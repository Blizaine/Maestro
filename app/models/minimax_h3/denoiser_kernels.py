"""Optional H3 denoiser kernels with an exact PyTorch fallback.

The narrow Comfy Kitchen device and capability gate follows Wan2GP at
2345ae148f82740f66e82c41292dbbdd592e713d. The H3 RMSNorm+RoPE operation uses
the out-of-place API so a failed kernel call leaves raw Q/K available for the
existing PyTorch implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from shared.kernels import kernel_policy
from shared.kernels.comfy_kitchen import (
    KitchenCudaBackend,
    is_compiling,
    kitchen_backend_for,
    report_kernel_failure,
    report_kernel_used,
)


@dataclass(frozen=True)
class PreparedRotary:
    cosine: torch.Tensor
    sine: torch.Tensor
    backend: KitchenCudaBackend

    def __iter__(self):
        yield self.cosine
        yield self.sine

    def __getitem__(self, index: int) -> torch.Tensor:
        return (self.cosine, self.sine)[index]


def prepare_rope(rotary):
    """Attach an available CUDA backend once, without materializing RoPE tables."""

    if rotary is None or not kernel_policy.allow_approximate():
        return rotary
    if torch.is_grad_enabled() or not torch.is_inference_mode_enabled() or is_compiling():
        return rotary
    try:
        cosine, sine = rotary
    except (TypeError, ValueError):
        return rotary
    if (
        not torch.is_tensor(cosine)
        or not torch.is_tensor(sine)
        or cosine.ndim != 2
        or sine.shape != cosine.shape
        or cosine.device.type != "cuda"
        or sine.device != cosine.device
        or cosine.shape[-1] < 2
        or cosine.shape[-1] % 2
    ):
        return rotary
    backend = kitchen_backend_for(cosine.device, "rms_rope_split_half")
    native = None if backend is None else getattr(backend.module, "_C", None)
    if (
        backend is None
        or not callable(getattr(backend.module, "rms_rope_split_half", None))
        or not callable(getattr(native, "rms_rope", None))
    ):
        return rotary
    return PreparedRotary(cosine, sine, backend)


def can_rms_rope(
    hidden_states: torch.Tensor,
    rotary,
    q_norm: nn.Module,
    k_norm: nn.Module,
) -> bool:
    if not isinstance(rotary, PreparedRotary):
        return False
    if (
        torch.is_grad_enabled()
        or not torch.is_inference_mode_enabled()
        or is_compiling()
        or not kernel_policy.allow_approximate()
        or hidden_states.device != rotary.cosine.device
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_cuda
    ):
        return False
    for norm in (q_norm, k_norm):
        if getattr(norm, "_forward_hooks", None) or getattr(norm, "_forward_pre_hooks", None):
            return False
        weight = getattr(norm, "weight", None)
        if (
            not torch.is_tensor(weight)
            or weight.device != hidden_states.device
            or weight.dtype != hidden_states.dtype
            or weight.ndim != 1
        ):
            return False
    return True


def _pack_split_half_rope(cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    """Convert H3's repeated split-half vectors to Kitchen's matrix layout."""

    pairs = cosine.shape[-1] // 2
    return torch.stack(
        (
            cosine[..., :pairs],
            -sine[..., :pairs],
            sine[..., :pairs],
            cosine[..., :pairs],
        ),
        dim=-1,
    ).reshape(cosine.shape[0], pairs, 2, 2)[None, :, None]


def rms_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary,
    start: int,
    stop: int,
    q_norm: nn.Module,
    k_norm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run one bounded out-of-place H3 Q/K RMSNorm+RoPE tile, if supported."""

    if not isinstance(rotary, PreparedRotary) or not kernel_policy.allow_approximate():
        return None
    if (
        torch.is_grad_enabled()
        or not torch.is_inference_mode_enabled()
        or is_compiling()
        or query.device.type != "cuda"
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or key.device != query.device
        or query.ndim != 4
        or key.shape != query.shape
        or query.shape[-1] < 32
        or query.shape[-1] % 32
        or getattr(q_norm, "_forward_hooks", None)
        or getattr(q_norm, "_forward_pre_hooks", None)
        or getattr(k_norm, "_forward_hooks", None)
        or getattr(k_norm, "_forward_pre_hooks", None)
    ):
        return None
    cosine = rotary.cosine[start:stop]
    sine = rotary.sine[start:stop]
    pairs = cosine.shape[-1] // 2
    if (
        cosine.shape != sine.shape
        or cosine.ndim != 2
        or cosine.shape[0] != query.shape[1]
        or cosine.shape[-1] % 2
        or not 0 < pairs * 2 <= query.shape[-1]
        or cosine.device != query.device
        or sine.device != query.device
        or cosine.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or sine.dtype != cosine.dtype
    ):
        return None
    q_scale = getattr(q_norm, "weight", None)
    k_scale = getattr(k_norm, "weight", None)
    if (
        not torch.is_tensor(q_scale)
        or not torch.is_tensor(k_scale)
        or q_scale.shape != (query.shape[-1],)
        or k_scale.shape != (key.shape[-1],)
        or q_scale.device != query.device
        or k_scale.device != key.device
        or q_scale.dtype != query.dtype
        or k_scale.dtype != key.dtype
    ):
        return None

    # Comfy Kitchen stores a complex RoPE matrix per pair; the local H3
    # embedding stores repeated split-half cos/sin vectors. Pack only this
    # bounded tile so long packed sequences do not gain a second full RoPE map.
    try:
        freqs = _pack_split_half_rope(cosine, sine)
        result = rotary.backend.module.rms_rope_split_half(
            query,
            key,
            freqs,
            q_scale,
            k_scale,
            epsilon=float(getattr(q_norm, "eps", 1e-6)),
            rot_dim=pairs * 2,
        )
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or any(
                not torch.is_tensor(value)
                or value.shape != query.shape
                or value.device != query.device
                or value.dtype != query.dtype
                for value in result
            )
        ):
            raise RuntimeError("Comfy Kitchen returned incompatible H3 Q/K tensors")
    except Exception as error:
        report_kernel_failure("H3 RMSNorm + split-half RoPE", error)
        return None
    report_kernel_used("H3 RMSNorm + split-half RoPE (SM120 BF16)")
    return result


__all__ = ["PreparedRotary", "can_rms_rope", "prepare_rope", "rms_rope"]
