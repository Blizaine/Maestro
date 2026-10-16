# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Portable PyTorch execution for Fish Audio's routed SwiGLU experts.

This is the compiler-free CPU baseline. Fish's local subclass can dispatch to
the optional pinned native CPU library or to its optional no-graph Triton GPU
kernel when those paths are available; the quantized checkpoint tensors stay
in the separately managed expert bank.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .experts import ExpertBank


_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _regular_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build Fish ConvRot's normalized regular Hadamard matrix (H4 Kronecker powers)."""

    size = int(size)
    device = torch.device(device)
    if size < 4 or size & (size - 1) or size & 0x55555555 == 0:
        raise ValueError(f"ConvRot group size must be a power of four, got {size}")
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    h4 = torch.tensor(
        ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1)),
        device=device,
        dtype=dtype,
    )
    hadamard = h4
    while hadamard.shape[0] < size:
        hadamard = torch.kron(hadamard, h4)
    hadamard = hadamard * (size ** -0.5)
    _HADAMARD_CACHE[key] = hadamard
    return hadamard


def _rotate_activation(input_tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply the checkpoint's row-vector ConvRot transform to each feature group."""

    group_size = int(group_size or 0)
    if group_size <= 0:
        return input_tensor
    features = int(input_tensor.shape[-1])
    if features % group_size:
        raise ValueError(
            f"ConvRot input features {features} are not divisible by group size {group_size}"
        )
    original_shape = input_tensor.shape
    matrix = _regular_hadamard(group_size, input_tensor.device, input_tensor.dtype)
    return torch.matmul(
        input_tensor.reshape(-1, features // group_size, group_size), matrix
    ).reshape(original_shape)


class ExpertFormat:
    """Checkpoint layout and CPU dequantizer for one routed expert format."""

    name = ""

    def tensors(self, experts: int, hidden: int, intermediate: int):
        raise NotImplementedError

    def matches(self, tensors: Mapping[str, torch.Tensor]) -> bool:
        raise NotImplementedError

    def dequantize(self, tensors: Mapping[str, torch.Tensor]):
        """Return float32 ``(gate+up, down)`` matrices for one expert."""
        raise NotImplementedError


class SwiGLUExperts(ExpertBank):
    """CPU-executed SwiGLU expert bank with FP8-block and INT4-group weights."""

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        intermediate: int,
        *,
        formats: Sequence[ExpertFormat],
        device: str | torch.device = "meta",
        use_triton: bool = False,
    ) -> None:
        super().__init__(num_experts)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.formats = tuple(formats)
        self.format: ExpertFormat | None = None
        self.rotation = 0
        self.use_triton = bool(use_triton)
        self._expert_format_error = None

    @staticmethod
    def pack(x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        """Keep the upstream call contract; portable execution reads tensors directly."""

        return torch.cat(
            (
                x.reshape(-1).float().contiguous().view(torch.int32),
                topk_weights.reshape(-1).float().contiguous().view(torch.int32),
                topk_ids.reshape(-1).to(torch.int32),
            )
        )

    def _set_format(self, expert_format: ExpertFormat) -> None:
        for name in self.expert_tensor_names:
            self._buffers.pop(name, None)
        declarations = expert_format.tensors(self.num_experts, self.hidden, self.intermediate)
        for name, (shape, dtype) in declarations.items():
            self.register_buffer(name, torch.empty(shape, dtype=dtype, device="meta"))
        self.expert_tensor_names = tuple(declarations)
        self.format = expert_format

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        rotation = state_dict.pop(prefix + "convrot_group_size", None)
        if rotation is not None:
            self.rotation = int(rotation.item())

        for candidate in self.formats:
            declarations = candidate.tensors(self.num_experts, self.hidden, self.intermediate)
            names = tuple(declarations)
            if not all(prefix + name in state_dict for name in names):
                continue
            tensors = {name: state_dict[prefix + name] for name in names}
            if candidate.matches(tensors):
                self._set_format(candidate)
                break
        else:
            expected = ", ".join(candidate.name for candidate in self.formats) or "no registered formats"
            error_msgs.append(f"No supported Fish Audio expert format found for {prefix[:-1]} ({expected})")
            return

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        packed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute selected experts on CPU, then return the original shape/device."""

        if self.format is None:
            raise RuntimeError("Fish Audio expert bank was used before its checkpoint format was loaded")
        if x.shape[-1] != self.hidden:
            raise ValueError(f"expert input width {x.shape[-1]} does not match hidden size {self.hidden}")
        flat = x.reshape(-1, self.hidden)
        ids = topk_ids.reshape(flat.shape[0], -1).to(device="cpu", dtype=torch.long)
        weights = topk_weights.reshape(flat.shape[0], -1).to(device="cpu", dtype=torch.float32)
        if ids.shape != weights.shape:
            raise ValueError("topk_ids and topk_weights must have the same shape")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.num_experts):
            raise ValueError("routed expert id is outside the loaded bank")

        x_cpu = flat.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if self.rotation:
            x_cpu = _rotate_activation(x_cpu, self.rotation)
        output = torch.zeros((flat.shape[0], self.hidden), dtype=torch.float32, device="cpu")
        for expert_id in torch.unique(ids, sorted=True).tolist():
            rows, slots = torch.where(ids == expert_id)
            selected = x_cpu.index_select(0, rows)
            tensors = {name: getattr(self, name)[expert_id] for name in self.expert_tensor_names}
            w13, w2 = self.format.dequantize(tensors)
            gate, up = F.linear(selected, w13).chunk(2, dim=-1)
            intermediate = F.silu(gate) * up
            if self.rotation:
                intermediate = _rotate_activation(intermediate, self.rotation)
            values = F.linear(intermediate, w2)
            values.mul_(weights[rows, slots, None])
            output.index_add_(0, rows, values)

        return output.to(device=x.device, dtype=x.dtype).reshape(x.shape)

    def queue_forward(self, *args, **kwargs):
        raise RuntimeError(
            "Fish Audio's portable CPU experts cannot run inside CUDA graphs; select the legacy LM engine"
        )

    def raise_job_errors(self) -> None:
        return None
