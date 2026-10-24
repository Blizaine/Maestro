# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Portable dequantizers for the released Drama 3 expert checkpoints.

The layouts mirror Fish Audio's block-FP8 release and its INT4 group-scale
release.  These are intentionally plain PyTorch implementations so Windows
Python/Torch installations do not need a compiler or shipped native library.
"""

from __future__ import annotations

import torch
from mmgp.expert_compute import ExpertFormat, SwiGLUExperts

_REPORTED_BACKENDS = set()


class Fp8BlockExperts(ExpertFormat):
    name = "fp8_block"
    block = 128

    def tensors(self, experts: int, hidden: int, intermediate: int):
        block = self.block
        return {
            "w13_weight": ((experts, 2 * intermediate, hidden), torch.float8_e4m3fn),
            "w2_weight": ((experts, hidden, intermediate), torch.float8_e4m3fn),
            "w13_scale_inv": ((experts, 2 * intermediate // block, hidden // block), torch.float32),
            "w2_scale_inv": ((experts, hidden // block, intermediate // block), torch.float32),
        }

    def matches(self, tensors):
        return (
            tensors["w13_weight"].dtype == torch.float8_e4m3fn
            and tensors["w2_weight"].dtype == torch.float8_e4m3fn
            and tensors["w13_scale_inv"].dtype == torch.float32
            and tensors["w2_scale_inv"].dtype == torch.float32
        )

    def dequantize(self, tensors):
        def matrix(weight, scale):
            rows, cols = weight.shape
            block = self.block
            return (
                weight.float()
                .reshape(rows // block, block, cols // block, block)
                .mul(scale.float()[:, None, :, None])
                .reshape(rows, cols)
            )

        return (
            matrix(tensors["w13_weight"], tensors["w13_scale_inv"]),
            matrix(tensors["w2_weight"], tensors["w2_scale_inv"]),
        )


class Int4GroupExperts(ExpertFormat):
    """Symmetric INT4, zero point 8, with FP16 scales per 128 input columns."""

    name = "int4_group"
    block = 128

    def tensors(self, experts: int, hidden: int, intermediate: int):
        block = self.block
        return {
            "w13_weight": ((experts, 2 * intermediate, hidden // 2), torch.uint8),
            "w2_weight": ((experts, hidden, intermediate // 2), torch.uint8),
            "w13_scales": ((experts, 2 * intermediate, hidden // block), torch.float16),
            "w2_scales": ((experts, hidden, intermediate // block), torch.float16),
        }

    def matches(self, tensors):
        return (
            tensors["w13_weight"].dtype == torch.uint8
            and tensors["w2_weight"].dtype == torch.uint8
            and tensors["w13_scales"].dtype == torch.float16
            and tensors["w2_scales"].dtype == torch.float16
        )

    def dequantize(self, tensors):
        def matrix(weight, scale):
            rows, packed_cols = weight.shape
            values = torch.stack((weight & 15, weight >> 4), dim=-1)
            values = values.reshape(rows, packed_cols * 2).float().sub_(8.0)
            values = values.reshape(rows, -1, self.block)
            return (values * scale.float().unsqueeze(-1)).reshape(rows, packed_cols * 2)

        return (
            matrix(tensors["w13_weight"], tensors["w13_scales"]),
            matrix(tensors["w2_weight"], tensors["w2_scales"]),
        )


EXPERT_FORMATS = (Fp8BlockExperts(), Int4GroupExperts())


class FishSwiGLUExperts(SwiGLUExperts):
    """Drama 3 bank with an optional zero-copy native CPU execution path."""

    def __init__(self, *args, use_triton=False, **kwargs):
        super().__init__(*args, use_triton=use_triton, **kwargs)
        self._fish_execution_backend = "not_run"
        self._fish_gpu_tensors = None
        self._fish_gpu_pointers = None
        self._fish_gpu_device = None
        self._fish_gpu_cache = None
        self._fish_decode_workspace = None

    @property
    def execution_backend(self) -> str:
        return self._fish_execution_backend

    def release_gpu_experts(self) -> None:
        """Release optional full-bank GPU copies while retaining CPU checkpoint storage."""

        self._fish_gpu_tensors = None
        self._fish_gpu_pointers = None
        self._fish_gpu_device = None
        self._fish_gpu_cache = None
        self._fish_decode_workspace = None

    def release_decode_cache(self):
        self._fish_gpu_cache = None
        self._fish_decode_workspace = None

    def supports_gpu_device(self, device: torch.device) -> bool:
        """Return whether this packed expert format is supported by Triton on a device."""

        device = torch.device(device)
        if device.type != "cuda" or self.format is None:
            return False
        if self.format.name == "fp8_block":
            return torch.cuda.get_device_capability(device) >= (8, 9)
        return self.format.name == "int4_group"

    def _require_gpu_format_support(self, device: torch.device) -> None:
        if self.supports_gpu_device(device):
            return
        if self.format is not None and self.format.name == "fp8_block":
            raise ValueError(
                "FP8 Triton expert kernels require CUDA compute capability 8.9 or newer; "
                "use the CPU fallback on this device"
            )
        kind = "unloaded" if self.format is None else self.format.name
        raise ValueError(
            f"Fish {kind} Triton expert placement is unsupported on {device}; "
            "use the CPU fallback"
        )

    def place_cached_experts(self, capacity, topk, device):
        device = self._canonical_cuda_device(device)
        if device.type != "cuda":
            raise ValueError("Fish GPU expert cache placement requires a CUDA device")
        if not self.use_triton:
            raise RuntimeError("Fish GPU experts are available only for the vLLM engine")
        self._require_gpu_format_support(device)
        from .expert_cache import PackedExpertCache
        self._fish_gpu_cache = PackedExpertCache(self, capacity, topk, device)

    @staticmethod
    def _canonical_cuda_device(device: torch.device) -> torch.device:
        """Resolve an unindexed CUDA alias before comparing tensor devices."""

        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def place_gpu_experts(self, device: torch.device) -> int:
        """Copy all routed tensors to a device and build the Triton pointer table."""

        device = self._canonical_cuda_device(device)
        if device.type != "cuda":
            raise ValueError("Fish GPU expert placement requires a CUDA device")
        if not self.use_triton:
            raise RuntimeError("Fish GPU experts are available only for the vLLM engine")
        self._require_gpu_format_support(device)
        if self._fish_gpu_tensors is not None:
            if self._fish_gpu_device != device:
                raise RuntimeError("Fish expert bank is already placed on another CUDA device")
            return self.expert_nbytes
        tensors = tuple(getattr(self, name).to(device=device) for name in self.expert_tensor_names)
        count = self.num_experts
        pointers = torch.empty((count, len(tensors)), dtype=torch.int64, device=device)
        for index, tensor in enumerate(tensors):
            stride = tensor[0].numel() * tensor.element_size()
            pointers[:, index] = torch.arange(count, dtype=torch.int64, device=device) * stride + tensor.data_ptr()
        self._fish_gpu_tensors = tensors
        self._fish_gpu_pointers = pointers
        self._fish_gpu_device = tensors[0].device
        return self.expert_nbytes

    def _gpu_forward(self, x, topk_ids, topk_weights, *, check_route_bounds=True,
                     gpu_pointers=None, gpu_device=None):
        activation_device = self._canonical_cuda_device(x.device)
        expert_device = self._canonical_cuda_device(gpu_device if gpu_device is not None else self._fish_gpu_device)
        if not x.is_cuda or activation_device != expert_device:
            raise RuntimeError(
                "Fish GPU expert weights and activations must be on the same CUDA device "
                f"(activation={x.device}, experts={self._fish_gpu_device})"
            )
        from .expert_triton import forward as triton_forward

        shape = x.shape
        flat = x.reshape(-1, self.hidden)
        ids = topk_ids.reshape(flat.shape[0], -1).contiguous()
        probabilities = topk_weights.reshape(flat.shape[0], -1).float().contiguous()
        if ids.shape != probabilities.shape:
            raise ValueError("topk_ids and topk_weights must have the same shape")
        if (
            check_route_bounds
            and ids.numel()
            and (int(ids.min()) < 0 or int(ids.max()) >= self.num_experts)
        ):
            raise ValueError("routed expert id is outside the loaded bank")
        workspace = {}
        if flat.shape[0] == 1:
            key = (flat.device, flat.dtype, ids.shape[-1])
            if self._fish_decode_workspace is None or self._fish_decode_workspace[0] != key:
                self._fish_decode_workspace = (key, {
                    "mid": torch.empty((ids.shape[-1], self.intermediate), dtype=torch.float32, device=flat.device),
                    "out": torch.empty_like(flat),
                    "down_scratch": torch.empty((ids.shape[-1], self.hidden), dtype=torch.float32, device=flat.device),
                })
            workspace = self._fish_decode_workspace[1]
        result = triton_forward(
            self.format.name,
            flat.contiguous(),
            ids,
            probabilities,
            self._fish_gpu_pointers if gpu_pointers is None else gpu_pointers,
            self.intermediate,
            128,
            rotation=self.rotation,
            single_token_rows=16,
            parallel_single_token_down=flat.shape[0] == 1,
            **workspace,
        )
        return result.to(dtype=x.dtype).reshape(shape)

    @torch.inference_mode()
    def forward(self, x, topk_ids, topk_weights, packed=None, *, check_route_bounds=True):
        if self._fish_gpu_tensors is not None:
            self._fish_execution_backend = "gpu_triton"
            if self._fish_execution_backend not in _REPORTED_BACKENDS:
                print("[Fish Audio] Routed expert backend: gpu_triton (full bank resident)")
                _REPORTED_BACKENDS.add(self._fish_execution_backend)
            return self._gpu_forward(
                x,
                topk_ids,
                topk_weights,
                check_route_bounds=check_route_bounds,
            )
        if self._fish_gpu_cache is not None and x.is_cuda and x.numel() == self.hidden:
            self._fish_execution_backend = "gpu_cached_triton"
            pointers = self._fish_gpu_cache.prepare(topk_ids.reshape(1, -1))
            return self._gpu_forward(x, topk_ids, topk_weights, check_route_bounds=False,
                                     gpu_pointers=pointers, gpu_device=self._fish_gpu_cache.device)
        from .native_cpu import forward as native_forward

        result = native_forward(self, x, topk_ids, topk_weights)
        if result is not None:
            self._fish_execution_backend = "native_cpu"
            if self._fish_execution_backend not in _REPORTED_BACKENDS:
                print("[Fish Audio] Routed expert backend: native_cpu")
                _REPORTED_BACKENDS.add(self._fish_execution_backend)
            return result
        self._fish_execution_backend = "portable_torch"
        if self._fish_execution_backend not in _REPORTED_BACKENDS:
            print("[Fish Audio] Routed expert backend: portable_torch")
            _REPORTED_BACKENDS.add(self._fish_execution_backend)
        return super().forward(x, topk_ids, topk_weights, packed)
