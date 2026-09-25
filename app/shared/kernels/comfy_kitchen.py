"""Narrow, optional Comfy Kitchen adapters for supported CUDA kernels.

Kernel imports are lazy because Comfy Kitchen may be installed without its
native CUDA extension. In particular, a callable exported Python wrapper does
not prove that a CUDA implementation is present or enabled.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch

from shared.kernels import kernel_policy


_SUPPORTED_SM = (12, 0)
_USED_KERNELS: set[str] = set()
_FAILURE_REASONS: set[str] = set()
_SCRATCH_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class KitchenCudaBackend:
    package: Any
    module: Any
    capabilities: frozenset[str]


def is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except (AttributeError, RuntimeError):
        return False


def _cuda_backend(
    required_capabilities: set[str],
    device: torch.device | None = None,
) -> KitchenCudaBackend | None:
    """Return only a registered, enabled CUDA backend with the requested ops."""

    if not torch.cuda.is_available() or torch.version.hip is not None:
        return None
    try:
        selected_device = torch.cuda.current_device() if device is None else device
        if torch.cuda.get_device_capability(selected_device) != _SUPPORTED_SM:
            return None
        package = importlib.import_module("comfy_kitchen")
        list_backends = getattr(package, "list_backends", None)
        if not callable(list_backends):
            return None
        backends = list_backends()
        status = backends.get("cuda", {}) if isinstance(backends, dict) else {}
        if not isinstance(status, dict) or not status.get("available") or status.get("disabled"):
            return None
        capabilities = frozenset(status.get("capabilities", ()))
        if not required_capabilities.issubset(capabilities):
            return None
        module = importlib.import_module("comfy_kitchen.backends.cuda")
        return KitchenCudaBackend(package, module, capabilities)
    except Exception:
        # Package import, backend registration, and capability enumeration are
        # all optional. A broken or CPU-only install simply keeps Torch active.
        return None


def kitchen_backend_for(device: torch.device, *capabilities: str) -> KitchenCudaBackend | None:
    if device.type != "cuda":
        return None
    return _cuda_backend(set(capabilities), device)


def report_kernel_used(name: str) -> None:
    if name not in _USED_KERNELS:
        _USED_KERNELS.add(name)
        print(f"[Comfy Kitchen] {name} kernel is active.")


def report_kernel_failure(name: str, error: Exception) -> None:
    if name not in _FAILURE_REASONS:
        _FAILURE_REASONS.add(name)
        detail = str(error).strip().splitlines()
        detail = detail[-1] if detail else type(error).__name__
        print(f"[Comfy Kitchen] {name} kernel unavailable at runtime; using PyTorch ({detail}).")


def _is_fake_tensor(value: torch.Tensor) -> bool:
    try:
        from torch._subclasses.fake_tensor import is_fake

        return bool(is_fake(value))
    except Exception:
        return False


def _has_unhandled_mm_lora_hook(module: torch.nn.Module, qlinear_type: type) -> bool:
    """Reject unknown MMGP wrappers while allowing Maestro's native ConvRot hook.

    Maestro's wrapper calls the original QLinear forward before applying LoRA
    deltas in source space, so the prequantized base path remains correct. Any
    other MMGP wrapper is left to its normal module call.
    """

    # Forward hooks run after the native qtype has filled the caller-owned
    # tile. Their ordering and semantics are unknown, so preserve their normal
    # module route rather than silently bypassing them.
    if getattr(module, "_forward_hooks", None) or getattr(module, "_forward_pre_hooks", None):
        return True

    forward = getattr(module, "forward", None)
    wrapper = getattr(forward, "func", None)
    local_native_wrapper = (
        getattr(wrapper, "__name__", None) == "_native_convrot_lora_forward"
        and callable(getattr(module, "_mm_convrot_native_forward", None))
    )
    if local_native_wrapper:
        # Even when adapters are currently deselected, retained MMGP tensors
        # and wrappers make the exact module-call route significant. Keep the
        # fast path only for the empty native hook state.
        return bool(getattr(module, "_mm_lora_data", None))

    mmgp_state = any(
        hasattr(module, name)
        for name in (
            "_mm_lora_old_forward",
            "_mm_lora_data",
            "_mm_lora_model",
            "_mm_convrot_native_forward",
        )
    )
    if mmgp_state:
        return True
    forward_func = getattr(forward, "__func__", forward)
    return forward_func is not qlinear_type.forward


def _supports_convrot_projection(module: torch.nn.Module, qlinear_type: type, x: torch.Tensor) -> bool:
    if not isinstance(module, qlinear_type) or _has_unhandled_mm_lora_hook(module, qlinear_type):
        return False
    weight = getattr(module, "weight", None)
    group_size = int(getattr(module, "_convrot_group_size", 0) or 0)
    if not torch.is_tensor(weight):
        return False
    qweight = getattr(module, "qweight", None)
    data = getattr(qweight, "_data", None)
    scale = getattr(qweight, "_scale", None)
    return bool(
        group_size == 256
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == x.dtype
        and weight.device == x.device
        and torch.is_tensor(data)
        and torch.is_tensor(scale)
        and data.device == x.device
        and data.dtype == torch.int8
        and data.is_contiguous()
        and scale.device == x.device
        and scale.numel() == module.out_features
        and module.in_features == x.shape[-1]
        and 256 <= module.in_features <= 16384
        and module.in_features % group_size == 0
        and module.out_features >= 8
        and module.out_features % 8 == 0
        and (module.bias is None or (module.bias.device == x.device and module.bias.dtype == x.dtype))
    )


def _prequantized_kitchen_linear(
    backend: Any,
    qweight: torch.Tensor,
    bias: torch.Tensor | None,
    quantized_input: torch.Tensor,
    input_scale: torch.Tensor,
    output: torch.Tensor,
    stream: int,
) -> torch.Tensor:
    """Write one ConvRot projection into a caller-owned output tile."""

    data = getattr(qweight, "_data", None)
    scale = getattr(qweight, "_scale", None)
    if not torch.is_tensor(data) or not torch.is_tensor(scale):
        raise TypeError("Kitchen ConvRot projection requires a Quanto INT8 weight")
    if output.ndim != 2 or output.shape != (quantized_input.shape[0], data.shape[0]):
        raise ValueError("Kitchen ConvRot output tile has an incompatible shape")
    if output.dtype != torch.bfloat16 or output.device != data.device:
        raise ValueError("Kitchen ConvRot output must be BF16 on the weight device")

    wrap = backend._wrap_for_dlpack
    weight_scale = scale.reshape(-1).to(device=output.device, dtype=torch.float32).contiguous()
    bias_arg = (
        backend._gemm_vector_arg(bias, output.device, output.dtype)
        if bias is not None
        else backend._empty_cuda_tensor(output.device, output.dtype)
    )
    used = backend._C.cutlass_int8_dequant(
        wrap(quantized_input),
        wrap(data),
        wrap(input_scale),
        wrap(weight_scale),
        wrap(bias_arg),
        wrap(output),
        backend.DTYPE_TO_CODE[output.dtype],
        stream,
    )
    if not used:
        raise RuntimeError("Comfy Kitchen rejected the shared-input ConvRot output tile")
    return output


def _has_native_int8_projection_symbols(module: Any) -> bool:
    native = getattr(module, "_C", None)
    return bool(
        callable(getattr(module, "quantize_int8_rowwise_convrot64", None))
        and callable(getattr(native, "cutlass_int8_dequant", None))
    )


def _int8_projection_backend(device: torch.device) -> KitchenCudaBackend | None:
    backend = kitchen_backend_for(device, "int8_linear")
    if (
        backend is None
        or getattr(backend.module, "_DISABLE_CUTLASS_INT8", False)
        or not _has_native_int8_projection_symbols(backend.module)
        or not callable(getattr(backend.module, "_convrot_fused_shared_memory_fits", None))
    ):
        return None
    return backend


def _validate_project_many(modules: tuple[torch.nn.Module, ...], x: torch.Tensor) -> tuple[type, Any] | None:
    if (
        not modules
        or not kernel_policy.allow_approximate()
        or not torch.is_inference_mode_enabled()
        or torch.is_grad_enabled()
        or is_compiling()
        or _is_fake_tensor(x)
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        return None
    try:
        from shared.qtypes.int8_convrot import QLinearInt8ConvRot
    except Exception:
        return None
    if not all(_supports_convrot_projection(module, QLinearInt8ConvRot, x) for module in modules):
        return None
    # The registry documents the INT8 linear operation, but the shared-input
    # quantizer is a low-level helper and is intentionally absent from the
    # public capability list. Probe that implementation separately.
    backend = _int8_projection_backend(x.device)
    if backend is None:
        return None
    try:
        if not all(
            backend.module._convrot_fused_shared_memory_fits(x, module.in_features, 256)
            for module in modules
        ):
            return None
    except Exception:
        return None
    return QLinearInt8ConvRot, backend.module


def project_int8_convrot_many(
    modules: tuple[torch.nn.Module, ...],
    x: torch.Tensor,
) -> tuple[torch.Tensor, ...] | None:
    """Share rowwise ConvRot quantization across compatible linear layers.

    Returns ``None`` if the backend, model layers, or current execution mode
    cannot use the verified SM120 BF16 path. We keep the full-precision input
    and invoke each original module under its existing hooks so the caller can
    use its normal fallback and any known native LoRA wrapper remains active.
    """

    validated = _validate_project_many(modules, x)
    if validated is None:
        return None
    qlinear_type, backend = validated
    flat = x.reshape(-1, x.shape[-1])
    outputs = [
        torch.empty((flat.shape[0], module.out_features), device=x.device, dtype=torch.bfloat16)
        for module in modules
    ]
    # Quantized input, its row scales, one weight's output tile, and an
    # optional contiguous source tile stay within this small scratch target.
    per_row_bytes = flat.shape[-1] + 4 + (flat.shape[-1] * x.element_size() if not flat.is_contiguous() else 0)
    tile_rows = max(32, (_SCRATCH_BYTES // max(1, per_row_bytes)) // 32 * 32)
    try:
        stream = torch.cuda.current_stream(x.device).cuda_stream
        for start in range(0, flat.shape[0], tile_rows):
            stop = min(start + tile_rows, flat.shape[0])
            tile = flat[start:stop]
            quantized_input, input_scale = backend.quantize_int8_rowwise_convrot64(tile, 256)
            for module, output in zip(modules, outputs, strict=True):
                output_tile = output[start:stop]
                previous = getattr(module, "_maestro_kitchen_prequantized_input", None)
                module._maestro_kitchen_prequantized_input = (
                    quantized_input,
                    input_scale,
                    output_tile,
                    backend,
                    stream,
                )
                try:
                    result = module(tile)
                    if result.shape != output_tile.shape or result.dtype != output_tile.dtype:
                        raise RuntimeError("ConvRot module returned an incompatible Kitchen output")
                    if result.data_ptr() != output_tile.data_ptr():
                        output_tile.copy_(result)
                finally:
                    if previous is None:
                        del module._maestro_kitchen_prequantized_input
                    else:
                        module._maestro_kitchen_prequantized_input = previous
            del quantized_input, input_scale, tile
    except Exception as error:
        # A failed optional tile never mutates the model input. Discard all
        # partial outputs and let the caller replay the original module math.
        report_kernel_failure("shared-input INT8 ConvRot projection", error)
        return None

    report_kernel_used("shared-input INT8 ConvRot projection (SM120 BF16)")
    return tuple(output.reshape(*x.shape[:-1], output.shape[-1]) for output in outputs)


def use_prequantized_convrot_input(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """Consume the scoped LTX tile marker from the native QLinear forward."""

    prepared = getattr(module, "_maestro_kitchen_prequantized_input", None)
    if prepared is None:
        return None
    if len(prepared) != 5:
        raise ValueError("Malformed temporary Comfy Kitchen ConvRot input")
    quantized_input, input_scale, output, backend, stream = prepared
    expected = (x.reshape(-1, x.shape[-1]).shape[0], module.out_features)
    if output.shape != expected or output.dtype != x.dtype or output.device != x.device:
        raise ValueError("Temporary Comfy Kitchen ConvRot output does not match its module input")
    return _prequantized_kitchen_linear(
        backend,
        module.qweight,
        module.bias,
        quantized_input,
        input_scale,
        output,
        stream,
    )


__all__ = [
    "KitchenCudaBackend",
    "is_compiling",
    "kitchen_backend_for",
    "project_int8_convrot_many",
    "report_kernel_failure",
    "report_kernel_used",
    "use_prequantized_convrot_input",
]
