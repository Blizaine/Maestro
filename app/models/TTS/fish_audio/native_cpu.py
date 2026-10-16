# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Optional x86 CPU kernels for Drama 3's routed expert banks.

The bundled library is a standalone C++ ABI copied from WanGP at the pinned
revision documented in ``native_cpu/THIRD_PARTY.md``. It does not depend on
PyTorch's C++ ABI. Import and load failures are intentionally non-fatal: the
caller can continue with the portable PyTorch implementation.
"""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

import torch


_FP8_BLOCK = 0
_INT4_GROUP = 1
_BLOCK = 128
_library = None
_load_attempted = False
_load_error: str | None = None


def _library_path() -> Path | None:
    machine = platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        return None
    if os.name == "nt":
        name = "cpu_experts_win_amd64.dll"
    elif os.name == "posix" and platform.system() == "Linux":
        name = "cpu_experts_linux_x86_64.so"
    else:
        return None
    return Path(__file__).with_name("native_cpu") / name


def _load_library():
    global _library, _load_attempted, _load_error
    if _load_attempted:
        return _library
    _load_attempted = True
    path = _library_path()
    if path is None or not path.is_file():
        _load_error = "No bundled native expert library for this platform"
        return None
    try:
        lib = ctypes.CDLL(str(path))
        lib.moe_supported.restype = ctypes.c_int
        if not lib.moe_supported():
            _load_error = "CPU does not support AVX2, FMA and F16C"
            return None
        lib.moe_forward.argtypes = [ctypes.c_int] * 2 + [ctypes.c_void_p] * 9 + [ctypes.c_int] * 7
        lib.moe_forward.restype = ctypes.c_int
        _library = lib
    except (OSError, AttributeError) as exc:
        # Missing optional VC++/OpenMP runtimes, or an unsupported binary, must
        # not make the portable Torch engine unavailable.
        _load_error = str(exc)
    return _library


def availability() -> tuple[bool, str | None]:
    """Return whether the optional native path can load on this CPU."""

    library = _load_library()
    return library is not None, _load_error


def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


@torch.inference_mode()
def forward(bank, x: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor, *, threads: int | None = None):
    """Compute a complete CPU expert call, or return ``None`` if unavailable.

    Expert buffers are read in place. Only activations, routing metadata and
    the float32 result are allocated for the current call.
    """

    library = _load_library()
    if library is None:
        return None
    if bank.format is None or bank.rotation < 0:
        raise RuntimeError("Fish Audio expert bank has invalid format metadata")
    if bank.format.name == "fp8_block":
        format_id = _FP8_BLOCK
    elif bank.format.name == "int4_group":
        format_id = _INT4_GROUP
    else:
        return None

    names = bank.expert_tensor_names
    if len(names) != 4:
        raise RuntimeError("Fish Audio native expert format must provide four checkpoint tensors")
    weights_13, weights_2, scales_13, scales_2 = (getattr(bank, name) for name in names)
    tensors = (weights_13, weights_2, scales_13, scales_2)
    if any(t.device.type != "cpu" or not t.is_contiguous() for t in tensors):
        raise RuntimeError("Fish Audio expert buffers must remain contiguous on CPU")

    if x.ndim < 1 or x.shape[-1] != bank.hidden:
        width = x.shape[-1] if x.ndim else 0
        raise ValueError(f"expert input width {width} does not match hidden size {bank.hidden}")
    x_cpu = x.reshape(-1, bank.hidden).detach().to(device="cpu", dtype=torch.float32).contiguous()
    ids_cpu = ids.reshape(x_cpu.shape[0], -1).to(device="cpu", dtype=torch.int32).contiguous()
    probabilities_cpu = probabilities.reshape(x_cpu.shape[0], -1).to(device="cpu", dtype=torch.float32).contiguous()
    if ids_cpu.shape != probabilities_cpu.shape:
        raise ValueError("topk_ids and topk_weights must have the same shape")
    if ids_cpu.numel() and (int(ids_cpu.min()) < 0 or int(ids_cpu.max()) >= bank.num_experts):
        raise ValueError("routed expert id is outside the loaded bank")
    output = torch.empty((x_cpu.shape[0], bank.hidden), dtype=torch.float32, device="cpu")
    thread_count = int(threads or min(os.cpu_count() or 1, 32))

    result = library.moe_forward(
        format_id,
        int(bank.rotation),
        _pointer(weights_13),
        _pointer(weights_2),
        _pointer(scales_13),
        _pointer(scales_2),
        _pointer(x_cpu),
        _pointer(ids_cpu),
        _pointer(probabilities_cpu),
        ctypes.c_void_p(),
        _pointer(output),
        x_cpu.shape[0],
        bank.hidden,
        bank.intermediate,
        bank.num_experts,
        ids_cpu.shape[1],
        _BLOCK,
        thread_count,
    )
    if result < 0:
        raise RuntimeError(f"Fish Audio native expert kernel failed ({result})")
    return output.to(device=x.device, dtype=x.dtype).reshape(x.shape)
