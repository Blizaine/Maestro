"""Print a concise startup audit of the active acceleration environment.

This intentionally exits successfully even when an optional kernel is broken;
the message tells Pinokio users to run Update while preserving Maestro's
existing fallback behavior for manually managed environments.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys

from services.runtime_compat import evaluate_runtime, format_runtime_warnings


def _import_ok(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _version_at_least(value: str | None, minimum: tuple[int, int]) -> bool:
    numbers = [int(token) for token in re.findall(r"\d+", value or "")[:2]]
    while len(numbers) < 2:
        numbers.append(0)
    return tuple(numbers[:2]) >= minimum


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"[Runtime][ACTION REQUIRED] PyTorch could not load: {exc}")
        print("[Runtime][ACTION REQUIRED] Run Update in Pinokio to repair the Maestro runtime.")
        return 0

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = "CUDA unavailable"
    capability: tuple[int, int] | None = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            capability = tuple(torch.cuda.get_device_capability(0))
        except Exception:
            pass

    # Importing lightx2v_kernel registers its torch.ops namespace. A package
    # can exist but still fail here when it was built for the wrong Torch/CUDA
    # ABI, which is exactly the public-v1.6.5 RTX 5090 failure mode.
    triton_ok = _import_ok("triton")
    triton_version = _distribution_version("triton-windows", "triton")
    sage_ok = _import_ok("sageattention")
    flash_ok = _import_ok("flash_attn")
    lightx2v_import_ok = _import_ok("lightx2v_kernel")
    lightx2v_ops = bool(
        hasattr(torch.ops, "lightx2v_kernel")
        and hasattr(torch.ops.lightx2v_kernel, "cutlass_scaled_nvfp4_mm_sm120")
    )
    lightx2v_ok = lightx2v_import_ok and lightx2v_ops
    sol_capability = capability in {(8, 9), (9, 0), (10, 0), (12, 0)}
    sol_ready = bool(
        cuda_available
        and sol_capability
        and triton_ok
        and _version_at_least(triton_version, (3, 6))
    )

    torch_version = str(getattr(torch, "__version__", "unknown"))
    cuda_version = str(getattr(torch.version, "cuda", None) or "none")
    cap_label = "unknown" if capability is None else f"sm_{capability[0]}{capability[1]}"
    print(
        f"[Runtime] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} | "
        f"PyTorch {torch_version} | CUDA {cuda_version} | {gpu_name} ({cap_label})"
    )
    print(
        "[Runtime] Kernels: "
        f"Triton={'ready' if triton_ok else 'missing'}"
        f"{f' ({triton_version})' if triton_version else ''}, "
        f"SageAttention={'ready' if sage_ok else 'missing'}, "
        f"Lightx2v NVFP4={'ready' if lightx2v_ok else 'missing'}, "
        f"FlashAttention={'ready' if flash_ok else 'missing'}, "
        f"H3 Sol Engine={'ready' if sol_ready else 'unavailable in this runtime'}"
    )

    warnings = evaluate_runtime(
        python_version=sys.version.split()[0],
        torch_version=torch_version,
        cuda_version=cuda_version,
        capability=capability,
        triton_available=triton_ok,
        sage_available=sage_ok,
        lightx2v_available=lightx2v_ok,
    )
    if warnings:
        print(f"[Runtime][ACTION REQUIRED] {format_runtime_warnings(warnings)}.")
        print(
            "[Runtime][ACTION REQUIRED] Run Update in Pinokio. Maestro will create "
            "the dedicated RTX 50 Python 3.11 / CUDA 13 environment automatically."
        )
        print(
            "[Runtime][ACTION REQUIRED] If Update already completed, use "
            "Advanced > Repair RTX 50 Runtime in Maestro's Pinokio menu."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
