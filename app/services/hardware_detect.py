"""
Hardware detection for the Performance Auto-Tune feature.

Single entry point: `detect_hardware()` returns a dict describing the
user's GPU + RAM + which acceleration kernels are actually installed.
The recommendation engine in `perf_recommend.py` consumes this and
returns the settings to apply.

Design notes:
- All heavy imports (torch, pynvml) are wrapped in try/except so this
  module loads cleanly on systems without CUDA — the AMD/CPU fallback
  path returns a minimal dict with `cuda_available=False`.
- Detection is deliberately cheap: no model loads, no kernel init.
  Should run in <100ms on any system. Safe to call from a sync HTTP
  endpoint without spawning a thread.
- Capability detection (sm89, sm120, sage, etc.) is conservative —
  we check both GPU capability AND that the relevant kernel module
  imports cleanly. A user with an RTX 4090 but no FP8 kernels
  installed will report `supports_fp8=False`, because we can't
  actually use FP8 in that state.
"""
from __future__ import annotations

import platform as host_platform
import sys
from pathlib import Path
from typing import Optional

# RAM detection — psutil is a hard dependency of the app, no fallback needed.
import psutil

_APP_ROOT = Path(__file__).resolve().parents[1]

# RAM is reported in binary GiB, while tier names describe the nominal
# 32/64-GB machine classes. A small allowance covers firmware-reserved RAM
# and reporting differences without admitting a materially smaller tier.
RAM_TIER_TOLERANCE_GB = 0.5


def ram_tier_for_gb(ram_gb: float) -> str:
    """Return the nominal RAM tier with a 0.5-GiB boundary tolerance.

    Hardware capacities are converted from bytes to GiB and rounded to one
    decimal place, matching the VRAM probe. The tolerance applies only to the
    32/64-GiB RAM tier boundaries; values below 31.5 or 63.5 GiB respectively
    remain in the smaller tier.
    """
    if ram_gb >= 64.0 - RAM_TIER_TOLERANCE_GB:
        return "high"
    if ram_gb >= 32.0 - RAM_TIER_TOLERANCE_GB:
        return "low"
    return "very_low"


def _detect_driver_version() -> str:
    """Read the NVIDIA driver version without creating a CUDA context."""
    try:
        import pynvml
        # Keep NVML initialized for the app's telemetry readers, which share
        # this process-wide client and also leave it initialized at startup.
        pynvml.nvmlInit()
        value = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return str(value) if value else "unknown"
    except Exception:
        return "unknown"


def _supports_mmgp_allocator(
    *,
    cuda_available: bool,
    nvidia_cuda: bool,
    system: str,
    machine: str,
    app_root: Optional[Path] = None,
) -> bool:
    """Check for a bundled allocator binary on a supported NVIDIA platform.

    This is a filesystem capability check only. It does not import MMGP's
    allocator, create a CUDA context, allocate memory, or execute a kernel.
    Runtime startup remains responsible for loading it and falling back.
    """
    if not cuda_available or not nvidia_cuda:
        return False

    machine = str(machine or "").lower()
    if system == "win32" and machine in ("amd64", "x86_64"):
        filename = "vmm_alloc_win_amd64.dll"
    elif system == "linux" and machine in ("x86_64", "amd64"):
        filename = "vmm_alloc_linux_x86_64.so"
    else:
        return False

    root = app_root if app_root is not None else _APP_ROOT
    return (Path(root) / "mmgp" / "allocator" / filename).is_file()


def _detect_gpu() -> dict:
    """Detect GPU specs. Returns CUDA-only fields; safe to call without CUDA.

    Returns dict with keys:
      cuda_available: bool
      gpu_name: str (empty if no CUDA)
      gpu_vram_gb: float (0 if no CUDA)
      gpu_capability: str (e.g. "sm89", "sm120", or "" if no CUDA)
      gpu_capability_tuple: tuple[int, int] | None
    """
    out = {
        "cuda_available": False,
        "gpu_name": "",
        "gpu_vram_gb": 0.0,
        "gpu_capability": "",
        "gpu_capability_tuple": None,
        "nvidia_cuda": False,
        "torch_version": "unknown",
        "runtime_version": "unknown",
    }
    try:
        import torch
        torch_version = getattr(torch, "__version__", None)
        version = getattr(torch, "version", None)
        cuda_runtime = getattr(version, "cuda", None)
        hip_runtime = getattr(version, "hip", None)
        out["torch_version"] = str(torch_version) if torch_version else "unknown"
        runtime_version = cuda_runtime or hip_runtime
        out["runtime_version"] = str(runtime_version) if runtime_version else "unknown"
        out["nvidia_cuda"] = bool(cuda_runtime and not hip_runtime)
        if not torch.cuda.is_available():
            return out
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        out["cuda_available"] = True
        out["gpu_name"] = torch.cuda.get_device_name(0)
        # total_memory is bytes; convert to GB rounded to 1 decimal
        out["gpu_vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
        out["gpu_capability"] = f"sm{major}{minor}"
        out["gpu_capability_tuple"] = (major, minor)
    except Exception:
        # Any failure (driver issue, AMD without ROCm, etc.) → return
        # the no-CUDA defaults. Recommendation engine will pick the
        # AMD/CPU fallback profile.
        pass
    return out


def _detect_kernel_support(gpu_cap: Optional[tuple]) -> dict:
    """Detect which acceleration kernels are actually usable.

    Each "supports_X" flag requires BOTH:
      1. Hardware capability (compute capability ≥ minimum), AND
      2. The kernel module imports without error.

    Without (2), the kernel might be on PATH but missing CUDA libs,
    a wrong torch version, etc. — checking the import is the only
    reliable way to know it'll actually load at generation time.
    """
    out = {
        "supports_fp8": False,
        "supports_nvfp4": False,
        "supports_sage": False,      # sage v1 (sm70+)
        "supports_sage2": False,     # sage v2 (sm89+)
        "supports_flash": False,
        "supports_triton": False,
    }
    if gpu_cap is None:
        return out

    major, _minor = gpu_cap

    # FP8: needs sm89+ (RTX 40xx+) AND torch built with FP8 dtypes.
    # Torch 2.1+ has float8_e4m3fn; check defensively.
    try:
        import torch
        has_fp8_dtype = hasattr(torch, "float8_e4m3fn")
        out["supports_fp8"] = (major >= 8) and has_fp8_dtype
        # The above is loose — sm80 (A100) technically supports some FP8.
        # For consumer GPUs, sm89 (RTX 4080/4090) is the practical floor
        # for transformer FP8 kernels. Use sm89 as the gate:
        if major == 8:
            out["supports_fp8"] = (gpu_cap >= (8, 9)) and has_fp8_dtype
    except Exception:
        pass

    # NVFP4: needs sm120+ (RTX 50xx) AND comfy_kitchen or lightx2v
    # kernel module installed.
    if major >= 12:
        try:
            import torch  # noqa: F401
            # The Lightx2v package registers custom torch.ops at import time.
            # Merely checking torch.ops before importing it incorrectly reports
            # a healthy RTX 50 install as lacking NVFP4 support.
            try:
                import lightx2v_kernel  # noqa: F401
            except Exception:
                pass
            has_kitchen = hasattr(getattr(__import__("torch").ops, "comfy_kitchen", None) or object(), "scaled_mm_nvfp4")
            has_lightx2v = hasattr(getattr(__import__("torch").ops, "lightx2v_kernel", None) or object(), "cutlass_scaled_nvfp4_mm_sm120")
            out["supports_nvfp4"] = bool(has_kitchen or has_lightx2v)
        except Exception:
            pass

    # Triton — required for sage and several other kernels
    try:
        import triton  # noqa: F401
        out["supports_triton"] = True
    except Exception:
        pass

    # Sage v1: sm70+ AND triton AND sageattention installed
    if major >= 7 and out["supports_triton"]:
        try:
            import sageattention  # noqa: F401
            out["supports_sage"] = True
            # Sage v2: sm89+ AND sage installed AND specific kernel symbol
            if gpu_cap >= (8, 9):
                try:
                    from sageattention import _qattn_sm89  # noqa: F401
                    out["supports_sage2"] = True
                except Exception:
                    pass
        except Exception:
            pass

    # Flash attention: pure import check (it bundles its own CUDA)
    try:
        import flash_attn  # noqa: F401
        out["supports_flash"] = True
    except Exception:
        pass

    return out


def detect_hardware() -> dict:
    """Detect user's hardware + acceleration kernel availability.

    Returns a dict suitable for JSON serialization (all values are
    primitives or simple lists). Safe to call on any system —
    fields default to "no CUDA" values when detection fails.

    Schema (all keys always present):
      cuda_available: bool
      gpu_name: str             — friendly GPU name, e.g. "NVIDIA GeForce RTX 4090"
      gpu_vram_gb: float        — total VRAM in GB, 0.0 if no CUDA
      gpu_capability: str       — e.g. "sm89", "" if no CUDA
      ram_gb: float             — total system RAM in GB
      ram_available_gb: float  — currently available system RAM in GB
      cpu_count: int            — logical CPU count
      driver_version: str       — NVIDIA driver version, or "unknown"
      torch_version: str        — PyTorch version, or "unknown"
      runtime_version: str      — CUDA/HIP runtime version, or "unknown"
      supports_mmgp_allocator: bool — bundled VMM allocator is present on this platform
      supports_fp8: bool        — FP8 quantization usable
      supports_nvfp4: bool      — NVFP4 quantization usable (RTX 50xx)
      supports_sage: bool       — sage attention v1 usable
      supports_sage2: bool      — sage attention v2 usable (sm89+)
      supports_flash: bool      — flash attention usable
      supports_triton: bool     — triton compiler available
      ram_tier: str             — "high" (nominal 64GB; ≥63.5GiB) | "low" (nominal 32GB; ≥31.5GiB) | "very_low"
      vram_tier: str            — "high" (≥24GB) | "low" (12-23GB) | "tight" (<12GB) | "none" (no CUDA)
    """
    gpu = _detect_gpu()
    kernels = _detect_kernel_support(gpu["gpu_capability_tuple"])

    try:
        memory = psutil.virtual_memory()
        ram_gb = round(memory.total / (1024 ** 3), 1)
        ram_available_gb = round(memory.available / (1024 ** 3), 1)
    except Exception:
        ram_gb = 0.0
        ram_available_gb = 0.0
    cpu_count = psutil.cpu_count(logical=True) or 1

    # VRAM is converted from bytes to GiB and rounded to 0.1 above in
    # _detect_gpu(). Apply the same units and rounding convention to RAM, but
    # allow a 0.5-GiB tolerance at nominal 32/64-GiB tier boundaries because
    # system firmware can reserve part of installed physical memory.
    ram_tier = ram_tier_for_gb(ram_gb)

    vram_gb = gpu["gpu_vram_gb"]
    if not gpu["cuda_available"]:
        vram_tier = "none"
    elif vram_gb >= 24:
        vram_tier = "high"
    elif vram_gb >= 12:
        vram_tier = "low"
    else:
        vram_tier = "tight"

    out = {
        "cuda_available": gpu["cuda_available"],
        "gpu_name": gpu["gpu_name"],
        "gpu_vram_gb": gpu["gpu_vram_gb"],
        "gpu_capability": gpu["gpu_capability"],
        "ram_gb": ram_gb,
        "ram_available_gb": ram_available_gb,
        "cpu_count": cpu_count,
        "platform": sys.platform,
        "machine": host_platform.machine(),
        "driver_version": (
            _detect_driver_version()
            if gpu["cuda_available"] and gpu["nvidia_cuda"]
            else "unknown"
        ),
        "torch_version": gpu["torch_version"],
        "runtime_version": gpu["runtime_version"],
        "supports_mmgp_allocator": _supports_mmgp_allocator(
            cuda_available=gpu["cuda_available"],
            nvidia_cuda=gpu["nvidia_cuda"],
            system=sys.platform,
            machine=host_platform.machine(),
        ),
        "ram_tier": ram_tier,
        "vram_tier": vram_tier,
        **kernels,
    }
    # gpu_capability_tuple is internal — not JSON-friendly, drop it
    # from the public schema.
    return out
