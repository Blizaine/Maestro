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

from typing import Optional

# RAM detection — psutil is a hard dependency of the app, no fallback needed.
import psutil


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
    }
    try:
        import torch
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
      cpu_count: int            — logical CPU count
      supports_fp8: bool        — FP8 quantization usable
      supports_nvfp4: bool      — NVFP4 quantization usable (RTX 50xx)
      supports_sage: bool       — sage attention v1 usable
      supports_sage2: bool      — sage attention v2 usable (sm89+)
      supports_flash: bool      — flash attention usable
      supports_triton: bool     — triton compiler available
      ram_tier: str             — "high" (≥64GB) | "low" (32-63GB) | "very_low" (<32GB)
      vram_tier: str            — "high" (≥24GB) | "low" (12-23GB) | "tight" (<12GB) | "none" (no CUDA)
    """
    gpu = _detect_gpu()
    kernels = _detect_kernel_support(gpu["gpu_capability_tuple"])

    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    cpu_count = psutil.cpu_count(logical=True) or 1

    # Tier classification — matches Wan2GP's existing thresholds at
    # app/wgp.py:10640-10645. perf_recommend.py reads these tiers
    # directly, so changing them in one place flows through to the
    # recommendation table.
    if ram_gb >= 64:
        ram_tier = "high"
    elif ram_gb >= 32:
        ram_tier = "low"
    else:
        ram_tier = "very_low"

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
        "cpu_count": cpu_count,
        "ram_tier": ram_tier,
        "vram_tier": vram_tier,
        **kernels,
    }
    # gpu_capability_tuple is internal — not JSON-friendly, drop it
    # from the public schema.
    return out
