"""
OOM detection for the Performance Auto-Tune OOM recovery banner.

Single entry point: `detect_oom(exception, current_coefficient)` returns
either None (not an OOM failure) or a dict with structured info the UI
can use to suggest a fix.

Used by:
  - app/launch.py (Studio job exception handlers)
  - app/services/director_pipeline.py (Director pipeline exception handler)

Both surfaces attach the returned dict (when non-None) to the failure
payload as `oom_info`. The frontend OomRecoveryBanner watches for this
field and surfaces a "Lower VRAM headroom?" banner with a one-click
permanent-fix button.
"""
from __future__ import annotations

from typing import Optional


# Substrings that reliably appear in CUDA OOM tracebacks. We match on
# the stringified exception (str(e)) AND the exception type name to
# catch both raised `torch.cuda.OutOfMemoryError` and string-wrapped
# variants ("CUDA error: out of memory", etc.) that some library
# layers re-raise as RuntimeError.
_OOM_SIGNATURES = (
    "out of memory",          # the canonical message in CUDA OOMs
    "cuda out of memory",     # explicit CUDA prefix variant
    "outofmemoryerror",       # the exception class name (lowercased)
    "cublas_status_alloc",    # cuBLAS allocator failure (sometimes wraps OOM)
    "cudnn_status_alloc",     # cuDNN allocator failure (same)
)


def _suggest_lower_coefficient(current: float) -> Optional[float]:
    """Return the suggested next-lower coefficient, or None if already at floor.

    Drop by 0.10 with a floor of 0.50 (the slider's minimum). Below
    that, the user genuinely needs a different solution (smaller
    model, lower resolution, etc.) — coefficient can't help anymore.
    """
    if current <= 0.50:
        return None
    suggested = round(current - 0.10, 2)
    return max(suggested, 0.50)


def detect_oom(exception: Exception, current_coefficient: float = 0.80) -> Optional[dict]:
    """Inspect an exception and return OOM info if it looks like a CUDA OOM.

    Returns None when the exception is not OOM-related. When OOM is
    detected, returns a dict suitable for JSON serialization:
      {
        "is_oom": True,
        "current_coefficient": 0.80,
        "suggested_coefficient": 0.70,  # or None if already at floor
        "message": str(exception)[:300],   # truncated for UI display
      }

    The current_coefficient parameter should be sourced from
    server_config["vram_safety_coefficient"] at the call site so the
    UI can display the actual current value.
    """
    err_str = str(exception).lower()
    err_type = type(exception).__name__.lower()
    haystack = err_str + " " + err_type

    if not any(sig in haystack for sig in _OOM_SIGNATURES):
        return None

    suggested = _suggest_lower_coefficient(current_coefficient)
    return {
        "is_oom": True,
        "current_coefficient": current_coefficient,
        "suggested_coefficient": suggested,
        # Truncate to keep UI banners readable. Full traceback is
        # already in the server log via traceback.print_exc().
        "message": str(exception)[:300],
    }
