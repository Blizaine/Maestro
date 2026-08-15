"""Pure compatibility checks for Maestro's hardware acceleration runtime."""
from __future__ import annotations

import re
from typing import Iterable


def version_tuple(value: object, parts: int = 2) -> tuple[int, ...]:
    """Return the first numeric version components without importing packaging."""
    numbers = [int(item) for item in re.findall(r"\d+", str(value or ""))[:parts]]
    return tuple(numbers + [0] * (parts - len(numbers)))


def is_rtx50_capability(capability: object) -> bool:
    if isinstance(capability, (tuple, list)) and capability:
        try:
            return int(capability[0]) >= 12
        except (TypeError, ValueError):
            return False
    normalized = str(capability or "").lower().replace("_", "")
    match = re.search(r"sm\s*(\d+)", normalized)
    return bool(match and int(match.group(1)) >= 120)


def evaluate_runtime(
    *,
    python_version: object,
    torch_version: object,
    cuda_version: object,
    capability: object,
    triton_available: bool,
    sage_available: bool,
    lightx2v_available: bool,
) -> list[str]:
    """Return actionable warnings for a runtime that would throttle Blackwell."""
    if not is_rtx50_capability(capability):
        return []

    warnings: list[str] = []
    if version_tuple(python_version) < (3, 11):
        warnings.append("Python 3.11 or newer is required for the RTX 50 acceleration wheels")
    if version_tuple(torch_version) < (2, 10):
        warnings.append("PyTorch 2.10 or newer is required for the tested RTX 50 runtime")
    if version_tuple(cuda_version) < (13, 0):
        warnings.append("a CUDA 13 PyTorch build is required for native RTX 50 kernels")
    if not triton_available:
        warnings.append("Triton is unavailable")
    # SageAttention is a useful dense backend, but it is not part of the H3
    # Sol readiness contract. Linux can safely run the bundled Sol Triton
    # kernels (or SDPA) when an optional Sage wheel is unavailable.
    _ = sage_available
    if not lightx2v_available:
        warnings.append("the Lightx2v NVFP4 kernel is unavailable; NVFP4 will use a very slow fallback")
    return warnings


def format_runtime_warnings(warnings: Iterable[str]) -> str:
    items = [str(item).strip() for item in warnings if str(item).strip()]
    return "; ".join(items)


def recommend_h3_lora_pin_budget_mb(
    configured_budget_mb: object,
    *,
    full_checkpoint: bool,
    platform_name: object,
) -> float:
    """Return a safe MMGP page-locked-memory budget for H3 LoRAs.

    MMGP's default ``-1`` means "pin every adapter".  A Full H3 pipeline can
    already occupy nearly all of Windows' practical page-locked-memory pool,
    even when plenty of ordinary system RAM remains available.  MMGP 3.7.6
    then raises an AssertionError if the next adapter cannot be pinned.

    Keep the existing default for pruned H3, other model families, and hosts
    where pinned allocation is not known to have this practical ceiling.  For
    a Full H3 checkpoint on Windows, turn the implicit unlimited adapter
    budget into zero (ordinary/pageable RAM).  An explicit non-negative user
    budget continues to win so advanced users with unusually large hosts can
    opt back in.  Other platforms still receive the guarded runtime fallback
    if their allocator reports the same failure.
    """
    try:
        budget_mb = float(configured_budget_mb)
    except (TypeError, ValueError):
        budget_mb = -1.0
    is_windows = str(platform_name or "").strip().lower() in {"nt", "win32", "windows"}
    if full_checkpoint and is_windows and budget_mb < 0:
        return 0.0
    return budget_mb


def is_mmgp_lora_pinning_assertion(error: BaseException) -> bool:
    """Identify MMGP's failed-pinned-allocation assertion by traceback.

    LoRA validation contains other assertions that must still surface.  The
    recoverable failure is specifically the assertion raised from MMGP's
    private ``_pin_sd_to_memory`` helper after a pinned allocation fails.
    """
    if not isinstance(error, AssertionError):
        return False
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == "_pin_sd_to_memory":
            return True
        traceback = traceback.tb_next
    return False
