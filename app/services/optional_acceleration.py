"""Make optional compiled acceleration packages genuinely optional.

Some Windows FlashAttention wheels can be present in package metadata while
their compiled CUDA extension cannot load.  Diffusers checks the metadata
before importing its attention dispatcher, so that half-installed state can
otherwise prevent Maestro from starting even though SageAttention and SDPA
are valid fallbacks.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType


def _disable_diffusers_flash_detection(
    import_module: Callable[[str], ModuleType],
) -> None:
    """Tell already/importable Diffusers utilities not to import FlashAttention."""

    try:
        diffusers_imports = import_module("diffusers.utils.import_utils")
    except Exception:
        return

    # Diffusers 0.36 derives this flag from package metadata.  A broken DLL
    # still looks installed, so override the cached result after a real import
    # probe has proved the extension unusable.
    if hasattr(diffusers_imports, "_flash_attn_available"):
        diffusers_imports._flash_attn_available = False


def prepare_optional_flash_attention(
    *,
    import_module: Callable[[str], ModuleType] | None = None,
    emit: Callable[[str], None] = print,
) -> bool:
    """Probe FlashAttention and safely disable it when its DLL cannot load.

    Returns ``True`` when the compiled package is usable.  On any import or
    symbol-load failure, future imports behave as though FlashAttention were
    absent so the rest of the application can select SageAttention or SDPA.
    """

    importer = import_module or importlib.import_module
    try:
        flash_attn = importer("flash_attn")
        # Importing the package should load the extension, but checking the two
        # entry points also catches incomplete or incompatible wheel layouts.
        getattr(flash_attn, "flash_attn_func")
        getattr(flash_attn, "flash_attn_varlen_func")
        return True
    except Exception as exc:
        # A failed extension import can leave partially initialized modules
        # behind.  Clear them before importing Diffusers' lightweight utility
        # module, then install a sentinel that makes later optional imports
        # raise ModuleNotFoundError instead of retrying the broken DLL.
        for name in tuple(sys.modules):
            if name == "flash_attn" or name.startswith("flash_attn."):
                sys.modules.pop(name, None)

        _disable_diffusers_flash_detection(importer)
        sys.modules["flash_attn"] = None

        detail = str(exc).strip() or type(exc).__name__
        emit(
            "[Runtime] FlashAttention could not load "
            f"({detail}); continuing with SageAttention/SDPA. "
            "Run Update in Pinokio to repair the optional kernel."
        )
        return False
