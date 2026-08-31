"""Make optional compiled acceleration packages genuinely optional.

Some Windows FlashAttention wheels can be present in package metadata while
their compiled CUDA extension cannot load, or can import while containing no
kernel for the active GPU. Diffusers checks package availability rather than
that full runtime contract, so those states can otherwise prevent Maestro
from starting or generating even though SageAttention and SDPA are valid
fallbacks.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import sys
from collections.abc import Callable
from types import ModuleType


_FLASH_ATTENTION_USABLE: bool | None = None
_FLASH_ATTENTION_DETAIL: str | None = None

# The Windows wheels installed by Maestro contain architecture-specific
# cubins rather than a portable implementation. Importing the extension only
# proves that its Python/Torch ABI matches; CUDA does not report a missing GPU
# target until the first real attention call. Keep this manifest aligned with
# the exact wheels selected in the root torch.js launcher.
_WINDOWS_FLASH_CUBINS: tuple[
    tuple[str, tuple[tuple[int, int], ...]], ...
] = (
    ("2.7.4", ((8, 9),)),
    ("2.8.2", ((10, 0), (12, 0))),
    ("2.8.3", ((8, 0), (9, 0), (10, 0), (11, 0), (12, 0))),
)


def _normalize_capability(value) -> tuple[int, int] | None:
    try:
        major, minor = value
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


def _cubin_supports_device(
    cubin_capability: tuple[int, int],
    device_capability: tuple[int, int],
) -> bool:
    """Apply CUDA cubin compatibility within one compute-capability family."""

    cubin_major, cubin_minor = cubin_capability
    device_major, device_minor = device_capability
    return cubin_major == device_major and cubin_minor <= device_minor


def flash_attention_wheel_compatibility(
    *,
    platform_name: str,
    package_version: str | None,
    cuda_capability,
) -> tuple[bool, str | None]:
    """Validate known architecture-specific Windows FlashAttention wheels.

    Linux packages and unknown/custom Windows builds retain their existing
    behavior. Maestro's pinned Windows wheels are deterministic, so known
    incompatible devices can safely use SDPA without ever launching a CUDA
    kernel that is absent from the binary.
    """

    capability = _normalize_capability(cuda_capability)
    version = str(package_version or "").strip()
    if platform_name != "win32" or capability is None or not version:
        return True, None

    compiled_capabilities = next(
        (
            cubins
            for prefix, cubins in _WINDOWS_FLASH_CUBINS
            if version.startswith(prefix)
        ),
        None,
    )
    if compiled_capabilities is None:
        return True, None
    if any(
        _cubin_supports_device(cubin, capability)
        for cubin in compiled_capabilities
    ):
        return True, None

    device_label = f"sm_{capability[0]}{capability[1]}"
    compiled_label = ", ".join(
        f"sm_{major}{minor}" for major, minor in compiled_capabilities
    )
    return False, (
        f"FlashAttention {version} contains CUDA kernels for {compiled_label}, "
        f"not this GPU ({device_label})"
    )


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


def _disable_flash_attention(
    *,
    import_module: Callable[[str], ModuleType],
    emit: Callable[[str], None],
    detail: str,
    repair_hint: bool = True,
) -> bool:
    global _FLASH_ATTENTION_USABLE, _FLASH_ATTENTION_DETAIL

    # A failed or incompatible extension import can leave partially
    # initialized modules behind. Clear them before importing Diffusers'
    # lightweight utility module, then install a sentinel that makes later
    # optional imports behave as though FlashAttention were absent.
    for name in tuple(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            sys.modules.pop(name, None)

    _disable_diffusers_flash_detection(import_module)
    sys.modules["flash_attn"] = None
    _FLASH_ATTENTION_USABLE = False
    _FLASH_ATTENTION_DETAIL = detail
    message = (
        "[Runtime] FlashAttention is unavailable "
        f"({detail}); continuing with SageAttention/SDPA."
    )
    if repair_hint:
        message += " Run Update in Pinokio to repair the optional kernel."
    emit(message)
    return False


def prepare_optional_flash_attention(
    *,
    import_module: Callable[[str], ModuleType] | None = None,
    emit: Callable[[str], None] = print,
    platform_name: str | None = None,
    package_version: str | None = None,
    cuda_capability=None,
) -> bool:
    """Validate FlashAttention before any model can select its CUDA backend.

    Returns ``True`` when the compiled package is importable and its known
    Windows wheel contains a compatible GPU target. On failure, future imports
    behave as though FlashAttention were absent so models select SDPA instead.
    """

    global _FLASH_ATTENTION_USABLE, _FLASH_ATTENTION_DETAIL
    importer = import_module or importlib.import_module
    try:
        flash_attn = importer("flash_attn")
        # Importing the package should load the extension, but checking the two
        # entry points also catches incomplete or incompatible wheel layouts.
        getattr(flash_attn, "flash_attn_func")
        getattr(flash_attn, "flash_attn_varlen_func")
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        package_is_absent = (
            isinstance(exc, ModuleNotFoundError)
            and getattr(exc, "name", None) == "flash_attn"
        )
        return _disable_flash_attention(
            import_module=importer,
            emit=emit,
            detail=detail,
            repair_hint=not package_is_absent,
        )

    detected_version = package_version
    if detected_version is None:
        try:
            detected_version = importlib.metadata.version("flash-attn")
        except importlib.metadata.PackageNotFoundError:
            detected_version = str(getattr(flash_attn, "__version__", ""))

    detected_capability = cuda_capability
    if detected_capability is None:
        try:
            torch = importer("torch")
            if torch.cuda.is_available():
                detected_capability = torch.cuda.get_device_capability()
        except Exception:
            detected_capability = None

    compatible, detail = flash_attention_wheel_compatibility(
        platform_name=platform_name or sys.platform,
        package_version=detected_version,
        cuda_capability=detected_capability,
    )
    if not compatible:
        return _disable_flash_attention(
            import_module=importer,
            emit=emit,
            detail=detail or "the installed wheel does not support this GPU",
        )

    _FLASH_ATTENTION_USABLE = True
    _FLASH_ATTENTION_DETAIL = None
    return True


def optional_flash_attention_available() -> bool:
    """Return the startup result, probing lazily for direct pipeline imports."""

    if _FLASH_ATTENTION_USABLE is None:
        return prepare_optional_flash_attention()
    return _FLASH_ATTENTION_USABLE


def optional_flash_attention_detail() -> str | None:
    return _FLASH_ATTENTION_DETAIL
