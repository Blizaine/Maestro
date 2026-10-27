"""Select one MMGP implementation before the application imports ``mmgp``.

The selection is process-wide because both implementations use the same Python
package name and keep mutable allocator, hook, and residency state in module
globals. This module deliberately does not read or persist server preferences;
the startup owner must pass the saved value explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.machinery
import json
import math
import os
from pathlib import Path
import platform as platform_module
import sys
from typing import Any, Mapping


LEGACY_RUNTIME = "legacy"
EXPERIMENTAL_PROFILE0_RUNTIME = "wangp9905"
EXPERIMENTAL_PROFILE0_LABEL = "Experimental Profile0 (WanGP9905)"
UPSTREAM_COMMIT = "9905f0df941e5ec7b0c80e32555d3a61fe415726"
UPSTREAM_METADATA_VERSION = "4.3.0"
UPSTREAM_RUNTIME_VERSION = "4.4.0"

PROFILE0_DEFAULT_OPTIONS = {
    "pinPool": True,
    "cpuCompute": "auto",
    "autoPreload": True,
    "cudaGraphs": False,
}
_PROFILE_CHOICES = frozenset((0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0))
_PROFILE_UPDATE_KEYS = ("video_profile", "image_profile", "audio_profile")
_VRAM_ALLOCATOR_CHOICES = frozenset(("default", "vmm", "vmm_spill"))
FISH_RUNTIME_LIMITATION = (
    "The WanGP9905 expert-bank API is incompatible with Maestro's current Fish "
    "Audio adapter. Use the default MMGP 4.0 runtime for Fish until its formats, "
    "residency hooks, CPU execution, and graph paths are ported and validated."
)


class MemoryRuntimeError(RuntimeError):
    """A runtime choice cannot be used safely in this process."""


class RuntimeRestartRequired(MemoryRuntimeError):
    """The requested MMGP package differs from the one already imported."""


@dataclass(frozen=True)
class RuntimeSelection:
    """Resolved package path; pass this to the later profile/model guards."""

    backend: str
    app_dir: Path
    module_root: Path
    package_dir: Path

    @property
    def label(self) -> str:
        if self.backend == EXPERIMENTAL_PROFILE0_RUNTIME:
            return EXPERIMENTAL_PROFILE0_LABEL
        return "Maestro MMGP 4.0 (default)"

    @property
    def pinned_commit(self) -> str | None:
        return UPSTREAM_COMMIT if self.backend == EXPERIMENTAL_PROFILE0_RUNTIME else None


@dataclass(frozen=True)
class AllocatorAssetStatus:
    """Static VMM asset check; it never imports MMGP or touches CUDA."""

    path: Path | None
    present: bool
    reason: str | None


def _allocator_binary_status(
    selection: RuntimeSelection,
    kind: str,
    *,
    platform: str | None = None,
    machine: str | None = None,
) -> AllocatorAssetStatus:
    """Check a staged allocator binary without importing/loading it."""
    actual_platform = sys.platform if platform is None else platform
    actual_machine = platform_module.machine() if machine is None else machine
    normalized_machine = actual_machine.lower()
    if normalized_machine in ("amd64", "x64"):
        normalized_machine = "amd64" if actual_platform == "win32" else "x86_64"
    stem = {"vram": "vmm_alloc", "ram": "ram_alloc"}.get(kind)
    if stem is None:
        raise ValueError(f"Unknown allocator kind {kind!r}.")
    filename = {
        ("win32", "amd64"): f"{stem}_win_amd64.dll",
        ("linux", "x86_64"): f"{stem}_linux_x86_64.so",
    }.get((actual_platform, normalized_machine))
    if filename is None:
        return AllocatorAssetStatus(
            None,
            False,
            f"the pinned {kind.upper()} allocator has no binary for {actual_platform}/{actual_machine}",
        )
    path = (selection.package_dir / "allocator" / filename).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        return AllocatorAssetStatus(
            path,
            False,
            f"{filename} is absent or empty; the pinned C++ source is present but this staged copy omits a usable native asset",
        )
    return AllocatorAssetStatus(path, True, None)


def active_runtime_selection(
    app_dir: str | os.PathLike[str], selection: RuntimeSelection | None = None
) -> RuntimeSelection | None:
    """Return the explicitly wired selection or match the already imported module.

    The fallback is read-only: it never imports MMGP or guesses from the saved
    preference. Unknown origins remain unknown so the settings API does not
    report an unverified runtime as active.
    """
    if isinstance(selection, RuntimeSelection):
        return selection
    module = sys.modules.get("mmgp")
    origin = _module_origin(module) if module is not None else None
    if origin is None:
        return None
    for backend in (LEGACY_RUNTIME, EXPERIMENTAL_PROFILE0_RUNTIME):
        candidate = _selection_for_backend(app_dir, backend)
        if origin == (candidate.package_dir / "__init__.py").resolve():
            return candidate
    return None


def _normalise_runtime(preference: str | None) -> str:
    if preference is None or preference == LEGACY_RUNTIME:
        return LEGACY_RUNTIME
    if preference == EXPERIMENTAL_PROFILE0_RUNTIME:
        return EXPERIMENTAL_PROFILE0_RUNTIME
    raise ValueError(
        f"Unknown MMGP runtime preference {preference!r}; expected "
        f"{LEGACY_RUNTIME!r} or {EXPERIMENTAL_PROFILE0_RUNTIME!r}."
    )


def _saved_runtime(config: Mapping[str, Any]) -> str:
    """Resolve an explicit saved value strictly, defaulting only when absent."""
    if not isinstance(config, Mapping):
        raise ValueError("The saved server configuration must be a JSON object.")
    if "memory_runtime" not in config:
        return LEGACY_RUNTIME
    value = config["memory_runtime"]
    if type(value) is not str or value not in (
        LEGACY_RUNTIME,
        EXPERIMENTAL_PROFILE0_RUNTIME,
    ):
        raise ValueError(
            f"Invalid memory_runtime {value!r}; expected an explicit "
            f"{LEGACY_RUNTIME!r} or {EXPERIMENTAL_PROFILE0_RUNTIME!r}."
        )
    return value


def read_saved_runtime(config_path: str | os.PathLike[str]) -> str:
    """Read the memory backend preference without importing Torch/MMGP or writing config.

    A missing file or missing preference keeps the historical MMGP 4.0
    backend. If the key exists, its value must be one of the two exact saved
    choices; null, empty, or unknown values are configuration errors.
    """
    try:
        with Path(config_path).open("r", encoding="utf-8") as reader:
            config = json.load(reader)
    except FileNotFoundError:
        return LEGACY_RUNTIME
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MemoryRuntimeError(
            f"Could not read the saved memory runtime from {config_path}: {error}"
        ) from error
    return _saved_runtime(config)


def startup_config_path(
    argv: list[str] | tuple[str, ...],
    *,
    cwd: str | os.PathLike[str] | None = None,
    config_filename: str = "wgp_config.json",
) -> str:
    """Resolve the same config file WGP's argparse startup will load.

    ``--config`` is a store action, so the last occurrence wins. An existing
    custom directory takes precedence; when it is absent or lacks the config,
    WGP falls back to the app's working-directory config and otherwise creates
    the requested custom config later in startup. This helper is intentionally
    pure and must remain safe before Torch or MMGP imports.
    """
    root = os.path.abspath(os.getcwd() if cwd is None else os.fspath(cwd))
    config_dir = ""
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--config":
            if index + 1 < len(argv) and not str(argv[index + 1]).startswith("-"):
                config_dir = str(argv[index + 1])
                index += 1
        elif isinstance(argument, str) and argument.startswith("--config="):
            config_dir = argument.partition("=")[2]
        index += 1

    config_dir = config_dir.strip()
    if not config_dir:
        return os.path.join(root, config_filename)

    requested_dir = os.path.abspath(os.path.join(root, config_dir))
    requested_path = os.path.join(requested_dir, config_filename)
    if os.path.isfile(requested_path):
        return requested_path

    fallback_path = os.path.join(root, config_filename)
    if os.path.isfile(fallback_path):
        return fallback_path
    return requested_path


def _selection_for_backend(app_dir: str | os.PathLike[str], backend: str) -> RuntimeSelection:
    app_root = Path(app_dir).resolve()
    module_root = (
        app_root
        if backend == LEGACY_RUNTIME
        else app_root / "mmgp_backends" / "v43"
    ).resolve()
    return RuntimeSelection(
        backend=backend,
        app_dir=app_root,
        module_root=module_root,
        package_dir=(module_root / "mmgp").resolve(),
    )


def _same_path(left: str | os.PathLike[str], right: Path) -> bool:
    try:
        return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(
            os.path.abspath(os.fspath(right))
        )
    except (TypeError, ValueError):
        return False


def _module_origin(module: Any) -> Path | None:
    origin = getattr(module, "__file__", None)
    if not origin:
        return None
    try:
        return Path(origin).resolve()
    except (OSError, RuntimeError):
        return None


def select_runtime(
    app_dir: str | os.PathLike[str], preference: str | None = None
) -> RuntimeSelection:
    """Put the selected ``mmgp`` package first on ``sys.path`` before import.

    ``None`` and ``"legacy"`` both select the current MMGP 4.0 implementation.
    ``"wangp9905"`` selects the immutable experimental source snapshot. Calling
    this function after a different ``mmgp`` package has loaded raises a clear
    restart error instead of trying to reload or alias modules.
    """

    backend = _normalise_runtime(preference)
    resolved = _selection_for_backend(app_dir, backend)
    app_root = resolved.app_dir
    module_root = resolved.module_root
    package_dir = resolved.package_dir
    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        raise MemoryRuntimeError(
            f"MMGP runtime {backend!r} is incomplete: expected {init_path}."
        )

    loaded = sys.modules.get("mmgp")
    partial = [name for name in sys.modules if name.startswith("mmgp.")]
    if loaded is None and partial:
        raise RuntimeRestartRequired(
            "MMGP submodules are already imported without the package root; "
            "restart the server before selecting a memory runtime."
        )
    if loaded is not None:
        origin = _module_origin(loaded)
        if origin != init_path.resolve():
            raise RuntimeRestartRequired(
                f"MMGP was already imported from {origin}; selecting {backend!r} "
                "requires a server restart."
            )

    # Remove duplicate/stale MMGP roots belonging to this app, then put exactly
    # one selected root first. The app root is also needed for its other modules.
    managed_roots = (
        app_root,
        (app_root / "mmgp_backends" / "v43").resolve(),
    )
    sys.path[:] = [
        entry
        for entry in sys.path
        if not any(_same_path(entry, root) for root in managed_roots)
    ]
    sys.path.insert(0, os.fspath(module_root))
    if not _same_path(module_root, app_root):
        sys.path.insert(1, os.fspath(app_root))

    if loaded is None:
        spec = importlib.machinery.PathFinder.find_spec("mmgp", [os.fspath(module_root)])
        expected = os.fspath(init_path.resolve())
        if spec is None or spec.origin is None or not _same_path(spec.origin, Path(expected)):
            raise MemoryRuntimeError(
                f"Python would not resolve MMGP from the selected source at {init_path}."
            )

    return RuntimeSelection(
        backend=resolved.backend,
        app_dir=resolved.app_dir,
        module_root=resolved.module_root,
        package_dir=resolved.package_dir,
    )


def runtime_snapshot(
    selection: RuntimeSelection | None, cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe saved/active runtimes and the static Profile 0/Fish gates.

    This is read-only and does not probe CUDA or load the native allocator.
    ``selection`` is the backend selected at process start; ``cfg`` is the
    current saved server configuration.
    """
    saved = _saved_runtime(cfg)
    if not isinstance(selection, RuntimeSelection):
        return {
            "memory_runtime": saved,
            "saved_runtime": saved,
            "active_runtime": None,
            "restart_required": True,
            "selected_label": (
                EXPERIMENTAL_PROFILE0_LABEL
                if saved == EXPERIMENTAL_PROFILE0_RUNTIME
                else "Maestro MMGP 4.0 (default)"
            ),
            "active_label": "MMGP source not verified",
            "experimental_runtime_selectable": False,
            "experimental_runtime_capability_reason": (
                "The active MMGP source is not verified; restart selection has not been verified."
            ),
            "profile0_native_available": False,
            "profile0_native_abi_verified": False,
            "profile0_native_validation_reason": "The active MMGP source is not verified.",
            "profile0_ram_native_available": False,
            "profile0_capability_reason": (
                "The active MMGP import does not match a known local backend; "
                "restart selection has not been verified."
            ),
            "profile0_ram_capability_reason": "The active MMGP source is not verified.",
            "profile0_saved_configuration_eligible": False,
            "fish_supported": saved != EXPERIMENTAL_PROFILE0_RUNTIME,
            "fish_limitation": None if saved != EXPERIMENTAL_PROFILE0_RUNTIME else FISH_RUNTIME_LIMITATION,
            "active_fish_supported": None,
            "active_fish_limitation": "The active MMGP source is not verified.",
        }
    experimental = _selection_for_backend(
        selection.app_dir, EXPERIMENTAL_PROFILE0_RUNTIME
    )
    native = _allocator_binary_status(experimental, "vram")
    ram_native = _allocator_binary_status(experimental, "ram")
    allocator = cfg.get("vram_allocator", "default")
    profile_zero_selected = any(
        key in cfg and _profile_value(cfg[key]) == 0
        for key in _PROFILE_UPDATE_KEYS
    )
    if not native.present:
        capability_reason = native.reason
    elif saved != EXPERIMENTAL_PROFILE0_RUNTIME:
        capability_reason = (
            "Select Experimental Profile0 (WanGP9905) and vram_allocator='vmm' "
            "before saving a Profile 0 memory profile."
        )
    elif allocator != "vmm":
        capability_reason = "Profile 0 requires the saved vram_allocator='vmm' choice."
    elif not profile_zero_selected:
        capability_reason = (
            "The VMM binary and settings are ready for Profile 0, but no saved "
            "video, image, or audio profile is 0."
        )
    else:
        capability_reason = (
            "The saved settings satisfy static file-presence checks only; startup "
            "must validate the VMM binary ABI, NVIDIA driver/device support, and "
            "activate VMM after restart."
        )

    selected_fish_supported = saved != EXPERIMENTAL_PROFILE0_RUNTIME
    active_fish_supported = selection.backend != EXPERIMENTAL_PROFILE0_RUNTIME
    return {
        "saved_runtime": saved,
        "memory_runtime": saved,
        "active_runtime": selection.backend,
        "restart_required": saved != selection.backend,
        "selected_label": (
            EXPERIMENTAL_PROFILE0_LABEL
            if saved == EXPERIMENTAL_PROFILE0_RUNTIME
            else "Maestro MMGP 4.0 (default)"
        ),
        "active_label": selection.label,
        "experimental_runtime_selectable": native.present,
        "experimental_runtime_capability_reason": (
            "The pinned VMM file is present; file presence does not establish ABI or GPU safety."
            if native.present else native.reason
        ),
        "profile0_native_available": native.present,
        "profile0_native_abi_verified": False,
        "profile0_native_validation_reason": (
            "This snapshot checks file presence only; the native ABI, driver, and GPU behavior have not been validated."
        ),
        "profile0_capability_reason": capability_reason,
        "profile0_ram_native_available": ram_native.present,
        "profile0_ram_capability_reason": ram_native.reason or (
            "The companion RAM allocator binary is present; startup must verify "
            "its PyTorch ABI before enabling it."
        ),
        "profile0_saved_configuration_eligible": (
            saved == EXPERIMENTAL_PROFILE0_RUNTIME
            and allocator == "vmm"
            and native.present
        ),
        "fish_supported": selected_fish_supported,
        "fish_limitation": None if selected_fish_supported else FISH_RUNTIME_LIMITATION,
        "active_fish_supported": active_fish_supported,
        "active_fish_limitation": None if active_fish_supported else FISH_RUNTIME_LIMITATION,
    }


def validate_runtime_updates(
    body: Mapping[str, Any],
    cfg: Mapping[str, Any],
    selection: RuntimeSelection,
) -> dict[str, Any]:
    """Validate a partial System API update without mutating body or config.

    The returned mapping contains only the runtime, allocator, and profile
    fields present in ``body``. Callers can validate this result alongside
    their other request fields, then mutate/persist only after all validation
    succeeds.
    """
    if not isinstance(body, Mapping):
        raise ValueError("Settings must be a JSON object.")
    if not isinstance(cfg, Mapping):
        raise ValueError("The saved server configuration must be a JSON object.")
    if not isinstance(selection, RuntimeSelection):
        raise ValueError("An active RuntimeSelection is required to validate updates.")

    saved_backend = _saved_runtime(cfg)
    validated: dict[str, Any] = {}
    if "memory_runtime" in body:
        saved_backend = _saved_runtime({"memory_runtime": body["memory_runtime"]})
        validated["memory_runtime"] = saved_backend

    if body.get("memory_runtime") == EXPERIMENTAL_PROFILE0_RUNTIME:
        native = allocator_asset_status(
            _selection_for_backend(selection.app_dir, EXPERIMENTAL_PROFILE0_RUNTIME)
        )
        if not native.present:
            raise ValueError(
                "Experimental Profile0 (WanGP9905) cannot be selected or saved until "
                f"its pinned VMM allocator asset is present: {native.reason}."
            )

    profiles: dict[str, Any] = {}
    for key in _PROFILE_UPDATE_KEYS:
        if key in cfg:
            _profile_value(cfg[key])
            profiles[key] = cfg[key]
        if key in body:
            _profile_value(body[key])
            profiles[key] = body[key]
            validated[key] = body[key]

    allocator = body.get("vram_allocator", cfg.get("vram_allocator", "default"))
    if "vram_allocator" in body:
        if type(allocator) is not str or allocator not in _VRAM_ALLOCATOR_CHOICES:
            raise ValueError(f"Invalid vram_allocator: {allocator!r}")
        validated["vram_allocator"] = allocator

    if saved_backend == EXPERIMENTAL_PROFILE0_RUNTIME:
        read_ahead = body.get("read_ahead", cfg.get("read_ahead", False))
        if type(read_ahead) is not bool:
            raise ValueError(f"Invalid read_ahead: {read_ahead!r}")
        if read_ahead:
            raise ValueError(
                "The Experimental Profile0 (WanGP9905) runtime removed read-ahead. "
                "Turn read-ahead off explicitly before selecting this runtime."
            )
        ram_allocator = body.get("ram_allocator", cfg.get("ram_allocator", "default"))
        if type(ram_allocator) is not str or ram_allocator not in ("default", "mmgp"):
            raise ValueError(f"Invalid ram_allocator: {ram_allocator!r}")
        if ram_allocator == "mmgp":
            native_ram = _allocator_binary_status(
                _selection_for_backend(selection.app_dir, EXPERIMENTAL_PROFILE0_RUNTIME),
                "ram",
            )
            if not native_ram.present:
                raise ValueError(
                    "The Experimental Profile0 (WanGP9905) runtime cannot use the "
                    f"saved MMGP RAM allocator until its companion binary is present: {native_ram.reason}. "
                    "Change RAM Allocator to PyTorch default explicitly first."
                )

    if any(_profile_value(value) == 0 for value in profiles.values()):
        if saved_backend != EXPERIMENTAL_PROFILE0_RUNTIME:
            raise ValueError(
                "Profile 0 requires memory_runtime='wangp9905' (Experimental "
                "Profile0 (WanGP9905)); existing profiles are preserved."
            )
        if allocator != "vmm":
            raise ValueError(
                "Profile 0 requires vram_allocator='vmm'; default or vmm_spill "
                "does not satisfy this runtime gate."
            )
        native = allocator_asset_status(
            _selection_for_backend(selection.app_dir, EXPERIMENTAL_PROFILE0_RUNTIME)
        )
        if not native.present:
            raise ValueError(
                "Profile 0 cannot be saved until the pinned native VMM allocator "
                f"is present: {native.reason}."
            )

    return validated


def validate_system_memory_updates(
    body: Mapping[str, Any],
    cfg: Mapping[str, Any],
    selection: RuntimeSelection,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate System API memory fields before any config mutation.

    Returns ``(memory_updates, runtime_updates)``. This combines the existing
    Maestro memory-setting validator with runtime-specific compatibility
    checks, while leaving persistence and server mutation to the API layer.
    """
    from services.memory_settings import validate_memory_updates

    memory_updates = validate_memory_updates(body)
    runtime_updates = validate_runtime_updates(body, cfg, selection)
    return memory_updates, runtime_updates


def verify_runtime_import(selection: RuntimeSelection, module: Any | None = None) -> str:
    """Confirm the real import came from the selected source and version."""

    module = sys.modules.get("mmgp") if module is None else module
    if module is None:
        raise MemoryRuntimeError(
            "MMGP has not been imported yet; select its runtime before importing it."
        )
    origin = _module_origin(module)
    expected = (selection.package_dir / "__init__.py").resolve()
    if origin != expected:
        raise RuntimeRestartRequired(
            f"Imported MMGP from {origin}, expected {expected}; restart the server "
            "to avoid mixing backend module state."
        )
    version = str(getattr(module, "__version__", ""))
    required = "4.0.0" if selection.backend == LEGACY_RUNTIME else UPSTREAM_RUNTIME_VERSION
    if version != required:
        raise MemoryRuntimeError(
            f"MMGP source at {origin} reports version {version!r}; expected {required!r}."
        )
    return version


def _profile_value(profile: int | float) -> float:
    if isinstance(profile, bool):
        raise ValueError("A boolean is not a memory profile number.")
    try:
        value = float(profile)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid memory profile {profile!r}.") from error
    if not math.isfinite(value) or value not in _PROFILE_CHOICES:
        raise ValueError(
            f"Unsupported memory profile {profile!r}; valid choices are "
            "0, 1, 2, 3, 3.5, 4, 4.5, and 5."
        )
    return value


def require_profile(selection: RuntimeSelection, profile: int | float) -> int | float:
    """Validate a profile without converting or remapping the caller's value."""

    value = _profile_value(profile)
    if value == 0 and selection.backend != EXPERIMENTAL_PROFILE0_RUNTIME:
        raise MemoryRuntimeError(
            "Profile 0 requires the explicitly selected Experimental Profile0 "
            "(WanGP9905) runtime; the default MMGP 4.0 backend supports profiles 1-5."
        )
    return profile


def profile_options(
    selection: RuntimeSelection,
    profile: int | float,
    manual_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return backend-safe MMGP keyword overrides, preserving manual settings.

    Profile 0 defaults to the upstream shared pin pool, automatic CPU compute,
    and dynamic preload. Existing manual options are copied over those defaults.
    WanGP9905 drops the disabled legacy readAhead kwarg and empty budget sentinel;
    enabled readAhead and CUDA graphs fail closed until compatibility is tested.
    """

    value = _profile_value(profile)
    supplied = dict(manual_options or {})
    if selection.backend == EXPERIMENTAL_PROFILE0_RUNTIME:
        require_profile(selection, profile)
        # Maestro's init_pipe always adds readAhead=False as a legacy default.
        # MMGP 4.4 removed that API, so pass no kwarg for the disabled case and
        # fail closed when a user explicitly relies on its old behavior.
        read_ahead = supplied.pop("readAhead", None)
        if read_ahead not in (None, False):
            raise MemoryRuntimeError(
                "The Experimental Profile0 (WanGP9905) backend removed MMGP's "
                "readAhead option; disable read-ahead before loading this runtime."
            )
        # init_pipe creates {} when a pipe supplies no budgets. Passing an
        # empty override erases the selected MMGP profile's own budgets.
        if supplied.get("budgets") == {}:
            supplied.pop("budgets")

    if value == 0:
        require_profile(selection, profile)
        options = dict(PROFILE0_DEFAULT_OPTIONS)
        options.update(supplied)
    else:
        options = supplied

    graph_request = options.get("cudaGraphs", False)
    if (
        selection.backend == EXPERIMENTAL_PROFILE0_RUNTIME
        and graph_request not in (None, False, [], (), "")
    ):
        raise MemoryRuntimeError(
            "WanGP9905 CUDA graph capture is disabled in this staging because "
            "Maestro graph and Fish expert parity has not been validated."
        )
    return options


def allocator_asset_status(
    selection: RuntimeSelection,
    *,
    platform: str | None = None,
    machine: str | None = None,
) -> AllocatorAssetStatus:
    """Check only the platform VMM filename; this does not load the library."""

    if selection.backend != EXPERIMENTAL_PROFILE0_RUNTIME:
        return AllocatorAssetStatus(None, False, "the selected runtime is not WanGP9905")
    return _allocator_binary_status(
        selection, "vram", platform=platform, machine=machine
    )


def require_profile_runtime(
    selection: RuntimeSelection,
    profile: int | float,
    *,
    allocator_active: str | None = None,
    platform: str | None = None,
    machine: str | None = None,
) -> None:
    """Fail before generation unless Profile 0 has its VMM allocator active."""

    require_profile(selection, profile)
    if _profile_value(profile) != 0:
        return
    status = allocator_asset_status(selection, platform=platform, machine=machine)
    if not status.present:
        raise MemoryRuntimeError(
            "Profile 0 needs the pinned MMGP VMM allocator before it can run: "
            f"{status.reason}. Build/verify the platform library and restart; "
            "automatic preload otherwise disables itself."
        )
    if allocator_active != "vmm":
        raise MemoryRuntimeError(
            f"Profile 0 needs the MMGP VMM allocator active before model loading; "
            f"the current allocator is {allocator_active!r}. Restart after selecting "
            "the experimental runtime and VMM allocator."
        )


def require_model_family(selection: RuntimeSelection, model_family: str) -> None:
    """Reject current Fish Audio models on the incompatible MMGP API snapshot."""

    family = str(model_family).strip().casefold().replace("-", "_").replace(" ", "_")
    if selection.backend == EXPERIMENTAL_PROFILE0_RUNTIME and (
        family.startswith("fish") or "fish_audio" in family
    ):
        raise MemoryRuntimeError(FISH_RUNTIME_LIMITATION)


def graph_policy(
    selection: RuntimeSelection,
    profile: int | float,
    *,
    model_family: str | None = None,
    requested: Any = False,
) -> bool:
    """Current conservative graph policy for the selected runtime/profile."""

    value = _profile_value(profile)
    wants_graphs = bool(requested)
    if model_family is not None:
        require_model_family(selection, model_family)
    if selection.backend == EXPERIMENTAL_PROFILE0_RUNTIME:
        if wants_graphs:
            raise MemoryRuntimeError(
                "CUDA graphs are not enabled with the staged WanGP9905 backend; "
                "the upstream graph implementation has not been validated against "
                "Maestro's model, Fish, cleanup, or restart paths."
            )
        return False
    # Preserve Maestro's existing Fish whole-frame graph allowance exactly.
    if wants_graphs and value not in (1.0, 3.0):
        raise MemoryRuntimeError(
            "Maestro Fish whole-frame graphs are supported only for profiles 1 and 3."
        )
    return wants_graphs
