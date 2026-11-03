"""Read-only discovery and inventory of local ComfyUI model files.

This module deliberately does not import ComfyUI, load model tensors, or
modify discovered installations. Model hashing is only done for files that
match an explicitly supplied known asset with both an expected size and a
SHA-256 digest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml


MAX_CONFIG_BYTES = 1024 * 1024
MAX_CONFIG_FILES = 32
MAX_CONFIG_ROOTS = 256
MAX_SCAN_DEPTH = 8
MAX_SCAN_FILES = 25_000
MAX_SCAN_ENTRIES = 100_000
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024

_CATEGORY_ALIASES = {
    "checkpoint": "checkpoints",
    "checkpoints": "checkpoints",
    "diffusion_model": "diffusion_models",
    "diffusion_models": "diffusion_models",
    "unet": "diffusion_models",
    "clip": "text_encoders",
    "text_encoder": "text_encoders",
    "text_encoders": "text_encoders",
    "loras": "loras",
    "lora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "controlnets": "controlnet",
    "clip_vision": "clip_vision",
    "embeddings": "embeddings",
    "embedding": "embeddings",
    "upscale_model": "upscale_models",
    "upscale_models": "upscale_models",
    "hypernetwork": "hypernetworks",
    "hypernetworks": "hypernetworks",
    "style_model": "style_models",
    "style_models": "style_models",
    "model_patch": "model_patches",
    "model_patches": "model_patches",
    "classifiers": "classifiers",
    "configs": "configs",
    "vae_approx": "vae_approx",
    "animatediff_models": "animatediff_models",
    "animatediff_motion_lora": "animatediff_motion_lora",
    "animatediff_video_formats": "animatediff_video_formats",
    "ipadapter": "ipadapter",
    "liveportrait": "liveportrait",
    "sams": "sams",
    "blip": "blip",
    "cogvideo": "cogvideo",
    "xlabs": "xlabs",
    "instantid": "instantid",
    "layerstyle": "layerstyle",
    "llm": "llm",
    "joy_caption": "joy_caption",
    "gligen": "gligen",
    "photomaker": "photomaker",
    "insightface": "insightface",
    "diffusers": "diffusers",
    "sam": "sam",
    "ultralytics": "ultralytics",
}
_MODEL_SUFFIXES = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".ggml",
    ".onnx",
    ".model",
    ".weights",
    ".tflite",
    ".pb",
    ".engine",
}
_CONFIG_NAMES = ("extra_model_paths.yaml", "extra_models_config.yaml")
_ENV_VAR_PATTERN = re.compile(r"%([^%]+)%")


def discover_comfyui_installations(
    app_root: Path, extra_roots: list[str] | None = None
) -> list[dict[str, Any]]:
    """Find likely ComfyUI installs and configured model roots near ``app_root``.

    Discovery checks a fixed set of nearby Pinokio layouts, plus paths passed
    explicitly in ``extra_roots``. It does not walk parent trees or disks.
    Existing ComfyUI model folders and safe-loaded extra-model YAML files are
    included as ``model_paths`` entries.
    """
    anchor = Path(app_root).expanduser()
    candidates: list[tuple[Path, str]] = []
    candidates.extend((path, "desktop") for path in _desktop_install_candidates())
    for base in (anchor, anchor.parent, anchor.parent.parent):
        candidates.extend(
            [
                (base / "app" / "ComfyUI", "discovered"),
                (base / "ComfyUI", "discovered"),
                (base / "ComfyUI_windows_portable", "discovered"),
                (base / "app", "discovered"),
                (base, "discovered"),
            ]
        )

    explicit = [str(item) for item in (extra_roots or [])[:MAX_CONFIG_ROOTS]]
    candidates.extend((Path(item).expanduser(), "configured") for item in explicit)

    installations: dict[str, dict[str, Any]] = {}
    config_files_seen: set[str] = set()
    configs_read = 0

    for candidate, source in candidates:
        install = _identify_install(candidate, allow_outer=source == "configured")
        if install is None:
            continue

        install_root, comfy_root, evidence = install
        install_key = _path_key(install_root)
        record = installations.get(install_key)
        if record is None:
            record = {
                "id": _stable_id("comfyui", install_root),
                "name": install_root.name or "ComfyUI",
                "path": _path_text(install_root),
                "model_paths": [],
                "source": source,
                "available": _readable_directory(install_root),
                "evidence": evidence,
            }
            installations[install_key] = record
        elif source == "configured":
            record["source"] = "configured"

        model_paths: list[dict[str, str]] = record["model_paths"]
        for default_path in _default_model_paths(comfy_root):
            _append_model_path(model_paths, default_path["category"], Path(default_path["path"]))

        for config_path in _candidate_config_files(
            install_root, comfy_root, anchor, include_desktop=source == "desktop"
        ):
            if configs_read >= MAX_CONFIG_FILES:
                break
            config_key = _path_key(config_path)
            if config_key in config_files_seen:
                continue
            config_files_seen.add(config_key)
            entries, read = _read_model_config(config_path)
            if read:
                configs_read += 1
                for category, path in entries:
                    if _readable_directory(path):
                        _append_model_path(model_paths, category, path)

    # Explicit model directories need not themselves be ComfyUI source trees.
    # Keep them visible as configured roots instead of pretending they are a
    # validated ComfyUI install.
    for configured in (Path(item).expanduser() for item in explicit):
        if _identify_install(configured, allow_outer=True) is not None:
            continue
        model_paths = _configured_model_paths(configured)
        if not model_paths:
            continue
        key = _path_key(configured)
        if key in installations:
            for item in model_paths:
                _append_model_path(
                    installations[key]["model_paths"], item["category"], Path(item["path"])
                )
            installations[key]["source"] = "configured"
            continue
        installations[key] = {
            "id": _stable_id("comfyui-model-root", configured),
            "name": configured.name or "Configured model root",
            "path": _path_text(configured),
            "model_paths": model_paths,
            "source": "configured_model_root",
            "available": _readable_directory(configured),
            "evidence": "explicit_model_root",
        }

    results = list(installations.values())
    for item in results:
        item["model_paths"].sort(key=lambda path: (path["category"], _path_key(Path(path["path"]))))
    results.sort(key=lambda item: _path_key(Path(item["path"])))
    return results


def scan_comfyui_models(
    root_paths: list[str], known_assets: list[dict] | None = None
) -> dict[str, Any]:
    """Inventory model files beneath the supplied ComfyUI/model directories.

    Files are never opened except for opt-in digest candidates and safetensors
    headers associated with an explicitly supplied tensor-layout signature.
    The scan skips symlinks, has global depth/file/entry limits, and reports
    ``truncated`` when any limit stops traversal.
    """
    root_inputs = [Path(value).expanduser() for value in root_paths[:MAX_CONFIG_ROOTS]]
    roots: list[dict[str, Any]] = []
    scan_roots: list[tuple[Path, str]] = []
    seen_scan_roots: set[str] = set()

    for requested in root_inputs:
        available = _readable_directory(requested)
        roots.append({"path": _path_text(requested), "available": available})
        if not available:
            continue
        for scan_path, category in _expand_scan_root(requested):
            key = _path_key(scan_path)
            if key in seen_scan_roots:
                continue
            seen_scan_roots.add(key)
            scan_roots.append((scan_path, category))

    assets_by_filename = _known_assets_by_filename(known_assets or [])
    files: list[dict[str, Any]] = []
    files_seen: set[str] = set()
    visited_entries = 0
    truncated = len(root_paths) > MAX_CONFIG_ROOTS

    for scan_root, root_category in scan_roots:
        stack: list[tuple[Path, int]] = [(scan_root, 0)]
        while stack:
            directory, depth = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        visited_entries += 1
                        if visited_entries > MAX_SCAN_ENTRIES:
                            truncated = True
                            stack.clear()
                            break
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if depth >= MAX_SCAN_DEPTH:
                                    truncated = True
                                    continue
                                stack.append((Path(entry.path), depth + 1))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            path = Path(entry.path)
                            if path.suffix.casefold() not in _MODEL_SUFFIXES:
                                continue
                            path_key = _path_key(path)
                            if path_key in files_seen:
                                continue
                            if len(files) >= MAX_SCAN_FILES:
                                truncated = True
                                stack.clear()
                                break
                            files_seen.add(path_key)
                            try:
                                size = entry.stat(follow_symlinks=False).st_size
                            except OSError:
                                continue
                            category = _category_for_file(path, scan_root, root_category)
                            compatibility, reason, known_asset = _classify_file(
                                path, size, category, assets_by_filename.get(path.name.casefold(), [])
                            )
                            try:
                                relative_path = str(path.relative_to(scan_root))
                            except ValueError:
                                relative_path = path.name
                            model_file = {
                                "id": _stable_id("comfyui-model", path),
                                "category": category,
                                "path": _path_text(path),
                                "filename": path.name,
                                "size": size,
                                "compatibility": compatibility,
                                "reason": reason,
                                "relative_path": relative_path,
                            }
                            if known_asset is not None:
                                model_file["known_asset_id"] = known_asset.get("id")
                            files.append(model_file)
                        except OSError:
                            # Individual unreadable entries do not abort a scan.
                            continue
            except OSError:
                continue
            if truncated and (visited_entries > MAX_SCAN_ENTRIES or len(files) >= MAX_SCAN_FILES):
                break

    files.sort(key=lambda item: (_path_key(Path(item["path"])), item["category"]))
    return {"roots": roots, "files": files, "truncated": truncated}


def _identify_install(
    candidate: Path, *, allow_outer: bool = False
) -> tuple[Path, Path, str] | None:
    """Return (display root, actual ComfyUI root, evidence) for a candidate."""
    root = _absolute(candidate)
    if not _readable_directory(root):
        return None
    inner = root / "ComfyUI"
    if _has_comfy_markers(root):
        return root, root, "comfyui_markers"
    outer_layout = (
        root.name.casefold().startswith("comfyui_windows_portable")
        or _readable_directory(root / "python_embeded")
        or _regular_file(root / "run_nvidia_gpu.bat")
    )
    if (allow_outer or outer_layout) and _readable_directory(inner) and _has_comfy_markers(inner):
        return root, inner, "portable_comfyui_markers"
    resources_comfy = root / "resources" / "ComfyUI"
    if _readable_directory(resources_comfy) and _has_comfy_markers(resources_comfy):
        return root, resources_comfy, "desktop_resources_markers"
    if _readable_directory(root / "models"):
        return root, root, "models_directory"
    if _readable_directory(inner / "models"):
        return root, inner, "portable_models_directory"
    return None


def _has_comfy_markers(path: Path) -> bool:
    return _regular_file(path / "main.py") and _regular_file(path / "folder_paths.py")


def _default_model_paths(comfy_root: Path) -> list[dict[str, str]]:
    return _model_paths_from_base(comfy_root)


def _desktop_install_candidates() -> list[Path]:
    """Return exact ComfyUI Desktop install locations for the current user."""
    roots: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata and os.name == "nt":
        local_appdata = str(Path.home() / "AppData" / "Local")
    if local_appdata:
        roots.append(Path(local_appdata) / "Programs" / "ComfyUI")
    if os.name == "darwin":
        roots.extend(
            [
                Path("/Applications/ComfyUI.app/Contents/Resources/ComfyUI"),
                Path.home() / "Applications" / "ComfyUI.app" / "Contents" / "Resources" / "ComfyUI",
            ]
        )
    return roots


def _candidate_config_files(*roots: Path, include_desktop: bool = False) -> list[Path]:
    candidates: dict[str, Path] = {}
    for root in roots:
        for name in _CONFIG_NAMES:
            paths = (
                root / name,
                root / "user" / "default" / name,
                root / "ComfyUI" / name,
                root / "ComfyUI" / "user" / "default" / name,
            )
            for path in paths:
                candidates.setdefault(_path_key(path), path)

    # ComfyUI Desktop's user config belongs only to a recognized Desktop
    # installation; nearby standalone and portable installs stay isolated.
    if include_desktop:
        desktop_paths: list[Path] = []
        appdata = os.environ.get("APPDATA")
        if appdata:
            desktop_paths.append(Path(appdata) / "ComfyUI" / "extra_models_config.yaml")
        if os.name == "nt":
            desktop_paths.append(
                Path.home() / "AppData" / "Roaming" / "ComfyUI" / "extra_models_config.yaml"
            )
        elif os.name == "darwin":
            desktop_paths.append(
                Path.home() / "Library" / "Application Support" / "ComfyUI" / "extra_models_config.yaml"
            )
        else:
            local_config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
            desktop_paths.append(Path(local_config) / "ComfyUI" / "extra_models_config.yaml")
        for path in desktop_paths:
            candidates.setdefault(_path_key(path), path)
    return list(candidates.values())


def _read_model_config(path: Path) -> tuple[list[tuple[str, Path]], bool]:
    if not _regular_file(path):
        return [], False
    try:
        if path.stat().st_size > MAX_CONFIG_BYTES:
            return [], True
        with path.open("r", encoding="utf-8-sig") as handle:
            raw = handle.read(MAX_CONFIG_BYTES + 1)
        if len(raw.encode("utf-8")) > MAX_CONFIG_BYTES:
            return [], True
        parsed = yaml.safe_load(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError):
        # Malformed and inaccessible configs are ignored, but never executed.
        return [], True
    if not isinstance(parsed, (Mapping, list)):
        return [], True

    entries: list[tuple[str, Path]] = []
    _collect_config_paths(parsed, path.parent, path.parent, entries)
    return _dedupe_category_paths(entries), True


def _collect_config_paths(
    node: Any, config_dir: Path, inherited_base: Path, out: list[tuple[str, Path]]
) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_config_paths(item, config_dir, inherited_base, out)
        return
    if not isinstance(node, Mapping):
        return

    local_base = inherited_base
    raw_base = node.get("base_path")
    if isinstance(raw_base, str) and raw_base.strip():
        resolved = _config_path(raw_base, config_dir, inherited_base)
        if resolved is not None:
            local_base = resolved
            for model_path in _model_paths_from_base(local_base):
                out.append((model_path["category"], Path(model_path["path"])))

    for key, value in node.items():
        category = _CATEGORY_ALIASES.get(str(key).strip().casefold())
        if category:
            for raw_path in _path_values(value):
                resolved = _config_path(raw_path, config_dir, local_base)
                if resolved is not None and _readable_directory(resolved):
                    out.append((category, resolved))
        elif isinstance(value, (Mapping, list)):
            _collect_config_paths(value, config_dir, local_base, out)


def _path_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        for line in value.splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith("#"):
                yield candidate
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield from _path_values(item)
            elif isinstance(item, Mapping):
                for key in ("path", "paths", "directory", "directories"):
                    if key in item:
                        yield from _path_values(item[key])
    elif isinstance(value, Mapping):
        for key in ("path", "paths", "directory", "directories"):
            if key in value:
                yield from _path_values(value[key])


def _config_path(value: str, config_dir: Path, base: Path) -> Path | None:
    expanded = _expand_env_path(value.strip())
    if not expanded:
        return None
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return _absolute(path)


def _expand_env_path(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    # os.path.expandvars on POSIX does not expand Windows-style %NAME% entries.
    if os.name != "nt":
        expanded = _ENV_VAR_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(0)), expanded)
    return expanded


def _configured_model_paths(root: Path) -> list[dict[str, str]]:
    if not _readable_directory(root):
        return []
    category = _category_for_name(root.name)
    try:
        with os.scandir(root) as entries:
            children = []
            for index, entry in enumerate(entries):
                if index >= MAX_SCAN_ENTRIES:
                    break
                if not entry.is_symlink() and entry.is_dir(follow_symlinks=False):
                    children.append(entry)
    except OSError:
        return []

    result: list[dict[str, str]] = []
    for entry in children:
        child_category = _category_for_name(entry.name)
        if child_category:
            result.append({"category": child_category, "path": _path_text(Path(entry.path))})
    if category:
        result.append({"category": category, "path": _path_text(root)})
    elif not result:
        # A plain configured folder may contain direct model files.
        result.append({"category": "models", "path": _path_text(root)})
    return _dedupe_model_paths(result)


def _model_paths_from_base(base: Path) -> list[dict[str, str]]:
    """Find canonical category folders under a ComfyUI data or models base."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for models in (base / "models", base):
        if not _readable_directory(models):
            continue
        key = _path_key(models)
        if key in seen:
            continue
        seen.add(key)
        try:
            with os.scandir(models) as entries:
                for index, entry in enumerate(entries):
                    if index >= MAX_SCAN_ENTRIES:
                        break
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        continue
                    category = _category_for_name(entry.name)
                    if category:
                        result.append({"category": category, "path": _path_text(Path(entry.path))})
        except OSError:
            continue
    return _dedupe_model_paths(result)


def _expand_scan_root(root: Path) -> list[tuple[Path, str]]:
    if root.is_symlink() or not _readable_directory(root):
        return []
    models = root / "models"
    comfy_child = root / "ComfyUI"
    if _has_comfy_markers(root) and _readable_directory(models):
        return _expand_models_directory(models)
    if _readable_directory(comfy_child) and _has_comfy_markers(comfy_child):
        if _readable_directory(comfy_child / "models"):
            return _expand_models_directory(comfy_child / "models")
        return []
    if root.name.casefold() == "models":
        return _expand_models_directory(root)

    category = _category_for_name(root.name)
    if category:
        return [(root, category)]

    if _readable_directory(models):
        return _expand_models_directory(models)
    if _readable_directory(comfy_child / "models"):
        return _expand_models_directory(comfy_child / "models")
    return [(root, "models")]


def _expand_models_directory(models: Path) -> list[tuple[Path, str]]:
    # Traverse the aggregate models directory as one bounded root. Per-file
    # categories are recovered from relative path segments by the scanner.
    return [(models, "models")] if _readable_directory(models) else []


def _category_for_file(path: Path, scan_root: Path, root_category: str) -> str:
    try:
        relative_parts = path.relative_to(scan_root).parts[:-1]
    except ValueError:
        relative_parts = ()
    for part in reversed(relative_parts):
        category = _category_for_name(part)
        if category:
            return category
    if root_category == "models" and relative_parts:
        return re.sub(r"[^a-z0-9_.-]+", "_", relative_parts[0].casefold()) or root_category
    return root_category


def _category_for_name(name: str) -> str | None:
    folded = name.casefold().replace("-", "_").replace(" ", "_")
    return _CATEGORY_ALIASES.get(folded)


def _known_assets_by_filename(known_assets: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for asset in known_assets:
        if not isinstance(asset, Mapping):
            continue
        filename = asset.get("filename")
        if not filename and isinstance(asset.get("path"), str):
            filename = Path(str(asset["path"])).name
        if isinstance(filename, str) and filename.strip():
            result.setdefault(Path(filename).name.casefold(), []).append(asset)
    return result


def _classify_file(
    path: Path,
    size: int,
    category: str,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, str, Mapping[str, Any] | None]:
    if not candidates:
        return (
            "comfyui_only",
            "Found in a ComfyUI model path; no byte-verified Maestro asset match was supplied.",
            None,
        )

    fingerprints = {_asset_fingerprint(asset) for asset in candidates}
    if len(fingerprints) > 1:
        return (
            "needs_review",
            "Multiple known assets share this filename with different expectations.",
            None,
        )
    asset = candidates[0]
    expected_size = _asset_size(asset)
    expected_hash = _asset_hash(asset)
    layout = _asset_tensor_layout(asset)

    # A category mismatch is useful context but not proof that a file is
    # universally unsupported, so leave it for review.
    expected_category = asset.get("category")
    if isinstance(expected_category, str) and _category_for_name(expected_category) != category:
        return "needs_review", "Filename matches a known asset in another model category.", asset

    if expected_hash and expected_size is not None:
        if size != expected_size:
            return "needs_review", "Known filename matched, but the file size differs from the expected asset.", asset
        actual_hash = _sha256_file(path)
        if actual_hash is None:
            return "needs_review", "Could not read the candidate to verify its expected SHA-256 digest.", asset
        if actual_hash == expected_hash:
            return "native", "File size and SHA-256 match the supplied known asset.", asset
        return "needs_review", "Known filename and size matched, but its SHA-256 digest differs.", asset

    if layout and path.suffix.casefold() == ".safetensors":
        actual_layout = _read_safetensors_layout(path)
        if actual_layout is None:
            return "needs_review", "Could not read a bounded safetensors header for tensor-layout verification.", asset
        if _layout_matches(actual_layout, layout):
            return "native", "Safetensors tensor layout matches the supplied known asset signature.", asset
        return "needs_review", "Safetensors tensor layout does not match the supplied known asset signature.", asset

    return (
        "needs_review",
        "Known filename matched, but no complete byte or tensor-layout proof was supplied.",
        asset,
    )


def _asset_fingerprint(asset: Mapping[str, Any]) -> tuple[Any, ...]:
    layout = _asset_tensor_layout(asset)
    layout_key = tuple(
        sorted((name, tuple(spec["shape"]), spec.get("dtype")) for name, spec in layout.items())
    )
    return (
        _asset_hash(asset),
        _asset_size(asset),
        layout_key,
        _category_for_name(str(asset.get("category", ""))) if asset.get("category") else None,
    )


def _asset_hash(asset: Mapping[str, Any]) -> str | None:
    value = asset.get("expected_sha256", asset.get("sha256", asset.get("expected_hash")))
    if not isinstance(value, str):
        return None
    value = value.strip().casefold()
    if value.startswith("sha256:"):
        value = value[7:]
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        return None
    return value


def _asset_size(asset: Mapping[str, Any]) -> int | None:
    value = asset.get("expected_size", asset.get("size"))
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _asset_tensor_layout(asset: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = asset.get("tensor_layout", asset.get("safetensors_layout", asset.get("safetensors")))
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, descriptor in raw.items():
        if not isinstance(name, str) or not name or not isinstance(descriptor, Mapping):
            continue
        shape = descriptor.get("shape")
        if not isinstance(shape, (list, tuple)) or any(
            isinstance(part, bool) or not isinstance(part, int) or part < 0 for part in shape
        ):
            continue
        dtype = descriptor.get("dtype")
        result[name] = {
            "shape": tuple(shape),
            "dtype": dtype.upper() if isinstance(dtype, str) else None,
        }
    return result


def _read_safetensors_layout(path: Path) -> dict[str, dict[str, Any]] | None:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                return None
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size < 2 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
                return None
            header_bytes = handle.read(header_size)
            if len(header_bytes) != header_size:
                return None
        header = json.loads(header_bytes.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, struct.error):
        return None
    if not isinstance(header, Mapping):
        return None
    result: dict[str, dict[str, Any]] = {}
    for name, descriptor in header.items():
        if name == "__metadata__" or not isinstance(name, str) or not isinstance(descriptor, Mapping):
            continue
        shape = descriptor.get("shape")
        dtype = descriptor.get("dtype")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            continue
        if any(isinstance(part, bool) or not isinstance(part, int) or part < 0 for part in shape):
            continue
        result[name] = {"shape": tuple(shape), "dtype": dtype.upper()}
    return result


def _layout_matches(
    actual: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Mapping[str, Any]]
) -> bool:
    if not expected:
        return False
    for name, descriptor in expected.items():
        found = actual.get(name)
        if found is None or tuple(found.get("shape", ())) != tuple(descriptor["shape"]):
            return False
        expected_dtype = descriptor.get("dtype")
        if expected_dtype and found.get("dtype") != expected_dtype:
            return False
    return True


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _append_model_path(target: list[dict[str, str]], category: str, path: Path) -> None:
    if not _readable_directory(path):
        return
    key = (category, _path_key(path))
    if any((item["category"], _path_key(Path(item["path"]))) == key for item in target):
        return
    target.append({"category": category, "path": _path_text(path)})


def _dedupe_category_paths(items: Iterable[tuple[str, Path]]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[tuple[str, str]] = set()
    for category, path in items:
        key = (category, _path_key(path))
        if key not in seen:
            seen.add(key)
            result.append((category, path))
    return result


def _dedupe_model_paths(items: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item["category"], _path_key(Path(item["path"])))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _readable_directory(path: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_dir():
            return False
        with os.scandir(path):
            return True
    except OSError:
        return False


def _regular_file(path: Path) -> bool:
    try:
        return not path.is_symlink() and path.is_file()
    except OSError:
        return False


def _absolute(path: Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, RuntimeError, ValueError):
        return path


def _path_text(path: Path) -> str:
    return os.path.normpath(os.fspath(_absolute(path)))


def _path_key(path: Path) -> str:
    return os.path.normcase(_path_text(path))


def _stable_id(prefix: str, path: Path) -> str:
    digest = hashlib.sha256(_path_key(path).encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return f"{prefix}:{digest}"
