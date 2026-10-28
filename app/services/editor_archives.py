"""Portable, integrity-checked Editor project archives.

Archives contain a bounded JSON project document, a SHA-256 manifest, and only
media referenced by timeline clips or their alternate takes. Media members are
stored without recompression to keep export CPU cost low. Import stages files
inside the requested workspace before atomically publishing them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from typing import Any

from services.editor_projects import (
    MAX_EDITOR_ASSETS,
    EditorProjectError,
    _is_under,
    _project_lock,
    _project_path,
    editor_project_directory,
    normalize_editor_project,
    resolve_editor_asset,
    save_editor_project,
    workspace_directory,
)


EDITOR_ARCHIVE_FORMAT = "maestro-editor-project"
EDITOR_ARCHIVE_VERSION = 1
EDITOR_ARCHIVE_METADATA_MAX_BYTES = 32 * 1024 * 1024
EDITOR_ARCHIVE_UNPACKED_MAX_BYTES = 20 * 1024 * 1024 * 1024
EDITOR_ARCHIVE_MAX_ENTRIES = MAX_EDITOR_ASSETS + 2
_COPY_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_PATH_KEYS = {
    "path",
    "file_path",
    "filepath",
    "local_path",
    "absolute_path",
    "resolved_path",
    "thumbnail_path",
    "proxy_path",
}
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class EditorProjectArchiveError(EditorProjectError):
    """An Editor package was missing, invalid, unsafe, or over its limits."""


class EditorProjectArchiveCancelled(RuntimeError):
    """The caller cancelled an Editor package operation."""


def _cancel_if_requested(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check and cancel_check():
        raise EditorProjectArchiveCancelled("Editor package operation cancelled")


def _progress(
    callback: Callable[[int, str], None] | None,
    percent: int,
    phase: str,
) -> None:
    if callback:
        try:
            callback(max(0, min(100, int(percent))), phase)
        except Exception:
            # Progress reporting is advisory; a broken callback must not leave
            # a half-published project or package.
            pass


def _is_absolute_host_path(value: str) -> bool:
    return (
        os.path.isabs(value)
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
        or value.startswith("\\\\")
        or value.startswith("//")
        or value.lower().startswith("file://")
    )


def _strip_host_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _PATH_KEYS:
                continue
            result[str(key)] = _strip_host_paths(child)
        return result
    if isinstance(value, list):
        return [_strip_host_paths(child) for child in value]
    if isinstance(value, str) and _is_absolute_host_path(value):
        raise EditorProjectArchiveError(
            "Project metadata contains an absolute host path and cannot be packaged"
        )
    return copy.deepcopy(value)


def _referenced_asset_ids(project: Mapping[str, Any]) -> set[str]:
    referenced: set[str] = set()
    assets = project.get("assets")
    assets = assets if isinstance(assets, Mapping) else {}
    tracks = project.get("tracks")
    for track in tracks if isinstance(tracks, list) else []:
        if not isinstance(track, Mapping):
            continue
        for item in track.get("items") or []:
            if not isinstance(item, Mapping):
                continue
            asset_id = str(item.get("asset_id") or "")
            if asset_id in assets:
                referenced.add(asset_id)
            for take_id in item.get("take_asset_ids") or []:
                take_id = str(take_id or "")
                if take_id in assets:
                    referenced.add(take_id)
    return referenced


def _safe_archive_member_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise EditorProjectArchiveError("Invalid package member path")
    if "\\" in value or "\0" in value or ":" in value:
        raise EditorProjectArchiveError("Package paths must use safe relative POSIX paths")
    if value.startswith("/"):
        raise EditorProjectArchiveError("Absolute package paths are not allowed")
    parts = value.split("/")
    if len(parts) > 16 or any(part in {"", ".", ".."} for part in parts):
        raise EditorProjectArchiveError("Package path traverses outside its root")
    for part in parts:
        if part.endswith((".", " ")) or any(ord(char) < 32 for char in part):
            raise EditorProjectArchiveError("Package path contains an invalid filename")
        if part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise EditorProjectArchiveError("Package path uses a reserved filename")
    path = PurePosixPath(value)
    if path.is_absolute() or path.parts != tuple(parts):
        raise EditorProjectArchiveError("Invalid package member path")
    return "/".join(parts)


def _hash_file(
    path: str,
    *,
    cancel_check: Callable[[], bool] | None,
    progress: Callable[[int, str], None] | None,
    percent_start: int,
    percent_end: int,
    progress_phase: str,
    progress_total_bytes: int,
    progress_done_bytes: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        while True:
            _cancel_if_requested(cancel_check)
            chunk = source.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if progress_total_bytes > 0:
                percent = percent_start + int(
                    (percent_end - percent_start)
                    * min(1.0, (progress_done_bytes + size) / progress_total_bytes)
                )
                _progress(progress, percent, progress_phase)
    return digest.hexdigest(), size


def _portable_project(
    normalized: Mapping[str, Any], referenced_ids: set[str]
) -> dict[str, Any]:
    portable = copy.deepcopy(dict(normalized))
    portable_assets: dict[str, dict[str, Any]] = {}
    for asset_id in sorted(referenced_ids):
        asset = portable["assets"][asset_id]
        asset = copy.deepcopy(dict(asset))
        for key in ("path", "workspace", "missing", "preview"):
            asset.pop(key, None)
        asset["origin"] = "project"
        portable_assets[asset_id] = asset
    portable["assets"] = portable_assets
    return _strip_host_paths(portable)


def _package_filename(asset: Mapping[str, Any], asset_id: str, ordinal: int) -> str:
    name = os.path.basename(str(asset.get("name") or "media"))
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._ ")
    if not name:
        name = "media"
    if name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        name = f"asset-{name}"
    return _safe_archive_member_path(f"media/{ordinal:04d}_{asset_id}_{name}")


def _package_target(output_path: str) -> tuple[str, str]:
    requested = os.path.abspath(output_path)
    parent_requested = os.path.dirname(requested)
    os.makedirs(parent_requested, exist_ok=True)
    parent = os.path.realpath(parent_requested)
    destination = os.path.join(parent, os.path.basename(requested))
    if os.path.islink(destination):
        raise EditorProjectArchiveError("Package output cannot be a symbolic link")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    return parent, destination


def export_editor_project_archive(
    project: Mapping[str, Any],
    *,
    save_root: str,
    workspace: str,
    uploads_root: str,
    output_path: str,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> str:
    """Write an atomic portable package for referenced timeline media only."""
    normalized = normalize_editor_project(project, workspace=workspace)
    referenced_ids = _referenced_asset_ids(normalized)
    if len(referenced_ids) > MAX_EDITOR_ASSETS:
        raise EditorProjectArchiveError("Project references too many media assets")

    sources: list[dict[str, Any]] = []
    total_media_bytes = 0
    for index, asset_id in enumerate(sorted(referenced_ids), start=1):
        _cancel_if_requested(cancel_check)
        asset = normalized["assets"][asset_id]
        try:
            source_path = resolve_editor_asset(
                asset,
                save_root=save_root,
                workspace=workspace,
                uploads_root=uploads_root,
            )
        except (EditorProjectError, OSError) as error:
            raise EditorProjectArchiveError(
                f"Cannot package referenced media '{asset.get('name') or asset_id}': {error}"
            ) from error
        try:
            stat_result = os.stat(source_path)
            if not os.path.isfile(source_path) or stat_result.st_size <= 0:
                raise EditorProjectArchiveError(
                    f"Referenced media '{asset.get('name') or asset_id}' is empty or unavailable"
                )
        except OSError as error:
            raise EditorProjectArchiveError(
                f"Cannot package referenced media '{asset.get('name') or asset_id}'"
            ) from error
        total_media_bytes += stat_result.st_size
        if total_media_bytes > EDITOR_ARCHIVE_UNPACKED_MAX_BYTES:
            raise EditorProjectArchiveError(
                "Referenced media exceeds the 20 GiB unpacked package limit"
            )
        sources.append({
            "asset_id": asset_id,
            "asset": asset,
            "source_path": source_path,
            "member": _package_filename(asset, asset_id, index),
            "initial_size": stat_result.st_size,
        })

    portable_project = _portable_project(normalized, referenced_ids)
    project_bytes = json.dumps(
        portable_project, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
    if len(project_bytes) > EDITOR_ARCHIVE_METADATA_MAX_BYTES:
        raise EditorProjectArchiveError("Project metadata exceeds the 32 MiB package limit")

    _progress(progress_callback, 1, "Hashing referenced media")
    media_manifest: dict[str, dict[str, Any]] = {}
    total_hash_bytes = max(1, total_media_bytes)
    hashed_bytes = 0
    for index, record in enumerate(sources):
        digest, size = _hash_file(
            record["source_path"],
            cancel_check=cancel_check,
            progress=progress_callback,
            percent_start=2,
            percent_end=25,
            progress_phase="Hashing referenced media",
            progress_total_bytes=total_hash_bytes,
            progress_done_bytes=hashed_bytes,
        )
        hashed_bytes += size
        if size != record["initial_size"]:
            raise EditorProjectArchiveError("Source media changed while packaging")
        record["size"] = size
        record["sha256"] = digest
        media_manifest[record["asset_id"]] = {
            "path": record["member"],
            "size": size,
            "sha256": digest,
        }
        _progress(
            progress_callback,
            2 + int(23 * (index + 1) / max(1, len(sources))),
            "Hashing referenced media",
        )

    manifest = {
        "format": EDITOR_ARCHIVE_FORMAT,
        "version": EDITOR_ARCHIVE_VERSION,
        "project": {
            "path": "project.json",
            "size": len(project_bytes),
            "sha256": hashlib.sha256(project_bytes).hexdigest(),
        },
        "assets": media_manifest,
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
    if len(manifest_bytes) > EDITOR_ARCHIVE_METADATA_MAX_BYTES:
        raise EditorProjectArchiveError("Package manifest exceeds the 32 MiB metadata limit")
    unpacked_size = len(project_bytes) + len(manifest_bytes) + total_media_bytes
    if unpacked_size > EDITOR_ARCHIVE_UNPACKED_MAX_BYTES:
        raise EditorProjectArchiveError("Package exceeds the 20 GiB unpacked size limit")

    parent, destination = _package_target(output_path)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".maestro-editor-package-", suffix=".tmp", dir=parent
    )
    os.close(descriptor)
    try:
        _cancel_if_requested(cancel_check)
        _progress(progress_callback, 27, "Writing package")
        with zipfile.ZipFile(temporary_path, "w", allowZip64=True) as archive:
            archive.writestr(
                "project.json", project_bytes, compress_type=zipfile.ZIP_DEFLATED
            )
            archive.writestr(
                "manifest.json", manifest_bytes, compress_type=zipfile.ZIP_DEFLATED
            )
            copied_bytes = 0
            for index, record in enumerate(sources):
                _cancel_if_requested(cancel_check)
                info = zipfile.ZipInfo(record["member"])
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                copied_hash = hashlib.sha256()
                copied_size = 0
                with open(record["source_path"], "rb") as source, archive.open(
                    info, "w", force_zip64=True
                ) as packaged:
                    while True:
                        _cancel_if_requested(cancel_check)
                        chunk = source.read(_COPY_CHUNK_BYTES)
                        if not chunk:
                            break
                        packaged.write(chunk)
                        copied_hash.update(chunk)
                        copied_size += len(chunk)
                        copied_bytes += len(chunk)
                        percent = 28 + int(
                            67 * copied_bytes / max(1, total_media_bytes)
                        )
                        _progress(progress_callback, percent, "Writing package media")
                if copied_size != record["size"] or copied_hash.hexdigest() != record["sha256"]:
                    raise EditorProjectArchiveError(
                        f"Source media '{record['asset'].get('name') or record['asset_id']}' changed while packaging"
                    )
                _progress(
                    progress_callback,
                    28 + int(67 * (index + 1) / max(1, len(sources))),
                    "Writing package media",
                )
        _cancel_if_requested(cancel_check)
        if os.path.islink(destination) or os.path.lexists(destination):
            raise FileExistsError(destination)
        if os.path.dirname(os.path.realpath(temporary_path)) != parent:
            raise EditorProjectArchiveError("Package staging file escaped its output directory")
        os.replace(temporary_path, destination)
        _progress(progress_callback, 100, "Package ready")
        return destination
    except Exception:
        if os.path.isfile(temporary_path) and not os.path.islink(temporary_path):
            if os.path.dirname(os.path.realpath(temporary_path)) == parent:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
        raise


def _json_object_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EditorProjectArchiveError(f"Package JSON contains duplicate key '{key}'")
        result[key] = value
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _validate_archive_members(
    infos: list[zipfile.ZipInfo],
) -> dict[str, zipfile.ZipInfo]:
    if len(infos) > EDITOR_ARCHIVE_MAX_ENTRIES:
        raise EditorProjectArchiveError("Package contains too many entries")
    by_name: dict[str, zipfile.ZipInfo] = {}
    folded_names: set[str] = set()
    unpacked = 0
    metadata_bytes = 0
    for info in infos:
        name = _safe_archive_member_path(info.filename)
        if info.is_dir():
            raise EditorProjectArchiveError("Directory entries are not allowed in a package")
        if name.casefold() in folded_names:
            raise EditorProjectArchiveError("Package contains duplicate member names")
        folded_names.add(name.casefold())
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if stat.S_ISLNK(mode) or (file_type not in {0, stat.S_IFREG}):
            raise EditorProjectArchiveError("Package contains a symbolic link or special file")
        if info.flag_bits & 0x1:
            raise EditorProjectArchiveError("Encrypted package entries are not supported")
        if info.file_size < 0:
            raise EditorProjectArchiveError("Package entry has an invalid size")
        unpacked += info.file_size
        if name in {"project.json", "manifest.json"}:
            metadata_bytes += info.file_size
            if info.file_size > EDITOR_ARCHIVE_METADATA_MAX_BYTES:
                raise EditorProjectArchiveError("Package metadata exceeds the 32 MiB limit")
        if unpacked > EDITOR_ARCHIVE_UNPACKED_MAX_BYTES:
            raise EditorProjectArchiveError("Package exceeds the 20 GiB unpacked size limit")
        by_name[name] = info
    if metadata_bytes > EDITOR_ARCHIVE_METADATA_MAX_BYTES:
        raise EditorProjectArchiveError("Package metadata exceeds the 32 MiB limit")
    return by_name


def _safe_import_roots(save_root: str, workspace: str) -> tuple[str, str]:
    save_root_real = os.path.realpath(os.path.abspath(save_root))
    workspace_path = os.path.abspath(workspace_directory(save_root_real, workspace))
    os.makedirs(workspace_path, exist_ok=True)
    workspace_root = os.path.realpath(workspace_path)
    if not _is_under(workspace_root, save_root_real):
        raise EditorProjectArchiveError("Import workspace escaped the output root")
    project_path = os.path.abspath(editor_project_directory(save_root_real, workspace))
    if os.path.islink(project_path):
        raise EditorProjectArchiveError("Editor project directory cannot be a symbolic link")
    os.makedirs(project_path, exist_ok=True)
    project_root = os.path.realpath(project_path)
    if os.path.dirname(project_root) != workspace_root:
        raise EditorProjectArchiveError("Editor project directory escaped the workspace")
    imports_path = os.path.join(project_root, "imports")
    if os.path.islink(imports_path):
        raise EditorProjectArchiveError("Editor imports directory cannot be a symbolic link")
    os.makedirs(imports_path, exist_ok=True)
    imports_root = os.path.realpath(imports_path)
    if os.path.dirname(imports_root) != project_root:
        raise EditorProjectArchiveError("Editor imports directory escaped its project root")
    return save_root_real, imports_root


def _remove_import_directory(imports_root: str, path: str) -> None:
    """Remove one direct staged/imported child after rechecking its exact root."""
    root = os.path.realpath(os.path.abspath(imports_root))
    candidate = os.path.abspath(path)
    if os.path.islink(candidate):
        return
    resolved = os.path.realpath(candidate)
    if resolved == root or os.path.dirname(resolved) != root:
        return
    try:
        shutil.rmtree(resolved)
    except OSError:
        pass


def _copy_verified_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: str,
    *,
    expected_size: int,
    expected_hash: str,
    staging_root: str,
    cancel_check: Callable[[], bool] | None,
    progress_callback: Callable[[int, str], None] | None,
    progress_start: int,
    progress_end: int,
    progress_done: int,
    progress_total: int,
) -> None:
    destination_parent = os.path.dirname(destination)
    os.makedirs(destination_parent, exist_ok=True)
    if not _is_under(destination_parent, staging_root):
        raise EditorProjectArchiveError("Extracted media escaped its staging directory")
    if os.path.islink(destination) or os.path.lexists(destination):
        raise EditorProjectArchiveError("Package contains conflicting media paths")
    digest = hashlib.sha256()
    size = 0
    with archive.open(info, "r") as source, open(destination, "xb") as target:
        while True:
            _cancel_if_requested(cancel_check)
            chunk = source.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > expected_size:
                raise EditorProjectArchiveError("Package member size does not match its manifest")
            digest.update(chunk)
            target.write(chunk)
            percent = progress_start + int(
                (progress_end - progress_start)
                * min(1.0, (progress_done + size) / max(1, progress_total))
            )
            _progress(progress_callback, percent, "Importing package media")
    if size != expected_size or digest.hexdigest() != expected_hash:
        try:
            os.remove(destination)
        except OSError:
            pass
        raise EditorProjectArchiveError("Package media failed SHA-256 verification")


def import_editor_project_archive(
    archive_path: str,
    *,
    save_root: str,
    workspace: str,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Verify and import a package as a new project in a contained workspace."""
    _cancel_if_requested(cancel_check)
    archive_path = os.path.realpath(os.path.abspath(archive_path))
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(archive_path)
    save_root_real = os.path.realpath(os.path.abspath(save_root))
    imports_root: str | None = None
    stage: str | None = None
    final_directory: str | None = None
    project_id: str | None = None
    project_saved = False
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = _validate_archive_members(archive.infolist())
            if not {"project.json", "manifest.json"}.issubset(members):
                raise EditorProjectArchiveError("Package is missing its project or manifest")
            project_bytes = archive.read(members["project.json"])
            manifest_bytes = archive.read(members["manifest.json"])
            try:
                manifest = json.loads(
                    manifest_bytes.decode("utf-8"),
                    object_pairs_hook=_json_object_no_duplicate_keys,
                )
                project_payload = json.loads(
                    project_bytes.decode("utf-8"),
                    object_pairs_hook=_json_object_no_duplicate_keys,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EditorProjectArchiveError("Package metadata is not valid UTF-8 JSON") from error
            if not isinstance(manifest, Mapping) or not isinstance(project_payload, Mapping):
                raise EditorProjectArchiveError("Package metadata must contain JSON objects")
            if (
                manifest.get("format") != EDITOR_ARCHIVE_FORMAT
                or manifest.get("version") != EDITOR_ARCHIVE_VERSION
            ):
                raise EditorProjectArchiveError("Unsupported Editor package format")
            project_record = manifest.get("project")
            if not isinstance(project_record, Mapping) or project_record.get("path") != "project.json":
                raise EditorProjectArchiveError("Package project manifest is invalid")
            if (
                project_record.get("size") != len(project_bytes)
                or not _is_sha256(project_record.get("sha256"))
                or hashlib.sha256(project_bytes).hexdigest() != project_record.get("sha256")
            ):
                raise EditorProjectArchiveError("Package project JSON failed SHA-256 verification")
            if _strip_host_paths(project_payload) != project_payload:
                raise EditorProjectArchiveError("Package project contains local host paths")

            try:
                imported = normalize_editor_project(project_payload, workspace=workspace)
            except (EditorProjectError, TypeError, ValueError) as error:
                raise EditorProjectArchiveError(f"Invalid packaged Editor project: {error}") from error
            manifest_assets = manifest.get("assets")
            if not isinstance(manifest_assets, Mapping):
                raise EditorProjectArchiveError("Package asset manifest is invalid")
            project_assets = imported.get("assets") or {}
            referenced_ids = _referenced_asset_ids(imported)
            if set(manifest_assets) != set(project_assets) or set(project_assets) != referenced_ids:
                raise EditorProjectArchiveError(
                    "Package media list does not match the project's active and alternate takes"
                )
            if len(project_assets) > MAX_EDITOR_ASSETS:
                raise EditorProjectArchiveError("Package references too many media assets")

            expected_names = {"project.json", "manifest.json"}
            validated_assets: list[tuple[str, str, int, str, zipfile.ZipInfo]] = []
            for asset_id in sorted(project_assets):
                record = manifest_assets.get(asset_id)
                if not isinstance(record, Mapping):
                    raise EditorProjectArchiveError("Package asset manifest entry is invalid")
                member = _safe_archive_member_path(record.get("path"))
                if not member.startswith("media/") or member == "media/":
                    raise EditorProjectArchiveError("Package media must be stored below media/")
                size = record.get("size")
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise EditorProjectArchiveError("Package media size is invalid")
                digest = record.get("sha256")
                if not _is_sha256(digest):
                    raise EditorProjectArchiveError("Package media hash is invalid")
                info = members.get(member)
                if info is None or info.file_size != size:
                    raise EditorProjectArchiveError("Package media size does not match its manifest")
                if info.compress_type != zipfile.ZIP_STORED:
                    raise EditorProjectArchiveError("Package media must use ZIP_STORED entries")
                expected_names.add(member)
                validated_assets.append((asset_id, member, size, digest, info))
            if set(members) != expected_names:
                raise EditorProjectArchiveError("Package contains unreferenced or unexpected files")

            _progress(progress_callback, 5, "Verifying package")
            total_media_bytes = sum(entry[2] for entry in validated_assets)
            if len(project_bytes) + len(manifest_bytes) + total_media_bytes > EDITOR_ARCHIVE_UNPACKED_MAX_BYTES:
                raise EditorProjectArchiveError("Package exceeds the 20 GiB unpacked size limit")
            for asset_id, _member, expected_size, expected_hash, info in validated_assets:
                _cancel_if_requested(cancel_check)
                digest = hashlib.sha256()
                measured_size = 0
                with archive.open(info, "r") as media:
                    while True:
                        _cancel_if_requested(cancel_check)
                        chunk = media.read(_COPY_CHUNK_BYTES)
                        if not chunk:
                            break
                        measured_size += len(chunk)
                        digest.update(chunk)
                if measured_size != expected_size or digest.hexdigest() != expected_hash:
                    raise EditorProjectArchiveError(
                        f"Package media '{asset_id}' failed SHA-256 verification"
                    )

            # Do not create workspace output paths until the complete manifest
            # and every media hash have passed validation.
            save_root_real, imports_root = _safe_import_roots(save_root, workspace)
            stage = tempfile.mkdtemp(prefix=".import-staging-", dir=imports_root)
            imported["assets"] = {
                asset_id: imported["assets"][asset_id]
                for asset_id, *_ in validated_assets
            }
            media_root = os.path.join(stage, "media")
            os.makedirs(media_root, exist_ok=True)
            copied_bytes = 0
            for index, (asset_id, member, size, digest, info) in enumerate(validated_assets):
                _cancel_if_requested(cancel_check)
                member = _safe_archive_member_path(member)
                target = os.path.join(stage, *member.split("/"))
                if not _is_under(target, stage):
                    raise EditorProjectArchiveError("Package media path escaped its staging root")
                _copy_verified_member(
                    archive,
                    info,
                    target,
                    expected_size=size,
                    expected_hash=digest,
                    staging_root=stage,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                    progress_start=8,
                    progress_end=85,
                    progress_done=copied_bytes,
                    progress_total=total_media_bytes,
                )
                copied_bytes += size
                _progress(
                    progress_callback,
                    8 + int(77 * (index + 1) / max(1, len(validated_assets))),
                    "Importing package media",
                )

        _cancel_if_requested(cancel_check)
        _progress(progress_callback, 90, "Publishing imported project")
        assert imports_root is not None and stage is not None
        project_id = uuid.uuid4().hex[:16]
        project_path = _project_path(save_root_real, workspace, project_id)
        final_directory = os.path.join(imports_root, project_id)
        with _project_lock:
            if os.path.lexists(project_path) or os.path.lexists(final_directory):
                raise EditorProjectArchiveError("Could not allocate a unique imported project ID")
            if (
                os.path.islink(stage)
                or os.path.dirname(os.path.realpath(stage)) != imports_root
                or os.path.islink(final_directory)
                or os.path.dirname(os.path.realpath(final_directory)) != imports_root
            ):
                raise EditorProjectArchiveError("Import staging directory escaped its output root")
            os.rename(stage, final_directory)
            stage = ""

            imported["id"] = project_id
            imported["workspace"] = workspace
            imported["revision"] = 0
            imported["created_at"] = time.time()
            imported["updated_at"] = imported["created_at"]
            for asset_id, member, _size, _digest, _info in validated_assets:
                asset_path = os.path.join(final_directory, *member.split("/"))
                if not _is_under(asset_path, save_root_real):
                    raise EditorProjectArchiveError("Imported media escaped its output root")
                imported["assets"][asset_id]["path"] = asset_path
                imported["assets"][asset_id]["origin"] = "project"
                imported["assets"][asset_id]["workspace"] = workspace
                imported["assets"][asset_id].pop("missing", None)
                imported["assets"][asset_id].pop("preview", None)
            _cancel_if_requested(cancel_check)
            saved = save_editor_project(
                save_root_real,
                workspace,
                imported,
                expected_revision=0,
            )
            project_saved = True
        _progress(progress_callback, 100, "Project imported")
        return saved
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as error:
        if isinstance(error, EditorProjectArchiveCancelled):
            raise
        raise EditorProjectArchiveError(f"Invalid Editor project package: {error}") from error
    finally:
        if stage and imports_root:
            _remove_import_directory(imports_root, stage)
        if final_directory and not project_saved and imports_root:
            # The final destination is always a freshly allocated direct child;
            # re-resolve its parent and target before recursive cleanup.
            _remove_import_directory(imports_root, final_directory)
