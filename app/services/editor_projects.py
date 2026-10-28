"""Project persistence and FFmpeg rendering for Maestro Editor.

The Editor is deliberately non-destructive: project JSON contains references
to source media plus timeline instructions, while export always writes a new
file.  This module has no FastAPI or WanGP imports so its persistence, path
validation, and render-graph compiler stay inexpensive to unit test.
"""
from __future__ import annotations

import copy
from array import array
from collections import OrderedDict
import functools
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from services.editor_animation import (
    EditorAnimationError,
    KEYFRAME_VALUE_RANGES,
    editor_color_filters,
    editor_keyframe_expression,
    normalize_editor_color,
    normalize_editor_keyframes,
)
from services.editor_keying import (
    EditorKeyingError,
    editor_keying_filters,
    normalize_editor_keying,
)


EDITOR_SCHEMA_VERSION = 6
EDITOR_PROJECT_DIR = ".maestro_editor"
EDITOR_RECOVERY_DIR = "history"
EDITOR_RECOVERY_REVISIONS = 20
EDITOR_MEDIA_CACHE_DIR = "media_cache"
EDITOR_MEDIA_PREVIEW_VERSION = 3
EDITOR_MEDIA_PREVIEW_WORKERS = 2
EDITOR_MEDIA_PREVIEW_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024
EDITOR_MEDIA_PREVIEW_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
EDITOR_MEDIA_PREVIEW_ACTIVE_SECONDS = 60 * 60
EDITOR_MEDIA_PREVIEW_CLEANUP_INTERVAL_SECONDS = 60 * 60
EDITOR_PROJECT_SUMMARY_CACHE_MAX_ENTRIES = 256
MAX_EDITOR_PROJECT_DURATION_SECONDS = 24 * 60 * 60
MAX_EDITOR_ASSETS = 2000
MAX_EDITOR_TRACKS = 100
MAX_EDITOR_ITEMS_PER_TRACK = 5000
MAX_EDITOR_TAKE_IDS_PER_ITEM = 100
MAX_EDITOR_TAKE_STATES_PER_ITEM = 100
MAX_EDITOR_AI_HISTORY_PER_ITEM = 100
MAX_EDITOR_WINDOW_PROMPTS_PER_ITEM = 100
MAX_EDITOR_MARKERS = 1000
MAX_EDITOR_EXPORTS = 50
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_PREVIEW_ID_RE = re.compile(r"^[a-f0-9]{24}$")
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_MEDIA_EXTENSIONS = {
    ".aac", ".flac", ".gif", ".jpeg", ".jpg", ".m4a", ".mkv",
    ".mov", ".mp3", ".mp4", ".ogg", ".png", ".wav", ".webm",
    ".webp",
}
_VIDEO_EXTENSIONS = {".gif", ".mkv", ".mov", ".mp4", ".webm"}
_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}
_EDITOR_TRANSITIONS = {"none", "dissolve", "fade_black"}
_EDITOR_AI_TOOLS = {
    "retake", "edit_anything", "recast", "repaint", "outpaint", "upscale",
    "film_grain", "revoice", "director_rerun",
}
_EDITOR_UPSCALE_METHODS = {
    "", "flashvsr2", "flashvsr3", "flashvsr4", "flashvsr2pass2",
    "flashvsr2pass4",
    "flashvsrplus2", "flashvsrplus3", "flashvsrplus4", "flashvsrplus2pass2", "flashvsrplus2pass4",
}
_EDITOR_FONT_FAMILIES = {
    "Arial",
    "Arial Black",
    "Georgia",
    "Times New Roman",
    "Verdana",
    "Trebuchet MS",
    "Courier New",
    "Impact",
}
_project_lock = threading.RLock()
_project_summary_cache_lock = threading.RLock()
_project_summary_cache: OrderedDict[str, tuple[int, int, dict[str, Any]]] = OrderedDict()
_preview_builds_lock = threading.RLock()
_preview_build_slots = threading.BoundedSemaphore(EDITOR_MEDIA_PREVIEW_WORKERS)
_preview_builds: dict[str, "_PreviewBuildFlight"] = {}
_preview_cache_cleanup_lock = threading.Lock()
_preview_cache_cleanup_times: OrderedDict[str, float] = OrderedDict()


class _PreviewBuildFlight:
    def __init__(self, *, needs_proxy: bool, proxy_profile: str) -> None:
        self.needs_proxy = needs_proxy
        self.proxy_profile = proxy_profile
        self.event = threading.Event()
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None


class EditorProjectError(ValueError):
    """A project or asset was invalid or unsafe."""


class EditorProjectRevisionConflict(EditorProjectError):
    """A project save was based on a revision that is no longer current."""

    def __init__(self, expected_revision: int, current_revision: int) -> None:
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        super().__init__(
            "Editor project changed since it was loaded "
            f"(expected revision {expected_revision}, current revision {current_revision})"
        )


class EditorRenderCancelled(RuntimeError):
    """The user cancelled an Editor render."""


def _finite_number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _bounded_number(value: Any, default: float, low: float, high: float) -> float:
    return min(high, max(low, _finite_number(value, default)))


def _editor_revision(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EditorProjectError("Invalid Editor project revision")
    return value


def _finite_project_time(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise EditorProjectError(f"Editor {label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise EditorProjectError(f"Editor {label} must be a finite number")
    return parsed


def _validate_editor_project_input(project: Mapping[str, Any]) -> None:
    """Reject oversized or non-finite timeline input before normalization.

    Normalization historically sliced collection limits and replaced invalid
    numbers with defaults. That is useful for old optional fields, but silently
    dropping authored clips/assets is not safe when saving a project.
    """
    if not isinstance(project, Mapping):
        raise EditorProjectError("Editor project must be an object")

    assets = project.get("assets")
    if isinstance(assets, Mapping) and len(assets) > MAX_EDITOR_ASSETS:
        raise EditorProjectError(
            f"Editor project has more than {MAX_EDITOR_ASSETS} assets"
        )

    tracks = project.get("tracks")
    if not isinstance(tracks, list):
        tracks = []
    if len(tracks) > MAX_EDITOR_TRACKS:
        raise EditorProjectError(
            f"Editor project has more than {MAX_EDITOR_TRACKS} tracks"
        )
    for track_index, track in enumerate(tracks):
        if not isinstance(track, Mapping):
            continue
        items = track.get("items")
        if not isinstance(items, list):
            continue
        if len(items) > MAX_EDITOR_ITEMS_PER_TRACK:
            raise EditorProjectError(
                f"Editor track {track_index + 1} has more than "
                f"{MAX_EDITOR_ITEMS_PER_TRACK} items"
            )
        for item_index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            location = f"track {track_index + 1} item {item_index + 1}"
            # Keep the existing defaults for omitted fields, but reject explicit
            # NaN/Infinity or unparsable values instead of converting them to 0/1.
            start = _finite_project_time(item.get("start", 0.0), label=f"{location} start")
            _finite_project_time(
                item.get("duration", 1.0), label=f"{location} duration"
            )
            if start > MAX_EDITOR_PROJECT_DURATION_SECONDS:
                raise EditorProjectError(
                    f"Editor {location} starts beyond the 24-hour project limit"
                )

            raw_duration = _finite_project_time(
                item.get("duration", 1.0), label=f"{location} duration"
            )
            bounded_duration = min(
                MAX_EDITOR_PROJECT_DURATION_SECONDS,
                max(1 / 240, raw_duration),
            )
            try:
                keyframes = normalize_editor_keyframes(
                    item.get("keyframes"), duration=bounded_duration
                )
                normalize_editor_color(item.get("color"))
            except EditorAnimationError as error:
                raise EditorProjectError(f"Editor {location}: {error}") from error
            track_type = str(track.get("type") or "video").lower()
            supported_by_track = {
                "video": set(KEYFRAME_VALUE_RANGES),
                "audio": {"volume"},
                # drawtext supports a per-frame fontsize expression, so text
                # scale can be rendered while keeping alignment tied to the
                # current glyph bounds. Rotation is intentionally rejected:
                # drawtext has no rotation option and a separate raster layer
                # would change text alignment/font-metric behavior.
                "text": {"x", "y", "scale", "opacity"},
            }
            allowed_properties = supported_by_track.get(track_type)
            if allowed_properties is not None:
                unsupported = set(keyframes) - allowed_properties
                if unsupported:
                    property_name = sorted(unsupported)[0]
                    raise EditorProjectError(
                        f"Editor {location}: keyframes.{property_name} is not supported "
                        f"on a {track_type} track"
                    )
            if "color" in item and track_type != "video":
                raise EditorProjectError(
                    f"Editor {location}: color adjustments require a video track"
                )
            if "keying" in item and track_type != "video":
                raise EditorProjectError(
                    f"Editor {location}: keying requires a video track"
                )
            if "keying" in item:
                try:
                    normalize_editor_keying(item.get("keying"))
                except EditorKeyingError as error:
                    raise EditorProjectError(f"Editor {location}: {error}") from error
            if "transition_overlap" in item:
                overlap = _finite_project_time(
                    item.get("transition_overlap"),
                    label=f"{location} transition overlap",
                )
                if overlap <= 0 or overlap > MAX_EDITOR_PROJECT_DURATION_SECONDS:
                    raise EditorProjectError(
                        f"Editor {location}: transition overlap must be between 0 and 24 hours"
                    )

            take_ids = item.get("take_asset_ids")
            if isinstance(take_ids, list) and len(take_ids) > MAX_EDITOR_TAKE_IDS_PER_ITEM:
                raise EditorProjectError(
                    f"Editor {location} has more than {MAX_EDITOR_TAKE_IDS_PER_ITEM} takes"
                )
            take_states = item.get("take_states")
            if (
                isinstance(take_states, Mapping)
                and len(take_states) > MAX_EDITOR_TAKE_STATES_PER_ITEM
            ):
                raise EditorProjectError(
                    f"Editor {location} has more than "
                    f"{MAX_EDITOR_TAKE_STATES_PER_ITEM} take states"
                )
            history = item.get("ai_history")
            if isinstance(history, list) and len(history) > MAX_EDITOR_AI_HISTORY_PER_ITEM:
                raise EditorProjectError(
                    f"Editor {location} has more than "
                    f"{MAX_EDITOR_AI_HISTORY_PER_ITEM} AI history entries"
                )
            director = item.get("director")
            window_prompts = (
                director.get("window_prompts")
                if isinstance(director, Mapping)
                else None
            )
            if (
                isinstance(window_prompts, list)
                and len(window_prompts) > MAX_EDITOR_WINDOW_PROMPTS_PER_ITEM
            ):
                raise EditorProjectError(
                    f"Editor {location} has more than "
                    f"{MAX_EDITOR_WINDOW_PROMPTS_PER_ITEM} Director window prompts"
                )

    markers = project.get("markers")
    if isinstance(markers, list):
        if len(markers) > MAX_EDITOR_MARKERS:
            raise EditorProjectError(
                f"Editor project has more than {MAX_EDITOR_MARKERS} markers"
            )
        for marker_index, marker in enumerate(markers):
            if not isinstance(marker, Mapping):
                continue
            marker_time = _finite_project_time(
                marker.get("time", 0.0), label=f"marker {marker_index + 1} time"
            )
            if marker_time > MAX_EDITOR_PROJECT_DURATION_SECONDS:
                raise EditorProjectError(
                    f"Editor marker {marker_index + 1} is beyond the 24-hour project limit"
                )

    exports = project.get("exports")
    if isinstance(exports, list) and len(exports) > MAX_EDITOR_EXPORTS:
        raise EditorProjectError(
            f"Editor project has more than {MAX_EDITOR_EXPORTS} export records"
        )
    auto_history = project.get("auto_edit_history", [])
    if not isinstance(auto_history, list) or len(auto_history) > 20:
        raise EditorProjectError("Editor auto-edit history must contain at most 20 records")
    for entry in auto_history:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("action_ids"), list):
            raise EditorProjectError("Editor auto-edit history has an invalid record")
        if len(entry["action_ids"]) > 64 or any(not isinstance(value, str) or len(value) > 200 for value in entry["action_ids"]):
            raise EditorProjectError("Editor auto-edit history has invalid action ids")
        if not isinstance(entry.get("proposal_id"), str) or len(entry["proposal_id"]) > 160:
            raise EditorProjectError("Editor auto-edit history has an invalid proposal id")
        if not isinstance(entry.get("summary", ""), str) or len(entry.get("summary", "")) > 400:
            raise EditorProjectError("Editor auto-edit history summary is too long")


def _editor_font_family(value: Any) -> str:
    family = str(value or "Arial").strip()
    return family if family in _EDITOR_FONT_FAMILIES else "Arial"


def _safe_identifier(value: Any, *, fallback: Optional[str] = None) -> str:
    text = str(value or "").strip()
    if _PROJECT_ID_RE.fullmatch(text):
        return text
    if fallback is not None:
        return fallback
    raise EditorProjectError("Invalid Editor project identifier")


def _safe_workspace_name(value: Any) -> str:
    text = str(value or "default").strip() or "default"
    if text == "default":
        return text
    if (
        text in {".", ".."}
        or os.path.basename(text) != text
        or any(character in text for character in ("/", "\\", "\0"))
    ):
        raise EditorProjectError("Invalid workspace name")
    return text


def _is_under(path: str, root: str) -> bool:
    try:
        candidate = os.path.normcase(os.path.realpath(os.path.abspath(path)))
        boundary = os.path.normcase(os.path.realpath(os.path.abspath(root)))
        return os.path.commonpath([candidate, boundary]) == boundary
    except (OSError, ValueError):
        return False


def _safe_join(root: str, *parts: str) -> Optional[str]:
    candidate = os.path.realpath(os.path.join(root, *parts))
    return candidate if _is_under(candidate, root) else None


def workspace_directory(save_root: str, workspace: str) -> str:
    workspace = _safe_workspace_name(workspace)
    root = os.path.realpath(os.path.abspath(save_root))
    return root if workspace == "default" else os.path.join(root, workspace)


def editor_project_directory(save_root: str, workspace: str) -> str:
    return os.path.join(workspace_directory(save_root, workspace), EDITOR_PROJECT_DIR)


def _project_path(save_root: str, workspace: str, project_id: str) -> str:
    safe_id = _safe_identifier(project_id)
    return os.path.join(editor_project_directory(save_root, workspace), f"{safe_id}.json")


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.isfile(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass


def create_editor_project(
    *,
    name: str = "Untitled project",
    workspace: str = "default",
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
) -> dict[str, Any]:
    now = time.time()
    project_id = uuid.uuid4().hex[:16]
    return {
        "schema_version": EDITOR_SCHEMA_VERSION,
        "revision": 0,
        "id": project_id,
        "name": (str(name or "Untitled project").strip() or "Untitled project")[:120],
        "workspace": _safe_workspace_name(workspace),
        "created_at": now,
        "updated_at": now,
        "canvas": {
            "width": int(min(7680, max(64, width))),
            "height": int(min(4320, max(64, height))),
            "fps": _bounded_number(fps, 30.0, 1.0, 120.0),
            "background": "#000000",
        },
        "assets": {},
        "markers": [],
        "tracks": [
            {
                "id": "video-main",
                "name": "Main video",
                "type": "video",
                "z_index": 0,
                "muted": False,
                "locked": False,
                "items": [],
            },
            {
                "id": "audio-main",
                "name": "Audio",
                "type": "audio",
                "z_index": 0,
                "muted": False,
                "locked": False,
                "items": [],
            },
            {
                "id": "titles-main",
                "name": "Titles",
                "type": "text",
                "z_index": 10,
                "muted": False,
                "locked": False,
                "items": [],
            },
        ],
        "export": {
            "quality": "high",
            "codec": "h264",
            "encoder": "auto",
            "include_audio": True,
            "resolution": "canvas",
            "frame_rate": "project",
            "filename": "",
            "spatial_upsampling": "",
            "film_grain_intensity": 0.0,
            "film_grain_saturation": 0.5,
        },
        "exports": [],
    }


def normalize_editor_project(project: Mapping[str, Any], *, workspace: str | None = None) -> dict[str, Any]:
    _validate_editor_project_input(project)
    normalized = copy.deepcopy(dict(project))
    try:
        source_schema_version = int(project.get("schema_version", 0))
    except (TypeError, ValueError, OverflowError):
        source_schema_version = 0
    if isinstance(project.get("schema_version"), bool):
        source_schema_version = 0
    normalized["schema_version"] = EDITOR_SCHEMA_VERSION
    normalized["revision"] = _editor_revision(normalized.get("revision", 0))
    normalized["id"] = _safe_identifier(normalized.get("id"), fallback=uuid.uuid4().hex[:16])
    normalized["name"] = (
        str(normalized.get("name") or "Untitled project").strip() or "Untitled project"
    )[:120]
    normalized["workspace"] = _safe_workspace_name(workspace or normalized.get("workspace"))
    now = time.time()
    normalized["created_at"] = _finite_number(normalized.get("created_at"), now)
    # Normalization also runs for read/list operations. Preserve the stored
    # modification time here; save_editor_project is the only operation that
    # should advance it. Otherwise merely opening the Editor makes every
    # project appear newly modified and destroys useful recency ordering.
    normalized["updated_at"] = _finite_number(normalized.get("updated_at"), now)

    canvas = normalized.get("canvas") if isinstance(normalized.get("canvas"), Mapping) else {}
    background = str(canvas.get("background") or "#000000")
    normalized["canvas"] = {
        "width": int(_bounded_number(canvas.get("width"), 1920, 64, 7680)),
        "height": int(_bounded_number(canvas.get("height"), 1080, 64, 4320)),
        "fps": _bounded_number(canvas.get("fps"), 30.0, 1.0, 120.0),
        "background": background if _HEX_COLOR_RE.fullmatch(background) else "#000000",
    }

    raw_assets = normalized.get("assets")
    assets: dict[str, dict[str, Any]] = {}
    if isinstance(raw_assets, Mapping):
        for key, value in list(raw_assets.items())[:2000]:
            if not isinstance(value, Mapping):
                continue
            asset_id = _safe_identifier(value.get("id") or key, fallback=uuid.uuid4().hex[:16])
            media_type = str(value.get("type") or "video").lower()
            if media_type not in {"video", "image", "audio"}:
                continue
            origin = str(value.get("origin") or "output").lower()
            if origin not in {"output", "upload", "project"}:
                origin = "output"
            normalized_asset = {
                **dict(value),
                "id": asset_id,
                "name": os.path.basename(str(value.get("name") or "asset"))[:255],
                "type": media_type,
                "origin": origin,
                "duration": max(0.0, _finite_number(value.get("duration"), 0.0)),
                "width": max(0, int(_finite_number(value.get("width"), 0))),
                "height": max(0, int(_finite_number(value.get("height"), 0))),
                "fps": max(0.0, _finite_number(value.get("fps"), 0.0)),
                "has_audio": bool(value.get("has_audio", media_type == "audio")),
            }
            # Availability and generated preview metadata are runtime state,
            # not part of the portable, non-destructive project document.
            normalized_asset.pop("missing", None)
            normalized_asset.pop("preview", None)
            assets[asset_id] = normalized_asset
    normalized["assets"] = assets

    raw_tracks = normalized.get("tracks")
    tracks: list[dict[str, Any]] = []
    if isinstance(raw_tracks, list):
        for raw_track in raw_tracks[:100]:
            if not isinstance(raw_track, Mapping):
                continue
            track_type = str(raw_track.get("type") or "video").lower()
            if track_type not in {"video", "audio", "text"}:
                continue
            track_id = _safe_identifier(raw_track.get("id"), fallback=uuid.uuid4().hex[:16])
            items: list[dict[str, Any]] = []
            raw_items = raw_track.get("items")
            if isinstance(raw_items, list):
                for raw_item in raw_items[:5000]:
                    if not isinstance(raw_item, Mapping):
                        continue
                    item = dict(raw_item)
                    item["id"] = _safe_identifier(item.get("id"), fallback=uuid.uuid4().hex[:16])
                    item["start"] = max(0.0, _finite_number(item.get("start"), 0.0))
                    item["duration"] = _bounded_number(
                        item.get("duration"),
                        1.0,
                        1 / 240,
                        MAX_EDITOR_PROJECT_DURATION_SECONDS,
                    )
                    item["source_in"] = max(0.0, _finite_number(item.get("source_in"), 0.0))
                    item["speed"] = _bounded_number(item.get("speed"), 1.0, 0.1, 8.0)
                    item["volume"] = _bounded_number(item.get("volume"), 1.0, 0.0, 4.0)
                    item["opacity"] = _bounded_number(item.get("opacity"), 1.0, 0.0, 1.0)
                    asset_id = str(item.get("asset_id") or "")
                    if track_type != "text" and asset_id not in assets:
                        continue
                    asset = assets.get(asset_id)
                    if "keying" in item:
                        try:
                            keying = normalize_editor_keying(item.get("keying"))
                        except EditorKeyingError as error:
                            raise EditorProjectError(
                                f"Editor track {len(tracks) + 1} item {len(items) + 1}: {error}"
                            ) from error
                        if keying is not None:
                            if asset and asset.get("type") not in {"video", "image"}:
                                raise EditorProjectError(
                                    f"Editor track {len(tracks) + 1} item {len(items) + 1}: "
                                    "keying requires a video or image asset"
                                )
                            item["keying"] = keying
                        else:
                            item.pop("keying", None)
                    if asset and asset.get("type") != "image":
                        asset_duration = max(0.0, _finite_number(asset.get("duration"), 0.0))
                        if asset_duration > 0:
                            minimum_timeline_duration = 1 / 240
                            minimum_source_duration = minimum_timeline_duration * item["speed"]
                            item["source_in"] = min(
                                item["source_in"],
                                max(0.0, asset_duration - minimum_source_duration),
                            )
                            available_duration = max(
                                minimum_timeline_duration,
                                (asset_duration - item["source_in"]) / item["speed"],
                            )
                            item["duration"] = min(item["duration"], available_duration)
                    item["fade_in"] = _bounded_number(
                        item.get("fade_in"), 0.0, 0.0, item["duration"]
                    )
                    item["fade_out"] = _bounded_number(
                        item.get("fade_out"), 0.0, 0.0, item["duration"]
                    )
                    transition_in = str(item.get("transition_in") or "none")
                    transition_out = str(item.get("transition_out") or "none")
                    item["transition_in"] = (
                        transition_in if transition_in in _EDITOR_TRANSITIONS else "none"
                    )
                    item["transition_out"] = (
                        transition_out if transition_out in _EDITOR_TRANSITIONS else "none"
                    )
                    if "transition_overlap" in item:
                        item["transition_overlap"] = _finite_project_time(
                            item["transition_overlap"],
                            label="transition overlap",
                        )
                    link_group_id = str(item.get("link_group_id") or "").strip()
                    if link_group_id and _PROJECT_ID_RE.fullmatch(link_group_id):
                        item["link_group_id"] = link_group_id
                    else:
                        item.pop("link_group_id", None)
                    raw_take_ids = item.get("take_asset_ids")
                    take_ids: list[str] = []
                    if isinstance(raw_take_ids, list):
                        for raw_take_id in raw_take_ids[:100]:
                            take_id = str(raw_take_id or "")
                            if take_id in assets and take_id not in take_ids:
                                take_ids.append(take_id)
                    if asset_id and asset_id in assets and asset_id not in take_ids:
                        take_ids.insert(0, asset_id)
                    if take_ids:
                        item["take_asset_ids"] = take_ids
                    else:
                        item.pop("take_asset_ids", None)
                    raw_take_states = item.get("take_states")
                    take_states: dict[str, dict[str, float]] = {}
                    if isinstance(raw_take_states, Mapping):
                        for take_id in take_ids:
                            raw_take_state = raw_take_states.get(take_id)
                            if not isinstance(raw_take_state, Mapping):
                                continue
                            take_asset = assets.get(take_id)
                            take_speed = _bounded_number(
                                raw_take_state.get("speed"), 1.0, 0.1, 8.0
                            )
                            take_source_in = max(
                                0.0, _finite_number(raw_take_state.get("source_in"), 0.0)
                            )
                            take_duration = max(
                                0.0, _finite_number(
                                    take_asset.get("duration") if isinstance(take_asset, Mapping) else 0.0,
                                    0.0,
                                )
                            )
                            if take_duration > 0:
                                take_source_in = min(
                                    take_source_in,
                                    max(0.0, take_duration - (1 / 240) * take_speed),
                                )
                            take_states[take_id] = {
                                "source_in": take_source_in,
                                "speed": take_speed,
                            }
                    # The active timeline item's values are authoritative. A stale
                    # cached take state must never move the current edit on reload.
                    if asset_id and asset_id in take_ids:
                        take_states[asset_id] = {
                            "source_in": item["source_in"],
                            "speed": item["speed"],
                        }
                    if take_states:
                        item["take_states"] = take_states
                    else:
                        item.pop("take_states", None)
                    raw_history = item.get("ai_history")
                    ai_history: list[dict[str, Any]] = []
                    if isinstance(raw_history, list):
                        for raw_entry in raw_history[-100:]:
                            if not isinstance(raw_entry, Mapping):
                                continue
                            history_asset_id = str(raw_entry.get("asset_id") or "")
                            history_tool = str(raw_entry.get("tool") or "")
                            if history_asset_id not in assets or history_tool not in _EDITOR_AI_TOOLS:
                                continue
                            ai_history.append({
                                "id": _safe_identifier(
                                    raw_entry.get("id"), fallback=uuid.uuid4().hex[:16]
                                ),
                                "tool": history_tool,
                                "asset_id": history_asset_id,
                                "created_at": _finite_number(raw_entry.get("created_at"), now),
                            })
                    if ai_history:
                        item["ai_history"] = ai_history
                    else:
                        item.pop("ai_history", None)
                    raw_director = item.get("director")
                    if isinstance(raw_director, Mapping):
                        pipeline_id = str(raw_director.get("pipeline_id") or "").strip()
                        try:
                            clip_index = int(raw_director.get("clip_index"))
                        except (TypeError, ValueError):
                            clip_index = -1
                        if _PROJECT_ID_RE.fullmatch(pipeline_id) and clip_index >= 0:
                            director_workspace = _safe_workspace_name(
                                raw_director.get("workspace") or normalized["workspace"]
                            )
                            window_prompts = raw_director.get("window_prompts")
                            draft_video_prompt = raw_director.get("draft_video_prompt")
                            item["director"] = {
                                "pipeline_id": pipeline_id,
                                "clip_index": clip_index,
                                "pipeline_type": str(
                                    raw_director.get("pipeline_type") or ""
                                )[:80],
                                "workspace": director_workspace,
                                "video_prompt": str(
                                    raw_director.get("video_prompt") or ""
                                )[:200000],
                                "window_prompts": [
                                    str(prompt)[:200000]
                                    for prompt in window_prompts[:100]
                                    if str(prompt).strip()
                                ] if isinstance(window_prompts, list) else [],
                                **({"draft_video_prompt": str(draft_video_prompt)[:200000]}
                                   if draft_video_prompt is not None else {}),
                            }
                        else:
                            item.pop("director", None)
                    else:
                        item.pop("director", None)
                    transform = item.get("transform") if isinstance(item.get("transform"), Mapping) else {}
                    text_style_source = item.get("style") if isinstance(item.get("style"), Mapping) else {}
                    if track_type == "text":
                        legacy_transform_rotation = _finite_number(
                            transform.get("rotation"), 0.0
                        )
                        legacy_style_rotation = _finite_number(
                            text_style_source.get("rotation"), 0.0
                        )
                        if source_schema_version >= 6:
                            if abs(legacy_transform_rotation) > 1e-6 or abs(legacy_style_rotation) > 1e-6:
                                raise EditorProjectError(
                                    "Editor text rotation is not supported; use visual clips for rotation"
                                )
                        elif abs(legacy_transform_rotation) > 1e-6 or abs(legacy_style_rotation) > 1e-6:
                            legacy_rotation: dict[str, float] = {}
                            if abs(legacy_transform_rotation) > 1e-6:
                                legacy_rotation["transform"] = legacy_transform_rotation
                            if abs(legacy_style_rotation) > 1e-6:
                                legacy_rotation["style"] = legacy_style_rotation
                            item["legacy_text_rotation"] = legacy_rotation
                            warnings = normalized.get("compatibility_warnings")
                            if not isinstance(warnings, list):
                                warnings = []
                            warning = (
                                "Legacy text rotation was reset to 0 because Editor text rotation is unsupported."
                            )
                            if warning not in warnings:
                                warnings.append(warning)
                            normalized["compatibility_warnings"] = warnings[:20]
                    item["transform"] = {
                        "x": _finite_number(transform.get("x"), 0.0),
                        "y": _finite_number(transform.get("y"), 0.0),
                        "scale": _bounded_number(transform.get("scale"), 1.0, 0.05, 4.0),
                        "rotation": (
                            0.0 if track_type == "text" and source_schema_version < 6
                            else _bounded_number(transform.get("rotation"), 0.0, -360.0, 360.0)
                        ),
                    }
                    item["fit"] = "cover" if str(item.get("fit")) == "cover" else "contain"
                    item["muted"] = bool(item.get("muted", False))
                    item["disabled"] = bool(item.get("disabled", False))
                    item["pinned"] = bool(item.get("pinned", False))
                    if track_type == "text":
                        style = text_style_source
                        color = str(style.get("color") or "#ffffff")
                        background_color = str(style.get("background_color") or "#000000")
                        text_align = str(style.get("text_align") or "center")
                        item["style"] = {
                            "x": _finite_number(style.get("x"), 0.0),
                            "y": _finite_number(style.get("y"), 0.0),
                            "font_family": _editor_font_family(style.get("font_family")),
                            "font_size": int(_bounded_number(style.get("font_size"), 64, 8, 400)),
                            "color": color if _HEX_COLOR_RE.fullmatch(color) else "#ffffff",
                            "background_color": background_color
                            if _HEX_COLOR_RE.fullmatch(background_color) else "#000000",
                            "background_opacity": _bounded_number(
                                style.get("background_opacity"), 0.32, 0.0, 1.0
                            ),
                            "text_align": text_align
                            if text_align in {"left", "center", "right"} else "center",
                        }
                    try:
                        keyframes = normalize_editor_keyframes(
                            item.get("keyframes"), duration=item["duration"]
                        )
                        color_adjustments = normalize_editor_color(item.get("color"))
                    except EditorAnimationError as error:
                        raise EditorProjectError(
                            f"Editor track {len(tracks) + 1} item {len(items) + 1}: {error}"
                        ) from error
                    if keyframes:
                        item["keyframes"] = keyframes
                    else:
                        item.pop("keyframes", None)
                    if color_adjustments is not None:
                        item["color"] = color_adjustments
                    else:
                        item.pop("color", None)
                    items.append(item)
            # Legacy overlaps still normalize to a butt edit. The only
            # preserved overlap is an explicit incoming dissolve whose marker
            # agrees with the authored clip geometry.
            items.sort(key=lambda item: (item["start"], item["id"]))
            previous_end = 0.0
            previous_item: dict[str, Any] | None = None
            frame_tolerance = 1.0 / float(normalized["canvas"]["fps"])
            for item_index, item in enumerate(items):
                overlap = max(0.0, previous_end - item["start"])
                explicit_overlap = "transition_overlap" in item
                if explicit_overlap:
                    declared_overlap = float(item["transition_overlap"])
                    if track_type != "video" or previous_item is None:
                        raise EditorProjectError(
                            "Editor transition overlap requires a preceding video clip"
                        )
                    if overlap <= 1e-9:
                        raise EditorProjectError(
                            "Editor transition overlap is declared but clips do not overlap"
                        )
                    if item.get("transition_in") != "dissolve":
                        raise EditorProjectError(
                            "Editor transition overlap requires transition_in='dissolve'"
                        )
                    if abs(declared_overlap - overlap) > frame_tolerance:
                        raise EditorProjectError(
                            "Editor transition_overlap does not match the clip overlap "
                            "within one frame"
                        )
                    if (
                        overlap > float(previous_item["duration"]) + 1e-9
                        or overlap > float(item["duration"]) + 1e-9
                    ):
                        raise EditorProjectError(
                            "Editor transition overlap cannot exceed either clip duration"
                        )
                    if float(item.get("fade_in") or 0.0) + 1e-9 < overlap:
                        raise EditorProjectError(
                            "Editor transition overlap requires fade_in to cover the overlap"
                        )
                    if float(previous_item.get("fade_out") or 0.0) > 1e-9:
                        raise EditorProjectError(
                            "Editor transition overlap requires an opaque preceding clip "
                            "with no fade_out"
                        )
                    if previous_item.get("disabled"):
                        raise EditorProjectError(
                            "Editor transition overlap requires an enabled preceding clip"
                        )
                    if any(
                        float(older["start"]) + float(older["duration"])
                        > float(item["start"]) + 1e-9
                        for older in items[:item_index - 1]
                    ):
                        raise EditorProjectError(
                            "Editor transition overlap cannot overlap more than two clips"
                        )
                    item["transition_overlap"] = round(overlap, 8)
                elif overlap > 1e-9:
                    item["start"] = previous_end
                previous_end = item["start"] + item["duration"]
                previous_item = item
            tracks.append({
                **dict(raw_track),
                "id": track_id,
                "name": str(raw_track.get("name") or track_type.title())[:80],
                "type": track_type,
                "z_index": int(_finite_number(raw_track.get("z_index"), 0)),
                "muted": bool(raw_track.get("muted", False)),
                "locked": bool(raw_track.get("locked", False)),
                "volume": _bounded_number(raw_track.get("volume"), 1.0, 0.0, 4.0),
                "items": items,
            })
    if not tracks:
        tracks = create_editor_project(workspace=normalized["workspace"])["tracks"]
    normalized["tracks"] = tracks
    raw_markers = normalized.get("markers")
    markers: list[dict[str, Any]] = []
    if isinstance(raw_markers, list):
        for raw_marker in raw_markers[:1000]:
            if not isinstance(raw_marker, Mapping):
                continue
            color = str(raw_marker.get("color") or "#f59e0b")
            markers.append({
                "id": _safe_identifier(
                    raw_marker.get("id"), fallback=uuid.uuid4().hex[:16]
                ),
                "time": max(0.0, _finite_number(raw_marker.get("time"), 0.0)),
                "label": (str(raw_marker.get("label") or "Marker").strip() or "Marker")[:120],
                "color": color if _HEX_COLOR_RE.fullmatch(color) else "#f59e0b",
            })
    markers.sort(key=lambda marker: (marker["time"], marker["id"]))
    normalized["markers"] = markers
    export = normalized.get("export") if isinstance(normalized.get("export"), Mapping) else {}
    quality = str(export.get("quality") or "high").lower()
    codec = str(export.get("codec") or "h264").lower()
    encoder = str(export.get("encoder") or "auto").lower()
    resolution = str(export.get("resolution") or "canvas").lower()
    spatial_upsampling = str(export.get("spatial_upsampling") or "").lower()
    raw_frame_rate = export.get("frame_rate", "project")
    try:
        numeric_frame_rate = int(raw_frame_rate)
    except (TypeError, ValueError):
        numeric_frame_rate = 0
    normalized["export"] = {
        "quality": quality if quality in {"draft", "balanced", "high"} else "high",
        "codec": codec if codec in {"h264", "h265"} else "h264",
        "encoder": encoder
        if encoder in {"auto", "software", "nvidia", "intel", "apple"}
        else "auto",
        "include_audio": bool(export.get("include_audio", True)),
        "resolution": resolution
        if resolution in {"canvas", "2160p", "1080p", "720p", "480p"}
        else "canvas",
        "frame_rate": numeric_frame_rate
        if numeric_frame_rate in {24, 30, 60}
        else "project",
        "filename": str(export.get("filename") or "").strip()[:120],
        "spatial_upsampling": spatial_upsampling
        if spatial_upsampling in _EDITOR_UPSCALE_METHODS
        else "",
        "film_grain_intensity": max(
            0.0,
            min(1.0, _finite_number(export.get("film_grain_intensity"), 0.0)),
        ),
        "film_grain_saturation": max(
            0.0,
            min(1.0, _finite_number(export.get("film_grain_saturation"), 0.5)),
        ),
    }
    raw_exports = normalized.get("exports")
    export_history: list[dict[str, Any]] = []
    if isinstance(raw_exports, list):
        for raw_record in raw_exports[-50:]:
            if not isinstance(raw_record, Mapping):
                continue
            filename = os.path.basename(str(raw_record.get("filename") or ""))[:255]
            if not filename:
                continue
            record_codec = str(raw_record.get("codec") or "h264").lower()
            record_quality = str(raw_record.get("quality") or "high").lower()
            record_encoder = str(raw_record.get("encoder") or "auto").lower()
            record_upscale = str(
                raw_record.get("spatial_upsampling") or ""
            ).lower()
            export_history.append({
                "id": _safe_identifier(
                    raw_record.get("id"), fallback=uuid.uuid4().hex[:16]
                ),
                "filename": filename,
                "workspace": _safe_workspace_name(
                    raw_record.get("workspace") or normalized["workspace"]
                ),
                "created_at": _finite_number(raw_record.get("created_at"), now),
                "duration": max(0.0, _finite_number(raw_record.get("duration"), 0.0)),
                "width": max(0, int(_finite_number(raw_record.get("width"), 0))),
                "height": max(0, int(_finite_number(raw_record.get("height"), 0))),
                "fps": max(0.0, _finite_number(raw_record.get("fps"), 0.0)),
                "codec": record_codec if record_codec in {"h264", "h265"} else "h264",
                "quality": record_quality
                if record_quality in {"draft", "balanced", "high"}
                else "high",
                "encoder": record_encoder
                if record_encoder in {"auto", "software", "nvidia", "intel", "apple"}
                else "auto",
                "spatial_upsampling": record_upscale
                if record_upscale in _EDITOR_UPSCALE_METHODS
                else "",
                "film_grain_intensity": max(
                    0.0,
                    min(
                        1.0,
                        _finite_number(raw_record.get("film_grain_intensity"), 0.0),
                    ),
                ),
                "film_grain_saturation": max(
                    0.0,
                    min(
                        1.0,
                        _finite_number(raw_record.get("film_grain_saturation"), 0.5),
                    ),
                ),
            })
    export_history.sort(key=lambda record: record["created_at"], reverse=True)
    normalized["exports"] = export_history[:50]
    for track_index, track in enumerate(tracks):
        for item_index, item in enumerate(track["items"]):
            start = float(item["start"])
            end = start + float(item["duration"])
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start > MAX_EDITOR_PROJECT_DURATION_SECONDS
                or end > MAX_EDITOR_PROJECT_DURATION_SECONDS
            ):
                raise EditorProjectError(
                    f"Editor track {track_index + 1} item {item_index + 1} "
                    "ends beyond the 24-hour project limit"
                )
    return normalized


def save_editor_project(
    save_root: str,
    workspace: str,
    project: Mapping[str, Any],
    *,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Persist one project snapshot and advance its optimistic revision.

    A revision in the payload (or ``expected_revision``) enables stale-write
    rejection. Callers that omit a revision retain the older last-write-wins
    behavior while saved results still receive an incremented revision.
    """
    normalized = normalize_editor_project(project, workspace=workspace)
    path = _project_path(save_root, normalized["workspace"], normalized["id"])
    revision_was_explicit = expected_revision is not None or "revision" in project
    expected = (
        _editor_revision(expected_revision)
        if expected_revision is not None
        else _editor_revision(project["revision"])
        if "revision" in project
        else None
    )
    with _project_lock:
        current_revision = 0
        stored: Mapping[str, Any] | None = None
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            if not isinstance(stored, Mapping):
                raise EditorProjectError("Stored Editor project must be an object")
            current_revision = _editor_revision(stored.get("revision", 0))
            if revision_was_explicit and expected != current_revision:
                raise EditorProjectRevisionConflict(expected, current_revision)
            _archive_editor_project_revision(
                save_root,
                normalized["workspace"],
                normalized["id"],
                current_revision,
                stored,
            )
        normalized["revision"] = current_revision + 1
        normalized["updated_at"] = time.time()
        _atomic_write_json(path, normalized)
        _store_project_summary_cache(path, os.stat(path), _editor_project_summary(normalized))
    return normalized


def _editor_project_recovery_directory(
    save_root: str,
    workspace: str,
    project_id: str,
    *,
    create: bool,
) -> str:
    safe_id = _safe_identifier(project_id)
    project_dir_path = os.path.abspath(editor_project_directory(save_root, workspace))
    save_root_real = os.path.realpath(os.path.abspath(save_root))
    if os.path.islink(project_dir_path):
        raise EditorProjectError("Invalid Editor project directory")
    project_dir = os.path.realpath(project_dir_path)
    if not _is_under(project_dir, save_root_real):
        raise EditorProjectError("Editor project directory escaped its output root")

    root_path = os.path.join(project_dir, EDITOR_RECOVERY_DIR)
    if os.path.islink(root_path):
        raise EditorProjectError("Invalid Editor recovery directory")
    if create:
        os.makedirs(root_path, exist_ok=True)
    if not os.path.isdir(root_path):
        return root_path
    root = os.path.realpath(root_path)
    if os.path.dirname(root) != project_dir:
        raise EditorProjectError("Editor recovery directory escaped its project root")

    project_history_path = os.path.join(root, safe_id)
    if os.path.islink(project_history_path):
        raise EditorProjectError("Invalid Editor recovery history")
    if create:
        os.makedirs(project_history_path, exist_ok=True)
    if not os.path.isdir(project_history_path):
        return project_history_path
    project_history = os.path.realpath(project_history_path)
    if os.path.dirname(project_history) != root:
        raise EditorProjectError("Editor recovery history escaped its recovery root")
    return project_history


def _archive_editor_project_revision(
    save_root: str,
    workspace: str,
    project_id: str,
    revision: int,
    stored_project: Mapping[str, Any],
) -> None:
    history_dir = _editor_project_recovery_directory(
        save_root, workspace, project_id, create=True
    )
    snapshot = os.path.join(history_dir, f"r{revision}.json")
    if os.path.islink(snapshot):
        raise EditorProjectError("Invalid Editor recovery snapshot")
    if not os.path.lexists(snapshot):
        _atomic_write_json(snapshot, stored_project)

    revisions: list[tuple[int, str]] = []
    try:
        with os.scandir(history_dir) as entries:
            for entry in entries:
                match = re.fullmatch(r"r(\d+)\.json", entry.name)
                if not match or entry.is_symlink():
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                revisions.append((int(match.group(1)), entry.path))
    except OSError:
        return
    revisions.sort(reverse=True)
    for _revision, old_path in revisions[EDITOR_RECOVERY_REVISIONS:]:
        # Recovery pruning deletes only a direct regular file under the exact
        # per-project history directory; it never follows symlinks or recurses.
        if os.path.islink(old_path) or os.path.dirname(os.path.realpath(old_path)) != history_dir:
            continue
        try:
            os.remove(old_path)
        except OSError:
            pass


def list_editor_project_revisions(
    save_root: str,
    workspace: str,
    project_id: str,
) -> list[dict[str, Any]]:
    """List bounded prior snapshots available for safe project recovery."""
    history_dir = _editor_project_recovery_directory(
        save_root, workspace, project_id, create=False
    )
    if not os.path.isdir(history_dir):
        return []
    revisions: list[dict[str, Any]] = []
    with _project_lock, os.scandir(history_dir) as entries:
        for entry in entries:
            match = re.fullmatch(r"r(\d+)\.json", entry.name)
            if not match or entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            if os.path.dirname(os.path.realpath(entry.path)) != history_dir:
                continue
            try:
                filename_revision = int(match.group(1))
                with open(entry.path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, Mapping):
                    continue
                if _editor_revision(payload.get("revision", 0)) != filename_revision:
                    continue
                project = normalize_editor_project(payload, workspace=workspace)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            revisions.append({
                "revision": filename_revision,
                "name": project["name"],
                "updated_at": project["updated_at"],
                "duration": editor_project_duration(project),
                "asset_count": len(project.get("assets") or {}),
            })
    revisions.sort(key=lambda item: item["revision"], reverse=True)
    return revisions[:EDITOR_RECOVERY_REVISIONS]


def restore_editor_project_revision(
    save_root: str,
    workspace: str,
    project_id: str,
    revision: int,
    *,
    expected_revision: int,
) -> dict[str, Any]:
    """Restore a snapshot while advancing the live revision monotonically."""
    safe_id = _safe_identifier(project_id)
    revision = _editor_revision(revision)
    if revision < 0:
        raise EditorProjectError("Invalid Editor recovery revision")
    expected = _editor_revision(expected_revision)
    path = _project_path(save_root, workspace, safe_id)
    history_dir = _editor_project_recovery_directory(
        save_root, workspace, safe_id, create=False
    )
    snapshot = os.path.join(history_dir, f"r{revision}.json")
    if os.path.islink(snapshot) or os.path.dirname(os.path.realpath(snapshot)) != history_dir:
        raise FileNotFoundError(revision)
    with _project_lock:
        if not os.path.isfile(path):
            raise FileNotFoundError(safe_id)
        with open(path, "r", encoding="utf-8") as handle:
            current = json.load(handle)
        current_revision = _editor_revision(current.get("revision", 0))
        if current_revision != expected:
            raise EditorProjectRevisionConflict(expected, current_revision)
        if not os.path.isfile(snapshot):
            raise FileNotFoundError(revision)
        with open(snapshot, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if not isinstance(previous, Mapping):
            raise EditorProjectError("Stored Editor recovery snapshot must be an object")
        if _editor_revision(previous.get("revision", 0)) != revision:
            raise EditorProjectError(
                "Stored Editor recovery snapshot revision does not match its filename"
            )
        restored = normalize_editor_project(previous, workspace=workspace)
        restored["id"] = safe_id
        restored["revision"] = current_revision
        return save_editor_project(
            save_root,
            workspace,
            restored,
            expected_revision=current_revision,
        )


def load_editor_project(save_root: str, workspace: str, project_id: str) -> dict[str, Any]:
    path = _project_path(save_root, workspace, project_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(project_id)
    with _project_lock, open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return normalize_editor_project(payload, workspace=workspace)


def list_editor_projects(save_root: str, workspace: str) -> list[dict[str, Any]]:
    directory = editor_project_directory(save_root, workspace)
    if not os.path.isdir(directory):
        return []
    summaries: list[dict[str, Any]] = []
    with _project_lock:
        for entry in os.scandir(directory):
            if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".json"):
                continue
            try:
                before = entry.stat(follow_symlinks=False)
                cached = _get_project_summary_cache(
                    entry.path,
                    mtime_ns=before.st_mtime_ns,
                    size=before.st_size,
                )
                if cached is not None:
                    summaries.append(cached)
                    continue
                with open(entry.path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                project = normalize_editor_project(payload, workspace=workspace)
                summary = _editor_project_summary(project)
                after = os.stat(entry.path, follow_symlinks=False)
                if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
                    _store_project_summary_cache(entry.path, after, summary)
                summaries.append(summary)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                _invalidate_project_summary_cache(entry.path)
                continue
    summaries.sort(key=lambda item: item["updated_at"], reverse=True)
    return summaries


def delete_editor_project(save_root: str, workspace: str, project_id: str) -> bool:
    path = _project_path(save_root, workspace, project_id)
    with _project_lock:
        if not os.path.isfile(path):
            return False
        os.remove(path)
        _invalidate_project_summary_cache(path)
    return True


def editor_project_duration(project: Mapping[str, Any]) -> float:
    duration = 0.0
    for track in project.get("tracks") or []:
        if not isinstance(track, Mapping):
            continue
        for item in track.get("items") or []:
            if not isinstance(item, Mapping) or item.get("disabled"):
                continue
            duration = max(
                duration,
                max(0.0, _finite_number(item.get("start"), 0.0))
                + max(0.0, _finite_number(item.get("duration"), 0.0)),
            )
    return duration


def _editor_project_summary(project: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": project["id"],
        "name": project["name"],
        "workspace": project["workspace"],
        "created_at": project["created_at"],
        "updated_at": project["updated_at"],
        "duration": editor_project_duration(project),
        "asset_count": len(project.get("assets") or {}),
    }


def _project_summary_cache_key(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _get_project_summary_cache(
    path: str,
    *,
    mtime_ns: int,
    size: int,
) -> dict[str, Any] | None:
    key = _project_summary_cache_key(path)
    with _project_summary_cache_lock:
        cached = _project_summary_cache.get(key)
        if cached is None:
            return None
        cached_mtime_ns, cached_size, summary = cached
        if (cached_mtime_ns, cached_size) != (mtime_ns, size):
            del _project_summary_cache[key]
            return None
        _project_summary_cache.move_to_end(key)
        return dict(summary)


def _store_project_summary_cache(
    path: str,
    stat_result: os.stat_result,
    summary: Mapping[str, Any],
) -> None:
    key = _project_summary_cache_key(path)
    with _project_summary_cache_lock:
        _project_summary_cache[key] = (
            stat_result.st_mtime_ns,
            stat_result.st_size,
            dict(summary),
        )
        _project_summary_cache.move_to_end(key)
        while len(_project_summary_cache) > EDITOR_PROJECT_SUMMARY_CACHE_MAX_ENTRIES:
            _project_summary_cache.popitem(last=False)


def _invalidate_project_summary_cache(path: str) -> None:
    key = _project_summary_cache_key(path)
    with _project_summary_cache_lock:
        _project_summary_cache.pop(key, None)


def resolve_editor_asset(
    asset: Mapping[str, Any],
    *,
    save_root: str,
    workspace: str,
    uploads_root: str,
) -> str:
    name = os.path.basename(str(asset.get("name") or ""))
    path_hint = str(asset.get("path") or "")
    allowed_roots = [os.path.realpath(save_root), os.path.realpath(uploads_root)]
    candidates: list[str] = []
    if path_hint:
        candidates.append(path_hint)
    origin = str(asset.get("origin") or "output").lower()
    if origin == "upload":
        candidates.append(os.path.join(uploads_root, name))
    else:
        asset_workspace = asset.get("workspace")
        if asset_workspace:
            candidates.append(
                os.path.join(
                    workspace_directory(save_root, str(asset_workspace)),
                    name,
                )
            )
        candidates.append(os.path.join(workspace_directory(save_root, workspace), name))
        candidates.append(os.path.join(save_root, name))
    for candidate in candidates:
        if not os.path.isabs(candidate):
            candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate) and any(_is_under(candidate, root) for root in allowed_roots):
            return os.path.realpath(candidate)
    raise EditorProjectError(f"Editor source media was not found: {name or 'unnamed asset'}")


def probe_media(path: str, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    command = [
        ffprobe,
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise EditorProjectError((result.stderr or "Unable to inspect media").strip()[:500])
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    extension = Path(path).suffix.lower()
    try:
        video_frame_count = int((video or {}).get("nb_frames") or 0)
    except (TypeError, ValueError):
        # ffprobe commonly emits "N/A" for stills and some container formats.
        video_frame_count = 0
    if extension in _AUDIO_EXTENSIONS and video is None:
        media_type = "audio"
    elif extension in _IMAGE_EXTENSIONS or (video and video_frame_count == 1):
        media_type = "image"
    else:
        media_type = "video" if video else "audio"
    duration = _finite_number((payload.get("format") or {}).get("duration"), 0.0)
    if duration <= 0 and video:
        duration = _finite_number(video.get("duration"), 0.0)
    rate = str((video or {}).get("avg_frame_rate") or "0/1")
    try:
        numerator, denominator = rate.split("/", 1)
        fps = float(numerator) / max(1.0, float(denominator))
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "name": os.path.basename(path),
        "type": media_type,
        "duration": max(0.0, duration),
        "width": int((video or {}).get("width") or 0),
        "height": int((video or {}).get("height") or 0),
        "fps": max(0.0, fps),
        "has_audio": audio is not None,
        "audio_channels": int((audio or {}).get("channels") or 0),
        "audio_sample_rate": int((audio or {}).get("sample_rate") or 0),
        "size": os.path.getsize(path),
    }


def inspect_editor_assets(
    project: Mapping[str, Any],
    *,
    save_root: str,
    workspace: str,
    uploads_root: str,
) -> list[dict[str, Any]]:
    """Return source availability without mutating or leaking project state."""
    normalized = normalize_editor_project(project, workspace=workspace)
    results: list[dict[str, Any]] = []
    for asset_id, asset in normalized["assets"].items():
        try:
            path = resolve_editor_asset(
                asset,
                save_root=save_root,
                workspace=workspace,
                uploads_root=uploads_root,
            )
            results.append({"asset_id": asset_id, "available": True, "path": path})
        except (EditorProjectError, FileNotFoundError, OSError) as error:
            results.append({
                "asset_id": asset_id,
                "available": False,
                "error": str(error),
            })
    return results


def editor_media_cache_directory(save_root: str, workspace: str) -> str:
    return os.path.join(
        editor_project_directory(save_root, workspace),
        EDITOR_MEDIA_CACHE_DIR,
    )


def _editor_preview_id(path: str) -> str:
    stat = os.stat(path)
    identity = f"{os.path.realpath(path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8", errors="surrogatepass")).hexdigest()[:24]


def _editor_media_cache_root(save_root: str, workspace: str) -> str:
    requested_root = os.path.abspath(editor_media_cache_directory(save_root, workspace))
    project_root = os.path.realpath(editor_project_directory(save_root, workspace))
    if os.path.islink(requested_root):
        raise EditorProjectError("Invalid Editor preview cache root")
    resolved_root = os.path.realpath(requested_root)
    if os.path.dirname(resolved_root) != project_root:
        raise EditorProjectError("Editor preview cache must stay inside its project cache root")
    return resolved_root


def _editor_media_cache_entry_path(root: str, preview_id: str) -> str:
    if not _PREVIEW_ID_RE.fullmatch(str(preview_id or "")):
        raise EditorProjectError("Invalid Editor preview identifier")
    candidate = os.path.join(root, preview_id)
    if os.path.islink(candidate):
        raise EditorProjectError("Invalid Editor preview cache entry")
    resolved = os.path.realpath(candidate)
    if os.path.dirname(resolved) != root:
        raise EditorProjectError("Editor preview cache entry escaped its cache root")
    return resolved


def _editor_media_cache_file_path(cache_dir: str, filename: str) -> str:
    candidate = os.path.join(cache_dir, filename)
    if os.path.islink(candidate):
        raise EditorProjectError("Invalid Editor preview cache file")
    resolved = os.path.realpath(candidate)
    if os.path.dirname(resolved) != cache_dir:
        raise EditorProjectError("Editor preview cache file escaped its cache entry")
    return candidate


def _safe_preview_cache_entry(root: str, entry_name: str) -> str | None:
    if not _PREVIEW_ID_RE.fullmatch(entry_name):
        return None
    candidate = os.path.join(root, entry_name)
    if os.path.islink(candidate) or not os.path.isdir(candidate):
        return None
    resolved = os.path.realpath(candidate)
    if os.path.dirname(resolved) != root or resolved == root:
        return None
    return resolved


def _preview_cache_entry_size(cache_dir: str) -> int:
    total = 0
    for current, directories, filenames in os.walk(cache_dir, followlinks=False):
        directories[:] = [
            name for name in directories
            if not os.path.islink(os.path.join(current, name))
        ]
        for filename in filenames:
            path = os.path.join(current, filename)
            try:
                if not os.path.islink(path):
                    total += os.lstat(path).st_size
            except OSError:
                continue
    return total


def _prune_editor_media_cache(
    root: str,
    *,
    now: float | None = None,
    max_bytes: int = EDITOR_MEDIA_PREVIEW_CACHE_MAX_BYTES,
    ttl_seconds: float = EDITOR_MEDIA_PREVIEW_CACHE_TTL_SECONDS,
    active_seconds: float = EDITOR_MEDIA_PREVIEW_ACTIVE_SECONDS,
) -> None:
    """Prune only direct, non-symlink preview entries in this exact cache root."""
    root = os.path.realpath(os.path.abspath(root))
    if not os.path.isdir(root):
        return
    now = time.time() if now is None else now
    with _preview_builds_lock:
        active_entries = set(_preview_builds)
        entries: list[tuple[str, str, float, int]] = []
        total_bytes = 0
        try:
            with os.scandir(root) as scan:
                for entry in scan:
                    try:
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                        cache_dir = _safe_preview_cache_entry(root, entry.name)
                        if cache_dir is None:
                            continue
                        stat_result = os.stat(cache_dir, follow_symlinks=False)
                        size = _preview_cache_entry_size(cache_dir)
                    except OSError:
                        continue
                    entries.append((entry.name, cache_dir, stat_result.st_mtime, size))
                    total_bytes += size
        except OSError:
            return

        def remove_entry(entry_name: str, cache_dir: str) -> bool:
            # Re-resolve immediately before deletion. Never pass an unchecked or
            # nested target to rmtree, even when the directory has been replaced.
            resolved = _safe_preview_cache_entry(root, entry_name)
            if resolved is None or resolved != cache_dir or resolved in active_entries:
                return False
            try:
                shutil.rmtree(resolved)
                return True
            except OSError:
                return False

        candidates: list[tuple[str, str, float, int]] = []
        for entry_name, cache_dir, last_used, size in entries:
            if cache_dir in active_entries or now - last_used < active_seconds:
                continue
            if now - last_used > ttl_seconds:
                if remove_entry(entry_name, cache_dir):
                    total_bytes -= size
                continue
            candidates.append((entry_name, cache_dir, last_used, size))

        if total_bytes > max_bytes:
            for entry_name, cache_dir, _last_used, size in sorted(
                candidates, key=lambda value: value[2]
            ):
                if total_bytes <= max_bytes:
                    break
                if remove_entry(entry_name, cache_dir):
                    total_bytes -= size


def _cleanup_editor_media_cache_if_due(root: str) -> None:
    root = os.path.realpath(os.path.abspath(root))
    now = time.time()
    with _preview_cache_cleanup_lock:
        last_cleanup = _preview_cache_cleanup_times.get(root, 0.0)
        if now - last_cleanup < EDITOR_MEDIA_PREVIEW_CLEANUP_INTERVAL_SECONDS:
            return
        _prune_editor_media_cache(root, now=now)
        _preview_cache_cleanup_times[root] = now
        _preview_cache_cleanup_times.move_to_end(root)
        while len(_preview_cache_cleanup_times) > 64:
            _preview_cache_cleanup_times.popitem(last=False)


def resolve_editor_media_cache_file(
    save_root: str,
    workspace: str,
    preview_id: str,
    kind: str,
) -> str:
    if not _PREVIEW_ID_RE.fullmatch(str(preview_id or "")):
        raise EditorProjectError("Invalid Editor preview identifier")
    filename = {
        "thumbnail": "thumbnail.jpg",
        "proxy": "proxy.mp4",
        "proxy-mobile": "proxy-mobile-v3.mp4",
    }.get(str(kind or ""))
    if not filename:
        raise EditorProjectError("Invalid Editor preview type")
    root = _editor_media_cache_root(save_root, workspace)
    cache_dir = _editor_media_cache_entry_path(root, preview_id)
    path = _editor_media_cache_file_path(cache_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(filename)
    # Touch the entry directory so TTL/LRU cleanup treats recently served files
    # as active for a conservative window. Keep cleanup and URL resolution from
    # racing over the same cache entry.
    with _preview_builds_lock:
        if _safe_preview_cache_entry(root, preview_id) != cache_dir:
            raise FileNotFoundError(filename)
        try:
            os.utime(cache_dir, None)
        except OSError:
            pass
    return path


def _run_preview_command(command: list[str], *, timeout: int = 180) -> bool:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            startupinfo=(
                subprocess.STARTUPINFO()
                if os.name == "nt"
                else None
            ),
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _build_editor_waveform(
    path: str,
    *,
    ffmpeg: str,
    duration: float,
    bins: int = 512,
) -> list[float]:
    """Stream mono PCM into coarse peak buckets with bounded memory."""
    duration = max(
        0.0,
        min(
            MAX_EDITOR_PROJECT_DURATION_SECONDS,
            _finite_number(duration, 0.0),
        ),
    )
    if duration <= 0 or bins <= 0:
        return []

    sample_rate = 2000
    expected_samples = max(1, int(math.ceil(duration * sample_rate)))
    target_peak_count = min(max(1, bins * 16), max(1, int(math.ceil(duration * 10))))
    samples_per_peak = max(1, int(math.ceil(expected_samples / target_peak_count)))
    peaks = array("f")
    current_peak = 0.0
    current_count = 0
    sample_count = 0
    output_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
    stop_reader = threading.Event()
    reader_errors: list[BaseException] = []
    process: subprocess.Popen[bytes] | None = None
    reader: threading.Thread | None = None
    sentinel: bytes | None = None

    try:
        process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-t",
                f"{duration:.8f}",
                "-f",
                "f32le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            startupinfo=(
                subprocess.STARTUPINFO()
                if os.name == "nt"
                else None
            ),
        )
        assert process.stdout is not None

        def read_chunks() -> None:
            try:
                while not stop_reader.is_set():
                    chunk = process.stdout.read(64 * 1024)
                    if not chunk:
                        break
                    while not stop_reader.is_set():
                        try:
                            output_queue.put(chunk, timeout=0.2)
                            break
                        except queue.Full:
                            continue
            except BaseException as error:
                reader_errors.append(error)
            finally:
                while not stop_reader.is_set():
                    try:
                        output_queue.put(sentinel, timeout=0.2)
                        break
                    except queue.Full:
                        continue

        reader = threading.Thread(target=read_chunks, daemon=True)
        reader.start()
        deadline = time.monotonic() + 180
        remainder = bytearray()
        reached_eof = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(ffmpeg, 180)
            try:
                chunk = output_queue.get(timeout=min(0.2, remaining))
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    reached_eof = True
                    break
                continue
            if chunk is sentinel:
                reached_eof = True
                break
            remainder.extend(chunk)
            usable_length = len(remainder) - (len(remainder) % 4)
            if usable_length == 0:
                continue
            sample_bytes = remainder[:usable_length]
            del remainder[:usable_length]
            samples = array("f")
            samples.frombytes(sample_bytes)
            for value in samples:
                if sample_count >= expected_samples:
                    break
                if math.isfinite(value):
                    current_peak = max(current_peak, abs(value))
                current_count += 1
                sample_count += 1
                if current_count >= samples_per_peak:
                    peaks.append(current_peak)
                    current_peak = 0.0
                    current_count = 0

        if not reached_eof:
            return []
        process.wait(timeout=max(0.1, deadline - time.monotonic()))
        if process.returncode != 0 or reader_errors:
            return []
        if current_count:
            peaks.append(current_peak)
        if not peaks:
            return []

        block = max(1, math.ceil(len(peaks) / bins))
        downsampled = [
            max(peaks[start:start + block], default=0.0)
            for start in range(0, len(peaks), block)
        ][:bins]
        maximum = max(0.001, *downsampled)
        return [round(max(0.02, min(1.0, value / maximum)), 4) for value in downsampled]
    except (OSError, subprocess.SubprocessError, queue.Empty):
        return []
    finally:
        stop_reader.set()
        if process is not None and process.poll() is None:
            process.kill()
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.SubprocessError:
                pass
            if process.stdout is not None:
                process.stdout.close()
        if reader is not None:
            reader.join(timeout=1)


def build_editor_media_preview(
    asset: Mapping[str, Any],
    *,
    save_root: str,
    workspace: str,
    uploads_root: str,
    include_proxy: bool = False,
    proxy_profile: str = "auto",
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    """Build reusable disk-cached thumbnails, waveform peaks, and a proxy.

    Browser-only analysis forced remote/mobile clients to download and decode
    entire source files whenever the Editor reopened. The cache is keyed by
    path, size, and mtime, so replacing a source invalidates it naturally.
    """
    path = resolve_editor_asset(
        asset,
        save_root=save_root,
        workspace=workspace,
        uploads_root=uploads_root,
    )
    preview_id = _editor_preview_id(path)
    media_type = str(asset.get("type") or "video")
    duration = max(0.0, _finite_number(asset.get("duration"), 0.0))
    profile = "mobile" if str(proxy_profile or "auto").strip().lower() == "mobile" else "auto"
    needs_proxy = _editor_preview_needs_proxy(asset, include_proxy, profile)
    root = _editor_media_cache_root(save_root, workspace)
    cache_dir = _editor_media_cache_entry_path(root, preview_id)

    while True:
        with _preview_builds_lock:
            flight = _preview_builds.get(cache_dir)
            owner = flight is None
            if owner:
                flight = _PreviewBuildFlight(
                    needs_proxy=needs_proxy,
                    proxy_profile=profile,
                )
                _preview_builds[cache_dir] = flight
        assert flight is not None
        if owner:
            try:
                with _preview_build_slots:
                    _cleanup_editor_media_cache_if_due(root)
                    result = _build_editor_media_preview(
                        path,
                        preview_id=preview_id,
                        save_root=save_root,
                        workspace=workspace,
                        cache_root=root,
                        cache_dir=cache_dir,
                        media_type=media_type,
                        duration=duration,
                        has_audio=bool(asset.get("has_audio", media_type == "audio")),
                        needs_proxy=needs_proxy,
                        proxy_profile=profile,
                        ffmpeg=ffmpeg,
                    )
                flight.result = result
                return result
            except BaseException as error:
                flight.error = error
                raise
            finally:
                with _preview_builds_lock:
                    if _preview_builds.get(cache_dir) is flight:
                        del _preview_builds[cache_dir]
                    flight.event.set()

        flight.event.wait()
        if flight.error is not None:
            raise flight.error
        result = flight.result
        if result is None:
            continue
        if _preview_result_satisfies(result, needs_proxy, profile):
            return copy.deepcopy(result)
        # Avoid immediately repeating an expensive failed build for callers
        # that requested the same proxy profile. Different requests run one
        # serialized follow-up flight after the existing build finishes.
        same_attempt = needs_proxy and flight.needs_proxy and flight.proxy_profile == profile
        if same_attempt:
            return copy.deepcopy(result)


def _editor_preview_needs_proxy(
    asset: Mapping[str, Any],
    include_proxy: bool,
    proxy_profile: str,
) -> bool:
    if not include_proxy or str(asset.get("type") or "video") != "video":
        return False
    if proxy_profile == "mobile":
        return True
    width = max(0, int(_finite_number(asset.get("width"), 0)))
    height = max(0, int(_finite_number(asset.get("height"), 0)))
    size = max(0, int(_finite_number(asset.get("size"), 0)))
    return width > 1280 or height > 720 or size > 256 * 1024 * 1024


def _preview_result_satisfies(
    metadata: Mapping[str, Any],
    needs_proxy: bool,
    proxy_profile: str,
) -> bool:
    if not needs_proxy:
        return True
    return bool(
        metadata.get("proxy_mobile")
        if proxy_profile == "mobile"
        else metadata.get("proxy")
    )


def _build_editor_media_preview(
    path: str,
    *,
    preview_id: str,
    save_root: str,
    workspace: str,
    cache_root: str,
    cache_dir: str,
    media_type: str,
    duration: float,
    has_audio: bool,
    needs_proxy: bool,
    proxy_profile: str,
    ffmpeg: str,
) -> dict[str, Any]:
    if _editor_media_cache_root(save_root, workspace) != cache_root:
        raise EditorProjectError("Editor preview cache root changed during build")
    os.makedirs(cache_root, exist_ok=True)
    if _editor_media_cache_root(save_root, workspace) != cache_root:
        raise EditorProjectError("Editor preview cache root changed during build")
    os.makedirs(cache_dir, exist_ok=True)
    if _editor_media_cache_entry_path(cache_root, preview_id) != cache_dir:
        raise EditorProjectError("Editor preview cache entry changed during build")

    metadata_path = _editor_media_cache_file_path(cache_dir, "preview.json")
    metadata: dict[str, Any] = {}
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if (
                isinstance(loaded, dict)
                and loaded.get("version") == EDITOR_MEDIA_PREVIEW_VERSION
                and loaded.get("preview_id") == preview_id
            ):
                metadata = loaded
        except (OSError, ValueError):
            metadata = {}

    thumbnail_path = _editor_media_cache_file_path(cache_dir, "thumbnail.jpg")
    if media_type == "video" and not os.path.isfile(thumbnail_path):
        interval = max(0.12, duration / 8.0) if duration > 0 else 1.0
        filter_chain = (
            f"fps=1/{interval:.8f},"
            "scale=160:90:force_original_aspect_ratio=increase,"
            "crop=160:90,tile=8x1"
        )
        temporary_thumbnail = _editor_media_cache_file_path(
            cache_dir, f"thumbnail.{uuid.uuid4().hex}.part.jpg"
        )
        built_thumbnail = _run_preview_command([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
            "-vf", filter_chain, "-frames:v", "1", "-q:v", "5", temporary_thumbnail,
        ])
        if built_thumbnail and os.path.isfile(temporary_thumbnail):
            os.replace(temporary_thumbnail, thumbnail_path)
        elif os.path.isfile(temporary_thumbnail):
            os.remove(temporary_thumbnail)

    waveform = metadata.get("waveform") if isinstance(metadata.get("waveform"), list) else []
    if has_audio and not waveform:
        waveform = _build_editor_waveform(path, ffmpeg=ffmpeg, duration=duration)

    mobile_proxy = proxy_profile == "mobile"
    standard_proxy_path = _editor_media_cache_file_path(cache_dir, "proxy.mp4")
    mobile_proxy_path = _editor_media_cache_file_path(cache_dir, "proxy-mobile-v3.mp4")
    proxy_path = mobile_proxy_path if mobile_proxy else standard_proxy_path
    if needs_proxy and not os.path.isfile(proxy_path):
        temporary_proxy = _editor_media_cache_file_path(
            cache_dir, f"proxy.{uuid.uuid4().hex}.part.mp4"
        )
        video_filter = (
            "scale=854:480:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            "format=yuv420p"
            if mobile_proxy else
            "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,format=yuv420p"
        )
        built = _run_preview_command([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
            "-map", "0:v:0", "-map", "0:a?", "-vf", video_filter,
            "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "27" if mobile_proxy else "25",
            *(["-profile:v", "main", "-level:v", "3.1"] if mobile_proxy else []),
            "-c:a", "aac", "-b:a", "80k" if mobile_proxy else "96k",
            "-movflags", "+faststart", temporary_proxy,
        ], timeout=900)
        if built and os.path.isfile(temporary_proxy):
            os.replace(temporary_proxy, proxy_path)
        elif os.path.isfile(temporary_proxy):
            os.remove(temporary_proxy)

    metadata = {
        "version": EDITOR_MEDIA_PREVIEW_VERSION,
        "preview_id": preview_id,
        "waveform": waveform,
        "thumbnail": os.path.isfile(thumbnail_path),
        "proxy": os.path.isfile(standard_proxy_path),
        "proxy_mobile": os.path.isfile(mobile_proxy_path),
        "updated_at": time.time(),
    }
    _atomic_write_json(metadata_path, metadata)
    return metadata


@functools.lru_cache(maxsize=4)
def editor_export_capabilities(ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    """Probe FFmpeg's hardware encoders, including a minimal real encode.

    ``ffmpeg -encoders`` only reports compiled-in wrappers. A one-frame null
    encode also checks that the runtime can initialize each advertised codec
    and its device, so auto-selection does not pick an unusable backend.
    """
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        listing = result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        listing = ""
    def runtime_available(name: str) -> bool:
        if name not in listing:
            return False
        try:
            probe = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:r=1",
                    "-frames:v",
                    "1",
                    "-an",
                    "-c:v",
                    name,
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                timeout=6,
                startupinfo=(
                    subprocess.STARTUPINFO()
                    if os.name == "nt"
                    else None
                ),
            )
            return probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    encoders = {"software": True}
    for backend, h264, h265 in (
        ("nvidia", "h264_nvenc", "hevc_nvenc"),
        ("intel", "h264_qsv", "hevc_qsv"),
        ("apple", "h264_videotoolbox", "hevc_videotoolbox"),
    ):
        encoders[backend] = (
            runtime_available(h264) and runtime_available(h265)
        )
    recommended = next(
        (name for name in ("nvidia", "apple", "intel") if encoders[name]),
        "software",
    )
    return {"encoders": encoders, "recommended": recommended}


def _atempo_chain(speed: float) -> list[str]:
    value = max(0.1, min(8.0, float(speed)))
    filters: list[str] = []
    while value > 2.0 + 1e-6:
        filters.append("atempo=2.0")
        value /= 2.0
    while value < 0.5 - 1e-6:
        filters.append("atempo=0.5")
        value /= 0.5
    filters.append(f"atempo={value:.8f}")
    return filters


def _escape_drawtext(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\\")
        .replace("'", r"\'")
        .replace(":", r"\:")
        .replace("%", r"\%")
        .replace("\n", r"\n")
    )


def _drawtext_font_option(font_family: Any = None) -> str:
    """Return a portable explicit font option when a system font is known.

    Windows FFmpeg builds often include fontconfig support without shipping a
    fontconfig configuration file. Letting drawtext auto-discover a font then
    fails even though Windows fonts are present. An explicit font keeps title
    export working while Linux/macOS retain sensible native fallbacks.
    """
    configured = os.environ.get("MAESTRO_EDITOR_FONT", "").strip()
    windows_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    family = _editor_font_family(font_family)
    family_candidates = {
        "Arial": [
            os.path.join(windows_fonts, "arial.ttf"),
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "Arial Black": [
            os.path.join(windows_fonts, "ariblk.ttf"),
            "/System/Library/Fonts/Supplemental/Arial Black.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
        "Georgia": [
            os.path.join(windows_fonts, "georgia.ttf"),
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ],
        "Times New Roman": [
            os.path.join(windows_fonts, "times.ttf"),
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ],
        "Verdana": [
            os.path.join(windows_fonts, "verdana.ttf"),
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "Trebuchet MS": [
            os.path.join(windows_fonts, "trebuc.ttf"),
            "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "Courier New": [
            os.path.join(windows_fonts, "cour.ttf"),
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ],
        "Impact": [
            os.path.join(windows_fonts, "impact.ttf"),
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        ],
    }
    candidates = [configured, *family_candidates[family]]
    font = next((candidate for candidate in candidates if candidate and os.path.isfile(candidate)), "")
    if not font:
        return ""
    escaped = font.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    return f":fontfile='{escaped}'"


def _quality_settings(value: str, codec: str = "h264") -> tuple[str, str]:
    quality = str(value or "high").lower()
    if codec == "h265":
        if quality == "draft":
            return "veryfast", "30"
        if quality == "balanced":
            return "fast", "25"
        return "medium", "20"
    if quality == "draft":
        return "veryfast", "27"
    if quality == "balanced":
        return "fast", "22"
    return "medium", "18"


def resolve_editor_export_encoder(
    export_settings: Mapping[str, Any],
    capabilities: Mapping[str, Any] | None = None,
) -> tuple[str, str, list[str]]:
    """Return the selected backend, FFmpeg codec, and quality arguments."""
    codec = str(export_settings.get("codec") or "h264")
    quality = str(export_settings.get("quality") or "high")
    requested = str(export_settings.get("encoder") or "auto")
    available = (
        capabilities.get("encoders")
        if isinstance(capabilities, Mapping)
        and isinstance(capabilities.get("encoders"), Mapping)
        else {}
    )
    if requested == "auto":
        selected = next(
            (name for name in ("nvidia", "apple", "intel") if available.get(name)),
            "software",
        )
    elif requested in {"nvidia", "apple", "intel"} and available.get(requested):
        selected = requested
    else:
        selected = "software"

    if selected == "software":
        preset, crf = _quality_settings(quality, codec)
        return selected, "libx265" if codec == "h265" else "libx264", [
            "-preset", preset, "-crf", crf,
        ]

    quality_level = {"draft": "30", "balanced": "24", "high": "19"}.get(quality, "19")
    if selected == "nvidia":
        encoder = "hevc_nvenc" if codec == "h265" else "h264_nvenc"
        preset = "p4" if quality == "draft" else "p5"
        return selected, encoder, [
            "-preset", preset, "-rc", "vbr", "-cq", quality_level, "-b:v", "0",
        ]
    if selected == "intel":
        encoder = "hevc_qsv" if codec == "h265" else "h264_qsv"
        preset = "faster" if quality == "draft" else "medium"
        return selected, encoder, [
            "-preset", preset, "-global_quality", quality_level,
        ]

    encoder = "hevc_videotoolbox" if codec == "h265" else "h264_videotoolbox"
    bitrate = {"draft": "4M", "balanced": "8M", "high": "16M"}.get(quality, "16M")
    return "apple", encoder, ["-b:v", bitrate, "-realtime", "true"]


def editor_export_dimensions(project: Mapping[str, Any]) -> tuple[int, int, float]:
    """Return even delivery dimensions and FPS without mutating the edit canvas."""
    normalized = normalize_editor_project(project)
    canvas = normalized["canvas"]
    export = normalized["export"]
    width = int(canvas["width"])
    height = int(canvas["height"])
    resolution = str(export.get("resolution") or "canvas")
    if resolution != "canvas":
        edge = int(resolution.removesuffix("p"))
        if width == height:
            width = height = edge
        elif width > height:
            width = max(2, int(round((edge * width / height) / 2) * 2))
            height = edge
        else:
            height = max(2, int(round((edge * height / width) / 2) * 2))
            width = edge
    frame_rate = export.get("frame_rate")
    fps = float(frame_rate if frame_rate in {24, 30, 60} else canvas["fps"])
    return width, height, fps


def _editor_item_keyframes(item: Mapping[str, Any]) -> Mapping[str, list[Mapping[str, Any]]]:
    keyframes = item.get("keyframes")
    return keyframes if isinstance(keyframes, Mapping) else {}


def _editor_item_animation_expression(
    item: Mapping[str, Any],
    property_name: str,
    fallback: float,
    *,
    time_variable: str,
    clip_start: float | None = None,
) -> str:
    time_expression = (
        time_variable
        if clip_start is None
        else f"({time_variable}-{clip_start:.8f})"
    )
    return editor_keyframe_expression(
        _editor_item_keyframes(item),
        property_name,
        fallback=fallback,
        time_expression=time_expression,
    )


def _editor_alpha_ramp_expression(
    *,
    time_variable: str,
    clip_start: float,
    clip_duration: float,
    fade_in: float,
    fade_out: float,
) -> str:
    local_time = f"({time_variable}-{clip_start:.8f})"
    ramps: list[str] = []
    if fade_in > 1e-9:
        ramps.append(f"min(1,max(0,{local_time}/{fade_in:.8f}))")
    if fade_out > 1e-9:
        ramps.append(
            f"min(1,max(0,({clip_duration:.8f}-{local_time})/{fade_out:.8f}))"
        )
    return "*".join(ramps) if ramps else "1"


def _editor_animated_rotation_bounds(
    width: int,
    height: int,
    *,
    maximum_scale: float,
) -> tuple[int, int]:
    diagonal = math.hypot(width * maximum_scale, height * maximum_scale)
    bound = max(2, int(math.ceil(diagonal / 2.0) * 2))
    return bound, bound


def compile_editor_render(
    project: Mapping[str, Any],
    *,
    save_root: str,
    workspace: str,
    uploads_root: str,
    filter_script_path: str,
    output_path: str,
    ffmpeg: str = "ffmpeg",
    encoder_capabilities: Mapping[str, Any] | None = None,
) -> tuple[list[str], float]:
    """Compile a project into one FFmpeg command and return it with duration."""
    project = normalize_editor_project(project, workspace=workspace)
    duration = editor_project_duration(project)
    if duration <= 0:
        raise EditorProjectError("Add at least one timeline item before exporting")
    canvas = project["canvas"]
    width = int(canvas["width"])
    height = int(canvas["height"])
    fps = float(canvas["fps"])
    export_width, export_height, export_fps = editor_export_dimensions(project)
    background = str(canvas["background"]).lstrip("#")
    assets = project["assets"]

    inputs: list[str] = []
    source_entries: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]] = []
    for track in project["tracks"]:
        # Track mute is part of the edit, not merely a preview preference.
        # Skipping every muted media track here keeps the exported composition
        # identical to the Editor preview.  Previously muted video tracks were
        # hidden in preview but still rendered (and their audio still mixed).
        if track.get("muted"):
            continue
        for item in track.get("items") or []:
            if item.get("disabled") or track.get("type") == "text":
                continue
            asset = assets.get(str(item.get("asset_id") or ""))
            if not asset:
                continue
            path = resolve_editor_asset(
                asset,
                save_root=save_root,
                workspace=workspace,
                uploads_root=uploads_root,
            )
            input_index = len(source_entries)
            if asset.get("type") == "image":
                inputs.extend(["-loop", "1", "-framerate", f"{fps:.8f}", "-i", path])
            else:
                inputs.extend(["-i", path])
            source_entries.append((track, item, asset, input_index))

    filters: list[str] = [
        f"color=c=0x{background}:s={width}x{height}:r={fps:.8f}:d={duration:.8f}[editor_base_0]"
    ]
    base_label = "editor_base_0"
    visual_entries = sorted(
        (entry for entry in source_entries if entry[0].get("type") == "video"),
        key=lambda entry: (int(entry[0].get("z_index") or 0), float(entry[1].get("start") or 0)),
    )
    for visual_index, (_track, item, asset, input_index) in enumerate(visual_entries, start=1):
        start = float(item["start"])
        clip_duration = float(item["duration"])
        source_in = float(item.get("source_in") or 0)
        speed = float(item.get("speed") or 1)
        source_span = clip_duration * speed
        keyframes = _editor_item_keyframes(item)
        transform = item.get("transform") if isinstance(item.get("transform"), Mapping) else {}
        scale = _bounded_number(transform.get("scale"), 1.0, 0.05, 4.0)
        x = _finite_number(transform.get("x"), 0.0)
        y = _finite_number(transform.get("y"), 0.0)
        rotation = _finite_number(transform.get("rotation"), 0.0)
        opacity = _bounded_number(item.get("opacity"), 1.0, 0.0, 1.0)
        animated_scale = bool(keyframes.get("scale"))
        animated_rotation = bool(keyframes.get("rotation"))
        animated_opacity = bool(keyframes.get("opacity"))
        fade_in = min(clip_duration, _bounded_number(item.get("fade_in"), 0.0, 0.0, clip_duration))
        fade_out = min(clip_duration, _bounded_number(item.get("fade_out"), 0.0, 0.0, clip_duration))
        transition_overlap = _finite_number(item.get("transition_overlap"), 0.0)
        render_fade_in = transition_overlap if transition_overlap > 0 else fade_in
        transition_in = str(item.get("transition_in") or "none")
        transition_out = str(item.get("transition_out") or "none")
        fit = str(item.get("fit") or "contain")
        clip_label = f"editor_clip_{visual_index}"
        output_label = f"editor_base_{visual_index}"
        raw_clip_label = (
            f"editor_clip_raw_{visual_index}" if animated_opacity else clip_label
        )
        if asset.get("type") == "image":
            prefix = f"[{input_index}:v]trim=duration={clip_duration:.8f},setpts=PTS-STARTPTS+{start:.8f}/TB"
        else:
            prefix = (
                f"[{input_index}:v]trim=start={source_in:.8f}:duration={source_span:.8f},"
                f"setpts=(PTS-STARTPTS)/{speed:.8f}+{start:.8f}/TB"
            )
        spatial_prefix = prefix
        keying = item.get("keying")
        if isinstance(keying, Mapping) and keying.get("enabled"):
            trimmed_label = f"editor_key_trimmed_{visual_index}"
            keyed_native_label = f"editor_key_native_{visual_index}"
            filters.append(f"{prefix},format=rgba[{trimmed_label}]")
            filters.extend(editor_keying_filters(
                keying,
                input_label=trimmed_label,
                output_label=keyed_native_label,
                suffix=str(visual_index),
            ))
            spatial_prefix = f"[{keyed_native_label}]null"
        resize_scale = 1.0 if animated_scale else scale
        target_w = max(2, int(round(width * resize_scale / 2) * 2))
        target_h = max(2, int(round(height * resize_scale / 2) * 2))
        if fit == "cover":
            resize = (
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )
        else:
            resize = f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease"
        scale_filter = ""
        if animated_scale:
            scale_expression = _editor_item_animation_expression(
                item, "scale", scale, time_variable="t", clip_start=start
            )
            scale_filter = (
                ",scale=w='max(2,trunc(iw*("
                f"{scale_expression})/2)*2)':"
                "h='max(2,trunc(ih*("
                f"{scale_expression})/2)*2)':eval=frame"
            )
        rotation_filter = ""
        if animated_rotation:
            rotation_expression = _editor_item_animation_expression(
                item, "rotation", rotation, time_variable="t", clip_start=start
            )
            max_scale = (
                max(float(point["value"]) for point in keyframes["scale"])
                if animated_scale else scale
            )
            rotation_width, rotation_height = _editor_animated_rotation_bounds(
                width,
                height,
                maximum_scale=max_scale,
            )
            rotation_filter = (
                f",rotate=angle='({rotation_expression})*PI/180':c=none:"
                f"ow={rotation_width}:oh={rotation_height}"
            )
        elif animated_scale and abs(rotation) > 0.001:
            rotation_width, rotation_height = _editor_animated_rotation_bounds(
                width,
                height,
                maximum_scale=max(
                    float(point["value"]) for point in keyframes["scale"]
                ),
            )
            radians = rotation * math.pi / 180.0
            rotation_filter = (
                f",rotate={radians:.10f}:c=none:"
                f"ow={rotation_width}:oh={rotation_height}"
            )
        elif abs(rotation) > 0.001:
            radians = rotation * math.pi / 180.0
            rotation_filter = (
                f",rotate={radians:.10f}:c=none:ow=rotw({radians:.10f}):oh=roth({radians:.10f})"
            )
        color_filters = editor_color_filters(
            item.get("color") if isinstance(item.get("color"), Mapping) else None
        )
        fade_filters: list[str] = []
        if render_fade_in > 1e-6:
            if transition_in == "fade_black":
                fade_filters.append(
                    f"fade=t=in:st={start:.8f}:d={render_fade_in:.8f}:color=black"
                )
            elif not animated_opacity:
                fade_filters.append(
                    f"fade=t=in:st={start:.8f}:d={render_fade_in:.8f}:alpha=1"
                )
        if fade_out > 1e-6:
            if transition_out == "fade_black":
                fade_filters.append(
                    f"fade=t=out:st={max(start, start + clip_duration - fade_out):.8f}:"
                    f"d={fade_out:.8f}:color=black"
                )
            elif not animated_opacity:
                fade_filters.append(
                    f"fade=t=out:st={max(start, start + clip_duration - fade_out):.8f}:"
                    f"d={fade_out:.8f}:alpha=1"
                )
        fade_chain = "".join(f",{value}" for value in fade_filters)
        color_chain = "".join(f",{value}" for value in color_filters)
        static_opacity_filter = (
            "" if animated_opacity else f",colorchannelmixer=aa={opacity:.8f}"
        )
        if animated_opacity:
            spatial_chain = f"{spatial_prefix},fps={fps:.8f},{resize},format=rgba"
            if fit == "contain":
                spatial_chain += (
                    f",pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black@0"
                )
            spatial_chain += f"{scale_filter}{rotation_filter}"
        else:
            spatial_chain = (
                f"{spatial_prefix},fps={fps:.8f},{resize}{scale_filter}{rotation_filter},format=rgba"
            )
        if color_filters:
            pre_color_label = f"editor_pre_color_{visual_index}"
            color_input_label = f"editor_color_input_{visual_index}"
            alpha_input_label = f"editor_color_alpha_input_{visual_index}"
            color_alpha_label = f"editor_color_alpha_{visual_index}"
            colored_label = f"editor_colored_{visual_index}"
            filters.append(
                f"{spatial_chain},split=2"
                f"[{color_input_label}][{alpha_input_label}]"
            )
            filters.append(
                f"[{alpha_input_label}]format=yuva444p,alphaextract[{color_alpha_label}]"
            )
            filters.append(
                f"[{color_input_label}]{color_chain[1:]},format=rgba[{colored_label}]"
            )
            filters.append(
                f"[{colored_label}][{color_alpha_label}]alphamerge[{pre_color_label}]"
            )
            filters.append(
                f"[{pre_color_label}]{fade_chain[1:] if fade_chain else 'null'}"
                f"{static_opacity_filter}"
                f"[{raw_clip_label}]"
            )
        else:
            filters.append(
                f"{spatial_chain}{fade_chain}{static_opacity_filter}"
                f"[{raw_clip_label}]"
            )
        if animated_opacity:
            opacity_expression = _editor_item_animation_expression(
                item, "opacity", opacity, time_variable="T", clip_start=start
            )
            alpha_ramp = _editor_alpha_ramp_expression(
                time_variable="T",
                clip_start=start,
                clip_duration=clip_duration,
                fade_in=(
                    render_fade_in
                    if transition_in != "fade_black" else 0.0
                ),
                fade_out=fade_out if transition_out != "fade_black" else 0.0,
            )
            alpha_expression = f"({opacity_expression})*({alpha_ramp})"
            mask_label = f"editor_opacity_mask_{visual_index}"
            filters.append(
                f"color=c=white:s=2x2:r={fps:.8f}:d={clip_duration:.8f},"
                f"setpts=PTS+{start:.8f}/TB,format=gray,"
                f"geq=lum='255*({alpha_expression})',"
                f"scale={target_w}:{target_h}{scale_filter}{rotation_filter}"
                f"[{mask_label}]"
            )
            filters.append(
                f"[{raw_clip_label}]split=2"
                f"[editor_opacity_color_{visual_index}]"
                f"[editor_opacity_source_alpha_{visual_index}]"
            )
            filters.append(
                f"[editor_opacity_source_alpha_{visual_index}]"
                f"format=yuva444p,alphaextract,format=gray"
                f"[editor_opacity_alpha_{visual_index}]"
            )
            filters.append(
                f"[editor_opacity_alpha_{visual_index}][{mask_label}]"
                f"blend=all_mode=multiply:shortest=1"
                f"[editor_opacity_combined_alpha_{visual_index}]"
            )
            filters.append(
                f"[editor_opacity_color_{visual_index}]"
                f"[editor_opacity_combined_alpha_{visual_index}]"
                f"alphamerge[{clip_label}]"
            )
        x_expression = _editor_item_animation_expression(
            item, "x", x, time_variable="t", clip_start=start
        )
        y_expression = _editor_item_animation_expression(
            item, "y", y, time_variable="t", clip_start=start
        )
        overlay_eval = (
            ":eval=frame"
            if keyframes.get("x") or keyframes.get("y") or animated_scale
            else ""
        )
        filters.append(
            f"[{base_label}][{clip_label}]overlay="
            f"x='(main_w-overlay_w)/2+({x_expression})':"
            f"y='(main_h-overlay_h)/2+({y_expression})'"
            f"{overlay_eval}:eof_action=pass:shortest=0:"
            f"enable='between(t,{start:.8f},{start + clip_duration:.8f})'"
            f"[{output_label}]"
        )
        base_label = output_label

    text_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for track in project["tracks"]:
        if track.get("type") != "text" or track.get("muted"):
            continue
        for item in track.get("items") or []:
            if not item.get("disabled"):
                text_entries.append((track, item))
    text_entries.sort(key=lambda entry: (int(entry[0].get("z_index") or 0), float(entry[1].get("start") or 0)))
    for text_index, (_track, item) in enumerate(text_entries, start=1):
        text = _escape_drawtext(str(item.get("text") or "Text"))
        style = item.get("style") if isinstance(item.get("style"), Mapping) else {}
        font_size = int(_bounded_number(style.get("font_size"), 64, 8, 400))
        color = str(style.get("color") or "#ffffff")
        if not _HEX_COLOR_RE.fullmatch(color):
            color = "#ffffff"
        start = float(item["start"])
        end = start + float(item["duration"])
        x = _finite_number(style.get("x"), 0.0)
        y = _finite_number(style.get("y"), 0.0)
        x_expression_value = _editor_item_animation_expression(
            item, "x", x, time_variable="t", clip_start=start
        )
        y_expression_value = _editor_item_animation_expression(
            item, "y", y, time_variable="t", clip_start=start
        )
        background_color = str(style.get("background_color") or "#000000")
        if not _HEX_COLOR_RE.fullmatch(background_color):
            background_color = "#000000"
        background_opacity = _bounded_number(style.get("background_opacity"), 0.32, 0.0, 1.0)
        text_align = str(style.get("text_align") or "center")
        if text_align == "left":
            x_expression = f"w/2+({x_expression_value})"
        elif text_align == "right":
            x_expression = f"w/2-text_w+({x_expression_value})"
        else:
            x_expression = f"(w-text_w)/2+({x_expression_value})"
        fade_in = min(float(item["duration"]), _bounded_number(item.get("fade_in"), 0.0, 0.0, float(item["duration"])))
        fade_out = min(float(item["duration"]), _bounded_number(item.get("fade_out"), 0.0, 0.0, float(item["duration"])))
        text_opacity = _bounded_number(item.get("opacity"), 1.0, 0.0, 1.0)
        text_scale_expression = _editor_item_animation_expression(
            item,
            "scale",
            _bounded_number(
                (item.get("transform") or {}).get("scale")
                if isinstance(item.get("transform"), Mapping) else None,
                1.0,
                0.05,
                4.0,
            ),
            time_variable="t",
            clip_start=start,
        )
        font_size_expression = f"'{font_size}*({text_scale_expression})'"
        opacity_expression = _editor_item_animation_expression(
            item,
            "opacity",
            text_opacity,
            time_variable="t",
            clip_start=start,
        )
        alpha_parts: list[str] = [opacity_expression]
        if fade_in > 1e-6:
            alpha_parts.append(f"if(lt(t,{start + fade_in:.8f}),(t-{start:.8f})/{fade_in:.8f},1)")
        if fade_out > 1e-6:
            alpha_parts.append(f"if(gt(t,{end - fade_out:.8f}),({end:.8f}-t)/{fade_out:.8f},1)")
        alpha_expression = "*".join(alpha_parts) if alpha_parts else "1"
        output_label = f"editor_text_{text_index}"
        font_option = _drawtext_font_option(style.get("font_family"))
        filters.append(
            f"[{base_label}]drawtext=text='{text}'{font_option}:fontcolor={color}:fontsize={font_size_expression}:"
            f"x='{x_expression}':y='(h-text_h)/2+({y_expression_value})':alpha='{alpha_expression}':"
            f"box={1 if background_opacity > 1e-6 else 0}:"
            f"boxcolor={background_color}@{background_opacity:.8f}:boxborderw=12:"
            f"enable='between(t,{start:.8f},{end:.8f})'"
            f"[{output_label}]"
        )
        base_label = output_label
    frame_rate_filter = (
        f"fps={export_fps:.8f},"
        if abs(export_fps - fps) > 1e-9
        else ""
    )
    if export_width != width or export_height != height:
        filters.append(
            f"[{base_label}]scale={export_width}:{export_height}:flags=lanczos,"
            f"{frame_rate_filter}format=yuv420p[editor_video]"
        )
    else:
        filters.append(
            f"[{base_label}]{frame_rate_filter}format=yuv420p[editor_video]"
        )

    audio_labels: list[str] = []
    include_audio = bool((project.get("export") or {}).get("include_audio", True))
    if include_audio:
        for audio_index, (track, item, asset, input_index) in enumerate(source_entries, start=1):
            if track.get("muted") or item.get("muted"):
                continue
            if asset.get("type") == "image" or not asset.get("has_audio"):
                continue
            start = float(item["start"])
            clip_duration = float(item["duration"])
            source_in = float(item.get("source_in") or 0)
            speed = float(item.get("speed") or 1)
            source_span = clip_duration * speed
            volume = _bounded_number(item.get("volume"), 1.0, 0.0, 4.0)
            track_volume = _bounded_number(track.get("volume"), 1.0, 0.0, 4.0)
            fade_in = min(clip_duration, _bounded_number(item.get("fade_in"), 0.0, 0.0, clip_duration))
            fade_out = min(clip_duration, _bounded_number(item.get("fade_out"), 0.0, 0.0, clip_duration))
            chain = [
                f"atrim=start={source_in:.8f}:duration={source_span:.8f}",
                "asetpts=PTS-STARTPTS",
                *_atempo_chain(speed),
                "aresample=48000",
                "aformat=sample_fmts=fltp:channel_layouts=stereo",
            ]
            if fade_in > 1e-6:
                chain.append(f"afade=t=in:st=0:d={fade_in:.8f}")
            if fade_out > 1e-6:
                chain.append(
                    f"afade=t=out:st={max(0.0, clip_duration - fade_out):.8f}:d={fade_out:.8f}"
                )
            if _editor_item_keyframes(item).get("volume"):
                volume_expression = _editor_item_animation_expression(
                    item,
                    "volume",
                    volume,
                    time_variable="t",
                )
                chain.append(
                    f"volume='{volume_expression}*{track_volume:.8f}':eval=frame"
                )
            else:
                chain.append(f"volume={volume * track_volume:.8f}")
            delay_ms = max(0, int(round(start * 1000)))
            if delay_ms:
                chain.append(f"adelay={delay_ms}:all=1")
            label = f"editor_audio_{audio_index}"
            filters.append(f"[{input_index}:a]{','.join(chain)}[{label}]")
            audio_labels.append(label)
    if audio_labels:
        joined = "".join(f"[{label}]" for label in audio_labels)
        filters.append(
            f"{joined}amix=inputs={len(audio_labels)}:duration=longest:normalize=0,"
            f"alimiter=limit=0.98,atrim=duration={duration:.8f}[editor_audio]"
        )
    elif include_audio:
        filters.append(f"anullsrc=r=48000:cl=stereo:d={duration:.8f}[editor_audio]")

    os.makedirs(os.path.dirname(filter_script_path), exist_ok=True)
    with open(filter_script_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(";\n".join(filters))

    export_settings = project.get("export") or {}
    codec = str(export_settings.get("codec") or "h264")
    _encoder_name, video_codec, encoder_args = resolve_editor_export_encoder(
        export_settings,
        encoder_capabilities,
    )
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        *inputs,
        # Keep the older spelling for the CUDA 12.8 / legacy Pinokio runtime.
        # New FFmpeg accepts it (with a deprecation warning) and older builds
        # do not understand the newer `-/filter_complex` file-option syntax.
        "-filter_complex_script", filter_script_path,
        "-map", "[editor_video]",
        *(["-map", "[editor_audio]"] if include_audio else []),
        "-r", f"{export_fps:.8f}",
        "-t", f"{duration:.8f}",
        "-c:v", video_codec,
        *encoder_args,
        "-pix_fmt", "yuv420p",
        *(["-tag:v", "hvc1"] if codec == "h265" else []),
        *(["-c:a", "aac", "-b:a", "192k"] if include_audio else ["-an"]),
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        "-nostats",
        output_path,
    ]
    return command, duration


def _safe_output_stem(name: str) -> str:
    requested = os.path.splitext(os.path.basename(str(name or "Editor export")))[0]
    stem = re.sub(r"[^A-Za-z0-9 _-]+", "", requested).strip()
    stem = re.sub(r"\s+", " ", stem)[:64]
    return stem or "Editor export"


def _is_hardware_encoder_initialization_failure(stderr: str) -> bool:
    """Identify device/encoder startup failures safe to retry in software."""
    message = str(stderr or "").lower()
    markers = (
        "no nvenc capable devices found",
        "cannot load libcuda",
        "cannot initialize cuda",
        "cannot init cuda",
        "error initializing an mfx session",
        "error initializing an internal mfx session",
        "no device available",
        "failed to create device",
        "failed to initialize hardware device",
        "device setup failed",
        "cannot create compression session",
        "failed to create compression session",
    )
    return any(marker in message for marker in markers)


def render_editor_project(
    project: Mapping[str, Any],
    *,
    save_root: str,
    workspace: str,
    uploads_root: str,
    ffmpeg: str = "ffmpeg",
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> str:
    project = normalize_editor_project(project, workspace=workspace)
    output_dir = workspace_directory(save_root, workspace)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    custom_name = str((project.get("export") or {}).get("filename") or "").strip()
    stem = (
        f"{timestamp}_{_safe_output_stem(custom_name)}"
        if custom_name
        else f"{timestamp}_{_safe_output_stem(project['name'])}_edit"
    )
    output_path = os.path.join(output_dir, f"{stem}.mp4")
    suffix = 2
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{stem}({suffix}).mp4")
        suffix += 1
    temporary_dir = tempfile.mkdtemp(prefix="maestro-editor-render-", dir=output_dir)
    # Render the unpublished part file directly inside the workspace. On
    # Windows, a file moved out of a Python-created private temp directory can
    # retain that directory's restrictive ACL and become unreadable to the
    # gallery/ffprobe process. A hidden sibling part inherits the workspace ACL
    # and is still atomically renamed only after FFmpeg finishes successfully.
    temporary_output = os.path.join(output_dir, f".{stem}.{uuid.uuid4().hex[:8]}.part.mp4")
    filter_script = os.path.join(temporary_dir, "timeline_filter.txt")
    process: subprocess.Popen[str] | None = None
    try:
        capabilities = editor_export_capabilities(ffmpeg)
        command, duration = compile_editor_render(
            project,
            save_root=save_root,
            workspace=workspace,
            uploads_root=uploads_root,
            filter_script_path=filter_script,
            output_path=temporary_output,
            ffmpeg=ffmpeg,
            encoder_capabilities=capabilities,
        )
        if progress_callback:
            progress_callback(1, "Preparing timeline")
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        def _execute(render_command: list[str], phase: str) -> str:
            nonlocal process
            process = subprocess.Popen(
                render_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                startupinfo=startupinfo,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            progress_lines: queue.Queue[str | None] = queue.Queue()
            reader_stop = threading.Event()
            stderr_tail = bytearray()
            stderr_tail_lock = threading.Lock()

            def drain_progress() -> None:
                try:
                    for line in process.stdout:
                        if reader_stop.is_set():
                            break
                        progress_lines.put(line)
                finally:
                    progress_lines.put(None)

            def drain_stderr() -> None:
                while not reader_stop.is_set():
                    chunk = process.stderr.read(8192)
                    if not chunk:
                        return
                    encoded = chunk.encode("utf-8", errors="replace")
                    with stderr_tail_lock:
                        stderr_tail.extend(encoded)
                        if len(stderr_tail) > 64 * 1024:
                            del stderr_tail[:-64 * 1024]

            progress_reader = threading.Thread(target=drain_progress, daemon=True)
            stderr_reader = threading.Thread(target=drain_stderr, daemon=True)
            progress_reader.start()
            stderr_reader.start()
            saw_progress_eof = False
            while True:
                if cancel_check and cancel_check():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    reader_stop.set()
                    progress_reader.join(timeout=1)
                    stderr_reader.join(timeout=1)
                    raise EditorRenderCancelled("Editor export cancelled")
                try:
                    line = progress_lines.get(timeout=0.1)
                except queue.Empty:
                    line = ""
                if line is None:
                    saw_progress_eof = True
                    line = ""
                if line:
                    key, _, value = line.strip().partition("=")
                    if key in {"out_time_us", "out_time_ms"}:
                        try:
                            microseconds = float(value)
                            percent = int(min(98, max(1, (microseconds / 1_000_000.0) / duration * 100)))
                            if progress_callback:
                                progress_callback(percent, phase)
                        except (TypeError, ValueError, ZeroDivisionError):
                            pass
                if process.poll() is not None and saw_progress_eof and progress_lines.empty():
                    break
            process.wait()
            reader_stop.set()
            progress_reader.join(timeout=1)
            stderr_reader.join(timeout=1)
            with stderr_tail_lock:
                return stderr_tail.decode("utf-8", errors="replace")

        stderr = _execute(command, "Rendering timeline")
        selected_encoder, _codec, _args = resolve_editor_export_encoder(
            project.get("export") or {}, capabilities
        )
        failed = process.returncode != 0 or not os.path.isfile(temporary_output)
        if (
            failed
            and selected_encoder != "software"
            and _is_hardware_encoder_initialization_failure(stderr)
            and str(
            (project.get("export") or {}).get("encoder") or "auto"
            ) == "auto"
        ):
            if os.path.isfile(temporary_output):
                os.remove(temporary_output)
            if progress_callback:
                progress_callback(1, "Hardware export unavailable; retrying in software")
            software_capabilities = {
                "encoders": {
                    "software": True,
                    "nvidia": False,
                    "intel": False,
                    "apple": False,
                },
                "recommended": "software",
            }
            command, duration = compile_editor_render(
                project,
                save_root=save_root,
                workspace=workspace,
                uploads_root=uploads_root,
                filter_script_path=filter_script,
                output_path=temporary_output,
                ffmpeg=ffmpeg,
                encoder_capabilities=software_capabilities,
            )
            stderr = _execute(command, "Rendering timeline in software")
            failed = process.returncode != 0 or not os.path.isfile(temporary_output)
        if failed:
            raise RuntimeError((stderr or "FFmpeg could not render this Editor project")[-2000:])
        os.replace(temporary_output, output_path)
        sidecar = {
            "generation_mode": "video",
            "tool": "editor",
            "tool_source": project["name"],
            "params": {
                "model_type": "editor",
                "edit_sub_mode": "timeline",
                "editor_project_id": project["id"],
                "editor_workspace": workspace,
            },
            "created_at": time.time(),
        }
        _atomic_write_json(os.path.splitext(output_path)[0] + ".meta.json", sidecar)
        if progress_callback:
            progress_callback(100, "Export complete")
        return output_path
    finally:
        if process is not None and process.poll() is None:
            process.kill()
        if os.path.isfile(temporary_output):
            try:
                os.remove(temporary_output)
            except OSError:
                pass
        shutil.rmtree(temporary_dir, ignore_errors=True)
