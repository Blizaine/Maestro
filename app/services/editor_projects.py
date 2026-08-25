"""Project persistence and FFmpeg rendering for Maestro Editor.

The Editor is deliberately non-destructive: project JSON contains references
to source media plus timeline instructions, while export always writes a new
file.  This module has no FastAPI or WanGP imports so its persistence, path
validation, and render-graph compiler stay inexpensive to unit test.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


EDITOR_SCHEMA_VERSION = 1
EDITOR_PROJECT_DIR = ".maestro_editor"
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_MEDIA_EXTENSIONS = {
    ".aac", ".flac", ".gif", ".jpeg", ".jpg", ".m4a", ".mkv",
    ".mov", ".mp3", ".mp4", ".ogg", ".png", ".wav", ".webm",
    ".webp",
}
_VIDEO_EXTENSIONS = {".gif", ".mkv", ".mov", ".mp4", ".webm"}
_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}
_project_lock = threading.RLock()


class EditorProjectError(ValueError):
    """A project or asset was invalid or unsafe."""


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
            "include_audio": True,
        },
    }


def normalize_editor_project(project: Mapping[str, Any], *, workspace: str | None = None) -> dict[str, Any]:
    if not isinstance(project, Mapping):
        raise EditorProjectError("Editor project must be an object")
    normalized = copy.deepcopy(dict(project))
    normalized["schema_version"] = EDITOR_SCHEMA_VERSION
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
            assets[asset_id] = {
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
                    item["duration"] = _bounded_number(item.get("duration"), 1.0, 1 / 240, 86400.0)
                    item["source_in"] = max(0.0, _finite_number(item.get("source_in"), 0.0))
                    item["speed"] = _bounded_number(item.get("speed"), 1.0, 0.1, 8.0)
                    item["volume"] = _bounded_number(item.get("volume"), 1.0, 0.0, 4.0)
                    item["opacity"] = _bounded_number(item.get("opacity"), 1.0, 0.0, 1.0)
                    item["disabled"] = bool(item.get("disabled", False))
                    if track_type != "text" and str(item.get("asset_id") or "") not in assets:
                        continue
                    items.append(item)
            tracks.append({
                **dict(raw_track),
                "id": track_id,
                "name": str(raw_track.get("name") or track_type.title())[:80],
                "type": track_type,
                "z_index": int(_finite_number(raw_track.get("z_index"), 0)),
                "muted": bool(raw_track.get("muted", False)),
                "locked": bool(raw_track.get("locked", False)),
                "items": items,
            })
    if not tracks:
        tracks = create_editor_project(workspace=normalized["workspace"])["tracks"]
    normalized["tracks"] = tracks
    export = normalized.get("export") if isinstance(normalized.get("export"), Mapping) else {}
    normalized["export"] = {
        **dict(export),
        "quality": str(export.get("quality") or "high"),
        "codec": str(export.get("codec") or "h264"),
        "include_audio": bool(export.get("include_audio", True)),
    }
    return normalized


def save_editor_project(save_root: str, workspace: str, project: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_editor_project(project, workspace=workspace)
    normalized["updated_at"] = time.time()
    path = _project_path(save_root, normalized["workspace"], normalized["id"])
    with _project_lock:
        _atomic_write_json(path, normalized)
    return normalized


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
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            try:
                with open(entry.path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                project = normalize_editor_project(payload, workspace=workspace)
                summaries.append({
                    "id": project["id"],
                    "name": project["name"],
                    "workspace": project["workspace"],
                    "created_at": project["created_at"],
                    "updated_at": project["updated_at"],
                    "duration": editor_project_duration(project),
                    "asset_count": len(project.get("assets") or {}),
                })
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    summaries.sort(key=lambda item: item["updated_at"], reverse=True)
    return summaries


def delete_editor_project(save_root: str, workspace: str, project_id: str) -> bool:
    path = _project_path(save_root, workspace, project_id)
    with _project_lock:
        if not os.path.isfile(path):
            return False
        os.remove(path)
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


def _drawtext_font_option() -> str:
    """Return a portable explicit font option when a system font is known.

    Windows FFmpeg builds often include fontconfig support without shipping a
    fontconfig configuration file. Letting drawtext auto-discover a font then
    fails even though Windows fonts are present. An explicit font keeps title
    export working while Linux/macOS retain sensible native fallbacks.
    """
    configured = os.environ.get("MAESTRO_EDITOR_FONT", "").strip()
    candidates = [
        configured,
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    font = next((candidate for candidate in candidates if candidate and os.path.isfile(candidate)), "")
    if not font:
        return ""
    escaped = font.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    return f":fontfile='{escaped}'"


def _quality_settings(value: str) -> tuple[str, str]:
    quality = str(value or "high").lower()
    if quality == "draft":
        return "veryfast", "27"
    if quality == "balanced":
        return "fast", "22"
    return "medium", "18"


def compile_editor_render(
    project: Mapping[str, Any],
    *,
    save_root: str,
    workspace: str,
    uploads_root: str,
    filter_script_path: str,
    output_path: str,
    ffmpeg: str = "ffmpeg",
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
        transform = item.get("transform") if isinstance(item.get("transform"), Mapping) else {}
        scale = _bounded_number(transform.get("scale"), 1.0, 0.05, 4.0)
        x = _finite_number(transform.get("x"), 0.0)
        y = _finite_number(transform.get("y"), 0.0)
        rotation = _finite_number(transform.get("rotation"), 0.0)
        opacity = _bounded_number(item.get("opacity"), 1.0, 0.0, 1.0)
        fit = str(item.get("fit") or "contain")
        clip_label = f"editor_clip_{visual_index}"
        output_label = f"editor_base_{visual_index}"
        if asset.get("type") == "image":
            prefix = f"[{input_index}:v]trim=duration={clip_duration:.8f},setpts=PTS-STARTPTS+{start:.8f}/TB"
        else:
            prefix = (
                f"[{input_index}:v]trim=start={source_in:.8f}:duration={source_span:.8f},"
                f"setpts=(PTS-STARTPTS)/{speed:.8f}+{start:.8f}/TB"
            )
        target_w = max(2, int(round(width * scale / 2) * 2))
        target_h = max(2, int(round(height * scale / 2) * 2))
        if fit == "cover":
            resize = (
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )
        else:
            resize = f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease"
        rotation_filter = ""
        if abs(rotation) > 0.001:
            radians = rotation * math.pi / 180.0
            rotation_filter = (
                f",rotate={radians:.10f}:c=none:ow=rotw({radians:.10f}):oh=roth({radians:.10f})"
            )
        filters.append(
            f"{prefix},fps={fps:.8f},{resize}{rotation_filter},format=rgba,"
            f"colorchannelmixer=aa={opacity:.8f}[{clip_label}]"
        )
        x_expression = f"(main_w-overlay_w)/2+({x:.8f})"
        y_expression = f"(main_h-overlay_h)/2+({y:.8f})"
        filters.append(
            f"[{base_label}][{clip_label}]overlay=x='{x_expression}':y='{y_expression}':"
            f"eof_action=pass:shortest=0:enable='between(t,{start:.8f},{start + clip_duration:.8f})'"
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
        output_label = f"editor_text_{text_index}"
        font_option = _drawtext_font_option()
        filters.append(
            f"[{base_label}]drawtext=text='{text}'{font_option}:fontcolor={color}:fontsize={font_size}:"
            f"x='(w-text_w)/2+({x:.8f})':y='(h-text_h)/2+({y:.8f})':"
            f"box=1:boxcolor=black@0.32:boxborderw=12:enable='between(t,{start:.8f},{end:.8f})'"
            f"[{output_label}]"
        )
        base_label = output_label
    filters.append(f"[{base_label}]format=yuv420p[editor_video]")

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
            chain = [
                f"atrim=start={source_in:.8f}:duration={source_span:.8f}",
                "asetpts=PTS-STARTPTS",
                *_atempo_chain(speed),
                "aresample=48000",
                "aformat=sample_fmts=fltp:channel_layouts=stereo",
                f"volume={volume * track_volume:.8f}",
            ]
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
    else:
        filters.append(f"anullsrc=r=48000:cl=stereo:d={duration:.8f}[editor_audio]")

    os.makedirs(os.path.dirname(filter_script_path), exist_ok=True)
    with open(filter_script_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(";\n".join(filters))

    preset, crf = _quality_settings((project.get("export") or {}).get("quality"))
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
        "-map", "[editor_audio]",
        "-r", f"{fps:.8f}",
        "-t", f"{duration:.8f}",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", crf,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        "-nostats",
        output_path,
    ]
    return command, duration


def _safe_output_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9 _-]+", "", str(name or "Editor export")).strip()
    stem = re.sub(r"\s+", " ", stem)[:64]
    return stem or "Editor export"


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
    stem = f"{timestamp}_{_safe_output_stem(project['name'])}_edit"
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
        command, duration = compile_editor_render(
            project,
            save_root=save_root,
            workspace=workspace,
            uploads_root=uploads_root,
            filter_script_path=filter_script,
            output_path=temporary_output,
            ffmpeg=ffmpeg,
        )
        if progress_callback:
            progress_callback(1, "Preparing timeline")
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            startupinfo=startupinfo,
        )
        assert process.stdout is not None
        while True:
            if cancel_check and cancel_check():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise EditorRenderCancelled("Editor export cancelled")
            line = process.stdout.readline()
            if line:
                key, _, value = line.strip().partition("=")
                if key in {"out_time_us", "out_time_ms"}:
                    try:
                        microseconds = float(value)
                        percent = int(min(98, max(1, (microseconds / 1_000_000.0) / duration * 100)))
                        if progress_callback:
                            progress_callback(percent, "Rendering timeline")
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
            if process.poll() is not None:
                break
            if not line:
                time.sleep(0.05)
        stderr = process.stderr.read() if process.stderr is not None else ""
        if process.returncode != 0 or not os.path.isfile(temporary_output):
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
