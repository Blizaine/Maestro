"""Create bounded, non-destructive excerpts from gallery audio and video."""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Mapping
from pathlib import PureWindowsPath
from typing import Any

from services.editor_projects import (
    EditorProjectError,
    probe_media,
    workspace_directory,
)
from services.gallery_library import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from services.media_info import record_upload
from services.upload_media import upload_path

FFMPEG_TIMEOUT_SECONDS = 300
FFMPEG_THREADS = 2
_TRIM_SLOT = threading.BoundedSemaphore(1)


class GalleryMediaTrimError(ValueError):
    """A gallery asset or requested excerpt cannot be trimmed."""


class GalleryMediaTrimBusy(RuntimeError):
    """Another CPU-bound gallery trim is already running."""


class GalleryMediaTrimTimeout(TimeoutError):
    """FFmpeg exceeded the bounded trim duration."""


class GalleryMediaTrimUnavailable(RuntimeError):
    """A required media binary or writable destination is unavailable."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GalleryMediaTrimError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise GalleryMediaTrimError(f"{label} must be a finite number.")
    return result


def _safe_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise GalleryMediaTrimError("A valid gallery asset name is required.")
    if (
        value.startswith(".")
        or os.path.basename(value) != value
        or PureWindowsPath(value).name != value
        or PureWindowsPath(value).drive
        or any(character in value for character in ("/", "\\", ":", "\0"))
    ):
        raise GalleryMediaTrimError("Gallery asset names must be plain filenames.")
    return value


def _contained(path: str, root: str) -> bool:
    try:
        candidate = os.path.normcase(os.path.realpath(os.path.abspath(path)))
        boundary = os.path.normcase(os.path.realpath(os.path.abspath(root)))
        return os.path.commonpath([candidate, boundary]) == boundary
    except (OSError, ValueError):
        return False


def _source_path(
    asset: Mapping[str, Any],
    *,
    save_root: str,
    workspace: str,
    uploads_root: str,
) -> str:
    name = _safe_filename(asset.get("name"))
    origin = asset.get("origin")
    if not isinstance(origin, str) or origin not in {"output", "upload"}:
        raise GalleryMediaTrimError("Gallery asset origin must be 'output' or 'upload'.")

    if origin == "upload":
        try:
            root = os.path.realpath(os.path.abspath(uploads_root))
            candidate = upload_path(root, name)
        except ValueError as error:
            raise GalleryMediaTrimError(str(error)) from error
    else:
        selected_workspace = asset.get("workspace") or workspace
        if not isinstance(selected_workspace, str) or not selected_workspace.strip():
            raise GalleryMediaTrimError("Choose the output workspace that contains this asset.")
        selected_workspace = selected_workspace.strip()
        base = os.path.realpath(os.path.abspath(save_root))
        try:
            workspace_root = workspace_directory(base, selected_workspace)
        except EditorProjectError as error:
            raise GalleryMediaTrimError(str(error)) from error
        # A named workspace symlink could silently alias another workspace.
        # Reject it and every resolved root outside the configured output tree.
        if selected_workspace != "default" and os.path.islink(workspace_root):
            raise GalleryMediaTrimError("Symlinked output workspaces cannot be trimmed.")
        root = os.path.realpath(workspace_root)
        if not _contained(root, base):
            raise GalleryMediaTrimError("The selected output workspace is outside the outputs folder.")
        candidate = os.path.join(root, name)

    if os.path.islink(candidate):
        raise GalleryMediaTrimError("Symlinked gallery media cannot be trimmed.")
    resolved = os.path.realpath(candidate)
    if (
        not _contained(resolved, root)
        or os.path.normcase(os.path.dirname(resolved)) != os.path.normcase(root)
    ):
        raise GalleryMediaTrimError("Gallery media must be directly inside its selected folder.")
    if not os.path.isfile(resolved):
        raise FileNotFoundError("Gallery media was not found in the selected origin and workspace.")
    return resolved


def _seconds(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


def _run_ffmpeg(command: list[str], *, timeout: int) -> None:
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as error:
        raise GalleryMediaTrimTimeout(
            f"Excerpt creation exceeded the {timeout}-second limit."
        ) from error
    except OSError as error:
        raise GalleryMediaTrimUnavailable(f"FFmpeg could not be started: {error}") from error
    if result.returncode:
        detail = (result.stderr or "FFmpeg returned an error.").strip().splitlines()
        raise GalleryMediaTrimError(f"Could not create the excerpt: {(detail[-1] if detail else 'FFmpeg failed')[:400]}")


def trim_gallery_media(
    asset: Mapping[str, Any],
    start_time: Any,
    end_time: Any,
    *,
    save_root: str,
    workspace: str,
    uploads_root: str,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    timeout: int = FFMPEG_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Write a unique audio/video excerpt into uploads without changing source."""
    if not isinstance(asset, Mapping):
        raise GalleryMediaTrimError("Asset must be an object.")
    start = _number(start_time, "Start time")
    end = _number(end_time, "End time")
    if start < 0 or end <= start:
        raise GalleryMediaTrimError("Choose a range with start time before end time and start at zero or later.")

    source = _source_path(
        asset,
        save_root=save_root,
        workspace=workspace,
        uploads_root=uploads_root,
    )
    if ffmpeg is None or ffprobe is None:
        from shared.utils.video_decode import _resolve_media_binary

        ffmpeg_bin = ffmpeg or _resolve_media_binary("ffmpeg")
        ffprobe_bin = ffprobe or _resolve_media_binary("ffprobe")
    else:
        ffmpeg_bin, ffprobe_bin = ffmpeg, ffprobe
    if not ffmpeg_bin or not os.path.isfile(ffmpeg_bin):
        raise GalleryMediaTrimUnavailable("FFmpeg is unavailable; cannot create this excerpt.")
    if not ffprobe_bin or not os.path.isfile(ffprobe_bin):
        raise GalleryMediaTrimUnavailable("FFprobe is unavailable; cannot inspect this media.")

    try:
        source_info = probe_media(source, ffprobe=ffprobe_bin)
    except subprocess.TimeoutExpired as error:
        raise GalleryMediaTrimTimeout("Media inspection exceeded its 60-second limit.") from error
    except (EditorProjectError, OSError, ValueError) as error:
        raise GalleryMediaTrimError(f"Could not inspect gallery media: {str(error)[:400]}") from error
    extension = os.path.splitext(source)[1].lower()
    if extension in AUDIO_EXTENSIONS and source_info.get("has_audio"):
        # FFprobe reports embedded album art as a video stream. Gallery audio
        # formats remain audio sources, and the command below maps `a:0` only.
        media_type = "audio"
    elif extension in VIDEO_EXTENSIONS and source_info.get("width") and source_info.get("height"):
        media_type = "video"
    else:
        media_type = source_info.get("type")
    duration = _number(source_info.get("duration"), "Source duration")
    if media_type not in {"audio", "video"} or duration <= 0:
        raise GalleryMediaTrimError("Only audio and video files with a readable duration can be trimmed.")
    if end > duration:
        raise GalleryMediaTrimError(f"End time must be within the source duration ({duration:.6f} seconds).")

    if media_type == "audio":
        sample_rate = int(source_info.get("audio_sample_rate") or 0)
        channels = int(source_info.get("audio_channels") or 0)
        if sample_rate <= 0 or channels <= 0:
            raise GalleryMediaTrimError("The audio source has no readable sample rate or channel count.")
        suffix, mime_type = ".wav", "audio/wav"
    else:
        fps = _number(source_info.get("fps") or 0.0, "Source frame rate")
        if fps <= 0:
            raise GalleryMediaTrimError("The video source has no readable frame rate.")
        suffix, mime_type = ".mp4", "video/mp4"

    uploads = os.path.realpath(os.path.abspath(uploads_root))
    try:
        os.makedirs(uploads, exist_ok=True)
    except OSError as error:
        raise GalleryMediaTrimUnavailable("The uploads folder could not be created.") from error
    if not os.path.isdir(uploads) or not os.access(uploads, os.W_OK):
        raise GalleryMediaTrimUnavailable("The uploads folder is not writable.")
    if not _TRIM_SLOT.acquire(blocking=False):
        raise GalleryMediaTrimBusy("Another gallery excerpt is being created. Try again shortly.")

    temporary_path = None
    try:
        token = uuid.uuid4().hex
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".gallery-trim-{token}-", suffix=suffix, dir=uploads
            )
        except OSError as error:
            raise GalleryMediaTrimUnavailable("A temporary excerpt file could not be allocated.") from error
        os.close(descriptor)
        cut_duration = end - start
        command = [
            ffmpeg_bin,
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-filter_threads", str(FFMPEG_THREADS),
            "-threads", str(FFMPEG_THREADS),
            "-ss", _seconds(start), "-i", source,
            "-t", _seconds(cut_duration),
        ]
        if media_type == "audio":
            command.extend([
                "-map", "0:a:0", "-vn",
                "-af", f"atrim=duration={_seconds(cut_duration)},asetpts=PTS-STARTPTS",
                "-c:a", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels),
                "-threads", str(FFMPEG_THREADS),
            ])
        else:
            command.extend([
                "-map", "0:v:0", "-vf", "setpts=PTS-STARTPTS",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                "-threads", str(FFMPEG_THREADS),
            ])
            if source_info.get("has_audio"):
                command.extend([
                    "-map", "0:a:0", "-af",
                    f"atrim=duration={_seconds(cut_duration)},asetpts=PTS-STARTPTS",
                    "-c:a", "aac", "-b:a", "192k",
                ])
            else:
                command.append("-an")
            command.extend(["-movflags", "+faststart"])
        command.append(temporary_path)
        _run_ffmpeg(command, timeout=timeout)

        try:
            output_info = probe_media(temporary_path, ffprobe=ffprobe_bin)
        except subprocess.TimeoutExpired as error:
            raise GalleryMediaTrimTimeout("Excerpt verification exceeded its 60-second limit.") from error
        except (EditorProjectError, OSError, ValueError) as error:
            raise GalleryMediaTrimError(f"Could not verify the excerpt: {str(error)[:400]}") from error
        output_duration = _number(output_info.get("duration"), "Excerpt duration")
        if output_duration <= 0 or not os.path.getsize(temporary_path):
            raise GalleryMediaTrimError("FFmpeg produced an empty excerpt.")
        if media_type == "audio" and (
            output_info.get("audio_sample_rate") != sample_rate
            or output_info.get("audio_channels") != channels
        ):
            raise GalleryMediaTrimError("The excerpt did not preserve the source audio format.")

        filename = f"gallery-trim-{token}{suffix}"
        destination = os.path.join(uploads, filename)
        if os.path.exists(destination):
            raise GalleryMediaTrimUnavailable("A unique excerpt filename could not be allocated.")
        try:
            os.replace(temporary_path, destination)
        except OSError as error:
            raise GalleryMediaTrimUnavailable("The excerpt could not be saved to uploads.") from error
        temporary_path = None
        record_upload(destination, f"Excerpt of {os.path.basename(source)}")
        return {
            "filename": filename,
            "media_type": media_type,
            "mime_type": mime_type,
            "duration": output_duration,
        }
    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        _TRIM_SLOT.release()
