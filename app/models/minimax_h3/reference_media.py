"""Prepare Ref2VA media to the official per-reference duration envelope."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path

from .reference_manifest import validate_reference_manifest


MINIMAX_H3_REFERENCE_MIN_SECONDS = 2.0
MINIMAX_H3_REFERENCE_MAX_SECONDS = 15.0
MINIMAX_H3_REFERENCE_TOTAL_SECONDS = 15.0


def allocate_reference_durations(
    durations: list[float],
    *,
    total_limit: float = MINIMAX_H3_REFERENCE_TOTAL_SECONDS,
) -> list[float]:
    """Fairly allocate a total duration budget while preserving short clips.

    This is a water-filling allocation.  For example, three ten-second clips
    become three five-second clips, while 2/10/10 becomes 2/6.5/6.5.
    """

    values = [min(float(value), MINIMAX_H3_REFERENCE_MAX_SECONDS) for value in durations]
    if any(value < MINIMAX_H3_REFERENCE_MIN_SECONDS for value in values):
        raise ValueError("MiniMax H3 Omni reference clips must each be at least 2 seconds long.")
    if sum(values) <= total_limit + 1e-6:
        return values

    remaining = set(range(len(values)))
    result = [0.0] * len(values)
    budget = float(total_limit)
    while remaining:
        share = budget / len(remaining)
        short = [index for index in remaining if values[index] <= share + 1e-9]
        if not short:
            for index in remaining:
                result[index] = share
            break
        for index in short:
            result[index] = values[index]
            budget -= values[index]
            remaining.remove(index)
    if any(value < MINIMAX_H3_REFERENCE_MIN_SECONDS - 1e-6 for value in result):
        raise ValueError("The selected references cannot fit MiniMax H3's 15-second total reference budget.")
    return [round(value, 3) for value in result]


def _probe_container_duration(path: str) -> float:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        duration = float(completed.stdout.strip())
    except FileNotFoundError as error:
        raise ValueError("FFprobe is required to inspect MiniMax H3 references.") from error
    except (ValueError, subprocess.SubprocessError) as error:
        raise ValueError(f"Could not read reference duration: {os.path.basename(path)}") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Could not read reference duration: {os.path.basename(path)}")
    return duration


def _probe_duration(path: str) -> float:
    duration = _probe_container_duration(path)
    if duration < MINIMAX_H3_REFERENCE_MIN_SECONDS:
        raise ValueError(
            f"{os.path.basename(path)} is {duration:.2f}s; MiniMax H3 references must be at least 2 seconds."
        )
    return duration


def _probe_stream(path: str, selector: str) -> dict | None:
    """Return the first selected stream's metadata without decoding its media."""

    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", selector,
                "-show_entries", "stream=duration,avg_frame_rate,r_frame_rate,sample_rate",
                "-of", "json", path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        streams = json.loads(completed.stdout).get("streams") or []
    except FileNotFoundError as error:
        raise ValueError("FFprobe is required to inspect MiniMax H3 references.") from error
    except (json.JSONDecodeError, subprocess.SubprocessError) as error:
        raise ValueError(f"Could not inspect reference media: {os.path.basename(path)}") from error
    return streams[0] if streams else None


def _stream_duration(path: str, stream: dict) -> float:
    try:
        duration = float(stream.get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    if math.isfinite(duration) and duration > 0:
        return duration
    return _probe_container_duration(path)


def _window_cache_path(
    source: str,
    start_time: float,
    duration: float,
    kind: str,
    *,
    include_audio: bool = True,
    target_fps: int = 24,
    pad_to_duration: bool = True,
) -> Path:
    source_path = Path(source).resolve()
    stat = source_path.stat()
    signature = json.dumps(
        [
            str(source_path), stat.st_size, stat.st_mtime_ns, round(start_time, 6),
            round(duration, 6), kind, bool(include_audio), int(target_fps), bool(pad_to_duration),
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(signature).hexdigest()[:24]
    root = Path.cwd() / "uploads" / "h3_reference_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{digest}{'.mp4' if kind == 'video-window' else '.wav'}"


def _run_cached_ffmpeg(command: list[str], destination: Path, source: str) -> str:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        subprocess.run(command + [str(temporary)], capture_output=True, text=True, check=True, timeout=600)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise ValueError("FFmpeg produced an empty reference segment.")
        os.replace(temporary, destination)
    except FileNotFoundError as error:
        raise ValueError("FFmpeg is required to prepare MiniMax H3 reference windows.") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or str(error)).strip()[-500:]
        raise ValueError(f"Could not slice {os.path.basename(source)}: {detail}") from error
    finally:
        if temporary.is_file():
            try:
                temporary.unlink()
            except OSError:
                pass
    return str(destination)


def extract_reference_video_window(
    source: str,
    start_time: float,
    duration: float,
    *,
    include_audio: bool = True,
    target_fps: int = 24,
) -> str:
    """Cache one bounded, timeline-aligned video window at the H3 frame rate.

    Seeking is accurate for transcoded output. If the requested start lies
    beyond the video EOF, the last source frame is used as the segment's first
    frame, then held for the requested duration; the beginning is never replayed.
    """

    start_time = max(0.0, float(start_time))
    duration = max(1.0 / max(1, int(target_fps)), float(duration))
    target_fps = max(1, int(target_fps))
    destination = _window_cache_path(
        source, start_time, duration, "video-window",
        include_audio=include_audio, target_fps=target_fps,
    )
    if destination.is_file() and destination.stat().st_size > 0:
        return str(destination)

    stream = _probe_stream(source, "v:0")
    if stream is None:
        raise ValueError(f"No video stream was found in {os.path.basename(source)}.")
    source_duration = _stream_duration(source, stream)
    try:
        source_fps = float(Fraction(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24/1")))
    except (ValueError, ZeroDivisionError):
        source_fps = 24.0
    source_fps = source_fps if source_fps > 0 else 24.0
    # Keep one source frame available to clone when a later target window
    # starts at or beyond EOF.
    seek_time = min(start_time, max(0.0, source_duration - (1.0 / source_fps)))
    frame_count = max(1, int(round(duration * target_fps)))
    frame_duration = frame_count / float(target_fps)
    filter_graph = (
        f"fps={target_fps},tpad=stop_mode=clone:stop_duration={frame_duration:.6f},"
        f"trim=duration={frame_duration:.6f},setpts=PTS-STARTPTS"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{seek_time:.6f}", "-i", source,
        "-map", "0:v:0", "-an", "-vf", filter_graph,
        "-frames:v", str(frame_count), "-vsync", "cfr", "-r", str(target_fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-f", "mp4",
    ]
    return _run_cached_ffmpeg(command, destination, source)


def _write_silence_window(destination: Path, duration: float, source: str) -> str:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo",
        "-t", f"{duration:.6f}", "-ar", "32000", "-ac", "2",
        "-c:a", "pcm_s16le", "-f", "wav",
    ]
    return _run_cached_ffmpeg(command, destination, source)


def extract_reference_audio_window(
    source: str,
    start_time: float,
    duration: float,
    *,
    include_audio: bool = True,
    pad_to_duration: bool = True,
) -> str | None:
    """Cache a bounded soundtrack window, optionally padding its tail with silence."""

    if not include_audio:
        return None
    start_time = max(0.0, float(start_time))
    duration = max(0.001, float(duration))
    destination = _window_cache_path(
        source,
        start_time,
        duration,
        "audio-window",
        include_audio=include_audio,
        pad_to_duration=pad_to_duration,
    )
    if destination.is_file() and destination.stat().st_size > 0:
        return str(destination)

    stream = _probe_stream(source, "a:0")
    if stream is None:
        return None
    source_duration = _stream_duration(source, stream)
    if start_time >= source_duration - 1e-6:
        return _write_silence_window(destination, duration, source)

    filters = []
    if pad_to_duration:
        filters.append(f"apad=pad_dur={duration:.6f}")
    filters.extend((f"atrim=duration={duration:.6f}", "asetpts=PTS-STARTPTS"))
    filter_graph = ",".join(filters)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_time:.6f}", "-i", source,
        "-map", "0:a:0", "-vn", "-af", filter_graph,
        "-t", f"{duration:.6f}", "-ar", "32000", "-ac", "2",
        "-c:a", "pcm_s16le", "-f", "wav",
    ]
    try:
        return _run_cached_ffmpeg(command, destination, source)
    except ValueError as error:
        message = str(error).lower()
        if pad_to_duration and (
            "does not contain any stream" in message or "output file is empty" in message
        ):
            return _write_silence_window(destination, duration, source)
        raise


def _cache_path(source: str, duration: float, kind: str, include_audio: bool = True) -> Path:
    source_path = Path(source).resolve()
    stat = source_path.stat()
    signature = json.dumps(
        [str(source_path), stat.st_size, stat.st_mtime_ns, round(duration, 3), kind, include_audio],
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(signature).hexdigest()[:20]
    root = Path.cwd() / "uploads" / "h3_reference_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{digest}{'.mp4' if kind == 'video' else '.wav'}"


def _trim_media(source: str, duration: float, kind: str, *, include_audio: bool = True) -> str:
    destination = _cache_path(source, duration, kind, include_audio)
    if destination.is_file() and destination.stat().st_size > 0:
        return str(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source,
        "-t", f"{duration:.3f}",
    ]
    if kind == "video":
        command += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
        command += ["-c:a", "aac", "-b:a", "192k"] if include_audio else ["-an"]
        command += ["-movflags", "+faststart", "-f", "mp4"]
    else:
        command += ["-vn", "-acodec", "pcm_s16le", "-f", "wav"]
    command.append(str(temporary))
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
        os.replace(temporary, destination)
    except FileNotFoundError as error:
        raise ValueError("FFmpeg is required to prepare MiniMax H3 references.") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or str(error)).strip()[-500:]
        raise ValueError(f"Could not trim {os.path.basename(source)}: {detail}") from error
    finally:
        if temporary.is_file():
            try:
                temporary.unlink()
            except OSError:
                pass
    return str(destination)


def normalize_reference_manifest(references) -> list[dict]:
    """Return an H3-compliant manifest backed by cached derived media.

    The originals are never modified.  Exact drive audio is routed through
    Maestro's target soundtrack path and is intentionally excluded from the
    Ref2VA reference-audio budget.
    """

    items = validate_reference_manifest(references, require_files=True)

    prepared_items = [
        item for item in items
        if item["type"] == "video"
        or (item["type"] == "audio" and item.get("audio_intent", "voice") != "drive")
    ]
    if prepared_items and all(
        isinstance(item.get("effective_duration_seconds"), (int, float))
        and MINIMAX_H3_REFERENCE_MIN_SECONDS
        <= float(item["effective_duration_seconds"])
        <= MINIMAX_H3_REFERENCE_MAX_SECONDS
        for item in prepared_items
    ):
        video_total = sum(
            float(item["effective_duration_seconds"])
            for item in prepared_items if item["type"] == "video"
        )
        audio_total = sum(
            float(item["effective_duration_seconds"])
            for item in prepared_items if item["type"] == "audio"
        )
        if (
            video_total <= MINIMAX_H3_REFERENCE_TOTAL_SECONDS + 1e-6
            and audio_total <= MINIMAX_H3_REFERENCE_TOTAL_SECONDS + 1e-6
        ):
            return items

    video_indices: list[int] = []
    video_durations: list[float] = []
    audio_indices: list[int] = []
    audio_durations: list[float] = []
    for index, item in enumerate(items):
        if item["type"] == "video":
            video_indices.append(index)
            video_durations.append(_probe_duration(item["path"]))
        elif item["type"] == "audio" and item.get("audio_intent", "voice") != "drive":
            audio_indices.append(index)
            audio_durations.append(_probe_duration(item["path"]))

    video_targets = allocate_reference_durations(video_durations) if video_durations else []
    audio_targets = allocate_reference_durations(audio_durations) if audio_durations else []

    for index, source_duration, target_duration in zip(video_indices, video_durations, video_targets):
        item = items[index]
        source_path = item["path"]
        follows_timeline = bool(item.get("follow_timeline", False))
        item["source_duration_seconds"] = round(source_duration, 3)
        item["effective_duration_seconds"] = target_duration
        include_audio = bool(item.get("include_audio", True))
        if not follows_timeline and source_duration > target_duration + 0.01:
            item["path"] = _trim_media(source_path, target_duration, "video", include_audio=include_audio)
            print(
                "[MiniMax H3 Ref2VA] Trimmed reference video "
                f"{os.path.basename(str(item.get('filename') or source_path))}: "
                f"{source_duration:.2f}s -> {target_duration:.2f}s (cached derivative)."
            )
        if item.get("audio_path"):
            if follows_timeline:
                # Replacement audio shares the video window and stays intact so
                # later passes can seek into its original timeline.
                item["audio_effective_duration_seconds"] = target_duration
            else:
                attached_duration = _probe_duration(item["audio_path"])
                attached_target = min(target_duration, attached_duration, MINIMAX_H3_REFERENCE_MAX_SECONDS)
                if attached_duration > attached_target + 0.01:
                    item["audio_path"] = _trim_media(item["audio_path"], attached_target, "audio")

    for index, source_duration, target_duration in zip(audio_indices, audio_durations, audio_targets):
        item = items[index]
        source_path = item["path"]
        item["source_duration_seconds"] = round(source_duration, 3)
        item["effective_duration_seconds"] = target_duration
        follows_timeline = item.get("audio_intent", "voice") == "style"
        if not follows_timeline and source_duration > target_duration + 0.01:
            item["path"] = _trim_media(source_path, target_duration, "audio")
            print(
                "[MiniMax H3 Ref2VA] Trimmed reference audio "
                f"{os.path.basename(str(item.get('filename') or source_path))}: "
                f"{source_duration:.2f}s -> {target_duration:.2f}s (cached derivative)."
            )
    return items
