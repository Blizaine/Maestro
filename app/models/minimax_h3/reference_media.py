"""Prepare Ref2VA media to the official per-reference duration envelope."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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


def _probe_duration(path: str) -> float:
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
    if duration < MINIMAX_H3_REFERENCE_MIN_SECONDS:
        raise ValueError(
            f"{os.path.basename(path)} is {duration:.2f}s; MiniMax H3 references must be at least 2 seconds."
        )
    return duration


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
        item["source_duration_seconds"] = round(source_duration, 3)
        item["effective_duration_seconds"] = target_duration
        include_audio = bool(item.get("include_audio", True))
        if source_duration > target_duration + 0.01:
            item["path"] = _trim_media(item["path"], target_duration, "video", include_audio=include_audio)
            print(
                "[MiniMax H3 Ref2VA] Trimmed reference video "
                f"{os.path.basename(str(item.get('filename') or item['path']))}: "
                f"{source_duration:.2f}s -> {target_duration:.2f}s (cached derivative)."
            )
        if item.get("audio_path"):
            attached_duration = _probe_duration(item["audio_path"])
            attached_target = min(target_duration, attached_duration, MINIMAX_H3_REFERENCE_MAX_SECONDS)
            if attached_duration > attached_target + 0.01:
                item["audio_path"] = _trim_media(item["audio_path"], attached_target, "audio")

    for index, source_duration, target_duration in zip(audio_indices, audio_durations, audio_targets):
        item = items[index]
        item["source_duration_seconds"] = round(source_duration, 3)
        item["effective_duration_seconds"] = target_duration
        if source_duration > target_duration + 0.01:
            item["path"] = _trim_media(item["path"], target_duration, "audio")
            print(
                "[MiniMax H3 Ref2VA] Trimmed reference audio "
                f"{os.path.basename(str(item.get('filename') or item['path']))}: "
                f"{source_duration:.2f}s -> {target_duration:.2f}s (cached derivative)."
            )
    return items
