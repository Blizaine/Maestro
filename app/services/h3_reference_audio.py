"""Canonical source-audio roles for MiniMax H3 video references.

The original UI manifest remains the saved source of truth. This module
creates request-local standalone audio references for roles that condition
H3, while output-only soundtracks stay outside the conditioning manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal


H3VideoAudioRole = Literal[
    "ignore", "legacy", "drive", "voice", "style", "sound", "output_only",
]

VIDEO_AUDIO_ROLE_TO_INTENT: dict[str, str] = {
    "drive": "drive",
    "voice": "voice",
    "style": "style",
    "sound": "sound",
}

VIDEO_AUDIO_ROLES = {"ignore", "legacy", "output_only", *VIDEO_AUDIO_ROLE_TO_INTENT}
_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
_DERIVED_AUDIO_MARKER = "_h3_video_audio_source_id"
_VIDEO_AUDIO_MARKER = "_h3_video_audio_source_id"


def normalize_video_audio_role(reference: dict[str, Any]) -> H3VideoAudioRole:
    """Read the selected video audio role without changing the manifest.

    Older saved references have no role field.  Their prior checkbox behavior
    remains authoritative: an enabled soundtrack is ``legacy`` and a disabled
    soundtrack is ``ignore``.
    """

    raw = reference.get("video_audio_role")
    if raw is None or raw == "":
        return "ignore" if reference.get("include_audio", True) is False else "legacy"
    role = str(raw).strip().lower()
    if role not in VIDEO_AUDIO_ROLES:
        choices = ", ".join(sorted(VIDEO_AUDIO_ROLES))
        raise ValueError(
            f"Video reference has invalid audio role {raw!r}; expected one of: {choices}."
        )
    return role  # type: ignore[return-value]


def video_drive_prompt_context(
    reference: dict[str, Any],
    *,
    video_index: int,
    role: str,
) -> list[str]:
    """Build pure exact-drive prose and its optional measured duration clock."""

    lines = [
        f"The exact target soundtrack supplies the performance and timing from "
        f"<Video {int(video_index)}> for {role}; preserve its waveform and audible "
        "timeline exactly and synchronize visible action to it."
    ]
    duration = _positive_duration(reference.get("audio_duration_seconds"))
    if duration is not None:
        lines.append(f"H3_PERFORMANCE_AUDIO_CLOCK duration_seconds={repr(duration)}")
    return lines


def video_audio_contributes_reference_ordinal(reference: dict[str, Any]) -> bool:
    """Whether this video currently owns an Audio ordinal in the mixed manifest."""

    role = normalize_video_audio_role(reference)
    if role == "ignore":
        return False
    if role == "legacy":
        return bool(
            (reference.get("has_audio") or reference.get("audio_path"))
            and reference.get("include_audio", True)
        )
    # Exact drives are target conditioning, never an ``<Audio N>``. For other
    # typed roles, canonicalization moves the numbered slot to a standalone
    # row.  Raw uncanonicalized manifests still need their effective role
    # reflected in read-only prompt/signature numbering.
    if role == "drive":
        return False
    if role == "output_only":
        # This soundtrack is restored only after generation. It has no H3
        # audio-reference slot and must never shift authored Audio labels.
        return False
    # Once canonicalized, this visual row is only the video reference; the
    # derived audio row owns the single Audio ordinal. Keep counting raw typed
    # video rows before conversion so read-only prompt remapping still matches
    # the user's original manifest.
    if reference.get(_VIDEO_AUDIO_MARKER) or reference.get("_h3_video_audio_role_routed"):
        return False
    return True


def _positive_duration(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _probe_audio(path: str) -> tuple[bool, float | None]:
    """Return whether an audio stream exists and its measured duration, if known."""

    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=duration:format=duration", "-of", "json", path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
    except FileNotFoundError as error:
        raise ValueError("FFprobe is required to inspect the selected soundtrack.") from error
    except (json.JSONDecodeError, subprocess.SubprocessError) as error:
        raise ValueError(
            f"Could not inspect soundtrack audio: {os.path.basename(path)}"
        ) from error

    streams = payload.get("streams") or []
    if not streams:
        return False, None
    duration = _positive_duration(streams[0].get("duration"))
    if duration is not None:
        return True, duration
    if Path(path).suffix.lower() in _AUDIO_EXTENSIONS:
        format_info = payload.get("format") or {}
        duration = _positive_duration(format_info.get("duration"))
    return True, duration


def _probe_audio_duration(path: str) -> float | None:
    """Compatibility wrapper for call sites that only need a duration."""

    return _probe_audio(path)[1]


def _extract_full_audio(source: str, cache_dir: str | os.PathLike[str] | None) -> str:
    """Decode one source audio stream into a cached WAV without changing it."""

    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise ValueError(f"Selected soundtrack was not found: {source_path.name}.")
    stat = source_path.stat()
    cache_root = (
        Path(cache_dir).resolve()
        if cache_dir is not None
        else Path.cwd() / "uploads" / "h3_reference_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    signature = json.dumps(
        [str(source_path), stat.st_size, stat.st_mtime_ns, "full-audio-v1"],
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(signature).hexdigest()[:24]
    destination = cache_root / f"{digest}.wav"
    if destination.is_file() and destination.stat().st_size > 0:
        return str(destination)

    # Multiple windows can prepare the same reference concurrently.  Give each
    # FFmpeg invocation its own temporary file, then atomically publish the
    # completed cache entry.
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{digest}.", suffix=".wav.tmp", dir=cache_root, delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source_path), "-map", "0:a:0", "-vn",
        "-c:a", "pcm_s16le", "-f", "wav", str(temporary),
    ]
    try:
        subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=600,
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise ValueError("FFmpeg produced an empty soundtrack WAV.")
        try:
            os.replace(temporary, destination)
        except PermissionError:
            # Windows can deny replacing a cache another concurrent request
            # has just published while its first reader still holds it open.
            if destination.is_file() and destination.stat().st_size > 0:
                return str(destination)
            raise
    except FileNotFoundError as error:
        raise ValueError("FFmpeg is required to extract audio from the selected video.") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or str(error)).strip()[-400:]
        raise ValueError(
            f"Could not extract soundtrack audio from {source_path.name}: {detail}"
        ) from error
    finally:
        if temporary.is_file():
            try:
                temporary.unlink()
            except OSError:
                pass
    return str(destination)


def _source_audio_path(
    reference: dict[str, Any],
    *,
    index: int,
    role: H3VideoAudioRole,
    cache_dir: str | os.PathLike[str] | None,
) -> tuple[str, float]:
    attached_path = str(reference.get("audio_path") or "").strip()
    video_path = str(reference.get("path") or "").strip()
    if attached_path:
        source_path = attached_path
    else:
        if reference.get("refmod_path"):
            raise ValueError(
                f"Reference {index} uses an encoded RefMod video, which does not expose its source audio. "
                "Attach an audio file or use an original video reference."
            )
        if reference.get("has_audio") is False:
            source_name = os.path.basename(video_path) or f"reference {index}"
            raise ValueError(
                f"Reference {index} selected {role} audio, but {source_name} has no audio track. "
                "Attach an audio file or choose Ignore source audio."
            )
        source_path = video_path

    if not source_path:
        raise ValueError(
            f"Reference {index} selected {role} audio, but no audio source is available. "
            "Attach an audio file or choose Ignore source audio."
        )
    if not os.path.isfile(source_path):
        raise ValueError(f"Reference {index} soundtrack was not found: {source_path}.")

    has_audio, duration = _probe_audio(source_path)
    if not has_audio:
        source_name = os.path.basename(source_path) or f"reference {index}"
        raise ValueError(
            f"Reference {index} selected {role} audio, but no readable audio track was found in {source_name}. "
            "Attach an audio file or choose Ignore source audio."
        )

    extension = Path(source_path).suffix.lower()
    if extension in _AUDIO_EXTENSIONS and duration is not None:
        prepared_path = source_path
    else:
        prepared_path = _extract_full_audio(source_path, cache_dir)

    # Use the audio stream's own measured duration.  In particular, an
    # externally attached soundtrack can be longer or shorter than its video.
    prepared_duration = _probe_audio_duration(prepared_path)
    if prepared_duration is None:
        raise ValueError(
            f"Reference {index} soundtrack extraction did not produce a readable audio track."
        )
    return prepared_path, prepared_duration


def canonicalize_video_audio_roles(
    references: list[dict[str, Any]] | None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Return an idempotent request-local manifest with typed video audio rows.

    Legacy paired soundtracks remain attached to their video. Explicit
    conditioning roles become standalone audio rows: ``drive`` is routed
    through the exact target-audio path and ``voice``/``style``/``sound`` use
    their established reference behavior. ``output_only`` stays on its video
    row with audio disabled. Existing derived rows are rebuilt from their
    video source, making repeated calls stable.
    """

    if references is None:
        return []
    if not isinstance(references, list):
        raise ValueError("MiniMax H3 references must be a list.")

    base: list[dict[str, Any]] = []
    existing_derived: set[str] = set()
    for index, raw in enumerate(references, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Reference {index} must be an object.")
        item = dict(raw)
        if item.get("type") == "audio" and item.get(_DERIVED_AUDIO_MARKER):
            existing_derived.add(str(item[_DERIVED_AUDIO_MARKER]))
            continue
        base.append(item)

    result: list[dict[str, Any]] = []
    video_ordinal = 0
    for index, item in enumerate(base, start=1):
        kind = str(item.get("type") or item.get("kind") or "").strip().lower()
        item["type"] = kind
        if kind != "video":
            result.append(item)
            continue

        video_ordinal += 1
        role = normalize_video_audio_role(item)
        raw_role = item.get("video_audio_role")
        if raw_role is not None and str(raw_role).strip():
            # Normalize casing/whitespace in the request-local copy only.
            item["video_audio_role"] = role
        if role == "legacy":
            # Selecting the explicit legacy option restores the old checkbox's
            # enabled behavior.  For old records with no role, retain its exact
            # saved checkbox value.
            if raw_role is not None and str(raw_role).strip():
                item["include_audio"] = True
            else:
                item["include_audio"] = bool(item.get("include_audio", True))
            item.pop(_VIDEO_AUDIO_MARKER, None)
            item.pop("_h3_video_audio_role_routed", None)
            item.pop("_h3_video_audio_routed_role", None)
            result.append(item)
            continue
        if role == "ignore":
            item["include_audio"] = False
            item.pop(_VIDEO_AUDIO_MARKER, None)
            item.pop("_h3_video_audio_role_routed", None)
            item.pop("_h3_video_audio_routed_role", None)
            result.append(item)
            continue

        if role == "output_only":
            # Keep the visual row in the H3 manifest, with embedded audio
            # disabled. The output mixer resolves its source separately from
            # this request-local visual manifest.
            item["include_audio"] = False
            item.pop("audio_duration_seconds", None)
            item.pop(_VIDEO_AUDIO_MARKER, None)
            item.pop("_h3_video_audio_role_routed", None)
            item.pop("_h3_video_audio_routed_role", None)
            result.append(item)
            continue

        source_key = hashlib.sha256(
            "\0".join((str(video_ordinal), str(item.get("id") or ""), str(item.get("path") or ""))).encode("utf-8")
        ).hexdigest()[:16]
        item[_VIDEO_AUDIO_MARKER] = source_key
        item["include_audio"] = False
        result.append(item)

        # If a prior exact-drive split removed its derived drive row, the
        # marker records that routing was already completed for this request.
        if (
            item.get("_h3_video_audio_role_routed")
            and item.get("_h3_video_audio_routed_role") == role
            and source_key not in existing_derived
        ):
            continue
        item.pop("_h3_video_audio_role_routed", None)
        item.pop("_h3_video_audio_routed_role", None)

        source_path, audio_duration = _source_audio_path(
            item, index=index, role=role, cache_dir=cache_dir,
        )
        item["audio_duration_seconds"] = audio_duration
        derived: dict[str, Any] = {
            "id": f"h3-video-audio-{source_key}",
            "type": "audio",
            "path": source_path,
            "filename": str(
                item.get("audio_filename")
                or f"{Path(str(item.get('filename') or item.get('path') or 'video')).stem} audio.wav"
            ),
            "role": str(item.get("role") or f"audio from video {video_ordinal}"),
            "audio_intent": VIDEO_AUDIO_ROLE_TO_INTENT[role],
            "duration_seconds": audio_duration,
            "source_duration_seconds": audio_duration,
            "effective_duration_seconds": audio_duration,
            "has_audio": True,
            _DERIVED_AUDIO_MARKER: source_key,
        }
        for field in ("library_character_id", "character_name"):
            if item.get(field):
                derived[field] = item[field]
        result.append(derived)

    return result


def resolve_output_only_audio_path(
    references,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> str | None:
    """Resolve the single request-only output soundtrack from H3 video refs.

    ``references`` may be one manifest or a list of per-clip manifests. Repeated
    copies of the same selected source across per-clip manifests are one output
    soundtrack; two output-only rows in one manifest or different sources
    across clips are ambiguous and rejected.

    The return value is always a seekable WAV for WGP's timeline slicer. An
    attached audio file takes precedence over the video's first audio stream.
    Nothing is attached to a Ref2VA audio row or copied into saved settings.
    """

    if references is None:
        return None
    if not isinstance(references, list):
        raise ValueError("MiniMax H3 references must be a list.")

    if any(isinstance(item, list) for item in references):
        manifests = references
        if any(not isinstance(manifest, list) for manifest in manifests):
            raise ValueError("Per-clip MiniMax H3 references must be lists.")
    else:
        manifests = [references]

    resolved: dict[str, str] = {}
    for manifest_index, manifest in enumerate(manifests, start=1):
        candidates = []
        for reference_index, item in enumerate(manifest, start=1):
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type") or item.get("kind") or "").strip().lower()
            if kind == "video" and normalize_video_audio_role(item) == "output_only":
                candidates.append((reference_index, item))
        if len(candidates) > 1:
            raise ValueError(
                f"Reference manifest {manifest_index} selects more than one "
                "Keep source audio (output only) soundtrack. Choose one source per generation."
            )
        for reference_index, item in candidates:
            source_path, _duration = _source_audio_path(
                item,
                index=reference_index,
                role="output_only",
                cache_dir=cache_dir,
            )
            if Path(source_path).suffix.lower() != ".wav":
                source_path = _extract_full_audio(source_path, cache_dir)
            resolved[os.path.normcase(os.path.realpath(source_path))] = source_path

    if not resolved:
        return None
    if len(resolved) > 1:
        raise ValueError(
            "Per-clip references select different Keep source audio (output only) "
            "soundtracks. Use one soundtrack for the full generation."
        )
    return next(iter(resolved.values()))

def measure_video_audio_metadata(
    references: list[dict[str, Any]] | None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Copy only measured soundtrack durations back onto original video rows.

    Planning needs the exact drive length before its prompt clock is built, but
    saved Studio/Director manifests must retain their original row structure.
    This helper performs the same on-demand source inspection as
    ``canonicalize_video_audio_roles`` and returns fresh source-shaped rows;
    derived standalone audio references and internal routing markers never
    escape this boundary.
    """

    if references is None:
        return []
    if not isinstance(references, list):
        raise ValueError("MiniMax H3 references must be a list.")

    original_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(references, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Reference {index} must be an object.")
        if raw.get("type") == "audio" and raw.get(_DERIVED_AUDIO_MARKER):
            continue
        row = dict(raw)
        row.pop(_VIDEO_AUDIO_MARKER, None)
        row.pop("_h3_video_audio_role_routed", None)
        row.pop("_h3_video_audio_routed_role", None)
        if (
            str(row.get("type") or row.get("kind") or "").strip().lower() == "video"
            and normalize_video_audio_role(row) == "output_only"
        ):
            row.pop("audio_duration_seconds", None)
        original_rows.append(row)

    expanded = canonicalize_video_audio_roles(
        [dict(item) for item in references], cache_dir=cache_dir,
    )
    measured_durations = [
        item.get("audio_duration_seconds")
        for item in expanded
        if str(item.get("type") or item.get("kind") or "").strip().lower() == "video"
    ]
    result: list[dict[str, Any]] = []
    video_index = 0
    for item in original_rows:
        kind = str(item.get("type") or item.get("kind") or "").strip().lower()
        if kind == "video":
            duration = measured_durations[video_index] if video_index < len(measured_durations) else None
            video_index += 1
            if duration is not None:
                item["audio_duration_seconds"] = duration
        result.append(item)
    return result
