"""CPU-only technical take review helpers and a durable polish-budget ledger.

inspect_take checks media structure and software decodability, then extracts
small local preview artifacts. It does not assess creative quality, run ASR/VLM,
fetch models, contact a cloud service, or retry on a GPU. Callers must pass media
and output paths they have already authorized.

PolishBudgetReviewStore is scoped by the caller-selected directory (normally one
owner/project). A unique claim is persisted before the caller starts a renderer
request. Claims are idempotent by request ID and exact payload; they count
against configured limits permanently unless an operator explicitly repairs
the ledger outside this API. This module never submits or cancels work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - platform-specific import
    import msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    msvcrt = None

try:  # pragma: no cover - platform-specific import
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None


_REVIEW_SCHEMA_VERSION = 1
_STORE_SCHEMA_VERSION = 1
_MAX_STDOUT_BYTES = 1024 * 1024
_MAX_STDERR_BYTES = 16 * 1024
_MAX_SUMMARY_CHARS = 320
_MAX_OBSERVATIONS = 12
_MAX_OBSERVATION_CHARS = 500
_MAX_DECODE_SECONDS = 120
_MAX_PROBE_SECONDS = 20
_MAX_STILL_SECONDS = 20
_MAX_AUDIO_PREVIEW_SECONDS = 30
_AUDIO_PREVIEW_SECONDS = 8
_WINDOWS_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_REVIEW_CATEGORIES = frozenset({
    "action", "audio", "composition", "continuity", "lighting", "motion",
    "prompt_adherence", "subject_identity", "visual_quality", "other",
})
_REVIEW_SEVERITIES = frozenset({"low", "medium", "high", "critical"})
_GENERIC_OBSERVATIONS = frozenset({
    "nothing to improve", "nothing needs changing", "no visible issues",
    "no apparent issues",
})
_GENERIC_OBSERVATION_RE = re.compile(
    r"^(?:(?:the|this) (?:take|clip|video|scene)|it|everything) "
    r"(?:looks|is|seems|feels) (?:good|bad|great|fine|nice|okay|ok)"
    r"(?: overall| with no (?:issues|problems))?[.!?]*$",
    re.IGNORECASE,
)


class ReviewValidationError(ValueError):
    """Raised when an agent-authored creative assessment violates the schema."""


class ReviewStoreError(RuntimeError):
    """Raised when the persistent review ledger is invalid or cannot be used."""


class ReviewStoreCorruptError(ReviewStoreError):
    """Raised rather than overwriting an unreadable existing ledger."""


class RequestCollisionError(ReviewStoreError):
    """Raised when an existing request ID is reused with different inputs."""


class BudgetExceededError(ReviewStoreError):
    """Raised when a new polish claim would exceed a configured limit."""


def director_take_signature(state: Any, index: int) -> str:
    """Hash the selected Director clip's creative state for stale-run checks.

    Injected output-path aliases are deliberately omitted from the clip payload;
    callers bind the actual selected media path and content fingerprint
    separately. The pipeline parameters and model/type are included so an edit
    to the render setup invalidates a previously inspected take signature.
    """
    if not isinstance(state, Mapping):
        raise ValueError("Director pipeline state must be an object")
    if type(index) is not int or index < 0:
        raise ValueError("clip index must be a non-negative integer")
    clips = state.get("clips")
    if not isinstance(clips, list) or index >= len(clips):
        raise ValueError("selected Director clip does not exist")
    clip = clips[index]
    if not isinstance(clip, Mapping):
        raise ValueError("selected Director clip must be an object")
    payload = {
        "clip": {
            key: value for key, value in clip.items()
            if key not in {"video_path", "video_file", "output_path"}
        },
        "params": state.get("_params_snapshot"),
        "video_model": state.get("video_model"),
        "pipeline_type": state.get("pipeline_type"),
    }
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _clean_text(value: Any, *, field: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str):
        raise ReviewValidationError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not minimum <= len(cleaned) <= maximum:
        raise ReviewValidationError(
            f"{field} must contain {minimum} to {maximum} non-whitespace characters"
        )
    return cleaned


def normalize_take_review(body: Any) -> dict[str, Any]:
    """Validate and normalize an explicit creative assessment from an agent.

    The accepted object has exactly verdict, summary, and observations. Every
    verdict, including keep, requires at least one categorized, severity-rated,
    non-generic observation. An empty assessment cannot turn into an automatic
    keep.
    """
    if not isinstance(body, Mapping):
        raise ReviewValidationError("review must be an object")
    if set(body) != {"verdict", "summary", "observations"}:
        raise ReviewValidationError(
            "review must contain exactly verdict, summary, and observations"
        )

    verdict = body.get("verdict")
    if not isinstance(verdict, str) or verdict not in {"keep", "needs_work", "reject"}:
        raise ReviewValidationError("verdict must be keep, needs_work, or reject")

    summary = _clean_text(
        body.get("summary"), field="summary", minimum=8, maximum=_MAX_SUMMARY_CHARS
    )
    observations = body.get("observations")
    if not isinstance(observations, list) or not 1 <= len(observations) <= _MAX_OBSERVATIONS:
        raise ReviewValidationError(
            f"observations must contain 1 to {_MAX_OBSERVATIONS} items"
        )

    normalized_observations: list[dict[str, str]] = []
    for index, item in enumerate(observations):
        if not isinstance(item, Mapping) or set(item) != {"category", "severity", "text"}:
            raise ReviewValidationError(
                f"observations[{index}] must contain exactly category, severity, and text"
            )
        category = item.get("category")
        severity = item.get("severity")
        if not isinstance(category, str) or category not in _REVIEW_CATEGORIES:
            raise ReviewValidationError(f"observations[{index}].category is unsupported")
        if not isinstance(severity, str) or severity not in _REVIEW_SEVERITIES:
            raise ReviewValidationError(f"observations[{index}].severity is unsupported")
        text = _clean_text(
            item.get("text"),
            field=f"observations[{index}].text",
            minimum=12,
            maximum=_MAX_OBSERVATION_CHARS,
        )
        if (
            text.casefold().rstrip(".!? ") in _GENERIC_OBSERVATIONS
            or _GENERIC_OBSERVATION_RE.fullmatch(text)
        ):
            raise ReviewValidationError(
                f"observations[{index}].text must describe a concrete observation"
            )
        normalized_observations.append({
            "category": category,
            "severity": severity,
            "text": text,
        })

    return {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "creative_status": "assessed",
        "verdict": verdict,
        "summary": summary,
        "observations": normalized_observations,
    }


def _copy_tail(pipe: Any, tail: bytearray, total: list[int], limit: int) -> None:
    while True:
        chunk = pipe.read(8192)
        if not chunk:
            return
        total[0] += len(chunk)
        tail.extend(chunk)
        if len(tail) > limit:
            del tail[: len(tail) - limit]


def _run_bounded(command: list[str], timeout: float, *, capture_stdout: bool) -> dict[str, Any]:
    """Run a no-shell subprocess while bounding time and retained output."""
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=_WINDOWS_CREATE_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        return {
            "status": "unavailable", "returncode": None, "stdout": b"", "stderr": b"",
            "error": f"Executable unavailable: {exc}",
        }
    except OSError as exc:
        return {
            "status": "failed", "returncode": None, "stdout": b"", "stderr": b"",
            "error": f"Could not start executable: {exc}",
        }

    stdout_tail = bytearray()
    stdout_total = [0]
    stderr_tail = bytearray()
    stderr_total = [0]
    readers: list[threading.Thread] = []
    if capture_stdout and process.stdout is not None:
        readers.append(threading.Thread(
            target=_copy_tail,
            args=(process.stdout, stdout_tail, stdout_total, _MAX_STDOUT_BYTES),
            daemon=True,
        ))
    if process.stderr is not None:
        readers.append(threading.Thread(
            target=_copy_tail,
            args=(process.stderr, stderr_tail, stderr_total, _MAX_STDERR_BYTES),
            daemon=True,
        ))
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
    finally:
        for reader in readers:
            reader.join(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    status = "timed_out" if timed_out else ("ok" if process.returncode == 0 else "failed")
    error = f"Process exceeded {timeout:g} second timeout" if timed_out else None
    return {
        "status": status,
        "returncode": process.returncode,
        "stdout": bytes(stdout_tail),
        "stdout_truncated": stdout_total[0] > _MAX_STDOUT_BYTES,
        "stderr": bytes(stderr_tail),
        "stderr_truncated": stderr_total[0] > _MAX_STDERR_BYTES,
        "error": error,
    }


def _decode_bytes(value: bytes) -> str:
    return value.decode("utf-8", errors="replace").strip()


def _command_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {"status": result["status"]}
    if result.get("returncode") is not None:
        evidence["returncode"] = result["returncode"]
    stderr = _decode_bytes(result.get("stderr", b""))
    if stderr:
        evidence["stderr_tail"] = stderr[-_MAX_STDERR_BYTES:]
    if result.get("stderr_truncated"):
        evidence["stderr_truncated"] = True
    if result.get("error"):
        evidence["error"] = result["error"]
    return evidence


def _resolve_executable(value: Any, name: str) -> str | None:
    if value is None:
        return shutil.which(name)
    try:
        executable = os.fspath(value)
    except TypeError as exc:
        raise TypeError(f"{name}_path must be a path-like value or None") from exc
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError(f"{name}_path must be a non-empty path or None")
    return executable


def _positive_duration(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive number or None")
    duration = float(value)
    if not math.isfinite(duration) or not 0 < duration <= 24 * 60 * 60:
        raise ValueError(f"{field} must be greater than 0 and at most 24 hours")
    return duration


def _safe_artifact_path(directory: Path, token: str, label: str, suffix: str) -> Path:
    return directory / f"take-review-{token}-{label}{suffix}"


def _partial_path(target: Path) -> Path:
    return target.with_name(f"{target.stem}.partial-{uuid.uuid4().hex}{target.suffix}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = _partial_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _fraction(value: Any) -> float | None:
    if not isinstance(value, str) or not value or value in {"N/A", "0/0"}:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            result = float(numerator) / denominator_value
        else:
            result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _parse_probe(stdout: bytes) -> dict[str, Any]:
    document = json.loads(stdout.decode("utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("ffprobe JSON root must be an object")
    streams = document.get("streams")
    if not isinstance(streams, list):
        raise ValueError("ffprobe JSON did not contain a stream list")
    video_streams: list[dict[str, Any]] = []
    audio_streams: list[dict[str, Any]] = []
    for stream in streams:
        if not isinstance(stream, Mapping):
            continue
        stream_type = stream.get("codec_type")
        normalized: dict[str, Any] = {
            "index": stream.get("index") if isinstance(stream.get("index"), int) else None,
            "codec": stream.get("codec_name") if isinstance(stream.get("codec_name"), str) else None,
        }
        stream_duration = _fraction(stream.get("duration"))
        if stream_duration is not None and stream_duration >= 0:
            normalized["duration_seconds"] = stream_duration
        if stream_type == "video":
            for field in ("width", "height"):
                if isinstance(stream.get(field), int) and stream[field] > 0:
                    normalized[field] = stream[field]
            for source, dest in (("avg_frame_rate", "average_frame_rate"), ("r_frame_rate", "frame_rate")):
                parsed = _fraction(stream.get(source))
                if parsed is not None:
                    normalized[dest] = parsed
            sar = stream.get("sample_aspect_ratio")
            if isinstance(sar, str) and sar not in {"", "N/A", "0:1"}:
                normalized["sample_aspect_ratio"] = sar
            video_streams.append(normalized)
        elif stream_type == "audio":
            for field in ("channels", "sample_rate"):
                raw_value = stream.get(field)
                if isinstance(raw_value, int):
                    normalized[field] = raw_value
                elif field == "sample_rate" and isinstance(raw_value, str) and raw_value.isdigit():
                    normalized[field] = int(raw_value)
            audio_streams.append(normalized)

    format_info = document.get("format")
    if not isinstance(format_info, Mapping):
        format_info = {}
    duration = _fraction(format_info.get("duration"))
    if duration is None or duration < 0:
        durations = [
            stream["duration_seconds"]
            for stream in video_streams + audio_streams
            if "duration_seconds" in stream
        ]
        duration = max(durations) if durations else None

    geometry: dict[str, Any] = {
        "width": None, "height": None, "sample_aspect_ratio": None,
        "display_aspect_ratio": None,
    }
    if video_streams:
        first_video = video_streams[0]
        geometry["width"] = first_video.get("width")
        geometry["height"] = first_video.get("height")
        geometry["sample_aspect_ratio"] = first_video.get("sample_aspect_ratio", "1:1")
        width = first_video.get("width")
        height = first_video.get("height")
        sar_value = geometry["sample_aspect_ratio"]
        sar = _fraction(sar_value.replace(":", "/")) if isinstance(sar_value, str) else 1.0
        if width and height and sar and sar > 0:
            geometry["display_aspect_ratio"] = round((width * sar) / height, 6)

    return {
        "duration_seconds": duration,
        "format_name": format_info.get("format_name") if isinstance(format_info.get("format_name"), str) else None,
        "video_streams": video_streams,
        "audio_streams": audio_streams,
        "geometry": geometry,
    }


def _run_artifact_command(command: list[str], temporary: Path, target: Path, timeout: float) -> dict[str, Any]:
    result = _run_bounded(command, timeout, capture_stdout=False)
    evidence = _command_evidence(result)
    if result["status"] == "ok" and temporary.is_file() and temporary.stat().st_size > 0:
        try:
            os.replace(temporary, target)
            evidence.update({"status": "created", "path": str(target)})
            return evidence
        except OSError as exc:
            evidence["status"] = "failed"
            evidence["error"] = f"Could not atomically publish artifact: {exc}"
    elif result["status"] == "ok":
        evidence["status"] = "failed"
        evidence["error"] = "Command succeeded but produced no artifact"
    if temporary.exists():
        evidence["partial_path"] = str(temporary)
    return evidence


def _make_contact_sheet(paths: list[str], target: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return {"status": "unavailable", "reason": "Pillow is not installed"}
    if not paths:
        return {"status": "skipped", "reason": "No stills were extracted"}

    cell_width, image_height, label_height = 320, 190, 26
    sheet = Image.new("RGB", (cell_width * len(paths), image_height + label_height), "#111827")
    draw = ImageDraw.Draw(sheet)
    errors: list[str] = []
    resampling = getattr(Image, "Resampling", Image)
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as source:
                frame = source.convert("RGB")
                frame.thumbnail((cell_width, image_height), resampling.LANCZOS)
                x = index * cell_width + (cell_width - frame.width) // 2
                y = (image_height - frame.height) // 2
                sheet.paste(frame, (x, y))
            draw.text((index * cell_width + 8, image_height + 5), f"Frame {index + 1}", fill="white")
        except Exception as exc:
            errors.append(f"{Path(path).name}: {exc}")
    temporary = _partial_path(target)
    try:
        sheet.save(temporary, format="PNG")
        os.replace(temporary, target)
        result: dict[str, Any] = {"status": "created", "path": str(target)}
        if errors:
            result["errors"] = errors
        return result
    except Exception as exc:
        if temporary.exists():
            return {"status": "failed", "error": str(exc), "partial_path": str(temporary)}
        return {"status": "failed", "error": str(exc)}
    finally:
        sheet.close()


def inspect_take(
    media_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    ffmpeg_path: str | os.PathLike[str] | None = None,
    ffprobe_path: str | os.PathLike[str] | None = None,
    expected_duration: float | None = None,
) -> dict[str, Any]:
    """Return a bounded technical review packet and local preview artifacts.

    Media must be an existing regular file. The caller supplies an authorized
    output directory. ffmpeg/ffprobe are invoked with argv arrays and no shell;
    probe, decode, still, and audio commands all have fixed timeouts. Decode and
    extraction force FFmpeg's software path. Creative verdict remains explicitly
    unassessed and must be supplied separately through normalize_take_review.
    """
    try:
        media = Path(media_path)
    except TypeError as exc:
        raise TypeError("media_path must be path-like") from exc
    if not media.exists():
        raise FileNotFoundError(f"Media file does not exist: {media}")
    if not media.is_file():
        raise ValueError("media_path must name a regular file")
    media = media.resolve()
    expected = _positive_duration(expected_duration, "expected_duration")

    try:
        output = Path(output_dir)
    except TypeError as exc:
        raise TypeError("output_dir must be path-like") from exc
    if output.exists() and not output.is_dir():
        raise ValueError("output_dir must name a directory")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()

    ffmpeg = _resolve_executable(ffmpeg_path, "ffmpeg")
    ffprobe = _resolve_executable(ffprobe_path, "ffprobe")
    token = uuid.uuid4().hex[:12]
    packet_path = _safe_artifact_path(output, token, "packet", ".json")

    probe_status: dict[str, Any]
    parsed_probe: dict[str, Any] | None = None
    if ffprobe is None:
        probe_status = {"status": "unavailable", "error": "ffprobe executable was not found"}
    else:
        probe_command = [
            ffprobe, "-v", "error", "-show_entries",
            "format=duration,format_name:stream=index,codec_type,codec_name,width,height,"
            "avg_frame_rate,r_frame_rate,duration,sample_aspect_ratio,channels,sample_rate",
            "-of", "json", str(media),
        ]
        probe_result = _run_bounded(probe_command, _MAX_PROBE_SECONDS, capture_stdout=True)
        probe_status = _command_evidence(probe_result)
        if probe_result["status"] == "ok":
            if probe_result.get("stdout_truncated"):
                probe_status = {"status": "failed", "error": "ffprobe output exceeded the safety limit"}
            else:
                try:
                    parsed_probe = _parse_probe(probe_result["stdout"])
                    probe_status["status"] = "available"
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    probe_status = {"status": "failed", "error": f"Could not parse ffprobe output: {exc}"}

    duration = parsed_probe.get("duration_seconds") if parsed_probe else None
    video_streams = parsed_probe.get("video_streams", []) if parsed_probe else []
    audio_streams = parsed_probe.get("audio_streams", []) if parsed_probe else []
    has_media_stream = bool(video_streams or audio_streams)

    if ffmpeg is None:
        decode = {"status": "unavailable", "error": "ffmpeg executable was not found"}
    elif parsed_probe is None:
        decode = {"status": "unknown", "reason": "ffprobe did not establish the media streams"}
    elif not has_media_stream:
        decode = {"status": "skipped", "reason": "No audio or video streams were found"}
    else:
        decode_command = [
            ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-hwaccel", "none",
            "-i", str(media), "-map", "0:v?", "-map", "0:a?", "-f", "null", "-",
        ]
        decode_result = _run_bounded(decode_command, _MAX_DECODE_SECONDS, capture_stdout=False)
        decode = _command_evidence(decode_result)
        if decode["status"] == "ok":
            decode["status"] = "passed"
        decode["scope"] = "all_video_and_audio_streams"

    stills: dict[str, Any] = {"status": "skipped", "paths": [], "sample_times_seconds": []}
    still_paths: list[str] = []
    still_errors: list[dict[str, Any]] = []
    sample_times: list[float] = []
    still_basis = duration if duration is not None and duration > 0 else expected
    if ffmpeg is None:
        stills = {"status": "unavailable", "reason": "ffmpeg executable was not found", "paths": [], "sample_times_seconds": []}
    elif not video_streams:
        reason = "ffprobe did not establish a video stream" if parsed_probe is None else "No video stream was found"
        stills = {"status": "skipped", "reason": reason, "paths": [], "sample_times_seconds": []}
    elif still_basis is None:
        stills = {"status": "skipped", "reason": "Duration is unknown; five spread samples cannot be selected", "paths": [], "sample_times_seconds": []}
    else:
        fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
        frame_rate = None
        if video_streams:
            first_video = video_streams[0]
            frame_rate = first_video.get("average_frame_rate") or first_video.get("frame_rate")
        last_sample = max(0.0, still_basis - 0.001)
        if isinstance(frame_rate, (int, float)) and frame_rate > 0:
            last_sample = max(0.0, still_basis - 1.0 / frame_rate)
        for index, fraction in enumerate(fractions, start=1):
            sample_time = min(max(0.0, still_basis * fraction), last_sample)
            sample_times.append(round(sample_time, 3))
            target = _safe_artifact_path(output, token, f"still-{index:02d}", ".png")
            temporary = _partial_path(target)
            command = [
                ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-hwaccel", "none",
                "-ss", f"{sample_time:.3f}", "-i", str(media), "-frames:v", "1", "-an",
                "-sn", "-dn", "-vf", "scale=640:640:force_original_aspect_ratio=decrease",
                "-y", str(temporary),
            ]
            artifact = _run_artifact_command(command, temporary, target, _MAX_STILL_SECONDS)
            if artifact.get("status") == "created":
                still_paths.append(artifact["path"])
            else:
                artifact["sample_time_seconds"] = sample_times[-1]
                still_errors.append(artifact)
        stills = {
            "status": "complete" if len(still_paths) == 5 else ("partial" if still_paths else "failed"),
            "paths": still_paths,
            "sample_times_seconds": sample_times,
        }
        if still_errors:
            stills["errors"] = still_errors

    contact_target = _safe_artifact_path(output, token, "contact-sheet", ".png")
    if still_paths:
        contact_sheet = _make_contact_sheet(still_paths, contact_target)
    else:
        contact_sheet = {"status": "skipped", "reason": "No stills were extracted"}

    if ffmpeg is None:
        audio_preview = {"status": "unavailable", "reason": "ffmpeg executable was not found"}
    elif not audio_streams:
        reason = "ffprobe did not establish an audio stream" if parsed_probe is None else "No audio stream was found"
        audio_preview = {"status": "skipped", "reason": reason}
    else:
        audio_target = _safe_artifact_path(output, token, "audio-preview", ".wav")
        audio_temporary = _partial_path(audio_target)
        audio_command = [
            ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-hwaccel", "none",
            "-i", str(media), "-t", str(_AUDIO_PREVIEW_SECONDS), "-map", "0:a:0", "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", "-y",
            str(audio_temporary),
        ]
        audio_preview = _run_artifact_command(
            audio_command, audio_temporary, audio_target, _MAX_AUDIO_PREVIEW_SECONDS
        )

    geometry = parsed_probe.get("geometry") if parsed_probe else {
        "width": None, "height": None, "sample_aspect_ratio": None,
        "display_aspect_ratio": None,
    }
    duration_delta = None
    if expected is not None and duration is not None:
        duration_delta = round(duration - expected, 3)
    packet: dict[str, Any] = {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "review_type": "technical_take_review",
        "creative_status": "not_assessed",
        "creative_assessment": None,
        "source": {"name": media.name, "size_bytes": media.stat().st_size},
        "technical": {
            "probe": probe_status,
            "duration_seconds": duration,
            "expected_duration_seconds": expected,
            "duration_delta_seconds": duration_delta,
            "geometry": geometry,
            "video_streams": video_streams,
            "audio": {
                "present": bool(audio_streams) if parsed_probe is not None else None,
                "streams": audio_streams,
            },
            "container_format": parsed_probe.get("format_name") if parsed_probe else None,
            "software_decode": decode,
        },
        "artifacts": {
            "stills": stills,
            "contact_sheet": contact_sheet,
            "audio_preview": audio_preview,
            "packet": {"status": "written", "path": str(packet_path)},
        },
        "creative_review": {
            "status": "pending",
            "unassessed": ["visual quality", "prompt adherence", "continuity", "creative intent"],
        },
    }
    try:
        _write_json_atomic(packet_path, packet)
    except Exception as exc:
        packet["artifacts"]["packet"] = {"status": "failed", "path": str(packet_path), "error": str(exc)}
    return packet


_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock_for(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.resolve(strict=False)))
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _STORE_LOCKS[key] = lock
        return lock


@contextmanager
def _exclusive_file_lock(path: Path):
    """Lock across threads and processes on supported Windows/POSIX hosts."""
    thread_lock = _thread_lock_for(path)
    with thread_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if msvcrt is not None:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise ReviewStoreError("Timed out waiting for the review ledger lock")
                        time.sleep(0.05)
            elif fcntl is not None:  # pragma: no cover - platform-specific branch
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            else:  # pragma: no cover - unsupported platform
                raise ReviewStoreError("No supported file-locking primitive is available")
            try:
                yield
            finally:
                handle.seek(0)
                if msvcrt is not None:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                elif fcntl is not None:  # pragma: no cover - platform-specific branch
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_store_limit(value: Any, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{field} must be an integer from {low} to {high}")
    return value


def _ledger_text(value: Any, field: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(ord(char) < 32 for char in cleaned):
        raise ValueError(f"{field} must be 1 to {maximum} printable characters")
    return cleaned


class PolishBudgetReviewStore:
    """Durable, caller-scoped intent ledger for bounded polish renders.

    Construct one store per trusted owner/project directory. Budget values are
    persisted and must match on later construction; changing limits never
    resets existing intents. claim is the durable pre-render reservation. A
    caller should submit only after it receives the claim, then persist the
    owned renderer receipt with bind_receipt. Ambiguous or failed renderer
    attempts remain counted; there is deliberately no automatic release path.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        max_renders: int,
        max_retakes_per_clip: int,
    ) -> None:
        self.max_renders = _validate_store_limit(max_renders, "max_renders", 1, 12)
        self.max_retakes_per_clip = _validate_store_limit(
            max_retakes_per_clip, "max_retakes_per_clip", 1, 3
        )
        try:
            self.directory = Path(directory)
        except TypeError as exc:
            raise TypeError("directory must be path-like") from exc
        if self.directory.exists() and not self.directory.is_dir():
            raise ValueError("directory must name a directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory = self.directory.resolve()
        self._path = self.directory / "polish-review-ledger.json"
        self._lock_path = self.directory / ".polish-review-ledger.lock"
        with _exclusive_file_lock(self._lock_path):
            state = self._read_state()
            if state is None:
                self._write_state(self._empty_state())

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": _STORE_SCHEMA_VERSION,
            "max_renders": self.max_renders,
            "max_retakes_per_clip": self.max_retakes_per_clip,
            "intents": [],
        }

    def _read_state(self) -> dict[str, Any] | None:
        if not self._path.exists():
            return None
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewStoreCorruptError(f"Could not read review ledger: {exc}") from exc
        if not isinstance(state, dict) or state.get("schema_version") != _STORE_SCHEMA_VERSION:
            raise ReviewStoreCorruptError("Review ledger has an unsupported schema")
        persisted_renders = state.get("max_renders")
        persisted_retakes = state.get("max_retakes_per_clip")
        if (
            isinstance(persisted_renders, bool)
            or not isinstance(persisted_renders, int)
            or isinstance(persisted_retakes, bool)
            or not isinstance(persisted_retakes, int)
            or persisted_renders != self.max_renders
            or persisted_retakes != self.max_retakes_per_clip
        ):
            raise ReviewStoreError("Configured polish limits do not match the persisted ledger")
        intents = state.get("intents")
        if not isinstance(intents, list):
            raise ReviewStoreCorruptError("Review ledger intents must be a list")
        if len(intents) > self.max_renders:
            raise ReviewStoreCorruptError("Review ledger contains more intents than its render limit")
        seen: set[str] = set()
        seen_receipts: set[str] = set()
        clip_claims: dict[str, int] = {}
        for intent in intents:
            required = {
                "request_id", "clip_id", "source_fingerprint", "request_fingerprint",
                "intent_id", "status", "receipt_id", "created_at",
            }
            if not isinstance(intent, dict) or not required <= set(intent):
                raise ReviewStoreCorruptError("Review ledger contains a malformed intent")
            try:
                for field in (
                    "request_id", "clip_id", "source_fingerprint",
                    "request_fingerprint", "intent_id", "created_at",
                ):
                    _ledger_text(intent[field], field)
            except (TypeError, ValueError) as exc:
                raise ReviewStoreCorruptError(f"Review ledger contains invalid {field}") from exc
            status = intent.get("status")
            if not isinstance(status, str) or status not in {"claimed", "submitted"}:
                raise ReviewStoreCorruptError("Review ledger contains an unsupported intent status")
            receipt_id = intent.get("receipt_id")
            if status == "claimed" and receipt_id is not None:
                raise ReviewStoreCorruptError("Claimed intent cannot already have a receipt")
            if status == "submitted":
                try:
                    receipt_id = _ledger_text(receipt_id, "receipt_id")
                except (TypeError, ValueError) as exc:
                    raise ReviewStoreCorruptError("Submitted intent is missing its receipt") from exc
                if receipt_id in seen_receipts:
                    raise ReviewStoreCorruptError("Review ledger contains a duplicate renderer receipt")
                seen_receipts.add(receipt_id)
            if intent["request_id"] in seen:
                raise ReviewStoreCorruptError("Review ledger contains duplicate request IDs")
            seen.add(intent["request_id"])
            clip_claims[intent["clip_id"]] = clip_claims.get(intent["clip_id"], 0) + 1
        if any(count > self.max_retakes_per_clip for count in clip_claims.values()):
            raise ReviewStoreCorruptError("Review ledger exceeds a per-clip retake limit")
        return state

    def _write_state(self, state: Mapping[str, Any]) -> None:
        try:
            _write_json_atomic(self._path, state)
        except OSError as exc:
            raise ReviewStoreError(f"Could not persist review ledger: {exc}") from exc

    def claim(
        self,
        request_id: str,
        clip_id: str,
        source_fingerprint: str,
        request_fingerprint: str,
    ) -> dict[str, Any]:
        """Persist a claim, returning its existing or newly created intent."""
        intent, _created = self.claim_once(
            request_id, clip_id, source_fingerprint, request_fingerprint
        )
        return intent

    def claim_once(
        self,
        request_id: str,
        clip_id: str,
        source_fingerprint: str,
        request_fingerprint: str,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically return an intent and whether this call created it.

        The created flag is decided inside the same cross-process lock as the
        durable read/modify/write. Dispatchers must submit renderer work only
        when it is true; duplicate request IDs return the same intent with
        false and consume no additional budget.
        """
        request_id = _ledger_text(request_id, "request_id")
        clip_id = _ledger_text(clip_id, "clip_id")
        source_fingerprint = _ledger_text(source_fingerprint, "source_fingerprint")
        request_fingerprint = _ledger_text(request_fingerprint, "request_fingerprint")
        payload = {
            "clip_id": clip_id,
            "source_fingerprint": source_fingerprint,
            "request_fingerprint": request_fingerprint,
        }
        with _exclusive_file_lock(self._lock_path):
            state = self._read_state()
            if state is None:
                raise ReviewStoreError("Review ledger disappeared after initialization")
            for intent in state["intents"]:
                if intent["request_id"] == request_id:
                    existing_payload = {key: intent.get(key) for key in payload}
                    if existing_payload != payload:
                        raise RequestCollisionError(
                            "request_id is already bound to different clip/source/request fingerprints"
                        )
                    return copy.deepcopy(intent), False
            if len(state["intents"]) >= self.max_renders:
                raise BudgetExceededError("Project polish render limit has been reached")
            clip_claims = sum(1 for intent in state["intents"] if intent.get("clip_id") == clip_id)
            if clip_claims >= self.max_retakes_per_clip:
                raise BudgetExceededError("Per-clip retake limit has been reached")

            intent = {
                "request_id": request_id,
                **payload,
                "intent_id": uuid.uuid4().hex,
                "status": "claimed",
                "receipt_id": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            state["intents"].append(intent)
            self._write_state(state)
            return copy.deepcopy(intent), True

    def bind_receipt(self, request_id: str, receipt_id: str) -> dict[str, Any]:
        """Bind the caller-owned native receipt after an accepted renderer POST."""
        request_id = _ledger_text(request_id, "request_id")
        receipt_id = _ledger_text(receipt_id, "receipt_id")
        with _exclusive_file_lock(self._lock_path):
            state = self._read_state()
            if state is None:
                raise ReviewStoreError("Review ledger disappeared after initialization")
            intent = next((item for item in state["intents"] if item["request_id"] == request_id), None)
            if intent is None:
                raise ReviewStoreError("Cannot bind a receipt before claiming its request")
            current_receipt = intent.get("receipt_id")
            if current_receipt is not None and current_receipt != receipt_id:
                raise RequestCollisionError("request_id is already bound to a different renderer receipt")
            if current_receipt is None:
                if any(
                    item["request_id"] != request_id and item.get("receipt_id") == receipt_id
                    for item in state["intents"]
                ):
                    raise RequestCollisionError("renderer receipt is already bound to another request")
                intent["receipt_id"] = receipt_id
                intent["status"] = "submitted"
                intent["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._write_state(state)
            return copy.deepcopy(intent)

    def get(self, request_id: str) -> dict[str, Any] | None:
        """Return one JSON-ready intent, including any receipt binding."""
        request_id = _ledger_text(request_id, "request_id")
        with _exclusive_file_lock(self._lock_path):
            state = self._read_state()
            if state is None:
                raise ReviewStoreError("Review ledger disappeared after initialization")
            intent = next((item for item in state["intents"] if item["request_id"] == request_id), None)
            return copy.deepcopy(intent) if intent is not None else None

    def list_intents(self) -> list[dict[str, Any]]:
        """Return all durable budget intents as JSON-ready dictionaries."""
        with _exclusive_file_lock(self._lock_path):
            state = self._read_state()
            if state is None:
                raise ReviewStoreError("Review ledger disappeared after initialization")
            return copy.deepcopy(state["intents"])

    def snapshot(self) -> dict[str, Any]:
        """Return persisted limits and current counts for diagnostics/API output."""
        intents = self.list_intents()
        per_clip: dict[str, int] = {}
        for intent in intents:
            clip_id = intent["clip_id"]
            per_clip[clip_id] = per_clip.get(clip_id, 0) + 1
        return {
            "max_renders": self.max_renders,
            "max_retakes_per_clip": self.max_retakes_per_clip,
            "renders_claimed": len(intents),
            "per_clip_claims": per_clip,
            "remaining_renders": max(0, self.max_renders - len(intents)),
            "intents": intents,
        }
