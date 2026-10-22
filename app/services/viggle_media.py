"""Prepare Animate's selected source range without changing saved job inputs."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import subprocess
import threading
import time
from tempfile import TemporaryDirectory

FPS = 24
_PROBE_TIMEOUT_SECONDS = 15
_PACKET_FALLBACK_LIMIT = 200_000
_PROBE_CACHE_LIMIT = 32
_PROBE_CACHE: OrderedDict[tuple[str, int, int], dict] = OrderedDict()
_PROBE_CACHE_LOCK = threading.Lock()


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _positive(value) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _rate(value) -> float | None:
    try:
        numerator, denominator = str(value).split("/", 1)
        denominator = float(denominator)
        result = float(numerator) / denominator
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _duration_tag(value) -> float | None:
    """Read FFprobe duration tags, which commonly store an absolute endpoint."""
    if not isinstance(value, str):
        return _positive(value)
    value = value.strip()
    if ":" not in value:
        return _positive(value)
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes = int(parts[0]), int(parts[1])
        seconds = float(parts[2])
    except (TypeError, ValueError, OverflowError):
        return None
    result = hours * 3600 + minutes * 60 + seconds
    return result if math.isfinite(result) and result > 0 else None


def _source_identity(source: Path) -> tuple[Path, tuple[str, int, int]]:
    path = source.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("The Animate control video is missing. Load it again before generating.")
    stat = path.stat()
    return path, (str(path), stat.st_size, stat.st_mtime_ns)


def _run_probe(command: list[str], aborted, *, timeout_seconds: float = _PROBE_TIMEOUT_SECONDS) -> str:
    """Run FFprobe with polling cancellation and a hard wall-clock bound."""
    if aborted():
        raise InterruptedError("Animate media inspection cancelled.")
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as error:
        raise ValueError(f"FFprobe could not be started: {error}") from error
    with process:
        try:
            while True:
                if aborted():
                    raise InterruptedError("Animate media inspection cancelled.")
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("FFprobe exceeded its inspection time limit.")
                try:
                    stdout, stderr = process.communicate(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
    if process.returncode:
        detail = stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise ValueError(f"FFprobe could not inspect the control video: {detail or 'media inspection failed'}")
    return stdout.decode("utf-8", errors="replace")


def _probe_json(command: list[str], aborted) -> dict | None:
    text = _run_probe(command, aborted)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _ffprobe_metadata(source: Path, aborted) -> dict | None:
    return _probe_json([
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=index,codec_type,disposition,start_time,duration,duration_ts,time_base,nb_frames,avg_frame_rate,r_frame_rate:stream_tags:format=start_time,duration",
        "-of", "json", str(source),
    ], aborted)


def _video_stream(metadata: dict | None) -> dict | None:
    if not isinstance(metadata, dict):
        return None
    streams = metadata.get("streams")
    if not isinstance(streams, list):
        return None
    videos = [stream for stream in streams
              if isinstance(stream, dict) and stream.get("codec_type") == "video"]
    if not videos:
        return None
    return next((stream for stream in videos
                 if not (stream.get("disposition") or {}).get("attached_pic")), videos[0])


def _metadata_duration(stream: dict | None) -> float | None:
    if not isinstance(stream, dict):
        return None
    duration = _positive(stream.get("duration"))
    if duration is not None:
        return duration

    duration_ticks = _positive(stream.get("duration_ts"))
    time_base = _rate(stream.get("time_base"))
    if duration_ticks is not None and time_base is not None:
        # _rate parses rational values as numerator / denominator; a time base
        # is seconds per tick.
        duration = duration_ticks * time_base
        if math.isfinite(duration) and duration > 0:
            return duration

    start = _number(stream.get("start_time")) or 0.0
    tags = stream.get("tags")
    if isinstance(tags, dict):
        for key, value in tags.items():
            if str(key).upper().startswith("DURATION"):
                endpoint = _duration_tag(value)
                if endpoint is not None and endpoint > start:
                    return endpoint - start
    return None


def _packet_duration(source: Path, aborted) -> tuple[float | None, bool]:
    metadata = _probe_json([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts_time,dts_time,duration_time",
        "-read_intervals", f"%+#{_PACKET_FALLBACK_LIMIT}",
        "-of", "json", str(source),
    ], aborted)
    packets = metadata.get("packets") if isinstance(metadata, dict) else None
    if not isinstance(packets, list) or not packets:
        return None, False
    truncated = len(packets) >= _PACKET_FALLBACK_LIMIT
    timestamps: list[float] = []
    ends: list[float] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        timestamp = _number(packet.get("pts_time"))
        if timestamp is None:
            timestamp = _number(packet.get("dts_time"))
        if timestamp is None:
            continue
        timestamps.append(timestamp)
        packet_duration = _positive(packet.get("duration_time"))
        if packet_duration is not None:
            ends.append(timestamp + packet_duration)
    if not timestamps:
        return None, truncated

    first, last = min(timestamps), max(timestamps)
    end = max(ends, default=last)
    # Some codecs omit the final packet duration. Estimate only that last
    # frame's extent from recent presentation-time gaps, preserving VFR timing.
    if end <= last + 1e-9:
        ordered = sorted(set(timestamps))
        gaps = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
        if gaps:
            recent = sorted(gaps[-30:])
            step = recent[len(recent) // 2]
            end = last + step
    elapsed = end - first
    return (elapsed if math.isfinite(elapsed) and elapsed > 0 else None), truncated


def _frame_count_duration(stream: dict | None) -> float | None:
    if not isinstance(stream, dict):
        return None
    try:
        count = int(stream.get("nb_read_frames") or stream.get("nb_frames"))
    except (TypeError, ValueError, OverflowError):
        return None
    rate = _rate(stream.get("avg_frame_rate"))
    if count <= 0 or rate is None:
        return None
    duration = count / rate
    return duration if math.isfinite(duration) and duration > 0 else None


def probe_control_video(source: Path | str, *, aborted=lambda: False) -> dict:
    """Return the elapsed duration of the first playable video track.

    Container duration is reported separately because it may include an audio
    tail. When stream duration metadata is absent, stream tags and bounded
    packet timestamps are preferred over that container-wide value.
    """
    if aborted():
        raise InterruptedError("Animate media inspection cancelled.")
    try:
        path, identity = _source_identity(Path(source))
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("The Animate control video"):
            raise
        raise ValueError("The Animate control video is missing. Load it again before generating.") from error
    with _PROBE_CACHE_LOCK:
        cached = _PROBE_CACHE.get(identity)
        if cached is not None:
            _PROBE_CACHE.move_to_end(identity)
            return dict(cached)

    try:
        metadata = _ffprobe_metadata(path, aborted)
        stream = _video_stream(metadata)
        format_info = metadata.get("format") if isinstance(metadata, dict) else None
        container_duration = _positive(format_info.get("duration")) if isinstance(format_info, dict) else None
        duration = _metadata_duration(stream)
        if duration is None:
            packet_duration, truncated = _packet_duration(path, aborted)
            # A bounded packet prefix is never a complete track duration. Only
            # metadata that describes the whole stream may replace it.
            duration = None if truncated else packet_duration
            if duration is None and stream is not None:
                # A finite frame count and average rate are a last metadata
                # fallback for unusually long files where packet scan hit its cap.
                duration = _frame_count_duration(stream)
    except InterruptedError:
        raise
    except TimeoutError as error:
        raise ValueError("Could not read the control video's duration before the probe timed out.") from error
    if duration is None and stream is not None:
        streams = metadata.get("streams") if isinstance(metadata, dict) else []
        has_audio = any(isinstance(item, dict) and item.get("codec_type") == "audio"
                        for item in (streams or []))
        if not has_audio:
            duration = container_duration
    if duration is None:
        raise ValueError("Could not read the control video's duration.")

    result = {"duration_seconds": duration}
    if container_duration is not None:
        result["container_duration_seconds"] = container_duration
    try:
        if _source_identity(path)[1] != identity:
            raise ValueError("The Animate control video changed while it was being inspected. Try again.")
    except OSError as error:
        raise ValueError("The Animate control video changed while it was being inspected. Try again.") from error
    with _PROBE_CACHE_LOCK:
        _PROBE_CACHE[identity] = dict(result)
        _PROBE_CACHE.move_to_end(identity)
        while len(_PROBE_CACHE) > _PROBE_CACHE_LIMIT:
            _PROBE_CACHE.popitem(last=False)
    return result


def _selection_bounds(body: dict) -> tuple[float, float] | None:
    start, end = body.get("_viggle_trim_start"), body.get("_viggle_trim_end")
    if start is None and end is None:
        return None
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Choose both a start and end time for the Animate clip.") from None
    if (not math.isfinite(start) or not math.isfinite(end) or start < 0
            or not 1 / FPS - 1e-9 <= end - start <= 3600):
        raise ValueError("The Animate trim must span at least one frame and at most one hour.")
    return start, end


def _selection_frame(body: dict, start: float, end: float) -> float:
    character = body.get("viggle_character") or {}
    if not isinstance(character, dict):
        raise ValueError("Choose a valid Animate character reference.")
    frame = character.get("frame_seconds", body.get("_viggle_frame_seconds", start))
    try:
        frame = float(frame)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Choose a frame inside the selected Animate range.") from None
    if not math.isfinite(frame) or not start <= frame < end:
        raise ValueError("Choose a frame inside the selected Animate range.")
    return frame


def selected_range(body: dict) -> tuple[float, float] | None:
    """Validate source-relative bounds; old jobs without a selection stay valid."""
    selection = _selection_bounds(body)
    if selection is None:
        return None
    start, end = selection
    _selection_frame(body, start, end)
    return start, end


def normalize_control_selection(body: dict, *, resolve_media, aborted=lambda: False) -> dict:
    """Return source-verified canonical job fields without mutating ``body``."""
    selection = _selection_bounds(body)
    if selection is None:
        return {}
    start, requested_end = selection
    frame = _selection_frame(body, start, requested_end)
    try:
        resolved = resolve_media(body.get("video_guide"))
    except InterruptedError:
        raise
    if not resolved or not Path(resolved).is_file():
        raise ValueError("The Animate control video is missing. Load it again before generating.")
    info = probe_control_video(resolved, aborted=aborted)
    duration = info["duration_seconds"]
    tolerance = 1 / FPS + 1e-9
    end = requested_end
    if end > duration:
        container_duration = info.get("container_duration_seconds")
        matches_container_end = (container_duration is not None
                                 and abs(end - container_duration) <= tolerance)
        if end <= duration + tolerance or matches_container_end:
            end = duration
        else:
            raise ValueError("The Animate trim is outside the control video. Reload it and choose a valid range.")
    if start >= duration or start >= end:
        raise ValueError("The Animate trim is outside the control video. Reload it and choose a valid range.")
    span = end - start
    if span < 1 / FPS - 1e-9:
        raise ValueError("The Animate trim must span at least one frame and at most one hour.")

    updates = {
        "_viggle_source_seconds": duration,
        "_viggle_trim_start": start,
        "_viggle_trim_end": end,
    }
    try:
        requested_frames = int(float(body.get("video_length") or 124))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Viggle duration must be between one frame and one hour.") from None
    if requested_frames < 1 or requested_frames > 24 * 3600:
        raise ValueError("Viggle duration must be between one frame and one hour.")
    updates["video_length"] = min(requested_frames, max(1, math.floor(span * FPS + 1e-6)))

    if frame >= end:
        # The submitted frame was valid for the client's wider, full-file
        # selection, but the authoritative video stream ends sooner.
        frame = max(start, end - 1 / FPS)
        character = body.get("viggle_character")
        if isinstance(character, dict) and character:
            canonical_character = copy.deepcopy(character)
            canonical_character["frame_seconds"] = frame
            updates["viggle_character"] = canonical_character
        updates["_viggle_frame_seconds"] = frame
    return updates


def _run(command: list[str], aborted) -> str:
    """Keep media preparation cancellable, including while FFmpeg is encoding."""
    if aborted():
        raise InterruptedError("Animate media preparation cancelled.")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
        try:
            while True:
                if aborted():
                    raise InterruptedError("Animate media preparation cancelled.")
                try:
                    stdout, stderr = process.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
        if process.returncode:
            detail = stderr.decode("utf-8", errors="replace").strip()[-1500:]
            raise ValueError(f"Could not prepare the Animate trim: {detail or 'media decoding failed'}")
        return stdout.decode("utf-8", errors="replace")


@contextmanager
def prepare_control_media(body: dict, *, resolve_media, aborted, update):
    """Yield runtime-only paths, then remove them after generation or failure.

    Character preparation must run first against the original video: its frame
    timestamp and cache signature remain absolute. Both input-audio choices use
    the same source interval as the motion, keeping the soundtrack in sync.
    """
    selection = _selection_bounds(body)
    if selection is None:
        yield {}
        return
    effective = dict(body)
    resolved = resolve_media(effective.get("video_guide"))
    if not resolved or not Path(resolved).is_file():
        raise ValueError("The Animate control video is missing. Load it again before generating.")
    canonical = normalize_control_selection(
        effective, resolve_media=lambda _: resolved, aborted=aborted
    )
    effective.update(canonical)
    start, end = selected_range(effective)
    source = Path(resolved)
    duration = canonical["_viggle_source_seconds"]
    length = end - start
    runtime = {"video_length": canonical["video_length"]}
    if start == 0 and abs(end - duration) < 0.001:
        yield runtime
        return

    update("Trimming Animate control video…")
    with TemporaryDirectory(prefix="maestro-viggle-trim-") as directory:
        target = Path(directory) / "control.mkv"
        # Re-encode for accurate cuts between keyframes. Keep the original
        # resolution/frame rate and lossless audio; the H3 loader resamples to 24 fps.
        common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        _run([*common, "-ss", str(start), "-i", str(source), "-t", str(length),
              "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "fast",
              "-crf", "18", "-c:a", "pcm_s16le", str(target)], aborted)
        runtime["video_guide"] = str(target)
        if effective.get("audio_prompt_type") == "A" and effective.get("audio_guide"):
            update("Trimming Animate audio…")
            audio = Path(directory) / "audio.wav"
            _run([*common, "-ss", str(start), "-i", str(resolve_media(effective["audio_guide"])),
                  "-t", str(length), "-vn", "-c:a", "pcm_s16le", str(audio)], aborted)
            runtime["audio_guide"] = str(audio)
        yield runtime
