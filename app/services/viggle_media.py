"""Prepare Animate's selected source range without changing saved job inputs."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

FPS = 24


def selected_range(body: dict) -> tuple[float, float] | None:
    """Validate source-relative bounds; old jobs without a selection stay valid."""
    start, end = body.get("_viggle_trim_start"), body.get("_viggle_trim_end")
    if start is None and end is None:
        return None
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        raise ValueError("Choose both a start and end time for the Animate clip.") from None
    if (not math.isfinite(start) or not math.isfinite(end) or start < 0
            or not 1 / FPS - 1e-9 <= end - start <= 3600):
        raise ValueError("The Animate trim must span at least one frame and at most one hour.")
    character = body.get("viggle_character") or {}
    if not isinstance(character, dict):
        raise ValueError("Choose a valid Animate character reference.")
    frame = character.get("frame_seconds", body.get("_viggle_frame_seconds", start))
    try:
        frame = float(frame)
    except (TypeError, ValueError):
        raise ValueError("Choose a frame inside the selected Animate range.") from None
    if not math.isfinite(frame) or not start <= frame < end:
        raise ValueError("Choose a frame inside the selected Animate range.")
    return start, end


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
    selection = selected_range(body)
    if selection is None:
        yield {}
        return
    start, end = selection
    resolved = resolve_media(body["video_guide"])
    if not resolved or not Path(resolved).is_file():
        raise ValueError("The Animate control video is missing. Load it again before generating.")
    source = Path(resolved)
    metadata = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration:format=duration", "-of", "json", str(source)], aborted))
    streams = metadata.get("streams") or []
    try:
        duration = float(streams[0].get("duration") or metadata["format"]["duration"])
    except (IndexError, KeyError, TypeError, ValueError):
        raise ValueError("Could not read the control video's duration.") from None
    if not math.isfinite(duration) or start >= duration or end > duration + 1 / FPS:
        raise ValueError("The Animate trim is outside the control video. Reload it and choose a valid range.")
    end = min(end, duration)
    length = end - start
    runtime = {"video_length": min(int(body.get("video_length") or 124),
                                   max(1, math.floor(length * FPS + 1e-6)))}
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
        if body.get("audio_prompt_type") == "A" and body.get("audio_guide"):
            update("Trimming Animate audio…")
            audio = Path(directory) / "audio.wav"
            _run([*common, "-ss", str(start), "-i", str(resolve_media(body["audio_guide"])),
                  "-t", str(length), "-vn", "-c:a", "pcm_s16le", str(audio)], aborted)
            runtime["audio_guide"] = str(audio)
        yield runtime
