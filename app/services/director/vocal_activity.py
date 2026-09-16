"""Bounded vocal-activity evidence for Director music-video intervals."""

from __future__ import annotations

import math
import os
import wave
from typing import Any, Optional


_MAX_INTERVAL_SECONDS = 30.0
_MIN_RMS = 0.002
_MIN_ACTIVE_FRACTION = 0.002
_ACTIVE_SAMPLE_LEVEL = 0.01
_SILENT_RMS = 0.00002
_SILENT_PEAK = 0.0001
_MAX_SAMPLE_RATE = 384_000
_MAX_CHANNELS = 32
_READ_CHUNK_FRAMES = 12_000
_NON_VOCAL_TEXT = {"instrumental", "music", "no vocals", "silence"}


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _has_timestamped_vocal(
    lyrics: Optional[list[dict]], start: float, end: float,
) -> bool:
    for entry in lyrics or []:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text or text.strip("[] ").casefold() in _NON_VOCAL_TEXT:
            continue
        item_start = _number(entry.get("start"))
        item_end = _number(entry.get("end"))
        if item_start is None or item_end is None or item_end <= item_start:
            continue
        if item_start < end and item_end > start:
            return True
    return False


def _iter_pcm(raw: bytes, sample_width: int):
    if sample_width == 1:
        yield from (value - 128 for value in raw)
        return
    if sample_width == 2:
        for offset in range(0, len(raw) - 1, 2):
            yield int.from_bytes(raw[offset:offset + 2], "little", signed=True)
        return
    if sample_width == 3:
        for offset in range(0, len(raw) - 2, 3):
            value = int.from_bytes(raw[offset:offset + 3], "little", signed=False)
            if value & 0x800000:
                value -= 1 << 24
            yield value
        return
    if sample_width == 4:
        for offset in range(0, len(raw) - 3, 4):
            yield int.from_bytes(raw[offset:offset + 4], "little", signed=True)


def _classify_wave_interval(
    source: wave.Wave_read, start: float, end: float,
) -> str:
    rate = source.getframerate()
    frames = source.getnframes()
    width = source.getsampwidth()
    channels = source.getnchannels()
    if (
        rate <= 0 or rate > _MAX_SAMPLE_RATE
        or frames <= 0 or channels <= 0 or channels > _MAX_CHANNELS
        or width not in (1, 2, 3, 4)
    ):
        return "unknown"
    start_frame = max(0, int(math.floor(start * rate)))
    end_frame = int(math.ceil(end * rate))
    if start < 0 or end <= start or start_frame >= frames or end_frame > frames:
        return "unknown"
    requested = end_frame - start_frame
    if requested > int(rate * _MAX_INTERVAL_SECONDS):
        return "unknown"
    scale = float(1 << (width * 8 - 1))
    sum_squares = 0.0
    sample_count = 0
    active_count = 0
    peak = 0.0
    source.setpos(start_frame)
    remaining = requested
    while remaining:
        chunk_frames = min(remaining, _READ_CHUNK_FRAMES)
        raw = source.readframes(chunk_frames)
        if len(raw) != chunk_frames * channels * width:
            return "unknown"
        for sample in _iter_pcm(raw, width):
            level = abs(sample) / scale
            sum_squares += level * level
            sample_count += 1
            if level >= _ACTIVE_SAMPLE_LEVEL:
                active_count += 1
            peak = max(peak, level)
        remaining -= chunk_frames
    if not sample_count:
        return "unknown"
    rms = math.sqrt(sum_squares / sample_count)
    active_fraction = active_count / sample_count
    if rms >= _MIN_RMS and active_fraction >= _MIN_ACTIVE_FRACTION:
        return "active"
    if rms <= _SILENT_RMS and peak <= _SILENT_PEAK:
        return "silent"
    return "unknown"


def classify_vocal_intervals(
    clips: list[dict],
    lyrics: Optional[list[dict]],
    vocals_path: Optional[str],
) -> list[str]:
    """Return ``active``, ``silent``, or ``unknown`` for each clip.

    Timed transcript entries are explicit positive evidence. Absence of a
    transcript is never silence. A pre-existing separated-vocal WAV may supply
    positive or negative evidence, and only the requested clip bytes are read.
    """

    source = None
    path = str(vocals_path or "").strip()
    if path and os.path.isfile(path):
        try:
            source = wave.open(path, "rb")
            if source.getcomptype() != "NONE":
                source.close()
                source = None
        except (OSError, EOFError, wave.Error):
            source = None
    results = []
    try:
        for clip in clips:
            start = _number(clip.get("start")) if isinstance(clip, dict) else None
            end = _number(clip.get("end")) if isinstance(clip, dict) else None
            if start is None or end is None or end <= start:
                results.append("unknown")
            elif _has_timestamped_vocal(lyrics, start, end):
                results.append("active")
            elif source is not None:
                try:
                    results.append(_classify_wave_interval(source, start, end))
                except (EOFError, OSError, OverflowError, ValueError, wave.Error):
                    results.append("unknown")
            else:
                results.append("unknown")
    finally:
        if source is not None:
            source.close()
    return results
