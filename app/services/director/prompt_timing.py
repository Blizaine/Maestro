"""Validation helpers for clip-local visual timecodes in Director prompts."""

from __future__ import annotations

import math
import re
from typing import Any


_AT_TIME_RE = re.compile(r"\bAt\s+(?P<value>[^\s,;!?]+)", re.IGNORECASE)
_TIME_LIKE_RE = re.compile(r"^(?:\d+|mm|\?{2,})[:.]", re.IGNORECASE)
_LOCAL_TIMECODE_RE = re.compile(
    r"^(?P<minutes>\d{2,}):(?P<seconds>\d{2})(?:\.(?P<fraction>\d{1,3}))?$"
)
_DIALOGUE_TAG_RE = re.compile(r"<d\b[^>]*>.*?</d\s*>", re.IGNORECASE | re.DOTALL)
_SOURCE_TIMING_LABEL_RE = re.compile(
    r"\b(?:source[ -]?(?:song|audio)|song|audio|lyrics?)\b"
    r"(?:[ \t]+(?:lyric|music|vocal|beat|entry))*[ \t]+"
    r"(?:cue|offset|time(?:code|ing)?|timestamp)[ \t]*(?::[ \t]*)?$",
    re.IGNORECASE,
)


def _mask_dialogue_tags(text: str) -> str:
    """Keep offsets stable while excluding spoken words from visual timing checks."""

    return _DIALOGUE_TAG_RE.sub(lambda match: " " * len(match.group(0)), text)


def _is_source_timing_reference(text: str, start: int) -> bool:
    """Return true when an ``At`` time belongs to an explicitly labeled cue."""

    boundaries = [
        text.rfind(marker, 0, start)
        for marker in (".", "!", "?", ";", "\n", "\r")
    ]
    sentence_start = max(boundaries, default=-1) + 1
    prefix = text[max(sentence_start, start - 160):start]
    return bool(_SOURCE_TIMING_LABEL_RE.search(prefix))


def _format_local_timecode(seconds: float) -> str:
    total_millis = max(0, round(seconds * 1000))
    minutes, remainder = divmod(total_millis, 60_000)
    return f"{minutes:02d}:{remainder / 1000:06.3f}"


def music_video_visual_timing_issue(
    prompt: Any,
    duration_seconds: Any,
    *,
    tolerance_seconds: float = 0.05,
) -> str | None:
    """Return the first malformed or out-of-range visual timecode issue.

    Visual cut markers may use either ``[Shot N] At MM:SS[.mmm]`` or a plain
    ``At MM:SS[.mmm]`` sequence in the visual body. One to three fractional
    digits are accepted. The function ignores words
    inside H3 ``<d>`` dialogue tags and timestamps explicitly labeled as
    source-song, lyric, or audio cue timing.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        return None

    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError, OverflowError):
        duration = math.nan
    if not math.isfinite(duration) or duration <= 0:
        duration = math.nan

    text = _mask_dialogue_tags(prompt)
    for match in _AT_TIME_RE.finditer(text):
        raw_value = match.group("value").strip("([{<").rstrip(".,:)]}>")
        if not _TIME_LIKE_RE.match(raw_value):
            continue
        if _is_source_timing_reference(text, match.start()):
            continue

        time_match = _LOCAL_TIMECODE_RE.fullmatch(raw_value)
        seconds_field = int(time_match.group("seconds")) if time_match else -1
        if not time_match or seconds_field >= 60:
            return (
                f"visual timecode '{raw_value}' is malformed; use local "
                "MM:SS or MM:SS.mmm timecodes with seconds below 60"
            )

        fraction = time_match.group("fraction") or ""
        visual_seconds = (
            int(time_match.group("minutes")) * 60
            + seconds_field
            + (int(fraction) / (10 ** len(fraction)) if fraction else 0.0)
        )
        if math.isfinite(duration) and visual_seconds > duration + max(
            0.0, float(tolerance_seconds)
        ):
            return (
                f"visual timecode {raw_value} exceeds local clip duration "
                f"{_format_local_timecode(duration)}"
            )

    return None
