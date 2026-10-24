"""Fish Drama 3 duration budgets.

Auto is a generous generation ceiling, not a pacing target. It counts printable
Unicode code points after removing speaker labels/tags, square-bracket cues,
and paired delivery tags, then budgets ``min(3600, 8 + ceil(units / 4))``
seconds. The text itself is never rewritten here; EOS and cancellation still
end a take.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

AUTO_DURATION_MAX_SECONDS = 3600
AUTO_DURATION_MARGIN_SECONDS = 8
AUTO_DURATION_UNITS_PER_SECOND = 4
LONGFORM_CHUNK_MAX_SECONDS = 120
LONGFORM_CHUNK_MAX_UNITS = (
    LONGFORM_CHUNK_MAX_SECONDS - AUTO_DURATION_MARGIN_SECONDS
) * AUTO_DURATION_UNITS_PER_SECOND

_SPEAKER_LABELS = re.compile(
    r"^[ \t]*Speaker[ \t]*\d+[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE
)
_SPEAKER_TAGS = re.compile(r"<\|speaker:\d+\|>")
_DIRECTION_CUES = re.compile(r"\[[^\]]*\]", re.DOTALL)
_DELIVERY_TAGS = re.compile(r"</?(?:whisper|stress|slow)>", re.IGNORECASE)


@dataclass(frozen=True)
class DurationBudget:
    frames: int
    automatic: bool
    seconds: float
    script_units: int | None = None


def script_speech_units(script: str) -> int:
    """Count visible Unicode code points without speaker or performance cues."""
    analysis_text = _SPEAKER_LABELS.sub(" ", script)
    analysis_text = _SPEAKER_TAGS.sub(" ", analysis_text)
    analysis_text = _DIRECTION_CUES.sub(" ", analysis_text)
    analysis_text = _DELIVERY_TAGS.sub(" ", analysis_text)
    return sum(
        not character.isspace()
        and not unicodedata.category(character).startswith("C")
        for character in analysis_text
    )


def duration_frame_budget(
    duration_seconds: float,
    script: str,
    sample_rate: int,
    frame_length: int,
) -> DurationBudget:
    """Resolve Auto or a manual maximum to an integer frame budget.

    Positive durations retain the existing floor-to-codec-frame rule. Zero
    selects the script-derived Auto budget. Negative and non-finite requests
    are rejected instead of silently becoming a one-frame cap.
    """
    try:
        seconds = float(duration_seconds)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "duration_seconds must be 0 (Auto) or a finite positive number"
        ) from error
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            "duration_seconds must be 0 (Auto) or a finite positive number"
        )
    if seconds > AUTO_DURATION_MAX_SECONDS:
        raise ValueError(
            f"duration_seconds must be at most {AUTO_DURATION_MAX_SECONDS} seconds"
        )
    if sample_rate <= 0 or frame_length <= 0:
        raise ValueError("sample_rate and frame_length must be positive")

    if seconds == 0:
        units = script_speech_units(script)
        estimate_seconds = min(
            AUTO_DURATION_MAX_SECONDS,
            AUTO_DURATION_MARGIN_SECONDS
            + (units + AUTO_DURATION_UNITS_PER_SECOND - 1)
            // AUTO_DURATION_UNITS_PER_SECOND,
        )
        frames = max(1, math.floor(estimate_seconds * sample_rate / frame_length))
        return DurationBudget(frames, True, estimate_seconds, units)

    frames = max(1, math.floor(seconds * sample_rate / frame_length))
    return DurationBudget(frames, False, seconds)
