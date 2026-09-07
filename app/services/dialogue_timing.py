"""Shared dialogue planning pace and H3 speech interval calculation."""

from __future__ import annotations


DIALOGUE_DEFAULT_WORDS_PER_SECOND = 2.8
DIALOGUE_MAX_WORDS_PER_SECOND = 3.0


def h3_dialogue_schedule(
    word_count: int,
    duration_seconds: float | None,
) -> tuple[float, float, float]:
    """Use the default pace, letting dense lines occupy the full clip.

    Admission checks own the maximum word budget. Timing must not compress a
    valid line to reserve silence: pauses use only time left after the words.
    """

    duration = max(2.0, float(duration_seconds or 8.0))
    speech_duration = min(
        duration,
        max(0, word_count) / DIALOGUE_DEFAULT_WORDS_PER_SECOND,
    )
    start = min(0.25, max(0.0, duration - speech_duration) / 2)
    return duration, start, start + speech_duration
