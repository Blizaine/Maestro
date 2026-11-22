"""Live denoising-step timing for Studio progress cards."""

from __future__ import annotations

import re


_DENOISING_PHASE = re.compile(
    r"\b(Denoising(?:\s+(?:First|Second|Third|\d+(?:st|nd|rd|th))\s+Pass)?)\b",
    re.IGNORECASE,
)
_PROGRESS_CONTEXT = re.compile(
    r"\b(?:Prompt|Sample|Sliding Window)\s+\d+\s*/\s*\d+\b",
    re.IGNORECASE,
)


class StudioStepTiming:
    """Average elapsed monotonic time per step in the active denoise pass.

    Samples arrive from the model's existing per-step progress callback, so
    model loading, queue waits, VAE decode and post-processing are excluded.
    Each new denoising stage/window/counter resets the average.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._key: tuple[str, tuple[str, ...], int] | None = None
        self._started_at: float | None = None
        self._start_step: int | None = None
        self._last_step: int | None = None
        self._seconds_per_step: float | None = None

    def observe(self, step: int, total: int, phase: str, *, now: float) -> float | None:
        """Return the current pass average, or ``None`` until two steps exist."""
        label = str(phase or "")
        match = _DENOISING_PHASE.search(label)
        try:
            step = int(step)
            total = int(total)
            now = float(now)
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return None

        # The common worker also prefixes speech-frame progress with Denoising.
        if match is None or "generating speech" in label.casefold() or step <= 0 or total <= 0:
            self.reset()
            return None

        contexts = tuple(item.casefold() for item in _PROGRESS_CONTEXT.findall(label))
        key = (match.group(1).casefold(), contexts, total)
        if (
            key != self._key
            or self._last_step is None
            or step < self._last_step
        ):
            self._key = key
            self._started_at = now
            self._start_step = step
            self._last_step = step
            self._seconds_per_step = None
            return None

        if step == self._last_step:
            return self._seconds_per_step

        self._last_step = step
        completed_steps = step - int(self._start_step or step)
        if completed_steps <= 0 or self._started_at is None:
            return None
        self._seconds_per_step = max(0.0, now - self._started_at) / completed_steps
        return self._seconds_per_step
