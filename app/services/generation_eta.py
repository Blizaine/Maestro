"""Adaptive ETA estimates for long, multi-clip generation jobs.

The sampler's console ETA assumes every denoising step costs the same amount.
That is a poor fit for First Block Cache: the warm-up steps run at full cost,
then later steps become cheaper as cache hits accumulate.  This module keeps
the estimator independent from FastAPI and model code so both the job worker
and its tests can feed it ordinary progress observations.
"""
from __future__ import annotations

import math
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


_WINDOW_RE = re.compile(r"(?:sliding\s+)?window\s+(\d+)\s*/\s*(\d+)", re.I)


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed > 0 else default


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def task_workload(task: Mapping[str, Any]) -> float:
    """Return a relative render-work score for one parsed generation task.

    Director clips in one queue normally share model, resolution, and step
    count, leaving frame count as the main variable.  The modest super-linear
    frame exponent matches attention/runtime growth better than a flat seconds
    per frame rule, while the other factors keep mixed task manifests sane.
    """

    params = task.get("params") if isinstance(task, Mapping) else None
    if not isinstance(params, Mapping):
        params = {}
    frames = _positive_int(
        params.get("video_length", task.get("video_length", 1)),
        1,
    )
    steps = _positive_int(params.get("num_inference_steps", 1), 1)
    resolution = str(params.get("resolution") or "832x480").lower()
    match = re.search(r"(\d+)\s*x\s*(\d+)", resolution)
    if match:
        pixels = max(1, int(match.group(1)) * int(match.group(2)))
    else:
        pixels = 832 * 480
    pixel_factor = (pixels / float(832 * 480)) ** 0.85
    return max(1.0, (frames ** 1.15) * steps * pixel_factor)


def task_cache_start_percent(task: Mapping[str, Any]) -> Optional[float]:
    """Return First Block Cache's warm-up percentage for ``task``."""

    params = task.get("params") if isinstance(task, Mapping) else None
    if not isinstance(params, Mapping):
        return None
    if str(params.get("skip_steps_cache_type") or "").lower() != "first_block":
        return None
    try:
        value = float(params.get("skip_steps_start_step_perc", 25))
    except (TypeError, ValueError):
        value = 25.0
    return min(100.0, max(0.0, value))


def _join_overhead_seconds(tasks: Sequence[Mapping[str, Any]]) -> float:
    """Estimate Director's final ffmpeg re-encode, if this is a clip group."""

    if len(tasks) <= 1:
        return 0.0
    total_duration = 0.0
    largest_pixels = 832 * 480
    has_concat_group = False
    for task in tasks:
        params = task.get("params") if isinstance(task, Mapping) else None
        if not isinstance(params, Mapping):
            params = {}
        group = params.get("multi_clip_info")
        if isinstance(group, Mapping) and not group.get("defer_concat", False):
            has_concat_group = True
        frames = _positive_int(params.get("video_length", 1), 1)
        model_type = str(params.get("model_type") or "").lower()
        default_fps = 25.0 if model_type.startswith("ltx") else 24.0
        fps = _positive_float(params.get("fps"), default_fps)
        total_duration += frames / fps
        match = re.search(
            r"(\d+)\s*x\s*(\d+)",
            str(params.get("resolution") or "832x480").lower(),
        )
        if match:
            largest_pixels = max(
                largest_pixels,
                int(match.group(1)) * int(match.group(2)),
            )
    if not has_concat_group:
        return 0.0
    pixel_factor = (largest_pixels / float(1280 * 720)) ** 0.7
    # libx264's "fast" preset is generally several times real-time at 720p,
    # but CPU, storage, resolution, and audio muxing vary widely. Keep the
    # allowance conservative and deliberately coarse.
    return min(300.0, max(8.0, total_duration * 0.25 * pixel_factor))


def _weighted_mean(samples: Sequence[tuple[float, int]]) -> Optional[float]:
    valid = [(value, max(1, weight)) for value, weight in samples if value > 0]
    if not valid:
        return None
    total_weight = sum(weight for _, weight in valid)
    return sum(value * weight for value, weight in valid) / total_weight


def _adaptive_rate(samples: Sequence[tuple[float, int]]) -> Optional[float]:
    """Blend the whole-phase mean with an EWMA of the latest observations."""

    if not samples:
        return None
    mean = _weighted_mean(samples)
    ewma = None
    for value, weight in samples[-12:]:
        if value <= 0:
            continue
        # A callback can jump several steps when the consumer is busy. Give
        # that interval more authority without expanding it into N entries.
        alpha = min(0.72, 0.28 + 0.06 * max(1, weight))
        ewma = value if ewma is None else alpha * value + (1.0 - alpha) * ewma
    if mean is None:
        return ewma
    if ewma is None:
        return mean
    return 0.55 * mean + 0.45 * ewma


@dataclass
class _CompletedTask:
    workload: float
    seconds: float
    non_step_seconds: float


class AdaptiveGenerationEta:
    """Learn clip and whole-job ETAs from live sampler observations."""

    def __init__(
        self,
        tasks: Sequence[Mapping[str, Any]],
        *,
        now_fn=time.monotonic,
    ) -> None:
        self._now = now_fn
        self._workloads = [task_workload(task) for task in tasks] or [1.0]
        self._cache_start_percents = [
            task_cache_start_percent(task) for task in tasks
        ] or [None]
        self._join_overhead = _join_overhead_seconds(tasks)
        self._completed: list[Optional[_CompletedTask]] = [
            None for _ in self._workloads
        ]
        self._current_index: Optional[int] = None
        self._task_started_at: Optional[float] = None
        self._first_progress_at: Optional[float] = None
        self._phase_started_at: Optional[float] = None
        self._phase_identity: Optional[tuple[Any, ...]] = None
        self._last_step = 0
        self._last_step_at: Optional[float] = None
        self._total_steps = 0
        self._samples: list[tuple[float, int, bool]] = []
        self._finished_step_seconds = 0.0
        self._window_index = 1
        self._window_total = 1
        self._window_started_at: Optional[float] = None
        self._completed_window_seconds: list[float] = []
        self._last_snapshot: dict[str, Any] = {}

    @property
    def total_tasks(self) -> int:
        return len(self._workloads)

    def start_task(self, index: int, *, now: Optional[float] = None) -> None:
        now = self._now() if now is None else float(now)
        self._current_index = min(max(0, int(index)), self.total_tasks - 1)
        self._task_started_at = now
        self._first_progress_at = None
        self._phase_started_at = None
        self._phase_identity = None
        self._last_step = 0
        self._last_step_at = None
        self._total_steps = 0
        self._samples = []
        self._finished_step_seconds = 0.0
        self._window_index = 1
        self._window_total = 1
        self._window_started_at = None
        self._completed_window_seconds = []
        self._last_snapshot = self.snapshot(now=now)

    @staticmethod
    def _phase_name(message: str) -> str:
        value = str(message or "").lower()
        if "denois" in value:
            return "denoising"
        if "sampl" in value:
            return "sampling"
        if "diffusion" in value:
            return "diffusion"
        return re.sub(r"\s*\|.*$", "", value).strip() or "steps"

    def _finish_phase(self, now: float) -> None:
        if self._phase_started_at is None:
            return
        elapsed = max(0.0, now - self._phase_started_at)
        self._finished_step_seconds += elapsed

    def _finish_window(self, now: float) -> None:
        """Record one complete internal sliding window.

        A window can contain several sampler passes plus VAE/handoff work.
        Recording the whole wall-clock interval instead of individual phases
        lets later-window estimates learn the real pipeline cost, including
        LTX's multi-pass schedules and H3's continuation preparation.
        """

        if self._window_started_at is None:
            return
        elapsed = max(0.0, now - self._window_started_at)
        if elapsed > 0:
            self._completed_window_seconds.append(elapsed)
            self._completed_window_seconds = (
                self._completed_window_seconds[-12:]
            )

    def observe_progress(
        self,
        step: int,
        total_steps: int,
        message: str,
        *,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        now = self._now() if now is None else float(now)
        if self._current_index is None:
            self.start_task(0, now=now)
        step = max(0, int(step or 0))
        total_steps = max(0, int(total_steps or 0))
        window_match = _WINDOW_RE.search(str(message or ""))
        window_index = int(window_match.group(1)) if window_match else 1
        window_total = int(window_match.group(2)) if window_match else 1
        phase_identity = (
            window_index,
            window_total,
            self._phase_name(message),
            total_steps,
        )

        if self._phase_identity != phase_identity or step < self._last_step:
            if self._phase_identity is not None:
                self._finish_phase(now)
            if self._window_started_at is None:
                self._window_started_at = now
            elif window_index != self._window_index:
                self._finish_window(now)
                self._window_started_at = now
            self._phase_identity = phase_identity
            self._phase_started_at = now
            self._last_step = step
            self._last_step_at = now
            self._total_steps = total_steps
            self._samples = []
            self._window_index = max(1, window_index)
            self._window_total = max(self._window_index, window_total)
        else:
            if (
                self._last_step_at is not None
                and step > self._last_step
                and total_steps > 0
            ):
                delta_steps = step - self._last_step
                rate = max(0.001, (now - self._last_step_at) / delta_steps)
                warmup = self._cache_start_step(total_steps)
                is_post_cache = warmup is not None and step > warmup
                self._samples.append((rate, delta_steps, is_post_cache))
                self._samples = self._samples[-30:]
            self._last_step = step
            self._last_step_at = now
            self._total_steps = total_steps

        if self._first_progress_at is None:
            self._first_progress_at = now
        self._last_snapshot = self.snapshot(now=now)
        return dict(self._last_snapshot)

    def observe_status(
        self,
        message: str,
        *,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Refresh an estimate while encoding/decoding or saving."""

        now = self._now() if now is None else float(now)
        if self._current_index is None:
            self.start_task(0, now=now)
        window_match = _WINDOW_RE.search(str(message or ""))
        if window_match:
            window_index = max(1, int(window_match.group(1)))
            window_total = max(window_index, int(window_match.group(2)))
            if self._window_started_at is None:
                self._window_started_at = now
            elif window_index != self._window_index:
                if self._phase_started_at is not None:
                    self._finish_phase(now)
                self._finish_window(now)
                self._window_started_at = now
                self._phase_started_at = None
                self._phase_identity = None
                self._last_step = 0
                self._last_step_at = None
                self._total_steps = 0
                self._samples = []
            self._window_index = window_index
            self._window_total = window_total
        if (
            self._phase_started_at is not None
            and self._total_steps > 0
            and self._last_step >= self._total_steps
        ):
            # The first status after the final denoising callback marks the
            # beginning of VAE/save work. Closing the phase here lets a
            # completed clip teach subsequent clips their real non-step
            # overhead instead of pretending decode time was another step.
            self._finish_phase(now)
            self._phase_started_at = None
        self._last_snapshot = self.snapshot(now=now)
        return dict(self._last_snapshot)

    def _cache_start_step(self, total_steps: int) -> Optional[int]:
        if self._current_index is None:
            return None
        percent = self._cache_start_percents[self._current_index]
        if percent is None:
            return None
        return int(percent * max(0, total_steps) / 100.0)

    def _phase_remaining(self) -> Optional[float]:
        if self._total_steps <= 0 or self._last_step >= self._total_steps:
            return 0.0 if self._total_steps > 0 else None
        if not self._samples:
            return None

        warmup = self._cache_start_step(self._total_steps)
        pre = [(rate, weight) for rate, weight, post in self._samples if not post]
        post = [(rate, weight) for rate, weight, post in self._samples if post]
        all_samples = [(rate, weight) for rate, weight, _ in self._samples]
        if warmup is None:
            rate = _adaptive_rate(all_samples)
            return None if rate is None else rate * (self._total_steps - self._last_step)

        pre_rate = _adaptive_rate(pre) or _adaptive_rate(all_samples)
        post_rate = _adaptive_rate(post)
        if pre_rate is None:
            return None
        # Before cache hits have been observed, use a conservative reduction.
        # As soon as real post-warm-up samples arrive they fully replace this
        # prior, which is what lets the ETA follow different thresholds,
        # resolutions, GPUs, and Sol configurations without hard-coded tables.
        if post_rate is None:
            post_rate = pre_rate * 0.58
        if self._last_step < warmup:
            pre_steps = warmup - self._last_step
            post_steps = self._total_steps - warmup
            return pre_steps * pre_rate + post_steps * post_rate
        return (self._total_steps - self._last_step) * post_rate

    def _completed_rate(self) -> Optional[float]:
        rates = [
            item.seconds / item.workload
            for item in self._completed
            if item is not None and item.workload > 0
        ]
        return statistics.median(rates) if rates else None

    def _non_step_overhead(self) -> float:
        values = [
            item.non_step_seconds
            for item in self._completed
            if item is not None and item.non_step_seconds >= 0
        ]
        if values:
            return max(2.0, statistics.median(values))
        return 10.0

    def _predicted_task_seconds(
        self,
        index: int,
        *,
        now: Optional[float] = None,
    ) -> Optional[float]:
        completed = self._completed[index]
        if completed is not None:
            return completed.seconds
        rate = self._completed_rate()
        if rate is not None:
            return max(1.0, rate * self._workloads[index])
        if (
            self._current_index is not None
            and self._first_progress_at is not None
        ):
            phase_remaining = self._phase_remaining()
            if phase_remaining is not None:
                now = self._now() if now is None else float(now)
                current_total = max(
                    1.0,
                    now - self._first_progress_at
                    + phase_remaining
                    + self._non_step_overhead(),
                )
                current_work = self._workloads[self._current_index]
                return current_total * self._workloads[index] / current_work
        return None

    def snapshot(self, *, now: Optional[float] = None) -> dict[str, Any]:
        now = self._now() if now is None else float(now)
        if self._current_index is None:
            return {
                "current_clip": 0,
                "total_clips": self.total_tasks,
                "current_window": 0,
                "total_windows": 1,
                "window_eta_seconds": None,
                "clip_eta_seconds": None,
                "generation_eta_seconds": None,
                "project_eta_seconds": None,
                "eta_confidence": "calibrating",
                "eta_basis": "waiting-for-first-clip",
                "clip_estimates": [],
            }

        current = self._current_index
        elapsed_origin = (
            self._first_progress_at
            if self._first_progress_at is not None
            else self._task_started_at
            if self._task_started_at is not None
            else now
        )
        elapsed = max(
            0.0,
            now - elapsed_origin,
        )
        phase_remaining = self._phase_remaining()
        window_elapsed = max(
            0.0,
            now - (
                self._window_started_at
                if self._window_started_at is not None
                else elapsed_origin
            ),
        )
        live_window_remaining = (
            phase_remaining + self._non_step_overhead()
            if phase_remaining is not None else None
        )
        completed_window_total = (
            statistics.median(self._completed_window_seconds)
            if self._completed_window_seconds else None
        )
        history_window_remaining = (
            max(0.0, completed_window_total - window_elapsed)
            if completed_window_total is not None else None
        )
        sample_steps = sum(weight for _, weight, _ in self._samples)
        if (
            live_window_remaining is not None
            and history_window_remaining is not None
        ):
            live_weight = min(0.8, 0.35 + 0.05 * sample_steps)
            window_remaining = (
                live_weight * live_window_remaining
                + (1.0 - live_weight) * history_window_remaining
            )
        else:
            window_remaining = (
                live_window_remaining
                if live_window_remaining is not None
                else history_window_remaining
            )

        live_remaining: Optional[float] = None
        if window_remaining is not None:
            live_remaining = window_remaining
            if self._window_total > 1 and self._window_index < self._window_total:
                completed_window = (
                    completed_window_total
                    if completed_window_total is not None
                    else max(1.0, window_elapsed + window_remaining)
                )
                live_remaining += (
                    self._window_total - self._window_index
                ) * completed_window

        history_total = self._predicted_task_seconds(current, now=now)
        history_remaining = (
            max(0.0, history_total - elapsed)
            if history_total is not None
            else None
        )
        if live_remaining is not None and history_remaining is not None:
            live_weight = min(0.85, 0.35 + 0.05 * sample_steps)
            clip_remaining = (
                live_weight * live_remaining
                + (1.0 - live_weight) * history_remaining
            )
        else:
            clip_remaining = (
                live_remaining
                if live_remaining is not None
                else history_remaining
            )
        if self._completed[current] is not None:
            clip_remaining = 0.0

        clip_estimates: list[dict[str, Any]] = []
        future_seconds = 0.0
        all_future_known = True
        for index in range(self.total_tasks):
            completed = self._completed[index]
            predicted = self._predicted_task_seconds(index, now=now)
            status = (
                "completed" if completed is not None
                else "current" if index == current
                else "pending"
            )
            if index == current and clip_remaining is not None:
                predicted = elapsed + clip_remaining
            clip_estimates.append({
                "clip": index + 1,
                "status": status,
                "seconds": (
                    max(0, int(round(predicted)))
                    if predicted is not None else None
                ),
            })
            if index > current:
                if predicted is None:
                    all_future_known = False
                else:
                    future_seconds += predicted

        project_remaining = None
        if clip_remaining is not None and all_future_known:
            remaining_boundaries = max(0, self.total_tasks - current - 1)
            # Covers the small per-clip cleanup/continuation handoff and final
            # concat that live sampler timings do not include.
            orchestration_overhead = (
                4.0 * remaining_boundaries + self._join_overhead
            )
            project_remaining = (
                clip_remaining + future_seconds + orchestration_overhead
            )

        completed_count = sum(item is not None for item in self._completed)
        post_cache_steps = sum(
            weight for _, weight, post in self._samples if post
        )
        if sample_steps < 2 and completed_count == 0:
            confidence = "calibrating"
        elif completed_count >= 2 and (post_cache_steps >= 3 or self._cache_start_step(self._total_steps) is None):
            confidence = "high"
        elif completed_count >= 1 or sample_steps >= 4:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "current_clip": current + 1,
            "total_clips": self.total_tasks,
            "current_window": self._window_index,
            "total_windows": self._window_total,
            "window_eta_seconds": (
                max(0, int(round(window_remaining)))
                if window_remaining is not None else None
            ),
            "clip_eta_seconds": (
                max(0, int(round(clip_remaining)))
                if clip_remaining is not None else None
            ),
            # Generic alias used by Studio. For a one-task sliding-window
            # render this is the whole sequence; for a multi-task Studio job
            # it includes every remaining clip and the final join.
            "generation_eta_seconds": (
                max(0, int(round(project_remaining)))
                if project_remaining is not None else None
            ),
            "project_eta_seconds": (
                max(0, int(round(project_remaining)))
                if project_remaining is not None else None
            ),
            "eta_confidence": confidence,
            "eta_basis": (
                "live-cache-aware"
                if self._cache_start_step(self._total_steps) is not None
                else "live-adaptive"
            ),
            "clip_estimates": clip_estimates,
        }

    def complete_task(
        self,
        seconds: Optional[float],
        *,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        now = self._now() if now is None else float(now)
        if self._current_index is None:
            return self.snapshot(now=now)
        if self._phase_started_at is not None:
            self._finish_phase(now)
            self._phase_started_at = None
        fallback_start = (
            self._first_progress_at
            if self._first_progress_at is not None
            else self._task_started_at
            if self._task_started_at is not None
            else now
        )
        actual = _positive_float(seconds, max(0.0, now - fallback_start))
        non_step = max(0.0, actual - self._finished_step_seconds)
        self._completed[self._current_index] = _CompletedTask(
            workload=self._workloads[self._current_index],
            seconds=actual,
            non_step_seconds=non_step,
        )
        self._last_snapshot = self.snapshot(now=now)
        return dict(self._last_snapshot)
