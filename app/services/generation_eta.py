"""Adaptive ETA estimates for long, multi-clip generation jobs.

The sampler's console ETA assumes every denoising step costs the same amount.
That is a poor fit for First Block Cache: the warm-up steps run at full cost,
then later steps become cheaper as cache hits accumulate.  This module keeps
the estimator independent from FastAPI and model code so both the job worker
and its tests can feed it ordinary progress observations.
"""
from __future__ import annotations

import math
import hashlib
import json
import os
import re
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


_WINDOW_RE = re.compile(r"(?:sliding\s+)?window\s+(\d+)\s*/\s*(\d+)", re.I)
_HISTORY_SCHEMA_VERSION = 1
_DEFAULT_HISTORY_FILENAME = "generation_eta_history.sqlite3"


def _task_params(task: Mapping[str, Any]) -> Mapping[str, Any]:
    params = task.get("params") if isinstance(task, Mapping) else None
    return params if isinstance(params, Mapping) else {}


def _resolution_pixels(params: Mapping[str, Any]) -> int:
    resolution = str(params.get("resolution") or "832x480").lower()
    match = re.search(r"(\d+)\s*x\s*(\d+)", resolution)
    if not match:
        return 832 * 480
    return max(1, int(match.group(1)) * int(match.group(2)))


def _window_geometry(task: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Return ``(total_frames, window_frames, overlap, window_count)``.

    Sliding-window pipelines repeatedly render the native window size; they do
    not run attention across the full final sequence at once.  Treating a
    four-window sequence as one 1,300-frame context made duration scaling far
    too pessimistic and prevented a completed sequence from teaching a later
    run with a different number of windows.
    """

    params = _task_params(task)
    total_frames = _positive_int(
        params.get("video_length", task.get("video_length", 1)),
        1,
    )
    raw_window = params.get("sliding_window_size")
    try:
        window_frames = int(raw_window)
    except (TypeError, ValueError):
        window_frames = total_frames
    if window_frames <= 0:
        window_frames = total_frames
    window_frames = min(total_frames, window_frames)
    try:
        overlap = max(0, int(params.get("sliding_window_overlap") or 0))
    except (TypeError, ValueError):
        overlap = 0
    overlap = min(overlap, max(0, window_frames - 1))
    if total_frames <= window_frames:
        return total_frames, window_frames, overlap, 1
    stride = max(1, window_frames - overlap)
    window_count = 1 + int(math.ceil((total_frames - window_frames) / stride))
    return total_frames, window_frames, overlap, max(1, window_count)


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

    params = _task_params(task)
    frames, window_frames, _overlap, window_count = _window_geometry(task)
    steps = _positive_int(params.get("num_inference_steps", 1), 1)
    pixels = _resolution_pixels(params)
    pixel_factor = (pixels / float(832 * 480)) ** 0.85
    if window_count > 1 and window_frames < frames:
        frame_work = window_count * (window_frames ** 1.15)
    else:
        frame_work = frames ** 1.15
    return max(1.0, frame_work * steps * pixel_factor)


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


def _media_count(value: Any) -> int:
    if value in (None, "", False):
        return 0
    if isinstance(value, (list, tuple, set)):
        return len([item for item in value if item not in (None, "", False)])
    return 1


def _stable_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _TaskProfile:
    exact_key: str
    family_key: str
    workload: float
    window_count: int
    details: dict[str, Any]


def _task_profile(task: Mapping[str, Any]) -> _TaskProfile:
    params = _task_params(task)
    total_frames, window_frames, overlap, window_count = _window_geometry(task)
    try:
        window_discard = max(
            0,
            int(params.get("sliding_window_discard_last_frames") or 0),
        )
    except (TypeError, ValueError):
        window_discard = 0
    loras = params.get("activated_loras") or []
    if not isinstance(loras, (list, tuple)):
        loras = [loras]
    lora_names = sorted(
        os.path.basename(str(item)).lower()
        for item in loras
        if str(item or "").strip()
    )
    conditioning = {
        "start_images": _media_count(params.get("image_start")),
        "end_images": _media_count(params.get("image_end")),
        "video_guides": _media_count(params.get("video_guide")),
        "audio_guides": sum(
            _media_count(params.get(key))
            for key in (
                "audio_guide", "audio_guide2", "audio_guide3",
                "audio_guide4", "audio_guide5", "audio_guide6",
            )
        ),
        "omni_references": _media_count(
            params.get("minimax_h3_references")
        ),
    }
    optimization = {
        "attention": str(params.get("override_attention") or "default").lower(),
        "cache": str(params.get("skip_steps_cache_type") or "off").lower(),
        "cache_multiplier": str(params.get("skip_steps_multiplier") or ""),
        "cache_start": str(params.get("skip_steps_start_step_perc") or ""),
        "turbo": bool(params.get("minimax_h3_turbo_mode")),
        "turbo_preset": str(params.get("minimax_h3_turbo_preset") or ""),
        "loras": lora_names,
        "lora_weights": str(params.get("loras_multipliers") or ""),
        "pipeline": (
            "progressive" if params.get("progressive_pipeline")
            else "single" if params.get("single_stage_pipeline")
            else "standard"
        ),
        "memory_profile": str(
            params.get("profile") or params.get("mmgp_profile") or ""
        ),
        "vram_coefficient": str(
            params.get("vram_safety_coefficient") or ""
        ),
        "attention_mode": str(params.get("attention_mode") or ""),
        "compile": bool(
            params.get("compile") or params.get("compile_modules")
        ),
    }
    details = {
        "schema": _HISTORY_SCHEMA_VERSION,
        "model_type": str(
            params.get("model_type")
            or task.get("model_type")
            or "unknown"
        ).lower(),
        "resolution": str(params.get("resolution") or "832x480").lower(),
        "steps": _positive_int(params.get("num_inference_steps"), 1),
        "window_frames": window_frames,
        "window_overlap": overlap,
        "window_discard": window_discard,
        "text_encoder": str(params.get("minimax_h3_text_encoder") or ""),
        "transformer_quantization": str(
            params.get("transformer_quantization") or ""
        ),
        "conditioning": conditioning,
        "optimization": optimization,
    }
    family = {
        "schema": details["schema"],
        "model_type": details["model_type"],
        "resolution": details["resolution"],
        "text_encoder": details["text_encoder"],
        "transformer_quantization": details["transformer_quantization"],
        "conditioning": conditioning,
        "optimization": optimization,
    }
    return _TaskProfile(
        exact_key=_stable_digest(details),
        family_key=_stable_digest(family),
        workload=task_workload(task),
        window_count=window_count,
        details={
            **details,
            "total_frames": total_frames,
            "window_count": window_count,
        },
    )


def _hardware_history_key(hardware: Optional[Mapping[str, Any]]) -> str:
    hardware = hardware if isinstance(hardware, Mapping) else {}
    try:
        vram = round(float(hardware.get("gpu_vram_gb") or 0.0), 1)
    except (TypeError, ValueError):
        vram = 0.0
    try:
        ram = round(float(hardware.get("ram_gb") or 0.0) / 8.0) * 8
    except (TypeError, ValueError):
        ram = 0
    signature = {
        "gpu": re.sub(
            r"\s+",
            " ",
            str(hardware.get("gpu_name") or "unknown").strip().lower(),
        ),
        "vram_gb": vram,
        "capability": str(hardware.get("gpu_capability") or ""),
        "ram_gb_bucket": ram,
        "cpu_count": _positive_int(hardware.get("cpu_count"), 1),
    }
    return _stable_digest(signature)


@dataclass(frozen=True)
class HistoricalEtaEstimate:
    active_seconds: float
    wall_seconds: float
    window_seconds: Optional[float]
    non_step_seconds: Optional[float]
    sample_count: int
    match: str


class GenerationEtaHistory:
    """Small local timing database used to seed future render ETAs.

    The database contains performance metadata only; prompts and media paths
    are deliberately excluded.  SQLite gives us atomic cross-thread writes,
    remains easy to inspect or delete, and is available in Python's standard
    library on every supported platform.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        hardware: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if path is None:
            app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            path = os.path.join(app_root, "settings", _DEFAULT_HISTORY_FILENAME)
        self.path = os.path.abspath(path)
        self.hardware_key = _hardware_history_key(hardware)
        self._lock = threading.RLock()
        self._enabled = True
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._initialize()
        except (OSError, sqlite3.Error) as exc:
            self._enabled = False
            print(f"[ETA] Local timing history unavailable ({exc}).")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eta_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    hardware_key TEXT NOT NULL,
                    exact_key TEXT NOT NULL,
                    family_key TEXT NOT NULL,
                    workload REAL NOT NULL,
                    active_seconds REAL NOT NULL,
                    wall_seconds REAL NOT NULL,
                    non_step_seconds REAL,
                    window_seconds_json TEXT,
                    source_key TEXT,
                    profile_json TEXT NOT NULL,
                    UNIQUE(hardware_key, source_key)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS eta_samples_exact "
                "ON eta_samples(hardware_key, exact_key, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS eta_samples_family "
                "ON eta_samples(hardware_key, family_key, created_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eta_import_roots (
                    root TEXT PRIMARY KEY,
                    newest_mtime REAL NOT NULL,
                    scanned_at REAL NOT NULL
                )
                """
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(
        self,
        task: Mapping[str, Any],
        active_seconds: float,
        *,
        wall_seconds: Optional[float] = None,
        non_step_seconds: Optional[float] = None,
        window_seconds: Optional[Sequence[float]] = None,
        source_key: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> bool:
        if not self._enabled:
            return False
        active = _positive_float(active_seconds, 0.0)
        if active <= 0:
            return False
        wall = _positive_float(wall_seconds, active)
        wall = max(active, wall)
        valid_windows = [
            float(value)
            for value in (window_seconds or [])
            if _positive_float(value, 0.0) > 0
        ]
        profile = _task_profile(task)
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO eta_samples (
                        created_at, hardware_key, exact_key, family_key,
                        workload, active_seconds, wall_seconds,
                        non_step_seconds, window_seconds_json, source_key,
                        profile_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        float(created_at or time.time()),
                        self.hardware_key,
                        profile.exact_key,
                        profile.family_key,
                        profile.workload,
                        active,
                        wall,
                        (
                            max(0.0, float(non_step_seconds))
                            if non_step_seconds is not None else None
                        ),
                        json.dumps(valid_windows),
                        source_key,
                        json.dumps(profile.details, sort_keys=True),
                    ),
                )
                inserted = cursor.rowcount > 0
                if inserted:
                    connection.execute(
                        """
                        DELETE FROM eta_samples
                        WHERE hardware_key = ? AND exact_key = ? AND id NOT IN (
                            SELECT id FROM eta_samples
                            WHERE hardware_key = ? AND exact_key = ?
                            ORDER BY created_at DESC LIMIT 80
                        )
                        """,
                        (
                            self.hardware_key, profile.exact_key,
                            self.hardware_key, profile.exact_key,
                        ),
                    )
                    connection.execute(
                        """
                        DELETE FROM eta_samples WHERE id NOT IN (
                            SELECT id FROM eta_samples
                            ORDER BY created_at DESC LIMIT 3000
                        )
                        """
                    )
                return inserted
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            print(f"[ETA] Could not save local timing sample ({exc}).")
            return False

    @staticmethod
    def _decode_windows(value: Any) -> list[float]:
        try:
            parsed = json.loads(value or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [
            float(item)
            for item in parsed
            if _positive_float(item, 0.0) > 0
        ]

    def estimate(self, task: Mapping[str, Any]) -> Optional[HistoricalEtaEstimate]:
        if not self._enabled:
            return None
        profile = _task_profile(task)
        columns = (
            "active_seconds, wall_seconds, non_step_seconds, workload, "
            "window_seconds_json"
        )
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    f"SELECT {columns} FROM eta_samples "
                    "WHERE hardware_key = ? AND exact_key = ? "
                    "ORDER BY created_at DESC LIMIT 40",
                    (self.hardware_key, profile.exact_key),
                ).fetchall()
                match = "exact"
                if not rows:
                    rows = connection.execute(
                        f"SELECT {columns} FROM eta_samples "
                        "WHERE hardware_key = ? AND family_key = ? "
                        "ORDER BY created_at DESC LIMIT 40",
                        (self.hardware_key, profile.family_key),
                    ).fetchall()
                    match = "family"
        except (OSError, sqlite3.Error):
            return None
        if not rows:
            return None

        rates = [
            float(row[0]) / float(row[3])
            for row in rows
            if _positive_float(row[0], 0.0) > 0
            and _positive_float(row[3], 0.0) > 0
        ]
        if not rates:
            return None
        active = statistics.median(rates) * profile.workload
        history_windows: list[float] = []
        residuals: list[float] = []
        if match == "exact":
            for row in rows:
                decoded = self._decode_windows(row[4])
                history_windows.extend(decoded)
                if decoded:
                    residuals.append(max(0.0, float(row[0]) - sum(decoded)))
            if history_windows:
                median_window = statistics.median(history_windows)
                residual = statistics.median(residuals) if residuals else 0.0
                active = median_window * profile.window_count + residual
            else:
                median_window = None
        else:
            median_window = None

        setup_values = [
            max(0.0, float(row[1]) - float(row[0]))
            for row in rows
            if _positive_float(row[1], 0.0) > 0
            and _positive_float(row[0], 0.0) > 0
        ]
        non_step_values = [
            max(0.0, float(row[2]))
            for row in rows
            if row[2] is not None
        ]
        setup = statistics.median(setup_values) if setup_values else 0.0
        return HistoricalEtaEstimate(
            active_seconds=max(1.0, active),
            wall_seconds=max(1.0, active + setup),
            window_seconds=median_window,
            non_step_seconds=(
                statistics.median(non_step_values)
                if non_step_values else None
            ),
            sample_count=len(rows),
            match=match,
        )

    def bootstrap_from_sidecars(
        self,
        root: str,
        *,
        max_files: int = 300,
    ) -> int:
        """Import recent completed gallery timings once per output folder."""

        if not self._enabled or not root or not os.path.isdir(root):
            return 0
        normalized_root = os.path.normcase(os.path.abspath(root))
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT newest_mtime FROM eta_import_roots WHERE root = ?",
                    (normalized_root,),
                ).fetchone()
                previous_mtime = float(row[0]) if row else 0.0
        except (OSError, sqlite3.Error, TypeError, ValueError):
            previous_mtime = 0.0

        candidates: list[tuple[float, str]] = []
        newest_seen = previous_mtime
        try:
            for directory, _subdirs, filenames in os.walk(normalized_root):
                for filename in filenames:
                    if not filename.lower().endswith(".meta.json"):
                        continue
                    path = os.path.join(directory, filename)
                    try:
                        modified = os.path.getmtime(path)
                    except OSError:
                        continue
                    newest_seen = max(newest_seen, modified)
                    if modified + 1e-6 >= previous_mtime:
                        candidates.append((modified, path))
        except OSError:
            return 0
        candidates.sort(reverse=True)
        imported = 0
        for _modified, path in candidates[:max(1, int(max_files))]:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    sidecar = json.load(handle)
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            params = sidecar.get("params")
            if not isinstance(params, Mapping) or not params.get("model_type"):
                continue
            mode = str(
                sidecar.get("generation_mode")
                or params.get("generation_mode")
                or "video"
            ).lower()
            if mode != "video":
                continue
            timing = sidecar.get("multi_window_timing")
            windows: list[float] = []
            if isinstance(timing, Mapping):
                expected = _positive_int(timing.get("window_count"), 1)
                completed = _positive_int(timing.get("completed_windows"), 0)
                raw_windows = timing.get("window_generation_seconds") or []
                if not isinstance(raw_windows, list):
                    raw_windows = []
                windows = [
                    float(value)
                    for value in raw_windows
                    if _positive_float(value, 0.0) > 0
                ]
                # Intermediate per-window gallery files carry partial timing.
                # Only the final, complete sequence is a valid training sample.
                if expected > 1 and (
                    completed < expected or len(windows) < expected
                ):
                    continue
            active = _positive_float(
                (
                    timing.get("total_generation_seconds")
                    if isinstance(timing, Mapping) else None
                )
                or sidecar.get("generation_time"),
                0.0,
            )
            if active <= 0:
                continue
            sidecar_job_id = str(sidecar.get("job_id") or "").strip()
            clip_info = params.get("multi_clip_info")
            if sidecar_job_id:
                if isinstance(clip_info, Mapping):
                    try:
                        task_index = max(0, int(clip_info.get("index") or 0))
                    except (TypeError, ValueError):
                        task_index = 0
                else:
                    try:
                        task_index = max(
                            0,
                            int(sidecar.get("director_clip_index") or 0),
                        )
                    except (TypeError, ValueError):
                        task_index = 0
                # Matches the key written directly by the live job worker,
                # preventing the later sidecar scan from counting one render
                # twice.
                source_key = f"job:{sidecar_job_id}:task:{task_index}"
            else:
                source_key = "sidecar:" + hashlib.sha256(
                    os.path.normcase(os.path.abspath(path)).encode("utf-8")
                ).hexdigest()
            if self.record(
                {"params": dict(params)},
                active,
                wall_seconds=(
                    active
                    if isinstance(clip_info, Mapping)
                    else _positive_float(
                        sidecar.get("job_elapsed_time"), active
                    )
                ),
                window_seconds=windows,
                source_key=source_key,
                created_at=_positive_float(
                    sidecar.get("created_at"), os.path.getmtime(path)
                ),
            ):
                imported += 1

        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO eta_import_roots(root, newest_mtime, scanned_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(root) DO UPDATE SET
                        newest_mtime = excluded.newest_mtime,
                        scanned_at = excluded.scanned_at
                    """,
                    (normalized_root, newest_seen, time.time()),
                )
        except (OSError, sqlite3.Error):
            pass
        return imported


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
        history_store: Optional[GenerationEtaHistory] = None,
        now_fn=time.monotonic,
    ) -> None:
        self._now = now_fn
        self._tasks = list(tasks) or [{"params": {}}]
        self._task_profiles = [_task_profile(task) for task in self._tasks]
        self._workloads = [profile.workload for profile in self._task_profiles]
        self._cache_start_percents = [
            task_cache_start_percent(task) for task in self._tasks
        ]
        self._join_overhead = _join_overhead_seconds(self._tasks)
        self._history_store = history_store
        self._history_estimates = [
            history_store.estimate(task) if history_store is not None else None
            for task in self._tasks
        ]
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

    @property
    def history_sample_count(self) -> int:
        return max(
            (
                estimate.sample_count
                for estimate in self._history_estimates
                if estimate is not None
            ),
            default=0,
        )

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
        self._window_total = self._task_profiles[self._current_index].window_count
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
        if self._current_index is not None:
            prior = self._history_estimates[self._current_index]
            if prior is not None and prior.non_step_seconds is not None:
                values.append(prior.non_step_seconds)
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
        prior = self._history_estimates[index]
        if prior is not None:
            # Before the first sampler callback, include the model/setup tail
            # observed by earlier equivalent jobs. Once rendering begins that
            # cost has already elapsed, so the active-generation prior is the
            # correct remaining-time baseline.
            if index == self._current_index and self._first_progress_at is None:
                return prior.wall_seconds
            return prior.active_seconds
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
                "eta_history_samples": 0,
                "eta_history_match": None,
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
        observed_window_total = (
            statistics.median(self._completed_window_seconds)
            if self._completed_window_seconds else None
        )
        current_prior = self._history_estimates[current]
        prior_window_total = (
            current_prior.window_seconds
            if current_prior is not None else None
        )
        if observed_window_total is not None and prior_window_total is not None:
            observed_weight = min(
                0.9,
                0.45 + 0.18 * len(self._completed_window_seconds),
            )
            expected_window_total = (
                observed_weight * observed_window_total
                + (1.0 - observed_weight) * prior_window_total
            )
        else:
            expected_window_total = (
                observed_window_total
                if observed_window_total is not None
                else prior_window_total
            )
        expected_window_remaining = (
            max(0.0, expected_window_total - window_elapsed)
            if expected_window_total is not None else None
        )
        if (
            expected_window_total is not None
            and expected_window_remaining == 0.0
            and self._completed[current] is None
        ):
            # A decode/save phase can occasionally run past the learned
            # median. Keep the completion clock ahead of the current time
            # instead of displaying a stale timestamp in the past while the
            # next status callback supplies more evidence.
            expected_window_remaining = max(
                8.0,
                min(180.0, expected_window_total * 0.10),
            )
        sample_steps = sum(weight for _, weight, _ in self._samples)
        if (
            live_window_remaining is not None
            and expected_window_remaining is not None
        ):
            live_weight = min(0.8, 0.35 + 0.05 * sample_steps)
            window_remaining = (
                live_weight * live_window_remaining
                + (1.0 - live_weight) * expected_window_remaining
            )
        else:
            window_remaining = (
                live_window_remaining
                if live_window_remaining is not None
                else expected_window_remaining
            )

        live_remaining: Optional[float] = None
        if window_remaining is not None:
            live_remaining = window_remaining
            if self._window_total > 1 and self._window_index < self._window_total:
                completed_window = (
                    expected_window_total
                    if expected_window_total is not None
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
        if (
            self._first_progress_at is None
            and current_prior is not None
            and history_remaining is not None
        ):
            clip_remaining = history_remaining
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
        history_samples = current_prior.sample_count if current_prior else 0
        history_match = current_prior.match if current_prior else None
        post_cache_steps = sum(
            weight for _, weight, post in self._samples if post
        )
        if sample_steps < 2 and completed_count == 0:
            if history_match == "exact" and history_samples >= 3:
                confidence = "high"
            elif history_match == "exact" and history_samples >= 1:
                confidence = "medium"
            elif history_samples >= 1:
                confidence = "low"
            else:
                confidence = "calibrating"
        elif completed_count >= 2 and (post_cache_steps >= 3 or self._cache_start_step(self._total_steps) is None):
            confidence = "high"
        elif completed_count >= 1 or sample_steps >= 4 or history_samples >= 1:
            confidence = "medium"
        else:
            confidence = "low"

        cache_active = self._cache_start_step(self._total_steps) is not None
        if cache_active and sample_steps > 0:
            eta_basis = "live-cache-aware"
        elif history_samples > 0 and sample_steps > 0:
            eta_basis = "historical-adaptive"
        elif history_samples > 0:
            eta_basis = "historical"
        else:
            eta_basis = "live-adaptive"

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
            "eta_basis": eta_basis,
            "eta_history_samples": history_samples,
            "eta_history_match": history_match,
            "clip_estimates": clip_estimates,
        }

    def complete_task(
        self,
        seconds: Optional[float],
        *,
        wall_seconds: Optional[float] = None,
        window_seconds: Optional[Sequence[float]] = None,
        source_key: Optional[str] = None,
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
        if self._history_store is not None:
            self._history_store.record(
                self._tasks[self._current_index],
                actual,
                wall_seconds=wall_seconds,
                non_step_seconds=non_step,
                window_seconds=window_seconds,
                source_key=source_key,
            )
            self._history_estimates[self._current_index] = (
                self._history_store.estimate(
                    self._tasks[self._current_index]
                )
            )
        self._last_snapshot = self.snapshot(now=now)
        return dict(self._last_snapshot)
