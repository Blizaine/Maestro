"""Thread-safe state transitions for Maestro's in-process generation jobs.

Jobs are mutable dictionaries shared by API handlers and background workers.
This module keeps terminal-state changes and abort-state registration atomic so
that cancellation cannot be lost to a late ``completed``/``failed`` write.
It deliberately has no dependency on ``launch.py`` or model code, which keeps
the race behavior testable without loading the generation engine.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_lifecycle_lock = threading.RLock()
_registrations: dict[
    str,
    tuple[
        MutableMapping[str, Any],
        MutableMapping[str, Any],
        Callable[[], None] | None,
    ],
] = {}
_terminal_listeners: set[Callable[[Mapping[str, Any], str], None]] = set()


@dataclass(frozen=True)
class CancelResult:
    """Result of a cancellation request."""

    changed: bool
    was_running: bool
    abort_signalled: bool


GENERATED_MEDIA_EXTENSIONS = frozenset({
    ".aac", ".flac", ".gif", ".jpeg", ".jpg", ".m4a", ".mkv", ".mov",
    ".mp3", ".mp4", ".ogg", ".png", ".wav", ".webm", ".webp",
})


def collect_job_outputs(
    gen: Mapping[str, Any],
    out_dir: str,
    before: set[str] | None = None,
    *,
    allow_legacy_fallback: bool = False,
) -> list[str]:
    """Return only files explicitly registered by this generation state.

    WGP records generated media in ``file_list`` and ``audio_file_list``.
    Those lists are job-local, unlike a directory before/after scan that can
    accidentally claim a concurrent pipeline operation's output.  A guarded
    one-file fallback remains for older non-Director generators that do not
    register their result.
    """
    output_root = os.path.normcase(os.path.realpath(os.path.abspath(out_dir)))
    owned: list[str] = []
    seen: set[str] = set()
    for list_name in ("artifact_list", "file_list", "audio_file_list"):
        values = gen.get(list_name) or []
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            if isinstance(value, tuple):
                value = value[0] if value else None
            if not isinstance(value, str) or not value:
                continue
            # WGP registers both bare filenames (``clip.jpg``) and paths that
            # are already rooted at a relative output directory
            # (``outputs/clip.jpg``).  Blindly joining every relative value to
            # ``out_dir`` turns the latter into ``outputs/outputs/clip.jpg``
            # and silently loses the generated artifact.  Try the two exact
            # interpretations, accepting only an existing direct child of the
            # resolved output root so the ownership boundary remains strict.
            candidates = [value] if os.path.isabs(value) else [
                value,
                os.path.join(out_dir, value),
            ]
            candidate = None
            checked: set[str] = set()
            for registered_path in candidates:
                resolved = os.path.realpath(os.path.abspath(registered_path))
                normalized = os.path.normcase(resolved)
                if normalized in checked:
                    continue
                checked.add(normalized)
                if (
                    os.path.normcase(os.path.dirname(resolved)) == output_root
                    and os.path.isfile(resolved)
                ):
                    candidate = resolved
                    break
            if candidate is None:
                continue
            filename = os.path.basename(candidate)
            if (
                filename.startswith("_continuation_")
                or filename in seen
            ):
                continue
            seen.add(filename)
            owned.append(filename)

    if owned or not allow_legacy_fallback:
        return owned

    # Legacy fallback: accept exactly one newly-created, non-temporary media
    # file.  Ambiguity is safer to report as no output than to claim another
    # operation's artifact and stamp it with this job's metadata.
    try:
        candidates = []
        for filename in sorted(set(os.listdir(out_dir)) - set(before or ())):
            extension = os.path.splitext(filename)[1].lower()
            if (
                extension not in GENERATED_MEDIA_EXTENSIONS
                or filename.startswith("_")
                or not os.path.isfile(os.path.join(out_dir, filename))
            ):
                continue
            candidates.append(filename)
    except OSError:
        return []
    return candidates if len(candidates) == 1 else []


def call_with_sticky_interrupt(
    abort_state: Mapping[str, Any],
    model: Any,
    callable_: Callable[..., Any],
    *args: Any,
    poll_interval: float = 0.02,
    **kwargs: Any,
) -> Any:
    """Run a model call while making a durable abort survive model resets.

    Several model wrappers clear ``_interrupt`` at the beginning of their
    ``generate`` method.  A cancellation can land just before that reset, so
    relay the durable job abort flag until the call exits.  The normal direct
    interrupt remains the fast path; this closes the reset race.
    """

    def _reassert_interrupt() -> None:
        try:
            setattr(model, "_interrupt", True)
        except Exception:
            pass

    if abort_state.get("abort", False):
        _reassert_interrupt()
        return None

    stopped = threading.Event()

    def _relay() -> None:
        while not stopped.wait(poll_interval):
            if abort_state.get("abort", False):
                _reassert_interrupt()

    relay = threading.Thread(
        target=_relay,
        daemon=True,
        name="maestro_abort_relay",
    )
    relay.start()
    try:
        # Close the window between the first check and relay startup.
        if abort_state.get("abort", False):
            _reassert_interrupt()
            return None
        return callable_(*args, **kwargs)
    finally:
        if abort_state.get("abort", False):
            _reassert_interrupt()
        stopped.set()
        relay.join(timeout=max(0.1, poll_interval * 2))


def is_cancel_requested(job: MutableMapping[str, Any]) -> bool:
    """Return whether cancellation is durable for ``job``."""
    with _lifecycle_lock:
        return bool(job.get("cancel_requested")) or job.get("status") == "cancelled"


def snapshot_job(job: MutableMapping[str, Any]) -> dict[str, Any]:
    """Return a consistent shallow snapshot for API polling."""
    with _lifecycle_lock:
        snapshot = dict(job)
        if isinstance(snapshot.get("output_files"), list):
            snapshot["output_files"] = list(snapshot["output_files"])
        if isinstance(snapshot.get("clip_output_files"), dict):
            snapshot["clip_output_files"] = dict(snapshot["clip_output_files"])
        return snapshot


def record_job_outputs(
    job: MutableMapping[str, Any],
    output_files: list[str],
    *,
    clip_output_files: Mapping[int | str, str] | None = None,
    join_output_file: str | None = None,
) -> list[str]:
    """Merge discovered outputs/artifact metadata without changing status."""
    with _lifecycle_lock:
        merged = list(job.get("output_files") or [])
        for filename in output_files:
            if filename not in merged:
                merged.append(filename)
        job["output_files"] = merged
        if clip_output_files:
            clip_outputs = dict(job.get("clip_output_files") or {})
            for index, filename in clip_output_files.items():
                try:
                    key = str(int(index))
                except (TypeError, ValueError):
                    continue
                if filename:
                    clip_outputs[key] = filename
            job["clip_output_files"] = clip_outputs
        if join_output_file:
            job["join_output_file"] = join_output_file
        return list(merged)


def try_start(job: MutableMapping[str, Any], **updates: Any) -> bool:
    """Atomically move a queued job to running unless it was cancelled."""
    if "status" in updates:
        raise ValueError("status must be changed through a lifecycle transition")
    with _lifecycle_lock:
        if is_cancel_requested(job):
            job["status"] = "cancelled"
            job["message"] = "Cancelled"
            return False
        if job.get("status") != "queued":
            return False
        job.update(updates)
        job["status"] = "running"
        return True


def release_held(job: MutableMapping[str, Any], **updates: Any) -> bool:
    """Atomically release a user-held job into the runnable queue.

    Studio's explicit queue action captures a complete request without
    starting a worker.  Only the queue dispatcher may perform this
    ``held`` -> ``queued`` transition, and cancellation always wins.
    """
    if "status" in updates:
        raise ValueError("status must be changed through a lifecycle transition")
    with _lifecycle_lock:
        if is_cancel_requested(job):
            job["status"] = "cancelled"
            job["message"] = "Cancelled"
            return False
        if job.get("status") != "held":
            return False
        job.update(updates)
        job["status"] = "queued"
        return True


def try_requeue(job: MutableMapping[str, Any], **updates: Any) -> bool:
    """Return a multi-phase job to queued unless cancellation won first."""
    if "status" in updates:
        raise ValueError("status must be changed through a lifecycle transition")
    with _lifecycle_lock:
        if is_cancel_requested(job):
            job["status"] = "cancelled"
            job["message"] = "Cancelled"
            return False
        if job.get("status") != "running":
            return False
        job.update(updates)
        job["status"] = "queued"
        return True


def update_job(job: MutableMapping[str, Any], **updates: Any) -> bool:
    """Update a live job without replacing a terminal/cancelled message."""
    if "status" in updates:
        raise ValueError("status must be changed through a lifecycle transition")
    with _lifecycle_lock:
        if is_cancel_requested(job) or job.get("status") != "running":
            return False
        job.update(updates)
        return True


def register_abort_state(
    job: MutableMapping[str, Any],
    job_id: str,
    active_states: MutableMapping[str, MutableMapping[str, Any]],
    state: MutableMapping[str, Any],
    *,
    interrupt_model: Callable[[], None] | None = None,
) -> bool:
    """Register a worker's abort dictionary unless cancellation won first.

    Dummy states for non-Wan tools still receive ``abort=True`` but have no
    interrupt callback. Callback ownership is tracked separately by state
    identity so an old worker cannot interrupt or unregister a newer phase.
    """
    with _lifecycle_lock:
        state.setdefault("abort", False)
        if is_cancel_requested(job) or job.get("status") != "running":
            state["abort"] = True
            return False
        active_states[job_id] = state
        _registrations[job_id] = (job, state, interrupt_model)
        return True


def unregister_abort_state(
    job_id: str,
    active_states: MutableMapping[str, MutableMapping[str, Any]],
    state: MutableMapping[str, Any] | None = None,
) -> None:
    """Remove only the abort state owned by the finishing worker."""
    with _lifecycle_lock:
        current = active_states.get(job_id)
        if current is not None and (state is None or current is state):
            active_states.pop(job_id, None)
        registration = _registrations.get(job_id)
        if registration is not None and (
            state is None or registration[1] is state
        ):
            _registrations.pop(job_id, None)


def request_cancel(
    job: MutableMapping[str, Any],
    *,
    job_id: str | None = None,
    active_states: MutableMapping[str, MutableMapping[str, Any]] | None = None,
) -> CancelResult:
    """Atomically request cancellation and signal the matching active state."""
    with _lifecycle_lock:
        status = job.get("status")
        if status in TERMINAL_STATUSES:
            return CancelResult(False, False, False)

        was_running = status == "running"
        job["cancel_requested"] = True
        job["message"] = "Cancelled"

        abort_signalled = False
        state = active_states.get(job_id) if active_states is not None and job_id else None
        registration = _registrations.get(job_id) if job_id else None
        if (
            was_running
            and state is not None
            and registration is not None
            and registration[0] is job
            and registration[1] is state
        ):
            state["abort"] = True
            abort_signalled = True
            if registration[2] is not None:
                try:
                    registration[2]()
                except Exception:
                    pass

        job["status"] = "cancelled"

        return CancelResult(True, was_running, abort_signalled)


def finish_job(
    job: MutableMapping[str, Any],
    status: str,
    **updates: Any,
) -> bool:
    """Publish a completed/failed result unless cancellation already won."""
    if status not in {"completed", "failed"}:
        raise ValueError(f"Invalid terminal job status: {status}")
    if "status" in updates:
        raise ValueError("status must be changed through a lifecycle transition")
    with _lifecycle_lock:
        if is_cancel_requested(job):
            job["status"] = "cancelled"
            job["message"] = "Cancelled"
            return False
        if job.get("status") != "running":
            return False
        job.update(updates)
        job["status"] = status
        snapshot = dict(job)
        listeners = tuple(_terminal_listeners)

    # Notification/logging listeners must never hold the lifecycle lock or
    # make a successful model result fail. They receive a snapshot so an
    # observer cannot mutate the canonical job after publication.
    for listener in listeners:
        try:
            listener(snapshot, status)
        except Exception:
            pass
    return True


def register_terminal_listener(
    listener: Callable[[Mapping[str, Any], str], None],
) -> None:
    """Subscribe to successfully-published completed/failed job states."""
    with _lifecycle_lock:
        _terminal_listeners.add(listener)


def unregister_terminal_listener(
    listener: Callable[[Mapping[str, Any], str], None],
) -> None:
    """Remove a terminal listener. Primarily useful for tests/embedders."""
    with _lifecycle_lock:
        _terminal_listeners.discard(listener)


def acquire_generation_slot(
    generation_lock: threading.Lock,
    job: MutableMapping[str, Any],
    *,
    poll_interval: float = 0.1,
) -> bool:
    """Wait for the GPU lock while allowing a queued cancellation to exit."""
    while not is_cancel_requested(job):
        if generation_lock.acquire(timeout=poll_interval):
            if is_cancel_requested(job):
                generation_lock.release()
                return False
            return True
    return False


@contextmanager
def generation_slot(
    generation_lock: threading.Lock,
    job: MutableMapping[str, Any],
    *,
    poll_interval: float = 0.1,
) -> Iterator[bool]:
    """Context manager form of :func:`acquire_generation_slot`."""
    acquired = acquire_generation_slot(
        generation_lock, job, poll_interval=poll_interval,
    )
    try:
        yield acquired
    finally:
        if acquired:
            generation_lock.release()
