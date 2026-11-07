"""Model-backend integration for Agent Access jobs and assets.

This module contains no model imports and never starts or stops an application
process. ``launch.py`` supplies the live job registry and native upload/prepare
callbacks; the controller remains responsible for its own network listener.
"""

from __future__ import annotations

import errno
import inspect
import json
import os
from pathlib import Path, PureWindowsPath
import re
import threading
import time
from typing import Any, Callable, Mapping
import uuid

from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile
from starlette.requests import Request as StarletteRequest

from .assets import AssetStore, AssetStoreError
from .context import principal_context
from .jobs import JobRegistry, JournalError, SubmissionJournal


_PRIVATE_DIRECTOR_KEYS = frozenset({
    "_director_pipeline_id",
    "_director_queue_entry_id",
    "_director_detached_operation",
    "_agent_access_owner_id",
    "_agent_access_session_id",
})
_ABSOLUTE_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:api_?key|token|access_token|refresh_token|password|secret(?:_?key)?|private_key|authorization|authorization_header|bearer_token)$",
    re.I,
)
_PATH_KEYS = frozenset({
    "path", "image", "image_path", "image_start", "image_end", "video_path", "audio_path",
    "input_path", "input_file", "source_path", "file_path", "output_path", "output_dir",
    "out_dir", "working_directory", "work_dir", "image_refs", "ref_images", "reference_images",
    "video_refs", "audio_refs", "control_video", "video_guide", "audio_guide", "audio_prompt",
    "voice_ref", "audio_reference", "reference_video", "reference_audio", "init_image",
    "model", "model_path", "checkpoint", "ckpt", "lora", "lora_path",
})
_APPROVED_MEDIA_ROOTS = (
    "uploads", "outputs", "outputs2", "storage", "ckpts", "loras", "loras_i2v", "temp",
)
_MAX_PREPARATION_BYTES = 8 * 1024 * 1024
_TERMINAL_STATUSES = frozenset({
    "cancelled", "canceled", "complete", "completed", "done", "error",
    "failed", "finished", "interrupted", "stopped", "success", "succeeded",
})
_PENDING_STATUSES = frozenset({"held", "queued", "running"})


def process_is_alive(pid: int) -> bool:
    """Return false only when the OS confirms that a PID does not exist."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("A verified process ID is required")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_INVALID_PARAMETER = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == ERROR_INVALID_PARAMETER:
                return False
            # Access denied and other failures do not prove that the process
            # exited, so startup must conservatively refuse takeover.
            return True
        try:
            exit_code = wintypes.DWORD()
            get_exit_code = kernel32.GetExitCodeProcess
            get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            get_exit_code.restype = wintypes.BOOL
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == STILL_ACTIVE
        finally:
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            close_handle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno == errno.ESRCH or getattr(error, "winerror", None) in {87, 1168}:
            return False
        # Permission and transient system errors are conservative: they do not
        # prove that the old backend is gone.
        return True


def is_admin(principal: Any) -> bool:
    return bool(principal and (getattr(principal, "owner", False) or "admin" in (getattr(principal, "scopes", ()) or ())))


def is_restricted_agent(principal: Any) -> bool:
    return bool(principal is not None and not is_admin(principal))


def strip_private_director_fields(value: Any) -> Any:
    """Copy JSON-like input while removing request-forgeable internal markers."""
    if isinstance(value, dict):
        return {
            key: strip_private_director_fields(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key.casefold() in _PRIVATE_DIRECTOR_KEYS)
        }
    if isinstance(value, list):
        return [strip_private_director_fields(item) for item in value]
    if isinstance(value, tuple):
        return [strip_private_director_fields(item) for item in value]
    return value


def _is_absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or bool(_ABSOLUTE_WINDOWS_PATH.match(value))


def _path_identity(value: str) -> str:
    try:
        return os.path.normcase(str(Path(value).resolve(strict=False)))
    except (OSError, RuntimeError, ValueError):
        return os.path.normcase(os.path.abspath(value))


def _approved_local_path(
    value: str,
    app_root: Path,
    additional_media_roots: tuple[Path, ...] = (),
) -> bool:
    if not _is_absolute_path(value):
        return True
    candidate = Path(value)
    if not candidate.is_absolute():
        # A Windows absolute path cannot be validated on a non-Windows host.
        return False
    try:
        resolved = candidate.resolve(strict=False)
        roots = [
            *((app_root / name).resolve(strict=False) for name in _APPROVED_MEDIA_ROOTS),
            *additional_media_roots,
        ]
    except (OSError, RuntimeError, ValueError):
        return False
    for root in roots:
        if resolved == root:
            return True
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _safe_preparation(
    value: Any,
    known_assets: Mapping[str, str],
    app_root: Path,
    *,
    additional_media_roots: tuple[Path, ...] = (),
    depth: int = 0,
    key: str = "",
    budget: list[int] | None = None,
) -> Any:
    """Keep round-trippable prepared values within the API response bound."""
    if budget is None:
        budget = [100_000]
    budget[0] -= 1
    if budget[0] < 0 or depth > 40:
        raise HTTPException(413, "Prepared settings exceed the configured complexity limit")
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            safe_key = str(key)
            if _SECRET_KEY.search(safe_key.replace("-", "_")):
                continue
            result[safe_key] = _safe_preparation(
                item, known_assets, app_root,
                additional_media_roots=additional_media_roots,
                depth=depth + 1, key=safe_key, budget=budget,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _safe_preparation(
                item, known_assets, app_root,
                additional_media_roots=additional_media_roots,
                depth=depth + 1, key=key, budget=budget,
            )
            for item in value
        ]
    if isinstance(value, str):
        normalized_key = key.casefold().replace("-", "_")
        is_path_value = (
            normalized_key in _PATH_KEYS
            or normalized_key.endswith(("_path", "_file", "_dir", "_directory"))
        )
        if is_path_value:
            if value in known_assets:
                return known_assets[value]
            path_identity = _path_identity(value)
            if path_identity in known_assets:
                return known_assets[path_identity]
        if is_path_value and _is_absolute_path(value) and not _approved_local_path(
            value, app_root, additional_media_roots,
        ):
            raise HTTPException(422, "Prepared settings reference a local file outside approved Maestro storage")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise HTTPException(422, "Prepared settings contain a value that cannot be returned as JSON")


def _asset_paths(payload: Any, store: AssetStore, principal_id: str) -> dict[str, str]:
    """Build an owner-scoped reverse map for paths resolved by the request boundary."""
    result: dict[str, str] = {}

    def add(path: str | os.PathLike[str], reference: str) -> None:
        path_text = os.fspath(path)
        result[path_text] = reference
        result[_path_identity(path_text)] = reference

    def visit(value: Any, key: str = "", depth: int = 0) -> None:
        if depth > 32:
            return
        if isinstance(value, str) and value.startswith(("asset:", "maestro-asset:")):
            add(store.resolve(principal_id, value), value)
        elif isinstance(value, str) and (
            key.casefold().replace("-", "_") in _PATH_KEYS
            or key.casefold().endswith(("_path", "_file", "_dir", "_directory"))
        ):
            reference_for_path = getattr(store, "reference_for_path", None)
            if callable(reference_for_path):
                reference = reference_for_path(principal_id, value)
                if reference:
                    add(value, reference)
        elif isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key), depth + 1)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key, depth + 1)

    visit(payload)
    return result


class AgentAccessBackend:
    """Durable ownership and lifecycle helpers for one model-backend process."""

    def __init__(
        self,
        runtime,
        app_root: str | os.PathLike[str],
        *,
        session_id: str | None = None,
        journal: SubmissionJournal | None = None,
        asset_store: AssetStore | None = None,
        additional_media_roots: Callable[[], Any] | None = None,
        pid_checker: Callable[[int], bool] = process_is_alive,
        clock: Callable[[], float] = time.time,
        recover_previous: bool = True,
    ):
        self.runtime = runtime
        self.app_root = Path(app_root).resolve()
        self.session_id = session_id or str(uuid.uuid4())
        self.journal = journal or SubmissionJournal(runtime.directory, clock=clock)
        self._pid_checker = pid_checker
        self._clock = clock
        if recover_previous:
            self.recover_previous_session()
        self.asset_store = asset_store or AssetStore(runtime.directory / "assets", self.app_root, clock=clock)
        self._additional_media_roots_callback = additional_media_roots
        self.jobs: JobRegistry | None = None
        self._snapshot_stop = threading.Event()
        self._snapshot_thread: threading.Thread | None = None
        self._snapshot_lock = threading.RLock()
        self._snapshot_signatures: dict[str, str] = {}
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False

    def additional_media_roots(self) -> tuple[Path, ...]:
        """Resolve trusted server-configured output roots for preparation checks."""
        callback = self._additional_media_roots_callback
        if callback is None:
            return ()
        try:
            configured = callback()
        except Exception as error:
            raise HTTPException(500, "Configured Maestro media roots are unavailable") from error
        if configured is None:
            return ()
        if isinstance(configured, (str, os.PathLike)):
            values = (configured,)
        elif isinstance(configured, (list, tuple, set, frozenset)):
            values = configured
        else:
            raise HTTPException(500, "Configured Maestro media roots are invalid")
        roots: list[Path] = []
        for value in values:
            if not isinstance(value, (str, os.PathLike)):
                raise HTTPException(500, "Configured Maestro media roots are invalid")
            if isinstance(value, str) and not value.strip():
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.app_root / path
            try:
                roots.append(path.resolve(strict=False))
            except (OSError, RuntimeError, ValueError) as error:
                raise HTTPException(500, "Configured Maestro media roots are invalid") from error
        return tuple(roots)

    def recover_previous_session(self) -> dict[str, Any] | None:
        """Mark old jobs interrupted only after the OS confirms the backend is dead."""
        previous = self.runtime.backend()
        if not previous:
            return None
        pid = previous.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeError("Agent Access has an unverifiable backend PID; refusing takeover")
        if self._pid_checker(pid):
            raise RuntimeError("Agent Access backend is already registered to a live process; refusing takeover")
        old_session = previous.get("session_id")
        if old_session:
            self.journal.mark_session_interrupted(str(old_session), verified_stopped=True)
        return previous

    def registry(self, recovered: Mapping[str, Any] | None = None) -> JobRegistry:
        """Create the passive registry with synchronous initial persistence."""
        from .context import current_principal_id

        self.jobs = JobRegistry(
            current_principal=current_principal_id,
            current_session=lambda: self.session_id,
            owner_resolver=self.resolve_background_job_owner,
            on_insert=self._persist_insert,
        )
        # Archive recovery is not a new submission. Install recovered rows
        # without invoking the initial-dispatch persistence callback, then
        # restore their immutable owner/session from the journal below.
        for job_id, value in (recovered or {}).items():
            dict.__setitem__(self.jobs, job_id, value)
            self.jobs._owners[job_id] = (None, self.session_id)
        for job_id in list(self.jobs.keys()):
            durable = self.journal.get_job_any(str(job_id))
            if durable is None:
                resolved = self.resolve_background_job_owner(job_id, self.jobs[job_id])
                if resolved and resolved.get("principal_id"):
                    self.jobs.restore_owner(job_id, resolved["principal_id"], resolved.get("session_id"))
                    self._persist_insert(job_id, self.jobs[job_id], resolved)
                continue
            saved_status = str((durable.get("snapshot") or {}).get("status") or "").casefold()
            if saved_status == "interrupted" and isinstance(self.jobs[job_id], dict):
                self.jobs[job_id]["status"] = "interrupted"
                self.jobs[job_id]["message"] = "Backend stopped before job completion"
                self.jobs[job_id].setdefault("completed_at", durable.get("updated_at"))
            self.jobs.restore_owner(job_id, durable["principal_id"], durable.get("session_id"))
        return self.jobs

    def _persist_insert(self, job_id: Any, job: Any, owner: dict[str, str | None]) -> None:
        principal_id = owner.get("principal_id")
        if not isinstance(principal_id, str) or not principal_id:
            return
        self.journal.record_job(job, principal_id, owner.get("session_id"))

    def resolve_background_job_owner(self, _job_id: Any, job: Any) -> dict[str, str | None] | None:
        """Resolve a child job only through a server-recorded pipeline or queue owner."""
        if principal_context.get() is not None or not isinstance(job, Mapping):
            return None
        params = job.get("params")
        if not isinstance(params, Mapping):
            return None
        pipeline_id = params.get("_director_pipeline_id")
        if isinstance(pipeline_id, str) and pipeline_id:
            owner = self.journal.resource_owner("pipeline", pipeline_id)
            if owner is not None:
                return {"principal_id": owner["principal_id"], "session_id": self.session_id}
        queue_id = params.get("_director_queue_entry_id")
        if isinstance(queue_id, str) and queue_id:
            owner = self.journal.resource_owner("queue", queue_id)
            if owner is not None:
                return {"principal_id": owner["principal_id"], "session_id": self.session_id}
        return None

    def bind_pipeline_owner(self, pipeline_id: str, principal_id: str | None) -> None:
        if principal_id:
            self.journal.register_resource_owner("pipeline", pipeline_id, principal_id, self.session_id)

    def bind_queue_owner(self, entry_id: str, principal_id: str | None) -> None:
        if principal_id:
            self.journal.register_resource_owner("queue", entry_id, principal_id, self.session_id)

    def inherit_pipeline_queue_owner(self, pipeline_id: str, queue_entry_id: Any) -> None:
        if not isinstance(queue_entry_id, str) or not queue_entry_id:
            return
        if self.journal.resource_owner("pipeline", pipeline_id) is not None:
            return
        owner = self.journal.resource_owner("queue", queue_entry_id)
        if owner is not None:
            self.journal.register_resource_owner(
                "pipeline", pipeline_id, owner["principal_id"], owner.get("session_id")
            )

    def resource_owner(self, resource_type: str, resource_id: str) -> dict[str, str | None] | None:
        return self.journal.resource_owner(resource_type, resource_id)

    def can_access_job(self, job_id: Any, principal: Any | None = None) -> bool:
        principal = principal if principal is not None else principal_context.get()
        if principal is None or is_admin(principal):
            return True
        principal_id = getattr(principal, "id", None)
        if not isinstance(principal_id, str):
            return False
        owner = self.jobs.ownership_for(job_id) if self.jobs is not None else None
        if owner and owner.get("principal_id") == principal_id:
            return True
        durable = self.journal.get_job_any(str(job_id))
        return bool(durable and durable.get("principal_id") == principal_id)

    def require_job_access(self, job_id: Any, principal: Any | None = None) -> None:
        principal = principal if principal is not None else principal_context.get()
        if is_restricted_agent(principal) and not self.can_access_job(job_id, principal):
            raise HTTPException(404, "Job not found")

    def can_access_resource(self, resource_type: str, resource_id: str, principal: Any | None = None) -> bool:
        principal = principal if principal is not None else principal_context.get()
        if principal is None or is_admin(principal):
            return True
        owner = self.journal.resource_owner(resource_type, resource_id)
        return bool(owner and owner.get("principal_id") == getattr(principal, "id", None))

    def require_resource_access(self, resource_type: str, resource_id: str, principal: Any | None = None) -> None:
        principal = principal if principal is not None else principal_context.get()
        if is_restricted_agent(principal) and not self.can_access_resource(resource_type, resource_id, principal):
            noun = "pipeline" if resource_type == "pipeline" else "queue entry"
            raise HTTPException(404, f"Director {noun} not found")

    def require_collection_access(self, resource_type: str, resource_ids, principal: Any | None = None) -> None:
        principal = principal if principal is not None else principal_context.get()
        if not is_restricted_agent(principal):
            return
        for resource_id in resource_ids:
            if not self.can_access_resource(resource_type, str(resource_id), principal):
                raise HTTPException(403, "This Director action also affects work owned by another client")

    def public_job(self, job_id: str, principal: Any | None = None) -> dict[str, Any] | None:
        principal = principal if principal is not None else principal_context.get()
        if principal is None:
            return None
        if is_admin(principal):
            record = self.journal.get_job_any(job_id)
        else:
            record = self.journal.get_job(job_id, getattr(principal, "id", ""))
        if record is None:
            return None
        snapshot = dict(record.get("snapshot") or {})
        for key in ("output_dir", "output_paths"):
            snapshot.pop(key, None)
        return {
            "job_id": record["job_id"],
            "job": snapshot,
            "updated_at": record["updated_at"],
        }

    def snapshot_job(self, job_id: Any) -> dict[str, Any] | None:
        if self.jobs is None:
            return None
        return self.jobs.record_snapshot(self.journal, job_id)

    def snapshot_all(self) -> int:
        if self.jobs is None:
            return 0
        count = 0
        for job_id in list(self.jobs.keys()):
            owner = self.jobs.ownership_for(job_id) or {}
            if not owner.get("principal_id"):
                continue
            try:
                snapshot = SubmissionJournal._job_snapshot(self.jobs[job_id], str(job_id))
                signature = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError, OverflowError):
                continue
            if self._snapshot_signatures.get(str(job_id)) == signature:
                continue
            try:
                self.journal.record_job(self.jobs[job_id], owner["principal_id"], owner.get("session_id"))
            except JournalError:
                continue
            self._snapshot_signatures[str(job_id)] = signature
            count += 1
        return count

    def start_snapshotter(self, *, interval: float = 1.0) -> None:
        if self.jobs is None:
            raise RuntimeError("Job registry must be installed before snapshots start")
        if self._snapshot_thread is not None and self._snapshot_thread.is_alive():
            return
        self._snapshot_stop.clear()

        def run() -> None:
            while not self._snapshot_stop.wait(interval):
                try:
                    self._bind_background_jobs()
                    self.snapshot_all()
                except Exception:
                    # Passive persistence must never change generation outcome.
                    continue

        self._snapshot_thread = threading.Thread(
            target=run, name="maestro-agent-job-snapshots", daemon=True,
        )
        self._snapshot_thread.start()

    def _bind_background_jobs(self) -> None:
        if self.jobs is None:
            return
        for job_id, job in list(self.jobs.items()):
            owner = self.jobs.ownership_for(job_id) or {}
            if owner.get("principal_id"):
                continue
            resolved = self.resolve_background_job_owner(job_id, job)
            if resolved and resolved.get("principal_id"):
                try:
                    self.jobs.bind_owner(job_id, resolved["principal_id"], resolved.get("session_id"))
                except JournalError:
                    continue

    def shutdown(self, backend_url: str, *, launcher_script: str) -> None:
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        self._snapshot_stop.set()
        worker = self._snapshot_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        try:
            self._bind_background_jobs()
            self.snapshot_all()
            self.runtime.register_backend(
                backend_url, ready=False, session_id=self.session_id,
                launcher_script=launcher_script,
            )
        except Exception:
            # Shutdown remains passive if settings are temporarily unavailable.
            pass

    @staticmethod
    def idle_evidence(
        *,
        jobs: Mapping[str, Any],
        engine_lock: Any,
        active_states: Any,
        director_pipelines: list[Mapping[str, Any]],
        director_queue: Mapping[str, Any],
        active_pipeline_statuses: tuple[str, ...] = ("queued", "planning", "running", "paused"),
    ) -> dict[str, Any]:
        """Produce conservative, read-only evidence for an idle engine."""
        if engine_lock is None or not callable(getattr(engine_lock, "locked", None)):
            return {"idle": False, "certain": False, "reason": "Generation lock state is unavailable"}
        if not isinstance(active_states, Mapping):
            return {"idle": False, "certain": False, "reason": "Active generation state is unavailable"}
        if not isinstance(jobs, Mapping) or not isinstance(director_queue, Mapping) or not isinstance(director_pipelines, list):
            return {"idle": False, "certain": False, "reason": "Queue or Director state is unavailable"}
        busy_reasons: list[str] = []
        try:
            if engine_lock.locked():
                busy_reasons.append("The generation engine is locked")
            if active_states:
                busy_reasons.append("A generation task is active")
            for job in jobs.values():
                if not isinstance(job, Mapping):
                    return {"idle": False, "certain": False, "reason": "Generation job state is malformed"}
                status = str(job.get("status") or job.get("state") or "").casefold()
                if status in _PENDING_STATUSES:
                    busy_reasons.append("A generation job is held, queued, or running")
                    break
                if status not in _TERMINAL_STATUSES:
                    return {"idle": False, "certain": False, "reason": "Generation job state is unrecognized"}
            for pipeline in director_pipelines:
                if not isinstance(pipeline, Mapping):
                    return {"idle": False, "certain": False, "reason": "Director pipeline state is malformed"}
                status = str(pipeline.get("status") or "").casefold()
                if status in active_pipeline_statuses:
                    busy_reasons.append("A Director pipeline is active or paused")
                    break
                if status not in _TERMINAL_STATUSES and status != "crashed":
                    return {"idle": False, "certain": False, "reason": "Director pipeline state is unrecognized"}
            entries = director_queue.get("entries")
            if not isinstance(entries, list):
                return {"idle": False, "certain": False, "reason": "Director queue state is unavailable"}
            for entry in entries:
                if not isinstance(entry, Mapping):
                    return {"idle": False, "certain": False, "reason": "Director queue entry is malformed"}
                status = str(entry.get("status") or "").casefold()
                if status in _PENDING_STATUSES:
                    busy_reasons.append("A Director project is held, queued, or running")
                    break
                if status not in _TERMINAL_STATUSES:
                    return {"idle": False, "certain": False, "reason": "Director queue state is unrecognized"}
            if director_queue.get("running") not in (None, False, True):
                return {"idle": False, "certain": False, "reason": "Director queue state is malformed"}
            if director_queue.get("running") is True and not busy_reasons:
                busy_reasons.append("Director queue is running")
        except Exception:
            return {"idle": False, "certain": False, "reason": "Backend busy state could not be verified"}
        if busy_reasons:
            return {"idle": False, "certain": True, "reason": busy_reasons[0]}
        return {"idle": True, "certain": True, "reason": "No active or queued generation work"}


def create_router(
    backend: AgentAccessBackend,
    *,
    prepare_generation: Callable[..., Any],
    native_upload: Callable[..., Any],
    idle_probe: Callable[[], Mapping[str, Any]],
    creative=None,
    native_director=None,
    production_callbacks=None,
) -> APIRouter:
    """Create the model backend's narrow Agent Access API surface."""
    router = APIRouter()
    from .http import management_router
    production = None
    if production_callbacks is not None:
        from .production_routes import ProductionService, create_production_router
        production = ProductionService(backend, **production_callbacks)
        router.include_router(create_production_router(production))
    if creative is not None:
        from .creative_routes import create_creative_router
        router.include_router(create_creative_router(
            backend, creative, native_prepare=prepare_generation, native_director=native_director,
            production=production,
        ))

    async def request_console_code():
        # Print only in the independent controller terminal, outside model-job
        # stdout capture. Verify its pinned installation certificate on loopback.
        import ssl
        import httpx
        cert = backend.runtime.directory / "tls" / "server-cert.pem"
        if not cert.is_file():
            raise RuntimeError("Start Agent Access from Pinokio first")
        try:
            context = ssl.create_default_context(cafile=str(cert))
            async with httpx.AsyncClient(verify=context, trust_env=False, follow_redirects=False, timeout=5) as client:
                response = await client.post(
                    backend.runtime.local_url + "/api/v1/agent-access/request-console-code",
                    headers={"Origin": backend.runtime.local_url},
                )
                if response.status_code != 200:
                    raise RuntimeError("The Agent Access terminal is not ready")
        except (OSError, httpx.HTTPError) as error:
            raise RuntimeError("Open Agent Access in Pinokio before requesting a code") from error

    router.include_router(management_router(
        backend.runtime,
        request_console_code=request_console_code,
        local_device_access=True,
    ))

    def principal_required():
        principal = principal_context.get()
        if principal is None or not isinstance(getattr(principal, "id", None), str):
            raise HTTPException(401, "Pair this agent before using Maestro")
        return principal

    @router.get("/api/v1/agent-access/jobs/{job_id}")
    def get_agent_access_job(job_id: str):
        principal = principal_required()
        record = backend.public_job(job_id, principal)
        if record is None:
            raise HTTPException(404, "Job not found")
        return record

    @router.get("/api/v1/agent-access/submissions")
    def get_agent_access_submission(request_id: str):
        principal = principal_required()
        try:
            record = backend.journal.find_submission(principal.id, request_id)
        except JournalError as error:
            raise HTTPException(error.status, error.message) from error
        if record is None:
            raise HTTPException(404, "Submission not found")
        return record

    @router.post("/api/v1/agent-access/prepare")
    async def prepare_agent_access_generation(request: Request):
        principal = principal_required()
        try:
            body = await request.json()
        except Exception as error:
            raise HTTPException(400, "Use a JSON object for generation preparation") from error
        if not isinstance(body, dict):
            raise HTTPException(400, "Use a JSON object for generation preparation")
        body = strip_private_director_fields(body)
        try:
            reverse_assets = _asset_paths(body, backend.asset_store, principal.id)
            resolved = backend.asset_store.resolve_tree(principal.id, body)
            prepared = prepare_generation(resolved, prepare_only=True)
            if inspect.isawaitable(prepared):
                prepared = await prepared
            safe_prepared = _safe_preparation(
                prepared, reverse_assets, backend.app_root,
                additional_media_roots=backend.additional_media_roots(),
            )
            try:
                encoded = json.dumps(
                    safe_prepared, ensure_ascii=False, allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
                raise HTTPException(422, "Prepared settings are not valid JSON") from error
            if len(encoded) > _MAX_PREPARATION_BYTES:
                raise HTTPException(413, "Prepared settings exceed the configured response limit")
            return safe_prepared
        except (AssetStoreError, JournalError) as error:
            raise HTTPException(error.status, error.message) from error
        except HTTPException:
            raise
        except ValueError as error:
            raise HTTPException(400, str(error)[:1000]) from error

    @router.post("/api/v1/agent-access/assets/uploads/{upload_id}/import")
    async def import_agent_access_asset(upload_id: str):
        principal = principal_required()
        try:
            record, staging_path = backend.asset_store.finish(principal.id, upload_id)
            if not isinstance(record, Mapping) or not isinstance(record.get("filename"), str):
                raise HTTPException(500, "Verified upload metadata is unavailable")
            if not isinstance(staging_path, Path):
                staging_path = Path(staging_path)
            if staging_path.is_symlink() or not staging_path.is_file():
                raise HTTPException(500, "Verified upload staging file is unavailable")
            size = staging_path.stat().st_size
            request_scope = {
                "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                "method": "POST", "scheme": "http", "path": "/api/v1/upload",
                "raw_path": b"/api/v1/upload", "query_string": b"",
                "headers": [(b"content-length", str(size).encode("ascii"))],
                "server": ("127.0.0.1", 80), "client": ("127.0.0.1", 1),
            }
            request_for_upload = StarletteRequest(request_scope)
            with staging_path.open("rb") as stream:
                upload = UploadFile(file=stream, filename=record["filename"])
                try:
                    result = native_upload(request_for_upload, upload, reuse_identical=True)
                    if inspect.isawaitable(result):
                        result = await result
                finally:
                    await upload.close()
            if not isinstance(result, Mapping) or not isinstance(result.get("path"), str):
                raise HTTPException(500, "Native upload did not return a trusted asset path")
            reference = backend.asset_store.register(principal.id, upload_id, result["path"])
            return {"reference": reference}
        except (AssetStoreError, JournalError) as error:
            raise HTTPException(error.status, error.message) from error

    @router.get("/api/v1/agent-access/idle")
    async def agent_access_idle_probe():
        principal = principal_required()
        if not is_admin(principal):
            raise HTTPException(403, "This operation requires admin permission")
        try:
            result = idle_probe()
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, Mapping):
                raise ValueError("idle probe returned no evidence")
            idle, certain, reason = result.get("idle"), result.get("certain"), result.get("reason")
            if not isinstance(idle, bool) or not isinstance(certain, bool) or not isinstance(reason, str):
                raise ValueError("idle probe returned malformed evidence")
            return {"idle": idle and certain, "certain": certain, "reason": reason[:512]}
        except Exception:
            return {"idle": False, "certain": False, "reason": "Backend busy state could not be verified"}

    return router


__all__ = [
    "AgentAccessBackend",
    "create_router",
    "is_admin",
    "is_restricted_agent",
    "process_is_alive",
    "strip_private_director_fields",
]
