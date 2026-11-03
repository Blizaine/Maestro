"""Bridge the optional ComfyUI adapter into Maestro's existing owned job queue."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import secrets
import threading
import time
import uuid

from .comfyui_runtime import ComfyService, ServiceError, _read_json
from .job_lifecycle import request_cancel


class ComfyJobAdapter:
    def __init__(
        self, app_root, *, jobs, workspace_dir, start_worker, require_job_access,
        principal_id, link_folders, known_assets, resolve_media,
        finish_job, update_job, is_cancelled, unload_native, authorize_media=None,
        is_owner=None,
    ):
        self.jobs = jobs
        self.workspace_dir = workspace_dir
        self.start_worker = start_worker
        self.resolve_media = resolve_media
        self.finish_job = finish_job
        self.update_job = update_job
        self.is_cancelled = is_cancelled
        self.unload_native = unload_native
        self.lock = threading.RLock()
        self.active_monitors = set()
        self.service = ComfyService(
            app_root, submit=self.submit, require_job_access=require_job_access,
            principal_id=principal_id, link_folders=link_folders, known_assets=known_assets,
            authorize_media=authorize_media,
            is_owner=is_owner, resume=self.resume,
            job_exists=lambda job_id: job_id in self.jobs,
        )

    def submit(self, prepared: dict) -> dict:
        with self.lock:
            job_id = uuid.uuid4().hex[:8]
            receipt, fresh = self.service.reserve(prepared, job_id)
            if not fresh:
                existing = self.jobs.get(receipt["job_id"])
                return {
                    "job_id": receipt["job_id"],
                    "status": existing.get("status") if existing else receipt["state"],
                    "client_submission_id": receipt["submission_id"],
                    "replayed": True,
                }
            workflow = prepared["workflow"]
            prompt = prepared["inputs"].get("prompt") or prepared["inputs"].get("positive_prompt") or workflow["name"]
            self.jobs[job_id] = {
                "id": job_id, "status": "queued", "progress": 0, "step": 0,
                "total_steps": 0, "phase": "", "message": "Queued · ComfyUI",
                "created_at": time.time(), "show_in_gallery": True,
                "client_submission_id": prepared["client_submission_id"],
                "workspace": prepared["workspace"], "out_dir": self.workspace_dir(prepared["workspace"]),
                "params": {
                    "model_type": "comfyui_workflow", "prompt": str(prompt),
                    "generation_mode": workflow["kind"], "generation_engine": "comfyui",
                    "comfyui_workflow_id": workflow["id"],
                    "comfyui_workflow_name": workflow["name"],
                    "_comfyui_submission": prepared["client_submission_id"],
                },
                "output_files": [], "error": None,
            }
            self.start_worker(job_id)
            return {"job_id": job_id, "status": "queued", "client_submission_id": prepared["client_submission_id"]}

    def capture_engine(self, body: dict) -> dict:
        explicit = body.get("generation_engine")
        if explicit not in {None, "native", "comfyui"}:
            raise ServiceError(422, "Generation engine must be native or comfyui.")
        config = self.service._config()
        route = config["routes"].get(str(body.get("model_type") or ""))
        if explicit == "native" or (explicit is None and not config["enabled"]) or not route or not route.get("enabled"):
            if explicit == "comfyui":
                raise ServiceError(409, "Configure a ComfyUI workflow mapping for this model first.")
            return {"engine": "native"}
        if not config["enabled"]:
            raise ServiceError(409, "Enable ComfyUI integration or choose the Native engine.")
        return {"engine": "comfyui", "route": deepcopy(route)}

    def resume(self, receipt: dict) -> dict:
        """Resume GET monitoring of a dispatched intent, never prompt submission."""
        with self.lock:
            job_id = receipt["job_id"]
            existing = self.jobs.get(job_id)
            if job_id in self.active_monitors or (existing and existing.get("status") in {"queued", "running"}):
                return {"job_id": job_id, "status": existing["status"], "replayed": True}
            prepared = receipt["prepared"]
            workflow = prepared["workflow"]
            out_dir = Path(receipt.get("out_dir") or self.workspace_dir(prepared["workspace"])).resolve()
            base = Path(self.workspace_dir("default")).resolve()
            if out_dir != base and base not in out_dir.parents:
                raise ServiceError(409, "The saved workflow output directory is outside the current output library; preserve it for review.")
            job = {
                "id": job_id, "status": "queued", "progress": 0, "step": 0,
                "total_steps": 0, "phase": "comfyui", "message": "Reconnecting to owned ComfyUI work",
                "created_at": receipt["created_at"], "show_in_gallery": True,
                "client_submission_id": receipt["submission_id"],
                "workspace": prepared["workspace"], "out_dir": str(out_dir),
                "params": {"model_type": "comfyui_workflow", "generation_mode": workflow["kind"],
                           "prompt": prepared["inputs"].get("prompt") or prepared["inputs"].get("positive_prompt") or workflow["name"],
                           "generation_engine": "comfyui", "comfyui_workflow_id": workflow["id"],
                           "comfyui_workflow_name": workflow["name"], "_comfyui_submission": receipt["submission_id"]},
                "output_files": [], "error": None, "comfyui_monitor_only": True,
                "comfyui_cancel_requested": bool(receipt.get("cancel_requested")),
            }
            if existing is None and hasattr(self.jobs, "restore_owner"):
                dict.__setitem__(self.jobs, job_id, job)
                if receipt.get("owner_id"):
                    self.jobs.restore_owner(job_id, receipt["owner_id"], None)
            else:
                self.jobs[job_id] = job
            self.start_worker(job_id)
            return {"job_id": job_id, "status": "queued", "client_submission_id": receipt["submission_id"]}

    def attach_native(self, body: dict, workspace: str, job_id: str, submission: str | None, *, engine_snapshot=None) -> dict | None:
        snapshot = engine_snapshot or self.capture_engine(body)
        if snapshot["engine"] == "native":
            body.pop("generation_engine", None)
            return None
        normalized = dict(body)
        normalized["generation_engine"] = "comfyui"
        if normalized.get("seed") in {None, -1, "-1"}:
            normalized["seed"] = secrets.randbits(32)
        selected = self.service.native_request(normalized, frozen_route=snapshot["route"])
        if selected is None:
            body.pop("generation_engine", None)
            return None
        body["seed"] = normalized["seed"]
        try:
            canonical = str(uuid.UUID(str(submission))) if submission else str(uuid.uuid4())
        except ValueError:
            canonical = str(uuid.uuid5(uuid.NAMESPACE_URL, "maestro-comfyui:" + str(submission)))
        prepared = self.service.prepare(
            selected["workflow_id"],
            {"client_submission_id": canonical, "inputs": selected["inputs"], "workspace": workspace},
            native_model_type=selected["native_model_type"],
        )
        prepared["native_route"] = selected["native_route"]
        ownership = self.jobs.ownership_for(job_id) if hasattr(self.jobs, "ownership_for") else None
        receipt, fresh = self.service.reserve(prepared, job_id, owner_id=(ownership or {}).get("principal_id"))
        if not fresh:
            return {"job_id": receipt["job_id"], "status": self.jobs.get(receipt["job_id"], {}).get("status", receipt["state"]), "client_submission_id": submission}
        body["_comfyui_submission"] = canonical
        body["generation_engine"] = "comfyui"
        body["comfyui_workflow_id"] = selected["workflow_id"]
        return None

    def execute(self, job: dict, *, finalize=True) -> bool:
        with self.lock:
            if job["id"] in self.active_monitors:
                raise ServiceError(409, "This workflow already has an active monitor.")
            self.active_monitors.add(job["id"])
        try:
            return self._execute(job, finalize=finalize)
        finally:
            with self.lock:
                self.active_monitors.discard(job["id"])

    def _execute(self, job: dict, *, finalize=True) -> bool:
        submission = job.get("params", {}).get("_comfyui_submission")
        prepared = self.service.prepared_submission(submission)
        receipt = self.service._receipt_path(submission)
        intent = _read_json(receipt, {})
        cancellation = bool(intent.get("cancel_requested") or job.get("comfyui_cancel_requested"))
        def cancelled():
            return cancellation or self.is_cancelled(job)
        if prepared.get("native_route") and not job.get("comfyui_monitor_only"):
            body = deepcopy(job["params"])
            selected = self.service.native_request(body, frozen_route=prepared["native_route"])
            if selected is None:
                raise ServiceError(409, "The frozen workflow mapping is unavailable.")
            prepared["inputs"] = selected["inputs"]
        if cancelled() and intent.get("state") == "prepared":
            self.service.cancel_prepared(submission)
            return False
        # The caller owns Maestro's normal generation slot. Release only its
        # idle native models, never another ComfyUI user's running workflow.
        self.unload_native()
        started = time.monotonic()
        files = self.service.execute(
            prepared, Path(job["out_dir"]),
            cancelled=cancelled,
            progress=lambda message: self.update_job(job, phase="comfyui", message=message),
            resolve_media=self.resolve_media,
        )
        if cancelled():
            self.service.discard_cancelled_outputs(submission, Path(job["out_dir"]))
            request_cancel(job)
            return False
        if not files:
            raise ServiceError(422, "ComfyUI did not return verified output media.")
        params = job["params"]
        params["comfyui_workflow_name"] = prepared["workflow"]["name"]
        params["generation_engine"] = "comfyui"
        job["generation_time"] = time.monotonic() - started
        if finalize:
            completed = bool(self.finish_job(
                job, "completed", output_files=files, progress=100, step=0,
                total_steps=0, phase="", message="Done · ComfyUI",
            ))
            if not completed and self.is_cancelled(job):
                self.service.discard_cancelled_outputs(submission, Path(job["out_dir"]))
                job["output_files"] = []
            return completed
        self.update_job(job, output_files=files, progress=100, phase="", message="ComfyUI output ready.")
        return True


def resolve_workflow_media(app_root, output_root, raw: str, workspace: str) -> Path:
    """Accept only media already imported into Maestro's managed media roots."""
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ServiceError(422, "Upload or select a Maestro media reference first.")
    app = Path(app_root).resolve()
    outputs = Path(output_root)
    if not outputs.is_absolute():
        outputs = app / outputs
    outputs = outputs.resolve()
    roots = [outputs]
    for relative in ("uploads", "settings/characters"):
        root = (app / relative).resolve()
        # Managed reference directories cannot delegate access to arbitrary
        # external folders through a root junction or symlink.
        if os.path.normcase(os.path.commonpath([str(app), str(root)])) == os.path.normcase(str(app)):
            roots.append(root)
    candidate = Path(raw)
    possibilities = [candidate] if candidate.is_absolute() else [
        app / candidate, outputs / workspace / candidate, outputs / candidate, app / "uploads" / candidate,
    ]
    for possibility in possibilities:
        try:
            resolved = possibility.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not resolved.is_file() or resolved.suffix.lower() not in {
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".mp4", ".mov",
            ".webm", ".mkv", ".wav", ".mp3", ".flac", ".ogg", ".m4a",
        }:
            continue
        for root in roots:
            try:
                if os.path.normcase(os.path.commonpath([str(root), str(resolved)])) == os.path.normcase(str(root)):
                    return resolved
            except ValueError:
                pass
    raise ServiceError(422, "The workflow reference is outside Maestro's approved media folders.")
