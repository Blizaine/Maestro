"""Authenticated title, finishing and take-review operations for agents.

All writing remains with the connected agent. Inspection is CPU-only; a
creative review never starts a render. Retakes consume a durable explicit
budget before the native Director callback is invoked.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import io
import json
from pathlib import Path
import re
import threading
import uuid

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .backend import _approved_local_path
from .http import json_body, scope_required
from .state import atomic_json
from services.production_review import ReviewStoreError


async def awaited(value):
    return await value if inspect.isawaitable(value) else value


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProductionService:
    def __init__(self, backend, *, output_root, export_editor, rerun_director, director_snapshot, idle_probe=None):
        self.backend = backend
        self.output_root = output_root
        self.export_editor = export_editor
        self.rerun_director = rerun_director
        self.director_snapshot = director_snapshot
        self.idle_probe = idle_probe
        self._lock = threading.RLock()

    def owner_dir(self, principal):
        directory = self.backend.runtime.directory / "production" / hashlib.sha256(principal.id.encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def media(self, value, principal, workspace="default"):
        if isinstance(value, str) and value.startswith(("asset:", "maestro-asset:")):
            path = self.backend.asset_store.resolve(principal.id, value)
        elif isinstance(value, dict) and set(value) <= {"filename", "workspace"}:
            from services.editor_projects import workspace_directory
            name = value.get("filename")
            if not isinstance(name, str) or not name or any(c in name for c in "/\\:\x00") or name in {".", ".."}:
                raise ValueError("Use a media basename and workspace, or an uploaded asset ID.")
            path = Path(workspace_directory(str(self.output_root()), value.get("workspace") or workspace)) / name
        elif isinstance(value, str) and Path(value).is_absolute():
            path = Path(value)
        else:
            raise ValueError("Media needs an uploaded asset ID or {filename, workspace}.")
        path = Path(path).resolve(strict=True)
        if not _approved_local_path(str(path), self.backend.app_root, self.backend.additional_media_roots()):
            raise ValueError("Media is outside approved Maestro storage.")
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Select an existing supported media file.")
        return path

    def workspace(self, name):
        from services.editor_projects import workspace_directory
        root = Path(self.output_root()).resolve()
        directory = Path(workspace_directory(str(root), name or "default"))
        if not directory.resolve().is_relative_to(root):
            raise ValueError("Workspace redirects outside Maestro output storage.")
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.resolve().is_relative_to(root):
            raise ValueError("Workspace redirects outside Maestro output storage.")
        return directory

    def create_title(self, params, principal):
        from services.title_cards import render_title_card
        identifier = uuid.uuid4().hex
        path = self.workspace(params.get("workspace")) / f"title-card-{identifier}.mp4"
        image = self.media(params["image_asset"], principal, params.get("workspace")) if params.get("image_asset") else None
        job_id = identifier[:8]
        snapshot = {"id": job_id, "kind": "title_card", "status": "running", "progress": 0,
                    "output_files": [], "error": None, "params": {"workspace": params.get("workspace") or "default"},
                    "message": "Rendering exact-text title card on CPU"}
        if self.backend.jobs is not None:
            self.backend.jobs[job_id] = snapshot
        try:
            result = render_title_card(params["card"], path, width=params["width"], height=params["height"], fps=params["fps"], image_path=image)
        except Exception as exc:
            snapshot.update(status="failed", message=str(exc)[:2000], error=str(exc)[:2000])
            if self.backend.jobs is not None:
                self.backend.snapshot_job(job_id)
            raise ValueError(f"Title-card rendering failed: {exc}") from exc
        snapshot.update(status="completed", progress=100, output_files=[str(path)], message="Exact-text title card ready")
        if self.backend.jobs is not None:
            self.backend.snapshot_job(job_id)
        return {"job_id": job_id, "status": "completed", "filename": path.name, "workspace": params.get("workspace") or "default", "duration_seconds": result.get("duration_seconds", result.get("duration")), "card": result.get("card", params["card"]), "warnings": result.get("warnings", [])}

    async def finish(self, body, principal):
        from services.editor_projects import probe_media, save_editor_project
        from services.production_finishing import build_finish_project
        if type(body.get("export", False)) is not bool:
            raise ValueError("export must be a boolean; saving a project alone does not render it.")
        self.workspace(body.get("workspace") or "default")
        def resolve(value):
            path = self.media(value, principal, body.get("workspace") or "default")
            return {**probe_media(str(path)), "path": str(path), "workspace": body.get("workspace") or "default"}
        project = await run_in_threadpool(build_finish_project, body, resolve)
        saved = await run_in_threadpool(save_editor_project, str(self.output_root()), project["workspace"], project, expected_revision=0)
        result = {"project_id": saved["id"], "workspace": saved["workspace"], "revision": saved["revision"], "duration_seconds": max(item["start"] + item["duration"] for item in saved["tracks"][0]["items"]), "exported": False}
        if body.get("export"):
            result["export"] = await awaited(self.export_editor({"project_id": saved["id"], "workspace": saved["workspace"]}))
            result["exported"] = True
            result["job_id"] = (result["export"] or {}).get("job_id")
        return result

    def inspect_take(self, body, principal):
        from services.production_review import inspect_take
        from PIL import Image
        if set(body) - {"asset", "workspace", "expected_duration", "include_audio"} or type(body.get("include_audio", False)) is not bool:
            raise ValueError("Use asset, workspace, expected_duration and boolean include_audio.")
        path = self.media(body.get("asset"), principal, body.get("workspace") or "default")
        if path.stat().st_size > 1024 * 1024 * 1024:
            raise ValueError("Take inspection accepts media up to 1 GiB.")
        identifier = str(uuid.uuid4())
        directory = self.owner_dir(principal) / "reviews" / identifier
        directory.mkdir(parents=True, exist_ok=True)
        before = fingerprint(path)
        packet = inspect_take(path, directory, expected_duration=body.get("expected_duration"))
        if fingerprint(path) != before:
            raise ValueError("The source changed during inspection; inspect its current revision.")
        atomic_json(directory / "source.json", {"path": str(path), "fingerprint": before, "packet": packet})
        artifacts = packet.get("artifacts") or {}
        # Keep local/cache paths private. The MCP tool emits actual media blocks.
        result = {key: value for key, value in packet.items() if key not in {"artifacts", "source", "source_path", "media_path", "output_dir"}}
        result.update(review_id=identifier, source_fingerprint=before, creative_status="not_assessed")
        sheet = artifacts.get("contact_sheet")
        if isinstance(sheet, dict):
            sheet = sheet.get("path")
        if sheet and Path(sheet).is_file():
            with Image.open(sheet) as image:
                image.thumbnail((1600, 1200))
                data = io.BytesIO()
                image.convert("RGB").save(data, format="JPEG", quality=82)
            if data.tell() <= 1024 * 1024:
                result["contact_sheet_base64"] = base64.b64encode(data.getvalue()).decode("ascii")
        audio = artifacts.get("audio_preview")
        if isinstance(audio, dict):
            audio = audio.get("path")
        if body.get("include_audio") is True and audio and Path(audio).is_file() and Path(audio).stat().st_size <= 1024 * 1024:
            result["audio_preview_base64"] = base64.b64encode(Path(audio).read_bytes()).decode("ascii")
        return result

    def review_source(self, review_id, principal):
        identifier = str(uuid.UUID(str(review_id)))
        path = self.owner_dir(principal) / "reviews" / identifier / "source.json"
        if not path.is_file():
            raise HTTPException(404, "Take review not found for this agent")
        source = json.loads(path.read_text(encoding="utf-8"))
        self.media(source["path"], principal)
        if fingerprint(source["path"]) != source["fingerprint"]:
            raise HTTPException(409, "Take changed since inspection; inspect it again")
        return source, path.parent

    def record_review(self, body, principal):
        from services.production_review import normalize_take_review
        source, directory = self.review_source(body.get("review_id"), principal)
        review = normalize_take_review(body.get("review"))
        atomic_json(directory / "creative-review.json", {"source_fingerprint": source["fingerprint"], "review": review})
        return {"review_id": str(body["review_id"]), "source_fingerprint": source["fingerprint"], "review": review, "render_submitted": False}

    def budget(self, principal, budget_id, limits=None):
        from services.production_review import PolishBudgetReviewStore
        directory = self.owner_dir(principal) / "budgets" / str(uuid.UUID(str(budget_id)))
        if limits is None:
            config = directory / "limits.json"
            if not config.is_file():
                raise HTTPException(404, "Polish budget not found for this agent")
            limits = json.loads(config.read_text(encoding="utf-8"))
        else:
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / "limits.json", limits)
        return PolishBudgetReviewStore(directory, **limits)

    async def polish(self, body, principal):
        from services.production_review import director_take_signature
        source, directory = self.review_source(body.get("review_id"), principal)
        review_path = directory / "creative-review.json"
        if not review_path.is_file():
            raise ValueError("Record concrete creative observations before requesting a retake.")
        if json.loads(review_path.read_text(encoding="utf-8"))["review"]["verdict"] == "keep":
            raise ValueError("A kept take has no approved retake request.")
        pid = body.get("pipeline_id")
        index = body.get("clip_index")
        if not isinstance(pid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", pid) or type(index) is not int or index < 0:
            raise ValueError("Use pipeline_id and a zero-based clip_index.")
        snapshot = await awaited(self.director_snapshot(pid))
        clips = snapshot.get("clips") or []
        if index >= len(clips):
            raise ValueError("The selected Director clip no longer exists.")
        clip = clips[index]
        if clip.get("video_stale"):
            raise HTTPException(409, "The selected clip was edited after its render; inspect and review its current plan first")
        selected = clip.get("video_path") or clip.get("video_file") or clip.get("output_path")
        if not selected or Path(selected).resolve() != Path(source["path"]).resolve():
            raise HTTPException(409, "The selected Director take changed or does not match the inspected media")
        prompt = body.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 100000):
            raise ValueError("A revised prompt must be nonempty text up to 100,000 characters.")
        store = self.budget(principal, body.get("budget_id"))
        request = {"pipeline_id": pid, "clip_index": index, "prompt": prompt, "review_id": body["review_id"]}
        request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if self.idle_probe is not None:
            idle = await awaited(self.idle_probe())
            if idle.get("idle") is not True or idle.get("certain") is not True:
                raise HTTPException(409, "Maestro has active or uncertain work; defer polishing without consuming a new attempt")
        # Never retry a claimed request automatically. The controller also
        # journals request IDs; an uncertain receipt requires reconciliation.
        with self._lock:
            intent, created = store.claim_once(body.get("request_id"), f"{pid}:{index}", source["fingerprint"], request_hash)
            if not created:
                return {"intent": intent, "submitted_again": False, "next": "Inspect the original Director pipeline/receipt; do not resubmit an uncertain request."}
        result = await awaited(self.rerun_director(pid, index, {"prompt": prompt, "_expected_take": {
            "signature": director_take_signature(snapshot, index), "path": source["path"], "fingerprint": source["fingerprint"],
        }}))
        receipt = str((result or {}).get("job_id") or (result or {}).get("pipeline_id") or pid)
        store.bind_receipt(body["request_id"], receipt)
        return {"intent": store.get(body["request_id"]), "result": result,
                "job_id": (result or {}).get("job_id"), "submitted_again": False}


def create_production_router(service):
    router = APIRouter(prefix="/api/v1/agent-access/production")

    def error(exc):
        if isinstance(exc, HTTPException):
            return exc
        return HTTPException(409 if isinstance(exc, ReviewStoreError) else 422, str(exc)[:2000])

    @router.post("/finish")
    async def finish(request: Request):
        principal = scope_required("manage")
        try:
            return await service.finish(await json_body(request), principal)
        except (ValueError, TypeError, OSError, HTTPException, ReviewStoreError) as exc:
            raise error(exc) from exc

    @router.post("/inspect")
    async def inspect_take(request: Request):
        principal = scope_required("observe")
        try:
            return await run_in_threadpool(service.inspect_take, await json_body(request), principal)
        except (ValueError, TypeError, OSError, HTTPException, ReviewStoreError) as exc:
            raise error(exc) from exc

    @router.post("/reviews")
    async def review(request: Request):
        principal = scope_required("manage")
        try:
            return await run_in_threadpool(service.record_review, await json_body(request), principal)
        except (ValueError, TypeError, OSError, HTTPException, ReviewStoreError) as exc:
            raise error(exc) from exc

    @router.post("/budgets")
    async def budget(request: Request):
        principal = scope_required("generate")
        body = await json_body(request)
        try:
            if set(body) != {"max_renders", "max_retakes_per_clip"}:
                raise ValueError("Explicit max_renders and max_retakes_per_clip are required.")
            identifier = str(uuid.uuid4())
            store = service.budget(principal, identifier, body)
            return {"budget_id": identifier, "budget": store.snapshot()}
        except (ValueError, TypeError, OSError, HTTPException, ReviewStoreError) as exc:
            raise error(exc) from exc

    @router.get("/budgets/{budget_id}")
    def budget_status(budget_id: str):
        principal = scope_required("observe")
        try:
            return service.budget(principal, budget_id).snapshot()
        except (ValueError, TypeError, OSError, HTTPException, ReviewStoreError) as exc:
            raise error(exc) from exc

    @router.post("/polish")
    async def polish(request: Request):
        principal = scope_required("generate")
        try:
            return await service.polish(await json_body(request), principal)
        except (ValueError, TypeError, OSError, HTTPException, ReviewStoreError) as exc:
            raise error(exc) from exc

    return router
