"""Native creative context, mechanical plan review and explicit submission."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .authoring_plans import AuthoringPlanError, compute_studio_timeline, normalize_authored_director_plan, prepare_authored_generation
from .assets import AssetStoreError
from .backend import _asset_paths, _safe_preparation, strip_private_director_fields
from .creative_context import CreativeError
from .http import json_body, owner_required, scope_required
from .state import atomic_json


async def _await(value):
    return await value if inspect.isawaitable(value) else value


def create_creative_router(backend, service, *, native_prepare, native_director) -> APIRouter:
    router = APIRouter()

    def safe(value, assets=None):
        return _safe_preparation(value, assets or {}, backend.app_root, additional_media_roots=backend.additional_media_roots())

    def draft_path(principal_id, draft_id):
        actor = hashlib.sha256(principal_id.encode()).hexdigest()
        return backend.runtime.directory / "authoring-drafts" / actor / (str(uuid.UUID(draft_id)) + ".json")

    def reject(principal, context_id, plan, error):
        identifier = str(uuid.uuid4())
        atomic_json(draft_path(principal.id, identifier), {
            "context_id": context_id, "plan": plan, "reason": str(error)[:4000],
        })
        backend.runtime.audit(principal.id, "creative.plan.rejected", "rejected")
        return HTTPException(getattr(error, "status", 422), detail={
            "code": "invalid_authored_plan", "message": str(error)[:4000],
            "draft_id": identifier, "next": "Read the saved draft and reasons; refresh stale context or revise only the reported fields.",
        })

    @router.get("/api/v1/agent-access/creative-defaults")
    def creative_defaults():
        scope_required("observe")
        try:
            return service.default_settings()
        except CreativeError as error:
            raise HTTPException(error.status, str(error)) from error

    @router.put("/api/v1/agent-access/creative-defaults")
    async def save_creative_defaults(request: Request):
        principal = owner_required()
        body = await json_body(request)
        if set(body) != {"defaults"}:
            raise HTTPException(422, "Use a defaults object; other settings are not changed.")
        try:
            result = service.save_defaults(body["defaults"])
            backend.runtime.audit(principal.id, "creative.defaults", "saved")
            return result
        except CreativeError as error:
            raise HTTPException(error.status, str(error)) from error

    @router.post("/api/v1/agent-access/creative-context")
    async def creative_context(request: Request):
        principal = scope_required("observe")
        body = await json_body(request)
        # Resolve owned uploaded assets while preserving explicit author IDs.
        try:
            reverse = _asset_paths(body, backend.asset_store, principal.id)
            resolved = backend.asset_store.resolve_tree(principal.id, body)
            if isinstance(resolved.get("references"), list):
                for row in resolved["references"]:
                    if isinstance(row, dict) and row.get("asset"):
                        row.setdefault("path", row["asset"])
            # Check storage boundaries before CPU inspection opens any media.
            safe(resolved)
            # Native CPU beat/section analysis can take seconds. Preserve
            # auth context while keeping preview/job polling responsive.
            result = await run_in_threadpool(service.create_context, resolved, principal.id)
            return safe(result, reverse)
        except (CreativeError, AuthoringPlanError, AssetStoreError, ValueError) as error:
            raise HTTPException(getattr(error, "status", 422), str(error)) from error

    @router.post("/api/v1/agent-access/prepare-authored")
    async def prepare_authored(request: Request):
        principal = scope_required("generate")
        body = await json_body(request)
        context_id, plan = body.get("context_id"), body.get("plan")
        try:
            context = service.lookup(context_id, principal.id)
            if not isinstance(plan, dict):
                raise CreativeError("plan must be an object containing context_id and the requested native prompts.")
            if context["workflow"].startswith("director."):
                authored = normalize_authored_director_plan(context, plan)
                payload = {**strip_private_director_fields(context["params"]), **authored["native_params"]}
                payload.update(video_model=context["model_type"], pipeline_type="music_video" if context["workflow"] == "director.music_video" else "short_film_story", creative_author="connected_agent")
                # The native pipeline consumes video_params, not Studio's flat
                # generation object. Prepared plans bypass all local writing.
                payload["video_params"] = copy.deepcopy(context["params"].get("video_params") or context["params"])
                payload["_reference_style"] = str(plan.get("reference_style") or "")[:500]
                result = {"params": payload, "plan_metadata": authored["plan_metadata"]}
                kind = "director"
            else:
                authored = prepare_authored_generation(context, plan)
                payload = {**strip_private_director_fields(context["params"]), **authored["native_params"]}
                payload.update(prompt_enhancer="", enhance_on_generation=False)
                result = await _await(native_prepare(payload, prepare_only=True))
                result["plan_metadata"] = authored["plan_metadata"]
                kind = "studio"
            identifier = service.remember_prepared(context_id, principal.id, kind, result)
            return {"prepared_id": identifier, "context_id": context_id, "kind": kind, "local_writer": False, "submitted": False, "prepared": safe(result)}
        except (CreativeError, AuthoringPlanError, ValueError, HTTPException) as error:
            # Keep the exact rejected draft for its author, outside public
            # outputs and without logging scripts or private media paths.
            if isinstance(error, HTTPException):
                error = CreativeError(str(error.detail), error.status_code)
            raise reject(principal, context_id, plan, error) from error

    @router.post("/api/v1/agent-access/submit-authored")
    async def submit_authored(request: Request):
        principal = scope_required("generate")
        body = await json_body(request)
        try:
            kind, context, prepared = service.prepared(body.get("prepared_id"), principal.id)
            # Recheck live memory/profile facts; never adapt reviewed external
            # shot/window plans behind the author's back.
            current = copy.deepcopy(context)
            if service.resolve is not None:
                await run_in_threadpool(service.resolve, current, {"params": current["params"], "references": current["references"]})
                if (
                    current.get("execution_profile") != context.get("execution_profile")
                    or compute_studio_timeline(current) != context["timeline"]
                    or current["params"].get("resolution") != context["params"].get("resolution")
                ):
                    raise CreativeError("Execution geometry/profile changed; refresh the context and review the plan before submitting.", 409)
            params = copy.deepcopy(prepared["params"])
            if kind == "director":
                return await _await(native_director(params))
            if prepared.get("workspace"):
                params["workspace"] = prepared["workspace"]
            params.update(prompt_enhancer="", enhance_on_generation=False)
            return await _await(native_prepare(params, prepare_only=False))
        except (CreativeError, AuthoringPlanError, ValueError) as error:
            raise HTTPException(getattr(error, "status", 422), str(error)) from error

    @router.get("/api/v1/agent-access/authoring-drafts/{draft_id}")
    def authoring_draft(draft_id: str):
        principal = scope_required("observe")
        try:
            path = draft_path(principal.id, draft_id)
        except (ValueError, TypeError) as error:
            raise HTTPException(404, "Authored draft not found") from error
        if path.exists() and path.stat().st_size > 8 * 1024 * 1024:
            raise HTTPException(413, "Authored draft exceeds the readback limit")
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not value:
            raise HTTPException(404, "Authored draft not found")
        return safe(value)

    return router
