"""Authenticated-host ComfyUI API boundary.

The router keeps request validation and HTTP error mapping at the edge. The
injected ComfyService owns persistence, runtime state, workflow execution, and
job ownership checks.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from .comfyui_guide import get_comfyui_guide


_log = logging.getLogger(__name__)

PathValue = Annotated[StrictStr, Field(min_length=1, max_length=4096)]
WorkflowName = Annotated[StrictStr, Field(min_length=1, max_length=120)]
WorkflowId = Annotated[StrictStr, Field(min_length=1, max_length=128)]
WorkspaceName = Annotated[StrictStr, Field(min_length=1, max_length=120)]
WorkflowKind = Literal["image", "video", "audio"]


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _EmptyRequest(_RequestModel):
    pass


class SettingsRequest(_RequestModel):
    enabled: StrictBool
    endpoint: Annotated[StrictStr, Field(min_length=1, max_length=2048)]
    execution_timeout: Annotated[StrictInt, Field(ge=30, le=14400)] | None = None
    dedicated: StrictBool | None = None
    routes: dict[StrictStr, Any] | None = None


class DiscoverModelsRequest(_RequestModel):
    paths: list[PathValue] | None = Field(default=None, max_length=64)


class ScanModelsRequest(_RequestModel):
    paths: list[PathValue] = Field(max_length=256)


class LinkModelsRequest(_RequestModel):
    paths: list[PathValue] = Field(max_length=256)


class SaveWorkflowRequest(_RequestModel):
    name: WorkflowName
    kind: WorkflowKind
    workflow: dict[str, Any]
    bindings: list[Any]
    description: Annotated[StrictStr, Field(max_length=1000)] | None = None


class RunWorkflowRequest(_RequestModel):
    client_submission_id: UUID
    inputs: dict[str, Any]
    workspace: WorkspaceName


class ValidateWorkflowRequest(_RequestModel):
    inputs: dict[str, Any] = Field(default_factory=dict)


class SaveRouteRequest(_RequestModel):
    workflow_id: WorkflowId
    enabled: StrictBool
    field_bindings: dict[StrictStr, StrictStr]
    input_defaults: dict[str, Any] = Field(default_factory=dict)
    kind: WorkflowKind


def create_router(service) -> APIRouter:
    """Create the ComfyUI API routes around an injected synchronous service."""
    router = APIRouter()

    @router.get("/api/v1/comfyui/status")
    def status():
        return _call(service.status)

    @router.get("/api/v1/comfyui/guide")
    def guide():
        return get_comfyui_guide()

    @router.put("/api/v1/comfyui/settings")
    def configure(payload: SettingsRequest):
        return _call(service.configure, payload.model_dump(exclude_unset=True))

    @router.post("/api/v1/comfyui/connection/test")
    def test_connection(_payload: _EmptyRequest):
        return _call(service.test_connection)

    @router.post("/api/v1/comfyui/models/discover")
    def discover_models(payload: DiscoverModelsRequest):
        return _call(service.discover, payload.paths)

    @router.post("/api/v1/comfyui/models/scan")
    def scan_models(payload: ScanModelsRequest):
        return _call(service.scan, payload.paths)

    @router.post("/api/v1/comfyui/models/link")
    def link_models(payload: LinkModelsRequest):
        return _call(service.link, payload.paths)

    @router.get("/api/v1/comfyui/workflows")
    def workflows():
        return _call(service.workflows)

    @router.post("/api/v1/comfyui/workflows")
    def save_workflow(payload: SaveWorkflowRequest):
        return _call(service.save_workflow, payload.model_dump(exclude_unset=True))

    @router.delete("/api/v1/comfyui/workflows/{workflow_id}")
    def delete_workflow(workflow_id: WorkflowId):
        return _call(service.delete_workflow, workflow_id)

    @router.post("/api/v1/comfyui/workflows/{workflow_id}/validate")
    def validate_workflow(workflow_id: WorkflowId, payload: ValidateWorkflowRequest):
        if "inputs" in payload.model_fields_set:
            return _call(service.validate_workflow, workflow_id, payload.inputs)
        return _call(service.validate_workflow, workflow_id)

    @router.post("/api/v1/comfyui/workflows/{workflow_id}/run")
    def run_workflow(workflow_id: WorkflowId, payload: RunWorkflowRequest):
        request = payload.model_dump()
        request["client_submission_id"] = str(request["client_submission_id"])
        return _call(service.run_workflow, workflow_id, request)

    @router.get("/api/v1/comfyui/jobs/by-submission/{uuid}")
    def submission_receipt(uuid: UUID):
        return _call(service.submission_receipt, str(uuid))

    @router.post("/api/v1/comfyui/jobs/by-submission/{uuid}/resume")
    def resume_submission(uuid: UUID):
        return _call(service.resume_submission, str(uuid))

    @router.post("/api/v1/comfyui/jobs/by-submission/{uuid}/abandon")
    def abandon_submission(uuid: UUID):
        return _call(service.abandon_submission, str(uuid))

    @router.put("/api/v1/comfyui/routes/{model_type}")
    def save_route(
        payload: SaveRouteRequest,
        model_type: Annotated[
            str,
            Path(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$"),
        ],
    ):
        request = payload.model_dump(exclude_unset=True)
        request["model_type"] = model_type
        return _call(service.save_route, request)

    return router


def _call(method, *args):
    """Call a service method and translate expected errors without leaking internals."""
    try:
        return method(*args)
    except HTTPException:
        raise
    except Exception as error:
        status = getattr(error, "status", None)
        if isinstance(status, int) and 400 <= status <= 599:
            message = getattr(error, "message", None)
            if not isinstance(message, str) or not message:
                message = str(error) if isinstance(error, ValueError) else "ComfyUI request failed"
            raise HTTPException(status, message[:1000]) from error
        if isinstance(error, KeyError):
            detail = str(error).strip("'\"") or "Resource not found"
            raise HTTPException(404, detail[:1000]) from error
        if isinstance(error, ValueError):
            detail = str(error) or "Invalid ComfyUI request"
            raise HTTPException(422, detail[:1000]) from error
        _log.exception("Unexpected ComfyUI service failure")
        raise HTTPException(500, "ComfyUI request failed") from error
