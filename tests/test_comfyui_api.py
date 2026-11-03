"""FastAPI contract tests for the ComfyUI HTTP boundary."""
from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.comfyui_api import create_router


class FakeComfyService:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.failures: dict[str, Exception] = {}
        self.context_values: list[object] = []

    def _record(self, name: str, *args):
        self.calls.append((name, args))
        principal = _principal_context.get()
        if principal is not None:
            self.context_values.append(principal)
        if name in self.failures:
            raise self.failures[name]
        return {"method": name}

    def status(self):
        return self._record("status")

    def configure(self, payload):
        return self._record("configure", payload)

    def test_connection(self):
        return self._record("test_connection")

    def discover_connections(self):
        return self._record("discover_connections")

    def discover(self, paths):
        return self._record("discover", paths)

    def scan(self, paths):
        return self._record("scan", paths)

    def link(self, paths):
        return self._record("link", paths)

    def workflows(self):
        return self._record("workflows")

    def workflow_options(self, workflow_id):
        return self._record("workflow_options", workflow_id)

    def save_workflow(self, payload):
        return self._record("save_workflow", payload)

    def delete_workflow(self, workflow_id):
        return self._record("delete_workflow", workflow_id)

    def validate_workflow(self, workflow_id, *args):
        return self._record("validate_workflow", workflow_id, *args)

    def run_workflow(self, workflow_id, payload):
        return self._record("run_workflow", workflow_id, payload)

    def submission_receipt(self, submission_id):
        return self._record("submission_receipt", submission_id)

    def resume_submission(self, submission_id):
        return self._record("resume_submission", submission_id)

    def save_route(self, payload):
        return self._record("save_route", payload)


_principal_context: ContextVar[object | None] = ContextVar("comfy_test_principal", default=None)


def _client(service: FakeComfyService) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(service))
    return TestClient(app)


def test_setup_discovery_and_workflow_options_are_typed_explicit_operations():
    service = FakeComfyService()
    client = _client(service)
    assert client.post("/api/v1/comfyui/connection/discover", json={}).status_code == 200
    assert client.get("/api/v1/comfyui/workflows/known-template/options").status_code == 200
    assert service.calls == [("discover_connections", ()), ("workflow_options", ("known-template",))]
    assert client.post("/api/v1/comfyui/connection/discover", json={"arbitrary_endpoint": "http://example.com"}).status_code == 422


def test_status_settings_and_connection_use_injected_service():
    service = FakeComfyService()
    client = _client(service)

    assert client.get("/api/v1/comfyui/status").json() == {"method": "status"}
    guide = client.get("/api/v1/comfyui/guide")
    assert guide.status_code == 200
    guide_content = str(guide.json())
    for required in (
        "API-format prompt graph",
        "current selected",
        "native engine as the default",
        "does not install",
        "GET /api/v1/comfyui/jobs/by-submission/{uuid}",
        "POST /api/v1/comfyui/jobs/by-submission/{uuid}/resume",
        "global interrupt",
        "arbitrary",
    ):
        assert required.casefold() in guide_content.casefold()
    settings = {
        "enabled": True,
        "endpoint": "http://127.0.0.1:8188",
        "execution_timeout": 600,
        "dedicated": False,
        "routes": {"image": {"workflow_id": "workflow-1"}},
    }
    assert client.put("/api/v1/comfyui/settings", json=settings).status_code == 200
    assert client.post("/api/v1/comfyui/connection/test", json={}).status_code == 200
    assert service.calls[1] == ("configure", (settings,))
    assert service.calls[2] == ("test_connection", ())


def test_model_routes_validate_and_forward_bounded_paths():
    service = FakeComfyService()
    client = _client(service)

    assert client.post("/api/v1/comfyui/models/discover", json={}).status_code == 200
    assert client.post("/api/v1/comfyui/models/discover", json={"paths": ["C:/ComfyUI"]}).status_code == 200
    assert client.post("/api/v1/comfyui/models/scan", json={"paths": ["C:/ComfyUI/models"]}).status_code == 200
    assert client.post("/api/v1/comfyui/models/link", json={"paths": ["C:/ComfyUI/models"]}).status_code == 200

    assert service.calls[0] == ("discover", (None,))
    assert service.calls[1] == ("discover", (["C:/ComfyUI"],))
    assert service.calls[2] == ("scan", (["C:/ComfyUI/models"],))
    assert service.calls[3] == ("link", (["C:/ComfyUI/models"],))
    assert client.post("/api/v1/comfyui/models/scan", json={"paths": [], "arbitrary": True}).status_code == 422
    assert client.post("/api/v1/comfyui/models/discover", json={"paths": ["x"] * 65}).status_code == 422
    assert client.post("/api/v1/comfyui/models/scan", json={"paths": ["x"] * 257}).status_code == 422
    assert client.post("/api/v1/comfyui/models/link", json={"paths": ["x"] * 257}).status_code == 422


def test_workflow_lifecycle_forwards_only_validated_payloads():
    service = FakeComfyService()
    client = _client(service)
    workflow = {"nodes": [{"id": 1, "type": "KSampler"}], "links": []}
    save_payload = {
        "name": "Still image",
        "kind": "image",
        "workflow": workflow,
        "bindings": [{"name": "prompt", "node_id": "1", "field": "text"}],
        "description": "A sample graph",
    }

    assert client.get("/api/v1/comfyui/workflows").status_code == 200
    assert client.post("/api/v1/comfyui/workflows", json=save_payload).status_code == 200
    assert client.delete("/api/v1/comfyui/workflows/workflow-1").status_code == 200
    assert client.post("/api/v1/comfyui/workflows/workflow-1/validate", json={}).status_code == 200
    assert client.post(
        "/api/v1/comfyui/workflows/workflow-1/validate", json={"inputs": {}}
    ).status_code == 200
    assert client.post(
        "/api/v1/comfyui/workflows/workflow-1/validate",
        json={"inputs": {"checkpoint": "selected.safetensors"}},
    ).status_code == 200

    submission_id = uuid4()
    run_payload = {
        "client_submission_id": str(submission_id),
        "inputs": {"prompt": "A lighthouse"},
        "workspace": "Research",
    }
    assert client.post("/api/v1/comfyui/workflows/workflow-1/run", json=run_payload).status_code == 200
    assert client.get(f"/api/v1/comfyui/jobs/by-submission/{submission_id}").status_code == 200
    assert client.post(
        f"/api/v1/comfyui/jobs/by-submission/{submission_id}/resume"
    ).status_code == 200

    assert service.calls[1] == ("save_workflow", (save_payload,))
    assert service.calls[2] == ("delete_workflow", ("workflow-1",))
    assert service.calls[3] == ("validate_workflow", ("workflow-1",))
    assert service.calls[4] == (
        "validate_workflow",
        ("workflow-1", {}),
    )
    assert service.calls[5] == (
        "validate_workflow",
        ("workflow-1", {"checkpoint": "selected.safetensors"}),
    )
    expected_run = {**run_payload, "client_submission_id": str(submission_id)}
    assert service.calls[6] == ("run_workflow", ("workflow-1", expected_run))
    assert service.calls[7] == ("submission_receipt", (str(submission_id),))
    assert service.calls[8] == ("resume_submission", (str(submission_id),))

    assert client.post("/api/v1/comfyui/workflows", json={**save_payload, "unexpected": 1}).status_code == 422
    assert client.post("/api/v1/comfyui/workflows", json={**save_payload, "kind": "other"}).status_code == 422
    assert client.post("/api/v1/comfyui/workflows/workflow-1/validate", json={"unexpected": 1}).status_code == 422


def test_run_and_route_payloads_are_validated_and_model_type_is_merged():
    service = FakeComfyService()
    client = _client(service)
    route_payload = {
        "workflow_id": "workflow-1",
        "enabled": True,
        "field_bindings": {"prompt": "inputs.prompt"},
        "kind": "image",
    }

    response = client.put("/api/v1/comfyui/routes/image", json=route_payload)
    assert response.status_code == 200
    assert service.calls[-1] == (
        "save_route",
        ({**route_payload, "model_type": "image"},),
    )
    assert client.put("/api/v1/comfyui/routes/image", json={**route_payload, "model_type": "video"}).status_code == 422
    assert client.put("/api/v1/comfyui/routes/bad%2Fproxy", json=route_payload).status_code in {404, 422}

    bad_run = {
        "client_submission_id": "not-a-uuid",
        "inputs": {},
        "workspace": "Research",
    }
    assert client.post("/api/v1/comfyui/workflows/workflow-1/run", json=bad_run).status_code == 422
    assert client.post(
        "/api/v1/comfyui/workflows/workflow-1/run",
        json={**bad_run, "client_submission_id": str(uuid4()), "workspace": "x" * 121},
    ).status_code == 422


def test_settings_timeout_and_forbidden_extra_fields_are_rejected():
    client = _client(FakeComfyService())
    base = {"enabled": True, "endpoint": "http://127.0.0.1:8188"}

    assert client.put("/api/v1/comfyui/settings", json={**base, "execution_timeout": 29}).status_code == 422
    assert client.put("/api/v1/comfyui/settings", json={**base, "execution_timeout": 14401}).status_code == 422
    assert client.put("/api/v1/comfyui/settings", json={**base, "unknown": True}).status_code == 422
    assert client.post("/api/v1/comfyui/connection/test", json={"url": "http://elsewhere"}).status_code == 422


def test_sync_service_call_preserves_contextvar_principal():
    service = FakeComfyService()
    client = _client(service)
    token = _principal_context.set("principal-123")
    try:
        assert client.get("/api/v1/comfyui/status").status_code == 200
    finally:
        _principal_context.reset(token)

    assert service.context_values == ["principal-123"]


def test_service_errors_map_to_http_without_leaking_unexpected_exception():
    class ServiceError(ValueError):
        def __init__(self, status, message):
            super().__init__(message)
            self.status = status
            self.message = message

    service = FakeComfyService()
    client = _client(service)

    service.failures["status"] = ServiceError(409, "ComfyUI is already running")
    response = client.get("/api/v1/comfyui/status")
    assert response.status_code == 409
    assert response.json()["detail"] == "ComfyUI is already running"

    service.failures["status"] = ValueError("Invalid endpoint")
    assert client.get("/api/v1/comfyui/status").status_code == 422
    service.failures["status"] = KeyError("workflow")
    assert client.get("/api/v1/comfyui/status").status_code == 404

    service.failures["status"] = RuntimeError("private stack details")
    response = client.get("/api/v1/comfyui/status")
    assert response.status_code == 500
    assert response.json()["detail"] == "ComfyUI request failed"
    assert "private stack details" not in response.text
