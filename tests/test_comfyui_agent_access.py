"""Pure scope-classification tests for the ComfyUI API namespace."""
try:
    from services.agent_access.catalog import required_scope
except ModuleNotFoundError:
    from app.services.agent_access.catalog import required_scope


def test_comfyui_reads_and_owned_submission_receipts_are_observe():
    assert required_scope("GET", "/api/v1/comfyui/status") == "observe"
    assert required_scope("GET", "/api/v1/comfyui/guide") == "observe"
    assert required_scope("GET", "/api/v1/comfyui/workflows") == "observe"
    assert required_scope("GET", "/api/v1/comfyui/jobs/by-submission/123") == "observe"
    assert required_scope("HEAD", "/api/v1/comfyui/unknown/read") == "observe"


def test_exact_submission_resume_route_is_generate():
    submission_id = "6d6c2a95-967b-42d4-8d18-6a19f684c38c"
    assert required_scope(
        "POST", f"/api/v1/comfyui/jobs/by-submission/{submission_id}/resume"
    ) == "generate"
    assert required_scope(
        "POST", f"/api/v1/comfyui/jobs/by-submission/{submission_id}/resume/again"
    ) == "admin"
    assert required_scope(
        "POST", "/api/v1/comfyui/jobs/by-submission/not-a-uuid/resume"
    ) == "admin"
    for action in ("resume", "abandon"):
        assert required_scope("POST", f"/api/v1/comfyui/jobs/by-submission/{{uuid}}/{action}") == "generate"
        assert required_scope("POST", f"/api/v1/comfyui/jobs/by-submission/{submission_id}/{action}") == "generate"


def test_only_exact_registered_workflow_run_route_is_generate():
    assert required_scope("POST", "/api/v1/comfyui/workflows/workflow-123/run") == "generate"
    assert required_scope("POST", "/api/v1/comfyui/workflows/run/run") == "generate"
    assert required_scope("POST", "/api/v1/comfyui/workflows/workflow-123/run/queue") == "admin"
    assert required_scope("POST", "/api/v1/comfyui/workflows/workflow-123/validate/run") == "admin"
    assert required_scope("POST", "/api/v1/comfyui/workflows//run") == "admin"


def test_connection_and_model_management_require_manage():
    assert required_scope("POST", "/api/v1/comfyui/connection/test") == "manage"
    assert required_scope("POST", "/api/v1/comfyui/models/discover") == "manage"
    assert required_scope("POST", "/api/v1/comfyui/models/scan") == "manage"
    assert required_scope("POST", "/api/v1/comfyui/models/link") == "manage"
    assert required_scope("POST", "/api/v1/comfyui/models/discover-extra") == "admin"


def test_workflow_configuration_and_unknown_writes_require_admin():
    assert required_scope("POST", "/api/v1/comfyui/workflows") == "admin"
    assert required_scope("POST", "/api/v1/comfyui/workflows/import") == "admin"
    assert required_scope("DELETE", "/api/v1/comfyui/workflows/workflow-123") == "admin"
    assert required_scope("PUT", "/api/v1/comfyui/settings") == "admin"
    assert required_scope("PUT", "/api/v1/comfyui/routes/image") == "admin"
    assert required_scope("PATCH", "/api/v1/comfyui/routes/image") == "admin"
    assert required_scope("POST", "/api/v1/comfyui/unknown/write") == "admin"


def test_native_route_scopes_are_unchanged():
    assert required_scope("GET", "/api/v1/models") == "observe"
    assert required_scope("POST", "/api/v1/generate") == "generate"
    assert required_scope("POST", "/api/v1/media/trim") == "manage"
    assert required_scope("POST", "/api/v1/system-config") == "admin"
