"""Restart gaps must never permit duplicate dispatch or erase ownership."""
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.services.comfyui_api import create_router
from app.services.comfyui_runtime import ComfyService, ServiceError
from app.services.agent_access.catalog import required_scope
from app.services.comfyui_integration import ComfyJobAdapter


class QuietClient:
    def __init__(self, _endpoint):
        pass

    def close(self):
        pass

    def queue(self):
        return {"queue_running": [], "queue_pending": []}


def service(tmp_path, *, exists=False):
    current = {"id": "agent-a"}

    def missing_job(_job_id):
        raise ServiceError(404, "Job not recovered")

    result = ComfyService(tmp_path, client_factory=QuietClient,
                         principal_id=lambda: current["id"], is_owner=lambda: False,
                         require_job_access=missing_job, job_exists=lambda _id: exists)
    result.configure({"enabled": True})
    request = result.prepare("cpu_empty_image_smoke", {
        "client_submission_id": str(uuid.uuid4()), "inputs": {},
    })
    result.reserve(request, "lost-job")
    return result, request["client_submission_id"], current


def test_dispatch_crash_without_visible_queue_entry_blocks_native(tmp_path):
    controller, submission, _ = service(tmp_path)
    controller._update_receipt(submission, state="dispatching")
    with pytest.raises(ServiceError, match="reconciliation"):
        controller.assert_native_idle()


def test_orphan_intent_can_be_dismissed_without_generation(tmp_path):
    controller, submission, _ = service(tmp_path)
    app = FastAPI()
    app.include_router(create_router(controller))
    with TestClient(app) as client:
        route = f"/api/v1/comfyui/jobs/by-submission/{submission}/abandon"
        assert required_scope("POST", route) == "generate"
        response = client.post(route)
        assert response.status_code == 200
        assert response.json()["state"] == "cancelled"
    assert controller.status()["pending_submissions"] == []
    assert controller.configure({"enabled": False})["enabled"] is False


def test_foreign_agent_cannot_discard_an_orphan_intent(tmp_path):
    controller, submission, current = service(tmp_path)
    current["id"] = "agent-b"
    with pytest.raises(ServiceError) as error:
        controller.abandon_submission(submission)
    assert error.value.status == 404
    assert controller.status()["pending_submissions"] == []


def test_dispatched_intent_cannot_be_dismissed_as_unsubmitted(tmp_path):
    controller, submission, _ = service(tmp_path)
    controller._update_receipt(submission, state="uncertain")
    with pytest.raises(ServiceError, match="undispatched"):
        controller.abandon_submission(submission)


def test_queue_owned_intent_uses_normal_job_cancellation(tmp_path):
    controller, submission, _ = service(tmp_path, exists=True)
    # A real queue row cannot bypass its normal ownership check through a
    # receipt cleanup operation, even when the receipt's owner matches.
    with pytest.raises(ServiceError) as error:
        controller.abandon_submission(submission)
    assert error.value.status == 404


@pytest.mark.parametrize("outside", [False, True])
def test_recovered_monitor_retains_approved_director_output_folder(tmp_path, outside):
    jobs, started = {}, []
    base = tmp_path / "outputs"
    adapter = ComfyJobAdapter(tmp_path, jobs=jobs,
        workspace_dir=lambda name: str(base if name == "default" else base / name),
        start_worker=started.append, require_job_access=lambda _id: None,
        principal_id=lambda: None, link_folders=lambda values: values,
        known_assets=lambda: [], resolve_media=lambda *_args: None,
        finish_job=lambda *_args, **_kwargs: True, update_job=lambda *_args, **_kwargs: True,
        is_cancelled=lambda job: bool(job.get("cancel_requested")), unload_native=lambda: None)
    adapter.service.client_factory = QuietClient
    adapter.service.configure({"enabled": True})
    submission = str(uuid.uuid4())
    prepared = adapter.service.prepare("cpu_empty_image_smoke", {"client_submission_id": submission, "inputs": {}})
    adapter.service.reserve(prepared, "director-child")
    target = tmp_path / "unapproved" if outside else base / "project" / "clips"
    adapter.service._update_receipt(submission, state="running", prompt_id="owned-prompt", out_dir=str(target))
    if outside:
        with pytest.raises(ServiceError, match="outside"):
            adapter.service.resume_submission(submission)
        assert not started
    else:
        adapter.service.resume_submission(submission)
        assert Path(jobs["director-child"]["out_dir"]) == target
        assert started == ["director-child"]
