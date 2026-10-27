from __future__ import annotations

import asyncio
from functools import wraps
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI
import httpx
import pytest


import sys
_APP = Path(__file__).resolve().parents[1] / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from services.worlds_workflow import (  # noqa: E402
    WORLD_GUIDANCE,
    WORLD_MODEL_TYPE,
    WORLD_RESOLUTION,
    WORLD_STEPS,
    WORLD_VIDEO_LENGTH,
    build_generation_body,
    build_native_prompt,
    canonical_request_digest,
    create_worlds_router,
    normalize_world_request,
)
from services.worlds_storage import WorldsStorageError  # noqa: E402


def _run_async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


def _payload(operation: str, **specific):
    fields = {
        "operation": operation,
        "acknowledge_pixelumm_noncommercial": True,
        **specific,
    }
    return {
        **fields,
        "client_request_id": str(uuid.uuid4()),
        "request_digest": canonical_request_digest(fields),
    }


class WorldsHarness:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.root / "state"
        self.state_dir.mkdir()
        self.source = self.state_dir / "saved-world.wgpworld"
        self.source.write_bytes(b"PK\x03\x04synthetic-world")
        self.video = self.state_dir / "recording.mp4"
        self.video.write_bytes(b"\x00\x00\x00\x18ftypmp42 synthetic")
        self.poster = self.state_dir / "capture.jpg"
        self.poster.write_bytes(b"\xff\xd8synthetic-jpeg\xff\xd9")
        self.jobs: dict[str, dict] = {}
        self.receipts: dict[str, dict] = {}
        self.worlds: dict[str, dict] = {
            "world-1": {
                "world_id": "world-1", "title": "Saved scene", "description": "Pine forest",
                "recording_available": True, "segment_count": 1,
                "video_url": "/media/world-1.mp4", "poster_url": "/media/world-1.jpg",
                "private_path": str(self.source),
            }
        }
        self.submit_calls: list[dict] = []
        self.cancel_calls: list[str] = []
        self.publish_calls: list[str] = []
        self.lookup_calls = 0
        self.record_thread_id: int | None = None
        self.raise_after_accept = False
        self.rejection_detail: str | None = None
        self.publication_error: Exception | None = None
        self.publication_error_once = False
        self.bad_verification = False

    async def submit(self, body: dict):
        self.submit_calls.append(dict(body))
        submission_id = body["_client_submission_id"]
        if self.rejection_detail is not None:
            return {
                "rejected": True,
                "detail": self.rejection_detail,
                "client_submission_id": submission_id,
            }
        job_id = f"job-{len(self.submit_calls)}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "client_submission_id": submission_id,
            "status": "queued",
            "output_files": [],
        }
        if self.raise_after_accept:
            raise OSError("simulated lost submit response")
        return {"job_id": job_id, "client_submission_id": submission_id, "status": "queued"}

    def publish(self, job: dict, receipt: dict):
        self.publish_calls.append(receipt["client_request_id"])
        if self.publication_error is not None:
            error = self.publication_error
            if self.publication_error_once:
                self.publication_error = None
            raise error
        if self.bad_verification:
            return {"verified": True, "video_verified": True, "world_file_verified": False}
        world_id = receipt["request"].get("world_id", "world-created")
        world = {
            "world_id": world_id,
            "title": "Created forest" if world_id == "world-created" else "Saved scene",
            "description": receipt["request"].get("description", "Pine forest"),
            "video_url": f"/media/{world_id}.mp4",
            "poster_url": f"/media/{world_id}.jpg",
            "recording_available": True,
            "segment_count": 1 if world_id == "world-created" else 2,
        }
        self.worlds[world_id] = world
        return {
            "verified": True,
            "video_verified": True,
            "world_file_verified": True,
            "world": world,
        }

    def router(self):
        return create_worlds_router(
            workspace_root=self.root,
            submit_generation=self.submit,
            lookup_jobs=self.lookup_jobs_fn,
            get_job_status=lambda job_id: self.jobs.get(job_id),
            cancel_generation=self.cancel,
            list_worlds=lambda: list(self.worlds.values()),
            get_world=lambda world_id: self.worlds.get(world_id),
            resolve_world_file=lambda world_id: self.source,
            publish_completed_generation=self.publish,
            save_world=lambda world_id: self.source,
            record_world=self.record,
            capture_world=lambda world_id: self.poster,
            read_receipt=lambda request_id: self.receipts.get(request_id),
            write_receipt=self.write_receipt,
        )

    def lookup_jobs_fn(self):
        self.lookup_calls += 1
        return list(self.jobs.values())

    async def write_receipt(self, request_id: str, receipt: dict):
        self.receipts[request_id] = dict(receipt)

    def cancel(self, job_id: str):
        self.cancel_calls.append(job_id)

    def record(self, world_id: str):
        self.record_thread_id = threading.get_ident()
        return self.video

    def client(self) -> httpx.AsyncClient:
        app = FastAPI()
        app.include_router(self.router())
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_request_contract_digest_prompts_and_fixed_generation_defaults():
    create = _payload("create", description="A quiet lake")
    request_id, fields, digest = normalize_world_request(create)
    assert request_id == create["client_request_id"]
    assert digest == create["request_digest"]
    assert build_native_prompt(fields) == "A quiet lake"

    move = _payload("move", world_id="world-1", action="Forward", duration_ms=800)
    _, move_fields, _ = normalize_world_request(move)
    assert build_native_prompt(move_fields) == "[Forward 0.8s]"
    body = build_generation_body(move_fields, client_submission_id=move["client_request_id"], custom_guide="C:/safe/world.wgpworld")
    assert body == {
        "model_type": WORLD_MODEL_TYPE,
        "prompt": "[Forward 0.8s]",
        "num_inference_steps": WORLD_STEPS,
        "guidance_scale": WORLD_GUIDANCE,
        "video_length": WORLD_VIDEO_LENGTH,
        "resolution": WORLD_RESOLUTION,
        "custom_settings": {
            "world_output": "world_video",
            "acknowledge_pixelumm_noncommercial": True,
        },
        "_client_submission_id": move["client_request_id"],
        "custom_guide": "C:/safe/world.wgpworld",
    }


@pytest.mark.parametrize("bad", [800.0, True, 0, 10_001])
def test_move_duration_requires_bounded_integer_milliseconds(bad):
    fields = {"operation": "move", "acknowledge_pixelumm_noncommercial": True,
              "world_id": "world-1", "action": "Wait", "duration_ms": bad}
    payload = {**fields, "client_request_id": str(uuid.uuid4()),
               "request_digest": canonical_request_digest(fields)}
    with pytest.raises(Exception):
        normalize_world_request(payload)


@pytest.mark.parametrize("mutation", ["digest", "unknown", "license", "path"])
def test_request_rejects_mismatched_or_unsafe_client_fields(mutation):
    payload = _payload("create", description="Test")
    if mutation == "digest":
        payload["request_digest"] = "0" * 64
    elif mutation == "unknown":
        payload["model_type"] = "other"
    elif mutation == "license":
        payload["acknowledge_pixelumm_noncommercial"] = False
    else:
        payload["custom_guide"] = "../../somewhere.wgpworld"
    with pytest.raises(Exception):
        normalize_world_request(payload)


@_run_async_test
async def test_create_move_save_reopen_record_capture_and_publication(tmp_path):
    h = WorldsHarness(tmp_path)
    async with h.client() as client:
        create = _payload("create", description="A quiet pine forest")
        accepted = await client.post("/api/v1/worlds/requests", json=create)
        assert accepted.status_code == 200
        assert accepted.json()["state"] == "queued"
        assert h.submit_calls[0]["prompt"] == "A quiet pine forest"
        assert h.submit_calls[0]["custom_settings"] == {
            "world_output": "world_video",
            "acknowledge_pixelumm_noncommercial": True,
        }
        assert h.submit_calls[0]["_client_submission_id"] == create["client_request_id"]
        assert "custom_guide" not in h.submit_calls[0]
        assert h.publish_calls == [], "a queued job cannot publish a world before completion"

        # Reposting the identical client request only reconciles its receipt.
        duplicate = await client.post("/api/v1/worlds/requests", json=create)
        assert duplicate.status_code == 200
        assert len(h.submit_calls) == 1

        created_job = h.jobs["job-1"]
        created_job.update(status="completed", output_files=["world.mp4", "world.wgpworld"])
        before_complete = len(h.publish_calls)
        running = await client.get(f"/api/v1/worlds/requests/{create['client_request_id']}")
        assert running.status_code == 200
        assert running.json()["state"] == "published"
        assert len(h.publish_calls) == before_complete + 1
        assert running.json()["world"]["world_id"] == "world-created"
        assert "private_path" not in running.text

        move = _payload("move", world_id="world-created", action="Forward", duration_ms=800)
        moved = await client.post("/api/v1/worlds/requests", json=move)
        assert moved.status_code == 200
        assert h.submit_calls[1]["prompt"] == "[Forward 0.8s]"
        assert h.submit_calls[1]["custom_guide"] == str(h.source.resolve())
        h.jobs["job-2"].update(status="completed", output_files=["move.mp4", "move.wgpworld"])
        done = await client.get(f"/api/v1/worlds/requests/{move['client_request_id']}")
        assert done.json()["state"] == "published"

        world = await client.get("/api/v1/worlds/world-created")
        assert world.status_code == 200
        assert "private_path" not in world.text
        save = await client.post("/api/v1/worlds/world-created/save", json={"acknowledge_pixelumm_noncommercial": True})
        assert save.status_code == 200
        assert save.headers["content-disposition"].endswith('"world-created.wgpworld"')
        assert save.content.startswith(b"PK")

        recording = await client.post(
            "/api/v1/worlds/world-created/record",
            json={"acknowledge_pixelumm_noncommercial": True},
        )
        assert recording.status_code == 200
        assert recording.content.endswith(b"synthetic")
        assert h.record_thread_id != threading.get_ident(), "CPU recording callback runs off the event loop thread"
        capture = await client.get("/api/v1/worlds/world-created/capture")
        assert capture.status_code == 200
        assert capture.headers["content-type"] == "image/jpeg"
        assert capture.content.startswith(b"\xff\xd8")


@_run_async_test
async def test_digest_reuse_conflict_ambiguous_submit_and_get_only_reconcile(tmp_path):
    h = WorldsHarness(tmp_path)
    h.raise_after_accept = True
    async with h.client() as client:
        bad_digest = _payload("create", description="A test world")
        bad_digest["request_digest"] = "0" * 64
        bad = await client.post("/api/v1/worlds/requests", json=bad_digest)
        assert bad.status_code == 409
        assert h.submit_calls == []

        original = _payload("create", description="A sandstone canyon")
        accepted = await client.post("/api/v1/worlds/requests", json=original)
        assert accepted.status_code == 200
        assert accepted.json()["state"] == "queued"
        assert len(h.submit_calls) == 1

        conflict_fields = {"operation": "create", "acknowledge_pixelumm_noncommercial": True,
                           "description": "Different place"}
        conflict = {**conflict_fields,
                    "client_request_id": original["client_request_id"],
                    "request_digest": canonical_request_digest(conflict_fields)}
        response = await client.post("/api/v1/worlds/requests", json=conflict)
        assert response.status_code == 409
        assert len(h.submit_calls) == 1

        # A subsequent status read uses the ID lookup only; it never repeats POST.
        status = await client.get(f"/api/v1/worlds/requests/{original['client_request_id']}")
        assert status.json()["job_id"] == "job-1"
        assert len(h.submit_calls) == 1


@_run_async_test
async def test_cancel_requires_exact_owner_and_is_one_shot(tmp_path):
    h = WorldsHarness(tmp_path)
    async with h.client() as client:
        request = _payload("create", description="A cave")
        accepted = await client.post("/api/v1/worlds/requests", json=request)
        assert accepted.status_code == 200
        h.jobs["job-1"]["client_submission_id"] = str(uuid.uuid4())
        foreign = await client.post(f"/api/v1/worlds/requests/{request['client_request_id']}/cancel")
        assert foreign.status_code == 409
        assert h.cancel_calls == []

        h.jobs["job-1"]["client_submission_id"] = request["client_request_id"]
        owned = await client.post(f"/api/v1/worlds/requests/{request['client_request_id']}/cancel")
        assert owned.status_code == 200
        assert owned.json()["cancel_state"] == "sent"
        again = await client.post(f"/api/v1/worlds/requests/{request['client_request_id']}/cancel")
        assert again.status_code == 200
        assert h.cancel_calls == ["job-1"]


@_run_async_test
async def test_publication_waits_for_verified_video_and_sidecar(tmp_path):
    h = WorldsHarness(tmp_path)
    h.bad_verification = True
    async with h.client() as client:
        request = _payload("create", description="A snowy station")
        await client.post("/api/v1/worlds/requests", json=request)
        h.jobs["job-1"].update(status="completed", output_files=["fake.mp4", "fake.wgpworld"])
        receipt = await client.get(f"/api/v1/worlds/requests/{request['client_request_id']}")
        assert receipt.json()["state"] == "verification_failed"
        assert "world" not in receipt.json()
        assert "world-created" not in {world["world_id"] for world in (await client.get("/api/v1/worlds")).json()["worlds"]}
        again = await client.get(f"/api/v1/worlds/requests/{request['client_request_id']}")
        duplicate = await client.post("/api/v1/worlds/requests", json=request)
        assert again.json()["state"] == duplicate.json()["state"] == "verification_failed"
        assert len(h.publish_calls) == 1, "verification failures are terminal and never republish"
        assert len(h.submit_calls) == 1, "reconciliation never resubmits a terminal request"


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ValueError("invalid world metadata"), id="value-error"),
        pytest.param(FileNotFoundError("missing output"), id="missing-file"),
        pytest.param(WorldsStorageError("archive integrity check failed"), id="worlds-storage-error"),
    ],
)
@_run_async_test
async def test_known_publication_validation_errors_are_terminal(tmp_path, error):
    h = WorldsHarness(tmp_path)
    h.publication_error = error
    async with h.client() as client:
        request = _payload("create", description="A broken output")
        await client.post("/api/v1/worlds/requests", json=request)
        h.jobs["job-1"].update(status="completed", output_files=["bad.mp4", "bad.wgpworld"])

        first = await client.get(f"/api/v1/worlds/requests/{request['client_request_id']}")
        assert first.json()["state"] == "verification_failed"
        assert first.json()["detail"]
        assert len(first.json()["detail"]) <= 400

        again = await client.get(f"/api/v1/worlds/requests/{request['client_request_id']}")
        duplicate = await client.post("/api/v1/worlds/requests", json=request)
        assert again.json()["state"] == duplicate.json()["state"] == "verification_failed"
        assert len(h.publish_calls) == 1, "known invalid output is not retried"
        assert len(h.submit_calls) == 1, "duplicate requests never create another job"


@_run_async_test
async def test_prequeue_rejection_is_durable_and_does_not_reconcile_or_retry(tmp_path):
    h = WorldsHarness(tmp_path)
    h.rejection_detail = "Model preflight rejected the request.\nNo job was created."
    async with h.client() as client:
        request = _payload("create", description="A rejected output")
        accepted = await client.post("/api/v1/worlds/requests", json=request)
        assert accepted.json()["state"] == "failed"
        assert accepted.json()["detail"] == "Model preflight rejected the request. No job was created."
        assert h.lookup_calls == 0, "a matching prequeue rejection is already a conclusive outcome"

        again = await client.get(f"/api/v1/worlds/requests/{request['client_request_id']}")
        duplicate = await client.post("/api/v1/worlds/requests", json=request)
        assert again.json()["state"] == duplicate.json()["state"] == "failed"
        assert len(h.submit_calls) == 1
        assert len(h.publish_calls) == 0
        assert h.lookup_calls == 0

        h.rejection_detail = "x" * 500
        bounded_request = _payload("create", description="A rejected output with a long reason")
        bounded = await client.post("/api/v1/worlds/requests", json=bounded_request)
        assert bounded.json()["state"] == "failed"
        assert len(bounded.json()["detail"]) == 400
        assert len(h.submit_calls) == 2


@_run_async_test
async def test_unknown_publication_error_stays_get_reconcilable_without_resubmit(tmp_path):
    h = WorldsHarness(tmp_path)
    h.publication_error = RuntimeError("transient publisher failure")
    h.publication_error_once = True
    async with h.client() as client:
        request = _payload("create", description="A temporarily unavailable output")
        await client.post("/api/v1/worlds/requests", json=request)
        h.jobs["job-1"].update(status="completed", output_files=["world.mp4", "world.wgpworld"])

        pending = await client.get(f"/api/v1/worlds/requests/{request['client_request_id']}")
        assert pending.json()["state"] == "publication_pending"
        assert len(h.publish_calls) == 1

        recovered = await client.get(f"/api/v1/worlds/requests/{request['client_request_id']}")
        assert recovered.json()["state"] == "published"
        assert len(h.publish_calls) == 2, "a later GET can retry the idempotent publication"
        assert len(h.submit_calls) == 1, "unknown publication errors never authorize another submission"


@_run_async_test
async def test_traversal_and_license_boundaries(tmp_path):
    h = WorldsHarness(tmp_path)
    async with h.client() as client:
        bad_id = await client.get("/api/v1/worlds/..%2Foutside")
        assert bad_id.status_code == 404
        no_ack = await client.post("/api/v1/worlds/world-1/save", json={"acknowledge_pixelumm_noncommercial": False})
        assert no_ack.status_code == 403
        no_ack_record = await client.post("/api/v1/worlds/world-1/record", json={})
        assert no_ack_record.status_code == 403

    outside = tmp_path.parent / "outside.wgpworld"
    outside.write_bytes(b"outside")
    h.source = outside
    async with h.client() as client:
        move = _payload("move", world_id="world-1", action="Wait", duration_ms=800)
        response = await client.post("/api/v1/worlds/requests", json=move)
        assert response.status_code == 404
        assert h.submit_calls == []


@_run_async_test
async def test_recording_is_unavailable_without_a_saved_clip_chain(tmp_path):
    h = WorldsHarness(tmp_path)
    h.worlds["world-1"]["recording_available"] = False
    async with h.client() as client:
        response = await client.post(
            "/api/v1/worlds/world-1/record",
            json={"acknowledge_pixelumm_noncommercial": True},
        )
    assert response.status_code == 409
    assert h.record_thread_id is None
