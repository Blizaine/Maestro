"""Model-backend integration contracts without importing launch.py or models."""

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import io


_APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(_APP))

from fastapi import FastAPI, HTTPException  # noqa: E402
from services.agent_access.auth import Principal  # noqa: E402
from services.agent_access.backend import (  # noqa: E402
    AgentAccessBackend,
    create_router,
    strip_private_director_fields,
)
from services.agent_access.boundary import AccessBoundary  # noqa: E402
from services.agent_access.context import principal_context  # noqa: E402
from services.agent_access.jobs import JournalError, SubmissionJournal  # noqa: E402
from services.agent_access.state import RuntimeState  # noqa: E402


def _route_endpoint(router, path, method="GET"):
    for route in router.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", {method}):
            endpoint = getattr(route, "endpoint", None)
            if endpoint is not None:
                return endpoint
    raise AssertionError(f"No direct {method} endpoint for {path}")


class FakeStore:
    def __init__(self, root: Path, stage: Path | None = None):
        self.root = root
        self.stage = stage
        self.registered = None

    def resolve(self, principal_id, reference):
        if principal_id != "actor-a" or reference != "asset:asset-a":
            from services.agent_access.assets import AssetStoreError
            raise AssetStoreError(404, "Asset not found")
        return self.root / "uploads" / "clip.mp4"

    def resolve_tree(self, principal_id, payload):
        if isinstance(payload, dict):
            return {key: self.resolve_tree(principal_id, value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self.resolve_tree(principal_id, value) for value in payload]
        if isinstance(payload, str) and payload.startswith(("asset:", "maestro-asset:")):
            return str(self.resolve(principal_id, payload))
        return payload

    def reference_for_path(self, principal_id, backend_path):
        if principal_id == "actor-a" and Path(backend_path) == self.root / "uploads" / "clip.mp4":
            return "asset:asset-a"
        return None

    def finish(self, principal_id, upload_id):
        if principal_id != "actor-a" or upload_id != "upload-a":
            from services.agent_access.assets import AssetStoreError
            raise AssetStoreError(404, "Upload not found")
        return {"filename": "clip.mp4"}, self.stage

    def register(self, principal_id, upload_id, path):
        self.registered = (principal_id, upload_id, path)
        return "asset:asset-a"


class FakeRuntime:
    def __init__(self, directory: Path, registration=None):
        self.directory = directory
        self.registration = registration or {}

    def backend(self):
        return dict(self.registration)


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class AgentAccessBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = self.root / "settings"
        self.settings.mkdir()
        self.app_root = self.root / "app"
        for name in ("uploads", "outputs", "storage"):
            (self.app_root / name).mkdir(parents=True)
        self.runtime = FakeRuntime(self.settings / "agent-access")
        self.journal = SubmissionJournal(self.settings / "agent-access")
        self.store = FakeStore(self.app_root)
        self.backend = AgentAccessBackend(
            self.runtime, self.app_root, session_id="session-current",
            journal=self.journal, asset_store=self.store,
        )
        self.actor = Principal("actor-a", "client-a", ("observe", "generate", "manage"))
        self.other = Principal("actor-b", "client-b", ("observe", "generate", "manage"))
        self.admin = Principal("admin-a", "client-admin", ("admin",))

    def test_first_owned_assignment_is_durable_before_dispatch_and_immutable(self):
        jobs = self.backend.registry()
        token = principal_context.set(self.actor)
        try:
            jobs["job-a"] = {
                "id": "job-a", "status": "queued", "prompt": "not persisted",
                "params": {"prompt": "private request text", "api_key": "private-key"},
                "created_at": 1,
            }
        finally:
            principal_context.reset(token)

        record = self.journal.get_job("job-a", "actor-a")
        self.assertIsNotNone(record)
        self.assertEqual(record["session_id"], "session-current")
        self.assertNotIn("params", record["snapshot"])
        self.assertNotIn("private-key", str(record))
        self.assertTrue(self.backend.can_access_job("job-a", self.actor))
        self.assertFalse(self.backend.can_access_job("job-a", self.other))

        with self.assertRaises(JournalError) as reassigned:
            self.journal.record_job(jobs["job-a"], "actor-b", "session-other")
        self.assertEqual(reassigned.exception.status, 409)

    def test_trusted_pipeline_owner_is_captured_before_child_insert(self):
        self.backend.bind_pipeline_owner("pipeline-a", "actor-a")
        jobs = self.backend.registry()
        jobs["child-a"] = {
            "id": "child-a", "status": "queued",
            "params": {"_director_pipeline_id": "pipeline-a"},
        }
        record = self.journal.get_job("child-a", "actor-a")
        self.assertIsNotNone(record)
        self.assertEqual(record["session_id"], "session-current")
        self.assertEqual(jobs.ownership_for("child-a")["principal_id"], "actor-a")

    def test_child_of_existing_pipeline_uses_current_session_without_reassigning_pipeline(self):
        self.journal.register_resource_owner("pipeline", "pipeline-old", "actor-a", "session-old")
        jobs = self.backend.registry()
        resolved = self.backend.resolve_background_job_owner(
            "child-new", {"id": "child-new", "params": {"_director_pipeline_id": "pipeline-old"}},
        )
        self.assertEqual(resolved, {"principal_id": "actor-a", "session_id": "session-current"})
        jobs["child-new"] = {
            "id": "child-new", "status": "queued",
            "params": {"_director_pipeline_id": "pipeline-old"},
        }
        child = self.journal.get_job("child-new", "actor-a")
        pipeline = self.journal.resource_owner("pipeline", "pipeline-old")
        self.assertEqual(child["session_id"], "session-current")
        self.assertEqual(pipeline["session_id"], "session-old")

    def test_resource_owner_and_session_cannot_be_reassigned(self):
        self.backend.bind_queue_owner("queue-a", "actor-a")
        with self.assertRaises(JournalError) as reassigned:
            self.journal.register_resource_owner("queue", "queue-a", "actor-b", "session-b")
        self.assertEqual(reassigned.exception.status, 409)

    def test_dead_session_recovery_marks_pending_only_and_restores_owner(self):
        old = SubmissionJournal(self.settings / "agent-access")
        old.record_job({"id": "pending-a", "status": "running"}, "actor-a", "session-old")
        old.record_job({"id": "done-a", "status": "completed"}, "actor-a", "session-old")
        registration = {
            "pid": 43210, "session_id": "session-old", "url": "http://127.0.0.1:7860",
        }
        runtime = FakeRuntime(self.settings / "agent-access", registration)
        recovered = AgentAccessBackend(
            runtime, self.app_root, session_id="session-new", journal=old,
            asset_store=self.store, pid_checker=lambda _pid: False,
        )
        records = {item["job_id"]: item for item in old.list_jobs("actor-a")}
        self.assertEqual(records["pending-a"]["snapshot"]["status"], "interrupted")
        self.assertEqual(records["done-a"]["snapshot"]["status"], "completed")

        registry = recovered.registry({
            "pending-a": {"id": "pending-a", "status": "held"},
            "done-a": {"id": "done-a", "status": "completed"},
        })
        self.assertEqual(registry["pending-a"]["status"], "interrupted")
        self.assertEqual(registry.ownership_for("pending-a")["session_id"], "session-old")

    def test_live_or_unverifiable_pid_refuses_takeover(self):
        journal = SubmissionJournal(self.settings / "agent-access")
        journal.record_job({"id": "pending", "status": "running"}, "actor-a", "session-live")
        runtime = FakeRuntime(self.settings / "agent-access", {
            "pid": 42, "session_id": "session-live", "url": "http://127.0.0.1:7860",
        })
        with self.assertRaises(RuntimeError):
            AgentAccessBackend(
                runtime, self.app_root, journal=journal, asset_store=self.store,
                pid_checker=lambda _pid: True,
            )
        self.assertEqual(journal.get_job("pending", "actor-a")["snapshot"]["status"], "running")

    def test_request_private_director_markers_are_removed_recursively(self):
        cleaned = strip_private_director_fields({
            "_director_pipeline_id": "forged",
            "nested": {"_agent_access_owner_id": "forged", "value": 3},
        })
        self.assertEqual(cleaned, {"nested": {"value": 3}})

    def test_agent_router_uses_durable_owner_prepare_asset_import_and_admin_idle(self):
        jobs = self.backend.registry()
        token = principal_context.set(self.actor)
        try:
            jobs["job-a"] = {"id": "job-a", "status": "completed", "output_dir": str(self.app_root)}
        finally:
            principal_context.reset(token)

        stage = self.root / "clip.mp4"
        stage.write_bytes(b"verified bytes")
        self.store.stage = stage
        native_path = self.app_root / "uploads" / "native.mp4"

        async def native_upload(request, file, *, reuse_identical):
            self.assertEqual(request.headers.get("content-length"), str(stage.stat().st_size))
            self.assertTrue(reuse_identical)
            native_path.write_bytes(await file.read())
            return {"path": str(native_path)}

        async def prepare(body, *, prepare_only):
            self.assertTrue(prepare_only)
            self.assertNotIn("_director_pipeline_id", body)
            return {
                "params": body,
                "out_dir": str(self.app_root / "outputs"),
                "workspace": "default",
            }

        router = create_router(
            self.backend,
            prepare_generation=prepare,
            native_upload=native_upload,
            idle_probe=lambda: {"idle": True, "certain": True, "reason": "No active work"},
        )
        endpoints = {
            getattr(route, "path", None): getattr(route, "endpoint", None)
            for route in router.routes
            if getattr(route, "endpoint", None) is not None
        }

        token = principal_context.set(self.actor)
        try:
            durable = endpoints["/api/v1/agent-access/jobs/{job_id}"]("job-a")
            self.assertEqual(durable["job"]["status"], "completed")
            self.assertNotIn("output_dir", durable["job"])

            prepared = asyncio.run(endpoints["/api/v1/agent-access/prepare"](
                FakeRequest({
                    "model_type": "model-a", "prompt": "Keep this prompt",
                    "reference_video": "asset:asset-a",
                    "_director_pipeline_id": "forged",
                })
            ))
            self.assertEqual(prepared["params"]["prompt"], "Keep this prompt")
            self.assertEqual(prepared["params"]["reference_video"], "asset:asset-a")
            self.assertEqual(prepared["out_dir"], str(self.app_root / "outputs"))

            imported = asyncio.run(endpoints["/api/v1/agent-access/assets/uploads/{upload_id}/import"]("upload-a"))
            self.assertEqual(imported, {"reference": "asset:asset-a"})
            self.assertEqual(self.store.registered, ("actor-a", "upload-a", str(native_path)))

            with self.assertRaises(HTTPException) as denied:
                asyncio.run(endpoints["/api/v1/agent-access/idle"]())
            self.assertEqual(denied.exception.status_code, 403)
        finally:
            principal_context.reset(token)

        reserved = self.journal.reserve(
            "actor-a", "request-recovery-a", "POST", "/api/v1/jobs",
            {"prompt": "never return request details"},
        )
        self.journal.complete(reserved["reservation_id"], 202, b'{"job_id":"job-a"}')
        submission_endpoint = _route_endpoint(router, "/api/v1/agent-access/submissions")
        token = principal_context.set(self.actor)
        try:
            receipt = submission_endpoint("request-recovery-a")
            self.assertEqual(receipt["result"], {"job_id": "job-a"})
            self.assertNotIn("body", receipt)
            with self.assertRaises(HTTPException) as hidden:
                submission_endpoint("request-from-another-principal")
            self.assertEqual(hidden.exception.status_code, 404)
        finally:
            principal_context.reset(token)

        token = principal_context.set(self.other)
        try:
            with self.assertRaises(HTTPException) as cross_principal:
                submission_endpoint("request-recovery-a")
            self.assertEqual(cross_principal.exception.status_code, 404)
        finally:
            principal_context.reset(token)

        token = principal_context.set(self.admin)
        try:
            idle = asyncio.run(endpoints["/api/v1/agent-access/idle"]())
            self.assertEqual(idle["idle"], True)
            self.assertTrue(idle["certain"])
        finally:
            principal_context.reset(token)

    def test_prepare_round_trips_boundary_resolved_assets_and_preserves_long_speech(self):
        router = create_router(
            self.backend,
            prepare_generation=lambda body, *, prepare_only: {
                "params": body,
                "output_dir": str(self.app_root / "outputs"),
                "api_key": "must not be returned",
            },
            native_upload=lambda *_args, **_kwargs: None,
            idle_probe=lambda: {"idle": True, "certain": True, "reason": "No active work"},
        )
        endpoint = next(
            getattr(route, "endpoint") for route in router.routes
            if getattr(route, "path", None) == "/api/v1/agent-access/prepare"
            and getattr(route, "endpoint", None) is not None
        )
        speech = "A long spoken line. " * 1200
        token = principal_context.set(self.actor)
        try:
            prepared = asyncio.run(endpoint(FakeRequest({
                "speech": speech,
                "reference_video": str(self.app_root / "uploads" / "clip.mp4"),
            })))
        finally:
            principal_context.reset(token)

        self.assertEqual(prepared["params"]["speech"], speech)
        self.assertEqual(prepared["params"]["reference_video"], "asset:asset-a")
        self.assertEqual(prepared["output_dir"], str(self.app_root / "outputs"))
        self.assertNotIn("api_key", prepared)

    def test_prepare_rejects_unapproved_local_paths_without_invalid_sentinel(self):
        router = create_router(
            self.backend,
            prepare_generation=lambda _body, *, prepare_only: {
                "params": {"image_path": str(self.root / "outside.png")},
            },
            native_upload=lambda *_args, **_kwargs: None,
            idle_probe=lambda: {"idle": True, "certain": True, "reason": "No active work"},
        )
        endpoint = next(
            getattr(route, "endpoint") for route in router.routes
            if getattr(route, "path", None) == "/api/v1/agent-access/prepare"
            and getattr(route, "endpoint", None) is not None
        )
        token = principal_context.set(self.actor)
        try:
            with self.assertRaises(HTTPException) as denied:
                asyncio.run(endpoint(FakeRequest({"prompt": "safe"})))
        finally:
            principal_context.reset(token)
        self.assertEqual(denied.exception.status_code, 422)

    def test_prepare_accepts_only_server_configured_additional_output_roots(self):
        configured_root = self.root / "configured-save-path"
        configured_root.mkdir()
        backend = AgentAccessBackend(
            self.runtime, self.app_root, session_id="session-current",
            journal=self.journal, asset_store=self.store,
            additional_media_roots=lambda: [configured_root],
        )
        router = create_router(
            backend,
            prepare_generation=lambda _body, *, prepare_only: {
                "out_dir": str(configured_root / "render-1"),
            },
            native_upload=lambda *_args, **_kwargs: None,
            idle_probe=lambda: {"idle": True, "certain": True, "reason": "No active work"},
        )
        endpoint = _route_endpoint(router, "/api/v1/agent-access/prepare", "POST")
        token = principal_context.set(self.actor)
        try:
            prepared = asyncio.run(endpoint(FakeRequest({"prompt": "safe"})))
        finally:
            principal_context.reset(token)
        self.assertEqual(prepared["out_dir"], str(configured_root / "render-1"))

    def test_native_management_status_and_login_routes_are_public_default_off(self):
        import httpx

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app_root = root / "app"
            app_root.mkdir()
            for name in ("uploads", "outputs", "storage"):
                (app_root / name).mkdir()
            runtime = RuntimeState(root / "settings" / "agent-access", app_root=app_root)
            journal = SubmissionJournal(runtime.directory)
            backend = AgentAccessBackend(
                runtime, app_root, session_id="session-asgi", journal=journal,
            )
            router = create_router(
                backend,
                prepare_generation=lambda body, *, prepare_only: {"params": body},
                native_upload=lambda *_args, **_kwargs: {"path": str(app_root / "uploads" / "x")},
                idle_probe=lambda: {"idle": True, "certain": True, "reason": "No active work"},
            )
            app = FastAPI()
            app.include_router(router)
            app = AccessBoundary(app, runtime=runtime, journal=journal, app_root=app_root)

            captured = io.StringIO()

            async def request_public_routes():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://maestro.test") as client:
                    status = await client.get("/api/v1/agent-access/status")
                    login = await client.get("/agent-access/login")
                    return status, login

            with redirect_stdout(captured):
                status, login = asyncio.run(request_public_routes())

        self.assertEqual(status.status_code, 200)
        self.assertFalse(status.json()["enabled"])
        self.assertTrue(status.json()["available"])
        self.assertFalse(status.json()["authenticated"])
        self.assertIn("mcp_url", status.json())
        self.assertEqual(login.status_code, 200)
        self.assertIn("Open Maestro securely", login.text)
        self.assertEqual(captured.getvalue(), "")

    def test_idle_evidence_is_false_or_uncertain_when_busy_state_is_present_or_missing(self):
        class Locked:
            def __init__(self, value):
                self.value = value

            def locked(self):
                return self.value

        idle = AgentAccessBackend.idle_evidence(
            jobs={}, engine_lock=Locked(False), active_states={},
            director_pipelines=[], director_queue={"entries": []},
        )
        self.assertEqual(idle["idle"], True)
        busy = AgentAccessBackend.idle_evidence(
            jobs={"job": {"status": "queued"}}, engine_lock=Locked(False), active_states={},
            director_pipelines=[], director_queue={"entries": []},
        )
        self.assertEqual(busy["idle"], False)
        self.assertTrue(busy["certain"])
        uncertain = AgentAccessBackend.idle_evidence(
            jobs={}, engine_lock=None, active_states={},
            director_pipelines=[], director_queue={"entries": []},
        )
        self.assertEqual(uncertain, {"idle": False, "certain": False, "reason": "Generation lock state is unavailable"})
        unknown_job = AgentAccessBackend.idle_evidence(
            jobs={"job": {"status": "preparing"}}, engine_lock=Locked(False), active_states={},
            director_pipelines=[], director_queue={"entries": []},
        )
        self.assertFalse(unknown_job["idle"])
        self.assertFalse(unknown_job["certain"])


if __name__ == "__main__":
    unittest.main()
