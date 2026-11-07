from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
from fastapi import HTTPException

from services.agent_access.creative_context import CreativeError, CreativeService
from services.agent_access.production_routes import ProductionService, fingerprint
from services.production_finishing import build_finish_project
from services.production_review import BudgetExceededError, RequestCollisionError
from services.agent_access.state import atomic_json


def creator(tmp_path):
    def forbidden(*args):
        raise AssertionError("A CPU title context must not inspect/load a model or writer")
    return CreativeService(tmp_path, models=forbidden, model_def=forbidden,
                           defaults=forbidden, options=forbidden, preferences=forbidden,
                           guides=forbidden, session=lambda: "session", resolve=forbidden)


def test_title_context_is_model_free_bound_and_does_not_change_defaults(tmp_path):
    service = creator(tmp_path)
    context = service.create_context({"workflow": "title.card", "brief": "An anime end card", "duration_seconds": 2, "resolution": "640x360"}, "one")
    assert context["timeline"]["total_frames"] == 48
    assert context["local_writer"] is False
    assert context["params"]["width"] == 640
    assert service.lookup(context["context_id"], "one")["workflow"] == "title.card"
    with pytest.raises(CreativeError):
        service.lookup(context["context_id"], "two")
    assert not (tmp_path / "creative-defaults.json").exists()


@pytest.mark.parametrize("body", [{"duration_seconds": True}, {"duration_seconds": 0}, {"resolution": "641x360"}, {"model_type": "some-model"}, {"references": [{"id": "person"}]}, {"params": {"prompt": "hidden"}}])
def test_title_context_rejects_unreviewed_geometry_and_character_refs(tmp_path, body):
    with pytest.raises(CreativeError):
        creator(tmp_path).create_context({"workflow": "title.card", **body}, "one")


def resolver(name):
    kind = "audio" if name.endswith(".wav") else "image" if name.endswith(".png") else "video"
    return {"name": name, "path": str(Path(name).absolute()), "type": kind, "duration": 8 if kind == "audio" else 3 if kind == "video" else 0,
            "width": 640, "height": 360, "fps": 24, "has_audio": kind != "image"}


def test_finish_keeps_order_and_audio_and_ducks_only_declared_narration():
    body = {"clips": [{"asset": "opening.png", "duration": 1}, {"asset": "take.mp4", "duration": 3}, {"asset": "end.png", "duration": 1}],
            "narration": [{"asset": "voice.wav", "start": 1, "duration": 2}],
            "music": [{"asset": "music.wav", "duration": 5}]}
    original = copy.deepcopy(body)
    project = build_finish_project(body, resolver)
    assert body == original
    assert [item["start"] for item in project["tracks"][0]["items"]] == [0, 1, 4]
    assert all(not item["muted"] for item in project["tracks"][0]["items"])
    assert project["export"]["encoder"] == "software"
    assert project["export"]["include_audio"] is True
    music = next(track for track in project["tracks"] if track["name"] == "Music")["items"][0]
    assert music["volume"] == pytest.approx(10 ** (-18 / 20))
    assert min(point["value"] for point in music["keyframes"]["volume"]) == pytest.approx(10 ** (-30 / 20))
    assert max(point["value"] for point in music["keyframes"]["volume"]) == pytest.approx(music["volume"])


@pytest.mark.parametrize("body", [
    {"clips": [{"asset": "take.mp4", "duration": 4}]},
    {"clips": [{"asset": "take.mp4"}], "music": [{"asset": "music.wav", "duration": 4}]},
    {"clips": [{"asset": "take.mp4", "include_audio": "false"}]},
    {"clips": [{"asset": "take.mp4"}], "speech_ranges": [[2, 1]]},
    {"clips": [{"asset": "take.mp4"}], "canvas": {"width": 641}},
])
def test_finish_never_silently_clips_authored_timing_or_changes_types(body):
    with pytest.raises(ValueError):
        build_finish_project(body, resolver)


def production(tmp_path):
    root = tmp_path / "app"
    outputs = root / "outputs"
    outputs.mkdir(parents=True)
    backend = SimpleNamespace(app_root=root, runtime=SimpleNamespace(directory=root / "settings"), jobs={}, additional_media_roots=lambda: (), snapshot_job=lambda _: None)
    owner = SimpleNamespace(id="agent-one", owner=False, scopes=("observe", "generate", "manage"))
    calls = []
    service = ProductionService(backend, output_root=lambda: outputs,
                                export_editor=lambda body: {"job_id": "export01"},
                                rerun_director=lambda *args: calls.append(args) or {"job_id": "retake01"},
                                director_snapshot=lambda pid: {"clips": [{"video_path": str(outputs / "take.mp4")}]},
                                idle_probe=lambda: {"idle": True, "certain": True})
    return service, owner, outputs, calls


def reviewed_take(service, owner, outputs):
    media = outputs / "take.mp4"
    media.write_bytes(b"owned original media")
    identifier = str(uuid.uuid4())
    directory = service.owner_dir(owner) / "reviews" / identifier
    directory.mkdir(parents=True)
    atomic_json(directory / "source.json", {"path": str(media), "fingerprint": fingerprint(media)})
    review = {"verdict": "needs_work", "summary": "The camera movement jumps at the end.", "observations": [{"category": "motion", "severity": "medium", "text": "The final shot jumps left immediately before the cut."}]}
    service.record_review({"review_id": identifier, "review": review}, owner)
    return identifier, media


def test_media_boundary_rejects_traversal_private_files_and_urls(tmp_path):
    service, owner, outputs, _ = production(tmp_path)
    private = service.backend.app_root / "settings" / "secret.wav"
    private.parent.mkdir(exist_ok=True)
    private.write_bytes(b"private")
    for value in ({"filename": "../secret.wav"}, str(private), "https://example.org/audio.wav"):
        with pytest.raises(ValueError):
            service.media(value, owner)
    media = outputs / "take.mp4"
    media.write_bytes(b"media")
    assert service.media({"filename": media.name}, owner) == media


def test_polish_is_owned_fingerprint_bound_idempotent_and_budgeted(tmp_path):
    service, owner, outputs, calls = production(tmp_path)
    review_id, media = reviewed_take(service, owner, outputs)
    budget_id = str(uuid.uuid4())
    service.budget(owner, budget_id, {"max_renders": 1, "max_retakes_per_clip": 1})
    body = {"budget_id": budget_id, "review_id": review_id, "pipeline_id": "pipeline1", "clip_index": 0, "request_id": "one-owned-request", "prompt": "Preserve the subject; use a smooth locked camera."}
    first = asyncio.run(service.polish(body, owner))
    second = asyncio.run(service.polish(body, owner))
    assert len(calls) == 1
    assert first["intent"]["receipt_id"] == "retake01"
    assert second["submitted_again"] is False
    with pytest.raises(RequestCollisionError):
        asyncio.run(service.polish({**body, "prompt": "A different instruction"}, owner))
    with pytest.raises(BudgetExceededError):
        asyncio.run(service.polish({**body, "request_id": "second-request"}, owner))
    media.write_bytes(b"a replacement take")
    with pytest.raises(HTTPException) as error:
        asyncio.run(service.polish(body, owner))
    assert error.value.status_code == 409


def test_foreign_review_and_busy_machine_do_not_submit_or_consume_new_budget(tmp_path):
    service, owner, outputs, calls = production(tmp_path)
    review_id, _ = reviewed_take(service, owner, outputs)
    foreign = SimpleNamespace(id="agent-two")
    with pytest.raises(HTTPException) as error:
        service.review_source(review_id, foreign)
    assert error.value.status_code == 404
    budget_id = str(uuid.uuid4())
    store = service.budget(owner, budget_id, {"max_renders": 2, "max_retakes_per_clip": 1})
    service.idle_probe = lambda: {"idle": False, "certain": True}
    with pytest.raises(HTTPException):
        asyncio.run(service.polish({"budget_id": budget_id, "review_id": review_id, "pipeline_id": "pipeline1", "clip_index": 0, "request_id": "deferred"}, owner))
    assert calls == []
    assert store.list_intents() == []


def test_title_prepare_submit_boundary_is_owned_and_idempotent(tmp_path):
    import httpx
    from fastapi import FastAPI
    from services.agent_access.auth import Principal
    from services.agent_access.boundary import AccessBoundary
    from services.agent_access.jobs import SubmissionJournal
    from services.agent_access.creative_routes import create_creative_router

    async def run():
        principal = Principal("grant-a", "client-a", ("observe", "generate"), False)
        runtime = SimpleNamespace(directory=tmp_path, origin="https://maestro.test",
            local_url="https://maestro.test/mcp", security_required=True,
            auth=SimpleNamespace(enabled=True, authenticate=lambda token: principal),
            audit=lambda *args: None, public_status=lambda: {"controller_url": "https://maestro.test", "local_setup_url": "https://maestro.test"})
        backend = SimpleNamespace(app_root=tmp_path, additional_media_roots=lambda: (), runtime=runtime,
                                  asset_store=SimpleNamespace(resolve_tree=lambda actor, value: value))
        calls = []
        def render(params, actor):
            calls.append(copy.deepcopy(params))
            return {"job_id": "card-once", "status": "completed"}
        app = FastAPI()
        app.include_router(create_creative_router(backend, creator(tmp_path), native_prepare=None,
            native_director=None, production=SimpleNamespace(create_title=render)))
        boundary = AccessBoundary(app, runtime=runtime, journal=SubmissionJournal(tmp_path / "journal"), app_root=tmp_path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=boundary), base_url="https://maestro.test",
                                     headers={"Authorization": "Bearer test"}) as client:
            context = await client.post("/api/v1/agent-access/creative-context", json={"workflow": "title.card", "duration_seconds": 2}, headers={"Idempotency-Key": "context"})
            assert context.status_code == 200, context.text
            context_id = context.json()["context_id"]
            prepared = await client.post("/api/v1/agent-access/prepare-authored", json={"context_id": context_id,
                "plan": {"context_id": context_id, "card": {"text": "MAESTRO", "role": "opening"}}}, headers={"Idempotency-Key": "prepare"})
            assert prepared.status_code == 200, prepared.text
            assert calls == []
            body = {"prepared_id": prepared.json()["prepared_id"]}
            first = await client.post("/api/v1/agent-access/submit-authored", json=body, headers={"Idempotency-Key": "once"})
            repeated = await client.post("/api/v1/agent-access/submit-authored", json=body, headers={"Idempotency-Key": "once"})
            assert first.status_code == repeated.status_code == 200, first.text
            assert first.json() == repeated.json()
            assert len(calls) == 1
            assert calls[0]["card"]["duration_seconds"] == 2
    asyncio.run(run())


def test_native_polish_job_has_pollable_snapshot_and_result():
    import ast
    import threading
    source = Path(__file__).resolve().parents[1] / "app/launch.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_agent_production_rerun")
    jobs = {}
    async def rerun(*args):
        return {"filename": "new-take.mp4"}
    class ImmediateThread:
        def __init__(self, *, target, **kwargs):
            self.target = target
        def start(self):
            self.target()
    namespace = {"_agent_require_director_pipeline": lambda pid: None,
        "_agent_access_idle_probe": lambda: {"idle": True, "certain": True},
        "uuid": uuid, "_jobs": jobs, "asyncio": asyncio, "HTTPException": HTTPException,
        "JSONResponse": type(None), "_AgentProductionRequest": lambda body: body,
        "rerun_pipeline_clip_video": rerun, "threading": SimpleNamespace(Thread=ImmediateThread),
        "_agent_access_backend": SimpleNamespace(snapshot_job=lambda job_id: None)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    result = namespace["_agent_production_rerun"]("owned-pipeline", 0, {})
    snapshot = jobs[result["job_id"]]
    assert snapshot["status"] == "completed"
    assert snapshot["error"] is None
    assert snapshot["output_files"] == ["new-take.mp4"]
    assert snapshot["params"]["_director_pipeline_id"] == "owned-pipeline"
    assert len(result["job_id"]) == 32
