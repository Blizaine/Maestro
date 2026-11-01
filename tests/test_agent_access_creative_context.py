from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from services.agent_access.context import principal_context
from services.agent_access.creative_context import CreativeError, CreativeService, compatible_workflows, normalize_defaults
from services.agent_access.creative_routes import create_creative_router


def service(tmp_path, *, clock=lambda: 0, session=lambda: "session-1"):
    rows = [
        {"model_type": "minimax_h3_pika_flex", "name": "Pika Frames", "architecture": "minimax_h3", "is_t2v": True, "fps": 24, "is_downloaded": True},
        {"model_type": "minimax_h3_ref2va_pika_flex", "name": "Pika References", "architecture": "minimax_h3_ref2va", "is_t2v": True, "supports_ref_images": True, "fps": 24, "is_downloaded": True},
        {"model_type": "flux2_klein_9b", "name": "Flux", "family": "flux2", "is_downloaded": True},
        {"model_type": "yue2", "name": "YuE2", "family": "tts", "is_downloaded": True},
        {"model_type": "fish_drama3_int4", "name": "Fish", "family": "tts", "is_downloaded": True},
        {"model_type": "missing_image", "name": "Uninstalled", "family": "flux2", "is_downloaded": False},
    ]
    def model_def(mt):
        row = next(row for row in rows if row["model_type"] == mt)
        return {"architecture": row.get("architecture", mt), "fps": row.get("fps", 24), "frames_minimum": 124, "frames_steps": 17, "frames_maximum": 362, "image_outputs": row.get("family") == "flux2", "audio_only": row.get("family") == "tts", "image_ref_choices": {"selection": ["identity"]} if "ref2va" in mt else None}
    def defaults(mt):
        return {"model_type": mt, "video_length": 124, "sliding_window_size": 124, "sliding_window_overlap": 0, "duration_seconds": 120 if mt == "yue2" else 60, "resolution": "832x480", "prompt": "OLD PERSONAL PROMPT", "image_start": "/private/old-picture.png", "audio_guide2": "/private/old-voice.wav", "seed": 17, "activated_loras": ["old.safetensors"], "batch_size": 6, "repeat_generation": 3}
    return CreativeService(
        tmp_path, models=lambda: {"models": rows}, model_def=model_def,
        defaults=defaults, options=lambda mt: {"speech_auto_duration": mt.startswith("fish")},
        preferences=lambda: {"selected_model_per_mode": {"video": "minimax_h3_ref2va_pika_flex", "image": "flux2_klein_9b"}, "selected_model_per_audio_sub_mode": {"speech": "fish_drama3_int4", "music": "yue2"}},
        guides=lambda *args: {"format_guide": "Native guide", "guide_revision": "guide-1"},
        clock=clock, session=session,
    )


def test_model_defaults_follow_compatible_workflow_without_old_creative_inputs(tmp_path):
    value = service(tmp_path).create_context({"workflow": "video.frames", "brief": "Anime karate in a dojo"}, "agent-a")
    assert value["model_type"] == "minimax_h3_pika_flex"
    assert value["timeline"]["total_frames"] == 124
    assert value["timeline"]["windows"][0]["end_seconds"] == pytest.approx(5.167, abs=.001)
    assert value["brief"] == "Anime karate in a dojo"
    assert value["local_writer"] is False
    assert "model_def" not in value
    for key in ("prompt", "image_start", "audio_guide2", "seed", "activated_loras"):
        assert key not in value["params"]
    assert value["params"]["batch_size"] == value["params"]["repeat_generation"] == 1
    assert not (tmp_path / "creative-defaults.json").exists()


def test_explicit_agent_defaults_override_selection_and_do_not_load_missing_models(tmp_path):
    creator = service(tmp_path)
    creator.save_defaults({"video.frames": {"model_type": "minimax_h3_pika_flex", "resolution": "1280x704"}})
    value = creator.create_context({"workflow": "video.frames", "brief": "A dojo"}, "a")
    assert value["selection_source"] == "agent_default"
    assert value["params"]["resolution"] == "1280x704"
    with pytest.raises(CreativeError, match="not downloaded"):
        creator.create_context({"workflow": "image.generate", "model_type": "missing_image"}, "a")
    with pytest.raises(CreativeError, match="does not support"):
        creator.create_context({"workflow": "audio.music", "model_type": "flux2_klein_9b"}, "a")


def test_speech_auto_and_song_cap_are_distinct(tmp_path):
    creator = service(tmp_path)
    speech = creator.create_context({"workflow": "audio.speech", "brief": "Tell a complete story"}, "a")
    assert speech["params"]["duration_seconds"] == 0
    assert speech["params"]["multi_prompts_gen_type"] == 2
    song = creator.create_context({"workflow": "audio.music", "brief": "Karate song"}, "a")
    assert song["params"]["duration_seconds"] == 120
    creator.save_defaults({"audio.speech": {"duration_seconds": 600}})
    assert creator.create_context({"workflow": "audio.speech"}, "a")["params"]["duration_seconds"] == 600


def test_context_and_prepared_revisions_are_owned_bounded_and_expire(tmp_path):
    now, revision = [0], ["one"]
    creator = service(tmp_path, clock=lambda: now[0], session=lambda: revision[0])
    context = creator.create_context({"workflow": "image.generate"}, "a")
    pid = creator.remember_prepared(context["context_id"], "a", "studio", {"params": {"prompt": "Exact script"}})
    with pytest.raises(CreativeError, match="unavailable"):
        creator.prepared(pid, "b")
    kind, _, prepared = creator.prepared(pid, "a")
    assert kind == "studio"
    prepared["params"]["prompt"] = "Tampered"
    assert creator.prepared(pid, "a")[2]["params"]["prompt"] == "Exact script"
    revision[0] = "two"
    with pytest.raises(CreativeError, match="restarted"):
        creator.prepared(pid, "a")
    revision[0] = "one"
    now[0] = 1801
    with pytest.raises(CreativeError, match="expired"):
        creator.lookup(context["context_id"], "a")


@pytest.mark.parametrize("value", [None, [], {"unknown": {}}, {"audio.speech": {"duration_seconds": True}}, {"audio.music": {"duration_seconds": 0}}, {"image.generate": {"duration_seconds": 5}}, {"video.frames": {"resolution": "bad"}}])
def test_invalid_defaults_fail_without_touching_other_settings(value):
    with pytest.raises(CreativeError):
        normalize_defaults(value)


def test_image_edit_ability_does_not_hide_text_generation_but_native_overrides_do():
    model = {"model_type": "fibo2_dev", "family": "fibo2"}
    definition = {"image_outputs": True, "instruction_image_edit": True, "image_ref_choices": {"choices": [["No image", ""], ["Reference", "I"]]}}
    assert compatible_workflows(model, definition) == ["image.generate", "image.edit"]
    assert compatible_workflows(model, {**definition, "capability_overrides": {"text_to_image": False}}) == ["image.edit"]
    assert compatible_workflows(model, {**definition, "at_least_one_image_ref_needed": True}) == ["image.edit"]


def test_prepare_and_submit_are_separate_and_rejections_are_saved_for_the_author(tmp_path):
    async def run():
        creator = service(tmp_path)
        calls = []
        backend = SimpleNamespace(
            app_root=tmp_path, additional_media_roots=lambda: (),
            runtime=SimpleNamespace(directory=tmp_path, audit=lambda *args: None),
            asset_store=SimpleNamespace(resolve_tree=lambda actor, value: value),
        )
        def native_prepare(params, *, prepare_only):
            calls.append((prepare_only, copy.deepcopy(params)))
            return {"params": copy.deepcopy(params), "workspace": "Test"} if prepare_only else {"job_id": "owned-job"}
        app = FastAPI()
        @app.middleware("http")
        async def identity(request, next_handler):
            token = principal_context.set(SimpleNamespace(id=request.headers.get("x-test-actor", "a"), owner=False, scopes=("observe", "generate")))
            try:
                return await next_handler(request)
            finally:
                principal_context.reset(token)
        app.include_router(create_creative_router(backend, creator, native_prepare=native_prepare, native_director=None))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            ctx = (await client.post("/api/v1/agent-access/creative-context", json={"workflow": "video.frames", "brief": "Karate in anime style"})).json()
            assert not calls
            plan = {"context_id": ctx["context_id"], "prompt": "An anime karate practice, no speech."}
            prepared = await client.post("/api/v1/agent-access/prepare-authored", json={"context_id": ctx["context_id"], "plan": plan})
            assert prepared.status_code == 200, prepared.text
            result = prepared.json()
            assert result["submitted"] is False
            assert [entry[0] for entry in calls] == [True]
            denied = await client.post("/api/v1/agent-access/submit-authored", json={"prepared_id": result["prepared_id"]}, headers={"x-test-actor": "b"})
            assert denied.status_code == 409
            submitted = await client.post("/api/v1/agent-access/submit-authored", json={"prepared_id": result["prepared_id"]})
            assert submitted.json()["job_id"] == "owned-job"
            assert [entry[0] for entry in calls] == [True, False]
            assert calls[-1][1]["prompt"] == plan["prompt"]
            assert calls[-1][1]["prompt_enhancer"] == ""
            bad = await client.post("/api/v1/agent-access/prepare-authored", json={"context_id": ctx["context_id"], "plan": {"context_id": "wrong", "prompt": "Keep this rejected draft"}})
            assert bad.status_code == 422
            draft = bad.json()["detail"]["draft_id"]
            readback = await client.get("/api/v1/agent-access/authoring-drafts/" + draft)
            assert readback.json()["plan"]["prompt"] == "Keep this rejected draft"
            assert (await client.get("/api/v1/agent-access/authoring-drafts/" + draft, headers={"x-test-actor": "b"})).status_code == 404
    asyncio.run(run())


def test_authored_submit_uses_durable_boundary_receipt_on_retry(tmp_path):
    from services.agent_access.auth import Principal
    from services.agent_access.boundary import AccessBoundary
    from services.agent_access.jobs import SubmissionJournal

    async def run():
        creator = service(tmp_path)
        principal = Principal("grant-a", "client-a", ("observe", "generate"), False)
        runtime = SimpleNamespace(
            directory=tmp_path, origin="https://maestro.test", local_url="https://maestro.test/mcp", security_required=True,
            auth=SimpleNamespace(enabled=True, authenticate=lambda token: principal),
            audit=lambda *args: None, public_status=lambda: {"controller_url": "https://maestro.test", "local_setup_url": "https://maestro.test"},
        )
        backend = SimpleNamespace(app_root=tmp_path, additional_media_roots=lambda: (), runtime=runtime, asset_store=SimpleNamespace(resolve_tree=lambda actor, value: value))
        submitted = []
        def native(params, *, prepare_only):
            if prepare_only:
                return {"params": params, "workspace": "Test"}
            submitted.append(copy.deepcopy(params))
            return {"job_id": "owned-once"}
        app = FastAPI()
        app.include_router(create_creative_router(backend, creator, native_prepare=native, native_director=None))
        boundary = AccessBoundary(app, runtime=runtime, journal=SubmissionJournal(tmp_path / "journal"), app_root=tmp_path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=boundary), base_url="https://maestro.test", headers={"Authorization": "Bearer test-token"}) as client:
            context_response = await client.post("/api/v1/agent-access/creative-context", json={"workflow": "image.generate"}, headers={"Idempotency-Key": "context"})
            assert context_response.status_code == 200, context_response.text
            context_id = context_response.json()["context_id"]
            prepared = await client.post("/api/v1/agent-access/prepare-authored", json={"context_id": context_id, "plan": {"context_id": context_id, "prompt": "An illustrative dojo"}}, headers={"Idempotency-Key": "prepare"})
            assert prepared.status_code == 200, prepared.text
            body = {"prepared_id": prepared.json()["prepared_id"]}
            first = await client.post("/api/v1/agent-access/submit-authored", json=body, headers={"Idempotency-Key": "submit-once"})
            repeated = await client.post("/api/v1/agent-access/submit-authored", json=body, headers={"Idempotency-Key": "submit-once"})
            assert first.status_code == repeated.status_code == 200
            assert first.json() == repeated.json() == {"job_id": "owned-once"}
            assert len(submitted) == 1
    asyncio.run(run())
