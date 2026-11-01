"""Song-to-Director handoff using native adapters and mocked render dispatch."""
import asyncio
import copy
import sys
import threading
from types import SimpleNamespace

from fastapi import FastAPI
import httpx

from services.agent_access.context import principal_context
from services.agent_access.creative_routes import create_creative_router
from services.agent_access.http import scope_required
from services.agent_access.native_creative import make_native_creative_service


def setup_workflow(tmp_path, monkeypatch, analyze_override=None):
    from services.agent_access import authoring_plans
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    song = uploads / "generated-song.wav"
    song.write_bytes(b"rendered audio fixture")
    character = uploads / "character.png"
    character.write_bytes(b"image fixture")
    scene = uploads / "scene.png"
    scene.write_bytes(b"image fixture")
    monkeypatch.setitem(sys.modules, "models.minimax_h3.ref2va", SimpleNamespace(align_ref2va_voice_reference_order=lambda prompt, refs: (prompt, refs, {})))
    monkeypatch.setattr("services.h3_sequence_planner._reference_context", lambda refs: ("Reference identity and scene", "Preserve reference roles", "identity"))
    def resolve(context, request, hardware, **kwargs):
        if context["workflow"].startswith("director."):
            context["execution_profile"] = {"fps": 24, "frames_minimum": 124, "effective_max_frames": 243, "frame_step": 17}
            context["shot_image_policy"] = "direct_references"
    monkeypatch.setattr(authoring_plans, "resolve_native_context", resolve)
    backend = SimpleNamespace(
        app_root=tmp_path, session_id="fixture-session", additional_media_roots=lambda: (),
        runtime=SimpleNamespace(directory=tmp_path / "private", audit=lambda *args: None),
        asset_store=SimpleNamespace(resolve_tree=lambda actor, value: value),
    )
    model = "minimax_h3_ref2va_fused_turbo"
    analyzed, studio_calls, director_calls = [], [], []
    def analyze(path):
        assert scope_required("observe").id == "agent", "thread pool must retain the principal"
        analyzed.append(path)
        if analyze_override:
            analyze_override()
        return {"duration": 12.125, "bpm": 120, "beats": [{"time": i / 2, "strength": 1} for i in range(24)], "downbeats": [0, 4, 8], "sections": [{"start": 0, "end": 6, "label": "intro", "energy": 0.2}, {"start": 6, "end": 12.125, "label": "chorus", "energy": 0.8}], "music_cues": [{"time": 6, "type": "percussion_entry"}]}
    service = make_native_creative_service(
        backend, models=lambda: [{"model_type": "yue2", "is_downloaded": True}, {"model_type": model, "is_t2v": True, "is_downloaded": True}],
        model_def=lambda mt: {"audio_only": True, "group": "music"} if mt == "yue2" else {"architecture": "minimax_h3_ref2va", "director_video_strategy": "omni_reference", "omni_reference": True, "any_audio_prompt": True, "audio_guide_window_slicing": True, "fps": 24, "frames_minimum": 124, "frames_maximum": 243, "frames_steps": 17},
        defaults=lambda mt: {"video_length": 243}, options=lambda mt: {}, preferences=lambda: {}, hardware=lambda: {},
        lora_hint=lambda *args: "", injected_keyframes=lambda params: [], mature_mode=lambda: False,
        probe_audio=lambda path: 12.125, analyze_music=analyze,
    )
    def studio(params, *, prepare_only):
        studio_calls.append((prepare_only, copy.deepcopy(params)))
        return {"params": params} if prepare_only else {"job_id": "song-job", "audio_path": str(song)}
    def director(params):
        director_calls.append(copy.deepcopy(params))
        return {"pipeline_id": "reviewed-agent-pipeline"}
    app = FastAPI()
    @app.middleware("http")
    async def identity(request, next_handler):
        token = principal_context.set(SimpleNamespace(id="agent", owner=False, scopes=("observe", "generate")))
        try:
            return await next_handler(request)
        finally:
            principal_context.reset(token)
    @app.get("/ping")
    async def ping():
        return {"ready": True}
    app.include_router(create_creative_router(backend, service, native_prepare=studio, native_director=director))
    request = {"workflow": "director.music_video", "brief": "The robot finds a garden; illustrated style.", "params": {"audio_path": str(song)}, "references": [{"id": "hero", "type": "image", "role": "character", "path": str(character)}, {"id": "garden", "type": "image", "role": "scene", "path": str(scene)}]}
    return app, request, analyzed, studio_calls, director_calls, song


def test_song_guides_analysis_roles_and_authored_prompts_reach_exact_submission(tmp_path, monkeypatch):
    app, request, analyzed, studio_calls, director_calls, song = setup_workflow(tmp_path, monkeypatch)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/agent-access/creative-context", json={"workflow": "audio.music", "model_type": "yue2", "brief": "A song about a robot finding a garden."})
            assert response.status_code == 200, response.text
            music = response.json()
            assert music["guides"]["format_guide"]
            assert music["local_writer"] is False
            lyrics, style = "[verse]\nSteel hands find a seed\n[chorus]\nLet the garden grow", "Bright electronic pop with warm vocals"
            prepared = await client.post("/api/v1/agent-access/prepare-authored", json={"context_id": music["context_id"], "plan": {"context_id": music["context_id"], "prompt": lyrics, "style": style}})
            assert prepared.status_code == 200, prepared.text
            rendered = await client.post("/api/v1/agent-access/submit-authored", json={"prepared_id": prepared.json()["prepared_id"]})
            assert rendered.json()["job_id"] == "song-job"
            assert studio_calls[-1][1]["prompt"] == lyrics
            assert studio_calls[-1][1]["alt_prompt"] == style
            response = await client.post("/api/v1/agent-access/creative-context", json=request)
            assert response.status_code == 200, response.text
            context = response.json()
            assert context["model_type"] == "minimax_h3_ref2va_fused_turbo"
            assert [row["image_intent"] for row in context["references"]] == ["identity", "scene"]
            assert context["music_analysis"]["beat_count"] == 24
            assert context["timeline"]["director_timeline"]["timing_basis"] == "beat_and_section_analysis"
            assert "music_video_workflow" in context["guides"]["director_guidance"]
            clips, previous = [], None
            for i, window in enumerate(context["timeline"]["windows"]):
                identifier = f"authored-{i}"
                clips.append({"id": identifier, "timeline_id": window["id"], "start_frame": window["start_frame"], "end_frame": window["end_frame"], "duration_frames": window["duration_frames"], "reference_ids": [row["id"] for row in context["references"]], "previous_clip_id": previous, "video_prompt": f"Illustrated robot in the garden, authored scene {i}, following the supplied soundtrack."})
                previous = identifier
            prepared = await client.post("/api/v1/agent-access/prepare-authored", json={"context_id": context["context_id"], "plan": {"context_id": context["context_id"], "clips": clips}})
            assert prepared.status_code == 200, prepared.text
            assert not director_calls, "planning and preparation must not render"
            rendered = await client.post("/api/v1/agent-access/submit-authored", json={"prepared_id": prepared.json()["prepared_id"]})
            assert rendered.status_code == 200, rendered.text
            assert rendered.json()["pipeline_id"] == "reviewed-agent-pipeline"
            assert analyzed == [str(song)], "submission must use the original bound analysis"
            native = director_calls[0]
            assert native["creative_author"] == "connected_agent"
            assert [row["video_prompt"] for row in native["prepared_clip_plans"]] == [row["video_prompt"] for row in clips]
            assert sum(row["output_frames"] for row in native["prepared_planned_clips"]) == 291
            assert any(row.get("music_cues") for row in native["prepared_planned_clips"])
            assert [row["image_intent"] for row in native["video_params"]["minimax_h3_references"]] == ["identity", "scene"]
            song.write_bytes(b"replaced after review")
            changed = await client.post("/api/v1/agent-access/submit-authored", json={"prepared_id": prepared.json()["prepared_id"]})
            assert changed.status_code == 409
            assert len(director_calls) == 1
    asyncio.run(run())


def test_native_analysis_does_not_block_other_api_requests(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def wait():
        started.set()
        assert release.wait(3)
    app, request, *_ = setup_workflow(tmp_path, monkeypatch, wait)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            task = asyncio.create_task(client.post("/api/v1/agent-access/creative-context", json=request))
            try:
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                assert started.is_set()
                ping = await asyncio.wait_for(client.get("/ping"), timeout=0.5)
                assert ping.json() == {"ready": True}
                assert not task.done(), "audio analysis should still be waiting off the event loop"
            finally:
                release.set()
            assert (await task).status_code == 200
    asyncio.run(run())


def test_outside_storage_audio_is_rejected_before_any_analysis(tmp_path, monkeypatch):
    app, request, analyzed, *_ = setup_workflow(tmp_path, monkeypatch)
    request["params"]["audio_path"] = str(tmp_path / "settings" / "private.wav")
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/agent-access/creative-context", json=request)
            assert response.status_code == 422
            assert "outside approved Maestro storage" in response.text
            assert not analyzed
    asyncio.run(run())
