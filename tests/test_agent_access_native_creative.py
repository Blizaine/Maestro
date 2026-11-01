"""Native guide/adapter checks use registry data and CPU-only callbacks."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from services.agent_access.native_creative import make_native_creative_service
from services.authoring_guides import authoring_guide_bundle


def test_tts_compatibility_module_preserves_every_shared_constant():
    from services import tts_prompt_guidance
    source = Path(__file__).resolve().parents[1] / "app/models/TTS/prompt_enhancers.py"
    spec = importlib.util.spec_from_file_location("tts_compatibility_check", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = {name for name in vars(tts_prompt_guidance) if name.isupper()}
    assert names == set(module.__all__)
    assert all(getattr(module, name) == getattr(tts_prompt_guidance, name) for name in names)


@pytest.mark.parametrize("model", ["minimax_h3_fused_turbo", "minimax_h3_ref2va_fused_turbo", "minimax_h3_ref2va_pika_flex", "flux2_klein_9b", "flux3_klein", "qwen_image_21", "krea2_raw", "moss_image_sft", "moss_image_edit", "fibo2_dev", "yue2", "fish_drama3_int4"])
def test_model_guides_are_available_without_importing_a_model_runtime(monkeypatch, model):
    # Supplying registry data must never fall back to a runtime import.
    monkeypatch.setitem(sys.modules, "wgp", SimpleNamespace())
    workflow = "audio.music" if model == "yue2" else "audio.speech" if model.startswith("fish") else "video.references" if "ref2va" in model else "video.frames" if model.startswith("minimax") else "image.generate"
    result = authoring_guide_bundle(model, workflow, {"writer_instructions": "Keep my anime style; no realism.", "references": []}, {})
    assert len(result["format_guide"]) > 100
    assert result["writer_instructions"].startswith("Keep my anime")
    assert "Preserve the user's requested medium, style" in result["authoring_contract"]
    assert "do not silently replace it with independently voiced Story scenes" in result["authoring_contract"]
    assert len(result["guide_revision"]) == 64


def test_context_uses_resolved_params_and_can_recheck_private_reference_inputs(tmp_path, monkeypatch):
    from services.agent_access import authoring_plans
    seen = []
    def resolve(context, request, hardware, **kwargs):
        seen.append(copy.deepcopy(context["references"]))
        context["params"] = {**context["params"], "resolution": "1280x704"}
        return context
    monkeypatch.setattr(authoring_plans, "resolve_native_context", resolve)
    # The image adapter binds the source in native params, while author-facing
    # references expose only stable IDs and semantic roles.
    backend = SimpleNamespace(runtime=SimpleNamespace(directory=tmp_path), session_id="session")
    creator = make_native_creative_service(
        backend, models=lambda: [{"model_type": "flux2_klein_9b", "family": "flux2", "supports_ref_images": True, "is_downloaded": True}],
        model_def=lambda mt: {"image_outputs": True}, defaults=lambda mt: {}, options=lambda mt: {},
        preferences=lambda: {}, hardware=lambda: {}, lora_hint=lambda *args: "", injected_keyframes=lambda params: [], mature_mode=lambda: False,
    )
    context = creator.create_context({"workflow": "image.edit", "references": [{"id": "room", "type": "image", "role": "scene", "path": str(tmp_path / "room.png")}]}, "a")
    assert context["params"]["resolution"] == "1280x704"
    assert context["params"]["image_refs"] == [str(tmp_path / "room.png")]
    assert "_input_references" not in context
    assert "path" not in context["references"][0]
    assert context["reference_roles"]["scene"] != context["reference_roles"]["composition"]
    stored = creator.lookup(context["context_id"], "a")
    creator.resolve(stored, {})
    assert stored["params"]["image_refs"] == [str(tmp_path / "room.png")]
    assert len(seen) == 2


def test_director_context_uses_exact_audio_and_nested_video_profile(tmp_path, monkeypatch):
    from services.agent_access import authoring_plans
    measured = []
    analyzed = []
    song = tmp_path / "owned-song.wav"
    song.write_bytes(b"owned soundtrack fixture")
    def analyze(path):
        analyzed.append(path)
        return {"duration": 63.25, "bpm": 120, "beats": [{"time": 0.5, "strength": 1}], "sections": [{"start": 0, "end": 63.25, "label": "verse", "energy": 0.5}]}
    def resolve(context, request, hardware, **kwargs):
        assert context["params"]["pipeline_type"] == "music_video"
        assert context["params"]["source_duration_seconds"] == 63.25
        assert context["params"]["video_params"]["resolution"] == "1280x704"
        context["execution_profile"] = {"frames_minimum": 124, "effective_max_frames": 243, "frame_step": 17, "fps": 24}
    monkeypatch.setattr(authoring_plans, "resolve_native_context", resolve)
    backend = SimpleNamespace(runtime=SimpleNamespace(directory=tmp_path), session_id="session")
    creator = make_native_creative_service(
        backend, models=lambda: [{"model_type": "minimax_h3_fused_turbo", "is_t2v": True, "is_downloaded": True}],
        model_def=lambda mt: {"architecture": "minimax_h3", "director_video_strategy": "bounded_start_end", "image_prompt_types_allowed": "S", "any_audio_prompt": True, "audio_guide_window_slicing": True, "frames_minimum": 124, "frames_maximum": 243, "frames_steps": 17, "fps": 24},
        defaults=lambda mt: {"resolution": "1280x704", "video_length": 243}, options=lambda mt: {}, preferences=lambda: {}, hardware=lambda: {},
        lora_hint=lambda *args: "", injected_keyframes=lambda params: [], mature_mode=lambda: False,
        probe_audio=lambda path: measured.append(path) or 63.25,
        analyze_music=analyze,
    )
    context = creator.create_context({"workflow": "director.music_video", "params": {"audio_path": str(song)}}, "a")
    assert measured == [str(song)]
    assert analyzed == [str(song)]
    assert context["source_duration_seconds"] == 63.25
    assert context["timeline"]["director_profile"]["target_total_frames"] == 1518
    assert context["params"]["shot_image_guidance"] == "prompt_only"
    assert context["guides"]["director_guidance"]["shot_writing"]
    assert context["guides"]["director_guidance"]["music_video_workflow"]
    assert context["music_analysis"]["beat_count"] == 1
    assert context["music_analysis"]["vocal_timing"] == "not_analyzed"
    assert "beats" not in context["music_analysis"]
    stored = creator.lookup(context["context_id"], "a")
    assert len(stored["music_analysis"]["beats"]) == 1
    creator.resolve(stored, {})
    assert analyzed == [str(song)], "submission rechecks must reuse the bound analysis"


def test_native_omni_scene_role_survives_and_can_be_rechecked(tmp_path, monkeypatch):
    from services.agent_access import authoring_plans
    monkeypatch.setattr(authoring_plans, "resolve_native_context", lambda *args, **kwargs: None)
    # Voice alignment belongs to the already-loaded render backend; there is
    # no audio in this fixture and the controller test env has no Torch.
    monkeypatch.setitem(sys.modules, "models.minimax_h3.ref2va", SimpleNamespace(align_ref2va_voice_reference_order=lambda prompt, refs: (prompt, refs, {})))
    monkeypatch.setattr("services.h3_sequence_planner._reference_context", lambda refs: ("Scene reference", "Preserve room layout", "scene"))
    image = tmp_path / "room.png"
    image.write_bytes(b"owned image fixture")
    backend = SimpleNamespace(runtime=SimpleNamespace(directory=tmp_path), session_id="session")
    creator = make_native_creative_service(
        backend, models=lambda: [{"model_type": "minimax_h3_ref2va_fused_turbo", "is_t2v": True, "is_downloaded": True}],
        model_def=lambda mt: {"architecture": "minimax_h3_ref2va", "omni_reference": True, "fps": 24, "frames_minimum": 124, "frames_maximum": 362, "frames_steps": 17},
        defaults=lambda mt: {"video_length": 124, "sliding_window_size": 124}, options=lambda mt: {}, preferences=lambda: {}, hardware=lambda: {},
        lora_hint=lambda *args: "", injected_keyframes=lambda params: [], mature_mode=lambda: False,
    )
    context = creator.create_context({"workflow": "video.references", "brief": "A kitten enters this room.", "references": [{"id": "room", "type": "image", "role": "scene", "path": str(image)}]}, "a")
    assert context["params"]["minimax_h3_references"][0]["image_intent"] == "scene"
    assert context["references"][0]["image_intent"] == "scene"
    assert "path" not in context["references"][0]
    stored = creator.lookup(context["context_id"], "a")
    creator.resolve(stored, {})
    assert stored["params"]["minimax_h3_references"][0]["image_intent"] == "scene"
