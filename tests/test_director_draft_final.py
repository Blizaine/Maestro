from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import director_draft_final as draft_final
from services import director_pipeline as pipeline


class _HeldThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def start(self):
        return None


def _source_state(tmp_path: Path) -> dict:
    return {
        "pipeline_id": "draft1234",
        "project_id": "project1234",
        "status": "completed",
        "seamless": False,
        "pipeline_type": "short_film",
        "workspace": "default",
        "video_model": "test_video_model",
        "image_model": "test_image_model",
        "director_resolution_preset": "480p",
        "director_aspect_ratio": "16:9",
        "shot_image_policy": "generate",
        "video_params": {"resolution": "640x360"},
        "_params_snapshot": {
            "pipeline_type": "short_film",
            "video_model": "test_video_model",
            "video_params": {"resolution": "640x360"},
            "planned_clips": [{"duration_sec": 1.0}],
        },
        "clips": [{
            "index": 0,
            "tag": "good",
            "planned_clip": {"duration_sec": 1.0},
            "video_filename": "draft-shot.mp4",
            "video_stale": False,
            "image_prompt": "unchanged still prompt",
            "video_prompt": "unchanged motion prompt",
            "start_image_filename": "source-image.png",
            "keyframe_filenames": ["source-keyframe.png"],
            "keyframe_prompts": [],
            "window_prompts": [],
            "window_count": 1,
        }],
    }


def _patch_pipeline_for_child(monkeypatch, tmp_path: Path, source_state: dict):
    source_file = tmp_path / "_director_pipeline_draft1234.json"
    source_file.write_text(json.dumps(source_state), encoding="utf-8")
    runtime_pipelines: dict[str, dict] = {}

    monkeypatch.setattr(pipeline, "_claim_pipeline_operation", lambda pid: True)
    monkeypatch.setattr(pipeline, "_release_pipeline_operation", lambda pid: None)
    monkeypatch.setattr(pipeline, "load_pipeline_state", lambda out_dir, pid: source_state if pid == "draft1234" else None)
    monkeypatch.setattr(pipeline, "_find_pipeline_file", lambda out_dir, pid: str(source_file) if pid == "draft1234" else None)
    monkeypatch.setattr(pipeline, "_wgp", SimpleNamespace(get_model_def=lambda model_id: {
        "resolution_presets": {"720p": {"values": {"16:9": "1280x720"}}},
    }))
    monkeypatch.setattr(pipeline, "_create_director_video_execution_profile", lambda params, model_def=None: {
        "normalized_resolution": params["video_params"]["resolution"],
        "effective_max_frames": None,
        "video_strategy": "bounded_start_end",
    })
    monkeypatch.setattr(pipeline, "validate_director_execution_frames", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_pipeline_lock", threading.RLock())
    monkeypatch.setattr(pipeline, "_pipeline_file_lock", threading.RLock())
    monkeypatch.setattr(pipeline, "_pipelines", runtime_pipelines)
    monkeypatch.setattr(pipeline, "_pipeline_threads", {})
    monkeypatch.setattr(draft_final.threading, "Thread", _HeldThread)

    def save_json(path, state):
        Path(path).write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(pipeline, "_write_pipeline_json_unlocked", save_json)
    return source_file, runtime_pipelines


def test_resolution_preset_uses_explicit_canvas_and_rejects_auto_size():
    model = {"resolution_presets": {
        "720p": {"values": {"16:9": "1280x720", "auto": "auto_720p"}},
    }}

    assert draft_final._resolution_for_preset(model, "720p", "16:9") == "1280x720"
    assert draft_final._resolution_for_preset(model, "720p", "auto", "640x360") == "1280x720"
    with pytest.raises(draft_final.DirectorDraftFinalError, match="explicit Final resolution preset"):
        draft_final._resolution_for_preset(model, "auto", "16:9")


def test_source_recipe_requires_verified_actual_seed_and_exact_lineage(tmp_path, monkeypatch):
    media = tmp_path / "draft-shot.mp4"
    media.write_bytes(b"draft-media")
    sidecar = {
        "director_pipeline_id": "draft1234",
        "director_clip_index": 0,
        "params": {"seed": 8080},
    }
    (tmp_path / "draft-shot.meta.json").write_text(json.dumps(sidecar), encoding="utf-8")
    recipe = {
        "seed": 8080,
        "prompt": "exact rendered prompt",
        "resolution": "640x360",
        "video_length": 16,
        "minimax_h3_references": [
            {"path": str(media), "weight": 1.0},
        ],
    }
    monkeypatch.setattr(
        "services.director_clip_settings.enrich_director_output_metadata",
        lambda *args, **kwargs: {"params": recipe},
    )
    state = {"workspace": "default", "clips": [{"video_filename": media.name}]}

    saved = draft_final._read_source_recipe(
        out_dir=str(tmp_path), pipeline_id="draft1234", state=state,
        state_dir=str(tmp_path), clip_index=0,
    )
    assert saved["seed"] == 8080
    assert saved["recipe"]["prompt"] == "exact rendered prompt"
    assert saved["recipe"]["minimax_h3_references"][0]["path"] == str(media)

    state["clips"][0]["approved_take_filename"] = media.name
    state["clips"][0]["approved_take_fingerprint"] = "an-older-review-fingerprint"
    with pytest.raises(draft_final.DirectorDraftFinalError, match="changed after approval"):
        draft_final._read_source_recipe(
            out_dir=str(tmp_path), pipeline_id="draft1234", state=state,
            state_dir=str(tmp_path), clip_index=0,
        )

    state["clips"][0].pop("approved_take_filename")
    state["clips"][0].pop("approved_take_fingerprint")
    sidecar["params"]["seed"] = -1
    (tmp_path / "draft-shot.meta.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(draft_final.DirectorDraftFinalError, match="actual generation seed"):
        draft_final._read_source_recipe(
            out_dir=str(tmp_path), pipeline_id="draft1234", state=state,
            state_dir=str(tmp_path), clip_index=0,
        )


def test_final_child_renders_exact_prompt_seed_and_refs_and_keeps_both_takes(tmp_path, monkeypatch):
    state = _source_state(tmp_path)
    draft_media = tmp_path / "draft-shot.mp4"
    draft_media.write_bytes(b"original-draft")
    _source_file, runtime_pipelines = _patch_pipeline_for_child(monkeypatch, tmp_path, state)
    captured = {}
    item = {
        "filename": draft_media.name,
        "path": str(draft_media),
        "seed": 8080,
        "resolution": "640x360",
        "fingerprint": draft_final._fingerprint(str(draft_media)),
        "recipe": {
            "seed": 8080,
            "prompt": "exact rendered prompt with no planning rewrite",
            "resolution": "640x360",
            "video_length": 16,
            "audio_guide": str(tmp_path / "archived-audio.wav"),
            "minimax_h3_references": [{"path": str(tmp_path / "ordered-ref.png"), "weight": 0.75}],
        },
    }
    for asset in ("archived-audio.wav", "ordered-ref.png"):
        (tmp_path / asset).write_bytes(b"asset")
    monkeypatch.setattr(draft_final, "_read_source_recipe", lambda **kwargs: json.loads(json.dumps(item)))

    child_id, resolution = draft_final._make_final_child("unused-root", "draft1234", "720p")
    assert resolution == "1280x720"
    released_source_operations = []
    monkeypatch.setattr(
        pipeline, "_release_pipeline_operation",
        lambda pid: released_source_operations.append(pid),
    )

    def submit(params, timeout_s, workspace, out_dir):
        captured.update(params)
        result_name = "final-shot.mp4"
        (Path(out_dir) / result_name).write_bytes(b"final-media")
        return [result_name]

    monkeypatch.setattr(pipeline, "_submit_and_wait", submit)
    monkeypatch.setattr(pipeline, "_save_pipeline_state", lambda pid: True)

    def update(pid, **changes):
        runtime_pipelines[pid].update(changes)

    monkeypatch.setattr(pipeline, "_update_pipeline", update)
    draft_final._render_final_worker(child_id, "unused-root")

    runtime = runtime_pipelines[child_id]
    assert captured["prompt"] == "exact rendered prompt with no planning rewrite"
    assert captured["seed"] == 8080
    assert captured["resolution"] == "1280x720"
    assert captured["audio_guide"] == str(tmp_path / "archived-audio.wav")
    assert captured["minimax_h3_references"] == [{"path": str(tmp_path / "ordered-ref.png"), "weight": 0.75}]
    assert runtime["status"] == "completed"
    assert [take["stage"] for take in runtime["_clip_takes"][0]] == ["draft", "final"]
    assert runtime["_clip_takes"][0][0]["filename"] == draft_media.name
    assert runtime["_clip_takes"][0][1]["filename"] == "final-shot.mp4"
    assert released_source_operations == ["draft1234"]


def test_take_selection_persists_and_rejects_deleted_source_draft(tmp_path, monkeypatch):
    media = tmp_path / "draft-shot.mp4"
    media.write_bytes(b"approved-draft")
    draft_take = {
        "id": "draft:source123:0",
        "label": "Draft",
        "stage": "draft",
        "pipeline_id": "source123",
        "filename": media.name,
        "fingerprint": draft_final._fingerprint(str(media)),
    }
    child = {
        "draft_final": {"stage": "final"},
        "clips": [{"takes": [draft_take], "video_filename": "final-shot.mp4"}],
    }
    source = {"clips": [{
        "video_filename": media.name,
        "video_stale": False,
        "tag": "good",
        "approved_take_filename": media.name,
        "approved_take_fingerprint": draft_take["fingerprint"],
    }]}
    state_file = tmp_path / "_director_pipeline_child123.json"
    state_file.write_text("{}", encoding="utf-8")
    runtime = {"child123": {"_clip_video_files": ["final-shot.mp4"]}}
    monkeypatch.setattr(pipeline, "_claim_pipeline_operation", lambda pid: True)
    monkeypatch.setattr(pipeline, "_release_pipeline_operation", lambda pid: None)
    monkeypatch.setattr(pipeline, "_pipeline_lock", threading.RLock())
    monkeypatch.setattr(pipeline, "_pipelines", runtime)
    monkeypatch.setattr(pipeline, "_find_pipeline_file", lambda out_dir, pid: str(state_file))
    monkeypatch.setattr(
        pipeline, "load_pipeline_state",
        lambda out_dir, pid: child if pid == "child123" else source,
    )

    def update(_out_dir, _pid, apply):
        apply(child)
        return child

    monkeypatch.setattr(pipeline, "_update_saved_pipeline", update)
    selected = draft_final.select_pipeline_clip_take(
        str(tmp_path), "child123", 0, draft_take["id"],
    )
    assert selected["filename"] == media.name
    assert child["clips"][0]["selected_take_id"] == draft_take["id"]
    assert runtime["child123"]["_clip_video_files"] == [media.name]

    monkeypatch.setattr(pipeline, "load_pipeline_state", lambda out_dir, pid: child if pid == "child123" else None)
    with pytest.raises(draft_final.DirectorDraftFinalError, match="source Draft pipeline was deleted"):
        draft_final.select_pipeline_clip_take(
            str(tmp_path), "child123", 0, draft_take["id"],
        )


def test_review_router_registers_only_director_local_routes():
    router = draft_final.create_director_draft_final_router(
        authorize_pipeline=lambda pid: None,
        initialize_pipeline=lambda: None,
        get_output_dir=lambda: "outputs",
    )

    assert {route.path for route in router.routes} == {
        "/api/v1/director/pipelines/{pid}/finalize",
        "/api/v1/director/pipelines/{pid}/clips/{clip_index}/inspect",
        "/api/v1/director/pipelines/{pid}/clips/{clip_index}/take",
    }
