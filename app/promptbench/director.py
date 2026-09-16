"""Prepare Director's real H3 planning and final preflight, without a project or render."""
from copy import deepcopy
import uuid


def prepare_director(params, model, settings):
    from services import director_pipeline as pipeline
    from services.director_video_strategy import apply_independent_shot_context
    fps = float(model.get("fps") or 24)
    references = deepcopy(params.get("minimax_h3_references") or [])
    images = [reference for reference in references if reference.get("type") == "image"]
    director_params = {
        "scene_description": params["prompt"], "target_duration": params["video_length"] / fps,
        "pipeline_type": "short_film_story", "video_model": params["model_type"],
        "image_model": "flux2_klein_9b", "use_director_v2": True, "seamless": False,
        "_director_shot_image_policy": "direct_references" if model.get("omni_reference") else "prompt_only",
        "fps": fps, "video_params": deepcopy(params),
        "reference_image_path": params.get("image_start"),
        "minimax_h3_references": references,
        "character_ref_paths": [reference["path"] for reference in images],
        "character_ref_labels": [str(reference.get("role") or "Reference") for reference in images],
        "llm_model_id": settings["llm_model_id"], "llm_device": "cuda", "llm_provider": "local",
    }
    # No entry is created in _pipelines. Production planning's optional progress,
    # logging and save callbacks therefore cannot create/overwrite user projects.
    clips, timeline = pipeline._run_planning("promptbench-" + uuid.uuid4().hex, director_params, "short_film_story")
    clips = apply_independent_shot_context(clips)
    clips, timeline = pipeline.prepare_director_timeline(director_params, clips, timeline)
    durations = [clip["duration_sec"] for clip in timeline]
    modes = ["ref2va" if model.get("omni_reference") else "i2va" if index == 0 and params.get("image_start") else "t2va"
             for index in range(len(clips))]
    pipeline._preflight_h3_director_prompts(params["model_type"], clips, prompt_modes=modes,
                                          durations=durations, reference_manifests=[references] * len(clips))
    return {"params": {"h3_window_prompts": [clip["video_prompt"] for clip in clips],
                       "prompt": params["prompt"], "model_type": params["model_type"]},
            "director_clip_plans": clips, "director_timeline": timeline,
            "diagnostics_scope": "Review planner repairs/fallbacks in the captured calls; Director does not return Studio's warning ledger.",
            "evaluation_scope": "Director story planning, native timeline and H3 final preflight; no images or video generated"}
