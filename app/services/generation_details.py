"""Small, credential-free settings snapshots for generation job cards.

Read the job's frozen parameters, never the current browser or server defaults.
Only known generation controls are public; paths, provider settings and internal
planning state must not leak into the polling/reconnection payload.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from urllib.parse import unquote, urlsplit


DISPLAY_PARAMETERS = frozenset({
    "model_type", "prompt", "negative_prompt", "generation_mode",
    "_studio_video_workflow", "resolution", "video_length", "force_fps",
    "num_inference_steps", "guidance_scale", "guidance2_scale", "seed",
    "sample_solver", "flow_shift", "audio_flow_shift", "audio_guidance_scale",
    "override_attention", "loras_multipliers",
    "minimax_h3_turbo_mode", "minimax_h3_turbo_preset",
    "minimax_h3_experimental_loras",
    "minimax_h3_text_encoder", "minimax_h3_video_vae", "minimax_h3_multi_window",
    "sliding_window_size", "sliding_window_overlap", "denoising_strength",
    "skip_steps_cache_type", "skip_steps_multiplier", "skip_steps_start_step_perc",
    "spatial_upsampling", "temporal_upsampling", "repeat_generation",
    "generation_engine", "comfyui_workflow_id", "comfyui_workflow_name",
})
DISPLAY_PERFORMANCE = frozenset({
    "applied", "profile", "transformer_budget_mb", "attention_head_split",
    "read_ahead", "reserved_ram_fraction", "source",
})


def _scalar(value):
    return isinstance(value, (str, bool, int)) or (
        isinstance(value, float) and math.isfinite(value)
    )


def _lora_name(value: str) -> str:
    # Imported adapters can be selected by a URL. Show their filename without
    # signed query strings or local directory names.
    try:
        path = urlsplit(value).path if "://" in value else value
    except ValueError:
        path = value.split("?", 1)[0].split("#", 1)[0]
    return unquote(path.replace("\\", "/").rsplit("/", 1)[-1])


def is_terminal_studio_failure(job: Mapping) -> bool:
    params = job.get("params")
    return (
        job.get("status") in {"failed", "cancelled"}
        and job.get("kind", "generation") == "generation"
        and isinstance(params, Mapping)
        and not params.get("_director_pipeline_id")
    )


def public_generation_details(job: Mapping) -> dict | None:
    params = job.get("params")
    if not isinstance(params, Mapping):
        return None
    visible = {
        key: value for key, value in params.items()
        if key in DISPLAY_PARAMETERS and _scalar(value)
    }
    selected = params.get("activated_loras")
    if isinstance(selected, str):
        selected = [selected] if selected else []
    if isinstance(selected, (tuple, list)):
        visible["activated_loras"] = [_lora_name(name) for name in selected if isinstance(name, str)]
    result = {"params": visible}
    workspace = job.get("workspace")
    if isinstance(workspace, str):
        result["workspace"] = workspace
    plan = job.get("performance_plan")
    if isinstance(plan, Mapping):
        result["performance_plan"] = {
            key: value for key, value in plan.items()
            if key in DISPLAY_PERFORMANCE and _scalar(value)
        }
    return result
