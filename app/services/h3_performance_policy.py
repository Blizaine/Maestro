"""Workload-bounded H3 attention head-split advice.

Standard-library-only policy logic; it reads shared numeric thresholds and
never mutates the saved configuration.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path


_POLICY_PATH = Path(__file__).resolve().parents[1] / "shared" / "h3_performance_policy.json"
_RESOLUTION_RE = re.compile(r"^\s*(\d+)\s*[x\u00d7]\s*(\d+)\s*$", re.IGNORECASE)
_REFERENCE_KEYS = (
    "image_refs", "scene_references", "video_guide", "audio_guide", "video_source",
    "minimax_h3_references", "audio_guide2", "audio_guide3", "image_start", "image_end",
)
_LORA_KEYS = ("activated_loras", "loras_multipliers", "loras_multipliers2")
_SPLIT_LABELS = {0: "Off", 1: "Low", 2: "Medium"}


@lru_cache(maxsize=1)
def _policy() -> dict:
    try:
        value = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _positive_int(value):
    result = _number(value)
    if result is None or result <= 0 or not result.is_integer():
        return None
    return int(result)


def _nonempty(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return any(_nonempty(item) for item in value)
    return bool(value)


def _truthy_mode(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "video"}
    return bool(value)


def _scope_reason(hardware, model_def, params, policy):
    scope = policy.get("scope", {})
    if not all(isinstance(item, Mapping) for item in (hardware, model_def, params, scope)):
        return "invalid_inputs"

    architecture = str(model_def.get("architecture") or "").strip().lower()
    prefix = str(scope.get("architecture_prefix") or "minimax_h3").lower()
    if not architecture.startswith(prefix):
        return "non_h3"
    if model_def.get(scope.get("full_checkpoint_flag") or "minimax_h3_full_checkpoint"):
        return "full_checkpoint"
    if architecture.endswith("_full") or architecture.endswith("_full_checkpoint"):
        return "full_checkpoint"
    if architecture in set(scope.get("excluded_architectures") or ()):
        return "excluded_architecture"
    if any(bool(model_def.get(key)) for key in scope.get("excluded_flags") or ()):
        return "excluded_model_flag"

    model_type = str(params.get("model_type") or "").strip().lower()
    if model_type and (not model_type.startswith(prefix) or "viggle" in model_type or "voice_audio" in model_type):
        return "non_video_model_type"
    if any(_truthy_mode(params.get(key)) for key in scope.get("excluded_params") or ()):
        return "image_mode"
    if _truthy_mode(params.get("audio_only")):
        return "audio_only"
    requested_output = params.get("_performance_output_type", params.get("output_type"))
    if requested_output and str(requested_output).lower() != str(scope.get("output_type") or "video").lower():
        return "non_video_output"

    vram = _number(hardware.get("gpu_vram_gb"))
    ram = _number(hardware.get("ram_gb"))
    if vram is None or ram is None:
        return "unknown_hardware"
    if not float(scope.get("gpu_vram_gb_min", 11.0)) <= vram <= float(scope.get("gpu_vram_gb_max", 12.5)):
        return "vram_out_of_scope"
    if ram < float(scope.get("ram_gb_min", 30.0)):
        return "ram_out_of_scope"
    return None


def _aligned_dimension(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def _resolution_area(params, model_def, geometry):
    raw = params.get("resolution")
    alignment = _positive_int(geometry.get("canvas_alignment")) or 1
    if isinstance(raw, Mapping):
        width, height = _positive_int(raw.get("width")), _positive_int(raw.get("height"))
        if width and height:
            width, height = _aligned_dimension(width, alignment), _aligned_dimension(height, alignment)
            return {"pixels": width * height, "width": width, "height": height,
                    "label": f"{width}x{height}", "rounded": True}

    if isinstance(raw, str):
        match = _RESOLUTION_RE.match(raw)
        if match:
            requested_width, requested_height = int(match.group(1)), int(match.group(2))
            if requested_width <= 0 or requested_height <= 0:
                return None
            width = _aligned_dimension(requested_width, alignment)
            height = _aligned_dimension(requested_height, alignment)
            return {"pixels": width * height, "width": width, "height": height,
                    "label": f"{width}x{height}",
                    "rounded": width != requested_width or height != requested_height}

    field = str(geometry.get("resolution_pixels_field") or "auto_resolution_pixels")
    pixel_maps = [
        model_def.get(field),
        model_def.get("auto_resolution_budgets"),
        model_def.get("auto_resolution_pixels"),
    ]
    for policy_name in ("omni_sequence_memory_policy", "sliding_window_memory_policy"):
        nested = model_def.get(policy_name)
        if isinstance(nested, Mapping):
            pixel_maps.append(nested.get("auto_resolution_pixels"))
    if isinstance(raw, str):
        preset = raw.strip().lower()
        for pixel_map in pixel_maps:
            if not isinstance(pixel_map, Mapping):
                continue
            value = pixel_map.get(preset)
            if value is None:
                value = next((item for key, item in pixel_map.items()
                              if str(key).strip().lower() == preset), None)
            pixels = _positive_int(value)
            if pixels:
                return {"pixels": pixels, "width": None, "height": None,
                        "label": raw.strip(), "rounded": False}
            if isinstance(value, str):
                match = _RESOLUTION_RE.match(value)
                if match:
                    requested_width, requested_height = int(match.group(1)), int(match.group(2))
                    width = _aligned_dimension(requested_width, alignment)
                    height = _aligned_dimension(requested_height, alignment)
                    return {"pixels": width * height, "width": width, "height": height,
                            "label": f"{width}x{height}",
                            "rounded": width != requested_width or height != requested_height}
    return None


def largest_actual_h3_pass(params):
    """Return the largest actual pass, or None when the request is ambiguous.

    Sequential H3/Omni plans carry per-clip geometry. A normal request marked
    single-window uses video_length even if sliding_window_size is stale.
    """
    if not isinstance(params, Mapping):
        return None

    plan = params.get("h3_window_plan")
    plan = plan if isinstance(plan, Mapping) else {}
    multi_window = params.get("minimax_h3_multi_window")
    per_clip = params.get("per_clip_frames")
    if not isinstance(per_clip, (list, tuple)) or not per_clip:
        per_clip = plan.get("per_clip_frames")
    is_multi_prompt_sequence = _positive_int(params.get("multi_prompts_gen_type")) == 3
    if isinstance(per_clip, (list, tuple)) and per_clip and (
        multi_window is not False or bool(plan.get("per_clip_frames")) or is_multi_prompt_sequence
    ):
        parsed = [_positive_int(item) for item in per_clip]
        if any(item is None for item in parsed):
            return None
        total = _positive_int(params.get("video_length"))
        largest = max(parsed)
        return min(total, largest) if total else largest

    if multi_window is False:
        return _positive_int(params.get("video_length"))

    plan_window = _positive_int(plan.get("window_frames"))
    total = _positive_int(params.get("video_length"))
    if plan_window:
        return min(total, plan_window) if total else plan_window
    if total is None:
        total = _positive_int(plan.get("total_frames"))
    if total is None:
        return None

    raw_window = params.get("sliding_window_size")
    if raw_window in (None, "", 0, "0"):
        raw_window = plan.get("window_frames")
    if raw_window not in (None, "", 0, "0"):
        window = _positive_int(raw_window)
        if window is None:
            return None
        return min(total, window)
    return total


def _target_rows(frames, pixels, geometry):
    names = (
        "video_latent_frame_offset", "video_latent_frame_divisor",
        "video_latent_multiplier", "video_latent_bias", "area_divisor",
        "audio_latent_fps", "audio_latent_rate", "audio_latent_multiplier",
    )
    offset, divisor, multiplier, bias, area_divisor, audio_fps, audio_rate, audio_multiplier = (
        _positive_int(geometry.get(name)) for name in names
    )
    if None in (offset, divisor, multiplier, bias, area_divisor, audio_fps, audio_rate, audio_multiplier) or frames < offset:
        return None, None, None
    video_latents = math.ceil((frames - offset) / divisor) * multiplier + bias
    audio_latents = round(frames / audio_fps * audio_rate) * audio_multiplier
    return video_latents * (pixels / area_divisor) + audio_latents, video_latents, audio_latents


def _normalize_attention(value):
    return str(value or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _has_int8_convrot_checkpoint(model_def):
    selected = model_def.get("_checkpoint_files")
    if selected:
        files = selected if isinstance(selected, (list, tuple)) else (selected,)
        # The actual selection takes precedence over template URLs/metadata,
        # which can advertise other quantizations alongside this checkpoint.
        return all("int8_convrot" in str(item).lower().replace("-", "_") for item in files)
    for key in (
        "checkpoint_name", "checkpoint_file", "selected_model_file", "model_filename",
        "model_path", "_checkpoint_files", "quantization", "weight_quantization", "checkpoint_quantization",
    ):
        if "int8_convrot" in str(model_def.get(key) or "").lower().replace("-", "_"):
            return True
    if model_def.get("minimax_h3_fused_turbo"):
        urls = model_def.get("URLs") or ()
        if isinstance(urls, str):
            urls = (urls,)
        if any("int8_convrot" in str(url).lower().replace("-", "_") for url in urls):
            return True
    return False


def _benchmark_mismatches(hardware, config, model_def, params, benchmark):
    mismatches = []
    actual_profile = _number(params.get("override_profile"))
    if actual_profile is None or actual_profile < 0:
        actual_profile = _number(config.get("video_profile"))
    if actual_profile != _number(benchmark.get("profile")):
        mismatches.append("video profile")

    quant = str(config.get("transformer_quantization") or "").strip().lower().replace("-", "_")
    if "int8" not in quant:
        mismatches.append("INT8 ConvRot quantization")
    if not _has_int8_convrot_checkpoint(model_def):
        mismatches.append("selected INT8 ConvRot checkpoint identity")
    active_allocator = hardware.get("allocator_active")
    if active_allocator is not None and str(active_allocator).strip():
        actual_allocator = str(active_allocator).strip().lower()
    else:
        # A requested VMM value can still be pending restart. Without runtime
        # telemetry it is not evidence that this job will use that allocator.
        actual_allocator = str(config.get("vram_allocator") or "").strip().lower()
        mismatches.append("active allocator state (not detected; requested setting may be pending restart)")
    if actual_allocator != str(benchmark.get("allocator") or "").lower():
        mismatches.append("active VMM allocator")
    if config.get("smart_memory_pinning") is not benchmark.get("smart_memory_pinning"):
        mismatches.append("smart memory pinning")
    if config.get("read_ahead") is not benchmark.get("read_ahead"):
        mismatches.append("read-ahead setting")

    expected_encoder = str(benchmark.get("text_encoder") or "").strip().lower()
    requested_encoder = str(params.get("minimax_h3_text_encoder") or "").strip().lower()
    if requested_encoder not in {"", "auto", expected_encoder}:
        mismatches.append("Q2K text encoder")

    fused = bool(model_def.get("minimax_h3_fused_turbo"))
    variant_name = "fused_turbo" if fused else "pruned"
    variant = benchmark.get("variants", {}).get(variant_name, {})
    expected_steps = _positive_int(variant.get("steps"))
    actual_steps = _positive_int(params.get("num_inference_steps"))
    if actual_steps is None:
        actual_steps = _positive_int(model_def.get("num_inference_steps"))
    if expected_steps is None or actual_steps != expected_steps:
        mismatches.append(f"{variant_name} step count")

    expected_attention = _normalize_attention(variant.get("attention"))
    actual_attention = params.get("override_attention") or model_def.get("override_attention")
    if not actual_attention and fused and model_def.get("sla_attention_default"):
        actual_attention = "sla"
    normalized_actual = _normalize_attention(actual_attention)
    if variant_name == "pruned" and normalized_actual in {"sageattention2", "sageattn2"}:
        normalized_actual = "sage2"
    if not normalized_actual or normalized_actual != expected_attention:
        mismatches.append(f"{variant_name} attention backend")

    gpu_name = str(hardware.get("gpu_name") or "").lower()
    expected_gpu = str(benchmark.get("gpu_name_contains") or "").lower()
    if not expected_gpu or expected_gpu not in gpu_name:
        mismatches.append("GPU model")
    return mismatches


def _has_matching_completed_point(frames, area, split, benchmark):
    for point in benchmark.get("completed_points") or ():
        point_frames = _positive_int(point.get("frames"))
        width, height = _positive_int(point.get("width")), _positive_int(point.get("height"))
        if point_frames == frames and width and height and width * height == area and point.get("split") == split:
            return True
    return False


def h3_performance_advice(hardware, config, model_def, params) -> dict | None:
    """Return temporary split advice for the declared H3 hardware/workload scope.

    This helper is informational unless the performance planner separately
    confirms Auto-Tune owns attention_head_split for the current config.
    """
    policy = _policy()
    if _scope_reason(hardware, model_def, params, policy):
        return None

    geometry, bands, benchmark = policy.get("geometry", {}), policy.get("bands", {}), policy.get("benchmark", {})
    frames = largest_actual_h3_pass(params)
    canvas = _resolution_area(params, model_def, geometry)
    warnings = [
        "Target-row bands are a workload estimate, not the full packed sequence or a memory guarantee; fit and completion are not promised."
    ]
    reasons = []

    if frames is None or canvas is None:
        warnings.append("H3 pass length or canvas geometry is unknown; keep the saved head-split level for this job.")
        if frames is not None:
            reasons.append(f"Largest requested H3 pass: {frames} frames.")
        reasons.append(
            f"Canvas area could not be resolved from the request or the model's {geometry.get('resolution_pixels_field', 'auto_resolution_pixels')} map."
            if canvas is None else f"Resolved H3 canvas: {canvas['label']} ({canvas['pixels']:,} pixels)."
        )
        return {"attention_head_split": None, "target_rows": None, "split_label": None,
                "risk": "caution", "warnings": warnings, "reasons": reasons}

    pixels = canvas["pixels"]
    target_rows, video_latents, audio_latents = _target_rows(frames, pixels, geometry)
    if target_rows is None:
        warnings.append("The shared H3 geometry policy is invalid for this pass; keep the saved head-split level.")
        return {"attention_head_split": None, "target_rows": None, "split_label": None,
                "risk": "caution", "warnings": warnings,
                "reasons": ["H3 target rows could not be calculated from the shared policy constants."]}

    try:
        off_max = float(bands["off_max_target_rows"])
        low_max = float(bands["low_max_target_rows"])
        medium_max = float(bands["medium_max_target_rows"])
    except (KeyError, TypeError, ValueError, OverflowError):
        warnings.append("The shared H3 target-row bands are invalid; keep the saved head-split level.")
        return {"attention_head_split": None, "target_rows": round(target_rows, 2), "split_label": None,
                "risk": "caution", "warnings": warnings,
                "reasons": ["The shared H3 target-row bands could not be read."]}

    split = 0 if target_rows <= off_max else 1 if target_rows <= low_max else 2
    max_frames = _positive_int(geometry.get("medium_max_frames"))
    max_area = _positive_int(geometry.get("medium_max_area_pixels"))
    outside_envelope = (
        (max_frames is not None and frames > max_frames)
        or (max_area is not None and pixels > max_area)
        or target_rows > medium_max
    )
    if outside_envelope:
        split = max(2, split)

    if canvas["rounded"]:
        reasons.append(
            f"Canvas estimate rounds dimensions up to the H3 {geometry.get('canvas_alignment', 32)}-pixel alignment: {canvas['label']}."
        )
    divisor = float(geometry["area_divisor"])
    reasons.append(
        f"Largest requested pass is {frames} frames at {canvas['label']}; estimated target load is {target_rows:,.0f} rows "
        f"({video_latents} video latents * {pixels / divisor:g} area units + {audio_latents} audio latents)."
    )
    reasons.append(f"H3 policy recommends {_SPLIT_LABELS[split]} head splitting for this workload.")

    risk = "experimental" if outside_envelope else "caution"
    if outside_envelope:
        warnings.append(
            "This pass exceeds the completed 362-frame or 1920x1088 area envelope. Medium is the minimum advice; larger workloads remain experimental and no High level is inferred."
        )
    if any(_nonempty(params.get(key)) for key in _REFERENCE_KEYS):
        warnings.append("References and guides add unmeasured memory and runtime; their content and count are not represented in the target-row estimate.")
    if any(_nonempty(params.get(key)) for key in _LORA_KEYS):
        warnings.append("LoRAs were not included in the completed large-canvas benchmark evidence; their memory and runtime impact is unverified.")

    mismatches = _benchmark_mismatches(hardware, config, model_def, params, benchmark)
    if mismatches:
        warnings.append(
            "Completed runs used profile 4, an identified INT8 ConvRot checkpoint, VMM, smart memory pinning, read-ahead off, and the listed model-specific steps and attention backend. Settings not confirmed: "
            + ", ".join(mismatches) + ". This advice is unverified for the active recipe."
        )

    fused = bool(model_def.get("minimax_h3_fused_turbo"))
    steps = _positive_int(params.get("num_inference_steps"))
    pruned_long_1080 = (
        not fused and steps is not None and steps >= 20
        and frames >= 243 and pixels >= 1920 * 1088
    )
    if pruned_long_1080:
        warnings.append(
            "Long Pruned20/SageAttention2 1080p evidence is slow: the 243-frame Low run took about 116 minutes, and the 362-frame Medium run was cancelled after an hours-long projection. Those observations are context for this job, not its completion-time estimate."
        )

    geometry_has_completed_point = _has_matching_completed_point(frames, pixels, split, benchmark)
    measured_point = (
        not outside_envelope
        and not mismatches
        and not any(_nonempty(params.get(key)) for key in _REFERENCE_KEYS + _LORA_KEYS)
        and geometry_has_completed_point
    )
    if measured_point:
        risk = "measured"
    if pruned_long_1080:
        risk = "slow"
    if outside_envelope:
        risk = "experimental"

    gpu_name = str(hardware.get("gpu_name") or "").lower()
    expected_gpu = str(benchmark.get("gpu_name_contains") or "").lower()
    if expected_gpu not in gpu_name:
        warnings.append("This GPU is inside the nominal 12 GB scope but differs from the benchmark card; treat the selected split as unverified.")
        if risk == "measured":
            risk = "caution"
    if risk == "caution" and not outside_envelope and not mismatches and not geometry_has_completed_point:
        warnings.append("This geometry has no exact completed benchmark point.")
    return {
        "attention_head_split": split,
        "target_rows": round(target_rows, 2),
        "split_label": _SPLIT_LABELS[split],
        "risk": risk,
        "warnings": list(dict.fromkeys(warnings)),
        "reasons": reasons,
    }
