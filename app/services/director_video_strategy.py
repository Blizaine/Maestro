"""Model-aware rendering strategies for Director video jobs.

Director's original renderer assumes a rolling-window model with generic
start-frame and audio-guide inputs.  Bounded models such as MiniMax H3 need a
different contract: every queued task must fit the model's native duration
lattice, and specialized inputs are assembled independently for each shot.

The helpers in this module are deliberately pure so planning, Dashboard
repair, and regression tests all share the same timing decisions.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


ROLLING_WINDOW = "rolling_window"
BOUNDED_START_END = "bounded_start_end"
OMNI_REFERENCE = "omni_reference"

DIRECTOR_VIDEO_EXECUTION_PROFILE_VERSION = 1

# Model capability: how Director may supply visual guidance for each shot.
SHOT_IMAGES_REQUIRED = "required"
SHOT_IMAGES_OPTIONAL = "optional"
SHOT_IMAGES_DIRECT_REFERENCES = "direct_references"

# User-facing request values.  ``auto`` is resolved once at submission time
# and the effective result is persisted with the project so Dashboard repair
# never mistakes an intentionally image-free shot for a missing artifact.
SHOT_IMAGE_AUTO = "auto"
SHOT_IMAGE_PROMPT_ONLY = "prompt_only"
SHOT_IMAGE_GENERATE = "generate"

SHOT_IMAGE_POLICIES = {
    SHOT_IMAGE_PROMPT_ONLY,
    SHOT_IMAGE_GENERATE,
    SHOT_IMAGES_DIRECT_REFERENCES,
}


def _normalize_director_h3_resolution(
    value: Any,
    model_def: Mapping[str, Any],
) -> str:
    """Resolve Auto/legacy canvases using model-published H3 metadata."""

    requested = str(value or "").strip().lower().replace(" ", "")
    fallbacks = dict(model_def.get("auto_resolution_fallbacks") or {})
    if requested in fallbacks:
        return str(fallbacks[requested])

    supported: list[str] = []
    for item in model_def.get("resolutions") or []:
        candidate = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
        candidate = str(candidate or "").strip().lower()
        if re.fullmatch(r"\d+x\d+", candidate) and candidate not in supported:
            supported.append(candidate)
    for preset in (model_def.get("resolution_presets") or {}).values():
        if not isinstance(preset, Mapping):
            continue
        for candidate in (preset.get("values") or {}).values():
            candidate = str(candidate or "").strip().lower()
            if re.fullmatch(r"\d+x\d+", candidate) and candidate not in supported:
                supported.append(candidate)

    if requested in supported or not supported:
        return requested
    match = re.fullmatch(r"(\d+)x(\d+)", requested)
    if not match:
        return str(fallbacks.get("auto") or supported[0])
    width, height = (int(part) for part in match.groups())
    orientation = 0 if width == height else (1 if width > height else -1)
    candidates: list[tuple[str, int, int]] = []
    for candidate in supported:
        candidate_width, candidate_height = (
            int(part) for part in candidate.split("x", 1)
        )
        candidate_orientation = (
            0 if candidate_width == candidate_height
            else 1 if candidate_width > candidate_height
            else -1
        )
        if candidate_orientation == orientation:
            candidates.append((candidate, candidate_width, candidate_height))
    if not candidates:
        return supported[0]

    target_aspect = width / max(1, height)
    target_area = max(1, width * height)

    def score(item: tuple[str, int, int]) -> float:
        _, candidate_width, candidate_height = item
        aspect_error = abs(
            (candidate_width / candidate_height) - target_aspect
        ) / max(target_aspect, 1e-6)
        area_error = abs(
            (candidate_width * candidate_height) - target_area
        ) / target_area
        return aspect_error * 8 + area_error

    return min(candidates, key=score)[0]


def _recommended_director_h3_profile(
    total_vram_gb: Any,
    resolution: Any,
    model_def: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the lightweight memory policy published by the H3 handler."""

    normalized_resolution = _normalize_director_h3_resolution(
        resolution,
        model_def,
    )
    try:
        width, height = (
            int(part) for part in normalized_resolution.split("x", 1)
        )
        pixels = max(1, width * height)
    except (TypeError, ValueError):
        pixels = 864 * 480
    try:
        total_vram_gb = float(total_vram_gb)
    except (TypeError, ValueError):
        total_vram_gb = 0.0

    full_checkpoint = bool(model_def.get("minimax_h3_full_checkpoint", False))
    policy = (
        model_def.get("director_memory_policy")
        or model_def.get("sliding_window_memory_policy")
        or {}
    )
    architectural_maximum = int(model_def.get("frames_maximum") or 345)
    if total_vram_gb <= 0 or not policy:
        return {
            "supported": True,
            "frames": architectural_maximum,
            "fallback_resolution": None,
            "gpu_vram_gb": total_vram_gb,
            "resolution": normalized_resolution,
            "pixels": pixels,
            "checkpoint": "full" if full_checkpoint else "pruned",
        }

    for band in policy.get("resolution_bands") or []:
        if pixels < int(band.get("min_pixels", 0)):
            continue
        for tier in band.get("vram_tiers") or []:
            maximum_vram = tier.get("max_vram_gb")
            if maximum_vram is None or total_vram_gb <= float(maximum_vram):
                frames = tier.get("frames")
                frames = int(frames) if frames is not None else None
                return {
                    "supported": frames is not None and frames > 0,
                    "frames": frames,
                    "fallback_resolution": tier.get("fallback_resolution"),
                    "gpu_vram_gb": total_vram_gb,
                    "resolution": normalized_resolution,
                    "pixels": pixels,
                    "checkpoint": "full" if full_checkpoint else "pruned",
                }
        break
    return {
        "supported": True,
        "frames": architectural_maximum,
        "fallback_resolution": None,
        "gpu_vram_gb": total_vram_gb,
        "resolution": normalized_resolution,
        "pixels": pixels,
        "checkpoint": "full" if full_checkpoint else "pruned",
    }


def build_director_video_execution_profile(
    model_type: str,
    model_def: Mapping[str, Any] | None,
    video_params: Mapping[str, Any] | None,
    hardware: Mapping[str, Any] | None,
    *,
    manual_max_frames: Any = None,
    resolution_preset: str = "",
    aspect_ratio: str = "",
) -> dict[str, Any]:
    """Resolve Director's authoritative per-shot generation contract.

    ``frames_maximum`` in a model definition is an architectural limit.  H3's
    practical one-pass limit can be lower at a large canvas or on a smaller
    GPU.  Director must know that effective limit before its LLM allocates
    dialogue and action; allowing the runtime to shrink a shot later causes
    premature action, repeated dialogue, and avoidable OOM failures.

    The returned object is JSON-safe and is persisted with the project so a
    Dashboard rerun can reproduce the same timing contract.
    """

    model_def = dict(model_def or {})
    video_params = dict(video_params or {})
    hardware = dict(hardware or {})
    strategy = video_strategy(model_def)
    architecture = str(
        model_def.get("architecture")
        or model_def.get("base_model_type")
        or model_type
        or ""
    )
    architecture_lower = architecture.lower()
    model_type_lower = str(model_type or "").lower()
    is_h3 = (
        architecture_lower.startswith("minimax_h3")
        or model_type_lower.startswith("minimax_h3")
    )

    try:
        fps = float(model_def.get("fps") or 24)
    except (TypeError, ValueError):
        fps = 24.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    try:
        minimum_frames = max(1, int(model_def.get("frames_minimum") or 1))
    except (TypeError, ValueError):
        minimum_frames = 1
    try:
        frame_step = max(1, int(model_def.get("frames_steps") or 1))
    except (TypeError, ValueError):
        frame_step = 1
    raw_architectural_maximum = model_def.get("frames_maximum")
    try:
        architectural_maximum = (
            max(minimum_frames, int(raw_architectural_maximum))
            if raw_architectural_maximum is not None
            else None
        )
    except (TypeError, ValueError):
        architectural_maximum = None

    requested_resolution = str(video_params.get("resolution") or "").strip()
    normalized_resolution = requested_resolution
    recommendation: dict[str, Any] | None = None
    recommended_maximum = architectural_maximum
    checkpoint = "full" if model_def.get("minimax_h3_full_checkpoint") else "pruned"

    if is_h3:
        recommendation = _recommended_director_h3_profile(
            hardware.get("gpu_vram_gb", 0.0),
            requested_resolution,
            model_def,
        )
        normalized_resolution = str(
            recommendation.get("resolution") or requested_resolution
        )
        raw_recommended = recommendation.get("frames")
        recommended_maximum = (
            int(raw_recommended) if raw_recommended is not None else None
        )

    override_frames: int | None = None
    if manual_max_frames not in (None, "", 0, "0"):
        try:
            override_frames = int(manual_max_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Director's manual maximum shot length must be a frame count."
            ) from exc
        if architectural_maximum is None:
            raise ValueError(
                f"{model_type} does not publish a bounded shot length to override."
            )
        if not (
            minimum_frames <= override_frames <= architectural_maximum
            and (override_frames - minimum_frames) % frame_step == 0
        ):
            raise ValueError(
                f"Director's manual maximum for {model_type} must be on its "
                f"{minimum_frames}-{architectural_maximum} frame lattice "
                f"(step {frame_step})."
            )

    if is_h3 and recommended_maximum is None and override_frames is None:
        fallback = (recommendation or {}).get("fallback_resolution")
        vram = float((recommendation or {}).get("gpu_vram_gb") or 0.0)
        suffix = f" Choose {fallback}." if fallback else " Choose a lower resolution."
        raise ValueError(
            f"H3 has no automatic one-pass profile for {normalized_resolution} "
            f"on this {vram:g} GB GPU.{suffix} Advanced users can set a manual "
            "maximum shot length to try this combination experimentally."
        )

    effective_maximum = (
        override_frames
        if override_frames is not None
        else recommended_maximum
    )
    if (
        architectural_maximum is not None
        and effective_maximum is not None
    ):
        effective_maximum = min(
            architectural_maximum, int(effective_maximum)
        )

    try:
        detected_vram = float(hardware.get("gpu_vram_gb") or 0.0)
    except (TypeError, ValueError):
        detected_vram = 0.0

    return {
        "version": DIRECTOR_VIDEO_EXECUTION_PROFILE_VERSION,
        "policy_version": DIRECTOR_VIDEO_EXECUTION_PROFILE_VERSION,
        "model_type": str(model_type or ""),
        "architecture": architecture,
        "video_strategy": strategy,
        "checkpoint": checkpoint if is_h3 else None,
        "is_minimax_h3": is_h3,
        "omni_reference": strategy == OMNI_REFERENCE,
        "requested_resolution": requested_resolution,
        "normalized_resolution": normalized_resolution,
        "resolution_preset": str(resolution_preset or ""),
        "aspect_ratio": str(aspect_ratio or ""),
        "fps": fps,
        "frames_minimum": minimum_frames,
        "frame_step": frame_step,
        "architectural_max_frames": architectural_maximum,
        "recommended_max_frames": recommended_maximum,
        "effective_max_frames": effective_maximum,
        "effective_max_seconds": (
            effective_maximum / fps
            if effective_maximum is not None else None
        ),
        "gpu_vram_gb": detected_vram,
        "manual_override": override_frames is not None,
        "manual_max_frames": override_frames,
        "hardware_supported": bool(
            not is_h3 or (recommendation or {}).get("supported", True)
        ),
        "fallback_resolution": (
            (recommendation or {}).get("fallback_resolution")
        ),
        "turbo_mode": bool(video_params.get("minimax_h3_turbo_mode")),
        "turbo_preset": str(
            video_params.get("minimax_h3_turbo_preset") or ""
        ),
        "sol_attention": video_params.get("override_attention") == "sol",
        "first_block_cache": (
            video_params.get("skip_steps_cache_type") == "first_block"
        ),
        "first_block_cache_multiplier": video_params.get(
            "skip_steps_multiplier"
        ),
        "first_block_cache_warmup": video_params.get(
            "skip_steps_start_step_perc"
        ),
        "activated_lora_count": len(
            video_params.get("activated_loras") or []
        ),
    }


def validate_director_execution_frames(
    profile: Mapping[str, Any] | None,
    frames: Any,
    *,
    label: str = "Director shot",
) -> int:
    """Validate one bounded shot against its persisted execution profile."""

    try:
        frame_count = int(frames)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has no valid frame count.") from exc
    if not profile:
        return frame_count
    try:
        minimum = int(profile.get("frames_minimum") or 1)
        step = max(1, int(profile.get("frame_step") or 1))
        maximum = profile.get("effective_max_frames")
        maximum = int(maximum) if maximum is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("Director's saved video execution profile is invalid.") from exc
    if frame_count < minimum or (frame_count - minimum) % step != 0:
        raise ValueError(
            f"{label} has {frame_count} frames, outside the saved "
            f"{minimum}+{step}n timing lattice. Re-plan the project."
        )
    if maximum is not None and frame_count > maximum:
        seconds = profile.get("effective_max_seconds")
        limit = (
            f"{float(seconds):.2f} seconds / {maximum} frames"
            if seconds is not None else f"{maximum} frames"
        )
        raise ValueError(
            f"{label} has {frame_count} frames but this Director execution "
            f"profile allows one native pass of at most {limit}. Re-plan the "
            "project for the selected resolution and GPU."
        )
    return frame_count


_TIMELINE_LABEL_RE = re.compile(
    r"\b(?:window|shot)\s+\d+\s*(?:\([^)]*\))?\s*:\s*",
    re.IGNORECASE,
)
_DEPENDENT_OPENING_RE = re.compile(
    r"^\s*(?:"
    r"begin this portion of the scene(?: and leave the action able to continue)?|"
    r"continue directly from the preceding portion of the same scene|"
    r"continue from (?:the )?(?:previous|preceding) (?:shot|scene|portion)"
    r")\s*[.!:-]*\s*",
    re.IGNORECASE,
)
_DIALOGUE_TAG_RE = re.compile(r"<d\b[^>]*>.*?</d>", re.IGNORECASE | re.DOTALL)


def _compact_context(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{shortened}..." if shortened else text[:limit]


def _shot_value(shot: Any, name: str, default: Any = "") -> Any:
    if isinstance(shot, Mapping):
        return shot.get(name, default)
    return getattr(shot, name, default)


def build_independent_shot_context(
    scene_description: str,
    shot: Any | None = None,
) -> str:
    """Build a stable visual-world anchor for an independent H3 shot.

    H3 bounded shots do not inherit semantic context from an earlier render.
    Carry the user's original world/franchise request plus the structured shot
    setting and cast into every queued prompt.  The wording explicitly limits
    the anchor to continuity so plot beats and dialogue remain shot-local.
    """

    project = _compact_context(scene_description, 700)
    environment = _compact_context(_shot_value(shot, "environment"), 320)
    spatial_setup = _compact_context(_shot_value(shot, "spatial_setup"), 420)
    subjects: list[str] = []
    for subject in (_shot_value(shot, "subjects_on_screen", []) or []):
        name = _compact_context(_shot_value(subject, "speaker_name"), 80)
        description = _compact_context(
            _shot_value(subject, "visual_description"), 180
        )
        wardrobe = _compact_context(_shot_value(subject, "wardrobe"), 220)
        position = _compact_context(
            _shot_value(subject, "position_or_relation"), 220
        )
        label = " — ".join(item for item in (name, description) if item)
        subject_details = [label] if label else []
        if wardrobe:
            subject_details.append(f"wardrobe: {wardrobe}")
        if position:
            subject_details.append(f"opening position: {position}")
        label = "; ".join(subject_details)
        if label and label not in subjects:
            subjects.append(label)

    details = []
    if project:
        details.append(f"project world/franchise: {project}")
    if environment:
        details.append(f"this shot's physical setting: {environment}")
    if spatial_setup:
        details.append(f"exact opening blocking: {spatial_setup}")
    if subjects:
        details.append(f"visible cast: {'; '.join(subjects)}")
    if not details:
        return ""
    return (
        "PROJECT CONTINUITY (visual world only; use only the action and "
        "dialogue specified for this shot): "
        + ". ".join(details)
        + "."
    )


def _inject_context_anchor(text: str, anchor: str) -> str:
    text = str(text or "").strip()
    anchor = str(anchor or "").strip()
    if not anchor or anchor in text:
        return text
    # Preserve H3's Context-IR top-level label by placing continuity inside
    # integrated_multimodal_description rather than ahead of the format.
    match = re.match(
        r"(integrated_multimodal_description\s*:\s*)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1)}{anchor} {text[match.end():]}".strip()
    return f"{anchor} {text}".strip()


def _inject_blocking_detail(text: str, closing_blocking: str) -> str:
    """Keep the planner's exact final arrangement through prompt polish."""

    text = str(text or "").strip()
    closing_blocking = re.sub(
        r"\s+", " ", str(closing_blocking or "")
    ).strip()
    if not closing_blocking:
        return text
    if re.search(r"\bFINAL\s+BLOCKING\s*:", text, flags=re.IGNORECASE):
        return text
    statement = (
        "FINAL BLOCKING: The shot ends with this exact arrangement, with any "
        "required sitting, standing, walking, or repositioning shown visibly "
        f"beforehand: {closing_blocking}."
    )
    if statement in text:
        return text
    boundary = re.search(
        r"\b(?:overall_soundscape|non_diegetic_music)\s*:",
        text,
        flags=re.IGNORECASE,
    )
    if boundary:
        return (
            f"{text[:boundary.start()].rstrip()} {statement} "
            f"{text[boundary.start():].lstrip()}"
        ).strip()
    return f"{text} {statement}".strip()


def apply_independent_shot_context(
    clip_plans: Sequence[dict[str, Any]],
    *,
    scene_description: str = "",
    shots: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Attach/re-attach self-contained context to bounded shot prompts.

    The private anchor survives bounded segmentation and third-pass polish.
    Calling this helper again after polish restores the exact project context
    if an LLM rewrite omitted it.
    """

    plans = list(clip_plans)
    for index, plan in enumerate(plans):
        shot = shots[index] if shots and index < len(shots) else None
        anchor = str(plan.get("_director_context_anchor") or "").strip()
        if not anchor:
            anchor = build_independent_shot_context(scene_description, shot)
        if not anchor:
            continue
        plan["_director_context_anchor"] = anchor
        plan["video_prompt"] = _inject_context_anchor(
            _clean_independent_prompt(plan.get("video_prompt", ""), anchor),
            anchor,
        )
        plan["video_prompt"] = _inject_blocking_detail(
            plan["video_prompt"],
            plan.get("_director_closing_blocking", ""),
        )
        windows = []
        for item in (plan.get("window_prompts") or []):
            if isinstance(item, Mapping):
                updated = copy.deepcopy(dict(item))
                field = "prompt" if "prompt" in updated else "text"
                updated[field] = _inject_context_anchor(
                    _clean_independent_prompt(updated.get(field, ""), anchor),
                    anchor,
                )
                windows.append(updated)
            else:
                windows.append(
                    _inject_context_anchor(
                        _clean_independent_prompt(str(item or ""), anchor),
                        anchor,
                    )
                )
        if windows:
            plan["window_prompts"] = windows
    return plans


def video_strategy(model_def: Mapping[str, Any] | None) -> str:
    """Return the Director rendering strategy declared by a model."""

    if not isinstance(model_def, Mapping):
        return ROLLING_WINDOW
    explicit = str(model_def.get("director_video_strategy") or "").strip()
    if explicit in {ROLLING_WINDOW, BOUNDED_START_END, OMNI_REFERENCE}:
        return explicit
    return ROLLING_WINDOW


def shot_image_support(model_def: Mapping[str, Any] | None) -> str:
    """Return the visual-guidance contract declared by a video model."""

    if not isinstance(model_def, Mapping):
        return SHOT_IMAGES_REQUIRED
    explicit = str(
        model_def.get("director_shot_image_support") or ""
    ).strip()
    if explicit in {
        SHOT_IMAGES_REQUIRED,
        SHOT_IMAGES_OPTIONAL,
        SHOT_IMAGES_DIRECT_REFERENCES,
    }:
        return explicit
    return SHOT_IMAGES_REQUIRED


def resolve_shot_image_policy(
    model_def: Mapping[str, Any] | None,
    requested: str | None,
    *,
    has_visual_references: bool,
) -> str:
    """Resolve Director's saved shot-image behavior for one project.

    Existing Director models keep their required start-image workflow.
    MiniMax H3 FL2VA can render from prompts alone, while Ref2VA normally
    consumes the user's references directly.  Explicit generated guidance is
    still available for either H3 flavor.
    """

    support = shot_image_support(model_def)
    requested = str(requested or SHOT_IMAGE_AUTO).strip().lower()
    if requested not in {
        SHOT_IMAGE_AUTO,
        SHOT_IMAGE_PROMPT_ONLY,
        SHOT_IMAGE_GENERATE,
    }:
        requested = SHOT_IMAGE_AUTO

    if support == SHOT_IMAGES_REQUIRED:
        return SHOT_IMAGE_GENERATE
    if requested == SHOT_IMAGE_GENERATE:
        return SHOT_IMAGE_GENERATE
    if support == SHOT_IMAGES_DIRECT_REFERENCES:
        return SHOT_IMAGES_DIRECT_REFERENCES
    if requested == SHOT_IMAGE_PROMPT_ONLY:
        return SHOT_IMAGE_PROMPT_ONLY
    return (
        SHOT_IMAGE_GENERATE
        if has_visual_references
        else SHOT_IMAGE_PROMPT_ONLY
    )


def shot_images_required(policy: str | None) -> bool:
    """Whether an effective project policy owns per-shot image artifacts."""

    return str(policy or SHOT_IMAGE_GENERATE) == SHOT_IMAGE_GENERATE


def _duration_seconds(clip: Mapping[str, Any]) -> float:
    try:
        duration = float(clip.get("duration_sec") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        try:
            duration = float(clip.get("end") or 0) - float(clip.get("start") or 0)
        except (TypeError, ValueError):
            duration = 0.0
    return max(0.0, duration)


def _video_text(plan: Mapping[str, Any]) -> str:
    windows = [
        str(item.get("prompt") or item.get("text") or "")
        if isinstance(item, Mapping)
        else str(item or "")
        for item in (plan.get("window_prompts") or [])
    ]
    windows = [item.strip() for item in windows if item.strip()]
    if windows:
        return " Then, without a discontinuity, ".join(windows)
    return str(plan.get("video_prompt") or "").strip()


def _merge_units(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Merge adjacent sub-minimum cuts into one renderable H3 shot."""

    plan = copy.deepcopy(first["plan"])
    first_text = _video_text(first["plan"])
    second_text = _video_text(second["plan"])
    if first_text and second_text:
        plan["video_prompt"] = (
            f"{first_text} Then, within the same continuous shot, {second_text}"
        )
    else:
        plan["video_prompt"] = first_text or second_text
    plan["window_prompts"] = []
    plan["window_count"] = 1
    plan["keyframe_prompts"] = []

    first_image = str(first["plan"].get("image_prompt") or "").strip()
    second_image = str(second["plan"].get("image_prompt") or "").strip()
    if first_image and second_image:
        plan["image_prompt"] = (
            f"{first_image} The shot will progress naturally toward: {second_image}"
        )
    else:
        plan["image_prompt"] = first_image or second_image

    planned = copy.deepcopy(first["planned"])
    planned["end"] = second["planned"].get("end", planned.get("end"))
    planned["duration_sec"] = first["duration"] + second["duration"]
    planned["label"] = " + ".join(
        item
        for item in (
            str(first["planned"].get("label") or "").strip(),
            str(second["planned"].get("label") or "").strip(),
        )
        if item
    ) or "combined"
    return {
        "plan": plan,
        "planned": planned,
        "duration": first["duration"] + second["duration"],
        "source_indices": [*first["source_indices"], *second["source_indices"]],
    }


def _merge_short_units(
    units: list[dict[str, Any]],
    *,
    minimum_seconds: float,
    maximum_seconds: float,
) -> list[dict[str, Any]]:
    """Coalesce rapid cuts when the target model cannot render them."""

    units = list(units)
    while len(units) > 1:
        short_index = next(
            (index for index, unit in enumerate(units) if unit["duration"] < minimum_seconds),
            None,
        )
        if short_index is None:
            break
        candidates: list[tuple[float, int, int]] = []
        if short_index > 0:
            total = units[short_index - 1]["duration"] + units[short_index]["duration"]
            if total <= maximum_seconds:
                candidates.append((total, short_index - 1, short_index))
        if short_index + 1 < len(units):
            total = units[short_index]["duration"] + units[short_index + 1]["duration"]
            if total <= maximum_seconds:
                candidates.append((total, short_index, short_index + 1))
        if not candidates:
            break
        _, left, right = min(candidates, key=lambda item: item[0])
        units[left : right + 1] = [_merge_units(units[left], units[right])]
    return units


def _segment_plan(
    plan: Mapping[str, Any],
    segment_index: int,
    segment_count: int,
) -> dict[str, Any]:
    segmented = copy.deepcopy(dict(plan))
    source_parts = [
        str(item.get("prompt") or item.get("text") or "")
        if isinstance(item, Mapping)
        else str(item or "")
        for item in (plan.get("window_prompts") or [])
    ]
    source_parts = [item.strip() for item in source_parts if item.strip()]
    if not source_parts:
        source_parts = [str(plan.get("video_prompt") or "").strip()]

    anchor = str(plan.get("_director_context_anchor") or "").strip()
    chunks = _partition_independent_prompt(source_parts, segment_count, anchor)
    text = chunks[min(segment_index, len(chunks) - 1)] if chunks else ""
    text = _inject_context_anchor(text, anchor)

    if segment_count > 1:
        image_prompt = str(plan.get("image_prompt") or "").strip()
        if image_prompt:
            segmented["image_prompt"] = (
                f"Independent composition for shot {segment_index + 1} of "
                f"{segment_count}. {image_prompt}"
            )
    segmented["video_prompt"] = text
    segmented["window_prompts"] = []
    segmented["window_count"] = 1
    segmented["keyframe_prompts"] = []
    segmented["_director_segment_index"] = segment_index
    segmented["_director_segment_count"] = segment_count
    return segmented


def _clean_independent_prompt(text: str, anchor: str = "") -> str:
    cleaned = str(text or "")
    if anchor:
        cleaned = cleaned.replace(anchor, " ")
    cleaned = _DEPENDENT_OPENING_RE.sub("", cleaned)
    cleaned = _TIMELINE_LABEL_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _mask_dialogue(text: str) -> tuple[str, dict[str, str]]:
    values: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        key = f"__MAESTRO_DIALOGUE_{len(values)}__"
        values[key] = match.group(0)
        return key

    return _DIALOGUE_TAG_RE.sub(replace, text), values


def _restore_dialogue(text: str, values: Mapping[str, str]) -> str:
    for key, value in values.items():
        text = text.replace(key, value)
    return text


def _partition_independent_prompt(
    source_parts: Sequence[str],
    segment_count: int,
    anchor: str = "",
) -> list[str]:
    """Partition source prose into unique, chronological standalone chunks."""

    segment_count = max(1, int(segment_count))
    cleaned = " ".join(
        item
        for item in (
            _clean_independent_prompt(part, anchor) for part in source_parts
        )
        if item
    )
    if segment_count == 1:
        return [cleaned]
    if not cleaned:
        return [""] * segment_count

    masked, dialogue = _mask_dialogue(cleaned)
    units = [
        item.strip()
        for item in re.split(r"(?<=[.!?;])\s+|\s*\n+\s*", masked)
        if item.strip()
    ]
    if len(units) < segment_count:
        units = [
            item.strip()
            for item in re.split(r"(?<=[.!?;,])\s+", masked)
            if item.strip()
        ]
    if len(units) < segment_count:
        words = masked.split()
        units = [
            " ".join(words[start:end])
            for start, end in (
                (
                    round(index * len(words) / segment_count),
                    round((index + 1) * len(words) / segment_count),
                )
                for index in range(segment_count)
            )
            if end > start
        ]

    # Greedily form contiguous groups with balanced word counts. Each source
    # sentence/dialogue token is assigned exactly once, avoiding the old
    # midpoint selection that duplicated an entire 20-second window.
    groups: list[str] = []
    cursor = 0
    for group_index in range(segment_count):
        remaining_groups = segment_count - group_index
        remaining_units = len(units) - cursor
        if remaining_units <= 0:
            groups.append("")
            continue
        if remaining_groups == 1:
            end = len(units)
        else:
            remaining_words = sum(len(item.split()) for item in units[cursor:])
            target_words = max(1, remaining_words / remaining_groups)
            end = cursor
            used_words = 0
            max_end = len(units) - (remaining_groups - 1)
            while end < max_end:
                next_words = len(units[end].split())
                if end > cursor and used_words + next_words > target_words:
                    break
                used_words += next_words
                end += 1
            end = max(cursor + 1, end)
        groups.append(_restore_dialogue(" ".join(units[cursor:end]), dialogue))
        cursor = end

    return [re.sub(r"\s+", " ", group).strip() for group in groups]


def _nearest_bounded_frame_schedule(
    requested_frames: Sequence[int],
    *,
    minimum_frames: int,
    maximum_frames: int,
    frame_step: int,
) -> list[int]:
    """Snap independent shots while carrying residual timing error."""

    minimum_frames = max(1, int(minimum_frames))
    maximum_frames = max(minimum_frames, int(maximum_frames))
    frame_step = max(1, int(frame_step))
    valid = list(range(minimum_frames, maximum_frames + 1, frame_step))
    residual = 0.0
    schedule: list[int] = []
    for raw in requested_frames:
        target = max(1.0, float(raw)) + residual
        chosen = min(valid, key=lambda value: (abs(value - target), value))
        residual = target - chosen
        schedule.append(chosen)
    return schedule


def adapt_bounded_timeline(
    clip_plans: Sequence[Mapping[str, Any]],
    planned_clips: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    minimum_frames: int,
    maximum_frames: int,
    frame_step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a Director plan into bounded model-native independent shots.

    Long rolling-window scenes are divided before image generation, while
    adjacent cuts shorter than the model minimum are combined.  The returned
    durations already lie on the model's frame lattice, so generation,
    Dashboard reruns, and source-audio slicing share exact boundaries.
    """

    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 24.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    minimum_seconds = minimum_frames / fps
    maximum_seconds = maximum_frames / fps

    units: list[dict[str, Any]] = []
    count = max(len(clip_plans), len(planned_clips))
    for index in range(count):
        plan = copy.deepcopy(
            dict(clip_plans[index]) if index < len(clip_plans) else {}
        )
        planned = copy.deepcopy(
            dict(planned_clips[index]) if index < len(planned_clips) else {}
        )
        duration = _duration_seconds(planned)
        if duration <= 0:
            duration = minimum_seconds
        units.append(
            {
                "plan": plan,
                "planned": planned,
                "duration": duration,
                "source_indices": [index],
            }
        )

    units = _merge_short_units(
        units,
        minimum_seconds=minimum_seconds,
        maximum_seconds=maximum_seconds,
    )

    segmented: list[dict[str, Any]] = []
    for unit in units:
        segment_count = max(1, math.ceil(unit["duration"] / maximum_seconds))
        segment_duration = unit["duration"] / segment_count
        for segment_index in range(segment_count):
            segmented.append(
                {
                    "plan": _segment_plan(
                        unit["plan"], segment_index, segment_count
                    ),
                    "planned": copy.deepcopy(unit["planned"]),
                    "duration": segment_duration,
                    "source_indices": list(unit["source_indices"]),
                    "segment_index": segment_index,
                    "segment_count": segment_count,
                }
            )

    requested = [max(1, round(unit["duration"] * fps)) for unit in segmented]
    schedule = _nearest_bounded_frame_schedule(
        requested,
        minimum_frames=minimum_frames,
        maximum_frames=maximum_frames,
        frame_step=frame_step,
    )

    try:
        cursor = float(planned_clips[0].get("start") or 0) if planned_clips else 0.0
    except (TypeError, ValueError):
        cursor = 0.0
    result_plans: list[dict[str, Any]] = []
    result_clips: list[dict[str, Any]] = []
    for unit, frames in zip(segmented, schedule):
        duration = frames / fps
        plan = unit["plan"]
        plan["_director_source_clip_indices"] = unit["source_indices"]
        plan["_director_generation_frames"] = frames
        planned = unit["planned"]
        planned.update(
            {
                "start": cursor,
                "end": cursor + duration,
                "duration_sec": duration,
                "duration_frames": frames,
                "_director_source_clip_indices": unit["source_indices"],
                "_director_segment_index": unit["segment_index"],
                "_director_segment_count": unit["segment_count"],
            }
        )
        cursor += duration
        result_plans.append(plan)
        result_clips.append(planned)
    return result_plans, result_clips
