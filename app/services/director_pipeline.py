"""Server-side Director pipeline.

Orchestrates the full Director flow (LLM planning → image gen → video gen)
in a background thread so it can run without the browser being open.

Supports two planning backends:
  - Legacy: direct calls to llm_service (old monolithic approach)
  - New:    DirectorOrchestrator with layered architecture (planners → renderers → validators)

Controlled by feature flags in params or server config.
"""

import os
import re
import time
import json
import uuid
import math
import threading
import traceback
from functools import wraps
from typing import Optional

from services.job_lifecycle import (
    GENERATED_MEDIA_EXTENSIONS,
    request_cancel,
    snapshot_job,
)
from services.director_model_compat import (
    DIRECTOR_PIPELINE_TYPES,
    assess_director_model,
)
from services.director_video_strategy import (
    BOUNDED_START_END,
    OMNI_REFERENCE,
    ROLLING_WINDOW,
    SHOT_IMAGE_GENERATE,
    SHOT_IMAGE_PROMPT_ONLY,
    SHOT_IMAGE_POLICIES,
    SHOT_IMAGES_DIRECT_REFERENCES,
    adapt_bounded_timeline,
    apply_independent_shot_context,
    build_director_video_execution_profile,
    resolve_shot_image_policy,
    shot_images_required,
    validate_director_execution_frames,
    video_strategy,
)

# These will be set by launch.py on startup
_jobs: dict = None          # reference to launch._jobs
_run_generation = None      # reference to launch._run_generation
_wgp = None                 # reference to wgp module
_gen_lock = None            # reference to launch._gen_lock
_active_gen_states = None   # reference to launch._active_gen_states (abort signaling)

_pipelines: dict = {}
_pipeline_lock = threading.Lock()
_pipeline_file_lock = threading.RLock()
_pipeline_threads: dict[str, threading.Thread] = {}
_pipeline_child_jobs: dict[str, set[str]] = {}
_pipeline_starting: set[str] = set()
_pipeline_operations: set[str] = set()
_pipeline_deleting: set[str] = set()
_pipeline_repairs: dict[str, dict] = {}
_REPAIR_ACTIVE_STATUSES = {"queued", "running", "cancelling"}
_GENERATION_SETTLE_GRACE_S = 10.0
_CANCELLED_ARTIFACT_FIELDS = {
    "output_files",
    "clip_images",
    "_clip_keyframes",
    "_clip_video_files",
    "_clip_timings",
}


class PipelineBusyError(RuntimeError):
    """Raised when a Dashboard mutation conflicts with active pipeline work."""


class DirectorModelCompatibilityError(ValueError):
    """Raised before Director submits work to an incompatible model."""


def _director_hardware_snapshot() -> dict:
    """Read the same cached hardware facts used by Studio's H3 preflight."""

    try:
        from launch import _get_cached_hardware

        return dict(_get_cached_hardware() or {})
    except Exception:
        try:
            from services.hardware_detect import detect_hardware

            return dict(detect_hardware() or {})
        except Exception:
            return {"gpu_vram_gb": 0.0}


def _create_director_video_execution_profile(
    params: dict,
    *,
    model_def: Optional[dict] = None,
    hardware: Optional[dict] = None,
) -> dict:
    """Build a trusted profile and normalize the submitted video canvas."""

    video_model = params.get("video_model") or "ltx2_22B_distilled_1_1"
    if model_def is None:
        getter = getattr(_wgp, "get_model_def", None)
        model_def = getter(video_model) if callable(getter) else {}
    model_def = dict(model_def or {})
    video_params = dict(params.get("video_params") or {})
    video_loras = dict(params.get("video_loras") or {})
    profile_inputs = {
        **video_params,
        "activated_loras": video_loras.get("activated_loras", []) or [],
    }
    profile = build_director_video_execution_profile(
        video_model,
        model_def,
        profile_inputs,
        hardware if hardware is not None else _director_hardware_snapshot(),
        manual_max_frames=params.get("director_max_shot_frames"),
        resolution_preset=params.get("director_resolution_preset", ""),
        aspect_ratio=params.get("director_aspect_ratio", ""),
    )
    normalized_resolution = profile.get("normalized_resolution")
    if normalized_resolution:
        video_params["resolution"] = normalized_resolution
    if profile.get("turbo_mode"):
        from models.minimax_h3.turbo import MINIMAX_H3_TURBO_PRESET_STEPS

        video_params["num_inference_steps"] = MINIMAX_H3_TURBO_PRESET_STEPS
    params["video_params"] = video_params
    params["_director_video_execution_profile"] = profile
    return profile


def _director_video_execution_profile(params: dict) -> dict:
    profile = params.get("_director_video_execution_profile")
    return dict(profile) if isinstance(profile, dict) else {}


def _director_effective_max_frames(
    params: dict,
    model_def: dict,
) -> int:
    profile = _director_video_execution_profile(params)
    value = profile.get("effective_max_frames")
    if value is None:
        value = model_def.get("frames_maximum") or 345
    return int(value)


def _saved_director_video_execution_profile(
    state: dict,
    *,
    model_def: Optional[dict] = None,
) -> dict:
    """Read a saved profile, deriving one for projects created before it."""

    saved = state.get("video_execution_profile")
    snapshot = state.get("_params_snapshot") or {}
    if not isinstance(saved, dict):
        saved = snapshot.get("_director_video_execution_profile")
    if isinstance(saved, dict) and saved.get("effective_max_frames"):
        return dict(saved)

    params = dict(snapshot)
    params.setdefault("video_model", state.get("video_model"))
    params["video_params"] = dict(
        state.get("video_params") or params.get("video_params") or {}
    )
    params["video_loras"] = dict(
        state.get("video_loras") or params.get("video_loras") or {}
    )
    return _create_director_video_execution_profile(
        params,
        model_def=model_def,
    )


def _validate_saved_profile_for_current_hardware(
    state: dict,
    profile: dict,
    model_def: dict,
    frame_values,
) -> None:
    """Reject an auto-planned H3 shot that is unsafe on the current GPU.

    The saved profile remains the reproducibility contract when a project is
    moved to a larger card.  On a smaller card, however, forcing that old pass
    size would either OOM or invite the generic runtime to split a prompt that
    Director did not pace as sliding windows.  An explicit manual override is
    still honored because the user already opted out of Auto's guardrail.
    """

    if (
        not profile.get("is_minimax_h3")
        or profile.get("manual_override")
    ):
        return

    snapshot = dict(state.get("_params_snapshot") or {})
    snapshot["video_model"] = (
        snapshot.get("video_model")
        or state.get("video_model")
        or profile.get("model_type")
    )
    snapshot["video_params"] = dict(
        state.get("video_params") or snapshot.get("video_params") or {}
    )
    saved_resolution = profile.get("normalized_resolution")
    if saved_resolution:
        snapshot["video_params"]["resolution"] = saved_resolution
    snapshot["video_loras"] = dict(
        state.get("video_loras") or snapshot.get("video_loras") or {}
    )
    snapshot.pop("director_max_shot_frames", None)
    snapshot.pop("_director_video_execution_profile", None)
    current_profile = _create_director_video_execution_profile(
        snapshot,
        model_def=model_def,
    )
    current_maximum = current_profile.get("effective_max_frames")
    saved_maximum = profile.get("effective_max_frames")
    if current_maximum is None or saved_maximum is None:
        return
    current_maximum = int(current_maximum)
    saved_maximum = int(saved_maximum)
    if current_maximum >= saved_maximum:
        return

    for index, frames in enumerate(frame_values):
        if frames is None:
            continue
        if int(frames) > current_maximum:
            fps = float(profile.get("fps") or 24)
            raise ValueError(
                f"Saved Director shot {index + 1} requires {int(frames)} "
                f"frames, but Auto allows one {current_maximum}-frame "
                f"({current_maximum / fps:.2f}s) H3 pass at "
                f"{saved_resolution or 'this resolution'} on the current "
                "GPU. Re-plan at a lower resolution, or explicitly use a "
                "manual maximum shot override if you want to try it."
            )


def _prepare_director_generation_params(params: dict) -> None:
    """Apply Director-only H3 guarantees before publishing a child job."""

    model_type = str(params.get("model_type") or "")
    if not model_type.lower().startswith("minimax_h3"):
        return
    profile = params.get("_director_video_execution_profile")
    if isinstance(profile, dict):
        frame_values = params.get("per_clip_frames")
        if not isinstance(frame_values, (list, tuple)):
            frame_values = [params.get("video_length")]
        for index, frames in enumerate(frame_values):
            validate_director_execution_frames(
                profile,
                frames,
                label=f"Director shot {index + 1}",
            )
        # Director has already planned every H3 child as one hardware-safe
        # native pass. Prevent the generic runtime policy from silently
        # shrinking it into prompt-unaware continuation windows.
        params["sliding_window_memory_override"] = True

    if params.get("minimax_h3_turbo_mode") is True:
        from models.minimax_h3.turbo import normalize_minimax_h3_turbo_request

        getter = getattr(_wgp, "get_model_def", None)
        model_def = getter(model_type) if callable(getter) else {}
        normalize_minimax_h3_turbo_request(
            params,
            full_checkpoint=bool(
                (model_def or {}).get("minimax_h3_full_checkpoint", False)
            ),
        )


class _RepairCancelledError(RuntimeError):
    """Internal control-flow exception for a server-owned repair batch."""


def _director_model_assessment(model_type: str) -> tuple[dict, dict] | None:
    """Resolve one model and its Director capability assessment.

    Model-free unit tests inject a deliberately tiny ``_wgp`` stub.  Runtime
    always supplies ``get_model_def``; returning ``None`` for those stubs keeps
    unrelated pipeline lifecycle tests isolated from the model registry.
    """
    getter = getattr(_wgp, "get_model_def", None)
    if not callable(getter):
        return None
    model_def = getter(model_type)
    if not model_def:
        raise DirectorModelCompatibilityError(
            f"Director model '{model_type}' is not available. Choose another model.",
        )
    family_getter = getattr(_wgp, "get_model_family", None)
    architecture_getter = getattr(_wgp, "get_base_model_type", None)
    try:
        family = family_getter(model_type, for_ui=True) if callable(family_getter) else ""
    except Exception:
        family = ""
    try:
        architecture = architecture_getter(model_type) if callable(architecture_getter) else ""
    except Exception:
        architecture = ""
    return model_def, assess_director_model(
        model_type,
        model_def,
        family=family,
        architecture=architecture,
    )


def _director_visual_reference_paths(params: dict) -> list[str]:
    """Return user-supplied visual references in stable manifest order."""

    paths: list[str] = []
    primary = str(params.get("reference_image_path") or "").strip()
    if primary:
        paths.append(primary)
    for key in ("character_ref_paths", "location_ref_paths"):
        for value in params.get(key) or []:
            candidate = str(value or "").strip()
            if candidate:
                paths.append(candidate)
    return paths


def _director_has_visual_references(
    params: dict,
    *,
    existing_only: bool = False,
) -> bool:
    paths = _director_visual_reference_paths(params)
    if not existing_only:
        return bool(paths)
    return any(os.path.isfile(path) for path in paths)


def _director_effective_shot_image_policy(params: dict) -> str:
    """Return a resolved policy, retaining generated images as legacy default."""

    saved = str(params.get("_director_shot_image_policy") or "").strip()
    if saved in SHOT_IMAGE_POLICIES:
        return saved

    # Fresh API submissions are resolved in start_pipeline. Calls that bypass
    # it (old saved projects and isolated tests) must keep Director's historic
    # required-image behavior rather than silently changing semantics.
    return SHOT_IMAGE_GENERATE


def _resolve_fresh_shot_image_policy(params: dict) -> str:
    """Resolve a new submission against the selected video's capabilities."""

    getter = getattr(_wgp, "get_model_def", None)
    if not callable(getter):
        return SHOT_IMAGE_GENERATE
    video_model = params.get("video_model") or "ltx2_22B_distilled_1_1"
    model_def = getter(video_model) or {}
    return resolve_shot_image_policy(
        model_def,
        params.get("shot_image_guidance"),
        has_visual_references=_director_has_visual_references(params),
    )


def _saved_pipeline_shot_image_policy(state: dict) -> str:
    """Read a persisted policy; pre-feature projects required start images."""

    saved = str(state.get("shot_image_policy") or "").strip()
    if saved in SHOT_IMAGE_POLICIES:
        return saved
    snapshot = state.get("_params_snapshot") or {}
    saved = str(snapshot.get("_director_shot_image_policy") or "").strip()
    if saved in SHOT_IMAGE_POLICIES:
        return saved
    return SHOT_IMAGE_GENERATE


def _validate_director_models(
    params: dict,
    *,
    stages: tuple[str, ...] = ("image", "video"),
) -> None:
    """Reject model/workflow combinations Director cannot drive safely."""
    registry_methods = (
        "get_model_def",
        "get_model_family",
        "get_base_model_type",
    )
    if not all(callable(getattr(_wgp, name, None)) for name in registry_methods):
        return

    effective_policy = _director_effective_shot_image_policy(params)
    # A direct image rerun still needs a valid image model. During a complete
    # pipeline, however, prompt-only/direct-reference H3 projects have no
    # image stage and should not be blocked by an irrelevant image selector.
    validate_image_stage = "image" in stages and (
        "video" not in stages or shot_images_required(effective_policy)
    )
    if validate_image_stage:
        image_model = params.get("image_model") or "flux2_klein_9b"
        resolved = _director_model_assessment(image_model)
        if resolved is not None:
            model_def, assessment = resolved
            capability = assessment["image"]
            if not capability["compatible"]:
                name = model_def.get("name", image_model)
                raise DirectorModelCompatibilityError(
                    f"{name} cannot be used as Director's image model: "
                    f"{capability['reason']} Choose a reference-editing image model.",
                )

    if "video" not in stages:
        return

    pipeline_type = params.get("pipeline_type") or "music_video"
    if pipeline_type not in DIRECTOR_PIPELINE_TYPES:
        raise DirectorModelCompatibilityError(
            f"Unknown Director workflow '{pipeline_type}'.",
        )
    video_model = params.get("video_model") or "ltx2_22B_distilled_1_1"
    resolved = _director_model_assessment(video_model)
    if resolved is None:
        return
    model_def, assessment = resolved
    capability = assessment["video"][pipeline_type]
    name = model_def.get("name", video_model)
    workflow_labels = {
        "music_video": "Music Video",
        "short_film_audio": "audio-driven Short Film",
        "short_film_story": "story-driven Short Film",
    }
    if not capability["compatible"]:
        raise DirectorModelCompatibilityError(
            f"{name} cannot be used for Director {workflow_labels[pipeline_type]}: "
            f"{capability['reason']} Choose a compatible video model.",
        )
    if params.get("seamless"):
        seamless = assessment["video"]["seamless"]
        if not seamless["compatible"]:
            raise DirectorModelCompatibilityError(
                f"{name} cannot be used with Director Seamless: "
                f"{seamless['reason']} Turn off Seamless or choose another model.",
            )
    if params.get("voice_reference") and not assessment["supports_voice_reference"]:
        raise DirectorModelCompatibilityError(
            f"{name} does not support Director Voice Reference. "
            "Remove the voice reference or choose LTX-2 or MiniMax H3 Omni Reference.",
        )
    if (
        effective_policy == SHOT_IMAGES_DIRECT_REFERENCES
        and not _director_has_visual_references(params, existing_only=True)
    ):
        raise DirectorModelCompatibilityError(
            f"{name} needs at least one valid main, character, or location "
            "image when Director uses references directly. Add a visual "
            "reference, choose Generate shot images, or use MiniMax H3 FL2VA."
        )


def _director_params_from_saved_state(state: dict) -> dict:
    """Reconstruct compatibility-relevant params from a saved pipeline."""
    params = dict(state.get("_params_snapshot") or {})
    for key in (
        "pipeline_type",
        "seamless",
        "image_model",
        "video_model",
    ):
        if state.get(key) is not None:
            params[key] = state[key]
    params["_director_shot_image_policy"] = (
        _saved_pipeline_shot_image_policy(state)
    )
    profile = state.get("video_execution_profile")
    if isinstance(profile, dict):
        params["_director_video_execution_profile"] = profile
    return params


def _limit_director_image_refs(
    model_type: str,
    refs: list[str],
    *,
    pid: str,
) -> list[str]:
    """Honor a compatible editor's reference limit, preserving source first."""
    try:
        resolved = _director_model_assessment(model_type)
    except DirectorModelCompatibilityError:
        return refs
    if resolved is None:
        return refs
    _, assessment = resolved
    maximum = assessment.get("max_image_refs")
    if not isinstance(maximum, int) or maximum <= 0 or len(refs) <= maximum:
        return refs
    print(
        f"[Pipeline {pid}] {model_type} accepts {maximum} image reference(s); "
        f"using the source plus the first {max(0, maximum - 1)} supplemental "
        f"reference(s) and skipping {len(refs) - maximum}.",
    )
    return refs[:maximum]


def _has_runtime_model_registry() -> bool:
    return all(
        callable(getattr(_wgp, name, None))
        for name in (
            "get_model_def",
            "get_model_family",
            "get_base_model_type",
        )
    )


def _director_supports_frame_injection(model_type: str) -> bool:
    """Whether Director may generate and submit intermediate keyframes."""
    # Preserve isolation for model-free unit tests. Runtime always has the
    # complete registry and therefore takes the explicit capability path.
    if not _has_runtime_model_registry():
        return True
    try:
        model_def = _wgp.get_model_def(model_type) or {}
    except Exception:
        return False
    return bool(model_def.get("custom_frames_injection"))


def _director_native_window_frames(
    model_type: str,
    model_def: dict,
    *,
    fps: float,
    min_frames: int,
    latent_size: int,
) -> int | None:
    """Resolve the selected model's trained/default rolling-window length."""
    if not _has_runtime_model_registry():
        return None

    sources: list[dict] = []
    defaults_getter = getattr(_wgp, "get_default_settings", None)
    if callable(defaults_getter):
        try:
            defaults = defaults_getter(model_type)
            if isinstance(defaults, dict):
                sources.append(defaults)
        except Exception:
            pass
    settings = model_def.get("settings")
    if isinstance(settings, dict):
        sources.append(settings)

    candidate = None
    for source in sources:
        for key in ("sliding_window_size", "video_length"):
            try:
                value = float(source.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                candidate = round(value)
                break
        if candidate is not None:
            break

    # Wan/LTX-V definitions without an explicit default are trained around a
    # short native shot. Five seconds is the safe generic window; models with
    # longer native contexts (LTX-2, Ovi 10s, LongCat, Hunyuan) publish theirs.
    if candidate is None:
        candidate = round(5 * fps)

    latent_size = max(1, int(latent_size or 1))
    min_frames = max(1, int(min_frames or 1))
    return max(
        ((max(1, int(candidate)) - 1) // latent_size) * latent_size + 1,
        min_frames,
    )


def _claim_pipeline_operation_locked(pid: str) -> bool:
    """Reserve a terminal pipeline while ``_pipeline_lock`` is held."""
    if (
        pid in _pipeline_threads
        or bool(_pipeline_child_jobs.get(pid))
        or pid in _pipeline_starting
        or pid in _pipeline_operations
        or pid in _pipeline_deleting
        or _pipelines.get(pid, {}).get("status") in {
            "queued", "planning", "running", "paused",
        }
    ):
        return False
    _pipeline_operations.add(pid)
    return True


def _claim_pipeline_operation(pid: str) -> bool:
    """Reserve a terminal pipeline for one Dashboard mutation."""
    with _pipeline_lock:
        return _claim_pipeline_operation_locked(pid)


def _release_pipeline_operation(pid: str) -> None:
    with _pipeline_lock:
        _pipeline_operations.discard(pid)


def _claim_pipeline_delete(pid: str) -> bool:
    """Reserve deletion before taking the state-file lock."""
    with _pipeline_lock:
        pipeline = _pipelines.get(pid)
        if (
            pid in _pipeline_threads
            or bool(_pipeline_child_jobs.get(pid))
            or pid in _pipeline_starting
            or pid in _pipeline_operations
            or pid in _pipeline_deleting
            or (
                pipeline
                and pipeline.get("status") in {
                    "queued", "planning", "running", "paused",
                }
            )
        ):
            return False
        _pipeline_deleting.add(pid)
        return True


def _release_pipeline_delete(pid: str) -> None:
    with _pipeline_lock:
        _pipeline_deleting.discard(pid)


def _exclusive_pipeline_operation(function):
    """Keep delete/resume/live saves away from a Dashboard media mutation."""
    @wraps(function)
    def wrapped(out_dir: str, pid: str, *args, **kwargs):
        if not _claim_pipeline_operation(pid):
            raise PipelineBusyError(
                "Pipeline is still active; try again shortly.",
            )
        try:
            return function(out_dir, pid, *args, **kwargs)
        finally:
            _release_pipeline_operation(pid)
    return wrapped


# ── Reference art-style lock ────────────────────────────────────────────
# Flux Klein only honors a reference's art style when the MEDIUM IS NAMED
# AT THE START of the prompt ("Maintain the same black and white hand
# drawn art style. ..."). A trailing referential anchor ("...preserve the
# art style of the reference image") demonstrably does NOT hold it — the
# output comes back photorealistic. So the pipeline asks the vision LLM
# once per run to NAME the reference's medium concretely, and the phrase
# is prepended to every image prompt deterministically at generation time
# (instead of trusting the 4B planner to follow a guide rule, which it
# provably doesn't do reliably).

_STYLE_DESCRIBE_PROMPT = (
    "Name the visual medium and art style of this image in one short phrase "
    "of 3 to 8 words. Examples: 'black and white hand-drawn pencil sketch', "
    "'watercolor illustration', 'flat-color anime', 'oil painting', "
    "'photorealistic photograph'. Reply with ONLY the phrase, nothing else."
)


def _normalize_style_phrase(raw: str) -> str:
    """Reduce the vision LLM's style answer to a clean, prefix-able phrase.

    Returns "" for photographic references (photorealism is the image
    model's default — a prefix would add nothing) and for answers that
    don't look like a short phrase (refusals, prose, thinking spill).
    """
    s = (raw or "").strip()
    if not s:
        return ""
    s = s.splitlines()[0].strip()
    s = s.strip('"').strip("'").lstrip("-*# ").rstrip(".").strip()
    if not s or len(s) > 80:
        return ""
    low = s.lower()
    if "photo" in low or "realistic" in low:
        return ""
    # Avoid "...style art style" when composing the prefix sentence.
    for suffix in (" art style", " style"):
        if low.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    # Mid-sentence position: "Maintain the same simple black line..." —
    # the vision model tends to capitalize its answer.
    if s and s[0].isupper() and (len(s) < 2 or not s[1].isupper()):
        s = s[0].lower() + s[1:]
    return s


def _style_prefix_for(style: str) -> str:
    """The exact lead sentence validated to hold Klein to a medium."""
    style = (style or "").strip()
    return f"Maintain the same {style} art style. " if style else ""


# Motion-photography effects have no place in a START-FRAME prompt — the
# frame must be sharp for the video model to animate from. The music-video
# planner still writes them ("A strong motion blur effect is present on
# the background...") because its energy-focused rules leak into image
# prompts, and Klein complies with an image-wrecking smear. Deterministic
# strip, same philosophy as the style prefix: don't trust the 4B.
_MOTION_EFFECT_RE = re.compile(
    r"motion[- ]?blur|speed[- ]?lines|long[- ]?exposure|camera shake|blur effect",
    re.IGNORECASE,
)


def _strip_motion_effects(prompt: str) -> str:
    """Drop sentences/clauses that request motion-photography effects."""
    if not prompt or not _MOTION_EFFECT_RE.search(prompt):
        return prompt
    parts = re.split(r"(?<=[.;!?])\s+", prompt)
    kept = [s for s in parts if not _MOTION_EFFECT_RE.search(s)]
    cleaned = " ".join(kept).strip()
    return cleaned if cleaned else prompt

# ── Pipeline State Persistence ─────────────────────────────────────────────

PIPELINE_STATE_VERSION = 1
_PIPELINE_FILE_PREFIX = "_director_pipeline_"


def _write_pipeline_json_unlocked(filepath: str, state: dict) -> None:
    """Atomically replace one pipeline JSON file while its file lock is held."""
    temp_filepath = (
        f"{filepath}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(temp_filepath, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False, default=str)
        os.replace(temp_filepath, filepath)
    finally:
        if os.path.isfile(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError:
                pass


def _map_completed_clip_videos(
    output_files: list[str], clip_count: int,
) -> list[Optional[str]]:
    """Map an unambiguous multi-clip output prefix to its planned clips."""
    if clip_count <= 0:
        return []
    video_exts = {".mp4", ".webm", ".mkv", ".mov"}
    clips = [
        filename for filename in output_files
        if os.path.splitext(filename)[1].lower() in video_exts
        and "_multiclip" not in os.path.splitext(filename)[0].lower()
    ]
    if not clips or len(clips) > clip_count:
        return []
    return clips + [None] * (clip_count - len(clips))


def _clip_video_slots(
    output_files: list[str], clip_count: int,
) -> list[Optional[str]]:
    """Preserve explicit sparse clip indices, with legacy prefix fallback."""
    indexed = getattr(output_files, "clip_output_files", None)
    if isinstance(indexed, dict) and indexed and clip_count > 0:
        slots: list[Optional[str]] = [None] * clip_count
        for index, filename in indexed.items():
            try:
                position = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= position < clip_count and filename:
                slots[position] = filename
        if any(slots):
            return slots
    return _map_completed_clip_videos(output_files, clip_count)


def _save_pipeline_state(pid: str) -> bool:
    """Serialize one live pipeline snapshot without racing other writers."""
    with _pipeline_file_lock:
        return _save_pipeline_state_locked(pid)


def _save_pipeline_state_locked(pid: str) -> bool:
    """Serialize pipeline state to JSON on disk. Called at phase boundaries."""
    with _pipeline_lock:
        p = _pipelines.get(pid)
        if not p:
            return False
        p = dict(p)  # shallow copy for safe access outside lock

    out_dir = p.get("out_dir") or (_wgp.save_path if _wgp else "outputs")
    params = p.get("params", {})

    # Build per-clip state
    clip_plans = p.get("clip_plans", [])
    clip_images = p.get("clip_images", [])
    pre_polish = p.get("_clip_plans_pre_polish", [])
    clip_timings = p.get("_clip_timings", {})

    # Per-clip video filenames. Multi-clip output files are emitted in clip
    # order, followed by the optional *_multiclip join. Preserve a completed
    # prefix after cancellation so the Dashboard can rerun/rejoin those clips.
    clip_videos = p.get("_clip_video_files") or []
    if not clip_videos and not params.get("seamless", True):
        clip_videos = _clip_video_slots(
            p.get("output_files") or [], len(clip_plans),
        )

    clips = []
    for i, plan in enumerate(clip_plans):
        clip_state = {
            "index": i,
            "planned_clip": p.get("_planned_clips", [{}] * (i + 1))[i] if i < len(p.get("_planned_clips", [])) else None,
            "image_prompt": plan.get("image_prompt", ""),
            "video_prompt": plan.get("video_prompt", ""),
            "visual_changes": plan.get("visual_changes", []) or [],
            "image_source": plan.get("image_source", "original"),
            "keyframe_prompts": plan.get("keyframe_prompts", []) or [],
            "window_prompts": plan.get("window_prompts", []) or [],
            "window_count": plan.get("window_count", 1),
            "_director_dialogue_beats": (
                plan.get("_director_dialogue_beats", []) or []
            ),
            "_director_subjects_on_screen": (
                plan.get("_director_subjects_on_screen", []) or []
            ),
            "_director_duration_sec": plan.get("_director_duration_sec"),
            "_director_vocal_contract": plan.get("_director_vocal_contract"),
            "_director_h3_source_prompt": plan.get("_director_h3_source_prompt"),
            "_director_h3_compiled_prompt": plan.get("_director_h3_compiled_prompt"),
            "_director_h3_prompt_mode": plan.get("_director_h3_prompt_mode"),
            "_director_h3_model_family": plan.get("_director_h3_model_family"),
            "_director_speaker_registry": plan.get("_director_speaker_registry"),
            "_director_project_context": plan.get("_director_project_context"),
            "_director_environment": plan.get("_director_environment"),
            "_director_opening_blocking": plan.get("_director_opening_blocking"),
            "_director_closing_blocking": plan.get("_director_closing_blocking"),
            "_director_audio_plan": plan.get("_director_audio_plan"),
            "image_prompt_pre_polish": pre_polish[i].get("image_prompt", "") if i < len(pre_polish) else None,
            "video_prompt_pre_polish": pre_polish[i].get("video_prompt", "") if i < len(pre_polish) else None,
            # Per-window and per-keyframe pre-polish snapshots so the
            # Dashboard can show before/after diffs for windowed shots
            # (≥21s) and for keyframe prompts. Without these, windowed
            # shots showed no polish diff because video_prompt is
            # skipped by Pass 3 when window_prompts exist (its content
            # is unused at generation time anyway).
            "window_prompts_pre_polish": pre_polish[i].get("window_prompts", []) if i < len(pre_polish) else None,
            "keyframe_prompts_pre_polish": pre_polish[i].get("keyframe_prompts", []) if i < len(pre_polish) else None,
            "start_image_filename": clip_images[i] if i < len(clip_images) else None,
            "keyframe_filenames": (p.get("_clip_keyframes", []) or [])[i] if i < len(p.get("_clip_keyframes", [])) else [],
            "video_filename": clip_videos[i] if i < len(clip_videos) else None,
            "video_stale": False,
            "tag": (p.get("_clip_tags", []) or [])[i] if i < len(p.get("_clip_tags", [])) else None,
            "image_gen_time_sec": clip_timings.get(f"image_{i}"),
            "video_gen_time_sec": clip_timings.get(f"video_{i}"),
        }
        clips.append(clip_state)

    state = {
        "version": PIPELINE_STATE_VERSION,
        "pipeline_id": pid,
        "created_at": p.get("created_at"),
        "completed_at": p.get("_completed_at"),
        "status": p.get("status", "unknown"),
        "pipeline_type": params.get("pipeline_type", "music_video"),
        "scene_description": params.get("scene_description", ""),
        "reference_image_path": params.get("reference_image_path"),
        # A no-reference run creates its own visual anchor inside the output
        # directory.  Keep the basename separate from the user's input path so
        # reruns and resume can reuse it without pretending the user uploaded
        # a reference image.
        "generated_reference_image_filename": (
            params.get("generated_reference_image_filename")
            or p.get("generated_reference_image_filename")
        ),
        "character_ref_paths": params.get("character_ref_paths", []),
        "location_ref_paths": params.get("location_ref_paths", []),
        "auto_mode": params.get("auto_mode", True),
        "seamless": params.get("seamless", True),
        "image_model": params.get("image_model", ""),
        "video_model": params.get("video_model", ""),
        "shot_image_policy": _director_effective_shot_image_policy(params),
        "shot_image_guidance": params.get("shot_image_guidance", "auto"),
        "image_loras": params.get("image_loras", {}),
        "video_loras": params.get("video_loras", {}),
        "image_params": params.get("image_params", {}),
        "video_params": params.get("video_params", {}),
        "director_resolution_preset": params.get("director_resolution_preset"),
        "director_aspect_ratio": params.get("director_aspect_ratio"),
        "video_execution_profile": params.get(
            "_director_video_execution_profile", {}
        ),
        "llm_log": p.get("_llm_log"),
        "clips": clips,
        "output_files": p.get("output_files", []),
        "total_time_sec": (time.time() - p["created_at"]) if p.get("created_at") else None,
        # Full original request params, verbatim (it's the JSON dict the
        # endpoint received, so it's serializable). This is what makes a
        # crashed pipeline faithfully resumable — music-video mode in
        # particular depends on the analyzed audio track, character list, and
        # per-clip frame counts that the flattened per-clip state above does
        # not carry. resume_pipeline() rehydrates from here.
        "_params_snapshot": params,
    }

    try:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, f"{_PIPELINE_FILE_PREFIX}{pid}.json")
        _write_pipeline_json_unlocked(filepath, state)
        return True
    except Exception as e:
        print(f"[Pipeline] Failed to save state for {pid}: {e}")
        return False


def _normalize_interrupted_repair(state: dict, pid: str) -> bool:
    """Mark a persisted active repair interrupted when its worker is gone.

    Browser reloads leave the non-daemon worker registered, so they continue
    normally.  A Maestro process restart removes the registry; changing the
    saved status makes that distinction visible and leaves Repair available as
    an idempotent resume-from-disk operation.
    """
    repair = state.get("repair")
    if not isinstance(repair, dict):
        return False
    if repair.get("status") not in _REPAIR_ACTIVE_STATUSES:
        return False
    operation_id = repair.get("operation_id")
    with _pipeline_lock:
        control = _pipeline_repairs.get(pid)
        worker_present = bool(
            control
            and control.get("operation_id") == operation_id
        )
    if worker_present:
        return False

    now = time.time()
    repair.update({
        "status": "interrupted",
        "phase": "interrupted",
        "clip_index": None,
        "message": "Repair was interrupted when Maestro stopped. Start Repair again to continue.",
        "error": "Maestro stopped before the repair finished.",
        "updated_at": now,
        "completed_at": now,
    })
    return True


def list_pipeline_states(out_dir: str) -> list[dict]:
    """Scan directory for saved pipeline state files. Returns summary list."""
    results = []
    if not os.path.isdir(out_dir):
        return results
    # Scan top-level and workspace subdirectories
    dirs_to_scan = [out_dir]
    for name in os.listdir(out_dir):
        sub = os.path.join(out_dir, name)
        if os.path.isdir(sub):
            dirs_to_scan.append(sub)

    for scan_dir in dirs_to_scan:
        for fname in os.listdir(scan_dir):
            if fname.startswith(_PIPELINE_FILE_PREFIX) and fname.endswith(".json"):
                try:
                    filepath = os.path.join(scan_dir, fname)
                    with _pipeline_file_lock:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        # Normalize and replace the exact snapshot read while
                        # retaining the file lock. Releasing it between read
                        # and write let a repair worker publish newer progress
                        # that this stale list snapshot then overwrote.
                        pid = data.get("pipeline_id", "")
                        changed = _normalize_interrupted_repair(data, pid)

                        # Detect stale "running" pipelines while retaining the
                        # same serialization boundary as repair normalization.
                        status = data.get("status", "unknown")
                        with _pipeline_lock:
                            pipeline_present = pid in _pipelines
                        if status == "running" and not pipeline_present:
                            data["status"] = "crashed"
                            status = "crashed"
                            changed = True
                        if changed:
                            _write_pipeline_json_unlocked(filepath, data)
                    results.append({
                        "id": pid,
                        "status": status,
                        "pipeline_type": data.get("pipeline_type", ""),
                        "created_at": data.get("created_at"),
                        "clip_count": len(data.get("clips", [])),
                        "output_count": len(data.get("output_files", [])),
                        "scene_description": (data.get("scene_description", "") or "")[:100],
                        "workspace": os.path.basename(scan_dir) if scan_dir != out_dir else "default",
                        "repair_status": (data.get("repair") or {}).get("status"),
                        "_filepath": filepath,
                    })
                except Exception:
                    pass
    results.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return results


def _backfill_clip_video_filenames(state: dict, state_dir: str) -> dict:
    """Derive per-clip video filenames from output_files when absent.

    Multi-clip (non-seamless) runs produce one video per clip, in clip
    order, plus a trailing *_multiclip.mp4 join — but the runtime never
    recorded them per clip (_clip_video_files was a dead key), leaving
    every clip's video_filename null. That made the Dashboard count all
    clips as "missing" and broke Rejoin (needs >= 2 per-clip files).
    Fill only null entries (a rerun clip's filename must survive), only
    when the per-clip count matches exactly, and only for files that
    still exist next to the pipeline file. Seamless runs (one combined
    output) never match the count and are left untouched.
    """
    clips = state.get("clips") or []
    outputs = [
        filename for filename in (state.get("output_files") or [])
        if "_multiclip" not in os.path.splitext(filename)[0].lower()
    ]
    if not clips or len(outputs) != len(clips):
        return state
    for i, clip in enumerate(clips):
        if not clip.get("video_filename") and os.path.isfile(os.path.join(state_dir, outputs[i])):
            clip["video_filename"] = outputs[i]
    return state


_SAVED_MEDIA_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp"},
    "video": {".mkv", ".mov", ".mp4", ".webm"},
}


def _invalid_saved_media_numbers(
    filenames: list,
    expected_count: int,
    output_dir: str,
    media_kind: str,
) -> list[int]:
    """Return 1-based slots without a non-empty direct-child media file."""
    allowed_extensions = _SAVED_MEDIA_EXTENSIONS.get(media_kind)
    if allowed_extensions is None:
        raise ValueError(f"Unsupported saved media kind: {media_kind}")
    output_root = os.path.realpath(os.path.abspath(output_dir))
    normalized_root = os.path.normcase(output_root)
    invalid = []
    for index in range(expected_count):
        filename = filenames[index] if index < len(filenames) else ""
        if (
            not isinstance(filename, str)
            or not filename
            or os.path.basename(filename) != filename
        ):
            invalid.append(index + 1)
            continue
        candidate = os.path.realpath(os.path.join(output_root, filename))
        if (
            os.path.normcase(os.path.dirname(candidate)) != normalized_root
            or os.path.splitext(filename)[1].lower() not in allowed_extensions
            or not os.path.isfile(candidate)
        ):
            invalid.append(index + 1)
            continue
        try:
            if os.path.getsize(candidate) <= 0:
                invalid.append(index + 1)
        except OSError:
            invalid.append(index + 1)
    return invalid


def _require_video_start_images(
    clip_images: list,
    clip_count: int,
    output_dir: str,
) -> None:
    """Stop the video phase rather than silently falling back to T2V."""
    invalid = _invalid_saved_media_numbers(
        clip_images, clip_count, output_dir, "image",
    )
    if not invalid:
        return
    invalid_labels = ", ".join(str(index) for index in invalid)
    raise RuntimeError(
        "Start-image generation did not produce valid recorded files for "
        f"shot(s) {invalid_labels}; video generation was not started. "
        "Use the Dashboard to regenerate the missing images."
    )


def _repair_saved_h3_frame_lattice(state: dict) -> bool:
    """Upgrade legacy Director clip timing to H3's native frame lattice.

    Older projects and generic media-duration fallbacks can contain ordinary
    24-fps counts such as 120 frames for five seconds. H3 requires 124 frames
    followed by 17-frame increments. Repair the complete saved timeline as a
    unit, mark any already-rendered affected clips stale, and let Dashboard
    repair regenerate them instead of failing before a job can be queued.
    """

    if not isinstance(state, dict):
        return False
    snapshot = state.get("_params_snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    video_model = str(
        state.get("video_model") or snapshot.get("video_model") or ""
    )
    model_def = {}
    try:
        getter = getattr(_wgp, "get_model_def", None)
        if callable(getter):
            model_def = getter(video_model) or {}
    except Exception:
        model_def = {}
    architecture = str(model_def.get("architecture") or video_model).lower()
    if not architecture.startswith("minimax_h3"):
        return False

    try:
        fps = float(model_def.get("fps") or snapshot.get("fps") or 24)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("invalid fps")
    except (TypeError, ValueError):
        fps = 24.0
    minimum = int(model_def.get("frames_minimum") or 124)
    maximum = int(model_def.get("frames_maximum") or 345)
    frame_step = int(model_def.get("frames_steps") or 17)
    clips = state.get("clips") or []
    if not isinstance(clips, list) or not clips:
        return False

    from models.minimax_h3.minimax_h3_handler import (
        normalize_h3_clip_frame_schedule,
    )

    requested: list[int] = []
    metadata_missing: list[bool] = []
    for clip in clips:
        planned = clip.get("planned_clip") if isinstance(clip, dict) else None
        planned = planned if isinstance(planned, dict) else {}
        raw_frames = planned.get("duration_frames")
        missing = raw_frames in (None, "", 0, "0")
        try:
            frame_count = int(round(float(raw_frames)))
        except (TypeError, ValueError, OverflowError):
            frame_count = 0
        if frame_count <= 0:
            try:
                duration = float(planned.get("duration_sec") or 0)
            except (TypeError, ValueError):
                duration = 0.0
            if duration <= 0:
                try:
                    duration = float(planned.get("end") or 0) - float(
                        planned.get("start") or 0
                    )
                except (TypeError, ValueError):
                    duration = 0.0
            if duration <= 0 and isinstance(clip, dict):
                try:
                    duration = float(clip.get("_director_duration_sec") or 0)
                except (TypeError, ValueError):
                    duration = 0.0
            frame_count = round(max(0.0, duration) * fps)
        requested.append(frame_count)
        metadata_missing.append(missing)

    repaired = normalize_h3_clip_frame_schedule(
        requested,
        minimum_frames=minimum,
        maximum_frames=maximum,
        frame_step=frame_step,
    )
    changed_indices = [
        index
        for index, (before, after, missing) in enumerate(
            zip(requested, repaired, metadata_missing)
        )
        if missing or before != after
    ]
    if not changed_indices:
        return False
    # Retiming one clip shifts the source-media start time of every clip that
    # follows it. Regenerate that entire suffix so old audio/video slices are
    # never reused against the repaired timeline.
    stale_indices = set(range(min(changed_indices), len(clips)))

    first_plan = clips[0].get("planned_clip") or {}
    try:
        cursor = float(first_plan.get("start") or 0.0)
    except (TypeError, ValueError):
        cursor = 0.0
    snapshot_plans = snapshot.get("planned_clips")
    if not isinstance(snapshot_plans, list):
        snapshot_plans = []

    for index, (clip, frame_count) in enumerate(zip(clips, repaired)):
        if not isinstance(clip, dict):
            continue
        planned = clip.get("planned_clip")
        if not isinstance(planned, dict):
            planned = {}
            clip["planned_clip"] = planned
        duration = frame_count / fps
        planned.update({
            "start": cursor,
            "end": cursor + duration,
            "duration_sec": duration,
            "duration_frames": frame_count,
        })
        clip["_director_duration_sec"] = duration
        if index in stale_indices:
            clip["video_stale"] = True
        if index < len(snapshot_plans) and isinstance(snapshot_plans[index], dict):
            snapshot_plans[index].update(planned)
        cursor += duration

    if snapshot:
        snapshot["planned_clips"] = snapshot_plans
    state["_h3_frame_lattice_repair"] = {
        "version": 1,
        "clip_indices": [index + 1 for index in changed_indices],
        "stale_clip_indices": [index + 1 for index in sorted(stale_indices)],
        "original_frames": requested,
        "repaired_frames": repaired,
    }
    print(
        f"[Director {state.get('id') or 'saved'}] Repaired legacy H3 clip "
        f"frame schedule: {requested} -> {repaired}."
    )
    return True


def load_pipeline_state(out_dir: str, pid: str) -> Optional[dict]:
    """Load a saved state while serialized against deletion/replacement."""
    with _pipeline_file_lock:
        return _load_pipeline_state_locked(out_dir, pid)


def _load_pipeline_state_locked(out_dir: str, pid: str) -> Optional[dict]:
    """Load a saved pipeline state by ID. Searches out_dir and subdirectories."""
    target = f"{_PIPELINE_FILE_PREFIX}{pid}.json"
    # Search top-level
    filepath = os.path.join(out_dir, target)
    if os.path.isfile(filepath):
        with _pipeline_file_lock:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            state_changed = _normalize_interrupted_repair(state, pid)
            state_changed = _repair_saved_h3_frame_lattice(state) or state_changed
            if state_changed:
                _write_pipeline_json_unlocked(filepath, state)
            return _backfill_clip_video_filenames(state, out_dir)
    # Search subdirectories (workspaces)
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            sub = os.path.join(out_dir, name, target)
            if os.path.isfile(sub):
                with _pipeline_file_lock:
                    with open(sub, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    state_changed = _normalize_interrupted_repair(state, pid)
                    state_changed = (
                        _repair_saved_h3_frame_lattice(state) or state_changed
                    )
                    if state_changed:
                        _write_pipeline_json_unlocked(sub, state)
                    return _backfill_clip_video_filenames(
                        state, os.path.join(out_dir, name),
                    )
    return None


def update_clip_tag(out_dir: str, pid: str, clip_index: int, tag: Optional[str]) -> bool:
    if not _claim_pipeline_operation(pid):
        raise PipelineBusyError("Pipeline is still active; try again shortly.")
    try:
        with _pipeline_file_lock:
            return _update_clip_tag_locked(out_dir, pid, clip_index, tag)
    finally:
        _release_pipeline_operation(pid)


def update_clip_prompt(out_dir: str, pid: str, clip_index: int,
                       field: str, value: str) -> bool:
    """Update image_prompt or video_prompt for a clip without
    triggering generation.  Returns True on success, raises
    PipelineBusyError when the pipeline is still active."""
    if not _claim_pipeline_operation(pid):
        raise PipelineBusyError("Pipeline is still active; try again shortly.")
    try:
        def _update(state):
            clips = state.get("clips", [])
            if clip_index < 0 or clip_index >= len(clips):
                return
            state["clips"][clip_index][field] = value
        _update_saved_pipeline(out_dir, pid, _update)
        return True
    finally:
        _release_pipeline_operation(pid)


def _update_clip_tag_locked(out_dir: str, pid: str, clip_index: int, tag: Optional[str]) -> bool:
    """Update the tag on a specific clip in a saved pipeline state."""
    state = load_pipeline_state(out_dir, pid)
    if not state:
        return False
    clips = state.get("clips", [])
    if clip_index < 0 or clip_index >= len(clips):
        return False
    clips[clip_index]["tag"] = tag

    # Find and overwrite the file
    target = f"{_PIPELINE_FILE_PREFIX}{pid}.json"
    for search_dir in [out_dir] + [os.path.join(out_dir, d) for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]:
        filepath = os.path.join(search_dir, target)
        if os.path.isfile(filepath):
            _write_pipeline_json_unlocked(filepath, state)
            return True
    return False


def _find_pipeline_file(out_dir: str, pid: str) -> Optional[str]:
    """Find the JSON file path for a saved pipeline."""
    target = f"{_PIPELINE_FILE_PREFIX}{pid}.json"
    filepath = os.path.join(out_dir, target)
    if os.path.isfile(filepath):
        return filepath
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            sub = os.path.join(out_dir, name, target)
            if os.path.isfile(sub):
                return sub
    return None


def _update_saved_pipeline(out_dir: str, pid: str, updater) -> Optional[dict]:
    with _pipeline_file_lock:
        return _update_saved_pipeline_locked(out_dir, pid, updater)


def _update_saved_pipeline_locked(out_dir: str, pid: str, updater) -> Optional[dict]:
    """Load a saved pipeline, apply an updater function, save back, and return the state."""
    filepath = _find_pipeline_file(out_dir, pid)
    if not filepath:
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        state = _backfill_clip_video_filenames(
            json.load(f), os.path.dirname(filepath),
        )
    updater(state)
    _write_pipeline_json_unlocked(filepath, state)
    return state


# Pipeline statuses whose run thread is (or may become) alive — a paused
# pipeline is blocked in _wait_for_resume and resurrects its state file
# on resume, so deletion must refuse these, not just "running".
_ACTIVE_PIPELINE_STATUSES = ("queued", "planning", "running", "paused")


def any_pipeline_active() -> bool:
    """True when any in-memory pipeline has a live (or resumable-in-place)
    run thread. Used by workspace deletion: between generation jobs a
    pipeline holds no _jobs entry yet will recreate its workspace folder
    on its next step."""
    with _pipeline_lock:
        return bool(
            _pipeline_threads
            or _pipeline_child_jobs
            or _pipeline_starting
            or _pipeline_operations
            or _pipeline_deleting
        ) or any(
            p.get("status") in _ACTIVE_PIPELINE_STATUSES
            for p in _pipelines.values()
        )


def delete_pipeline(out_dir: str, pid: str) -> dict:
    """Serialize deletion against every pipeline-state reader and writer."""
    if not _claim_pipeline_delete(pid):
        return {"ok": False, "error": "running"}
    try:
        with _pipeline_file_lock:
            return _delete_pipeline_locked(out_dir, pid)
    finally:
        _release_pipeline_delete(pid)


def _delete_pipeline_locked(out_dir: str, pid: str) -> dict:
    """Delete a saved pipeline and every media file it produced.

    Refuses while the pipeline is running OR paused in memory: its state
    file is re-written at phase boundaries (and on resume) and would
    resurrect mid-delete, and popping a paused pipeline's entry crashes
    its blocked run thread. The media set is the union of filenames the
    state JSON references (start images, keyframes, clip videos,
    joins/rejoins) and any media in the same folder whose .meta.json
    sidecar carries this pipeline's id stamp — the second set catches
    superseded rerun files the JSON no longer points at. Shared inputs
    in uploads/ (the song, character and location refs) are absolute
    paths outside the pipeline folder and are never touched.
    """
    with _pipeline_lock:
        mem = _pipelines.get(pid)
        if (
            pid in _pipeline_threads
            or bool(_pipeline_child_jobs.get(pid))
            or pid in _pipeline_starting
            or pid in _pipeline_operations
            or (
                mem and mem.get("status") in _ACTIVE_PIPELINE_STATUSES
            )
        ):
            return {"ok": False, "error": "running"}
    filepath = _find_pipeline_file(out_dir, pid)
    if not filepath:
        return {"ok": False, "error": "not_found"}
    pipeline_dir = os.path.dirname(filepath)

    state = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state = _backfill_clip_video_filenames(json.load(f), pipeline_dir)
    except Exception:
        pass

    names = set()
    if state:
        for clip in state.get("clips", []) or []:
            if clip.get("start_image_filename"):
                names.add(clip["start_image_filename"])
            for kf in clip.get("keyframe_filenames") or []:
                if kf:
                    names.add(kf)
            if clip.get("video_filename"):
                names.add(clip["video_filename"])
        for out in state.get("output_files", []) or []:
            if out:
                names.add(out)
    try:
        dir_entries = os.listdir(pipeline_dir)
    except OSError:
        dir_entries = []
    # Sidecar names strip the media extension ("clip_0.mp4" ->
    # "clip_0.meta.json"), so map extensionless base -> real media file
    # before sweeping; adding the bare base would silently no-op.
    base_to_media = {}
    ambiguous_media_bases = set()
    for entry in dir_entries:
        if entry.endswith(".meta.json") or entry.startswith(_PIPELINE_FILE_PREFIX):
            continue
        stem, extension = os.path.splitext(entry)
        if extension.lower() not in GENERATED_MEDIA_EXTENSIONS:
            continue
        existing = base_to_media.setdefault(stem, entry)
        if existing != entry:
            ambiguous_media_bases.add(stem)
    for fname in dir_entries:
        if not fname.endswith(".meta.json"):
            continue
        try:
            with open(os.path.join(pipeline_dir, fname), "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get("director_pipeline_id") == pid:
            sidecar_stem = fname[: -len(".meta.json")]
            media = meta.get("output_filename")
            if not (
                isinstance(media, str)
                and media == os.path.basename(media)
                and os.path.splitext(media)[0] == sidecar_stem
                and os.path.splitext(media)[1].lower()
                    in GENERATED_MEDIA_EXTENSIONS
                and os.path.isfile(os.path.join(pipeline_dir, media))
            ):
                media = (
                    None if sidecar_stem in ambiguous_media_bases
                    else base_to_media.get(sidecar_stem)
                )
            if media:
                names.add(media)
            else:
                # Orphan sidecar (media already gone) — remove it directly.
                try:
                    os.remove(os.path.join(pipeline_dir, fname))
                except OSError:
                    pass

    from services.win_safe_files import safe_delete, safe_join_under, favorites_lock
    deleted = 0
    deferred = 0
    errors = []
    cleanup_blocked = False
    for name in sorted(names):
        # State filenames are relative; contain them to the pipeline folder
        # (symlink-resolving join) so a tampered state file cannot reach
        # outside it.
        target = safe_join_under(pipeline_dir, name)
        if target is None:
            errors.append(f"skipped suspicious path: {name}")
            cleanup_blocked = True
            continue
        # retries=1: bulk sweep — locked files go straight to the
        # trash-rename path instead of sleeping through backoff per file.
        result = safe_delete(target, retries=1)
        if result.get("deferred"):
            deferred += 1
        elif result.get("deleted"):
            deleted += 1
        elif result.get("reason") == "locked":
            errors.append(name)
            cleanup_blocked = True
            # Preserve ownership companions so a later retry can still find
            # and safely remove this media.
            continue
        elif not result.get("deleted") and result.get("reason") != "not_found":
            errors.append(name)
            cleanup_blocked = True
            continue
        artifact_base = os.path.splitext(target)[0]
        # WGP may write metadata JSON or an alpha-frame ZIP beside the media
        # without registering those companions in its gallery list. Removing
        # them with their owned media prevents cancelled window artifacts from
        # accumulating invisibly.
        for companion_ext in (".meta.json", ".json", ".zip"):
            companion = artifact_base + companion_ext
            companion_result = safe_delete(companion, retries=1)
            if companion_result.get("reason") == "locked":
                errors.append(os.path.basename(companion))
                cleanup_blocked = True

    # Un-favorite everything that vanished (per-workspace .favorites.json).
    # Lock shared with launch.py's favorites endpoints — both sides do
    # read-modify-write on the same file from threadpool handlers.
    with favorites_lock:
        fav_path = os.path.join(pipeline_dir, ".favorites.json")
        if os.path.isfile(fav_path):
            try:
                with open(fav_path, "r", encoding="utf-8") as f:
                    favs = json.load(f)
                if isinstance(favs, list):
                    kept = [n for n in favs if n not in names]
                    if len(kept) != len(favs):
                        with open(fav_path, "w", encoding="utf-8") as f:
                            json.dump(sorted(kept), f)
            except Exception:
                pass

    # Current rerun slices are unique and cleaned in rerun_clip_video. Sweep
    # any historical/crash leftovers only when this was the folder's last
    # pipeline, because older names were not pipeline-scoped.
    try:
        others = [n for n in os.listdir(pipeline_dir)
                  if n.startswith(_PIPELINE_FILE_PREFIX) and n.endswith(".json")
                  and n != os.path.basename(filepath)]
        if not others:
            for n in os.listdir(pipeline_dir):
                if n.startswith("_rerun_audio_") and n.endswith(".wav"):
                    safe_delete(os.path.join(pipeline_dir, n))
    except OSError:
        pass

    delete_error = None
    if cleanup_blocked:
        # The state file is the recovery marker for retrying a partial delete.
        # Never erase it while owned media or companions are still locked.
        state_removed = False
        delete_error = "media_locked"
    else:
        state_result = safe_delete(filepath, retries=1)
        state_removed = bool(state_result.get("deleted")) or (
            state_result.get("reason") == "not_found"
        )
        if not state_removed:
            errors.append("state file is locked")
            delete_error = "state_file_locked"
    if state_removed:
        with _pipeline_lock:
            _pipelines.pop(pid, None)

    try:
        from services.search_index import get_search_index
        get_search_index().invalidate()
    except Exception:
        pass

    return {
        "ok": state_removed,
        **({"error": delete_error} if delete_error else {}),
        "dir": pipeline_dir, "media_total": len(names),
        "media_deleted": deleted, "media_deferred": deferred, "errors": errors,
    }


@_exclusive_pipeline_operation
def rerun_clip_image(out_dir: str, pid: str, clip_index: int, prompt_override: str = None) -> dict:
    return _rerun_clip_image_impl(out_dir, pid, clip_index, prompt_override)


def _rerun_clip_image_impl(out_dir: str, pid: str, clip_index: int, prompt_override: str = None) -> dict:
    """Re-generate the start image for a single clip. Returns {job_id, filename} or raises."""
    state = load_pipeline_state(out_dir, pid)
    if not state:
        raise ValueError(f"Pipeline {pid} not found")
    clips = state.get("clips", [])
    if clip_index < 0 or clip_index >= len(clips):
        raise ValueError(f"Clip index {clip_index} out of range (0-{len(clips)-1})")

    clip = clips[clip_index]
    prompt = prompt_override or clip.get("image_prompt", "")
    if not prompt:
        raise ValueError("No image prompt for this clip")

    # Reference art-style lock: reruns re-apply the detected style prefix
    # (the pipeline prepends it at generation time, so the saved
    # image_prompt does not carry it). Motion-effect strip mirrors
    # _gen_image for the same reason.
    prompt = _strip_motion_effects(prompt)
    _style_prefix = _style_prefix_for((state.get("_params_snapshot") or {}).get("_reference_style") or "")
    if _style_prefix and not prompt.lower().startswith("maintain the same"):
        prompt = _style_prefix + prompt

    # Get image gen params from the saved pipeline state
    image_model = state.get("image_model") or "flux2_klein_9b"
    image_loras = state.get("image_loras") or {}
    image_params = state.get("image_params") or {}
    validation_params = _director_params_from_saved_state(state)
    validation_params["image_model"] = image_model
    _validate_director_models(validation_params, stages=("image",))

    # Determine the output directory before resolving the generated anchor:
    # unlike the user's upload path, that anchor is stored as a basename in
    # the pipeline workspace so saved pipelines remain portable.
    pipeline_file = _find_pipeline_file(out_dir, pid)
    clip_out_dir = os.path.dirname(pipeline_file) if pipeline_file else out_dir

    user_ref_path = state.get("reference_image_path") or ""
    ref_path = user_ref_path if os.path.isfile(user_ref_path) else ""
    persisted_anchor = state.get("generated_reference_image_filename") or ""
    anchor_to_persist = ""
    if (
        not ref_path
        and persisted_anchor
        and os.path.basename(persisted_anchor) == persisted_anchor
    ):
        candidate = os.path.join(clip_out_dir, persisted_anchor)
        if os.path.isfile(candidate):
            ref_path = candidate

    # Backward-compatible recovery for pipelines saved before generated
    # anchors were persisted: a valid first clip image is the safest visual
    # identity reference available.
    if not ref_path:
        for saved_clip in clips:
            saved_start = saved_clip.get("start_image_filename") or ""
            if not saved_start or os.path.basename(saved_start) != saved_start:
                continue
            candidate = os.path.join(clip_out_dir, saved_start)
            if os.path.isfile(candidate):
                ref_path = candidate
                anchor_to_persist = saved_start
                break

    # Build refs: main + character + location
    all_refs = []
    seen_refs = set()
    if ref_path:
        resolved_ref = os.path.normcase(os.path.realpath(ref_path))
        seen_refs.add(resolved_ref)
        all_refs.append(ref_path)
    for cp in (state.get("character_ref_paths") or []):
        resolved = os.path.normcase(os.path.realpath(cp)) if cp else ""
        if cp and os.path.isfile(cp) and resolved not in seen_refs:
            seen_refs.add(resolved)
            all_refs.append(cp)
    for lp in (state.get("location_ref_paths") or []):
        resolved = os.path.normcase(os.path.realpath(lp)) if lp else ""
        if lp and os.path.isfile(lp) and resolved not in seen_refs:
            seen_refs.add(resolved)
            all_refs.append(lp)
    all_refs = _limit_director_image_refs(
        image_model,
        all_refs,
        pid=pid,
    )

    gen_params = {
        "model_type": image_model,
        "prompt": prompt,
        "image_refs": all_refs,
        "image_mode": 1,
        "image_prompt_type": "",
        "num_inference_steps": image_params.get("num_inference_steps", 8),
        "guidance_scale": image_params.get("guidance_scale", 1),
        # A legacy no-reference pipeline must bootstrap with plain T2I.  Once
        # this image is saved below it becomes the durable anchor for every
        # later clip rerun.
        "video_prompt_type": "KI" if all_refs else "",
        "resolution": image_params.get("resolution", "1280x720"),
        "seed": -1,
        "settings_version": 2.52,
        "generation_mode": "image",
        "repeat_generation": 1,
        "negative_prompt": "",
        "video_length": 1,
        "activated_loras": image_loras.get("activated_loras", []),
        "loras_multipliers": " ".join(
            m.split(";")[0] for m in (image_loras.get("loras_multipliers", "") or "").split(" ") if m
        ),
        "_director_pipeline_id": pid,
        "_director_detached_operation": True,
    }
    with _pipeline_lock:
        repair_control = _pipeline_repairs.get(pid)
        repair_operation_id = (
            repair_control.get("operation_id") if repair_control else None
        )
    if repair_operation_id:
        gen_params["_director_repair_operation_id"] = repair_operation_id

    output_files = _submit_and_wait(gen_params, timeout_s=600, out_dir=clip_out_dir)
    new_filename = output_files[0] if output_files else ""

    if not new_filename:
        raise RuntimeError(
            "Start-image generation completed without a recorded output."
        )

    if not ref_path:
        anchor_to_persist = new_filename

    # Update the saved pipeline state
    def _update(s):
        s["clips"][clip_index]["start_image_filename"] = new_filename
        # A video generated from the previous start image is still useful
        # history, but it no longer represents this clip's current inputs.
        # Keep its filename for playback/ownership and mark it for regeneration.
        s["clips"][clip_index]["video_stale"] = bool(
            s["clips"][clip_index].get("video_filename")
        )
        if prompt_override:
            s["clips"][clip_index]["image_prompt"] = prompt_override
        if anchor_to_persist:
            s["generated_reference_image_filename"] = anchor_to_persist
            snapshot = s.get("_params_snapshot")
            if isinstance(snapshot, dict):
                snapshot["generated_reference_image_filename"] = (
                    anchor_to_persist
                )
    _update_saved_pipeline(out_dir, pid, _update)

    return {"filename": new_filename, "clip_index": clip_index}


def _slice_audio_segment(src_path: str, start_sec: float, duration_sec: float, dst_path: str) -> None:
    """Cut [start, start+duration] out of the source audio with ffmpeg.

    Mirrors shared/utils/audio_video.py's plain-subprocess ffmpeg usage.
    Output is normalized wav so the generation's audio loader never has to
    care what container the song came in.
    """
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{max(0.0, float(start_sec)):.3f}",
        "-t", f"{max(0.1, float(duration_sec)):.3f}",
        "-i", src_path,
        "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
        dst_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _audio_timeline_start(planned_clips: list[dict]) -> float:
    """Return the source-audio time represented by video frame zero."""
    if not planned_clips:
        return 0.0
    try:
        start_sec = float((planned_clips[0] or {}).get("start", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(start_sec) or start_sec <= 0:
        return 0.0
    return start_sec


def _director_reference_label(params: dict, kind: str, index: int) -> str:
    """Return a stable, human-readable role for an H3 Director reference."""

    labels = params.get(f"{kind}_ref_labels") or []
    if index < len(labels) and str(labels[index] or "").strip():
        return str(labels[index]).strip()
    if kind == "character":
        characters = params.get("characters") or []
        if index < len(characters):
            character = characters[index]
            if isinstance(character, dict):
                name = str(character.get("name") or "").strip()
                if name:
                    return name
    noun = "character" if kind == "character" else "location"
    return f"{noun} {index + 1}"


def _director_h3_reference_manifest(
    params: dict,
    clip_image_path: str | None,
    *,
    out_dir: str,
    drive_audio_path: str | None = None,
) -> list[dict]:
    """Compile Director assets into one Ref2VA manifest for a single shot.

    When present, a generated shot image is a soft composition/cast reference,
    not a fixed first frame. In the normal Ref2VA workflow it is omitted and
    original character/location uploads are mapped directly. Audio roles are
    explicit because H3 treats a song or dialogue timeline very differently
    from a voice sample.
    """

    images: list[dict] = []
    seen_images: set[str] = set()

    def add_image(path: str, role: str, intent: str) -> None:
        if len(images) >= 9:
            return
        candidate = str(path or "").strip()
        if not candidate or not os.path.isfile(candidate):
            return
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized in seen_images:
            return
        seen_images.add(normalized)
        images.append({
            "type": "image",
            "path": candidate,
            "role": role,
            "image_intent": intent,
        })

    add_image(
        clip_image_path,
        "the intended composition, cast placement, wardrobe, and setting for this shot",
        "composition",
    )

    primary_reference = str(params.get("reference_image_path") or "").strip()
    if not primary_reference:
        generated_anchor = str(
            params.get("generated_reference_image_filename") or ""
        ).strip()
        if generated_anchor and os.path.basename(generated_anchor) == generated_anchor:
            primary_reference = os.path.join(out_dir, generated_anchor)
    add_image(
        primary_reference,
        "the primary cast identity and appearance",
        "identity",
    )

    for index, path in enumerate(params.get("character_ref_paths") or []):
        label = _director_reference_label(params, "character", index)
        add_image(
            path,
            f"the identity and appearance of {label}",
            "identity",
        )
    for index, path in enumerate(params.get("location_ref_paths") or []):
        label = _director_reference_label(params, "location", index)
        add_image(
            path,
            f"the environment and location named {label}",
            "scene",
        )

    references: list[dict] = list(images)
    if drive_audio_path and os.path.isfile(drive_audio_path):
        references.append({
            "type": "audio",
            "path": drive_audio_path,
            "role": "the exact performance and timing for this shot",
            "audio_intent": "drive",
        })

    voice_reference = str(params.get("voice_reference") or "").strip()
    if voice_reference and os.path.isfile(voice_reference):
        drive_normalized = (
            os.path.normcase(os.path.abspath(drive_audio_path))
            if drive_audio_path else ""
        )
        voice_normalized = os.path.normcase(os.path.abspath(voice_reference))
        if voice_normalized != drive_normalized:
            voice_role = str(params.get("voice_reference_role") or "").strip()
            if not voice_role:
                voice_role = _director_reference_label(params, "character", 0)
            references.append({
                "type": "audio",
                "path": voice_reference,
                "role": f"the voice of {voice_role}",
                "audio_intent": "voice",
            })
    return references


def _director_same_logical_scene(
    first_plan: dict,
    first_clip: dict,
    second_plan: dict,
    second_clip: dict,
) -> bool:
    """Whether adjacent bounded shots should share a final-frame handoff."""

    # Native H3 planning can deliberately mark a pair as one uninterrupted
    # shot continued across the model's duration boundary. Only that explicit
    # strategy may use the true final frame as the next start frame; ordinary
    # same-scene editorial cuts remain independent renders.
    first_group = str(
        first_clip.get("_director_continuity_group")
        or first_plan.get("_director_continuity_group")
        or ""
    ).strip()
    second_group = str(
        second_clip.get("_director_continuity_group")
        or second_plan.get("_director_continuity_group")
        or ""
    ).strip()
    second_strategy = str(
        second_clip.get("_director_continuity_strategy")
        or second_plan.get("_director_continuity_strategy")
        or ""
    ).strip().lower()
    if (
        first_group
        and first_group == second_group
        and second_strategy == "extend_previous"
    ):
        return True

    first_sources = (
        first_clip.get("_director_source_clip_indices")
        or first_plan.get("_director_source_clip_indices")
        or []
    )
    second_sources = (
        second_clip.get("_director_source_clip_indices")
        or second_plan.get("_director_source_clip_indices")
        or []
    )
    try:
        shared = set(int(value) for value in first_sources) & set(
            int(value) for value in second_sources
        )
    except (TypeError, ValueError):
        return False
    if not shared:
        return False
    try:
        first_segment = int(first_clip.get("_director_segment_index", 0) or 0)
        second_segment = int(second_clip.get("_director_segment_index", 0) or 0)
    except (TypeError, ValueError):
        return True
    return second_segment == first_segment + 1


def _extract_director_continuation_frame(
    video_path: str,
    destination: str,
) -> str:
    """Write the true final frame of one bounded H3 segment as a PNG."""

    import decord
    from PIL import Image as PILImage

    reader = decord.VideoReader(video_path)
    try:
        if len(reader) <= 0:
            raise ValueError("video contains no frames")
        frame = reader[len(reader) - 1].asnumpy()
    finally:
        del reader
    PILImage.fromarray(frame).save(destination)
    return destination


def _quantize_clip_frame_schedule(
    requested_frames: list[float], min_frames: int, latent_size: int,
) -> list[int]:
    """Match Director's carried rounding for a sequence of clip lengths."""
    latent_size = max(1, int(latent_size or 1))
    min_frames = max(1, int(min_frames or 1))
    carried: list[int] = []
    carry = 0.0
    for frame_count in requested_frames:
        target = float(frame_count) + carry
        quantized = max(
            round((target - 1) / latent_size) * latent_size + 1,
            min_frames,
        )
        carry = target - quantized
        carried.append(int(quantized))
    return carried


@_exclusive_pipeline_operation
def rerun_clip_video(out_dir: str, pid: str, clip_index: int, prompt_override: str = None) -> dict:
    return _rerun_clip_video_impl(out_dir, pid, clip_index, prompt_override)


def _rerun_clip_video_impl(out_dir: str, pid: str, clip_index: int, prompt_override: str = None) -> dict:
    """Re-generate the video for a single clip. Returns {job_id, filename} or raises."""
    state = load_pipeline_state(out_dir, pid)
    if not state:
        raise ValueError(f"Pipeline {pid} not found")
    clips = state.get("clips", [])
    if clip_index < 0 or clip_index >= len(clips):
        raise ValueError(f"Clip index {clip_index} out of range (0-{len(clips)-1})")

    clip = clips[clip_index]
    prompt = prompt_override or clip.get("video_prompt", "")
    if not prompt:
        raise ValueError("No video prompt for this clip")

    snapshot = state.get("_params_snapshot") or {}
    video_model = state.get("video_model") or "ltx2_22B_distilled_1_1"
    prompt_plan = {
        "video_prompt": prompt,
        "_director_h3_source_prompt": (
            prompt
            if prompt_override is not None
            else clip.get("_director_h3_source_prompt") or prompt
        ),
        "_director_h3_compiled_prompt": (
            "" if prompt_override is not None
            else clip.get("_director_h3_compiled_prompt") or prompt
        ),
        "_director_dialogue_beats": (
            [] if prompt_override is not None
            else clip.get("_director_dialogue_beats", []) or []
        ),
        "_director_subjects_on_screen": (
            clip.get("_director_subjects_on_screen", []) or []
        ),
        "_director_duration_sec": clip.get("_director_duration_sec"),
        "_director_h3_prompt_mode": clip.get("_director_h3_prompt_mode"),
        "_director_h3_model_family": clip.get("_director_h3_model_family"),
        "_director_speaker_registry": clip.get("_director_speaker_registry") or {},
        "_director_project_context": (
            clip.get("_director_project_context")
            or state.get("scene_description")
            or snapshot.get("scene_description")
            or ""
        ),
        "_director_opening_blocking": clip.get("_director_opening_blocking", ""),
        "_director_closing_blocking": clip.get("_director_closing_blocking", ""),
        "_director_audio_plan": clip.get("_director_audio_plan") or {},
    }
    _preflight_h3_director_prompts(video_model, [prompt_plan], pid=pid)
    prompt = prompt_plan["video_prompt"]
    video_loras = state.get("video_loras") or {}
    video_params = state.get("video_params") or {}
    shot_image_policy = _saved_pipeline_shot_image_policy(state)
    uses_shot_images = shot_images_required(shot_image_policy)
    validation_params = _director_params_from_saved_state(state)
    validation_params["video_model"] = video_model
    _validate_director_models(validation_params, stages=("video",))

    # Determine the output directory
    pipeline_file = _find_pipeline_file(out_dir, pid)
    clip_out_dir = os.path.dirname(pipeline_file) if pipeline_file else out_dir

    # Generated-image projects retain the strict I2V contract. Prompt-only
    # and direct-reference projects intentionally have no start-image file.
    start_path = ""
    if uses_shot_images:
        start_img = clip.get("start_image_filename")
        if _invalid_saved_media_numbers(
            [start_img], 1, clip_out_dir, "image",
        ):
            raise ValueError(
                "This clip has no valid start image. Regenerate its start image "
                "before regenerating video."
            )
        start_path = os.path.join(clip_out_dir, start_img)

    # Reconstruct the SAME carried frame schedule used by a full Director run.
    # Generators only accept lengths on a model-specific latent lattice. A
    # standalone rerun previously floored this one clip independently, losing
    # as many as latent_size-1 frames every time (over a second on a 32-frame
    # lattice). Those losses shifted every later cut against the soundtrack.
    fps = snapshot.get("fps", 16)
    model_def = {}
    try:
        model_def = _wgp.get_model_def(video_model) or {}
        if model_def and model_def.get("fps"):
            fps = model_def["fps"]
    except Exception:
        pass
    try:
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("invalid fps")
    except (TypeError, ValueError):
        fps = 16.0
    director_strategy = video_strategy(model_def)
    execution_profile = _saved_director_video_execution_profile(
        state,
        model_def=model_def,
    )
    try:
        min_frames, _, latent_size = _wgp.get_model_min_frames_and_step(video_model)
    except Exception:
        min_frames, latent_size = 17, 8

    requested_frames = []
    planned_clips = []
    for saved_clip in clips:
        saved_plan = saved_clip.get("planned_clip") or {}
        planned_clips.append(saved_plan)
        try:
            saved_duration = float(saved_plan.get("duration_sec") or 0)
        except (TypeError, ValueError):
            saved_duration = 0.0
        if saved_duration <= 0:
            try:
                saved_duration = float(saved_plan.get("end", 0) or 0) - float(
                    saved_plan.get("start", 0) or 0
                )
            except (TypeError, ValueError):
                saved_duration = 0.0
        if saved_duration > 0:
            frame_count = round(saved_duration * fps)
        else:
            try:
                frame_count = int(saved_plan.get("duration_frames") or 0)
            except (TypeError, ValueError):
                frame_count = 0
            if frame_count <= 0:
                frame_count = round(20 * fps)
        requested_frames.append(max(
            frame_count, round(5 * fps),
        ))
    if director_strategy in {BOUNDED_START_END, OMNI_REFERENCE}:
        frame_schedule = []
        maximum_frames = int(
            execution_profile.get("effective_max_frames")
            or model_def.get("frames_maximum")
            or 345
        )
        frame_step = int(model_def.get("frames_steps") or 17)
        for index, saved_clip in enumerate(clips):
            saved_plan = saved_clip.get("planned_clip") or {}
            try:
                frame_count = int(saved_plan.get("duration_frames") or 0)
            except (TypeError, ValueError):
                frame_count = 0
            if not (
                min_frames <= frame_count <= maximum_frames
                and (frame_count - min_frames) % max(1, frame_step) == 0
            ):
                raise ValueError(
                    f"Saved shot {index + 1} does not have a valid native "
                    f"{video_model} duration within this project's "
                    f"{maximum_frames}-frame one-pass limit. Re-plan this "
                    "older project before rerunning it."
                )
            validate_director_execution_frames(
                execution_profile,
                frame_count,
                label=f"Saved Director shot {index + 1}",
            )
            frame_schedule.append(frame_count)
        _validate_saved_profile_for_current_hardware(
            state,
            execution_profile,
            model_def,
            frame_schedule,
        )
    else:
        frame_schedule = _quantize_clip_frame_schedule(
            requested_frames, min_frames, latent_size,
        )
    video_length = frame_schedule[clip_index]
    native_window_frames = _director_native_window_frames(
        video_model,
        model_def,
        fps=fps,
        min_frames=min_frames,
        latent_size=latent_size,
    )
    try:
        saved_window_count = int(clip.get("window_count", 1) or 1)
    except (TypeError, ValueError):
        saved_window_count = 1
    clip_uses_planned_windows = (
        saved_window_count > 1 or bool(clip.get("keyframe_prompts"))
    )
    if director_strategy != ROLLING_WINDOW:
        rerun_window_frames = video_length
    elif (
        native_window_frames is not None
        and (
            not model_def.get("custom_frames_injection")
            or clip_uses_planned_windows
        )
    ):
        rerun_window_frames = native_window_frames
    else:
        rerun_window_frames = video_length + latent_size + 1
    print(
        f"[Pipeline {pid}] Clip {clip_index} rerun frame budget: "
        f"{video_length} frames at {fps:g} fps ({video_length / fps:.3f}s)"
    )

    continuation_path = None
    if (
        director_strategy == BOUNDED_START_END
        and not uses_shot_images
        and clip_index > 0
    ):
        previous_clip = clips[clip_index - 1]
        if _director_same_logical_scene(
            previous_clip,
            previous_clip.get("planned_clip") or {},
            clip,
            clip.get("planned_clip") or {},
        ):
            previous_video = previous_clip.get("video_filename")
            if _invalid_saved_media_numbers(
                [previous_video], 1, clip_out_dir, "video",
            ):
                raise ValueError(
                    "This shot continues the preceding H3 segment. "
                    "Regenerate the preceding video clip first."
                )
            pid_token = re.sub(r"[^A-Za-z0-9_-]", "_", pid)[:32]
            continuation_path = os.path.join(
                clip_out_dir,
                f"_director_h3_continue_{pid_token}_c{clip_index}_"
                f"{uuid.uuid4().hex[:8]}.png",
            )
            _extract_director_continuation_frame(
                os.path.join(clip_out_dir, previous_video),
                continuation_path,
            )
            start_path = continuation_path

    gen_params = {
        "model_type": video_model,
        "prompt": prompt,
        "image_mode": 0,
        "image_prompt_type": (
            "" if director_strategy == OMNI_REFERENCE
            else "S" if start_path
            else ""
        ),
        "num_inference_steps": video_params.get("num_inference_steps", 8),
        "guidance_scale": video_params.get("guidance_scale", 1),
        "resolution": (
            execution_profile.get("normalized_resolution")
            or video_params.get("resolution", "1280x720")
        ),
        "video_length": video_length,
        # Match the full Director run: LTX-2 may use one expanded window for
        # an ordinary shot, while other story models retain their native
        # rolling-window length. _submit_and_wait returns the final cumulative
        # output, so a multi-window rerun still records the complete clip.
        "sliding_window_size": rerun_window_frames,
        "seed": -1,
        "settings_version": 2.52,
        "generation_mode": "video",
        "repeat_generation": 1,
        "negative_prompt": "",
        "activated_loras": video_loras.get("activated_loras", []),
        "loras_multipliers": " ".join(
            m.split(";")[0] for m in (video_loras.get("loras_multipliers", "") or "").split(" ") if m
        ),
        "_director_pipeline_id": pid,
        "_director_detached_operation": True,
        "_director_video_execution_profile": execution_profile,
        "minimax_h3_turbo_mode": bool(
            video_params.get("minimax_h3_turbo_mode")
        ),
    }
    with _pipeline_lock:
        repair_control = _pipeline_repairs.get(pid)
        repair_operation_id = (
            repair_control.get("operation_id") if repair_control else None
        )
    if repair_operation_id:
        gen_params["_director_repair_operation_id"] = repair_operation_id
    if start_path and director_strategy != OMNI_REFERENCE:
        gen_params["image_start"] = start_path
    if (
        uses_shot_images
        and director_strategy == BOUNDED_START_END
        and clip_index + 1 < len(clips)
    ):
        next_clip = clips[clip_index + 1]
        first_plan = clip.get("planned_clip") or {}
        second_plan = next_clip.get("planned_clip") or {}
        if _director_same_logical_scene(clip, first_plan, next_clip, second_plan):
            next_image = next_clip.get("start_image_filename")
            next_path = os.path.join(clip_out_dir, next_image) if next_image else ""
            if next_path and os.path.isfile(next_path):
                gen_params["image_end"] = next_path
                gen_params["image_prompt_type"] = "SE"

    # Soundtrack conditioning. The original pipeline run passes the FULL
    # song as audio_guide (audio_prompt_type "A") and wgp slices it across
    # clips internally — a single-clip rerun gets none of that context, so
    # without this block the model invents its own audio and the
    # regenerated clip no longer matches the music video's soundtrack.
    # Slice the song to this clip's window and condition on it, mirroring
    # the segment the clip was originally generated against.
    pipeline_type = state.get("pipeline_type") or snapshot.get("pipeline_type") or "music_video"
    audio_path = snapshot.get("audio_path") or ""
    audio_origin_frames = round(_audio_timeline_start(planned_clips) * fps)
    clip_start = (
        audio_origin_frames + sum(frame_schedule[:clip_index])
    ) / fps
    clip_duration_sec = video_length / fps
    slice_path = None
    if (
        director_strategy == OMNI_REFERENCE
        and pipeline_type != "short_film_story"
        and (not audio_path or not os.path.isfile(audio_path))
    ):
        raise ValueError(
            "This H3 Omni project no longer has its source soundtrack or "
            "dialogue audio. Restore that file before rerunning the clip."
        )
    if pipeline_type != "short_film_story" and audio_path and os.path.isfile(audio_path):
        pid_token = re.sub(r"[^A-Za-z0-9_-]", "_", pid)[:32]
        slice_path = os.path.join(
            clip_out_dir,
            f"_rerun_audio_{pid_token}_c{clip_index}_{uuid.uuid4().hex[:8]}.wav",
        )
        try:
            _slice_audio_segment(
                audio_path, clip_start, clip_duration_sec, slice_path,
            )
            if director_strategy != OMNI_REFERENCE:
                gen_params["audio_prompt_type"] = "A"
                gen_params["audio_guide"] = slice_path
                if snapshot.get("audio_scale") is not None:
                    gen_params["audio_scale"] = snapshot["audio_scale"]
            print(f"[Pipeline {pid}] Clip {clip_index} rerun conditioned on song segment "
                  f"{float(clip_start):.3f}s-"
                  f"{float(clip_start) + float(clip_duration_sec):.3f}s")
        except Exception as e:
            if director_strategy == OMNI_REFERENCE:
                if slice_path and os.path.isfile(slice_path):
                    try:
                        os.remove(slice_path)
                    except OSError:
                        pass
                raise RuntimeError(
                    f"Could not prepare H3 Omni audio for clip {clip_index + 1}: {e}"
                ) from e
            print(f"[Pipeline {pid}] Clip {clip_index} audio slice failed; "
                  f"regenerating without soundtrack conditioning: {e}")

    if director_strategy == OMNI_REFERENCE:
        manifest_params = dict(snapshot)
        manifest_params.setdefault(
            "generated_reference_image_filename",
            state.get("generated_reference_image_filename"),
        )
        gen_params["minimax_h3_references"] = _director_h3_reference_manifest(
            manifest_params,
            start_path if uses_shot_images else None,
            out_dir=clip_out_dir,
            drive_audio_path=slice_path,
        )
        if not any(
            reference.get("type") in {"image", "video"}
            for reference in gen_params["minimax_h3_references"]
        ):
            raise ValueError(
                "This H3 Omni project no longer has a valid main, character, "
                "or location image. Restore a visual reference before rerunning."
            )
        gen_params["minimax_h3_reference_detail"] = "match"

    if str(video_model or "").lower().startswith("minimax_h3"):
        final_prompt_mode = (
            "ref2va"
            if director_strategy == OMNI_REFERENCE
            else "fl2va"
            if gen_params.get("image_end")
            else "i2va"
            if start_path
            else "t2va"
        )
        final_references = (
            [gen_params.get("minimax_h3_references") or []]
            if director_strategy == OMNI_REFERENCE
            else None
        )
        _preflight_h3_director_prompts(
            video_model,
            [prompt_plan],
            pid=pid,
            prompt_modes=[final_prompt_mode],
            durations=[clip_duration_sec],
            reference_manifests=final_references,
        )
        prompt = prompt_plan["video_prompt"]
        gen_params["prompt"] = prompt

    try:
        output_files = _submit_and_wait(
            gen_params, timeout_s=23600, out_dir=clip_out_dir,
        )
    finally:
        if slice_path and os.path.isfile(slice_path):
            try:
                os.remove(slice_path)
            except OSError:
                pass
        if continuation_path and os.path.isfile(continuation_path):
            try:
                os.remove(continuation_path)
            except OSError:
                pass
    # Sliding-window generations save CUMULATIVE progress files (each save
    # is the video so far) — the LAST file is the complete clip. With the
    # single-window sizing above there is normally exactly one file, but
    # taking the last is correct in every case; taking the first recorded
    # a 5s preview of a 13s clip.
    new_filename = output_files[-1] if output_files else ""

    if not new_filename:
        raise RuntimeError(
            "Video generation completed without a recorded output."
        )

    def _update(s):
        s["clips"][clip_index]["video_filename"] = new_filename
        s["clips"][clip_index]["video_stale"] = False
        s["clips"][clip_index]["video_prompt"] = prompt
        s["clips"][clip_index]["_director_vocal_contract"] = (
            prompt_plan.get("_director_vocal_contract")
        )
        for key in (
            "_director_h3_source_prompt",
            "_director_h3_compiled_prompt",
            "_director_h3_prompt_mode",
            "_director_h3_model_family",
            "_director_speaker_registry",
            "_director_project_context",
            "_director_opening_blocking",
            "_director_closing_blocking",
            "_director_audio_plan",
        ):
            if prompt_plan.get(key) is not None:
                s["clips"][clip_index][key] = prompt_plan.get(key)
        if new_filename not in s.get("output_files", []):
            s.setdefault("output_files", []).append(new_filename)
        s["video_execution_profile"] = execution_profile
        snapshot_params = s.get("_params_snapshot")
        if isinstance(snapshot_params, dict):
            snapshot_params["_director_video_execution_profile"] = (
                execution_profile
            )
            snapshot_video_params = snapshot_params.setdefault(
                "video_params", {}
            )
            snapshot_video_params["resolution"] = gen_params["resolution"]
    _update_saved_pipeline(out_dir, pid, _update)

    return {"filename": new_filename, "clip_index": clip_index}


@_exclusive_pipeline_operation
def rejoin_clips(out_dir: str, pid: str) -> dict:
    return _rejoin_clips_impl(out_dir, pid)


def _rejoin_clips_impl(out_dir: str, pid: str) -> dict:
    """Re-join all clips from a saved pipeline using current best versions. Returns {filename}."""
    state = load_pipeline_state(out_dir, pid)
    if not state:
        raise ValueError(f"Pipeline {pid} not found")

    pipeline_file = _find_pipeline_file(out_dir, pid)
    clip_out_dir = os.path.dirname(pipeline_file) if pipeline_file else out_dir

    clips = state.get("clips", [])
    stale_clip_numbers = [
        str(index + 1)
        for index, clip in enumerate(clips)
        if clip.get("video_stale")
    ]
    if stale_clip_numbers:
        raise ValueError(
            "Regenerate stale video clip(s) "
            f"{', '.join(stale_clip_numbers)} before rejoining."
        )

    if shot_images_required(_saved_pipeline_shot_image_policy(state)):
        invalid_start_numbers = _invalid_saved_media_numbers(
            [clip.get("start_image_filename") for clip in clips],
            len(clips),
            clip_out_dir,
            "image",
        )
        if invalid_start_numbers:
            invalid_labels = ", ".join(
                str(index) for index in invalid_start_numbers
            )
            raise ValueError(
                "Regenerate missing or invalid start image(s) for clip(s) "
                f"{invalid_labels} before rejoining."
            )

    invalid_video_numbers = _invalid_saved_media_numbers(
        [clip.get("video_filename") for clip in clips],
        len(clips),
        clip_out_dir,
        "video",
    )
    if invalid_video_numbers:
        invalid_labels = ", ".join(
            str(index) for index in invalid_video_numbers
        )
        raise ValueError(
            "Regenerate missing or invalid video clip(s) "
            f"{invalid_labels} before rejoining."
        )

    video_files = [
        os.path.join(clip_out_dir, clip["video_filename"])
        for clip in clips
    ]

    if len(video_files) < 2:
        raise ValueError(f"Need at least 2 video clips to rejoin, found {len(video_files)}")

    # Lay the pristine source song over the rejoined video, exactly like the
    # original pipeline's multiclip join does — per-clip embedded audio is a
    # windowed generation, the full track is the real soundtrack. Story-mode
    # pipelines (no song) concat with the clips' own audio.
    snapshot = state.get("_params_snapshot") or {}
    audio_path = snapshot.get("audio_path") or None
    if audio_path and not os.path.isfile(audio_path):
        audio_path = None
    audio_start_sec = _audio_timeline_start([
        clip.get("planned_clip") or {} for clip in clips
    ]) if audio_path else 0.0

    import time as _time
    timestamp = _time.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    output_name = f"{timestamp}_rejoin_multiclip.mp4"
    output_path = os.path.join(clip_out_dir, output_name)

    try:
        # concatenate_multi_clip_videos is the join the original pipeline
        # uses (ffmpeg concat FILTER, re-encodes to a uniform format). The
        # previously-called wgp.concatenate_videos never existed — this path
        # was unreachable until the video_filename backfill fix, so the
        # AttributeError only surfaced now.
        ok = _wgp.concatenate_multi_clip_videos(
            video_files,
            output_path,
            audio_path,
            audio_start_sec=audio_start_sec,
        )
        if not ok or not os.path.isfile(output_path):
            raise RuntimeError("ffmpeg concatenation failed (see server log for the clip that broke it)")
        print(f"[Pipeline] Rejoined {len(video_files)} clips -> {output_name}")

        # Update pipeline state
        def _update(s):
            if output_name not in s.get("output_files", []):
                s.setdefault("output_files", []).append(output_name)
        _update_saved_pipeline(out_dir, pid, _update)

        return {"filename": output_name}
    except Exception as e:
        raise RuntimeError(f"Rejoin failed: {e}")


def _plan_pipeline_repair(out_dir: str, pid: str, state: dict) -> dict:
    """Build a deterministic repair plan from recorded files on disk."""
    pipeline_file = _find_pipeline_file(out_dir, pid)
    if not pipeline_file:
        raise ValueError(f"Pipeline {pid} not found")
    clip_out_dir = os.path.dirname(pipeline_file)
    clips = state.get("clips") or []

    requires_shot_images = shot_images_required(
        _saved_pipeline_shot_image_policy(state)
    )
    invalid_images = (
        {
            number - 1
            for number in _invalid_saved_media_numbers(
                [clip.get("start_image_filename") for clip in clips],
                len(clips),
                clip_out_dir,
                "image",
            )
        }
        if requires_shot_images
        else set()
    )
    invalid_videos = {
        number - 1
        for number in _invalid_saved_media_numbers(
            [clip.get("video_filename") for clip in clips],
            len(clips),
            clip_out_dir,
            "video",
        )
    }
    image_indices = sorted(invalid_images)
    video_indices = sorted(
        invalid_videos
        | invalid_images
        | {
            index
            for index, clip in enumerate(clips)
            if clip.get("video_stale")
        }
    )

    missing_image_prompts = [
        index + 1 for index in image_indices
        if not str(clips[index].get("image_prompt") or "").strip()
    ]
    if missing_image_prompts:
        labels = ", ".join(str(index) for index in missing_image_prompts)
        raise ValueError(
            f"Missing image prompt for repair clip(s) {labels}."
        )
    missing_video_prompts = [
        index + 1 for index in video_indices
        if not str(clips[index].get("video_prompt") or "").strip()
    ]
    if missing_video_prompts:
        labels = ", ".join(str(index) for index in missing_video_prompts)
        raise ValueError(
            f"Missing video prompt for repair clip(s) {labels}."
        )

    should_rejoin = len(clips) >= 2
    return {
        "image_indices": image_indices,
        "video_indices": video_indices,
        "should_rejoin": should_rejoin,
        "clip_count": len(clips),
        "total": (
            len(image_indices)
            + len(video_indices)
            + (1 if should_rejoin else 0)
        ),
    }


def _repair_queue_message(plan: dict) -> str:
    parts = []
    image_count = len(plan["image_indices"])
    video_count = len(plan["video_indices"])
    if image_count:
        parts.append(f"{image_count} image{'s' if image_count != 1 else ''}")
    if video_count:
        parts.append(f"{video_count} video{'s' if video_count != 1 else ''}")
    if plan["should_rejoin"]:
        parts.append("final join")
    return "Queued " + (", ".join(parts) if parts else "repair check")


def _persist_repair_state_unlocked(
    out_dir: str,
    pid: str,
    control: dict,
    *,
    replace: bool = False,
    **updates,
) -> Optional[dict]:
    """Persist repair status while the caller holds control['state_lock']."""
    operation_id = control["operation_id"]
    now = time.time()

    def _update(state):
        existing = state.get("repair")
        if (
            not replace
            and isinstance(existing, dict)
            and existing.get("operation_id") != operation_id
        ):
            return
        repair = {} if replace else dict(existing or {})
        repair.update(updates)
        repair["operation_id"] = operation_id
        repair["updated_at"] = now
        state["repair"] = repair

    saved = _update_saved_pipeline(out_dir, pid, _update)
    repair = (saved or {}).get("repair")
    if not isinstance(repair, dict) or repair.get("operation_id") != operation_id:
        return None
    snapshot = dict(repair)
    with _pipeline_lock:
        current = _pipeline_repairs.get(pid)
        if current is control:
            current["snapshot"] = snapshot
    return snapshot


def _persist_repair_state(
    out_dir: str,
    pid: str,
    control: dict,
    *,
    replace: bool = False,
    **updates,
) -> Optional[dict]:
    with control["state_lock"]:
        return _persist_repair_state_unlocked(
            out_dir, pid, control, replace=replace, **updates,
        )


def _raise_if_repair_cancelled(control: dict) -> None:
    if control["cancel_event"].is_set():
        raise _RepairCancelledError("Repair cancelled")


def _finish_pipeline_repair(
    out_dir: str,
    pid: str,
    control: dict,
    *,
    status: str,
    phase: str,
    current: int,
    total: int,
    message: str,
    error: Optional[str] = None,
    result_filename: Optional[str] = None,
) -> Optional[dict]:
    with control["state_lock"]:
        # Decide completion-versus-cancellation while holding the same lock
        # used by cancel_pipeline_repair. Whichever path enters first wins:
        # completion marks the control as finishing, while cancellation sets
        # the absorbing event before a terminal snapshot can be chosen.
        with _pipeline_lock:
            current_control = _pipeline_repairs.get(pid)
            if current_control is control:
                current_control["finishing"] = True
            cancel_requested = control["cancel_event"].is_set()
        if status == "completed" and cancel_requested:
            status = "cancelled"
            phase = "cancelled"
            message = "Repair cancelled"
            error = None
        return _persist_repair_state_unlocked(
            out_dir,
            pid,
            control,
            status=status,
            phase=phase,
            current=current,
            total=total,
            clip_index=None,
            message=message,
            error=error,
            cancel_requested=cancel_requested,
            completed_at=time.time(),
            result_filename=result_filename,
        )


def _run_pipeline_repair(
    out_dir: str,
    pid: str,
    control: dict,
    plan: dict,
) -> None:
    """Run one full Dashboard repair independently of the browser."""
    current = 0
    total = plan["total"]
    clip_count = plan["clip_count"]
    result_filename = None
    try:
        _raise_if_repair_cancelled(control)
        _persist_repair_state(
            out_dir,
            pid,
            control,
            status="running",
            phase="images" if plan["image_indices"] else "videos",
            current=current,
            total=total,
            clip_index=None,
            message="Starting repair",
            error=None,
        )

        for clip_index in plan["image_indices"]:
            _raise_if_repair_cancelled(control)
            _persist_repair_state(
                out_dir,
                pid,
                control,
                status="running",
                phase="images",
                current=current,
                total=total,
                clip_index=clip_index,
                message=f"Generating start image for clip {clip_index + 1} of {clip_count}",
                error=None,
            )
            _rerun_clip_image_impl(out_dir, pid, clip_index)
            _raise_if_repair_cancelled(control)
            current += 1
            _persist_repair_state(
                out_dir,
                pid,
                control,
                status="running",
                phase="images",
                current=current,
                total=total,
                clip_index=clip_index,
                message=f"Finished start image for clip {clip_index + 1}",
                error=None,
            )

        for clip_index in plan["video_indices"]:
            _raise_if_repair_cancelled(control)
            _persist_repair_state(
                out_dir,
                pid,
                control,
                status="running",
                phase="videos",
                current=current,
                total=total,
                clip_index=clip_index,
                message=f"Generating video for clip {clip_index + 1} of {clip_count}",
                error=None,
            )
            _rerun_clip_video_impl(out_dir, pid, clip_index)
            _raise_if_repair_cancelled(control)
            current += 1
            _persist_repair_state(
                out_dir,
                pid,
                control,
                status="running",
                phase="videos",
                current=current,
                total=total,
                clip_index=clip_index,
                message=f"Finished video for clip {clip_index + 1}",
                error=None,
            )

        if plan["should_rejoin"]:
            _raise_if_repair_cancelled(control)
            _persist_repair_state(
                out_dir,
                pid,
                control,
                status="running",
                phase="rejoin",
                current=current,
                total=total,
                clip_index=None,
                message=f"Joining {clip_count} repaired clips",
                error=None,
            )
            result = _rejoin_clips_impl(out_dir, pid)
            result_filename = result.get("filename")
            _raise_if_repair_cancelled(control)
            current += 1

        _finish_pipeline_repair(
            out_dir,
            pid,
            control,
            status="completed",
            phase="completed",
            current=current,
            total=total,
            message=(
                "Repair complete and clips joined"
                if plan["should_rejoin"]
                else "Repair complete"
            ),
            result_filename=result_filename,
        )
    except (GenerationCancelledError, _RepairCancelledError):
        _finish_pipeline_repair(
            out_dir,
            pid,
            control,
            status="cancelled",
            phase="cancelled",
            current=current,
            total=total,
            message="Repair cancelled",
        )
    except Exception as exc:
        print(f"[Pipeline {pid}] Repair failed: {exc}")
        traceback.print_exc()
        _finish_pipeline_repair(
            out_dir,
            pid,
            control,
            status="failed",
            phase="failed",
            current=current,
            total=total,
            message="Repair stopped after an error",
            error=str(exc),
        )
    finally:
        with _pipeline_lock:
            if _pipeline_repairs.get(pid) is control:
                _pipeline_repairs.pop(pid, None)
        _release_pipeline_operation(pid)


def _run_pipeline_repair_after_ready(
    out_dir: str,
    pid: str,
    control: dict,
    plan: dict,
) -> None:
    """Keep even a zero-unit worker alive until start publication finishes."""
    ready_event = control.get("ready_event")
    if ready_event is not None:
        ready_event.wait()
    # The starter owns cleanup when publication itself failed. In the rare
    # case a Thread implementation began running before start() raised, do
    # not let that worker execute a repair after the failed reservation.
    if control.get("start_error") is not None:
        return
    _run_pipeline_repair(out_dir, pid, control, plan)


def _repair_start_result(pid: str, control: dict) -> dict:
    """Wait for an atomic start reservation to publish its first snapshot."""
    ready_event = control.get("ready_event")
    if ready_event is not None:
        ready_event.wait()
    start_error = control.get("start_error")
    if start_error is not None:
        raise start_error
    return {
        "pipeline_id": pid,
        "repair": dict(control.get("snapshot") or {}),
    }


def start_pipeline_repair(out_dir: str, pid: str) -> dict:
    """Start or reconnect to a server-owned repair batch."""
    with _pipeline_lock:
        existing = _pipeline_repairs.get(pid)
        if existing is not None:
            control = existing
            starter = False
        else:
            # Claim the operation and publish a reservation in one critical
            # section. A simultaneous duplicate now waits for this starter's
            # persisted snapshot instead of falling into the claim gap and
            # receiving a spurious busy response.
            if not _claim_pipeline_operation_locked(pid):
                raise PipelineBusyError(
                    "Pipeline is still active; try again shortly."
                )
            operation_id = uuid.uuid4().hex[:12]
            control = {
                "operation_id": operation_id,
                "snapshot": {},
                "cancel_event": threading.Event(),
                "state_lock": threading.Lock(),
                "finishing": False,
                "thread": None,
                "ready_event": threading.Event(),
                "start_error": None,
            }
            _pipeline_repairs[pid] = control
            starter = True

    if not starter:
        return _repair_start_result(pid, control)

    try:
        state = load_pipeline_state(out_dir, pid)
        if not state:
            raise ValueError(f"Pipeline {pid} not found")
        plan = _plan_pipeline_repair(out_dir, pid, state)
        required_stages = tuple(
            stage
            for stage, indices in (
                ("image", plan["image_indices"]),
                ("video", plan["video_indices"]),
            )
            if indices
        )
        if required_stages:
            _validate_director_models(
                _director_params_from_saved_state(state),
                stages=required_stages,
            )
        started_at = time.time()
        initial = {
            "operation_id": control["operation_id"],
            "status": "queued",
            "phase": "queued",
            "current": 0,
            "total": plan["total"],
            "clip_index": None,
            "message": _repair_queue_message(plan),
            "error": None,
            "cancel_requested": False,
            "started_at": started_at,
            "updated_at": started_at,
            "completed_at": None,
            "result_filename": None,
        }
        with _pipeline_lock:
            if _pipeline_repairs.get(pid) is control:
                control["snapshot"] = dict(initial)

        persisted = _persist_repair_state(
            out_dir, pid, control, replace=True, **initial,
        )
        if not persisted:
            raise RuntimeError("Could not persist repair status")

        thread = threading.Thread(
            target=_run_pipeline_repair_after_ready,
            args=(out_dir, pid, control, plan),
            daemon=False,
            name=f"director-repair-{pid}",
        )
        with _pipeline_lock:
            control["thread"] = thread
        thread.start()
        control["ready_event"].set()
        return {"pipeline_id": pid, "repair": persisted}
    except BaseException as exc:
        try:
            _finish_pipeline_repair(
                out_dir,
                pid,
                control,
                status="failed",
                phase="failed",
                current=0,
                total=(control.get("snapshot") or {}).get("total", 0),
                message="Could not start repair",
                error=str(exc),
            )
        except Exception:
            traceback.print_exc()
        with _pipeline_lock:
            control["start_error"] = exc
            if _pipeline_repairs.get(pid) is control:
                _pipeline_repairs.pop(pid, None)
        control["ready_event"].set()
        _release_pipeline_operation(pid)
        raise


def cancel_pipeline_repair(out_dir: str, pid: str) -> Optional[dict]:
    """Request cancellation and abort the repair's in-flight child job."""
    with _pipeline_lock:
        control = _pipeline_repairs.get(pid)
        if not control:
            return None

    # A newly reserved repair has not persisted its operation snapshot yet.
    # Wait outside both locks so the starter can publish (or fail), then
    # revalidate the exact control below. The worker uses the same gate, so
    # cancel never acts on an old/no repair record during this handshake.
    ready_event = control.get("ready_event")
    if ready_event is not None:
        ready_event.wait()

    with control["state_lock"]:
        with _pipeline_lock:
            current = _pipeline_repairs.get(pid)
            if current is not control or current.get("finishing"):
                return dict(control.get("snapshot") or {})
            control["cancel_event"].set()
            # Keep the registry lock through job selection and abort. Without
            # this boundary the old repair could tear down, a successor could
            # register the same pid, and this late abort would cancel the
            # successor's child job instead.
            _abort_pipeline_jobs(pid)
        snapshot = _persist_repair_state_unlocked(
            out_dir,
            pid,
            control,
            status="cancelling",
            message="Cancelling repair after the current model step",
            cancel_requested=True,
        )
    return snapshot


def init(jobs_dict, run_gen_fn, wgp_module, gen_lock=None, active_gen_states=None):
    """Called by launch.py to wire up shared references."""
    global _jobs, _run_generation, _wgp, _gen_lock, _active_gen_states
    _jobs = jobs_dict
    _run_generation = run_gen_fn
    _wgp = wgp_module
    _gen_lock = gen_lock
    _active_gen_states = active_gen_states


class _DirectorOutputs(list):
    """List-compatible outputs that retain exact Director clip ownership."""

    def __init__(self, values, clip_output_files=None):
        super().__init__(values)
        self.clip_output_files = dict(clip_output_files or {})


class _GenerationTimeoutError(RuntimeError):
    def __init__(self, output_files: _DirectorOutputs):
        super().__init__("Generation timed out")
        self.output_files = output_files


class GenerationCancelledError(RuntimeError):
    """A detached Dashboard generation was cancelled after settling."""

    def __init__(self, output_files: _DirectorOutputs):
        super().__init__("Re-run cancelled")
        self.output_files = output_files


def _director_job_outputs(job: dict) -> _DirectorOutputs:
    """Collapse multi-window files to the final output for each clip."""
    snapshot = snapshot_job(job)
    output_files = list(snapshot.get("output_files") or [])
    clip_outputs = snapshot.get("clip_output_files") or {}
    if not isinstance(clip_outputs, dict) or not clip_outputs:
        return _DirectorOutputs(output_files)

    indexed = []
    for index, filename in clip_outputs.items():
        try:
            indexed.append((int(index), filename))
        except (TypeError, ValueError):
            continue
    indexed.sort(key=lambda item: item[0])
    collapsed = [filename for _, filename in indexed if filename]
    join_output = snapshot.get("join_output_file")
    if join_output and join_output not in collapsed:
        collapsed.append(join_output)
    return _DirectorOutputs(
        collapsed or output_files,
        {index: filename for index, filename in indexed if filename},
    )


def _submit_and_wait(params: dict, timeout_s: float = 600, workspace: str = None, out_dir: str = None) -> list[str]:
    """Submit a generation job and block until it completes.

    Returns list of output filenames. Raises on failure/timeout.
    """
    _prepare_director_generation_params(params)
    job_id = uuid.uuid4().hex[:8]
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "step": 0,
        "total_steps": 0,
        "phase": "",
        "message": "Queued",
        "created_at": time.time(),
        "params": params,
        "output_files": [],
        "error": None,
        "workspace": workspace,
        "out_dir": out_dir,
    }
    _dir_pid = params.get("_director_pipeline_id")
    _detached_operation = bool(params.get("_director_detached_operation"))
    _repair_operation_id = params.get("_director_repair_operation_id")
    _skip_generation = False

    def _run_tracked_generation() -> None:
        try:
            # A repair cancellation may win before this newly published
            # child thread begins executing. Do not invoke generation for a
            # detached repair child that registration already made terminal.
            # Ordinary pipeline cancellation still enters _run_generation so
            # its existing settle path can publish already-produced outputs.
            if _skip_generation:
                return
            _run_generation(job_id)
        finally:
            if _dir_pid:
                with _pipeline_lock:
                    child_jobs = _pipeline_child_jobs.get(_dir_pid)
                    if child_jobs is not None:
                        child_jobs.discard(job_id)
                        if not child_jobs:
                            _pipeline_child_jobs.pop(_dir_pid, None)

    # Run generation in a separate thread (it acquires _gen_lock internally).
    # The child lease outlives this waiter if cancellation cannot settle
    # promptly, keeping destructive Dashboard actions away from a live writer.
    # Non-daemon so the process stays alive if browser disconnects mid-generation.
    thread = threading.Thread(target=_run_tracked_generation, daemon=False)
    try:
        if _dir_pid:
            # Publish, lease, recheck repair cancellation, and start under one
            # registry boundary. If cancel scanned before this child existed,
            # its operation-scoped event is observed here before generation;
            # if it scans after, the job is already visible to that scan.
            with _pipeline_lock:
                _jobs[job_id] = job
                _pipeline_child_jobs.setdefault(_dir_pid, set()).add(job_id)
                if _detached_operation and _repair_operation_id:
                    repair_control = _pipeline_repairs.get(_dir_pid)
                    if (
                        repair_control is not None
                        and repair_control.get("operation_id")
                            == _repair_operation_id
                        and repair_control["cancel_event"].is_set()
                    ):
                        request_cancel(job)
                        _skip_generation = True
                elif not _detached_operation:
                    pipeline_cancelled = (
                        _pipelines.get(_dir_pid, {}).get("status")
                        == "cancelled"
                    )
                    if pipeline_cancelled:
                        request_cancel(job)
                thread.start()
        else:
            _jobs[job_id] = job
            thread.start()
    except BaseException:
        if _dir_pid:
            with _pipeline_lock:
                child_jobs = _pipeline_child_jobs.get(_dir_pid)
                if child_jobs is not None:
                    child_jobs.discard(job_id)
                    if not child_jobs:
                        _pipeline_child_jobs.pop(_dir_pid, None)
        raise

    # Wait for completion, mirroring job progress to pipeline status
    deadline = time.time() + timeout_s
    _abort_signalled = False
    while time.time() < deadline:
        j = _jobs.get(job_id)
        if not j:
            raise RuntimeError("Job disappeared")
        if j["status"] == "completed":
            return _director_job_outputs(j)
        if j["status"] == "cancelled":
            # Keep whatever clips finished before the abort (multi-clip
            # jobs accrue output_files per clip) — callers tolerate a
            # partial or empty list and check the pipeline status.
            print(f"[Pipeline] Job {job_id} cancelled")
            # Cancellation is published immediately. Settle the child only in
            # this background pipeline thread so it can publish files that
            # completed before the abort took effect.
            thread.join(timeout=_GENERATION_SETTLE_GRACE_S)
            if thread.is_alive():
                print(
                    f"[Pipeline] Job {job_id} is still shutting down; "
                    "pipeline remains busy"
                )
            settled = _jobs.get(job_id) or j
            settled_outputs = _director_job_outputs(settled)
            if _detached_operation:
                raise GenerationCancelledError(settled_outputs)
            return settled_outputs
        if j["status"] == "failed":
            err = j.get("error") or "Generation failed"
            print(f"[Pipeline] Job {job_id} failed: {err}")
            raise RuntimeError(err)
        # Backstop for stop_pipeline's abort: if the pipeline was cancelled
        # while this job runs (e.g. the job was submitted in the window
        # after the stop endpoint scanned _jobs), signal abort from here.
        if _dir_pid and not _detached_operation and not _abort_signalled:
            with _pipeline_lock:
                _cancelled = _pipelines.get(_dir_pid, {}).get("status") == "cancelled"
            if _cancelled:
                _abort_pipeline_jobs(_dir_pid)
                _abort_signalled = True
        # Mirror denoising step progress to pipeline status
        # Only update step/total_steps and message — preserve current/total for pipeline-level counts
        if _dir_pid and (j.get("step", 0) > 0 or j.get("total_steps", 0) > 0):
            with _pipeline_lock:
                p = _pipelines.get(_dir_pid)
                if p and "progress" in p:
                    p["progress"]["step"] = j.get("step", 0)
                    p["progress"]["total_steps"] = j.get("total_steps", 0)
                    p["progress"]["message"] = j.get("phase") or j.get("message") or "Generating..."
        time.sleep(min(1.0, max(0.01, deadline - time.time())))

    request_cancel(
        job,
        job_id=job_id,
        active_states=_active_gen_states or {},
    )
    thread.join(timeout=_GENERATION_SETTLE_GRACE_S)
    if thread.is_alive():
        print(
            f"[Pipeline] Timed-out job {job_id} is still shutting down; "
            "pipeline remains busy"
        )
    settled = _jobs.get(job_id) or job
    raise _GenerationTimeoutError(_director_job_outputs(settled))


def _update_pipeline(pid: str, **kwargs):
    """Thread-safe update; cancellation is an absorbing terminal state."""
    with _pipeline_lock:
        pipeline = _pipelines.get(pid)
        if not pipeline:
            return False
        if pipeline.get("status") == "cancelled":
            # Finished clips may still be reported after an in-flight abort,
            # but no later phase, completion, or failure may replace Stop.
            if set(kwargs) - _CANCELLED_ARTIFACT_FIELDS:
                return False
        pipeline.update(kwargs)
        return True


def _start_pipeline_worker(pid: str, *, resume: bool = False) -> None:
    """Start and track a Director worker until its ``finally`` completes."""
    thread = threading.Thread(
        target=_run_pipeline,
        args=(pid,),
        kwargs={"resume": resume},
        daemon=False,
    )
    with _pipeline_lock:
        if pid in _pipeline_threads:
            raise RuntimeError(f"Pipeline {pid} already has a worker")
        if _pipeline_child_jobs.get(pid):
            raise RuntimeError(
                f"Pipeline {pid} still has a generation child"
            )
        _pipeline_threads[pid] = thread
    try:
        thread.start()
    except BaseException as exc:
        with _pipeline_lock:
            if _pipeline_threads.get(pid) is thread:
                _pipeline_threads.pop(pid, None)
            pipeline = _pipelines.get(pid)
            if pipeline and pipeline.get("status") not in {
                "completed", "failed", "cancelled",
            }:
                pipeline["status"] = "failed"
                pipeline["phase"] = "failed"
                pipeline["error"] = f"Could not start pipeline worker: {exc}"
                pipeline["_completed_at"] = time.time()
                pipeline["progress"] = {
                    "current": 0,
                    "total": 0,
                    "message": "Could not start pipeline worker",
                    "step": 0,
                    "total_steps": 0,
                }
        _save_pipeline_state(pid)
        raise


def start_pipeline(params: dict) -> str:
    """Start a new director pipeline. Returns pipeline_id."""
    # Internal resume metadata must never be accepted from a fresh API request.
    # Otherwise a caller could nominate unrelated workspace media as this
    # pipeline's generated anchor and later influence repair/cleanup behavior.
    params.pop("generated_reference_image_filename", None)
    params.pop("_director_shot_image_policy", None)
    params.pop("_director_video_execution_profile", None)
    params["_director_shot_image_policy"] = (
        _resolve_fresh_shot_image_policy(params)
    )
    _validate_director_models(params)
    execution_profile = _create_director_video_execution_profile(params)
    if execution_profile.get("is_minimax_h3"):
        print(
            "[Pipeline] H3 Director execution profile: "
            f"{execution_profile.get('normalized_resolution')}, "
            f"{execution_profile.get('gpu_vram_gb', 0):g} GB, "
            f"one-pass max {execution_profile.get('effective_max_frames')} "
            f"frames ({execution_profile.get('effective_max_seconds', 0):.2f}s)"
            + (
                " [manual override]"
                if execution_profile.get("manual_override") else ""
            )
        )
    pid = uuid.uuid4().hex[:8]

    # Capture workspace at submission time — not at execution time
    workspace = params.pop("workspace", None)
    if workspace:
        # Resolve the output directory now, while we know the intended workspace
        from launch import _workspace_dir
        out_dir = _workspace_dir(workspace)
        print(f"[Pipeline] Workspace={workspace}, out_dir={out_dir}, wgp.save_path={_wgp.save_path}")
    else:
        out_dir = _wgp.save_path
        workspace = None
        print(f"[Pipeline] No workspace, using wgp.save_path={out_dir}")

    pipeline = {
        "id": pid,
        "status": "running",
        "phase": "planning",
        "auto_mode": params.get("auto_mode", True),
        "progress": {"current": 0, "total": 0, "message": "Starting...", "step": 0, "total_steps": 0},
        "clip_plans": [],
        "clip_images": [],         # filenames of generated start images
        "output_files": [],
        "error": None,
        "created_at": time.time(),
        "params": params,
        "pause_reason": None,
        "workspace": workspace,
        "out_dir": out_dir,
        # For LLM streaming: the frontend polls /api/v1/llm/stream-status
        "llm_streaming": False,
    }

    with _pipeline_lock:
        _pipelines[pid] = pipeline

    # Non-daemon so pipeline survives browser disconnect during overnight runs.
    _start_pipeline_worker(pid)

    return pid


def get_pipeline(pid: str) -> Optional[dict]:
    with _pipeline_lock:
        p = _pipelines.get(pid)
        return dict(p) if p else None


def get_pipeline_status(pid: str, out_dir: str) -> Optional[dict]:
    """Return live status or a terminal disk snapshot after a UI reconnect.

    Browser tabs can survive a Maestro restart while the in-memory registry
    cannot. Returning the saved terminal/crashed state lets the frontend stop
    polling instead of issuing a 404 every two seconds forever.
    """

    live = get_pipeline(pid)
    if live is not None:
        return live
    saved = load_pipeline_state(out_dir, pid)
    if not saved:
        return None

    saved_status = str(saved.get("status") or "unknown").strip().lower()
    if saved_status not in {"completed", "failed", "cancelled", "crashed"}:
        saved_status = "crashed"
    # Keep the existing live-status API contract for older browser bundles:
    # they already stop polling on "failed" but do not know "crashed".
    response_status = "failed" if saved_status == "crashed" else saved_status
    clips = saved.get("clips") or []
    message = {
        "completed": "Director generation completed",
        "cancelled": "Director generation cancelled",
        "failed": "Director generation failed",
        "crashed": "Director generation was interrupted when Maestro stopped",
    }.get(saved_status, "Saved Director generation")
    return {
        "id": pid,
        "status": response_status,
        "phase": response_status,
        "auto_mode": bool(saved.get("auto_mode", True)),
        "progress": {
            "current": len([
                clip for clip in clips if clip.get("video_filename")
            ]),
            "total": len(clips),
            "message": message,
            "step": 0,
            "total_steps": 0,
        },
        "clip_plans": [{
            "image_prompt": clip.get("image_prompt", ""),
            "video_prompt": clip.get("video_prompt", ""),
            "window_prompts": clip.get("window_prompts", []) or [],
            "keyframe_prompts": clip.get("keyframe_prompts", []) or [],
        } for clip in clips],
        "clip_images": [
            clip.get("start_image_filename") or "" for clip in clips
        ],
        "output_files": saved.get("output_files", []) or [],
        "error": saved.get("error") or (
            "Maestro no longer has a live worker for this Director run."
            if saved_status == "crashed" else None
        ),
        "pause_reason": None,
        "llm_streaming": False,
        "recovered_from_disk": True,
    }


def continue_pipeline(pid: str, updates: Optional[dict] = None):
    """Resume a paused pipeline, optionally with updated clip_plans."""
    with _pipeline_lock:
        p = _pipelines.get(pid)
        if not p or p["status"] != "paused":
            return False
        if updates:
            if "clip_plans" in updates:
                p["clip_plans"] = updates["clip_plans"]
        p["status"] = "running"
        p["pause_reason"] = None
    return True


def _find_pipeline_state_file(pid: str, out_dir: str) -> Optional[str]:
    """Locate a saved pipeline JSON by id under out_dir or a workspace subdir."""
    fname = f"{_PIPELINE_FILE_PREFIX}{pid}.json"
    candidates = [os.path.join(out_dir, fname)]
    try:
        for name in os.listdir(out_dir):
            sub = os.path.join(out_dir, name)
            if os.path.isdir(sub):
                candidates.append(os.path.join(sub, fname))
    except OSError:
        pass
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def resume_pipeline(pid: str, out_dir: str) -> tuple[bool, str]:
    """Rehydrate a crashed pipeline from disk and re-run it.

    Reuses the planning (and start images, when their files still exist)
    that completed before the crash; only the video phase re-runs. Returns
    (ok, message). Requires a state file that carries the full params
    snapshot (written since the resume feature shipped) — older crash files
    can't be resumed faithfully and report so.
    """
    with _pipeline_lock:
        existing = _pipelines.get(pid)
        if (
            pid in _pipeline_threads
            or bool(_pipeline_child_jobs.get(pid))
            or pid in _pipeline_starting
            or pid in _pipeline_operations
            or pid in _pipeline_deleting
            or (
                existing
                and existing.get("status") in (
                    "running", "queued", "planning",
                )
            )
        ):
            return False, "Pipeline is already running."
        _pipeline_starting.add(pid)
    try:
        return _resume_pipeline_reserved(pid, out_dir)
    finally:
        with _pipeline_lock:
            _pipeline_starting.discard(pid)


def _resume_pipeline_reserved(pid: str, out_dir: str) -> tuple[bool, str]:
    """Resume implementation after ``pid`` has been atomically reserved."""
    state_path = _find_pipeline_state_file(pid, out_dir)
    if not state_path:
        return False, "No saved state found for this pipeline."
    try:
        with _pipeline_file_lock:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if _repair_saved_h3_frame_lattice(data):
                _write_pipeline_json_unlocked(state_path, data)
    except Exception as e:
        return False, f"Could not read saved pipeline state: {e}"

    params = data.get("_params_snapshot")
    if not isinstance(params, dict):
        return False, (
            "This pipeline was created before resume support and can't be "
            "resumed — start a new generation."
        )

    params["_director_shot_image_policy"] = (
        _saved_pipeline_shot_image_policy(data)
    )

    try:
        _validate_director_models(params, stages=("video",))
    except DirectorModelCompatibilityError as exc:
        return False, str(exc)

    video_model = params.get("video_model") or data.get("video_model")
    try:
        resume_model_def = _wgp.get_model_def(video_model) or {}
        execution_profile = _saved_director_video_execution_profile(
            data,
            model_def=resume_model_def,
        )
    except ValueError as exc:
        return False, str(exc)
    params["_director_video_execution_profile"] = execution_profile
    params.setdefault("video_params", {})
    if execution_profile.get("normalized_resolution"):
        params["video_params"]["resolution"] = execution_profile[
            "normalized_resolution"
        ]

    # Rebuild the generation-driving structures from the saved per-clip state.
    saved_clips = data.get("clips", []) or []
    try:
        saved_frame_values = []
        for index, saved_clip in enumerate(saved_clips):
            planned = saved_clip.get("planned_clip") or {}
            frames = planned.get("duration_frames")
            saved_frame_values.append(frames)
            if frames is not None:
                validate_director_execution_frames(
                    execution_profile,
                    frames,
                    label=f"Saved Director shot {index + 1}",
                )
        _validate_saved_profile_for_current_hardware(
            data,
            execution_profile,
            resume_model_def,
            saved_frame_values,
        )
    except ValueError as exc:
        return False, str(exc)
    clip_plans = [{
        "image_prompt": c.get("image_prompt", ""),
        "video_prompt": c.get("video_prompt", ""),
        "visual_changes": c.get("visual_changes", []) or [],
        "image_source": c.get("image_source", "original"),
        "keyframe_prompts": c.get("keyframe_prompts", []) or [],
        "window_prompts": c.get("window_prompts", []) or [],
        "window_count": c.get("window_count", 1),
        "_director_dialogue_beats": (
            c.get("_director_dialogue_beats", []) or []
        ),
        "_director_subjects_on_screen": (
            c.get("_director_subjects_on_screen", []) or []
        ),
        "_director_duration_sec": c.get("_director_duration_sec"),
        "_director_vocal_contract": c.get("_director_vocal_contract"),
        "_director_h3_source_prompt": c.get("_director_h3_source_prompt"),
        "_director_h3_compiled_prompt": c.get("_director_h3_compiled_prompt"),
        "_director_h3_prompt_mode": c.get("_director_h3_prompt_mode"),
        "_director_h3_model_family": c.get("_director_h3_model_family"),
        "_director_speaker_registry": c.get("_director_speaker_registry"),
        "_director_project_context": c.get("_director_project_context"),
        "_director_environment": c.get("_director_environment"),
        "_director_opening_blocking": c.get("_director_opening_blocking"),
        "_director_closing_blocking": c.get("_director_closing_blocking"),
        "_director_audio_plan": c.get("_director_audio_plan"),
    } for c in saved_clips]
    planned_clips = [c.get("planned_clip") for c in saved_clips]
    clip_images = [c.get("start_image_filename") for c in saved_clips]
    clip_keyframes = [c.get("keyframe_filenames", []) or [] for c in saved_clips]

    workspace = data.get("workspace") if data.get("workspace") not in ("default", None) else None
    resume_out_dir = os.path.dirname(state_path)

    pipeline = {
        "id": pid,
        "status": "running",
        "phase": "resuming",
        "auto_mode": params.get("auto_mode", True),
        "progress": {"current": 0, "total": 0, "message": "Resuming…", "step": 0, "total_steps": 0},
        "clip_plans": clip_plans,
        "_planned_clips": planned_clips,
        "clip_images": clip_images,
        "_clip_keyframes": clip_keyframes,
        "_clip_video_files": [
            c.get("video_filename") for c in saved_clips
        ],
        "output_files": data.get("output_files", []) or [],
        "_llm_log": data.get("llm_log"),
        "error": None,
        "created_at": data.get("created_at") or time.time(),
        "params": params,
        "pause_reason": None,
        "workspace": workspace,
        "out_dir": resume_out_dir,
        "llm_streaming": False,
    }
    with _pipeline_lock:
        _pipelines[pid] = pipeline

    _start_pipeline_worker(pid, resume=True)
    return True, "resumed"


def _abort_pipeline_jobs(pid: str):
    """Signal wgp abort for this pipeline's queued/running generation jobs.

    Mirrors the Studio cancel endpoint (launch.cancel_job): flip the job's
    gen-state abort flag and the model's _interrupt so the denoise loop
    stops within a step. Without this, Stop only takes effect at the next
    phase/clip boundary — the in-flight clip runs to completion, 10+
    minutes of GPU work after the user pressed Stop on slower cards.
    """
    if not _jobs:
        return
    for job_id, job in list(_jobs.items()):
        params = job.get("params") or {}
        if params.get("_director_pipeline_id") != pid:
            continue
        result = request_cancel(
            job,
            job_id=job_id,
            active_states=_active_gen_states or {},
        )
        if result.abort_signalled:
            print(f"[Pipeline {pid}] Abort signalled for in-flight job {job_id}")


def stop_pipeline(pid: str) -> bool:
    with _pipeline_lock:
        p = _pipelines.get(pid)
        if not p or p.get("status") in ("completed", "failed", "cancelled"):
            return False
        p["status"] = "cancelled"
        p["phase"] = "cancelled"
        p["pause_reason"] = None
        p["_completed_at"] = time.time()
        p["progress"] = {
            "current": 0,
            "total": 0,
            "message": "Cancelled",
            "step": 0,
            "total_steps": 0,
        }
    _abort_pipeline_jobs(pid)
    persisted = _save_pipeline_state(pid)
    with _pipeline_lock:
        current = _pipelines.get(pid)
        if current is not None:
            current["_state_persisted"] = persisted
    return True


def _run_pipeline(pid: str, resume: bool = False):
    """Main pipeline thread — runs the full Director flow.

    When resume=True the pipeline was rehydrated from a crashed state
    (see resume_pipeline): planning + prompt-polish are skipped when the
    saved clip_plans are present, and start-image generation is skipped
    when the saved images still exist on disk. Only the (atomic) video
    generation phase re-runs — so a crash 2 hours into a run doesn't
    throw away the LLM planning that already succeeded.
    """
    try:
        with _pipeline_lock:
            p = _pipelines.get(pid)
            if not p or p.get("status") == "cancelled":
                return
        params = p["params"]
        pipeline_out_dir = p.get("out_dir") or _wgp.save_path
        pipeline_workspace = p.get("workspace")
        shot_image_policy = _director_effective_shot_image_policy(params)
        requires_shot_images = shot_images_required(shot_image_policy)

        # Work already completed before a crash (empty on a fresh run).
        resume_plans = (p.get("clip_plans") or None) if resume else None
        resume_images = (p.get("clip_images") or None) if resume else None

        pipeline_type = params.get("pipeline_type", "music_video")  # music_video | short_film_audio | short_film_story
        auto_mode = params.get("auto_mode", True)

        # ── Disk preflight ─────────────────────────────────────────────
        # A Director run writes gigabytes (per-clip images + video + the
        # final concat). Fail fast with a clear message instead of dying
        # halfway through with a truncated "No space left on device" write.
        try:
            import shutil as _shutil
            free_gb = _shutil.disk_usage(pipeline_out_dir).free / (1024 ** 3)
            if free_gb < 3:
                raise RuntimeError(
                    f"Only {free_gb:.1f} GB free on the output drive — not "
                    f"enough for a Director run. Free up space and try again."
                )
        except RuntimeError:
            raise
        except Exception:
            pass  # disk_usage can fail on odd mounts; don't block on the check itself

        # ── Wait for GPU if jobs are running ────────────────────────────
        # LLM needs GPU (CUDA), so we must wait for generation queue to drain.
        # In auto mode this is expected (fire-and-forget). In non-auto mode
        # the user is waiting interactively, so we still wait but they can cancel.
        if not _wait_for_gpu(pid):
            return  # cancelled while waiting

        # ── Phase 1: LLM Planning ──────────────────────────────────────
        _update_pipeline(pid, phase="planning", llm_streaming=True,
                         progress={"current": 0, "total": 1, "message": "Planning with LLM...", "step": 0, "total_steps": 0})

        planning_start = time.time()
        if resume_plans:
            # Reuse the planning that already succeeded before the crash.
            clip_plans = resume_plans
            planned_clips = p.get("_planned_clips") or []
            print(f"[Pipeline {pid}] Resume: reusing {len(clip_plans)} planned clips — skipping LLM planning + polish")
        else:
            try:
                clip_plans, planned_clips = _run_planning(pid, params, pipeline_type)
            except Exception as plan_err:
                print(f"[Pipeline] Planning error: {plan_err}")
                import traceback
                traceback.print_exc()
                raise
        planning_time = time.time() - planning_start

        if not clip_plans:
            raise RuntimeError("Planning produced no clip plans")

        # H3 does not expose Director's rolling-window contract. Convert the
        # plan before prompt polish and image generation so every downstream
        # artifact (start images, source-audio slices, repair metadata, and
        # generated clips) shares the same native 17n+5 timing lattice.
        if not resume_plans:
            video_model = params.get("video_model") or "ltx2_22B_distilled_1_1"
            try:
                selected_video_def = _wgp.get_model_def(video_model) or {}
            except Exception:
                selected_video_def = {}
            selected_strategy = video_strategy(selected_video_def)
            if selected_strategy in {BOUNDED_START_END, OMNI_REFERENCE}:
                model_fps = float(selected_video_def.get("fps") or 24)
                minimum_frames = int(selected_video_def.get("frames_minimum") or 124)
                maximum_frames = _director_effective_max_frames(
                    params, selected_video_def,
                )
                frame_step = int(selected_video_def.get("frames_steps") or 17)
                original_count = len(clip_plans)
                clip_plans, planned_clips = adapt_bounded_timeline(
                    clip_plans,
                    planned_clips,
                    fps=model_fps,
                    minimum_frames=minimum_frames,
                    maximum_frames=maximum_frames,
                    frame_step=frame_step,
                )
                params["_director_video_strategy"] = selected_strategy
                params["planned_clips"] = planned_clips
                print(
                    f"[Pipeline {pid}] Adapted {original_count} planned scene(s) "
                    f"to {len(clip_plans)} native {video_model} shot(s) "
                    f"({minimum_frames}-{maximum_frames} frames, step {frame_step})."
                )

        # Store planned clips for persistence
        _update_pipeline(pid, _planned_clips=planned_clips)

        # Capture LLM logs — collect all passes from the pipeline's accumulated log
        try:
            from services import llm_service
            # The pipeline accumulates logs via _append_llm_log during planning
            accumulated = _pipelines.get(pid, {}).get("_llm_passes", [])
            # Also capture the final state as a fallback
            if not accumulated:
                accumulated = [{
                    "pass": "planning",
                    "system_prompt": getattr(llm_service, '_last_system_prompt', '') or '',
                    "user_prompt": getattr(llm_service, '_last_user_prompt', '') or '',
                    "response_text": getattr(llm_service, '_stream_buffer', '') or '',
                    "thinking_text": getattr(llm_service, '_last_thinking_text', None),
                }]
            llm_log = {
                "provider": params.get("llm_provider", "local"),
                "model_id": params.get("llm_model_id", ""),
                "passes": accumulated,
                # Keep flat fields for backward compat — use last pass
                "system_prompt": accumulated[-1].get("system_prompt", "") if accumulated else "",
                "response_text": accumulated[-1].get("response_text", "") if accumulated else "",
                "thinking_text": accumulated[-1].get("thinking_text") if accumulated else None,
                "planning_time_sec": round(planning_time, 2),
            }
            # On resume, keep the rehydrated original log instead of clobbering
            # it with an empty re-capture (there was no fresh planning stream).
            if not resume_plans:
                _update_pipeline(pid, _llm_log=llm_log)
        except Exception:
            pass

        # ── Optional: Third-pass prompt polish ────────────────────────
        services = _wgp.server_config.get("services", {}) if _wgp else {}
        # Default "third_pass" is model-aware. Architectures that benefit
        # from a dialect rewrite keep it; native H3 video prompts bypass the
        # creative rewrite and proceed to deterministic continuity/dialogue
        # preflight. H3-generated image prompts can still be polished for the
        # selected image model.
        polish_mode = services.get("director_prompt_polish", "third_pass")

        # Snapshot pre-polish prompts for comparison
        import copy
        _update_pipeline(pid, _clip_plans_pre_polish=copy.deepcopy(clip_plans))

        # On resume the saved clip_plans are ALREADY polished — re-polishing
        # would compound edits and drift the prompts, so skip the whole block.
        if resume_plans:
            pass
        elif polish_mode == "third_pass" and clip_plans:
            try:
                from services.director.prompt_polish import (
                    polish_prompts_third_pass,
                    should_polish_director_video_prompts,
                )
                provider = services.get("llm_provider", "local")
                nsfw = services.get("nsfw_mode", False) and provider not in {"openai", "anthropic"}
                video_model = params.get("video_model", "")
                image_model = params.get("image_model", "")
                polish_video_prompts = should_polish_director_video_prompts(
                    video_model
                )
                polish_image_prompts = bool(requires_shot_images)
                if polish_video_prompts or polish_image_prompts:
                    polish_label = (
                        "Polishing generated image prompts..."
                        if not polish_video_prompts
                        else "Polishing prompts (3rd pass)..."
                    )
                    _update_pipeline(
                        pid,
                        phase="polishing_prompts",
                        llm_streaming=False,
                        progress={
                            "current": 0,
                            "total": len(clip_plans),
                            "message": polish_label,
                            "step": 0,
                            "total_steps": 0,
                        },
                    )
                else:
                    _update_pipeline(
                        pid,
                        _polish_mode_used="h3_native_preflight",
                    )
                    print(
                        f"[Pipeline {pid}] Skipping creative third-pass "
                        "polish for native MiniMax H3 video prompts; "
                        "deterministic continuity and dialogue preflight "
                        "remain enabled."
                    )
                video_loras = (params.get("video_loras") or {}).get("activated_loras", [])
                image_loras = (params.get("image_loras") or {}).get("activated_loras", [])
                ref_paths = []
                rip = params.get("reference_image_path")
                if rip:
                    ref_paths.append(rip)
                for cp in (params.get("character_ref_paths") or []):
                    if cp:
                        ref_paths.append(cp)
                # Pass character profiles into polish so the LLM has a
                # definitive name → descriptor mapping. Without this, polish
                # silently substitutes generic "the woman" / "the man" for
                # any character name it encounters — catastrophic for
                # non-human characters (Lumi the unicorn became "the woman"
                # in test 03). characters comes from params.characters,
                # the same list passed to the planner.
                characters = params.get("characters", []) or []
                if polish_video_prompts or polish_image_prompts:
                    clip_plans = polish_prompts_third_pass(
                        clip_plans, video_model, image_model, nsfw,
                        video_loras=video_loras, image_loras=image_loras,
                        image_paths=ref_paths or None,
                        characters=characters,
                        preserve_video_character_names=(
                            str(video_model).lower().startswith("minimax_h3")
                            and shot_image_policy in {
                                SHOT_IMAGE_PROMPT_ONLY,
                                SHOT_IMAGES_DIRECT_REFERENCES,
                            }
                        ),
                        polish_video_prompts=polish_video_prompts,
                        polish_image_prompts=polish_image_prompts,
                    )
                    _capture_llm_pass(pid, "third_pass_polish")
                    print(
                        "[Pipeline] Model-aware third-pass polish completed "
                        f"for {len(clip_plans)} clips"
                    )
            except Exception as e:
                print(f"[Pipeline] Prompt polish failed (non-fatal): {e}")
        elif polish_mode in ("full_guide", "light_guide"):
            # For inject modes, polish happened inside the planner — note it in the log
            _update_pipeline(pid, _polish_mode_used=polish_mode)

        # Bounded shots have no semantic memory of the preceding generation.
        # Re-attach the stored world/location anchor after any LLM polish so a
        # rewrite cannot reduce a recognizable set to a generic room.
        clip_plans = apply_independent_shot_context(clip_plans)
        _preflight_h3_director_prompts(
            params.get("video_model", ""),
            clip_plans,
            pid=pid,
        )

        _update_pipeline(pid, clip_plans=clip_plans, llm_streaming=False)
        _save_pipeline_state(pid)  # Save after planning

        # Check cancellation
        if _pipelines[pid]["status"] == "cancelled":
            return

        # In non-auto mode, pause for user review after planning
        if not auto_mode:
            _update_pipeline(pid, status="paused", pause_reason="review_prompts",
                             progress={"current": 1, "total": 3, "message": "Review prompts", "step": 0, "total_steps": 0})
            _save_pipeline_state(pid)  # Save paused state so Dashboard shows it
            _wait_for_resume(pid)
            if _pipelines[pid]["status"] == "cancelled":
                return
            # Reload clip_plans in case user edited them
            clip_plans = _pipelines[pid]["clip_plans"]

        # ── Phase 2: Generate Start Images ──────────────────────────────
        # Generate start images only when the selected model/policy uses them.
        # When no reference image was provided in that workflow,
        # _run_image_generation creates an establishing/anchor image first and
        # adopts it as the shared reference, so every clip shares a look —
        # instead of skipping image gen and going straight to text-to-video.
        if requires_shot_images:
            _update_pipeline(pid, phase="generating_images",
                             progress={"current": 0, "total": len(clip_plans), "message": "Generating start images...", "step": 0, "total_steps": 0})
        else:
            guidance_label = (
                "direct references"
                if shot_image_policy == SHOT_IMAGES_DIRECT_REFERENCES
                else "video prompts"
            )
            _update_pipeline(pid, phase="preparing_video",
                             progress={"current": 0, "total": len(clip_plans), "message": f"Using {guidance_label}; no shot images needed", "step": 0, "total_steps": 0})

        # ── Detect the reference's art style while the LLM is still up ──
        # One vision call naming the medium concretely; the phrase gets
        # prepended to every image prompt in _run_image_generation (see
        # the module-level "Reference art-style lock" note). Skipped when
        # already detected (resume) or the reference is photographic.
        from services import llm_service
        _style_ref = params.get("reference_image_path") or ""
        if (
            requires_shot_images
            and "_reference_style" not in params
            and _style_ref
            and os.path.isfile(_style_ref)
        ):
            _style_phrase = ""
            try:
                if llm_service.is_loaded() and getattr(llm_service, "_vision_available", False):
                    _style_raw = llm_service.generate(
                        _STYLE_DESCRIBE_PROMPT,
                        max_new_tokens=48,
                        temperature=0.1,
                        image_paths=[_style_ref],
                        enable_thinking=False,
                    )
                    _style_phrase = _normalize_style_phrase(_style_raw)
                    print(f"[Pipeline {pid}] Reference art style: {_style_phrase!r} (raw: {str(_style_raw)[:80]!r})")
            except Exception as e:
                print(f"[Pipeline {pid}] Style detection skipped (non-fatal): {e}")
            # Record even when empty ("" = photographic / undetected) so
            # resume doesn't re-run the detection.
            params["_reference_style"] = _style_phrase
            _update_pipeline(pid, _reference_style=_style_phrase)

        # Unload LLM to free VRAM
        try:
            if llm_service.is_loaded():
                llm_service.unload_model()
        except Exception as e:
            print(f"[Pipeline] LLM unload warning (non-fatal): {e}")

        # On resume, reuse the start images that already generated before the
        # crash — but only if every file still exists (a wiped/half-written
        # output dir falls back to regenerating them, which is safer than
        # feeding missing paths into video generation).
        _resume_imgs_ok = requires_shot_images and bool(resume_images) and all(
            f and os.path.isfile(os.path.join(pipeline_out_dir, f)) for f in resume_images
        )
        if not requires_shot_images:
            clip_images = [""] * len(clip_plans)
            clip_keyframes = [[] for _ in clip_plans]
            print(
                f"[Pipeline {pid}] Shot images skipped by saved policy "
                f"'{shot_image_policy}'."
            )
        elif _resume_imgs_ok:
            clip_images = resume_images
            clip_keyframes = p.get("_clip_keyframes") or [[] for _ in clip_images]
            print(f"[Pipeline {pid}] Resume: reusing {len(clip_images)} start images — skipping image generation")
        else:
            if resume_images:
                print(f"[Pipeline {pid}] Resume: saved start images missing on disk — regenerating")
            clip_images, clip_keyframes = _run_image_generation(pid, params, clip_plans, out_dir=pipeline_out_dir, workspace=pipeline_workspace)

        _update_pipeline(pid, clip_images=clip_images, _clip_keyframes=clip_keyframes)
        _save_pipeline_state(pid)  # Save after image generation

        if _pipelines[pid]["status"] == "cancelled":
            return

        if requires_shot_images:
            _require_video_start_images(
                clip_images, len(clip_plans), pipeline_out_dir,
            )

        # In non-auto mode, pause for image review
        if not auto_mode and requires_shot_images:
            _update_pipeline(pid, status="paused", pause_reason="review_images",
                             progress={"current": 2, "total": 3, "message": "Review images", "step": 0, "total_steps": 0})
            _wait_for_resume(pid)
            if _pipelines[pid]["status"] == "cancelled":
                return

            # Review can be open for hours; a gallery cleanup or manual rename
            # during that pause must not silently turn a planned I2V shot into
            # unconditioned T2V.
            _require_video_start_images(
                clip_images, len(clip_plans), pipeline_out_dir,
            )

        # ── Phase 3: Generate Video ─────────────────────────────────────
        _update_pipeline(pid, phase="generating_video",
                         progress={"current": 0, "total": 1, "message": "Generating video...", "step": 0, "total_steps": 0})

        output_files = _run_video_generation(pid, params, clip_plans, planned_clips, clip_images, clip_keyframes, out_dir=pipeline_out_dir, workspace=pipeline_workspace)

        # A Stop during the video phase lands here after the abort. Record
        # whatever clips finished (the Dashboard can rerun/rejoin them),
        # but don't overwrite the cancelled status with "completed".
        if _pipelines[pid]["status"] == "cancelled":
            print(f"[Pipeline {pid}] Cancelled during video generation — keeping {len(output_files or [])} finished clip(s)")
            artifacts = {"output_files": output_files or []}
            if not params.get("seamless", True):
                clip_videos = _clip_video_slots(
                    output_files or [], len(clip_plans),
                )
                if clip_videos:
                    artifacts["_clip_video_files"] = clip_videos
            _update_pipeline(pid, **artifacts)
            _save_pipeline_state(pid)
            return

        completed_clip_videos = []
        if not params.get("seamless", True):
            completed_clip_videos = _clip_video_slots(
                output_files or [], len(clip_plans),
            )
        completed = _update_pipeline(
            pid,
            status="completed",
            phase="completed",
            output_files=output_files,
            _clip_video_files=completed_clip_videos,
            _completed_at=time.time(),
            progress={
                "current": 3, "total": 3, "message": "Done!",
                "step": 0, "total_steps": 0,
            },
        )
        if not completed:
            _update_pipeline(
                pid,
                output_files=output_files or [],
                _clip_video_files=completed_clip_videos,
            )
        _save_pipeline_state(pid)  # Save on completion

    except Exception as e:
        import traceback
        partial_outputs = getattr(e, "output_files", None)
        if partial_outputs:
            artifact_updates = {"output_files": partial_outputs}
            with _pipeline_lock:
                current_pipeline = _pipelines.get(pid) or {}
                current_plans = current_pipeline.get("clip_plans") or []
                current_params = current_pipeline.get("params") or {}
            if not current_params.get("seamless", True):
                clip_slots = _clip_video_slots(
                    partial_outputs, len(current_plans),
                )
                if clip_slots:
                    artifact_updates["_clip_video_files"] = clip_slots
            _update_pipeline(pid, **artifact_updates)
        # Special-case the safety scanner. Don't print a stack trace for
        # safety violations — they're a clean refusal, not a crash, and
        # the user-visible message is purpose-built. Other exceptions
        # keep the existing traceback dump for debugging.
        try:
            from services.director.safety_scan import SafetyViolationError
        except Exception:
            SafetyViolationError = None  # type: ignore
        if SafetyViolationError is not None and isinstance(e, SafetyViolationError):
            print(
                f"[Pipeline {pid}] Safety scan blocked generation. "
                f"source={e.source} matched={e.matched_terms}"
            )
            user_msg = (
                "Generation aborted: the input contained content involving "
                f"minors in a prohibited context (matched terms: "
                f"{', '.join(e.matched_terms)}). The system refuses to "
                f"generate this category of content. Please revise your "
                f"concept to use only adult characters (18+)."
            )
            _update_pipeline(
                pid, status="failed", error=user_msg,
                _completed_at=time.time(),
                progress={"current": 0, "total": 0,
                          "message": "Generation aborted (safety policy)",
                          "step": 0, "total_steps": 0},
            )
            _save_pipeline_state(pid)
            return
        traceback.print_exc()
        # Tag with OOM info if applicable so the UI can surface the
        # OOM recovery banner. detect_oom returns None for non-OOM
        # failures, in which case oom_info stays absent.
        _oom_info = None
        try:
            from services.oom_detect import detect_oom
            import wgp as _wgp_mod
            _coef = float(_wgp_mod.server_config.get("vram_safety_coefficient", 0.80))
            _oom_info = detect_oom(e, _coef)
        except Exception:
            pass  # Never fail a failure handler
        _update_pipeline(pid, status="failed", error=str(e),
                         oom_info=_oom_info,
                         _completed_at=time.time(),
                         progress={"current": 0, "total": 0, "message": f"Error: {e}", "step": 0, "total_steps": 0})
        _save_pipeline_state(pid)  # Save on failure too
    finally:
        with _pipeline_lock:
            current = _pipeline_threads.get(pid)
            if current is threading.current_thread():
                _pipeline_threads.pop(pid, None)


def _wait_for_resume(pid: str, poll_interval: float = 1.0):
    """Block until pipeline is resumed, cancelled, or removed."""
    while True:
        with _pipeline_lock:
            p = _pipelines.get(pid)
            if not p:
                return
            if p["status"] != "paused":
                return
        time.sleep(poll_interval)


def _wait_for_gpu(pid: str, poll_interval: float = 2.0):
    """Block until no generation jobs are actively running on GPU.

    Checks both _gen_lock availability and active job statuses.
    Returns False if pipeline was cancelled while waiting.
    """
    _update_pipeline(pid, progress={
        "current": 0, "total": 1,
        "message": "Waiting for GPU (generation queue)...",
        "step": 0, "total_steps": 0,
    })

    while True:
        if _pipelines.get(pid, {}).get("status") == "cancelled":
            return False

        # Check if any jobs are currently running
        active_jobs = [j for j in _jobs.values()
                       if j.get("status") in ("queued", "running")]
        if not active_jobs:
            return True

        time.sleep(poll_interval)


# ── Planning Phase ──────────────────────────────────────────────────────

def _ensure_llm_loaded(params: dict):
    """Load/reload LLM if needed. Shared between legacy and new planning."""
    from services import llm_service

    services_cfg = _wgp.server_config.get("services", {}) if _wgp else {}
    desired_model = params.get("llm_model_id") or services_cfg.get("llm_model_id", "Abhiray/gemma-4-E4B-it-heretic-GGUF")
    desired_device = params.get("llm_device") or services_cfg.get("llm_device", "cpu")
    desired_provider = params.get("llm_provider") or services_cfg.get("llm_provider", "local")
    desired_remote_url = services_cfg.get("llm_remote_url", "")
    desired_api_key = ""
    if desired_provider == "openai":
        desired_api_key = services_cfg.get("openai_api_key", "")
    elif desired_provider == "anthropic":
        desired_api_key = services_cfg.get("anthropic_api_key", "")

    # Free GPU memory before running a local CUDA LLM. Director planning
    # fires right after image edits / audio analysis: memory profiles keep
    # the last generation model resident, and torch's caching allocator
    # holds whatever Whisper / the vocal separator reserved — none of it
    # available to the llama-server SUBPROCESS. The server then loads its
    # weights fine but aborts (CUDA OOM → connection reset by peer) when
    # the vision encode spikes during the first planning request; the
    # identical request verified fine on a free GPU. Guarded by _gen_lock
    # so an active generation is never released mid-run; wgp reloads the
    # gen model transparently on its next job (reload_needed).
    if desired_provider == "local" and desired_device == "cuda" and _wgp is not None:
        acquired = _gen_lock.acquire(blocking=False) if _gen_lock is not None else True
        if acquired:
            try:
                if getattr(_wgp, "wan_model", None) is not None:
                    print("[Pipeline] Releasing generation model VRAM before LLM planning")
                    _wgp.release_model()
                else:
                    import gc
                    import torch
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"[Pipeline] Pre-LLM VRAM release skipped: {e}")
            finally:
                if _gen_lock is not None:
                    _gen_lock.release()
        else:
            print("[Pipeline] Generation in progress — skipping pre-LLM VRAM release")

    if llm_service.is_loaded():
        status = llm_service.get_status()
        if status.get("model_id") != desired_model or status.get("provider") != desired_provider:
            llm_service.unload_model()
            llm_service.load_model(model_id=desired_model, device=desired_device, provider=desired_provider, remote_url=desired_remote_url, api_key=desired_api_key)
    else:
        llm_service.load_model(model_id=desired_model, device=desired_device, provider=desired_provider, remote_url=desired_remote_url, api_key=desired_api_key)


def _capture_llm_pass(pid: str, pass_name: str):
    """Capture the current LLM state as a pass and append to the pipeline's log.

    Captures both system_prompt AND user_prompt so the Director Dashboard
    can render the full LLM input. Previously the dashboard only stored
    system_prompt, which made it look like the user's story description
    was missing from Pass 1's input — but it was always being sent as
    a separate user message; the dashboard just wasn't capturing it.
    """
    try:
        from services import llm_service
        pass_entry = {
            "pass": pass_name,
            "system_prompt": getattr(llm_service, '_last_system_prompt', '') or '',
            "user_prompt": getattr(llm_service, '_last_user_prompt', '') or '',
            "response_text": getattr(llm_service, '_stream_buffer', '') or '',
            "thinking_text": getattr(llm_service, '_last_thinking_text', None),
        }
        with _pipeline_lock:
            p = _pipelines.get(pid)
            if p:
                passes = p.get("_llm_passes", [])
                passes.append(pass_entry)
                p["_llm_passes"] = passes
    except Exception:
        pass


def _run_planning(pid: str, params: dict, pipeline_type: str):
    """Run LLM planning and return (clip_plans, planned_clips).

    Uses the new DirectorOrchestrator when use_director_v2 flag is set,
    otherwise falls back to legacy llm_service calls.
    """
    _ensure_llm_loaded(params)

    # Default v2 — see launch.py services-config comment for rationale.
    # The params dict is built from servicesConfig in the frontend, so
    # this default only fires for direct API callers that didn't pass
    # the flag at all. Keeping it consistent with the services-config
    # default here so the legacy path isn't accidentally hit.
    use_v2 = params.get("use_director_v2", True)
    execution_profile = _director_video_execution_profile(params)
    if execution_profile.get("is_minimax_h3") and not use_v2:
        # The legacy planner only understands generic 20-second rolling
        # windows. H3 needs the native-shot planner so its dialogue and action
        # are written against the effective one-pass limit.
        print(
            f"[Pipeline {pid}] MiniMax H3 requires Director v2 native-shot "
            "planning; ignoring the legacy Director toggle for this run."
        )
        use_v2 = True

    if use_v2:
        return _run_planning_v2(pid, params, pipeline_type)
    else:
        return _run_planning_legacy(pid, params, pipeline_type)


def _run_planning_v2(pid: str, params: dict, pipeline_type: str):
    """New architecture: DirectorOrchestrator with planners + renderers."""
    from services import llm_service
    from services.director.orchestrator import DirectorOrchestrator, DirectorFlags

    # Build feature flags from params
    flags_dict = params.get("director_flags", {})
    flags = DirectorFlags.from_dict(flags_dict) if flags_dict else DirectorFlags()

    # Wrap LLM functions to capture each pass for the dashboard log
    _pass_counter = [0]
    def _logged_generate(*args, **kwargs):
        result = llm_service.generate(*args, **kwargs)
        _pass_counter[0] += 1
        _capture_llm_pass(pid, f"generate_{_pass_counter[0]}")
        return result

    def _logged_streaming(*args, **kwargs):
        result = llm_service.generate_streaming(*args, **kwargs)
        _pass_counter[0] += 1
        _capture_llm_pass(pid, f"streaming_{_pass_counter[0]}")
        return result

    # Create orchestrator with logged LLM functions
    director = DirectorOrchestrator(
        llm_generate=_logged_generate,
        llm_generate_streaming=_logged_streaming,
        flags=flags,
    )

    # Map pipeline_type to skill_type
    skill_map = {
        "music_video": "music_video",
        "short_film_audio": "short_film",
        "short_film_story": "short_film",
        "podcast": "podcast",
        "viral_video": "viral_video",
    }
    skill_type = skill_map.get(pipeline_type, "music_video")

    # Build planner kwargs
    scene_description = params.get("scene_description", "")
    reference_image_path = params.get("reference_image_path")
    planned_clips = params.get("planned_clips", [])

    # Read NSFW from server config (persisted setting, not per-request)
    services_cfg = _wgp.server_config.get("services", {}) if _wgp else {}
    nsfw = services_cfg.get("nsfw_mode", False)
    # Multi-shot LoRA mode — passes through to Pass 2 so it can emit
    # storyboard-format video_prompts for medium-length shots. See
    # the toggle's comment in launch.py for behavior details.
    multishot_lora_mode = services_cfg.get("director_multishot_lora_mode", False)

    seamless = params.get("seamless", True)
    selected_video_model = params.get("video_model", "")
    try:
        selected_video_def = (
            _wgp.get_model_def(selected_video_model)
            if _wgp and selected_video_model
            else None
        ) or {}
    except Exception:
        selected_video_def = {}
    selected_video_strategy = video_strategy(selected_video_def)
    effective_max_frames = _director_effective_max_frames(
        params, selected_video_def,
    )

    # Audio-analysis workflows arrive with a coarse clip timeline before the
    # LLM writes prompts. For bounded H3, divide that timeline now so the LLM
    # receives the exact number and duration of native shots. Splitting after
    # prompt generation forced one long action/dialogue description across
    # multiple hardware windows and made later windows repeat or improvise.
    if (
        pipeline_type != "short_film_story"
        and selected_video_strategy in {BOUNDED_START_END, OMNI_REFERENCE}
        and planned_clips
    ):
        placeholder_plans = [
            {"video_prompt": "", "image_prompt": ""}
            for _ in planned_clips
        ]
        original_planning_count = len(planned_clips)
        _, planned_clips = adapt_bounded_timeline(
            placeholder_plans,
            planned_clips,
            fps=float(selected_video_def.get("fps") or 24),
            minimum_frames=int(
                selected_video_def.get("frames_minimum") or 124
            ),
            maximum_frames=effective_max_frames,
            frame_step=int(selected_video_def.get("frames_steps") or 17),
        )
        params["planned_clips"] = planned_clips
        print(
            f"[Pipeline {pid}] Pre-segmented {original_planning_count} "
            f"audio timeline item(s) into {len(planned_clips)} "
            f"hardware-safe native shot(s) before prompt planning "
            f"(max {effective_max_frames} frames)."
        )
    # Pass video_model and image_model to every planner so Pass 2 can
    # route its prompt guides correctly. Previously these only flowed
    # into polish_block construction (when polish_mode was on); now the
    # planner gets them unconditionally so it can pick the right
    # dialect-aware guide files (ltx2_shot_breakdown.md for LTX-2,
    # flux_image_edit_pass2.md for Flux.2 Klein, etc.).
    planner_kwargs = {
        "reference_image_path": reference_image_path,
        "speaker_mappings": params.get("speaker_mappings"),
        "characters": params.get("characters", []),
        "nsfw": nsfw,
        "seamless": seamless,
        "video_model": params.get("video_model", ""),
        "image_model": params.get("image_model", ""),
        "shot_image_policy": _director_effective_shot_image_policy(params),
        "multishot_lora_mode": multishot_lora_mode,
    }

    if pipeline_type == "short_film_story":
        native_bounded = selected_video_strategy in {
            BOUNDED_START_END,
            OMNI_REFERENCE,
        }
        planner_kwargs.update({
            "story_description": scene_description,
            "target_duration": params.get("target_duration", 60),
            "target_scenes": params.get("target_scenes"),
            "narrative_mode": params.get("narrative_mode", False),
            "fps": (
                selected_video_def.get("fps", 24)
                if native_bounded else params.get("fps", 16)
            ),
            "frames_steps": (
                selected_video_def.get("frames_steps", 17)
                if native_bounded else params.get("frames_steps", 8)
            ),
            "frames_minimum": (
                selected_video_def.get("frames_minimum", 124)
                if native_bounded else params.get("frames_minimum", 41)
            ),
            "frames_maximum": (
                effective_max_frames
                if native_bounded else None
            ),
        })
    elif pipeline_type == "short_film_audio":
        planner_kwargs.update({
            "clips": planned_clips,
            "story_description": scene_description,
            "audio_path": params.get("audio_path"),
            "lyrics": params.get("lyrics"),
        })
    elif pipeline_type in ("podcast", "viral_video"):
        planner_kwargs.update({
            "clips": planned_clips if planned_clips else None,
            "transcript": params.get("lyrics"),
            "audio_path": params.get("audio_path"),
            "concept": scene_description,
            "visual_style": params.get("visual_style", ""),
            "target_duration": params.get("target_duration", 30),
            "platform": params.get("platform", "general"),
            "style": params.get("style", "cinematic"),
        })
    else:
        # Music video
        planner_kwargs.update({
            "clips": planned_clips,
            "scene_description": scene_description,
            "lyrics": params.get("lyrics"),
            "bpm": params.get("bpm", 120),
        })

    # Inject LoRA guides + model dialect guides into the planner only for
    # the full/light_guide inject modes (legacy paths). Default mode
    # "third_pass" deliberately skips this. Architectures that need a
    # separate dialect rewrite receive it after planning; native H3 keeps its
    # dedicated Pass 2 output and proceeds directly to deterministic preflight.
    polish_mode = services_cfg.get("director_prompt_polish", "third_pass")
    if polish_mode in ("full_guide", "light_guide"):
        from services.director.prompt_polish import build_polish_block
        guide_mode = "full" if polish_mode == "full_guide" else "light"
        video_model = params.get("video_model", "")
        image_model = params.get("image_model", "")
        video_loras = (params.get("video_loras") or {}).get("activated_loras", [])
        image_loras = (params.get("image_loras") or {}).get("activated_loras", [])
        polish_block = build_polish_block(video_model, image_model, guide_mode,
                                          video_loras=video_loras, image_loras=image_loras)
        if polish_block:
            planner_kwargs["polish_block"] = polish_block
            print(f"[Pipeline {pid}] Injected {guide_mode} polish block ({len(polish_block)} chars)")

    # Also pass character/location ref labels and paths for image prompt rules
    planner_kwargs["character_ref_paths"] = params.get("character_ref_paths", [])
    planner_kwargs["character_ref_labels"] = params.get("character_ref_labels", [])
    planner_kwargs["location_ref_paths"] = params.get("location_ref_paths", [])
    planner_kwargs["location_ref_labels"] = params.get("location_ref_labels", [])

    # Plan
    print(f"[Pipeline {pid}] Planning with DirectorOrchestrator (skill={skill_type})...")
    plan = director.plan(skill_type, **planner_kwargs)

    # Store the production plan in pipeline state for later reference
    _update_pipeline(pid, production_plan=plan.to_dict())

    # Render only the prompt families this saved workflow will consume.  In
    # prompt-only/direct-reference H3 projects the planner already writes a
    # complete video prompt, and asking the image renderer to synthesize an
    # unused still prompt both wastes work and leaks misleading image cards
    # into Director chat.
    has_reference = bool(reference_image_path)
    render_prompt_type = (
        "both"
        if shot_images_required(_director_effective_shot_image_policy(params))
        else "video"
    )
    rendered = director.render_plan(
        plan,
        prompt_type=render_prompt_type,
        has_reference=has_reference,
    )
    clip_plans = director.plan_to_clip_plans(rendered)

    # Build planned_clips from shot data (for story mode which creates clips)
    if pipeline_type == "short_film_story":
        cumulative = 0.0
        # Get FPS from model definition for accurate frame count
        fps = params.get("fps", 16)
        try:
            vm = params.get("video_model", "")
            md = _wgp.get_model_def(vm) if vm else None
            if md and md.get("fps"):
                fps = md["fps"]
        except Exception:
            pass
        new_clips = []
        for shot in plan.shots:
            duration_frames = shot.metadata.get("duration_frames") if shot.metadata else int(shot.duration_sec * fps)
            new_clips.append({
                "start": cumulative,
                "end": cumulative + shot.duration_sec,
                "duration_sec": shot.duration_sec,
                "duration_frames": duration_frames,
                "label": shot.narrative_role or shot.scene_type or "scene",
                "beat_count": 0,
            })
            cumulative += shot.duration_sec
        planned_clips = new_clips

    # Normalize
    if clip_plans and isinstance(clip_plans[0], str):
        clip_plans = [{"video_prompt": p, "image_prompt": ""} for p in clip_plans]

    # Preserve the planner's shot-state contract on both parallel structures.
    # Prompt-only H3 needs this after third-pass polish, and FL2VA uses the
    # explicit extend_previous marker to decide whether a true final frame may
    # become the next shot's start frame.  Ordinary same-scene cuts remain
    # independent even when they share wardrobe and blocking continuity.
    video_model_lower = str(params.get("video_model") or "").lower()
    is_h3_model = video_model_lower.startswith("minimax_h3")
    h3_model_family = (
        "ref2va" if video_model_lower.startswith("minimax_h3_ref2va") else "base"
    )
    h3_initial_prompt_mode = (
        "ref2va"
        if h3_model_family == "ref2va"
        else "i2va"
        if shot_images_required(_director_effective_shot_image_policy(params))
        else "t2va"
    )
    for index, shot in enumerate(plan.shots):
        metadata = getattr(shot, "metadata", None) or {}
        continuity_group = str(metadata.get("continuity_group") or "").strip()
        closing_blocking = str(
            metadata.get("closing_blocking")
            or getattr(shot, "ending_beat", "")
            or ""
        ).strip()
        shot_state = {
            "_director_continuity_strategy": getattr(
                shot, "continuity_strategy", "independent"
            ),
            "_director_continuity_group": continuity_group,
            "_director_opening_blocking": getattr(
                shot, "spatial_setup", ""
            ),
            "_director_closing_blocking": closing_blocking,
            # Retain the structured H3 speech contract through prompt polish,
            # persistence, resume, and Dashboard reruns. The final preflight
            # compiler uses these fields as the authoritative dialogue source.
            "_director_dialogue_beats": [
                (
                    beat.to_dict()
                    if callable(getattr(beat, "to_dict", None))
                    else dict(beat)
                    if isinstance(beat, dict)
                    else dict(vars(beat))
                )
                for beat in (getattr(shot, "dialogue_beats", None) or [])
            ],
            "_director_subjects_on_screen": [
                (
                    subject.to_dict()
                    if callable(getattr(subject, "to_dict", None))
                    else dict(subject)
                    if isinstance(subject, dict)
                    else dict(vars(subject))
                )
                for subject in (
                    getattr(shot, "subjects_on_screen", None) or []
                )
            ],
            "_director_duration_sec": getattr(shot, "duration_sec", None),
        }
        if is_h3_model:
            audio_plan = getattr(shot, "audio_plan", None)
            shot_state.update({
                # Keep the planner's audiovisual description immutable. The
                # final compiler may run once for review and again after the
                # actual start/end/reference assets are known.
                "_director_h3_source_prompt": (
                    clip_plans[index].get("video_prompt", "")
                    if index < len(clip_plans) else ""
                ),
                "_director_h3_prompt_mode": h3_initial_prompt_mode,
                "_director_h3_model_family": h3_model_family,
                "_director_project_context": scene_description,
                "_director_environment": getattr(shot, "environment", ""),
                "_director_audio_plan": (
                    audio_plan.to_dict()
                    if callable(getattr(audio_plan, "to_dict", None))
                    else dict(audio_plan)
                    if isinstance(audio_plan, dict)
                    else {}
                ),
            })
        if index < len(clip_plans):
            clip_plans[index].update(shot_state)
        if index < len(planned_clips):
            planned_clips[index].update(shot_state)

    if selected_video_strategy in {BOUNDED_START_END, OMNI_REFERENCE}:
        clip_plans = apply_independent_shot_context(
            clip_plans,
            scene_description=scene_description,
            shots=plan.shots,
        )

    # Debug: log shot structure
    for idx, cp in enumerate(clip_plans):
        kf_count = len(cp.get("keyframe_prompts", []) or [])
        wc = cp.get("window_count", 1)
        pc = planned_clips[idx] if idx < len(planned_clips) else {}
        dur = pc.get("duration_sec", pc.get("duration_frames", "?"))
        print(f"[Pipeline] Shot {idx+1}: duration={dur}s, windows={wc}, keyframes={kf_count}, prompt_len={len(cp.get('video_prompt',''))}")

    return clip_plans, planned_clips


def _run_planning_legacy(pid: str, params: dict, pipeline_type: str):
    """Legacy planning: direct calls to llm_service functions."""
    from services import llm_service

    scene_description = params.get("scene_description", "")
    reference_image_path = params.get("reference_image_path")
    speaker_mappings = params.get("speaker_mappings", [])
    characters = params.get("characters", [])
    audio_path = params.get("audio_path")
    planned_clips = params.get("planned_clips", [])
    fps = params.get("fps", 16)
    frames_steps = params.get("frames_steps", 8)
    frames_minimum = params.get("frames_minimum", 41)

    if pipeline_type == "short_film_story":
        # Path C: Full story-based planning
        target_duration = params.get("target_duration", 60)
        narrative_mode = params.get("narrative_mode", False)

        result = llm_service.plan_short_film_from_story(
            story_description=scene_description,
            characters=characters,
            reference_image_path=reference_image_path,
            target_duration=target_duration,
            narrative_mode=narrative_mode,
            fps=fps,
            frames_steps=frames_steps,
            frames_minimum=frames_minimum,
        )
        planned_clips = result.get("clips", [])
        clip_plans = result.get("clip_plans", [])

    elif pipeline_type == "short_film_audio":
        # Path B: Short film with uploaded dialogue audio
        result = llm_service.plan_short_film_prompts(
            clips=planned_clips,
            scene_description=scene_description,
            lyrics=params.get("lyrics", ""),
            reference_image_path=reference_image_path,
            speaker_mappings=speaker_mappings,
            characters=characters,
            prompt_type="both",
        )
        clip_plans = result if isinstance(result, list) else result.get("clip_plans", [])

    else:
        # Music video flow
        result = llm_service.plan_clip_prompts_and_images(
            clips=planned_clips,
            scene_description=scene_description,
            lyrics=params.get("lyrics", ""),
            bpm=params.get("bpm"),
            reference_image_path=reference_image_path,
            speaker_mappings=speaker_mappings,
            prompt_type="both",
        )
        clip_plans = result if isinstance(result, list) else result.get("clip_plans", [])

    # Normalize clip_plans to list of dicts
    if clip_plans and isinstance(clip_plans[0], str):
        clip_plans = [{"video_prompt": p, "image_prompt": ""} for p in clip_plans]

    return clip_plans, planned_clips


# ── Image Generation Phase ──────────────────────────────────────────────

def _run_image_generation(pid: str, params: dict, clip_plans: list[dict], out_dir: str = None, workspace: str = None) -> tuple[list[str], list[list[str]]]:
    """Generate start images and keyframe images per clip.

    Returns:
        (clip_images, clip_keyframes) where:
        - clip_images[i] = start image filename for clip i
        - clip_keyframes[i] = list of keyframe image filenames for clip i (may be empty)
    """
    _validate_director_models(params, stages=("image",))
    ref_image_path = params.get("reference_image_path")
    character_ref_paths = params.get("character_ref_paths", []) or []
    location_ref_paths = params.get("location_ref_paths", []) or []
    image_model = params.get("image_model") or "flux2_klein_9b"
    image_params = params.get("image_params", {})
    image_loras = params.get("image_loras", {})
    video_model = params.get("video_model") or "ltx2_22B_distilled_1_1"
    supports_frame_injection = _director_supports_frame_injection(video_model)

    # Diagnostic-only log: report what the frontend sent so a future
    # "I selected N LoRAs but only K were applied" report has data we
    # can correlate against the [LoRA] Loading line wgp prints.
    _activated_in = list(image_loras.get("activated_loras", []) or [])
    _mults_in = image_loras.get("loras_multipliers", "") or ""
    if _activated_in:
        print(
            f"[Pipeline {pid}] Image LoRAs received: {len(_activated_in)} | "
            f"model={image_model} | "
            f"names={[os.path.basename(n) for n in _activated_in]} | "
            f"multipliers={_mults_in!r}"
        )

    # ── Filter image LoRAs to those that exist in the image model's dir ──
    # The frontend's DirectorLoraSelector filters available LoRAs by
    # model directory, but `savedLoraPerMode.image` persists across
    # sessions and can hold stale activations from a previous model
    # selection (e.g. an LTX-2 LoRA name that's never been in the
    # flux2_klein_9b/ directory). Without this filter, wgp.validate_task
    # rejects the entire task with "The following Loras files are missing
    # or invalid: [...]" and image gen never starts.
    #
    # This is a file-EXISTENCE check only — no architecture detection,
    # no dim peeking. Just: is the .safetensors actually in the right
    # directory? If not, drop it with a clear warning so the user knows
    # to re-select their image LoRAs for the active model.
    try:
        if _activated_in:
            try:
                _lora_dir = _wgp.get_lora_dir(image_model)
            except Exception:
                _lora_dir = ""
            if _lora_dir and os.path.isdir(_lora_dir):
                _existing = {
                    f for f in os.listdir(_lora_dir)
                    if f.lower().endswith((".safetensors", ".sft"))
                }
                _mult_tokens = _mults_in.split()
                _kept: list[str] = []
                _kept_mults: list[str] = []
                _skipped: list[str] = []
                for _idx, _name in enumerate(_activated_in):
                    _basename = os.path.basename(_name)
                    if _basename in _existing:
                        _kept.append(_name)
                        if _idx < len(_mult_tokens):
                            _kept_mults.append(_mult_tokens[_idx])
                    else:
                        _skipped.append(_basename)
                if _skipped:
                    _warn = (
                        f"Skipped {len(_skipped)} image LoRA(s) not present in "
                        f"{os.path.basename(_lora_dir)}/: {_skipped}. These were "
                        f"likely activated when a different image model was selected, "
                        f"and the saved selection persisted across sessions. Re-select "
                        f"the LoRAs you want for {image_model} in the Director image "
                        f"LoRA panel to clear the stale entries."
                    )
                    print(f"[Pipeline {pid}] {_warn}")
                    _existing_warnings = _pipelines.get(pid, {}).get("lora_warnings", []) or []
                    _update_pipeline(pid, lora_warnings=[*_existing_warnings, _warn])
                _activated_in = _kept
                _mults_in = " ".join(_kept_mults)
                image_loras = {
                    "activated_loras": _activated_in,
                    "loras_multipliers": _mults_in,
                }
                print(
                    f"[Pipeline {pid}] Image LoRAs after existence filter: "
                    f"{len(_kept)} kept, {len(_skipped)} skipped"
                )
    except Exception as _e:
        print(f"[Pipeline {pid}] LoRA file-existence filter skipped: {_e}")

    resolution = image_params.get("resolution", "1280x720")
    steps = image_params.get("num_inference_steps", 8)
    guidance = image_params.get("guidance_scale", 1)
    spatial_upsampling = params.get("image_spatial_upsampling", "")
    film_grain_intensity = params.get("image_film_grain_intensity", 0)
    film_grain_saturation = params.get("image_film_grain_saturation", 0.5)

    if not out_dir:
        out_dir = _wgp.save_path

    # Resume and Dashboard repairs can carry a generated anchor even though
    # the user-facing reference path is intentionally still empty.
    if not (ref_image_path and os.path.isfile(ref_image_path)):
        generated_anchor = params.get(
            "generated_reference_image_filename", "",
        )
        if (
            generated_anchor
            and os.path.basename(generated_anchor) == generated_anchor
        ):
            generated_anchor_path = os.path.join(out_dir, generated_anchor)
            if os.path.isfile(generated_anchor_path):
                ref_image_path = generated_anchor_path

    # Build full refs list: main scene + character refs + location refs. Keep
    # character and location refs separate so a generated identity anchor can
    # use the former without allowing location imagery to dominate the cast.
    valid_character_refs = [
        p for p in character_ref_paths if p and os.path.isfile(p)
    ]
    valid_location_refs = [
        p for p in location_ref_paths if p and os.path.isfile(p)
    ]
    extra_refs = valid_character_refs + valid_location_refs
    print(f"[Pipeline {pid}] Image refs: main={ref_image_path}, chars={len(character_ref_paths)}, locs={len(location_ref_paths)}, extra_valid={len(extra_refs)}")

    # Count total images to generate (start images + keyframes)
    total_images = len(clip_plans)
    planned_keyframes = sum(
        len(plan.get("keyframe_prompts", []) or [])
        for plan in clip_plans
    )
    if supports_frame_injection:
        total_images += planned_keyframes
    elif planned_keyframes:
        print(
            f"[Pipeline {pid}] {video_model} does not support injected "
            f"keyframes; skipping {planned_keyframes} intermediate image(s) "
            "and using each shot's start frame only.",
        )

    clip_images: list[str] = []
    clip_keyframes: list[list[str]] = []
    image_count = 0

    # Reference art-style lock: the exact lead sentence validated to hold
    # Klein to a stylized medium. Applied to EVERY image prompt (start
    # images, keyframes, anchor) at generation time — after polish, and
    # regardless of whether the planner remembered to name the medium.
    _style_prefix = _style_prefix_for(params.get("_reference_style") or "")

    def _gen_image(
        prompt: str,
        source_ref: str,
        include_extra_refs: bool = True,
        supplemental_refs: Optional[list[str]] = None,
    ) -> str:
        """Generate a single image using source_ref + optional extra refs."""
        nonlocal image_count
        _pre_strip = prompt
        prompt = _strip_motion_effects(prompt or "")
        if prompt != _pre_strip:
            print(f"[Pipeline {pid}] Stripped motion-effect language from image prompt")
        if _style_prefix and not prompt.lower().startswith("maintain the same"):
            prompt = _style_prefix + prompt
        all_refs = []
        seen_refs = set()
        selected_extra_refs = (
            extra_refs if supplemental_refs is None else supplemental_refs
        )
        for candidate in [source_ref] + (
            selected_extra_refs if include_extra_refs else []
        ):
            if not candidate or not os.path.isfile(candidate):
                continue
            resolved = os.path.normcase(os.path.realpath(candidate))
            if resolved in seen_refs:
                continue
            seen_refs.add(resolved)
            all_refs.append(candidate)
        all_refs = _limit_director_image_refs(
            image_model,
            all_refs,
            pid=pid,
        )
        print(f"[Pipeline {pid}] _gen_image: {len(all_refs)} refs: {[os.path.basename(r) for r in all_refs]}")
        gen_params: dict = {
            "model_type": image_model,
            "prompt": prompt,
            "image_refs": all_refs,
            "image_mode": 1,
            "image_prompt_type": "",
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            # 'I' carries an image reference; a ref-less anchor is plain T2I.
            "video_prompt_type": "KI" if all_refs else "",
            "resolution": resolution,
            "seed": -1,
            "settings_version": 2.52,
            "generation_mode": "image",
            "repeat_generation": 1,
            "negative_prompt": "",
            "video_length": 1,
            "activated_loras": image_loras.get("activated_loras", []),
            "loras_multipliers": image_loras.get("loras_multipliers", ""),
            "_director_pipeline_id": pid,
        }
        if spatial_upsampling:
            gen_params["spatial_upsampling"] = spatial_upsampling
        if film_grain_intensity > 0:
            gen_params["film_grain_intensity"] = film_grain_intensity
            gen_params["film_grain_saturation"] = film_grain_saturation

        output_files = _submit_and_wait(gen_params, timeout_s=600, workspace=workspace, out_dir=out_dir)
        if not output_files or not output_files[0]:
            raise RuntimeError(
                "Image generation completed without a recorded output."
            )
        image_count += 1
        return output_files[0]

    # If no reference image was provided, generate a single establishing /
    # "anchor" image from the scene description and adopt it as the shared
    # reference, so every clip's start image keeps a consistent look instead of
    # each being generated independently with no visual through-line.
    if not (ref_image_path and os.path.isfile(ref_image_path)):
        scene_desc = (params.get("scene_description") or "").strip()
        first_shot_prompt = (
            clip_plans[0].get("image_prompt", "") if clip_plans else ""
        ).strip()
        anchor_subject = first_shot_prompt or scene_desc or (
            "cinematic establishing shot"
        )
        anchor_prompt = (
            "Create a definitive cinematic character anchor for visual "
            "continuity. Clearly establish the recurring subject or people, "
            "especially faces, hair, wardrobe, body attributes, and overall "
            f"design. {anchor_subject}"
        )
        character_profiles = []
        for character in params.get("characters", []) or []:
            if not isinstance(character, dict):
                continue
            name = str(
                character.get("name")
                or character.get("display_name")
                or ""
            ).strip()
            description = str(
                character.get("description")
                or character.get("physical_description")
                or character.get("visual_description")
                or ""
            ).strip()
            wardrobe = str(character.get("wardrobe") or "").strip()
            profile = ": ".join(part for part in (name, description) if part)
            if wardrobe:
                profile = f"{profile}; wardrobe: {wardrobe}" if profile else wardrobe
            if profile:
                character_profiles.append(profile)
        if character_profiles:
            anchor_prompt += (
                " Recurring character profiles: "
                + " | ".join(character_profiles)
                + "."
            )
        if valid_character_refs:
            anchor_prompt += (
                " Use the provided character reference image(s) as the "
                "definitive identity and appearance source."
            )
        if scene_desc and scene_desc.lower() not in anchor_subject.lower():
            anchor_prompt += f" Project concept: {scene_desc}"
        total_images += 1
        _update_pipeline(pid, progress={
            "current": 0,
            "total": total_images,
            "message": "Generating establishing image",
            "step": 0, "total_steps": 0,
        })
        print(f"[Pipeline {pid}] No reference image — generating establishing/anchor image first.")
        anchor_file = _gen_image(
            anchor_prompt,
            "",
            supplemental_refs=valid_character_refs,
        )
        anchor_path = os.path.realpath(os.path.join(out_dir, anchor_file))
        output_root = os.path.realpath(os.path.abspath(out_dir))
        if (
            os.path.normcase(os.path.dirname(anchor_path))
                != os.path.normcase(output_root)
            or not os.path.isfile(anchor_path)
        ):
            raise RuntimeError(
                "The generated Director anchor could not be found in the "
                "pipeline output directory; video generation was not started."
            )
        ref_image_path = anchor_path
        params["generated_reference_image_filename"] = anchor_file
        _update_pipeline(
            pid, generated_reference_image_filename=anchor_file,
        )
        print(f"[Pipeline {pid}] Adopted establishing image as shared reference: {anchor_file}")

    for i, plan in enumerate(clip_plans):
        if _pipelines[pid]["status"] == "cancelled":
            return clip_images, clip_keyframes

        # ── Determine image source: original reference or previous scene's output ──
        image_source = plan.get("image_source", "original")
        source_ref = ref_image_path  # default: user's original reference

        if image_source == "previous" and i > 0 and clip_images[i - 1]:
            prev_img_path = os.path.join(out_dir, clip_images[i - 1])
            if os.path.isfile(prev_img_path):
                source_ref = prev_img_path
                print(f"[Pipeline {pid}] Shot {i+1}: using previous scene output as source ({clip_images[i-1]})")

        _update_pipeline(pid, progress={
            "current": image_count,
            "total": total_images,
            "message": f"Shot {i + 1}: generating start image ({image_source})",
            "step": 0, "total_steps": 0,
        })

        prompt = plan.get("image_prompt", "")
        ref_exists = os.path.isfile(source_ref) if source_ref else False
        print(f"[Pipeline {pid}] Shot {i+1} start image: source={image_source}, ref={source_ref} (exists={ref_exists}), prompt='{prompt[:60]}...'")

        img_t0 = time.time()
        try:
            if image_source == "previous" and source_ref != ref_image_path:
                # Dual reference: previous scene output as primary + original reference for character identity
                # _gen_image puts source_ref first, then extra_refs (which includes character/location refs).
                # We temporarily prepend the original ref to extra_refs so the model sees both.
                saved_extras = extra_refs[:]
                extra_refs.insert(0, ref_image_path)
                start_img = _gen_image(prompt, source_ref, include_extra_refs=True)
                extra_refs[:] = saved_extras  # restore
            else:
                start_img = _gen_image(prompt, ref_image_path)
            clip_images.append(start_img)
        except _GenerationTimeoutError:
            raise
        except Exception as e:
            print(f"[Pipeline {pid}] Shot {i+1} start image failed: {e}")
            clip_images.append("")
        # Record per-clip image timing
        timings = _pipelines.get(pid, {}).get("_clip_timings", {})
        timings[f"image_{i}"] = round(time.time() - img_t0, 2)
        _update_pipeline(pid, _clip_timings=timings)

        # ── Generate keyframes (chained from previous output) ──
        keyframe_prompts = (
            plan.get("keyframe_prompts", []) or []
        ) if supports_frame_injection else []
        shot_keyframes: list[str] = []

        if keyframe_prompts and clip_images[-1]:
            # Chain: each keyframe edits from the previous image
            chain_ref = os.path.join(out_dir, clip_images[-1])  # start from the start image

            for ki, kf_prompt in enumerate(keyframe_prompts):
                if _pipelines[pid]["status"] == "cancelled":
                    break

                # Ensure kf_prompt is a string (LLM may return dicts or other types)
                if isinstance(kf_prompt, dict):
                    kf_prompt = kf_prompt.get("prompt", kf_prompt.get("image_prompt", str(kf_prompt)))
                elif not isinstance(kf_prompt, str):
                    kf_prompt = str(kf_prompt)
                if not kf_prompt or not kf_prompt.strip():
                    continue

                _update_pipeline(pid, progress={
                    "current": image_count,
                    "total": total_images,
                    "message": f"Shot {i + 1}: keyframe {ki + 1}/{len(keyframe_prompts)}",
                    "step": 0, "total_steps": 0,
                })

                print(f"[Pipeline {pid}] Shot {i+1} keyframe {ki+1}: chain_ref='{os.path.basename(chain_ref)}', prompt='{str(kf_prompt)[:60]}...'")

                try:
                    kf_img = _gen_image(kf_prompt, chain_ref)
                    shot_keyframes.append(kf_img)
                    # Chain: next keyframe edits from this one
                    if kf_img:
                        chain_ref = os.path.join(out_dir, kf_img)
                except _GenerationTimeoutError:
                    raise
                except Exception as e:
                    print(f"[Pipeline {pid}] Shot {i+1} keyframe {ki+1} failed: {e}")
                    shot_keyframes.append("")

        clip_keyframes.append(shot_keyframes)

    _update_pipeline(pid, progress={
        "current": total_images,
        "total": total_images,
        "message": "All images generated",
        "step": 0, "total_steps": 0,
    })

    return clip_images, clip_keyframes


# ── Video Generation Phase ──────────────────────────────────────────────

def _preflight_h3_director_prompts(
    video_model: str,
    clip_plans: list[dict],
    *,
    pid: str = "",
    prompt_modes: Optional[list[str]] = None,
    durations: Optional[list[float]] = None,
    reference_manifests: Optional[list[list[dict]]] = None,
) -> list[dict]:
    """Compile official H3 Context-IR and validate it before GPU work."""

    if not str(video_model or "").lower().startswith("minimax_h3"):
        return clip_plans
    from services.director.h3_dialogue import compile_h3_clip_plans

    if (
        prompt_modes is None
        and str(video_model or "").lower().startswith("minimax_h3_ref2va")
    ):
        prompt_modes = ["ref2va"] * len(clip_plans)

    compile_h3_clip_plans(
        clip_plans,
        prompt_modes=prompt_modes,
        durations=durations,
        reference_manifests=reference_manifests,
    )
    label = f"[Pipeline {pid}]" if pid else "[Pipeline]"
    dialogue_count = sum(
        str(plan.get("video_prompt") or "").lower().count("<d>")
        for plan in clip_plans
    )
    print(
        f"{label} H3 prompt preflight passed for {len(clip_plans)} "
        f"shot(s), {dialogue_count} canonical dialogue line(s), official "
        "Context-IR field order verified."
    )
    return clip_plans


def _run_video_generation(pid: str, params: dict, clip_plans: list[dict],
                          planned_clips: list[dict], clip_images: list[str],
                          clip_keyframes: Optional[list[list[str]]] = None,
                          out_dir: str = None, workspace: str = None) -> list[str]:
    """Generate multi-clip video with optional keyframe injection. Returns list of output filenames."""
    _validate_director_models(params, stages=("video",))
    video_model = params.get("video_model") or "ltx2_22B_distilled_1_1"
    _preflight_h3_director_prompts(video_model, clip_plans, pid=pid)
    video_params = params.get("video_params", {})
    video_loras = params.get("video_loras", {})
    # Mirror of the image-LoRA file-existence filter — see _run_image_generation
    # for the rationale. Filter video_loras to those actually present in
    # video_model's LoRA directory so a stale activation from a different
    # video model doesn't crash wgp validation upfront.
    try:
        _vid_activated = list(video_loras.get("activated_loras", []) or [])
        _vid_mults = video_loras.get("loras_multipliers", "") or ""
        if _vid_activated:
            print(
                f"[Pipeline {pid}] Video LoRAs received: {len(_vid_activated)} | "
                f"model={video_model} | "
                f"names={[os.path.basename(n) for n in _vid_activated]} | "
                f"multipliers={_vid_mults!r}"
            )
            try:
                _vid_lora_dir = _wgp.get_lora_dir(video_model)
            except Exception:
                _vid_lora_dir = ""
            if _vid_lora_dir and os.path.isdir(_vid_lora_dir):
                _vid_existing = {
                    f for f in os.listdir(_vid_lora_dir)
                    if f.lower().endswith((".safetensors", ".sft"))
                }
                _vid_mult_tokens = _vid_mults.split()
                _vid_kept: list[str] = []
                _vid_kept_mults: list[str] = []
                _vid_skipped: list[str] = []
                for _idx, _name in enumerate(_vid_activated):
                    _basename = os.path.basename(_name)
                    if _basename in _vid_existing:
                        _vid_kept.append(_name)
                        if _idx < len(_vid_mult_tokens):
                            _vid_kept_mults.append(_vid_mult_tokens[_idx])
                    else:
                        _vid_skipped.append(_basename)
                if _vid_skipped:
                    _warn = (
                        f"Skipped {len(_vid_skipped)} video LoRA(s) not present in "
                        f"{os.path.basename(_vid_lora_dir)}/: {_vid_skipped}. These "
                        f"were likely activated when a different video model was "
                        f"selected. Re-select your video LoRAs for {video_model}."
                    )
                    print(f"[Pipeline {pid}] {_warn}")
                    _exw = _pipelines.get(pid, {}).get("lora_warnings", []) or []
                    _update_pipeline(pid, lora_warnings=[*_exw, _warn])
                video_loras = {
                    "activated_loras": _vid_kept,
                    "loras_multipliers": " ".join(_vid_kept_mults),
                }
                print(
                    f"[Pipeline {pid}] Video LoRAs after existence filter: "
                    f"{len(_vid_kept)} kept, {len(_vid_skipped)} skipped"
                )
    except Exception as _e:
        print(f"[Pipeline {pid}] Video LoRA file-existence filter skipped: {_e}")

    audio_path = params.get("audio_path")
    seamless = params.get("seamless", True)
    pipeline_type = params.get("pipeline_type", "music_video")
    # Get FPS from model definition (reliable) — don't trust frontend default of 16
    fps = params.get("fps", 16)
    model_def = {}
    try:
        model_def = _wgp.get_model_def(video_model) or {}
        if model_def and model_def.get("fps"):
            fps = model_def["fps"]
    except Exception:
        pass
    director_strategy = video_strategy(model_def)
    execution_profile = _director_video_execution_profile(params)
    shot_image_policy = _director_effective_shot_image_policy(params)
    uses_shot_images = shot_images_required(shot_image_policy)
    if director_strategy != ROLLING_WINDOW:
        seamless = False
    print(
        f"[Pipeline] Video gen: fps={fps}, video_model={video_model}, "
        f"strategy={director_strategy}, shot_images={shot_image_policy}"
    )

    resolution = (
        execution_profile.get("normalized_resolution")
        or video_params.get("resolution", "1280x720")
    )
    steps = video_params.get("num_inference_steps", 8)
    guidance = video_params.get("guidance_scale", 1)
    spatial_upsampling = params.get("video_spatial_upsampling", "")
    film_grain_intensity = params.get("video_film_grain_intensity", 0)
    film_grain_saturation = params.get("video_film_grain_saturation", 0.5)
    self_refiner = params.get("video_self_refiner", 0)

    if not out_dir:
        out_dir = _wgp.save_path

    # Quantize helper
    try:
        _min_f, _fs, _latent = _wgp.get_model_min_frames_and_step(video_model)
    except Exception:
        _min_f, _fs, _latent = 17, 8, 8

    def _quantize_frames(cf):
        return max((cf - 1) // _latent * _latent + 1, _min_f)

    native_window_frames = _director_native_window_frames(
        video_model,
        model_def,
        fps=fps,
        min_frames=_min_f,
        latent_size=_latent,
    )
    supports_frame_injection = bool(model_def.get("custom_frames_injection"))

    # ── SEAMLESS MODE: one continuous rolling window generation ──────
    # Instead of separate per-clip jobs, build ONE generation that looks like
    # Studio mode: rolling windows with per-window prompts + keyframe injection.
    if seamless:
        window_prompts_all = []  # One prompt per rolling window
        keyframe_images = []     # All keyframe images in order
        keyframe_frame_positions = []  # Absolute frame numbers (1-indexed for wgp parser)

        # Track cumulative frame position as we go through scenes
        cumulative_frames = 0

        for i, plan in enumerate(clip_plans):
            pc = planned_clips[i] if i < len(planned_clips) else {}
            dur_sec = pc.get("duration_sec", pc.get("end", 0) - pc.get("start", 0))
            if dur_sec <= 0:
                dur_sec = 20
            scene_frames = round(dur_sec * fps)

            wp = plan.get("window_prompts") or []
            wp = [w.get("prompt", w.get("text", str(w))) if isinstance(w, dict) else str(w) for w in wp]
            if len(wp) > 1:
                for w_prompt in wp:
                    window_prompts_all.append(w_prompt)
            else:
                vp = plan.get("video_prompt", "")
                if vp:
                    window_prompts_all.append(vp)

            # Mid-scene keyframes from the LLM (injected at mid-point of this scene)
            if clip_keyframes and i < len(clip_keyframes):
                kf_list = clip_keyframes[i]
                if kf_list:
                    # Distribute mid-scene keyframes evenly across the scene
                    num_kf = len(kf_list)
                    for ki, kf_file in enumerate(kf_list):
                        if kf_file:
                            kf_path = os.path.join(out_dir, kf_file)
                            if os.path.isfile(kf_path):
                                # Position: evenly spaced within the scene
                                kf_pos = cumulative_frames + int(scene_frames * (ki + 1) / (num_kf + 1))
                                keyframe_images.append(kf_path)
                                keyframe_frame_positions.append(kf_pos + 1)  # 1-indexed for wgp parser

            # Scene boundary keyframe: inject next scene's start image at the end of this scene
            if i < len(clip_plans) - 1:
                next_img = clip_images[i + 1] if i + 1 < len(clip_images) else ""
                if next_img:
                    next_path = os.path.join(out_dir, next_img)
                    if os.path.isfile(next_path):
                        boundary_frame = cumulative_frames + scene_frames
                        keyframe_images.append(next_path)
                        keyframe_frame_positions.append(boundary_frame)  # 1-indexed (approx)

            cumulative_frames += scene_frames

        total_frames = _quantize_frames(cumulative_frames)
        sliding_window_frames = (
            native_window_frames
            if native_window_frames is not None
            else _quantize_frames(round(20 * fps))
        )

        # First scene's start image
        first_start = ""
        if clip_images and clip_images[0]:
            first_path = os.path.join(out_dir, clip_images[0])
            if os.path.isfile(first_path):
                first_start = first_path

        prompt_text = "\n".join(window_prompts_all)

        print(f"[Pipeline {pid}] Seamless mode: {len(window_prompts_all)} windows, "
              f"{len(keyframe_images)} keyframes at frames {keyframe_frame_positions}, "
              f"{total_frames} total frames ({total_frames/fps:.1f}s)")

    # ── STANDARD MODE: separate per-clip generation ─────────────────
    else:
        prompts = []
        image_start_paths = []
        image_end_paths = []
        per_clip_frames = []
        has_sliding_window = False
        h3_timing_repaired = False
        is_minimax_h3 = str(
            model_def.get("architecture") or video_model
        ).lower().startswith("minimax_h3")
        bounded_director = director_strategy in {
            BOUNDED_START_END,
            OMNI_REFERENCE,
        }
        bounded_minimum = int(model_def.get("frames_minimum") or _min_f)
        bounded_maximum = int(
            execution_profile.get("effective_max_frames")
            or model_def.get("frames_maximum")
            or bounded_minimum
        )
        bounded_step = int(model_def.get("frames_steps") or _fs or 1)

        for i, plan in enumerate(clip_plans):
            wp = plan.get("window_prompts") or []
            wp = [w.get("prompt", w.get("text", str(w))) if isinstance(w, dict) else str(w) for w in wp]
            if len(wp) > 1:
                prompts.append("\n".join(wp))
            else:
                vp = plan.get("video_prompt", "")
                pc = planned_clips[i] if i < len(planned_clips) else {}
                dur = pc.get("duration_sec", pc.get("end", 0) - pc.get("start", 0))
                if dur > 32 and vp:
                    print(f"[Pipeline] WARNING: Clip {i+1} is {dur:.0f}s but has no window_prompts")
                prompts.append(vp)

            img_file = (
                clip_images[i]
                if uses_shot_images and i < len(clip_images)
                else ""
            )
            if img_file:
                img_path = os.path.join(out_dir, img_file)
                image_start_paths.append(img_path if os.path.isfile(img_path) else "")
            else:
                image_start_paths.append("")

            pc = planned_clips[i] if i < len(planned_clips) else {}
            end_path = ""
            if (
                uses_shot_images
                and director_strategy == BOUNDED_START_END
                and i + 1 < len(clip_plans)
            ):
                next_pc = planned_clips[i + 1] if i + 1 < len(planned_clips) else {}
                if _director_same_logical_scene(plan, pc, clip_plans[i + 1], next_pc):
                    next_file = clip_images[i + 1] if i + 1 < len(clip_images) else ""
                    candidate = os.path.join(out_dir, next_file) if next_file else ""
                    if candidate and os.path.isfile(candidate):
                        end_path = candidate
            image_end_paths.append(end_path)

            if bounded_director:
                try:
                    clip_frames = int(
                        pc.get("duration_frames")
                        or plan.get("_director_generation_frames")
                        or 0
                    )
                except (TypeError, ValueError):
                    clip_frames = 0
                if clip_frames <= 0:
                    duration = pc.get("duration_sec") or (
                        pc.get("end", 0) - pc.get("start", 0)
                    )
                    clip_frames = round(float(duration or 0) * fps)
                if not is_minimax_h3 and not (
                    bounded_minimum <= clip_frames <= bounded_maximum
                    and (clip_frames - bounded_minimum)
                    % max(1, bounded_step) == 0
                ):
                    raise RuntimeError(
                        f"Director shot {i + 1} has {clip_frames} frames, outside "
                        f"{video_model}'s native {bounded_minimum}-{bounded_maximum} "
                        f"frame lattice (step {bounded_step}). Re-plan the project "
                        "before generation."
                    )
                if not is_minimax_h3:
                    validate_director_execution_frames(
                        execution_profile,
                        clip_frames,
                        label=f"Director shot {i + 1}",
                    )
                per_clip_frames.append(clip_frames)
                continue

            window_prompts = plan.get("window_prompts", []) or []
            window_count = plan.get("window_count", 1) or 1
            if len(window_prompts) > 1 and window_count <= 1:
                window_count = len(window_prompts)
            has_keyframes = (
                supports_frame_injection
                and bool(plan.get("keyframe_prompts"))
            )
            num_keyframes = len(plan.get("keyframe_prompts", []) or [])

            if window_count > 1 or has_keyframes:
                shot_duration = pc.get("duration_sec", pc.get("end", 0) - pc.get("start", 0))
                if shot_duration <= 0:
                    shot_duration = 20 * max(window_count, num_keyframes + 1)
                clip_frames = max(round(shot_duration * fps), round(5 * fps))
                per_clip_frames.append(clip_frames)
                has_sliding_window = True
            else:
                # SECONDS are the fps-agnostic ground truth. planned_clips
                # from plan_clip_structure carry start/end (+duration_frames)
                # but NO duration_sec — the old `get("duration_sec", 0)`
                # fell straight through to duration_frames, which the
                # frontend may have had computed at the WRONG model's fps
                # (modelOptions belongs to the Studio-selected model, e.g.
                # ACE-Step right after generating the track → fps 16). A
                # 26s clip became 26x16=416 frames, rendered at LTX-2's 25
                # fps = 16.6s — every music-video clip silently shortened
                # by 16/25.
                dur_sec = pc.get("duration_sec") or (pc.get("end", 0) - pc.get("start", 0))
                clip_frames = round(dur_sec * fps) if dur_sec > 0 else pc.get("duration_frames", round(20 * fps))
                if clip_frames > round(32 * fps):
                    has_sliding_window = True
                per_clip_frames.append(max(clip_frames, round(5 * fps)))

        if is_minimax_h3 and bounded_director:
            from models.minimax_h3.minimax_h3_handler import (
                normalize_h3_clip_frame_schedule,
            )

            requested_h3_schedule = list(per_clip_frames)
            per_clip_frames = normalize_h3_clip_frame_schedule(
                requested_h3_schedule,
                minimum_frames=bounded_minimum,
                maximum_frames=bounded_maximum,
                frame_step=bounded_step,
            )
            h3_timing_repaired = per_clip_frames != requested_h3_schedule
            if h3_timing_repaired:
                print(
                    f"[Pipeline {pid}] Repaired Director H3 frame schedule "
                    f"before queueing: {requested_h3_schedule} -> "
                    f"{per_clip_frames}."
                )
            for clip_index, frame_count in enumerate(per_clip_frames):
                validate_director_execution_frames(
                    execution_profile,
                    frame_count,
                    label=f"Director shot {clip_index + 1}",
                )

        if h3_timing_repaired:
            # Keep uploaded-audio/video slicing and every later Dashboard
            # repair on the same continuous timeline as the repaired frame
            # schedule. Carried residual rounding prevents each small lattice
            # adjustment from accumulating into A/V drift across the project.
            try:
                timeline_cursor = float(
                    (planned_clips[0] if planned_clips else {}).get("start")
                    or 0.0
                )
            except (TypeError, ValueError):
                timeline_cursor = 0.0
            for clip_index, frame_count in enumerate(per_clip_frames):
                if clip_index >= len(planned_clips):
                    break
                duration_sec = frame_count / float(fps)
                planned_clip = planned_clips[clip_index]
                planned_clip.update({
                    "start": timeline_cursor,
                    "end": timeline_cursor + duration_sec,
                    "duration_sec": duration_sec,
                    "duration_frames": frame_count,
                })
                if clip_index < len(clip_plans):
                    clip_plans[clip_index]["_director_generation_frames"] = (
                        frame_count
                    )
                    clip_plans[clip_index]["_director_duration_sec"] = (
                        duration_sec
                    )
                timeline_cursor += duration_sec
            params["planned_clips"] = planned_clips
            _update_pipeline(
                pid,
                _planned_clips=planned_clips,
                clip_plans=clip_plans,
            )

        # Quantize to the model's (latent*n + 1) frame lattice WITHOUT letting
        # the error compound. Floor-snapping each clip independently lost 0-7
        # frames per clip (an 8s clip = 200 frames @25fps floors to 193 —
        # exactly the "7 frames short" the user measured), while the song
        # plays on at true time — so cuts drifted off the planned musical
        # break points by seconds near the end of a song. Instead, round each
        # clip to the NEAREST valid length and carry the residual into the
        # next clip: every cumulative boundary stays within half a latent
        # step (±4 frames ≈ 0.16s) of the planned beat, forever.
        if director_strategy == ROLLING_WINDOW:
            per_clip_frames = _quantize_clip_frame_schedule(
                per_clip_frames, _min_f, _latent,
            )
        total_frames = sum(per_clip_frames)
        max_clip_frames = max(per_clip_frames) if per_clip_frames else round(5 * fps)
        # LTX-2 has been user-validated with a single expanded window for
        # ordinary shots up to Director's ~32s planning cap. Other eligible
        # story models must stay on their own native/trained window length;
        # forcing a Wan/Ovi/LongCat model into an LTX-sized window can OOM or
        # severely degrade it. The task engine still publishes only the final
        # cumulative output for each clip after rolling-window generation.
        if director_strategy != ROLLING_WINDOW:
            sliding_window_frames = max_clip_frames
        elif native_window_frames is None:
            sliding_window_frames = (
                round(20 * fps) if has_sliding_window
                else max_clip_frames + _latent + 1
            )
        elif supports_frame_injection and not has_sliding_window:
            sliding_window_frames = max_clip_frames + _latent + 1
        else:
            sliding_window_frames = native_window_frames

        for ci, cf in enumerate(per_clip_frames):
            wp_count = len((clip_plans[ci].get("window_prompts") or []) if ci < len(clip_plans) else [])
            wc = clip_plans[ci].get("window_count", 1) if ci < len(clip_plans) else 1
            print(f"[Pipeline {pid}] Clip {ci+1}: {cf} frames ({cf/fps:.1f}s), windows={wc}, window_prompts={wp_count}")

    # Build audio params
    audio_params: dict = {}
    per_clip_h3_references: list[list[dict]] = []
    temporary_h3_audio: list[str] = []
    audio_start_sec = (
        _audio_timeline_start(planned_clips)
        if pipeline_type != "short_film_story" and audio_path
        else 0.0
    )
    if director_strategy == OMNI_REFERENCE:
        if uses_shot_images and (
            not image_start_paths or not all(image_start_paths)
        ):
            raise RuntimeError(
                "MiniMax H3 Omni Director needs a valid generated composition "
                "image for every shot. Repair the missing start images first."
            )
        if (
            pipeline_type != "short_film_story"
            and (not audio_path or not os.path.isfile(audio_path))
        ):
            raise RuntimeError(
                "MiniMax H3 Omni Director needs the uploaded soundtrack or "
                "dialogue audio for this workflow."
            )
        pid_token = re.sub(r"[^A-Za-z0-9_-]", "_", pid)[:32]
        cumulative_frames = 0
        try:
            for index, (clip_image, clip_frames) in enumerate(
                zip(image_start_paths, per_clip_frames)
            ):
                drive_slice = None
                if pipeline_type != "short_film_story":
                    drive_slice = os.path.join(
                        out_dir,
                        f"_director_h3_audio_{pid_token}_c{index}_{uuid.uuid4().hex[:8]}.wav",
                    )
                    clip_start = audio_start_sec + cumulative_frames / fps
                    _slice_audio_segment(
                        audio_path,
                        clip_start,
                        clip_frames / fps,
                        drive_slice,
                    )
                    temporary_h3_audio.append(drive_slice)
                manifest = _director_h3_reference_manifest(
                    params,
                    clip_image if uses_shot_images else None,
                    out_dir=out_dir,
                    drive_audio_path=drive_slice,
                )
                if not any(
                    reference.get("type") in {"image", "video"}
                    for reference in manifest
                ):
                    raise RuntimeError(
                        "MiniMax H3 Omni Director has no valid visual "
                        "reference. Restore a main, character, or location "
                        "image, or enable generated shot images."
                    )
                per_clip_h3_references.append(manifest)
                cumulative_frames += clip_frames
        except Exception:
            for temporary_path in temporary_h3_audio:
                if os.path.isfile(temporary_path):
                    try:
                        os.remove(temporary_path)
                    except OSError:
                        pass
            raise
        print(
            f"[Pipeline {pid}] Built {len(per_clip_h3_references)} H3 Omni "
            "shot manifest(s) with explicitly mapped visual and audio roles."
        )
    elif pipeline_type == "short_film_story":
        audio_params["audio_prompt_type"] = ""
    elif audio_path:
        audio_params["audio_prompt_type"] = "A"
        audio_params["audio_guide"] = audio_path
        # Music analysis may intentionally omit a silent intro. Align model
        # conditioning to the source-audio time represented by video frame 0.
        audio_params["audio_frame_offset"] = round(audio_start_sec * fps)
        audio_scale = params.get("audio_scale")
        if audio_scale is not None:
            audio_params["audio_scale"] = audio_scale

    # ── Build gen_params based on mode ──────────────────────────────
    lora_params = {
        "activated_loras": video_loras.get("activated_loras", []),
        "loras_multipliers": " ".join(
            m.split(";")[0] for m in (video_loras.get("loras_multipliers", "") or "").split(" ") if m
        ),
    }

    if seamless:
        # Seamless: ONE generation job with rolling windows + keyframe injection
        gen_params: dict = {
            "model_type": video_model,
            "prompt": prompt_text,
            "image_mode": 0,
            "multi_prompts_gen_type": 0,  # Rolling window mode (one prompt per window)
            "image_prompt_type": "S" if first_start else "",
            "video_prompt_type": "",
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "resolution": resolution,
            "video_length": total_frames,
            "sliding_window_size": sliding_window_frames,
            "seed": -1,
            "settings_version": 2.52,
            "generation_mode": "video",
            "repeat_generation": 1,
            "negative_prompt": "",
            "self_refiner_setting": self_refiner,
            "_director_pipeline_id": pid,
            **lora_params,
            **audio_params,
        }
        if first_start:
            gen_params["image_start"] = first_start
        # Keyframe injection via image_refs + frames_positions (numeric absolute positions)
        if keyframe_images:
            gen_params["image_refs"] = keyframe_images
            gen_params["frames_positions"] = " ".join(str(p) for p in keyframe_frame_positions)
            existing_vpt = gen_params.get("video_prompt_type", "")
            if "KFI" not in existing_vpt:
                gen_params["video_prompt_type"] = existing_vpt + "KFI"
            print(f"[Pipeline {pid}] Seamless keyframes: {len(keyframe_images)} images at frames {keyframe_frame_positions}")

    else:
        # Standard: separate per-clip generation jobs
        CLIP_SEPARATOR = "\n---CLIP_BOUNDARY---\n"

        has_any_start = any(p for p in image_start_paths)
        has_any_end = any(p for p in image_end_paths)
        if (
            uses_shot_images
            and director_strategy == BOUNDED_START_END
            and (
            not image_start_paths or not all(image_start_paths)
            )
        ):
            raise RuntimeError(
                "MiniMax H3 FL2VA Director needs a valid generated start "
                "image for every shot. Repair the missing start images first."
            )
        if not has_any_start:
            image_start_paths = []
        if not has_any_end:
            image_end_paths = []

        # Prompt-only FL2VA shots may continue from the preceding generated
        # final frame when they are duration-split segments or the planner
        # explicitly marks a literal same-composition continuation. Ordinary
        # editorial cuts remain true T2V shots and do not inherit composition.
        per_clip_continue_from_previous = [False] * len(clip_plans)
        if director_strategy == BOUNDED_START_END and not uses_shot_images:
            for index in range(1, len(clip_plans)):
                previous_clip = (
                    planned_clips[index - 1]
                    if index - 1 < len(planned_clips)
                    else {}
                )
                current_clip = (
                    planned_clips[index]
                    if index < len(planned_clips)
                    else {}
                )
                per_clip_continue_from_previous[index] = (
                    _director_same_logical_scene(
                        clip_plans[index - 1],
                        previous_clip,
                        clip_plans[index],
                        current_clip,
                    )
                )

        if str(video_model or "").lower().startswith("minimax_h3"):
            final_prompt_modes: list[str] = []
            for index in range(len(clip_plans)):
                if director_strategy == OMNI_REFERENCE:
                    final_prompt_modes.append("ref2va")
                    continue
                has_start = bool(
                    index < len(image_start_paths) and image_start_paths[index]
                ) or bool(per_clip_continue_from_previous[index])
                has_end = bool(
                    index < len(image_end_paths) and image_end_paths[index]
                )
                final_prompt_modes.append(
                    "fl2va" if has_start and has_end
                    else "i2va" if has_start
                    else "l2va" if has_end
                    else "t2va"
                )
            _preflight_h3_director_prompts(
                video_model,
                clip_plans,
                pid=pid,
                prompt_modes=final_prompt_modes,
                durations=[frames / fps for frames in per_clip_frames],
                reference_manifests=(
                    per_clip_h3_references
                    if director_strategy == OMNI_REFERENCE
                    else None
                ),
            )
            prompts = [str(plan.get("video_prompt") or "") for plan in clip_plans]

        prompt_text = CLIP_SEPARATOR.join(prompts)

        ipt = (
            ""
            if director_strategy == OMNI_REFERENCE
            else "SE" if has_any_start and has_any_end
            else "S" if has_any_start
            else ""
        )

        gen_params: dict = {
            "model_type": video_model,
            "prompt": prompt_text,
            "image_mode": 0,
            "multi_prompts_gen_type": 3,  # Multi-clip mode
            "image_prompt_type": ipt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "resolution": resolution,
            "video_length": total_frames,
            "sliding_window_size": sliding_window_frames,
            "per_clip_frames": per_clip_frames,
            "multi_clip_audio_start_sec": audio_start_sec,
            "seed": -1,
            "settings_version": 2.52,
            "generation_mode": "video",
            "repeat_generation": 1,
            "negative_prompt": "",
            "self_refiner_setting": self_refiner,
            "_director_pipeline_id": pid,
            **lora_params,
            **audio_params,
        }
        if has_any_start and director_strategy != OMNI_REFERENCE:
            gen_params["image_start"] = image_start_paths
        if has_any_end:
            gen_params["image_end"] = image_end_paths
        if any(per_clip_continue_from_previous):
            gen_params["per_clip_continue_from_previous"] = (
                per_clip_continue_from_previous
            )
        if director_strategy == OMNI_REFERENCE:
            gen_params["per_clip_minimax_h3_references"] = per_clip_h3_references
            gen_params["minimax_h3_reference_detail"] = "match"
            if pipeline_type != "short_film_story" and audio_path:
                # Ref2VA receives per-shot slices through its manifest, but
                # the final join still uses the pristine continuous source
                # track to avoid audible boundaries between generated clips.
                gen_params["multi_clip_concat_audio"] = audio_path
        # Per-clip keyframe injection
        if supports_frame_injection and clip_keyframes:
            per_clip_kf_paths: list[list[str]] = []
            for i, kf_list in enumerate(clip_keyframes):
                paths = []
                for kf_file in kf_list:
                    if kf_file:
                        kf_path = os.path.join(out_dir, kf_file)
                        if os.path.isfile(kf_path):
                            paths.append(kf_path)
                per_clip_kf_paths.append(paths)
            if any(paths for paths in per_clip_kf_paths):
                gen_params["per_clip_keyframes"] = per_clip_kf_paths
                print(f"[Pipeline {pid}] Keyframe injection: {[len(p) for p in per_clip_kf_paths]} keyframes per clip")

    # Common params
    if execution_profile.get("is_minimax_h3"):
        gen_params["_director_video_execution_profile"] = execution_profile
        gen_params["minimax_h3_turbo_mode"] = bool(
            video_params.get("minimax_h3_turbo_mode")
        )
    voice_ref = params.get("voice_reference")
    if voice_ref and director_strategy != OMNI_REFERENCE:
        gen_params["voice_reference"] = voice_ref
        gen_params["identity_guidance_scale"] = params.get("identity_guidance_scale", 3.0)
        print(f"[Pipeline {pid}] Voice reference: {voice_ref}, identity_scale={gen_params['identity_guidance_scale']}")
    if spatial_upsampling:
        gen_params["spatial_upsampling"] = spatial_upsampling
    if film_grain_intensity > 0:
        gen_params["film_grain_intensity"] = film_grain_intensity
        gen_params["film_grain_saturation"] = film_grain_saturation

    # Track progress by monitoring the generation job
    try:
        output_files = _submit_and_wait(
            gen_params,
            timeout_s=28000,
            workspace=workspace,
            out_dir=out_dir,
        )  # extended timeout for long videos
    finally:
        for temporary_path in temporary_h3_audio:
            if os.path.isfile(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
    return output_files
