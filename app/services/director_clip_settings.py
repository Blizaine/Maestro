"""Read-time reconstruction of a single Director clip's Studio recipe.

Director sidecars written before this module existed describe the entire
multi-clip request, even when attached to one clip file.  This module keeps the
sidecar as the source of its historical prompt/settings and consults the
saved pipeline only for clip identity, timing, and archived assets.  It never
rewrites a pipeline or an existing sidecar.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from urllib.parse import quote


DIRECTOR_CLIP_SETTINGS_VERSION = 1
_PIPELINE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CLIP_SEPARATOR = "\n---CLIP_BOUNDARY---\n"
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov"}
_DROP_VALUE = object()
_AUDIO_SLICE_LOCK = threading.RLock()


def json_safe_generation_params(
    params,
    *,
    path_params=None,
    attachment_keys=(),
    omit_keys=(),
):
    """Copy scalar settings without runtime state or loaded media objects.

    WGP attaches PIL images and its live ``state`` object to validated task
    params. Sidecars need the effective scalar values plus the original
    manifest paths, never a copy of those runtime objects.
    """

    omitted = {str(key).lower() for key in omit_keys}
    safe = _json_value(params, omitted, set(), 0)
    if not isinstance(safe, dict):
        safe = {}
    if path_params is not None:
        paths = _json_value(path_params, omitted, set(), 0)
        if isinstance(paths, dict):
            for key in attachment_keys:
                if key in paths:
                    safe[key] = paths[key]
    return safe


def _json_value(value, omitted, seen, depth):
    if depth > 30:
        return _DROP_VALUE
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _DROP_VALUE
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        identity = id(value)
        if identity in seen:
            return _DROP_VALUE
        seen.add(identity)
        result = {}
        for key, child in value.items():
            if not isinstance(key, str) or key.lower() == "state" or key.lower() in omitted:
                continue
            safe_child = _json_value(child, omitted, seen, depth + 1)
            if safe_child is not _DROP_VALUE:
                result[key] = safe_child
        seen.remove(identity)
        return result
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return _DROP_VALUE
        seen.add(identity)
        result = []
        for child in value:
            safe_child = _json_value(child, omitted, seen, depth + 1)
            if safe_child is not _DROP_VALUE:
                result.append(safe_child)
        seen.remove(identity)
        return result
    # PIL.Image instances retain the source path when opened from an image
    # attachment. Use it only when it still names a real file; never stringify
    # a model/runtime object into a bogus path.
    if type(value).__module__.startswith("PIL."):
        filename = getattr(value, "filename", None)
        if isinstance(filename, str) and os.path.isfile(filename):
            return filename
    if type(value).__module__.startswith("numpy"):
        try:
            return _json_value(value.item(), omitted, seen, depth + 1)
        except Exception:
            pass
    return _DROP_VALUE


def enrich_director_output_metadata(
    out_dir: str,
    workspace: str,
    filename: str,
    metadata: dict,
) -> dict:
    """Return a safe single-clip recipe or an actionable restore error.

    ``metadata`` is the in-memory sidecar/embedded object returned by the
    gallery API.  The function is deliberately read-only with respect to
    saved pipeline JSON and sidecars.  Audio slices are the only possible
    write, and they are stable, scoped derivatives under the workspace.
    """

    result = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
    params = result.get("params")
    params = params if isinstance(params, dict) else {}
    pipeline_id = result.get("director_pipeline_id") or params.get(
        "_director_pipeline_id"
    )
    if not pipeline_id:
        return result

    extension = os.path.splitext(str(filename))[1].lower()
    if extension not in _VIDEO_EXTENSIONS:
        # Director stills are not video clip recipes.
        return result

    result["director_pipeline_id"] = str(pipeline_id)
    result["director_clip_settings_version"] = (
        DIRECTOR_CLIP_SETTINGS_VERSION
    )
    if result.get("director_artifact_kind") == "joined":
        result.pop("director_clip_index", None)
        result.pop("director_clip_settings_error", None)
        return result
    state, state_root, state_error = _read_pipeline_state(out_dir, pipeline_id)
    trusted_child_recipe = (
        result.get("director_clip_settings_provenance")
        == "per_output_effective_params"
    )
    if state_error and trusted_child_recipe and "was not found" in state_error:
        # Future sidecars carry their own exact effective child parameters,
        # native/output timing, and audio offset. They remain usable if a user
        # later removes the mutable Director plan.
        state, state_root, state_error = {}, os.path.realpath(out_dir), None
    if state_error:
        return _restore_error(
            result,
            "Director clip settings are unavailable: " + state_error,
        )

    clip_index, provenance, classification_error = _identify_clip(
        result, params, state, filename,
    )
    if classification_error:
        kind = "joined" if classification_error == "joined" else "unknown"
        result["director_artifact_kind"] = kind
        if kind == "joined":
            result.pop("director_clip_index", None)
            result.pop("director_clip_settings_error", None)
            return result
        return _restore_error(
            result,
            "Could not identify this Director video as one saved clip or a "
            "joined output. Open the Director pipeline to inspect its plan.",
            kind=kind,
        )

    result["director_artifact_kind"] = "clip"
    result["director_clip_index"] = clip_index
    result["director_clip_settings_provenance"] = provenance
    result.pop("director_clip_settings_error", None)

    clips = state.get("clips") if isinstance(state, dict) else None
    clip = (
        clips[clip_index]
        if isinstance(clips, list) and 0 <= clip_index < len(clips)
        and isinstance(clips[clip_index], dict)
        else {}
    )
    planned = clip.get("planned_clip") if isinstance(clip, dict) else {}
    planned = planned if isinstance(planned, dict) else {}
    snapshot = state.get("_params_snapshot") or {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}

    recipe = copy.deepcopy(params)
    result.pop("h3_window_plan", None)
    for key in list(result):
        if str(key).lower().startswith(("_prompt_enhancement", "prompt_enhancement")):
            result.pop(key, None)
    raw_prompt = recipe.get("prompt")
    prompt, prompt_error = _prompt_for_clip(raw_prompt, clip_index, state)
    if prompt_error:
        return _restore_error(
            result,
            prompt_error,
            kind="clip",
            clip_index=clip_index,
            provenance=provenance,
        )
    recipe["prompt"] = prompt

    native_frames, output_frames, fps = _clip_timing(
        recipe, planned, state, snapshot, result,
    )
    if native_frames is None or fps is None:
        return _restore_error(
            result,
            "The saved Director state does not contain a usable native clip "
            "duration and frame rate. The full-run duration was not restored.",
            kind="clip",
            clip_index=clip_index,
            provenance=provenance,
        )

    recipe["video_length"] = native_frames
    recipe["duration_seconds"] = round(native_frames / fps, 6)
    recipe["trim_tail_frames"] = max(0, native_frames - output_frames)
    try:
        old_window = int(recipe.get("sliding_window_size") or native_frames)
    except (TypeError, ValueError):
        old_window = native_frames
    recipe["sliding_window_size"] = min(max(old_window, 1), native_frames)
    recipe["multi_prompts_gen_type"] = 0
    if str(recipe.get("model_type") or "").lower().startswith("minimax_h3"):
        recipe["h3_window_prompts"] = [prompt]

    video_model = str(
        recipe.get("model_type") or state.get("video_model") or ""
    )
    recipe["model_type"] = video_model
    _restore_saved_model_settings(recipe, state, provenance)
    _sanitize_multiclip_params(recipe)

    required_asset_error = _restore_clip_assets(
        recipe,
        result,
        state,
        state_root,
        planned,
        clip,
        clip_index,
        fps,
        native_frames,
        workspace,
        out_dir,
        filename,
    )
    if required_asset_error:
        return _restore_error(
            result,
            required_asset_error,
            kind="clip",
            clip_index=clip_index,
            provenance=provenance,
        )

    result["params"] = recipe
    result["director_clip_timing"] = {
        "native_frames": native_frames,
        "output_frames": output_frames,
        "fps": fps,
        "native_duration_seconds": round(native_frames / fps, 6),
        "output_duration_seconds": round(output_frames / fps, 6),
        "trim_tail_frames": max(0, native_frames - output_frames),
    }
    return result


def _restore_error(
    metadata: dict,
    message: str,
    *,
    kind: str = "unknown",
    clip_index: int | None = None,
    provenance: str | None = None,
) -> dict:
    metadata["director_artifact_kind"] = kind
    metadata["director_clip_settings_version"] = (
        DIRECTOR_CLIP_SETTINGS_VERSION
    )
    metadata["director_clip_settings_error"] = message
    if clip_index is not None:
        metadata["director_clip_index"] = clip_index
    if provenance:
        metadata["director_clip_settings_provenance"] = provenance
    # Never leave a whole-project prompt available to a Restore button after
    # we have determined that this is an unsafe clip recipe.
    metadata["params"] = None
    return metadata


def _read_pipeline_state(out_dir: str, pipeline_id):
    token = str(pipeline_id or "")
    if not _PIPELINE_ID_RE.fullmatch(token):
        return None, None, "the pipeline id is invalid"
    root = os.path.realpath(out_dir)
    target = f"_director_pipeline_{token}.json"
    candidates = []
    try:
        direct = os.path.join(root, target)
        if os.path.isfile(direct) and _inside(root, direct):
            candidates.append((direct, root))
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                candidate_root = os.path.realpath(entry.path)
                if not _inside(root, candidate_root):
                    continue
                candidate = os.path.join(candidate_root, target)
                if os.path.isfile(candidate) and _inside(root, candidate):
                    candidates.append((candidate, candidate_root))
    except OSError:
        return None, None, "the output folder cannot be scanned"
    if not candidates:
        return None, None, f"saved pipeline {token} was not found"
    if len(candidates) != 1:
        return None, None, f"saved pipeline {token} is ambiguous"
    path, state_root = candidates[0]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None, f"saved pipeline {token} could not be read"
    if not isinstance(state, dict):
        return None, None, f"saved pipeline {token} has an invalid format"
    saved_id = state.get("pipeline_id") or state.get("id")
    if saved_id and str(saved_id) != token:
        return None, None, f"saved pipeline {token} did not match its filename"
    return state, state_root, None


def _identify_clip(metadata, params, state, filename):
    clips = state.get("clips")
    if not isinstance(clips, list):
        clips = []
    explicit = _nonnegative_int(metadata.get("director_clip_index"))
    provenance = metadata.get("director_clip_settings_provenance")
    if explicit is None:
        explicit = _nonnegative_int(params.get("_director_clip_index"))
        if explicit is not None:
            provenance = "director_rerun_request_marker"
    if explicit is not None:
        if explicit >= len(clips) and not (
            metadata.get("director_clip_settings_provenance")
            == "per_output_effective_params"
        ):
            return None, None, "Director clip index is outside the saved plan"
        if not provenance:
            provenance = "legacy_batch_sidecar"
        return explicit, provenance, None

    exact = []
    basename = os.path.basename(str(filename))
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            continue
        saved_name = clip.get("video_filename")
        if saved_name and os.path.basename(str(saved_name)) == basename:
            exact.append(index)
    if len(exact) == 1:
        return exact[0], "legacy_rerun_pipeline_match", None
    if len(exact) > 1:
        return None, None, "matching clip filenames are ambiguous"

    # New joined outputs carry an explicit kind. For older output files,
    # require both the concrete join naming and membership in saved outputs.
    if metadata.get("director_artifact_kind") == "joined":
        return None, None, "joined"
    output_files = state.get("output_files") or []
    is_saved_output = any(
        os.path.basename(str(saved)) == basename
        for saved in output_files
        if isinstance(saved, str)
    )
    if (
        is_saved_output
        and "_multiclip" in os.path.splitext(basename)[0].lower()
        and params.get("multi_prompts_gen_type") == 3
    ):
        return None, None, "joined"
    return None, None, "unmatched"


def _prompt_for_clip(raw_prompt, clip_index, state):
    if not isinstance(raw_prompt, str) or not raw_prompt:
        return None, "This Director sidecar has no saved prompt for its clip."
    if _CLIP_SEPARATOR not in raw_prompt:
        return raw_prompt, None
    parts = raw_prompt.split(_CLIP_SEPARATOR)
    clips = state.get("clips") or []
    if len(parts) != len(clips):
        return None, (
            "The sidecar's saved clip prompts do not match the saved Director "
            "clip count, so the historical prompt cannot be identified safely."
        )
    if clip_index >= len(parts):
        return None, "The sidecar does not contain a prompt for this clip."
    return parts[clip_index], None


def _clip_timing(recipe, planned, state, snapshot, metadata):
    # The sidecar belongs to the output being restored. Prefer the original
    # request's per-clip values because a later pipeline edit may have changed
    # the current plan's duration.
    clip_index = _nonnegative_int(metadata.get("director_clip_index"))
    provenance = metadata.get("director_clip_settings_provenance")
    direct_clip_sidecar = provenance in {
        "per_output_effective_params",
        "director_rerun_request_marker",
        "legacy_rerun_pipeline_match",
    }
    per_clip_frames = recipe.get("per_clip_frames")
    native = _positive_int(_indexed_value(per_clip_frames, clip_index or 0))
    if native is None:
        timing = metadata.get("director_clip_timing") or {}
        native = _positive_int(timing.get("native_frames"))
    if native is None and direct_clip_sidecar:
        native = _positive_int(recipe.get("video_length"))
    if native is None:
        native = _positive_int(planned.get("duration_frames"))
    if native is None:
        native = _positive_int(recipe.get("video_length"))

    per_clip_outputs = recipe.get("per_clip_output_frames")
    output = _positive_int(_indexed_value(per_clip_outputs, clip_index or 0))
    if output is None:
        timing = metadata.get("director_clip_timing") or {}
        output = _positive_int(timing.get("output_frames"))
    if output is None and direct_clip_sidecar:
        trim = _nonnegative_int(recipe.get("trim_tail_frames")) or 0
        output = max(1, native - trim) if native is not None else None
    if output is None:
        output = _positive_int(planned.get("output_frames"))
    if output is None:
        trim = _nonnegative_int(recipe.get("trim_tail_frames")) or 0
        output = max(1, native - trim) if native is not None else None
    if native is not None and output is not None:
        output = min(output, native)

    timing = metadata.get("director_clip_timing") or {}
    recipe_model = str(recipe.get("model_type") or "")
    state_model = str(state.get("video_model") or "")
    state_profile_matches = bool(
        not recipe_model or not state_model or recipe_model == state_model
    )
    profile = state.get("video_execution_profile") or {}
    profile = profile if isinstance(profile, dict) else {}
    snapshot_profile = snapshot.get("_director_video_execution_profile") or {}
    snapshot_profile = snapshot_profile if isinstance(snapshot_profile, dict) else {}
    fps = _positive_float(
        timing.get("fps")
        or recipe.get("fps")
        or (
            profile.get("fps")
            or snapshot_profile.get("fps")
            or snapshot.get("fps")
            if state_profile_matches else None
        )
    )
    if native is None or output is None:
        return None, None, fps
    return native, output, fps


def _restore_saved_model_settings(recipe, state, provenance):
    params = recipe
    state_model = str(state.get("video_model") or "")
    recipe_model = str(params.get("model_type") or "")
    state_model_matches = bool(recipe_model and recipe_model == state_model)
    saved = state.get("video_params") or {} if state_model_matches else {}
    if not isinstance(saved, dict):
        saved = {}
    if not params.get("model_type"):
        params["model_type"] = state.get("video_model") or ""

    # Older sidecars omitted the values enforced by the fused H3 boundary.
    profile = state.get("video_execution_profile") or {}
    if not isinstance(profile, dict):
        profile = {}
    snapshot = state.get("_params_snapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    snapshot_profile = snapshot.get("_director_video_execution_profile") or {}
    if not isinstance(snapshot_profile, dict):
        snapshot_profile = {}
    own_profile = params.get("_director_video_execution_profile") or {}
    if not isinstance(own_profile, dict):
        own_profile = {}
    is_fused = bool(
        own_profile.get("fused_turbo")
        or "fused_turbo" in str(params.get("model_type") or "").lower()
        or (
            state_model_matches
            and (profile.get("fused_turbo") or snapshot_profile.get("fused_turbo"))
        )
    )
    for key in (
        "num_inference_steps", "activated_loras", "loras_multipliers",
        "resolution", "override_attention", "denoising_strength",
        "guidance_scale", "flow_shift", "audio_flow_shift",
        "skip_steps_cache_type",
        "minimax_h3_text_encoder", "minimax_h3_video_vae", "minimax_h3_reference_detail",
    ):
        if (key not in params or params.get(key) is None) and key in saved:
            params[key] = copy.deepcopy(saved[key])
    if is_fused:
        params["minimax_h3_turbo_mode"] = False
        params["minimax_h3_turbo_preset"] = ""
        params["guidance_scale"] = 1.0
        params["flow_shift"] = 12.0
        params["audio_flow_shift"] = 3.0
        params["skip_steps_cache_type"] = ""
        attention = str(
            params.get("override_attention")
            or saved.get("override_attention")
            or ""
        ).strip().lower()
        params["override_attention"] = (
            attention if attention in {"sla", "veda", "sdpa"} else "sla"
        )


def _sanitize_multiclip_params(params):
    for key in list(params):
        lowered = str(key).lower()
        if (
            lowered.startswith("_director_")
            or lowered.startswith("director_")
            or lowered.startswith("per_clip_")
            or lowered.startswith("multi_clip_")
            or lowered in {
                "multi_clip_info", "multi_clip_concat_audio",
                "multi_clip_audio_start_sec", "audio_frame_offset",
                "video_frame_offset", "_omni_sequence_continuity",
                "_h3_window_plan_reviewed", "_reviewed_h3_runtime_geometry",
                "h3_window_plan", "h3_window_plan_signature",
            }
            or lowered.startswith("_prompt_enhancement")
        ):
            params.pop(key, None)
    params["multi_prompts_gen_type"] = 0
    params["repeat_generation"] = 1


def _restore_clip_assets(
    params,
    metadata,
    state,
    state_root,
    planned,
    clip,
    clip_index,
    fps,
    native_frames,
    workspace,
    out_dir,
    filename,
):
    uploads = metadata.get("upload_filenames")
    uploads = copy.deepcopy(uploads) if isinstance(uploads, dict) else {}
    urls = {}
    manifest = state.get("asset_manifest") or {}
    manifest = manifest if isinstance(manifest, dict) else {}
    snapshot = state.get("_params_snapshot") or {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}

    def register(key, value):
        resolved = _resolve_asset(out_dir, state_root, value)
        relative = _safe_relative_asset(out_dir, resolved) if resolved else None
        if not relative:
            return None
        uploads[key] = relative
        url = _asset_url(relative, workspace)
        urls[key] = url
        return os.path.join(out_dir, *relative.split("/"))

    # Preserve per-clip Omni references, including their role/intent metadata.
    original_params = metadata.get("params") or {}
    per_clip_refs = original_params.get("per_clip_minimax_h3_references")
    if isinstance(per_clip_refs, list):
        if clip_index >= len(per_clip_refs):
            return (
                "The sidecar's per-clip identity references do not match its "
                "saved clip index. Restore the original reference manifest."
            )
        selected_refs = per_clip_refs[clip_index]
    elif "minimax_h3_references" in original_params:
        # An explicit empty list is meaningful: this child used no identity
        # references, even if the project now has a different manifest.
        selected_refs = original_params.get("minimax_h3_references")
    else:
        selected_refs = None
    expected_ref_count = len(selected_refs) if isinstance(selected_refs, list) else 0
    if selected_refs is None:
        selected_refs = _manifest_reference_records(manifest)
        expected_ref_count = len(selected_refs)
    elif not isinstance(selected_refs, list):
        return "The saved identity-reference list has an invalid format."
    refs = []
    for item in selected_refs if isinstance(selected_refs, list) else []:
        if not isinstance(item, dict):
            continue
        path_value = item.get("path")
        if isinstance(path_value, dict):
            path_value = path_value.get("path") or path_value.get("serve_path")
        resolved = _resolve_asset(out_dir, state_root, path_value)
        if not resolved:
            continue
        rel = _safe_relative_asset(out_dir, resolved)
        if not rel:
            continue
        restored = copy.deepcopy(item)
        restored["path"] = resolved
        restored["url"] = _asset_url(rel, workspace)
        refs.append(restored)
    if expected_ref_count and len(refs) != expected_ref_count:
        return (
            "One or more archived identity references used by this Director "
            "clip are missing. Restore the reference assets before loading "
            "this recipe."
        )
    if refs:
        params["minimax_h3_references"] = refs
        ref_rel = [
            _safe_relative_asset(out_dir, item["path"]) for item in refs
        ]
        uploads["minimax_h3_references"] = ref_rel
        urls["minimax_h3_references"] = [
            _asset_url(path, workspace) for path in ref_rel if path
        ]

    timing_metadata = metadata.get("director_clip_timing") or {}
    if not isinstance(timing_metadata, dict):
        timing_metadata = {}
    original_model = str(
        original_params.get("model_type") or params.get("model_type") or ""
    )
    saved_model = str(state.get("video_model") or "")
    state_profile_matches = bool(
        not original_model or not saved_model or original_model == saved_model
    )
    profile = state.get("video_execution_profile") or {}
    profile = profile if isinstance(profile, dict) else {}
    strategy = str(
        timing_metadata.get("video_strategy")
        or original_params.get("_director_video_strategy")
        or (
            profile.get("video_strategy")
            if state_profile_matches else ""
        )
        or ""
    ).lower()
    image_policy = str(
        timing_metadata.get("shot_image_policy")
        or original_params.get("_director_shot_image_policy")
        or (
            state.get("shot_image_policy")
            if state_profile_matches else ""
        )
        or ""
    ).lower()
    is_omni = (
        strategy == "omni_reference"
        or image_policy == "direct_references"
        or "ref2va" in str(params.get("model_type") or "").lower()
    )
    if is_omni:
        params.pop("image_start", None)
        params.pop("image_end", None)
        params["image_prompt_type"] = ""
        for key in ("image_start", "image_end"):
            uploads.pop(key, None)
            urls.pop(key, None)
        params["_studio_video_workflow"] = "references"
    else:
        start_value = _indexed_value(original_params.get("image_start"), clip_index)
        end_value = _indexed_value(original_params.get("image_end"), clip_index)
        image_mode = str(original_params.get("image_prompt_type") or "").upper()
        direct_child = (
            metadata.get("director_clip_settings_provenance")
            == "per_output_effective_params"
        )
        used_frame_strategy = bool(
            image_policy == "generate"
            or (direct_child and (start_value or end_value))
            or (image_mode in {"S", "SE", "E"} and (start_value or end_value))
        )
        if used_frame_strategy:
            if not start_value and image_policy == "generate":
                start_name = clip.get("start_image_filename")
                if start_name and os.path.basename(str(start_name)) == str(start_name):
                    start_value = os.path.join(state_root, str(start_name))
            if start_value:
                start_path = register("image_start", start_value)
                if not start_path:
                    return "The saved start image for this Director clip is missing. Restore it before loading this recipe."
                params["image_start"] = start_path
                params["_studio_video_workflow"] = "frames"
            if end_value:
                end_path = register("image_end", end_value)
                if not end_path:
                    return "The saved end image for this Director clip is missing. Restore it before loading this recipe."
                params["image_end"] = end_path
                params["image_prompt_type"] = "SE" if start_value else "E"
                params["_studio_video_workflow"] = "frames"

    # Retain the user's voice/identity reference as a reference. It is never
    # treated as driving audio and is never sliced with the soundtrack.
    voice_value = original_params.get("voice_reference")
    if not voice_value:
        voice_value = snapshot.get("voice_reference")
    if voice_value:
        voice_path = register("voice_reference", voice_value)
        if not voice_path:
            return (
                "The archived voice reference used by this Director clip is "
                "missing. Restore that asset before loading this recipe."
            )
        params["voice_reference"] = voice_path

    # Other reference assets come from the output's own metadata where
    # possible; only archived paths inside the workspace are exposed.
    for key in ("image_refs", "audio_guide2", "audio_guide3", "audio_guide4",
                "audio_guide5", "audio_guide6", "audio_conditioning_guide"):
        value = original_params.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            resolved_values = []
            rel_values = []
            for entry in value:
                path_value = (
                    entry.get("path") or entry.get("filename")
                    if isinstance(entry, dict) else entry
                )
                if isinstance(path_value, dict):
                    path_value = path_value.get("path") or path_value.get(
                        "serve_path"
                    )
                resolved = _resolve_asset(out_dir, state_root, path_value)
                rel = _safe_relative_asset(out_dir, resolved) if resolved else None
                if rel:
                    resolved_values.append(resolved)
                    rel_values.append(rel)
            if len(resolved_values) != len(value):
                return (
                    f"One or more archived {key.replace('_', ' ')} assets "
                    "used by this Director clip are missing. Restore those "
                    "assets before loading this recipe."
                )
            if resolved_values:
                params[key] = resolved_values
                uploads[key] = rel_values
                urls[key] = [_asset_url(path, workspace) for path in rel_values]
        else:
            resolved = _resolve_asset(out_dir, state_root, value)
            if not resolved:
                return (
                    f"The archived {key.replace('_', ' ')} asset used by "
                    "this Director clip is missing. Restore that asset "
                    "before loading this recipe."
                )
            rel = _safe_relative_asset(out_dir, resolved)
            if rel:
                params[key] = resolved
                uploads[key] = rel
                urls[key] = _asset_url(rel, workspace)

    # A music-driving guide must be the precise native clip window. Never
    # return the original full song at offset zero as a clip recipe.
    pipeline_type = state.get("pipeline_type") or snapshot.get("pipeline_type") or ""
    has_story_drive = bool(snapshot.get("_director_omni_drive_audio"))
    drive_audio_expected = bool(
        original_params.get("audio_guide")
        or snapshot.get("audio_path")
        or _manifest_asset_path(manifest, "audio_path")
    ) and (pipeline_type != "short_film_story" or has_story_drive)
    if drive_audio_expected:
        # The output's own archived path is the best source when it still
        # exists. Pipeline assets are the recovery fallback for reruns whose
        # temporary generation slice was removed after completion.
        source_audio = _first_resolved_asset(
            out_dir,
            state_root,
            (metadata.get("director_clip_timing") or {}).get(
                "audio_source_path"
            ),
            original_params.get("audio_guide"),
            _manifest_asset_path(manifest, "audio_path"),
            snapshot.get("audio_path"),
        )
        if not source_audio:
            return (
                "The archived soundtrack for this Director clip is missing. "
                "Restore the soundtrack asset before loading this recipe."
            )
        source_name = os.path.basename(source_audio).lower()
        source_is_clip_slice = (
            source_name.startswith("_rerun_audio_")
            or "_director_clip_settings_slices" in source_audio.replace("\\", "/")
        )
        start_sec = _clip_audio_start(
            planned, original_params, metadata, clip_index, fps,
        )
        if start_sec is None and not source_is_clip_slice:
            return (
                "The saved Director state has no exact soundtrack offset for "
                "this clip, so a full-song-at-zero recipe was withheld."
            )
        duration_sec = native_frames / fps
        if source_is_clip_slice:
            slice_path, slice_error = source_audio, None
        else:
            slice_path, slice_error = _materialize_audio_slice(
                out_dir,
                str(metadata.get("director_pipeline_id") or ""),
                clip_index,
                source_audio,
                start_sec,
                duration_sec,
            )
        if not slice_path:
            return (
                "Could not prepare the archived soundtrack slice for this "
                f"Director clip: {slice_error}"
            )
        params["audio_guide"] = slice_path
        params.pop("audio_frame_offset", None)
        params.pop("multi_clip_concat_audio", None)
        params.pop("multi_clip_audio_start_sec", None)
        relative = _safe_relative_asset(out_dir, slice_path)
        if relative:
            uploads["audio_guide"] = relative
            urls["audio_guide"] = _asset_url(relative, workspace)
        prompt_type = str(original_params.get("audio_prompt_type") or "")
        if prompt_type:
            params["audio_prompt_type"] = prompt_type

        # If the saved model used a separate vocal conditioning source, slice
        # that same interval too. Voice identity references remain untouched.
        conditioning_value = original_params.get("audio_conditioning_guide")
        vocal_slice = None
        if conditioning_value:
            existing_conditioning = _resolve_asset(
                out_dir, state_root, conditioning_value,
            )
            if existing_conditioning and os.path.basename(
                existing_conditioning
            ).lower().startswith("_rerun_vocals_"):
                vocal_slice = existing_conditioning
        if conditioning_value and not vocal_slice:
            vocal_source = _first_resolved_asset(
                out_dir,
                state_root,
                _manifest_asset_path(manifest, "audio_vocals_path"),
                snapshot.get("audio_vocals_path"),
            )
            if not vocal_source:
                return (
                    "The archived vocal-conditioning source for this "
                    "Director clip is missing. Restore it before loading "
                    "this recipe."
                )
            vocal_slice, _ = _materialize_audio_slice(
                out_dir,
                str(metadata.get("director_pipeline_id") or "") + "v",
                clip_index,
                vocal_source,
                start_sec,
                duration_sec,
            )
            if not vocal_slice:
                return (
                    "Could not prepare the archived vocal-conditioning "
                    "slice for this Director clip."
                )
        if vocal_slice:
            params["audio_conditioning_guide"] = vocal_slice
            relative = _safe_relative_asset(out_dir, vocal_slice)
            if relative:
                uploads["audio_conditioning_guide"] = relative
                urls["audio_conditioning_guide"] = _asset_url(
                    relative, workspace,
                )

    # Safe workspace-relative names let Studio re-fetch archived nested
    # assets through /api/v1/file without relying on source machine paths.
    metadata["upload_filenames"] = uploads
    if urls:
        metadata["director_clip_asset_urls"] = urls
    else:
        metadata.pop("director_clip_asset_urls", None)
    return None


def _manifest_reference_records(manifest):
    entries = manifest.get("minimax_h3_references") or []
    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        nested = entry.get("path")
        if isinstance(nested, dict):
            records.append({"path": nested.get("path") or nested.get("serve_path")})
    return records


def _manifest_asset_path(manifest, key):
    value = manifest.get(key)
    if isinstance(value, dict):
        return value.get("path") or value.get("serve_path")
    if isinstance(value, str):
        return value
    return None


def _clip_audio_start(planned, original_params, metadata, clip_index, fps):
    timing = metadata.get("director_clip_timing") or {}
    explicit = _nonnegative_float(timing.get("audio_start_sec"))
    if explicit is not None:
        return explicit
    base = _nonnegative_float(timing.get("audio_base_start_sec"))
    offset_frames = _nonnegative_float(original_params.get("audio_frame_offset"))
    own_outputs = original_params.get("per_clip_output_frames")
    own_base = _nonnegative_float(
        original_params.get("multi_clip_audio_start_sec")
    )
    if isinstance(own_outputs, list) and clip_index < len(own_outputs):
        try:
            offset = sum(
                max(0, int(value)) for value in own_outputs[:clip_index]
            )
        except (TypeError, ValueError, OverflowError):
            return None
        return (own_base if own_base is not None else (base or 0.0)) + offset / fps
    if offset_frames is not None and metadata.get(
        "director_clip_settings_provenance"
    ) in {"per_output_effective_params", "director_rerun_request_marker"}:
        return (own_base if own_base is not None else (base or 0.0)) + offset_frames / fps
    if planned.get("music_timing_version") == 1:
        return _nonnegative_float(planned.get("start"))
    return None


def _materialize_audio_slice(out_dir, pipeline_id, clip_index, source, start, duration):
    # Metadata requests can arrive together for the same clip. Serialize cache
    # publication so Windows readers never race a directory create or replace.
    with _AUDIO_SLICE_LOCK:
        return _materialize_audio_slice_locked(
            out_dir, pipeline_id, clip_index, source, start, duration,
        )


def _materialize_audio_slice_locked(out_dir, pipeline_id, clip_index, source, start, duration):
    temp_path = None
    try:
        source = os.path.realpath(source)
        root = os.path.realpath(out_dir)
        if not _inside(root, source) or not os.path.isfile(source):
            return None, "source audio is outside the saved workspace"
        token = re.sub(r"[^A-Za-z0-9_-]", "_", pipeline_id)[:64] or "director"
        stat = os.stat(source)
        signature = "|".join((
            os.path.normcase(source), str(stat.st_size), str(stat.st_mtime_ns),
            f"{start:.6f}", f"{duration:.6f}",
        ))
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
        relative_dir = os.path.join(
            "_director_clip_settings_slices", token,
        )
        slice_dir = os.path.realpath(os.path.join(root, relative_dir))
        if not _inside(root, slice_dir):
            return None, "slice destination is outside the saved workspace"
        os.makedirs(slice_dir, exist_ok=True)
        dst = os.path.join(
            slice_dir, f"clip-{int(clip_index)}-{digest}.wav",
        )
        if _is_valid_wav_slice(dst):
            return dst, None
        # Reuse the pipeline's established ffmpeg helper, but do not reuse its
        # temporary-file lifecycle (these are durable gallery restore assets).
        from services.director_pipeline import _slice_audio_segment
        descriptor, temp_path = tempfile.mkstemp(
            prefix=f".clip-{int(clip_index)}-",
            suffix=".wav",
            dir=slice_dir,
        )
        os.close(descriptor)
        os.unlink(temp_path)
        _slice_audio_segment(source, start, duration, temp_path)
        if not _is_valid_wav_slice(temp_path):
            return None, "the audio slice helper produced an invalid WAV file"
        os.replace(temp_path, dst)
        temp_path = None
        return dst, None
    except Exception as error:
        return None, str(error)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _is_valid_wav_slice(path):
    try:
        if os.path.getsize(path) <= 44:
            return False
        with open(path, "rb") as handle:
            header = handle.read(12)
        return len(header) == 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    except OSError:
        return False


def _resolve_asset(out_dir, state_root, value):
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    root = os.path.realpath(out_dir)
    candidates = []
    if os.path.isabs(value):
        candidates.append(value)
    else:
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or ".." in normalized.split("/"):
            return None
        candidates.extend((
            os.path.join(root, *normalized.split("/")),
            os.path.join(state_root, *normalized.split("/")),
        ))
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if _inside(root, real) and os.path.isfile(real):
            return real
    return None


def _first_resolved_asset(out_dir, state_root, *values):
    for value in values:
        resolved = _resolve_asset(out_dir, state_root, value)
        if resolved:
            return resolved
    return None


def _safe_relative_asset(out_dir, value):
    if not isinstance(value, str) or not value:
        return None
    root = os.path.realpath(out_dir)
    real = os.path.realpath(value)
    if not _inside(root, real) or not os.path.isfile(real):
        return None
    relative = os.path.relpath(real, root).replace(os.sep, "/")
    if relative.startswith("../") or relative == "..":
        return None
    return relative


def _asset_url(relative, workspace):
    query = f"?workspace={quote(str(workspace or ''), safe='')}" if workspace else ""
    return f"/api/v1/file/{quote(relative, safe='/')}{query}"


def _indexed_value(value, index):
    if isinstance(value, list):
        return value[index] if 0 <= index < len(value) else None
    return value


def _inside(root, path):
    try:
        resolved_root = os.path.normcase(os.path.realpath(root))
        resolved_path = os.path.normcase(os.path.realpath(path))
        return os.path.commonpath((resolved_root, resolved_path)) == resolved_root
    except (OSError, ValueError):
        return False


def _nonnegative_int(value):
    if isinstance(value, bool):
        return None
    try:
        numeric = int(value)
        if numeric < 0 or float(value) != numeric:
            return None
        return numeric
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_int(value):
    value = _nonnegative_int(value)
    return value if value is not None and value > 0 else None


def _positive_float(value):
    try:
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _nonnegative_float(value):
    try:
        number = float(value)
        if math.isfinite(number) and number >= 0:
            return number
    except (TypeError, ValueError, OverflowError):
        pass
    return None
