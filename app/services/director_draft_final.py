"""Draft-to-final rendering and per-shot inspection for Director projects.

Final renders are based on the immutable recipe recorded beside each selected
Draft clip. Planning is never re-run. A final child keeps the source Draft
takes as selectable alternatives and adds final-resolution renders as they
complete.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Mapping

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool


_PIPELINE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RESOLUTION_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")


class DirectorDraftFinalError(ValueError):
    """A requested final render cannot safely use its selected Draft take."""


def _fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise DirectorDraftFinalError(
            "A selected Director media file changed or became unavailable while it was verified."
        ) from exc
    return digest.hexdigest()


def _read_source_recipe(
    *,
    out_dir: str,
    pipeline_id: str,
    state: dict,
    state_dir: str,
    clip_index: int,
) -> dict:
    """Load a clip's exact per-output recipe and verify its real seed/media."""
    clips = state.get("clips") or []
    clip = clips[clip_index]
    filename = clip.get("video_filename")
    if clip.get("video_stale"):
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} has changed since its Draft was rendered. "
            "Regenerate the Draft shot before creating a Final."
        )
    if (
        not isinstance(filename, str)
        or not filename
        or os.path.basename(filename) != filename
    ):
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} has no saved per-shot Draft take. "
            "A joined Seamless output cannot supply a verifiable per-shot seed."
        )
    media_path = os.path.join(state_dir, filename)
    if not os.path.isfile(media_path) or os.path.getsize(media_path) <= 0:
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} Draft take is missing or empty. Restore or "
            "regenerate that take before creating a Final."
        )
    media_fingerprint = _fingerprint(media_path)
    approved_filename = clip.get("approved_take_filename")
    approved_fingerprint = clip.get("approved_take_fingerprint")
    if approved_fingerprint and (
        approved_filename != filename
        or approved_fingerprint != media_fingerprint
    ):
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} changed after approval. Review and mark the "
            "current Draft take Good before creating a Final."
        )

    sidecar_path = os.path.join(
        state_dir, os.path.splitext(filename)[0] + ".meta.json",
    )
    if not os.path.isfile(sidecar_path):
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} has no output metadata sidecar. Its actual "
            "seed cannot be verified, so Final rendering was not started."
        )
    try:
        with open(sidecar_path, "r", encoding="utf-8") as source:
            sidecar = json.load(source)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} output metadata could not be read; its "
            "actual seed cannot be verified."
        ) from exc
    if not isinstance(sidecar, dict):
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} output metadata is invalid; its actual "
            "seed cannot be verified."
        )
    sidecar_params = sidecar.get("params")
    if not isinstance(sidecar_params, dict):
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} output metadata has no saved generation "
            "recipe. Final rendering was not started."
        )
    seed = sidecar_params.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} output metadata does not contain its actual "
            "generation seed. Final rendering was not started."
        )
    source_pipeline_id = (
        sidecar.get("director_pipeline_id")
        or sidecar_params.get("_director_pipeline_id")
    )
    if str(source_pipeline_id or "") != pipeline_id:
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} metadata does not belong to the selected "
            "Director Draft. Final rendering was not started."
        )
    actual_index = sidecar.get("director_clip_index")
    if (
        isinstance(actual_index, bool)
        or not isinstance(actual_index, int)
        or actual_index != clip_index
    ):
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} metadata does not identify that exact shot. "
            "Final rendering was not started."
        )

    workspace = str(state.get("workspace") or "default")
    try:
        from services.director_clip_settings import enrich_director_output_metadata

        restored = enrich_director_output_metadata(
            out_dir, workspace, filename, sidecar,
        )
    except Exception as exc:
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} recipe could not be reconstructed safely: {exc}"
        ) from exc
    recipe = restored.get("params") if isinstance(restored, dict) else None
    if not isinstance(recipe, dict):
        detail = (restored or {}).get("director_clip_settings_error")
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} no longer has a complete saved recipe. "
            f"{detail or 'Restore its missing references or audio assets first.'}"
        )
    if recipe.get("seed") != seed:
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} recipe seed does not match the output sidecar. "
            "Final rendering was not started."
        )
    if not str(recipe.get("prompt") or "").strip():
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} saved recipe has no rendered prompt."
        )
    for key in (
        "audio_guide", "audio_conditioning_guide", "voice_reference",
        "image_start", "image_end",
    ):
        value = recipe.get(key)
        if isinstance(value, str) and value and not os.path.isfile(value):
            raise DirectorDraftFinalError(
                f"Shot {clip_index + 1} required {key.replace('_', ' ')} asset "
                "is missing. Restore that file before creating a Final."
            )
    def reference_path(value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("path") or value.get("serve_path") or value.get("filename")
        return value

    for key in (
        "minimax_h3_references", "image_refs", "audio_guide2",
        "audio_guide3", "audio_guide4", "audio_guide5", "audio_guide6",
    ):
        references = recipe.get(key)
        if not isinstance(references, list):
            continue
        for index, reference in enumerate(references, start=1):
            value = reference.get("path") if isinstance(reference, dict) else reference
            value = reference_path(value)
            if not isinstance(value, str) or not os.path.isfile(value):
                raise DirectorDraftFinalError(
                    f"Shot {clip_index + 1} ordered {key.replace('_', ' ')} "
                    f"reference {index} is missing. Restore that reference before "
                    "creating a Final."
                )
    for key in (
        "audio_guide2", "audio_guide3", "audio_guide4", "audio_guide5",
        "audio_guide6", "audio_conditioning_guide",
    ):
        value = recipe.get(key)
        if isinstance(value, str) and value and not os.path.isfile(value):
            raise DirectorDraftFinalError(
                f"Shot {clip_index + 1} required {key.replace('_', ' ')} asset "
                "is missing. Restore that file before creating a Final."
            )

    return {
        "filename": filename,
        "path": media_path,
        "seed": seed,
        "resolution": str(recipe.get("resolution") or ""),
        "fingerprint": media_fingerprint,
        "recipe": copy.deepcopy(recipe),
        "sidecar": copy.deepcopy(sidecar),
    }


def _resolution_for_preset(
    model_def: Mapping[str, Any],
    preset_name: str,
    aspect_ratio: str,
    draft_resolution: str = "",
) -> str:
    """Resolve a published model preset for the project's selected aspect."""
    name = str(preset_name or "").strip().lower()
    if not name or name == "auto":
        raise DirectorDraftFinalError("Choose an explicit Final resolution preset.")
    presets = model_def.get("resolution_presets")
    preset = presets.get(name) if isinstance(presets, Mapping) else None
    values = preset.get("values") if isinstance(preset, Mapping) else None
    if isinstance(values, Mapping):
        ratio = str(aspect_ratio or "auto").strip()
        selected = values.get(ratio) if ratio != "auto" else None
        if isinstance(selected, str) and _RESOLUTION_RE.fullmatch(selected):
            return selected
        # When the project uses Auto aspect, retain the Draft canvas shape
        # wherever the model publishes a matching numeric preset canvas.
        draft_match = _RESOLUTION_RE.fullmatch(str(draft_resolution or ""))
        if draft_match:
            draft_width, draft_height = (int(value) for value in draft_match.groups())
            target_ratio = draft_width / max(1, draft_height)
            candidates = [
                value for value in values.values()
                if isinstance(value, str) and _RESOLUTION_RE.fullmatch(value)
            ]
            if candidates:
                return min(
                    candidates,
                    key=lambda value: abs(
                        int(value.split("x", 1)[0])
                        / max(1, int(value.split("x", 1)[1]))
                        - target_ratio
                    ),
                )
        if isinstance(selected, str) and selected.startswith("auto_"):
            return selected
        automatic = values.get("auto")
        if isinstance(automatic, str) and automatic.startswith("auto_"):
            return automatic
        candidates = [
            value for value in values.values()
            if isinstance(value, str) and _RESOLUTION_RE.fullmatch(value)
        ]
        if candidates:
            return candidates[0]
    # Some older model definitions expose only a fixed resolution catalog.
    # Permit an exact catalog entry as the explicit selection.
    for item in model_def.get("resolutions") or []:
        value = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
        if isinstance(value, str) and value == preset_name and _RESOLUTION_RE.fullmatch(value):
            return value
    raise DirectorDraftFinalError(
        f"{preset_name!r} is not a supported resolution preset for this model."
    )


def _safe_relative_to(root: str, path: str) -> str:
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    try:
        if os.path.commonpath([root_real, path_real]) != root_real:
            raise ValueError("review artifacts must remain inside the output folder")
    except ValueError as exc:
        raise ValueError("review artifacts must remain inside the output folder") from exc
    return os.path.relpath(path_real, root_real).replace("\\", "/")


def _reference_evidence(state: Mapping[str, Any], recipe: Mapping[str, Any]) -> dict:
    """Report attached location/voice assets without claiming media matching."""
    locations = []
    raw_locations = state.get("location_ref_paths") or []
    labels = state.get("location_ref_labels") or []
    for index, value in enumerate(raw_locations):
        path = str(value or "")
        if not path:
            continue
        label = str(labels[index] if index < len(labels) else "").strip()
        locations.append({
            "label": label or f"Location reference {index + 1}",
            "status": "available" if os.path.isfile(path) else "missing",
            "filename": os.path.basename(path),
        })
    voice = str(recipe.get("voice_reference") or "").strip()
    return {
        "location_references": locations,
        "voice_reference": {
            "status": "not_attached" if not voice else (
                "available" if os.path.isfile(voice) else "missing"
            ),
            "filename": os.path.basename(voice) if voice else None,
        },
        "voice_verification": {
            "status": "not_verified",
            "message": (
                "Audio presence and preview do not verify speaker identity, "
                "voice ownership, or dialogue-to-picture sync."
            ),
        },
    }


def _relative_review_packet(packet: Mapping[str, Any], output_root: str) -> dict:
    """Keep inspection responses workspace-relative and safe for file URLs."""

    def rewrite(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            result = {}
            for child_key, child in value.items():
                rewritten = rewrite(child, child_key)
                if rewritten is not None:
                    result[child_key] = rewritten
            return result
        if isinstance(value, list):
            return [item for child in value if (item := rewrite(child, key)) is not None]
        if key in {"path", "partial_path", "paths"} and isinstance(value, str):
            if os.path.isfile(value):
                try:
                    return _safe_relative_to(output_root, value)
                except ValueError:
                    return None
            return None
        return copy.deepcopy(value)

    return rewrite(packet)


def inspect_pipeline_clip(out_dir: str, pipeline_id: str, clip_index: int) -> dict:
    """Inspect the currently selected rendered shot with CPU-only media tools."""
    if not _PIPELINE_ID_RE.fullmatch(str(pipeline_id or "")):
        raise DirectorDraftFinalError("The Director pipeline id is invalid.")
    from services import director_pipeline as pipeline

    state = pipeline.load_pipeline_state(out_dir, pipeline_id)
    if not state:
        raise DirectorDraftFinalError(f"Director pipeline {pipeline_id} was not found.")
    clips = state.get("clips") or []
    if type(clip_index) is not int or clip_index < 0 or clip_index >= len(clips):
        raise DirectorDraftFinalError("The selected Director shot does not exist.")
    clip = clips[clip_index]
    if clip.get("video_stale"):
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} is stale. Regenerate it before inspection."
        )
    filename = clip.get("video_filename")
    if not isinstance(filename, str) or os.path.basename(filename) != filename:
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} has no saved per-shot video to inspect."
        )
    state_file = pipeline._find_pipeline_file(out_dir, pipeline_id)
    if not state_file:
        raise DirectorDraftFinalError(f"Director pipeline {pipeline_id} was not found.")
    state_dir = os.path.dirname(state_file)
    media_path = os.path.join(state_dir, filename)
    if not os.path.isfile(media_path) or os.path.getsize(media_path) <= 0:
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} selected take is missing or empty."
        )
    expected_duration = None
    planned = clip.get("planned_clip") or {}
    try:
        expected_duration = float(planned.get("duration_sec"))
    except (TypeError, ValueError):
        pass

    try:
        from services.production_review import inspect_take

        review_dir = os.path.join(
            state_dir, "_director_reviews", pipeline_id, f"shot-{clip_index + 1}",
        )
        packet = inspect_take(
            media_path, review_dir, expected_duration=expected_duration,
        )
    except Exception as exc:
        raise DirectorDraftFinalError(str(exc)) from exc

    artifacts = packet.get("artifacts") or {}
    path_map: dict[str, Any] = {}
    for key in ("contact_sheet", "audio_preview", "packet"):
        item = artifacts.get(key)
        path = item.get("path") if isinstance(item, dict) else None
        if path and os.path.isfile(path):
            path_map[key] = _safe_relative_to(state_dir, path)
    stills = artifacts.get("stills") or {}
    path_map["stills"] = [
        _safe_relative_to(state_dir, path)
        for path in stills.get("paths", [])
        if isinstance(path, str) and os.path.isfile(path)
    ]

    sidecar_params: dict = {}
    sidecar_path = os.path.join(
        state_dir, os.path.splitext(filename)[0] + ".meta.json",
    )
    try:
        with open(sidecar_path, "r", encoding="utf-8") as source:
            metadata = json.load(source)
        if isinstance(metadata, dict) and isinstance(metadata.get("params"), dict):
            sidecar_params = metadata["params"]
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    response = {
        "pipeline_id": pipeline_id,
        "clip_index": clip_index,
        "filename": filename,
        "fingerprint": _fingerprint(media_path),
        "review": _relative_review_packet(packet, state_dir),
        "artifact_filenames": path_map,
        "reference_evidence": _reference_evidence(state, sidecar_params),
        "workspace": state.get("workspace") or "default",
    }
    return response


def _resolve_take(state: dict, clip_index: int, take_id: str) -> tuple[dict, dict]:
    clips = state.get("clips") or []
    if type(clip_index) is not int or clip_index < 0 or clip_index >= len(clips):
        raise DirectorDraftFinalError("The selected Director shot does not exist.")
    clip = clips[clip_index]
    takes = clip.get("takes") or []
    take = next(
        (item for item in takes if isinstance(item, dict) and item.get("id") == take_id),
        None,
    )
    if not take:
        raise DirectorDraftFinalError("That take is no longer available for this shot.")
    if clip.get("video_stale"):
        raise DirectorDraftFinalError(
            f"Shot {clip_index + 1} settings changed after rendering. Regenerate "
            "the shot before selecting either take."
        )
    filename = take.get("filename")
    if not isinstance(filename, str) or os.path.basename(filename) != filename:
        raise DirectorDraftFinalError("The selected take has an invalid media filename.")
    return clip, take


def select_pipeline_clip_take(
    out_dir: str, pipeline_id: str, clip_index: int, take_id: str,
) -> dict:
    """Select Draft or Final for a child shot, checking file and lineage."""
    from services import director_pipeline as pipeline

    if not _PIPELINE_ID_RE.fullmatch(str(pipeline_id or "")):
        raise DirectorDraftFinalError("The Director pipeline id is invalid.")
    if not pipeline._claim_pipeline_operation(pipeline_id):
        raise pipeline.PipelineBusyError("Pipeline is still active; try again shortly.")
    try:
        state = pipeline.load_pipeline_state(out_dir, pipeline_id)
        if not state or not state.get("draft_final"):
            raise DirectorDraftFinalError("This pipeline has no Draft/Final takes.")
        clip, take = _resolve_take(state, clip_index, take_id)
        source_pipeline_id = take.get("pipeline_id")
        if take.get("stage") == "draft":
            if not _PIPELINE_ID_RE.fullmatch(str(source_pipeline_id or "")):
                raise DirectorDraftFinalError(
                    "The saved Draft take has no valid source pipeline lineage."
                )
            source_state = pipeline.load_pipeline_state(out_dir, source_pipeline_id)
            if not source_state:
                raise DirectorDraftFinalError(
                    "The source Draft pipeline was deleted. This Draft take can no "
                    "longer be selected; choose the Final take instead."
                )
            source_clips = source_state.get("clips") or []
            source_clip = source_clips[clip_index] if clip_index < len(source_clips) else None
            if (
                not isinstance(source_clip, dict)
                or source_clip.get("video_stale")
                or source_clip.get("video_filename") != take["filename"]
            ):
                raise DirectorDraftFinalError(
                    "The source Draft shot changed after Final rendering. Inspect or "
                    "render the updated Draft before selecting it."
                )
            if source_clip.get("tag") != "good":
                raise DirectorDraftFinalError(
                    "This Draft shot is no longer approved. Review the current "
                    "source take and mark it Good before selecting it here."
                )
            if take.get("fingerprint") and source_clip.get("approved_take_fingerprint") and (
                source_clip.get("approved_take_filename") != take["filename"]
                or source_clip.get("approved_take_fingerprint") != take["fingerprint"]
            ):
                raise DirectorDraftFinalError(
                    "This Draft take changed after approval. Review and approve the "
                    "current source take before selecting it here."
                )
        target = pipeline._find_pipeline_file(out_dir, pipeline_id)
        if not target:
            raise DirectorDraftFinalError("The Director pipeline state file is missing.")
        state_dir = os.path.dirname(target)
        media_path = os.path.join(state_dir, take["filename"])
        if not os.path.isfile(media_path) or os.path.getsize(media_path) <= 0:
            raise DirectorDraftFinalError(
                f"The selected {take.get('label', 'take')} file is missing or empty."
            )
        expected_fingerprint = take.get("fingerprint")
        if expected_fingerprint and _fingerprint(media_path) != expected_fingerprint:
            raise DirectorDraftFinalError(
                f"The selected {take.get('label', 'take')} media changed after "
                "inspection. Inspect the current file before selecting it."
            )
        clip["video_filename"] = take["filename"]
        clip["selected_take_id"] = take_id
        clip["video_stale"] = False
        updated = pipeline._update_saved_pipeline(
            out_dir,
            pipeline_id,
            lambda saved: saved.get("clips", [])[clip_index].update({
                "video_filename": take["filename"],
                "selected_take_id": take_id,
                "video_stale": False,
            }),
        )
        if not updated:
            raise DirectorDraftFinalError("The selected take could not be saved to the pipeline.")
        with pipeline._pipeline_lock:
            live = pipeline._pipelines.get(pipeline_id)
            if live is not None:
                clip_count = len(state.get("clips") or [])
                slots = live.setdefault("_clip_video_files", [None] * clip_count)
                if len(slots) < clip_count:
                    slots.extend([None] * (clip_count - len(slots)))
                slots[clip_index] = take["filename"]
                selected = live.setdefault("_clip_selected_take_ids", {})
                selected[str(clip_index)] = take_id
        return {
            "pipeline_id": pipeline_id,
            "clip_index": clip_index,
            "selected_take_id": take_id,
            "filename": take["filename"],
        }
    finally:
        pipeline._release_pipeline_operation(pipeline_id)


def _validate_source_pipeline(state: dict, pipeline_id: str) -> None:
    if (state.get("draft_final") or {}).get("stage") == "final":
        raise DirectorDraftFinalError(
            "Create a Final from an approved Draft pipeline, not another Final revision."
        )
    if state.get("status") != "completed":
        raise DirectorDraftFinalError(
            "Create a Final from a completed Draft pipeline."
        )
    if state.get("seamless", True):
        raise DirectorDraftFinalError(
            "This Draft was rendered as one Seamless video, so it has no "
            "verifiable per-shot seeds. Turn off Seamless and render a Draft "
            "before creating per-shot Finals."
        )
    if not state.get("clips"):
        raise DirectorDraftFinalError("The selected Draft has no saved shots.")
    unapproved = [
        index + 1
        for index, clip in enumerate(state.get("clips") or [])
        if not isinstance(clip, dict) or clip.get("tag") != "good"
    ]
    if unapproved:
        preview = ", ".join(str(index) for index in unapproved[:8])
        extra = "…" if len(unapproved) > 8 else ""
        raise DirectorDraftFinalError(
            f"Approve every Draft shot with Good before creating a Final. "
            f"Shots needing approval: {preview}{extra}."
        )
    if any(
        repair.get("status") in {"queued", "running", "cancelling"}
        for repair in [state.get("repair") or {}]
    ):
        raise DirectorDraftFinalError("Wait for the Draft repair to finish before finalizing.")


def _make_final_child(
    out_dir: str, pipeline_id: str, resolution_preset: str,
) -> tuple[str, str]:
    """Validate all immutable shot recipes and register a child worker."""
    from services import director_pipeline as pipeline

    if not _PIPELINE_ID_RE.fullmatch(str(pipeline_id or "")):
        raise DirectorDraftFinalError("The Director pipeline id is invalid.")
    if not pipeline._claim_pipeline_operation(pipeline_id):
        raise pipeline.PipelineBusyError("Pipeline is still active; try again shortly.")
    source_operation_transferred = False
    child_id = uuid.uuid4().hex[:8]
    try:
        source_state = pipeline.load_pipeline_state(out_dir, pipeline_id)
        if not source_state:
            raise DirectorDraftFinalError(f"Director pipeline {pipeline_id} was not found.")
        _validate_source_pipeline(source_state, pipeline_id)
        source_file = pipeline._find_pipeline_file(out_dir, pipeline_id)
        if not source_file:
            raise DirectorDraftFinalError("The saved Draft state file is missing.")
        source_dir = os.path.dirname(source_file)

        source_items = [
            _read_source_recipe(
                out_dir=out_dir,
                pipeline_id=pipeline_id,
                state=source_state,
                state_dir=source_dir,
                clip_index=index,
            )
            for index in range(len(source_state["clips"]))
        ]

        model_id = str(source_state.get("video_model") or "").strip()
        model_def = pipeline._wgp.get_model_def(model_id) or {}
        preset = str(resolution_preset or "").strip().lower()
        resolution_value = _resolution_for_preset(
            model_def,
            preset,
            str(source_state.get("director_aspect_ratio") or "auto"),
            str((source_items[0].get("recipe") or {}).get("resolution") or ""),
        )
        params = copy.deepcopy(source_state.get("_params_snapshot") or {})
        params.update({
            "pipeline_type": source_state.get("pipeline_type") or "music_video",
            "seamless": False,
            "image_model": source_state.get("image_model") or params.get("image_model"),
            "video_model": model_id,
            "workspace": source_state.get("workspace") or "default",
            "_director_project_id": source_state.get("project_id") or pipeline_id,
            "_director_parent_pipeline_id": pipeline_id,
            "_director_render_stage": "final",
            "_director_final_resolution_preset": preset,
            "_director_shot_image_policy": source_state.get("shot_image_policy")
                or pipeline._saved_pipeline_shot_image_policy(source_state),
        })
        video_params = copy.deepcopy(source_state.get("video_params") or params.get("video_params") or {})
        video_params["resolution"] = resolution_value
        params["video_params"] = video_params
        params["director_resolution_preset"] = preset
        params["director_aspect_ratio"] = source_state.get("director_aspect_ratio") or "auto"
        params["_director_video_execution_profile"] = None
        profile = pipeline._create_director_video_execution_profile(params, model_def=model_def)
        final_resolution = str(
            profile.get("normalized_resolution")
            or (params.get("video_params") or {}).get("resolution")
            or resolution_value
        )
        if final_resolution:
            params.setdefault("video_params", {})["resolution"] = final_resolution

        recipes = []
        for index, item in enumerate(source_items):
            recipe = item["recipe"]
            try:
                native_frames = int(recipe.get("video_length") or 0)
            except (TypeError, ValueError):
                native_frames = 0
            if native_frames <= 0:
                raise DirectorDraftFinalError(
                    f"Shot {index + 1} recipe has no native frame length."
                )
            if profile.get("effective_max_frames") is not None:
                pipeline.validate_director_execution_frames(
                    profile, native_frames, label=f"Final shot {index + 1}",
                )
            recipe["resolution"] = final_resolution
            recipe["seed"] = item["seed"]
            recipe["_director_video_execution_profile"] = copy.deepcopy(profile)
            recipe["_director_pipeline_id"] = child_id
            recipe["_director_clip_index"] = index
            recipe["_director_video_strategy"] = profile.get("video_strategy")
            recipe["_director_shot_image_policy"] = params[
                "_director_shot_image_policy"
            ]
            recipe["_director_detached_operation"] = False
            recipe["_director_render_stage"] = "final"
            recipes.append(item)

        child_state = copy.deepcopy(source_state)
        child_state.update({
            "version": pipeline.PIPELINE_STATE_VERSION,
            "pipeline_id": child_id,
            "project_id": params["_director_project_id"],
            "parent_pipeline_id": pipeline_id,
            "status": "running",
            "phase": "final_render",
            "created_at": time.time(),
            "completed_at": None,
            "progress": {
                "current": 0,
                "total": len(recipes),
                "message": f"Preparing Final at {final_resolution}...",
                "step": 0,
                "total_steps": 0,
            },
            "error": None,
            "output_files": [],
            "director_resolution_preset": preset,
            "video_params": copy.deepcopy(params["video_params"]),
            "video_execution_profile": copy.deepcopy(profile),
            "render_stage": "final",
            "draft_final": {
                "stage": "final",
                "source_pipeline_id": pipeline_id,
                "source_resolution_preset": source_state.get("director_resolution_preset"),
                "final_resolution_preset": preset,
                "final_resolution": final_resolution,
                "source_take_count": len(recipes),
            },
            "_params_snapshot": params,
        })
        draft_takes = []
        selected_ids = {}
        for index, (clip, item) in enumerate(zip(child_state["clips"], recipes)):
            take_id = f"draft:{pipeline_id}:{index}"
            take = {
                "id": take_id,
                "label": "Draft",
                "stage": "draft",
                "pipeline_id": pipeline_id,
                "filename": item["filename"],
                "resolution": item["resolution"],
                "seed": item["seed"],
                "fingerprint": item["fingerprint"],
                "approved": True,
            }
            clip["takes"] = [take]
            clip["selected_take_id"] = take_id
            clip["video_filename"] = item["filename"]
            clip["video_stale"] = False
            draft_takes.append([take])
            selected_ids[str(index)] = take_id

        # Final children reuse the source project's images and Draft media.
        # Record those shared names so deleting the child never deletes files
        # owned by its parent Draft.
        shared_source_files = {
            filename
            for clip in child_state["clips"]
            for filename in [
                clip.get("start_image_filename"),
                *(clip.get("keyframe_filenames") or []),
                clip.get("video_filename"),
            ]
            if isinstance(filename, str) and os.path.basename(filename) == filename
        }
        generated_anchor = child_state.get("generated_reference_image_filename")
        if isinstance(generated_anchor, str) and os.path.basename(generated_anchor) == generated_anchor:
            shared_source_files.add(generated_anchor)
        child_state["draft_final"]["shared_source_files"] = sorted(shared_source_files)
        child_state["draft_final"]["source_approved"] = True
        for clip in child_state["clips"]:
            clip["tag"] = None
            clip.pop("approved_take_filename", None)
            clip.pop("approved_take_fingerprint", None)

        target_file = os.path.join(source_dir, f"_director_pipeline_{child_id}.json")
        with pipeline._pipeline_file_lock:
            if os.path.exists(target_file):
                raise DirectorDraftFinalError("Could not reserve a new Final pipeline id.")
            pipeline._write_pipeline_json_unlocked(target_file, child_state)

        clip_plans = []
        for clip in child_state["clips"]:
            plan = {
                "image_prompt": clip.get("image_prompt", ""),
                "video_prompt": clip.get("video_prompt", ""),
                "visual_changes": clip.get("visual_changes", []) or [],
                "image_source": clip.get("image_source", "original"),
                "keyframe_prompts": clip.get("keyframe_prompts", []) or [],
                "window_prompts": clip.get("window_prompts", []) or [],
                "window_count": clip.get("window_count", 1),
            }
            for key, value in clip.items():
                if key.startswith("_director_"):
                    plan[key] = copy.deepcopy(value)
            clip_plans.append(plan)
        runtime = {
            "id": child_id,
            "status": "running",
            "phase": "final_render",
            "auto_mode": True,
            "progress": copy.deepcopy(child_state["progress"]),
            "clip_plans": clip_plans,
            "_planned_clips": [
                copy.deepcopy(clip.get("planned_clip") or {})
                for clip in child_state["clips"]
            ],
            "clip_images": [clip.get("start_image_filename") or "" for clip in child_state["clips"]],
            "_clip_keyframes": [clip.get("keyframe_filenames") or [] for clip in child_state["clips"]],
            "_clip_video_files": [item["filename"] for item in recipes],
            "_clip_takes": draft_takes,
            "_clip_selected_take_ids": selected_ids,
            "_clip_tags": [clip.get("tag") for clip in child_state["clips"]],
            "_clip_approved_take_filenames": [None] * len(child_state["clips"]),
            "_clip_approved_take_fingerprints": [None] * len(child_state["clips"]),
            "_director_final_recipes": [item["recipe"] for item in recipes],
            "_director_final_source_recipes": recipes,
            "_director_final_resolution": final_resolution,
            "_director_source_pipeline_id": pipeline_id,
            "_llm_log": child_state.get("llm_log"),
            "_llm_drafts": child_state.get("llm_drafts") or [],
            "_planning_checkpoint": None,
            "output_files": [],
            "created_at": child_state["created_at"],
            "_completed_at": None,
            "params": params,
            "pause_reason": None,
            "workspace": child_state.get("workspace"),
            "out_dir": source_dir,
            "draft_final": child_state["draft_final"],
            "error": None,
            "llm_streaming": False,
        }
        with pipeline._pipeline_lock:
            if child_id in pipeline._pipelines:
                try:
                    os.remove(target_file)
                except OSError:
                    pass
                raise DirectorDraftFinalError("Could not reserve a new Final pipeline id.")
        try:
            worker = threading.Thread(
                target=_render_final_worker,
                args=(child_id, out_dir),
                name=f"director-final-{child_id}",
                daemon=False,
            )
        except BaseException:
            try:
                os.remove(target_file)
            except OSError:
                pass
            raise
        with pipeline._pipeline_lock:
            pipeline._pipelines[child_id] = runtime
            pipeline._pipeline_threads[child_id] = worker
        source_operation_transferred = True
        try:
            worker.start()
        except BaseException:
            source_operation_transferred = False
            with pipeline._pipeline_lock:
                pipeline._pipeline_threads.pop(child_id, None)
                pipeline._pipelines.pop(child_id, None)
            try:
                os.remove(target_file)
            except OSError:
                pass
            raise
        return child_id, final_resolution
    finally:
        if not source_operation_transferred:
            pipeline._release_pipeline_operation(pipeline_id)


def _render_final_worker(child_id: str, base_out_dir: str) -> None:
    """Render exact per-shot recipes at the selected Final resolution."""
    from services import director_pipeline as pipeline

    worker = threading.current_thread()
    source_pipeline_id = None
    try:
        runtime = pipeline.get_pipeline(child_id) or {}
        source_pipeline_id = runtime.get("_director_source_pipeline_id")
        recipes = runtime.get("_director_final_recipes") or []
        source_recipes = runtime.get("_director_final_source_recipes") or []
        total = len(recipes)
        for index, recipe in enumerate(recipes):
            current = pipeline.get_pipeline(child_id) or {}
            if current.get("status") == "cancelled":
                return
            source = source_recipes[index]
            if not os.path.isfile(source["path"]) or _fingerprint(source["path"]) != source["fingerprint"]:
                raise DirectorDraftFinalError(
                    f"Shot {index + 1} Draft media changed before rendering. "
                    "Inspect or restore that take, then create the Final again."
                )
            current_recipe = copy.deepcopy(recipe)
            current_recipe["_director_clip_index"] = index
            output_files = pipeline._submit_and_wait(
                current_recipe,
                timeout_s=7200,
                workspace=runtime.get("workspace"),
                out_dir=runtime.get("out_dir"),
            )
            filename = output_files[-1] if output_files else ""
            if not filename or not isinstance(filename, str) or os.path.basename(filename) != filename:
                raise RuntimeError(
                    f"Final render for shot {index + 1} completed without a recorded output."
                )
            final_path = os.path.join(runtime.get("out_dir") or base_out_dir, filename)
            if not os.path.isfile(final_path) or os.path.getsize(final_path) <= 0:
                raise RuntimeError(
                    f"Final render for shot {index + 1} did not produce a valid media file."
                )
            final_take = {
                "id": f"final:{child_id}:{index}",
                "label": "Final",
                "stage": "final",
                "pipeline_id": child_id,
                "filename": filename,
                "resolution": runtime.get("_director_final_resolution"),
                "seed": source["seed"],
                "fingerprint": _fingerprint(final_path),
            }
            with pipeline._pipeline_lock:
                live = pipeline._pipelines.get(child_id)
                if live is None:
                    raise RuntimeError("The Final pipeline was removed while rendering.")
                takes = live.setdefault("_clip_takes", [[] for _ in range(total)])
                while len(takes) <= index:
                    takes.append([])
                takes[index] = [
                    item for item in takes[index]
                    if not (isinstance(item, dict) and item.get("stage") == "final")
                ] + [final_take]
                live.setdefault("_clip_video_files", [None] * total)[index] = filename
                live.setdefault("_clip_selected_take_ids", {})[str(index)] = final_take["id"]
                live.setdefault("output_files", [])
                if filename not in live["output_files"]:
                    live["output_files"].append(filename)
                live["progress"] = {
                    "current": index + 1,
                    "total": total,
                    "message": f"Final rendered shot {index + 1}/{total}",
                    "step": 0,
                    "total_steps": 0,
                }
            pipeline._save_pipeline_state(child_id)

        current = pipeline.get_pipeline(child_id) or {}
        if current.get("status") == "cancelled":
            return
        pipeline._update_pipeline(
            child_id,
            status="completed",
            phase="completed",
            _completed_at=time.time(),
            progress={
                "current": total,
                "total": total,
                "message": "Final rendering completed",
                "step": 0,
                "total_steps": 0,
            },
        )
    except Exception as exc:
        current = pipeline.get_pipeline(child_id) or {}
        if current.get("status") != "cancelled":
            pipeline._update_pipeline(
                child_id,
                status="failed",
                phase="failed",
                error=str(exc),
                _completed_at=time.time(),
                progress={
                    "current": len((current.get("output_files") or [])),
                    "total": len(current.get("_director_final_recipes") or []),
                    "message": "Final rendering failed",
                    "step": 0,
                    "total_steps": 0,
                },
            )
    finally:
        try:
            pipeline._save_pipeline_state(child_id)
        finally:
            with pipeline._pipeline_lock:
                if pipeline._pipeline_threads.get(child_id) is worker:
                    pipeline._pipeline_threads.pop(child_id, None)
            if source_pipeline_id:
                pipeline._release_pipeline_operation(source_pipeline_id)


def finalize_pipeline(out_dir: str, pipeline_id: str, resolution_preset: str) -> dict:
    """Create a Final child without invoking the planner or changing prompts."""
    child_id, resolution = _make_final_child(
        out_dir, pipeline_id, resolution_preset,
    )
    return {
        "pipeline_id": child_id,
        "source_pipeline_id": pipeline_id,
        "status": "running",
        "render_stage": "final",
        "resolution": resolution,
    }


def create_director_draft_final_router(
    *,
    authorize_pipeline: Callable[[str], Any],
    initialize_pipeline: Callable[[], Any],
    get_output_dir: Callable[[], str],
) -> APIRouter:
    """Build the additive Director Draft/Final + shot-inspection routes.

    `authorize_pipeline` should be launch.py's existing
    `_agent_require_director_pipeline`; it owns the same pipeline authorization
    check as every other saved Director action.
    """
    router = APIRouter()

    @router.post("/api/v1/director/pipelines/{pid}/finalize")
    async def finalize_director_pipeline_route(pid: str, request: Request):
        authorize_pipeline(pid)
        initialize_pipeline()
        try:
            body = await request.json()
            preset = body.get("resolution_preset") if isinstance(body, dict) else None
            if not isinstance(preset, str) or not preset.strip():
                raise DirectorDraftFinalError("Choose an explicit Final resolution preset.")
            return await run_in_threadpool(
                finalize_pipeline, get_output_dir(), pid, preset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            from services.director_pipeline import PipelineBusyError

            if isinstance(exc, PipelineBusyError):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    @router.post("/api/v1/director/pipelines/{pid}/clips/{clip_index}/inspect")
    async def inspect_director_pipeline_clip(pid: str, clip_index: int):
        authorize_pipeline(pid)
        initialize_pipeline()
        try:
            return await run_in_threadpool(
                inspect_pipeline_clip, get_output_dir(), pid, clip_index,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.put("/api/v1/director/pipelines/{pid}/clips/{clip_index}/take")
    async def select_director_pipeline_take(
        pid: str, clip_index: int, request: Request,
    ):
        authorize_pipeline(pid)
        initialize_pipeline()
        try:
            body = await request.json()
            take_id = body.get("take_id") if isinstance(body, dict) else None
            if not isinstance(take_id, str) or not take_id.strip():
                raise DirectorDraftFinalError("Choose a valid Draft or Final take.")
            return await run_in_threadpool(
                select_pipeline_clip_take,
                get_output_dir(), pid, clip_index, take_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            from services.director_pipeline import PipelineBusyError

            if isinstance(exc, PipelineBusyError):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    return router
