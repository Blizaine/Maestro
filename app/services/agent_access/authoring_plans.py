"""Pure helpers for externally authored Studio and Director plans.

These helpers validate a plan against a server-owned creation context and
translate it into the request fields consumed by Maestro's existing native
generation paths.  They deliberately do not call an LLM or adapt authored
timelines behind the caller's back.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any


class AuthoringPlanError(ValueError):
    """An authored plan is missing required fields or no longer fits context."""


_DIRECTOR_WORKFLOWS = {"director.story", "director.music_video"}
_SINGLE_PROMPT_WORKFLOWS = {
    "image.generate",
    "image.edit",
    "audio.music",
    "audio.speech",
    "audio.sfx",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuthoringPlanError(f"{label} must be an object.")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuthoringPlanError(f"{label} is required and must be a non-empty string.")
    return value


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise AuthoringPlanError(f"{label} must be an integer of at least {minimum}.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AuthoringPlanError(
            f"{label} must be an integer of at least {minimum}."
        ) from error
    if result < minimum or str(value).strip() not in {str(result), f"{result}.0"}:
        raise AuthoringPlanError(f"{label} must be an integer of at least {minimum}.")
    return result


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AuthoringPlanError(f"{label} must be a number greater than {minimum}.") from error
    if not math.isfinite(result) or result <= minimum:
        raise AuthoringPlanError(f"{label} must be a number greater than {minimum}.")
    return result


def _context_parts(context: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any]]:
    context_id = _required_text(context.get("context_id"), "context.context_id")
    workflow = _required_text(context.get("workflow"), "context.workflow")
    params = context.get("params") or {}
    model_def = context.get("model_def") or {}
    return (
        context_id,
        workflow,
        _mapping(params, "context.params"),
        _mapping(model_def, "context.model_def"),
    )


def _reference_ids(context: Mapping[str, Any]) -> list[str]:
    refs = context.get("references") or []
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        raise AuthoringPlanError("context.references must be an array.")
    identifiers: list[str] = []
    for index, item in enumerate(refs, start=1):
        row = _mapping(item, f"context.references[{index}]")
        identifier = _required_text(row.get("id"), f"context.references[{index}].id")
        if identifier in identifiers:
            raise AuthoringPlanError(f"context.references contains duplicate ID {identifier!r}.")
        identifiers.append(identifier)
    return identifiers


def _ordered_reference_ids(value: Any, expected: list[str], label: str) -> list[str]:
    if value is None and not expected:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AuthoringPlanError(f"{label} must list the reference IDs from the current context.")
    if len(set(value)) != len(value):
        raise AuthoringPlanError(f"{label} contains duplicate reference IDs.")
    if value != expected:
        raise AuthoringPlanError(
            f"{label} does not match the current context references. "
            "Refresh the creation context and author the plan against its current reference IDs."
        )
    return list(value)


def _continuity_mode(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized not in {"single", "continuous", "independent"}:
        raise AuthoringPlanError(
            "The creation context has no valid continuity mode; refresh it before authoring."
        )
    return normalized


def _window_rows(
    geometry: Sequence[Mapping[str, Any]],
    *,
    references: list[str],
    continuity: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_id: str | None = None
    for index, raw in enumerate(geometry, start=1):
        row = dict(raw)
        identifier = f"window-{index:03d}"
        rows.append({
            "id": identifier,
            "index": index,
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "start_seconds": float(row.get("start_seconds", 0.0)),
            "end_seconds": float(row.get("end_seconds", 0.0)),
            "reference_ids": list(references),
            "previous_window_id": previous_id if continuity == "continuous" else None,
        })
        previous_id = identifier
    return rows


def _ltx_pass_geometry(
    total_frames: int,
    window_frames: int,
    *,
    overlap_frames: int,
    discard_frames: int,
    fps: float,
) -> list[dict[str, Any]]:
    """Describe pass spans using the same stride/count as the native LTX helper."""

    from services.ltx_window_planner import compute_ltx_window_count

    count = compute_ltx_window_count(
        total_frames,
        window_frames,
        overlap_frames=overlap_frames,
        discard_frames=discard_frames,
    )
    stride = window_frames - overlap_frames - discard_frames
    if count > 1 and stride <= 0:
        raise AuthoringPlanError(
            "LTX Window Length must exceed overlap plus discarded tail frames."
        )
    rows = []
    for index in range(count):
        start = index * max(1, stride)
        end = min(total_frames, start + window_frames)
        rows.append({
            "index": index + 1,
            "start_frame": start,
            "end_frame": end,
            "start_seconds": round(start / fps, 3),
            "end_seconds": round(end / fps, 3),
            "pass_start_frame": start,
            "pass_end_frame": end,
            "overlap_before_frames": overlap_frames if index else 0,
            "discarded_tail_frames": discard_frames if index < count - 1 else 0,
        })
    return rows


def _director_profile(
    context: Mapping[str, Any],
    params: Mapping[str, Any],
    model_def: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = context.get("execution_profile") or params.get("_director_video_execution_profile") or {}
    if not isinstance(supplied, Mapping):
        supplied = {}
    fps = _number(
        context.get("fps") or supplied.get("fps") or model_def.get("fps") or 24,
        "Director frame rate",
    )
    minimum = _integer(
        supplied.get("frames_minimum", model_def.get("frames_minimum", 1)),
        "Director minimum shot frames",
    )
    step = _integer(
        supplied.get("frame_step", model_def.get("frames_steps", 1)),
        "Director shot frame step",
    )
    maximum_value = supplied.get("effective_max_frames", model_def.get("frames_maximum"))
    maximum = _integer(maximum_value, "Director maximum shot frames")
    if maximum < minimum:
        raise AuthoringPlanError(
            "Director execution profile has a maximum shorter than its minimum. Refresh the context."
        )
    target_frames = context.get("target_duration_frames")
    if target_frames is None:
        if str(context.get("workflow") or "") == "director.music_video":
            target_seconds = (
                context.get("source_duration_seconds")
                or params.get("source_duration_seconds")
                or context.get("target_duration_seconds")
            )
        else:
            target_seconds = (
                context.get("target_duration_seconds")
                or params.get("target_duration_seconds")
                or params.get("duration_seconds")
                or params.get("source_duration_seconds")
            )
        if target_seconds is not None:
            target_frames = max(1, round(_number(target_seconds, "Director target duration") * fps))
    if target_frames is None and params.get("video_length") is not None:
        target_frames = _integer(params.get("video_length"), "Director target duration frames")
    if target_frames is not None:
        target_frames = _integer(target_frames, "Director target duration frames")
    return {
        "fps": fps,
        "minimum_frames": minimum,
        "maximum_frames": maximum,
        "frame_step": step,
        "target_total_frames": target_frames,
    }


def _music_analysis_rows(value: Any, label: str, *, optional: bool = False) -> Sequence[Any]:
    if value is None and optional:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuthoringPlanError(f"context.music_analysis.{label} must be an array.")
    return value


def _music_analysis_number(
    value: Any,
    label: str,
    *,
    minimum: float = 0.0,
    inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuthoringPlanError(f"context.music_analysis.{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or (result < minimum if inclusive else result <= minimum):
        comparison = "at least" if inclusive else "greater than"
        raise AuthoringPlanError(
            f"context.music_analysis.{label} must be {comparison} {minimum:g}."
        )
    return result


def _music_analysis_status(analysis: Mapping[str, Any], key: str) -> str:
    value = analysis.get(key)
    if isinstance(value, Mapping):
        value = value.get("status")
    return str(value or "").strip().casefold()


def _validated_music_analysis(
    value: Any,
    *,
    target_duration: float,
) -> Mapping[str, Any]:
    analysis = _mapping(value, "context.music_analysis")
    duration = _music_analysis_number(
        analysis.get("duration"), "duration", minimum=0.0, inclusive=False
    )
    # CPU decoding and container probes can differ slightly for compressed
    # audio. The native source frame count remains the schedule's authority.
    duration_tolerance = max(0.25, target_duration * 0.001)
    if abs(duration - target_duration) > duration_tolerance:
        raise AuthoringPlanError(
            "context.music_analysis.duration no longer matches the current soundtrack. "
            "Refresh the creation context and its analysis."
        )
    no_reliable_beats = (
        _music_analysis_status(analysis, "beat_detection") == "no_reliable_beats"
    )
    _music_analysis_number(
        analysis.get("bpm"),
        "bpm",
        minimum=0.0,
        inclusive=no_reliable_beats,
    )

    beats = _music_analysis_rows(analysis.get("beats"), "beats")
    if not beats and not no_reliable_beats:
        raise AuthoringPlanError(
            "context.music_analysis.beats is empty; refresh the context instead of using duration-only timing."
        )
    if beats and no_reliable_beats:
        raise AuthoringPlanError(
            "context.music_analysis.beat_detection reports no reliable beats but also supplies beat rows."
        )
    for index, raw in enumerate(beats, start=1):
        row = _mapping(raw, f"context.music_analysis.beats[{index}]")
        time = _music_analysis_number(row.get("time"), f"beats[{index}].time")
        if time > duration + duration_tolerance:
            raise AuthoringPlanError(
                f"context.music_analysis.beats[{index}].time falls after the analyzed soundtrack."
            )
        if row.get("strength") is not None:
            _music_analysis_number(row["strength"], f"beats[{index}].strength")

    sections = _music_analysis_rows(analysis.get("sections"), "sections")
    for index, raw in enumerate(sections, start=1):
        row = _mapping(raw, f"context.music_analysis.sections[{index}]")
        start = _music_analysis_number(row.get("start"), f"sections[{index}].start")
        end = _music_analysis_number(row.get("end"), f"sections[{index}].end")
        if end <= start or end > duration + duration_tolerance:
            raise AuthoringPlanError(
                f"context.music_analysis.sections[{index}] has invalid soundtrack bounds."
            )
        if row.get("energy") is not None:
            _music_analysis_number(row["energy"], f"sections[{index}].energy")

    downbeats = _music_analysis_rows(
        analysis.get("downbeats"), "downbeats", optional=True
    )
    for index, raw in enumerate(downbeats, start=1):
        time = _music_analysis_number(raw, f"downbeats[{index}]")
        if time > duration + duration_tolerance:
            raise AuthoringPlanError(
                f"context.music_analysis.downbeats[{index}] falls after the analyzed soundtrack."
            )

    lyrics = _music_analysis_rows(analysis.get("lyrics"), "lyrics", optional=True)
    for index, raw in enumerate(lyrics, start=1):
        row = _mapping(raw, f"context.music_analysis.lyrics[{index}]")
        start = _music_analysis_number(row.get("start"), f"lyrics[{index}].start")
        end = _music_analysis_number(row.get("end"), f"lyrics[{index}].end")
        if end <= start or end > duration + duration_tolerance:
            raise AuthoringPlanError(
                f"context.music_analysis.lyrics[{index}] has invalid soundtrack bounds."
            )

    cues = _music_analysis_rows(
        analysis.get("music_cues"), "music_cues", optional=True
    )
    for index, raw in enumerate(cues, start=1):
        row = _mapping(raw, f"context.music_analysis.music_cues[{index}]")
        time = _music_analysis_number(row.get("time"), f"music_cues[{index}].time")
        if time > duration + duration_tolerance:
            raise AuthoringPlanError(
                f"context.music_analysis.music_cues[{index}] falls after the analyzed soundtrack."
            )

    percussion = analysis.get("percussion_activity")
    if percussion is not None:
        intervals = _music_analysis_rows(percussion, "percussion_activity")
        for index, raw in enumerate(intervals, start=1):
            row = _mapping(
                raw, f"context.music_analysis.percussion_activity[{index}]"
            )
            start = _music_analysis_number(
                row.get("start"), f"percussion_activity[{index}].start"
            )
            end = _music_analysis_number(
                row.get("end"), f"percussion_activity[{index}].end"
            )
            if end <= start or end > duration + duration_tolerance:
                raise AuthoringPlanError(
                    "context.music_analysis.percussion_activity["
                    f"{index}] has invalid soundtrack bounds."
                )

    return analysis


def _music_window_context(
    analysis: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    start: float,
    end: float,
    timing_basis: str,
) -> dict[str, Any]:
    """Attach a compact, server-derived music summary to one output window."""

    beat_times = sorted(
        float(beat["time"])
        for beat in analysis.get("beats") or []
        if start <= float(beat["time"]) < end
    )
    downbeat_times = sorted(
        float(time)
        for time in analysis.get("downbeats") or []
        if start <= float(time) < end
    )
    result: dict[str, Any] = {
        "timing_basis": timing_basis,
        "bpm": float(analysis["bpm"]),
        "beat_count": len(beat_times),
        "opening_beat_times": beat_times[:3],
        "ending_beat_times": beat_times[-3:],
        "downbeat_count": len(downbeat_times),
        "opening_downbeat_times": downbeat_times[:2],
        "music_cues": copy.deepcopy(row.get("music_cues") or []),
        "percussion_activity": row.get("percussion_activity", "unknown"),
    }
    for key in (
        "analysis_method",
        "method",
        "transcription_status",
        "beat_detection",
        "vocal_timing",
    ):
        if analysis.get(key) is not None:
            result[key] = copy.deepcopy(analysis[key])
    for key in (
        "section_label",
        "energy",
        "label",
        "suggested_prompt_hint",
        "dominant_speaker",
    ):
        if row.get(key) is not None:
            result[key] = copy.deepcopy(row[key])
    lyrics = [
        copy.deepcopy(line)
        for line in analysis.get("lyrics") or []
        if float(line["start"]) < end and float(line["end"]) > start
    ]
    if lyrics:
        result["lyrics"] = lyrics
    return result


def _music_video_editorial_schedule(
    profile: Mapping[str, Any],
    references: list[str],
    *,
    params: Mapping[str, Any] | None = None,
    model_def: Mapping[str, Any] | None = None,
    music_analysis: Any = None,
    analysis_supplied: bool = False,
    music_analysis_source: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build exact-source slots from analyzed music or a labeled legacy fallback."""

    target = profile.get("target_total_frames")
    if target is None:
        raise AuthoringPlanError(
            "The current music-video context has no exact source-audio duration. Add or select the soundtrack before authoring."
        )
    target = _integer(target, "Director source-audio duration frames")
    fps = _number(profile.get("fps"), "Director frame rate")
    minimum = _integer(profile.get("minimum_frames"), "Director minimum shot frames")
    maximum = _integer(profile.get("maximum_frames"), "Director maximum shot frames")
    step = _integer(profile.get("frame_step"), "Director shot frame step")

    from services.director_music_timing import prepare_music_timeline

    target_duration = target / fps
    timing_basis = "legacy_duration_only"
    analysis: Mapping[str, Any] = {}
    if analysis_supplied:
        analysis = _validated_music_analysis(
            music_analysis, target_duration=target_duration
        )
        from services.director_music_timing import (
            plan_capped_music_clips,
            resolve_clip_limits,
            resolve_music_pacing_bias,
        )

        maximum_on_lattice = minimum + ((maximum - minimum) // step) * step
        if maximum_on_lattice < minimum:
            raise AuthoringPlanError(
                "Director execution profile has no valid native shot length. Refresh the context."
            )
        resolved_model = dict(model_def or {})
        resolved_model.update({
            "fps": fps,
            "frames_minimum": minimum,
            "frames_steps": step,
            "frames_maximum": maximum_on_lattice,
        })
        resolved_clip_profile = {
            "fps": fps,
            "effective_max_frames": maximum_on_lattice,
            "manual_max_frames": maximum_on_lattice,
        }
        try:
            limits = resolve_clip_limits(
                resolved_model,
                (params or {}).get("director_music_clip_seconds"),
                resolved_clip_profile,
            )
            energy_bias = resolve_music_pacing_bias(
                (params or {}).get("energy_bias"), resolved_model
            )
            maximum_frames = min(maximum_on_lattice, int(limits["max_frames"]))
            planned = plan_capped_music_clips(
                analysis,
                maximum_seconds=maximum_frames / fps,
                fps=fps,
                frames_steps=step,
                frames_minimum=minimum,
                total_duration=target_duration,
                energy_bias=energy_bias,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise AuthoringPlanError(
                f"The current music analysis cannot produce a valid native schedule: {error}"
            ) from error
        if not planned:
            raise AuthoringPlanError(
                "The current music analysis produced no editorial clips; refresh the creation context."
            )
        timing_basis = (
            "section_and_duration_analysis"
            if not analysis.get("beats")
            else "beat_and_section_analysis"
        )
        _, scheduled = prepare_music_timeline(
            planned,
            planned,
            fps=fps,
            minimum_frames=minimum,
            maximum_frames=maximum_frames,
            frame_step=step,
        )
    else:
        _, scheduled = prepare_music_timeline(
            [{}],
            [{
                "start": 0.0,
                "end": target_duration,
                "duration_sec": target_duration,
                "duration_frames": target,
                "output_frames": target,
                "music_timing_version": 1,
            }],
            fps=fps,
            minimum_frames=minimum,
            maximum_frames=maximum,
            frame_step=step,
        )
    windows: list[dict[str, Any]] = []
    previous_id: str | None = None
    editorial_cursor = 0
    for index, row in enumerate(scheduled, start=1):
        start = round(float(row["start"]) * fps)
        end = round(float(row["end"]) * fps)
        output_frames = int(row["output_frames"])
        native_frames = int(row["duration_frames"])
        if start != editorial_cursor or end - start != output_frames:
            raise AuthoringPlanError(
                "The native music-timing helper did not preserve contiguous source-audio frames; refresh the context."
            )
        identifier = f"music-shot-{index:03d}"
        windows.append({
            "id": identifier,
            "index": index,
            "start_frame": start,
            "end_frame": end,
            "duration_frames": output_frames,
            "output_frames": output_frames,
            "native_duration_frames": native_frames,
            "trim_frames": max(0, native_frames - output_frames),
            "start_seconds": start / fps,
            "end_seconds": end / fps,
            "duration_seconds": output_frames / fps,
            "native_duration_seconds": native_frames / fps,
            "reference_ids": list(references),
            "previous_window_id": previous_id,
            "timing_basis": timing_basis,
        })
        if analysis_supplied:
            windows[-1].update(_music_window_context(
                analysis,
                row,
                start=start / fps,
                end=end / fps,
                timing_basis=timing_basis,
            ))
        previous_id = identifier
        editorial_cursor = end
    if editorial_cursor != target:
        raise AuthoringPlanError(
            "The native music-timing helper did not cover the exact soundtrack duration; refresh the context."
        )
    native_total = sum(row["native_duration_frames"] for row in windows)
    schedule = {
        "kind": "exact_source_audio",
        "requested_total_frames": target,
        "editorial_total_frames": editorial_cursor,
        "native_total_frames": native_total,
        "padding_frames": max(0, native_total - editorial_cursor),
        "final_trim_frames": max(0, native_total - editorial_cursor),
        "trim_semantics": "each native render segment is trimmed to its exact source-audio output_frames",
        "fps": fps,
        "timing_basis": timing_basis,
        "windows": windows,
    }
    if analysis_supplied:
        schedule["analysis_duration_seconds"] = float(analysis["duration"])
        schedule["bpm"] = float(analysis["bpm"])
        for key in (
            "analysis_method",
            "method",
            "transcription_status",
            "beat_detection",
            "vocal_timing",
        ):
            if analysis.get(key) is not None:
                schedule[key] = copy.deepcopy(analysis[key])
        if isinstance(music_analysis_source, Mapping):
            schedule["music_analysis_source"] = copy.deepcopy(dict(music_analysis_source))
    return windows, schedule


def resolve_native_context(
    context: Mapping[str, Any],
    request: Mapping[str, Any] | None,
    hardware: Mapping[str, Any] | None,
    *,
    lora_system_hint: str = "",
    injected_keyframes: list[dict[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Normalize native limits before showing the context to an author.

    The server resolves saved defaults and request overrides first. This
    callback then applies the same CPU-side memory/window/profile helpers used
    by the generation paths, so the external plan is authored against the
    effective geometry instead of a later clamped request.
    """

    if not isinstance(context, dict):
        raise AuthoringPlanError("context must be a mutable object.")
    request = _mapping(request or {}, "request")
    hardware = _mapping(hardware or {}, "hardware")
    _, workflow, params_view, model_def = _context_parts(context)
    params = copy.deepcopy(dict(params_view))
    model_type = str(context.get("model_type") or "")
    if lora_system_hint:
        context["lora_system_hint"] = str(lora_system_hint)
    if injected_keyframes is not None:
        context["injected_keyframes"] = copy.deepcopy(injected_keyframes)

    if workflow in _DIRECTOR_WORKFLOWS:
        from services.director_video_strategy import (
            build_director_video_execution_profile,
            resolve_shot_image_policy,
        )

        video_model = str(params.get("video_model") or model_type)
        video_params = params.get("video_params") or {}
        if not isinstance(video_params, Mapping):
            raise AuthoringPlanError("context.params.video_params must be an object.")
        profile = build_director_video_execution_profile(
            video_model,
            model_def,
            video_params,
            hardware,
            manual_max_frames=params.get("director_max_shot_frames"),
            resolution_preset=str(params.get("director_resolution_preset") or ""),
            aspect_ratio=str(params.get("director_aspect_ratio") or ""),
        )
        refs = context.get("references") or []
        has_visual_references = any(
            str((item.get("type") or item.get("kind") or item.get("role") or "")).casefold()
            in {"image", "picture", "character", "location", "visual", "reference"}
            for item in refs
            if isinstance(item, Mapping)
        )
        policy = resolve_shot_image_policy(
            model_def,
            params.get("shot_image_guidance"),
            has_visual_references=has_visual_references,
        )
        context["params"] = params
        context["execution_profile"] = profile
        context["shot_image_policy"] = policy
        context["fps"] = profile.get("fps") or model_def.get("fps") or 24
        return context

    if workflow not in {"video.frames", "video.references"}:
        context["params"] = params
        return context

    defaults = model_def.get("sliding_window_defaults") or {}
    if not isinstance(defaults, Mapping):
        defaults = {}
    try:
        total = _integer(params.get("video_length"), "params.video_length")
    except AuthoringPlanError:
        raise
    default_window = (
        defaults.get("window_default")
        or model_def.get("frames_maximum")
        or total
    )
    window = _integer(
        params.get("sliding_window_size") or default_window,
        "params.sliding_window_size",
    )
    overlap_default = defaults.get("overlap_default", 0)
    if params.get("sliding_window_overlap") is None:
        params["sliding_window_overlap"] = overlap_default
    if params.get("sliding_window_discard_last_frames") is None:
        params["sliding_window_discard_last_frames"] = int(
            defaults.get("discard_last_frames") or defaults.get("discard_default") or 0
        )
    params["sliding_window_size"] = window

    lower_model = model_type.casefold()
    architecture = str(model_def.get("architecture") or "").casefold()
    is_h3 = architecture.startswith("minimax_h3") or lower_model.startswith("minimax_h3")
    is_omni = bool(model_def.get("omni_reference"))
    is_ltx = bool(model_def.get("multi_window_sequence_controls"))

    if workflow == "video.references" and is_h3 and is_omni:
        from models.minimax_h3.minimax_h3_handler import (
            apply_h3_native_omni_memory_policy,
            apply_h3_omni_sequence_memory_policy,
            normalize_h3_overlap_frames,
        )

        explicit_single = request.get("minimax_h3_reference_sequence") is False
        sequence_enabled = (
            request.get("minimax_h3_reference_sequence") is True
            or params.get("minimax_h3_reference_sequence") is True
        )
        maximum = _integer(model_def.get("frames_maximum", window), "model_def.frames_maximum")
        current_window = _integer(
            params.get("minimax_h3_sequence_clip_frames")
            or params.get("sliding_window_size")
            or maximum,
            "MiniMax H3 Omni native window",
        )
        explicit_window_override = (
            params.get("minimax_h3_sequence_memory_override") is True
            or params.get("sliding_window_memory_override") is True
        )
        if total > maximum and explicit_single:
            raise AuthoringPlanError(
                f"The requested {total}-frame duration exceeds the {maximum}-frame "
                "MiniMax H3 Omni single-pass limit. Enable Reference Sequence or choose a shorter duration."
            )
        if not sequence_enabled and not explicit_single and total > current_window:
            sequence_enabled = True

        if not sequence_enabled:
            probe = copy.deepcopy(params)
            adjustment = apply_h3_native_omni_memory_policy(probe, dict(model_def), dict(hardware))
            requires_sequence = bool(adjustment and adjustment.get("unsupported"))
            if int(probe.get("video_length") or total) < total:
                requires_sequence = True
            if requires_sequence:
                if explicit_single or explicit_window_override:
                    detail = (
                        adjustment.get("message")
                        if adjustment and adjustment.get("unsupported")
                        else "the native one-pass memory profile is shorter than the requested duration"
                    )
                    raise AuthoringPlanError(
                        f"This explicit one-pass request cannot preserve {total} frames: {detail}"
                    )
                sequence_enabled = True
            else:
                params["minimax_h3_reference_sequence"] = False
                params["minimax_h3_sequence_clip_frames"] = min(total, current_window)
                if params.get("minimax_h3_sequence_continuity") is None:
                    params["minimax_h3_sequence_continuity"] = True
        if sequence_enabled:
            if explicit_single:
                raise AuthoringPlanError(
                    "The requested duration needs H3 Reference Sequence, but the request explicitly disabled it."
                )
            params["minimax_h3_reference_sequence"] = True
            params.setdefault("minimax_h3_sequence_prompt_mode", "auto")
            sequence_adjustment = apply_h3_omni_sequence_memory_policy(
                params, dict(model_def), dict(hardware)
            )
            if sequence_adjustment and sequence_adjustment.get("unsupported"):
                raise AuthoringPlanError(str(sequence_adjustment["message"]))
            maximum = _integer(
                params.get("minimax_h3_sequence_clip_frames")
                or (params.get("sliding_window_size") if model_def.get("minimax_h3_pika_flex") else None)
                or model_def.get("frames_maximum", current_window),
                "MiniMax H3 Omni sequence window",
            )
            params["sliding_window_size"] = (
                maximum if model_def.get("minimax_h3_pika_flex") else min(total, maximum)
            )
            native = params.get("minimax_h3_sequence_continuity") is not False
            params["minimax_h3_sequence_continuity"] = native
            if native:
                params["sliding_window_overlap"] = normalize_h3_overlap_frames(
                    params.get("sliding_window_overlap", overlap_default),
                    window_frames=params["sliding_window_size"],
                )
            else:
                params["sliding_window_overlap"] = 0
        context["params"] = params
        context["fps"] = model_def.get("fps") or context.get("fps") or 24
        return context

    if workflow == "video.frames" and is_h3:
        from models.minimax_h3.minimax_h3_handler import (
            apply_h3_window_memory_policy,
            normalize_h3_window_geometry,
            resolve_h3_long_sequence_discard_frames,
        )

        explicit_single = request.get("minimax_h3_multi_window") is False
        multi = request.get("minimax_h3_multi_window") is True or params.get("minimax_h3_multi_window") is True
        maximum = _integer(model_def.get("frames_maximum", window), "model_def.frames_maximum")
        if explicit_single:
            params["minimax_h3_multi_window"] = False
            if total > maximum:
                raise AuthoringPlanError(
                    f"The requested {total}-frame duration exceeds the {maximum}-frame "
                    "H3 single-pass limit. Enable multi-window or choose a shorter duration."
                )
            if total > window:
                params["sliding_window_size"] = total
        elif total > window:
            multi = True
            params["minimax_h3_multi_window"] = True
        else:
            params["minimax_h3_multi_window"] = multi

        if not explicit_single and not multi:
            single_pass_probe = copy.deepcopy(params)
            single_pass_probe["minimax_h3_multi_window"] = False
            normalize_h3_window_geometry(single_pass_probe, dict(model_def))
            if int(single_pass_probe.get("video_length") or 0) != total:
                multi = True
                params["minimax_h3_multi_window"] = True

        adjustment = apply_h3_window_memory_policy(params, dict(model_def), dict(hardware))
        if adjustment and adjustment.get("unsupported"):
            raise AuthoringPlanError(str(adjustment["message"]))
        effective_window = _integer(params.get("sliding_window_size") or window, "effective H3 window")
        if not multi and total > effective_window:
            if explicit_single:
                raise AuthoringPlanError(
                    f"The requested {total}-frame H3 single pass exceeds the current "
                    f"memory-safe {effective_window}-frame window. Enable multi-window or choose a shorter duration."
                )
            multi = True
            params["minimax_h3_multi_window"] = True
        normalize_h3_window_geometry(params, dict(model_def))
        if int(params.get("video_length") or 0) != total:
            raise AuthoringPlanError(
                "The selected H3 settings would shorten the requested duration. "
                "Refresh the context or choose a native timeline that preserves every requested frame."
            )
        params["sliding_window_discard_last_frames"] = resolve_h3_long_sequence_discard_frames(
            params.get("custom_settings"),
            enabled=(
                params.get("minimax_h3_multi_window") is True
                and int(params.get("video_length") or 0) > int(params.get("sliding_window_size") or 0)
            ),
        )
        context["params"] = params
        context["fps"] = model_def.get("fps") or context.get("fps") or 24
        return context

    if workflow == "video.frames" and is_ltx:
        explicit_single = request.get("ltx_multi_window") is False
        multi = request.get("ltx_multi_window") is True or params.get("ltx_multi_window") is True
        maximum = _integer(
            model_def.get("frames_maximum") or defaults.get("window_max") or window,
            "model_def.frames_maximum",
        )
        if total > window:
            if explicit_single:
                if total > maximum:
                    raise AuthoringPlanError(
                        f"The requested {total}-frame duration exceeds the {maximum}-frame "
                        "LTX single-pass limit. Enable multi-window or choose a shorter duration."
                    )
                params["sliding_window_size"] = total
                params["ltx_multi_window"] = False
            else:
                params["ltx_multi_window"] = True
                multi = True
        else:
            params["ltx_multi_window"] = multi
        if params.get("ltx_window_prompt_mode") is None:
            params["ltx_window_prompt_mode"] = "auto"
        context["params"] = params
        context["fps"] = model_def.get("fps") or context.get("fps") or 24
        return context

    # Generic video models support a single pass in this external authoring
    # surface. Raise when neither the selected pass nor the published model
    # maximum can cover the requested timeline; never report a shorter clip.
    maximum = _integer(
        model_def.get("frames_maximum") or defaults.get("window_max") or window,
        "model_def.frames_maximum",
    )
    if total > window:
        if total > maximum:
            raise AuthoringPlanError(
                f"The requested {total}-frame duration exceeds this model's {maximum}-frame "
                "single-pass limit, and no supported external window plan is available."
            )
        params["sliding_window_size"] = total
    context["params"] = params
    context["fps"] = model_def.get("fps") or context.get("fps") or 24
    return context


def compute_studio_timeline(context: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic model-native geometry for one creation context.

    ``context_id`` is copied from the authoritative context store. Callers can
    expose this result to an authoring agent and later require the same ID when
    validating its response.
    """

    context = _mapping(context, "context")
    context_id, workflow, params, model_def = _context_parts(context)
    references = _reference_ids(context)
    if workflow in _DIRECTOR_WORKFLOWS:
        profile = _director_profile(context, params, model_def)
        continuity = _continuity_mode(context.get("continuity_mode") or "continuous")
        director_timeline: dict[str, Any] = {
            "requested_total_frames": profile.get("target_total_frames"),
            "fps": profile["fps"],
            "native_minimum_frames": profile["minimum_frames"],
            "native_maximum_frames": profile["maximum_frames"],
            "native_frame_step": profile["frame_step"],
            "native_total_frames": None,
            "padding_frames": None,
            "final_trim_frames": 0,
            "trim_semantics": "no final trim; render uses native shot durations",
        }
        pipeline_type = str(params.get("pipeline_type") or "")
        requested_total = profile.get("target_total_frames")
        architecture = str(model_def.get("architecture") or context.get("model_type") or "").casefold()
        is_h3 = architecture.startswith("minimax_h3") or str(context.get("model_type") or "").casefold().startswith("minimax_h3")
        if pipeline_type == "music_video" and requested_total is not None:
            _, exact_schedule = _music_video_editorial_schedule(
                profile,
                references,
                params=params,
                model_def=model_def,
                music_analysis=context.get("music_analysis"),
                analysis_supplied="music_analysis" in context,
                music_analysis_source=context.get("music_analysis_source"),
            )
            director_timeline.update(exact_schedule)
        elif is_h3 and requested_total is not None:
            from services.h3_sequence_planner import compute_h3_sequence_clips

            geometry, padding = compute_h3_sequence_clips(
                requested_total,
                min_clip_frames=profile["minimum_frames"],
                max_clip_frames=profile["maximum_frames"],
                frame_step=profile["frame_step"],
                fps=profile["fps"],
            )
            windows = []
            previous_id = None
            for index, item in enumerate(geometry, start=1):
                identifier = f"shot-{index:03d}"
                windows.append({
                    "id": identifier,
                    "index": index,
                    "start_frame": int(item["start_frame"]),
                    "end_frame": int(item["end_frame"]),
                    "duration_frames": int(item["frames"]),
                    "start_seconds": float(item["start_seconds"]),
                    "end_seconds": float(item["end_seconds"]),
                    "duration_seconds": float(item["duration_seconds"]),
                    "requested_output_frames": max(
                        0,
                        min(int(item["end_frame"]), requested_total)
                        - int(item["start_frame"]),
                    ),
                    "reference_ids": list(references),
                    "previous_window_id": previous_id if continuity == "continuous" else None,
                })
                previous_id = identifier
            native_total = sum(int(item["frames"]) for item in geometry)
            director_timeline.update({
                "kind": "h3_native_story_schedule",
                "native_total_frames": native_total,
                "padding_frames": int(padding),
                "final_trim_frames": 0,
                "trim_semantics": "no final trim; requested duration is approximated by the native clip schedule",
            })
            director_timeline["windows"] = windows
        else:
            director_timeline["kind"] = "authored_native_clips"
        return {
            "context_id": context_id,
            "workflow": workflow,
            "kind": "director",
            "continuity_mode": continuity,
            "reference_ids": references,
            "windows": director_timeline.get("windows", []),
            "director_profile": profile,
            "director_timeline": director_timeline,
            "plan_schema": {
                "mode": "director_clips",
                "required_fields": ["context_id", "clips"],
                "clip_fields": [
                    "id", "start_frame", "end_frame", "duration_frames",
                    "previous_clip_id", "reference_ids", "video_prompt",
                ],
                "window_prompt_fields": [
                    "id", "start_frame", "end_frame", "prompt",
                    "reference_ids", "previous_window_id",
                ],
                "time_basis": (
                    "editorial_source_frames"
                    if director_timeline["kind"] == "exact_source_audio"
                    else "native_frames"
                ),
                "timeline_id_required": director_timeline["kind"] in {
                    "h3_native_story_schedule", "exact_source_audio",
                },
                "native_duration_field": (
                    "native_duration_frames"
                    if director_timeline["kind"] == "exact_source_audio"
                    else "duration_frames"
                ),
                "output_duration_field": (
                    "output_frames"
                    if director_timeline["kind"] == "exact_source_audio"
                    else None
                ),
                "requested_total_frames": director_timeline.get("requested_total_frames"),
                "native_total_frames": director_timeline.get("native_total_frames"),
                "padding_frames": director_timeline.get("padding_frames"),
                "final_trim_frames": director_timeline.get("final_trim_frames"),
                "trim_semantics": director_timeline.get("trim_semantics"),
            },
        }

    if workflow in _SINGLE_PROMPT_WORKFLOWS:
        return {
            "context_id": context_id,
            "workflow": workflow,
            "kind": "single",
            "continuity_mode": "single",
            "reference_ids": references,
            "window_count": 1,
            "plan_schema": {
                "mode": "single",
                "required_fields": ["context_id", "prompt"],
                "optional_fields": ["style"],
                "style_native_field": "alt_prompt" if workflow == "audio.music" else None,
            },
            "windows": [{
                "id": "window-001",
                "index": 1,
                "start_frame": None,
                "end_frame": None,
                "start_seconds": None,
                "end_seconds": None,
                "reference_ids": list(references),
                "previous_window_id": None,
            }],
        }

    if workflow not in {"video.frames", "video.references"}:
        raise AuthoringPlanError(f"Unsupported creative workflow {workflow!r}.")

    fps = _number(context.get("fps") or model_def.get("fps") or 24, "Video frame rate")
    total = _integer(params.get("video_length"), "params.video_length")
    window = _integer(
        params.get("sliding_window_size") or model_def.get("frames_maximum") or total,
        "params.sliding_window_size",
    )
    overlap = _integer(params.get("sliding_window_overlap", 0), "params.sliding_window_overlap", minimum=0)
    discard = _integer(
        params.get("sliding_window_discard_last_frames", 0),
        "params.sliding_window_discard_last_frames",
        minimum=0,
    )
    lower_model = str(context.get("model_type") or "").casefold()
    architecture = str(model_def.get("architecture") or "").casefold()
    is_h3 = architecture.startswith("minimax_h3") or lower_model.startswith("minimax_h3")
    is_omni = bool(model_def.get("omni_reference"))
    is_ltx = bool(model_def.get("multi_window_sequence_controls"))
    geometry: list[dict[str, Any]]
    continuity = "independent"
    geometry_kind = "single_pass"

    if workflow == "video.references" and is_omni and params.get("minimax_h3_reference_sequence") is True:
        from services.h3_sequence_planner import (
            compute_h3_native_sequence_windows,
            compute_h3_sequence_clips,
        )

        minimum = _integer(model_def.get("frames_minimum", 124), "model_def.frames_minimum")
        maximum = _integer(
            params.get("minimax_h3_sequence_clip_frames")
            or (params.get("sliding_window_size") if model_def.get("minimax_h3_pika_flex") else None)
            or model_def.get("frames_maximum", window),
            "model_def.frames_maximum",
        )
        step = _integer(model_def.get("frames_steps", 17), "model_def.frames_steps")
        native = params.get("minimax_h3_sequence_continuity") is not False
        if native:
            geometry = compute_h3_native_sequence_windows(
                total,
                window_frames=maximum,
                overlap_frames=overlap,
                fps=fps,
            )
            continuity = "continuous"
            geometry_kind = "h3_native_sequence"
        else:
            geometry, _ = compute_h3_sequence_clips(
                total,
                min_clip_frames=minimum,
                max_clip_frames=maximum,
                frame_step=step,
                fps=fps,
            )
            continuity = "independent"
            geometry_kind = "h3_hard_cut_sequence"
    elif workflow == "video.frames" and is_h3 and params.get("minimax_h3_multi_window") is True:
        from services.h3_window_planner import compute_h3_window_boundaries

        geometry = compute_h3_window_boundaries(
            total,
            window,
            fps=fps,
            overlap_frames=overlap,
            discard_frames=discard,
        )
        continuity = "continuous"
        geometry_kind = "h3_sliding_windows"
    elif workflow == "video.frames" and is_ltx and params.get("ltx_multi_window") is True and total > window:
        try:
            geometry = _ltx_pass_geometry(
                total,
                window,
                overlap_frames=overlap,
                discard_frames=discard,
                fps=fps,
            )
        except ValueError as error:
            raise AuthoringPlanError(str(error)) from error
        continuity = "continuous"
        geometry_kind = "ltx_sliding_windows"
    else:
        if total > window:
            raise AuthoringPlanError(
                f"The requested {total}-frame timeline exceeds the {window}-frame native pass, "
                "and this model has no enabled supported continuation path. Refresh native context "
                "or select an available multi-window workflow."
            )
        effective_total = total
        geometry = [{
            "index": 1,
            "start_frame": 0,
            "end_frame": effective_total,
            "start_seconds": 0.0,
            "end_seconds": round(effective_total / fps, 3),
            "duration_frames": effective_total,
            "duration_seconds": round(effective_total / fps, 3),
        }]

    windows = _window_rows(geometry, references=references, continuity=continuity)
    for output, source in zip(windows, geometry):
        for key in (
            "frames",
            "duration_frames",
            "duration_seconds",
            "pass_start_frame",
            "pass_end_frame",
            "overlap_before_frames",
            "discarded_tail_frames",
        ):
            if key in source:
                output[key] = source[key]
    return {
        "context_id": context_id,
        "workflow": workflow,
        "kind": geometry_kind,
        "continuity_mode": continuity,
        "reference_ids": references,
        "fps": fps,
        "total_frames": total,
        "window_frames": window,
        "overlap_frames": overlap,
        "discard_frames": discard,
        "window_count": len(windows),
        "windows": windows,
        "plan_schema": {
            "mode": "windows" if len(windows) > 1 else "single",
            "required_fields": (
                ["context_id", "windows"] if len(windows) > 1 else ["context_id", "prompt"]
            ),
            "window_fields": [
                "id", "index", "start_frame", "end_frame", "prompt",
                "reference_ids", "previous_window_id",
            ],
        },
    }


def _require_context_binding(context: Mapping[str, Any], plan: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    context_id, workflow, params, model_def = _context_parts(context)
    supplied_id = _required_text(plan.get("context_id"), "plan.context_id")
    if supplied_id != context_id:
        raise AuthoringPlanError(
            "This plan was authored against a different creation context. Refresh the context and try again."
        )
    timeline = context.get("timeline")
    if not isinstance(timeline, Mapping) or timeline.get("context_id") != context_id:
        raise AuthoringPlanError("The current context has no matching server-computed timeline; refresh it.")
    return context_id, workflow, params, model_def, timeline


def _validate_authored_windows(
    timeline: Mapping[str, Any],
    plan: Mapping[str, Any],
    expected_references: list[str],
) -> list[dict[str, Any]]:
    expected = timeline.get("windows")
    actual = plan.get("windows")
    if not isinstance(expected, list) or not isinstance(actual, list):
        raise AuthoringPlanError("A multi-window plan must include a windows array from the current timeline.")
    if len(actual) != len(expected):
        raise AuthoringPlanError(
            f"The current timeline needs exactly {len(expected)} window prompts; received {len(actual)}. "
            "Refresh the context if duration or window settings changed."
        )
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    _continuity_mode(timeline.get("continuity_mode"))
    for index, (raw, geometry) in enumerate(zip(actual, expected), start=1):
        row = _mapping(raw, f"plan.windows[{index}]")
        window_id = _required_text(row.get("id"), f"plan.windows[{index}].id")
        if window_id != geometry.get("id"):
            raise AuthoringPlanError(
                f"plan.windows[{index}].id must be {geometry.get('id')!r} from the current timeline."
            )
        if window_id in seen_ids:
            raise AuthoringPlanError(f"plan.windows[{index}].id is duplicated.")
        seen_ids.add(window_id)
        for key in ("index", "start_frame", "end_frame"):
            expected_value = geometry.get(key)
            if expected_value is None:
                continue
            value = row.get(key)
            if isinstance(value, bool) or value != expected_value:
                raise AuthoringPlanError(
                    f"plan.windows[{index}].{key} must match the current timeline value {expected_value!r}. "
                    "Refresh the context and re-author against its exact geometry."
                )
        prompt = _required_text(row.get("prompt"), f"plan.windows[{index}].prompt")
        if "\n" in prompt or "\r" in prompt:
            raise AuthoringPlanError(
                f"plan.windows[{index}].prompt must be a single line so the native window selector keeps one prompt per pass."
            )
        refs = _ordered_reference_ids(
            row.get("reference_ids", [] if not expected_references else None),
            expected_references,
            f"plan.windows[{index}].reference_ids",
        )
        expected_previous = geometry.get("previous_window_id")
        supplied_previous = row.get("previous_window_id")
        if supplied_previous != expected_previous:
            expected_text = repr(expected_previous)
            raise AuthoringPlanError(
                f"plan.windows[{index}].previous_window_id must be {expected_text} for this timeline's continuity mode."
            )
        normalized.append({
            "id": window_id,
            "index": index,
            "start_frame": geometry.get("start_frame"),
            "end_frame": geometry.get("end_frame"),
            "prompt": prompt,
            "reference_ids": refs,
            "previous_window_id": expected_previous,
        })
    return normalized


def _h3_first_last_metadata(
    context: Mapping[str, Any],
    params: Mapping[str, Any],
    model_def: Mapping[str, Any],
    timeline: Mapping[str, Any],
    windows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    from services.h3_window_planner import (
        h3_window_plan_signature,
        lora_guidance_fingerprint,
    )

    prompt = str(context.get("brief") or "").strip()
    model_type = str(context.get("model_type") or "")
    resolution = str(params.get("resolution") or "")
    planning_style = str(params.get("minimax_h3_sequence_prompt_mode") or "auto")
    if planning_style not in {"creative", "adaptive"}:
        planning_style = "faithful"
    lora_hint = str(context.get("lora_system_hint") or "")
    signature = h3_window_plan_signature(
        prompt,
        model_type=model_type,
        resolution=resolution,
        total_frames=int(timeline["total_frames"]),
        window_frames=int(timeline["window_frames"]),
        overlap_frames=int(timeline.get("overlap_frames") or 0),
        discard_frames=int(timeline.get("discard_frames") or 0),
        fps=float(timeline.get("fps") or model_def.get("fps") or 24),
        has_start_image=bool(params.get("image_start")),
        has_end_image=bool(params.get("image_end")),
        camera_coverage=str(params.get("minimax_h3_camera_coverage") or "auto"),
        planning_style=planning_style,
        lora_system_hint=lora_hint,
        injected_keyframes=context.get("injected_keyframes"),
    )
    plan = {
        "plan_kind": "sliding_window",
        "source_prompt": prompt,
        "signature": signature,
        "model_type": model_type,
        "resolution": resolution,
        "total_frames": int(timeline["total_frames"]),
        "window_frames": int(timeline["window_frames"]),
        "overlap_frames": int(timeline.get("overlap_frames") or 0),
        "discard_frames": int(timeline.get("discard_frames") or 0),
        "fps": float(timeline.get("fps") or 24),
        "camera_coverage": str(params.get("minimax_h3_camera_coverage") or "auto"),
        "planning_style": planning_style,
        "lora_guidance_fingerprint": lora_guidance_fingerprint(lora_hint),
        "windows": [
            {
                "index": item["index"],
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "prompt": item["prompt"],
            }
            for item in windows
        ],
        "window_prompts": [item["prompt"] for item in windows],
        "planned_by": "external_agent",
    }
    return signature, plan


def prepare_authored_generation(
    context: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an authored Studio plan and return native request overrides."""

    context = _mapping(context, "context")
    plan = _mapping(plan, "plan")
    context_id, workflow, params, model_def, timeline = _require_context_binding(context, plan)
    expected_references = _reference_ids(context)
    expected_windows = timeline.get("windows")
    if not isinstance(expected_windows, list) or not expected_windows:
        raise AuthoringPlanError("The current context has no usable timeline; refresh it.")

    supplied_windows = plan.get("windows")
    if len(expected_windows) == 1 and supplied_windows is None:
        prompt = _required_text(plan.get("prompt"), "plan.prompt")
        if workflow == "audio.music":
            native = {"prompt": prompt}
            style = plan.get("style")
            if style is not None:
                native["alt_prompt"] = _required_text(style, "plan.style")
        else:
            native = {"prompt": prompt}
        if plan.get("reference_ids") is not None:
            _ordered_reference_ids(plan.get("reference_ids"), expected_references, "plan.reference_ids")
        return {
            "native_params": native,
            "plan_metadata": {
                "context_id": context_id,
                "workflow": workflow,
                "window_count": 1,
                "reference_ids": expected_references,
                "authored_by": "external_agent",
            },
        }

    windows = _validate_authored_windows(timeline, plan, expected_references)
    prompts = [item["prompt"] for item in windows]
    timeline_kind = str(timeline.get("kind") or "")
    native: dict[str, Any]
    metadata: dict[str, Any] = {
        "context_id": context_id,
        "workflow": workflow,
        "window_count": len(windows),
        "continuity_mode": timeline.get("continuity_mode"),
        "reference_ids": expected_references,
        "windows": windows,
        "authored_by": "external_agent",
    }

    if workflow == "video.frames" and timeline_kind == "h3_sliding_windows":
        signature, h3_plan = _h3_first_last_metadata(context, params, model_def, timeline, windows)
        native = {
            "prompt": "\n".join(prompts),
            "_h3_original_prompt": str(context.get("brief") or "").strip(),
            "h3_window_prompts": prompts,
            "h3_window_plan_signature": signature,
            "h3_window_plan": h3_plan,
            "_h3_window_plan_reviewed": True,
            "minimax_h3_multi_window": True,
            "minimax_h3_sequence_prompt_mode": str(
                params.get("minimax_h3_sequence_prompt_mode") or "auto"
            ),
            "multi_prompts_gen_type": 2,
        }
        metadata["h3_window_plan_signature"] = signature
        metadata["h3_window_plan"] = h3_plan
    elif workflow == "video.references" and timeline_kind in {
        "h3_native_sequence",
        "h3_hard_cut_sequence",
    }:
        native_continuation = timeline_kind == "h3_native_sequence"
        native = {
            "prompt": "\n".join(prompts),
            "minimax_h3_reference_sequence": True,
            "minimax_h3_sequence_prompt_mode": "manual",
            "minimax_h3_sequence_continuity": native_continuation,
        }
        from services.h3_sequence_planner import build_manual_h3_reference_sequence_plan

        h3_plan = build_manual_h3_reference_sequence_plan(
            native["prompt"],
            model_type=str(context.get("model_type") or ""),
            resolution=str(params.get("resolution") or ""),
            total_frames=int(timeline["total_frames"]),
            references=list(params.get("minimax_h3_references") or []),
            min_clip_frames=int(model_def.get("frames_minimum") or 124),
            max_clip_frames=int(model_def.get("frames_maximum") or timeline.get("window_frames") or 345),
            frame_step=int(model_def.get("frames_steps") or 17),
            fps=float(timeline.get("fps") or 24),
            camera_coverage=str(params.get("minimax_h3_camera_coverage") or "auto"),
            overlap_frames=int(timeline.get("overlap_frames") or 0),
            native_continuation=native_continuation,
        )
        native["h3_window_prompts"] = prompts
        native["h3_window_plan_signature"] = h3_plan["signature"]
        native["h3_window_plan"] = h3_plan
        metadata["h3_window_plan_signature"] = h3_plan["signature"]
        metadata["h3_window_plan"] = h3_plan
    elif workflow == "video.frames" and timeline_kind == "ltx_sliding_windows":
        native = {
            "prompt": "\n".join(prompts),
            "ltx_multi_window": True,
            "ltx_window_prompt_mode": "manual",
            "ltx_window_prompts": prompts,
            "multi_prompts_gen_type": 1,
        }
    else:
        raise AuthoringPlanError(
            f"A per-window authored plan is not supported for {workflow!r} in the current native mode."
        )

    return {"native_params": native, "plan_metadata": metadata}


def _validate_director_window_prompts(
    context: Mapping[str, Any],
    clip: Mapping[str, Any],
    *,
    clip_index: int,
    clip_start: int,
    clip_end: int,
    references: list[str],
    continuity: str,
    expected_geometry: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    supplied = clip.get("window_prompts")
    if supplied is None:
        if len(expected_geometry) > 1:
            raise AuthoringPlanError(
                f"plan.clips[{clip_index}].window_prompts must provide all "
                f"{len(expected_geometry)} native prompts for this clip."
            )
        return None, []
    if not isinstance(supplied, list):
        raise AuthoringPlanError(f"plan.clips[{clip_index}].window_prompts must be an array.")
    if not supplied:
        raise AuthoringPlanError(f"plan.clips[{clip_index}].window_prompts cannot be empty when supplied.")
    if len(supplied) != len(expected_geometry):
        raise AuthoringPlanError(
            f"plan.clips[{clip_index}].window_prompts needs exactly "
            f"{len(expected_geometry)} prompts for the current native geometry; "
            f"received {len(supplied)}. Refresh or revise the window plan."
        )
    result: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    previous_id: str | None = None
    for window_index, (raw, geometry) in enumerate(zip(supplied, expected_geometry), start=1):
        row = _mapping(raw, f"plan.clips[{clip_index}].window_prompts[{window_index}]")
        window_id = _required_text(
            row.get("id"), f"plan.clips[{clip_index}].window_prompts[{window_index}].id"
        )
        prompt = _required_text(
            row.get("prompt"), f"plan.clips[{clip_index}].window_prompts[{window_index}].prompt"
        )
        if "\n" in prompt or "\r" in prompt:
            raise AuthoringPlanError(
                f"plan.clips[{clip_index}].window_prompts[{window_index}].prompt must be a single line."
            )
        start = _integer(
            row.get("start_frame"),
            f"plan.clips[{clip_index}].window_prompts[{window_index}].start_frame",
            minimum=0,
        )
        end = _integer(
            row.get("end_frame"),
            f"plan.clips[{clip_index}].window_prompts[{window_index}].end_frame",
        )
        expected_start = clip_start + int(geometry["start_frame"])
        expected_end = clip_start + int(geometry["end_frame"])
        if start != expected_start or end != expected_end or not (clip_start <= start < end <= clip_end):
            raise AuthoringPlanError(
                f"plan.clips[{clip_index}].window_prompts[{window_index}] must match "
                f"native frames {expected_start}–{expected_end}; refresh or revise its geometry."
            )
        refs = _ordered_reference_ids(
            row.get("reference_ids", [] if not references else None),
            references,
            f"plan.clips[{clip_index}].window_prompts[{window_index}].reference_ids",
        )
        expected_previous = previous_id if continuity == "continuous" else None
        if row.get("previous_window_id") != expected_previous:
            raise AuthoringPlanError(
                f"plan.clips[{clip_index}].window_prompts[{window_index}].previous_window_id "
                f"must be {expected_previous!r} for this timeline's continuity mode."
            )
        result.append({"id": window_id, "start_frame": start, "end_frame": end, "prompt": prompt})
        provenance.append({
            "id": window_id,
            "start_frame": start,
            "end_frame": end,
            "reference_ids": refs,
            "previous_window_id": expected_previous,
        })
        previous_id = window_id
    return result, provenance


def _director_clip_window_geometry(
    context: Mapping[str, Any],
    params: Mapping[str, Any],
    model_def: Mapping[str, Any],
    *,
    duration_frames: int,
    fps: float,
) -> list[dict[str, Any]]:
    """Compute prompt spans only when the selected Director mode is windowed."""

    if bool(model_def.get("multi_window_sequence_controls")) and params.get("ltx_multi_window") is True:
        from services.ltx_window_planner import compute_ltx_window_count

        profile = context.get("execution_profile") or params.get("_director_video_execution_profile") or {}
        if not isinstance(profile, Mapping):
            profile = {}
        window_frames = _integer(
            params.get("sliding_window_size") or profile.get("effective_max_frames")
            or model_def.get("frames_maximum") or duration_frames,
            "Director LTX window length",
        )
        overlap = _integer(params.get("sliding_window_overlap") or 0, "Director LTX overlap", minimum=0)
        discard = _integer(
            params.get("sliding_window_discard_last_frames") or 0,
            "Director LTX discarded tail",
            minimum=0,
        )
        try:
            count = compute_ltx_window_count(
                duration_frames,
                window_frames,
                overlap_frames=overlap,
                discard_frames=discard,
            )
        except ValueError as error:
            raise AuthoringPlanError(str(error)) from error
        return _ltx_pass_geometry(
            duration_frames,
            window_frames,
            overlap_frames=overlap,
            discard_frames=discard,
            fps=fps,
        ) if count > 1 else [{"index": 1, "start_frame": 0, "end_frame": duration_frames}]

    architecture = str(model_def.get("architecture") or context.get("model_type") or "").casefold()
    if architecture.startswith("minimax_h3") and params.get("minimax_h3_multi_window") is True:
        from services.h3_window_planner import compute_h3_window_boundaries

        profile = context.get("execution_profile") or params.get("_director_video_execution_profile") or {}
        if not isinstance(profile, Mapping):
            profile = {}
        window_frames = _integer(
            params.get("sliding_window_size") or profile.get("effective_max_frames")
            or model_def.get("frames_maximum") or duration_frames,
            "Director H3 window length",
        )
        overlap = _integer(params.get("sliding_window_overlap") or 0, "Director H3 overlap", minimum=0)
        discard = _integer(
            params.get("sliding_window_discard_last_frames") or 0,
            "Director H3 discarded tail",
            minimum=0,
        )
        return compute_h3_window_boundaries(
            duration_frames,
            window_frames,
            fps=fps,
            overlap_frames=overlap,
            discard_frames=discard,
        )
    return [{"index": 1, "start_frame": 0, "end_frame": duration_frames}]


def _director_image_policy(
    context: Mapping[str, Any],
    params: Mapping[str, Any],
    model_def: Mapping[str, Any],
) -> tuple[str, Any]:
    try:
        from services.director_video_strategy import (
            resolve_shot_image_policy,
            shot_images_required,
        )

        policy = str(context.get("shot_image_policy") or params.get("shot_image_policy") or "")
        if not policy:
            policy = resolve_shot_image_policy(
                model_def,
                params.get("shot_image_guidance"),
                has_visual_references=any(
                    str(item.get("type") or item.get("kind") or item.get("role") or "").casefold()
                    in {"image", "picture", "character", "location", "visual", "reference"}
                    for item in (context.get("references") or [])
                    if isinstance(item, Mapping)
                ),
            )
        return policy, shot_images_required
    except ImportError:
        policy = str(context.get("shot_image_policy") or params.get("shot_image_policy") or "required")

        def required(value: Any) -> bool:
            return str(value).casefold() in {"required", "generate"}

        return policy, required


def _authored_clip_prompts(
    clip: Mapping[str, Any],
    *,
    index: int,
    requires_image: bool,
) -> tuple[str, str | None]:
    video_prompt = _required_text(clip.get("video_prompt"), f"plan.clips[{index}].video_prompt")
    image_prompt = clip.get("image_prompt")
    if requires_image:
        image_prompt = _required_text(image_prompt, f"plan.clips[{index}].image_prompt")
    elif image_prompt is not None:
        image_prompt = _required_text(image_prompt, f"plan.clips[{index}].image_prompt")
    return video_prompt, image_prompt


def _normalize_h3_story_schedule(
    context: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    context_id: str,
    workflow: str,
    params: Mapping[str, Any],
    model_def: Mapping[str, Any],
    timeline: Mapping[str, Any],
    raw_clips: list[Any],
) -> dict[str, Any]:
    director_timeline = timeline.get("director_timeline")
    expected = director_timeline.get("windows") if isinstance(director_timeline, Mapping) else None
    if not isinstance(expected, list) or not expected:
        raise AuthoringPlanError("The current H3 Director context has no native shot schedule; refresh it.")
    if len(raw_clips) != len(expected):
        raise AuthoringPlanError(
            f"The current H3 Director context requires {len(expected)} native shots; "
            f"received {len(raw_clips)}. Refresh the context before revising the screenplay."
        )
    policy, image_required = _director_image_policy(context, params, model_def)
    requires_image = bool(image_required(policy))
    references = _reference_ids(context)
    continuity = _continuity_mode(timeline.get("continuity_mode") or "continuous")
    fps = float(director_timeline.get("fps") or 24)
    plans: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    previous_source_id: str | None = None
    for index, (raw, geometry) in enumerate(zip(raw_clips, expected), start=1):
        clip = _mapping(raw, f"plan.clips[{index}]")
        source_id = _required_text(clip.get("id"), f"plan.clips[{index}].id")
        if source_id in source_ids:
            raise AuthoringPlanError(f"plan.clips[{index}].id is duplicated: {source_id!r}.")
        source_ids.add(source_id)
        timeline_id = _required_text(clip.get("timeline_id"), f"plan.clips[{index}].timeline_id")
        if timeline_id != geometry.get("id"):
            raise AuthoringPlanError(
                f"plan.clips[{index}].timeline_id must be {geometry.get('id')!r} from the current context."
            )
        expected_values = {
            "start_frame": int(geometry["start_frame"]),
            "end_frame": int(geometry["end_frame"]),
            "duration_frames": int(geometry["duration_frames"]),
        }
        for key, value in expected_values.items():
            if clip.get(key) != value:
                raise AuthoringPlanError(
                    f"plan.clips[{index}].{key} must match the current H3 schedule value {value}. "
                    "Refresh the context and re-author against its native shot durations."
                )
        expected_previous = previous_source_id if continuity == "continuous" else None
        if clip.get("previous_clip_id") != expected_previous:
            raise AuthoringPlanError(
                f"plan.clips[{index}].previous_clip_id must be {expected_previous!r}."
            )
        refs = _ordered_reference_ids(
            clip.get("reference_ids", [] if not references else None),
            references,
            f"plan.clips[{index}].reference_ids",
        )
        video_prompt, image_prompt = _authored_clip_prompts(
            clip, index=index, requires_image=requires_image
        )
        geometry_for_clip = _director_clip_window_geometry(
            context,
            params,
            model_def,
            duration_frames=expected_values["duration_frames"],
            fps=fps,
        )
        windows, provenance = _validate_director_window_prompts(
            context,
            clip,
            clip_index=index,
            clip_start=expected_values["start_frame"],
            clip_end=expected_values["end_frame"],
            references=references,
            continuity=continuity,
            expected_geometry=geometry_for_clip,
        )
        item = {
            "video_prompt": video_prompt,
            "agent_source_clip_id": source_id,
            "agent_timeline_id": timeline_id,
            "agent_previous_clip_id": expected_previous,
            "agent_reference_ids": refs,
        }
        if image_prompt is not None:
            item["image_prompt"] = image_prompt
        if windows is not None:
            item["window_prompts"] = windows
            item["agent_window_provenance"] = provenance
        for key in ("label", "scene_type", "keyframe_prompts", "narrative_role"):
            if clip.get(key) is not None:
                item[key] = clip[key]
        duration = expected_values["duration_frames"]
        start = expected_values["start_frame"]
        end = expected_values["end_frame"]
        plans.append(item)
        planned.append({
            "start": start / fps,
            "end": end / fps,
            "duration_sec": duration / fps,
            "duration_frames": duration,
            "agent_source_clip_id": source_id,
            "agent_timeline_id": timeline_id,
            "agent_previous_clip_id": expected_previous,
            "agent_reference_ids": refs,
        })
        previous_source_id = source_id
    native_total = sum(item["duration_frames"] for item in planned)
    return {
        "native_params": {
            "prepared_clip_plans": plans,
            "prepared_planned_clips": planned,
        },
        "plan_metadata": {
            "context_id": context_id,
            "workflow": workflow,
            "clip_count": len(plans),
            "requested_total_frames": director_timeline.get("requested_total_frames"),
            "native_total_frames": native_total,
            "padding_frames": max(0, native_total - int(director_timeline.get("requested_total_frames") or native_total)),
            "final_trim_frames": 0,
            "trim_semantics": director_timeline.get("trim_semantics"),
            "fps": fps,
            "continuity_mode": continuity,
            "reference_ids": references,
            "authored_by": "external_agent",
        },
    }


def _normalize_director_music_timeline(
    context: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    context_id: str,
    workflow: str,
    params: Mapping[str, Any],
    model_def: Mapping[str, Any],
    timeline: Mapping[str, Any],
    raw_clips: list[Any],
) -> dict[str, Any]:
    director_timeline = timeline.get("director_timeline")
    target = director_timeline.get("requested_total_frames") if isinstance(director_timeline, Mapping) else None
    if target is None:
        raise AuthoringPlanError(
            "The current music-video context has no exact source-audio duration. Add or select the soundtrack before authoring."
        )
    target = int(target)
    expected_windows = director_timeline.get("windows") if isinstance(director_timeline, Mapping) else None
    if not isinstance(expected_windows, list) or not expected_windows:
        raise AuthoringPlanError(
            "The current music-video context has no native editorial schedule. Refresh the context before authoring."
        )
    if len(raw_clips) != len(expected_windows):
        raise AuthoringPlanError(
            f"The current soundtrack schedule requires {len(expected_windows)} editorial shots; "
            f"received {len(raw_clips)}. Refresh the context before revising the screenplay."
        )
    profile = timeline.get("director_profile") or {}
    fps = float(profile.get("fps") or 24)
    minimum = int(profile.get("minimum_frames") or model_def.get("frames_minimum") or 1)
    maximum = int(profile.get("maximum_frames") or model_def.get("frames_maximum") or minimum)
    step = int(profile.get("frame_step") or model_def.get("frames_steps") or 1)
    references = _reference_ids(context)
    continuity = _continuity_mode(timeline.get("continuity_mode") or "continuous")
    policy, image_required = _director_image_policy(context, params, model_def)
    requires_image = bool(image_required(policy))
    raw_plans: list[dict[str, Any]] = []
    raw_timeline: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    ordered_source_ids: list[str] = []
    cursor = 0
    previous_source_id: str | None = None
    for index, (raw, geometry) in enumerate(zip(raw_clips, expected_windows), start=1):
        clip = _mapping(raw, f"plan.clips[{index}]")
        source_id = _required_text(clip.get("id"), f"plan.clips[{index}].id")
        if source_id in source_ids:
            raise AuthoringPlanError(f"plan.clips[{index}].id is duplicated: {source_id!r}.")
        source_ids.add(source_id)
        ordered_source_ids.append(source_id)
        timeline_id = _required_text(clip.get("timeline_id"), f"plan.clips[{index}].timeline_id")
        if timeline_id != geometry.get("id"):
            raise AuthoringPlanError(
                f"plan.clips[{index}].timeline_id must be {geometry.get('id')!r} from the current soundtrack schedule."
            )
        start = _integer(clip.get("start_frame"), f"plan.clips[{index}].start_frame", minimum=0)
        end = _integer(clip.get("end_frame"), f"plan.clips[{index}].end_frame")
        duration = _integer(clip.get("duration_frames"), f"plan.clips[{index}].duration_frames")
        expected_values = {
            "start_frame": int(geometry["start_frame"]),
            "end_frame": int(geometry["end_frame"]),
            "duration_frames": int(geometry["duration_frames"]),
        }
        for key, value in expected_values.items():
            if clip.get(key) != value:
                raise AuthoringPlanError(
                    f"plan.clips[{index}].{key} must match current soundtrack schedule value {value}. "
                    "Refresh the context before revising the shot boundaries."
                )
        if start != cursor or end != start + duration:
            raise AuthoringPlanError(
                f"plan.clips[{index}] must start at editorial frame {cursor} and end at "
                "start_frame + duration_frames."
            )
        expected_previous = previous_source_id if continuity == "continuous" else None
        if clip.get("previous_clip_id") != expected_previous:
            raise AuthoringPlanError(
                f"plan.clips[{index}].previous_clip_id must be {expected_previous!r}."
            )
        refs = _ordered_reference_ids(
            clip.get("reference_ids", [] if not references else None),
            references,
            f"plan.clips[{index}].reference_ids",
        )
        video_prompt, image_prompt = _authored_clip_prompts(
            clip, index=index, requires_image=requires_image
        )
        if clip.get("window_prompts") is not None:
            raise AuthoringPlanError(
                f"plan.clips[{index}] is an editorial music-video segment; place one authored prompt in video_prompt. "
                "The native timeline helper will preserve the segment ID and pad/trim its render duration."
            )
        plan_row = {
            "video_prompt": video_prompt,
            "agent_source_clip_id": source_id,
            "agent_timeline_id": timeline_id,
            "agent_previous_clip_id": expected_previous,
            "agent_reference_ids": refs,
        }
        if image_prompt is not None:
            plan_row["image_prompt"] = image_prompt
        for key in ("label", "scene_type", "keyframe_prompts", "narrative_role"):
            if clip.get(key) is not None:
                plan_row[key] = clip[key]
        raw_plans.append(plan_row)
        prepared_clip = {
            "start": start / fps,
            "end": end / fps,
            "duration_sec": duration / fps,
            "duration_frames": duration,
            "output_frames": duration,
            "agent_source_clip_id": source_id,
            "agent_timeline_id": timeline_id,
            "agent_previous_clip_id": expected_previous,
            "agent_reference_ids": refs,
            "music_timing_version": 1,
        }
        for key in (
            "timing_basis",
            "bpm",
            "beat_count",
            "opening_beat_times",
            "ending_beat_times",
            "downbeat_count",
            "opening_downbeat_times",
            "music_cues",
            "percussion_activity",
            "section_label",
            "energy",
            "suggested_prompt_hint",
            "dominant_speaker",
            "lyrics",
            "analysis_method",
            "method",
            "transcription_status",
            "beat_detection",
            "vocal_timing",
        ):
            if key in geometry:
                prepared_clip[key] = copy.deepcopy(geometry[key])
        raw_timeline.append(prepared_clip)
        cursor = end
        previous_source_id = source_id
    if cursor != target:
        raise AuthoringPlanError(
            f"The authored editorial timeline covers {cursor} frames, but the current soundtrack has "
            f"{target} frames. Re-author the clip boundaries against the exact source duration."
        )

    from services.director_music_timing import prepare_music_timeline

    prepared_plans, prepared_timeline = prepare_music_timeline(
        raw_plans,
        raw_timeline,
        fps=fps,
        minimum_frames=minimum,
        maximum_frames=maximum,
        frame_step=step,
    )
    native_total = sum(int(item["duration_frames"]) for item in prepared_timeline)
    editorial_total = sum(int(item["output_frames"]) for item in prepared_timeline)
    return {
        "native_params": {
            "prepared_clip_plans": prepared_plans,
            "prepared_planned_clips": prepared_timeline,
        },
        "plan_metadata": {
            "context_id": context_id,
            "workflow": workflow,
            "clip_count": len(prepared_plans),
            "editorial_total_frames": editorial_total,
            "requested_total_frames": target,
            "native_total_frames": native_total,
            "padding_frames": max(0, native_total - editorial_total),
            "final_trim_frames": max(0, native_total - editorial_total),
            "trim_semantics": "trim each native clip to output_frames, preserving the exact source-audio timeline",
            "fps": fps,
            "timing_basis": director_timeline.get("timing_basis", "legacy_duration_only"),
            "continuity_mode": continuity,
            "reference_ids": references,
            "source_clip_ids": ordered_source_ids,
            "authored_by": "external_agent",
        },
    }


def normalize_authored_director_plan(
    context: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize a complete authored Director plan for ``start_pipeline``.

    Story clips are bound to the server's native H3 schedule or must already
    satisfy the current frame profile. Music-video editorial timing remains
    exact and uses Maestro's native pad/trim helper, preserving source IDs.
    """

    context = _mapping(context, "context")
    plan = _mapping(plan, "plan")
    context_id, workflow, params, model_def, timeline = _require_context_binding(context, plan)
    if workflow not in _DIRECTOR_WORKFLOWS or timeline.get("kind") != "director":
        raise AuthoringPlanError("The current context is not a Director authoring context.")
    profile = timeline.get("director_profile")
    if not isinstance(profile, Mapping):
        raise AuthoringPlanError("The current context is missing Director frame limits; refresh it.")
    fps = _number(profile.get("fps"), "Director frame rate")
    minimum = _integer(profile.get("minimum_frames"), "Director minimum shot frames")
    maximum = _integer(profile.get("maximum_frames"), "Director maximum shot frames")
    step = _integer(profile.get("frame_step"), "Director shot frame step")
    expected_refs = _reference_ids(context)
    raw_clips = plan.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise AuthoringPlanError("plan.clips must contain at least one authored clip.")
    director_timeline = timeline.get("director_timeline") or {}
    if director_timeline.get("kind") == "h3_native_story_schedule":
        return _normalize_h3_story_schedule(
            context,
            plan,
            context_id=context_id,
            workflow=workflow,
            params=params,
            model_def=model_def,
            timeline=timeline,
            raw_clips=raw_clips,
        )
    if director_timeline.get("kind") == "exact_source_audio":
        return _normalize_director_music_timeline(
            context,
            plan,
            context_id=context_id,
            workflow=workflow,
            params=params,
            model_def=model_def,
            timeline=timeline,
            raw_clips=raw_clips,
        )

    try:
        from services.director_video_strategy import (
            resolve_shot_image_policy,
            shot_images_required,
        )

        policy = str(context.get("shot_image_policy") or params.get("shot_image_policy") or "")
        if not policy:
            policy = resolve_shot_image_policy(
                model_def,
                params.get("shot_image_guidance"),
                has_visual_references=any(
                    str(item.get("role") or "").casefold() in {"image", "character", "location", "reference"}
                    for item in (context.get("references") or [])
                    if isinstance(item, Mapping)
                ),
            )
    except ImportError:
        policy = str(context.get("shot_image_policy") or params.get("shot_image_policy") or "required")

        def shot_images_required(value: Any) -> bool:
            return str(value).casefold() in {"required", "generate"}

    continuity = _continuity_mode(timeline.get("continuity_mode") or "continuous")
    prepared_plans: list[dict[str, Any]] = []
    prepared_timeline: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    cursor = 0
    prior_clip_id: str | None = None
    for index, raw in enumerate(raw_clips, start=1):
        clip = _mapping(raw, f"plan.clips[{index}]")
        clip_id = _required_text(clip.get("id"), f"plan.clips[{index}].id")
        if clip_id in all_ids:
            raise AuthoringPlanError(f"plan.clips[{index}].id is duplicated: {clip_id!r}.")
        all_ids.add(clip_id)
        start = _integer(clip.get("start_frame"), f"plan.clips[{index}].start_frame", minimum=0)
        end = _integer(clip.get("end_frame"), f"plan.clips[{index}].end_frame")
        duration = _integer(clip.get("duration_frames"), f"plan.clips[{index}].duration_frames")
        if start != cursor or end != start + duration:
            raise AuthoringPlanError(
                f"plan.clips[{index}] must start at frame {cursor} and end at "
                f"start_frame + duration_frames; refresh the timeline or revise this clip."
            )
        if duration < minimum or duration > maximum or (duration - minimum) % step:
            raise AuthoringPlanError(
                f"Director clip {clip_id!r} has {duration} frames; the current model accepts "
                f"{minimum}+{step}n frames up to {maximum}. Re-author its duration for this model."
            )
        refs = _ordered_reference_ids(
            clip.get("reference_ids", [] if not expected_refs else None),
            expected_refs,
            f"plan.clips[{index}].reference_ids",
        )
        expected_previous = prior_clip_id if continuity == "continuous" else None
        if clip.get("previous_clip_id") != expected_previous:
            raise AuthoringPlanError(
                f"plan.clips[{index}].previous_clip_id must be {expected_previous!r} "
                "for this timeline's continuity mode."
            )
        video_prompt = _required_text(clip.get("video_prompt"), f"plan.clips[{index}].video_prompt")
        image_prompt = clip.get("image_prompt")
        if shot_images_required(policy):
            image_prompt = _required_text(image_prompt, f"plan.clips[{index}].image_prompt")
        elif image_prompt is not None:
            image_prompt = _required_text(image_prompt, f"plan.clips[{index}].image_prompt")

        expected_window_geometry = _director_clip_window_geometry(
            context,
            params,
            model_def,
            duration_frames=duration,
            fps=fps,
        )
        windows, provenance = _validate_director_window_prompts(
            context,
            clip,
            clip_index=index,
            clip_start=start,
            clip_end=end,
            references=expected_refs,
            continuity=continuity,
            expected_geometry=expected_window_geometry,
        )
        prepared = {
            "video_prompt": video_prompt,
            "agent_source_clip_id": clip_id,
            "agent_previous_clip_id": expected_previous,
            "agent_reference_ids": refs,
        }
        if image_prompt is not None:
            prepared["image_prompt"] = image_prompt
        if windows is not None:
            prepared["window_prompts"] = windows
            prepared["agent_window_provenance"] = provenance
        for passthrough in ("label", "scene_type", "keyframe_prompts", "narrative_role"):
            value = clip.get(passthrough)
            if value is not None:
                prepared[passthrough] = value
        seconds = duration / fps
        planned = {
            "start": start / fps,
            "end": end / fps,
            "duration_sec": seconds,
            "duration_frames": duration,
            "agent_source_clip_id": clip_id,
            "agent_previous_clip_id": expected_previous,
            "agent_reference_ids": refs,
        }
        prepared_plans.append(prepared)
        prepared_timeline.append(planned)
        cursor = end
        prior_clip_id = clip_id

    return {
        "native_params": {
            "prepared_clip_plans": prepared_plans,
            "prepared_planned_clips": prepared_timeline,
        },
        "plan_metadata": {
            "context_id": context_id,
            "workflow": workflow,
            "clip_count": len(prepared_plans),
            "native_total_frames": cursor,
            "requested_total_frames": profile.get("target_total_frames"),
            "padding_frames": None,
            "final_trim_frames": 0,
            "trim_semantics": "native clip schedule; no silent truncation or duration rewrite",
            "fps": fps,
            "continuity_mode": continuity,
            "reference_ids": expected_refs,
            "authored_by": "external_agent",
        },
    }
