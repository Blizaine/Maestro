"""Reference-driven multi-clip planning for MiniMax H3 Omni.

Ref2VA has no native sliding-window contract. Maestro therefore plans bounded
native clips, runs each with the same canonical references, and joins them as
an editorial sequence. Generated continuity pictures are injected later by
the queue worker and never replace the user's identity references.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from services.h3_story_ledger import H3_STORY_LEDGER_VERSION, plan_h3_story_segments
from services.h3_window_planner import (
    _UNREQUESTED_SPECTACLE_PATTERNS,
    _compact,
    _fallback_plan,
    _infer_camera_coverage,
    _narrative_dialogue_expected,
    _normalized_window_shots,
    _shot_prompt_sentence,
    normalize_h3_camera_coverage,
)


_H3_SEQUENCE_PLANNER_VERSION = 2 + H3_STORY_LEDGER_VERSION


def compute_h3_sequence_clips(
    total_frames: int,
    *,
    min_clip_frames: int = 124,
    max_clip_frames: int = 345,
    frame_step: int = 17,
    fps: float = 24.0,
) -> tuple[list[dict[str, Any]], int]:
    """Partition a requested timeline into balanced valid H3 clip lengths."""

    total = max(1, int(total_frames))
    minimum = max(1, int(min_clip_frames))
    maximum = max(minimum, int(max_clip_frames))
    step = max(1, int(frame_step))
    fps_value = max(1.0, float(fps))
    if total <= maximum:
        count = 1
    else:
        count = int(math.ceil(total / float(maximum)))

    # Every valid H3 clip is minimum + N*step. Choose the smallest aggregate
    # lattice at or above the requested duration, then distribute increments
    # evenly so one clip is not dramatically longer than its neighbors.
    base_sum = count * minimum
    increment_count = max(0, int(math.ceil((total - base_sum) / float(step))))
    max_increments = count * ((maximum - minimum) // step)
    increment_count = min(max_increments, increment_count)
    quotient, remainder = divmod(increment_count, count)
    frame_counts = [
        minimum + (quotient + (1 if index < remainder else 0)) * step
        for index in range(count)
    ]
    frame_counts = [min(maximum, value) for value in frame_counts]

    clips: list[dict[str, Any]] = []
    cursor = 0
    for index, frames in enumerate(frame_counts):
        end = cursor + frames
        clips.append({
            "index": index + 1,
            "frames": frames,
            "start_frame": cursor,
            "end_frame": end,
            "start_seconds": round(cursor / fps_value, 3),
            "end_seconds": round(end / fps_value, 3),
            "duration_seconds": round(frames / fps_value, 3),
        })
        cursor = end
    return clips, max(0, cursor - total)


def h3_sequence_plan_signature(
    prompt: str,
    *,
    model_type: str,
    resolution: str,
    total_frames: int,
    min_clip_frames: int,
    max_clip_frames: int,
    frame_step: int,
    fps: float,
    references: list[dict[str, Any]],
    camera_coverage: str = "auto",
) -> str:
    reference_contract = [
        {
            "type": item.get("type") or item.get("kind"),
            "path": item.get("path"),
            "role": item.get("role"),
            "image_intent": item.get("image_intent"),
            "audio_intent": item.get("audio_intent"),
            "include_audio": item.get("include_audio"),
            "audio_path": item.get("audio_path"),
        }
        for item in (references or [])
        if isinstance(item, dict)
    ]
    payload = {
        "planner_version": _H3_SEQUENCE_PLANNER_VERSION,
        "prompt": str(prompt or "").strip(),
        "model_type": str(model_type or ""),
        "resolution": str(resolution or ""),
        "total_frames": int(total_frames),
        "min_clip_frames": int(min_clip_frames),
        "max_clip_frames": int(max_clip_frames),
        "frame_step": int(frame_step),
        "fps": round(float(fps), 6),
        "references": reference_contract,
        "camera_coverage": normalize_h3_camera_coverage(camera_coverage),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _reference_context(references: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Return relationship, retention, and official task-type summaries."""

    from models.minimax_h3.ref2va import validate_reference_manifest

    # Enhancement is useful before the generation manifest is complete. The
    # actual Ref2VA generation path still performs strict file + visual-media
    # validation; planning only needs whatever labels are currently present.
    items = validate_reference_manifest(
        references,
        require_files=False,
        require_visual=False,
        allow_empty=True,
    )
    picture = video = audio = 0
    relationships: list[str] = []
    retention: list[str] = []
    task_types = ["reference generation"]
    for item in items:
        kind = item["type"]
        role = item.get("role") or f"the supplied {kind} reference"
        if kind == "image":
            picture += 1
            intent = item.get("image_intent", "identity")
            if intent == "composition":
                relationships.append(
                    f"<Picture {picture}> is a composition and blocking reference for {role}, not an identity source."
                )
                retention.append(f"<Picture {picture}>: weak_reference")
            elif intent == "scene":
                relationships.append(f"<Picture {picture}> defines the location and environment for {role}.")
                retention.append(f"<Picture {picture}>: partially_preserved")
            elif intent == "style":
                relationships.append(f"<Picture {picture}> is a visual-style reference for {role}.")
                retention.append(f"<Picture {picture}>: weak_reference")
            else:
                relationships.append(
                    f"<Picture {picture}> defines the identity and appearance of {role}; reject its source background, framing, and pose."
                )
                retention.append(f"<Picture {picture}>: fully_preserved for identity and appearance")
        elif kind == "video":
            video += 1
            relationships.append(
                f"<Video {video}> provides motion, camera, scene, or timing reference for {role}."
            )
            retention.append(f"<Video {video}>: partially_preserved")
            if (item.get("has_audio") or item.get("audio_path")) and item.get("include_audio", True):
                audio += 1
                relationships.append(
                    f"<Audio {audio}> is the soundtrack paired with <Video {video}> and keeps its audible timeline."
                )
                retention.append(f"<Audio {audio}>: partially_copy")
                if "audio reuse" not in task_types:
                    task_types.append("audio reuse")
        else:
            audio += 1
            intent = item.get("audio_intent", "voice")
            if intent == "drive":
                relationships.append(
                    f"<Audio {audio}> is performance-driving audio for {role}; preserve its audible timeline."
                )
                retention.append(f"<Audio {audio}>: partially_copy")
                if "audio reuse" not in task_types:
                    task_types.append("audio reuse")
            elif intent == "style":
                relationships.append(
                    f"<Audio {audio}> supplies sound, music, rhythm, or texture style for {role}, without copying its signal."
                )
                retention.append(f"<Audio {audio}>: weak_reference")
                if "audio reference" not in task_types:
                    task_types.append("audio reference")
            else:
                relationships.append(
                    f"<Audio {audio}> supplies voice timbre, emotion, and delivery for {role}, without copying its words or timing."
                )
                retention.append(f"<Audio {audio}>: reference")
                if "audio reference" not in task_types:
                    task_types.append("audio reference")
    return " ".join(relationships), "; ".join(retention), " + ".join(task_types)


def _schema(clip_count: int) -> dict[str, Any]:
    dialogue = {
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "speaker_id": {"type": "string"},
            "language": {"type": "string"},
            "delivery": {"type": "string"},
            "action": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["speaker", "speaker_id", "language", "delivery", "action", "text"],
        "additionalProperties": False,
    }
    shot = {
        "type": "object",
        "properties": {
            "shot": {"type": "integer"},
            "start_seconds": {"type": "number"},
            "end_seconds": {"type": "number"},
            "transition": {"type": "string"},
            "framing": {"type": "string"},
            "camera": {"type": "string"},
            "action": {"type": "string"},
            "dialogue": {"type": "array", "items": dialogue},
            "sound_effects": {"type": "string"},
        },
        "required": [
            "shot", "start_seconds", "end_seconds", "transition", "framing",
            "camera", "action", "dialogue", "sound_effects",
        ],
        "additionalProperties": False,
    }
    clip = {
        "type": "object",
        "properties": {
            "clip": {"type": "integer"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "opening_state": {"type": "string"},
            "coverage": {"type": "string"},
            "pacing": {"type": "string"},
            "shots": {"type": "array", "minItems": 1, "maxItems": 4, "items": shot},
            "closing_state": {"type": "string"},
        },
        "required": [
            "clip", "title", "summary", "opening_state", "coverage",
            "pacing", "shots", "closing_state",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "subject_definitions": {"type": "string"},
            "retention_analysis": {"type": "string"},
            "setting_continuity": {"type": "string"},
            "visual_style": {"type": "string"},
            "ambient_audio": {"type": "string"},
            "music": {"type": "string"},
            "clips": {
                "type": "array",
                "minItems": clip_count,
                "maxItems": clip_count,
                "items": clip,
            },
        },
        "required": [
            "subject_definitions", "retention_analysis", "setting_continuity",
            "visual_style", "ambient_audio", "music", "clips",
        ],
        "additionalProperties": False,
    }


def _plan_violations(
    source_prompt: str,
    plan: dict[str, Any] | None,
    *,
    clip_count: int,
    expect_dialogue: bool,
) -> list[str]:
    if not isinstance(plan, dict):
        return ["invalid plan"]
    clips = plan.get("clips") or []
    violations: list[str] = []
    if len(clips) != clip_count:
        violations.append(f"returned {len(clips)} clips instead of {clip_count}")
    lowered_source = str(source_prompt or "").casefold()
    lowered_plan = json.dumps(plan, ensure_ascii=False).casefold()
    for pattern in _UNREQUESTED_SPECTACLE_PATTERNS:
        match = re.search(pattern, lowered_plan, flags=re.IGNORECASE)
        if match and not re.search(pattern, lowered_source, flags=re.IGNORECASE):
            violations.append(f"invented unrequested power/effect: {match.group(0).strip()}")
            break
    if any(
        not isinstance(item, dict)
        or not any(
            isinstance(shot, dict) and str(shot.get("action") or "").strip()
            for shot in (item.get("shots") or [])
        )
        for item in clips
    ):
        violations.append("one or more clips contain no visible shot action")
    if expect_dialogue and not any(
        isinstance(line, dict) and str(line.get("text") or "").strip()
        for item in clips if isinstance(item, dict)
        for shot in (item.get("shots") or []) if isinstance(shot, dict)
        for line in (shot.get("dialogue") or [])
    ):
        violations.append("left a long named-character interaction entirely mute")
    return violations


def compile_h3_reference_sequence_prompts(
    plan: dict[str, Any],
    clips: list[dict[str, Any]],
    *,
    reference_relationships: str,
    default_retention: str,
    task_types: str,
) -> list[dict[str, Any]]:
    planned_clips = plan.get("clips") if isinstance(plan, dict) else None
    if not isinstance(planned_clips, list) or len(planned_clips) != len(clips):
        raise ValueError("H3 Omni sequence plan does not match its clip geometry.")

    subjects = _compact(plan.get("subject_definitions") or reference_relationships, 700)
    if reference_relationships and reference_relationships not in subjects:
        subjects = f"{subjects}. {reference_relationships}" if subjects else reference_relationships
    retention = _compact(plan.get("retention_analysis") or default_retention, 600)
    if default_retention and default_retention not in retention:
        retention = f"{retention}; {default_retention}" if retention else default_retention
    setting = _compact(plan.get("setting_continuity"), 260)
    style = _compact(plan.get("visual_style"), 220)
    ambient = _compact(plan.get("ambient_audio") or "Natural location ambience", 190)
    music = _compact(plan.get("music") or "N/A", 130)
    speaker_ids: dict[str, str] = {}
    compiled: list[dict[str, Any]] = []

    for geometry, item in zip(clips, planned_clips):
        if not isinstance(item, dict):
            raise ValueError(f"H3 Omni sequence clip {geometry['index']} is invalid.")
        duration = float(geometry["duration_seconds"])
        shots = _normalized_window_shots(item, duration)
        if not shots:
            raise ValueError(f"H3 Omni sequence clip {geometry['index']} has no shots.")
        coverage = _compact(item.get("coverage") or "cinematic editorial coverage", 90)
        pacing = _compact(item.get("pacing") or "natural real-time pacing", 110)
        opening = _compact(item.get("opening_state") or "The scene begins in a clear composition", 210)
        closing = _compact(item.get("closing_state") or "The beat settles in a clear final composition", 210)
        preamble = " ".join(part for part in (
            f"{setting}." if setting else "",
            f"{style}." if style else "",
            f"Opening state: {opening}.",
            f"Coverage is {coverage}; pacing is {pacing}. Slow motion occurs only when explicitly requested.",
        ) if part)
        detailed = " ".join(
            _shot_prompt_sentence(
                shot,
                speaker_ids=speaker_ids,
                preamble=preamble if shot_index == 0 else "",
            )
            for shot_index, shot in enumerate(shots)
        )
        has_dialogue = any(
            isinstance(line, dict) and str(line.get("text") or "").strip()
            for shot in shots for line in (shot.get("dialogue") or [])
        )
        detailed += (
            " Only the tagged lines are spoken once in order. Outside those lines, everyone is silent with mouths closed; no background voices or gibberish."
            if has_dialogue
            else " Everyone remains silent with mouths closed; no voices, muttering, or gibberish."
        )
        detailed += f" The final composition is {closing}."
        effects = "; ".join(
            effect for effect in (
                _compact(shot.get("sound_effects"), 130) for shot in shots
            ) if effect.casefold() not in {"", "n/a", "none"}
        )
        soundscape = ambient + (f". Synchronized effects: {effects}" if effects else "")
        summary = _compact(item.get("summary") or item.get("title") or "The requested story advances", 240)
        if not summary.startswith("["):
            summary = f"[{task_types}] {summary}"
        prompt = "\n\n".join([
            f"subject_definitions: {subjects}",
            f"summary: {summary}",
            f"retention_analysis: {retention}",
            f"detailed_description: {detailed}",
            f"overall_soundscape: {soundscape}.",
            f"non_diegetic_music: {music}",
        ])
        compiled.append({
            **geometry,
            "title": _compact(item.get("title") or f"Clip {geometry['index']}", 90),
            "opening_state": opening,
            "closing_state": closing,
            "coverage": coverage,
            "pacing": pacing,
            "shot_count": len(shots),
            "shots": shots,
            "prompt": prompt,
        })
    return compiled


def _fallback_sequence_plan(
    prompt: str,
    clips: list[dict[str, Any]],
    *,
    reference_relationships: str,
    default_retention: str,
    camera_coverage: str,
) -> dict[str, Any]:
    fallback = _fallback_plan(
        prompt,
        len(clips),
        window_durations=[float(item["duration_seconds"]) for item in clips],
        camera_coverage=camera_coverage,
    )
    planned_clips = []
    for window in fallback["windows"]:
        planned_clips.append({
            "clip": window["window"],
            "title": window["title"],
            "summary": window["title"],
            "opening_state": (
                fallback["initial_state"]
                if window["window"] == 1
                else "The established story resumes in a fresh editorial composition"
            ),
            "coverage": window["coverage"],
            "pacing": window["pacing"],
            "shots": window["shots"],
            "closing_state": window["closing_state"],
        })
    return {
        "subject_definitions": reference_relationships,
        "retention_analysis": default_retention,
        "setting_continuity": fallback["setting_continuity"],
        "visual_style": fallback["visual_continuity"],
        "ambient_audio": fallback["ambient_audio"],
        "music": fallback["music"],
        "clips": planned_clips,
    }


def plan_h3_reference_sequence(
    prompt: str,
    *,
    model_type: str,
    resolution: str,
    total_frames: int,
    references: list[dict[str, Any]],
    min_clip_frames: int = 124,
    max_clip_frames: int = 345,
    frame_step: int = 17,
    fps: float = 24.0,
    camera_coverage: str = "auto",
    image_paths: list[str] | None = None,
    nsfw: bool = False,
) -> dict[str, Any]:
    """Plan independent H3 Omni clips that share canonical references."""

    camera_coverage = normalize_h3_camera_coverage(camera_coverage)
    clips, trim_tail_frames = compute_h3_sequence_clips(
        total_frames,
        min_clip_frames=min_clip_frames,
        max_clip_frames=max_clip_frames,
        frame_step=frame_step,
        fps=fps,
    )
    relationships, default_retention, task_types = _reference_context(references)
    signature = h3_sequence_plan_signature(
        prompt,
        model_type=model_type,
        resolution=resolution,
        total_frames=total_frames,
        min_clip_frames=min_clip_frames,
        max_clip_frames=max_clip_frames,
        frame_step=frame_step,
        fps=fps,
        references=references,
        camera_coverage=camera_coverage,
    )
    if len(clips) <= 1:
        return {
            "source_prompt": str(prompt or ""),
            "signature": signature,
            "planned_by": "not_needed",
            "plan_kind": "reference_sequence",
            "camera_coverage": camera_coverage,
            "total_frames": int(total_frames),
            "window_frames": int(max_clip_frames),
            "window_count": 1,
            "per_clip_frames": [clips[0]["frames"]],
            "trim_tail_frames": trim_tail_frames,
            "windows": [],
            "window_prompts": [],
        }

    expect_dialogue = _narrative_dialogue_expected(prompt, len(clips))
    resolved_coverage = _infer_camera_coverage(prompt, camera_coverage)
    story_ledger: dict[str, Any] | None = None
    try:
        staged = plan_h3_story_segments(
            prompt,
            segment_durations=[float(item["duration_seconds"]) for item in clips],
            mode="reference_sequence",
            camera_coverage=resolved_coverage,
            reference_context=" ".join(
                part for part in (relationships, default_retention) if part
            ),
            expect_dialogue=expect_dialogue,
            image_paths=image_paths,
            nsfw=nsfw,
        )
        planned_by = staged["planned_by"]
        story_ledger = staged["ledger"]
        planned_clips = []
        for index, segment in enumerate(staged["segments"]):
            planned_clips.append({
                **segment,
                "clip": index + 1,
            })
        plan = {
            "subject_definitions": story_ledger.get("subject_continuity", ""),
            "retention_analysis": default_retention,
            "setting_continuity": story_ledger.get("setting_continuity", ""),
            "visual_style": story_ledger.get("visual_continuity", ""),
            "ambient_audio": story_ledger.get("ambient_audio", ""),
            "music": story_ledger.get("music", "N/A"),
            "clips": planned_clips,
        }
        compiled = compile_h3_reference_sequence_prompts(
            plan,
            clips,
            reference_relationships=relationships,
            default_retention=default_retention,
            task_types=task_types,
        )
    except Exception as error:
        print(f"[MiniMax H3 Omni] Sequence planner fallback: {error}")
        planned_by = "deterministic_fallback"
        plan = _fallback_sequence_plan(
            prompt,
            clips,
            reference_relationships=relationships,
            default_retention=default_retention,
            camera_coverage=camera_coverage,
        )
        compiled = compile_h3_reference_sequence_prompts(
            plan,
            clips,
            reference_relationships=relationships,
            default_retention=default_retention,
            task_types=task_types,
        )

    return {
        "source_prompt": str(prompt or ""),
        "signature": signature,
        "planned_by": planned_by,
        "plan_kind": "reference_sequence",
        "camera_coverage": camera_coverage,
        "total_frames": int(total_frames),
        "window_frames": int(max_clip_frames),
        "window_count": len(compiled),
        "resolution": str(resolution or ""),
        "model_type": str(model_type or ""),
        "per_clip_frames": [int(item["frames"]) for item in clips],
        "trim_tail_frames": int(trim_tail_frames),
        "subject_continuity": plan.get("subject_definitions", ""),
        "setting_continuity": plan.get("setting_continuity", ""),
        "story_ledger": story_ledger,
        "windows": compiled,
        "window_prompts": [item["prompt"] for item in compiled],
    }
