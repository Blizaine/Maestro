"""Reference-driven multi-clip planning for MiniMax H3 Omni.

Ref2VA can run either as native continuation windows or as independent
editorial clips. Native continuation carries recent motion and synchronized
audio while repeating the same canonical references; hard-cut mode retains
Maestro's independent-clip path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from services.h3_story_ledger import (
    H3_STORY_LEDGER_VERSION,
    extract_h3_source_intent,
    plan_h3_story_segments,
    recover_h3_plain_story,
)
from services.h3_window_planner import (
    _UNREQUESTED_SPECTACLE_PATTERNS,
    _compact,
    _fallback_plan,
    _infer_camera_coverage,
    _narrative_dialogue_expected,
    _normalized_window_shots,
    _shot_prompt_sentence,
    compute_h3_window_boundaries,
    normalize_h3_camera_coverage,
)


_H3_SEQUENCE_PLANNER_VERSION = 4 + H3_STORY_LEDGER_VERSION
_H3_CLIP_BOUNDARY = "\n---CLIP_BOUNDARY---\n"


def resolve_h3_sequence_source_prompt(
    current_prompt: str,
    cached_plan: dict[str, Any] | None,
    cached_prompts: list[str] | None,
) -> str:
    """Recover the user's story prompt from a serialized H3 sequence.

    Generation sidecars retain the compiled per-window Context-IR so a run is
    reproducible.  In hard-cut mode that compiled text also becomes the task's
    runtime ``prompt``.  If settings are reloaded and the sequence geometry
    changes, feeding that runtime prompt back to the planner makes reference
    declarations look like story actions.  Restore ``source_prompt`` only
    when the current value is demonstrably the cached runtime serialization;
    a genuinely edited user prompt must always win.
    """

    current = str(current_prompt or "").replace("\r\n", "\n").replace(
        "\r", "\n"
    ).strip()
    if not isinstance(cached_plan, dict):
        return current
    source = str(cached_plan.get("source_prompt") or "").replace(
        "\r\n", "\n"
    ).replace("\r", "\n").strip()
    if not source:
        return current
    if current == source:
        return source
    if not isinstance(cached_prompts, list):
        return current
    prompts = [
        str(item).replace("\r\n", "\n").replace("\r", "\n").strip()
        for item in cached_prompts
        if isinstance(item, str) and item.strip()
    ]
    if not prompts:
        return current
    serialized = _H3_CLIP_BOUNDARY.join(prompts)
    if current == serialized or (len(prompts) == 1 and current == prompts[0]):
        return source
    return current


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


def compute_h3_native_sequence_windows(
    total_frames: int,
    *,
    window_frames: int = 345,
    overlap_frames: int = 18,
    fps: float = 24.0,
) -> list[dict[str, Any]]:
    """Describe the committed timeline owned by native Ref2VA passes."""

    boundaries = compute_h3_window_boundaries(
        total_frames,
        window_frames,
        fps=fps,
        overlap_frames=overlap_frames,
        discard_frames=0,
    )
    clips: list[dict[str, Any]] = []
    for item in boundaries:
        frames = max(1, int(item["end_frame"]) - int(item["start_frame"]))
        clips.append({
            **item,
            "frames": frames,
            "duration_seconds": round(frames / max(1.0, float(fps)), 3),
        })
    return clips


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
    overlap_frames: int = 0,
    native_continuation: bool = False,
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
        "overlap_frames": int(overlap_frames),
        "native_continuation": bool(native_continuation),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def reviewed_h3_sequence_plan_matches(
    plan: Any,
    window_prompts: Any,
    *,
    source_prompt: str,
    model_type: str,
    resolution: str,
    geometry: list[dict[str, Any]],
    window_frames: int,
    camera_coverage: str,
    overlap_frames: int,
    native_continuation: bool,
) -> bool:
    """Return whether a visible Omni sequence plan still fits this request."""

    if not isinstance(plan, dict) or not isinstance(window_prompts, (list, tuple)):
        return False
    prompts = [
        item.strip()
        for item in window_prompts
        if isinstance(item, str) and item.strip()
    ]
    windows = plan.get("windows")
    if not prompts or not isinstance(windows, list):
        return False
    if len(prompts) != len(geometry) or len(windows) != len(geometry):
        return False
    if str(plan.get("plan_kind") or "") != "reference_sequence":
        return False
    if str(plan.get("source_prompt") or "").strip() != str(source_prompt or "").strip():
        return False
    if str(plan.get("model_type") or "") != str(model_type or ""):
        return False
    if str(plan.get("resolution") or "") != str(resolution or ""):
        return False
    if int(plan.get("window_frames") or 0) != int(window_frames):
        return False
    if normalize_h3_camera_coverage(plan.get("camera_coverage")) != normalize_h3_camera_coverage(camera_coverage):
        return False
    if bool(plan.get("native_continuation")) != bool(native_continuation):
        return False
    if int(plan.get("overlap_frames") or 0) != int(overlap_frames):
        return False
    if [int(value) for value in (plan.get("per_clip_frames") or [])] != [
        int(item.get("frames") or 0) for item in geometry
    ]:
        return False

    for index, (window, expected, prompt) in enumerate(
        zip(windows, geometry, prompts),
        start=1,
    ):
        if not isinstance(window, dict):
            return False
        try:
            geometry_matches = (
                int(window.get("index") or index) == int(expected.get("index") or index)
                and int(window.get("start_frame") or 0) == int(expected.get("start_frame") or 0)
                and int(window.get("end_frame") or 0) == int(expected.get("end_frame") or 0)
            )
        except (TypeError, ValueError):
            return False
        if not geometry_matches or str(window.get("prompt") or "").strip() != prompt:
            return False
    return True


def parse_h3_manual_sequence_prompts(
    prompt: str,
    *,
    expected_count: int,
    native_continuation: bool,
) -> list[str]:
    """Parse and strictly validate one user-authored prompt per H3 pass."""

    lines = [
        line.strip()
        for line in str(prompt or "").replace("\r\n", "\n").replace(
            "\r", "\n"
        ).split("\n")
        if line.strip()
    ]
    expected = max(1, int(expected_count))
    if len(lines) != expected:
        unit = "window" if native_continuation else "clip"
        raise ValueError(
            "Manual MiniMax H3 Omni sequence needs exactly "
            f"{expected} non-empty prompt "
            f"{'line' if expected == 1 else 'lines'} "
            f"({unit} 1 through {unit} {expected}); received {len(lines)}. "
            "Add or remove prompt lines, or adjust Duration / Sequence Window "
            "Length."
        )
    return lines


def build_manual_h3_reference_sequence_plan(
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
    overlap_frames: int = 0,
    native_continuation: bool = False,
) -> dict[str, Any]:
    """Build a reproducible Ref2VA plan without invoking an LLM.

    Each non-empty source line is passed to the matching native continuation
    window (or independent hard-cut clip) exactly as authored.
    """

    if native_continuation:
        clips = compute_h3_native_sequence_windows(
            total_frames,
            window_frames=max_clip_frames,
            overlap_frames=overlap_frames,
            fps=fps,
        )
        trim_tail_frames = 0
    else:
        clips, trim_tail_frames = compute_h3_sequence_clips(
            total_frames,
            min_clip_frames=min_clip_frames,
            max_clip_frames=max_clip_frames,
            frame_step=frame_step,
            fps=fps,
        )

    prompts = parse_h3_manual_sequence_prompts(
        prompt,
        expected_count=len(clips),
        native_continuation=native_continuation,
    )
    source_prompt = "\n".join(prompts)
    camera_coverage = normalize_h3_camera_coverage(camera_coverage)
    signature = h3_sequence_plan_signature(
        source_prompt,
        model_type=model_type,
        resolution=resolution,
        total_frames=total_frames,
        min_clip_frames=min_clip_frames,
        max_clip_frames=max_clip_frames,
        frame_step=frame_step,
        fps=fps,
        references=references,
        camera_coverage=camera_coverage,
        overlap_frames=overlap_frames,
        native_continuation=native_continuation,
    )
    label = "Window" if native_continuation else "Clip"
    windows = [
        {
            **geometry,
            "title": f"Manual {label} {geometry['index']}",
            "opening_state": "",
            "closing_state": "",
            "coverage": "manual",
            "pacing": "manual",
            "shot_count": 0,
            "prompt": window_prompt,
        }
        for geometry, window_prompt in zip(clips, prompts)
    ]
    return {
        "source_prompt": source_prompt,
        "signature": signature,
        "planned_by": "manual",
        "plan_kind": "reference_sequence",
        "camera_coverage": camera_coverage,
        "total_frames": int(total_frames),
        "window_frames": int(max_clip_frames),
        "window_count": len(windows),
        "resolution": str(resolution or ""),
        "model_type": str(model_type or ""),
        "per_clip_frames": [int(item["frames"]) for item in clips],
        "trim_tail_frames": int(trim_tail_frames),
        "overlap_frames": int(overlap_frames),
        "native_continuation": bool(native_continuation),
        "subject_continuity": "",
        "setting_continuity": "",
        "story_ledger": None,
        "windows": windows,
        "window_prompts": prompts,
    }


def _reference_context(references: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Return relationship, retention, and official task-type summaries."""

    from models.minimax_h3.reference_manifest import validate_reference_manifest

    # Enhancement is useful before the generation manifest is complete. The
    # actual Ref2VA generation path still performs strict file + visual-media
    # validation; planning only needs whatever labels are currently present.
    items = validate_reference_manifest(
        references,
        require_files=False,
        require_visual=False,
        allow_empty=True,
    )
    picture = video = audio = subject = 0
    relationships: list[str] = []
    retention: list[str] = []
    task_types = ["reference generation"]
    role_subjects: dict[str, int] = {}
    character_subjects: dict[str, int] = {}
    pending_voice: list[tuple[int, dict, str]] = []
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
                subject += 1
                relationships.append(
                    f"<Subject {subject}> is the environment and location for "
                    f"{role}, defined by <Picture {picture}>."
                )
                retention.append(
                    f"<Subject {subject}>: partially_preserved - preserve the "
                    "referenced architecture, materials, lighting context, and "
                    "location identity."
                )
            elif intent == "style":
                subject += 1
                relationships.append(
                    f"<Subject {subject}> is the visual treatment for {role}, "
                    f"guided by <Picture {picture}>."
                )
                retention.append(
                    f"<Subject {subject}>: weak_reference - retain broad "
                    "similarity in medium, palette, lighting language, and texture."
                )
            else:
                subject += 1
                role_subjects[str(role).strip().casefold()] = subject
                character_key = str(item.get("library_character_id") or "").strip()
                if character_key:
                    character_subjects[character_key] = subject
                relationships.append(
                    f"<Subject {subject}> is {role}, whose identity and appearance "
                    f"come from <Picture {picture}>; reject the picture's source "
                    "background, framing, composition, and pose."
                )
                retention.append(
                    f"<Subject {subject}>: fully_preserved - preserve the identity "
                    f"and appearance defined by <Picture {picture}>."
                )
        elif kind == "video":
            video += 1
            video_intent = item.get("video_intent", "motion")
            if video_intent == "character":
                subject += 1
                role_subjects[str(role).strip().casefold()] = subject
                character_key = str(item.get("library_character_id") or "").strip()
                if character_key:
                    character_subjects[character_key] = subject
                relationships.append(
                    f"<Subject {subject}> is {role}, whose identity, appearance, and characteristic "
                    f"motion come from <Video {video}>; reject its source background, framing, camera, "
                    "edit rhythm, opening frame, and action."
                )
                retention.append(
                    f"<Subject {subject}>: fully_preserved - preserve the identity and appearance "
                    f"defined by <Video {video}> while generating each requested target action."
                )
            elif video_intent == "scene":
                relationships.append(
                    f"<Video {video}> provides environment, lighting, and scene continuity for {role}; "
                    "do not copy incidental people as target identities."
                )
                retention.append(f"<Video {video}>: partially_preserved")
            else:
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
            intent = item.get("audio_intent", "voice")
            if intent == "drive":
                relationships.append(
                    f"The exact target soundtrack supplies the performance and timing for {role}; "
                    "preserve its waveform and audible timeline exactly and synchronize visible action to it."
                )
                # This is Maestro's target conditioning track, not an Omni
                # reference tensor, so it intentionally has no <Audio N>
                # label in the full-reference media manifest.
                retention.append("Exact target soundtrack: fully_preserved")
                if "audio reuse" not in task_types:
                    task_types.append("audio reuse")
            elif intent == "style":
                audio += 1
                relationships.append(
                    f"<Audio {audio}> supplies sound, music, rhythm, or texture style for {role}, without copying its signal."
                )
                retention.append(f"<Audio {audio}>: weak_reference")
                if "audio reference" not in task_types:
                    task_types.append("audio reference")
            else:
                audio += 1
                character_key = str(item.get("library_character_id") or "").strip()
                mapped_subject = (
                    character_subjects.get(character_key)
                    if character_key else role_subjects.get(str(role).strip().casefold())
                )
                if mapped_subject is None and (character_key or str(role).strip()):
                    pending_voice.append((audio, item, role))
                    continue
                target = (
                    f"<Subject {mapped_subject}>"
                    if mapped_subject else str(role)
                )
                relationships.append(
                    f"<Audio {audio}> supplies voice timbre, emotion, and delivery "
                    f"for {target}, without copying its words or timing."
                )
                retention.append(f"<Audio {audio}>: reference")
                if "audio reference" not in task_types:
                    task_types.append("audio reference")
    for audio_index, item, role in pending_voice:
        character_key = str(item.get("library_character_id") or "").strip()
        mapped_subject = (
            character_subjects.get(character_key)
            if character_key else role_subjects.get(str(role).strip().casefold())
        )
        target = f"<Subject {mapped_subject}>" if mapped_subject else str(role)
        relationships.append(
            f"<Audio {audio_index}> supplies voice timbre, emotion, and delivery "
            f"for {target}, without copying its words or timing."
        )
        retention.append(f"<Audio {audio_index}>: reference")
        if "audio reference" not in task_types:
            task_types.append("audio reference")
    return " ".join(relationships), "; ".join(retention), " + ".join(task_types)


def _ref2va_style_opening(style: str) -> str:
    """Return MiniMax's required pre-[Shot 1] full-reference style sentence."""

    value = _compact(style, 220).rstrip(" .")
    if not value:
        return (
            "The target video maintains the requested visual style, lighting, "
            "color, and cinematic texture."
        )
    if re.match(r"^(?:the|this)\s+target\s+video\b", value, re.IGNORECASE):
        return f"{value}."
    return f"The target video uses {value[0].lower() + value[1:]}."


def _officialize_subject_definitions(value: str) -> str:
    """Promote simple planner speaker mappings to official Subject labels."""

    text = deduped = str(value or "").strip()
    if not text or re.search(r"<Subject\s+\d+>", text, re.IGNORECASE):
        return text
    pairs = re.findall(
        r"\b(S\d+)\s+is\s+([^;,.]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not pairs:
        return text
    definitions = [
        f"<Subject {index}> is {name.strip()} ({speaker.upper()})."
        for index, (speaker, name) in enumerate(pairs, start=1)
        if name.strip()
    ]
    if not definitions:
        return text
    # The promoted definitions carry the same information more explicitly.
    # Remove only the narrow legacy "S1 is Name" clauses and retain unrelated
    # planner context.
    deduped = re.sub(
        r"(?i)(?:stable\s+speaking\s+identit(?:y|ies)\s*:\s*)?"
        r"S\d+\s+is\s+[^;,.]+[;,.]?\s*",
        "",
        deduped,
    ).strip(" ;,.")
    return " ".join([*definitions, deduped]).strip()


def _label_ref2va_subjects_in_description(
    description: str,
    subject_definitions: str,
) -> str:
    """Place each reusable Subject label at its first clear visible use."""

    result = str(description or "")
    bindings: list[tuple[str, str]] = []
    for match in re.finditer(
        r"<Subject\s+(\d+)>\s+is\s+(.+?)"
        r"(?=\s+\(S\d+(?:,S\d+)*\)|,\s+(?:whose|with|defined|from)\b|[;.]|$)",
        str(subject_definitions or ""),
        flags=re.IGNORECASE,
    ):
        label = f"<Subject {int(match.group(1))}>"
        name = match.group(2).strip(" ,;:.-")
        if not name or name.casefold().startswith((
            "the visual style", "the visual treatment", "the environment",
        )):
            continue
        bindings.append((label, name))

    missing: list[str] = []
    for label, name in bindings:
        if label in result:
            continue
        pattern = re.compile(
            rf"(?<![\w>]){re.escape(name)}(?![\w<])",
            flags=re.IGNORECASE,
        )
        result, count = pattern.subn(f"{label} ({name})", result, count=1)
        if not count:
            missing.append(f"{label} ({name}) is visible in the opening composition.")
    if missing:
        shot = re.search(r"\[Shot\s+1\]", result, flags=re.IGNORECASE)
        if shot:
            result = (
                f"{result[:shot.end()]} {' '.join(missing)} "
                f"{result[shot.end():].lstrip()}"
            )
        else:
            result = f"{' '.join(missing)} {result}".strip()
    return result


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

    reference_slot_pattern = re.compile(
        r"<(?:Picture|Video|Audio)\s+\d+>",
        flags=re.IGNORECASE,
    )

    def dedupe_sentences(text: str) -> str:
        clauses = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        kept: list[str] = []
        seen: set[str] = set()
        for clause in clauses:
            clause = clause.strip()
            key = re.sub(r"[^a-z0-9<>]+", " ", clause.casefold()).strip()
            if clause and key not in seen:
                kept.append(clause)
                seen.add(key)
        return " ".join(kept)

    # The model may echo Maestro's canonical Picture/Video/Audio contracts,
    # sometimes twice and sometimes with subtly different wording. Treat
    # those contracts as application-owned data: retain only the model's
    # subject/voice identity mapping, then append the canonical relationships
    # exactly once. This is structural rather than wording-based deduplication.
    raw_subjects = str(plan.get("subject_definitions") or "").strip()
    subject_identity_clauses: list[str] = []
    for clause in re.split(r"(?<=[.!?])\s+", raw_subjects):
        match = reference_slot_pattern.search(clause)
        if match and not re.search(r"<Subject\s+\d+>", clause, re.IGNORECASE):
            prefix = clause[:match.start()].strip(" ;,.-")
            if prefix:
                subject_identity_clauses.append(prefix + ".")
            continue
        if clause.strip():
            subject_identity_clauses.append(clause.strip())
    subjects = _officialize_subject_definitions(_compact(
        dedupe_sentences(" ".join(
            part for part in (
                " ".join(subject_identity_clauses),
                dedupe_sentences(reference_relationships),
            ) if part
        )),
        1100,
    ))

    # Reference retention is likewise owned by the manifest. Keep any
    # non-reference analysis the planner supplied, then add each canonical
    # slot contract once.
    raw_plan_retention = str(plan.get("retention_analysis") or "").strip()
    non_reference_retention = "; ".join(
        clause.strip()
        for clause in re.split(r"\s*;\s*", raw_plan_retention)
        if clause.strip() and not reference_slot_pattern.search(clause)
    )
    raw_retention = "; ".join(
        part for part in (non_reference_retention, str(default_retention or "").strip())
        if part
    )
    retention_clauses: list[str] = []
    seen_retention: set[str] = set()
    for clause in re.split(r"\s*;\s*", raw_retention):
        key = re.sub(r"[^a-z0-9<>]+", " ", clause.casefold()).strip()
        if clause and key not in seen_retention:
            retention_clauses.append(clause.strip())
            seen_retention.add(key)
    retention = _compact("; ".join(retention_clauses), 600)
    setting = _compact(plan.get("setting_continuity"), 260)
    style = _compact(plan.get("visual_style"), 220)
    ambient = _compact(plan.get("ambient_audio") or "Natural location ambience", 190)
    music = _compact(plan.get("music") or "N/A", 130)
    source_intent = (
        plan.get("source_intent")
        if isinstance(plan.get("source_intent"), dict)
        else {}
    )
    requested_nonverbal = _compact(
        plan.get("requested_nonverbal_vocals")
        or source_intent.get("requested_nonverbal_vocals"),
        180,
    )
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
        pacing = _compact(item.get("pacing") or "natural real-time pacing", 180)
        opening = _compact(item.get("opening_state") or "The scene begins in a clear composition", 210)
        closing = _compact(item.get("closing_state") or "The beat settles in a clear final composition", 210)
        pacing_sentence = f"Coverage is {coverage}; pacing is {pacing}."
        if "slow motion" not in pacing.casefold():
            pacing_sentence += " Slow motion occurs only when explicitly requested."
        preamble = " ".join(part for part in (
            f"{setting}." if setting else "",
            f"Opening state: {opening}.",
            pacing_sentence,
        ) if part)
        shot_timeline = " ".join(
            _shot_prompt_sentence(
                shot,
                speaker_ids=speaker_ids,
                preamble=preamble if shot_index == 0 else "",
            )
            for shot_index, shot in enumerate(shots)
        )
        detailed = f"{_ref2va_style_opening(style)} {shot_timeline}".strip()
        detailed = _label_ref2va_subjects_in_description(detailed, subjects)
        has_dialogue = any(
            isinstance(line, dict) and str(line.get("text") or "").strip()
            for shot in shots for line in (shot.get("dialogue") or [])
        )
        nonverbal_value = (
            requested_nonverbal.partition(":")[2]
            if ":" in requested_nonverbal else requested_nonverbal
        )
        nonverbal_terms = [
            term.strip().casefold()
            for term in nonverbal_value.split(",")
            if term.strip()
        ]
        clip_performance = " ".join(
            str(shot.get(field) or "")
            for shot in shots
            for field in ("action", "sound_effects")
        ).casefold()
        local_nonverbal = (
            requested_nonverbal
            if any(term in clip_performance for term in nonverbal_terms)
            else ""
        )
        nonverbal_clause = f" {local_nonverbal}." if local_nonverbal else ""
        detailed += (
            " Only the tagged words are spoken once in order. Outside those lines, "
            "there are no additional spoken words, muttering, or gibberish."
            + nonverbal_clause
            if has_dialogue
            else " No spoken words, muttering, or gibberish occur in this clip."
            + nonverbal_clause
        )
        detailed += f" The final composition is {closing}."
        unique_effects: list[str] = []
        seen_effects: set[str] = set()
        for shot in shots:
            for raw_effect in re.split(r"\s*;\s*", str(shot.get("sound_effects") or "")):
                effect = _compact(raw_effect, 130)
                key = effect.casefold()
                if key in {"", "n/a", "none"} or key in seen_effects:
                    continue
                seen_effects.add(key)
                unique_effects.append(effect)
        if len(unique_effects) > 1:
            unique_effects = [
                effect for effect in unique_effects
                if not effect.casefold().startswith("natural synchronized effects")
            ]
        effects = "; ".join(unique_effects)
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
    overlap_frames: int = 0,
    native_continuation: bool = False,
) -> dict[str, Any]:
    """Plan H3 Omni windows that share canonical references."""

    raw_prompt = str(prompt or "").strip()
    prompt = recover_h3_plain_story(raw_prompt)
    source_intent = extract_h3_source_intent(prompt)
    if prompt != raw_prompt:
        print(
            "[MiniMax H3 Omni] Recovered the original story from an existing "
            "single-clip Context-IR prompt before sequence planning."
        )

    camera_coverage = normalize_h3_camera_coverage(camera_coverage)
    if native_continuation:
        clips = compute_h3_native_sequence_windows(
            total_frames,
            window_frames=max_clip_frames,
            overlap_frames=overlap_frames,
            fps=fps,
        )
        trim_tail_frames = 0
    else:
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
        overlap_frames=overlap_frames,
        native_continuation=native_continuation,
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
            "overlap_frames": int(overlap_frames),
            "native_continuation": bool(native_continuation),
            "windows": [],
            "window_prompts": [],
        }

    expect_dialogue = _narrative_dialogue_expected(prompt, len(clips))
    resolved_coverage = _infer_camera_coverage(prompt, camera_coverage)
    story_ledger: dict[str, Any] | None = None
    planning_warnings: list[str] = []
    try:
        staged = plan_h3_story_segments(
            prompt,
            segment_durations=[float(item["duration_seconds"]) for item in clips],
            mode=(
                "reference_sequence_continuation"
                if native_continuation
                else "reference_sequence"
            ),
            camera_coverage=resolved_coverage,
            # Retention belongs in retention_analysis. Putting it into the
            # subject field made every fallback prompt repeat the same
            # fully_preserved/reference list twice.
            reference_context=relationships,
            expect_dialogue=expect_dialogue,
            image_paths=image_paths,
            nsfw=nsfw,
        )
        planned_by = staged["planned_by"]
        planning_warnings = list(staged.get("planning_warnings") or [])
        source_intent = staged.get("source_intent") or source_intent
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
            "source_intent": source_intent,
            "requested_nonverbal_vocals": story_ledger.get(
                "requested_nonverbal_vocals",
                source_intent.get("requested_nonverbal_vocals", ""),
            ),
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
        planning_warnings.append(
            "The H3 reference-sequence compiler could not use the AI camera "
            "plan, so Maestro generated a source-faithful deterministic plan."
        )
        plan = _fallback_sequence_plan(
            prompt,
            clips,
            reference_relationships=relationships,
            default_retention=default_retention,
            camera_coverage=camera_coverage,
        )
        plan["source_intent"] = source_intent
        plan["requested_nonverbal_vocals"] = source_intent.get(
            "requested_nonverbal_vocals", ""
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
        "planning_warnings": list(dict.fromkeys(planning_warnings)),
        "plan_kind": "reference_sequence",
        "camera_coverage": camera_coverage,
        "total_frames": int(total_frames),
        "window_frames": int(max_clip_frames),
        "window_count": len(compiled),
        "resolution": str(resolution or ""),
        "model_type": str(model_type or ""),
        "per_clip_frames": [int(item["frames"]) for item in clips],
        "trim_tail_frames": int(trim_tail_frames),
        "overlap_frames": int(overlap_frames),
        "native_continuation": bool(native_continuation),
        "subject_continuity": plan.get("subject_definitions", ""),
        "setting_continuity": plan.get("setting_continuity", ""),
        "story_ledger": story_ledger,
        "source_intent": source_intent,
        "windows": compiled,
        "window_prompts": [item["prompt"] for item in compiled],
    }
