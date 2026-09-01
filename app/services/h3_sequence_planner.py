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
    H3DialogueTimingError,
    H3_STORY_LEDGER_VERSION,
    _active_h3_cast_names,
    _merge_h3_cast_names,
    _same_h3_cast_identity,
    _source_requests_multiple_cast_instances,
    extract_h3_source_intent,
    normalize_h3_planning_style,
    plan_h3_story_segments,
    recover_h3_plain_story,
)
from services.h3_window_planner import (
    _UNREQUESTED_SPECTACLE_PATTERNS,
    _compact,
    _fallback_plan,
    _h3_shot_timestamp,
    _infer_camera_coverage,
    _creative_dialogue_expected,
    _normalized_window_shots,
    compute_h3_window_boundaries,
    normalize_h3_camera_coverage,
)


# Increment whenever the compiled Context-IR contract changes.  The signature
# is persisted with reviewed/generated window prompts, so this prevents a run
# restored from gallery metadata from silently reusing pre-fix dialogue and
# reference bindings.
_H3_SEQUENCE_PLANNER_VERSION = 6 + H3_STORY_LEDGER_VERSION
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
    planning_style: str = "faithful",
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
        "planning_style": normalize_h3_planning_style(planning_style),
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
    planning_style: str = "faithful",
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
    if normalize_h3_planning_style(plan.get("planning_style")) != normalize_h3_planning_style(planning_style):
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

    from models.minimax_h3.ref2va import canonicalize_ref2va_reference_order
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
    _prompt, items, _order_remap = canonicalize_ref2va_reference_order("", items)
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
                    f"<Subject {subject}> is {role} from <Picture {picture}>, preserving "
                    "identity and visible appearance; the source background, framing, "
                    "composition, and pose do not define the target scene."
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
                    f"<Subject {subject}> is {role} from <Video {video}>, preserving identity, "
                    "appearance, and characteristic motion while using the newly described target "
                    "scene and action."
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
                    f"<Audio {audio}> is the voice-timbre reference for {target}, guiding "
                    "emotion and delivery without reusing the source words or timing."
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
            f"<Audio {audio_index}> is the voice-timbre reference for {target}, guiding "
            "emotion and delivery without reusing the source words or timing."
        )
        retention.append(f"<Audio {audio_index}>: reference")
        if "audio reference" not in task_types:
            task_types.append("audio reference")
    return "\n".join(relationships), "\n".join(retention), " + ".join(task_types)


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

    text = str(value or "").strip()
    if not text:
        return text
    pairs = re.findall(
        r"\b(S\d+)\s+is\s+([^;,.]+)",
        text,
        flags=re.IGNORECASE,
    )
    # Speaker IDs are ordered by the first vocal event in *this generation
    # pass*. They are not stable character IDs and must never survive beside
    # the canonical Subject/Picture/Audio map across continuation windows.
    # Keep the names for legacy promotion only when no Subject map exists.
    deduped = re.sub(
        r"(?i)\bstable\s+speaking\s+identit(?:y|ies)\s*:\s*"
        r"S\d+\s+is\s+[^.;<]+(?:\s*;\s*S\d+\s+is\s+[^.;<]+)*\s*\.?",
        "",
        text,
    ).strip(" ;,.")
    if re.search(r"<Subject\s+\d+>", text, re.IGNORECASE):
        return deduped
    if not pairs:
        return deduped
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
    """Label a reusable Subject only where that character actually appears.

    Every reference remains declared in ``subject_definitions`` and is still
    supplied to Ref2VA.  A character absent from the current story window must
    not be invented in its opening composition merely to mention its Subject
    token; doing so made future entrants appear early and destabilized staging.
    """

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

    for label, name in bindings:
        if label in result:
            continue
        pattern = re.compile(
            rf"(?<![\w>]){re.escape(name)}(?![\w<])",
            flags=re.IGNORECASE,
        )
        result = pattern.sub(f"{label} ({name})", result, count=1)
    return result


def _ref2va_prompt_bindings(
    subject_definitions: str,
) -> tuple[dict[str, int], dict[int, int]]:
    """Read immutable character and voice bindings from canonical Context-IR."""

    aliases: dict[str, int] = {}
    for match in re.finditer(
        r"<Subject\s+(\d+)>\s+is\s+(.+?)"
        r"(?=\s+from\s+<(?:Picture|Video)\s+\d+>|"
        r",\s+(?:whose|with|defined|from|preserving)\b|[;.\r\n]|$)",
        str(subject_definitions or ""),
        flags=re.IGNORECASE,
    ):
        subject = int(match.group(1))
        name = re.sub(r"\s+", " ", match.group(2)).strip(" ,;:.-")
        if name:
            aliases[name.casefold()] = subject

    audio_by_subject: dict[int, int] = {}
    for match in re.finditer(
        r"<Audio\s+(\d+)>[^.\r\n]{0,240}?"
        r"(?:for|to|of)\s+<Subject\s+(\d+)>",
        str(subject_definitions or ""),
        flags=re.IGNORECASE,
    ):
        audio_by_subject.setdefault(int(match.group(2)), int(match.group(1)))
    return aliases, audio_by_subject


def _ref2va_bind_audio_speakers(
    subject_definitions: str,
    subject_speakers: dict[int, str],
) -> str:
    """Bind voice Audio, visual Subject, and local speaker ID together."""

    result = str(subject_definitions or "")
    for subject, speaker in subject_speakers.items():
        result = re.sub(
            rf"(<Audio\s+\d+>[^.\r\n]{{0,260}}?"
            rf"(?:for|to|of)\s+<Subject\s+{subject}>)"
            r"(?:\s*\(S\d+\))?",
            rf"\1 ({speaker})",
            result,
            flags=re.IGNORECASE,
        )
    return result


def _replace_ref2va_names_with_subjects(
    value: str,
    aliases: dict[str, int],
) -> str:
    """Use MiniMax's stable Subject labels instead of ambiguous bare names."""

    result = str(value or "")
    for alias, subject in sorted(aliases.items(), key=lambda row: -len(row[0])):
        pattern = re.compile(
            rf"(?<![\w>]){re.escape(alias)}(?![\w<])",
            flags=re.IGNORECASE,
        )
        result = pattern.sub(f"<Subject {subject}>", result)
    return result


def _h3_plan_dialogue_speakers(plan: dict[str, Any]) -> list[str]:
    speakers: list[str] = []
    for clip in plan.get("clips") or []:
        if not isinstance(clip, dict):
            continue
        for shot in clip.get("shots") or []:
            if not isinstance(shot, dict):
                continue
            for line in shot.get("dialogue") or []:
                if not isinstance(line, dict):
                    continue
                speaker = _compact(line.get("speaker"), 80)
                if speaker and speaker.casefold() != "speaker" and speaker not in speakers:
                    speakers.append(speaker)
    return speakers


def _h3_plan_cast_names(
    plan: dict[str, Any],
    subject_aliases: dict[str, int],
) -> list[str]:
    """Return one project-wide cast inventory without treating places as cast."""

    source_intent = (
        plan.get("source_intent")
        if isinstance(plan.get("source_intent"), dict)
        else {}
    )
    intent_names = [
        _compact(value, 100)
        for value in (source_intent.get("cast_names") or [])
        if _compact(value, 100)
    ]
    reference_names = [
        alias for alias, _subject in sorted(
            subject_aliases.items(),
            key=lambda item: item[1],
        )
    ]
    dialogue_names = _h3_plan_dialogue_speakers(plan)
    return _merge_h3_cast_names(intent_names, reference_names, dialogue_names)


def _h3_cast_display(
    name: str,
    subject_aliases: dict[str, int],
) -> str:
    for alias, subject in subject_aliases.items():
        if _same_h3_cast_identity(name, alias):
            return f"<Subject {subject}>"
    return name


def _h3_native_cast_definition(
    cast_names: list[str],
    subject_aliases: dict[str, int],
) -> str:
    native = [
        name for name in cast_names
        if not any(_same_h3_cast_identity(name, alias) for alias in subject_aliases)
    ]
    if not native:
        return ""
    if len(native) == 1:
        return (
            f"{native[0]} is a named prompt-native recurring character without a media-reference binding; "
            "keep this identity and appearance stable across shots and clips."
        )
    return (
        f"{', '.join(native)} are named prompt-native recurring characters without media-reference bindings; "
        "keep each identity and appearance stable across shots and clips."
    )


def _h3_active_cast_for_clip(
    cast_names: list[str],
    item: dict[str, Any],
    subject_aliases: dict[str, int],
) -> list[str]:
    clip_text = json.dumps(item, ensure_ascii=False)
    active = _active_h3_cast_names(cast_names, clip_text)
    for name in cast_names:
        label = _h3_cast_display(name, subject_aliases)
        match = re.search(r"<Subject\s+(\d+)>", label, flags=re.IGNORECASE)
        if match and re.search(
            rf"<Subject\s+{match.group(1)}>",
            clip_text,
            flags=re.IGNORECASE,
        ) and name not in active:
            active.append(name)
    # A generic one-character camera plan still unambiguously belongs to the
    # sole saved principal. Larger casts must be named locally so future
    # entrants are not forced into an earlier clip.
    if not active and len(cast_names) == 1:
        active = list(cast_names)
    return active


def _h3_window_cast_contract(
    active_cast: list[str],
    *,
    source_prompt: str,
    subject_aliases: dict[str, int],
) -> str:
    if not active_cast:
        return ""
    exact: list[str] = []
    multiples: list[str] = []
    for name in active_cast:
        display = _h3_cast_display(name, subject_aliases)
        target = multiples if _source_requests_multiple_cast_instances(source_prompt, name) else exact
        target.append(display)
    clauses: list[str] = []
    if exact:
        clauses.append(
            "Principal cast in this clip: exactly one " + ", exactly one ".join(exact)
        )
    if multiples:
        clauses.append(
            "Preserve the explicitly requested number of " + ", ".join(multiples)
        )
    clauses.append(
        "Keep every principal identity distinct and preserve these counts through every cut"
    )
    clauses.append(
        "Any already-established background extras remain anonymous and visually distinct from the principals"
    )
    if len(active_cast) > 4:
        clauses.append(
            "Establish the full group once, then use smaller motivated coverage without duplicating anyone"
        )
    return ". ".join(clauses)


_RESTARTED_BLOCKING_RE = re.compile(
    r"\b(?:approach(?:es|ed|ing)?|arriv(?:e|es|ed|ing)|enter(?:s|ed|ing)?|"
    r"reach(?:es|ed|ing)?|walk(?:s|ed|ing)?|sit(?:s|ting)?\s+down|takes?\s+(?:his|her|their|the)\s+seat)\b",
    flags=re.IGNORECASE,
)


def _stabilize_ref2va_hold_shots(shots: list[dict[str, Any]]) -> None:
    """Remove camera prose that restarts a beat already completed on screen."""

    for shot in shots[1:]:
        action = str(shot.get("action") or "")
        if not action.startswith("The immediate result of the preceding assigned event"):
            continue
        framing = str(shot.get("framing") or "")
        if _RESTARTED_BLOCKING_RE.search(framing):
            size = re.search(
                r"\b(?:extreme\s+close-up|close-up|medium\s+close-up|medium-wide|medium|wide)\b",
                framing,
                flags=re.IGNORECASE,
            )
            prefix = size.group(0) if size else "reaction"
            shot["framing"] = (
                f"{prefix} reaction composition preserving the already-established blocking"
            )
        camera = str(shot.get("camera") or "")
        if _RESTARTED_BLOCKING_RE.search(camera):
            shot["camera"] = (
                "a motivated static hold or subtle reaction reframe preserving the established geography"
            )


def _clean_ref2va_action(
    value: str,
    *,
    setting: str = "",
    opening: str = "",
) -> str:
    """Remove compiler/meta prose that H3 can mistakenly perform as speech."""

    text = re.sub(
        r"(?i)^\s*(?:visual direction only|silent visual action)\s*,?\s*"
        r"never spoken narration\s*:\s*",
        "",
        str(value or "").strip(),
    )
    text = re.sub(
        r"(?i)\s*\.?(?:\s+No words are spoken or mouthed in this shot;.*)$",
        "",
        text,
    ).strip(" .")
    clauses = [
        clause.strip(" .")
        for clause in re.split(
            r"(?i)(?:\.\s+Then\s+|\s+Then\s+|\.\s+)",
            text,
        )
        if clause.strip(" .")
    ]
    context_tokens = set(re.findall(
        r"[a-z0-9]+",
        f"{setting} {opening}".casefold(),
    ))
    cleaned: list[str] = []
    for clause in clauses:
        if re.search(
            r"(?i)\b(?:visibly\s+)?(?:delivers?|performs?|speaks?|says?)\b"
            r"[^.]{0,80}\b(?:assigned\s+)?dialogue\b",
            clause,
        ):
            continue
        clause_tokens = set(re.findall(r"[a-z0-9]+", clause.casefold()))
        overlap = (
            len(clause_tokens & context_tokens) / max(1, len(clause_tokens))
            if context_tokens else 0.0
        )
        # Pure scene-establishing prose is already expressed by the setting
        # and opening state. Repeating it inside the timed action is what made
        # H3 audibly narrate phrases such as "Yoda is in Dagobah."
        if overlap >= 0.72 and not re.search(
            r"(?i)\b(?:walk|run|turn|wave|raise|lower|pick|drop|fight|hit|"
            r"punch|snap|fall|move|look|nod|sit|stand\s+up|enter|exit|"
            r"breathe|sigh)\w*\b",
            clause,
        ):
            continue
        cleaned.append(clause)
    return ". Then ".join(cleaned).strip(" .")


def _clean_ref2va_state(value: str) -> str:
    """Keep a handoff state visual instead of re-triggering prior speech."""

    text = str(value or "").strip()
    text = re.sub(
        r"(?i)^the immediate visible state is the result of this event:\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)\s+while\s+(?:saying|speaking|delivering|performing)\b"
        r"[^,.!?;]*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:visibly\s+)?delivers?\s+the\s+(?:assigned\s+)?dialogue\s+line\b",
        "holds the described expression",
        text,
    )
    return re.sub(r"\s{2,}", " ", text).strip(" .")


def _ref2va_dialogue_sentence(
    item: Any,
    *,
    speaker_ids: dict[str, str],
    subject_aliases: dict[str, int],
    audio_by_subject: dict[int, int],
) -> str:
    """Render one line in MiniMax's documented Subject/Speaker syntax."""

    if not isinstance(item, dict):
        return ""
    words = " ".join(str(item.get("text") or "").split()).strip()
    if not words:
        return ""
    speaker = _compact(item.get("speaker") or "Speaker", 80)
    key = speaker.casefold()
    stable_id = speaker_ids.get(key)
    if not stable_id:
        stable_id = f"S{len(speaker_ids) + 1}"
        speaker_ids[key] = stable_id
    language = _compact(item.get("language") or "English", 30)
    subject = subject_aliases.get(key)
    if subject is None:
        # Preserve advanced/unreferenced speakers through the base H3 syntax.
        from services.h3_window_planner import _dialogue_sentence

        return _dialogue_sentence(item, speaker_ids)
    audio = audio_by_subject.get(subject)
    delivery = _compact(item.get("delivery") or "speaks naturally", 100)
    action = _compact(item.get("action") or "", 120)
    if re.search(
        r"(?i)\b(?:only\s+.+?mouth|every\s+other\s+visible\s+mouth|"
        r"established\s+target\s+scene)\b",
        action,
    ):
        action = ""
    is_voiceover = bool(re.search(
        r"(?i)\b(?:off[- ]camera|off[- ]screen|voice[- ]?over|unseen)\b",
        f"{delivery} {action}",
    ))
    if is_voiceover:
        lead = f"<Subject {subject}> ({stable_id}) says in an off-screen voiceover"
    else:
        lead = f"<Subject {subject}> ({stable_id})"
        if action:
            lead += f" {action.strip()} and"
        lead += " says"
    if audio is not None:
        lead += f" in the voice referenced from <Audio {audio}>"
    if delivery and delivery.casefold() not in {"speaks naturally", "says naturally"}:
        delivery = re.sub(r"(?i)^\s*(?:speaks?|says?)\s+(?:with|in)?\s*", "", delivery)
        if delivery:
            lead += f" with {delivery.strip()} delivery"
    return f"{lead}, <d>[{language}] {words}</d>."


def _ref2va_shot_prompt_sentence(
    shot: dict[str, Any],
    *,
    speaker_ids: dict[str, str],
    subject_aliases: dict[str, int],
    audio_by_subject: dict[int, int],
    preamble: str = "",
    setting: str = "",
    opening: str = "",
) -> str:
    number = int(shot["shot"])
    start = float(shot["start_seconds"])
    end = float(shot["end_seconds"])
    transition = _compact(shot.get("transition"), 70)
    framing = _compact(shot.get("framing"), 130)
    camera = _compact(shot.get("camera"), 170)
    action = _clean_ref2va_action(
        shot.get("action") or "",
        setting=setting,
        opening=opening,
    )
    action = _replace_ref2va_names_with_subjects(action, subject_aliases)
    if number == 1:
        lead = f"[Shot 1] {preamble} From {start:.2f} to {end:.2f} seconds, {framing}".strip()
    elif transition.casefold().startswith(("continuous", "without a cut", "reframe")):
        lead = (
            f"[Shot {number}] At {_h3_shot_timestamp(start)}, without a cut, "
            f"reframe to {framing}; continue through {_h3_shot_timestamp(end)}"
        )
    else:
        lead = (
            f"[Shot {number}] At {_h3_shot_timestamp(start)}, "
            f"{transition or 'hard cut'} to {framing}; continue through "
            f"{_h3_shot_timestamp(end)}"
        )
    details = "; ".join(part for part in (camera, action) if part)
    sentence = f"{lead}; {details}." if details else f"{lead}."
    dialogue = " ".join(
        value
        for value in (
            _ref2va_dialogue_sentence(
                line,
                speaker_ids=speaker_ids,
                subject_aliases=subject_aliases,
                audio_by_subject=audio_by_subject,
            )
            for line in (shot.get("dialogue") or [])
        )
        if value
    )
    return f"{sentence} {dialogue}".strip()


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
    source_prompt: str = "",
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

    # Reference labels are application-owned.  Mixing model-authored wardrobe
    # prose into this block previously produced malformed definitions such as
    # ``black t-shirt <Subject 1> is ...``.  Use the canonical manifest by
    # itself; visual prose belongs in the chronological shot description.
    raw_subjects = str(plan.get("subject_definitions") or "").strip()
    canonical_subjects = str(reference_relationships or "").strip()
    subjects = (
        canonical_subjects
        if canonical_subjects
        else _officialize_subject_definitions(dedupe_sentences(raw_subjects))
    )
    subject_aliases, audio_by_subject = _ref2va_prompt_bindings(subjects)
    cast_names = _h3_plan_cast_names(plan, subject_aliases)
    native_cast_definition = _h3_native_cast_definition(
        cast_names,
        subject_aliases,
    )
    subjects = "\n".join(
        part for part in (subjects, native_cast_definition) if part
    )
    source_prompt = str(
        source_prompt or plan.get("source_prompt") or ""
    ).strip()

    # Reference retention is likewise owned by the manifest. Keep any
    # non-reference analysis the planner supplied, then add each canonical
    # slot contract once.
    raw_plan_retention = str(plan.get("retention_analysis") or "").strip()
    non_reference_retention = "; ".join(
        clause.strip()
        for clause in re.split(r"\s*;\s*", raw_plan_retention)
        if clause.strip() and not reference_slot_pattern.search(clause)
    )
    optional_retention = _compact(non_reference_retention, 240)
    canonical_retention = str(default_retention or "").strip()
    raw_retention = "\n".join(
        part for part in (optional_retention, canonical_retention) if part
    )
    retention_clauses: list[str] = []
    seen_retention: set[str] = set()
    for clause in re.split(r"\s*(?:;|\r?\n)\s*", raw_retention):
        key = re.sub(r"[^a-z0-9<>]+", " ", clause.casefold()).strip()
        if clause and key not in seen_retention:
            retention_clauses.append(clause.strip())
            seen_retention.add(key)
    retention = "\n".join(retention_clauses)
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
    compiled: list[dict[str, Any]] = []

    for geometry, item in zip(clips, planned_clips):
        if not isinstance(item, dict):
            raise ValueError(f"H3 Omni sequence clip {geometry['index']} is invalid.")
        duration = float(geometry["duration_seconds"])
        shots = _normalized_window_shots(item, duration)
        if not shots:
            raise ValueError(f"H3 Omni sequence clip {geometry['index']} has no shots.")
        _stabilize_ref2va_hold_shots(shots)
        # MiniMax speaker IDs are scoped to one generated clip/window and are
        # assigned by first vocal-event order. Subject IDs remain stable across
        # the whole project. Reusing the semantic ledger's global S-labels made
        # a later window's sole speaker (for example Subject 3) collide with the
        # character who happened to be S1 in the first window.
        speaker_ids: dict[str, str] = {}
        for shot in shots:
            local_dialogue: list[Any] = []
            for raw_line in shot.get("dialogue") or []:
                if not isinstance(raw_line, dict):
                    local_dialogue.append(raw_line)
                    continue
                line = dict(raw_line)
                speaker = _compact(line.get("speaker") or "Speaker", 80)
                key = speaker.casefold()
                if key not in speaker_ids:
                    speaker_ids[key] = f"S{len(speaker_ids) + 1}"
                line["speaker_id"] = speaker_ids[key]
                local_dialogue.append(line)
            shot["dialogue"] = local_dialogue
        subject_speakers = {
            subject_aliases[alias]: speaker
            for alias, speaker in speaker_ids.items()
            if alias in subject_aliases
        }
        window_subjects = _ref2va_bind_audio_speakers(
            subjects,
            subject_speakers,
        )
        active_cast = _h3_active_cast_for_clip(
            cast_names,
            item,
            subject_aliases,
        )
        cast_contract = _h3_window_cast_contract(
            active_cast,
            source_prompt=source_prompt,
            subject_aliases=subject_aliases,
        )
        source_intent = (
            plan.get("source_intent")
            if isinstance(plan.get("source_intent"), dict)
            else {}
        )
        blocking_contract = _compact(
            source_intent.get("blocking_contract"),
            360,
        )
        blocking_names = _active_h3_cast_names(cast_names, blocking_contract)
        if blocking_names and not all(
            any(_same_h3_cast_identity(name, active) for active in active_cast)
            for name in blocking_names
        ):
            blocking_contract = ""
        coverage = _compact(item.get("coverage") or "cinematic editorial coverage", 90)
        pacing = _compact(item.get("pacing") or "natural real-time pacing", 180)
        opening = _compact(
            _clean_ref2va_state(
                item.get("opening_state") or "The scene begins in a clear composition"
            ),
            210,
        )
        closing = _compact(
            _clean_ref2va_state(
                item.get("closing_state") or "The beat settles in a clear final composition"
            ),
            210,
        )
        pacing_sentence = f"Coverage is {coverage}; pacing is {pacing}."
        if "slow motion" not in pacing.casefold():
            pacing_sentence += " Slow motion occurs only when explicitly requested."
        preamble = " ".join(part for part in (
            f"{setting}." if setting else "",
            f"Cast contract: {cast_contract}." if cast_contract else "",
            f"Blocking contract: {blocking_contract}." if blocking_contract else "",
            f"Opening state: {opening}.",
            pacing_sentence,
        ) if part)
        preamble = _replace_ref2va_names_with_subjects(preamble, subject_aliases)
        shot_timeline = " ".join(
            _ref2va_shot_prompt_sentence(
                shot,
                speaker_ids=speaker_ids,
                subject_aliases=subject_aliases,
                audio_by_subject=audio_by_subject,
                preamble=preamble if shot_index == 0 else "",
                setting=setting,
                opening=opening,
            )
            for shot_index, shot in enumerate(shots)
        )
        detailed = f"{_ref2va_style_opening(style)} {shot_timeline}".strip()
        detailed = _replace_ref2va_names_with_subjects(detailed, subject_aliases)
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
        if has_dialogue:
            detailed += " The tagged dialogue is performed once in the listed order."
        elif not local_nonverbal:
            detailed += " The characters remain silent during this clip."
        detailed += nonverbal_clause
        detailed += (
            " The final composition is "
            f"{_replace_ref2va_names_with_subjects(closing, subject_aliases)}."
        )
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
        summary = _clean_ref2va_action(
            item.get("summary") or item.get("title") or "The requested story advances",
            setting=setting,
            opening=opening,
        ) or _compact(item.get("title") or "The requested story advances", 180)
        summary = _compact(
            _replace_ref2va_names_with_subjects(summary, subject_aliases),
            240,
        )
        if not summary.startswith("["):
            summary = f"[{task_types}] {summary}"
        prompt = "\n\n".join([
            f"subject_definitions: {window_subjects}",
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
            "active_cast": active_cast,
            "cast_contract": cast_contract,
            "blocking_contract": blocking_contract,
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
    planning_style: str = "faithful",
) -> dict[str, Any]:
    """Plan H3 Omni windows that share canonical references."""

    raw_prompt = str(prompt or "").strip()
    prompt = recover_h3_plain_story(raw_prompt)
    source_intent = extract_h3_source_intent(prompt)
    if prompt != raw_prompt and "subject_definitions:" in raw_prompt.casefold():
        print(
            "[MiniMax H3 Omni] Recovered the original story from an existing "
            "single-clip Context-IR prompt before sequence planning."
        )

    camera_coverage = normalize_h3_camera_coverage(camera_coverage)
    planning_style = normalize_h3_planning_style(planning_style)
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
        planning_style=planning_style,
    )
    if len(clips) <= 1:
        return {
            "source_prompt": str(prompt or ""),
            "signature": signature,
            "planned_by": "not_needed",
            "plan_kind": "reference_sequence",
            "camera_coverage": camera_coverage,
            "planning_style": planning_style,
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

    expect_dialogue = (
        _creative_dialogue_expected(prompt, len(clips))
        if planning_style == "creative"
        else bool(re.search(r"[\"“][^\"”]+[\"”]", prompt))
    )
    resolved_coverage = _infer_camera_coverage(prompt, camera_coverage)
    story_ledger: dict[str, Any] | None = None
    planning_warnings: list[str] = []
    planning_diagnostics: list[str] = []
    planning_notes: list[str] = []
    dialogue_fragments: list[dict[str, Any]] = []
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
            planning_style=planning_style,
            image_paths=image_paths,
            nsfw=nsfw,
        )
        planned_by = staged["planned_by"]
        planning_warnings = list(staged.get("planning_warnings") or [])
        planning_diagnostics = list(staged.get("planning_diagnostics") or [])
        planning_notes = list(staged.get("planning_notes") or [])
        dialogue_fragments = list(staged.get("dialogue_fragments") or [])
        source_intent = staged.get("source_intent") or source_intent
        story_ledger = staged["ledger"]
        planned_clips = []
        for index, segment in enumerate(staged["segments"]):
            planned_clips.append({
                **segment,
                "clip": index + 1,
            })
        plan = {
            "source_prompt": prompt,
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
    except H3DialogueTimingError:
        raise
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
        plan["source_prompt"] = prompt
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
        "planning_diagnostics": list(dict.fromkeys(planning_diagnostics)),
        "planning_notes": list(dict.fromkeys(planning_notes)),
        "plan_kind": "reference_sequence",
        "camera_coverage": camera_coverage,
        "planning_style": planning_style,
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
        "dialogue_fragments": dialogue_fragments,
        "source_intent": source_intent,
        "windows": compiled,
        "window_prompts": [item["prompt"] for item in compiled],
    }
