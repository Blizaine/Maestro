"""
Short Film Planner — creates a ProductionPlan from story/audio inputs.

Supports two paths:
  1. Audio-driven: dialogue audio + transcript → scene plans
  2. Story-driven: story description + characters → scene plans (no audio)

Outputs: ProductionPlan with ShotPlan objects (NOT final prompts).
"""

from __future__ import annotations
import copy
import json
import math
import os
import re
from typing import Optional, Any

from ..schema import (
    ProductionPlan, ShotPlan, CharacterProfile, ReferenceAssets,
    AssetRef, SubjectRef, DialogueBeat, CameraPlan, AudioPlan,
    VALID_CONTINUITY_STRATEGIES,
)
from ..policies import build_character_rules_block, build_camera_style_block
from ..guide_loader import load_guide as _load_guide_helper
from ..h3_dialogue import (
    compile_h3_vocal_contract as _inject_h3_vocal_contract,
    h3_dialogue_budget_violations as _h3_dialogue_budget_violations,
    normalize_h3_text as _normalize_h3_text,
)
from .base import BasePlanner


# Video-model architecture → Pass 2 shot-breakdown guide file.
# Currently only LTX-2/LTX-V have a dedicated Pass-2 guide. Other
# video families share the LTX-2 rules as a best-effort fallback
# until per-model Pass-2 guides land in Phase 3.
_VIDEO_PASS2_GUIDE_MAP = {
    "minimax_h3_ref2va": "minimax_h3_shot_breakdown.md",
    "minimax_h3": "minimax_h3_shot_breakdown.md",
    "ltx2": "ltx2_shot_breakdown.md",
    "ltxv": "ltx2_shot_breakdown.md",
}


# ── Pass 2 JSON output schemas (llama-server grammar constraint) ──────
# These mirror the JSON examples embedded in the Pass 2 / fallback system
# prompts. llama-server compiles the schema to a GBNF grammar that masks
# every token which would break it, so a constrained pass physically
# cannot emit prose, markdown fences, or repeat-loop garbage (the Gemma 4
# 12B failure: 96K chars of looping pseudo-JSON on a 5-min film).
#
# additionalProperties=False is the actual loop-killer: a grammar-compiled
# closed object emits each key AT MOST ONCE, in this defined order, so the
# "repeat the same field/object until max_tokens" failure class becomes
# unrepresentable. The flip side: any field a prompt's output spec asks
# for MUST be listed here, in spec order, or the grammar will forbid the
# model from writing it. If you add a field to a Pass 2 output spec,
# add it to _SHOT_PROPERTIES too.
#
# Strings stay unbounded (creative prose can't be length-capped at the
# grammar level) — intra-string repetition remains covered by the
# registry-level repeat penalties in llm_service.

_SUBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "visual_description": {"type": "string"},
        "character_id": {"type": "string"},
        "speaker_name": {"type": "string"},
        "position_or_relation": {"type": "string"},
        "wardrobe": {"type": "string"},
    },
    "required": ["visual_description"],
    "additionalProperties": False,
}

_DIALOGUE_BEAT_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker_id": {"type": "string"},
        "spoken_text": {"type": "string"},
        "delivery": {"type": "string"},
        "physical_cue": {"type": "string"},
        "priority": {"type": "string"},
    },
    "required": ["spoken_text"],
    "additionalProperties": False,
}

_CAMERA_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "framing": {"type": "string"},
        "angle": {"type": "string"},
        "movement": {"type": "string"},
        "movement_intensity": {"type": "string"},
        "lens_feel": {"type": "string"},
        "reframing_notes": {"type": "string"},
    },
    "required": ["framing"],
    "additionalProperties": False,
}

_AUDIO_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string"},
        "ambience": {"type": "string"},
        "effects": {"type": "array", "items": {"type": "string"}},
        "vocal_style": {"type": "string"},
        "timing_anchor": {"type": "string"},
        "lip_sync_critical": {"type": "boolean"},
    },
    "required": ["mode"],
    "additionalProperties": False,
}

# Union of every field the story-mode spec, the audio-mode spec, and the
# single-pass fallback spec request, in spec order (story-mode order;
# audio-mode-only fields slot where its spec shows them). Per-call-site
# `required` lists pick which subset the grammar forces.
_SHOT_PROPERTIES = {
    "title": {"type": "string"},
    "duration_sec": {"type": "number"},
    "scene_goal": {"type": "string"},
    "narrative_role": {"type": "string"},
    "scene_type": {"type": "string"},
    "continuity_strategy": {"type": "string"},
    "continuity_group": {"type": "string"},
    "subjects_on_screen": {"type": "array", "items": _SUBJECT_SCHEMA},
    "spatial_setup": {"type": "string"},
    "environment": {"type": "string"},
    "visual_style": {"type": "string"},
    "lighting": {"type": "string"},
    "mood": {"type": "string"},
    "action_beats": {"type": "array", "items": {"type": "string"}},
    "dialogue_beats": {"type": "array", "items": _DIALOGUE_BEAT_SCHEMA},
    "camera_plan": _CAMERA_PLAN_SCHEMA,
    "audio_plan": _AUDIO_PLAN_SCHEMA,
    "ending_beat": {"type": "string"},
    "closing_blocking": {"type": "string"},
    "image_source": {"type": "string"},
    "image_prompt": {"type": "string"},
    "visual_changes": {"type": "array", "items": {"type": "string"}},
    "video_prompt": {"type": "string"},
    "multishot": {"type": "boolean"},
    "keyframe_prompts": {"type": "array", "items": {"type": "string"}},
    "window_prompts": {"type": "array", "items": {"type": "string"}},
}


_SHOT_IMAGE_FIELDS = frozenset({
    "image_source",
    "image_prompt",
    "visual_changes",
    "keyframe_prompts",
})

# H3 remains natural around two spoken words per second. A small 0.1 margin
# avoids rejecting a 29-word line in the model's 14.375-second maximum clip
# solely because the old floor-based budget rounded 28.75 down to 28.
_H3_DIALOGUE_WORDS_PER_SECOND = 2.1


def _h3_preferred_native_durations(
    *,
    fps: int,
    frames_minimum: int,
    frames_maximum: int,
    frames_steps: int,
) -> list[float]:
    """Return a compact set of valid, human-friendly H3 shot lengths."""

    fps = max(1, int(fps or 24))
    frames_minimum = max(1, int(frames_minimum or 124))
    frames_maximum = max(frames_minimum, int(frames_maximum or frames_minimum))
    frames_steps = max(1, int(frames_steps or 17))
    valid = list(range(frames_minimum, frames_maximum + 1, frames_steps))
    if not valid:
        valid = [frames_minimum]
    targets = [8.0, 10.0, 12.0, 14.0]
    selected: list[int] = []
    for target in targets:
        if target < frames_minimum / fps or target > frames_maximum / fps:
            continue
        nearest = min(
            valid,
            key=lambda frames: (abs(frames / fps - target), frames),
        )
        if nearest not in selected:
            selected.append(nearest)
    if not selected:
        selected.append(valid[-1])
    elif valid[-1] not in selected:
        # Always advertise the actual execution ceiling. When four friendly
        # targets were already selected, replace the longest near-target
        # instead of hiding the final hardware-safe native duration.
        if len(selected) >= 4:
            selected[-1] = valid[-1]
        else:
            selected.append(valid[-1])
    return [frames / fps for frames in sorted(selected)]


_H3_VOICE_BIBLE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "character_name": {"type": "string"},
            "personality_engine": {"type": "string"},
            "speech_pattern": {"type": "string"},
            "relationship_behavior": {"type": "string"},
            "performance_direction": {"type": "string"},
            "avoid": {"type": "string"},
        },
        "required": [
            "character_name",
            "personality_engine",
            "speech_pattern",
            "relationship_behavior",
            "performance_direction",
            "avoid",
        ],
        "additionalProperties": False,
    },
    "minItems": 0,
    "maxItems": 16,
}


def _h3_table_read_schema(turn_count: int) -> dict:
    """Closed schema for the dialogue-only H3 table-read revision."""

    turn_count = max(1, int(turn_count or 1))
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "turn": {"type": "integer"},
                "speaker_name": {"type": "string"},
                "original_text": {"type": "string"},
                "revised_text": {"type": "string"},
                "delivery": {"type": "string"},
            },
            "required": [
                "turn",
                "speaker_name",
                "original_text",
                "revised_text",
                "delivery",
            ],
            "additionalProperties": False,
        },
        "minItems": turn_count,
        "maxItems": turn_count,
    }


def _shot_list_schema(
    min_items: int,
    max_items: int,
    required: list[str],
    *,
    include_image_fields: bool = True,
) -> dict:
    """JSON schema for a Pass 2 shot list: a bounded array of closed shot objects."""
    properties = {
        key: value
        for key, value in _SHOT_PROPERTIES.items()
        if include_image_fields or key not in _SHOT_IMAGE_FIELDS
    }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": [field for field in required if field in properties],
            "additionalProperties": False,
        },
        "minItems": max(1, min_items),
        "maxItems": max(1, max_items),
    }


def _discard_unused_image_fields(shot_dicts: list[dict]) -> list[dict]:
    """Defensively remove still-image planning data from video-only output."""

    for shot in shot_dicts:
        if not isinstance(shot, dict):
            continue
        for field in _SHOT_IMAGE_FIELDS:
            shot.pop(field, None)
    return shot_dicts


def _fit_bounded_frame_schedule(
    durations: list[float],
    *,
    target_duration: float,
    fps: float,
    minimum_frames: int,
    maximum_frames: int,
    frame_step: int,
    minimum_frames_by_item: Optional[list[int]] = None,
) -> list[int]:
    """Fit independent shots to a bounded model lattice near target runtime.

    ``minimum_frames_by_item`` supplies optional per-shot lower bounds. The
    bounds are snapped upward to the same model frame lattice and are useful
    when a shot needs enough time for immutable dialogue. If those floors need
    more time than the requested project runtime, the schedule grows only by
    the minimum representable amount instead of silently squeezing speech.
    """

    if not durations:
        return []
    fps = max(1.0, float(fps or 24))
    minimum_frames = max(1, int(minimum_frames))
    maximum_frames = max(minimum_frames, int(maximum_frames))
    frame_step = max(1, int(frame_step))
    valid = list(range(minimum_frames, maximum_frames + 1, frame_step))
    count = len(durations)
    item_minimums: list[int] = []
    for index in range(count):
        requested = minimum_frames
        if minimum_frames_by_item and index < len(minimum_frames_by_item):
            try:
                requested = max(requested, int(minimum_frames_by_item[index]))
            except (TypeError, ValueError):
                requested = minimum_frames
        item_minimums.append(next(
            (candidate for candidate in valid if candidate >= requested),
            valid[-1],
        ))
    target_frames = round(max(0.0, float(target_duration)) * fps)
    target_frames = max(sum(item_minimums), target_frames)
    target_frames = min(count * maximum_frames, target_frames)

    positive = [max(0.01, float(value or 0)) for value in durations]
    raw_total = sum(positive)
    scaled = [value / raw_total * target_frames for value in positive]
    schedule = []
    for index, desired in enumerate(scaled):
        eligible = [
            candidate for candidate in valid
            if candidate >= item_minimums[index]
        ]
        schedule.append(min(
            eligible,
            key=lambda candidate: (abs(candidate - desired), candidate),
        ))

    # The requested total may not be exactly representable because each shot
    # advances by frame_step. Make only changes that reduce total timing error.
    while True:
        total = sum(schedule)
        current_error = abs(target_frames - total)
        direction = 1 if total < target_frames else -1
        candidates = [
            index
            for index, frames in enumerate(schedule)
            if (
                direction > 0 and frames + frame_step <= maximum_frames
            ) or (
                direction < 0 and frames - frame_step >= item_minimums[index]
            )
        ]
        if not candidates:
            break
        if direction > 0:
            index = max(candidates, key=lambda item: scaled[item] - schedule[item])
        else:
            index = max(candidates, key=lambda item: schedule[item] - scaled[item])
        revised_total = total + direction * frame_step
        if abs(target_frames - revised_total) >= current_error:
            break
        schedule[index] += direction * frame_step
    return schedule


def _sanitize_h3_independent_prompt(value: Any) -> str:
    """Remove rolling-window commands that are invalid for a native H3 shot."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(
        r"^\s*(?:begin this portion of the scene(?: and leave the action able "
        r"to continue)?|continue directly from the preceding portion of the "
        r"same scene|continue from (?:the )?(?:previous|preceding) "
        r"(?:shot|scene|portion))\s*[.!:-]*\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\bwindow\s+\d+\s*(?:\([^)]*\))?\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _h3_plain_dialogue_text(value: Any) -> str:
    """Return canonical spoken words without H3 markup or a language prefix."""

    text = _normalize_h3_text(value)
    text = re.sub(r"<\s*/?\s*d\s*>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
    return re.sub(r"\s+", " ", text)


def _h3_screenplay_speaker_heading(value: Any) -> tuple[str, bool] | None:
    """Recognize the screenplay speaker headings emitted by Director Pass 1."""

    text = str(value or "").strip()
    centered = re.fullmatch(
        r"<center>\s*([^<\r\n]+?)\s*</center>",
        text,
        flags=re.IGNORECASE,
    )
    if centered:
        return _normalize_h3_text(centered.group(1)).strip(), True

    markdown = re.fullmatch(r"\*\*\s*([^*\r\n]+?)\s*\*\*", text)
    if markdown:
        text = markdown.group(1).strip()

    # Standard screenplay headings are short uppercase names. Exclude scene
    # headings and structural labels so they cannot become phantom speakers.
    if not re.fullmatch(r"[A-Z][A-Z0-9 .'\-()]{0,60}", text):
        return None
    upper = text.upper()
    if upper.startswith(("INT.", "EXT.", "INT/EXT.", "I/E.")):
        return None
    if upper in {
        "SCREENPLAY", "FADE IN", "FADE OUT", "CUT TO", "SMASH CUT TO",
        "THE END", "ACTION", "DIALOGUE", "CONTINUED",
    }:
        return None
    name = re.sub(
        r"\s*\((?:CONT['’]?D|V\.?O\.?|O\.?S\.?)\)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return (_normalize_h3_text(name), False) if name else None


def _extract_h3_screenplay_dialogue(screenplay: Any) -> list[dict[str, str]]:
    """Extract the immutable speaker/word stream from Pass 1 screenplay text.

    Director asks for either centered Markdown dialogue blocks or conventional
    uppercase screenplay headings. Parsing that small contract is safer than
    treating a later, potentially truncated shot-plan response as the script.
    """

    text = _normalize_h3_text(screenplay)
    text = re.sub(
        r"<(think|thinking|seed:think|reasoning|reflection)>.*?</\1>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    manifest: list[dict[str, str]] = []
    index = 0
    while index < len(lines):
        heading = _h3_screenplay_speaker_heading(lines[index])
        if not heading:
            index += 1
            continue
        speaker_name, centered = heading
        index += 1
        spoken_lines: list[str] = []
        while index < len(lines):
            raw = lines[index]
            stripped = raw.strip()
            if _h3_screenplay_speaker_heading(stripped):
                break
            if not stripped:
                index += 1
                if spoken_lines:
                    break
                continue
            if centered and not stripped.startswith(">"):
                break
            if not centered and re.match(
                r"^(?:INT\.|EXT\.|INT/EXT\.|I/E\.)\s+",
                stripped,
                flags=re.IGNORECASE,
            ):
                break
            dialogue = re.sub(r"^>\s?", "", stripped).strip()
            if re.fullmatch(r"\([^)]*\)", dialogue):
                index += 1
                continue
            spoken_lines.append(dialogue)
            index += 1
        spoken = _h3_plain_dialogue_text(" ".join(spoken_lines))
        if spoken:
            manifest.append({
                "speaker_name": speaker_name,
                "spoken_text": spoken,
            })
    return manifest


def _h3_dialogue_word_fingerprint(value: Any) -> tuple[str, ...]:
    """Compare spoken words while ignoring punctuation and Markdown emphasis."""

    text = _h3_plain_dialogue_text(value).casefold()
    text = text.replace("’", "'").replace("‘", "'")
    return tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text, flags=re.UNICODE))


def _h3_speaker_name_tokens(value: Any) -> tuple[str, ...]:
    text = _normalize_h3_text(value).casefold()
    return tuple(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _normalize_h3_voice_bible(
    rows: Any,
    *,
    supported_character_text: str,
) -> list[dict[str, str]]:
    """Validate a compact cast voice bible without trusting invented names."""

    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        for envelope_key in ("characters", "cast", "voice_bible", "profiles"):
            nested = rows[0].get(envelope_key)
            if isinstance(nested, list):
                rows = nested
                break
    supported_tokens = set(_h3_speaker_name_tokens(supported_character_text))
    supported_folded = _normalize_h3_text(supported_character_text).casefold()
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    required = (
        "character_name",
        "personality_engine",
        "speech_pattern",
        "relationship_behavior",
        "performance_direction",
        "avoid",
    )
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        values = {
            field: re.sub(
                r"\s+",
                " ",
                _normalize_h3_text(raw.get(field) or ""),
            ).strip(" .")
            for field in required
        }
        if any(not values[field] for field in required):
            continue
        name = values["character_name"]
        key = name.casefold()
        name_tokens = _h3_speaker_name_tokens(name)
        # The voice-bible model may know additional franchise characters, but
        # it must never add them to a user's cast. Accept a full supplied name
        # or a distinctive supplied first/name token only.
        supported = key in supported_folded or any(
            len(token) >= 3 and token in supported_tokens
            for token in name_tokens
        )
        if not supported or key in seen:
            continue
        seen.add(key)
        normalized.append(values)
    return normalized[:16]


def _format_h3_voice_bible(rows: list[dict[str, str]]) -> str:
    """Render structured voice profiles as compact binding LLM guidance."""

    lines: list[str] = []
    for row in rows or []:
        lines.append(
            f"- {row['character_name']}: personality/behavior: "
            f"{row['personality_engine']}; speech: {row['speech_pattern']}; "
            f"relationships: {row['relationship_behavior']}; performance: "
            f"{row['performance_direction']}; avoid: {row['avoid']}."
        )
    return "\n".join(lines)


def _h3_user_locked_dialogue_map(
    value: Any,
) -> dict[tuple[str, ...], str]:
    """Map user-authored literal dialogue fingerprints to their exact text."""

    text = _normalize_h3_text(value)
    candidates: list[str] = []
    patterns = (
        r'<\s*d\s*>\s*(?:\[[^\]]+\]\s*)?(.+?)<\s*/\s*d\s*>',
        r'"([^"\r\n]{1,600})"',
        r'\u201c([^\u201d\r\n]{1,600})\u201d',
    )
    for pattern in patterns:
        candidates.extend(
            match.group(1)
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
    locked: dict[tuple[str, ...], str] = {}
    for candidate in candidates:
        exact_text = _h3_plain_dialogue_text(candidate)
        fingerprint = _h3_dialogue_word_fingerprint(exact_text)
        if fingerprint:
            locked.setdefault(fingerprint, exact_text)
    return locked


def _h3_user_locked_dialogue_fingerprints(value: Any) -> set[tuple[str, ...]]:
    """Find literal user-authored dialogue that a table read must not rewrite."""

    return set(_h3_user_locked_dialogue_map(value))


def _apply_h3_character_table_read(
    manifest: list[dict[str, Any]],
    rows: Any,
    *,
    story_description: str,
    max_spoken_words: int,
    maximum_line_words: int = 30,
) -> tuple[list[dict[str, Any]], int]:
    """Install a dialogue-only revision after strict identity/order checks.

    The screenplay manifest remains the structural authority. The editor may
    improve only the spoken words and performance direction; it cannot add,
    remove, reorder, or reassign turns. Literal dialogue quoted by the user is
    restored deterministically even if the editor attempts to change it.
    """

    if not manifest:
        return [], 0
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        for envelope_key in (
            "turns", "dialogue_turns", "revisions", "table_read",
        ):
            nested = rows[0].get(envelope_key)
            if isinstance(nested, list):
                rows = nested
                break
    if not isinstance(rows, list) or len(rows) != len(manifest):
        raise ValueError(
            "table read did not return exactly one row per screenplay turn"
        )

    by_turn: dict[int, dict] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("table read contains a non-object row")
        try:
            turn = int(raw.get("turn"))
        except (TypeError, ValueError):
            raise ValueError("table read contains an invalid turn index")
        if turn in by_turn or not 1 <= turn <= len(manifest):
            raise ValueError("table read contains duplicate or out-of-range turns")
        by_turn[turn] = raw
    if set(by_turn) != set(range(1, len(manifest) + 1)):
        raise ValueError("table read changed the screenplay turn sequence")

    locked = _h3_user_locked_dialogue_map(story_description)
    revised_manifest: list[dict[str, Any]] = []
    changed = 0
    original_word_count = 0
    revised_word_count = 0
    for turn, original in enumerate(manifest, start=1):
        raw = by_turn[turn]
        original_speaker = str(original.get("speaker_name") or "").strip()
        returned_speaker = str(raw.get("speaker_name") or "").strip()
        if (
            _h3_speaker_name_tokens(original_speaker)
            != _h3_speaker_name_tokens(returned_speaker)
        ):
            raise ValueError(f"table read reassigned spoken turn {turn}")
        original_text = _h3_plain_dialogue_text(original.get("spoken_text"))
        if (
            _h3_dialogue_word_fingerprint(raw.get("original_text"))
            != _h3_dialogue_word_fingerprint(original_text)
        ):
            raise ValueError(f"table read changed the source text for turn {turn}")

        candidate = _h3_plain_dialogue_text(raw.get("revised_text"))
        if not candidate:
            raise ValueError(f"table read removed spoken turn {turn}")
        original_fingerprint = _h3_dialogue_word_fingerprint(original_text)
        if original_fingerprint in locked:
            candidate = locked[original_fingerprint]

        # A line must fit the effective native pass selected for this run.
        # Leave an already-long screenplay line untouched so the established
        # duration allocator can report/handle it instead of accepting a new
        # table-read regression.
        if len(candidate.split()) > max(1, int(maximum_line_words)):
            candidate = original_text

        updated = copy.deepcopy(original)
        updated["spoken_text"] = candidate
        delivery = re.sub(
            r"\s+",
            " ",
            _normalize_h3_text(raw.get("delivery") or ""),
        ).strip(" .")
        if delivery:
            source_beat = dict(updated.get("source_beat") or {})
            source_beat["delivery"] = delivery
            updated["source_beat"] = source_beat
        revised_manifest.append(updated)
        original_word_count += len(original_text.split())
        revised_word_count += len(candidate.split())
        if _h3_dialogue_word_fingerprint(candidate) != original_fingerprint:
            changed += 1

    # The characterization pass may tighten an over-budget screenplay, but it
    # may never create a new timing overrun or make an existing one worse.
    allowed_total = max(int(max_spoken_words or 0), original_word_count)
    if revised_word_count > allowed_total:
        raise ValueError(
            "table read increased dialogue beyond the available timing budget"
        )
    return revised_manifest, changed


def _h3_subject_matches_speaker(subject: dict, speaker_name: str) -> bool:
    wanted = _h3_speaker_name_tokens(speaker_name)
    if not wanted:
        return False
    candidates = [
        subject.get("speaker_name"),
        subject.get("visual_description"),
    ]
    for candidate in candidates:
        tokens = _h3_speaker_name_tokens(candidate)
        if not tokens:
            continue
        if tokens == wanted or tokens[:len(wanted)] == wanted:
            return True
        # Pass 2 sometimes expands a one-name heading (JOEY) to a full name
        # and occasionally misspells the surname. The unique first name is
        # still a reliable local identity anchor for that shot.
        if len(wanted) == 1 and tokens[0] == wanted[0]:
            return True
        if len(wanted) > 1 and tokens[0] == wanted[0] and len(tokens) > 1:
            return True
    return False


def _reconcile_h3_dialogue_manifest(
    items: list[dict],
    manifest: list[dict[str, Any]],
    *,
    known_items: Optional[list[dict]] = None,
    allow_manifest_restore: bool = False,
    allow_manifest_sentence_splits: bool = False,
) -> list[dict]:
    """Bind exact screenplay lines to their semantic LLM-planned shot slots.

    Dialogue is never moved by duration. The planned line must remain in the
    shot whose visible cast contains its screenplay speaker. Missing lines,
    reordering, or ambiguous rewrites are rejected before video jobs queue.
    ``allow_manifest_restore`` is reserved for a whole-plan repair whose turn
    count and visible speakers have already survived validation. It lets the
    locked screenplay manifest replace a repair model's duplicated or altered
    words and incorrect speaker ID without trusting any rewritten dialogue.
    ``allow_manifest_sentence_splits`` accepts only the exact same locked
    speaker/word stream after the deterministic timing allocator has split an
    overlong screenplay turn at sentence boundaries. It never accepts extra,
    missing, reordered, or rewritten planner dialogue.
    """

    planned: list[tuple[int, dict, dict]] = []
    for shot_index, raw in enumerate(items or []):
        for beat in raw.get("dialogue_beats") or []:
            if isinstance(beat, dict) and _h3_plain_dialogue_text(
                beat.get("spoken_text")
            ):
                planned.append((shot_index, raw, beat))

    if len(planned) != len(manifest) and allow_manifest_sentence_splits:
        canonical_source = _h3_manifest_dialogue_source(
            manifest,
            [*(known_items or []), *(items or [])],
        )
        if (
            _h3_dialogue_signature(items)
            == _h3_dialogue_signature(canonical_source)
        ):
            for event in _h3_dialogue_events(items):
                shot_index = int(event.get("source_index") or 0)
                if not _h3_speaker_is_visible(
                    items[shot_index],
                    event.get("speaker_key") or "",
                ):
                    raise ValueError(
                        "the deterministic dialogue allocation placed a "
                        f"speaker outside the visible cast of shot "
                        f"{shot_index + 1}"
                    )
            return items

    if len(planned) != len(manifest):
        raise ValueError(
            f"the screenplay contains {len(manifest)} spoken turns but the "
            f"shot plan contains {len(planned)}"
        )

    known_ids: dict[str, str] = {}
    used_ids: set[str] = set()
    for collection in (known_items or [], items or []):
        for raw in collection:
            for subject in raw.get("subjects_on_screen") or []:
                if not isinstance(subject, dict):
                    continue
                character_id = str(subject.get("character_id") or "").strip()
                if not character_id:
                    continue
                used_ids.add(character_id)
                for entry in manifest:
                    speaker_name = entry.get("speaker_name") or ""
                    if _h3_subject_matches_speaker(subject, speaker_name):
                        known_ids.setdefault(speaker_name.casefold(), character_id)

    for entry in manifest:
        speaker_name = str(entry.get("speaker_name") or "speaker").strip()
        key = speaker_name.casefold()
        if key in known_ids:
            continue
        requested_id = str(entry.get("speaker_id") or "").strip()
        if requested_id and requested_id not in used_ids:
            character_id = requested_id
        else:
            base = re.sub(r"[^a-z0-9]+", "_", key).strip("_") or "speaker"
            character_id = f"dialogue_{base}"
            suffix = 2
            while character_id in used_ids:
                character_id = f"dialogue_{base}_{suffix}"
                suffix += 1
        known_ids[key] = character_id
        used_ids.add(character_id)

    for event_index, ((shot_index, raw, beat), entry) in enumerate(
        zip(planned, manifest),
        start=1,
    ):
        canonical_text = _h3_plain_dialogue_text(entry.get("spoken_text"))
        planned_fingerprint = _h3_dialogue_word_fingerprint(
            beat.get("spoken_text")
        )
        canonical_fingerprint = _h3_dialogue_word_fingerprint(canonical_text)
        speaker_name = str(entry.get("speaker_name") or "speaker").strip()
        character_id = known_ids[speaker_name.casefold()]
        subjects = [
            subject for subject in (raw.get("subjects_on_screen") or [])
            if isinstance(subject, dict)
        ]
        speaker_subjects = [
            subject for subject in subjects
            if _h3_subject_matches_speaker(subject, speaker_name)
        ]
        if not speaker_subjects:
            raise ValueError(
                f"spoken turn {event_index} belongs to {speaker_name}, but "
                f"that person is not visible in shot {shot_index + 1}"
            )

        if planned_fingerprint != canonical_fingerprint:
            declared_id = str(beat.get("speaker_id") or "").strip()
            sole_matching_subject = (
                len(subjects) == 1 and len(speaker_subjects) == 1
            )
            allowed_ids = {
                value for value in (
                    character_id,
                    str(entry.get("speaker_id") or "").strip(),
                ) if value
            }
            if (
                not (declared_id and declared_id in allowed_ids)
                and not sole_matching_subject
                and not allow_manifest_restore
            ):
                raise ValueError(
                    f"spoken turn {event_index} changed or moved relative to "
                    "the screenplay"
                )
            if (
                allow_manifest_restore
                and not (declared_id and declared_id in allowed_ids)
                and not sole_matching_subject
            ):
                # The repair model attached a rewritten/duplicated line to a
                # different visible person. The manifest supplies the words
                # and speaker; retain only the shot slot and replace cues that
                # may describe the incorrect speaker.
                beat["delivery"] = "natural and context-appropriate"
                beat["physical_cue"] = (
                    f"{speaker_name} visibly delivers the line while "
                    "remaining in the described blocking."
                )

        for subject in speaker_subjects:
            subject["character_id"] = character_id
            subject.setdefault("speaker_name", speaker_name.title())
        beat["speaker_id"] = character_id
        beat["spoken_text"] = canonical_text

        source_beat = entry.get("source_beat")
        if isinstance(source_beat, dict):
            for field in ("delivery", "physical_cue", "priority"):
                # A validated table read is upstream of the immutable
                # screenplay manifest, so its performance direction is more
                # authoritative than Pass 2's generic "conversational" label.
                if source_beat.get(field):
                    beat[field] = source_beat[field]

    return items


def _h3_dialogue_manifest_prompt(manifest: list[dict[str, Any]]) -> str:
    payload = [
        {
            "turn": index,
            "speaker_name": entry.get("speaker_name") or "speaker",
            "spoken_text": entry.get("spoken_text") or "",
        }
        for index, entry in enumerate(manifest, start=1)
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _h3_manifest_dialogue_source(
    manifest: list[dict[str, Any]],
    known_items: list[dict],
) -> list[dict]:
    """Build an authoritative dialogue event stream independent of Pass 2.

    Pass 1's screenplay is the source of truth for words and speaker order.
    Pass 2 may still provide useful subject identity/wardrobe templates, but
    its dialogue array is only a placement hint and may contain duplicated or
    omitted turns. This representation lets the deterministic allocator place
    every locked screenplay turn into a visual plan without trusting Pass 2's
    rewritten dialogue.
    """

    templates: dict[str, dict] = {}
    used_ids: set[str] = set()
    for raw in known_items or []:
        if not isinstance(raw, dict):
            continue
        for subject in raw.get("subjects_on_screen") or []:
            if not isinstance(subject, dict):
                continue
            character_id = str(subject.get("character_id") or "").strip()
            if character_id:
                used_ids.add(character_id)
            for entry in manifest:
                speaker_name = str(entry.get("speaker_name") or "").strip()
                key = speaker_name.casefold()
                if key and key not in templates and _h3_subject_matches_speaker(
                    subject,
                    speaker_name,
                ):
                    templates[key] = copy.deepcopy(subject)

    ids_by_speaker: dict[str, str] = {}
    source_items: list[dict] = []
    for entry in manifest:
        speaker_name = str(entry.get("speaker_name") or "speaker").strip()
        key = speaker_name.casefold()
        subject = copy.deepcopy(templates.get(key) or {})
        character_id = str(subject.get("character_id") or "").strip()
        if not character_id:
            character_id = ids_by_speaker.get(key, "")
        if not character_id:
            requested_id = str(entry.get("speaker_id") or "").strip()
            if requested_id and requested_id not in used_ids:
                character_id = requested_id
            else:
                base = re.sub(r"[^a-z0-9]+", "_", key).strip("_") or "speaker"
                character_id = f"dialogue_{base}"
                suffix = 2
                while character_id in used_ids:
                    character_id = f"dialogue_{base}_{suffix}"
                    suffix += 1
        ids_by_speaker[key] = character_id
        used_ids.add(character_id)
        subject.update({
            "character_id": character_id,
            "speaker_name": (
                str(subject.get("speaker_name") or "").strip()
                or speaker_name.title()
            ),
            "visual_description": (
                str(subject.get("visual_description") or "").strip()
                or speaker_name
            ),
            "position_or_relation": (
                str(subject.get("position_or_relation") or "").strip()
                or "visible in the shot near the other speaking characters"
            ),
            "wardrobe": str(subject.get("wardrobe") or "").strip(),
        })

        source_beat = entry.get("source_beat")
        source_beat = source_beat if isinstance(source_beat, dict) else {}
        source_items.append({
            "subjects_on_screen": [subject],
            "dialogue_beats": [{
                "speaker_id": character_id,
                "spoken_text": _h3_plain_dialogue_text(
                    entry.get("spoken_text")
                ),
                "delivery": str(
                    source_beat.get("delivery")
                    or "natural and context-appropriate"
                ).strip(),
                "physical_cue": str(
                    source_beat.get("physical_cue")
                    or f"{speaker_name} visibly delivers the line."
                ).strip(),
                "priority": str(source_beat.get("priority") or "high").strip(),
            }],
        })
    return source_items


def _h3_native_structure_issues(
    items: list[dict],
    required: list[str],
    *,
    minimum_items: int,
    maximum_items: int,
) -> list[str]:
    """Detect truncated json_repair output before normalization masks it."""

    issues: list[str] = []
    if not minimum_items <= len(items or []) <= maximum_items:
        issues.append(
            f"returned {len(items or [])} shots; expected "
            f"{minimum_items}-{maximum_items}"
        )
    for index, raw in enumerate(items or [], start=1):
        if not isinstance(raw, dict):
            issues.append(f"shot {index} is not an object")
            continue
        missing = [field for field in required if field not in raw]
        if missing:
            issues.append(f"shot {index} is missing {', '.join(missing)}")
    return issues


def _h3_planner_token_budget(target_duration: float) -> int:
    """Leave enough room for complete, self-contained H3 shot JSON."""

    # Keep enough headroom for the 32K-context Director models' system prompt
    # and (when enabled) their separate reasoning budget. H3's native plan is
    # unusually verbose because each independently generated shot must repeat
    # its complete world, cast, blocking, audio, and prompt context. The prior
    # 200-token/second allowance hit its exact ceiling on a valid 90-second
    # plan and left the final object half-written.
    return min(23000, max(12288, int(math.ceil(target_duration * 240))))


def _h3_dialogue_events(items: list[dict]) -> list[dict]:
    """Capture immutable dialogue plus its original speaker/subject context."""

    events: list[dict] = []
    for shot_index, raw in enumerate(items or []):
        subjects = [
            subject for subject in (raw.get("subjects_on_screen") or [])
            if isinstance(subject, dict)
        ]
        for beat in raw.get("dialogue_beats") or []:
            if not isinstance(beat, dict):
                continue
            spoken = _h3_plain_dialogue_text(beat.get("spoken_text"))
            if not spoken:
                continue
            canonical = dict(beat)
            canonical["spoken_text"] = spoken
            speaker_key = re.sub(
                r"\s+", " ", str(beat.get("speaker_id") or "")
            ).strip().casefold()
            source_subject = None
            for subject in subjects:
                keys = {
                    re.sub(r"\s+", " ", str(subject.get(field) or ""))
                    .strip().casefold()
                    for field in ("character_id", "speaker_name")
                }
                if speaker_key and speaker_key in keys:
                    source_subject = dict(subject)
                    break
            events.append({
                "beat": canonical,
                "speaker_key": speaker_key,
                "source_index": shot_index,
                "source_subject": source_subject,
            })
    return events


def _h3_dialogue_signature(items: list[dict]) -> list[tuple[str, str]]:
    """Compare exact words and speakers while allowing sentence re-bucketing."""

    signature: list[tuple[str, str]] = []
    for event in _h3_dialogue_events(items):
        signature.extend(
            (event["speaker_key"], token)
            for token in event["beat"]["spoken_text"].split()
        )
    return signature


def _h3_speaker_is_visible(raw: dict, speaker_key: str) -> bool:
    if not speaker_key:
        return True
    for subject in raw.get("subjects_on_screen") or []:
        if not isinstance(subject, dict):
            continue
        keys = {
            re.sub(r"\s+", " ", str(subject.get(field) or ""))
            .strip().casefold()
            for field in ("character_id", "speaker_name")
        }
        if speaker_key in keys:
            return True
    return False


def _h3_rebuilt_visual_prompt(raw: dict) -> str:
    """Rebuild dialogue-free visual prose from the repair's structured data."""

    def clean(value: Any) -> str:
        text = _normalize_h3_text(value)
        if re.search(r"<\s*d\s*>", text, flags=re.IGNORECASE):
            return ""
        return re.sub(r"\s+", " ", text).strip(" .")

    parts: list[str] = []
    for field in ("scene_goal", "environment", "spatial_setup"):
        value = clean(raw.get(field))
        if value and value.casefold() not in {item.casefold() for item in parts}:
            parts.append(value)

    subject_details: list[str] = []
    for subject in raw.get("subjects_on_screen") or []:
        if not isinstance(subject, dict):
            continue
        name = clean(
            subject.get("speaker_name")
            or subject.get("character_id")
            or subject.get("visual_description")
        )
        description = clean(subject.get("visual_description"))
        wardrobe = clean(subject.get("wardrobe"))
        position = clean(subject.get("position_or_relation"))
        bits = [name]
        if description and description.casefold() != name.casefold():
            bits.append(description)
        if wardrobe:
            bits.append(f"wearing {wardrobe}")
        if position:
            bits.append(f"positioned {position}")
        if any(bits):
            subject_details.append(", ".join(bit for bit in bits if bit))
    if subject_details:
        parts.append("Visible cast: " + "; ".join(subject_details))

    actions = [clean(value) for value in raw.get("action_beats") or []]
    actions = [value for value in actions if value]
    if actions:
        parts.append("Action: " + " Then ".join(actions))

    camera = raw.get("camera_plan") or {}
    camera_bits = [
        clean(camera.get(field))
        for field in (
            "framing", "angle", "movement", "movement_intensity",
            "lens_feel", "reframing_notes",
        )
    ]
    camera_bits = [value for value in camera_bits if value]
    if camera_bits:
        parts.append("Camera: " + ", ".join(camera_bits))
    for label, field in (("Lighting", "lighting"), ("Mood", "mood")):
        value = clean(raw.get(field))
        if value:
            parts.append(f"{label}: {value}")
    ending = clean(raw.get("ending_beat") or raw.get("closing_blocking"))
    if ending:
        parts.append("Final beat: " + ending)

    audio = raw.get("audio_plan") or {}
    sound_bits = [clean(audio.get("ambience"))]
    sound_bits.extend(clean(value) for value in audio.get("effects") or [])
    sound_bits = [value for value in sound_bits if value]
    soundscape = ", ".join(sound_bits) or "Natural scene-appropriate stereo ambience"

    music = "N/A"
    old_prompt = str(raw.get("video_prompt") or "")
    music_match = re.search(
        r"\bnon_diegetic_music\s*:\s*(.+?)\s*$",
        old_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if music_match:
        candidate = clean(music_match.group(1))
        if candidate and len(candidate) <= 500:
            music = candidate

    body = ". ".join(value for value in parts if value)
    return (
        f"{body}. overall_soundscape: {soundscape}. "
        f"non_diegetic_music: {music}"
    ).strip()


def _complete_h3_truncated_tail(
    items: list[dict],
    required: list[str],
) -> list[str]:
    """Complete a token-capped final shot without inventing story content.

    ``json_repair`` can recover an array whose final object was cut off at the
    output-token ceiling. Recovery is safe only when every earlier shot is
    complete and the final shot already contains its semantic core: identity,
    setting, blocking, actions, and immutable dialogue. In that narrow case,
    the missing suffix consists only of derived production fields that Maestro
    can reconstruct deterministically. If dialogue or any other semantic core
    field is absent, leave the object untouched so normal validation rejects
    it instead of silently dropping or inventing screenplay content.

    Returns the names of fields filled on success, otherwise an empty list.
    """

    if not items or not isinstance(items[-1], dict):
        return []
    if any(
        not isinstance(raw, dict)
        or any(field not in raw for field in required)
        for raw in items[:-1]
    ):
        return []

    tail = items[-1]
    missing = [field for field in required if field not in tail]
    if not missing:
        return []

    semantic_core = {
        "title", "duration_sec", "scene_goal", "narrative_role",
        "scene_type", "continuity_strategy", "continuity_group",
        "subjects_on_screen", "spatial_setup", "environment",
        "visual_style", "lighting", "mood", "action_beats",
        "dialogue_beats",
    }
    recoverable_suffix = {
        "camera_plan", "audio_plan", "ending_beat", "closing_blocking",
        "image_source", "image_prompt", "visual_changes", "video_prompt",
        "multishot", "window_prompts",
    }
    if any(field not in tail for field in semantic_core):
        return []
    if any(field not in recoverable_suffix for field in missing):
        return []
    if not isinstance(tail.get("subjects_on_screen"), list):
        return []
    if not isinstance(tail.get("action_beats"), list):
        return []
    if not isinstance(tail.get("dialogue_beats"), list):
        return []

    dialogue_beats = [
        beat for beat in tail.get("dialogue_beats") or []
        if isinstance(beat, dict)
        and _h3_plain_dialogue_text(beat.get("spoken_text"))
    ]
    action_beats = [
        re.sub(r"\s+", " ", str(value or "")).strip()
        for value in tail.get("action_beats") or []
    ]
    action_beats = [value for value in action_beats if value]
    spatial_setup = re.sub(
        r"\s+", " ", str(tail.get("spatial_setup") or "")
    ).strip()
    scene_goal = re.sub(
        r"\s+", " ", str(tail.get("scene_goal") or "")
    ).strip()

    if "camera_plan" in missing:
        tail["camera_plan"] = {
            "framing": "medium shot",
            "angle": "eye level",
            "movement": "static hold",
            "movement_intensity": "static",
            "lens_feel": "natural cinematic perspective",
            "reframing_notes": "Keep every visible subject clearly framed.",
        }
    if "audio_plan" in missing:
        has_dialogue = bool(dialogue_beats)
        tail["audio_plan"] = {
            "mode": "dialogue_driven" if has_dialogue else "ambient_only",
            "ambience": "Natural scene-appropriate stereo ambience",
            "effects": [],
            "vocal_style": "Natural character voices",
            "timing_anchor": "audio" if has_dialogue else "video",
            "lip_sync_critical": has_dialogue,
        }
    if "ending_beat" in missing:
        tail["ending_beat"] = (
            action_beats[-1] if action_beats else scene_goal or spatial_setup
        )
    if "closing_blocking" in missing:
        tail["closing_blocking"] = spatial_setup or str(
            tail.get("ending_beat") or scene_goal
        ).strip()
    if "image_source" in missing:
        strategy = str(tail.get("continuity_strategy") or "").casefold()
        tail["image_source"] = (
            "previous" if strategy in {"continuous", "extend_previous"}
            else "original"
        )
    if "image_prompt" in missing:
        static_parts = [
            str(tail.get(field) or "").strip()
            for field in (
                "environment", "visual_style", "lighting", "spatial_setup",
            )
        ]
        static_parts.extend(
            ", ".join(
                str(subject.get(field) or "").strip()
                for field in (
                    "visual_description", "wardrobe", "position_or_relation",
                )
                if str(subject.get(field) or "").strip()
            )
            for subject in tail.get("subjects_on_screen") or []
            if isinstance(subject, dict)
        )
        tail["image_prompt"] = ". ".join(
            value.strip(" .") for value in static_parts if value.strip(" .")
        )
    if "visual_changes" in missing:
        tail["visual_changes"] = []
    if "video_prompt" in missing:
        tail["video_prompt"] = _h3_rebuilt_visual_prompt(tail)
    if "multishot" in missing:
        tail["multishot"] = False
    if "window_prompts" in missing:
        tail["window_prompts"] = []
    return missing


def _restore_h3_dialogue_after_pacing_repair(
    original: list[dict],
    repaired: list[dict],
    durations: list[float],
    *,
    words_per_second: float = _H3_DIALOGUE_WORDS_PER_SECOND,
) -> list[dict]:
    """Overlay immutable dialogue onto an LLM-repaired visual shot plan.

    The repair model may change shot count, blocking, or timing, but it is not
    trusted to rewrite spoken words. A small dynamic program re-buckets whole
    speaker turns (or complete sentences when one turn is too large) across
    the repaired shot capacities without changing word order or speakers.
    """

    events = _h3_dialogue_events(original)
    if not events:
        for raw in repaired:
            raw["dialogue_beats"] = []
            raw["video_prompt"] = _h3_rebuilt_visual_prompt(raw)
        return repaired
    if not repaired or len(durations) != len(repaired):
        raise ValueError("the repaired shot schedule is incomplete")

    capacities = [
        max(0, int(math.floor(max(0.0, float(duration)) * words_per_second)))
        for duration in durations
    ]
    maximum_capacity = max(capacities, default=0)
    if maximum_capacity <= 0:
        raise ValueError("the repaired shot schedule has no dialogue capacity")

    # A multi-sentence turn may cross a shot boundary, but individual
    # sentences remain intact. This preserves the exact word/speaker stream.
    split_events: list[dict] = []
    for event in events:
        spoken = event["beat"]["spoken_text"]
        if len(spoken.split()) <= maximum_capacity:
            split_events.append(event)
            continue
        sentences = [
            value.strip()
            for value in re.split(r"(?<=[.!?])\s+", spoken)
            if value.strip()
        ]
        if not sentences or any(
            len(sentence.split()) > maximum_capacity for sentence in sentences
        ):
            raise ValueError(
                "one scripted sentence is longer than MiniMax H3's maximum "
                "single-shot dialogue budget"
            )
        groups: list[list[str]] = []
        current: list[str] = []
        current_words = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current and current_words + sentence_words > maximum_capacity:
                groups.append(current)
                current = []
                current_words = 0
            current.append(sentence)
            current_words += sentence_words
        if current:
            groups.append(current)
        for group in groups:
            clone = dict(event)
            clone["beat"] = dict(event["beat"])
            clone["beat"]["spoken_text"] = " ".join(group)
            split_events.append(clone)
    events = split_events

    total_words = sum(len(event["beat"]["spoken_text"].split()) for event in events)
    if total_words > sum(capacities):
        raise ValueError(
            f"the scripted dialogue needs {total_words} words of capacity but "
            f"the repaired timeline provides only {sum(capacities)}"
        )

    repair_slots = [
        shot_index
        for shot_index, raw in enumerate(repaired)
        for beat in (raw.get("dialogue_beats") or [])
        if isinstance(beat, dict) and _h3_plain_dialogue_text(beat.get("spoken_text"))
    ]
    desired: list[int] = []
    for event_index, event in enumerate(events):
        if repair_slots and (len(repair_slots) >= 2 or len(events) == 1):
            slot_index = (
                0 if len(events) == 1 else
                round(event_index * (len(repair_slots) - 1) / (len(events) - 1))
            )
            desired.append(repair_slots[slot_index])
        else:
            source_count = max(1, len(original))
            desired.append(min(
                len(repaired) - 1,
                round(
                    (event["source_index"] + 0.5)
                    * len(repaired) / source_count - 0.5
                ),
            ))

    event_words = [len(event["beat"]["spoken_text"].split()) for event in events]
    event_count = len(events)
    shot_count = len(repaired)
    infinity = float("inf")
    costs = [[infinity] * (event_count + 1) for _ in range(shot_count + 1)]
    parents: list[list[int | None]] = [
        [None] * (event_count + 1) for _ in range(shot_count + 1)
    ]
    costs[0][0] = 0.0
    for shot_no in range(shot_count):
        for start in range(event_count + 1):
            if not math.isfinite(costs[shot_no][start]):
                continue
            words = 0
            for end in range(start, event_count + 1):
                if end > start:
                    words += event_words[end - 1]
                if words > capacities[shot_no]:
                    break
                segment_cost = 0.0
                for event_no in range(start, end):
                    segment_cost += abs(shot_no - desired[event_no]) * 10.0
                    if not _h3_speaker_is_visible(
                        repaired[shot_no], events[event_no]["speaker_key"]
                    ):
                        segment_cost += 1.0
                candidate = costs[shot_no][start] + segment_cost
                if candidate < costs[shot_no + 1][end]:
                    costs[shot_no + 1][end] = candidate
                    parents[shot_no + 1][end] = start

    if not math.isfinite(costs[shot_count][event_count]):
        raise ValueError(
            "the original complete dialogue turns cannot fit the repaired "
            "per-shot timing without changing words"
        )

    assignments: list[list[int]] = [[] for _ in repaired]
    end = event_count
    for shot_no in range(shot_count, 0, -1):
        start = parents[shot_no][end]
        if start is None:
            raise ValueError("the deterministic dialogue allocation is incomplete")
        assignments[shot_no - 1] = list(range(start, end))
        end = start

    for shot_no, raw in enumerate(repaired):
        raw["dialogue_beats"] = [
            dict(events[event_no]["beat"])
            for event_no in assignments[shot_no]
        ]
        subjects = [
            subject for subject in (raw.get("subjects_on_screen") or [])
            if isinstance(subject, dict)
        ]
        for event_no in assignments[shot_no]:
            event = events[event_no]
            source_subject = event.get("source_subject")
            if not source_subject or _h3_speaker_is_visible(raw, event["speaker_key"]):
                continue
            subjects.append(dict(source_subject))
            raw["subjects_on_screen"] = subjects
        audio = dict(raw.get("audio_plan") or {})
        if raw["dialogue_beats"]:
            audio.update({
                "mode": "dialogue_driven",
                "timing_anchor": "audio",
                "lip_sync_critical": True,
            })
        raw["audio_plan"] = audio
        raw["video_prompt"] = _h3_rebuilt_visual_prompt(raw)

    if _h3_dialogue_signature(repaired) != _h3_dialogue_signature(original):
        raise ValueError("the deterministic dialogue overlay failed its integrity check")
    return repaired


def _coalesce_h3_dialogue_shots(
    shot_dicts: list[dict],
    *,
    fps: float,
    minimum_frames: int,
    maximum_frames: int,
    frame_step: int,
    minimum_shots: int,
    words_per_second: float = _H3_DIALOGUE_WORDS_PER_SECOND,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Merge safe adjacent conversation beats into native H3 clips.

    Pass 2 sometimes treats a speaker change as an edit even though H3 can
    cover a short exchange with internal speaker-motivated cuts and reframes.
    Only adjacent shots in the same uninterrupted location are eligible. Both
    must contain dialogue, every speaker must already be visible in the first
    frame, the combined action load must remain modest, and the complete exact
    dialogue stream must fit one legal H3 generation. Merges are pairwise so a
    compact reaction cannot swallow an entire sequence of distinct visual
    beats, and ``minimum_shots`` preserves enough clips to cover the requested
    project runtime at the model's maximum frame count.
    """

    items = [copy.deepcopy(raw) for raw in (shot_dicts or [])]
    if len(items) < 2:
        return items, []

    fps = max(1.0, float(fps or 24))
    minimum_frames = max(1, int(minimum_frames or 1))
    maximum_frames = max(minimum_frames, int(maximum_frames or minimum_frames))
    frame_step = max(1, int(frame_step or 1))
    minimum_shots = max(1, int(minimum_shots or 1))
    merge_budget = max(0, len(items) - minimum_shots)
    if merge_budget <= 0:
        return items, []

    valid_frames = list(range(
        minimum_frames,
        maximum_frames + 1,
        frame_step,
    ))
    maximum_words = int(math.floor(
        maximum_frames / fps * max(0.1, float(words_per_second)),
    ))

    def normalized_key(value: Any) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            " ",
            _normalize_h3_text(value).casefold(),
        ).strip()

    def dialogue_beats(raw: dict) -> list[dict]:
        return [
            beat for beat in (raw.get("dialogue_beats") or [])
            if isinstance(beat, dict)
            and _h3_plain_dialogue_text(beat.get("spoken_text"))
        ]

    def spoken_words(raw: dict) -> int:
        return sum(
            len(_h3_plain_dialogue_text(beat.get("spoken_text")).split())
            for beat in dialogue_beats(raw)
        )

    def speaker_keys(raw: dict) -> list[str]:
        return [
            re.sub(
                r"\s+",
                " ",
                str(beat.get("speaker_id") or ""),
            ).strip().casefold()
            for beat in dialogue_beats(raw)
            if str(beat.get("speaker_id") or "").strip()
        ]

    def unique_text(values: list[Any]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                result.append(text)
        return result

    def combine_text(first: Any, second: Any, *, separator: str = "; ") -> str:
        return separator.join(unique_text([first, second]))

    def pair_frame_count(first: dict, second: dict) -> Optional[int]:
        words = spoken_words(first) + spoken_words(second)
        if not 0 < words <= maximum_words:
            return None
        dialogue_floor = math.ceil(
            words * fps / max(0.1, float(words_per_second)),
        )
        longest_existing = max(
            int(round(float(first.get("duration_sec") or 0) * fps)),
            int(round(float(second.get("duration_sec") or 0) * fps)),
        )
        requested = max(minimum_frames, dialogue_floor, longest_existing)
        return next(
            (frames for frames in valid_frames if frames >= requested),
            None,
        )

    def can_merge(first: dict, second: dict) -> tuple[bool, Optional[int]]:
        first_beats = dialogue_beats(first)
        second_beats = dialogue_beats(second)
        if not first_beats or not second_beats:
            return False, None
        first_group = normalized_key(first.get("continuity_group"))
        second_group = normalized_key(second.get("continuity_group"))
        if not first_group or first_group != second_group:
            return False, None
        first_environment = normalized_key(first.get("environment"))
        second_environment = normalized_key(second.get("environment"))
        if (
            first_environment
            and second_environment
            and first_environment != second_environment
        ):
            return False, None
        if first.get("multishot") is True or second.get("multishot") is True:
            return False, None
        actions = [
            value for value in [
                *(first.get("action_beats") or []),
                *(second.get("action_beats") or []),
            ]
            if str(value or "").strip()
        ]
        if len(actions) > 6:
            return False, None
        speakers = speaker_keys(first) + speaker_keys(second)
        if len(set(speakers)) < 2:
            return False, None
        if any(
            not _h3_speaker_is_visible(first, speaker_key)
            for speaker_key in speakers
        ):
            return False, None
        frames = pair_frame_count(first, second)
        return frames is not None, frames

    def speaker_sequence(raws: list[dict], subjects: list[dict]) -> str:
        names_by_key: dict[str, str] = {}
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            name = str(
                subject.get("speaker_name")
                or subject.get("character_id")
                or "the current speaker"
            ).strip()
            for field in ("character_id", "speaker_name"):
                key = re.sub(
                    r"\s+",
                    " ",
                    str(subject.get(field) or ""),
                ).strip().casefold()
                if key:
                    names_by_key[key] = name
        sequence: list[str] = []
        for raw in raws:
            for key in speaker_keys(raw):
                name = names_by_key.get(key, key or "the current speaker")
                # Keep a later return to the same person (Ross, Monica, Ross)
                # because that order is the internal camera choreography. Only
                # collapse an accidental immediately repeated speaker label.
                if not sequence or sequence[-1].casefold() != name.casefold():
                    sequence.append(name)
        return ", then ".join(sequence) or "each current speaker in order"

    def merged_subjects(first: dict, second: dict) -> list[dict]:
        subjects = [
            copy.deepcopy(subject)
            for subject in (first.get("subjects_on_screen") or [])
            if isinstance(subject, dict)
        ]
        known: dict[str, dict] = {}
        for index, subject in enumerate(subjects):
            for field in ("character_id", "speaker_name"):
                key = normalized_key(subject.get(field))
                if key:
                    known[key] = subject
            known.setdefault(f"index {index}", subject)
        for subject in second.get("subjects_on_screen") or []:
            if not isinstance(subject, dict):
                continue
            keys = [
                normalized_key(subject.get(field))
                for field in ("character_id", "speaker_name")
            ]
            existing = next((known[key] for key in keys if key in known), None)
            if existing is None:
                clone = copy.deepcopy(subject)
                subjects.append(clone)
                for key in keys:
                    if key:
                        known[key] = clone
                continue
            for field in (
                "visual_description", "character_id", "speaker_name",
                "wardrobe",
            ):
                if not existing.get(field) and subject.get(field):
                    existing[field] = copy.deepcopy(subject[field])
        return subjects

    intensity_order = {"static": 0, "subtle": 1, "moderate": 2, "dynamic": 3}

    def merge_pair(first: dict, second: dict, frames: int) -> dict:
        merged = copy.deepcopy(first)
        subjects = merged_subjects(first, second)
        first_camera = dict(first.get("camera_plan") or {})
        second_camera = dict(second.get("camera_plan") or {})
        sequence = speaker_sequence([first, second], subjects)
        opening_framing = str(
            first_camera.get("framing") or "an ensemble dialogue frame"
        ).strip()
        closing_framing = str(
            second_camera.get("framing") or opening_framing
        ).strip()
        first_movement = str(
            first_camera.get("movement") or "a stable opening hold"
        ).strip()
        second_movement = str(
            second_camera.get("movement") or "a stable closing hold"
        ).strip()
        intensity = max(
            (
                str(first_camera.get("movement_intensity") or "subtle"),
                str(second_camera.get("movement_intensity") or "subtle"),
            ),
            key=lambda value: intensity_order.get(value, 1),
        )
        prior_notes = combine_text(
            first_camera.get("reframing_notes"),
            second_camera.get("reframing_notes"),
        )
        conversation_notes = (
            "Treat this as one continuous H3 conversation clip. Begin on "
            f"{opening_framing}; before each tagged line, use a clean "
            "speaker-motivated internal cut or reframe in this order: "
            f"{sequence}. Hold each speaker's unobstructed face and mouth for "
            f"their complete line, include natural reactions, and finish on "
            f"{closing_framing}."
        )
        merged["title"] = combine_text(
            first.get("title"),
            second.get("title"),
            separator=" / ",
        )
        merged["duration_sec"] = frames / fps
        merged["scene_goal"] = combine_text(
            first.get("scene_goal"),
            second.get("scene_goal"),
            separator=" Then ",
        )
        merged["narrative_role"] = (
            second.get("narrative_role") or first.get("narrative_role")
        )
        merged["scene_type"] = "dialogue"
        merged["subjects_on_screen"] = subjects
        merged["action_beats"] = unique_text([
            *(first.get("action_beats") or []),
            *(second.get("action_beats") or []),
        ])
        merged["dialogue_beats"] = [
            copy.deepcopy(beat)
            for beat in [*dialogue_beats(first), *dialogue_beats(second)]
        ]
        merged["camera_plan"] = {
            "framing": (
                "continuous dialogue coverage beginning on "
                f"{opening_framing} and ending on {closing_framing}"
            ),
            "angle": (
                first_camera.get("angle") or second_camera.get("angle")
            ),
            "movement": (
                f"{first_movement}; then speaker-motivated internal cuts and "
                f"reframes; finish with {second_movement}"
            ),
            "movement_intensity": intensity,
            "lens_feel": (
                first_camera.get("lens_feel")
                or second_camera.get("lens_feel")
            ),
            "reframing_notes": combine_text(
                prior_notes,
                conversation_notes,
                separator=". ",
            ),
        }
        first_audio = dict(first.get("audio_plan") or {})
        second_audio = dict(second.get("audio_plan") or {})
        merged["audio_plan"] = {
            "mode": "dialogue_driven",
            "ambience": combine_text(
                first_audio.get("ambience"),
                second_audio.get("ambience"),
            ),
            "effects": unique_text([
                *(first_audio.get("effects") or []),
                *(second_audio.get("effects") or []),
            ]),
            "vocal_style": combine_text(
                first_audio.get("vocal_style"),
                second_audio.get("vocal_style"),
            ),
            "timing_anchor": "audio",
            "lip_sync_critical": True,
        }
        merged["ending_beat"] = (
            second.get("ending_beat") or first.get("ending_beat")
        )
        merged["closing_blocking"] = (
            second.get("closing_blocking")
            or second.get("spatial_setup")
            or first.get("closing_blocking")
        )
        merged["visual_changes"] = unique_text([
            *(first.get("visual_changes") or []),
            *(second.get("visual_changes") or []),
        ])
        merged["multishot"] = False
        merged["window_prompts"] = []
        merged["video_prompt"] = _h3_rebuilt_visual_prompt(merged)
        return merged

    original_signature = _h3_dialogue_signature(items)
    merged_items: list[dict] = []
    merges: list[tuple[int, int]] = []
    index = 0
    while index < len(items):
        if index + 1 < len(items) and merge_budget > 0:
            allowed, frames = can_merge(items[index], items[index + 1])
            if allowed and frames is not None:
                merged_items.append(merge_pair(
                    items[index],
                    items[index + 1],
                    frames,
                ))
                merges.append((index + 1, index + 2))
                merge_budget -= 1
                index += 2
                continue
        merged_items.append(items[index])
        index += 1

    if _h3_dialogue_signature(merged_items) != original_signature:
        raise ValueError(
            "H3 conversation coalescing changed the locked dialogue stream"
        )
    return merged_items, merges


def _insert_h3_visual_detail(prompt: str, label: str, detail: str) -> str:
    """Place deterministic blocking details inside H3's visual section."""

    prompt = str(prompt or "").strip()
    detail = re.sub(r"\s+", " ", str(detail or "")).strip()
    if not detail:
        return prompt
    statement = f"{label}: {detail}"
    if statement in prompt:
        return prompt
    boundary = re.search(
        r"\b(?:overall_soundscape|non_diegetic_music)\s*:",
        prompt,
        flags=re.IGNORECASE,
    )
    if boundary:
        return (
            f"{prompt[:boundary.start()].rstrip()} {statement}. "
            f"{prompt[boundary.start():].lstrip()}"
        ).strip()
    return f"{prompt} {statement}.".strip()


def _enforce_h3_speaker_visual_contract(
    shot_dicts: list[dict],
    voice_bible: Optional[list[dict[str, str]]] = None,
) -> list[dict]:
    """Keep every H3 speaker visible and carry stable performance guidance."""

    profiles = {
        str(row.get("character_name") or "").strip().casefold(): row
        for row in (voice_bible or [])
        if isinstance(row, dict) and row.get("character_name")
    }
    subject_templates: dict[str, dict] = {}
    for raw in shot_dicts or []:
        if not isinstance(raw, dict):
            continue
        for subject in raw.get("subjects_on_screen") or []:
            if not isinstance(subject, dict):
                continue
            name = str(subject.get("speaker_name") or "").strip().casefold()
            if name:
                subject_templates.setdefault(name, copy.deepcopy(subject))

    def profile_for(name: str) -> Optional[dict[str, str]]:
        key = str(name or "").strip().casefold()
        if key in profiles:
            return profiles[key]
        wanted = _h3_speaker_name_tokens(key)
        for profile_name, profile in profiles.items():
            candidate = _h3_speaker_name_tokens(profile_name)
            if wanted and candidate and wanted[0] == candidate[0]:
                return profile
        return None

    for shot_index, raw in enumerate(shot_dicts or [], start=1):
        if not isinstance(raw, dict):
            continue
        subjects = [
            subject for subject in (raw.get("subjects_on_screen") or [])
            if isinstance(subject, dict)
        ]
        structured_cast_text = " ".join([
            str(raw.get("spatial_setup") or ""),
            *(str(value or "") for value in raw.get("action_beats") or []),
            str(raw.get("ending_beat") or ""),
            str(raw.get("closing_blocking") or ""),
        ])
        folded_cast_text = structured_cast_text.casefold()
        for template_name, template in subject_templates.items():
            name_tokens = _h3_speaker_name_tokens(template_name)
            aliases = [template_name]
            if name_tokens and len(name_tokens[0]) >= 3:
                aliases.append(name_tokens[0])
            mention = next(
                (
                    re.search(
                        rf"\b{re.escape(alias)}\b",
                        folded_cast_text,
                        flags=re.IGNORECASE,
                    )
                    for alias in aliases
                    if re.search(
                        rf"\b{re.escape(alias)}\b",
                        folded_cast_text,
                        flags=re.IGNORECASE,
                    )
                ),
                None,
            )
            if not mention or any(
                any(
                    candidate_tokens
                    and name_tokens
                    and candidate_tokens[0] == name_tokens[0]
                    for candidate_tokens in (
                        _h3_speaker_name_tokens(subject.get("speaker_name")),
                        _h3_speaker_name_tokens(subject.get("character_id")),
                    )
                )
                for subject in subjects
            ):
                continue
            nearby = folded_cast_text[
                max(0, mention.start() - 35):mention.end() + 55
            ]
            if re.search(
                r"\b(?:off[- ]?screen|off[- ]?camera|outside the frame)\b",
                nearby,
            ):
                continue
            restored = copy.deepcopy(template)
            restored["position_or_relation"] = (
                "in the exact position and pose stated in spatial_setup"
            )
            subjects.append(restored)
        raw["subjects_on_screen"] = subjects
        visible_speakers: list[str] = []
        performance_directions: list[str] = []
        for beat in raw.get("dialogue_beats") or []:
            if not isinstance(beat, dict) or not _h3_plain_dialogue_text(
                beat.get("spoken_text")
            ):
                continue
            speaker_id = str(beat.get("speaker_id") or "").strip()
            subject = next(
                (
                    item for item in subjects
                    if speaker_id.casefold() in {
                        str(item.get("character_id") or "").strip().casefold(),
                        str(item.get("speaker_name") or "").strip().casefold(),
                    }
                ),
                None,
            )
            if subject is None:
                raise ValueError(
                    f"H3 speaker {speaker_id or 'unknown'} is not visible in "
                    f"shot {shot_index}"
                )
            speaker_name = str(
                subject.get("speaker_name")
                or subject.get("character_id")
                or speaker_id
                or "the speaker"
            ).strip()
            if speaker_name not in visible_speakers:
                visible_speakers.append(speaker_name)
            profile = profile_for(speaker_name)
            performance = re.sub(
                r"\s+",
                " ",
                str((profile or {}).get("performance_direction") or ""),
            ).strip(" .")
            delivery = re.sub(
                r"\s+", " ", str(beat.get("delivery") or "")
            ).strip(" .")
            if performance:
                if performance.casefold() not in delivery.casefold():
                    delivery = (
                        f"{performance}; {delivery}" if delivery else performance
                    )
                if performance not in performance_directions:
                    performance_directions.append(performance)
            beat["delivery"] = delivery or "natural and character-appropriate"

        if not visible_speakers:
            continue
        names = ", ".join(visible_speakers)
        visibility = (
            f"Keep {names} visibly framed whenever they speak. Reframe to the "
            "current speaker before each line; their face and mouth remain "
            "unobstructed for the complete line, and reaction framing follows "
            "only after that line ends"
        )
        camera = dict(raw.get("camera_plan") or {})
        notes = re.sub(
            r"\s+", " ", str(camera.get("reframing_notes") or "")
        ).strip(" .")
        if "mouth remain unobstructed" not in notes.casefold():
            camera["reframing_notes"] = (
                f"{notes}. {visibility}" if notes else visibility
            )
        raw["camera_plan"] = camera
        audio = dict(raw.get("audio_plan") or {})
        if performance_directions:
            audio["vocal_style"] = "; ".join(performance_directions)
        raw["audio_plan"] = audio
        # Rebuild from the now-validated structured shot. This removes stale
        # Pass 2 cast/camera prose and all embedded dialogue so the canonical
        # H3 compiler can inject the table-read words and delivery exactly
        # once from dialogue_beats.
        raw["video_prompt"] = _h3_rebuilt_visual_prompt(raw)
        raw["video_prompt"] = _insert_h3_visual_detail(
            raw.get("video_prompt", ""),
            "SPEAKER VISIBILITY",
            visibility,
        )
    return shot_dicts


def _h3_subject_key(subject: dict, index: int) -> str:
    return str(
        subject.get("character_id")
        or subject.get("speaker_name")
        or f"subject_{index}"
    ).strip().casefold()


def _prepare_h3_prompt_only_continuity(shot_dicts: list[dict]) -> list[dict]:
    """Normalize H3 shot state and choreograph adjacent blocking handoffs."""

    prior_group = ""
    wardrobe_by_scene_subject: dict[tuple[str, str], str] = {}
    opening_blocking: list[str] = []
    final_instructions: dict[int, str] = {}
    for shot_index, raw in enumerate(shot_dicts):
        group = re.sub(
            r"[^a-z0-9_-]+",
            "_",
            str(raw.get("continuity_group") or f"scene_{shot_index + 1}")
            .strip()
            .casefold(),
        ).strip("_") or f"scene_{shot_index + 1}"
        raw["continuity_group"] = group
        requested = str(raw.get("continuity_strategy") or "").strip().lower()
        if requested not in VALID_CONTINUITY_STRATEGIES:
            requested = "continuous" if group == prior_group else "independent"
        if shot_index == 0 or group != prior_group:
            requested = "independent"
        elif requested != "extend_previous":
            # A shared group already means uninterrupted place and story time.
            # Treat an inconsistent "independent" label as an ordinary edit so
            # the preceding shot still earns this shot's opening blocking.
            requested = "continuous"
        raw["continuity_strategy"] = requested

        subjects = raw.get("subjects_on_screen") or []
        opening_subjects: list[str] = []
        for subject_index, subject in enumerate(subjects):
            if not isinstance(subject, dict):
                continue
            key = _h3_subject_key(subject, subject_index)
            wardrobe = re.sub(
                r"\s+", " ", str(subject.get("wardrobe") or "")
            ).strip()
            canonical_key = (group, key)
            if not wardrobe:
                wardrobe = wardrobe_by_scene_subject.get(canonical_key, "")
            if wardrobe:
                subject["wardrobe"] = wardrobe
                wardrobe_by_scene_subject.setdefault(canonical_key, wardrobe)
            name = str(
                subject.get("speaker_name")
                or subject.get("character_id")
                or subject.get("visual_description")
                or f"subject {subject_index + 1}"
            ).strip()
            position = re.sub(
                r"\s+",
                " ",
                str(subject.get("position_or_relation") or "unspecified position"),
            ).strip()
            description = re.sub(
                r"\s+", " ", str(subject.get("visual_description") or "")
            ).strip()
            subject_bits = [name]
            if description:
                subject_bits.append(description)
            if wardrobe:
                subject_bits.append(f"wearing {wardrobe}")
            subject_bits.append(f"positioned {position}")
            opening_subjects.append(", ".join(subject_bits))

        spatial_setup = re.sub(
            r"\s+", " ", str(raw.get("spatial_setup") or "")
        ).strip()
        opening_detail = "; ".join(opening_subjects)
        if spatial_setup:
            opening_detail = (
                f"{spatial_setup}. {opening_detail}"
                if opening_detail else spatial_setup
            )
        raw["video_prompt"] = _insert_h3_visual_detail(
            raw.get("video_prompt", ""),
            "OPENING CONTINUITY",
            opening_detail,
        )
        opening_blocking.append(opening_detail)
        prior_group = group

    # If the next shot remains in the same uninterrupted place/time, make any
    # blocking change occur visibly before the edit. This turns a next-shot
    # state such as "Joey is seated" into an action in the preceding clip.
    for index in range(1, len(shot_dicts)):
        previous = shot_dicts[index - 1]
        current = shot_dicts[index]
        if (
            current.get("continuity_strategy")
            not in {"continuous", "extend_previous"}
            or current.get("continuity_group")
            != previous.get("continuity_group")
        ):
            continue
        next_opening = opening_blocking[index]
        if not next_opening:
            continue
        previous["closing_blocking"] = next_opening
        transition = (
            "During the final beat, the visible characters naturally move "
            f"into this exact blocking before the shot ends: {next_opening}"
        )
        action_beats = list(previous.get("action_beats") or [])
        if transition not in action_beats:
            action_beats.append(transition)
        previous["action_beats"] = action_beats
        previous["ending_beat"] = transition
        final_instructions[index - 1] = transition

    for index, raw in enumerate(shot_dicts):
        closing = re.sub(
            r"\s+",
            " ",
            str(raw.get("closing_blocking") or raw.get("ending_beat") or ""),
        ).strip()
        raw["closing_blocking"] = closing
        raw["video_prompt"] = _insert_h3_visual_detail(
            raw.get("video_prompt", ""),
            "FINAL BLOCKING",
            final_instructions.get(index, closing),
        )
    return shot_dicts


def _model_specific_pass2_notes(video_model: str) -> str:
    """Per-checkpoint prompting notes for the active model, or "" if none.

    CivitAI / HF checkpoint imports carry a generated prompting DELTA (trigger
    words + preferred style, see Phase 2) stored inline on the model_def as
    `enhance_guide_text`. Surfacing it in Director's Pass-2 system prompt lets a
    custom (often NSFW) checkpoint prompt as well in Director as it does in
    Studio — closing the gap where Director ignored per-checkpoint guides.

    Only the inline delta is used. Built-in fine-tunes that ship a file-based
    `enhance_guide` (Sulphur, 10Eros) are intentionally NOT pulled in here: those
    are full Studio-format "rewrite the prompt" guides that would conflict with
    Director's shot-breakdown instructions and JSON output contract.
    """
    # Do not import wgp from this leaf helper. Its module owns the application
    # CLI parser, so importing it from an isolated planner/test process can
    # consume that process's argv. In Maestro proper, wgp is already loaded.
    import sys
    wgp_module = sys.modules.get("wgp")
    get_model_def = getattr(wgp_module, "get_model_def", None)
    if not callable(get_model_def):
        return ""
    try:
        md = get_model_def(video_model)
    except Exception:
        return ""
    notes = (md or {}).get("enhance_guide_text")
    if not (isinstance(notes, str) and notes.strip()):
        return ""
    return (
        "MODEL-SPECIFIC PROMPTING NOTES — the active checkpoint is a community "
        "fine-tune with its own conventions. Apply these to every shot prompt; "
        "they augment trigger words and style, they do not override the shot "
        "structure or output format:\n" + notes.strip()
    )


def _route_video_pass2_guide(video_model: str) -> str:
    """Pick the Pass 2 video guide for `video_model`, plus any per-checkpoint notes."""
    if not video_model:
        return _load_guide_helper("ltx2_shot_breakdown.md") or ""
    model_lower = video_model.lower()
    best_match: str | None = None
    best_len = 0
    for prefix, guide_file in _VIDEO_PASS2_GUIDE_MAP.items():
        if model_lower.startswith(prefix) and len(prefix) > best_len:
            best_match = guide_file
            best_len = len(prefix)
    chosen = best_match or "ltx2_shot_breakdown.md"
    if not best_match:
        print(f"[ShortFilmPlanner] No Pass-2 video guide for model={video_model!r}; falling back to {chosen}")
    guide = _load_guide_helper(chosen) or ""

    # Layer the active checkpoint's per-model prompting delta (Phase 2) on top.
    delta = _model_specific_pass2_notes(video_model)
    if delta:
        guide = f"{guide}\n\n{delta}" if guide else delta
    return guide


def _video_character_name_rules(preserve_names: bool) -> str:
    """Return model-aware naming rules for generated video prompts."""

    if preserve_names:
        return """H3 CHARACTER NAMING — preserve trained and mapped identities:
- In video_prompt and window_prompts, preserve every proper character/person name and its series, film, or franchise exactly as supplied.
- Repeat a recognizable trained identity such as "Dwight from The Office" verbatim in each shot where that identity appears; do not replace it with "the man" or "the character".
- For a user-reference identity, keep its supplied name/label together with useful visible traits so Ref2VA can map the prompt role to the labeled reference.
- Names inside quoted dialogue also remain verbatim.
- Image-model naming rules apply only to image_prompt; never use them to remove names from the H3 video prompt."""
    return """NAME CONVERSION — the screenplay may use character names, but prompts MUST NOT:
- Replace every character name with their descriptor + "from the reference image".
  PRESERVE the age/role descriptor from the screenplay — do NOT normalize to "man"/"woman".
  "teen boy Tommy" → "the teen boy from the reference image"
  "elderly Mrs. Chen" → "the elderly woman from the reference image"
  "Dr. Ava" → "the female doctor from the reference image"
  "little girl Sarah" → "the young girl from the reference image"
- Names are ONLY allowed inside quoted dialogue in video_prompt.
- NOT "Ava looks annoyed" → YES "the woman from the reference image looks annoyed"."""


class ShortFilmPlanner(BasePlanner):
    skill_type = "short_film"

    def plan(
        self,
        story_description: str = "",
        clips: Optional[list[dict]] = None,
        audio_path: Optional[str] = None,
        reference_image_path: Optional[str] = None,
        characters: Optional[list[dict]] = None,
        lyrics: Optional[list[dict]] = None,
        speaker_mappings: Optional[dict] = None,
        target_duration: int = 60,
        target_scenes: Optional[int] = None,
        narrative_mode: bool = True,
        fps: int = 24,
        frames_steps: int = 8,
        frames_minimum: int = 41,
        frames_maximum: Optional[int] = None,
        **kwargs,
    ) -> ProductionPlan:
        """Create a ProductionPlan for a short film.

        If `clips` are provided → audio-driven mode (scenes follow audio structure).
        If no clips → story-driven mode (LLM plans scene structure from scratch).
        """
        has_reference = bool(reference_image_path)
        is_audio_mode = bool(clips)
        # Store extra ref info for use in private methods
        self._num_character_refs = len(kwargs.get("character_ref_paths", []) or [])
        self._num_location_refs = len(kwargs.get("location_ref_paths", []) or [])
        self._character_ref_labels = kwargs.get("character_ref_labels")
        self._location_ref_labels = kwargs.get("location_ref_labels")
        self._character_ref_paths_raw = kwargs.get("character_ref_paths", [])
        self._location_ref_paths_raw = kwargs.get("location_ref_paths", [])
        self._seamless = kwargs.get("seamless", True)
        # Capture model identifiers for Pass-2 dialect-aware guide routing.
        # These flow from director_pipeline.py's planner_kwargs and let
        # _run_story_mode + _plan_audio_driven pick the correct video and
        # image guide files (ltx2_shot_breakdown.md for LTX-2,
        # flux_image_edit_pass2.md for Flux.2 Klein, etc.).
        self._video_model = kwargs.get("video_model", "") or ""
        self._image_model = kwargs.get("image_model", "") or ""
        shot_image_policy = str(kwargs.get("shot_image_policy") or "")
        self._shot_image_policy = shot_image_policy
        self._uses_generated_shot_images = shot_image_policy not in {
            "prompt_only",
            "direct_references",
        }
        self._preserve_video_character_names = (
            self._video_model.lower().startswith("minimax_h3")
            and shot_image_policy in {"prompt_only", "direct_references"}
        )

        # Normalize speaker_mappings: frontend sends list, we need dict
        if isinstance(speaker_mappings, list):
            sm_dict: dict = {}
            for entry in speaker_mappings:
                if isinstance(entry, dict):
                    sid = entry.get("speakerId") or entry.get("speaker_id", "")
                    if sid:
                        sm_dict[sid] = {"name": entry.get("name", ""), "role": entry.get("role", "")}
            speaker_mappings = sm_dict

        # Build character profiles
        char_profiles = self._build_characters(characters)

        # Build reference assets
        ref_assets = ReferenceAssets(
            start_image=AssetRef(id="ref_image", type="image", uri=reference_image_path) if has_reference else None,
            audio=AssetRef(id="audio", type="audio", uri=audio_path) if audio_path else None,
            transcript="\n".join(l.get("text", "") for l in (lyrics or []) if l.get("text", "").strip()),
        )

        nsfw = kwargs.get("nsfw", False)
        polish_block = kwargs.get("polish_block", "")
        # Multi-shot LoRA mode — when on, Pass 2 emits storyboard-format
        # video_prompts for medium-length shots. See the toggle's
        # comment in launch.py for behavior details. Threaded through
        # to _plan_story_driven below.
        multishot_lora_mode = kwargs.get("multishot_lora_mode", False)
        if (
            multishot_lora_mode
            and self._video_model.lower().startswith("minimax_h3")
        ):
            # Maestro's multi-shot toggle targets an LTX IC-LoRA and its
            # ``Shot N (Camera, Xs)`` trigger syntax. H3 has its own native
            # timeline language and must not inherit that LoRA-only format.
            multishot_lora_mode = False
            print(
                "[ShortFilmPlanner] Ignoring LTX Multi-Shot LoRA mode for "
                "MiniMax H3."
            )

        if is_audio_mode:
            shots = self._plan_audio_driven(
                clips=clips,
                story_description=story_description,
                lyrics=lyrics,
                speaker_mappings=speaker_mappings,
                reference_image_path=reference_image_path,
                char_profiles=char_profiles,
                has_reference=has_reference,
                nsfw=nsfw,
                polish_block=polish_block,
            )
        else:
            shots, title = self._plan_story_driven(
                story_description=story_description,
                reference_image_path=reference_image_path,
                char_profiles=char_profiles,
                has_reference=has_reference,
                target_duration=target_duration,
                target_scenes=target_scenes,
                narrative_mode=narrative_mode,
                fps=fps,
                frames_steps=frames_steps,
                frames_minimum=frames_minimum,
                frames_maximum=frames_maximum,
                nsfw=nsfw,
                polish_block=polish_block,
                multishot_lora_mode=multishot_lora_mode,
            )

        total_duration = sum(s.duration_sec for s in shots) if shots else target_duration

        return ProductionPlan(
            skill_type="short_film",
            title=getattr(self, '_last_title', None),
            global_style=story_description,
            total_duration_sec=total_duration,
            reference_assets=ref_assets,
            characters=char_profiles if char_profiles else None,
            shots=shots,
            continuity_notes=[
                "Short film — maintain visual and narrative continuity across shots",
                "Match camera complexity to emotional content",
                "Dialogue must appear in video prompts with speaker cues",
            ],
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _build_all_image_paths(self, reference_image_path: Optional[str], has_reference: bool) -> Optional[list[str]]:
        """Build image_paths list with ALL reference images (main + character + location)."""
        paths = []
        if has_reference and reference_image_path:
            paths.append(reference_image_path)
        for cp in (getattr(self, '_character_ref_paths_raw', None) or []):
            if cp and os.path.isfile(cp):
                paths.append(cp)
        for lp in (getattr(self, '_location_ref_paths_raw', None) or []):
            if lp and os.path.isfile(lp):
                paths.append(lp)
        return paths if paths else None

    # ── Character Building ───────────────────────────────────────────

    def _build_characters(self, characters: Optional[list[dict]]) -> list[CharacterProfile]:
        if not characters:
            return []
        return [
            CharacterProfile(
                id=f"char_{i}",
                display_name=c.get("name", ""),
                physical_description=c.get("description", "person"),
            )
            for i, c in enumerate(characters)
        ]

    # ── Audio-Driven Planning ────────────────────────────────────────

    # H3 Character Voice and Table-Read Passes

    def _build_h3_character_voice_bible(
        self,
        *,
        story_description: str,
        char_profiles: list[CharacterProfile],
    ) -> list[dict[str, str]]:
        """Create a compact, reusable characterization guide before Pass 1."""

        if not (self._generate or self._generate_streaming):
            return []
        supplied_characters = "\n".join(
            f"- {profile.display_name or profile.id}: "
            f"{profile.physical_description}"
            for profile in char_profiles or []
        ) or (
            "- No separate character cards were supplied. Use only people "
            "named in the concept."
        )
        system_prompt = """You are a character and dialogue editor preparing a compact voice bible before a screenplay is written.

Return ONLY a JSON array. Include one object for each person who may speak in the supplied concept, and no one else.

For an established fictional character named by the user, use the character's established personality, vocabulary, sentence rhythm, comic or dramatic behavior, and relationships to the other supplied characters. Capture why the character is recognizable beyond a single stereotype. Write entirely original guidance: do not quote, reproduce, or request signature dialogue or catchphrases.

For an original character, infer only what the concept and supplied character card support. Do not invent a biography that changes the story. Do not list a TV series, franchise, location, or group as a character.

Fields:
- character_name: the exact supplied character name.
- personality_engine: the motives, habits, contradictions, and behavioral logic that shape this character.
- speech_pattern: vocabulary, sentence length, rhythm, interruptions, formality, and recurring conversational behavior.
- relationship_behavior: how this character specifically talks and reacts to the other supplied cast.
- performance_direction: concise audible cadence, energy, register, and emotional delivery guidance; describe qualities, not an actor voice clone.
- avoid: generic caricatures, vocabulary this person would not use, and other out-of-character failure modes.

Keep each field concise and practical for a small local screenwriting model."""
        user_prompt = f"""Build the character voice bible for this project.

PROJECT CONCEPT:
{story_description}

SUPPLIED CHARACTER CARDS:
{supplied_characters}"""
        try:
            rows = self._call_llm_json(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=3072,
                thinking_budget=2048,
                temperature=0.45,
                streaming=True,
                frequency_penalty=0.1,
                presence_penalty=0.05,
                json_schema=_H3_VOICE_BIBLE_SCHEMA,
            )
            supported_text = "\n".join([
                story_description,
                *(
                    f"{profile.id} {profile.display_name} "
                    f"{profile.physical_description}"
                    for profile in char_profiles or []
                ),
            ])
            bible = _normalize_h3_voice_bible(
                rows,
                supported_character_text=supported_text,
            )
            if bible:
                print(
                    "[ShortFilmPlanner] Built H3 character voice bible for "
                    f"{len(bible)} supplied character(s)."
                )
            else:
                print(
                    "[ShortFilmPlanner] H3 voice-bible response contained no "
                    "validated supplied characters; using the screenplay rules."
                )
            return bible
        except Exception as exc:
            print(
                "[ShortFilmPlanner] H3 voice-bible pass was unavailable; "
                f"continuing with the screenplay rules ({exc})."
            )
            return []

    def _run_h3_character_table_read(
        self,
        *,
        story_description: str,
        screenplay: str,
        manifest: list[dict[str, Any]],
        voice_bible: list[dict[str, str]],
        max_spoken_words: int,
        maximum_line_words: int,
    ) -> list[dict[str, Any]]:
        """Polish only spoken words before the H3 manifest becomes immutable."""

        if not manifest or not (self._generate or self._generate_streaming):
            return manifest
        locked = _h3_user_locked_dialogue_fingerprints(story_description)
        payload = [
            {
                "turn": index,
                "speaker_name": entry.get("speaker_name") or "speaker",
                "original_text": entry.get("spoken_text") or "",
                "user_locked": (
                    _h3_dialogue_word_fingerprint(entry.get("spoken_text"))
                    in locked
                ),
            }
            for index, entry in enumerate(manifest, start=1)
        ]
        bible_text = _format_h3_voice_bible(voice_bible) or (
            "No structured voice bible was available. Infer distinct speech "
            "only from the project concept and screenplay context."
        )
        system_prompt = f"""You are the H3 CHARACTER TABLE-READ editor. Improve only the dialogue of an already structured screenplay.

Return ONLY one JSON array row for every supplied dialogue turn. Keep the same turn number, speaker, order, intent, plot facts, and conversational response relationship. Do not add or remove turns. original_text must be copied exactly into the corresponding output row.

Make each revised_text sound unmistakably appropriate to that character: established personality, vocabulary, syntax, cadence, comic or dramatic mechanism, and relationship to the person being addressed. Preserve nuance; do not reduce a character to one exaggerated trait. Write fresh dialogue and never copy famous lines or catchphrases.

If user_locked is true, revised_text MUST exactly equal original_text. Otherwise tighten stiff, formal, generic, or AI-like phrasing while preserving meaning. Keep the whole exchange within the stated spoken-word budget and never make an individual turn longer than {maximum_line_words} words.

delivery is a concise performance direction for that specific line. Describe cadence, energy, pitch/register, hesitation, interruption, or emotional pressure. Do not request an exact actor voice or voice impersonation."""
        user_prompt = f"""Perform a dialogue-only table read for this H3 Director screenplay.

PROJECT CONCEPT:
{story_description}

CHARACTER VOICE BIBLE (binding):
{bible_text}

MAXIMUM SPOKEN WORDS ACROSS ALL TURNS: {max_spoken_words}

DIALOGUE TURN MANIFEST:
{json.dumps(payload, ensure_ascii=False, indent=2)}

FULL SCREENPLAY FOR ACTION AND RELATIONSHIP CONTEXT:
{screenplay}"""
        try:
            rows = self._call_llm_json(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=max(2048, min(8192, len(manifest) * 190)),
                thinking_budget=4096,
                temperature=0.65,
                streaming=True,
                frequency_penalty=0.12,
                presence_penalty=0.04,
                json_schema=_h3_table_read_schema(len(manifest)),
            )
            revised, changed = _apply_h3_character_table_read(
                manifest,
                rows,
                story_description=story_description,
                max_spoken_words=max_spoken_words,
                maximum_line_words=maximum_line_words,
            )
            print(
                "[ShortFilmPlanner] H3 character table read validated "
                f"{len(revised)} turn(s) and revised {changed}; dialogue is now locked."
            )
            return revised
        except Exception as exc:
            print(
                "[ShortFilmPlanner] H3 character table read failed validation; "
                f"locking the original screenplay dialogue instead ({exc})."
            )
            return manifest

    # Audio-Driven Planning

    def _plan_audio_driven(
        self,
        clips: list[dict],
        story_description: str,
        lyrics: Optional[list[dict]],
        speaker_mappings: Optional[dict],
        reference_image_path: Optional[str],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        nsfw: bool = False,
        polish_block: str = "",
    ) -> list[ShotPlan]:
        """Plan shots from existing audio-segmented clips."""
        from ..nsfw_guidance import inject_nsfw_if_enabled

        speaker_names = {}
        if speaker_mappings:
            for sid, info in speaker_mappings.items():
                speaker_names[sid] = info.get("name", sid)

        # Build clip contexts
        clip_contexts = []
        for i, clip in enumerate(clips):
            start_sec = clip.get("start", 0)
            end_sec = clip.get("end", start_sec + 5)
            duration = end_sec - start_sec
            label = clip.get("label", "scene")

            # Gather dialogue
            dialogue_lines = []
            speakers_in_clip = set()
            if lyrics:
                for l in lyrics:
                    if l.get("start", 0) < end_sec and l.get("end", 0) > start_sec:
                        spk = l.get("speaker", "")
                        text = l.get("text", "")
                        if text.strip():
                            spk_name = speaker_names.get(spk, spk) if spk else ""
                            dialogue_lines.append(f'{spk_name}: "{text}"' if spk_name else f'"{text}"')
                            if spk:
                                speakers_in_clip.add(spk)

            # Characters on screen
            char_info = ""
            if speakers_in_clip and char_profiles:
                on_screen = [speaker_names.get(s, s) for s in speakers_in_clip]
                char_info = f" On screen: {', '.join(on_screen)}."

            dialogue_text = ""
            if dialogue_lines:
                dialogue_text = f" Dialogue: {' / '.join(dialogue_lines[:4])}"

            ctx = f"Shot {i + 1}: {label}, {duration:.1f}s.{char_info}{dialogue_text}"
            clip_contexts.append(ctx)

        # Build full transcript for context
        full_transcript = ""
        if lyrics:
            lines = []
            for l in lyrics:
                spk = l.get("speaker", "")
                text = l.get("text", "")
                if text.strip():
                    spk_name = speaker_names.get(spk, spk) if spk else ""
                    t_start = l.get("start", 0)
                    t_end = l.get("end", 0)
                    prefix = f"[{t_start:.1f}-{t_end:.1f}s] {spk_name}: " if spk_name else f"[{t_start:.1f}-{t_end:.1f}s] "
                    lines.append(f"{prefix}{text}")
            full_transcript = "\n".join(lines)

        # Call LLM
        preserve_names = bool(
            getattr(self, "_preserve_video_character_names", False)
        )
        uses_generated_images = bool(
            getattr(self, "_uses_generated_shot_images", True)
        )
        char_rules = build_character_rules_block(
            has_reference or bool(getattr(self, "_num_character_refs", 0)),
            char_profiles if char_profiles else None,
            preserve_names=preserve_names,
        )
        camera_block = build_camera_style_block()
        # Audio-driven mode also uses dialect-aware Pass 2 guides — see
        # _route_video_pass2_guide / get_image_prompt_rules for routing.
        video_model = getattr(self, '_video_model', '') or ''
        image_model = getattr(self, '_image_model', '') or ''
        video_guide = _route_video_pass2_guide(video_model)
        video_name_rules = _video_character_name_rules(
            preserve_names,
        )
        visual_strategy_rules = (
            "H3 DIRECT VIDEO GUIDANCE:\n"
            "- No generated start frame will be supplied. Make every video_prompt "
            "self-contained: name the setting, composition, visible identities and "
            "traits, wardrobe, action, camera, dialogue, ambience, and sound.\n"
            "- Character/location references are soft identity and scene guidance, "
            "not fixed opening frames. Describe the finished shot rather than an "
            "instruction to copy or replace a reference.\n"
            "- Do not create image_prompt, image_source, visual_changes, or "
            "keyframe_prompts. Those fields are intentionally absent from the "
            "video-only output schema."
            if not uses_generated_images else ""
        )

        image_prompt_rules = ""
        if uses_generated_images:
            from ..image_prompt_rules import get_image_prompt_rules
            image_prompt_rules = get_image_prompt_rules(
                has_reference,
                num_character_refs=getattr(self, '_num_character_refs', 0),
                num_location_refs=getattr(self, '_num_location_refs', 0),
                character_ref_labels=getattr(self, '_character_ref_labels', None),
                location_ref_labels=getattr(self, '_location_ref_labels', None),
                seamless=getattr(self, '_seamless', True),
                image_model=image_model,
            )

        image_planning_rules = (
            """- image_prompt is the VERY FIRST FRAME — BEFORE any action in the video_prompt begins.
  It must be a FROZEN STILL PHOTOGRAPH — no motion, no action, no verbs of movement.
  Show the INITIAL STATE: if the scene involves removing clothing, the clothing is still ON.
  If a character enters the room, the room is EMPTY (or show whoever is already there).
  If something will be revealed, it is still hidden. The video_prompt describes the change.
  Include \"create new scene, [environment].\" at the start."""
            if uses_generated_images else ""
        )
        image_output_fields = (
            '''    "image_source": "original or previous — original=edit from user's reference photo, previous=edit from last scene's output (use for same-location continuity)",
    "image_prompt": "FIRST FRAME BEFORE action — initial state, static pose, environment. No motion verbs.",
    "visual_changes": ["what visually transforms during this scene — e.g. 'shirt is removed', 'man enters from doorway'"],
'''
            if uses_generated_images else ""
        )
        image_output_notes = (
            """- image_source: "original" = edit from user's reference photo (default). "previous" = edit from last scene's
  output (for same-location continuity). First scene must always be "original".
- FIELD ORDER: Write image_prompt FIRST (starting state), then visual_changes, then video_prompt.
  image_prompt shows the BEFORE state. visual_changes lists what transforms. video_prompt describes the action.
- visual_changes: If it says "shirt removed", image_prompt must show shirt still ON.
- keyframe_prompts: Only when the video model needs visual info it can't generate from the start image.
"""
            if uses_generated_images else ""
        )

        system_prompt = f"""You are a cinematic scene planner for a short film with dialogue audio. Output ONLY the JSON array.

{f"You are given a REFERENCE PHOTO of the characters. Use their visible appearance in all prompts." if has_reference else ""}

{visual_strategy_rules}

You are planning visuals for a scene where the AUDIO ALREADY EXISTS. The dialogue is pre-recorded.
Your job is to create compelling VISUALS that match the dialogue — environments, staging, camera work,
character actions, and facial expressions that bring the audio to life.

FULL DIALOGUE TRANSCRIPT:
{full_transcript if full_transcript else "(no transcript available)"}

STORY CONCEPT: {story_description}

Plan each shot as a structured scene — deciding visuals, camera, action, mood,
and how dialogue is staged. Write a DETAILED {"video_prompt and image_prompt" if uses_generated_images else "video_prompt"} for each shot.

{char_rules}

{camera_block}

SHORT FILM PLANNING RULES:
- The audio is PRE-RECORDED — you are planning VISUALS to match existing dialogue.
- Focus on acting, body language, and emotional expression that matches what's being said.
- Stage dialogue naturally — characters should have physical business while speaking.
- Match camera complexity to emotional tone: steady for intimate, dynamic for action.
- Each shot should advance the story or reveal character.
- Describe the ENVIRONMENT in detail for each shot (room, furniture, lighting, time of day).
- video_prompt MUST be a full detailed paragraph (80-150 words) — NOT a brief label.
{image_planning_rules}

VIDEO PROMPT (video_prompt) — follow the LTX-2 style guide below closely:
- One single flowing paragraph, present tense, 4-8 sentences.
- Start with shot type and visual style early.
- Characters: {"preserve supplied proper names and add useful visible traits (clothing, hair, posture, expression)" if preserve_names else "describe by visible traits (clothing, hair, posture, expression)"}.
- Emotion through PHYSICAL CUES only (jaw tightens, fists clench, shoulders drop) — never abstract labels like "serious expression" or "looks determined".
- Action: chronological order — setup, movement, reaction, final beat.
- Camera: explicit movement tied to the subject (slow dolly in, tracking left, orbit around, handheld follow) — never vague ("digital drift", "cinematic camera").
- Audio: include ambient sound when relevant, and any other sounds or sound effects that are relevant to the scene.
- Dialogue: in quotes with delivery cue if present.
- NEVER say montage, quick cuts, cut to.
{video_name_rules}

{image_prompt_rules}

REFERENCE — LTX-2 video style guide:
{video_guide if video_guide else "(no guide loaded)"}

OUTPUT FORMAT — respond with ONLY a JSON array:
[
  {{
    "scene_goal": "What this shot achieves in the story",
    "scene_type": "dialogue|action|opening|closing|reaction",
    "subjects_on_screen": [
      {{"visual_description": "the woman in the white coat", "position_or_relation": "foreground left"}}
    ],
    "spatial_setup": "How subjects are arranged",
    "environment": "Setting description",
    "visual_style": "Visual look",
    "lighting": "Lighting description",
    "mood": "Emotional tone",
    "action_beats": ["Physical actions in chronological order"],
    "dialogue_beats": [
      {{"speaker_id": "char_0", "spoken_text": "Actual dialogue", "delivery": "softly", "physical_cue": "leans forward"}}
    ],
    "camera_plan": {{
      "framing": "medium shot",
      "movement": "slow push in",
      "movement_intensity": "subtle"
    }},
    "audio_plan": {{
      "mode": "dialogue_driven",
      "lip_sync_critical": true
    }},
    "ending_beat": "Final visual moment",
{image_output_fields}    "video_prompt": "Full flowing paragraph for video generation — describes the action...",
    "window_prompts": ["(OPTIONAL) Window 1 — first ~20s of action...", "Window 2 — next ~20s, continues from where window 1 ends..."]
  }}
]

{image_output_notes}

WINDOW PROMPTS vs VIDEO PROMPT — use ONE or the OTHER, never both:
- Scenes 20s or under: write video_prompt, leave window_prompts as [].
- Scenes over 20s: write window_prompts, leave video_prompt as "".
  Each window covers ~20s. Windows play SEQUENTIALLY — window 2 continues exactly
  where window 1 left off, picking up the action mid-flow.
  CRITICAL: The video model only sees the last few frames — it has NO memory of
  earlier action or sound. Each window must briefly re-establish ongoing state
  (e.g. "the audience continues cheering" or "rain still falling") before
  describing new action. Without this, ongoing activity abruptly stops.
  Example: Window 1 delivers the joke → Window 2: "The audience continues laughing
  and clapping. She takes a bow, wipes her brow, and walks to stage left..."
Output exactly {len(clips)} shot plans. Go:"""

        # Inject model-specific prompt polish guide if provided
        if polish_block:
            system_prompt = f"{system_prompt}\n\n{polish_block}"

        # Mature-mode guidance is now SELF-GATING: the version-controlled
        # clinical guides apply only when the scene is actually sexual and tell
        # the model to write normally otherwise, so the block can be injected
        # whenever mature mode is on without harming clean scenes. This replaced
        # the old keyword pre-scan, which depended on an explicit wordlist that
        # cannot live in the version-controlled repo and missed scenes phrased
        # without its keywords.
        effective_nsfw = nsfw
        system_prompt = inject_nsfw_if_enabled(
            system_prompt,
            effective_nsfw,
            "both" if uses_generated_images else "video",
        )
        # Note: audio mode doesn't load keyframe_rules.md as a separate
        # block (the keyframe guidance is inlined in the output spec
        # below).

        # `/no_think` prefix suppresses Qwen3 internal reasoning for this turn
        # — see story-mode pass 2 for full rationale. Pass 2 is structured-JSON
        # planning where thinking adds no creative value and on Qwen3.6-27B
        # has been observed to spiral. The marker is enforced by Qwen's Jinja
        # template directly, bypassing the broken `enable_thinking` kwarg path.
        user_prompt = f"""/no_think

TASK: Plan visuals for each of these {len(clips)} dialogue segments. Output exactly {len(clips)} shot plans — no more, no less.

CRITICAL OUTPUT REQUIREMENTS:
- Output EXACTLY {len(clips)} shots, one per audio clip below
- The audio is already recorded — write {"video_prompt and image_prompt" if uses_generated_images else "a self-contained video_prompt"} that brings each segment to life visually
{("- Use keyframes ONLY when the video model needs visual info not in the start image; the model handles dialogue, gestures, and expressions on its own" if uses_generated_images else "- Do not output any still-image or keyframe fields")}

Shots to plan:
{chr(10).join(clip_contexts)}"""

        image_paths = self._build_all_image_paths(reference_image_path, has_reference)
        # Video-only H3 plans omit four still-image fields, so reserve a smaller
        # per-shot budget instead of inviting unused elaboration.
        # `/no_think` above suppresses Qwen thinking. `thinking_budget=None`
        # delegates to _call_llm_json's model-aware default: Qwen→0 (off),
        # Gemma→4096 (on, to help small Gemma models follow structured-output
        # rules like the strict 20s window threshold).
        per_shot_tokens = 1600 if uses_generated_images else 1200
        max_tokens = max(8192, len(clips) * per_shot_tokens + 4096)

        # Grammar constraint (applies on thinking-off models' first attempt
        # + everyone's retry — see _call_llm_json). minItems == maxItems ==
        # len(clips) makes the "output EXACTLY {len(clips)} shots" rule
        # grammar-enforced, not just prompted: the model cannot close the
        # array early or run past the clip count. keyframe_prompts /
        # window_prompts stay optional (spec tags them OPTIONAL).
        audio_schema = _shot_list_schema(
            min_items=len(clips),
            max_items=len(clips),
            required=[
                "scene_goal", "scene_type", "subjects_on_screen",
                "spatial_setup", "environment", "visual_style", "lighting",
                "mood", "action_beats", "dialogue_beats", "camera_plan",
                "audio_plan", "ending_beat", "image_source", "image_prompt",
                "visual_changes", "video_prompt",
            ],
            include_image_fields=uses_generated_images,
        )

        shot_dicts = self._call_llm_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            thinking_budget=None,
            image_paths=image_paths,
            json_schema=audio_schema,
        )

        if not uses_generated_images:
            _discard_unused_image_fields(shot_dicts)
        return self._convert_audio_shots(shot_dicts, clips, char_profiles, has_reference)

    def _convert_audio_shots(
        self,
        shot_dicts: list[dict],
        clips: list[dict],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
    ) -> list[ShotPlan]:
        """Convert LLM output to ShotPlan objects for audio-driven mode."""
        shots = []
        for i, clip in enumerate(clips):
            raw = shot_dicts[i] if i < len(shot_dicts) else {}
            duration = clip.get("end", 0) - clip.get("start", 0)

            subjects = [SubjectRef.from_dict(s) if isinstance(s, dict) else SubjectRef(visual_description=str(s))
                        for s in raw.get("subjects_on_screen", [])]

            cam_raw = raw.get("camera_plan", {})
            camera = CameraPlan(
                framing=cam_raw.get("framing", "medium shot"),
                angle=cam_raw.get("angle"),
                movement=cam_raw.get("movement"),
                movement_intensity=cam_raw.get("movement_intensity", "subtle"),
            )

            audio_raw = raw.get("audio_plan", {})
            audio = AudioPlan(
                mode=audio_raw.get("mode", "dialogue_driven"),
                ambience=audio_raw.get("ambience"),
                timing_anchor="audio",
                lip_sync_critical=audio_raw.get("lip_sync_critical", True),
            )

            dialogue_beats = None
            if raw.get("dialogue_beats"):
                dialogue_beats = [DialogueBeat.from_dict(db) for db in raw["dialogue_beats"]]

            shot = ShotPlan(
                shot_id=self._make_shot_id(i, "sf"),
                index=i,
                duration_sec=duration,
                skill_type="short_film",
                scene_goal=raw.get("scene_goal", f"Shot {i + 1}"),
                scene_type=raw.get("scene_type", "dialogue"),
                source_mode_preference="a2v" if audio_raw.get("lip_sync_critical") else ("i2v" if has_reference else "t2v"),
                image_strategy="reference_edit" if has_reference else "fresh_generation",
                continuity_strategy="continuous" if i > 0 else "independent",
                subjects_on_screen=subjects,
                spatial_setup=raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                visual_style=raw.get("visual_style", ""),
                lighting=raw.get("lighting", ""),
                mood=raw.get("mood", ""),
                action_beats=raw.get("action_beats", []),
                dialogue_beats=dialogue_beats,
                camera_plan=camera,
                audio_plan=audio,
                ending_beat=raw.get("ending_beat", ""),
                metadata={
                    "clip_start": clip.get("start", 0),
                    "clip_end": clip.get("end", 0),
                },
                # LLM-generated prompts (used directly, skipping renderer pass 2)
                video_prompt=raw.get("video_prompt"),
                image_prompt=raw.get("image_prompt"),
                window_prompts=raw.get("window_prompts"),
                visual_changes=raw.get("visual_changes"),
                image_source=raw.get("image_source"),
                keyframe_prompts=raw.get("keyframe_prompts"),
            )
            shots.append(shot)

        return shots

    # ── Story-Driven Planning ────────────────────────────────────────

    def _plan_story_driven(
        self,
        story_description: str,
        reference_image_path: Optional[str],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        target_duration: int,
        target_scenes: Optional[int],
        narrative_mode: bool,
        fps: int,
        frames_steps: int,
        frames_minimum: int,
        frames_maximum: Optional[int] = None,
        nsfw: bool = False,
        polish_block: str = "",
        multishot_lora_mode: bool = False,
    ) -> tuple[list[ShotPlan], Optional[str]]:
        """Two-pass story-driven planning.

        Pass 1 — Screenplay: LLM writes the full story as a flowing script.
        Pass 2 — Shot breakdown: LLM converts the screenplay into minimum shots with prompts.

        Args:
            multishot_lora_mode: When True, Pass 2 emits storyboard-format
                video_prompts for medium-length shots (20-30s) suitable
                for IC-LoRA-trained multi-shot models (Maque AI LTX-2.3
                IC-LoRA and similar). Short reaction shots (≤15s) and
                long sustained shots (40s+) keep the regular flowing
                video_prompt format.
        """
        from ..nsfw_guidance import inject_nsfw_if_enabled
        from ..safety_scan import (
            assert_no_minor_content,
            collect_pass2_text,
        )

        if target_scenes is None:
            target_scenes = max(2, min(20, target_duration // 20))

        preserve_names = bool(
            getattr(self, "_preserve_video_character_names", False)
        )
        is_h3_native = str(
            getattr(self, "_video_model", "") or ""
        ).lower().startswith("minimax_h3")
        uses_generated_images = bool(
            getattr(self, "_uses_generated_shot_images", True)
        )

        image_paths = self._build_all_image_paths(reference_image_path, has_reference)

        # ── PRE-PASS-1 SAFETY SCAN: user concept ────────────────────────
        # Scan the user's input concept BEFORE running Pass 1. Catches
        # obviously-prohibited concepts ~30s earlier and avoids burning
        # an LLM call on something we'll abort anyway. Same scanner /
        # same hybrid co-occurrence policy as the post-Pass-1 check.
        assert_no_minor_content(story_description, source="user concept")

        # ── PASS 1: Screenplay ───────────────────────────────────────────
        story_guide = ""
        if narrative_mode:
            story_guide = self._load_guide("Expert short-form storyteller.md")

        narrative_block = ""
        if narrative_mode and story_guide:
            narrative_block = f"""\nNARRATIVE GUIDE:\n{story_guide}\n
Structure the story with: setup, rising conflict, climax, resolution."""

        char_block = ""
        if char_profiles:
            char_lines = []
            for c in char_profiles:
                identity = (
                    f"{c.id} / {c.display_name}"
                    if preserve_names and c.display_name
                    else c.id
                )
                char_lines.append(f"- {identity}: {c.physical_description}")
            char_block = (
                (
                    "\nCHARACTERS (preserve supplied proper names in both "
                    "camera-visible action and dialogue, paired with useful "
                    "appearance details):\n"
                    if preserve_names
                    else "\nCHARACTERS (use appearance descriptions in the "
                    "screenplay — names are allowed ONLY in dialogue):\n"
                )
                + "\n".join(char_lines)
                + "\n\nNOTE on character descriptions: the descriptions above are "
                "REFERENCE-PHOTO descriptions — they describe how each person LOOKS "
                "in the photo the user uploaded. They are an IDENTITY hint (face, "
                "build, gender) for the image generator. The actual STORY may "
                "transform these characters into other roles (a 'man in black' "
                "from the reference photo can become a knight in armor, a wizard, "
                "a CEO, a vampire — whatever the story needs). When you write the "
                "screenplay, describe characters as they appear IN THE STORY, not "
                "as they appear in the reference photo. The image generator will "
                "blend the reference face with the story's costume/role to render "
                "the transformed character correctly."
            )

        from ..guide_loader import load_guide
        screenplay_rules = load_guide("screenplay_writing_rules.md")
        if preserve_names:
            screenplay_rules += (
                "\n\nH3 NAMED-IDENTITY OVERRIDE — this supersedes the "
                "screenplay rule that normally removes names from action lines. "
                "Preserve every proper name, performer, character, series, film, "
                "or franchise supplied by the user exactly as written in both "
                "action and dialogue. Pair names with camera-visible traits when "
                "useful, and never invent an unsupplied name."
            )

        # ── Hard length budget (CRITICAL) ────────────────────────────
        # The screenplay LLM consistently overshoots target duration —
        # observed in production: a 180s target produced a 358s
        # screenplay (~5.5 minutes of content). Without a concrete word
        # budget, "let scenes breathe" and "substantial dialogue"
        # guidance from screenplay_writing_rules.md compounds with the
        # LLM's natural tendency to elaborate, and Pass 2 inherits a
        # too-dense screenplay that no amount of consolidation can
        # actually fit.
        #
        # Math: at ~2 spoken words/sec, target_duration sets the
        # dialogue ceiling. Action lines add ~50% on top (they're
        # silent but they consume screen time).
        max_spoken_words = target_duration * 2  # 2 wps
        max_total_words = int(target_duration * 4.5)  # action + dialogue
        # Suggest a reasonable scene count window. Cinematic average is
        # ~10-25s/scene; we anchor at the wider end to prevent shot
        # explosion at Pass 2.
        scene_count_low = max(2, target_duration // 30)
        scene_count_high = max(scene_count_low + 1, target_duration // 15)

        length_budget_block = f"""
HARD LENGTH BUDGET — NON-NEGOTIABLE FOR THIS SCREENPLAY:
- Target duration: {target_duration} seconds.
- Maximum SPOKEN dialogue across the entire screenplay: {max_spoken_words} words.
  (At ~2 words/second, dialogue alone fills the runtime if you write more.)
- Maximum TOTAL screenplay length (dialogue + action lines + scene headings):
  approximately {max_total_words} words.
- Aim for {scene_count_low}-{scene_count_high} distinct scenes total.
  Fewer, fuller scenes always beat many short ones.

WHEN YOU NOTICE THE SCREENPLAY GETTING LONG — CUT, DON'T SPLIT:
- If you have written more than {max_spoken_words} words of dialogue, you are
  OVER BUDGET. Do NOT split into more scenes. Do NOT add a Pass 2
  consolidation step — there isn't one. Instead:
    * DROP a beat entirely (does the story actually need this exchange?).
    * SHORTEN a beat (one back-and-forth instead of three).
    * CONDENSE multi-line speeches into a single direct line.
- A {target_duration}-second film is SHORT. Pick the {scene_count_low}-{scene_count_high} most
  essential beats and write THOSE well. Save the rest for a longer cut.

WHY THIS MATTERS:
- Downstream Pass 2 splits the screenplay into shots; the video model
  generates each shot. If your screenplay implies 300 seconds of action,
  Pass 2 has TWO bad choices: (a) inflate total runtime to {int(target_duration * 1.7)}+
  seconds (overshoots user's target), or (b) cram 300s of content into
  {target_duration}s of shots (rushed, characters speak too fast, motion blurs).
  Both produce a worse film than a {target_duration}s screenplay paced for {target_duration}s.
"""

        h3_voice_bible: list[dict[str, str]] = []
        h3_character_block = ""
        if is_h3_native:
            print("[ShortFilmPlanner] Pass 0: Building H3 character voice bible...")
            h3_voice_bible = self._build_h3_character_voice_bible(
                story_description=story_description,
                char_profiles=char_profiles,
            )
            voice_bible_text = _format_h3_voice_bible(h3_voice_bible)
            h3_character_block = """
H3 CHARACTER-AUTHENTICITY RULES:
- Before drafting, use the binding voice bible below as the cast's dialogue and relationship logic.
- For an established fictional character named by the user, write fresh dialogue consistent with the character's established personality, vocabulary, syntax, cadence, comic/dramatic mechanism, and relationships. Do not copy famous dialogue or catchphrases.
- Do not reduce a recognizable character to one generic trait. A line that could be reassigned to another cast member without sounding wrong must be rewritten.
- Preserve every literal line supplied by the user exactly. Character-authentic writing changes generated dialogue, never user-authored dialogue.
- Silently conduct a table read before returning the screenplay: remove generic sitcom filler, stiff exposition, invented gimmicks, and words the named speaker would not naturally choose.
"""
            if voice_bible_text:
                h3_character_block += (
                    "\nBINDING CHARACTER VOICE BIBLE:\n" + voice_bible_text
                )

        print("[ShortFilmPlanner] Pass 1: Writing screenplay...")
        pass1_system = f"""You are an acclaimed screenwriter celebrated for dialogue that sounds like real people actually talking — never stiff, formal, stagey, or "AI-like." You give every character a distinct, believable voice, and you fully commit to whatever tone, era, or style the concept calls for. Write a complete short film screenplay.

{f"You are given a REFERENCE PHOTO of the characters. Use their visible appearance in the script." if has_reference else ""}
{char_block}
{narrative_block}

{screenplay_rules}
{h3_character_block}
{length_budget_block}"""

        if polish_block:
            pass1_system = f"{pass1_system}\n\n{polish_block}"
        pass1_system = inject_nsfw_if_enabled(pass1_system, nsfw, "screenplay")

        pass1_user = f"Write a short film screenplay based on this concept:\n\n{story_description}"

        # Repetition penalties are critical at Pass 1's scale (~18k token
        # output budget for a 180s film). Without them, models — especially
        # Qwen3.5/3.6 — can lock into a repetition cascade and generate
        # the same paragraph endlessly until the token budget runs out.
        # Phase 0.1 added stronger penalties (0.3 / 0.1) to Pass 2's
        # JSON output via _call_llm_json. Pass 1 is creative writing where
        # too much penalty hurts natural dialogue flow, so we use softer
        # values here — just enough to break repetition cascades without
        # discouraging legitimate word reuse in dialogue ("yes", "no",
        # character names, etc.).
        # Output token cap aligned to the word budget. Without this cap,
        # max_new_tokens defaulted to target_duration * 100 (180s →
        # 18000 tokens) which gave the LLM no signal to stop. User
        # reported a 1317-word screenplay against an 810-word budget
        # for a 180s target. Capping at ~3 tokens/word (generous for
        # English screenplay formatting) lets the LLM go ~50% over
        # budget before hitting the wall — a soft enforcement that
        # leaves room for the prompt-level guidance to do its job
        # without truncating mid-screenplay when the LLM lands close
        # to budget.
        #
        # The thinking_budget is independent — chain-of-thought
        # reasoning gets its own pool and doesn't count against this
        # cap.
        _output_token_cap = max(2000, max_total_words * 3)
        screenplay = self._generate_streaming(
            prompt=pass1_user,
            system_prompt=pass1_system,
            max_new_tokens=_output_token_cap,
            temperature=0.8,
            thinking_budget=16384,
            image_paths=image_paths or [],
            frequency_penalty=0.15,
            presence_penalty=0.05,
        )

        print(f"[ShortFilmPlanner] Screenplay: {len(screenplay)} chars")

        # ── Post-Pass-1 length warning ───────────────────────────────
        # Cheap word count to compare against the budget set in the
        # length_budget_block above. If we're over, we don't fail or
        # truncate — Pass 2 has its own duration constraints — but we
        # log so the user can see when Pass 1 ignored its budget.
        # Persistent over-budget output across runs is the signal that
        # the screenplay-LLM model is too aggressive for the budget
        # wording (consider switching models or temperature).
        _word_count = len(screenplay.split())
        if _word_count > max_total_words * 1.15:
            print(
                f"[ShortFilmPlanner] ⚠ Pass 1 over budget: {_word_count} words "
                f"(budget was {max_total_words}, +{_word_count - max_total_words}). "
                f"Pass 2 will compress; expect possible runtime overshoot."
            )
        else:
            print(
                f"[ShortFilmPlanner] Pass 1 word count: {_word_count} "
                f"(budget {max_total_words})"
            )

        # ── POST-PASS-1 SAFETY SCAN ─────────────────────────────────────
        # Catches anything the prompt-level prohibition rule failed to
        # prevent. Raises SafetyViolationError; pipeline error handler
        # in director_pipeline.py converts to a clean user-visible
        # message in chat.
        assert_no_minor_content(screenplay, source="screenplay (Pass 1)")

        # H3 renders independent bounded shots rather than 20-second rolling
        # windows. Plan directly on its native duration lattice so legacy
        # Window 1/2 prose never reaches the later compatibility adapter.
        if is_h3_native:
            screenplay_dialogue_manifest = _extract_h3_screenplay_dialogue(
                screenplay
            )
            if screenplay_dialogue_manifest:
                print(
                    "[ShortFilmPlanner] Pass 1.5: Running H3 character "
                    "table read before dialogue lock..."
                )
                screenplay_dialogue_manifest = (
                    self._run_h3_character_table_read(
                        story_description=story_description,
                        screenplay=screenplay,
                        manifest=screenplay_dialogue_manifest,
                        voice_bible=h3_voice_bible,
                        max_spoken_words=max_spoken_words,
                        maximum_line_words=max(
                            1,
                            int(math.floor(
                                (float(frames_maximum or 345) / max(1, fps))
                                * _H3_DIALOGUE_WORDS_PER_SECOND
                            )),
                        ),
                    )
                )
                assert_no_minor_content(
                    "\n".join(
                        str(entry.get("spoken_text") or "")
                        for entry in screenplay_dialogue_manifest
                    ),
                    source="H3 character table read",
                )
            return self._plan_story_h3_native(
                story_description=story_description,
                screenplay=screenplay or story_description,
                screenplay_dialogue_manifest=screenplay_dialogue_manifest,
                character_voice_bible=h3_voice_bible,
                reference_image_path=reference_image_path,
                char_profiles=char_profiles,
                has_reference=has_reference,
                target_duration=target_duration,
                fps=fps,
                frames_steps=frames_steps,
                frames_minimum=frames_minimum,
                frames_maximum=frames_maximum,
                nsfw=nsfw,
                polish_block=polish_block,
            )

        if not screenplay or len(screenplay) < 50:
            print("[ShortFilmPlanner] Screenplay too short, falling back to single-pass")
            return self._plan_story_single_pass(
                story_description, reference_image_path, char_profiles,
                has_reference, target_duration, target_scenes, narrative_mode,
                fps, frames_steps, frames_minimum, nsfw, polish_block,
            )

        # ── PASS 2: Shot Breakdown ───────────────────────────────────────
        print("[ShortFilmPlanner] Pass 2: Breaking screenplay into shots...")

        char_rules = build_character_rules_block(
            has_reference or bool(getattr(self, "_num_character_refs", 0)),
            char_profiles if char_profiles else None,
            preserve_names=preserve_names,
        )
        # video_guide now merged into ltx2_shot_breakdown.md — no separate load needed

        # Pull video/image model identifiers from the planner kwargs.
        # These flow from director_pipeline.py's planner_kwargs and let
        # us route Pass-2 guides correctly (LTX-2 video gets LTX-2
        # shot breakdown, Flux.2 Klein image gets Flux Pass-2 rules,
        # etc.) rather than always loading the legacy hardcoded files.
        video_model = getattr(self, '_video_model', '') or ''
        image_model = getattr(self, '_image_model', '') or ''

        image_prompt_rules = ""
        if uses_generated_images:
            from ..image_prompt_rules import get_image_prompt_rules
            image_prompt_rules = get_image_prompt_rules(
                has_reference,
                num_character_refs=getattr(self, '_num_character_refs', 0),
                num_location_refs=getattr(self, '_num_location_refs', 0),
                character_ref_labels=getattr(self, '_character_ref_labels', None),
                location_ref_labels=getattr(self, '_location_ref_labels', None),
                seamless=getattr(self, '_seamless', True),
                image_model=image_model,
            )

        # Load all guide content from .md files. Video shot-breakdown
        # currently routes only to LTX-2 vs. a generic fallback —
        # other model families share the LTX-2 rules until per-model
        # Pass-2 video guides land in Phase 3.
        shot_structure = load_guide("shot_structure_rules.md")
        video_rules = _route_video_pass2_guide(video_model)
        video_name_rules = _video_character_name_rules(
            preserve_names,
        )
        visual_strategy_rules = (
            "H3 DIRECT VIDEO GUIDANCE:\n"
            "- No generated start frame will be supplied. video_prompt or "
            "window_prompts must be fully self-contained with setting, "
            "composition, identities and "
            "visible traits, wardrobe, action, camera, dialogue, ambience, and "
            "synchronized sound.\n"
            "- Character/location references are soft guidance rather than fixed "
            "opening frames. Describe the finished target shot.\n"
            "- Do not create image_prompt, image_source, visual_changes, or "
            "keyframe_prompts. Those fields are intentionally absent from the "
            "video-only output schema."
            if not uses_generated_images else ""
        )
        speaker_name_note = (
            "- subjects_on_screen[i].speaker_name: when the screenplay uses a "
            "proper name, record that exact name. Preserve it in H3 video_prompt "
            "and window_prompts, together with useful visible traits."
            if preserve_names else
            "- subjects_on_screen[i].speaker_name: REQUIRED when the screenplay "
            "calls a character by a personal name. Record the EXACT name the "
            "screenplay uses for this character in this shot (e.g. 'Nancy', "
            "'Blaine'). The downstream prompt-polish layer uses this to substitute "
            "the screenplay-invented name with the visual descriptor everywhere "
            "it appears in narrative prose. Without it, names like 'Blaine' leak "
            "into video and image prompts where the generation model has no idea "
            "who that is. If the character has no spoken name in the screenplay "
            "(background extra, unnamed character), set to null or omit the field."
        )

        # Mature-mode guidance is self-gating (see audio-mode pass 2): the
        # version-controlled clinical guides apply only to scenes that are
        # actually sexual, so the block is injected whenever mature mode is on.
        # (Replaces the old explicit-keyword pre-scan, which can't be version-
        # controlled and missed scenes phrased without its keywords.)
        effective_nsfw = nsfw

        # Keyframe guidance is useful only when Director will actually render
        # still-image artifacts for this project.
        keyframe_note = ""
        if uses_generated_images:
            keyframe_note = load_guide("keyframe_rules.md") or "keyframe_prompts: use when the scene involves a visible state change (character enters, clothing removed, moves to new position)."

        image_output_fields = (
            '''    "image_source": "original or previous",
    "image_prompt": "FIRST FRAME BEFORE action begins — the starting visual state. Static pose, environment, lighting. No motion verbs.",
    "visual_changes": ["list what visually transforms — e.g. 'character removes jacket', 'new person enters room', 'camera reveals second character'"],
'''
            if uses_generated_images else ""
        )
        keyframe_output_field = (
            '    "keyframe_prompts": ["(OPTIONAL) only when model needs visual info it cannot generate from the start image"],\n'
            if uses_generated_images else ""
        )
        video_only_subject_note = (
            """- subjects_on_screen[i].visual_description: describe how the character looks IN THIS SHOT, including current wardrobe and story state. Keep each mapped H3 identity name/label and useful visible traits consistent across shots."""
            if not uses_generated_images else ""
        )
        subject_appearance_notes = (
            """- subjects_on_screen[i].visual_description: describe how the character LOOKS IN THIS SHOT per the screenplay, not merely the user's reference-photo clothing. The reference establishes identity, while this field carries the current costume and story state. For example, a reference subject in a black shirt can be a knight in silver armor in one shot and wear a linen tunic later; describe each on-screen state accurately."""
            if uses_generated_images else video_only_subject_note
        )
        image_workflow_notes = (
            """- image_source: "original" (default) = edit from the user's uploaded reference photo. Use for most scenes.
  "previous" = edit from the previous scene's generated output. Use when scenes share the same location
  and visual continuity must carry forward. First scene must always be "original".
- FIELD ORDER MATTERS: Write image_prompt FIRST (the starting state), then visual_changes
  (what transforms), then video_prompt (the action). The start frame must show the BEFORE state.
- visual_changes: List every visible transformation. If a shirt is removed, image_prompt shows it ON;
  if a person enters, image_prompt shows the room before that entrance.
- keyframe_prompts: DEFAULT IS EMPTY. The video model already animates movement, dialogue, expressions,
  camera, and lighting. Add a keyframe only for specific visual information that cannot be inferred from
  the start image and available references, such as a new unmapped identity or a required end-state.
  Each keyframe edits from the start image, so describe only that specific visual change.
"""
            if uses_generated_images else ""
        )

        pass2_system = f"""You are a film director breaking a screenplay into shots. Output ONLY the JSON array.

{char_rules}

{shot_structure}

{keyframe_note}

{video_rules}

{image_prompt_rules}

{visual_strategy_rules}


OUTPUT — respond with ONLY a JSON array:
[
  {{
    "title": "Shot title",
    "duration_sec": 20,
    "scene_goal": "What this shot achieves",
    "narrative_role": "setup|rising_action|climax|resolution",
    "scene_type": "dialogue|action|opening|closing",
    "subjects_on_screen": [{{"visual_description": "woman in red", "character_id": "char_0", "speaker_name": "Nancy"}}],
    "environment": "Setting details",
    "visual_style": "Style",
    "lighting": "Lighting",
    "mood": "Tone",
    "action_beats": ["Action 1", "Action 2"],
    "camera_plan": {{"framing": "medium shot", "movement": "slow push in", "movement_intensity": "subtle"}},
    "ending_beat": "Final moment",
{image_output_fields}    "video_prompt": "Full detailed paragraph describing action — MUST include ALL dialogue in quotes with delivery cues. Physical actions, camera movement, atmosphere.",
    "multishot": false,
{keyframe_output_field}    "window_prompts": []
  }}
]

- multishot: false by default. Set true when the MULTI-SHOT LORA
  MODE block is in this system prompt (above) AND at least one of
  this shot's generations (video_prompt for a 20s shot, or any entry
  in window_prompts for 40s+ shots) uses the storyboard Format B
  instead of the flowing Format A paragraph. The storyboard format
  is the "Shot 1 (Camera, Xs): ..." structured form that the IC-LoRA
  renders as internal camera cuts. The decision is per generation,
  not per shot — a 40s shot can have one storyboard window and one
  flowing window, in which case multishot still equals true.

{subject_appearance_notes}

{speaker_name_note}
{image_workflow_notes}
- window_prompts vs. video_prompt is determined by duration_sec ALONE.
  Use this STRICT decision (no soft zone, no "around 20"):
    duration_sec ≤ 20  → video_prompt populated, window_prompts MUST be []
    duration_sec ≥ 21  → window_prompts populated, video_prompt MUST be ""
  Every shot uses EXACTLY one of the two — never both, never neither.
  21s, 22s, 25s ALL count as "≥ 21" → these MUST use window_prompts.
  Window count for ≥ 21s shots:
    21-40s → 2 windows
    41-60s → 3 windows
    61-80s → 4 windows
  Each window covers ~20s of video. Windows play SEQUENTIALLY — window 2
  continues exactly where window 1 left off, picking up the action mid-flow.
  The video model only sees the last few frames between windows — re-establish
  ongoing state (crowd cheering, rain falling, music playing) at the start
  of each window.
- Each window prompt MUST be a full detailed paragraph (80-150 words).
  Do NOT reuse the same prompt for multiple windows — each window describes
  a different portion of the scene's action chronologically.
{video_name_rules}

PACING — match shot length to story beat, not to a "preferred" average:
- Total duration must sum to ~{target_duration}s.
- KEEP CONVERSATIONS TOGETHER. If two characters are mid-exchange, that is ONE shot — do NOT cut
  mid-conversation into separate shots. A 40s dialogue is one 40s scene with window_prompts,
  not three 13s scenes. Cutting mid-dialogue forces new start images, breaks character
  consistency, and wastes generation time.
- Cut to a new shot when ANY of these is true: location changes, a new character enters,
  a significant time jump, a clear story beat ends and a new one begins, OR a brief reaction
  is the entire dramatic point of the moment.
- Shot-length menu (use the whole range — variety is good filmmaking, not bias toward long):
    * 3-8s   — single reaction, glance, visual punctuation, establishing detail
    * 6-15s  — brief action, transition, short establishing shot
    * 15-40s — dialogue exchange, focused continuous action (one or two windows)
    * 40-80s — sustained scene (multiple windows, conversation that earns its length)
- STRICT 20s threshold: shots ≤ 20s use a single video_prompt with window_prompts=[];
  shots ≥ 21s use window_prompts (one per ~20s slice) with video_prompt="".
  21s, 22s, 25s ALL require window_prompts — there is no "soft zone".
Go:"""

        # ── Multi-shot LoRA mode injection ───────────────────────────
        # When the user has enabled multi-shot LoRA mode (a toggle in
        # services config; defaults off), Pass 2 gets supplementary
        # guidance for mixed-format output.
        #
        # Architecture (revised after first user test):
        # The unit of decision is the GENERATION, not the shot. A
        # generation is one LTX-2 call producing ≤20s of video. Mapping:
        #   - 20s shot = 1 generation (the video_prompt itself)
        #   - 40s shot = 2 generations (each window_prompt is one)
        #   - 60s shot = 3 generations (each window_prompt is one)
        #   - 80s shot = 4 generations (each window_prompt is one)
        #
        # For EACH generation independently, the LLM picks one of two
        # formats:
        #   1. SINGLE-CAMERA FLOWING (default): a flowing paragraph
        #      describing one continuous take.
        #   2. STORYBOARD MULTI-SHOT: a series of "Shot N (Camera, Xs):
        #      description" blocks describing internal camera cuts that
        #      the IC-LoRA will render within the single generation.
        #
        # When to use storyboard: dialogue exchanges, multi-beat
        # interaction, scenes where camera variety helps. When to keep
        # flowing: sustained single beats (a kiss, a sex act, a held
        # reaction), punchy moments, ambient establishing shots.
        #
        # Each window in a 40s+ shot can use a different format —
        # window 1 might be storyboard (dialogue) while window 2 is
        # flowing (the kiss that follows). The decision is per
        # generation.
        if multishot_lora_mode:
            _multishot_block = (
                "\n\n"
                "═══════════════════════════════════════════════════════\n"
                "MULTI-SHOT LORA MODE — USE FORMAT B FOR DIALOGUE\n"
                "═══════════════════════════════════════════════════════\n\n"

                "AN IC-LORA IS LOADED. It renders internal camera cuts "
                "inside one ~20s generation IF you write the prompt in "
                "Format B (storyboard structure). If you write Format A "
                "(flowing prose), the LoRA produces one camera angle and "
                "is doing nothing useful. For a dialogue-heavy film, "
                "Format B should be the default — Format A is the "
                "EXCEPTION for sustained beats.\n\n"

                "FORMAT B — STORYBOARD (default for dialogue/interaction):\n"
                "  Shot 1 (Wide Shot, 5s): description of action this angle.\n"
                "  Shot 2 (Medium Shot, 7s): continuation in new angle.\n"
                "  Shot 3 (Close-up, 4s): continuation in another angle.\n"
                "  Shot 4 (Two-Shot, 4s): final angle of the ~20s slice.\n\n"

                "FORMAT A — FLOWING (only for sustained single beats):\n"
                "A normal flowing paragraph describing ONE continuous "
                "camera take. Use ONLY for: a kiss, a sex act, a held "
                "reaction, a slow push-in — beats that would be RUINED "
                "by camera cuts.\n\n"

                "RULES THAT DO NOT CHANGE:\n"
                "1. The duration→field rule is unchanged. 20s shots use "
                "video_prompt; 40s/60s/80s shots use window_prompts "
                "(one entry per 20s). Format A/B is the CONTENT inside "
                "each field, never which field is populated. Putting "
                "Format B inside video_prompt of a 40s shot triggers "
                "snap-down to 20s and loses content.\n"
                "2. Camera type parens contain ONLY the shot type, "
                "never a character name. 'Close-up', not 'Close-up on "
                "Henry'. Names go inside dialogue quotes in the "
                "description text.\n"
                "3. Two-Shot and Over-the-Shoulder REQUIRE two "
                "characters on screen. For solo moments use "
                "Wide/Medium/Close-up.\n"
                "4. Internal shot durations sum to ~20s; each one is "
                "3-8 seconds; 2-5 internal shots per 20s generation.\n\n"

                "CAMERA TYPES: Wide Shot, Medium Shot, Medium Close-up, "
                "Close-up, Extreme Close-up, Two-Shot, Over-the-Shoulder, "
                "Side Shot, Overhead, Low Angle.\n\n"

                "EXAMPLE — 20s dialogue, Format B:\n"
                "  video_prompt: \"Shot 1 (Wide Shot, 5s): The woman in "
                "russet dress steps onto the porch. Shot 2 (Medium Shot, "
                "7s): The man in cowboy hat turns toward her. He says, "
                "'You're back early.' Shot 3 (Close-up, 4s): Her hand "
                "rests on the railing. Shot 4 (Two-Shot, 4s): She nods.\"\n"
                "  window_prompts: []\n"
                "  multishot: true\n\n"

                "EXAMPLE — 40s dialogue, BOTH windows in Format B:\n"
                "  video_prompt: \"\"\n"
                "  window_prompts: [\n"
                "    \"Shot 1 (Wide Shot, 5s): The woman stands at the "
                "porch railing. Shot 2 (Over-the-Shoulder, 8s): The man "
                "approaches from behind. Shot 3 (Close-up, 7s): He says, "
                "'Sun's setting.' She replies, 'I noticed.'\",\n"
                "    \"Shot 1 (Medium Shot, 6s): They stand close. Shot 2 "
                "(Side Shot, 7s): The man turns his head. He says, 'Stay "
                "a while.' Shot 3 (Close-up, 7s): She tilts her chin up. "
                "She replies, 'I'm not going anywhere.'\"\n"
                "  ]\n"
                "  multishot: true\n\n"

                "EXAMPLE — 40s mixed (dialogue then sustained kiss):\n"
                "  video_prompt: \"\"\n"
                "  window_prompts: [\n"
                "    \"Shot 1 (Medium Shot, 6s): He leans in. He whispers, "
                "'The flame needs kindling.' Shot 2 (Close-up, 7s): Her "
                "breathing hitches. Shot 3 (Two-Shot, 7s): Her hands rest "
                "flat on his chest.\",\n"
                "    \"A slow push-in on the two embracing. He wraps his "
                "arms around her shoulders. The kiss deepens. The camera "
                "holds steady as the light fades to amber.\"\n"
                "  ]\n"
                "  multishot: true   # true because window 1 uses Format B\n\n"

                "EXAMPLE — 20s sustained shot, Format A:\n"
                "  video_prompt: \"A slow push-in on the embracing couple. "
                "Their lips press together. He cups her jaw. The camera "
                "holds steady as the kiss deepens.\"\n"
                "  window_prompts: []\n"
                "  multishot: false\n\n"

                "═══════════════════════════════════════════════════════\n"
                "EXPECTATION: 60-80% of generations should be FORMAT B. If "
                "your final output has ZERO Format B generations on a "
                "script with dialogue, you have UNDERUSED the LoRA and the "
                "user paid for it to do nothing. Re-plan: every window "
                "containing dialogue or character interaction MUST be "
                "Format B. Only sustained beats stay Format A.\n"
                "═══════════════════════════════════════════════════════\n"
            )
            pass2_system = f"{pass2_system}{_multishot_block}"

        if polish_block:
            pass2_system = f"{pass2_system}\n\n{polish_block}"

        # `effective_nsfw` was computed above; reuse it for the
        # inject_nsfw_if_enabled call. The injected guides are
        # self-gating (apply only when a scene is actually sexual).
        pass2_system = inject_nsfw_if_enabled(
            pass2_system,
            effective_nsfw,
            "both" if uses_generated_images else "video",
        )

        # Compute a permissive shot count range so the LLM has creative
        # freedom to match shot length to story beat. Earlier versions
        # used target//35..target//20 which forced a 60s film into 2-3
        # long shots — fine for sustained dialogue, terrible for
        # reaction beats and montages. New range: at least 2 shots
        # (no single-shot films), up to roughly target/8 (allowing a
        # mix of 4-8s reaction beats with longer scenes). The LLM
        # decides where on that spectrum each story sits.
        # Shot count guidance. The high cap is the single biggest lever
        # for forcing the LLM to use long buckets. Math:
        #
        # If shot_count_high = target / 25, the LLM CANNOT hit target
        # using only 20s shots — that would require more shots than the
        # cap allows (180s / 20 = 9, but cap is 7). To hit target, the
        # LLM is forced to mix in 40s/60s/80s buckets. This is the
        # only reliable way to get long shots; prompt-level "use long
        # buckets" guidance alone has been observed to be ignored
        # (latest user test: 10 × 20s = 200s for 180s target).
        #
        # - shot_count_low = target / 40: lower bound, allows long-shot-
        #   dominated films (e.g. 180s = 3 × 60s, 4 shots — though the
        #   floor of max(2, ...) usually wins for short targets).
        # - shot_count_high = target / 25: upper bound, forces long
        #   buckets when target requires it. 180s → 7. 300s → 12.
        #   60s → 2 max from formula but floor brings it to 4 via
        #   max(low+2, ...).
        #
        # The previous target/15 cap let 180s have 12 × 20s = 240s
        # (within accept-zone of 207s ceiling), so the LLM picked the
        # safe all-20s option. target/25 forces the math.
        shot_count_low = max(2, target_duration // 40)
        shot_count_high = max(shot_count_low + 2, target_duration // 25)

        # Pass 2 user prompt construction:
        # 1. /no_think at the top suppresses Qwen3 internal reasoning for
        #    this turn (enforced in Qwen's Jinja chat template directly).
        #    On Qwen3.6-27B, thinking has been observed to spiral into
        #    multi-thousand-token loops that exhaust the budget before
        #    producing actual output. /no_think bypasses the broken
        #    `enable_thinking` chat_template_kwarg path on some llama.cpp
        #    builds. Other models simply ignore the marker.
        # 2. Hard duration + shot-count constraint at the very top — this
        #    used to be buried at line ~643 of the system prompt, but the
        #    LLM ignored it under cognitive load. Hoisting to the user
        #    prompt's first paragraph anchors output structure decisively.
        # 3. The screenplay itself goes last so it remains in the model's
        #    most-recent attention window.
        # Multi-shot LoRA anchor injected into the POPULATION RULE
        # in pass2_user below. Empty string when multi-shot mode is
        # off; a short pointer when on. LLM weighs user-prompt rules
        # more heavily than system-prompt, so the storyboard-format
        # decision needs a visible mention here to avoid the LLM
        # cramming storyboard content into the wrong field (observed
        # production bug: 40s shots ended up with populated
        # video_prompt + empty window_prompts, then snap-down lost
        # half the runtime).
        multishot_user_anchor = (
            "   MULTI-SHOT LORA MODE: storyboard format goes INSIDE "
            "the field this rule says to populate. For a 40s shot, "
            "that means TWO entries in window_prompts, each "
            "independently formatted as storyboard OR flowing. NEVER "
            "put a storyboard inside video_prompt of a 40s+ shot — "
            "the system will snap the duration down to 20s. See the "
            "MULTI-SHOT LORA MODE block in the system prompt above "
            "for examples."
        ) if multishot_lora_mode else ""
        generation_inputs = (
            "a single prompt + start frame"
            if uses_generated_images
            else "a self-contained prompt plus any mapped H3 references"
        )
        keyframe_user_rule = (
            "- Use keyframes ONLY when the video model needs visual info it "
            "cannot generate from the start image (new character entry, "
            "clothing reveal, dramatic state change). Do NOT use keyframes as "
            "a substitute for animating dialogue — the video model handles all "
            "talking, gestures, and expressions on its own."
            if uses_generated_images
            else "- Do not output image_prompt, image_source, visual_changes, "
            "or keyframe_prompts; this H3 workflow renders directly from video "
            "prompts and mapped references."
        )

        pass2_user = f"""/no_think

TASK: Break this {target_duration}-second screenplay into {shot_count_low}-{shot_count_high} distinct shots.

CRITICAL OUTPUT REQUIREMENTS (these override any conflicting system-prompt guidance):

1. EXACTLY {shot_count_low} TO {shot_count_high} SHOTS. No more. Going over this count
   means you're fragmenting — every shot under 20s is a sign you cut where
   the video model could have rendered continuous action. Re-merge.

2. SHOT DURATION MUST BE ONE OF: 20, 40, 60, 80 seconds.
   - 20s = single beat (a transition, a brief reaction, an
     establishing moment, a short dialogue exchange). One prompt,
     no windows.
   - 40s = TWO connected beats that flow together as one continuous
     scene (an extended dialogue, a foreplay-to-act transition, a
     slow reveal, a full kiss + embrace). USE FREELY — don't default
     to two 20s shots when the screenplay has a continuous 40s beat.
     Two windows.
   - 60s = THREE connected beats in one sustained scene (a long
     romantic encounter, a sex sequence, a confrontation that builds
     and breaks). Three windows. Common in NSFW films and any film
     with sustained dramatic scenes — don't avoid 60s shots.
   - 80s = FOUR connected beats in a single uninterrupted sequence
     (a sustained sex act, a long climactic confrontation, an extended
     seduction). Four windows. Use when the screenplay has a beat
     that genuinely needs the breathing room.
   - HEURISTIC: aim for variety. A {target_duration}s film with NINE
     20s shots feels choppy; a film with three 20s shots + two 40s +
     one 60s feels cinematic. Mix the bucket sizes.
   - NEVER 5, 8, 10, 15, 22, 25, 30, 35, 45, 50, 55, 65, 70, 75. Those
     all create stranded short tail windows that render as sluggish stubs.

3. TOTAL duration_sec MUST sum to {target_duration} seconds (±5%).
   With 20s shots that's exactly {target_duration // 20} shots. With one
   40s shot mixed in, the rest fit into {(target_duration - 40) // 20} 20s shots.

4. POPULATION RULE — single hard threshold (THIS RULE OVERRIDES THE
   MULTI-SHOT LORA MODE BLOCK BELOW IF YOU TRY TO BREAK IT):
   - duration_sec == 20 → populate video_prompt, window_prompts=[]
   - duration_sec ∈ {{40, 60, 80}} → populate window_prompts (one per 20s),
     video_prompt=""
   Each window is a full paragraph (80-150 words) describing 20s of action.
   {multishot_user_anchor}

5. THE VIDEO MODEL HANDLES INTRA-SHOT PROGRESSION. ONE 20s shot can show
   the woman walking closer, raising her hand to his chest, kneeling, and
   beginning a new action — the model renders all of that from
   {generation_inputs}. You do NOT need separate shots for "she steps
   closer", "her hand moves", "she kneels", "she begins to..." — those
   are micro-beats, NOT shot boundaries.

   ONLY cut to a new shot when ONE of these changes:
     - LOCATION (different room, indoor↔outdoor)
     - TIME (skip ahead — "later that evening")
     - CAST (a new character enters / someone exits)
     - DRAMATIC PIVOT (clear emotional inflection)
   DO NOT cut for: position, gesture, expression, camera movement, or
   action progression within an ongoing scene.

WHEN THE SCREENPLAY IS TOO DENSE FOR {target_duration}s — DROP CONTENT, DON'T ADD SHOTS:
The user asked for {target_duration} seconds. If the screenplay implies more, do NOT
solve it by adding more shots or stretching duration_sec. Instead:
  * DROP whole beats from the screenplay (a transition, a redundant line).
  * MERGE adjacent beats into one shot — most multi-beat content fits in
    a single 20s shot's prompt.
  * SHORTEN dialogue (cut the second back-and-forth, condense speeches).
A {shot_count_high}-shot film at 20s each is {shot_count_high * 20}s. If your plan
exceeds that count or that total, you are fragmenting or over-budget — re-plan.

SHOT BOUNDARIES (do not overlap):
Each shot covers a distinct, NON-overlapping span of the screenplay's
timeline. If Shot 2 covers minute 0:00-0:30 of action, Shot 3 starts at 0:30
and never re-uses lines from Shot 2. Do NOT include the same dialogue
exchange across multiple shots.

The user's original request:
{story_description}

Shot-construction rules:
- KEEP CONTINUOUS ACTION TOGETHER — physical progression that flows from one beat to the next is ONE shot. See the WRONG/RIGHT examples above. The video model handles intra-shot action progression; do not fragment.
- KEEP CONVERSATIONS TOGETHER — one conversation = one shot, using window_prompts if over 20s.
- MIX BUCKET SIZES. Use 40s for connected dialogue/action pairs. Use 60s for long romantic / dramatic / sex scenes. Use 80s for genuinely sustained sequences. With only {shot_count_high} shots allowed total, you CANNOT hit {target_duration}s using only 20s — the math forces you to use longer buckets. That is intentional: longer buckets produce more cinematic, less choppy films.
- Only cut to a new shot when location changes, a new character enters, or there's a clear dramatic beat transition (see strict criteria above).
- Preserve ALL dialogue from the screenplay verbatim — but each line goes in EXACTLY ONE shot/window, never repeated.
{keyframe_user_rule}

SCREENPLAY:
{screenplay}"""

        # Video-only H3 plans omit four still-image fields, so reserve a smaller
        # output budget than the generated-image contract.
        # `/no_think` above suppresses Qwen thinking. `thinking_budget=None`
        # delegates to _call_llm_json's model-aware default (Qwen→0, Gemma→4096).
        # Gemma 4B specifically benefits from thinking when planning the strict
        # 20s window threshold and total-duration arithmetic.
        tokens_per_second = 100 if uses_generated_images else 80
        max_tokens = max(8192, target_duration * tokens_per_second)

        # Grammar constraint (thinking-off models' first attempt + every
        # retry — see _call_llm_json). The shot-count bounds make the
        # prompt's "{shot_count_low}-{shot_count_high} shots" rule grammar-
        # enforced, and the closed shot object makes the observed failure
        # (Gemma 4 12B looping 96K chars of repeating shot pseudo-JSON)
        # unrepresentable. keyframe_prompts stays optional (spec tags it
        # OPTIONAL); window_prompts is required because the ≤20s/≥21s
        # pairing rule expects an explicit [] on short shots.
        pass2_schema = _shot_list_schema(
            min_items=shot_count_low,
            max_items=shot_count_high,
            required=[
                "title", "duration_sec", "scene_goal", "narrative_role",
                "scene_type", "subjects_on_screen", "environment",
                "visual_style", "lighting", "mood", "action_beats",
                "camera_plan", "ending_beat", "image_source", "image_prompt",
                "visual_changes", "video_prompt", "multishot",
                "window_prompts",
            ],
            include_image_fields=uses_generated_images,
        )

        shot_dicts = self._call_llm_json(
            user_prompt=pass2_user,
            system_prompt=pass2_system,
            max_tokens=max_tokens,
            thinking_budget=None,
            image_paths=image_paths,
            json_schema=pass2_schema,
        )
        if not uses_generated_images:
            _discard_unused_image_fields(shot_dicts)

        # ── POST-PASS-2 SAFETY SCAN ─────────────────────────────────────
        # Defense in depth — Pass 2's structured output (image/video
        # prompts, action beats, dialogue, subjects) gets concatenated
        # and scanned the same way the screenplay was. Catches the case
        # where Pass 1 produced clean text but Pass 2's expansion
        # introduced minor + sexual co-occurrence.
        assert_no_minor_content(
            collect_pass2_text(shot_dicts), source="shot list (Pass 2)"
        )

        # ── CHARACTER DESCRIPTOR CANONICALIZATION ────────────────────
        # User-reported bug: uploaded selfie tagged "man in black",
        # screenplay turned the character into a knight in silver armor,
        # but Pass 2 inconsistently described them — some shots said
        # "man in black" (the user's reference descriptor), others said
        # "knight in silver armor" (the in-story appearance). Result: the
        # image generator put the character in armor in some scenes and
        # back into a black shirt in others.
        #
        # Prompt-level guidance to use the in-story descriptor was added
        # in commit 9263c8a but the LLM still doesn't follow it
        # consistently. This is the deterministic safety net.
        #
        # Algorithm:
        # 1. For each character_id, collect every visual_description used
        #    across shots.
        # 2. Filter out descriptors that match the user's char_profile
        #    descriptor (case-insensitive) — those are the ones we want
        #    to REPLACE.
        # 3. Pick the most-common non-user descriptor as the "canonical
        #    in-story descriptor" for that character.
        # 4. Replace the user's descriptor with the canonical one in:
        #    - subjects_on_screen[i].visual_description
        #    - video_prompt
        #    - image_prompt
        #    - window_prompts entries
        #    - keyframe_prompts entries
        #
        # Only fires when the canonical descriptor appears in ≥2 shots —
        # if there's only a one-off transformation, the LLM may have
        # intended a one-shot variation (flashback, costume change) and
        # we should not force consistency.
        try:
            from collections import Counter as _Counter, defaultdict as _DefaultDict

            user_descriptors_by_cid: dict[str, str] = {}
            for c in (char_profiles or []):
                cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
                desc = (
                    getattr(c, "physical_description", None)
                    or (c.get("physical_description") if isinstance(c, dict) else None)
                    or ""
                )
                if cid and desc:
                    user_descriptors_by_cid[cid] = desc.strip().lower()

            descs_by_cid: dict[str, list[str]] = _DefaultDict(list)
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                for subj in (sd.get("subjects_on_screen") or []):
                    if not isinstance(subj, dict):
                        continue
                    cid = subj.get("character_id")
                    vd = (subj.get("visual_description") or "").strip()
                    if cid and vd:
                        descs_by_cid[cid].append(vd)

            canonical_by_cid: dict[str, str] = {}
            for cid, descs in descs_by_cid.items():
                user_desc = user_descriptors_by_cid.get(cid, "")
                if not user_desc:
                    continue
                non_user = [d for d in descs if d.strip().lower() != user_desc]
                if not non_user:
                    continue  # all match user descriptor — no transformation
                counter = _Counter(non_user)
                most_common, count = counter.most_common(1)[0]
                # Require ≥2 occurrences to consider it canonical.
                # Single-shot variations are likely intentional (flashback,
                # costume change) and should not be forced across the
                # whole production.
                if count >= 2:
                    canonical_by_cid[cid] = most_common

            if canonical_by_cid:
                import re as _re_can
                for cid, canonical in canonical_by_cid.items():
                    user_desc_raw = next(
                        (
                            (getattr(c, "physical_description", None)
                             or (c.get("physical_description") if isinstance(c, dict) else None))
                            for c in (char_profiles or [])
                            if (getattr(c, "id", None) == cid
                                or (isinstance(c, dict) and c.get("id") == cid))
                        ),
                        None,
                    )
                    if not user_desc_raw:
                        continue
                    user_desc_raw = user_desc_raw.strip()
                    pat = _re_can.compile(
                        r"\b" + _re_can.escape(user_desc_raw) + r"\b",
                        _re_can.IGNORECASE,
                    )
                    replacements = 0
                    for sd in shot_dicts:
                        if not isinstance(sd, dict):
                            continue
                        # subjects_on_screen
                        for subj in (sd.get("subjects_on_screen") or []):
                            if not isinstance(subj, dict):
                                continue
                            if subj.get("character_id") != cid:
                                continue
                            vd = (subj.get("visual_description") or "").strip()
                            if vd.lower() == user_desc_raw.lower():
                                subj["visual_description"] = canonical
                                replacements += 1
                        # text fields
                        for field in ("video_prompt", "image_prompt"):
                            text = sd.get(field) or ""
                            if text:
                                new_text, n = pat.subn(canonical, text)
                                if n:
                                    sd[field] = new_text
                                    replacements += n
                        # array text fields
                        for arr_field in ("window_prompts", "keyframe_prompts"):
                            arr = sd.get(arr_field) or []
                            if not isinstance(arr, list):
                                continue
                            new_arr = []
                            for item in arr:
                                if isinstance(item, str):
                                    new_item, n = pat.subn(canonical, item)
                                    if n:
                                        replacements += n
                                    new_arr.append(new_item)
                                else:
                                    new_arr.append(item)
                            sd[arr_field] = new_arr
                    if replacements:
                        print(
                            f"[ShortFilmPlanner] Canonicalized {cid} "
                            f"descriptor across {replacements} location(s): "
                            f"replaced reference description '{user_desc_raw}' "
                            f"with in-story description '{canonical}'. "
                            f"(LLM was inconsistent — some shots used the "
                            f"reference photo's description, others used "
                            f"the screenplay's transformed description; "
                            f"forcing the transformed one for consistency.)"
                        )
        except Exception as _canon_err:
            print(f"[ShortFilmPlanner] Descriptor canonicalization skipped: {_canon_err}")

        # ── POST-PASS-2 OVER-FRAGMENTATION MERGE ──────────────────────
        # When the LLM emits way more shots than the target shot-count
        # range (e.g. 36 shots for a 180s target where the range is
        # 6-12), merge adjacent short shots into single 20s shots.
        # Without this step, every short shot gets snap-up'd to 20s by
        # the per-shot post-process, ballooning the total runtime,
        # which then triggers the duration scale-down — and the result
        # is N tiny shots crammed into target seconds, the worst of
        # both worlds.
        #
        # Merge strategy: walk the shot list in order, accumulating
        # adjacent short shots (≤15s) into one merged shot until the
        # accumulated duration would exceed 20s. Concatenate their
        # video_prompts (with " " separator), drop their keyframes
        # (stale after merge), keep the FIRST shot's image_prompt and
        # subjects_on_screen (since the merged shot opens on that
        # frame). Boundary detection: stop accumulating when location
        # or scene_type changes — those are real shot boundaries even
        # in a fragmented run.
        try:
            _max_shots = max(2, target_duration // 15)  # generous ceiling
            if len(shot_dicts) > _max_shots * 1.3 and shot_dicts:
                pre_merge_count = len(shot_dicts)
                merged_shots: list[dict] = []
                bucket: list[dict] = []
                bucket_dur = 0

                def _flush_bucket():
                    nonlocal bucket, bucket_dur
                    if not bucket:
                        return
                    if len(bucket) == 1:
                        merged_shots.append(bucket[0])
                    else:
                        head = dict(bucket[0])
                        # Concatenate video_prompts in order, preserving
                        # each shot's intended action sequence.
                        prompts = []
                        for s in bucket:
                            vp = (s.get("video_prompt") or "").strip()
                            if vp:
                                prompts.append(vp)
                        if prompts:
                            head["video_prompt"] = " ".join(prompts)
                        head["window_prompts"] = []
                        head["duration_sec"] = 20
                        # Drop keyframes — they were placed for the
                        # original tiny shots and don't fit a single
                        # merged 20s shot.
                        head["keyframe_prompts"] = []
                        # Concatenate action_beats for downstream tools
                        # that read them.
                        all_beats: list = []
                        for s in bucket:
                            ab = s.get("action_beats") or []
                            if isinstance(ab, list):
                                all_beats.extend(ab)
                        if all_beats:
                            head["action_beats"] = all_beats
                        merged_shots.append(head)
                    bucket = []
                    bucket_dur = 0

                for sd in shot_dicts:
                    if not isinstance(sd, dict):
                        merged_shots.append(sd)
                        continue
                    dur = int(sd.get("duration_sec", 0) or 0)
                    has_windows = bool(sd.get("window_prompts"))
                    # Don't merge: long shots, multi-window shots, or
                    # shots that change location/scene-type from the
                    # bucket head.
                    is_short = (0 < dur <= 15) and not has_windows
                    boundary = False
                    if bucket and is_short:
                        head = bucket[0]
                        if (sd.get("environment") and head.get("environment")
                                and sd.get("environment") != head.get("environment")):
                            boundary = True
                        if (sd.get("scene_type") and head.get("scene_type")
                                and sd.get("scene_type") != head.get("scene_type")):
                            boundary = True
                    if not is_short or boundary:
                        _flush_bucket()
                        merged_shots.append(sd)
                        continue
                    # Would adding this shot push the bucket past 20s?
                    if bucket_dur + dur > 20 and bucket:
                        _flush_bucket()
                    bucket.append(sd)
                    bucket_dur += dur
                _flush_bucket()

                if len(merged_shots) < pre_merge_count:
                    print(
                        f"[ShortFilmPlanner] ⚠ Pass 2 over-fragmented: "
                        f"{pre_merge_count} shots > {_max_shots} expected. "
                        f"Merged adjacent short shots → {len(merged_shots)} shots. "
                        f"Each merged shot's video_prompts concatenated; "
                        f"keyframes dropped (stale after merge)."
                    )
                    shot_dicts[:] = merged_shots
        except Exception as _merge_err:
            print(f"[ShortFilmPlanner] Adjacent-shot merge skipped: {_merge_err}")

        # ── POST-PASS-2 DURATION ENFORCEMENT ─────────────────────────
        # User-reported lesson from production: scaling 20s shots down
        # to 17-18s "to hit the exact target" is pointless. The user
        # would rather have clean 20-second buckets and slightly miss
        # the runtime target than hit the runtime exactly with awkward
        # mid-bucket durations that violate the model's window-
        # threshold rules.
        #
        # Three-tier policy:
        #
        # Tier 1 — accept (≤15% over):
        #   The LLM's overshoot is small enough to live with. Log it
        #   and move on. This handles the common case where Pass 1
        #   was a bit dense and Pass 2 ended at, say, 200s for a 180s
        #   target. User gets a 200s film with clean buckets — better
        #   than an exact 180s film with 17s shots.
        #
        # Tier 2 — bucket-aware reduction (15% to 50% over):
        #   Find shots in larger buckets (40/60/80s) and snap each
        #   down to the next-smaller bucket until total fits. Each
        #   snap removes exactly 20s of runtime AND one window of
        #   content (the last window of that shot). Preserves the
        #   bucket grid; the only "compression" is dropping content,
        #   not stretching it.
        #
        # Tier 3 — proportional fallback (>50% over):
        #   Runaway LLM. Apply proportional scale, then run a final
        #   snap-to-bucket cleanup that rounds each shot back to a
        #   valid bucket value (20/40/60/80). The result may exceed
        #   target after rounding — accepted as a known fail mode for
        #   pathological inputs.
        _raw_total = sum(
            int(sd.get("duration_sec", 0) or 0)
            for sd in shot_dicts
            if isinstance(sd, dict)
        )
        _ceiling = int(target_duration * 1.15)
        _scale_threshold = int(target_duration * 1.50)

        def _snap_bucket(sd: dict) -> None:
            """Snap a single shot's duration_sec to nearest valid bucket
            and align window_prompts/video_prompt accordingly. Idempotent.
            """
            d = int(sd.get("duration_sec", 0) or 0)
            if d <= 0 or d in (20, 40, 60, 80):
                return
            if d < 20:
                new_d = 20
            else:
                tail = d % 20
                if tail == 0:
                    return
                new_d = (d - tail) if tail <= 10 else (d + (20 - tail))
                new_d = max(20, new_d)
            sd["duration_sec"] = new_d
            # Adjust windows to match new bucket count
            n_target = max(1, new_d // 20)
            wps = sd.get("window_prompts") or []
            if new_d == 20 and wps:
                # Convert windows to a single video_prompt
                sd["video_prompt"] = " ".join(str(w) for w in wps)
                sd["window_prompts"] = []
            elif new_d > 20 and len(wps) > n_target:
                # Trim excess windows (merge into last)
                kept = list(wps[:n_target - 1])
                merged = " ".join(str(w) for w in wps[n_target - 1:])
                kept.append(merged)
                sd["window_prompts"] = kept

        if _raw_total <= _ceiling:
            # Tier 1
            _delta = _raw_total - target_duration
            _sign = "+" if _delta >= 0 else ""
            print(
                f"[ShortFilmPlanner] Pass 2 duration: {_raw_total}s "
                f"({len(shot_dicts)} shots) vs {target_duration}s target "
                f"({_sign}{_delta}s, within {_ceiling}s ceiling — no compression)."
            )
        elif _raw_total <= _scale_threshold:
            # Tier 2 — bucket-aware reduction
            _bucket_down = {80: 60, 60: 40, 40: 20}
            excess = _raw_total - target_duration
            print(
                f"[ShortFilmPlanner] ⚠ Pass 2 over budget: "
                f"{_raw_total}s total vs {target_duration}s target "
                f"(ceiling {_ceiling}s, +{_raw_total - target_duration}s overrun). "
                f"Bucket-down: snapping large shots to smaller buckets."
            )
            # Sort largest-bucket-first so we prefer reducing 60s→40s
            # over 40s→20s when the choice exists (preserves more
            # sustained scenes).
            candidates = sorted(
                [sd for sd in shot_dicts
                 if isinstance(sd, dict)
                 and sd.get("duration_sec") in _bucket_down],
                key=lambda s: -int(s.get("duration_sec", 0) or 0),
            )
            snapped: list[str] = []
            for sd in candidates:
                if excess <= 0:
                    break
                cur = int(sd.get("duration_sec", 0) or 0)
                nxt = _bucket_down[cur]
                # Drop the last window's content (it's the one being
                # cut). For 40s→20s that means drop one window AND
                # convert the surviving window to video_prompt.
                wps = list(sd.get("window_prompts") or [])
                if wps:
                    wps = wps[:-1]
                    if nxt == 20:
                        sd["video_prompt"] = (
                            " ".join(str(w) for w in wps) if wps else
                            sd.get("video_prompt", "") or ""
                        )
                        sd["window_prompts"] = []
                    else:
                        sd["window_prompts"] = wps
                sd["duration_sec"] = nxt
                excess -= (cur - nxt)
                snapped.append(
                    f"'{sd.get('title', 'untitled')}' {cur}s→{nxt}s"
                )
            _new_total = sum(
                int(sd.get("duration_sec", 0) or 0)
                for sd in shot_dicts
                if isinstance(sd, dict)
            )
            if snapped:
                print(
                    f"[ShortFilmPlanner] Bucket-down: "
                    f"{', '.join(snapped)}. "
                    f"New total: {_new_total}s "
                    f"({_raw_total - _new_total}s removed)."
                )
            else:
                # No bucket-down candidates (all shots already 20s).
                # Accept the overshoot rather than chop content.
                print(
                    f"[ShortFilmPlanner] No bucket-down candidates "
                    f"(all shots are 20s). Accepting {_new_total}s "
                    f"overshoot vs {target_duration}s target."
                )
            # Always run snap-cleanup so any leftover non-bucket dur
            # (e.g. from earlier snap-up steps) gets normalized.
            for sd in shot_dicts:
                if isinstance(sd, dict):
                    _snap_bucket(sd)
        else:
            # Tier 3 — runaway. Proportional scale + bucket cleanup.
            scale = target_duration / _raw_total if _raw_total else 1.0
            print(
                f"[ShortFilmPlanner] ⚠ Pass 2 SEVERELY over budget: "
                f"{_raw_total}s total vs {target_duration}s target "
                f"(ceiling {_ceiling}s, +{_raw_total - target_duration}s, "
                f">{int((_scale_threshold/target_duration - 1) * 100)}% over). "
                f"Proportional scale {scale:.2%} + bucket cleanup."
            )
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                old_dur = int(sd.get("duration_sec", 0) or 0)
                if old_dur <= 0:
                    continue
                sd["duration_sec"] = max(3, int(old_dur * scale))
            for sd in shot_dicts:
                if isinstance(sd, dict):
                    _snap_bucket(sd)
            _new_total = sum(
                int(sd.get("duration_sec", 0) or 0)
                for sd in shot_dicts
                if isinstance(sd, dict)
            )
            print(
                f"[ShortFilmPlanner] After scale + bucket cleanup: "
                f"{_new_total}s ({len(shot_dicts)} shots)."
            )

        # Deterministic post-process: fix structural rule violations the LLM
        # makes despite all prompt-level guidance. Two passes:
        #
        # 1. WINDOW COUNT OVERSHOOT — Gemma 4B sometimes emits 3 windows
        #    for a 35s shot when the formula calls for 2. Trim excess
        #    windows and merge their content into the last surviving one.
        #
        # 2. RUSHED TAIL WINDOW — when duration_sec is not a multiple of 20
        #    (e.g. 25s, 35s, 45s), the backend allocates 20s to each full
        #    window and gives the tail window only the remainder. A 25s
        #    shot with 2 windows gets W1=20s, W2=5s — the 5s window is
        #    far too short to fit the dialogue/action the LLM wrote for
        #    it. Empirically, anything <10s of tail is "rushed". Fix by
        #    merging the rushed tail into the previous window AND snapping
        #    duration_sec down to the resulting clean multiple of 20.
        import math as _math
        for sd in shot_dicts:
            try:
                dur = int(sd.get("duration_sec", 0) or 0)
                wps = sd.get("window_prompts", []) or []
                if dur <= 20:
                    continue
                # ── Pass 0: shot violates the "≥21s = use window_prompts"
                # rule by populating video_prompt instead. Common LLM
                # violation, especially Gemma 4B on NSFW screenplays
                # where attention to structural rules drops. Snap down
                # to 20s so the shot fits a single video_prompt cleanly,
                # since the LLM clearly intended one continuous block of
                # action (not multiple windows).
                if not wps:
                    vp = sd.get("video_prompt", "") or ""
                    if vp.strip():
                        sd["duration_sec"] = 20
                        # Drop keyframes — they were placed for the LLM's
                        # original (longer, multi-stage) intent. After
                        # snapping to a single 20s video_prompt, those
                        # keyframes are stale visual references that
                        # over-constrain a now-simpler shot.
                        had_kfs = bool(sd.get("keyframe_prompts"))
                        if had_kfs:
                            sd["keyframe_prompts"] = []
                        print(
                            f"[ShortFilmPlanner] Snap-down (video_prompt only) in '{sd.get('title', 'untitled')}': "
                            f"{dur}s → 20s — LLM populated video_prompt for a >20s shot instead of window_prompts; "
                            "treating as single 20s shot to match the LLM's structural intent"
                            + (" (also cleared stale keyframes)" if had_kfs else "")
                        )
                    # If both video_prompt and window_prompts are empty,
                    # nothing to do — the shot is malformed.
                    continue
                # ── Pass 0c: reconcile MIXED-STATE shots ──────────────
                # The strict rule is "≤20s → video_prompt only; ≥21s →
                # window_prompts only." The LLM sometimes violates it
                # by populating BOTH. The polish layer
                # (prompt_polish.py:1046) and the gen layer both pick
                # window_prompts when it has 2+ entries and silently
                # drop video_prompt — so any dialogue the LLM put in
                # video_prompt gets discarded before it reaches the
                # video model.
                #
                # Reconcile here based on where the actual dialogue
                # lives (detected by quoted text containing 3+ words).
                # The user-reported failure looked exactly like this:
                # 25s shot with full scene + dialogue in video_prompt
                # and short "same scene, medium shot..." stub strings
                # in window_prompts (the LLM treated them as keyframes).
                # Detect quoted-dialogue spans of ≥3 words, accepting
                # straight + smart quotes. `re.finditer` caches the
                # compiled pattern internally so repeated calls are cheap.
                import re as _re_dlg
                _DIALOGUE_PAT = r'[\"\'“”‘’]([^\"\'“”‘’]{12,})[\"\'“”‘’]'
                def _has_dialogue(text: str) -> bool:
                    if not isinstance(text, str) or not text.strip():
                        return False
                    for m in _re_dlg.finditer(_DIALOGUE_PAT, text):
                        if len(m.group(1).split()) >= 3:
                            return True
                    return False

                vp_text = (sd.get("video_prompt") or "").strip()
                if vp_text and wps:
                    vp_has_dialogue = _has_dialogue(vp_text)
                    wps_have_dialogue = any(_has_dialogue(w) for w in wps if isinstance(w, str))
                    vp_words = len(vp_text.split())
                    wp_words_max = max((len(w.split()) for w in wps if isinstance(w, str)), default=0)

                    # CASE A: video_prompt has dialogue, window_prompts
                    # don't. The LLM put the real scene content in
                    # video_prompt and treated window_prompts as
                    # keyframe-shaped stubs (e.g. "same scene, close-up
                    # of her face..."). Collapse to a 20s single shot
                    # using video_prompt — the dialogue must be
                    # preserved or the scene loses its core content.
                    if vp_has_dialogue and not wps_have_dialogue:
                        wp_count_before = len(wps)
                        sd["window_prompts"] = []
                        sd["duration_sec"] = 20
                        had_kfs = bool(sd.get("keyframe_prompts"))
                        if had_kfs:
                            sd["keyframe_prompts"] = []
                        wps = []
                        print(
                            f"[ShortFilmPlanner] Mixed-state reconciled (Case A) in '{sd.get('title', 'untitled')}': "
                            f"video_prompt has dialogue ({vp_words}w), window_prompts don't "
                            f"({wp_count_before} stubs, max {wp_words_max}w) → collapsed to 20s single "
                            f"video_prompt. Without this, the polish layer would skip video_prompt entirely "
                            f"(because window_prompts has 2+ entries) and the dialogue would be silently "
                            f"dropped before video gen."
                            + (" (also cleared stale keyframes)" if had_kfs else "")
                        )
                        continue
                    # CASE B: window_prompts have dialogue (LLM
                    # followed the rule for windows but ALSO left a
                    # stale video_prompt). Clear video_prompt so the
                    # unused field doesn't confuse anyone downstream.
                    if wps_have_dialogue:
                        sd["video_prompt"] = ""
                        print(
                            f"[ShortFilmPlanner] Mixed-state reconciled (Case B) in '{sd.get('title', 'untitled')}': "
                            f"both fields populated; window_prompts have the dialogue, video_prompt cleared "
                            f"(was {vp_words}w of redundant content the polish layer would have ignored)"
                        )
                    # CASE C: neither has dialogue (action-only scene
                    # where the LLM violated the either/or rule). Keep
                    # window_prompts since the duration calls for them,
                    # clear video_prompt.
                    else:
                        sd["video_prompt"] = ""
                        print(
                            f"[ShortFilmPlanner] Mixed-state reconciled (Case C) in '{sd.get('title', 'untitled')}': "
                            f"both fields populated, no dialogue in either; window_prompts kept "
                            f"(matches {dur}s duration), video_prompt cleared (was {vp_words}w)"
                        )

                # ── Pass 0b: window-count UNDERSHOOT. LLM produced fewer
                # windows than the duration calls for (e.g. 30s shot with
                # only 1 window_prompt). Without this fix, the wgp pipeline
                # generates the full duration anyway and uses the single
                # window prompt for both windows — producing the action-
                # looping behavior the original rule was designed to
                # prevent. Snap duration down to 20 × len(wps) so the
                # shot fits the actual window count cleanly. We lose the
                # missing window's worth of intended runtime but avoid
                # repeating the same prompt across two windows.
                expected_pre = max(1, _math.ceil(dur / 20.0))
                actual_pre = len(wps)
                if actual_pre < expected_pre:
                    new_dur = 20 * actual_pre
                    if new_dur < dur:
                        sd["duration_sec"] = new_dur
                        had_kfs = bool(sd.get("keyframe_prompts"))
                        # If snapped down to a single window, switch to
                        # video_prompt to satisfy the strict ≤20s rule
                        # (window_prompts is for >20s shots only).
                        # Also drop keyframes — they were placed for the
                        # LLM's original (longer) intent and are stale
                        # references on a now-simpler single-prompt shot.
                        if actual_pre == 1:
                            sd["video_prompt"] = str(wps[0])
                            sd["window_prompts"] = []
                            wps = []
                            if had_kfs:
                                sd["keyframe_prompts"] = []
                        print(
                            f"[ShortFilmPlanner] Snap-down (window undershoot) in '{sd.get('title', 'untitled')}': "
                            f"{dur}s → {new_dur}s — LLM emitted {actual_pre} "
                            f"window(s) for a {dur}s shot (needed {expected_pre}); "
                            "duration trimmed to match actual window count"
                            + (" (also cleared stale keyframes)" if had_kfs and actual_pre == 1 else "")
                        )
                        # Update dur for subsequent passes; if windows
                        # got cleared (snap to single video_prompt),
                        # skip the rest of the per-window passes.
                        dur = new_dur
                        if not wps:
                            continue
                # ── Pass 1: window-count overshoot ─────────────────────
                expected = max(1, _math.ceil(dur / 20.0))
                actual = len(wps)
                if actual > expected:
                    keep = list(wps[: expected - 1])
                    merged_tail = " ".join(str(w) for w in wps[expected - 1:])
                    keep.append(merged_tail)
                    sd["window_prompts"] = keep
                    wps = keep
                    print(
                        f"[ShortFilmPlanner] Fixed window overshoot in '{sd.get('title', 'untitled')}': "
                        f"{actual} → {expected} windows for {dur}s shot (excess merged into last window)"
                    )
                # ── Pass 2: snap to multiple-of-20 duration buckets ────
                # User-facing rule: shots are EITHER ≤20s (single
                # video_prompt, no windows) OR exactly a multiple of 20s
                # (40, 60, 80) for sustained continuous action that
                # genuinely warrants the longer runtime. NEVER 22s, 25s,
                # 30s, 35s, 45s — these create stranded tail windows
                # (e.g. a 25s shot is W1=20s + W2=5s, where W2 renders
                # as a sluggish stub and the cut into the next shot
                # feels jagged).
                #
                # The Pass 2 user prompt already tells the LLM "duration
                # MUST be one of 20/40/60/80". This post-process is the
                # safety net for when the LLM picks an invalid value
                # anyway. Snap direction picks the NEAREST valid bucket:
                #
                #   tail = duration_sec % 20
                #   tail == 0       → already valid, no change
                #   1 ≤ tail ≤ 10   → snap DOWN (subtract tail, merge
                #                     last window's content into previous)
                #   11 ≤ tail ≤ 19  → snap UP (add 20-tail seconds, last
                #                     window covers a longer effective
                #                     time but receives no extra content)
                #
                # Why split the snap direction at the midpoint: if the
                # LLM wrote 25s of content (tail=5), it sized only ~15-30
                # words for the tail window. Snapping down merges those
                # words into the previous 20s window — minor compression,
                # acceptable. If the LLM wrote 35s of content (tail=15),
                # the tail window has a near-full 60-100 words. Cramming
                # those into the previous 20s window would rush dialogue
                # significantly. Snapping up to 40s preserves pacing
                # (the 5s expansion just gives the last window a few
                # extra seconds of breathing room). 40s shots are
                # explicitly allowed by the new rule.
                #
                # Special case: 1-window shots whose duration_sec exceeds
                # 20 by 1-10s (e.g. 25s with no windows) snap down to
                # 20s and stay single-video_prompt. Anything ≥ 21s
                # should already be in window form per the threshold
                # rules, but we handle the malformed case defensively.
                n = len(wps)
                tail_seconds = dur % 20
                if dur > 0 and tail_seconds != 0:
                    had_kfs = bool(sd.get("keyframe_prompts"))
                    cleared_kfs = False
                    if tail_seconds <= 10:
                        # Snap DOWN: drop the tail.
                        new_dur = dur - tail_seconds
                        if new_dur < 20:
                            new_dur = 20  # never go below the minimum
                        if n == 0:
                            # 1-window shot (≤20s case shouldn't reach
                            # here, but defensive). Just clamp duration.
                            sd["duration_sec"] = new_dur
                            print(
                                f"[ShortFilmPlanner] Snap-down (tail {tail_seconds}s) "
                                f"in '{sd.get('title', 'untitled')}': "
                                f"{dur}s → {new_dur}s (no windows)"
                            )
                        elif n == 1:
                            # Was 21-30s with one window. Snap to 20s,
                            # convert window to video_prompt.
                            sd["duration_sec"] = new_dur
                            if new_dur == 20:
                                sd["video_prompt"] = str(wps[0])
                                sd["window_prompts"] = []
                                if had_kfs:
                                    sd["keyframe_prompts"] = []
                                    cleared_kfs = True
                            print(
                                f"[ShortFilmPlanner] Snap-down (tail {tail_seconds}s) "
                                f"in '{sd.get('title', 'untitled')}': "
                                f"{dur}s → {new_dur}s "
                                + ("(now single video_prompt)" if new_dur == 20 else "")
                                + (" (also cleared stale keyframes)" if cleared_kfs else "")
                            )
                        else:
                            # Multi-window: merge last window into previous.
                            merged = str(wps[-2]) + " " + str(wps[-1])
                            new_windows = list(wps[:-2]) + [merged]
                            sd["window_prompts"] = new_windows
                            sd["duration_sec"] = new_dur
                            if len(new_windows) == 1:
                                sd["video_prompt"] = merged
                                sd["window_prompts"] = []
                                if had_kfs:
                                    sd["keyframe_prompts"] = []
                                    cleared_kfs = True
                            print(
                                f"[ShortFilmPlanner] Snap-down (tail {tail_seconds}s) "
                                f"in '{sd.get('title', 'untitled')}': "
                                f"{dur}s ({n} windows) → {new_dur}s "
                                f"({len(new_windows)} window(s)) — small tail merged into previous"
                                + (" (also cleared stale keyframes)" if cleared_kfs else "")
                            )
                    else:
                        # tail 11-19s → snap UP (preserve content, accept
                        # a few extra seconds of runtime). The new_dur is
                        # the next multiple of 20.
                        new_dur = dur + (20 - tail_seconds)
                        sd["duration_sec"] = new_dur
                        # If we started with no windows but now need them
                        # (≤20s → >20s wouldn't happen here since dur was
                        # already > 20 to have a non-zero tail; but
                        # defensive against edge cases like dur=11):
                        if new_dur > 20 and n == 0:
                            # Originating shot was malformed (single
                            # video_prompt with dur > 20). Convert to
                            # window form.
                            sd["window_prompts"] = [
                                str(sd.get("video_prompt", "") or ""),
                                "",  # second window blank — Pass 2 LLM
                                     # didn't intend a multi-window shot
                            ][:max(1, _math.ceil(new_dur / 20.0))]
                            sd["video_prompt"] = ""
                        print(
                            f"[ShortFilmPlanner] Snap-up (tail {tail_seconds}s) "
                            f"in '{sd.get('title', 'untitled')}': "
                            f"{dur}s → {new_dur}s — last window covers "
                            f"a slightly longer effective time, content unchanged"
                        )

                    # Diagnostic: warn when a window's content looks
                    # over-stuffed for its allocated time. Doesn't fix
                    # anything but flags pacing problems for future
                    # iteration. ~150 words/20s ≈ 7.5 words/s, so
                    # window with > 10 words/s of content is suspect.
                    try:
                        for wi, wp in enumerate(wps):
                            if not isinstance(wp, str):
                                continue
                            window_seconds = (
                                20 if wi < n - 1
                                else max(1, dur - 20 * (n - 1))
                            )
                            word_count = len(wp.split())
                            words_per_sec = word_count / window_seconds
                            if words_per_sec > 10:
                                print(
                                    f"[ShortFilmPlanner] Pacing warning in "
                                    f"'{sd.get('title', 'untitled')}' "
                                    f"window {wi+1}: {word_count} words for "
                                    f"{window_seconds}s ({words_per_sec:.1f} w/s) "
                                    f"— may render rushed"
                                )
                    except Exception:
                        pass
            except Exception as e:
                print(f"[ShortFilmPlanner] Duration post-process skipped a shot: {e}")

        # ── Image-prompt sanitization (Layer 1) ──────────────────────
        # Strip GARMENT BAN violations and narrative-filler phrases the
        # image model can't render. Runs on every shot's image_prompt
        # AND each keyframe_prompt regardless of whether Pass 3 polish
        # is enabled — Pass 2 LLM (especially Gemma 4B on NSFW) routinely
        # writes "white sweater" / "grey shirt" and emotion fillers like
        # "showing the heat of the moment" despite the rules. Pass 3
        # runs the same sanitizer again with the descriptor-dedupe pass
        # added (since it has the name_to_descriptor map). No-op when
        # the LLM already followed the rules.
        try:
            from ..prompt_polish import sanitize_image_prompt as _sanitize_ip
            for sd in shot_dicts:
                ip = sd.get("image_prompt") or ""
                if ip.strip():
                    sd["image_prompt"] = _sanitize_ip(
                        ip, log_prefix=f"[ShortFilmPlanner Pass2 image sanitize '{sd.get('title', 'untitled')}']"
                    )
                kfs = sd.get("keyframe_prompts") or []
                if isinstance(kfs, list) and kfs:
                    cleaned_kfs = []
                    for ki, kf in enumerate(kfs):
                        if isinstance(kf, str) and kf.strip():
                            cleaned_kfs.append(_sanitize_ip(
                                kf, log_prefix=f"[ShortFilmPlanner Pass2 keyframe[{ki}] sanitize '{sd.get('title', 'untitled')}']"
                            ))
                        else:
                            cleaned_kfs.append(kf)
                    sd["keyframe_prompts"] = cleaned_kfs
        except Exception as e:
            print(f"[ShortFilmPlanner] Image-prompt sanitization skipped: {e}")

        # ── Sex-act leet trigger strip (always-on safety net) ────────
        # User-reported leak: a SFW music video had "bl0wj0b" in a
        # keyframe_prompt. Same risk applies to short films when a
        # user has NSFW LoRAs in their video_loras selection from
        # prior testing and runs a SFW concept. Strip from image and
        # keyframe fields ALWAYS (still images don't use video LoRA
        # triggers). Strip from video/window fields when nsfw=False.
        try:
            from ..prompt_polish import strip_sex_act_leet_tokens as _strip_leet
            leet_count = 0
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                ip = sd.get("image_prompt") or ""
                if ip:
                    new_ip, n = _strip_leet(ip)
                    if n:
                        sd["image_prompt"] = new_ip
                        leet_count += n
                kfs = sd.get("keyframe_prompts") or []
                if isinstance(kfs, list):
                    new_kfs = []
                    for kf in kfs:
                        if isinstance(kf, str):
                            new_kf, n = _strip_leet(kf)
                            new_kfs.append(new_kf)
                            leet_count += n
                        else:
                            new_kfs.append(kf)
                    sd["keyframe_prompts"] = new_kfs
                if not nsfw:
                    vp = sd.get("video_prompt") or ""
                    if vp:
                        new_vp, n = _strip_leet(vp)
                        if n:
                            sd["video_prompt"] = new_vp
                            leet_count += n
                    wps_local = sd.get("window_prompts") or []
                    if isinstance(wps_local, list):
                        new_wps = []
                        for w in wps_local:
                            if isinstance(w, str):
                                new_w, n = _strip_leet(w)
                                new_wps.append(new_w)
                                leet_count += n
                            else:
                                new_wps.append(w)
                        sd["window_prompts"] = new_wps
            if leet_count:
                print(
                    f"[ShortFilmPlanner] Stripped {leet_count} sex-act leet "
                    f"trigger token(s) — LLM placed them in fields where they "
                    f"don't belong (still images or SFW video context)."
                )
        except Exception as e:
            print(f"[ShortFilmPlanner] Leet trigger strip skipped: {e}")

        # ── Storyboard camera-name leak strip (Multi-Shot LoRA mode) ─
        # When Pass 2 produced Format B storyboard prompts, the LLM
        # sometimes embeds character names inside the camera-type
        # parens ("Shot 2 (Close-up on Henry, 7s):"). The IC-LoRA was
        # trained on clean camera-type tokens; names in the parens
        # break the trained pattern. Strip the "on Henry" / "of Mary"
        # / "from Mary" / "with Mary" / "over Mary's shoulder" leak
        # everywhere it appears (video_prompt and each window_prompts
        # entry).
        try:
            from ..prompt_polish import strip_storyboard_camera_name_leaks
            total_stripped = 0
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                vp = sd.get("video_prompt") or ""
                if vp:
                    new_vp, n = strip_storyboard_camera_name_leaks(vp)
                    if n:
                        sd["video_prompt"] = new_vp
                        total_stripped += n
                wps_local = sd.get("window_prompts") or []
                if isinstance(wps_local, list):
                    new_wps = []
                    for w in wps_local:
                        if isinstance(w, str):
                            new_w, n = strip_storyboard_camera_name_leaks(w)
                            new_wps.append(new_w)
                            total_stripped += n
                        else:
                            new_wps.append(w)
                    sd["window_prompts"] = new_wps
            if total_stripped:
                print(
                    f"[ShortFilmPlanner] Stripped {total_stripped} character-"
                    f"name leak(s) from storyboard camera-type parens "
                    f"(e.g. 'Close-up on Henry' → 'Close-up')."
                )
        except Exception as e:
            print(f"[ShortFilmPlanner] Storyboard camera-name strip skipped: {e}")

        # Deduplicate scenes
        seen_goals = set()
        unique_dicts = []
        for sd in shot_dicts:
            goal = sd.get("scene_goal", "")
            if goal not in seen_goals:
                seen_goals.add(goal)
                unique_dicts.append(sd)

        shots = self._convert_story_shots(unique_dicts, char_profiles, has_reference, fps, frames_steps, frames_minimum)

        # Extract title from first shot if available
        title = shot_dicts[0].get("title") if shot_dicts else None
        self._last_title = title

        return shots, title

    def _plan_story_h3_native(
        self,
        *,
        story_description: str,
        screenplay: str,
        reference_image_path: Optional[str],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        target_duration: int,
        fps: int,
        frames_steps: int,
        frames_minimum: int,
        frames_maximum: Optional[int],
        screenplay_dialogue_manifest: Optional[list[dict[str, Any]]] = None,
        character_voice_bible: Optional[list[dict[str, str]]] = None,
        nsfw: bool = False,
        polish_block: str = "",
    ) -> tuple[list[ShotPlan], Optional[str]]:
        """Break a screenplay directly into self-contained native H3 shots."""

        from ..nsfw_guidance import inject_nsfw_if_enabled
        from ..safety_scan import assert_no_minor_content, collect_pass2_text

        uses_generated_images = bool(
            getattr(self, "_uses_generated_shot_images", True)
        )
        preserve_names = bool(
            getattr(self, "_preserve_video_character_names", False)
        )
        fps = max(1, int(fps or 24))
        if frames_maximum is None:
            # Direct planner callers predate model-aware kwargs and still pass
            # the generic 41/8 defaults. H3's built-in native contract is
            # 124..345 frames on a 17-frame lattice at 24 fps.
            frames_steps = 17
            frames_minimum = 124
            frames_maximum = 345
        frames_steps = max(1, int(frames_steps or 17))
        frames_minimum = max(1, int(frames_minimum or 124))
        frames_maximum = max(
            frames_minimum,
            int(frames_maximum or 345),
        )
        minimum_seconds = frames_minimum / fps
        maximum_seconds = frames_maximum / fps
        maximum_dialogue_words = int(math.floor(
            maximum_seconds * _H3_DIALOGUE_WORDS_PER_SECOND,
        ))
        shot_count_low = max(1, math.ceil(target_duration / maximum_seconds))
        maximum_by_runtime = max(
            shot_count_low,
            math.floor(target_duration / minimum_seconds),
        )
        # Prefer ordinary editorial-length clips while respecting a smaller
        # hardware-safe ceiling (for example 5.17s at wide 1080p on 24 GB).
        shot_count_high = max(
            shot_count_low,
            min(maximum_by_runtime, math.ceil(target_duration / 7.5)),
        )
        preferred_durations = _h3_preferred_native_durations(
            fps=fps,
            frames_minimum=frames_minimum,
            frames_maximum=frames_maximum,
            frames_steps=frames_steps,
        )
        preferred_duration_text = ", ".join(
            f"{duration:.2f}s" for duration in preferred_durations
        )
        example_duration = preferred_durations[-1]

        char_rules = build_character_rules_block(
            has_reference or bool(getattr(self, "_num_character_refs", 0)),
            char_profiles if char_profiles else None,
            preserve_names=preserve_names,
        )
        video_rules = _route_video_pass2_guide(
            getattr(self, "_video_model", "") or "minimax_h3"
        )
        video_name_rules = _video_character_name_rules(preserve_names)

        image_rules = ""
        image_fields = ""
        image_requirements: list[str] = []
        if uses_generated_images:
            from ..image_prompt_rules import get_image_prompt_rules
            image_rules = get_image_prompt_rules(
                has_reference,
                num_character_refs=getattr(self, "_num_character_refs", 0),
                num_location_refs=getattr(self, "_num_location_refs", 0),
                character_ref_labels=getattr(self, "_character_ref_labels", None),
                location_ref_labels=getattr(self, "_location_ref_labels", None),
                seamless=getattr(self, "_seamless", True),
                image_model=getattr(self, "_image_model", "") or "",
            )
            image_fields = '''    "image_source": "original or previous",
    "image_prompt": "Static first-frame composition before the action",
    "visual_changes": [],
'''
            image_requirements = [
                "image_source", "image_prompt", "visual_changes",
            ]

        if screenplay_dialogue_manifest is None:
            screenplay_dialogue_manifest = _extract_h3_screenplay_dialogue(
                screenplay
            )
        else:
            screenplay_dialogue_manifest = copy.deepcopy(
                screenplay_dialogue_manifest
            )
        character_voice_bible = copy.deepcopy(character_voice_bible or [])
        voice_bible_text = _format_h3_voice_bible(character_voice_bible)
        dialogue_manifest_json = _h3_dialogue_manifest_prompt(
            screenplay_dialogue_manifest
        )
        print(
            "[ShortFilmPlanner] Locked "
            f"{len(screenplay_dialogue_manifest)} screenplay dialogue turn(s) "
            "before H3 shot planning."
        )

        voice_bible_block = (
            "CHARACTER VOICE BIBLE — BINDING FOR PERFORMANCE AND BEHAVIOR:\n"
            f"{voice_bible_text}\n"
            "Use this only to stage character-appropriate reactions, delivery, "
            "and conversational behavior. The immutable dialogue manifest "
            "already contains the final spoken words; do not rewrite them."
            if voice_bible_text else
            "CHARACTER VOICE BIBLE: No separate profile was validated. Preserve "
            "the screenplay's speaker identities and dialogue exactly."
        )

        pass2_system = f"""You are a film director breaking a screenplay into shots for MiniMax H3. Output ONLY the JSON array.

H3 NATIVE SHOT CONTRACT — NON-NEGOTIABLE:
- Every array item is ONE bounded H3 generation lasting {minimum_seconds:.2f}-{maximum_seconds:.2f} seconds.
- Use video_prompt for every item and set window_prompts to []. Never write 20-second windows, timeline ranges, or prompt instructions referring to a previous/preceding shot. The structured continuity_strategy and continuity_group fields are required planning metadata, while every video_prompt must remain self-contained.
- Every video_prompt must stand alone. Restate the exact physical setting, all visible people, their appearance/wardrobe, action, camera, lighting, dialogue, ambience, effects, and music needed in that clip.
- WARDROBE IS STATE: on every appearance, subjects_on_screen must give each person a complete head-to-toe wardrobe (colors, materials, layers, accessories, and visible footwear). Repeat the same wardrobe wording in every shot within a continuity_group unless the screenplay explicitly changes it, and show that change visibly before using the new wardrobe.
- BLOCKING IS STATE: spatial_setup and each subject's position_or_relation describe the FIRST FRAME precisely: screen-left/center/right, foreground/midground/background, standing/seated/leaning, facing direction, and nearby furniture or props. Repeat that opening blocking in video_prompt.
- closing_blocking describes the final positions at the end of the shot. When the next shot in the same continuity_group opens with different blocking, the current shot must visibly show the person walking, sitting, standing, turning, or otherwise moving into that next arrangement before the cut.
- continuity_strategy is "independent" for a new place/time, "continuous" for a normal editorial cut within the same scene, or "extend_previous" ONLY when the next generation should start from the literal final frame with the same camera axis/composition and no intended cut. Use extend_previous sparingly.
- continuity_group is a short stable ID such as kitchen_morning_1. Reuse it only while place and story time remain uninterrupted; change it for any location or time jump.
- WORLD CONTINUITY IS REQUIRED: preserve any supplied TV show, film, performer, franchise, historical era, city, named venue, room, or recognizable set. Repeat the relevant world/franchise and full location in EACH video_prompt; never collapse a named series and its recognizable apartment set into a generic kitchen.
- Each screenplay event and each spoken line appears in exactly one shot. Do not duplicate dialogue across adjacent shots. Preserve scripted dialogue verbatim.
- CONVERSATION PACKING IS REQUIRED: a change of speaker is not by itself a reason to start another array item. Within the same uninterrupted location and story beat, prefer one native clip ({preferred_duration_text}) containing 2-4 alternating dialogue turns when their combined total is no more than {maximum_dialogue_words} words. Keep a brief reaction such as "What?", a gasp, or a one-line reply in the surrounding exchange instead of wasting a separate minimum-length clip.
- INTERNAL CAMERA EDITING IS SUPPORTED: inside one bounded H3 clip, the camera may begin on an ensemble frame, cut or reframe to each current speaker before their tagged line, hold their unobstructed face and mouth through the complete line, capture reactions, and finish on a new composition. Describe that chronological coverage in camera_plan and action_beats. Prefer the lower end of the requested shot-count range for a continuous dialogue scene.
- DIALOGUE MUST NOT LIVE ONLY IN dialogue_beats. Every dialogue_beats[].spoken_text must also appear exactly once in the same shot's video_prompt as <d>[English] Exact words</d>, with the speaker ID/name, delivery, and physical cue outside the tag. If dialogue_beats is empty, explicitly state that no one speaks, mouths remain closed, and no muttering, gibberish, or speech-like vocalization occurs.
- SPEAKER VISIBILITY IS REQUIRED: every person who delivers a line must have a complete subjects_on_screen entry and remain visibly framed with an unobstructed face and mouth for the full line. Reframe to the current speaker before speech; reaction framing may follow only after the spoken line is complete.
- CAST LIST CONSISTENCY IS REQUIRED: every person mentioned in spatial_setup, action_beats, dialogue_beats, ending_beat, closing_blocking, or video_prompt must appear in subjects_on_screen. Do not mention a bystander in blocking while omitting that person from the visible cast.
- A shot may follow another in the finished edit, but its prompt must describe its own opening state instead of saying "continue", "as before", "the push-in continues", or similar.
- multishot is always false because this is not the LTX Multi-Shot LoRA format. That field does NOT prohibit H3 from making speaker-motivated internal cuts, reframes, and reaction coverage inside its one bounded generation.

{char_rules}

{video_name_rules}

{video_rules}

{voice_bible_block}

{image_rules}

OUTPUT — one closed object per native shot:
[
  {{
    "title": "Shot title",
    "duration_sec": {example_duration:.2f},
    "scene_goal": "Unique story beat",
    "narrative_role": "setup|rising_action|climax|resolution",
    "scene_type": "dialogue|action|opening|closing",
    "continuity_strategy": "independent|continuous|extend_previous",
    "continuity_group": "stable_scene_id",
    "subjects_on_screen": [{{"visual_description": "Stable identity and physical appearance", "character_id": "char_0", "speaker_name": "Exact supplied name", "position_or_relation": "screen-left foreground, standing beside the counter and facing screen-right", "wardrobe": "mustard-yellow cotton shirt, brown tie, dark slacks, black belt, black shoes"}}],
    "spatial_setup": "Exact first-frame screen blocking for every visible person and important prop",
    "environment": "Exact world/franchise and complete physical location",
    "visual_style": "Series/film visual language and medium",
    "lighting": "Shot lighting",
    "mood": "Tone",
    "action_beats": ["Chronological visible actions"],
    "dialogue_beats": [{{"speaker_id": "char_0", "spoken_text": "Exact words", "delivery": "Delivery", "physical_cue": "Visible cue", "priority": "high"}}],
    "camera_plan": {{"framing": "medium shot", "movement": "slow push in", "movement_intensity": "subtle"}},
    "audio_plan": {{"mode": "dialogue_driven", "ambience": "Location ambience", "effects": ["Synchronized practical effects"], "vocal_style": "Natural voices", "timing_anchor": "audio", "lip_sync_critical": true}},
    "ending_beat": "Visible end state",
    "closing_blocking": "Exact final screen positions and poses after all movement",
{image_fields}    "video_prompt": "MiniMax H3 Context-IR prompt with integrated_multimodal_description, overall_soundscape, and non_diegetic_music",
    "multishot": false,
    "window_prompts": []
  }}
]"""
        if polish_block:
            pass2_system = f"{pass2_system}\n\n{polish_block}"
        pass2_system = inject_nsfw_if_enabled(
            pass2_system,
            nsfw,
            "both" if uses_generated_images else "video",
        )

        pass2_user = f"""/no_think

TASK: Convert this {target_duration}-second screenplay into {shot_count_low}-{shot_count_high} self-contained native H3 shots.

Total duration should remain approximately {target_duration} seconds. Each duration_sec must be between {minimum_seconds:.2f} and {maximum_seconds:.2f} seconds; prefer these valid native durations when pacing permits: {preferred_duration_text}. Do not output a duration above {maximum_seconds:.2f} seconds.

PROJECT WORLD SOURCE OF TRUTH:
{story_description}

IMMUTABLE SCREENPLAY DIALOGUE MANIFEST:
{dialogue_manifest_json}

The manifest is authoritative. Emit every listed turn exactly once, in that
order, with the same speaker_name represented by dialogue_beats[].speaker_id
and a matching visible subjects_on_screen entry. Do not add dialogue. An empty
manifest means every shot is silent.

Repeat the relevant show/movie/franchise and exact physical location from that source of truth in every shot's environment AND video_prompt. Character names alone are not enough. Only the action/dialogue assigned to that one shot may occur in its prompt.

Continuity audit before responding:
1. Assign one stable full wardrobe to each character for each continuity_group and repeat it in every appearance.
2. Compare every shot's closing_blocking with the next shot's spatial_setup.
3. If the same-scene positions differ, put the required movement in the earlier shot's action_beats and video_prompt so the next opening is earned on screen.
4. Use extend_previous only for a literal seamless continuation with unchanged camera composition. Use continuous for ordinary same-scene cuts.
5. Cross-check every dialogue_beats entry against video_prompt. Copy each spoken_text verbatim into one <d>[English] ...</d> tag. For a silent shot, forbid invented speech and gibberish explicitly.

SCREENPLAY:
{screenplay}"""

        required = [
            "title", "duration_sec", "scene_goal", "narrative_role",
            "scene_type", "continuity_strategy", "continuity_group",
            "subjects_on_screen", "spatial_setup", "environment",
            "visual_style", "lighting", "mood", "action_beats",
            "dialogue_beats", "camera_plan", "audio_plan", "ending_beat",
            "closing_blocking",
            *image_requirements,
            "video_prompt", "multishot", "window_prompts",
        ]

        def _strengthen_native_schema(configured: dict) -> dict:
            configured["items"]["properties"]["window_prompts"] = {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 0,
            }
            configured["items"]["properties"]["subjects_on_screen"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": dict(_SUBJECT_SCHEMA["properties"]),
                    "required": [
                        "visual_description", "character_id", "speaker_name",
                        "position_or_relation", "wardrobe",
                    ],
                    "additionalProperties": False,
                },
            }
            dialogue_schema = dict(_DIALOGUE_BEAT_SCHEMA)
            dialogue_schema["required"] = [
                "speaker_id", "spoken_text", "delivery", "physical_cue",
                "priority",
            ]
            configured["items"]["properties"]["dialogue_beats"] = {
                "type": "array",
                "items": dialogue_schema,
            }
            return configured

        schema = _shot_list_schema(
            min_items=shot_count_low,
            max_items=shot_count_high,
            required=required,
            include_image_fields=uses_generated_images,
        )
        schema = _strengthen_native_schema(schema)

        def _normalize_native_shots(items: list[dict]) -> list[dict]:
            normalized: list[dict] = []
            if not uses_generated_images:
                _discard_unused_image_fields(items)
            for raw in items:
                if not isinstance(raw, dict):
                    continue
                windows = raw.get("window_prompts") or []
                prompt = raw.get("video_prompt") or " ".join(
                    str(item.get("prompt") or item.get("text") or "")
                    if isinstance(item, dict) else str(item or "")
                    for item in windows
                )
                raw["video_prompt"] = _sanitize_h3_independent_prompt(prompt)
                raw["window_prompts"] = []
                raw["multishot"] = False
                normalized.append(raw)
            return normalized

        def _apply_native_schedule(
            items: list[dict],
            *,
            protect_dialogue: bool = False,
        ) -> list[int]:
            raw_durations = []
            dialogue_frame_floors: list[int] = []
            for raw in items:
                try:
                    raw_durations.append(float(raw.get("duration_sec") or 10))
                except (TypeError, ValueError):
                    raw_durations.append(10.0)
                if protect_dialogue:
                    spoken_words = sum(
                        len(_h3_plain_dialogue_text(
                            beat.get("spoken_text")
                        ).split())
                        for beat in (raw.get("dialogue_beats") or [])
                        if isinstance(beat, dict)
                    )
                    dialogue_frame_floors.append(math.ceil(
                        spoken_words * fps / _H3_DIALOGUE_WORDS_PER_SECOND
                    ))
            frame_schedule = _fit_bounded_frame_schedule(
                raw_durations,
                target_duration=target_duration,
                fps=fps,
                minimum_frames=frames_minimum,
                maximum_frames=frames_maximum,
                frame_step=frames_steps,
                minimum_frames_by_item=(
                    dialogue_frame_floors if protect_dialogue else None
                ),
            )
            for raw, frame_count in zip(items, frame_schedule):
                raw["duration_sec"] = frame_count / fps
            return frame_schedule

        def _configure_native_schema(max_items: int) -> dict:
            configured = _shot_list_schema(
                min_items=shot_count_low,
                max_items=max_items,
                required=required,
                include_image_fields=uses_generated_images,
            )
            return _strengthen_native_schema(configured)

        def _compile_locked_dialogue(
            items: list[dict],
            current_schedule: list[int],
            *,
            known_items: Optional[list[dict]] = None,
        ) -> tuple[list[dict], list[int], str]:
            """Install Pass 1 dialogue without trusting Pass 2 to copy it.

            The visual planner's dialogue beats are placement hints only. If
            their count/speaker mapping is intact, overwrite them directly
            from the immutable manifest. Otherwise allocate the complete
            manifest across the visual shots, first within the current timing
            and then within H3's legal maximum before fitting the smallest
            dialogue-safe frame schedule.
            """

            candidate = copy.deepcopy(items)
            known = [
                raw for raw in (known_items or [])
                if isinstance(raw, dict)
            ]
            try:
                _reconcile_h3_dialogue_manifest(
                    candidate,
                    screenplay_dialogue_manifest,
                    known_items=known,
                    allow_manifest_restore=True,
                )
                mode = "canonicalized existing dialogue slots"
            except ValueError:
                source = _h3_manifest_dialogue_source(
                    screenplay_dialogue_manifest,
                    [*known, *candidate],
                )
                allocation_errors: list[str] = []
                attempts = [
                    (
                        [frames / fps for frames in current_schedule],
                        False,
                    ),
                    (
                        [frames_maximum / fps] * len(candidate),
                        True,
                    ),
                ]
                allocated = None
                allocated_schedule: list[int] = []
                for durations, needs_refit in attempts:
                    trial = copy.deepcopy(candidate)
                    try:
                        trial = _restore_h3_dialogue_after_pacing_repair(
                            source,
                            trial,
                            durations,
                        )
                    except ValueError as error:
                        allocation_errors.append(str(error))
                        continue
                    trial_schedule = (
                        _apply_native_schedule(trial, protect_dialogue=True)
                        if needs_refit else list(current_schedule)
                    )
                    violations = _h3_dialogue_budget_violations(
                        trial,
                        [frames / fps for frames in trial_schedule],
                        words_per_second=_H3_DIALOGUE_WORDS_PER_SECOND,
                    )
                    if violations:
                        allocation_errors.append(
                            "allocated dialogue still exceeded legal clip timing"
                        )
                        continue
                    allocated = trial
                    allocated_schedule = trial_schedule
                    break
                if allocated is None:
                    raise ValueError(
                        next(
                            (message for message in reversed(allocation_errors) if message),
                            "the complete screenplay dialogue could not be allocated",
                        )
                    )
                candidate = allocated
                current_schedule = allocated_schedule
                mode = "allocated every manifest turn into visual shot slots"

            # Canonical slots can still be clustered too tightly. First give
            # their existing shots enough legal time; if that cannot fit, use
            # the same deterministic allocator to redistribute whole turns.
            fitted_schedule = _apply_native_schedule(
                candidate,
                protect_dialogue=True,
            )
            violations = _h3_dialogue_budget_violations(
                candidate,
                [frames / fps for frames in fitted_schedule],
                words_per_second=_H3_DIALOGUE_WORDS_PER_SECOND,
            )
            if violations:
                source = copy.deepcopy(candidate)
                redistributed = copy.deepcopy(candidate)
                redistributed = _restore_h3_dialogue_after_pacing_repair(
                    source,
                    redistributed,
                    [frames_maximum / fps] * len(redistributed),
                )
                fitted_schedule = _apply_native_schedule(
                    redistributed,
                    protect_dialogue=True,
                )
                violations = _h3_dialogue_budget_violations(
                    redistributed,
                    [frames / fps for frames in fitted_schedule],
                    words_per_second=_H3_DIALOGUE_WORDS_PER_SECOND,
                )
                if violations:
                    raise ValueError(
                        "the complete screenplay dialogue still exceeds H3's "
                        "legal clip timing after deterministic allocation"
                    )
                candidate = redistributed
                mode += " and redistributed crowded turns"

            _reconcile_h3_dialogue_manifest(
                candidate,
                screenplay_dialogue_manifest,
                known_items=known,
                allow_manifest_sentence_splits=True,
            )
            return candidate, fitted_schedule, mode

        image_paths = self._build_all_image_paths(
            reference_image_path, has_reference
        )
        print(
            "[ShortFilmPlanner] Pass 2: Planning native MiniMax H3 shots "
            f"({minimum_seconds:.2f}-{maximum_seconds:.2f}s, "
            f"{shot_count_low}-{shot_count_high} shots)..."
        )
        planner_token_budget = _h3_planner_token_budget(target_duration)
        raw_shot_dicts = self._call_llm_json(
            user_prompt=pass2_user,
            system_prompt=pass2_system,
            max_tokens=planner_token_budget,
            thinking_budget=None,
            temperature=0.4,
            image_paths=image_paths,
            json_schema=schema,
        )
        structure_issues = _h3_native_structure_issues(
            raw_shot_dicts,
            required,
            minimum_items=shot_count_low,
            maximum_items=shot_count_high,
        )
        shot_dicts = _normalize_native_shots(raw_shot_dicts)
        schedule = _apply_native_schedule(shot_dicts)

        dialogue_integrity_error: Optional[str] = None
        try:
            _reconcile_h3_dialogue_manifest(
                shot_dicts,
                screenplay_dialogue_manifest,
            )
        except ValueError as error:
            dialogue_integrity_error = str(error)

        dialogue_violations = _h3_dialogue_budget_violations(
            shot_dicts,
            [frames / fps for frames in schedule],
            words_per_second=_H3_DIALOGUE_WORDS_PER_SECOND,
        )
        if not structure_issues and (
            dialogue_integrity_error or dialogue_violations
        ):
            try:
                shot_dicts, schedule, repair_mode = _compile_locked_dialogue(
                    shot_dicts,
                    schedule,
                    known_items=shot_dicts,
                )
            except ValueError as error:
                dialogue_integrity_error = str(error)
            else:
                dialogue_integrity_error = None
                dialogue_violations = []
                print(
                    "[ShortFilmPlanner] Deterministic H3 dialogue compiler "
                    f"{repair_mode}; skipped the whole-plan LLM repair."
                )
        if structure_issues or dialogue_integrity_error or dialogue_violations:
            original_shot_dicts = copy.deepcopy(shot_dicts)
            issue_messages = [
                f"- Incomplete structured output: {issue}"
                for issue in structure_issues
            ]
            if dialogue_integrity_error:
                issue_messages.append(
                    "- Dialogue integrity: " + dialogue_integrity_error
                )
            issue_messages.extend(
                f"- Shot {item['index'] + 1} ({item['title']}): "
                f"{item['word_count']} spoken words but only "
                f"{item['word_budget']} fit in {item['duration_sec']:.2f}s."
                for item in dialogue_violations
            )
            issue_lines = "\n".join(issue_messages)
            repair_max_items = max(shot_count_high, maximum_by_runtime)
            repair_user = f"""{pass2_user}

H3 WHOLE-PLAN REPAIR - YOUR PREVIOUS PLAN WAS REJECTED:
{issue_lines}

Rewrite the COMPLETE plan from the screenplay, including its ending. The
immutable dialogue manifest above must appear exactly once, in order, with
each line kept in the visual shot whose action and visible speaker match it.
You may use {shot_count_low}-{repair_max_items} shots. Increase a shot only up
to {maximum_seconds:.2f}s or use additional self-contained shots. Never return
a partial array, truncate a line, move dialogue to an unrelated visual beat,
nest <d> tags, or exceed roughly two spoken words per second in any shot.

Keep structured metadata concise so the complete array fits: one short
sentence per descriptive metadata field and at most three action beats. Put
the complete standalone generation description in video_prompt instead of
repeating that prose across every metadata field."""
            print(
                "[ShortFilmPlanner] H3 plan failed completeness, dialogue, "
                "or pacing validation; requesting one whole-plan repair "
                "before generation."
            )
            repaired_raw = self._call_llm_json(
                user_prompt=repair_user,
                system_prompt=pass2_system,
                max_tokens=planner_token_budget,
                thinking_budget=0,
                temperature=0.3,
                image_paths=image_paths,
                json_schema=_configure_native_schema(repair_max_items),
            )
            recovered_tail_fields = _complete_h3_truncated_tail(
                repaired_raw,
                required,
            )
            if recovered_tail_fields:
                print(
                    "[ShortFilmPlanner] Recovered token-capped final H3 shot "
                    "from its complete semantic core; deterministically "
                    "filled: " + ", ".join(recovered_tail_fields)
                )
            repair_structure_issues = _h3_native_structure_issues(
                repaired_raw,
                required,
                minimum_items=shot_count_low,
                maximum_items=repair_max_items,
            )
            if repair_structure_issues:
                raise RuntimeError(
                    "MiniMax H3 returned an incomplete shot plan after its "
                    "automatic repair ("
                    + "; ".join(repair_structure_issues)
                    + "). No video jobs were queued."
                )
            shot_dicts = _normalize_native_shots(repaired_raw)
            if not shot_dicts:
                raise RuntimeError(
                    "MiniMax H3 whole-plan repair returned no usable "
                    "shots. No video jobs were queued."
                )
            schedule = _apply_native_schedule(shot_dicts)
            try:
                shot_dicts, schedule, repair_mode = _compile_locked_dialogue(
                    shot_dicts,
                    schedule,
                    known_items=original_shot_dicts,
                )
            except ValueError as error:
                raise RuntimeError(
                    "MiniMax H3's repaired visual shot plan could not accept "
                    "the locked screenplay dialogue: "
                    f"{error}. No video jobs were queued."
                ) from error
            print(
                "[ShortFilmPlanner] Deterministic H3 dialogue compiler "
                f"{repair_mode}; the visual repair's rewritten dialogue was "
                "ignored."
            )
            dialogue_violations = _h3_dialogue_budget_violations(
                shot_dicts,
                [frames / fps for frames in schedule],
                words_per_second=_H3_DIALOGUE_WORDS_PER_SECOND,
            )
            if dialogue_violations:
                # The LLM repair may preserve every line but still cluster too
                # much speech into a few shots. Resolve that deterministically:
                # first redistribute time without extending the target runtime,
                # then re-bucket complete dialogue turns, and only then permit
                # the smallest model-lattice runtime extension required.
                authoritative_shots = copy.deepcopy(shot_dicts)
                original_schedule_total = sum(schedule)

                retimed_shots = copy.deepcopy(authoritative_shots)
                retimed_schedule = _apply_native_schedule(
                    retimed_shots,
                    protect_dialogue=True,
                )
                retimed_violations = _h3_dialogue_budget_violations(
                    retimed_shots,
                    [frames / fps for frames in retimed_schedule],
                )
                if (
                    not retimed_violations
                    and sum(retimed_schedule)
                    <= original_schedule_total + frames_steps
                ):
                    shot_dicts = retimed_shots
                    schedule = retimed_schedule
                    dialogue_violations = []
                    print(
                        "[ShortFilmPlanner] Deterministic H3 dialogue timing "
                        "repair reallocated shot duration without changing "
                        "the screenplay dialogue."
                    )

                normal_allocation_error: Optional[str] = None
                if dialogue_violations:
                    allocated_shots = copy.deepcopy(authoritative_shots)
                    try:
                        allocated_shots = _restore_h3_dialogue_after_pacing_repair(
                            authoritative_shots,
                            allocated_shots,
                            [frames / fps for frames in schedule],
                        )
                    except ValueError as error:
                        normal_allocation_error = str(error)
                    else:
                        allocated_violations = _h3_dialogue_budget_violations(
                            allocated_shots,
                            [frames / fps for frames in schedule],
                        )
                        if not allocated_violations:
                            shot_dicts = allocated_shots
                            dialogue_violations = []
                            print(
                                "[ShortFilmPlanner] Deterministic H3 dialogue "
                                "timing repair redistributed complete turns "
                                "across the existing shot schedule without "
                                "changing any words or speakers."
                            )

                extended_allocation_error: Optional[str] = None
                if dialogue_violations:
                    # Use maximum legal per-shot capacity only to find a valid
                    # semantic allocation, then immediately shrink every shot
                    # back to the smallest dialogue-safe bounded schedule.
                    allocated_shots = copy.deepcopy(authoritative_shots)
                    try:
                        allocated_shots = _restore_h3_dialogue_after_pacing_repair(
                            authoritative_shots,
                            allocated_shots,
                            [frames_maximum / fps] * len(allocated_shots),
                        )
                    except ValueError as error:
                        extended_allocation_error = str(error)
                    else:
                        allocated_schedule = _apply_native_schedule(
                            allocated_shots,
                            protect_dialogue=True,
                        )
                        allocated_violations = _h3_dialogue_budget_violations(
                            allocated_shots,
                            [frames / fps for frames in allocated_schedule],
                        )
                        if not allocated_violations:
                            shot_dicts = allocated_shots
                            schedule = allocated_schedule
                            dialogue_violations = []
                            added_seconds = max(
                                0.0,
                                (sum(schedule) - original_schedule_total) / fps,
                            )
                            print(
                                "[ShortFilmPlanner] Deterministic H3 dialogue "
                                "timing repair preserved every scripted word "
                                "and speaker"
                                + (
                                    f" by extending the plan {added_seconds:.2f}s."
                                    if added_seconds > 0.01 else "."
                                )
                            )

                if dialogue_violations:
                    remaining = "; ".join(
                        f"shot {item['index'] + 1}: {item['word_count']}/"
                        f"{item['word_budget']} words"
                        for item in dialogue_violations
                    )
                    allocation_details = next(
                        (
                            detail for detail in (
                                extended_allocation_error,
                                normal_allocation_error,
                            )
                            if detail
                        ),
                        "the exact dialogue could not be allocated safely",
                    )
                    raise RuntimeError(
                        "MiniMax H3 dialogue cannot fit the available legal "
                        "clip timing without changing scripted words ("
                        f"{remaining}; {allocation_details}). No video jobs "
                        "were queued."
                    )

        if not uses_generated_images and len(shot_dicts) > shot_count_low:
            original_shot_dicts = copy.deepcopy(shot_dicts)
            try:
                compacted_shots, merged_pairs = _coalesce_h3_dialogue_shots(
                    shot_dicts,
                    fps=fps,
                    minimum_frames=frames_minimum,
                    maximum_frames=frames_maximum,
                    frame_step=frames_steps,
                    minimum_shots=shot_count_low,
                    words_per_second=_H3_DIALOGUE_WORDS_PER_SECOND,
                )
                if merged_pairs:
                    compacted_schedule = _apply_native_schedule(
                        compacted_shots,
                        protect_dialogue=True,
                    )
                    compacted_violations = _h3_dialogue_budget_violations(
                        compacted_shots,
                        [frames / fps for frames in compacted_schedule],
                        words_per_second=_H3_DIALOGUE_WORDS_PER_SECOND,
                    )
                    if compacted_violations:
                        raise ValueError(
                            "a merged conversation exceeded its fitted H3 "
                            "dialogue timing"
                        )
                    _reconcile_h3_dialogue_manifest(
                        compacted_shots,
                        screenplay_dialogue_manifest,
                        known_items=original_shot_dicts,
                        allow_manifest_sentence_splits=True,
                    )
                    shot_dicts = compacted_shots
                    schedule = compacted_schedule
                    pair_text = ", ".join(
                        f"{first}+{second}" for first, second in merged_pairs
                    )
                    print(
                        "[ShortFilmPlanner] H3 conversation packing merged "
                        f"adjacent visual shots {pair_text}; "
                        f"{len(original_shot_dicts)} planned clips became "
                        f"{len(shot_dicts)} native conversation clips without "
                        "changing dialogue."
                    )
            except ValueError as error:
                print(
                    "[ShortFilmPlanner] H3 conversation packing kept the "
                    f"original visual edits ({error})."
                )

        print(
            "[ShortFilmPlanner] H3 shot plan verified: "
            f"{len(shot_dicts)} complete shot(s), "
            f"{len(screenplay_dialogue_manifest)} screenplay dialogue turn(s) "
            "preserved in semantic shot order."
        )
        shot_dicts = _enforce_h3_speaker_visual_contract(
            shot_dicts,
            character_voice_bible,
        )
        shot_dicts = _prepare_h3_prompt_only_continuity(shot_dicts)

        assert_no_minor_content(
            collect_pass2_text(shot_dicts), source="shot list (H3 native Pass 2)"
        )

        shots = self._convert_story_shots(
            shot_dicts,
            char_profiles,
            has_reference,
            fps,
            frames_steps,
            frames_minimum,
            frames_maximum=frames_maximum,
        )
        title = shot_dicts[0].get("title") if shot_dicts else None
        self._last_title = title
        return shots, title

    def _convert_story_shots(
        self,
        shot_dicts: list[dict],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        fps: int,
        frames_steps: int,
        frames_minimum: int,
        frames_maximum: Optional[int] = None,
    ) -> list[ShotPlan]:
        """Convert LLM output to ShotPlan objects for story-driven mode."""
        shots = []
        for i, raw in enumerate(shot_dicts):
            duration = raw.get("duration_sec", raw.get("duration", 15))

            # Snap duration to valid frame count
            raw_frames = int(round(duration * fps))
            if frames_maximum is not None:
                frames_maximum = max(frames_minimum, int(frames_maximum))
                lattice_index = round(
                    (raw_frames - frames_minimum) / max(1, frames_steps)
                )
                snapped = frames_minimum + max(0, lattice_index) * max(1, frames_steps)
                snapped = min(frames_maximum, snapped)
            else:
                snapped = max(
                    frames_minimum,
                    ((raw_frames - 1) // frames_steps) * frames_steps + 1,
                )
            duration = snapped / fps

            subjects = [SubjectRef.from_dict(s) if isinstance(s, dict) else SubjectRef(visual_description=str(s))
                        for s in raw.get("subjects_on_screen", [])]

            cam_raw = raw.get("camera_plan", {})
            camera = CameraPlan(
                framing=cam_raw.get("framing", "medium shot"),
                angle=cam_raw.get("angle"),
                movement=cam_raw.get("movement"),
                movement_intensity=cam_raw.get("movement_intensity", "subtle"),
            )

            audio_raw = raw.get("audio_plan", {})
            has_dialogue = bool(raw.get("dialogue_beats"))
            audio = AudioPlan(
                mode=audio_raw.get("mode", "dialogue_driven" if has_dialogue else "ambient_only"),
                ambience=audio_raw.get("ambience"),
                timing_anchor="audio" if has_dialogue else "video",
                lip_sync_critical=audio_raw.get("lip_sync_critical", has_dialogue),
            )

            is_h3_native = bool(
                frames_maximum is not None
                and str(getattr(self, "_video_model", "") or "")
                .lower().startswith("minimax_h3")
            )
            dialogue_beats = None
            if raw.get("dialogue_beats"):
                dialogue_beats = [DialogueBeat.from_dict(db) for db in raw["dialogue_beats"]]
                # Never truncate structured dialogue here. video_prompt was
                # already written from the same exact lines, so changing only
                # DialogueBeat.spoken_text creates split/nested <d> blocks and
                # turns H3 speech into gibberish. Native H3 planning performs
                # a whole-plan pacing repair before this conversion instead.
                if not is_h3_native:
                    word_budget = int(duration * 2.5)
                    total_words = sum(
                        len(db.spoken_text.split()) for db in dialogue_beats
                    )
                    if total_words > word_budget * 1.5:
                        for db in dialogue_beats:
                            words = db.spoken_text.split()
                            max_words = max(
                                3,
                                int(len(words) * word_budget / total_words),
                            )
                            db.spoken_text = " ".join(words[:max_words])

            vocal_contract = ""
            if is_h3_native:
                raw["video_prompt"], vocal_contract = _inject_h3_vocal_contract(
                    raw.get("video_prompt", ""),
                    subjects,
                    dialogue_beats or [],
                )

            continuity_strategy = str(
                raw.get("continuity_strategy")
                or ("continuous" if i > 0 else "independent")
            ).strip().lower()
            if continuity_strategy not in VALID_CONTINUITY_STRATEGIES:
                continuity_strategy = "continuous" if i > 0 else "independent"

            shot = ShotPlan(
                shot_id=self._make_shot_id(i, "sf"),
                index=i,
                duration_sec=duration,
                skill_type="short_film",
                scene_goal=raw.get("scene_goal", f"Scene {i + 1}"),
                narrative_role=raw.get("narrative_role"),
                scene_type=raw.get("scene_type", "dialogue" if has_dialogue else "action"),
                source_mode_preference="i2v" if has_reference else "t2v",
                image_strategy="reference_edit" if has_reference else "fresh_generation",
                continuity_strategy=continuity_strategy,
                subjects_on_screen=subjects,
                spatial_setup=raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                visual_style=raw.get("visual_style", ""),
                lighting=raw.get("lighting", ""),
                mood=raw.get("mood", ""),
                action_beats=raw.get("action_beats", []),
                dialogue_beats=dialogue_beats,
                camera_plan=camera,
                audio_plan=audio,
                ending_beat=raw.get("ending_beat", ""),
                metadata={
                    "title": raw.get("title", ""),
                    "duration_frames": snapped,
                    "continuity_group": raw.get("continuity_group", ""),
                    "closing_blocking": raw.get(
                        "closing_blocking", raw.get("ending_beat", "")
                    ),
                    "vocal_contract": vocal_contract,
                },
                # LLM-generated prompts (used directly, skipping renderer pass 2)
                video_prompt=raw.get("video_prompt"),
                image_prompt=raw.get("image_prompt"),
                window_prompts=raw.get("window_prompts"),
                visual_changes=raw.get("visual_changes"),
                image_source=raw.get("image_source"),
                keyframe_prompts=raw.get("keyframe_prompts"),
            )
            shots.append(shot)

        return shots

    # ── Single-Pass Fallback ─────────────────────────────────────────

    def _plan_story_single_pass(
        self,
        story_description: str,
        reference_image_path: Optional[str],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        target_duration: int,
        target_scenes: Optional[int],
        narrative_mode: bool,
        fps: int,
        frames_steps: int,
        frames_minimum: int,
        nsfw: bool = False,
        polish_block: str = "",
    ) -> tuple[list[ShotPlan], Optional[str]]:
        """Fallback single-pass planning if the screenplay pass fails."""
        from ..nsfw_guidance import inject_nsfw_if_enabled

        if target_scenes is None:
            target_scenes = max(2, min(20, target_duration // 20))

        preserve_names = bool(
            getattr(self, "_preserve_video_character_names", False)
        )
        uses_generated_images = bool(
            getattr(self, "_uses_generated_shot_images", True)
        )
        char_rules = build_character_rules_block(
            has_reference or bool(getattr(self, "_num_character_refs", 0)),
            char_profiles if char_profiles else None,
            preserve_names=preserve_names,
        )
        video_guide = _route_video_pass2_guide(
            getattr(self, "_video_model", "") or ""
        )
        video_name_rules = _video_character_name_rules(preserve_names)
        visual_strategy_rules = (
            "No generated start frame will be supplied. Make every video_prompt "
            "self-contained with the complete visible scene and synchronized "
            "sound. Do not create image_prompt, image_source, visual_changes, "
            "or keyframe_prompts."
            if not uses_generated_images else ""
        )
        fallback_output_fields = (
            "title, duration_sec, scene_goal, video_prompt, image_prompt"
            if uses_generated_images
            else "title, duration_sec, scene_goal, video_prompt"
        )
        fallback_image_rule = (
            "- image_prompt is the FIRST FRAME BEFORE action begins — initial "
            "state, static poses, zero motion verbs. If something changes in "
            "the scene, the image shows the BEFORE state."
            if uses_generated_images
            else "- Omit every still-image and keyframe field."
        )

        system_prompt = f"""You are a short film director. Create a scene plan. Output ONLY the JSON array.

{f"You are given a REFERENCE PHOTO." if has_reference else ""}

{char_rules}

{video_name_rules}

{visual_strategy_rules}

{video_guide}

- Total duration must sum to ~{target_duration}s. YOU decide how many scenes based on the story.
- KEEP CONVERSATIONS TOGETHER — do not split dialogue across multiple shots. One conversation = one shot.
- Only cut when the location changes or a clear story beat transition happens.
- Prefer 20-40s shots. Shots over 20s need window_prompts.
- Output ONLY a JSON array with {fallback_output_fields} per scene.
{fallback_image_rule}


Go:"""

        if polish_block:
            system_prompt = f"{system_prompt}\n\n{polish_block}"
        system_prompt = inject_nsfw_if_enabled(
            system_prompt,
            nsfw,
            "both" if uses_generated_images else "video",
        )

        # Single-pass fallback also gets the safety scan — it bypasses
        # Pass 1 entirely, so the post-Pass-1 scan above doesn't run for
        # this code path. Mirror the same hybrid co-occurrence check on
        # the user's concept (pre-call) and on the structured shot list
        # (post-call).
        from ..safety_scan import (
            assert_no_minor_content,
            collect_pass2_text,
        )
        assert_no_minor_content(story_description, source="user concept")

        image_paths = self._build_all_image_paths(reference_image_path, has_reference)
        # Grammar constraint — this path runs with thinking_budget=4096, so
        # the schema only fires on the parse-failure retry (see
        # _call_llm_json). The fallback spec asks for just five fields; the
        # rest of _SHOT_PROPERTIES stays available but optional. +2 slack
        # on maxItems since the prompt lets the LLM choose the scene count.
        fallback_schema = _shot_list_schema(
            min_items=2,
            max_items=max(4, target_scenes + 2),
            required=["title", "duration_sec", "scene_goal", "image_prompt", "video_prompt"],
            include_image_fields=uses_generated_images,
        )
        shot_dicts = self._call_llm_json(
            user_prompt=f"Story: {story_description}",
            system_prompt=system_prompt,
            max_tokens=max(4096, target_duration * 60),
            thinking_budget=4096,
            image_paths=image_paths,
            json_schema=fallback_schema,
        )
        if not uses_generated_images:
            _discard_unused_image_fields(shot_dicts)

        assert_no_minor_content(
            collect_pass2_text(shot_dicts), source="shot list (single-pass fallback)"
        )

        seen_goals = set()
        unique_dicts = []
        for sd in shot_dicts:
            goal = sd.get("scene_goal", sd.get("title", ""))
            if goal not in seen_goals:
                seen_goals.add(goal)
                unique_dicts.append(sd)

        shots = self._convert_story_shots(unique_dicts, char_profiles, has_reference, fps, frames_steps, frames_minimum)
        title = shot_dicts[0].get("title") if shot_dicts else None
        self._last_title = title
        return shots, title
