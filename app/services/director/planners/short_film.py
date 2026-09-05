"""
Short Film Planner — creates a ProductionPlan from story/audio inputs.

Supports two paths:
  1. Audio-driven: dialogue audio + transcript → scene plans
  2. Story-driven: story description + characters → scene plans (no audio)

Outputs: ProductionPlan with ShotPlan objects (NOT final prompts).
"""

from __future__ import annotations
import copy
import difflib
import json
import math
import os
import re
from typing import Optional, Any, Sequence

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
from ..long_form_story import (
    LONG_FORM_STORY_BIBLE_REVISION,
    LONG_FORM_STORY_BIBLE_SCHEMA,
    apply_long_form_recurring_motifs,
    audit_long_form_plan,
    ensure_long_form_location_coverage,
    format_long_form_story_bible,
    long_form_outline_quality_issues,
    long_form_story_bible_quality_issues,
    normalize_long_form_outline,
    normalize_long_form_sequence_states,
    normalize_long_form_story_bible,
    place_chapter_motifs_in_sequences,
    resolve_locked_dialogue_speakers,
    sanitize_long_form_shot_dicts,
)
from .base import BasePlanner
from services.text_integrity import repair_text


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
    "story_scene_number": {"type": "integer"},
    "causal_handoff": {"type": "string"},
    "persistent_story_state": {"type": "string"},
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

# Dialogue polishing is deliberately split into compact, scene-sized batches.
# Large H3 screenplays can contain 40+ turns; asking a thinking model to return
# every row in one response caused the final JSON array to truncate and made
# Maestro discard an otherwise useful table read in its entirety.
_H3_TABLE_READ_CHUNK_SIZE = 12

# Qwen3.8's creative screenplay pass can use substantially more reasoning than
# its eventual answer. The allowance is additive to the screenplay's content
# cap, but llama.cpp does not reserve the latter: if reasoning consumes the
# entire combined limit, content is empty. Keep the proven creative headroom
# and pair it with the explicit recovery path below rather than silently
# planning shots from an absent screenplay.
_H3_SCREENPLAY_THINKING_BUDGET_SHORT = 16384
_H3_SCREENPLAY_THINKING_BUDGET_MEDIUM = 24576
_H3_SCREENPLAY_THINKING_BUDGET_LONG = 32768
_H3_SCREENPLAY_MIN_CHARS = 50

# Long-form Director uses a three-level plan: complete-film outline, bounded
# chapters, then screenplay-sized sequences.  Five minutes remains a useful
# story/continuity chapter, but it is too large for one reliable local-model
# screenplay response.  Ninety seconds keeps Qwen/Gemma creative calls small
# while still giving each sequence room for several native video shots.
_DIRECTOR_LONG_FORM_CHAPTER_SECONDS = 300
_DIRECTOR_LONG_FORM_SEQUENCE_SECONDS = 90
_DIRECTOR_LONG_FORM_PLAN_REVISION = 4

_LONG_FORM_PLAN_TEXT_FIELDS = (
    "title",
    "location_time",
    "objective",
    "opening_state",
    "inherited_state",
    "closing_state",
    "causal_handoff",
    "persistent_state",
    "motif_variation_contract",
)
_LONG_FORM_DIALOGUE_MODES = frozenset({
    "silent",
    "visual",
    "sparse",
    "natural",
    "dialogue_forward",
})
_LONG_FORM_SPEECH_CUE_RE = re.compile(
    r"\b(?:asks?|answers?|argues?|banter|conversation|debates?|dialogue|"
    r"explains?|interrogates?|monologue|replies?|responds?|says?|speaks?|"
    r"tells?|yells?|whispers?)\b",
    flags=re.IGNORECASE,
)


def _h3_screenplay_thinking_budget(target_duration: float) -> int:
    """Scale creative reasoning headroom without penalizing short films."""

    try:
        duration = max(0.0, float(target_duration or 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 120.0:
        return _H3_SCREENPLAY_THINKING_BUDGET_SHORT
    if duration <= 180.0:
        return _H3_SCREENPLAY_THINKING_BUDGET_MEDIUM
    return _H3_SCREENPLAY_THINKING_BUDGET_LONG


def _bounded_long_form_durations(
    total_seconds: int,
    *,
    maximum_seconds: int = _DIRECTOR_LONG_FORM_SEQUENCE_SECONDS,
) -> list[int]:
    """Split a duration into even, integer, bounded planning sequences."""

    total = max(1, int(total_seconds or 1))
    maximum = max(1, int(maximum_seconds or 1))
    count = max(1, math.ceil(total / maximum))
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _redact_long_form_dialogue(
    story_description: str,
    locked_dialogue: list[dict[str, Any]],
) -> str:
    """Replace literal dialogue with stable IDs in shared chapter context.

    The complete concept is supplied to every chapter for global fidelity.
    Leaving quotes in that shared text made each chapter's local validator
    demand every line in the film.  Placeholders retain chronology without
    leaking the literal line into chapters that do not own it.
    """

    redacted = str(story_description or "")
    for line in sorted(
        locked_dialogue,
        key=lambda item: int(item.get("source_offset") or 0),
        reverse=True,
    ):
        try:
            start = int(line.get("source_offset"))
            end = int(line.get("source_end"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > len(redacted):
            continue
        dialogue_id = str(line.get("dialogue_id") or "D?").upper()
        redacted = (
            redacted[:start]
            + f"[{dialogue_id}: exact dialogue assigned by Maestro]"
            + redacted[end:]
        )
    return repair_text(redacted)


def _normalize_long_form_dialogue_ownership(
    rows: list[dict[str, Any]],
    *,
    allowed_dialogue: list[dict[str, Any]],
    source_length: int = 0,
    source_events: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Make each locked dialogue ID belong to exactly one ordered row.

    The model may propose ownership, but it cannot reorder the immutable user
    dialogue.  Proposed positions are monotonically clamped in source order;
    missing IDs receive a deterministic position.
    """

    normalized = [dict(row) for row in rows]
    if not normalized:
        return normalized
    allowed_ids = [
        str(item.get("dialogue_id") or "").upper()
        for item in allowed_dialogue
        if item.get("dialogue_id")
    ]
    allowed_set = set(allowed_ids)
    proposed_owner: dict[str, int] = {}
    event_owner: dict[str, int] = {}
    for row_index, row in enumerate(normalized):
        for raw_id in row.get("source_event_ids") or []:
            event_id = str(raw_id or "").upper()
            if event_id:
                event_owner.setdefault(event_id, row_index)
        for raw_id in row.get("dialogue_ids") or []:
            dialogue_id = str(raw_id or "").upper()
            if dialogue_id in allowed_set:
                proposed_owner.setdefault(dialogue_id, row_index)
        row["dialogue_ids"] = []

    # A speech event and its exact words are one semantic unit. The redacted
    # source event retains a [D#: ...] marker, so use its normalized owner as
    # the binding owner for that dialogue ID. Without this link an architect
    # could assign "Thanos says [D1]" to one chapter and D1's words to another.
    linked_owner: dict[str, int] = {}
    for event in source_events or []:
        event_id = str(event.get("event_id") or "").upper()
        owner = event_owner.get(event_id)
        if owner is None:
            continue
        for dialogue_id in re.findall(
            r"\[(D\d+)\s*:",
            str(event.get("text") or ""),
            flags=re.IGNORECASE,
        ):
            normalized_id = dialogue_id.upper()
            if normalized_id in allowed_set:
                linked_owner.setdefault(normalized_id, owner)

    allowed_order = {
        dialogue_id: index for index, dialogue_id in enumerate(allowed_ids)
    }
    last_owner = 0
    for item in allowed_dialogue:
        dialogue_id = str(item.get("dialogue_id") or "").upper()
        if dialogue_id in linked_owner:
            owner = linked_owner[dialogue_id]
        elif dialogue_id in proposed_owner:
            owner = proposed_owner[dialogue_id]
        elif source_length > 0:
            try:
                ratio = max(0.0, min(
                    0.999999,
                    float(item.get("source_offset") or 0) / source_length,
                ))
                owner = int(ratio * len(normalized))
            except (TypeError, ValueError, ZeroDivisionError):
                owner = 0
        else:
            rank = allowed_order.get(dialogue_id, 0)
            owner = int(
                ((rank + 1) * len(normalized)) / (len(allowed_ids) + 1)
            )
        owner = max(last_owner, min(len(normalized) - 1, owner))
        normalized[owner].setdefault("dialogue_ids", []).append(dialogue_id)
        last_owner = owner
    return normalized


def _normalize_long_form_event_ownership(
    rows: list[dict[str, Any]],
    *,
    source_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign every ordered source event to one and only one plan row."""

    normalized = [dict(row) for row in rows]
    if not normalized:
        return normalized
    event_ids = [
        str(item.get("event_id") or "").upper()
        for item in source_events
        if item.get("event_id")
    ]
    allowed = set(event_ids)
    proposed_owner: dict[str, int] = {}
    for row_index, row in enumerate(normalized):
        for raw_id in row.get("source_event_ids") or []:
            event_id = str(raw_id or "").upper()
            if event_id in allowed:
                proposed_owner.setdefault(event_id, row_index)
        row["source_event_ids"] = []

    # Treat the architect's placement as a creative preference, not permission
    # to dump the complete source story into the final chapter.  A 60-minute
    # Gemma plan legally assigned five of six source events to chapter 12,
    # leaving the first fifty-five minutes to invent unrelated filler.  Keep
    # proposals inside a generous band around source order so sparse concepts
    # still expand across the available runtime without becoming a rigid beat
    # grid.
    placement_band = max(1, int(math.ceil(len(normalized) / 6.0)))
    last_owner = 0
    for event_index, event_id in enumerate(event_ids):
        expected = (
            0
            if len(event_ids) <= 1 else
            int(round(
                event_index * (len(normalized) - 1)
                / (len(event_ids) - 1)
            ))
        )
        proposed = proposed_owner.get(event_id)
        if proposed is None:
            proposed = expected
        elif len(normalized) >= 4 and len(event_ids) >= 2:
            proposed = max(
                expected - placement_band,
                min(expected + placement_band, proposed),
            )
        owner = max(last_owner, min(len(normalized) - 1, proposed))
        normalized[owner].setdefault("source_event_ids", []).append(event_id)
        last_owner = owner
    return normalized


def _normalize_long_form_plan_references(
    rows: list[dict[str, Any]],
    *,
    allowed_dialogue: list[dict[str, Any]],
    source_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove registry references from rows that do not own them.

    The ID arrays are authoritative.  Local models sometimes assign ``D1`` to
    one chapter while writing "after delivering D1" in several other chapter
    summaries.  Those prose leaks cause later screenplay calls to perform the
    supposedly absent line.  Preserve an ID or its literal words only inside
    the row that owns it; replace an out-of-scope reference with a neutral
    continuity phrase that cannot be mistaken for an instruction to speak.
    """

    normalized = [dict(row) for row in rows]
    dialogue_by_id = {
        str(item.get("dialogue_id") or "").upper(): str(
            item.get("text") or item.get("exact_text") or ""
        ).strip()
        for item in allowed_dialogue or []
        if str(item.get("dialogue_id") or "").strip()
    }
    event_ids = {
        str(item.get("event_id") or "").upper()
        for item in source_events or []
        if str(item.get("event_id") or "").strip()
    }

    for row in normalized:
        owned_dialogue = {
            str(value or "").upper()
            for value in row.get("dialogue_ids") or []
        }
        owned_events = {
            str(value or "").upper()
            for value in row.get("source_event_ids") or []
        }
        for field in _LONG_FORM_PLAN_TEXT_FIELDS:
            value = str(row.get(field) or "")
            if not value:
                continue

            def replace_dialogue_marker(match: re.Match[str]) -> str:
                dialogue_id = str(
                    match.group(1) or match.group(2) or ""
                ).upper()
                return (
                    match.group(0)
                    if dialogue_id in owned_dialogue else
                    "the established story state"
                )

            value = re.sub(
                r"\[(D\d+)\s*:[^\]]*\]|\b(D\d+)\b",
                replace_dialogue_marker,
                value,
                flags=re.IGNORECASE,
            )
            for dialogue_id, exact_text in dialogue_by_id.items():
                if dialogue_id in owned_dialogue or not exact_text:
                    continue
                value = re.sub(
                    re.escape(exact_text),
                    "the established story state",
                    value,
                    flags=re.IGNORECASE,
                )

            def replace_event_marker(match: re.Match[str]) -> str:
                event_id = str(
                    match.group(1) or match.group(2) or ""
                ).upper()
                return (
                    match.group(0)
                    if event_id in owned_events else
                    "the established story progression"
                )

            value = re.sub(
                r"\[(E\d+)\s*:[^\]]*\]|\b(E\d+)\b",
                replace_event_marker,
                value,
                flags=re.IGNORECASE,
            )
            # Only IDs from the immutable registry are meaningful.  Removing
            # hallucinated IDs as well prevents them from becoming new local
            # obligations in the next planner pass.
            value = re.sub(
                r"\b(?:D|E)\d+\b",
                "the established story beat",
                value,
                flags=re.IGNORECASE,
            ) if not (owned_dialogue or owned_events) else value
            row[field] = re.sub(r"\s+", " ", value).strip()
    return normalized


def _normalize_long_form_dialogue_targets(
    rows: list[dict[str, Any]],
    *,
    durations: Sequence[int | float],
    allowed_dialogue: list[dict[str, Any]],
    source_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Give every bounded sequence an explicit, safe dialogue target.

    This value is the sole source of long-form dialogue density.  Internal
    production-contract prose is deliberately excluded, so words such as
    "dialogue manifest" can never turn a silent establishing sequence into a
    forced ten-turn exchange.
    """

    normalized = [dict(row) for row in rows]
    dialogue_by_id = {
        str(item.get("dialogue_id") or "").upper(): item
        for item in allowed_dialogue or []
    }
    event_by_id = {
        str(item.get("event_id") or "").upper(): item
        for item in source_events or []
    }
    for index, row in enumerate(normalized):
        try:
            duration = max(1.0, float(durations[index]))
        except (IndexError, TypeError, ValueError):
            duration = 60.0
        owned_dialogue = [
            dialogue_by_id[str(value or "").upper()]
            for value in row.get("dialogue_ids") or []
            if str(value or "").upper() in dialogue_by_id
        ]
        owned_events = [
            event_by_id[str(value or "").upper()]
            for value in row.get("source_event_ids") or []
            if str(value or "").upper() in event_by_id
        ]
        local_intent = " ".join([
            *(str(row.get(field) or "") for field in _LONG_FORM_PLAN_TEXT_FIELDS),
            *(str(item.get("text") or "") for item in owned_events),
        ])
        requested_mode = str(row.get("dialogue_mode") or "").strip().casefold()
        requested_mode = requested_mode.replace("-", "_").replace(" ", "_")
        if requested_mode not in _LONG_FORM_DIALOGUE_MODES:
            if _H3_SILENT_STORY_RE.search(local_intent):
                requested_mode = "silent"
            elif _H3_DIALOGUE_FORWARD_RE.search(local_intent):
                requested_mode = "dialogue_forward"
            elif _LONG_FORM_SPEECH_CUE_RE.search(local_intent):
                requested_mode = "natural"
            else:
                requested_mode = "visual"
        if owned_dialogue and requested_mode == "silent":
            requested_mode = "natural"

        defaults = {
            "silent": (0, 0),
            "visual": (0, 0),
            "sparse": (
                max(1, int(round(duration / 30.0))),
                max(6, int(round(duration * 0.20))),
            ),
            "natural": (
                max(2, int(round(duration / 15.0))),
                max(12, int(round(duration * 0.50))),
            ),
            "dialogue_forward": (
                max(3, int(math.ceil(duration / 8.0))),
                max(18, int(round(duration * 0.75))),
            ),
        }
        default_turns, default_words = defaults[requested_mode]
        locked_words = sum(
            len(str(item.get("text") or item.get("exact_text") or "").split())
            for item in owned_dialogue
        )
        hard_turn_cap = max(1, int(math.ceil(duration / 6.0)))
        hard_word_cap = max(1, int(math.floor(duration * 1.65)))
        try:
            requested_turns = int(row.get("dialogue_target_turns"))
        except (TypeError, ValueError):
            requested_turns = default_turns
        try:
            requested_words = int(row.get("dialogue_target_words"))
        except (TypeError, ValueError):
            requested_words = default_words
        if requested_mode in {"silent", "visual"} and not owned_dialogue:
            requested_turns = 0
            requested_words = 0
        row["dialogue_mode"] = requested_mode
        row["dialogue_target_turns"] = max(
            len(owned_dialogue),
            min(hard_turn_cap, max(0, requested_turns)),
        )
        row["dialogue_target_words"] = max(
            locked_words,
            min(hard_word_cap, max(0, requested_words)),
        )
    return normalized


def _format_long_form_source_events(
    event_ids: Sequence[str],
    source_events: list[dict[str, Any]],
) -> str:
    event_map = {
        str(item.get("event_id") or "").upper(): str(
            item.get("text") or ""
        ).strip()
        for item in source_events
    }
    lines = [
        f"{str(event_id).upper()}: {event_map[str(event_id).upper()]}"
        for event_id in event_ids or []
        if event_map.get(str(event_id).upper())
    ]
    return "\n".join(lines) or "No explicit source event is owned here; advance the binding chapter objective."


def _format_long_form_plan_row(value: Optional[dict[str, Any]]) -> str:
    """Render a chapter/sequence contract without JSON quote noise."""

    if not value:
        return "None."
    labels = (
        ("title", "Title"),
        ("location_id", "Location ID"),
        ("location_time", "Location/time"),
        ("objective", "Objective"),
        ("opening_state", "Opening state"),
        ("inherited_state", "Inherited state"),
        ("inherited_character_state", "Inherited character state"),
        ("closing_state", "Closing state"),
        ("causal_handoff", "Causal handoff"),
        ("persistent_state", "Persistent state"),
        ("motif_variation_contract", "Motif variation"),
    )
    lines = [
        f"{label}: {str(value.get(key) or '').strip()}"
        for key, label in labels
        if str(value.get(key) or "").strip()
    ]
    event_ids = [
        str(item or "").upper()
        for item in value.get("source_event_ids") or []
        if str(item or "").strip()
    ]
    dialogue_ids = [
        str(item or "").upper()
        for item in value.get("dialogue_ids") or []
        if str(item or "").strip()
    ]
    if event_ids:
        lines.append("Owned source events: " + ", ".join(event_ids))
    if dialogue_ids:
        lines.append("Owned dialogue: " + ", ".join(dialogue_ids))
    motif_ids = [
        str(item or "").upper()
        for item in value.get("recurring_motif_ids") or []
        if str(item or "").strip()
    ]
    cast_present = [
        str(item or "").strip()
        for item in value.get("cast_present") or []
        if str(item or "").strip()
    ]
    if motif_ids:
        lines.append("Active recurring motifs: " + ", ".join(motif_ids))
    if cast_present:
        lines.append("Cast present: " + ", ".join(cast_present))
    state_changes = [
        str(item or "").strip()
        for item in value.get("character_state_changes") or []
        if str(item or "").strip()
    ]
    if state_changes:
        lines.append("Named character state changes: " + "; ".join(state_changes))
    return "\n".join(lines) or "None."


def _format_long_form_locked_dialogue(
    dialogue_ids: Sequence[str],
    locked_dialogue: list[dict[str, Any]],
) -> str:
    by_id = {
        str(item.get("dialogue_id") or "").upper(): item
        for item in locked_dialogue
    }
    lines: list[str] = []
    for raw_id in dialogue_ids or []:
        dialogue_id = str(raw_id or "").upper()
        item = by_id.get(dialogue_id)
        if not item:
            continue
        speaker = str(item.get("speaker") or "Speaker").strip()
        spoken = str(item.get("text") or "").replace('"', "'").strip()
        delivery = str(item.get("delivery") or "speaks naturally").strip()
        lines.append(
            f'{dialogue_id}: {speaker} {delivery} and says "{spoken}"'
        )
    return "\n".join(lines) or "None in this sequence."


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
    "maxItems": 40,
}


_STORY_CONTINUITY_FIELDS = (
    "scene_number",
    "location_time",
    "active_objective",
    "story_purpose",
    "opening_cause",
    "visible_beats",
    "choice_or_discovery",
    "outgoing_handoff",
    "persistent_state_after",
)


def _story_continuity_blueprint_schema(
    minimum_scenes: int,
    maximum_scenes: int,
) -> dict:
    """Closed schema for Director's causal scene-chain planning pass."""

    minimum_scenes = max(2, int(minimum_scenes or 2))
    maximum_scenes = max(minimum_scenes, int(maximum_scenes or minimum_scenes))
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "scene_number": {"type": "integer"},
                "location_time": {"type": "string"},
                "active_objective": {"type": "string"},
                "story_purpose": {"type": "string"},
                "opening_cause": {"type": "string"},
                "visible_beats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                },
                "choice_or_discovery": {"type": "string"},
                "outgoing_handoff": {"type": "string"},
                "persistent_state_after": {"type": "string"},
            },
            "required": list(_STORY_CONTINUITY_FIELDS),
            "additionalProperties": False,
        },
        "minItems": minimum_scenes,
        "maxItems": maximum_scenes,
    }


def _normalize_story_continuity_blueprint(
    rows: Any,
    *,
    minimum_scenes: int,
    maximum_scenes: int,
) -> list[dict[str, Any]]:
    """Accept only a complete, concrete causal scene chain.

    The grammar normally guarantees this shape. The explicit validation also
    protects Gemma's thinking-on path and remote providers that cannot enforce
    a local llama.cpp JSON grammar.
    """

    minimum_scenes = max(2, int(minimum_scenes or 2))
    maximum_scenes = max(minimum_scenes, int(maximum_scenes or minimum_scenes))
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for expected_number, raw in enumerate(rows[:maximum_scenes], start=1):
        if not isinstance(raw, dict):
            return []
        scene: dict[str, Any] = {"scene_number": expected_number}
        for field in _STORY_CONTINUITY_FIELDS:
            if field in {"scene_number", "visible_beats"}:
                continue
            value = re.sub(r"\s+", " ", str(raw.get(field) or "")).strip()
            if not value:
                return []
            scene[field] = value
        visible_beats = [
            re.sub(r"\s+", " ", str(beat or "")).strip()
            for beat in (raw.get("visible_beats") or [])
        ]
        visible_beats = [beat for beat in visible_beats if beat]
        if not visible_beats:
            return []
        scene["visible_beats"] = visible_beats[:8]
        normalized.append(scene)
    if len(normalized) < minimum_scenes:
        return []
    return normalized


def _format_story_continuity_blueprint(rows: list[dict[str, Any]]) -> str:
    """Render the architect pass as compact binding context for later passes."""

    if not rows:
        return "(No separate story-architect blueprint was available.)"
    return json.dumps(rows, ensure_ascii=False, indent=2)


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


def _repair_h3_screenplay_speaker_headings(
    screenplay: Any,
    canonical_names: Sequence[str],
) -> tuple[str, list[tuple[str, str]]]:
    """Correct an evident misspelled heading before dialogue becomes locked.

    The screenwriter occasionally emits a near-name such as ``THORNS`` while
    the surrounding action still says ``Thanos``.  If that typo reaches Pass
    2 it becomes a new visible person.  Repairs require either a strong unique
    spelling match or a moderate match to the one canonical person named in
    the immediately preceding action paragraph.  Uncertain headings remain
    untouched so an original supporting character cannot be silently merged.
    """

    text = _normalize_h3_text(screenplay)
    names: list[str] = []
    seen: set[str] = set()
    for raw_name in canonical_names or []:
        name = re.sub(r"\s+", " ", str(raw_name or "")).strip(" .")
        key = " ".join(_h3_speaker_name_tokens(name))
        if not key or key in seen or key in {
            "he", "her", "him", "narration", "she", "speaker", "they",
            "them", "voice", "voiceover",
        }:
            continue
        seen.add(key)
        names.append(name)
    if not text or not names:
        return text, []

    canonical_keys = {
        " ".join(_h3_speaker_name_tokens(name)): name for name in names
    }
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    learned: dict[str, str] = {}
    repairs: list[tuple[str, str]] = []

    def previous_paragraph(index: int) -> str:
        cursor = index - 1
        while cursor >= 0 and not lines[cursor].strip():
            cursor -= 1
        block: list[str] = []
        while cursor >= 0 and lines[cursor].strip():
            # Do not treat a preceding dialogue block as action evidence.
            if _h3_screenplay_speaker_heading(lines[cursor]):
                break
            block.append(lines[cursor].strip())
            cursor -= 1
        return " ".join(reversed(block))

    for index, raw_line in enumerate(lines):
        heading = _h3_screenplay_speaker_heading(raw_line)
        if not heading:
            continue
        supplied_name, _centered = heading
        supplied_key = " ".join(_h3_speaker_name_tokens(supplied_name))
        if not supplied_key or supplied_key in canonical_keys:
            continue
        target = learned.get(supplied_key, "")
        ratios = sorted(
            (
                difflib.SequenceMatcher(None, supplied_key, key).ratio(),
                name,
            )
            for key, name in canonical_keys.items()
        )
        ratios.reverse()
        if not target and ratios:
            best_ratio, best_name = ratios[0]
            runner_up = ratios[1][0] if len(ratios) > 1 else 0.0
            if best_ratio >= 0.78 and best_ratio - runner_up >= 0.08:
                target = best_name
            elif best_ratio >= 0.62 and best_ratio - runner_up >= 0.08:
                context = previous_paragraph(index)
                mentioned = [
                    name for name in names
                    if re.search(
                        rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
                        context,
                        flags=re.IGNORECASE,
                    )
                ]
                if len(mentioned) == 1 and mentioned[0] == best_name:
                    target = best_name
        if not target:
            continue
        learned[supplied_key] = target
        repairs.append((supplied_name, target))
        stripped = raw_line.strip()
        replacement = target.upper()
        if re.fullmatch(
            r"<center>\s*[^<\r\n]+?\s*</center>",
            stripped,
            flags=re.IGNORECASE,
        ):
            lines[index] = f"<center>{replacement}</center>"
        elif re.fullmatch(r"\*\*\s*[^*\r\n]+?\s*\*\*", stripped):
            lines[index] = f"**{replacement}**"
        else:
            continuation = re.search(
                r"\s*(\((?:CONT['’]?D|V\.?O\.?|O\.?S\.?)\))\s*$",
                stripped,
                flags=re.IGNORECASE,
            )
            lines[index] = (
                f"{replacement} {continuation.group(1)}"
                if continuation else replacement
            )

    return "\n".join(lines), list(dict.fromkeys(repairs))


def _h3_dialogue_word_fingerprint(value: Any) -> tuple[str, ...]:
    """Compare spoken words while ignoring punctuation and Markdown emphasis."""

    text = _h3_plain_dialogue_text(value).casefold()
    text = text.replace("’", "'").replace("‘", "'")
    return tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text, flags=re.UNICODE))


def _h3_speaker_name_tokens(value: Any) -> tuple[str, ...]:
    text = _normalize_h3_text(value).casefold()
    return tuple(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


_H3_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|\d+)"
)
_H3_FIXED_WORD_RANGE_RE = re.compile(
    rf"\b(?:(?:often|usually|typically|generally|mostly)\s+)?"
    rf"(?:only\s+)?{_H3_NUMBER_WORD}\s*(?:-|–|—|to|through)\s*"
    rf"{_H3_NUMBER_WORD}\s+words?\b",
    flags=re.IGNORECASE,
)
_H3_FIXED_WORD_CEILING_RE = re.compile(
    rf"\b(?:(?:never|no|not)\s+(?:more\s+than|over)|under|at\s+most)\s+"
    rf"{_H3_NUMBER_WORD}\s+words?\b",
    flags=re.IGNORECASE,
)
_H3_MICRO_REPLY_RE = re.compile(
    r"\b(?:single|one|two)[ -]word\s+"
    r"(?:lines?|sentences?|answers?|responses?|replies?)\b",
    flags=re.IGNORECASE,
)

# A table read is allowed to make dialogue more character-specific, but it
# must not behave like a thesaurus pass that upgrades ordinary speech into
# consultant, academic, or production-note language.  This is deliberately a
# *relative* guard: a character who already speaks formally keeps that voice;
# Maestro only rejects a revision that introduces several new elevated
# markers which were absent from the screenplay line it was meant to polish.
_H3_ELEVATED_DIALOGUE_RE = re.compile(
    r"\b(?:adequate|ascertain|commence|consequently|endeavo(?:u)?r|entropy|"
    r"facilitate|fortitude|fundamentally|henceforth|imperative|inherently|"
    r"localized|methodology|nevertheless|operational|optimal|parameters?|"
    r"possess|precisely|subsequent|terminate|utilize|volatile)\b",
    flags=re.IGNORECASE,
)
_H3_ELEVATED_DIALOGUE_PHRASES = (
    "core issue",
    "stable foundation",
    "structural fortitude",
    "visual appeal",
    "i grant you",
    "one could argue",
    "it is simply",
)


def _h3_elevated_dialogue_score(value: Any) -> int:
    """Count conspicuously formal markers in one spoken line."""

    text = _h3_plain_dialogue_text(value).casefold()
    return (
        len(_H3_ELEVATED_DIALOGUE_RE.findall(text))
        + sum(text.count(phrase) for phrase in _H3_ELEVATED_DIALOGUE_PHRASES)
    )


def _h3_table_read_formalization_regressed(
    original_text: Any,
    candidate_text: Any,
) -> bool:
    """Return True when a dialogue polish newly formalizes plain speech."""

    return (
        _h3_elevated_dialogue_score(candidate_text)
        - _h3_elevated_dialogue_score(original_text)
        >= 2
    )


def _sanitize_h3_voice_guidance(value: Any) -> str:
    """Remove fixed micro-line prescriptions from character guidance.

    A voice profile should constrain diction, syntax, subtext, and rhythm. It
    must not turn a concise character into a screenplay made entirely from
    one- and two-word fragments. This also protects saved/generated profiles
    produced by older prompt wording.
    """

    text = re.sub(
        r"\s+",
        " ",
        _normalize_h3_text(value or ""),
    ).strip(" .")
    if not text:
        return ""
    text = _H3_FIXED_WORD_RANGE_RE.sub(
        "with line length varied to the dramatic beat",
        text,
    )
    text = _H3_FIXED_WORD_CEILING_RE.sub(
        "without a fixed word-count ceiling",
        text,
    )
    text = _H3_MICRO_REPLY_RE.sub("occasional clipped replies", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,;])\s*[,;]+", r"\1", text)
    return text.strip(" .")


def _h3_dialogue_quality_metrics(
    manifest: list[dict[str, Any]],
    *,
    story_description: str = "",
    maximum_line_words: Optional[int] = None,
) -> dict[str, Any]:
    """Measure obvious dialogue-collapse failures without judging prose taste."""

    locked = _h3_user_locked_dialogue_fingerprints(story_description)
    editable: list[dict[str, Any]] = []
    for turn, entry in enumerate(manifest or [], start=1):
        text = _h3_plain_dialogue_text(entry.get("spoken_text"))
        fingerprint = _h3_dialogue_word_fingerprint(text)
        if not fingerprint or fingerprint in locked:
            continue
        editable.append({
            "turn": turn,
            "speaker": str(entry.get("speaker_name") or "").strip().casefold(),
            "fingerprint": fingerprint,
            "words": len(text.split()),
        })

    count = len(editable)
    micro = [entry for entry in editable if entry["words"] <= 2]
    average_words = (
        sum(entry["words"] for entry in editable) / count
        if count else 0.0
    )
    by_fingerprint: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for entry in editable:
        if entry["words"] <= 3:
            by_fingerprint.setdefault(entry["fingerprint"], []).append(entry)
    cross_speaker_duplicates = {
        fingerprint: entries
        for fingerprint, entries in by_fingerprint.items()
        if len(entries) > 1
        and len({entry["speaker"] for entry in entries}) > 1
    }
    line_ceiling = max(1, int(maximum_line_words or 0)) if maximum_line_words else 0
    overlong = [
        entry for entry in editable
        if line_ceiling and entry["words"] > line_ceiling
    ]

    micro_pathology = (
        count >= 6
        and len(micro) >= max(3, int(math.ceil(count * 0.35)))
    )
    average_pathology = count >= 8 and average_words < 4.0
    problem_turns: set[int] = set()
    if micro_pathology:
        problem_turns.update(entry["turn"] for entry in micro)
    if average_pathology:
        problem_turns.update(
            entry["turn"] for entry in editable if entry["words"] <= 3
        )
    for entries in cross_speaker_duplicates.values():
        problem_turns.update(entry["turn"] for entry in entries)
    problem_turns.update(entry["turn"] for entry in overlong)

    issues: list[str] = []
    if micro_pathology:
        issues.append(
            f"{len(micro)} of {count} generated turns contain only one or two words"
        )
    if average_pathology:
        issues.append(
            f"generated dialogue averages only {average_words:.1f} words per turn"
        )
    if cross_speaker_duplicates:
        examples = ", ".join(
            " ".join(fingerprint)
            for fingerprint in list(cross_speaker_duplicates)[:3]
        )
        issues.append(
            "different characters share interchangeable short replies"
            + (f" ({examples})" if examples else "")
        )
    if overlong:
        examples = ", ".join(
            f"turn {entry['turn']} ({entry['words']}/{line_ceiling} words)"
            for entry in overlong[:4]
        )
        issues.append(
            "generated dialogue exceeds one native H3 clip and must be "
            f"shortened rather than split ({examples})"
        )

    micro_ratio = len(micro) / count if count else 0.0
    score = 0.0
    if count >= 6:
        score += micro_ratio * 10.0
    if count >= 8:
        score += max(0.0, 4.0 - average_words) * 2.0
    score += len(cross_speaker_duplicates) * 2.0
    score += sum(
        4.0 + max(0, entry["words"] - line_ceiling) * 0.5
        for entry in overlong
    )
    return {
        "editable_turns": count,
        "micro_turns": len(micro),
        "average_words": average_words,
        "cross_speaker_duplicates": len(cross_speaker_duplicates),
        "overlong_turns": len(overlong),
        "problem_turns": problem_turns,
        "issues": issues,
        "score": score,
    }


def _h3_dialogue_quality_issues(
    manifest: list[dict[str, Any]],
    *,
    story_description: str = "",
) -> list[str]:
    return list(_h3_dialogue_quality_metrics(
        manifest,
        story_description=story_description,
    )["issues"])


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
            field: _sanitize_h3_voice_guidance(raw.get(field) or "")
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
        speech_guard = (
            "Line length varies with the dramatic beat; concise delivery still "
            "uses complete conversational thoughts and substantive responses "
            "when the exchange allows"
        )
        if "line length varies" not in values["speech_pattern"].casefold():
            values["speech_pattern"] = (
                f"{values['speech_pattern']}. {speech_guard}"
            )
        avoid_guard = (
            "chains of interchangeable fragments or making every response a "
            "one-liner"
        )
        if "interchangeable fragment" not in values["avoid"].casefold():
            values["avoid"] = f"{values['avoid']}; {avoid_guard}"
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
    if lines:
        lines.append(
            "- Global dialogue rule: these profiles govern diction, syntax, "
            "subtext, and cadence, never a fixed word count. Vary line length "
            "and write complete responsive thoughts instead of chains of "
            "interchangeable fragments."
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


_H3_EXPLICIT_SPEECH_PATTERNS = (
    re.compile(
        r'<\s*d\s*>\s*(?:\[[^\]]+\]\s*)?(.+?)<\s*/\s*d\s*>',
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:says?|said|asks?|asked|replies?|replied|yells?|yelled|"
        r"shouts?|shouted|whispers?|whispered|calls?|called|speaks?|spoke)\b"
        r"[^\"\u201c\r\n]{0,120}[\"\u201c]([^\"\u201d\r\n]{1,600})"
        r"[\"\u201d]",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"[\"\u201c]([^\"\u201d\r\n]{1,600})[\"\u201d]"
        r"[^.?!\r\n]{0,120}\b(?:says?|said|asks?|asked|replies?|replied|"
        r"yells?|yelled|shouts?|shouted|whispers?|whispered|calls?|called)\b",
        flags=re.IGNORECASE,
    ),
)


def _h3_explicit_story_dialogue_fingerprints(
    value: Any,
) -> dict[tuple[str, ...], str]:
    """Return dialogue the user's concept explicitly identifies as speech.

    The broader literal-dialogue lock intentionally protects every quoted
    string once it appears in a screenplay manifest. Screenplay validation is
    narrower: quoted project titles and named objects must not be mistaken for
    required spoken lines, so only dialogue tags and quotes attached to a
    speech verb are required here.
    """

    text = _normalize_h3_text(value)
    found: dict[tuple[str, ...], str] = {}
    for pattern in _H3_EXPLICIT_SPEECH_PATTERNS:
        for match in pattern.finditer(text):
            spoken = _h3_plain_dialogue_text(match.group(1))
            fingerprint = _h3_dialogue_word_fingerprint(spoken)
            if fingerprint:
                found.setdefault(fingerprint, spoken)
    return found


def _h3_screenplay_recovery_reasons(
    screenplay: Any,
    *,
    story_description: str,
) -> list[str]:
    """Identify failures that must never reach H3 shot planning silently."""

    text = _normalize_h3_text(screenplay).strip()
    reasons: list[str] = []
    if len(text) < _H3_SCREENPLAY_MIN_CHARS:
        reasons.append("the screenplay answer is empty or truncated")

    required = _h3_explicit_story_dialogue_fingerprints(story_description)
    if required:
        present = {
            _h3_dialogue_word_fingerprint(entry.get("spoken_text"))
            for entry in _extract_h3_screenplay_dialogue(text)
        }
        missing = [
            spoken
            for fingerprint, spoken in required.items()
            if fingerprint not in present
        ]
        if missing:
            preview = "; ".join(repr(line) for line in missing[:3])
            reasons.append(
                "explicit user dialogue is missing from canonical screenplay "
                f"blocks ({preview})"
            )
    return reasons


def _h3_screenplay_budget_metrics(screenplay: Any) -> dict[str, int]:
    """Measure the screenplay budget that must be settled before locking."""

    manifest = _extract_h3_screenplay_dialogue(screenplay)
    line_words = [
        len(_h3_plain_dialogue_text(item.get("spoken_text")).split())
        for item in manifest
    ]
    return {
        "total_words": len(_normalize_h3_text(screenplay).split()),
        "spoken_words": sum(line_words),
        "maximum_line_words": max(line_words, default=0),
        "turns": len(manifest),
    }


def _h3_screenplay_budget_issues(
    screenplay: Any,
    *,
    story_description: str = "",
    max_total_words: int,
    max_spoken_words: int,
    maximum_line_words: int,
) -> list[str]:
    """Return hard pre-lock screenplay budget violations."""

    metrics = _h3_screenplay_budget_metrics(screenplay)
    issues: list[str] = []
    if metrics["total_words"] > max(1, int(max_total_words or 0)):
        issues.append(
            f"{metrics['total_words']} total words exceed the "
            f"{max_total_words}-word sequence budget"
        )
    if metrics["spoken_words"] > max(0, int(max_spoken_words or 0)):
        issues.append(
            f"{metrics['spoken_words']} spoken words exceed the "
            f"{max_spoken_words}-word dialogue budget"
        )
    locked = _h3_user_locked_dialogue_map(story_description)
    generated_sentence_words: list[int] = []
    for item in _extract_h3_screenplay_dialogue(screenplay):
        spoken = _h3_plain_dialogue_text(item.get("spoken_text"))
        if _h3_dialogue_word_fingerprint(spoken) in locked:
            continue
        sentences = [
            value.strip()
            for value in re.split(r"(?<=[.!?])\s+", spoken)
            if value.strip()
        ] or [spoken]
        generated_sentence_words.extend(
            len(_h3_plain_dialogue_text(value).split())
            for value in sentences
        )
    maximum_generated_sentence_words = max(generated_sentence_words, default=0)
    if maximum_generated_sentence_words > max(1, int(maximum_line_words or 1)):
        issues.append(
            "one generated sentence contains "
            f"{maximum_generated_sentence_words} "
            f"words; the native maximum is {maximum_line_words}"
        )
    return issues


def _truncate_h3_generated_line(value: Any, maximum_words: int) -> str:
    """Shorten generated dialogue only, preferring complete sentences."""

    text = _h3_plain_dialogue_text(value)
    limit = max(0, int(maximum_words or 0))
    words = text.split()
    if len(words) <= limit:
        return text
    if limit <= 0:
        return ""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    selected: list[str] = []
    used = 0
    for sentence in sentences:
        count = len(sentence.split())
        if used + count > limit:
            break
        selected.append(sentence)
        used += count
    if selected:
        return " ".join(selected)
    shortened = " ".join(words[:limit]).rstrip(" ,;:-")
    if shortened and shortened[-1:] not in ".!?":
        shortened += "."
    return shortened


def _fit_h3_dialogue_manifest_to_budget(
    manifest: list[dict[str, Any]],
    *,
    story_description: str,
    max_spoken_words: int,
    maximum_line_words: int,
) -> list[dict[str, Any]]:
    """Emergency-fit generated lines while preserving literal user dialogue."""

    source = [copy.deepcopy(item) for item in manifest or []]
    if not source:
        return []
    locked = _h3_user_locked_dialogue_map(story_description)
    generated_indices: list[int] = []
    allocations: dict[int, int] = {}
    locked_words = 0
    for index, item in enumerate(source):
        fingerprint = _h3_dialogue_word_fingerprint(item.get("spoken_text"))
        count = len(_h3_plain_dialogue_text(item.get("spoken_text")).split())
        if fingerprint in locked:
            item["spoken_text"] = locked[fingerprint]
            locked_words += count
        else:
            generated_indices.append(index)
            allocations[index] = min(count, max(1, int(maximum_line_words or 1)))

    available = max(0, int(max_spoken_words or 0) - locked_words)
    requested = sum(allocations.values())
    if requested > available and generated_indices:
        # Keep every generated conversational turn when possible. Allocate a
        # small complete-thought floor, then distribute remaining capacity in
        # source order until the hard global budget is exhausted.
        revised = {index: 0 for index in generated_indices}
        floor = min(3, available // len(generated_indices))
        for index in generated_indices:
            revised[index] = min(floor, allocations[index])
        remaining = available - sum(revised.values())
        while remaining > 0:
            advanced = False
            for index in generated_indices:
                if revised[index] >= allocations[index]:
                    continue
                revised[index] += 1
                remaining -= 1
                advanced = True
                if remaining <= 0:
                    break
            if not advanced:
                break
        allocations = revised

    fitted: list[dict[str, Any]] = []
    for index, item in enumerate(source):
        fingerprint = _h3_dialogue_word_fingerprint(item.get("spoken_text"))
        if fingerprint in locked:
            fitted.append(item)
            continue
        shortened = _truncate_h3_generated_line(
            item.get("spoken_text"),
            allocations.get(index, 0),
        )
        if not shortened:
            continue
        item["spoken_text"] = shortened
        fitted.append(item)
    return fitted


def _build_h3_budgeted_screenplay_fallback(
    *,
    story_description: str,
    story_blueprint: list[dict[str, Any]],
    screenplay: str,
    max_total_words: int,
    max_spoken_words: int,
    maximum_line_words: int,
) -> str:
    """Build a bounded canonical screenplay if focused compression fails.

    The binding continuity blueprint still reaches Pass 2 separately, so this
    emergency representation can remain compact without losing source events.
    It preserves speaker order and every literal user line; only generated
    prose/dialogue may be shortened.
    """

    source_manifest = _extract_h3_screenplay_dialogue(screenplay)
    # A failed creative/compression response may omit a literal user line.
    # The deterministic compiler is the last recovery boundary, so seed any
    # missing locked lines directly from the source registry before fitting
    # generated dialogue.  This benefits H3 and the shared long-form LTX path
    # without inventing or rewriting a word.
    present_fingerprints = {
        _h3_dialogue_word_fingerprint(item.get("spoken_text"))
        for item in source_manifest
    }
    try:
        from services.h3_story_ledger import extract_locked_dialogue

        for item in extract_locked_dialogue(story_description):
            spoken = _h3_plain_dialogue_text(item.get("text"))
            fingerprint = _h3_dialogue_word_fingerprint(spoken)
            if not fingerprint or fingerprint in present_fingerprints:
                continue
            source_manifest.append({
                "speaker_name": str(item.get("speaker") or "Speaker").strip(),
                "spoken_text": spoken,
                "delivery": str(item.get("delivery") or "").strip(),
                "physical_cue": "",
            })
            present_fingerprints.add(fingerprint)
    except Exception:
        pass

    manifest = _fit_h3_dialogue_manifest_to_budget(
        source_manifest,
        story_description=story_description,
        max_spoken_words=max_spoken_words,
        maximum_line_words=maximum_line_words,
    )
    action_fragments: list[str] = []
    for row in story_blueprint or []:
        if not isinstance(row, dict):
            continue
        for field in (
            "opening_cause", "active_objective", "visible_beats",
            "choice_or_discovery", "outgoing_handoff",
        ):
            value = row.get(field)
            if isinstance(value, list):
                action_fragments.extend(str(item).strip() for item in value if str(item).strip())
            elif str(value or "").strip():
                action_fragments.append(str(value).strip())
    if not action_fragments:
        action_fragments = [
            "The established characters perform the requested sequence in "
            "source order and reach its required visible ending."
        ]

    dialogue_blocks = [
        f"{str(item.get('speaker_name') or 'SPEAKER').upper()}\n"
        f"{_h3_plain_dialogue_text(item.get('spoken_text'))}"
        for item in manifest
        if _h3_plain_dialogue_text(item.get("spoken_text"))
    ]
    fixed = "INT. ESTABLISHED STORY LOCATION - CONTINUOUS\n\n"
    fixed += "\n\n".join(dialogue_blocks)
    fixed += "\n\nFADE OUT."
    remaining = max(0, int(max_total_words or 1) - len(fixed.split()))
    action_text = " ".join(action_fragments)
    action_text = _truncate_h3_generated_line(action_text, remaining)
    return (
        "INT. ESTABLISHED STORY LOCATION - CONTINUOUS\n\n"
        + (action_text or "")
        + ("\n\n" + "\n\n".join(dialogue_blocks) if dialogue_blocks else "")
        + "\n\nFADE OUT."
    )


def _long_form_screenplay_budget_issues(
    screenplay: Any,
    *,
    story_description: str = "",
    max_total_words: int,
    max_spoken_words: int,
) -> list[str]:
    """Apply the shared sequence budget without H3's per-clip line cap.

    LTX can carry dialogue across its rolling windows, so a single spoken turn
    does not need to fit H3's native 14.4-second lattice.  The complete bounded
    sequence still has the same hard action/dialogue capacity, however.  Using
    the common screenplay parser here prevents long-form LTX from receiving an
    oversized screenplay that Pass 2 can only rush, truncate, or stretch.
    """

    return _h3_screenplay_budget_issues(
        screenplay,
        story_description=story_description,
        max_total_words=max_total_words,
        max_spoken_words=max_spoken_words,
        # A sequence-wide cap disables H3's per-native-clip sentence check
        # while retaining the shared total/spoken-word validation.
        maximum_line_words=max(1, int(max_spoken_words or 1)),
    )


def _build_long_form_budgeted_screenplay_fallback(
    *,
    story_description: str,
    story_blueprint: list[dict[str, Any]],
    screenplay: str,
    max_total_words: int,
    max_spoken_words: int,
) -> str:
    """Compile a bounded canonical screenplay for non-H3 long-form models."""

    return _build_h3_budgeted_screenplay_fallback(
        story_description=story_description,
        story_blueprint=story_blueprint,
        screenplay=screenplay,
        max_total_words=max_total_words,
        max_spoken_words=max_spoken_words,
        maximum_line_words=max(1, int(max_spoken_words or 1)),
    )


def _restore_missing_h3_screenplay_dialogue(
    screenplay: Any,
    *,
    story_description: str,
) -> str:
    """Restore omitted locked lines as canonical screenplay performances.

    This is intentionally narrow: it never invents or rewrites words.  It is
    used only after the creative attempt and focused recovery both returned a
    usable screenplay but omitted an exact line owned by this bounded
    sequence.  An empty/truncated screenplay still fails rather than being
    papered over.
    """

    text = _normalize_h3_text(screenplay).strip()
    if len(text) < _H3_SCREENPLAY_MIN_CHARS:
        return text
    required = _h3_explicit_story_dialogue_fingerprints(story_description)
    present = {
        _h3_dialogue_word_fingerprint(entry.get("spoken_text"))
        for entry in _extract_h3_screenplay_dialogue(text)
    }
    missing = {
        fingerprint: spoken
        for fingerprint, spoken in required.items()
        if fingerprint not in present
    }
    if not missing:
        return text

    try:
        from services.h3_story_ledger import extract_locked_dialogue

        locked = extract_locked_dialogue(story_description)
    except Exception:
        locked = []
    speaker_by_fingerprint = {
        _h3_dialogue_word_fingerprint(item.get("text")): str(
            item.get("speaker") or "Speaker"
        ).strip()
        for item in locked
    }
    blocks: list[str] = []
    for fingerprint, spoken in missing.items():
        speaker = speaker_by_fingerprint.get(fingerprint, "Speaker")
        heading = re.sub(r"[^A-Za-z0-9 .'-]+", "", speaker).strip().upper()
        if not heading:
            heading = "SPEAKER"
        blocks.append(
            "The ongoing action creates a natural opening for the locked "
            "line. The speaker visibly addresses the established listener.\n\n"
            f"{heading}\n{spoken}"
        )
    insertion = "\n\n".join(blocks)
    fade_match = re.search(r"\n\s*FADE OUT\.?\s*$", text, re.IGNORECASE)
    if fade_match:
        return (
            text[:fade_match.start()].rstrip()
            + "\n\n"
            + insertion
            + "\n\nFADE OUT."
        )
    return text.rstrip() + "\n\n" + insertion


_H3_DIALOGUE_FORWARD_RE = re.compile(
    r"\b(?:"
    r"dialogue(?:[- ](?:heavy|driven|forward))?"
    r"|banter"
    r"|conversation[- ]driven"
    r"|(?:witty|sharp|funny|natural|realistic|strong|substantial)\s+dialogue"
    r"|(?:lots?|plenty)\s+of\s+dialogue"
    r")\b",
    flags=re.IGNORECASE,
)
_H3_SILENT_STORY_RE = re.compile(
    r"\b(?:no|without|zero)\s+(?:spoken\s+)?dialogue\b"
    r"|\b(?:minimal|sparse|limited|very little)\s+(?:spoken\s+)?dialogue\b"
    r"|\b(?:silent film|mostly silent|entirely silent|wordless)\b",
    flags=re.IGNORECASE,
)


def _h3_dialogue_density_targets(
    story_description: Any,
    *,
    target_duration: int,
) -> Optional[dict[str, int]]:
    """Return minimum dialogue targets only when the user asks for dialogue.

    These are pacing floors rather than quotas. They keep an explicitly
    dialogue-forward film from collapsing into a handful of one-line shots
    separated by long silent tails, which invites H3 to improvise speech.
    Action-first and intentionally silent concepts remain untouched.
    """

    concept = _normalize_h3_text(story_description).strip()
    if (
        not concept
        or _H3_SILENT_STORY_RE.search(concept)
        or not _H3_DIALOGUE_FORWARD_RE.search(concept)
    ):
        return None
    duration = max(8, int(target_duration or 0))
    minimum_turns = max(3, min(48, int(math.ceil(duration / 8.0))))
    minimum_words = max(
        minimum_turns * 5,
        int(round(duration * 0.70)),
    )
    return {
        "minimum_turns": minimum_turns,
        "minimum_words": minimum_words,
    }


def _h3_dialogue_density_issue(
    screenplay: Any,
    *,
    story_description: Any,
    target_duration: int,
) -> str:
    """Describe a sparse dialogue-forward screenplay without rejecting it."""

    targets = _h3_dialogue_density_targets(
        story_description,
        target_duration=target_duration,
    )
    return _h3_dialogue_density_issue_for_targets(screenplay, targets)


def _h3_dialogue_density_issue_for_targets(
    screenplay: Any,
    targets: Optional[dict[str, int]],
) -> str:
    """Check a resolved local dialogue contract without re-inferring intent.

    Long-form sequence prompts contain internal phrases such as "dialogue
    manifest". Re-scanning that wrapper used to make silent visual sequences
    look dialogue-forward even when the sequence architect requested none.
    """

    if not targets:
        return ""
    manifest = _extract_h3_screenplay_dialogue(screenplay)
    spoken_words = sum(
        len(_h3_plain_dialogue_text(entry.get("spoken_text")).split())
        for entry in manifest
    )
    turn_count = len(manifest)
    if (
        turn_count >= targets["minimum_turns"]
        and spoken_words >= targets["minimum_words"]
    ):
        return ""
    return (
        "the dialogue-forward concept produced only "
        f"{turn_count} spoken turns / {spoken_words} words; target at least "
        f"{targets['minimum_turns']} responsive turns / "
        f"{targets['minimum_words']} words across the complete film"
    )


def _apply_h3_character_table_read(
    manifest: list[dict[str, Any]],
    rows: Any,
    *,
    story_description: str,
    max_spoken_words: int,
    maximum_line_words: int = 30,
    turn_offset: int = 0,
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
    turn_offset = max(0, int(turn_offset or 0))
    expected_turns = set(range(
        turn_offset + 1,
        turn_offset + len(manifest) + 1,
    ))
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
        if turn in by_turn or turn not in expected_turns:
            raise ValueError("table read contains duplicate or out-of-range turns")
        by_turn[turn] = raw
    if set(by_turn) != expected_turns:
        raise ValueError("table read changed the screenplay turn sequence")

    locked = _h3_user_locked_dialogue_map(story_description)
    revised_manifest: list[dict[str, Any]] = []
    changed = 0
    original_word_count = 0
    revised_word_count = 0
    for turn, original in enumerate(manifest, start=turn_offset + 1):
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
        user_locked = original_fingerprint in locked
        if user_locked:
            candidate = locked[original_fingerprint]

        # A line must fit the effective native pass selected for this run.
        # Generated dialogue must fit one native clip. Splitting one character's
        # sentence across independent H3 generations produces abrupt partial
        # performances and weakens identity/context in the second clip. Literal
        # user-authored lines remain immutable; generated overlong lines must be
        # shortened by the table-read pass instead of silently restored.
        line_ceiling = max(1, int(maximum_line_words))
        if len(candidate.split()) > line_ceiling:
            if user_locked or len(original_text.split()) <= line_ceiling:
                candidate = original_text
            else:
                raise ValueError(
                    f"table read did not shorten generated turn {turn} to "
                    f"{line_ceiling} words"
                )
        # The roleplay/table-read pass should remove stiff prose, never make a
        # natural line sound like a thesaurus or corporate rewrite. Preserve
        # the screenplay line when the proposed edit introduces multiple new
        # elevated-language markers. This is comparative, so deliberately
        # formal characters and source lines are not flattened.
        if _h3_table_read_formalization_regressed(original_text, candidate):
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
    locked_word_count = sum(
        len(_h3_plain_dialogue_text(item.get("spoken_text")).split())
        for item in manifest
        if _h3_dialogue_word_fingerprint(item.get("spoken_text")) in locked
    )
    allowed_total = max(int(max_spoken_words or 0), locked_word_count)
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
    causal_handoff = clean(raw.get("causal_handoff"))
    if causal_handoff:
        parts.append("Story handoff: " + causal_handoff)
    persistent_state = clean(raw.get("persistent_story_state"))
    if persistent_state:
        parts.append("Continuity state: " + persistent_state)
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
    actions = [clean(value) for value in raw.get("action_beats") or []]
    actions = [value for value in actions if value]
    if actions:
        parts.append("Action: " + " Then ".join(actions))
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


def _expand_h3_dialogue_coverage_slots(
    shot_dicts: list[dict],
    *,
    desired_count: int,
) -> list[dict]:
    """Add same-scene coverage slots without inventing story events.

    A visual repair may return fewer shots than the locked conversation needs.
    These neutral reaction/coverage shots give the deterministic dialogue
    allocator legal native clips to work with. They inherit cast, setting, and
    continuity from neighboring footage, but never duplicate an action or a
    spoken turn from the LLM plan.
    """

    source = [copy.deepcopy(item) for item in shot_dicts or [] if isinstance(item, dict)]
    target = max(len(source), int(desired_count or 0))
    if not source or target <= len(source):
        return source

    extra = target - len(source)
    base_extras, remainder = divmod(extra, len(source))
    expanded: list[dict] = []
    coverage_number = 0
    for index, raw in enumerate(source):
        expanded.append(raw)
        clone_count = base_extras + (1 if index < remainder else 0)
        for _ in range(clone_count):
            coverage_number += 1
            clone = copy.deepcopy(raw)
            title = str(raw.get("title") or f"Shot {index + 1}").strip()
            clone["title"] = f"{title} — dialogue coverage {coverage_number}"
            clone["scene_goal"] = (
                "Continue the same exchange through a distinct speaker or "
                "listener reaction without replaying the prior action"
            )
            clone["narrative_role"] = "dialogue coverage"
            clone["scene_type"] = "dialogue reaction"
            clone["continuity_strategy"] = "continuous"
            clone["causal_handoff"] = str(
                raw.get("ending_beat")
                or raw.get("causal_handoff")
                or "The ongoing exchange continues in the same place and time"
            ).strip()
            clone["action_beats"] = [
                "The visible speaker and listeners continue the established "
                "exchange in real time; reactions advance without replaying "
                "an earlier entrance, gesture, impact, or reveal."
            ]
            clone["dialogue_beats"] = []
            clone["camera_plan"] = {
                "framing": "medium conversational coverage",
                "angle": "eye level",
                "movement": "subtle motivated reframe",
                "movement_intensity": "subtle",
                "lens_feel": "natural cinematic perspective",
                "reframing_notes": (
                    "Keep the active speaker visible and include the listener's "
                    "specific reaction when composition permits."
                ),
            }
            clone["audio_plan"] = {
                "mode": "ambient_only",
                "ambience": str(
                    (raw.get("audio_plan") or {}).get("ambience")
                    or "Continue the established natural ambience"
                ),
                "effects": [],
                "vocal_style": "Natural character voices",
                "timing_anchor": "video",
                "lip_sync_critical": False,
            }
            clone["ending_beat"] = (
                "A specific visible reaction or reply advances the same "
                "conversation toward its next beat"
            )
            clone["closing_blocking"] = str(
                raw.get("closing_blocking")
                or raw.get("spatial_setup")
                or "Preserve the established same-scene blocking"
            ).strip()
            if "image_source" in clone:
                clone["image_source"] = "previous"
            clone["multishot"] = False
            clone["window_prompts"] = []
            clone["video_prompt"] = _h3_rebuilt_visual_prompt(clone)
            expanded.append(clone)
    return expanded


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
        merged["story_scene_number"] = (
            first.get("story_scene_number")
            or second.get("story_scene_number")
        )
        merged["causal_handoff"] = (
            first.get("causal_handoff")
            or second.get("causal_handoff")
        )
        merged["persistent_story_state"] = (
            second.get("persistent_story_state")
            or first.get("persistent_story_state")
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
    *,
    project_context: str = "",
    allowed_character_names: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Keep speakers visible and enforce Director's closed canonical cast."""

    profiles = {
        str(row.get("character_name") or "").strip().casefold(): row
        for row in (voice_bible or [])
        if isinstance(row, dict) and row.get("character_name")
    }
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

    # First normalize abbreviated labels ("George") to the complete identity
    # validated by Pass 0 ("George Costanza"). The stable character_id remains
    # unchanged, so dialogue ownership and reference mapping do not move.
    for raw in shot_dicts or []:
        if not isinstance(raw, dict):
            continue
        for subject in raw.get("subjects_on_screen") or []:
            if not isinstance(subject, dict):
                continue
            profile = profile_for(subject.get("speaker_name") or "")
            canonical_name = str(
                (profile or {}).get("character_name") or ""
            ).strip()
            if canonical_name:
                subject["speaker_name"] = canonical_name

    # H3 may freely populate a scene with unnamed extras, but a named cameo is
    # a story/cast decision. Remove named silent characters introduced by the
    # planner when they are absent from the user concept and validated voice
    # bible. Their blocking can remain as a generic silent background patron.
    generic_roles = {
        "barista", "bystander", "cashier", "clerk", "crowd", "customer",
        "driver", "extra", "guard", "listener", "passerby", "patron",
        "server", "staff", "waiter", "waitress",
    }
    speaking_keys: set[str] = set()
    allowed_name_tokens = [
        _h3_speaker_name_tokens(name)
        for name in (allowed_character_names or [])
        if _h3_speaker_name_tokens(name)
    ]
    for raw in shot_dicts or []:
        if not isinstance(raw, dict):
            continue
        for beat in raw.get("dialogue_beats") or []:
            if isinstance(beat, dict) and _h3_plain_dialogue_text(
                beat.get("spoken_text")
            ):
                speaking_keys.add(
                    str(beat.get("speaker_id") or "").strip().casefold()
                )

    unsupported: dict[str, set[str]] = {}
    context_folded = str(project_context or "").casefold()
    if context_folded:
        for raw in shot_dicts or []:
            if not isinstance(raw, dict):
                continue
            for subject in raw.get("subjects_on_screen") or []:
                if not isinstance(subject, dict):
                    continue
                character_id = str(
                    subject.get("character_id") or ""
                ).strip()
                name = str(
                    subject.get("speaker_name") or character_id
                ).strip()
                name_tokens = _h3_speaker_name_tokens(name)
                first_name = name_tokens[0] if name_tokens else ""
                folded_name = name.casefold()
                is_generic = (
                    folded_name in generic_roles
                    or first_name in generic_roles
                    or folded_name.startswith(("the ", "a ", "an "))
                )
                is_spoken = (
                    character_id.casefold() in speaking_keys
                    or folded_name in speaking_keys
                )
                is_supported = bool(profile_for(name)) or bool(
                    folded_name and re.search(
                        rf"(?<![a-z0-9]){re.escape(folded_name)}(?![a-z0-9])",
                        context_folded,
                    )
                ) or bool(
                    first_name and len(first_name) >= 3 and re.search(
                        rf"(?<![a-z0-9]){re.escape(first_name)}(?![a-z0-9])",
                        context_folded,
                    )
                ) or any(
                    name_tokens
                    and allowed_tokens
                    and name_tokens[0] == allowed_tokens[0]
                    for allowed_tokens in allowed_name_tokens
                )
                if is_generic or is_spoken or is_supported:
                    continue
                identity_key = (character_id or folded_name).casefold()
                aliases = unsupported.setdefault(identity_key, set())
                aliases.add(name)
                visual = str(subject.get("visual_description") or "").strip()
                visual_name = re.match(
                    r"^([A-Z][A-Za-z0-9'’-]+(?:\s+[A-Z][A-Za-z0-9'’-]+){0,2})"
                    r"(?=\s*(?:\(|,|—|–|-))",
                    visual,
                )
                if visual_name:
                    aliases.add(visual_name.group(1).strip())
                if character_id:
                    aliases.add(character_id)

    def neutralize_unrequested(value: Any, aliases: set[str]) -> Any:
        if isinstance(value, str):
            result = value
            character_ids = [
                alias for alias in aliases
                if re.fullmatch(r"(?:char|subject|speaker)[_-]?\d+", alias, re.I)
            ]
            for alias in character_ids:
                result = re.sub(
                    rf"\s*\(\s*{re.escape(alias)}\s*\)",
                    "",
                    result,
                    flags=re.IGNORECASE,
                )
            for alias in sorted(aliases - set(character_ids), key=len, reverse=True):
                if alias:
                    result = re.sub(
                        rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                        "a silent background patron",
                        result,
                        flags=re.IGNORECASE,
                    )
            for alias in character_ids:
                result = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                    "the silent background patron",
                    result,
                    flags=re.IGNORECASE,
                )
            return re.sub(r"\s+", " ", result).strip()
        if isinstance(value, list):
            return [neutralize_unrequested(item, aliases) for item in value]
        if isinstance(value, dict):
            return {
                key: neutralize_unrequested(item, aliases)
                for key, item in value.items()
            }
        return value

    if unsupported:
        affected_fields = (
            "title", "narrative_role", "scene_goal", "environment",
            "visual_style", "spatial_setup", "action_beats", "camera_plan",
            "lighting", "mood", "ending_beat", "closing_blocking",
            "causal_handoff", "persistent_story_state", "audio_plan",
            "video_prompt", "window_prompts",
        )
        for raw in shot_dicts or []:
            if not isinstance(raw, dict):
                continue
            raw["subjects_on_screen"] = [
                subject
                for subject in (raw.get("subjects_on_screen") or [])
                if not (
                    isinstance(subject, dict)
                    and str(
                        subject.get("character_id")
                        or subject.get("speaker_name")
                        or ""
                    ).strip().casefold() in unsupported
                )
            ]
            for aliases in unsupported.values():
                for field in affected_fields:
                    if field in raw:
                        raw[field] = neutralize_unrequested(raw[field], aliases)
        removed = sorted({
            alias
            for aliases in unsupported.values()
            for alias in aliases
            if not re.fullmatch(r"(?:char|subject|speaker)[_-]?\d+", alias, re.I)
        })
        print(
            "[ShortFilmPlanner] Removed unrequested named H3 cast from the "
            "shot plan: " + ", ".join(removed)
        )

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
        speaker_sequence: list[str] = []
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
            speaker_sequence.append(speaker_name)
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
        camera_text = " ".join(
            re.sub(r"\s+", " ", str(camera.get(field) or "")).strip()
            for field in (
                "framing", "angle", "movement", "movement_intensity",
                "lens_feel", "reframing_notes",
            )
        ).casefold()
        has_close_coverage = bool(re.search(
            r"\b(?:close[- ]?ups?|medium close[- ]?ups?|"
            r"over[- ]the[- ]shoulder|reaction shots?)\b",
            camera_text,
        ))
        has_coverage_transition = bool(re.search(
            r"\b(?:cut(?:s|ting)?|refram(?:e|es|ing)|"
            r"alternat(?:e|es|ing)|transition(?:s|ing)?|reaction)\b",
            camera_text,
        ))
        needs_dialogue_coverage = (
            not has_close_coverage
            if len(speaker_sequence) == 1
            else not (has_close_coverage and has_coverage_transition)
        )
        if needs_dialogue_coverage:
            opening_framing = re.sub(
                r"\s+",
                " ",
                str(camera.get("framing") or "readable ensemble frame"),
            ).strip(" .")
            line_order = " then ".join(speaker_sequence)
            if len(speaker_sequence) == 1:
                coverage = (
                    f"Begin on {opening_framing} long enough to establish the "
                    f"action, then use a clean motivated reframe or internal "
                    f"cut to a medium close-up of {speaker_sequence[0]} before "
                    "the tagged line. Hold the unobstructed face and mouth for "
                    "the complete line, then return to the action or a motivated "
                    "listener reaction for the closing composition"
                )
                camera["framing"] = (
                    f"{opening_framing}, then a medium close-up of "
                    f"{speaker_sequence[0]}, ending on the action or reaction"
                )
            else:
                coverage = (
                    f"Begin on {opening_framing} only long enough to establish "
                    "the geography, then use clean speaker-motivated internal "
                    "cuts or reframes with alternating medium close-ups and "
                    "over-the-shoulder reactions in this exact line order: "
                    f"{line_order}. Hold each unobstructed face and mouth for "
                    "the complete line; reactions follow only after each line, "
                    "then finish on the motivated closing composition"
                )
                camera["framing"] = (
                    f"{opening_framing}, then alternating medium close-ups and "
                    "over-the-shoulder reactions"
                )
            movement = re.sub(
                r"\s+", " ", str(camera.get("movement") or "")
            ).strip(" .")
            coverage_movement = (
                "speaker-motivated internal cuts and reframes in exact dialogue order"
            )
            camera["movement"] = (
                f"{movement}; {coverage_movement}"
                if movement else coverage_movement
            )
        else:
            coverage = ""
        notes = re.sub(
            r"\s+", " ", str(camera.get("reframing_notes") or "")
        ).strip(" .")
        if coverage and coverage.casefold() not in notes.casefold():
            notes = f"{notes}. {coverage}" if notes else coverage
        if "mouth remain unobstructed" not in notes.casefold():
            notes = f"{notes}. {visibility}" if notes else visibility
        camera["reframing_notes"] = notes
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
    return shot_dicts


def _h3_subject_key(subject: dict, index: int) -> str:
    return str(
        subject.get("character_id")
        or subject.get("speaker_name")
        or f"subject_{index}"
    ).strip().casefold()


def _prepare_h3_story_continuity(
    shot_dicts: list[dict],
    story_blueprint: list[dict[str, Any]],
) -> list[dict]:
    """Carry the screenplay's causal scene ledger into executable shots.

    ``continuity_group`` describes physical place/time and may legitimately
    reset the start frame. It must not reset the plot. This reconciler maps
    ordered scene groups back to the architect blueprint, preserves explicit
    model-authored mappings when valid, and installs the opening/outgoing
    story handoffs as visible action beats so they survive prompt rebuilding.
    """

    if not shot_dicts or not story_blueprint:
        return shot_dicts
    scene_count = len(story_blueprint)
    run_numbers: list[int] = []
    prior_group = object()
    run_index = -1
    for index, raw in enumerate(shot_dicts):
        group = str(raw.get("continuity_group") or f"shot_{index + 1}")
        if group != prior_group:
            run_index += 1
            prior_group = group
        run_numbers.append(run_index)
    run_count = max(run_numbers, default=-1) + 1

    mapped_numbers: list[int] = []
    last_number = 1
    for shot_index, (raw, current_run) in enumerate(
        zip(shot_dicts, run_numbers)
    ):
        try:
            explicit = int(raw.get("story_scene_number") or 0)
        except (TypeError, ValueError):
            explicit = 0
        if shot_index == 0:
            scene_number = 1
        elif (
            1 <= explicit <= scene_count
            and last_number <= explicit <= last_number + 1
        ):
            scene_number = explicit
        elif run_count <= scene_count:
            scene_number = min(scene_count, current_run + 1)
        else:
            scene_number = min(
                scene_count,
                1 + int(math.floor(current_run * scene_count / run_count)),
            )
        scene_number = max(last_number, scene_number)
        raw["story_scene_number"] = scene_number
        mapped_numbers.append(scene_number)
        last_number = scene_number

    def add_visible_action(raw: dict, detail: str, *, first: bool) -> None:
        detail = re.sub(r"\s+", " ", str(detail or "")).strip()
        if not detail:
            return
        existing_text = " ".join([
            str(raw.get("scene_goal") or ""),
            *(str(value or "") for value in raw.get("action_beats") or []),
            str(raw.get("ending_beat") or ""),
        ]).casefold()
        if detail.casefold() in existing_text:
            return
        actions = list(raw.get("action_beats") or [])
        if first:
            actions.insert(0, detail)
        else:
            actions.append(detail)
        raw["action_beats"] = actions

    represented: set[int] = set(mapped_numbers)
    for scene_number in sorted(represented):
        indices = [
            index
            for index, mapped in enumerate(mapped_numbers)
            if mapped == scene_number
        ]
        if not indices:
            continue
        scene = story_blueprint[scene_number - 1]
        first_index, last_index = indices[0], indices[-1]
        first_shot = shot_dicts[first_index]
        last_shot = shot_dicts[last_index]
        opening = str(scene.get("opening_cause") or "").strip()
        outgoing = str(scene.get("outgoing_handoff") or "").strip()
        active_objective = str(scene.get("active_objective") or "").strip()
        state_after = str(scene.get("persistent_state_after") or "").strip()
        inherited_state = ""
        if scene_number > 1:
            inherited_state = str(
                story_blueprint[scene_number - 2].get(
                    "persistent_state_after"
                ) or ""
            ).strip()

        # The architect ledger is authoritative for cross-scene causality.
        # Pass 2 may phrase the same bridge differently, but it must not swap
        # in a new reason or unrelated incident during coverage planning.
        first_shot["causal_handoff"] = opening
        if scene_number > 1:
            add_visible_action(first_shot, opening, first=True)
        for index in indices:
            raw = shot_dicts[index]
            if index != first_index:
                raw["causal_handoff"] = str(
                    shot_dicts[index - 1].get("ending_beat")
                    or scene.get("story_purpose")
                    or ""
                ).strip()
            state_bits = []
            if active_objective:
                state_bits.append(f"Active objective: {active_objective}")
            if inherited_state:
                state_bits.append(f"Inherited state: {inherited_state}")
            if state_after:
                state_bits.append(f"State carried forward: {state_after}")
            raw["persistent_story_state"] = ". ".join(state_bits)

        add_visible_action(last_shot, outgoing, first=False)
        ending = re.sub(
            r"\s+", " ", str(last_shot.get("ending_beat") or "")
        ).strip()
        if outgoing and outgoing.casefold() not in ending.casefold():
            last_shot["ending_beat"] = (
                f"{ending}. {outgoing}" if ending else outgoing
            )

    # Dialogue shots are rebuilt again by the speaker-visibility pass. Silent
    # shots skip that branch, so rebuild every prompt here to guarantee the
    # causal ledger reaches both kinds of H3 generation.
    for raw in shot_dicts:
        raw["video_prompt"] = _h3_rebuilt_visual_prompt(raw)

    missing = sorted(set(range(1, scene_count + 1)) - represented)
    if missing:
        print(
            "[ShortFilmPlanner] H3 shot plan did not expose coverage for "
            "story blueprint scene(s) "
            + ", ".join(str(number) for number in missing)
            + "; preserving the screenplay plan without inventing footage."
        )
    return shot_dicts


def _prepare_long_form_ltx_prompt_contract(
    shot_dicts: list[dict],
    story_blueprint: list[dict[str, Any]],
    *,
    uses_generated_images: bool,
    screenplay_dialogue_manifest: Optional[list[dict[str, Any]]] = None,
) -> list[dict]:
    """Make every bounded LTX sequence executable and self-contained.

    Long-form Director plans are generated one bounded sequence at a time.
    The screenplay architecture is therefore reliable even for hour-long
    projects, but a local model can still leave a prompt string blank or omit
    the opening/closing bridge when translating one sequence into LTX shots.
    Repair those derived production fields from the structured shot data and
    the binding story ledger before the sequence is checkpointed.

    Image prompts remain static first-frame descriptions. Video/window prompts
    receive the causal action and repeat concrete visible identities because
    every LTX rolling window is a fresh text-conditioning pass.
    """

    if not shot_dicts:
        return shot_dicts

    scene = next(
        (row for row in story_blueprint or [] if isinstance(row, dict)),
        {},
    )

    def clean(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip(" .")

    def append_unique(base: Any, detail: Any, *, prepend: bool = False) -> str:
        text = clean(base)
        addition = clean(detail)
        if not addition or addition.casefold() in text.casefold():
            return text
        if not text:
            return addition + ("." if addition[-1:] not in ".!?" else "")
        if addition[-1:] not in ".!?":
            addition += "."
        return f"{addition} {text}" if prepend else f"{text} {addition}"

    def subject_details(raw: dict) -> list[str]:
        values: list[str] = []
        for index, subject in enumerate(raw.get("subjects_on_screen") or []):
            if not isinstance(subject, dict):
                continue
            description = clean(subject.get("visual_description"))
            name = clean(
                subject.get("speaker_name")
                or subject.get("character_id")
                or f"subject {index + 1}"
            )
            wardrobe = clean(subject.get("wardrobe"))
            position = clean(subject.get("position_or_relation"))
            bits = [description or name]
            if wardrobe:
                bits.append(f"wearing {wardrobe}")
            if position:
                bits.append(f"positioned {position}")
            value = ", ".join(bit for bit in bits if bit)
            if value:
                values.append(value)
        return values

    def identity_context(raw: dict) -> str:
        values = subject_details(raw)
        return "Visible subjects: " + "; ".join(values) if values else ""

    def camera_context(raw: dict) -> str:
        camera = raw.get("camera_plan") or {}
        bits = [
            clean(camera.get(field))
            for field in (
                "framing", "angle", "movement", "movement_intensity",
                "lens_feel", "reframing_notes",
            )
        ]
        bits = [value for value in bits if value]
        return "Camera " + ", ".join(bits) if bits else ""

    def dialogue_details(raw: dict) -> list[str]:
        values: list[str] = []
        subjects = raw.get("subjects_on_screen") or []
        speaker_names: dict[str, str] = {}
        for index, subject in enumerate(subjects):
            if not isinstance(subject, dict):
                continue
            speaker_id = clean(subject.get("character_id")).casefold()
            speaker_name = clean(
                subject.get("speaker_name")
                or subject.get("visual_description")
                or f"subject {index + 1}"
            )
            if speaker_id:
                speaker_names[speaker_id] = speaker_name
        for beat in raw.get("dialogue_beats") or []:
            if not isinstance(beat, dict):
                continue
            spoken = _h3_plain_dialogue_text(beat.get("spoken_text"))
            if not spoken:
                continue
            speaker_id = clean(beat.get("speaker_id"))
            speaker = speaker_names.get(
                speaker_id.casefold(), speaker_id or "The visible speaker"
            )
            delivery = clean(beat.get("delivery"))
            cue = clean(beat.get("physical_cue"))
            line = f'{speaker} says "{spoken}"'
            if delivery:
                line += f" {delivery}"
            if cue:
                line += f" while {cue}"
            values.append(line)
        return values

    def base_video_prompt(raw: dict, *, include_environment: bool) -> str:
        parts: list[str] = []
        identities = identity_context(raw)
        if identities:
            parts.append(identities)
        if include_environment:
            environment = clean(raw.get("environment"))
            if environment:
                parts.append(environment)
        actions = [clean(value) for value in raw.get("action_beats") or []]
        parts.extend(value for value in actions if value)
        parts.extend(dialogue_details(raw))
        camera = camera_context(raw)
        if camera:
            parts.append(camera)
        ending = clean(raw.get("ending_beat"))
        if ending:
            parts.append(ending)
        if not parts:
            parts.append(
                clean(raw.get("scene_goal"))
                or "The requested visible story beat unfolds clearly"
            )
        return ". ".join(part.strip(" .") for part in parts if part.strip(" .")) + "."

    def base_image_prompt(raw: dict) -> str:
        parts = [
            clean(raw.get(field))
            for field in (
                "environment", "visual_style", "lighting", "mood",
                "spatial_setup",
            )
        ]
        identities = identity_context(raw)
        if identities:
            parts.append(identities)
        parts = [part for part in parts if part]
        return ". ".join(part.strip(" .") for part in parts) + ("." if parts else "")

    def make_window_fallbacks(raw: dict, count: int) -> list[str]:
        count = max(1, int(count or 1))
        identities = identity_context(raw)
        environment = "" if uses_generated_images else clean(raw.get("environment"))
        camera = camera_context(raw)
        events = [
            clean(value) for value in raw.get("action_beats") or []
            if clean(value)
        ]
        events.extend(dialogue_details(raw))
        ending = clean(raw.get("ending_beat"))
        if ending:
            events.append(ending)
        if not events:
            events = [
                clean(raw.get("scene_goal"))
                or "The requested visible story beat advances"
            ]
        buckets: list[list[str]] = [[] for _ in range(count)]
        for event_index, event in enumerate(events):
            bucket_index = min(
                count - 1,
                int(event_index * count / max(1, len(events))),
            )
            buckets[bucket_index].append(event)
        prompts: list[str] = []
        for index, bucket in enumerate(buckets):
            if not bucket:
                bucket = [
                    clean(raw.get("scene_goal"))
                    or "The visible action advances toward the ending beat"
                ]
            parts = [identities, environment, *bucket, camera]
            prompts.append(
                ". ".join(
                    part.strip(" .") for part in parts if part.strip(" .")
                ) + "."
            )
        return prompts

    opening = clean(scene.get("opening_cause"))
    objective = clean(scene.get("active_objective"))
    outgoing = clean(scene.get("outgoing_handoff"))
    persistent = clean(scene.get("persistent_state_after"))
    first_shot = shot_dicts[0]
    last_shot = shot_dicts[-1]

    if opening:
        first_actions = [clean(value) for value in first_shot.get("action_beats") or []]
        if opening.casefold() not in " ".join(first_actions).casefold():
            first_shot["action_beats"] = [opening, *first_actions]
        first_shot["causal_handoff"] = opening
    if outgoing:
        last_actions = [clean(value) for value in last_shot.get("action_beats") or []]
        if outgoing.casefold() not in " ".join(last_actions).casefold():
            last_shot["action_beats"] = [*last_actions, outgoing]
        last_shot["ending_beat"] = append_unique(
            last_shot.get("ending_beat"), outgoing
        )

    for index, raw in enumerate(shot_dicts):
        state_bits = []
        if objective:
            state_bits.append(f"Active objective: {objective}")
        if persistent:
            state_bits.append(f"State carried forward: {persistent}")
        if state_bits:
            raw["persistent_story_state"] = ". ".join(state_bits)

        identities = identity_context(raw)
        if uses_generated_images:
            image_prompt = clean(raw.get("image_prompt"))
            if not image_prompt:
                image_prompt = base_image_prompt(raw)
            else:
                for detail in [
                    clean(raw.get("environment")),
                    clean(raw.get("spatial_setup")),
                    identities,
                ]:
                    image_prompt = append_unique(image_prompt, detail)
            if (
                index == 0
                and not _h3_explicit_story_dialogue_fingerprints(opening)
                and not re.search(r'["\u201c\u201d]', opening)
            ):
                # A bounded sequence is not a new film. Its first generated
                # frame must depict the visible state inherited from the prior
                # sequence rather than resetting to a generic establishment.
                image_prompt = append_unique(image_prompt, opening)
            raw["image_prompt"] = image_prompt
            if clean(raw.get("image_source")).casefold() not in {
                "original", "previous",
            }:
                raw["image_source"] = "previous" if index else "original"

        try:
            duration = max(1.0, float(raw.get("duration_sec") or 20.0))
        except (TypeError, ValueError):
            duration = 20.0
        video_prompt = clean(raw.get("video_prompt"))
        window_prompts = [
            clean(value)
            for value in raw.get("window_prompts") or []
            if clean(value)
        ]
        if duration <= 20.0:
            if not video_prompt and window_prompts:
                video_prompt = " ".join(window_prompts)
            if not video_prompt:
                video_prompt = base_video_prompt(
                    raw,
                    include_environment=not uses_generated_images,
                )
            if identities:
                video_prompt = append_unique(
                    video_prompt, identities, prepend=True
                )
            if index == 0:
                video_prompt = append_unique(video_prompt, opening, prepend=True)
            if index == len(shot_dicts) - 1:
                video_prompt = append_unique(video_prompt, outgoing)
            raw["video_prompt"] = video_prompt
            raw["window_prompts"] = []
            continue

        expected_windows = max(2, int(math.ceil(duration / 20.0)))
        fallbacks = make_window_fallbacks(raw, expected_windows)
        if not window_prompts and video_prompt:
            sentences = [
                clean(value)
                for value in re.split(r"(?<=[.!?])\s+", video_prompt)
                if clean(value)
            ]
            if sentences:
                buckets: list[list[str]] = [
                    [] for _ in range(expected_windows)
                ]
                for sentence_index, sentence in enumerate(sentences):
                    bucket_index = min(
                        expected_windows - 1,
                        int(
                            sentence_index * expected_windows
                            / max(1, len(sentences))
                        ),
                    )
                    buckets[bucket_index].append(sentence)
                window_prompts = [
                    ". ".join(bucket) + ("." if bucket else "")
                    for bucket in buckets
                ]
        while len(window_prompts) < expected_windows:
            window_prompts.append(fallbacks[len(window_prompts)])
        window_prompts = [
            value or fallbacks[prompt_index]
            for prompt_index, value in enumerate(window_prompts)
        ]
        for prompt_index, value in enumerate(window_prompts):
            if identities:
                value = append_unique(value, identities, prepend=True)
            if index == 0 and prompt_index == 0:
                value = append_unique(value, opening, prepend=True)
            if (
                index == len(shot_dicts) - 1
                and prompt_index == len(window_prompts) - 1
            ):
                value = append_unique(value, outgoing)
            window_prompts[prompt_index] = value
        raw["video_prompt"] = ""
        raw["window_prompts"] = window_prompts

    # Pass 2 is instructed to preserve screenplay dialogue, but unlike H3 the
    # legacy LTX schema stores those words directly in prompt prose rather than
    # a required dialogue_beats array. A local model can therefore return
    # structurally valid JSON while silently omitting a line. Restore any
    # missing canonical screenplay turn into a chronological generation slot
    # before the sequence is checkpointed. Existing prompt-authored dialogue
    # remains untouched.
    prompt_slots: list[tuple[dict, str, Optional[int]]] = []
    for raw in shot_dicts:
        video_prompt = clean(raw.get("video_prompt"))
        if video_prompt:
            prompt_slots.append((raw, "video_prompt", None))
        for prompt_index, value in enumerate(raw.get("window_prompts") or []):
            if clean(value):
                prompt_slots.append((raw, "window_prompts", prompt_index))

    def slot_text(slot: tuple[dict, str, Optional[int]]) -> str:
        raw, field, prompt_index = slot
        if prompt_index is None:
            return clean(raw.get(field))
        values = raw.get(field) or []
        return clean(values[prompt_index]) if prompt_index < len(values) else ""

    def spoken_occurrences(text: str, spoken: str) -> int:
        haystack = _h3_dialogue_word_fingerprint(text)
        needle = _h3_dialogue_word_fingerprint(spoken)
        if not needle or len(needle) > len(haystack):
            return 0
        width = len(needle)
        return sum(
            haystack[index:index + width] == needle
            for index in range(len(haystack) - width + 1)
        )

    dialogue_manifest = [
        item for item in screenplay_dialogue_manifest or []
        if isinstance(item, dict)
        and _h3_plain_dialogue_text(item.get("spoken_text"))
    ]
    required_occurrences: dict[tuple[str, ...], int] = {}
    for dialogue_index, item in enumerate(dialogue_manifest):
        spoken = _h3_plain_dialogue_text(item.get("spoken_text"))
        fingerprint = _h3_dialogue_word_fingerprint(spoken)
        required_occurrences[fingerprint] = (
            required_occurrences.get(fingerprint, 0) + 1
        )
        present_count = sum(
            spoken_occurrences(slot_text(slot), spoken)
            for slot in prompt_slots
        )
        if present_count >= required_occurrences[fingerprint]:
            continue
        if not prompt_slots:
            break
        target_index = min(
            len(prompt_slots) - 1,
            int(dialogue_index * len(prompt_slots) / max(1, len(dialogue_manifest))),
        )
        raw, field, prompt_index = prompt_slots[target_index]
        speaker = clean(item.get("speaker_name")) or "The visible speaker"
        line = f'{speaker} says "{spoken}"'
        if prompt_index is None:
            raw[field] = append_unique(raw.get(field), line)
        else:
            values = list(raw.get(field) or [])
            values[prompt_index] = append_unique(values[prompt_index], line)
            raw[field] = values
        print(
            "[ShortFilmPlanner] Restored one screenplay dialogue turn "
            "missing from the long-form LTX video prompts."
        )

    return shot_dicts


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

        if not is_audio_mode and target_duration > _DIRECTOR_LONG_FORM_CHAPTER_SECONDS:
            self._configure_planning_runtime(
                kwargs,
                kind="short_film_story",
                fingerprint_payload={
                    "planner_revision": _DIRECTOR_LONG_FORM_PLAN_REVISION,
                    "story_description": story_description,
                    "target_duration": target_duration,
                    "target_scenes": target_scenes,
                    "narrative_mode": narrative_mode,
                    "video_model": self._video_model,
                    "image_model": self._image_model,
                    "shot_image_policy": shot_image_policy,
                    "characters": characters or [],
                    "character_ref_labels": self._character_ref_labels or [],
                    "location_ref_labels": self._location_ref_labels or [],
                    "fps": fps,
                    "frames_steps": frames_steps,
                    "frames_minimum": frames_minimum,
                    "frames_maximum": frames_maximum,
                },
            )
        elif is_audio_mode and len(clips or []) > 12:
            self._configure_planning_runtime(
                kwargs,
                kind="short_film_audio",
                fingerprint_payload={
                    "planner_revision": 2,
                    "story_description": story_description,
                    "clips": clips or [],
                    "lyrics": lyrics or [],
                    "speaker_mappings": speaker_mappings or {},
                    "characters": characters or [],
                    "reference_image_path": reference_image_path,
                    "video_model": self._video_model,
                    "image_model": self._image_model,
                    "shot_image_policy": shot_image_policy,
                },
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
        elif target_duration > _DIRECTOR_LONG_FORM_CHAPTER_SECONDS:
            shots, title = self._plan_long_story_driven(
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
                "Short film — preserve the causal story blueprint across shots and locations",
                "Match camera complexity to emotional content",
                "Dialogue must appear in video prompts with speaker cues",
            ],
        )

    def _plan_long_story_driven(
        self,
        *,
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
        frames_maximum: Optional[int],
        nsfw: bool,
        polish_block: str,
        multishot_lora_mode: bool,
    ) -> tuple[list[ShotPlan], Optional[str]]:
        """Plan a long film as chapters containing bounded sequences.

        Literal user dialogue is registered once for the complete film, then
        assigned to exactly one chapter and one sequence.  Each sequence is at
        most ninety seconds, independently validated, and durably checkpointed.
        This keeps local-model context bounded and makes a one-hour plan
        resumable instead of all-or-nothing.
        """

        try:
            from services.h3_story_ledger import (
                extract_locked_dialogue,
                extract_source_events,
            )

            locked_dialogue = extract_locked_dialogue(story_description)
        except Exception:
            locked_dialogue = []
            extract_source_events = None
        redacted_story = _redact_long_form_dialogue(
            story_description,
            locked_dialogue,
        )
        source_events = (
            extract_source_events(redacted_story)
            if callable(extract_source_events) else []
        )
        dialogue_manifest = [
            {
                "dialogue_id": str(item.get("dialogue_id") or "").upper(),
                "speaker": item.get("speaker") or "Speaker",
                "exact_text": item.get("text") or "",
            }
            for item in locked_dialogue
        ]

        chapter_count = max(
            2,
            math.ceil(
                target_duration / _DIRECTOR_LONG_FORM_CHAPTER_SECONDS
            ),
        )
        base_duration, remainder = divmod(int(target_duration), chapter_count)
        chapter_durations = [
            base_duration + (1 if index < remainder else 0)
            for index in range(chapter_count)
        ]
        sequence_durations_by_chapter = [
            _bounded_long_form_durations(duration)
            for duration in chapter_durations
        ]
        total_sequences = sum(
            len(durations) for durations in sequence_durations_by_chapter
        )
        checkpoint = copy.deepcopy(
            self._planning_resume_checkpoint or {}
        )
        checkpoint.setdefault("completed_sequences", {})
        checkpoint.setdefault("chapter_sequences", {})

        # A complete-film story bible is the durable memory shared by every
        # bounded chapter and sequence.  Without it, a one-hour project asks
        # the local model to rediscover the premise, cast, locations, tone,
        # recurring gags, and ending dozens of times.  Existing checkpoints
        # remain resumable: when an older outline already exists we derive a
        # deterministic compatibility bible rather than inserting a surprise
        # LLM call into the middle of a resumed project.
        supplied_character_names = [
            str(
                getattr(profile, "display_name", "")
                or getattr(profile, "id", "")
                or ""
            ).strip()
            for profile in char_profiles or []
            if str(
                getattr(profile, "display_name", "")
                or getattr(profile, "id", "")
                or ""
            ).strip()
        ]
        story_bible = checkpoint.get("story_bible")
        story_bible_generated_now = False
        existing_plan = bool(
            isinstance(checkpoint.get("outline"), list)
            or checkpoint.get("completed_sequences")
        )
        if not isinstance(story_bible, dict):
            story_bible_candidate: Any = None
            if not existing_plan:
                self._emit_planning_progress(
                    message="Building the complete-film story bible...",
                    current=0,
                    total=total_sequences,
                    stage="long_form_story_bible",
                    chapter=0,
                    chapter_count=chapter_count,
                )
                try:
                    story_bible_candidate = self._call_llm_json(
                        user_prompt=(
                            f"Build the binding story bible for one {target_duration}-second "
                            f"film divided into {chapter_count} chapters.\n\n"
                            "Identify the premise engine, tone, final payoff, complete "
                            "canonical cast, and a location registry before any chapter "
                            "is written. Put the central/recurring protagonists first in "
                            "canonical_characters, and enumerate every named character "
                            "who may speak later so a bounded sequence cannot invent a "
                            "speaker. When the user requests many/different shows, "
                            "worlds, rooms, or encounters, pre-plan enough DISTINCT "
                            "locations and supporting ensembles to sustain the film; do "
                            "not cycle through only the examples.\n\n"
                            "A recurring motif is an action/dialogue pattern the user "
                            "intends to happen in multiple encounters (for example, "
                            "'in every room', 'each time', or 'whatever show'). Put its "
                            "source_event_ids and dialogue_ids in recurring_motifs and "
                            "choose a realistic occurrence range. One-time plot events "
                            "must not become motifs. Every repetition must evolve through "
                            "a different setup, reaction, escalation, and consequence.\n\n"
                            "Lock state rules: disappearances, injuries, knowledge, "
                            "relationships, wardrobe damage, and prop ownership persist "
                            "until an explicit on-screen event changes them. Lock the "
                            "user's genre and ending; never invent a redemptive or serious "
                            "arc merely to fill time. Do not write screenplay dialogue.\n\n"
                            f"SOURCE EVENTS:\n{json.dumps(source_events, ensure_ascii=False, indent=2)}\n\n"
                            f"LOCKED DIALOGUE:\n{json.dumps(dialogue_manifest, ensure_ascii=False, indent=2)}\n\n"
                            f"SUPPLIED CHARACTER NAMES:\n{json.dumps(supplied_character_names, ensure_ascii=False)}\n\n"
                            f"USER CONCEPT:\n{redacted_story}"
                        ),
                        system_prompt=(
                            "You are Maestro's complete-film continuity producer. "
                            "Return only the requested compact JSON object. Preserve the "
                            "user's creative intent while making it executable across "
                            "many independently planned video clips."
                        ),
                        max_tokens=min(9000, 2600 + chapter_count * 360),
                        thinking_budget=0,
                        temperature=0.5,
                        image_paths=self._build_all_image_paths(
                            reference_image_path,
                            has_reference,
                        ),
                        json_schema=LONG_FORM_STORY_BIBLE_SCHEMA,
                    )
                    story_bible_generated_now = isinstance(
                        story_bible_candidate,
                        dict,
                    )
                except InterruptedError:
                    raise
                except Exception as exc:
                    print(
                        "[ShortFilmPlanner] Story-bible planning was unavailable; "
                        f"using the deterministic continuity contract ({exc})."
                    )
            else:
                print(
                    "[ShortFilmPlanner] Resuming a legacy long-form checkpoint "
                    "with a deterministic compatibility story bible."
                )
            story_bible = normalize_long_form_story_bible(
                story_bible_candidate,
                story_description=story_description,
                locked_dialogue=locked_dialogue,
                source_events=source_events,
                character_names=supplied_character_names,
                chapter_count=chapter_count,
            )
        else:
            story_bible = normalize_long_form_story_bible(
                story_bible,
                story_description=story_description,
                locked_dialogue=locked_dialogue,
                source_events=source_events,
                character_names=supplied_character_names,
                chapter_count=chapter_count,
            )
        bible_issues = long_form_story_bible_quality_issues(
            story_bible,
            chapter_count=chapter_count,
        )
        if story_bible_generated_now and bible_issues:
            self._emit_planning_progress(
                message="Expanding the story bible for long-form variety...",
                current=0,
                total=total_sequences,
                stage="long_form_story_bible_repair",
                chapter=0,
                chapter_count=chapter_count,
            )
            try:
                repaired_story_bible = self._call_llm_json(
                    user_prompt=(
                        "Revise this complete-film story bible to fix only the "
                        "listed structural coverage gaps. Return the entire revised "
                        "object. Preserve its premise, tone, ending, source IDs, and "
                        "existing canonical entries. Add distinct concrete locations "
                        "and named speaking ensemble members where requested; do not "
                        "add filler, screenplay dialogue, or a new story arc. Put the "
                        "central/recurring protagonists first in canonical_characters.\n\n"
                        "GAPS:\n- " + "\n- ".join(bible_issues) + "\n\n"
                        "CURRENT STORY BIBLE:\n"
                        f"{json.dumps(story_bible, ensure_ascii=False, indent=2)}\n\n"
                        f"USER CONCEPT:\n{redacted_story}"
                    ),
                    system_prompt=(
                        "You are Maestro's long-form continuity editor. Return "
                        "only the requested complete JSON story bible."
                    ),
                    max_tokens=min(10000, 3000 + chapter_count * 420),
                    thinking_budget=0,
                    temperature=0.38,
                    image_paths=None,
                    json_schema=LONG_FORM_STORY_BIBLE_SCHEMA,
                )
                story_bible = normalize_long_form_story_bible(
                    repaired_story_bible,
                    story_description=story_description,
                    locked_dialogue=locked_dialogue,
                    source_events=source_events,
                    character_names=supplied_character_names,
                    chapter_count=chapter_count,
                )
                remaining_bible_issues = long_form_story_bible_quality_issues(
                    story_bible,
                    chapter_count=chapter_count,
                )
                if remaining_bible_issues:
                    print(
                        "[ShortFilmPlanner] Focused story-bible expansion left "
                        "non-fatal coverage warnings: "
                        + "; ".join(remaining_bible_issues)
                    )
            except InterruptedError:
                raise
            except Exception as exc:
                print(
                    "[ShortFilmPlanner] Focused story-bible expansion was "
                    f"unavailable; continuing with the safe base bible ({exc})."
                )
        locked_dialogue = resolve_locked_dialogue_speakers(
            locked_dialogue,
            story_description=story_description,
            story_bible=story_bible,
        )
        dialogue_manifest = [
            {
                "dialogue_id": str(item.get("dialogue_id") or "").upper(),
                "speaker": item.get("speaker") or "Speaker",
                "exact_text": item.get("text") or "",
            }
            for item in locked_dialogue
        ]
        checkpoint.update({
            "story_bible_revision": LONG_FORM_STORY_BIBLE_REVISION,
            "story_bible": story_bible,
            "stage": "story_bible_complete",
        })
        self._publish_planning_checkpoint(checkpoint)

        chapter_schema = {
            "type": "array",
            "minItems": chapter_count,
            "maxItems": chapter_count,
            "items": {
                "type": "object",
                "properties": {
                    "chapter": {"type": "integer"},
                    "title": {"type": "string"},
                    "location_id": {"type": "string"},
                    "location_time": {"type": "string"},
                    "objective": {"type": "string"},
                    "opening_state": {"type": "string"},
                    "closing_state": {"type": "string"},
                    "causal_handoff": {"type": "string"},
                    "persistent_state": {"type": "string"},
                    "character_state_changes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 30,
                    },
                    "cast_present": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 30,
                    },
                    "recurring_motif_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    },
                    "dialogue_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": len(locked_dialogue),
                    },
                    "source_event_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": len(source_events),
                    },
                },
                "required": [
                    "chapter", "title", "location_id", "location_time", "objective", "opening_state",
                    "closing_state", "causal_handoff", "persistent_state",
                    "character_state_changes", "cast_present",
                    "recurring_motif_ids", "dialogue_ids",
                    "source_event_ids",
                ],
                "additionalProperties": False,
            },
        }
        geometry = "\n".join(
            f"- Chapter {index + 1}: {duration} seconds; "
            f"{len(sequence_durations_by_chapter[index])} bounded sequences"
            for index, duration in enumerate(chapter_durations)
        )
        outline = checkpoint.get("outline")
        outline_generated_now = False
        if not isinstance(outline, list) or len(outline) != chapter_count:
            self._emit_planning_progress(
                message=f"Planning the complete {chapter_count}-chapter story arc...",
                current=0,
                total=total_sequences,
                stage="long_form_outline",
                chapter=0,
                chapter_count=chapter_count,
            )
            try:
                outline = self._call_llm_json(
                    user_prompt=(
                        f"Design exactly {chapter_count} causal chapters for one "
                        f"{target_duration}-second film.\n\n"
                        f"CHAPTER GEOMETRY:\n{geometry}\n\n"
                        f"BINDING COMPLETE-FILM STORY BIBLE:\n"
                        f"{json.dumps(story_bible, ensure_ascii=False, indent=2)}\n\n"
                        "Every chapter must advance the same story. Never recap, "
                        "reset, or repeat prior action. A location change must be "
                        "caused by the preceding visible decision or consequence. "
                        "Carry objectives, knowledge, relationships, injuries, "
                        "wardrobe, props, and unresolved danger forward. Only the "
                        "final chapter resolves the requested outcome. Use only "
                        "canonical location_id values and list the cast physically "
                        "present. Record every named physical availability change "
                        "as 'Name: concrete new state' in character_state_changes "
                        "(death, injury, disappearance, transformation, return, or "
                        "restoration); never reintroduce an unavailable character "
                        "until an explicit restoration event. Assign every one-time dialogue_id and "
                        "source_event_id to exactly one chapter in source order. "
                        "Assign each recurring_motif_id to the number of chapters "
                        "required by the bible; its attached IDs intentionally recur "
                        "there, but each performance must be newly staged and advance "
                        "the story. Do not write new dialogue.\n\n"
                        f"IMMUTABLE SOURCE EVENT REGISTRY:\n"
                        f"{json.dumps(source_events, ensure_ascii=False, indent=2)}\n\n"
                        f"LOCKED DIALOGUE REGISTRY:\n"
                        f"{json.dumps(dialogue_manifest, ensure_ascii=False, indent=2)}\n\n"
                        f"USER CONCEPT WITH DIALOGUE PLACEHOLDERS:\n{redacted_story}"
                    ),
                    system_prompt=(
                        "You are Maestro's long-form story architect. Return only "
                        "the requested JSON array. Build one coherent film in "
                        "bounded chapters, not unrelated shorts."
                    ),
                    max_tokens=min(6000, 1100 + chapter_count * 320),
                    thinking_budget=0,
                    temperature=0.45,
                    image_paths=self._build_all_image_paths(
                        reference_image_path,
                        has_reference,
                    ),
                    json_schema=chapter_schema,
                )
                outline_generated_now = isinstance(outline, list)
            except InterruptedError:
                raise
            except Exception as exc:
                print(
                    "[ShortFilmPlanner] Long-form chapter outline failed; "
                    f"using a deterministic causal scaffold ({exc})."
                )
                outline = []
        if outline_generated_now:
            outline_issues = long_form_outline_quality_issues(
                outline,
                story_bible=story_bible,
                chapter_count=chapter_count,
            )
            if outline_issues:
                self._emit_planning_progress(
                    message="Repairing the complete-film chapter arc...",
                    current=0,
                    total=total_sequences,
                    stage="long_form_outline_repair",
                    chapter=0,
                    chapter_count=chapter_count,
                )
                try:
                    repaired_outline = self._call_llm_json(
                        user_prompt=(
                            f"Revise this outline into exactly {chapter_count} "
                            "causal chapters and fix only the listed structural "
                            "gaps. Preserve all source_event_ids, dialogue_ids, "
                            "recurring_motif_ids, exact story facts, and the ending. "
                            "Use the registered locations with purposeful variety, "
                            "give every chapter a distinct dramatic objective, and "
                            "keep each opening causally attached to the preceding "
                            "closing. Return the entire revised array.\n\n"
                            "GAPS:\n- " + "\n- ".join(outline_issues) + "\n\n"
                            "BINDING STORY BIBLE:\n"
                            f"{json.dumps(story_bible, ensure_ascii=False, indent=2)}\n\n"
                            "CURRENT OUTLINE:\n"
                            f"{json.dumps(outline, ensure_ascii=False, indent=2)}"
                        ),
                        system_prompt=(
                            "You are Maestro's long-form chapter editor. Return "
                            "only the requested complete JSON array."
                        ),
                        max_tokens=min(6500, 1300 + chapter_count * 350),
                        thinking_budget=0,
                        temperature=0.35,
                        image_paths=None,
                        json_schema=chapter_schema,
                    )
                    if (
                        isinstance(repaired_outline, list)
                        and len(repaired_outline) == chapter_count
                    ):
                        outline = repaired_outline
                    else:
                        print(
                            "[ShortFilmPlanner] Focused chapter repair returned "
                            "an incomplete outline; keeping the original arc."
                        )
                except InterruptedError:
                    raise
                except Exception as exc:
                    print(
                        "[ShortFilmPlanner] Focused chapter repair was unavailable; "
                        f"keeping the original arc ({exc})."
                    )
        if not isinstance(outline, list) or len(outline) != chapter_count:
            print(
                "[ShortFilmPlanner] Long-form chapter outline was incomplete; "
                "using a deterministic causal scaffold."
            )
            outline = [
                {
                    "chapter": index + 1,
                    "title": f"Chapter {index + 1}",
                    "location_id": f"chapter_{index + 1}_location",
                    "location_time": "The story's established location and time",
                    "objective": (
                        f"Advance the user's story through chapter {index + 1} "
                        f"of {chapter_count} without replaying earlier events"
                    ),
                    "opening_state": (
                        "Establish the requested opening situation"
                        if index == 0 else
                        f"Show the concrete result of chapter {index}'s ending"
                    ),
                    "closing_state": (
                        "Complete the requested outcome"
                        if index + 1 == chapter_count else
                        f"Create a visible cause that launches chapter {index + 2}"
                    ),
                    "causal_handoff": "Carry the preceding visible consequence forward",
                    "persistent_state": "Preserve all established character and world state",
                    "character_state_changes": [],
                    "cast_present": (
                        [
                            item.get("name")
                            for item in story_bible.get("canonical_characters") or []
                            if isinstance(item, dict) and item.get("name")
                        ][:1]
                        if story_bible.get("allow_cast_expansion") else
                        [
                            item.get("name")
                            for item in story_bible.get("canonical_characters") or []
                            if isinstance(item, dict) and item.get("name")
                        ]
                    ),
                    "recurring_motif_ids": [],
                    "dialogue_ids": [],
                    "source_event_ids": [],
                }
                for index in range(chapter_count)
            ]
        outline = _normalize_long_form_event_ownership(
            outline,
            source_events=source_events,
        )
        outline = _normalize_long_form_dialogue_ownership(
            outline,
            allowed_dialogue=locked_dialogue,
            source_length=max(1, len(story_description)),
            source_events=source_events,
        )
        outline, story_bible = normalize_long_form_outline(
            outline,
            story_bible=story_bible,
            chapter_count=chapter_count,
        )
        outline, location_coverage_warnings = ensure_long_form_location_coverage(
            outline,
            story_bible=story_bible,
        )
        if location_coverage_warnings:
            print(
                "[ShortFilmPlanner] Long-form location coverage repair: "
                + "; ".join(location_coverage_warnings)
            )
        outline = apply_long_form_recurring_motifs(
            outline,
            story_bible=story_bible,
        )
        outline = _normalize_long_form_plan_references(
            outline,
            allowed_dialogue=locked_dialogue,
            source_events=source_events,
        )
        checkpoint.update({
            "stage": "chapters_planned",
            "target_duration": target_duration,
            "chapter_durations": chapter_durations,
            "sequence_durations_by_chapter": sequence_durations_by_chapter,
            "dialogue_manifest": dialogue_manifest,
            "source_event_manifest": source_events,
            "redacted_story": redacted_story,
            "story_bible_revision": LONG_FORM_STORY_BIBLE_REVISION,
            "story_bible": story_bible,
            "outline": outline,
            "location_coverage_repairs": location_coverage_warnings,
            "total_sequences": total_sequences,
        })
        self._publish_planning_checkpoint(checkpoint)

        is_h3_native = self._video_model.lower().startswith("minimax_h3")
        cached_voice_bible = checkpoint.get("h3_voice_bible")
        if is_h3_native and not isinstance(cached_voice_bible, list):
            self._emit_planning_progress(
                message="Building the shared character voice bible...",
                current=len(checkpoint.get("completed_sequences") or {}),
                total=total_sequences,
                stage="long_form_voice_bible",
                chapter=0,
                chapter_count=chapter_count,
            )
            self._long_form_voice_bible_max_items = max(
                1,
                len(story_bible.get("canonical_characters") or []),
            )
            cached_voice_bible = self._build_h3_character_voice_bible(
                story_description=(
                    f"{story_description}\n\nBINDING COMPLETE-FILM STORY BIBLE:\n"
                    f"{format_long_form_story_bible(story_bible)}"
                ),
                char_profiles=char_profiles,
            )
            checkpoint["h3_voice_bible"] = cached_voice_bible
            checkpoint["stage"] = "voice_bible_complete"
            self._publish_planning_checkpoint(checkpoint)

        dialogue_by_id = {
            str(item.get("dialogue_id") or "").upper(): item
            for item in locked_dialogue
        }
        all_shots: list[ShotPlan] = []
        first_title: Optional[str] = None
        completed_sequences = checkpoint.setdefault(
            "completed_sequences", {}
        )
        completed_count = 0

        # Reuse global creative context in every bounded sequence. Rebuilding
        # the voice bible and inner story architect forty times for an hour-long
        # film would add cost without adding continuity.
        self._long_form_story_blueprint_override = []
        self._long_form_story_bible_override = story_bible
        # Forty-eight 8K reasoning passes would make a one-hour project spend
        # hundreds of thousands of tokens before writing any screenplay.
        # The film/chapter/sequence architecture is already locked by this
        # point, so a compact 4K creative allowance is sufficient; the exact
        # dialogue restorer remains the deterministic final safety net.
        self._long_form_screenplay_thinking_budget_override = 4096
        if is_h3_native:
            self._long_form_h3_voice_bible_override = cached_voice_bible or []
        try:
            for chapter_index, (chapter, chapter_duration, sequence_durations) in enumerate(
                zip(outline, chapter_durations, sequence_durations_by_chapter),
                start=1,
            ):
                previous_chapter = (
                    outline[chapter_index - 2]
                    if chapter_index > 1 else None
                )
                next_chapter = (
                    outline[chapter_index]
                    if chapter_index < chapter_count else None
                )
                chapter_dialogue_ids = [
                    str(value or "").upper()
                    for value in chapter.get("dialogue_ids") or []
                    if str(value or "").upper() in dialogue_by_id
                ]
                chapter_dialogue = [
                    dialogue_by_id[dialogue_id]
                    for dialogue_id in chapter_dialogue_ids
                ]
                chapter_event_ids = [
                    str(value or "").upper()
                    for value in chapter.get("source_event_ids") or []
                ]
                source_event_by_id = {
                    str(item.get("event_id") or "").upper(): item
                    for item in source_events
                }
                chapter_events = [
                    source_event_by_id[event_id]
                    for event_id in chapter_event_ids
                    if event_id in source_event_by_id
                ]
                sequence_count = len(sequence_durations)
                sequence_key = str(chapter_index)
                sequences = checkpoint["chapter_sequences"].get(sequence_key)
                sequence_schema = {
                    "type": "array",
                    "minItems": sequence_count,
                    "maxItems": sequence_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "sequence": {"type": "integer"},
                            "title": {"type": "string"},
                            "location_id": {"type": "string"},
                            "location_time": {"type": "string"},
                            "objective": {"type": "string"},
                            "opening_state": {"type": "string"},
                            "closing_state": {"type": "string"},
                            "causal_handoff": {"type": "string"},
                            "persistent_state": {"type": "string"},
                            "character_state_changes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 30,
                            },
                            "cast_present": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 30,
                            },
                            "recurring_motif_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 12,
                            },
                            "dialogue_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": len(chapter_dialogue),
                            },
                            "source_event_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": len(chapter_events),
                            },
                            "dialogue_mode": {
                                "type": "string",
                                "enum": [
                                    "silent", "visual", "sparse", "natural",
                                    "dialogue_forward",
                                ],
                            },
                            "dialogue_target_turns": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 24,
                            },
                            "dialogue_target_words": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 180,
                            },
                        },
                        "required": [
                            "sequence", "title", "location_id", "location_time", "objective",
                            "opening_state", "closing_state",
                            "causal_handoff", "persistent_state",
                            "character_state_changes", "cast_present",
                            "recurring_motif_ids",
                            "dialogue_ids", "source_event_ids", "dialogue_mode",
                            "dialogue_target_turns", "dialogue_target_words",
                        ],
                        "additionalProperties": False,
                    },
                }
                if not isinstance(sequences, list) or len(sequences) != sequence_count:
                    self._emit_planning_progress(
                        message=(
                            f"Planning chapter {chapter_index}/{chapter_count} "
                            f"into {sequence_count} bounded sequences..."
                        ),
                        current=completed_count,
                        total=total_sequences,
                        stage="long_form_sequence_outline",
                        chapter=chapter_index,
                        chapter_count=chapter_count,
                        sequence=0,
                        sequence_count=sequence_count,
                    )
                    local_geometry = "\n".join(
                        f"- Sequence {index + 1}: {duration} seconds"
                        for index, duration in enumerate(sequence_durations)
                    )
                    chapter_bible_context = format_long_form_story_bible(
                        story_bible,
                        cast_names=chapter.get("cast_present") or [],
                        location_ids=[
                            value.get("location_id")
                            for value in (
                                previous_chapter,
                                chapter,
                                next_chapter,
                            )
                            if isinstance(value, dict)
                            and value.get("location_id")
                        ],
                    )
                    try:
                        sequences = self._call_llm_json(
                            user_prompt=(
                                f"Divide chapter {chapter_index} into exactly "
                                f"{sequence_count} causal screenplay sequences.\n\n"
                                f"SEQUENCE GEOMETRY:\n{local_geometry}\n\n"
                                f"BINDING COMPLETE-FILM STORY BIBLE:\n"
                                f"{chapter_bible_context}\n\n"
                                f"CHAPTER PLAN:\n"
                                f"{json.dumps(chapter, ensure_ascii=False, indent=2)}\n\n"
                                f"PREVIOUS CHAPTER:\n"
                                f"{json.dumps(previous_chapter, ensure_ascii=False, indent=2) if previous_chapter else 'Opening chapter.'}\n\n"
                                f"NEXT CHAPTER:\n"
                                f"{json.dumps(next_chapter, ensure_ascii=False, indent=2) if next_chapter else 'Final chapter.'}\n\n"
                                "Assign every listed source_event_id and dialogue_id "
                                "to exactly one sequence in source order. Assign each "
                                "chapter recurring_motif_id to exactly one sequence, "
                                "where its attached event and dialogue are performed as "
                                "this chapter's unique variation. Each "
                                "sequence begins on the visible "
                                "result of the prior sequence and ends with a "
                                "concrete cause for the next. Never recap or preview. "
                                "Record every named death, disappearance, injury, "
                                "transformation, return, or restoration in "
                                "character_state_changes as 'Name: concrete new state'. "
                                "Respect inherited_character_state and never place an "
                                "unavailable character on screen before restoration. "
                                "Set dialogue_mode and realistic dialogue targets for "
                                "this sequence's dramatic job: visual means dialogue "
                                "may occur but has no minimum; silent explicitly forbids "
                                "speech; sparse, natural, and dialogue_forward progressively "
                                "increase performed conversation. Targets include locked "
                                "lines and must leave time for visible action. Do not infer "
                                "dialogue from the registry labels themselves. Never mention "
                                "a D# or E# identifier in prose unless that same sequence "
                                "owns the identifier.\n\n"
                                f"CHAPTER SOURCE EVENT REGISTRY:\n"
                                f"{json.dumps(chapter_events, ensure_ascii=False, indent=2)}\n\n"
                                f"CHAPTER DIALOGUE REGISTRY:\n"
                                f"{json.dumps([{'dialogue_id': item.get('dialogue_id'), 'speaker': item.get('speaker'), 'exact_text': item.get('text')} for item in chapter_dialogue], ensure_ascii=False, indent=2)}"
                            ),
                            system_prompt=(
                                "You are Maestro's sequence architect. Return only "
                                "the requested JSON array. Plan causally connected "
                                "bounded sequences, not miniature restarts."
                            ),
                            max_tokens=max(2200, sequence_count * 520),
                            thinking_budget=0,
                            temperature=0.42,
                            image_paths=None,
                            json_schema=sequence_schema,
                        )
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        print(
                            "[ShortFilmPlanner] Long-form sequence outline "
                            f"failed for chapter {chapter_index}; using its "
                            f"deterministic sequence scaffold ({exc})."
                        )
                        sequences = []
                if not isinstance(sequences, list) or len(sequences) != sequence_count:
                    sequences = [
                        {
                            "sequence": index + 1,
                            "title": f"{chapter.get('title') or f'Chapter {chapter_index}'} — sequence {index + 1}",
                            "location_id": (
                                chapter.get("location_id")
                                or f"chapter_{chapter_index}_location"
                            ),
                            "location_time": (
                                chapter.get("location_time")
                                or "Continue the chapter's established place and time"
                            ),
                            "objective": (
                                f"Advance chapter {chapter_index} through a new "
                                f"visible beat {index + 1} of {sequence_count}"
                            ),
                            "opening_state": (
                                chapter.get("opening_state")
                                if index == 0 else
                                f"The visible result of sequence {index}"
                            ),
                            "closing_state": (
                                chapter.get("closing_state")
                                if index + 1 == sequence_count else
                                f"A concrete cause launches sequence {index + 2}"
                            ),
                            "causal_handoff": "Carry the completed visible state forward",
                            "persistent_state": chapter.get("persistent_state") or "Preserve established state",
                            "character_state_changes": [],
                            "cast_present": list(chapter.get("cast_present") or []),
                            "recurring_motif_ids": [],
                            "dialogue_ids": [],
                            "source_event_ids": [],
                            "dialogue_mode": "visual",
                            "dialogue_target_turns": 0,
                            "dialogue_target_words": 0,
                        }
                        for index in range(sequence_count)
                    ]
                sequences = _normalize_long_form_event_ownership(
                    sequences,
                    source_events=chapter_events,
                )
                sequences = _normalize_long_form_dialogue_ownership(
                    sequences,
                    allowed_dialogue=chapter_dialogue,
                    source_events=chapter_events,
                )
                sequences = place_chapter_motifs_in_sequences(
                    sequences,
                    chapter=chapter,
                    story_bible=story_bible,
                )
                chapter_state_changes = [
                    str(value or "").strip()
                    for value in chapter.get("character_state_changes") or []
                    if str(value or "").strip()
                ]
                represented_state_changes = {
                    str(value or "").strip().casefold()
                    for row in sequences
                    for value in row.get("character_state_changes") or []
                    if str(value or "").strip()
                }
                if sequences:
                    final_sequence_changes = list(
                        sequences[-1].get("character_state_changes") or []
                    )
                    for change in chapter_state_changes:
                        if change.casefold() not in represented_state_changes:
                            final_sequence_changes.append(change)
                    sequences[-1]["character_state_changes"] = (
                        final_sequence_changes
                    )
                for local_index, row in enumerate(sequences):
                    row["sequence"] = local_index + 1
                    row["location_id"] = (
                        row.get("location_id")
                        or chapter.get("location_id")
                        or f"chapter_{chapter_index}_location"
                    )
                    row["cast_present"] = list(
                        row.get("cast_present")
                        or chapter.get("cast_present")
                        or []
                    )
                    prior = sequences[local_index - 1] if local_index else None
                    inherited = (
                        "; ".join(filter(None, [
                            str(prior.get("closing_state") or "").strip(),
                            str(prior.get("persistent_state") or "").strip(),
                        ]))
                        if prior else
                        str(chapter.get("opening_state") or "").strip()
                    )
                    row["inherited_state"] = (
                        inherited
                        or "The complete visible state established immediately before this sequence"
                    )
                sequences = normalize_long_form_sequence_states(
                    sequences,
                    story_bible=story_bible,
                    inherited_unavailable=(
                        chapter.get("_character_availability_before") or {}
                    ),
                )
                sequences = _normalize_long_form_plan_references(
                    sequences,
                    allowed_dialogue=chapter_dialogue,
                    source_events=chapter_events,
                )
                sequences = _normalize_long_form_dialogue_targets(
                    sequences,
                    durations=sequence_durations,
                    allowed_dialogue=chapter_dialogue,
                    source_events=chapter_events,
                )
                checkpoint["chapter_sequences"][sequence_key] = sequences
                checkpoint["stage"] = "sequence_outlines_ready"
                self._publish_planning_checkpoint(checkpoint)

                for sequence_index, (sequence, sequence_duration) in enumerate(
                    zip(sequences, sequence_durations),
                    start=1,
                ):
                    completed_key = f"{chapter_index}:{sequence_index}"
                    cached = completed_sequences.get(completed_key)
                    if isinstance(cached, dict) and cached.get("shots"):
                        sequence_shots = [
                            ShotPlan.from_dict(item)
                            for item in cached.get("shots") or []
                            if isinstance(item, dict)
                        ]
                        sequence_title = cached.get("title")
                    else:
                        self._emit_planning_progress(
                            message=(
                                f"Writing chapter {chapter_index}/{chapter_count}, "
                                f"sequence {sequence_index}/{sequence_count}..."
                            ),
                            current=completed_count,
                            total=total_sequences,
                            stage="long_form_sequence_screenplay",
                            chapter=chapter_index,
                            chapter_count=chapter_count,
                            sequence=sequence_index,
                            sequence_count=sequence_count,
                        )
                        sequence_dialogue_ids = [
                            str(value or "").upper()
                            for value in sequence.get("dialogue_ids") or []
                        ]
                        sequence_event_ids = [
                            str(value or "").upper()
                            for value in sequence.get("source_event_ids") or []
                        ]
                        previous_sequence = (
                            sequences[sequence_index - 2]
                            if sequence_index > 1 else None
                        )
                        next_sequence = (
                            sequences[sequence_index]
                            if sequence_index < sequence_count else None
                        )
                        sequence_bible_context = format_long_form_story_bible(
                            story_bible,
                            cast_names=sequence.get("cast_present") or [],
                            location_ids=[
                                value.get("location_id")
                                for value in (
                                    previous_sequence,
                                    sequence,
                                    next_sequence,
                                )
                                if isinstance(value, dict)
                                and value.get("location_id")
                            ],
                        )
                        sequence_story = (
                            "LONG-FORM PRODUCTION CONTRACT\n"
                            f"This is chapter {chapter_index} of {chapter_count}, "
                            f"sequence {sequence_index} of {sequence_count}, in "
                            "one continuous film. Write and direct only this "
                            "bounded sequence. Do not restart, recap, preview a "
                            "later sequence, or turn it into a standalone short.\n\n"
                            "GLOBAL USER CONCEPT — dialogue placeholders retain "
                            "story order. It is global context only; enact only "
                            "the source_event_ids owned below:\n"
                            f"{redacted_story}\n\n"
                            "BINDING COMPLETE-FILM STORY BIBLE — preserve its "
                            "premise engine, canonical cast and places, recurring "
                            "motifs, tone, state rules, and ending:\n"
                            f"{sequence_bible_context}\n\n"
                            f"BINDING CHAPTER:\n"
                            f"{_format_long_form_plan_row(chapter)}\n\n"
                            f"BINDING CURRENT SEQUENCE:\n"
                            f"{_format_long_form_plan_row(sequence)}\n\n"
                            f"PREVIOUS SEQUENCE RESULT:\n"
                            f"{_format_long_form_plan_row(previous_sequence or previous_chapter) if (previous_sequence or previous_chapter) else 'This is the film opening.'}\n\n"
                            f"NEXT SEQUENCE — HANDOFF ONLY:\n"
                            f"{_format_long_form_plan_row(next_sequence or next_chapter) if (next_sequence or next_chapter) else 'This is the final sequence; complete the requested outcome.'}\n\n"
                            "IMMUTABLE SOURCE EVENTS OWNED ONLY BY THIS SEQUENCE "
                            "— enact each once, in order. Events assigned to "
                            "other sequences are context, not action:\n"
                            f"{_format_long_form_source_events(sequence_event_ids, source_events)}\n\n"
                            "LOCKED USER DIALOGUE OWNED ONLY BY THIS SEQUENCE — "
                            "place each line once in canonical screenplay format, "
                            "with its assigned speaker and exact words:\n"
                            f"{_format_long_form_locked_dialogue(sequence_dialogue_ids, locked_dialogue)}"
                        )
                        sequence_target_scenes = None
                        if target_scenes is not None:
                            sequence_target_scenes = max(
                                2,
                                int(round(
                                    target_scenes
                                    * sequence_duration
                                    / target_duration
                                )),
                            )
                        owned_event_texts = [
                            str(source_event_by_id[event_id].get("text") or "").strip()
                            for event_id in sequence_event_ids
                            if event_id in source_event_by_id
                            and str(source_event_by_id[event_id].get("text") or "").strip()
                        ]
                        self._long_form_story_blueprint_override = [{
                            "scene_number": 1,
                            "location_time": (
                                str(sequence.get("location_time") or "").strip()
                                or "Continue the established location and story time"
                            ),
                            "active_objective": (
                                str(sequence.get("objective") or "").strip()
                                or "Advance the current chapter objective"
                            ),
                            "story_purpose": (
                                str(sequence.get("objective") or "").strip()
                                or "Advance this bounded part of the film"
                            ),
                            "opening_cause": (
                                str(sequence.get("opening_state") or "").strip()
                                or "Open on the visible result of the preceding sequence"
                            ),
                            "visible_beats": (
                                owned_event_texts
                                or [str(sequence.get("objective") or "Advance the sequence")]
                            ),
                            "choice_or_discovery": (
                                str(sequence.get("closing_state") or "").strip()
                                or "The sequence produces its required visible change"
                            ),
                            "outgoing_handoff": (
                                str(sequence.get("causal_handoff") or "").strip()
                                or str(sequence.get("closing_state") or "").strip()
                                or "The ending visibly causes the next sequence"
                            ),
                            "persistent_state_after": (
                                "; ".join(filter(None, [
                                    str(sequence.get("persistent_state") or "").strip(),
                                    *(
                                        str(value or "").strip()
                                        for value in sequence.get(
                                            "character_state_changes",
                                            [],
                                        )
                                    ),
                                ]))
                                or "Preserve every established identity, prop, relationship, and physical state"
                            ),
                        }]
                        sequence_shots, sequence_title = self._plan_story_driven(
                            story_description=sequence_story,
                            reference_image_path=reference_image_path,
                            char_profiles=char_profiles,
                            has_reference=has_reference,
                            target_duration=sequence_duration,
                            target_scenes=sequence_target_scenes,
                            narrative_mode=narrative_mode,
                            fps=fps,
                            frames_steps=frames_steps,
                            frames_minimum=frames_minimum,
                            frames_maximum=frames_maximum,
                            nsfw=nsfw,
                            polish_block=polish_block,
                            multishot_lora_mode=multishot_lora_mode,
                            dialogue_density_override={
                                "mode": sequence.get("dialogue_mode") or "visual",
                                "minimum_turns": int(
                                    sequence.get("dialogue_target_turns") or 0
                                ),
                                "minimum_words": int(
                                    sequence.get("dialogue_target_words") or 0
                                ),
                            },
                            dialogue_intent_text="\n".join([
                                *owned_event_texts,
                                str(sequence.get("objective") or ""),
                                str(sequence.get("opening_state") or ""),
                                str(sequence.get("closing_state") or ""),
                            ]),
                        )
                        completed_sequences[completed_key] = {
                            "title": sequence_title,
                            "shots": [shot.to_dict() for shot in sequence_shots],
                        }
                        checkpoint["stage"] = "sequence_complete"
                        checkpoint["completed_sequence_count"] = (
                            completed_count + 1
                        )
                        checkpoint["current_chapter"] = chapter_index
                        checkpoint["current_sequence"] = sequence_index
                        self._publish_planning_checkpoint(checkpoint)

                    if first_title is None and sequence_title:
                        first_title = sequence_title
                    for shot in sequence_shots:
                        global_index = len(all_shots)
                        shot.index = global_index
                        shot.shot_id = self._make_shot_id(global_index, "sf")
                        metadata = dict(shot.metadata or {})
                        metadata.update({
                            "long_form_chapter": chapter_index,
                            "long_form_chapter_count": chapter_count,
                            "long_form_chapter_title": chapter.get("title"),
                            "long_form_sequence": sequence_index,
                            "long_form_sequence_count": sequence_count,
                            "long_form_sequence_title": sequence.get("title"),
                            "long_form_causal_handoff": sequence.get("causal_handoff"),
                            "long_form_persistent_state": sequence.get("persistent_state"),
                            "long_form_inherited_character_state": sequence.get(
                                "inherited_character_state"
                            ),
                            "long_form_character_availability_before": dict(
                                sequence.get("_character_availability_before") or {}
                            ),
                            "long_form_character_state_changes": list(
                                sequence.get("character_state_changes") or []
                            ),
                            "long_form_location_id": sequence.get("location_id"),
                            "long_form_cast_present": list(
                                sequence.get("cast_present") or []
                            ),
                            "long_form_recurring_motif_ids": list(
                                sequence.get("recurring_motif_ids") or []
                            ),
                            "long_form_story_bible_revision": (
                                LONG_FORM_STORY_BIBLE_REVISION
                            ),
                        })
                        shot.metadata = metadata
                        all_shots.append(shot)
                    completed_count += 1
                    self._emit_planning_progress(
                        message=(
                            f"Planned {completed_count}/{total_sequences} "
                            "long-form sequences"
                        ),
                        current=completed_count,
                        total=total_sequences,
                        stage="long_form_sequence_complete",
                        chapter=chapter_index,
                        chapter_count=chapter_count,
                        sequence=sequence_index,
                        sequence_count=sequence_count,
                    )
        finally:
            for attribute in (
                "_long_form_story_blueprint_override",
                "_long_form_story_bible_override",
                "_long_form_screenplay_thinking_budget_override",
                "_long_form_h3_voice_bible_override",
                "_long_form_voice_bible_max_items",
            ):
                if hasattr(self, attribute):
                    delattr(self, attribute)

        # Run one non-creative whole-film audit after assembly.  It never asks
        # the LLM to rewrite hours of completed work and therefore cannot turn
        # a late typo into another multi-hour planning restart.  Conservative
        # speaker cleanup also covers shots restored from an older checkpoint.
        final_dicts, final_quality_warnings = sanitize_long_form_shot_dicts(
            [shot.to_dict() for shot in all_shots],
            story_bible=story_bible,
        )
        all_shots = [ShotPlan.from_dict(item) for item in final_dicts]
        for index, shot in enumerate(all_shots):
            shot.index = index
            shot.shot_id = self._make_shot_id(index, "sf")
            metadata = dict(shot.metadata or {})
            if final_quality_warnings:
                metadata["long_form_quality_warnings"] = list(
                    final_quality_warnings
                )
            shot.metadata = metadata
        quality_report = audit_long_form_plan(
            final_dicts,
            story_bible=story_bible,
            target_duration=target_duration,
        )
        quality_report["repairs"] = list(final_quality_warnings)

        checkpoint["stage"] = "complete"
        checkpoint["completed_sequence_count"] = total_sequences
        checkpoint["quality_report"] = quality_report
        checkpoint["complete"] = True
        self._publish_planning_checkpoint(checkpoint)
        self._last_title = first_title
        return all_shots, first_title

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

    def _build_story_continuity_blueprint(
        self,
        *,
        story_description: str,
        char_profiles: list[CharacterProfile],
        target_duration: int,
        minimum_scenes: int,
        maximum_scenes: int,
        nsfw: bool = False,
    ) -> list[dict[str, Any]]:
        """Plan one causal film before screenplay prose and shot coverage.

        Director previously asked one pass to discover the plot, write natural
        dialogue, obey a hard runtime, and format a screenplay simultaneously.
        Small local models could satisfy the dialogue and formatting contracts
        while silently turning the middle into disconnected location vignettes.
        This compact structured pass gives the writer and director the same
        explicit scene chain without making the video model reason about it.
        """

        if not (self._generate or self._generate_streaming):
            return []
        from ..nsfw_guidance import inject_nsfw_if_enabled

        supplied_characters = "\n".join(
            f"- {profile.display_name or profile.id}: "
            f"{profile.physical_description}"
            for profile in char_profiles or []
        ) or "- No separate character cards were supplied."
        system_prompt = f"""You are Maestro's story architect. Build the causal scene chain for ONE complete {target_duration}-second short film before another writer turns it into screenplay prose.

Return ONLY a JSON array containing {minimum_scenes}-{maximum_scenes} scene objects. This is story architecture, not a shot list and not a screenplay. Do not write camera coverage or dialogue prose.

ONE FILM, NOT DISCONNECTED VIGNETTES:
- Preserve every requested event, outcome, relationship, named identity, location requirement, and literal user-authored line in its original order. Do not replace the user's plot with a different premise.
- Establish one central dramatic objective or conflict. Every scene must change that same story by causing a decision, discovery, obstacle, consequence, escalation, climax, or resolution.
- If the user leaves a broad act underspecified (for example, "they team up and fight crime"), invent the SMALLEST coherent plot needed to complete it: one case, one objective, and one escalating chain. Do not fill runtime with unrelated criminals, emergencies, villains, locations, or sketches.
- Scene N+1 must happen BECAUSE OF something visible or spoken in scene N. Apply this removal test: if scene N could be deleted without changing why scene N+1 happens, repair the handoff.
- A location or time change is welcome when dramatically motivated. The outgoing scene must establish the decision, discovery, pursuit, dispatch, departure, consequence, or time transition; the next scene must open on its concrete result or arrival. Never teleport merely for visual variety.
- Carry persistent state forward: current objective, information learned, relationship changes, injuries, dirt or wardrobe damage, important props, who possesses them, and unresolved danger.
- Escalate toward one climax, then pay off both the external objective and the central relationship or character change. The final scene must resolve this film, not advertise an unrelated sequel.
- Use only camera-observable physical events in visible_beats. Dialogue will be authored in the next pass.
- Do not use generic placeholders such as "the story continues," "they move on," "next scene," or "another incident occurs." State the concrete cause and visible result.

FIELD CONTRACT:
- location_time: exact physical place and story time.
- active_objective: what the central character or team is trying to achieve NOW.
- story_purpose: the unique narrative change this scene earns.
- opening_cause: for scene 1, the initiating situation; afterward, the exact prior action, choice, discovery, or consequence that causes this scene and how its opening visibly shows the result.
- visible_beats: ordered camera-observable actions that deliver the scene.
- choice_or_discovery: the new choice, information, reversal, or consequence produced here.
- outgoing_handoff: for every non-final scene, the concrete on-screen beat that motivates the next scene; for the final scene, the completed payoff.
- persistent_state_after: the objective, knowledge, relationship, physical damage, wardrobe, and prop state that the next scene inherits."""
        system_prompt = inject_nsfw_if_enabled(
            system_prompt,
            nsfw,
            "screenplay",
        )
        user_prompt = f"""/no_think

Create the binding causal scene blueprint for this short film.

USER CONCEPT — SOURCE OF TRUTH:
{story_description}

SUPPLIED CHARACTER CARDS:
{supplied_characters}

Target runtime: {target_duration} seconds.
Return {minimum_scenes}-{maximum_scenes} scenes. Preserve the user's ordered events, and make every location change and plot beat causally earned."""
        try:
            rows = self._call_llm_json(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=max(3072, maximum_scenes * 640),
                # This is a closed structured planning artifact. Keep Qwen's
                # thinking disabled so reasoning cannot consume the JSON body.
                thinking_budget=0,
                temperature=0.55,
                streaming=True,
                frequency_penalty=0.16,
                presence_penalty=0.06,
                json_schema=_story_continuity_blueprint_schema(
                    minimum_scenes,
                    maximum_scenes,
                ),
            )
            blueprint = _normalize_story_continuity_blueprint(
                rows,
                minimum_scenes=minimum_scenes,
                maximum_scenes=maximum_scenes,
            )
            if blueprint:
                print(
                    "[ShortFilmPlanner] Story architect locked a causal "
                    f"chain of {len(blueprint)} scene(s)."
                )
                return blueprint
            print(
                "[ShortFilmPlanner] Story architect returned an incomplete "
                "scene chain; screenplay pass will use the causal rules "
                "directly."
            )
        except Exception as exc:
            print(
                "[ShortFilmPlanner] Story architect pass was unavailable; "
                f"screenplay pass will use the causal rules directly ({exc})."
            )
        return []

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

For an established fictional character named by the user, use the character's established personality, vocabulary, sentence rhythm, comic or dramatic behavior, and relationships to the other supplied characters. Capture why the character is recognizable beyond a single stereotype. Copyright-safe originality means fresh dialogue, not bland dialogue: do not quote or reproduce signature lines or catchphrases, but do preserve the character's recognizable decision-making and conversational behavior.

CASTING IS NOT CHARACTERIZATION: When the concept says "Actor as Character," the actor supplies visual casting and screen presence while the fictional Character supplies biography, nationality, accent, vocabulary, slang, relationships, and behavior. Never transfer an actor's real nationality, accent, public persona, or personal speech habits into the role unless the user explicitly requests that adaptation. If a performer is named without playing a fictional role, follow the user's description rather than inventing private traits.

For an original character, infer only what the concept and supplied character card support. Do not invent a biography that changes the story. Do not list a TV series, franchise, location, or group as a character.

Fields:
- character_name: the exact supplied character name.
- personality_engine: concrete wants, pressure responses, defenses, habits, contradictions, and choices that shape this character. Describe playable behavior, not an abstract theme such as order, chaos, perfection, or grit.
- speech_pattern: concrete vocabulary level, syntax, rhythm, interruptions, formality, subtext, humor, and recurring conversational tactics. Describe how the character constructs and avoids thoughts, not a fixed sentence or word count.
- relationship_behavior: how this character specifically pursues, resists, needles, reassures, lies to, protects, or reacts to each other supplied cast member.
- performance_direction: concise audible cadence, energy, register, and emotional delivery guidance; describe qualities, not an actor voice clone.
- avoid: generic caricatures, vocabulary this person would not use, and other out-of-character failure modes.

VOICE-BIBLE QUALITY RULES:
- Silently inhabit each character in the exact project situation before writing the profile: what do they want right now, what are they unwilling to admit, and what verbal tactic do they use on the other person? Output the profile, not the rehearsal and not sample dialogue.
- Character traits are ACTING INSTRUCTIONS, never proposed conversation topics. A disciplined character behaves and speaks decisively; they do not explain "consistency." A rough character chooses blunt concrete words; they do not lecture about "chaos" or "grit."
- Prefer concrete, speakable language over psychological, academic, corporate, or thesaurus language. Record the tempting-but-wrong register in avoid so the screenplay writer can reject it.
- Apply the name-hidden test: the eventual wording, tactics, and reactions should identify the speaker even with the character heading removed.
- Never prescribe a numerical word range, maximum word count, or constant one-word/two-word response pattern for a character.
- "Terse," "direct," or "economical" describes diction and cadence, not permanently tiny lines. Every character must be able to express complete responsive thoughts when the dramatic beat needs them.
- Preserve recognizable differences through vocabulary, syntax, subtext, conversational tactics, and relationships. Do not manufacture distinctness by reducing everyone to fragments or catchphrases.
- Encourage varied line lengths across an exchange: clipped reactions where natural, fuller responses where meaning, comedy, conflict, or emotion needs room.

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
                max_tokens=min(
                    10000,
                    max(
                        3072,
                        1200 + int(getattr(
                            self,
                            "_long_form_voice_bible_max_items",
                            8,
                        )) * 220,
                    ),
                ),
                # Model-aware structured output: Qwen runs this directly with
                # grammar; Gemma may retain its proven compact reasoning path.
                thinking_budget=None,
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
        """Polish spoken words in bounded batches before the manifest locks."""

        if not manifest or not (self._generate or self._generate_streaming):
            return manifest
        manifest = copy.deepcopy(manifest)
        locked = _h3_user_locked_dialogue_fingerprints(story_description)
        bible_text = _format_h3_voice_bible(voice_bible) or (
            "No structured voice bible was available. Infer distinct speech "
            "only from the project concept and screenplay context."
        )
        system_prompt = f"""You are the H3 CHARACTER TABLE-READ editor. Improve only the dialogue of an already structured screenplay. This is an actor's table read, not a thesaurus rewrite.

Return ONLY one JSON array row for every turn in the supplied MANIFEST BATCH. Keep the same global turn number, speaker, order, intent, plot facts, and conversational response relationship. Do not add or remove turns. original_text must be copied exactly into the corresponding output row. The full screenplay is context only; never return rows for dialogue turns outside the supplied batch.

Make each revised_text sound unmistakably appropriate to that character: established personality, vocabulary, syntax, cadence, comic or dramatic mechanism, and relationship to the person being addressed. Preserve nuance; do not reduce a character to one exaggerated trait. Write fresh dialogue and never copy famous lines or catchphrases.

Before revising each turn, silently inhabit the speaker and rehearse the exchange from inside their point of view: What do they want from the listener right now? What are they hiding or refusing to say directly? Do they attack, dodge, charm, tease, reassure, lecture, deflect, or go quiet? How do they react to the exact preceding line or visible action? Then write only revised_text and delivery.

CHARACTER-VOICE ACCEPTANCE TEST:
- Apply the name-hidden recognition test: with speaker_name hidden, a viewer familiar with the character should still recognize who chose those words and that conversational tactic.
- Express personality THROUGH diction, rhythm, subtext, reaction, and behavior. Never make characters discuss voice-bible abstractions such as their own order, chaos, consistency, perfection, grit, worldview, or personality unless the story itself requires that literal topic.
- Plain spoken words beat elevated synonyms. Never replace "good enough" with "adequate," "problem" with "core issue," or ordinary banter with academic, corporate, therapeutic, technical, or production-design language merely to make it different.
- Preserve a natural original line when it already passes the test. revised_text may exactly equal original_text. Change only what becomes more speakable, responsive, and character-authentic.
- Keep the character's intelligence and nuance, but do not confuse intelligence with formality. Read every revision aloud mentally; if the person would not actually say it under this pressure, rewrite it.

If user_locked is true, revised_text MUST exactly equal original_text. Otherwise tighten stiff, formal, generic, fragmentary, or AI-like phrasing while preserving meaning, plot facts, and the line's immediate dramatic job. A terse character may use economical diction, but "terse" never imposes a fixed word count. Vary line lengths, use complete responsive thoughts, preserve subtext, and let each reply react specifically to what was just said or done. Do not turn an exchange into a chain of one-word or two-word fragments. Keep the batch within its stated spoken-word budget and never make an individual turn longer than {maximum_line_words} words.

delivery is a concise performance direction for that specific line. Describe cadence, energy, pitch/register, hesitation, interruption, or emotional pressure. Do not request an exact actor voice or voice impersonation."""

        total_turns = len(manifest)
        locked_total_words = sum(
            len(_h3_plain_dialogue_text(entry.get("spoken_text")).split())
            for entry in manifest
            if _h3_dialogue_word_fingerprint(entry.get("spoken_text")) in locked
        )
        allowed_total_words = max(
            max(0, int(max_spoken_words or 0)),
            locked_total_words,
        )
        initial_metrics = _h3_dialogue_quality_metrics(
            manifest,
            story_description=story_description,
            maximum_line_words=maximum_line_words,
        )
        if initial_metrics["issues"]:
            print(
                "[ShortFilmPlanner] H3 dialogue quality repair requested: "
                + "; ".join(initial_metrics["issues"])
            )

        def run_batches(
            source_manifest: list[dict[str, Any]],
            *,
            quality_issues: list[str],
            pass_label: str,
            only_problem_turns: Optional[set[int]] = None,
        ) -> tuple[list[dict[str, Any]], int]:
            revised_manifest = copy.deepcopy(source_manifest)
            source_total_words = sum(
                len(_h3_plain_dialogue_text(entry.get("spoken_text")).split())
                for entry in source_manifest
            )
            remaining_capacity = max(
                0,
                allowed_total_words - source_total_words,
            )
            successful_batches = 0
            batch_count = int(math.ceil(
                total_turns / max(1, _H3_TABLE_READ_CHUNK_SIZE)
            ))
            for start in range(0, total_turns, _H3_TABLE_READ_CHUNK_SIZE):
                end = min(total_turns, start + _H3_TABLE_READ_CHUNK_SIZE)
                global_turns = set(range(start + 1, end + 1))
                if (
                    only_problem_turns is not None
                    and not (global_turns & only_problem_turns)
                ):
                    continue
                chunk = source_manifest[start:end]
                payload = [
                    {
                        "turn": index,
                        "speaker_name": entry.get("speaker_name") or "speaker",
                        "original_text": entry.get("spoken_text") or "",
                        "previous_turn": (
                            {
                                "speaker_name": manifest[index - 2].get(
                                    "speaker_name"
                                ) or "speaker",
                                "spoken_text": manifest[index - 2].get(
                                    "spoken_text"
                                ) or "",
                            }
                            if index > 1 else None
                        ),
                        "user_locked": (
                            _h3_dialogue_word_fingerprint(entry.get("spoken_text"))
                            in locked
                        ),
                    }
                    for index, entry in enumerate(chunk, start=start + 1)
                ]
                chunk_original_words = sum(
                    len(_h3_plain_dialogue_text(entry.get("spoken_text")).split())
                    for entry in chunk
                )
                # Divide any available expansion room deterministically by
                # turn count. The integer prefix calculation guarantees all
                # batch allowances sum to no more than the global budget.
                capacity_before = (remaining_capacity * start) // total_turns
                capacity_after = (remaining_capacity * end) // total_turns
                chunk_word_budget = (
                    chunk_original_words
                    + capacity_after
                    - capacity_before
                )
                issue_block = (
                    "QUALITY REPAIR REQUIRED:\n- "
                    + "\n- ".join(quality_issues)
                    + "\nExpand or reshape generated fragmentary turns where "
                    "needed while keeping the exchange natural and within budget."
                    if quality_issues else
                    "QUALITY CHECK: Preserve natural variety and improve only lines "
                    "that are stiff, generic, or out of character."
                )
                user_prompt = f"""Perform a dialogue-only table read for this H3 Director screenplay.

TABLE-READ PASS: {pass_label}
DIALOGUE TURN BATCH: global turns {start + 1}-{end} of {total_turns} (batch {start // _H3_TABLE_READ_CHUNK_SIZE + 1}/{batch_count})

PROJECT CONCEPT:
{story_description}

CHARACTER VOICE BIBLE (binding for character identity, not line length):
{bible_text}

{issue_block}

MAXIMUM SPOKEN WORDS IN THIS BATCH: {chunk_word_budget}
MAXIMUM WORDS IN ANY ONE TURN: {maximum_line_words}

DIALOGUE TURN MANIFEST BATCH:
{json.dumps(payload, ensure_ascii=False, indent=2)}

FULL SCREENPLAY FOR ACTION AND RELATIONSHIP CONTEXT ONLY:
{screenplay}"""
                try:
                    rows = self._call_llm_json(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_tokens=max(
                            1536,
                            min(4096, len(chunk) * 220 + 512),
                        ),
                        # This is an exact schema transformation, not a
                        # creative planning pass. Thinking previously consumed
                        # the response ceiling before all rows were emitted.
                        thinking_budget=0,
                        temperature=0.65,
                        streaming=True,
                        frequency_penalty=0.12,
                        presence_penalty=0.04,
                        json_schema=_h3_table_read_schema(len(chunk)),
                    )
                    revised_chunk, _ = _apply_h3_character_table_read(
                        chunk,
                        rows,
                        story_description=story_description,
                        max_spoken_words=chunk_word_budget,
                        maximum_line_words=maximum_line_words,
                        turn_offset=start,
                    )
                    revised_manifest[start:end] = revised_chunk
                    successful_batches += 1
                except Exception as exc:
                    # Keep only this batch's screenplay lines. Validated work
                    # from earlier/later batches remains usable.
                    print(
                        "[ShortFilmPlanner] H3 table-read batch "
                        f"{start // _H3_TABLE_READ_CHUNK_SIZE + 1}/{batch_count} "
                        f"failed validation; preserving its {len(chunk)} original "
                        f"turn(s) ({exc})."
                    )
            return revised_manifest, successful_batches

        revised, successful_batches = run_batches(
            manifest,
            quality_issues=list(initial_metrics["issues"]),
            pass_label="primary character table read",
        )
        revised_metrics = _h3_dialogue_quality_metrics(
            revised,
            story_description=story_description,
            maximum_line_words=maximum_line_words,
        )

        # One focused retry is allowed only when an objective collapse signal
        # remains. Re-run the small batches that contain problem turns instead
        # of paying for or risking another whole-screenplay transformation.
        if revised_metrics["issues"]:
            problem_turns = set(revised_metrics["problem_turns"])
            if not problem_turns:
                problem_turns = set(range(1, total_turns + 1))
            print(
                "[ShortFilmPlanner] H3 table read retained dialogue-quality "
                "issues; retrying only affected batch(es)."
            )
            retry, retry_batches = run_batches(
                revised,
                quality_issues=list(revised_metrics["issues"]),
                pass_label="targeted dialogue-quality retry",
                only_problem_turns=problem_turns,
            )
            retry_metrics = _h3_dialogue_quality_metrics(
                retry,
                story_description=story_description,
                maximum_line_words=maximum_line_words,
            )
            if retry_metrics["score"] < revised_metrics["score"]:
                revised = retry
                revised_metrics = retry_metrics
                successful_batches += retry_batches
            else:
                print(
                    "[ShortFilmPlanner] Targeted H3 dialogue retry did not "
                    "improve the objective quality score; keeping the primary "
                    "validated table read."
                )

        final_total_words = sum(
            len(_h3_plain_dialogue_text(entry.get("spoken_text")).split())
            for entry in revised
        )
        if final_total_words > allowed_total_words:
            # This should be unreachable once Pass 1's hard pre-lock budget
            # has run, but retain a deterministic final fit. Literal user
            # dialogue remains immutable; only generated lines are shortened.
            print(
                "[ShortFilmPlanner] H3 table read exceeded the global dialogue "
                "budget after batching; fitting generated lines before lock."
            )
            revised = _fit_h3_dialogue_manifest_to_budget(
                revised,
                story_description=story_description,
                max_spoken_words=allowed_total_words,
                maximum_line_words=maximum_line_words,
            )

        if revised_metrics.get("overlong_turns") and successful_batches:
            raise RuntimeError(
                "MiniMax H3 character table read could not shorten generated "
                "dialogue to the native clip limit after its targeted repair. "
                "No video jobs were queued."
            )
        if revised_metrics.get("overlong_turns"):
            print(
                "[ShortFilmPlanner] H3 table read was unavailable for an "
                "overlong generated turn; retaining the established deterministic "
                "shot allocator as the compatibility fallback."
            )

        changed = sum(
            _h3_dialogue_word_fingerprint(after.get("spoken_text"))
            != _h3_dialogue_word_fingerprint(before.get("spoken_text"))
            for before, after in zip(manifest, revised)
        )
        print(
            "[ShortFilmPlanner] H3 character table read validated "
            f"{successful_batches} batch(es), {len(revised)} turn(s), and "
            f"revised {changed}; dialogue is now locked."
        )
        if revised_metrics["issues"]:
            print(
                "[ShortFilmPlanner] H3 dialogue remains intentionally concise "
                "after bounded repair: " + "; ".join(revised_metrics["issues"])
            )
        return revised

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
        _bounded_batch: bool = False,
    ) -> list[ShotPlan]:
        """Plan shots from existing audio-segmented clips."""
        from ..nsfw_guidance import inject_nsfw_if_enabled

        if len(clips) > 12 and not _bounded_batch:
            def call_batch(
                batch_number: int,
                start: int,
                batch_clips: list[dict],
                previous: Optional[dict],
            ) -> list[dict]:
                end = start + len(batch_clips)
                start_sec = float(batch_clips[0].get("start", 0) or 0)
                end_sec = float(
                    batch_clips[-1].get("end", start_sec) or start_sec
                )
                batch_lyrics = [
                    row for row in (lyrics or [])
                    if float(row.get("start", 0) or 0) < end_sec
                    and float(row.get("end", 0) or 0) > start_sec
                ]
                previous_ending = str(
                    (previous or {}).get("ending_beat") or ""
                ).strip()
                batch_story = (
                    f"{story_description}\n\n"
                    "LONG-FORM AUDIO TIMELINE CONTRACT:\n"
                    f"Plan only global clips {start + 1}-{end} of {len(clips)}. "
                    "The audio and transcript timestamps are immutable. Continue "
                    "the same staging, identities, props, and story state; never "
                    "restart or repeat completed action.\n"
                    f"Previous planned ending: "
                    f"{previous_ending or 'No prior clip; establish the opening.'}"
                )
                batch_shots = self._plan_audio_driven(
                    clips=batch_clips,
                    story_description=batch_story,
                    lyrics=batch_lyrics,
                    speaker_mappings=speaker_mappings,
                    reference_image_path=reference_image_path,
                    char_profiles=char_profiles,
                    has_reference=has_reference,
                    nsfw=nsfw,
                    polish_block=polish_block,
                    _bounded_batch=True,
                )
                return [shot.to_dict() for shot in batch_shots]

            def fallback_shot(index: int, clip: dict) -> dict:
                duration = max(
                    0.1,
                    float(clip.get("end", 0) or 0)
                    - float(clip.get("start", 0) or 0),
                )
                return ShotPlan(
                    shot_id=self._make_shot_id(index, "sf"),
                    index=index,
                    duration_sec=duration,
                    skill_type="short_film",
                    scene_goal=f"Continue audio segment {index + 1}",
                    scene_type="dialogue",
                    source_mode_preference="a2v",
                    image_strategy=(
                        "reference_edit" if has_reference
                        else "fresh_generation"
                    ),
                    continuity_strategy=(
                        "continuous" if index else "independent"
                    ),
                    subjects_on_screen=[],
                    spatial_setup="Maintain the established staging",
                    environment="",
                    visual_style="",
                    lighting="",
                    mood="",
                    action_beats=["The visible performance follows the audio"],
                    camera_plan=CameraPlan(framing="medium shot"),
                    audio_plan=AudioPlan(
                        mode="dialogue_driven",
                        timing_anchor="audio",
                        lip_sync_critical=True,
                    ),
                    ending_beat="The performance continues",
                ).to_dict()

            serialized = self._run_checkpointed_json_batches(
                items=clips,
                batch_size=12,
                checkpoint_key="short_film_audio_batches",
                stage="short_film_audio_batch",
                progress_label="audio-film",
                call_batch=call_batch,
                fallback_factory=fallback_shot,
            )
            shots = [ShotPlan.from_dict(row) for row in serialized]
            for index, shot in enumerate(shots):
                shot.index = index
                shot.shot_id = self._make_shot_id(index, "sf")
            return shots

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
        # Long audio timelines arrive here in bounded batches. Keep those
        # structured calls deterministic and inexpensive; short projects retain
        # the model-aware historical default.
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
            thinking_budget=0 if _bounded_batch else None,
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
        dialogue_density_override: Optional[dict[str, Any]] = None,
        dialogue_intent_text: Optional[str] = None,
    ) -> tuple[list[ShotPlan], Optional[str]]:
        """Layered story-driven planning.

        Pass 0A — Story architecture: LLM locks a causal scene chain.
        Pass 0B — Character architecture (H3): LLM builds a voice bible.
        Pass 1 — Screenplay: LLM writes the complete performed story.
        Pass 1.5 — Table read (H3): LLM polishes dialogue without changing plot.
        Pass 2 — Direction: LLM converts the screenplay into executable shots.

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
            assert_no_minor_content_in_pass2,
        )

        if target_scenes is None:
            target_scenes = max(2, min(20, target_duration // 20))

        preserve_names = bool(
            getattr(self, "_preserve_video_character_names", False)
        )
        is_h3_native = str(
            getattr(self, "_video_model", "") or ""
        ).lower().startswith("minimax_h3")
        h3_maximum_line_words = max(
            1,
            int(math.floor(
                (float(frames_maximum or 345) / max(1, fps))
                * _H3_DIALOGUE_WORDS_PER_SECOND
            )),
        )
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

        story_continuity_blueprint: list[dict[str, Any]] = []
        has_long_form_override = hasattr(
            self, "_long_form_story_blueprint_override"
        )
        # One- or two-scene micro-shorts already have an obvious local
        # handoff and do not justify another LLM round trip. Longer films are
        # where disconnected vignette drift appears and need the explicit
        # architecture pass.
        if has_long_form_override:
            story_continuity_blueprint = copy.deepcopy(
                getattr(self, "_long_form_story_blueprint_override") or []
            )
        elif scene_count_high >= 4:
            print(
                "[ShortFilmPlanner] Pass 0A: Building causal story "
                "architecture..."
            )
            story_continuity_blueprint = (
                self._build_story_continuity_blueprint(
                    story_description=story_description,
                    char_profiles=char_profiles,
                    target_duration=target_duration,
                    minimum_scenes=scene_count_low,
                    maximum_scenes=scene_count_high,
                    nsfw=nsfw,
                )
            )
        self._last_story_continuity_blueprint = copy.deepcopy(
            story_continuity_blueprint
        )
        story_blueprint_text = _format_story_continuity_blueprint(
            story_continuity_blueprint
        )
        is_long_form_sequence = bool(
            has_long_form_override and story_continuity_blueprint
        )
        story_continuity_block = f"""
CAUSAL FILM CONTINUITY — BINDING STORY CONTRACT:
- Write ONE coherent film, never a reel of loosely related scenes. Establish a central objective or conflict, escalate it, reach one climax, and earn a resolution.
- Every scene after the first must be caused by an action, choice, discovery, pursuit, dispatch, departure, or consequence in the scene before it. A new scene heading is not permission to teleport.
- When place or time changes, visibly establish the reason before the cut and open the next scene on the resulting arrival or consequence. Location variety is welcome only when the story motivates it.
- End every non-final scene with a concrete handoff that makes the next scene necessary. Apply the deletion test: if removing the prior scene would not change why the next scene happens, repair the transition.
- Carry forward the active objective, knowledge, relationship change, injuries, dirt or wardrobe damage, and important props. A character cannot forget what just happened merely because the camera cut.
- Do not invent an unrelated villain, emergency, crime, location, or subplot to fill runtime. When the user's later act is broad, choose the smallest single conflict that pays off the established setup and relationship.
- Preserve the user's ordered events and literal spoken lines. The structured blueprint may clarify bridges, but it never overrides the user's source concept.
- In screenplay prose, the first camera-observable actions after each new scene heading must show its opening cause/result, and the final actions before the next heading must show its outgoing handoff. Do not print blueprint labels or planning commentary.

BINDING STORY-ARCHITECT BLUEPRINT:
{story_blueprint_text}
"""

        h3_voice_bible: list[dict[str, str]] = []
        h3_character_block = ""
        h3_dialogue_density = None
        long_form_dialogue_density = None
        long_form_dialogue_mode = "visual"
        if is_h3_native:
            if hasattr(self, "_long_form_h3_voice_bible_override"):
                h3_voice_bible = copy.deepcopy(
                    getattr(self, "_long_form_h3_voice_bible_override") or []
                )
                print(
                    "[ShortFilmPlanner] Reusing long-form H3 character "
                    "voice bible."
                )
            else:
                print("[ShortFilmPlanner] Pass 0: Building H3 character voice bible...")
                h3_voice_bible = self._build_h3_character_voice_bible(
                    story_description=story_description,
                    char_profiles=char_profiles,
                )
            voice_bible_text = _format_h3_voice_bible(h3_voice_bible)
            h3_character_block = f"""
H3 CHARACTER-AUTHENTICITY RULES:
- Before drafting, use the binding voice bible below as the cast's dialogue and relationship logic.
- For an established fictional character named by the user, write fresh dialogue consistent with the character's established personality, vocabulary, syntax, cadence, comic/dramatic mechanism, and relationships. Do not copy famous dialogue or catchphrases.
- Treat "Actor as Character" as visual casting, not a personality transplant. The fictional character—not the performer—governs nationality, accent, slang, biography, relationships, and dialogue unless the user explicitly requests a reinterpretation.
- Silently embody the speaker before every generated line. Decide what they want from the listener in that beat, what they will not admit, and the character-specific tactic they use to get it. Write the result, never the rehearsal or an explanation of the profile.
- Character traits are performance engines, not dialogue subjects. Show discipline through decisive choices, anger through pressure and word choice, wit through a targeted reaction, and intelligence through insight. Do not make the cast literally debate abstract profile labels such as order, chaos, consistency, perfection, grit, worldview, or personality unless the user's story makes that the subject.
- Use the name-hidden test: if a generated line could belong to another member of the cast after removing the heading, rewrite its diction, syntax, subtext, or tactic until the speaker is recognizable.
- Prefer plain, speakable, situation-specific words. Do not "improve" dialogue with academic, corporate, therapeutic, technical, or thesaurus phrasing that the character would not choose aloud under pressure.
- Do not reduce a recognizable character to one generic trait. A line that could be reassigned to another cast member without sounding wrong must be rewritten.
- Preserve every literal line supplied by the user exactly. Character-authentic writing changes generated dialogue, never user-authored dialogue.
- Treat the named cast as CLOSED: use only people named in the user's concept, supplied character references, and binding voice bible. Unnamed background extras are allowed when the setting needs them, but never add a named cameo, familiar franchise character, friend, relative, or celebrity the user did not request.
- Every generated spoken turn must be a complete thought of no more than {h3_maximum_line_words} words so it can be performed inside one native H3 clip. Never end a generated turn with a continuation ellipsis and never split one character's sentence across clips; shorten it or turn the next idea into a natural reply from another character.
- Treat "terse," "direct," and "economical" as style guidance, never as a fixed line-length rule. Use varied line lengths and complete responsive thoughts; do not build an exchange from a chain of one-word or two-word fragments.
- Every generated reply must respond specifically to the preceding action or line, carry subtext or character intent, and move the relationship or story forward. Prefer substantial back-and-forth over disconnected labels, counters, and generic acknowledgements.
- Silently conduct a table read before returning the screenplay: remove generic sitcom filler, stiff exposition, invented gimmicks, and words the named speaker would not naturally choose.
"""
            if voice_bible_text:
                h3_character_block += (
                    "\nBINDING CHARACTER VOICE BIBLE:\n" + voice_bible_text
                )
            if dialogue_density_override is not None:
                try:
                    override_turns = max(
                        0,
                        int(dialogue_density_override.get("minimum_turns") or 0),
                    )
                except (TypeError, ValueError):
                    override_turns = 0
                try:
                    override_words = max(
                        0,
                        int(dialogue_density_override.get("minimum_words") or 0),
                    )
                except (TypeError, ValueError):
                    override_words = 0
                override_mode = str(
                    dialogue_density_override.get("mode") or "visual"
                ).strip().casefold()
                h3_dialogue_density = (
                    {
                        "minimum_turns": override_turns,
                        "minimum_words": override_words,
                    }
                    if override_turns or override_words else None
                )
                if override_mode == "silent":
                    h3_character_block += """

H3 SILENT-SEQUENCE CONTRACT — REQUIRED BY THE SEQUENCE ARCHITECT:
- Write no spoken dialogue in this bounded sequence. Use visible action,
  reactions, ambience, and synchronized nonverbal effects only.
"""
            else:
                h3_dialogue_density = _h3_dialogue_density_targets(
                    (
                        dialogue_intent_text
                        if dialogue_intent_text is not None else
                        story_description
                    ),
                    target_duration=target_duration,
                )
            if h3_dialogue_density:
                h3_character_block += f"""

H3 DIALOGUE-FORWARD PACING — REQUIRED BY THE USER'S CONCEPT:
- Write at least {h3_dialogue_density['minimum_turns']} responsive spoken turns and approximately {h3_dialogue_density['minimum_words']} or more spoken words across the complete {target_duration}-second screenplay, while remaining below the hard maximum above.
- These are creative pacing floors, not permission to pad. Build natural back-and-forth exchanges with setup, response, escalation, and payoff; distribute them across the story instead of isolating one short line inside a long silent scene.
- Preserve action-only beats where action is dramatically stronger, but do not turn a request for dialogue or banter into mostly silent spectacle.
- Give each exchange enough specific character intent that the lines could not be reassigned to another cast member.
"""

        long_form_dialogue_block = ""
        if (
            is_long_form_sequence
            and not is_h3_native
            and dialogue_density_override is not None
        ):
            try:
                sequence_turns = max(
                    0,
                    int(dialogue_density_override.get("minimum_turns") or 0),
                )
            except (TypeError, ValueError):
                sequence_turns = 0
            try:
                sequence_words = max(
                    0,
                    int(dialogue_density_override.get("minimum_words") or 0),
                )
            except (TypeError, ValueError):
                sequence_words = 0
            sequence_dialogue_mode = str(
                dialogue_density_override.get("mode") or "visual"
            ).strip().casefold()
            long_form_dialogue_mode = sequence_dialogue_mode
            if sequence_turns or sequence_words:
                long_form_dialogue_density = {
                    "minimum_turns": sequence_turns,
                    "minimum_words": sequence_words,
                }
                long_form_dialogue_block = f"""

LONG-FORM SEQUENCE DIALOGUE CONTRACT:
- This bounded sequence calls for at least {sequence_turns} responsive spoken turns and approximately {sequence_words} or more spoken words, while remaining below the hard screenplay budget.
- Keep each complete line with its speaker and preserve every user-authored line exactly. Dialogue must advance this sequence's local objective; do not borrow lines or events owned by another sequence.
"""
            elif sequence_dialogue_mode == "silent":
                long_form_dialogue_block = """

LONG-FORM SILENT-SEQUENCE CONTRACT:
- Write no spoken dialogue in this bounded sequence. Use visible action, reaction, ambience, and synchronized nonverbal effects only.
"""

        print("[ShortFilmPlanner] Pass 1: Writing screenplay...")
        pass1_system = f"""You are an acclaimed screenwriter celebrated for dialogue that sounds like real people actually talking — never stiff, formal, stagey, or "AI-like." You give every character a distinct, believable voice, and you fully commit to whatever tone, era, or style the concept calls for. Write a complete short film screenplay.

{f"You are given a REFERENCE PHOTO of the characters. Use their visible appearance in the script." if has_reference else ""}
{char_block}
{narrative_block}
{story_continuity_block}

{screenplay_rules}
{h3_character_block}
{long_form_dialogue_block}
{length_budget_block}"""

        if polish_block:
            pass1_system = f"{pass1_system}\n\n{polish_block}"
        pass1_system = inject_nsfw_if_enabled(pass1_system, nsfw, "screenplay")

        pass1_user = repair_text(
            f"Write a short film screenplay based on this concept:\n\n{story_description}"
        )
        pass1_system = repair_text(pass1_system)

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
        # thinking_budget enlarges the combined response allowance and, on
        # current local Qwen llama.cpp builds, also sets the per-request
        # reasoning ceiling. H3 still validates the returned screenplay and
        # retries once without thinking rather than trusting an empty or
        # prematurely stopped answer.
        _output_token_cap = max(2000, max_total_words * 3)
        screenplay = repair_text(
            self._generate_streaming(
                prompt=pass1_user,
                system_prompt=pass1_system,
                max_new_tokens=_output_token_cap,
                temperature=0.8,
                # Creative reasoning remains enabled for Qwen3.8. Its observed
                # screenplay planning routinely needs more than 8K reasoning
                # tokens before it begins the answer. Preserve the proven 16K
                # short-film path and scale to 24K/32K for longer films, with
                # the validated non-thinking recovery below as a final guard.
                thinking_budget=int(getattr(
                    self,
                    "_long_form_screenplay_thinking_budget_override",
                    _h3_screenplay_thinking_budget(target_duration),
                )),
                image_paths=image_paths or [],
                frequency_penalty=0.15,
                presence_penalty=0.05,
            )
        )

        if is_h3_native:
            screenplay_recovery_reasons = _h3_screenplay_recovery_reasons(
                screenplay,
                story_description=story_description,
            )
            dialogue_density_issue = _h3_dialogue_density_issue_for_targets(
                screenplay,
                h3_dialogue_density,
            )
            if dialogue_density_issue:
                screenplay_recovery_reasons.append(dialogue_density_issue)
            if screenplay_recovery_reasons:
                print(
                    "[ShortFilmPlanner] Pass 1 returned an unusable H3 "
                    "screenplay ("
                    + "; ".join(screenplay_recovery_reasons)
                    + "); retrying once without thinking..."
                )
                recovery_system = pass1_system + """

H3 SCREENPLAY RECOVERY — FINAL ANSWER REQUIRED:
- Return the complete finished screenplay now. Do not output analysis, planning notes, a synopsis, or a thinking block.
- Use canonical screenplay scene headings, action, uppercase speaker headings, and spoken dialogue beneath each speaker heading.
- Preserve every explicit user-authored spoken line exactly and place it in a canonical dialogue block.
- Write the complete character-authentic exchanges and story beats; do not reduce the result to a visual outline or silent shot list.
"""
                if h3_dialogue_density:
                    recovery_system += f"""
- This is explicitly dialogue-forward. Return at least {h3_dialogue_density['minimum_turns']} responsive spoken turns and approximately {h3_dialogue_density['minimum_words']} or more spoken words across the screenplay. Use natural multi-turn exchanges rather than isolated micro-lines or filler.
"""
                recovery_user = (
                    pass1_user
                    + "\n\nThe prior attempt did not produce a usable final "
                    "screenplay. Return only the finished screenplay."
                )
                screenplay = repair_text(
                    self._generate_streaming(
                        prompt=repair_text(recovery_user),
                        system_prompt=repair_text(recovery_system),
                        max_new_tokens=_output_token_cap,
                        temperature=0.8,
                        thinking_budget=0,
                        enable_thinking=False,
                        image_paths=image_paths or [],
                        frequency_penalty=0.15,
                        presence_penalty=0.05,
                    )
                )
                remaining_reasons = _h3_screenplay_recovery_reasons(
                    screenplay,
                    story_description=story_description,
                )
                if (
                    remaining_reasons
                    and all(
                        reason.startswith(
                            "explicit user dialogue is missing"
                        )
                        for reason in remaining_reasons
                    )
                ):
                    screenplay = _restore_missing_h3_screenplay_dialogue(
                        screenplay,
                        story_description=story_description,
                    )
                    remaining_reasons = _h3_screenplay_recovery_reasons(
                        screenplay,
                        story_description=story_description,
                    )
                    if not remaining_reasons:
                        print(
                            "[ShortFilmPlanner] Restored omitted immutable "
                            "dialogue in its assigned bounded sequence."
                        )
                if remaining_reasons:
                    raise RuntimeError(
                        "MiniMax H3 screenplay generation failed its automatic "
                        "recovery: "
                        + "; ".join(remaining_reasons)
                        + ". No video jobs were queued."
                    )
                remaining_density_issue = _h3_dialogue_density_issue_for_targets(
                    screenplay,
                    h3_dialogue_density,
                )
                if remaining_density_issue:
                    print(
                        "[ShortFilmPlanner] H3 screenplay recovery remains "
                        f"dialogue-light ({remaining_density_issue}); continuing "
                        "without blocking the Director run."
                    )
                print(
                    "[ShortFilmPlanner] H3 screenplay recovery succeeded; "
                    "continuing with canonical dialogue lock."
                )

            # Pass 2 cannot safely "compress" a screenplay after its words
            # become the immutable dialogue manifest. Settle every timing and
            # length violation here, while generated prose and dialogue may
            # still be edited. This is especially important for long-form
            # planning, where a 75-second sequence previously reached Pass 2
            # with more than twice its complete screenplay budget.
            budget_issues = _h3_screenplay_budget_issues(
                screenplay,
                story_description=story_description,
                max_total_words=max_total_words,
                max_spoken_words=max_spoken_words,
                maximum_line_words=h3_maximum_line_words,
            )
            if budget_issues:
                source_screenplay = screenplay
                print(
                    "[ShortFilmPlanner] Pass 1 exceeded the hard H3 pre-lock "
                    "budget (" + "; ".join(budget_issues) + "); running one "
                    "focused screenplay compression pass."
                )
                compression_system = f"""You are Maestro's H3 screenplay timing editor. Return only a complete canonical screenplay, never analysis, JSON, or a synopsis.

Compress the supplied screenplay so it can be performed in exactly {target_duration} seconds:
- At most {max_total_words} total words, including action and headings.
- At most {max_spoken_words} spoken words across all dialogue.
- Every generated spoken turn is at most {h3_maximum_line_words} words and remains a complete thought.
- Preserve every literal user-authored dialogue line exactly, including speaker and order.
- Preserve the binding story events, causal handoff, ending, cast identities, and existing order.
- Do not add dialogue, events, characters, locations, or planning commentary.
- Cut redundant action prose and generated conversational filler first. Combine action descriptions and shorten generated dialogue without converting the screenplay to an outline.
- Use canonical scene headings, action paragraphs, uppercase speaker headings, and dialogue beneath its speaker heading."""
                compression_user = f"""Compress this screenplay to the stated hard budget.

BINDING STORY BLUEPRINT:
{story_blueprint_text}

SCREENPLAY TO COMPRESS:
{source_screenplay}"""
                compressed = ""
                try:
                    compressed = repair_text(
                        self._generate_streaming(
                            prompt=repair_text(compression_user),
                            system_prompt=repair_text(compression_system),
                            max_new_tokens=max(1200, max_total_words * 3),
                            temperature=0.35,
                            thinking_budget=0,
                            enable_thinking=False,
                            image_paths=image_paths or [],
                            frequency_penalty=0.1,
                            presence_penalty=0.02,
                        )
                    )
                except InterruptedError:
                    raise
                except Exception as exc:
                    print(
                        "[ShortFilmPlanner] Focused H3 screenplay compression "
                        f"was unavailable ({exc}); using the deterministic "
                        "bounded screenplay compiler."
                    )

                compression_reasons = _h3_screenplay_recovery_reasons(
                    compressed,
                    story_description=story_description,
                ) if compressed else ["the compression answer is empty"]
                compression_budget_issues = _h3_screenplay_budget_issues(
                    compressed,
                    story_description=story_description,
                    max_total_words=max_total_words,
                    max_spoken_words=max_spoken_words,
                    maximum_line_words=h3_maximum_line_words,
                ) if compressed else ["the compression answer is empty"]
                used_deterministic_budget_fallback = False
                if not compression_reasons and not compression_budget_issues:
                    screenplay = compressed
                    print(
                        "[ShortFilmPlanner] Focused H3 screenplay compression "
                        "satisfied the hard pre-lock budget."
                    )
                else:
                    used_deterministic_budget_fallback = True
                    if compressed:
                        print(
                            "[ShortFilmPlanner] Focused H3 screenplay compression "
                            "did not validate ("
                            + "; ".join([
                                *compression_reasons,
                                *compression_budget_issues,
                            ])
                            + "); using the deterministic bounded screenplay "
                            "compiler."
                        )
                    screenplay = _build_h3_budgeted_screenplay_fallback(
                        story_description=story_description,
                        story_blueprint=story_continuity_blueprint,
                        screenplay=source_screenplay,
                        max_total_words=max_total_words,
                        max_spoken_words=max_spoken_words,
                        maximum_line_words=h3_maximum_line_words,
                    )

                final_reasons = _h3_screenplay_recovery_reasons(
                    screenplay,
                    story_description=story_description,
                )
                final_budget_issues = _h3_screenplay_budget_issues(
                    screenplay,
                    story_description=story_description,
                    max_total_words=max_total_words,
                    max_spoken_words=max_spoken_words,
                    maximum_line_words=h3_maximum_line_words,
                )
                if final_reasons or final_budget_issues:
                    raise RuntimeError(
                        "MiniMax H3 screenplay cannot fit this bounded "
                        "sequence before dialogue lock: "
                        + "; ".join([*final_reasons, *final_budget_issues])
                        + ". Shorten literal user dialogue assigned to this "
                        "sequence or use a longer duration. No video jobs "
                        "were queued."
                    )
                if used_deterministic_budget_fallback:
                    print(
                        "[ShortFilmPlanner] Deterministic H3 screenplay compiler "
                        "settled the hard pre-lock budget."
                    )

        elif is_long_form_sequence:
            # Long-form LTX uses the same bounded chapter/sequence architect
            # as H3. Historically, however, an oversized LTX screenplay only
            # emitted a warning and was handed to Pass 2, where the model had
            # to rush it, stretch runtime, or omit events. Settle the sequence
            # before image/video prompts are authored, while generated prose
            # and dialogue are still editable.
            ltx_max_spoken_words = (
                0
                if long_form_dialogue_mode == "silent"
                else max_spoken_words
            )
            screenplay_recovery_reasons = _h3_screenplay_recovery_reasons(
                screenplay,
                story_description=story_description,
            )
            screenplay_budget_issues = _long_form_screenplay_budget_issues(
                screenplay,
                story_description=story_description,
                max_total_words=max_total_words,
                max_spoken_words=ltx_max_spoken_words,
            )
            dialogue_density_issue = _h3_dialogue_density_issue_for_targets(
                screenplay,
                long_form_dialogue_density,
            )
            focused_reasons = [
                *screenplay_recovery_reasons,
                *screenplay_budget_issues,
            ]
            if dialogue_density_issue:
                focused_reasons.append(dialogue_density_issue)

            if focused_reasons:
                source_screenplay = screenplay
                print(
                    "[ShortFilmPlanner] Long-form LTX screenplay needs one "
                    "focused timing repair ("
                    + "; ".join(focused_reasons)
                    + ")."
                )
                dialogue_rule = (
                    "- This sequence is silent. Remove every spoken line and "
                    "perform the story through visible action and sound."
                    if ltx_max_spoken_words <= 0 else
                    f"- At most {ltx_max_spoken_words} spoken words across "
                    "all dialogue."
                )
                density_rule = ""
                if long_form_dialogue_density:
                    density_rule = (
                        "\n- Preserve the requested conversational pacing: "
                        f"at least {long_form_dialogue_density['minimum_turns']} "
                        "responsive turns and approximately "
                        f"{long_form_dialogue_density['minimum_words']} or more "
                        "spoken words, without exceeding the hard maximum."
                    )
                compression_system = f"""You are Maestro's long-form LTX screenplay timing editor. Return only a complete canonical screenplay, never analysis, JSON, a synopsis, or planning commentary.

Repair and compress the supplied bounded sequence so it can be performed in exactly {target_duration} seconds:
- At most {max_total_words} total words, including action and headings.
{dialogue_rule}{density_rule}
- Preserve every literal user-authored dialogue line exactly, including its speaker and order.
- Preserve the binding source events, opening cause, causal handoff, ending, cast identities, and existing order.
- Do not add dialogue, events, characters, locations, or subplots from another sequence.
- Cut redundant action prose and generated conversational filler first. Keep complete speakable sentences; never split one sentence merely to satisfy length.
- Write concrete camera-observable opening state, action, and ending state. Pass 2 must be able to derive a static first-frame image prompt and chronological LTX video/window prompts from this screenplay.
- Use canonical scene headings, action paragraphs, uppercase speaker headings, and dialogue beneath its speaker heading."""
                compression_user = f"""Repair this bounded sequence to the stated hard budget.

BINDING STORY BLUEPRINT:
{story_blueprint_text}

SCREENPLAY TO REPAIR:
{source_screenplay}"""
                repaired_screenplay = ""
                try:
                    repaired_screenplay = repair_text(
                        self._generate_streaming(
                            prompt=repair_text(compression_user),
                            system_prompt=repair_text(compression_system),
                            max_new_tokens=max(1200, max_total_words * 3),
                            temperature=0.35,
                            thinking_budget=0,
                            enable_thinking=False,
                            image_paths=image_paths or [],
                            frequency_penalty=0.1,
                            presence_penalty=0.02,
                        )
                    )
                except InterruptedError:
                    raise
                except Exception as exc:
                    print(
                        "[ShortFilmPlanner] Focused long-form LTX screenplay "
                        f"repair was unavailable ({exc}); using the "
                        "deterministic bounded screenplay compiler."
                    )

                if repaired_screenplay:
                    repaired_screenplay = _restore_missing_h3_screenplay_dialogue(
                        repaired_screenplay,
                        story_description=story_description,
                    )
                repaired_reasons = (
                    _h3_screenplay_recovery_reasons(
                        repaired_screenplay,
                        story_description=story_description,
                    )
                    if repaired_screenplay else
                    ["the repair answer is empty"]
                )
                repaired_budget_issues = (
                    _long_form_screenplay_budget_issues(
                        repaired_screenplay,
                        story_description=story_description,
                        max_total_words=max_total_words,
                        max_spoken_words=ltx_max_spoken_words,
                    )
                    if repaired_screenplay else
                    ["the repair answer is empty"]
                )
                if not repaired_reasons and not repaired_budget_issues:
                    screenplay = repaired_screenplay
                    print(
                        "[ShortFilmPlanner] Focused long-form LTX screenplay "
                        "repair satisfied the hard sequence budget."
                    )
                else:
                    if repaired_screenplay:
                        print(
                            "[ShortFilmPlanner] Focused long-form LTX "
                            "screenplay repair did not validate ("
                            + "; ".join([
                                *repaired_reasons,
                                *repaired_budget_issues,
                            ])
                            + "); using the deterministic bounded screenplay "
                            "compiler."
                        )
                    screenplay = _build_long_form_budgeted_screenplay_fallback(
                        story_description=story_description,
                        story_blueprint=story_continuity_blueprint,
                        screenplay=source_screenplay,
                        max_total_words=max_total_words,
                        max_spoken_words=ltx_max_spoken_words,
                    )

                final_reasons = _h3_screenplay_recovery_reasons(
                    screenplay,
                    story_description=story_description,
                )
                final_budget_issues = _long_form_screenplay_budget_issues(
                    screenplay,
                    story_description=story_description,
                    max_total_words=max_total_words,
                    max_spoken_words=ltx_max_spoken_words,
                )
                if final_reasons or final_budget_issues:
                    raise RuntimeError(
                        "Long-form LTX screenplay cannot fit this bounded "
                        "sequence before image/video prompt planning: "
                        + "; ".join([*final_reasons, *final_budget_issues])
                        + ". Shorten literal user dialogue assigned to this "
                        "sequence or use a longer duration. No video jobs "
                        "were queued."
                    )
                remaining_density_issue = (
                    _h3_dialogue_density_issue_for_targets(
                        screenplay,
                        long_form_dialogue_density,
                    )
                )
                if remaining_density_issue:
                    print(
                        "[ShortFilmPlanner] Long-form LTX screenplay remains "
                        f"dialogue-light ({remaining_density_issue}); "
                        "continuing without inventing filler."
                    )

        if is_h3_native:
            long_form_bible = getattr(
                self,
                "_long_form_story_bible_override",
                None,
            )
            canonical_speakers = [
                str(row.get("character_name") or "").strip()
                for row in h3_voice_bible
                if isinstance(row, dict) and row.get("character_name")
            ]
            if isinstance(long_form_bible, dict):
                canonical_speakers.extend(
                    str(row.get("name") or "").strip()
                    for row in long_form_bible.get("canonical_characters") or []
                    if isinstance(row, dict) and row.get("name")
                )
            canonical_speakers.extend(
                str(
                    getattr(profile, "display_name", "")
                    or getattr(profile, "id", "")
                    or ""
                ).strip()
                for profile in (char_profiles or [])
            )
            screenplay, heading_repairs = _repair_h3_screenplay_speaker_headings(
                screenplay,
                canonical_speakers,
            )
            if heading_repairs:
                print(
                    "[ShortFilmPlanner] Corrected screenplay speaker heading "
                    "drift before dialogue lock: "
                    + ", ".join(
                        f"{source} -> {target}"
                        for source, target in heading_repairs
                    )
                )

        print(f"[ShortFilmPlanner] Screenplay: {len(screenplay)} chars")

        # ── Post-Pass-1 length warning ───────────────────────────────
        # Short non-H3 projects retain the historical advisory. H3 and every
        # bounded long-form sequence have already satisfied the hard budget
        # above, before immutable dialogue or image/video prompt planning.
        _word_count = len(screenplay.split())
        if _word_count > max_total_words * 1.15:
            print(
                f"[ShortFilmPlanner] ⚠ Pass 1 over budget: {_word_count} words "
                f"(budget was {max_total_words}, +{_word_count - max_total_words}). "
                "The selected short-form non-H3 shot planner will pace the available "
                "material; expect possible runtime overshoot."
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
                        maximum_line_words=h3_maximum_line_words,
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
                story_continuity_blueprint=story_continuity_blueprint,
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
        is_ltx_family = str(video_model).strip().casefold().startswith("ltx")

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
        direct_video_family = (
            "MiniMax H3"
            if is_h3_native else
            "LTX" if is_ltx_family else "DIRECT VIDEO"
        )
        visual_strategy_rules = (
            f"{direct_video_family} DIRECT VIDEO GUIDANCE:\n"
            "- No generated start frame will be supplied. video_prompt or "
            "window_prompts must be fully self-contained with setting, "
            "composition, identities and "
            "visible traits, wardrobe, action, camera, dialogue, ambience, and "
            "synchronized sound.\n"
            "- Any supplied character, location, start-image, audio, or control "
            "references remain conditioning inputs; do not mistake them for a "
            "Director-generated shot image. Describe the complete target shot.\n"
            "- Do not create image_prompt, image_source, visual_changes, or "
            "keyframe_prompts. Those fields are intentionally absent from the "
            "video-only output schema."
            if not uses_generated_images else ""
        )
        speaker_name_note = (
            "- subjects_on_screen[i].speaker_name: when the screenplay uses a "
            "proper name, record that exact name. Preserve it in video_prompt "
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
            """- subjects_on_screen[i].visual_description: describe how the character looks IN THIS SHOT, including current wardrobe and story state. Keep each mapped identity name/label and useful visible traits consistent across shots."""
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

CAUSAL STORY CONTINUITY — DIRECTOR RESPONSIBILITY:
- The screenplay and binding blueprint below describe one film. Preserve their central objective, scene order, consequences, relationship changes, physical damage, and prop state.
- A location/time cut must be motivated by the outgoing scene and must open on the resulting arrival or consequence. Never create an unrelated scene merely for visual variety.
- Every shot advances the same causal chain. Do not invent a new antagonist, emergency, crime, objective, location, or subplot that is absent from both authorities.
- When a scene spans multiple shots/windows, preserve its geography and state. When a new scene begins, carry forward everything the screenplay has not visibly changed.

BINDING STORY-ARCHITECT BLUEPRINT:
{story_blueprint_text}


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
            else "a self-contained prompt plus any supplied conditioning references"
        )
        keyframe_user_rule = (
            "- Use keyframes ONLY when the video model needs visual info it "
            "cannot generate from the start image (new character entry, "
            "clothing reveal, dramatic state change). Do NOT use keyframes as "
            "a substitute for animating dialogue — the video model handles all "
            "talking, gestures, and expressions on its own."
            if uses_generated_images
            else "- Do not output image_prompt, image_source, visual_changes, "
            "or keyframe_prompts; this workflow renders directly from video "
            "prompts and supplied conditioning references."
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

BINDING STORY-ARCHITECT BLUEPRINT:
{story_blueprint_text}

Treat the screenplay as the final performed story and the blueprint as its
causal continuity ledger. At every location/time change, include the written
handoff and resulting arrival/consequence in the appropriate shot prompts.

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
        if (
            is_long_form_sequence
            and str(video_model or "").strip().casefold().startswith("ltx")
        ):
            shot_dicts = _prepare_long_form_ltx_prompt_contract(
                shot_dicts,
                story_continuity_blueprint,
                uses_generated_images=uses_generated_images,
                screenplay_dialogue_manifest=(
                    _extract_h3_screenplay_dialogue(screenplay)
                ),
            )

        # ── POST-PASS-2 SAFETY SCAN ─────────────────────────────────────
        # Defense in depth — Pass 2's structured output (image/video
        # prompts, action beats, dialogue, subjects) gets concatenated
        # and scanned the same way the screenplay was. Catches the case
        # where Pass 1 produced clean text but Pass 2's expansion
        # introduced minor + sexual co-occurrence.
        assert_no_minor_content_in_pass2(
            shot_dicts, source="shot list (Pass 2)"
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

        # Duration/window normalization above may merge prompts, convert a
        # window back to video_prompt, or create a newly required tail window.
        # Re-apply the idempotent long-form LTX contract before image cleanup
        # so the final cached sequence—not merely the raw LLM answer—has every
        # executable image/video field and causal boundary.
        if is_long_form_sequence and is_ltx_family:
            shot_dicts = _prepare_long_form_ltx_prompt_contract(
                shot_dicts,
                story_continuity_blueprint,
                uses_generated_images=uses_generated_images,
                screenplay_dialogue_manifest=(
                    _extract_h3_screenplay_dialogue(screenplay)
                ),
            )

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

        long_form_bible = getattr(
            self,
            "_long_form_story_bible_override",
            None,
        )
        if isinstance(long_form_bible, dict):
            shot_dicts, quality_warnings = sanitize_long_form_shot_dicts(
                shot_dicts,
                story_bible=long_form_bible,
            )
            if quality_warnings:
                print(
                    "[ShortFilmPlanner] Long-form dialogue/cast repair: "
                    + "; ".join(quality_warnings)
                )
                for shot_dict in shot_dicts:
                    shot_dict["long_form_quality_warnings"] = list(
                        quality_warnings
                    )

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
        story_continuity_blueprint: Optional[list[dict[str, Any]]] = None,
        nsfw: bool = False,
        polish_block: str = "",
    ) -> tuple[list[ShotPlan], Optional[str]]:
        """Break a screenplay directly into self-contained native H3 shots."""

        from ..nsfw_guidance import inject_nsfw_if_enabled
        from ..safety_scan import (
            assert_no_minor_content_in_pass2,
        )

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
        story_continuity_blueprint = copy.deepcopy(
            story_continuity_blueprint or []
        )
        story_blueprint_text = _format_story_continuity_blueprint(
            story_continuity_blueprint
        )
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
- Every video_prompt must stand alone, but it must also be economical: normally 180-450 words and never more than 650 words. State the exact physical setting, visible people, essential appearance/wardrobe, chronological action, camera, lighting, dialogue, ambience, effects, and music once each. Do not paste the screenplay, story bible, production contract, or continuity metadata into the prompt; translate them into concrete visible/audible instructions. Never restate the same opening, action, speaker, or ending in multiple differently worded paragraphs.
- WARDROBE IS STATE: on every appearance, subjects_on_screen must give each person a complete head-to-toe wardrobe (colors, materials, layers, accessories, and visible footwear). Repeat the same wardrobe wording in every shot within a continuity_group unless the screenplay explicitly changes it, and show that change visibly before using the new wardrobe.
- BLOCKING IS STATE: spatial_setup and each subject's position_or_relation describe the FIRST FRAME precisely: screen-left/center/right, foreground/midground/background, standing/seated/leaning, facing direction, and nearby furniture or props. Repeat that opening blocking in video_prompt.
- closing_blocking describes the final positions at the end of the shot. When the next shot in the same continuity_group opens with different blocking, the current shot must visibly show the person walking, sitting, standing, turning, or otherwise moving into that next arrangement before the cut.
- continuity_strategy is "independent" for a new place/time, "continuous" for a normal editorial cut within the same scene, or "extend_previous" ONLY when the next generation should start from the literal final frame with the same camera axis/composition and no intended cut. Use extend_previous sparingly.
- continuity_group is a short stable ID such as kitchen_morning_1. Reuse it only while place and story time remain uninterrupted; change it for any location or time jump.
- "independent" means the video generation does not inherit a literal prior frame; it NEVER means narratively unrelated. Every shot remains part of one causal film.
- CAUSAL SCENE HANDOFFS ARE REQUIRED: when continuity_group changes, the final shot in the outgoing group must contain the screenplay's visible decision, discovery, departure, pursuit, dispatch, or consequence that motivates the change, and the first shot in the new group must open on its concrete result or arrival. Never teleport characters to a visually interesting location without a story reason.
- STORY STATE IS CONTINUOUS ACROSS GROUPS: carry the active objective, information learned, relationship changes, injuries, dirt or wardrobe damage, and important props even when the physical location changes.
- story_scene_number maps every shot to the corresponding ordered scene in the binding blueprint. Multiple coverage shots may share one story_scene_number, but scene numbers never move backward or skip the screenplay's causal events.
- causal_handoff is camera-observable story context, not an editing instruction. On the first shot of a scene it states the exact prior cause and visible arrival/result; on later coverage in the same scene it states the immediately preceding action or consequence that this shot advances.
- persistent_story_state states the current objective, knowledge, relationship, physical damage, wardrobe/prop state, and unresolved danger that remain true at this point. Repeat the relevant state instead of silently resetting it at a cut.
- Do not invent a new villain, emergency, crime, objective, or location that is absent from both the screenplay and the binding story blueprint. Shot planning directs and photographs the written film; it does not replace it with disconnected coverage.
- WORLD CONTINUITY IS REQUIRED: preserve any supplied TV show, film, performer, franchise, historical era, city, named venue, room, or recognizable set. Repeat the relevant world/franchise and full location in EACH video_prompt; never collapse a named series and its recognizable apartment set into a generic kitchen.
- Each screenplay event and each spoken line appears in exactly one shot. Do not duplicate dialogue across adjacent shots. Preserve scripted dialogue verbatim.
- CONVERSATION PACKING IS REQUIRED: a change of speaker is not by itself a reason to start another array item. Within the same uninterrupted location and story beat, prefer one native clip ({preferred_duration_text}) containing 2-4 alternating dialogue turns when their combined total is no more than {maximum_dialogue_words} words. Keep a brief reaction such as "What?", a gasp, or a one-line reply in the surrounding exchange instead of wasting a separate minimum-length clip.
- INTERNAL CAMERA EDITING IS SUPPORTED: inside one bounded H3 clip, the camera may begin on an ensemble frame, cut or reframe to each current speaker before their tagged line, hold their unobstructed face and mouth through the complete line, capture reactions, and finish on a new composition. Describe that chronological coverage in camera_plan and action_beats. Prefer the lower end of the requested shot-count range for a continuous dialogue scene.
- DIALOGUE CAMERA COVERAGE IS REQUIRED: do not leave a dialogue scene entirely in a wide master. Use a wide or two-shot only to establish geography or cover large action, then cut or reframe to motivated medium close-ups or over-the-shoulder angles for each line and listener reactions. Vary shot size by dramatic purpose across adjacent clips; do not default every clip to the same wide framing.
- DIALOGUE MUST NOT LIVE ONLY IN dialogue_beats. Every dialogue_beats[].spoken_text must also appear exactly once in the same shot's video_prompt as <d>[English] Exact words</d>, with the speaker ID/name, delivery, and physical cue outside the tag. If dialogue_beats is empty, explicitly state that no one speaks, mouths remain closed, and no muttering, gibberish, or speech-like vocalization occurs.
- SPEAKER VISIBILITY IS REQUIRED: every person who delivers a line must have a complete subjects_on_screen entry and remain visibly framed with an unobstructed face and mouth for the full line. Reframe to the current speaker before speech; reaction framing may follow only after the spoken line is complete.
- CAST LIST CONSISTENCY IS REQUIRED: every person mentioned in spatial_setup, action_beats, dialogue_beats, ending_beat, closing_blocking, or video_prompt must appear in subjects_on_screen. Do not mention a bystander in blocking while omitting that person from the visible cast.
- NAMED CAST IS CLOSED: subjects_on_screen may contain only named people present in the user concept, supplied character references, binding voice bible, or locked screenplay dialogue. Setting-appropriate silent extras may appear only as generic roles such as "barista" or "background patron". Never add a named cameo or another familiar character from the franchise.
- A shot may follow another in the finished edit, but its prompt must describe its own opening state instead of saying "continue", "as before", "the push-in continues", or similar.
- multishot is always false because this is not the LTX Multi-Shot LoRA format. That field does NOT prohibit H3 from making speaker-motivated internal cuts, reframes, and reaction coverage inside its one bounded generation.

{char_rules}

{video_name_rules}

{video_rules}

{voice_bible_block}

BINDING CAUSAL STORY BLUEPRINT:
{story_blueprint_text}

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
    "story_scene_number": 1,
    "causal_handoff": "Concrete prior cause and visible opening result for this shot",
    "persistent_story_state": "Current objective, knowledge, relationship, damage, wardrobe, props, and unresolved danger",
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

BINDING CAUSAL STORY BLUEPRINT:
{story_blueprint_text}

The screenplay is the final performed story, and the blueprint is its causal
continuity ledger. Preserve the screenplay's exact events and dialogue while
using the blueprint to keep objectives, locations, consequences, relationship
changes, physical damage, and props connected across scene boundaries. Do not
add a location or incident merely for shot variety.

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
6. At every continuity_group change, verify that the prior ending visibly motivates the new location/time and that the new opening shows the resulting arrival or consequence. "Independent" is a render boundary, not a story reset.
7. Compare persistent state across that boundary: objective, knowledge, relationship, injuries, wardrobe damage, and important props must carry forward unless the screenplay visibly changes them.
8. Remove any unrelated incident, antagonist, or location that is not supported by the screenplay and story blueprint.
9. Fill story_scene_number, causal_handoff, and persistent_story_state for every shot. Verify that scene numbers are monotonic and that adjacent handoffs describe one cause-and-effect chain rather than summaries of unrelated events.

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
        # New local generations are grammar-required to expose the causal
        # ledger, while the runtime completeness validator remains backward
        # compatible with saved plans and test fixtures from before the field
        # existed. _normalize_native_shots supplies safe fallbacks, then the
        # architect reconciler replaces them from the binding blueprint.
        schema_required = [
            *required[:7],
            "story_scene_number", "causal_handoff",
            "persistent_story_state",
            *required[7:],
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
            required=schema_required,
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
                try:
                    raw["story_scene_number"] = max(
                        1,
                        int(raw.get("story_scene_number") or 1),
                    )
                except (TypeError, ValueError):
                    raw["story_scene_number"] = 1
                raw["causal_handoff"] = re.sub(
                    r"\s+",
                    " ",
                    str(
                        raw.get("causal_handoff")
                        or raw.get("scene_goal")
                        or "Opening story beat"
                    ),
                ).strip()
                raw["persistent_story_state"] = re.sub(
                    r"\s+",
                    " ",
                    str(
                        raw.get("persistent_story_state")
                        or raw.get("ending_beat")
                        or raw.get("scene_goal")
                        or "Story state remains continuous"
                    ),
                ).strip()
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
                required=schema_required,
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
                if allocated is None and len(candidate) < maximum_by_runtime:
                    # The repair may have returned visually valid coverage but
                    # too few independent native clips for the immutable turn
                    # boundaries. Add the smallest number of neutral,
                    # same-scene coverage slots that can hold the conversation
                    # instead of asking another LLM to rewrite locked words.
                    for desired_count in range(
                        len(candidate) + 1,
                        maximum_by_runtime + 1,
                    ):
                        trial = _expand_h3_dialogue_coverage_slots(
                            candidate,
                            desired_count=desired_count,
                        )
                        try:
                            trial = _restore_h3_dialogue_after_pacing_repair(
                                source,
                                trial,
                                [frames_maximum / fps] * len(trial),
                            )
                            trial_schedule = _apply_native_schedule(
                                trial,
                                protect_dialogue=True,
                            )
                        except ValueError as error:
                            allocation_errors.append(str(error))
                            continue
                        violations = _h3_dialogue_budget_violations(
                            trial,
                            [frames / fps for frames in trial_schedule],
                            words_per_second=_H3_DIALOGUE_WORDS_PER_SECOND,
                        )
                        if violations:
                            allocation_errors.append(
                                "expanded dialogue coverage still exceeded "
                                "legal clip timing"
                            )
                            continue
                        allocated = trial
                        allocated_schedule = trial_schedule
                        print(
                            "[ShortFilmPlanner] Added "
                            f"{desired_count - len(candidate)} deterministic "
                            "same-scene H3 dialogue coverage slot(s)."
                        )
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
        long_form_bible = getattr(
            self,
            "_long_form_story_bible_override",
            None,
        )
        if isinstance(long_form_bible, dict):
            shot_dicts, quality_warnings = sanitize_long_form_shot_dicts(
                shot_dicts,
                story_bible=long_form_bible,
            )
            if quality_warnings:
                print(
                    "[ShortFilmPlanner] Long-form H3 dialogue/cast repair: "
                    + "; ".join(quality_warnings)
                )
                for shot_dict in shot_dicts:
                    shot_dict["long_form_quality_warnings"] = list(
                        quality_warnings
                    )
        shot_dicts = _prepare_h3_story_continuity(
            shot_dicts,
            story_continuity_blueprint,
        )
        shot_dicts = _enforce_h3_speaker_visual_contract(
            shot_dicts,
            character_voice_bible,
            project_context=story_description,
            allowed_character_names=[
                str(
                    getattr(profile, "display_name", "")
                    or getattr(profile, "id", "")
                    or ""
                ).strip()
                for profile in (char_profiles or [])
            ],
        )
        shot_dicts = _prepare_h3_prompt_only_continuity(shot_dicts)

        assert_no_minor_content_in_pass2(
            shot_dicts, source="shot list (H3 native Pass 2)"
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
                    "story_scene_number": raw.get("story_scene_number"),
                    "causal_handoff": raw.get("causal_handoff", ""),
                    "persistent_story_state": raw.get(
                        "persistent_story_state", ""
                    ),
                    "closing_blocking": raw.get(
                        "closing_blocking", raw.get("ending_beat", "")
                    ),
                    "vocal_contract": vocal_contract,
                    "long_form_quality_warnings": raw.get(
                        "long_form_quality_warnings", []
                    ),
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
            assert_no_minor_content_in_pass2,
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

        assert_no_minor_content_in_pass2(
            shot_dicts, source="shot list (single-pass fallback)"
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
