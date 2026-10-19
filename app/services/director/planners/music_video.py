"""
Music Video Planner — creates a ProductionPlan from song analysis data.

Inputs: song audio, transcript/lyrics, beat map, section labels,
performer map, optional reference image, user scene concept.

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
    SpeakerMapEntry,
)
from ..policies import build_character_rules_block, build_camera_style_block
from ..vocal_activity import classify_vocal_intervals
from ..music_performance import MUSIC_PERFORMANCE_RULES, PERFORMANCE_ROLES
from ..prompt_timing import (
    music_video_visual_timing_issue,
    normalize_music_video_visual_timecodes,
)
from ..physical_continuity import (
    GROUNDED_STATE_VALUES,
    MAX_PHYSICAL_FACTS,
    MAX_PHYSICAL_TEXT_CHARS,
    MAX_PHYSICAL_TRANSITIONS,
    MAX_PHYSICAL_VALUE_CHARS,
    PHYSICAL_STATE_PREDICATES,
    format_physical_state_for_prompt,
    music_video_progression_issue,
    normalize_physical_state,
    physical_transition_actions_missing_from_prompt,
    physical_state_json_schema,
    source_physical_state_issues,
    validate_physical_state_sequence,
)
from .base import BasePlanner
from ..music_narrative import (
    build_music_narrative_contract,
    format_music_combat_context,
    format_music_narrative_context,
    music_narrative_row_issue,
    validate_music_narrative_allocation,
)
from ..lyrics_context import lyrics_context_enabled, lyric_visual_policy


def _compact_lyrics_for_context(
    value: Any,
    *,
    max_words: int = 48,
    max_chars: int = 320,
) -> str:
    """Bound noisy transcription text before it reaches the planning LLM."""

    words = re.findall(r"\S+", str(value or ""))
    compact: list[str] = []
    index = 0
    truncated = False
    while index < len(words) and len(compact) < max_words:
        word = words[index]
        key = re.sub(r"(^\W+|\W+$)", "", word).casefold()
        run_end = index + 1
        while run_end < len(words):
            candidate = re.sub(
                r"(^\W+|\W+$)", "", words[run_end],
            ).casefold()
            if not key or candidate != key:
                break
            run_end += 1
        run_length = run_end - index
        if run_length >= 4:
            compact.extend(words[index:index + 3])
            compact.append("[repeated refrain]")
        else:
            compact.extend(words[index:run_end])
        index = run_end

    if index < len(words) or len(compact) > max_words:
        truncated = True
    text = " ".join(compact[:max_words])
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
        truncated = True
    if truncated:
        text = f"{text} [transcript excerpt truncated]".strip()
    return text


def _music_clip_duration_seconds(clip: dict) -> Optional[float]:
    """Resolve a clip's actual local duration without using its song offset."""

    def finite_number(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    start = finite_number(clip.get("start"))
    end = finite_number(clip.get("end"))
    if start is not None and end is not None and end > start:
        return end - start

    for key in ("duration_sec", "duration_seconds", "local_duration_sec"):
        duration = finite_number(clip.get(key))
        if duration is not None and duration > 0:
            return duration

    frames = finite_number(clip.get("duration_frames"))
    fps = finite_number(clip.get("fps") or clip.get("frame_rate"))
    if frames is not None and frames > 0 and fps is not None and fps > 0:
        return frames / fps
    return None


def _has_meaningful_cast_identity(subject: dict) -> bool:
    """Check that a visible cast member has a stable identity field."""

    generic_values = {
        "a character",
        "a person",
        "character",
        "none",
        "n a",
        "na",
        "nobody",
        "person",
        "performer",
        "speaker",
        "singer",
        "subject",
        "the character",
        "the person",
        "the performer",
        "the singer",
        "the speaker",
        "the subject",
        "unknown",
        "unknown person",
        "unknown speaker",
        "vocalist",
    }

    def usable(value: Any) -> bool:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
        normalized = re.sub(r"^(?:the|a|an)\s+", "", normalized)
        numeric_slot_id = re.fullmatch(
            r"(?:char|character|subject|speaker)\s*\d+",
            normalized,
            flags=re.IGNORECASE,
        )
        return bool(
            normalized
            and normalized not in generic_values
            and numeric_slot_id is None
        )

    return usable(subject.get("character_id")) or usable(subject.get("speaker_name"))


# ── Section-based visual strategy ────────────────────────────────────

_SECTION_VISUAL_STRATEGY = {
    "intro": {
        "camera_default": "wide establishing shot",
        "movement_intensity": "subtle",
        "energy": "building",
        "hints": "atmospheric, slow reveal, moody lighting, set the tone",
    },
    "verse": {
        "camera_default": "medium shot",
        "movement_intensity": "subtle",
        "energy": "steady",
        "hints": "storytelling, character focus, steady camera, intimate",
    },
    "chorus": {
        "camera_default": "dynamic angle",
        "movement_intensity": "dynamic",
        "energy": "peak",
        "hints": "bold energy, wide and close-up mix, confident movement",
    },
    "bridge": {
        "camera_default": "unique angle",
        "movement_intensity": "moderate",
        "energy": "contrasting",
        "hints": "change of scenery, dreamy or surreal, unexpected perspective",
    },
    "outro": {
        "camera_default": "wide shot",
        "movement_intensity": "subtle",
        "energy": "fading",
        "hints": "pulling back, reflective, fading light, resolution",
    },
    "instrumental": {
        "camera_default": "sweeping shot",
        "movement_intensity": "moderate",
        "energy": "atmospheric",
        "hints": "environment focus, dramatic sweep, textures, abstract visuals",
    },
}


_MUSIC_IMAGE_FIELDS = frozenset({
    "image_source",
    "image_prompt",
    "visual_changes",
    "keyframe_prompts",
})

_MUSIC_SHOT_PROPERTIES = {
    "scene_goal": {"type": "string"},
    "scene_type": {"type": "string"},
    "subjects_on_screen": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "character_id": {"type": "string"},
                "speaker_name": {"type": "string"},
                "visual_description": {"type": "string"},
                "position_or_relation": {"type": "string"},
                "performance_role": {"type": "string", "enum": list(PERFORMANCE_ROLES)},
            },
            "required": ["visual_description", "performance_role"],
            "additionalProperties": False,
        },
    },
    "spatial_setup": {"type": "string"},
    "continuity_strategy": {
        "type": "string",
        "enum": ["continuous", "independent", "extend_previous"],
    },
    "environment": {"type": "string"},
    "visual_style": {"type": "string"},
    "lighting": {"type": "string"},
    "mood": {"type": "string"},
    "action_beats": {"type": "array", "items": {"type": "string"}},
    "physical_state": physical_state_json_schema(),
    "source_event_ids": {
        "type": "array",
        "items": {"type": "string", "maxLength": 80},
        "maxItems": 16,
    },
    "camera_plan": {
        "type": "object",
        "properties": {
            "framing": {"type": "string"},
            "angle": {"type": "string"},
            "movement": {"type": "string"},
            "movement_intensity": {"type": "string"},
            "lens_feel": {"type": "string"},
        },
        "required": ["framing"],
        "additionalProperties": False,
    },
    "ending_beat": {"type": "string"},
    "image_source": {"type": "string"},
    "image_prompt": {"type": "string"},
    "visual_changes": {"type": "array", "items": {"type": "string"}},
    "video_prompt": {"type": "string"},
    "keyframe_prompts": {"type": "array", "items": {"type": "string"}},
    "window_prompts": {"type": "array", "items": {"type": "string"}},
}

_EXPLICIT_PHYSICAL_RELATION_RE = re.compile(
    r"\b(?:riders?|riding|rides|rode|aboard|astride|mounted|mounts?|mounting|"
    r"carried\s+by|carrying|carries|sits?\s+(?:on|atop|astride)|"
    r"sitting\s+(?:on|atop|astride)|seated\s+(?:on|atop|astride)|"
    r"stands?\s+(?:on|atop)|standing\s+(?:on|atop)|"
    r"gripping|grips|touching|touches|"
    r"leaning\s+against|leans\s+against)\b",
    re.IGNORECASE,
)
_EXPLICIT_HOLD_VERB_RE = re.compile(r"\b(?:hold|holds|holding|held)\b", re.IGNORECASE)
_NONCONTACT_HOLD_HEADS = frozenset(
    {
        "pose", "posture", "stance", "gaze", "note", "breath", "line",
        "melody", "tune", "chord", "rhythm", "beat", "frame", "framing", "shot",
        "focus", "attention", "position", "stillness",
    }
)
_HOLD_COMPLEMENT_MODIFIERS = frozenset(
    {
        "a", "an", "the", "its", "his", "her", "their", "our", "my",
        "your", "this", "that", "one", "both", "very", "still", "steady",
        "high", "low", "determined", "focused", "sustained", "long", "final",
        "last", "fixed", "upright", "composed", "wide", "in", "at", "with",
        "to", "on", "of", "by", "up", "down", "into",
    }
)
_HOLD_CLAUSE_BOUNDARY_RE = re.compile(r"\b(?:and|but|while|as|then)\b|[,;.!?]")
_NEGATED_PHYSICAL_RELATION_RE = re.compile(
    r"\b(?:unsupported|unmounted|not\s+(?:mounted|seated|sitting|"
    r"standing\s+on|touching|in\s+contact|holding|gripping)|"
    r"without\s+(?:support|contact|a\s+grip))\b",
    re.IGNORECASE,
)


def _source_establishes_physical_relation(
    scene_description: Any,
    row: dict[str, Any],
) -> bool:
    """Use a bounded phrase check to catch omitted declared support/contact."""

    if not _has_explicit_physical_relation(str(scene_description or "")):
        return False
    row_text = [
        str(row.get(field) or "")
        for field in ("scene_goal", "video_prompt", "ending_beat")
    ]
    beats = row.get("action_beats")
    if isinstance(beats, list):
        row_text.extend(str(value) for value in beats[:4] if value)
    return _has_explicit_physical_relation(" ".join(row_text))


def _camera_is_hold_subject(text: str, verb_start: int) -> bool:
    """Recognize a literal camera-as-subject framing direction, not an operator."""

    prefix = text[max(0, verb_start - 160) : verb_start]
    boundaries = list(_HOLD_CLAUSE_BOUNDARY_RE.finditer(prefix))
    if boundaries:
        prefix = prefix[boundaries[-1].end() :]
    return bool(
        re.fullmatch(
            r"\s*(?:the\s+)?camera(?:\s+(?:now|then|finally))?\s*",
            prefix,
            re.IGNORECASE,
        )
    )


def _has_explicit_physical_relation(text: str) -> bool:
    """Match contact/support language without treating artistic ``holds`` as contact.

    For hold/held, only the immediate complement is inspected. Abstract heads
    such as a pose, note, gaze, or camera frame are not physical relations;
    concrete objects and people remain eligible contact anchors.
    """

    if _EXPLICIT_PHYSICAL_RELATION_RE.search(text):
        return True
    for match in _EXPLICIT_HOLD_VERB_RE.finditer(text[:20000]):
        if _camera_is_hold_subject(text, match.start()):
            continue
        tail = text[match.end() :]
        boundary = _HOLD_CLAUSE_BOUNDARY_RE.search(tail)
        if boundary:
            tail = tail[: boundary.start()]
        complement = re.findall(r"[a-z]+(?:['-][a-z]+)?", tail.casefold())[:8]
        meaningful = [
            word for word in complement
            if word not in _HOLD_COMPLEMENT_MODIFIERS
        ]
        if not meaningful:
            continue
        if meaningful[0] in _NONCONTACT_HOLD_HEADS:
            continue
        return True
    return False


def _assigned_source_events_for_clip(clip: dict[str, Any]) -> Any:
    """Read explicit event text attached to a clip, when the caller supplies it."""

    for field in ("source_events", "source_event_texts", "assigned_source_events"):
        value = clip.get(field)
        if value:
            return value
    single_event = clip.get("source_event") or clip.get("source_event_text")
    return single_event if single_event else None


def _physical_prompt_conflict_issue(row: dict[str, Any]) -> Optional[str]:
    """Catch direct negation of a declared final support/contact relation."""

    state = normalize_physical_state(row.get("physical_state"))
    prompt = str(row.get("video_prompt") or "")
    if state is None or not _NEGATED_PHYSICAL_RELATION_RE.search(prompt):
        return None
    prompt_tokens = {
        token.casefold() for token in re.findall(r"[a-z0-9]+", prompt)
    }
    for fact in state["closing"]:
        predicate = fact["predicate"]
        if predicate == "support":
            relational = re.search(
                r"\b(?:on|onto|atop|astride|mounted|seated|sitting|"
                r"standing|supported|carried|riding|aboard)\b",
                fact["value"], re.IGNORECASE,
            )
        elif predicate == "contact":
            relational = re.search(
                r"\b(?:touch|touching|contact|holding|gripping|grip|hand|"
                r"hands|embracing|embrace)\b",
                fact["value"], re.IGNORECASE,
            )
        else:
            relational = None
        if not relational:
            continue
        anchors = {
            token.casefold()
            for token in re.findall(r"[a-z0-9]+", fact["value"])
            if len(token) > 2 and token.casefold() not in {
                "the", "and", "with", "onto", "atop", "astride", "mounted",
                "seated", "sitting", "standing", "supported", "carried",
                "riding", "aboard", "touching", "contact", "holding",
                "gripping", "grip", "hands",
            }
        }
        if anchors and not anchors.intersection(prompt_tokens):
            continue
        return (
            "video_prompt directly negates the declared closing "
            f"{predicate} relation for {fact['subject_id']!r}"
        )
    return None


def _music_shot_schema(
    count: int,
    *,
    include_image_fields: bool,
    require_cast_identity: bool = False,
    require_physical_state: bool = False,
    omit_physical_state: bool = False,
    require_continuity_strategy: bool = False,
    require_native_single_clip: bool = False,
    narrative_event_ids: Optional[list[str]] = None,
) -> dict:
    properties = {
        key: value
        for key, value in _MUSIC_SHOT_PROPERTIES.items()
        if include_image_fields or key not in _MUSIC_IMAGE_FIELDS
    }
    if require_native_single_clip:
        # H3's timeline is already divided into native clips before planning.
        # Rolling windows would replace the clip's authored prompt with a
        # concatenation of separate timelines, each starting at zero.
        properties["window_prompts"] = {
            **properties["window_prompts"], "maxItems": 0,
        }
    if require_cast_identity:
        # Grammar-constrain the structural identity fields up front rather
        # than spending a semantic repair solely to request missing keys.
        cast = properties["subjects_on_screen"]
        properties["subjects_on_screen"] = {
            **cast,
            "items": {
                **cast["items"],
                "required": [*cast["items"]["required"], "character_id", "speaker_name"],
            },
        }
    required = [
        "scene_goal",
        "scene_type",
        "subjects_on_screen",
        "environment",
        "visual_style",
        "lighting",
        "mood",
        "action_beats",
        "camera_plan",
        "ending_beat",
        "image_source",
        "image_prompt",
        "visual_changes",
        "video_prompt",
        "window_prompts",
    ]
    if omit_physical_state:
        # H3 music direction is authored as visible prose. Keep the auxiliary
        # ledger available to strict callers, but out of production H3 output.
        properties.pop("physical_state", None)
    elif require_physical_state:
        # New H3 plans must explicitly declare an empty ledger when no
        # physical relation is established. Code derives its continuity
        # marker from the single planned cut strategy.
        properties["physical_state"] = _h3_physical_state_proposal_schema()
        required.extend(("physical_state", "continuity_strategy"))
    if require_continuity_strategy and "continuity_strategy" in properties:
        required.append("continuity_strategy")
    if narrative_event_ids:
        properties["source_event_ids"] = {
            **properties["source_event_ids"],
            "items": {"type": "string", "enum": narrative_event_ids},
        }
        required.append("source_event_ids")
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": [field for field in required if field in properties],
            "additionalProperties": False,
        },
        "minItems": max(1, count),
        "maxItems": max(1, count),
    }


def _h3_physical_state_proposal_schema() -> dict[str, Any]:
    """Schema for compact H3 state proposals; code derives from/closing values."""

    fact_schema = {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string", "maxLength": 80},
            "predicate": {
                "type": "string",
                "enum": list(PHYSICAL_STATE_PREDICATES),
            },
            "value": {"type": "string", "maxLength": MAX_PHYSICAL_VALUE_CHARS},
        },
        "required": ["subject_id", "predicate", "value"],
        "additionalProperties": False,
    }
    transition_schema = {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string", "maxLength": 80},
            "predicate": {
                "type": "string",
                "enum": list(PHYSICAL_STATE_PREDICATES),
            },
            "to_value": {"type": "string", "maxLength": MAX_PHYSICAL_VALUE_CHARS},
            "action": {"type": "string", "maxLength": MAX_PHYSICAL_TEXT_CHARS},
        },
        "required": ["subject_id", "predicate", "to_value", "action"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "opening": {
                "type": "array",
                "items": fact_schema,
                "maxItems": MAX_PHYSICAL_FACTS,
            },
            "transitions": {
                "type": "array",
                "items": transition_schema,
                "maxItems": MAX_PHYSICAL_TRANSITIONS,
            },
        },
        "required": ["opening", "transitions"],
        "additionalProperties": False,
    }


def _compact_state_text(
    value: Any,
    *,
    field: str,
    max_length: int,
    location: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} needs a non-empty {field}")
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) > max_length:
        raise ValueError(f"{location}.{field} exceeds {max_length} characters")
    return cleaned


def _compact_fact_key(fact: dict[str, str]) -> tuple[str, str]:
    return (fact["subject_id"].casefold(), fact["predicate"])


def _is_prop_manipulation_posture_mislabel(
    row: dict[str, Any], subject_id: str, action: str, value: str,
) -> bool:
    match = re.search(
        r"\b(?P<verb>twirl(?:s|ed|ing)?|brandish(?:es|ed|ing)?|"
        r"wave(?:s|d|ing)?|swing(?:s|ing)?|swung)\s+"
        r"(?:(?:a|an|the|its|his|her|their)\s+)?(?P<object>[\w’'-]+)\b",
        action, re.IGNORECASE,
    )
    if match is None:
        return False
    if _h3_action_beat_for_state_fact(
        row, {"subject_id": subject_id, "value": action},
        additional_roots={"brandish", "wave"},
    ) is None:
        return False
    root = "swing" if match.group("verb").casefold() == "swung" else re.match(
        r"(?:twirl|brandish|wave|swing)", match.group("verb").casefold(),
    ).group(0)
    value_tokens = re.findall(r"[a-z0-9]+", value.casefold())
    value_roots = {
        stem for token in value_tokens for stem in _h3_action_match_roots(token)
    }
    object_token = re.findall(r"[a-z0-9]+", match.group("object").casefold())
    return root in value_roots and bool(object_token) and object_token[0] in value_tokens


_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD = "_h3_physical_state_proposal_error"


def _strip_h3_physical_state_fields(row: Any) -> None:
    """Discard the auxiliary ledger from H3 music rows and cached drafts."""

    if not isinstance(row, dict):
        return
    row.pop("physical_state", None)
    row.pop(_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD, None)


def _materialize_h3_physical_state_proposals_strict(
    shot_dicts: list[Any],
    *,
    previous_physical_state: Any = None,
) -> list[Any]:
    """Expand compact H3 proposals into the strict canonical physical ledger.

    The proposal owns only explicit opening facts, destinations, and actions.
    Current values and complete closing state are derived deterministically.
    Existing full ledgers pass through unchanged for strict legacy validation.
    """

    if not isinstance(shot_dicts, list):
        raise ValueError("H3 shot plans must be an array")
    rows = copy.deepcopy(shot_dicts)
    prior_state = normalize_physical_state(previous_physical_state)

    def fact_list(value: Any, *, location: str) -> list[dict[str, str]]:
        if not isinstance(value, list):
            raise ValueError(f"{location} must be an array")
        if len(value) > MAX_PHYSICAL_FACTS:
            raise ValueError(
                f"{location} has more than {MAX_PHYSICAL_FACTS} facts"
            )
        facts: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for index, raw in enumerate(value):
            item_location = f"{location}[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{item_location} must be an object")
            if set(raw) != {"subject_id", "predicate", "value"}:
                raise ValueError(
                    f"{item_location} must contain only subject_id, predicate, value"
                )
            subject_id = _compact_state_text(
                raw.get("subject_id"), field="subject_id", max_length=80,
                location=item_location,
            )
            predicate = raw.get("predicate")
            if predicate not in PHYSICAL_STATE_PREDICATES:
                raise ValueError(
                    f"{item_location}.predicate must be one of "
                    f"{', '.join(PHYSICAL_STATE_PREDICATES)}"
                )
            fact_value = _compact_state_text(
                raw.get("value"), field="value",
                max_length=MAX_PHYSICAL_VALUE_CHARS, location=item_location,
            )
            if predicate == "grounded" and fact_value not in GROUNDED_STATE_VALUES:
                raise ValueError(
                    f"{item_location}.value for predicate 'grounded' must be "
                    "exactly 'grounded' or 'airborne'"
                )
            fact = {
                "subject_id": subject_id,
                "predicate": predicate,
                "value": fact_value,
            }
            key = _compact_fact_key(fact)
            if key in seen:
                raise ValueError(f"{item_location} duplicates a physical fact key")
            seen.add(key)
            facts.append(fact)
        return facts

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"H3 shot {row_index + 1} must be an object")
        raw_state = row.get("physical_state")
        if not isinstance(raw_state, dict):
            raise ValueError(f"H3 shot {row_index + 1} needs physical_state")

        # Keep any legacy/full ledger exactly as proposed; existing row and
        # sequence validators remain responsible for accepting or rejecting it.
        is_legacy_full_state = "closing" in raw_state
        if is_legacy_full_state:
            try:
                prior_state = normalize_physical_state(raw_state)
            except (TypeError, ValueError):
                prior_state = None
            continue

        if set(raw_state) not in (
            {"opening", "transitions"},
            {"opening", "transitions", "continuity"},
        ):
            raise ValueError(
                f"H3 shot {row_index + 1} compact physical_state must contain "
                "only opening and transitions, with optional legacy continuity"
            )
        strategy = row.get("continuity_strategy")
        if not isinstance(strategy, str) or strategy not in {
            "continuous", "extend_previous", "independent",
        }:
            raise ValueError(
                f"H3 shot {row_index + 1} continuity_strategy is invalid"
            )
        continuity = raw_state.get(
            "continuity", "independent" if strategy == "independent" else "continuous",
        )
        if not isinstance(continuity, str) or continuity not in {
            "continuous", "independent", "time_jump",
        }:
            raise ValueError(
                f"H3 shot {row_index + 1} physical_state.continuity is invalid"
            )
        if strategy == "independent":
            if continuity not in {"independent", "time_jump"}:
                raise ValueError(
                    f"H3 shot {row_index + 1} independent continuity_strategy "
                    "requires independent or time_jump physical continuity"
                )
        elif continuity != "continuous":
            raise ValueError(
                f"H3 shot {row_index + 1} continuous continuity_strategy "
                "requires continuous physical continuity"
            )

        proposed_opening = fact_list(
            raw_state.get("opening"),
            location=f"H3 shot {row_index + 1} physical_state.opening",
        )
        raw_transitions = raw_state.get("transitions")
        if not isinstance(raw_transitions, list):
            raise ValueError(
                f"H3 shot {row_index + 1} physical_state.transitions must be an array"
            )
        if len(raw_transitions) > MAX_PHYSICAL_TRANSITIONS:
            raise ValueError(
                f"H3 shot {row_index + 1} has more than "
                f"{MAX_PHYSICAL_TRANSITIONS} transitions"
            )

        visible_ids: set[str] = set()
        subjects = row.get("subjects_on_screen")
        if isinstance(subjects, list):
            for subject in subjects:
                if not isinstance(subject, dict):
                    continue
                identifier = subject.get("character_id") or subject.get("speaker_name")
                if isinstance(identifier, str) and identifier.strip():
                    visible_ids.add(identifier.strip().casefold())

        def check_visible_owner(fact: dict[str, str], location: str) -> None:
            if visible_ids and fact["subject_id"].casefold() not in visible_ids:
                raise ValueError(
                    f"{location} subject_id {fact['subject_id']!r} is not in "
                    "subjects_on_screen"
                )

        is_discontinuity = continuity in {"independent", "time_jump"}
        prior_closing = (
            prior_state["closing"]
            if prior_state is not None and not is_discontinuity else []
        )
        if is_discontinuity and prior_state and prior_state["closing"] and not proposed_opening:
            raise ValueError(
                f"H3 shot {row_index + 1} must explicitly establish opening facts "
                "when resetting an active physical state"
            )

        current: dict[tuple[str, str], dict[str, str]] = {}
        opening: list[dict[str, str]] = []
        for fact in prior_closing:
            copied = dict(fact)
            key = _compact_fact_key(copied)
            current[key] = copied
            opening.append(copied)

        for fact in proposed_opening:
            check_visible_owner(fact, f"H3 shot {row_index + 1} opening")
            key = _compact_fact_key(fact)
            existing = current.get(key)
            if existing is not None:
                if existing["value"] != fact["value"]:
                    raise ValueError(
                        f"H3 shot {row_index + 1} opening conflicts with carried "
                        f"{fact['subject_id']} {fact['predicate']} state: "
                        f"expected {existing['value']!r}, received {fact['value']!r}. "
                        "Omit this carried opening key and use a visible transition "
                        "to change it, or plan an intentional independent cut."
                    )
                continue
            if len(opening) >= MAX_PHYSICAL_FACTS:
                raise ValueError(
                    f"H3 shot {row_index + 1} opening has more than "
                    f"{MAX_PHYSICAL_FACTS} facts after carry-forward"
                )
            current[key] = dict(fact)
            opening.append(dict(fact))

        transitions: list[dict[str, str]] = []
        unchanged_props = {
            subject: item["value"] for (subject, predicate), item in current.items()
            if predicate == "prop"
        }
        prop_transition_subjects = {
            item["subject_id"].casefold() for item in raw_transitions
            if isinstance(item, dict) and item.get("predicate") == "prop"
            and isinstance(item.get("subject_id"), str)
        }
        for transition_index, raw_transition in enumerate(raw_transitions):
            location = (
                f"H3 shot {row_index + 1} physical_state.transitions"
                f"[{transition_index}]"
            )
            if not isinstance(raw_transition, dict):
                raise ValueError(f"{location} must be an object")
            if set(raw_transition) != {
                "subject_id", "predicate", "to_value", "action",
            }:
                raise ValueError(
                    f"{location} must contain only subject_id, predicate, "
                    "to_value, and action"
                )
            subject_id = _compact_state_text(
                raw_transition.get("subject_id"), field="subject_id",
                max_length=80, location=location,
            )
            predicate = raw_transition.get("predicate")
            if predicate not in PHYSICAL_STATE_PREDICATES:
                raise ValueError(
                    f"{location}.predicate must be one of "
                    f"{', '.join(PHYSICAL_STATE_PREDICATES)}"
                )
            to_value = _compact_state_text(
                raw_transition.get("to_value"), field="to_value",
                max_length=MAX_PHYSICAL_VALUE_CHARS, location=location,
            )
            if predicate == "grounded" and to_value not in GROUNDED_STATE_VALUES:
                raise ValueError(
                    f"{location}.to_value for predicate 'grounded' must be "
                    "exactly 'grounded' or 'airborne'"
                )
            action = _compact_state_text(
                raw_transition.get("action"), field="action",
                max_length=MAX_PHYSICAL_TEXT_CHARS, location=location,
            )
            transition_fact = {
                "subject_id": subject_id,
                "predicate": predicate,
                "value": to_value,
            }
            check_visible_owner(transition_fact, location)
            key = _compact_fact_key(transition_fact)
            before = current.get(key)
            if before is None:
                raise ValueError(
                    f"{location} has no explicit opening/current fact for "
                    f"{subject_id} {predicate}"
                )
            if (
                predicate == "posture"
                and subject_id.casefold() not in prop_transition_subjects
                and to_value == unchanged_props.get(subject_id.casefold())
                and _is_prop_manipulation_posture_mislabel(
                    row, subject_id, action, to_value,
                )
            ):
                continue
            transitions.append({
                "subject_id": subject_id,
                "predicate": predicate,
                "from_value": before["value"],
                "to_value": to_value,
                "action": action,
            })
            current[key] = transition_fact

        if len(current) > MAX_PHYSICAL_FACTS:
            raise ValueError(
                f"H3 shot {row_index + 1} closing has more than "
                f"{MAX_PHYSICAL_FACTS} facts"
            )
        materialized = {
            "opening": opening,
            "transitions": transitions,
            "closing": list(current.values()),
            "continuity": continuity,
        }
        row["physical_state"] = normalize_physical_state(materialized)
        prior_state = row["physical_state"]

    return rows


def _materialize_h3_physical_state_proposals(
    shot_dicts: list[Any],
    *,
    previous_physical_state: Any = None,
    preserve_invalid_tail: bool = False,
) -> list[Any]:
    """Materialize rows strictly, optionally retaining a valid prefix.

    Batch planning needs per-row recovery: a malformed later compact proposal
    must not erase earlier rows that can be checkpointed. The first malformed
    row is returned unchanged with a private error marker so the normal row
    validator rejects it and starts bounded tail repair. Direct callers remain
    strict by default.
    """

    if not preserve_invalid_tail:
        return _materialize_h3_physical_state_proposals_strict(
            shot_dicts,
            previous_physical_state=previous_physical_state,
        )
    if not isinstance(shot_dicts, list):
        raise ValueError("H3 shot plans must be an array")

    materialized_rows: list[Any] = []
    prior_state = copy.deepcopy(previous_physical_state)
    for row_index, raw_row in enumerate(shot_dicts):
        try:
            materialized = _materialize_h3_physical_state_proposals_strict(
                [raw_row],
                previous_physical_state=prior_state,
            )
        except (TypeError, ValueError) as exc:
            tail = copy.deepcopy(shot_dicts[row_index:])
            if tail and isinstance(tail[0], dict):
                tail[0][_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD] = str(exc)[:500]
            materialized_rows.extend(tail)
            break
        materialized_rows.extend(materialized)
        last_row = materialized[-1] if materialized else None
        prior_state = (
            copy.deepcopy(last_row.get("physical_state"))
            if isinstance(last_row, dict) else None
        )
    return materialized_rows


def _h3_metadata_repair_row_is_complete(
    row: Any,
    *,
    include_image_fields: bool,
) -> bool:
    """Keep state-only repair for shots whose visible plan is already usable."""

    if not isinstance(row, dict):
        return False
    required_text = [
        "scene_goal", "scene_type", "environment", "visual_style", "lighting",
        "mood", "ending_beat", "video_prompt",
    ]
    if include_image_fields:
        required_text.extend(("image_source", "image_prompt"))
    if any(
        not isinstance(row.get(field), str) or not row[field].strip()
        for field in required_text
    ):
        return False
    if include_image_fields and not isinstance(row.get("visual_changes"), list):
        return False

    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list) or any(
        not isinstance(subject, dict) or not _has_meaningful_cast_identity(subject)
        for subject in subjects
    ):
        return False
    action_beats = row.get("action_beats")
    if not isinstance(action_beats, list) or not any(
        isinstance(beat, str) and beat.strip() for beat in action_beats
    ):
        return False
    camera = row.get("camera_plan")
    if not isinstance(camera, dict) or not str(camera.get("framing") or "").strip():
        return False
    windows = row.get("window_prompts")
    if not isinstance(windows, list) or windows:
        return False
    if row.get("continuity_strategy") not in {
        "continuous", "independent", "extend_previous",
    }:
        return False

    proposal = row.get("physical_state")
    if not isinstance(proposal, dict):
        return False
    if set(proposal) - {"opening", "transitions", "continuity"}:
        return False
    return isinstance(proposal.get("opening"), list) and isinstance(
        proposal.get("transitions"), list
    )


_H3_STATE_ACTION_ROOTS = frozenset({
    "approach", "board", "climb", "crouch", "descend", "drop", "emerge",
    "fall", "face", "flare", "fly", "grip", "hover", "jump", "kneel",
    "land", "launch", "lean", "leap", "leave", "lift", "look", "lower",
    "move", "mount", "pounce", "reach", "release", "retreat", "ride", "rise",
    "roll", "run", "settle", "shift", "sit", "soar", "spin", "stand", "step",
    "swoop", "swing", "turn", "twirl", "walk",
})
_H3_ACTION_MATCH_STOP_WORDS = frozenset({
    "a", "an", "and", "as", "her", "his", "its", "of", "the",
    "their", "with",
})
_H3_SUPPORT_BOARDING_ACTION_ROOTS = frozenset({"board", "climb", "land", "mount"})
_H3_UNSUPPORTED_ACTION_MARKERS = frozenset({
    "already", "appear", "attempt", "can", "cannot", "could", "describe",
    "fail", "hope", "if", "imagine", "intend", "may", "maybe", "might",
    "never", "no", "not", "perhaps", "plan", "possible", "possibly", "pretend",
    "recall", "remember", "report", "say", "should", "talk", "tell", "teleport",
    "try", "unless", "unable", "want", "wish", "without", "would",
})


def _h3_action_match_tokens(value: Any) -> list[str]:
    normalized = re.sub(
        r"n['’]t\b", " not", str(value or "").casefold(),
    )
    return re.findall(r"[^\W_]+", normalized)


def _h3_action_match_roots(token: str) -> set[str]:
    """Return conservative English inflection stems for one action token."""

    roots = {token}
    suffix = "ing" if token.endswith("ing") else "ed" if token.endswith("ed") else ""
    if suffix and len(token) > len(suffix) + 2:
        base = token[:-len(suffix)]
        roots.update((base, base + "e"))
        if len(base) > 2 and base[-1] == base[-2]:
            roots.update((base[:-1], base[:-1] + "e"))
    elif token.endswith("ies") and len(token) > 4:
        roots.add(token[:-3] + "y")
    elif token.endswith("es") and len(token) > 4:
        roots.update((token[:-2], token[:-1]))
    elif token.endswith("s") and len(token) > 3:
        roots.add(token[:-1])
    return roots


def _h3_action_is_asserted(tokens: list[str], index: int) -> bool:
    token = tokens[index]
    previous = tokens[index - 1] if index else ""
    if token.endswith("s"):
        return True
    if token.endswith("ed"):
        return previous not in {
            "am", "are", "be", "been", "being", "has", "have", "had",
            "is", "was", "were",
        }
    if index > 0 and token.endswith("ing") and tokens[index - 1] in {
        "am", "are", "is", "was", "were",
    }:
        return True
    return (
        index > 1
        and tokens[index - 1] == "to"
        and tokens[index - 2] in {
            "begin", "begins", "continue", "continues", "proceed", "proceeds",
            "start", "starts",
        }
    )


def _h3_subject_aliases(row: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    aliases: list[tuple[str, tuple[str, ...]]] = []
    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        return aliases
    for subject in subjects:
        if not isinstance(subject, dict):
            continue
        subject_id = subject.get("character_id")
        if not isinstance(subject_id, str) or not subject_id.strip():
            continue
        names = [subject_id, subject.get("speaker_name")]
        for name in names:
            tokens = tuple(
                token for token in _h3_action_match_tokens(name)
                if token not in {"a", "an", "the"}
            )
            if tokens:
                aliases.append((subject_id.casefold(), tokens))
    return list(dict.fromkeys(aliases))


def _h3_action_beat_for_state_fact(
    row: dict[str, Any],
    fact: dict[str, str],
    *,
    additional_roots: frozenset[str] | set[str] = frozenset(),
) -> Optional[tuple[int, str]]:
    """Find one positive authored beat that explicitly enacts a fact value."""

    value_tokens = [
        token for token in _h3_action_match_tokens(fact.get("value"))
        if token not in _H3_ACTION_MATCH_STOP_WORDS
    ]
    action_roots = [
        root
        for token in value_tokens
        for root in _h3_action_match_roots(token)
        if root in _H3_STATE_ACTION_ROOTS or root in additional_roots
    ]
    if not value_tokens or not action_roots:
        return None
    action_roots = [action_roots[0]]

    aliases = _h3_subject_aliases(row)
    target_id = fact["subject_id"].casefold()
    if not any(owner == target_id for owner, _ in aliases):
        return None
    beats = row.get("action_beats")
    if not isinstance(beats, list):
        return None

    matches: list[tuple[int, str]] = []
    for beat_index, beat in enumerate(beats):
        if not isinstance(beat, str) or not beat.strip():
            continue
        beat_tokens = _h3_action_match_tokens(beat)
        if not beat_tokens or any(
            root in _H3_UNSUPPORTED_ACTION_MARKERS
            for token in beat_tokens
            for root in _h3_action_match_roots(token)
        ):
            continue
        beat_roots = set().union(*(
            _h3_action_match_roots(token) for token in beat_tokens
        ))
        if any(
            not (_h3_action_match_roots(token) & beat_roots)
            for token in value_tokens
        ):
            continue

        action_positions = [
            index for index, token in enumerate(beat_tokens)
            if (
                any(root in _h3_action_match_roots(token) for root in action_roots)
                and _h3_action_is_asserted(beat_tokens, index)
            )
        ]
        if not action_positions:
            continue

        # Attribute each candidate action to the nearest named subject before
        # its verb, so another visible subject cannot supply the evidence.
        owned_action = False
        for action_position in action_positions:
            preceding_mentions: list[tuple[int, str]] = []
            for owner, alias_tokens in aliases:
                span = len(alias_tokens)
                for start in range(0, action_position - span + 1):
                    if tuple(beat_tokens[start:start + span]) == alias_tokens:
                        preceding_mentions.append((start + span, owner))
            if not preceding_mentions:
                continue
            nearest_end = max(end for end, _ in preceding_mentions)
            nearest_owners = {
                owner for end, owner in preceding_mentions if end == nearest_end
            }
            if nearest_owners == {target_id}:
                owned_action = True
                break
        if owned_action:
            matches.append((beat_index, beat.strip()))

    if len(matches) != 1:
        return None
    return matches[0]


def _h3_action_beat_for_transition(
    row: dict[str, Any], transition: dict[str, Any],
) -> Optional[tuple[int, str]]:
    subject_id = transition.get("subject_id")
    action = transition.get("action")
    if not isinstance(subject_id, str) or not isinstance(action, str):
        return None
    action_token = next((
        token for token in _h3_action_match_tokens(action)
        if _h3_action_match_roots(token) & _H3_STATE_ACTION_ROOTS
    ), None)
    if action_token is None:
        return None
    return _h3_action_beat_for_state_fact(
        row, {"subject_id": subject_id, "value": action_token},
    )


def _bridge_h3_conflicting_openings(
    row: dict[str, Any],
    preceding_closing: list[dict[str, str]],
) -> Optional[dict[str, Any]]:
    """Bridge explicit opening actions from exact carried facts, or fail closed."""

    if row.get("continuity_strategy") not in {"continuous", "extend_previous"}:
        return None
    proposal = row.get("physical_state")
    if not isinstance(proposal, dict) or proposal.get("continuity") in {
        "independent", "time_jump",
    }:
        return None
    openings = proposal.get("opening")
    transitions = proposal.get("transitions")
    if not isinstance(openings, list) or not isinstance(transitions, list):
        return None

    prior_by_key = {
        _compact_fact_key(fact): fact for fact in preceding_closing
        if isinstance(fact, dict)
        and isinstance(fact.get("subject_id"), str)
        and isinstance(fact.get("predicate"), str)
        and isinstance(fact.get("value"), str)
    }
    conflicts: list[dict[str, str]] = []
    conflict_keys: set[tuple[str, str]] = set()
    for opening in openings:
        if not isinstance(opening, dict):
            return None
        key = _compact_fact_key(opening) if all(
            isinstance(opening.get(field), str)
            for field in ("subject_id", "predicate")
        ) else None
        prior = prior_by_key.get(key) if key is not None else None
        if prior is not None and opening.get("value") != prior.get("value"):
            if key in conflict_keys:
                return None
            conflict_keys.add(key)
            conflicts.append(opening)
    if not conflicts:
        return None

    bridge_specs: list[tuple[int, dict[str, str]]] = []
    for opening in conflicts:
        action_beat = _h3_action_beat_for_state_fact(row, opening)
        if action_beat is None:
            return None
        beat_index, beat_text = action_beat
        try:
            action = _compact_state_text(
                beat_text,
                field="action",
                max_length=MAX_PHYSICAL_TEXT_CHARS,
                location="H3 metadata repair action beat",
            )
            to_value = _compact_state_text(
                opening.get("value"),
                field="to_value",
                max_length=MAX_PHYSICAL_VALUE_CHARS,
                location="H3 metadata repair opening",
            )
        except ValueError:
            return None
        if opening.get("predicate") not in PHYSICAL_STATE_PREDICATES:
            return None
        if opening["predicate"] == "grounded" and to_value not in GROUNDED_STATE_VALUES:
            return None
        same_key_transitions: list[tuple[int, dict[str, Any]]] = []
        for transition_index, item in enumerate(transitions):
            if not isinstance(item, dict):
                continue
            transition_subject = item.get("subject_id")
            if (
                isinstance(transition_subject, str)
                and transition_subject.casefold() == opening["subject_id"].casefold()
                and item.get("predicate") == opening["predicate"]
            ):
                same_key_transitions.append((transition_index, item))
        destination_transitions = [
            transition_index for transition_index, item in same_key_transitions
            if item.get("to_value") == to_value
        ]
        if destination_transitions:
            if (
                not same_key_transitions
                or destination_transitions[0] != same_key_transitions[0][0]
            ):
                return None
            existing_action_beat = _h3_action_beat_for_transition(
                row, same_key_transitions[0][1],
            )
            if existing_action_beat is None or existing_action_beat[0] != beat_index:
                return None
            continue
        if same_key_transitions:
            first_transition = same_key_transitions[0][1]
            first_destination = first_transition.get("to_value")
            if not isinstance(first_destination, str):
                return None
            first_action_beat = _h3_action_beat_for_state_fact(
                row,
                {
                    "subject_id": opening["subject_id"],
                    "predicate": opening["predicate"],
                    "value": first_destination,
                },
            )
            if first_action_beat is None or beat_index >= first_action_beat[0]:
                return None
        bridge_specs.append((beat_index, {
            "subject_id": opening["subject_id"],
            "predicate": opening["predicate"],
            "to_value": to_value,
            "action": action,
        }))

    if len(transitions) + len(bridge_specs) > MAX_PHYSICAL_TRANSITIONS:
        return None

    bridge_indices = [beat_index for beat_index, _ in bridge_specs]
    if bridge_indices != list(range(len(bridge_indices))):
        return None
    if bridge_indices:
        existing_beat_indices: list[int] = []
        for item in transitions:
            if not isinstance(item, dict):
                return None
            action_beat = _h3_action_beat_for_transition(row, item)
            if action_beat is None or action_beat[0] <= bridge_indices[-1]:
                return None
            existing_beat_indices.append(action_beat[0])
        if existing_beat_indices != sorted(existing_beat_indices):
            return None

    candidate = copy.deepcopy(row)
    candidate.pop(_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD, None)
    candidate_state = copy.deepcopy(proposal)
    candidate_state["opening"] = [
        opening for opening in openings if opening not in conflicts
    ]
    candidate_state["transitions"] = [
        item for _, item in sorted(bridge_specs, key=lambda spec: spec[0])
    ] + copy.deepcopy(transitions)
    candidate["physical_state"] = candidate_state
    return candidate


def _h3_word_tokens(value: Any) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _h3_contains_phrase(text: Any, phrase: Any) -> bool:
    words = _h3_word_tokens(text)
    wanted = _h3_word_tokens(phrase)
    return bool(wanted) and any(
        words[index:index + len(wanted)] == wanted
        for index in range(len(words) - len(wanted) + 1)
    )


def _h3_source_support_patch_spec(
    issues: list[str],
    *,
    row: dict[str, Any],
    cast_ids: list[dict[str, str]],
    preceding_closing: list[dict[str, str]],
    assigned_actions: list[dict[str, Any]],
    repair_seed: dict[str, list[dict[str, str]]],
) -> Optional[dict[str, Any]]:
    """Allow one additive support patch only when its start and owner are grounded."""

    obligations: dict[tuple[str, str], set[str]] = {}
    for issue in issues:
        match = re.search(
            r"source action ['\"](.+?)['\"] requires a "
            r"([a-z_]+(?:/[a-z_]+)*) fact in physical_state\.(opening|closing)",
            issue,
        )
        if match is None:
            continue
        phrase, predicates, endpoint = match.groups()
        for predicate in predicates.split("/"):
            if predicate in PHYSICAL_STATE_PREDICATES:
                obligations.setdefault((phrase.casefold(), predicate), set()).add(endpoint)
    missing_both = [
        (phrase, predicate)
        for (phrase, predicate), endpoints in obligations.items()
        if endpoints == {"opening", "closing"}
    ]
    if len(missing_both) != 1 or missing_both[0][1] != "support":
        return None
    phrase, predicate = missing_both[0]

    seed_items = repair_seed.get("opening", []) + repair_seed.get("transitions", [])
    if any(
        isinstance(item, dict) and item.get("predicate") == predicate
        for item in seed_items
    ):
        return None
    matching_actions = [
        action for action in assigned_actions
        if isinstance(action, dict)
        and _h3_contains_phrase(action.get("text"), phrase)
        and isinstance(action.get("actor_aliases"), list)
        and action.get("actor_aliases")
        and isinstance(action.get("target_aliases"), list)
        and action.get("target_aliases")
    ]
    if len(matching_actions) != 1:
        return None
    source_action = matching_actions[0]
    source_text = source_action.get("text")
    if not isinstance(source_text, str):
        return None
    target_aliases = [
        alias for alias in source_action["target_aliases"]
        if isinstance(alias, str) and alias.strip()
    ]
    matched_target_aliases = [
        alias for alias in target_aliases
        if _h3_contains_phrase(source_text, alias)
    ]
    if len(matched_target_aliases) != 1:
        return None
    target_alias = matched_target_aliases[0]

    owner_ids = {
        cast.get("character_id")
        for cast in cast_ids
        if isinstance(cast.get("character_id"), str)
        and any(
            isinstance(alias, str)
            and any(_h3_contains_phrase(name, alias) for name in cast.values())
            for alias in source_action["actor_aliases"]
        )
    }
    if len(owner_ids) != 1:
        return None
    owner_id = next(iter(owner_ids))
    has_grounded_start = any(
        isinstance(item, dict)
        and isinstance(item.get("subject_id"), str)
        and item["subject_id"].casefold() == owner_id.casefold()
        and isinstance(item.get("value"), str)
        and (
            (item.get("predicate") == "grounded" and item["value"] == "grounded")
            or (
                item.get("predicate") == "contact"
                and (
                    item["value"].casefold() == "grounded"
                    or _h3_support_ground_start(item["value"])
                )
            )
        )
        for item in preceding_closing
    )
    if not has_grounded_start:
        return None
    beats = row.get("action_beats")
    if not isinstance(beats, list):
        return None
    source_roots = list(dict.fromkeys(
        root
        for token in _h3_action_match_tokens(source_text)
        for root in _h3_action_match_roots(token)
        if root in _H3_SUPPORT_BOARDING_ACTION_ROOTS
    ))
    if len(source_roots) != 1:
        return None
    source_tokens = _h3_action_match_tokens(source_text)
    if any(
        root in _H3_UNSUPPORTED_ACTION_MARKERS
        for token in source_tokens
        for root in _h3_action_match_roots(token)
    ) or not any(
        _h3_action_is_asserted(source_tokens, token_index)
        and bool(_h3_action_match_roots(token) & set(source_roots))
        for token_index, token in enumerate(source_tokens)
    ):
        return None
    action_candidates: list[tuple[int, str]] = []
    for beat_index, beat in enumerate(beats[:MAX_PHYSICAL_TRANSITIONS]):
        if (
            not isinstance(beat, str)
            or not beat.strip()
            or len(beat) > MAX_PHYSICAL_TEXT_CHARS
            or not _h3_contains_phrase(beat, target_alias)
        ):
            continue
        single_beat_row = {**row, "action_beats": [beat]}
        if _h3_action_beat_for_state_fact(
            single_beat_row,
            {"subject_id": owner_id, "value": source_roots[0]},
        ) is not None:
            action_candidates.append((beat_index, beat.strip()))
    if len(action_candidates) != 1:
        return None
    _, action_beat = action_candidates[0]
    return {
        "predicate": predicate,
        "subject_id": owner_id,
        "source_action": source_action,
        "target_alias": target_alias,
        "opening_value": "on the ground",
        "to_value": f"on {target_alias}",
        "action": action_beat,
    }


def _h3_support_ground_start(value: str) -> bool:
    return bool(re.fullmatch(
        r"(?:(?:standing|stood|sitting|seated|resting)\s+)?"
        r"(?:on|upon)\s+(?:the\s+)?(?:ground|floor|earth|soil|pavement)",
        re.sub(r"\s+", " ", value).strip(),
        re.IGNORECASE,
    ))


def _h3_support_value_matches_target(value: str, aliases: list[str]) -> bool:
    if re.search(
        r"\b(?:hover(?:s|ed|ing)?|float(?:s|ed|ing)?)\s+above\b|"
        r"\b(?:as\s+if|as\s+though|pretend(?:s|ed|ing)?|"
        r"imagin(?:e|es|ed|ing))\b|"
        r"\b(?:not|never|without|no)\b[^,.;]{0,32}"
        r"\b(?:supported\s+by|mounted\s+on|riding(?:\s+on)?|"
        r"on|onto|upon|astride|aboard)\b|"
        r"\b(?:as\s+if|as\s+though|pretend(?:s|ed|ing)?|"
        r"imagin(?:e|es|ed|ing))\b[^,.;]{0,32}"
        r"\b(?:supported\s+by|mounted\s+on|riding(?:\s+on)?|"
        r"on|onto|upon|astride|aboard)\b",
        value,
        re.IGNORECASE,
    ):
        return False
    relation = re.search(
        r"\b(?:supported\s+by|mounted\s+on|riding\s+on|riding|"
        r"on|onto|upon|astride|aboard)\b",
        value,
        re.IGNORECASE,
    )
    if relation is None:
        return False
    if re.search(r"\b(?:contact|touch(?:es|ed|ing)?)\b", value, re.IGNORECASE):
        return False
    tail = list(_h3_word_tokens(value[relation.end():]))
    if tail and tail[0] in {"a", "an", "the"}:
        tail = tail[1:]
    return any(
        (alias_words := list(_h3_word_tokens(alias)))
        and tail[:len(alias_words)] == alias_words
        for alias in aliases
    )


def _merge_h3_source_support_patch(
    row: dict[str, Any],
    repair_seed: dict[str, list[dict[str, str]]],
    patch: Any,
    spec: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if (
        not isinstance(patch, dict)
        or set(patch) != {"opening_additions", "transition_additions"}
    ):
        return None
    openings = patch.get("opening_additions")
    transitions = patch.get("transition_additions")
    if not isinstance(openings, list) or len(openings) != 1:
        return None
    if not isinstance(transitions, list) or len(transitions) != 1:
        return None

    opening = openings[0]
    transition_item = transitions[0]
    if not isinstance(opening, dict) or set(opening) != {
        "subject_id", "predicate", "value",
    }:
        return None
    if not isinstance(transition_item, dict) or set(transition_item) != {
        "subject_id", "predicate", "to_value", "action",
    }:
        return None
    owner_id = spec["subject_id"]
    predicate = spec["predicate"]
    if any(
        item.get("subject_id") != owner_id or item.get("predicate") != predicate
        for item in (opening, transition_item)
    ):
        return None
    if opening.get("value") != spec["opening_value"]:
        return None
    if transition_item.get("to_value") != spec["to_value"]:
        return None
    if transition_item.get("action") != spec["action"]:
        return None
    try:
        opening_value = _compact_state_text(
            opening.get("value"), field="value",
            max_length=MAX_PHYSICAL_VALUE_CHARS,
            location="H3 source support opening addition",
        )
        to_value = _compact_state_text(
            transition_item.get("to_value"), field="to_value",
            max_length=MAX_PHYSICAL_VALUE_CHARS,
            location="H3 source support transition addition",
        )
        action = _compact_state_text(
            transition_item.get("action"), field="action",
            max_length=MAX_PHYSICAL_TEXT_CHARS,
            location="H3 source support transition addition",
        )
    except ValueError:
        return None
    if not _h3_support_ground_start(opening_value):
        return None
    if not _h3_support_value_matches_target(to_value, [spec["target_alias"]]):
        return None
    if action != spec["action"]:
        return None

    seed_openings = repair_seed.get("opening", [])
    seed_transitions = repair_seed.get("transitions", [])
    opening_key = (owner_id.casefold(), predicate)
    if any(
        isinstance(item, dict)
        and isinstance(item.get("subject_id"), str)
        and _compact_fact_key(item) == opening_key
        for item in seed_openings
    ):
        return None
    if any(
        isinstance(item, dict)
        and isinstance(item.get("subject_id"), str)
        and item.get("subject_id", "").casefold() == owner_id.casefold()
        and item.get("predicate") == predicate
        for item in seed_transitions
    ):
        return None

    added_transition = {
        "subject_id": owner_id,
        "predicate": predicate,
        "to_value": to_value,
        "action": action,
    }
    added_beat = _h3_action_beat_for_transition(row, added_transition)
    if added_beat is None:
        return None
    tagged: list[tuple[int, int, dict[str, str]]] = []
    last_beat = -1
    for index, item in enumerate(seed_transitions):
        if not isinstance(item, dict):
            return None
        action_beat = _h3_action_beat_for_transition(row, item)
        if action_beat is None or action_beat[0] < last_beat:
            return None
        last_beat = action_beat[0]
        tagged.append((action_beat[0], index, copy.deepcopy(item)))
    tagged.append((added_beat[0], len(tagged), added_transition))
    tagged.sort(key=lambda item: (item[0], item[1]))

    physical_state = {
        "opening": copy.deepcopy(seed_openings) + [{
            "subject_id": owner_id,
            "predicate": predicate,
            "value": opening_value,
        }],
        "transitions": [item[2] for item in tagged],
    }
    continuity = row.get("physical_state", {}).get("continuity")
    if continuity in {"continuous", "independent", "time_jump"}:
        physical_state["continuity"] = continuity
    candidate = copy.deepcopy(row)
    candidate.pop(_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD, None)
    candidate["physical_state"] = physical_state
    return candidate


def _try_repair_h3_physical_state_only(
    planner: Any,
    *,
    index: int,
    row: dict[str, Any],
    issue: str,
    clip: dict[str, Any],
    previous: Optional[dict[str, Any]],
    contract: Any,
    allocation: Any,
    include_image_fields: bool,
) -> Optional[list[dict[str, Any]]]:
    """Repair one H3 compact-state materialization failure without rewriting its shot.

    ``None`` means this failure is outside the narrow metadata-only path and
    should use the ordinary row repair. An empty list means the metadata call
    was malformed; the batch runner owns the existing bounded retry. A
    schema-shaped proposal that still fails materialization returns its tagged
    row so the next attempt can see the updated, exact materializer error.
    """

    if (
        "compact physical-state proposal could not be materialized" not in issue
        or not isinstance(row, dict)
        or not row.get(_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD)
        or getattr(planner, "_last_music_narrative_issue", "")
        or getattr(planner, "_last_music_creative_issue", "")
        or not _h3_metadata_repair_row_is_complete(
            row,
            include_image_fields=include_image_fields,
        )
    ):
        return None

    source_ids = row.get("source_event_ids")
    source_ids = source_ids if isinstance(source_ids, list) else []
    assigned_ids = [
        item.get("event_id")
        for item in (allocation if isinstance(allocation, list) else [])
        if isinstance(item, dict) and item.get("clip_index") == index
        and isinstance(item.get("event_id"), str)
    ]
    event_ids = assigned_ids or [
        event_id for event_id in source_ids if isinstance(event_id, str)
    ]
    event_by_id = {
        str(event.get("event_id") or ""): event
        for event in (
            contract.get("events", [])
            if isinstance(contract, dict) and isinstance(contract.get("events"), list)
            else []
        )
        if isinstance(event, dict)
    }
    assigned_actions = []
    for event_id in event_ids:
        event = event_by_id.get(event_id)
        if event is None:
            continue
        action = {
            "event_id": event_id,
            "text": event.get("text", ""),
            "kind": event.get("kind", ""),
        }
        for key in ("actor_aliases", "target_aliases"):
            aliases = event.get(key)
            if isinstance(aliases, list):
                action[key] = [
                    value for value in aliases[:16]
                    if isinstance(value, str) and value.strip()
                ]
        assigned_actions.append(action)
    state = normalize_physical_state(
        previous.get("physical_state") if isinstance(previous, dict) else None
    )
    preceding_closing = state["closing"] if state else []
    proposal = row.get("physical_state")
    proposal = proposal if isinstance(proposal, dict) else {}
    original_proposal = copy.deepcopy(proposal)
    opening = proposal.get("opening")
    opening = opening if isinstance(opening, list) else []
    transitions = proposal.get("transitions")
    transitions = transitions if isinstance(transitions, list) else []
    carries_previous = (
        row.get("continuity_strategy") in {"continuous", "extend_previous"}
        and proposal.get("continuity") not in {"independent", "time_jump"}
    )
    bridge_source_issues: list[str] = []
    bridge_conflicts: list[dict[str, str]] = []
    if carries_previous and preceding_closing:
        prior_by_key = {
            _compact_fact_key(item): item for item in preceding_closing
            if isinstance(item, dict)
            and isinstance(item.get("subject_id"), str)
            and isinstance(item.get("predicate"), str)
            and isinstance(item.get("value"), str)
        }
        for opening_item in opening:
            if not isinstance(opening_item, dict):
                continue
            subject_id = opening_item.get("subject_id")
            predicate = opening_item.get("predicate")
            value = opening_item.get("value")
            if not all(isinstance(item, str) for item in (subject_id, predicate, value)):
                continue
            prior = prior_by_key.get((subject_id.casefold(), predicate))
            if prior is not None and prior.get("value") != value:
                bridge_conflicts.append({
                    "subject_id": subject_id,
                    "predicate": predicate,
                    "authoritative_value": prior["value"],
                    "proposed_value": value,
                })
        bridge_candidate = _bridge_h3_conflicting_openings(
            row, preceding_closing,
        )
        if bridge_candidate is not None:
            try:
                bridged = _materialize_h3_physical_state_proposals(
                    [bridge_candidate],
                    previous_physical_state=(
                        previous.get("physical_state")
                        if isinstance(previous, dict) else None
                    ),
                    preserve_invalid_tail=True,
                )
            except (TypeError, ValueError):
                bridged = []
            if (
                len(bridged) == 1
                and isinstance(bridged[0], dict)
                and not bridged[0].get(_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD)
            ):
                source_events = _assigned_source_events_for_clip(clip) or assigned_actions
                bridge_source_issues = source_physical_state_issues(
                    [bridged[0]], [source_events],
                )
                if not bridge_source_issues:
                    return bridged
                row = bridged[0]
                proposal = row["physical_state"]
                opening = proposal["opening"]
                transitions = proposal["transitions"]

    known_current_keys: list[dict[str, str]] = []
    known_key_pairs: set[tuple[str, str]] = set()
    known_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    conflicting_openings: list[dict[str, str]] = []

    def add_known_key(raw: Any, source: str) -> None:
        if not isinstance(raw, dict):
            return
        subject_id = raw.get("subject_id")
        predicate = raw.get("predicate")
        value = raw.get("value")
        if not all(isinstance(item, str) and item.strip() for item in (subject_id, predicate, value)):
            return
        key = (subject_id.casefold(), predicate)
        if key in known_by_pair:
            return
        known_key_pairs.add(key)
        known_fact = {
            "subject_id": subject_id,
            "predicate": predicate,
            "value": value,
            "source": source,
        }
        known_by_pair[key] = known_fact
        known_current_keys.append(known_fact)

    if carries_previous:
        for fact_item in preceding_closing:
            add_known_key(fact_item, "preceding_closing")
    for fact_item in opening[:MAX_PHYSICAL_FACTS]:
        if not isinstance(fact_item, dict):
            continue
        subject_id = fact_item.get("subject_id")
        predicate = fact_item.get("predicate")
        value = fact_item.get("value")
        if not all(
            isinstance(item, str) and item.strip()
            for item in (subject_id, predicate, value)
        ):
            continue
        known = known_by_pair.get((subject_id.casefold(), predicate))
        if known is None:
            add_known_key(fact_item, "proposed_opening")
        elif known["value"] != value:
            conflicting_openings.append({
                "subject_id": subject_id,
                "predicate": predicate,
                "authoritative_value": known["value"],
                "authoritative_source": known["source"],
                "proposed_value": value,
            })

    missing_transition_keys: list[dict[str, str]] = []
    for transition_item in transitions[:MAX_PHYSICAL_TRANSITIONS]:
        if not isinstance(transition_item, dict):
            continue
        subject_id = transition_item.get("subject_id")
        predicate = transition_item.get("predicate")
        if not isinstance(subject_id, str) or not subject_id.strip() or not isinstance(predicate, str):
            continue
        if (subject_id.casefold(), predicate) in known_key_pairs:
            continue
        missing_transition_keys.append({
            "subject_id": subject_id,
            "predicate": predicate,
            "to_value": str(transition_item.get("to_value") or ""),
            "action": str(transition_item.get("action") or ""),
            "required_start_fact": "an explicit opening fact or carried prior closing fact",
        })
    cast_ids = []
    subjects = row.get("subjects_on_screen")
    if isinstance(subjects, list):
        for subject in subjects[:32]:
            if not isinstance(subject, dict):
                continue
            identity = {
                key: subject[key]
                for key in ("character_id", "speaker_name")
                if isinstance(subject.get(key), str) and subject[key].strip()
            }
            if identity:
                cast_ids.append(identity)

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "physical_state": _h3_physical_state_proposal_schema(),
            },
            "required": ["physical_state"],
            "additionalProperties": False,
        },
        "minItems": 1,
        "maxItems": 1,
    }
    repair_seed_state = row.get("physical_state")
    repair_seed = None
    if isinstance(repair_seed_state, dict):
        seed_opening = repair_seed_state.get("opening")
        seed_transitions = repair_seed_state.get("transitions")
        repair_seed = {
            "opening": [
                {
                    key: item[key]
                    for key in ("subject_id", "predicate", "value")
                    if key in item
                }
                for item in seed_opening[:MAX_PHYSICAL_FACTS]
                if isinstance(item, dict)
            ] if isinstance(seed_opening, list) else [],
            "transitions": [
                {
                    key: item[key]
                    for key in ("subject_id", "predicate", "to_value", "action")
                    if key in item
                }
                for item in seed_transitions[:MAX_PHYSICAL_TRANSITIONS]
                if isinstance(item, dict)
            ] if isinstance(seed_transitions, list) else [],
        }
    source_issue_summary = "; ".join(bridge_source_issues)
    context = {
        "target_clip_index_zero_based": index,
        "target_clip_number": index + 1,
        "musical_clip_context": clip,
        "validation_issue": (
            f"Outstanding source physical-state issues: {source_issue_summary}"
            if bridge_source_issues else issue
        ),
        "resolved_original_issue": issue if bridge_source_issues else None,
        "cast_ids": cast_ids,
        "preceding_authoritative_closing_facts": preceding_closing,
        "known_current_physical_keys": known_current_keys,
        "conflicting_proposed_opening_keys": conflicting_openings,
        "bridged_conflicting_opening_keys": (
            [] if bridge_source_issues else bridge_conflicts
        ),
        "bridge_candidate_source_physical_state_issues": bridge_source_issues,
        "transition_keys_missing_prior_or_opening_fact": missing_transition_keys,
        "source_event_ids": event_ids,
        "assigned_source_actions": assigned_actions,
        "existing_action_beats": row.get("action_beats", []),
        "continuity_strategy": row.get("continuity_strategy", ""),
        "original_visible_prompt": row.get("video_prompt", ""),
        "original_compact_physical_state_proposal": (
            None if bridge_source_issues else original_proposal
        ),
        "repair_seed_compact_physical_state_proposal": repair_seed,
        "clip_source_actions": _assigned_source_events_for_clip(clip),
    }
    source_support_patch_spec = (
        _h3_source_support_patch_spec(
            bridge_source_issues,
            row=row,
            cast_ids=cast_ids,
            preceding_closing=preceding_closing,
            assigned_actions=assigned_actions,
            repair_seed=repair_seed,
        )
        if bridge_source_issues and repair_seed is not None else None
    )
    if bridge_source_issues:
        user_prompt = (
            "Repair only compact physical_state metadata for this H3 clip. Return "
            "exactly one array item containing only physical_state; the caller keeps "
            "all visible shot fields unchanged. The repair seed already contains the "
            "deterministic opening bridges. Preserve those bridges and every supported "
            "contact or posture fact. Treat posture as body pose and prop as object "
            "manipulation; preserve seated/standing pose during unrelated prop gestures. "
            "The initial carried-state mismatch is resolved; "
            "repair only the outstanding source issues. Returning the seed unchanged "
            "is invalid.\n\n"
            "Use actor_aliases and target_aliases in assigned_source_actions to bind "
            "each required relation to its source owner and target. Add only facts or "
            "transitions supported by those source actions, authored beats, or carried "
            "physical facts. A required opening relation may describe the supported "
            "pre-action state when carried grounded/contact facts and the visible action "
            "establish it. Do not invent a starting relation or target. Closing is "
            "derived and cannot be supplied as a field: to reach a required closing "
            "relation, include its supported opening fact and an authored transition "
            "to the exact supported target.\n\n"
            + json.dumps(context, ensure_ascii=False)
        )
    else:
        user_prompt = (
            "Repair only the compact physical_state metadata for this one H3 music-video clip. "
            "Return exactly one array item containing only physical_state. The caller will "
            "preserve the original visible prompt, action_beats, source_event_ids, cast IDs, "
            "continuity_strategy, and every other shot field unchanged.\n\n"
            "Treat posture as body pose and prop as object manipulation; preserve "
            "seated/standing pose during unrelated prop gestures unless the body visibly "
            "changes pose.\n\n"
            "Use the validation issue and visible evidence below. The known-current list "
            "shows keys available at the clip start, with each key listed once. The "
            "preceding_authoritative_closing_facts are authoritative for carried values. "
            "Omit every value listed in conflicting_proposed_opening_keys from opening. "
            "When one of those different values is the destination of an explicit positive "
            "action already present in the visible prompt or action beats, represent that "
            "change as a transition from the exact carried value; otherwise keep the carried "
            "value unchanged. Preserve existing supported transitions and their specific "
            "destination values and actions. Every transition key must already "
            "have an explicit opening fact or a carried prior closing fact. For each key "
            "listed as missing, add a genuinely supported opening fact before keeping a "
            "relevant transition; otherwise omit that transition. A transition cannot "
            "supply its own starting fact. For a continuous clip, treat preceding closing "
            "facts as authoritative and inherit unchanged keys. Add explicit supported "
            "opening facts only for genuinely new keys. For predicate grounded, use exactly "
            "'grounded' or 'airborne' as the value; this restriction applies only to that "
            "predicate. Keep descriptive posture, contact, support, location, and prop values "
            "specific; never replace them with the generic word 'grounded'. "
            "Do not encode unchanged or decorative motion (for example, cloth fluttering "
            "in persistent wind) as a physical-state transition. Omit irrelevant state "
            "metadata while leaving the original visible prose intact. Keep each remaining "
            "transition action grounded in the original prompt or action beats. Do not "
            "change or paraphrase the visible shot. If a repair seed is supplied, preserve "
            "its already-added deterministic opening bridges and all supported existing "
            "facts and transitions. Address every listed source physical-state issue by "
            "adding required opening/closing relation facts or transitions only when the "
            "assigned source action or authored beats explicitly support their owner and "
            "target. Never erase a supported contact/posture fact or infer a target the "
            "source does not name.\n\n"
            + json.dumps(context, ensure_ascii=False)
        )
    if source_support_patch_spec is not None:
        response = [{
            "physical_state_patch": {
                "opening_additions": [{
                    "subject_id": source_support_patch_spec["subject_id"],
                    "predicate": source_support_patch_spec["predicate"],
                    "value": source_support_patch_spec["opening_value"],
                }],
                "transition_additions": [{
                    "subject_id": source_support_patch_spec["subject_id"],
                    "predicate": source_support_patch_spec["predicate"],
                    "to_value": source_support_patch_spec["to_value"],
                    "action": source_support_patch_spec["action"],
                }],
            },
        }]
    else:
        try:
            response = planner._call_llm_json(
                user_prompt=user_prompt,
                system_prompt=(
                    "Edit only compact physical-state metadata. Follow the supplied strict "
                    "JSON schema and return JSON only. Do not rewrite the shot or invent "
                    "visible actions."
                ),
                max_tokens=1800,
                temperature=0.1,
                thinking_budget=0,
                json_schema=schema,
            )
        except InterruptedError:
            raise
        except Exception:
            return []

    if source_support_patch_spec is not None:
        if (
            not isinstance(response, list)
            or len(response) != 1
            or not isinstance(response[0], dict)
            or set(response[0]) != {"physical_state_patch"}
        ):
            return []
        candidate = _merge_h3_source_support_patch(
            row,
            repair_seed,
            response[0]["physical_state_patch"],
            source_support_patch_spec,
        )
        if candidate is None:
            return []
    else:
        if (
            not isinstance(response, list)
            or len(response) != 1
            or not isinstance(response[0], dict)
            or set(response[0]) != {"physical_state"}
            or not isinstance(response[0].get("physical_state"), dict)
            or set(response[0]["physical_state"]) != {"opening", "transitions"}
        ):
            return []
        candidate = copy.deepcopy(row)
        candidate.pop(_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD, None)
        candidate["physical_state"] = copy.deepcopy(response[0]["physical_state"])
    try:
        materialized = _materialize_h3_physical_state_proposals(
            [candidate],
            previous_physical_state=(
                previous.get("physical_state") if isinstance(previous, dict) else None
            ),
            preserve_invalid_tail=True,
        )
    except (TypeError, ValueError):
        return []
    if len(materialized) != 1 or not isinstance(materialized[0], dict):
        return []
    if (
        source_support_patch_spec is not None
        and materialized[0].get(_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD)
    ):
        return []
    if source_support_patch_spec is not None:
        source_events = _assigned_source_events_for_clip(clip) or assigned_actions
        if source_physical_state_issues([materialized[0]], [source_events]):
            return []
    return materialized


def _discard_unused_image_fields(shot_dicts: list[dict]) -> list[dict]:
    for shot in shot_dicts:
        if isinstance(shot, dict):
            for field in _MUSIC_IMAGE_FIELDS:
                shot.pop(field, None)
    return shot_dicts


# ── Performer Map Parsing ────────────────────────────────────────────

_PRONOUN_MAP = {
    "he": "man", "she": "woman", "him": "man", "her": "woman",
    "guy": "man", "girl": "woman", "boy": "man",
    "male": "man", "female": "woman",
    "rapper": "man", "singer": "woman",
}

_SECTION_ALIASES = {
    "verse": ["verse", "verses", "rap", "raps"],
    "chorus": ["chorus", "choruses", "hook", "hooks", "sing", "sings"],
    "bridge": ["bridge", "bridges"],
    "intro": ["intro", "introduction"],
    "outro": ["outro", "ending"],
    "instrumental": ["instrumental", "break"],
}


def _parse_performer_map(scene_description: str) -> dict[str, str]:
    """Extract section→performer mapping from natural language scene description.

    Returns: {"verse": "the man in the dark jacket", "chorus": "the woman", ...}
    """
    result = {}
    if not scene_description:
        return result

    text = scene_description.lower()
    # Pattern: "the man raps the verses", "chorus by the woman", etc.
    for section, aliases in _SECTION_ALIASES.items():
        for alias in aliases:
            patterns = [
                rf'(\bthe\s+\w+(?:\s+\w+)?)\s+(?:raps?|sings?|performs?|does?|handles?)\s+(?:the\s+)?{alias}',
                rf'{alias}\s+(?:by|from|performed by|sung by|rapped by)\s+(\bthe\s+\w+(?:\s+\w+)?)',
                rf'(\bthe\s+\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:on|in|doing)\s+(?:the\s+)?{alias}',
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    performer = m.group(1).strip()
                    # Normalize pronouns
                    for pronoun, replacement in _PRONOUN_MAP.items():
                        performer = re.sub(rf'\b{pronoun}\b', replacement, performer)
                    result[section] = performer
                    break
    return result


class MusicVideoPlanner(BasePlanner):
    skill_type = "music_video"

    _LONG_FORM_BATCH_SIZE = 6

    def plan(
        self,
        clips: list[dict],
        scene_description: str,
        lyrics: Optional[list[dict]] = None,
        bpm: float = 120.0,
        reference_image_path: Optional[str] = None,
        speaker_mappings: Optional[dict] = None,
        characters: Optional[list[dict]] = None,
        **kwargs,
    ) -> ProductionPlan:
        """Create a ProductionPlan for a music video.

        Args:
            clips: List of PlannedClip dicts with beat_count, duration_frames, label, start, end, etc.
            scene_description: User's scene concept/vibe/setting.
            lyrics: Transcribed lyrics with optional speaker tags.
            bpm: Song tempo.
            reference_image_path: Path to reference photo (optional).
            speaker_mappings: {speaker_id: {name, role}} from UI.
            characters: List of character dicts [{name, description}].
        """
        has_reference = bool(reference_image_path)
        video_model = str(kwargs.get("video_model") or "")
        self._use_lyrics_context = lyrics_context_enabled(kwargs.get("use_lyrics_context"))
        shot_image_policy = str(kwargs.get("shot_image_policy") or "")
        self._uses_generated_shot_images = shot_image_policy not in {
            "prompt_only",
            "direct_references",
        }
        self._preserve_video_character_names = (
            video_model.lower().startswith("minimax_h3")
            and shot_image_policy in {"prompt_only", "direct_references"}
        )
        performer_map = _parse_performer_map(scene_description)
        source_audio_drives_vocals = video_model.lower().startswith(
            ("minimax_h3", "ltx2_25")
        )
        vocal_activity = (
            classify_vocal_intervals(
                clips, lyrics, kwargs.get("audio_vocals_path")
            )
            if source_audio_drives_vocals else []
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
        char_profiles = self._build_characters(characters, speaker_mappings, lyrics, performer_map)

        # Build speaker lookup
        speaker_names = self._build_speaker_names(speaker_mappings, lyrics)

        self._configure_planning_runtime(
            kwargs,
            kind="music_video",
            fingerprint_payload={
                "planner_revision": 25,
                **(
                    {"h3_music_physical_state_policy": "visual-prose-v1"}
                    if video_model.lower().startswith("minimax_h3") else {}
                ),
                "use_lyrics_context": self._use_lyrics_context,
                "scene_description": scene_description,
                "clips": clips,
                "lyrics": lyrics or [],
                "bpm": bpm,
                "reference_image_path": reference_image_path,
                "speaker_mappings": speaker_mappings or {},
                "characters": characters or [],
                "video_model": video_model,
                "image_model": kwargs.get("image_model", ""),
                "shot_image_policy": shot_image_policy,
                "vocal_activity": vocal_activity,
                "character_ref_labels": kwargs.get("character_ref_labels") or [],
                "source_participant_references": [
                    ref.get("name") or ref.get("role", "")
                    for ref in (kwargs.get("minimax_h3_references") or [])
                    if isinstance(ref, dict) and ref.get("type") in {"image", "video"}
                ],
                "location_ref_labels": kwargs.get("location_ref_labels") or [],
            },
        )

        self._music_narrative_contract = (
            build_music_narrative_contract(
                scene_description,
                participant_hints=[
                    *(kwargs.get("character_ref_labels") or []),
                    *[
                        ref.get("name") or ref.get("role", "")
                        for ref in (kwargs.get("minimax_h3_references") or [])
                        if isinstance(ref, dict) and ref.get("type") in {"image", "video"}
                    ],
                ],
            )
            if video_model.lower().startswith("minimax_h3") else {"events": [], "source": scene_description}
        )
        self._music_arc = self._plan_music_arc(
            clips, scene_description, lyrics=lyrics,
        ) if len(clips) >= 4 else []
        if self._music_arc:
            from ..music_arc import arc_allocation
            self._music_narrative_allocation = arc_allocation(self._music_arc)
        else:
            self._music_narrative_allocation = self._allocate_narrative_actions(
                self._music_narrative_contract, clips, scene_description,
            )
        self._last_music_narrative_issue = ""

        # Build reference assets
        ref_assets = ReferenceAssets(
            start_image=AssetRef(id="ref_image", type="image", uri=reference_image_path) if has_reference else None,
            lyrics=(self._format_lyrics_text(lyrics)
                    if lyrics and self._use_lyrics_context else None),
            speaker_map=[
                SpeakerMapEntry(speaker_id=sid, name=info.get("name", ""), voice_description=info.get("role", ""))
                for sid, info in (speaker_mappings or {}).items()
            ] if speaker_mappings else None,
        )

        # Build clip context for the LLM. H3 and LTX-2.5 receive the actual
        # source-audio slice for every shot, so a transcript must not become
        # newly authored prompt dialogue. LTX-2.5 also responds much more
        # reliably when ``lip-syncs`` is stated explicitly.
        clip_contexts = self._build_clip_contexts(
            clips,
            lyrics,
            performer_map,
            speaker_names,
            speaker_mappings,
            source_audio_drives_vocals=source_audio_drives_vocals,
            vocal_activity=vocal_activity,
            use_lyrics_context=self._use_lyrics_context,
        )

        # Call LLM for creative planning
        nsfw = kwargs.get("nsfw", False)
        shot_dicts = self._plan_with_llm(
            clips=clips,
            clip_contexts=clip_contexts,
            scene_description=scene_description,
            bpm=bpm,
            has_reference=has_reference,
            reference_image_path=reference_image_path,
            char_profiles=char_profiles,
            performer_map=performer_map,
            nsfw=nsfw,
            **{k: v for k, v in kwargs.items() if k not in ("nsfw",)},
        )

        # ── Image-prompt sanitization (Layer 1) ──────────────────────
        # Strip GARMENT BAN violations and narrative-filler phrases the
        # image model can't render. Mirrors the same hook in short_film.py
        # so music-video shots also benefit from the deterministic cleanup
        # regardless of whether Pass 3 polish is enabled. See
        # prompt_polish.sanitize_image_prompt for the full ruleset.
        try:
            from ..prompt_polish import sanitize_image_prompt as _sanitize_ip
            for sd in shot_dicts:
                ip = sd.get("image_prompt") or ""
                if ip.strip():
                    sd["image_prompt"] = _sanitize_ip(
                        ip, log_prefix=f"[MusicVideoPlanner Pass2 image sanitize '{sd.get('scene_goal', 'untitled')[:40]}']"
                    )
                kfs = sd.get("keyframe_prompts") or []
                if isinstance(kfs, list) and kfs:
                    cleaned_kfs = []
                    for ki, kf in enumerate(kfs):
                        if isinstance(kf, str) and kf.strip():
                            cleaned_kfs.append(_sanitize_ip(
                                kf, log_prefix=f"[MusicVideoPlanner Pass2 keyframe[{ki}] sanitize]"
                            ))
                        else:
                            cleaned_kfs.append(kf)
                    sd["keyframe_prompts"] = cleaned_kfs
        except Exception as e:
            print(f"[MusicVideoPlanner] Image-prompt sanitization skipped: {e}")

        # ── Sex-act leet trigger strip (always-on safety net) ────────
        # User-reported leak: a SFW music video about a football coach
        # had "bl0wj0b" in a keyframe_prompt because the LLM treated
        # leet-coded LoRA triggers as generic energy boosters. The
        # leet tokens (bl0wj0b, m15510n4ry, c0wg1rl, etc.) are SEX-ACT
        # video-LoRA triggers — they should NEVER appear in:
        #   - image_prompt or keyframe_prompts (still images, video
        #     LoRA triggers don't apply)
        #   - SFW content of any kind
        # Strip runs on EVERY prompt field unconditionally for image
        # and keyframe fields. For video_prompt and window_prompts,
        # strip only when the surrounding screenplay/scene context
        # is SFW (handled below via nsfw flag).
        try:
            from ..prompt_polish import strip_sex_act_leet_tokens
            total_stripped = 0
            for sd in shot_dicts:
                if not isinstance(sd, dict):
                    continue
                # Image + keyframe: always strip (still images don't
                # use video LoRA triggers regardless of NSFW mode).
                ip = sd.get("image_prompt") or ""
                if ip:
                    new_ip, n = strip_sex_act_leet_tokens(ip)
                    if n:
                        sd["image_prompt"] = new_ip
                        total_stripped += n
                kfs = sd.get("keyframe_prompts") or []
                if isinstance(kfs, list):
                    new_kfs = []
                    for kf in kfs:
                        if isinstance(kf, str):
                            new_kf, n = strip_sex_act_leet_tokens(kf)
                            new_kfs.append(new_kf)
                            total_stripped += n
                        else:
                            new_kfs.append(kf)
                    sd["keyframe_prompts"] = new_kfs
                # Video / windows: strip only if NSFW mode is OFF
                # (effective_nsfw could differ from request nsfw; we
                # use the planner's own nsfw flag which is wired
                # from services config).
                if not nsfw:
                    vp = sd.get("video_prompt") or ""
                    if vp:
                        new_vp, n = strip_sex_act_leet_tokens(vp)
                        if n:
                            sd["video_prompt"] = new_vp
                            total_stripped += n
                    wps = sd.get("window_prompts") or []
                    if isinstance(wps, list):
                        new_wps = []
                        for w in wps:
                            if isinstance(w, str):
                                new_w, n = strip_sex_act_leet_tokens(w)
                                new_wps.append(new_w)
                                total_stripped += n
                            else:
                                new_wps.append(w)
                        sd["window_prompts"] = new_wps
            if total_stripped:
                print(
                    f"[MusicVideoPlanner] Stripped {total_stripped} sex-act "
                    f"leet trigger token(s) (bl0wj0b/m15510n4ry/c0wg1rl/etc.) "
                    f"from prompts that should never contain them. The LLM "
                    f"misplaced trigger words from the active LoRA selection."
                )
        except Exception as e:
            print(f"[MusicVideoPlanner] Leet trigger strip skipped: {e}")

        # Convert LLM output to ShotPlan objects
        shots = self._convert_to_shots(
            shot_dicts=shot_dicts,
            clips=clips,
            char_profiles=char_profiles,
            has_reference=has_reference,
            performer_map=performer_map,
            lyrics=lyrics,
            speaker_names=speaker_names,
            vocal_activity=vocal_activity,
            project_context=scene_description,
            ignore_physical_state_ledger=video_model.lower().startswith("minimax_h3"),
        )

        total_duration = sum(c.get("end", 0) - c.get("start", 0) for c in clips) if clips else None

        return ProductionPlan(
            skill_type="music_video",
            title=None,
            global_style=scene_description,
            total_duration_sec=total_duration,
            reference_assets=ref_assets,
            characters=char_profiles if char_profiles else None,
            shots=shots,
            continuity_notes=[
                "Music video — visual variety across sections is important",
                "Chorus clips should feel higher energy than verses",
                "Performer must be visible when assigned to a clip",
            ],
        )

    # ── Whole-song creative direction ────────────────────────────────

    def _plan_music_arc(self, clips, scene_description, *, lyrics=None):
        """Plan progression before detail, bounded even for hour-long songs."""
        from ..music_arc import (
            creative_plan_schema, normalize_music_arc, music_arc_quality_issues,
            music_arc_action_issues, music_arc_repetition_findings,
            music_arc_reprise_reason_findings,
            ground_music_combat_arc,
        )
        cached = self._planning_resume_checkpoint.get("music_creative_arc")
        contract = self._music_narrative_contract
        self._music_arc_assignment_notes = []
        self._music_arc_quality_notes = []
        combat_recovery_notes = []
        group_size = max(1, math.ceil(len(clips) / 48))
        groups = []
        for start in range(0, len(clips), group_size):
            end = min(len(clips), start + group_size)
            groups.append({
                "start": clips[start].get("start", 0),
                "end": clips[end - 1].get("end", 0),
                "label": clips[start].get("label") or clips[start].get("section_label") or "scene",
                "energy": clips[start].get("energy"),
            })

        def quality_notes_for(outline):
            """Map zero-based outline findings onto global detailed-clip ranges."""
            repetition_findings = music_arc_repetition_findings(outline)
            reprise_findings = music_arc_reprise_reason_findings(outline)
            reprise_by_index = {finding["clip_index"]: finding for finding in reprise_findings}
            paired_indexes = {finding["clip_index"] for finding in repetition_findings}
            notes = copy.deepcopy(combat_recovery_notes)

            def clip_range(index):
                first = index * group_size
                return first, min(len(clips), first + group_size) - 1

            def clip_label(first, last):
                return (
                    f"global clip {first + 1}" if first == last
                    else f"global clips {first + 1}-{last + 1}"
                )

            for finding in repetition_findings:
                current = finding["clip_index"]
                prior = finding["prior_clip_index"]
                current_first, current_last = clip_range(current)
                prior_first, prior_last = clip_range(prior)
                message = (
                    f"{clip_label(current_first, current_last)} has a very similar visual beat and "
                    f"purpose, environment, and camera treatment to {clip_label(prior_first, prior_last)}. "
                    "This is advisory; develop distinct visible progression while preserving the concept, "
                    "cast, source assignment, and pacing."
                )
                if current in reprise_by_index:
                    message += " The reprise label has no concise reason; no reason was invented."
                notes.append({
                    "kind": "repeated_visual_treatment",
                    "clip_index": current_first,
                    "clip_end_index": current_last,
                    "prior_clip_index": prior_first,
                    "prior_clip_end_index": prior_last,
                    "evidence": copy.deepcopy(finding["evidence"]),
                    "message": message[:360],
                })
                if len(notes) >= 12:
                    break

            if len(notes) < 12:
                for finding in reprise_findings:
                    index = finding["clip_index"]
                    if index in paired_indexes:
                        continue
                    first, last = clip_range(index)
                    notes.append({
                        "kind": "missing_reprise_reason",
                        "clip_index": first,
                        "clip_end_index": last,
                        "prior_clip_index": None,
                        "prior_clip_end_index": None,
                        "evidence": copy.deepcopy(finding["evidence"]),
                        "message": (
                            f"{clip_label(first, last)} is marked as an intentional reprise without a "
                            "concise reason. Keep this as an advisory; do not invent a reason."
                        )[:360],
                    })
                    if len(notes) >= 12:
                        break
            if len(notes) < 12:
                early_allowed = bool(re.search(
                    r"\b(?:immediately|right away|from the start)\b",
                    scene_description, re.I,
                ))
                pacing_issues = [
                    issue for issue in music_arc_quality_issues(
                        outline, contract, allow_early_events=early_allowed,
                    )
                    if issue.startswith("Repairable planning issue:")
                ]
                if pacing_issues:
                    assigned_groups = [
                        row["clip_index"] for row in outline
                        if row.get("source_event_ids")
                    ]
                    if assigned_groups:
                        first, _first_end = clip_range(min(assigned_groups))
                        _last_start, last = clip_range(max(assigned_groups))
                        notes.append({
                            "kind": "source_event_pacing",
                            "clip_index": first,
                            "clip_end_index": last,
                            "prior_clip_index": None,
                            "prior_clip_end_index": None,
                            "evidence": {"assigned_group_indices": assigned_groups[:16]},
                            "message": (
                                "Source-action payoffs cluster near the opening. This is advisory: "
                                "keep their exact source order and assignments, and give later sections "
                                "room to develop the resulting state."
                            ),
                        })
            return notes

        if isinstance(cached, list):
            try:
                normalized = normalize_music_arc(cached, clips, contract)
                self._music_arc_assignment_notes = music_arc_action_issues(normalized, contract)
                saved_notes = self._planning_resume_checkpoint.get("music_arc_quality_notes")
                if isinstance(saved_notes, list):
                    self._music_arc_quality_notes = copy.deepcopy(
                        [note for note in saved_notes if isinstance(note, dict)][:12]
                    )
                else:
                    compact_cached = []
                    for group_index, clip_index in enumerate(range(0, len(normalized), group_size)):
                        row = copy.deepcopy(normalized[clip_index])
                        row["clip_index"] = group_index
                        compact_cached.append(row)
                    self._music_arc_quality_notes = quality_notes_for(compact_cached)
                    checkpoint = copy.deepcopy(self._planning_resume_checkpoint)
                    checkpoint["music_arc_quality_notes"] = copy.deepcopy(self._music_arc_quality_notes)
                    self._publish_planning_checkpoint(checkpoint)
                return normalized
            except ValueError:
                pass
        # Long timelines receive one overview item per bounded group, then the
        # detailed writer develops that beat over its assigned native clips.
        timeline = [{"clip_index": i, **c} for i, c in enumerate(groups)]
        schema = creative_plan_schema()
        schema["minItems"] = schema["maxItems"] = len(groups)
        event_field = schema["items"]["properties"]["source_event_ids"]
        event_ids = [event["event_id"] for event in contract.get("events", [])]
        if event_ids:
            # Keyed assignments let the grammar enforce exactly one slot for
            # each source action. Repeating an ID across 23 independent rows
            # was a bookkeeping trap even when their visual beats were sound.
            schema["items"]["properties"].pop("source_event_ids")
            schema["items"]["required"].remove("source_event_ids")
            response_schema = {
                "type": "object",
                "properties": {
                    "beats": schema,
                    "source_event_clips": {
                        "type": "object",
                        "properties": {event_id: {"type": "integer", "minimum": 0,
                                                   "maximum": len(groups) - 1}
                                       for event_id in event_ids},
                        "required": event_ids,
                        "additionalProperties": False,
                    },
                },
                "required": ["beats", "source_event_clips"],
                "additionalProperties": False,
            }
        else:
            event_field["maxItems"] = 0
            response_schema = schema
        lyric_context = ""
        if self._use_lyrics_context and lyrics:
            lyric_context = "\nOptional thematic lyric context:\n" + _compact_lyrics_for_context(
                self._format_lyrics_text(lyrics), max_words=900, max_chars=6000,
            )
        prompt = (
            f"User visual concept: {scene_description}\n"
            + lyric_visual_policy(self._use_lyrics_context)
            + "\nPlan the COMPLETE song before writing detailed shots. Output one compact "
            "creative beat for each timeline item. Develop the requested narrative, "
            "performance, dance, atmosphere or abstract idea in its own form; do not "
            "force a story, new characters, costume changes or a new location. Preserve "
            "explicit fixed-camera, single-location, continuous-shot or immediate-action "
            "requests. Choose a deliberate opening, development, contrast and ending "
            "appropriate to THIS concept and the music. Let important moments breathe. "
            "Preserve every explicitly requested part of a mixed concept: vocal performance "
            "does not replace requested story action or its other participants. "
            "Place enabling actions before their consequences, but do not rush every "
            "requested action into the opening. Distribute the main developments across "
            "the song's sections; later clips need meaningful development, not camera "
            "synonyms for the same action. Cutaways and performance inserts may develop "
            "the concept without resetting persistent physical facts. An intentional "
            "reprise needs a specific purpose and new development or treatment. "
            "Only source_event_ids below are immutable obligations: assign each once "
            "in order; other creative beats may be invented within the concept. "
            "A continuing state (e.g. travelling together) does not require replaying "
            "the transition that established it. Visibly complete each assigned "
            "source action in its item, describing the action rather than merely "
            "writing its event ID. Following items begin from its completed result; "
            "do not spread an enabling action across unassigned items. Keep each text field concise.\n"
            "Required source actions: " + json.dumps(contract.get("events", []), ensure_ascii=False)
            + "\nSong timeline: " + json.dumps(timeline, ensure_ascii=False)
            + lyric_context
        )
        combat_context = format_music_combat_context(contract)
        if combat_context:
            prompt += "\n" + combat_context
        if event_ids:
            prompt += (
                "\nOutput an object with beats and source_event_clips. Each beat "
                "omits source_event_ids; source_event_clips maps each immutable "
                "source-event key to exactly one ZERO-BASED timeline item where "
                "its visible action occurs. Keep those clip indices in source order."
            )
        def parse_candidate(response, baseline=None):
            """Parse and validate one response, preserving locked source assignments."""
            rows = response
            assignments = None
            if event_ids:
                if (
                    not isinstance(response, list) or len(response) != 1
                    or not isinstance(response[0], dict)
                    or set(response[0]) != {"beats", "source_event_clips"}
                ):
                    raise ValueError("outline response must contain beats and source_event_clips")
                envelope = response[0]
                assignments = envelope.get("source_event_clips")
                rows = envelope.get("beats")
                if not isinstance(assignments, dict) or set(assignments) != set(event_ids):
                    raise ValueError("source_event_clips must contain exactly the required source-event keys")
                for event_id in event_ids:
                    index = assignments[event_id]
                    if (
                        not isinstance(index, int) or isinstance(index, bool)
                        or index < 0 or index >= len(groups)
                    ):
                        raise ValueError(f"source event {event_id} needs a valid zero-based timeline index")
                if (
                    not isinstance(rows, list) or len(rows) != len(groups)
                    or not all(isinstance(row, dict) for row in rows)
                ):
                    raise ValueError("beats must cover exactly the requested timeline items")
                rows = copy.deepcopy(rows)
                by_index = {}
                for row in rows:
                    index = row.get("clip_index")
                    if not isinstance(index, int) or isinstance(index, bool):
                        raise ValueError("each outline beat needs an integer clip_index")
                    if index in by_index:
                        raise ValueError("outline beat clip_index values must be unique")
                    if "source_event_ids" in row:
                        raise ValueError("outline beats must omit source_event_ids; use source_event_clips")
                    by_index[index] = row
                if set(by_index) != set(range(len(groups))):
                    raise ValueError("beats must cover each requested timeline item exactly once")
                if baseline is None:
                    for row in rows:
                        row["source_event_ids"] = []
                    for event_id in event_ids:
                        by_index[assignments[event_id]]["source_event_ids"].append(event_id)
                else:
                    baseline_by_index = {row["clip_index"]: row for row in baseline}
                    for row in rows:
                        # Source-event ownership is immutable during a creative repair.
                        row["source_event_ids"] = copy.deepcopy(
                            baseline_by_index[row["clip_index"]]["source_event_ids"]
                        )
            outline = normalize_music_arc(rows, groups, contract)
            outline, recovered = ground_music_combat_arc(outline, contract)
            if recovered:
                for note in recovered:
                    note = copy.deepcopy(note)
                    note["clip_index"] *= group_size
                    note["clip_end_index"] = note["clip_index"]
                    if note not in combat_recovery_notes:
                        combat_recovery_notes.append(note)
                # Restoring an action cannot bypass ordinary outline checks.
                outline = normalize_music_arc(outline, groups, contract)
            early = bool(re.search(
                r"\b(?:immediately|right away|from the start)\b",
                scene_description, re.I,
            ))
            issues = music_arc_quality_issues(outline, contract, allow_early_events=early)
            hard_issues = [
                issue for issue in issues
                if not issue.startswith("Repairable planning issue:")
            ]
            if hard_issues:
                raise ValueError("; ".join(hard_issues[:3]))
            return outline

        def expand_and_checkpoint(outline, notes):
            self._music_arc_assignment_notes = music_arc_action_issues(outline, contract)
            self._music_arc_quality_notes = copy.deepcopy(notes[:12])
            if self._music_arc_quality_notes:
                print(f"[MusicVideoPlanner] Whole-song outline accepted with "
                      f"{len(self._music_arc_quality_notes)} creative advisory note(s); "
                      "source-action and timeline checks passed.")
                for note in self._music_arc_quality_notes[:3]:
                    print(f"[MusicVideoPlanner] Advisory: {str(note.get('message') or '')[:360]}")
            expanded = []
            for i, row in enumerate(outline):
                start = i * group_size
                end = min(len(clips), start + group_size)
                for index in range(start, end):
                    item = copy.deepcopy(row)
                    item["clip_index"] = index
                    if index != start:
                        item["source_event_ids"] = []
                    expanded.append(item)
            checkpoint = copy.deepcopy(self._planning_resume_checkpoint)
            checkpoint["music_creative_arc"] = expanded
            checkpoint["music_arc_assignment_notes"] = copy.deepcopy(self._music_arc_assignment_notes)
            checkpoint["music_arc_quality_notes"] = copy.deepcopy(self._music_arc_quality_notes)
            checkpoint["music_narrative_contract"] = copy.deepcopy(contract)
            checkpoint["music_narrative_allocation"] = [
                {"event_id": event, "clip_index": row["clip_index"]}
                for row in expanded for event in row["source_event_ids"]
            ]
            self._publish_planning_checkpoint(checkpoint)
            return expanded

        def merged_repair(baseline, candidate, findings):
            """Use only improving edits to flagged later rows; keep the best valid arc."""
            merged = copy.deepcopy(baseline)
            baseline_findings = music_arc_repetition_findings(baseline)
            baseline_pairs = {
                (finding["clip_index"], finding["prior_clip_index"])
                for finding in baseline_findings
            }
            implicated = sorted({finding["clip_index"] for finding in findings})
            for index in implicated:
                candidate_by_index = {row["clip_index"]: row for row in candidate}
                current_by_index = {row["clip_index"]: row for row in merged}
                replacement = copy.deepcopy(candidate_by_index[index])
                replacement["source_event_ids"] = copy.deepcopy(
                    current_by_index[index]["source_event_ids"]
                )
                # A repair may improve creative fields but cannot invent or
                # remove reprise intent to evade the conservative comparison.
                replacement["intentional_reprise"] = current_by_index[index]["intentional_reprise"]
                replacement["reprise_reason"] = current_by_index[index]["reprise_reason"]
                current_by_index[index] = replacement
                trial = [current_by_index[i] for i in sorted(current_by_index)]
                trial_pairs = {
                    (finding["clip_index"], finding["prior_clip_index"])
                    for finding in music_arc_repetition_findings(trial)
                }
                if len(trial_pairs) < len(baseline_pairs):
                    merged = trial
                    baseline_pairs = trial_pairs
            # Validate the combined outline, including all immutable event slots.
            return normalize_music_arc(merged, groups, contract)

        issue = ""
        baseline = None
        baseline_findings = []
        best_outline = None
        for attempt in range(2):
            self._raise_if_planning_cancelled()
            self._emit_planning_progress(message="Planning progression across the whole song...",
                                         current=0, total=len(clips), stage="music_creative_arc")
            repair_context = ""
            if baseline is not None and baseline_findings:
                implicated = sorted({finding["clip_index"] for finding in baseline_findings})
                pair_details = [
                    {
                        "later_clip_index": finding["clip_index"],
                        "prior_clip_index": finding["prior_clip_index"],
                        "evidence": finding["evidence"],
                    }
                    for finding in baseline_findings
                ]
                repair_context = (
                    "\nOne focused creative repair is allowed. The valid outline below is the baseline. "
                    "Edit only the implicated later rows at zero-based indices "
                    + json.dumps(implicated)
                    + ". Preserve every unflagged row, cast, pacing, and each immutable source-event assignment. "
                    "Do not move source events, invent intentional_reprise labels, or fabricate reasons. "
                    "Develop only those later beats into distinct visible progression. Exact flagged pairs and evidence: "
                    + json.dumps(pair_details, ensure_ascii=False)
                    + "\nOriginal valid outline to edit in place:\n"
                    + json.dumps(baseline, ensure_ascii=False)
                )
            elif issue:
                repair_context = (
                    "\nRepair this outline's structural or source-assignment issue: " + issue
                    + "\nPreserve valid beats, cast, and pacing. Keep immutable source actions exactly once and in order."
                )
                if best_outline:
                    repair_context += "\nBest previously accepted outline:\n" + json.dumps(
                        best_outline, ensure_ascii=False,
                    )
            response = self._call_llm_json(
                user_prompt=prompt + repair_context,
                system_prompt="Direct a complete music video. Output only the requested JSON structure.",
                max_tokens=max(2048, 160 * len(groups) + 1024),
                temperature=0.55, thinking_budget=0,
                json_schema=response_schema,
            )
            try:
                candidate = parse_candidate(response, baseline=baseline)
                if baseline is None:
                    baseline = candidate
                    baseline_findings = music_arc_repetition_findings(baseline)
                    if not baseline_findings:
                        return expand_and_checkpoint(baseline, quality_notes_for(baseline))
                    best_outline = baseline
                else:
                    try:
                        merged = merged_repair(baseline, candidate, baseline_findings)
                    except ValueError:
                        merged = None
                    if merged is not None:
                        merged_findings = music_arc_repetition_findings(merged)
                        if len(merged_findings) < len(baseline_findings):
                            best_outline = merged
                    # The initial outline is already structurally valid. A failed,
                    # malformed, or still-repetitive optional edit cannot abort it.
                    chosen = best_outline or baseline
                    return expand_and_checkpoint(chosen, quality_notes_for(chosen))
            except ValueError as exc:
                issue = str(exc)
                if baseline is not None:
                    # Optional repetition repair errors are advisory after a valid baseline.
                    return expand_and_checkpoint(baseline, quality_notes_for(baseline))
        if baseline is not None:
            return expand_and_checkpoint(baseline, quality_notes_for(baseline))
        raise RuntimeError("Music-video whole-song planning needs revision: " + issue)

    # ── Character Building ───────────────────────────────────────────

    def _allocate_narrative_actions(
        self, contract: dict, clips: list[dict], scene_description: str,
    ) -> list[dict]:
        """Allocate source-owned actions before independently writing clips."""
        events = contract.get("events") or []
        if not events:
            return []
        cached = self._planning_resume_checkpoint.get("music_narrative_allocation")
        if isinstance(cached, list) and not validate_music_narrative_allocation(
            contract, cached, len(clips),
        ):
            return copy.deepcopy(cached)
        event_ids = [event["event_id"] for event in events]
        schema = {
            "type": "array", "minItems": len(events), "maxItems": len(events),
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "event_id": {"type": "string", "enum": event_ids},
                    "clip_index": {"type": "integer", "minimum": 0,
                                   "maximum": max(0, len(clips) - 1)},
                },
                "required": ["event_id", "clip_index"],
            },
        }
        prompt = (
            f"Original music-video brief: {scene_description}\n\n"
            "Assign each immutable action below to one global clip, in source "
            "order. clip_index is ZERO BASED. Establish the cast, then show "
            "boarding/climbing before riding or flight. Give the opening room "
            "for the user's concept; spread major developments across the song. Later clips can develop "
            "the resulting state with varied action and camera work. Never "
            "repeat a completed enabling action. Multiple ordered events may "
            "share a clip when necessary. Do not invent or rewrite events.\n"
            + json.dumps(events, ensure_ascii=False)
            + "\nAvailable clips:\n"
            + "\n".join(
                f"{index}: {clip.get('label') or clip.get('section_label') or 'scene'}, "
                f"{_music_clip_duration_seconds(clip):.3f}s"
                for index, clip in enumerate(clips)
            )
        )
        issue = ""
        for attempt in range(2):
            self._emit_planning_progress(
                message="Allocating required music-video story actions...",
                current=0, total=len(clips), stage="music_narrative_allocation",
            )
            allocation = self._call_llm_json(
                user_prompt=prompt + (f"\nRepair this allocation error: {issue}" if issue else ""),
                system_prompt="Allocate source actions to clips. Output only the requested JSON array.",
                max_tokens=max(1024, len(events) * 80), thinking_budget=0,
                json_schema=schema,
            )
            issues = validate_music_narrative_allocation(contract, allocation, len(clips))
            if not issues:
                checkpoint = copy.deepcopy(self._planning_resume_checkpoint)
                checkpoint["music_narrative_contract"] = copy.deepcopy(contract)
                checkpoint["music_narrative_allocation"] = copy.deepcopy(allocation)
                self._planning_resume_checkpoint = checkpoint
                self._publish_planning_checkpoint(checkpoint)
                return copy.deepcopy(allocation)
            issue = "; ".join(issues[:3])
        raise RuntimeError(f"Music-video story-action allocation failed: {issue}")

    def _build_characters(
        self,
        characters: Optional[list[dict]],
        speaker_mappings: Optional[dict],
        lyrics: Optional[list[dict]],
        performer_map: dict[str, str],
    ) -> list[CharacterProfile]:
        """Build character profiles from available sources."""
        profiles = []

        # From explicit characters
        if characters:
            for i, c in enumerate(characters):
                profiles.append(CharacterProfile(
                    id=f"char_{i}",
                    display_name=c.get("name", ""),
                    physical_description=c.get("description", "person"),
                ))

        # From speaker mappings (if no explicit characters)
        if not profiles and speaker_mappings:
            for sid, info in speaker_mappings.items():
                name = info.get("name", sid)
                role = info.get("role", "")
                profiles.append(CharacterProfile(
                    id=sid,
                    display_name=name,
                    physical_description=f"the {name}" if name else "a performer",
                    voice_description=role,
                ))

        # From performer map
        if not profiles and performer_map:
            seen = set()
            for section, performer in performer_map.items():
                if performer not in seen:
                    seen.add(performer)
                    profiles.append(CharacterProfile(
                        id=f"perf_{len(profiles)}",
                        display_name=None,
                        physical_description=performer,
                    ))

        return profiles

    def _build_speaker_names(
        self,
        speaker_mappings: Optional[dict],
        lyrics: Optional[list[dict]],
    ) -> dict[str, str]:
        """Map speaker_id → display name."""
        names: dict[str, str] = {}
        if speaker_mappings:
            for sid, info in speaker_mappings.items():
                names[sid] = info.get("name", sid)
        return names

    def _format_lyrics_text(self, lyrics: Optional[list[dict]]) -> str:
        """Format lyrics list into plain text."""
        if not lyrics:
            return ""
        return "\n".join(line.get("text", "") for line in lyrics if line.get("text", "").strip())

    # ── Clip Context Building ────────────────────────────────────────

    def _build_clip_contexts(
        self,
        clips: list[dict],
        lyrics: Optional[list[dict]],
        performer_map: dict[str, str],
        speaker_names: dict[str, str],
        speaker_mappings: Optional[dict],
        *,
        source_audio_drives_vocals: bool = False,
        vocal_activity: Optional[list[str]] = None,
        use_lyrics_context: bool = False,
    ) -> list[str]:
        """Build text descriptions for each clip (context for LLM)."""
        if source_audio_drives_vocals and vocal_activity is None:
            vocal_activity = classify_vocal_intervals(clips, lyrics, None)
        contexts = []
        for i, clip in enumerate(clips):
            section = (clip.get("label") or clip.get("section_label") or "verse").lower()
            beat_count = clip.get("beat_count", 8)
            start_sec = clip.get("start", 0)
            end_sec = clip.get("end", start_sec + 5)
            local_duration = _music_clip_duration_seconds(clip)

            # Gather overlapping lyrics
            lyrics_snippet = ""
            if lyrics:
                overlapping = [
                    l.get("text", "")
                    for l in lyrics
                    if l.get("start", 0) < end_sec and l.get("end", 0) > start_sec
                ]
                if overlapping:
                    lyrics_snippet = _compact_lyrics_for_context(
                        " ".join(overlapping)
                    )

            # Identify dominant speaker
            performer_hint = ""
            if section in performer_map:
                performer_hint = f" Performer: {performer_map[section]}."
            elif clip.get("dominant_speaker") and speaker_names.get(clip["dominant_speaker"]):
                name = speaker_names[clip["dominant_speaker"]]
                role = ""
                if speaker_mappings and clip["dominant_speaker"] in speaker_mappings:
                    role = speaker_mappings[clip["dominant_speaker"]].get("role", "")
                performer_hint = f" Performer: the {name}"
                if role:
                    performer_hint += f" ({role})"
                performer_hint += "."

            # Vocal info
            if source_audio_drives_vocals:
                activity = (
                    vocal_activity[i]
                    if vocal_activity and i < len(vocal_activity)
                    else "unknown"
                )
                if activity == "active":
                    vocal_info = (
                        "mapped source audio drives this interval and vocal "
                        "activity is present; synchronize the shot's existing "
                        "subjects and actions to that audio; only a person "
                        "explicitly assigned as a visible vocalist lip-syncs, "
                        "and only to their own audible part; do not add a person "
                        "or transcribe lyrics for the soundtrack"
                    )
                elif activity == "silent":
                    vocal_info = (
                        "mapped source audio drives this interval with no "
                        "detected vocal activity; do not depict singing or "
                        "lip-sync; preserve the shot's assigned actions and "
                        "explicit non-song expressions"
                    )
                else:
                    vocal_info = (
                        "mapped source audio drives this interval, but vocal "
                        "activity is unknown; do not invent lyrics or visible "
                        "singing, and do not add anyone to represent the "
                        "soundtrack; lip movement follows an actually audible "
                        "voice, not musical rhythm"
                    )
            elif use_lyrics_context:
                vocal_info = (
                    f'lyrics excerpt: "{lyrics_snippet}"'
                    if lyrics_snippet else "instrumental"
                )
            else:
                # Retain transcript-derived vocal/speaker timing without
                # exposing its words to creative planning on legacy models.
                vocal_info = (
                    "source recording has vocals in this interval; preserve "
                    "the assigned performer and timing without adding words"
                    if lyrics_snippet else "no transcribed vocals in this interval"
                )

            if source_audio_drives_vocals and use_lyrics_context and lyrics_snippet:
                vocal_info += f' Optional lyric imagery context: "{lyrics_snippet}"'

            ctx = f"Clip {i + 1}: {section}, {beat_count} beats, {vocal_info}.{performer_hint}"
            source_event_ids = clip.get("source_event_ids")
            if isinstance(source_event_ids, list):
                usable_event_ids = [
                    str(value).strip()[:80]
                    for value in source_event_ids[:16]
                    if isinstance(value, str) and value.strip()
                ]
                if usable_event_ids:
                    ctx += (
                        " Source event IDs for this clip: "
                        + ", ".join(usable_event_ids)
                        + ". Preserve these exact IDs in source_event_ids; "
                        "do not invent IDs."
                    )
            if local_duration is not None:
                end_timecode = (
                    f"{int(local_duration // 60):02d}:"
                    f"{local_duration % 60:06.3f}"
                )
                ctx += (
                    f" Local visual duration: {local_duration:.3f}s "
                    f"(clip-local timeline 00:00.000 through {end_timecode}); "
                    "visual cut timecodes in video_prompt reset to 00:00.000 "
                    "for this clip and must stay within this duration, using "
                    "exactly one decimal separator (for example 00:06.210). "
                    "Source-song lyric and cue offsets remain audio timing, "
                    "not visual cut timecodes."
                )
            else:
                ctx += (
                    " Local visual timecodes, if used, start at 00:00.000; "
                    "do not substitute source-song offsets for shot-local time."
                )
            from services.director.music_cues import format_music_cues
            cue_context = format_music_cues(clip)
            if cue_context:
                ctx += f" Source-song music timing: {cue_context}"
            contexts.append(ctx)

        return contexts

    @staticmethod
    def _music_video_row_issue(
        index: int,
        row: dict,
        previous_rows: list[dict],
        *,
        scene_description: str,
        include_image_fields: bool,
        duration_seconds: Optional[float] = None,
        validate_visual_timing: bool = False,
        require_stable_cast_identity: bool = False,
        require_source_physical_state: bool = False,
        ignore_physical_state_ledger: bool = False,
        require_native_single_clip: bool = False,
        source_events_for_row: Any = None,
        progression_context: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return why a music-video row is incomplete or not shot-specific."""

        if ignore_physical_state_ledger:
            _strip_h3_physical_state_fields(row)
            for previous in previous_rows:
                _strip_h3_physical_state_fields(previous)

        if validate_visual_timing:
            for field in ("video_prompt", "ending_beat"):
                value = row.get(field)
                if isinstance(value, str):
                    row[field] = normalize_music_video_visual_timecodes(value)
            action_beats = row.get("action_beats")
            if isinstance(action_beats, list):
                row["action_beats"] = [
                    normalize_music_video_visual_timecodes(beat)
                    if isinstance(beat, str) else beat
                    for beat in action_beats
                ]

        required_text = [
            "scene_goal",
            "scene_type",
            "environment",
            "visual_style",
            "lighting",
            "mood",
            "ending_beat",
            "video_prompt",
        ]
        proposal_error = row.get(_H3_PHYSICAL_STATE_PROPOSAL_ERROR_FIELD)
        if isinstance(proposal_error, str) and proposal_error:
            return (
                f"clip {index + 1} compact physical-state proposal could not "
                f"be materialized: {proposal_error}"
            )
        if include_image_fields:
            required_text.extend(("image_source", "image_prompt"))
        for field in required_text:
            if not isinstance(row.get(field), str) or not row[field].strip():
                return f"clip {index + 1} is missing a usable {field}"

        if not isinstance(row.get("subjects_on_screen"), list):
            return f"clip {index + 1} has no subjects_on_screen list"
        if require_stable_cast_identity:
            for subject_index, subject in enumerate(row["subjects_on_screen"]):
                if (
                    not isinstance(subject, dict)
                    or not _has_meaningful_cast_identity(subject)
                ):
                    return (
                        f"clip {index + 1} subject {subject_index + 1} needs a "
                        "stable, meaningful character_id or speaker_name"
                    )
        if not isinstance(row.get("action_beats"), list) or not any(
            isinstance(beat, str) and beat.strip()
            for beat in row.get("action_beats", [])
        ):
            return f"clip {index + 1} has no meaningful action_beats"
        camera = row.get("camera_plan")
        if not isinstance(camera, dict) or not str(
            camera.get("framing") or ""
        ).strip():
            return f"clip {index + 1} has no usable camera framing"
        if not isinstance(row.get("window_prompts"), list):
            return f"clip {index + 1} has no window_prompts list"
        if require_native_single_clip and row["window_prompts"]:
            return (
                f"clip {index + 1} is one native H3 clip; window_prompts must "
                "be empty. Put this clip's complete local timeline in "
                "video_prompt, starting at [Shot 1]."
            )
        if include_image_fields and not isinstance(row.get("visual_changes"), list):
            return f"clip {index + 1} has no visual_changes list"

        def normalized(text: Any) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()

        prompt = normalized(row.get("video_prompt"))
        concept = normalized(scene_description)
        if concept and prompt == concept:
            return (
                f"clip {index + 1} repeats the global scene description instead "
                "of providing a shot-specific video_prompt"
            )
        for previous in previous_rows:
            if prompt and prompt == normalized(previous.get("video_prompt")):
                return (
                    f"clip {index + 1} repeats a completed clip's video_prompt"
                )
        if validate_visual_timing:
            timing_issue = music_video_visual_timing_issue(
                row.get("video_prompt"), duration_seconds
            )
            if timing_issue:
                return f"clip {index + 1} {timing_issue}"

        prefix_rows = [*previous_rows, row]
        if not ignore_physical_state_ledger:
            physical_issues = validate_physical_state_sequence(
                prefix_rows,
                require_state_after_first=True,
            )
            if physical_issues:
                return physical_issues[0]

            physical_prompt_issue = _physical_prompt_conflict_issue(row)
            if physical_prompt_issue:
                return f"clip {index + 1} {physical_prompt_issue}"

            if require_source_physical_state:
                assigned_source_issues = source_physical_state_issues(
                    [row], [source_events_for_row],
                )
                if assigned_source_issues:
                    return f"clip {index + 1} {assigned_source_issues[0]}"

        if (
            not ignore_physical_state_ledger
            and require_source_physical_state
            and not source_events_for_row
            and _source_establishes_physical_relation(scene_description, row)
        ):
            state = normalize_physical_state(row.get("physical_state"))
            required_predicates = {"support", "contact"}
            opening_relations = {
                (fact["subject_id"].casefold(), fact["predicate"])
                for fact in (state or {}).get("opening", [])
                if fact["predicate"] in required_predicates
            }
            closing_relations = {
                (fact["subject_id"].casefold(), fact["predicate"])
                for fact in (state or {}).get("closing", [])
                if fact["predicate"] in required_predicates
            }
            if not opening_relations or not closing_relations:
                return (
                    f"clip {index + 1} keeps an explicitly established physical "
                    "relation in its planned action, so physical_state must name "
                    "a support/contact fact in both opening and closing"
                )

        progression_issue = music_video_progression_issue(
            prefix_rows,
            **(progression_context or {}),
        )
        if progression_issue:
            return progression_issue
        return None

    # ── LLM Planning Call ────────────────────────────────────────────

    def _plan_with_llm(
        self,
        clips: list[dict],
        clip_contexts: list[str],
        scene_description: str,
        bpm: float,
        has_reference: bool,
        reference_image_path: Optional[str],
        char_profiles: list[CharacterProfile],
        performer_map: dict[str, str],
        nsfw: bool = False,
        **kwargs,
    ) -> list[dict]:
        """Call LLM to generate structured shot plans."""
        from ..nsfw_guidance import inject_nsfw_if_enabled

        is_minimax_h3 = str(kwargs.get("video_model") or "").lower().startswith(
            "minimax_h3"
        )
        # H3 rows include the native prompt and full cast/camera direction.
        # Six such rows exceeded the former LTX-sized output allowance and
        # truncated otherwise usable long-song plans in the middle of JSON.
        batch_size = 3 if is_minimax_h3 else self._LONG_FORM_BATCH_SIZE
        if not kwargs.get("_bounded_music_batch"):
            invalid_drafts: dict[int, dict] = {}
            narrative_repair_attempts: dict[int, int] = {}
            forwarded_kwargs = {
                key: value for key, value in kwargs.items()
                if key not in {
                    "_bounded_music_batch",
                    "_music_global_start",
                    "_music_previous_physical_state",
                    "_preserve_short_music_reasoning",
                    "_planning_progress_callback",
                    "_planning_checkpoint_callback",
                    "_planning_cancelled_callback",
                    "_planning_checkpoint",
                }
            }

            def plan_batch(
                batch_number: int,
                start: int,
                batch_clips: list[dict],
                previous: Optional[dict],
                *,
                repair: bool,
            ) -> list[dict]:
                if is_minimax_h3:
                    _strip_h3_physical_state_fields(previous)
                    for rejected_row in invalid_drafts.values():
                        _strip_h3_physical_state_fields(rejected_row)
                end = start + len(batch_clips)
                previous_state = "No prior completed clip; establish the opening cleanly."
                if previous:
                    state_parts = []
                    for label, value in (
                        ("goal", previous.get("scene_goal")),
                        ("setting", previous.get("environment")),
                        ("ending", previous.get("ending_beat")),
                    ):
                        text = str(value or "").strip()
                        if text:
                            state_parts.append(f"{label}: {text}")
                    if not is_minimax_h3:
                        physical_state = normalize_physical_state(previous.get("physical_state"))
                        physical_state_text = format_physical_state_for_prompt(physical_state)
                        if physical_state_text:
                            state_parts.append(physical_state_text)
                    subjects = previous.get("subjects_on_screen")
                    if isinstance(subjects, list):
                        names = [
                            str(item.get("visual_description") or "").strip()
                            for item in subjects
                            if isinstance(item, dict)
                            and item.get("visual_description")
                        ]
                        if names:
                            state_parts.append("visible subjects: " + ", ".join(names))
                    if state_parts:
                        previous_state = "; ".join(state_parts)
                batch_concept = (
                    f"{scene_description}\n\n"
                    "LONG-FORM TIMELINE CONTRACT:\n"
                    f"This is planning batch {batch_number}, covering global "
                    f"clips {start + 1}-{end} of {len(clips)}. Continue the "
                    "same music video; preserve performer identity, wardrobe, "
                    "world, and visual grammar unless the song section motivates "
                    "a visible change. Give every requested clip its own "
                    "section-specific action, environment, and ending.\n"
                    f"Previous completed ending/state: {previous_state}. "
                    "Carry that visible state into the next clip without "
                    "repeating the previous shot."
                )
                narrative_contract = getattr(self, "_music_narrative_contract", {})
                creative_arc = getattr(self, "_music_arc", [])
                if creative_arc:
                    from ..music_arc import filter_music_arc_context, format_music_arc_context
                    # Creative shorthand cannot overrule the story schedule.
                    # A rejected beat must not be fed back as repair direction.
                    repair_indices = (
                        range(start, end)
                        if repair and getattr(self, "_last_music_narrative_issue", "")
                        else ()
                    )
                    arc_context = filter_music_arc_context(
                        creative_arc, narrative_contract, self._music_narrative_allocation,
                        suppressed_indices=repair_indices,
                    )
                    batch_concept += "\n\n" + format_music_arc_context(
                        arc_context, start, end, one_based=True,
                        quality_notes=getattr(self, "_music_arc_quality_notes", []),
                    )
                if narrative_contract.get("events"):
                    batch_concept += "\n\n" + format_music_narrative_context(
                        narrative_contract,
                        self._music_narrative_allocation,
                        start, end,
                    )
                if is_minimax_h3 and previous:
                    batch_concept += (
                        "\nAt this shot boundary, posture, head angle, gaze, "
                        "expression, and gesture are local visual direction and "
                        "may be re-established in the new shot without matching "
                        "prior wording. Preserve concrete source-established "
                        "support, contact, grounding, prop ownership, and location "
                        "in the visible action when they remain relevant; show a "
                        "change through the requested action. Do not invent a "
                        "mount, grip, contact, or starting position. Do not replay "
                        "a completed one-time transition from the previous clip. "
                        "An explicitly requested ongoing performance or conflict "
                        "may continue with new visible action."
                    )
                if repair:
                    batch_concept += (
                        "\n\nTAIL REPAIR: Earlier clips are already complete. "
                        "Return plans only for the listed missing global clips. "
                        "Keep their original clip numbers and musical contexts; "
                        "do not restart the video or repeat the global scene brief."
                    )
                    if getattr(self, "_last_music_narrative_issue", ""):
                        batch_concept += (
                            "\nRequired story-action repair: "
                            + self._last_music_narrative_issue
                            + "\nUse the accepted previous scene and the source-action "
                            "schedule to rebuild this clip's local action, environment "
                            "and subject positions. Revise any rejected draft details "
                            "that depict a pending milestone early. When this clip has "
                            "no assigned milestone, develop a distinct performance, "
                            "reaction or camera beat within the established scene, "
                            "preserving visible support and contact."
                        )
                    if getattr(self, "_last_music_creative_issue", ""):
                        batch_concept += "\nFOCUSED CREATIVE REPAIR:\n" + self._last_music_creative_issue
                    if getattr(self, "_last_music_row_issue", ""):
                        batch_concept += "\nREQUIRED VALIDATION REPAIR:\n" + self._last_music_row_issue
                    rejected = [
                        {"global_clip": index + 1, "draft": invalid_drafts[index]}
                        for index in range(start, end) if index in invalid_drafts
                    ]
                    if rejected:
                        batch_concept += (
                            "\nEDIT THE REJECTED DRAFT IN PLACE:\n"
                            "Fix the stated validation error while retaining the valid "
                            "visible actions, source-event obligations, stable subject IDs "
                            "and unaffected direction. Do not replace a correctly performed "
                            "action with a static pose. Keep physical relations accurate "
                            "in visible prose and preserve the source action and actor. "
                            "Source actions above take precedence over an inconsistent "
                            "draft or outline.\n"
                            + json.dumps(rejected, ensure_ascii=False)
                        )
                return self._plan_with_llm(
                    clips=batch_clips,
                    clip_contexts=clip_contexts[start:end],
                    scene_description=batch_concept,
                    bpm=bpm,
                    has_reference=has_reference,
                    reference_image_path=reference_image_path,
                    char_profiles=char_profiles,
                    performer_map=performer_map,
                    nsfw=nsfw,
                    _bounded_music_batch=True,
                    _music_global_start=start,
                    _preserve_short_music_reasoning=(
                        not repair and len(clips) <= self._LONG_FORM_BATCH_SIZE
                    ),
                    **forwarded_kwargs,
                )

            def call_batch(
                batch_number: int,
                start: int,
                batch_clips: list[dict],
                previous: Optional[dict],
            ) -> list[dict]:
                return plan_batch(
                    batch_number,
                    start,
                    batch_clips,
                    previous,
                    repair=False,
                )

            def repair_batch(
                batch_number: int,
                start: int,
                batch_clips: list[dict],
                previous: Optional[dict],
            ) -> list[dict]:
                if is_minimax_h3 and len(batch_clips) == 1:
                    # A later row in a rejected batch may have no cached draft
                    # until its first full-row repair. Count that writer call
                    # too, so source recovery fits the shared two-attempt cap.
                    narrative_repair_attempts[start] = narrative_repair_attempts.get(start, 0) + 1
                narrative_issue = getattr(self, "_last_music_narrative_issue", "")
                if (
                    is_minimax_h3
                    and len(batch_clips) == 1
                    and start in invalid_drafts
                    and narrative_issue
                    and not getattr(self, "_last_music_creative_issue", "")
                    and not narrative_allocation_issue
                ):
                    # A small writer can echo a conflicting outline when asked
                    # to regenerate every field. Repair the actual visual action
                    # with a compact source-owned request, then run the same full
                    # production row validator on the merged row below.
                    from ..music_narrative_repair import (
                        compile_music_narrative_bridge_row,
                        compile_music_narrative_source_row,
                        repair_music_narrative_visuals,
                    )

                    repaired = repair_music_narrative_visuals(
                        self,
                        index=start,
                        row=invalid_drafts[start],
                        issue=narrative_issue,
                        contract=narrative_contract,
                        allocation=narrative_allocation,
                        duration_seconds=_music_clip_duration_seconds(batch_clips[0]),
                    )
                    if repaired is not None:
                        return repaired
                    if narrative_repair_attempts[start] >= 2:
                        compiled = compile_music_narrative_source_row(
                            index=start,
                            row=invalid_drafts[start],
                            issue=narrative_issue,
                            contract=narrative_contract,
                            allocation=narrative_allocation,
                            previous=previous,
                        )
                        if compiled is None:
                            compiled = compile_music_narrative_bridge_row(
                                index=start,
                                row=invalid_drafts[start],
                                issue=narrative_issue,
                                contract=narrative_contract,
                                allocation=narrative_allocation,
                                previous=previous,
                            )
                        if compiled is not None:
                            print(
                                f"[MusicVideoPlanner] Clip {start + 1} compiled "
                                "from its source actions or established scene after "
                                "two writer repair attempts."
                            )
                            self._emit_planning_progress(
                                message=(f"Clip {start + 1}: compiled from "
                                         "the requested story and established scene"),
                                current=start,
                                total=len(clips),
                                stage="music_source_recovery",
                            )
                            return compiled
                return plan_batch(
                    batch_number,
                    start,
                    batch_clips,
                    previous,
                    repair=True,
                )

            uses_generated_images = bool(
                getattr(self, "_uses_generated_shot_images", True)
            )
            narrative_contract = getattr(self, "_music_narrative_contract", {})
            narrative_allocation = getattr(self, "_music_narrative_allocation", [])
            contract_events = (
                narrative_contract.get("events")
                if isinstance(narrative_contract, dict) else None
            )
            # Event IDs are planner-owned bookkeeping; visible actions remain
            # subject to the narrative row validator below.
            source_event_ids_by_clip: Optional[list[list[str]]] = None
            narrative_allocation_issue = ""
            if is_minimax_h3 and isinstance(contract_events, list):
                if any(
                    not isinstance(event, dict)
                    or not isinstance(event.get("event_id"), str)
                    or not event["event_id"].strip()
                    for event in contract_events
                ):
                    narrative_allocation_issue = (
                        "music narrative contract is incomplete or invalid"
                    )
                else:
                    allocation_issues = validate_music_narrative_allocation(
                        narrative_contract, narrative_allocation, len(clips),
                    )
                    if allocation_issues:
                        narrative_allocation_issue = (
                            "music narrative allocation is invalid: "
                            + allocation_issues[0]
                        )
                    else:
                        assignments = {
                            item["event_id"]: item["clip_index"]
                            for item in narrative_allocation
                        }
                        source_event_ids_by_clip = [
                            [
                                event["event_id"] for event in contract_events
                                if assignments[event["event_id"]] == clip_index
                            ]
                            for clip_index in range(len(clips))
                        ]

            def validate_music_row(index, row, previous_rows):
                if is_minimax_h3:
                    _strip_h3_physical_state_fields(row)
                    for previous in previous_rows:
                        _strip_h3_physical_state_fields(previous)
                if narrative_allocation_issue:
                    issue = narrative_allocation_issue
                    self._last_music_narrative_issue = issue
                    self._last_music_row_issue = issue
                    invalid_drafts[index] = copy.deepcopy(row)
                    return issue
                if (
                    source_event_ids_by_clip is not None
                    and isinstance(row, dict)
                    and 0 <= index < len(source_event_ids_by_clip)
                ):
                    row["source_event_ids"] = list(source_event_ids_by_clip[index])

                if contract_events:
                    issue = music_narrative_row_issue(
                        row,
                        contract=narrative_contract,
                        allocation=narrative_allocation,
                        index=index,
                    )
                    if issue:
                        self._last_music_narrative_issue = issue
                        invalid_drafts[index] = copy.deepcopy(row)
                        self._last_music_row_issue = issue
                        return issue
                self._last_music_narrative_issue = ""
                assigned_ids = {
                    item["event_id"] for item in narrative_allocation
                    if item["clip_index"] == index
                }
                assigned_events = [
                    event for event in narrative_contract.get("events", [])
                    if event["event_id"] in assigned_ids
                ]
                issue = self._music_video_row_issue(
                    index, row, previous_rows,
                    scene_description=scene_description,
                    include_image_fields=uses_generated_images,
                    duration_seconds=(
                        _music_clip_duration_seconds(clips[index])
                        if 0 <= index < len(clips) else None
                    ),
                    validate_visual_timing=is_minimax_h3,
                    require_stable_cast_identity=is_minimax_h3,
                    require_source_physical_state=False,
                    ignore_physical_state_ledger=is_minimax_h3,
                    require_native_single_clip=is_minimax_h3,
                    source_events_for_row=(
                        _assigned_source_events_for_clip(clips[index]) or assigned_events
                        if 0 <= index < len(clips) else None
                    ),
                    progression_context={
                        "sections": [
                            str(clips[i].get("label") or clips[i].get("section_label") or "")
                            for i in range(index + 1)
                        ],
                        "source_event_ids_by_index": [
                            clips[i].get("source_event_ids") for i in range(index + 1)
                        ],
                    },
                )
                self._last_music_row_issue = issue or ""
                if issue:
                    invalid_drafts[index] = copy.deepcopy(row)
                else:
                    invalid_drafts.pop(index, None)
                return issue
            rows = self._run_checkpointed_json_batches(
                items=clips,
                batch_size=batch_size,
                checkpoint_key="music_video_batches",
                stage="music_video_batch",
                progress_label="music-video",
                call_batch=call_batch,
                repair_batch=repair_batch,
                row_validator=validate_music_row,
                repair_batch_size=max(1, batch_size // 2),
                repair_attempts=2,
            )
            if getattr(self, "_music_arc", []):
                rows = self._review_music_plan(
                    rows, clips, clip_contexts, scene_description,
                    row_validator=validate_music_row,
                    repair_row=lambda index, feedback, current: plan_batch(
                        index + 1, index, clips[index:index + 1],
                        current[index - 1] if index else None, repair=True,
                    ),
                    batch_size=batch_size,
                    ignore_physical_state_ledger=is_minimax_h3,
                )
            return rows

        num_character_refs = len(kwargs.get("character_ref_paths", []) or [])
        num_location_refs = len(kwargs.get("location_ref_paths", []) or [])
        has_asset_references = bool(
            has_reference or num_character_refs or num_location_refs
        )
        preserve_names = bool(
            getattr(self, "_preserve_video_character_names", False)
        )
        uses_generated_images = bool(
            getattr(self, "_uses_generated_shot_images", True)
        )
        char_rules = build_character_rules_block(
            has_reference or bool(num_character_refs),
            char_profiles if char_profiles else None,
            preserve_names=preserve_names,
        )
        camera_block = build_camera_style_block()
        # video_guide now merged into ltx2_music_video_rules.md — no separate load needed
        if is_minimax_h3:
            physical_continuity_rules = (
                "H3 VISIBLE PHYSICAL CONTINUITY:\n"
                "- Describe concrete support and contact in the visible action: "
                "who is standing on, riding, holding, touching, or carrying what; "
                "state whether a subject is grounded or airborne when it matters. "
                "Preserve source-established prop ownership and location when "
                "they remain relevant, and show a change through its visible action.\n"
                "- Head angle, gaze, expression, gesture, and body pose are local "
                "shot direction, not exact-text state that must carry across cuts. "
                "A new shot may establish a different pose without a transition "
                "from a prior pose.\n"
                "- Do not invent a mount, grip, support relation, contact, or "
                "starting position the Scene Concept has not established. Keep "
                "actor, target, and action order faithful to the requested story."
            )
        else:
            physical_continuity_rules = (
                "PHYSICAL CONTINUITY:\n"
                "- Every clip must include physical_state. Use empty opening, "
                "transitions, and closing arrays only when no established "
                "physical relation is relevant to the action. If the Scene "
                "Concept or clip context establishes riding, carrying, holding, "
                "touching, standing on, or another support/contact relation, "
                "opening and closing facts must state that relation; never use "
                "an empty ledger to omit it. Use the same stable subject_id "
                "values throughout.\n"
                "- A transition must name the visible action that changes the "
                "fact. Keep every transition action in action_beats and make the "
                "support/contact facts visible in video_prompt.\n"
                "- Carry active closing facts into the next continuous clip's "
                "opening. Mark physical_state.continuity as independent or "
                "time_jump only when that clip is actually planned as a new "
                "place/time or a discontinuous story beat, and pair it with "
                "continuity_strategy independent. A camera or framing change "
                "alone keeps physical continuity intact.\n"
                "- Do not invent a mount, grip, support relation, or prop handoff "
                "that the Scene Concept has not established.\n"
                "- Treat physical_state as the blocking authority. Make "
                "video_prompt agree with the opening facts and transitions, "
                "and end on the closing facts; do not leave conflicting earlier "
                "blocking prose in the prompt.\n"
                "- When clip context supplies source event IDs, preserve those "
                "exact IDs in source_event_ids. Do not make up source event IDs."
            )

        image_prompt_rules = ""
        if uses_generated_images:
            from ..image_prompt_rules import get_image_prompt_rules
            image_prompt_rules = get_image_prompt_rules(
                has_reference,
                num_character_refs=num_character_refs,
                num_location_refs=num_location_refs,
                character_ref_labels=kwargs.get("character_ref_labels"),
                location_ref_labels=kwargs.get("location_ref_labels"),
                seamless=kwargs.get("seamless", True),
                image_model=kwargs.get("image_model", ""),
            )

        from ..guide_loader import load_guide
        video_model = str(kwargs.get("video_model") or "")
        music_video_rules = load_guide(
            "minimax_h3_shot_breakdown.md"
            if video_model.lower().startswith("minimax_h3")
            else "ltx2_music_video_rules.md"
        )
        h3_direct_rules = (
            "H3 DIRECT-REFERENCE MUSIC VIDEO:\n"
            "- No generated start frame will be supplied. Make each video_prompt "
            "self-contained with setting, composition, named identities plus "
            "visible traits, wardrobe, performance, camera, lighting, ambience, "
            "effects, and music.\n"
            "- The per-shot source-audio slice is mapped as driving audio. "
            "Synchronize this shot's assigned actions to it. Only a person "
            "explicitly assigned as a visible vocalist lip-syncs to their own "
            "audible part. Keep the source soundtrack as the audio source; do "
            "not add a person to represent it or transcribe lyrics.\n"
            "- Character/location references are soft guidance, not fixed first "
            "frames. Describe the finished target shot.\n"
            "- Do not create image_prompt, image_source, visual_changes, or "
            "keyframe_prompts. Those fields are intentionally absent from the "
            "video-only output schema."
            if not uses_generated_images else ""
        )
        reference_aesthetic_rules = (
            """VISUAL AESTHETIC — the reference photo defines the visual style for the entire music video.
Match its aesthetic (color grading, film texture, era, tone) in every image_prompt unless the
scene concept explicitly calls for a style change. End each image_prompt with
"Use lighting and color temp from reference image." to preserve the look."""
            if uses_generated_images and has_reference else ""
        )
        image_output_fields = (
            '''    "image_source": "original or previous",
    "image_prompt": "FIRST FRAME BEFORE action — initial state, static pose, environment. No motion verbs.",
    "visual_changes": ["what transforms during the clip — e.g. 'performer jumps off stage', 'lights shift to red'"],
'''
            if uses_generated_images else ""
        )
        keyframe_output_field = (
            '    "keyframe_prompts": [],\n'
            if uses_generated_images else ""
        )
        physical_state_output_field = (
            '' if is_minimax_h3 else
            '    "physical_state": {"opening": [], "transitions": [], "closing": []},\n'
        )
        image_workflow_notes = (
            """- image_source: "original" = user's reference photo (default). "previous" = previous scene's output for same-location continuity.
- FIELD ORDER: Write image_prompt FIRST (starting state), then visual_changes, then video_prompt.
- visual_changes: If the performer jumps off stage, image_prompt shows them still ON stage.
- keyframe_prompts: DEFAULT IS EMPTY. Add one only for a specific visual state the video model cannot infer from the start image and prompt; never for ordinary movement, camera, expression, lighting, or energy changes.
"""
            if uses_generated_images else ""
        )
        if uses_generated_images and has_reference:
            scene_anchoring_rules = """SCENE-ANCHORING (avoid off-topic content):
The user's main reference is visual ground truth. Every image_prompt and video_prompt must match its identity, setting, and aesthetic plus the Scene Concept. Do not invent unrelated worlds."""
        elif has_asset_references:
            scene_anchoring_rules = """SCENE-ANCHORING (avoid off-topic content):
Character references define identity and location references define the setting. Follow their labels and the Scene Concept in every self-contained video prompt; do not invent conflicting identities or settings."""
        else:
            scene_anchoring_rules = """SCENE-ANCHORING (avoid off-topic content):
No visual reference was provided. Follow the Scene Concept, which may be narrative, dance, environment, or performance-focused. Do not add people or objects solely because the soundtrack contains vocals. Keep the identity and role of any requested or explicitly assigned person consistent across clips."""

        subject_identity_field = (
            '"character_id": "lead_performer", "speaker_name": "", '
            if is_minimax_h3 else ""
        )

        system_prompt = f"""You are a music video director. Plan each clip AND write its prompts. Output ONLY the JSON array.

{f"You are given a REFERENCE PHOTO. Use it to identify appearance, clothing, and setting." if has_reference else ""}

{char_rules}

{camera_block}

{reference_aesthetic_rules}

MUSIC VIDEO RULES:
- Chorus = high energy, bold framing. Verse = intimate, character focus.
- Instrumental = environment, textures. Bridge = contrasting, unexpected.
- Vary visuals across clips. Performer must be visible when assigned.
- Keep anatomy and proportions stable. Convey energy through natural expression,
  gesture, movement, and camera work; do not invent expanding, pulsing, or bulging
  body parts. Preserve any transformation the Scene Concept explicitly requests.

{physical_continuity_rules}

{music_video_rules}

{MUSIC_PERFORMANCE_RULES}

{h3_direct_rules}

{image_prompt_rules}

For MiniMax H3, preserve supplied cast names exactly and emit both identity keys. For unnamed cast, use a stable semantic character_id such as lead_performer and an empty speaker_name; never use a placeholder name such as Unnamed or N/A. Subjects can be objects, animals, materials or abstract forms; do not add a person for an instrumental or abstract concept. Keep concrete support, contact, grounding, prop ownership, and location accurate in visible prose. Posture, gaze, head angle, expression, and gesture are local shot direction and may be re-established at a cut; do not invent a starting physical relation.

OUTPUT — respond with ONLY a JSON array:
[
  {{
    "scene_goal": "What this clip achieves",
    "scene_type": "performance|narrative|atmospheric",
    "continuity_strategy": "independent|continuous|extend_previous",
    "subjects_on_screen": [{{{subject_identity_field}"visual_description": "a visible person", "position_or_relation": "center frame", "performance_role": "non_vocal"}}],
    "environment": "Setting details",
    "visual_style": "Style",
    "lighting": "Lighting",
    "mood": "Tone",
    "action_beats": ["Action 1", "Action 2"],
{physical_state_output_field}    "camera_plan": {{"framing": "medium shot", "movement": "slow dolly in", "movement_intensity": "subtle"}},
    "ending_beat": "Final image",
{image_output_fields}    "video_prompt": "Energetic prompt describing the visible performance, action, sound, and camera.",
{keyframe_output_field}    "window_prompts": []
  }}
]

Notes:
{image_workflow_notes}
- window_prompts: {"always empty ([]). Each listed H3 clip is already one native generation. Put all local action and any consecutive visual cuts in video_prompt; never add rolling windows or restart the clip's clock." if is_minimax_h3 else "empty ([]) unless the scene needs >26s continuous video."}

{scene_anchoring_rules}

KEEP MUSIC-VIDEO PROMPTS SIMPLE:
For each scene, the music drives the pacing and energy. You only need to identify:
  - WHO is in frame ({"preserve user-supplied proper names and pair them with useful visible traits" if preserve_names else "the performer, by descriptor — never by name"})
  - CAMERA MOVEMENT (push-in, pull-back, orbit, handheld, low angle, etc.)
  - ATMOSPHERIC ELEMENTS (smoke, pyro, crowd cheering, lighting flashes, etc.)
  - The performer's BODY MOVEMENT in broad strokes (head bob, arms raised,
    walking forward, etc.) — but don't over-specify; the model interpolates.
{"Use the H3 Context-IR fields above. Be concise but complete; do not enforce the legacy 15-40 word LTX limit." if preserve_names else "Keep video_prompt 15-40 words. Anything longer is over-described for music video."}

{"Most scenes should use a single video_prompt with empty keyframe_prompts." if uses_generated_images else "Every scene should use video_prompt/window_prompts only; omit all still-image fields."}
Output exactly {len(clips)} objects. Go:"""

        # Inject model-specific prompt polish guide if provided
        polish_block = kwargs.get("polish_block", "")
        if polish_block:
            system_prompt = f"{system_prompt}\n\n{polish_block}"

        # Inject content guidance (NSFW or safety guardrails)
        system_prompt = inject_nsfw_if_enabled(
            system_prompt,
            nsfw,
            "both" if uses_generated_images else "video",
        )

        system_prompt += "\n\n" + lyric_visual_policy(
            bool(getattr(self, "_use_lyrics_context", False))
        )
        user_prompt = f"""Scene Concept: {scene_description}
Song tempo: {bpm:.0f} BPM

Clips:
{chr(10).join(clip_contexts)}

Write {len(clips)} structured shot plans. Go:"""

        narrative_contract = getattr(self, "_music_narrative_contract", {})
        if is_minimax_h3 and narrative_contract.get("events"):
            global_start = int(kwargs.get("_music_global_start") or 0)
            user_prompt += (
                "\n\nFINAL REQUIRED STORY ACTIONS: These override creative "
                "paraphrasing. In video_prompt, write each assigned source "
                "action clause unchanged, with its named actor, as an actual "
                "visible action. Expand the physical movement around that "
                "clause. Do not replace it with a scene_goal, summary, or "
                "action_beats entry. "
                + (
                    "For boarding/climbing, establish the actor beside the "
                    "target, show continuous contact during the climb, and "
                    "settle securely on it before riding or flight. "
                    if any(event.get("kind") == "boarding" for event in narrative_contract["events"])
                    else ""
                )
                + "\n"
                + format_music_narrative_context(
                    narrative_contract, self._music_narrative_allocation,
                    global_start, global_start + len(clips),
                )
            )

        # Send ALL reference images to the LLM (main + character + location refs)
        image_paths = []
        if has_reference and reference_image_path:
            image_paths.append(reference_image_path)
        for cp in (kwargs.get("character_ref_paths") or []):
            if cp and os.path.isfile(cp):
                image_paths.append(cp)
        for lp in (kwargs.get("location_ref_paths") or []):
            if lp and os.path.isfile(lp):
                image_paths.append(lp)
        if not image_paths:
            image_paths = None
        per_clip_tokens = (
            2800 if uses_generated_images else 2300
        ) if is_minimax_h3 else (700 if uses_generated_images else 900)
        max_tokens = max(4096, len(clips) * per_clip_tokens + 1024)
        shot_dicts = self._call_llm_json(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            # A long timeline is already divided into a bounded, explicit clip
            # batch. Keep that structured transformation grammar-constrained
            # and spend reasoning only on the historical non-H3 short-form path.
            thinking_budget=(
                4096 if kwargs.get("_preserve_short_music_reasoning") and not is_minimax_h3 else 0
            ),
            image_paths=image_paths,
            json_schema=_music_shot_schema(
                len(clips),
                include_image_fields=uses_generated_images,
                require_cast_identity=is_minimax_h3,
                omit_physical_state=is_minimax_h3,
                require_continuity_strategy=is_minimax_h3,
                require_native_single_clip=is_minimax_h3,
                narrative_event_ids=[
                    event["event_id"] for event in getattr(
                        self, "_music_narrative_contract", {},
                    ).get("events", [])
                ],
            ),
        )
        if is_minimax_h3:
            for row in shot_dicts:
                _strip_h3_physical_state_fields(row)
        if not uses_generated_images:
            _discard_unused_image_fields(shot_dicts)
        return shot_dicts

    def _review_music_plan(self, rows, clips, contexts, source, *, row_validator,
                           repair_row, batch_size,
                           ignore_physical_state_ledger=False):
        """Review actual visual bodies and repair only evidenced weak clips."""
        from ..music_arc import format_music_arc_context
        from ..music_review import (
            creative_review_schema, grounded_review_findings,
            music_review_fingerprint, music_visual_body,
            music_visual_reset_issue, repair_music_rows,
            repeated_music_visuals, justified_music_reprise,
        )
        arc = self._music_arc
        if ignore_physical_state_ledger:
            for row in rows:
                _strip_h3_physical_state_fields(row)
        cached = self._planning_resume_checkpoint.get("music_creative_review", {})
        if cached.get("fingerprint") == music_review_fingerprint(rows, arc):
            return rows
        repaired = set()
        review_notes = []

        def save(current, *, complete=False):
            if ignore_physical_state_ledger:
                for row in current:
                    _strip_h3_physical_state_fields(row)
            checkpoint = copy.deepcopy(self._planning_resume_checkpoint)
            batches = checkpoint.setdefault("music_video_batches", {})
            for start in range(0, len(current), batch_size):
                part = current[start:start + batch_size]
                batches[str(start)] = {"count": len(part), "expected_count": len(part),
                                       "rows": copy.deepcopy(part)}
            checkpoint["music_creative_review"] = {
                "fingerprint": music_review_fingerprint(current, arc) if complete else "",
                "repaired_clip_indices": sorted(repaired),
                "preserved_clip_indices": [i for i in range(len(current)) if i not in repaired],
                "review_notes": copy.deepcopy(review_notes),
            }
            self._publish_planning_checkpoint(checkpoint)

        def replace(index, feedback, current):
            self._raise_if_planning_cancelled()
            if ignore_physical_state_ledger:
                for row in current:
                    _strip_h3_physical_state_fields(row)
                continuity_instruction = (
                    "Keep source actions, stable subject IDs, and concrete support, "
                    "contact, grounding, prop ownership, and location accurate in "
                    "visible prose. Head angle, gaze, expression, gesture, and pose "
                    "are local direction and may change at a cut.\n"
                )
            else:
                continuity_instruction = (
                    "Preserve its authoritative opening and closing physical facts; "
                    "change only its visible treatment/action within those boundaries.\n"
                    "Required closing facts: " + json.dumps(
                        (current[index].get("physical_state") or {}).get("closing", []),
                        ensure_ascii=False,
                    ) + "\n"
                )
            self._last_music_narrative_issue = ""
            self._last_music_creative_issue = (
                feedback + "\nKeep the accepted neighboring clips unchanged. "
                "Keep this clip's source_event_ids and stable subject IDs. "
                + continuity_instruction
                + "Current clip to revise in place:\n"
                + json.dumps(current[index], ensure_ascii=False)
            )
            try:
                replacement = repair_row(index, feedback, current)
            finally:
                self._last_music_creative_issue = ""
            return replacement

        def keep_advisory(index, issue):
            if index in hard_findings:
                raise RuntimeError(f"Music-video clip {index + 1} creative repair needs revision: {issue}")
            review_notes.append({"clip_index": index, "severity": "advisory",
                                 "inconclusive": issue})
            print(f"[MusicVideoPlanner] Optional variety repair for clip {index + 1} "
                  "was inconclusive; keeping its validated original.")

        for round_index in range(2):
            findings = repeated_music_visuals(rows, arc)
            hard_findings = set(findings)
            for start in range(0, len(rows), 22):
                self._raise_if_planning_cancelled()
                end = min(len(rows), start + 24)
                self._emit_planning_progress(
                    message="Reviewing musical progression and visible shot variety...",
                    current=start, total=len(rows), stage="music_creative_review",
                )
                cards = [{"clip_index": i, "assigned_beat": arc[i]["visual_beat"],
                          "intentional_reprise": arc[i]["intentional_reprise"],
                          "reprise_reason": arc[i]["reprise_reason"],
                          "visual_body": music_visual_body(rows[i].get("video_prompt"))[:2600]}
                         for i in range(start, end)]
                request = (
                    f"User visual concept: {source}\n"
                    + format_music_arc_context(arc, start, end)
                    + "\nReview the actual visual bodies below, not identity headers, "
                    "summary text or repeated soundscape rules. Flag only material "
                    "failures: a whole-brief reset; replaying the same enabling action; "
                    "an unplanned repetitive visible sequence with only camera-word "
                    "changes; or contradicting the assigned creative beat. A persistent "
                    "physical state is not repetition by itself. Permit recurring "
                    "performance, motifs, chorus reprises, fixed cameras/locations and "
                    "abstract imagery when deliberate and developed. Do not demand "
                    "new characters, locations, story arcs or constant novelty. Do not "
                    "flag stylistic preferences. Quote exact visual-body evidence "
                    "(24+ characters); a repetition finding needs quotes from both "
                    "supplied clips. Use ZERO-BASED clip_index. Return [] if sound.\n"
                    + json.dumps(cards, ensure_ascii=False)
                )
                review = self._call_llm_json(
                    user_prompt=request, system_prompt="Review music-video direction conservatively. Output JSON only.",
                    max_tokens=2400, temperature=0.15, thinking_budget=0,
                    json_schema=creative_review_schema(),
                )
                try:
                    grounded = grounded_review_findings(review, rows, set(range(start, end)))
                    # A small writer's subjective arc-mismatch judgment must
                    # not rewrite an accepted shot. Source obligations already
                    # have strict validators; preserve this judgment for review.
                    for item in review:
                        index = item["clip_index"]
                        if item["category"] == "repeated_action" and not justified_music_reprise(arc[index]):
                            findings[index] = grounded[index]
                        else:
                            review_notes.append(item)
                except ValueError as exc:
                    review_notes.append({"window": [start, end], "inconclusive": str(exc)})
                    print(f"[MusicVideoPlanner] Creative reviewer was inconclusive: {exc}")
                for i in range(start, end):
                    issue = music_visual_reset_issue(rows[i].get("video_prompt"), source)
                    if issue:
                        findings[i] = issue
                        hard_findings.add(i)
            if not findings:
                save(rows, complete=True)
                return rows
            if round_index == 1:
                if hard_findings:
                    raise RuntimeError("Music-video creative review still found issues in clips "
                                       + ", ".join(str(i + 1) for i in sorted(hard_findings))
                                       + ". Accepted clips are saved; retry planning to repair these clips.")
                for index, feedback in sorted(findings.items()):
                    review_notes.append({"clip_index": index, "severity": "advisory",
                                         "inconclusive": feedback})
                print("[MusicVideoPlanner] Variety review remains inconclusive for clips "
                      + ", ".join(str(i + 1) for i in sorted(findings))
                      + "; keeping validated shots and saving advisory notes.")
                save(rows, complete=True)
                return rows
            rows = repair_music_rows(
                rows, findings, replace, row_validator, checkpoint=save,
                on_failure=keep_advisory, on_repaired=repaired.add,
            )
        return rows

    # ── Convert LLM Output to ShotPlans ──────────────────────────────

    def _convert_to_shots(
        self,
        shot_dicts: list[dict],
        clips: list[dict],
        char_profiles: list[CharacterProfile],
        has_reference: bool,
        performer_map: dict[str, str],
        lyrics: Optional[list[dict]],
        speaker_names: dict[str, str],
        vocal_activity: Optional[list[str]] = None,
        project_context: str = "",
        ignore_physical_state_ledger: bool = False,
    ) -> list[ShotPlan]:
        """Convert raw LLM JSON output into validated ShotPlan objects."""
        shots = []
        for i, clip in enumerate(clips):
            raw = dict(shot_dicts[i]) if i < len(shot_dicts) else {}
            if ignore_physical_state_ledger:
                _strip_h3_physical_state_fields(raw)
            section = (clip.get("label") or clip.get("section_label") or "verse").lower()
            strategy = _SECTION_VISUAL_STRATEGY.get(section, _SECTION_VISUAL_STRATEGY["verse"])
            duration = clip.get("end", 0) - clip.get("start", 0)

            # Parse subjects
            subjects = []
            for s in raw.get("subjects_on_screen", []):
                if isinstance(s, dict):
                    subjects.append(SubjectRef.from_dict(s))
                elif isinstance(s, str):
                    subjects.append(SubjectRef(visual_description=s))

            # Fallback subjects from performer map
            if not subjects and section in performer_map:
                subjects.append(SubjectRef(visual_description=performer_map[section], position_or_relation="center frame"))

            # Parse camera plan
            cam_raw = raw.get("camera_plan", {})
            camera = CameraPlan(
                framing=cam_raw.get("framing", strategy["camera_default"]),
                angle=cam_raw.get("angle"),
                movement=cam_raw.get("movement"),
                movement_intensity=cam_raw.get("movement_intensity", strategy["movement_intensity"]),
                lens_feel=cam_raw.get("lens_feel"),
            )

            # Parse audio plan
            audio_raw = raw.get("audio_plan", {})
            audio = AudioPlan(
                mode=audio_raw.get("mode", "music_driven"),
                ambience=audio_raw.get("ambience"),
                timing_anchor="audio",
                vocal_activity=(
                    vocal_activity[i]
                    if vocal_activity and i < len(vocal_activity) else None
                ),
            )

            # Keep role-specific mouth cues consistent across generated views
            # of the shot. The final H3 compiler repeats this after later polish.
            from ..music_performance import (
                constrain_music_performance,
                music_performance_direction,
                resolved_music_performance_role,
            )
            if audio.mode == "music_driven":
                for subject in subjects:
                    subject.performance_role = resolved_music_performance_role(
                        subject, project_context,
                    )
            for key in ("video_prompt", "image_prompt", "spatial_setup", "ending_beat"):
                if isinstance(raw.get(key), str):
                    raw[key] = constrain_music_performance(
                        raw[key], subjects, audio.vocal_activity, project_context=project_context,
                    )
            for key in ("window_prompts", "keyframe_prompts", "action_beats", "performance_beats"):
                if isinstance(raw.get(key), list):
                    raw[key] = [
                        constrain_music_performance(value, subjects, audio.vocal_activity, project_context=project_context)
                        if isinstance(value, str) else value for value in raw[key]
                    ]
            if audio.mode == "music_driven":
                direction = music_performance_direction(subjects, audio.vocal_activity, project_context=project_context)
                if direction and raw.get("video_prompt") and direction not in raw["video_prompt"]:
                    raw["video_prompt"] = f"{raw['video_prompt']} {direction}"
                if direction and raw.get("window_prompts"):
                    raw["window_prompts"] = [
                        f"{value} {direction}"
                        if isinstance(value, str) and direction not in value
                        else value
                        for value in raw["window_prompts"]
                    ]

            physical_state = (
                None if ignore_physical_state_ledger
                else normalize_physical_state(raw.get("physical_state"))
            )
            if physical_state is not None:
                raw["physical_state"] = physical_state
                physical_prompt = format_physical_state_for_prompt(
                    physical_state,
                    include_transitions=False,
                )
                if physical_prompt and isinstance(raw.get("video_prompt"), str):
                    if physical_prompt not in raw["video_prompt"]:
                        raw["video_prompt"] = (
                            f"{raw['video_prompt'].rstrip()} {physical_prompt}"
                        )
                if isinstance(raw.get("video_prompt"), str):
                    missing_action_clauses = physical_transition_actions_missing_from_prompt(
                        physical_state,
                        raw["video_prompt"],
                    )
                    if missing_action_clauses:
                        raw["video_prompt"] = (
                            f"{raw['video_prompt'].rstrip()} Transition detail to show: "
                            + "; then ".join(missing_action_clauses)
                            + "."
                        )
                transition_actions = [
                    transition["action"]
                    for transition in physical_state["transitions"]
                ]
                action_beats = raw.get("action_beats")
                if not isinstance(action_beats, list):
                    action_beats = []
                existing_actions = {
                    re.sub(r"\s+", " ", str(value)).strip().casefold()
                    for value in action_beats if isinstance(value, str)
                }
                for action in transition_actions:
                    key = re.sub(r"\s+", " ", action).strip().casefold()
                    if key not in existing_actions:
                        action_beats.append(action)
                        existing_actions.add(key)
                raw["action_beats"] = action_beats
                if physical_prompt and isinstance(raw.get("window_prompts"), list):
                    raw["window_prompts"] = [
                        f"{value.rstrip()} {physical_prompt}"
                        if isinstance(value, str) and physical_prompt not in value
                        else value
                        for value in raw["window_prompts"]
                    ]

            # Parse dialogue beats if present
            dialogue_beats = None
            if raw.get("dialogue_beats"):
                dialogue_beats = [DialogueBeat.from_dict(db) for db in raw["dialogue_beats"]]

            # Determine image strategy
            image_strategy = "reference_edit" if has_reference else "fresh_generation"
            if section == "instrumental" and not has_reference:
                image_strategy = "fresh_generation"

            metadata = {
                "section": section,
                "beat_count": clip.get("beat_count", 0),
                "bpm": clip.get("bpm", 120),
                "clip_start": clip.get("start", 0),
                "clip_end": clip.get("end", 0),
            }
            creative_arc = getattr(self, "_music_arc", [])
            if i < len(creative_arc):
                metadata["_director_music_arc"] = copy.deepcopy(creative_arc[i])
                metadata["_director_music_arc"]["source_brief"] = getattr(
                    self, "_music_narrative_contract", {},
                ).get("source", "")
            narrative_contract = getattr(self, "_music_narrative_contract", {})
            if narrative_contract.get("events"):
                metadata["_director_music_narrative"] = {
                    "contract": copy.deepcopy(narrative_contract),
                    "allocation": copy.deepcopy(self._music_narrative_allocation),
                    "index": i,
                    "source_event_ids": list(raw.get("source_event_ids") or []),
                }
                if raw.get("_music_source_compiled"):
                    metadata["_director_music_narrative"].update({
                        "source_compiled": True,
                        "recovery_reason": str(raw.get("_music_source_recovery_reason") or ""),
                    })
                    if raw.get("_music_bridge_compiled"):
                        metadata["_director_music_narrative"]["bridge_compiled"] = True
            if physical_state is not None:
                metadata["physical_state"] = physical_state
            source_event_ids = (
                raw.get("source_event_ids") if narrative_contract.get("events")
                else clip.get("source_event_ids")
            )
            if isinstance(source_event_ids, list):
                usable_event_ids = [
                    str(value).strip()[:80]
                    for value in source_event_ids[:16]
                    if isinstance(value, str) and value.strip()
                ]
                if usable_event_ids:
                    metadata["source_event_ids"] = usable_event_ids

            continuity_strategy = str(
                raw.get("continuity_strategy") or "independent"
            ).strip().casefold()
            if continuity_strategy not in {
                "continuous", "independent", "extend_previous",
            }:
                continuity_strategy = "independent"

            shot = ShotPlan(
                shot_id=self._make_shot_id(i, "mv"),
                index=i,
                duration_sec=duration,
                skill_type="music_video",
                scene_goal=raw.get("scene_goal", f"{section} clip — {strategy['energy']} energy"),
                narrative_role=section,
                scene_type=raw.get("scene_type", "performance" if section != "instrumental" else "atmospheric"),
                source_mode_preference="i2v" if has_reference else "t2v",
                image_strategy=image_strategy,
                continuity_strategy=continuity_strategy,
                subjects_on_screen=subjects,
                spatial_setup=raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                visual_style=raw.get("visual_style", ""),
                lighting=raw.get("lighting", ""),
                mood=raw.get("mood", strategy["energy"]),
                action_beats=raw.get("action_beats", []),
                performance_beats=raw.get("performance_beats"),
                dialogue_beats=dialogue_beats,
                camera_plan=camera,
                audio_plan=audio,
                ending_beat=raw.get("ending_beat", ""),
                metadata=metadata,
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
