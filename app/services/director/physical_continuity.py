"""Typed physical-state contracts shared by director planners.

These checks validate declared blocking facts and transitions. They do not
prove that a generated image or video depicts the declared state.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict


PHYSICAL_STATE_PREDICATES = (
    "support",
    "contact",
    "posture",
    "location",
    "grounded",
    "prop",
)
MAX_PHYSICAL_FACTS = 12
MAX_PHYSICAL_TRANSITIONS = 8
MAX_PHYSICAL_TEXT_CHARS = 240
MAX_PHYSICAL_VALUE_CHARS = 80
GROUNDED_STATE_VALUES = ("grounded", "airborne")

_TEMPORARY_POSE_OR_PROP_ACTION_RE = re.compile(
    r"\b(?:holding|holds?|gripping|grips?|standing(?!\s+collar\b)|stands?|seated|sitting|sits?|"
    r"(?<![-\w])facing(?!\s+eyes?\b)|faces?|leaning|leans?|sliding|slides?|"
    r"riding(?!\s+boots?\b)|rides?|mounted|mounting|"
    r"climbing|climbs?|kneeling|kneels?|lying|lies?|resting|rests?|turning|turns?|"
    r"looking|looks?|grounded|airborne|wielding|wields?|carrying|carries?|"
    r"brandishing|brandishes|drawing|draws?|unsheathing|unsheathes?|"
    r"unholstering|unholsters?|sheathed|sheathe|sheathes|holstered|holsters?|"
    r"strapped|straps?|fastened|fastens?|clipped|clips?|tucked|tucks?|"
    r"stowed|stows?|unsheathed|unholstered|secured|secures?|"
    r"held|holds?|drawn|grasping|grasps?)\b",
    re.IGNORECASE,
)
_TEMPORARY_PROP_RELATION_RE = re.compile(
    r"\s+with\s+(?:(?:a|an|the|his|her|their|its|my|your|our)\s+)?"
    r"[A-Za-z][\w’'-]*(?:\s+[A-Za-z][\w’'-]*){0,5}\s+"
    r"(?:sheathed|holstered|strapped|fastened|clipped|tucked|stowed|"
    r"unsheathed|unholstered|drawn|held|gripped|wielded|secured)\b"
    r"[^,.;!?]*",
    re.IGNORECASE,
)


def identity_only_description(value: Any) -> str:
    """Remove temporary pose and prop clauses from a subject description.

    H3 subject definitions persist across a clip's full duration. When a typed
    physical ledger owns the changing pose or prop relation, retain stable
    appearance text that precedes that blocking and discard the temporal clause.
    Wardrobe remains a separate field and is not filtered here.
    """

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""

    kept_sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Attached/stowed prop clauses often describe the same fact that the
        # physical ledger owns (for example, "with a sword sheathed at her hip").
        sentence = _TEMPORARY_PROP_RELATION_RE.sub("", sentence)
        sentence = re.sub(r"\s+([,.;:!?])", r"\1", sentence).strip(" ,;:-")
        if not sentence:
            continue

        match = _TEMPORARY_POSE_OR_PROP_ACTION_RE.search(sentence)
        if match:
            prefix = sentence[:match.start()]
            # Drop the whole clause when the action is explicitly introduced
            # by a conjunction or comma; otherwise preserve the preceding
            # stable identity description and cut from the action verb.
            separator = re.search(r"(?:,\s*|\s+(?:and|while|as|then)\s+)\s*$", prefix, re.I)
            if separator:
                cut_at = separator.start()
            else:
                auxiliary = re.search(r"\b(?:is|are|was|were|now|already)\s+$", prefix, re.I)
                cut_at = auxiliary.start() if auxiliary else match.start()

            # A pronoun-led sentence that consists of blocking has no stable
            # identity material to retain.
            stable_prefix = sentence[:cut_at].strip(" ,;:-")
            if re.match(r"^(?:it|he|she|they)\b", stable_prefix, re.I) and not re.search(
                r"\b(?:fur|feather|scale|eye|eyes|mane|horn|tail|paw|paws|"
                r"coat|skin|whisker|whiskers|color|coloured|colored)\b",
                stable_prefix,
                re.I,
            ):
                continue
            sentence = stable_prefix

        sentence = re.sub(r"\s+([,.;:!?])", r"\1", sentence)
        sentence = re.sub(r"([,;:])\s*[,;:]+", r"\1", sentence)
        sentence = sentence.strip(" ,;:-")
        if sentence:
            kept_sentences.append(sentence)

    return ". ".join(kept_sentences)


class PhysicalFact(TypedDict):
    subject_id: str
    predicate: Literal[
        "support", "contact", "posture", "location", "grounded", "prop"
    ]
    value: str


class PhysicalTransition(TypedDict):
    subject_id: str
    predicate: Literal[
        "support", "contact", "posture", "location", "grounded", "prop"
    ]
    from_value: str | None
    to_value: str
    action: str


class _EstablishmentPhysicalTransition(PhysicalTransition, total=False):
    operation: Literal["establish"]
    event_id: str


class PhysicalState(TypedDict, total=False):
    opening: list[PhysicalFact]
    transitions: list[PhysicalTransition]
    closing: list[PhysicalFact]
    continuity: Literal["continuous", "independent", "time_jump"]


def _physical_fact_schema() -> dict[str, Any]:
    # Keep this model-facing grammar flat: llama.cpp alternatives can discard
    # sibling required fields. Predicate-specific rules stay in normalization.
    return {
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


def _physical_transition_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string", "maxLength": 80},
            "predicate": {
                "type": "string",
                "enum": list(PHYSICAL_STATE_PREDICATES),
            },
            "from_value": {
                "type": ["string", "null"],
                "maxLength": MAX_PHYSICAL_VALUE_CHARS,
                "description": "Null only when operation is establish.",
            },
            "to_value": {"type": "string", "maxLength": MAX_PHYSICAL_VALUE_CHARS},
            "action": {"type": "string", "maxLength": MAX_PHYSICAL_TEXT_CHARS},
            "operation": {"type": "string", "enum": ["establish"]},
            "event_id": {"type": "string", "maxLength": 80},
        },
        "required": [
            "subject_id", "predicate", "from_value", "to_value", "action",
        ],
        "additionalProperties": False,
    }


PHYSICAL_STATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "opening": {
            "type": "array",
            "maxItems": MAX_PHYSICAL_FACTS,
            "items": _physical_fact_schema(),
        },
        "transitions": {
            "type": "array",
            "maxItems": MAX_PHYSICAL_TRANSITIONS,
            "items": _physical_transition_schema(),
        },
        "closing": {
            "type": "array",
            "maxItems": MAX_PHYSICAL_FACTS,
            "items": _physical_fact_schema(),
        },
        "continuity": {
            "type": "string",
            "enum": ["continuous", "independent", "time_jump"],
        },
    },
    "required": ["opening", "transitions", "closing"],
    "additionalProperties": False,
}


_STATE_KEYS = ("opening", "transitions", "closing")
_ALLOWED_CONTINUITY = {"continuous", "independent", "time_jump"}


def physical_state_json_schema() -> dict[str, Any]:
    """Return a fresh JSON-schema fragment for an optional shot property."""

    return copy.deepcopy(PHYSICAL_STATE_JSON_SCHEMA)


def physical_state_contract_instructions() -> str:
    """Return concise generation rules for a typed physical-state ledger."""

    return (
        "PHYSICAL STATE CONTRACT:\n"
        "- Use the exact stable character_id from subjects_on_screen as subject_id; "
        "copy each ID exactly wherever the same actor appears. For support/contact, "
        "subject_id is the actor whose relation is being stated, and the counterpart "
        "belongs in the short value. Never invent prop IDs or treat a prop as cast; "
        "put a prop in the actor-owned prop fact instead.\n"
        "- Keep every fact value a compact state atom of at most 80 characters, such "
        "as 'astride char_1', 'touching char_1', 'upright', or 'holding lantern'. "
        "Do not put paragraphs, event narration, or multiple synonyms in a value.\n"
        "- Use support for what bears or carries an actor, contact for touch/grip, "
        "posture for body pose, and location for place. Keep grounded separate from "
        "posture and set grounded.value to exactly 'grounded' or 'airborne'.\n"
        "- Include only facts needed to show the opening, action, and ending. Put "
        "natural-language motion in ordered transitions[].action; use compact state "
        "atoms in from_value and to_value. Keep action_beats and video_prompt aligned "
        "with those transition actions. Preserve state atoms verbatim: in a continuous "
        "shot, copy the preceding closing subject_id, predicate, and value exactly "
        "into opening. Each ordered transition.from_value must exactly match the "
        "current value for that subject and predicate; its to_value becomes the new "
        "current value. The final current value must exactly match closing, unless a "
        "later ordered transition updates it. Do not paraphrase state values. A missing "
        "opening key may use operation='establish' only when the exact assigned source "
        "event explicitly establishes that actor-owned relation; this requires event "
        "provenance and is not a way to infer an earlier state."
    )


def _clean_text(value: Any, *, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} needs a non-empty {field}")
    cleaned = re.sub(r"\s+", " ", value).strip()
    maximum = (
        80 if field in {"subject_id", "event_id"}
        else MAX_PHYSICAL_VALUE_CHARS
        if field in {"value", "from_value", "to_value"}
        else MAX_PHYSICAL_TEXT_CHARS
    )
    if len(cleaned) > maximum:
        raise ValueError(f"{location}.{field} exceeds {maximum} characters")
    return cleaned


def _text_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _normalize_predicate(value: Any, *, location: str) -> str:
    predicate = _clean_text(value, field="predicate", location=location)
    predicate = predicate.casefold().replace("-", "_").replace(" ", "_")
    if predicate not in PHYSICAL_STATE_PREDICATES:
        allowed = ", ".join(PHYSICAL_STATE_PREDICATES)
        raise ValueError(f"{location} has unsupported predicate {predicate!r}; use {allowed}")
    return predicate


def _normalize_state_value(
    value: Any,
    *,
    predicate: str,
    field: str,
    location: str,
) -> str:
    cleaned = _clean_text(value, field=field, location=location)
    if cleaned.casefold().replace("_", " ").strip() in {
        "unknown", "unspecified", "not specified", "not established",
    }:
        raise ValueError(
            f"{location}.{field} cannot use an unknown placeholder as a physical fact"
        )
    if predicate == "grounded":
        canonical = cleaned.casefold()
        if canonical not in GROUNDED_STATE_VALUES:
            raise ValueError(
                f"{location}.{field} for predicate 'grounded' must be exactly "
                "'grounded' or 'airborne'"
            )
        return canonical
    return cleaned


def _normalize_fact_list(value: Any, *, field: str) -> list[PhysicalFact]:
    if not isinstance(value, list):
        raise ValueError(f"physical_state.{field} must be an array")
    if len(value) > MAX_PHYSICAL_FACTS:
        raise ValueError(
            f"physical_state.{field} has more than {MAX_PHYSICAL_FACTS} facts"
        )

    facts: list[PhysicalFact] = []
    seen: dict[tuple[str, str], str] = {}
    for index, raw in enumerate(value):
        location = f"physical_state.{field}[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{location} must be an object")
        allowed = {"subject_id", "predicate", "value"}
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f"{location} has unsupported keys: {', '.join(sorted(extra))}")
        subject_id = _clean_text(raw.get("subject_id"), field="subject_id", location=location)
        predicate = _normalize_predicate(raw.get("predicate"), location=location)
        fact_value = _normalize_state_value(
            raw.get("value"), predicate=predicate, field="value", location=location,
        )
        key = (_text_key(subject_id), predicate)
        normalized_value = _text_key(fact_value)
        previous_value = seen.get(key)
        if previous_value is not None:
            if previous_value != normalized_value:
                raise ValueError(
                    f"{location} conflicts with another {predicate} fact for "
                    f"subject {subject_id!r}"
                )
            continue
        seen[key] = normalized_value
        facts.append({
            "subject_id": subject_id,
            "predicate": predicate,  # type: ignore[typeddict-item]
            "value": fact_value,
        })
    return facts


def normalize_physical_state(value: Any) -> dict[str, Any] | None:
    """Validate and canonicalize one optional physical-state object.

    An omitted state returns ``None``. Malformed typed state raises
    ``ValueError`` with a field-specific message so callers can surface or
    repair the contract issue.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("physical_state must be an object")
    extra = set(value) - {*_STATE_KEYS, "continuity"}
    if extra:
        raise ValueError(
            "physical_state has unsupported keys: " + ", ".join(sorted(extra))
        )
    missing = [key for key in _STATE_KEYS if key not in value]
    if missing:
        raise ValueError(
            "physical_state is missing required keys: " + ", ".join(missing)
        )

    opening = _normalize_fact_list(value["opening"], field="opening")
    closing = _normalize_fact_list(value["closing"], field="closing")
    raw_transitions = value["transitions"]
    if not isinstance(raw_transitions, list):
        raise ValueError("physical_state.transitions must be an array")
    if len(raw_transitions) > MAX_PHYSICAL_TRANSITIONS:
        raise ValueError(
            "physical_state.transitions has more than "
            f"{MAX_PHYSICAL_TRANSITIONS} transitions"
        )
    transitions: list[PhysicalTransition | _EstablishmentPhysicalTransition] = []
    for index, raw in enumerate(raw_transitions):
        location = f"physical_state.transitions[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{location} must be an object")
        allowed = {
            "subject_id", "predicate", "from_value", "to_value", "action",
            "operation", "event_id",
        }
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f"{location} has unsupported keys: {', '.join(sorted(extra))}")
        predicate = _normalize_predicate(raw.get("predicate"), location=location)
        operation_value = raw.get("operation")
        if "operation" in raw and operation_value is None:
            raise ValueError(f"{location}.operation must be 'establish' when provided")
        if operation_value is not None:
            if not isinstance(operation_value, str) or operation_value.strip().casefold() != "establish":
                raise ValueError(f"{location}.operation must be 'establish' when provided")
            operation = "establish"
            if "from_value" not in raw or raw.get("from_value") is not None:
                raise ValueError(f"{location}.from_value must be null for establish")
            event_id = _clean_text(raw.get("event_id"), field="event_id", location=location)
        else:
            operation = None
            if "event_id" in raw:
                raise ValueError(f"{location}.event_id requires operation='establish'")
            event_id = None
        transition: PhysicalTransition | _EstablishmentPhysicalTransition = {
            "subject_id": _clean_text(
                raw.get("subject_id"), field="subject_id", location=location,
            ),
            "predicate": predicate,  # type: ignore[typeddict-item]
            "from_value": (
                None if operation == "establish" else _normalize_state_value(
                    raw.get("from_value"), predicate=predicate,
                    field="from_value", location=location,
                )
            ),
            "to_value": _normalize_state_value(
                raw.get("to_value"), predicate=predicate,
                field="to_value", location=location,
            ),
            "action": _clean_text(
                raw.get("action"), field="action", location=location,
            ),
        }
        if operation == "establish":
            transition["operation"] = "establish"  # type: ignore[literal-required]
            transition["event_id"] = event_id  # type: ignore[literal-required]
        transitions.append(transition)

    continuity = str(value.get("continuity", "continuous") or "continuous")
    continuity = continuity.strip().casefold().replace("-", "_").replace(" ", "_")
    if continuity not in _ALLOWED_CONTINUITY:
        allowed = ", ".join(sorted(_ALLOWED_CONTINUITY))
        raise ValueError(
            f"physical_state.continuity must be one of: {allowed}"
        )
    return {
        "opening": opening,
        "transitions": transitions,
        "closing": closing,
        "continuity": continuity,
    }


def physical_state_from_row(row: Any) -> Any:
    """Read state from a raw planner row or a serialized ShotPlan."""

    if not isinstance(row, Mapping):
        return None
    if "physical_state" in row:
        return row.get("physical_state")
    metadata = row.get("metadata")
    return metadata.get("physical_state") if isinstance(metadata, Mapping) else None


def format_physical_state_for_prompt(
    value: Any,
    *,
    include_transitions: bool = True,
) -> str:
    """Render typed facts and, by default, transition actions as visible blocking."""

    state = normalize_physical_state(value)
    if state is None:
        return ""
    lines: list[str] = []
    continuity = state["continuity"]
    if continuity == "independent":
        lines.append(
            "This is an intentional independent cut; establish its opening "
            "facts directly."
        )
    elif continuity == "time_jump":
        lines.append(
            "This is an intentional time jump; establish its opening facts "
            "without implying an unseen transition."
        )
    if state["opening"]:
        lines.append("Opening physical facts: " + "; ".join(
            f"{fact['subject_id']} {fact['predicate']}: {fact['value']}"
            for fact in state["opening"]
        ) + ".")
    if include_transitions:
        for index, transition in enumerate(state["transitions"], start=1):
            if transition.get("operation") == "establish":
                lines.append(
                    f"Physical transition {index}: establish {transition['subject_id']} "
                    f"{transition['predicate']} as {transition['to_value']} when "
                    f"{transition['action']}."
                )
            else:
                lines.append(
                    f"Physical transition {index}: {transition['subject_id']} "
                    f"{transition['predicate']} changes from "
                    f"{transition['from_value']} to {transition['to_value']} as "
                    f"{transition['action']}."
                )
    if state["closing"]:
        lines.append("At the final frame: " + "; ".join(
            f"{fact['subject_id']} {fact['predicate']}: {fact['value']}"
            for fact in state["closing"]
        ) + ".")
    if not lines:
        return ""
    if include_transitions:
        lines.append(
            "Keep these facts physically readable in frame and show each "
            "stated transition with its action."
        )
    else:
        lines.append(
            "Keep these facts physically readable in frame; final-frame facts "
            "describe only the end of the clip."
        )
    return " ".join(lines)


def _facts_by_key(facts: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], PhysicalFact]:
    return {
        (_text_key(fact["subject_id"]), str(fact["predicate"])): {
            "subject_id": str(fact["subject_id"]),
            "predicate": str(fact["predicate"]),  # type: ignore[typeddict-item]
            "value": str(fact["value"]),
        }
        for fact in facts
    }


def _state_label(row: Mapping[str, Any], index: int) -> str:
    label = row.get("shot_id") or row.get("clip_id") or row.get("scene_id")
    return str(label) if label else f"shot {index + 1}"


_SOURCE_PHYSICAL_RELATION_PATTERNS: tuple[
    tuple[str, frozenset[str], re.Pattern[str]], ...
] = (
    (
        "support",
        frozenset({"support"}),
        re.compile(
            r"\b(?:climb(?:s|ed|ing)?\s+(?:aboard|on|onto|astride)|"
            r"aboard|astride|rides?|riding|rode|ridden|mounted|mounts?|mounting|"
            r"supported\s+by|"
            r"stands?\s+(?:on|atop|astride)|standing\s+(?:on|atop|astride)|"
            r"slide(?:s|d|ing)\s+down\s+(?:\w+'s\s+)?(?:the\s+)?"
            r"(?:shoulder|back|side))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "carrying",
        frozenset({"support", "contact"}),
        re.compile(
            r"\b(?:carry|carries|carrying|carried)\s+(?!(?:on|out|over|through|"
            r"forward|away)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "contact",
        frozenset({"contact"}),
        re.compile(
            r"\b(?:touch|touches|touched|touching|pat|pats|patted|patting|"
            r"grip|grips|gripped|gripping)(?=\s|\b)|"
            r"\b(?:use|uses|used|using)\s+(?:the|a|an|his|her|their|its)?\s*"
            r"(?:saddle\s+strap|strap|reins|handhold|handrail|rail|handle)\b|"
            r"\b(?:hold|holds|held|holding)\s+(?:on\s+to|onto|"
            r"(?:the|a|an|his|her|their|my|your)?\s*"
            r"(?:hand|hands|wrist|arm|strap|reins|rail|railing|handle|"
            r"sword|sleeve|shoulder|waist|partner|rider|child|person|"
            r"lantern|saddle|snout|face|head)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "grounding",
        frozenset({"grounded"}),
        re.compile(
            r"\b(?:lands?|landed|landing)(?=\s*(?:[,.;]|$)|\s+(?:on|onto|"
            r"safely|back)\b)|"
            r"\b(?:takes?|took|taking)\s+flight\b|"
            r"\b(?:lifts?|lifted|lifting)\s+off\b|"
            r"\bputs?\s+(?:both\s+)?(?:feet|paws)\s+(?:down\s+)?"
            r"(?:on|onto)\s+(?:the\s+)?ground\b",
            re.IGNORECASE,
        ),
    ),
)
_NEGATED_SOURCE_RELATION_RE = re.compile(
    r"\b(?:not|never|without|don't|doesn't|didn't|avoid|avoids|avoiding)"
    r"(?:\s+\w+){0,3}\s*$",
    re.IGNORECASE,
)
_ABSTRACT_CARRY_TARGET_RE = re.compile(
    r"^\s*(?:(?:the|a|an|his|her|their|my|your|our|its)\s+)?"
    r"(?:(?:long|high|low|sustained|held|main|melodic|beautiful|same)\s+){0,3}"
    r"(?:melody|song|music|tune|note|beat|rhythm|chorus|verse|lyric|lyrics|"
    r"voice|scene|story|mood|energy)\b",
    re.IGNORECASE,
)


def _source_event_texts(value: Any) -> list[str]:
    """Extract bounded event wording from one row's source-event payload."""

    if isinstance(value, str):
        return [value.strip()[:1600]] if value.strip() else []
    if isinstance(value, Mapping):
        direct_text = value.get("text") or value.get("description")
        if isinstance(direct_text, str) and direct_text.strip():
            return [direct_text.strip()[:1600]]
        value = value.get("events")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    output: list[str] = []
    for item in value[:24]:
        if isinstance(item, str) and item.strip():
            output.append(item.strip()[:1600])
        elif isinstance(item, Mapping):
            text = item.get("text") or item.get("description")
            if isinstance(text, str) and text.strip():
                output.append(text.strip()[:1600])
    return output


def _source_event_entries(value: Any) -> list[tuple[str | None, str]]:
    """Return source event IDs with their immutable wording when available."""

    if isinstance(value, Mapping):
        if isinstance(value.get("event_id"), str) and isinstance(value.get("text"), str):
            return [(value["event_id"], value["text"].strip())]
        nested = value.get("events")
        if nested is not None:
            return _source_event_entries(nested)
        return [(None, text) for text in _source_event_texts(value)]
    if isinstance(value, str):
        return [(None, value.strip())] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    output: list[tuple[str | None, str]] = []
    for item in value[:24]:
        if isinstance(item, str) and item.strip():
            output.append((None, item.strip()[:1600]))
        elif isinstance(item, Mapping):
            event_id = item.get("event_id")
            text = item.get("text") or item.get("description")
            if isinstance(text, str) and text.strip():
                output.append((event_id if isinstance(event_id, str) else None, text.strip()[:1600]))
    return output


_LOWERING_POSTURE_RE = re.compile(r"\blower(?:s|ed|ing)?\b", re.IGNORECASE)
_LOWERING_SHOULDER_RE = re.compile(
    r"\blower(?:s|ed|ing)?\b[^.;!?]{0,50}\bshoulder\b|"
    r"\bshoulder\b[^.;!?]{0,40}\blower(?:s|ed|ing)?\b",
    re.IGNORECASE,
)
_SOURCE_ACTOR_TERMINAL_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "takeoff",
        re.compile(
            r"\b(?:takes?|took|taking)\s+(?:flight|off)|"
            r"\b(?:lifts?|lifted|lifting)\s+off\b",
            re.IGNORECASE,
        ),
    ),
    (
        "landing",
        re.compile(
            r"\b(?:lands?|landed|alights?|alighted)\b|"
            r"\btouch(?:es|ed)\s+down\b",
            re.IGNORECASE,
        ),
    ),
    (
        "feet_ground",
        re.compile(
            r"\bputs?\s+(?:both\s+)?(?:feet|paws)\s+(?:down\s+)?"
            r"(?:on|onto)\s+(?:the\s+)?(?:(?:stone|bare|solid)\s+)?"
            r"(?:ground|floor|earth|soil|pavement|ledge)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dismount",
        re.compile(
            r"\b(?:slides?|sliding|climbs?|climbing)\s+down\s+"
            r"(?:(?:from|off)\s+)?(?:(?:its|their|his|her|the)\s+)?"
            r"(?:[A-Z][\w'-]*\s+)?(?:shoulder|back|side|mount|saddle)\b|"
            r"\bdismounts?|dismounting\b",
            re.IGNORECASE,
        ),
    ),
    (
        "lower_shoulder",
        _LOWERING_SHOULDER_RE,
    ),
)


def _named_source_actor_cues(
    row: Mapping[str, Any], text: str,
) -> list[tuple[str, str, int]]:
    """Resolve only unambiguous cast aliases used as direct grammatical subjects."""

    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        return []
    alias_owners: dict[str, set[str]] = {}
    for subject in subjects:
        if not isinstance(subject, Mapping):
            continue
        character_id = str(subject.get("character_id") or "").strip()
        if not character_id:
            continue
        for field in ("character_id", "speaker_name"):
            alias = str(subject.get(field) or "").strip()
            if alias:
                alias_owners.setdefault(alias.casefold(), set()).add(character_id)

    cues: list[tuple[str, str, int]] = []
    for alias_key, owners in alias_owners.items():
        if len(owners) != 1:
            continue
        # Recover original spelling from the row; matching itself is case-insensitive.
        alias = next(
            str(subject.get(field) or "").strip()
            for subject in subjects if isinstance(subject, Mapping)
            for field in ("character_id", "speaker_name")
            if str(subject.get(field) or "").strip().casefold() == alias_key
        )
        alias_matches = list(re.finditer(
            rf"(?<![\w]){re.escape(alias)}(?![\w])", text, re.IGNORECASE,
        ))
        for family, pattern in _SOURCE_ACTOR_TERMINAL_CUES:
            for cue_match in pattern.finditer(text):
                if family == "lower_shoulder":
                    # This cue is event-specific and the actor must be the
                    # grammatical subject of the lower-shoulder clause.
                    if not re.search(
                        rf"(?<![\w]){re.escape(alias)}(?![\w])\s+"
                        r"(?:slowly\s+|gently\s+|carefully\s+)?"
                        r"(?:lowers?|lowered|lowering)\b[^.;!?]{0,30}\bshoulder\b",
                        text, re.IGNORECASE,
                    ):
                        continue
                    if _NEGATED_SOURCE_RELATION_RE.search(
                        text[max(0, cue_match.start() - 48):cue_match.start()]
                    ):
                        continue
                    cues.append((next(iter(owners)), family, text.casefold().find("lower")))
                    break
                for alias_match in alias_matches:
                    if cue_match.start() < alias_match.end():
                        continue
                    before_alias = text[max(0, alias_match.start() - 24):alias_match.start()]
                    if re.search(
                        r"\b(?:to|at|with|from|for|of|near|toward|towards)\s+$",
                        before_alias,
                        re.IGNORECASE,
                    ):
                        continue
                    bridge = text[alias_match.end():cue_match.start()]
                    if len(bridge) > 220 or re.search(r"[.!?]", bridge):
                        continue
                    # A second named subject, including an unknown proper
                    # name, makes the cue ambiguous. Coordinated actions with
                    # an omitted subject remain supportable ("Kitten slides
                    # down ... and puts both feet on the ground").
                    if re.search(
                        r"\b(?:while|as|because|although|and|then)\s+[A-Z][\w'-]*|"
                        r"\b(?:as|while)\s+(?:he|she|they|it)\b|"
                        r"\b(?:while|as|because|although)\s+(?:a|an|the)\s+[\w'-]+\b|"
                        r"\b(?:watches?|sees?|follows?|looks?\s+at|turns?\s+to|faces?)\s+"
                        r"[A-Z][\w'-]*",
                        bridge,
                    ):
                        continue
                    other_known = any(
                        other_alias_key != alias_key
                        and re.search(
                            rf"(?<![\w]){re.escape(other_alias_key)}(?![\w])",
                            bridge, re.IGNORECASE,
                        )
                        for other_alias_key in alias_owners
                    )
                    if other_known:
                        continue
                    if _NEGATED_SOURCE_RELATION_RE.search(
                        text[max(0, cue_match.start() - 48):cue_match.start()]
                    ):
                        continue
                    # Reject a new proper-name subject even when that person
                    # is not in the visible cast list.
                    if re.search(r"\b(?:and|then|while|as)\s+[A-Z][a-z]+\s+", bridge):
                        continue
                    cues.append((next(iter(owners)), family, cue_match.start()))
                    break
    return list(dict.fromkeys(cues))


_ESTABLISHMENT_CONTACT_ACTION_RE = re.compile(
    r"\b(?:touch(?:es|ed|ing)?|pat(?:s|ted|ting)?|grip(?:s|ped|ping)?|"
    r"grasps?|grab(?:s|bed|bing)?|hold(?:s|ing|held)?|us(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_ESTABLISHMENT_SUPPORT_RELATION_RE = re.compile(
    r"\b(?:astride|aboard|mounted|riding|on|onto)\b", re.IGNORECASE,
)
_ESTABLISHMENT_TARGET_STOPWORDS = {
    "a", "an", "the", "his", "her", "their", "its", "my", "your", "our",
    "none", "no", "and", "of", "to", "with", "on", "onto", "as", "in",
    "at", "astride", "aboard", "mounted", "riding", "touching", "touch",
    "patting", "pat", "gripping", "grip", "holding", "hold", "using", "use",
}


def _cast_alias_owners(row: Mapping[str, Any]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        return aliases
    for subject in subjects:
        if not isinstance(subject, Mapping):
            continue
        character_id = str(subject.get("character_id") or "").strip()
        if not character_id:
            continue
        for field in ("character_id", "speaker_name"):
            alias = str(subject.get(field) or "").strip()
            if alias:
                aliases.setdefault(alias.casefold(), set()).add(character_id)
    return aliases


def _establishment_owner_is_direct(
    row: Mapping[str, Any], text: str, cue_position: int, subject_id: str,
) -> bool:
    aliases = _cast_alias_owners(row)
    alias_matches: list[tuple[int, int, str]] = []
    for alias_key, owners in aliases.items():
        if len(owners) != 1:
            continue
        alias = next(
            str(subject.get(field) or "").strip()
            for subject in row.get("subjects_on_screen", [])
            if isinstance(subject, Mapping)
            for field in ("character_id", "speaker_name")
            if str(subject.get(field) or "").strip().casefold() == alias_key
        )
        for match in re.finditer(
            rf"(?<![\w]){re.escape(alias)}(?![\w])", text[:cue_position], re.IGNORECASE,
        ):
            alias_matches.append((match.start(), match.end(), next(iter(owners))))
    if not alias_matches:
        return False

    last_start, last_end, last_owner = max(alias_matches, key=lambda item: item[1])
    if _text_key(last_owner) != _text_key(subject_id):
        return False
    clause_start = max(text.rfind(mark, 0, cue_position) for mark in ".!?;") + 1
    clause_aliases = [
        item for item in alias_matches
        if item[0] >= clause_start and item[1] <= cue_position
    ]
    # Coordinated named actors make the grammatical owner unclear, even if
    # the requested owner happens to be the last name before the action.
    if len({_text_key(item[2]) for item in clause_aliases}) > 1:
        return False
    before_alias = text[max(clause_start, last_start - 24):last_start]
    if re.search(r"\b(?:to|at|with|from|for|of|near|toward|towards)\s+$", before_alias, re.I):
        return False
    bridge = text[last_end:cue_position]
    if len(bridge) > 220:
        return False
    if re.search(
        r"\b(?:and|then|while|as|because|although)\s+(?:[A-Z][\w'-]*|he|she|they|it)\b",
        bridge,
    ):
        return False
    if re.search(r"[.!?]", bridge):
        return False
    return True


def _establishment_object_text(text: str, start: int) -> str:
    suffix = text[start:start + 120]
    # Stop at a new clause/action so a nearby but unrelated noun cannot prove
    # the proposed target relation.
    return re.split(
        r"[,.;!?]|\b(?:and|then|while|as|with)\b", suffix, maxsplit=1, flags=re.I,
    )[0]


def _establishment_target_terms(
    value: str, row: Mapping[str, Any] | None = None,
) -> set[str]:
    text = value
    if row is not None:
        for alias in _cast_alias_owners(row):
            text = re.sub(rf"(?<![\w]){re.escape(alias)}(?![\w])", " ", text, flags=re.I)
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][\w'-]*", text)
        if token.casefold() not in _ESTABLISHMENT_TARGET_STOPWORDS
        and not re.fullmatch(r"char_\d+", token, re.I)
    }


def _target_character_ids(value: str, row: Mapping[str, Any]) -> set[str]:
    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        return set()
    identifiers = {
        str(subject.get("character_id") or "").strip()
        for subject in subjects
        if isinstance(subject, Mapping)
        and str(subject.get("character_id") or "").strip()
    }
    identifiers = {
        identifier for identifier in identifiers
        if re.search(rf"(?<![\w]){re.escape(identifier)}(?![\w])", value, re.I)
    }
    for alias, owners in _cast_alias_owners(row).items():
        if len(owners) == 1 and re.search(
            rf"(?<![\w]){re.escape(alias)}(?![\w])", value, re.I,
        ):
            identifiers.update(owners)
    return identifiers


def _establishment_source_supports(
    row: Mapping[str, Any], transition: Mapping[str, Any], event_text: str,
) -> bool:
    subject_id = str(transition.get("subject_id") or "").strip()
    predicate = str(transition.get("predicate") or "")
    target = str(transition.get("to_value") or "").strip()
    action = str(transition.get("action") or "").strip()
    if not subject_id or not target or not action:
        return False

    if predicate == "grounded":
        expected_family = "takeoff" if target.casefold() == "airborne" else "landing"
        if target.casefold() not in {"airborne", "grounded"}:
            return False
        if target.casefold() == "grounded":
            allowed_families = {"landing", "feet_ground"}
        else:
            allowed_families = {expected_family}
        return any(
            cue_subject == subject_id and family in allowed_families
            and re.search(pattern.pattern, action, pattern.flags)
            for cue_subject, family, _ in _named_source_actor_cues(row, event_text)
            for family_name, pattern in _SOURCE_ACTOR_TERMINAL_CUES
            if family_name == family
        )

    if predicate == "support":
        relation = next(
            (match for match in _ESTABLISHMENT_SUPPORT_RELATION_RE.finditer(target)),
            None,
        )
        if relation is None or not re.search(
            rf"(?<![\w]){re.escape(relation.group(0))}(?![\w])", action, re.I,
        ):
            return False
        target_ids = _target_character_ids(target, row)
        target_terms = _establishment_target_terms(target, row)
        if not target_ids and not target_terms:
            return False
        if subject_id in target_ids or len(target_ids) > 1:
            return False
        for source_relation in _ESTABLISHMENT_SUPPORT_RELATION_RE.finditer(event_text):
            if source_relation.group(0).casefold() != relation.group(0).casefold():
                continue
            if _NEGATED_SOURCE_RELATION_RE.search(
                event_text[max(0, source_relation.start() - 48):source_relation.start()]
            ):
                continue
            if not _establishment_owner_is_direct(
                row, event_text, source_relation.start(), subject_id,
            ):
                continue
            target_text = _establishment_object_text(event_text, source_relation.end())
            if not target_terms.issubset(set(re.findall(r"[A-Za-z][\w'-]*", target_text.casefold()))):
                continue
            if not target_ids:
                continue
            target_id = next(iter(target_ids))
            subjects = row.get("subjects_on_screen", [])
            aliases = [
                str(subject.get(field) or "").strip()
                for subject in subjects if isinstance(subject, Mapping)
                and _text_key(str(subject.get("character_id") or "")) == _text_key(target_id)
                for field in ("character_id", "speaker_name")
                if str(subject.get(field) or "").strip()
            ] if isinstance(subjects, list) else []
            target_named = any(
                re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", target_text, re.I)
                for alias in aliases
            )
            possessive_back = bool(re.search(
                r"\b(?:its|their|his|her)\s+(?:back|shoulder|side|saddle)\b",
                target_text, re.I,
            ))
            other_ids = {
                str(subject.get("character_id") or "").strip()
                for subject in subjects if isinstance(subject, Mapping)
                and str(subject.get("character_id") or "").strip()
                and _text_key(str(subject.get("character_id") or "")) != _text_key(subject_id)
            } if isinstance(subjects, list) else set()
            if target_named or (possessive_back and other_ids == {target_id}):
                return True
        return False

    if predicate == "contact":
        target_ids = _target_character_ids(target, row)
        if subject_id in target_ids:
            return False
        terms = _establishment_target_terms(target, row)
        if not target_ids and not terms:
            return False
        for cue in _ESTABLISHMENT_CONTACT_ACTION_RE.finditer(event_text):
            if _NEGATED_SOURCE_RELATION_RE.search(
                event_text[max(0, cue.start() - 48):cue.start()]
            ):
                continue
            if not _establishment_owner_is_direct(row, event_text, cue.start(), subject_id):
                continue
            object_text = _establishment_object_text(event_text, cue.end())
            object_words = set(re.findall(r"[A-Za-z][\w'-]*", object_text.casefold()))
            if not terms.issubset(object_words):
                continue
            if not _ESTABLISHMENT_CONTACT_ACTION_RE.search(action):
                continue
            action_words = set(re.findall(r"[A-Za-z][\w'-]*", action.casefold()))
            if not terms.issubset(action_words):
                continue
            if target_ids:
                subjects = row.get("subjects_on_screen", [])
                for target_id in target_ids:
                    aliases = [
                        str(subject.get(field) or "").strip()
                        for subject in subjects if isinstance(subject, Mapping)
                        and _text_key(str(subject.get("character_id") or "")) == _text_key(target_id)
                        for field in ("character_id", "speaker_name")
                        if str(subject.get(field) or "").strip()
                    ] if isinstance(subjects, list) else []
                    if not any(
                        re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", object_text, re.I)
                        for alias in aliases
                    ):
                        break
                else:
                    return True
            else:
                return True
        return False

    # Event-scoped establishment stays narrow until a predicate has an
    # equivalently direct actor/action/value proof.
    return False


def _closing_fact_for_actor(
    state: Mapping[str, Any], subject_id: str, predicate: str,
) -> Mapping[str, Any] | None:
    return next((
        fact for fact in state.get("closing", [])
        if _text_key(str(fact.get("subject_id") or "")) == _text_key(subject_id)
        and fact.get("predicate") == predicate
    ), None)


def _named_terminal_state_issues(
    row: Mapping[str, Any], state: Mapping[str, Any], events: Sequence[tuple[str | None, str]], label: str,
) -> list[str]:
    issues: list[str] = []
    subject_ids = {
        str(subject.get("character_id") or "").strip()
        for subject in row.get("subjects_on_screen", [])
        if isinstance(subject, Mapping) and str(subject.get("character_id") or "").strip()
    } if isinstance(row.get("subjects_on_screen"), list) else set()
    terminal_cues: dict[str, list[tuple[int, int, str]]] = {}
    for event_index, (_, text) in enumerate(events):
        for subject_id, family, text_position in _named_source_actor_cues(row, text):
            if family not in {"takeoff", "landing", "feet_ground", "dismount"}:
                continue
            terminal_cues.setdefault(subject_id, []).append((event_index, text_position, family))

    for subject_id, actor_cues in terminal_cues.items():
        _, _, family = max(actor_cues, key=lambda cue: cue[:2])
        grounded = _closing_fact_for_actor(state, subject_id, "grounded")
        support = _closing_fact_for_actor(state, subject_id, "support")
        contact = _closing_fact_for_actor(state, subject_id, "contact")
        posture = _closing_fact_for_actor(state, subject_id, "posture")
        if family in {"takeoff", "landing", "feet_ground"}:
            expected = "airborne" if family == "takeoff" else "grounded"
            if grounded is None or str(grounded.get("value", "")).casefold() != expected:
                issues.append(
                    f"{label} explicit named-actor {family.replace('_', '-')} cue requires "
                    f"closing grounded='{expected}' for {subject_id!r}"
                )
        if family == "feet_ground":
            support_value = str(support.get("value") or "") if support else ""
            contact_value = str(contact.get("value") or "") if contact else ""
            latest_feet_cue = max(
                (cue for cue in actor_cues if cue[2] == "feet_ground"),
                key=lambda cue: cue[:2],
            )
            dismount_precedes_feet = any(
                cue[2] == "dismount" and cue[:2] < latest_feet_cue[:2]
                for cue in actor_cues
            )
            ground_relation = bool(re.search(
                r"\b(?:ground|floor|earth|soil|pavement|ledge)\b",
                support_value + " " + contact_value,
                re.IGNORECASE,
            ))
            seated = bool(posture and re.search(
                r"\b(?:seated|sitting|astride|mounted|riding)\b",
                str(posture.get("value") or ""), re.IGNORECASE,
            ))
            other_actor_support = any(
                re.search(rf"(?<![\w]){re.escape(other_id)}(?![\w])", support_value, re.IGNORECASE)
                for other_id in subject_ids - {subject_id}
            )
            if not ground_relation or (
                dismount_precedes_feet and (support is None or not re.search(
                    r"\b(?:ground|floor|earth|soil|pavement|ledge)\b",
                    support_value,
                    re.IGNORECASE,
                ) or seated or other_actor_support)
            ):
                issues.append(
                    f"{label} explicit named-actor feet-on-ground cue conflicts with closing "
                    f"support/contact or preceding dismount state for {subject_id!r}"
                )
        if family == "landing":
            location = _closing_fact_for_actor(state, subject_id, "location")
            location_value = str(location.get("value") or "") if location else ""
            if location and re.search(
                r"\b(?:aerial|airborne|in\s+flight|flying)\b",
                location_value,
                re.IGNORECASE,
            ) and not re.search(
                r"\b(?:no\s+longer|not|never|isn['’]?t|wasn['’]?t)\s+"
                r"(?:aerial|airborne|in\s+flight|flying)\b",
                location_value,
                re.IGNORECASE,
            ):
                issues.append(
                    f"{label} explicit named-actor landing cue conflicts with closing "
                    f"aerial location for {subject_id!r}"
                )
        if family == "dismount":
            support_value = str(support.get("value") or "") if support else ""
            other_actor_ids = subject_ids - {subject_id}
            remains_supported_by_actor = any(
                re.search(rf"(?<![\w]){re.escape(other_id)}(?![\w])", support_value, re.IGNORECASE)
                for other_id in other_actor_ids
            )
            seated = bool(posture and re.search(
                r"\b(?:seated|sitting|astride|mounted|riding)\b",
                str(posture.get("value") or ""), re.IGNORECASE,
            ))
            if support is None or remains_supported_by_actor or seated:
                issues.append(
                    f"{label} explicit named-actor dismount cue conflicts with closing "
                    f"support/posture for {subject_id!r}"
                )
    return issues


def _event_transition_cue_issues(
    row: Mapping[str, Any],
    state: Mapping[str, Any],
    events: Sequence[tuple[str | None, str]],
    label: str,
) -> tuple[list[str], set[tuple[str, str, str]]]:
    if not isinstance(state, Mapping):
        return [], set()
    event_text = {event_id: text for event_id, text in events if event_id}
    event_ids = row.get("_director_transition_event_ids")
    if not isinstance(event_ids, list):
        event_ids = []
    transitions = state.get("transitions", [])
    issues: list[str] = []
    established_keys: set[tuple[str, str, str]] = set()
    for index, transition in enumerate(transitions):
        event_id = event_ids[index] if index < len(event_ids) else None
        if transition.get("operation") == "establish":
            persisted_event_id = transition.get("event_id")
            if not isinstance(persisted_event_id, str) or not persisted_event_id:
                issues.append(
                    f"{label} establishment transition {index + 1} lacks exact source-event provenance"
                )
                continue
            if isinstance(event_id, str) and event_id != persisted_event_id:
                issues.append(
                    f"{label} establishment transition {index + 1} event_id conflicts "
                    "with its event-routing provenance"
                )
                continue
            event_text_value = event_text.get(persisted_event_id)
            if event_text_value is None:
                issues.append(
                    f"{label} establishment transition {index + 1} references missing or "
                    f"unassigned source event {persisted_event_id!r}"
                )
                continue
            if not _establishment_source_supports(row, transition, event_text_value):
                issues.append(
                    f"{label} establishment transition {index + 1} is not directly supported "
                    f"for {transition.get('subject_id')} {transition.get('predicate')} "
                    f"by exact source event {persisted_event_id!r}"
                )
                continue
            established_keys.add((
                persisted_event_id,
                _text_key(str(transition.get("subject_id") or "")),
                str(transition.get("predicate") or ""),
            ))
            continue
        if isinstance(event_id, str) and event_id in event_text:
            candidate_events = [(event_id, event_text[event_id])]
        elif len(events) == 1:
            # Full-ledger physical states have no compact-proposal sidecar. A
            # sole row-assigned event is still an exact source for validation.
            candidate_events = list(events)
        else:
            # With multiple assigned events and no event association, validate
            # against the set without guessing which event owns the transition.
            candidate_events = list(events)
        subject_id = str(transition.get("subject_id") or "")
        predicate = str(transition.get("predicate") or "")
        action = str(transition.get("action") or "")
        target = str(transition.get("to_value") or "")
        action_claims_shoulder_lowering = bool(_LOWERING_SHOULDER_RE.search(action))
        target_claims_lowering = predicate == "posture" and bool(_LOWERING_POSTURE_RE.search(target))
        if not action_claims_shoulder_lowering and not target_claims_lowering:
            continue
        # Shoulder lowering is the narrow posture event observed in the v8 failure.
        exact_shoulder_cue = any(
            (subject_id, "lower_shoulder") in {
                (cue_subject_id, family)
                for cue_subject_id, family, _ in _named_source_actor_cues(row, text)
            }
            for _, text in candidate_events
        )
        if predicate != "posture" or not exact_shoulder_cue or not action_claims_shoulder_lowering:
            source_label = (
                f"exact assigned source event {candidate_events[0][0]!r}"
                if len(candidate_events) == 1
                else "the assigned source events"
            )
            issues.append(
                f"{label} transition {index + 1} lowering posture/action is not supported by "
                f"{source_label}; use an event-matched shoulder-lowering transition"
            )
    return issues, established_keys


def _explicit_source_physical_relations(
    text: str,
) -> list[tuple[str, frozenset[str], str, int]]:
    relations: list[tuple[str, frozenset[str], str, int]] = []
    for kind, required_predicates, pattern in _SOURCE_PHYSICAL_RELATION_PATTERNS:
        for match in pattern.finditer(text):
            if _NEGATED_SOURCE_RELATION_RE.search(text[max(0, match.start() - 48):match.start()]):
                continue
            if kind == "carrying" and _ABSTRACT_CARRY_TARGET_RE.match(
                text[match.end():match.end() + 100]
            ):
                continue
            relation = (kind, required_predicates, match.group(0), match.start())
            if relation not in relations:
                relations.append(relation)
    return relations



_SOURCE_RELATION_UNRESOLVED_TARGET_RE = re.compile(
    r"^\s*(?:it|its|this|that|these|those|someone|somebody|something|"
    r"him|her|them|their|his|hers|theirs)\b",
    re.IGNORECASE,
)
_SOURCE_RELATION_TRAILING_MANNER_RE = re.compile(
    r"\s+(?:securely|firmly|tightly|carefully|gently|lightly|loosely|"
    r"deliberately|briefly|softly|quietly|quickly|slowly)\s*$",
    re.IGNORECASE,
)


def _source_relation_owner(
    row: Mapping[str, Any], text: str, cue_position: int,
) -> str | None:
    """Resolve only a unique visible cast actor directly owning the source cue."""

    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        return None
    owners = {
        str(subject.get("character_id") or "").strip()
        for subject in subjects
        if isinstance(subject, Mapping)
        and str(subject.get("character_id") or "").strip()
        and _establishment_owner_is_direct(
            row, text, cue_position, str(subject.get("character_id") or "").strip(),
        )
    }
    return next(iter(owners)) if len(owners) == 1 else None


def _source_relation_target(
    row: Mapping[str, Any],
    text: str,
    kind: str,
    phrase: str,
    cue_position: int,
) -> tuple[frozenset[str], str] | None:
    """Resolve the cue's immediate target as one cast actor or concrete noun."""

    pattern = next((
        candidate for pattern_kind, _, candidate in _SOURCE_PHYSICAL_RELATION_PATTERNS
        if pattern_kind == kind and (
            match := candidate.match(text, cue_position)
        ) and match.group(0).casefold() == phrase.casefold()
    ), None)
    if pattern is None:
        return None
    match = pattern.match(text, cue_position)
    if match is None:
        return None

    # Some cues embed the target inside the cue match rather than after it.
    if kind == "support" and re.search(r"\bdown\b", phrase, re.IGNORECASE):
        target_match = re.search(
            r"\bdown\s+(?:(?P<possessive>[A-Za-z][\w'-]*)'s\s+)?"
            r"(?:the\s+)?(?P<body>shoulder|back|side)\b",
            phrase,
            re.IGNORECASE,
        )
        if target_match:
            target_text = " ".join(
                part for part in (
                    target_match.group("possessive"), target_match.group("body"),
                ) if part
            )
        else:
            target_text = text[match.end():]
    elif kind == "support" and re.search(
        r"\bsupported\s+by\b", phrase, re.IGNORECASE,
    ):
        target_text = re.split(
            r"\bsupported\s+by\b", phrase, maxsplit=1, flags=re.IGNORECASE,
        )[-1]
    elif kind == "contact":
        verb = re.match(
            r"\s*(?:touch(?:es|ed|ing)?|pat(?:s|ted|ting)?|"
            r"grip(?:s|ped|ping)?|use(?:s|d|ing)?|hold(?:s|ing|held)?)\b",
            phrase,
            re.IGNORECASE,
        )
        target_text = (
            phrase[verb.end():] + text[match.end():]
            if verb else text[match.end():]
        )
    else:
        target_text = text[match.end():]

    target_text = _establishment_object_text(target_text, 0).strip()
    if not target_text or _SOURCE_RELATION_UNRESOLVED_TARGET_RE.match(target_text):
        return None
    target_text = _SOURCE_RELATION_TRAILING_MANNER_RE.sub("", target_text).strip()
    target_ids = _target_character_ids(target_text, row)
    if len(target_ids) > 1:
        return None
    if target_ids:
        return frozenset(target_ids), ""

    terms = _establishment_target_terms(target_text, row)
    if not terms:
        return None
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z][\w'-]*", target_text)
        if token.casefold() in terms
    ]
    return (frozenset(), tokens[-1]) if tokens else None


def _source_relation_value_matches(
    value: Any,
    row: Mapping[str, Any],
    target_ids: frozenset[str],
    target_anchor: str,
) -> bool:
    value_text = str(value or "").strip()
    if target_ids:
        return _target_character_ids(value_text, row) == set(target_ids)
    return bool(target_anchor) and target_anchor in _establishment_target_terms(value_text, row)


def _is_pre_relation_opening(kind: str, phrase: str, predicate: str, value: Any) -> bool:
    value_text = str(value or "").strip().casefold()
    if predicate == "contact" and kind in {"contact", "carrying"}:
        return value_text in {"none", "no contact"}
    if predicate == "support" and kind == "support" and re.search(
        r"\b(?:climb(?:s|ed|ing)?\s+(?:aboard|on|onto|astride)|"
        r"mount(?:s|ed|ing)?)\b",
        phrase,
        re.IGNORECASE,
    ):
        return value_text == "none" or bool(re.fullmatch(
            r"(?:(?:standing|stood|sitting|seated|resting)\s+)?"
            r"(?:(?:on|upon)\s+(?:the\s+)?)?"
            r"(?:ground|floor|earth|soil|pavement|grounded)",
            value_text,
            re.IGNORECASE,
        ))
    return False


def _source_relation_transition_proves_target(
    row: Mapping[str, Any],
    state: Mapping[str, Any],
    events: Sequence[tuple[str | None, str]],
    event_id: str | None,
    event_text: str,
    kind: str,
    owner_id: str,
    predicates: frozenset[str],
    target_ids: frozenset[str],
    target_anchor: str,
    *,
    allow_ordinary: bool = True,
) -> bool:
    sidecar = row.get("_director_transition_event_ids")
    sidecar = sidecar if isinstance(sidecar, list) else []
    for index, transition in enumerate(state.get("transitions", [])):
        if _text_key(str(transition.get("subject_id") or "")) != _text_key(owner_id):
            continue
        if transition.get("predicate") not in predicates or not _source_relation_value_matches(
            transition.get("to_value"), row, target_ids, target_anchor,
        ):
            continue
        if transition.get("operation") == "establish":
            if event_id and transition.get("event_id") == event_id and _establishment_source_supports(
                row, transition, event_text,
            ):
                return True
            continue
        if not allow_ordinary:
            continue

        transition_event_id = sidecar[index] if index < len(sidecar) else None
        if event_id and transition_event_id == event_id:
            return True
        # Legacy ledgers without a sidecar can bind only to a sole assigned event.
        if (
            transition_event_id in (None, "")
            and len(events) == 1
            and events[0] == (event_id, event_text)
        ):
            return True
    return False


def _source_relation_owner_target_issues(
    row: Mapping[str, Any],
    state: Mapping[str, Any],
    events: Sequence[tuple[str | None, str]],
    relations: Sequence[tuple[str | None, str, frozenset[str], str, int]],
    label: str,
) -> list[str]:
    issues: list[str] = []
    for event_id, kind, predicates, phrase, position in relations:
        if not (predicates & {"support", "contact"}):
            continue
        event_text = next(
            (text for candidate_id, text in events if candidate_id == event_id),
            "",
        ) if event_id else next(
            (text for text_id, text in events if text_id is None and text),
            "",
        )
        owner_id = _source_relation_owner(row, event_text, position)
        if not owner_id:
            continue
        resolved_target = _source_relation_target(
            row, event_text, kind, phrase, position,
        )
        if resolved_target is None:
            continue
        target_ids, target_anchor = resolved_target

        owner_values = [
            fact["value"]
            for endpoint in ("opening", "closing")
            for fact in state[endpoint]
            if fact["predicate"] in predicates
            and _text_key(fact["subject_id"]) == _text_key(owner_id)
        ]
        owner_values.extend(
            value
            for transition in state["transitions"]
            if transition["predicate"] in predicates
            and _text_key(transition["subject_id"]) == _text_key(owner_id)
            for value in (transition.get("from_value"), transition.get("to_value"))
            if value is not None
        )
        if not owner_values:
            issues.append(
                f"{label} source actor {owner_id!r} must own the explicit "
                f"{kind} relation in physical_state for {phrase!r}"
            )
            continue

        # Multi-event clips have no safe endpoint-to-event mapping here. Keep
        # the owner requirement, and require the source target somewhere in
        # that actor's declared trajectory without guessing a dismount order.
        if len(events) != 1:
            if not any(
                _source_relation_value_matches(value, row, target_ids, target_anchor)
                for value in owner_values
            ):
                issues.append(
                    f"{label} source action {phrase!r} target does not match any "
                    f"{kind} state owned by source actor {owner_id!r}"
                )
            continue

        for endpoint in ("opening", "closing"):
            endpoint_facts = [
                fact for fact in state[endpoint]
                if fact["predicate"] in predicates
            ]
            owner_facts = [
                fact for fact in endpoint_facts
                if _text_key(fact["subject_id"]) == _text_key(owner_id)
            ]
            if not endpoint_facts:
                # The existing predicate-presence guard reports this case.
                continue
            if not owner_facts:
                if endpoint == "opening" and _source_relation_transition_proves_target(
                    row, state, events, event_id, event_text, kind, owner_id,
                    predicates, target_ids, target_anchor, allow_ordinary=False,
                ):
                    continue
                issues.append(
                    f"{label} source actor {owner_id!r} must own the explicit "
                    f"{kind} relation in physical_state.{endpoint} for {phrase!r}"
                )
                continue

            if endpoint == "closing" and kind == "support" and re.search(
                r"\bslide(?:s|d|ing)\s+down\b", phrase, re.IGNORECASE,
            ):
                # This explicit departure cue names the support being left.
                # Require its owner at both endpoints and its target at the
                # opening, without treating the old support as the destination.
                continue
            if (
                endpoint == "opening"
                and any(
                    _is_pre_relation_opening(
                        kind, phrase, fact["predicate"], fact["value"],
                    ) for fact in owner_facts
                )
                and _source_relation_transition_proves_target(
                    row, state, events, event_id, event_text, kind, owner_id,
                    predicates, target_ids, target_anchor,
                )
            ):
                continue
            if any(
                _source_relation_value_matches(
                    fact["value"], row, target_ids, target_anchor,
                )
                for fact in owner_facts
            ):
                continue
            issues.append(
                f"{label} source action {phrase!r} target does not match the "
                f"{kind} fact owned by source actor {owner_id!r} in "
                f"physical_state.{endpoint}"
            )
    return issues

def source_physical_state_issues(
    rows: Sequence[Any],
    source_events_by_row: Sequence[Any],
) -> list[str]:
    """Require typed endpoint relations for explicit, row-assigned source actions.

    ``source_events_by_row`` is aligned with ``rows``. Each entry may be one
    source event string or a sequence of event strings/objects with ``text``.
    This deliberately checks source wording directly, without requiring the
    generated row to repeat the relation first.
    """

    if not isinstance(source_events_by_row, Sequence) or isinstance(
        source_events_by_row, (str, bytes),
    ):
        return []
    issues: list[str] = []
    for index, row in enumerate(rows or []):
        if not isinstance(row, Mapping) or index >= len(source_events_by_row):
            continue
        events = _source_event_entries(source_events_by_row[index])
        relations = [
            (event_id, *relation)
            for event_id, event_text in events
            for relation in _explicit_source_physical_relations(event_text)
        ]
        state_value = physical_state_from_row(row)
        try:
            state = normalize_physical_state(state_value)
        except ValueError:
            # The sequence validator reports malformed state independently.
            continue
        label = _state_label(row, index)
        if state is None:
            if relations:
                issues.append(
                    f"{label} has an assigned source action with an explicit physical "
                    "relation but no physical_state; declare its support/contact at "
                    "the opening and closing frames"
                )
            continue
        issues.extend(_named_terminal_state_issues(row, state, events, label))
        transition_issues, established_keys = _event_transition_cue_issues(
            row, state, events, label,
        )
        issues.extend(transition_issues)
        if not relations:
            continue
        for endpoint in ("opening", "closing"):
            facts = state[endpoint]
            for event_id, kind, predicates, phrase, source_position in relations:
                if not any(fact["predicate"] in predicates for fact in facts):
                    if endpoint == "opening" and isinstance(event_id, str):
                        established_here = any(
                            established_event_id == event_id
                            and established_predicate in predicates
                            and _establishment_owner_is_direct(
                                row, next((text for candidate_id, text in events
                                           if candidate_id == event_id), ""),
                                source_position, established_subject_id,
                            )
                            for (established_event_id, established_subject_id,
                                 established_predicate) in established_keys
                        )
                        if established_here:
                            continue
                    expected = "/".join(sorted(predicates))
                    issues.append(
                        f"{label} source action {phrase!r} requires a {expected} "
                        f"fact in physical_state.{endpoint}"
                    )
                    break
        issues.extend(_source_relation_owner_target_issues(
            row,
            state,
            events,
            relations,
            label,
        ))
    return issues


def physical_state_subject_issues(row: Any, *, index: int = 0) -> list[str]:
    """Reject physical-state owners that are not stable visible actor IDs."""

    if not isinstance(row, Mapping):
        return []
    if "subjects_on_screen" not in row:
        return []
    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        subjects = []
    stable_ids: set[str] = set()
    for subject in subjects:
        if not isinstance(subject, Mapping):
            continue
        identifier = subject.get("character_id") or subject.get("speaker_name")
        if isinstance(identifier, str) and identifier.strip():
            stable_ids.add(_text_key(identifier))
    try:
        state = normalize_physical_state(physical_state_from_row(row))
    except ValueError:
        return []
    if state is None:
        return []
    issues: list[str] = []
    seen: set[str] = set()
    for facts in (state["opening"], state["closing"], state["transitions"]):
        for fact in facts:
            subject_id = str(fact["subject_id"])
            key = _text_key(subject_id)
            if key in stable_ids or key in seen:
                continue
            seen.add(key)
            label = _state_label(row, index)
            issues.append(
                f"{label} physical_state subject_id {subject_id!r} is not a "
                "stable visible actor ID; use subjects_on_screen.character_id "
                "exactly and describe props in an actor-owned prop fact"
            )
    return issues


def validate_physical_state_sequence(
    rows: Sequence[Any],
    *,
    initial_facts: Sequence[Mapping[str, Any]] = (),
    require_state_after_first: bool = False,
) -> list[str]:
    """Return clear issues in declared blocking and adjacent shot handoffs.

    Physical state remains opt-in: an entirely untyped legacy plan validates
    without issues. Once a caller has activated state tracking, it can set
    ``require_state_after_first`` to require a typed object in every subsequent
    continuous shot. Explicit ``independent`` and ``time_jump`` states reset
    the handoff before validating their own opening, transitions, and closing.
    """

    issues: list[str] = []
    try:
        normalized_initial = _normalize_fact_list(
            list(initial_facts or []), field="initial_facts",
        )
    except (TypeError, ValueError) as exc:
        return [f"initial physical state: {exc}"]
    current = _facts_by_key(normalized_initial)
    active = bool(current)

    for index, raw_row in enumerate(rows or []):
        if not isinstance(raw_row, Mapping):
            continue
        label = _state_label(raw_row, index)
        raw_state = physical_state_from_row(raw_row)
        if raw_state is None:
            if active and require_state_after_first:
                issues.append(
                    f"{label} is missing physical_state while a continuous "
                    "physical state is active"
                )
            continue
        try:
            state = normalize_physical_state(raw_state)
        except ValueError as exc:
            issues.append(f"{label} has invalid physical_state: {exc}")
            continue
        if state is None:
            continue
        issues.extend(physical_state_subject_issues(raw_row, index=index))
        has_declared_facts = bool(
            state["opening"] or state["transitions"] or state["closing"]
        )
        active = active or has_declared_facts

        opening = _facts_by_key(state["opening"])
        transitions = state["transitions"]
        closing = _facts_by_key(state["closing"])
        issues.extend(_support_cycle_issues(opening, label, "opening"))
        issues.extend(_support_cycle_issues(closing, label, "closing"))
        discontinuity = state["continuity"] in {"independent", "time_jump"}

        if discontinuity:
            planned_strategy = str(
                raw_row.get("continuity_strategy") or ""
            ).strip().casefold().replace("-", "_").replace(" ", "_")
            if planned_strategy and planned_strategy != "independent":
                issues.append(
                    f"{label} declares physical_state.continuity="
                    f"{state['continuity']!r}, but continuity_strategy is "
                    f"{planned_strategy!r}; a physical discontinuity requires "
                    "a planned independent cut"
                )
            current = dict(opening)
        elif current:
            for key, prior_fact in current.items():
                opening_fact = opening.get(key)
                fact_label = (
                    f"{prior_fact['subject_id']} {prior_fact['predicate']}"
                )
                if opening_fact is None:
                    issues.append(
                        f"{label} opening physical_state omits carried fact "
                        f"{fact_label!r}"
                    )
                elif _text_key(opening_fact["value"]) != _text_key(prior_fact["value"]):
                    issues.append(
                        f"{label} opening {fact_label!r} is "
                        f"{opening_fact['value']!r}, but the preceding closing "
                        f"state is {prior_fact['value']!r}"
                    )
            for key, opening_fact in opening.items():
                current.setdefault(key, opening_fact)
        else:
            current = dict(opening)

        for transition_index, transition in enumerate(transitions, start=1):
            key = (
                _text_key(transition["subject_id"]),
                transition["predicate"],
            )
            before = current.get(key)
            fact_label = (
                f"{transition['subject_id']} {transition['predicate']}"
            )
            establishment = transition.get("operation") == "establish"
            if establishment:
                provenance_issue = _establishment_sequence_source_issue(
                    raw_row, transition, label, transition_index,
                )
                if provenance_issue:
                    issues.append(provenance_issue)
                    continue
                if before is not None:
                    issues.append(
                        f"{label} transition {transition_index} for {fact_label!r} "
                        "cannot establish a fact that already exists"
                    )
                    continue
            elif before is None:
                issues.append(
                    f"{label} transition {transition_index} for {fact_label!r} "
                    "has no opening/current fact to transition from"
                )
            elif _text_key(before["value"]) != _text_key(transition["from_value"]):
                issues.append(
                    f"{label} transition {transition_index} for {fact_label!r} "
                    f"starts at {transition['from_value']!r}, but the current "
                    f"state is {before['value']!r}"
                )
            current[key] = {
                "subject_id": transition["subject_id"],
                "predicate": transition["predicate"],  # type: ignore[typeddict-item]
                "value": transition["to_value"],
            }

        expected_keys = set(current)
        closing_keys = set(closing)
        for key in sorted(expected_keys - closing_keys):
            fact = current[key]
            issues.append(
                f"{label} closing physical_state omits "
                f"{fact['subject_id']} {fact['predicate']}"
            )
        for key in sorted(closing_keys - expected_keys):
            fact = closing[key]
            issues.append(
                f"{label} closing introduces {fact['subject_id']} "
                f"{fact['predicate']} without an opening fact or transition"
            )
        for key in sorted(expected_keys & closing_keys):
            expected = current[key]
            declared = closing[key]
            if _text_key(expected["value"]) != _text_key(declared["value"]):
                issues.append(
                    f"{label} transitions produce {expected['value']!r} for "
                    f"{expected['subject_id']} {expected['predicate']}, but "
                    f"closing declares {declared['value']!r}"
                )

    return issues


def _establishment_sequence_source_issue(
    row: Mapping[str, Any], transition: Mapping[str, Any], label: str,
    transition_index: int,
) -> str | None:
    event_id = str(transition.get("event_id") or "").strip()
    if not event_id:
        return (
            f"{label} establishment transition {transition_index} lacks exact source-event provenance"
        )
    assigned_ids = row.get("source_event_ids")
    if not isinstance(assigned_ids, list) or event_id not in assigned_ids:
        return (
            f"{label} establishment transition {transition_index} event_id {event_id!r} "
            "is not assigned to this row"
        )
    contract = row.get("_director_source_contract")
    if not isinstance(contract, Mapping) or not isinstance(contract.get("events"), list):
        return (
            f"{label} establishment transition {transition_index} has no embedded immutable "
            "source contract for exact event validation"
        )
    event = next((
        item for item in contract["events"]
        if isinstance(item, Mapping) and item.get("event_id") == event_id
    ), None)
    if not isinstance(event, Mapping) or not isinstance(event.get("text"), str):
        return (
            f"{label} establishment transition {transition_index} references unknown "
            f"source event {event_id!r}"
        )
    if not _establishment_source_supports(row, transition, event["text"]):
        return (
            f"{label} establishment transition {transition_index} is not directly supported "
            f"by exact source event {event_id!r}"
        )
    return None


def _support_cycle_issues(
    facts: Mapping[tuple[str, str], Mapping[str, Any]],
    label: str,
    endpoint: str,
) -> list[str]:
    """Reject cycles in explicit actor-to-actor support relations only."""

    known_subject_ids = {
        str(fact.get("subject_id") or "").strip()
        for fact in facts.values()
        if str(fact.get("subject_id") or "").strip()
    }
    edges: dict[str, set[str]] = {}
    for fact in facts.values():
        if fact.get("predicate") != "support":
            continue
        subject_id = str(fact.get("subject_id") or "").strip()
        value = str(fact.get("value") or "")
        targets: set[str] = set()
        for candidate in known_subject_ids:
            match = re.search(
                rf"(?<![\w]){re.escape(candidate)}(?![\w])", value, re.IGNORECASE,
            )
            if match and not _NEGATED_SOURCE_RELATION_RE.search(value[:match.start()]):
                targets.add(candidate)
        if targets:
            edges[subject_id] = targets

    cycles: list[str] = []
    reported: set[frozenset[str]] = set()
    for start in edges:
        stack: list[tuple[str, tuple[str, ...]]] = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            for target in edges.get(node, set()):
                if target == start:
                    cycle = frozenset(path)
                    if cycle not in reported:
                        reported.add(cycle)
                        cycles.append(
                            f"{label} {endpoint} physical_state has a cyclic actor support relation: "
                            + " -> ".join((*path, start))
                        )
                elif target not in path and len(path) < len(edges) + 1:
                    stack.append((target, (*path, target)))
    return cycles


_ACTION_STOPWORDS = {
    "a", "an", "and", "as", "at", "away", "by", "for", "from", "in",
    "into", "it", "its", "near", "of", "off", "on", "onto", "or",
    "over", "the", "their", "then", "through", "to", "toward",
    "towards", "under", "up", "with", "while", "within", "without",
    "frame", "framing", "camera", "shot", "lens", "view", "angle",
}
_ACTION_TOKEN_ALIASES = {
    "flies": "fly", "flying": "fly", "flew": "fly", "flight": "fly",
    "rides": "ride", "riding": "ride", "rode": "ride",
    "holds": "hold", "holding": "hold", "held": "hold",
    "grips": "grip", "gripping": "grip", "gripped": "grip",
    "banks": "bank", "banking": "bank", "dived": "dive",
    "dives": "dive", "diving": "dive", "glides": "glide",
    "gliding": "glide", "approaches": "approach", "approaching": "approach",
    "walks": "walk", "walking": "walk", "runs": "run", "running": "run",
    "turns": "turn", "turning": "turn", "lands": "land", "landing": "land",
    "leaps": "leap", "leaping": "leap", "jumps": "jump", "jumping": "jump",
    "raises": "raise", "raising": "raise", "reaches": "reach", "reaching": "reach",
    "waves": "wave", "waving": "wave", "spins": "spin", "spinning": "spin",
}
_MOTIF_OR_LOOP_RE = re.compile(
    r"\b(?:loop(?:ed|ing)?|refrain|intentional motif|recurring motif|chorus motif)\b",
    re.IGNORECASE,
)


def _canonical_action_token(token: str) -> str:
    word = token.casefold()
    if word in _ACTION_TOKEN_ALIASES:
        return _ACTION_TOKEN_ALIASES[word]
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 7 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 5 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _bounded_action_signature(row: Mapping[str, Any]) -> set[str]:
    chunks: list[str] = []
    action_beats = row.get("action_beats")
    if isinstance(action_beats, list):
        chunks.extend(str(value) for value in action_beats[:4] if value)
    state = physical_state_from_row(row)
    if isinstance(state, Mapping):
        transitions = state.get("transitions")
        if isinstance(transitions, list):
            chunks.extend(
                str(item.get("action") or "")
                for item in transitions[:3]
                if isinstance(item, Mapping)
            )
    for field in ("scene_goal", "ending_beat"):
        value = row.get(field)
        if value:
            chunks.append(str(value))
    tokens: list[str] = []
    for chunk in chunks:
        for raw in re.findall(r"[a-z0-9]+", chunk.casefold()):
            token = _canonical_action_token(raw)
            if len(token) < 3 or token in _ACTION_STOPWORDS:
                continue
            if token not in tokens:
                tokens.append(token)
            if len(tokens) >= 40:
                return set(tokens)
    return set(tokens)


def physical_transition_actions_missing_from_prompt(
    value: Any,
    prompt: Any,
) -> list[str]:
    """Return transition clauses not already described by the video prompt."""

    state = normalize_physical_state(value)
    if state is None:
        return []
    prompt_signature = _bounded_action_signature({
        "action_beats": [str(prompt or "")[:1600]],
    })
    missing: list[str] = []
    for transition in state["transitions"]:
        action = transition["action"]
        clauses = [
            clause.strip()
            for clause in re.split(
                r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\bwhile\b)\s*",
                action,
                flags=re.IGNORECASE,
            )
            if clause.strip()
        ][:4]
        if not clauses:
            continue
        for clause in clauses:
            signature = _bounded_action_signature({"action_beats": [clause[:240]]})
            if not signature:
                continue
            overlap = len(signature & prompt_signature)
            minimum_overlap = min(
                len(signature), max(2, (len(signature) * 72 + 99) // 100),
            )
            if overlap < minimum_overlap:
                missing.append(clause)
    return missing


def _camera_bucket(field: str, value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    if not text:
        return ""
    rules: tuple[tuple[str, str], ...]
    if field == "framing":
        rules = (
            (r"\bextreme close\b|\bmacro\b", "extreme_close"),
            (r"\bclose\b|\btight\b", "close"),
            (r"\bmedium\b|\bwaist\b", "medium"),
            (r"\bwide\b|\bestablishing\b|\bfull body\b", "wide"),
            (r"\bover the shoulder\b", "over_shoulder"),
        )
    elif field == "angle":
        rules = (
            (r"\boverhead\b|\btop down\b|\bbird.s eye\b", "overhead"),
            (r"\blow angle\b|\bground level\b", "low"),
            (r"\bhigh angle\b", "high"),
            (r"\bdutch\b|\bcanted\b", "dutch"),
            (r"\beye level\b|\beye line\b", "eye_level"),
        )
    else:
        rules = (
            (r"\borbit\b|\borbiting\b|\barc around\b", "orbit"),
            (r"\bdolly in\b|\bpush in\b|\bpushes in\b", "push_in"),
            (r"\bdolly out\b|\bpull back\b|\bpulls back\b", "pull_back"),
            (r"\bpan\b|\bpanning\b", "pan"),
            (r"\btilt\b|\btilting\b", "tilt"),
            (r"\bhandheld\b|\bshaky\b", "handheld"),
            (r"\btrack\b|\btracking\b|\bfollow\b|\bfollowing\b", "tracking"),
            (r"\bcrane\b|\bjib\b", "crane"),
            (r"\bzoom\b|\bzooming\b", "zoom"),
            (r"\bstatic\b|\blocked off\b|\bno movement\b", "static"),
        )
    for pattern, bucket in rules:
        if re.search(pattern, text):
            return bucket
    return ""


def _has_meaningful_camera_change(
    left: Mapping[str, Any], right: Mapping[str, Any],
) -> bool:
    left_camera = left.get("camera_plan")
    right_camera = right.get("camera_plan")
    if not isinstance(left_camera, Mapping) or not isinstance(right_camera, Mapping):
        return False
    return any(
        left_bucket
        and right_bucket
        and left_bucket != right_bucket
        for field in ("framing", "angle", "movement")
        for left_bucket, right_bucket in [
            (_camera_bucket(field, left_camera.get(field)),
             _camera_bucket(field, right_camera.get(field)))
        ]
    )


def _normalized_ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        _text_key(str(item))
        for item in value[:16]
        if isinstance(item, str) and item.strip()
    }


def _is_performance_or_explicit_loop(row: Mapping[str, Any]) -> bool:
    if str(row.get("scene_type") or "").strip().casefold() == "performance":
        return True
    text_parts = [str(row.get(field) or "") for field in ("scene_goal", "ending_beat")]
    beats = row.get("action_beats")
    if isinstance(beats, list):
        text_parts.extend(str(value) for value in beats[:4] if value)
    return bool(_MOTIF_OR_LOOP_RE.search(" ".join(text_parts)))


def music_video_progression_issue(
    rows: Sequence[Any],
    *,
    sections: Sequence[Any] | None = None,
    source_event_ids_by_index: Sequence[Any] | None = None,
    lookback: int = 4,
) -> str | None:
    """Flag a run of repeated planned actions lacking event or shot variation.

    The bounded signature combines action beats, transition actions, scene
    goals, and ending beats. It ignores word order and basic verb inflections,
    compares only the most recent few rows, and honors source event IDs,
    explicit motifs/loops, performance scenes, section changes, and meaningful
    camera changes. One or two repeated clips remain valid for sustained
    moments and short lyric transitions.
    """

    if not rows:
        return None
    current_index = len(rows) - 1
    current = rows[current_index]
    if not isinstance(current, Mapping) or _is_performance_or_explicit_loop(current):
        return None
    current_signature = _bounded_action_signature(current)
    if len(current_signature) < 3:
        return None
    current_section = (
        str(sections[current_index] or "").strip().casefold()
        if sections is not None and current_index < len(sections) else ""
    )
    current_event_ids = (
        _normalized_ids(source_event_ids_by_index[current_index])
        if source_event_ids_by_index is not None
        and current_index < len(source_event_ids_by_index)
        else set()
    )
    if not current_event_ids:
        current_event_ids = _normalized_ids(current.get("source_event_ids"))
    repeated_matches = 0
    bounded_lookback = max(1, min(8, int(lookback)))
    start = max(0, current_index - bounded_lookback)
    for previous_index in range(start, current_index):
        previous = rows[previous_index]
        if not isinstance(previous, Mapping) or _is_performance_or_explicit_loop(previous):
            continue
        previous_section = (
            str(sections[previous_index] or "").strip().casefold()
            if sections is not None and previous_index < len(sections) else ""
        )
        if current_section and previous_section and current_section != previous_section:
            continue
        previous_event_ids = (
            _normalized_ids(source_event_ids_by_index[previous_index])
            if source_event_ids_by_index is not None
            and previous_index < len(source_event_ids_by_index)
            else set()
        )
        if not previous_event_ids:
            previous_event_ids = _normalized_ids(previous.get("source_event_ids"))
        if current_event_ids and previous_event_ids:
            # Shared event IDs identify a deliberately reused source event;
            # distinct IDs identify a new planned event.
            continue
        previous_signature = _bounded_action_signature(previous)
        common = current_signature & previous_signature
        union = current_signature | previous_signature
        similarity = len(common) / len(union) if union else 0.0
        if len(common) < 3 or similarity < 0.66:
            continue
        if _has_meaningful_camera_change(current, previous):
            continue
        repeated_matches += 1

    if repeated_matches < 2:
        return None
    return (
        f"clip {current_index + 1} repeats the same planned action across "
        "several adjacent clips; add a distinct story or performance beat, "
        "a meaningful camera change, or the source event/motif marker when "
        "the repetition is intentional"
    )
