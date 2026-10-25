"""Bounded patches for source-action ownership and physical-state ledgers."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any

from .physical_continuity import (
    GROUNDED_STATE_VALUES,
    MAX_PHYSICAL_FACTS,
    MAX_PHYSICAL_TEXT_CHARS,
    MAX_PHYSICAL_TRANSITIONS,
    MAX_PHYSICAL_VALUE_CHARS,
    PHYSICAL_STATE_PREDICATES,
    normalize_physical_state,
    physical_state_contract_instructions,
    physical_state_json_schema,
)

_PATCH_KEYS = {"shot_index", "source_event_ids", "physical_state"}


def _compact_row_context(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    subjects = row.get("subjects_on_screen")
    cast = []
    if isinstance(subjects, list):
        for subject in subjects:
            if not isinstance(subject, Mapping):
                continue
            cast.append({
                key: subject[key]
                for key in ("character_id", "speaker_name")
                if key in subject
            })
    raw_physical_state = row.get("physical_state")
    if raw_physical_state is None:
        physical_state_context = {
            "status": "not_supplied",
            "ledger_included": False,
        }
    else:
        try:
            normalized_physical_state = normalize_physical_state(raw_physical_state)
        except ValueError as exc:
            # Never echo malformed state back as if it were a value worth
            # preserving. Keep the concrete validation error so the repair can
            # reconstruct the ledger from the source and authored actions.
            physical_state_context = {
                "status": "invalid_excluded",
                "ledger_included": False,
                "validation_error": str(exc),
            }
        else:
            physical_state_context = {
                "status": "draft_revisable",
                "ledger_included": normalized_physical_state is not None,
                "ledger": normalized_physical_state,
            }

    return {
        "shot_index": index + 1,
        "title": row.get("title"),
        "scene_goal": row.get("scene_goal"),
        "spatial_setup": row.get("spatial_setup"),
        "continuity_strategy": row.get("continuity_strategy"),
        "source_event_ids": row.get("source_event_ids"),
        "cast": cast,
        "action_beats": row.get("action_beats"),
        "ending_beat": row.get("ending_beat"),
        "prior_physical_state_context": physical_state_context,
    }


def build_source_physical_repair_request(
    rows: list[dict],
    contract: dict,
    issues: list[str],
) -> tuple[str, str, dict[str, Any]]:
    """Build a JSON-only request that can patch IDs and physical state per row.

    The returned response intentionally contains no editable shot prose. The
    caller applies only ``source_event_ids`` and ``physical_state`` and then
    reruns its ordinary ownership and sequence validators.
    """

    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("rows must be a list of shot objects")
    if not rows:
        raise ValueError("source/physical repair requires at least one existing shot row")
    if not isinstance(contract, Mapping):
        raise ValueError("contract must be an object")
    if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
        raise ValueError("issues must be a list of strings")

    events = contract.get("events")
    if not isinstance(events, list) or any(not isinstance(event, Mapping) for event in events):
        raise ValueError("contract.events must be a list of event objects")
    event_ids = [event.get("event_id") for event in events]
    if any(not isinstance(event_id, str) or not event_id for event_id in event_ids):
        raise ValueError("every source event must have a non-empty string event_id")

    row_count = len(rows)
    patch_schema = {
        "type": "object",
        "properties": {
            "shot_index": {
                "type": "integer",
                "minimum": 1,
                "maximum": row_count,
            },
            "source_event_ids": {
                "type": "array",
                "items": (
                    {"type": "string", "enum": event_ids}
                    if event_ids else {"type": "string"}
                ),
                "uniqueItems": True,
                "maxItems": len(event_ids),
            },
            "physical_state": physical_state_json_schema(),
        },
        "required": ["shot_index", "source_event_ids", "physical_state"],
        "additionalProperties": False,
    }
    schema = {
        "type": "array",
        "items": patch_schema,
        "minItems": row_count,
        "maxItems": row_count,
    }

    system_prompt = (
        "You repair only source-event ownership and typed physical state in an "
        "existing shot plan. Return only the requested JSON array. Do not rewrite "
        "or propose shot content. The immutable source contract is authoritative. "
        "A valid prior physical-state ledger is draft context only, not a "
        "constraint, and must be revised if it conflicts with source actions or "
        "facts. An invalid prior ledger is excluded; never copy values from it.\n\n"
        + physical_state_contract_instructions()
    )
    context = {
        "immutable_source_contract": contract,
        "original_shots": [
            _compact_row_context(row, index) for index, row in enumerate(rows)
        ],
        "validation_issues": issues,
    }
    user_prompt = (
        "Return exactly one patch for every existing shot, in the same order, "
        "with one-based shot_index values 1 through " + str(row_count) + ".\n"
        "Each patch may set only source_event_ids and physical_state. Preserve "
        "the existing shot count, timing, action and ending prose, subjects, "
        "dialogue, camera, and all other shot fields unchanged.\n"
        "Allocate every immutable source event ID exactly once in chronological "
        "order. Use only IDs in immutable_source_contract; do not invent, omit, "
        "repeat, or reorder events. Keep each event with the existing shot that "
        "performs it, resolving against the original source and the supplied "
        "action/ending context. Allocation example (format only): if one shot "
        "performs consecutive events E1 and E2 and a later shot performs E3, "
        "use [\"E1\",\"E2\"] for the first row and [\"E3\"] for the later "
        "row. Include every performed event ID, not only the first or last one; "
        "use the actual IDs and actions from the supplied contract.\n"
        "Repair the physical-state ledgers for the same existing segments. "
        "Keep unchanged facts complete across continuous boundaries, copying their "
        "subject_id, predicate, and value verbatim. For every transition, copy "
        "from_value verbatim from the current fact and carry to_value verbatim "
        "into the resulting state. Put motion only in transition.action. Do not "
        "silently infer or copy a state that conflicts with an explicit action "
        "or transition; resolve the conflict using the source and original shot "
        "context. Do not add events or physical actions.\n"
        "Keep physical predicates distinct: a rider on a stationary grounded "
        "mount can have grounded='grounded' with support='astride char_1'. "
        "When that mount flies, rider and mount are grounded='airborne' while "
        "the rider keeps the same support. 'astride' or 'mounted' is not a "
        "grounded value.\n"
        "Return each physical_state with opening, transitions, and closing "
        "arrays, using the provided typed schema.\n\n"
        "REPAIR CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    return system_prompt, user_prompt, schema


def apply_source_physical_patch(
    rows: list[dict],
    patches: list[dict],
) -> list[dict]:
    """Return a deep-copied plan with only IDs and normalized states patched."""

    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("rows must be a list of shot objects")
    if not isinstance(patches, list):
        raise ValueError("patches must be an array")
    if len(patches) != len(rows):
        raise ValueError(
            f"patch count must match existing shot count ({len(rows)}); got {len(patches)}"
        )

    expected_indexes = list(range(1, len(rows) + 1))
    normalized_patches: list[tuple[list[str], dict[str, Any]]] = []
    for offset, patch in enumerate(patches):
        if not isinstance(patch, Mapping):
            raise ValueError(f"patch {offset + 1} must be an object")
        extra = set(patch) - _PATCH_KEYS
        missing = _PATCH_KEYS - set(patch)
        if extra:
            raise ValueError(
                f"patch {offset + 1} has unsupported keys: {', '.join(sorted(extra))}"
            )
        if missing:
            raise ValueError(
                f"patch {offset + 1} is missing required keys: {', '.join(sorted(missing))}"
            )

        shot_index = patch.get("shot_index")
        if isinstance(shot_index, bool) or not isinstance(shot_index, int):
            raise ValueError(f"patch {offset + 1} shot_index must be an integer")
        if shot_index != expected_indexes[offset]:
            raise ValueError(
                "patch shot_index values must be unique, complete, and in existing "
                f"row order 1..{len(rows)}; patch {offset + 1} has index {shot_index}"
            )

        source_event_ids = patch.get("source_event_ids")
        if not isinstance(source_event_ids, list) or any(
            not isinstance(event_id, str) for event_id in source_event_ids
        ):
            raise ValueError(
                f"patch {offset + 1} source_event_ids must be an array of strings"
            )

        physical_state = patch.get("physical_state")
        if physical_state is None:
            raise ValueError(f"patch {offset + 1} physical_state must be an object")
        try:
            normalized_state = normalize_physical_state(physical_state)
        except ValueError as exc:
            raise ValueError(f"patch {offset + 1} has invalid physical_state: {exc}") from exc
        if normalized_state is None:
            raise ValueError(f"patch {offset + 1} physical_state must be an object")
        normalized_patches.append((list(source_event_ids), normalized_state))

    patched_rows = copy.deepcopy(rows)
    for row, (source_event_ids, physical_state) in zip(patched_rows, normalized_patches):
        row["source_event_ids"] = source_event_ids
        row["physical_state"] = physical_state
    return patched_rows


def _compact_fact_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string", "minLength": 1, "maxLength": 80},
            "predicate": {
                "type": "string",
                "enum": list(PHYSICAL_STATE_PREDICATES),
            },
            "value": {"type": "string", "minLength": 1, "maxLength": MAX_PHYSICAL_VALUE_CHARS},
        },
        "required": ["subject_id", "predicate", "value"],
        "additionalProperties": False,
    }


def _compact_initial_facts_schema(
    known_cast_ids: list[str],
    *,
    use_object: bool,
) -> dict[str, Any]:
    if not use_object or not 1 <= len(known_cast_ids) <= 4:
        return {
            "type": "array",
            "maxItems": MAX_PHYSICAL_FACTS,
            "items": _compact_fact_schema(),
        }

    allowed_predicates = (
        tuple(PHYSICAL_STATE_PREDICATES)
        if len(known_cast_ids) <= 3
        else ("support", "contact", "grounded")
    )
    character_schema = {
        "type": "object",
        "properties": {
            predicate: {
                "type": "string",
                "maxLength": MAX_PHYSICAL_VALUE_CHARS,
                **({"enum": list(GROUNDED_STATE_VALUES)} if predicate == "grounded" else {}),
            }
            for predicate in allowed_predicates
        },
        "required": [],
        "maxProperties": len(allowed_predicates),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            character_id: character_schema for character_id in known_cast_ids
        },
        "required": list(known_cast_ids),
        "maxProperties": len(known_cast_ids),
        "additionalProperties": False,
    }


def _compact_transition_schema(
    event_ids: list[str],
    *,
    include_event_id: bool = True,
) -> dict[str, Any]:
    properties = {
        "subject_id": {"type": "string", "minLength": 1, "maxLength": 80},
        "predicate": {
            "type": "string",
            "enum": list(PHYSICAL_STATE_PREDICATES),
        },
        "to_value": {"type": "string", "minLength": 1, "maxLength": MAX_PHYSICAL_VALUE_CHARS},
        "action": {"type": "string", "minLength": 1, "maxLength": MAX_PHYSICAL_TEXT_CHARS},
    }
    required = ["subject_id", "predicate", "to_value", "action"]
    if not include_event_id:
        # Only immutable event-keyed proposals have enough provenance to add a
        # fact at the event that introduces it.
        properties["operation"] = {"type": "string", "enum": ["establish"]}
    if include_event_id:
        properties = {
            "event_id": {"type": "string", "enum": list(event_ids)},
            **properties,
        }
        required.insert(0, "event_id")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _compact_initial_facts_instruction(
    known_cast_ids: list[str],
    initial_facts_schema: Mapping[str, Any],
) -> str:
    if initial_facts_schema.get("type") != "object":
        return (
            "Use the legacy array of fact objects, with at most "
            f"{MAX_PHYSICAL_FACTS} total facts. Include every needed initial "
            "subject_id/predicate key only when the opening fact is source-supported. "
            "Omit unknown opening facts. Use only known cast "
            "character IDs where available; props belong in actor-owned prop facts."
        )

    if len(known_cast_ids) <= 2:
        return (
            "Use an object keyed by exactly these cast character IDs: "
            + ", ".join(known_cast_ids)
            + ". Each character object is sparse: include only opening facts "
            "supported by the source/context and needed to replay the assigned "
            "events. Omit unknown predicates; never use 'unknown' or "
            "'unspecified'. Grounded, when included, must be exactly 'grounded' "
            "or 'airborne' and supported by context. The 12-fact total cap is strict."
        )

    optional = (
        " Location, posture, and prop are optional only when there is room under "
        f"the strict {MAX_PHYSICAL_FACTS}-fact total cap."
        if len(known_cast_ids) == 3
        else " No optional predicates fit under the strict fact cap."
    )
    return (
        "Use an object keyed by exactly these cast character IDs: "
        + ", ".join(known_cast_ids)
        + ". Each character object is sparse: include only opening facts supported "
        "by the source/context and needed to replay the assigned events. Omit "
        "unknown predicates; never use 'unknown' or 'unspecified'. Grounded, "
        "when included, must be exactly 'grounded' or 'airborne' and supported by context."
        + optional
        + " Do not exceed the strict 12-fact total cap."
    )


def _contract_event_ids(contract: Mapping[str, Any]) -> list[str]:
    events = contract.get("events")
    if not isinstance(events, list) or any(not isinstance(event, Mapping) for event in events):
        raise ValueError("contract.events must be a list of event objects")
    event_ids = [event.get("event_id") for event in events]
    if any(not isinstance(event_id, str) or not event_id for event_id in event_ids):
        raise ValueError("every source event must have a non-empty string event_id")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("contract.events must have unique event_id values")
    return event_ids


def _ordered_compact_event_items(
    event_shots: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> list[tuple[str, Any]]:
    """Recover authoritative event order when the source contract is present.

    The fixed-shot planner normally passes raw rows before attaching the
    contract. In that case source contracts use generated E1..En IDs, so their
    numeric suffix gives the same order. For other IDs, JSON object insertion
    order is retained and the later source-contract validator remains the
    authority for exact membership/order.
    """

    embedded_contracts = [
        row.get("_director_source_contract")
        for row in rows
        if isinstance(row.get("_director_source_contract"), Mapping)
    ]
    expected_event_ids: list[str] | None = None
    if embedded_contracts:
        expected_event_ids = _contract_event_ids(embedded_contracts[0])
        for embedded in embedded_contracts[1:]:
            if _contract_event_ids(embedded) != expected_event_ids:
                raise ValueError("shot rows contain conflicting source contracts")
        if set(event_shots) != set(expected_event_ids):
            raise ValueError(
                "event_shots keys must exactly match the immutable source contract event IDs"
            )
        return [(event_id, event_shots[event_id]) for event_id in expected_event_ids]

    if all(re.fullmatch(r"E\d+", event_id) for event_id in event_shots):
        ordered = sorted(event_shots.items(), key=lambda item: int(item[0][1:]))
        numeric_ids = [int(event_id[1:]) for event_id, _ in ordered]
        if numeric_ids != list(range(1, len(numeric_ids) + 1)):
            raise ValueError("event_shots must contain every generated source event ID E1..En")
        return ordered
    return list(event_shots.items())


def _known_cast_ids(rows: list[Mapping[str, Any]]) -> set[str]:
    known: set[str] = set()
    for row in rows:
        subjects = row.get("subjects_on_screen")
        if not isinstance(subjects, list):
            continue
        for subject in subjects:
            if not isinstance(subject, Mapping):
                continue
            character_id = subject.get("character_id")
            if isinstance(character_id, str) and character_id.strip():
                known.add(character_id.strip())
    return known


def _state_key(subject_id: str, predicate: str) -> tuple[str, str]:
    return (" ".join(subject_id.split()).casefold(), predicate)


def _normalized_prior_continuity(row: Mapping[str, Any]) -> str:
    try:
        state = normalize_physical_state(row.get("physical_state"))
    except ValueError:
        return "continuous"
    return state["continuity"] if state is not None else "continuous"


def _normalize_compact_initial_facts(
    raw_facts: Any,
    *,
    known_cast_ids: set[str],
) -> list[dict[str, str]]:
    if isinstance(raw_facts, Mapping):
        ordered_cast_ids = sorted(known_cast_ids)
        if not 1 <= len(ordered_cast_ids) <= 4:
            raise ValueError(
                "initial_facts object requires one to four known cast character IDs"
            )
        if any(not isinstance(character_id, str) for character_id in raw_facts):
            raise ValueError("initial_facts object keys must be character ID strings")
        supplied_cast_ids = set(raw_facts)
        expected_cast_ids = set(ordered_cast_ids)
        if supplied_cast_ids != expected_cast_ids:
            raise ValueError(
                "initial_facts object keys must exactly match known cast character IDs; "
                f"missing={sorted(expected_cast_ids - supplied_cast_ids)}, "
                f"extra={sorted(supplied_cast_ids - expected_cast_ids)}"
            )

        core_predicates = ("support", "contact", "grounded")
        allowed_predicates = (
            set(PHYSICAL_STATE_PREDICATES)
            if len(ordered_cast_ids) <= 3
            else set(core_predicates)
        )
        flattened_facts: list[dict[str, Any]] = []
        for character_id in ordered_cast_ids:
            state = raw_facts[character_id]
            if not isinstance(state, Mapping):
                raise TypeError(f"initial_facts[{character_id!r}] must be an object")
            supplied_predicates = set(state)
            extra_predicates = supplied_predicates - allowed_predicates
            if extra_predicates:
                raise ValueError(
                    f"initial_facts[{character_id!r}] has unsupported predicates: "
                    + ", ".join(sorted(map(str, extra_predicates)))
                )
            for predicate in PHYSICAL_STATE_PREDICATES:
                if predicate in state:
                    flattened_facts.append({
                        "subject_id": character_id,
                        "predicate": predicate,
                        "value": state[predicate],
                    })
        raw_facts = flattened_facts
    elif not isinstance(raw_facts, list):
        raise TypeError("initial_facts must be an array or compact character-state object")
    if len(raw_facts) > MAX_PHYSICAL_FACTS:
        raise ValueError(
            f"initial_facts has more than {MAX_PHYSICAL_FACTS} facts"
        )
    seen: set[tuple[str, str]] = set()
    for index, raw_fact in enumerate(raw_facts):
        if not isinstance(raw_fact, Mapping):
            raise ValueError(f"initial_facts[{index}] must be an object")
        extra = set(raw_fact) - {"subject_id", "predicate", "value"}
        missing = {"subject_id", "predicate", "value"} - set(raw_fact)
        if extra:
            raise ValueError(
                f"initial_facts[{index}] has unsupported keys: {', '.join(sorted(extra))}"
            )
        if missing:
            raise ValueError(
                f"initial_facts[{index}] is missing required keys: "
                f"{', '.join(sorted(missing))}"
            )
        subject_id = raw_fact.get("subject_id")
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError(f"initial_facts[{index}].subject_id must be a non-empty string")
        if known_cast_ids and subject_id.strip() not in known_cast_ids:
            raise ValueError(
                f"initial_facts[{index}] subject_id {subject_id.strip()!r} "
                "is not a known cast character_id"
            )
        predicate = raw_fact.get("predicate")
        if not isinstance(predicate, str):
            raise ValueError(f"initial_facts[{index}].predicate must be a string")
        normalized_predicate = predicate.strip().casefold().replace("-", "_").replace(" ", "_")
        if normalized_predicate not in PHYSICAL_STATE_PREDICATES:
            raise ValueError(
                f"initial_facts[{index}] has unsupported predicate "
                f"{normalized_predicate!r}"
            )
        key = _state_key(subject_id.strip(), normalized_predicate)
        if key in seen:
            raise ValueError(
                "initial_facts contains a duplicate subject_id/predicate key: "
                f"{subject_id.strip()} {normalized_predicate}"
            )
        seen.add(key)

    normalized = normalize_physical_state({
        "opening": raw_facts,
        "transitions": [],
        "closing": raw_facts,
    })
    if normalized is None:  # pragma: no cover - the explicit object above cannot be null
        raise ValueError("initial_facts must be an array")
    return normalized["opening"]


def build_compact_source_physical_repair_request(
    rows: list[dict],
    contract: dict,
    issues: list[str],
) -> tuple[str, str, dict[str, Any]]:
    """Build a fixed-shot request with compact proposals and deterministic replay.

    Unlike the legacy row patch, this wire format asks for one canonical initial
    ledger and transitions only. Opening and closing states are assembled in
    code so the model cannot paraphrase or inconsistently repeat carried facts.
    """

    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("rows must be a list of shot objects")
    if not rows:
        raise ValueError("source/physical repair requires at least one existing shot row")
    if not isinstance(contract, Mapping):
        raise ValueError("contract must be an object")
    if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
        raise ValueError("issues must be a list of strings")

    event_ids = _contract_event_ids(contract)
    row_count = len(rows)
    known_cast_ids = sorted(_known_cast_ids(rows))
    anchors = ordered_source_assignments(rows, contract)
    initial_facts_schema = _compact_initial_facts_schema(
        known_cast_ids,
        use_object=bool(event_ids),
    )
    event_transitions_schema = {
        "type": "object",
        "properties": {
            event_id: {
                "type": "array",
                "maxItems": MAX_PHYSICAL_TRANSITIONS,
                "items": _compact_transition_schema(
                    event_ids,
                    include_event_id=False,
                ),
            }
            for event_id in event_ids
        },
        "required": list(event_ids),
        "maxProperties": len(event_ids),
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "event_shots": {
                "type": "object",
                "properties": {
                    event_id: {
                        "type": "integer",
                        "minimum": max([1] + [shot for key, shot in anchors.items()
                                               if event_ids.index(key) <= event_ids.index(event_id)]),
                        "maximum": min([row_count] + [shot for key, shot in anchors.items()
                                                     if event_ids.index(key) >= event_ids.index(event_id)]),
                    }
                    for event_id in event_ids
                },
                "required": event_ids,
                "additionalProperties": False,
            },
            "initial_facts": initial_facts_schema,
            "event_transitions": event_transitions_schema,
        },
        "required": ["event_shots", "initial_facts", "event_transitions"],
        "additionalProperties": False,
    }

    system_prompt = (
        "Repair only source-event allocation and typed physical state for the "
        "existing fixed shot rows. Return only the requested JSON object. Do not "
        "rewrite or propose shot content. The immutable source contract and this "
        "response's event_shots allocation are the only action authority. Existing "
        "shot prose and prior source_event_ids may be stale draft context and must "
        "never move a state change to a different event's shot. Valid prior physical "
        "ledgers are revisable draft context; invalid prior ledgers are excluded "
        "and must never be copied. Allocate source events to existing shots, then "
        "describe physical state only with initial facts and transitions keyed by "
        "their immutable source event. "
        "Do not propose opening or closing arrays; the caller derives those by "
        "replaying the transitions. Never invent a missing fact, state value, or "
        "transition action. A fact first established by an exact later source event "
        "must remain absent from initial_facts and may be added only through an "
        "event-keyed operation='establish' transition with exact event provenance "
        "and direct source support. Never seed a later fact into the opening.\n\n"
        "PHYSICAL STATE CONTRACT:\n"
        "- Use only these predicates: "
        + ", ".join(PHYSICAL_STATE_PREDICATES)
        + ". subject_id must be an exact stable character_id from the cast when "
        "those IDs are available. Props belong in actor-owned prop facts, never as "
        "cast entities.\n"
        "- Keep fact values compact state atoms of at most "
        + str(MAX_PHYSICAL_VALUE_CHARS)
        + " characters and transition actions concise. Put narrative motion only "
        "in action; do not place event narration in a fact value.\n"
        "- Keep support, contact, posture, location, grounded, and prop distinct. "
        "For grounded, the only valid values are exactly 'grounded' or 'airborne'. "
        "'astride' and 'mounted' are support descriptions, never grounded values.\n"
        "- Facts flow from initial_facts through each ordered transition. A "
        "transition's from_value is copied from the current fact and its to_value "
        "becomes the next current fact. The caller derives complete row openings "
        "and closings; do not output either array or from_value.\n"
    )
    context = {
        "immutable_source_contract": contract,
        "preserve_ordered_assignments": anchors,
        "missing_event_ids": [event_id for event_id in event_ids if event_id not in anchors],
        "original_shots": [
            _compact_row_context(row, index) for index, row in enumerate(rows)
        ],
        "validation_issues": issues,
    }
    initial_facts_instruction = _compact_initial_facts_instruction(
        known_cast_ids,
        initial_facts_schema,
    )
    user_prompt = (
        "Return one JSON object with exactly event_shots, initial_facts, and "
        "event_transitions. event_shots must contain every immutable event_id "
        "exactly once as a key, in the same key order as immutable_source_contract.events; "
        "each value is the one-based existing shot that performs that event. In "
        "source order, shot numbers must be nondecreasing. Every event belongs to "
        "exactly one row; consecutive events may share a row. Example only: "
        "{\"E1\":1,\"E2\":1,\"E3\":2}. Use the actual event IDs from the "
        "contract. Do not create rows.\n"
        "Keep preserve_ordered_assignments unchanged. Correct missing_event_ids "
        "within the schema's shot bounds; those bounds preserve the existing "
        "unique chronological assignments. If no supported allocation fits, "
        "do not invent actions or change the accepted shots.\n"
        "event_transitions must be an object with exactly one key for every "
        "immutable event_id, including events with no state changes (use an empty "
        "array for those events). The list under an event_id contains only ordered "
        "physical changes caused by that event; do not put event_id or shot_index "
        "inside a transition. The caller routes each event's list to its one "
        "assigned shot using event_shots and applies events in immutable source "
        "order, so never duplicate a change in another event or shot. Do not move "
        "a change because old draft action_beats, ending_beat, or source_event_ids "
        "say it belongs elsewhere; those fields may be stale. The combined changes "
        "routed to any one shot must not exceed "
        + str(MAX_PHYSICAL_TRANSITIONS)
        + " transitions.\n"
        "initial_facts is the complete canonical fact set at the first shot's "
        "opening. "
        + initial_facts_instruction
        + "\n"
        "For each event_transitions entry, provide an ordered array of explicit "
        "state changes with subject_id, predicate, to_value, and a short action "
        "grounded in the immutable source event named by the object key. Do not provide "
        "from_value: the caller copies it from the current fact. A transition may "
        "normally update only a subject_id/predicate key already in initial_facts. "
        "In this event-keyed form only, operation='establish' may add one absent key "
        "at that exact event when the event text explicitly supports the named actor, "
        "target relation, and action. Direct establishment proof is supported only for "
        "support, contact, and grounded predicates; leave other facts absent until "
        "they have a valid initial fact or a normal change from a known value. It "
        "cannot replace an existing key "
        "or imply an earlier state. The caller derives every complete opening "
        "and closing by carrying facts forward and replaying these transitions.\n"
        "Keep predicate values compact and literal. For predicate grounded, "
        "to_value must be exactly 'grounded' or 'airborne'; support='astride char_1' "
        "does not mean grounded. A rider and mount are both airborne while flying, "
        "even if the rider remains astride. Preserve exact action order from the "
        "source. Do not change shot count, duration, prose, subjects, dialogue, "
        "camera, or any other existing field.\n"
        "The immutable source and shot context, plus validation feedback, follow.\n\n"
        "REPAIR CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    return system_prompt, user_prompt, schema


def ordered_source_assignments(rows: list[dict], contract: dict) -> dict[str, int]:
    """Pin only a unique, chronological partial allocation; never sort a bad one."""
    expected = _contract_event_ids(contract)
    found = [(event_id, index + 1) for index, row in enumerate(rows)
             for event_id in (row.get("source_event_ids") or [])]
    ids = [event_id for event_id, _ in found]
    if (any(not isinstance(event_id, str) for event_id in ids)
            or len(ids) != len(set(ids)) or any(event_id not in expected for event_id in ids)
            or ids != sorted(ids, key=expected.index)):
        return {}
    return dict(found)


def apply_validated_compact_patch(rows: list[dict], response: dict, contract: dict) -> list[dict]:
    """Commit a working copy only after complete source and physical validation."""
    from .source_contract import source_contract_issues
    from .physical_continuity import validate_physical_state_sequence

    working = copy.deepcopy(rows)
    for row in working:
        row["_director_source_contract"] = copy.deepcopy(contract)
    candidate = apply_compact_source_physical_patch(working, response)
    anchors = ordered_source_assignments(rows, contract)
    assigned = {event_id: index + 1 for index, row in enumerate(candidate)
                for event_id in row.get("source_event_ids", [])}
    if any(assigned.get(event_id) != shot for event_id, shot in anchors.items()):
        raise ValueError("repair moved an existing unique chronological source assignment")
    issues = source_contract_issues(candidate, contract, require_ids=True)
    issues.extend(validate_physical_state_sequence(candidate))
    if issues:
        raise ValueError("; ".join(issues))
    return candidate


def apply_compact_source_physical_patch(
    rows: list[dict],
    response: dict,
) -> list[dict]:
    """Apply a compact proposal by deterministically replaying physical state.

    Only ``source_event_ids`` and ``physical_state`` are changed. Physical
    ``from_value``, opening, and closing facts are always derived from the
    canonical initial facts and ordered transitions, never trusted to the model.
    """

    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("rows must be a list of shot objects")
    if not rows:
        raise ValueError("source/physical repair requires at least one existing shot row")
    if not isinstance(response, Mapping):
        raise ValueError("compact repair response must be an object")
    common_response_keys = {"event_shots", "initial_facts"}
    transition_response_keys = {"event_transitions", "shot_transitions"}
    allowed_response_keys = common_response_keys | transition_response_keys
    extra = set(response) - allowed_response_keys
    missing = common_response_keys - set(response)
    if extra:
        raise ValueError(
            "compact repair response has unsupported keys: " + ", ".join(sorted(extra))
        )
    if missing:
        raise ValueError(
            "compact repair response is missing required keys: " + ", ".join(sorted(missing))
        )
    transition_representations = transition_response_keys.intersection(response)
    if len(transition_representations) != 1:
        raise ValueError(
            "compact repair response must provide exactly one of event_transitions "
            "or shot_transitions"
        )
    event_keyed_transitions = "event_transitions" in transition_representations

    row_count = len(rows)
    event_shots = response.get("event_shots")
    if not isinstance(event_shots, Mapping):
        raise ValueError("event_shots must be an object")
    if any(not isinstance(event_id, str) or not event_id for event_id in event_shots):
        raise ValueError("event_shots keys must be non-empty event ID strings")
    ordered_event_items = _ordered_compact_event_items(event_shots, rows)
    event_order = {
        event_id: index for index, (event_id, _) in enumerate(ordered_event_items)
    }
    prior_shot_index = 0
    allocated_ids: list[list[str]] = [[] for _ in rows]
    for event_id, shot_index in ordered_event_items:
        if isinstance(shot_index, bool) or not isinstance(shot_index, int):
            raise ValueError(f"event_shots[{event_id!r}] must be an integer shot index")
        if not 1 <= shot_index <= row_count:
            raise ValueError(
                f"event_shots[{event_id!r}] must be between 1 and {row_count}"
            )
        if shot_index < prior_shot_index:
            raise ValueError(
                "event_shots shot indexes must be nondecreasing in source event order"
            )
        prior_shot_index = shot_index
        allocated_ids[shot_index - 1].append(event_id)

    known_cast_ids = _known_cast_ids(rows)
    raw_initial_facts = response.get("initial_facts")
    legacy_array_response = (
        isinstance(raw_initial_facts, list) and not event_keyed_transitions
    )
    initial_facts = _normalize_compact_initial_facts(
        raw_initial_facts,
        known_cast_ids=known_cast_ids,
    )
    if event_keyed_transitions:
        raw_event_transitions = response.get("event_transitions")
        if not isinstance(raw_event_transitions, Mapping):
            raise ValueError("event_transitions must be an object")
        expected_event_ids = [event_id for event_id, _ in ordered_event_items]
        if set(raw_event_transitions) != set(expected_event_ids):
            raise ValueError(
                "event_transitions keys must exactly match immutable source event IDs"
            )

        routed_transitions: list[list[dict[str, Any]]] = [
            [] for _ in rows
        ]
        transition_keys_without_event = {
            "subject_id", "predicate", "to_value", "action",
        }
        event_transition_keys = transition_keys_without_event | {"operation"}
        for event_id, shot_index in ordered_event_items:
            event_entries = raw_event_transitions[event_id]
            if not isinstance(event_entries, list):
                raise ValueError(f"event_transitions[{event_id!r}] must be an array")
            if len(event_entries) > MAX_PHYSICAL_TRANSITIONS:
                raise ValueError(
                    f"event_transitions[{event_id!r}] has more than "
                    f"{MAX_PHYSICAL_TRANSITIONS} transitions"
                )
            for transition_index, raw_transition in enumerate(event_entries):
                location = f"event_transitions[{event_id!r}][{transition_index}]"
                if not isinstance(raw_transition, Mapping):
                    raise ValueError(f"{location} must be an object")
                extra = set(raw_transition) - event_transition_keys
                missing = transition_keys_without_event - set(raw_transition)
                if extra:
                    raise ValueError(
                        f"{location} has unsupported keys: {', '.join(sorted(extra))}"
                    )
                if missing:
                    raise ValueError(
                        f"{location} is missing required keys: {', '.join(sorted(missing))}"
                    )
                shot_transitions = routed_transitions[shot_index - 1]
                if len(shot_transitions) >= MAX_PHYSICAL_TRANSITIONS:
                    raise ValueError(
                        f"event_transitions routed to shot {shot_index} has more than "
                        f"{MAX_PHYSICAL_TRANSITIONS} transitions"
                    )
                shot_transitions.append({
                    **dict(raw_transition),
                    # Event ownership is derived from the immutable object key,
                    # never accepted as a second model-supplied field.
                    "event_id": event_id,
                })
        raw_shot_transitions = [
            {"shot_index": shot_index, "transitions": transitions}
            for shot_index, transitions in enumerate(routed_transitions, start=1)
        ]
    else:
        raw_shot_transitions = response.get("shot_transitions")
        if not isinstance(raw_shot_transitions, list):
            raise ValueError("shot_transitions must be an array")
        if len(raw_shot_transitions) != row_count:
            raise ValueError(
                "shot_transitions count must match existing shot count "
                f"({row_count}); got {len(raw_shot_transitions)}"
            )

    normalized_states: list[dict[str, Any]] = []
    transition_event_ids_by_shot: list[list[str | None]] = []
    current: dict[tuple[str, str], dict[str, str]] = {
        _state_key(fact["subject_id"], fact["predicate"]): dict(fact)
        for fact in initial_facts
    }
    transition_keys_without_event = {"subject_id", "predicate", "to_value", "action"}
    for offset, raw_shot in enumerate(raw_shot_transitions):
        if not isinstance(raw_shot, Mapping):
            raise ValueError(f"shot_transitions[{offset}] must be an object")
        extra = set(raw_shot) - {"shot_index", "transitions"}
        missing = {"shot_index", "transitions"} - set(raw_shot)
        if extra:
            raise ValueError(
                f"shot_transitions[{offset}] has unsupported keys: {', '.join(sorted(extra))}"
            )
        if missing:
            raise ValueError(
                f"shot_transitions[{offset}] is missing required keys: "
                f"{', '.join(sorted(missing))}"
            )
        shot_index = raw_shot.get("shot_index")
        if isinstance(shot_index, bool) or not isinstance(shot_index, int):
            raise ValueError(f"shot_transitions[{offset}].shot_index must be an integer")
        if shot_index != offset + 1:
            raise ValueError(
                "shot_transitions shot_index values must be unique, complete, and "
                f"in existing row order 1..{row_count}; item {offset + 1} has {shot_index}"
            )
        raw_transitions = raw_shot.get("transitions")
        if not isinstance(raw_transitions, list):
            raise ValueError(f"shot_transitions[{offset}].transitions must be an array")
        if len(raw_transitions) > MAX_PHYSICAL_TRANSITIONS:
            raise ValueError(
                f"shot_transitions[{offset}].transitions has more than "
                f"{MAX_PHYSICAL_TRANSITIONS} transitions"
            )

        opening = [dict(fact) for fact in current.values()]
        normalized_transitions: list[dict[str, Any]] = []
        normalized_transition_event_ids: list[str | None] = []
        prior_transition_event_rank = -1
        for transition_index, raw_transition in enumerate(raw_transitions):
            location = f"shot_transitions[{offset}].transitions[{transition_index}]"
            if not isinstance(raw_transition, Mapping):
                raise ValueError(f"{location} must be an object")
            allowed_transition_keys = transition_keys_without_event | {"event_id"}
            if event_keyed_transitions:
                allowed_transition_keys.add("operation")
            required_transition_keys = set(transition_keys_without_event)
            if not legacy_array_response:
                required_transition_keys.add("event_id")
            extra = set(raw_transition) - allowed_transition_keys
            missing = required_transition_keys - set(raw_transition)
            if extra:
                raise ValueError(
                    f"{location} has unsupported keys: {', '.join(sorted(extra))}"
                )
            if missing:
                raise ValueError(
                    f"{location} is missing required keys: {', '.join(sorted(missing))}"
                )

            event_id = raw_transition.get("event_id")
            if "event_id" in raw_transition:
                if not isinstance(event_id, str) or not event_id:
                    raise ValueError(f"{location}.event_id must be a non-empty string")
                if event_id not in allocated_ids[offset]:
                    raise ValueError(
                        f"{location}.event_id {event_id!r} is not assigned to shot "
                        f"{offset + 1} by event_shots"
                    )
                event_rank = event_order[event_id]
                if event_rank < prior_transition_event_rank:
                    raise ValueError(
                        f"{location}.event_id transitions must follow source-event order "
                        "within each shot"
                    )
                prior_transition_event_rank = event_rank

            subject_id = raw_transition.get("subject_id")
            if not isinstance(subject_id, str) or not subject_id.strip():
                raise ValueError(f"{location}.subject_id must be a non-empty string")
            subject_id = subject_id.strip()
            if known_cast_ids and subject_id not in known_cast_ids:
                raise ValueError(
                    f"{location}.subject_id {subject_id!r} is not a known cast character_id"
                )
            predicate_value = raw_transition.get("predicate")
            if not isinstance(predicate_value, str):
                raise ValueError(f"{location}.predicate must be a string")
            predicate = predicate_value.strip().casefold().replace("-", "_").replace(" ", "_")
            if predicate not in PHYSICAL_STATE_PREDICATES:
                raise ValueError(f"{location} has unsupported predicate {predicate!r}")
            operation = raw_transition.get("operation")
            if "operation" in raw_transition and operation is None:
                raise ValueError(f"{location}.operation must be 'establish' when provided")
            if operation is not None:
                if not event_keyed_transitions:
                    raise ValueError(
                        f"{location}.operation is supported only for immutable event-keyed transitions"
                    )
                if operation != "establish":
                    raise ValueError(f"{location}.operation must be 'establish'")
            key = _state_key(subject_id, predicate)
            current_fact = current.get(key)
            if operation == "establish" and current_fact is not None:
                raise ValueError(
                    f"{location} cannot establish existing physical fact key "
                    f"{subject_id} {predicate}; use a normal change"
                )
            if current_fact is None and operation != "establish":
                raise ValueError(
                    f"{location} targets unknown physical fact key "
                    f"{subject_id} {predicate}; compact patches cannot infer a missing from-state"
                )
            if operation == "establish" and not isinstance(event_id, str):
                raise ValueError(
                    f"{location} establishment requires exact immutable event provenance"
                )
            if operation == "establish" and len(current) >= MAX_PHYSICAL_FACTS:
                raise ValueError(
                    f"{location} establishment would exceed {MAX_PHYSICAL_FACTS} physical facts"
                )

            # Normalize each proposal against the actual current fact. This
            # applies strict grounded enum validation and text-length checks,
            # while ensuring from_value is never supplied by the model.
            transition_input: dict[str, Any] = {
                "subject_id": subject_id,
                "predicate": predicate,
                "from_value": None if operation == "establish" else current_fact["value"],
                "to_value": raw_transition.get("to_value"),
                "action": raw_transition.get("action"),
            }
            if operation == "establish":
                transition_input["operation"] = "establish"
                transition_input["event_id"] = event_id
            candidate = normalize_physical_state({
                "opening": [dict(fact) for fact in current.values()],
                "transitions": [transition_input],
                "closing": [dict(fact) for fact in current.values()],
            })
            if candidate is None:  # pragma: no cover - explicit state object above
                raise ValueError(f"{location} could not be normalized")
            normalized_transition = candidate["transitions"][0]
            transition: dict[str, Any] = {
                "subject_id": subject_id if current_fact is None else current_fact["subject_id"],
                "predicate": predicate,
                "from_value": normalized_transition["from_value"],
                "to_value": normalized_transition["to_value"],
                "action": normalized_transition["action"],
            }
            if operation == "establish":
                transition["operation"] = "establish"
                transition["event_id"] = normalized_transition["event_id"]
            normalized_transitions.append(transition)
            normalized_transition_event_ids.append(
                event_id if isinstance(event_id, str) else None
            )
            current[key] = {
                "subject_id": subject_id if current_fact is None else current_fact["subject_id"],
                "predicate": predicate,
                "value": normalized_transition["to_value"],
            }

        closing = [dict(fact) for fact in current.values()]
        continuity = _normalized_prior_continuity(rows[offset])
        normalized_state = normalize_physical_state({
            "opening": opening,
            "transitions": normalized_transitions,
            "closing": closing,
            "continuity": continuity,
        })
        if normalized_state is None:  # pragma: no cover - explicit object above
            raise ValueError(f"shot {offset + 1} physical state could not be normalized")
        normalized_states.append(normalized_state)
        transition_event_ids_by_shot.append(normalized_transition_event_ids)

    patched_rows = copy.deepcopy(rows)
    for row, source_event_ids, physical_state, transition_event_ids in zip(
        patched_rows, allocated_ids, normalized_states, transition_event_ids_by_shot,
    ):
        row["source_event_ids"] = source_event_ids
        row["physical_state"] = physical_state
        # Keep event provenance private to validation; public physical_state
        # remains on the stable typed schema and never exposes routing metadata.
        row["_director_transition_event_ids"] = transition_event_ids
    return patched_rows
