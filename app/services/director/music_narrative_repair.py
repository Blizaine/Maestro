"""Focused visual repair for music narrative source-action failures."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from .music_narrative import (
    _action_matches,
    _allocation_map,
    _contract_events,
    _event_coverage_positions,
    _event_with_row_actor_identity,
    _is_nominal_action_reference,
    _is_static_modifier_match,
    format_music_narrative_context,
    music_narrative_row_issue,
    validate_music_narrative_allocation,
)
from .physical_continuity import identity_only_description


_VISUAL_FIELDS = (
    "scene_goal",
    "video_prompt",
    "action_beats",
    "ending_beat",
)
_SUBJECT_FIELDS = (
    "character_id",
    "speaker_name",
    "visual_description",
    "position_or_relation",
    "performance_role",
)
_CAMERA_FIELDS = (
    "framing",
    "angle",
    "movement",
    "movement_intensity",
    "lens_feel",
)

_MUSIC_NARRATIVE_VISUAL_PATCH_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "maxItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "scene_goal": {"type": "string"},
            "video_prompt": {"type": "string"},
            "action_beats": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "ending_beat": {"type": "string"},
        },
        "required": list(_VISUAL_FIELDS),
        "additionalProperties": False,
    },
}


def _stable_subjects(row: dict[str, Any]) -> list[dict[str, Any]]:
    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        return []
    return [
        {key: subject[key] for key in _SUBJECT_FIELDS if key in subject}
        for subject in subjects
        if isinstance(subject, dict)
    ]


def _stable_camera(row: dict[str, Any]) -> dict[str, Any]:
    camera = row.get("camera_plan")
    if not isinstance(camera, dict):
        return {}
    return {key: camera[key] for key in _CAMERA_FIELDS if key in camera}


def _valid_visual_patch(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != set(_VISUAL_FIELDS):
        return False
    for field in ("scene_goal", "video_prompt", "ending_beat"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            return False
    beats = value.get("action_beats")
    return (
        isinstance(beats, list)
        and bool(beats)
        and all(isinstance(beat, str) and beat.strip() for beat in beats)
    )


def repair_music_narrative_visuals(
    planner: Any,
    *,
    index: int,
    row: Any,
    issue: str,
    contract: Any,
    allocation: Any,
    duration_seconds: float,
) -> list[dict[str, Any]] | None:
    """Repair only visual prose rejected by the music narrative checker.

    The source-action checker remains authoritative. This bounded pass returns
    a patched copy only when its four visual fields satisfy that same checker.
    """

    if not isinstance(row, dict):
        return None

    source_context = format_music_narrative_context(
        contract, allocation, index, index + 1,
    )
    visual_context = {field: row.get(field) for field in _VISUAL_FIELDS}
    request = {
        "selected_duration_seconds": duration_seconds,
        "stable_subjects": _stable_subjects(row),
        "stable_camera": _stable_camera(row),
        "validation_reason": issue,
        "rejected_visual_fields": visual_context,
    }
    user_prompt = (
        "Repair the rejected visual fields for this one music-video clip. "
        "Copy each required named actor plus source action clause from the exact "
        "clip requirements below into video_prompt unchanged and in source order. "
        "Develop natural movement around that exact action while preserving its "
        "actor, target, verb phrase, and relation. These source clauses override "
        "the old scene goal and draft. Correct conflicting action prose in place; "
        "do not append a contradictory action or claim event IDs as coverage. "
        "Preserve the established cast, stable subjects and camera, other useful "
        "visible details, and any local timecodes. Fit the selected duration. "
        "Return only the four requested visual fields.\n\n"
        "SOURCE ACTION CONTEXT — completed, assigned, and pending actions:\n"
        + source_context
        + "\n\nCLIP REPAIR CONTEXT (JSON):\n"
        + json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    )
    system_prompt = (
        "You are a concise visual continuity editor. Return exactly one JSON "
        "array item matching the supplied schema. Edit only scene_goal, "
        "video_prompt, action_beats, and ending_beat. Preserve unrelated "
        "visible details, cast identity, and local timecodes while making the "
        "assigned source action literal, correctly owned, correctly targeted, "
        "and in source order."
    )
    response = planner._call_llm_json(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=2300,
        thinking_budget=0,
        temperature=0.2,
        json_schema=_MUSIC_NARRATIVE_VISUAL_PATCH_SCHEMA,
    )

    if not isinstance(response, list) or len(response) != 1:
        return None
    patch = response[0]
    if not _valid_visual_patch(patch):
        return None

    merged = copy.deepcopy(row)
    for field in _VISUAL_FIELDS:
        merged[field] = copy.deepcopy(patch[field])
    if music_narrative_row_issue(
        merged,
        contract=contract,
        allocation=allocation,
        index=index,
    ) is not None:
        return None
    return [merged]


_STRUCTURAL_TEXT_FIELDS = (
    "scene_goal",
    "scene_type",
    "environment",
    "visual_style",
    "lighting",
    "mood",
    "ending_beat",
    "video_prompt",
)
_SUBJECT_IDENTITY_FIELDS = (
    "character_id",
    "speaker_name",
    "visual_description",
    "identity",
    "identity_description",
    "description",
)
_NATIVE_TEXT_FIELDS = ("native_ir", "video_ir", "prompt_ir", "_director_native_ir")
_NARRATIVE_DEFINITION_FIELDS = (
    "subject_definitions",
    "subject_defs",
    "compiled_subject_definitions",
)
_ARTICLE_RE = re.compile(r"\b(the|a|an)\s+$", re.IGNORECASE)
_EXITING_BOARD_RE = re.compile(
    r"\b(?:land(?:s|ed|ing)?|disembark(?:s|ed|ing)?|"
    r"dismount(?:s|ed|ing)?|gets? off|steps? off|climbs? off)\b",
    re.IGNORECASE,
)
_TRAVEL_ACTION_RE = re.compile(
    r"\b(?:sail(?:s|ed|ing)?|fly|flies|flew|flying|soar(?:s|ed|ing)?|"
    r"glide(?:s|d|ing)?|travel(?:s|ed|ing)?|journey(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)


def _phrase_tokens(value: Any) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _validated_aliases(event: dict[str, Any], field: str) -> list[str] | None:
    raw = event.get(field)
    if not isinstance(raw, list) or not raw:
        return None
    aliases: list[str] = []
    for value in raw:
        if not isinstance(value, str) or not value.strip() or len(value) > 100:
            return None
        normalized = " ".join(_phrase_tokens(value))
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    if not aliases:
        return None
    longest = max(aliases, key=lambda item: len(_phrase_tokens(item)))
    longest_tokens = _phrase_tokens(longest)
    # The source parser emits a most-specific alias followed by suffix aliases.
    # Disjoint names are ambiguous, so refuse to compile them.
    for alias in aliases:
        alias_tokens = _phrase_tokens(alias)
        if not alias_tokens or longest_tokens[-len(alias_tokens):] != alias_tokens:
            return None
    return [longest]


def _phrase_present(text: Any, alias: str) -> bool:
    tokens = _phrase_tokens(text)
    wanted = _phrase_tokens(alias)
    if not tokens or not wanted:
        return False
    for offset in range(len(tokens) - len(wanted) + 1):
        if tokens[offset:offset + len(wanted)] == wanted:
            return True
    # Source actors can be named as compound objects, such as "sailboat" for
    # the contract alias "boat". Keep this narrow to avoid loose human matches.
    return len(wanted) == 1 and len(wanted[0]) >= 4 and any(
        token.endswith(wanted[0]) for token in tokens
    )


def _source_clause_positions(
    source: str,
    events: list[dict[str, Any]],
) -> dict[str, int] | None:
    positions: dict[str, int] = {}
    cursor = 0
    folded = source.casefold()
    for event in events:
        event_id = str(event.get("event_id") or "")
        text = event.get("text")
        if not event_id or not isinstance(text, str) or not text.strip():
            return None
        found = folded.find(text.casefold(), cursor)
        if found < 0:
            return None
        positions[event_id] = found
        cursor = found + len(text)
    return positions


def _surface_actor(source: str, clause_start: int, alias: str) -> str | None:
    prefix = source[:clause_start]
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    exact = list(re.finditer(rf"(?<![\w]){escaped}(?![\w])", prefix, re.IGNORECASE))
    if not exact and len(_phrase_tokens(alias)) == 1 and len(alias) >= 4:
        exact = list(re.finditer(
            rf"(?<![\w])[\w'-]*{escaped}(?![\w])", prefix, re.IGNORECASE,
        ))
    if not exact:
        return None
    match = exact[-1]
    display = match.group(0)
    article = _ARTICLE_RE.search(prefix[max(0, match.start() - 8):match.start()])
    if article:
        display = article.group(1) + " " + display
    return display.strip()


def _surface_target(event_text: str, aliases: list[str]) -> str | None:
    escaped = re.escape(aliases[0]).replace(r"\ ", r"\s+")
    matches = list(re.finditer(rf"(?<![\w]){escaped}(?![\w])", event_text, re.IGNORECASE))
    if not matches:
        return None
    match = matches[0]
    display = match.group(0)
    article = _ARTICLE_RE.search(event_text[max(0, match.start() - 8):match.start()])
    if article:
        display = article.group(1) + " " + display
    return display.strip()


def _row_has_actor(row: dict[str, Any], alias: str) -> bool:
    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        return False
    for subject in subjects:
        if not isinstance(subject, dict):
            continue
        if any(_phrase_present(subject.get(field), alias) for field in _SUBJECT_IDENTITY_FIELDS):
            return True
    return False


def _safe_detail(value: Any, *, limit: int = 180) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s+", " ", value).strip(" \t\r\n.;")
    if not text or len(text) > limit:
        return ""
    if re.search(r"\b(?:then|before|after|while|as|until|once)\b", text, re.I):
        return ""
    for start, end, _kind, verb in _action_matches(text):
        if not _is_static_modifier_match(text, start, end, verb) and not _is_nominal_action_reference(
            text, start, end, verb,
        ):
            return ""
    return text


def _stable_subject_copy(
    row: dict[str, Any],
    *,
    boarding: bool,
    carried: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    subjects = row.get("subjects_on_screen")
    if not isinstance(subjects, list):
        return []
    copied: list[dict[str, Any]] = []
    for subject in subjects:
        if not isinstance(subject, dict):
            continue
        clean = copy.deepcopy(subject)
        for field in ("visual_description", "identity", "identity_description", "description"):
            if isinstance(clean.get(field), str):
                clean[field] = identity_only_description(clean[field])
        if boarding:
            clean["position_or_relation"] = ""
        else:
            for actor, target in carried:
                if any(
                    _phrase_present(clean.get(field), actor)
                    for field in _SUBJECT_IDENTITY_FIELDS
                ):
                    clean["position_or_relation"] = f"securely aboard {target}"
                    break
        copied.append(clean)
    return copied


def _current_row_is_complete(row: dict[str, Any]) -> bool:
    if any(
        not isinstance(row.get(field), str) or not row[field].strip()
        for field in _STRUCTURAL_TEXT_FIELDS
    ):
        return False
    if not isinstance(row.get("subjects_on_screen"), list):
        return False
    if not isinstance(row.get("action_beats"), list) or not any(
        isinstance(item, str) and item.strip() for item in row["action_beats"]
    ):
        return False
    camera = row.get("camera_plan")
    return (
        isinstance(camera, dict)
        and isinstance(camera.get("framing"), str)
        and bool(camera["framing"].strip())
        and isinstance(row.get("window_prompts"), list)
    )


_BRIDGE_IDENTITY_FIELDS = (
    "character_id",
    "speaker_name",
    "visual_description",
    "identity",
    "identity_description",
    "description",
)
_BRIDGE_IDENTITY_STOPWORDS = {"a", "an", "and", "as", "of", "the", "to", "with"}


def _safe_text_parts(value: Any, *, limit: int = 180) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    parts = re.split(r"[,;.!?]+", value)
    safe = [_safe_detail(part, limit=limit) for part in parts]
    return [part for part in safe if part]


def _bridge_subject_texts(subject: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in _BRIDGE_IDENTITY_FIELDS:
        value = subject.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        if field in {"visual_description", "identity", "identity_description", "description"}:
            value = identity_only_description(value)
        normalized = " ".join(_phrase_tokens(value))
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def _bridge_subjects_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_name = _phrase_tokens(left.get("speaker_name"))
    right_name = _phrase_tokens(right.get("speaker_name"))
    if left_name and right_name:
        return left_name == right_name
    left_id = _phrase_tokens(left.get("character_id"))
    if left_id and left_id == _phrase_tokens(right.get("character_id")):
        return True
    # Row-local IDs may drift. Accept the same distinctive appearance, with
    # punctuation/connectives normalized, rather than guessing from resemblance.
    for left_text in _bridge_subject_texts(left):
        left_tokens = tuple(t for t in _phrase_tokens(left_text) if t not in _BRIDGE_IDENTITY_STOPWORDS)
        for right_text in _bridge_subject_texts(right):
            right_tokens = tuple(t for t in _phrase_tokens(right_text) if t not in _BRIDGE_IDENTITY_STOPWORDS)
            if len(left_tokens) >= 3 and left_tokens == right_tokens:
                return True
    return False


def _bridge_ending_state(previous: dict[str, Any], events: list[dict[str, Any]]) -> str:
    # Keep ownership intact: never split comma clauses and reassign them to
    # individual cast members. Recognized source transitions are not replayed.
    ending = str(previous.get("ending_beat") or "")
    for sentence in re.split(r"(?<=[.!?])\s+", ending):
        text = re.sub(r"\s+", " ", sentence).strip(" .")
        if text and len(text) <= 900 and not any(
            _event_coverage_positions(text, _event_with_row_actor_identity(previous, event))
            for event in events
        ):
            return text
    return ""


def _bridge_camera(row: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current = row.get("camera_plan")
    prior = previous.get("camera_plan")
    current = current if isinstance(current, dict) else {}
    prior = prior if isinstance(prior, dict) else {}
    camera: dict[str, Any] = {}
    for field in _CAMERA_FIELDS:
        value = _safe_detail(current.get(field), limit=80)
        if not value:
            value = _safe_detail(prior.get(field), limit=80)
        if value:
            camera[field] = value
    if not camera.get("framing"):
        camera["framing"] = "wide shot"
    return camera


def compile_music_narrative_bridge_row(
    *,
    index: int,
    row: Any,
    issue: str,
    contract: Any,
    allocation: Any,
    previous: Any = None,
) -> list[dict[str, Any]] | None:
    """Build a grounded no-event bridge from an accepted preceding clip.

    The bridge keeps the current row's cast and any safe camera choice, while
    sourcing visible state and location from the accepted previous row. It
    cannot perform or reallocate a source event.
    """

    if (
        not isinstance(row, dict)
        or not isinstance(previous, dict)
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 1
        or not isinstance(issue, str)
        or not issue.strip()
        or not _current_row_is_complete(row)
        or not _current_row_is_complete(previous)
        or row.get("image_prompt")
        or row.get("keyframe_prompts")
    ):
        return None

    events = _contract_events(contract)
    if not events or not isinstance(allocation, list):
        return None
    allocation_indexes = [
        item.get("clip_index")
        for item in allocation
        if isinstance(item, dict)
        and isinstance(item.get("clip_index"), int)
        and not isinstance(item.get("clip_index"), bool)
    ]
    clip_count = max([index + 1, *(value + 1 for value in allocation_indexes)])
    if validate_music_narrative_allocation(contract, allocation, clip_count):
        return None

    if music_narrative_row_issue(
        row, contract=contract, allocation=allocation, index=index,
    ) is None:
        return None
    if music_narrative_row_issue(
        previous, contract=contract, allocation=allocation, index=index - 1,
    ) is not None:
        return None
    assignments = _allocation_map(allocation)
    if any(
        assignments.get(str(event.get("event_id") or "")) == index
        for event in events
    ):
        return None
    expected_ids = [
        str(event.get("event_id") or "")
        for event in events
        if assignments.get(str(event.get("event_id") or "")) == index
    ]
    if expected_ids or row.get("source_event_ids") != []:
        return None

    previous_environment = ". ".join(_safe_text_parts(previous.get("environment")))
    previous_environment = _safe_detail(previous_environment, limit=360)
    if not previous_environment:
        return None

    current_subjects = row.get("subjects_on_screen")
    previous_subjects = previous.get("subjects_on_screen")
    if not isinstance(current_subjects, list) or not isinstance(previous_subjects, list):
        return None
    prior_subjects = [subject for subject in previous_subjects if isinstance(subject, dict)]
    ending_state = _bridge_ending_state(previous, events)
    matched_subjects: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for subject in current_subjects:
        if not isinstance(subject, dict):
            return None
        matches = [
            prior for prior in prior_subjects
            if _bridge_subjects_match(subject, prior)
        ]
        if len(matches) != 1:
            return None
        matched_subjects.append((subject, matches[0]))

    # A whole ending can include every visible actor. Require the same cast
    # when using it; a cutaway can carry only its own proven position fields.
    matched_prior = {id(prior) for _, prior in matched_subjects}
    if len(matched_prior) != len(matched_subjects):
        return None
    same_cast = len(matched_prior) == len(prior_subjects)
    merged = copy.deepcopy(row)
    merged_subjects: list[dict[str, Any]] = []
    visible_states: list[str] = []
    for current_subject, prior_subject in matched_subjects:
        copied = copy.deepcopy(current_subject)
        for field in ("visual_description", "identity", "identity_description", "description"):
            if isinstance(copied.get(field), str):
                copied[field] = identity_only_description(copied[field])
        position = _safe_detail(prior_subject.get("position_or_relation"), limit=360)
        copied["position_or_relation"] = position
        if position:
            label = str(current_subject.get("speaker_name") or current_subject.get("character_id") or "").strip()
            if label:
                visible_states.append(f"{label} remains {position}")
        elif not same_cast or not ending_state:
            return None
        merged_subjects.append(copied)
    if same_cast and ending_state:
        state_text = "; ".join([ending_state, *visible_states])
    elif visible_states:
        state_text = "; ".join(visible_states)
    elif not merged_subjects:
        state_text = f"The established setting remains {previous_environment}"
    else:
        return None

    camera = _bridge_camera(row, previous)
    camera_bits = [
        _safe_detail(camera.get(field), limit=80)
        for field in ("framing", "angle", "movement", "lens_feel")
    ]
    camera_bits = [value for value in camera_bits if value]
    appearance_bits = [
        _safe_detail(identity_only_description(subject.get("visual_description")), limit=240)
        for subject in merged_subjects
    ]
    appearance_bits = [value for value in appearance_bits if value]
    prompt_parts = ["[Shot 1]"]
    if camera_bits:
        prompt_parts.append("Composition: " + ", ".join(camera_bits) + ".")
    prompt_parts.append(f"Setting: {previous_environment}.")
    prompt_parts.append("Established visible state: " + state_text + ".")
    for appearance in appearance_bits:
        prompt_parts.append(f"Stable appearance: {appearance}.")
    prompt_parts.append(
        "The camera sustains a visual music bridge over the supplied soundtrack, "
        "which continues unchanged; add no dialogue or transcribed lyrics."
    )
    prompt = " ".join(prompt_parts)

    merged["subjects_on_screen"] = merged_subjects
    merged["camera_plan"] = camera
    merged["environment"] = previous_environment
    for field, fallback in (
        ("visual_style", "Cinematic visual style"),
        ("lighting", "Natural, consistent lighting"),
        ("mood", "Steady, reflective mood"),
    ):
        value = _safe_detail(previous.get(field)) or _safe_detail(row.get(field)) or fallback
        merged[field] = value
    merged["scene_goal"] = "Hold a camera bridge on the established visible state."
    merged["video_prompt"] = prompt
    merged["detailed_description"] = prompt
    merged["action_beats"] = [
        "The camera holds on the established visible state as the supplied soundtrack continues."
    ]
    merged["ending_beat"] = state_text
    merged["spatial_setup"] = state_text
    merged["window_prompts"] = []
    for field in _NARRATIVE_DEFINITION_FIELDS:
        if field in merged:
            merged[field] = []
    for field in _NATIVE_TEXT_FIELDS:
        if field in merged:
            merged[field] = prompt
    merged["_music_bridge_compiled"] = True
    merged["_music_source_compiled"] = True
    merged["_music_source_recovery_reason"] = issue

    if music_narrative_row_issue(
        merged, contract=contract, allocation=allocation, index=index,
    ) is not None:
        return None
    return [merged]


def _same_entity(left: str, right: str) -> bool:
    return _phrase_present(left, right) or _phrase_present(right, left)


def _completed_boarding_carries(
    *,
    index: int,
    current_events: list[dict[str, Any]],
    events: list[dict[str, Any]],
    assignments: dict[str, int],
    source_positions: dict[str, int],
    source: str,
    row: dict[str, Any],
) -> list[tuple[str, str]]:
    travel_events = [
        event for event in current_events
        if str(event.get("kind") or "").casefold() == "flight"
        or _TRAVEL_ACTION_RE.search(str(event.get("text") or ""))
    ]
    if not travel_events:
        return []
    carries: list[tuple[str, str]] = []
    for board in events:
        board_id = str(board.get("event_id") or "")
        board_index = assignments.get(board_id)
        if (
            str(board.get("kind") or "").casefold() != "boarding"
            or board_index is None
            or board_index >= index
        ):
            continue
        actor_aliases = _validated_aliases(board, "actor_aliases")
        target_aliases = _validated_aliases(board, "target_aliases")
        actor_alias = actor_aliases[0] if actor_aliases else ""
        target_alias = target_aliases[0] if target_aliases else ""
        board_position = source_positions.get(board_id, -1)
        if not actor_alias or not target_alias or not _row_has_actor(row, actor_alias):
            continue
        # A completed boarding carries its actor only when this travel action
        # belongs to the same source-owned vessel or vehicle.
        current_actor_aliases = [
            _validated_aliases(event, "actor_aliases")
            for event in travel_events
        ]
        if not any(
            aliases and _same_entity(aliases[0], target_alias)
            for aliases in current_actor_aliases
        ):
            continue
        exited = False
        for later in events:
            later_id = str(later.get("event_id") or "")
            later_index = assignments.get(later_id)
            later_text = str(later.get("text") or "")
            if later_index is None or not board_index < later_index < index:
                continue
            if source_positions.get(later_id, -1) <= board_position:
                continue
            later_aliases = _validated_aliases(later, "actor_aliases")
            if (
                later_aliases
                and _same_entity(later_aliases[0], actor_alias)
                and _EXITING_BOARD_RE.search(later_text)
            ):
                exited = True
                break
        if exited:
            continue
        actor = _surface_actor(source, board_position, actor_alias)
        target = _surface_target(str(board.get("text") or ""), target_aliases)
        if actor and target:
            carries.append((actor, target))
    return carries


def _compile_source_prompt(
    *,
    row: dict[str, Any],
    action_clauses: list[str],
    boarding_clauses: list[tuple[str, str]],
    carried: list[tuple[str, str]],
    include_details: bool,
) -> str:
    prompt_parts = ["[Shot 1]"]
    if include_details:
        camera = row.get("camera_plan")
        if isinstance(camera, dict):
            camera_bits = [
                _safe_detail(camera.get(field), limit=80)
                for field in ("framing", "angle", "movement", "lens_feel")
            ]
            camera_bits = [value for value in camera_bits if value]
            if camera_bits:
                prompt_parts.append("Composition: " + ", ".join(camera_bits) + ".")
        for label, field in (
            ("Setting", "environment"),
            ("Visual style", "visual_style"),
            ("Lighting", "lighting"),
            ("Mood", "mood"),
        ):
            value = _safe_detail(row.get(field))
            if value:
                prompt_parts.append(f"{label}: {value}.")
        subjects = row.get("subjects_on_screen")
        if isinstance(subjects, list):
            for subject in subjects:
                if not isinstance(subject, dict):
                    continue
                appearance = _safe_detail(
                    identity_only_description(subject.get("visual_description")),
                    limit=240,
                )
                if appearance:
                    prompt_parts.append(f"Stable appearance: {appearance}.")
    for actor, target in carried:
        prompt_parts.append(f"{actor} remains securely aboard {target}.")
    if boarding_clauses:
        actor, target = boarding_clauses[0]
        prompt_parts.append(f"At the start, {actor} is unboarded beside {target}.")
    prompt_parts.extend(action_clauses)
    for actor, target in boarding_clauses:
        prompt_parts.append(f"At the end, {actor} is securely aboard {target}.")
    prompt_parts.append(
        "The supplied soundtrack continues unchanged; do not add dialogue or transcribe lyrics."
    )
    return " ".join(prompt_parts)


def compile_music_narrative_source_row(
    *,
    index: int,
    row: Any,
    issue: str,
    contract: Any,
    allocation: Any,
    previous: Any = None,
) -> list[dict[str, Any]] | None:
    """Compile rejected story prose from assigned immutable source clauses.

    The deterministic fallback never calls a model and returns a patched copy
    only when the ordinary music narrative checker accepts it unchanged.
    """

    del previous  # Historical prose is deliberately not an authority here.
    if (
        not isinstance(row, dict)
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or not isinstance(issue, str)
        or not issue.strip()
        or not _current_row_is_complete(row)
    ):
        return None
    if music_narrative_row_issue(
        row, contract=contract, allocation=allocation, index=index,
    ) is None:
        return None

    events = _contract_events(contract)
    if not events:
        return None
    assignments = _allocation_map(allocation)
    current_events = [
        event for event in events
        if assignments.get(str(event.get("event_id") or "")) == index
    ]
    if not current_events:
        return None
    source = str(contract.get("source") or "") if isinstance(contract, dict) else ""
    positions = _source_clause_positions(source, events)
    if positions is None:
        return None

    action_clauses: list[str] = []
    boarding_clauses: list[tuple[str, str]] = []
    event_positions = {
        str(event.get("event_id") or ""): offset
        for offset, event in enumerate(events)
    }
    for event in current_events:
        event_id = str(event.get("event_id") or "")
        aliases = _validated_aliases(event, "actor_aliases")
        if not aliases:
            return None
        actor_alias = aliases[0]
        actor = _surface_actor(source, positions[event_id], actor_alias)
        if not actor:
            return None
        event_text = str(event.get("text") or "").strip()
        object_actor_known = any(
            str(prior.get("kind") or "").casefold() == "boarding"
            and assignments.get(str(prior.get("event_id") or ""), index + 1) <= index
            and event_positions.get(str(prior.get("event_id") or ""), len(events))
            < event_positions[event_id]
            and (targets := _validated_aliases(prior, "target_aliases"))
            and _same_entity(actor_alias, targets[0])
            for prior in events
        )
        # A source-established vessel/mount remains a known actor when the
        # current camera excludes it from the cast list. Its assigned source
        # action may be travel, attack or another verb; never invent an ID.
        if not _row_has_actor(row, actor_alias) and not object_actor_known:
            return None
        clause = f"{actor} {event_text}"
        if not clause.endswith((".", "!", "?")):
            clause += "."
        action_clauses.append(clause)
        if str(event.get("kind") or "").casefold() == "boarding":
            target_aliases = _validated_aliases(event, "target_aliases")
            target = _surface_target(event_text, target_aliases) if target_aliases else None
            if not target:
                return None
            boarding_clauses.append((actor, target))

    carried = _completed_boarding_carries(
        index=index,
        current_events=current_events,
        events=events,
        assignments=assignments,
        source_positions=positions,
        source=source,
        row=row,
    )
    merged = copy.deepcopy(row)
    merged["subjects_on_screen"] = _stable_subject_copy(
        merged,
        boarding=bool(boarding_clauses),
        carried=carried,
    )
    for field in _NARRATIVE_DEFINITION_FIELDS:
        if field in merged:
            merged[field] = []
    prompt = _compile_source_prompt(
        row=merged,
        action_clauses=action_clauses,
        boarding_clauses=boarding_clauses,
        carried=carried,
        include_details=True,
    )
    minimal_prompt = _compile_source_prompt(
        row=merged,
        action_clauses=action_clauses,
        boarding_clauses=boarding_clauses,
        carried=carried,
        include_details=False,
    )
    merged["video_prompt"] = prompt
    merged["detailed_description"] = prompt
    for field in _NATIVE_TEXT_FIELDS:
        if field in merged:
            merged[field] = prompt
    merged["scene_goal"] = "Show the assigned source actions in order: " + "; ".join(action_clauses)
    merged["action_beats"] = copy.deepcopy(action_clauses)
    merged["ending_beat"] = (
        f"{boarding_clauses[-1][0]} is securely aboard {boarding_clauses[-1][1]}"
        if boarding_clauses
        else action_clauses[-1]
    )
    merged["_music_source_compiled"] = True
    merged["_music_source_recovery_reason"] = issue

    if music_narrative_row_issue(
        merged, contract=contract, allocation=allocation, index=index,
    ) is not None:
        merged["video_prompt"] = minimal_prompt
        merged["detailed_description"] = minimal_prompt
        for field in _NATIVE_TEXT_FIELDS:
            if field in merged:
                merged[field] = minimal_prompt
        if music_narrative_row_issue(
            merged, contract=contract, allocation=allocation, index=index,
        ) is not None:
            return None
    return [merged]
