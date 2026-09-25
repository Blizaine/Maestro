"""Validated relationships between immutable H3 source events.

The catalog remains unchanged. A narrowly validated overview may be represented
by ordered detail obligations, but its original source text and relationship are
retained for audit. Semantic completeness is supplied by the caller as a
fail-closed validator over the immutable event text.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable, Iterable, Mapping, Sequence
import re
from typing import Any


CoverageValidator = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]]], bool | tuple[bool, str]
]


def normalize_h3_source_relationships(
    raw_relationships: Any,
    source_events: Sequence[Mapping[str, Any]],
    *,
    assigned_event_ids: Iterable[Any] | None = None,
    protected_parent_ids: Iterable[Any] = (),
    recurring_event_ids: Iterable[Any] = (),
    coverage_validator: CoverageValidator | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate overview-to-detail links without changing source identities.

    A relationship is accepted only when it points forward through known
    immutable IDs, its detail events are actually assigned once (when a
    proposed schedule is available), it does not own dialogue or explicit
    recurrence, and the caller independently proves complete semantic
    coverage. Missing or uncertain relationships leave every source event
    executable.
    """

    diagnostics: list[str] = []
    order: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(source_events):
        if not isinstance(event, Mapping):
            diagnostics.append(f"source event {index + 1} is malformed")
            continue
        event_id = str(event.get("event_id") or "").strip().upper()
        event_text = " ".join(str(event.get("text") or "").split())
        if not event_id or not event_text or event_id in by_id:
            diagnostics.append(f"source event {index + 1} has a missing or duplicate identity")
            continue
        order.append(event_id)
        by_id[event_id] = {**dict(event), "event_id": event_id, "text": event_text}

    if raw_relationships is None:
        return [], diagnostics
    if isinstance(raw_relationships, Mapping):
        raw_relationships = raw_relationships.get("source_relationships")
    if not isinstance(raw_relationships, list):
        return [], diagnostics + ["source_relationships must be a list"]
    if not raw_relationships:
        return [], diagnostics

    positions = {event_id: index for index, event_id in enumerate(order)}
    assigned = None
    if assigned_event_ids is not None:
        assigned = [str(value or "").strip().upper() for value in assigned_event_ids]
    protected = {str(value or "").strip().upper() for value in protected_parent_ids}
    recurring = {str(value or "").strip().upper() for value in recurring_event_ids}
    accepted: list[dict[str, Any]] = []
    seen_relation_ids: set[str] = set()
    seen_parents: set[str] = set()
    used_events: set[str] = set()

    for index, raw in enumerate(raw_relationships):
        label = f"source_relationships[{index}]"
        if not isinstance(raw, Mapping):
            diagnostics.append(f"{label} is malformed")
            continue
        relation_id = " ".join(str(raw.get("relation_id") or "").split()).upper()
        relation_type = " ".join(str(raw.get("relation_type") or "").split()).casefold()
        parent_id = str(raw.get("overview_event_id") or "").strip().upper()
        detail_values = raw.get("detail_event_ids")
        if not relation_id or relation_type != "overview_of" or not parent_id:
            diagnostics.append(f"{label} lacks a valid ID, overview_of type, or parent event")
            continue
        if raw.get("coverage_status") != "complete":
            diagnostics.append(f"{label} does not claim complete overview coverage")
            continue
        if relation_id in seen_relation_ids or parent_id in seen_parents:
            diagnostics.append(f"{label} duplicates a relation or overview event")
            continue
        if parent_id not in positions or not isinstance(detail_values, list):
            diagnostics.append(f"{label} references an unknown overview or malformed details")
            continue
        details = [str(value or "").strip().upper() for value in detail_values]
        if len(details) < 2 or any(not value or value not in positions for value in details):
            diagnostics.append(f"{label} requires at least two known detail events")
            continue
        if len(set(details)) != len(details):
            diagnostics.append(f"{label} repeats a detail event")
            continue
        if positions[parent_id] >= positions[details[0]]:
            diagnostics.append(f"{label} overview must precede its detail events")
            continue
        if [positions[value] for value in details] != sorted(positions[value] for value in details):
            diagnostics.append(f"{label} detail events are out of source order")
            continue
        if parent_id in protected or parent_id in recurring:
            diagnostics.append(f"{label} cannot demote a dialogue-anchored or recurring event")
            continue
        if any(by_id[parent_id].get(key) is not None for key in (
            "source_start_seconds", "source_end_seconds", "start_seconds", "end_seconds",
        )):
            diagnostics.append(f"{label} cannot demote an explicitly timed source event")
            continue
        if parent_id in details or any(value in seen_parents for value in details):
            diagnostics.append(f"{label} creates a nested or cyclic overview relation")
            continue
        if any(value in used_events for value in [parent_id, *details]):
            diagnostics.append(f"{label} overlaps another accepted source relationship")
            continue
        if assigned is not None:
            counts = {value: assigned.count(value) for value in set(assigned)}
            if any(counts.get(value, 0) != 1 for value in details):
                diagnostics.append(f"{label} details are not each assigned exactly once")
                continue
            if counts.get(parent_id, 0) > 1:
                diagnostics.append(f"{label} overview is assigned more than once")
                continue

        if coverage_validator is None:
            diagnostics.append(f"{label} has no semantic coverage validator")
            continue
        try:
            proof = coverage_validator(by_id[parent_id], [by_id[value] for value in details])
        except Exception as exc:  # Fail closed; callers decide how to report it.
            diagnostics.append(f"{label} semantic coverage check failed: {type(exc).__name__}")
            continue
        if isinstance(proof, tuple):
            complete, reason = bool(proof[0]), str(proof[1] or "")
        else:
            complete, reason = bool(proof), ""
        if not complete:
            diagnostics.append(f"{label} does not cover the complete overview" + (f": {reason}" if reason else ""))
            continue

        accepted.append({
            "relation_id": relation_id,
            "relation_type": "overview_of",
            "overview_event_id": parent_id,
            "detail_event_ids": details,
            "coverage_status": "complete",
            "source_evidence": {
                "overview": {"event_id": parent_id, "exact_text": by_id[parent_id]["text"]},
                "details": [
                    {"event_id": value, "exact_text": by_id[value]["text"]}
                    for value in details
                ],
            },
        })
        seen_relation_ids.add(relation_id)
        seen_parents.add(parent_id)
        used_events.update((parent_id, *details))

    # A child may not also be the parent of a later link. Checking after
    # normalization catches reverse-order nested graphs without trusting IDs
    # to be numerically contiguous.
    parent_ids = {item["overview_event_id"] for item in accepted}
    invalid_parents = {
        item["overview_event_id"]
        for item in accepted
        if parent_ids.intersection(item["detail_event_ids"])
    }
    if invalid_parents:
        accepted = [item for item in accepted if item["overview_event_id"] not in invalid_parents]
        diagnostics.append("source relationships cannot nest overview parents")
    return accepted, diagnostics


def executable_h3_source_event_ids(
    source_events: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return original event IDs excluding only validated overview parents."""

    overview_ids = {
        str(item.get("overview_event_id") or "").strip().upper()
        for item in relationships
        if isinstance(item, Mapping)
    }
    return [
        str(item.get("event_id") or "").strip().upper()
        for item in source_events
        if isinstance(item, Mapping)
        and str(item.get("event_id") or "").strip().upper() not in overview_ids
    ]


_THRESHOLD_PHRASE = re.compile(
    r"\b(?:the|same|that|this|a|an)\s+"
    r"(?P<modifiers>(?:[A-Za-z][A-Za-z’'-]*\s+){0,3})"
    r"(?P<head>door|gate|hatch)\b",
    re.IGNORECASE,
)
_THRESHOLD_STATE_WORDS = {
    "ajar", "closed", "open", "locked", "unlocked", "shut",
}
_PASSAGE_ACTION = re.compile(
    r"\b(?:pass(?:es|ed)?|walk(?:s|ed)?|move(?:s|d)?|cross(?:es|ed)?|"
    r"step(?:s|ped)?|travel(?:s|ed)?)\b"
    r"[^.!?;]{0,80}?\b(?:through|across|past)\s+"
    r"(?:the|same|that|this|a|an)\s+"
    r"(?:[A-Za-z][A-Za-z’'-]*\s+){0,3}(?:door|gate|hatch)\b",
    re.IGNORECASE,
)
_ACTIVE_OPENER = re.compile(
    r"\b(?P<actor>[A-Z][A-Za-z’'-]*)\s+"
    r"(?:opens?|opened|unlocks?|unlocked)\s+"
    r"(?:the|same|that|this|a|an)\s+"
    r"(?:[A-Za-z][A-Za-z’'-]*\s+){0,3}(?P<head>door|gate|hatch)\b",
    re.IGNORECASE,
)
_NAMED_TRAVELER = re.compile(
    r"\b(?P<actor>[A-Z][A-Za-z’'-]*)\s+"
    r"(?:enters?|entered|passes?\s+through|passed\s+through|"
    r"walks?\s+through|walked\s+through|steps?\s+through|stepped\s+through|"
    r"crosses?|crossed)\b",
    re.IGNORECASE,
)
_LATER_OCCASION_OR_RETURN = re.compile(
    r"\b(?:again|another\s+(?:visit|trip|journey|crossing)|"
    r"(?:later|next|following|subsequent|previous|earlier)\s+"
    r"(?:day|evening|night|morning|visit|trip|journey|crossing)|"
    r"on\s+(?:the\s+)?(?:next|following|subsequent|previous|later)\b|"
    r"return(?:s|ed|ing)?\s+(?:to|through|back)|go(?:es|t|ing)?\s+back)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_ENABLER = re.compile(
    r"\b(?:if|unless)\b.{0,80}\b(?:opens?|opened|opening|unlocks?|unlocked|unlocking|"
    r"enters?|entered|entering|passes?|passed|passing|crosses?|crossed|crossing)\b|"
    r"\b(?:could|would|might|may|should)\b.{0,60}\b(?:opens?|opened|opening|unlocks?|"
    r"unlocked|unlocking|enters?|entered|entering|passes?|passed|passing|crosses?|crossed|crossing)\b|"
    r"\b(?:plan(?:s|ned)?|intend(?:s|ed)?|hope(?:s|d)?|want(?:s|ed)?)\s+to\b"
    r".{0,60}\b(?:open(?:s|ed|ing)?|unlock(?:s|ed|ing)?|enter(?:s|ed|ing)?|"
    r"pass(?:es|ed|ing)?|cross(?:es|ed|ing)?)\b",
    re.IGNORECASE,
)
_NEGATED_OR_FAILED_ACTION = re.compile(
    r"\b(?:fail(?:s|ed)?|try|tries|tried|attempt(?:s|ed)?)\s+to\b.{0,60}\b"
    r"(?:open|unlock|enter|pass|cross|walk)\b|"
    r"\b(?:cannot|can't|does\s+not|doesn't|do\s+not|don't|never)\b.{0,50}\b"
    r"(?:open|unlock|enter|pass|cross|walk)\b|"
    r"\bwithout\s+(?:opening|unlocking|entering|passing|crossing)\b",
    re.IGNORECASE,
)
_PERSISTENT_CLOSED_PASSAGE = re.compile(
    r"\b(?:while|as|when)\b.{0,48}\b(?:still\s+)?(?:closed|locked|shut)\b|"
    r"\b(?:still\s+closed|remains?\s+(?:closed|locked|shut)|"
    r"stays?\s+(?:closed|locked|shut))\b.{0,48}\b"
    r"(?:pass|walk|cross|enter|step|move)\b",
    re.IGNORECASE,
)
_MAGICAL_PASSAGE = re.compile(
    r"\b(?:magic(?:al(?:ly)?)?|teleport(?:s|ed|ing)?|phase(?:s|d|ing)?|"
    r"ghost(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_DIALOGUE_OR_QUOTE = re.compile(
    r'"[^"\r\n]*"|“[^”\r\n]*”|‘[^’\r\n]*’|`[^`\r\n]*`|'
    r"(?<!\w)'[^'\r\n]+'(?!\w)|\b(?:says?|said|asks?|asked|speaks?|spoke|dialogue)\b",
    re.IGNORECASE,
)
_EVENT_TIMING_FIELDS = (
    "source_start_seconds", "source_end_seconds", "start_seconds", "end_seconds",
    "timecode", "time_range", "source_time_range",
)


def _event_is_timed(event: Mapping[str, Any]) -> bool:
    return any(event.get(field) is not None for field in _EVENT_TIMING_FIELDS)


def _event_has_dialogue(event: Mapping[str, Any]) -> bool:
    if any(event.get(field) for field in ("dialogue_ids", "locked_dialogue", "dialogue_anchor", "dialogue_id")):
        return True
    return bool(_DIALOGUE_OR_QUOTE.search(str(event.get("text") or "")))


def _threshold_mentions(text: str) -> list[tuple[str, str, str]]:
    """Return (head, qualifier, phrase) for explicit threshold noun phrases."""
    mentions: list[tuple[str, str, str]] = []
    for match in _THRESHOLD_PHRASE.finditer(text):
        head = match.group("head").casefold()
        words = [
            word.casefold()
            for word in re.findall(r"[A-Za-z][A-Za-z’'-]*", match.group("modifiers"))
            if word.casefold() not in _THRESHOLD_STATE_WORDS
        ]
        qualifier = " ".join(words)
        phrase = " ".join(part for part in (qualifier, head) if part)
        mentions.append((head, qualifier, phrase))
    return mentions


def _resolve_unique_threshold(texts: Sequence[str]) -> str | None:
    mentions = [mention for text in texts for mention in _threshold_mentions(text)]
    if not mentions:
        return None
    heads = {head for head, _qualifier, _phrase in mentions}
    if len(heads) != 1:
        return None
    qualified = {qualifier for _head, qualifier, _phrase in mentions if qualifier}
    if len(qualified) > 1:
        return None
    head = next(iter(heads))
    qualifier = next(iter(qualified), "")
    # A generic later reference such as “the door” is safe only when the
    # validated overview group names exactly one more specific threshold.
    return " ".join(part for part in (qualifier, head) if part)


def propose_h3_transport_enabler_groups(
    source_events: Sequence[Mapping[str, Any]],
    validated_relationships: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Propose adjacent same-passage enabler groups from validated overviews.

    The only accepted shape is an untimed, silent movement summary immediately
    followed by an untimed, silent detail that explicitly opens/unlocks the
    same unique threshold before a named traveler enters or passes through it.
    Both immutable event IDs remain separate and executable. The caller must
    recompute and validate the overview relationship before using this result.
    """
    events: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for index, raw in enumerate(source_events):
        if not isinstance(raw, Mapping):
            return []
        event_id = str(raw.get("event_id") or "").strip().upper()
        text = " ".join(str(raw.get("text") or "").split())
        if not event_id or not text or event_id in positions:
            return []
        positions[event_id] = index
        events.append({**dict(raw), "event_id": event_id, "text": text})

    event_by_id = {event["event_id"]: event for event in events}
    groups: list[dict[str, Any]] = []
    used: set[str] = set()
    for relation in validated_relationships:
        if not isinstance(relation, Mapping) or relation.get("relation_type") != "overview_of":
            continue
        if relation.get("coverage_status") != "complete":
            continue
        overview_id = str(relation.get("overview_event_id") or "").strip().upper()
        raw_detail_ids = relation.get("detail_event_ids")
        if not isinstance(raw_detail_ids, Sequence) or isinstance(raw_detail_ids, (str, bytes)):
            continue
        detail_ids = [str(value or "").strip().upper() for value in raw_detail_ids]
        if overview_id not in event_by_id or len(detail_ids) < 2:
            continue
        if any(value not in event_by_id for value in detail_ids) or len(set(detail_ids)) != len(detail_ids):
            continue
        if positions[overview_id] >= positions[detail_ids[0]]:
            continue
        if [positions[value] for value in detail_ids] != sorted(positions[value] for value in detail_ids):
            continue
        for summary_id, detail_id in zip(detail_ids, detail_ids[1:]):
            if summary_id in used or detail_id in used:
                continue
            summary = event_by_id[summary_id]
            detail = event_by_id[detail_id]
            summary_text, detail_text = summary["text"], detail["text"]
            if positions[detail_id] != positions[summary_id] + 1:
                continue
            if _event_is_timed(summary) or _event_is_timed(detail):
                continue
            if _event_has_dialogue(summary) or _event_has_dialogue(detail):
                continue
            if _LATER_OCCASION_OR_RETURN.search(summary_text + " " + detail_text):
                continue
            if _PERSISTENT_CLOSED_PASSAGE.search(summary_text + " " + detail_text):
                continue
            if _MAGICAL_PASSAGE.search(summary_text + " " + detail_text):
                continue
            if _HYPOTHETICAL_ENABLER.search(summary_text + " " + detail_text):
                continue
            if _NEGATED_OR_FAILED_ACTION.search(summary_text + " " + detail_text):
                continue

            passage = _PASSAGE_ACTION.search(summary_text)
            opener = _ACTIVE_OPENER.search(detail_text)
            if not passage or not opener:
                continue
            traveler = _NAMED_TRAVELER.search(detail_text, opener.end())
            if not traveler or opener.start() >= traveler.start():
                continue
            summary_thresholds = _threshold_mentions(passage.group(0))
            detail_thresholds = _threshold_mentions(detail_text)
            if not summary_thresholds or not detail_thresholds:
                continue
            summary_heads = {item[0] for item in summary_thresholds}
            if opener.group("head").casefold() not in summary_heads:
                continue
            relation_texts = [event_by_id[overview_id]["text"]] + [
                event_by_id[value]["text"] for value in detail_ids
            ]
            threshold = _resolve_unique_threshold(relation_texts)
            if not threshold:
                continue
            # Ensure this exact passage is the only movement through the
            # identified threshold in the pair; ambiguous second routes abstain.
            pair_thresholds = _threshold_mentions(summary_text + " " + detail_text)
            if {item[0] for item in pair_thresholds} != summary_heads:
                continue
            named_overview_people = {
                match.group(0).casefold()
                for match in re.finditer(r"\b[A-Z][A-Za-z’'-]*\b", event_by_id[overview_id]["text"])
            }
            opener_name = opener.group("actor").casefold()
            traveler_name = traveler.group("actor").casefold()
            if opener_name not in named_overview_people or traveler_name not in named_overview_people:
                continue
            groups.append({
                "relation_type": "same_passage_enabler",
                "overview_event_id": overview_id,
                "source_event_ids": [summary_id, detail_id],
                "threshold": threshold,
                "opener": opener.group("actor"),
                "traveler": traveler.group("actor"),
                "source_evidence": {
                    "overview": {
                        "event_id": overview_id,
                        "exact_text": event_by_id[overview_id]["text"],
                    },
                    "summary": {"event_id": summary_id, "exact_text": summary_text},
                    "enabler": {"event_id": detail_id, "exact_text": detail_text},
                },
            })
            used.update((summary_id, detail_id))
    return groups


_FALLBACK_SUBJECT = r"(?:they|[A-Z][A-Za-z’'-]*\s+and\s+[A-Z][A-Za-z’'-]*)"
_FALLBACK_APPROACH_VERB = (
    r"(?:cross(?:es|ed)?|walk(?:s|ed)?|move(?:s|d)?|travel(?:s|ed)?|step(?:s|ped)?)"
)
_FALLBACK_PASSAGE_VERB = r"(?:pass(?:es|ed)?|walk(?:s|ed)?|step(?:s|ped)?|cross(?:es|ed)?)"
_FALLBACK_DOOR_PHRASE = (
    r"(?:(?:[A-Za-z][A-Za-z’'-]*\s+){0,3})(?:door|gate|hatch)"
)
_FALLBACK_ROUTE_CONNECTOR = re.compile(
    r"\b(?:and|or|then|before|after|while|as|but|because|who|which|that|toward|towards)\b|"
    r"\b[A-Za-z][A-Za-z’'-]*(?:ing|ed)\b|\b(?:door|gate|hatch)\b",
    re.IGNORECASE,
)
_FALLBACK_APPROACH_AND_PASSAGE = re.compile(
    rf"^\s*(?P<subject>{_FALLBACK_SUBJECT})\s+"
    rf"(?P<approach>{_FALLBACK_APPROACH_VERB})\s+"
    r"(?P<route>[^,.!?;]+?)\s+and\s+"
    rf"(?P<passage>{_FALLBACK_PASSAGE_VERB})\s+through\s+"
    r"(?:the|same|that|this)\s+(?P<initial_state>closed|shut)\s+"
    rf"(?P<door>{_FALLBACK_DOOR_PHRASE})\s*[.!?]*$",
    re.IGNORECASE,
)
_FALLBACK_OPEN_THEN_ENTRY = re.compile(
    rf"^\s*(?P<opener>[A-Z][A-Za-z’'-]*)\s+"
    r"(?P<open_verb>opens|opened)\s+"
    rf"(?:the|same|that|this)\s+(?P<door>{_FALLBACK_DOOR_PHRASE})\s*"
    r"(?:,\s*|\s+and\s+)"
    r"(?P<traveler>[A-Z][A-Za-z’'-]*)\s+"
    r"(?P<entry_verb>enters|entered)\s+first\s*[.!?]*$",
    re.IGNORECASE,
)


def _capitalized_sentence_subject(value: str) -> str:
    value = value.strip()
    if value.casefold() == "they":
        return "They"
    return value[:1].upper() + value[1:]


def _fallback_group_matches(
    group: Mapping[str, Any],
    source_events: Sequence[Mapping[str, Any]],
    validated_relationships: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Recompute one group and require exact IDs, actors, and source evidence."""

    source_ids = group.get("source_event_ids")
    if not isinstance(source_ids, list) or len(source_ids) != 2:
        return None
    normalized_ids = [str(value or "").strip().upper() for value in source_ids]
    if any(not value for value in normalized_ids):
        return None
    recomputed = [
        candidate for candidate in propose_h3_transport_enabler_groups(
            source_events, validated_relationships,
        )
        if candidate.get("source_event_ids") == normalized_ids
        and candidate.get("overview_event_id")
        == str(group.get("overview_event_id") or "").strip().upper()
    ]
    if len(recomputed) != 1:
        return None
    candidate = recomputed[0]
    for key in (
        "relation_type", "overview_event_id", "source_event_ids", "threshold",
        "opener", "traveler", "source_evidence",
    ):
        supplied = group.get(key)
        expected = candidate.get(key)
        if key in {"overview_event_id"}:
            supplied = str(supplied or "").strip().upper()
        if supplied != expected:
            return None
    return candidate


def h3_same_passage_enabler_fallback_action(
    group: Mapping[str, Any],
    source_events: Sequence[Mapping[str, Any]],
    validated_relationships: Sequence[Mapping[str, Any]],
) -> str | None:
    """Compile a narrow, source-verified opening-before-passage fallback.

    The normal source order can mention passage through a closed threshold
    before the next event explains how it is opened. For the exact supported
    grammar, keep the approach, move only the opening clause ahead of the one
    passage, and bind the named traveler's first-entry detail to that same
    passage. Unsupported or altered evidence returns ``None`` so callers can
    fail closed instead of emitting the physically contradictory source order.
    """

    if not isinstance(group, Mapping):
        return None
    verified = _fallback_group_matches(group, source_events, validated_relationships)
    if verified is None:
        return None
    event_by_id = {
        str(event.get("event_id") or "").strip().upper(): event
        for event in source_events if isinstance(event, Mapping)
    }
    summary_id, detail_id = verified["source_event_ids"]
    summary = event_by_id.get(summary_id, {})
    detail = event_by_id.get(detail_id, {})
    if _event_is_timed(summary) or _event_is_timed(detail):
        return None
    if _event_has_dialogue(summary) or _event_has_dialogue(detail):
        return None

    summary_text = " ".join(str(summary.get("text") or "").split())
    detail_text = " ".join(str(detail.get("text") or "").split())
    route = _FALLBACK_APPROACH_AND_PASSAGE.fullmatch(summary_text)
    actions = _FALLBACK_OPEN_THEN_ENTRY.fullmatch(detail_text)
    if not route or not actions:
        return None
    route_text = route.group("route").strip()
    if not route_text or _FALLBACK_ROUTE_CONNECTOR.search(route_text):
        return None

    if (
        actions.group("opener").casefold() != verified["opener"].casefold()
        or actions.group("traveler").casefold() != verified["traveler"].casefold()
    ):
        return None

    threshold = str(verified.get("threshold") or "").strip()
    threshold_mentions = _threshold_mentions("the " + threshold)
    summary_mentions = _threshold_mentions("the " + route.group("initial_state") + " " + route.group("door"))
    detail_mentions = _threshold_mentions("the " + actions.group("door"))
    if (
        not threshold
        or len(threshold_mentions) != 1
        or len(summary_mentions) != 1
        or len(detail_mentions) != 1
        or threshold_mentions[0][0] != summary_mentions[0][0]
        or threshold_mentions[0][0] != detail_mentions[0][0]
    ):
        return None

    subject = _capitalized_sentence_subject(route.group("subject"))
    approach = route.group("approach").lower()
    passage = route.group("passage").lower()
    opener = actions.group("opener")
    open_verb = actions.group("open_verb").lower()
    traveler = actions.group("traveler")
    approach_clause = (
        f"{subject} {approach} {route_text} toward the closed {threshold}."
    )
    opening_clause = f"{opener} {open_verb} the {threshold}."
    passage_clause = (
        f"{subject} {passage} through the now-open {threshold}, "
        f"with {traveler} entering first."
    )
    return " ".join((approach_clause, opening_clause, passage_clause))


def co_locate_h3_enabler_beats(
    beats: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    source_events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Co-locate a proven adjacent enabler pair without retaining stale staging.

    Invalid, duplicate, timed, dialogue-bearing, or non-adjacent assignments
    are returned unchanged. Cross-window pairs move together to the later
    segment; the earlier slot becomes a neutral hold. Original model staging
    is preserved only in the returned audit receipts.
    """
    if any(not isinstance(beat, Mapping) for beat in beats):
        return [deepcopy(beat) for beat in beats], []  # type: ignore[list-item]
    result = [deepcopy(dict(beat)) for beat in beats]
    event_list = [dict(event) for event in source_events if isinstance(event, Mapping)]
    if len(event_list) != len(source_events):
        return result, []
    event_ids = [str(event.get("event_id") or "").strip().upper() for event in event_list]
    if any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids):
        return result, []
    event_by_id = {
        str(event.get("event_id") or "").strip().upper(): event
        for event in event_list
    }
    source_positions = {
        str(event.get("event_id") or "").strip().upper(): index
        for index, event in enumerate(event_list)
    }
    receipts: list[dict[str, Any]] = []
    consumed: set[str] = set()
    changed = False

    for group in groups:
        if not isinstance(group, Mapping) or group.get("relation_type") != "same_passage_enabler":
            continue
        values = [str(value or "").strip().upper() for value in group.get("source_event_ids", [])]
        if len(values) != 2 or not all(values) or len(set(values)) != 2:
            continue
        summary_id, detail_id = values
        if summary_id in consumed or detail_id in consumed:
            continue
        if summary_id not in event_by_id or detail_id not in event_by_id:
            continue
        if source_positions[detail_id] != source_positions[summary_id] + 1:
            continue
        evidence = group.get("source_evidence")
        if not isinstance(evidence, Mapping):
            continue
        summary_evidence = evidence.get("summary")
        detail_evidence = evidence.get("enabler")
        if (
            not isinstance(summary_evidence, Mapping)
            or not isinstance(detail_evidence, Mapping)
            or str(summary_evidence.get("event_id") or "").strip().upper() != summary_id
            or str(detail_evidence.get("event_id") or "").strip().upper() != detail_id
            or summary_evidence.get("exact_text") != event_by_id[summary_id].get("text")
            or detail_evidence.get("exact_text") != event_by_id[detail_id].get("text")
        ):
            continue
        if _event_is_timed(event_by_id[summary_id]) or _event_is_timed(event_by_id[detail_id]):
            continue
        if _event_has_dialogue(event_by_id[summary_id]) or _event_has_dialogue(event_by_id[detail_id]):
            continue

        event_assignments: dict[str, list[int]] = {summary_id: [], detail_id: []}
        for beat_index, beat in enumerate(result):
            ids = [str(value or "").strip().upper() for value in beat.get("source_event_ids", [])]
            for event_id in values:
                if event_id in ids:
                    event_assignments[event_id].append(beat_index)
        if all(len(event_assignments[value]) == 1 for value in values):
            first_index = event_assignments[summary_id][0]
            second_index = event_assignments[detail_id][0]
            if first_index == second_index:
                beat = result[first_index]
                if [str(value or "").strip().upper() for value in beat.get("source_event_ids", [])] != values:
                    continue
                if _beat_has_dialogue(beat) or _beat_is_timed(beat):
                    continue
                try:
                    segment = int(beat.get("segment"))
                except (TypeError, ValueError):
                    continue
                beat["_source_enabler_groups"] = [deepcopy(dict(group))]
                receipts.append({
                    "relation_type": "same_passage_enabler",
                    "source_event_ids": values,
                    "operation": "already_grouped",
                    "segment": segment,
                    "source_evidence": deepcopy(dict(evidence)),
                })
                consumed.update(values)
                changed = True
                continue
            if second_index != first_index + 1:
                continue
            summary_beat, detail_beat = result[first_index], result[second_index]
            summary_ids = [str(value or "").strip().upper() for value in summary_beat.get("source_event_ids", [])]
            detail_ids = [str(value or "").strip().upper() for value in detail_beat.get("source_event_ids", [])]
            if summary_ids != [summary_id] or detail_ids != [detail_id]:
                continue
            if _beat_has_dialogue(summary_beat) or _beat_has_dialogue(detail_beat):
                continue
            if _beat_is_timed(summary_beat) or _beat_is_timed(detail_beat):
                continue
            try:
                first_segment = int(summary_beat.get("segment"))
                second_segment = int(detail_beat.get("segment"))
            except (TypeError, ValueError):
                continue
            if second_segment not in {first_segment, first_segment + 1}:
                continue

            receipt = {
                "relation_type": "same_passage_enabler",
                "source_event_ids": values,
                "operation": "co_located",
                "from_segments": [first_segment, second_segment],
                "to_segment": second_segment,
                "source_evidence": deepcopy(dict(evidence)),
                "staging_provenance": [
                    {"source_event_id": summary_id, "original_beat": deepcopy(summary_beat)},
                    {"source_event_id": detail_id, "original_beat": deepcopy(detail_beat)},
                ],
            }
            threshold = str(group.get("threshold") or "the threshold")
            if first_segment == second_segment:
                result.pop(first_index)
                second_index -= 1
            else:
                result[first_index] = {
                    "beat_id": summary_beat.get("beat_id"),
                    "segment": first_segment,
                    "source_event_ids": [],
                    "dialogue_ids": [],
                    "description": (
                        "Hold the composition from the previous accepted frame. Keep all people and "
                        "props at their existing positions with the same ownership; do not advance the state."
                    ),
                    "state_after": (
                        "Unchanged from the previous accepted visible state: retain existing person, "
                        "threshold, and prop positions and ownership."
                    ),
                    "_source_enabler_connector": {
                        "relation_type": "same_passage_enabler",
                        "source_event_ids": values,
                        "threshold_identity": threshold,
                        "instruction": "Preserve the previous accepted composition and ownership; do not advance the linked state in this connector.",
                    },
                }
            joined = {
                "beat_id": detail_beat.get("beat_id"),
                "segment": second_segment,
                "source_event_ids": values,
                "dialogue_ids": [],
                "description": (
                    f"At the {threshold}, the assigned opener action precedes the assigned "
                    "traveler's passage. Preserve both separate source events and their actors."
                ),
                "_source_enabler_groups": [deepcopy(dict(group))],
            }
            result[second_index] = joined
            receipts.append(receipt)
            consumed.update(values)
            changed = True
        # An event duplicated across beats or mixed with other assigned source
        # IDs is intentionally left untouched.

    if changed:
        for index, beat in enumerate(result, start=1):
            beat["beat_id"] = f"B{index}"
    return result, receipts


def _beat_has_dialogue(beat: Mapping[str, Any]) -> bool:
    return any(beat.get(field) for field in ("dialogue_ids", "dialogue", "locked_dialogue", "dialogue_anchor"))


def _beat_is_timed(beat: Mapping[str, Any]) -> bool:
    return any(beat.get(field) is not None for field in (
        *_EVENT_TIMING_FIELDS, "authored_time", "start_frame", "end_frame",
    ))
