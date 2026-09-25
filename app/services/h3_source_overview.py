"""Conservative recognition of an introductory spatial-transport synopsis.

This helper uses only immutable source-event text and metadata. It proposes a
relationship without removing, merging, or renumbering any source event.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_TRANSPORT = re.compile(
    r"\b(?:move|moves|moved|carry|carries|carried|take|takes|took|bring|brings|"
    r"brought|transport|transports|transported)\b\s+"
    r"(?P<prop>.+?)\s+from\s+(?P<origin>.+?)\s+(?:to|into)\s+"
    r"(?P<destination>.+?)"
    r"(?=\s+(?:so|for|because)\b|\s+in\s+order\s+to\b|[,;.!?]|$)",
    re.IGNORECASE,
)
_ROUTE_WORD = re.compile(
    r"\b(?:cross(?:es|ed)?|pass(?:es|ed)?\s+through|travel(?:s|ed)?|"
    r"walk(?:s|ed)?\s+(?:along|through|across)|move(?:s|d)?\s+through|"
    r"leave(?:s|d)?|depart(?:s|ed)?|exit(?:s|ed)?)\b",
    re.IGNORECASE,
)
_ARRIVAL_WORD = re.compile(
    r"\b(?:reach(?:es|ed)?|arrive(?:s|d)?|enter(?:s|ed)?|"
    r"step(?:s|ped)?\s+(?:into|inside)|walk(?:s|ed)?\s+into|"
    r"go(?:es)?\s+into|went\s+into)\b",
    re.IGNORECASE,
)
_FAILED_ATTEMPT = re.compile(
    r"\b(?:try|tries|attempt|attempts|fail|fails|failed)\s+to|"
    r"\b(?:cannot|can't|could\s+not|unable\s+to|unsuccessfully)\b",
    re.IGNORECASE,
)
_ABANDONED_PROP_PREFIX = re.compile(
    r"\b(?:leave|leaves|left|drop|drops|dropped|abandon|abandons|abandoned)\b",
    re.IGNORECASE,
)
_NAME = re.compile(r"\b[A-Z][a-z]+(?:[’'-][A-Z]?[a-z]+)*\b")
_IGNORED_CAPITALS = {
    "a", "an", "the", "one", "two", "three", "four", "five", "six",
    "at", "only", "end", "then", "when", "while", "after", "before",
    "but", "and", "or", "once", "later", "finally", "initially",
}
_COLOR = re.compile(
    r"\b(?:red|orange|yellow|green|blue|violet|purple|pink|black|white|"
    r"brown|gray|grey|gold|golden|silver)\b",
    re.IGNORECASE,
)
_REPEAT = re.compile(
    r"\b(?:again|another\s+(?:trip|journey|crossing)|second\s+(?:trip|journey)|"
    r"return(?:s|ed)?\s+(?:to|back)|go(?:es|t)?\s+back|head(?:s|ed)?\s+back)\b",
    re.IGNORECASE,
)
_ALLOWED_PURPOSE = re.compile(
    r"^(?:so\s+(?:they|he|she|it|[A-Z][a-z]+)\s+can|in\s+order\s+to|for)\b"
    r"[^;.!?]{0,100}$",
    re.IGNORECASE,
)
_TIMING_FIELDS = (
    "source_start_seconds", "source_end_seconds", "start_seconds", "end_seconds",
    "timecode", "time_range", "source_time_range",
)
_PERSON_NAME = r"[A-Z][a-z]+(?:[’'-][A-Z]?[a-z]+)*"
_SUBJECT_GROUP = re.compile(
    rf"^\s*{_PERSON_NAME}(?:\s+(?:and|&)\s+{_PERSON_NAME})*\s*$"
)
_ROLE_NOUNS = {
    "adult", "adults", "person", "people", "volunteer", "volunteers", "staff",
    "member", "members", "partner", "partners", "worker", "workers", "artist",
    "artists", "curator", "curators", "guide", "guides", "courier", "couriers",
    "visitor", "visitors", "performer", "performers", "friend", "friends", "student",
    "students", "teacher", "teachers", "parent", "parents", "child", "children",
    "sibling", "siblings", "crew", "team", "group", "pair", "duo", "helper",
    "helpers", "technician", "technicians", "attendant", "attendants", "participant",
    "participants", "companion", "companions", "observer", "observers", "exhibitor",
    "exhibitors", "volunteer", "volunteers", "colleague", "colleagues",
}
_APPOSITIVE_CLAUSE = re.compile(
    r"\b(?:then|before|after|while|when|because|since|until|once|who|whom|whose|"
    r"which|that|where|although|unless)\b",
    re.IGNORECASE,
)
_APPOSITIVE_PREFIX_WORDS = {
    "a", "an", "the", "one", "two", "three", "four", "five", "six", "both",
    "several", "adult",
}


def recognize_introductory_transport_overview(
    source_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return a source-supported overview proposal, or abstain.

    The accepted form is intentionally narrow: an untimed, non-dialogue first
    event states a unique prop's route; the next event explicitly resets the
    chronology with ``At the start``; later source events prove an ordered
    passage from the origin and a successful arrival at the destination.
    """
    if len(source_events) < 4:
        return None

    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in source_events:
        if not isinstance(raw, Mapping):
            return None
        event_id = str(raw.get("event_id") or "").strip().upper()
        text = " ".join(str(raw.get("text") or "").split())
        if not event_id or not text or event_id in seen_ids:
            return None
        seen_ids.add(event_id)
        events.append({**dict(raw), "event_id": event_id, "text": text})

    overview = events[0]
    overview_text = overview["text"]
    if any(overview.get(key) is not None for key in _TIMING_FIELDS):
        return None
    if any(overview.get(key) for key in ("dialogue_ids", "locked_dialogue", "dialogue_anchor")):
        return None
    if re.match(r"^\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|[A-Z]{2,})\s*:\s*", overview_text):
        return None
    if re.search(r"[\"“”‘’]", overview_text) or re.search(
        r"\b(?:says?|said|asks?|asked|speaks?|spoke|dialogue|line of dialogue)\b",
        overview_text,
        re.IGNORECASE,
    ):
        return None
    if _FAILED_ATTEMPT.search(overview_text):
        return None

    synopsis_matches = list(_TRANSPORT.finditer(overview_text))
    if len(synopsis_matches) != 1:
        return None
    synopsis = synopsis_matches[0]
    prop_phrase = synopsis.group("prop").strip(" \t,;:-")
    origin = synopsis.group("origin").strip(" \t,;:-")
    destination = synopsis.group("destination").strip(" \t,;:-")
    if not prop_phrase or not origin or not destination:
        return None
    actor_prefix = overview_text[:synopsis.start()]
    participants = _parse_subject_prefix(actor_prefix)
    if not participants:
        return None
    tail = overview_text[synopsis.end():].strip(" \t,;:.!?-")
    if tail and not _ALLOWED_PURPOSE.fullmatch(tail):
        return None
    if not re.match(r"^(?:one|single|a|an|the)\b", prop_phrase, re.IGNORECASE):
        return None
    prop_words = re.findall(r"[A-Za-z]+", prop_phrase)
    if len(prop_words) < 2:
        return None
    prop_head = prop_words[-1].casefold()
    origin_terms = _place_terms(origin)
    destination_terms = _place_terms(destination)
    if not origin_terms or not destination_terms or origin_terms == destination_terms:
        return None

    # The reset must be adjacent to the overview. This avoids treating an
    # unrelated later trip as the promised detail of an opening synopsis.
    if not re.match(r"^At the start\b", events[1]["text"], re.IGNORECASE):
        return None
    if any(_FAILED_ATTEMPT.search(event["text"]) for event in events[1:]):
        return None

    origin_indexes: list[int] = []
    arrival_indexes: list[int] = []
    for index in range(1, len(events)):
        text = events[index]["text"]
        tokens = _tokens(text)
        if index >= 2 and origin_terms.issubset(tokens) and _ROUTE_WORD.search(text):
            origin_indexes.append(index)
        if index >= 2 and destination_terms.issubset(tokens) and _ARRIVAL_WORD.search(text):
            arrival_indexes.append(index)
        if _has_conflicting_prop(text, prop_head, prop_phrase):
            return None
        if _has_second_prop(text, prop_head):
            return None
        if _has_conflicting_route(text, origin_terms, destination_terms):
            return None

    ordered_route = next(
        (
            (origin_index, arrival_index)
            for origin_index in origin_indexes
            for arrival_index in arrival_indexes
            if origin_index < arrival_index
        ),
        None,
    )
    if ordered_route is None:
        return None
    origin_index, arrival_index = ordered_route
    route_events = events[1:arrival_index + 1]
    route_text = " ".join(event["text"] for event in route_events)

    # Named participants must occur in the source-supported trip itself, and
    # the relevant clauses may not introduce a different named actor.
    if any(not re.search(rf"\b{re.escape(name)}\b", route_text) for name in participants):
        return None
    route_names = _action_subject_names(route_text)
    if route_names - set(participants):
        return None

    # The prop must reach the destination, not merely appear in an opening
    # hold before the people travel without it. Reject explicit abandonment
    # and require a definite prop reference attached to the arrival evidence.
    if any(_abandons_prop(event["text"], prop_head) for event in route_events):
        return None
    arrival_text = events[arrival_index]["text"]
    if (not _prop_present_at_arrival(arrival_text, prop_head)
            or _has_unlisted_carrier(arrival_text, prop_head)):
        return None

    arrival_subjects = _action_subject_names(arrival_text)
    if arrival_subjects - set(participants):
        return None
    named_arrival = bool(arrival_subjects & set(participants))
    collective_arrival = bool(re.search(r"\bthey\b", arrival_text, re.IGNORECASE))
    if not (named_arrival or collective_arrival):
        return None
    if collective_arrival and not all(
        re.search(rf"\b{re.escape(name)}\b", route_text, re.IGNORECASE)
        or any(
            re.search(rf"\b{re.escape(name)}\b", event["text"], re.IGNORECASE)
            for event in route_events
        )
        for name in participants
    ):
        return None

    # A later return or repeated source-to-destination trip keeps the synopsis
    # executable, even if its first passage was otherwise fully described.
    for event in events[arrival_index + 1:]:
        text = event["text"]
        if _has_conflicting_prop(text, prop_head, prop_phrase):
            return None
        if _has_second_prop(text, prop_head):
            return None
        if _REPEAT.search(text) and re.search(rf"\b{re.escape(prop_head)}\b", text, re.IGNORECASE):
            return None
        if _has_conflicting_route(text, origin_terms, destination_terms):
            return None
        if _TRANSPORT.search(text) and re.search(
            rf"\b{re.escape(prop_head)}\b", text, re.IGNORECASE,
        ) and origin_terms.issubset(_tokens(text)):
            return None

    detail_events = events[1:arrival_index + 1]
    return {
        "relation_id": f"OVERVIEW_{overview['event_id']}",
        "relation_type": "overview_of",
        "overview_event_id": overview["event_id"],
        "detail_event_ids": [event["event_id"] for event in detail_events],
        "coverage_status": "complete",
        "invariant_context": {
            "participants": participants,
            "unique_prop": prop_phrase,
        },
        "route_evidence": {
            "origin": origin,
            "destination": destination,
            "origin_event_ids": [events[origin_index]["event_id"]],
            "arrival_event_ids": [events[arrival_index]["event_id"]],
        },
        "source_evidence": {
            "overview": {
                "event_id": overview["event_id"],
                "exact_text": overview_text,
            },
            "details": [
                {"event_id": event["event_id"], "exact_text": event["text"]}
                for event in detail_events
            ],
        },
    }


def _parse_subject_prefix(text: str) -> list[str] | None:
    """Accept only a name group and simple role appositives before the verb."""
    chunks = [chunk.strip() for chunk in text.strip(" \t,;").split(",")]
    if not 1 <= len(chunks) <= 3 or any(not chunk for chunk in chunks):
        return None
    subject_indexes = [index for index, chunk in enumerate(chunks) if _SUBJECT_GROUP.fullmatch(chunk)]
    if len(subject_indexes) != 1:
        return None
    subject_index = subject_indexes[0]
    if any(
        not _is_identity_appositive(chunk)
        for index, chunk in enumerate(chunks)
        if index != subject_index
    ):
        return None
    subject_text = chunks[subject_index]
    names = re.split(r"\s+(?:and|&)\s+", subject_text)
    return names if names and len(set(names)) == len(names) else None


def _is_identity_appositive(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    if not words or words[-1].casefold() not in _ROLE_NOUNS or _APPOSITIVE_CLAUSE.search(text):
        return False
    names = [
        match.group(0) for match in _NAME.finditer(text)
        if match.group(0).casefold() not in _APPOSITIVE_PREFIX_WORDS
    ]
    if names:
        return False
    # Clause-like participles and verb-shaped words are not identity labels.
    return not any(word.casefold().endswith(("ed", "ing")) for word in words[:-1])


def _action_subject_names(text: str) -> set[str]:
    verb = (
        r"(?:holds?|held|cross(?:es|ed)?|pass(?:es|ed)?|opens?|opened|enters?|"
        r"entered|closes?|closed|reaches?|reached|arrives?|arrived|leaves?|left|"
        r"carries?|carried|moves?|moved|takes?|took|brings?|brought|hands?|handed|"
        r"threads?|threaded|ties?|tied|sets?|set|places?|placed|walks?|walked)"
    )
    single = re.compile(rf"\b([A-Z][a-z]+)\s+{verb}\b")
    paired = re.compile(rf"\b([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s+{verb}\b")
    names = {match.group(1) for match in single.finditer(text)}
    names.update(name for match in paired.finditer(text) for name in match.groups())
    return names - _IGNORED_CAPITALS


def _tokens(text: str) -> set[str]:
    words = set()
    for raw in re.findall(r"[A-Za-z]+", text):
        word = raw.casefold()
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is", "ous", "ics")):
            word = word[:-1]
        words.add(word)
    return words


def _place_terms(text: str) -> set[str]:
    return _tokens(re.sub(r"\b(?:a|an|the|this|that|their|our)\b", " ", text, flags=re.I))


def _has_conflicting_prop(text: str, head: str, prop_phrase: str) -> bool:
    phrase_colors = {match.group(0).casefold() for match in _COLOR.finditer(prop_phrase)}
    for match in re.finditer(
        rf"\b((?:[A-Za-z]+\s+){{0,3}}){re.escape(head)}\b", text, re.IGNORECASE,
    ):
        modifiers = match.group(1)
        colors = {item.group(0).casefold() for item in _COLOR.finditer(modifiers)}
        if colors - phrase_colors:
            return True
    return False


def _has_second_prop(text: str, head: str) -> bool:
    return bool(re.search(
        rf"\b(?:another|a second|second|different|extra|additional)\s+"
        rf"(?:[a-z]+\s+){{0,2}}{re.escape(head)}\b",
        text,
        re.IGNORECASE,
    ))


def _has_conflicting_route(
    text: str,
    origin_terms: set[str],
    destination_terms: set[str],
) -> bool:
    for match in re.finditer(
        r"\bfrom\s+(.+?)\s+(?:(?:back|again)\s+)?(?:to|into)\s+"
        r"(.+?)(?=[,;.!?]|$)",
        text,
        re.IGNORECASE,
    ):
        start, end = _place_terms(match.group(1)), _place_terms(match.group(2))
        if destination_terms.issubset(start) and origin_terms.issubset(end):
            return True
        if origin_terms.issubset(start) and end and not destination_terms.issubset(end):
            # An explicitly named alternative endpoint is ambiguous; routes
            # expressed with "through" or "across" are not matched here.
            return True
    return False


def _abandons_prop(text: str, head: str) -> bool:
    head_ref = rf"(?:the|same|this|that)?\s*(?:[a-z]+\s+){{0,2}}{re.escape(head)}"
    if re.search(rf"{_ABANDONED_PROP_PREFIX.pattern}.{{0,48}}\b{head_ref}\b", text, re.I):
        return True
    if re.search(
        rf"\b{re.escape(head)}\b.{{0,24}}\b(?:remains?|stays?)\s+(?:at|in|inside|on)\b",
        text,
        re.I,
    ):
        return True
    if re.search(rf"\bwithout\s+(?:the\s+|same\s+)?{head_ref}\b", text, re.I):
        return True
    return False


def _prop_present_at_arrival(text: str, head: str) -> bool:
    reference = rf"(?:the|same|this|that)\s+(?:[a-z]+\s+){{0,3}}{re.escape(head)}"
    if not re.search(rf"\b{reference}\b", text, re.IGNORECASE):
        return False
    associated_action = re.search(
        rf"\b(?:carry|carries|carried|carrying|hold|holds|holding|held|take|takes|"
        rf"took|bring|brings|brought|hand|hands|handed|pass|passes|passed|move|moves|"
        rf"moved|transport|transports|transported)\b.{{0,64}}\b{reference}\b",
        text,
        re.IGNORECASE,
    )
    with_prop = re.search(rf"\bwith\s+{reference}\b", text, re.IGNORECASE)
    return bool(associated_action or with_prop)


def _has_unlisted_carrier(text: str, head: str) -> bool:
    action = (
        r"(?:carry|carries|carried|carrying|hold|holds|holding|held|take|takes|took|"
        r"bring|brings|brought|hand|hands|handed|pass|passes|passed|move|moves|moved)"
    )
    ref = rf"(?:the|same|this|that)\s+(?:[a-z]+\s+){{0,3}}{re.escape(head)}"
    return bool(re.search(
        rf"\b(?:someone|somebody|(?:a|an|the|another|different|some)\s+"
        rf"(?:[a-z]+\s+){{0,2}}[a-z]+)\s+{action}\b.{{0,64}}\b{ref}\b",
        text,
        re.IGNORECASE,
    ))
