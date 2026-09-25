"""Conservative source-grammar helpers for H3 event contracts.

These helpers annotate syntax without deleting or rewriting the immutable
source events. Uncertain language stays an ordinary action obligation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


_QUOTED = re.compile(r'"[^"\r\n]*"|“[^”\r\n]*”|‘[^’\r\n]*’|`[^`\r\n]*`')
_SINGLE_QUOTED = re.compile(r"(?<!\w)'[^'\r\n]+'(?!\w)")
_PURPOSE = re.compile(
    r"\bso\s+(?:they|he|she|it|we|I|you|"
    r"the\s+(?:pair|two|group|couple|team|volunteers|characters)|"
    r"[A-Z][\w’'-]*(?:\s+and\s+[A-Z][\w’'-]*)?)\s+"
    r"(?:can|could)\b[^.!?;\r\n]*?"
    r"(?=\s*(?:[;.!?\r\n]|,\s*(?:then\b|but\b|and\s+then\b|"
    r"(?:and\s+)?(?:(?i:they|he|she|it|we|I|you|the|a|an)\s+|"
    r"(?-i:[A-Z][\w’'-]*\s+))[A-Z\w’'-]+\b)|$))",
    re.IGNORECASE,
)
_END_WITH = re.compile(r"\bend\s+with\b", re.IGNORECASE)

_EVENT_TERMS = re.compile(
    r"\b(?:approach(?:es|ed|ing)?|arriv(?:e|es|ed|ing)|carry|carries|carried|"
    r"cross(?:es|ed|ing)?|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|"
    r"grab(?:s|bed|bing)?|hand(?:s|ed|ing)?|leave(?:s|d|ing)?|"
    r"lift(?:s|ed|ing)?|lower(?:s|ed|ing)?|move(?:s|d|ing)?|"
    r"opens|opening|closes?|closing|places?|placing|put(?:s|ting)|"
    r"picks?|picking|retrieves?|retrieving|reaches?|reaching|"
    r"returns?|returning|sets?|setting|steps?|stepping|takes?|taken|taking|"
    r"turns?|turning|walks?|walking|threads?|threading|ties?|tying|"
    r"unlocks?|unlocking|locks?|locking|begins?|beginning|starts?|starting|"
    r"finishes?|finishing|completes?|completing|reopens?|reopening|"
    r"smiles?|smiling|nods?|nodding|glances?|glancing|looks?|looking|"
    r"waves?|waving|speaks?|speaking|says|said|laughs?|laughing|"
    r"points?|pointing|examines?|examining|inspects?|inspecting|"
    r"leans?|leaning|bends?|bending|kneels?|kneeling|stands?|sits?)\b",
    re.IGNORECASE,
)
_STATE_ITEM = re.compile(
    r"^(?P<subject>(?-i:[A-Z][\w’'-]*(?:\s+(?-i:[A-Z][\w’'-]*))?)|"
    r"(?i:they|he|she|it|we|both|everyone)|"
    r"(?i:(?:(?:the|a|an)\s+)?(?:[\w’'-]+\s+){0,3}"
    r"(?:door|window|gate|lid|box|case|curtain|banner|spool|print|frame|"
    r"bouquet|lantern|volunteers?|florist|commuter|keeper|performer|"
    r"pair|group|couple|scene|composition|image|objects?|hands?|arms?|"
    r"table|worktable|hook|face|room|end)))\s+"
    r"(?:(?:is|are|remains?|stays?)\s+)?"
    r"(?P<predicate>holding|held|standing|sitting|waiting|resting|remaining|"
    r"staying|hanging|lying|positioned|attached|secured|fastened|tied|"
    r"visible|present|closed|open|locked|unlocked)\b"
    r"(?P<tail>.*)$",
    re.IGNORECASE,
)
_STATE_CONNECTOR = re.compile(
    r"\s+and\s+(?=(?:[A-Z][\w’'-]*|the\b|a\b|an\b|they\b|he\b|"
    r"she\b|it\b|we\b|both\b|everyone\b))"
)
_UNSAFE_STATE_TAIL = re.compile(
    r"\b(?:and|then|before|after|while|because|so|but|although|until)\b|"
    r"\b[\w’'-]+(?:ing|ed)\b",
    re.IGNORECASE,
)
_PRENOMINAL_COORDINATED_PARTICIPLES = re.compile(
    r"\b(?:the|a|an|this|that|these|those|one|same|another)"
    r"[ \t]+(?:[\w’'-]+,[ \t]+)?"
    r"(?P<first>[\w’'-]+(?:ed|en))(?:[ \t]*,[ \t]*)?[ \t]+and[ \t]+"
    r"(?P<second>[\w’'-]+(?:ed|en))"
    r"(?P<tail>(?:[ \t]+[\w’'-]+){3,4})"
    r"(?=[ \t]|[.,;:!?]|$)",
    re.IGNORECASE,
)
_PRENOMINAL_TAIL_FUNCTION_WORD = re.compile(
    r"^(?:the|a|an|this|that|these|those|one|same|another|by|of|in|on|at|"
    r"to|from|with|under|over|behind|before|after|through|across|beside|"
    r"near|during|while|because|then)$",
    re.IGNORECASE,
)

_OCCASION = r"(?:morning|evening|night|day|week|month|year|shift|visit)"
_PURE_TEMPORAL_CLAUSE = re.compile(
    rf"(?:(?:on|by|at|during|in|over|throughout|for)\s+)?"
    rf"(?:(?:the|a|an)\s+)?"
    rf"(?:first|last|final|next|following|previous|earlier|later|"
    rf"each|every|another|one|two|three|four|five|six|seven|eight|nine|"
    rf"ten|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
    rf"{_OCCASION}s?",
    re.IGNORECASE,
)


def _quoted_ranges(text: str) -> list[tuple[int, int]]:
    """Return paired quote ranges, preserving apostrophes in contractions."""
    matches = list(_QUOTED.finditer(text))
    matches.extend(_SINGLE_QUOTED.finditer(text))
    matches.sort(key=lambda match: match.start())
    ranges: list[tuple[int, int]] = []
    last_end = -1
    for match in matches:
        if match.start() >= last_end:
            ranges.append((match.start(), match.end()))
            last_end = match.end()
    return ranges


def _blank_ranges(text: str, ranges: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in ranges:
        for index in range(start, end):
            if chars[index] not in "\r\n":
                chars[index] = " "
    return "".join(chars)


def _quoted_mask(text: str) -> str:
    return _blank_ranges(text, _quoted_ranges(text))


def mask_modal_purpose(text: object) -> str:
    """Blank ``so <person> can/could ...`` purpose clauses in a copy.

    The returned string is the same length as the input and keeps line breaks
    and later clauses intact. The original input is never modified.
    """
    original = str(text or "")
    scan = _quoted_mask(original)
    ranges: list[tuple[int, int]] = []
    for match in _PURPOSE.finditer(scan):
        # Keep the comma/semicolon/sentence boundary itself and all following
        # action text. This avoids consuming a later finite clause.
        end = match.end()
        while end > match.start() and scan[end - 1].isspace():
            end -= 1
        ranges.append((match.start(), end))
    return _blank_ranges(original, ranges)


def mask_prenominal_participial_properties(text: object) -> str:
    """Blank coordinated prenominal participle properties in a same-length copy.

    The intentionally narrow form requires a determiner, a coordinated pair
    of ``-ed``/``-en`` modifiers, and a multiword noun description after them
    (for example, ``the solid, closed, and locked dark wood archive door``).
    This catches descriptive properties before a head noun while abstaining on
    finite closures, post-nominal participial clauses, resultative
    complements, and ambiguous short phrases. Quoted text is left untouched.
    """
    original = str(text or "")
    scan = _quoted_mask(original)
    ranges: list[tuple[int, int]] = []
    for match in _PRENOMINAL_COORDINATED_PARTICIPLES.finditer(scan):
        tail_words = re.findall(r"[\w’'-]+", match.group("tail"))
        if not tail_words or _PRENOMINAL_TAIL_FUNCTION_WORD.fullmatch(tail_words[0]):
            continue
        ranges.extend((match.start(group), match.end(group)) for group in ("first", "second"))
    return _blank_ranges(original, ranges)


def _end_with_scopes(text: str) -> list[tuple[int, int, int]]:
    """Return (marker start, content start, content end) for unquoted scopes."""
    masked = _quoted_mask(text)
    scopes: list[tuple[int, int, int]] = []
    for marker in _END_WITH.finditer(masked):
        end_match = re.search(r"[.!?;\r\n]", masked[marker.end():])
        content_end = marker.end() + end_match.start() if end_match else len(text)
        scopes.append((marker.start(), marker.end(), content_end))
    return scopes


def _classify_state_content(content: str) -> str | None:
    clean = content.strip(" \t\r\n,;:-–—")
    if not clean:
        return None
    if _EVENT_TERMS.search(clean):
        return "eventive"

    # Each comma/conjoined item must independently fit a small state form.
    # This intentionally leaves unfamiliar or mixed constructions untagged.
    comma_items = re.split(r"\s*,\s*", clean)
    state_items: list[str] = []
    for item in comma_items:
        item = re.sub(r"^\s*(?:and\s+)?", "", item, flags=re.IGNORECASE).strip()
        if not item:
            continue
        state_items.extend(_STATE_CONNECTOR.split(item))
    if not state_items:
        return None
    for item in state_items:
        item = item.strip(" \t,;:-–—")
        match = _STATE_ITEM.fullmatch(item)
        if not match:
            return None
        predicate = match.group("predicate").casefold()
        if predicate in {"closed", "open", "locked", "unlocked", "tied", "secured", "attached", "fastened", "positioned"} and not re.search(
            r"\b(?:door|window|gate|lid|box|case|curtain|banner|spool|print|frame|bouquet|lantern|table|worktable|hook|room|end)\b",
            match.group("subject"),
            re.IGNORECASE,
        ):
            return None
        tail = match.group("tail").strip()
        if _UNSAFE_STATE_TAIL.search(tail):
            return None
        # Do not accept a known action hidden in a permissive noun/preposition
        # tail (e.g. “holding a spool and shooting Ron”).
        if _EVENT_TERMS.search(tail):
            return None
    return "final_state"


def classify_end_with_scope(text: object) -> str | None:
    """Classify one explicit, unquoted ``End with`` sentence suffix.

    Returns ``"final_state"`` for a recognized purely stative ending,
    ``"eventive"`` when the scoped phrase clearly directs an action, and
    ``None`` for absent, mixed, or uncertain scopes.
    """
    source = str(text or "")
    scopes = _end_with_scopes(source)
    if len(scopes) != 1:
        return None
    _, content_start, content_end = scopes[0]
    content = _quoted_mask(source)[content_start:content_end]
    return _classify_state_content(content)


def annotate_final_state_events(
    events: Iterable[Mapping[str, Any]], *, source_text: object,
) -> list[dict[str, Any]]:
    """Copy events and tag wholly scoped, uniquely located stative endings.

    The exact original source is required because split event text may have
    lost the punctuation that bounds an ``End with`` sentence. An event is
    tagged only when its unchanged text occurs once in the original source,
    falls wholly after the marker and before that sentence boundary, is not
    quoted, and is itself a recognized state phrase.
    """
    source = str(source_text or "")
    scopes = _end_with_scopes(source)
    quoted = _quoted_ranges(source)
    masked = _quoted_mask(source)
    copied = [dict(event) if isinstance(event, Mapping) else {"text": str(event)} for event in events]

    for event in copied:
        raw = event.get("text")
        if not isinstance(raw, str) or not raw.strip():
            continue
        event_text = raw.strip()
        positions: list[int] = []
        cursor = 0
        while True:
            found = source.find(event_text, cursor)
            if found < 0:
                break
            positions.append(found)
            cursor = found + 1
        if len(positions) != 1:
            continue
        start = positions[0]
        end = start + len(event_text)
        if any(start < quote_end and end > quote_start for quote_start, quote_end in quoted):
            continue
        matching_scopes = [
            (marker_start, content_start, content_end)
            for marker_start, content_start, content_end in scopes
            if start >= marker_start and end <= content_end
        ]
        if len(matching_scopes) != 1:
            continue
        _, content_start, _ = matching_scopes[0]
        _, _, scope_end = matching_scopes[0]
        complete_scope = masked[content_start:scope_end]
        if _classify_state_content(complete_scope) != "final_state":
            continue
        scoped_text = event_text
        if start <= content_start < end:
            prefix_len = content_start - start
            scoped_text = event_text[prefix_len:]
        if not scoped_text.strip():
            continue
        # Ensure the located source span itself is unquoted and corresponds to
        # the exact string used for classification.
        if masked[start:end].strip() != event_text.strip() and start >= content_start:
            continue
        if _classify_state_content(scoped_text) == "final_state":
            event["requirement_kind"] = "final_state"
    return copied


def is_temporal_context_clause(text: object) -> bool:
    """Recognize a pure occasion noun phrase, never a clause with an action."""
    clean = str(text or "").strip(" \t\r\n,;:.!?()[]{}\"“”‘’")
    if not clean:
        return False
    masked = _quoted_mask(clean)
    if masked != clean:
        return False
    return bool(_PURE_TEMPORAL_CLAUSE.fullmatch(clean))
