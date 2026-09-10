"""Read explicit structure in imported video briefs without rewriting their prose.

Timestamped shot blocks are authored story units, not a bag of sentences. Keep
their camera, choreography and timing together; production notes are context.
This module uses only explicit labels, never guesses characters from capitals.
"""

from __future__ import annotations

from functools import lru_cache
import re


_CLOCK = r"(?:\d{1,2}:)?\d{1,3}(?:\.\d+)?"
_TIMED_HEADING = re.compile(
    rf"(?:【|\[|\()\s*(?P<start>{_CLOCK})\s*(?:s|sec(?:onds?)?)?\s*"
    rf"[-–—]\s*(?P<end>{_CLOCK})\s*(?:s|sec(?:onds?)?)?\s*(?:】|\]|\))",
    re.IGNORECASE,
)
_NOTES_LABEL = (
    r"(?:(?:vfx|cinematography|negative(?:\s+prompt)?|constraints|requirements)"
    r"(?:\s+(?:requirements|direction|notes|design))?|"
    r"(?:camera|visuals?|sound|audio|style|lighting)\s+(?:requirements|direction|notes|design))"
)
_NOTES_HEADING = re.compile(
    rf"(?:【|\[)\s*{_NOTES_LABEL}\s*(?:】|\])|"
    rf"(?:^|\n)\s*(?:#{{1,4}}\s*)?{_NOTES_LABEL}\s*:",
    re.IGNORECASE,
)
_PROFILE = re.compile(
    r"\b(?P<name>(?:Character|Subject|Actor)\s+(?:[A-Z](?![a-z])|\d+))"
    r"\s*(?:\((?P<details>[^)\r\n]{1,160})\))?\s*:\s*",
)


def _seconds(value: str) -> float:
    parts = value.split(":")
    return float(parts[-1]) + (60 * float(parts[-2]) if len(parts) > 1 else 0)


def explicit_character_profiles(source: str) -> list[dict[str, str]]:
    """Recognize inline or multiline Character A / Subject 1 definitions."""
    result = []
    seen = set()
    for match in _PROFILE.finditer(source):
        name = match.group("name")
        if name in seen:
            continue
        seen.add(name)
        result.append({"name": name, "details": (match.group("details") or "").strip()})
    return result


@lru_cache(maxsize=8)
def _timed_brief(source: str) -> tuple[str, tuple[tuple[float, float, str, int, int], ...]]:
    matches = list(_TIMED_HEADING.finditer(source))
    if len(matches) < 2:
        return "", ()
    ranges = [(_seconds(item.group("start")), _seconds(item.group("end"))) for item in matches]
    if any(end <= start for start, end in ranges) or any(
        start < ranges[index - 1][1] for index, (start, _end) in enumerate(ranges) if index
    ):
        return "", ()
    # A time range quoted in conversation is not an imported storyboard.
    if ranges[0][0] != 0 or any(
        start - ranges[index - 1][1] > 0.1 for index, (start, _end) in enumerate(ranges) if index
    ):
        return "", ()
    context = [source[:matches[0].start()].strip()]
    blocks = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        raw_body = source[match.end():stop]
        body_offset = match.end() + len(raw_body) - len(raw_body.lstrip(" \\\r\n"))
        body = raw_body.strip(" \\\r\n")
        body_end = stop
        # Notes inside an earlier timed block belong to that block. Only a
        # trailing production-notes section is shared across the sequence.
        note = _NOTES_HEADING.search(body) if index == len(matches) - 1 else None
        if note:
            context.append(body[note.start():].strip())
            body_end = body_offset + note.start()
            body = body[:note.start()].strip()
        if not body:
            return "", ()
        blocks.append((*ranges[index], body, body_offset, body_end))
    return "\n\n".join(item for item in context if item), tuple(blocks)


def authored_timed_brief(source: str) -> dict:
    """Return new containers so callers cannot mutate the cached source parse."""
    context, blocks = _timed_brief(str(source or ""))
    return {
        "context": context,
        "events": [
            {"text": text, "source_start_seconds": start, "source_end_seconds": end,
             "source_offset": offset, "source_end": stop}
            for start, end, text, offset, stop in blocks
        ],
    }


def explicit_negative_constraints(source: str) -> str:
    """Keep authored prohibition clauses as global directions, never as events."""
    # Only sentence-start prohibitions or a clear Throughout clause qualify;
    # incidental narration such as "he finds no key" is not a global rule.
    # A flattened paste can put a notes heading directly after a prohibition.
    # Give explicit section labels a boundary so their prose is not mistaken
    # for part of that prohibition.
    source = _NOTES_HEADING.sub("\n", source)
    clauses = re.split(r"(?<=[.!?])\s+|[\r\n]+", source)
    return " ".join(dict.fromkeys(
        item.strip() for item in clauses
        if re.match(r"\s*(?:Throughout\s*,\s*)?(?:no|never|without)\b", item, re.IGNORECASE)
    ))
