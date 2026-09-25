"""Small, conservative helpers for explicitly spaced story-event recurrence.

These helpers do not infer recurrence from repeated wording, a final-night
reference, or an ordinary "again". Callers should pass the immutable source
event catalog and the local visual action for the connective beat being
checked.
"""

from __future__ import annotations

import re
from copy import deepcopy
from collections.abc import Iterable, Mapping
from typing import Any


_OCCASION = r"(?:morning|evening|night|day|week|month|year|shift|visit)"
_OCCASION_RE = re.compile(_OCCASION, re.IGNORECASE)
_LATER_ORDINAL = r"(?:second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
_QUOTED_SPANS = re.compile(r'"[^"\r\n]*"|“[^”\r\n]*”|‘[^’\r\n]*’|`[^`\r\n]*`')
_STRAIGHT_SINGLE_QUOTED_SPAN = re.compile(r"(?<!\w)'[^'\r\n]+'(?!\w)")
_EXPLICIT_REPEAT = re.compile(
    rf"\b(?:each|every)(?:\s+and\s+every)?\s+"
    rf"(?:(?:single|other|alternate|alternating)\s+)?"
    rf"{_OCCASION_RE.pattern}s?\b"
    rf"|\b(?P<repeated>{_OCCASION_RE.pattern})s?\s+after\s+"
    rf"(?P=repeated)s?\b",
    re.IGNORECASE,
)
_LATER_OCCASION = re.compile(
    rf"\b(?:"
    rf"(?:(?:next|following|subsequent|another|different|separate|new|"
    rf"successive|consecutive)|distinct\s+(?:next|following|subsequent|"
    rf"another|different|separate|new))\s+{_OCCASION_RE.pattern}s?"
    rf"|(?:(?:the|a)\s+)?{_LATER_ORDINAL}\s+{_OCCASION_RE.pattern}s?"
    rf")\b",
    re.IGNORECASE,
)
_ELAPSED_OCCASIONS = re.compile(
    rf"\b(?:a|one|two|three|four|five|several|a\s+few|few|\d+)\s+"
    rf"{_OCCASION_RE.pattern}s?\s+later\b",
    re.IGNORECASE,
)
_OCCASION_AFTER_OCCASION = re.compile(
    rf"\b(?P<occasion>{_OCCASION_RE.pattern})s?\s+after\s+"
    rf"(?P=occasion)s?\b",
    re.IGNORECASE,
)
_LEADING_TRANSITION = re.compile(
    rf"^\s*(?:(?:on|by|during)\s+)?"
    rf"(?:(?:(?:the|a)\s+)?(?:next|following|subsequent|another|different|"
    rf"separate|new|successive|consecutive)\s+{_OCCASION_RE.pattern}s?"
    rf"|(?:a|one|two|three|four|five|several|a\s+few|few|\d+)\s+"
    rf"{_OCCASION_RE.pattern}s?\s+later"
    rf"|{_OCCASION_RE.pattern}s?\s+after\s+{_OCCASION_RE.pattern}s?)\b",
    re.IGNORECASE,
)
_VISUAL_TIME_CUE = (
    r"(?:"
    # A moving clock hand can show that this is a new time, but only when the
    # same clause connects that visible advancement to a distinct occasion.
    r"\b(?:clock|watch)\b[^.!?;:]{0,90}\b(?:hands?|minute hand|hour hand)\b"
    r"[^.!?;:]{0,60}\b(?:tick(?:s|ing)?\s+forward|"
    r"(?:advance|move|turn)(?:s|d|ing)?\s+forward|"
    r"(?:sweep(?:s|ing)?|swept)(?:\s+\w+){0,3}\s+"
    r"(?:forward|across|past|around|through))\b"
    r"|\b(?:second|minute|hour) hand\b[^.!?;:]{0,60}\b"
    r"(?:tick(?:s|ing)?\s+forward|(?:advance|move|turn)(?:s|d|ing)?\s+forward|"
    r"(?:sweep(?:s|ing)?|swept)(?:\s+\w+){0,3}\s+"
    r"(?:forward|across|past|around|through))\b"
    r"|\bcalendar\b[^.!?;:]{0,60}\b(?:pages?|date)\b"
    r"[^.!?;:]{0,50}\b(?:flip|turn|advance|change|move)(?:s|d|ing)?\b"
    r"|\b(?:daylight|sunlight|lighting|light|sky|shadows?)\b"
    r"[^.!?;:]{0,70}\b(?:shift|fade|change|darken|brighten)(?:s|d|ing)?\b"
    # A filmed transition can be expressed as a noun phrase rather than
    # "the light shifts". The enclosing pattern still requires an explicit
    # link to a distinct later occasion, and planned/narrated cues are excluded.
    r"|\b(?:shift|change)\s+in\b[^.!?;:]{0,70}"
    r"\b(?:lighting|light|glow|shadows?|reflections?)\b"
    r"|\b(?:time[- ]lapse|dissolve|fade|visual transition|transition shot)\b"
    r"[^.!?;:]{0,70}\b(?:jump|move|cut|fade|dissolve|transition|advance)(?:s|d|ing)?\b"
    r")"
)
_VISUAL_LATER_OCCASION = re.compile(
    rf"{_VISUAL_TIME_CUE}[^.!?;:]{{0,100}}\b(?:to|into|through|establishes?|marks?|signals?)\s+"
    rf"(?:(?:the|a|an)\s+)?{_LATER_OCCASION.pattern}\b",
    re.IGNORECASE,
)
_VISUAL_ELAPSED_OCCASIONS = re.compile(
    rf"\b(?:accelerated\s+)?time[- ]lapse\b[^.!?;:]{{0,100}}"
    rf"\b(?:mark|show|indicate|depict|establish|signal)(?:s|ed|ing)?\b"
    rf"[^.!?;:]{{0,80}}\b(?:passage|passing|elapsed)\b"
    rf"[^.!?;:]{{0,40}}\b(?:one|two|three|four|five|several|a\s+few|few|\d+)\s+"
    rf"{_OCCASION_RE.pattern}s?\b",
    re.IGNORECASE,
)
_VISUAL_PASSAGE_OF_TIME = re.compile(
    r"\b(?:(?:slow|brief|gradual)\s+)?"
    r"(?:dissolve|fade|crossfade|wipe|time[- ]lapse|visual transition|"
    r"transition(?: shot)?)\b"
    r"[^.!?;:\r\n]{0,70}\b(?:indicates?|shows?|marks?|signals?|"
    r"depicts?|establishes?|demonstrates?)\b"
    r"[^.!?;:\r\n]{0,70}\b(?:passage\s+of\s+time|passing\s+of\s+time|"
    r"time\s+passing|elapsed\s+time|time\s+has\s+passed)\b",
    re.IGNORECASE,
)
_VISUAL_CROSS_CLAUSE_CUE = re.compile(
    rf"(?:{_VISUAL_PASSAGE_OF_TIME.pattern}|"
    r"\b(?:(?:another|a|the)\s+)?(?:(?:slow|brief|gradual)\s+)?"
    r"(?:dissolve|fade|crossfade|wipe)\s+transition\b)",
    re.IGNORECASE,
)
_VISIBLE_CLOCK_LATER_OCCASION = re.compile(
    rf"\b(?:clock|watch|calendar|date|display|readout)\b"
    rf"[^.!?;\r\n]{{0,100}}\b(?:now\s+)?"
    rf"(?:reads?|shows?|displays?|reveals?|indicates?)\b"
    rf"[^.!?;\r\n]{{0,100}}{_LATER_OCCASION.pattern}",
    re.IGNORECASE,
)
_CROSS_CLAUSE_VISUAL_TIME = re.compile(
    rf"(?P<cue_clause>[^.!?;:\r\n]*?{_VISUAL_CROSS_CLAUSE_CUE.pattern}[^.!?;:\r\n]*)"
    rf"(?:[.!?;][ \t]*(?:\r?\n[ \t]*)?|\r?\n[ \t]*)"
    rf"(?P<occasion_clause>[^.!?;:\r\n]*?{_VISIBLE_CLOCK_LATER_OCCASION.pattern}[^.!?;:\r\n]*)",
    re.IGNORECASE,
)
_NARRATED_TIME_CUE_PREFIX = re.compile(
    r"\b(?:says?|describes?|mentions?|thinks?(?:\s+about|\s+of)?|"
    r"imagines?|dreams?(?:\s+of)?|remembers?|wonders?|plans?|intends?|"
    r"expects?|hopes?|wishes?|talks?\s+about)\b[^.!?;:]{0,90}$",
    re.IGNORECASE,
)
_PLANNED_VISUAL_TIME_CUE = re.compile(
    r"\b(?:plan(?:s|ned|ning)?|intend(?:s|ed|ing)?|hope(?:s|d|ing)?|"
    r"expect(?:s|ed|ing)?|wish(?:es|ed|ing)?|propos(?:e|es|ed|ing))\b"
    r"[^.!?;:]{0,100}\b(?:clock|watch|calendar|time[- ]lapse|lighting|"
    r"daylight|sunlight)\b|"
    r"\b(?:will|would|shall|is\s+going\s+to|are\s+going\s+to)\b"
    r"[^.!?;:]{0,100}\b(?:clock|watch|calendar|time[- ]lapse|lighting|"
    r"daylight|sunlight)\b|"
    r"\b(?:clock|watch|calendar|time[- ]lapse|lighting|daylight|sunlight)\b"
    r"[^.!?;:]{0,80}\b(?:planned|intended|proposed|imagined|will|would|"
    r"is\s+going\s+to|are\s+going\s+to)\b",
    re.IGNORECASE,
)
_PLANNED_VISUAL_TRANSITION = re.compile(
    r"\b(?:plan(?:s|ned|ning)?|intend(?:s|ed|ing)?|hope(?:s|d|ing)?|"
    r"expect(?:s|ed|ing)?|wish(?:es|ed|ing)?|propos(?:e|es|ed|ing)|"
    r"will|would|shall|is\s+going\s+to|are\s+going\s+to)\b",
    re.IGNORECASE,
)
_PLANNED_ACTION_SCOPE = re.compile(
    r"\b(?:plan(?:s|ned|ning)?|intend(?:s|ed|ing)?|hope(?:s|d|ing)?|"
    r"expect(?:s|ed|ing)?|wish(?:es|ed|ing)?|propos(?:e|es|ed|ing)|"
    r"will|would|shall|is\s+going\s+to|are\s+going\s+to|might|could)\b",
    re.IGNORECASE,
)
_STANDALONE_TIME_HEADING = re.compile(
    rf"^\s*(?:(?:the|a)\s+)?(?:next|following|subsequent|another|different|"
    rf"separate|new|successive|consecutive)\s+{_OCCASION_RE.pattern}s?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_SCENE_SLUGLINE_HEADER = re.compile(
    rf"^\s*(?:INT\.|EXT\.|INT/EXT\.|EXT/INT\.)\s+.+?[-–—:]\s*"
    rf"(?:(?:the|a)\s+)?(?:next|following|subsequent|another|different|"
    rf"separate|new|successive|consecutive)\s+{_OCCASION_RE.pattern}s?\s*",
    re.IGNORECASE,
)
_SCENE_SLUGLINE = re.compile(
    rf"{_SCENE_SLUGLINE_HEADER.pattern}(?:[-–—:].*)?$",
    re.IGNORECASE,
)
_MONTAGE = re.compile(r"\bmontage(?:\s+sequence)?\b", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:no|not|never|does\s+not|doesn't|did\s+not|didn't|will\s+not|"
    r"won't|cannot|can't|is\s+not|isn't|was\s+not|wasn't)\b",
    re.IGNORECASE,
)


def _mask_quoted_text(text: str) -> str:
    """Blank paired quotations while preserving line boundaries."""

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group())

    text = _QUOTED_SPANS.sub(blank, text)
    return _STRAIGHT_SINGLE_QUOTED_SPAN.sub(blank, text)


def explicitly_requests_spaced_recurrence(source_event_text: object) -> bool:
    """Return whether one source event explicitly describes recurring occasions.

    Accepted forms are ``each/every <occasion>`` and repeated-occasion forms
    such as ``night after night``. The interval noun keeps phrases like
    ``every person`` or ``each bouquet`` out of the recurrence path.
    """

    text = _mask_quoted_text(str(source_event_text or ""))
    return bool(_EXPLICIT_REPEAT.search(text))


def recurring_source_event_ids(
    source_events: Iterable[Mapping[str, Any]],
) -> frozenset[str]:
    """Collect IDs whose own source text explicitly requests spaced recurrence."""

    recurring: set[str] = set()
    for event in source_events:
        if not isinstance(event, Mapping):
            continue
        event_id = str(event.get("event_id") or "").strip().upper()
        if event_id and explicitly_requests_spaced_recurrence(event.get("text")):
            recurring.add(event_id)
    return frozenset(recurring)


def _cross_clause_visual_time_end(text: str) -> int | None:
    """Return the end of an immediately displayed later-time readout, if safe."""

    masked = _mask_quoted_text(text)
    for match in _CROSS_CLAUSE_VISUAL_TIME.finditer(masked):
        cue_clause = match.group("cue_clause")
        occasion_clause = match.group("occasion_clause")
        cue = _VISUAL_CROSS_CLAUSE_CUE.search(cue_clause)
        occasion = _VISIBLE_CLOCK_LATER_OCCASION.search(occasion_clause)
        if cue is None or occasion is None:
            continue
        if (
            _NEGATION.search(cue_clause)
            or _PLANNED_VISUAL_TIME_CUE.search(cue_clause)
            or _PLANNED_VISUAL_TRANSITION.search(cue_clause)
            or _NARRATED_TIME_CUE_PREFIX.search(cue_clause[:cue.start()])
            or _NEGATION.search(occasion_clause)
            or _PLANNED_VISUAL_TIME_CUE.search(occasion_clause)
            or _PLANNED_VISUAL_TRANSITION.search(occasion_clause)
            or _NARRATED_TIME_CUE_PREFIX.search(occasion_clause[:occasion.start()])
        ):
            continue
        return match.start("occasion_clause") + occasion.end()
    return None


def action_establishes_later_occasion(visual_action: object) -> bool:
    """Require a local action cue that separates this occurrence in time.

    Cues may start a sentence/clause, form a scene heading, explicitly mark a
    spaced montage, or show a clock/calendar/lighting transition directly
    linked to a later occasion. A time-lapse must explicitly mark elapsed
    occasions. Merely narrated, planned, quoted, or negated time phrases do
    not count. This intentionally favors a missed exception over waiving
    replay based on ambiguous prose.
    """

    text = _mask_quoted_text(str(visual_action or ""))
    if _cross_clause_visual_time_end(text) is not None:
        return True
    for line in text.splitlines() or [text]:
        if _SCENE_SLUGLINE.match(line):
            return True

        clauses = re.split(r"(?<=[.!?;:—–])\s+", line)
        for clause in clauses:
            if _STANDALONE_TIME_HEADING.match(clause):
                return True
            transition = _LEADING_TRANSITION.match(clause)
            if transition:
                remainder = clause[transition.end():]
                # A bare prose mention at the end of a clause is not a shown
                # transition; standalone headings are handled above.
                if remainder.strip(" \t,;:-–—") and not _NEGATION.search(clause):
                    return True
            # A specific visual time-passage cue may place the later occasion
            # after its establishing action rather than at the clause start.
            # Require a direct link (e.g. clock hands tick forward to a later
            # evening, calendar pages flip to the next day, or lighting shifts
            # into another evening); mere mentions and quoted/negated time do
            # not establish that the scene has advanced.
            visual_transition = _VISUAL_LATER_OCCASION.search(clause)
            if (
                visual_transition
                and not _NEGATION.search(clause)
                and not _PLANNED_VISUAL_TIME_CUE.search(clause)
                and not _NARRATED_TIME_CUE_PREFIX.search(
                    clause[:visual_transition.start()]
                )
            ):
                return True
            # An explicit time-lapse may establish several elapsed occasions
            # without naming a particular "next" one (for example, a clock
            # montage marking the passage of several nights). Require a
            # visible time-lapse, an elapsed-time verb, and an occasion count;
            # a bare clock hand or a future mention is insufficient.
            visual_elapsed = _VISUAL_ELAPSED_OCCASIONS.search(clause)
            if (
                visual_elapsed
                and not _NEGATION.search(clause)
                and not _PLANNED_VISUAL_TIME_CUE.search(clause)
                and not _NARRATED_TIME_CUE_PREFIX.search(
                    clause[:visual_elapsed.start()]
                )
            ):
                return True

        # A montage is an explicit visual transition only when its local text
        # also names a recurring or later occasion. A montage alone is not a
        # time cue, and the time phrase must be unquoted.
        if _MONTAGE.search(line) and not _NEGATION.search(line):
            if (
                _EXPLICIT_REPEAT.search(line)
                or _LATER_OCCASION.search(line)
                or _ELAPSED_OCCASIONS.search(line)
                or _OCCASION_AFTER_OCCASION.search(line)
            ):
                return True
    return False


def later_occasion_action_text(visual_action: object) -> str:
    """Return only visible action text after an explicit later-time cue.

    This is narrower than :func:`action_establishes_later_occasion`: a clock
    insert or a scene heading can establish that time has passed, but it is
    not itself evidence that a recurring source action happened afterward.
    The returned text starts after the cue so callers can check that the
    source action is actually performed in the later occasion. An action
    before a later time-lapse is deliberately excluded.
    """

    text = _mask_quoted_text(str(visual_action or ""))
    lines = text.splitlines() or [text]

    def suffix_after(line_index: int, offset: int) -> str:
        suffix_lines = [lines[line_index][offset:], *lines[line_index + 1:]]
        return "\n".join(suffix_lines).strip(" \t\r\n,;:-–—.!?")

    def usable_scope(scope: str) -> str:
        if not scope.strip():
            return ""
        first_clause = re.split(r"(?<=[.!?;:—–])\s+", scope, maxsplit=1)[0]
        if (
            _NEGATION.search(first_clause)
            or _PLANNED_ACTION_SCOPE.search(first_clause)
            or _NARRATED_TIME_CUE_PREFIX.search(first_clause)
        ):
            return ""
        return scope

    cross_clause_end = _cross_clause_visual_time_end(text)
    if cross_clause_end is not None:
        scope = usable_scope(text[cross_clause_end:].strip(" \t\r\n,;:-–—.!?"))
        if scope:
            return scope

    clause_separator = re.compile(r"(?<=[.!?;:—–])\s+")
    for line_index, line in enumerate(lines):
        if _SCENE_SLUGLINE.match(line):
            header = _SCENE_SLUGLINE_HEADER.match(line)
            offset = header.end() if header else len(line)
            scope = usable_scope(suffix_after(line_index, offset))
            if scope:
                return scope
            continue

        cursor = 0
        for clause in clause_separator.split(line):
            if not clause:
                continue
            clause_start = line.find(clause, cursor)
            if clause_start < 0:
                clause_start = cursor
            cursor = clause_start + len(clause)

            if _STANDALONE_TIME_HEADING.match(clause):
                scope = usable_scope(suffix_after(line_index, cursor))
                if scope:
                    return scope

            transition = _LEADING_TRANSITION.match(clause)
            if transition and not _NEGATION.search(clause):
                scope = usable_scope(suffix_after(
                    line_index, clause_start + transition.end(),
                ))
                if scope:
                    return scope

            visual_transition = _VISUAL_LATER_OCCASION.search(clause)
            if (
                visual_transition
                and not _NEGATION.search(clause)
                and not _PLANNED_VISUAL_TIME_CUE.search(clause)
                and not _NARRATED_TIME_CUE_PREFIX.search(
                    clause[:visual_transition.start()]
                )
            ):
                scope = usable_scope(suffix_after(
                    line_index, clause_start + visual_transition.end(),
                ))
                if scope:
                    return scope

            visual_elapsed = _VISUAL_ELAPSED_OCCASIONS.search(clause)
            if (
                visual_elapsed
                and not _NEGATION.search(clause)
                and not _PLANNED_VISUAL_TIME_CUE.search(clause)
                and not _NARRATED_TIME_CUE_PREFIX.search(
                    clause[:visual_elapsed.start()]
                )
            ):
                scope = usable_scope(suffix_after(
                    line_index, clause_start + visual_elapsed.end(),
                ))
                if scope:
                    return scope

        if _MONTAGE.search(line) and not _NEGATION.search(line):
            montage_cues = [
                match
                for pattern in (
                    _EXPLICIT_REPEAT,
                    _LATER_OCCASION,
                    _ELAPSED_OCCASIONS,
                    _OCCASION_AFTER_OCCASION,
                )
                if (match := pattern.search(line)) is not None
            ]
            if montage_cues:
                cue = min(montage_cues, key=lambda match: match.start())
                scope = usable_scope(suffix_after(line_index, cue.end()))
                if scope:
                    return scope
    return ""


def allows_spaced_recurrence(
    event_id: object,
    recurring_event_ids: Iterable[object],
    visual_action: object,
) -> bool:
    """Allow a local replay only for that explicitly recurring event and later time.

    ``visual_action`` must be the current connective beat's action text, not
    global context, a closing-state claim, or camera-only prose.
    """

    normalized_id = str(event_id or "").strip().upper()
    if not normalized_id:
        return False
    known_recurring_ids = {
        str(value or "").strip().upper()
        for value in recurring_event_ids
        if str(value or "").strip()
    }
    return normalized_id in known_recurring_ids and action_establishes_later_occasion(
        visual_action
    )


def expand_spaced_recurrence_phases(
    phases: list[dict[str, Any]], source_events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Give an explicitly recurring silent action two distinct occasions.

    This expands camera coverage only: the immutable source event and story
    schedule still appear once. Two is the minimum needed to show recurrence,
    not an inferred total number of visits. Authored counted/timed occurrences
    and speech-bearing phases stay with their original explicit schedule.
    """
    events = {str(event.get("event_id") or "").upper(): event for event in source_events}
    result = []
    for phase in phases:
        ids = [str(value).upper() for value in phase.get("source_event_ids", [])]
        event = events.get(ids[0]) if len(ids) == 1 else None
        if (not event or phase.get("dialogue_ids") or phase.get("_spaced_recurrence")
                or any(event.get(key) is not None for key in (
                    "start_seconds", "end_seconds", "source_start_seconds", "source_end_seconds",
                ))):
            result.append(phase)
            continue
        text = str(event.get("text") or "")
        cue = _EXPLICIT_REPEAT.search(_mask_quoted_text(text))
        if not cue:
            result.append(phase)
            continue
        if re.search(r"\b(?:other|alternate|alternating)\b", cue.group(), re.IGNORECASE):
            # A two-occasion rewrite must not erase an authored alternating
            # cadence. Keep that more specific schedule with the writer.
            result.append(phase)
            continue
        occasion_match = _OCCASION_RE.search(cue.group())
        if not occasion_match:
            result.append(phase)
            continue
        occasion = occasion_match.group().lower()
        # Preserve the user's complete action; only specialize its recurrence
        # cue to the occasion being shown. This does not infer a new prop,
        # participant, route, camera cut or intermediate physical result.
        source_before, source_after = text[:cue.start()], text[cue.end():]
        for occurrence in (1, 2):
            item = deepcopy(phase)
            item["beat_id"] = f"{phase['beat_id']}.R{occurrence}"
            item["_parent_beat_id"] = phase.get("_parent_beat_id") or phase["beat_id"]
            cue_text = f"on {'one' if occurrence == 1 else 'a subsequent'} {occasion}"
            action = source_before + cue_text + source_after
            action = action[0].upper() + action[1:] if action else action
            item["description"] = item["_canonical_action"] = action
            item["_spaced_recurrence"] = {
                "source_event_id": ids[0], "occurrence": occurrence,
                "minimum_occurrences": 2, "occasion": occasion,
            }
            # Do not copy a claimed overall result onto the first occasion.
            item["state_after"] = f"The visible result of this occasion: {action}"
            for duration_key in ("_action_seconds", "authored_duration"):
                if duration_key in item:
                    item[duration_key] = float(item[duration_key] or 0) / 2
            if occurrence == 2:
                item.pop("_start_frame_continuation", None)
            result.append(item)
    return result
