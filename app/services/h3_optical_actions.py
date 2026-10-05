"""Mask optical camera moves only in comparison text for action checks.

Source prose is immutable and must never be rewritten with this helper. It is
intended for parsers that compare physical actions and otherwise mistake a
lens/camera move for an action performed by a named subject in the same line.
"""

from __future__ import annotations

import re


_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?P<sentence>(?<=[.!?;])\s+)|"
    r"(?P<comma>,\s*(?:and|but|then|finally|afterward|afterwards|meanwhile|while|as|when)\s+)|"
    r"(?P<word>\s+(?:and|but|then|finally|afterward|afterwards|meanwhile|while|as|when|after|before|who)\s+)",
    re.IGNORECASE,
)
_LEADING_CONNECTOR_RE = re.compile(
    r"^\s*(?:(?:and\s+)?(?:then|finally|afterward|afterwards|meanwhile)\s+|"
    r"(?:and|but|while|as|when)\s+)+",
    re.IGNORECASE,
)
_CAMERA_SUBJECT_RE = re.compile(r"^(?:(?:the|a|an)\s+)?(?P<subject>lens|camera)\b", re.I)
_CAMERA_PREDICATE_RE = re.compile(
    r"^(?:(?:first|quick(?:ly)?|slow(?:ly)?|fast|rapidly|smoothly|gradually|gently|slightly|subtly)\s+)*"
    r"(?:"
    r"pull(?:s|ed|ing)?\s+(?:wide|back|in|out|away|closer|wider|to\b)|"
    r"push(?:es|ed|ing)?\s+(?:in|out|back|to\b|toward\b|towards\b)|"
    r"glid(?:e|es|ed|ing)\b|"
    r"(?:quick[- ]?)?cut(?:s|ting)?\s+to\b|"
    r"follow(?:s|ed|ing)?\b|track(?:s|ed|ing)?\b|"
    r"pan(?:s|ned|ning)?\b|tilt(?:s|ed|ing)?\b|doll(?:y|ies|ied|ying)\b|zoom(?:s|ed|ing)?\b|"
    r"mov(?:e|es|ed|ing)\s+(?:in|out|back|to\b|toward\b|towards\b|closer|wider)\b|"
    r"drift(?:s|ed|ing)?\b|sweep(?:s|t|ing)?\b|"
    r"rack(?:s|ed|ing)?\s+focus\b|focus(?:es|ed|ing)?\s+(?:on|from|to)\b|"
    r"refram(?:e|es|ed|ing)\b|fram(?:e|es|ed|ing)\b|settle(?:s|d|ing)?\s+(?:on|into)\b"
    r")",
    re.IGNORECASE,
)
_ACTOR_PRONOUN_RE = re.compile(r"^(?P<pronoun>he|she|they|who)\b(?=\s+\w)", re.I)
_GENERIC_ACTOR_RE = re.compile(
    r"^(?P<actor>[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,2})\b"
)


def _strip_leading_connectors(value: str) -> str:
    return _LEADING_CONNECTOR_RE.sub("", value).strip(" \t,;:-")


def _actor_at_start(
    value: str,
    cast_pattern: re.Pattern[str] | None,
) -> tuple[str | None, bool]:
    """Return (actor label, is pronoun) for a clause led by a person."""
    pronoun = _ACTOR_PRONOUN_RE.match(value)
    if pronoun:
        return pronoun.group("pronoun"), True
    if cast_pattern is not None:
        match = cast_pattern.match(value)
        if match:
            return match.group(0), False
        return None, False
    match = _GENERIC_ACTOR_RE.match(value)
    if match and match.group("actor").casefold() not in {"the", "a", "an"}:
        return match.group("actor"), False
    return None, False


def _camera_predicate_starts(value: str) -> bool:
    return bool(_CAMERA_PREDICATE_RE.match(value.strip()))


def _single_cast_target(
    value: str,
    cast_pattern: re.Pattern[str] | None,
) -> str | None:
    if cast_pattern is None:
        return None
    matches = list(cast_pattern.finditer(value))
    names = {match.group(0).casefold(): match.group(0) for match in matches}
    return next(iter(names.values())) if len(names) == 1 else None


def _resolve_subject_pronoun(value: str, target: str | None) -> str:
    if not target:
        return value
    match = _ACTOR_PRONOUN_RE.match(value)
    if not match:
        return value
    return target + value[match.end():]


def _camera_participle_suffix(
    value: str,
    subject_end: int,
    cast_pattern: re.Pattern[str] | None,
) -> str | None:
    """Retain an explicit target's participial action after a lens cue."""
    tail = value[subject_end:]
    normalized_tail = tail.lstrip()
    predicate = _CAMERA_PREDICATE_RE.match(normalized_tail)
    if not predicate:
        return None

    # Pull-wide / push-in cues can introduce a camera-owned infinitive target:
    # ``pulls wide to follow Eli cutting the rope``. The following participle
    # is a human action and must survive the optical mask.
    predicate_text = normalized_tail[:predicate.end()]
    camera_verb = re.match(
        r"(?:(?:first|quick(?:ly)?|slow(?:ly)?|fast|rapidly|smoothly|gradually|"
        r"gently|slightly|subtly)\s+)*(?P<verb>follow|track)(?:s|ed|ing)?\b",
        predicate_text,
        re.I,
    )
    target_source = normalized_tail[predicate.end():]
    if not camera_verb:
        infinitive = re.match(r"\s+to\s+(?:follow|track)\s+", normalized_tail[predicate.end():], re.I)
        if not infinitive:
            return None
        target_source = normalized_tail[predicate.end() + infinitive.end():]
    target_source = target_source.lstrip()

    target_match = cast_pattern.match(target_source) if cast_pattern is not None else None
    if target_match is None and cast_pattern is None:
        target_match = _GENERIC_ACTOR_RE.match(target_source)
    if target_match is None:
        # A known human action attached to an unregistered/ambiguous framing
        # target stays unresolved in the full clause rather than disappearing.
        generic_target = _GENERIC_ACTOR_RE.match(target_source)
        if generic_target and _has_participial_human_suffix(
            target_source[generic_target.end():],
        ):
            return value
        return None

    suffix = target_source[target_match.end():]
    participial_action = _participle_after_target(suffix)
    if participial_action is None:
        if re.search(r"\b[a-z][\w'’-]*ing\b", suffix, re.IGNORECASE):
            # An unfamiliar modifier or participial predicate may still be a
            # human action. Preserve the mixed camera clause for downstream
            # review instead of silently dropping that evidence.
            return value
        return None
    return target_match.group(0) + " " + participial_action


def _participle_after_target(suffix: str) -> str | None:
    match = re.match(
        r"\s+(?P<action>(?:(?:[a-z][\w'’-]*ly|visibly|still|already|actively|currently)\s+){0,3}"
        r"[a-z][\w'’-]*ing\b.*)$",
        suffix,
        re.IGNORECASE,
    )
    return match.group("action") if match else None


def _has_participial_human_suffix(suffix: str) -> bool:
    return _participle_after_target(suffix) is not None


def _clauses_with_separators(
    value: str,
    cast_pattern: re.Pattern[str] | None,
) -> list[tuple[str, str]]:
    boundary_re = _CLAUSE_BOUNDARY_RE
    # A comma can introduce an explicit new actor without a connective:
    # ``the camera pulls back, Eli opens the door``. Prefer the known cast;
    # callers without one may still use a capitalized person-name lead.
    actor_pattern = (
        cast_pattern.pattern
        if cast_pattern is not None
        else r"(?-i:[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,2})"
    )
    boundary_re = re.compile(
        rf"(?:{_CLAUSE_BOUNDARY_RE.pattern})|"
        rf"(?P<person_comma>,\s*(?={actor_pattern}\s+\w))",
        re.IGNORECASE,
    )
    clauses: list[tuple[str, str]] = []
    cursor = 0
    separator_before = ""
    for boundary in boundary_re.finditer(value):
        clauses.append((value[cursor:boundary.start()], separator_before))
        separator_before = boundary.group(0)
        cursor = boundary.end()
    clauses.append((value[cursor:], separator_before))
    return clauses


def _source_clause_text(raw_clause: str, separator_before: str) -> str:
    """Reattach meaningful connector words while dropping delimiter punctuation."""
    clause = raw_clause.strip()
    connector = separator_before.strip(" \t,;:-")
    return f"{connector} {clause}" if connector else clause


def _physical_action_analysis(
    value: object,
    cast_pattern: re.Pattern[str] | None = None,
) -> tuple[str, list[str]]:
    """Return comparison text and the pure optical clauses removed from it.

    Only clauses explicitly led by ``camera`` or ``lens`` (or unheaded camera
    predicates that continue such a clause) are eligible. Human-led clauses,
    including unknown predicates and actions involving camera equipment, stay
    intact. A human action subordinate to a camera move is retained, and a
    unique cast target can resolve its leading ``he``, ``she`` or ``they``.
    """
    text = str(value or "")
    if not text.strip():
        return text, []

    clauses = _clauses_with_separators(text, cast_pattern)
    kept: list[str] = []
    optical_only: list[str] = []
    camera_chain = False
    camera_target: str | None = None
    removed_since_last_kept = False
    previous_raw = ""

    for raw_clause, separator_before in clauses:
        clause = raw_clause.strip()
        if not clause:
            continue
        # A sentence boundary ends ellipsis-style camera-subject carryover.
        if previous_raw.rstrip().endswith((".", "!", "?", ";")):
            camera_chain = False
            camera_target = None

        core = _strip_leading_connectors(clause)
        if not core:
            previous_raw = raw_clause
            continue

        camera_subject = _CAMERA_SUBJECT_RE.match(core)
        if camera_subject:
            tail = core[camera_subject.end():].lstrip()
            is_camera_move = _camera_predicate_starts(tail)
            if is_camera_move:
                camera_chain = True
                camera_target = _single_cast_target(core, cast_pattern)
                human_suffix = _camera_participle_suffix(
                    core, camera_subject.end(), cast_pattern,
                )
                if human_suffix:
                    if human_suffix == core:
                        # The predicate is mixed/ambiguous; keep its evidence
                        # intact so downstream physical checks fail closed.
                        camera_chain = False
                        camera_target = None
                        retained_clause = clause
                    else:
                        retained_clause = human_suffix
                    if kept:
                        kept.append("; " if removed_since_last_kept else (separator_before or " "))
                    kept.append(retained_clause)
                    removed_since_last_kept = True
                    previous_raw = raw_clause
                    continue
                optical_only.append(_source_clause_text(raw_clause, separator_before))
                removed_since_last_kept = True
                previous_raw = raw_clause
                continue
            # An unknown or passive camera predicate is deliberately retained.
            camera_chain = False
            camera_target = None

        if camera_chain and _camera_predicate_starts(core):
            camera_target = _single_cast_target(core, cast_pattern) or camera_target
            human_suffix = _camera_participle_suffix(core, 0, cast_pattern)
            if human_suffix:
                # Inherited camera clauses can also carry a human action:
                # ``then follows Eli opening the case``. Preserve the action
                # (and keep ambiguous mixed clauses intact) just as for an
                # explicitly headed camera clause.
                if human_suffix == core:
                    camera_chain = False
                    camera_target = None
                    retained_clause = clause
                else:
                    retained_clause = human_suffix
                if kept:
                    kept.append("; " if removed_since_last_kept else (separator_before or " "))
                kept.append(retained_clause)
                removed_since_last_kept = True
                previous_raw = raw_clause
                continue
            optical_only.append(_source_clause_text(raw_clause, separator_before))
            removed_since_last_kept = True
            previous_raw = raw_clause
            continue

        actor, is_pronoun = _actor_at_start(core, cast_pattern)
        retained_clause = clause
        if (
            camera_chain
            and re.search(r"\bwho\b", separator_before, re.IGNORECASE)
        ):
            # A relative ``who`` explicitly binds the next predicate to the
            # named target; unlike an unmarked adjacent clause, this is not a
            # guess based on framing proximity.
            core = f"{camera_target} {core}" if camera_target else f"who {core}"
            actor, is_pronoun = (camera_target, False) if camera_target else ("who", True)
            retained_clause = core
        if actor:
            if is_pronoun:
                core = _resolve_subject_pronoun(core, camera_target)
                retained_clause = core
            # A human-led predicate, known or unknown, is never stripped.
            camera_chain = False
            camera_target = None
        else:
            # Do not let a camera lead leak across an unrelated unparsed clause.
            camera_chain = False
            camera_target = None

        if kept:
            # Retain a source conjunction only when adjacent retained clauses
            # still have one. Use a hard clause boundary across removed optics
            # so the downstream parser cannot bind both human actors together.
            if removed_since_last_kept:
                kept.append("; ")
            else:
                kept.append(separator_before or " ")
        kept.append(retained_clause)
        removed_since_last_kept = False
        previous_raw = raw_clause

    return "".join(kept).strip(), optical_only


def physical_action_text(
    value: object,
    cast_pattern: re.Pattern[str] | None = None,
) -> str:
    """Return comparison text with narrow camera-owned predicates removed."""
    physical_text, _ = _physical_action_analysis(value, cast_pattern)
    return physical_text


def optical_only_clauses(
    value: object,
    cast_pattern: re.Pattern[str] | None = None,
) -> list[str]:
    """Return original clauses removed as purely optical camera movement.

    The clauses are collected by the same pass as :func:`physical_action_text`;
    mixed, ambiguous, and human-action clauses are therefore never reported as
    optical-only.
    """
    _, optical_clauses = _physical_action_analysis(value, cast_pattern)
    return optical_clauses
