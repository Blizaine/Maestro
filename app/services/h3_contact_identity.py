"""Conservative checks for explicit source contact-instrument requirements."""

from __future__ import annotations

import re


_PART = (
    r"elbows?|knees?|shoulders?|forearms?|fists?|palms?|heads?|heels?|"
    r"foot|feet|boots?"
)
_SIDE = r"left|right"
_CONTACT_NOUN = (
    r"strike|smash|ram|check|slam|blow|jab|bash|thrust|butt|impact|"
    r"attack|punch|swing|swipe"
)
_CONTACT_VERB = (
    r"strike|strikes|struck|striking|smash|smashes|smashed|smashing|"
    r"hit|hits|hitting|ram|rams|rammed|ramming|drive|drives|drove|driving|"
    r"slam|slams|slammed|slamming|jab|jabs|jabbed|jabbing|bash|bashes|"
    r"bashed|bashing|thrust(?:s|ed|ing)?|punch|punches|"
    r"punched|punching|check|checks|checked|checking|swing|swings|swung|"
    r"swinging|throw|throws|threw|throwing|deliver|delivers|delivered|"
    r"delivering|attack|"
    r"attacks|attacked|attacking|connect|connects|connected|connecting"
)
_POSSESSOR = r"(?:my|your|his|her|their|its|the|a|an|[A-Z][\w'’-]*['’]s)"
_SUBJECT_POSSESSOR = r"(?:my|your|his|her|their|its|the|[A-Z][\w'’-]*['’]s)"
_CLAUSE_BREAK = r"(?:and|but|then|while|before|after|although)"


def _patterns() -> tuple[re.Pattern[str], ...]:
    """Return small, local grammars; body-part words alone are never enough."""

    part = rf"(?P<part>{_PART})"
    side = rf"(?:(?P<side>{_SIDE})\s+)?"
    return (
        # Instrument-headed action phrases: "left elbow smash", "knee strike".
        re.compile(
            rf"\b{side}{part}\s+(?P<attack>{_CONTACT_NOUN})\b", re.I,
        ),
        # A body part is explicitly used as an instrument: "strikes Eli with her elbow".
        re.compile(
            rf"\b(?:{_CONTACT_VERB})\b"
            rf"(?:\s+(?!{_CLAUSE_BREAK}\b)[\w'’-]+){{0,5}}\s+"
            rf"(?P<selector>(?:with|using)\s+(?:{_POSSESSOR}\s+)?"
            rf"{side}{part})\b",
            re.I,
        ),
        # Direct bodily contact: "drives her shoulder into Eli".
        re.compile(
            rf"\b(?:{_CONTACT_VERB})\b"
            rf"(?:\s+(?!{_CLAUSE_BREAK}\b)[\w'’-]+){{0,4}}\s+"
            rf"(?P<selector>(?:(?:{_POSSESSOR})\s+)?{side}{part})\s+"
            rf"(?:into|against|at|onto|to|through)\b",
            re.I,
        ),
        # The instrument is the grammatical subject: "her elbow strikes Eli".
        re.compile(
            rf"\b(?P<selector>(?:(?:{_SUBJECT_POSSESSOR})\s+)?{side}{part})\s+"
            rf"(?:{_CONTACT_VERB})\s+(?:into|against|at|onto|to|through|"
            rf"him|her|them|me|you|us|the\s+[\w'’-]+|"
            rf"(?-i:[A-Z][\w'’-]*))\b",
            re.I,
        ),
    )


_REQUIREMENT_PATTERNS = _patterns()

# These verbs name the contact instrument themselves. Keep this list explicit:
# words such as "elbow" or "shoulder" in a pose are not contact requirements.
_INSTRUMENT_VERBS = (
    (re.compile(r"\b(?P<selector>elbow(?:ed|ing))\b", re.I), "elbow"),
    (re.compile(r"\b(?P<selector>knee(?:d|ing))\b", re.I), "knee"),
    (re.compile(r"\b(?P<selector>headbutt(?:s|ed|ing))\b", re.I), "head"),
    (re.compile(r"\b(?P<selector>shoulder[- ]check(?:s|ed|ing))\b", re.I), "shoulder"),
    (re.compile(r"\b(?P<selector>forearm[- ](?:strike|smash|ram)(?:s|ed|ing))\b", re.I), "forearm"),
    (re.compile(r"\b(?P<selector>fist[- ](?:punch|strike|smash)(?:es|ed|ing))\b", re.I), "fist"),
    (re.compile(r"\b(?P<selector>punch(?:es|ed|ing))\b", re.I), "fist"),
    (re.compile(
        r"\b(?P<selector>elbows?\s+"
        r"(?:him|her|them|me|you|us|(?-i:[A-Z][\w'’-]*)))\b", re.I,
    ), "elbow"),
    (re.compile(
        r"\b(?P<selector>knees?\s+"
        r"(?:him|her|them|me|you|us|(?-i:[A-Z][\w'’-]*)))\b", re.I,
    ), "knee"),
)

_PLURAL_PARTS = {
    "elbows": "elbow", "knees": "knee", "shoulders": "shoulder",
    "forearms": "forearm", "fists": "fist", "palms": "palm",
    "heads": "head", "heels": "heel", "feet": "foot", "boots": "boot",
}


def _is_negated_contact(text: str, start: int) -> bool:
    """Exclude denied or absent actions while retaining misses and near misses."""

    prefix = text[max(0, start - 80):start]
    return bool(re.search(
        r"\b(?:no|never|without|neither|not)\s+"
        r"(?:(?:an?|the|any|single|one)\s+){0,2}$|"
        r"\b(?:do|does|did|will|would|could|should|can|must|might|may)\s+"
        r"not\s+(?:ever\s+)?$|"
        r"\b(?:don't|doesn't|didn't|won't|wouldn't|couldn't|shouldn't|can't|"
        r"mustn't|mightn't|mayn't)\s*$",
        prefix,
        re.I,
    ))


def _contact_instruments(text: str) -> list[tuple[str, str | None, str]]:
    """Extract only body parts grammatically linked to an attack or contact."""

    found: list[tuple[int, str, str | None, str]] = []
    for pattern in _REQUIREMENT_PATTERNS:
        for match in pattern.finditer(text):
            if _is_negated_contact(text, match.start()):
                continue
            part = _PLURAL_PARTS.get(
                match.group("part").casefold(), match.group("part").casefold(),
            )
            side = (match.groupdict().get("side") or "").casefold() or None
            selected = match.groupdict().get("selector")
            if selected:
                selector = selected.strip()
            elif "attack" in match.groupdict():
                selector = f"{side + ' ' if side else ''}{part} {match.group('attack')}"
            else:
                selector = match.group(0).strip()
            found.append((match.start(), part, side, selector))

    for pattern, part in _INSTRUMENT_VERBS:
        for match in pattern.finditer(text):
            if _is_negated_contact(text, match.start()):
                continue
            # Preserve side only when it is part of the explicit instrument
            # selector itself. These verb forms have no side qualifier.
            found.append((match.start(), part, None, match.group("selector")))

    found.sort(key=lambda item: item[0])
    unique: dict[tuple[str, str | None], str] = {}
    for _offset, part, side, selector in found:
        unique.setdefault((part, side), selector)
    return [(part, side, selector) for (part, side), selector in unique.items()]


def omitted_h3_contact_instruments(source_text: str, visible_action: str) -> list[str]:
    """List explicit source contact instruments absent from this visible beat.

    The grammar intentionally requires a body part to appear in a clear
    contact construction. A pose such as "shoulder-first" or "elbows tucked"
    does not establish an instrument, and a different body part never stands
    in for the required one. Misses and near misses remain attempts and retain
    their instrument requirement.
    """

    required = _contact_instruments(str(source_text or ""))
    visible = _contact_instruments(str(visible_action or ""))
    missing = []
    for part, required_side, selector in required:
        covered = any(
            visible_part == part
            and (required_side is None or visible_side == required_side)
            for visible_part, visible_side, _visible_selector in visible
        )
        if not covered:
            missing.append(selector)
    return missing
