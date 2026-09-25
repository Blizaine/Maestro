"""Narrow source-grammar helpers for directions that do not consume screen time.

These helpers classify syntax, not story meaning. Uncertain phrases remain in
the event catalog; the complete original directions remain in source intent.
"""

from __future__ import annotations

import re
from collections.abc import Callable


def strip_modified_context_headings(
    source: str, *, is_static: Callable[[str], bool],
) -> str:
    """Remove an entire descriptive heading, rather than strand its modifiers.

    Only a sentence-leading article + static modifiers + production noun is a
    heading. A clause such as 'Nora rewrites the story:' is not consumed.
    """
    pattern = re.compile(
        r"(?P<boundary>^|[.!?;\n]\s*)"
        r"(?P<heading>(?:an?|the)\s+(?P<modifiers>[^.!?;:\n]{1,72}?)\s+"
        r"(?:story|scene|film|video|sequence)\s*:)", re.IGNORECASE,
    )
    return pattern.sub(
        lambda m: m.group("boundary") if is_static(m.group("modifiers")) else m.group(0),
        source,
    )


_MEDIA_SUBJECT = re.compile(
    r"(?P<subject>(?:the|this|that|a|an)\s+"
    r"(?:(?:\d+(?:\.\d+)?)[ -](?:second|minute)s?[ -])?"
    r"(?:video|film|clip|scene|shot|camera|soundtrack|track|performance))"
    r"(?=\s|$)", re.IGNORECASE,
)


def media_subject_prefix(text: str) -> str:
    """Return an explicit media subject, including its duration modifier."""
    match = _MEDIA_SUBJECT.match(text.strip())
    return match.group("subject") if match else ""


def is_media_subject_only(text: str) -> bool:
    clean = text.strip(" \t\r\n,;:.!?")
    return bool(clean and media_subject_prefix(clean).casefold() == clean.casefold())


def is_media_no_additions(text: str) -> bool:
    """Recognize negative output/cast lists, not negative narrative actions."""
    clean = text.strip(" \t\r\n,;:.!?")
    if re.fullmatch(
        r"(?:please\s+)?(?:show|depict|include)\s+no\s+"
        r"(?:(?:additional|extra|other|new)\s+)?"
        r"(?:musicians?|singers?|bands?|orchestras?|audience|bystanders?)",
        clean, re.IGNORECASE,
    ):
        return True
    item = (
        r"(?:(?:new|additional|extra)\s+)?(?:dialogue|speech|voices?|vocals?|"
        r"instruments?|effects|sound\s+effects|lyrics?|subtitles?|captions?|"
        r"(?:replacement\s+or\s+supplemental\s+|replacement\s+|supplemental\s+)?music|"
        r"loops?|stretch(?:ing)?)"
    )
    return bool(re.fullmatch(
        r"(?:please\s+)?(?:add|include|write)\s+no\s+" + item
        + rf"(?:(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+){item})*",
        clean, re.IGNORECASE,
    ))


def is_story_scope_direction(text: str) -> bool:
    """A persistent location and a reference to an already specified ending.

    This deliberately does not consume a new ending action ('end with Nora
    opening the box'). Such action-bearing endings still belong on the timeline.
    """
    return bool(re.fullmatch(
        r"(?:please\s+)?keep\s+the\s+(?:story|scene|film|video)\s+"
        r"(?:at|in|inside|within)\s+(?:this|the\s+same|one|a\s+single)\s+"
        r"(?!and\b|then\b|before\b|after\b)[\w'-]+"
        r"(?:\s+(?!and\b|then\b|before\b|after\b)[\w'-]+){0,2}"
        r"(?:\s+and\s+(?:end|finish)\s+on\s+the\s+"
        r"(?:same|exchanged|completed|resulting|established|final)\s+"
        r"(?:gesture|composition|state|pose|tableau|image|outcome))?",
        text.strip(" \t\r\n,;:.!?"), re.IGNORECASE,
    ))


def is_audio_binding_direction(text: str) -> bool:
    """Keep a soundtrack binding/duration fact in context, not a camera card."""
    clean = text.strip(" \t\r\n,;:.!?")
    if re.fullmatch(
        r"(?:the\s+)?(?:source|supplied|reference)\s+(?:audio|track|soundtrack)\s+"
        r"is\s+\d+(?:\.\d+)?\s+(?:seconds?|minutes?)(?:\s+long)?",
        clean, re.IGNORECASE,
    ):
        return True
    return bool(re.fullmatch(
        r"use\s+<\s*Audio\s+\d+\s*>\s+as\s+(?:the\s+)?"
        r"(?:(?:exact|supplied|original|unedited|performance[- ]driving)\s+)*"
        r"(?:soundtrack|track|audio)"
        r"(?:,?\s+once)?(?:\s+at\s+its\s+(?:supplied|original)\s+speed)?"
        r"(?:\s+(?:and\s+)?without\s+(?:edits|editing))?",
        clean, re.IGNORECASE,
    ))
