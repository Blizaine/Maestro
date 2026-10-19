"""Shared local-window context without global setting itineraries."""
from __future__ import annotations

import re


_ITINERARY_RE = re.compile(
    r"(?i)\b(?:transitions?|travels?|moves?|progresses?|changes?|shift(?:s|ing)?)\b.{0,120}\b(?:from|to|through)\b"
    r"|^\s*from\b.{1,160}\bto\b|\b(?:then|followed by|subsequently|eventually)\b"
    r"|\b(?:starts?|begins?)\b.{0,160}\b(?:ends?|finishes?)\b"
)
_STYLE_ITINERARY_RE = re.compile(
    r"(?i)\b(?:lighting|setting|location|scene|environment|background|color palette)\b"
    r".{0,90}\b(?:transitions?|changes?|moves?|shift(?:s|ing)?)\b.{0,120}\b(?:from|to|through)\b"
)
_OPENING_PLACE_RE = re.compile(
    r"(?i)\b(?:the\s+)?(?:sequence|story|journey|video|film|narrative|scene)\s+"
    r"(?:begins?|starts?|opens?)\b"
    r"|\b(?:initial|opening|first)\s+(?:setting|location|place|environment)\b"
    r"|\b(?:setting|location|place|environment)\s+(?:at|during)\s+"
    r"(?:the\s+)?(?:start|beginning|opening)\b"
)
_STYLE_SHIFT_VERB_RE = re.compile(
    r"(?i)\b(?:transitions?|changes?|moves?|shift(?:s|ing)?)\b"
)
_STATIC_STYLE_SUBJECT_RE = re.compile(
    r"(?i)^\s*(?:(?:the|a|an)\s+)?"
    r"(?:lighting|setting|location|scene|environment|background|color palette)\s+"
    r"(?:is|are|remains?|stays?|looks?|feels?)\b"
)


def _static_visual_style_lead(sentence: str) -> str:
    """Retain a static aesthetic stated before a later-window style shift."""
    shift = _STYLE_SHIFT_VERB_RE.search(sentence)
    if not shift:
        return ""
    lead = sentence[:shift.start()].rstrip(" ,;:-")
    if not lead or not _STATIC_STYLE_SUBJECT_RE.match(lead):
        return ""
    return lead if lead.endswith((".", "!", "?")) else lead + "."


def stable_h3_window_context(value: str, *, visual_style: bool = False) -> str:
    """Keep shared static direction and effect mechanics, not a whole journey.

    Visual language may encode persistent power/effect sources. A beam
    travelling from eyes to a target is a mechanic rather than a location route.
    """
    itinerary = _STYLE_ITINERARY_RE if visual_style else _ITINERARY_RE
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(value or "")):
        if _OPENING_PLACE_RE.search(sentence):
            continue
        if itinerary.search(sentence):
            if visual_style:
                static_lead = _static_visual_style_lead(sentence)
                if static_lead:
                    kept.append(static_lead)
            continue
        kept.append(sentence)
    return " ".join(kept).rstrip(".")
