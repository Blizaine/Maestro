"""Shared local-window context without global setting itineraries."""
from __future__ import annotations

import re


_ITINERARY_RE = re.compile(
    r"(?i)\b(?:transitions?|travels?|moves?|progresses?|changes?)\b.{0,120}\b(?:from|to|through)\b"
    r"|^\s*from\b.{1,160}\bto\b|\b(?:then|followed by|subsequently|eventually)\b"
    r"|\b(?:starts?|begins?)\b.{1,160}\b(?:ends?|finishes?)\b"
)
_STYLE_ITINERARY_RE = re.compile(
    r"(?i)\b(?:lighting|setting|location|scene|environment|background|color palette)\b"
    r".{0,90}\b(?:transitions?|changes?|moves?)\b.{0,120}\b(?:from|to|through)\b"
)


def stable_h3_window_context(value: str, *, visual_style: bool = False) -> str:
    """Keep shared static direction and effect mechanics, not a whole journey.

    Visual language may encode persistent power/effect sources. A beam
    travelling from eyes to a target is a mechanic rather than a location route.
    """
    itinerary = _STYLE_ITINERARY_RE if visual_style else _ITINERARY_RE
    return " ".join(
        sentence for sentence in re.split(r"(?<=[.!?])\s+", str(value or ""))
        if not itinerary.search(sentence)
    ).rstrip('.')
