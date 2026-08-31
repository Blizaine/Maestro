"""Conservative text-integrity helpers for user and LLM authored content.

Maestro is UTF-8 end to end, but older saved projects and a few Windows HTTP
boundaries have historically decoded UTF-8 bytes as Windows-1252/Latin-1.
That turns text such as ``Wörter`` into ``WÃ¶rter`` and curly punctuation into
``â€™``.  The repair here is deliberately evidence-based: a re-decoded candidate
is accepted only when it contains fewer well-known mojibake markers than the
input, so valid international text is left alone.
"""

from __future__ import annotations

import re
from typing import Any


_KNOWN_REPLACEMENTS = {
    "\u00e2\u0080\u0098": "\u2018",
    "\u00e2\u0080\u0099": "\u2019",
    "\u00e2\u0080\u009c": "\u201c",
    "\u00e2\u0080\u009d": "\u201d",
    "\u00e2\u0080\u0093": "\u2013",
    "\u00e2\u0080\u0094": "\u2014",
    "\u00e2\u0080\u00a6": "\u2026",
    "\u00e2\u20ac\u02dc": "\u2018",
    "\u00e2\u20ac\u2122": "\u2019",
    "\u00e2\u20ac\u0153": "\u201c",
    "\u00e2\u20ac\u009d": "\u201d",
    "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u201d": "\u2014",
    "\u00e2\u20ac\u00a6": "\u2026",
    "\u00c2\u00a0": " ",
    # Observed in older Maestro Windows JSON: the leading UTF-8 byte for
    # punctuation became U+0101 before the continuation bytes survived.
    "\u0101\u0080\u0098": "\u2018",
    "\u0101\u0080\u0099": "\u2019",
    "\u0101\u0080\u009c": "\u201c",
    "\u0101\u0080\u009d": "\u201d",
    "\u0101\u0080\u0093": "\u2013",
    "\u0101\u0080\u0094": "\u2014",
    "\u0101\u0080\u00a6": "\u2026",
}

_SUSPICIOUS_SEQUENCE_RE = re.compile(
    r"(?:\u00c3[\u0080-\u00ff]|\u00c2[\u0080-\u00ff]|"
    r"\u00e2(?:\u0080|\u20ac).|\u00f0[\u0080-\u00ff]{2,3}|"
    r"\u00ef\u00bf\u00bd|\u0101[\u0080-\u00ff]{2})"
)


def _mojibake_score(text: str) -> int:
    """Return a rough cost for byte-decoding artifacts in ``text``."""

    score = len(_SUSPICIOUS_SEQUENCE_RE.findall(text)) * 4
    score += sum(3 for character in text if 0x80 <= ord(character) <= 0x9F)
    score += text.count("\ufffd") * 8
    # These lead characters are common evidence even when the remainder does
    # not match a complete sequence. They are lightly weighted because some
    # are valid letters in real languages.
    score += sum(text.count(character) for character in ("\u00c2", "\u00c3", "\u00e2", "\u00f0"))
    return score


def _decode_candidate(text: str, encoding: str) -> str | None:
    try:
        return text.encode(encoding).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def repair_text(value: Any) -> str:
    """Repair demonstrable UTF-8 mojibake without changing valid Unicode."""

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value or "")

    for broken, repaired in _KNOWN_REPLACEMENTS.items():
        text = text.replace(broken, repaired)

    # Double-encoded text needs two passes. Stop unless a candidate strictly
    # improves the artifact score; this is what protects legitimate Unicode.
    for _ in range(2):
        current_score = _mojibake_score(text)
        if current_score <= 0:
            break
        candidates = [
            candidate
            for candidate in (
                _decode_candidate(text, "cp1252"),
                _decode_candidate(text, "latin-1"),
            )
            if candidate is not None
        ]
        if not candidates:
            break
        best = min(candidates, key=_mojibake_score)
        if _mojibake_score(best) >= current_score:
            break
        text = best
        for broken, repaired in _KNOWN_REPLACEMENTS.items():
            text = text.replace(broken, repaired)

    # Known byte sequences have been repaired above. Any remaining C1 code
    # points are control characters, not displayable screenplay content.
    return "".join(
        character for character in text
        if not 0x80 <= ord(character) <= 0x9F
    )


def repair_payload(value: Any) -> Any:
    """Recursively repair text in JSON-like Director state and LLM output."""

    if isinstance(value, str) or isinstance(value, bytes):
        return repair_text(value)
    if isinstance(value, dict):
        return {
            repair_text(key) if isinstance(key, (str, bytes)) else key:
            repair_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [repair_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(repair_payload(item) for item in value)
    return value
