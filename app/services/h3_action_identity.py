"""Resolve narrowly scoped actor labels for deterministic action comparison."""

from __future__ import annotations

import re
from typing import Any


_SHORTHAND = re.compile(r"(?<![\w])(?P<label>[A-Z]|\d+)(?P<possessive>['’]s)?(?![\w])")
_CANONICAL_PREFIX_BEFORE_ALIAS = re.compile(r"\b(?:character|subject|actor|role)\s+$", re.I)
_SUBJECT_POSITION = re.compile(
    r"(?:^|[.!?;:,]|\b(?:then|afterward|afterwards|meanwhile|and|but|while|as|when|before|after)\b)\s*$",
    re.I,
)
_FINITE_AUXILIARIES = frozenset({
    "am", "is", "are", "was", "were", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "may",
    "might", "must", "should",
})
_IRREGULAR_FINITE_VERBS = frozenset({
    "ate", "began", "bent", "broke", "brought", "built", "bought",
    "caught", "chose", "came", "cut", "dealt", "did", "drew", "drank",
    "drove", "fell", "felt", "flew", "forgot", "found", "gave", "got",
    "grew", "held", "hit", "kept", "knew", "left", "lost", "made",
    "met", "paid", "put", "ran", "read", "rode", "rose", "said", "saw",
    "sat", "sent", "set", "shook", "shot", "shut", "sang", "sank",
    "slept", "slid", "spoke", "spent", "stood", "stuck", "swam", "took",
    "tore", "told", "threw", "understood", "woke", "wore", "won", "wrote",
})
_ACTION_WORD = re.compile(r"(?P<word>[a-z][a-z'’-]*)\b", re.I)


def _registered_canonical_label(alias: str, cast_pattern: Any) -> str | None:
    """Return one unambiguous canonical label recognized by the cast regex."""
    fullmatch = getattr(cast_pattern, "fullmatch", None)
    if fullmatch is None:
        return None
    matches = [
        f"{prefix} {alias}"
        for prefix in ("Character", "Subject", "Actor", "Role")
        if fullmatch(f"{prefix} {alias}")
    ]
    return matches[0] if len(matches) == 1 else None


def _has_finite_action_after(text: str, cast_action_re: re.Pattern[str],
                             preview_verbs: dict[str, str], token_stems: Any) -> bool:
    """Recognize a finite action predicate immediately after the alias."""
    tail = text.lstrip(" \t\r\n,;:-")
    # Allow a compact adverbial lead-in without accepting noun phrases like
    # "A very wide shot" as shorthand for a named character.
    for _ in range(3):
        adverb = re.match(r"([a-z][a-z'-]*ly)\s+", tail, re.I)
        if not adverb:
            break
        tail = tail[adverb.end():]
    word_match = _ACTION_WORD.match(tail)
    if not word_match:
        return False
    word = word_match.group("word").casefold()
    remainder = tail[word_match.end():]

    # Finite auxiliaries can introduce an action predicate ("A will break").
    if word in _FINITE_AUXILIARIES:
        if word in {"am", "is", "are", "was", "were"}:
            return True
        following = _ACTION_WORD.match(remainder.lstrip())
        if not following:
            return False
        predicate = following.group("word").casefold()
        stems = token_stems(predicate)
        return bool(
            any(stem in preview_verbs for stem in stems)
            or cast_action_re.match(remainder.lstrip())
        )

    stems = token_stems(word)
    known_action = bool(
        cast_action_re.match(tail)
        or any(stem in preview_verbs for stem in stems)
    )
    finite_form = (
        word.endswith(("s", "ed"))
        or word in _IRREGULAR_FINITE_VERBS
    )
    return known_action and finite_form


def expand_labeled_actor_shorthand(value: Any, cast_pattern: Any) -> str:
    """Expand registered uppercase/numeric actor shorthand in comparison text.

    Only labels uniquely registered as Character X, Subject X, Actor X or
    Role X are eligible. Plain aliases must be clause subjects followed by a
    finite action; possessives identify the actor directly. The input is
    returned unchanged for ambiguity or uncertainty.
    """
    text = str(value or "")
    if not text or cast_pattern is None:
        return text
    try:
        # This helper is called by the ledger's action-frame parser. Importing
        # lazily avoids a module cycle during ledger initialization.
        from services.h3_story_ledger import (
            _CAST_ACTION_RE, _H3_PREVIEW_VERB_CANONICAL,
            _h3_contract_token_stems,
        )
    except Exception:
        return text

    replacements: list[tuple[int, int, str]] = []
    for match in _SHORTHAND.finditer(text):
        alias = match.group("label")
        canonical = _registered_canonical_label(alias, cast_pattern)
        if canonical is None:
            continue

        prefix = text[:match.start()]
        if _CANONICAL_PREFIX_BEFORE_ALIAS.search(prefix):
            # Already inside a full cast label such as Character A.
            continue

        possessive = match.group("possessive")
        if possessive:
            replacement = canonical + possessive
        else:
            if not _SUBJECT_POSITION.search(prefix):
                continue
            if not _has_finite_action_after(
                text[match.end():], _CAST_ACTION_RE,
                _H3_PREVIEW_VERB_CANONICAL, _h3_contract_token_stems,
            ):
                continue
            replacement = canonical
        replacements.append((match.start(), match.end(), replacement))

    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text
