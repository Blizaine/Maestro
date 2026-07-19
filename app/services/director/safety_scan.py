"""
Base content-safety scanner for the Director pipeline.

Refuses screenplays / shot lists that depict minors in sexual contexts.
This is a HARD rule. It is not a guidance hint, not a user-toggleable
setting, and not gated on any optional download — it ships with the
public repo and is always active.

Hybrid co-occurrence detection: a match requires BOTH minor-vocabulary
AND sexual-vocabulary present in the same scanned blob. This avoids
false positives on legitimate non-sexual content involving minors
(children's films, family stories) AND on legitimate adult NSFW content.
A match aborts the generation pipeline via SafetyViolationError, which
the pipeline error handler in director_pipeline.py converts into a clean
user-visible message in the chat.

Forbidden vocabulary lives in inline Python tuples below — not JSON, not
a separate data file — so the rule cannot be silenced by deleting a data
file. Forking the project to remove these would require editing source.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable


class SafetyViolationError(RuntimeError):
    """Raised when scanned text contains hard-blocked content categories.

    Carries `source` (where the violating text came from — "user concept",
    "screenplay (Pass 1)", etc.) and `matched_terms` (the actual words
    that tripped the scan) so the pipeline error handler can build a
    deterministic, jailbreak-resistant user-facing message without
    reproducing the offending text itself.
    """

    def __init__(self, source: str, matched_terms: list[str]):
        self.source = source
        self.matched_terms = matched_terms
        super().__init__(
            f"Safety violation in {source}: blocked terms detected — "
            f"{', '.join(sorted(set(matched_terms)))}"
        )


# ── Forbidden vocabulary ─────────────────────────────────────────────
# Keep these lists complete and conservative. False positives are
# preferable to false negatives in this category. The hybrid co-occurrence
# check below mitigates false positives by requiring sexual vocabulary
# in the same blob.

_MINOR_TERMS: tuple[str, ...] = (
    # Direct age vocabulary.
    "child", "children", "kid", "kids", "minor", "minors", "underage",
    "adolescent", "adolescents", "youngster", "youngsters",
    "juvenile", "juveniles",
    # Very young.
    "baby", "babies", "infant", "infants", "toddler", "toddlers",
    "newborn", "newborns",
    # Pre-adult age descriptors.
    "pre-teen", "pre-teens", "preteen", "preteens", "tween", "tweens",
    "teen", "teens", "teenager", "teenagers", "teenage",
    # School-age contexts.
    "schoolgirl", "schoolgirls", "schoolboy", "schoolboys",
    "junior", "juniors",
    "elementary student", "elementary students",
    "middle schooler", "middle schoolers", "high schooler", "high schoolers",
    "middle school student", "middle school students",
    "high school student", "high school students",
    # Unambiguously young gendered phrases. Plain "girl" and "boy" are omitted
    # to avoid flagging common adult descriptions.
    "little girl", "little girls", "little boy", "little boys",
    "young girl", "young girls", "young boy", "young boys",
    # Family relationships that commonly imply a minor in prompt context.
    "daughter", "daughters", "son", "sons",
    "stepdaughter", "stepdaughters", "stepson", "stepsons",
    "niece", "nieces", "nephew", "nephews",
    "granddaughter", "granddaughters", "grandson", "grandsons",
    "under 18",
)

_SEXUAL_TERMS: tuple[str, ...] = (
    # Acts (verb / noun forms)
    "sex", "sexual", "sexually",
    "fuck", "fucking", "fucked", "fucks",
    "blowjob", "handjob", "rimjob", "cunnilingus", "fellatio",
    "intercourse", "coitus",
    "penetrate", "penetration", "penetrating", "penetrated",
    "thrusting", "thrust", "thrusts",
    "orgasm", "orgasmic", "ejaculate", "ejaculation", "ejaculating",
    "cum", "cumming", "creampie",
    "straddling", "straddles", "straddled",
    "riding", "humping", "grinding", "grinds",
    "fingering", "fingered",
    # Anatomy in sexual context
    "penis", "cock", "vagina", "pussy", "clitoris", "clit",
    "breasts", "boobs", "tits", "nipples", "genitals", "genitalia",
    "buttocks", "anus",
    "erection", "erect", "aroused", "horny",
    # State / posture descriptors that almost always imply sex context
    "naked", "nude", "topless", "bottomless",
    "undress", "undresses", "undressed", "undressing",
    "strip", "strips", "stripped", "stripping",
    "moaning", "moans",
    "pumping", "pounding",  # caught the original incident
)

# Numerical age patterns explicitly under 18: "16-year-old", "age: 16",
# "17 y/o", etc. These are minor-side evidence and still require separate
# sexual vocabulary before the scanner blocks the text.
_AGE_NUMERIC = re.compile(
    r"(?<!\w)(?:(?:age|aged)(?:\s*[:=]\s*|\s+)(?:[1-9]|1[0-7])|"
    r"(?:[1-9]|1[0-7])\s*(?:-\s*)?(?:(?:years?|yrs?)"
    r"(?:(?:\s*-\s*|\s+)old)?|y\.?\s*/?\s*o\.?))(?!\w)",
    re.IGNORECASE,
)


def _build_regex(terms: Iterable[str]) -> "re.Pattern[str]":
    """Compile a case-insensitive word-boundary alternation from a term list.

    Sorts terms by length descending so multi-word terms ("pre-teen",
    "little girl") match before their single-word substrings ("teen",
    "girl") would otherwise consume their head.
    """
    sorted_terms = sorted(terms, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(t) for t in sorted_terms) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


@lru_cache(maxsize=1)
def _minor_regex() -> "re.Pattern[str]":
    return _build_regex(_MINOR_TERMS)


@lru_cache(maxsize=1)
def _sexual_regex() -> "re.Pattern[str]":
    return _build_regex(_SEXUAL_TERMS)


def _hits(pattern: "re.Pattern[str]", text: str) -> list[str]:
    return [m.group(0).lower() for m in pattern.finditer(text)]


def screenplay_contains_minor_content(text: str) -> list[str]:
    """Hybrid co-occurrence scan.

    Returns the union of minor-vocabulary and sexual-vocabulary matches
    when BOTH categories are present in `text`. Returns empty list when:
      - text is empty / falsy
      - only minor vocabulary present (e.g. children's content)
      - only sexual vocabulary present (e.g. adult-only NSFW)

    The numerical-age regex (under 18) counts as minor vocabulary. It must
    still co-occur with separate sexual vocabulary before the scan blocks.

    Cheap to call repeatedly — regex objects are lru_cached at module
    level and the typical screenplay is only a few KB.
    """
    if not text:
        return []
    minor_hits = _hits(_minor_regex(), text)
    minor_hits += [m.group(0).lower() for m in _AGE_NUMERIC.finditer(text)]
    if not minor_hits:
        return []
    sexual_hits = _hits(_sexual_regex(), text)
    if not sexual_hits:
        return []
    return sorted(set(minor_hits + sexual_hits))


def assert_no_minor_content(text: str, source: str) -> None:
    """Raise SafetyViolationError if `text` contains minor + sexual co-occurrence.

    `source` is a short human-readable label describing where the text
    came from ("user concept", "screenplay (Pass 1)", "shot list (Pass 2)").
    It surfaces in the user-visible error and in logs so failures are
    traceable to the right pipeline stage.
    """
    matches = screenplay_contains_minor_content(text)
    if matches:
        raise SafetyViolationError(source=source, matched_terms=matches)


def collect_pass2_text(shot_dicts: list[dict]) -> str:
    """Recursively concatenate every string value from a Pass-2 shot list.

    Pass 2's output is nested structured JSON. Traversing all dictionary
    values and list/tuple members makes the safety boundary resilient to
    additions or reorganizations in the shot schema.
    """
    if not shot_dicts:
        return ""
    parts: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                collect(nested)

    collect(shot_dicts)
    return "\n".join(parts)
