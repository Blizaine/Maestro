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
    # Direct age vocabulary
    "child", "children", "kid", "kids", "minor", "minors", "underage",
    "adolescent", "youngster", "juvenile",
    # Very young
    "baby", "babies", "infant", "toddler", "newborn",
    # School-age contexts
    "elementary student", "middle schooler", "high schooler",
    "middle school student", "high school student",
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
    "naked", "nude", "topless", "bottomless", "stripped", "stripping",
    "moaning", "moans",
    "pumping", "pounding",  # caught the original incident
)

# Numerical age patterns explicitly under 18: "16-year-old", "17 yo",
# "12 years old", etc. Treated as a sexual-side trigger because if a
# screenplay mentions an under-18 age at all in any context near minor
# vocabulary, that's enough to abort.
_AGE_NUMERIC = re.compile(
    r"\b(?:[1-9]|1[0-7])[\s\-]?(?:year|yr|y\.?o\.?)s?(?:[\s\-]?old)?\b",
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

    The numerical-age regex (under 18) counts as a sexual-side trigger:
    any reference to an under-18 age co-occurring with minor vocabulary
    aborts.

    Cheap to call repeatedly — regex objects are lru_cached at module
    level and the typical screenplay is only a few KB.
    """
    if not text:
        return []
    minor_hits = _hits(_minor_regex(), text)
    if not minor_hits:
        return []
    sexual_hits = _hits(_sexual_regex(), text)
    sexual_hits += [m.group(0).lower() for m in _AGE_NUMERIC.finditer(text)]
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
    """Concatenate every free-text field from a Pass-2 shot list into one
    blob suitable for `assert_no_minor_content`.

    Pass 2's output is structured JSON; the text-bearing fields are
    spread across multiple keys per shot. This helper centralizes the
    extraction so callsites stay short and so any future shot-schema
    change updates the safety scan in one place.
    """
    if not shot_dicts:
        return ""
    text_fields = (
        "title", "scene_goal", "narrative_role", "scene_type",
        "environment", "visual_style", "lighting", "mood",
        "image_prompt", "video_prompt", "ending_beat",
    )
    array_fields = (
        "action_beats", "visual_changes",
        "keyframe_prompts", "window_prompts",
    )
    parts: list[str] = []
    for sd in shot_dicts:
        if not isinstance(sd, dict):
            continue
        for f in text_fields:
            v = sd.get(f)
            if isinstance(v, str):
                parts.append(v)
        for f in array_fields:
            v = sd.get(f) or []
            if isinstance(v, list):
                parts.extend(x for x in v if isinstance(x, str))
        for subj in (sd.get("subjects_on_screen") or []):
            if isinstance(subj, dict):
                for k in ("visual_description", "speaker_name",
                          "position_or_relation"):
                    v = subj.get(k)
                    if isinstance(v, str):
                        parts.append(v)
        for db in (sd.get("dialogue_beats") or []):
            if isinstance(db, dict):
                v = db.get("spoken_text")
                if isinstance(v, str):
                    parts.append(v)
    return "\n".join(parts)
