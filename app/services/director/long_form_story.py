"""Deterministic continuity support for long-form Director projects.

The creative model still invents the film.  This module owns the durable
contracts that a small local model should not have to rediscover in every
chapter: cast and location registries, recurring story machinery, handoffs,
and a conservative post-plan dialogue cleanup.
"""

from __future__ import annotations

import copy
import difflib
import re
from typing import Any, Iterable, Optional, Sequence


LONG_FORM_STORY_BIBLE_REVISION = 2


LONG_FORM_STORY_BIBLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "working_title": {"type": "string"},
        "logline": {"type": "string"},
        "story_engine": {"type": "string"},
        "tone_contract": {"type": "string"},
        "ending_contract": {"type": "string"},
        "allow_cast_expansion": {"type": "boolean"},
        "allow_location_expansion": {"type": "boolean"},
        "canonical_characters": {
            "type": "array",
            "maxItems": 40,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "initial_state": {"type": "string"},
                    "continuity_rules": {"type": "string"},
                },
                "required": [
                    "name", "role", "initial_state", "continuity_rules",
                ],
                "additionalProperties": False,
            },
        },
        "canonical_locations": {
            "type": "array",
            "maxItems": 40,
            "items": {
                "type": "object",
                "properties": {
                    "location_id": {"type": "string"},
                    "name": {"type": "string"},
                    "visual_identity": {"type": "string"},
                    "story_function": {"type": "string"},
                },
                "required": [
                    "location_id", "name", "visual_identity", "story_function",
                ],
                "additionalProperties": False,
            },
        },
        "recurring_motifs": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "motif_id": {"type": "string"},
                    "description": {"type": "string"},
                    "min_occurrences": {"type": "integer", "minimum": 2},
                    "max_occurrences": {"type": "integer", "minimum": 2},
                    "dialogue_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "source_event_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "variation_rule": {"type": "string"},
                },
                "required": [
                    "motif_id", "description", "min_occurrences",
                    "max_occurrences", "dialogue_ids", "source_event_ids",
                    "variation_rule",
                ],
                "additionalProperties": False,
            },
        },
        "world_rules": {"type": "array", "items": {"type": "string"}},
        "forbidden_drift": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "working_title", "logline", "story_engine", "tone_contract",
        "ending_contract", "allow_cast_expansion", "allow_location_expansion",
        "canonical_characters", "canonical_locations", "recurring_motifs",
        "world_rules", "forbidden_drift",
    ],
    "additionalProperties": False,
}


_GENERIC_SPEAKERS = {
    "he", "her", "him", "narration", "she", "speaker", "they", "them",
    "voice", "voiceover", "we", "you",
}
_PSEUDO_SPEAKERS = {
    "action", "ambience", "boom", "camera", "crash", "cut", "impact",
    "music", "silence", "snap", "sound", "the snap", "the snap hits",
    "thud", "whoosh",
}
_LOCATION_EXPANSION_RE = re.compile(
    r"\b(?:and\s+(?:many|several)\s+more|many\s+more|and\s+so\s+on|"
    r"various\s+(?:shows|worlds|rooms|locations|places|eras)|"
    r"different\s+(?:shows|worlds|rooms|locations|places|eras)|"
    r"go(?:es|ing)?\s+around\s+to|visits?\s+(?:different|many|several))\b",
    flags=re.IGNORECASE,
)
_CAST_EXPANSION_RE = re.compile(
    r"\b(?:and\s+(?:many|several)\s+more|many\s+more|and\s+so\s+on|"
    r"various\s+(?:shows|worlds|families|groups|characters|people|ensembles)|"
    r"different\s+(?:shows|worlds|families|groups|characters|people|ensembles)|"
    r"go(?:es|ing)?\s+around\s+to\s+different\s+(?:shows|worlds)|"
    r"visits?\s+(?:different|many|several)\s+(?:shows|worlds|families|groups))\b",
    flags=re.IGNORECASE,
)
_EXPANSION_RE = re.compile(
    rf"(?:{_LOCATION_EXPANSION_RE.pattern}|{_CAST_EXPANSION_RE.pattern})",
    flags=re.IGNORECASE,
)
_RECURRING_RE = re.compile(
    r"\b(?:each\s+time|every\s+time|whenever|repeatedly|again\s+and\s+again|"
    r"in\s+(?:each|every|whatever)\s+(?:room|show|world|place|location)|"
    r"whatever\s+(?:room|show|world|place|location)|go(?:es|ing)?\s+around\s+"
    r"to\s+different|visits?\s+(?:different\s+)?(?:shows|worlds|places|"
    r"locations))\b",
    flags=re.IGNORECASE,
)
_EVENT_ACTION_RE = re.compile(
    r"\b(?:says?|speaks?|asks?|replies?|snaps?|turns?\s+to\s+dust|"
    r"repeats?|performs?|does?\s+the\s+same|catchphrase)\b",
    flags=re.IGNORECASE,
)


def _clean(value: Any, *, limit: int = 800) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n,;:-")
    return text[:limit].rstrip() if len(text) > limit else text


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _slug(value: Any, fallback: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    return result[:48] or fallback


def _unique_strings(values: Iterable[Any], *, limit: int = 60) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value, limit=500)
        folded = cleaned.casefold()
        if not cleaned or folded in seen:
            continue
        seen.add(folded)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _extract_named_candidates(story: str) -> list[str]:
    without_quotes = re.sub(r'"[^"\r\n]*"|“[^”\r\n]*”', " ", story)
    result: list[str] = []
    rejected = {
        "a", "an", "and", "chapter", "director", "each", "every", "film",
        "first", "last", "many", "maestro", "scene", "the", "then", "tv",
        "video",
    }
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3}\b",
        without_quotes,
    ):
        name = _clean(match.group(0), limit=100)
        while name.split() and name.split()[0].casefold() in {"a", "an", "the", "then"}:
            name = " ".join(name.split()[1:])
        key = _key(name)
        if (
            not name
            or name.casefold() in rejected
            or key in _GENERIC_SPEAKERS
            or key in _PSEUDO_SPEAKERS
            or any(_key(existing) == key for existing in result)
        ):
            continue

        # A capitalized work or destination in a list is not a person.  The
        # previous fallback interpreted phrases such as "visits TV shows,
        # Friends, The Office, Parks and Rec" as five cast members (including
        # "He").  Those phantom identities then polluted every bounded
        # chapter's cast registry.  Keep this deliberately contextual so a
        # real character named in ordinary prose is still discovered.
        prefix = without_quotes[max(0, match.start() - 140):match.start()]
        sentence_start = max(
            without_quotes.rfind(marker, 0, match.start())
            for marker in ".!?"
        ) + 1
        sentence_end_candidates = [
            position for marker in ".!?"
            for position in [without_quotes.find(marker, match.end())]
            if position >= 0
        ]
        sentence_end = (
            min(sentence_end_candidates)
            if sentence_end_candidates else len(without_quotes)
        )
        sentence = without_quotes[sentence_start:sentence_end]
        visit_list = bool(
            re.search(r"\bvisits?\b", sentence, flags=re.IGNORECASE)
            and sentence.count(",") >= 2
        )
        if re.search(
            r"\b(?:visits?|travels?\s+through|go(?:es|ing)?\s+(?:around\s+)?to|"
            r"including|such\s+as)\s+(?:(?:different|many|several|iconic)\s+)?"
            r"(?:(?:tv|television|streaming)\s+)?"
            r"(?:shows?|series|sitcoms?|worlds?|locations?|places?|rooms?)"
            r"\b[^.!?]{0,100}$",
            prefix,
            flags=re.IGNORECASE,
        ) or re.search(
            r"\b(?:(?:tv|television|streaming)\s+)?"
            r"(?:show|series|sitcom|film|movie|episode)\s+"
            r"(?:called\s+|named\s+)?$",
            prefix,
            flags=re.IGNORECASE,
        ) or visit_list:
            continue
        result.append(name)
    return result


def _locked_dialogue_character_names(
    locked_dialogue: Sequence[dict[str, Any]],
) -> list[str]:
    """Return only concrete human speakers from the immutable dialogue map."""

    result: list[str] = []
    for item in locked_dialogue or []:
        speaker = _clean(item.get("speaker"), limit=100)
        key = _key(speaker)
        if (
            not speaker
            or key in _GENERIC_SPEAKERS
            or key in _PSEUDO_SPEAKERS
            or re.fullmatch(r"(?:speaker|subject|character)[ _-]*\d+", key)
        ):
            continue
        result.append(speaker)
    return _unique_strings(result, limit=40)


def _fallback_recurring_motif(
    story: str,
    *,
    locked_dialogue: Sequence[dict[str, Any]],
    source_events: Sequence[dict[str, Any]],
    chapter_count: int,
) -> list[dict[str, Any]]:
    if not _RECURRING_RE.search(story):
        return []
    reusable_events = [
        str(item.get("event_id") or "").upper()
        for item in source_events
        if item.get("event_id") and _EVENT_ACTION_RE.search(str(item.get("text") or ""))
    ]
    dialogue_ids = [
        str(item.get("dialogue_id") or "").upper()
        for item in locked_dialogue
        if item.get("dialogue_id")
    ]
    if not reusable_events and not dialogue_ids:
        return []
    occurrences = max(2, min(max(2, chapter_count - 2), 12))
    return [{
        "motif_id": "M1",
        "description": (
            "Repeat the user's central encounter or action pattern in each "
            "assigned chapter while changing the setting, reaction, obstacle, "
            "and consequence; it is a deliberate evolving motif, not reused footage"
        ),
        "min_occurrences": occurrences,
        "max_occurrences": occurrences,
        "dialogue_ids": dialogue_ids[:1] if len(dialogue_ids) == 1 else [],
        "source_event_ids": reusable_events,
        "variation_rule": (
            "Preserve the motif's required order and any attached exact line, "
            "but give every occurrence a new setup, reaction, escalation, and handoff"
        ),
    }]


def build_long_form_story_bible_fallback(
    story_description: str,
    *,
    locked_dialogue: Sequence[dict[str, Any]],
    source_events: Sequence[dict[str, Any]],
    character_names: Sequence[str],
    chapter_count: int,
) -> dict[str, Any]:
    """Create a safe story memory when the creative bible call is unavailable."""

    story = _clean(story_description, limit=12000)
    names = _unique_strings([
        *character_names,
        *_locked_dialogue_character_names(locked_dialogue),
        *_extract_named_candidates(story),
    ], limit=40)
    expansion = bool(_EXPANSION_RE.search(story))
    cast_expansion = bool(_CAST_EXPANSION_RE.search(story))
    location_expansion = bool(_LOCATION_EXPANSION_RE.search(story))
    return {
        "revision": LONG_FORM_STORY_BIBLE_REVISION,
        "registry_complete": False,
        "working_title": "Long-form Director project",
        "logline": story[:500],
        "story_engine": (
            "An escalating episodic progression whose variations cumulatively "
            "advance one story"
            if expansion else
            "One causal story whose scenes create the next scene"
        ),
        "tone_contract": (
            "Preserve the user's requested tone, genre, humor, intensity, and "
            "rating throughout; do not drift into a different moral or genre"
        ),
        "ending_contract": (
            "Resolve only the outcome requested or strongly implied by the user; "
            "do not invent redemption, reversal, or sequel setup"
        ),
        "allow_cast_expansion": cast_expansion,
        "allow_location_expansion": location_expansion,
        "canonical_characters": [
            {
                "name": name,
                "role": "User-specified character",
                "initial_state": "As established by the user concept",
                "continuity_rules": "Preserve identity and accumulated physical and relationship state",
            }
            for name in names
        ],
        "canonical_locations": [],
        "recurring_motifs": _fallback_recurring_motif(
            story,
            locked_dialogue=locked_dialogue,
            source_events=source_events,
            chapter_count=chapter_count,
        ),
        "world_rules": [
            "Every chapter begins from the concrete result of the prior chapter",
            "A character, prop, injury, disappearance, or fact persists until an on-screen event changes it",
            "A new location is reached through an established departure, transition, or consequence",
        ],
        "forbidden_drift": [
            "No unrelated filler vignette, unexplained reset, duplicate principal, or recap",
            "No invented ending that contradicts the user's tone or requested outcome",
            "No stage direction, sound effect, or camera instruction may become spoken dialogue",
        ],
    }


def normalize_long_form_story_bible(
    value: Any,
    *,
    story_description: str,
    locked_dialogue: Sequence[dict[str, Any]],
    source_events: Sequence[dict[str, Any]],
    character_names: Sequence[str],
    chapter_count: int,
) -> dict[str, Any]:
    """Validate an LLM-authored bible and merge immutable source facts."""

    fallback = build_long_form_story_bible_fallback(
        story_description,
        locked_dialogue=locked_dialogue,
        source_events=source_events,
        character_names=character_names,
        chapter_count=chapter_count,
    )
    if not isinstance(value, dict):
        return fallback
    bible = copy.deepcopy(fallback)
    for field in (
        "working_title", "logline", "story_engine", "tone_contract",
        "ending_contract",
    ):
        cleaned = _clean(value.get(field), limit=1200)
        if cleaned:
            bible[field] = cleaned
    for field in ("allow_cast_expansion", "allow_location_expansion"):
        if isinstance(value.get(field), bool):
            # Explicit user requests such as "many more" cannot be silently
            # narrowed by an architect that happens to return false.
            bible[field] = bool(value[field]) or bool(fallback[field])

    source_character_names = _unique_strings([
        *character_names,
        *_locked_dialogue_character_names(locked_dialogue),
        *_extract_named_candidates(story_description),
    ], limit=40)
    characters: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for raw in value.get("canonical_characters") or []:
        if not isinstance(raw, dict):
            continue
        name = _clean(raw.get("name"), limit=100)
        if not name or _key(name) in seen_names:
            continue
        role = _clean(raw.get("role"), limit=300) or "Story character"
        # Revision-1 deterministic fallback bibles labeled every extracted
        # capitalized noun as a user-specified character.  On resume, discard
        # only those legacy fallback entries that cannot be matched back to a
        # supplied reference, a real quoted-line speaker, or a safe character
        # candidate in the user's concept.  Creative LLM-authored supporting
        # cast remains untouched.
        if (
            role.casefold() == "user-specified character"
            and source_character_names
            and not _canonical_name(name, source_character_names)
        ):
            continue
        if _key(name) in _GENERIC_SPEAKERS or _key(name) in _PSEUDO_SPEAKERS:
            continue
        seen_names.add(_key(name))
        characters.append({
            "name": name,
            "role": role,
            "initial_state": _clean(raw.get("initial_state"), limit=500) or "As established in the concept",
            "continuity_rules": _clean(raw.get("continuity_rules"), limit=500) or "Preserve accumulated state",
        })
    for name in _unique_strings(character_names, limit=40):
        if _key(name) in seen_names:
            continue
        seen_names.add(_key(name))
        characters.append({
            "name": name,
            "role": "User-specified character",
            "initial_state": "As established by the user concept",
            "continuity_rules": "Preserve identity and accumulated state",
        })
    bible["canonical_characters"] = characters[:40] or fallback["canonical_characters"]

    locations: list[dict[str, str]] = []
    seen_locations: set[str] = set()
    for index, raw in enumerate(value.get("canonical_locations") or [], start=1):
        if not isinstance(raw, dict):
            continue
        name = _clean(raw.get("name"), limit=180)
        if not name or _key(name) in seen_locations:
            continue
        seen_locations.add(_key(name))
        locations.append({
            "location_id": _slug(raw.get("location_id") or name, f"location_{index}"),
            "name": name,
            "visual_identity": _clean(raw.get("visual_identity"), limit=500) or "Preserve the location's established visual identity",
            "story_function": _clean(raw.get("story_function"), limit=400) or "Advance the same story",
        })
    bible["canonical_locations"] = locations[:40]

    allowed_dialogue = {
        str(item.get("dialogue_id") or "").upper()
        for item in locked_dialogue if item.get("dialogue_id")
    }
    allowed_events = {
        str(item.get("event_id") or "").upper()
        for item in source_events if item.get("event_id")
    }
    motifs: list[dict[str, Any]] = []
    seen_motifs: set[str] = set()
    for index, raw in enumerate(value.get("recurring_motifs") or [], start=1):
        if not isinstance(raw, dict):
            continue
        description = _clean(raw.get("description"), limit=800)
        if not description:
            continue
        motif_id = str(raw.get("motif_id") or f"M{index}").upper()
        motif_id = re.sub(r"[^A-Z0-9_-]+", "", motif_id) or f"M{index}"
        if motif_id in seen_motifs:
            continue
        seen_motifs.add(motif_id)
        try:
            minimum = int(raw.get("min_occurrences") or 2)
        except (TypeError, ValueError):
            minimum = 2
        try:
            maximum = int(raw.get("max_occurrences") or minimum)
        except (TypeError, ValueError):
            maximum = minimum
        minimum = max(2, min(max(2, chapter_count), minimum))
        maximum = max(minimum, min(max(2, chapter_count), maximum))
        motifs.append({
            "motif_id": motif_id,
            "description": description,
            "min_occurrences": minimum,
            "max_occurrences": maximum,
            "dialogue_ids": _unique_strings(
                str(item or "").upper()
                for item in raw.get("dialogue_ids") or []
                if str(item or "").upper() in allowed_dialogue
            ),
            "source_event_ids": _unique_strings(
                str(item or "").upper()
                for item in raw.get("source_event_ids") or []
                if str(item or "").upper() in allowed_events
            ),
            "variation_rule": _clean(raw.get("variation_rule"), limit=700) or "Change setup, reaction, escalation, and consequence every time",
        })
    if not motifs:
        motifs = fallback["recurring_motifs"]
    bible["recurring_motifs"] = motifs[:12]
    bible["world_rules"] = _unique_strings([
        *(value.get("world_rules") or []), *fallback["world_rules"],
    ], limit=16)
    bible["forbidden_drift"] = _unique_strings([
        *(value.get("forbidden_drift") or []), *fallback["forbidden_drift"],
    ], limit=16)
    bible["revision"] = LONG_FORM_STORY_BIBLE_REVISION
    supplied_count = len(_unique_strings(character_names, limit=40))
    bible["registry_complete"] = bool(characters) and (
        not bible.get("allow_cast_expansion")
        or len(characters) > supplied_count
    )
    return bible


def long_form_story_bible_quality_issues(
    story_bible: dict[str, Any],
    *,
    chapter_count: int,
) -> list[str]:
    """Find macro-plan gaps worth one focused creative repair.

    This deliberately checks only structural coverage.  It does not grade the
    model's taste or rewrite its story, and it never interprets a simple list
    of visual destinations as a recurring plot motif.
    """

    issues: list[str] = []
    if story_bible.get("allow_location_expansion") and chapter_count > 1:
        location_count = len(story_bible.get("canonical_locations") or [])
        target = min(chapter_count, max(4, int(round(chapter_count * 0.67))))
        if location_count < target:
            issues.append(
                f"location registry has {location_count} distinct places; "
                f"the requested expanding story needs at least {target}"
            )
    if story_bible.get("allow_cast_expansion"):
        cast_count = len(story_bible.get("canonical_characters") or [])
        # A multi-world ensemble story needs more than one guest character,
        # but should not be forced to enumerate dozens of silent extras.
        target = min(8, max(3, int(round(chapter_count * 0.42))))
        if cast_count < target:
            issues.append(
                f"cast registry has {cast_count} named characters; the "
                f"requested ensemble expansion needs at least {target}"
            )
    return issues


def long_form_outline_quality_issues(
    rows: Sequence[dict[str, Any]],
    *,
    story_bible: dict[str, Any],
    chapter_count: int,
) -> list[str]:
    """Find a collapsed chapter arc before expensive sequence writing."""

    planned = [item for item in rows if isinstance(item, dict)][:chapter_count]
    if len(planned) != chapter_count:
        return [f"chapter outline has {len(planned)} of {chapter_count} chapters"]
    issues: list[str] = []
    objectives = {
        _key(item.get("objective")) for item in planned if _key(item.get("objective"))
    }
    minimum_objectives = min(chapter_count, max(3, int(round(chapter_count * 0.67))))
    if len(objectives) < minimum_objectives:
        issues.append(
            f"only {len(objectives)} distinct chapter objectives were planned; "
            f"at least {minimum_objectives} are required"
        )
    if story_bible.get("allow_location_expansion"):
        locations = {
            _key(item.get("location_id") or item.get("location_time"))
            for item in planned
            if _key(item.get("location_id") or item.get("location_time"))
        }
        registry_count = len(story_bible.get("canonical_locations") or [])
        minimum_locations = min(
            chapter_count,
            registry_count,
            max(3, int(round(chapter_count * 0.60))),
        )
        if minimum_locations and len(locations) < minimum_locations:
            issues.append(
                f"only {len(locations)} registered locations are represented; "
                f"at least {minimum_locations} are required"
            )
    return issues


def ensure_long_form_location_coverage(
    rows: Sequence[dict[str, Any]],
    *,
    story_bible: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Use unused approved locations when an expanding arc still repeats.

    The creative calls choose the location registry and chapter objectives.
    This safety net changes neither: it only assigns an unused registered
    location to later duplicate-location chapters until minimum coverage is
    reached.  It therefore remains useful when a small local model ignores a
    focused diversity repair without becoming a deterministic story writer.
    """

    planned = [dict(item) for item in rows if isinstance(item, dict)]
    if not story_bible.get("allow_location_expansion") or len(planned) < 2:
        return planned, []
    registry = [
        item for item in story_bible.get("canonical_locations") or []
        if isinstance(item, dict) and _clean(item.get("location_id"), limit=80)
    ]
    required = min(
        len(planned),
        len(registry),
        max(3, int(round(len(planned) * 0.60))),
    )
    if required <= 1:
        return planned, []
    registry_by_id = {
        _key(item.get("location_id")): item for item in registry
    }
    represented: set[str] = set()
    duplicate_indices: list[int] = []
    for index, row in enumerate(planned):
        location_key = _key(row.get("location_id"))
        if location_key and location_key not in represented:
            represented.add(location_key)
        else:
            duplicate_indices.append(index)
    unused = [
        item for item in registry
        if _key(item.get("location_id")) not in represented
    ]
    warnings: list[str] = []
    while len(represented) < required and duplicate_indices and unused:
        index = duplicate_indices.pop(0)
        location = unused.pop(0)
        location_id = _clean(location.get("location_id"), limit=80)
        location_name = _clean(location.get("name"), limit=180) or location_id
        if not location_id or _key(location_id) not in registry_by_id:
            continue
        row = planned[index]
        prior_opening = _clean(row.get("opening_state"), limit=650)
        row["location_id"] = location_id
        row["location_time"] = location_name
        row["opening_state"] = _clean(
            f"Arrive in {location_name} through the preceding causal handoff. "
            f"{prior_opening}",
            limit=900,
        )
        row["location_coverage_repaired"] = True
        represented.add(_key(location_id))
        warnings.append(
            f"chapter {index + 1}: assigned unused registered location "
            f"{location_name!r} to prevent long-form location cycling"
        )
    return planned, warnings


def resolve_locked_dialogue_speakers(
    locked_dialogue: Sequence[dict[str, Any]],
    *,
    story_description: str,
    story_bible: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve pronoun-only quote attribution against the preceding named cast."""

    result = [dict(item) for item in locked_dialogue]
    names = [
        _clean(item.get("name"), limit=100)
        for item in story_bible.get("canonical_characters") or []
        if isinstance(item, dict) and _clean(item.get("name"), limit=100)
    ]
    for item in result:
        speaker = _clean(item.get("speaker"), limit=100)
        if _key(speaker) not in _GENERIC_SPEAKERS:
            continue
        try:
            offset = int(item.get("source_offset") or 0)
        except (TypeError, ValueError):
            offset = 0
        prefix = story_description[:max(0, offset)].casefold()
        mentioned = [
            (prefix.rfind(name.casefold()), name)
            for name in names
            if prefix.rfind(name.casefold()) >= 0
        ]
        if mentioned:
            item["speaker"] = max(mentioned)[1]
        elif len(names) == 1:
            item["speaker"] = names[0]
    return result


def normalize_long_form_outline(
    rows: Sequence[dict[str, Any]],
    *,
    story_bible: dict[str, Any],
    chapter_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind chapter rows to the shared cast/location registry and handoffs."""

    bible = copy.deepcopy(story_bible)
    normalized = [dict(row) for row in rows if isinstance(row, dict)][:chapter_count]
    character_names = [
        _clean(item.get("name"), limit=100)
        for item in bible.get("canonical_characters") or []
        if isinstance(item, dict) and _clean(item.get("name"), limit=100)
    ]
    locations = [
        item for item in bible.get("canonical_locations") or []
        if isinstance(item, dict) and item.get("location_id")
    ]
    location_by_id = {
        _key(item["location_id"]): item for item in locations
    }
    location_by_name = {_key(item.get("name")): item for item in locations}
    unavailable_characters: dict[str, str] = {}

    for index, row in enumerate(normalized):
        row["chapter"] = index + 1
        raw_location_id = _key(row.get("location_id"))
        raw_location_name = _clean(row.get("location_time"), limit=240)
        location = location_by_id.get(raw_location_id) or location_by_name.get(
            _key(raw_location_name)
        )
        if location is None and raw_location_name and bible.get("allow_location_expansion"):
            location = {
                "location_id": _slug(raw_location_name, f"location_{len(locations) + 1}"),
                "name": raw_location_name,
                "visual_identity": "Preserve the exact established set geography and period details",
                "story_function": _clean(row.get("objective"), limit=400) or "Advance this chapter",
            }
            locations.append(location)
            location_by_id[_key(location["location_id"])] = location
            location_by_name[_key(location["name"])] = location
        elif location is None and locations:
            location = locations[index % len(locations)]
        if location:
            row["location_id"] = location["location_id"]
            row["location_time"] = raw_location_name or location["name"]
        else:
            row["location_id"] = _slug(raw_location_name, f"chapter_{index + 1}_location")

        cast = _unique_strings(row.get("cast_present") or [], limit=30)
        if not bible.get("allow_cast_expansion"):
            allowed = {_key(name): name for name in character_names}
            cast = [allowed[_key(name)] for name in cast if _key(name) in allowed]
        state_changes = _unique_strings(
            row.get("character_state_changes") or [],
            limit=30,
        )
        complete_row_text = " ".join([
            *(_clean(row.get(field), limit=900) for field in (
                "objective", "opening_state", "closing_state", "persistent_state",
            )),
            *state_changes,
        ])
        inherited_character_state = dict(unavailable_characters)
        for name in list(unavailable_characters):
            restore = bool(re.search(
                rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9]).{{0,100}}"
                r"\b(?:returns?|restored|resurrected|reappears?|reconstituted|revived)\b",
                complete_row_text,
                flags=re.IGNORECASE,
            ))
            if restore:
                unavailable_characters.pop(name, None)
        if not cast:
            # Missing cast must not silently put every person in the story
            # bible into every chapter.  Recover only characters explicitly
            # mentioned by this chapter, falling back to the first (lead)
            # registry entry when the architect omitted all cast information.
            cast = [
                name for name in character_names
                if re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
                    complete_row_text,
                    flags=re.IGNORECASE,
                )
            ]
            if not cast and character_names:
                cast = character_names[:1]
        filtered_cast: list[str] = []
        for name in cast:
            canonical = _canonical_name(name, character_names) or name
            unavailable = next(
                (
                    state_name for state_name in unavailable_characters
                    if _key(state_name) == _key(canonical)
                ),
                "",
            )
            if unavailable:
                continue
            filtered_cast.append(canonical)
        row["cast_present"] = _unique_strings(filtered_cast, limit=30)
        row["character_state_changes"] = state_changes
        row["_character_availability_before"] = inherited_character_state
        row["inherited_character_state"] = (
            "; ".join(
                f"{name}: {state}"
                for name, state in inherited_character_state.items()
            )
            or "No irreversible named-character state change is currently active"
        )
        row["recurring_motif_ids"] = _unique_strings(
            str(item or "").upper() for item in row.get("recurring_motif_ids") or []
        )
        if index == 0:
            row["inherited_state"] = "The initial state defined by the user concept and story bible"
        else:
            previous = normalized[index - 1]
            inherited = "; ".join(_unique_strings([
                previous.get("closing_state"), previous.get("persistent_state"),
            ], limit=2))
            row["inherited_state"] = inherited or "The complete visible result of the preceding chapter"
            opening = _clean(row.get("opening_state"), limit=500)
            prior_close = _clean(previous.get("closing_state"), limit=500)
            if prior_close and _key(prior_close) not in _key(opening):
                row["opening_state"] = _clean(
                    f"Open on the direct visible consequence of {prior_close}. {opening}",
                    limit=850,
                )
        for change in state_changes:
            folded = change.casefold()
            for name in character_names:
                if not re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
                    change,
                    flags=re.IGNORECASE,
                ):
                    continue
                if re.search(
                    r"\b(?:returns?|restored|resurrected|reappears?|reconstituted|revived)\b",
                    folded,
                ):
                    unavailable_characters.pop(name, None)
                elif re.search(
                    r"\b(?:dead|dies|killed|turns?\s+to\s+dust|disintegrat(?:es|ed)|"
                    r"vanish(?:es|ed)|erased\s+from\s+existence)\b",
                    folded,
                ):
                    unavailable_characters[name] = change
        row["_character_availability_after"] = dict(unavailable_characters)
    bible["canonical_locations"] = locations[:40]
    return normalized, bible


def normalize_long_form_sequence_states(
    rows: Sequence[dict[str, Any]],
    *,
    story_bible: dict[str, Any],
    inherited_unavailable: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Carry named-character availability through bounded sequences."""

    normalized = [dict(row) for row in rows if isinstance(row, dict)]
    names = [
        _clean(item.get("name"), limit=100)
        for item in story_bible.get("canonical_characters") or []
        if isinstance(item, dict) and _clean(item.get("name"), limit=100)
    ]
    unavailable = {
        _canonical_name(name, names) or _clean(name, limit=100): _clean(state, limit=700)
        for name, state in (inherited_unavailable or {}).items()
        if _clean(name, limit=100) and _clean(state, limit=700)
    }
    for index, row in enumerate(normalized):
        before = dict(unavailable)
        changes = _unique_strings(row.get("character_state_changes") or [], limit=30)
        row_text = " ".join([
            *(_clean(row.get(field), limit=900) for field in (
                "objective", "opening_state", "closing_state", "persistent_state",
            )),
            *changes,
        ])
        restoration_names = {
            name for name in list(unavailable)
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9]).{{0,100}}"
                r"\b(?:returns?|restored|resurrected|reappears?|reconstituted|revived)\b",
                row_text,
                flags=re.IGNORECASE,
            )
        }
        for name in restoration_names:
            unavailable.pop(name, None)
        cast = _unique_strings(row.get("cast_present") or [], limit=30)
        if not cast:
            cast = [
                name for name in names
                if re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
                    row_text,
                    flags=re.IGNORECASE,
                )
            ]
            if not cast and names:
                cast = names[:1]
        filtered_cast: list[str] = []
        for candidate in cast:
            canonical = _canonical_name(candidate, names) or candidate
            if any(_key(canonical) == _key(name) for name in unavailable):
                continue
            filtered_cast.append(canonical)
        row["cast_present"] = _unique_strings(filtered_cast, limit=30)
        row["character_state_changes"] = changes
        row["_character_availability_before"] = before
        row["inherited_character_state"] = (
            "; ".join(f"{name}: {state}" for name, state in before.items())
            or "No irreversible named-character state change is currently active"
        )
        for change in changes:
            folded = change.casefold()
            for name in names:
                if not re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
                    change,
                    flags=re.IGNORECASE,
                ):
                    continue
                if re.search(
                    r"\b(?:returns?|restored|resurrected|reappears?|reconstituted|revived)\b",
                    folded,
                ):
                    unavailable.pop(name, None)
                elif re.search(
                    r"\b(?:dead|dies|killed|turns?\s+to\s+dust|disintegrat(?:es|ed)|"
                    r"vanish(?:es|ed)|erased\s+from\s+existence)\b",
                    folded,
                ):
                    unavailable[name] = change
        row["_character_availability_after"] = dict(unavailable)
        if index and not row.get("inherited_state"):
            previous = normalized[index - 1]
            row["inherited_state"] = "; ".join(_unique_strings([
                previous.get("closing_state"), previous.get("persistent_state"),
            ], limit=2))
    return normalized


def apply_long_form_recurring_motifs(
    rows: Sequence[dict[str, Any]],
    *,
    story_bible: dict[str, Any],
) -> list[dict[str, Any]]:
    """Distribute each intentional recurring motif without duplicating footage."""

    normalized = [dict(row) for row in rows]
    if not normalized:
        return normalized
    valid_motifs = {
        str(item.get("motif_id") or "").upper(): item
        for item in story_bible.get("recurring_motifs") or []
        if isinstance(item, dict) and item.get("motif_id")
    }
    for row in normalized:
        row["recurring_motif_ids"] = [
            str(item or "").upper()
            for item in row.get("recurring_motif_ids") or []
            if str(item or "").upper() in valid_motifs
        ]
    for motif_id, motif in valid_motifs.items():
        proposed = [
            index for index, row in enumerate(normalized)
            if motif_id in row["recurring_motif_ids"]
        ]
        minimum = max(2, min(len(normalized), int(motif.get("min_occurrences") or 2)))
        maximum = max(minimum, min(len(normalized), int(motif.get("max_occurrences") or minimum)))
        selected = proposed[:maximum]
        if len(selected) < minimum:
            if minimum == 1:
                candidates = [len(normalized) // 2]
            else:
                candidates = [
                    int(round(position * (len(normalized) - 1) / (minimum - 1)))
                    for position in range(minimum)
                ]
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= minimum:
                    break
        selected = sorted(selected[:maximum])
        for index, row in enumerate(normalized):
            row["recurring_motif_ids"] = [
                item for item in row["recurring_motif_ids"] if item != motif_id
            ]
            if index not in selected:
                continue
            row["recurring_motif_ids"].append(motif_id)
            for field in ("dialogue_ids", "source_event_ids"):
                current = [str(item or "").upper() for item in row.get(field) or []]
                for item in motif.get(field) or []:
                    item = str(item or "").upper()
                    if item and item not in current:
                        current.append(item)
                row[field] = current
            variation = _clean(motif.get("variation_rule"), limit=500)
            if variation:
                row["motif_variation_contract"] = variation
    return normalized


def place_chapter_motifs_in_sequences(
    rows: Sequence[dict[str, Any]],
    *,
    chapter: dict[str, Any],
    story_bible: dict[str, Any],
) -> list[dict[str, Any]]:
    """Place each chapter-level motif in one bounded sequence."""

    normalized = [dict(row) for row in rows]
    motif_map = {
        str(item.get("motif_id") or "").upper(): item
        for item in story_bible.get("recurring_motifs") or []
        if isinstance(item, dict) and item.get("motif_id")
    }
    for row in normalized:
        row["recurring_motif_ids"] = []
    for raw_id in chapter.get("recurring_motif_ids") or []:
        motif_id = str(raw_id or "").upper()
        motif = motif_map.get(motif_id)
        if not motif or not normalized:
            continue
        obligations = {
            str(item or "").upper()
            for field in ("dialogue_ids", "source_event_ids")
            for item in motif.get(field) or []
        }
        owner = next((
            index for index, row in enumerate(normalized)
            if obligations.intersection({
                str(item or "").upper()
                for field in ("dialogue_ids", "source_event_ids")
                for item in row.get(field) or []
            })
        ), len(normalized) // 2)
        normalized[owner]["recurring_motif_ids"].append(motif_id)
        normalized[owner]["motif_variation_contract"] = _clean(
            chapter.get("motif_variation_contract")
            or motif.get("variation_rule"),
            limit=500,
        )
    return normalized


def format_long_form_story_bible(
    story_bible: dict[str, Any],
    *,
    cast_names: Optional[Sequence[str]] = None,
    location_ids: Optional[Sequence[str]] = None,
) -> str:
    """Render a compact production contract for bounded local-model calls."""

    requested_cast = {_key(item) for item in cast_names or [] if _key(item)}
    character_items = [
        item for item in story_bible.get("canonical_characters") or []
        if isinstance(item, dict) and _clean(item.get("name"), limit=100)
        and (
            cast_names is None
            or _key(item.get("name")) in requested_cast
        )
    ]
    characters = "; ".join(
        f"{_clean(item.get('name'), limit=100)} — "
        f"{_clean(item.get('role'), limit=180)}; initial: "
        f"{_clean(item.get('initial_state'), limit=220)}; continuity: "
        f"{_clean(item.get('continuity_rules'), limit=260)}"
        for item in character_items
    ) or "No named cast is locked for this bounded section."
    requested_locations = {
        _key(item) for item in location_ids or [] if _key(item)
    }
    location_items = [
        item for item in story_bible.get("canonical_locations") or []
        if isinstance(item, dict) and item.get("location_id")
        and (
            location_ids is None
            or _key(item.get("location_id")) in requested_locations
        )
    ]
    locations = "; ".join(
        f"{item.get('location_id')}: {_clean(item.get('name'), limit=140)} — "
        f"{_clean(item.get('visual_identity'), limit=240)}"
        for item in location_items
    ) or "Locations must remain faithful to the user concept."
    motifs = "; ".join(
        f"{item.get('motif_id')}: {_clean(item.get('description'), limit=300)}; "
        f"vary by {_clean(item.get('variation_rule'), limit=260)}"
        for item in story_bible.get("recurring_motifs") or []
        if isinstance(item, dict) and item.get("motif_id")
    ) or "No recurring motif is required."
    forbidden = "; ".join(
        _unique_strings(story_bible.get("forbidden_drift") or [], limit=12)
    )
    world_rules = "; ".join(
        _unique_strings(story_bible.get("world_rules") or [], limit=12)
    )
    return "\n".join([
        f"Premise engine: {_clean(story_bible.get('story_engine'), limit=700)}",
        f"Tone: {_clean(story_bible.get('tone_contract'), limit=700)}",
        f"Ending: {_clean(story_bible.get('ending_contract'), limit=700)}",
        f"Canonical cast: {characters}",
        f"Location registry: {locations}",
        f"Recurring motifs: {motifs}",
        f"World/state rules: {world_rules}",
        f"Forbidden drift: {forbidden}",
    ])


def _canonical_name(value: Any, names: Sequence[str]) -> str:
    candidate = _key(value)
    if not candidate:
        return ""
    exact = {_key(name): name for name in names}
    if candidate in exact:
        return exact[candidate]
    first_matches = [
        name for name in names
        if _key(name).split() and candidate.split()
        and _key(name).split()[0] == candidate.split()[0]
    ]
    if len(first_matches) == 1:
        return first_matches[0]
    best_name = ""
    best_score = 0.0
    for name in names:
        score = difflib.SequenceMatcher(None, candidate, _key(name)).ratio()
        if score > best_score:
            best_name, best_score = name, score
    return best_name if best_score >= 0.78 else ""


def _words(value: Any, limit: int) -> str:
    items = str(value or "").split()
    return " ".join(items[:max(0, int(limit))])


def _context_ir_section(prompt: str, label: str) -> str:
    labels = (
        "subject_definitions", "summary", "retention_analysis",
        "integrated_multimodal_description", "detailed_description",
        "overall_soundscape", "non_diegetic_music",
    )
    next_labels = "|".join(re.escape(item) for item in labels if item != label)
    match = re.search(
        rf"\b{re.escape(label)}\s*:\s*(.*?)(?=\b(?:{next_labels})\s*:|\Z)",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _clean(match.group(1), limit=6000) if match else ""


def compact_long_form_h3_prompt(shot: dict[str, Any]) -> tuple[str, bool]:
    """Recompile an overlong H3 prompt from its authoritative shot fields.

    Dialogue and multimodal reference mappings are never summarized.  The
    compactor removes duplicated screenplay/continuity prose and re-emits one
    concise, self-contained Context-IR description.
    """

    original = str(shot.get("video_prompt") or "").strip()
    if len(original.split()) <= 650 or not re.search(
        r"\b(?:integrated_multimodal_description|detailed_description)\s*:",
        original,
        flags=re.IGNORECASE,
    ):
        return original, False

    subject_definitions = _words(
        _context_ir_section(original, "subject_definitions"),
        180,
    )
    retention = _words(
        _context_ir_section(original, "retention_analysis"),
        90,
    )
    summary = _words(
        _context_ir_section(original, "summary")
        or shot.get("scene_goal"),
        55,
    )
    subjects = []
    subject_name_by_id: dict[str, str] = {}
    for subject in shot.get("subjects_on_screen") or []:
        if not isinstance(subject, dict):
            continue
        character_id = _clean(subject.get("character_id"), limit=80)
        name = _clean(subject.get("speaker_name") or character_id, limit=100)
        if character_id:
            subject_name_by_id[_key(character_id)] = name or character_id
        details = ", ".join(_unique_strings([
            name,
            _words(subject.get("visual_description"), 45),
            _words(subject.get("wardrobe"), 22),
            _words(subject.get("position_or_relation"), 28),
        ], limit=4))
        if details:
            subjects.append(details)

    camera = shot.get("camera_plan") or {}
    camera_text = ", ".join(_unique_strings([
        camera.get("framing") if isinstance(camera, dict) else "",
        camera.get("angle") if isinstance(camera, dict) else "",
        camera.get("movement") if isinstance(camera, dict) else "",
        camera.get("movement_intensity") if isinstance(camera, dict) else "",
        camera.get("lens_feel") if isinstance(camera, dict) else "",
        camera.get("reframing_notes") if isinstance(camera, dict) else "",
    ], limit=6))
    dialogue_parts: list[str] = []
    for beat in shot.get("dialogue_beats") or []:
        if not isinstance(beat, dict):
            continue
        spoken = _clean(beat.get("spoken_text"), limit=1200)
        if not spoken:
            continue
        speaker_id = _clean(beat.get("speaker_id"), limit=100)
        speaker = subject_name_by_id.get(_key(speaker_id), speaker_id or "Speaker")
        delivery = _words(beat.get("delivery"), 24) or "speaks naturally"
        cue = _words(beat.get("physical_cue"), 28)
        clause = f"{speaker} {delivery}"
        if cue:
            clause += f" while {cue}"
        clause += f": <d>[English] {spoken}</d>. Immediately afterward, {speaker} closes their mouth."
        dialogue_parts.append(clause)

    audio = shot.get("audio_plan") or {}
    soundscape = _words(
        _context_ir_section(original, "overall_soundscape")
        or (
            audio.get("ambience") if isinstance(audio, dict) else ""
        ),
        90,
    ) or "Natural synchronized ambience and effects appropriate to the visible action."
    effects = audio.get("effects") if isinstance(audio, dict) else []
    if effects:
        soundscape = _clean(
            soundscape + " Synchronized effects: " + "; ".join(
                _words(item, 22) for item in effects if _words(item, 22)
            ),
            limit=900,
        )
    music = _words(
        _context_ir_section(original, "non_diegetic_music"),
        45,
    ) or "N/A"
    detail_parts = [
        _words(shot.get("environment"), 75),
        _words(shot.get("visual_style"), 45),
        _words(shot.get("lighting"), 35),
        _words(shot.get("mood"), 25),
        (
            "Visible cast: " + "; ".join(subjects)
            if subjects else ""
        ),
        (
            "Opening blocking: " + _words(shot.get("spatial_setup"), 65)
            if shot.get("spatial_setup") else ""
        ),
        f"Camera: {_words(camera_text, 65)}" if camera_text else "",
        (
            "Chronological action: " + "; then ".join(
                _words(item, 55)
                for item in shot.get("action_beats") or []
                if _words(item, 55)
            )
        ),
        " ".join(dialogue_parts),
        (
            "No one else speaks. Outside the tagged lines, every mouth remains "
            "closed; no muttering, narration, repeated words, or gibberish."
            if dialogue_parts else
            "Everyone remains silent with mouths closed; no voices, muttering, narration, or gibberish."
        ),
        (
            "Ending state: " + _words(shot.get("ending_beat"), 65)
            if shot.get("ending_beat") else ""
        ),
        (
            "Closing blocking: " + _words(shot.get("closing_blocking"), 55)
            if shot.get("closing_blocking") else ""
        ),
    ]
    detailed = ". ".join(
        part.strip(" .") for part in detail_parts if str(part or "").strip(" .")
    )
    compiled_parts = []
    if subject_definitions:
        compiled_parts.append(f"subject_definitions: {subject_definitions}")
    if summary:
        compiled_parts.append(f"summary: {summary}")
    if retention:
        compiled_parts.append(f"retention_analysis: {retention}")
    compiled_parts.extend([
        f"integrated_multimodal_description: [Shot 1] {detailed}.",
        f"overall_soundscape: {soundscape}",
        f"non_diegetic_music: {music}",
    ])
    compiled = "\n\n".join(compiled_parts)
    # Reference mappings and exact dialogue can legitimately consume much of
    # the budget. If the first pass is still over, remove only redundant
    # summary/retention fields; never cut those protected contracts.
    if len(compiled.split()) > 650:
        compiled_parts = [
            item for item in compiled_parts
            if not item.startswith(("summary:", "retention_analysis:"))
        ]
        compiled = "\n\n".join(compiled_parts)
    return compiled.strip(), True


def sanitize_long_form_shot_dicts(
    shot_dicts: Sequence[dict[str, Any]],
    *,
    story_bible: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Repair phantom speakers and action-as-dialogue before rendering.

    This remains conservative: genuine supporting cast is allowed when the
    story bible explicitly permits expansion.  Only obvious sound/camera
    pseudo-speakers are removed without an unambiguous canonical match.
    """

    shots = [copy.deepcopy(item) for item in shot_dicts if isinstance(item, dict)]
    names = [
        _clean(item.get("name"), limit=100)
        for item in story_bible.get("canonical_characters") or []
        if isinstance(item, dict) and _clean(item.get("name"), limit=100)
    ]
    registry_complete = bool(story_bible.get("registry_complete"))
    generic_roles = {
        "announcer", "barista", "bystander", "cashier", "clerk", "customer",
        "driver", "guard", "host", "interviewer", "narrator", "patron",
        "reporter", "server", "staff", "waiter", "waitress",
    }
    warnings: list[str] = []
    for shot_index, shot in enumerate(shots, start=1):
        subjects = [
            item for item in shot.get("subjects_on_screen") or []
            if isinstance(item, dict)
        ]
        subject_by_id = {
            _key(item.get("character_id")): item for item in subjects
            if _key(item.get("character_id"))
        }
        canonical_replacements: list[tuple[str, str]] = []
        for subject in subjects:
            name = _clean(
                subject.get("speaker_name") or subject.get("character_id"),
                limit=100,
            )
            canonical = _canonical_name(name, names)
            if canonical:
                if name and _key(name) != _key(canonical):
                    canonical_replacements.append((name, canonical))
                subject["speaker_name"] = canonical

        unsupported_subjects: list[tuple[str, str]] = []
        if registry_complete and names:
            retained_subjects: list[dict[str, Any]] = []
            for subject in subjects:
                subject_id = _clean(subject.get("character_id"), limit=100)
                subject_name = _clean(
                    subject.get("speaker_name") or subject_id,
                    limit=100,
                )
                folded = _key(subject_name)
                is_generic = (
                    folded in generic_roles
                    or folded.startswith(("a ", "an ", "the "))
                    or not re.search(r"[a-z]", folded)
                )
                if _canonical_name(subject_name, names) or is_generic:
                    retained_subjects.append(subject)
                else:
                    unsupported_subjects.append((subject_id, subject_name))
            subjects = retained_subjects

        kept: list[dict[str, Any]] = []
        removed_beats: list[dict[str, Any]] = []
        for raw in shot.get("dialogue_beats") or []:
            if not isinstance(raw, dict):
                continue
            beat = dict(raw)
            spoken = _clean(beat.get("spoken_text"), limit=1200)
            speaker_id = _clean(beat.get("speaker_id"), limit=100)
            subject = subject_by_id.get(_key(speaker_id))
            speaker_name = _clean(
                (subject or {}).get("speaker_name") or speaker_id,
                limit=100,
            )
            canonical = _canonical_name(speaker_name, names)
            pseudo = _key(speaker_name) in _PSEUDO_SPEAKERS or bool(re.fullmatch(
                r"(?:sfx|fx|sound|camera|music|action)[ _-]*\d*",
                _key(speaker_name),
            ))
            if not spoken or pseudo:
                if pseudo:
                    warnings.append(
                        f"shot {shot_index}: removed non-character dialogue speaker {speaker_name!r}"
                    )
                removed_beats.append(beat)
                continue
            if canonical and subject is not None:
                subject["speaker_name"] = canonical
            elif canonical:
                beat["speaker_id"] = canonical
            elif (
                registry_complete
                and names
                and _key(speaker_name) not in generic_roles
            ):
                warnings.append(
                    f"shot {shot_index}: removed unsupported speaker {speaker_name!r}"
                )
                removed_beats.append(beat)
                continue
            beat["spoken_text"] = spoken
            kept.append(beat)
        for field in ("video_prompt",):
            prompt = str(shot.get(field) or "")
            if not prompt:
                continue
            for removed in removed_beats:
                spoken = _clean(removed.get("spoken_text"), limit=1200)
                if not spoken:
                    continue
                tagged = re.compile(
                    r"[^.!?<>]{0,260}<d>\s*(?:\[[^\]]+\]\s*)?"
                    + re.escape(spoken)
                    + r"\s*</d>[^.!?<>]{0,180}[.!?]?",
                    flags=re.IGNORECASE,
                )
                prompt = tagged.sub(" ", prompt)
            for subject_id, subject_name in unsupported_subjects:
                for alias in (subject_name, subject_id):
                    if alias:
                        prompt = re.sub(
                            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                            "a silent background extra",
                            prompt,
                            flags=re.IGNORECASE,
                        )
            for source_name, canonical_name in canonical_replacements:
                prompt = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(source_name)}(?![A-Za-z0-9])",
                    canonical_name,
                    prompt,
                    flags=re.IGNORECASE,
                )
            shot[field] = re.sub(r"\s+", " ", prompt).strip()
        cleaned_windows: list[Any] = []
        for prompt in shot.get("window_prompts") or []:
            if not isinstance(prompt, str):
                cleaned_windows.append(prompt)
                continue
            for removed in removed_beats:
                spoken = _clean(removed.get("spoken_text"), limit=1200)
                if spoken:
                    prompt = re.sub(
                        r"[^.!?<>]{0,260}<d>\s*(?:\[[^\]]+\]\s*)?"
                        + re.escape(spoken)
                        + r"\s*</d>[^.!?<>]{0,180}[.!?]?",
                        " ",
                        prompt,
                        flags=re.IGNORECASE,
                    )
            for subject_id, subject_name in unsupported_subjects:
                for alias in (subject_name, subject_id):
                    if alias:
                        prompt = re.sub(
                            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                            "a silent background extra",
                            prompt,
                            flags=re.IGNORECASE,
                        )
            for source_name, canonical_name in canonical_replacements:
                prompt = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(source_name)}(?![A-Za-z0-9])",
                    canonical_name,
                    prompt,
                    flags=re.IGNORECASE,
                )
            cleaned_windows.append(re.sub(r"\s+", " ", prompt).strip())
        if "window_prompts" in shot:
            shot["window_prompts"] = cleaned_windows
        if unsupported_subjects:
            warnings.extend(
                f"shot {shot_index}: removed unsupported cast member {name!r}"
                for _subject_id, name in unsupported_subjects
            )
            for field in (
                "spatial_setup", "environment", "action_beats", "ending_beat",
                "closing_blocking", "scene_goal", "persistent_story_state",
            ):
                value = shot.get(field)
                if not isinstance(value, (str, list)):
                    continue
                def neutralize(text: Any) -> str:
                    result = str(text or "")
                    for subject_id, subject_name in unsupported_subjects:
                        for alias in (subject_name, subject_id):
                            if alias:
                                result = re.sub(
                                    rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                                    "a silent background extra",
                                    result,
                                    flags=re.IGNORECASE,
                                )
                    for source_name, canonical_name in canonical_replacements:
                        result = re.sub(
                            rf"(?<![A-Za-z0-9]){re.escape(source_name)}(?![A-Za-z0-9])",
                            canonical_name,
                            result,
                            flags=re.IGNORECASE,
                        )
                    return re.sub(r"\s+", " ", result).strip()
                shot[field] = (
                    [neutralize(item) for item in value]
                    if isinstance(value, list) else neutralize(value)
                )
        shot["subjects_on_screen"] = subjects
        shot["dialogue_beats"] = kept
        compacted_prompt, compacted = compact_long_form_h3_prompt(shot)
        if compacted:
            shot["video_prompt"] = compacted_prompt
            warnings.append(
                f"shot {shot_index}: compacted an overlong H3 prompt to "
                f"{len(compacted_prompt.split())} words"
            )
    return shots, warnings


def audit_long_form_plan(
    shot_dicts: Sequence[dict[str, Any]],
    *,
    story_bible: dict[str, Any],
    target_duration: float,
) -> dict[str, Any]:
    """Return a compact durable quality report without causing a late failure."""

    prompt_words = [
        len(str(prompt or "").split())
        for item in shot_dicts if isinstance(item, dict)
        for prompt in [
            item.get("video_prompt"),
            *(item.get("window_prompts") or []),
        ]
        if str(prompt or "").strip()
    ]
    dialogue = [
        beat
        for item in shot_dicts if isinstance(item, dict)
        for beat in item.get("dialogue_beats") or []
        if isinstance(beat, dict) and _clean(beat.get("spoken_text"))
    ]
    duration = sum(
        float(item.get("duration_sec") or 0.0)
        for item in shot_dicts if isinstance(item, dict)
    )
    goals = [
        _key(item.get("scene_goal"))
        for item in shot_dicts if isinstance(item, dict)
        if _key(item.get("scene_goal"))
    ]
    duplicate_goals = len(goals) - len(set(goals))
    location_ids = {
        str((item.get("metadata") or {}).get("long_form_location_id") or "").strip()
        for item in shot_dicts if isinstance(item, dict)
        if str((item.get("metadata") or {}).get("long_form_location_id") or "").strip()
    }
    motif_coverage: dict[str, int] = {}
    for motif in story_bible.get("recurring_motifs") or []:
        if not isinstance(motif, dict) or not motif.get("motif_id"):
            continue
        motif_id = str(motif["motif_id"]).upper()
        chapters = {
            (item.get("metadata") or {}).get("long_form_chapter")
            for item in shot_dicts if isinstance(item, dict)
            if motif_id in {
                str(value or "").upper()
                for value in (item.get("metadata") or {}).get(
                    "long_form_recurring_motif_ids", []
                )
            }
        }
        motif_coverage[motif_id] = len(chapters)
    issues: list[str] = []
    issues.extend(
        long_form_story_bible_quality_issues(
            story_bible,
            chapter_count=max(
                1,
                len({
                    (item.get("metadata") or {}).get("long_form_chapter")
                    for item in shot_dicts if isinstance(item, dict)
                    if (item.get("metadata") or {}).get("long_form_chapter")
                }),
            ),
        )
    )
    if abs(duration - float(target_duration)) > max(2.0, float(target_duration) * 0.03):
        issues.append("planned duration differs materially from the requested runtime")
    if duplicate_goals:
        issues.append(f"{duplicate_goals} scene goals are exact duplicates")
    if story_bible.get("allow_location_expansion"):
        expected_locations = min(
            len(story_bible.get("canonical_locations") or []),
            max(
                1,
                len({
                    (item.get("metadata") or {}).get("long_form_chapter")
                    for item in shot_dicts if isinstance(item, dict)
                    if (item.get("metadata") or {}).get("long_form_chapter")
                }),
            ),
        )
        expected_locations = min(
            expected_locations,
            max(3, int(round(expected_locations * 0.60))),
        )
        if expected_locations and len(location_ids) < expected_locations:
            issues.append(
                f"only {len(location_ids)} locations reached the final shot plan; "
                f"at least {expected_locations} were expected"
            )
    state_change_keys = {
        (
            (item.get("metadata") or {}).get("long_form_chapter"),
            (item.get("metadata") or {}).get("long_form_sequence"),
            _key(change),
        )
        for item in shot_dicts if isinstance(item, dict)
        for change in (item.get("metadata") or {}).get(
            "long_form_character_state_changes",
            [],
        )
        if _key(change)
    }
    state_change_count = len(state_change_keys)
    for motif in story_bible.get("recurring_motifs") or []:
        if not isinstance(motif, dict) or not motif.get("motif_id"):
            continue
        motif_id = str(motif["motif_id"]).upper()
        minimum = int(motif.get("min_occurrences") or 0)
        if motif_coverage.get(motif_id, 0) < minimum:
            issues.append(
                f"{motif_id} appears in {motif_coverage.get(motif_id, 0)} "
                f"chapters but requires at least {minimum}"
            )
    return {
        "revision": LONG_FORM_STORY_BIBLE_REVISION,
        "target_duration_sec": float(target_duration),
        "planned_duration_sec": round(duration, 3),
        "shot_count": len(shot_dicts),
        "dialogue_turn_count": len(dialogue),
        "canonical_character_count": len(story_bible.get("canonical_characters") or []),
        "canonical_location_count": len(story_bible.get("canonical_locations") or []),
        "recurring_motif_count": len(story_bible.get("recurring_motifs") or []),
        "represented_location_count": len(location_ids),
        "recurring_motif_chapter_coverage": motif_coverage,
        "named_character_state_change_count": state_change_count,
        "duplicate_scene_goal_count": duplicate_goals,
        "maximum_prompt_words": max(prompt_words, default=0),
        "prompts_over_650_words": sum(1 for count in prompt_words if count > 650),
        "issues": issues,
    }
