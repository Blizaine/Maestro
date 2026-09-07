"""Dependency-light character/speaker matching shared by H3 planning and rendering."""
from __future__ import annotations

import re
from typing import Any

_DIALOGUE_TAG_RE = re.compile(r"<d(?:\s+[^>]*)?>(.*?)</d>", re.IGNORECASE | re.DOTALL)
_SPEECH_VERB_RE = re.compile(
    r"\b(?:say|says|said|ask|asks|asked|reply|replies|replied|respond|responds|"
    r"responded|answer|answers|answered|speak|speaks|spoke|shout|shouts|shouted|"
    r"yell|yells|yelled|whisper|whispers|whispered|exclaim|exclaims|exclaimed|"
    r"murmur|murmurs|murmured|call|calls|called|cry|cries|cried|add|adds|added|"
    r"remark|remarks|remarked|state|states|stated|declare|declares|declared|"
    r"warn|warns|warned|demand|demands|demanded|tell|tells|told)\b",
    re.IGNORECASE,
)
_DIALOGUE_OWNER_NAME_RE = re.compile(
    r"(?<!\w)([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})(?!\w)"
)
_DIALOGUE_OWNER_LEADING_WORDS = {
    "a", "after", "an", "and", "as", "at", "both", "during", "from",
    "he", "her", "his", "immediately", "in", "inside", "it", "later",
    "next", "only", "outside", "she", "the", "their", "then", "they",
    "this", "while", "with",
}

def _normalize_ref2va_speaker_alias(value: Any) -> str:
    alias = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    alias = re.sub(
        r"\s+(?:voice(?:\s+reference)?|audio(?:\s+reference)?|character\s+reference)\s*$",
        "",
        alias,
    )
    return alias.strip(" \t\r\n.,:;_-–—")


def _ref2va_alias_values(item: dict, fallback_role: str = "") -> list[str]:
    """Match RefMod packaging names and natural spelling without fuzzy guesses.

    Retain the full original name as an alias, then remove recognized file
    decoration and a trailing version. Compact keys make "elizadushku" and
    "Eliza Dushku" the same candidate. Callers discard keys shared by multiple
    characters, so two versions still need a version-specific name or Subject.
    """
    values: list[str] = []
    for raw in (item.get("character_name"), item.get("role"), fallback_role):
        alias = _normalize_ref2va_speaker_alias(raw)
        candidates = [alias]
        filename = re.split(r"[\\/]", alias)[-1]
        cleaned = re.sub(r"^minimaxh3_", "", filename, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:\.maestro)?\.safetensors$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:^|[_.\s-]+)refmod$", "", cleaned, flags=re.IGNORECASE)
        if cleaned != filename:
            candidates.append(cleaned)
            candidates.append(re.sub(r"[_.\s-]+v\d+(?:[_.]\d+)*$", "", cleaned, flags=re.IGNORECASE))
        for candidate in candidates:
            key = re.sub(r"[\s_-]+", "", candidate)
            if key and key not in values:
                values.append(key)
    return values


def _ref2va_alias_occurrences(source: str, aliases: dict[str, int]):
    occurrences: list[tuple[int, int, str, int]] = []
    for alias, subject in aliases.items():
        alias_pattern = r"[\s_-]*".join(re.escape(char) for char in alias)
        for match in re.finditer(
            rf"(?<!\w){alias_pattern}(?!\w)",
            source,
            re.IGNORECASE,
        ):
            occurrences.append((match.start(), match.end(), alias, subject))
    return sorted(occurrences, key=lambda row: (row[0], -(row[1] - row[0])))


def _clean_ref2va_dialogue_owner_name(value: str) -> str:
    """Return a human speaker label without sentence-leading glue words."""

    words = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n,;:-")
    parts = words.split()
    while len(parts) > 1 and parts[0].casefold() in _DIALOGUE_OWNER_LEADING_WORDS:
        parts.pop(0)
    candidate = " ".join(parts).strip()
    if not candidate or candidate.casefold() in _DIALOGUE_OWNER_LEADING_WORDS:
        return ""
    return candidate


def _resolve_ref2va_dialogue_owner_name(
    source: str,
    dialogue_start: int,
    dialogue_end: int,
) -> str:
    """Resolve an explicitly named speaker, including non-reference actors.

    Ref2VA's ``(Sx)`` marker identifies first-vocal-event order. It does not
    identify ``<Subject x>``.  Detecting the grammatical speaker separately is
    what lets a saved Blaine character share a scene with unreferenced Rachel
    and Ross without either guest inheriting Blaine's portrait or voice.
    """

    before = source[max(0, dialogue_start - 420):dialogue_start]
    after = source[dialogue_end:dialogue_end + 180]
    clause_start = max(
        before.rfind("."),
        before.rfind("!"),
        before.rfind("?"),
        before.rfind(";"),
        before.rfind("\n"),
    ) + 1
    clause = before[clause_start:]

    def occurrences(value: str):
        return [
            (match.start(), match.end(), _clean_ref2va_dialogue_owner_name(match.group(1)))
            for match in _DIALOGUE_OWNER_NAME_RE.finditer(value)
            if _clean_ref2va_dialogue_owner_name(match.group(1))
        ]

    # Direct screenplay syntax: ``Rachel: <d>...</d>``.
    for start, end, name in reversed(occurrences(clause)):
        tail = clause[end:]
        leading = clause[:start]
        if (
            re.fullmatch(r"\s*(?:'s\s+voice\s*)?[:,\-–—]\s*", tail, re.IGNORECASE)
            and not _SPEECH_VERB_RE.search(leading)
        ):
            return name

    # Postposed attribution: ``<d>...</d>, replies Rachel``.
    after_verb = re.match(
        rf"\s*[,;:\-–—]?\s*({_SPEECH_VERB_RE.pattern})",
        after,
        re.IGNORECASE,
    )
    if after_verb:
        candidates = [
            (start - after_verb.end(), -(end - start), name)
            for start, end, name in occurrences(after)
            if start >= after_verb.end()
        ]
        if candidates:
            return min(candidates)[-1]

    # Natural prose: ``Blaine turns to Yoda and says, ...``. The last name
    # before the speech verb is not necessarily the speaker, so reject names
    # introduced by object prepositions such as ``to`` or ``at``.
    verbs = list(_SPEECH_VERB_RE.finditer(clause))
    if verbs:
        verb = verbs[-1]
        candidates = []
        for start, end, name in occurrences(clause):
            if end > verb.start():
                continue
            leading = clause[max(0, start - 32):start]
            is_object = bool(re.search(
                r"(?:\bto|\bat|\btoward|\btowards|\bwith|\bbeside|\bnear|\bbehind)\s+$",
                leading,
                re.IGNORECASE,
            ))
            candidates.append((is_object, verb.start() - end, -(end - start), name))
        non_objects = [candidate for candidate in candidates if not candidate[0]]
        if non_objects:
            return min(non_objects)[-1]
        if candidates:
            return min(candidates)[-1]

    # Finished Context-IR often writes ``Rachel (S1) ...`` beside the tag
    # without repeating a speech verb. This marker can confirm the nearby
    # name, but it still never maps the name to Subject 1.
    marked = list(re.finditer(
        r"([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})"
        r"\s*\(S\d+\)",
        clause,
    ))
    if marked:
        return _clean_ref2va_dialogue_owner_name(marked[-1].group(1))
    return ""


def _resolve_ref2va_dialogue_speaker(
    source: str,
    dialogue_start: int,
    dialogue_end: int,
    speaker_aliases: dict[str, int],
    valid_subjects: set[int],
) -> int | None:
    """Resolve a dialogue speaker from natural-language cues around one line."""

    before_start = max(0, dialogue_start - 360)
    before = source[before_start:dialogue_start]
    after = source[dialogue_end:dialogue_end + 180]
    clause_start = max(
        before.rfind("."),
        before.rfind("!"),
        before.rfind("?"),
        before.rfind(";"),
        before.rfind("\n"),
    ) + 1
    clause = before[clause_start:]
    explicit_subjects = re.findall(
        r"<Subject\s+(\d+)>", clause, flags=re.IGNORECASE
    )
    if explicit_subjects:
        explicit_subject = int(explicit_subjects[-1])
        if explicit_subject in valid_subjects:
            return explicit_subject
    occurrences = _ref2va_alias_occurrences(clause, speaker_aliases)

    # Direct screenplay syntax: ``Yoda: "..."`` or ``Yoda's voice: "..."``.
    # Do not mistake the addressed character in natural prose such as
    # ``Thanos says to Yoda, <d>...</d>`` for a screenplay speaker label.  A
    # preceding speech verb means the grammatical-speaker pass below owns the
    # decision, where ``to Yoda`` is correctly treated as the object.
    for start, end, _alias, subject in reversed(occurrences):
        tail = clause[end:]
        leading = clause[:start]
        if (
            re.fullmatch(r"\s*(?:'s\s+voice\s*)?[:,\-–—]\s*", tail, re.IGNORECASE)
            and not _SPEECH_VERB_RE.search(leading)
        ):
            return subject

    # Postposed attribution is highly specific and takes precedence over an
    # unrelated speech verb in the preceding sentence.
    after_verb = re.match(
        rf"\s*[,;:\-–—]?\s*({_SPEECH_VERB_RE.pattern})",
        after,
        re.IGNORECASE,
    )
    if after_verb:
        after_occurrences = _ref2va_alias_occurrences(after, speaker_aliases)
        candidates = [
            (start - after_verb.end(), -len(alias), subject)
            for start, _end, alias, subject in after_occurrences
            if start >= after_verb.end()
        ]
        if candidates:
            return min(candidates)[-1]

    # Natural syntax before the line: ``Blaine turns to Yoda and says, ...``.
    # Select the last non-object character before the final speech verb.
    verbs = list(_SPEECH_VERB_RE.finditer(clause))
    if verbs:
        verb = verbs[-1]
        candidates = []
        for start, end, alias, subject in occurrences:
            if end > verb.start():
                continue
            leading = clause[max(0, start - 28):start]
            is_object = bool(re.search(
                r"(?:\bto|\bat|\btoward|\btowards|\bwith|\bbeside|\bnear|\bbehind)\s+$",
                leading,
                re.IGNORECASE,
            ))
            candidates.append((is_object, verb.start() - end, -len(alias), subject))
        non_objects = [candidate for candidate in candidates if not candidate[0]]
        if non_objects:
            return min(non_objects)[-1]
        if candidates:
            return min(candidates)[-1]

    # A manually authored Context-IR prompt may put the <d> tag in the sentence
    # after the named performance cue, for example: ``Yoda nods. He answers.
    # <d>...</d>``. Follow that short discourse chain, but only when the latest
    # named sentence has one unambiguous grammatical subject. This is purposely
    # conservative: ``Yoda and Blaine react. They answer.`` remains ambiguous.
    preceding_dialogue_end = 0
    for previous_match in _DIALOGUE_TAG_RE.finditer(source, 0, dialogue_start):
        preceding_dialogue_end = previous_match.end()
    discourse_start = max(preceding_dialogue_end, dialogue_start - 720)
    # Context-IR fields are independent contracts. Never reach backward from
    # detailed_description into subject_definitions/retention_analysis and use
    # a saved character named there as the grammatical speaker of a guest's
    # line. The latest field header is the hard discourse boundary.
    field_headers = list(re.finditer(
        r"(?im)^\s*(?:subject_definitions|summary|retention_analysis|"
        r"detailed_description|overall_soundscape|non_diegetic_music)\s*:\s*",
        source[discourse_start:dialogue_start],
    ))
    if field_headers:
        discourse_start += field_headers[-1].end()
    discourse = source[discourse_start:dialogue_start]
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?;])\s+|[\r\n]+", discourse)
        if segment.strip()
    ]
    for segment in reversed(segments):
        segment_occurrences = _ref2va_alias_occurrences(segment, speaker_aliases)
        if not segment_occurrences:
            continue
        candidate_subjects: set[int] = set()
        for start, _end, _alias, candidate_subject in segment_occurrences:
            leading = segment[max(0, start - 32):start]
            is_object = bool(re.search(
                r"(?:\bto|\bat|\btoward|\btowards|\bwith|\bbeside|\bnear|"
                r"\bbehind|\bfrom|\bfor|\bof|\bby)\s+$",
                leading,
                re.IGNORECASE,
            ))
            if not is_object:
                candidate_subjects.add(candidate_subject)
        if len(candidate_subjects) == 1:
            return next(iter(candidate_subjects))
        # Do not reach past a more recent sentence that names multiple possible
        # speakers. Guessing here would recreate the original voice-swap bug.
        return None
    return None


def _ambiguous_ref2va_dialogue_error(words: str) -> ValueError:
    excerpt = re.sub(r"\s+", " ", words).strip()[:80]
    return ValueError(
        "MiniMax H3 Omni could not determine which referenced character speaks "
        f"{excerpt!r}. Name the speaker beside the line (for example, Yoda says, "
        '"Do or do not.") or place that character\'s explicit <Subject N> tag '
        "beside the line. (Sx) labels only identify vocal-event order."
    )
