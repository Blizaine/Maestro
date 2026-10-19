"""Dependency-light character/speaker matching shared by H3 planning and rendering."""
from __future__ import annotations

import re
from typing import Any

_DIALOGUE_TAG_RE = re.compile(r"<d(?:\s+[^>]*)?>(.*?)</d>", re.IGNORECASE | re.DOTALL)
_SPEECH_VERB_RE = re.compile(
    r"\b(?:say|says|said|ask|asks|asked|reply|replies|replied|respond|responds|"
    r"responded|responding|answer|answers|answered|speak|speaks|spoke|shout|shouts|shouted|"
    r"yell|yells|yelled|whisper|whispers|whispered|exclaim|exclaims|exclaimed|"
    r"murmur|murmurs|murmured|call|calls|called|cry|cries|cried|add|adds|added|"
    r"remark|remarks|remarked|state|states|stated|declare|declares|declared|"
    r"warn|warns|warned|demand|demands|demanded|tell|tells|told|telling|"
    r"saying|speaking|asking|replying|explains?|explained|explaining|"
    r"informs?|informed|announces?|announced|calls?\s+out|called\s+out|"
    r"mumbl(?:es?|ed|ing)|murmuring|mutter(?:s|ed|ing)?|muffl(?:es?|ed|ing)|"
    r"sings?|singing|sang|raps?|rapping|narrat(?:es?|ed|ing))\b",
    re.IGNORECASE,
)
_POSSESSIVE_SPEECH_LABEL_RE = re.compile(
    r"['’]s\s+(?:voice|reply|whisper|answer|response)\s*[:,\-–—]\s*$",
    re.IGNORECASE,
)
_POSSESSIVE_BODY_SPEECH_RE = re.compile(
    rf"['’]s\s+(?:mouth\s+opens?|lips?\s+move)\s+(?:as|while)\s+"
    rf"(?:he|she|they)\s+(?:\w+ly\s+)?{_SPEECH_VERB_RE.pattern}\s*[:\-–—]\s*$",
    re.IGNORECASE,
)
_POSSESSIVE_EYES_ACTION_RE = re.compile(
    r"['’]s\s+eyes\s+(?:narrow|widen)(?:s|ed)?"
    r"(?:\s+(?:suddenly|slightly|briefly|wide|wider|narrower))*",
    re.IGNORECASE,
)
_DIALOGUE_OWNER_NAME_RE = re.compile(
    r"(?<!\w)([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})(?!\w)"
)
_DIALOGUE_OWNER_LEADING_WORDS = {
    "a", "after", "an", "and", "as", "at", "both", "during", "from",
    "he", "her", "his", "immediately", "in", "inside", "it", "later",
    "next", "now", "only", "outside", "she", "the", "their", "then", "they",
    "this", "while", "with",
}

# Shared by screenplay extraction and quoted-speech detection. A production
# field must not become a character just because its value follows a colon.
_PRODUCTION_LABELS = {
    "action", "actions", "ambiance", "ambience", "atmosphere", "audio",
    "audio design", "audio notes", "camera", "camera movement", "camera notes",
    "cast", "character", "characters", "cinematography", "color palette",
    "composition", "constraints", "continuity", "description", "dialogue",
    "director", "duration", "editing", "effects", "detailed description",
    "end", "example", "examples", "ext", "exterior", "fade in", "fade out", "format", "fps",
    "framing", "int", "interior", "language", "lighting", "location",
    "integrated multimodal description", "music", "non diegetic music",
    "negative prompt", "notes", "overall soundscape", "pacing", "pov",
    "instead", "prohibited", "forbidden", "disallowed",
    "instruction", "instructions", "prompt", "reference", "resolution", "retention analysis", "role",
    "prop", "props", "hero object", "hero prop", "set", "vehicle", "vehicles",
    "scene", "setting", "sfx", "shot", "sound", "sound design",
    "sound effects", "soundscape", "soundtrack", "style", "subject",
    "subject definitions", "summary", "template", "templates", "time", "timeline", "title", "tone", "transition",
    "vfx", "visual", "visual direction", "visual style", "visuals", "voice",
}
_PRODUCTION_LABEL_WORDS = {
    word for label in _PRODUCTION_LABELS for word in label.split()
} | {
    "and", "animation", "appearance", "background", "behavior", "blood", "body", "boosters",
    "characterization", "choreography", "cinematic", "closing", "clothing",
    "colorless", "colour", "core", "delivery", "density", "design", "directions", "environment",
    "facial", "film", "final", "foreground", "global", "guidance", "identity", "image",
    "initial", "injuries", "instructions", "lock", "locked", "logo", "logotype", "main", "motion", "movement",
    "opening", "outfit", "performance", "physical", "plan", "product", "production",
    "quality", "reference", "references", "requirements", "rules", "settings", "setup",
    "state", "structure", "system", "technique", "texture", "timing", "treatment",
    "wardrobe",
}


def is_h3_production_label(value: Any) -> bool:
    """Recognize production fields, including compound headings in briefs.

    Match combinations of production vocabulary rather than maintaining a
    separate exception for every heading ("Scene description", "Final state",
    "Visual requirements", etc.). Unknown names and character/role labels
    remain eligible to introduce real screenplay turns.
    """
    key = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    words = key.split()
    return bool(
        key in _PRODUCTION_LABELS
        or re.fullmatch(r"(?:scene|shot|subject|picture|video|audio|s)\s*\d+", key)
        or (len(words) > 1 and all(word in _PRODUCTION_LABEL_WORDS for word in words))
    )


class H3SpeakerBindingError(ValueError):
    """A real dialogue line lacks an unambiguous referenced speaker."""


def h3_action_beat_speaker(source: str, dialogue_start: int) -> str:
    """Recognize prose dialogue attributed by an adjacent named reaction.

    ``Alex stares at Sam. "Keep it."`` belongs to Alex, not the last name
    mentioned or the preceding turn. Stay within the same paragraph and one
    complete reaction sentence; headings, thoughts, and plural actors do not
    establish a speaker.
    """
    before = source[max(0, dialogue_start - 420):dialogue_start]
    before = re.split(r'["“”]|</d>|\n\s*\n', before, flags=re.I)[-1]
    reaction = re.search(
        r"(?:^|[.!?]\s+)([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})"
        r"\s+(?:(?:then|[\w-]+ly)\s+)?"
        r"(?i:stares?|looks?|glances?|glares?|nods?|shrugs?|smiles?|grins?|"
        r"frowns?|blinks?|sighs?|pauses?|hesitates?|turns?|leans?|squints?|"
        r"shakes?\s+(?:his|her|their)\s+head)\b"
        r"[^.!?;\r\n\"“”]{0,180}[.!?][ \t\r\n]*$", before,
    )
    if not reaction:
        return ""
    name = _clean_ref2va_dialogue_owner_name(reaction.group(1))
    if not name or is_h3_production_label(name):
        return ""
    # A second independent actor makes a reaction-only attribution ambiguous.
    if re.search(r"\b(?:and|while|but|as)\s+[A-Z][\w'’-]*\b", reaction.group(0)):
        return ""
    return name


def is_h3_spoken_quote(source: str, match: re.Match, *, allow_screenplay_label: bool = True) -> bool:
    """Recognize speech cues, without turning quoted names/styles into dialogue.

    Explicit <d> blocks are handled separately by callers. Quotes in reference
    definitions, music directions, and visual descriptions are not a request
    for another character to say those words.
    """

    start, end = match.span()
    if any(tag.start() <= start < tag.end() for tag in _DIALOGUE_TAG_RE.finditer(source)):
        return False
    fields = list(re.finditer(
        r"(?mi)^[ \t]*(subject_definitions|summary|retention_analysis|"
        r"detailed_description|integrated_multimodal_description|"
        r"overall_soundscape|non_diegetic_music)\s*:", source[:start],
    ))
    if fields and fields[-1].group(1).casefold() not in {
        "detailed_description", "integrated_multimodal_description",
    }:
        return False

    before = source[max(0, start - 220):start]
    # A previous line's speech cue must not capture the following quotation.
    before = re.split(r'["“”]|</d>', before, flags=re.IGNORECASE)[-1]
    after = source[end:end + 120]
    if re.search(r"(?i)\b(?:titled|entitled|called|named|captioned)\s*[:,-]?\s*$", before):
        return False
    written_text = re.search(
        r"(?i)\b(?:sign|banner|label|subtitle|caption|marquee|poster|billboard|"
        r"screen|monitor|display|neon|placard|headline|logo|shirt|door|wall|"
        r"on-screen\s+text)\b[^.!?\r\n]{0,80}"
        r"\b(?:reads?|reading|shows?|showing|displays?|displaying|bears?|bearing|"
        r"marked|printed|written|spells?|saying|says?|said|"
        r"with(?:\s+the)?\s+(?:text|words?|lettering))\s*[:,-]?\s*$", before,
    )
    if written_text and not re.search(
        r"(?i)\b(?:and|then|while|before|after|as)\b", written_text.group(0),
    ):
        # "Alice opens the door and says ..." is human dialogue, not text
        # printed on the door. A coordinated action breaks object ownership.
        return False
    if re.match(
        r"(?i)^\s*(?:appears?|is\s+(?:visible|written|printed|displayed)|glows?)"
        r"\b[^.!?\r\n]{0,70}\b(?:on|across|above|below|behind|over)\b", after,
    ):
        return False

    label = re.search(
        r"(?:^|\n)[ \t]*(?:[-*][ \t]+)?(?:\*\*)?"
        r"([\w '-]{1,80}?)(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?[ \t]*$",
        before,
    )
    # Check headings before speech verbs: "Final state" contains "state",
    # but a quoted visual description is still not a spoken declaration.
    if label and is_h3_production_label(label.group(1)) and (
        label.group(1).strip().casefold() != "dialogue"
    ):
        return False

    clause = re.split(r"[.!?;\n]", before)[-1]
    if re.search(r"(?i)\b(?:a|an|the|character|role|style)\s*$", clause):
        return False
    if _SPEECH_VERB_RE.search(clause) or re.search(
        r"(?i)\b(?:dialogue|lyrics?|voiceover)\s*:\s*$", clause,
    ):
        return True
    # H3-native ownership and screenplay labels also declare spoken content.
    if re.search(r"(?:<Subject\s+\d+>|\(S\d+\))\s*[:,]?\s*$", before, re.IGNORECASE):
        return True
    if allow_screenplay_label and label and not is_h3_production_label(label.group(1)):
        return True
    # Quotation followed by attribution: "Hello," Alex says.
    attribution = re.split(r'[.!?;"“”\n]', after.lstrip(" ,"))[0]
    post_verb = _SPEECH_VERB_RE.search(attribution)
    if post_verb and re.match(r"^[,\s]+", after) and (
        re.fullmatch(r"\s*(?:[A-Z][\w'’-]*|he|she|they)(?:\s+[\w'’-]+){0,4}\s+",
                     attribution[:post_verb.start()])
        or re.match(r"\s+(?:[A-Z][\w'’-]*|he|she|they)\b", attribution[post_verb.end():])
    ):
        return True
    if allow_screenplay_label and h3_action_beat_speaker(source, start):
        return True
    return not (source[:start] + source[end:]).strip(" \t\r\n.,;:!?-")


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


def _has_unintroduced_possessive_cue_name(value: str) -> bool:
    """Return whether a proper name appears as a new subject in a cue."""
    object_preposition = re.compile(
        r"(?:\bto|\bat|\btoward|\btowards|\bwith|\bbeside|\bnear|\bbehind)\s+$",
        re.IGNORECASE,
    )
    for match in _DIALOGUE_OWNER_NAME_RE.finditer(value):
        raw_name = match.group(1)
        if (
            re.fullmatch(r"S\d+", raw_name, re.IGNORECASE)
            and re.search(r"\(\s*$", value[:match.start()])
            and re.match(r"\s*\)", value[match.end():])
        ):
            continue
        leading = value[max(0, match.start() - 32):match.start()]
        if not object_preposition.search(leading):
            return True
    return False


def _is_leading_possessive_eyes_speech_cue(tail: str) -> bool:
    """Recognize a leading eye cue only when its same-subject pronoun speaks."""
    eyes = _POSSESSIVE_EYES_ACTION_RE.match(tail)
    if not eyes:
        return False
    remainder = tail[eyes.end():]
    leading = re.match(
        r"\s*(?:(?:,\s*(?:and\s+)?)|(?:and\s+))?"
        r"(?:(?:as|while)\s+)?(he|she|they)\b",
        remainder,
        re.IGNORECASE,
    )
    if not leading:
        return False
    pronoun = leading.group(1).casefold()
    possessive = {"he": "his", "she": "her", "they": "their"}[pronoun]

    voice_cue = re.match(
        rf"\s*(?:(?:,\s*(?:and\s+)?)|(?:and\s+))?"
        rf"(?:as|while)\s+{pronoun}\b"
        rf"[^.!?;:\r\n]{{0,120}}\b{possessive}\s+voice\s+"
        rf"(?:rising|rises|rose|raises|raised|softening|softens|"
        rf"shaking|shakes|quivering|quivers|cracking|cracks|"
        rf"breaking|breaks|falling|falls)\b"
        rf"(?:\s+(?:in|with)\s+(?:a|an|the|[a-z][a-z'-]*)"
        rf"(?:\s+[a-z][a-z'-]*){{0,4}})?\s*[:\-–—]\s*(?:\(S\d+\))?\s*$",
        tail[eyes.end():],
        re.IGNORECASE,
    )
    if voice_cue:
        cue_body = tail[eyes.end():eyes.end() + voice_cue.end()]
        if not _has_unintroduced_possessive_cue_name(cue_body):
            return True

    suffix = (
        r"(?:\s+(?:with|in)\s+(?:a|an|the|[a-z][a-z'-]*)"
        r"(?:\s+[a-z][a-z'-]*){0,4})?\s*[:\-–—]\s*(?:\(S\d+\))?\s*$"
    )
    speech_cue = re.search(
        rf"\b(?:as|while)\s+{pronoun}\s+(?:\w+ly\s+)?"
        rf"(?:{_SPEECH_VERB_RE.pattern}){suffix}",
        remainder,
        re.IGNORECASE,
    )
    if speech_cue:
        cue_body = remainder[leading.end():speech_cue.start()]
        if not _has_unintroduced_possessive_cue_name(cue_body):
            return True

    direct_speech_cue = re.match(
        rf"\s*(?:(?:,\s*(?:and\s+)?)|(?:and\s+))?{pronoun}\s+"
        rf"(?:\w+ly\s+)?(?:{_SPEECH_VERB_RE.pattern}){suffix}",
        remainder,
        re.IGNORECASE,
    )
    return bool(direct_speech_cue)


def _is_ref2va_possessive_speech_attribution(
    value: str,
    name_start: int,
    name_end: int,
) -> bool:
    """Accept only bounded, explicit possessive speech ownership cues.

    A speech noun followed by a screenplay delimiter is an explicit label
    ("Ana's response:"). A body-part cue is accepted only when the named
    character leads the clause and the same clause ends with that character's
    pronoun plus a speech verb ("Ana's mouth opens as she says:"). This keeps
    object mentions such as "Bea looks at Ana's mouth and says" with Bea.
    """
    tail = value[name_end:]
    if _POSSESSIVE_SPEECH_LABEL_RE.fullmatch(tail):
        return True
    prefix = value[:name_start]
    if not re.fullmatch(
        r"\s*(?:</d>\s*)?(?:(?:then|now)\s+)?",
        prefix,
        re.IGNORECASE,
    ):
        return False
    return bool(
        _POSSESSIVE_BODY_SPEECH_RE.fullmatch(tail)
        or _is_leading_possessive_eyes_speech_cue(tail)
    )


def _ref2va_alias_occurrences(source: str, aliases: dict[str, int]):
    occurrences: list[tuple[int, int, str, int]] = []
    for alias, subject in aliases.items():
        if is_h3_production_label(alias):
            continue
        alias_pattern = r"[\s_-]*".join(re.escape(char) for char in alias)
        for match in re.finditer(
            rf"(?<!\w){alias_pattern}(?!\w)",
            source,
            re.IGNORECASE,
        ):
            following = source[match.end():]
            if re.match(r"['’]s\b", following, re.IGNORECASE) and not (
                _is_ref2va_possessive_speech_attribution(
                    source, match.start(), match.end(),
                )
            ):
                # A possessive is an owner cue only for explicitly labeled
                # speech or a bounded direct body-part speech construction.
                continue
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


def _responding_gerund_object_spans(clause: str, occurrences, verb: re.Match) -> set[tuple[int, int]]:
    """Exclude prepositional object lists only for a responding gerund."""
    if verb.group(0).casefold() != "responding":
        return set()
    spans = sorted({
        (occurrence[0], occurrence[1])
        for occurrence in occurrences
        if occurrence[1] <= verb.start()
    })
    object_preposition = re.compile(
        r"(?:\bto|\bat|\btoward|\btowards|\bwith|\bbeside|\bnear|\bbehind)\s+$",
        re.IGNORECASE,
    )
    direct_objects = set()
    for index, (start, _end) in enumerate(spans):
        leading = clause[max(0, start - 32):start]
        if object_preposition.search(leading):
            direct_objects.add(index)

    object_indexes = set(direct_objects)
    separators = re.compile(r"\s*(?:(?:,|and|or|&)\s*)+", re.IGNORECASE)
    for direct_index in sorted(direct_objects):
        previous_index = direct_index
        for index in range(direct_index + 1, len(spans)):
            gap = re.sub(r"\(\s*S\d+\s*\)", "", clause[spans[previous_index][1]:spans[index][0]], flags=re.IGNORECASE)
            if not separators.fullmatch(gap):
                break
            object_indexes.add(index)
            previous_index = index
    return {spans[index] for index in object_indexes}


def _has_coordinated_responding_subject(
    clause: str,
    occurrences,
    verb: re.Match,
    object_spans: set[tuple[int, int]] | None = None,
) -> bool:
    """Do not guess one speaker when a responding gerund follows a compound subject."""
    if verb.group(0).casefold() != "responding":
        return False
    object_spans = object_spans or set()
    candidates = []
    for occurrence in occurrences:
        start, end = occurrence[0], occurrence[1]
        if end > verb.start() or (start, end) in object_spans:
            continue
        leading = clause[max(0, start - 32):start]
        if re.search(
            r"(?:\bto|\bat|\btoward|\btowards|\bwith|\bbeside|\bnear|\bbehind)\s+$",
            leading,
            re.IGNORECASE,
        ):
            continue
        candidates.append((start, end))
    if len(candidates) < 2:
        return False
    candidates.sort()
    left, right = candidates[-2:]
    gap = re.sub(
        r"\(\s*S\d+\s*\)", "", clause[left[1]:right[0]], flags=re.IGNORECASE,
    )
    return bool(re.fullmatch(
        r"\s*(?:,\s*)?(?:and|&)\s*", gap, re.IGNORECASE,
    ))

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
        found = []
        for match in _DIALOGUE_OWNER_NAME_RE.finditer(value):
            raw_name = match.group(1)
            end = match.end()
            if (
                re.fullmatch(r"S\d+", raw_name, re.IGNORECASE)
                and re.search(r"\(\s*$", value[:match.start()])
                and re.match(r"\s*\)", value[match.end():])
            ):
                # Parenthesized S-number tokens are H3 voice markers, not names.
                continue
            if re.search(r"['’]s$", raw_name, re.IGNORECASE):
                base_end = end - 2
                if not _is_ref2va_possessive_speech_attribution(
                    value, match.start(), base_end,
                ):
                    continue
                # Retain explicit possessive ownership while returning the
                # character name without its grammatical suffix.
                raw_name = raw_name[:-2]
                end = base_end
            name = _clean_ref2va_dialogue_owner_name(raw_name)
            if name and not is_h3_production_label(name):
                found.append((match.start(), end, name))
        return found

    # A leading same-subject eyes cue can attribute the line even when its
    # voice is described by a rising tone instead of a finite speech verb.
    for start, end, name in occurrences(clause):
        if (
            _is_ref2va_possessive_speech_attribution(clause, start, end)
            and _is_leading_possessive_eyes_speech_cue(clause[end:])
        ):
            return name

    # Direct screenplay syntax: ``Rachel: <d>...</d>``.
    for start, end, name in reversed(occurrences(clause)):
        tail = clause[end:]
        leading = clause[:start]
        if (
            re.fullmatch(
                r"\s*(?:['’]s\s+(?:voice|reply|whisper|answer|response)\s*)?[:,\-–—]\s*",
                tail,
                re.IGNORECASE,
            )
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

    # The other ordinary attribution order: ``<d>...</d>, Alex says``.
    post_clause = re.split(r"[.!?;\r\n]", after)[0]
    for start, end, name in occurrences(post_clause):
        if not re.search(r'<d\b|["“”]|\(S\d+\)', post_clause, re.IGNORECASE) and re.fullmatch(r"\s*[,;:\-–—]?\s*", after[:start]) and re.match(
            rf"\s+(?:\w+ly\s+)?(?:{_SPEECH_VERB_RE.pattern})", after[end:], re.IGNORECASE,
        ):
            return name

    # Natural prose: ``Blaine turns to Yoda and says, ...``. The last name
    # before the speech verb is not necessarily the speaker, so reject names
    # introduced by object prepositions such as ``to`` or ``at``.
    verbs = list(_SPEECH_VERB_RE.finditer(clause))
    if verbs:
        verb = verbs[-1]
        owner_occurrences = occurrences(clause)
        responding_objects = _responding_gerund_object_spans(
            clause, owner_occurrences, verb,
        )
        if _has_coordinated_responding_subject(
            clause, owner_occurrences, verb, responding_objects,
        ):
            return ""
        candidates = []
        for start, end, name in owner_occurrences:
            if end > verb.start() or (start, end) in responding_objects:
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
    return h3_action_beat_speaker(source, dialogue_start)


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
            re.fullmatch(
                r"\s*(?:['’]s\s+(?:voice|reply|whisper|answer|response)\s*)?[:,\-–—]\s*",
                tail,
                re.IGNORECASE,
            )
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

    post_clause = re.split(r"[.!?;\r\n]", after)[0]
    for start, end, _alias, subject in _ref2va_alias_occurrences(post_clause, speaker_aliases):
        if not re.search(r'<d\b|["“”]|\(S\d+\)', post_clause, re.IGNORECASE) and re.fullmatch(r"\s*[,;:\-–—]?\s*", after[:start]) and re.match(
            rf"\s+(?:\w+ly\s+)?(?:{_SPEECH_VERB_RE.pattern})", after[end:], re.IGNORECASE,
        ):
            return subject

    # Natural syntax before the line: ``Blaine turns to Yoda and says, ...``.
    # Select the last non-object character before the final speech verb.
    verbs = list(_SPEECH_VERB_RE.finditer(clause))
    if verbs:
        verb = verbs[-1]
        responding_objects = _responding_gerund_object_spans(
            clause, occurrences, verb,
        )
        if _has_coordinated_responding_subject(
            clause, occurrences, verb, responding_objects,
        ):
            return None
        candidates = []
        for start, end, alias, subject in occurrences:
            if end > verb.start() or (start, end) in responding_objects:
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

    # A named guest starts a new turn. Only unnamed/pronominal continuation
    # may inherit the preceding character; otherwise "Alex replies ... Jordan
    # adds ..." incorrectly gives Jordan Alex's saved identity and voice.
    owner = _resolve_ref2va_dialogue_owner_name(source, dialogue_start, dialogue_end)
    if owner:
        owner_subjects = {
            speaker_aliases[alias]
            for alias in _ref2va_alias_values({"character_name": owner})
            if alias in speaker_aliases
        }
        return next(iter(owner_subjects)) if len(owner_subjects) == 1 else None

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


def _ambiguous_ref2va_dialogue_error(words: str) -> H3SpeakerBindingError:
    excerpt = re.sub(r"\s+", " ", words).strip()[:80]
    return H3SpeakerBindingError(
        "MiniMax H3 Omni could not determine which referenced character speaks "
        f"{excerpt!r}. Name the speaker beside the line (for example, Alex says, "
        '"Hello.") or place that character\'s explicit <Subject N> tag '
        "beside the line. (Sx) labels only identify vocal-event order."
    )


_REF2VA_SUBJECT_DEFINITIONS_RE = re.compile(
    r"(?ims)^\s*subject_definitions\s*:\s*(.*?)(?="
    r"^\s*(?:summary|retention_analysis|detailed_description|"
    r"integrated_multimodal_description|overall_soundscape|"
    r"non_diegetic_music)\s*:|\Z)"
)
_REF2VA_SUBJECT_MARKER_RE = re.compile(
    r"<Subject\s+(\d+)>\s*\(S(\d+)\)", re.IGNORECASE,
)
_REF2VA_SPEAKER_MARKER_RE = re.compile(r"\(S(\d+)\)", re.IGNORECASE)
_REF2VA_CLIP_HEADING_RE = re.compile(
    r"(?im)(?:\[\s*(?:shot|scene)\s+\d+\s*\]|"
    r"(?:^|[\r\n])\s*(?:shot|scene)\s+\d+\s*:?)"
)


def _ref2va_declared_marker_subjects(
    source: str,
    valid_subjects: set[int],
) -> dict[int, int | None]:
    """Read only explicit Subject/S pairs in the prompt's definition field.

    ``None`` is a conflict sentinel.  In particular, an Audio ordinal is never
    treated as either a Subject number or a speaker number: the explicit
    ``<Subject N> (Sx)`` pair is the only supported identity declaration.
    """

    field = _REF2VA_SUBJECT_DEFINITIONS_RE.search(str(source or ""))
    if not field:
        return {}

    declared: dict[int, int | None] = {}

    def record(subject: int, marker: int) -> None:
        if subject not in valid_subjects:
            return
        if marker not in declared:
            declared[marker] = subject
        elif declared[marker] != subject:
            declared[marker] = None

    body = field.group(1)
    for match in _REF2VA_SUBJECT_MARKER_RE.finditer(body):
        record(int(match.group(1)), int(match.group(2)))
    return declared


def _ref2va_marker_owner_is_explicit(
    source: str,
    dialogue_start: int,
    dialogue_end: int,
    marker: re.Match | None,
    owner: str,
    valid_subjects: set[int],
) -> bool:
    """Whether the current line directly attributes a raw marker to its owner."""

    before_start = max(0, dialogue_start - 420)
    before = source[before_start:dialogue_start]
    # A Subject tag in the current clause is an explicit identity cue even if
    # prose elsewhere in the same sentence mentions another character.
    clause_start = max(
        before.rfind("."), before.rfind("!"), before.rfind("?"),
        before.rfind(";"), before.rfind("\n"),
    ) + 1
    explicit_subjects = [
        int(value) for value in re.findall(
            r"<Subject\s+(\d+)>", before[clause_start:], re.IGNORECASE,
        )
    ]
    if any(subject in valid_subjects for subject in explicit_subjects):
        return True
    if marker is None:
        if not owner:
            return False
        # A postposed speaker name is still an explicit attribution.
        after = source[dialogue_end:dialogue_end + 180]
        postposed = re.match(
            rf"\s*[,;:\-–—]?\s*(?:{_SPEECH_VERB_RE.pattern})"
            rf"\s+{re.escape(owner)}\b",
            after,
            re.IGNORECASE,
        )
        return bool(postposed)
    if not owner:
        return False

    # The marker is within the current event's cue. A nearby proper name binds
    # it directly; a distant action subject does not become the owner merely
    # because the marker is followed by "says".
    name_matches = list(re.finditer(
        re.escape(owner), before, re.IGNORECASE,
    ))
    if not name_matches:
        return False
    selected = name_matches[-1]
    marker_start = marker.start() - before_start
    marker_end = marker.end() - before_start
    if selected.end() <= marker_start:
        gap = before[selected.end():marker_start]
        if re.fullmatch(r"[\s,;:'’\-–—]*", gap):
            return True
        if _SPEECH_VERB_RE.search(gap):
            return True
    elif selected.start() >= marker_end:
        # ``(S1) Sam says`` is an explicit speaker cue; ``(S1) says`` has no
        # name and is resolved only by a definition or proven marker reuse.
        tail = before[selected.end():]
        if _SPEECH_VERB_RE.search(tail):
            return True

    # Postposed attribution can follow the dialogue tag rather than precede it.
    after = source[dialogue_end:dialogue_end + 180]
    if re.match(
        rf"\s*[,;:\-–—]?\s*(?:{_SPEECH_VERB_RE.pattern})"
        rf"\s+{re.escape(owner)}\b",
        after,
        re.IGNORECASE,
    ):
        return True
    return False


def _ref2va_marker_is_another_speaker_object(
    source: str,
    dialogue_start: int,
    marker: re.Match | None,
    dialogue_subject: int | None,
    speaker_aliases: dict[str, int],
) -> bool:
    """Do not learn a marker from a referenced name used as a speech object."""

    if marker is None or dialogue_subject is None:
        return False
    marker_prefix = source[max(0, marker.start() - 140):marker.start()]
    attached_name = re.search(
        r"([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})\s*$",
        marker_prefix,
    )
    if not attached_name:
        return False
    attached_subjects = {
        speaker_aliases[alias]
        for alias in _ref2va_alias_values({"character_name": attached_name.group(1)})
        if alias in speaker_aliases
    }
    if len(attached_subjects) != 1 or dialogue_subject in attached_subjects:
        return False
    clause = re.split(r"[.!?;\r\n]", marker_prefix)[-1]
    return bool(re.search(
        r"\b(?:to|at|toward|towards|with|beside|near|behind)\b[^.!?;:\r\n]*$",
        clause,
        re.IGNORECASE,
    ))


def _source_from_latest_clip_heading(
    source: str,
    dialogue_start: int,
    previous_dialogue_end: int = 0,
) -> str:
    """Blank prior events and clip context while preserving source offsets."""

    prefix = source[:dialogue_start]
    headings = list(_REF2VA_CLIP_HEADING_RE.finditer(prefix))
    context_floor = max(
        int(previous_dialogue_end), headings[-1].end() if headings else 0,
    )
    if not context_floor:
        return source
    chars = list(source)
    for index in range(context_floor):
        if chars[index] not in "\r\n":
            chars[index] = " "
    return "".join(chars)


def _ref2va_owner_identity(owner: tuple[int | None, str]) -> tuple[str, Any]:
    subject, name = owner
    if subject is not None:
        return "subject", subject
    return "guest", _normalize_ref2va_speaker_alias(name)


def _resolve_ref2va_dialogue_owners(
    source: str,
    spans: list[tuple[int, int, str]],
    speaker_aliases: dict[str, int],
    valid_subjects: set[int],
    *,
    strict: bool,
) -> list[tuple[int | None, str]]:
    """Shared per-prompt resolver; non-strict mode is for voice prefiltering."""

    source = str(source or "")
    valid_subjects = set(valid_subjects)
    declarations = _ref2va_declared_marker_subjects(source, valid_subjects)
    definitions_field = _REF2VA_SUBJECT_DEFINITIONS_RE.search(source)
    definitions_span = (
        (definitions_field.start(1), definitions_field.end(1))
        if definitions_field else None
    )
    # A binding is either a referenced Subject or a known named guest.
    marker_owners: dict[int, tuple[int | None, str] | None] = {}
    output: list[tuple[int | None, str]] = []
    previous_end = 0

    for start, end, words in spans:
        start, end = int(start), int(end)
        bounded_source = _source_from_latest_clip_heading(source, start, previous_end)
        subject = _resolve_ref2va_dialogue_speaker(
            bounded_source, start, end, speaker_aliases, valid_subjects,
        )
        owner = _resolve_ref2va_dialogue_owner_name(
            bounded_source, start, end,
        )
        if owner and is_h3_production_label(owner):
            owner = ""

        context_start = max(previous_end, start - 240)
        heading = list(_REF2VA_CLIP_HEADING_RE.finditer(source, context_start, start))
        if heading:
            context_start = max(context_start, heading[-1].end())
        markers = [
            match for match in _REF2VA_SPEAKER_MARKER_RE.finditer(
                source, context_start, start,
            )
            if not (
                definitions_span is not None
                and definitions_span[0] <= match.start() < definitions_span[1]
            )
        ]
        marker = markers[-1] if markers else None
        raw_marker = int(marker.group(1)) if marker else None
        explicit = _ref2va_marker_owner_is_explicit(
            bounded_source, start, end, marker, owner, valid_subjects,
        )
        natural_subject = subject
        marker_is_object = _ref2va_marker_is_another_speaker_object(
            source, start, marker, natural_subject, speaker_aliases,
        )

        declared: tuple[int | None, str] | None = None
        declaration_conflict = False
        if raw_marker is not None and raw_marker in declarations:
            declared_subject = declarations[raw_marker]
            if declared_subject is None:
                declaration_conflict = True
            else:
                declared = (declared_subject, "")

        proven = marker_owners.get(raw_marker) if raw_marker is not None else None
        marker_conflict = raw_marker is not None and raw_marker in marker_owners and proven is None

        # When a marker appears before a generic speech verb, the nearest
        # silent actor is not enough to establish that actor as its speaker.
        # Adjacent names, explicit Subject tags, direct speech labels, and
        # postposed names remain direct cues.
        marker_precedes_verb = bool(
            marker is not None
            and _SPEECH_VERB_RE.search(source[marker.end():start])
        )
        poisoned_marker = marker_conflict or declaration_conflict
        if raw_marker is not None and poisoned_marker and not explicit:
            subject, owner = None, ""
        elif (
            raw_marker is not None and marker_precedes_verb and not explicit
            and not marker_is_object and len(valid_subjects) > 1
        ):
            binding = declared if declared is not None else proven
            if binding is not None:
                subject, owner = binding
                if subject is not None:
                    owner = ""
            else:
                subject, owner = None, ""
        elif raw_marker is not None and not explicit:
            binding = declared if declared is not None else proven
            if binding is not None:
                subject, owner = binding
                if subject is not None:
                    owner = ""

        # An explicitly attributed current line stays readable even when its
        # raw marker conflicts with an earlier line/declaration. The marker is
        # then poisoned for future bare uses, which must not guess.
        if subject is None and not owner and len(valid_subjects) == 1 and not poisoned_marker:
            subject = next(iter(valid_subjects))

        if (
            strict and valid_subjects and raw_marker is not None
            and poisoned_marker and not explicit
        ):
            raise _ambiguous_ref2va_dialogue_error(words)
        if subject is None and not owner and len(valid_subjects) > 1 and strict:
            raise _ambiguous_ref2va_dialogue_error(words)

        current = (subject if subject in valid_subjects else None, owner)
        if raw_marker is not None:
            declaration_mismatch = bool(
                explicit
                and declared is not None
                and _ref2va_owner_identity(current) != _ref2va_owner_identity(declared)
            )
            if declaration_conflict or declaration_mismatch:
                marker_owners[raw_marker] = None
            elif marker_conflict:
                marker_owners[raw_marker] = None
            elif explicit and (current[0] is not None or current[1]):
                previous = marker_owners.get(raw_marker)
                if (
                    previous is not None
                    and _ref2va_owner_identity(previous) != _ref2va_owner_identity(current)
                ):
                    marker_owners[raw_marker] = None
                else:
                    marker_owners[raw_marker] = current
            elif raw_marker not in marker_owners and current[0] is not None and not marker_is_object:
                # Natural, unambiguous discourse may establish an owner only
                # when the marker does not precede an unattributed speech verb.
                marker_owners[raw_marker] = current

        output.append(current)
        previous_end = max(previous_end, end)

    return output


def resolve_ref2va_dialogue_owners(
    source: str,
    spans: list[tuple[int, int, str]],
    speaker_aliases: dict[str, int],
    valid_subjects: set[int],
) -> list[tuple[int | None, str]]:
    """Resolve dialogue spans with prompt-local Subject/S proof.

    Each result is ``(subject, owner_name)``. ``owner_name`` is the explicit
    human label when one was found (including a named guest); it remains empty
    for a bare line resolved from an explicit definition or prior marker proof.
    Raw S markers are scoped to this call and never numerically converted to
    Subjects. Multi-Subject ambiguity raises the normal H3 binding error.
    """

    return _resolve_ref2va_dialogue_owners(
        source, spans, speaker_aliases, valid_subjects, strict=True,
    )
