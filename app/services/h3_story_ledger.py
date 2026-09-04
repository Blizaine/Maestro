"""Shared staged story planning for MiniMax H3 timelines.

Both FL2VA sliding windows and Ref2VA editorial sequences need the same story
discipline: a requested event must happen once, quoted dialogue must remain
verbatim, and every segment must advance from a concrete state.  Asking a
small local LLM to write every finished H3 prompt in one response made those
contracts fragile and could hit the response-token ceiling.  This module keeps
the first response compact, then expands and validates one segment at a time.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
import re
from typing import Any, Callable

from services.director.long_form_story import (
    LONG_FORM_STORY_BIBLE_SCHEMA,
    build_long_form_story_bible_fallback,
    ensure_long_form_location_coverage,
    format_long_form_story_bible,
    normalize_long_form_outline,
    normalize_long_form_story_bible,
)


H3_STORY_LEDGER_VERSION = 24

_H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND = 2.1
# H3 can still deliver a clear line slightly above the conservative planning
# rate.  The render compiler may use this ceiling only to keep an exact turn
# on a sentence/clause boundary; the full sequence must continue to fit the
# conservative budget above.
_H3_DIALOGUE_MAX_WORDS_PER_SECOND = 2.3


class H3DialogueTimingError(ValueError):
    """Exact screenplay dialogue cannot fit without unsafe rewriting."""


def normalize_h3_planning_style(value: Any) -> str:
    """Return the durable AI-writing contract stored with every H3 plan."""

    return "creative" if str(value or "").strip().casefold() == "creative" else "faithful"


def _only_supplied_dialogue_requested(prompt: str) -> bool:
    """Whether creative planning must not add dialogue around quoted lines."""

    return bool(re.search(
        r"\b(?:only\s+(?:use|speak|say)?\s*(?:these|the supplied|the quoted)?\s*"
        r"(?:lines?|dialogue)|no\s+(?:extra|additional|other)\s+(?:lines?|dialogue)|"
        r"do\s+not\s+(?:add|invent|write)\s+(?:any\s+)?(?:extra\s+|additional\s+)?"
        r"(?:lines?|dialogue))\b",
        str(prompt or ""),
        flags=re.IGNORECASE,
    ))


def _creative_conversation_brief(prompt: str) -> bool:
    """Return whether Creative mode should sustain speech across windows.

    A character scene that explicitly revolves around telling, explaining,
    discussing, interviewing, or banter is different from an action scene
    that merely ends with one verbal reaction. The former needs an authored
    exchange from the opening window onward; otherwise H3 fills long silent
    stretches with improvised, unintelligible speech.
    """

    lowered = " ".join(str(prompt or "").split()).casefold()
    return bool(re.search(
        r"\b(?:talk(?:s|ed|ing)?(?:\s+(?:to|with|about))?|"
        r"convers(?:ation|e|es|ed|ing)|chat(?:s|ted|ting)?|"
        r"discuss(?:es|ed|ing)?|debate(?:s|d|ing)?|"
        r"argu(?:e|es|ed|ing)|banter(?:s|ed|ing)?|"
        r"interview(?:s|ed|ing)?|tell(?:s|ing)?|told|"
        r"explain(?:s|ed|ing)?|present(?:s|ed|ing)?|"
        r"announce(?:s|d|ing)?)\b",
        lowered,
    ))

UNREQUESTED_SPECTACLE_PATTERNS = (
    r"\bgolden\s+energy\b",
    r"\b(?:golden|blue|red|purple)\s+energy\s+(?:wave|pulse|blast)\b",
    r"\b(?:visible|glowing|luminous|colored|coloured)?\s*energy\s+"
    r"(?:wave|pulse|blast|beam|field|surge|aura)\b",
    r"\bforce\s+field\b",
    r"\btelekin(?:esis|etic)\b",
    r"\b(?:magic|magical)\s+(?:aura|blast|energy|shield|beam|wave|field)\b",
    r"\b(?:laser|lightning)\s+(?:beam|blast|bolt)\b",
)

_CONTEXT_IR_LABEL = re.compile(
    r"\b(subject_definitions|summary|retention_analysis|detailed_description|"
    r"integrated_multimodal_description|"
    r"overall_soundscape|non_diegetic_music)\s*:",
    flags=re.IGNORECASE,
)
_CONTEXT_IR_FIELD = re.compile(
    r"^[ \t]*(subject_definitions|summary|retention_analysis|"
    r"detailed_description|integrated_multimodal_description|"
    r"overall_soundscape|non_diegetic_music)\s*:\s*",
    flags=re.IGNORECASE | re.MULTILINE,
)
_SPEECH_VERB = re.compile(
    r"\b(?:says?|said|saying|speaks?|speaking|asks?|asked|replies?|replied|responds?|responded|"
    r"whispers?|whispered|shouts?|shouted|yells?|yelled|declares?|declared|"
    r"mumbles?|mumbled|mumbling|murmurs?|murmured|murmuring|mutters?|muttered|muttering|"
    r"muffles?|muffled|muffling|"
    r"states?|stated|tells?|telling|told|explains?|explained|explaining|"
    r"informs?|informed|informing|announces?|announced|announcing|"
    r"calls?\s+out|called\s+out)\b",
    flags=re.IGNORECASE,
)
_PROPER_NAME = re.compile(
    r"\b[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3}\b"
)
_PLACEHOLDER_DIALOGUE = re.compile(r"^[\s.…_-]*$")
_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "for",
    "from", "has", "have", "he", "her", "his", "in", "into", "is", "it",
    "its", "of", "on", "or", "she", "that", "the", "their", "them", "then",
    "they", "this", "through", "to", "toward", "with", "while",
}

_ACTION_VERBS = (
    "approach|arrive|attack|board|break|breathe|burst|climb|cross|descend|dive|drop|"
    "enter|exit|fall|fight|fly|grab|hold|jump|laugh|launch|leap|mount|"
    "move|plummet|race|reach|ride|run|save|smash|sprint|stand|step|"
    "take|turn|walk|yell"
)
_FAST_ACTION_RE = re.compile(
    r"\b(?:high[- ]speed|high rate of speed|extreme(?:ly)? fast|rapid|"
    r"supersonic|breakneck|plummet|free[- ]?fall|dive|race|"
    r"hurtl|speeding|never stopping|non[- ]stop)\w*\b",
    flags=re.IGNORECASE,
)

# Screenplay-style dialogue is common in pasted Studio prompts.  These labels
# describe metadata or Context-IR fields rather than speaking characters and
# must never be converted into H3 dialogue.
_SCREENPLAY_NON_SPEAKER_LABELS = {
    "action", "audio", "camera", "cast", "character", "characters",
    "detailed description", "dialogue", "director", "end", "ext",
    "exterior", "fade in", "fade out", "int", "interior", "location",
    "integrated multimodal description", "music", "non diegetic music",
    "notes", "overall soundscape", "pov",
    "prompt", "retention analysis", "scene", "shot", "style", "summary",
    "subject definitions", "time", "title", "transition", "visual",
}
_NON_CAST_PROPER_NAMES = {
    "anyone", "beat", "everybody", "everyone", "nobody", "no one",
    "camera", "okay", "ok", "someone", "starts", "that", "there", "these", "this",
    "those", "you",
}
_SCREENPLAY_DIALOGUE_RE = re.compile(
    r"(?m)^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?"
    r"(?P<speaker>[A-Za-z][A-Za-z0-9_'\u2019.\-]*"
    r"(?:[ \t]+[A-Za-z][A-Za-z0-9_'\u2019.\-]*){0,3})"
    r"(?:[ \t]*\((?P<delivery>[^)\r\n]{1,80})\))?"
    r"(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?"
    r"(?P<text>[^\r\n]+?)[ \t]*$"
)
_ENERGETIC_PERFORMANCE_RE = re.compile(
    r"\b(?:animated|animatedly|breathless|breathlessly|burst(?:s|ing)?\s+in|"
    r"eager|eagerly|energetic|energetically|enthusiastic|enthusiastically|"
    r"excited|excitedly|frantic|frantically|passionate|passionately|"
    r"rush(?:es|ed|ing)?\s+in|urgent|urgently)\b",
    flags=re.IGNORECASE,
)
_POV_RE = re.compile(r"\b(?:pov|first[- ]person)\b", flags=re.IGNORECASE)
_NONVERBAL_VOCAL_RE = re.compile(
    r"\b(?:laugh(?:s|ed|ing)?|giggl(?:e|es|ed|ing)|gasp(?:s|ed|ing)?|"
    r"grunt(?:s|ed|ing)?|sob(?:s|bed|bing)?|scream(?:s|ed|ing)?|"
    r"breath(?:es|ed|ing|less)?)\b",
    flags=re.IGNORECASE,
)
_STYLE_WORD_RE = re.compile(
    r"\b(?:cinematic|realistic|live[- ]action|film(?:ic)?|epic|thrilling|"
    r"dramatic|gritty|dark|bright|moody|stylized|rated[- ]?[rpg0-9+]+)\b",
    flags=re.IGNORECASE,
)
_H3_FULL_BODY_MOTION_RE = re.compile(
    r"\b(?:approach(?:es|ed|ing)?|arriv(?:e|es|ed|ing)|board(?:s|ed|ing)?|"
    r"climb(?:s|ed|ing)?|cross(?:es|ed|ing)?|descend(?:s|ed|ing)?|"
    r"div(?:e|es|ed|ing)|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|"
    r"fall(?:s|ing)?|fl(?:y|ies|ew|ying)|jump(?:s|ed|ing)?|"
    r"launch(?:es|ed|ing)?|leap(?:s|ed|ing)?|mount(?:s|ed|ing)?|"
    r"move(?:s|d|ing)?|plummet(?:s|ed|ing)?|race(?:s|d|ing)?|"
    r"rid(?:e|es|ing)|run(?:s|ning)?|sprint(?:s|ed|ing)?|"
    r"step(?:s|ped|ping)?|walk(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
_H3_BLOCKING_CHANGE_RE = re.compile(
    r"\b(?:crouch(?:es|ed|ing)?|grab(?:s|bed|bing)?|kneel(?:s|ed|ing)?|"
    r"(?:lie(?:s)?|lying)|lower(?:s|ed|ing)?|pick(?:s|ed|ing)?\s+up|"
    r"pivot(?:s|ed|ing)?|raise(?:s|d|ing)?|reach(?:es|ed|ing)?|"
    r"sit(?:s|ting)?(?:\s+down)?|stand(?:s|ing)?(?:\s+up)?|"
    r"turn(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
_H3_VISIBLE_REACTION_RE = re.compile(
    r"\b(?:breath(?:e|es|ed|ing)|gasp(?:s|ed|ing)?|glance(?:s|d|ing)?|"
    r"laugh(?:s|ed|ing)?|look(?:s|ed|ing)?|nod(?:s|ded|ding)?|"
    r"recoil(?:s|ed|ing)?|shrug(?:s|ged|ging)?|sigh(?:s|ed|ing)?|"
    r"smil(?:e|es|ed|ing)|stare(?:s|d|ing)?)\b",
    flags=re.IGNORECASE,
)
_H3_IMPACT_ACTION_RE = re.compile(
    r"\b(?:attack(?:s|ed|ing)?|break(?:s|ing)?|fight(?:s|ing)?|"
    r"hit(?:s|ting)?|kick(?:s|ed|ing)?|punch(?:es|ed|ing)?|"
    r"save(?:s|d|ing)?|smash(?:es|ed|ing)?|throw(?:s|ing)?|"
    r"tackle(?:s|d|ing)?)\b",
    flags=re.IGNORECASE,
)


def normalize_h3_dialogue_tags(value: Any) -> str:
    """Repair harmless whitespace in user-authored ``<d>`` tags.

    Mobile keyboards and pasted prompts commonly produce ``< d>`` or
    ``< / d >``.  MiniMax understands only the canonical spelling, and the
    story ledger must recognize the line before it removes dialogue from the
    visual event stream.  Normalize only the tag delimiters; spoken words are
    left byte-for-byte unchanged.
    """

    text = str(value or "")
    text = re.sub(r"<\s*d\s*>", "<d>", text, flags=re.IGNORECASE)
    return re.sub(r"<\s*/\s*d\s*>", "</d>", text, flags=re.IGNORECASE)


def _is_style_only_fragment(value: str) -> bool:
    """Return whether a fragment is a visual directive, not a story event."""

    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    if not text or not _STYLE_WORD_RE.search(text):
        return False
    # Style tails are commonly comma-separated adjective lists. A real event
    # has a concrete action verb and remains part of the immutable story.
    has_action = re.search(
        rf"\b(?:{_ACTION_VERBS})(?:s|es|ed|ing)?\b",
        text,
        flags=re.IGNORECASE,
    )
    return not has_action


def _is_subject_only_fragment(value: str) -> bool:
    """Drop orphaned names created while splitting a compound action.

    A construction such as ``Thanos, while standing nearby, snaps...`` can be
    split at the comma before the action splitter sees it.  The bare
    ``Thanos`` fragment is not a filmable event, but historically it became a
    ledger beat and then a full H3 shot.  Keep this deliberately narrow so a
    short imperative such as ``Run`` is not discarded.
    """

    text = sanitize_h3_prompt_text(value).strip(" \t\r\n-.,;:!?")
    return bool(re.fullmatch(
        r"(?:the\s+)?[A-Z][A-Za-z0-9_'’-]*"
        r"(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3}",
        text,
    ))


def _is_creation_directive(value: str) -> bool:
    """Identify a user instruction that establishes a work, not an action.

    ``Make a scene from Friends`` belongs in the shared story context. It is
    not something an actor should visibly perform in shot one. Treat only a
    narrow imperative form as metadata so physical uses of make/create remain
    valid story events.
    """

    text = sanitize_h3_prompt_text(value).strip(" \t\r\n-.,;:!?")
    return bool(re.fullmatch(
        r"(?:please\s+)?(?:make|create|generate|write|direct)\s+(?:me\s+)?"
        r"(?:a|an|the)?\s*(?:scene|video|clip|film|movie|sequence|episode)"
        r"(?:\s+(?:from|in|for|of|like|set\s+in)\s+.+)?",
        text,
        flags=re.IGNORECASE,
    ))


def _is_screenplay_performance_directive(value: str) -> bool:
    """Identify prose that only introduces the screenplay dialogue below it."""

    text = sanitize_h3_prompt_text(value).strip(" \t\r\n-.,;:!?")
    proper = _PROPER_NAME.pattern
    return bool(re.fullmatch(
        rf"(?:(?:{proper}|he|she|they)\s+)?"
        r"(?:starts?|begins?|continues?|keeps?)\s+"
        r"(?:(?:very|visibly)\s+)?"
        r"(?:(?:animatedly|breathlessly|calmly|eagerly|energetically|"
        r"enthusiastically|excitedly|frantically|nervously|passionately|"
        r"quietly|softly|urgently)\s+)?"
        r"(?:talking|speaking|telling|explaining)"
        rf"(?:\s+(?:to|with)\s+(?:{proper}|him|her|them))?"
        r"(?:\s+about\s+.+)?",
        text,
        flags=re.IGNORECASE,
    ))


def _collapse_duplicate_screenplay_entrances(fragments: list[str]) -> list[str]:
    """Drop a synopsis entrance repeated by a more concrete script entrance.

    Pasted scripts often start with a one-line premise (``George walks in``),
    followed by blocking (``George bursts through the door``) and screenplay
    dialogue.  Rendering both creates two copies of the same principal.  This
    deliberately applies only before the first screenplay speech cue and only
    when the later entrance names a physical threshold, with no exit/return
    language that would establish a genuine second entrance.
    """

    first_speech = next((
        index for index, value in enumerate(fragments)
        if re.fullmatch(
            r"[A-Z][A-Za-z0-9_'\u2019-]*(?:\s+[A-Z][A-Za-z0-9_'\u2019-]*){0,3}\s+speaks",
            value,
            flags=re.IGNORECASE,
        )
    ), len(fragments))
    seen: dict[str, int] = {}
    remove: set[int] = set()
    for index, value in enumerate(fragments[:first_speech]):
        entrance = _find_h3_opening_entrance(value)
        if not entrance:
            continue
        entrant = _normalize_key(entrance.group("entrant"))
        previous = seen.get(entrant)
        if previous is not None:
            between = " ".join(fragments[previous + 1:index + 1])
            genuine_reentry = bool(re.search(
                r"\b(?:again|another|next|second|returns?|re[- ]?enters?|"
                r"exits?|leaves?|left)\b",
                between,
                flags=re.IGNORECASE,
            ))
            concrete_threshold = bool(re.search(
                r"\b(?:door|doorway|entrance|gate|threshold)\b",
                value,
                flags=re.IGNORECASE,
            ))
            if concrete_threshold and not genuine_reentry:
                remove.add(previous)
        seen[entrant] = index
    return [value for index, value in enumerate(fragments) if index not in remove]


def _is_persistent_camera_directive(value: str) -> bool:
    """Identify camera/pacing requirements that should span every window."""

    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    if not text:
        return False
    return bool(
        re.match(
            r"^(?:extreme(?:ly)?|very)?\s*(?:exciting|fast|dynamic|kinetic)?"
            r"\s*(?:first[- ]person\s+)?pov\b",
            text,
            flags=re.IGNORECASE,
        )
        and re.search(r"\b(?:speed|view|camera|hands?|handle|stick)\b", text, re.IGNORECASE)
    )


def _is_persistent_audio_directive(value: str) -> bool:
    """Identify global ambience/voice-mixing notes, not plot events."""

    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    if not text:
        return False
    return bool(re.fullmatch(
        r"(?:atmospheric|natural|cinematic)?\s*(?:ambient|environmental)?\s*"
        r"(?:ambiance|ambience|room\s+tone|soundscape)|"
        r"character\s+voices?\s+(?:should\s+)?sound(?:s)?\s+natural(?:ly)?"
        r"(?:\s+in\s+(?:the|their)\s+environment)?|"
        r"voices?\s+(?:should\s+)?match(?:es)?\s+(?:the\s+)?(?:scene|location|environment)"
        r"(?:\s+acoustics?)?",
        text,
        flags=re.IGNORECASE,
    ))


_CAST_ACTION_RE = re.compile(
    r"\b(?:approach(?:es|ed|ing)?|arriv(?:e|es|ed|ing)|attack(?:s|ed|ing)?|"
    r"breathe(?:s|d|ing)?|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|fight(?:s|ing)?|"
    r"fly|flies|flew|flying|grab(?:s|bed|bing)?|hold(?:s|ing)?|jump(?:s|ed|ing)?|"
    r"laugh(?:s|ed|ing)?|look(?:s|ed|ing)?|move(?:s|d|ing)?|nod(?:s|ded|ding)?|"
    r"punch(?:es|ed|ing)?|raise(?:s|d|ing)?|react(?:s|ed|ing)?|run(?:s|ning)?|"
    r"say(?:s|ing)?|said|sit(?:s|ting)?|sat|speak(?:s|ing)?|spoke|stand(?:s|ing)?|stood|"
    r"tell(?:s|ing)?|told|turn(?:s|ed|ing)?|walk(?:s|ed|ing)?|wave(?:s|d|ing)?|"
    r"wear(?:s|ing)?|watch(?:es|ed|ing)?|is|are|was|were)\b",
    flags=re.IGNORECASE,
)
_CAST_INTERACTION_RE = re.compile(
    r"\b(?:alongside|beside|between|faces?|facing|fight(?:s|ing)?|helps?|meets?|punch(?:es|ed)?|"
    r"attacks?|saves?|sees?|watches?|with|looks?\s+at|speaks?\s+to|"
    r"talks?\s+to|asks?|tells?)\s+$",
    flags=re.IGNORECASE,
)
_SETTING_NAME_PREFIX_RE = re.compile(
    r"\b(?:apartment|building|cafe|café|city|coffee\s+shop|country|film|"
    r"game|location|model|movie|neighborhood|planet|restaurant|room|school|"
    r"series|show|street|studio|town|tv\s+show|version|village|world)"
    r"(?:\s+(?:called|named|on|from|in))?\s+$",
    flags=re.IGNORECASE,
)
_SETTING_NAME_SUFFIXES = {
    "apartment", "building", "cafe", "café", "city", "country", "film",
    "island", "mountain", "planet", "restaurant", "room", "school", "series",
    "show", "street", "studio", "town", "village", "world",
}


def _same_h3_cast_identity(left: Any, right: Any) -> bool:
    """Match a saved-character label to its prompt-native portrayal name."""

    a = _normalize_key(left)
    b = _normalize_key(right)
    if not a or not b:
        return False
    if a == b:
        return True
    # ``Henry Cavill as Superman`` and ``Superman`` describe one principal,
    # as do the equivalent ``played by`` forms. Ordinary nested names are not
    # merged merely because they share one token.
    return bool(
        ((" as " in f" {a} " or " played by " in f" {a} ") and f" {b} " in f" {a} ")
        or ((" as " in f" {b} " or " played by " in f" {b} ") and f" {a} " in f" {b} ")
    )


def _plain_h3_name_parts(value: Any) -> list[str]:
    """Return ordinary person-name tokens, excluding actor/role phrases."""

    text = sanitize_h3_prompt_text(value).strip(" ,;:.-")
    lowered = f" {text.casefold()} "
    if " as " in lowered or " played by " in lowered:
        return []
    return re.findall(r"[A-Za-z0-9_'’-]+", text)


def _source_separates_h3_name_aliases(
    prompt: Any,
    short_name: str,
    full_name: str,
) -> bool:
    """Preserve two people only when the source explicitly contrasts them."""

    source = str(prompt or "")
    short = re.escape(short_name)
    full = re.escape(full_name)
    return bool(re.search(
        rf"\b(?:{short})\b\s*(?:,\s*)?(?:and|with|beside|alongside|meets?)\s+"
        rf"\b(?:{full})\b|\b(?:{full})\b\s*(?:,\s*)?"
        rf"(?:and|with|beside|alongside|meets?)\s+\b(?:{short})\b",
        source,
        flags=re.IGNORECASE,
    ))


def _canonicalize_h3_cast_names(
    names: list[str],
    *,
    prompt: Any = "",
) -> list[str]:
    """Collapse an established full name and its later unambiguous shorthand.

    ``George Costanza`` followed by ``George`` is one principal. A shared
    first name remains ambiguous when two full cast names own it, and an
    explicit ``George and George Costanza`` construction remains two people.
    """

    deduplicated: list[str] = []
    for value in names:
        name = sanitize_h3_prompt_text(value).strip(" ,;:.-")
        if name and not any(
            _same_h3_cast_identity(name, existing)
            for existing in deduplicated
        ):
            deduplicated.append(name)

    long_names = [
        name for name in deduplicated
        if len(_plain_h3_name_parts(name)) >= 2
    ]
    replacements: dict[str, str] = {}
    for name in deduplicated:
        parts = _plain_h3_name_parts(name)
        if len(parts) != 1:
            continue
        token = _normalize_key(parts[0])
        candidates = [
            full_name
            for full_name in long_names
            if token in {
                _normalize_key(_plain_h3_name_parts(full_name)[0]),
                _normalize_key(_plain_h3_name_parts(full_name)[-1]),
            }
        ]
        if (
            len(candidates) == 1
            and not _source_separates_h3_name_aliases(
                prompt,
                name,
                candidates[0],
            )
        ):
            replacements[_normalize_key(name)] = candidates[0]

    canonical: list[str] = []
    for name in deduplicated:
        resolved = replacements.get(_normalize_key(name), name)
        if not any(
            _same_h3_cast_identity(resolved, existing)
            or _normalize_key(resolved) == _normalize_key(existing)
            for existing in canonical
        ):
            canonical.append(resolved)
    return canonical


def _h3_cast_aliases(name: str, all_names: list[str]) -> list[str]:
    """Return only aliases that identify one unambiguous active principal."""

    aliases = [sanitize_h3_prompt_text(name)]
    actor_role = re.fullmatch(r"(.+?)\s+as\s+(.+)", name, flags=re.IGNORECASE)
    played_by = re.fullmatch(
        r"(.+?)\s*\(\s*played\s+by\s+(.+?)\s*\)",
        name,
        flags=re.IGNORECASE,
    )
    if actor_role or played_by:
        aliases.extend([
            sanitize_h3_prompt_text((actor_role or played_by).group(1)),
            sanitize_h3_prompt_text((actor_role or played_by).group(2)),
        ])
        return list(dict.fromkeys(alias for alias in aliases if alias))

    parts = _plain_h3_name_parts(name)
    if len(parts) >= 2:
        for token in (parts[0], parts[-1]):
            normalized = _normalize_key(token)
            owners: list[str] = []
            for candidate in all_names:
                candidate_parts = _plain_h3_name_parts(candidate)
                if len(candidate_parts) >= 2 and normalized in {
                    _normalize_key(candidate_parts[0]),
                    _normalize_key(candidate_parts[-1]),
                }:
                    owners.append(candidate)
            if len(owners) == 1:
                aliases.append(token)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _resolve_h3_cast_name(value: Any, names: list[str]) -> str:
    """Resolve a dialogue/event shorthand to its canonical cast identity."""

    original = sanitize_h3_prompt_text(value).strip(" ,;:.-") or "Speaker"
    direct = [
        name for name in names
        if _same_h3_cast_identity(original, name)
        or _normalize_key(original) == _normalize_key(name)
    ]
    if len(direct) == 1:
        return direct[0]
    key = _normalize_key(original)
    aliases = [
        name for name in names
        if key in {
            _normalize_key(alias)
            for alias in _h3_cast_aliases(name, names)
        }
    ]
    return aliases[0] if len(aliases) == 1 else original


def _find_h3_opening_entrance(prompt: Any) -> re.Match[str] | None:
    """Find an explicitly requested entrance at the start of a scene.

    A location-establishing beat followed by dialogue does not need its own
    camera phase when every principal is already present.  This match is kept
    deliberately narrower than the general motion vocabulary so the opening
    pacing guard applies only when someone actually enters the scene.
    """

    proper = _PROPER_NAME.pattern
    return re.search(
        rf"(?P<entrant>{proper})\s+(?i:"
        r"enter(?:s|ed|ing)?|arriv(?:e|es|ed|ing)?|"
        r"(?:walk|run|step|come)(?:s|ed|ing)?\s+(?:in|into|through)|"
        r"burst(?:s|ed|ing)?\s+(?:in|into|through))\b",
        sanitize_h3_prompt_text(prompt),
    )


def _infer_h3_opening_state_contract(
    prompt: Any,
    cast_names: list[str],
) -> str:
    """Describe the frame immediately before a requested entrance begins."""

    source = sanitize_h3_prompt_text(prompt)
    proper = _PROPER_NAME.pattern
    entrance = _find_h3_opening_entrance(source)
    if not entrance:
        return ""
    entrant = _resolve_h3_cast_name(entrance.group("entrant"), cast_names)
    stationary = re.search(
        rf"(?P<target>{proper})\s*,\s*(?i:who\s+is)\s+"
        r"(?P<state>[^.!?]{1,140})",
        source[entrance.end():],
    )
    if stationary:
        target = _resolve_h3_cast_name(
            stationary.group("target"),
            cast_names,
        )
        state = sanitize_h3_prompt_text(stationary.group("state")).strip(" ,;:.-")
        return (
            f"{target} is {state} in the established location; {entrant} has "
            "not yet entered and remains outside the frame"
        )
    return (
        f"The established location is visible immediately before {entrant} "
        f"enters; {entrant} has not yet entered and remains outside the frame"
    )


def _canonicalize_h3_dialogue_speakers(
    dialogue: list[dict[str, Any]],
    cast_names: list[str],
) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for item in dialogue:
        repaired = dict(item)
        repaired["speaker"] = _resolve_h3_cast_name(
            repaired.get("speaker"),
            cast_names,
        )
        canonical.append(repaired)
    return canonical


def _reference_h3_cast_names(value: Any) -> list[str]:
    """Read human role names from Maestro-owned Subject bindings."""

    names: list[str] = []
    for match in re.finditer(
        r"<Subject\s+\d+>\s+is\s+(.+?)"
        r"(?=\s+from\s+<(?:Picture|Video)\s+\d+>|"
        r",\s+(?:whose|with|defined|from|preserving)\b|[;.\r\n]|$)",
        str(value or ""),
        flags=re.IGNORECASE,
    ):
        name = sanitize_h3_prompt_text(match.group(1)).strip(" ,;:.-")
        if name and not name.casefold().startswith((
            "the environment", "the location", "the visual style", "the visual treatment",
        )):
            names.append(name)
    return list(dict.fromkeys(names))


def _source_requests_multiple_cast_instances(prompt: Any, name: Any) -> bool:
    """Allow explicitly requested twins, clones, copies, or multiple versions."""

    source = str(prompt or "")
    escaped = re.escape(sanitize_h3_prompt_text(name))
    if not escaped:
        return False
    quantity = (
        r"(?:two|three|four|five|six|seven|eight|nine|ten|multiple|several|many|"
        r"a\s+pair\s+of|a\s+group\s+of)"
    )
    return bool(re.search(
        rf"(?:{quantity})\s+(?:identical\s+)?(?:copies?\s+of\s+)?{escaped}\b|"
        rf"\b{escaped}(?:es|s)?\b[^.!?]{{0,45}}\b(?:twins?|clones?|copies|"
        r"duplicates?|multiple\s+versions?)\b",
        source,
        flags=re.IGNORECASE,
    ))


def _derive_h3_cast_names(source: str, proper_names: list[str]) -> list[str]:
    """Separate named principals from works, products, and named locations."""

    speakers = {
        sanitize_h3_prompt_text(item.get("speaker")).casefold()
        for item in extract_locked_dialogue(source)
        if sanitize_h3_prompt_text(item.get("speaker")) not in {"", "Speaker"}
    }
    cast: list[str] = []
    for name in proper_names:
        clean = sanitize_h3_prompt_text(name)
        if not clean:
            continue
        if clean.casefold() in speakers:
            cast.append(clean)
            continue
        if clean.split()[-1].casefold() in _SETTING_NAME_SUFFIXES:
            continue
        evidence = False
        for match in re.finditer(
            rf"(?<![\w]){re.escape(clean)}(?![\w])",
            source,
            flags=re.IGNORECASE,
        ):
            before = source[max(0, match.start() - 90):match.start()]
            after = source[match.end():match.end() + 100]
            before_clause = re.split(r"[.!?;]", before)[-1]
            after_clause = re.split(r"[.!?;]", after)[0]
            setting_label = bool(_SETTING_NAME_PREFIX_RE.search(before_clause))
            location_preposition = bool(re.search(
                r"\b(?:at|from|in|inside|into|outside|through|to|toward)\s+"
                r"(?:(?:a|an|the)\s+)?$",
                before_clause,
                flags=re.IGNORECASE,
            ))
            followed_by_setting_noun = bool(re.match(
                r"\s+(?:apartment|building|cafe|café|city|coffee\s+shop|film|"
                r"island|movie|neighborhood|planet|restaurant|room|school|series|"
                r"show|shop|street|studio|town|village|world)\b",
                after,
                flags=re.IGNORECASE,
            ))
            if setting_label or location_preposition or followed_by_setting_noun:
                continue
            performs = bool(_CAST_ACTION_RE.search(after_clause))
            participates = bool(_CAST_INTERACTION_RE.search(before_clause))
            sentence_subject = not before_clause.strip(" ,:-") and performs
            if participates or sentence_subject or (
                performs
            ):
                evidence = True
                break
        if evidence:
            cast.append(clean)

    # Explicit actor/role notation represents one principal, not two copies.
    aliases: list[tuple[str, list[str]]] = []
    proper = _PROPER_NAME.pattern
    for pattern in (
        re.compile(rf"({proper})\s+as\s+({proper})"),
        re.compile(
            rf"({proper})\s*\(\s*played\s+by\s+({proper})\s*\)",
            flags=re.IGNORECASE,
        ),
    ):
        for match in pattern.finditer(source):
            aliases.append((
                sanitize_h3_prompt_text(match.group(0)),
                [sanitize_h3_prompt_text(match.group(1)), sanitize_h3_prompt_text(match.group(2))],
            ))
    merged: list[str] = []
    consumed: set[str] = set()
    for name in cast:
        if name.casefold() in consumed:
            continue
        alias = next(
            (
                (phrase, members) for phrase, members in aliases
                if any(_same_h3_cast_identity(name, member) for member in members)
            ),
            None,
        )
        if alias:
            phrase, members = alias
            consumed.update(member.casefold() for member in members)
            if phrase not in merged:
                merged.append(phrase)
        elif not any(_same_h3_cast_identity(name, existing) for existing in merged):
            merged.append(name)
    return _canonicalize_h3_cast_names(merged, prompt=source)


def _merge_h3_cast_names(
    *groups: list[str],
    prompt: Any = "",
) -> list[str]:
    raw: list[str] = []
    for group in groups:
        for value in group:
            name = sanitize_h3_prompt_text(value).strip(" ,;:.-")
            if name:
                raw.append(name)
    return _canonicalize_h3_cast_names(raw, prompt=prompt)


def _h3_cast_cardinality_contract(prompt: Any, names: list[str]) -> str:
    singular = [
        name for name in names
        if not _source_requests_multiple_cast_instances(prompt, name)
    ]
    plural = [name for name in names if name not in singular]
    parts: list[str] = []
    if singular:
        parts.append(
            "Keep exactly one identity instance of each named principal: "
            + ", ".join(singular)
        )
    if plural:
        parts.append(
            "Preserve the explicitly requested multiplicity for "
            + ", ".join(plural)
        )
    if names:
        parts.append(
            "Only principals named in a segment's assigned action or opening state appear there; later entrants do not appear early"
        )
    return ". ".join(parts)


def _infer_h3_blocking_contract(prompt: Any, names: list[str]) -> str:
    """Compile explicit relational blocking into a persistent screen map."""

    source = sanitize_h3_prompt_text(prompt)
    if len(names) < 2:
        return ""
    alternatives = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True)
    )
    between = re.search(
        rf"\b(?P<center>{alternatives})\b(?P<lead>[^.!?]{{0,180}}?)"
        rf"\b(?P<verb>sits?(?:\s+down)?|sat|stands?|stood|moves?)\b"
        rf"[^.!?]{{0,80}}?\bbetween\s+(?P<left>{alternatives})\s+and\s+"
        rf"(?P<right>{alternatives})\b",
        source,
        flags=re.IGNORECASE,
    )
    if between:
        resolved: dict[str, str] = {}
        for key in ("center", "left", "right"):
            value = sanitize_h3_prompt_text(between.group(key))
            resolved[key] = next(
                (name for name in names if name.casefold() == value.casefold()),
                value,
            )
        moving_entry = bool(re.search(
            r"\b(?:approach|arriv|enter|walk|run|move)\w*\b",
            between.group("lead"),
            flags=re.IGNORECASE,
        ))
        if moving_entry:
            return (
                f"Before {resolved['center']} sits, {resolved['left']} and {resolved['right']} occupy opposite seats with one empty place between them while {resolved['center']} remains separate. "
                f"After that sitting beat, preserve the stable screen order {resolved['left']} - {resolved['center']} - {resolved['right']}"
            )
        return (
            f"Preserve {resolved['center']} between {resolved['left']} and {resolved['right']} with the stable screen order "
            f"{resolved['left']} - {resolved['center']} - {resolved['right']}"
        )

    placements: list[str] = []
    for name in names:
        match = re.search(
            rf"\b{re.escape(name)}\b[^.!?]{{0,70}}?\b"
            r"(screen[- ]left|screen[- ]right|screen[- ]center|center frame)\b",
            source,
            flags=re.IGNORECASE,
        )
        if match:
            placements.append(f"{name} remains {match.group(1).lower()}")
    return "Preserve established screen geography: " + "; ".join(placements) if placements else ""


def _active_h3_cast_names(names: list[str], value: Any) -> list[str]:
    text = str(value or "")
    active: list[str] = []
    for name in names:
        aliases = _h3_cast_aliases(name, names)
        if any(re.search(
            rf"(?<![\w]){re.escape(alias)}(?![\w])",
            text,
            flags=re.IGNORECASE,
        ) for alias in aliases if alias):
            active.append(name)
    return active


def extract_h3_source_intent(prompt: str) -> dict[str, Any]:
    """Extract immutable camera, pacing, style, and vocal requirements.

    These facts are application-owned. They must survive even when the local
    planning LLM returns malformed JSON or exhausts its response budget.
    """

    raw_source = normalize_h3_dialogue_tags(prompt)
    locked_dialogue = extract_locked_dialogue(raw_source)
    source = sanitize_h3_prompt_text(raw_source)
    directive_source = sanitize_h3_prompt_text(_without_locked_dialogue(
        raw_source,
        locked_dialogue,
        keep_screenplay_speaker_cues=True,
    ))
    directive_source = re.sub(
        r"<d>\s*(?:\[[^\]]+\]\s*)?(?:(?!<d>).)*?</d>",
        ". ",
        directive_source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    lowered = directive_source.casefold()
    pov = bool(_POV_RE.search(directive_source))
    identity_match = re.search(
        r"\bviewer\s+is\s+([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})"
        r"(?=\s+(?:as|while|who|standing|walking|running|flying|riding)\b|[.,;:])",
        source,
    )
    if not identity_match:
        identity_match = re.search(
            r"\bPOV\s*:\s*(?:the\s+viewer\s+is\s+)?"
            r"([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})"
            r"(?=\s+(?:as|while|who|standing|walking|running|flying|riding)\b|[.,;:])",
            source,
            flags=re.IGNORECASE,
        )
    pov_identity = sanitize_h3_prompt_text(
        identity_match.group(1) if identity_match else ""
    )

    proper_names: list[str] = []
    names_source = directive_source
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3}\b",
        names_source,
    ):
        name = sanitize_h3_prompt_text(match.group(0))
        parts = name.split()
        while len(parts) > 1 and parts[0].casefold() in {
            "a", "an", "and", "both", "make", "the", "then", "extremely", "epic",
        }:
            parts.pop(0)
        name = " ".join(parts)
        if name.casefold() in {
            "a", "an", "and", "both", "make", "the", "then", "extremely", "epic",
            "friends", "maestro",
        } | _NON_CAST_PROPER_NAMES or (name.isupper() and len(name) <= 3):
            continue
        if name not in proper_names:
            proper_names.append(name)
    cast_names = _derive_h3_cast_names(raw_source, proper_names)
    cast_cardinality = _h3_cast_cardinality_contract(raw_source, cast_names)
    blocking_contract = _infer_h3_blocking_contract(raw_source, cast_names)

    style_fragments = [
        fragment.strip(" ,;:-.!?")
        for fragment in re.split(r"(?<=[.!?])\s+", directive_source)
        if _is_style_only_fragment(fragment)
    ]
    camera_fragments = [
        fragment.strip(" ,;:-.!?")
        for fragment in re.split(r"(?<=[.!?])\s+", directive_source)
        if _is_persistent_camera_directive(fragment)
    ]
    audio_fragments = [
        fragment.strip(" ,;:-.!?")
        for fragment in re.split(r"(?<=[.!?])\s+", directive_source)
        if _is_persistent_audio_directive(fragment)
    ]
    nonverbal = list(dict.fromkeys(
        match.group(0).casefold()
        for match in _NONVERBAL_VOCAL_RE.finditer(directive_source)
    ))
    hands_visible = bool(
        re.search(r"\b(?:both|two)?\s*hands?\b", directive_source, re.IGNORECASE)
        and re.search(r"\b(?:holding|gripping|grasping)\b", directive_source, re.IGNORECASE)
    )
    ongoing = bool(re.search(
        r"\b(?:never[- ]ending|never stopping|non[- ]stop|keeps? (?:moving|falling|flying)|"
        r"continues? indefinitely|ongoing)\b",
        lowered,
    ))

    perspective_parts: list[str] = []
    if pov:
        identity = f" of {pov_identity}" if pov_identity else ""
        perspective_parts.append(
            f"Lock the camera to the first-person POV{identity}; the viewpoint character never appears in an external shot"
        )
    if hands_visible:
        perspective_parts.append(
            "Keep the requested hands and held object visible naturally in the foreground during the moving POV action"
        )
    perspective_parts.extend(camera_fragments)

    energetic_performance = bool(_ENERGETIC_PERFORMANCE_RE.search(directive_source))
    pacing = (
        "extremely fast real-time movement with sustained forward momentum, decisive choreography, and no slow motion"
        if _FAST_ACTION_RE.search(directive_source)
        else "brisk, energetic real-time pacing with immediate expressive performance and no slow motion"
        if energetic_performance
        else "natural real-time pacing"
    )
    ambient_parts: list[str] = []
    if re.search(r"\bmountain|cliff|canyon|clouds?\b", lowered):
        ambient_parts.append("open-air mountain wind")
    if _FAST_ACTION_RE.search(directive_source):
        ambient_parts.append("speed-dependent rushing air")
    return {
        "first_person_pov": pov,
        "pov_identity": pov_identity,
        "proper_names": proper_names,
        "cast_names": cast_names,
        "cast_cardinality_contract": cast_cardinality,
        "blocking_contract": blocking_contract,
        "opening_state_contract": _infer_h3_opening_state_contract(
            raw_source,
            cast_names,
        ),
        "fast_action": bool(_FAST_ACTION_RE.search(directive_source)),
        "energetic_performance": energetic_performance,
        "opening_dialogue_id": _opening_h3_dialogue_id(
            raw_source,
            locked_dialogue,
            extract_source_events(raw_source),
        ),
        "ongoing_motion": ongoing,
        "hands_visible": hands_visible,
        "perspective_contract": ". ".join(perspective_parts),
        "style_contract": ". ".join(style_fragments),
        "pacing_contract": pacing,
        "requested_nonverbal_vocals": (
            "Requested nonverbal vocalizations remain audible: " + ", ".join(nonverbal)
            if nonverbal else ""
        ),
        "ambient_contract": "; ".join([*audio_fragments, *ambient_parts]),
    }


def sanitize_h3_prompt_text(value: Any) -> str:
    """Return value text that cannot become a prompt-template expression.

    WGP applies a lightweight ``{variable}`` template pass after enhancement.
    JSON-like prose from an LLM therefore must not retain literal braces.  We
    also neutralize nested Context-IR labels so every compiled prompt owns one
    and only one instance of each field.
    """

    text = str(value or "")
    text = (
        text.replace("{", "(")
        .replace("}", ")")
        .replace("â", "'")
        .replace("â", '"')
        .replace("â", '"')
    )
    text = _CONTEXT_IR_LABEL.sub(lambda match: f"{match.group(1)} -", text)
    return " ".join(text.split())


def recover_h3_plain_story(value: Any) -> str:
    """Unwrap an already-enhanced Context-IR prompt for story planning.

    Studio can legitimately enhance one native H3 clip before the user later
    enables a longer sequence. Feeding that six-field runtime prompt into the
    sequence planner treats reference contracts and timestamps as plot beats.
    Recover its human-readable summary and exact tagged dialogue instead.
    Plain user concepts pass through unchanged.
    """

    source = normalize_h3_dialogue_tags(value).strip()
    matches = list(_CONTEXT_IR_FIELD.finditer(source))
    labels = {match.group(1).casefold() for match in matches}
    if not {"summary", "detailed_description", "retention_analysis"}.issubset(labels):
        return source
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        fields[match.group(1).casefold()] = source[match.end():end].strip()
    summary = re.sub(
        r"^\s*\[[^\]\r\n]{1,160}\]\s*",
        "",
        fields.get("summary", ""),
    )
    summary = sanitize_h3_prompt_text(summary)
    if not summary:
        return source

    detailed = fields.get("detailed_description", "")
    dialogue_events: list[str] = []
    for match in re.finditer(
        r"<d>\s*(?:\[[^\]]+\]\s*)?((?:(?!<d>).)*?)\s*</d>",
        detailed,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        text = sanitize_h3_prompt_text(match.group(1)).strip()
        if not text or _PLACEHOLDER_DIALOGUE.fullmatch(text):
            continue
        prefix = detailed[max(0, match.start() - 260):match.start()]
        speaker = "Speaker"
        name_first = re.search(
            r"([A-Z][A-Za-z0-9_'â€™-]*(?:\s+[A-Z][A-Za-z0-9_'â€™-]*){0,3})"
            r"\s*\(S\d+\)[^.!?<>]{0,190}$",
            prefix,
        )
        id_first = re.search(r"\bS\d+\s*\(([^)]+)\)[^.!?<>]{0,190}$", prefix)
        if name_first:
            speaker = sanitize_h3_prompt_text(name_first.group(1))
        elif id_first:
            speaker = sanitize_h3_prompt_text(id_first.group(1))
        if text.casefold() not in summary.casefold():
            dialogue_events.append(f'{speaker} says "{text}".')
    return " ".join([summary, *dialogue_events]).strip()


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _content_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", str(value or "").casefold())
        if len(token) > 2 and token not in _CONTENT_STOPWORDS
    }


def _infer_quote_speaker(
    source: str,
    quote_start: int,
    *,
    context_start: int = 0,
) -> tuple[str, str]:
    """Infer one quote's speaker without reading through an earlier quote.

    Adjacent quoted turns often have only a short action between them. Looking
    back a fixed number of characters let the preceding speech verb and its
    entire quotation become the next line's delivery direction. Bound the
    search at the previous dialogue span so each quote owns only its local cue.
    """

    lower_bound = max(0, min(int(context_start or 0), quote_start))
    prefix = source[max(lower_bound, quote_start - 220):quote_start]
    verbs = list(_SPEECH_VERB.finditer(prefix))
    if not verbs:
        return "Speaker", "speaks naturally"
    verb = verbs[-1]
    names = list(_PROPER_NAME.finditer(prefix[:verb.start()]))
    speaker = names[-1].group(0) if names else "Speaker"
    post_modifier = sanitize_h3_prompt_text(prefix[verb.end():]).strip(" ,;:-")
    pre_modifier = ""
    if names:
        candidate = sanitize_h3_prompt_text(
            prefix[names[-1].end():verb.start()]
        ).strip(" ,;:-")
        if re.fullmatch(
            r"(?:(?:very|visibly)\s+)?(?:animatedly|breathlessly|calmly|"
            r"eagerly|energetically|enthusiastically|excitedly|frantically|"
            r"nervously|passionately|quietly|softly|urgently)",
            candidate,
            flags=re.IGNORECASE,
        ):
            pre_modifier = candidate
    vocal_verb = sanitize_h3_prompt_text(verb.group(0)).strip()
    expressive_verb = bool(re.match(
        r"^(?:mumbl|murmur|mutter|muffl)",
        vocal_verb,
        flags=re.IGNORECASE,
    ))
    delivery = (
        " ".join(part for part in (vocal_verb, post_modifier) if part)
        if expressive_verb else post_modifier
    )
    if pre_modifier:
        delivery = " ".join(
            part for part in ("speaks", pre_modifier, post_modifier) if part
        )
    # ``says to Yoda`` identifies the listener, not a vocal performance.
    # Passing it through produced awkward H3 instructions such as "speaks
    # with to Yoda delivery" and competed with the actual dialogue contract.
    if re.fullmatch(
        r"(?:to|at)\s+[A-Z][A-Za-z0-9_'â€™-]*(?:\s+[A-Z][A-Za-z0-9_'â€™-]*){0,3}",
        delivery,
        flags=re.IGNORECASE,
    ):
        delivery = ""
    if not delivery:
        delivery = "speaks naturally"
    elif len(delivery) > 100:
        delivery = delivery[-100:].lstrip(" ,;:-")
    return speaker, delivery


def _screenplay_dialogue_spans(source: str) -> list[dict[str, Any]]:
    """Return unambiguous ``CHARACTER: line`` screenplay rows.

    A colon is also used by camera notes and Context-IR, so only a short
    person-like label at the start of its own line is accepted.  The body is
    preserved verbatim except for optional wrapping quotation marks/Markdown.
    """

    spans: list[dict[str, Any]] = []
    for match in _SCREENPLAY_DIALOGUE_RE.finditer(source):
        speaker = sanitize_h3_prompt_text(match.group("speaker")).strip(" ,;:.-")
        speaker_key = _normalize_key(speaker)
        if (
            not speaker_key
            or speaker_key in _SCREENPLAY_NON_SPEAKER_LABELS
            or re.fullmatch(r"(?:scene|shot|subject|picture|video|audio|s)\s*\d+", speaker_key)
        ):
            continue
        text = str(match.group("text") or "").strip()
        text = re.sub(r"\*\*\s*$", "", text).strip()
        if len(text) >= 2 and (text[0], text[-1]) in {
            ('"', '"'), ("\u201c", "\u201d"), ("'", "'"), ("\u2018", "\u2019"),
        }:
            text = text[1:-1].strip()
        text = sanitize_h3_prompt_text(text)
        if not text or _PLACEHOLDER_DIALOGUE.fullmatch(text):
            continue
        raw_delivery = sanitize_h3_prompt_text(match.group("delivery")).strip(" ,;:.-")
        off_camera = bool(re.search(
            r"\b(?:v\.?\s*o\.?|o\.?\s*s\.?|off[- ]?camera|off[- ]?screen|voice[- ]?over)\b",
            raw_delivery,
            flags=re.IGNORECASE,
        ))
        cleaned_delivery = re.sub(
            r"\b(?:v\.?\s*o\.?|o\.?\s*s\.?|off[- ]?camera|off[- ]?screen|voice[- ]?over)\b",
            "",
            raw_delivery,
            flags=re.IGNORECASE,
        ).strip(" ,;:.-")
        if cleaned_delivery:
            delivery = (
                f"speaks {cleaned_delivery}"
                if cleaned_delivery.casefold().endswith("ly")
                else f"speaks with {cleaned_delivery} delivery"
            )
        else:
            delivery = "speaks naturally"
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "text": text,
            "speaker": speaker,
            "language": "English",
            "delivery": delivery,
            "off_camera": off_camera,
            "explicit_tag": False,
            "source_form": "screenplay",
        })
    return spans


def _without_locked_dialogue(
    source: str,
    locked_dialogue: list[dict[str, Any]],
    *,
    keep_screenplay_speaker_cues: bool,
) -> str:
    """Remove spoken words while preserving chronological screenplay cues."""

    cleaned = source
    for item in reversed(locked_dialogue):
        start = int(item.get("source_offset") or 0)
        end = int(item.get("source_end") or start)
        replacement = " . "
        if keep_screenplay_speaker_cues and item.get("source_form") == "screenplay":
            speaker = sanitize_h3_prompt_text(item.get("speaker")) or "Speaker"
            replacement = f"\n{speaker} speaks.\n"
        cleaned = cleaned[:start] + replacement + cleaned[end:]
    return cleaned


def extract_locked_dialogue(prompt: str) -> list[dict[str, Any]]:
    """Extract tagged, quoted, or screenplay-form dialogue before rewriting."""

    source = normalize_h3_dialogue_tags(prompt)
    tag_pattern = re.compile(
        r"<d>\s*(?:\[([^\]\r\n]+)\])?\s*((?:(?!<d>).)*?)\s*</d>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    quote_pattern = re.compile(r'"([^"\r\n]{1,600})"|“([^”\r\n]{1,600})”')
    spans: list[dict[str, Any]] = []
    occupied_ranges: list[tuple[int, int]] = []
    for match in tag_pattern.finditer(source):
        text = sanitize_h3_prompt_text(match.group(2) or "").strip()
        if not text or _PLACEHOLDER_DIALOGUE.fullmatch(text):
            continue
        occupied_ranges.append((match.start(), match.end()))
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "text": text,
            "language": sanitize_h3_prompt_text(match.group(1) or "English"),
            "explicit_tag": True,
            "source_form": "tagged",
        })
    for item in _screenplay_dialogue_spans(source):
        if any(start <= int(item["start"]) < end for start, end in occupied_ranges):
            continue
        occupied_ranges.append((int(item["start"]), int(item["end"])))
        spans.append(item)
    for match in quote_pattern.finditer(source):
        if any(start <= match.start() < end for start, end in occupied_ranges):
            continue
        text = sanitize_h3_prompt_text(match.group(1) or match.group(2) or "").strip()
        if not text or _PLACEHOLDER_DIALOGUE.fullmatch(text):
            continue
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "text": text,
            "language": "English",
            "explicit_tag": False,
            "source_form": "quoted",
        })
    spans.sort(key=lambda item: int(item["start"]))

    locked: list[dict[str, Any]] = []
    for span_index, span in enumerate(spans):
        start = int(span["start"])
        end = int(span["end"])
        text = str(span["text"])
        context_start = (
            int(spans[span_index - 1]["end"])
            if span_index else 0
        )
        screenplay = span.get("source_form") == "screenplay"
        speaker, delivery = (
            (
                sanitize_h3_prompt_text(span.get("speaker")) or "Speaker",
                sanitize_h3_prompt_text(span.get("delivery")) or "speaks naturally",
            )
            if screenplay else _infer_quote_speaker(
                source,
                start,
                context_start=context_start,
            )
        )
        # A quoted title should not silently become dialogue. Speech cues,
        # screenplay-style ``Name:`` labels, and a quote-only prompt are the
        # three unambiguous forms accepted here. An explicit <d> block is
        # already an unambiguous speech declaration.
        nearby = source[max(context_start, start - 120):start]
        label = re.search(
            r"([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})\s*:\s*$",
            nearby,
        )
        outside_quote = (source[:start] + source[end:]).strip(" \t\r\n.,;:!?-")
        if (
            not span["explicit_tag"]
            and not screenplay
            and not _SPEECH_VERB.search(nearby)
            and not label
            and outside_quote
        ):
            continue
        if label and not screenplay:
            speaker = label.group(1)
        speech_context = source[max(context_start, start - 240):start]
        off_camera = bool(span.get("off_camera")) or bool(re.search(
            r"\b(?:off[- ]camera|offscreen|off[- ]screen|pov\s+(?:off[- ]camera\s+)?voice)\b",
            speech_context,
            flags=re.IGNORECASE,
        ))
        locked.append({
            "dialogue_id": f"D{len(locked) + 1}",
            "speaker": sanitize_h3_prompt_text(speaker) or "Speaker",
            "language": str(span["language"] or "English"),
            "delivery": delivery,
            "text": text,
            "off_camera": off_camera,
            "source_offset": start,
            "source_end": end,
            "source_form": str(span.get("source_form") or "quoted"),
        })
    return locked


def _story_fragments(prompt: str) -> list[str]:
    # A user-authored line break is often the only boundary between two
    # actions in Studio's prompt box. Preserve it as sentence punctuation
    # before the general sanitizer collapses whitespace.
    raw_source = normalize_h3_dialogue_tags(prompt)
    locked_dialogue = extract_locked_dialogue(raw_source)
    has_screenplay_dialogue = any(
        item.get("source_form") == "screenplay" for item in locked_dialogue
    )
    raw_source = _without_locked_dialogue(
        raw_source,
        locked_dialogue,
        keep_screenplay_speaker_cues=True,
    )
    source = sanitize_h3_prompt_text(
        re.sub(
            r"(?:\r?\n)+",
            ". ",
            raw_source,
        )
    )
    pieces = re.split(
        r"(?<=[.!?])\s+|\s*;\s*|"
        r"\s+(?:(?:and\s+)?then(?:\s+then)*|after\s+that|next)\s+|"
        r",\s+(?:but|and)\s+|"
        r"\s+(?=(?:hard\s+cut|smash\s+cut|match\s+cut|cut\s+to|whip\s+pan)\b)",
        source,
        flags=re.IGNORECASE,
    )
    fragments: list[str] = []
    carried_subject = ""
    carried_motion = ""
    for piece in pieces:
        # Split coordinated physical actions without splitting compound names
        # such as "Hermione and Ron". This turns a long prose sentence into
        # usable immutable beats: laugh -> mount -> plummet -> canyon ->
        # waterfall -> cave, rather than one unfilmable mega-event.
        action_split = re.compile(
            rf"\s+(?:and|while|as)\s+(?=(?:(?:they|he|she|it|the\s+viewer|"
            rf"[A-Z][A-Za-z0-9_'’-]*)\s+)?(?:{_ACTION_VERBS})(?:s|es|ed|ing)?\b)|"
            r",\s+(?:and\s+)?(?=(?:through|between|into|out\s+of|over|under|past)\b)",
            flags=re.IGNORECASE,
        )
        subject_match = re.match(
            r"\s*(?:(?:and\s+)?then\s+)?(?P<subject>they|he|she|it|the\s+viewer)\b",
            piece,
            flags=re.IGNORECASE,
        )
        if not subject_match:
            subject_match = re.match(
                r"\s*(?:(?:and\s+)?then\s+)?(?P<subject>"
                r"[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})\b",
                piece,
            )
        shared_subject = sanitize_h3_prompt_text(
            subject_match.group("subject") if subject_match else ""
        )
        explicit_motion = (
            "flying" if re.search(r"\bfl(?:y|ies|ew|ying)\b", piece, re.IGNORECASE)
            else "falling" if re.search(r"\bfall(?:s|ing|en)?\b|\bplummet", piece, re.IGNORECASE)
            else ""
        )
        piece_needs_carried_subject = bool(re.match(
            rf"\s*(?:(?:and\s+)?then\s+)?(?:{_ACTION_VERBS})(?:s|es|ed|ing)?\b|"
            r"\s*(?:through|between|into|out\s+of|over|under|past)\b",
            piece,
            flags=re.IGNORECASE,
        ) or _SPEECH_VERB.match(piece))
        if not shared_subject and carried_subject and piece_needs_carried_subject:
            shared_subject = carried_subject
        if shared_subject:
            carried_subject = shared_subject
        shared_motion = explicit_motion or carried_motion or "moving"
        if explicit_motion:
            carried_motion = explicit_motion
        for subpiece in action_split.split(piece):
            value = re.sub(
                r"^(?:(?:and\s+)?then(?:\s+then)*|after\s+that|next)\s+",
                "",
                subpiece.strip(" ,;:-.!?"),
                flags=re.IGNORECASE,
            )
            if not value:
                continue
            if shared_subject and not re.match(
                r"^(?:they|he|she|it|the\s+viewer|"
                r"[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})\b",
                value,
            ):
                if re.match(
                    rf"^(?:{_ACTION_VERBS})(?:s|es|ed|ing)?\b",
                    value,
                    flags=re.IGNORECASE,
                ):
                    value = (
                        f"{shared_subject} "
                        f"{'keep' if shared_subject.casefold() == 'they' else 'keeps'} {value}"
                        if re.match(r"^[A-Za-z]+ing\b", value, flags=re.IGNORECASE)
                        else f"{shared_subject} {value}"
                    )
                elif re.match(
                    r"^(?:through|between|into|out\s+of|over|under|past)\b",
                    value,
                    flags=re.IGNORECASE,
                ):
                    keep = "keep" if shared_subject.casefold() == "they" else "keeps"
                    value = f"{shared_subject} {keep} {shared_motion} {value}"
                elif _SPEECH_VERB.match(value):
                    # ``Dwight then mutters ...`` is split at ``then`` by the
                    # chronological tokenizer. Carry the named subject into
                    # the speech cue so its locked quote remains anchorable.
                    value = f"{shared_subject} {value}"
            if re.fullmatch(
                r"(?:and\s+)?then|after\s+that|next",
                value,
                flags=re.IGNORECASE,
            ):
                continue
            if (
                _is_style_only_fragment(value)
                or _is_persistent_camera_directive(value)
                or _is_persistent_audio_directive(value)
                or _is_subject_only_fragment(value)
                or _is_creation_directive(value)
                or (
                    has_screenplay_dialogue
                    and _is_screenplay_performance_directive(value)
                )
            ):
                continue
            if re.fullmatch(
                r"(?:[A-Za-z]+verse|[A-Za-z]+)\s+style|"
                r"high[- ]speed\s+(?:action\s+)?(?:movie\s+)?(?:dynamic\s+)?"
                r"(?:superhero\s+)?(?:fight\s+)?scenes?",
                value,
                flags=re.IGNORECASE,
            ):
                continue
            fragments.append(value)

    # The action splitter intentionally separates long physical chains, but a
    # POV identity followed by its opening pose is one setup fact, not two
    # independent story events.  Keeping these together reduces artificial
    # bookkeeping without weakening the later action-by-action fidelity lock.
    compacted: list[str] = []
    for value in fragments:
        if (
            compacted
            and re.fullmatch(
                r"POV\s*:\s*The\s+viewer\s+is\s+.+",
                compacted[-1],
                flags=re.IGNORECASE,
            )
            and re.match(
                r"^(?:he|she|they|the\s+viewer)\s+(?:stands?|sits?|lies?|waits?)\b",
                value,
                flags=re.IGNORECASE,
            )
        ):
            compacted[-1] = f"{compacted[-1]} as {value}"
            continue
        if (
            compacted
            and _SPEECH_VERB.search(compacted[-1])
            and re.fullmatch(
                r"(?:(?:as|while)\s+)?(?:he|she|they)\s+"
                r"(?:introduce|gesture|motion|indicate|"
                r"point|present|nod|wave)(?:s|d|ed|ing)?\b.{0,80}",
                value,
                flags=re.IGNORECASE,
            )
        ):
            # A clause such as ``as he introduces them`` is performance for
            # the immediately preceding line, not a new story event or a
            # reason for H3 to improvise another utterance.
            dependent = re.sub(
                r"^(?:as|while)\s+",
                "",
                value,
                flags=re.IGNORECASE,
            )
            compacted[-1] = f"{compacted[-1]} while {dependent}"
            continue
        compacted.append(value)
    if has_screenplay_dialogue:
        compacted = _collapse_duplicate_screenplay_entrances(compacted)
    return compacted or ["Establish and carry out the requested scene"]


def extract_source_events(prompt: str) -> list[dict[str, str]]:
    """Create immutable source-event IDs used to prove coverage exactly once."""

    return [
        {"event_id": f"E{index + 1}", "text": fragment}
        for index, fragment in enumerate(_story_fragments(prompt))
    ]


def _filmable_source_event(value: Any) -> str:
    """Turn a dialogue cue into visible direction without duplicating its words.

    Quoted dialogue is removed before source-event extraction and restored from
    the locked dialogue catalog later.  That can leave bookkeeping fragments
    such as ``Hermione says``.  They are useful for chronological ownership,
    but are poor H3 action prose and can encourage the model to improvise a
    second line.  Keep mixed action-and-speech events intact; rewrite only a
    cue whose sole event is delivering the already-locked line.
    """

    text = sanitize_h3_prompt_text(value).strip(" \t\r\n-.,;:!?")
    pov_match = re.fullmatch(
        r"POV\s*:\s*The\s+viewer\s+is\s+(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if pov_match:
        identity = sanitize_h3_prompt_text(pov_match.group(1))
        return f"The camera is locked to {identity}'s first-person viewpoint"
    if re.match(r"^(?:he|she|they|it)\b", text, flags=re.IGNORECASE):
        text = text[:1].upper() + text[1:]
    speech = _SPEECH_VERB.search(text)
    if not text or not speech:
        return text

    before = text[:speech.start()].strip(" \t\r\n-.,;:!?")
    non_speech_actions = re.compile(
        r"\b(?:approach|arrive|attack|board|break|climb|cross|descend|dive|"
                r"drop|enter|exit|fall|fight|fly|grab|hold|jump|laugh|launch|leap|"
                r"mount|move|nod|pan|plummet|race|reach|ride|run|save|smash|sprint|stand|"
                r"step|take|turn|walk|wave|breathe)(?:s|es|ed|ing)?\b",
        flags=re.IGNORECASE,
    )
    if non_speech_actions.search(before):
        visible_action = re.sub(
            r"\b(?:and|while)\s*$",
            "",
            before,
            flags=re.IGNORECASE,
        ).strip(" \t\r\n-.,;:!?")
        # ``Camera pans to Dwight, and Dwight says ...`` otherwise leaves the
        # dangling phrase ``and Dwight`` in visual prose. Remove that cue and
        # retain a dependent physical performance after the speech verb.
        visible_action = re.sub(
            r"(?:,\s*)?\band\s+(?:the\s+)?"
            r"[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3}$",
            "",
            visible_action,
        ).strip(" \t\r\n-.,;:!?")
        after_speech = text[speech.end():]
        dependent = re.search(
            r"\b(?:while|as)\s+((?:he|she|they)\b.{1,100})$",
            after_speech,
            flags=re.IGNORECASE,
        )
        if dependent:
            visible_action = (
                f"{visible_action} while {sanitize_h3_prompt_text(dependent.group(1))}"
            )
        return visible_action

    off_camera = bool(re.search(
        r"\b(?:off[- ]camera|offscreen|off[- ]screen|pov\s+(?:off[- ]camera\s+)?voice)\b",
        text,
        flags=re.IGNORECASE,
    ))
    if off_camera:
        speaker_match = re.search(
            r"\bvoice\s+of\s+(.+?)\s+(?:says?|asks?|replies?|responds?|"
            r"whispers?|shouts?|yells?|declares?|states?|tells?|calls?\s+out)\b",
            text,
            flags=re.IGNORECASE,
        )
        speaker = sanitize_h3_prompt_text(
            speaker_match.group(1) if speaker_match else "the viewpoint character"
        )
        return (
            f"{speaker}'s unseen first-person voice delivers the assigned dialogue "
            "line off-camera while the POV remains locked"
        )

    speaker = re.sub(
        r"^(?:then\s+)?(?:the\s+)?",
        "",
        before,
        flags=re.IGNORECASE,
    ).strip()
    speaker = re.sub(
        r"\s+(?:(?:very|visibly)\s+)?(?:animatedly|breathlessly|calmly|"
        r"eagerly|energetically|enthusiastically|excitedly|frantically|"
        r"nervously|passionately|quietly|softly|urgently)\s*$",
        "",
        speaker,
        flags=re.IGNORECASE,
    ).strip()
    if not speaker or len(speaker.split()) > 8:
        return text
    delivery = text[speech.end():].strip(" \t\r\n-.,;:!?")
    delivery_suffix = f" {delivery}" if delivery else ""
    return (
        f"{speaker} visibly delivers the assigned dialogue line{delivery_suffix}"
    )


def _expected_dialogue_events(
    prompt: str,
    locked_dialogue: list[dict[str, Any]],
) -> dict[str, str]:
    events = extract_source_events(prompt)
    expected: dict[str, str] = {}
    cursor = 0
    for dialogue in locked_dialogue:
        speaker_tokens = _content_tokens(dialogue.get("speaker"))
        match_index: int | None = None
        for index in range(cursor, len(events)):
            text = events[index]["text"]
            if _SPEECH_VERB.search(text) and (
                not speaker_tokens or speaker_tokens & _content_tokens(text)
            ):
                match_index = index
                break
        if match_index is None:
            continue
        expected[str(dialogue.get("dialogue_id") or "").upper()] = events[match_index]["event_id"]
        cursor = match_index + 1
    return expected


_EXPLICIT_OPENING_DIALOGUE_DELAY_RE = re.compile(
    r"\b(?:much\s+later|later\s+(?:that|the\s+same)\s+(?:day|night)|"
    r"eventually|after\s+(?:a\s+long\s+time|waiting|watching|walking|"
    r"several|many|multiple|\d+\s*(?:seconds?|minutes?))|"
    r"for\s+(?:several|many|multiple|\d+)\s*(?:seconds?|minutes?)\s+before)\b",
    flags=re.IGNORECASE,
)


def _opening_h3_dialogue_id(
    prompt: str,
    locked_dialogue: list[dict[str, Any]],
    source_events: list[dict[str, str]] | None = None,
) -> str:
    """Return the first line that belongs in the opening H3 window.

    One compact setup/entrance sequence followed by a quoted speech cue
    describes an immediate performance, not a full silent establishing
    window. Longer action chains and explicit waits retain the semantic
    planner's freedom.
    """

    if not locked_dialogue:
        return ""
    events = source_events or extract_source_events(prompt)
    expected = _expected_dialogue_events(prompt, locked_dialogue)
    first = locked_dialogue[0]
    dialogue_id = str(first.get("dialogue_id") or "").upper()
    event_id = expected.get(dialogue_id)
    event_index = next(
        (
            index for index, event in enumerate(events)
            if event.get("event_id") == event_id
        ),
        None,
    )
    # ``walks into the room and walks up to Joey`` is deliberately split into
    # two immutable movement events, but remains one brief entrance. Three or
    # more preceding events represent a real visual sequence and are not
    # force-packed into the opening window.
    if event_index is None or event_index > 2:
        return ""
    prefix = str(prompt or "")[: int(first.get("source_offset") or 0)]
    if _EXPLICIT_OPENING_DIALOGUE_DELAY_RE.search(prefix):
        return ""
    return dialogue_id


def _ledger_schema(
    segment_count: int,
    *,
    source_event_count: int,
    locked_dialogue_count: int,
    allow_generated_dialogue: bool,
    minimum_generated_dialogue: int = 1,
) -> dict[str, Any]:
    """Schema for the semantic story schedule authored by the planning LLM.

    Maestro owns immutable event/dialogue catalogs and validates the result,
    but the planner owns the meaningful grouping and window allocation.  Beat
    IDs and exact prose are added locally after parsing so the model spends its
    capacity on directing the story rather than bookkeeping.
    """

    dialogue = {
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "language": {"type": "string"},
            "delivery": {"type": "string"},
            "text": {"type": "string"},
            "segment": {
                "type": "integer",
                "minimum": 1,
                "maximum": max(1, segment_count),
            },
        },
        "required": ["speaker", "language", "delivery", "text", "segment"],
        "additionalProperties": False,
    }
    generated_dialogue: dict[str, Any] = {
        "type": "array",
        "items": dialogue,
        "maxItems": max(0, segment_count * 2),
    }
    if allow_generated_dialogue:
        # Creative conversational briefs require an audible script. Keeping
        # this in the constrained schema prevents a capable LLM from spending
        # all of its output budget on visual beats and returning an empty
        # dialogue catalog that H3 can only fill with improvised gibberish.
        generated_dialogue["minItems"] = min(
            generated_dialogue["maxItems"],
            max(1, int(minimum_generated_dialogue)),
        )
    else:
        generated_dialogue["maxItems"] = 0
    beat = {
        "type": "object",
        "properties": {
            "segment": {
                "type": "integer",
                "minimum": 1,
                "maximum": max(1, segment_count),
            },
            "source_event_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": max(1, source_event_count),
            },
            "dialogue_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": max(0, locked_dialogue_count),
            },
            "description": {"type": "string"},
            "state_after": {"type": "string"},
            "sound_effects": {"type": "string"},
        },
        "required": [
            "segment", "source_event_ids", "dialogue_ids", "description",
            "state_after", "sound_effects",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "subject_continuity": {"type": "string"},
            "setting_continuity": {"type": "string"},
            "visual_continuity": {"type": "string"},
            "editing_style": {"type": "string"},
            "initial_state": {"type": "string"},
            "ambient_audio": {"type": "string"},
            "music": {"type": "string"},
            "required_final_outcome": {"type": "string"},
            "beats": {
                "type": "array",
                "items": beat,
                "minItems": max(1, segment_count),
                "maxItems": max(1, segment_count * 3),
            },
            "generated_dialogue": generated_dialogue,
        },
        "required": [
            "subject_continuity", "setting_continuity", "visual_continuity",
            "editing_style", "initial_state", "ambient_audio", "music",
            "required_final_outcome", "beats", "generated_dialogue",
        ],
        "additionalProperties": False,
    }


def _faithful_treatment_schema() -> dict[str, Any]:
    """Schema for the creative work that remains in faithful planning.

    A faithful request already gives Maestro an ordered source-event catalog
    and, when present, an exact dialogue catalog.  Asking a small local model
    to copy those internal IDs while also making directorial choices turns a
    straightforward writing task into brittle database transcription.  The
    model therefore supplies only the global cinematic treatment here;
    Maestro schedules the immutable story locally.
    """

    return {
        "type": "object",
        "properties": {
            "setting_continuity": {"type": "string"},
            "visual_continuity": {"type": "string"},
            "editing_style": {"type": "string"},
            "ambient_audio": {"type": "string"},
        },
        "required": [
            "setting_continuity",
            "visual_continuity",
            "editing_style",
            "ambient_audio",
        ],
        "additionalProperties": False,
    }


def _apply_faithful_treatment(
    canonical: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Overlay safe LLM direction without giving it story ownership."""

    ledger = deepcopy(canonical)
    if not isinstance(candidate, dict):
        return ledger
    cast_names = [
        sanitize_h3_prompt_text(name)
        for name in (
            (canonical.get("source_intent") or {}).get("cast_names") or []
        )
        if sanitize_h3_prompt_text(name)
    ]
    for field in (
        "setting_continuity",
        "visual_continuity",
        "editing_style",
    ):
        value = sanitize_h3_prompt_text(candidate.get(field))
        # A treatment field is a compact global direction, never a hidden
        # screenplay or per-character shot list. Reject malformed spillover
        # wholesale instead of truncating it into a misleading instruction.
        if (
            value
            and len(value.split()) <= 80
            and not re.search(r"(?:---|#{2,}|\*\*|\bsequence\s+progression\b)", value, re.IGNORECASE)
        ):
            ledger[field] = value
    ambient = sanitize_h3_nonverbal_audio(candidate.get("ambient_audio"))
    ambient_is_visual_plan = bool(re.search(
        r"(?:---|#{2,}|\*\*|\b(?:camera|close[- ]?up|framing|shot|"
        r"sequence\s+progression|interaction|establishment|dialogue|"
        r"center[- ]?frame|foreground)\b)",
        ambient,
        flags=re.IGNORECASE,
    ))
    ambient_names_story_cast = any(
        _speaker_name_present(ambient, name) for name in cast_names
    )
    if (
        ambient
        and len(ambient.split()) <= 55
        and not ambient_is_visual_plan
        and not ambient_names_story_cast
    ):
        ledger["ambient_audio"] = ambient
    elif ambient:
        print(
            "[MiniMax H3] Ignored malformed cinematic-treatment ambience "
            "that contained a shot plan or character blocking."
        )
    return ledger


def _dialogue_catalog(
    ledger: dict[str, Any],
    locked_dialogue: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    locked = [dict(item) for item in locked_dialogue]
    generated = [
        dict(item) for item in (ledger.get("generated_dialogue") or [])
        if isinstance(item, dict)
    ]
    catalog = locked + generated
    cast_names = list(
        (ledger.get("source_intent") or {}).get("cast_names") or []
    )
    speaker_ids: dict[str, str] = {}
    for item in catalog:
        speaker = _resolve_h3_cast_name(item.get("speaker"), cast_names)
        item["speaker"] = speaker
        key = speaker.casefold()
        speaker_ids.setdefault(key, f"S{len(speaker_ids) + 1}")
        item["speaker_id"] = speaker_ids[key]
    return catalog


def _camera_phase_beats(
    assigned_beats: list[dict[str, Any]],
    *,
    source_events: list[dict[str, str]],
    expected_dialogue_events: dict[str, str],
) -> list[dict[str, Any]]:
    """Expand coarse semantic beats into shot-local audiovisual phases.

    The story LLM is allowed to group several source events into one semantic
    beat.  That grouping is useful for window ownership, but it is too coarse
    for camera planning when the beat contains several speakers.  Previously
    all dialogue IDs from such a beat were attached to the first camera shot,
    while the LLM-authored later shots still described the corresponding
    characters speaking.  H3 then received the correct transcript beside the
    wrong face and could swap voices, repeat lines, or promote a portrait into
    target footage.

    Keep the LLM's segment allocation intact, but split a multi-event beat into
    ordered, event-local phases before camera planning.  Dialogue follows its
    immutable source-event anchor.  These phase IDs are internal to the local
    segment and deliberately do not alter the saved semantic ledger.
    """

    event_text = {
        str(item.get("event_id") or "").upper(): str(item.get("text") or "")
        for item in source_events
    }
    dialogue_event = {
        str(dialogue_id or "").upper(): str(event_id or "").upper()
        for dialogue_id, event_id in expected_dialogue_events.items()
    }
    phases: list[dict[str, Any]] = []
    for beat in assigned_beats:
        source_ids = [
            str(value or "").upper()
            for value in (beat.get("source_event_ids") or [])
            if str(value or "").upper() in event_text
        ]
        dialogue_ids = [
            str(value or "").upper()
            for value in (beat.get("dialogue_ids") or [])
            if str(value or "").strip()
        ]
        if len(source_ids) <= 1:
            phases.append(dict(beat))
            continue

        original_id = str(beat.get("beat_id") or f"B{len(phases) + 1}").upper()
        claimed_dialogue: set[str] = set()
        expanded: list[dict[str, Any]] = []
        for event_index, event_id in enumerate(source_ids, start=1):
            local_dialogue = [
                dialogue_id
                for dialogue_id in dialogue_ids
                if dialogue_event.get(dialogue_id) == event_id
            ]
            claimed_dialogue.update(local_dialogue)
            description = _filmable_source_event(event_text[event_id])
            phase = dict(beat)
            phase.update({
                "beat_id": f"{original_id}.{event_index}",
                "source_event_ids": [event_id],
                "dialogue_ids": local_dialogue,
                "description": description,
                "state_after": (
                    sanitize_h3_prompt_text(beat.get("state_after"))
                    if event_index == len(source_ids)
                    else f"The immediate visible state follows this event: {description}"
                ),
                "sound_effects": (
                    sanitize_h3_prompt_text(beat.get("sound_effects"))
                    if event_index == len(source_ids)
                    else "Natural synchronized effects for this visible event"
                ),
            })
            expanded.append(phase)

        # Generated dialogue and unusual source phrasing may not have a
        # recoverable E-id anchor. Preserve those lines once, in their original
        # order, on the final phase rather than dropping or duplicating them.
        unanchored = [
            dialogue_id
            for dialogue_id in dialogue_ids
            if dialogue_id not in claimed_dialogue
        ]
        if expanded and unanchored:
            expanded[-1]["dialogue_ids"].extend(unanchored)
        phases.extend(expanded or [dict(beat)])
    return phases


def _coalesce_camera_phases(
    phases: list[dict[str, Any]],
    *,
    target_count: int = 4,
) -> list[dict[str, Any]]:
    """Combine adjacent silent motion before compressing speaking coverage.

    A native H3 window commonly needs an entrance and an approach before a
    rapid dialogue exchange.  Those two continuous actions are naturally one
    moving shot.  Keeping them as separate internal phases can leave five
    phases for a four-shot camera plan, which previously forced two different
    speakers into one shot and later triggered a fallback.  Merge only
    adjacent action-only phases; dialogue turns remain discrete.
    """

    result = [deepcopy(item) for item in phases if isinstance(item, dict)]
    target = max(1, int(target_count))
    while len(result) > target:
        merge_index = next(
            (
                index
                for index in range(len(result) - 1)
                if not (result[index].get("dialogue_ids") or [])
                and not (result[index + 1].get("dialogue_ids") or [])
            ),
            None,
        )
        if merge_index is None:
            break
        left = result[merge_index]
        right = result[merge_index + 1]
        descriptions = [
            sanitize_h3_prompt_text(value)
            for value in (left.get("description"), right.get("description"))
            if sanitize_h3_prompt_text(value)
        ]
        effects = [
            sanitize_h3_prompt_text(value)
            for value in (left.get("sound_effects"), right.get("sound_effects"))
            if sanitize_h3_prompt_text(value)
            and sanitize_h3_prompt_text(value).casefold() not in {"n/a", "none"}
        ]
        merged = dict(left)
        merged.update({
            "beat_id": (
                f"{str(left.get('beat_id') or '').upper()}+"
                f"{str(right.get('beat_id') or '').upper()}"
            ).strip("+"),
            "source_event_ids": list(dict.fromkeys([
                str(value or "").upper()
                for item in (left, right)
                for value in (item.get("source_event_ids") or [])
                if str(value or "").strip()
            ])),
            "dialogue_ids": [],
            "description": ". Then ".join(descriptions),
            "state_after": sanitize_h3_prompt_text(
                right.get("state_after") or left.get("state_after")
            ),
            "sound_effects": "; ".join(dict.fromkeys(effects))
            or "Natural synchronized effects for the visible action",
        })
        result[merge_index:merge_index + 2] = [merged]
    return result


def _dialogue_word_count(value: Any) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", str(value or "")))


_DIALOGUE_BREAK_GLUE_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "just", "like", "nor", "of", "on", "or", "the", "to", "with",
}


def _safe_dialogue_word_break(
    source: str,
    matches: list[re.Match[str]],
    absolute_word_index: int,
) -> bool:
    """Reject boundaries that would visibly tear one spoken phrase apart."""

    if absolute_word_index <= 0 or absolute_word_index >= len(matches):
        return False
    left = matches[absolute_word_index - 1]
    right = matches[absolute_word_index]
    separator = source[left.end():right.start()]
    # A decimal/version number such as Qwen 3.8 has punctuation but no actual
    # inter-word pause. Likewise, keep a numeric suffix such as Sora 2 or
    # LTX 2.5 attached to the name it qualifies.
    if not any(character.isspace() for character in separator):
        return False
    left_word = left.group(0).casefold()
    right_word = right.group(0)
    if right_word[:1].isdigit():
        return False
    if left_word in _DIALOGUE_BREAK_GLUE_WORDS:
        return False
    return True


def _split_dialogue_text_for_capacities(
    text: str,
    capacities: list[int],
) -> list[str]:
    """Split one exact line at natural boundaries without changing its words."""

    source = sanitize_h3_prompt_text(text)
    matches = list(re.finditer(r"\b[\w'’-]+\b", source))
    if not matches:
        return [source] if source else []
    usable = [max(0, int(value)) for value in capacities]
    if sum(usable) < len(matches):
        raise H3DialogueTimingError(
            "MiniMax H3 exact dialogue exceeds the combined selected window timing. "
            "Increase total duration or add windows before generating."
        )

    fragments: list[str] = []
    word_index = 0
    char_index = 0
    for capacity_index, capacity in enumerate(usable):
        remaining_words = len(matches) - word_index
        if remaining_words <= 0:
            break
        if capacity <= 0:
            continue
        if remaining_words <= capacity:
            take = remaining_words
        else:
            future_capacity = sum(usable[capacity_index + 1:])
            minimum_take = max(1, remaining_words - future_capacity)
            maximum_take = min(capacity, remaining_words - 1)
            if maximum_take < minimum_take:
                continue
            sentence_boundaries: list[int] = []
            clause_boundaries: list[int] = []
            safe_word_boundaries: list[int] = []
            for candidate in range(minimum_take, maximum_take + 1):
                absolute_word_index = word_index + candidate
                end = matches[absolute_word_index - 1].end()
                next_start = (
                    matches[absolute_word_index].start()
                    if absolute_word_index < len(matches) else len(source)
                )
                punctuation = source[end:next_start]
                safe_break = _safe_dialogue_word_break(
                    source,
                    matches,
                    absolute_word_index,
                )
                if safe_break:
                    safe_word_boundaries.append(candidate)
                if safe_break and re.search(
                    r"[.!?](?:[\"'”’)]*)\s+$",
                    punctuation,
                ):
                    sentence_boundaries.append(candidate)
                elif safe_break and re.search(
                    r"[,;:](?:[\"'”’)]*)\s+$",
                    punctuation,
                ):
                    clause_boundaries.append(candidate)
            take = (
                sentence_boundaries[-1]
                if sentence_boundaries else
                clause_boundaries[-1]
                if clause_boundaries else
                safe_word_boundaries[-1]
                if safe_word_boundaries else
                maximum_take
            )

        word_end = matches[word_index + take - 1].end()
        next_word_start = (
            matches[word_index + take].start()
            if word_index + take < len(matches) else len(source)
        )
        # Keep punctuation attached to the phrase it closes, but not the
        # whitespace that separates it from the next fragment.
        boundary = word_end
        while boundary < next_word_start and not source[boundary].isspace():
            boundary += 1
        fragment = source[char_index:boundary].strip()
        if fragment:
            fragments.append(fragment)
        char_index = next_word_start
        word_index += take

    if word_index != len(matches):
        raise H3DialogueTimingError(
            "MiniMax H3 could not divide the exact dialogue safely across the "
            "selected windows. Increase total duration before generating."
        )
    if " ".join(" ".join(fragments).split()) != " ".join(source.split()):
        raise H3DialogueTimingError(
            "MiniMax H3 dialogue fragmentation changed the locked screenplay text; "
            "no generation was started."
        )
    return fragments


def _prepare_render_dialogue_schedule(
    beats: list[dict[str, Any]],
    dialogue_catalog: list[dict[str, Any]],
    *,
    segment_durations: list[float],
    source_events: list[dict[str, str]],
    expected_dialogue_events: dict[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    """Compile overlong exact dialogue into safe adjacent-window fragments.

    The semantic ledger keeps the user's original D-ids atomic so fidelity can
    be validated. Native H3 clips, however, have a finite speech budget. This
    render-only pass preserves the exact transcript while continuing a long
    turn across adjacent windows. It never asks the LLM to paraphrase, shorten,
    duplicate, or improvise a line.
    """

    durations = [max(0.1, float(value)) for value in segment_durations]
    budgets = [
        max(1, int(math.floor(value * _H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND)))
        for value in durations
    ]
    maximum_budgets = [
        max(1, int(math.floor(value * _H3_DIALOGUE_MAX_WORDS_PER_SECOND)))
        for value in durations
    ]
    segment_count = len(budgets)
    if not segment_count or not dialogue_catalog:
        return deepcopy(beats), deepcopy(dialogue_catalog), dict(expected_dialogue_events), []

    base_segments: dict[str, int] = {
        str(dialogue_id or "").upper(): int(beat.get("segment") or 0)
        for beat in beats
        for dialogue_id in (beat.get("dialogue_ids") or [])
    }
    event_text = {
        str(item.get("event_id") or "").upper(): str(item.get("text") or "")
        for item in source_events
    }
    event_order = {
        str(item.get("event_id") or "").upper(): index + 1
        for index, item in enumerate(source_events)
    }

    def is_pure_speech_event(dialogue_id: str) -> bool:
        event_id = str(expected_dialogue_events.get(dialogue_id) or "").upper()
        filmable = _filmable_source_event(event_text.get(event_id, ""))
        return bool(re.search(
            r"(?:visibly|voice)\s+delivers?\s+the\s+assigned\s+dialogue\s+line",
            filmable,
            flags=re.IGNORECASE,
        ))

    base_word_totals = Counter()
    for item in dialogue_catalog:
        dialogue_id = str(item.get("dialogue_id") or "").upper()
        base = base_segments.get(dialogue_id) or int(item.get("segment") or 1)
        base_word_totals[base] += _dialogue_word_count(item.get("text"))

    used = [0] * segment_count
    last_segment = 1
    allocations: dict[str, list[tuple[int, str]]] = {}
    total_words = sum(_dialogue_word_count(item.get("text")) for item in dialogue_catalog)
    if total_words > sum(budgets):
        raise H3DialogueTimingError(
            f"MiniMax H3 screenplay dialogue needs {total_words} spoken words, but "
            f"the selected windows safely fit {sum(budgets)}. Increase total duration "
            "or add windows before generating."
        )

    for item in dialogue_catalog:
        dialogue_id = str(item.get("dialogue_id") or "").upper()
        word_count = _dialogue_word_count(item.get("text"))
        if not dialogue_id or not word_count:
            continue
        base = max(1, min(
            segment_count,
            base_segments.get(dialogue_id) or int(item.get("segment") or 1),
        ))
        can_begin_previous = bool(
            base > 1
            and is_pure_speech_event(dialogue_id)
            and (
                word_count > budgets[base - 1]
                or base_word_totals[base] > budgets[base - 1]
            )
        )
        start = max(last_segment, base - 1 if can_begin_previous else base)
        available = [
            (segment, budgets[segment - 1] - used[segment - 1])
            for segment in range(start, segment_count + 1)
            if budgets[segment - 1] - used[segment - 1] > 0
        ]
        if not available or sum(capacity for _segment, capacity in available) < word_count:
            raise H3DialogueTimingError(
                f"MiniMax H3 cannot fit {item.get('speaker') or 'a speaker'}'s exact "
                "dialogue into the remaining selected windows. Increase total duration "
                "or add windows before generating."
            )

        # Prefer moving a short intact turn to the next available window over
        # creating a one- or two-word tail at the boundary. Long turns consume
        # each adjacent window in order and are split at punctuation when safe.
        selected: list[tuple[int, int]] = []
        first_segment, first_capacity = available[0]
        if word_count <= first_capacity:
            selected = [(first_segment, word_count)]
        elif available[1:] and word_count <= max(
            capacity for _segment, capacity in available[1:]
        ):
            target = next(
                (segment, capacity)
                for segment, capacity in available[1:]
                if capacity >= word_count
            )
            selected = [(target[0], word_count)]
        else:
            remaining = word_count
            for segment, capacity in available:
                if remaining <= 0:
                    break
                selected.append((segment, capacity))
                remaining -= capacity

        split_capacities = [capacity for _segment, capacity in selected]
        # Use a small amount of otherwise-unused local speech headroom only on
        # the final fragment. This lets a long exact turn break after a full
        # sentence instead of cutting a product/name phrase in half. Overall
        # sequence admission still uses the conservative 2.1 words/sec total.
        final_selected_segment = selected[-1][0]
        final_maximum_capacity = max(
            0,
            maximum_budgets[final_selected_segment - 1]
            - used[final_selected_segment - 1],
        )
        split_capacities[-1] = max(
            split_capacities[-1],
            final_maximum_capacity,
        )
        fragments = _split_dialogue_text_for_capacities(
            str(item.get("text") or ""),
            split_capacities,
        )
        if len(fragments) != len(selected):
            raise H3DialogueTimingError(
                "MiniMax H3 produced an incomplete exact-dialogue continuation plan; "
                "no generation was started."
            )
        allocations[dialogue_id] = [
            (segment, fragment)
            for (segment, _capacity), fragment in zip(selected, fragments)
        ]
        for (segment, _capacity), fragment in zip(selected, fragments):
            used[segment - 1] += _dialogue_word_count(fragment)
        last_segment = selected[-1][0]

    changed = any(
        len(parts) != 1
        or parts[0][0] != (
            base_segments.get(dialogue_id)
            or int(next(
                item.get("segment") or 1
                for item in dialogue_catalog
                if str(item.get("dialogue_id") or "").upper() == dialogue_id
            ))
        )
        for dialogue_id, parts in allocations.items()
    )
    if not changed:
        return deepcopy(beats), deepcopy(dialogue_catalog), dict(expected_dialogue_events), []

    phases_by_segment: dict[int, list[dict[str, Any]]] = {}
    phase_ordinal = 0
    for segment in range(1, segment_count + 1):
        assigned = [
            item for item in beats
            if isinstance(item, dict) and int(item.get("segment") or 0) == segment
        ]
        phases = _camera_phase_beats(
            assigned,
            source_events=source_events,
            expected_dialogue_events=expected_dialogue_events,
        )
        for phase in phases:
            phase_ordinal += 1
            phase["_render_order"] = min(
                [
                    event_order.get(str(event_id or "").upper(), 100000)
                    for event_id in (phase.get("source_event_ids") or [])
                ]
                or [100000 + phase_ordinal]
            ) * 100
        phases_by_segment[segment] = phases

    changed_ids = {
        dialogue_id for dialogue_id, parts in allocations.items()
        if len(parts) != 1 or parts[0][0] != base_segments.get(dialogue_id, parts[0][0])
    }
    relocated_events: set[str] = set()
    for dialogue_id in changed_ids:
        event_id = str(expected_dialogue_events.get(dialogue_id) or "").upper()
        relocate_event = bool(event_id and is_pure_speech_event(dialogue_id))
        if relocate_event:
            relocated_events.add(event_id)
        for phase in [item for values in phases_by_segment.values() for item in values]:
            phase["dialogue_ids"] = [
                str(value or "").upper()
                for value in (phase.get("dialogue_ids") or [])
                if str(value or "").upper() != dialogue_id
            ]
            if relocate_event:
                phase["source_event_ids"] = [
                    str(value or "").upper()
                    for value in (phase.get("source_event_ids") or [])
                    if str(value or "").upper() != event_id
                ]

    render_catalog: list[dict[str, Any]] = []
    render_expected = {
        dialogue_id: event_id
        for dialogue_id, event_id in expected_dialogue_events.items()
        if dialogue_id not in changed_ids
    }
    fragment_metadata: list[dict[str, Any]] = []
    for item in dialogue_catalog:
        dialogue_id = str(item.get("dialogue_id") or "").upper()
        parts = allocations.get(dialogue_id) or []
        if dialogue_id not in changed_ids:
            render_catalog.append(deepcopy(item))
            continue
        event_id = str(expected_dialogue_events.get(dialogue_id) or "").upper()
        fragment_ids: list[str] = []
        for fragment_index, (segment, fragment_text) in enumerate(parts, start=1):
            fragment_id = f"{dialogue_id}F{fragment_index}"
            fragment_ids.append(fragment_id)
            fragment = deepcopy(item)
            fragment.update({
                "dialogue_id": fragment_id,
                "source_dialogue_id": dialogue_id,
                "fragment_index": fragment_index,
                "fragment_count": len(parts),
                "text": fragment_text,
                "segment": segment,
            })
            if fragment_index > 1:
                fragment["delivery"] = (
                    "natural and continuous, without "
                    "restarting or repeating earlier words"
                )
            render_catalog.append(fragment)
            if fragment_index == 1 and event_id in relocated_events:
                render_expected[fragment_id] = event_id
            speaker = sanitize_h3_prompt_text(item.get("speaker")) or "The speaker"
            if len(parts) == 1:
                description = f"{speaker} visibly delivers the assigned dialogue line"
                state_after = f"the immediate visible state follows {speaker}'s completed line"
            elif fragment_index == 1:
                description = f"{speaker} visibly begins the assigned response"
                state_after = (
                    f"{speaker} is visibly mid-sentence in the final frame; the same "
                    "response continues without restarting"
                )
            elif fragment_index == len(parts):
                description = f"{speaker} visibly continues and completes the same response"
                state_after = f"the immediate visible state follows {speaker}'s completed response"
            else:
                description = f"{speaker} visibly continues the same uninterrupted response"
                state_after = (
                    f"{speaker} remains visibly mid-sentence in the final frame; the same "
                    "response continues without restarting"
                )
            source_ids = (
                [event_id]
                if fragment_index == 1 and event_id in relocated_events else []
            )
            order = (
                event_order.get(event_id, 100000) * 100
                + fragment_index
            )
            phases_by_segment[segment].append({
                "beat_id": f"RF{len(fragment_metadata) + fragment_index}",
                "segment": segment,
                "description": description,
                "source_event_ids": source_ids,
                "dialogue_ids": [fragment_id],
                "state_after": state_after,
                "sound_effects": "Natural synchronized effects for the visible performance",
                "_render_order": order,
            })
        fragment_metadata.append({
            "source_dialogue_id": dialogue_id,
            "speaker": sanitize_h3_prompt_text(item.get("speaker")) or "Speaker",
            "fragment_ids": fragment_ids,
            "segments": [segment for segment, _text in parts],
            "exact_text": sanitize_h3_prompt_text(item.get("text")),
        })

    # Moving an intact short response into the next window can put it on the
    # far side of a silent action that originally followed it.  For example,
    # if D7 moves from segment 4 to 5, an intervening "George sits down" beat
    # must move with that chronological boundary; otherwise flattening the
    # render phases produces E8, E10, E9 and the safety check correctly aborts.
    # Dialogue allocation is fixed by the speech budget above.  Re-home only
    # source-event phases around those fixed dialogue anchors, clamping silent
    # events between the preceding line's completion and the following line's
    # start.  This preserves every source event once and in order without
    # changing dialogue timing or text.
    event_dialogue_spans: dict[str, tuple[int, int]] = {}
    for dialogue_id, parts in allocations.items():
        event_id = str(expected_dialogue_events.get(dialogue_id) or "").upper()
        if not event_id or not parts:
            continue
        start = min(segment for segment, _text in parts)
        end = max(segment for segment, _text in parts)
        previous = event_dialogue_spans.get(event_id)
        event_dialogue_spans[event_id] = (
            min(previous[0], start) if previous else start,
            max(previous[1], end) if previous else end,
        )

    all_phases = [
        phase
        for segment_phases in phases_by_segment.values()
        for phase in segment_phases
    ]
    original_event_segments: dict[str, int] = {}
    for phase in all_phases:
        try:
            phase_segment = int(phase.get("segment") or 1)
        except (TypeError, ValueError):
            phase_segment = 1
        for event_id in phase.get("source_event_ids") or []:
            event_key = str(event_id or "").upper()
            if event_key:
                original_event_segments.setdefault(event_key, phase_segment)

    ordered_event_ids = [
        str(item.get("event_id") or "").upper()
        for item in source_events
        if str(item.get("event_id") or "").strip()
    ]
    next_dialogue_starts: list[int] = [segment_count] * len(ordered_event_ids)
    next_start = segment_count
    for event_index in range(len(ordered_event_ids) - 1, -1, -1):
        event_id = ordered_event_ids[event_index]
        if event_id in event_dialogue_spans:
            next_start = event_dialogue_spans[event_id][0]
        next_dialogue_starts[event_index] = next_start

    event_targets: dict[str, int] = {}
    preceding_completion = 1
    preceding_event_segment = 1
    for event_index, event_id in enumerate(ordered_event_ids):
        dialogue_span = event_dialogue_spans.get(event_id)
        if dialogue_span:
            target = dialogue_span[0]
            preceding_completion = max(preceding_completion, dialogue_span[1])
            preceding_event_segment = max(preceding_event_segment, dialogue_span[1])
        else:
            upper = max(preceding_completion, next_dialogue_starts[event_index])
            desired = original_event_segments.get(event_id, preceding_event_segment)
            target = max(
                preceding_completion,
                preceding_event_segment,
                min(upper, desired),
            )
            preceding_event_segment = target
        event_targets[event_id] = max(1, min(segment_count, target))

    rehomed_phases: dict[int, list[dict[str, Any]]] = {
        segment: [] for segment in range(1, segment_count + 1)
    }
    for phase in all_phases:
        source_ids = [
            str(event_id or "").upper()
            for event_id in (phase.get("source_event_ids") or [])
            if str(event_id or "").strip()
        ]
        try:
            phase_segment = int(phase.get("segment") or 1)
        except (TypeError, ValueError):
            phase_segment = 1
        if source_ids:
            phase_segment = max(
                event_targets.get(event_id, phase_segment)
                for event_id in source_ids
            )
            phase["segment"] = phase_segment
        phase_segment = max(1, min(segment_count, phase_segment))
        rehomed_phases[phase_segment].append(phase)
    phases_by_segment = rehomed_phases

    render_beats: list[dict[str, Any]] = []
    for segment in range(1, segment_count + 1):
        cleaned: list[dict[str, Any]] = []
        for phase in sorted(
            phases_by_segment[segment],
            key=lambda item: float(item.get("_render_order") or 0),
        ):
            source_ids = [
                str(value or "").upper()
                for value in (phase.get("source_event_ids") or [])
                if str(value or "").upper() in event_text
            ]
            dialogue_ids = [
                str(value or "").upper()
                for value in (phase.get("dialogue_ids") or [])
                if str(value or "").strip()
            ]
            if not source_ids and not dialogue_ids:
                continue
            phase["source_event_ids"] = source_ids
            phase["dialogue_ids"] = dialogue_ids
            if source_ids:
                phase["description"] = ". Then ".join(
                    _filmable_source_event(event_text[event_id])
                    for event_id in source_ids
                )
            phase.pop("_render_order", None)
            cleaned.append(phase)
        render_beats.extend(cleaned)

    for index, beat in enumerate(render_beats, start=1):
        beat["beat_id"] = f"B{index}"

    rendered_event_ids = [
        str(event_id or "").upper()
        for beat in render_beats
        for event_id in (beat.get("source_event_ids") or [])
    ]
    expected_event_ids = [
        str(item.get("event_id") or "").upper()
        for item in source_events
    ]
    if rendered_event_ids != expected_event_ids:
        raise H3DialogueTimingError(
            "MiniMax H3 dialogue fragmentation changed the screenplay event order; "
            "no generation was started."
        )

    rendered_words = Counter()
    for item in render_catalog:
        rendered_words[int(item.get("segment") or base_segments.get(
            str(item.get("dialogue_id") or "").upper(), 1
        ))] += _dialogue_word_count(item.get("text"))
    for segment, maximum_budget in enumerate(maximum_budgets, start=1):
        if rendered_words[segment] > maximum_budget:
            raise H3DialogueTimingError(
                f"MiniMax H3 exact dialogue still exceeds window {segment}'s safe "
                "natural-speech ceiling after fragmentation "
                f"({rendered_words[segment]}/{maximum_budget} words). "
                "Increase total duration before generating."
            )

    return render_beats, render_catalog, render_expected, fragment_metadata


def ledger_violations(
    prompt: str,
    ledger: dict[str, Any] | None,
    *,
    segment_count: int,
    locked_dialogue: list[dict[str, Any]],
    expect_dialogue: bool,
    allow_generated_dialogue: bool = False,
    require_dialogue_per_segment: bool = False,
    segment_durations: list[float] | None = None,
) -> list[str]:
    """Validate event ownership before expensive per-segment expansion."""

    if not isinstance(ledger, dict):
        return ["invalid story ledger"]
    violations: list[str] = []
    beats = [item for item in (ledger.get("beats") or []) if isinstance(item, dict)]
    if len(beats) < segment_count:
        violations.append(f"returned {len(beats)} beats for {segment_count} segments")
    ids = [str(item.get("beat_id") or "").upper() for item in beats]
    expected_ids = [f"B{index + 1}" for index in range(len(beats))]
    if ids != expected_ids:
        violations.append("beat IDs are not unique and sequential")
    segments: list[int] = []
    for item in beats:
        try:
            segment = int(item.get("segment"))
        except (TypeError, ValueError):
            segment = 0
        segments.append(segment)
        if not str(item.get("description") or "").strip():
            violations.append(f"{item.get('beat_id') or 'a beat'} has no visible event")
    if any(value < 1 or value > segment_count for value in segments):
        violations.append("one or more beats target an invalid segment")
    if segments != sorted(segments):
        violations.append("story beats are assigned out of order")
    missing_segments = sorted(set(range(1, segment_count + 1)) - set(segments))
    if missing_segments:
        violations.append("segments without a story beat: " + ", ".join(map(str, missing_segments)))
    if beats and segments[-1:] != [segment_count]:
        violations.append("the final story outcome is not assigned to the final segment")
    if any(count > 3 for count in Counter(segments).values()):
        violations.append("a segment has more than three story beats")
    normalized_descriptions = []
    for item in beats:
        description = _normalize_key(item.get("description"))
        if not description:
            continue
        # Screenplay rows intentionally become generic visual cues after the
        # exact spoken text is moved into the locked dialogue catalog. Several
        # distinct GEORGE: rows can therefore all canonicalize to "George
        # visibly delivers the assigned dialogue line." Include their source
        # ownership in the duplicate signature so those legitimate turns are
        # not mistaken for a repeated story event. Ordinary action prose stays
        # strict and still catches genuinely duplicated beats.
        if (
            item.get("dialogue_ids")
            and item.get("source_event_ids")
            and "visibly delivers the assigned dialogue line" in description
        ):
            description += " source " + " ".join(
                str(event_id or "").upper()
                for event_id in (item.get("source_event_ids") or [])
            )
        normalized_descriptions.append(description)
    if len(normalized_descriptions) != len(set(normalized_descriptions)):
        violations.append("a story event is duplicated across beats")

    source_event_ids = [item["event_id"] for item in extract_source_events(prompt)]
    referenced_event_ids = [
        str(event_id or "").upper()
        for beat in beats
        for event_id in (beat.get("source_event_ids") or [])
    ]
    if Counter(referenced_event_ids) != Counter(source_event_ids):
        violations.append("source event IDs are missing, foreign, or repeated")
    if referenced_event_ids and referenced_event_ids != source_event_ids:
        violations.append("source event order differs from the user's story order")
    if len(source_event_ids) > 1:
        final_event_segments = [
            int(beat.get("segment") or 0)
            for beat in beats
            if source_event_ids[-1] in [
                str(event_id or "").upper()
                for event_id in (beat.get("source_event_ids") or [])
            ]
        ]
        if final_event_segments != [segment_count]:
            violations.append("the final source event is not assigned to the final segment")

    generated = [
        item for item in (ledger.get("generated_dialogue") or [])
        if isinstance(item, dict)
    ]
    if locked_dialogue and generated and not allow_generated_dialogue:
        violations.append("invented extra dialogue despite locked user dialogue")
    locked_ids = [item["dialogue_id"] for item in locked_dialogue]
    generated_ids = [str(item.get("dialogue_id") or "").upper() for item in generated]
    all_ids = locked_ids + generated_ids
    if len(all_ids) != len(set(all_ids)):
        violations.append("dialogue IDs are duplicated")
    if generated_ids:
        expected_generated = [
            f"D{index}" for index in range(len(locked_ids) + 1, len(all_ids) + 1)
        ]
        if generated_ids != expected_generated:
            violations.append("generated dialogue IDs are not sequential")
    for item in generated:
        text = sanitize_h3_prompt_text(item.get("text"))
        if not text or _PLACEHOLDER_DIALOGUE.fullmatch(text):
            violations.append(f"{item.get('dialogue_id') or 'dialogue'} is empty or a placeholder")
    referenced_ids = [
        str(dialogue_id or "").upper()
        for beat in beats
        for dialogue_id in (beat.get("dialogue_ids") or [])
    ]
    if Counter(referenced_ids) != Counter(all_ids):
        violations.append("dialogue IDs are missing, duplicated, or assigned to multiple beats")
    locked_reference_order = [item for item in referenced_ids if item in set(locked_ids)]
    generated_reference_order = [item for item in referenced_ids if item in set(generated_ids)]
    if locked_reference_order != locked_ids:
        violations.append("locked dialogue order differs from the user's story order")
    if generated_reference_order != generated_ids:
        violations.append("generated dialogue order differs from its authored story order")
    expected_dialogue_events = _expected_dialogue_events(prompt, locked_dialogue)
    dialogue_beat_events = {
        str(dialogue_id or "").upper(): {
            str(event_id or "").upper()
            for event_id in (beat.get("source_event_ids") or [])
        }
        for beat in beats
        for dialogue_id in (beat.get("dialogue_ids") or [])
    }
    for dialogue_id, event_id in expected_dialogue_events.items():
        if event_id not in dialogue_beat_events.get(dialogue_id, set()):
            violations.append(f"{dialogue_id} moved away from its source speech event {event_id}")
    source_events = extract_source_events(prompt)
    opening_dialogue_id = _opening_h3_dialogue_id(
        prompt,
        locked_dialogue,
        source_events,
    )
    if opening_dialogue_id:
        opening_segments = [
            int(beat.get("segment") or 0)
            for beat in beats
            if opening_dialogue_id in {
                str(dialogue_id or "").upper()
                for dialogue_id in (beat.get("dialogue_ids") or [])
            }
        ]
        if opening_segments != [1]:
            violations.append(
                f"first requested dialogue {opening_dialogue_id} is delayed beyond segment 1"
            )
    if expect_dialogue and not all_ids:
        violations.append("the requested character interaction contains no dialogue")
    if require_dialogue_per_segment and all_ids:
        dialogue_segments = {
            int(beat.get("segment") or 0)
            for beat in beats
            if any(
                str(dialogue_id or "").upper() in set(all_ids)
                for dialogue_id in (beat.get("dialogue_ids") or [])
            )
        }
        silent_segments = sorted(
            set(range(1, segment_count + 1)) - dialogue_segments
        )
        if silent_segments:
            violations.append(
                "conversation-first Creative plan leaves segment(s) without "
                "authored dialogue: " + ", ".join(map(str, silent_segments))
            )
    if all_ids and segment_durations:
        dialogue_words = {
            str(item.get("dialogue_id") or "").upper(): _dialogue_word_count(
                item.get("text")
            )
            for item in [*locked_dialogue, *generated]
        }
        beat_segment = {
            str(dialogue_id or "").upper(): int(beat.get("segment") or 0)
            for beat in beats
            for dialogue_id in (beat.get("dialogue_ids") or [])
        }
        budgets = [
            max(1, int(math.floor(max(0.0, float(duration)) * 2.1)))
            for duration in segment_durations
        ]
        total_dialogue_words = sum(dialogue_words.values())
        if total_dialogue_words > sum(budgets):
            violations.append(
                f"screenplay dialogue uses {total_dialogue_words} words; the selected "
                f"windows safely fit {sum(budgets)}"
            )
        maximum_window_budget = max(budgets or [1])
        for segment_number, budget in enumerate(budgets, start=1):
            local_ids = [
                dialogue_id for dialogue_id in all_ids
                if beat_segment.get(dialogue_id) == segment_number
            ]
            word_count = sum(
                count for dialogue_id, count in dialogue_words.items()
                if beat_segment.get(dialogue_id) == segment_number
            )
            if word_count > budget:
                # One exact user-authored turn may legitimately continue over
                # adjacent native H3 windows. The render compiler fragments it
                # later without changing a word. Keep rejecting a pile-up of
                # ordinary short turns so the semantic planner can regroup
                # their whole speech events first.
                fragmentable = bool(
                    total_dialogue_words <= sum(budgets)
                    and any(
                        dialogue_words.get(dialogue_id, 0) > maximum_window_budget
                        for dialogue_id in local_ids
                    )
                )
                if fragmentable:
                    continue
                violations.append(
                    f"segment {segment_number} dialogue uses {word_count} words; budget is {budget}"
                )

    lowered_source = str(prompt or "").casefold()
    lowered_ledger = json.dumps(ledger, ensure_ascii=False).casefold()
    for pattern in UNREQUESTED_SPECTACLE_PATTERNS:
        match = re.search(pattern, lowered_ledger, flags=re.IGNORECASE)
        if match and not re.search(pattern, lowered_source, flags=re.IGNORECASE):
            violations.append(f"invented unrequested power/effect: {match.group(0).strip()}")
            break
    if not sanitize_h3_prompt_text(ledger.get("required_final_outcome")):
        violations.append("required final outcome is empty")
    return list(dict.fromkeys(violations))


def _deterministic_ledger(
    prompt: str,
    *,
    segment_count: int,
    segment_durations: list[float] | None = None,
    locked_dialogue: list[dict[str, Any]],
    camera_coverage: str,
    reference_context: str,
) -> dict[str, Any]:
    source_events = extract_source_events(prompt)
    fragments = [item["text"] for item in source_events]
    intent = extract_h3_source_intent(prompt)
    reference_cast = _reference_h3_cast_names(reference_context)
    cast_names = _merge_h3_cast_names(
        list(intent.get("cast_names") or []),
        reference_cast,
        prompt=prompt,
    )
    locked_dialogue = _canonicalize_h3_dialogue_speakers(
        locked_dialogue,
        cast_names,
    )
    intent["cast_names"] = cast_names
    intent["cast_cardinality_contract"] = _h3_cast_cardinality_contract(
        prompt,
        cast_names,
    )
    intent["blocking_contract"] = _infer_h3_blocking_contract(
        prompt,
        cast_names,
    )
    beats: list[dict[str, Any]] = []
    event_buckets: list[list[dict[str, str]]] = [[] for _ in range(segment_count)]
    expected_dialogue_events = _expected_dialogue_events(prompt, locked_dialogue)
    dialogue_by_event_source: dict[str, list[dict[str, Any]]] = {}
    for item in locked_dialogue:
        event_id = expected_dialogue_events.get(str(item.get("dialogue_id") or "").upper())
        if event_id:
            dialogue_by_event_source.setdefault(event_id, []).append(item)

    durations = [max(0.1, float(value)) for value in (segment_durations or [])]
    if len(durations) == segment_count and source_events:
        # Emergency fallback is still story-aware: use the actual spoken-word
        # cost plus a small action cost, then project that cumulative work onto
        # the available segment time. This avoids packing every early line into
        # window one merely because its speech cues are consecutive.
        event_costs: list[float] = []
        for event in source_events:
            spoken_words = sum(
                len(re.findall(r"\b[\w'’-]+\b", str(item.get("text") or "")))
                for item in dialogue_by_event_source.get(event["event_id"], [])
            )
            event_costs.append(1.15 + spoken_words / 2.1)
        total_cost = max(0.1, sum(event_costs))
        total_duration = max(0.1, sum(durations))
        thresholds: list[float] = []
        elapsed_duration = 0.0
        for duration in durations[:-1]:
            elapsed_duration += duration
            thresholds.append(total_cost * elapsed_duration / total_duration)
        elapsed_cost = 0.0
        current_segment = 0
        for index, (event, cost) in enumerate(zip(source_events, event_costs)):
            midpoint = elapsed_cost + cost * 0.5
            while (
                current_segment < segment_count - 1
                and midpoint > thresholds[current_segment]
            ):
                current_segment += 1
            if index == 0 and len(source_events) > 1:
                current_segment = 0
            if index == len(source_events) - 1:
                current_segment = segment_count - 1
            event_buckets[current_segment].append(event)
            elapsed_cost += cost
    else:
        for index, event in enumerate(source_events):
            # A single compound outcome belongs at the end; earlier segments
            # can then build toward it. With multiple events, anchor the first
            # and last to the timeline ends.
            target = (
                segment_count - 1
                if len(source_events) == 1
                else min(
                    segment_count - 1,
                    int(round(
                        index * (segment_count - 1)
                        / max(1, len(source_events) - 1)
                    )),
                )
            )
            event_buckets[target].append(event)

    # A simple entrance followed immediately by the first requested line is
    # one opening performance. Never let proportional scheduling turn that
    # into a complete silent native window.
    opening_dialogue_id = str(intent.get("opening_dialogue_id") or "").upper()
    opening_event_id = expected_dialogue_events.get(opening_dialogue_id)
    if opening_event_id:
        source_order = {
            event["event_id"]: index
            for index, event in enumerate(source_events)
        }
        for bucket_index, bucket in enumerate(event_buckets[1:], start=1):
            event_index = next(
                (
                    index for index, event in enumerate(bucket)
                    if event.get("event_id") == opening_event_id
                ),
                None,
            )
            if event_index is None:
                continue
            event_buckets[0].append(bucket.pop(event_index))
            event_buckets[0].sort(
                key=lambda event: source_order.get(event.get("event_id"), 10**9)
            )
            break

    # Prefer a meaningful physical handoff over an arbitrary proportional
    # split. Mounting/boarding and launching over an edge belong with the
    # setup window when the following window owns the sustained journey.
    handoff_re = re.compile(
        r"\b(?:laugh|mount|board|climb\s+(?:onto|aboard)|take\s+off|launch|"
        r"leap|jump|plummet\s+over)\b",
        flags=re.IGNORECASE,
    )
    if not durations:
        for bucket_index in range(max(0, segment_count - 1)):
            current = event_buckets[bucket_index]
            following = event_buckets[bucket_index + 1]
            moved = 0
            while len(following) > 1 and moved < 3 and handoff_re.search(following[0]["text"]):
                if re.search(r"\blaugh", following[0]["text"], re.IGNORECASE) and not any(
                    handoff_re.search(item["text"])
                    and not re.search(r"\blaugh", item["text"], re.IGNORECASE)
                    for item in following[1:3]
                ):
                    break
                current.append(following.pop(0))
                moved += 1
    else:
        def dialogue_words_for(events: list[dict[str, str]]) -> int:
            return sum(
                len(re.findall(r"\b[\w'’-]+\b", str(item.get("text") or "")))
                for event in events
                for item in dialogue_by_event_source.get(event["event_id"], [])
            )

        for bucket_index in range(max(0, segment_count - 1)):
            current = event_buckets[bucket_index]
            following = event_buckets[bucket_index + 1]
            # Keep a launch/handoff with its setup even when one final spoken
            # cue immediately precedes it. Leave the sustained journey or next
            # outcome for the following window. This preserves a natural cut
            # point without violating the duration-aware speech budget.
            handoff_index = next(
                (
                    index for index, event in enumerate(following[:4])
                    if handoff_re.search(event["text"])
                ),
                None,
            )
            if handoff_index is None:
                continue
            move_end = handoff_index
            while move_end < len(following) and handoff_re.search(following[move_end]["text"]):
                move_end += 1
            prefix = following[:move_end]
            if not prefix or len(following) <= len(prefix):
                continue
            dialogue_budget = max(1, int(math.floor(durations[bucket_index] * 2.1)))
            if dialogue_words_for(current + prefix) > dialogue_budget:
                continue
            current.extend(prefix)
            del following[:move_end]

    source_length = max(1, len(str(prompt or "")))
    event_segments = {
        event["event_id"]: segment_index + 1
        for segment_index, bucket in enumerate(event_buckets)
        for event in bucket
    }
    dialogue_by_event: dict[str, list[str]] = {}
    for item in locked_dialogue:
        dialogue_id = item["dialogue_id"]
        expected_event = expected_dialogue_events.get(dialogue_id)
        segment = event_segments.get(expected_event or "")
        if segment is None:
            ratio = min(1.0, max(0.0, int(item.get("source_offset") or 0) / source_length))
            segment = 1 + min(segment_count - 1, int(math.floor(ratio * segment_count)))
        if expected_event:
            dialogue_by_event.setdefault(expected_event, []).append(dialogue_id)
        else:
            bucket = event_buckets[max(0, min(segment_count - 1, segment - 1))]
            event_id = bucket[-1]["event_id"] if bucket else ""
            dialogue_by_event.setdefault(event_id, []).append(dialogue_id)

    def event_kind(event: dict[str, str], *, first_segment: bool) -> str:
        event_id = event["event_id"]
        text = event["text"]
        if event_id in dialogue_by_event or _SPEECH_VERB.search(text):
            return "dialogue"
        if first_segment and re.search(
            r"\b(?:pov|viewer|scene starts|begins|stands?|location|setting|"
            r"on top of|inside|outside)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return "setup"
        return "action"

    def partition_events(
        events: list[dict[str, str]],
        *,
        first_segment: bool,
    ) -> list[list[dict[str, str]]]:
        if not events:
            return []
        groups: list[list[dict[str, str]]] = []
        kinds: list[str] = []
        for event in events:
            kind = event_kind(event, first_segment=first_segment)
            if groups and kinds[-1] == kind:
                groups[-1].append(event)
            else:
                groups.append([event])
                kinds.append(kind)
        # Use available shot/beat capacity to keep long action chains
        # chronological instead of collapsing them into one mega-sentence.
        while len(groups) < 3:
            candidates = [
                (len(group), index)
                for index, group in enumerate(groups)
                if len(group) > 1 and kinds[index] != "dialogue"
            ]
            if not candidates:
                break
            _, index = max(candidates)
            group = groups[index]
            split = max(1, len(group) // 2)
            groups[index:index + 1] = [group[:split], group[split:]]
            kinds[index:index + 1] = [kinds[index], kinds[index]]
        while len(groups) > 3:
            merge_at = min(
                range(len(groups) - 1),
                key=lambda index: len(groups[index]) + len(groups[index + 1]),
            )
            groups[merge_at:merge_at + 2] = [groups[merge_at] + groups[merge_at + 1]]
            kinds[merge_at:merge_at + 2] = [
                kinds[merge_at] if kinds[merge_at] == kinds[merge_at + 1] else "action"
            ]
        return groups

    def effects_for(events: list[dict[str, str]]) -> str:
        text = " ".join(item["text"] for item in events).casefold()
        effects: list[str] = []
        if re.search(r"\blaugh", text):
            effects.append("the requested shared laughter")
        if re.search(r"\b(?:mount|broom|ride)\b", text):
            effects.append("hands tightening on broom handles and clothing shifting")
        if _FAST_ACTION_RE.search(text):
            effects.append("a rapidly intensifying wind rush")
        if re.search(r"\bwaterfalls?\b", text):
            effects.append("roaring water and synchronized spray")
        if re.search(r"\b(?:cave|canyon)\b", text):
            effects.append("fast environmental echoes")
        return "; ".join(effects) or "Natural synchronized effects for the visible action"

    def state_after(events: list[dict[str, str]], *, final: bool) -> str:
        last = sanitize_h3_prompt_text(events[-1]["text"] if events else "the requested beat")
        last = re.sub(r"^(?:then|next)\s+", "", last, flags=re.IGNORECASE)
        prefix_parts: list[str] = []
        if intent["first_person_pov"]:
            identity = f" {intent['pov_identity']}" if intent["pov_identity"] else ""
            prefix_parts.append(f"the first-person{identity} POV remains locked")
        if intent["hands_visible"] and re.search(
            r"\b(?:fl(?:y|ies|ew|ying)|fall(?:s|ing|en)?|drop(?:s|ped|ping)?|"
            r"plummet\w*|div\w*|rid\w*|brooms?|canyons?|waterfalls?|caves?)\b",
            last,
            flags=re.IGNORECASE,
        ):
            prefix_parts.append("the requested hands and held object remain visible in the foreground")
        physical = "; ".join(prefix_parts)
        if physical:
            physical += "; "
        if final and intent["ongoing_motion"]:
            return f"{physical}the requested motion is still actively continuing after {last}"
        return f"{physical}the immediate visible state is the result of this event: {last}"

    beat_number = 0
    for index in range(segment_count):
        assigned_events = event_buckets[index]
        next_event = next(
            (
                bucket[0]["text"]
                for bucket in event_buckets[index + 1:]
                if bucket
            ),
            "",
        )
        previous_event = next(
            (
                bucket[-1]["text"]
                for bucket in reversed(event_buckets[:index])
                if bucket
            ),
            "",
        )
        event_groups = partition_events(
            assigned_events,
            first_segment=index == 0,
        )
        if not event_groups:
            connective = (
                f"Show new physical progression from {previous_event} toward {next_event} without replaying either"
                if previous_event and next_event else
                f"Build visibly toward {next_event} without completing it"
                if next_event else
                f"Show new physical consequences after {previous_event} without replaying it"
                if previous_event else
                "Advance to a new visible story state without replaying an earlier action"
            )
            event_groups = [[{"event_id": "", "text": connective}]]
        for group in event_groups:
            beat_number += 1
            source_ids = [item["event_id"] for item in group if item["event_id"]]
            description = ". Then ".join(
                _filmable_source_event(item["text"]) for item in group
            )
            dialogue_ids = [
                dialogue_id
                for event in group
                for dialogue_id in dialogue_by_event.get(event["event_id"], [])
            ]
            beats.append({
                "beat_id": f"B{beat_number}",
                "segment": index + 1,
                "description": description,
                "source_event_ids": source_ids,
                "dialogue_ids": dialogue_ids,
                "state_after": state_after(
                    group,
                    final=(
                        index + 1 == segment_count
                        and group is event_groups[-1]
                    ),
                ),
                "sound_effects": effects_for(group),
            })

    names = list(intent.get("cast_names") or [])
    if intent["pov_identity"] and intent["pov_identity"] not in names:
        names.insert(0, intent["pov_identity"])
    if intent["first_person_pov"]:
        viewpoint = intent["pov_identity"] or "the viewpoint character"
        visible = [name for name in names if name.casefold() != viewpoint.casefold()]
        subject_continuity = (
            f"{viewpoint} remains the unseen first-person viewpoint"
            + (f"; {', '.join(visible)} retain their exact requested identities, appearance, wardrobe, and carried objects" if visible else "")
        )
    elif names:
        subject_continuity = (
            f"{', '.join(names)} retain their exact requested identities, appearance, wardrobe, and carried objects"
        )
    else:
        subject_continuity = "Keep every requested subject's identity, appearance, wardrobe, and carried objects unchanged"

    native_names = [
        name for name in names
        if not any(_same_h3_cast_identity(name, ref_name) for ref_name in reference_cast)
    ]
    continuity_parts: list[str] = []
    canonical_references = sanitize_h3_prompt_text(reference_context)
    if canonical_references:
        continuity_parts.append(canonical_references)
        if native_names:
            continuity_parts.append(
                f"{', '.join(native_names)} are named prompt-native recurring characters without media-reference bindings; preserve each requested identity, appearance, wardrobe, and carried objects"
            )
    else:
        continuity_parts.append(subject_continuity)
    continuity_parts.extend(
        value for value in (
            sanitize_h3_prompt_text(intent.get("cast_cardinality_contract")),
            sanitize_h3_prompt_text(intent.get("blocking_contract")),
        )
        if value
    )
    subject_continuity = ". ".join(
        part.strip(" .") for part in continuity_parts if part.strip(" .")
    )

    visual_contract = ". ".join(part for part in (
        intent["perspective_contract"],
        intent["style_contract"],
        "Keep lighting, color, screen direction, and established geography coherent",
    ) if part)
    first_state_events = event_buckets[0][:2] or source_events[:1]
    initial_state = ". ".join(
        _filmable_source_event(item["text"]) for item in first_state_events
    )
    opening_state_contract = sanitize_h3_prompt_text(
        intent.get("opening_state_contract")
    )
    if opening_state_contract:
        initial_state = opening_state_contract
    blocking_contract = sanitize_h3_prompt_text(intent.get("blocking_contract"))
    if blocking_contract:
        # The first half of a relational blocking contract describes the
        # composition *before* its assigned action. The old fallback used the
        # first event itself as opening_state (for example, already seated),
        # which forced shot one to restart that same entrance or seating beat.
        initial_state = re.split(
            r"\bAfter\s+that\b",
            blocking_contract,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .")
    final_outcome = _filmable_source_event(fragments[-1])
    if intent["ongoing_motion"]:
        final_outcome = f"The requested motion remains active after {final_outcome}"
    # A quote whose offset landed in an otherwise unexpected segment remains
    # assigned exactly once. No model-authored line is needed in fallback.
    return {
        "subject_continuity": subject_continuity,
        "setting_continuity": "Keep the requested location, geography, time of day, and background elements coherent",
        "visual_continuity": visual_contract,
        "editing_style": (
            "Locked continuous first-person POV with kinetic camera motion"
            if intent["first_person_pov"] and camera_coverage != "multi_shot"
            else "Motivated cinematic cuts and dynamic camera coverage"
            if camera_coverage == "multi_shot"
            else "A motivated cinematic camera follows the requested action"
        ),
        "initial_state": initial_state or "The requested scene begins in a clear readable composition",
        "ambient_audio": intent["ambient_contract"] or "Continuous natural nonverbal ambience appropriate to the requested location",
        "music": "N/A",
        "required_final_outcome": final_outcome,
        "beats": beats,
        "generated_dialogue": [],
        "source_intent": intent,
        "requested_nonverbal_vocals": intent["requested_nonverbal_vocals"],
        "sequence_shape": "ongoing" if intent["ongoing_motion"] else "resolved",
    }


_STORY_CONTEXT_FIELDS = (
    "subject_continuity",
    "setting_continuity",
    "visual_continuity",
    "editing_style",
    "initial_state",
    "ambient_audio",
    "music",
    "required_final_outcome",
)


def _anchor_immediate_opening_dialogue(
    prompt: str,
    ledger: dict[str, Any],
    *,
    locked_dialogue: list[dict[str, Any]],
    segment_count: int,
) -> bool:
    """Apply the source-owned opening-performance timing contract locally.

    The semantic planner may group and direct events, but it does not get to
    turn a brief entrance followed by the user's first line into a complete
    silent native clip. Move only the chronological prefix through that line
    into segment one. Any unsafe or incomplete schedule remains visible to the
    normal fidelity validator and still receives the focused repair/fallback.
    """

    source_events = extract_source_events(prompt)
    opening_dialogue_id = _opening_h3_dialogue_id(
        prompt,
        locked_dialogue,
        source_events,
    )
    if not opening_dialogue_id:
        return False
    beats = [
        item for item in (ledger.get("beats") or [])
        if isinstance(item, dict)
    ]
    owner_indexes = [
        index
        for index, beat in enumerate(beats)
        if opening_dialogue_id in {
            str(dialogue_id or "").upper()
            for dialogue_id in (beat.get("dialogue_ids") or [])
        }
    ]
    if len(owner_indexes) != 1:
        return False
    owner_index = owner_indexes[0]
    try:
        owner_segment = int(beats[owner_index].get("segment") or 0)
    except (TypeError, ValueError):
        return False
    if owner_segment <= 1 or owner_segment > max(1, int(segment_count)):
        return False

    for beat in beats[: owner_index + 1]:
        beat["segment"] = 1

    # Moving a compact entrance/dialogue prefix can expose an empty middle
    # segment in a three-beat schedule. Pull the earliest later non-final beat
    # into that gap when this is unambiguous; otherwise validation deliberately
    # leaves the plan for the focused LLM repair.
    final_source_event = source_events[-1]["event_id"] if source_events else ""
    source_map = {
        str(item.get("event_id") or "").upper(): _filmable_source_event(
            item.get("text")
        )
        for item in source_events
    }
    expected_dialogue_events = _expected_dialogue_events(
        prompt,
        locked_dialogue,
    )
    for missing_segment in range(2, max(1, int(segment_count))):
        if any(
            int(beat.get("segment") or 0) == missing_segment
            for beat in beats
        ):
            continue
        movable = next((
            beat
            for beat in beats[owner_index + 1:]
            if int(beat.get("segment") or 0) > missing_segment
            and final_source_event not in {
                str(event_id or "").upper()
                for event_id in (beat.get("source_event_ids") or [])
            }
        ), None)
        if movable is not None:
            movable["segment"] = missing_segment
            continue

        # A compact three-beat answer commonly groups every remaining source
        # event into the final beat. Split its non-final prefix locally so the
        # opening line can move forward without creating an empty middle clip.
        split_target: tuple[int, dict[str, Any], list[str]] | None = None
        for index, beat in enumerate(
            beats[owner_index + 1:],
            start=owner_index + 1,
        ):
            source_ids = [
                str(event_id or "").upper()
                for event_id in (beat.get("source_event_ids") or [])
            ]
            if (
                int(beat.get("segment") or 0) > missing_segment
                and len(source_ids) > 1
                and all(event_id in source_map for event_id in source_ids)
            ):
                split_target = (index, beat, source_ids)
                break
        if split_target is None:
            continue
        target_index, target, source_ids = split_target
        moved_source_ids = source_ids[:-1]
        retained_source_ids = source_ids[-1:]
        target_dialogue_ids = [
            str(dialogue_id or "").upper()
            for dialogue_id in (target.get("dialogue_ids") or [])
        ]
        moved_dialogue_ids = [
            dialogue_id
            for dialogue_id in target_dialogue_ids
            if str(expected_dialogue_events.get(dialogue_id) or "").upper()
            in set(moved_source_ids)
        ]
        moved_descriptions = [source_map[event_id] for event_id in moved_source_ids]
        retained_descriptions = [source_map[event_id] for event_id in retained_source_ids]
        split_beat = deepcopy(target)
        split_beat.update({
            "segment": missing_segment,
            "description": ". Then ".join(moved_descriptions),
            "source_event_ids": moved_source_ids,
            "dialogue_ids": moved_dialogue_ids,
            "state_after": (
                "The immediate visible state is the result of this event: "
                + moved_descriptions[-1]
            ),
            "sound_effects": "Natural synchronized effects for the visible action",
        })
        target["description"] = ". Then ".join(retained_descriptions)
        target["source_event_ids"] = retained_source_ids
        target["dialogue_ids"] = [
            dialogue_id
            for dialogue_id in target_dialogue_ids
            if dialogue_id not in set(moved_dialogue_ids)
        ]
        beats.insert(target_index, split_beat)

    for index, beat in enumerate(beats, start=1):
        beat["beat_id"] = f"B{index}"
    ledger["beats"] = beats
    return True


def _canonicalize_story_ledger(
    prompt: str,
    canonical: dict[str, Any],
    candidate: dict[str, Any] | None,
    *,
    locked_dialogue: list[dict[str, Any]],
    segment_count: int,
    allow_generated_dialogue: bool = False,
) -> dict[str, Any]:
    """Compile an LLM-authored semantic schedule against immutable catalogs.

    The model chooses which chronological events form a beat and which window
    owns that beat. Maestro assigns sequential beat/dialogue IDs, restores the
    user's exact event prose, and then validates coverage, order, timing, and
    speaker ownership. Nothing here silently replaces a rejected schedule with
    the deterministic one; that happens only after the focused repair fails.
    """

    ledger = {
        field: canonical.get(field, "")
        for field in _STORY_CONTEXT_FIELDS
    }
    ledger["beats"] = []
    ledger["generated_dialogue"] = []
    if not isinstance(candidate, dict):
        return ledger

    for field in _STORY_CONTEXT_FIELDS:
        value = sanitize_h3_prompt_text(candidate.get(field))
        if value:
            ledger[field] = value

    source_events = extract_source_events(prompt)
    canonical_cast_names = list(
        (canonical.get("source_intent") or {}).get("cast_names") or []
    )
    source_map = {
        item["event_id"]: _filmable_source_event(item["text"])
        for item in source_events
    }
    locked_dialogue_ids = [
        str(item.get("dialogue_id") or "").upper()
        for item in locked_dialogue
        if str(item.get("dialogue_id") or "").strip()
    ]
    locked_dialogue_id_set = set(locked_dialogue_ids)
    expected_dialogue_events = _expected_dialogue_events(prompt, locked_dialogue)
    locked_dialogue_by_event: dict[str, list[str]] = {}
    for dialogue_id in locked_dialogue_ids:
        event_id = str(expected_dialogue_events.get(dialogue_id) or "").upper()
        if event_id:
            locked_dialogue_by_event.setdefault(event_id, []).append(dialogue_id)
    canonical_by_signature = {
        (
            int(item.get("segment") or 0),
            tuple(str(event_id or "").upper() for event_id in (item.get("source_event_ids") or [])),
        ): item
        for item in (canonical.get("beats") or [])
        if isinstance(item, dict)
    }
    proposed_dialogue_segments: dict[str, int] = {}
    for beat in candidate.get("beats") or []:
        if not isinstance(beat, dict):
            continue
        try:
            proposed_segment = int(beat.get("segment") or 0)
        except (TypeError, ValueError):
            proposed_segment = 0
        for dialogue_id in beat.get("dialogue_ids") or []:
            proposed_dialogue_segments[str(dialogue_id or "").upper()] = proposed_segment

    raw_generated = [
        item for item in (candidate.get("generated_dialogue") or [])
        if isinstance(item, dict)
    ]
    if allow_generated_dialogue:
        for index, item in enumerate(raw_generated):
            dialogue_number = len(locked_dialogue) + index + 1
            try:
                segment = int(
                    item.get("segment")
                    or proposed_dialogue_segments.get(f"D{dialogue_number}")
                    or 0
                )
            except (TypeError, ValueError):
                segment = 0
            text = sanitize_h3_prompt_text(item.get("text"))
            if not text or segment < 1 or segment > segment_count:
                continue
            ledger["generated_dialogue"].append({
                "dialogue_id": f"D{dialogue_number}",
                "speaker": _resolve_h3_cast_name(
                    item.get("speaker"),
                    canonical_cast_names,
                ),
                "language": sanitize_h3_prompt_text(item.get("language")) or "English",
                "delivery": sanitize_h3_prompt_text(item.get("delivery")) or "speaks naturally and clearly",
                "text": text,
                "segment": segment,
            })

    for index, item in enumerate(candidate.get("beats") or []):
        if not isinstance(item, dict):
            continue
        try:
            segment = int(item.get("segment") or 0)
        except (TypeError, ValueError):
            segment = 0
        source_ids = [
            str(event_id or "").upper()
            for event_id in (item.get("source_event_ids") or [])
        ]
        proposed_dialogue_ids = [
            str(dialogue_id or "").upper()
            for dialogue_id in (item.get("dialogue_ids") or [])
        ]
        # Dialogue ownership is not a creative decision. The small planning
        # model may choose beat grouping and segment placement, but Maestro
        # already knows which immutable source event owns every quoted D-id.
        # Rebuild that binding from the locked catalog instead of accepting a
        # duplicated, omitted, or reordered model-authored dialogue_ids array.
        # Generated dialogue is attached below from its own canonical segment
        # field, so it is intentionally omitted here as well.
        dialogue_ids = [
            dialogue_id
            for event_id in source_ids
            for dialogue_id in locked_dialogue_by_event.get(event_id, [])
        ]
        # Preserve a genuinely unanchored locked quote once when the parser
        # could not associate it with a source speech event. Anchored dialogue
        # never trusts the model-authored D-id placement.
        dialogue_ids.extend(
            dialogue_id
            for dialogue_id in proposed_dialogue_ids
            if dialogue_id in locked_dialogue_id_set
            and not expected_dialogue_events.get(dialogue_id)
            and not any(
                dialogue_id in (existing.get("dialogue_ids") or [])
                for existing in ledger["beats"]
            )
        )
        exact_events = [source_map[event_id] for event_id in source_ids if event_id in source_map]
        canonical_match = canonical_by_signature.get((segment, tuple(source_ids)), {})
        description = (
            ". Then ".join(exact_events)
            if exact_events
            else sanitize_h3_prompt_text(item.get("description"))
        )
        state_after = sanitize_h3_prompt_text(item.get("state_after")) or sanitize_h3_prompt_text(
            canonical_match.get("state_after")
        )
        if not state_after and exact_events:
            state_after = f"The immediate visible state is the result of this event: {exact_events[-1]}"
        sound_effects = sanitize_h3_prompt_text(item.get("sound_effects")) or sanitize_h3_prompt_text(
            canonical_match.get("sound_effects")
        ) or "Natural synchronized effects for the visible action"
        ledger["beats"].append({
            "beat_id": f"B{index + 1}",
            "segment": segment,
            "description": description,
            "source_event_ids": source_ids,
            "dialogue_ids": dialogue_ids,
            "state_after": state_after,
            "sound_effects": sound_effects,
        })

    # Generated lines choose a segment in their own schema entry. Attach each
    # one to the last semantic beat in that window after the LLM beat schedule
    # has been compiled. Locked dialogue stays exactly where the model placed
    # its immutable D-id and is checked against the speech event catalog.
    for item in ledger["generated_dialogue"]:
        if any(
            item["dialogue_id"] in (beat.get("dialogue_ids") or [])
            for beat in ledger["beats"]
        ):
            continue
        segment_beats = [
            beat for beat in ledger["beats"]
            if int(beat.get("segment") or 0) == int(item.get("segment") or 0)
        ]
        if segment_beats:
            segment_beats[-1]["dialogue_ids"].append(item["dialogue_id"])
    ledger["ambient_audio"] = sanitize_h3_nonverbal_audio(
        ledger.get("ambient_audio")
    )
    _anchor_immediate_opening_dialogue(
        prompt,
        ledger,
        locked_dialogue=locked_dialogue,
        segment_count=segment_count,
    )
    return ledger


def _spread_generated_dialogue_across_segments(
    ledger: dict[str, Any],
    *,
    segment_count: int,
) -> None:
    """Distribute an authored conversation without rewriting its words.

    The semantic LLM owns the dialogue itself. Maestro only rebalances the
    ordered lines onto the available native H3 windows, ensuring a discussion
    does not spend its first 14 seconds silent and then cram two turns into a
    later pass. This is intentionally used only for unquoted, conversation-
    first Creative briefs; user-locked dialogue keeps its authored anchors.
    """

    generated = [
        item for item in (ledger.get("generated_dialogue") or [])
        if isinstance(item, dict)
        and str(item.get("dialogue_id") or "").strip()
        and str(item.get("text") or "").strip()
    ]
    beats = [
        item for item in (ledger.get("beats") or [])
        if isinstance(item, dict)
    ]
    count = max(1, int(segment_count))
    if not generated or not beats:
        return

    generated_ids = {
        str(item.get("dialogue_id") or "").upper()
        for item in generated
    }
    for beat in beats:
        beat["dialogue_ids"] = [
            str(dialogue_id or "").upper()
            for dialogue_id in (beat.get("dialogue_ids") or [])
            if str(dialogue_id or "").upper() not in generated_ids
        ]

    total = len(generated)
    for index, item in enumerate(generated):
        segment = (
            1
            if total == 1 else
            1 + int(round(index * (count - 1) / float(total - 1)))
        )
        segment = min(count, max(1, segment))
        item["segment"] = segment
        segment_beats = [
            beat for beat in beats
            if int(beat.get("segment") or 0) == segment
        ]
        if segment_beats:
            segment_beats[-1].setdefault("dialogue_ids", []).append(
                str(item.get("dialogue_id") or "").upper()
            )


def _salvage_creative_fallback(
    prompt: str,
    canonical_ledger: dict[str, Any],
    rejected_ledger: dict[str, Any] | None,
    *,
    locked_dialogue: list[dict[str, Any]],
    segment_count: int,
    segment_durations: list[float],
    spread_generated_dialogue: bool,
) -> tuple[dict[str, Any], int]:
    """Repair structure without throwing away a valid Creative script.

    A small planning model can write useful dialogue and cinematic context yet
    miss an immutable event ID, duplicate a beat, or place the final source
    event in the wrong window.  The old all-or-nothing fallback discarded the
    entire response in that case.  For Creative mode, rebuild event ownership
    from Maestro's deterministic ledger while transplanting only the
    canonicalized, non-placeholder dialogue and context that still pass the
    complete fidelity validator.
    """

    fallback = deepcopy(canonical_ledger)
    if not isinstance(rejected_ledger, dict):
        return fallback, 0

    generated: list[dict[str, Any]] = []
    seen_lines: set[tuple[str, str]] = set()
    next_number = len(locked_dialogue) + 1
    max_items = max(1, int(segment_count) * 2)
    for raw in rejected_ledger.get("generated_dialogue") or []:
        if not isinstance(raw, dict) or len(generated) >= max_items:
            continue
        text = sanitize_h3_prompt_text(raw.get("text"))
        speaker = sanitize_h3_prompt_text(raw.get("speaker"))
        if (
            not text
            or not speaker
            or _PLACEHOLDER_DIALOGUE.fullmatch(text)
        ):
            continue
        signature = (speaker.casefold(), _normalize_key(text))
        if not signature[1] or signature in seen_lines:
            continue
        seen_lines.add(signature)
        try:
            segment = int(raw.get("segment") or 1)
        except (TypeError, ValueError):
            segment = 1
        generated.append({
            "dialogue_id": f"D{next_number}",
            "speaker": speaker,
            "language": sanitize_h3_prompt_text(raw.get("language")) or "English",
            "delivery": sanitize_h3_prompt_text(raw.get("delivery")) or "speaks naturally and clearly",
            "text": text,
            "segment": min(max(1, int(segment_count)), max(1, segment)),
        })
        next_number += 1

    if not generated:
        return fallback, 0

    def attach_dialogue(target: dict[str, Any]) -> None:
        target["generated_dialogue"] = deepcopy(generated)
        generated_ids = {
            str(item.get("dialogue_id") or "").upper()
            for item in generated
        }
        beats = [
            item for item in (target.get("beats") or [])
            if isinstance(item, dict)
        ]
        for beat in beats:
            beat["dialogue_ids"] = [
                str(dialogue_id or "").upper()
                for dialogue_id in (beat.get("dialogue_ids") or [])
                if str(dialogue_id or "").upper() not in generated_ids
            ]
        if spread_generated_dialogue:
            _spread_generated_dialogue_across_segments(
                target,
                segment_count=segment_count,
            )
            return
        for item in target["generated_dialogue"]:
            segment_beats = [
                beat for beat in beats
                if int(beat.get("segment") or 0) == int(item.get("segment") or 0)
            ]
            if segment_beats:
                segment_beats[-1].setdefault("dialogue_ids", []).append(
                    str(item.get("dialogue_id") or "").upper()
                )

    # First preserve the safe high-level direction. If any such field caused
    # the rejection (for example an invented visual effect), retry with only
    # the authored dialogue on Maestro's canonical context.
    enriched = deepcopy(fallback)
    for field in (
        "subject_continuity",
        "setting_continuity",
        "visual_continuity",
        "editing_style",
        "initial_state",
        "ambient_audio",
        "music",
    ):
        value = sanitize_h3_prompt_text(rejected_ledger.get(field))
        if value:
            enriched[field] = value
    enriched["ambient_audio"] = sanitize_h3_nonverbal_audio(
        enriched.get("ambient_audio")
    )
    attach_dialogue(enriched)

    def violations_for(target: dict[str, Any]) -> list[str]:
        return ledger_violations(
            prompt,
            target,
            segment_count=segment_count,
            locked_dialogue=locked_dialogue,
            expect_dialogue=True,
            allow_generated_dialogue=True,
            require_dialogue_per_segment=spread_generated_dialogue,
            segment_durations=segment_durations,
        )

    if not violations_for(enriched):
        return enriched, len(generated)

    dialogue_only = deepcopy(fallback)
    attach_dialogue(dialogue_only)
    if not violations_for(dialogue_only):
        return dialogue_only, len(generated)
    return fallback, 0


def _lock_ledger_source_events(prompt: str, ledger: dict[str, Any]) -> None:
    """Replace LLM paraphrases with the user's immutable event wording."""

    source_events = extract_source_events(prompt)
    event_map = {
        item["event_id"]: _filmable_source_event(item["text"])
        for item in source_events
    }
    for beat in ledger.get("beats") or []:
        if not isinstance(beat, dict):
            continue
        exact = [
            event_map[str(event_id or "").upper()]
            for event_id in (beat.get("source_event_ids") or [])
            if str(event_id or "").upper() in event_map
        ]
        if exact:
            beat["description"] = ". Then ".join(exact)
    if source_events:
        ledger["required_final_outcome"] = _filmable_source_event(
            source_events[-1]["text"]
        )


def _segment_shot_limit(assigned_beats: list[dict[str, Any]]) -> int:
    """Allow extra cuts only when distinct dialogue phases require them."""

    return min(6, max(4, len(assigned_beats)))


def _segment_schema(
    segment_number: int,
    *,
    maximum_shots: int = 4,
) -> dict[str, Any]:
    shot = {
        "type": "object",
        "properties": {
            "shot": {"type": "integer"},
            "start_seconds": {"type": "number"},
            "end_seconds": {"type": "number"},
            "transition": {"type": "string"},
            "framing": {"type": "string"},
            "camera": {"type": "string"},
            "action": {"type": "string"},
            "sound_effects": {"type": "string"},
        },
        "required": [
            "shot", "start_seconds", "end_seconds", "transition", "framing",
            "camera", "action", "sound_effects",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "segment": {
                "type": "integer",
                "minimum": segment_number,
                "maximum": segment_number,
            },
            "title": {"type": "string"},
            "opening_state": {"type": "string"},
            "coverage": {"type": "string"},
            "pacing": {"type": "string"},
            "shots": {
                "type": "array",
                "minItems": 1,
                "maxItems": max(1, int(maximum_shots)),
                "items": shot,
            },
            "closing_state": {"type": "string"},
        },
        "required": [
            "segment", "title", "opening_state", "coverage", "pacing",
            "shots", "closing_state",
        ],
        "additionalProperties": False,
    }


def _strip_planner_speech_cues(value: Any, *, sound_field: bool = False) -> str:
    """Remove model-authored speech hints from application-owned dialogue.

    Exact spoken performance is attached later from the locked dialogue
    catalog.  Leaving phrases such as ``Yoda speaks`` in another shot (or
    ``Yoda's voice`` in sound effects) creates a second, untagged vocal event
    that H3 may fill with repeated or gibberish speech.
    """

    text = sanitize_h3_prompt_text(value)
    if not text:
        return ""
    speech_audio = re.compile(
        r"\b(?:voice|voices|dialogue|speech|spoken|vocal\s+line|"
        r"chatter|murmur(?:ing)?|babble|conversation|"
        r"people\s+(?:talking|speaking)|crowd\s+(?:talking|speaking))\b",
        flags=re.IGNORECASE,
    )
    clause_pattern = r"\s*;\s*|(?<=[.!?])\s+"
    if sound_field:
        # Soundscapes are commonly returned as comma-separated lists. Split
        # those items so one invalid "background chatter" phrase does not
        # discard valid music, cups, wind, or machinery in the same sentence.
        clause_pattern += r"|\s*,\s*(?:and\s+)?"
    clauses = re.split(clause_pattern, text)
    kept = [
        clause.strip()
        for clause in clauses
        if clause.strip()
        and not _SPEECH_VERB.search(clause)
        and not (sound_field and speech_audio.search(clause))
    ]
    return "; ".join(kept)


def sanitize_h3_nonverbal_audio(value: Any) -> str:
    """Return ambience that cannot invite untagged H3 speech.

    Preserve an explicit acoustic-matching direction such as ``character
    voices sound natural in the environment``. That controls the room sound
    of tagged dialogue; it does not request another speaker. Chatter, murmurs,
    and other speech-like background layers are removed.
    """

    source = sanitize_h3_prompt_text(value)
    acoustic_contracts = [
        clause.strip()
        for clause in re.split(r"\s*;\s*|(?<=[.!?])\s+", source)
        if clause.strip() and _is_persistent_audio_directive(clause)
        and re.search(r"\bvoices?\b", clause, flags=re.IGNORECASE)
    ]
    nonverbal = _strip_planner_speech_cues(source, sound_field=True)
    parts = [
        part for part in [nonverbal, *acoustic_contracts]
        if part
    ]
    return "; ".join(dict.fromkeys(parts)) or "Natural nonverbal location ambience"


def _speaker_name_present(value: Any, speaker: str) -> bool:
    """Return whether ``speaker`` is named as a whole phrase in ``value``."""

    text = sanitize_h3_prompt_text(value)
    name = sanitize_h3_prompt_text(speaker)
    if not text or not name:
        return False
    return bool(re.search(
        rf"(?<![\w]){re.escape(name)}(?![\w])",
        text,
        flags=re.IGNORECASE,
    ))


def _speaker_is_camera_focus(
    value: Any,
    speaker: str,
    all_speakers: list[str] | None = None,
) -> bool:
    """Return whether camera prose makes ``speaker`` the visual priority.

    Merely mentioning the speaker is not enough. ``George is dominant and
    gestures toward Joey`` contains Joey's name, but the camera still tells H3
    to animate George's face. Keep this intentionally narrower than general
    name matching so ordinary two-shots remain available.
    """

    text = sanitize_h3_prompt_text(value)
    name = sanitize_h3_prompt_text(speaker)
    if not text or not name:
        return False
    aliases = _h3_cast_aliases(name, all_speakers or [name])
    escaped = "(?:" + "|".join(re.escape(alias) for alias in aliases) + ")"
    return bool(
        re.search(
            rf"\b(?:focus(?:ed|es|ing)?|settles?|holds?)\s+"
            rf"(?:on|upon)\s+{escaped}(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"\b(?:frames?|center(?:ed|s|ing)?|centred|tracks?|"
            rf"emphasiz(?:e|es|ed|ing)|elevat(?:e|es|ed|ing))\b"
            rf"[^.;:]{{0,35}}(?<![\w]){escaped}(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"\b(?:look(?:s|ed|ing)?\s+(?:up\s+|down\s+)?(?:at|toward)|"
            rf"pan(?:s|ned|ning)?\s+(?:to|toward)|"
            rf"rack\s+focus\s+(?:to|onto)|push(?:es|ed|ing)?\s+in\s+on)\s+"
            rf"{escaped}(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"\brack\s+focus\s+from\b[^.;:]{{0,60}}\bto\s+"
            rf"{escaped}(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"(?<![\w]){escaped}(?![\w])[^.;:]{{0,55}}\b"
            r"(?:is|remains|becomes|stays)\s+(?:the\s+)?"
            r"(?:dominant|primary\s+focus|visual\s+focus|center(?:ed)?|"
            r"centred|center\s+frame|foregrounded|in\s+the\s+foreground|"
            r"filling\s+(?:most\s+of\s+)?the\s+frame)\b",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"\b(?:close[- ]?up|medium\s+close[- ]?up|reaction\s+shot)\b"
            rf"[^.;:]{{0,50}}\b(?:of|on)\s+{escaped}(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"\b(?:medium|wide|full|establishing)\s+shot\b"
            rf"[^.;:]{{0,24}}\b(?:of|on)\s+{escaped}(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"(?<![\w]){escaped}(?![\w])[^.;:]{{0,18}}\b"
            r"(?:close[- ]?up|reaction\s+shot)\b",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"(?<![\w]){escaped}(?![\w])[^.;:]{{0,35}}\b"
            r"(?:active\s+visible\s+speaker|carries\s+(?:the\s+)?visible\s+"
            r"speaking\s+performance)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _enforce_materialized_vocal_staging(
    *,
    framing: str,
    camera: str,
    action: str,
    dialogue_sources: list[dict[str, Any]],
    known_speakers: list[str],
    future_cast: list[str] | None = None,
) -> tuple[str, str, str]:
    """Keep the tagged voice, visible mouth, and camera subject inseparable.

    H3 can correctly select an Audio reference while lip-syncing the face that
    happens to dominate an adjacent camera/action sentence. It can also read
    an ordinary action clause aloud when that clause sits near dialogue. The
    story ledger owns both contracts: camera prose may choose coverage, but it
    may not choose a different visible speaker, and action prose is never an
    additional transcript.

    Do not turn speaker ownership into a close-up or face-hold instruction.
    Ref2VA may interpret that camera pressure as a request to materialize the
    speaker's identity portrait as target footage. Speaker ownership belongs
    beside the tagged line; camera fields remain ordinary target-scene prose.
    """

    framing = sanitize_h3_prompt_text(framing) or "cinematic medium shot"
    camera = sanitize_h3_prompt_text(camera) or "a motivated camera follows the action"
    action = sanitize_h3_prompt_text(action) or (
        "The established result remains visible without repeating an earlier event"
    )

    future_names = [
        sanitize_h3_prompt_text(name)
        for name in (future_cast or [])
        if sanitize_h3_prompt_text(name)
    ]
    visible_speakers: list[str] = []
    off_camera_speakers: list[str] = []
    for source in dialogue_sources:
        speaker = sanitize_h3_prompt_text(source.get("speaker")) or "Speaker"
        target = off_camera_speakers if bool(source.get("off_camera")) else visible_speakers
        if speaker.casefold() not in {item.casefold() for item in target}:
            target.append(speaker)

    # Put the no-narration instruction before the action. The downstream H3
    # compiler intentionally compacts long action fields, so a suffix could be
    # truncated precisely on the complex shots that need this protection most.
    if not dialogue_sources:
        if any(_speaker_name_present(framing, name) for name in future_names):
            framing = (
                "the same established composition with only already-introduced subjects"
            )
        if any(_speaker_name_present(camera, name) for name in future_names):
            camera = (
                "maintain the existing blocking and follow only already-introduced subjects"
            )
        action = (
            "Silent visual action, never spoken narration: "
            f"{action}. No words are spoken or mouthed in this shot; only "
            "explicitly requested nonverbal reactions may be heard"
        )
        return framing, camera, action

    action = f"Visual direction only, never spoken narration: {action}"
    unique_visible = list(dict.fromkeys(visible_speakers))
    # A camera LLM may preview a later entrant as a reaction angle even though
    # the story ledger has not introduced that person yet. Replace the invalid
    # angle with concrete coverage of the actual speaker when one exists;
    # generic "active cast" prose can make H3 invent a second stage position.
    active_speaker = unique_visible[0] if len(unique_visible) == 1 else ""
    if any(_speaker_name_present(framing, name) for name in future_names):
        framing = (
            f"medium shot on {active_speaker} in the existing composition; other "
            "already-present subjects remain in their established positions"
            if active_speaker else
            "the same established composition with only already-introduced subjects"
        )
    if any(_speaker_name_present(camera, name) for name in future_names):
        camera = (
            f"maintain the existing blocking and settle on {active_speaker} before "
            "the vocal line begins"
            if active_speaker else
            "maintain the existing blocking and follow only already-introduced subjects"
        )
    if len(unique_visible) == 1:
        speaker = unique_visible[0]
        inactive_speakers = [
            value for value in known_speakers
            if value.casefold() != speaker.casefold()
        ]
        framing_has_wrong_focus = any(
            _speaker_is_camera_focus(framing, other, known_speakers)
            for other in inactive_speakers
        )
        camera_has_wrong_focus = any(
            _speaker_is_camera_focus(camera, other, known_speakers)
            for other in inactive_speakers
        )
        if framing_has_wrong_focus:
            framing = (
                "medium scene composition in the established target setting; "
                f"{speaker} carries the visible speaking performance and "
                "listeners remain visually secondary"
            )
        elif not _speaker_name_present(framing, speaker):
            framing = (
                f"{framing}; {speaker} carries the visible speaking performance and "
                "listeners remain visually secondary"
            )
        if camera_has_wrong_focus:
            camera = (
                "maintain the established target-scene camera coverage; settle on "
                f"{speaker} before the vocal line begins, then show listener "
                "reactions only after the line ends"
            )
        elif not _speaker_name_present(camera, speaker):
            camera = (
                f"{camera}; settle on {speaker} before the vocal line begins, "
                "then show listener reactions only after the line ends"
            )
    elif unique_visible:
        active_names = {value.casefold() for value in unique_visible}
        inactive_speakers = [
            value for value in known_speakers
            if value.casefold() not in active_names
        ]
        framing_conflicts = (
            not any(_speaker_name_present(framing, speaker) for speaker in unique_visible)
            and any(_speaker_name_present(framing, other) for other in inactive_speakers)
        )
        camera_conflicts = (
            not any(_speaker_name_present(camera, speaker) for speaker in unique_visible)
            and any(_speaker_name_present(camera, other) for other in inactive_speakers)
        )
        if framing_conflicts:
            framing = "medium scene composition within the established target setting"
        if camera_conflicts:
            camera = "maintain the established target-scene camera coverage"
    return framing, camera, action


def _canonicalize_segment_contract(
    segment: dict[str, Any] | None,
    *,
    segment_number: int,
    duration: float,
    assigned_beats: list[dict[str, Any]],
    dialogue_catalog: list[dict[str, Any]],
    opening_state: str,
    source_intent: dict[str, Any],
) -> dict[str, Any] | None:
    """Apply immutable beat/dialogue ownership after creative camera planning.

    The local LLM now owns only coverage, choreography, and timing. Maestro
    owns IDs, exact dialogue, POV, pacing, and source-event inclusion, so a
    capable small model no longer fails because it copied one bookkeeping ID
    incorrectly.
    """

    if not isinstance(segment, dict):
        return None
    maximum_shots = _segment_shot_limit(assigned_beats)
    raw_shots = [
        dict(item) for item in (segment.get("shots") or [])
        if isinstance(item, dict)
    ][:maximum_shots]
    if not raw_shots:
        return segment

    total = max(0.1, float(duration))
    shot_count = len(raw_shots)
    minimum_shot = min(0.75, max(0.25, total / max(2, shot_count * 2)))
    cursor = 0.0
    for index, shot in enumerate(raw_shots):
        default_end = total * (index + 1) / shot_count
        try:
            requested_end = float(shot.get("end_seconds", default_end))
        except (TypeError, ValueError):
            requested_end = default_end
        remaining = shot_count - index - 1
        latest_end = total - minimum_shot * remaining
        end = total if index + 1 == shot_count else min(
            latest_end,
            max(cursor + minimum_shot, requested_end),
        )
        shot["shot"] = index + 1
        shot["start_seconds"] = round(cursor, 3)
        shot["end_seconds"] = round(end, 3)
        shot["transition"] = sanitize_h3_prompt_text(
            shot.get("transition")
            or ("opening composition" if index == 0 else "hard cut")
        )
        shot["framing"] = sanitize_h3_prompt_text(
            shot.get("framing") or "cinematic medium shot"
        )
        shot["camera"] = sanitize_h3_prompt_text(
            shot.get("camera") or "a motivated camera follows the action"
        )
        shot["action"] = (
            _strip_planner_speech_cues(shot.get("action"))
            if dialogue_catalog else sanitize_h3_prompt_text(shot.get("action"))
        )
        shot["sound_effects"] = (
            _strip_planner_speech_cues(
                shot.get("sound_effects"),
                sound_field=True,
            )
            if dialogue_catalog else sanitize_h3_prompt_text(shot.get("sound_effects"))
        ) or "Natural synchronized effects"
        cursor = end
    raw_shots[-1]["end_seconds"] = round(total, 3)

    assignments: list[list[dict[str, Any]]] = [[] for _ in raw_shots]
    beat_count = len(assigned_beats)
    for index, beat in enumerate(assigned_beats):
        target = min(shot_count - 1, int(index * shot_count / max(1, beat_count)))
        assignments[target].append(beat)

    dialogue_map = {
        str(item.get("dialogue_id") or "").upper(): item
        for item in dialogue_catalog
    }
    _apply_h3_filmable_shot_clock(
        raw_shots,
        assignments,
        dialogue_map,
        duration=total,
        only_when_squeezed=True,
    )
    if segment_number == 1 and source_intent.get("opening_dialogue_id"):
        _cap_h3_opening_dialogue_lead(
            raw_shots,
            assignments,
            duration=total,
            dialogue_id=str(source_intent.get("opening_dialogue_id")),
        )
    for shot, shot_beats in zip(raw_shots, assignments):
        beat_ids = [str(beat.get("beat_id") or "").upper() for beat in shot_beats]
        shot["beat_ids"] = beat_ids
        required = ". Then ".join(
            sanitize_h3_prompt_text(beat.get("description"))
            for beat in shot_beats
            if sanitize_h3_prompt_text(beat.get("description"))
        )
        # The camera planner owns framing, camera motion, transitions, and
        # timing. It does not own story action. Keeping its supplemental action
        # prose let a later shot repeat the preceding beat (or describe the
        # wrong character during a tagged line), even though the immutable beat
        # schedule was correct. Compile assigned story events verbatim and use
        # a neutral hold for any optional reaction angle with no assigned beat.
        shot["action"] = required or (
            "The immediate result of the preceding assigned event remains "
            "visible without repeating, restarting, or adding a story event"
        )

        existing = {
            str(item.get("dialogue_id") or "").upper(): item
            for item in (shot.get("dialogue") or [])
            if isinstance(item, dict)
        }
        dialogue_ids = [
            str(dialogue_id or "").upper()
            for beat in shot_beats
            for dialogue_id in (beat.get("dialogue_ids") or [])
        ]
        performances: list[dict[str, str]] = []
        for dialogue_id in dialogue_ids:
            source = dialogue_map.get(dialogue_id, {})
            proposed = existing.get(dialogue_id, {})
            off_camera = bool(source.get("off_camera"))
            performances.append({
                "dialogue_id": dialogue_id,
                "delivery": sanitize_h3_prompt_text(
                    proposed.get("delivery")
                    or source.get("delivery")
                    or "speaks naturally and clearly"
                ),
                "action": sanitize_h3_prompt_text(
                    proposed.get("action")
                    or (
                        "remaining off-camera at the unseen first-person viewpoint"
                        if off_camera
                        else "performing the assigned visible action"
                    )
                ),
            })
        shot["dialogue"] = performances

    result = dict(segment)
    result.update({
        "segment": segment_number,
        "opening_state": sanitize_h3_prompt_text(opening_state),
        "shots": raw_shots,
        "closing_state": sanitize_h3_prompt_text(
            assigned_beats[-1].get("state_after")
            if assigned_beats else segment.get("closing_state")
        ),
    })
    if source_intent.get("first_person_pov"):
        result["coverage"] = (
            "continuous locked first-person POV"
            if str(result.get("coverage") or "").casefold() != "multi_shot"
            else sanitize_h3_prompt_text(result.get("coverage"))
        )
    else:
        result["coverage"] = sanitize_h3_prompt_text(
            result.get("coverage") or "coherent cinematic coverage"
        )
    result["pacing"] = (
        sanitize_h3_prompt_text(source_intent.get("pacing_contract"))
        if (
            source_intent.get("fast_action")
            or source_intent.get("energetic_performance")
        ) else
        sanitize_h3_prompt_text(result.get("pacing") or "natural real-time pacing")
    )
    return result


def segment_violations(
    prompt: str,
    segment: dict[str, Any] | None,
    *,
    segment_number: int,
    duration: float,
    assigned_beats: list[dict[str, Any]],
    dialogue_catalog: list[dict[str, Any]],
) -> list[str]:
    """Validate one local camera plan without silently repairing its story."""

    if not isinstance(segment, dict):
        return ["invalid segment plan"]
    violations: list[str] = []
    try:
        returned_number = int(segment.get("segment"))
    except (TypeError, ValueError):
        returned_number = 0
    if returned_number != segment_number:
        violations.append(f"returned segment {returned_number} instead of {segment_number}")
    shots = [item for item in (segment.get("shots") or []) if isinstance(item, dict)]
    maximum_shots = _segment_shot_limit(assigned_beats)
    if not 1 <= len(shots) <= maximum_shots:
        violations.append(
            f"returned {len(shots)} shots instead of one to {maximum_shots}"
        )
        return violations

    assigned_beat_ids = [str(item.get("beat_id") or "").upper() for item in assigned_beats]
    used_beat_ids: list[str] = []
    beat_actions: dict[str, list[str]] = {}
    used_dialogue_ids: list[str] = []
    timing: list[tuple[float, float]] = []
    for index, shot in enumerate(shots):
        action = str(shot.get("action") or "").strip()
        if not action:
            violations.append(f"shot {index + 1} has no visible action")
        if "<d>" in action.casefold() or _CONTEXT_IR_LABEL.search(action):
            violations.append(f"shot {index + 1} embeds dialogue or Context-IR fields in its action")
        shot_beat_ids = [str(value or "").upper() for value in (shot.get("beat_ids") or [])]
        used_beat_ids.extend(shot_beat_ids)
        for beat_id in shot_beat_ids:
            beat_actions.setdefault(beat_id, []).append(action)
        for item in shot.get("dialogue") or []:
            if not isinstance(item, dict):
                violations.append(f"shot {index + 1} has an invalid dialogue performance")
                continue
            used_dialogue_ids.append(str(item.get("dialogue_id") or "").upper())
        try:
            start = float(shot.get("start_seconds"))
            end = float(shot.get("end_seconds"))
        except (TypeError, ValueError):
            violations.append(f"shot {index + 1} has invalid local timing")
            continue
        timing.append((start, end))

    if Counter(used_beat_ids) != Counter(assigned_beat_ids):
        violations.append("assigned beat IDs are missing, foreign, or repeated")
    for beat in assigned_beats:
        beat_id = str(beat.get("beat_id") or "").upper()
        required_tokens = _content_tokens(beat.get("description"))
        action_tokens = _content_tokens(" ".join(beat_actions.get(beat_id, [])))
        minimum_overlap = 1 if len(required_tokens) <= 4 else 2
        if required_tokens and len(required_tokens & action_tokens) < minimum_overlap:
            violations.append(f"{beat_id} shot action does not reflect its assigned source event")
    expected_dialogue_ids = [
        str(dialogue_id or "").upper()
        for beat in assigned_beats
        for dialogue_id in (beat.get("dialogue_ids") or [])
    ]
    known_dialogue_ids = {str(item.get("dialogue_id") or "").upper() for item in dialogue_catalog}
    if any(value not in known_dialogue_ids for value in used_dialogue_ids):
        violations.append("a shot uses an unknown dialogue ID")
    if used_dialogue_ids != expected_dialogue_ids:
        violations.append("dialogue IDs are missing, duplicated, or out of order")

    if len(timing) == len(shots):
        tolerance = 0.08
        if abs(timing[0][0]) > tolerance:
            violations.append("the local shot clock does not begin at 0.000 seconds")
        if abs(timing[-1][1] - duration) > tolerance:
            violations.append("the local shot clock does not end at the segment duration")
        minimum_shot = min(0.75, max(0.25, duration / max(2, len(shots) * 2)))
        for index, (start, end) in enumerate(timing):
            if start < -tolerance or end > duration + tolerance or end <= start:
                violations.append(f"shot {index + 1} is outside the local timeline")
            elif end - start < minimum_shot - tolerance:
                violations.append(f"shot {index + 1} is an unusably short tail shot")
            if index and abs(start - timing[index - 1][1]) > tolerance:
                violations.append(f"shot {index + 1} leaves a gap or overlap in the local timeline")

        if segment_number == 1:
            opening_dialogue_id = _opening_h3_dialogue_id(
                prompt,
                dialogue_catalog,
                extract_source_events(prompt),
            )
            if opening_dialogue_id in expected_dialogue_ids:
                dialogue_shot_index = next(
                    (
                        index
                        for index, shot in enumerate(shots)
                        if opening_dialogue_id in {
                            str(item.get("dialogue_id") or "").upper()
                            for item in (shot.get("dialogue") or [])
                            if isinstance(item, dict)
                        }
                    ),
                    None,
                )
                owner_index = next(
                    (
                        index
                        for index, beat in enumerate(assigned_beats)
                        if opening_dialogue_id in {
                            str(value or "").upper()
                            for value in (beat.get("dialogue_ids") or [])
                        }
                    ),
                    0,
                )
                lead_cap = min(4.5, max(2.0, duration * 0.30))
                if (
                    dialogue_shot_index is not None
                    and timing[dialogue_shot_index][0] > lead_cap + tolerance
                ):
                    violations.append(
                        "the first requested line begins too late in the opening segment"
                    )
                if (
                    dialogue_shot_index == 0
                    and owner_index > 0
                    and len(shots) == 1
                    and _find_h3_opening_entrance(prompt) is not None
                ):
                    violations.append(
                        "the opening entrance and first requested line need separate timed phases"
                    )

    lowered_source = str(prompt or "").casefold()
    lowered_segment = json.dumps(segment, ensure_ascii=False).casefold()
    # H3 can literalize orchestration vocabulary as scene content (for example,
    # turning "the next window" into an exterior view through a building
    # window).  Reject an LLM camera plan that introduces the word unless the
    # user's own concept actually requested a literal window.
    if (
        not re.search(r"\bwindows?\b", lowered_source)
        and re.search(r"\bwindows?\b", lowered_segment)
    ):
        violations.append("introduced the internal term 'window' as visible scene content")
    for pattern in UNREQUESTED_SPECTACLE_PATTERNS:
        match = re.search(pattern, lowered_segment, flags=re.IGNORECASE)
        if match and not re.search(pattern, lowered_source, flags=re.IGNORECASE):
            violations.append(f"invented unrequested power/effect: {match.group(0).strip()}")
            break
    if not sanitize_h3_prompt_text(segment.get("closing_state")):
        violations.append("closing state is empty")
    return list(dict.fromkeys(violations))


def _materialized_segment_violations(
    segment: dict[str, Any],
    *,
    known_speakers: list[str],
    future_cast: list[str] | None = None,
) -> list[str]:
    """Validate the final H3 prose after immutable dialogue is attached."""

    violations: list[str] = []
    for index, shot in enumerate(segment.get("shots") or []):
        if not isinstance(shot, dict):
            continue
        framing = sanitize_h3_prompt_text(shot.get("framing"))
        camera = sanitize_h3_prompt_text(shot.get("camera"))
        dialogue = [
            item for item in (shot.get("dialogue") or [])
            if isinstance(item, dict)
        ]
        visible_speakers = list(dict.fromkeys(
            sanitize_h3_prompt_text(item.get("speaker"))
            for item in dialogue
            if sanitize_h3_prompt_text(item.get("speaker"))
            and "off-camera" not in sanitize_h3_prompt_text(
                item.get("action")
            ).casefold()
        ))
        if len(visible_speakers) == 1:
            active = visible_speakers[0]
            framing_wrong = [
                other for other in known_speakers
                if other.casefold() != active.casefold()
                and _speaker_is_camera_focus(framing, other, known_speakers)
            ]
            camera_wrong = [
                other for other in known_speakers
                if other.casefold() != active.casefold()
                and _speaker_is_camera_focus(camera, other, known_speakers)
            ]
            if framing_wrong:
                violations.append(
                    f"shot {index + 1} framing still prioritizes {framing_wrong[0]} "
                    f"while {active} owns its dialogue"
                )
            if camera_wrong:
                violations.append(
                    f"shot {index + 1} camera still prioritizes {camera_wrong[0]} "
                    f"while {active} owns its dialogue"
                )
        for line in dialogue:
            delivery = sanitize_h3_prompt_text(line.get("delivery"))
            if re.search(r"[\"\u201c\u201d]|<\/?d\b", delivery, flags=re.IGNORECASE):
                violations.append(
                    f"shot {index + 1} contains quoted dialogue inside delivery direction"
                )
                break
        for future_name in future_cast or []:
            if any(_speaker_name_present(field, future_name) for field in (
                framing,
                camera,
            )):
                violations.append(
                    f"shot {index + 1} shows future entrant {future_name} before introduction"
                )
                break
            if _speaker_name_present(shot.get("sound_effects"), future_name):
                violations.append(
                    f"shot {index + 1} gives future entrant {future_name} an early sound cue"
                )
                break
    return violations


def _repair_materialized_segment_staging(
    segment: dict[str, Any],
    *,
    known_speakers: list[str],
    future_cast: list[str] | None = None,
) -> dict[str, Any]:
    """Apply a final deterministic safety net without changing story action."""

    repaired = deepcopy(segment)
    future_names = [
        sanitize_h3_prompt_text(name)
        for name in (future_cast or [])
        if sanitize_h3_prompt_text(name)
    ]
    for shot in repaired.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        framing = sanitize_h3_prompt_text(shot.get("framing"))
        camera = sanitize_h3_prompt_text(shot.get("camera"))
        dialogue = [
            item for item in (shot.get("dialogue") or [])
            if isinstance(item, dict)
        ]
        visible_speakers = list(dict.fromkeys(
            sanitize_h3_prompt_text(item.get("speaker"))
            for item in dialogue
            if sanitize_h3_prompt_text(item.get("speaker"))
            and "off-camera" not in sanitize_h3_prompt_text(
                item.get("action")
            ).casefold()
        ))
        active = visible_speakers[0] if len(visible_speakers) == 1 else ""
        if any(_speaker_name_present(framing, name) for name in future_names):
            framing = (
                f"medium shot on {active} in the existing composition; other "
                "already-present subjects remain in their established positions"
                if active else
                "the same established composition with only already-introduced subjects"
            )
        if any(_speaker_name_present(camera, name) for name in future_names):
            camera = (
                f"maintain the existing blocking and settle on {active} before "
                "the vocal line begins"
                if active else
                "maintain the existing blocking and follow only already-introduced subjects"
            )
        if any(
            _speaker_name_present(shot.get("sound_effects"), name)
            for name in future_names
        ):
            shot["sound_effects"] = (
                "Natural synchronized nonverbal effects for the visible action"
            )
        if len(visible_speakers) == 1:
            framing_wrong = any(
                other.casefold() != active.casefold()
                and _speaker_is_camera_focus(framing, other, known_speakers)
                for other in known_speakers
            )
            camera_wrong = any(
                other.casefold() != active.casefold()
                and _speaker_is_camera_focus(camera, other, known_speakers)
                for other in known_speakers
            )
            if framing_wrong:
                framing = (
                    "medium scene composition in the established target setting; "
                    f"{active} carries the visible speaking performance and "
                    "listeners remain visually secondary"
                )
            elif not _speaker_name_present(framing, active):
                framing = (
                    f"{framing}; {active} carries the visible speaking performance "
                    "and listeners remain visually secondary"
                )
            if camera_wrong:
                camera = (
                    "maintain the established target-scene camera coverage; settle on "
                    f"{active} before the vocal line begins, then show listener "
                    "reactions only after the line ends"
                )
            elif not _speaker_name_present(camera, active):
                camera = (
                    f"{camera}; settle on {active} before the vocal line begins, "
                    "then show listener reactions only after the line ends"
                )
        for line in dialogue:
            delivery = sanitize_h3_prompt_text(line.get("delivery"))
            if re.search(r"[\"\u201c\u201d]|<\/?d\b", delivery, flags=re.IGNORECASE):
                line["delivery"] = "speaks naturally"
        shot["framing"] = framing
        shot["camera"] = camera
    return repaired


def _h3_visible_action_seconds(
    bucket: list[dict[str, Any]],
    *,
    event_floor: int,
) -> float:
    """Estimate a filmable duration for immutable visible story action."""

    action_text = ". Then ".join(
        sanitize_h3_prompt_text(beat.get("description")) for beat in bucket
    )
    travel_count = len(_H3_FULL_BODY_MOTION_RE.findall(action_text))
    blocking_count = len(_H3_BLOCKING_CHANGE_RE.findall(action_text))
    reaction_count = len(_H3_VISIBLE_REACTION_RE.findall(action_text))
    impact_count = len(_H3_IMPACT_ACTION_RE.findall(action_text))
    physical_count = travel_count + blocking_count + reaction_count + impact_count
    if not physical_count:
        return float(event_floor)
    estimate = (
        0.75
        + travel_count * 2.0
        + blocking_count * 1.25
        + reaction_count * 0.9
        + impact_count * 1.2
    )
    # A fallback shot should remain filmable, but must not monopolize an
    # entire H3 window merely because a long action sentence repeats verbs.
    return max(float(event_floor), min(6.5, estimate))


def _h3_filmable_timing_weight(
    bucket: list[dict[str, Any]],
    dialogue_map: dict[str, dict[str, Any]],
) -> float:
    """Return the physical-or-spoken time floor for a group of story beats."""

    event_count = max(
        1,
        sum(max(1, len(beat.get("source_event_ids") or [])) for beat in bucket),
    )
    dialogue_ids = [
        str(dialogue_id or "").upper()
        for beat in bucket
        for dialogue_id in (beat.get("dialogue_ids") or [])
    ]
    dialogue_words = sum(
        len(re.findall(
            r"\b[\w'’-]+\b",
            str(dialogue_map.get(dialogue_id, {}).get("text") or ""),
        ))
        for dialogue_id in dialogue_ids
    )
    spoken_time = dialogue_words / 2.35 + len(dialogue_ids) * 0.35
    action_time = _h3_visible_action_seconds(bucket, event_floor=event_count)
    return max(float(event_count), action_time, spoken_time, 1.0)


def _apply_h3_filmable_shot_clock(
    shots: list[dict[str, Any]],
    assignments: list[list[dict[str, Any]]],
    dialogue_map: dict[str, dict[str, Any]],
    *,
    duration: float,
    only_when_squeezed: bool,
) -> None:
    """Give compound choreography and speech enough relative screen time."""

    if not shots:
        return
    total = max(0.1, float(duration))
    weights = [
        _h3_filmable_timing_weight(bucket, dialogue_map)
        if bucket else 1.0
        for bucket in assignments
    ]
    if only_when_squeezed:
        squeezed = False
        for shot, weight in zip(shots, weights):
            try:
                shot_duration = (
                    float(shot.get("end_seconds"))
                    - float(shot.get("start_seconds"))
                )
            except (TypeError, ValueError):
                squeezed = True
                break
            # A little tolerance preserves deliberate camera rhythms. Reflow
            # only when a proposed clock clearly cannot contain its immutable
            # action or exact dialogue.
            if shot_duration + 0.2 < min(total, weight):
                squeezed = True
                break
        if not squeezed:
            return

    total_weight = max(1.0, sum(weights))
    elapsed_weight = 0.0
    for index, (shot, weight) in enumerate(zip(shots, weights)):
        start = total * elapsed_weight / total_weight
        elapsed_weight += weight
        end = total if index + 1 == len(shots) else total * elapsed_weight / total_weight
        shot["start_seconds"] = round(start, 3)
        shot["end_seconds"] = round(end, 3)


def _cap_h3_opening_dialogue_lead(
    shots: list[dict[str, Any]],
    assignments: list[list[dict[str, Any]]],
    *,
    duration: float,
    dialogue_id: str,
) -> None:
    """Keep an immediate first requested line within the opening seconds."""

    target = str(dialogue_id or "").upper()
    if not target or not shots or len(shots) != len(assignments):
        return
    dialogue_shot = next(
        (
            index
            for index, bucket in enumerate(assignments)
            if any(
                target == str(dialogue_value or "").upper()
                for beat in bucket
                for dialogue_value in (beat.get("dialogue_ids") or [])
            )
        ),
        None,
    )
    if dialogue_shot is None or dialogue_shot == 0:
        return
    total = max(0.1, float(duration))
    lead_cap = min(4.5, max(2.0, total * 0.30))
    try:
        current_lead = float(
            shots[dialogue_shot].get("start_seconds") or 0.0
        )
    except (TypeError, ValueError):
        current_lead = total
    if current_lead <= lead_cap + 0.05:
        return

    shot_durations: list[float] = []
    for shot in shots:
        try:
            shot_duration = max(
                0.1,
                float(shot.get("end_seconds"))
                - float(shot.get("start_seconds")),
            )
        except (TypeError, ValueError):
            shot_duration = 1.0
        shot_durations.append(shot_duration)
    pre_total = max(0.1, sum(shot_durations[:dialogue_shot]))
    post_total = max(0.1, sum(shot_durations[dialogue_shot:]))
    cursor = 0.0
    for index, shot in enumerate(shots):
        local_total = lead_cap if index < dialogue_shot else total - lead_cap
        denominator = pre_total if index < dialogue_shot else post_total
        shot_duration = local_total * shot_durations[index] / denominator
        start = cursor
        cursor = total if index + 1 == len(shots) else cursor + shot_duration
        shot["start_seconds"] = round(start, 3)
        shot["end_seconds"] = round(cursor, 3)


def _fallback_segment(
    segment_number: int,
    *,
    duration: float,
    beats: list[dict[str, Any]],
    opening_state: str,
    camera_coverage: str,
    dialogue_catalog: list[dict[str, Any]] | None = None,
    source_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent = source_intent or {}
    locked_pov = bool(intent.get("first_person_pov"))
    # A continuous POV can still have multiple timed phases. Collapsing all
    # beats into one giant shot caused the compiler's safety compaction to cut
    # off launch and travel actions at the end of a busy window. Keep separate
    # phases, but describe each transition as an in-viewpoint reframe.
    shot_count = min(_segment_shot_limit(beats), max(1, len(beats)))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shot_count)]
    for index, beat in enumerate(beats):
        buckets[min(shot_count - 1, index)].append(beat)
    dialogue_map = {
        str(item.get("dialogue_id") or "").upper(): item
        for item in (dialogue_catalog or [])
    }
    weights = [
        _h3_filmable_timing_weight(bucket, dialogue_map)
        for bucket in buckets
    ]
    total_weight = max(1.0, sum(weights))
    elapsed_weight = 0.0
    shots: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        start = duration * elapsed_weight / total_weight
        elapsed_weight += weights[index]
        end = duration * elapsed_weight / total_weight
        beat_ids = [str(item.get("beat_id") or "") for item in bucket]
        dialogue_ids = [
            str(dialogue_id or "")
            for beat in bucket
            for dialogue_id in (beat.get("dialogue_ids") or [])
        ]
        action = ". Then ".join(
            sanitize_h3_prompt_text(item.get("description")) for item in bucket
        )
        shots.append({
            "shot": index + 1,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "transition": (
                "opening composition"
                if index == 0 else
                "without a cut, reframe within the locked first-person POV"
                if locked_pov and camera_coverage != "multi_shot" else
                "hard cut"
            ),
            "framing": (
                "the locked first-person POV with the requested foreground hands and held object"
                if locked_pov and intent.get("hands_visible") else
                "the locked first-person POV from the viewpoint character"
                if locked_pov else
                "a readable wide or medium-wide establishing view"
                if index == 0 else "a motivated medium or close reaction angle"
            ),
            "camera": (
                "a continuous kinetic first-person camera follows the requested motion without cutting outside the viewpoint"
                if locked_pov else
                "a dynamic camera follows the visible action in real time"
                if camera_coverage == "multi_shot"
                else "a coherent motivated camera follows the visible action"
            ),
            "beat_ids": beat_ids,
            "action": action or "The requested event advances visibly",
            "dialogue": [
                {
                    "dialogue_id": dialogue_id,
                    "delivery": sanitize_h3_prompt_text(
                        dialogue_map.get(dialogue_id, {}).get("delivery")
                        or "speaks naturally and clearly"
                    ),
                    "action": (
                        "remaining off-camera at the unseen first-person viewpoint"
                        if dialogue_map.get(dialogue_id, {}).get("off_camera")
                        else "performing the assigned visible action"
                    ),
                }
                for dialogue_id in dialogue_ids
            ],
            "sound_effects": "; ".join(
                sanitize_h3_prompt_text(item.get("sound_effects")) for item in bucket
                if sanitize_h3_prompt_text(item.get("sound_effects")).casefold() not in {"", "n/a", "none"}
            ) or "Natural synchronized effects for the visible action",
        })
    if segment_number == 1 and intent.get("opening_dialogue_id"):
        _cap_h3_opening_dialogue_lead(
            shots,
            buckets,
            duration=duration,
            dialogue_id=str(intent.get("opening_dialogue_id")),
        )
    return {
        "segment": segment_number,
        "title": f"Story segment {segment_number}",
        "opening_state": opening_state,
        "coverage": (
            "continuous locked first-person POV"
            if locked_pov and camera_coverage != "multi_shot" else
            "dynamic multi-shot cinematic coverage"
            if camera_coverage == "multi_shot" else "coherent cinematic coverage"
        ),
        "pacing": sanitize_h3_prompt_text(
            intent.get("pacing_contract")
            or "natural real-time pacing; no slow motion unless requested"
        ),
        "shots": shots,
        "closing_state": sanitize_h3_prompt_text(beats[-1].get("state_after")),
    }


_LONG_FORM_SEGMENTS_PER_CHAPTER = 24


def _plan_long_form_ledger(
    prompt: str,
    *,
    canonical_ledger: dict[str, Any],
    segment_durations: list[float],
    reference_context: str,
    generate: Callable[..., str],
    image_paths: list[str] | None,
    nsfw: bool,
    planning_style: str,
    allow_generated_dialogue: bool,
    locked_dialogue: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Expand a very long H3 concept in bounded chapter calls.

    The ordinary H3 planner intentionally gives every window its own camera
    planning call.  That is excellent for a handful of windows, but an hour
    can contain hundreds.  Long projects instead receive one compact chapter
    outline and one bounded expansion call per chapter.  Maestro still owns
    source-event order, exact dialogue IDs, and state handoffs; the LLM only
    supplies new visible progression between those immutable anchors.
    """

    from services.guide_loader import load_guide
    from services.h3_window_planner import _parse_json_object

    durations = [max(0.1, float(value)) for value in segment_durations]
    segment_count = len(durations)
    chapter_ranges = [
        (start, min(segment_count, start + _LONG_FORM_SEGMENTS_PER_CHAPTER))
        for start in range(0, segment_count, _LONG_FORM_SEGMENTS_PER_CHAPTER)
    ]
    chapter_count = len(chapter_ranges)
    warnings: list[str] = []
    planning_style = normalize_h3_planning_style(planning_style)
    guide = load_guide("enhance", "minimax_h3_story_ledger")
    if nsfw:
        guide += (
            "\n\nMATURE-MODE FIDELITY\nPreserve explicitly requested mature "
            "material. Do not censor it, add to it, or intensify it."
        )

    source_events = extract_source_events(prompt)
    source_cast_names = list(
        (canonical_ledger.get("source_intent") or {}).get("cast_names") or []
    )
    story_bible = build_long_form_story_bible_fallback(
        prompt,
        locked_dialogue=locked_dialogue,
        source_events=source_events,
        character_names=source_cast_names,
        chapter_count=chapter_count,
    )
    story_bible["story_engine"] = sanitize_h3_prompt_text(
        canonical_ledger.get("subject_continuity")
        or story_bible.get("story_engine")
    )
    story_bible["tone_contract"] = sanitize_h3_prompt_text(
        canonical_ledger.get("visual_continuity")
        or story_bible.get("tone_contract")
    )
    story_bible["ending_contract"] = sanitize_h3_prompt_text(
        canonical_ledger.get("required_final_outcome")
        or story_bible.get("ending_contract")
    )

    chapter_schema = {
        "type": "object",
        "properties": {
            "story_bible": LONG_FORM_STORY_BIBLE_SCHEMA,
            "chapters": {
                "type": "array",
                "minItems": chapter_count,
                "maxItems": chapter_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "chapter": {"type": "integer"},
                        "title": {"type": "string"},
                        "location_id": {"type": "string"},
                        "location_time": {"type": "string"},
                        "objective": {"type": "string"},
                        "opening_state": {"type": "string"},
                        "closing_state": {"type": "string"},
                        "continuity_notes": {"type": "string"},
                        "persistent_state": {"type": "string"},
                        "character_state_changes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 30,
                        },
                        "cast_present": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 30,
                        },
                    },
                    "required": [
                        "chapter", "title", "location_id", "location_time",
                        "objective", "opening_state", "closing_state",
                        "continuity_notes", "persistent_state",
                        "character_state_changes", "cast_present",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["story_bible", "chapters"],
        "additionalProperties": False,
    }
    chapter_geometry = "\n".join(
        f"- Chapter {index + 1}: segments {start + 1}-{end}; "
        f"{sum(durations[start:end]):.1f} seconds"
        for index, (start, end) in enumerate(chapter_ranges)
    )
    try:
        raw = generate(
            prompt=(
                "Create a compact complete-film story_bible and causal chapter "
                "outline for one continuous long-form video. The story bible "
                "must lock the premise engine, tone, ending, central cast, named "
                "speakers, distinct location registry, and persistent world rules "
                "before the chapters. When the concept asks for many or different "
                "places, worlds, rooms, or encounters, plan enough distinct places "
                "to sustain the full runtime rather than cycling through examples. "
                "Every chapter must advance to a new story state; never "
                "recap, restart, or repeat an earlier action. The final chapter "
                "alone completes the requested outcome. Keep identity, location "
                "logic, visual style, carried objects, injuries, and dialogue "
                "ownership coherent. Record named death, disappearance, injury, "
                "transformation, return, and restoration in character_state_changes. "
                + (
                    "The user concept is a creative brief: invent supporting causal story progression and a satisfying payoff without contradicting its requirements.\n\n"
                    if planning_style == "creative" else
                    "The user concept is locked: do not invent new plot events, outcomes, or dialogue.\n\n"
                )
                + f"CHAPTER GEOMETRY:\n{chapter_geometry}\n\n"
                f"REFERENCE CONTEXT:\n{reference_context or 'None.'}\n\n"
                f"USER CONCEPT:\n{prompt}"
            ),
            system_prompt=guide,
            max_new_tokens=min(7600, 1800 + chapter_count * 330),
            temperature=0.38 if planning_style == "creative" else 0.24,
            top_p=0.86,
            image_paths=image_paths or None,
            enable_thinking=False,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            json_schema=chapter_schema,
        )
        parsed = _parse_json_object(raw)
        story_bible = normalize_long_form_story_bible(
            parsed.get("story_bible") if isinstance(parsed, dict) else None,
            story_description=prompt,
            locked_dialogue=locked_dialogue,
            source_events=source_events,
            character_names=source_cast_names,
            chapter_count=chapter_count,
        )
        chapters = parsed.get("chapters") if isinstance(parsed, dict) else None
        if not isinstance(chapters, list) or len(chapters) != chapter_count:
            raise ValueError("chapter outline count did not match geometry")
    except Exception as error:
        print(f"[MiniMax H3] Long-form chapter-outline fallback: {error}")
        warnings.append(
            "The long-form chapter outline could not be expanded creatively, "
            "so Maestro retained its ordered duration-aware chapter scaffold."
        )
        chapters = [
            {
                "chapter": index + 1,
                "title": f"Chapter {index + 1}",
                "location_id": f"chapter_{index + 1}_location",
                "location_time": "Continue the established world and story time",
                "objective": (
                    f"Advance the requested concept through chapter {index + 1} "
                    f"of {chapter_count} without replaying earlier action"
                ),
                "opening_state": (
                    canonical_ledger.get("initial_state")
                    if index == 0 else
                    f"The visible result of chapter {index} carries forward"
                ),
                "closing_state": (
                    canonical_ledger.get("required_final_outcome")
                    if index + 1 == chapter_count else
                    f"A concrete new handoff into chapter {index + 2}"
                ),
                "continuity_notes": "Preserve every established visual and story state",
                "persistent_state": "Preserve accumulated identity, prop, relationship, and physical state",
                "character_state_changes": [],
                "cast_present": source_cast_names,
            }
            for index in range(chapter_count)
        ]

    chapters, story_bible = normalize_long_form_outline(
        chapters,
        story_bible=story_bible,
        chapter_count=chapter_count,
    )
    chapters, location_coverage_warnings = ensure_long_form_location_coverage(
        chapters,
        story_bible=story_bible,
    )
    if location_coverage_warnings:
        print(
            "[MiniMax H3] Long-form location coverage repair: "
            + "; ".join(location_coverage_warnings)
        )

    canonical_by_segment: dict[int, list[dict[str, Any]]] = {}
    for beat in canonical_ledger.get("beats") or []:
        if not isinstance(beat, dict):
            continue
        canonical_by_segment.setdefault(int(beat.get("segment") or 0), []).append(beat)

    expanded_beats: list[dict[str, Any]] = []
    generated_dialogue: list[dict[str, Any]] = []
    next_dialogue_number = len(locked_dialogue) + 1
    locked_dialogue_words = {
        str(item.get("dialogue_id") or "").upper(): len(
            re.findall(r"\b[\w'’-]+\b", str(item.get("text") or ""))
        )
        for item in locked_dialogue
    }
    beat_number = 0
    previous_state = sanitize_h3_prompt_text(canonical_ledger.get("initial_state"))
    for chapter_index, ((start, end), chapter) in enumerate(
        zip(chapter_ranges, chapters),
        start=1,
    ):
        local_count = end - start
        obligations: list[dict[str, Any]] = []
        for absolute_index in range(start, end):
            segment_number = absolute_index + 1
            mandatory = canonical_by_segment.get(segment_number, [])
            obligations.append({
                "window": segment_number,
                "duration_seconds": round(durations[absolute_index], 3),
                "required_events": [
                    sanitize_h3_prompt_text(item.get("description"))
                    for item in mandatory
                    if item.get("source_event_ids")
                ],
                "dialogue_ids": [
                    str(dialogue_id or "").upper()
                    for item in mandatory
                    for dialogue_id in (item.get("dialogue_ids") or [])
                ],
                "dialogue_word_budget": max(
                    0,
                    int(math.floor(durations[absolute_index] * 2.1))
                    - sum(
                        locked_dialogue_words.get(
                            str(dialogue_id or "").upper(),
                            0,
                        )
                        for item in mandatory
                        for dialogue_id in (item.get("dialogue_ids") or [])
                    ),
                ),
            })
        generated_line_schema = {
            "type": "object",
            "properties": {
                "speaker": {"type": "string"},
                "language": {"type": "string"},
                "delivery": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["speaker", "language", "delivery", "text"],
            "additionalProperties": False,
        }
        segment_schema = {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "minItems": local_count,
                    "maxItems": local_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "window": {"type": "integer"},
                            "supporting_progression": {"type": "string"},
                            "resulting_state": {"type": "string"},
                            "sound_effects": {"type": "string"},
                            "dialogue": {
                                "type": "array",
                                "items": generated_line_schema,
                                "maxItems": 2 if allow_generated_dialogue else 0,
                            },
                        },
                        "required": [
                            "window", "supporting_progression",
                            "resulting_state", "sound_effects", "dialogue",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["segments"],
            "additionalProperties": False,
        }
        next_chapter = chapters[chapter_index] if chapter_index < chapter_count else None
        chapter_bible_context = format_long_form_story_bible(
            story_bible,
            cast_names=chapter.get("cast_present") or [],
            location_ids=[
                value.get("location_id")
                for value in (
                    chapters[chapter_index - 2] if chapter_index > 1 else None,
                    chapter,
                    next_chapter,
                )
                if isinstance(value, dict) and value.get("location_id")
            ],
        )
        try:
            raw = generate(
                prompt=(
                    f"Expand chapter {chapter_index} of {chapter_count} into "
                    f"exactly {local_count} chronological video segments. Each "
                    "segment must create new visible action and a concrete ending "
                    "state. Never recap, preview a later beat, replay an action, "
                    "or put dialogue words in supporting_progression. Required "
                    "events and dialogue IDs are immutable anchors that Maestro "
                    "will insert separately; write only staging/progression around "
                    "them. Make the last resulting_state flow directly into the "
                    "next chapter.\n\n"
                + (
                        "Write short, character-specific dialogue in each segment's dialogue array when it advances the interaction. Respect dialogue_word_budget exactly; use an empty array when no line belongs there. Exact quoted lines are inserted separately and must not be repeated.\n\n"
                        if allow_generated_dialogue else
                        "Every dialogue array must be empty; do not invent spoken words.\n\n"
                    )
                    + f"BINDING STORY BIBLE:\n{chapter_bible_context}\n\n"
                    + f"CHAPTER PLAN:\n{json.dumps(chapter, ensure_ascii=False, indent=2)}\n\n"
                    f"PREVIOUS VISIBLE STATE:\n{previous_state}\n\n"
                    f"NEXT CHAPTER (for handoff only):\n"
                    f"{json.dumps(next_chapter, ensure_ascii=False, indent=2) if next_chapter else 'This is the final chapter.'}\n\n"
                    f"SEGMENT OBLIGATIONS:\n{json.dumps(obligations, ensure_ascii=False, indent=2)}\n\n"
                    f"GLOBAL USER CONCEPT:\n{prompt}"
                ),
                system_prompt=guide,
                max_new_tokens=min(7200, 900 + local_count * 235),
                temperature=0.4 if planning_style == "creative" else 0.28,
                top_p=0.88,
                image_paths=None,
                enable_thinking=False,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                json_schema=segment_schema,
            )
            parsed = _parse_json_object(raw)
            local_segments = parsed.get("segments") if isinstance(parsed, dict) else None
            if not isinstance(local_segments, list) or len(local_segments) != local_count:
                raise ValueError("chapter window count did not match geometry")
        except Exception as error:
            print(
                f"[MiniMax H3] Long-form chapter {chapter_index} fallback: {error}"
            )
            warnings.append(
                f"Long-form chapter {chapter_index}'s creative expansion failed; "
                "Maestro used its deterministic window progression for that chapter."
            )
            local_segments = [
                {
                    "window": absolute_index + 1,
                    "supporting_progression": "",
                    "resulting_state": "",
                    "sound_effects": "",
                    "dialogue": [],
                }
                for absolute_index in range(start, end)
            ]

        for offset, proposed in enumerate(local_segments):
            segment_number = start + offset + 1
            mandatory = canonical_by_segment.get(segment_number, [])
            mandatory_description = ". Then ".join(
                sanitize_h3_prompt_text(item.get("description"))
                for item in mandatory
                if sanitize_h3_prompt_text(item.get("description"))
            )
            supporting = sanitize_h3_prompt_text(
                proposed.get("supporting_progression")
                if isinstance(proposed, dict) else ""
            )
            if mandatory_description and supporting:
                description = f"{supporting}. Then {mandatory_description}"
            else:
                description = mandatory_description or supporting or (
                    "Advance to a new visible story state without replaying an earlier action"
                )
            resulting_state = sanitize_h3_prompt_text(
                proposed.get("resulting_state")
                if isinstance(proposed, dict) else ""
            ) or sanitize_h3_prompt_text(
                mandatory[-1].get("state_after") if mandatory else ""
            ) or f"A concrete new visible handoff after segment {segment_number}"
            sound_effects = sanitize_h3_prompt_text(
                proposed.get("sound_effects")
                if isinstance(proposed, dict) else ""
            ) or "; ".join(
                sanitize_h3_prompt_text(item.get("sound_effects"))
                for item in mandatory
                if sanitize_h3_prompt_text(item.get("sound_effects"))
            ) or "Natural synchronized effects for the visible action"
            beat_number += 1
            local_dialogue_ids: list[str] = []
            remaining_words = max(
                0,
                int(math.floor(durations[segment_number - 1] * 2.1))
                - sum(
                    locked_dialogue_words.get(
                        str(dialogue_id or "").upper(),
                        0,
                    )
                    for item in mandatory
                    for dialogue_id in (item.get("dialogue_ids") or [])
                ),
            )
            if allow_generated_dialogue and isinstance(proposed, dict):
                for raw_line in (proposed.get("dialogue") or [])[:2]:
                    if not isinstance(raw_line, dict) or remaining_words <= 0:
                        continue
                    text = sanitize_h3_prompt_text(raw_line.get("text"))
                    words = re.findall(r"\b[\w'’-]+\b", text)
                    if not words:
                        continue
                    if len(words) > remaining_words:
                        text = " ".join(words[:remaining_words]).strip()
                        words = words[:remaining_words]
                    if not text:
                        continue
                    dialogue_id = f"D{next_dialogue_number}"
                    next_dialogue_number += 1
                    generated_dialogue.append({
                        "dialogue_id": dialogue_id,
                        "speaker": sanitize_h3_prompt_text(raw_line.get("speaker")) or "Speaker",
                        "language": sanitize_h3_prompt_text(raw_line.get("language")) or "English",
                        "delivery": sanitize_h3_prompt_text(raw_line.get("delivery")) or "speaks naturally and clearly",
                        "text": text,
                        "segment": segment_number,
                    })
                    local_dialogue_ids.append(dialogue_id)
                    remaining_words -= len(words)
            expanded_beats.append({
                "beat_id": f"B{beat_number}",
                "segment": segment_number,
                "description": description,
                "source_event_ids": [
                    str(event_id or "").upper()
                    for item in mandatory
                    for event_id in (item.get("source_event_ids") or [])
                ],
                "dialogue_ids": [
                    str(dialogue_id or "").upper()
                    for item in mandatory
                    for dialogue_id in (item.get("dialogue_ids") or [])
                ] + local_dialogue_ids,
                "state_after": resulting_state,
                "sound_effects": sound_effects,
            })
            previous_state = resulting_state

    ledger = deepcopy(canonical_ledger)
    ledger["beats"] = expanded_beats
    ledger["generated_dialogue"] = generated_dialogue
    ledger["long_form_story_bible"] = story_bible
    ledger["long_form_chapters"] = chapters
    ledger["long_form_location_repairs"] = location_coverage_warnings
    if chapters:
        ledger["initial_state"] = sanitize_h3_prompt_text(
            chapters[0].get("opening_state")
        ) or ledger.get("initial_state")
        ledger["required_final_outcome"] = sanitize_h3_prompt_text(
            chapters[-1].get("closing_state")
        ) or ledger.get("required_final_outcome")
    return ledger, list(dict.fromkeys(warnings))


def _materialize_segment(
    segment: dict[str, Any],
    *,
    beats: list[dict[str, Any]],
    dialogue_catalog: list[dict[str, Any]],
    source_events: list[dict[str, str]],
    future_cast: list[str] | None = None,
) -> dict[str, Any]:
    dialogue_map = {
        str(item.get("dialogue_id") or "").upper(): item
        for item in dialogue_catalog
    }
    beat_map = {
        str(item.get("beat_id") or "").upper(): item
        for item in beats
    }
    event_map = {
        item["event_id"]: _filmable_source_event(item["text"])
        for item in source_events
    }
    known_speakers = list(dict.fromkeys(
        sanitize_h3_prompt_text(item.get("speaker")) or "Speaker"
        for item in dialogue_catalog
    ))
    shots: list[dict[str, Any]] = []
    for shot in segment.get("shots") or []:
        dialogue: list[dict[str, Any]] = []
        dialogue_sources: list[dict[str, Any]] = []
        for performance in shot.get("dialogue") or []:
            dialogue_id = str(performance.get("dialogue_id") or "").upper()
            source = dialogue_map[dialogue_id]
            dialogue_sources.append(source)
            speaker = sanitize_h3_prompt_text(source.get("speaker")) or "Speaker"
            off_camera = bool(source.get("off_camera"))
            dialogue.append({
                "speaker": speaker,
                "speaker_id": sanitize_h3_prompt_text(source.get("speaker_id")) or "S1",
                "language": sanitize_h3_prompt_text(source.get("language")) or "English",
                "delivery": (
                    sanitize_h3_prompt_text(performance.get("delivery"))
                    or sanitize_h3_prompt_text(source.get("delivery"))
                    or "speaks naturally"
                ),
                "action": (
                    "off-camera in the established target scene while every visible "
                    "mouth stays closed"
                    if off_camera else
                    f"in the established target scene, only {speaker}'s mouth moves while "
                    "every other visible mouth stays closed"
                ),
                # The text is inserted from the locked catalog, never copied
                # from the segment LLM response.
                "text": sanitize_h3_prompt_text(source.get("text")),
                "dialogue_id": dialogue_id,
            })
        required_events = [
            event_map[str(event_id or "").upper()]
            for beat_id in (shot.get("beat_ids") or [])
            for event_id in (
                beat_map.get(str(beat_id or "").upper(), {}).get("source_event_ids") or []
            )
            if str(event_id or "").upper() in event_map
        ]
        proposed_action = sanitize_h3_prompt_text(shot.get("action"))
        exact_action = ". Then ".join(required_events)
        if exact_action and _normalize_key(exact_action) not in _normalize_key(proposed_action):
            proposed_action = f"{exact_action}. {proposed_action}".strip()
        framing, camera, proposed_action = _enforce_materialized_vocal_staging(
            framing=sanitize_h3_prompt_text(shot.get("framing")),
            camera=sanitize_h3_prompt_text(shot.get("camera")),
            action=proposed_action,
            dialogue_sources=dialogue_sources,
            known_speakers=known_speakers,
            future_cast=future_cast,
        )
        sound_effects = sanitize_h3_prompt_text(shot.get("sound_effects"))
        if any(
            _speaker_name_present(sound_effects, name)
            for name in (future_cast or [])
        ):
            sound_effects = (
                "Natural synchronized nonverbal effects for the visible action"
            )
        shots.append({
            "shot": int(shot.get("shot") or len(shots) + 1),
            "start_seconds": float(shot.get("start_seconds") or 0.0),
            "end_seconds": float(shot.get("end_seconds") or 0.0),
            "transition": sanitize_h3_prompt_text(shot.get("transition")),
            "framing": framing,
            "camera": camera,
            "action": proposed_action,
            "dialogue": dialogue,
            "sound_effects": sound_effects,
        })
    shots = _split_h3_shots_at_speaker_changes(
        shots,
        known_speakers=known_speakers,
        future_cast=future_cast,
    )
    summary = "; then ".join(
        sanitize_h3_prompt_text(item.get("description")) for item in beats
    )
    return {
        "segment": int(segment.get("segment") or 0),
        "title": sanitize_h3_prompt_text(segment.get("title")),
        "summary": summary,
        "opening_state": sanitize_h3_prompt_text(segment.get("opening_state")),
        "coverage": sanitize_h3_prompt_text(segment.get("coverage")),
        "pacing": sanitize_h3_prompt_text(segment.get("pacing")),
        "shots": shots,
        "closing_state": (
            sanitize_h3_prompt_text(beats[-1].get("state_after"))
            if beats else sanitize_h3_prompt_text(segment.get("closing_state"))
        ),
    }


def _split_h3_shots_at_speaker_changes(
    shots: list[dict[str, Any]],
    *,
    known_speakers: list[str],
    future_cast: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Give every visible speaker change its own timed H3 camera phase.

    Exact-dialogue continuation can add a new speaker fragment after the
    four-shot camera plan has already been validated.  Leaving two different
    visible speakers in one shot gives H3 one face but two transcripts, and it
    can keep lip-syncing the first face to the second person's words.  Split
    only at an actual speaker change; consecutive lines from the same speaker
    remain together and ordinary action-only shots are untouched.
    """

    expanded: list[dict[str, Any]] = []
    for shot in shots:
        dialogue = [
            dict(item) for item in (shot.get("dialogue") or [])
            if isinstance(item, dict)
        ]
        groups: list[list[dict[str, Any]]] = []
        for line in dialogue:
            speaker_key = (
                sanitize_h3_prompt_text(line.get("speaker_id"))
                or sanitize_h3_prompt_text(line.get("speaker"))
                or "speaker"
            ).casefold()
            previous_key = ""
            if groups:
                previous = groups[-1][-1]
                previous_key = (
                    sanitize_h3_prompt_text(previous.get("speaker_id"))
                    or sanitize_h3_prompt_text(previous.get("speaker"))
                    or "speaker"
                ).casefold()
            if not groups or speaker_key != previous_key:
                groups.append([line])
            else:
                groups[-1].append(line)

        if len(groups) <= 1:
            item = dict(shot)
            item["dialogue"] = dialogue
            expanded.append(item)
            continue

        try:
            start = float(shot.get("start_seconds") or 0.0)
            end = float(shot.get("end_seconds") or start)
        except (TypeError, ValueError):
            start, end = 0.0, float(len(groups))
        total = max(0.1, end - start)
        weights = [
            max(
                1.0,
                sum(_dialogue_word_count(line.get("text")) for line in group)
                / 2.35
                + len(group) * 0.35,
            )
            for group in groups
        ]
        total_weight = max(1.0, sum(weights))
        cursor = start
        original_action = re.sub(
            r"^Visual direction only, never spoken narration:\s*",
            "",
            sanitize_h3_prompt_text(shot.get("action")),
            flags=re.IGNORECASE,
        )
        action_clauses = [
            clause.strip(" .")
            for clause in re.split(r"\s*\.\s*Then\s+|\s+Then\s+", original_action)
            if clause.strip(" .")
        ]
        group_tokens: list[list[str]] = []
        for group in groups:
            group_speaker = sanitize_h3_prompt_text(group[0].get("speaker"))
            group_tokens.append([
                token.casefold()
                for token in re.findall(r"[A-Za-z0-9_'’-]+", group_speaker)
                if len(token) > 1
            ])
        clause_owners: list[int | None] = []
        for clause in action_clauses:
            owner = next(
                (
                    group_index
                    for group_index, tokens in enumerate(group_tokens)
                    if any(
                        re.search(
                            rf"\b{re.escape(token)}\b",
                            clause,
                            flags=re.IGNORECASE,
                        )
                        for token in tokens
                    )
                ),
                None,
            )
            clause_owners.append(owner)
        # A shared/pronoun-led action ("they mount their brooms") still
        # belongs to the timeline. Attach it to the nearest named speaker
        # phase without duplicating or reordering it.
        for clause_index, owner in enumerate(clause_owners):
            if owner is not None:
                continue
            following = next(
                (
                    candidate
                    for candidate in clause_owners[clause_index + 1:]
                    if candidate is not None
                ),
                None,
            )
            preceding = next(
                (
                    candidate
                    for candidate in reversed(clause_owners[:clause_index])
                    if candidate is not None
                ),
                None,
            )
            clause_owners[clause_index] = (
                following if following is not None else
                preceding if preceding is not None else 0
            )
        group_actions: list[list[str]] = [[] for _ in groups]
        for clause, owner in zip(action_clauses, clause_owners):
            group_actions[int(owner or 0)].append(clause)
        for group_index, (group, weight) in enumerate(zip(groups, weights)):
            speaker = sanitize_h3_prompt_text(group[0].get("speaker")) or "The speaker"
            local_action = ". Then ".join(group_actions[group_index]) or (
                f"{speaker} visibly delivers only the assigned dialogue line"
            )
            off_camera = all(
                "off-camera" in str(line.get("action") or "").casefold()
                for line in group
            )
            framing, camera, action = _enforce_materialized_vocal_staging(
                framing=sanitize_h3_prompt_text(shot.get("framing")),
                camera=sanitize_h3_prompt_text(shot.get("camera")),
                action=local_action,
                dialogue_sources=[{
                    "speaker": speaker,
                    "off_camera": off_camera,
                }],
                known_speakers=known_speakers,
                future_cast=future_cast,
            )
            next_cursor = (
                end
                if group_index + 1 == len(groups)
                else cursor + total * weight / total_weight
            )
            expanded.append({
                **shot,
                "start_seconds": round(cursor, 3),
                "end_seconds": round(next_cursor, 3),
                "transition": (
                    sanitize_h3_prompt_text(shot.get("transition"))
                    if group_index == 0 else "hard cut"
                ),
                "framing": framing,
                "camera": camera,
                "action": action,
                "dialogue": group,
                "sound_effects": (
                    sanitize_h3_prompt_text(shot.get("sound_effects"))
                    if group_index == 0 else
                    "Natural synchronized effects for the visible performance"
                ),
            })
            cursor = next_cursor

    for index, shot in enumerate(expanded, start=1):
        shot["shot"] = index
    return expanded


def plan_h3_story_segments(
    prompt: str,
    *,
    segment_durations: list[float],
    mode: str,
    camera_coverage: str,
    reference_context: str = "",
    expect_dialogue: bool = False,
    planning_style: str = "faithful",
    image_paths: list[str] | None = None,
    nsfw: bool = False,
    llm_generate: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Create a compact ledger and expand one validated local segment at a time."""

    from services import llm_service
    from services.guide_loader import load_guide

    generate = llm_generate or llm_service.generate
    durations = [max(0.1, float(value)) for value in segment_durations]
    segment_count = len(durations)
    if not segment_count:
        raise ValueError("H3 story planning requires at least one segment.")
    locked_dialogue = extract_locked_dialogue(prompt)
    # Callers historically detected only quotation marks.  Treat any
    # canonical dialogue form, including ``CHARACTER: line`` screenplay rows,
    # as mandatory even when an older caller passes ``expect_dialogue=False``.
    expect_dialogue = bool(expect_dialogue or locked_dialogue)
    source_events = extract_source_events(prompt)
    source_intent = extract_h3_source_intent(prompt)
    source_cast_names = _merge_h3_cast_names(
        list(source_intent.get("cast_names") or []),
        _reference_h3_cast_names(reference_context),
        prompt=prompt,
    )
    source_intent["cast_names"] = source_cast_names
    source_intent["cast_cardinality_contract"] = _h3_cast_cardinality_contract(
        prompt,
        source_cast_names,
    )
    locked_dialogue = _canonicalize_h3_dialogue_speakers(
        locked_dialogue,
        source_cast_names,
    )
    planning_style = normalize_h3_planning_style(planning_style)
    planning_warnings: list[str] = []
    planning_diagnostics: list[str] = []
    planning_notes: list[str] = []
    allow_generated_dialogue = bool(
        planning_style == "creative"
        and expect_dialogue
        and not _only_supplied_dialogue_requested(prompt)
    )
    faithful_locked_schedule = bool(
        planning_style == "faithful" and locked_dialogue
    )
    spread_generated_dialogue = bool(
        allow_generated_dialogue
        and not locked_dialogue
        and segment_count > 1
        and segment_count <= _LONG_FORM_SEGMENTS_PER_CHAPTER
        and _creative_conversation_brief(prompt)
    )
    canonical_ledger = _deterministic_ledger(
        prompt,
        segment_count=segment_count,
        segment_durations=durations,
        locked_dialogue=locked_dialogue,
        camera_coverage=camera_coverage,
        reference_context=reference_context,
    )
    source_intent = dict(canonical_ledger.get("source_intent") or source_intent)
    expected_dialogue_events = _expected_dialogue_events(prompt, locked_dialogue)
    dialogue_by_event: dict[str, list[str]] = {}
    for dialogue_id, event_id in expected_dialogue_events.items():
        dialogue_by_event.setdefault(event_id, []).append(dialogue_id)
    source_event_lines = "\n".join(
        "- {event_id}: {text}{dialogue}".format(
            event_id=item["event_id"],
            text=item["text"],
            dialogue=(
                "; carries locked dialogue "
                + ", ".join(dialogue_by_event.get(item["event_id"], []))
                if dialogue_by_event.get(item["event_id"])
                else ""
            ),
        )
        for item in source_events
    )
    dialogue_lines = "\n".join(
        "- {dialogue_id}: speaker={speaker}; exact text={text}; anchored source event={event}".format(
            dialogue_id=item["dialogue_id"],
            speaker=item["speaker"],
            text=json.dumps(item["text"], ensure_ascii=False),
            event=expected_dialogue_events.get(item["dialogue_id"], "unanchored; assign once in story order"),
        )
        for item in locked_dialogue
    ) or "- None."
    geometry_lines = "\n".join(
        f"- Segment {index + 1}: {duration:.3f} seconds; total dialogue budget "
        f"at most {max(1, int(math.floor(duration * 2.1)))} spoken words"
        for index, duration in enumerate(durations)
    )
    maximum_window_words = max(
        [max(1, int(math.floor(duration * 2.1))) for duration in durations]
        or [1]
    )
    mechanically_continued_dialogue = [
        str(item.get("dialogue_id") or "").upper()
        for item in locked_dialogue
        if _dialogue_word_count(item.get("text")) > maximum_window_words
    ]
    continuation_instruction = (
        "\nLONG EXACT DIALOGUE CONTINUATION: Maestro will mechanically divide "
        + ", ".join(mechanically_continued_dialogue)
        + " across adjacent native segments after semantic validation. Keep each "
        "listed D-id atomic on its anchored source event in this JSON; do not "
        "rewrite, shorten, duplicate, split, or move it to evade a local word "
        "budget. Other dialogue must fit its assigned segment.\n"
        if mechanically_continued_dialogue else ""
    )
    opening_dialogue_instruction = (
        "\nOPENING PERFORMANCE: "
        f"{source_intent.get('opening_dialogue_id')} belongs in segment 1. "
        "Stage any brief entrance or establishing action in only the first few "
        "seconds, then begin that exact line. Never devote the complete opening "
        "segment to silent setup.\n"
        if source_intent.get("opening_dialogue_id") else ""
    )
    ledger_prompt = (
        f"Mode: {mode}. Camera coverage preference: {camera_coverage}.\n"
        f"Segment geometry:\n{geometry_lines}\n\n"
        f"Canonical reference context:\n{reference_context or 'No external reference map.'}\n\n"
        "APPLICATION-OWNED CAST AND BLOCKING CONTRACT (preserve exactly; only assigned active principals appear in a segment):\n"
        f"{canonical_ledger.get('subject_continuity')}\n\n"
        "IMMUTABLE SOURCE EVENTS (use every ID exactly once and in this order):\n"
        f"{source_event_lines}\n\n"
        "LOCKED USER DIALOGUE (never rewrite; put every D-id exactly once on its anchored event):\n"
        f"{dialogue_lines}\n\n"
        "MANDATORY OUTPUT CHECKSUM: before returning JSON, flatten source_event_ids "
        f"across all beats and verify the exact result is {[item['event_id'] for item in source_events]}. "
        "Then flatten only locked dialogue_ids across all beats and verify the exact result is "
        f"{[item['dialogue_id'] for item in locked_dialogue]}. Do not return until both arrays match.\n\n"
        f"{continuation_instruction}"
        f"{opening_dialogue_instruction}"
        "DIRECT THE SEMANTIC SCHEDULE. You own filmable beat grouping and segment allocation. "
        "Return beats in chronological order, with one to three beats in every segment. "
        "Do not repeat, recap, preview, omit, or reorder any source event. When multiple explicit E-ids exist, place the final E-id in the final segment; "
        "a single broad E-id may begin earlier and develop through concrete derived beats. "
        "Keep each locked dialogue ID with its anchored source event. Except for any D-id explicitly listed for mechanical "
        "continuation above, keep dialogue within that segment's total spoken-word budget. "
        "Use state_after to describe a concrete visual handoff into the next segment, not a generic continuation phrase.\n"
        f"Writing mode: {planning_style.upper()}. "
        + (
            "Treat the user concept as a creative brief. Build one causal full-duration scene with a clear opening, escalation, and payoff. "
            "You may create supporting filmable beats with empty source_event_ids, but never contradict, repeat, or complete an explicit source event early. "
            "When dialogue is allowed, this concept requires an audible authored script: never leave speaking, telling, explaining, discussing, or reacting implicit in visual prose, "
            "and never return generated_dialogue empty. Convert unquoted conversational content into concise, character-specific spoken lines. "
            "Give interacting characters natural back-and-forth and a verbal response when the brief establishes one. Exact quoted lines are immutable anchors; "
            "additional lines may surround them unless the user explicitly says only those lines. Every generated line must fit its segment's spoken-word budget. "
            if planning_style == "creative" else
            "Treat the user concept as locked source material. Distribute and stage only supplied events and exact dialogue; do not invent new plot events, outcomes, or spoken lines. "
        )
        + f"Dialogue policy: {'Add concise generated_dialogue entries that make the requested interaction feel authored and complete; select exactly one segment for each.' if allow_generated_dialogue else 'Do not add generated dialogue.'} "
        + (
            "This is a conversation-first brief: begin intelligible tagged dialogue in segment 1 and author at least one concise line for every segment. Do not spend a complete opening segment on silent setup; a brief entrance or establishing action may occupy only the first few seconds before speech begins.\n\n"
            if spread_generated_dialogue else "\n\n"
        )
        + "Also return shared continuity, setting, visual language, editing style, opening state, nonverbal ambience, music, and the visible final outcome.\n\n"
        + f"User concept:\n{prompt}"
    )
    if faithful_locked_schedule:
        ledger_prompt = (
            f"Mode: {mode}. Camera coverage preference: {camera_coverage}.\n"
            f"Total native segments: {segment_count}.\n\n"
            f"Canonical reference context:\n{reference_context or 'No external reference map.'}\n\n"
            "Maestro has already parsed, ordered, assigned, and timed every "
            "source event and exact dialogue line. Do not return a story "
            "schedule, IDs, beats, dialogue, or shot timings. Supply only a "
            "concise global cinematic treatment for the immutable story. "
            "Preserve the exact cast, setting, actions, tone, and outcome; do "
            "not add a character, plot event, effect, location, or spoken line.\n\n"
            f"Application-owned cast contract:\n{canonical_ledger.get('subject_continuity')}\n\n"
            f"Application-owned opening state:\n{canonical_ledger.get('initial_state')}\n\n"
            f"Application-owned final outcome:\n{canonical_ledger.get('required_final_outcome')}\n\n"
            f"User concept:\n{prompt}"
        )
    ledger_guide = load_guide(
        "enhance",
        "minimax_h3_story_treatment"
        if faithful_locked_schedule else
        "minimax_h3_story_ledger",
    )
    if nsfw:
        ledger_guide += (
            "\n\nMATURE-MODE FIDELITY\nPreserve explicitly requested mature material. "
            "Do not censor it, add to it, or intensify it."
        )
    ledger_schema = (
        _faithful_treatment_schema()
        if faithful_locked_schedule else
        _ledger_schema(
            segment_count,
            source_event_count=len(source_events),
            locked_dialogue_count=len(locked_dialogue),
            allow_generated_dialogue=allow_generated_dialogue,
            minimum_generated_dialogue=(
                segment_count if spread_generated_dialogue else 1
            ),
        )
    )
    long_form_hierarchical = (
        segment_count > _LONG_FORM_SEGMENTS_PER_CHAPTER
    )
    planned_by = "hierarchical_llm" if long_form_hierarchical else "llm"
    ledger: dict[str, Any] | None = None
    violations: list[str] = []
    raw = ""
    if long_form_hierarchical:
        ledger, long_form_warnings = _plan_long_form_ledger(
            prompt,
            canonical_ledger=canonical_ledger,
            segment_durations=durations,
            reference_context=reference_context,
            generate=generate,
            image_paths=image_paths,
            nsfw=nsfw,
            planning_style=planning_style,
            allow_generated_dialogue=allow_generated_dialogue,
            locked_dialogue=locked_dialogue,
        )
        planning_warnings.extend(long_form_warnings)
        if long_form_warnings:
            planned_by = "hierarchical_partial_fallback"
        violations = ledger_violations(
            prompt,
            ledger,
            segment_count=segment_count,
            locked_dialogue=locked_dialogue,
            expect_dialogue=expect_dialogue,
            allow_generated_dialogue=allow_generated_dialogue,
            require_dialogue_per_segment=False,
            segment_durations=durations,
        )
        if violations:
            print(
                "[MiniMax H3] Long-form ledger fallback: "
                + "; ".join(violations)
            )
            planning_diagnostics.extend(violations)
            planned_by = "deterministic_fallback"
            planning_warnings.append(
                "The hierarchical long-form schedule did not preserve every "
                "locked event, so Maestro used its deterministic duration-aware "
                "schedule instead."
            )
            ledger = deepcopy(canonical_ledger)
    else:
        try:
            raw = generate(
                prompt=ledger_prompt,
                system_prompt=ledger_guide,
                max_new_tokens=(
                    1000
                    if faithful_locked_schedule else
                    min(
                        4200,
                        max(1800, 950 + segment_count * 260 + len(source_events) * 150),
                    )
                ),
                temperature=0.38 if planning_style == "creative" else 0.22,
                top_p=0.84,
                image_paths=image_paths or None,
                enable_thinking=False,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                json_schema=ledger_schema,
            )
            from services.h3_window_planner import _parse_json_object

            candidate = _parse_json_object(raw)
            ledger = (
                _apply_faithful_treatment(canonical_ledger, candidate)
                if faithful_locked_schedule else
                _canonicalize_story_ledger(
                    prompt,
                    canonical_ledger,
                    candidate,
                    locked_dialogue=locked_dialogue,
                    segment_count=segment_count,
                    allow_generated_dialogue=allow_generated_dialogue,
                )
            )
            if spread_generated_dialogue:
                _spread_generated_dialogue_across_segments(
                    ledger,
                    segment_count=segment_count,
                )
            violations = ledger_violations(
                prompt,
                ledger,
                segment_count=segment_count,
                locked_dialogue=locked_dialogue,
                expect_dialogue=expect_dialogue,
                allow_generated_dialogue=allow_generated_dialogue,
                require_dialogue_per_segment=spread_generated_dialogue,
                segment_durations=durations,
            )
            if violations and not faithful_locked_schedule:
                print("[MiniMax H3] Story-schedule repair: " + "; ".join(violations))
                raw = generate(
                    prompt=(
                        ledger_prompt
                        + "\n\nPREVIOUS REJECTED STORY-SCHEDULE JSON:\n"
                        + json.dumps(candidate, ensure_ascii=False, indent=2)
                        + "\n\nREPAIR THE COMPLETE STORY SCHEDULE. Correct only these violations:\n- "
                        + "\n- ".join(violations)
                        + "\nReturn a complete replacement JSON object. Keep the immutable E-id and D-id catalogs exact; "
                        "you may regroup beats or move whole events between segments to satisfy timing."
                    ),
                    system_prompt=ledger_guide,
                    max_new_tokens=min(
                        4200,
                        max(1800, 950 + segment_count * 260 + len(source_events) * 150),
                    ),
                    temperature=0.08,
                    top_p=0.78,
                    image_paths=image_paths or None,
                    enable_thinking=False,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    json_schema=ledger_schema,
                )
                candidate = _parse_json_object(raw)
                ledger = _canonicalize_story_ledger(
                    prompt,
                    canonical_ledger,
                    candidate,
                    locked_dialogue=locked_dialogue,
                    segment_count=segment_count,
                    allow_generated_dialogue=allow_generated_dialogue,
                )
                if spread_generated_dialogue:
                    _spread_generated_dialogue_across_segments(
                        ledger,
                        segment_count=segment_count,
                    )
                violations = ledger_violations(
                    prompt,
                    ledger,
                    segment_count=segment_count,
                    locked_dialogue=locked_dialogue,
                    expect_dialogue=expect_dialogue,
                    allow_generated_dialogue=allow_generated_dialogue,
                    require_dialogue_per_segment=spread_generated_dialogue,
                    segment_durations=durations,
                )
            reflowable_timing = bool(
                violations
                and ledger
                and all(
                    re.fullmatch(
                        r"segment \d+ dialogue uses \d+ words; budget is \d+",
                        str(item or ""),
                    )
                    for item in violations
                )
            )
            if reflowable_timing:
                print(
                    "[MiniMax H3] Story-schedule timing reflow: "
                    + "; ".join(violations)
                )
                planning_notes.append(
                    "Maestro redistributed intact dialogue turns across adjacent "
                    "H3 windows after semantic planning so natural speech timing "
                    "stays within each native clip."
                )
                violations = []
            if violations or not ledger:
                raise ValueError("; ".join(violations or ["invalid story context JSON"]))
        except Exception as error:
            print(
                "[MiniMax H3] Shared-treatment fallback: "
                if faithful_locked_schedule else
                "[MiniMax H3] Story-schedule fallback: ",
                error,
                sep="",
            )
            diagnostic_items = violations or [sanitize_h3_prompt_text(error)]
            if not faithful_locked_schedule:
                planning_diagnostics.extend(
                    item for item in diagnostic_items if str(item or "").strip()
                )
            salvaged_ledger, salvaged_dialogue = (
                _salvage_creative_fallback(
                    prompt,
                    canonical_ledger,
                    ledger,
                    locked_dialogue=locked_dialogue,
                    segment_count=segment_count,
                    segment_durations=durations,
                    spread_generated_dialogue=spread_generated_dialogue,
                )
                if allow_generated_dialogue else
                (deepcopy(canonical_ledger), 0)
            )
            ledger = salvaged_ledger
            if salvaged_dialogue:
                planned_by = "hybrid_repair"
                planning_warnings.append(
                    "The AI story structure missed Maestro's fidelity checks, so Maestro repaired event timing while preserving "
                    f"{salvaged_dialogue} valid AI-authored dialogue line"
                    f"{'s' if salvaged_dialogue != 1 else ''} and safe creative direction."
                )
            else:
                if faithful_locked_schedule:
                    # The LLM supplies optional cinematic flavor only in
                    # faithful mode. Its failure cannot invalidate Maestro's
                    # locally owned event/dialogue schedule and should not be
                    # presented as a repaired story.
                    planned_by = "llm"
                    planning_notes.append(
                        "Maestro kept the exact locally scheduled story and "
                        "continued with per-window camera direction after the "
                        "optional shared cinematic treatment was unavailable."
                    )
                elif allow_generated_dialogue:
                    planned_by = "deterministic_fallback"
                    planning_warnings.append(
                        "The AI story schedule did not satisfy Maestro's fidelity checks after one focused repair. "
                        "The ordered source story and exact user-written dialogue remain intact, but no safe AI-authored dialogue "
                        "could be recovered, so Maestro used its duration-aware emergency schedule."
                    )
                else:
                    planned_by = "deterministic_fallback"
                    planning_warnings.append(
                        "The AI story schedule did not preserve Maestro's locked event and dialogue map after one focused repair, "
                        "so Maestro used its deterministic duration-aware timing schedule. Every supplied event and exact "
                        "user-written line remains intact."
                    )

    # Camera perspective, requested speed, style, nonverbal reactions, and
    # sequence shape are immutable even when the creative ledger succeeds.
    # Merge them after schema validation so the small LLM never owns them.
    # Principal identity, exact cast count, reference ownership, and blocking
    # are application-owned. A camera/story LLM may add useful choreography,
    # but it may not turn a location into a subject, drop an unreferenced
    # principal, or quietly duplicate a named character.
    ledger["subject_continuity"] = sanitize_h3_prompt_text(
        canonical_ledger.get("subject_continuity")
    )
    if source_intent.get("opening_state_contract"):
        ledger["initial_state"] = sanitize_h3_prompt_text(
            canonical_ledger.get("initial_state")
        )
        # Occupancy at an entrance boundary is source-owned.  A planning LLM
        # sometimes rewrites shared continuity as "George and Joey are already
        # seated" or carries "George bursts through the door" into every
        # segment.  Either instruction duplicates the entrant before shot one.
        ledger["setting_continuity"] = sanitize_h3_prompt_text(
            canonical_ledger.get("setting_continuity")
        )
        ledger["visual_continuity"] = sanitize_h3_prompt_text(
            canonical_ledger.get("visual_continuity")
        )
    ledger["source_intent"] = source_intent
    ledger["requested_nonverbal_vocals"] = source_intent[
        "requested_nonverbal_vocals"
    ]
    ledger["sequence_shape"] = (
        "ongoing" if source_intent["ongoing_motion"] else "resolved"
    )
    visual_parts: list[str] = []
    visual_keys: list[str] = []
    for part in (
        source_intent["perspective_contract"],
        source_intent["style_contract"],
        sanitize_h3_prompt_text(ledger.get("visual_continuity")),
    ):
        key = _normalize_key(part)
        if not key or any(key in existing or existing in key for existing in visual_keys):
            continue
        visual_parts.append(part)
        visual_keys.append(key)
    visual_contract = ". ".join(visual_parts)
    ledger["visual_continuity"] = visual_contract
    if source_intent["first_person_pov"]:
        ledger["editing_style"] = (
            "Locked continuous first-person POV; never cut to an external view"
            if camera_coverage != "multi_shot" else
            "First-person POV remains locked across motivated internal reframes"
        )
    if source_intent["ambient_contract"]:
        ambient = sanitize_h3_prompt_text(ledger.get("ambient_audio"))
        if source_intent["ambient_contract"].casefold() not in ambient.casefold():
            ledger["ambient_audio"] = "; ".join(
                part for part in (ambient, source_intent["ambient_contract"])
                if part
            )

    _lock_ledger_source_events(prompt, ledger)
    catalog = _dialogue_catalog(ledger, locked_dialogue)
    (
        render_beats,
        catalog,
        render_dialogue_events,
        dialogue_fragments,
    ) = _prepare_render_dialogue_schedule(
        list(ledger.get("beats") or []),
        catalog,
        segment_durations=durations,
        source_events=source_events,
        expected_dialogue_events=expected_dialogue_events,
    )
    if dialogue_fragments:
        planning_notes.append(
            "Maestro continued long exact dialogue across adjacent H3 windows "
            "to preserve every user-written word without rushing, repetition, "
            "or paraphrasing."
        )
    segment_guide = load_guide("enhance", "minimax_h3_story_segment")
    if nsfw:
        segment_guide += (
            "\n\nMATURE-MODE FIDELITY\nPreserve explicitly requested mature material. "
            "Do not censor it, add to it, or intensify it."
        )
    cast_names = list(source_intent.get("cast_names") or [])
    dialogue_by_id = {
        str(item.get("dialogue_id") or "").upper(): item
        for item in catalog
    }
    known_speakers = list(dict.fromkeys(
        sanitize_h3_prompt_text(item.get("speaker"))
        for item in catalog
        if sanitize_h3_prompt_text(item.get("speaker"))
    ))
    cast_first_segments: dict[str, int] = {}
    initial_cast = _active_h3_cast_names(
        cast_names,
        ledger.get("initial_state"),
    )
    for name in initial_cast:
        cast_first_segments[name] = 1
    for beat in render_beats:
        try:
            beat_segment = int(beat.get("segment") or 1)
        except (TypeError, ValueError):
            beat_segment = 1
        beat_speakers = [
            sanitize_h3_prompt_text(
                dialogue_by_id.get(str(dialogue_id or "").upper(), {}).get("speaker")
            )
            for dialogue_id in (beat.get("dialogue_ids") or [])
        ]
        beat_cast = _active_h3_cast_names(
            cast_names,
            " ".join([
                sanitize_h3_prompt_text(beat.get("description")),
                *beat_speakers,
            ]),
        )
        for name in beat_cast:
            cast_first_segments[name] = min(
                cast_first_segments.get(name, beat_segment),
                beat_segment,
            )
    segments: list[dict[str, Any]] = []
    previous_closing = sanitize_h3_prompt_text(ledger.get("initial_state"))
    for index, duration in enumerate(durations):
        segment_number = index + 1
        semantic_beats = [
            item for item in render_beats
            if isinstance(item, dict) and int(item.get("segment") or 0) == segment_number
        ]
        beats = _camera_phase_beats(
            semantic_beats,
            source_events=source_events,
            expected_dialogue_events=render_dialogue_events,
        )
        beats = _coalesce_camera_phases(beats, target_count=4)
        assigned_dialogue_ids = [
            str(dialogue_id or "").upper()
            for beat in beats
            for dialogue_id in (beat.get("dialogue_ids") or [])
        ]
        assigned_dialogue = [
            {
                "dialogue_id": item.get("dialogue_id"),
                "speaker": item.get("speaker"),
                "language": item.get("language"),
                "text": item.get("text"),
                "delivery": item.get("delivery"),
                "off_camera": bool(item.get("off_camera")),
            }
            for item in catalog
            if str(item.get("dialogue_id") or "").upper() in assigned_dialogue_ids
        ]
        future_cast = [
            name for name, first_segment in cast_first_segments.items()
            if first_segment > segment_number
        ]
        active_cast_text = " ".join([
            previous_closing,
            *(
                f"{beat.get('description', '')} {beat.get('state_after', '')}"
                for beat in beats
            ),
            *(str(item.get("speaker") or "") for item in assigned_dialogue),
        ])
        active_cast = _active_h3_cast_names(cast_names, active_cast_text)
        if not active_cast and cast_names:
            active_cast = _active_h3_cast_names(
                cast_names,
                ledger.get("initial_state") if segment_number == 1 else previous_closing,
            )
        active_cast_contract = _h3_cast_cardinality_contract(prompt, active_cast)
        blocking_contract = sanitize_h3_prompt_text(
            source_intent.get("blocking_contract")
        )
        blocking_cast = _active_h3_cast_names(cast_names, blocking_contract)
        if blocking_cast and not all(
            any(_same_h3_cast_identity(name, active) for active in active_cast)
            for name in blocking_cast
        ):
            blocking_contract = ""
        if mode == "sliding_window":
            mode_instruction = (
                "This is a frame-linked continuation. Its opening must exactly match the supplied previous frame/state. "
                "Do not restart or recap. Internal motivated cuts are allowed, but the segment boundary itself is not a story cut."
            )
        elif mode == "reference_sequence_continuation":
            mode_instruction = (
                "This is a native Ref2VA motion-and-audio overlap continuation. The canonical references remain identity guidance, "
                "not opening keyframes. Continue from the supplied previous state without restarting, restaging, or replaying an action. "
                "Internal motivated cuts are allowed, but the segment boundary itself is not a story cut."
            )
        else:
            mode_instruction = (
                "This is an independently generated editorial clip. Restate a complete readable opening composition, "
                "use the canonical references for identity, and advance only this clip's assigned beats."
            )
        dialogue_by_id = {
            str(item.get("dialogue_id") or "").upper(): item
            for item in assigned_dialogue
        }
        creative_beats = []
        for beat in beats:
            beat_dialogue = [
                dialogue_by_id.get(str(dialogue_id or "").upper())
                for dialogue_id in (beat.get("dialogue_ids") or [])
            ]
            creative_beats.append({
                "event": sanitize_h3_prompt_text(beat.get("description")),
                "dialogue_performances": [
                    {
                        "speaker": item.get("speaker"),
                        "off_camera": bool(item.get("off_camera")),
                        "spoken_words": _dialogue_word_count(item.get("text")),
                    }
                    for item in beat_dialogue
                    if isinstance(item, dict)
                ],
                "resulting_state": sanitize_h3_prompt_text(beat.get("state_after")),
                "sound_effects": sanitize_h3_prompt_text(beat.get("sound_effects")),
            })
        creative_dialogue = [
            {
                "speaker": item.get("speaker"),
                "language": item.get("language"),
                "exact_text": item.get("text"),
                "off_camera": bool(item.get("off_camera")),
            }
            for item in assigned_dialogue
        ]
        segment_prompt = (
            f"Segment {segment_number} of {segment_count}; local duration 0.000 to {duration:.3f} seconds.\n"
            f"{mode_instruction}\n\n"
            f"Shared subjects: {ledger.get('subject_continuity')}\n"
            f"Active principal cast for this segment: {active_cast_contract or 'Use only the principals required by the assigned events.'}\n"
            f"Blocking contract: {blocking_contract or 'Preserve the concrete opening geography and every completed state change across cuts.'}\n"
            f"Shared setting: {ledger.get('setting_continuity')}\n"
            f"Shared visual language: {ledger.get('visual_continuity')}\n"
            f"Editing style: {ledger.get('editing_style')}\n"
            f"Required pacing and performance energy: {source_intent.get('pacing_contract')}\n"
            f"Required opening state: {previous_closing}\n"
            f"Immutable chronological events (depict each once, in order):\n"
            f"{json.dumps(creative_beats, ensure_ascii=False, indent=2)}\n\n"
            f"Immutable dialogue performances (plan visible/off-camera performance but do not reproduce text in JSON):\n"
            f"{json.dumps(creative_dialogue, ensure_ascii=False, indent=2)}\n\n"
            "SPEAKER-CAMERA LOCK: for every visible dialogue performance, both "
            "framing and camera must settle on that exact speaker before the vocal "
            "line begins. A listener may react only after the line ends. Never frame "
            "one character while assigning another character's visible line. Keep a "
            "dependent gesture or introduction in the same shot as its owning line.\n\n"
            f"Original user concept for fidelity only:\n{prompt}"
        )
        if long_form_hierarchical:
            # Chapter expansion already supplied the local visible progression.
            # Compile the camera clock deterministically instead of making one
            # more LLM request for every window in a potentially hour-long run.
            segment = _fallback_segment(
                segment_number,
                duration=duration,
                beats=beats,
                opening_state=previous_closing,
                camera_coverage=camera_coverage,
                dialogue_catalog=catalog,
                source_intent=source_intent,
            )
            materialized = _materialize_segment(
                segment,
                beats=beats,
                dialogue_catalog=catalog,
                source_events=source_events,
                future_cast=future_cast,
            )
            final_errors = _materialized_segment_violations(
                materialized,
                known_speakers=known_speakers,
                future_cast=future_cast,
            )
            if final_errors:
                print(
                    f"[MiniMax H3] Segment {segment_number} final staging repair: "
                    + "; ".join(final_errors)
                )
                materialized = _repair_materialized_segment_staging(
                    materialized,
                    known_speakers=known_speakers,
                    future_cast=future_cast,
                )
            materialized["active_cast"] = list(active_cast)
            materialized["opening_state"] = previous_closing
            segments.append(materialized)
            previous_closing = materialized["closing_state"]
            continue
        schema = _segment_schema(
            segment_number,
            maximum_shots=_segment_shot_limit(beats),
        )
        segment: dict[str, Any] | None = None
        segment_errors: list[str] = []
        segment_token_budget = min(
            2600,
            max(
                1750,
                1050 + len(beats) * 260 + len(assigned_dialogue) * 120,
            ),
        )
        try:
            raw = generate(
                prompt=segment_prompt,
                system_prompt=segment_guide,
                max_new_tokens=segment_token_budget,
                temperature=0.24,
                top_p=0.86,
                image_paths=None,
                enable_thinking=False,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                json_schema=schema,
            )
            from services.h3_window_planner import _parse_json_object

            segment = _parse_json_object(raw)
            segment = _canonicalize_segment_contract(
                segment,
                segment_number=segment_number,
                duration=duration,
                assigned_beats=beats,
                dialogue_catalog=catalog,
                opening_state=previous_closing,
                source_intent=source_intent,
            )
            segment_errors = segment_violations(
                prompt,
                segment,
                segment_number=segment_number,
                duration=duration,
                assigned_beats=beats,
                dialogue_catalog=catalog,
            )
            if segment_errors:
                print(
                    f"[MiniMax H3] Segment {segment_number} repair: "
                    + "; ".join(segment_errors)
                )
                raw = generate(
                    prompt=(
                        segment_prompt
                        + "\n\nPREVIOUS REJECTED SEGMENT JSON:\n"
                        + json.dumps(segment, ensure_ascii=False, indent=2)
                        + "\n\nREPAIR ONLY THIS SEGMENT. Correct these violations:\n- "
                        + "\n- ".join(segment_errors)
                        + "\nReturn one complete local shot clock. Maestro will attach immutable event and dialogue ownership. "
                        "Do not add, repeat, recap, or preview any event."
                    ),
                    system_prompt=segment_guide,
                    max_new_tokens=segment_token_budget,
                    temperature=0.08,
                    top_p=0.78,
                    image_paths=None,
                    enable_thinking=False,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    json_schema=schema,
                )
                segment = _parse_json_object(raw)
                segment = _canonicalize_segment_contract(
                    segment,
                    segment_number=segment_number,
                    duration=duration,
                    assigned_beats=beats,
                    dialogue_catalog=catalog,
                    opening_state=previous_closing,
                    source_intent=source_intent,
                )
                segment_errors = segment_violations(
                    prompt,
                    segment,
                    segment_number=segment_number,
                    duration=duration,
                    assigned_beats=beats,
                    dialogue_catalog=catalog,
                )
            if segment_errors or not segment:
                raise ValueError("; ".join(segment_errors or ["invalid segment JSON"]))
        except Exception as error:
            print(f"[MiniMax H3] Segment {segment_number} fallback: {error}")
            planned_by = "deterministic_fallback"
            planning_warnings.append(
                f"Window {segment_number}'s camera plan did not satisfy Maestro's "
                "fidelity checks, so Maestro compiled that window directly from "
                "the locked source events."
            )
            segment = _fallback_segment(
                segment_number,
                duration=duration,
                beats=beats,
                opening_state=previous_closing,
                camera_coverage=camera_coverage,
                dialogue_catalog=catalog,
                source_intent=source_intent,
            )
        materialized = _materialize_segment(
            segment,
            beats=beats,
            dialogue_catalog=catalog,
            source_events=source_events,
            future_cast=future_cast,
        )
        final_errors = _materialized_segment_violations(
            materialized,
            known_speakers=known_speakers,
            future_cast=future_cast,
        )
        if final_errors:
            print(
                f"[MiniMax H3] Segment {segment_number} final staging repair: "
                + "; ".join(final_errors)
            )
            planning_warnings.append(
                f"Window {segment_number}'s final camera staging conflicted with "
                "its locked speaker map, so Maestro corrected only that staging "
                "while preserving its story action and exact dialogue."
            )
            materialized = _repair_materialized_segment_staging(
                materialized,
                known_speakers=known_speakers,
                future_cast=future_cast,
            )
        materialized["active_cast"] = list(active_cast)
        # Story state belongs to the ledger, not the camera expander. This also
        # makes FL2VA's next-frame continuation and Omni's editorial handoff
        # deterministic even if the segment LLM paraphrases opening_state.
        materialized["opening_state"] = previous_closing
        segments.append(materialized)
        previous_closing = materialized["closing_state"]

    # Remember every principal introduced so far, but do not require them to be
    # visible at the outgoing boundary.  Forcing a group composition into the
    # final second changes the whole diffusion sample: it can restage furniture,
    # replay an entrance, or make the actor walk to a duplicate location.  The
    # next segment instead receives this list as identity/wardrobe context for
    # anyone who remains visible or returns from briefly off camera.  Future
    # entrants are still excluded until their first assigned segment.
    for index, segment in enumerate(segments):
        segment_number = index + 1
        segment["continuity_handoff_cast"] = (
            [
                sanitize_h3_prompt_text(name)
                for name in cast_names
                if sanitize_h3_prompt_text(name)
                and int(cast_first_segments.get(name, segment_number + 1))
                <= segment_number
            ]
            if index + 1 < len(segments)
            else []
        )

    return {
        "planned_by": planned_by,
        "planning_warnings": list(dict.fromkeys(planning_warnings)),
        "planning_diagnostics": list(dict.fromkeys(planning_diagnostics)),
        "planning_notes": list(dict.fromkeys(planning_notes)),
        "source_intent": source_intent,
        "ledger": ledger,
        "locked_dialogue": [
            {
                key: value for key, value in item.items()
                if key not in {"source_offset", "source_end"}
            }
            for item in locked_dialogue
        ],
        "dialogue_fragments": dialogue_fragments,
        "segments": segments,
    }
