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

from models.minimax_h3.speakers import is_h3_production_label

from services.h3_authored_brief import (
    authored_optical_settings,
    authored_sound_cues,
    is_standalone_sound_cue,
    authored_timed_brief,
    character_profile_spans,
    explicit_character_profiles,
    explicit_negative_constraints,
    positive_instruction_text,
    production_note_spans,
    video_direction_source,
)

from services.dialogue_timing import (
    DIALOGUE_DEFAULT_WORDS_PER_SECOND as _H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND,
    DIALOGUE_MAX_WORDS_PER_SECOND as _H3_DIALOGUE_MAX_WORDS_PER_SECOND,
)
from services.dialogue_writing import (
    conversation_brief,
    creative_dialogue_budget,
    creative_dialogue_expected,
    dialogue_forbidden,
    only_supplied_dialogue_requested,
    requested_dialogue_topics,
)
from services.director.long_form_story import (
    LONG_FORM_STORY_BIBLE_SCHEMA,
    build_long_form_story_bible_fallback,
    ensure_long_form_location_coverage,
    format_long_form_story_bible,
    normalize_long_form_outline,
    normalize_long_form_story_bible,
)


H3_STORY_LEDGER_VERSION = 97


_H3_WINDOW_BOOKKEEPING_RE = re.compile(
    r"\b(?:generation|denoising|inference|latent|context)[\s-]+windows?\b"
    r"|\bsliding[\s-]+window[\s-]+(?:pass|index|boundary|overlap|size)\b"
    r"|\bwindow\s+#?\s*\d+\b",
    flags=re.IGNORECASE,
)


def has_h3_window_bookkeeping(text: str, *, source_prompt: str = "") -> bool:
    """Detect explicit generation bookkeeping, not physical scene windows.

    Bare 'window', 'next window' and even 'sliding windows' can describe real
    architecture. Only reject unambiguous pipeline terminology or numbered
    window labels, allowing the same literal wording supplied by the user.
    """

    def terms(value: str) -> set[str]:
        return {
            re.sub(r"[\s-]+", " ", match.group(0).casefold())
            for match in _H3_WINDOW_BOOKKEEPING_RE.finditer(str(value or ""))
        }

    return bool(terms(text) - terms(source_prompt))


class H3DialogueTimingError(ValueError):
    """Exact screenplay dialogue cannot fit without unsafe rewriting."""


def normalize_h3_planning_style(value: Any) -> str:
    """Return the durable AI-writing contract stored with every H3 plan."""

    from services.adaptive_enhancement import normalize_writing_style

    return normalize_writing_style(value)


def _load_h3_planning_guide(name: str, *, nsfw: bool) -> str:
    """Share Studio's optional content guidance across H3 planning stages."""
    from services.guide_loader import load_guide

    guide = load_guide("enhance", name)
    if nsfw:
        content_guide = load_guide("enhance", "nsfw_shared")
        if content_guide:
            # Keep the stage's JSON, timing and reference contract after the
            # shared writing guidance, as in Studio's final output contract.
            return f"{content_guide}\n\n{guide}"
    return guide


def _only_supplied_dialogue_requested(prompt: str) -> bool:
    """Whether creative planning must not add dialogue around quoted lines."""
    return only_supplied_dialogue_requested(prompt)


def _creative_conversation_brief(prompt: str) -> bool:
    """Return whether Creative mode should sustain speech across windows.

    A character scene that explicitly revolves around telling, explaining,
    discussing, interviewing, or banter is different from an action scene
    that merely ends with one verbal reaction. The former needs an authored
    exchange from the opening window onward; otherwise H3 fills long silent
    stretches with improvised, unintelligible speech.
    """

    return conversation_brief(prompt)

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
    r"\b(?:says?|said|saying|speaks?|speaking|asks?|asked|answers?|answered|answering|"
    r"replies?|replied|responds?|responded|"
    r"whispers?|whispered|shouts?|shouted|yells?|yelled|declares?|declared|"
    r"mumbles?|mumbled|mumbling|murmurs?|murmured|murmuring|mutters?|muttered|muttering|"
    r"muffles?|muffled|muffling|"
    r"states?|stated|tells?|telling|told|explains?|explained|explaining|"
    r"informs?|informed|informing|announces?|announced|announcing|"
    r"exclaims?|exclaimed|exclaiming|"
    r"calls?\s+out|called\s+out)\b",
    flags=re.IGNORECASE,
)


def _spectacle_violations(source: str, draft: Any) -> list[str]:
    """Check positive effect claims against positive source permissions.

    Surface wording is not an ability: an authored cursed-energy effect can
    become an energy field without introducing a new class of power. Keep
    unrelated mechanisms separate, and never treat a prohibition as permission
    or as an effect the writer actually performed.
    """
    def text_values(value: Any) -> str:
        if isinstance(value, dict):
            return "\n".join(text_values(v) for v in value.values())
        if isinstance(value, list):
            return "\n".join(text_values(v) for v in value)
        return value if isinstance(value, str) else ""

    def positive(text: str) -> str:
        # Keep sentence boundaries so a trailing negative list stays a rule.
        return "\n".join(
            line for line in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
            if not re.match(r"\s*(?:no|never|without|prohibited|forbidden|disallowed)\b", line, re.I)
        )

    source_positive = positive(source)
    draft_positive = positive(text_values(draft))
    negatives = explicit_negative_constraints(source)
    energy_requested = bool(re.search(
        r"\b(?:cursed|arcane|magical|supernatural|spiritual|psychic|cosmic)[-\s]+(?:energy|fluid|power)\b|"
        r"\b(?:colou?red|glowing|luminous|blue|red|purple|golden)[-\s]+energy\b",
        source_positive, re.I,
    ))
    forbids_magic = bool(re.search(
        r"\b(?:no|without)\s+(?:any\s+)?(?:magic|magical\s+(?:powers|effects)|supernatural\s+(?:powers|effects))\b",
        negatives, re.I,
    ))
    forbids_energy = bool(re.search(
        r"\b(?:no|without)\s+(?:(?:any|colou?red|glowing)\s+)?energ(?:y|ies)\b", negatives, re.I,
    ))
    for pattern in UNREQUESTED_SPECTACLE_PATTERNS:
        match = re.search(pattern, draft_positive, re.I)
        if not match:
            continue
        effect = match.group(0).strip()
        energy_effect = bool(re.search(r"\benergy\b", effect, re.I))
        source_matches = bool(re.search(pattern, source_positive, re.I))
        physical_effect_requested = source_matches and bool(re.search(r"\b(?:laser|lightning)\b", effect, re.I))
        forbidden = ((forbids_magic and not physical_effect_requested) or (forbids_energy and energy_effect)
                     or bool(re.search(pattern, negatives, re.I)))
        permitted = (source_matches
                     or (energy_effect and energy_requested))
        if forbidden or not permitted:
            return [f"invented unrequested power/effect: {effect}"]
    return []
_PLANNER_CONVERSATION_RE = re.compile(
    r"\b(?:have|has|having|had)\s+(?:[\w-]+[,\s]+){0,4}(?:conversation|discussion)\b",
    flags=re.IGNORECASE,
)
_PLANNER_ADVICE_RE = re.compile(
    r"\b(?:gives?|gave|giving|offers?|offered|offering)\s+"
    r"(?:[\w'’-]+[,\s]+){0,6}(?:advice|guidance|reassurance|recommendations?|"
    r"suggestions?|(?:practical|helpful|useful)\s+(?:\w+\s+)?tips?)\b",
    flags=re.IGNORECASE,
)
_PLANNER_SPEECH_VERB = re.compile(
    _SPEECH_VERB.pattern + r"|\b(?:talk(?:s|ed|ing)?|answer(?:s|ed|ing)?|asking|replying|responding|"
    r"interject(?:s|ed|ing)?|exclaim(?:s|ed|ing)?|argu(?:e|es|ed|ing)|"
    r"advis(?:e|es|ed|ing)|recommend(?:s|ed|ing)?|suggest(?:s|ed|ing)?|reassur(?:e|es|ed|ing))\b|"
    + _PLANNER_CONVERSATION_RE.pattern + "|" + _PLANNER_ADVICE_RE.pattern,
    flags=re.IGNORECASE,
)
_PROPER_NAME = re.compile(
    r"\b[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3}\b"
)
_PLACEHOLDER_DIALOGUE = re.compile(r"^[\s.…_-]*$")


def _is_generated_stage_direction(value: str) -> bool:
    """Bracketed acting notes are not words to feed to the speech generator.

    Used only on AI-authored lines. Supplied quotations remain immutable.
    Required physical actions are independently retained by the source ledger.
    """
    text = str(value or "").strip()
    if not ((text.startswith("(") and text.endswith(")"))
            or (text.startswith("[") and text.endswith("]"))
            or (text.startswith("*") and text.endswith("*"))):
        return False
    return bool(re.match(
        r"(?:he\s+|she\s+|they\s+)?(?:smil\w*|grin\w*|nod\w*|gestur\w*|paus\w*|"
        r"laugh\w*|sigh\w*|chuckl\w*|look\w*|turn\w*|walk\w*|step\w*|taking\s+a\s+step|"
        r"silence|silently|no\s+(?:dialogue|speech))\b", text[1:-1].strip(), re.I,
    ))
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

_NON_CAST_PROPER_NAMES = {
    "anyone", "beat", "everybody", "everyone", "nobody", "no one", "no",
    "camera", "okay", "ok", "someone", "starts", "that", "there", "these", "this",
    "those", "you", "he", "she", "they", "we", "it", "i", "his", "her",
    "their", "our", "them", "its", "us", "your", "my",
    # Clause operators are not actors merely because a later name performs
    # an action in the same sentence. Explicit cast definitions still win.
    "at", "after", "before", "during", "until", "when", "while", "by",
    "near", "toward", "towards", "finally", "eventually", "meanwhile",
    "as", "outside", "inside",
    # Role/age adjectives can begin a sentence before the actual proper name.
    "adult", "young", "elderly",
}
_SCREENPLAY_DIALOGUE_RE = re.compile(
    r"(?m)^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?"
    r"(?P<speaker>[A-Za-z][A-Za-z0-9_'\u2019.\-]*"
    r"(?:[ \t]+[A-Za-z][A-Za-z0-9_'\u2019.\-]*){0,3})"
    r"(?:[ \t]*\((?P<delivery>[^)\r\n]{1,80})\))?"
    r"(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?"
    r"(?P<text>[^\r\n]+?)[ \t]*$"
)
_DIALOGUE_QUOTE_RE = re.compile(
    r'["“]([^"”\r\n]{1,600})["”]|'
    r"(?<!\w)['‘]((?:[^'’\r\n]|(?<=\w)['’](?=\w)){1,600})['’](?!\w)"
)
# Descriptive principals are as valid as proper names. Keep articles and a
# bounded adjective phrase so possessives and nearby listeners cannot steal
# the subject of an attribution ("Sydney smiles. The older man then says...").
_DESCRIPTIVE_PERSON = (
    r"(?:the|an?|another)\s+(?:[\w-]+\s+){0,3}"
    r"(?:man|woman|boy|girl|person|child|officer|doctor|nurse|teacher|"
    r"singer|drummer|guitarist|monk|fighter|waiter|waitress|driver|captain)"
)


def _descriptive_speaker_name(source: str, name: str) -> str:
    """Resolve an unambiguous bare role to its established description."""
    clean = re.sub(r"^(?:the|an?|another)\s+", "", name, flags=re.I).strip()
    if len(clean.split()) != 1:
        return clean
    descriptions = list(dict.fromkeys(
        re.sub(r"^(?:the|an?|another)\s+", "", match.group(), flags=re.I).strip()
        for match in re.finditer(rf"\b{_DESCRIPTIVE_PERSON}\b", source, re.I)
        if match.group().split()[-1].casefold() == clean.casefold()
    ))
    specific = [value for value in descriptions if len(value.split()) > 1]
    return specific[0] if len(specific) == 1 else clean
_SCREENPLAY_PROFILE_TEXT_RE = re.compile(
    r"^(?:biased\s+towards?\b|speciali[sz](?:es?|ed|ing)\s+in\b|"
    r"(?:appearance|wardrobe|outfit|clothing|hairstyle|facial\s+features|"
    r"body\s+proportions|personality|fighting\s+style)\s*:|"
    r"(?:[A-Za-z][A-Za-z-]*\s+){0,4}[A-Za-z][A-Za-z-]*-"
    r"(?:clothed|robed|haired|skinned)\b)",
    flags=re.IGNORECASE,
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
    return bool(re.fullmatch(_DESCRIPTIVE_PERSON, text, flags=re.I) or re.fullmatch(
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
    ) or re.fullmatch(
        r"(?:please\s+)?(?:write|plan|describe)\s+(?:the\s+)?"
        r"(?:(?:shot[- ](?:for|by)[- ]shot|continuously\s+chained|"
        r"continuous|connected|detailed|fighting|fight|action)\s+)*choreography",
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
    # Global coverage preferences constrain action instead of consuming a
    # separate timed camera tour. Specific reveals or moves tied to an event
    # deliberately do not match this adjective-only form.
    camera_unit = (
        r"(?:(?:a|an|the)\s+)?"
        r"(?:(?:dynamic|cinematic|motivated|fluid|smooth|handheld|kinetic|fast|"
        r"energetic|restrained|static|continuous|single[- ]take|wide|close|"
        r"low[- ]angle|high[- ]angle|realistic|natural|conversational|alternating|medium)\s+)*"
        r"(?:camera(?:\s+(?:movements?|work|coverage))?|camerawork|cinematography|"
        r"framing|shots?|cuts?|close[- ]ups?|pacing|shallow\s+depth\s+of\s+field)"
        r"(?:\s+on\s+(?:the\s+)?(?:current|active)\s+speaker)?"
    )
    if re.fullmatch(
        r"(?:please\s+)?(?:use|keep|maintain|favor|prefer)\s+"
        + camera_unit + r"(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)" + camera_unit + r")*"
        r"(?:\s+(?:throughout|throughout\s+(?:the\s+)?(?:scene|film|video)))?",
        text, flags=re.IGNORECASE,
    ) or re.fullmatch(
        r"(?:please\s+)?keep\s+(?:the\s+)?camera\s+moving\s+with\s+(?:the\s+)?action",
        text, flags=re.IGNORECASE,
    ):
        return True
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
    r"\b(?:talk(?:s|ed|ing)?|discuss(?:es|ed|ing)?|chat(?:s|ted|ting)?|"
    r"explain(?:s|ed|ing)?|ask(?:s|ed|ing)?|answer(?:s|ed|ing)?|"
    r"argu(?:e|es|ed|ing)|debat(?:e|es|ed|ing)|"
    r"approach(?:es|ed|ing)?|arriv(?:e|es|ed|ing)|attack(?:s|ed|ing)?|"
    r"breathe(?:s|d|ing)?|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|fight(?:s|ing)?|"
    r"fly|flies|flew|flying|grab(?:s|bed|bing)?|hold(?:s|ing)?|jump(?:s|ed|ing)?|"
    r"laugh(?:s|ed|ing)?|look(?:s|ed|ing)?|move(?:s|d|ing)?|nod(?:s|ded|ding)?|"
    r"punch(?:es|ed|ing)?|raise(?:s|d|ing)?|react(?:s|ed|ing)?|run(?:s|ning)?|"
    r"see(?:s|ing)?|saw|notice(?:s|d|ing)?|rush(?:es|ed|ing)|"
    r"leap(?:s|ed|ing)?|exclaim(?:s|ed|ing)?|"
    r"assist(?:s|ed|ing)?|help(?:s|ed|ing)?|invit(?:e|es|ed|ing)|meet(?:s|ing)?|met|"
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

_H3_IMPERATIVE_DIRECTION_RE = re.compile(
    r"^(?:please\s+)?(?P<verb>keep|maintain|preserve|ensure)\b(?P<body>.+)$",
    flags=re.IGNORECASE | re.DOTALL,
)
_H3_CONTINUITY_DIRECTION_RE = re.compile(
    r"\b(?:continuity|entrances?|exits?|listeners?|room\s+geography|screen\s+geography|"
    r"doors?['\u2019]s?\s+(?:open|closed)|doors?\s+(?:open|closed)|open\s+or\s+closed\s+state|"
    r"state\s+changes?)\b.*\b(?:coherent|consistent|continuity|across\s+(?:cuts?|the\s+transition)|"
    r"throughout|between\s+(?:shots?|rooms?|windows?))\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def _is_persistent_action_direction(value: str) -> bool:
    """Keep general choreography preferences out of the timed event catalog."""
    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    quality = r"(?:connected|readable|clear|continuous|coherent|fluid)"
    action = r"(?:actions?|exchanges?|choreography|movement|geography)"
    unit = rf"(?:(?:a|the)\s+)?{quality}\s+{action}"
    scope = (r"(?:\s+(?:throughout|across)(?:\s+the)?"
             r"(?:\s+\d+\s+seconds?|\s+(?:scene|clip|film|video)))?")
    return bool(re.fullmatch(
        rf"(?:please\s+)?(?:show|keep|maintain|use)\s+{unit}"
        rf"(?:\s+and\s+{unit})*{scope}", text, re.I,
    ) or re.fullmatch(
        rf"(?:please\s+)?(?:keep|maintain)\s+(?:the\s+)?{action}"
        rf"(?:\s+and\s+{action})*\s+{quality}{scope}", text, re.I,
    ))


def _is_h3_preservation_contract(value: str) -> bool:
    """Recognize declarative fidelity checklists, not visible imperatives."""

    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    direction = _H3_IMPERATIVE_DIRECTION_RE.match(text)
    if not direction:
        return False
    body = sanitize_h3_prompt_text(direction.group("body"))
    if re.fullmatch(
        r"(?:the|an?)\s+(?:[a-z-]+\s+){1,4}"
        r"(?:closed|open|stationary|unchanged|intact)\s+throughout"
        r"(?:\s+(?:the\s+)?(?:scene|film|video|conversation))?",
        body, flags=re.I,
    ):
        return True  # A persistent prop state, not an extra action to perform.
    # Concrete named blocking such as ``Keep Mara beside the door`` remains a
    # filmable event. A checklist of states/relations constrains every event.
    return bool(
        re.search(
            r"\b(?:action\s+order|chronology|continuity|direction|distance|"
            r"geography|identity|landmarks?|locks?|ownership|positions?|"
            r"speakers?|states?|dialogue|speech|wording|quotes?|quotations?|"
            r"(?:exact|supplied|written)\s+(?:lines?|words?))\b",
            body,
            flags=re.IGNORECASE,
        )
        and not re.search(
            rf"\b(?:{_ACTION_VERBS})(?:s|es|ed|ing)?\b",
            body,
            flags=re.IGNORECASE,
        )
    )


def _is_h3_negative_constraint_fragment(value: str) -> bool:
    """Keep a production restriction out of the chronological event list."""

    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    if not re.match(r"^(?:no|without)\b", text, flags=re.IGNORECASE):
        return False
    # A grammatical human subject makes this a negative story event regardless
    # of the particular action verb: ``No one catches the glass`` and
    # ``Without anyone intervening`` must remain on the timeline. Bare noun
    # lists such as ``No dialogue, flashback, or visitor`` remain constraints.
    human_subject = re.match(
        r"^(?:no\s+(?:one|body|person|character|adult|child|man|woman|worker|"
        r"visitor|driver|operator|assistant)|without\s+(?:anyone|anybody|someone|"
        r"somebody))\b(?P<remainder>.*)",
        text,
        flags=re.IGNORECASE,
    )
    if human_subject:
        remainder = human_subject.group("remainder")
        if re.match(r"\s+\S+", remainder):
            return False
        if re.match(
            r"\s*,\s*(?:including\s+[^,.;!?]{1,80},\s*)?"
            r"(?:(?:he|she|they|[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,3})\s+)?"
            r"(?!no\b|without\b)[A-Za-z][\w'’-]*\s+\S+",
            remainder,
        ):
            return False
    return True


def _is_h3_performance_direction(value: str) -> bool:
    """Recognize a scene-wide acting note without consuming screen time."""

    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    return bool(re.fullmatch(
        r"(?:convey|express|communicate)\s+(?:the\s+)?"
        r"[a-z][a-z -]{0,45}\s+(?:through|with|by)\s+"
        r"[a-z][a-z ,'-]{1,100}",
        text,
        flags=re.IGNORECASE,
    ) or re.fullmatch(
        r"(?:the\s+)?(?:listener|non[- ]speaking\s+character)\s+"
        r"(?:reacts?|listens?)\s+(?:silently|without\s+speaking)", text, re.I,
    ))


def _is_h3_no_addition_contract(value: str) -> bool:
    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    return bool(re.fullmatch(
        r"(?:(?:add|include|write)\s+no|"
        r"(?:do\s+not|don['’]t|never)\s+(?:add|include|invent|write))\s+"
        r"(?:(?:any|other|extra|new|additional|unscripted)\s+){0,3}"
        r"(?:dialogue|speech|voices?|music|subtitles?|captions?|narration)",
        text,
        flags=re.IGNORECASE,
    ))


_H3_FACT_NAME = r"[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3}"
_H3_FACT_SUBJECT = (
    rf"(?:{_H3_FACT_NAME}(?:\s+and\s+{_H3_FACT_NAME}){{0,2}}|"
    r"(?i:he|she|they|it|(?:the|both)\s+(?:heroes|heroines|fighters|characters|"
    r"women|men|hero|heroine|fighter|woman|man)))"
)


def _is_persistent_ability_fact(value: str) -> bool:
    """Recognize bare capability definitions, not their use or acquisition.

    Full-sentence matching deliberately leaves qualified and mixed actions
    alone: 'can fly across the gap' or 'can fly and punches Dev' are events.
    """
    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    if re.match(r"^(?:now|then|finally|after|before|once|suddenly|eventually)\b", text, re.I):
        return False
    movement = r"(?:fly|levitate|teleport|turn invisible|breathe fire)"
    ability = (r"(?:heat[- ]vision|laser vision|superhuman strength|super strength|"
               r"super speed|telekinesis|teleportation|flight|invisibility)")
    trait = r"(?:super[- ]?human|fast|strong|agile|powerful|invulnerable)"
    separator = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)"
    traits = rf"(?:is|are)\s+(?:both\s+)?{trait}(?:{separator}{trait})*"
    predicate = rf"(?:can\s+{movement}|(?:has|have)\s+{ability}|{traits})"
    # Match the complete coordinated statement before the action splitter.
    # Otherwise "They are strong, and they can fly" becomes two timed events.
    capability = (
        rf"{_H3_FACT_SUBJECT}(?i:\s+{predicate})"
        rf"(?:(?i:{separator})(?:{_H3_FACT_SUBJECT}\s+)?"
        rf"(?i:{predicate}|{movement}))*"
    )
    owner = rf"(?:(?i:her|his|their|its)|{_H3_FACT_NAME}['’]s)"
    mechanism = (
        rf"{owner}(?i:\s+(?:flight\s+is\s+(?:innate|natural)\s+levitation|"
        r"heat[- ]vision\s+(?:comes|originates)\s+from\s+(?:her|his|their|its)\s+eyes))"
    )
    return bool(re.fullmatch(rf"(?:{capability}|{mechanism}(?:\s+and\s+{mechanism})*)", text))


def _is_persistent_setting_fact(value: str) -> bool:
    """Recognize a bare scene location, keeping entrances and timed moves intact."""
    text = sanitize_h3_prompt_text(value).strip(" ,;:-.!?")
    lead = (
        rf"(?:{_H3_FACT_SUBJECT}\s+(?i:is|are)|"
        r"(?i:the\s+(?:scene|fight|story|action)\s+(?:is\s+set|takes\s+place)))"
    )
    match = re.fullmatch(rf"{lead}\s+(?i:in|inside|at)\s+(.+)", text)
    if not match:
        return False
    location = match.group(1)
    # Do not consume a move, an action, or a temporal qualification as scenery.
    if re.search(r"\b(?:and|but|then|after|before|while|until|once|when|now|"
                 r"finally|into|from|through)\b", location, re.I) or _CAST_ACTION_RE.search(location):
        return False
    return bool(re.fullmatch(
        r"(?:[\w'’-]+\s+){0,8}(?:temple|courtyard|monastery|mountains?|"
        r"castle|forest|desert|beach|city|village|street|rooftop|room|office|"
        r"kitchen|garage|hall|station|warehouse|arena)", location, re.I,
    ))


def _h3_persistent_instruction_spans(source: str) -> list[tuple[int, int, str]]:
    """Find unheaded production directions that constrain, but do not enact, the story."""

    from services.adaptive_enhancement import is_writing_instruction

    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"(?:^|(?<=[.!?;])\s+|[\r\n]+)([^.!?;\r\n]+[.!?]?)", source):
        text = sanitize_h3_prompt_text(match.group(1)).strip(" \t\r\n")
        direction = _H3_IMPERATIVE_DIRECTION_RE.match(text.rstrip(".!?"))
        if (
            (direction and _H3_CONTINUITY_DIRECTION_RE.search(direction.group("body")))
            or _is_persistent_camera_directive(text)
            or _is_creation_directive(text)
            or _is_persistent_action_direction(text)
            or _is_h3_preservation_contract(text)
            or _is_h3_negative_constraint_fragment(text)
            or _is_h3_performance_direction(text)
            or _is_h3_no_addition_contract(text)
            or _is_persistent_ability_fact(text)
            or _is_persistent_setting_fact(text)
            or is_writing_instruction(text.rstrip(".!?"))
        ):
            start, end = match.span(1)
            spans.append((start, end, text))
    return spans


def _h3_production_note_spans(source: str) -> tuple[tuple[int, int], ...]:
    """Include native Context-IR metadata fields in descriptive spans.

    A Studio prompt can contain a partial native wrapper: a story-bearing
    ``integrated_multimodal_description`` followed by sound and music fields.
    Those trailing field values remain global production context and must not
    become timed story events or heuristic cast names.
    """

    spans = list(production_note_spans(source))
    fields = list(_CONTEXT_IR_FIELD.finditer(source))
    for index, match in enumerate(fields):
        if match.group(1).casefold() not in {
            "subject_definitions", "retention_analysis",
            "overall_soundscape", "non_diegetic_music",
        }:
            continue
        end = fields[index + 1].start() if index + 1 < len(fields) else len(source)
        spans.append((match.start(), end))
    return tuple(sorted(set(spans)))


def _h3_without_production_note_spans(source: str) -> str:
    """Mask headed production notes while preserving source offsets."""

    masked = source
    for start, end in reversed(_h3_production_note_spans(source)):
        masked = masked[:start] + (" " * (end - start)) + masked[end:]
    return masked


def _is_h3_imperative_name_occurrence(source: str, start: int, end: int, name: str) -> bool:
    """Reject a clause-leading direction verb without blacklisting a person's name."""

    before_clause = re.split(r"[.!?;\r\n]", source[:start])[-1]
    if before_clause.strip(" \t,;:-"):
        return False
    after = source[end:end + 100]
    instruction_spans = [
        text for span_start, span_end, text in _h3_persistent_instruction_spans(source)
        if span_start <= start < span_end
    ]
    if any(
        _is_creation_directive(text) or _is_persistent_camera_directive(text)
        for text in instruction_spans
    ):
        return True
    if name.casefold() not in {"keep", "maintain", "preserve", "ensure"}:
        return False
    if instruction_spans:
        return True
    predicate = after.lstrip(" ,;:-")
    finite = re.match(r"[A-Za-z]+", predicate)
    if (
        finite
        and _CAST_ACTION_RE.match(predicate)
        and (
            finite.group(0).casefold().endswith(("s", "ed"))
            or finite.group(0).casefold() in {
                "flew", "said", "sat", "saw", "spoke", "stood", "told", "was", "were",
            }
        )
    ):
        return False
    return not bool(re.match(
        rf"\s+and\s+{_PROPER_NAME.pattern}\s+(?:{_ACTION_VERBS})(?:s|es|ed|ing)?\b",
        after,
        flags=re.IGNORECASE,
    ))


def _is_h3_physical_direction_name_occurrence(
    source: str, start: int, end: int, name: str,
) -> bool:
    """Accept named participants inside a visible imperative staging clause."""

    clause_start = max(
        source.rfind(mark, 0, start) for mark in (".", "!", "?", ";", "\n", "\r")
    ) + 1
    clause_end_candidates = [
        pos for mark in (".", "!", "?", ";", "\n", "\r")
        if (pos := source.find(mark, end)) >= 0
    ]
    clause_end = min(clause_end_candidates, default=len(source))
    clause = sanitize_h3_prompt_text(source[clause_start:clause_end]).strip()
    direction = _H3_IMPERATIVE_DIRECTION_RE.match(clause)
    if not direction or _H3_CONTINUITY_DIRECTION_RE.search(direction.group("body")):
        return False
    return start >= clause_start + direction.start("body")
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
        # Sentence-level chronology words can be captured with the following
        # proper name (``Next Luis says``). They are discourse markers, not a
        # first name. Strip them only when a second capitalized name follows.
        name = re.sub(
            r"^(?:first|next|then|finally|afterward|later)\s+"
            r"(?=[A-Z][A-Za-z0-9_'’-]*(?:\s|$))",
            "",
            name,
            flags=re.IGNORECASE,
        ).strip()
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
    packaged = [name for name in canonical if _is_h3_packaged_name(name)]
    replacements = _h3_reference_name_replacements(
        [name for name in canonical if name not in packaged], packaged,
    )
    return list(dict.fromkeys(replacements.get(name, name) for name in canonical))


def _is_h3_packaged_name(name: str) -> bool:
    return bool(re.search(r"^minimaxh3_|(?:[_.\s-]refmod|\.safetensors)$", name, re.IGNORECASE))


def _h3_reference_name_replacements(names: list[str], references: list[str]) -> dict[str, str]:
    """Resolve packaged labels only when both sides have one unique owner.

    Share the renderer's exact RefMod aliases; never guess by substring or
    merge two versions of a saved character through their common base name.
    """
    from models.minimax_h3.speakers import _ref2va_alias_values

    references = list(dict.fromkeys(references))
    names = list(dict.fromkeys(names))
    reference_keys = {name: set(_ref2va_alias_values({"character_name": name})) for name in references}
    proposals: dict[str, list[str]] = {}
    for name in names:
        keys = set(_ref2va_alias_values({"character_name": name}))
        matches = [reference for reference, aliases in reference_keys.items() if keys & aliases]
        if len(matches) == 1 and _is_h3_packaged_name(matches[0]):
            proposals.setdefault(matches[0], []).append(name)
    return {reference: owners[0] for reference, owners in proposals.items() if len(owners) == 1}


def canonicalize_h3_reference_names(context: str, cast_names: list[str]) -> str:
    """Give Subject declarations the same names used by the requested cast."""
    replacements = _h3_reference_name_replacements(cast_names, _reference_h3_cast_names(context))
    result = str(context or "")
    for reference, name in replacements.items():
        result = re.sub(
            r"(<Subject\s+\d+>\s+is\s+)" + re.escape(reference) + r"(?=\s+from\s+<|[,;.\r\n]|$)",
            lambda match: match.group(1) + name, result, flags=re.IGNORECASE,
        )
    return result


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
    original = _h3_reference_name_replacements(names, [original]).get(original, original)
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
    source = normalize_h3_dialogue_tags(str(prompt or ""))
    # Reported arrivals inside speech are not visible entrances. Extract before
    # collapsing line breaks so unquoted screenplay rows remain recognizable.
    for line in reversed(extract_locked_dialogue(source)):
        start, end = int(line["source_offset"]), int(line["source_end"])
        source = source[:start] + " " * (end - start) + source[end:]
    source = sanitize_h3_prompt_text(source)
    entrance = re.search(
        rf"(?P<entrant>{proper})\s+(?i:"
        r"enter(?:s|ed|ing)?|arriv(?:e|es|ed|ing)?|"
        r"(?:walk|run|step|come)(?:s|ed|ing)?\s+(?:in|into|through)|"
        r"burst(?:s|ed|ing)?\s+(?:in|into|through))\b",
        source,
    )
    if entrance:
        # Crossing a room's threshold is not necessarily the character's
        # first appearance. A listener already visible in the doorway must
        # not acquire a contradictory "outside the frame" opening contract.
        entrant = entrance.group("entrant")
        if re.search(
            rf"(?<![\w]){re.escape(entrant)}(?![\w])\s*,?\s*"
            r"(?:(?:is|was)\s+)?(?:already\s+)?"
            r"(?:stands?|standing|sits?|sitting|waits?|waiting|"
            r"watches|watching|listens?|listening|holds?|holding)\b",
            source[:entrance.start()],
            flags=re.IGNORECASE,
        ):
            return None
    return entrance


def _infer_h3_opening_state_contract(
    prompt: Any,
    cast_names: list[str],
) -> str:
    """Describe the frame immediately before a requested entrance begins."""

    proper = _PROPER_NAME.pattern
    entrance = _find_h3_opening_entrance(prompt)
    if not entrance:
        return ""
    entrant = _resolve_h3_cast_name(entrance.group("entrant"), cast_names)
    stationary = re.search(
        rf"(?P<target>{proper})\s*,\s*(?i:who\s+is)\s+"
        r"(?P<state>[^.!?]{1,140})",
        entrance.string[entrance.end():],
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
            "the supplied image reference", "the supplied video reference",
        )):
            names.append(name)
    return list(dict.fromkeys(names))


def _source_requests_multiple_cast_instances(prompt: Any, name: Any) -> bool:
    """Allow explicitly requested twins, clones, copies, or multiple versions."""

    source = str(prompt or "")
    escaped = re.escape(sanitize_h3_prompt_text(name))
    if not escaped:
        return False
    for profile in explicit_character_profiles(source):
        if _normalize_key(profile["name"]) == _normalize_key(name) and re.search(
            r"\b(?:horde|crowd|army|swarm|group|ensemble|team)\s*(?:of\b|$)", profile["details"], re.I,
        ):
            return True
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

    profile_names = [item["name"] for item in explicit_character_profiles(source)] + [
        sanitize_h3_prompt_text(match.group("speaker"))
        for match in _SCREENPLAY_DIALOGUE_RE.finditer(source)
        if _is_screenplay_speaker_label(match.group("speaker"))
        and _SCREENPLAY_PROFILE_TEXT_RE.match(match.group("text").strip())
    ]
    spoken_names = [
        sanitize_h3_prompt_text(item.get("speaker"))
        for item in extract_locked_dialogue(source)
        if sanitize_h3_prompt_text(item.get("speaker")) not in {"", "Speaker"}
    ]
    # Explicit cast definitions plus a closed-cast instruction are stronger
    # evidence than incidental capitalized camera/VFX prose ("Lens", "Fist",
    # "Massive Mach", etc.). Keep real authored speakers, including silent
    # profile owners who would otherwise be missed by the name heuristics.
    profile_count = len({name.casefold() for name in profile_names})
    count_words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    quantity = str(profile_count)
    if profile_count < len(count_words):
        quantity += "|" + count_words[profile_count]
    # Standalone cast limits also appear as "No second person." Do not match
    # a local staging instruction such as "No second person enters until...".
    next_person = {1: "second", 2: "third"}.get(profile_count)
    closed_cardinality = bool(next_person and re.search(
        rf"\bno\s+{next_person}\s+(?:person|character|actor)\s*(?:[.!?;]|$)",
        source, flags=re.IGNORECASE,
    ))
    if profile_names and (closed_cardinality or re.search(
        r"\bno\s+(?:third\s+parties|(?:other|extra|additional)\s+"
        r"(?:characters|people|persons|cast\s+members))\b|"
        rf"\bonly\s+(?:{quantity})\s+[^.!?\r\n]{{0,80}}"
        r"\b(?:characters|people|persons|men|women|males|females|artists|fighters|monks)\b",
        source, flags=re.IGNORECASE,
    )):
        return _canonicalize_h3_cast_names(profile_names + spoken_names, prompt=source)
    speakers = {
        name.casefold() for name in spoken_names
    }
    timed_events = authored_timed_brief(source)["events"]
    evidence_source = ". ".join(event["text"] for event in timed_events) if timed_events else source
    cast: list[str] = list(profile_names)
    for name in proper_names:
        clean = sanitize_h3_prompt_text(name)
        if not clean:
            continue
        if any(clean.casefold() == profile.casefold() + "s"
               and _source_requests_multiple_cast_instances(source, profile)
               for profile in profile_names):
            continue  # A declared crowd's plural is not another principal.
        if clean.casefold() in speakers:
            cast.append(clean)
            continue
        if clean.split()[-1].casefold() in _SETTING_NAME_SUFFIXES:
            continue
        evidence = False
        for match in re.finditer(
            rf"(?<![\w]){re.escape(clean)}(?![\w])",
            evidence_source,
            flags=re.IGNORECASE,
        ):
            before = evidence_source[max(0, match.start() - 90):match.start()]
            after = evidence_source[match.end():match.end() + 100]
            before_clause = re.split(r"[.!?;]", before)[-1]
            after_clause = re.split(r"[.!?;]", after)[0]
            if _is_h3_imperative_name_occurrence(
                evidence_source, match.start(), match.end(), clean
            ):
                continue
            if _is_h3_physical_direction_name_occurrence(
                evidence_source, match.start(), match.end(), clean
            ):
                evidence = True
                break
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
            # A verb later in the clause may belong to somebody else:
            # "WHIP whip-pan: Nora turns" does not make WHIP an actor.
            predicate = re.sub(r"^(?:\s+[\w-]+ly\b)*\s*", "", after_clause)
            performs = bool(_CAST_ACTION_RE.match(predicate) if timed_events
                            else _CAST_ACTION_RE.search(after_clause))
            participates = bool(_CAST_INTERACTION_RE.search(before_clause) or re.search(
                r"\b(?:near|beside|next\s+to)\s+\w+\s+(?:is|stands?)\s*$",
                before_clause, flags=re.I,
            ))
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
    return _canonicalize_h3_cast_names(merged + spoken_names, prompt=source)


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
    raw_directive_source = video_direction_source(_without_locked_dialogue(
        raw_source,
        locked_dialogue,
        keep_screenplay_speaker_cues=True,
    ))
    directive_source = sanitize_h3_prompt_text(raw_directive_source)
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
    # Section boundaries must be interpreted before whitespace is flattened.
    timed_events = authored_timed_brief(raw_directive_source)["events"]
    names_source = (
        ". ".join(event["text"] for event in timed_events)
        if timed_events else _h3_without_production_note_spans(raw_directive_source)
    )
    # Authored timeline headings label blocks rather than people. Mask the
    # whole bracket before scanning proper names so arbitrary titles such as
    # Establish, Fill, Water, or a user's custom phase name cannot become cast.
    names_source = re.sub(
        r"\[\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*s?"
        r"(?:\s*\|[^\]]*)?\]",
        " ",
        names_source,
        flags=re.IGNORECASE,
    )
    names_source = sanitize_h3_prompt_text(names_source)
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3}\b",
        names_source,
    ):
        name = sanitize_h3_prompt_text(match.group(0))
        # Full numbered cast definitions are collected separately. A numbered
        # production label must not contribute just its noun as a new actor.
        if re.match(r"\s+\d+\b", names_source[match.end():]):
            continue
        parts = name.split()
        if parts and parts[0].casefold() in {
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        } and re.match(
            r"\s+(?:[\w-]+\s+){0,4}(?:people|men|women|friends|fighters|artists|monks|characters|actors|"
            r"soldiers|children|siblings|colleagues|coworkers|jedi)\b",
            names_source[match.start() + len(parts[0]):], re.I,
        ):
            continue
        while len(parts) > 1 and parts[0].casefold() in {
            "a", "an", "and", "both", "in", "keep", "maintain", "make", "preserve",
            "ensure", "the", "first", "next", "then", "finally", "afterward", "later",
            "extremely", "epic",
        }:
            parts.pop(0)
        name = " ".join(parts)
        if name.casefold() in {"finish", "begin", "end", "start"} and re.match(
            r"\s+(?:(?:the\s+)?(?:scene|video|film)\s+)?(?:with|by)\b",
            names_source[match.end():], flags=re.I,
        ):
            continue
        if name.casefold() in {
            "a", "an", "and", "both", "in", "make", "the", "then", "extremely", "epic",
            "friends", "maestro", "each", "every", "rear", "flanking", "residual",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "hundreds", "thousands", "millions", "myriad",
        } | _NON_CAST_PROPER_NAMES or (name.isupper() and len(name) <= 3):
            continue
        if (
            len(parts) == 1
            and name.casefold() in {"one", "single"}
            and re.match(
                r"\s+(?:continuous|unbroken|real[- ]time|tracking|camera|shot|take)\b",
                names_source[match.end():],
                flags=re.IGNORECASE,
            )
        ):
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
    # A filmed subject holding a prop does not put that prop in the camera
    # operator's hands. Require an explicit viewpoint-owned hand direction.
    hands_visible = pov and any(
        re.search(r"\b(?:hold\w*|grip\w*|grasp\w*)\b", clause, re.I)
        and re.search(
            r"\b(?:my|our|your|viewer['’]s|camera operator['’]s)\s+"
            r"(?:\w+\s+){0,2}hands?\b|"
            r"^\s*(?:(?:show|keep)\s+)?(?:(?:both|two)\s+)?hands?\b|"
            r"\b(?:first[- ]person|POV)\b[^.!?]{0,50}\bhands?\b",
            clause, re.I,
        )
        for clause in re.split(r"(?<=[.!?])\s+|[\r\n]+", raw_directive_source)
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

    positive_direction = sanitize_h3_prompt_text(positive_instruction_text(raw_directive_source))
    energetic_performance = bool(_ENERGETIC_PERFORMANCE_RE.search(positive_direction))
    pacing = (
        "extremely fast real-time movement with sustained forward momentum, decisive choreography, and no slow motion"
        if _FAST_ACTION_RE.search(positive_direction)
        else "brisk, energetic real-time pacing with immediate expressive performance and no slow motion"
        if energetic_performance
        else "natural real-time pacing"
    )
    requested_slow_motion = any(
        not re.search(
            r"\b(?:no|not|without|never|avoid)\s+(?:any\s+)?$|"
            r"\b(?:no|without)\s+(?:(?:cuts?|dialogue|speech|music|magic|subtitles?)\s*,\s*)+(?:(?:and|or)\s+)?$|"
            r"\b(?:prohibited|forbidden|disallowed)\s*:[^.!?]*$",
            positive_direction[:match.start()],
            flags=re.IGNORECASE,
        )
        for match in re.finditer(
            r"\b(?:slow[- ]mo(?:tion)?|time\s+dilation)\b",
            positive_direction, flags=re.IGNORECASE,
        )
    )
    if requested_slow_motion:
        pacing = (
            "Brief requested slow-motion accents and time-dilation at specified beats, then "
            "immediate high-speed bursts; preserve explosive acceleration and momentum between accents"
            if _FAST_ACTION_RE.search(positive_direction) else
            "Follow the requested changes of pace: preserve slow-motion and time-dilation "
            "at the specified beats, with returns to real-time action as directed"
        )
    ambient_parts: list[str] = []
    if re.search(r"\bmountain|cliff|canyon|clouds?\b", lowered):
        ambient_parts.append("open-air mountain wind")
    if _FAST_ACTION_RE.search(directive_source):
        ambient_parts.append("speed-dependent rushing air")
    video_source = video_direction_source(raw_source)
    video_dialogue = locked_dialogue if video_source == raw_source else extract_locked_dialogue(video_source)
    timed_context = authored_timed_brief(video_source)["context"]
    persistent_directions = [
        text for start, end, text in _h3_persistent_instruction_spans(video_source)
        if not any(start < line["source_end"] and line["source_offset"] < end
                   for line in video_dialogue)
    ]
    production_directions = [
        sanitize_h3_prompt_text(video_source[start:end])
        for start, end in _h3_production_note_spans(video_source)
    ]
    global_instructions = "\n\n".join(dict.fromkeys(
        item for item in [timed_context, *persistent_directions, *production_directions] if item
    ))
    return {
        "first_person_pov": pov,
        "pov_identity": pov_identity,
        "proper_names": proper_names,
        "cast_names": cast_names,
        "cast_profiles": explicit_character_profiles(raw_source),
        "global_instructions": global_instructions,
        "negative_constraints": explicit_negative_constraints(raw_source),
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


def _source_owns_h3_subject_appearance(
    prompt: Any,
    *,
    start_frame_supplied: bool = False,
    reference_context: Any = "",
) -> bool:
    """Return whether media or structured source text owns visual identity.

    An open one-line concept can benefit from writer-authored visual design.
    A Frames/Ref2VA input or native Context-IR prompt already owns identity,
    appearance, opening state and physical mechanics; appending a speculative
    writer profile can contradict the supplied person or named character.
    """

    reference_text = sanitize_h3_prompt_text(reference_context)
    visual_reference = bool(re.search(
        r"\bvisual\s+(?:identity|appearance)\b|\bappearance\s+reference\b|"
        r"\bsaved\s+reference\s+character\b|"
        r"\bpreserv(?:e|es|ed|ing)\s+(?:the\s+)?identity\b|"
        r"\bexact\s+(?:supplied\s+)?first\s+frame\b",
        reference_text,
        flags=re.IGNORECASE,
    ))
    return bool(
        start_frame_supplied
        or visual_reference
        or _CONTEXT_IR_FIELD.search(str(prompt or ""))
    )


def _lock_h3_source_owned_context(
    ledger: dict[str, Any], canonical: dict[str, Any], *,
    lock_initial_state: bool = True,
    lock_mechanics: bool = True,
) -> None:
    """Restore visual facts that a structured source or media input owns."""

    fields = ["subject_continuity", "required_final_outcome"]
    if lock_initial_state:
        fields.append("initial_state")
    if lock_mechanics:
        fields.append("motion_mechanics")
    for field in fields:
        ledger[field] = sanitize_h3_prompt_text(canonical.get(field))


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
    role = re.search(
        rf"\b(?P<speaker>{_DESCRIPTIVE_PERSON})\s+"
        r"(?:(?:then|[\w-]+ly)\s+)*$",
        prefix[:verb.start()], flags=re.I,
    )
    if role:
        speaker = _descriptive_speaker_name(source[:quote_start], role.group("speaker"))
    speaker = re.sub(
        r"^(?:first|next|then|finally|afterward|later)\s+"
        r"(?=[A-Z][A-Za-z0-9_'’-]*(?:\s|$))",
        "",
        speaker,
        flags=re.IGNORECASE,
    ).strip()
    post_modifier = sanitize_h3_prompt_text(prefix[verb.end():]).strip(" ,;:-")
    pre_modifier = ""
    if names and not role:
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


def _is_screenplay_speaker_label(value: Any) -> bool:
    """Keep production headings out of both screenplay and quoted dialogue."""

    key = _normalize_key(value)
    return bool(key and not is_h3_production_label(key))


def _screenplay_dialogue_spans(source: str) -> list[dict[str, Any]]:
    """Return unambiguous ``CHARACTER: line`` screenplay rows.

    A colon is also used by camera notes and Context-IR, so only a short
    person-like label at the start of its own line is accepted.  The body is
    preserved verbatim except for optional wrapping quotation marks/Markdown.
    """

    candidates = [
        match for match in _SCREENPLAY_DIALOGUE_RE.finditer(source)
        if _is_screenplay_speaker_label(match.group("speaker"))
    ]
    notes = production_note_spans(source)
    profiles = character_profile_spans(source)
    outside_notes = source
    for start, stop in reversed(notes):
        outside_notes = outside_notes[:start] + " " * (stop - start) + outside_notes[stop:]
    established_names = {
        _normalize_key(match.group()) for match in _PROPER_NAME.finditer(outside_notes)
    } | {_normalize_key(item["name"]) for item in explicit_character_profiles(source)}
    # A brief can label character descriptions exactly like screenplay turns.
    # Honor scene-wide silence for ambiguous, unquoted rows, but do not infer
    # silence from words a character actually says ("No dialogue today.").
    brief = source
    for match in reversed(candidates):
        brief = brief[:match.start()] + "\n" + brief[match.end():]
    brief = re.sub(r"<d>.*?</d>", "", brief, flags=re.IGNORECASE | re.DOTALL)
    silent_brief = dialogue_forbidden(_DIALOGUE_QUOTE_RE.sub("", brief))

    spans: list[dict[str, Any]] = []
    for match in candidates:
        speaker = sanitize_h3_prompt_text(match.group("speaker")).strip(" ,;:.-")
        raw_text = str(match.group("text") or "")
        text = raw_text.strip()
        end = match.end()
        quoted = _DIALOGUE_QUOTE_RE.match(text)
        tagged = re.match(r"<d>", text, flags=re.IGNORECASE)
        in_notes = any(start <= match.start() < stop for start, stop in notes)
        # Cast descriptions stay descriptive even when they contain a quoted
        # nickname. Other production sections can contain real screenplay
        # turns; a heading alone must not swallow a complete spoken quote or
        # an established character's first/second-person utterance.
        explicit_delivery = bool(_SPEECH_VERB.search(match.group("delivery") or ""))
        if in_notes and not tagged and not explicit_delivery:
            in_profile = any(start <= match.start() < stop for start, stop in profiles)
            complete_quote = bool(quoted and not text[quoted.end():].strip(" \t*"))
            established_utterance = bool(
                _normalize_key(speaker) in established_names
                and re.search(r"\b(?:I|me|my|mine|we|us|our|ours|you|your|yours)\b", text, re.IGNORECASE)
                and not _SCREENPLAY_PROFILE_TEXT_RE.match(text)
            )
            if in_profile or not (complete_quote or established_utterance):
                continue
        if not quoted and not tagged and (
            silent_brief or _SCREENPLAY_PROFILE_TEXT_RE.match(text)
        ):
            # Keep cast/profile prose in the visual story. Explicit quoted or
            # tagged lines still belong to the user and retain timing checks.
            continue
        if quoted:
            # A quoted turn ends at its closing quote. Directions after it
            # remain source events instead of consuming the speech budget.
            if text[quoted.end():].strip(" \t*"):
                end = (
                    match.start("text")
                    + len(raw_text) - len(raw_text.lstrip())
                    + quoted.end()
                )
            text = quoted.group(1) or quoted.group(2)
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
            "end": end,
            "content_start": match.start("text"),
            "text": text,
            "speaker": speaker,
            "language": "English",
            "delivery": delivery,
            "off_camera": off_camera,
            "explicit_tag": False,
            "source_form": "screenplay",
        })
    return spans


_SPEECH_ACTION_LINK_RE = re.compile(
    r"\s*[,;]?\s*\b(?:(?:right|immediately|just|only)\s+)?"
    r"(?:before|after|while|as|when|until)\b", re.IGNORECASE,
)


def _without_locked_dialogue(
    source: str,
    locked_dialogue: list[dict[str, Any]],
    *,
    keep_screenplay_speaker_cues: bool,
) -> str:
    """Remove spoken words while preserving chronological screenplay cues."""

    from models.minimax_h3.speakers import h3_action_beat_speaker

    cleaned = source
    for item in reversed(locked_dialogue):
        start = int(item.get("source_offset") or 0)
        end = int(item.get("source_end") or start)
        linked_action = bool(_SPEECH_ACTION_LINK_RE.match(cleaned[end:]))
        replacement = " " if linked_action else " . "
        if keep_screenplay_speaker_cues and (
            item.get("source_form") == "screenplay"
            or h3_action_beat_speaker(source, start)
        ):
            speaker = sanitize_h3_prompt_text(item.get("speaker")) or "Speaker"
            replacement = f"\n{speaker} speaks" + (" " if linked_action else ".\n")
        cleaned = cleaned[:start] + replacement + cleaned[end:]
    return cleaned


def _explicit_speech_action_order(prompt: str) -> dict[str, dict[str, str]]:
    """Keep explicit speech/action dependencies in the camera writing contract."""
    source = normalize_h3_dialogue_tags(prompt)
    order = {}
    for line in extract_locked_dialogue(source):
        suffix = source[int(line['source_end']):]
        link = _SPEECH_ACTION_LINK_RE.match(suffix)
        if not link:
            continue
        action = sanitize_h3_prompt_text(re.split(r'[.!?\r\n]', suffix[link.end():], maxsplit=1)[0])
        if not action:
            continue
        relation = link.group().strip(' ,;').split()[-1].casefold()
        slot = {'before': 'after_speech', 'after': 'before_speech'}.get(relation, 'during_speech')
        order[line['dialogue_id']] = {slot: action}
    return order


def extract_locked_dialogue(prompt: str) -> list[dict[str, Any]]:
    """Extract tagged, quoted, or screenplay-form dialogue before rewriting."""

    from models.minimax_h3.speakers import h3_action_beat_speaker, is_h3_spoken_quote

    source = normalize_h3_dialogue_tags(prompt)
    tag_pattern = re.compile(
        r"<d>\s*(?:\[([^\]\r\n]+)\])?\s*((?:(?!<d>).)*?)\s*</d>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    spans: list[dict[str, Any]] = []
    for match in tag_pattern.finditer(source):
        text = sanitize_h3_prompt_text(match.group(2) or "").strip()
        if not text or _PLACEHOLDER_DIALOGUE.fullmatch(text):
            continue
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "text": text,
            "language": sanitize_h3_prompt_text(match.group(1) or "English"),
            "explicit_tag": True,
            "source_form": "tagged",
        })
    for item in _screenplay_dialogue_spans(source):
        tagged = [
            span for span in spans
            if span["explicit_tag"]
            and int(item["start"]) < int(span["end"])
            and int(span["start"]) < int(item["end"])
        ]
        if tagged:
            # ``Name: <d>line</d>`` is one line, not a screenplay row plus
            # another tagged line. Retain the leading speaker/delivery cue
            # without absorbing any action after the closing tag.
            for span in tagged:
                if int(span["start"]) == int(item["content_start"]):
                    span.update({
                        "start": item["start"],
                        "speaker": item["speaker"],
                        "delivery": item["delivery"],
                        "off_camera": item["off_camera"],
                        "source_form": "screenplay",
                    })
            continue
        spans.append(item)
    occupied_ranges = [(int(span["start"]), int(span["end"])) for span in spans]
    notes = production_note_spans(source)
    for match in _DIALOGUE_QUOTE_RE.finditer(source):
        if any(match.start() < end and start < match.end() for start, end in occupied_ranges):
            continue
        in_notes = any(start <= match.start() < end for start, end in notes)
        if not is_h3_spoken_quote(source, match, allow_screenplay_label=not in_notes):
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
        action_speaker = h3_action_beat_speaker(source, start)
        speaker, delivery = (
            (
                sanitize_h3_prompt_text(span.get("speaker")) or "Speaker",
                sanitize_h3_prompt_text(span.get("delivery")) or "speaks naturally",
            )
            if span.get("speaker") else (action_speaker, "speaks naturally")
            if action_speaker else _infer_quote_speaker(
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
        if label and not _is_screenplay_speaker_label(label.group(1)):
            label = None
        outside_quote = (source[:start] + source[end:]).strip(" \t\r\n.,;:!?-")
        if (
            not span["explicit_tag"]
            and not screenplay
            and not _SPEECH_VERB.search(nearby)
            and not label
            and not action_speaker
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
    # Descriptive sections constrain the scene globally. Retain any explicit
    # speech in them, but do not schedule every costume/specification sentence
    # as a chronological event. Recompute offsets after removing the sections.
    for start, end in reversed(_h3_production_note_spans(raw_source)):
        speech = "\n".join(
            f'{item["speaker"]}{" (off-camera)" if item.get("off_camera") else ""}: '
            f'<d>[{item["language"]}]{item["text"]}</d>'
            for item in locked_dialogue if start <= item["source_offset"] < end
        )
        raw_source = raw_source[:start] + "\n" + speech + "\n" + raw_source[end:]
    locked_dialogue = extract_locked_dialogue(raw_source)
    has_screenplay_dialogue = any(
        item.get("source_form") == "screenplay" for item in locked_dialogue
    )
    raw_source = _without_locked_dialogue(
        raw_source,
        locked_dialogue,
        keep_screenplay_speaker_cues=True,
    )
    # Persistent continuity directions constrain every shot but are not
    # themselves timed visible actions. Named physical staging imperatives,
    # such as ``Keep Mara beside the door``, intentionally remain here.
    for start, end, _text in reversed(_h3_persistent_instruction_spans(raw_source)):
        raw_source = raw_source[:start] + ". " + raw_source[end:]
    source = sanitize_h3_prompt_text(
        re.sub(
            r"(?:\r?\n)+",
            ". ",
            raw_source,
        )
    )
    boundaries = re.compile(
        r"(?<=[.!?])\s+|\s*;(?!\s*(?:it|this|that|thereby)\b)\s*|"
        r"(?P<temporal>\s+(?:(?:and\s+)?then(?:\s+then)*|after\s+that|next(?!\s+to\b))\s+)|"
        r",\s+(?:but|and)\s+|"
        r"\s+(?=(?:hard\s+cut|smash\s+cut|match\s+cut|cut\s+to|whip\s+pan)\b)",
        flags=re.IGNORECASE,
    )
    pieces: list[str] = []
    start = 0
    clause_prefix = ""
    for boundary in boundaries.finditer(source):
        head = (clause_prefix + source[start:boundary.start()]).strip(" ,")
        if boundary.group("temporal") and (
            _is_subject_only_fragment(head)
            or re.fullmatch(r"they|he|she|it|the viewer", head, flags=re.IGNORECASE)
        ):
            # "Michael then invites/offers/apologizes ..." has only one
            # predicate. Splitting before that predicate strands its actor;
            # preserving the clause works without an exhaustive verb list.
            clause_prefix = head + " "
            start = boundary.end()
            continue
        pieces.append(clause_prefix + source[start:boundary.start()])
        clause_prefix = ""
        start = boundary.end()
    pieces.append(clause_prefix + source[start:])
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
            rf"\s*(?:(?:and\s+)?then\s+)?(?P<subject>{_DESCRIPTIVE_PERSON}|they|he|she|it|the\s+viewer)\b",
            piece,
            flags=re.IGNORECASE,
        )
        if not subject_match:
            subject_match = re.match(
                r"\s*(?:(?:and\s+)?then\s+)?(?P<subject>"
                r"[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})\b",
                piece,
            )
        if (
            subject_match
            and subject_match.group("subject").casefold() in {
                "after", "at", "before", "continuing", "during", "finally",
                "first", "in", "inside", "later", "next", "on", "one", "then",
            }
        ):
            subject_match = None
        if not subject_match:
            # Introductory setting and timing phrases commonly precede the
            # actual actor. Find the nearest explicit pronoun/name that owns a
            # finite action instead of carrying ``At`` or ``After`` as cast.
            subject_match = re.search(
                rf"\b(?P<subject>they|he|she|it|the\s+viewer)"
                rf"\s+(?=(?:{_ACTION_VERBS})(?:s|es|ed|ing)?\b)",
                piece,
                flags=re.IGNORECASE,
            )
        if not subject_match:
            subject_match = re.search(
                rf"\b(?P<subject>[A-Z][A-Za-z0-9_'’-]*"
                rf"(?:\s+[A-Z][A-Za-z0-9_'’-]*){{0,3}})"
                rf"\s+(?=(?i:(?:{_ACTION_VERBS})(?:s|es|ed|ing)?\b))",
                piece,
            )
        if not subject_match:
            subject_match = re.search(
                r"\b(?:adult|young|elderly)\s+(?:[a-z][a-z'-]*\s+){0,3}"
                r"(?P<subject>[A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,2})"
                r"(?=\s+[a-z])",
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
        ) or _SPEECH_VERB.match(piece) or _CAST_ACTION_RE.match(piece.lstrip(" ,")))
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
                elif _SPEECH_VERB.match(value) or _CAST_ACTION_RE.match(value):
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
                or _is_h3_preservation_contract(value)
                or _is_h3_negative_constraint_fragment(value)
                or _is_h3_performance_direction(value)
                or _is_h3_no_addition_contract(value)
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


def extract_source_events(prompt: str) -> list[dict[str, Any]]:
    """Create immutable source-event IDs used to prove coverage exactly once."""

    normalized = normalize_h3_dialogue_tags(prompt)
    timed = authored_timed_brief(normalized)["events"]
    if timed:
        locked = extract_locked_dialogue(normalized)
        def timed_event_text(item: dict[str, Any]) -> str:
            text = sanitize_h3_prompt_text(_without_locked_dialogue(
                item["text"], [
                    {**line, "source_offset": line["source_offset"] - item["source_offset"],
                     "source_end": line["source_end"] - item["source_offset"]}
                    for line in locked
                    if item["source_offset"] <= line["source_offset"] < item["source_end"]
                ], keep_screenplay_speaker_cues=True,
            ))
            parts = re.split(r"(?<=[.!?])\s+", text)
            kept = [
                part for part in parts
                if not (
                    _is_h3_preservation_contract(part)
                    or _is_h3_negative_constraint_fragment(part)
                    or _is_h3_performance_direction(part)
                    or _is_h3_no_addition_contract(part)
                )
            ]
            return sanitize_h3_prompt_text(" ".join(kept))
        return [
            {
                **item,
                "event_id": f"E{index + 1}",
                "text": timed_event_text(item),
            }
            for index, item in enumerate(timed)
            if timed_event_text(item)
        ]
    from services.adaptive_enhancement import is_writing_instruction

    return [
        {"event_id": f"E{index + 1}", "text": fragment}
        for index, fragment in enumerate(
            fragment for fragment in _story_fragments(prompt)
            if not is_writing_instruction(fragment)
        )
    ]


def _find_spoken_verb(text: str) -> re.Match[str] | None:
    """Find a vocal cue without treating an object's state as speech."""

    for match in _PLANNER_SPEECH_VERB.finditer(text):
        before, after = text[:match.start()], text[match.end():]
        # Visual evidence can "suggest" something without anyone speaking.
        # Treating that relative clause as dialogue also detached a prop's
        # movement and assigned it to the preceding named character.
        if match.group(0).lower().startswith('suggest') and (
            re.search(
                r'\b(?:expressions?|looks?|gazes?|postures?|movements?|motions?|'
                r'shifts?|vibrations?|gestures?)\s+(?:(?:that|which)\s+)?$', before, re.I,
            ) or (match.group(0).lower() == 'suggesting' and re.search(r',\s*$', before))
        ):
            continue
        # "Muffled" is an acoustic adjective here, not an untagged utterance.
        if match.group(0).lower() == 'muffled' and re.match(
            r'\s+(?:(?:soft|heavy|distant|rhythmic|faint)\s+){0,2}'
            r'(?:thumps?|rumbles?|impacts?|footsteps?|knocks?|clangs?|bangs?)\b', after, re.I,
        ):
            continue
        if match.group(0).lower() in {"state", "states"}:
            if re.search(
                r"\b(?:the|an?|its|their|his|her|initial|final|current|previous|"
                r"physical|visible|open|closed|unchanged|[\w]+['’]s)\s+$", before, re.I,
            ) or re.match(
                r"\s+(?:is|are|remains?|stays?|coherent|consistent|unchanged|"
                r"matches?|must|should)\b", after, re.I,
            ):
                continue
        return match
    return None


def strip_h3_source_clock_cues(value: Any) -> str:
    """Remove absolute authored timestamps from window-local action prose.

    The compiler supplies each window and shot with its own local clock. Keep
    durations such as ``holds for two seconds`` intact; remove only explicit
    ``at/by N seconds`` and ``at the N-second mark`` source positions.
    """

    return sanitize_h3_prompt_text(re.sub(
        r"\b(?:at|by)\s+(?:the\s+)?(?:approximately\s+)?\d+(?:\.\d+)?\s*"
        r"(?:(?:[-\s]?(?:seconds?|secs?|s)\s+mark)\b|(?:seconds?|secs?|s)\b)\s*,?",
        "",
        sanitize_h3_prompt_text(value),
        flags=re.IGNORECASE,
    ))


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
    # Inline source clocks belong to the complete authored timeline. Each H3
    # continuation prompt starts its own local clock at 0.000, so copying
    # ``At 13.5 seconds`` into a later window creates a contradictory native
    # instruction. The ledger already retains authored offsets for assignment;
    # remove only explicit numeric source-clock phrases from filmable prose.
    text = strip_h3_source_clock_cues(text).strip(" \t\r\n-.,;:!?")
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
    speech = _find_spoken_verb(text)
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
    if (non_speech_actions.search(text[speech.end():])
            or _SPEECH_ACTION_LINK_RE.search(text[speech.end():])):
        # A timed block can contain speech followed by further choreography.
        # Removing everything after its first cue would delete that action.
        return text
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
    if events and all("source_offset" in event for event in events):
        # Several lines can belong to the same authored time block. Anchor
        # each by its actual source location instead of advancing one event
        # after every line and moving the next speaker to a later window.
        for dialogue in locked_dialogue:
            offset = int(dialogue.get("source_offset") or 0)
            for event in events:
                if event["source_offset"] <= offset < event["source_end"]:
                    expected[str(dialogue.get("dialogue_id") or "").upper()] = event["event_id"]
                    break
        return expected
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


def _final_source_event_segment(
    source_events: list[dict[str, Any]],
    locked_dialogue: list[dict[str, Any]],
    expected_dialogue_events: dict[str, str],
    segment_durations: list[float] | None,
) -> int:
    """Return the last native window that can actually perform the ending.

    A restored or programmatic request can leave a tiny residual window after
    the ordinary full windows. That tail remains useful for holding the visible
    consequence, but it cannot own a physical ending or a locked spoken line.
    Walk backward only while a window is shorter than the ending's minimum
    action-and-speech clock; ordinary tails still own the final source event.
    """

    durations = [max(0.1, float(value)) for value in (segment_durations or [])]
    if not source_events or not durations:
        return max(1, len(durations))
    final_event_id = str(source_events[-1].get("event_id") or "").upper()
    spoken_words = sum(
        _dialogue_word_count(item.get("text"))
        for item in locked_dialogue
        if expected_dialogue_events.get(
            str(item.get("dialogue_id") or "").upper()
        ) == final_event_id
    )
    authored_seconds = max(
        0.0,
        float(source_events[-1].get("source_end_seconds", 0) or 0)
        - float(source_events[-1].get("source_start_seconds", 0) or 0),
    )
    minimum_seconds = max(
        authored_seconds,
        1.15 + spoken_words / _H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND,
    )
    if durations[-1] + 0.05 >= minimum_seconds:
        return len(durations)
    # A short-tail repair is legal only when a preceding native window can
    # actually perform the ending. If no window fits (for example a four-second
    # authored ending adapted to uniformly shorter windows), keep the event in
    # its chronological final slot and let the ordinary feasibility guard reject
    # the impossible request instead of silently moving it earlier.
    for index in range(len(durations) - 2, -1, -1):
        if durations[index] + 0.05 >= minimum_seconds:
            return index + 1
    return len(durations)


_EXPLICIT_OPENING_DIALOGUE_DELAY_RE = re.compile(
    r"\b(?:(?:at|near|towards?|by)\s+(?:the\s+)?(?:end|ending|final|finale|climax)|"
    r"(?:only|not\s+until)\s+after|only\s+when|"
    r"much\s+later|later\s+(?:that|the\s+same)\s+(?:day|night)|"
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
    if (_EXPLICIT_OPENING_DIALOGUE_DELAY_RE.search(prefix)
            or _EXPLICIT_OPENING_DIALOGUE_DELAY_RE.search(events[event_index]["text"])):
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
            "source_event_id": {
                "type": "string", "enum": ["", *[f"E{i + 1}" for i in range(source_event_count)]],
            },
            "segment": {
                "type": "integer",
                "minimum": 1,
                "maximum": max(1, segment_count),
            },
        },
        "required": ["speaker", "language", "delivery", "text", "segment", "source_event_id"],
        "additionalProperties": False,
    }
    generated_dialogue: dict[str, Any] = {
        "type": "array",
        "items": dialogue,
        "maxItems": max(0, segment_count * 6),
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
                "items": {"type": "string", **(
                    {"enum": [f"E{i + 1}" for i in range(source_event_count)]}
                    if source_event_count else {}
                )},
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
            "motion_mechanics": {"type": "string"},
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
                # Leave space for source actions AND supporting reactions.
                # A fixed three/window cap can force a writer to omit the
                # ending once all of its slots have been used.
                "maxItems": max(1, segment_count * 3, source_event_count + segment_count * 2),
            },
            "generated_dialogue": generated_dialogue,
        },
        "required": [
            "subject_continuity", "setting_continuity", "motion_mechanics", "visual_continuity",
            "editing_style", "initial_state", "ambient_audio", "music",
            "required_final_outcome", "beats", "generated_dialogue",
        ],
        "additionalProperties": False,
    }


def _faithful_treatment_schema(
    cast_names: list[str] | None = None, *,
    resolve_adaptation: bool = False, start_frame: bool = False,
) -> dict[str, Any]:
    """Schema for the creative work that remains in faithful planning.

    A faithful request already gives Maestro an ordered source-event catalog
    and, when present, an exact dialogue catalog.  Asking a small local model
    to copy those internal IDs while also making directorial choices turns a
    straightforward writing task into brittle database transcription.  The
    model therefore supplies only the global cinematic treatment here;
    Maestro schedules the immutable story locally.
    """

    appearance = (
        {"type": "object", "properties": {name: {"type": "string"} for name in cast_names},
         "required": list(cast_names), "additionalProperties": False}
        if cast_names else {"type": "string"}
    )
    schema = {
        "type": "object",
        "properties": {
            "character_appearance": appearance,
            "setting_continuity": {"type": "string"},
            "motion_mechanics": {"type": "string"},
            "visual_continuity": {"type": "string"},
            "editing_style": {"type": "string"},
            "ambient_audio": {"type": "string"},
        },
        "required": [
            "character_appearance",
            "setting_continuity",
            "motion_mechanics",
            "visual_continuity",
            "editing_style",
            "ambient_audio",
        ],
        "additionalProperties": False,
    }
    if resolve_adaptation:
        schema["properties"]["source_adaptation"] = {"type": "string"}
        schema["required"].append("source_adaptation")
    if start_frame:
        # Ground the scene before writing style. Putting the image observation
        # last encouraged the model to copy the template's setting first and
        # then contradict it with the actual frame.
        schema["properties"] = {
            "initial_state": {"type": "string"},
            "setting_continuity": schema["properties"]["setting_continuity"],
            "character_appearance": appearance,
            **schema["properties"],
        }
        schema["required"] = list(schema["properties"])
    return schema


def _h3_contract_token_stems(value: Any) -> set[str]:
    """Normalize simple inflection for local source/camera fidelity checks."""

    stems: set[str] = set()
    for token in _content_tokens(value):
        stem = token
        if len(stem) > 5 and stem.endswith("ing"):
            stem = stem[:-3]
        elif len(stem) > 4 and stem.endswith("ied"):
            stem = stem[:-3] + "y"
        elif len(stem) > 4 and stem.endswith("ed"):
            stem = stem[:-2]
        elif len(stem) > 4 and re.search(r"(?:s|x|z|ch|sh)es$", stem):
            stem = stem[:-2]
        elif len(stem) > 3 and stem.endswith("s") and not stem.endswith("ss"):
            stem = stem[:-1]
        # English drops the silent e before -ing/-ed. Match charge/charging,
        # release/released and move/moving without requiring copied wording.
        if len(stem) > 3 and stem.endswith("e"):
            stem = stem[:-1]
        stems.add(stem)
    return stems


def _h3_visible_chronology_text(value: Any) -> str:
    """History, namesakes and comparisons do not order two on-screen actions."""
    text = sanitize_h3_prompt_text(value)
    # Imported revision notes such as "(~1s earlier than before)" compare
    # versions; they do not order this action before the next source event.
    # Keep "before" when it actually introduces a following action clause.
    text = re.sub(r"\bthan\s+before\b(?=\s*(?:[).,;!?\]}]|$))",
                  "than previously", text, flags=re.I)
    text = re.sub(r"\bafter\s+(?:(?:many|several|\d+)\s+)?(?:years?|months?|decades?)\s+apart\b",
                  "", text, flags=re.I)
    return re.sub(
        r"\b(named|naming|names|modeled|modelled)(\s+(?:it|them|the\s+[\w-]+))?\s+after\b",
        r"\1\2 for", text, flags=re.I,
    )


def _h3_contract_clauses(value: Any) -> list[str]:
    """Split explicit event steps without treating every ``and`` as a cut."""

    text = _h3_visible_chronology_text(value)
    # A named appositive is one subject, not three separate physical steps:
    # "Two pilots, Iris and Omar, are checking ...". Keep the role with its
    # predicate; cast ownership is checked separately, and an isolated role
    # cannot be demonstrated by copying that noun into each action card.
    text = re.sub(
        r"^((?i:the|an?|two|three|both|several)\s+[^,;.!?]+),\s*"
        r"[A-Z][\w'’-]*(?:\s+(?:and\s+)?[A-Z][\w'’-]*)*,\s*"
        r"((?i:is|are|was|were)\b.*)$", r"\1 \2", text,
    )
    clauses = []
    for part in re.split(
        r"\s*[,;]\s*|\b(?:only\s+(?:then|after)|then|before|after|until)\b",
        text,
        flags=re.IGNORECASE,
    ):
        # Keep descriptive lead-ins with their concrete actions so a faithful
        # paraphrase need not repeat every adjective. Separate optical settings:
        # carrying "film grain" must not count as carrying an adjacent action.
        action_sentences = []
        for sentence in re.split(r"(?<=[.!?])\s+", part):
            if authored_optical_settings(sentence) or is_standalone_sound_cue(sentence):
                if action_sentences:
                    clauses.append(" ".join(action_sentences))
                    action_sentences = []
                clauses.append(sentence)
            else:
                action_sentences.append(sentence)
        if action_sentences:
            clauses.append(" ".join(action_sentences))
    clauses = [part.strip(" ,;:-.!?") for part in clauses]
    speech_only = re.compile(
        r"(?:(?:who|[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)\s+)?"
        r"(?:says?|replies|answers?|exclaims?|asks?|shouts?|whispers?|"
        r"declares?|announces?|calls\s+out|cries\s+out)",
        flags=re.IGNORECASE,
    )
    return [
        part for part in clauses
        if len(_h3_contract_token_stems(part)) >= 2
        and not speech_only.fullmatch(part)
    ]


def _camera_repair_feedback(
    violations: list[str], assigned_beats: list[dict[str, Any]],
) -> list[str]:
    """Translate internal beat IDs into the card keys the writer received."""
    feedback = []
    for error in violations:
        for index, beat in enumerate(assigned_beats, 1):
            prefix = str(beat.get("beat_id") or "").upper() + " "
            if error.startswith(prefix):
                error = f"event_cards.event_{index}: {error[len(prefix):]}"
                break
        feedback.append(error)
    return feedback


def _h3_required_relation_markers(value: Any) -> list[str]:
    text = _h3_visible_chronology_text(value).casefold()
    return [
        marker for marker in ("only after", "before", "until", "after")
        if re.search(r"\b" + re.escape(marker).replace(r"\ ", r"\s+") + r"\b", text)
    ]


def _h3_ordered_relation_pairs(value: Any) -> list[tuple[set[str], set[str], str]]:
    """Return earlier/later token sets and their explicit ordering marker."""

    text = _h3_visible_chronology_text(value)
    pairs: list[tuple[set[str], set[str], str]] = []
    for match in re.finditer(r"\b(only\s+after|before|after)\b", text, flags=re.I):
        marker = re.sub(r"\s+", " ", match.group(1).casefold())
        before = re.split(r"[.;]", text[:match.start()])[-1].strip(" ,:-")
        after = re.split(r"[.;]", text[match.end():])[0].strip(" ,:-")
        if not before or not after:
            continue
        left = _h3_contract_token_stems(before)
        right = _h3_contract_token_stems(after)
        if not left or not right:
            continue
        earlier, later = (left, right) if marker == "before" else (right, left)
        pairs.append((earlier, later, marker))
    return pairs


def _h3_relation_preserved(source: Any, action: Any, marker: str) -> bool:
    """Accept strict before/after paraphrases while rejecting simultaneity."""

    action_text = sanitize_h3_prompt_text(action)
    literal = r"\b" + re.escape(marker).replace(r"\ ", r"\s+") + r"\b"
    if marker == "until":
        return bool(re.search(literal, action_text, flags=re.IGNORECASE))
    source_pairs = _h3_ordered_relation_pairs(source)
    action_pairs = _h3_ordered_relation_pairs(action_text)
    for source_earlier, source_later, _source_marker in source_pairs:
        for action_earlier, action_later, _action_marker in action_pairs:
            earlier_need = 1 if min(len(source_earlier), len(action_earlier)) <= 3 else 2
            later_need = 1 if min(len(source_later), len(action_later)) <= 3 else 2
            if (
                len(source_earlier & action_earlier) >= earlier_need
                and len(source_later & action_later) >= later_need
            ):
                return True
    # Keep the legacy lexical check only for unusually terse fragments whose
    # two relation sides could not be resolved. When both sides are available,
    # a reversed relation must fail even if it repeats the same keyword.
    return bool(
        not source_pairs
        and re.search(literal, action_text, flags=re.IGNORECASE)
    )


def _h3_missing_relation_markers(source: Any, action: Any) -> list[str]:
    """Return each explicit source ordering whose semantic pair is absent."""

    source_pairs = _h3_ordered_relation_pairs(source)
    action_pairs = _h3_ordered_relation_pairs(action)
    missing: list[str] = []
    for source_earlier, source_later, marker in source_pairs:
        preserved = False
        for action_earlier, action_later, _action_marker in action_pairs:
            earlier_need = 1 if min(len(source_earlier), len(action_earlier)) <= 3 else 2
            later_need = 1 if min(len(source_later), len(action_later)) <= 3 else 2
            if (
                len(source_earlier & action_earlier) >= earlier_need
                and len(source_later & action_later) >= later_need
            ):
                preserved = True
                break
        if not preserved:
            missing.append(marker)
    if "until" in _h3_required_relation_markers(source) and not re.search(
        r"\buntil\b", sanitize_h3_prompt_text(action), flags=re.IGNORECASE,
    ):
        missing.append("until")
    return missing


def _apply_faithful_treatment(
    canonical: dict[str, Any],
    candidate: dict[str, Any] | None,
    *, resolve_adaptation: bool = False, start_frame: bool = False,
) -> dict[str, Any]:
    """Overlay safe LLM direction without giving it story ownership."""

    ledger = deepcopy(canonical)
    if not isinstance(candidate, dict):
        return ledger
    ledger["motion_mechanics"] = sanitize_h3_prompt_text(candidate.get("motion_mechanics"))
    cast_names = [
        sanitize_h3_prompt_text(name)
        for name in (
            (canonical.get("source_intent") or {}).get("cast_names") or []
        )
        if sanitize_h3_prompt_text(name)
    ]
    appearance = candidate.get("character_appearance")
    if isinstance(appearance, dict):
        ledger["character_appearance"] = " ".join(
            f"{name}: {sanitize_h3_prompt_text(appearance[name])}"
            for name in cast_names if sanitize_h3_prompt_text(appearance.get(name))
        )
    elif isinstance(appearance, str):
        # Older saved treatment responses used one shared description.
        ledger["character_appearance"] = sanitize_h3_prompt_text(appearance)
    for field in (
        "setting_continuity",
        "visual_continuity",
        "editing_style",
    ):
        resolving_source = resolve_adaptation or start_frame
        if resolving_source:
            # These canonical fields can contain the very source descriptors
            # being replaced. A rejected response must not resurrect them.
            ledger[field] = ""
        value = sanitize_h3_prompt_text(candidate.get(field))
        # A treatment field is a compact global direction, never a hidden
        # screenplay or per-character shot list. Reject malformed spillover
        # wholesale instead of truncating it into a misleading instruction.
        if (
            value
            # A coherent treatment can need a short paragraph. The old
            # 80-word ceiling rejected valid adapted style and silently
            # restored incompatible cast/duration notes from the source.
            and len(value.split()) <= 180
            and not re.search(r"(?:---|#{2,}|\*\*|\bsequence\s+progression\b)", value, re.IGNORECASE)
        ):
            ledger[field] = value
        elif resolving_source:
            ledger.setdefault("_treatment_review_fields", []).append(field)
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
    for field, allowed in (
        ("source_adaptation", resolve_adaptation),
        ("initial_state", start_frame),
    ):
        value = sanitize_h3_prompt_text(candidate.get(field))
        if allowed and value and len(value.split()) <= 180:
            ledger[field] = value
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
    preserve_adaptation: bool = False,
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
        if len(source_ids) <= 1 or (preserve_adaptation and not dialogue_ids):
            # A developed silent beat already carries its action, cause and
            # resulting state. Replacing it with individual source clauses
            # erases that adaptation and can replay an impact in the next
            # window. Spoken phases still split for exact speaker ownership.
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
                **({"_action_seconds": beat["_action_seconds"] / len(source_ids)}
                   if "_action_seconds" in beat else {}),
            })
            expanded.append(phase)

        # Generated connective lines may not have an E-id anchor. Keep them
        # beside their neighboring turns in authored order: appending them
        # after an already-anchored reaction can put the answer after its reply.
        # A leading connective joins the next anchored phase, so it cannot
        # accidentally speak before an earlier source entrance has happened.
        unanchored = [
            dialogue_id
            for dialogue_id in dialogue_ids
            if dialogue_id not in claimed_dialogue
        ]
        if expanded and unanchored:
            anchored_phases = {
                did: index for index, phase in enumerate(expanded)
                for did in phase["dialogue_ids"]
            }
            for phase in expanded:
                phase["dialogue_ids"] = []
            previous_phase = None
            for position, did in enumerate(dialogue_ids):
                phase_index = anchored_phases.get(did)
                if phase_index is None:
                    phase_index = previous_phase if previous_phase is not None else next(
                        (anchored_phases[later] for later in dialogue_ids[position + 1:]
                         if later in anchored_phases), len(expanded) - 1,
                    )
                expanded[phase_index]["dialogue_ids"].append(did)
                previous_phase = phase_index
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
        # Merge the smallest adjacent action groups first. Always taking the
        # first pair folds a long prompt into one huge opening shot followed
        # by tiny tails, which the filmable clock then rejects itself.
        merge_index = min(
            (
                index
                for index in range(len(result) - 1)
                if not (result[index].get("dialogue_ids") or [])
                and not (result[index + 1].get("dialogue_ids") or [])
            ),
            key=lambda index: sum(
                max(1, len(item.get("source_event_ids") or []))
                for item in result[index:index + 2]
            ),
            default=None,
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
            "authored_duration": sum(float(item.get("authored_duration") or 0) for item in (left, right)),
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
        # Let the final fragment use the remaining local budget so a long
        # exact turn can break at a sentence boundary. Both local and full
        # sequence admission use the same maximum speech rate.
        final_selected_segment = selected[-1][0]
        final_maximum_capacity = max(
            0,
            budgets[final_selected_segment - 1]
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
    for segment, maximum_budget in enumerate(budgets, start=1):
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
    # Beat count is a writing preference, not elapsed time: four brief turns
    # can fit where one long speech cannot. The constrained schema bounds the
    # draft size; the final physical/speech clock checks actual feasibility.
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
    source_events = extract_source_events(prompt)
    referenced_event_ids = [
        str(event_id or "").upper()
        for beat in beats
        for event_id in (beat.get("source_event_ids") or [])
    ]
    if Counter(referenced_event_ids) != Counter(source_event_ids):
        violations.append("source event IDs are missing, foreign, or repeated")
        counts = Counter(referenced_event_ids)
        missing = [event_id for event_id in source_event_ids if not counts[event_id]]
        repeated = [event_id for event_id, count in counts.items() if count > 1]
        foreign = [event_id for event_id in counts if event_id not in source_event_ids]
        violations.append(
            f"Event assignment correction: missing={missing}; repeated={repeated}; foreign={foreign}. "
            "Assign each source ID once; supporting reactions use empty source_event_ids. "
            "Include the final physical action, not only its state in required_final_outcome."
        )
    if referenced_event_ids and referenced_event_ids != source_event_ids:
        violations.append("source event order differs from the user's story order")
    if len(source_event_ids) > 1:
        expected_dialogue_events = _expected_dialogue_events(prompt, locked_dialogue)
        final_source_segment = _final_source_event_segment(
            source_events,
            locked_dialogue,
            expected_dialogue_events,
            segment_durations,
        ) if segment_durations else segment_count
        final_event_segments = [
            int(beat.get("segment") or 0)
            for beat in beats
            if source_event_ids[-1] in [
                str(event_id or "").upper()
                for event_id in (beat.get("source_event_ids") or [])
            ]
        ]
        if final_event_segments != [final_source_segment]:
            violations.append(
                "the final source event is not assigned to its last usable segment "
                f"({final_source_segment})"
            )

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
            max(1, int(math.floor(max(0.0, float(duration)) * _H3_DIALOGUE_MAX_WORDS_PER_SECOND)))
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

    violations.extend(_spectacle_violations(str(prompt or ""), ledger))
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
    reference_context = canonicalize_h3_reference_names(reference_context, intent["cast_names"])
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
    final_source_segment = _final_source_event_segment(
        source_events,
        locked_dialogue,
        expected_dialogue_events,
        segment_durations,
    ) if segment_durations else segment_count
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
            authored_cost = float(event.get("source_end_seconds", 0)) - float(event.get("source_start_seconds", 0))
            event_costs.append(
                authored_cost if authored_cost > 0 else
                1.15 + spoken_words / _H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND
            )
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
                current_segment = final_source_segment - 1
            event_buckets[current_segment].append(event)
            elapsed_cost += cost
    else:
        for index, event in enumerate(source_events):
            # A single compound outcome belongs at the end; earlier segments
            # can then build toward it. With multiple events, anchor the first
            # and last to the timeline ends.
            target = (
                final_source_segment - 1
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
    elif not any("source_start_seconds" in event for event in source_events):
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
            dialogue_budget = max(1, int(math.floor(durations[bucket_index] * _H3_DIALOGUE_MAX_WORDS_PER_SECOND)))
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
        if all("source_start_seconds" in event for event in events):
            return [[event] for event in events]
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
                "authored_duration": sum(
                    float(item.get("source_end_seconds", 0)) - float(item.get("source_start_seconds", 0))
                    for item in group
                ),
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
    for profile in intent.get("cast_profiles") or []:
        if profile.get("details"):
            continuity_parts.append(f"{profile['name']}: {profile['details']}")
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
    if source_events and "source_start_seconds" in source_events[0]:
        initial_state = re.split(r"(?<=[.!?])\s+", source_events[0]["text"], maxsplit=1)[0]
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
        "motion_mechanics": "",
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
    "motion_mechanics",
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
    preserve_adaptation: bool = False,
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
            if not text or _is_generated_stage_direction(text) or segment < 1 or segment > segment_count:
                continue
            ledger["generated_dialogue"].append({
                "dialogue_id": f"D{len(locked_dialogue) + len(ledger['generated_dialogue']) + 1}",
                "speaker": _resolve_h3_cast_name(
                    item.get("speaker"),
                    canonical_cast_names,
                ),
                "language": sanitize_h3_prompt_text(item.get("language")) or "English",
                "delivery": sanitize_h3_prompt_text(item.get("delivery")) or "speaks naturally and clearly",
                "text": text,
                "segment": segment,
                "source_event_id": str(item.get("source_event_id") or "").upper(),
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
        if (not source_ids and ledger['beats']
                and not locked_dialogue_id_set.intersection(item.get('dialogue_ids') or [])):
            # The array is chronological. A supporting reaction sometimes
            # repeats the preceding window number after the story has already
            # advanced. Correct that label without rescheduling any source
            # event or exact quotation, instead of rejecting the whole draft.
            segment = max(segment, ledger['beats'][-1]['segment'])
        continuation_ids = []
        if preserve_adaptation and index:
            # A broad concept may inspire every beat. Its ownership is recorded
            # once; later concrete progression is not another occurrence of
            # the user's whole brief. An ongoing activity can likewise span
            # windows before a separately specified ending. Keep concrete
            # one-off actions and dialogue events subject to duplicate checks.
            previous_ids = {value for beat in ledger["beats"] for value in beat["source_event_ids"]}
            previous_tail = next((value for beat in reversed(ledger['beats'])
                                  for value in reversed(beat['source_event_ids'])), None)
            def continuing_activity(event_id):
                text = _h3_visible_chronology_text(source_map.get(event_id, ''))
                return bool(
                    source_events and event_id == previous_tail
                    and event_id != source_events[-1]['event_id']
                    and not locked_dialogue_by_event.get(event_id)
                    and (re.search(
                        r'\b(?:is|are)\s+(?:(?:engaged|locked)\s+)?in\s+(?:an?|the)\s+'
                        r'(?:\w+[ -]+){0,3}(?:fight|battle|duel|race|chase|conversation|debate|dance)\b',
                        text, re.I) or _PLANNER_CONVERSATION_RE.search(text))
                    and not re.search(r'\b(?:then|after|before|until|wins?|loses?|lands?|breaks?|smashes?|ends?)\b', text, re.I)
                    # "how he finally opened it" is a conversation topic,
                    # whereas a leading "Finally" marks a concluding action.
                    and not re.search(r'(?:^|[.;])\s*finally\b', text, re.I)
                )
            continuation_ids = list(dict.fromkeys(
                value for value in [*source_ids, *(item.get("_continuation_event_ids") or [])]
                if value in previous_ids and continuing_activity(value)
            ))
            source_ids = [value for value in source_ids if value not in previous_ids
                          or not (len(source_events) == 1 or continuing_activity(value))]
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
        if preserve_adaptation and sanitize_h3_prompt_text(item.get("description")):
            description = sanitize_h3_prompt_text(item["description"])
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
            **({"_continuation_event_ids": continuation_ids} if continuation_ids else {}),
        })

    # The source order is authoritative; window labels are scheduling hints.
    # If every event is intact but those labels run backwards, repair only
    # the labels using the existing duration-aware schedule. Retain the AI's
    # choreography, scene state and complete spoken script.
    beat_segments = [beat['segment'] for beat in ledger['beats']]
    if (beat_segments != sorted(beat_segments)
            and [eid for beat in ledger['beats'] for eid in beat['source_event_ids']]
            == [event['event_id'] for event in source_events]):
        event_segments = {eid: beat['segment'] for beat in canonical['beats']
                          for eid in beat['source_event_ids']}
        previous_segment = 1
        for beat in ledger['beats']:
            beat['segment'] = max(previous_segment, max(
                (event_segments[eid] for eid in beat['source_event_ids']), default=previous_segment))
            previous_segment = beat['segment']
        for item in ledger['generated_dialogue']:
            item['segment'] = event_segments.get(item.get('source_event_id'), item['segment'])
        ledger['_timing_labels_repaired'] = True

    # New dialogue carries its source action, so greetings stay with the
    # greeting and later arguments stay with the argument. Older drafts with
    # only a segment retain their previous placement for compatibility.
    previous_owner = None
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
            owner = next((
                beat for beat in segment_beats
                if item.get("source_event_id") in [
                    *(beat.get("source_event_ids") or []), *(beat.get("_continuation_event_ids") or []),
                ]
            ), segment_beats[-1])
            if previous_owner is not None and ledger['beats'].index(owner) < ledger['beats'].index(previous_owner):
                # A follow-up question can cite the earlier question's E-id
                # even though the authored answer has already moved the
                # conversation forward. Keep that AI turn in the ongoing
                # discussion; never move exact user speech or one-off actions.
                ongoing_event = next((eid for eid in previous_owner['source_event_ids']
                    if _PLANNER_CONVERSATION_RE.search(source_map.get(eid, '')) or re.search(
                        r'\b(?:discuss(?:es|ed|ing)?|talk(?:s|ed|ing)?)\b', source_map.get(eid, ''), re.I)), None)
                if ongoing_event:
                    owner = previous_owner
                    item['segment'] = owner['segment']
                    item['source_event_id'] = ongoing_event
            owner["dialogue_ids"].append(item["dialogue_id"])
            previous_owner = owner
    ledger["ambient_audio"] = sanitize_h3_nonverbal_audio(
        ledger.get("ambient_audio")
    )
    _anchor_immediate_opening_dialogue(
        prompt,
        ledger,
        locked_dialogue=locked_dialogue,
        segment_count=segment_count,
    )
    from promptbench.experiments import action_first_enabled
    if action_first_enabled():
        from promptbench.story_time import retain_reservations
        retain_reservations(ledger, candidate)
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

    # A writer that supplies event anchors has already placed the exchange
    # beside its action. Rebalancing those lines by their list index can move a
    # greeting to a later argument or make somebody speak before they arrive.
    # Retain the legacy distribution only for drafts without event placement.
    if any(
        item.get("source_event_id") in (beat.get("source_event_ids") or [])
        and item["dialogue_id"] in (beat.get("dialogue_ids") or [])
        for item in generated for beat in beats
    ):
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
            or _is_generated_stage_direction(text)
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
        "motion_mechanics",
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


def _segment_shot_limit(assigned_beats: list[dict[str, Any]], *, event_cards: bool = False) -> int:
    """Allow extra cuts only when distinct dialogue phases require them."""

    if event_cards:
        # Each silent event has up to four phases. A speaking event has one
        # performance per line, optional physical lead-ins, and a follow-through.
        # These are schema-bounded slots, not model-selected dialogue ownership.
        return max(1, sum(
            2 * len(beat.get("dialogue_ids") or []) + 1
            if beat.get("dialogue_ids") else 4
            for beat in assigned_beats
        ))
    turns = sum(len(beat.get("dialogue_ids") or []) for beat in assigned_beats)
    return min(8, max(4, len(assigned_beats) + turns))


def _required_speech_action_cards(beat: dict[str, Any]) -> tuple[set[str], bool]:
    ids = [str(did).upper() for did in (beat.get('dialogue_ids') or [])]
    order = beat.get('_speech_action_order') or {}
    lead_ins = set()
    for index, did in enumerate(ids):
        if (order.get(did, {}).get('before_speech')
                or (index and order.get(ids[index - 1], {}).get('after_speech'))):
            lead_ins.add(did)
    return lead_ins, bool(ids and order.get(ids[-1], {}).get('after_speech'))


def _camera_event_card_schema(segment_number: int, assigned_beats: list[dict[str, Any]]) -> dict[str, Any]:
    """Require coverage by construction while leaving visual writing to the LLM."""

    def object_schema(properties):
        return {
            "type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False,
        }

    # Structured writers emit properties in this order. Resolve the movement
    # and its consequence before committing to coverage: choosing a cut first
    # encouraged splitting a strike from contact/recoil into separate shots.
    card = object_schema({name: {"type": "string"} for name in (
        "action", "framing", "camera", "transition", "sound_effects",
    )})
    # Reactions and gaze belong inside a speaking performance. Optional
    # setups/reactions otherwise triple its phases and crowd out the words.
    # Source-ordered actions (e.g. entering before speaking) still get cards.
    optional_card = {"type": "null"}
    events = {}
    has_speech = any(beat.get("dialogue_ids") for beat in assigned_beats)
    for index, beat in enumerate(assigned_beats, start=1):
        dialogue_ids = [str(did).upper() for did in (beat.get("dialogue_ids") or [])]
        if dialogue_ids:
            required_lead_ins, required_follow_through = _required_speech_action_cards(beat)
            properties = {
                did: object_schema({"lead_in": card if did in required_lead_ins else optional_card, "performance": card})
                for did in dialogue_ids
            }
            properties["follow_through"] = card if required_follow_through else optional_card
        else:
            grounded_opening = bool(index == 1 and beat.get("_start_frame_continuation"))
            if grounded_opening:
                opening_card = deepcopy(card)
                # Make the bridge from the photographed pose an explicit
                # writing step, still within the same timed action card.
                opening_card["properties"] = {
                    "recovery": {"type": "string"}, **opening_card["properties"],
                }
                opening_card["required"] = list(opening_card["properties"])
                opening_card["properties"]["transition"]["const"] = "continue supplied frame"
                opening_card["properties"]["framing"]["const"] = "The supplied frame's exact opening composition"
            properties = {"opening": opening_card} if grounded_opening else {}
            properties["phases"] = {
                "type": "array", "items": card,
                "minItems": 0 if grounded_opening else 1,
                # In a speaking window, one continuous performance per silent
                # event leaves time for the words. Splitting a smile or an
                # invitation into several reaction cuts starves its dialogue.
                # Silent action sequences retain their richer phase coverage.
                "maxItems": 1 if has_speech else (3 if grounded_opening else 4),
            }
        events[f"event_{index}"] = object_schema(properties)
    return object_schema({
        "segment": {"type": "integer", "minimum": segment_number, "maximum": segment_number},
        "title": {"type": "string"}, "coverage": {"type": "string"},
        "pacing": {"type": "string"}, "event_cards": object_schema(events),
        "closing_state": {"type": "string"},
    })


def _expand_camera_event_cards(
    segment: dict[str, Any], *, assigned_beats: list[dict[str, Any]],
    segment_number: int, duration: float,
) -> dict[str, Any]:
    """Compile required event/line cards in source order, including remote output.

    A camera model used to omit the final line or event while filling a freely
    sized shots array; a repair often repeated the identical omission. Fixed
    required keys let the grammar protect coverage without asking the model to
    copy and count identifiers. Do not invent a card if a provider ignores the
    schema: incomplete writing still needs repair/review.
    """

    events = segment.get("event_cards")
    expected_keys = [f"event_{i}" for i in range(1, len(assigned_beats) + 1)]
    if not isinstance(events, dict) or set(events) != set(expected_keys):
        raise ValueError(f"Write every required event card: {', '.join(expected_keys)}.")
    shots = []
    fields = ("transition", "framing", "camera", "action", "sound_effects")

    def add(card, event_index, dialogue_ids, *, optional=False, opening=False):
        if card is None and optional:
            return
        expected_fields = (*fields, "recovery") if opening else fields
        if not isinstance(card, dict) or set(card) != set(expected_fields) or any(
            not isinstance(card.get(name), str) for name in expected_fields
        ) or not card["action"].strip():
            raise ValueError(f"Event {event_index} needs a complete camera/action card.")
        card = {name: card[name] for name in expected_fields}
        card["action"] = strip_h3_source_clock_cues(card["action"])
        if opening:
            recovery = card.pop("recovery").strip().rstrip(".")
            if recovery:
                card["action"] = f"{recovery}. {card['action']}"
        shots.append({**card, "event_indices": [event_index], "dialogue_ids": dialogue_ids})

    for index, beat in enumerate(assigned_beats, start=1):
        event = events[f"event_{index}"]
        dialogue_ids = [str(did).upper() for did in (beat.get("dialogue_ids") or [])]
        grounded_opening = bool(index == 1 and beat.get("_start_frame_continuation") and not dialogue_ids)
        keys = [*dialogue_ids, "follow_through"] if dialogue_ids else (
            ["opening", "phases"] if grounded_opening else ["phases"]
        )
        if not isinstance(event, dict) or set(event) != set(keys):
            raise ValueError(f"Event {index} needs exactly these cards: {', '.join(keys)}.")
        if dialogue_ids:
            required_lead_ins, required_follow_through = _required_speech_action_cards(beat)
            for did in dialogue_ids:
                turn = event[did]
                if not isinstance(turn, dict) or set(turn) != {"lead_in", "performance"}:
                    raise ValueError(f"Write lead_in and performance for {did} in event {index}.")
                add(turn["lead_in"], index, [], optional=did not in required_lead_ins)
                add(turn["performance"], index, [did])
            add(event["follow_through"], index, [], optional=not required_follow_through)
        else:
            if grounded_opening:
                add(event["opening"], index, [], opening=True)
            phases = event["phases"]
            if not isinstance(phases, list) or not (
                (0 if grounded_opening else 1) <= len(phases) <= (3 if grounded_opening else 4)
            ):
                raise ValueError(f"Event {index} needs one to four advancing action phases.")
            for card in phases:
                add(card, index, [])
    if not shots:
        raise ValueError("Write at least one camera/action card.")
    if any(beat.get("_action_seconds") for beat in assigned_beats):
        from promptbench.story_time import reserve_camera_actions
        reserve_camera_actions(shots, assigned_beats)
    for index, shot in enumerate(shots):
        shot.update({
            "shot": index + 1,
            "start_seconds": duration * index / len(shots),
            "end_seconds": duration * (index + 1) / len(shots),
        })
    result = {key: value for key, value in segment.items() if key != "event_cards"}
    return {**result, "segment": segment_number, "shots": shots, "camera_contract": "event_cards"}


def _segment_schema(
    segment_number: int,
    *,
    maximum_shots: int = 4,
    event_count: int | None = None,
    minimum_shots: int = 1,
    dialogue_ids: list[str] | None = None,
    assigned_beats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if assigned_beats:
        return _camera_event_card_schema(segment_number, assigned_beats)
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
    if event_count is not None:
        shot["properties"]["event_indices"] = {
            "type": "array", "minItems": 1,
            "items": {"type": "integer", "minimum": 1, "maximum": max(1, event_count)},
        }
        shot["required"].append("event_indices")
    if dialogue_ids:
        shot["properties"]["dialogue_ids"] = {
            "type": "array", "items": {"type": "string", "enum": dialogue_ids},
            **({"maxItems": 1} if len(dialogue_ids) <= maximum_shots else {}),
        }
        shot["required"].append("dialogue_ids")
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
                "minItems": max(1, min(int(minimum_shots), int(maximum_shots))),
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


def _strip_planner_speech_cues(
    value: Any, *, sound_field: bool = False, assigned_speakers: tuple[str, ...] = (),
) -> str:
    """Remove model-authored speech hints from application-owned dialogue.

    Exact spoken performance is attached later from the locked dialogue
    catalog.  Leaving phrases such as ``Yoda speaks`` in another shot (or
    ``Yoda's voice`` in sound effects) creates a second, untagged vocal event
    that H3 may fill with repeated or gibberish speech.
    """

    text = sanitize_h3_prompt_text(value)
    if not text:
        return ""
    # Keep a conversational cue atomic before comma-based clause splitting.
    # Its subject matter belongs in dialogue, not in visible choreography.
    text = _PLANNER_CONVERSATION_RE.sub("talk", text)
    text = _PLANNER_ADVICE_RE.sub("advises", text)
    if sound_field:
        # Resolve whole sentence/semicolon negative blocks before comma
        # splitting can detach a shared "No" from later list items.
        blocks = re.split(r"\s*;\s*|(?<=[.!?])\s+", text)
        retained_blocks: list[str] = []
        for block in blocks:
            if not re.match(r"^No\b", block.strip(), flags=re.IGNORECASE):
                retained_blocks.append(block)
                continue
            contrast = re.search(
                r"(?:,\s*|\s+)(?:but|only|just|instead|while)\s+(.+)$",
                block.strip().rstrip(".!?"),
                flags=re.IGNORECASE,
            )
            tail = contrast.group(1).strip() if contrast else ""
            if tail and not re.match(r"^no\b", tail, flags=re.IGNORECASE):
                retained_blocks.append(tail)
        text = "; ".join(retained_blocks)
    text = re.sub(r"<d>.*?</d>", ". ", text, flags=re.IGNORECASE | re.DOTALL)
    # Keep a quoted utterance atomic before splitting sentences. Removing
    # only `Mara says, "Welcome.` leaves the rest of her multi-sentence line
    # in visual action, where H3 can speak it a second time without a tag.
    # Quoted visual labels and signs are not transcripts.
    quotes = _DIALOGUE_QUOTE_RE
    visible_text = re.compile(
        r"\b(?:sign|label|poster|screen|caption|banner)\b.{0,35}\b(?:says?|reads?)\b",
        flags=re.IGNORECASE,
    )
    def remove_spoken_quote(match: re.Match[str]) -> str:
        lead = re.split(r"[.!?;\n]", text[:match.start()])[-1]
        if _find_spoken_verb(lead) and not visible_text.search(lead):
            return "."  # Keep the boundary before the next physical action.
        return match.group(0)
    text = quotes.sub(remove_spoken_quote, text)
    speech_audio = re.compile(
        r"\b(?:voice|voices|dialogue|speech|spoken|vocal\s+line|"
        r"vocali[sz]ations?|"
        r"chatter|murmur(?:ing)?|babble|conversation|"
        r"announc(?:e(?:s|d|ments?)?|ing|ers?)|public[ -]address|"
        r"p\.?\s*a\.?\s+(?:system|speaker|announcements?)|paging|"
        r"people\s+(?:talking|speaking)|crowd\s+(?:talking|speaking))\b",
        flags=re.IGNORECASE,
    )
    overlapping_speech = re.compile(
        r"\b(?:voices?|words|dialogue|speech|vocali[sz]ations?)\b[^.;]{0,80}\boverlap(?:ping|s|ped)?\b|"
        r"\boverlapping\s+(?:voices?|words|dialogue|speech|vocali[sz]ations?)\b|"
        r"\btalk(?:ing|s|ed)?\s+over\b",
        flags=re.IGNORECASE,
    )
    clause_pattern = r"\s*;\s*|(?<=[.!?])\s+"
    if sound_field:
        # Soundscapes are commonly returned as comma-separated lists. Split
        # those items so one invalid "background chatter" phrase does not
        # discard valid music, cups, wind, or machinery in the same sentence.
        clause_pattern += r"|\s*,\s*(?:and\s+)?"
    clauses = re.split(clause_pattern, text)
    kept: list[str] = []
    acting_subject = ""

    def acting_name(part: str) -> str:
        part = re.sub(r"^(?:(?:then|as|while|before|after|when|once|during)\s+)+", "", part, flags=re.I)
        subject = re.match(_PROPER_NAME.pattern, part)
        name = subject.group(0) if subject else ""
        if name in assigned_speakers:
            return name
        if name.lower() in _CONTENT_STOPWORDS or _CAST_ACTION_RE.fullmatch(name.lower()) or _find_spoken_verb(name):
            return ""
        return name

    def keep_acting(part: str) -> None:
        nonlocal acting_subject
        # A removed speech cue may own the following dependent movement:
        # 'Mara says ... while pointing' must not lose Mara with the words.
        name = acting_name(part)
        if name:
            acting_subject = name
        elif acting_subject and re.match(r"^[a-z]+ing\b", part, re.I):
            part = f"{acting_subject} is {part}"
        elif acting_subject and (
            re.match(r"^then\s+", part, flags=re.IGNORECASE)
            or _CAST_ACTION_RE.match(part[:1].lower() + part[1:])
        ):
            part = f"{acting_subject} {part[:1].lower() + part[1:]}"
        kept.append(part)

    for clause in clauses:
        clause = clause.strip()
        if not clause.strip(" ;."):
            continue
        if sound_field:
            if not _find_spoken_verb(clause) and not speech_audio.search(clause) and not overlapping_speech.search(clause):
                kept.append(clause.strip(" ;."))
            continue
        if visible_text.search(clause) or not (_find_spoken_verb(clause) or overlapping_speech.search(clause)):
            keep_acting(clause)
            continue
        subject = acting_name(clause)
        if subject:
            acting_subject = subject
        # Preparing to speak is silent acting, not an extra utterance. Keep
        # its physical clause instead of discarding the entire sentence.
        clause = re.sub(
            r"\b(prepares?|preparing|prepared|ready)\s+to\s+(?:speak|talk|reply|respond)\b",
            r"\1 for the exchange", clause, flags=re.IGNORECASE,
        )
        clause = re.sub(
            r"\bbefore\s+(?:(?:he|she|they)\s+)?(?:speaks?|speaking|talking)\b",
            "before the exchange", clause, flags=re.IGNORECASE,
        )
        clause = re.sub(
            r"\bhaving\s+(?:just\s+)?finished\s+(?:speaking|talking)\b",
            "with mouth now closed", clause, flags=re.IGNORECASE,
        )
        clause = re.sub(
            r"\bafter\s+(?:speaking|talking)\b",
            "after the assigned line", clause, flags=re.IGNORECASE,
        )
        if not (_find_spoken_verb(clause) or overlapping_speech.search(clause)):
            keep_acting(clause)
            continue
        # A mixed acting sentence can have a vocal clause between two useful
        # physical clauses. Remove just the vocal clause, including any
        # unquoted transcript, and keep gestures/gaze/movement on either side.
        parts = re.split(
            r"\s*,\s*|\s+(?:and|while|as)\s+|\s+(?=speaking\b)", clause, flags=re.IGNORECASE,
        )
        owner = next((name for name in assigned_speakers if re.match(
            rf"^(?:the\s+)?{re.escape(name)}(?![\w])", clause, flags=re.IGNORECASE,
        )), "")
        in_speech = False
        for part in parts:
            part = part.strip(" ;.")
            if not part:
                continue
            if _find_spoken_verb(part) or overlapping_speech.search(part):
                in_speech = True
                if owner and re.match(rf"^(?:the\s+)?{re.escape(owner)}(?![\w])", part, re.I):
                    # The catalog, not this prose, owns the actual words and
                    # speaker. This is only the already-assigned performance.
                    kept.append(f"{owner} performs the assigned line")
                continue
            if in_speech and re.match(r"^(?:why|how|what|whether|that)\b", part, re.I):
                continue  # A coordinated topic still belongs to the speech.
            in_speech = False
            if not speech_audio.search(part):
                keep_acting(part)
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
            rf"\bthen\s+(?:cuts?\s+)?(?:to|onto)\s+{escaped}(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"\b(?:cut(?:s|ting)?|switch(?:es|ing)?)\s+between\b[^.;:]{{0,100}}{escaped}(?![\w])",
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
            rf"rack\s+focus\s+(?:to|onto)|push(?:es|ed|ing)?\s+in\s+(?:(?:slowly|quickly|slightly|gently)\s+)?on)\s+"
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
        # A duplicated camera instruction in action can otherwise override
        # the actual speaker's camera field. Preserve all physical clauses.
        clauses = re.split(r"(?<=[.!?])\s+|;\s*", action)
        kept = [clause for clause in clauses if not (
            re.match(r"(?:the\s+)?camera\b", clause, flags=re.IGNORECASE)
            and any(_speaker_is_camera_focus(clause, other, known_speakers) for other in inactive_speakers)
        )]
        if len(kept) != len(clauses):
            action = "; ".join(kept)
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
                "a medium shot in the established target scene frames "
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


def _has_ordered_shot_coverage(groups: list[list[Any]], expected: list[Any]) -> bool:
    """Allow a beat to continue across adjacent shots, never to restart later."""

    covered: list[Any] = []
    for group in groups:
        if not isinstance(group, list) or not group:
            return False
        for index, value in enumerate(group):
            if value in group[:index]:
                return False
            if not covered or covered[-1] != value:
                covered.append(value)
    return covered == expected


def _canonicalize_segment_contract(
    segment: dict[str, Any] | None,
    *,
    segment_number: int,
    duration: float,
    assigned_beats: list[dict[str, Any]],
    dialogue_catalog: list[dict[str, Any]],
    opening_state: str,
    source_intent: dict[str, Any],
    source_events: list[dict[str, Any]] | None = None,
    use_camera_handoff: bool = False,
) -> dict[str, Any] | None:
    """Attach exact dialogue and timing to the LLM's paired camera/action.

    Local event indices tie each shot to its source material. Complete action
    prose stays with the camera that was written for it. Legacy saved plans
    without this mapping retain their deterministic compilation behavior.
    """

    if not isinstance(segment, dict):
        return None
    if "event_cards" in segment:
        try:
            segment = _expand_camera_event_cards(
                segment, assigned_beats=assigned_beats,
                segment_number=segment_number, duration=duration,
            )
        except ValueError as error:
            return {**segment, "event_assignment_error": str(error)}
    maximum_shots = _segment_shot_limit(
        assigned_beats, event_cards=segment.get("camera_contract") == "event_cards",
    )
    raw_shots = [
        dict(item) for item in (segment.get("shots") or [])
        if isinstance(item, dict)
    ]
    if not raw_shots or len(raw_shots) > maximum_shots:
        # Let validation request a complete, correctly sized plan. Trimming
        # a continuation shot here can discard the event's final action.
        return segment

    total = max(0.1, float(duration))
    shot_count = len(raw_shots)
    minimum_shot = _h3_minimum_shot_seconds(total, shot_count)
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
        shot["action"] = strip_h3_source_clock_cues(
            _strip_planner_speech_cues(
                shot.get("action"),
                assigned_speakers=tuple(
                    str(line.get("speaker") or "") for line in dialogue_catalog
                    if line.get("dialogue_id") in (shot.get("dialogue_ids") or [])
                    and not line.get("off_camera") and line.get("speaker")
                ),
            )
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

    beat_count = len(assigned_beats)
    semantic_actions = any("event_indices" in shot for shot in raw_shots)
    explicit_dialogue = semantic_actions and any("dialogue_ids" in shot for shot in raw_shots)
    assignments: list[list[dict[str, Any]]] = [[] for _ in raw_shots]
    if semantic_actions:
        groups = [shot.get("event_indices") for shot in raw_shots]
        indices = [
            value for group in groups
            for value in (group if isinstance(group, list) else [group])
        ]
        if (
            any(type(value) is not int for value in indices)
            or not _has_ordered_shot_coverage(groups, list(range(1, beat_count + 1)))
        ):
            # Keep the actual response available for the existing focused
            # repair. Raising here used to bypass repair and replace useful
            # model writing with the deterministic fallback immediately.
            return {**segment, "event_assignment_error": (
                f"Cover every numbered event 1 through {beat_count} in order. "
                "An event may continue across consecutive shots: [1], [1], [2] is valid. "
                "List it only once within a shot, and do not return to an earlier "
                f"event after advancing to the next. The current mapping is {groups}."
            )}
        if explicit_dialogue:
            expected = [
                str(did).upper() for beat in assigned_beats
                for did in (beat.get("dialogue_ids") or [])
            ]
            speech_groups = [shot.get("dialogue_ids") for shot in raw_shots]
            valid_groups = all(
                isinstance(group, list) and all(isinstance(did, str) for did in group)
                for group in speech_groups
            )
            normalized = [[did.upper() for did in group] for group in speech_groups] if valid_groups else []
            allowed = [
                {str(did).upper() for number in group for did in (assigned_beats[number - 1].get("dialogue_ids") or [])}
                for group in groups
            ]
            if (
                not valid_groups or [did for group in normalized for did in group] != expected
                or any(not set(group) <= owners for group, owners in zip(normalized, allowed))
            ):
                return {**segment, "dialogue_assignment_error": (
                    f"Assign dialogue_ids {expected} exactly once in that order, beside their event. "
                    "Use [] for silent shots. A line may belong to a later shot continuing its event; "
                    f"the current placement is {speech_groups}."
                )}
            if len(expected) <= maximum_shots and any(len(group) > 1 for group in normalized):
                return {**segment, "dialogue_assignment_error": (
                    "Give each dialogue_id its own camera phase; use at most one line per shot. "
                    "Continue the same event in adjacent shots for its successive speaker turns. "
                    "This keeps each person's physical performance beside that person's line."
                )}
            for shot, group in zip(raw_shots, normalized):
                shot["dialogue_ids"] = group
        coverage_counts = Counter(indices)
        coverage_weights = Counter()
        shot_weights = [
            (shot["end_seconds"] - shot["start_seconds"]) / len(group)
            for shot, group in zip(raw_shots, groups)
        ]
        for group, weight in zip(groups, shot_weights):
            for number in group:
                coverage_weights[number] += weight
        seen_events: set[int] = set()
        for shot, bucket, group, weight in zip(raw_shots, assignments, groups, shot_weights):
            for number in group:
                beat = dict(assigned_beats[number - 1])
                # The camera writer owns local speech placement, so an
                # entrance can precede its greeting and a conversation can
                # follow the move into a room. Legacy plans keep first-shot
                # placement; neither path repeats a line on a continuation.
                if explicit_dialogue:
                    beat["dialogue_ids"] = [
                        did for did in (beat.get("dialogue_ids") or [])
                        if str(did).upper() in shot["dialogue_ids"]
                    ]
                elif number in seen_events:
                    beat["dialogue_ids"] = []
                seen_events.add(number)
                if coverage_counts[number] > 1 and beat.get("authored_duration"):
                    # Count a timed event's duration once, shared among its
                    # shots in their proposed proportions.
                    beat["authored_duration"] = (
                        float(beat["authored_duration"]) * weight / coverage_weights[number]
                    )
                bucket.append(beat)
    else:
        # Compatibility for saved plans and older callers. New camera
        # requests return explicit local ownership with their action prose.
        for index, beat in enumerate(assigned_beats):
            target = min(shot_count - 1, int(index * shot_count / max(1, beat_count)))
            assignments[target].append(beat)

    dialogue_map = {
        str(item.get("dialogue_id") or "").upper(): item
        for item in dialogue_catalog
    }
    source_timing = {event['event_id']: event['text'] for event in (source_events or [])}
    for shot, bucket in zip(raw_shots, assignments):
        event_ids = [eid for beat in bucket for eid in (beat.get('source_event_ids') or [])]
        if event_ids and all(eid in source_timing for eid in event_ids):
            # Minimum action time comes from the actual source event, not
            # decorative prose such as "her composure cracks" or "the box
            # sits motionless". The AI's local action still owns clock weights.
            shot['_timing_source_action'] = '. Then '.join(source_timing[eid] for eid in event_ids)
    timing_assignments = assignments
    if semantic_actions:
        # The writer's local action describes only this part of a continuing
        # event. Timing the entire coarse beat in every shot squeezes later
        # action and charges repeated speech time.
        timing_assignments = [[{
            "description": shot["action"],
            "source_event_ids": [beat.get("beat_id") for beat in bucket],
            "dialogue_ids": [
                dialogue_id for beat in bucket for dialogue_id in (beat.get("dialogue_ids") or [])
            ],
        }] for shot, bucket in zip(raw_shots, assignments)]
    _apply_h3_filmable_shot_clock(
        raw_shots,
        timing_assignments,
        dialogue_map,
        duration=total,
        only_when_squeezed=True,
    )
    authored_weights = [sum(float(beat.get("authored_duration") or 0) for beat in group) for group in assignments]
    if authored_weights and all(weight > 0 for weight in authored_weights):
        cursor = 0.0
        authored_durations = _h3_filmable_shot_durations(authored_weights, total)
        for index, (shot, shot_duration) in enumerate(zip(raw_shots, authored_durations)):
            shot["start_seconds"] = round(cursor, 3)
            cursor += shot_duration
            shot["end_seconds"] = round(total if index + 1 == len(raw_shots) else cursor, 3)
    if segment_number == 1 and source_intent.get("opening_dialogue_id"):
        _cap_h3_opening_dialogue_lead(
            raw_shots,
            assignments,
            duration=total,
            dialogue_id=str(source_intent.get("opening_dialogue_id")),
        )
    # Camera writing owns staging and movement. Literal user-authored optics
    # belong to the compiler, just like exact dialogue. Retain them in their
    # own event's camera field instead of repeatedly asking an LLM to copy
    # them and replacing a good window when the focused rewrite drops one.
    # Read only original source events, never AI-proposed beat descriptions.
    source_optics = {
        str(event.get("event_id") or "").upper(): authored_optical_settings(event.get("text"))
        for event in (source_events or [])
    }
    source_sounds = {
        str(event.get("event_id") or "").upper(): authored_sound_cues(event.get("text"))
        for event in (source_events or [])
    }
    retained_optics: set[str] = set()
    for shot, shot_beats in zip(raw_shots, assignments):
        beat_ids = [str(beat.get("beat_id") or "").upper() for beat in shot_beats]
        shot["beat_ids"] = beat_ids
        for beat in shot_beats:
            for event_id in (beat.get("source_event_ids") or []):
                event_id = str(event_id or "").upper()
                if event_id in retained_optics:
                    continue
                retained_optics.add(event_id)
                local_text = _normalize_key(" ".join(str(shot.get(field) or "")
                                                     for field in ("action", "framing", "camera")))
                cues = [cue for cue in source_optics.get(event_id, [])
                        if _normalize_key(cue) not in local_text]
                if cues:
                    shot["camera"] = ". ".join([shot["camera"].rstrip(" ."), *cues])
        required = ". Then ".join(
            sanitize_h3_prompt_text(beat.get("description"))
            for beat in shot_beats
            if sanitize_h3_prompt_text(beat.get("description"))
        )
        # Only legacy plans need action reconstructed. New plans explicitly
        # pair their source events with the writer's camera and choreography.
        if not semantic_actions:
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

    # Literal sound directions are compiler-owned, like optical settings.
    # Retain each cue once in its own event, never globally or in another beat.
    # Preserve an existing placement anywhere in that event's phases; otherwise
    # a trailing cue goes on its last phase and a leading cue on its first.
    for event in (source_events or []):
        event_id = str(event.get("event_id") or "").upper()
        cues = source_sounds.get(event_id, [])
        if not cues:
            continue
        local_shots = [shot for shot, bucket in zip(raw_shots, assignments)
                       if any(event_id in [str(value).upper() for value in beat.get("source_event_ids", [])]
                              for beat in bucket)]
        if not local_shots:
            continue
        local_text = _normalize_key(" ".join(str(shot.get(field) or "")
            for shot in local_shots for field in ("action", "sound_effects")))
        source_text = _normalize_key(event.get("text"))
        for cue in cues:
            if _normalize_key(cue) in local_text:
                continue
            target = local_shots[-1] if source_text.endswith(_normalize_key(cue)) else local_shots[0]
            target["sound_effects"] = "; ".join(
                part for part in (sanitize_h3_prompt_text(target.get("sound_effects")), cue) if part
            )

    result = dict(segment)
    result["semantic_actions"] = semantic_actions
    result.update({
        "segment": segment_number,
        "opening_state": sanitize_h3_prompt_text(opening_state),
        "shots": raw_shots,
        "closing_state": sanitize_h3_prompt_text(
            segment.get("closing_state")
            if use_camera_handoff and sanitize_h3_prompt_text(segment.get("closing_state")) else
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
    if segment.get("event_assignment_error"):
        return [str(segment["event_assignment_error"])]
    if segment.get("dialogue_assignment_error"):
        return [str(segment["dialogue_assignment_error"])]
    violations: list[str] = []
    try:
        returned_number = int(segment.get("segment"))
    except (TypeError, ValueError):
        returned_number = 0
    if returned_number != segment_number:
        violations.append(f"returned segment {returned_number} instead of {segment_number}")
    shots = [item for item in (segment.get("shots") or []) if isinstance(item, dict)]
    maximum_shots = _segment_shot_limit(
        assigned_beats, event_cards=segment.get("camera_contract") == "event_cards",
    )
    if not 1 <= len(shots) <= maximum_shots:
        violations.append(
            f"returned {len(shots)} shots instead of one to {maximum_shots}"
        )
        return violations

    assigned_beat_ids = [str(item.get("beat_id") or "").upper() for item in assigned_beats]
    source_event_map = {
        str(item.get("event_id") or "").upper(): sanitize_h3_prompt_text(item.get("text"))
        for item in extract_source_events(prompt)
    }
    used_beat_ids: list[str] = []
    beat_actions: dict[str, list[str]] = {}
    beat_camera: dict[str, list[str]] = {}
    beat_audio: dict[str, list[str]] = {}
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
            beat_camera.setdefault(beat_id, []).append(" ".join(
                str(shot.get(field) or "") for field in ("camera", "framing")
            ))
            beat_audio.setdefault(beat_id, []).append(str(shot.get("sound_effects") or ""))
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

    if segment.get("semantic_actions"):
        coverage = [
            [str(value or "").upper() for value in (shot.get("beat_ids") or [])]
            for shot in shots
        ]
        if not _has_ordered_shot_coverage(coverage, assigned_beat_ids):
            violations.append("assigned beat coverage is missing, foreign, or out of order")
    elif Counter(used_beat_ids) != Counter(assigned_beat_ids):
        violations.append("assigned beat IDs are missing, foreign, or repeated")
    for beat in assigned_beats:
        beat_id = str(beat.get("beat_id") or "").upper()
        action_text = " ".join(beat_actions.get(beat_id, []))
        # Imported event cards contain both action and cinematography. Their
        # local camera/framing fields are rendered beside the action; a pull
        # back or close-up need not be copied into the action string as well.
        # Do not credit other beats, future states or global notes. Audio is
        # evidence only for standalone sound cues, never for physical action.
        action_tokens = _h3_contract_token_stems(
            " ".join([action_text, *beat_camera.get(beat_id, [])])
        )
        beat_source_text = " ".join(
            source_event_map.get(str(event_id or "").upper(), "")
            for event_id in (beat.get("source_event_ids") or [])
        )
        source_locked = len(
            _h3_contract_token_stems(beat_source_text)
            & _h3_contract_token_stems(beat.get("description"))
        ) >= 2
        if segment.get("semantic_actions") and not source_locked:
            continue
        required_action = sanitize_h3_prompt_text(beat.get("description"))
        if segment.get("semantic_actions"):
            # Validate user-owned physical actions, not the earlier writer's
            # embellishments or spoken explanations removed from visual prose.
            source_requirement = (
                required_action if _normalize_key(required_action) in _normalize_key(prompt)
                else beat_source_text
            )
            required_action = (
                _strip_planner_speech_cues(source_requirement)
                if _find_spoken_verb(source_requirement) else source_requirement
            )
        required_parts = (
            _h3_contract_clauses(required_action)
            if (
                segment.get("semantic_actions")
                and beat.get("source_event_ids")
                and len(sanitize_h3_prompt_text(beat.get("description"))) <= 500
            ) else
            [required_action]
        )
        for part in required_parts:
            required_tokens = _h3_contract_token_stems(part)
            minimum_overlap = 1 if len(required_tokens) <= 4 else 2
            evidence_tokens = action_tokens
            if is_standalone_sound_cue(part):
                evidence_tokens = action_tokens | _h3_contract_token_stems(
                    " ".join(beat_audio.get(beat_id, []))
                )
            if re.match(r"^(?:outside|inside|in|at|near|beside)\b", part, re.I):
                evidence_tokens = action_tokens | _h3_contract_token_stems(segment.get("opening_state"))
            if "meet" in required_tokens and (
                action_tokens & {"greet", "arriv", "approach", "encounter"}
                or re.search(r"\b(?:towards?\s+(?:each\s+other|one\s+another)|"
                             r"clos(?:e|es|ing)\s+(?:(?:the|their|initial)\s+){0,2}(?:distance|gap))\b",
                             action_text, flags=re.I)
            ):
                # A visible arrival/greeting depicts a meeting; demanding the
                # word "meet" rejects normal character-specific staging.
                evidence_tokens = evidence_tokens | {"meet"}
            if required_tokens and len(required_tokens & evidence_tokens) < minimum_overlap:
                violations.append(
                    f"{beat_id} shot action omits required source step: {part}"
                )
        for marker in (
            _h3_missing_relation_markers(required_action, action_text)
            if source_locked else []
        ):
            violations.append(
                f"{beat_id} shot action drops the explicit '{marker}' chronology relation"
            )

    if segment.get("semantic_actions"):
        source_events = extract_source_events(prompt)
        source_positions = {
            str(item.get("event_id") or "").upper(): index
            for index, item in enumerate(source_events)
        }
        assigned_source_ids = [
            str(event_id or "").upper()
            for beat in assigned_beats
            for event_id in (beat.get("source_event_ids") or [])
            if str(event_id or "").upper() in source_positions
        ]
        if assigned_source_ids:
            last_position = max(source_positions[event_id] for event_id in assigned_source_ids)
            # A recurring two-word name is not evidence that its later action
            # already happened. Remove whole cast identifiers before comparing
            # predicates; do not globally discard their component words (a
            # character called Power Girl does not make all "power" irrelevant).
            cast_names = extract_h3_source_intent(prompt).get("cast_names") or []
            cast_pattern = (
                re.compile(
                    r"(?<![\w])(?:" + "|".join(
                        re.escape(name) for name in sorted(cast_names, key=len, reverse=True)
                    ) + r")(?![\w])",
                    flags=re.IGNORECASE,
                ) if cast_names else None
            )

            def event_action_tokens(value: Any) -> set[str]:
                text = sanitize_h3_prompt_text(value)
                return _h3_contract_token_stems(
                    cast_pattern.sub(" ", text) if cast_pattern else text
                ) - {"both"}  # A shared subject is not evidence of a later action.

            current_contract_tokens = event_action_tokens(" ".join(
                sanitize_h3_prompt_text(beat.get("description")) + " "
                + sanitize_h3_prompt_text(beat.get("state_after"))
                for beat in assigned_beats
            ))
            for future in source_events[last_position + 1:]:
                future_text = sanitize_h3_prompt_text(future.get("text"))
                future_actions = event_action_tokens(future_text)
                distinctive = future_actions - current_contract_tokens
                # Retain a complete short outcome such as "Maya Chen wins".
                # Its one predicate is meaningful; the name alone is not.
                short_named_outcome = bool(
                    len(future_actions) == 1 and cast_pattern
                    and cast_pattern.search(future_text)
                    and re.search(
                        cast_pattern.pattern + r"\s+(?!(?:is|are|was|were|looks?|feels?|seems?)\b)\w+",
                        future_text, flags=re.I,
                    )
                )
                minimum_overlap = 1 if short_named_outcome else 2
                if len(distinctive) < minimum_overlap:
                    continue
                for index, shot in enumerate(shots):
                    overlap = distinctive & event_action_tokens(shot.get("action"))
                    if (
                        len(overlap) >= minimum_overlap
                        and len(overlap) / len(distinctive) >= 0.35
                    ):
                        violations.append(
                            f"shot {index + 1} previews later source event "
                            f"{future.get('event_id')} through unassigned actions: "
                            + ", ".join(sorted(overlap)[:5])
                        )
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
        minimum_shot = _h3_minimum_shot_seconds(duration, len(shots))
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
                extract_locked_dialogue(prompt),
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
    # Titles/indices describe the plan in Maestro's UI, not the generated scene.
    scene_content = json.dumps({
        key: segment.get(key)
        for key in ("opening_state", "coverage", "pacing", "shots", "closing_state")
    }, ensure_ascii=False)
    if has_h3_window_bookkeeping(scene_content, source_prompt=lowered_source):
        violations.append("introduced generation-window bookkeeping into scene content")
    violations.extend(_spectacle_violations(str(prompt or ""), segment))
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

    from services.h3_action_clock import estimate_h3_action_seconds

    action_text = ". Then ".join(
        sanitize_h3_prompt_text(beat.get("description")) for beat in bucket
    )
    return estimate_h3_action_seconds(action_text, event_floor=event_floor)


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
    spoken_time = dialogue_words / _H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND + len(dialogue_ids) * 0.35
    action_time = _h3_visible_action_seconds(bucket, event_floor=event_count)
    return max(float(event_count), action_time, spoken_time, 1.0)


def _h3_minimum_shot_seconds(duration: float, shot_count: int) -> float:
    count = max(1, shot_count)
    return min(duration / count, 0.75, max(0.25, duration / max(2, count * 2)))


def _h3_filmable_shot_durations(
    weights: list[float], duration: float, *, minimums: list[float] | None = None,
) -> list[float]:
    """Fit relative action/speech weights without creating invalid tail shots."""

    if not weights:
        return []
    remaining = max(0.1, float(duration))
    minimum = _h3_minimum_shot_seconds(remaining, len(weights))
    floors = [max(minimum, value) for value in minimums] if minimums is not None else [minimum] * len(weights)
    if sum(floors) > remaining + 1e-6:
        # This draft needs shorter AI speech (or review for immutable quotes).
        # Keep a usable timeline for that edit, never drop a shot or line.
        floors = [minimum] * len(weights)
    weights = [max(0.1, weight) for weight in weights]
    durations = [0.0] * len(weights)
    pending = list(range(len(weights)))
    while pending:
        total_weight = sum(weights[index] for index in pending)
        short = [
            index for index in pending
            if remaining * weights[index] / total_weight < floors[index]
        ]
        if not short:
            for index in pending:
                durations[index] = remaining * weights[index] / total_weight
            break
        for index in short:
            durations[index] = floors[index]
            remaining -= floors[index]
            pending.remove(index)
    return durations


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

    speech_floors = []
    has_dialogue = any(beat.get("dialogue_ids") for bucket in assignments for beat in bucket)
    for shot, bucket in zip(shots, assignments):
        ids = [str(did).upper() for beat in bucket for did in (beat.get("dialogue_ids") or [])]
        words = sum(_dialogue_word_count(dialogue_map.get(did, {}).get("text")) for did in ids)
        floor = words / _H3_DIALOGUE_MAX_WORDS_PER_SECOND + len(ids) * 0.2
        if has_dialogue and not ids:
            # Keep actual arrival/travel phases readable instead of borrowing
            # virtually all their time to fit an overlong AI exchange.
            floor = min(2.0, _h3_filmable_timing_weight(bucket, dialogue_map))
            from services.h3_action_clock import is_h3_stationary_reaction
            if is_h3_stationary_reaction(shot.get('_timing_source_action') or ' '.join(
                str(beat.get('description') or '') for beat in bucket
            )):
                # Elaborate facial prose does not require a separate two-second
                # pause. Let these reactions use spare time after exact speech,
                # while preserving travel/action floors and explicit holds.
                floor = min(floor, 0.75)
        speech_floors.append(floor)
    durations = None
    if any(shot.get("_action_floor") for shot in shots):
        from promptbench.story_time import action_first_clock
        durations = action_first_clock(shots, weights, total, speech_floors)
    if durations is None:
        durations = _h3_filmable_shot_durations(weights, total, minimums=speech_floors)
    cursor = 0.0
    for index, (shot, shot_duration) in enumerate(zip(shots, durations)):
        start = cursor
        end = total if index + 1 == len(shots) else cursor + shot_duration
        shot["start_seconds"] = round(start, 3)
        shot["end_seconds"] = round(end, 3)
        cursor = end


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
    minimum = _h3_minimum_shot_seconds(total, len(shots))
    if (dialogue_shot * minimum > lead_cap
            or (len(shots) - dialogue_shot) * minimum > total - lead_cap):
        return  # An over-segmented opening needs writing repair, not invalid clocks.
    # Scaling each part proportionally can turn an already-minimal phase into
    # an unusably short tail. Reuse the bounded allocator and retain every phase.
    adjusted = _h3_filmable_shot_durations(
        shot_durations[:dialogue_shot], lead_cap,
        minimums=[minimum] * dialogue_shot,
    ) + _h3_filmable_shot_durations(
        shot_durations[dialogue_shot:], total - lead_cap,
        minimums=[minimum] * (len(shots) - dialogue_shot),
    )
    cursor = 0.0
    for index, (shot, shot_duration) in enumerate(zip(shots, adjusted)):
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
    durations = _h3_filmable_shot_durations(weights, duration)
    cursor = 0.0
    shots: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        start = cursor
        end = duration if index + 1 == len(buckets) else cursor + durations[index]
        cursor = end
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
    guide = _load_h3_planning_guide("minimax_h3_story_ledger", nsfw=nsfw)
    if planning_style == "adaptive":
        from services.adaptive_enhancement import adaptive_writing_guide
        guide += "\n\n" + adaptive_writing_guide(prompt)

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
                    if planning_style in {"creative", "adaptive"} else
                    "The user concept is locked: do not invent new plot events, outcomes, or dialogue.\n\n"
                )
                + f"CHAPTER GEOMETRY:\n{chapter_geometry}\n\n"
                f"REFERENCE CONTEXT:\n{reference_context or 'None.'}\n\n"
                f"USER CONCEPT:\n{prompt}"
            ),
            system_prompt=guide,
            max_new_tokens=min(7600, 1800 + chapter_count * 330),
            temperature=0.38 if planning_style in {"creative", "adaptive"} else 0.24,
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
                "dialogue_writing_target": (
                    budget.instruction() if allow_generated_dialogue and
                    (budget := creative_dialogue_budget(prompt, durations[absolute_index])) else ""
                ),
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
                    int(math.floor(durations[absolute_index] * _H3_DIALOGUE_MAX_WORDS_PER_SECOND))
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
                                "maxItems": 6 if allow_generated_dialogue else 0,
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
                        f"Write character-specific dialogue in each segment's dialogue array when it advances the interaction. Aim for {_H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND:g} words per second during speech, allowing up to {_H3_DIALOGUE_MAX_WORDS_PER_SECOND:g}. Respect dialogue_word_budget as a hard maximum; leave time for action and pauses and use an empty array when no line belongs there. Exact quoted lines are inserted separately and must not be repeated.\n\n"
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
                temperature=0.4 if planning_style in {"creative", "adaptive"} else 0.28,
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
                int(math.floor(durations[segment_number - 1] * _H3_DIALOGUE_MAX_WORDS_PER_SECOND))
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
                for raw_line in (proposed.get("dialogue") or [])[:6]:
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
        if (
            not segment.get("semantic_actions")
            and exact_action and _normalize_key(exact_action) not in _normalize_key(proposed_action)
        ):
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
            **({'_timing_source_action': shot['_timing_source_action']}
               if shot.get('_timing_source_action') else {}),
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
        # Canonicalization already chose the story state. In an imported
        # timeline, its concrete camera-authored handoff must not be replaced
        # here with a copy of the entire completed source event.
        "closing_state": sanitize_h3_prompt_text(segment.get("closing_state")) or (
            sanitize_h3_prompt_text(beats[-1].get("state_after")) if beats else ""
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
                / _H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND
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


def _complete_creative_dialogue(
    prompt: str,
    ledger: dict[str, Any],
    *,
    canonical_ledger: dict[str, Any],
    locked_dialogue: list[dict[str, Any]],
    durations: list[float],
    generate: Callable[..., str],
    system_prompt: str,
    copyedit_system_prompt: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    from services.h3_dialogue_writing import complete_creative_dialogue

    return complete_creative_dialogue(
        prompt, ledger, canonical_ledger=canonical_ledger,
        locked_dialogue=locked_dialogue, durations=durations,
        generate=generate, system_prompt=system_prompt,
        copyedit_system_prompt=copyedit_system_prompt,
    )


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
    has_start_image: bool = False,
    nsfw: bool = False,
    llm_generate: Callable[..., str] | None = None,
    resume: dict | None = None,
) -> dict[str, Any]:
    """Create a compact ledger and expand one validated local segment at a time."""

    from services import llm_service

    with llm_service.keep_loaded():
        context = deepcopy(resume["context"]) if resume else _prepare_h3_story_context(
            prompt, segment_durations=segment_durations, mode=mode,
            camera_coverage=camera_coverage, reference_context=reference_context,
            expect_dialogue=expect_dialogue, planning_style=planning_style,
            image_paths=image_paths, has_start_image=has_start_image,
            nsfw=nsfw, llm_generate=llm_generate,
        )
        return _render_h3_story_segments(
            context, generate=llm_generate or llm_service.generate, resume=resume,
        )


def _prepare_h3_story_context(
    prompt: str,
    *,
    segment_durations: list[float],
    mode: str,
    camera_coverage: str,
    reference_context: str,
    expect_dialogue: bool,
    planning_style: str,
    image_paths: list[str] | None,
    has_start_image: bool,
    nsfw: bool,
    llm_generate: Callable[..., str] | None,
) -> dict[str, Any]:
    from services import llm_service

    generate = llm_generate or llm_service.generate
    durations = [max(0.1, float(value)) for value in segment_durations]
    segment_count = len(durations)
    if not segment_count:
        raise ValueError("H3 story planning requires at least one segment.")
    locked_dialogue = extract_locked_dialogue(prompt)
    # Callers historically detected only quotation marks.  Treat any
    # canonical dialogue form, including ``CHARACTER: line`` screenplay rows,
    # as mandatory even when an older caller passes ``expect_dialogue=False``.
    expect_dialogue = bool(
        locked_dialogue or (
            not dialogue_forbidden(prompt)
            and (expect_dialogue or (
                normalize_h3_planning_style(planning_style) in {"creative", "adaptive"}
                and creative_dialogue_expected(prompt)
            ))
        )
    )
    if normalize_h3_planning_style(planning_style) == "adaptive":
        from services.adaptive_enhancement import adaptive_dialogue_expected
        expect_dialogue = adaptive_dialogue_expected(prompt)
    source_events = extract_source_events(prompt)
    has_authored_timing = bool(authored_timed_brief(prompt)["events"])
    source_intent = extract_h3_source_intent(prompt)
    reference_context = canonicalize_h3_reference_names(reference_context, source_intent["cast_names"])
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
    start_frame_supplied = bool(mode == "sliding_window" and has_start_image and image_paths)
    planning_warnings: list[str] = []
    planning_diagnostics: list[str] = []
    planning_notes: list[str] = []
    allow_generated_dialogue = bool(
        planning_style in {"creative", "adaptive"}
        and expect_dialogue
        and not dialogue_forbidden(prompt)
        and not _only_supplied_dialogue_requested(prompt)
    )
    if planning_style == "adaptive":
        from services.adaptive_enhancement import adaptive_dialogue_expansion_requested
        allow_generated_dialogue = adaptive_dialogue_expansion_requested(prompt)
    from promptbench.experiments import action_first_enabled, planning_thinking_enabled
    action_first = bool(action_first_enabled() and allow_generated_dialogue
                        and not locked_dialogue and 1 < segment_count <= _LONG_FORM_SEGMENTS_PER_CHAPTER)
    planning_thinking = planning_thinking_enabled()
    schedule_expect_dialogue = expect_dialogue and not action_first
    expected_dialogue_events = _expected_dialogue_events(prompt, locked_dialogue)
    fully_scripted_dialogue = (
        len(locked_dialogue) >= 2
        and len(expected_dialogue_events) == len(locked_dialogue)
    )
    # Detailed silent scripts have the same immutable event order as dialogue
    # scripts. Brief outlines that need additional intermediate progression
    # still use the semantic scheduler. A complete quoted exchange also has
    # an immutable chronology without timestamps: allocate its speech clock
    # here and let the writer direct performance, rather than rediscover IDs
    # or crowd the last window while leaving spare time in the first.
    faithful_locked_schedule = bool(
        (planning_style == "faithful" or (
            planning_style == "adaptive" and (has_authored_timing or fully_scripted_dialogue)
            and not allow_generated_dialogue
        ))
        and (locked_dialogue or len(source_events) >= max(2, segment_count))
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
    if start_frame_supplied:
        # A Frames image owns the initial visible scene. The copied script's
        # costume/pose descriptions must not become a competing immutable lock.
        # Keep role names and cardinality; the vision treatment describes their
        # actual appearance without changing event or dialogue ownership.
        canonical_ledger["subject_continuity"] = " ".join(filter(None, (
            reference_context,
            "Map the named roles to the people visible in the supplied first frame; "
            "keep their observed appearance and wardrobe throughout unless an assigned action changes them.",
            source_intent.get("cast_cardinality_contract"),
        )))
        canonical_ledger["initial_state"] = (
            "Begin in the supplied first frame's exact composition, poses, contacts and lighting. "
            "Continue the movement already visible before transitioning into the first requested action."
        )
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
        f"at most {max(1, int(math.floor(duration * _H3_DIALOGUE_MAX_WORDS_PER_SECOND)))} spoken words; "
        f"aim for {_H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND:g} words per second during speech, "
        f"up to {_H3_DIALOGUE_MAX_WORDS_PER_SECOND:g}, leaving time for action and pauses. "
        + (budget.instruction() if allow_generated_dialogue and
           (budget := creative_dialogue_budget(prompt, duration)) else "")
        for index, duration in enumerate(durations)
    )
    maximum_window_words = max(
        [max(1, int(math.floor(duration * _H3_DIALOGUE_MAX_WORDS_PER_SECOND))) for duration in durations]
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
    final_source_segment = _final_source_event_segment(
        source_events,
        locked_dialogue,
        _expected_dialogue_events(prompt, locked_dialogue),
        durations,
    )
    ending_assignment_instruction = (
        f"\nSHORT RESIDUAL TAIL: Segment {segment_count} is too short to perform "
        f"the final source event safely. Place the final E-id in segment "
        f"{final_source_segment}. Give every later residual segment a new "
        "derived visible hold or consequence with empty source_event_ids; do "
        "not replay or advance that source event there.\n"
        if final_source_segment < segment_count else
        "\nPlace the final E-id in the final segment.\n"
    )
    ledger_prompt = (
        f"Mode: {mode}. Camera coverage preference: {camera_coverage}.\n"
        f"Segment geometry:\n{geometry_lines}\n\n"
        f"Canonical reference context:\n{reference_context or 'No external reference map.'}\n\n"
        "APPLICATION-OWNED CAST AND BLOCKING CONTRACT (preserve exactly; only assigned active principals appear in a segment):\n"
        f"{canonical_ledger.get('subject_continuity')}\n\n"
        "PERSISTENT SCENE FACTS AND DIRECTIONS (apply throughout, not separate chronological events):\n"
        f"{source_intent.get('global_instructions') or 'None.'}\n\n"
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
        f"{ending_assignment_instruction}"
        "DIRECT THE SEMANTIC SCHEDULE. You own filmable beat grouping and segment allocation. "
        "Return beats in chronological order. Prefer one to three cohesive beats per segment; "
        "use additional short beats when needed for the source actions and ending, within its time budget. "
        "Do not repeat a source ID for a reaction or continuing conversation; those supporting beats use empty source_event_ids. "
        "Do not repeat, recap, preview, omit, or reorder any source event. "
        "a single broad E-id may begin earlier and develop through concrete derived beats. "
        "Keep each locked dialogue ID with its anchored source event. Except for any D-id explicitly listed for mechanical "
        "continuation above, keep dialogue within that segment's total spoken-word budget. "
        "Use state_after to describe a concrete visual handoff into the next segment, not a generic continuation phrase.\n"
        f"Writing mode: {planning_style.upper()}. "
        + (
            "Treat the user concept as a creative brief. Build one causal full-duration scene with a clear opening, escalation, and payoff. "
            "You may create supporting filmable beats with empty source_event_ids, but never contradict, repeat, or complete an explicit source event early. "
            "When dialogue is allowed, this concept requires an audible authored script: never leave speaking, telling, explaining, discussing, or reacting implicit in visual prose, "
            "and never return generated_dialogue empty. Develop unquoted conversational content into character-specific spoken exchanges that meet each segment's word target. "
            "Give interacting characters natural back-and-forth and a verbal response when the brief establishes one. Exact quoted lines are immutable anchors; "
            "additional lines may surround them unless the user explicitly says only those lines. Every generated line must fit its segment's spoken-word budget. "
            if planning_style in {"creative", "adaptive"} else
            "Treat the user concept as locked source material. Distribute and stage only supplied events and exact dialogue; do not invent new plot events, outcomes, or spoken lines. "
        )
        + f"Dialogue policy: {'Write developed, character-specific generated_dialogue entries; select exactly one segment for each. Use up to six turns per segment within its total word budget.' if allow_generated_dialogue else 'Do not add generated dialogue.'} "
        + (
            "This is a conversation-first brief: begin intelligible tagged dialogue in segment 1 and use each segment's spoken-word allocation for a developed exchange. Prefer two to four natural turns when characters interact. Do not spend a complete opening segment on silent setup; a brief entrance or establishing action may occupy only the first few seconds before speech begins.\n\n"
            if spread_generated_dialogue else "\n\n"
        )
        + "Also return shared continuity, setting, visual language, editing style, opening state, nonverbal ambience, music, and the visible final outcome.\n\n"
        + (f"Required spoken talking points (cover their substance in generated_dialogue, not only beat descriptions):\n{json.dumps(requested_dialogue_topics(prompt), ensure_ascii=False)}\n\n" if allow_generated_dialogue and requested_dialogue_topics(prompt) else "")
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
        if planning_style == "adaptive":
            ledger_prompt += (
                "\n\nSOURCE ADAPTATION: Resolve the user's requested changes to this copied brief. "
                "A request to adapt, replace or change something overrides the corresponding detail "
                "inside the pasted template. source_adaptation records those specific replacements "
                "and any necessary opening bridge, in at most 120 words. Keep the cast role names, "
                "attack ownership, causal event order, restrictions and ending. Do not rewrite the "
                "event schedule or merely say 'preserve consistency'. The selected native duration "
                "supersedes durations written for another model."
            )
        if start_frame_supplied:
            ledger_prompt = (
                "EXACT START FRAME: The first attached image is the actual frame at 0 seconds, "
                "not an identity sample. Use its visible wardrobe, physical contacts, posture, "
                "screen positions, camera view, lighting and immediate surroundings. Describe that "
                "observed state in initial_state, without advancing the story. setting_continuity "
                "must describe the same observed place and lighting, not copy the template's weather "
                "or replace its foreground set. Use character_appearance "
                "to map each named role to the visible person. A copied opening pose or costume cannot "
                "replace this image. The camera writer will connect this observed state to the "
                "first scripted action; describe the actual current pose, not its recovery or a "
                "different pose from the template. Preserve later requested "
                "geography through motivated movement/reveals rather than teleporting to a replacement set.\n\n"
            ) + ledger_prompt
    ledger_guide = _load_h3_planning_guide(
        "minimax_h3_story_treatment"
        if faithful_locked_schedule else
        "minimax_h3_story_ledger",
        nsfw=nsfw,
    )
    if planning_style == "adaptive":
        from services.adaptive_enhancement import adaptive_writing_guide
        ledger_guide += "\n\n" + adaptive_writing_guide(prompt)
    ledger_schema = (
        _faithful_treatment_schema(
            list(source_intent.get("cast_names") or []),
            resolve_adaptation=planning_style == "adaptive", start_frame=start_frame_supplied,
        )
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
    if action_first:
        from promptbench.story_time import schedule_request
        ledger_prompt, ledger_guide, ledger_schema = schedule_request(
            ledger_schema, prompt=prompt, durations=durations, canonical=canonical_ledger,
            events=source_events, references=reference_context,
        )
    # Even a short, silent brief needs developed action and a concrete handoff
    # for every requested segment. Size that space by the output schedule, not
    # just by how many events the user supplied. Spoken screenplays also need
    # a line catalog. The focused dialogue pass owns wording/length repair.
    ledger_token_budget = min(
        4200,
        max(3200 if allow_generated_dialogue else 1800,
            950 + segment_count * 600 + len(source_events) * 150),
    )
    schedule_dialogue_durations = None if allow_generated_dialogue else durations
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
            expect_dialogue=schedule_expect_dialogue,
            allow_generated_dialogue=allow_generated_dialogue,
            require_dialogue_per_segment=False,
            segment_durations=schedule_dialogue_durations,
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
                    (1800 if start_frame_supplied or planning_style == "adaptive" else 1000)
                    if faithful_locked_schedule else
                    ledger_token_budget
                ),
                temperature=0.38 if planning_style in {"creative", "adaptive"} else 0.22,
                top_p=0.84,
                image_paths=image_paths or None,
                enable_thinking=planning_thinking,
                thinking_budget=2048 if planning_thinking else 0,
                reasoning_effort="medium" if planning_thinking else None,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                json_schema=None if planning_thinking else ledger_schema,
            )
            from services.h3_window_planner import _parse_json_object

            candidate = _parse_json_object(raw)
            ledger = (
                _apply_faithful_treatment(
                    canonical_ledger, candidate, resolve_adaptation=planning_style == "adaptive",
                    start_frame=start_frame_supplied,
                )
                if faithful_locked_schedule else
                _canonicalize_story_ledger(
                    prompt,
                    canonical_ledger,
                    candidate,
                    locked_dialogue=locked_dialogue,
                    segment_count=segment_count,
                    allow_generated_dialogue=allow_generated_dialogue,
                    preserve_adaptation=planning_style == "adaptive",
                )
            )
            treatment_review_fields = ledger.pop("_treatment_review_fields", [])
            if treatment_review_fields:
                planning_warnings.append(
                    "The AI's shared cinematic treatment needs review for "
                    + ", ".join(field.replace("_", " ") for field in treatment_review_fields)
                    + ". Maestro omitted those invalid fields instead of restoring "
                    "potentially conflicting descriptions from the copied source."
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
                expect_dialogue=schedule_expect_dialogue,
                allow_generated_dialogue=allow_generated_dialogue,
                require_dialogue_per_segment=False,
                segment_durations=schedule_dialogue_durations,
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
                        "you may regroup beats or move whole events between segments to satisfy timing. "
                        "Check that window numbers never decrease as the chronological beat list advances."
                    ),
                    system_prompt=ledger_guide,
                    max_new_tokens=ledger_token_budget,
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
                    preserve_adaptation=planning_style == "adaptive",
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
                    expect_dialogue=schedule_expect_dialogue,
                    allow_generated_dialogue=allow_generated_dialogue,
                    require_dialogue_per_segment=False,
                    segment_durations=schedule_dialogue_durations,
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
                        "Maestro restored the ordered source story and exact user-written dialogue using its "
                        "duration-aware emergency schedule."
                    )
                else:
                    planned_by = "deterministic_fallback"
                    planning_warnings.append(
                        "The AI story schedule did not preserve Maestro's locked event and dialogue map after one focused repair, "
                        "so Maestro used its deterministic duration-aware timing schedule. Every supplied event and exact "
                        "user-written line remains intact."
                    )

    if planning_thinking:
        planning_notes.append(
            "Prompt-bench planning-thinking candidate: the existing story-ledger call used a bounded reasoning budget."
        )
    if action_first:
        from promptbench.story_time import reserve_time
        reserve_time(ledger, durations)
        planning_notes.append("Prompt-bench action-first candidate: physical action reserved before authored speech.")
    if allow_generated_dialogue:
        ledger, _ = _complete_creative_dialogue(
            prompt, ledger, canonical_ledger=canonical_ledger,
            locked_dialogue=locked_dialogue, durations=durations,
            generate=generate, system_prompt=_load_h3_planning_guide("minimax_h3_dialogue", nsfw=nsfw),
            copyedit_system_prompt=_load_h3_planning_guide("minimax_h3_dialogue_copyedit", nsfw=nsfw),
        )
        # This pass uses the whole-window writing target. Review the final
        # speech after camera staging and copyediting, when arrival/travel
        # phases have their own time and earlier density warnings can be stale.

    # Camera perspective, requested speed, style, nonverbal reactions, and
    # sequence shape are immutable even when the creative ledger succeeds.
    # Merge them after schema validation so the small LLM never owns them.
    # Principal identity, exact cast count, reference ownership, and blocking
    # are application-owned. A camera/story LLM may add useful choreography,
    # but it may not turn a location into a subject, drop an unreferenced
    # principal, or quietly duplicate a named character.
    authored_subjects = sanitize_h3_prompt_text(ledger.get("subject_continuity"))
    source_owns_subject_appearance = _source_owns_h3_subject_appearance(
        prompt,
        start_frame_supplied=start_frame_supplied,
        reference_context=reference_context,
    )
    ledger["subject_continuity"] = sanitize_h3_prompt_text(
        canonical_ledger.get("subject_continuity")
    )
    if (
        planning_style == "adaptive"
        and not faithful_locked_schedule
        and not source_owns_subject_appearance
        and authored_subjects
    ):
        # The canonical contract still owns supplied identities/bindings. Keep
        # the writer's developed appearance and contrast as well; replacing it
        # with "keep identities unchanged" erases the cast of an open brief.
        if authored_subjects not in ledger["subject_continuity"]:
            ledger["subject_continuity"] += ". " + authored_subjects
    if source_owns_subject_appearance:
        # These fields can otherwise stage a later character at the opening,
        # invent anatomy/propulsion, or replace the authored resting outcome.
        # Source events still receive cinematic camera/action development.
        structured_source = bool(_CONTEXT_IR_FIELD.search(str(prompt or "")))
        structured_text_owns_state = bool(structured_source and not start_frame_supplied)
        _lock_h3_source_owned_context(
            ledger,
            canonical_ledger,
            lock_initial_state=structured_text_owns_state,
            lock_mechanics=structured_text_owns_state,
        )
    if (
        faithful_locked_schedule
        and not _CONTEXT_IR_FIELD.search(str(prompt or ""))
        and ledger.get("character_appearance")
    ):
        # A Frames/Ref2VA treatment may describe the supplied pixels and role
        # map. Keep that observed profile after restoring the canonical media
        # contract; native Context-IR already contains its own visual facts.
        ledger["subject_continuity"] += " " + sanitize_h3_prompt_text(
            ledger["character_appearance"]
        )
    if source_intent.get("opening_state_contract") and not start_frame_supplied:
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
    if source_intent.get("global_instructions") and faithful_locked_schedule:
        # The LLM has already adapted the production notes. Re-inserting the
        # entire original style essay crowds the actual local action out.
        visual_contract = sanitize_h3_prompt_text(ledger.get("visual_continuity"))
    # Ability mechanics are authored separately from media bindings and style.
    # Ref2VA uses the canonical reference manifest for subject_definitions, so
    # mechanics left only in LLM subject prose disappear during compilation.
    mechanics = sanitize_h3_prompt_text(ledger.get("motion_mechanics"))
    if mechanics and mechanics.casefold() not in {"n/a", "none"} and mechanics not in visual_contract:
        visual_contract = ". ".join(part for part in (visual_contract, mechanics) if part)
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

    if planning_style != "adaptive" or faithful_locked_schedule:
        _lock_ledger_source_events(prompt, ledger)
    # Adaptive prose has already passed schedule validation. The source event
    # catalog below remains immutable and segment checks still require its
    # facts; do not replace useful staging with the original one-line concept.
    catalog = _dialogue_catalog(ledger, locked_dialogue)
    # Render phases need the same event bindings as the writer. Generated
    # speech is not an exact user quote and must not be guessed from source
    # text, but its explicit event anchor keeps it beside the right action.
    render_dialogue_anchors = dict(expected_dialogue_events)
    for item in ledger.get("generated_dialogue") or []:
        event_id = item.get("source_event_id")
        if event_id and any(
            item["dialogue_id"] in (beat.get("dialogue_ids") or [])
            and event_id in (beat.get("source_event_ids") or [])
            for beat in ledger.get("beats") or []
        ):
            render_dialogue_anchors[item["dialogue_id"]] = event_id
    try:
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
            expected_dialogue_events=render_dialogue_anchors,
        )
    except H3DialogueTimingError:
        if not ledger.get("generated_dialogue"):
            raise
        # An AI wording failure must not be reported as an oversized user
        # screenplay, nor erase the draft. First verify the user's own exact
        # lines can fit; that error still requires a real duration change.
        locked_ids = {item["dialogue_id"] for item in locked_dialogue}
        locked_beats = deepcopy(ledger.get("beats") or [])
        for beat in locked_beats:
            beat["dialogue_ids"] = [did for did in beat.get("dialogue_ids", []) if did in locked_ids]
        _prepare_render_dialogue_schedule(
            locked_beats, [item for item in catalog if item["dialogue_id"] in locked_ids],
            segment_durations=durations, source_events=source_events,
            expected_dialogue_events=expected_dialogue_events,
        )
        render_beats = deepcopy(ledger.get("beats") or [])
        render_dialogue_events = render_dialogue_anchors
        dialogue_fragments = []
        planning_notes.append(
            "The initial AI script exceeded its speech allocation; final camera timing and copyediting determine whether it fits."
        )
    if dialogue_fragments:
        planning_notes.append(
            "Maestro continued long exact dialogue across adjacent H3 windows "
            "to preserve every user-written word without rushing, repetition, "
            "or paraphrasing."
        )
    return {
        "action_first": action_first,
        "allow_generated_dialogue": allow_generated_dialogue,
        "camera_coverage": camera_coverage,
        "catalog": catalog,
        "dialogue_fragments": dialogue_fragments,
        "durations": durations,
        "faithful_locked_schedule": faithful_locked_schedule,
        "has_authored_timing": has_authored_timing,
        "image_paths": image_paths,
        "ledger": ledger,
        "locked_dialogue": locked_dialogue,
        "long_form_hierarchical": long_form_hierarchical,
        "mode": mode,
        "nsfw": nsfw,
        "planned_by": planned_by,
        "planning_diagnostics": planning_diagnostics,
        "planning_notes": planning_notes,
        "planning_style": planning_style,
        "planning_warnings": planning_warnings,
        "prompt": prompt,
        "render_beats": render_beats,
        "render_dialogue_events": render_dialogue_events,
        "segment_count": segment_count,
        "source_events": source_events,
        "source_intent": source_intent,
        "start_frame_supplied": start_frame_supplied,
    }


def _render_h3_story_segments(context: dict[str, Any], *, generate, resume: dict | None = None) -> dict[str, Any]:
    # Freeze the story and speech clock before camera writing mutates local
    # catalog entries. Retrying a window never reruns the story/dialogue writer.
    checkpoint_context = deepcopy(context)
    saved_segments = (resume or {}).get("segments") or []
    retry_windows = set((resume or {}).get("retry_windows") or [])
    if saved_segments:
        print("[MiniMax H3] Retrying windows " + ", ".join(map(str, sorted(retry_windows)))
              + f"; keeping {len(saved_segments) - len(retry_windows)} saved camera plans.")
    action_first = context["action_first"]
    allow_generated_dialogue = context["allow_generated_dialogue"]
    camera_coverage = context["camera_coverage"]
    catalog = context["catalog"]
    dialogue_fragments = context["dialogue_fragments"]
    durations = context["durations"]
    faithful_locked_schedule = context["faithful_locked_schedule"]
    has_authored_timing = context["has_authored_timing"]
    image_paths = context["image_paths"]
    ledger = context["ledger"]
    locked_dialogue = context["locked_dialogue"]
    long_form_hierarchical = context["long_form_hierarchical"]
    mode = context["mode"]
    nsfw = context["nsfw"]
    planned_by = context["planned_by"]
    planning_diagnostics = context["planning_diagnostics"]
    planning_notes = context["planning_notes"]
    planning_style = context["planning_style"]
    planning_warnings = context["planning_warnings"]
    prompt = context["prompt"]
    render_beats = context["render_beats"]
    render_dialogue_events = context["render_dialogue_events"]
    segment_count = context["segment_count"]
    source_events = context["source_events"]
    source_intent = context["source_intent"]
    start_frame_supplied = context["start_frame_supplied"]
    segment_guide = _load_h3_planning_guide("minimax_h3_story_segment", nsfw=nsfw)
    if planning_style == "adaptive":
        from services.adaptive_enhancement import adaptive_writing_guide
        segment_guide += "\n\n" + adaptive_writing_guide(prompt)
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
    review_windows: set[int] = set()
    speech_action_order = _explicit_speech_action_order(prompt)
    previous_closing = sanitize_h3_prompt_text(ledger.get("initial_state"))
    for index, duration in enumerate(durations):
        segment_number = index + 1
        from services.studio_enhancement import check_cancelled
        check_cancelled()
        saved_segment = saved_segments[index] if saved_segments else None
        if saved_segment and segment_number not in retry_windows:
            segments.append(deepcopy(saved_segment))
            previous_closing = saved_segment["closing_state"]
            continue
        # Both neighbouring windows are already authored. Repair inside this
        # window's existing entry/exit states instead of moving their story.
        if saved_segment:
            previous_closing = saved_segment["opening_state"]
        semantic_beats = [
            item for item in render_beats
            if isinstance(item, dict) and int(item.get("segment") or 0) == segment_number
        ]
        beats = _camera_phase_beats(
            semantic_beats,
            source_events=source_events,
            expected_dialogue_events=render_dialogue_events,
            preserve_adaptation=planning_style == "adaptive" and not has_authored_timing,
        )
        if not has_authored_timing:
            # A supplied timed shot is already an authored unit of action.
            # Merging ten such units into four large cards encouraged writers
            # to omit the middle exchanges while still covering each card.
            # Keep each timed event independently required by the schema.
            beats = _coalesce_camera_phases(beats, target_count=4)
        if segment_number == 1 and start_frame_supplied and beats and not beats[0].get("dialogue_ids"):
            beats[0]["_start_frame_continuation"] = True
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
        if camera_coverage == "continuous":
            mode_instruction = mode_instruction.replace("Internal motivated cuts are allowed, but ", "")
            mode_instruction += (
                " This scene is one unbroken camera take. Timed phases are movements within that take, "
                "not new shots or cuts; use continuous reframing for each later phase."
            )
        dialogue_by_id = {
            str(item.get("dialogue_id") or "").upper(): item
            for item in assigned_dialogue
        }
        creative_beats = []
        authored_total = sum(float(beat.get("authored_duration") or 0) for beat in beats)
        for event_number, beat in enumerate(beats, start=1):
            beat_dialogue = [
                dialogue_by_id.get(str(dialogue_id or "").upper())
                for dialogue_id in (beat.get("dialogue_ids") or [])
            ]
            local_order = {}
            for item in beat_dialogue:
                if not isinstance(item, dict):
                    continue
                source_id = item.get('source_dialogue_id') or item.get('dialogue_id')
                requirements = dict(speech_action_order.get(source_id) or {})
                if int(item.get('fragment_index') or 1) > 1:
                    requirements.pop('before_speech', None)
                if int(item.get('fragment_index') or 1) < int(item.get('fragment_count') or 1):
                    requirements.pop('after_speech', None)
                if requirements:
                    local_order[item['dialogue_id']] = requirements
            if local_order:
                beat['_speech_action_order'] = local_order
            creative_beats.append({
                "event_index": event_number,
                **({
                    "source_requirements": [
                        event for event in source_events
                        if event["event_id"] in (beat.get("source_event_ids") or [])
                    ],
                    "staging_draft": sanitize_h3_prompt_text(beat.get("description")),
                } if planning_style == "adaptive" else {
                    "event": sanitize_h3_prompt_text(beat.get("description")),
                }),
                **({"duration_seconds": round(duration * float(beat.get("authored_duration") or 0) / authored_total, 3)} if authored_total > 0 else {}),
                "dialogue_performances": [
                    {
                        "dialogue_id": item.get("dialogue_id"),
                        "speaker": item.get("speaker"),
                        "off_camera": bool(item.get("off_camera")),
                        "spoken_words": _dialogue_word_count(item.get("text")),
                        "speaking_seconds": round(_dialogue_word_count(item.get("text")) / _H3_DIALOGUE_PREFERRED_WORDS_PER_SECOND + 0.2, 2),
                        **({"required_action_order": local_order[item['dialogue_id']]} if item['dialogue_id'] in local_order else {}),
                    }
                    for item in beat_dialogue
                    if isinstance(item, dict)
                ],
                ("draft_resulting_state" if planning_style == "adaptive" else "resulting_state"):
                    sanitize_h3_prompt_text(beat.get("state_after")),
                "sound_effects": sanitize_h3_prompt_text(beat.get("sound_effects")),
            })
            if planning_style == "adaptive" and faithful_locked_schedule:
                # This path has no AI staging draft: both fields repeat the
                # complete source event, often tripling a long imported script.
                # Send each assigned event once and let camera writing stage it.
                creative_beats[-1].pop("staging_draft", None)
                creative_beats[-1].pop("draft_resulting_state", None)
        creative_dialogue = [
            {
                "dialogue_id": item.get("dialogue_id"),
                "speaker": item.get("speaker"),
                "language": item.get("language"),
                "exact_text": item.get("text"),
                "off_camera": bool(item.get("off_camera")),
            }
            for item in assigned_dialogue
        ]
        production_directions = source_intent.get("global_instructions") or prompt
        if planning_style == "adaptive" and faithful_locked_schedule:
            # The vision/treatment pass has resolved the copied brief already.
            # Repeating its old costume and setting essay in every camera call
            # used to undo that resolution, especially in later windows.
            production_directions = "\n".join(str(source_intent.get(key) or "") for key in (
                "negative_constraints", "perspective_contract", "pacing_contract",
            ))
        if planning_style == "adaptive" and not source_intent.get("global_instructions"):
            # The complete plot already belongs to the story schedule. Feeding
            # it to every camera pass invites replay of earlier/later actions.
            production_directions = "\n".join(str(source_intent.get(key) or "") for key in (
                "negative_constraints", "style_contract", "perspective_contract", "pacing_contract",
            ))
        event_heading = (
            "Assigned chronological events (depict each once, in order):"
            if planning_style == "adaptive" else "Immutable chronological events (depict each once, in order):"
        )
        segment_prompt = (
            f"Segment {segment_number} of {segment_count}; local duration 0.000 to {duration:.3f} seconds.\n"
            f"{mode_instruction}\n\n"
            f"Shared subjects: {ledger.get('subject_continuity')}\n"
            f"Active principal cast for this segment: {active_cast_contract or 'Use only the principals required by the assigned events.'}\n"
            f"Blocking contract: {blocking_contract or 'Preserve the concrete opening geography and every completed state change across cuts.'}\n"
            f"Shared setting: {ledger.get('setting_continuity')}\n"
            f"Shared visual language: {ledger.get('visual_continuity')}\n"
            f"Editing style: {ledger.get('editing_style')}\n"
            + (f"Previous camera geography (carry its landmarks and axis forward): "
               f"{segments[-1].get('coverage')}\n" if segments else "")
            + f"Required pacing and performance energy: {source_intent.get('pacing_contract')}\n"
            f"Required opening state: {previous_closing}\n"
            + (f"Resolved source adaptation (takes precedence over copied descriptors in event text): "
               f"{ledger['source_adaptation']}\n" if ledger.get("source_adaptation") else "")
            + ("Exact frame continuity: shared observed appearances and the required opening state "
               "take precedence over incompatible costume, pose or opening-camera descriptions in the "
               "copied events. Begin from that state and show the necessary transition before the "
               "next attack; do not cut immediately to an incompatible composition.\n"
               if start_frame_supplied else "")
            + f"{event_heading}\n"
            f"{json.dumps(creative_beats, ensure_ascii=False, indent=2)}\n\n"
            f"Immutable dialogue performances (plan visible/off-camera performance but do not reproduce text in JSON):\n"
            f"{json.dumps(creative_dialogue, ensure_ascii=False, indent=2)}\n\n"
            "Write the required event_cards object: event_1, event_2, and so on. "
            "The schema already assigns every event and speaking turn; fill those cards rather than "
            "choosing identifiers, allocating timestamps, or returning a shots array. "
            "First use coverage to establish concrete camera geography and the requested camera energy: "
            "relevant landmarks, the local action axis and motivated changes of angle or distance. "
            "Stable geography does not require a fixed camera or make every journey travel in the "
            "same screen direction; an explicitly shown return reverses its outbound route. "
            "Use one or two sentences, not a generic label such as cinematic impact angles. "
            "Carry those anchors into the action and camera fields; choose a wider view or a shown "
            "reframe when needed to make a change of direction legible. "
            "For a silent event, use one to four phases for its distinct causal exchanges. An event "
            "can contain several exchanges and camera changes; do not default to one long shot for "
            "a whole action sequence. For dynamic multi-shot fighting, give successive attacks, "
            "counters or recoveries into new attacks their own purposeful coverage while preserving "
            "all source actions and their available time. If one exchange fills a short event, one "
            "card is enough. Write action first, then choose coverage that proves it. Keep an attack, "
            "its contact or defense, and the immediate "
            "physical response together; do not spend one card launching a strike and the next "
            "showing it connect. Adjacent source paragraphs may continue the same flight or impact: "
            "advance from the point already reached instead of launching or colliding again. "
            "For a speaking event, fill each required dialogue key (D1, D2, etc.) with a performance "
            "card showing that speaker's physical delivery in the established scene. "
            "The speaker's face stays readable throughout their line; the others listen. "
            "Honor explicitly off-camera voices. Exact words are inserted by the compiler. "
            "Each line's lead_in is null unless a physical action must happen BEFORE that line, "
            "such as arriving or crossing a doorway. Put that action in the lead_in card, "
            "then write performance from the resulting position. Do not use lead_in for decorative "
            "pauses, repeated setup, or reaction shots. Honor required_action_order: before_speech goes "
            "in lead_in, during_speech stays in performance, and after_speech goes in follow_through "
            "(or the next line's lead_in when another line follows). Depict the whole required action "
            "and its consequence there, not just in closing_state. follow_through is null unless an action must "
            "happen AFTER the event's final line. Plan both people entering before closing a door. "
            "Keep cards concise and paired with their camera; use enough detail for the actual "
            "choreography. Speaking cards need concise physical acting; include small expressions "
            "and gestures during the assigned line. Null lead_in/follow_through slots stay null. "
            "Use 30–90 action words only for substantial choreography. The compiler calculates the local clock "
            "from speaking time, physical action and any authored event durations. "
            "Do not reproduce speech text in action or sound_effects.\n\n"
            "Write closing_state as the literal visible situation in the FINAL frame after these "
            "events: current positions, prop ownership, open/closed doors and lasting changes. "
            "Do not retell completed actions, list instructions, predict what will happen next, "
            "or start an event belonging to a later segment. The next camera writer starts from "
            "this already-achieved state; nothing here should be performed again.\n\n"
            f"User production directions and constraints:\n{production_directions}"
        )
        if action_first:
            segment_prompt += (
                f"\nReserved action clock: {json.dumps(ledger['_story_time'][index], ensure_ascii=False)}. "
                "Give reserved movements their own silent action cards; write natural concise performances in the remaining time. "
                "Show the action reaching the required state; do not substitute an intention or invitation for its completion. "
                + (f"Required visible ending: {ledger.get('required_final_outcome')}." if segment_number == segment_count else "")
            )
        if saved_segment:
            segment_prompt += (
                "\nREPAIR THIS WINDOW ONLY. The story schedule and other windows are already saved. "
                "Complete the assigned actions between the required opening above and this exact "
                f"visible ending: {saved_segment['closing_state']}. "
                "Stage the final action so it reaches that ending; do not change either boundary "
                "or add a transition/event belonging to a neighbouring window."
            )
        if planning_style == "adaptive":
            segment_prompt += (
                "\nThe source_requirements contain the user's actual events; preserve their actions, "
                "participants, explicit blocking and outcomes. staging_draft and draft_resulting_state "
                "are the AI's proposed staging, not additional user requirements. Before writing cards, "
                "reconcile that draft with the required opening state and the user's requirements. "
                "Preserve meaning, not copied figurative wording. Explain airborne movement only from "
                "a cause established by the source: gravity and momentum, tension in a named web, rope "
                "or cable, a shown surface push-off, an impact, requested unaided flight, or stated "
                "equipment. Do not invent innate flight, costume parts, footwear details, emissions or "
                "anatomy. Carry only relevant shared ability mechanics into the airborne action. "
                "Correct contradictory aim, trajectory, contact, prop operation or active effects with "
                "the smallest necessary staging adjustment. Do not copy a physical contradiction merely "
                "because it appears in the draft. Keep event order, requested progression and exact dialogue. "
                "An empty source_requirements list permits the assigned connective beat, not a new plot. "
                "\nGive each numbered event its own timed action phase or consecutive phases. In a continuous take, "
                "these phases have no cuts. Depict the action causing the resulting state inside action; "
                "a closing_state alone does not show how an impact or change happened."
            )
        if beats and beats[0].get("_start_frame_continuation"):
            segment_prompt += (
                "\n\nFIRST-FRAME ACTION: event_1.opening is the first timed action card. "
                "Begin in the required opening state's camera composition and immediately advance "
                "the movement already visible. Write recovery first: the short physical bridge from "
                "the photographed pose, including any release, lowering of an extended limb, landing "
                "or turn needed for the first scripted action. Use an empty recovery only when no "
                "bridge is needed. Then action begins AFTER that recovery and carries the first "
                "scripted attack through contact and the opponent's immediate response. The compiler "
                "joins recovery and action into this one timed phase; do not repeat recovery in action. "
                "This card is motion, not a restatement or hold of the still image. Keep the original "
                "first event's attack ownership and outcome: opening plus its zero-to-three following "
                "phases must complete that entire event, including the cause of the next event. "
                "Do not replace the first attack with the photographed pose. Prefer completing the "
                "recovery, first attack and immediate response in opening, leaving phases empty. "
                "If another card is needed, end opening on a completed physical change, not a strike "
                "whose contact is deferred to the next card. Continue forward without replay. "
                "Move the camera from its observed position "
                "before adopting a new framing; do not begin with a cut."
            )
        if long_form_hierarchical and not saved_segment:
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
            event_count=len(beats),
            dialogue_ids=assigned_dialogue_ids,
            assigned_beats=beats,
            minimum_shots=(
                len(beats) if beats and (planning_style == "adaptive" or all(float(beat.get("authored_duration") or 0) > 0 for beat in beats))
                else 1
            ),
        )
        if saved_segment:
            schema["properties"]["closing_state"] = {
                "type": "string", "enum": [saved_segment["closing_state"]],
            }
        segment: dict[str, Any] | None = None
        segment_errors: list[str] = []
        # Reserve enough output for complete JSON plus readable choreography;
        # the former ~2K cap cut off otherwise useful four-shot responses.
        segment_token_budget = min(6144, max(4096, 1400 + len(beats) * 650 + len(assigned_dialogue) * 150))
        camera_images = image_paths[:1] if start_frame_supplied and segment_number == 1 else None
        camera_guide = segment_guide
        if camera_images:
            camera_guide = (
                "The attached image is the actual first video frame. Direct its visible movement "
                "into the assigned story, rather than replacing its pose with a copied opening. "
                "The first action starts in this image's framing and physical contacts. Show any "
                "necessary release, limb recovery, landing or turn before the scripted strike. "
                "Preserve that strike's owner, target and result. An opening pose or camera angle "
                "inside the pasted event is superseded by this frame; the event's meaningful action "
                "and outcome are still required.\n\n"
            ) + segment_guide
        try:
            raw = generate(
                prompt=segment_prompt,
                system_prompt=camera_guide,
                max_new_tokens=segment_token_budget,
                temperature=0.24,
                top_p=0.86,
                image_paths=camera_images,
                enable_thinking=False,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                json_schema=schema,
            )
            from services.h3_window_planner import _parse_json_object

            segment = _parse_json_object(raw, allow_repair=False)
            camera_draft = deepcopy(segment)
            segment = _canonicalize_segment_contract(
                segment,
                segment_number=segment_number,
                duration=duration,
                assigned_beats=beats,
                dialogue_catalog=catalog,
                opening_state=previous_closing,
                source_intent=source_intent,
                source_events=source_events,
                use_camera_handoff=faithful_locked_schedule and segment_number < segment_count,
            )
            segment_errors = segment_violations(
                prompt,
                segment,
                segment_number=segment_number,
                duration=duration,
                assigned_beats=beats,
                dialogue_catalog=catalog,
            )
            from services.h3_camera_fidelity import (
                clear_confirmed_coverage_errors, review_missing_camera_actions,
            )
            coverage_receipts = review_missing_camera_actions(
                segment_errors, segment, assigned_beats=beats,
                source_events=source_events, generate=generate,
            )
            segment_errors = clear_confirmed_coverage_errors(
                segment_errors, segment or {}, coverage_receipts,
            )
            if coverage_receipts:
                print(f"[MiniMax H3] Segment {segment_number}: source coverage review "
                      f"confirmed {len(coverage_receipts)} faithful paraphrase(s).")
            if segment_errors:
                print(
                    f"[MiniMax H3] Segment {segment_number} repair: "
                    + "; ".join(segment_errors)
                )
                feedback = _camera_repair_feedback(segment_errors, beats)
                card_matches = [re.match(r"event_cards\.(event_\d+):", item) for item in feedback]
                repair_keys = (
                    list(dict.fromkeys(match.group(1) for match in card_matches))
                    if all(card_matches) and isinstance(camera_draft, dict)
                    and isinstance(camera_draft.get("event_cards"), dict) else []
                )
                repair_prompt, repair_schema, rejected = segment_prompt, schema, raw
                if repair_keys:
                    # Repair only failed cards. Rewriting a whole good window
                    # to restore one missing detail used to change other
                    # actions, introduce fresh errors, and force a fallback.
                    selected_events = [
                        {"event_card": f"event_{i}", **event}
                        for i, event in enumerate(creative_beats, 1)
                        if f"event_{i}" in repair_keys
                    ]
                    repair_prompt = segment_prompt.replace(
                        json.dumps(creative_beats, ensure_ascii=False, indent=2),
                        json.dumps(selected_events, ensure_ascii=False, indent=2), 1,
                    )
                    cards_schema = schema["properties"]["event_cards"]
                    repair_schema = {
                        "type": "object", "additionalProperties": False,
                        "required": ["segment", "event_cards"],
                        "properties": {
                            "segment": schema["properties"]["segment"],
                            "event_cards": {
                                **cards_schema, "required": repair_keys,
                                "properties": {key: cards_schema["properties"][key] for key in repair_keys},
                            },
                        },
                    }
                    rejected = json.dumps({
                        "segment": segment_number,
                        "event_cards": {key: camera_draft["event_cards"].get(key) for key in repair_keys},
                    }, ensure_ascii=False)
                    repair_prompt += (
                        "\nOnly the selected events are editable. Other cards are already saved. "
                        "Do not write their actions into the selected cards. Return only the named "
                        "event_card keys; preserve their source action, camera and effects."
                    )
                raw = generate(
                    prompt=(
                        repair_prompt
                        + "\n\nPREVIOUS REJECTED SEGMENT JSON:\n"
                        + rejected
                        + "\n\nREPAIR ONLY THIS SEGMENT. Correct these violations:\n- "
                        + "\n- ".join(feedback)
                        + "\nReturn the complete event_cards object required by the schema. Maestro supplies the clock and ownership. "
                        "Correct the named cards and preserve the other cards. "
                        "Do not add, repeat, recap, or preview any event."
                    ),
                    system_prompt=camera_guide,
                    max_new_tokens=min(8192, segment_token_budget * 2),
                    temperature=0.08,
                    top_p=0.78,
                    image_paths=camera_images,
                    enable_thinking=False,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    json_schema=repair_schema,
                )
                segment = _parse_json_object(raw, allow_repair=False)
                if repair_keys:
                    patches = segment.get("event_cards") if isinstance(segment, dict) else None
                    if not isinstance(patches, dict) or any(key not in patches for key in repair_keys):
                        raise ValueError("camera repair omitted a requested event card")
                    # Ignore unrequested cards/metadata even if a remote
                    # writer does not honor the restricted output schema.
                    camera_draft["event_cards"].update({key: patches[key] for key in repair_keys})
                    segment = camera_draft
                segment = _canonicalize_segment_contract(
                    segment,
                    segment_number=segment_number,
                    duration=duration,
                    assigned_beats=beats,
                    dialogue_catalog=catalog,
                    opening_state=previous_closing,
                    source_intent=source_intent,
                    source_events=source_events,
                    use_camera_handoff=faithful_locked_schedule and segment_number < segment_count,
                )
                segment_errors = segment_violations(
                    prompt,
                    segment,
                    segment_number=segment_number,
                    duration=duration,
                    assigned_beats=beats,
                    dialogue_catalog=catalog,
                )
                # A focused repair may leave other, already-reviewed cards
                # untouched. Reuse their evidence only while it still matches.
                segment_errors = clear_confirmed_coverage_errors(
                    segment_errors, segment or {}, coverage_receipts,
                )
            if segment_errors or not segment:
                raise ValueError("; ".join(segment_errors or ["invalid segment JSON"]))
        except InterruptedError:
            raise
        except Exception as error:
            print(f"[MiniMax H3] Segment {segment_number} fallback: {error}")
            review_windows.add(segment_number)
            planned_by = "deterministic_fallback"
            planning_diagnostics.extend(
                f"Window {segment_number}: {item}"
                for item in (segment_errors or [sanitize_h3_prompt_text(error)])
                if item
            )
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
        if any(shot.get("dialogue") for shot in segment.get("shots", [])):
            from services.h3_dialogue_writing import fit_camera_dialogue

            camera_warnings = fit_camera_dialogue(
                prompt, segment, catalog, ledger.get("generated_dialogue") or [],
                generate=generate,
                system_prompt=_load_h3_planning_guide("minimax_h3_dialogue_copyedit", nsfw=nsfw),
            )
            planning_warnings.extend(camera_warnings)
            if camera_warnings:
                review_windows.add(segment_number)
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
            review_windows.add(segment_number)
            materialized = _repair_materialized_segment_staging(
                materialized,
                known_speakers=known_speakers,
                future_cast=future_cast,
            )
        materialized["active_cast"] = list(active_cast)
        # A segment cannot rewrite its supplied opening. The preceding
        # completed state is shared by frame-linked and editorial handoffs.
        materialized["opening_state"] = previous_closing
        if saved_segment:
            materialized["closing_state"] = saved_segment["closing_state"]
        segments.append(materialized)
        previous_closing = materialized["closing_state"]

    if segments and source_intent.get("global_instructions") and not action_first:
        ledger["required_final_outcome"] = segments[-1]["closing_state"]

    # Remember every principal introduced so far, but do not require them to be
    # visible at the outgoing boundary.  Forcing a group composition into the
    # final second changes the whole diffusion sample: it can restage furniture,
    # replay an entrance, or make the actor walk to a duplicate location.  The
    # next segment instead receives this list as identity/wardrobe context for
    # anyone who remains visible or returns from briefly off camera.  Future
    # entrants are still excluded until their first assigned segment.
    for index, segment in enumerate(segments):
        segment_number = index + 1
        if camera_coverage == "continuous":
            segment["coverage"] = "single continuous shot"
            for phase in (segment.get("shots") or [])[1:]:
                phase["transition"] = "continuous reframe"
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

    if allow_generated_dialogue:
        from services.h3_dialogue_writing import creative_dialogue_windows

        for audit in creative_dialogue_windows(
            prompt, ledger, locked_dialogue, durations, camera_segments=segments,
        ):
            if audit["problems"]:
                review_windows.add(int(audit["segment"]))
                planning_warnings.append(
                    f"AI dialogue needs review in window {audit['segment']}: "
                    + "; ".join(audit["problems"])
                    + ". Automatic writing repair was unsuccessful; review or enhance again before generating."
                )

    # Camera copyediting can shorten AI-authored lines. Keep that accepted
    # wording on the next repair as well as the materialized window prompts.
    checkpoint_context["catalog"] = deepcopy(catalog)
    checkpoint_context["ledger"] = deepcopy(ledger)
    return {
        "planned_by": planned_by,
        "camera_checkpoint": {
            "version": 1,
            "context": checkpoint_context,
            "segments": deepcopy(segments),
            "review_windows": sorted(review_windows),
        },
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
