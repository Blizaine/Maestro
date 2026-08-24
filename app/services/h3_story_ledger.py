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


H3_STORY_LEDGER_VERSION = 4

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
    r"overall_soundscape|non_diegetic_music)\s*:",
    flags=re.IGNORECASE,
)
_CONTEXT_IR_FIELD = re.compile(
    r"^[ \t]*(subject_definitions|summary|retention_analysis|"
    r"detailed_description|overall_soundscape|non_diegetic_music)\s*:\s*",
    flags=re.IGNORECASE | re.MULTILINE,
)
_SPEECH_VERB = re.compile(
    r"\b(?:says?|said|asks?|asked|replies?|replied|responds?|responded|"
    r"whispers?|whispered|shouts?|shouted|yells?|yelled|declares?|declared|"
    r"states?|stated|tells?|told|calls?\s+out|called\s+out)\b",
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
    "approach|arrive|attack|board|break|climb|cross|descend|dive|drop|"
    "enter|exit|fall|fight|fly|grab|hold|jump|laugh|launch|leap|mount|"
    "move|plummet|race|reach|ride|run|save|smash|sprint|stand|step|"
    "take|turn|walk|yell"
)
_FAST_ACTION_RE = re.compile(
    r"\b(?:high[- ]speed|high rate of speed|extreme(?:ly)? fast|rapid|"
    r"supersonic|breakneck|frantic|plummet|free[- ]?fall|dive|race|"
    r"hurtl|speeding|never stopping|non[- ]stop)\w*\b",
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


def extract_h3_source_intent(prompt: str) -> dict[str, Any]:
    """Extract immutable camera, pacing, style, and vocal requirements.

    These facts are application-owned. They must survive even when the local
    planning LLM returns malformed JSON or exhausts its response budget.
    """

    source = sanitize_h3_prompt_text(prompt)
    lowered = source.casefold()
    pov = bool(_POV_RE.search(source))
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
    names_source = re.sub(r'"[^"\r\n]*"|“[^”\r\n]*”', "", source)
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9_'’-]+(?:\s+[A-Z][A-Za-z0-9_'’-]+){1,3}\b",
        names_source,
    ):
        name = sanitize_h3_prompt_text(match.group(0))
        if name.casefold().split()[0] in {"the", "then", "extremely", "epic"}:
            continue
        if name not in proper_names:
            proper_names.append(name)

    style_fragments = [
        fragment.strip(" ,;:-.!?")
        for fragment in re.split(r"(?<=[.!?])\s+", source)
        if _is_style_only_fragment(fragment)
    ]
    camera_fragments = [
        fragment.strip(" ,;:-.!?")
        for fragment in re.split(r"(?<=[.!?])\s+", source)
        if _is_persistent_camera_directive(fragment)
    ]
    nonverbal = list(dict.fromkeys(
        match.group(0).casefold()
        for match in _NONVERBAL_VOCAL_RE.finditer(source)
    ))
    hands_visible = bool(
        re.search(r"\b(?:both|two)?\s*hands?\b", source, re.IGNORECASE)
        and re.search(r"\b(?:holding|gripping|grasping)\b", source, re.IGNORECASE)
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

    pacing = (
        "extremely fast real-time movement with sustained forward momentum, decisive choreography, and no slow motion"
        if _FAST_ACTION_RE.search(source)
        else "natural real-time pacing"
    )
    ambient_parts: list[str] = []
    if re.search(r"\bmountain|cliff|canyon|clouds?\b", lowered):
        ambient_parts.append("open-air mountain wind")
    if _FAST_ACTION_RE.search(source):
        ambient_parts.append("speed-dependent rushing air")
    return {
        "first_person_pov": pov,
        "pov_identity": pov_identity,
        "proper_names": proper_names,
        "fast_action": bool(_FAST_ACTION_RE.search(source)),
        "ongoing_motion": ongoing,
        "hands_visible": hands_visible,
        "perspective_contract": ". ".join(perspective_parts),
        "style_contract": ". ".join(style_fragments),
        "pacing_contract": pacing,
        "requested_nonverbal_vocals": (
            "Requested nonverbal vocalizations remain audible: " + ", ".join(nonverbal)
            if nonverbal else ""
        ),
        "ambient_contract": "; ".join(ambient_parts),
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

    source = str(value or "").strip()
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
        r"<d>\s*(?:\[[^\]]+\]\s*)?(.*?)\s*</d>",
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


def _infer_quote_speaker(source: str, quote_start: int) -> tuple[str, str]:
    prefix = source[max(0, quote_start - 220):quote_start]
    verbs = list(_SPEECH_VERB.finditer(prefix))
    if not verbs:
        return "Speaker", "speaks naturally"
    verb = verbs[-1]
    names = list(_PROPER_NAME.finditer(prefix[:verb.start()]))
    speaker = names[-1].group(0) if names else "Speaker"
    delivery = sanitize_h3_prompt_text(prefix[verb.end():]).strip(" ,;:-")
    if not delivery:
        delivery = "speaks naturally"
    elif len(delivery) > 100:
        delivery = delivery[-100:].lstrip(" ,;:-")
    return speaker, delivery


def extract_locked_dialogue(prompt: str) -> list[dict[str, Any]]:
    """Extract user-authored quoted lines before any LLM rewriting occurs."""

    source = str(prompt or "")
    pattern = re.compile(r'"([^"\r\n]{1,600})"|“([^”\r\n]{1,600})”')
    locked: list[dict[str, Any]] = []
    for match in pattern.finditer(source):
        text = sanitize_h3_prompt_text(match.group(1) or match.group(2) or "").strip()
        if not text or _PLACEHOLDER_DIALOGUE.fullmatch(text):
            continue
        speaker, delivery = _infer_quote_speaker(source, match.start())
        # A quoted title should not silently become dialogue. Speech cues,
        # screenplay-style ``Name:`` labels, and a quote-only prompt are the
        # three unambiguous forms accepted here.
        nearby = source[max(0, match.start() - 120):match.start()]
        label = re.search(
            r"([A-Z][A-Za-z0-9_'’-]*(?:\s+[A-Z][A-Za-z0-9_'’-]*){0,3})\s*:\s*$",
            nearby,
        )
        outside_quote = (source[:match.start()] + source[match.end():]).strip(" \t\r\n.,;:!?-")
        if not _SPEECH_VERB.search(nearby) and not label and outside_quote:
            continue
        if label:
            speaker = label.group(1)
        speech_context = source[max(0, match.start() - 240):match.start()]
        off_camera = bool(re.search(
            r"\b(?:off[- ]camera|offscreen|off[- ]screen|pov\s+(?:off[- ]camera\s+)?voice)\b",
            speech_context,
            flags=re.IGNORECASE,
        ))
        locked.append({
            "dialogue_id": f"D{len(locked) + 1}",
            "speaker": sanitize_h3_prompt_text(speaker) or "Speaker",
            "language": "English",
            "delivery": delivery,
            "text": text,
            "off_camera": off_camera,
            "source_offset": match.start(),
            "source_end": match.end(),
        })
    return locked


def _story_fragments(prompt: str) -> list[str]:
    # A user-authored line break is often the only boundary between two
    # actions in Studio's prompt box. Preserve it as sentence punctuation
    # before the general sanitizer collapses whitespace.
    source = sanitize_h3_prompt_text(
        re.sub(r"(?:\r?\n)+", ". ", str(prompt or ""))
    )
    # Preserve a sentence boundary where quoted dialogue was removed so a
    # following physical event cannot collapse into the preceding speech cue.
    for item in reversed(extract_locked_dialogue(source)):
        source = (
            source[: int(item["source_offset"])]
            + "."
            + source[int(item["source_end"]):]
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
        ))
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
            if re.fullmatch(
                r"(?:and\s+)?then|after\s+that|next",
                value,
                flags=re.IGNORECASE,
            ):
                continue
            if _is_style_only_fragment(value) or _is_persistent_camera_directive(value):
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
        compacted.append(value)
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
        r"mount|move|plummet|race|reach|ride|run|save|smash|sprint|stand|"
        r"step|take|turn|walk)(?:s|es|ed|ing)?\b",
        flags=re.IGNORECASE,
    )
    if non_speech_actions.search(before):
        return text

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


def _ledger_schema(segment_count: int, *, allow_generated_dialogue: bool) -> dict[str, Any]:
    """Schema for creative context layered over Maestro's fixed story schedule.

    Event IDs, beat grouping, dialogue ownership, and segment allocation are
    deliberately absent.  Those are deterministic compiler responsibilities;
    the LLM supplies continuity language and, only when requested, dialogue.
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
    if not allow_generated_dialogue:
        generated_dialogue["maxItems"] = 0
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
            "generated_dialogue": generated_dialogue,
        },
        "required": [
            "subject_continuity", "setting_continuity", "visual_continuity",
            "editing_style", "initial_state", "ambient_audio", "music",
            "required_final_outcome", "generated_dialogue",
        ],
        "additionalProperties": False,
    }


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
    speaker_ids: dict[str, str] = {}
    for item in catalog:
        speaker = sanitize_h3_prompt_text(item.get("speaker")) or "Speaker"
        key = speaker.casefold()
        speaker_ids.setdefault(key, f"S{len(speaker_ids) + 1}")
        item["speaker_id"] = speaker_ids[key]
    return catalog


def ledger_violations(
    prompt: str,
    ledger: dict[str, Any] | None,
    *,
    segment_count: int,
    locked_dialogue: list[dict[str, Any]],
    expect_dialogue: bool,
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
    normalized_descriptions = [
        _normalize_key(item.get("description")) for item in beats
        if (item.get("source_event_ids") or [])
        and _normalize_key(item.get("description"))
    ]
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

    generated = [
        item for item in (ledger.get("generated_dialogue") or [])
        if isinstance(item, dict)
    ]
    if locked_dialogue and generated:
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
        violations.append("dialogue IDs are missing, duplicated, reordered, or assigned to multiple beats")
    if referenced_ids and referenced_ids != all_ids:
        violations.append("dialogue order differs from the locked story order")
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
    if expect_dialogue and not all_ids:
        violations.append("the requested character interaction contains no dialogue")
    if generated and segment_durations:
        generated_words = {
            str(item.get("dialogue_id") or "").upper(): len(
                re.findall(r"\b[\w'’-]+\b", str(item.get("text") or ""))
            )
            for item in generated
        }
        beat_segment = {
            str(dialogue_id or "").upper(): int(beat.get("segment") or 0)
            for beat in beats
            for dialogue_id in (beat.get("dialogue_ids") or [])
        }
        for segment_number, duration in enumerate(segment_durations, start=1):
            word_count = sum(
                count for dialogue_id, count in generated_words.items()
                if beat_segment.get(dialogue_id) == segment_number
            )
            budget = max(1, int(math.floor(max(0.0, float(duration)) * 2.1)))
            if word_count > budget:
                violations.append(
                    f"segment {segment_number} generated dialogue uses {word_count} words; budget is {budget}"
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
    locked_dialogue: list[dict[str, Any]],
    camera_coverage: str,
    reference_context: str,
) -> dict[str, Any]:
    source_events = extract_source_events(prompt)
    fragments = [item["text"] for item in source_events]
    intent = extract_h3_source_intent(prompt)
    beats: list[dict[str, Any]] = []
    event_buckets: list[list[dict[str, str]]] = [[] for _ in range(segment_count)]
    for index, event in enumerate(source_events):
        # A single compound outcome belongs at the end; earlier segments can
        # then build toward it instead of performing it repeatedly. With two
        # or more events, anchor the first and last to the timeline ends.
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

    # Prefer a meaningful physical handoff over an arbitrary proportional
    # split. Mounting/boarding and launching over an edge belong with the
    # setup window when the following window owns the sustained journey.
    handoff_re = re.compile(
        r"\b(?:laugh|mount|board|climb\s+(?:onto|aboard)|take\s+off|launch|"
        r"leap|jump|plummet\s+over)\b",
        flags=re.IGNORECASE,
    )
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

    source_length = max(1, len(str(prompt or "")))
    expected_dialogue_events = _expected_dialogue_events(prompt, locked_dialogue)
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

    names = list(intent["proper_names"])
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

    visual_contract = ". ".join(part for part in (
        intent["perspective_contract"],
        intent["style_contract"],
        "Keep lighting, color, screen direction, and established geography coherent",
    ) if part)
    first_state_events = event_buckets[0][:2] or source_events[:1]
    initial_state = ". ".join(
        _filmable_source_event(item["text"]) for item in first_state_events
    )
    final_outcome = _filmable_source_event(fragments[-1])
    if intent["ongoing_motion"]:
        final_outcome = f"The requested motion remains active after {final_outcome}"
    # A quote whose offset landed in an otherwise unexpected segment remains
    # assigned exactly once. No model-authored line is needed in fallback.
    return {
        "subject_continuity": sanitize_h3_prompt_text(reference_context) or subject_continuity,
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


def _merge_story_context(
    canonical: dict[str, Any],
    candidate: dict[str, Any] | None,
    *,
    locked_dialogue: list[dict[str, Any]],
    segment_count: int,
) -> dict[str, Any]:
    """Layer creative context onto Maestro's immutable story schedule.

    The model is intentionally unable to change event ownership, beat order,
    segment allocation, or locked dialogue placement.  Older mocked responses
    and cached development payloads may still contain ``beats``; their state
    prose is accepted only when the complete schedule signature matches.
    """

    ledger = deepcopy(canonical)
    if not isinstance(candidate, dict):
        return ledger

    for field in _STORY_CONTEXT_FIELDS:
        value = sanitize_h3_prompt_text(candidate.get(field))
        if value:
            ledger[field] = value

    proposed_beats = [
        item for item in (candidate.get("beats") or [])
        if isinstance(item, dict)
    ]
    proposed_by_signature: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    proposed_dialogue_segments: dict[str, int] = {}
    for item in proposed_beats:
        try:
            segment = int(item.get("segment") or 0)
        except (TypeError, ValueError):
            segment = 0
        source_ids = tuple(
            str(event_id or "").upper()
            for event_id in (item.get("source_event_ids") or [])
        )
        if segment and source_ids:
            proposed_by_signature[(segment, source_ids)] = item
        for dialogue_id in (item.get("dialogue_ids") or []):
            proposed_dialogue_segments[str(dialogue_id or "").upper()] = segment

    for beat in ledger.get("beats") or []:
        signature = (
            int(beat.get("segment") or 0),
            tuple(
                str(event_id or "").upper()
                for event_id in (beat.get("source_event_ids") or [])
            ),
        )
        enrichment = proposed_by_signature.get(signature)
        if not enrichment:
            continue
        # Keep Maestro's exact event wording and schedule, but preserve useful
        # creative result/effect prose from a response that matched it exactly.
        for field in ("state_after", "sound_effects"):
            value = sanitize_h3_prompt_text(enrichment.get(field))
            if value:
                beat[field] = value

    # Locked user dialogue is already attached to its source event by the
    # deterministic compiler.  Only synthesize dialogue when the caller asked
    # for it and supplied no exact lines to preserve.
    ledger["generated_dialogue"] = []
    if locked_dialogue:
        return ledger

    raw_generated = [
        item for item in (candidate.get("generated_dialogue") or [])
        if isinstance(item, dict)
    ]
    for index, item in enumerate(raw_generated):
        legacy_id = str(item.get("dialogue_id") or f"D{index + 1}").upper()
        try:
            segment = int(
                item.get("segment")
                or proposed_dialogue_segments.get(legacy_id)
                or 0
            )
        except (TypeError, ValueError):
            segment = 0
        if segment < 1 or segment > segment_count:
            continue
        text = sanitize_h3_prompt_text(item.get("text"))
        if not text:
            continue
        dialogue_id = f"D{index + 1}"
        generated = {
            "dialogue_id": dialogue_id,
            "speaker": sanitize_h3_prompt_text(item.get("speaker")) or "Speaker",
            "language": sanitize_h3_prompt_text(item.get("language")) or "English",
            "delivery": sanitize_h3_prompt_text(item.get("delivery")) or "speaks naturally and clearly",
            "text": text,
        }
        ledger["generated_dialogue"].append(generated)
        segment_beats = [
            beat for beat in ledger.get("beats") or []
            if int(beat.get("segment") or 0) == segment
        ]
        if segment_beats:
            segment_beats[-1].setdefault("dialogue_ids", []).append(dialogue_id)
    return ledger


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


def _segment_schema(segment_number: int) -> dict[str, Any]:
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
                "maxItems": 4,
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
    raw_shots = [
        dict(item) for item in (segment.get("shots") or [])
        if isinstance(item, dict)
    ][:4]
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
        shot["action"] = sanitize_h3_prompt_text(shot.get("action"))
        shot["sound_effects"] = sanitize_h3_prompt_text(
            shot.get("sound_effects") or "Natural synchronized effects"
        )
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
    for shot, shot_beats in zip(raw_shots, assignments):
        beat_ids = [str(beat.get("beat_id") or "").upper() for beat in shot_beats]
        shot["beat_ids"] = beat_ids
        required = ". Then ".join(
            sanitize_h3_prompt_text(beat.get("description"))
            for beat in shot_beats
            if sanitize_h3_prompt_text(beat.get("description"))
        )
        # One LLM-authored camera shot may cover several canonical beats. A
        # loose token-overlap check could be satisfied by only the first beat,
        # leaving a later beat absent and causing the stricter per-beat
        # validator to reject an otherwise usable plan. Put Maestro's complete
        # ordered beat text first unless it is already present verbatim after
        # normalization; creative choreography remains as supplemental prose.
        if (
            required
            and _normalize_key(required) not in _normalize_key(shot.get("action"))
        ):
            shot["action"] = ". ".join(
                part for part in (required, shot["action"]) if part
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
        if source_intent.get("fast_action") else
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
    if not 1 <= len(shots) <= 4:
        violations.append(f"returned {len(shots)} shots instead of one to four")
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

    lowered_source = str(prompt or "").casefold()
    lowered_segment = json.dumps(segment, ensure_ascii=False).casefold()
    for pattern in UNREQUESTED_SPECTACLE_PATTERNS:
        match = re.search(pattern, lowered_segment, flags=re.IGNORECASE)
        if match and not re.search(pattern, lowered_source, flags=re.IGNORECASE):
            violations.append(f"invented unrequested power/effect: {match.group(0).strip()}")
            break
    if not sanitize_h3_prompt_text(segment.get("closing_state")):
        violations.append("closing state is empty")
    return list(dict.fromkeys(violations))


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
    shot_count = min(4, max(1, len(beats)))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shot_count)]
    for index, beat in enumerate(beats):
        buckets[min(shot_count - 1, index)].append(beat)
    dialogue_map = {
        str(item.get("dialogue_id") or "").upper(): item
        for item in (dialogue_catalog or [])
    }
    weights: list[float] = []
    for bucket in buckets:
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
            len(re.findall(r"\b[\w'’-]+\b", str(dialogue_map.get(dialogue_id, {}).get("text") or "")))
            for dialogue_id in dialogue_ids
        )
        spoken_time = dialogue_words / 2.35 + len(dialogue_ids) * 0.35
        weights.append(max(float(event_count), spoken_time, 1.0))
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


def _materialize_segment(
    segment: dict[str, Any],
    *,
    beats: list[dict[str, Any]],
    dialogue_catalog: list[dict[str, Any]],
    source_events: list[dict[str, str]],
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
    shots: list[dict[str, Any]] = []
    for shot in segment.get("shots") or []:
        dialogue: list[dict[str, Any]] = []
        for performance in shot.get("dialogue") or []:
            dialogue_id = str(performance.get("dialogue_id") or "").upper()
            source = dialogue_map[dialogue_id]
            speaker = sanitize_h3_prompt_text(source.get("speaker")) or "Speaker"
            dialogue.append({
                "speaker": speaker,
                "speaker_id": sanitize_h3_prompt_text(source.get("speaker_id")) or "S1",
                "language": sanitize_h3_prompt_text(source.get("language")) or "English",
                "delivery": (
                    sanitize_h3_prompt_text(performance.get("delivery"))
                    or sanitize_h3_prompt_text(source.get("delivery"))
                    or "speaks naturally"
                ),
                "action": sanitize_h3_prompt_text(performance.get("action")),
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
        shots.append({
            "shot": int(shot.get("shot") or len(shots) + 1),
            "start_seconds": float(shot.get("start_seconds") or 0.0),
            "end_seconds": float(shot.get("end_seconds") or 0.0),
            "transition": sanitize_h3_prompt_text(shot.get("transition")),
            "framing": sanitize_h3_prompt_text(shot.get("framing")),
            "camera": sanitize_h3_prompt_text(shot.get("camera")),
            "action": proposed_action,
            "dialogue": dialogue,
            "sound_effects": sanitize_h3_prompt_text(shot.get("sound_effects")),
        })
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


def plan_h3_story_segments(
    prompt: str,
    *,
    segment_durations: list[float],
    mode: str,
    camera_coverage: str,
    reference_context: str = "",
    expect_dialogue: bool = False,
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
    source_events = extract_source_events(prompt)
    source_intent = extract_h3_source_intent(prompt)
    planning_warnings: list[str] = []
    allow_generated_dialogue = bool(expect_dialogue and not locked_dialogue)
    canonical_ledger = _deterministic_ledger(
        prompt,
        segment_count=segment_count,
        locked_dialogue=locked_dialogue,
        camera_coverage=camera_coverage,
        reference_context=reference_context,
    )
    dialogue_lines = "\n".join(
        f"- {item['dialogue_id']}: speaker={item['speaker']}; exact text={json.dumps(item['text'], ensure_ascii=False)}"
        for item in locked_dialogue
    ) or "- None."
    geometry_lines = "\n".join(
        f"- Segment {index + 1}: {duration:.3f} seconds; generated-dialogue budget "
        f"at most {max(1, int(math.floor(duration * 2.1)))} spoken words"
        for index, duration in enumerate(durations)
    )
    schedule_lines = "\n".join(
        "- Segment {segment}, {beat_id}: source events {events}; locked dialogue {dialogue}; {description}".format(
            segment=int(item.get("segment") or 0),
            beat_id=item.get("beat_id"),
            events=", ".join(item.get("source_event_ids") or []) or "connective beat",
            dialogue=", ".join(item.get("dialogue_ids") or []) or "none",
            description=item.get("description"),
        )
        for item in canonical_ledger.get("beats") or []
    )
    ledger_prompt = (
        f"Mode: {mode}. Camera coverage preference: {camera_coverage}.\n"
        f"Segment geometry:\n{geometry_lines}\n\n"
        f"Canonical reference context:\n{reference_context or 'No external reference map.'}\n\n"
        "MAESTRO'S LOCKED STORY SCHEDULE (read-only; do not reproduce or reschedule it in JSON):\n"
        f"{schedule_lines}\n\n"
        f"LOCKED USER DIALOGUE (context only; never rewrite or reproduce it in JSON):\n{dialogue_lines}\n\n"
        f"Dialogue policy: {'Write concise generated_dialogue entries and select the segment where each line naturally occurs.' if allow_generated_dialogue else 'Do not add generated dialogue.'}\n\n"
        "Return continuity, setting, visual-language, opening-state, ambience, music, and final-outcome context. "
        "Maestro—not you—owns event IDs, beat grouping, segment allocation, and locked dialogue placement.\n\n"
        f"User concept:\n{prompt}"
    )
    ledger_guide = load_guide("enhance", "minimax_h3_story_ledger")
    if nsfw:
        ledger_guide += (
            "\n\nMATURE-MODE FIDELITY\nPreserve explicitly requested mature material. "
            "Do not censor it, add to it, or intensify it."
        )
    ledger_schema = _ledger_schema(
        segment_count,
        allow_generated_dialogue=allow_generated_dialogue,
    )
    planned_by = "llm"
    ledger: dict[str, Any] | None = None
    violations: list[str] = []
    raw = ""
    try:
        raw = generate(
            prompt=ledger_prompt,
            system_prompt=ledger_guide,
            max_new_tokens=min(2200, max(1000, 650 + segment_count * 170)),
            temperature=0.22,
            top_p=0.84,
            image_paths=image_paths or None,
            enable_thinking=False,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            json_schema=ledger_schema,
        )
        from services.h3_window_planner import _parse_json_object

        candidate = _parse_json_object(raw)
        ledger = _merge_story_context(
            canonical_ledger,
            candidate,
            locked_dialogue=locked_dialogue,
            segment_count=segment_count,
        )
        violations = ledger_violations(
            prompt,
            ledger,
            segment_count=segment_count,
            locked_dialogue=locked_dialogue,
            expect_dialogue=expect_dialogue,
            segment_durations=durations,
        )
        if violations:
            print("[MiniMax H3] Story-context repair: " + "; ".join(violations))
            raw = generate(
                prompt=(
                    ledger_prompt
                    + "\n\nPREVIOUS REJECTED CONTEXT JSON:\n"
                    + json.dumps(candidate, ensure_ascii=False, indent=2)
                    + "\n\nREPAIR ONLY THE CREATIVE CONTEXT. Correct these violations:\n- "
                    + "\n- ".join(violations)
                    + "\nDo not return or alter story beats, event IDs, or locked dialogue IDs. "
                    "Maestro already owns the complete schedule."
                ),
                system_prompt=ledger_guide,
                max_new_tokens=min(2200, max(1000, 650 + segment_count * 170)),
                temperature=0.08,
                top_p=0.78,
                image_paths=image_paths or None,
                enable_thinking=False,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                json_schema=ledger_schema,
            )
            candidate = _parse_json_object(raw)
            ledger = _merge_story_context(
                canonical_ledger,
                candidate,
                locked_dialogue=locked_dialogue,
                segment_count=segment_count,
            )
            violations = ledger_violations(
                prompt,
                ledger,
                segment_count=segment_count,
                locked_dialogue=locked_dialogue,
                expect_dialogue=expect_dialogue,
                segment_durations=durations,
            )
        if violations or not ledger:
            raise ValueError("; ".join(violations or ["invalid story context JSON"]))
    except Exception as error:
        print(f"[MiniMax H3] Story-context fallback: {error}")
        planned_by = "deterministic_fallback"
        planning_warnings.append(
            "The AI continuity pass did not satisfy Maestro's fidelity checks. "
            "The ordered story and exact dialogue remain intact, and Maestro used its compiled continuity context."
        )
        ledger = deepcopy(canonical_ledger)

    # Camera perspective, requested speed, style, nonverbal reactions, and
    # sequence shape are immutable even when the creative ledger succeeds.
    # Merge them after schema validation so the small LLM never owns them.
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
    if catalog:
        stable_speakers: list[tuple[str, str]] = []
        seen_speaker_ids: set[str] = set()
        for item in catalog:
            speaker_id = str(item.get("speaker_id") or "")
            if speaker_id in seen_speaker_ids:
                continue
            seen_speaker_ids.add(speaker_id)
            stable_speakers.append((speaker_id, sanitize_h3_prompt_text(item.get("speaker"))))
        speaker_map = "; ".join(
            f"{speaker_id} is {speaker}"
            for speaker_id, speaker in stable_speakers
        )
        subjects = sanitize_h3_prompt_text(ledger.get("subject_continuity"))
        if speaker_map and speaker_map.casefold() not in subjects.casefold():
            ledger["subject_continuity"] = (
                f"Stable speaking identities: {speaker_map}. {subjects}"
                if subjects else f"Stable speaking identities: {speaker_map}"
            )
    segment_guide = load_guide("enhance", "minimax_h3_story_segment")
    if nsfw:
        segment_guide += (
            "\n\nMATURE-MODE FIDELITY\nPreserve explicitly requested mature material. "
            "Do not censor it, add to it, or intensify it."
        )
    segments: list[dict[str, Any]] = []
    previous_closing = sanitize_h3_prompt_text(ledger.get("initial_state"))
    for index, duration in enumerate(durations):
        segment_number = index + 1
        beats = [
            item for item in ledger.get("beats") or []
            if isinstance(item, dict) and int(item.get("segment") or 0) == segment_number
        ]
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
        if mode == "sliding_window":
            mode_instruction = (
                "This is a frame-linked continuation. Its opening must exactly match the supplied previous frame/state. "
                "Do not restart or recap. Internal motivated cuts are allowed, but the segment boundary itself is not a story cut."
            )
        elif mode == "reference_sequence_continuation":
            mode_instruction = (
                "This is a native Ref2VA motion-and-audio overlap continuation. The canonical references remain identity guidance, "
                "not opening keyframes. Continue from the supplied previous state without restarting, restaging, or replaying an action. "
                "Internal motivated cuts are allowed, but the window boundary itself is not a story cut."
            )
        else:
            mode_instruction = (
                "This is an independently generated editorial clip. Restate a complete readable opening composition, "
                "use the canonical references for identity, and advance only this clip's assigned beats."
            )
        creative_beats = [
            {
                "event": sanitize_h3_prompt_text(beat.get("description")),
                "resulting_state": sanitize_h3_prompt_text(beat.get("state_after")),
                "sound_effects": sanitize_h3_prompt_text(beat.get("sound_effects")),
            }
            for beat in beats
        ]
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
            f"Shared setting: {ledger.get('setting_continuity')}\n"
            f"Shared visual language: {ledger.get('visual_continuity')}\n"
            f"Editing style: {ledger.get('editing_style')}\n"
            f"Required opening state: {previous_closing}\n"
            f"Immutable chronological events (depict each once, in order):\n"
            f"{json.dumps(creative_beats, ensure_ascii=False, indent=2)}\n\n"
            f"Immutable dialogue performances (plan visible/off-camera performance but do not reproduce text in JSON):\n"
            f"{json.dumps(creative_dialogue, ensure_ascii=False, indent=2)}\n\n"
            f"Original user concept for fidelity only:\n{prompt}"
        )
        schema = _segment_schema(segment_number)
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
        )
        # Story state belongs to the ledger, not the camera expander. This also
        # makes FL2VA's next-frame continuation and Omni's editorial handoff
        # deterministic even if the segment LLM paraphrases opening_state.
        materialized["opening_state"] = previous_closing
        segments.append(materialized)
        previous_closing = materialized["closing_state"]

    return {
        "planned_by": planned_by,
        "planning_warnings": list(dict.fromkeys(planning_warnings)),
        "source_intent": source_intent,
        "ledger": ledger,
        "locked_dialogue": [
            {
                key: value for key, value in item.items()
                if key not in {"source_offset", "source_end"}
            }
            for item in locked_dialogue
        ],
        "segments": segments,
    }
