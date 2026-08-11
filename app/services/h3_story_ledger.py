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
import json
import math
import re
from typing import Any, Callable


H3_STORY_LEDGER_VERSION = 2

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
        locked.append({
            "dialogue_id": f"D{len(locked) + 1}",
            "speaker": sanitize_h3_prompt_text(speaker) or "Speaker",
            "language": "English",
            "delivery": delivery,
            "text": text,
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
    for piece in pieces:
        value = piece.strip(" ,;:-.!?")
        if not value:
            continue
        if re.fullmatch(
            r"(?:and\s+)?then|after\s+that|next",
            value,
            flags=re.IGNORECASE,
        ):
            continue
        if re.fullmatch(
            r"(?:dark\s+)?cinematic|(?:[A-Za-z]+verse|[A-Za-z]+)\s+style|"
            r"rated[- ]?[rpg0-9+]+(?:\s+graphic)?(?:,\s*realistic(?:\s+film\s+scene)?)?|"
            r"realistic(?:\s+film\s+scene)?|"
            r"high[- ]speed\s+(?:action\s+)?(?:movie\s+)?(?:dynamic\s+)?(?:superhero\s+)?(?:fight\s+)?scenes?",
            value,
            flags=re.IGNORECASE,
        ):
            continue
        fragments.append(value)
    return fragments or ["Establish and carry out the requested scene"]


def extract_source_events(prompt: str) -> list[dict[str, str]]:
    """Create immutable source-event IDs used to prove coverage exactly once."""

    return [
        {"event_id": f"E{index + 1}", "text": fragment}
        for index, fragment in enumerate(_story_fragments(prompt))
    ]


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
    dialogue = {
        "type": "object",
        "properties": {
            "dialogue_id": {"type": "string"},
            "speaker": {"type": "string"},
            "language": {"type": "string"},
            "delivery": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["dialogue_id", "speaker", "language", "delivery", "text"],
        "additionalProperties": False,
    }
    beat = {
        "type": "object",
        "properties": {
            "beat_id": {"type": "string"},
            "segment": {"type": "integer"},
            "description": {"type": "string"},
            "source_event_ids": {"type": "array", "items": {"type": "string"}},
            "dialogue_ids": {"type": "array", "items": {"type": "string"}},
            "state_after": {"type": "string"},
            "sound_effects": {"type": "string"},
        },
        "required": [
            "beat_id", "segment", "description", "source_event_ids", "dialogue_ids",
            "state_after", "sound_effects",
        ],
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
            "beats": {
                "type": "array",
                "minItems": segment_count,
                "maxItems": segment_count * 3,
                "items": beat,
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
        if _normalize_key(item.get("description"))
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
    source_length = max(1, len(str(prompt or "")))
    expected_dialogue_events = _expected_dialogue_events(prompt, locked_dialogue)
    event_segments = {
        event["event_id"]: segment_index + 1
        for segment_index, bucket in enumerate(event_buckets)
        for event in bucket
    }
    dialogue_by_segment: dict[int, list[str]] = {}
    for item in locked_dialogue:
        dialogue_id = item["dialogue_id"]
        expected_event = expected_dialogue_events.get(dialogue_id)
        segment = event_segments.get(expected_event or "")
        if segment is None:
            ratio = min(1.0, max(0.0, int(item.get("source_offset") or 0) / source_length))
            segment = 1 + min(segment_count - 1, int(math.floor(ratio * segment_count)))
        dialogue_by_segment.setdefault(segment, []).append(dialogue_id)
    for index in range(segment_count):
        assigned_events = event_buckets[index]
        fragment = " Then ".join(item["text"] for item in assigned_events)
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
        if fragment and index == 0:
            description = f"Establish the requested scene and begin this event: {fragment}"
        elif fragment and index + 1 == segment_count:
            description = f"Complete the requested story and final outcome: {fragment}"
        elif fragment:
            description = f"Advance the requested story without repeating or finishing early: {fragment}"
        elif next_event and previous_event:
            description = (
                "Show a new intermediate progression from the preceding beat "
                "toward the next requested event, without replaying or "
                f"completing either endpoint: {next_event}"
            )
        elif next_event:
            phase = "Begin" if index == 0 else "Continue"
            description = (
                f"{phase} a new visible buildup toward the upcoming requested "
                f"event without completing it yet: {next_event}"
            )
        elif previous_event:
            description = (
                "Show new physical consequences of the preceding requested "
                f"event without replaying it: {previous_event}"
            )
        else:
            description = "Advance to a new visible story state without replaying an earlier action"

        if index + 1 == segment_count:
            state_after = (
                "The final frame clearly shows the completed requested outcome: "
                f"{fragment or fragments[-1]}"
            )
        elif fragment:
            state_after = (
                "The final frame shows the immediate visible result of this beat: "
                f"{fragment}"
            )
        elif next_event:
            state_after = (
                "The final frame shows a new intermediate state leading toward, "
                f"but not yet completing: {next_event}"
            )
        else:
            state_after = (
                "The final frame shows new consequences after the preceding beat: "
                f"{previous_event}"
            )
        beats.append({
            "beat_id": f"B{index + 1}",
            "segment": index + 1,
            "description": description,
            "source_event_ids": [item["event_id"] for item in assigned_events],
            "dialogue_ids": dialogue_by_segment.get(index + 1, []),
            "state_after": state_after,
            "sound_effects": "Natural synchronized effects for the visible action",
        })
    # A quote whose offset landed in an otherwise unexpected segment remains
    # assigned exactly once. No model-authored line is needed in fallback.
    return {
        "subject_continuity": (
            sanitize_h3_prompt_text(reference_context)
            or "Keep every requested subject's identity, appearance, wardrobe, and carried objects unchanged"
        ),
        "setting_continuity": "Keep the requested location, geography, time of day, and background elements coherent",
        "visual_continuity": "Keep lighting, color, screen direction, and established geography coherent",
        "editing_style": (
            "Motivated cinematic cuts and dynamic camera coverage"
            if camera_coverage == "multi_shot"
            else "A motivated cinematic camera follows the requested action"
        ),
        "initial_state": "The requested scene begins in a clear readable composition",
        "ambient_audio": "Continuous natural nonverbal ambience appropriate to the requested location",
        "music": "N/A",
        "required_final_outcome": fragments[-1],
        "beats": beats,
        "generated_dialogue": [],
    }


def _lock_ledger_source_events(prompt: str, ledger: dict[str, Any]) -> None:
    """Replace LLM paraphrases with the user's immutable event wording."""

    source_events = extract_source_events(prompt)
    event_map = {item["event_id"]: item["text"] for item in source_events}
    for beat in ledger.get("beats") or []:
        if not isinstance(beat, dict):
            continue
        exact = [
            event_map[str(event_id or "").upper()]
            for event_id in (beat.get("source_event_ids") or [])
            if str(event_id or "").upper() in event_map
        ]
        if exact:
            beat["description"] = " Then ".join(exact)
    if source_events:
        ledger["required_final_outcome"] = source_events[-1]["text"]


def _segment_schema(segment_number: int) -> dict[str, Any]:
    performance = {
        "type": "object",
        "properties": {
            "dialogue_id": {"type": "string"},
            "delivery": {"type": "string"},
            "action": {"type": "string"},
        },
        "required": ["dialogue_id", "delivery", "action"],
        "additionalProperties": False,
    }
    shot = {
        "type": "object",
        "properties": {
            "shot": {"type": "integer"},
            "start_seconds": {"type": "number"},
            "end_seconds": {"type": "number"},
            "transition": {"type": "string"},
            "framing": {"type": "string"},
            "camera": {"type": "string"},
            "beat_ids": {"type": "array", "items": {"type": "string"}},
            "action": {"type": "string"},
            "dialogue": {"type": "array", "items": performance},
            "sound_effects": {"type": "string"},
        },
        "required": [
            "shot", "start_seconds", "end_seconds", "transition", "framing",
            "camera", "beat_ids", "action", "dialogue", "sound_effects",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "segment": {"type": "integer", "minimum": segment_number, "maximum": segment_number},
            "title": {"type": "string"},
            "opening_state": {"type": "string"},
            "coverage": {"type": "string"},
            "pacing": {"type": "string"},
            "shots": {"type": "array", "minItems": 1, "maxItems": 4, "items": shot},
            "closing_state": {"type": "string"},
        },
        "required": [
            "segment", "title", "opening_state", "coverage", "pacing",
            "shots", "closing_state",
        ],
        "additionalProperties": False,
    }


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
) -> dict[str, Any]:
    shot_count = min(4, max(1, len(beats)))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shot_count)]
    for index, beat in enumerate(beats):
        buckets[min(shot_count - 1, index)].append(beat)
    shots: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        start = duration * index / shot_count
        end = duration * (index + 1) / shot_count
        beat_ids = [str(item.get("beat_id") or "") for item in bucket]
        dialogue_ids = [
            str(dialogue_id or "")
            for beat in bucket
            for dialogue_id in (beat.get("dialogue_ids") or [])
        ]
        action = ". ".join(
            sanitize_h3_prompt_text(item.get("description")) for item in bucket
        )
        shots.append({
            "shot": index + 1,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "transition": "opening composition" if index == 0 else "hard cut",
            "framing": (
                "a readable wide or medium-wide establishing view"
                if index == 0 else "a motivated medium or close reaction angle"
            ),
            "camera": (
                "a dynamic camera follows the visible action in real time"
                if camera_coverage == "multi_shot"
                else "a coherent motivated camera follows the visible action"
            ),
            "beat_ids": beat_ids,
            "action": action or "The requested event advances visibly",
            "dialogue": [
                {
                    "dialogue_id": dialogue_id,
                    "delivery": "speaks naturally and clearly",
                    "action": "maintaining the visible performance",
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
            "dynamic multi-shot cinematic coverage"
            if camera_coverage == "multi_shot" else "coherent cinematic coverage"
        ),
        "pacing": "natural real-time pacing; no slow motion unless requested",
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
    event_map = {item["event_id"]: item["text"] for item in source_events}
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
    allow_generated_dialogue = bool(expect_dialogue and not locked_dialogue)
    dialogue_lines = "\n".join(
        f"- {item['dialogue_id']}: speaker={item['speaker']}; exact text={json.dumps(item['text'], ensure_ascii=False)}"
        for item in locked_dialogue
    ) or "- None."
    geometry_lines = "\n".join(
        f"- Segment {index + 1}: {duration:.3f} seconds; generated-dialogue budget "
        f"at most {max(1, int(math.floor(duration * 2.1)))} spoken words"
        for index, duration in enumerate(durations)
    )
    event_lines = "\n".join(
        f"- {item['event_id']}: {item['text']}"
        for item in source_events
    )
    ledger_prompt = (
        f"Mode: {mode}. Camera coverage preference: {camera_coverage}.\n"
        f"Segment geometry:\n{geometry_lines}\n\n"
        f"Canonical reference context:\n{reference_context or 'No external reference map.'}\n\n"
        f"LOCKED SOURCE EVENTS (assign every E-ID exactly once and in order; never omit one):\n{event_lines}\n\n"
        f"LOCKED USER DIALOGUE (assign IDs to beats; never rewrite or reproduce its text in JSON):\n{dialogue_lines}\n\n"
        f"Dialogue policy: {'Write concise generated_dialogue entries because an interaction needs speech.' if allow_generated_dialogue else 'Do not add generated dialogue.'}\n\n"
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
    try:
        raw = generate(
            prompt=ledger_prompt,
            system_prompt=ledger_guide,
            max_new_tokens=min(3200, max(1400, 800 + segment_count * 260)),
            temperature=0.22,
            top_p=0.84,
            image_paths=image_paths or None,
            enable_thinking=False,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            json_schema=ledger_schema,
        )
        from services.h3_window_planner import _parse_json_object

        ledger = _parse_json_object(raw)
        violations = ledger_violations(
            prompt,
            ledger,
            segment_count=segment_count,
            locked_dialogue=locked_dialogue,
            expect_dialogue=expect_dialogue,
            segment_durations=durations,
        )
        if violations:
            print("[MiniMax H3] Story-ledger repair: " + "; ".join(violations))
            raw = generate(
                prompt=(
                    ledger_prompt
                    + "\n\nREPAIR ONLY THE STORY LEDGER. Correct these violations:\n- "
                    + "\n- ".join(violations)
                    + "\nEvery beat must occur once, every segment must own at least one beat, "
                    "the final outcome belongs only to the final segment, and locked dialogue IDs "
                    "must appear exactly once in order."
                ),
                system_prompt=ledger_guide,
                max_new_tokens=min(3200, max(1400, 800 + segment_count * 260)),
                temperature=0.08,
                top_p=0.78,
                image_paths=image_paths or None,
                enable_thinking=False,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                json_schema=ledger_schema,
            )
            ledger = _parse_json_object(raw)
            violations = ledger_violations(
                prompt,
                ledger,
                segment_count=segment_count,
                locked_dialogue=locked_dialogue,
                expect_dialogue=expect_dialogue,
                segment_durations=durations,
            )
        if violations or not ledger:
            raise ValueError("; ".join(violations or ["invalid story ledger JSON"]))
    except Exception as error:
        print(f"[MiniMax H3] Story-ledger fallback: {error}")
        planned_by = "deterministic_fallback"
        ledger = _deterministic_ledger(
            prompt,
            segment_count=segment_count,
            locked_dialogue=locked_dialogue,
            camera_coverage=camera_coverage,
            reference_context=reference_context,
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
        segment_prompt = (
            f"Segment {segment_number} of {segment_count}; local duration 0.000 to {duration:.3f} seconds.\n"
            f"{mode_instruction}\n\n"
            f"Shared subjects: {ledger.get('subject_continuity')}\n"
            f"Shared setting: {ledger.get('setting_continuity')}\n"
            f"Shared visual language: {ledger.get('visual_continuity')}\n"
            f"Editing style: {ledger.get('editing_style')}\n"
            f"Required opening state: {previous_closing}\n"
            f"Assigned beats (use each beat_id exactly once):\n{json.dumps(beats, ensure_ascii=False, indent=2)}\n\n"
            f"Assigned dialogue (use each dialogue_id exactly once; do not copy text into action):\n"
            f"{json.dumps(assigned_dialogue, ensure_ascii=False, indent=2)}\n\n"
            f"Original user concept for fidelity only:\n{prompt}"
        )
        schema = _segment_schema(segment_number)
        segment: dict[str, Any] | None = None
        segment_errors: list[str] = []
        try:
            raw = generate(
                prompt=segment_prompt,
                system_prompt=segment_guide,
                max_new_tokens=1350,
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
                        + "\n\nREPAIR ONLY THIS SEGMENT. Correct these violations:\n- "
                        + "\n- ".join(segment_errors)
                        + "\nReturn one complete local shot clock. Use each assigned beat ID and dialogue ID exactly once. "
                        "Do not add, repeat, recap, or preview any event."
                    ),
                    system_prompt=segment_guide,
                    max_new_tokens=1350,
                    temperature=0.08,
                    top_p=0.78,
                    image_paths=None,
                    enable_thinking=False,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    json_schema=schema,
                )
                segment = _parse_json_object(raw)
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
            segment = _fallback_segment(
                segment_number,
                duration=duration,
                beats=beats,
                opening_state=previous_closing,
                camera_coverage=camera_coverage,
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
