"""Deterministic MiniMax H3 dialogue compilation and validation.

H3 treats ``<d>[Language] ...</d>`` blocks as an audio-generation contract.
These helpers keep that contract out of best-effort prompt rewriting: structured
Director dialogue is canonical, existing blocks are replaced as whole units,
and malformed or contradictory prompts are rejected before GPU generation.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from services.dialogue_timing import (
    DIALOGUE_MAX_WORDS_PER_SECOND,
    h3_dialogue_schedule,
)
from services.h3_prompt_budget import (
    H3_ENHANCED_TEXT_TOKEN_TARGET as _H3_DIRECTOR_TEXT_TOKEN_BUDGET,
    fit_h3_base_prompt,
    h3_prompt_token_count,
)
from services.text_integrity import repair_text
from .physical_continuity import identity_only_description, normalize_physical_state


class H3DialogueContractError(ValueError):
    """Raised when an H3 prompt cannot be made safe for native speech."""


_H3_DIALOGUE_TOKEN_RE = re.compile(r"<\s*(/?)\s*d\s*>", re.IGNORECASE)
_H3_STRICT_DIALOGUE_RE = re.compile(
    r"<d>\s*\[([^\]\r\n]+)\]\s+(.+?)\s*</d>",
    re.IGNORECASE | re.DOTALL,
)
_H3_SPOKEN_INSTRUCTION_RE = re.compile(
    r"\b(?:says?|said|speaks?|spoke|speaking|talks?|talking|tells?|told|"
    r"shouts?|shouting|whispers?|whispering|calls?|calling|recites?|"
    r"reciting|utters?|uttering|sings?|sang|singing|lip[ -]?syncs?)\b",
    re.IGNORECASE,
)
_H3_TEXTUAL_SPEECH_SUBJECT_RE = re.compile(
    r"\b(?:sign|poster|label|banner|screen|subtitle|caption|placard|"
    r"note|letter|message|card|word|title|speech\s+bubble|bubble|"
    r"display|text)\s+(?:that\s+)?$",
    re.IGNORECASE,
)
_H3_MUSIC_SUBJECT_CONTINUATION_RE = re.compile(
    r"^(?:(?:the|a|an)\s+)?(?:(?:non[ -]?diegetic|background|"
    r"orchestral|cinematic|instrumental|hopeful|heroic|soaring|gentle|"
    r"peaceful|sustained|uplifting|symphonic|dramatic)\s+)*(?:music|"
    r"score|soundtrack|orchestra|strings?|violins?|violas?|cellos?|"
    r"brass|horns?|piano|guitar|percussion|instruments?|melodies?|"
    r"themes?|chords?|notes?|symphonies|songs?)\b",
    re.IGNORECASE,
)
_H3_MUSIC_FOLLOWTHROUGH_RE = re.compile(
    r"^(?:then\s+)?(?:is|was|becomes?|became|swells?|builds?|rises?|"
    r"crescendo(?:es)?|resolves?|fades?|plays?|underscores?|accompanies|"
    r"continues?|starts?|begins?|ends?|sustains?|opens?|climaxes?)\b",
    re.IGNORECASE,
)
_H3_INDEPENDENT_CLAUSE_START_RE = re.compile(
    r"^(?:(?:the|a|an)\s+)?[A-Za-z][\w’'-]*(?:\s+[A-Za-z][\w’'-]*){0,2}"
    r"\s+[A-Za-z][\w’'-]+\b",
    re.IGNORECASE,
)
_H3_MUSIC_INSTRUCTION_RE = re.compile(
    r"\b(?P<subject>(?:(?:non[ -]?diegetic|background|orchestral|cinematic|"
    r"instrumental|hopeful|heroic|soaring|gentle|peaceful|sustained|"
    r"uplifting|symphonic|dramatic)\s+)*(?:music|score|soundtrack))\b"
    r"\s+(?:(?:slowly|gently|gradually|softly|quietly|briskly|"
    r"steadily|then)\s+){0,3}"
    r"(?P<verb>is|was|becomes|became|swells?|builds?|rises?|crescendo(?:es)?|"
    r"resolves?|fades?|plays?|underscores?|accompanies|continues?|starts?|"
    r"begins?|ends?|sustains?|opens?|climaxes?)\b",
    re.IGNORECASE,
)
_H3_MUSIC_DIRECTIVE_RE = re.compile(
    r"\b(?:add|compose|generate|include|introduce|play|write|create)\s+"
    r"(?:a|an|the|some)?\s*(?:non[ -]?diegetic\s+|background\s+|"
    r"orchestral\s+|cinematic\s+|instrumental\s+)*(?:music|score|soundtrack)\b",
    re.IGNORECASE,
)
_H3_VOCAL_SECTION_RE = re.compile(
    r"\s*(?:DIALOGUE AND VOCAL PERFORMANCE|"
    r"SILENCE AND VOCAL PERFORMANCE)\s*:.*?"
    r"(?=\b(?:FINAL BLOCKING|overall_soundscape|"
    r"non_diegetic_music)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_H3_SOUND_BOUNDARY_RE = re.compile(
    r"\b(?:overall_soundscape|non_diegetic_music)\s*:",
    re.IGNORECASE,
)
_H3_SPEECHLIKE_AMBIENCE_REPLACEMENTS = (
    (
        re.compile(
            r"\b(?:(?:coffee\s*shop|cafe|restaurant|office|party|crowd|"
            r"background)\s+)?chatter\b",
            re.IGNORECASE,
        ),
        "nonverbal room tone",
    ),
    (
        re.compile(
            r"\b(?:indistinct|distant|background|murmuring|soft|low)\s+"
            r"(?:voices?|conversations?|talking|speech)\b",
            re.IGNORECASE,
        ),
        "nonverbal room tone",
    ),
)

_H3_BASE_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_H3_REF2VA_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_H3_ALL_FIELDS = tuple(dict.fromkeys((*_H3_REF2VA_FIELDS, *_H3_BASE_FIELDS)))
_H3_CUSTOM_SECTION_RE = re.compile(
    r"\s+(?:OPENING CONTINUITY|FINAL BLOCKING|"
    r"DIALOGUE AND VOCAL PERFORMANCE|SILENCE AND VOCAL PERFORMANCE)\s*:.*?"
    r"(?=\s+(?:OPENING CONTINUITY|FINAL BLOCKING|"
    r"DIALOGUE AND VOCAL PERFORMANCE|SILENCE AND VOCAL PERFORMANCE)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_H3_SOUND_PROSE_RE = re.compile(
    r"\b(?:(?:the|overall)\s+)?(?:ambient\s+)?soundscape\s+(?:is\s+dominated\s+by|includes|contains|features|is)\s+"
    r"(.+?)(?=\s+(?:non-diegetic\s+music|the\s+final\s+beat|"
    r"closing\s+blocking|opening\s+continuity|final\s+blocking)\b|\s*\[Shot\s+\d+\]|(?<=[.!?])\s+|$)",
    re.IGNORECASE | re.DOTALL,
)
_H3_MUSIC_PROSE_RE = re.compile(
    r"\bnon-diegetic\s+music\s+is\s+(.+?)"
    r"(?=\s+(?:the\s+final\s+beat|closing\s+blocking|"
    r"opening\s+continuity|final\s+blocking)\b|\s*\[Shot\s+\d+\]|(?<=[.!?])\s+|$)",
    re.IGNORECASE | re.DOTALL,
)
_H3_SCENE_TIMESTAMP_PATTERN = r"\d{2,}:[0-5]\d(?:\.\d{1,3})?"
_H3_EMPTY_SCENE_TRANSITION_RE = re.compile(
    r"<scenetrans>\s*\[Shot\s+\d+\]\s+At\s+"
    + _H3_SCENE_TIMESTAMP_PATTERN
    + r"\s*</scenetrans>",
    re.IGNORECASE,
)
_H3_TIMED_SHOT_MARKER_RE = re.compile(
    r"\[Shot\s+(?P<number>\d+)\](?=\s+At\s+(?P<timestamp>"
    + _H3_SCENE_TIMESTAMP_PATTERN
    + r")"
    + r"\b)",
    re.IGNORECASE,
)
_H3_SHOT_MARKER_RE = re.compile(r"\[Shot\s+(?P<number>\d+)\]", re.IGNORECASE)
_H3_SHOT_REFERENCE_PREFIX_RE = re.compile(
    r"\b(?:throughout|during|within|from|appears\s+in)\s*$",
    re.IGNORECASE,
)
_H3_AUXILIARY_SECTION_BOUNDARY_RE = re.compile(
    r"\b(?:integrated_multimodal_description|detailed_description|"
    r"overall_soundscape|non_diegetic_music|dialogue\s+beats?)\s*:|\[Shot\s+\d+\]",
    re.IGNORECASE,
)
_H3_META_SENTENCE_RE = re.compile(
    r"\bMiniMax H3 generates synchronized picture and stereo sound\.\s*",
    re.IGNORECASE,
)
_H3_MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u0080\u0098": "\u2018",
    "\u00e2\u0080\u0099": "\u2019",
    "\u00e2\u0080\u009c": "\u201c",
    "\u00e2\u0080\u009d": "\u201d",
    "\u00e2\u0080\u0093": "\u2013",
    "\u00e2\u0080\u0094": "\u2014",
    "\u00e2\u0080\u00a6": "\u2026",
    "\u00e2\u20ac\u02dc": "\u2018",
    "\u00e2\u20ac\u2122": "\u2019",
    "\u00e2\u20ac\u0153": "\u201c",
    "\u00e2\u20ac\u009d": "\u201d",
    "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u201d": "\u2014",
    "\u00e2\u20ac\u00a6": "\u2026",
    "\u00c2\u00a0": " ",
    # Some older Windows JSON round-trips decoded the leading UTF-8 byte for
    # punctuation as U+0101 instead of U+00E2 before preserving the two C1
    # continuation bytes. Repair that observed Maestro variant as well.
    "\u0101\u0080\u0098": "\u2018",
    "\u0101\u0080\u0099": "\u2019",
    "\u0101\u0080\u009c": "\u201c",
    "\u0101\u0080\u009d": "\u201d",
    "\u0101\u0080\u0093": "\u2013",
    "\u0101\u0080\u0094": "\u2014",
    "\u0101\u0080\u00a6": "\u2026",
}


def _field(value: Any, key: str, default: Any = "") -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalized_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dialogue_spans(prompt: str) -> tuple[list[tuple[int, int]], bool]:
    """Return balanced top-level dialogue spans and whether markup was bad."""

    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    malformed = False
    for token in _H3_DIALOGUE_TOKEN_RE.finditer(prompt or ""):
        closing = bool(token.group(1))
        if not closing:
            if depth == 0:
                start = token.start()
            else:
                malformed = True
            depth += 1
            continue
        if depth == 0:
            malformed = True
            continue
        depth -= 1
        if depth == 0 and start >= 0:
            spans.append((start, token.end()))
            start = -1
    if depth:
        malformed = True
    return spans, malformed


def _audio_policy_flag(audio_policy: Mapping[str, Any] | None, key: str) -> bool | None:
    """Return an explicitly supplied audio permission, preserving legacy defaults."""

    if not isinstance(audio_policy, Mapping) or key not in audio_policy:
        return None
    value = audio_policy.get(key)
    return value if isinstance(value, bool) else None


def _has_spoken_beats(dialogue_beats: Sequence[Any] | None) -> bool:
    return any(
        _normalized_space(_field(beat, "spoken_text", ""))
        for beat in dialogue_beats or []
    )


def _is_negated_instruction(text: str, start: int) -> bool:
    """Recognize nearby negative wording without treating it as an instruction."""

    boundary = max(
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
        text.rfind(";", 0, start),
        text.rfind("\n", 0, start),
    )
    prefix = text[max(boundary + 1, start - 90):start]
    if re.search(
        r"\b(?:no|not|never|without|don't|doesn't|didn't|cannot|can't|"
        r"do\s+not|does\s+not|did\s+not|cannot)\s+"
        r"(?:[\w’'-]+\s+){0,4}$",
        prefix,
        flags=re.IGNORECASE,
    ):
        return True
    suffix = text[start:start + 100]
    return bool(re.match(
        r"\s*(?:(?:non[ -]?diegetic|background|orchestral|cinematic|"
        r"instrumental|hopeful|heroic|soaring|gentle|peaceful|sustained|"
        r"uplifting|symphonic|dramatic)\s+)*(?:music|score|soundtrack)?\s*"
        r"(?:is|was|remains?|stays?)\s+"
        r"(?:not|absent|silent|off|missing|muted|disabled|unavailable|"
        r"unheard|inaudible)\b",
        suffix,
        flags=re.IGNORECASE,
    ))


def _same_sentence(text: str, start: int, end: int) -> str:
    left = max(
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
        text.rfind(";", 0, start),
        text.rfind("\n", 0, start),
    ) + 1
    right_candidates = [
        position for marker in ".!?;\n"
        if (position := text.find(marker, end)) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:right]


def _is_generated_speech_instruction(text: str, cue: re.Match[str]) -> bool:
    """Ignore visible text and nonverbal calls while retaining real voice cues."""

    if _is_negated_instruction(text, cue.start()):
        return False
    boundary = max(
        text.rfind(".", 0, cue.start()),
        text.rfind("!", 0, cue.start()),
        text.rfind("?", 0, cue.start()),
        text.rfind(";", 0, cue.start()),
        text.rfind("\n", 0, cue.start()),
    ) + 1
    prefix = text[boundary:cue.start()]
    suffix = text[cue.end():]
    if _H3_TEXTUAL_SPEECH_SUBJECT_RE.search(prefix):
        return False

    verb = cue.group(0).casefold().replace(" ", "-")
    if verb in {"call", "calls", "calling"}:
        # Preparing a phone call is a visible action, not an instruction to
        # generate a voice. Keep explicit quoted words classified as speech.
        if (
            re.search(r"\b(?:smartphone|phone|handset)\b[^.!?;\n]*\bto\s+$", prefix, re.IGNORECASE)
            and not re.match(r"\s*(?:out\s*)?(?:[:,—-]\s*)?[\"'“]", suffix)
        ):
            return False
        # Bare calls commonly describe birdsong, a call button, or a person
        # summoning a lift. Treat it as speech only when words are explicit.
        return bool(re.match(
            r"\s*(?:out\s*)?(?:[:,—-]\s*)?[\"'“][^\"'”]{1,}"
            r"|\s+(?:out\s+)?(?:for\s+(?:help|someone|the\s+others?)|"
            r"(?:her|his|their|the\s+\w+['’]s)\s+name|"
            r"to\s+(?:the\s+)?(?:other\s+\w+|\w+))\b",
            suffix,
            flags=re.IGNORECASE,
        ))

    if verb in {"say", "says", "said"} and re.match(
        r"\s*(?:nothing|no\s+(?:words?|speech|dialogue)|silently)\b",
        suffix,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _independent_continuation_start(text: str, start: int, end: int) -> int | None:
    """Find an action/SFX/camera clause after a music cue without a verb list."""

    tail = text[start:end]
    for boundary in re.finditer(
        r"(?:,\s*|\s+)(?:and|then|while|as)\s+",
        tail,
        flags=re.IGNORECASE,
    ):
        clause = tail[boundary.end():].lstrip()
        if (
            _H3_MUSIC_SUBJECT_CONTINUATION_RE.match(clause)
            or _H3_MUSIC_FOLLOWTHROUGH_RE.match(clause)
        ):
            continue
        if _H3_INDEPENDENT_CLAUSE_START_RE.match(clause):
            return start + boundary.start()
    return None


def _is_mapped_audio_instruction(
    text: str,
    start: int,
    end: int,
    *,
    has_driving_audio: bool,
) -> bool:
    if not has_driving_audio:
        return False
    sentence = _same_sentence(text, start, end)
    return bool(re.search(
        r"\b(?:mapped\s+driving\s+audio|supplied\s+(?:source\s+)?"
        r"(?:audio|track|soundtrack|song)|provided\s+(?:audio|track|"
        r"soundtrack)|source\s+(?:audio|track|soundtrack|song)|"
        r"<Audio\s+\d+>)\b",
        sentence,
        flags=re.IGNORECASE,
    ))


def _h3_music_instruction_matches(text: str) -> list[re.Match[str]]:
    matches = [*_H3_MUSIC_INSTRUCTION_RE.finditer(text or "")]
    matches.extend(_H3_MUSIC_DIRECTIVE_RE.finditer(text or ""))
    return sorted(matches, key=lambda match: match.start())


def _strip_generated_music_instructions(
    value: Any,
    *,
    has_driving_audio: bool = False,
) -> str:
    """Remove positive score cues while preserving nearby visual and SFX prose."""

    text = str(value or "")
    removals: list[tuple[int, int]] = []
    for match in _h3_music_instruction_matches(text):
        if _is_negated_instruction(text, match.start()) or _is_mapped_audio_instruction(
            text, match.start(), match.end(), has_driving_audio=has_driving_audio,
        ):
            continue

        start = match.start()
        prefix = text[max(0, start - 12):start]
        conjunction = re.search(r"\b(?:and|while|as)\s+$", prefix, re.IGNORECASE)
        if conjunction:
            start -= len(prefix) - conjunction.start()

        sentence_end_candidates = [
            position for marker in ".!?;\n"
            if (position := text.find(marker, match.end())) >= 0
        ]
        sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(text)
        comma_positions = [
            position for marker in ",;"
            if (position := text.find(marker, match.end(), sentence_end)) >= 0
        ]
        verb = str(match.groupdict().get("verb") or "").casefold()
        end = sentence_end
        if comma_positions and verb not in {"is", "was", "becomes", "became"}:
            comma = min(comma_positions)
            tail = text[comma + 1:sentence_end]
            if re.match(
                r"\s*(?:matching|underscoring|which|that|thereby|"
                r"building\s+(?:to|toward)|rising\s+(?:to|toward))\b",
                tail,
                flags=re.IGNORECASE,
            ):
                end = sentence_end
            else:
                end = comma

        # Retain independent movement, camera, and SFX clauses even when their
        # verb is outside the original whitelist (for example ``unlocks`` or
        # ``pans``). Musical follow-through clauses remain part of the removed
        # score instruction.
        continuation_start = _independent_continuation_start(
            text, match.end(), end,
        )
        if continuation_start is not None:
            end = continuation_start
        removals.append((start, end))

    result = text
    for start, end in sorted(removals, reverse=True):
        result = result[:start] + " " + result[end:]
    result = re.sub(r"\s+([,.;!?])", r"\1", result)
    result = re.sub(r"([,;])\s*[,;]+", r"\1", result)
    result = re.sub(r"\b(?:and|while|as)\s*([,.;!?])", r"\1", result, flags=re.I)
    result = re.sub(r"([.!?])\s*([.!?])", r"\1", result)
    result = _normalized_space(result).strip(" ,;:")
    return re.sub(r"^(?:and|while|as|then)\s+", "", result, flags=re.IGNORECASE)


def _strip_music_mentions_from_soundscape(
    value: Any,
    *,
    has_driving_audio: bool = False,
) -> str:
    """Drop score-only soundscape clauses while retaining ambience and effects."""

    text = _strip_generated_music_instructions(value, has_driving_audio=has_driving_audio)
    clauses = re.split(r"\s*(?:,|;|\band\b)\s*", text, flags=re.IGNORECASE)
    kept: list[str] = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if not re.search(r"\b(?:music|score|soundtrack)\b", clause, re.IGNORECASE):
            kept.append(clause)
            continue
        noun = re.search(r"\b(?:music|score|soundtrack)\b", clause, re.IGNORECASE)
        if (
            _is_negated_instruction(clause, noun.start())
            or _is_mapped_audio_instruction(
                clause, noun.start(), noun.end(), has_driving_audio=has_driving_audio,
            )
        ):
            kept.append(clause)
    return ", ".join(kept)


def _strip_forbidden_dialogue_blocks(value: Any) -> str:
    """Remove planner-authored tags and only the speech lead-in attached to them."""

    text = str(value or "")
    spans, _malformed = _dialogue_spans(text)
    for start, end in reversed(spans):
        prefix_start = max(
            text.rfind(".", 0, start),
            text.rfind("!", 0, start),
            text.rfind("?", 0, start),
            text.rfind(";", 0, start),
            text.rfind("\n", 0, start),
        ) + 1
        prefix = text[prefix_start:start]
        lead = re.search(
            r"(?i)(?:(?:[A-Z][\w’'-]*|the\s+[A-Z][\w’'-]*)\s+)?"
            r"(?:says?|speaks?|sang|sings?|whispers?|shouts?|calls?|"
            r"utters?|recites?|tells?)\s*(?:[:,—-]\s*)?$",
            prefix,
        )
        remove_start = prefix_start + lead.start() if lead else start
        text = text[:remove_start] + " " + text[end:]
    text = _strip_dialogue_for_driving_audio(text)
    text = re.sub(r"([.!?])(?:\s*[,;:.!?])+", r"\1", text)
    return _normalized_space(text)


def _audio_policy_violations(
    prompt: str,
    audio_policy: Mapping[str, Any] | None,
    *,
    dialogue_beats: Sequence[Any] | None = None,
    dialogue_manifest_authoritative: bool = False,
    has_driving_audio: bool = False,
) -> list[str]:
    """Find final prompt instructions that contradict explicit audio permissions."""

    text = normalize_h3_text(prompt)
    errors: list[str] = []
    speech_forbidden = (
        _audio_policy_flag(audio_policy, "generated_speech_allowed") is False
        or (
            dialogue_manifest_authoritative
            and not _has_spoken_beats(dialogue_beats)
        )
    )
    if speech_forbidden:
        spans, _malformed = _dialogue_spans(text)
        if spans:
            errors.append("dialogue tags contradict the authoritative no-generated-speech policy")
        if not has_driving_audio:
            without_tags = _replace_spans(text, spans, [""] * len(spans))
            for cue in _H3_SPOKEN_INSTRUCTION_RE.finditer(without_tags):
                if _is_generated_speech_instruction(without_tags, cue):
                    errors.append(
                        "generated speech instructions contradict the authoritative no-generated-speech policy"
                    )
                    break

    if _audio_policy_flag(audio_policy, "generated_music_allowed") is False:
        fields = _extract_h3_fields(text)
        music_field = _normalized_space(fields.get("non_diegetic_music", ""))
        no_music_values = {"", "n/a", "none", "no music", "no non-diegetic music", "not applicable"}
        if music_field.casefold() not in no_music_values and not re.search(
            r"\b(?:mapped\s+driving\s+audio|supplied\s+(?:source\s+)?"
            r"(?:audio|track|soundtrack|song)|provided\s+(?:audio|track|"
            r"soundtrack)|source\s+(?:audio|track|soundtrack|song)|"
            r"<Audio\s+\d+>)\b",
            music_field,
            flags=re.IGNORECASE,
        ):
            errors.append(
                "non_diegetic_music contains generated music despite the audio policy"
            )
        for cue in _h3_music_instruction_matches(text):
            if _is_negated_instruction(text, cue.start()) or _is_mapped_audio_instruction(
                text, cue.start(), cue.end(), has_driving_audio=has_driving_audio,
            ):
                continue
            errors.append("generated music instructions contradict the audio policy")
            break
    return list(dict.fromkeys(errors))


def _h3_visual_shot_markers(visual: str) -> list[re.Match[str]]:
    """Return cut labels outside dialogue and explicit reference phrases."""

    spans, _ = _dialogue_spans(visual)
    span_index = 0
    markers: list[re.Match[str]] = []
    for marker in _H3_SHOT_MARKER_RE.finditer(visual or ""):
        while span_index < len(spans) and marker.start() >= spans[span_index][1]:
            span_index += 1
        if (
            span_index < len(spans)
            and spans[span_index][0] <= marker.start() < spans[span_index][1]
        ):
            continue
        preceding = visual[max(0, marker.start() - 45):marker.start()]
        if _H3_SHOT_REFERENCE_PREFIX_RE.search(preceding):
            continue
        markers.append(marker)
    return markers


def _replace_spans(
    prompt: str,
    spans: Sequence[tuple[int, int]],
    replacements: Sequence[str],
) -> str:
    result = prompt
    for (start, end), replacement in reversed(list(zip(spans, replacements))):
        result = f"{result[:start]}{replacement}{result[end:]}"
    return result


def _strip_dialogue_for_driving_audio(prompt: str) -> str:
    """Remove generated dialogue when mapped audio owns every audible vocal.

    Music-video planning sometimes receives a noisy transcription containing a
    repeated refrain. If an LLM copies that transcript into a ``<d>`` block and
    reaches its output limit before ``</d>``, the visual plan is still safe to
    use: the mapped soundtrack, rather than prompt-authored speech, is the vocal
    source. Remove balanced blocks, malformed nested blocks, unmatched closing
    tags, and an unterminated final block without altering surrounding visual
    instructions.
    """

    text = str(prompt or "")
    pieces: list[str] = []
    cursor = 0
    depth = 0
    open_start = -1
    for token in _H3_DIALOGUE_TOKEN_RE.finditer(text):
        closing = bool(token.group(1))
        if not closing:
            if depth == 0:
                pieces.append(text[cursor:token.start()])
                open_start = token.start()
            depth += 1
            continue
        if depth:
            depth -= 1
            if depth == 0:
                cursor = token.end()
                open_start = -1
            continue

        # A closing tag without an opening tag is markup noise. Preserve the
        # prose around it, but do not leave the invalid token in the prompt.
        pieces.append(text[cursor:token.start()])
        cursor = token.end()

    if depth:
        # Preserve official sound/music fields if malformed raw text includes
        # them after an unterminated dialogue block. Context-IR parsing usually
        # separates these fields before this helper runs, but this also makes
        # the recovery safe for legacy free-form prompts.
        boundary = _H3_SOUND_BOUNDARY_RE.search(text, max(0, open_start))
        cursor = boundary.start() if boundary else len(text)
    pieces.append(text[cursor:])
    return _normalized_space(" ".join(pieces))


def _dialogue_payload(value: Any) -> tuple[str, str]:
    """Return ``(language, words)`` from plain or nested H3 dialogue."""

    # Structured Director beats may carry a language hint separately from
    # their plain spoken_text. Older saved beats still use the historical
    # English default, while an inline [Language] prefix remains authoritative
    # when present.
    text_value = _field(value, "spoken_text", value)
    text = _H3_DIALOGUE_TOKEN_RE.sub("", str(text_value or "")).strip()
    language = _normalized_space(_field(value, "language", "")) or "English"
    language_prefix = re.match(r"^\[([^\]]+)\]\s*(.*)$", text, re.DOTALL)
    if language_prefix:
        language = language_prefix.group(1).strip() or language
        text = language_prefix.group(2).strip()
    if not text:
        raise H3DialogueContractError("MiniMax H3 dialogue contains an empty line.")
    return language, text


def h3_dialogue_tag(spoken_text: Any) -> str:
    """Build exactly one canonical H3 dialogue block."""

    language, words = _dialogue_payload(spoken_text)
    return f"<d>[{language}] {words}</d>"


def _replace_first_outside_dialogue(
    prompt: str,
    needle: str,
    replacement: str,
) -> tuple[str, bool]:
    """Replace an exact plain-text line without touching existing H3 tags."""

    if not needle:
        return prompt, False
    spans, malformed = _dialogue_spans(prompt)
    if malformed:
        return prompt, False
    cursor = 0
    for start, end in [*spans, (len(prompt), len(prompt))]:
        found = prompt.find(needle, cursor, start)
        if found >= 0:
            return (
                f"{prompt[:found]}{replacement}{prompt[found + len(needle):]}",
                True,
            )
        cursor = end
    return prompt, False


def _rewrite_outside_dialogue(prompt: str, rewrite) -> str:
    spans, malformed = _dialogue_spans(prompt)
    if malformed:
        raise H3DialogueContractError(
            "MiniMax H3 dialogue tags are unbalanced after compilation."
        )
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(rewrite(prompt[cursor:start]))
        parts.append(prompt[start:end])
        cursor = end
    parts.append(rewrite(prompt[cursor:]))
    return "".join(parts)


def _strip_empty_h3_scene_transitions(prompt: str) -> str:
    """Remove marker-only transition shells while protecting dialogue text."""

    text = str(prompt or "")
    pieces: list[str] = []
    cursor = 0
    protected_start = -1
    depth = 0
    for token in _H3_DIALOGUE_TOKEN_RE.finditer(text):
        if not token.group(1):
            if depth == 0:
                pieces.append(
                    _H3_EMPTY_SCENE_TRANSITION_RE.sub("", text[cursor:token.start()])
                )
                protected_start = token.start()
            depth += 1
            continue

        if depth == 0:
            continue
        depth -= 1
        if depth == 0 and protected_start >= 0:
            pieces.append(text[protected_start:token.end()])
            cursor = token.end()
            protected_start = -1

    if depth and protected_start >= 0:
        # Leave an unterminated speech block for the dialogue compiler's
        # existing repair/rejection path.
        pieces.append(text[protected_start:])
    else:
        pieces.append(_H3_EMPTY_SCENE_TRANSITION_RE.sub("", text[cursor:]))
    return "".join(pieces)


def _normalize_h3_compiled_visual_shots(visual: str) -> str:
    """Remove empty transition shells and localize timed shot labels.

    Director planners can number a prompt's internal cuts across the full
    sequence. H3 receives one clip at a time, so only timestamped cut labels
    are rebased to that clip's local sequence. Untimed markers stay intact for
    the strict contract validator to judge. Tagged dialogue is kept verbatim.
    """

    text = _strip_empty_h3_scene_transitions(str(visual or ""))
    spans, malformed = _dialogue_spans(text)
    if malformed:
        raise H3DialogueContractError(
            "MiniMax H3 dialogue tags are unbalanced after compilation."
        )

    prose_segments: list[str] = []
    dialogue_segments: list[str] = []
    cursor = 0
    for start, end in spans:
        prose_segments.append(text[cursor:start])
        dialogue_segments.append(text[start:end])
        cursor = end
    prose_segments.append(text[cursor:])

    timed_markers: list[re.Match[str]] = []
    for segment in prose_segments:
        for marker in _H3_TIMED_SHOT_MARKER_RE.finditer(segment):
            preceding = segment[max(0, marker.start() - 45):marker.start()]
            if not _H3_SHOT_REFERENCE_PREFIX_RE.search(preceding):
                timed_markers.append(marker)

    # A monotonic sequence starting beyond Shot 1 can be a planner-wide
    # sequence label, so translate it by one constant offset. Preserve every
    # gap between source labels: compressing [Shot 1], [Shot 3] into [Shot 1],
    # [Shot 2] would hide a malformed local sequence from strict validation.
    # A decrease is also left intact so validation still rejects cut reversal.
    source_numbers = [int(marker.group("number")) for marker in timed_markers]
    can_rebase = bool(source_numbers and source_numbers[0] > 1) and all(
        current >= previous
        for previous, current in zip(source_numbers, source_numbers[1:])
    )
    first_local_number = 2
    if timed_markers:
        first = timed_markers[0]
        first_timestamp = first.group("timestamp")
        clock, _, fraction = first_timestamp.partition(".")
        minutes, seconds = (int(part) for part in clock.split(":"))
        begins_at_zero = minutes == 0 and seconds == 0 and not int(fraction or "0")
        if begins_at_zero:
            # A source's first frame is this clip's Shot 1, even when the
            # planner assigned it a sequence-wide number.
            first_local_number = 1

    source_to_local_offset = (
        first_local_number - source_numbers[0]
        if can_rebase
        else 0
    )

    def localize_segment(segment: str) -> str:
        def localize(match: re.Match[str]) -> str:
            if not can_rebase:
                return match.group(0)
            preceding = segment[max(0, match.start() - 45):match.start()]
            if _H3_SHOT_REFERENCE_PREFIX_RE.search(preceding):
                return match.group(0)

            source_number = int(match.group("number"))
            return f"[Shot {source_number + source_to_local_offset}]"

        return _H3_TIMED_SHOT_MARKER_RE.sub(localize, segment)

    parts: list[str] = []
    for index, segment in enumerate(prose_segments):
        parts.append(localize_segment(segment))
        if index < len(dialogue_segments):
            parts.append(dialogue_segments[index])
    return "".join(parts)


def _sanitize_scripted_ambience(prompt: str) -> str:
    """Remove requests for synthetic background speech around exact lines."""

    def sanitize(segment: str) -> str:
        for pattern, replacement in _H3_SPEECHLIKE_AMBIENCE_REPLACEMENTS:
            segment = pattern.sub(replacement, segment)
        return re.sub(
            r"\bnonverbal room tone(?:\s*(?:,|and)\s*nonverbal room tone)+\b",
            "nonverbal room tone",
            segment,
            flags=re.IGNORECASE,
        )

    return _rewrite_outside_dialogue(prompt, sanitize)


def _insert_visual_detail(prompt: str, label: str, detail: str) -> str:
    prompt = str(prompt or "").strip()
    detail = _normalized_space(detail)
    statement = f"{label}: {detail}"
    suffix = "" if statement.endswith((".", "!", "?")) else "."
    boundary = _H3_SOUND_BOUNDARY_RE.search(prompt)
    if boundary:
        return (
            f"{prompt[:boundary.start()].rstrip()} {statement}{suffix} "
            f"{prompt[boundary.start():].lstrip()}"
        ).strip()
    return f"{prompt} {statement}{suffix}".strip()


def _speaker_map(subjects: Iterable[Any]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for index, subject in enumerate(subjects or []):
        character_id = _normalized_space(_field(subject, "character_id", ""))
        speaker_name = (
            _h3_identity_label(_field(subject, "speaker_name", ""))
            or _h3_identity_label(_field(subject, "visual_description", ""))
            or _h3_identity_label(character_id)
            or f"visible subject {index + 1}"
        )
        stable_id = f"(S{index + 1})"
        if character_id:
            result[character_id] = (stable_id, speaker_name)
            result[character_id.casefold()] = (stable_id, speaker_name)
        result.setdefault(speaker_name.casefold(), (stable_id, speaker_name))
    return result


def validate_h3_vocal_contract(
    prompt: str,
    dialogue_beats: Sequence[Any] | None = None,
) -> list[str]:
    """Return structural/semantic H3 dialogue errors for a final prompt."""

    prompt = normalize_h3_text(prompt)
    spans, malformed = _dialogue_spans(prompt)
    errors: list[str] = []
    if malformed:
        errors.append("dialogue tags are nested or unbalanced")
    blocks = [prompt[start:end] for start, end in spans]
    actual_words: list[str] = []
    for index, block in enumerate(blocks):
        strict = _H3_STRICT_DIALOGUE_RE.fullmatch(block.strip())
        if not strict or _H3_DIALOGUE_TOKEN_RE.search(strict.group(2) if strict else ""):
            errors.append(f"dialogue block {index + 1} is not canonical")
            continue
        actual_words.append(_normalized_space(strict.group(2)))

    expected_words = []
    for beat in dialogue_beats or []:
        spoken = normalize_h3_text(_field(beat, "spoken_text", ""))
        if _normalized_space(spoken):
            try:
                expected_words.append(_normalized_space(_dialogue_payload(spoken)[1]))
            except H3DialogueContractError as exc:
                errors.append(str(exc))

    if expected_words:
        if len(blocks) != len(expected_words):
            errors.append(
                f"expected {len(expected_words)} dialogue block(s), found {len(blocks)}"
            )
        if actual_words != expected_words:
            errors.append("dialogue words or speaker order differ from dialogue_beats")
    elif not blocks:
        has_silence_contract = bool(re.search(
            r"\bno (?:one|character) speaks\b",
            prompt,
            flags=re.IGNORECASE,
        ))
        has_mapped_audio_contract = (
            "mapped driving audio" in prompt.casefold()
            and "do not generate additional dialogue" in prompt.casefold()
        )
        if not (has_silence_contract or has_mapped_audio_contract):
            errors.append("silent prompt has no explicit H3 silence contract")
    return errors


def compile_h3_vocal_contract(
    prompt: str,
    subjects: Sequence[Any] | None,
    dialogue_beats: Sequence[Any] | None,
) -> tuple[str, str]:
    """Compile a final, idempotent H3 speech/silence contract.

    Existing dialogue is replaced by *whole top-level blocks*. Balanced nested
    blocks from older Maestro projects are therefore repaired without ever
    performing the unsafe ``prompt.replace(spoken_text, tag)`` operation that
    originally split sentences and produced gibberish.
    """

    prompt = str(prompt or "").strip()
    valid_beats: list[tuple[Any, str, str]] = []
    for beat in dialogue_beats or []:
        spoken = _field(beat, "spoken_text", "")
        if not _normalized_space(spoken):
            continue
        _, words = _dialogue_payload(spoken)
        valid_beats.append((beat, words, h3_dialogue_tag(spoken)))

    # Older saved projects do not carry structured dialogue metadata. If they
    # already contain a vocal section, preserve its speaker assignments and
    # repair the full prompt in place instead of deleting the only copy of a
    # line that had been appended inside that section.
    existing_vocal_section = _H3_VOCAL_SECTION_RE.search(prompt)
    if not valid_beats and existing_vocal_section:
        spans, malformed = _dialogue_spans(prompt)
        if malformed and not spans:
            raise H3DialogueContractError(
                "MiniMax H3 dialogue tags are unbalanced and cannot be repaired safely."
            )
        if spans:
            prompt = _replace_spans(
                prompt,
                spans,
                [h3_dialogue_tag(prompt[start:end]) for start, end in spans],
            )
            prompt = _sanitize_scripted_ambience(prompt)
        _, remains_malformed = _dialogue_spans(prompt)
        if remains_malformed:
            raise H3DialogueContractError(
                "MiniMax H3 dialogue tags remain unbalanced after repair."
            )
        errors = validate_h3_vocal_contract(prompt, [])
        if errors:
            raise H3DialogueContractError(
                "Invalid MiniMax H3 vocal contract: " + "; ".join(errors)
            )
        contract_match = _H3_VOCAL_SECTION_RE.search(prompt)
        contract = _normalized_space(
            contract_match.group(0) if contract_match else ""
        )
        return prompt, contract

    prompt = _H3_VOCAL_SECTION_RE.sub(" ", prompt).strip()
    spans, malformed = _dialogue_spans(prompt)
    if malformed and not spans:
        raise H3DialogueContractError(
            "MiniMax H3 dialogue tags are unbalanced and cannot be repaired safely."
        )

    instructions: list[str] = []
    if valid_beats:
        replacements = [
            valid_beats[index][2] if index < len(valid_beats) else ""
            for index in range(len(spans))
        ]
        prompt = _replace_spans(prompt, spans, replacements)
        _, remains_malformed = _dialogue_spans(prompt)
        if remains_malformed:
            raise H3DialogueContractError(
                "MiniMax H3 dialogue tags remain unbalanced after repair."
            )

        mapped = _speaker_map(subjects or [])
        for index in range(len(spans), len(valid_beats)):
            beat, words, tag = valid_beats[index]
            prompt, replaced = _replace_first_outside_dialogue(
                prompt, words, tag,
            )
            if replaced:
                continue
            speaker_id = _normalized_space(_field(beat, "speaker_id", ""))
            stable_id, speaker_name = mapped.get(
                speaker_id,
                mapped.get(
                    speaker_id.casefold(),
                    (speaker_id or "(S1)", speaker_id or "the visible speaker"),
                ),
            )
            delivery = _normalized_space(_field(beat, "delivery", ""))
            physical_cue = _normalized_space(_field(beat, "physical_cue", ""))
            delivery_text = f" with a {delivery} delivery" if delivery else ""
            instruction = (
                f"{stable_id} {speaker_name} speaks{delivery_text}: {tag}."
            )
            if physical_cue:
                instruction += f" While speaking, {physical_cue}"
            instructions.append(instruction)

        prompt = _sanitize_scripted_ambience(prompt)
        guard = (
            "Only these explicitly tagged lines are spoken, once each and in "
            "the listed order. Do not add, paraphrase, repeat, or improvise "
            "dialogue, muttering, murmuring, gibberish, or speech-like "
            "vocalizations. No background or crowd voices are audible. Anyone "
            "not currently delivering a tagged line remains silent with their "
            "mouth closed except for explicitly described nonverbal actions."
        )
        detail = " ".join([*instructions, guard])
        label = "DIALOGUE AND VOCAL PERFORMANCE"
    elif spans:
        replacements = [h3_dialogue_tag(prompt[start:end]) for start, end in spans]
        prompt = _replace_spans(prompt, spans, replacements)
        _, remains_malformed = _dialogue_spans(prompt)
        if remains_malformed:
            raise H3DialogueContractError(
                "MiniMax H3 dialogue tags remain unbalanced after repair."
            )
        prompt = _sanitize_scripted_ambience(prompt)
        detail = (
            "Only the explicitly tagged dialogue already present in this "
            "prompt is spoken. Do not add, paraphrase, repeat, or improvise "
            "any other words, muttering, murmuring, gibberish, or speech-like "
            "vocalizations. No background or crowd voices are audible. "
            "Non-speaking characters keep their mouths closed."
        )
        label = "DIALOGUE AND VOCAL PERFORMANCE"
    else:
        detail = (
            "No one speaks in this shot. All visible people remain silent and "
            "keep their mouths closed except for explicitly described nonverbal "
            "actions. Generate no muttering, murmuring, gibberish, invented "
            "words, background voices, or speech-like vocalizations."
        )
        label = "SILENCE AND VOCAL PERFORMANCE"

    prompt = _insert_visual_detail(prompt, label, detail)
    errors = validate_h3_vocal_contract(prompt, dialogue_beats)
    if errors:
        raise H3DialogueContractError(
            "Invalid MiniMax H3 vocal contract: " + "; ".join(errors)
        )
    return prompt, f"{label}: {detail}"


def h3_dialogue_budget_violations(
    shot_dicts: Sequence[Mapping[str, Any]],
    durations: Sequence[float] | None = None,
    *,
    words_per_second: float = DIALOGUE_MAX_WORDS_PER_SECOND,
) -> list[dict[str, Any]]:
    """Describe shots whose complete dialogue cannot fit their duration."""

    violations: list[dict[str, Any]] = []
    for index, shot in enumerate(shot_dicts):
        try:
            duration = float(
                durations[index]
                if durations is not None and index < len(durations)
                else shot.get("duration_sec") or 0
            )
        except (TypeError, ValueError):
            duration = 0.0
        word_count = 0
        for beat in shot.get("dialogue_beats") or []:
            spoken = _field(beat, "spoken_text", "")
            if not _normalized_space(spoken):
                continue
            try:
                words = _dialogue_payload(spoken)[1]
            except H3DialogueContractError:
                words = str(spoken or "")
            word_count += len(words.split())
        budget = max(1, int(math.floor(max(0.0, duration) * words_per_second)))
        if word_count > budget:
            violations.append({
                "index": index,
                "title": _normalized_space(shot.get("title") or f"Shot {index + 1}"),
                "duration_sec": duration,
                "word_count": word_count,
                "word_budget": budget,
            })
    return violations


def normalize_h3_text(value: Any) -> str:
    """Repair common UTF-8 mojibake before H3 tokenization.

    Saved Director projects created through a Windows code-page boundary can
    contain the three code points for a UTF-8 punctuation byte sequence (for
    example ``\u00e2\u0080\u0099``) instead of the intended apostrophe.  The
    C1 control bytes are especially harmful in literal dialogue, so repair the
    known sequences and remove any orphaned controls deterministically.
    """

    return repair_text(value)


def _extract_h3_fields(text: str) -> dict[str, str]:
    """Extract known Context-IR fields even from legacy one-line prompts."""

    matches = list(re.finditer(
        r"(?i)(?<![A-Za-z0-9_])(" + "|".join(
            re.escape(field) for field in _H3_ALL_FIELDS
        ) + r")\s*:",
        text,
    ))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        if name in fields:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[name] = text[match.end():end].strip()
    return fields


def _strip_h3_custom_sections(text: str) -> str:
    previous = None
    result = str(text or "")
    while result != previous:
        previous = result
        result = _H3_CUSTOM_SECTION_RE.sub(" ", result)
    return result


def _trim_sentence(value: Any) -> str:
    return _normalized_space(value).strip(" .")


def _meaningful_context_present(context: str, body: str) -> bool:
    stop = {
        "about", "after", "again", "being", "from", "have", "into",
        "make", "named", "show", "starring", "that", "their", "this",
        "with", "when", "where", "while",
    }
    words = {
        word.casefold()
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{3,}", context)
        if word.casefold() not in stop
    }
    if not words:
        return True
    body_words = {
        word.casefold()
        for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{3,}", body)
    }
    required = min(4, max(2, math.ceil(len(words) * 0.45)))
    return len(words & body_words) >= required


_H3_GENERIC_IDENTITY_LABELS = {
    "character", "crowd", "customer", "extra", "listener", "man",
    "narrator", "patron", "person", "speaker", "subject", "the man",
    "the woman", "visible speaker", "woman", "unnamed", "n/a", "n a",
    "na",
}
_H3_NONIDENTITY_REFERENCE_ALIASES = {
    "he", "her", "hers", "him", "his", "it", "its", "she", "their",
    "theirs", "them", "they", "we", "you",
}


def _h3_anchor_key(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        normalize_h3_text(value).casefold(),
    ).strip()


_H3_PLACEHOLDER_IDENTITY_KEYS = {"unnamed", "n a", "na"}


def _h3_identity_label(value: Any) -> str:
    """Return a usable identity label, excluding empty planner placeholders."""

    label = _normalized_space(value)
    key = _h3_anchor_key(label)
    key = re.sub(r"^(?:the|a|an)\s+", "", key)
    if key in _H3_PLACEHOLDER_IDENTITY_KEYS:
        return ""
    return label


def _h3_is_placeholder_identity(value: Any) -> bool:
    return bool(value) and not _h3_identity_label(value)


def _h3_anchor_present(anchor: Any, text: Any) -> bool:
    wanted = _h3_anchor_key(anchor)
    return not wanted or wanted in _h3_anchor_key(text)


def _h3_context_anchors(values: Iterable[Any]) -> list[str]:
    anchors: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        anchor = _normalized_space(normalize_h3_text(value)).strip(" .;:-")
        key = _h3_anchor_key(anchor)
        if not key or key in seen:
            continue
        seen.add(key)
        anchors.append(anchor)
    return anchors


def _ensure_h3_context_anchors(body: str, anchors: Iterable[Any]) -> str:
    required = _h3_context_anchors(anchors)
    missing = [anchor for anchor in required if not _h3_anchor_present(anchor, body)]
    if not missing:
        return body
    return _normalized_space(
        f"Canonical identity and world: {'; '.join(missing)}. {body}"
    )


def _source_prompt_parts(
    prompt: str,
    *,
    project_context: str = "",
    context_anchors: Sequence[str] | None = None,
    opening_blocking: str = "",
    closing_blocking: str = "",
    audio_plan: Mapping[str, Any] | None = None,
    audio_policy: Mapping[str, Any] | None = None,
    has_driving_audio: bool = False,
) -> tuple[str, str, str, list[str]]:
    """Recover one concise visual timeline and the two official sound fields."""

    text = normalize_h3_text(prompt).strip()
    spans, _ = _dialogue_spans(text)
    existing_blocks = [
        h3_dialogue_tag(normalize_h3_text(text[start:end]))
        for start, end in spans
    ]
    fields = _extract_h3_fields(text)
    body = (
        fields.get("detailed_description")
        or fields.get("integrated_multimodal_description")
        or text
    )
    soundscape = fields.get("overall_soundscape", "")
    music = fields.get("non_diegetic_music", "")
    if not (
        fields.get("detailed_description")
        or fields.get("integrated_multimodal_description")
    ):
        boundary = _H3_SOUND_BOUNDARY_RE.search(body)
        if boundary:
            body = body[:boundary.start()]

    if not soundscape:
        match = _H3_SOUND_PROSE_RE.search(body)
        if match:
            soundscape = match.group(1)
            body = f"{body[:match.start()]} {body[match.end():]}"
    if not music:
        match = _H3_MUSIC_PROSE_RE.search(body)
        if match:
            music = match.group(1)
            body = f"{body[:match.start()]} {body[match.end():]}"

    # Legacy Maestro added a long context anchor before the actual prompt.
    # The planner's native body starts at this sentence, so retain everything
    # after it and discard only the duplicated wrapper.
    marker = _H3_META_SENTENCE_RE.search(body)
    if marker and body.lstrip().lower().startswith("project continuity"):
        body = body[marker.end():]
    body = _H3_META_SENTENCE_RE.sub("", body)
    body = _strip_h3_custom_sections(body)
    # A planner may serialize a transition with no action after its timed
    # marker. Remove that empty shell while the body is still source text so
    # Ref2VA's style-opening split cannot mistake the trailing marker for the
    # boundary between its opening and visual timeline.
    body = _strip_empty_h3_scene_transitions(body)
    body = re.sub(
        r"^\s*(?:integrated_multimodal_description|detailed_description)\s*:\s*",
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(r"^\s*\[Shot\s+1\]\s*", "", body, flags=re.IGNORECASE)
    body = _normalized_space(body)

    plan = audio_plan if isinstance(audio_plan, Mapping) else {}
    audio_mode = _normalized_space(_field(plan, "mode", "")).casefold()
    context = _normalized_space(normalize_h3_text(project_context))
    # The song-wide brief can name actions assigned to later clips. Keep it
    # available to vocal helpers, while this shot's prose and anchors below
    # remain authoritative for its visual action.
    if (
        context
        and audio_mode != "music_driven"
        and not _meaningful_context_present(context, body)
    ):
        if len(context) > 360:
            context = context[:360].rsplit(" ", 1)[0].rstrip(" ,;:-") + "..."
        body = f"Project context: {context}. {body}".strip()
    body = _ensure_h3_context_anchors(body, context_anchors or [])

    opening = _normalized_space(normalize_h3_text(opening_blocking))
    if opening and opening.casefold() not in body.casefold():
        body = f"Opening composition: {opening}. {body}".strip()
    closing = _normalized_space(normalize_h3_text(closing_blocking))
    # Native shot prompts already carry a concise ``Final beat``. Appending
    # Director's expanded closing cast table repeated every identity, outfit,
    # and position near the end of the prompt, making the actual action and
    # dialogue less prominent to the conditioner.
    if (
        closing
        and "final beat:" not in body.casefold()
        and closing.casefold() not in body.casefold()
    ):
        body = f"{body} By the final beat, {closing}.".strip()

    if not _trim_sentence(soundscape):
        sound_bits = []
        ambience = _trim_sentence(plan.get("ambience", ""))
        if ambience:
            sound_bits.append(ambience)
        sound_bits.extend(
            _trim_sentence(effect)
            for effect in (plan.get("effects") or [])
            if _trim_sentence(effect)
        )
        soundscape = ", ".join(sound_bits) or "Natural scene-appropriate stereo ambience"
    # Small local planners occasionally emit the requested Context-IR labels
    # inline after first writing a prose prompt. The prose music extractor can
    # then capture the repeated visual/sound fields (and even a non-canonical
    # ``<d>`` block) as part of non_diegetic_music. Auxiliary fields never own
    # dialogue, so trim at the first nested field and remove any dialogue block
    # before final validation. The visual compiler retains/canonicalizes the
    # authoritative copy when scripted dialogue is actually present.
    def clean_auxiliary(value: Any) -> str:
        cleaned = normalize_h3_text(value)
        boundary = _H3_AUXILIARY_SECTION_BOUNDARY_RE.search(cleaned)
        if boundary:
            cleaned = cleaned[:boundary.start()]
        return _strip_dialogue_for_driving_audio(cleaned)

    soundscape = _trim_sentence(
        _sanitize_scripted_ambience(clean_auxiliary(soundscape))
    )
    music = _trim_sentence(clean_auxiliary(music)) or "N/A"
    if music.casefold() in {"n/a", "none", "no music"}:
        music = "N/A"
    if _audio_policy_flag(audio_policy, "generated_music_allowed") is False:
        body = _strip_generated_music_instructions(
            body,
            has_driving_audio=has_driving_audio,
        )
        soundscape = _strip_music_mentions_from_soundscape(
            soundscape,
            has_driving_audio=has_driving_audio,
        ) or "Natural scene-appropriate stereo ambience"
        music = (
            "Use the mapped driving audio according to retention_analysis."
            if has_driving_audio
            else "N/A"
        )
    return body, soundscape, music, existing_blocks


def _speaker_registry_entry(
    registry: Mapping[str, Any], key: str,
) -> tuple[str, str] | None:
    value = registry.get(key) or registry.get(key.casefold())
    if isinstance(value, Mapping):
        stable_id = _normalized_space(value.get("stable_id", ""))
        name = _normalized_space(value.get("speaker_name", ""))
    elif value:
        stable_id = _normalized_space(value)
        name = ""
    else:
        return None
    if stable_id and not stable_id.startswith("("):
        stable_id = f"({stable_id})"
    return stable_id, _h3_identity_label(name)


def _subject_name_for_key(subjects: Sequence[Any], key: str) -> str:
    folded = key.casefold()
    for subject in subjects or []:
        character_id = _normalized_space(_field(subject, "character_id", ""))
        speaker_name = _h3_identity_label(_field(subject, "speaker_name", ""))
        if folded == character_id.casefold() and not _h3_is_placeholder_identity(key):
            return speaker_name or _h3_identity_label(character_id) or "the visible speaker"
        if speaker_name and folded == speaker_name.casefold():
            return speaker_name
    return _h3_identity_label(key) or "the visible speaker"


def _set_h3_subject_field(subject: Any, key: str, value: Any) -> None:
    if isinstance(subject, MutableMapping):
        subject[key] = value
    elif hasattr(subject, key):
        setattr(subject, key, value)


def _h3_subjects_for_physical_state(
    subjects: Sequence[Any] | None,
    physical_state: Any,
) -> Sequence[Any] | None:
    """Keep temporary blocking out of H3's clip-long subject definitions."""

    state = normalize_physical_state(physical_state)
    if not state or not any(
        state.get(key) for key in ("opening", "transitions", "closing")
    ):
        return subjects

    sanitized_subjects: list[Any] = []
    for subject in subjects or []:
        if isinstance(subject, Mapping):
            serialized = dict(subject)
        else:
            try:
                serialized = dict(vars(subject))
            except TypeError:
                # H3 planner subjects are JSON mappings. Unknown opaque subject
                # objects remain untouched rather than being mutated in place.
                sanitized_subjects.append(subject)
                continue
        description = _field(serialized, "visual_description", "")
        if description:
            serialized["visual_description"] = identity_only_description(description)
        sanitized_subjects.append(serialized)
    return sanitized_subjects


def _leading_h3_identity(value: Any) -> str:
    text = _normalized_space(value)
    proper = r"[A-Z][A-Za-z0-9'’-]+"
    match = (
        # Shot planners most commonly use ``Rachel: ...`` or
        # ``George Costanza (char_2): ...``. A single capitalized token is
        # trustworthy before those explicit identity delimiters.
        re.match(rf"^({proper}(?:\s+{proper}){{0,2}})(?=\s*(?:\(|:))", text)
        # Before ordinary descriptive punctuation, require a multi-token name
        # so adjectives such as "Massive," or hyphenated wardrobe colors do
        # not become phantom characters.
        or re.match(rf"^({proper}(?:\s+{proper}){{1,2}})(?=\s*(?:,|—|–|-))", text)
        or re.match(
            rf"^({proper}(?:\s+{proper}){{0,2}})(?=\s+(?:is|wears|stands|"
            r"sits|walks|enters|from|in)\b)",
            text,
        )
    )
    if not match:
        return ""
    candidate = match.group(1).strip()
    if (
        candidate.casefold() in _H3_GENERIC_IDENTITY_LABELS
        or _h3_is_placeholder_identity(candidate)
    ):
        return ""
    return candidate


_H3_LOCAL_CAST_SLOT_RE = re.compile(
    r"^(?:char(?:acter)?|subject|speaker)[_-]?\d+$",
    re.IGNORECASE,
)


def _h3_identity_names_match(left: Any, right: Any) -> bool:
    """Return true for a full cast name and its unambiguous short form."""

    left_key = _h3_anchor_key(left)
    right_key = _h3_anchor_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_parts = left_key.split()
    right_parts = right_key.split()
    if len(left_parts) == 1:
        return left_parts[0] in {right_parts[0], right_parts[-1]}
    if len(right_parts) == 1:
        return right_parts[0] in {left_parts[0], left_parts[-1]}
    return False


def _h3_subject_identity_evidence(subject: Any) -> tuple[str, list[str]]:
    """Read the local shot's identity without trusting a stale global label.

    Bounded long-form sequence writers legitimately reuse ``char_1`` and
    ``char_2`` for different guest casts.  Older project compilation then
    stamped one globally selected name onto every matching slot.  The leading
    name in the shot-local visual description is more authoritative when it
    directly contradicts that stale label.
    """

    name = _h3_identity_label(_field(subject, "speaker_name", ""))
    inferred = _leading_h3_identity(
        _field(subject, "visual_description", "")
    )
    if inferred and name and not _h3_identity_names_match(inferred, name):
        return inferred, [inferred]
    candidates = [
        candidate for candidate in (name, inferred)
        if candidate
        and candidate.casefold() not in _H3_GENERIC_IDENTITY_LABELS
        and not _h3_is_placeholder_identity(candidate)
        and not _H3_LOCAL_CAST_SLOT_RE.fullmatch(candidate)
    ]
    if not candidates:
        return "", []
    best = max(
        candidates,
        key=lambda candidate: (
            len(_h3_anchor_key(candidate).split()),
            int(not candidate.isupper()),
            len(candidate),
        ),
    )
    return best, candidates


def _h3_identity_slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _h3_anchor_key(value)).strip("_")
    return slug[:56] or "subject"


def _h3_direct_speaker_mentions(value: Any, name: str) -> int:
    """Count explicit action statements which assign speech to ``name``.

    This deliberately recognizes only direct grammatical attributions such as
    ``Thanos speaks`` or ``Thanos begins to speak``.  A loose proximity search
    would misread action such as ``Thanos watches as Rachel speaks`` and move
    Rachel's line to the wrong face.
    """

    text = normalize_h3_text(value)
    clean_name = _normalized_space(name)
    if not text or not clean_name:
        return 0
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(clean_name)}(?![A-Za-z0-9])"
        r"(?:\s*\([^)]{0,80}\))?\s+"
        r"(?:visibly\s+)?"
        r"(?:(?:begins?|starts?|continues?|finishes?)\s+(?:to\s+)?)?"
        r"(?:speaks?|says?|asks?|replies?|answers?|shouts?|yells?|"
        r"declares?|whispers?|delivers?)\b",
        flags=re.IGNORECASE,
    )
    return len(pattern.findall(text))


def _repair_h3_phantom_dialogue_subjects(
    clip_plans: Sequence[Mapping[str, Any]],
) -> None:
    """Merge a misspelled dialogue-only person back into the visible actor.

    A long-form screenplay occasionally misspells an established heading
    (for example ``THANOS`` -> ``THORNS``).  Pass 2 historically converted the
    typo into a new ``dialogue_thorns`` subject, placed it beside Thanos, and
    then handed both identities to Ref2VA.  That is a literal instruction to
    render the principal twice.

    Repair is intentionally evidence-bound: the unknown subject must use a
    synthetic ``dialogue_*`` id, another visible subject must be explicitly
    assigned speech in the shot action, and that attribution must identify one
    unique person.  Once established, the same typo is repaired across later
    clips in the bounded sequence.  Ambiguous or genuinely independent
    speakers are left untouched for the normal contract validator.
    """

    alias_targets: dict[str, str] = {}

    # Discover a reliable identity for each synthetic alias before mutating
    # anything, so a later clip can reuse evidence established in an earlier
    # one even when the later action contains only a reaction beat.
    for plan in clip_plans:
        if not isinstance(plan, Mapping):
            continue
        subjects = [
            subject for subject in plan.get("_director_subjects_on_screen") or []
            if isinstance(subject, Mapping)
        ]
        speaking_ids = {
            _normalized_space(beat.get("speaker_id", "")).casefold()
            for beat in plan.get("_director_dialogue_beats") or []
            if isinstance(beat, Mapping)
            and _normalized_space(beat.get("spoken_text", ""))
        }
        source = " ".join(filter(None, [
            _normalized_space(plan.get("_director_h3_source_prompt", "")),
            _normalized_space(plan.get("video_prompt_pre_polish", "")),
        ]))
        for phantom in subjects:
            phantom_id = _normalized_space(phantom.get("character_id", ""))
            if (
                not phantom_id.casefold().startswith("dialogue_")
                or phantom_id.casefold() not in speaking_ids
            ):
                continue
            phantom_name = _normalized_space(
                _h3_identity_label(phantom.get("speaker_name", "")) or phantom_id
            )
            alias_keys = {
                phantom_id.casefold(),
                phantom_name.casefold(),
            }
            if any(key in alias_targets for key in alias_keys):
                continue
            attributed: list[tuple[int, str]] = []
            for candidate in subjects:
                candidate_id = _normalized_space(candidate.get("character_id", ""))
                if not candidate_id or candidate_id.casefold() == phantom_id.casefold():
                    continue
                candidate_name, _evidence = _h3_subject_identity_evidence(candidate)
                if not candidate_name:
                    candidate_name = (
                        _h3_identity_label(candidate.get("speaker_name", ""))
                        or _h3_identity_label(candidate_id)
                    )
                count = _h3_direct_speaker_mentions(source, candidate_name)
                if count:
                    attributed.append((count, candidate_name))
            if not attributed:
                continue
            best_count = max(count for count, _name in attributed)
            best_names = {
                name for count, name in attributed if count == best_count
            }
            if len(best_names) != 1:
                continue
            target_name = next(iter(best_names))
            for key in alias_keys:
                alias_targets[key] = target_name

    repaired = 0
    for plan in clip_plans:
        if not isinstance(plan, MutableMapping):
            continue
        subjects = [
            subject for subject in plan.get("_director_subjects_on_screen") or []
            if isinstance(subject, MutableMapping)
        ]
        replacements: list[tuple[str, str, str, str]] = []
        for phantom in subjects:
            phantom_id = _normalized_space(phantom.get("character_id", ""))
            phantom_name = _normalized_space(
                _h3_identity_label(phantom.get("speaker_name", "")) or phantom_id
            )
            target_name = (
                alias_targets.get(phantom_id.casefold())
                or alias_targets.get(phantom_name.casefold())
            )
            if not target_name:
                continue
            target = next((
                candidate for candidate in subjects
                if candidate is not phantom
                and _h3_identity_names_match(
                    _h3_subject_identity_evidence(candidate)[0]
                    or _h3_identity_label(candidate.get("speaker_name", ""))
                    or _h3_identity_label(candidate.get("character_id", "")),
                    target_name,
                )
            ), None)
            if target is None:
                continue
            target_id = _normalized_space(target.get("character_id", ""))
            canonical_name = _normalized_space(
                _h3_identity_label(target.get("speaker_name", "")) or target_name
            )
            if not target_id or not canonical_name:
                continue
            replacements.append((
                phantom_id,
                phantom_name,
                target_id,
                canonical_name,
            ))

        if not replacements:
            continue
        phantom_ids = {old_id.casefold() for old_id, _old, _new_id, _new in replacements}
        plan["_director_subjects_on_screen"] = [
            subject for subject in subjects
            if _normalized_space(subject.get("character_id", "")).casefold()
            not in phantom_ids
        ]
        for beat in plan.get("_director_dialogue_beats") or []:
            if not isinstance(beat, MutableMapping):
                continue
            speaker_id = _normalized_space(beat.get("speaker_id", ""))
            for old_id, old_name, new_id, new_name in replacements:
                if speaker_id.casefold() != old_id.casefold():
                    continue
                beat["speaker_id"] = new_id
                cue = _normalized_space(beat.get("physical_cue", ""))
                if cue:
                    beat["physical_cue"] = re.sub(
                        rf"(?<![A-Za-z0-9]){re.escape(old_name)}(?![A-Za-z0-9])",
                        new_name,
                        cue,
                        flags=re.IGNORECASE,
                    )
                repaired += 1
                break

        for field in (
            "_director_h3_source_prompt",
            "_director_opening_blocking",
            "_director_closing_blocking",
        ):
            value = plan.get(field)
            if not isinstance(value, str):
                continue
            for old_id, old_name, new_id, new_name in replacements:
                value = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(old_id)}(?![A-Za-z0-9])",
                    new_id,
                    value,
                    flags=re.IGNORECASE,
                )
                value = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(old_name)}(?![A-Za-z0-9])",
                    new_name,
                    value,
                    flags=re.IGNORECASE,
                )
            plan[field] = value
        # Force an immediate clean compilation rather than comparing against
        # a wrapper produced before the identity merge.
        plan.pop("_director_h3_compiled_prompt", None)
        plan.pop("_director_speaker_registry", None)
        plan["_director_h3_phantom_speaker_repaired"] = True

    if repaired:
        # The registry is copied onto every saved clip.  One phantom entry in
        # a single shot therefore contaminates otherwise unrelated clips too;
        # discard every cached copy and rebuild it from the cleaned subjects.
        for plan in clip_plans:
            if isinstance(plan, MutableMapping):
                plan.pop("_director_speaker_registry", None)
        print(
            "[MiniMax H3] Merged "
            f"{repaired} phantom dialogue turn(s) back into their explicitly "
            "attributed visible character."
        )


def _canonicalize_h3_project_subject_names(
    clip_plans: Sequence[Mapping[str, Any]],
) -> None:
    """Keep stable identities without conflating bounded-sequence cast slots.

    Long-form Director calls are intentionally planned in small independent
    batches.  A local model may use ``char_1`` for Monica in one batch, Pam in
    the next, and Leslie later.  Treating that local slot as a project-global
    identity caused the compiler to relabel every guest as one person and gave
    Ref2VA contradictory identity evidence that could materialize duplicate
    principals.  Rebind only demonstrably reused slots; meaningful IDs and
    explicitly separate same-name characters remain unchanged.
    """

    project_context = " ".join(
        _normalized_space(plan.get("_director_project_context", ""))
        for plan in clip_plans
        if isinstance(plan, Mapping)
    )
    occurrences: list[dict[str, Any]] = []
    identity_groups: list[dict[str, Any]] = []
    groups_by_original_id: dict[str, set[int]] = {}

    def matching_group(identity: str) -> int:
        matches = [
            index for index, group in enumerate(identity_groups)
            if any(
                _h3_identity_names_match(identity, candidate)
                for candidate in group["candidates"]
            )
        ]
        # A one-word shorthand can be ambiguous when two full names share it.
        # Do not merge through that ambiguity.
        if len(matches) == 1:
            return matches[0]
        identity_groups.append({"candidates": [], "occurrences": []})
        return len(identity_groups) - 1

    for plan_index, plan in enumerate(clip_plans):
        for subject_index, subject in enumerate(
            plan.get("_director_subjects_on_screen") or []
        ):
            character_id = _normalized_space(_field(subject, "character_id", ""))
            if not character_id or _h3_is_placeholder_identity(character_id):
                continue
            original_key = character_id.casefold()
            identity, candidates = _h3_subject_identity_evidence(subject)
            if not identity:
                identity = character_id
                candidates = (
                    [character_id]
                    if not _h3_is_placeholder_identity(character_id)
                    else []
                )
            group_index = matching_group(identity)
            group = identity_groups[group_index]
            for candidate in candidates:
                if candidate and not any(
                    _h3_anchor_key(candidate) == _h3_anchor_key(existing)
                    for existing in group["candidates"]
                ):
                    group["candidates"].append(candidate)
            occurrence = {
                "plan_index": plan_index,
                "subject_index": subject_index,
                "subject": subject,
                "original_id": character_id,
                "original_key": original_key,
                "group_index": group_index,
            }
            occurrences.append(occurrence)
            group["occurrences"].append(occurrence)
            groups_by_original_id.setdefault(original_key, set()).add(group_index)

    def score(name: str) -> tuple[int, int, int, int]:
        words = re.findall(r"[A-Za-z0-9'’-]+", name)
        context_key = _h3_anchor_key(project_context)
        first_token_present = bool(
            words
            and re.search(
                rf"(?:^|\s){re.escape(words[0].casefold())}(?:\s|$)",
                context_key,
            )
        )
        return (
            int(_h3_anchor_present(name, project_context) or first_token_present),
            len(words),
            int(not name.isupper()),
            len(name),
        )

    canonical_by_group = {
        index: max(group["candidates"], key=score)
        for index, group in enumerate(identity_groups)
        if group["candidates"]
    }
    unstable_ids = {
        key for key, groups in groups_by_original_id.items()
        if len(groups) > 1
    }
    used_ids = {
        occurrence["original_id"].casefold()
        for occurrence in occurrences
        if occurrence["original_key"] not in unstable_ids
    }
    generated_ids: dict[int, str] = {}
    for occurrence in occurrences:
        if occurrence["original_key"] not in unstable_ids:
            continue
        group_index = occurrence["group_index"]
        if group_index in generated_ids:
            continue
        canonical_name = canonical_by_group.get(group_index, "subject")
        base = f"director_identity_{_h3_identity_slug(canonical_name)}"
        candidate = base
        suffix = 2
        while candidate.casefold() in used_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        generated_ids[group_index] = candidate
        used_ids.add(candidate.casefold())

    plan_rebindings: dict[int, dict[str, set[str]]] = {}
    repaired_occurrences = 0
    for occurrence in occurrences:
        subject = occurrence["subject"]
        group_index = occurrence["group_index"]
        canonical_name = canonical_by_group.get(group_index)
        if canonical_name:
            _set_h3_subject_field(subject, "speaker_name", canonical_name)
        if occurrence["original_key"] not in unstable_ids:
            continue
        target_id = generated_ids[group_index]
        _set_h3_subject_field(subject, "character_id", target_id)
        plan_rebindings.setdefault(
            occurrence["plan_index"], {}
        ).setdefault(occurrence["original_key"], set()).add(target_id)
        repaired_occurrences += 1

    for plan_index, plan in enumerate(clip_plans):
        mappings = plan_rebindings.get(plan_index, {})
        for beat in plan.get("_director_dialogue_beats") or []:
            if not isinstance(beat, MutableMapping):
                continue
            speaker_id = _normalized_space(beat.get("speaker_id", ""))
            targets = mappings.get(speaker_id.casefold(), set())
            if len(targets) == 1:
                beat["speaker_id"] = next(iter(targets))
            elif len(targets) > 1:
                raise H3DialogueContractError(
                    "A long-form shot reused one local character ID for "
                    "multiple visible people, so dialogue ownership is ambiguous."
                )
        if mappings and isinstance(plan, MutableMapping):
            # Any saved registry was keyed by the now-invalid local slots.
            # Rebuild it deterministically from the repaired project cast.
            plan.pop("_director_speaker_registry", None)
            plan["_director_h3_identity_rebound"] = True

    if repaired_occurrences:
        print(
            "[MiniMax H3] Rebound "
            f"{repaired_occurrences} long-form cast occurrence(s) from "
            f"{len(unstable_ids)} reused local character slot(s)."
        )


def _h3_source_requests_multiple_instances(source: Any, name: Any) -> bool:
    """Allow twins/copies only when the source explicitly requests them."""

    text = normalize_h3_text(source)
    identity = _normalized_space(name)
    if not text or not identity:
        return False
    escaped = re.escape(identity)
    quantity = (
        r"(?:two|three|four|five|six|seven|eight|nine|ten|multiple|"
        r"several|many|a\s+pair\s+of|a\s+group\s+of)"
    )
    return bool(re.search(
        rf"(?:{quantity})\s+(?:identical\s+)?(?:copies?\s+of\s+)?"
        rf"{escaped}(?:es|s)?\b|\b{escaped}(?:es|s)?\b[^.!?]{{0,45}}\b"
        r"(?:twins?|clones?|copies|duplicates?|multiple\s+versions?)\b",
        text,
        flags=re.IGNORECASE,
    ))


def _h3_plan_context_anchors(plan: Mapping[str, Any]) -> list[str]:
    anchors: list[str] = []
    for subject in plan.get("_director_subjects_on_screen") or []:
        name = (
            _h3_identity_label(_field(subject, "speaker_name", ""))
            or _h3_identity_label(_field(subject, "character_id", ""))
        )
        folded = name.casefold()
        if (
            name
            and folded not in _H3_GENERIC_IDENTITY_LABELS
            and not _h3_is_placeholder_identity(name)
            and not re.fullmatch(r"(?:char|subject|speaker)[_-]?\d+", folded)
        ):
            anchors.append(name)
    environment = _normalized_space(plan.get("_director_environment", ""))
    if environment:
        # Environment is already Director's concise, shot-specific world ledger.
        # Cap pathological legacy values while retaining named franchise/venue data.
        words = environment.split()
        anchors.append(" ".join(words[:32]).rstrip(" .;:-"))
    return _h3_context_anchors(anchors)


def _build_stable_speaker_registry(
    clip_plans: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Assign speaker IDs once per Director project, not once per shot."""

    registry: dict[str, dict[str, str]] = {}
    used_numbers: set[int] = set()
    for plan in clip_plans:
        existing = plan.get("_director_speaker_registry") or {}
        if not isinstance(existing, Mapping):
            continue
        for raw_key, raw_value in existing.items():
            key = _normalized_space(raw_key).casefold()
            if _h3_is_placeholder_identity(raw_key):
                continue
            entry = _speaker_registry_entry(existing, str(raw_key))
            if not key or not entry or not entry[0]:
                continue
            match = re.fullmatch(r"\(S(\d+)\)", entry[0], re.IGNORECASE)
            if not match:
                continue
            used_numbers.add(int(match.group(1)))
            registry[key] = {
                "stable_id": f"(S{int(match.group(1))})",
                "speaker_name": entry[1],
            }

    next_number = 1
    for plan in clip_plans:
        subjects = plan.get("_director_subjects_on_screen") or []
        for beat in plan.get("_director_dialogue_beats") or []:
            raw_key = _normalized_space(_field(beat, "speaker_id", ""))
            name = _subject_name_for_key(subjects, raw_key)
            if _h3_is_placeholder_identity(raw_key):
                # A placeholder speaker token cannot become a project-wide
                # identity alias. Its native label stays neutral below.
                continue
            key = (raw_key or name).casefold()
            if not key or key in registry:
                continue
            while next_number in used_numbers:
                next_number += 1
            registry[key] = {
                "stable_id": f"(S{next_number})",
                "speaker_name": name,
            }
            used_numbers.add(next_number)
            next_number += 1
    # Existing saved projects may carry a registry compiled before a later
    # shot supplied the character's complete canonical name. The project-wide
    # subject ledger above is authoritative for labels while stable IDs remain.
    for plan in clip_plans:
        for subject in plan.get("_director_subjects_on_screen") or []:
            character_id = _normalized_space(_field(subject, "character_id", ""))
            speaker_name = _h3_identity_label(_field(subject, "speaker_name", ""))
            entry = registry.get(character_id.casefold())
            if entry is not None and speaker_name:
                entry["speaker_name"] = speaker_name
    return registry


_H3_VISUAL_LABELS = (
    "Visible cast",
    "Action",
    "Camera",
    "Lighting",
    "Mood",
    "Final beat",
    "SPEAKER VISIBILITY",
    "By the final beat",
)

_H3_VISUAL_LABEL_RE = re.compile(
    r"(?:^|(?<=[.!?;\n]))\s*(?P<label>"
    + "|".join(re.escape(item) for item in _H3_VISUAL_LABELS)
    + r")\s*:\s*",
    re.IGNORECASE,
)


def _h3_labeled_value(body: str, label: str) -> str:
    # Native Director fields begin sentences. A phrase such as "cinematic
    # action: ..." inside the world description is ordinary prose, not a
    # field delimiter. Use the same boundary for extraction and insertion.
    fields = list(_H3_VISUAL_LABEL_RE.finditer(body))
    for index, field in enumerate(fields):
        if field.group("label").casefold() == label.casefold():
            end = fields[index + 1].start() if index + 1 < len(fields) else len(body)
            return _normalized_space(body[field.end():end])
    return ""


def _compact_h3_visual_body(
    body: str,
    subjects: Sequence[Any],
    registry: Mapping[str, Any],
    *,
    closing_blocking: str = "",
) -> str:
    """Remove planner repetition while retaining H3's highest-value visuals.

    Director's planning JSON intentionally contains redundant continuity data.
    Sending every copy can dilute late dialogue and final-action instructions
    even though H3 now receives the complete sequence. Rebuild the visual body
    from its structured labels so identity, wardrobe, action, camera, and the
    final state each appear once.
    """

    text = _normalized_space(body)
    spans, _ = _dialogue_spans(text)
    if spans:
        text = _normalized_space(_replace_spans(text, spans, [""] * len(spans)))

    label_positions = [match.start() for match in _H3_VISUAL_LABEL_RE.finditer(text)]
    if not label_positions:
        return text

    # A cosmetic token target cannot justify severing an action chain, a
    # camera's destination or the final state. Keep unique field content;
    # the shared Studio fitter removes redundant boilerplate afterward.
    intro = _normalized_space(text[:min(label_positions)])
    cast: list[str] = []
    for subject in subjects or []:
        character_id = _normalized_space(_field(subject, "character_id", ""))
        speaker_name = (
            _h3_identity_label(_field(subject, "speaker_name", ""))
            or _h3_identity_label(character_id)
        )
        entry = _speaker_registry_entry(registry, character_id)
        stable_id = entry[0] if entry else ""
        visual = _normalized_space(_field(subject, "visual_description", ""))
        wardrobe = _normalized_space(_field(subject, "wardrobe", ""))
        position = _normalized_space(_field(subject, "position_or_relation", ""))
        bits = [f"{speaker_name} {stable_id}".strip(), visual]
        if wardrobe:
            bits.append(f"wearing {wardrobe}")
        if position:
            bits.append(position)
        compact_subject = ", ".join(bit for bit in bits if bit)
        if compact_subject:
            cast.append(compact_subject)

    action = _h3_labeled_value(text, "Action")
    camera = _h3_labeled_value(text, "Camera")
    lighting = _h3_labeled_value(text, "Lighting")
    mood = _h3_labeled_value(text, "Mood")
    final = _h3_labeled_value(text, "Final beat")
    if not final:
        final = closing_blocking

    parts = [intro]
    if cast:
        parts.append(f"Cast: {'; '.join(cast)}.")
    if camera:
        parts.append(f"Camera: {camera}.")
    if action:
        parts.append(f"Action: {action}.")
    light_and_mood = "; ".join(item for item in (lighting, mood) if item)
    if light_and_mood:
        parts.append(f"Lighting and mood: {light_and_mood}.")
    if final:
        parts.append(f"Final frame: {final}.")
    result = _normalized_space(" ".join(part for part in parts if part))
    for subject in subjects or []:
        key = _normalized_space(_field(subject, "character_id", ""))
        name = _h3_identity_label(_field(subject, "speaker_name", ""))
        if name and name != key and re.fullmatch(r"(?i)char(?:acter)?[_ -]?\d+", key):
            entry = _speaker_registry_entry(registry, key)
            label = f"{name} {entry[0]}" if entry else name
            result = re.sub(rf"\b{re.escape(key)}\b", lambda _: label, result, flags=re.I)
            result = re.sub(re.escape(label) + rf"\s*\({re.escape(name)}\)", lambda _: label, result, flags=re.I)
    return result


def _compact_h3_line_delivery(value: Any) -> str:
    parts = [
        part.strip(" .")
        for part in str(value or "").split(";")
        if part.strip(" .")
    ]
    if not parts:
        return ""
    # Director joins a project-wide voice bible to a concise line-specific
    # direction with semicolons.  Keep the line direction, not the repeated
    # multi-sentence bible on every turn.
    selected = parts[-2] if len(parts) >= 3 else parts[-1]
    return _normalized_space(selected)


def _h3_dialogue_timing_clause(
    beats: Sequence[Mapping[str, str]],
    duration_seconds: float,
) -> str:
    word_count = sum(len(beat.get("words", "").split()) for beat in beats)
    duration, start, end = h3_dialogue_schedule(word_count, duration_seconds)
    return (
        f"Dialogue timing: mouths stay closed with no human voice from 0.00 "
        f"to {start:.2f} seconds; the tagged lines run once in order from "
        f"about {start:.2f} to {end:.2f} seconds; after {end:.2f} seconds, "
        f"everyone remains silent through {duration:.2f} seconds."
    )


def _insert_h3_vocal_detail(body: str, detail: str) -> str:
    """Keep dialogue ahead of camera/action detail in the token stream."""

    boundary = next((
        field for field in _H3_VISUAL_LABEL_RE.finditer(body)
        if field.group("label").casefold() in {"camera", "action"}
    ), None)
    if boundary:
        return _normalized_space(
            f"{body[:boundary.start()]} {detail} {body[boundary.start():]}"
        )
    return _normalized_space(f"{body} {detail}")


def _ensure_speaker_before_tag(
    body: str,
    tag: str,
    *,
    speaker_name: str,
    stable_id: str,
    cursor: int,
) -> tuple[str, int]:
    tag_index = body.find(tag, cursor)
    if tag_index < 0:
        return body, cursor
    nearby_start = max(0, tag_index - 260)
    nearby = body[nearby_start:tag_index]
    if stable_id in nearby:
        return body, tag_index + len(tag)

    name_matches = list(re.finditer(
        re.escape(speaker_name), nearby, flags=re.IGNORECASE,
    )) if speaker_name else []
    if name_matches:
        insertion = nearby_start + name_matches[-1].end()
        body = f"{body[:insertion]} {stable_id}{body[insertion:]}"
        tag_index += len(stable_id) + 1
    else:
        prefix = f"{speaker_name or 'The visible speaker'} {stable_id} says: "
        body = f"{body[:tag_index]}{prefix}{body[tag_index:]}"
        tag_index += len(prefix)
    return body, tag_index + len(tag)


def _compile_official_dialogue(
    body: str,
    subjects: Sequence[Any],
    dialogue_beats: Sequence[Any],
    registry: Mapping[str, Any],
    existing_blocks: Sequence[str],
    *,
    has_driving_audio: bool = False,
    duration_seconds: float = 0.0,
    music_driven: bool = False,
    vocal_activity: str | None = None,
    project_context: str = "",
    generated_speech_allowed: bool | None = None,
    dialogue_manifest_authoritative: bool = False,
) -> tuple[str, str]:
    """Place exact tagged lines and stable speaker IDs in the visual field."""

    valid_beats: list[dict[str, str]] = []
    authoritative_empty = (
        dialogue_manifest_authoritative and not _has_spoken_beats(dialogue_beats)
    )
    speech_forbidden = generated_speech_allowed is False or authoritative_empty
    for beat in ([] if has_driving_audio or speech_forbidden else (dialogue_beats or [])):
        spoken = normalize_h3_text(_field(beat, "spoken_text", ""))
        if not _normalized_space(spoken):
            continue
        beat_payload = dict(beat) if isinstance(beat, Mapping) else {
            "spoken_text": spoken,
            "language": _field(beat, "language", ""),
        }
        beat_payload["spoken_text"] = spoken
        language, words = _dialogue_payload(beat_payload)
        speaker_key = _normalized_space(_field(beat, "speaker_id", ""))
        entry = _speaker_registry_entry(registry, speaker_key)
        if entry:
            stable_id, speaker_name = entry
        else:
            speaker_name = _subject_name_for_key(subjects, speaker_key)
            stable_id = f"(S{len(valid_beats) + 1})"
        valid_beats.append({
            "words": normalize_h3_text(words),
            "tag": f"<d>[{language}] {normalize_h3_text(words)}</d>",
            "stable_id": stable_id,
            "speaker_name": speaker_name,
            "delivery": _normalized_space(_field(beat, "delivery", "")),
            "physical_cue": _normalized_space(_field(beat, "physical_cue", "")),
        })

    body = _strip_h3_custom_sections(body)
    if has_driving_audio:
        # The mapped audio is authoritative. Transcript annotations are not a
        # second speech-generation request: they are used for timing and
        # performance planning only. Remove any prompt-authored tags and their
        # words while retaining the surrounding visual instructions.
        body = _strip_dialogue_for_driving_audio(body)
        existing_blocks = []
    elif speech_forbidden:
        # Planner-written spans are untrusted when a reviewed audio policy or
        # authoritative empty manifest says that this shot has no generated
        # speech. Remove the tag and an immediately attached ``Kitten says:``
        # lead-in, while leaving visual actions around it intact.
        body = _strip_forbidden_dialogue_blocks(body)
        existing_blocks = []
    spans, malformed = _dialogue_spans(body)
    if valid_beats and spans:
        # A complete top-level block is safely identifiable as dialogue even
        # if it contains duplicated inner tags. Drop that whole block first;
        # recovery below is reserved for an unmatched outer duplicate whose
        # trailing depth-one prose may contain visual action/continuity.
        body = _replace_spans(body, spans, [""] * len(spans))
        spans, malformed = _dialogue_spans(body)
    if malformed and valid_beats:
        # The visual prompt is best-effort prose; a duplicated nested opener
        # cannot override the exact structured dialogue and speaker map. Its
        # inner tagged payload is discarded as untrusted speech, while prose
        # after the inner close remains available as action/continuity.
        if not _has_nested_dialogue_open(body):
            raise H3DialogueContractError(
                "MiniMax H3 dialogue tags are unbalanced and cannot be repaired safely."
            )
        body = _strip_malformed_dialogue_markup_for_structured_beats(
            body,
            [beat["words"] for beat in valid_beats],
        )
        spans, malformed = _dialogue_spans(body)
    if malformed:
        raise H3DialogueContractError(
            "MiniMax H3 dialogue tags are unbalanced and cannot be repaired safely."
        )

    if valid_beats:
        # Structured dialogue is authoritative. Remove any planner-authored
        # copies and insert one concise, prominent sequence so visual prose
        # cannot bury the exact lines and their timing cues.
        body = _replace_spans(body, spans, [""] * len(spans))
        dialogue_parts: list[str] = []
        for beat in valid_beats:
            delivery = _compact_h3_line_delivery(beat["delivery"])
            delivery_text = f" {delivery}" if delivery else ""
            sentence = (
                f"{beat['speaker_name']} {beat['stable_id']} speaks"
                f"{delivery_text}: {beat['tag']}."
            )
            if beat["physical_cue"]:
                cue = _normalized_space(beat["physical_cue"])
                if cue:
                    sentence += f" While speaking, {cue}."
            dialogue_parts.append(sentence)

        timing = _h3_dialogue_timing_clause(valid_beats, duration_seconds)
        guard = (
            "Only the tagged lines are spoken, once each in order. After the "
            "final tagged line, every character remains silent with their "
            "mouth closed; no invented dialogue, muttering, gibberish, "
            "speech-like vocalization, or background voice occurs."
        )
        vocal_detail = (
            f"{timing} Dialogue: {' '.join(dialogue_parts)} {guard} "
            "Keep each current speaker visibly framed with an unobstructed "
            "face and mouth through their complete line."
        )
        body = _insert_h3_vocal_detail(body, vocal_detail)
        contract = " ".join(
            f"{beat['speaker_name']} {beat['stable_id']}: {beat['tag']}"
            for beat in valid_beats
        )
    else:
        canonical_blocks = [h3_dialogue_tag(block) for block in existing_blocks]
        if spans:
            canonical_blocks = [h3_dialogue_tag(body[start:end]) for start, end in spans]
            body = _replace_spans(body, spans, canonical_blocks)
        elif canonical_blocks:
            additions = " ".join(
                f"A visible speaker (S{index}) says: {tag}."
                for index, tag in enumerate(canonical_blocks, start=1)
            )
            body = f"{body} {additions}".strip()
        if canonical_blocks:
            guard = (
                "Only the tagged lines are spoken; everyone remains silent "
                "with their mouth closed at all other times. Generate no "
                "muttering, gibberish, or additional speech-like vocalization."
            )
            if "only the tagged lines are spoken" not in body.casefold():
                body = f"{body} {guard}".strip()
            contract = " ".join(canonical_blocks)
        elif has_driving_audio:
            driver_contract = (
                "Any audible voice or vocal comes only from the mapped driving "
                "audio and remains synchronized to it; do not generate "
                "additional dialogue, gibberish, or speech-like vocalization."
            )
            if "mapped driving audio" not in body.casefold():
                body = _insert_h3_vocal_detail(body, driver_contract)
            if music_driven:
                from .music_performance import music_performance_direction
                direction = music_performance_direction(subjects, vocal_activity, project_context=project_context)
                if direction and direction not in body:
                    body = _insert_h3_vocal_detail(body, direction)
                if direction:
                    driver_contract = f"{driver_contract} {direction}"
            contract = driver_contract
        else:
            silence = (
                "No character speaks; all visible mouths remain closed, and "
                "no muttering, gibberish, speech-like vocalization, or "
                "background voice occurs."
            )
            if not re.search(r"\bno (?:one|character) speaks\b", body, re.IGNORECASE):
                body = _insert_h3_vocal_detail(body, silence)
            contract = silence

    body = _normalized_space(body)
    return body, contract


def _strip_malformed_dialogue_markup_for_structured_beats(
    prompt: str,
    authoritative_lines: Sequence[str],
) -> str:
    """Keep visual prose while discarding malformed dialogue delimiters.

    This recovery is safe only when structured dialogue supplies the exact
    replacement lines and speaker order. Content at a duplicated inner tag
    depth is discarded as untrusted generated speech; text after its close is
    retained as visual prose. Exact copies of authoritative lines in retained
    tagged prose are removed before the canonical lines are reinserted once.
    """

    text = str(prompt or "")
    parts: list[tuple[str, bool, int]] = []
    cursor = 0
    depth = 0
    for token in _H3_DIALOGUE_TOKEN_RE.finditer(text):
        content = text[cursor:token.start()]
        if content:
            marked = depth > 0
            level = depth
            if parts and parts[-1][1:] == (marked, level):
                prior, _, _ = parts[-1]
                parts[-1] = (prior + content, marked, level)
            else:
                parts.append((content, marked, level))
        if not token.group(1):
            depth += 1
        elif depth:
            depth -= 1
        cursor = token.end()
    tail = text[cursor:]
    if tail:
        marked = depth > 0
        level = depth
        if parts and parts[-1][1:] == (marked, level):
            prior, _, _ = parts[-1]
            parts[-1] = (prior + tail, marked, level)
        else:
            parts.append((tail, marked, level))

    cleaned: list[str] = []
    for content, marked, level in parts:
        if marked and level > 1:
            continue
        if not marked:
            cleaned.append(content)
            continue

        had_authoritative_line = False
        for line in authoritative_lines:
            words = _normalized_space(line)
            if not words:
                continue
            phrase = r"\s+".join(re.escape(word) for word in words.split())
            content, count = re.subn(
                rf"(?<!\w){phrase}(?!\w)",
                "",
                content,
            )
            had_authoritative_line = had_authoritative_line or count > 0
        if had_authoritative_line:
            content = re.sub(
                r"^\s*\[[^\]\r\n]+\]\s*",
                "",
                content,
                count=1,
            )
        cleaned.append(content)
    return "".join(cleaned)


def _has_nested_dialogue_open(prompt: str) -> bool:
    """Return whether any dialogue opener occurs before the prior one closes."""

    depth = 0
    for token in _H3_DIALOGUE_TOKEN_RE.finditer(str(prompt or "")):
        if not token.group(1):
            if depth:
                return True
            depth = 1
        elif depth:
            depth -= 1
    return False


def _ref2va_inferred_visible_identity(subject: Any) -> str:
    """Infer a short visible label only when a cast member has no usable name.

    Semantic planner IDs such as ``lead_performer`` are stable keys, not names
    that appear in visual prose. A capitalized identity at the start of the
    subject's description can supply the visible label, but generic role words
    such as ``performer`` are not identities.
    """

    if _h3_identity_label(_field(subject, "speaker_name", "")):
        return ""
    description = _normalized_space(_field(subject, "visual_description", ""))
    opening = re.split(
        r"[,;:(]|\s+(?:from|with|wearing|whose|in|is|stands?|sits?|holds?|"
        r"approach(?:es|ing)?|standing|sitting|holding|posing|looking|walking|"
        r"running|clad)\b",
        description,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .,:;—–-")
    opening = re.sub(
        r"^(?:(?:the|a|an|same|single)\s+)+", "", opening, flags=re.IGNORECASE,
    )
    if not opening or not re.fullmatch(
        r"[A-Z][A-Za-z0-9'’–-]*(?:\s+[A-Z][A-Za-z0-9'’–-]*){0,2}",
        opening,
    ):
        return ""
    label_key = _h3_anchor_key(opening)
    generic_labels = _H3_GENERIC_IDENTITY_LABELS | {
        "actor", "actress", "dancer", "musician", "performer", "singer",
    }
    if (
        label_key in {_h3_anchor_key(label) for label in generic_labels}
        or _H3_LOCAL_CAST_SLOT_RE.fullmatch(opening)
        or _h3_is_placeholder_identity(opening)
    ):
        return ""
    return opening


def _ref2va_compound_identity_alias(value: str) -> str:
    """Read a semantic identity before neutral planner-ID suffixes."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+", value):
        return ""
    parts = re.split(r"[_-]+", value)
    original_count = len(parts)
    while parts and (
        parts[-1].isdigit()
        or parts[-1].casefold() in {"actor", "character", "creature", "entity", "subject"}
    ):
        parts.pop()
    if not parts or len(parts) == original_count:
        return ""
    alias = " ".join(parts)
    return "" if alias.casefold() in {
        "actor", "char", "character", "creature", "entity", "identity", "native", "speaker", "subject",
    } else alias


def _ref2va_identity_aliases(subject: Any) -> list[str]:
    """Read explicit names or a conservative leading identity noun phrase.

    Older music plans have only visual_description ("the Kitten, ...").
    Match word boundaries and require a unique subject; shared traits never
    justify picking the first member of an ensemble.
    """
    aliases = [
        _h3_identity_label(_field(subject, key, ""))
        for key in ("speaker_name", "character_id")
    ]
    inferred = _ref2va_inferred_visible_identity(subject)
    if inferred:
        aliases.append(inferred)
    for name in list(aliases):
        compound_alias = _ref2va_compound_identity_alias(name)
        if compound_alias:
            aliases.append(compound_alias)
        # Qualified names and semantic IDs can share an explicit identity noun
        # with their reference role ("Northern Kingdom Dragon" / "Dragon").
        # Retain the full name; a shared short alias must still resolve uniquely.
        readable_name = re.sub(r"[_-]+", " ", compound_alias or name).strip()
        name_parts = readable_name.split()
        if 1 < len(name_parts) <= 6:
            aliases.append(readable_name)
            noun = name_parts[-1].strip("'’-")
            if re.fullmatch(r"[A-Za-z][A-Za-z'’-]+", noun) and noun.casefold() not in {
                "person", "man", "woman", "child", "boy", "girl", "performer",
                "singer", "subject", "character", "figure", "people", "actor",
                "creature", "entity",
            }:
                aliases.append(noun)
        # Planner names and reference roles commonly differ only by a leading
        # article ("the Kitten" vs "Kitten"). Keep the original alias and add
        # the article-free form so a native Subject ID remains authoritative.
        name_without_article = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I)
        if name_without_article and name_without_article.casefold() != name.casefold():
            aliases.append(name_without_article)
        name_for_parts = name_without_article or name
        if re.fullmatch(
            r"[A-Z][A-Za-z'’-]+(?: [A-Z][A-Za-z'’-]+){1,2}",
            name_for_parts,
        ):
            aliases.extend(
                part for part in name_for_parts.split()
                if part.casefold() not in {"the", "a", "an"}
            )
    description = _normalized_space(_field(subject, "visual_description", ""))
    opening = re.split(
        r"[(:;]|\s+(?:from|with|wearing|whose|in|is|stands?|sits?|holds?|"
        r"approach(?:es|ing)?|standing|sitting|holding|posing|looking|walking|running|clad)\b", description, maxsplit=1, flags=re.I,
    )[0].strip(" ,.;")
    opening = re.sub(r"^(?:(?:the|a|an|same|single)\s+)+", "", opening, flags=re.I)
    if (
        opening
        and len(opening.split()) <= 6
        and not _h3_is_placeholder_identity(opening)
    ):
        aliases.append(opening)
        # A descriptor such as "single warrior kitten" may be called simply
        # "kitten" in the reference label. Generic human nouns are excluded.
        noun = opening.split()[-1].strip("'’-")
        if noun.casefold() not in {
            "person", "man", "woman", "child", "boy", "girl", "performer",
            "singer", "subject", "character", "figure", "people",
        }:
            aliases.append(noun)
    return list(dict.fromkeys(
        value for value in aliases
        if len(value) >= 2
        and _h3_anchor_key(value) not in _H3_NONIDENTITY_REFERENCE_ALIASES
        and not _h3_is_placeholder_identity(value)
    ))


def _mapped_ref2va_subject(role: str, subjects: Sequence[Any]) -> int | None:
    scored: list[tuple[tuple[int, int], int]] = []
    for index, subject in enumerate(subjects):
        explicit = _ref2va_identity_aliases({
            key: _field(subject, key, "") for key in ("speaker_name", "character_id")
        })
        def match_length(aliases: Sequence[str]) -> int:
            return max((len(alias) for alias in aliases if re.search(
                rf"(?<![\w]){re.escape(alias)}(?![\w])", role, re.I,
            )), default=0)
        explicit_length = match_length(explicit)
        inferred_length = match_length(_ref2va_identity_aliases(subject))
        scored.append(((2, explicit_length) if explicit_length else (1, inferred_length), index))
    best = max((score for score, _index in scored), default=(0, 0))
    matches = [index for score, index in scored if best[1] and score == best]
    if len(matches) > 1:
        raise H3DialogueContractError(
            f"Reference identity '{role}' matches multiple visible subjects; "
            "use a distinct character name in the reference and cast list."
        )
    return matches[0] if matches else None


def _ref2va_cast_from_source(
    prompt: str, subjects: Sequence[Any],
) -> tuple[list[Any], dict[int, str]]:
    """Recover reviewed cast definitions when structured cast metadata is absent.

    Explicit Context-IR edits are authoritative. Retaining their numbered
    definitions also retains non-reference actors (e.g. a mounted creature).
    Existing structured cast stays authoritative for legacy compiled copies.
    """
    field = _extract_h3_fields(prompt).get("subject_definitions", "")
    matches = list(re.finditer(r"<Subject\s+(\d+)>\s*(?:\(S\d+\)\s*)?is\s+", field, re.I))
    if not matches:
        return list(subjects), {}
    numbers = [int(match[1]) for match in matches]
    if numbers != list(range(1, len(matches) + 1)):
        raise H3DialogueContractError("Reviewed subject definitions must have unique consecutive subject numbers.")
    recovered: list[Any] = []
    definitions: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(field)
        # Other reference definitions follow the final cast member in a
        # compiled six-field prompt. Rebuild those from the current manifest,
        # rather than swallowing them into that last character definition.
        reference_definition = re.search(
            r"(?:^|(?<=[.!?])\s+|\n)\s*<(?:Picture|Video|Audio)\s+\d+>"
            r"\s+(?:is|provides|supplies)\b", field[match.end():end], re.I,
        )
        if reference_definition:
            end = match.end() + reference_definition.start()
        definition = field[match.start():end].strip()
        description = field[match.end():end].strip()
        subject = {"visual_description": description}
        aliases = _ref2va_identity_aliases(subject)
        # Keep descriptions out of display names; full reviewed prose below
        # remains the authority for identity, wardrobe, and reference binding.
        if aliases:
            subject["speaker_name"] = _leading_h3_identity(description) or aliases[-1]
        recovered.append(subject)
        definitions[index + 1] = definition
    if subjects:
        if len(subjects) > len(recovered):
            raise H3DialogueContractError("Reviewed subject definitions omit a visible cast member.")
        for index, subject in enumerate(subjects):
            aliases = _ref2va_identity_aliases(subject)
            if aliases and not any(re.search(
                rf"(?<![\w]){re.escape(alias)}(?![\w])",
                str(recovered[index]["visual_description"]), re.I,
            ) for alias in aliases):
                raise H3DialogueContractError(
                    f"Reviewed <Subject {index + 1}> contradicts the saved cast; "
                    "update the cast identity together with the prompt."
                )
            recovered[index] = subject
    return recovered, definitions


def _collapse_ref2va_compiler_identity_duplicates(
    prompt: str, references: Sequence[Mapping[str, Any]] | None,
) -> str:
    """Repair only redundant identity definitions in the compiler's template.

    Older native prompts can contain both a semantic cast ID and an appended
    reference-only definition for that same identity. Rich authored cast
    definitions and ambiguous identities remain untouched.
    """
    field = _extract_h3_fields(prompt).get("subject_definitions", "")
    if not field:
        return prompt
    _cast, definitions = _ref2va_cast_from_source(prompt, [])
    template_pattern = re.compile(
        r"<Subject\s+\d+>\s+is\s+(.+?),\s+whose\s+facial,\s+bodily,\s+"
        r"and\s+character\s+identity\s+come\s+from\s+<Picture\s+(\d+)>\.", re.I,
    )
    identity_references: list[tuple[int, str]] = []
    if references:
        picture_no = 0
        for reference in references:
            if str(reference.get("type") or "").casefold() != "image":
                continue
            picture_no += 1
            if str(reference.get("image_intent") or "identity").casefold() == "identity":
                identity_references.append((picture_no, _trim_sentence(reference.get("role", ""))))
    else:
        # The first rerun preflight precedes concrete asset resolution. Exact
        # compiler-owned definitions already prove their picture/name mapping.
        identity_references = [
            (int(match[2]), match[1]) for definition in definitions.values()
            if (match := template_pattern.fullmatch(definition))
        ]
    replacements: dict[int, int] = {}
    for picture_no, role in identity_references:
        role_name = re.sub(r"^(?:the|a|an)\s+", "", role, flags=re.I)
        if not role_name or _h3_source_requests_multiple_instances(prompt, role_name):
            continue
        # Fail closed when the authored text could describe distinct instances,
        # including plural nouns and a second/another named identity.
        escaped_role = re.escape(role_name)
        if re.search(
            rf"\b{escaped_role}(?:s|es)\b|\b(?:another|second|additional|different)\s+"
            rf"(?:[A-Za-z'-]+\s+){{0,3}}{escaped_role}\b", prompt, re.I,
        ):
            continue
        role_key = _h3_anchor_key(role_name)
        generated: list[int] = []
        existing: list[int] = []
        for number, definition in definitions.items():
            template = template_pattern.fullmatch(definition)
            if template:
                template_name = re.sub(r"^(?:the|a|an)\s+", "", template[1], flags=re.I)
                if int(template[2]) == picture_no and _h3_anchor_key(template_name) == role_key:
                    generated.append(number)
                continue
            label = re.match(
                r"<Subject\s+\d+>\s*(?:\(S\d+\)\s*)?is\s+([^:;,\n]+)",
                definition, re.I,
            )
            if not label:
                continue
            identity = label[1].strip(" .")
            pictures = {int(value) for value in re.findall(r"<Picture\s+(\d+)>", definition, re.I)}
            if pictures - {picture_no}:
                continue
            identity = re.sub(r"^(?:the|a|an)\s+", "", identity, flags=re.I)
            identity = _ref2va_compound_identity_alias(identity) or identity
            readable_identity = re.sub(r"[_-]+", " ", identity).strip()
            if _h3_anchor_key(readable_identity) == role_key or re.search(
                rf"(?<![\w]){re.escape(role_name)}$", readable_identity, re.I,
            ):
                existing.append(number)
        if len(generated) == 1 and len(existing) == 1:
            replacements[generated[0]] = existing[0]
    if not replacements:
        return prompt
    surviving = [number for number in definitions if number not in replacements]
    renumbered = {number: index + 1 for index, number in enumerate(surviving)}
    cleaned_field = field
    for number in replacements:
        cleaned_field = cleaned_field.replace(definitions[number], "", 1)
    cleaned = prompt.replace(field, cleaned_field.strip(), 1)
    return re.sub(
        r"<Subject\s+(\d+)>",
        lambda match: f"<Subject {renumbered.get(replacements.get(int(match[1]), int(match[1])), int(match[1]))}>",
        cleaned, flags=re.I,
    )


def _reference_relationships(
    references: Sequence[Mapping[str, Any]] | None,
    subjects: Sequence[Any] | None = None,
    registry: Mapping[str, Any] | None = None,
    source_text: str = "",
) -> tuple[
    list[str],
    list[str],
    bool,
    dict[int, list[str]],
    list[str],
    list[str],
]:
    """Build MiniMax's documented Ref2VA reference contract.

    Visual and audio retention markers intentionally use different fixed
    vocabularies. Identity/location pictures are bound through ``<Subject N>``
    instead of being misrepresented as concrete keyframes.
    """

    definitions: list[str] = []
    retention: list[str] = []
    subject_sources: dict[int, list[str]] = {}
    detail_bindings: list[str] = []
    task_types: list[str] = ["reference generation"]
    picture_no = video_no = audio_no = 0
    has_driving_audio = False
    next_subject_no = len(subjects or []) + 1
    registry = registry if isinstance(registry, Mapping) else {}

    def mapped_subject(role: str) -> tuple[int, str, str] | None:
        matched_index = _mapped_ref2va_subject(role, subjects or [])
        if matched_index is None:
            return None
        subject = (subjects or [])[matched_index]
        character_id = _normalized_space(_field(subject, "character_id", ""))
        aliases = _ref2va_identity_aliases(subject)
        subject_name = (
            _h3_identity_label(_field(subject, "speaker_name", ""))
            or _ref2va_inferred_visible_identity(subject)
            or _h3_identity_label(character_id)
            or (aliases[0] if aliases else f"visible subject {matched_index + 1}")
        )
        entry = _speaker_registry_entry(registry, character_id or subject_name)
        return matched_index + 1, subject_name, entry[0] if entry else ""

    def add_task_type(value: str) -> None:
        if value not in task_types:
            task_types.append(value)

    for reference in references or []:
        kind = str(reference.get("type") or "").strip().lower()
        role = _trim_sentence(reference.get("role", "")) or f"the supplied {kind} reference"
        subject_match = (
            mapped_subject(role)
            if kind != "image" or str(reference.get("image_intent") or "identity").lower() == "identity"
            else None
        )
        mapped_role = (
            f"<Subject {subject_match[0]}> ({subject_match[1]})"
            if subject_match else role
        )
        if kind == "image":
            picture_no += 1
            label = f"<Picture {picture_no}>"
            intent = str(reference.get("image_intent") or "identity").lower()
            if intent == "object":
                from models.minimax_h3.reference_manifest import object_reference_prompt_contract

                definition, analysis = object_reference_prompt_contract(label, role)
                definitions.append(definition)
                retention.append(analysis)
                detail_bindings.append(
                    f"The requested object uses {label}'s design and appearance with "
                    "the count, scale, placement, and action described in the target scene."
                )
                continue
            if intent == "composition":
                definitions.append(
                    f"{label} is the soft composition and cast-layout anchor "
                    f"for [Shot 1], showing {mapped_role}."
                )
                retention.append(
                    f"{label} ([Shot 1] composition anchor): partially_preserved - "
                    "retain the intended subject placement, wardrobe, setting, "
                    "and spatial relationships while generating natural motion "
                    "rather than a frozen opening frame."
                )
                detail_bindings.append(
                    f"{label} softly guides the opening composition and cast layout."
                )
                continue

            if intent in {"scene", "style"}:
                subject_match = None
            if subject_match:
                subject_no, subject_name, _ = subject_match
            else:
                subject_no = next_subject_no
                next_subject_no += 1
                aliases = _ref2va_identity_aliases({"visual_description": role})
                subject_name = aliases[0] if intent == "identity" and aliases else role

            if intent == "scene":
                source_clause = f"environment and location identity come from {label}"
                explanation = (
                    f"preserve the architecture, materials, lighting context, and "
                    f"location identity supplied by {label}, but not incidental characters"
                )
                marker = "fully_preserved"
                detail_bindings.append(
                    f"<Subject {subject_no}> is the environment established by {label}."
                )
            elif intent == "style":
                source_clause = f"visual style is guided by {label}"
                explanation = (
                    f"retain broad similarity to the medium, palette, lighting "
                    f"language, and texture of {label}, but not its characters, pose, "
                    "framing, or exact composition"
                )
                marker = "weak_reference"
                detail_bindings.append(
                    f"The visual treatment of <Subject {subject_no}> follows {label} broadly."
                )
            else:
                source_clause = (
                    f"facial, bodily, and character identity come from {label}"
                )
                allow_multiple = _h3_source_requests_multiple_instances(
                    source_text,
                    subject_name,
                )
                uniqueness = (
                    " This mapping is exactly one physical instance of the "
                    "character; do not create a second copy, clone, reflection, "
                    "portrait, screen image, background likeness, or literal "
                    "reference-image cutaway."
                    if not allow_multiple else ""
                )
                explanation = (
                    f"preserve the facial, bodily, and character identity supplied "
                    f"by {label}; use it for identity only, follow the target "
                    "shot's explicitly described wardrobe and lighting, and do "
                    "not copy its background, source location, framing, "
                    "composition, pose, source lighting, or opening-still appearance"
                    f"{uniqueness}"
                )
                marker = "fully_preserved"
                detail_bindings.append(
                    f"<Subject {subject_no}> uses facial, bodily, and character "
                    f"identity from {label} as the same single character already "
                    "described in this shot, never as inserted source footage or "
                    "an additional character."
                    if not allow_multiple else
                    f"<Subject {subject_no}> uses facial, bodily, and character "
                    f"identity from {label} for the explicitly requested multiple "
                    "instances."
                )

            if subject_match:
                subject_sources.setdefault(subject_no, []).append(source_clause)
            else:
                definitions.append(
                    f"<Subject {subject_no}> is {subject_name}, whose {source_clause}."
                )
            retention.append(
                f"<Subject {subject_no}> (appears in [Shot 1]): {marker} - {explanation}."
            )
        elif kind == "video":
            video_no += 1
            label = f"<Video {video_no}>"
            definitions.append(f"{label} provides motion and temporal reference for {mapped_role}.")
            retention.append(
                f"{label} (motion, camera, and temporal structure): weak_reference - "
                "retain only the requested motion, timing, camera, or scene traits "
                "while generating the described target video."
            )
            detail_bindings.append(
                f"The requested motion and temporal behavior follow {label} without copying it as source footage."
            )
        elif kind == "audio":
            audio_no += 1
            label = f"<Audio {audio_no}>"
            intent = str(reference.get("audio_intent") or "voice").lower()
            if intent == "drive":
                has_driving_audio = True
                add_task_type("audio reuse")
                definitions.append(
                    f"{label} is the performance-driving audio timeline for {mapped_role}."
                )
                retention.append(
                    f"{label}: partially_copy - reuse its audible content and "
                    "timing while synchronizing visible action and lip movement; "
                    "additional scene ambience or practical effects may be mixed around it."
                )
                detail_bindings.append(
                    f"Visible performance and lip movement remain synchronized to {label} throughout [Shot 1]."
                )
            elif intent == "style":
                add_task_type("audio reference")
                definitions.append(
                    f"{label} is the audio-style reference for {mapped_role}."
                )
                retention.append(
                    f"{label}: weak_reference - retain broad similarity to its "
                    "rhythm, texture, and style without copying its words, exact "
                    "timing, or waveform."
                )
                detail_bindings.append(
                    f"The requested audio treatment broadly follows {label}."
                )
            elif intent == "sound":
                add_task_type("audio reference")
                definitions.append(
                    f"{label} is a reusable sound-effect reference for {role}."
                )
                retention.append(
                    f"{label}: reference - retain the sound effect's timbre and "
                    "texture without copying its waveform or original timing."
                )
                detail_bindings.append(
                    f"Generate the requested matching sound effects using {label}'s "
                    "timbre and texture, synchronized to the visible actions in this window."
                )
            else:
                add_task_type("audio reference")
                speaker_suffix = ""
                if subject_match and subject_match[2]:
                    speaker_suffix = f" {subject_match[2]}"
                audio_target = (
                    f"<Subject {subject_match[0]}>"
                    if subject_match else mapped_role
                )
                definitions.append(
                    f"{label} is the voice-timbre reference for "
                    f"{audio_target}{speaker_suffix}."
                )
                retention.append(
                    f"{label}: reference - the target speaker follows its voice "
                    "timbre, emotion, and delivery without copying the source "
                    "words, timing, or waveform."
                )
                detail_bindings.append(
                    f"Newly scripted dialogue for {audio_target}{speaker_suffix} follows the voice timbre and delivery of {label}."
                )
    return (
        definitions,
        retention,
        has_driving_audio,
        subject_sources,
        detail_bindings,
        task_types,
    )


def _ref2va_subject_definitions(
    subjects: Sequence[Any],
    registry: Mapping[str, Any],
    source_bindings: Mapping[int, Sequence[str]] | None = None,
) -> list[str]:
    definitions: list[str] = []
    source_bindings = source_bindings or {}
    for index, subject in enumerate(subjects or [], start=1):
        character_id = _normalized_space(_field(subject, "character_id", ""))
        aliases = _ref2va_identity_aliases(subject)
        name = (
            _h3_identity_label(_field(subject, "speaker_name", ""))
            or _ref2va_inferred_visible_identity(subject)
            or _h3_identity_label(character_id)
            or next(iter(aliases), f"visible subject {index}")
        )
        entry = _speaker_registry_entry(registry, character_id or name)
        speaker = f" {entry[0]}" if entry and entry[0] else ""
        description = _trim_sentence(_field(subject, "visual_description", ""))
        wardrobe = _trim_sentence(_field(subject, "wardrobe", ""))
        details = "; ".join(
            item for item in (
                description,
                f"wearing {wardrobe}" if wardrobe else "",
                *source_bindings.get(index, ()),
            ) if item
        )
        definitions.append(
            f"<Subject {index}> is {name}{speaker}" + (f": {details}." if details else ".")
        )
    return definitions


def _label_ref2va_subjects_in_body(
    body: str,
    subjects: Sequence[Any],
) -> str:
    """Insert each official subject label at its first named appearance."""

    result = body
    missing: list[str] = []
    for index, subject in enumerate(subjects or [], start=1):
        label = f"<Subject {index}>"
        if label in result:
            continue
        name = (
            _h3_identity_label(_field(subject, "speaker_name", ""))
            or _ref2va_inferred_visible_identity(subject)
            or _h3_identity_label(_field(subject, "character_id", ""))
        )
        if (
            not _h3_identity_label(_field(subject, "speaker_name", ""))
            and not _ref2va_inferred_visible_identity(subject)
        ):
            # Semantic IDs such as lead_performer need not occur in natural
            # visual prose. Bind a unique visible identity alias instead.
            for alias in _ref2va_identity_aliases(subject):
                if not re.search(rf"(?<![\w>]){re.escape(alias)}(?![\w<])", result, re.I):
                    continue
                try:
                    matched = _mapped_ref2va_subject(alias, subjects)
                except H3DialogueContractError:
                    continue
                if matched == index - 1:
                    name = alias
                    break
        if not name:
            continue
        pattern = re.compile(rf"(?<![\w>]){re.escape(name)}(?![\w<])", re.IGNORECASE)
        result, count = pattern.subn(f"{label} ({name})", result, count=1)
        if not count:
            missing.append(f"{label} ({name}) is visible in the described blocking.")
    if missing:
        shot = re.search(r"\[Shot\s+1\]", result, flags=re.IGNORECASE)
        if shot:
            result = (
                f"{result[:shot.end()]} {' '.join(missing)} "
                f"{result[shot.end():].lstrip()}"
            )
        else:
            result = f"{' '.join(missing)} {result}".strip()
    return result


def _split_ref2va_style_opening(body: str) -> tuple[str, str]:
    """Separate or synthesize MiniMax's pre-[Shot 1] style opening."""

    text = _normalized_space(body)
    shot = re.search(r"\[Shot\s+1\]", text, flags=re.IGNORECASE)
    timed_first_shot = bool(
        shot and _H3_TIMED_SHOT_MARKER_RE.match(text[shot.start():])
    )
    if shot and text[:shot.start()].strip():
        opening = _normalized_space(text[:shot.start()])
        # Keep a real timestamped first marker in the timeline. It may be
        # needed to distinguish a malformed [Shot 1], [Shot 3] sequence from
        # a valid local sequence after the compiler adds its own Shot 1.
        timeline = text[shot.start() if timed_first_shot else shot.end():].strip()
        return opening.rstrip(" .") + ".", timeline
    if shot and not timed_first_shot:
        text = text[shot.end():].strip()

    # Free-form prose after a Lighting/Mood label can include the whole action
    # timeline. Echoing that span here performed source transitions twice.
    # Keep authored lighting and mood in the body, with a neutral style opening.
    opening = (
        "The target video maintains the requested visual style, lighting, "
        "color, and cinematic texture."
    )
    return opening, text


def _alignment_header(
    mode: str,
    duration_seconds: float,
    final_shot_number: int = 1,
) -> str:
    duration = max(0.0, float(duration_seconds or 0.0))
    final_shot_number = max(1, int(final_shot_number or 1))
    if mode == "i2va":
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if mode == "fl2va":
        return (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
            f"target video; Picture 2 (from Shot {final_shot_number}) aligns with the "
            f"{duration:.2f}-second "
            "mark of the target video."
        )
    if mode == "l2va":
        return (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {final_shot_number}]) aligns with the "
            f"{duration:.2f}-second mark of the target video."
        )
    return ""


def compile_h3_official_prompt(
    prompt: str,
    subjects: Sequence[Any] | None,
    dialogue_beats: Sequence[Any] | None,
    *,
    mode: str = "t2va",
    duration_seconds: float = 0.0,
    references: Sequence[Mapping[str, Any]] | None = None,
    speaker_registry: Mapping[str, Any] | None = None,
    project_context: str = "",
    context_anchors: Sequence[str] | None = None,
    visual_style: str = "",
    opening_blocking: str = "",
    closing_blocking: str = "",
    audio_plan: Mapping[str, Any] | None = None,
    audio_policy: Mapping[str, Any] | None = None,
    dialogue_manifest_authoritative: bool = False,
    strict_audio_policy: bool = False,
) -> tuple[str, str]:
    """Compile Director metadata into MiniMax's documented Context-IR shape."""

    mode = str(mode or "t2va").strip().lower()
    if mode not in {"t2va", "i2va", "fl2va", "l2va", "ref2va"}:
        raise H3DialogueContractError(f"Unknown MiniMax H3 prompt mode: {mode}")
    registry = speaker_registry if isinstance(speaker_registry, Mapping) else {}
    reviewed_definitions: dict[int, str] = {}
    if mode == "ref2va":
        prompt = _collapse_ref2va_compiler_identity_duplicates(prompt, references)
        subjects, reviewed_definitions = _ref2va_cast_from_source(prompt, subjects or [])
    (
        reference_definitions,
        retention,
        has_driving_audio,
        subject_sources,
        detail_bindings,
        task_types,
    ) = _reference_relationships(
        references if mode == "ref2va" else None,
        subjects or [],
        registry,
        source_text=f"{project_context}\n{prompt}",
    )
    audio_mode = _normalized_space(_field(audio_plan or {}, "mode", "")).casefold()
    if audio_mode == "music_driven":
        from .music_performance import constrain_music_performance
        activity = _field(audio_plan or {}, "vocal_activity", None)
        prompt = constrain_music_performance(
            prompt, subjects, activity, project_context=project_context,
        )
        opening_blocking = constrain_music_performance(
            opening_blocking, subjects, activity, project_context=project_context,
        )
        closing_blocking = constrain_music_performance(
            closing_blocking, subjects, activity, project_context=project_context,
        )
    if has_driving_audio or audio_mode in {"audio_driven", "music_driven"}:
        # Supplied driving audio owns every audible voice. Structured beats in
        # an audio-driven plan are transcript annotations for timing and
        # performance only; they must not become a second generated-speech
        # request. Story-driven plans without source audio keep their exact
        # structured dialogue as the spoken authority.
        dialogue_beats = []
    if audio_mode in {"audio_driven", "music_driven"}:
        # Initial Director preflight runs before concrete Ref2VA manifests are
        # assembled. The shot's audio plan still proves that a mapped source
        # track owns the vocals, so do not temporarily treat it as a silent or
        # prompt-scripted shot.
        has_driving_audio = True
    if strict_audio_policy:
        violations = _audio_policy_violations(
            prompt,
            audio_policy,
            dialogue_beats=dialogue_beats,
            dialogue_manifest_authoritative=dialogue_manifest_authoritative,
            has_driving_audio=has_driving_audio,
        )
        if violations:
            raise H3DialogueContractError(
                "Audio policy violation: " + "; ".join(violations)
            )
    if (
        _audio_policy_flag(audio_policy, "generated_speech_allowed") is False
        or (
            dialogue_manifest_authoritative
            and not _has_spoken_beats(dialogue_beats)
        )
    ) and not has_driving_audio:
        # The policy controls generated speech, not a supplied driving track.
        # In a visual-only shot, an empty authoritative manifest means there
        # are no exact lines available for the planner to add or promote.
        dialogue_beats = []
    body, soundscape, music, existing_blocks = _source_prompt_parts(
        prompt,
        project_context=project_context,
        context_anchors=context_anchors,
        opening_blocking=opening_blocking,
        closing_blocking=closing_blocking,
        audio_plan=audio_plan,
        audio_policy=audio_policy,
        has_driving_audio=has_driving_audio,
    )
    style_text = _normalized_space(str(visual_style or "")).rstrip(" .")
    if style_text:
        style_direction = (
            "Visual style across the entire frame: "
            f"{style_text}. Apply this style consistently to every subject, "
            "the environment and background, and all linework, shading, "
            "color treatment, and rendering."
        )
        if _normalized_space(style_direction).casefold() not in _normalized_space(body).casefold():
            # Ref2VA treats prose before its first shot marker as the visual-
            # style opening. Give this explicit direction that same placement
            # while keeping it out of the immutable source prompt. Other H3
            # modes retain it as an introduction to their complete timeline.
            if mode == "ref2va" and not re.search(
                r"\[Shot\s+1\]", body, flags=re.IGNORECASE,
            ):
                body = f"{style_direction} [Shot 1] {body}".strip()
            else:
                body = f"{style_direction} {body}".strip()
    speech_forbidden = (
        _audio_policy_flag(audio_policy, "generated_speech_allowed") is False
        or (
            dialogue_manifest_authoritative
            and not _has_spoken_beats(dialogue_beats)
        )
    )
    if speech_forbidden and not has_driving_audio:
        # Compacting and Ref2VA subject labeling can remove tags before the
        # vocal compiler sees them. Strip the attached planner lead-in first so
        # a remnant such as ``The Kitten says:`` cannot survive as an instruction.
        body = _strip_forbidden_dialogue_blocks(body)
        existing_blocks = []
    if mode == "ref2va":
        style_opening, body = _split_ref2va_style_opening(body)
        # Bind visual prose before inserting compiler-owned vocal directions.
        # Otherwise the inserted labels alter those exact canonical sentences
        # and a second preflight can no longer remove/rebuild them once.
        body = _label_ref2va_subjects_in_body(body, subjects or [])
        body, vocal_contract = _compile_official_dialogue(
            body,
            subjects or [],
            dialogue_beats or [],
            registry,
            existing_blocks,
            has_driving_audio=has_driving_audio,
            duration_seconds=duration_seconds,
            music_driven=audio_mode == "music_driven",
            vocal_activity=_field(audio_plan or {}, "vocal_activity", None),
            project_context=project_context,
            generated_speech_allowed=_audio_policy_flag(
                audio_policy, "generated_speech_allowed",
            ),
            dialogue_manifest_authoritative=dialogue_manifest_authoritative,
        )
        if not _H3_TIMED_SHOT_MARKER_RE.match(body.lstrip()):
            body = re.sub(r"^\s*\[Shot\s+1\]\s*", "", body, flags=re.IGNORECASE)
        body = f"[Shot 1] {body}".strip()
        subject_definitions = _ref2va_subject_definitions(
            subjects or [],
            registry,
            subject_sources,
        )
        for index, reviewed in reviewed_definitions.items():
            missing_sources = [
                clause for clause in subject_sources.get(index, ())
                if any(label not in reviewed for label in re.findall(r"<Picture \d+>", clause))
            ]
            subject_definitions[index - 1] = reviewed.rstrip(" .") + (
                "; " + "; ".join(missing_sources) if missing_sources else ""
            ) + "."
        definitions = "\n".join([*subject_definitions, *reference_definitions])
        if not definitions:
            definitions = "Use the explicitly described subjects and target scene."
        retention_text = "\n".join(retention) or (
            "Preserve the explicitly described identities, wardrobe, setting, action, and audio roles."
        )
        summary_body = re.sub(r"^\[Shot\s+1\]\s*", "", body, flags=re.IGNORECASE)
        summary_body = _H3_STRICT_DIALOGUE_RE.sub(
            "scripted dialogue",
            summary_body,
        )
        summary = _trim_sentence(re.split(r"(?<=[.!?])\s+", summary_body, maxsplit=1)[0])
        if len(summary) > 320:
            summary = summary[:320].rsplit(" ", 1)[0].rstrip(" ,;:-") + "..."
        summary = (
            f"[{' + '.join(task_types)}] "
            f"{summary or 'A complete audiovisual shot matching the mapped subjects and requested action.'}"
        )
        if detail_bindings:
            body = re.sub(r"^\[Shot\s+1\]\s*", "", body, flags=re.IGNORECASE)
            body = f"[Shot 1] {' '.join(detail_bindings)} {body}".strip()
        # The style opening may already contain the world anchor inserted
        # during source recovery. Check the complete visual field so moving
        # that opening does not cause a second copy to be added before it.
        body = f"{style_opening} {body}".strip()
        body = _ensure_h3_context_anchors(body, context_anchors or [])
        body = _normalize_h3_compiled_visual_shots(body)
        if has_driving_audio and music == "N/A":
            music = "Use the mapped driving audio according to retention_analysis."
        compiled = (
            f"subject_definitions: {definitions}\n\n"
            f"summary: {summary}\n\n"
            f"retention_analysis: {retention_text}\n\n"
            f"detailed_description: {body}\n\n"
            f"overall_soundscape: {soundscape}.\n\n"
            f"non_diegetic_music: {music}"
        )
    else:
        compact_body = _compact_h3_visual_body(
            body, subjects or [], registry, closing_blocking=closing_blocking,
        )
        compact_body = _ensure_h3_context_anchors(compact_body, context_anchors or [])
        compiled_body, vocal_contract = _compile_official_dialogue(
            compact_body, subjects or [], dialogue_beats or [], registry, existing_blocks,
            has_driving_audio=has_driving_audio, duration_seconds=duration_seconds,
            music_driven=audio_mode == "music_driven",
            vocal_activity=_field(audio_plan or {}, "vocal_activity", None),
            project_context=project_context,
            generated_speech_allowed=_audio_policy_flag(
                audio_policy, "generated_speech_allowed",
            ),
            dialogue_manifest_authoritative=dialogue_manifest_authoritative,
        )
        if not _H3_TIMED_SHOT_MARKER_RE.match(compiled_body.lstrip()):
            compiled_body = re.sub(
                r"^\s*\[Shot\s+1\]\s*",
                "",
                compiled_body,
                flags=re.IGNORECASE,
            )
        compiled_body = f"[Shot 1] {compiled_body}".strip()
        compiled_body = _normalize_h3_compiled_visual_shots(compiled_body)
        shot_numbers = [
            int(marker.group("number"))
            for marker in _h3_visual_shot_markers(compiled_body)
        ]
        header = _alignment_header(mode, duration_seconds, max(shot_numbers) if shot_numbers else 1)
        compiled = (
            f"integrated_multimodal_description: {compiled_body}\n\n"
            f"overall_soundscape: {_normalized_space(soundscape)}.\n\n"
            f"non_diegetic_music: {_normalized_space(music)}"
        )
        if header:
            compiled = f"{header}\n\n{compiled}"
        initial_tokens = final_tokens = h3_prompt_token_count(compiled)
        if final_tokens > _H3_DIRECTOR_TEXT_TOKEN_BUDGET:
            # Give unusually dense Director prompts one final structure-aware
            # quality pass. Exact dialogue, timing, and first/final states are
            # protected; if the target cannot be reached safely the complete
            # prompt remains valid and is sent to H3 intact.
            fitted = fit_h3_base_prompt(
                compiled,
                target_tokens=_H3_DIRECTOR_TEXT_TOKEN_BUDGET,
            )
            compiled = fitted.prompt
            final_tokens = fitted.token_count
        if final_tokens < initial_tokens:
            print(
                "[MiniMax H3] Compacted Director prompt from "
                f"{initial_tokens} to {final_tokens} text tokens for clearer "
                "instruction adherence."
            )
        elif final_tokens > _H3_DIRECTOR_TEXT_TOKEN_BUDGET:
            print(
                "[MiniMax H3] Preserving complete Director prompt at "
                f"{final_tokens} text tokens; the quality target is not a "
                "generation limit."
            )
    compiled = normalize_h3_text(compiled).strip()
    policy_errors = _audio_policy_violations(
        compiled,
        audio_policy,
        dialogue_beats=dialogue_beats,
        dialogue_manifest_authoritative=dialogue_manifest_authoritative,
        has_driving_audio=has_driving_audio,
    )
    if policy_errors:
        raise H3DialogueContractError(
            "Invalid MiniMax H3 audio policy: " + "; ".join(policy_errors)
        )
    return compiled, vocal_contract


def validate_final_h3_subject_bindings(
    prompt: str,
    expected_bindings: Mapping[int, int | None] | None = None,
) -> list[str]:
    """Check final Ref2VA picture ownership by subject number.

    ``expected_bindings`` maps each identity picture number to its expected
    subject number. Use ``None`` when the reference has no safe existing-cast
    match; that picture must still bind to exactly one subject. This pure
    check can also be called at the final render boundary after prompt review.
    """

    definitions_text = _extract_h3_fields(str(prompt or "")).get(
        "subject_definitions", "",
    )
    definitions = list(re.finditer(
        r"<Subject\s+(\d+)>\s*(?:\(S\d+\)\s*)?is\s+",
        definitions_text,
        re.I,
    ))
    owners_by_picture: dict[int, list[int]] = {}
    for index, match in enumerate(definitions):
        end = (
            definitions[index + 1].start()
            if index + 1 < len(definitions)
            else len(definitions_text)
        )
        body = definitions_text[match.end():end]
        subject_number = int(match[1])
        picture_numbers = {
            int(value) for value in re.findall(r"<Picture\s+(\d+)>", body, re.I)
        }
        for picture_number in picture_numbers:
            owners_by_picture.setdefault(picture_number, []).append(subject_number)

    expected = (
        expected_bindings
        if expected_bindings is not None
        else {picture_number: None for picture_number in owners_by_picture}
    )
    errors: list[str] = []
    for raw_picture_number, raw_subject_number in expected.items():
        picture_number = int(raw_picture_number)
        label = f"<Picture {picture_number}>"
        owners = owners_by_picture.get(picture_number, [])
        if raw_subject_number is None:
            if len(owners) != 1:
                errors.append(
                    f"Identity reference {label} must have exactly one subject binding; "
                    f"found {owners}"
                )
            continue
        expected_subject_number = int(raw_subject_number)
        if owners != [expected_subject_number]:
            errors.append(
                f"Identity reference {label} must bind only to "
                f"<Subject {expected_subject_number}>; found subject bindings {owners}"
            )
    return errors


def validate_h3_prompt_contract(
    prompt: str,
    dialogue_beats: Sequence[Any] | None = None,
    *,
    mode: str = "t2va",
    references: Sequence[Mapping[str, Any]] | None = None,
    subjects: Sequence[Any] | None = None,
    context_anchors: Sequence[str] | None = None,
    duration_seconds: float = 0.0,
    audio_policy: Mapping[str, Any] | None = None,
    dialogue_manifest_authoritative: bool = False,
    has_driving_audio: bool = False,
) -> list[str]:
    """Validate the official field order plus Maestro's exact dialogue data."""

    text = str(prompt or "")
    mode = str(mode or "t2va").strip().lower()
    expected = _H3_REF2VA_FIELDS if mode == "ref2va" else _H3_BASE_FIELDS
    errors = validate_h3_vocal_contract(text, dialogue_beats)
    has_driving_audio = has_driving_audio or any(
        str(_field(reference, "type", "")).strip().casefold() == "audio"
        and str(_field(reference, "audio_intent", "voice") or "voice")
        .strip().casefold() == "drive"
        for reference in references or []
    )
    errors.extend(_audio_policy_violations(
        text,
        audio_policy,
        dialogue_beats=dialogue_beats,
        dialogue_manifest_authoritative=dialogue_manifest_authoritative,
        has_driving_audio=has_driving_audio,
    ))
    positions: list[int] = []
    for field in expected:
        matches = list(re.finditer(
            rf"(?mi)^\s*{re.escape(field)}\s*:", text,
        ))
        if len(matches) != 1:
            errors.append(f"expected one {field} field, found {len(matches)}")
        elif matches:
            positions.append(matches[0].start())
    if len(positions) == len(expected) and positions != sorted(positions):
        errors.append("Context-IR fields are out of order")
    unexpected = set(_H3_ALL_FIELDS) - set(expected)
    for field in unexpected:
        if re.search(rf"(?mi)^\s*{re.escape(field)}\s*:", text):
            errors.append(f"unexpected {field} field for {mode}")
    extracted_fields = _extract_h3_fields(text)
    visual_field = "detailed_description" if mode == "ref2va" else "integrated_multimodal_description"
    visual = extracted_fields.get(visual_field, "")
    # Reference bindings mention an existing shot (for example, "throughout
    # [Shot 1]"); those mentions do not introduce a camera cut. Legacy inline
    # Context-IR can also repeat the current marker before more action prose.
    # Collapse that continuation, while rejecting skipped/global shot indices
    # or a return to an earlier shot after a later cut.
    shot_numbers: list[int] = []
    visual_markers = _h3_visual_shot_markers(visual)
    for marker in visual_markers:
        number = int(marker.group("number"))
        if not shot_numbers or number != shot_numbers[-1]:
            shot_numbers.append(number)
    if shot_numbers and shot_numbers != list(range(1, len(shot_numbers) + 1)):
        errors.append("visual shot numbers must be consecutive within this clip, starting at [Shot 1]")
    if duration_seconds > 0:
        from .prompt_timing import music_video_visual_timing_issue
        timing_issue = music_video_visual_timing_issue(visual, duration_seconds)
        if timing_issue:
            errors.append(timing_issue)
    for field in ("overall_soundscape", "non_diegetic_music"):
        if re.search(r"\[Shot\s+\d+\]", extracted_fields.get(field, ""), re.I):
            errors.append(f"{field} contains visual shot instructions")
    if mode == "ref2va":
        if not shot_numbers or shot_numbers[0] != 1:
            errors.append("detailed_description is missing [Shot 1]")
        elif not visual[:visual_markers[0].start()].strip():
            errors.append(
                "detailed_description is missing the visual-style opening before [Shot 1]"
            )
    elif not re.match(r"^\s*\[Shot\s+1\]", visual, flags=re.IGNORECASE):
        errors.append(f"{visual_field} does not begin with [Shot 1]")
    if re.search(
        r"\b(?:PROJECT CONTINUITY|OPENING CONTINUITY|FINAL BLOCKING|"
        r"DIALOGUE AND VOCAL PERFORMANCE|SILENCE AND VOCAL PERFORMANCE)\s*:",
        text,
        flags=re.IGNORECASE,
    ):
        errors.append("legacy Maestro prompt wrapper remains in the H3 payload")
    if any(0x80 <= ord(character) <= 0x9F for character in text):
        errors.append("prompt contains an orphaned C1 control character")
    required_anchors = list(context_anchors or [])
    for subject in subjects or []:
        name = (
            _h3_identity_label(_field(subject, "speaker_name", ""))
            or _h3_identity_label(_field(subject, "character_id", ""))
        )
        folded = name.casefold()
        if (
            name
            and folded not in _H3_GENERIC_IDENTITY_LABELS
            and not _h3_is_placeholder_identity(name)
            and not re.fullmatch(r"(?:char|subject|speaker)[_-]?\d+", folded)
        ):
            required_anchors.append(name)
    for anchor in _h3_context_anchors(required_anchors):
        if not _h3_anchor_present(anchor, text):
            errors.append(f"missing canonical identity/world context: {anchor}")
    if mode == "i2va" and not text.startswith(
        "For the target video, at 0.00 seconds into the target video, <Picture 1>"
    ):
        errors.append("I2VA prompt is missing the official 0.00-second alignment line")
    if mode == "fl2va" and not text.startswith(
        "How the reference pictures align with the target video"
    ):
        errors.append("FL2VA prompt is missing the official two-picture alignment line")
    if mode == "l2va" and not text.startswith(
        "How the reference pictures align with the target video — <Picture 1>"
    ):
        errors.append("L2VA prompt is missing the official last-picture alignment line")
    if mode == "t2va" and text.startswith((
        "For the target video,", "How the reference pictures align",
    )):
        errors.append("T2VA prompt must not contain a picture-alignment header")
    if mode == "ref2va":
        summary = extracted_fields.get("summary", "")
        if not re.match(
            r"^\[(?:reference generation|video editing|video continuation|"
            r"keyframe completion|audio reuse|audio reference)(?:\s+\+\s+"
            r"(?:reference generation|video editing|video continuation|"
            r"keyframe completion|audio reuse|audio reference))*\]\s+\S",
            summary,
            flags=re.IGNORECASE,
        ):
            errors.append("Ref2VA summary is missing its official task-type prefix")

        definitions_text = extracted_fields.get("subject_definitions", "")
        definitions = list(re.finditer(r"<Subject\s+(\d+)>\s*(?:\(S\d+\)\s*)?is\s+", definitions_text, re.I))
        numbers = [int(match[1]) for match in definitions]
        if len(numbers) != len(set(numbers)):
            errors.append("Ref2VA defines the same subject number more than once")
        undefined = set(int(value) for value in re.findall(r"<Subject\s+(\d+)>", visual, re.I)) - set(numbers)
        if undefined:
            errors.append("Ref2VA visual action uses undefined subjects: " + ", ".join(str(number) for number in sorted(undefined)))
        picture_no = 0
        expected_picture_bindings: dict[int, int | None] = {}
        for reference in references or []:
            if str(reference.get("type") or "").lower() != "image":
                continue
            picture_no += 1
            if str(reference.get("image_intent") or "identity").lower() != "identity":
                continue
            try:
                mapped = _mapped_ref2va_subject(str(reference.get("role") or ""), subjects or [])
            except H3DialogueContractError as exc:
                errors.append(str(exc))
                continue
            expected_picture_bindings[picture_no] = (
                mapped + 1 if mapped is not None else None
            )
        errors.extend(validate_final_h3_subject_bindings(
            text,
            expected_picture_bindings,
        ))

        if references:
            expected_labels: list[str] = []
            counts = {"image": 0, "video": 0, "audio": 0}
            for reference in references:
                kind = str(reference.get("type") or "").strip().lower()
                if kind not in counts:
                    continue
                counts[kind] += 1
                noun = {"image": "Picture", "video": "Video", "audio": "Audio"}[kind]
                expected_labels.append(f"<{noun} {counts[kind]}>")
            missing_labels = [label for label in expected_labels if label not in text]
            if missing_labels:
                errors.append(
                    "Ref2VA prompt does not map supplied references: "
                    + ", ".join(missing_labels)
                )

            retention_text = extracted_fields.get("retention_analysis", "")
            if counts["image"] + counts["video"] and not re.search(
                r":\s*(?:fully_preserved|partially_preserved|"
                r"attribute_transfer|weak_reference)\s+-",
                retention_text,
            ):
                errors.append("Ref2VA visual retention uses no official marker")
            if counts["audio"] and not re.search(
                r":\s*(?:fully_copy|partially_copy|reference|weak_reference)\s+-",
                retention_text,
            ):
                errors.append("Ref2VA audio retention uses no official marker")
    return list(dict.fromkeys(errors))


def _retime_h3_visual_timecodes(
    prompt: str,
    authored_duration: Any,
    native_duration: Any,
) -> str:
    """Scale visual-local timecodes when native fitting changes clip length.

    The planner's cue is first validated against its authored duration. Only
    eligible visual ``At MM:SS`` cues are then scaled; dialogue and explicitly
    labeled source-audio timing stay byte-for-byte intact.
    """

    from . import prompt_timing

    try:
        authored_seconds = float(authored_duration)
        native_seconds = float(native_duration)
    except (TypeError, ValueError, OverflowError):
        return prompt
    if (
        not math.isfinite(authored_seconds)
        or authored_seconds <= 0
        or not math.isfinite(native_seconds)
        or native_seconds <= 0
    ):
        return prompt

    authored_issue = prompt_timing.music_video_visual_timing_issue(
        prompt, authored_seconds
    )
    if authored_issue:
        raise H3DialogueContractError(
            f"authored visual timing is invalid: {authored_issue}"
        )
    if math.isclose(authored_seconds, native_seconds, rel_tol=0.0, abs_tol=1e-9):
        return prompt

    masked_prompt = prompt_timing._mask_dialogue_tags(prompt)
    ratio = native_seconds / authored_seconds
    replacements: list[tuple[int, int, str]] = []
    for match in prompt_timing._AT_TIME_RE.finditer(masked_prompt):
        if prompt_timing._is_source_timing_reference(
            masked_prompt, match.start()
        ):
            continue
        token = match.group("value")
        timecode = token.strip("([{<").rstrip(".,:)]}>")
        if not prompt_timing._TIME_LIKE_RE.match(timecode):
            continue
        parsed = prompt_timing._LOCAL_TIMECODE_RE.fullmatch(timecode)
        # The authored-duration validator above rejects malformed candidates.
        if not parsed:
            continue
        seconds_field = int(parsed.group("seconds"))
        fraction = parsed.group("fraction") or ""
        visual_seconds = (
            int(parsed.group("minutes")) * 60
            + seconds_field
            + (
                int(fraction) / (10 ** len(fraction))
                if fraction else 0.0
            )
        )
        prefix_length = len(token) - len(token.lstrip("([{<"))
        suffix_length = len(token) - len(token.rstrip(".,:)]}>"))
        suffix = token[len(token) - suffix_length:] if suffix_length else ""
        replacement = (
            token[:prefix_length]
            + prompt_timing._format_local_timecode(visual_seconds * ratio)
            + suffix
        )
        replacements.append((*match.span("value"), replacement))

    if not replacements:
        return prompt
    retimed = prompt
    for start, end, replacement in reversed(replacements):
        retimed = retimed[:start] + replacement + retimed[end:]
    return retimed


def compile_h3_clip_plans(
    clip_plans: Sequence[MutableMapping[str, Any]],
    *,
    prompt_modes: Sequence[str] | None = None,
    durations: Sequence[float] | None = None,
    reference_manifests: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> Sequence[MutableMapping[str, Any]]:
    """Compile every saved/renderable H3 clip immediately before queuing.

    ``_director_h3_source_prompt`` remains the immutable planner result. This
    lets generation recompile a prompt from T2VA to I2VA/FL2VA after the actual
    start/end files are known without nesting an alignment header or six-field
    Ref2VA wrapper around a previously compiled prompt.
    """

    references_by_plan = [
        (
            reference_manifests[index]
            if reference_manifests is not None and index < len(reference_manifests)
            else plan.get("_director_h3_reference_manifest") or []
        )
        for index, plan in enumerate(clip_plans)
    ]

    # Audio transcript beats remain useful identity and continuity metadata.
    # Track separately which beats are actual generated dialogue so prompt
    # compilation and final validation never treat a source transcript as a
    # second speech request.
    transcript_only = []
    for plan, references in zip(clip_plans, references_by_plan):
        audio_mode = _normalized_space(
            _field(plan.get("_director_audio_plan") or {}, "mode", "")
        ).casefold()
        has_driving_audio_reference = any(
            str(_field(reference, "type", "")).strip().casefold() == "audio"
            and str(_field(reference, "audio_intent", "voice") or "voice")
            .strip().casefold() == "drive"
            for reference in references or []
        )
        transcript_only.append(
            audio_mode in {"audio_driven", "music_driven"}
            or has_driving_audio_reference
        )

    for index, (plan, has_driving_audio) in enumerate(zip(clip_plans, transcript_only)):
        audio_policy = plan.get("_director_audio_policy")
        dialogue_manifest_authoritative = bool(
            plan.get("_director_dialogue_manifest_authoritative")
        )
        beats = plan.get("_director_dialogue_beats") or []
        source = str(
            plan.get("_director_h3_source_prompt") or plan.get("video_prompt") or ""
        )
        if plan.get("_director_prompt_user_edited"):
            violations = _audio_policy_violations(
                source,
                audio_policy,
                dialogue_beats=beats,
                dialogue_manifest_authoritative=dialogue_manifest_authoritative,
                has_driving_audio=has_driving_audio,
            )
            if violations:
                raise H3DialogueContractError(
                    f"Shot {index + 1}: Audio policy violation: "
                    + "; ".join(violations)
                )

        speech_forbidden = (
            _audio_policy_flag(audio_policy, "generated_speech_allowed") is False
            or (dialogue_manifest_authoritative and not _has_spoken_beats(beats))
        )
        if speech_forbidden and not has_driving_audio:
            # Empty authoritative source dialogue must not leave planner-added
            # beat metadata or lip-sync timing active in the saved clip plan.
            plan["_director_dialogue_beats"] = []
            audio_plan = plan.get("_director_audio_plan")
            if isinstance(audio_plan, MutableMapping):
                if _normalized_space(_field(audio_plan, "mode", "")).casefold() == "dialogue_driven":
                    audio_plan["mode"] = "ambient_only"
                audio_plan["timing_anchor"] = "video"
                audio_plan["lip_sync_critical"] = False

    for index, (plan, references) in enumerate(zip(clip_plans, references_by_plan)):
        source = str(plan.get("_director_h3_source_prompt") or plan.get("video_prompt") or "")
        current = str(plan.get("video_prompt") or "")
        previous = str(plan.get("_director_h3_compiled_prompt") or "")
        if source and previous and current != previous:
            source = current
        mode = (
            prompt_modes[index]
            if prompt_modes is not None and index < len(prompt_modes)
            else plan.get("_director_h3_prompt_mode") or plan.get("_director_h3_model_family") or "t2va"
        )
        repaired = (
            _collapse_ref2va_compiler_identity_duplicates(source, references)
            if str(mode).strip().casefold() == "ref2va" else source
        )
        if repaired != source:
            plan["_director_h3_source_prompt"] = repaired
            if current == source:
                plan["video_prompt"] = repaired
            plan["_director_h3_compiled_prompt"] = ""
        if plan.get("_director_prompt_user_edited"):
            source = str(plan.get("_director_h3_source_prompt") or plan.get("video_prompt") or "")
            if _extract_h3_fields(source).get("subject_definitions"):
                recovered, _definitions = _ref2va_cast_from_source(source, [])
                if recovered:
                    previous_subjects = plan.get("_director_subjects_on_screen") or []
                    for subject in recovered:
                        try:
                            previous = _mapped_ref2va_subject(subject.get("speaker_name", ""), previous_subjects)
                        except H3DialogueContractError:
                            previous = None
                        if previous is not None:
                            role = _field(previous_subjects[previous], "performance_role", "")
                            if role:
                                subject["performance_role"] = role
                    plan["_director_subjects_on_screen"] = recovered
                    plan["_director_opening_blocking"] = ""
                    plan["_director_closing_blocking"] = ""
                    plan.pop("_director_speaker_registry", None)

    _repair_h3_phantom_dialogue_subjects(clip_plans)
    _canonicalize_h3_project_subject_names(clip_plans)
    registry = _build_stable_speaker_registry(clip_plans)
    for index, plan in enumerate(clip_plans):
        beats = plan.get("_director_dialogue_beats") or []
        for beat in beats:
            if isinstance(beat, MutableMapping) and "spoken_text" in beat:
                beat["spoken_text"] = normalize_h3_text(beat["spoken_text"])

        current_prompt = str(plan.get("video_prompt", "") or "")
        last_compiled = str(plan.get("_director_h3_compiled_prompt", "") or "")
        source_prompt = plan.get("_director_h3_source_prompt")
        if source_prompt and last_compiled and current_prompt != last_compiled:
            # Prompt review/editing happens after the first preflight. Treat a
            # changed compiled prompt as the user's new authoritative source.
            source_prompt = current_prompt
            plan["_director_h3_source_prompt"] = source_prompt
        if not source_prompt:
            source_prompt = current_prompt
            plan["_director_h3_source_prompt"] = source_prompt
        mode = (
            prompt_modes[index]
            if prompt_modes is not None and index < len(prompt_modes)
            else plan.get("_director_h3_prompt_mode")
            or (
                "ref2va"
                if plan.get("_director_h3_model_family") == "ref2va"
                else "t2va"
            )
        )
        duration = (
            durations[index]
            if durations is not None and index < len(durations)
            else plan.get("_director_duration_sec") or 0.0
        )
        authored_duration = plan.get("_director_authored_duration_sec")
        if plan.get("_director_prompt_user_edited"):
            # Edited prompt text is authored against the current native clip
            # duration, so it must remain strict instead of inheriting the
            # planner's pre-fit timing scale.
            authored_duration = duration
        try:
            compile_source_prompt = (
                _retime_h3_visual_timecodes(
                    str(source_prompt), authored_duration, duration
                )
                if authored_duration is not None
                else str(source_prompt)
            )
        except H3DialogueContractError as error:
            raise H3DialogueContractError(
                f"Shot {index + 1}: {error}"
            ) from error
        references = references_by_plan[index]
        speech_beats = [] if transcript_only[index] else beats
        audio_policy = plan.get("_director_audio_policy")
        dialogue_manifest_authoritative = bool(
            plan.get("_director_dialogue_manifest_authoritative")
        )
        subjects = plan.get("_director_subjects_on_screen") or []
        subject_descriptions = (
            _h3_subjects_for_physical_state(
                subjects,
                plan.get("_director_physical_state"),
            )
            if str(mode).strip().casefold() == "ref2va"
            else subjects
        )
        context_anchors = _h3_plan_context_anchors(plan)
        plan["_director_required_context_anchors"] = context_anchors
        prompt, contract = compile_h3_official_prompt(
            compile_source_prompt,
            subject_descriptions or [],
            speech_beats,
            mode=mode,
            duration_seconds=duration,
            references=references,
            speaker_registry=registry,
            project_context=plan.get("_director_project_context", ""),
            context_anchors=context_anchors,
            # Reviewed prompt text is authoritative after a user edit; stale
            # planner metadata must not reimpose the earlier art direction.
            visual_style=(
                ""
                if plan.get("_director_prompt_user_edited")
                else plan.get("_director_visual_style", "")
            ),
            opening_blocking=plan.get("_director_opening_blocking", ""),
            closing_blocking=plan.get("_director_closing_blocking", ""),
            audio_plan=plan.get("_director_audio_plan") or {},
            audio_policy=audio_policy,
            dialogue_manifest_authoritative=dialogue_manifest_authoritative,
            strict_audio_policy=bool(plan.get("_director_prompt_user_edited")),
        )
        plan["video_prompt"] = prompt
        plan["_director_h3_compiled_prompt"] = prompt
        plan["_director_vocal_contract"] = contract
        plan["_director_h3_prompt_mode"] = mode
        plan["_director_speaker_registry"] = registry
        errors = validate_h3_prompt_contract(
            prompt,
            speech_beats,
            mode=mode,
            references=references,
            subjects=subject_descriptions or [],
            context_anchors=context_anchors,
            duration_seconds=duration,
            audio_policy=audio_policy,
            dialogue_manifest_authoritative=dialogue_manifest_authoritative,
            has_driving_audio=transcript_only[index],
        )
        if errors:
            from services.enhancement_diagnostics import record_rejection
            record_rejection(errors, stage=f"Director H3 shot {index + 1}")
            raise H3DialogueContractError(
                f"Shot {index + 1}: " + "; ".join(errors)
            )
    return clip_plans
