"""Bounded long-form speech writing for native TTS prompt enhancement.

Runtime is estimated from script words at 130 words per minute. This is a
planning estimate only; generated speech duration depends on the voice and
delivery. Qualitative requests for a long story/script default to about ten
minutes unless the brief supplies an explicit shorter or longer target.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable


WORDS_PER_MINUTE = 130
MAX_TARGET_MINUTES = 60
MAX_TARGET_WORDS = WORDS_PER_MINUTE * MAX_TARGET_MINUTES
MAX_WRITER_INSTRUCTIONS = 8_000
LONG_FORM_WORD_THRESHOLD = 900
DEFAULT_QUALITATIVE_LONG_MINUTES = 10
SECTION_TARGET_WORDS = 650
MAX_GENERATION_CALLS = 30
MAX_PLAN_CHARS = 6_000
MAX_PRIOR_TEXT_CHARS = 3_600
MAX_PLAN_TOKENS = 768
MAX_SECTION_TOKENS = 1_280


@dataclass(frozen=True)
class SpeechTarget:
    """A bounded word/runtime target inferred from settings or the brief."""

    minutes: float = 0.0
    words: int = 0
    source: str = ""
    capped: bool = False

    @property
    def is_long_form(self) -> bool:
        return self.words > LONG_FORM_WORD_THRESHOLD


_DURATION_ONES = "one|two|three|four|five|six|seven|eight|nine"
_DURATION_TEENS = (
    "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    "eighteen|nineteen"
)
_DURATION_UNITS = rf"(?:{_DURATION_ONES}|{_DURATION_TEENS})"
_DURATION_TENS = "twenty|thirty|forty|fifty"
_DURATION_NUMBER_WORD = (
    rf"(?:{_DURATION_UNITS}|(?:{_DURATION_TENS})(?:[- ](?:{_DURATION_ONES}))?|sixty)"
)
_DURATION_NUMBER = rf"(?<![\w-])(?P<value>\d+(?:\.\d+)?|{_DURATION_NUMBER_WORD})"
_DURATION_WORD_VALUES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60,
}
_DURATION_UNIT = r"(?:minutes?|mins?)"
_DURATION_MEDIA = r"(?:story|dialogue|conversation|monologue|script|speech|narration)"
_DURATION_PATTERNS = (
    # "10-minute story", "10 minutes of dialogue", or "10 minutes long".
    re.compile(
        rf"\b{_DURATION_NUMBER}\s*\+?\s*(?:[- ]\s*)?{_DURATION_UNIT}"
        rf"\s*(?:long\b|of\s+(?:spoken\s+)?{_DURATION_MEDIA}\b|{_DURATION_MEDIA}\b)",
        re.IGNORECASE,
    ),
    # "story for/about 10 minutes", "dialogue lasting 10 minutes".
    re.compile(
        rf"\b{_DURATION_MEDIA}\b\s+(?:(?:that|which)\s+)?"
        rf"(?:for|about|around|in|approximately|roughly|lasting|lasts?(?:\s+for)?|running|runs?(?:\s+for)?)\s+"
        rf"(?:about\s+|around\s+|roughly\s+)?{_DURATION_NUMBER}\s*\+?\s*{_DURATION_UNIT}\b"
        rf"(?!\s+ago)",
        re.IGNORECASE,
    ),
    # Direct length controls such as "make it 10 minutes" or "runtime: 10 min".
    re.compile(
        rf"\b(?:make\s+it|target(?:\s+length)?|aim(?:\s+for)?|duration(?:\s+of)?|runtime(?:\s+of)?)\s*[:=]?\s*"
        rf"(?:about\s+|around\s+|roughly\s+)?{_DURATION_NUMBER}\s*\+?\s*{_DURATION_UNIT}\b",
        re.IGNORECASE,
    ),
    # "about 10 minutes of speech" and an explicit writer command.
    re.compile(
        rf"\b(?:write|create|produce|tell)\s+(?:for\s+)?(?:about\s+|around\s+|roughly\s+)?"
        rf"{_DURATION_NUMBER}\s*\+?\s*{_DURATION_UNIT}\s+of\s+{_DURATION_MEDIA}\b",
        re.IGNORECASE,
    ),
)
_HOUR_PATTERNS = (
    re.compile(
        rf"\b{_DURATION_NUMBER}\s*[- ]?\s*hours?\s*(?:long\b|{_DURATION_MEDIA}\b)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_DURATION_MEDIA}\b\s+(?:for|about|around|in|lasting|lasts?(?:\s+for)?)\s+"
        rf"(?:about\s+|around\s+)?{_DURATION_NUMBER}\s*hours?\b(?!\s+ago)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:make\s+it|target(?:\s+length)?|aim(?:\s+for)?|duration(?:\s+of)?|runtime(?:\s+of)?)\s*[:=]?\s*"
        rf"(?:about\s+|around\s+)?{_DURATION_NUMBER}\s*hours?\b",
        re.IGNORECASE,
    ),
)
_SECOND_PATTERNS = (
    re.compile(
        rf"\b{_DURATION_NUMBER}\s*[- ]?\s*seconds?\s*(?:long\b|{_DURATION_MEDIA}\b)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_DURATION_MEDIA}\b\s+(?:for|about|around|lasting|lasts?(?:\s+for)?)\s+"
        rf"(?:about\s+|around\s+)?{_DURATION_NUMBER}\s*seconds?\b(?!\s+ago)",
        re.IGNORECASE,
    ),
)
_WORD_COUNT_RE = re.compile(
    r"\bword\s+count\s*[:=]\s*(?P<value>\d[\d,]*)\b",
    re.IGNORECASE,
)
_DIRECT_WORDS_RE = re.compile(
    r"\b(?:write|create|produce|aim(?:ing)?\s+for|target(?:ing)?|around|about|"
    r"approximately|approx\.?|exactly|roughly|at\s+least|up\s+to)\s+"
    r"(?P<value>\d[\d,]*)\s*(?:spoken\s+)?words?\b",
    re.IGNORECASE,
)
_HYPHENATED_WORDS_RE = re.compile(
    rf"\b(?P<value>\d[\d,]*)[- ]word\s+(?:{_DURATION_MEDIA})\b", re.IGNORECASE
)
_QUALITATIVE_LONG_RE = re.compile(
    r"\b(?:long(?:[ -]form)?|extended|full[ -]length)\s+"
    r"(?:form\s+)?(?:story|dialogue|conversation|monologue|script|speech)\b",
    re.IGNORECASE,
)
_QUALITATIVE_SHORT_RE = re.compile(
    r"\b(?:short|brief|concise)\s+(?:story|dialogue|conversation|monologue|script|speech)\b"
    r"|\b(?:keep|make|write|produce|aim(?:ing)?\s+for)\b[^.!?\n]{0,48}"
    r"\b(?:short|brief|concise)\b",
    re.IGNORECASE,
)
_PART_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:part|chapter|section)\s+"
    r"(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"(?:\b.*)?$",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\b[\w][\w'’\-]*\b", re.UNICODE)


def _parse_number(match: re.Match | None) -> float:
    if not match:
        return 0.0
    raw_value = match.group("value").strip().casefold()
    parts = raw_value.replace("-", " ").split()
    if parts and parts[0] in _DURATION_WORD_VALUES:
        value = float(_DURATION_WORD_VALUES[parts[0]])
        if len(parts) == 2 and parts[1] in _DURATION_WORD_VALUES and value % 10 == 0:
            value += _DURATION_WORD_VALUES[parts[1]]
        elif len(parts) != 1:
            return 0.0
        return value
    try:
        value = float(raw_value.replace(",", ""))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def _target_from_minutes(minutes: float, source: str) -> SpeechTarget:
    capped = minutes > MAX_TARGET_MINUTES
    minutes = min(float(minutes), float(MAX_TARGET_MINUTES))
    words = max(1, int(round(minutes * WORDS_PER_MINUTE)))
    return SpeechTarget(minutes=minutes, words=words, source=source, capped=capped)


def resolve_speech_target(prompt: str, requested_minutes: object = 0) -> SpeechTarget:
    """Resolve explicit targets and qualitative long-form requests.

    A positive setting wins. With a zero setting, explicit minutes take
    precedence over explicit word counts. A qualitative "long story/dialogue"
    request defaults to ten minutes. Clear short/brief requests keep the
    ordinary one-call behavior unless they also contain an explicit size.
    Every inferred target is capped at sixty minutes.
    """
    try:
        configured = float(requested_minutes or 0)
    except (TypeError, ValueError):
        configured = 0.0
    if math.isfinite(configured) and configured > 0:
        return _target_from_minutes(configured, "setting")

    source_text = str(prompt or "")
    minute_match = next(
        (match for pattern in _DURATION_PATTERNS if (match := pattern.search(source_text))),
        None,
    )
    minutes = _parse_number(minute_match)
    if minutes:
        return _target_from_minutes(minutes, "prompt_minutes")

    hour_match = next(
        (match for pattern in _HOUR_PATTERNS if (match := pattern.search(source_text))),
        None,
    )
    hours = _parse_number(hour_match)
    if hours:
        return _target_from_minutes(hours * 60, "prompt_hours")

    seconds_match = next(
        (match for pattern in _SECOND_PATTERNS if (match := pattern.search(source_text))),
        None,
    )
    seconds = _parse_number(seconds_match)
    if seconds:
        return _target_from_minutes(seconds / 60, "prompt_seconds")

    word_match = (
        _WORD_COUNT_RE.search(source_text)
        or _DIRECT_WORDS_RE.search(source_text)
        or _HYPHENATED_WORDS_RE.search(source_text)
    )
    word_count = int(_parse_number(word_match)) if word_match else 0
    if word_count:
        capped = word_count > MAX_TARGET_WORDS
        words = min(word_count, MAX_TARGET_WORDS)
        return SpeechTarget(
            minutes=words / WORDS_PER_MINUTE,
            words=words,
            source="prompt_words",
            capped=capped,
        )

    if _QUALITATIVE_SHORT_RE.search(source_text):
        return SpeechTarget()
    if _QUALITATIVE_LONG_RE.search(source_text):
        return _target_from_minutes(
            DEFAULT_QUALITATIVE_LONG_MINUTES, "prompt_qualitative_long"
        )
    return SpeechTarget()


def bound_writer_instructions(instructions: object) -> tuple[str, bool]:
    """Return bounded style guidance and whether it was shortened."""
    value = str(instructions or "").strip()
    if len(value) <= MAX_WRITER_INSTRUCTIONS:
        return value, False
    return value[:MAX_WRITER_INSTRUCTIONS].rstrip(), True


def record_speech_writer_warning(message: str) -> None:
    """Attach a non-blocking warning to the current enhancement request."""
    try:
        from services.studio_enhancement import record_review_warning

        record_review_warning(str(message))
    except Exception as error:
        # A diagnostic failure must never discard a usable script draft.
        print(f"[SpeechWriter] Could not record enhancement warning: {error}")


def _check_cancelled() -> None:
    try:
        from services.studio_enhancement import check_cancelled

        check_cancelled()
    except ImportError:
        return


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def _without_part_headings(text: str) -> str:
    lines = [
        line.rstrip()
        for line in str(text or "").splitlines()
        if not _PART_HEADING_RE.match(line.strip())
    ]
    return "\n".join(lines).strip()


def _remove_repeated_prefix(previous: str, continuation: str) -> str:
    """Remove exact word overlap when a model repeats its continuation cue."""
    previous_tokens = list(re.finditer(r"\S+", previous or ""))
    next_tokens = list(re.finditer(r"\S+", continuation or ""))
    if not previous_tokens or not next_tokens:
        return str(continuation or "").strip()

    def normalize(token: str) -> str:
        return re.sub(r"[^\w'’\-]", "", token).casefold()

    previous_words = [normalize(match.group()) for match in previous_tokens[-120:]]
    next_words = [normalize(match.group()) for match in next_tokens]
    max_overlap = min(len(previous_words), len(next_words))
    for overlap in range(max_overlap, 2, -1):
        if previous_words[-overlap:] == next_words[:overlap]:
            return continuation[next_tokens[overlap - 1].end():].lstrip()
    return str(continuation or "").strip()


def _format_minutes(minutes: float) -> str:
    return f"{minutes:g}"


def generate_long_form_script(
    *,
    generate: Callable[..., str],
    prompt: str,
    system_prompt: str,
    target: SpeechTarget,
    max_new_tokens: int,
    temperature: float,
    is_truncated: Callable[[], bool] | None = None,
) -> str:
    """Generate one assembled TTS script through a bounded plan and sections.

    The planner plus section drafts and one continuation per section stay
    below thirty calls for the sixty-minute cap. Previous spoken text supplied
    as context is limited to a short rolling tail, while the complete source
    brief and bounded plan remain available to every section.
    """
    target_words = min(max(1, int(target.words)), MAX_TARGET_WORDS)
    target_minutes = min(max(0.01, float(target.minutes)), float(MAX_TARGET_MINUTES))
    try:
        requested_token_limit = int(max_new_tokens)
    except (TypeError, ValueError):
        requested_token_limit = MAX_SECTION_TOKENS
    if requested_token_limit <= 0:
        requested_token_limit = MAX_SECTION_TOKENS
    per_call_token_limit = min(requested_token_limit, MAX_SECTION_TOKENS)
    section_count = max(1, math.ceil(target_words / SECTION_TARGET_WORDS))
    section_words = math.ceil(target_words / section_count)
    calls = 0

    def call(call_prompt: str, call_system: str, token_budget: int) -> tuple[str, bool]:
        nonlocal calls
        if calls >= MAX_GENERATION_CALLS:
            return "", False
        _check_cancelled()
        calls += 1
        output = str(generate(
            prompt=call_prompt,
            system_prompt=call_system,
            max_new_tokens=max(1, min(int(token_budget), per_call_token_limit)),
            temperature=temperature,
            image_paths=None,
            enable_thinking=False,
            thinking_budget=0,
        ) or "")
        return output, bool(is_truncated and is_truncated())

    plan_prompt = (
        "Prepare private planning notes for a complete spoken script. This planning response will not be "
        "included in the spoken output. Preserve the user's premise, named characters, requested language, "
        "and all explicit facts. Establish a small cast and stable voices, a causal beginning-to-ending arc, "
        f"and {section_count} chronological section beats of about {section_words} spoken words each. "
        "Give each section a distinct beat and handoff; reserve a satisfying resolution for the final section. "
        "Do not write the dialogue or narration itself, and do not put chapter or part headings in the final script.\n\n"
        f"Approximate target: {target_words} spoken words, or {target_minutes:g} minutes at about "
        f"{WORDS_PER_MINUTE} words per minute.\n\nBrief:\n{prompt}"
    )
    planner_system = (
        f"{system_prompt}\n\n"
        "PLANNING CALL ONLY: Return concise private planning notes, not spoken lines or narration. "
        "The final script will follow the model-native TTS output format and speaker conventions."
    )
    section_system = system_prompt.replace(
        "Output only the complete requested spoken script.",
        "Output only the requested spoken-script portion.",
    )
    try:
        plan, plan_truncated = call(plan_prompt, planner_system, MAX_PLAN_TOKENS)
    except InterruptedError:
        raise
    except Exception:
        record_speech_writer_warning(
            "The private long-form planning call failed; section drafts will continue from the original brief."
        )
        plan, plan_truncated = "", False
    plan = str(plan or "").strip()[:MAX_PLAN_CHARS]
    if not plan:
        record_speech_writer_warning(
            "The long-form speech plan was empty; section drafts were generated from the original brief."
        )
        plan = "No planning notes were returned. Follow the original brief as one continuous story and finish it."
    elif plan_truncated:
        record_speech_writer_warning(
            "The long-form speech plan reached its token limit; available planning notes were retained."
        )

    sections: list[str] = []
    empty_sections: list[int] = []
    truncated_sections: list[int] = []
    under_length_sections: list[tuple[int, int]] = []
    stopped_early = False

    for index in range(section_count):
        if calls >= MAX_GENERATION_CALLS:
            record_speech_writer_warning(
                f"Long-form speech writing stopped at section {index + 1} because the {MAX_GENERATION_CALLS}-call limit was reached."
            )
            break

        beat_start = index * section_words + 1
        beat_end = min(target_words, (index + 1) * section_words)
        previous_context = "\n".join(sections)[-MAX_PRIOR_TEXT_CHARS:]
        section_prompt = (
            "Write the next portion of the complete spoken script from this brief. Output only speech in the "
            "model's required native TTS format. Do not include explanations, markdown, production notes, "
            "chapter names, part headings, or labels such as 'Section 1'. Never print or speak these instructions. "
            "Preserve the cast, names, language, speaker count, and exact speaker/tag conventions required by the "
            "model guide. Keep the same narrative voice and emotional style. Do not recap or repeat earlier lines. "
            f"This portion should contain about {section_words} spoken words (overall script words {beat_start}–{beat_end}). "
            "Develop its assigned beat fully rather than summarizing it. The final portion must resolve the main story "
            "and feel complete; do not end on a cliffhanger.\n\n"
            f"Original brief:\n{prompt}\n\n"
            f"Private continuity and section plan:\n{plan}\n\n"
            f"Current section: {index + 1} of {section_count}.\n"
            f"Assigned section beat: section {index + 1}, approximately words {beat_start}–{beat_end}.\n"
        )
        if previous_context:
            section_prompt += (
                "Recent script context for continuity only (do not repeat it):\n"
                f"{previous_context}\n"
            )
        section_prompt += "\nNow write only this section of the spoken script."

        try:
            output, was_truncated = call(
                section_prompt, section_system,
                min(MAX_SECTION_TOKENS, max(768, int(section_words * 1.75 + 192))),
            )
        except InterruptedError:
            raise
        except Exception:
            stopped_early = True
            record_speech_writer_warning(
                f"Long-form speech writing stopped at section {index + 1} after a model call failed; "
                "the available script draft was retained for review."
            )
            break
        section = _without_part_headings(output)
        if not section:
            empty_sections.append(index + 1)
            record_speech_writer_warning(
                f"Long-form speech section {index + 1} returned no spoken text; other generated sections were retained."
            )
            if was_truncated:
                truncated_sections.append(index + 1)
                record_speech_writer_warning(
                    f"Long-form speech section {index + 1} hit the token limit before returning spoken text; the section remains incomplete."
                )
            continue
        section_words_actual = _word_count(section)
        if section_words_actual < int(section_words * 0.65):
            under_length_sections.append((index + 1, section_words_actual))

        if was_truncated:
            continuation_prompt = (
                "Continue the same spoken script from the exact point where the excerpt stops. Use the same cast, "
                "story state, language, writing style, and model-native TTS speaker/tag format. Do not repeat any "
                "words from the excerpt and do not add a heading. Continue this section's assigned beat. "
                + ("Finish the complete story with a clear ending." if index == section_count - 1 else
                   "Reach a natural handoff into the next planned section.")
                + f"\n\nOriginal brief:\n{prompt}\n\nPlan:\n{plan}\n\n"
                + f"Section excerpt to continue:\n{section[-MAX_PRIOR_TEXT_CHARS:]}"
            )
            try:
                continuation, continuation_truncated = call(
                    continuation_prompt, section_system, MAX_SECTION_TOKENS
                )
            except InterruptedError:
                raise
            except Exception:
                continuation = ""
                continuation_truncated = False
                record_speech_writer_warning(
                    f"Long-form speech section {index + 1} continuation failed; its available excerpt was retained for review."
                )
            continuation = _without_part_headings(continuation)
            continuation = _remove_repeated_prefix(section, continuation)
            if continuation:
                section = (section.rstrip() + "\n" + continuation.lstrip()).strip()
            else:
                record_speech_writer_warning(
                    f"Long-form speech section {index + 1} reached its token limit and its continuation made no progress; the available draft was retained."
                )
                truncated_sections.append(index + 1)
            if continuation_truncated:
                truncated_sections.append(index + 1)
                record_speech_writer_warning(
                    f"Long-form speech section {index + 1} also reached the token limit during continuation; review the assembled ending and handoff."
                )

        sections.append(section)

    assembled = "\n".join(section for section in sections if section).strip()
    actual_words = _word_count(assembled)
    if empty_sections:
        record_speech_writer_warning(
            "The assembled long-form speech draft has missing section(s) "
            + ", ".join(map(str, empty_sections))
            + "; review it before synthesis."
        )
    for section_number, section_words_actual in under_length_sections:
        record_speech_writer_warning(
            f"Long-form speech section {section_number} contains approximately {section_words_actual} words against a target of "
            f"{section_words}; the section draft was retained for review."
        )
    if actual_words < int(target_words * 0.75):
        record_speech_writer_warning(
            f"The long-form speech draft contains approximately {actual_words} words against a target of "
            f"{target_words}; the requested spoken runtime is approximate and the draft may be incomplete."
        )
    if stopped_early or section_count in truncated_sections:
        record_speech_writer_warning(
            "The final speech draft may be incomplete in native TTS syntax or end before its planned resolution because a model call failed or the last section was token-limited; review its speaker tags and ending."
        )

    if not assembled:
        raise RuntimeError(
            "Speech writing returned no spoken script text. Review the enhancement warning and retry; "
            "any generated planning or raw drafts remain available in the enhancement diagnostics."
        )

    print(
        f"[SpeechWriter] Long-form draft: {calls} model call(s), {actual_words}/{target_words} words, "
        f"{section_count} planned section(s)"
    )
    return assembled
