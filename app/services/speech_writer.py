"""Bounded long-form speech writing for native TTS prompt enhancement.

Runtime is estimated from script words at 130 words per minute. This is a
planning estimate only; generated speech duration depends on the voice and
delivery. Qualitative requests for a long story/script default to about ten
minutes unless the brief or selected clip supplies a target.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable

from services.enhancement_diagnostics import record_advisory
from services.fish_script_format import FishScriptFormat


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
MIN_SECTION_COVERAGE = 0.85
MIN_OVERALL_COVERAGE = 0.95
DUPLICATE_MIN_COVERAGE = 0.80
DUPLICATE_MIN_WORDS = 80
MIN_DUPLICATE_TEXT_VARIETY = 0.08
MAX_DUPLICATE_CONTEXT_WORDS = 1_200


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
    r"\b(?:long(?:[ -]form)?|extended|full[ -]length)"
    r"(?P<modifiers>(?:(?:[ \t]+|[ \t]*[,;:()—–][ \t]*)"
    r"(?:(?:and|or|but|yet|&)[ \t]+)?[\w][\w'’\-]*){0,6}?)"
    r"(?:[ \t]+|[ \t]*[,;:()—–][ \t]*)"
    r"(?:form[ \t]+)?(?:story|dialogue|conversation|monologue|script|speech)\b",
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
_SPEECH_CUE_RE = re.compile(r"\[[^\]\r\n]*\]")
_TTS_TAG_RE = re.compile(r"<[^>\r\n]*>")
_TTS_TAG_TOKEN_RE = re.compile(r"<\s*(/?)\s*([A-Za-z][\w:.-]*)\b([^<>]*?)>")
_SPEAKER_LABEL_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:speaker|voice|character)[ _-]*\d+(?:[ \t]*\{[^}\r\n]*\})?"
    r"|[A-Z][\w'’\-]{0,39}(?:[ \t]+[A-Z][\w'’\-]{0,39})?)[ \t]*:[ \t]*"
)
_QUALITATIVE_MODIFIER_LINK_WORDS = frozenset({"and", "or", "but", "yet"})
_QUALITATIVE_MODIFIER_WORDS = frozenset({
    "bad", "big", "bold", "bright", "cool", "dark", "deep", "dry", "fair",
    "fun", "good", "great", "grim", "hard", "hot", "large", "light", "loud",
    "mild", "new", "odd", "old", "plain", "real", "red", "rich", "safe",
    "sharp", "small", "soft", "tall", "true", "warm", "weak", "weird", "wide",
    "wild", "wise", "young", "eerie", "spooky", "full", "complete", "fresh",
    "ghost", "halloween", "science", "fiction", "horror", "fantasy", "mystery",
})
_QUALITATIVE_NONLENGTH_NOUNS = frozenset({"hair", "hall", "hallway", "life", "lifetime"})
_QUALITATIVE_ADJECTIVE_SUFFIXES = (
    "able", "ible", "al", "ary", "ed", "ent", "ful", "ic", "ing", "ish",
    "ive", "less", "like", "ly", "ory", "ous", "some", "y",
)


def _is_qualitative_modifier(word: str) -> bool:
    """Recognize common adjectives and genres while rejecting incidental nouns."""
    parts = word.casefold().strip("-'’").split("-")
    if any(part in _QUALITATIVE_NONLENGTH_NOUNS for part in parts):
        return False
    return any(
        part in _QUALITATIVE_MODIFIER_WORDS
        or part.endswith(_QUALITATIVE_ADJECTIVE_SUFFIXES)
        for part in parts
        if part
    )


def _has_qualitative_long_request(prompt: str) -> bool:
    """Match a bounded adjective phrase between a length cue and speech type."""
    for match in _QUALITATIVE_LONG_RE.finditer(prompt):
        modifiers = _WORD_RE.findall(match.group("modifiers"))
        if all(
            word.casefold() in _QUALITATIVE_MODIFIER_LINK_WORDS
            or _is_qualitative_modifier(word)
            for word in modifiers
        ):
            return True
    return False


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


def resolve_speech_target(
    prompt: str,
    requested_minutes: object = 0,
    *,
    clip_duration_seconds: object = None,
) -> SpeechTarget:
    """Resolve explicit targets and qualitative long-form requests.

    A positive setting wins, followed by numeric size in the brief. Clear
    short/brief requests keep the ordinary one-call behavior. Otherwise a
    selected clip duration sets the target before qualitative long-form
    requests fall back to ten minutes. Every inferred target is capped at
    sixty minutes.
    """
    try:
        configured = float(requested_minutes or 0)
    except (TypeError, ValueError, OverflowError):
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

    try:
        clip_seconds = (
            0.0
            if isinstance(clip_duration_seconds, bool)
            else float(clip_duration_seconds or 0)
        )
    except (TypeError, ValueError, OverflowError):
        clip_seconds = 0.0
    if math.isfinite(clip_seconds) and clip_seconds > 0:
        return _target_from_minutes(clip_seconds / 60, "clip_duration")

    if _has_qualitative_long_request(source_text):
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


def _spoken_tokens(text: str) -> list[str]:
    """Return words that will be spoken, excluding TTS directions and labels."""
    value = _without_part_headings(text)
    value = _SPEECH_CUE_RE.sub(" ", value)
    value = _TTS_TAG_RE.sub(" ", value)
    value = _SPEAKER_LABEL_RE.sub("", value)
    return [token.casefold() for token in _WORD_RE.findall(value)]


def _spoken_word_count(text: str) -> int:
    return len(_spoken_tokens(text))


def _has_substantial_duplicate(previous: str, current: str) -> bool:
    """Detect a section that substantially repeats earlier spoken wording.

    A long exact run of normalized spoken words is resilient to cue, tag,
    punctuation, and case changes while ignoring short repeated refrains. The
    comparison is bounded to recent context and deliberately conservative.
    """
    prior_words = _spoken_tokens(previous)[-MAX_DUPLICATE_CONTEXT_WORDS:]
    current_words = _spoken_tokens(current)
    if len(current_words) < DUPLICATE_MIN_WORDS:
        return False
    if len(set(current_words)) / len(current_words) < MIN_DUPLICATE_TEXT_VARIETY:
        return False

    longest = 0
    lengths = [0] * (len(current_words) + 1)
    for prior_word in prior_words:
        for index in range(len(current_words) - 1, -1, -1):
            if prior_word == current_words[index]:
                lengths[index + 1] = lengths[index] + 1
                longest = max(longest, lengths[index + 1])
            else:
                lengths[index + 1] = 0
    return longest >= max(
        DUPLICATE_MIN_WORDS,
        math.ceil(len(current_words) * DUPLICATE_MIN_COVERAGE),
    )


def _without_part_headings(text: str) -> str:
    lines = [
        line.rstrip()
        for line in str(text or "").splitlines()
        if not _PART_HEADING_RE.match(line.strip())
    ]
    return "\n".join(lines).strip()


def _without_part_headings_preserve_leading_space(text: str) -> tuple[str, str]:
    """Strip headings while retaining the continuation's original left seam."""
    raw = str(text or "")
    lines = raw.splitlines(keepends=True)
    body = "".join(
        line for line in lines
        if not _PART_HEADING_RE.match(line.strip())
    )
    leading = re.match(r"[ \t\r\n]*", body)
    leading_whitespace = leading.group(0) if leading else ""
    return body.strip(), leading_whitespace


def _bounded_script_context(text: str, limit: int = MAX_PRIOR_TEXT_CHARS) -> str:
    """Keep an exact recent excerpt without starting inside words or TTS cues.

    Only the private context is shortened. Prefer complete paragraphs, then
    lines, then a safe word boundary; retain the exact ending for continuation.
    An unsafe scope longer than the allowance is omitted rather than fabricated.
    """
    value = str(text or "")
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    floor = len(value) - limit
    seen: set[int] = set()
    for pattern in (r"\n[ \t]*\n", r"[\r\n]+", r"\s+"):
        inspected = 0
        for match in re.finditer(pattern, value[floor:]):
            cut = floor + match.end()
            if cut in seen:
                continue
            seen.add(cut)
            inspected += 1
            prefix = value[:cut]
            line_start = value.rfind("\n", 0, cut) + 1
            label = _SPEAKER_LABEL_RE.match(value, line_start)
            inside_labeled_turn = label is not None and label.start() < cut
            if (not inside_labeled_turn and not _has_unclosed_native_syntax(prefix)
                    and prefix.count("<|phoneme_start|>") == prefix.count("<|phoneme_end|>")):
                return value[cut:]
            if inspected >= 64:
                break
    return ""


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


def _has_open_native_token(text: str) -> bool:
    """Detect an unfinished bracket cue or angle-tag token at a text seam."""
    value = str(text or "")
    open_angles = 0
    open_brackets = 0
    for index, character in enumerate(value):
        if character == "<" and (
            index + 1 == len(value)
            or re.match(r"[A-Za-z/]", value[index + 1])
        ):
            open_angles += 1
        elif character == ">" and open_angles:
            open_angles -= 1
        elif character == "[":
            open_brackets += 1
        elif character == "]" and open_brackets:
            open_brackets -= 1
    return bool(open_angles or open_brackets)


def _has_open_angle_token(text: str) -> bool:
    """Detect an unfinished angle-tag token, whose name may span the seam."""
    value = str(text or "")
    open_angles = 0
    for index, character in enumerate(value):
        if character == "<" and (
            index + 1 == len(value)
            or re.match(r"[A-Za-z/]", value[index + 1])
        ):
            open_angles += 1
        elif character == ">" and open_angles:
            open_angles -= 1
    return bool(open_angles)


def _has_unclosed_native_syntax(text: str) -> bool:
    """Check for unfinished cue tokens and unbalanced paired TTS tags."""
    value = str(text or "")
    if _has_open_native_token(value):
        return True

    cue_depth = 0
    for character in value:
        if character == "[":
            cue_depth += 1
        elif character == "]":
            if not cue_depth:
                return True
            cue_depth -= 1

    stack: list[str] = []
    malformed = False
    for match in _TTS_TAG_TOKEN_RE.finditer(value):
        closing, raw_name, attributes = match.groups()
        name = raw_name.casefold()
        if closing:
            if stack and stack[-1] == name:
                stack.pop()
            else:
                malformed = True
        elif not attributes.rstrip().endswith("/"):
            stack.append(name)
    return bool(stack or malformed)


def _join_script_continuation(
    previous: str,
    addition: str,
    *,
    leading_whitespace: str = "",
) -> str:
    """Join native syntax fragments directly and keep prose on a word seam."""
    left = str(previous or "").rstrip()
    right = str(addition or "").lstrip()
    if not left:
        return right
    if leading_whitespace:
        separator = leading_whitespace
    elif _has_open_angle_token(left):
        separator = ""
    elif _has_open_native_token(left) and right.startswith(("]", ">")):
        separator = ""
    else:
        separator = "\n"
    return (left + separator + right).strip()


def _has_uncertain_word_boundary(
    previous: str,
    addition: str,
    *,
    leading_whitespace: str = "",
) -> bool:
    """Report a token-limit seam that may have split a word without whitespace."""
    left = str(previous or "").rstrip()
    right = str(addition or "").lstrip()
    return bool(
        left
        and right
        and not leading_whitespace
        and not _has_open_angle_token(left)
        and left[-1].isalnum()
        and right[0].isalnum()
    )


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
    finalize_section: Callable[..., str] | None = None,
    script_format: FishScriptFormat | None = None,
) -> str:
    """Generate one assembled TTS script through a bounded plan and sections.

    Planning, section drafts, truncation continuations, focused repairs, and
    optional section finalization share a hard thirty-call cap. Previous
    model-authored speech supplied as context is limited to a short rolling
    tail, while the complete source brief and bounded plan remain available to
    every section.
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

    def normalize_script_fragment(text: str, *, stage: str) -> str:
        if script_format is None:
            return str(text or "")
        normalized, changed_cues = script_format.normalize_fragment(str(text or ""))
        if changed_cues:
            record_advisory("fish_script_format_normalization", {
                "stage": stage,
                "converted_delivery_cues": len(changed_cues),
            })
        return normalized

    def recover_script_continuation(
        previous: str, addition: str, *, stage: str
    ) -> tuple[str, str]:
        if script_format is None:
            return previous, addition
        recovered_previous, recovered_addition, recovered = script_format.recover_continuation(
            previous, addition
        )
        if recovered:
            record_advisory("fish_script_format_seam_recovery", {
                "stage": stage,
                "boundary": "independent_native_directive",
            })
        return recovered_previous, recovered_addition

    def call(
        call_prompt: str,
        call_system: str,
        token_budget: int,
        *,
        call_temperature: float | None = None,
    ) -> tuple[str, bool]:
        nonlocal calls
        if calls >= MAX_GENERATION_CALLS:
            return "", False
        _check_cancelled()
        calls += 1
        output = str(generate(
            prompt=call_prompt,
            system_prompt=call_system,
            max_new_tokens=max(1, min(int(token_budget), per_call_token_limit)),
            temperature=temperature if call_temperature is None else call_temperature,
            image_paths=None,
            enable_thinking=False,
            thinking_budget=0,
        ) or "")
        return output, bool(is_truncated and is_truncated())

    def continue_partial_output(
        output: str,
        was_truncated: bool,
        *,
        section_index: int,
        output_kind: str,
    ) -> tuple[str, bool, str, bool]:
        """Finish a token-limited output while retaining its exact useful text."""
        current = str(output or "").strip()
        pending = bool(was_truncated)
        failure = ""
        uncertain_word_boundary = False
        while pending:
            if calls >= MAX_GENERATION_CALLS:
                failure = "call_limit"
                break
            final_section = section_index == section_count - 1
            continuation_prompt = (
                "Continue the same spoken script from the exact point where the preceding response stops. "
                "That response reached its output token limit. If it ended inside a sentence, speaker turn, "
                "native TTS tag, or performance cue, finish that syntax naturally before adding new material. "
                "Preserve the cast, story state, language, and native speaker/tag format. Do not repeat any "
                "words from the excerpt and do not add a heading. "
                + (
                    "Finish the complete story with a clear, coherent ending."
                    if final_section else
                    "Finish this section's assigned beat and reach a natural handoff into the next section."
                )
                + f"\n\nOriginal brief:\n{prompt}\n\nPlan:\n{plan}\n\n"
                + f"Partial {output_kind} response to continue:\n"
                + _bounded_script_context(current)
            )
            try:
                continuation_raw, pending = call(
                    continuation_prompt, section_system, MAX_SECTION_TOKENS
                )
            except InterruptedError:
                raise
            except Exception:
                failure = "call_failed"
                break

            continuation, leading_whitespace = _without_part_headings_preserve_leading_space(
                continuation_raw
            )
            continuation = normalize_script_fragment(
                continuation, stage=f"{output_kind} continuation"
            )
            addition = _remove_repeated_prefix(current, continuation)
            syntax_was_open = _has_unclosed_native_syntax(current)
            join_previous, addition = recover_script_continuation(
                current, addition, stage=f"{output_kind} continuation"
            )
            joined = _join_script_continuation(
                join_previous, addition, leading_whitespace=leading_whitespace
            )
            syntax_progress = (
                syntax_was_open and not _has_unclosed_native_syntax(joined)
            )
            if (
                not addition
                or (
                    not _spoken_word_count(addition)
                    and not syntax_progress
                )
                or (
                    _has_substantial_duplicate(current, addition)
                    and not syntax_progress
                )
            ):
                pending = True
                failure = "no_progress"
                break
            uncertain_word_boundary = uncertain_word_boundary or _has_uncertain_word_boundary(
                current, addition, leading_whitespace=leading_whitespace
            )
            current = joined

        if not pending and _has_unclosed_native_syntax(current):
            failure = failure or "native_syntax"
        return current, pending or _has_unclosed_native_syntax(current), failure, uncertain_word_boundary

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
    section_system = (
        system_prompt.replace(
            "Output only the complete requested spoken script.",
            "Output only the requested spoken-script portion.",
        )
        + "\n\nLONG-FORM CONTEXT RULE: Preserve exact user-authored wording when the user supplied "
        "words intended to be spoken, along with all explicit user-supplied performance cues and native TTS tags. "
        "The section prompt labels earlier model-authored speech as continuity context already written; it is not "
        "immutable user source wording. Continue after that text's events without copying, paraphrasing, or restarting "
        "them. Keep each speaker's identity, accent, and baseline register stable across sections. "
        "Let emotion, pace, emphasis, and intensity follow the meaning of the spoken text, using only the "
        "selected model's supported performance controls where useful. Those acting changes do not require "
        "inventing a new voice or an explicit user instruction for every phrase. Honor requests for plain, "
        "untagged, or deliberately uniform delivery."
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
    section_coverage_words: list[int] = []
    empty_sections: list[int] = []
    truncated_sections: list[int] = []
    duplicate_sections: list[int] = []
    uncertain_boundary_sections: list[int] = []
    stopped_early = False

    for index in range(section_count):
        if calls >= MAX_GENERATION_CALLS:
            record_speech_writer_warning(
                f"Long-form speech writing stopped at section {index + 1} because the {MAX_GENERATION_CALLS}-call limit was reached."
            )
            break

        beat_start = index * section_words + 1
        beat_end = min(target_words, (index + 1) * section_words)
        previous_context = _bounded_script_context("\n".join(sections))
        section_prompt = (
            "Write the next portion of the complete spoken script from this brief. Output only speech in the "
            "model's required native TTS format, including the speaker labels and performance controls "
            "supported by that guide. These native cues and word/phrase tags are script syntax, not production "
            "commentary; use them where the guide calls for expressive delivery. Do not include explanations, "
            "Markdown fences, external production commentary, chapter names, part headings, or labels such as "
            "'Section 1'. Never print or speak these instructions. "
            "Preserve the cast, names, language, speaker count, and exact speaker/tag conventions required by the "
            "model guide. Keep each speaker's identity, accent, and baseline register stable across sections. "
            "Direct meaningful emotional turns and important phrases using the model's native controls; "
            "vary the acting without changing the voice identity, and honor any plain-delivery request. "
            "Preserve exact user-supplied words "
            "and explicit TTS cues. Treat prior model-authored section text as continuity context already written: "
            "continue after its events without copying, paraphrasing, or restarting them. "
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
                "Recent model-authored script context for continuity only; it is already part of the script, "
                "not new user-supplied source wording. Continue after it without repeating or paraphrasing it:\n"
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
        section = _without_part_headings(
            normalize_script_fragment(output, stage=f"section {index + 1}")
        )
        if not section or not _spoken_word_count(section):
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

        section_incomplete = False
        section_boundary_uncertain = False
        completion_failure = ""
        if was_truncated:
            section, section_incomplete, completion_failure, section_boundary_uncertain = continue_partial_output(
                section,
                was_truncated,
                section_index=index,
                output_kind="section excerpt",
            )
            if section_incomplete:
                truncated_sections.append(index + 1)
                if completion_failure == "no_progress":
                    record_speech_writer_warning(
                        f"Long-form speech section {index + 1} reached its token limit and its continuation made no progress; the available draft was retained."
                    )
                elif completion_failure == "call_limit":
                    record_speech_writer_warning(
                        f"Long-form speech section {index + 1} continuation could not finish within the {MAX_GENERATION_CALLS}-call limit; its available draft was retained."
                    )
                elif completion_failure == "native_syntax":
                    record_speech_writer_warning(
                        f"Long-form speech section {index + 1} continuation left native TTS syntax open; the available draft was retained for review."
                    )
                else:
                    record_speech_writer_warning(
                        f"Long-form speech section {index + 1} continuation failed before the token-limited excerpt could be completed; its available draft was retained."
                    )
                if completion_failure not in {"no_progress", "native_syntax"}:
                    record_speech_writer_warning(
                        f"Long-form speech section {index + 1} also reached the token limit during continuation; review the assembled ending and handoff."
                    )
        if _has_unclosed_native_syntax(section):
            section_incomplete = True
            if index + 1 not in truncated_sections:
                truncated_sections.append(index + 1)
        if section_boundary_uncertain:
            uncertain_boundary_sections.append(index + 1)
            record_speech_writer_warning(
                f"Long-form speech section {index + 1} crossed a token-limit seam without whitespace between ordinary words; a safe line break was retained, but the intended word boundary is uncertain."
            )

        prior_text = "\n".join(sections)
        section_words_actual = _spoken_word_count(section)
        duplicate = _has_substantial_duplicate(prior_text, section)
        below_target = section_words_actual < math.ceil(section_words * MIN_SECTION_COVERAGE)
        preserve_finished_ending = (
            index == section_count - 1 and not section_incomplete and not duplicate
        )
        needs_completion_repair = section_incomplete and not duplicate
        repair_attempted = False
        if duplicate or needs_completion_repair or (below_target and not preserve_finished_ending):
            if calls >= MAX_GENERATION_CALLS:
                reason = "repeated earlier wording" if duplicate else "below its spoken-word target"
                record_speech_writer_warning(
                    f"Long-form speech section {index + 1} could not be repaired because the "
                    f"{MAX_GENERATION_CALLS}-call limit was reached; its available draft was retained despite {reason}."
                )
            else:
                if duplicate:
                    repair_prompt = (
                        "Focused section revision: this section substantially repeats earlier model-authored text. "
                        "Replace only the current section with a distinct continuation that starts after the earlier "
                        "events. Preserve the original brief's facts, names, user-supplied exact wording, explicit TTS "
                        "cues, speaker identities, accents, baseline registers, and native tag format. Keep any useful "
                        "new details from the current draft when they do not repeat earlier material. Aim for about "
                        f"{section_words} spoken words. Return only the replacement section.\n\n"
                        f"Original brief:\n{prompt}\n\nPlan:\n{plan}\n\n"
                        "Earlier model-authored script (continuity only; already included in the final script):\n"
                        f"{_bounded_script_context(prior_text)}\n\n"
                        "Current section draft to revise; retain its new details where possible:\n"
                        f"{_bounded_script_context(section)}"
                    )
                    repair_budget = min(
                        MAX_SECTION_TOKENS,
                        max(768, int(section_words * 1.75 + 192)),
                    )
                    repair_kind = "replacement"
                elif needs_completion_repair:
                    finish_instruction = (
                        "Complete the story with a coherent ending and resolve the main conflict."
                        if index == section_count - 1 else
                        "Complete this section's assigned beat and reach a natural handoff."
                    )
                    repair_prompt = (
                        "Focused section completion: the current spoken draft reached a token limit or could not be "
                        "continued. Preserve every existing line exactly as written and return only new speech to "
                        "append from its exact ending. If the last sentence, speaker turn, native TTS tag, or "
                        "performance cue is incomplete, finish that syntax naturally. Do not recap, repeat, or add "
                        "speech after the requested ending. Preserve the original brief's facts, names, user-supplied "
                        "exact wording, explicit TTS cues, speaker identities, accents, baseline registers, and native "
                        f"tag format. {finish_instruction}\n\n"
                        f"Original brief:\n{prompt}\n\nPlan:\n{plan}\n\n"
                        f"Current section to complete:\n{_bounded_script_context(section)}"
                    )
                    repair_budget = MAX_SECTION_TOKENS
                    repair_kind = "completion"
                else:
                    missing_words = max(1, section_words - section_words_actual)
                    finish_instruction = (
                        "Finish the complete story with a coherent, satisfying ending; resolve the main conflict and do not end on a cliffhanger."
                        if index == section_count - 1 else
                        "Reach a natural handoff into the next planned section."
                    )
                    repair_prompt = (
                        "Focused section expansion: preserve every existing line exactly as written and return only "
                        "new speech to append to the current section. Continue from its exact ending; do not recap or "
                        "repeat the current section or earlier sections. Preserve the original brief's facts, names, "
                        "user-supplied exact wording, explicit TTS cues, speaker identities, accents, baseline registers, "
                        "and native tag format. Add about "
                        f"{missing_words} spoken words, developing the assigned beat instead of summarizing it. "
                        f"{finish_instruction}\n\n"
                        f"Original brief:\n{prompt}\n\nPlan:\n{plan}\n\n"
                        "Earlier model-authored script (continuity only; already included in the final script):\n"
                        f"{_bounded_script_context(prior_text)}\n\n"
                        "Current section to continue from:\n"
                        f"{_bounded_script_context(section)}"
                    )
                    repair_budget = min(
                        MAX_SECTION_TOKENS,
                        max(384, int(missing_words * 1.75 + 128)),
                    )
                    repair_kind = "expansion"

                try:
                    repair_output_raw, repair_truncated = call(
                        repair_prompt, section_system, repair_budget
                    )
                    repair_output, repair_leading_whitespace = _without_part_headings_preserve_leading_space(
                        repair_output_raw
                    )
                    repair_output = normalize_script_fragment(
                        repair_output, stage=f"focused {repair_kind} section {index + 1}"
                    )
                    repair_attempted = True
                except InterruptedError:
                    raise
                except Exception:
                    repair_output = ""
                    repair_leading_whitespace = ""
                    repair_truncated = False
                    record_speech_writer_warning(
                        f"Long-form speech section {index + 1} focused {repair_kind} failed; its original draft was retained."
                    )

                if repair_truncated:
                    repair_output, repair_incomplete, _repair_completion_failure, repair_boundary_uncertain = continue_partial_output(
                        repair_output,
                        repair_truncated,
                        section_index=index,
                        output_kind=f"focused {repair_kind}",
                    )
                    section_boundary_uncertain = section_boundary_uncertain or repair_boundary_uncertain
                else:
                    repair_incomplete = False
                if repair_kind == "replacement" and _has_unclosed_native_syntax(repair_output):
                    repair_incomplete = True
                if repair_incomplete:
                    record_speech_writer_warning(
                        f"Long-form speech section {index + 1} focused {repair_kind} and its continuation could not finish; the original section was retained and the partial repair remains available in generation history."
                    )

                if repair_incomplete:
                    pass
                elif duplicate:
                    if (
                        _spoken_word_count(repair_output) > 0
                        and not _has_substantial_duplicate(prior_text, repair_output)
                        and not _has_unclosed_native_syntax(repair_output)
                    ):
                        section = repair_output
                        section_incomplete = False
                        if index + 1 in truncated_sections:
                            truncated_sections.remove(index + 1)
                        duplicate = False
                    else:
                        record_speech_writer_warning(
                            f"Long-form speech section {index + 1} still substantially repeats earlier text after one focused revision; its original draft was retained for review."
                        )
                elif needs_completion_repair:
                    addition = _remove_repeated_prefix(section, repair_output)
                    join_previous, addition = recover_script_continuation(
                        section, addition, stage=f"focused completion section {index + 1}"
                    )
                    candidate = _join_script_continuation(
                        join_previous, addition,
                        leading_whitespace=repair_leading_whitespace,
                    )
                    syntax_progress = (
                        _has_unclosed_native_syntax(section)
                        and not _has_unclosed_native_syntax(candidate)
                    )
                    section_boundary_uncertain = section_boundary_uncertain or _has_uncertain_word_boundary(
                        section, addition,
                        leading_whitespace=repair_leading_whitespace,
                    )
                    if (
                        (_spoken_word_count(addition) > 0 or syntax_progress)
                        and not _has_substantial_duplicate(f"{prior_text}\n{section}", addition)
                        and not _has_unclosed_native_syntax(candidate)
                    ):
                        section = candidate
                        section_incomplete = False
                        if index + 1 in truncated_sections:
                            truncated_sections.remove(index + 1)
                    else:
                        record_speech_writer_warning(
                            f"Long-form speech section {index + 1} completion made no distinct progress or left native TTS syntax open; its available draft was retained."
                        )
                else:
                    addition = _remove_repeated_prefix(section, repair_output)
                    join_previous, addition = recover_script_continuation(
                        section, addition, stage=f"focused expansion section {index + 1}"
                    )
                    candidate = _join_script_continuation(
                        join_previous, addition,
                        leading_whitespace=repair_leading_whitespace,
                    )
                    section_boundary_uncertain = section_boundary_uncertain or _has_uncertain_word_boundary(
                        section, addition,
                        leading_whitespace=repair_leading_whitespace,
                    )
                    if (
                        _spoken_word_count(addition) > 0
                        and not _has_substantial_duplicate(f"{prior_text}\n{section}", addition)
                        and not _has_unclosed_native_syntax(candidate)
                    ):
                        section = candidate
                    else:
                        record_speech_writer_warning(
                            f"Long-form speech section {index + 1} focused expansion made no distinct progress or left native TTS syntax open; its original draft was retained."
                        )

        if _has_unclosed_native_syntax(section):
            section_incomplete = True
            if index + 1 not in truncated_sections:
                truncated_sections.append(index + 1)
            record_speech_writer_warning(
                f"Long-form speech section {index + 1} retains an unclosed native TTS tag or cue; the available draft was kept for review."
            )

        if section_boundary_uncertain and index + 1 not in uncertain_boundary_sections:
            uncertain_boundary_sections.append(index + 1)
            record_speech_writer_warning(
                f"Long-form speech section {index + 1} crossed a token-limit seam without whitespace between ordinary words; a safe line break was retained, but the intended word boundary is uncertain."
            )

        if finalize_section is not None:
            remaining_sections = section_count - index - 1
            reserve_calls = 2 * remaining_sections + (1 if remaining_sections else 0)
            finalizer_limit = MAX_GENERATION_CALLS - reserve_calls

            def remaining_finalizer_calls() -> int:
                return max(0, finalizer_limit - calls)

            def bounded_finalizer_call(
                *,
                prompt: str,
                system_prompt: str,
                max_new_tokens: int,
                temperature: float,
                **_options: object,
            ) -> tuple[str, bool]:
                if remaining_finalizer_calls() <= 0:
                    return "", False
                return call(
                    prompt, system_prompt, max_new_tokens,
                    call_temperature=temperature,
                )

            try:
                finalized = finalize_section(
                    text=section,
                    generate=bounded_finalizer_call,
                    remaining_calls=remaining_finalizer_calls,
                    reserve_calls=reserve_calls,
                    max_new_tokens=min(1_024, per_call_token_limit),
                    temperature=temperature,
                    section_incomplete=section_incomplete,
                )
            except InterruptedError:
                raise
            except Exception:
                record_speech_writer_warning(
                    f"Long-form speech section {index + 1} finalization failed; its available draft was retained."
                )
            else:
                if isinstance(finalized, str) and finalized.strip():
                    candidate = normalize_script_fragment(
                        finalized.strip(), stage=f"finalized section {index + 1}"
                    )
                    if section_incomplete and candidate != section:
                        record_speech_writer_warning(
                            f"Long-form speech section {index + 1} finalization changed an incomplete draft; its prior text was retained."
                        )
                    elif _has_unclosed_native_syntax(candidate):
                        record_speech_writer_warning(
                            f"Long-form speech section {index + 1} finalization left native TTS syntax open; its prior draft was retained."
                        )
                    else:
                        section = candidate
                elif finalized is not None:
                    record_speech_writer_warning(
                        f"Long-form speech section {index + 1} finalization returned no usable text; its available draft was retained."
                    )

        section_words_actual = _spoken_word_count(section)
        duplicate = _has_substantial_duplicate(prior_text, section)
        if duplicate:
            duplicate_sections.append(index + 1)
            record_speech_writer_warning(
                f"Long-form speech section {index + 1} remains a substantial repetition of earlier text; its words are not counted twice toward target coverage."
            )
        if section_words_actual < math.ceil(section_words * MIN_SECTION_COVERAGE):
            if preserve_finished_ending and not repair_attempted:
                record_speech_writer_warning(
                    f"Long-form speech section {index + 1} contains approximately {section_words_actual} spoken words against a target of "
                    f"{section_words}; it ended without a token-limit signal, so the draft was retained without a length-only expansion."
                )
            elif repair_attempted:
                record_speech_writer_warning(
                    f"Long-form speech section {index + 1} contains approximately {section_words_actual} spoken words against a target of "
                    f"{section_words}; its available draft was retained after one bounded repair attempt."
                )
            else:
                record_speech_writer_warning(
                    f"Long-form speech section {index + 1} contains approximately {section_words_actual} spoken words against a target of "
                    f"{section_words}; its available draft was retained because the call limit or a model error prevented repair."
                )

        sections.append(section)
        section_coverage_words.append(0 if duplicate else section_words_actual)

    assembled = "\n".join(section for section in sections if section).strip()
    actual_words = _spoken_word_count(assembled)
    covered_words = sum(section_coverage_words)
    if empty_sections:
        record_speech_writer_warning(
            "The assembled long-form speech draft has missing section(s) "
            + ", ".join(map(str, empty_sections))
            + "; review it before synthesis."
        )
    if covered_words < int(target_words * MIN_OVERALL_COVERAGE):
        coverage_label = "distinct spoken words" if duplicate_sections else "spoken words"
        coverage_percent = int(round(covered_words / max(1, target_words) * 100))
        record_speech_writer_warning(
            f"The long-form speech draft contains approximately {covered_words} {coverage_label} against a target of "
            f"{target_words} ({coverage_percent}% coverage); the requested spoken runtime is approximate and the draft remains short."
        )
    if actual_words > math.ceil(target_words * 1.25):
        record_speech_writer_warning(
            f"The long-form speech draft contains approximately {actual_words} spoken words against a target of "
            f"{target_words}; it is substantially longer than requested. The complete text was retained. "
            "Review its length against the synthesis maximum, which could stop before the full script is spoken."
        )
    if stopped_early or truncated_sections:
        record_speech_writer_warning(
            "The final speech draft may be incomplete in native TTS syntax or end before its planned resolution because a model call failed, output reached a token limit, or native syntax remained open; review its speaker tags and ending."
        )

    if not assembled:
        raise RuntimeError(
            "Speech writing returned no spoken script text. Review the enhancement warning and retry; "
            "any generated planning or raw drafts remain available in the enhancement diagnostics."
        )

    print(
        f"[SpeechWriter] Long-form draft: {calls} model call(s), {covered_words}/{target_words} distinct spoken words "
        f"({actual_words} including repeated sections), "
        f"{section_count} planned section(s)"
    )
    return assembled
