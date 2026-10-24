# Adapted from fishaudio/fish-speech, revision 1ff3c3ed2f80d3106a83909dd25b83129c3d816f.
# Copyright Fish Audio contributors. See LICENSE_FISH_AUDIO for the upstream license.
import re
import unicodedata
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from .config import FishOmniConfig
from .duration import LONGFORM_CHUNK_MAX_UNITS

DRAMA3_TTS_INSTRUCTION = "convert the provided text to speech"
DRAMA3_REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
DRAMA3_MESSAGE_END = "<|im_end|>\n"
DRAMA3_GENERATION_START = "<|im_start|>assistant\n<|voice|>"
# Only the prompt builder writes these; other special tokens in caller text are content.
DRAMA3_STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
# The tokenizer has one token per speaker, <|speaker:0|> to <|speaker:15|>.
MAX_SPEAKER_ID = 15
# The Unicode White_Space property; str.strip() without arguments also strips
# U+001C..U+001F, which would change the empty check and the speaker-tag rule.
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)
DRAMA3_MAX_PROMPT_ROWS = 8192
class Drama3PromptTooLongError(ValueError):
    """The prompt has more than DRAMA3_MAX_PROMPT_ROWS rows."""


def validate_drama3_text(text: str, location: str) -> None:
    """Non-empty, no structural tokens, and only canonical speaker tags 0..15."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise ValueError(f"{location}: text must not be empty")
    for token in DRAMA3_STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(
                f"{location}: {token} is written by the prompt builder"
            )
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG_PATTERN.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER_ID:
            raise ValueError(
                f"{location}: speaker tags must be <|speaker:0|> to "
                f"<|speaker:{MAX_SPEAKER_ID}|>; got {text!r}"
            )


# WanGP's dialogue convention, shared with its other speech models: "Speaker N:" (1..16) at the start of a line.
SPEAKER_LINE_PATTERN = re.compile(r"^([ \t]*)Speaker[ \t]*(\d+)[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)


# Drama 3's paired tags apply to the words between them; an unclosed one spoils the speech that follows.
PAIRED_TAGS = ("whisper", "stress", "slow")


def check_paired_tags(text: str, location: str) -> None:
    """Only Drama 3's paired tags may use angle brackets (besides <|...|> tokens), each one closed."""
    for match in re.finditer(r"</?([A-Za-z][^<>|]*)>?", text):  # tag-like: "<" before a letter (not "x < y" or "3<4")
        if match.group(1) not in PAIRED_TAGS or not match.group(0).endswith(">"):
            raise ValueError(f"{location}: {match.group(0)!r} is not a Drama 3 tag; tags are <whisper>, <stress> and <slow> around words, other directions go in square brackets such as [smiling]")
    for tag in PAIRED_TAGS:
        if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
            raise ValueError(f"{location}: close each <{tag}> with </{tag}> after the words it applies to")


def speaker_lines_to_tags(text: str) -> str:
    """'Speaker N:' line labels -> Drama 3's <|speaker:N-1|>; native tags are kept."""
    def tag(match):
        number = int(match.group(2))
        if not 1 <= number <= MAX_SPEAKER_ID + 1:
            raise ValueError(f"speaker labels go from Speaker 1: to Speaker {MAX_SPEAKER_ID + 1}:; got Speaker {number}:")
        return f"{match.group(1)}<|speaker:{number - 1}|> "
    return SPEAKER_LINE_PATTERN.sub(tag, text)


def dialogue_parts(text: str, lines_per_part: int) -> list[str]:
    """The script in parts of lines_per_part whole lines (one part when 0 or shorter). A line without a leading tag continues
    the speaker before it, so a part starting there gets that speaker's tag."""
    lines = [line for line in text.splitlines() if line.strip(UNICODE_WHITE_SPACE)]
    if lines_per_part <= 0 or len(lines) <= lines_per_part:
        return [text]
    parts, speaker = [], "<|speaker:0|>"
    for start in range(0, len(lines), lines_per_part):
        group = lines[start:start + lines_per_part]
        if SPEAKER_TAG_PATTERN.match(group[0].lstrip(UNICODE_WHITE_SPACE)) is None:
            group[0] = f"{speaker} {group[0].lstrip(UNICODE_WHITE_SPACE)}"
        for line in group:
            for match in SPEAKER_TAG_PATTERN.finditer(line):
                speaker = match.group(0)
        parts.append("\n".join(group))
    return parts


_SQUARE_CUE = re.compile(r"\[[^\]]*\]", re.DOTALL)
_DRAMA3_TAG_TOKEN = re.compile(r"<\|speaker:\d+\|>|</?(?:whisper|stress|slow)>", re.IGNORECASE)
_PAIRED_TAG_TOKEN = re.compile(r"<(\/)?(whisper|stress|slow)>", re.IGNORECASE)
_SENTENCE_ENDINGS = ".!?…。！？؟"
_SENTENCE_CLOSERS = "'\"’”»)]}」』"


@dataclass(frozen=True)
class Drama3TextChunk:
    """One lossless source span plus conditioning carried across a cut.

    ``source_start`` and ``source_end`` are half-open offsets into the exact
    submitted script. ``prompt_text`` may repeat active speaker/style controls
    as conditioning, while ``source_text`` is never duplicated or rewritten.
    """

    index: int
    source_start: int
    source_end: int
    source_text: str
    prompt_text: str
    speech_units: int


def _speech_unit_prefix(text: str) -> list[int]:
    ignored = bytearray(len(text))
    for pattern in (SPEAKER_LINE_PATTERN, SPEAKER_TAG_PATTERN, _SQUARE_CUE, _DRAMA3_TAG_TOKEN):
        for match in pattern.finditer(text):
            ignored[match.start():match.end()] = b"\x01" * (match.end() - match.start())
    prefix = [0]
    for index, character in enumerate(text):
        counted = (
            ignored[index] == 0
            and not character.isspace()
            and not unicodedata.category(character).startswith("C")
        )
        prefix.append(prefix[-1] + int(counted))
    return prefix


def _safe_unicode_boundary(text: str, position: int) -> bool:
    if position <= 0 or position >= len(text):
        return True
    right, left = text[position], text[position - 1]
    if unicodedata.category(right).startswith("M") or left == "\u200d" or right == "\u200d":
        return False
    regional = lambda char: 0x1F1E6 <= ord(char) <= 0x1F1FF
    if regional(left) and regional(right):
        count = 0
        cursor = position - 1
        while cursor >= 0 and regional(text[cursor]):
            count += 1
            cursor -= 1
        return count % 2 == 0
    return True


def _chunk_boundaries(text: str) -> tuple[list[int], list[int], list[int]]:
    """Return sentence/line, whitespace, and grapheme-safe split offsets."""
    atoms = []
    for pattern in (SPEAKER_LINE_PATTERN, _SQUARE_CUE, _DRAMA3_TAG_TOKEN):
        atoms.extend(pattern.finditer(text))
    atoms.sort(key=lambda match: (match.start(), -match.end()))
    atom_at = {match.start(): match.end() for match in atoms}
    natural, whitespace, fallback = set(), set(), set()
    index = 0
    while index < len(text):
        atom_end = atom_at.get(index)
        if atom_end is not None:
            index = atom_end
            fallback.add(index)
            continue

        character = text[index]
        if character.isspace():
            end = index + 1
            while end < len(text) and text[end].isspace() and end not in atom_at:
                end += 1
            whitespace.add(end)
            fallback.add(end)
            if "\n" in text[index:end] or "\r" in text[index:end]:
                natural.add(end)
            index = end
            continue

        end = index + 1
        if character in _SENTENCE_ENDINGS:
            while end < len(text) and text[end] in _SENTENCE_ENDINGS + _SENTENCE_CLOSERS:
                end += 1
            if end == len(text) or text[end].isspace():
                while end < len(text) and text[end].isspace() and end not in atom_at:
                    end += 1
                natural.add(end)
        if _safe_unicode_boundary(text, end):
            fallback.add(end)
        index += 1
    return sorted(natural), sorted(whitespace), sorted(fallback)


def _line_group_spans(text: str, lines_per_part: int) -> list[tuple[int, int]]:
    if lines_per_part <= 0:
        return [(0, len(text))] if text else []
    spans = []
    group_start = offset = line_count = 0
    for line in text.splitlines(keepends=True):
        offset += len(line)
        if line.strip(UNICODE_WHITE_SPACE):
            line_count += 1
            if line_count >= lines_per_part:
                spans.append((group_start, offset))
                group_start, line_count = offset, 0
    if group_start < len(text):
        spans.append((group_start, len(text)))
    return spans or ([(0, len(text))] if text else [])


def _speaker_before(text: str, offset: int) -> int:
    events = []
    for match in SPEAKER_LINE_PATTERN.finditer(text, 0, offset):
        if match.end() <= offset:
            number = int(match.group(2))
            if 1 <= number <= MAX_SPEAKER_ID + 1:
                events.append((match.start(), number - 1))
    for match in SPEAKER_TAG_PATTERN.finditer(text, 0, offset):
        if match.end() <= offset:
            events.append((match.start(), int(match.group(1))))
    return max(events, default=(-1, 0))[1]


def _opening_cues_before_line_offset(text: str, offset: int) -> list[str]:
    line_start = text.rfind("\n", 0, offset) + 1
    prefix = text[line_start:offset]
    match = SPEAKER_LINE_PATTERN.match(prefix)
    cursor = match.end() if match else 0
    if match is None:
        native_speaker = SPEAKER_TAG_PATTERN.match(prefix.lstrip(UNICODE_WHITE_SPACE))
        if native_speaker:
            leading = len(prefix) - len(prefix.lstrip(UNICODE_WHITE_SPACE))
            cursor = leading + native_speaker.end()
    cues = []
    while cursor < len(prefix) and prefix[cursor].isspace():
        cursor += 1
    while cursor < len(prefix) and prefix[cursor] == "[":
        close = prefix.find("]", cursor + 1)
        if close < 0:
            break
        cues.append(prefix[cursor:close + 1])
        cursor = close + 1
        while cursor < len(prefix) and prefix[cursor].isspace():
            cursor += 1
    return cues


def _active_paired_tags(text: str, offset: int) -> list[str]:
    active = []
    for match in _PAIRED_TAG_TOKEN.finditer(text, 0, offset):
        tag = match.group(2).lower()
        if match.group(1):
            for index in range(len(active) - 1, -1, -1):
                if active[index] == tag:
                    del active[index]
                    break
        else:
            active.append(tag)
    return active


def _split_span(
    start: int,
    end: int,
    units: list[int],
    boundaries: tuple[list[int], list[int], list[int]],
    max_units: int,
) -> list[tuple[int, int]]:
    natural, whitespace, fallback = boundaries
    spans = []
    cursor = start
    while units[end] - units[cursor] > max_units:
        limit = units[cursor] + max_units

        def latest(candidates: list[int], cursor: int, limit: int) -> int | None:
            result = None
            for boundary in candidates[bisect_right(candidates, cursor):]:
                if boundary >= end or units[boundary] > limit:
                    break
                if units[boundary] > units[cursor]:
                    result = boundary
            return result

        split = latest(natural, cursor, limit) or latest(whitespace, cursor, limit) or latest(fallback, cursor, limit)
        if split is None:
            raise ValueError("Fish long-form chunker could not find a safe text boundary")
        spans.append((cursor, split))
        cursor = split
    if cursor < end:
        spans.append((cursor, end))
    return spans


def drama3_text_chunks(
    text: str,
    lines_per_part: int = 0,
    max_speech_units: int = LONGFORM_CHUNK_MAX_UNITS,
) -> list[Drama3TextChunk]:
    """Split a script into bounded, lossless chunks with speaker/style continuity.

    Newlines and sentence endings are preferred; oversized sentences fall back
    to whitespace, then Unicode-safe codepoint boundaries for scripts without
    spaces. Explicit line cuts form an additional upper bound. A continuation
    chunk carries active speaker, line-opening cue, and paired-tag state only
    as prompt metadata; concatenating its source spans reproduces ``text``.
    """
    if max_speech_units <= 0:
        raise ValueError("max_speech_units must be positive")
    if not text:
        return []
    unit_prefix = _speech_unit_prefix(text)
    boundaries = _chunk_boundaries(text)
    spans = []
    for start, end in _line_group_spans(text, int(lines_per_part)):
        spans.extend(_split_span(start, end, unit_prefix, boundaries, max_speech_units))

    # An explicit line cut can leave trailing blank lines as a whitespace-only
    # span. Fold those into a neighboring source span instead of asking the
    # model to synthesize an empty part; cue-only spans remain valid parts.
    coalesced = []
    leading_whitespace_start = None
    for start, end in spans:
        if not text[start:end].strip(UNICODE_WHITE_SPACE):
            if coalesced:
                coalesced[-1] = (coalesced[-1][0], end)
            elif leading_whitespace_start is None:
                leading_whitespace_start = start
            continue
        if not coalesced and leading_whitespace_start is not None:
            start = leading_whitespace_start
        coalesced.append((start, end))
    if coalesced:
        spans = coalesced

    chunks = []
    for index, (start, end) in enumerate(spans, 1):
        source = text[start:end]
        leading = source.lstrip(UNICODE_WHITE_SPACE)
        has_speaker = bool(
            SPEAKER_TAG_PATTERN.match(leading)
            or SPEAKER_LINE_PATTERN.match(leading)
        )
        prefix = ""
        if not has_speaker:
            # The first unlabelled turn defaults to Speaker 1; later chunks
            # carry the most recent explicit speaker across the cut.
            prefix += f"<|speaker:{_speaker_before(text, start)}|> "
        if start > 0:
            prefix += " ".join(_opening_cues_before_line_offset(text, start))
            if prefix and not prefix.endswith(" "):
                prefix += " "
        active_at_start = _active_paired_tags(text, start)
        active_at_end = _active_paired_tags(text, end)
        prefix += "".join(f"<{tag}>" for tag in active_at_start)
        suffix = "".join(f"</{tag}>" for tag in reversed(active_at_end))
        prompt_text = speaker_lines_to_tags(prefix + source + suffix)
        chunks.append(
            Drama3TextChunk(
                index=index,
                source_start=start,
                source_end=end,
                source_text=source,
                prompt_text=prompt_text,
                speech_units=unit_prefix[end] - unit_prefix[start],
            )
        )
    return chunks


def reference_transcripts(text: str, count: int) -> list[str]:
    """One transcript per voice reference ('' when not given): 'Speaker N:' sections, else one line per reference; with a
    single reference, the whole text."""
    matches = list(SPEAKER_LINE_PATTERN.finditer(text))
    if matches:
        values = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            values[int(match.group(2))] = text[match.end():end].strip(UNICODE_WHITE_SPACE)
        return [values.get(number, "") for number in range(1, count + 1)]
    if count == 1:
        return [text.strip(UNICODE_WHITE_SPACE)]
    lines = [line.strip(UNICODE_WHITE_SPACE) for line in text.splitlines() if line.strip(UNICODE_WHITE_SPACE)]
    return (lines + [""] * count)[:count]


def with_speaker_tag(text: str) -> str:
    """Prefix <|speaker:0|> unless a tag already leads the text after white space."""
    if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:0|> {text}"
    return text


def build_drama3_prompt(
    text: str,
    tokenizer: Tokenizer,
    config: FishOmniConfig,
    prompt_texts: Sequence[str] = (),
    prompt_tokens: Sequence[torch.Tensor] = (),
    history: Sequence[tuple[str, torch.Tensor]] = (),
) -> torch.Tensor:
    """CPU int64 rows [rows, 1 + num_codebooks] ending in the open assistant voice turn.

    Text rows are [token, 0, ..., 0]; audio rows are [c0 + semantic_start, c0..cN-1].
    Every literal is tokenized on its own: tokenizing the joined string changes ids
    at the seams. history holds earlier (text, codes) chunks, written as completed
    user and assistant voice turns before the current one.
    """
    num_codebooks = config.audio_decoder.num_codebooks
    codebook_size = config.audio_decoder.vocab_size
    for token, expected in (
        ("<|semantic:0|>", config.semantic_start_token_id),
        (f"<|semantic:{codebook_size - 1}|>", config.semantic_end_token_id),
        ("<|im_end|>", config.eos_token_id),
    ):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"tokenizer/config mismatch for {token}")
    if len(prompt_texts) != len(prompt_tokens):
        raise ValueError("every reference needs both a transcript and codes")
    validate_drama3_text(text, "text")

    def check_codes(codes: torch.Tensor, location: str) -> None:
        if (
            codes.dtype.is_floating_point
            or codes.dtype == torch.bool
            or codes.ndim != 2
            or codes.shape[0] != num_codebooks
            or codes.shape[1] == 0
        ):
            raise ValueError(
                f"{location}: codes must be integer [{num_codebooks}, frames >= 1]; "
                f"got {codes.dtype} {tuple(codes.shape)}"
            )
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise ValueError(
                f"{location}: codes must be in [0, {codebook_size})"
            )

    segments: list[str | torch.Tensor] = [
        "<|im_start|>system\n",
        DRAMA3_REFERENCE_INSTRUCTION if prompt_tokens else DRAMA3_TTS_INSTRUCTION,
    ]

    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_tokens), 1):
        location = f"reference {index}"
        validate_drama3_text(transcript, location)
        check_codes(codes, location)
        # The tag must follow "Text: " directly; leading white space moves behind it.
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(DRAMA3_MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        location = f"history {index}"
        validate_drama3_text(turn_text, location)
        check_codes(codes, location)
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(turn_text),
            DRAMA3_MESSAGE_END,
            DRAMA3_GENERATION_START,
            codes,
            DRAMA3_MESSAGE_END,
        ]
    segments += [
        "<|im_start|>user\n",
        with_speaker_tag(text),
        DRAMA3_MESSAGE_END,
        DRAMA3_GENERATION_START,
    ]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = torch.tensor(tokenizer.encode(segment).ids, dtype=torch.long, device="cpu")
            text_rows = torch.zeros((len(ids), 1 + num_codebooks), dtype=torch.long, device="cpu")
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            main = codes[:1] + config.semantic_start_token_id
            rows.append(torch.cat((main, codes)).T)
    rows = torch.cat(rows)
    if len(rows) > DRAMA3_MAX_PROMPT_ROWS:
        raise Drama3PromptTooLongError(
            f"prompt has {len(rows)} rows; at most {DRAMA3_MAX_PROMPT_ROWS}"
        )
    return rows


def fit_history_to_prompt_rows(
    history: Sequence[tuple[str, torch.Tensor]],
    tokenizer: Tokenizer,
    available_rows: int,
) -> list[tuple[str, torch.Tensor]]:
    """Keep the newest bounded history that fits after references and script.

    Reference codes and current script rows are never dropped. Keep history
    turns whole so their transcripts remain aligned with their generated audio.
    """
    remaining = max(0, int(available_rows))
    kept = []

    def encoded_rows(value: str) -> int:
        return len(tokenizer.encode(value).ids)

    for text, codes in reversed(history):
        fixed_rows = sum(
            encoded_rows(segment)
            for segment in (
                "<|im_start|>user\n",
                with_speaker_tag(text),
                DRAMA3_MESSAGE_END,
                DRAMA3_GENERATION_START,
                DRAMA3_MESSAGE_END,
            )
        )
        turn_rows = fixed_rows + int(codes.shape[-1])
        if turn_rows > remaining:
            break
        kept.append((text, codes))
        remaining -= turn_rows
    kept.reverse()
    return kept
