"""Prompt and input contract for H3's hidden-video audio-only preset."""
from __future__ import annotations

import math
import random
import re
import unicodedata
from dataclasses import dataclass, replace

from services.dialogue_timing import DIALOGUE_DEFAULT_WORDS_PER_SECOND, DIALOGUE_MAX_WORDS_PER_SECOND


MIN_AUDIO_SECONDS = 5.0
MAX_SEGMENT_SECONDS = 45.0
MAX_AUDIO_SECONDS = 300.0
MAX_REFERENCE_SECONDS = 15.0
SEGMENT_PAUSE_SECONDS = 0.18
_SPEAKER = re.compile(r"^\s*(?:Speaker\s*)?(\d+)\s*(?:\{([^{}\n]*)\})?\s*:\s*", re.I | re.M)
_DIALOGUE = re.compile(r"<d>(.*?)</d>", re.I | re.S)
_DIRECTION = re.compile(r"\[([^\[\]]+)\]")
_LANGUAGES = {
    "english": "en", "french": "fr", "spanish": "es", "german": "de",
    "italian": "it", "portuguese": "pt", "chinese": "zh", "mandarin": "zh",
    "japanese": "ja", "korean": "ko", "russian": "ru", "arabic": "ar",
    "hindi": "hi", "dutch": "nl", "polish": "pl", "turkish": "tr",
}


@dataclass(frozen=True)
class AudioSegment:
    text: str
    speaker: int = 1
    direction: str = "natural, expressive delivery"
    language: str = "Original language"
    language_code: str | None = None
    duration_s: float = MIN_AUDIO_SECONDS
    seed: int = 0
    # Preserve a complete manual H3 prompt for a single pass, or a soundscape.
    native_prompt: str | None = None
    sound_only: bool = False


def audio_duration(seconds, *, single_segment=False):
    maximum = MAX_SEGMENT_SECONDS if single_segment else MAX_AUDIO_SECONDS
    try:
        duration = float(15 if seconds is None else seconds)
    except (TypeError, ValueError) as error:
        raise ValueError(f"H3 Voice Audio duration must be between 5 and {maximum:g} seconds") from error
    if not math.isfinite(duration) or not MIN_AUDIO_SECONDS <= duration <= maximum:
        raise ValueError(f"H3 Voice Audio duration must be between 5 and {maximum:g} seconds")
    return duration


def normalize_audio_settings(inputs, *, validate_prompt=True):
    """Shared Studio/API/Classic contract: one complete script per audio job."""
    from .packing import align_num_frames

    seconds = audio_duration(inputs.get("duration_seconds", 15))
    if validate_prompt:
        plan_audio_request(inputs.get("prompt"), seconds)
    frames = align_num_frames(round(min(seconds, MAX_SEGMENT_SECONDS) * 24))
    inputs.update({
        "resolution": "32x32", "duration_seconds": seconds, "video_length": frames,
        "sliding_window_size": frames, "sliding_window_overlap": 0,
        "sliding_window_discard_last_frames": 0, "force_fps": "24",
        "multi_prompts_gen_type": 2, "video_prompt_type": "", "image_prompt_type": "",
        "minimax_h3_multi_window": False, "minimax_h3_reference_sequence": False,
        "minimax_h3_turbo_mode": False, "minimax_h3_turbo_preset": "",
    })
    return inputs


def normalized_words(text):
    # Keep non-Latin speech intact for both timing and Whisper alignment.
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
    return re.findall(r"[\u3400-\u9fff\u3040-\u30ff]|[^\W_]+(?:'[^\W_]+)?", text)


def _spoken_turn(body, speaker=1, header_direction=""):
    if speaker not in (1, 2):
        raise ValueError("H3 Voice Audio supports Speaker 1 and Speaker 2")
    directions = [part.strip() for part in re.split(r"[,;]", header_direction or "") if part.strip()]
    for direction in _DIRECTION.findall(body):
        directions.extend(part.strip() for part in re.split(r"[,;]", direction) if part.strip())
    body = _DIRECTION.sub(" ", body)
    body = re.sub(r"</?d>", "", body, flags=re.I)
    text = re.sub(r"\s+", " ", body).strip().strip('"\u201c\u201d')
    if not normalized_words(text):
        raise ValueError(f"H3 Voice Audio Speaker {speaker} has no spoken text")
    language, code, acting = "Original language", None, []
    for direction in directions:
        key = direction.casefold()
        if key in _LANGUAGES:
            language, code = direction, _LANGUAGES[key]
        elif key in _LANGUAGES.values():
            language = next(name.title() for name, value in _LANGUAGES.items() if value == key)
            code = key
        else:
            acting.append(direction)
    return AudioSegment(text=text, speaker=speaker, direction="; ".join(acting) or "natural, expressive delivery",
                        language=language, language_code=code)


def _parse_turns(text):
    headers = list(_SPEAKER.finditer(text))
    if not headers:
        return [_spoken_turn(text)]
    if text[:headers[0].start()].strip():
        raise ValueError("Put H3 voice directions inside the first Speaker block in square brackets")
    turns, languages = [], {}
    for index, header in enumerate(headers):
        body = text[header.end():headers[index + 1].start() if index + 1 < len(headers) else len(text)]
        turn = _spoken_turn(body, int(header[1]), header[2])
        if turn.language_code:
            languages[turn.speaker] = (turn.language, turn.language_code)
        elif turn.speaker in languages:
            turn = replace(turn, language=languages[turn.speaker][0], language_code=languages[turn.speaker][1])
        turns.append(turn)
    return turns


def _natural_seconds(text):
    # Speech itself uses the shared 2.8 w/s pace. A brief lead-in/tail allowance
    # avoids chopping the last word; confident Whisper alignment removes it.
    return max(MIN_AUDIO_SECONDS, len(normalized_words(text)) / DIALOGUE_DEFAULT_WORDS_PER_SECOND
               + len(re.findall(r"[.!?;:\u3002\uff01\uff1f]", text)) * 0.12 + 1.4)


def _split_turn(turn):
    """Prefer sentence boundaries, then word boundaries; never discard script text."""
    pieces, current = [], ""
    sentences = re.split(r"(?<=[.!?;\u3002\uff01\uff1f])\s+", turn.text)
    for sentence in sentences:
        # A very long sentence still has to fit. Character tokens also support
        # scripts without spaces, such as Chinese and Japanese.
        tokens = re.findall(r"[\u3400-\u9fff\u3040-\u30ff]|[^\s\u3400-\u9fff\u3040-\u30ff]+|\s+", sentence)
        if current and _natural_seconds(current + " " + sentence) <= MAX_SEGMENT_SECONDS:
            current += " " + sentence
            continue
        if current:
            pieces.append(current.strip())
            current = ""
        for token in tokens:
            candidate = current + token
            if current.strip() and _natural_seconds(candidate) > MAX_SEGMENT_SECONDS:
                pieces.append(current.strip())
                current = token.lstrip()
            else:
                current = candidate
    if current.strip():
        pieces.append(current.strip())
    return [replace(turn, text=piece) for piece in pieces]


def plan_audio_request(prompt, seconds=15, seed=0):
    """Plan at most five minutes without ever sending over 45s to the model.

    Speech duration is a ceiling, not padding: an undersized budget is rejected
    before queueing and surplus time is left unused after the scripted speech.
    """
    duration = audio_duration(seconds)
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("Enter dialogue or an audio description")
    native = "subject_definitions:" in text and "detailed_description:" in text
    dialogue = list(_DIALOGUE.finditer(text)) if native else []
    sound = text.lower().startswith("sound:") or (native and not dialogue)
    if text.lower().startswith("sound:") and not text[6:].strip():
        raise ValueError("Enter an audio description after Sound:")
    base_seed = random.randrange(2**31) if seed is None or int(seed) < 0 else int(seed)
    if sound:
        count = math.ceil(duration / MAX_SEGMENT_SECONDS)
        return [AudioSegment(text="", native_prompt=text, sound_only=True, duration_s=duration / count,
                             seed=(base_seed + index * 1000) % (2**31)) for index in range(count)]
    if native and duration <= MAX_SEGMENT_SECONDS:
        spoken = " ".join(_spoken_turn(match[1]).text for match in dialogue)
        if len(normalized_words(spoken)) > duration * DIALOGUE_MAX_WORDS_PER_SECOND:
            raise ValueError("The H3 spoken script exceeds 3 words per second; increase the maximum audio duration")
        return [AudioSegment(text=spoken, native_prompt=text, duration_s=duration, seed=base_seed)]
    if native:
        # Long enhanced/manual prompts keep their literal <d> lines. Compile
        # only the current utterance so later dialogue cannot leak into a pass.
        turns = []
        previous_end = 0
        for match in dialogue:
            subjects = re.findall(r"\(S([12])\)", text[previous_end:match.start()], flags=re.I)
            turns.append(_spoken_turn(match[1], int(subjects[-1]) if subjects else 1))
            previous_end = match.end()
    else:
        turns = _parse_turns(text)
    segments = [segment for turn in turns for segment in _split_turn(turn)]
    pauses = SEGMENT_PAUSE_SECONDS * (len(segments) - 1)
    minimums = [max(MIN_AUDIO_SECONDS, len(normalized_words(segment.text)) / DIALOGUE_MAX_WORDS_PER_SECOND)
                for segment in segments]
    required = sum(minimums) + pauses
    if required > duration + 1e-6:
        raise ValueError(f"The complete H3 script needs at least {math.ceil(required)} seconds at up to 3 words per second "
                         f"including segment pauses; the selected maximum is {duration:g}s (limit 300s). "
                         "Increase the duration or shorten the script.")
    natural = [min(MAX_SEGMENT_SECONDS, _natural_seconds(segment.text)) for segment in segments]
    slack = sum(natural) - sum(minimums)
    scale = min(1.0, max(0.0, (duration - required) / slack)) if slack > 0 else 0.0
    return [replace(segment, duration_s=min(MAX_SEGMENT_SECONDS, minimum + (preferred - minimum) * scale),
                    seed=(base_seed + index * 1000) % (2**31))
            for index, (segment, minimum, preferred) in enumerate(zip(segments, minimums, natural))]


def segment_prompt(segment, has_reference=False):
    if segment.native_prompt is not None:
        return segment.native_prompt
    speaker = f"Speaker {segment.speaker} (S{segment.speaker})"
    subject = (f"<Audio 1> supplies only the voice timbre, accent and vocal identity of {speaker}."
               if has_reference else f"{speaker} has a distinct, stable natural voice.")
    return (f"subject_definitions: {subject} summary: [audio reference] One clean isolated dialogue turn. "
            "retention_analysis: Retain the reference voice only; do not copy its words, timing or background noise. "
            f"detailed_description: A neutral close-microphone studio shot. {speaker}, {segment.direction}, says exactly once "
            f"<d>[{segment.language}] {segment.text}</d> After the complete line, the speaker stops. "
            "overall_soundscape: Clean speech, natural breaths, no other voices, extra words or repetitions. "
            "non_diegetic_music: N/A")


def audio_request(prompt, seconds, guides=()):
    duration = audio_duration(seconds, single_segment=True)
    references = [{"type": "audio", "path": str(path), "role": f"Speaker {index + 1}",
                   "audio_intent": "voice"} for index, path in enumerate(guides) if path]
    if len(references) > 2:
        raise ValueError("H3 Voice Audio accepts at most two audio references")
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("Enter dialogue or an audio description")
    if "subject_definitions:" in text and "detailed_description:" in text:
        return text, duration, references
    definitions = " ".join(f"<Audio {index + 1}> supplies the vocal identity of speaker (S{index + 1})."
                           for index in range(len(references))) or "A single natural speaker (S1)."
    retention = " ".join(f"<Audio {index + 1}>: reference - preserve voice timbre only."
                         for index in range(len(references))) or "N/A"
    # A soundscape request is explicit; ordinary plain text is spoken verbatim.
    if text.lower().startswith("sound:"):
        detail, soundscape = "A static neutral view with no visible action.", text[6:].strip()
    else:
        parts = []
        for line in text.splitlines():
            match = re.match(r"\s*(?:speaker\s*)?([12])\s*:\s*(.+)", line, re.I)
            speaker, words = (int(match[1]), match[2]) if match else (1, line.strip())
            if not words:
                continue
            dialogue = words if "<d>" in words else f"<d>[English] {words}</d>"
            parts.append(f"Speaker (S{speaker}) says once, clearly and naturally, {dialogue}")
        detail = " ".join(parts) + " After the scripted lines, all speakers stop; the remaining time is silent."
        soundscape = "Clean natural speech, no additional words, repeated lines or vocalizations."
    native = (f"subject_definitions: {definitions} summary: [audio reference] A clear audio performance. "
              f"retention_analysis: {retention} detailed_description: {detail} "
              f"overall_soundscape: {soundscape} non_diegetic_music: N/A")
    return native, duration, references
