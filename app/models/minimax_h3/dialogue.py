"""H3 audio assembly and Whisper boundary alignment.

Whisper alignment is adapted from Wan2GP v12.71, commit
1e1dd2757f24923f008593d9d4ec09062234be20 (WanGP Community License 2.0).
See UPSTREAM.md and ../../LICENSES/WanGP-Community-2.0.txt.
Maestro owns planning, queue integration, reference files and cancellation.
"""
from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torchaudio.functional as audio_F

from .voice_audio import (
    MAX_REFERENCE_SECONDS, SEGMENT_PAUSE_SECONDS, normalized_words as _normalize_words,
    plan_audio_request, segment_prompt,
)

H3_DIALOGUE_SILENCE_THRESHOLD = 0.012
H3_DIALOGUE_BOUNDARY_PADDING_SECONDS = 0.14


def load_dialogue_whisper() -> torch.nn.Module:
    from shared.deepy.transcription import _load_whisper_medium

    model = _load_whisper_medium(torch.device("cpu"))
    alignment_heads = model.alignment_heads
    del model._buffers["alignment_heads"]
    object.__setattr__(model, "alignment_heads", alignment_heads)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module._lock_dtype = torch.float32
    model._offload_hooks = ["transcribe"]
    model._model_dtype = torch.float16
    model._budget = 0
    return model.eval().requires_grad_(False)


def _fuzzy_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    distances = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        diagonal = distances[0]
        distances[0] = row
        for column, right_char in enumerate(right, 1):
            previous = distances[column]
            distances[column] = diagonal if left_char == right_char else 1 + min(diagonal, distances[column], distances[column - 1])
            diagonal = previous
    return 1.0 - distances[-1] / max(len(left), len(right)) >= 0.55


def _align_words(transcribed: list[str], expected: list[str]) -> list[tuple[str, int | None]]:
    rows, columns, gap = len(transcribed), len(expected), -1
    scores = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        scores[row][0] = scores[row - 1][0] + gap
    for column in range(1, columns + 1):
        scores[0][column] = scores[0][column - 1] + gap
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            diagonal = scores[row - 1][column - 1] + (2 if _fuzzy_match(transcribed[row - 1], expected[column - 1]) else -1)
            scores[row][column] = max(diagonal, scores[row - 1][column] + gap, scores[row][column - 1] + gap)
    aligned = []
    row, column = rows, columns
    while row or column:
        match_score = 2 if row and column and _fuzzy_match(transcribed[row - 1], expected[column - 1]) else -1
        if row and column and scores[row][column] == scores[row - 1][column - 1] + match_score:
            aligned.append(("match" if match_score == 2 else "substitution", column - 1))
            row -= 1
            column -= 1
        elif row and scores[row][column] == scores[row - 1][column] + gap:
            aligned.append(("insertion", None))
            row -= 1
        else:
            column -= 1
    return list(reversed(aligned))


def _transcribe_words(model: torch.nn.Module, audio: torch.Tensor, sample_rate: int, language: str | None) -> list[dict]:
    mono = audio.detach().float().mean(dim=0, keepdim=True).cpu()
    if int(sample_rate) != 16000:
        mono = audio_F.resample(mono, int(sample_rate), 16000)
    result = model.transcribe(mono.squeeze(0).numpy(), language=language, word_timestamps=True,
                              fp16=getattr(model, "_model_dtype", torch.float32) == torch.float16, verbose=None)
    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            for normalized in _normalize_words(word.get("word", "")):
                words.append({"word": normalized, "start": float(word.get("start", 0.0)), "end": float(word.get("end", 0.0)),
                              "probability": float(word.get("probability", 1.0))})
    return words


def _reliable_boundary_word(word: dict) -> bool:
    duration = word["end"] - word["start"]
    return word["probability"] >= 0.35 and duration <= 0.5 + 0.15 * len(word["word"])


def _quiet_boundary(mono: torch.Tensor, sample_rate: int, start: int, stop: int, prefer_last: bool) -> int | None:
    start, stop = max(0, start), min(mono.numel(), stop)
    hop = max(1, round(sample_rate * 0.01))
    quiet = []
    for position in range(start, stop, hop):
        window = mono[position:min(stop, position + hop)]
        if window.numel() and float(window.square().mean().sqrt()) < H3_DIALOGUE_SILENCE_THRESHOLD:
            quiet.append(position)
    if not quiet:
        return None
    return quiet[-1] if prefer_last else quiet[0]


def trim_dialogue_surplus(model: torch.nn.Module, audio: torch.Tensor, sample_rate: int, expected_text: str,
                          language: str | None, verbose: bool = False) -> torch.Tensor:
    expected = _normalize_words(expected_text)
    transcribed = _transcribe_words(model, audio, sample_rate, language)
    if not expected or not transcribed:
        return audio
    alignment = _align_words([word["word"] for word in transcribed], expected)
    matching = sum(label == "match" for label, _ in alignment)
    if matching < max(1, math.ceil(len(expected) * 0.5)):
        if verbose:
            print("[MiniMax H3 Dialogue] Whisper alignment was not confident enough to trim this segment")
        return audio
    first = next((index for index, (_, expected_index) in enumerate(alignment) if expected_index == 0), None)
    last = next((index for index in range(len(alignment) - 1, -1, -1) if alignment[index][1] == len(expected) - 1), None)
    if first is None or last is None:
        return audio
    mono = audio.detach().float().mean(dim=0).cpu()
    start_sample, stop_sample = 0, audio.shape[-1]
    leading_insertion = first > 0 and any(label == "insertion" for label, _ in alignment[:first])
    # Never advance past an uncertain *expected* word to find a convenient
    # boundary: that can delete the requested opening word (for example Yes).
    reliable_first = alignment[first][0] == "match" and _reliable_boundary_word(transcribed[first])
    if reliable_first and (leading_insertion or transcribed[first]["start"] > 0.5):
        intended_start = round(transcribed[first]["start"] * sample_rate)
        padding = round(H3_DIALOGUE_BOUNDARY_PADDING_SECONDS * sample_rate)
        search_start = max(0, intended_start - round(0.5 * sample_rate))
        start_sample = _quiet_boundary(mono, sample_rate, search_start, max(search_start + 1, intended_start - padding), True)
        start_sample = max(0, intended_start - padding) if start_sample is None else start_sample
    trailing_duration = audio.shape[-1] / sample_rate - transcribed[last]["end"]
    trailing_insertion = last + 1 < len(transcribed) and any(label == "insertion" for label, _ in alignment[last + 1:])
    reliable_last = alignment[last][0] == "match" and _reliable_boundary_word(transcribed[last])
    if reliable_last and (trailing_insertion or trailing_duration > 0.5):
        intended_end = round(transcribed[last]["end"] * sample_rate)
        padding = round(H3_DIALOGUE_BOUNDARY_PADDING_SECONDS * sample_rate)
        search_stop = min(audio.shape[-1], intended_end + round(0.5 * sample_rate))
        padded_end = min(search_stop, intended_end + padding)
        stop_sample = _quiet_boundary(mono, sample_rate, padded_end, search_stop, False)
        stop_sample = min(audio.shape[-1], intended_end + padding) if stop_sample is None else max(padded_end, stop_sample)
    if stop_sample <= start_sample:
        return audio
    if verbose and (start_sample or stop_sample < audio.shape[-1]):
        print(f"[MiniMax H3 Dialogue] Whisper trimmed {start_sample / sample_rate:.2f}s before and {(audio.shape[-1] - stop_sample) / sample_rate:.2f}s after the expected speech")
    return audio[..., start_sample:stop_sample]


def _write_reference(path, audio, sample_rate):
    import soundfile as sf

    sf.write(str(path), audio.detach().float().cpu().transpose(0, 1).numpy(), sample_rate, subtype="FLOAT")
    return str(path)


def _voice_references(folder, guides):
    """Keep the uploaded-reference budget separate from the output duration."""
    from .ref2va import decode_reference_audio, prepare_reference_waveform

    decoded = {speaker: decode_reference_audio(path) for speaker, path in guides.items()}
    total = sum(audio.shape[-1] / rate for audio, rate in decoded.values())
    budget = MAX_REFERENCE_SECONDS / len(decoded) if total > MAX_REFERENCE_SECONDS else MAX_REFERENCE_SECONDS
    paths = {}
    for speaker, (audio, rate) in decoded.items():
        if audio.shape[-1] / rate < 2:
            raise ValueError(f"Audio Reference {speaker} must be at least 2 seconds long")
        audio = prepare_reference_waveform(audio, rate, 32000, budget, pad_to_duration=False)
        paths[speaker] = _write_reference(Path(folder) / f"uploaded-speaker-{speaker}.wav", audio, 32000)
    return paths


def generate_audio(pipeline, input_prompt, *, duration_seconds=15, audio_guide=None, audio_guide2=None,
                   audio_prompt_type="", seed=0, sampling_steps=20, callback=None,
                   set_progress_status=None, **generation_kwargs):
    """Generate bounded passes, align speech once, and publish one audio result."""
    segments = plan_audio_request(input_prompt, duration_seconds, seed)
    flags = str(audio_prompt_type or "").upper()
    guides = {speaker: path for speaker, flag, path in ((1, "A", audio_guide), (2, "B", audio_guide2)) if flag in flags}
    if any(not path for path in guides.values()):
        raise ValueError("Upload an audio reference for every selected H3 voice")

    def status(message):
        if set_progress_status is not None:
            set_progress_status(message)

    status(f"Planning H3 audio ({len(segments)} segments, up to {float(duration_seconds):g}s)")
    if pipeline._interrupt:
        return None
    generated = []
    # References are private job intermediates, never library outputs. The
    # context cleans them on success, cancellation, and generation failure.
    with TemporaryDirectory(prefix="maestro-h3-voice-") as folder:
        references = _voice_references(folder, guides) if guides else {}
        for index, segment in enumerate(segments):
            if pipeline._interrupt:
                return None
            label = f"H3 audio segment {index + 1}/{len(segments)}"
            status(f"{label} ({segment.duration_s:.1f}s)")

            def segment_callback(*args, **kwargs):
                kwargs["denoising_extra"] = label
                if kwargs.get("override_num_inference_steps", -1) > 0:
                    kwargs["total_steps_hint"] = kwargs["override_num_inference_steps"] * len(segments)
                return callback(*args, **kwargs)

            if segment.native_prompt is not None:
                # A complete native prompt retains its original Audio N order.
                first, second = references.get(1), references.get(2)
                segment_flags = ("A" if first else "") + ("B" if second else "")
            else:
                first, second = references.get(segment.speaker), None
                segment_flags = "A" if first else ""
            result = pipeline.generate(
                input_prompt=segment_prompt(segment, bool(first)), duration_seconds=segment.duration_s,
                audio_guide=first, audio_guide2=second, audio_prompt_type=segment_flags,
                seed=segment.seed, sampling_steps=sampling_steps,
                callback=segment_callback if callback is not None else None,
                set_progress_status=lambda message: status(f"{label} | {message}"),
                _audio_segment=True, _audio_speaker=segment.speaker if segment.native_prompt is None else None,
                **generation_kwargs,
            )
            if result is None or pipeline._interrupt:
                return None
            audio = result["x"].detach().float().cpu()
            sample_rate = int(result["audio_sampling_rate"])
            if sample_rate != 32000 or audio.ndim != 2 or audio.shape[0] != 2 or not audio.shape[-1]:
                raise ValueError("H3 audio segment did not return nonempty 32 kHz stereo audio")
            audio = audio[..., :round(segment.duration_s * sample_rate)]
            generated.append((segment, audio))
            if not segment.sound_only and segment.native_prompt is None and segment.speaker not in references:
                # The first generated turn anchors this speaker for all later
                # turns. Take its beginning so a long trailing silence cannot
                # replace the voice reference.
                reference = audio[..., :round(MAX_REFERENCE_SECONDS * sample_rate)]
                references[segment.speaker] = _write_reference(
                    Path(folder) / f"generated-speaker-{segment.speaker}.wav", reference, sample_rate)

        aligned = []
        for index, (segment, audio) in enumerate(generated):
            if pipeline._interrupt:
                return None
            if segment.text and not segment.sound_only:
                status(f"H3 audio segment {index + 1}/{len(segments)} | Trimming script boundaries")
                audio = trim_dialogue_surplus(pipeline.dialogue_whisper, audio, sample_rate,
                                              segment.text, segment.language_code, verbose=True)
            if pipeline._interrupt:
                return None
            aligned.append(audio)

    pause_samples = 0 if segments[0].sound_only else round(SEGMENT_PAUSE_SECONDS * sample_rate)
    # Maestro may set Torch's default device to CUDA for the model. Decoded
    # segments live on CPU, including for multi-minute output assembly.
    pause = torch.zeros((2, pause_samples), dtype=aligned[0].dtype, device="cpu")
    pieces = [piece for index, audio in enumerate(aligned) for piece in ((pause,) if index else ()) + (audio,)]
    output = torch.cat(pieces, dim=-1)[..., :round(float(duration_seconds) * sample_rate)]
    actual_duration = output.shape[-1] / sample_rate
    status(f"Combined H3 audio ({len(segments)} segments, {actual_duration:.1f}s)")
    return {"x": output, "audio_sampling_rate": sample_rate,
            "overridden_inputs": {"resolution": "32x32", "video_length": round(actual_duration * 24),
                                  "duration_seconds": round(actual_duration, 3), "audio_segment_count": len(segments)}}
