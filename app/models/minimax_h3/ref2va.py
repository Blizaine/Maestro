"""Official MiniMax H3 Ref2VA media preparation and packed layout.

This is a local-file-oriented port of the Hugging Face Diffusers Ref2VA
blocks pinned in UPSTREAM.md. Maestro keeps request validation separate from
media decoding so malformed jobs fail before they enter the generation queue.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    _fill_audio_condition_positions,
    _spatial_position_grid,
    _temporal_position_grid,
    _temporal_position_span,
    _unpack_condition_anchor,
    resolve_canvas_size,
)
from .reference_manifest import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    validate_reference_manifest,
)


MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2
_REFERENCE_TAG_RE = re.compile(r"<(?:Picture|Video|Audio)\s+\d+>", re.IGNORECASE)
_PICTURE_TAG_RE = re.compile(r"<Picture\s+(\d+)>", re.IGNORECASE)


@dataclass
class MiniMaxH3PreparedReference:
    """One prepared Ref2VA reference, kept in request order."""

    kind: str
    has_audio: bool = False
    image: Any = None
    frames: Any = None
    waveform: torch.Tensor | None = None
    block_timestamps: list[float] = field(default_factory=list)
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0
    role: str = ""
    audio_intent: str = ""
    image_intent: str = ""

    @property
    def num_video_rows(self) -> int:
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS


def ensure_ref2va_prompt_relationships(
    prompt: str,
    references,
    *,
    duration_seconds: float | None = None,
) -> str:
    """Compile a raw Ref2VA request into explicit six-field Context-IR.

    MiniMax H3 uses natural-language Context-IR to decide whether audio is
    copied/reused or merely referenced. Media tensors alone cannot communicate
    that distinction, so a raw Studio prompt receives a complete relationship
    map and literal dialogue tags. An already enhanced/tagged prompt is
    preserved verbatim.
    """

    text = str(prompt or "").strip()
    if _REFERENCE_TAG_RE.search(text):
        return text

    items = validate_reference_manifest(references, require_files=False)
    picture_index = 0
    video_index = 0
    audio_index = 0
    subject_index = 0
    relationships: list[str] = []
    retention: list[str] = []
    role_subjects: dict[str, int] = {}
    character_subjects: dict[str, int] = {}
    pending_voice_references: list[tuple[int, dict, str]] = []
    opening_subjects: list[str] = []

    for item in items:
        kind = item["type"]
        role = item.get("role") or f"the supplied {kind} reference"
        if kind == "image":
            picture_index += 1
            intent = item.get("image_intent", "identity")
            if intent == "composition":
                relationships.append(
                    f"<Picture {picture_index}> is a soft composition and cast-layout reference for {role} "
                    "that preserves the intended subjects, wardrobe, setting, and spatial arrangement "
                    "while generating a naturally moving opening rather than copying the picture as a "
                    "frozen first frame."
                )
                retention.append(
                    f"<Picture {picture_index}> ([Shot 1] composition anchor): weak_reference - "
                    "retain broad subject placement, setting, and spatial relationships."
                )
            elif intent == "scene":
                subject_index += 1
                relationships.append(
                    f"<Subject {subject_index}> is the environment and location for {role}, defined by "
                    f"<Picture {picture_index}>; its architecture, materials, lighting context, and scene "
                    "identity apply without treating incidental people as target character identities."
                )
                retention.append(
                    f"<Subject {subject_index}> (appears in [Shot 1]): fully_preserved - preserve the "
                    "referenced architecture, materials, lighting context, and location identity."
                )
                opening_subjects.append(
                    f"<Subject {subject_index}> establishes the requested environment and location."
                )
            elif intent == "style":
                subject_index += 1
                relationships.append(
                    f"<Subject {subject_index}> is the visual treatment for {role}, guided by "
                    f"<Picture {picture_index}>; use its medium, palette, lighting language, and texture "
                    "without copying its people, pose, framing, or exact composition."
                )
                retention.append(
                    f"<Subject {subject_index}>: weak_reference - retain broad similarity in medium, "
                    "palette, lighting language, and texture."
                )
            else:
                subject_index += 1
                role_subjects[str(role).strip().casefold()] = subject_index
                character_key = str(item.get("library_character_id") or "").strip()
                if character_key:
                    character_subjects[character_key] = subject_index
                relationships.append(
                    f"<Subject {subject_index}> is {role}, whose visual identity and appearance come from "
                    f"<Picture {picture_index}>; use it as identity evidence only, not as an opening "
                    "freeze-frame, source location, background, composition, framing, or pose."
                )
                retention.append(
                    f"<Subject {subject_index}> (appears in [Shot 1]): fully_preserved - preserve the "
                    f"identity and appearance defined by <Picture {picture_index}>."
                )
                opening_subjects.append(
                    f"<Subject {subject_index}> ({role}) appears with the referenced identity and "
                    "appearance in the described frame position and performs the requested action."
                )
            continue

        if kind == "video":
            next_video_index = video_index + 1
            has_soundtrack = bool(item.get("has_audio") or item.get("audio_path"))
            if has_soundtrack and item.get("include_audio", True):
                audio_index += 1
                relationships.append(
                    f"<Audio {audio_index}> is the synchronized soundtrack paired with "
                    f"<Video {next_video_index}>; reuse its audible timeline and synchronize "
                    "visible action and lip movement to it."
                )
                retention.append(
                    f"<Audio {audio_index}>: partially_copy - reuse the enabled soundtrack timeline "
                    "while allowing synchronized scene effects."
                )
            video_index = next_video_index
            video_intent = item.get("video_intent", "motion")
            if video_intent == "character":
                subject_index += 1
                role_subjects[str(role).strip().casefold()] = subject_index
                character_key = str(item.get("library_character_id") or "").strip()
                if character_key:
                    character_subjects[character_key] = subject_index
                relationships.append(
                    f"<Subject {subject_index}> is {role}, whose identity, appearance, and characteristic "
                    f"motion come from <Video {video_index}>; use the video as character evidence only, "
                    "not as the target opening frame, source location, background, composition, camera, "
                    "edit rhythm, or action to copy."
                )
                retention.append(
                    f"<Subject {subject_index}> (appears in [Shot 1]): fully_preserved - preserve the "
                    f"identity and appearance defined by <Video {video_index}> while generating the "
                    "requested target action and setting."
                )
                opening_subjects.append(
                    f"<Subject {subject_index}> ({role}) appears with the referenced identity and "
                    "appearance in the described frame position and performs the requested action."
                )
            elif video_intent == "scene":
                relationships.append(
                    f"<Video {video_index}> provides environment, lighting, and scene continuity for {role}; "
                    "do not copy incidental people as target character identities."
                )
                retention.append(
                    f"<Video {video_index}>: partially_preserved - retain the requested environment, "
                    "lighting, and scene continuity."
                )
            else:
                relationships.append(
                    f"<Video {video_index}> provides motion, camera, scene, and temporal reference for {role}."
                )
                retention.append(
                    f"<Video {video_index}>: partially_preserved - retain the requested motion, camera, "
                    "scene, and temporal structure."
                )
            continue

        audio_index += 1
        intent = item.get("audio_intent", "voice")
        if intent == "drive":
            relationships.append(
                f"<Audio {audio_index}> is the performance-driving audio timeline for {role} "
                "and supplies the audible content synchronized to visible action and lip movement."
            )
            retention.append(
                f"<Audio {audio_index}>: partially_copy - reuse its audible content and timeline "
                "while allowing synchronized scene effects."
            )
        elif intent == "style":
            relationships.append(
                f"<Audio {audio_index}> is an audio style, rhythm, and texture reference for {role} "
                "without copying its waveform, source words, or exact timing."
            )
            retention.append(
                f"<Audio {audio_index}>: weak_reference - retain broad similarity in sound, rhythm, "
                "texture, or music style."
            )
        else:
            character_key = str(item.get("library_character_id") or "").strip()
            mapped_subject = (
                character_subjects.get(character_key)
                if character_key else role_subjects.get(str(role).strip().casefold())
            )
            if mapped_subject is None and (character_key or str(role).strip()):
                pending_voice_references.append((audio_index, item, role))
                continue
            target = (
                f"<Subject {mapped_subject}> (S{mapped_subject})"
                if mapped_subject else str(role)
            )
            relationships.append(
                f"<Audio {audio_index}> is a voice-timbre, emotion, and delivery reference for {target}; "
                "generate only explicitly requested dialogue and do not copy its source words, timing, "
                "or waveform."
            )
            retention.append(
                f"<Audio {audio_index}>: reference - use its voice timbre, emotion, and delivery "
                "without copying the source signal, words, or timing."
            )

    for pending_audio_index, item, role in pending_voice_references:
        character_key = str(item.get("library_character_id") or "").strip()
        mapped_subject = (
            character_subjects.get(character_key)
            if character_key else role_subjects.get(str(role).strip().casefold())
        )
        target = (
            f"<Subject {mapped_subject}> (S{mapped_subject})"
            if mapped_subject else str(role)
        )
        relationships.append(
            f"<Audio {pending_audio_index}> is a voice-timbre, emotion, and delivery reference for {target}; "
            "generate only explicitly requested dialogue and do not copy its source words, timing, or waveform."
        )
        retention.append(
            f"<Audio {pending_audio_index}>: reference - use its voice timbre, emotion, and delivery "
            "without copying the source signal, words, or timing."
        )

    dialogue_counter = 0
    dialogue_word_count = 0

    def is_visible_text_quote(match) -> bool:
        before = text[max(0, match.start() - 150):match.start()]
        after = text[match.end():match.end() + 100]
        if re.search(
            r"(?i)\b(?:titled|entitled|called|named|captioned)\s*[:,-]?\s*$",
            before,
        ):
            return True
        visible_noun = re.search(
            r"(?i)\b(?:sign|banner|label|subtitle|caption|marquee|poster|billboard|"
            r"screen|monitor|display|neon|placard|headline|logo|shirt|door|wall)\b",
            before,
        )
        visible_cue = re.search(
            r"(?i)\b(?:reads?|reading|shows?|showing|displays?|displaying|bears?|"
            r"bearing|marked|printed|written|spells?|saying|with(?:\s+the)?\s+"
            r"(?:text|words?|lettering))\s*[:,-]?\s*$",
            before,
        )
        if visible_noun and visible_cue:
            return True
        return bool(re.match(
            r"(?i)^\s*(?:appears?|is\s+(?:visible|written|printed|displayed)|glows?)"
            r"\b[^.!?\r\n]{0,70}\b(?:on|across|above|below|behind|over)\b",
            after,
        ))

    def compile_dialogue(match):
        nonlocal dialogue_counter, dialogue_word_count
        if is_visible_text_quote(match):
            return match.group(0)
        dialogue_counter += 1
        words = (match.group(1) or match.group(2) or "").strip()
        dialogue_word_count += len(words.split())
        return f"(S{dialogue_counter}) <d>[English] {words}</d>"

    compiled_target = re.sub(
        r'"([^"\r\n]{1,500})"|“([^”\r\n]{1,500})”',
        compile_dialogue,
        text,
    )
    relationship_block = " ".join(relationships)
    retention_block = " ".join(retention)
    if dialogue_counter:
        duration = max(2.0, float(duration_seconds or 8.0))
        speech_duration = min(
            max(1.0, dialogue_word_count / 2.0),
            max(1.0, duration * 0.55),
        )
        dialogue_start = min(
            max(0.5, duration * 0.2),
            max(0.25, duration - speech_duration - 0.75),
        )
        dialogue_end = min(duration - 0.25, dialogue_start + speech_duration)
        dialogue_rule = (
            f"From 0.00 to {dialogue_start:.2f} seconds, show active scene-appropriate nonverbal "
            "action rather than idle staring; every mouth stays closed and the audio contains no "
            f"human voice. Begin the first tagged line at {dialogue_start:.2f} seconds and finish "
            f"all dialogue by {dialogue_end:.2f} seconds. From {dialogue_end:.2f} to "
            f"{duration:.2f} seconds, fill the remaining timeline with concrete nonverbal action, "
            "reactions, camera development, ambience, and synchronized practical effects. The tagged "
            "lines are the only spoken words; outside them there are no voices, whispers, grunts, "
            "audible breathing, or speech-like vocalizations, and every mouth remains closed."
        )
    else:
        dialogue_rule = (
            "Do not generate dialogue, voices, or speech-like vocalizations unless a <d> block is supplied."
        )
    has_mapped_music = any(item.get("audio_intent") in {"drive", "style"} for item in items)
    requests_music = bool(re.search(r"\b(?:music|song|score|soundtrack)\b", text, re.IGNORECASE))
    music_direction = (
        "Use only the mapped audio reference according to its assigned retention role."
        if has_mapped_music
        else "Follow only the music explicitly requested in the target description."
        if requests_music
        else "N/A"
    )
    task_types = ["reference generation"]
    if any(
        item.get("audio_intent") == "drive"
        or (
            item.get("type") == "video"
            and (item.get("has_audio") or item.get("audio_path"))
            and item.get("include_audio", True)
        )
        for item in items
    ):
        task_types.append("audio reuse")
    if any(
        item.get("type") == "audio"
        and item.get("audio_intent", "voice") in {"voice", "style"}
        for item in items
    ):
        task_types.append("audio reference")
    opening_subject_block = " ".join(opening_subjects)
    return (
        f"subject_definitions: {relationship_block}\n\n"
        f"summary: [{' + '.join(task_types)}] A finished video matching the requested action, "
        "identity, setting, reference roles, and explicitly tagged dialogue.\n\n"
        f"retention_analysis: {retention_block}\n\n"
        "detailed_description: The target video maintains the requested visual style, lighting, "
        "color, and cinematic texture. "
        f"[Shot 1] {opening_subject_block} The finished target video follows this request: "
        f"{compiled_target} {dialogue_rule}\n\n"
        "overall_soundscape: Continuous scene-appropriate stereo ambience and synchronized practical "
        "sound effects begin at the first frame and continue naturally underneath any scripted dialogue. "
        "Outside tagged dialogue there are no human voices, whispers, grunts, audible breathing, or "
        "speech-like vocalizations.\n\n"
        f"non_diegetic_music: {music_direction}"
    )


def add_ref2va_continuation_context(
    prompt: str,
    *,
    picture_offset: int = 1,
) -> str:
    """Reserve leading Picture labels for native Ref2VA continuation frames.

    Ref2VA presents a carried boundary frame to Qwen before the user's
    canonical references, exactly as upstream does for start/end conditions.
    The boundary therefore becomes ``<Picture 1>`` and only the user's
    Picture labels shift; Video and Audio numbering is unaffected. The
    transformer still receives the boundary as an exact keyframe condition,
    not as a general identity reference.
    """

    offset = max(0, int(picture_offset or 0))
    normalized = str(prompt or "").strip()
    if not normalized or offset <= 0:
        return normalized

    shifted = _PICTURE_TAG_RE.sub(
        lambda match: f"<Picture {int(match.group(1)) + offset}>",
        normalized,
    )
    continuity = (
        "<Picture 1> is the exact final frame carried from the preceding "
        "window. Use it only as the opening composition and motion boundary; "
        "continue forward without restaging it or treating it as a new "
        "identity reference."
    )
    retention = "<Picture 1>: exact opening-boundary continuity"

    subject_pattern = re.compile(
        r"(^\s*subject_definitions\s*:\s*)",
        re.IGNORECASE | re.MULTILINE,
    )
    if subject_pattern.search(shifted):
        shifted = subject_pattern.sub(
            lambda match: f"{match.group(1)}{continuity} ",
            shifted,
            count=1,
        )
    else:
        shifted = f"subject_definitions: {continuity}\n\n{shifted}"

    retention_pattern = re.compile(
        r"(^\s*retention_analysis\s*:\s*)",
        re.IGNORECASE | re.MULTILINE,
    )
    if retention_pattern.search(shifted):
        shifted = retention_pattern.sub(
            lambda match: f"{match.group(1)}{retention}; ",
            shifted,
            count=1,
        )
    else:
        shifted = f"{shifted}\n\nretention_analysis: {retention}"
    return shifted


def _decode_audio_stream(av, container, stream) -> tuple[torch.Tensor, int]:
    sample_rate = int(stream.codec_context.sample_rate)
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        chunks.extend(torch.from_numpy(item.to_ndarray()) for item in resampler.resample(frame))
    chunks.extend(torch.from_numpy(item.to_ndarray()) for item in resampler.resample(None))
    if not chunks:
        raise ValueError("The selected audio stream contains no decodable samples.")
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def decode_reference_video(path: str, *, decode_audio: bool = True):
    import av

    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream was found in {path}.")
            stream = container.streams.video[0]
            frames = []
            rotation = 0.0
            for frame in container.decode(stream):
                rotation = float(getattr(frame, "rotation", 0.0) or 0.0)
                frames.append(frame.to_ndarray(format="rgb24"))
            frame_rate = float(stream.average_rate or stream.guessed_rate or 0)
            soundtrack = None
            if decode_audio and container.streams.audio:
                container.seek(0)
                soundtrack = _decode_audio_stream(av, container, container.streams.audio[0])
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not decode reference video {os.path.basename(path)}: {error}") from error
    if not frames:
        raise ValueError(f"No video frames were found in {path}.")
    if frame_rate <= 0:
        raise ValueError(f"Reference video {os.path.basename(path)} has no valid frame rate.")
    pixels = np.stack(frames)
    turns = round(rotation / 90.0) % 4
    if turns:
        pixels = np.ascontiguousarray(np.rot90(pixels, k=-turns, axes=(1, 2)))
    return pixels, frame_rate, soundtrack


def decode_reference_audio(path: str) -> tuple[torch.Tensor, int]:
    import av

    try:
        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError(f"No audio stream was found in {path}.")
            return _decode_audio_stream(av, container, container.streams.audio[0])
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Could not decode reference audio {os.path.basename(path)}: {error}") from error


def reference_media_to_uint8(media) -> np.ndarray:
    if isinstance(media, list):
        return np.stack([reference_media_to_uint8(item) for item in media])
    if isinstance(media, Image.Image):
        return np.asarray(media.convert("RGB"))
    if isinstance(media, torch.Tensor):
        media = media.movedim(-3, -1).cpu().numpy()
    media = np.asarray(media)
    if media.dtype != np.uint8:
        media = (media * 255.0).round().clip(0, 255).astype(np.uint8)
    return media


def resolve_reference_image_size(
    width: int,
    height: int,
    *,
    detail: str = "match",
    target_height: int | None = None,
    target_width: int | None = None,
) -> tuple[int, int]:
    """Resolve official maximum detail or Maestro's consumer-friendly match size."""

    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    if detail == "max":
        scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    elif detail == "match":
        if not target_height or not target_width:
            raise ValueError("Matched reference detail needs the target height and width.")
        scale = min(1.0, math.sqrt((target_height * target_width) / float(height * width)))
    else:
        raise ValueError("Reference detail must be 'match' or 'max'.")
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def prepare_reference_image(image: Image.Image, height: int, width: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    if fps == MINIMAX_H3_FPS:
        return frames
    scale = MINIMAX_H3_FPS / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5))
    return np.repeat(frames, repeats, axis=0)


def resolve_reference_video_size(
    width: int,
    height: int,
    *,
    detail: str = "match",
    target_height: int | None = None,
    target_width: int | None = None,
) -> tuple[int, int]:
    """Resolve Ref2VA video detail without silently exceeding the output area.

    The official high-detail path keeps MiniMax's 768px-short-edge canvas.
    Maestro's default ``match`` path instead preserves the reference aspect
    ratio while bounding its pixel area to the requested output.  Reference
    video rows share the transformer's attention sequence with the generated
    clip, so decoding a 480/544p job's reference at 768p can more than double
    the denoising working set and exhaust a 24 GB card.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"A reference video must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference video must be within 1:4 and 4:1, got {width}x{height}.")
    if detail == "max":
        return resolve_canvas_size(width, height)
    if detail != "match":
        raise ValueError("Reference detail must be 'match' or 'max'.")
    if not target_height or not target_width:
        raise ValueError("Matched reference detail needs the target height and width.")

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    scale = min(1.0, math.sqrt((target_height * target_width) / float(height * width)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def prepare_reference_frames(
    frames: np.ndarray,
    num_frames: int,
    *,
    detail: str = "max",
    target_height: int | None = None,
    target_width: int | None = None,
) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"A reference video must contain RGB frames, got {tuple(frames.shape)}.")
    frames = frames[:num_frames]
    height, width = resolve_reference_video_size(
        frames.shape[2],
        frames.shape[1],
        detail=detail,
        target_height=target_height,
        target_width=target_width,
    )
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def prepare_reference_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
    max_duration: float,
    *,
    start_time: float = 0.0,
    pad_to_duration: bool = False,
) -> torch.Tensor:
    waveform = torch.as_tensor(waveform, device=torch.device("cpu"))
    if waveform.ndim != 2 or waveform.shape[0] not in (1, MINIMAX_H3_AUDIO_CHANNELS):
        raise ValueError(
            "A reference soundtrack must be a mono or stereo (channels, samples) waveform, "
            f"got {tuple(waveform.shape)}."
        )
    if sample_rate <= 0:
        raise ValueError(f"A reference soundtrack must have a positive sample rate, got {sample_rate}.")
    duration_samples = max(1, int(round(max_duration * sample_rate)))
    start_sample = max(0, int(round(max(0.0, float(start_time)) * sample_rate)))
    waveform = waveform.to(torch.float32)[:, start_sample : start_sample + duration_samples]
    if pad_to_duration and waveform.shape[-1] < duration_samples:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, duration_samples - waveform.shape[-1]),
        )
    if waveform.shape[0] == 1:
        waveform = waveform.expand(MINIMAX_H3_AUDIO_CHANNELS, -1).contiguous()
    if sample_rate != target_sample_rate:
        import torchaudio

        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
    if pad_to_duration:
        target_samples = max(1, int(round(max_duration * target_sample_rate)))
        waveform = waveform[:, :target_samples]
        if waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, target_samples - waveform.shape[-1]),
            )
    return waveform.contiguous()


def prepare_references(
    manifest,
    *,
    num_frames: int,
    target_height: int,
    target_width: int,
    audio_sample_rate: int = 32000,
    detail: str = "match",
    timeline_start_frame: int = 0,
) -> list[MiniMaxH3PreparedReference]:
    """Decode and prepare every reference without changing target geometry."""

    items = validate_reference_manifest(manifest, require_files=True)
    max_duration = num_frames / MINIMAX_H3_FPS
    timeline_start_frame = max(0, int(timeline_start_frame or 0))
    timeline_start_time = timeline_start_frame / MINIMAX_H3_FPS
    prepared: list[MiniMaxH3PreparedReference] = []

    for item in items:
        kind = item["type"]
        reference = MiniMaxH3PreparedReference(
            kind=kind,
            role=item.get("role", ""),
            audio_intent=item.get("audio_intent", ""),
            image_intent=item.get("image_intent", ""),
        )

        if kind == "image":
            with Image.open(item["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                height, width = resolve_reference_image_size(
                    *image.size,
                    detail=detail,
                    target_height=target_height,
                    target_width=target_width,
                )
                reference.image = prepare_reference_image(image, height, width).copy()
        elif kind == "video":
            wants_embedded_audio = bool(item.get("include_audio", True)) and not item.get("audio_path")
            if item.get("has_audio") is False:
                wants_embedded_audio = False
            frames, fps, soundtrack = decode_reference_video(item["path"], decode_audio=wants_embedded_audio)
            frames = resample_reference_frames(reference_media_to_uint8(frames), fps)
            source_height, source_width = frames.shape[1:3]
            reference.frames = prepare_reference_frames(
                frames,
                num_frames,
                detail=detail,
                target_height=target_height,
                target_width=target_width,
            )
            prepared_height, prepared_width = reference.frames.shape[1:3]
            print(
                "[MiniMax H3 Ref2VA] Prepared reference video "
                f"{source_width}x{source_height} -> {prepared_width}x{prepared_height} "
                f"({reference.frames.shape[0]} frames, detail={detail})."
            )
            if item.get("include_audio", True):
                if item.get("audio_path"):
                    soundtrack = decode_reference_audio(item["audio_path"])
                if soundtrack is not None:
                    waveform, sample_rate = soundtrack
                    reference.waveform = prepare_reference_waveform(
                        waveform, sample_rate, audio_sample_rate, max_duration
                    )
                    reference.has_audio = reference.waveform.shape[-1] > 0
        else:
            waveform, sample_rate = decode_reference_audio(item["path"])
            intent = item.get("audio_intent", "voice")
            follows_sequence_timeline = intent in {"drive", "style"}
            segment_start_time = timeline_start_time if follows_sequence_timeline else 0.0
            reference.waveform = prepare_reference_waveform(
                waveform,
                sample_rate,
                audio_sample_rate,
                max_duration,
                start_time=segment_start_time,
                pad_to_duration=follows_sequence_timeline,
            )
            reference.has_audio = True
            if follows_sequence_timeline:
                print(
                    "[MiniMax H3 Ref2VA] Prepared "
                    f"{intent} audio timeline segment "
                    f"{segment_start_time:.2f}-{segment_start_time + max_duration:.2f}s "
                    f"from {os.path.basename(item['path'])}."
                )

        prepared.append(reference)
    return prepared


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    stride = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    timestamps = [index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for index in range(0, len(timestamps), MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    ]
    return [frames[index] for index in indices], block_timestamps


def trim_reference_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"A reference video must have at least one frame, got {num_frames}.")
    return (
        max(1, (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK)
        * MINIMAX_H3_FRAMES_PER_CHUNK
        + MINIMAX_H3_LATENTS_PER_CHUNK
    )


def build_ref2va_presentation(
    tokenizer,
    prompt: str,
    references: list[MiniMaxH3PreparedReference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
) -> tuple[list[int], list[int]]:
    def text(value: str):
        token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return token_ids, [MINIMAX_H3_TEXT_TAG] * len(token_ids)

    def vision(pad_token: str, num_tokens: int):
        token_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return token_ids, [MINIMAX_H3_VIDEO_TAG] * len(token_ids)

    token_ids: list[int] = []
    token_tags: list[int] = []

    def emit(segment):
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if reference.kind == "image":
            counts["image"] += 1
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
        elif reference.kind == "video":
            counts["video"] += 1
            emit(text(f"<Video {counts['video']}>: "))
            for timestamp in reference.block_timestamps:
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
    emit(text(prompt))
    return token_ids, token_tags


def _reference_temporal_span(num_latent_frames: int) -> float:
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        for index in range(num_latent_frames)
    )


def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(position_ids, rows: slice, num_audio_latents: int, rotary_time: float, width_grid):
    if num_audio_latents == 0:
        return
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: list[MiniMaxH3PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors=(),
    audio_condition_anchors=(),
    target_condition_audio_latents: int = 0,
    target_condition_video_frames: int = 0,
) -> MiniMaxH3PackedSequence:
    """Build Ref2VA references plus optional native continuation history.

    The packed order mirrors WanGP 12.44: keyframe video, keyframe audio,
    canonical ordered references, target audio, then target video. Rotary
    time still places canonical references before the carried history and the
    newly generated target, so the references remain authoritative while the
    history supplies local motion and sound continuity.
    """

    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    target_frame_grid, target_width_grid = _frame_position_grid(
        latent_height,
        latent_width,
        patch_h,
        patch_w,
    )
    rows_per_target_frame = target_frame_grid.shape[0]
    num_target_video_rows = num_latent_frames * rows_per_target_frame
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    keyframe_frames = sum(
        _unpack_condition_anchor(anchor)[1]
        for anchor in keyframe_anchors
    )
    keyframe_video_rows = keyframe_frames * rows_per_target_frame
    keyframe_audio_latents = sum(
        int(anchor[1]) if isinstance(anchor, tuple) else 1
        for anchor in audio_condition_anchors
    )
    keyframe_audio_rows = (
        keyframe_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    )
    num_reference_video_rows = sum(reference.num_video_rows for reference in references if reference.kind != "audio")
    num_reference_audio_rows = sum(reference.num_audio_rows for reference in references)
    sequence_length = (
        num_text_tokens
        + keyframe_video_rows
        + keyframe_audio_rows
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    keyframe_start = num_text_tokens
    keyframe_audio_start = keyframe_start + keyframe_video_rows
    reference_start = keyframe_audio_start + keyframe_audio_rows
    video_indices: list[torch.Tensor] = (
        [torch.arange(keyframe_start, keyframe_audio_start)]
        if keyframe_video_rows
        else []
    )
    audio_indices: list[torch.Tensor] = (
        [torch.arange(keyframe_audio_start, reference_start)]
        if keyframe_audio_rows
        else []
    )
    cursor = reference_start
    rotary_time = float(num_text_tokens)
    for reference in references:
        if reference.kind == "image":
            rows = slice(cursor, cursor + reference.num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(reference.latent_height, reference.latent_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif reference.kind == "audio":
            rows = slice(cursor, cursor + reference.num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, reference.num_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(reference.num_audio_latents)
        elif reference.kind == "video":
            audio_rows = slice(cursor, cursor + reference.num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + reference.num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))
            frame_grid, width_grid = _frame_position_grid(
                reference.latent_height, reference.latent_width, patch_h, patch_w
            )
            _fill_audio_positions(position_ids, audio_rows, reference.num_audio_latents, rotary_time, width_grid)
            frame_time = _temporal_position_grid(reference.num_latent_frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.num_latent_frames, 1)
            rotary_time += max(float(reference.num_audio_latents), _reference_temporal_span(reference.num_latent_frames))
        else:
            raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}.")

    history_frames = sum(
        _unpack_condition_anchor(anchor)[1]
        for anchor in keyframe_anchors
        if _unpack_condition_anchor(anchor)[0] == "history"
    )
    target_origin = rotary_time + _temporal_position_span(history_frames)
    target_times = _temporal_position_grid(
        num_latent_frames,
        target_origin,
    )
    condition_cursor = keyframe_start
    history_time = rotary_time
    for entry in keyframe_anchors:
        anchor, condition_frames, frame_index = _unpack_condition_anchor(entry)
        if condition_frames <= 0:
            raise ValueError(
                "MiniMax H3 Ref2VA condition anchors must contain at least "
                f"one latent frame, got {entry!r}."
            )
        rows = slice(
            condition_cursor,
            condition_cursor + condition_frames * rows_per_target_frame,
        )
        condition = position_ids[rows].view(
            condition_frames,
            rows_per_target_frame,
            3,
        )
        if anchor == "history":
            condition[:, :, 0] = _temporal_position_grid(
                condition_frames,
                history_time,
            )[:, None]
            history_time += _temporal_position_span(condition_frames)
        elif anchor == "first":
            condition[:, :, 0] = target_times[:condition_frames, None]
        elif anchor == "last":
            condition[:, :, 0] = (
                target_origin
                + _temporal_position_span(num_latent_frames)
                - _ROPE_FRAME_RESCALE
            )
        elif anchor == "frame":
            if frame_index is None:
                raise ValueError(
                    "A MiniMax H3 Ref2VA 'frame' condition needs a target "
                    "frame index."
                )
            condition[:, :, 0] = (
                target_origin + frame_index * _ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(
                f"Unknown MiniMax H3 Ref2VA keyframe anchor {anchor!r}."
            )
        condition[:, :, 1:] = target_frame_grid[None]
        condition_cursor = rows.stop

    _fill_audio_condition_positions(
        position_ids,
        keyframe_audio_start,
        audio_condition_anchors,
        rotary_time,
        target_origin,
        target_width_grid,
    )
    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        target_origin,
        target_width_grid,
    )
    position_ids[video_start:, 0] = target_times.repeat_interleave(
        target_frame_grid.shape[0]
    )
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=(
            keyframe_video_rows + num_reference_video_rows
        ),
        num_condition_audio_rows=(
            keyframe_audio_rows + num_reference_audio_rows
        ),
        num_target_condition_audio_latents=max(
            0,
            int(target_condition_audio_latents),
        ),
        num_target_condition_video_rows=(
            max(0, int(target_condition_video_frames))
            * rows_per_target_frame
        ),
    )
