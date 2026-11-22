"""Single-shot Viggle Remake recipes adapted from WanGP 5b63ae9."""

from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path


MIN_FRAMES = 90
MAX_FRAMES = 345
FPS = 24
STEPS = 6
ALIGNED_GUIDE_LORA_FILE = "minimax_h3_character_swap_v1.1_ref2va_fro.safetensors"
ALIGNED_GUIDE_LORA_REPO = "Alissonerdx/Minimax-H3-ComfyUI"
ALIGNED_GUIDE_LORA_REVISION = "5a2770de134c2f006790dddcb9ecaad3d88836f2"
ALIGNED_GUIDE_LORA_SIZE = 80_673_112
ALIGNED_GUIDE_LORA_SHA256 = "ab21167bed71db99ac09527ad1f26f5f3517687ebf2e1f1bd5e91cc0bdf9b9c3"
ALIGNED_GUIDE_LORA_URL = (
    f"https://huggingface.co/{ALIGNED_GUIDE_LORA_REPO}/resolve/"
    f"{ALIGNED_GUIDE_LORA_REVISION}/loras/{ALIGNED_GUIDE_LORA_FILE}"
)
ALIGNED_GUIDE_DEFAULT_PROMPT = "The replacement character is shown in Picture 1."
_VERIFIED_ALIGNED_GUIDE_LORAS = set()
_SIGMAS = {
    "swap": (1.0, 0.990476, 0.965243, 0.923077, 0.8, 0.631579),
    "edit": (1.0, 0.983607, 0.96, 0.923077, 0.857143, 0.705882),
}


def sigmas(mode, shift=12.0):
    ratio = shift / 12.0
    return (*[ratio * value / (1.0 + (ratio - 1.0) * value) for value in _SIGMAS[mode]], 0.0)


def model_definition(base, mode):
    aligned_guide = mode == "swap_aligned"
    recipe_mode = "swap" if aligned_guide else mode
    if recipe_mode not in _SIGMAS:
        raise ValueError("Unknown Viggle Remake recipe.")
    result = {
        **base,
        "minimax_h3_viggle_remake": recipe_mode,
        "minimax_h3_lora_workflow": "ref2va" if aligned_guide else "fl2va",
        "omni_reference": bool(aligned_guide),
        "director_video_strategy": "",
        "director_reference_mode": "none",
        "director_audio_input_mode": "none",
        "director_shot_image_support": "none",
        "frames_minimum": MIN_FRAMES,
        "frames_maximum": MAX_FRAMES,
        "frames_selection_maximum": MAX_FRAMES,
        "sliding_window_size_locked": True,
        "sliding_window_defaults": {
            **base["sliding_window_defaults"],
            "window_min": MAX_FRAMES,
            "window_max": MAX_FRAMES,
            "window_default": MAX_FRAMES,
            "overlap_default": 17,
        },
        "sliding_window_auto_prompt_pacing": False,
        "extract_guide_from_window_start": True,
        "preserve_guide_frame_count": True,
        "minimax_h3_bundled_lora_reserve_gb": 3.5,
        "custom_frames_injection": False,
        "video_continuation": False,
        "control_video_trim": False,
        "set_video_prompt_type": "IVU2" if aligned_guide else "IVU",
        "image_prompt_types_allowed": "T",
        "any_image_refs_relative_size": False,
        "fit_into_canvas_image_refs": 1,
        "no_background_removal": True,
        "image_ref_choices": {"choices": [("Edited frames", "I")], "letters_filter": "I", "default": "I"},
        "guide_custom_choices": {"choices": [("Source video", "VU")], "letters_filter": "VU", "default": "VU"},
        "any_audio_prompt": True,
        "audio_prompt_choices": True,
        "output_audio_is_input_audio": True,
        "loras_disabled": True,
        "minimax_h3_turbo_eligible": False,
        "minimax_h3_two_pass_compatible": False,
        "minimax_h3_latent_upscaler_enabled": False,
        "first_block_cache": False,
        "lock_inference_steps": True,
        "inference_steps_min": STEPS,
        "inference_steps_max": STEPS,
        "lock_guidance_scale": True,
        "supports_latent_preview": False,
        "custom_settings": [],
        "runtime_custom_settings": [],
        "vae_upsamplers": {},
        "selector_help": (
            "Replace the main subject with one edited identity image. The aligned guide supplies each source frame at the matching output time; the original soundtrack is preserved."
            if aligned_guide
            else "Edit an existing shot with one to four painted frames and its original soundtrack."
        ),
    }
    if aligned_guide:
        result.update({
            "minimax_h3_aligned_guide": True,
            "minimax_h3_two_pass_compatible": False,
            "minimax_h3_turbo_eligible": False,
            "minimax_h3_accelerators_allowed": False,
            "loras_disabled": True,
            "lock_inference_steps": False,
            "lock_guidance_scale": False,
            # The BFS adapter uses the native Ref2VA sampler, not Viggle's
            # baked six-step recipe. Preserve the resolved checkpoint range.
            "inference_steps_min": max(2, int(base.get("inference_steps_min", 2))),
            "inference_steps_max": base.get("inference_steps_max", 50),
            "image_prompt_types_allowed": "I",
            "image_ref_choices": {
                "choices": [("Replacement identity", "I")],
                "letters_filter": "I",
                "default": "I",
            },
            "guide_custom_choices": {
                "choices": [("Source video — aligned guide", "VU2")],
                "letters_filter": "VU2",
                "default": "VU2",
            },
        })
    return result


def validate_remake_inputs(inputs, *, aligned_guide=False):
    if not inputs.get("video_guide"):
        raise ValueError("Viggle Remake needs a source video.")
    refs = inputs.get("image_refs")
    if aligned_guide:
        if not isinstance(refs, (list, tuple)) or len(refs) != 1:
            raise ValueError("BFS Aligned Swap needs exactly one edited replacement identity image.")
    elif not isinstance(refs, (list, tuple)) or not 1 <= len(refs) <= 4:
        raise ValueError("Viggle Remake needs one to four edited source frames.")
    if any(item is None or (isinstance(item, str) and not item.strip()) for item in refs):
        raise ValueError("Viggle Remake needs one to four edited source frames.")
    if inputs.get("video_mask") is not None:
        raise ValueError("This Viggle Remake recipe supports whole-shot edits.")
    return len(refs)


def verify_aligned_guide_lora(path):
    """Require the exact revision-pinned BFS adapter bytes."""

    candidate = Path(path).resolve(strict=True)
    if candidate.name != ALIGNED_GUIDE_LORA_FILE:
        raise ValueError("BFS Aligned Swap requires its pinned character-swap adapter.")
    stat = candidate.stat()
    key = (
        str(candidate),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        ALIGNED_GUIDE_LORA_SHA256,
    )
    if key in _VERIFIED_ALIGNED_GUIDE_LORAS:
        return str(candidate)
    if stat.st_size != ALIGNED_GUIDE_LORA_SIZE:
        raise ValueError("BFS Aligned Swap adapter size differs from the pinned upstream asset.")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ALIGNED_GUIDE_LORA_SHA256:
        raise ValueError("BFS Aligned Swap adapter checksum differs from the pinned upstream asset.")
    current = candidate.stat()
    if (current.st_size, current.st_mtime_ns, current.st_ctime_ns) != key[1:4]:
        raise ValueError("BFS Aligned Swap adapter changed during verification.")
    _VERIFIED_ALIGNED_GUIDE_LORAS.add(key)
    return str(candidate)


def validate_aligned_guide_lora_names(paths):
    """Require exactly the pinned BFS adapter without opening the asset."""

    if isinstance(paths, (str, os.PathLike)):
        candidates = [paths]
    else:
        try:
            candidates = list(paths or ())
        except TypeError as error:
            raise ValueError("BFS Aligned Swap needs its pinned character-swap adapter.") from error
    if len(candidates) != 1:
        raise ValueError("BFS Aligned Swap requires exactly its one pinned character-swap adapter.")
    candidate = os.fspath(candidates[0])
    if os.path.basename(candidate.replace("\\", "/")) != ALIGNED_GUIDE_LORA_FILE:
        raise ValueError("BFS Aligned Swap requires only its pinned character-swap adapter.")
    return candidate


def validate_aligned_guide_loras(paths):
    """Validate the one required adapter and its pinned bytes."""

    candidate = validate_aligned_guide_lora_names(paths)
    return verify_aligned_guide_lora(candidate)


def _seconds(value):
    if isinstance(value, bool):
        raise ValueError("Remake timestamps must be numbers in seconds.")
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Remake timestamps must be numbers in seconds.") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Remake timestamps must be finite and nonnegative.")
    return seconds


def frame_times(inputs, count):
    values = inputs.get("viggle_remake_frame_times")
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError("Set a source timestamp for every edited frame.")
    times = [_seconds(value) for value in values]
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("Edited frame timestamps must be in strictly increasing order.")
    return times


def normalize_request(inputs, *, aligned_guide=False):
    count = validate_remake_inputs(inputs, aligned_guide=aligned_guide)
    if not aligned_guide:
        frame_times(inputs, count)
    inputs.pop("_enhance_on_generation", None)
    if aligned_guide:
        if not str(inputs.get("prompt") or "").strip():
            inputs["prompt"] = ALIGNED_GUIDE_DEFAULT_PROMPT
        inputs.update({
            "prompt_enhancer": "",
            "video_prompt_type": "IVU2",
            "audio_prompt_type": str(inputs.get("audio_prompt_type") or ""),
            "activated_loras": [ALIGNED_GUIDE_LORA_URL],
            "loras_multipliers": "1.0",
            "minimax_h3_turbo_mode": False,
            "minimax_h3_turbo_preset": "",
            "sample_solver": "euler",
            "guidance_phases": 1,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "denoising_strength": 1.0,
            "masking_strength": 1.0,
            "image_mode": 0,
            "multi_prompts_gen_type": 2,
            "image_prompt_type": "I",
            "settings_version": 2.58,
        })
        return
    inputs.update({
        "num_inference_steps": STEPS,
        "guidance_scale": 1.0,
        "guidance_phases": 1,
        "flow_shift": 12.0,
        "sample_solver": "euler",
        "image_mode": 0,
        "multi_prompts_gen_type": 2,
        "prompt_enhancer": "",
        "image_prompt_type": "",
        "video_prompt_type": "IVU",
        "audio_prompt_type": "K",
        "force_fps": "24",
        "sliding_window_size": MAX_FRAMES,
        "sliding_window_overlap": 17,
        "sliding_window_discard_last_frames": 0,
        "minimax_h3_turbo_mode": False,
        "denoising_strength": 1.0,
        "masking_strength": 1.0,
        "skip_steps_cache_type": "",
        "settings_version": 2.58,
    })


def _remake_prompt(description, mode, times, *, aligned_guide=False):
    description = str(description).strip()
    if aligned_guide:
        identity = (
            "the character shown in <Picture 1>"
            if description == ALIGNED_GUIDE_DEFAULT_PROMPT
            else description or "the character shown in <Picture 1>"
        )
        return (
            "body_swap: subject_definitions: <Subject 1> is the replacement character shown in <Picture 1>; "
            f"the character identity is {identity}. The aligned guide supplies the source scene and performance frame by frame.\n\n"
            "summary:\n[video generation + reference] Replace the main guide subject with <Subject 1>, keeping the scene, camera, timing, and performance.\n\n"
            "detailed_description:\n[Shot 1] <Subject 1> takes the guide subject's place. Preserve the source framing, environment, lighting, body pose, gestures, head motion, gaze, speech, and facial expression at each matching frame. Keep the identity, face, hair, skin, body shape, and clothing from <Picture 1> stable.\n\n"
            "overall_soundscape:\nThe source soundtrack, unchanged.\n\nnon_diegetic_music:\nN/A"
        )
    if description.startswith("subject_definitions:"):
        return description
    intent = "video editing + character replacement" if mode == "swap" else "video editing"
    pictures = "\n".join(
        f"<Picture {index}> is an edited target frame at {time:.3f} seconds."
        for index, time in enumerate(times, 1)
    )
    return (
        f"subject_definitions:\n<Video 1> is the source video.\n{pictures}\n\n"
        f"summary:\n[{intent}] {description}\n\n"
        "retention_analysis:\n<Video 1>: Preserve camera, movement, cut timing and soundtrack. "
        "Retain the people and scene details that are outside the requested change.\n\n"
        f"detailed_description:\n[Shot 1] {description} Match each edited picture at its specified moment.\n\n"
        "overall_soundscape:\nThe source soundtrack, unchanged.\n\nnon_diegetic_music:\nN/A"
    )


@contextmanager
def prepare_remake_media(inputs, mode, *, resolve_media, aborted, update, aligned_guide=False):
    from services.viggle_media import prepare_control_media, probe_control_video
    from shared.utils.audio_video import extract_audio_tracks

    count = validate_remake_inputs(inputs, aligned_guide=aligned_guide)
    times = [] if aligned_guide else frame_times(inputs, count)
    source = resolve_media(inputs["video_guide"])
    if not source:
        raise ValueError("The Remake source video could not be found.")
    info = probe_control_video(source, aborted=aborted)
    duration = info["duration_seconds"]
    start = _seconds(inputs.get("viggle_trim_start") or 0)
    raw_end = inputs.get("viggle_trim_end")
    end = duration if raw_end is None else _seconds(raw_end)
    if duration < end <= duration + 1 / FPS:
        end = duration
    if not 0 <= start < end <= duration:
        raise ValueError("The Remake range must be inside the source video.")
    count_frames = math.floor((end - start) * FPS + 1e-6)
    if not MIN_FRAMES <= count_frames <= MAX_FRAMES:
        raise ValueError("Choose a single shot from 3.75 to 14.37 seconds.")
    if any(not start <= time < end for time in times):
        raise ValueError("Every edited frame must fall inside the Remake range.")
    settings = dict(inputs)
    normalize_request(settings, aligned_guide=aligned_guide)
    settings.update({
        "video_guide": str(source),
        "_viggle_trim_start": start,
        "_viggle_trim_end": end,
        "video_length": count_frames,
        "prompt": _remake_prompt(
            inputs.get("prompt") or "",
            mode,
            [time - start for time in times],
            aligned_guide=aligned_guide,
        ),
        "video_prompt_type": "IVU2" if aligned_guide else "IVU",
        "audio_prompt_type": "K" if extract_audio_tracks(str(source), query_only=True) else "",
    })
    with prepare_control_media(settings, resolve_media=resolve_media, aborted=aborted, update=update) as prepared:
        yield {**settings, **prepared}


def prepare_references(
    input_frames,
    images,
    history_count,
    frame_num,
    height,
    width,
    *,
    aligned_guide=False,
):
    import numpy as np
    import torch
    from PIL import Image

    from .packing import prepare_keyframe_image
    from .ref2va import MiniMaxH3PreparedReference

    video = input_frames
    if video is not None and isinstance(video, torch.Tensor) and video.ndim == 3:
        video = video.unsqueeze(1)
    if (
        video is None
        or not isinstance(video, torch.Tensor)
        or video.ndim != 4
        or video.shape[1] < 1
    ):
        video = None
    if video is None or not images:
        raise ValueError("Viggle Remake requires a source video and edited reference image.")
    if aligned_guide and len(images) != 1:
        raise ValueError("BFS Aligned Swap requires exactly one replacement identity image.")
    if not aligned_guide and not 1 <= len(images) <= 4:
        raise ValueError("Viggle Remake requires one to four edited frames.")
    references = []
    if not aligned_guide:
        from .viggle import _window_pixels

        pixels = _window_pixels(video, history_count, frame_num, height, width)
        references.append(MiniMaxH3PreparedReference(kind="video", frames=pixels))
    for item in images:
        if isinstance(item, Image.Image):
            image = item.convert("RGB")
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().to("cpu")
            if tensor.ndim == 4:
                tensor = tensor[:, 0]
            if tensor.ndim != 3:
                raise ValueError("A Viggle Remake edited frame must be a CHW image.")
            if tensor.dtype == torch.uint8:
                pixels = tensor.permute(1, 2, 0).numpy()
            else:
                pixels = tensor.float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
                pixels = pixels.permute(1, 2, 0).numpy()
            image = Image.fromarray(pixels).convert("RGB")
        else:
            image = Image.fromarray(np.asarray(item).astype(np.uint8)).convert("RGB")
        if image is None:
            raise ValueError("A Viggle Remake edited frame could not be decoded.")
        references.append(MiniMaxH3PreparedReference(
            kind="image", image=prepare_keyframe_image(image, height, width),
        ))
    return references


def text_presentation(prompt, frame_count, image_count, *, aligned_guide=False):
    if aligned_guide:
        if image_count != 1:
            raise ValueError("BFS Aligned Swap prompt conditioning requires exactly one identity image.")
        return f"<Picture 1>: replacement identity. {prompt}"
    timestamps = list(range(0, frame_count, FPS // 2))
    if len(timestamps) % 2:
        timestamps.append(timestamps[-1])
    times = "".join(
        f"<{(left + right) / (2 * FPS):.1f} seconds>"
        for left, right in zip(timestamps[::2], timestamps[1::2])
    )
    pictures = "".join(f"<Picture {index}>: " for index in range(1, image_count + 1))
    return f"<Video 1>: {times}{pictures}{prompt}"
