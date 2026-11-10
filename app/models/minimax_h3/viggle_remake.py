"""Single-shot Viggle Remake recipes adapted from WanGP 5b63ae9."""

from contextlib import contextmanager
import math


MIN_FRAMES = 90
MAX_FRAMES = 345
FPS = 24
STEPS = 6
_SIGMAS = {
    "swap": (1.0, 0.990476, 0.965243, 0.923077, 0.8, 0.631579),
    "edit": (1.0, 0.983607, 0.96, 0.923077, 0.857143, 0.705882),
}


def sigmas(mode, shift=12.0):
    ratio = shift / 12.0
    return (*[ratio * value / (1.0 + (ratio - 1.0) * value) for value in _SIGMAS[mode]], 0.0)


def model_definition(base, mode):
    if mode not in _SIGMAS:
        raise ValueError("Unknown Viggle Remake recipe.")
    return {
        **base,
        "minimax_h3_viggle_remake": mode,
        "minimax_h3_lora_workflow": "fl2va",
        "omni_reference": False,
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
        "set_video_prompt_type": "IVU",
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
        "selector_help": "Edit an existing shot with one to four painted frames and its original soundtrack.",
    }


def validate_remake_inputs(inputs):
    if not inputs.get("video_guide"):
        raise ValueError("Viggle Remake needs a source video.")
    refs = inputs.get("image_refs")
    if not isinstance(refs, (list, tuple)) or not 1 <= len(refs) <= 4:
        raise ValueError("Viggle Remake needs one to four edited source frames.")
    if any(item is None or (isinstance(item, str) and not item.strip()) for item in refs):
        raise ValueError("Viggle Remake needs one to four edited source frames.")
    if inputs.get("video_mask") is not None:
        raise ValueError("This Viggle Remake recipe supports whole-shot edits.")
    return len(refs)


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


def normalize_request(inputs):
    frame_times(inputs, validate_remake_inputs(inputs))
    inputs.pop("_enhance_on_generation", None)
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


def _remake_prompt(description, mode, times):
    description = str(description).strip()
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
def prepare_remake_media(inputs, mode, *, resolve_media, aborted, update):
    from services.viggle_media import prepare_control_media, probe_control_video
    from shared.utils.audio_video import extract_audio_tracks

    count = validate_remake_inputs(inputs)
    times = frame_times(inputs, count)
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
    normalize_request(settings)
    settings.update({
        "video_guide": str(source),
        "_viggle_trim_start": start,
        "_viggle_trim_end": end,
        "video_length": count_frames,
        "prompt": _remake_prompt(inputs["prompt"], mode, [time - start for time in times]),
        "audio_prompt_type": "K" if extract_audio_tracks(str(source), query_only=True) else "",
    })
    with prepare_control_media(settings, resolve_media=resolve_media, aborted=aborted, update=update) as prepared:
        yield {**settings, **prepared}


def prepare_references(input_frames, images, history_count, frame_num, height, width):
    from .minimax_h3_main import _as_video_tensor, _tensor_to_pil
    from .packing import prepare_keyframe_image
    from .ref2va import MiniMaxH3PreparedReference
    from .viggle import _window_pixels

    video = _as_video_tensor(input_frames)
    if video is None or not images or not 1 <= len(images) <= 4:
        raise ValueError("Viggle Remake requires a video and one to four edited frames.")
    pixels = _window_pixels(video, history_count, frame_num, height, width)
    references = [MiniMaxH3PreparedReference(kind="video", frames=pixels)]
    for item in images:
        image = _tensor_to_pil(item)
        if image is None:
            raise ValueError("A Viggle Remake edited frame could not be decoded.")
        references.append(MiniMaxH3PreparedReference(
            kind="image", image=prepare_keyframe_image(image, height, width),
        ))
    return references


def text_presentation(prompt, frame_count, image_count):
    timestamps = list(range(0, frame_count, FPS // 2))
    if len(timestamps) % 2:
        timestamps.append(timestamps[-1])
    times = "".join(
        f"<{(left + right) / (2 * FPS):.1f} seconds>"
        for left, right in zip(timestamps[::2], timestamps[1::2])
    )
    pictures = "".join(f"<Picture {index}>: " for index in range(1, image_count + 1))
    return f"<Video 1>: {times}{pictures}{prompt}"
