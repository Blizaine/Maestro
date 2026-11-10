"""Open-H3-Regenerate-2K-IC finishing for an already rendered draft.

Reference conditioning and the three-step HyperFlow tail follow the
Efficient-Large-Model model card and WanGP revision 5b63ae9.
"""

from contextlib import contextmanager
import math


STEPS = 3
TASK = (
    "Enhance <Video 1> into a detailed high-resolution version of the same clip. "
    "Preserve identities, layout, lettering, camera movement, actions, timing, and lip sync."
)


def model_definition(base):
    return {
        **base,
        "minimax_h3_regenerate_ic": True,
        "omni_reference": False,
        "director_video_strategy": "",
        "director_reference_mode": "none",
        "director_audio_input_mode": "none",
        "director_shot_image_support": "none",
        "frames_minimum": 90,
        "frames_maximum": 345,
        "frames_selection_maximum": 345,
        "sliding_window_size_locked": True,
        "sliding_window_defaults": {
            **base["sliding_window_defaults"],
            "window_min": 345, "window_max": 345, "window_default": 345,
            "overlap_default": 17,
        },
        "sliding_window_auto_prompt_pacing": False,
        "extract_guide_from_window_start": True,
        "preserve_guide_frame_count": True,
        # MMGP's grouped projections expand the three shipped adapters
        # to about 10.3 GiB. Account for them outside the base checkpoint.
        "minimax_h3_bundled_lora_reserve_gb": 10.3,
        "control_video_trim": False,
        "set_video_prompt_type": "VU",
        "guide_custom_choices": {"choices": [("Source draft", "VU")], "letters_filter": "VU", "default": "VU"},
        "video_continuation": False,
        "custom_frames_injection": False,
        "output_audio_is_input_audio": True,
        "any_audio_prompt": True,
        "audio_prompt_choices": True,
        "loras_disabled": True,
        "lock_inference_steps": True,
        "inference_steps_min": STEPS,
        "inference_steps_max": STEPS,
        "lock_guidance_scale": True,
        "minimax_h3_two_pass_compatible": False,
        "minimax_h3_latent_upscaler_enabled": True,
        "first_block_cache": False,
        "supports_latent_preview": False,
        "custom_settings": [],
        "runtime_custom_settings": [],
        "vae_upsamplers": {},
        "selector_help": "Finish an existing draft at 2K with its original soundtrack.",
    }


def normalize_request(inputs):
    if not inputs.get("video_guide"):
        raise ValueError("Choose the draft to finish at 2K.")
    if inputs.get("video_mask") is not None:
        raise ValueError("Regenerate 2K IC refines the full source clip.")
    inputs.pop("_enhance_on_generation", None)
    inputs.update({
        "prompt": str(inputs.get("prompt") or TASK),
        "num_inference_steps": STEPS,
        "guidance_scale": 1.0,
        "guidance_phases": 1,
        "flow_shift": 12.0,
        "sample_solver": "euler",
        "image_mode": 0,
        "multi_prompts_gen_type": 2,
        "prompt_enhancer": "",
        "image_prompt_type": "",
        "video_prompt_type": "VU",
        "audio_prompt_type": "K",
        "force_fps": "24",
        "sliding_window_size": 345,
        "sliding_window_overlap": 17,
        "sliding_window_discard_last_frames": 0,
        "minimax_h3_turbo_mode": False,
        "denoising_strength": 1.0,
        "masking_strength": 1.0,
        "skip_steps_cache_type": "",
        "settings_version": 2.58,
    })


def target_canvas(width, height):
    if min(width, height) < 1:
        raise ValueError("The draft has an invalid canvas.")
    max_width, max_height = (2560, 1440) if width >= height else (1440, 2560)
    scale = min(max_width / width, max_height / height)
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


@contextmanager
def prepare_media(inputs, *, resolve_media, aborted, update):
    from services.viggle_media import probe_control_video
    from shared.utils.audio_video import extract_audio_tracks
    from shared.utils.utils import get_video_info

    settings = dict(inputs)
    normalize_request(settings)
    source = resolve_media(settings["video_guide"])
    if not source:
        raise ValueError("The source draft could not be found.")
    info = probe_control_video(source, aborted=aborted)
    count = math.floor(info["duration_seconds"] * 24 + 1e-6)
    if not 90 <= count <= 345:
        raise ValueError("Choose a single draft from 3.75 to 14.37 seconds.")
    _, width, height, _ = get_video_info(str(source))
    out_width, out_height = target_canvas(width, height)
    settings.update({
        "video_guide": str(source),
        "video_length": count,
        "resolution": f"{out_width}x{out_height}",
        "audio_prompt_type": "K" if extract_audio_tracks(str(source), query_only=True) else "",
    })
    yield settings


def prepare_latents(model, input_frames, height, width, frame_num):
    import torch

    from .minimax_h3_main import VIDEO_LATENTS_MEAN, VIDEO_LATENTS_STD, _as_video_tensor
    from .ref2va import MiniMaxH3PreparedReference
    from .two_pass import phase1_dimensions
    from .runtime import runtime
    from .viggle import _window_pixels

    source = _as_video_tensor(input_frames)
    if source is None or source.shape[1] < 1:
        raise ValueError("Regenerate 2K IC needs a readable source draft.")
    if model.latent_upscaler is None:
        raise RuntimeError("Regenerate 2K IC requires the H3 latent upscaler.")
    draft_height, draft_width = phase1_dimensions(height, width)
    rgb = _window_pixels(source, 0, frame_num, draft_height, draft_width)
    reference = MiniMaxH3PreparedReference(kind="video", frames=rgb)
    rgb_tensor = torch.from_numpy(rgb).permute(3, 0, 1, 2)
    pixels = model._condition_pixels(rgb_tensor, draft_height, draft_width)
    draft = model.vae.encode(pixels, return_dict=False)[0].mode()
    device = runtime.execution_device(model.latent_upscaler)
    dtype = torch.bfloat16
    mean = torch.tensor(VIDEO_LATENTS_MEAN, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(VIDEO_LATENTS_STD, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    draft = (draft.to(device=device, dtype=dtype) - mean) / std
    ratio = model.vae.spatial_compression_ratio
    target = (int(draft.shape[2]), height // ratio, width // ratio)
    scale = (target[1] / draft.shape[-2] + target[2] / draft.shape[-1]) / 2
    refined = model.latent_upscaler(draft, scale, target_size=target)
    return reference, refined.detach().to(device="cpu", dtype=torch.float32)
