"""Viggle Animate's fixed conditioning, adapted from pinned Wan2GP v12.72.

See UPSTREAM.md for source revision and licensing. The video reference MUST
precede the edited picture; the released text embedding encodes that order.
"""
from __future__ import annotations

import math
import os

ARCHITECTURE = "viggle_animate"
REVISION = "fa7ed035f21d341439d4dd763a020fc4a2482c43"
REPO = "DeepBeepMeep/MiniMax-H3"
PROMPT_FILE = "viggle_animate_fixed_prompt.safetensors"
PROMPT_PATH = os.path.join("viggle_animate", PROMPT_FILE)
WINDOW_FRAMES = 124
OVERLAP_FRAMES = 18
STEPS = 3
PROMPT = "Viggle Animate: propagate the edited frame through the control video."


def model_definition(base):
    result = dict(base)
    # Reuse the H3 media runtime, but keep Viggle outside ordinary Omni,
    # Director and text enhancement routing in Maestro's public capabilities.
    result.update({
        "minimax_h3_viggle": True, "omni_reference": False,
        "director_video_strategy": "", "director_reference_mode": "none",
        "director_audio_input_mode": "none", "director_shot_image_support": "none",
        "text_encoder_URLs": [], "text_encoder_folder": None,
        "minimax_h3_text_encoder_variants": {}, "minimax_h3_text_encoder_default": None,
        "compatible_text_encoder_paths": {}, "compatible_model_paths": {},
        "compatible_model_qkv_layouts": {}, "minimax_h3_qkv_layout": "interleaved",
        "minimax_h3_sampler": "euler", "frames_maximum": WINDOW_FRAMES,
        "frames_selection_maximum": WINDOW_FRAMES, "sliding_window_size_locked": True,
        "sliding_window_defaults": {**base["sliding_window_defaults"],
            "window_min": WINDOW_FRAMES, "window_max": WINDOW_FRAMES,
            "window_default": WINDOW_FRAMES, "overlap_default": OVERLAP_FRAMES},
        "sliding_window_memory_policy": None, "omni_sequence_memory_policy": None,
        "sliding_window_auto_prompt_pacing": False, "video_continuation": False,
        "sliding_window_end_image_at_final": False, "custom_frames_injection": False,
        "extract_guide_from_window_start": True, "control_video_trim_disabled": False,
        "control_video_trim": False, "image_prompt_types_allowed": "T",
        "image_outputs": False, "end_frames_always_enabled": False,
        "i2v_class": False, "t2v_class": False, "supports_reference_audio": False,
        "guidance_max_phases": 0, "visible_phases": 0,
        "lock_inference_steps": True, "inference_steps_min": STEPS, "inference_steps_max": STEPS,
        "lock_guidance_scale": True, "loras_disabled": True,
        "sol_attention": False, "sla_attention": False, "first_block_cache": False,
        "custom_settings": [], "runtime_custom_settings": [],
        "one_image_ref_needed": True, "no_background_removal": True,
        "any_image_refs_relative_size": False, "fit_into_canvas_image_refs": 1,
        "image_ref_choices": {"choices": [("Edited frame", "I")], "letters_filter": "I", "default": "I"},
        "guide_custom_choices": {"choices": [("Control video", "VU")], "letters_filter": "V-U", "default": "VU"},
        "any_audio_prompt": True, "audio_prompt_choices": True, "output_audio_is_input_audio": True,
        "audio_prompt_type_sources": {"selection": ["", "K", "A"], "letters_filter": "AK",
            "labels": {"": "No input audio", "K": "Use control video audio", "A": "Use custom audio"}, "default": ""},
        "selector_help": "Animate one edited frame through a control video. Fixed prompt, three Euler evaluations, 5.2-second windows. Keep the edited frame's pose, framing and background aligned with the source.",
        "profiles_dir": [ARCHITECTURE],
    })
    return result


def normalize_settings(inputs, *, validate_media=True, allow_preparation=False):
    """Enforce the distilled recipe for UI, restored jobs, and API callers."""
    refs = inputs.get("image_refs") or []
    prepare_character = allow_preparation and bool(inputs.get("viggle_character"))
    if prepare_character:
        from services.viggle_preparation import normalize_options
        inputs["viggle_character"] = normalize_options(inputs["viggle_character"])
    if validate_media:
        if not inputs.get("video_guide"):
            raise ValueError("Viggle Animate needs a control video.")
        if not prepare_character and (not isinstance(refs, (list, tuple)) or len(refs) != 1):
            raise ValueError("Viggle Animate needs exactly one edited frame from the control video.")
    frames = float(inputs.get("video_length") or WINDOW_FRAMES)
    if not math.isfinite(frames) or not 1 <= frames <= 24 * 3600:
        raise ValueError("Viggle duration must be between one frame and one hour.")
    from services.viggle_media import selected_range
    selection = selected_range(inputs)
    if selection is not None:
        start, end = selection
        inputs.update(_viggle_trim_start=start, _viggle_trim_end=end)
        frames = min(frames, max(1, math.floor((end - start) * 24 + 1e-6)))
    # New recipes must not pass through the legacy pre-2.35 audio migrations.
    inputs["settings_version"] = max(float(inputs.get("settings_version") or 0), 2.58)
    if inputs.get("resolution") == "auto480p":
        inputs["resolution"] = "auto_480p"
    inputs.update({"prompt": PROMPT, "num_inference_steps": STEPS, "flow_shift": 3.0,
        "audio_flow_shift": 3.0, "sample_solver": "euler", "guidance_scale": 1.0,
        "guidance_phases": 1, "sliding_window_size": WINDOW_FRAMES,
        "sliding_window_overlap": OVERLAP_FRAMES, "sliding_window_discard_last_frames": 0,
        "video_prompt_type": "IVU", "image_prompt_type": "", "image_mode": 0,
        "multi_prompts_gen_type": 2, "image_refs_relative_size": 100,
        "remove_background_images_ref": 0, "skip_steps_cache_type": "",
        "override_attention": "",
        "minimax_h3_turbo_mode": False, "activated_loras": [], "loras_multipliers": "",
        "force_fps": "24", "video_length": int(round(frames)),
        "denoising_strength": 1.0, "masking_strength": 1.0,
        "_studio_video_workflow": "animate"})
    for key in ("image_start", "image_end", "video_source", "frames_positions", "video_mask",
                "minimax_h3_references", "minimax_h3_reference_sequence", "minimax_h3_multi_window"):
        if key in inputs:
            inputs[key] = None
    inputs["custom_settings"] = {}
    audio = str(inputs.get("audio_prompt_type") or "")
    if not validate_media:
        audio = "".join(letter for letter in audio if letter not in "VNL")
    if audio not in ("", "K", "A"):
        raise ValueError("Choose no input audio, the video's soundtrack, or custom audio for Viggle.")
    if validate_media and audio == "A" and not inputs.get("audio_guide"):
        raise ValueError("Upload custom audio or choose another Viggle audio mode.")
    inputs["audio_prompt_type"] = audio


def load_conditioner():
    import torch
    from safetensors.torch import load_file
    from shared.utils import files_locator as fl
    tensors = load_file(fl.locate_file(PROMPT_PATH), device="cpu")
    embeds, tags = tensors["prompt_embeds"], tensors["text_token_tags"]
    if tuple(embeds.shape) != (1, 362, 5120) or tuple(tags.shape) != (362,):
        raise ValueError("Viggle's fixed prompt has an unexpected shape; re-download its conditioning asset.")

    class FixedConditioner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("prompt_embeds", embeds)
            self.register_buffer("text_tags", tags)
            self._interrupt = False

        def forward_ref2va(self, _prompt, device, _references):
            return (self.prompt_embeds.to(device=device, dtype=torch.bfloat16),
                    self.text_tags.to(device=device))

    return FixedConditioner()


def prepare_window_references(input_frames, edited_images, history_count, frame_num, height, width):
    import numpy as np
    import torch
    from .minimax_h3_main import _prepare_control_video_tensor, _tensor_to_pil, prepare_keyframe_image
    from .ref2va import MiniMaxH3PreparedReference
    video = _prepare_control_video_tensor(input_frames, height, width)
    if video is None or not edited_images or len(edited_images) != 1:
        raise ValueError("Viggle requires a control video window and one edited frame.")
    # WanGP extracts from window_start, including the motion-history prefix.
    # The reference covers only this pass's target, in video-then-image order.
    if video.shape[1] < frame_num:
        video = torch.cat([video, video[:, -1:].expand(-1, frame_num - video.shape[1], -1, -1)], dim=1)
    video = video[:, history_count:frame_num]
    if video.shape[1] < 5 or (video.shape[1] - 5) % 17:
        raise ValueError("Viggle's control window must follow the H3 17n+5 frame alignment.")
    pixels = video.add(1).mul(127.5).round().clamp(0, 255).permute(1, 2, 3, 0).numpy().astype(np.uint8)
    image = _tensor_to_pil(edited_images[0])
    if image is None:
        raise ValueError("The Viggle edited frame is not a readable image.")
    image = prepare_keyframe_image(image, height, width, stretch=True)
    return [MiniMaxH3PreparedReference(kind="video", frames=pixels),
            MiniMaxH3PreparedReference(kind="image", image=image)]
