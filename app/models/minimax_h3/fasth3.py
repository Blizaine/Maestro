"""Pinned FastH3 V2 T2AV recipe, distinct from H3 Turbo and OmniRef PDD.

Contract: FastVideo/FastVideo-FastH3-8-Step-V2@3da2ddfe1954d9cda4c05b643dc0f26007a655c5.
The eight training rungs are shifted once per modality, then terminated by
zero. They are not an ordinary eight-step uniform or RES schedule.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

FASTH3_REPO = "FastVideo/FastVideo-FastH3-Comfy"
FASTH3_REVISION = "ec1e3aa374a91c57b0b94a1623b7e657c0498cf2"
FASTH3_FILE = "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
FASTH3_SIZE = 22128378696
FASTH3_SHA256 = "0922785978dc9bfe1adf27d8b291b0ca763f9f165f882e6cb297c72fbb6deda8"
FASTH3_RUNGS = (999, 874, 749, 624, 500, 375, 250, 125)
FASTH3_HELP = (
    "FASTH3 V2 — EXPERIMENTAL\n"
    "Eight-step text-to-video with generated audio. Uses its trained VSA-H3 "
    "attention (80% sparsity, 4×4×4 tiles) and fixed Euler schedule with video/audio "
    "shifts 10/3. Requires CUDA/Triton and BF16 GPU support. This checkpoint was not distilled for "
    "start/end frames, Omni references, source audio or continuation. Those inputs "
    "and additional LoRAs are unavailable. Begin with 480p / 5.2 seconds. "
    "Real-time generation on consumer GPUs has not been verified in Maestro."
)
_VERIFIED: set[tuple[str, int, int, int, str]] = set()


def fasth3_sigmas(shift: float) -> tuple[float, ...]:
    if shift not in (10.0, 3.0):
        raise ValueError("FastH3 V2 requires video/audio shifts 10/3.")
    return tuple(shift * (r / 1000.0) / (1 + (shift - 1) * (r / 1000.0))
                 for r in FASTH3_RUNGS) + (0.0,)


def verify_fasth3_checkpoint(filename: str) -> None:
    """Fail before MMGP allocation if the downloaded weights are not the pin."""
    verify_checkpoint_pin(filename, FASTH3_SIZE, FASTH3_SHA256, "FastH3 V2")


def verify_checkpoint_pin(filename: str, expected_size: int, expected_sha256: str, label: str) -> None:
    path = Path(filename).resolve(strict=True)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected_sha256)
    if key in _VERIFIED:
        return
    if stat.st_size != expected_size:
        raise ValueError(f"{label} requires its pinned INT8 ConvRot checkpoint; the file size differs.")
    print("[FastH3] Verifying pinned checkpoint SHA-256…", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"{label} checkpoint checksum differs from the published pin. Re-download the model.")
    current = path.stat()
    if (current.st_size, current.st_mtime_ns, current.st_ctime_ns) != key[1:4]:
        raise ValueError("FastH3 checkpoint changed during verification.")
    _VERIFIED.add(key)


def validate_fasth3_request(inputs: dict) -> None:
    """Reject unsupported conditioning rather than silently dropping it."""
    media = (
        "image_start", "image_end", "image_refs", "input_frames", "input_ref_images",
        "input_masks", "input_video", "input_waveform", "minimax_h3_references",
        "video_guide", "video_source", "audio_guide", "audio_guide2", "frames_to_inject",
    )
    for key in media:
        value = inputs.get(key)
        if value is not None and not (isinstance(value, (list, tuple, dict, str)) and len(value) == 0):
            raise ValueError("FastH3 V2 supports text-to-video with generated audio only; remove reference images, frames, video and source audio.")
    if any(str(inputs.get(key) or "") for key in ("image_prompt_type", "video_prompt_type", "audio_prompt_type")):
        raise ValueError("FastH3 V2 supports text-to-video only; switch off media conditioning.")
    if inputs.get("prefix_frames_count") or inputs.get("window_start_frame_no") or inputs.get("minimax_h3_reference_sequence") or inputs.get("minimax_h3_multi_window"):
        raise ValueError("FastH3 V2 does not support continuation windows.")
    if inputs.get("activated_loras") or inputs.get("loras_selected") or inputs.get("minimax_h3_turbo_mode"):
        raise ValueError("FastH3 V2 uses its distilled recipe; additional LoRAs/Turbo are unavailable.")
    if inputs.get("multi_prompts_gen_type") in (3, "3"):
        raise ValueError("FastH3 V2 currently accepts one complete prompt per generation. Queue separate prompts to batch clips.")
    steps = inputs.get("num_inference_steps", inputs.get("sampling_steps", 8))
    if isinstance(steps, bool) or steps is None or float(steps) != 8:
        raise ValueError("FastH3 V2 requires exactly eight model evaluations.")
    if int(inputs.get("video_length", inputs.get("frame_num", 124))) > 345:
        raise ValueError("FastH3 V2 supports one native window of at most 14.4 seconds.")
    if float(inputs.get("denoising_strength", 1.0)) != 1.0:
        raise ValueError("FastH3 V2 requires the complete eight-step schedule.")
    custom = inputs.get("custom_settings") or {}
    if custom.get("audio_refinement") == "enabled":
        raise ValueError("FastH3 V2 does not support the ordinary H3 audio-refinement pass.")
    inputs.update(num_inference_steps=8, guidance_scale=1.0, flow_shift=10.0,
                  audio_flow_shift=3.0, override_attention="sdpa",
                  skip_steps_cache_type="", minimax_h3_turbo_mode=False,
                  minimax_h3_turbo_preset="", minimax_h3_extended_duration=False,
                  multi_prompts_gen_type=2,
                  sliding_window_size=int(inputs.get("video_length", inputs.get("frame_num", 124))),
                  sliding_window_overlap=0, sliding_window_discard_last_frames=0)


def fasth3_capabilities() -> dict:
    """Remove capabilities inherited from the ordinary FL2VA handler."""
    return {
        "minimax_h3_fasth3": True, "minimax_h3_qkv_layout": "grouped",
        "minimax_h3_sampler": "euler", "minimax_h3_video_shift": 10.0,
        "minimax_h3_audio_shift": 3.0, "i2v_class": False,
        "omni_reference": False, "image_prompt_types_allowed": "",
        "end_frames_always_enabled": False, "custom_frames_injection": False,
        "sliding_window": False, "video_continuation": False,
        "sliding_window_end_image_at_final": False,
        "sliding_window_auto_prompt_pacing": False, "sliding_window_memory_policy": None,
        "sliding_window_audio_history": False, "any_audio_prompt": False,
        "audio_prompt_choices": False, "supports_reference_audio": False,
        "audio_prompt_type_sources": {}, "infer_audio_prompt_from_guide": False,
        "video_to_video_inpaint": False, "guide_custom_choices": {},
        "video_guide_outpainting": [], "mask_preprocessing": {},
        "director_video_strategy": "", "director_reference_mode": "none",
        "director_memory_policy": None, "omni_sequence_memory_policy": None,
        "director_audio_input_mode": "none", "director_shot_image_support": "none",
        "director_endpoint_continuity": False, "custom_settings": [],
        "runtime_custom_settings": [], "audio_guide_window_slicing": False,
        "sol_attention": False, "sla_attention": False, "sla_attention_default": False,
        "sla_attention_experimental": False, "veda_attention": False,
        "veda_attention_experimental": False, "first_block_cache": False,
        "loras_disabled": True, "profiles_dir": [], "compatible_model_paths": {},
        "compatible_model_qkv_layouts": {}, "lock_inference_steps": True,
        "inference_steps_min": 8, "inference_steps_max": 8,
        "inference_steps_label": "Denoising Steps",
        "inference_steps_help": "FastH3 V2 uses eight fixed DMD steps with trained VSA-H3 attention.",
        "lock_guidance_scale": True, "minimax_h3_turbo_mode_default": False,
        "minimax_h3_unaccelerated_default_steps": 8,
        "selector_help": FASTH3_HELP, "frames_maximum": 345,
    }
