"""Pinned Pika H3 Flex LoRAs and conservative native recipe metadata.

This catalogue describes the official Comfy-formatted BF16 adapters. It does
not enable or require FastH3, Turbo, PDD, or other acceleration paths.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import fasth3
from .pika_flex import normalize_pika_flex_attention

PIKA_FLEX_REPO = "Pika-Labs/Pika-H3-Flex"
PIKA_FLEX_REVISION = "0ddf55ad7067ca1e76812253bd7b4600df87f759"
PIKA_FLEX_MODEL_REPO = "MiniMaxAI/MiniMax-H3"
PIKA_FLEX_MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
PIKA_FLEX_BASE_REPO = "Comfy-Org/MiniMax-H3"
PIKA_FLEX_BASE_REVISION = "0543966fbdce5ba05709a8f2031c94bdba629b4a"
PIKA_FLEX_BASE_SIZE = 20_958_205_608
PIKA_FLEX_FRAMES_CHECKPOINT_FILE = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
PIKA_FLEX_FRAMES_CHECKPOINT_SHA256 = "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993"
PIKA_FLEX_REFERENCES_CHECKPOINT_FILE = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
PIKA_FLEX_REFERENCES_CHECKPOINT_SHA256 = "f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9"
PIKA_FLEX_LORA_SIZE = 1_956_172_176
PIKA_FLEX_FRAMES_FILE = "pika_h3_flex_fl2va_comfy_bf16.safetensors"
PIKA_FLEX_FRAMES_SHA256 = "68783e6447696ec365d13c0554941e070bee4cfea13c37da52498ec4f9a53f89"
PIKA_FLEX_REFERENCES_FILE = "pika_h3_flex_ref2va_comfy_bf16.safetensors"
PIKA_FLEX_REFERENCES_SHA256 = "e4d058fbd027377b925a0ee607b514ddd9037873fee3afa05d73b302e6f7f8e1"
PIKA_FLEX_KNOTS = (1000, 750, 500, 250, 100, 40, 0)
PIKA_FLEX_STEPS = 6
PIKA_FLEX_VIDEO_SHIFT = 12.0
PIKA_FLEX_AUDIO_SHIFT = 3.0
PIKA_FLEX_GUIDANCE_SCALE = 1.0
PIKA_FLEX_LORA_STRENGTH = 1.0
PIKA_FLEX_FRAME_MINIMUM = 124
PIKA_FLEX_FRAME_STEP = 17
PIKA_FLEX_FRAME_MAXIMUM = 362
PIKA_FLEX_FPS = 24
PIKA_FLEX_DEFAULT_RESOLUTION = "832x480"
PIKA_FLEX_REFERENCE_LIMITS = {"image": 9, "video": 3, "audio": 3, "total": 12}

# Each entry is an official Pika H3 Flex aspect-ratio bucket from Diffusers.
PIKA_FLEX_RESOLUTION_BUCKETS = {
    "480p": (
        "960x416", "832x480", "704x480", "640x480", "608x480",
        "480x480", "480x608", "480x640", "480x704", "480x832", "416x960",
    ),
    "768p": (
        "1536x672", "1344x768", "1152x768", "1024x768", "960x768",
        "768x768", "768x960", "768x1024", "768x1152", "768x1344", "672x1536",
    ),
}

PIKA_FLEX_CHECKPOINT_SPECS = {
    False: {
        "file": PIKA_FLEX_FRAMES_CHECKPOINT_FILE,
        "sha256": PIKA_FLEX_FRAMES_CHECKPOINT_SHA256,
        "label": "Pika H3 Flex Frames FP8 checkpoint",
    },
    True: {
        "file": PIKA_FLEX_REFERENCES_CHECKPOINT_FILE,
        "sha256": PIKA_FLEX_REFERENCES_CHECKPOINT_SHA256,
        "label": "Pika H3 Flex References FP8 checkpoint",
    },
}
PIKA_FLEX_LORA_SPECS = {
    False: {
        "file": PIKA_FLEX_FRAMES_FILE,
        "sha256": PIKA_FLEX_FRAMES_SHA256,
        "label": "Pika H3 Flex Frames LoRA",
        "url": (
            f"https://huggingface.co/{PIKA_FLEX_REPO}/resolve/{PIKA_FLEX_REVISION}/"
            f"comfy/loras/{PIKA_FLEX_FRAMES_FILE}"
        ),
    },
    True: {
        "file": PIKA_FLEX_REFERENCES_FILE,
        "sha256": PIKA_FLEX_REFERENCES_SHA256,
        "label": "Pika H3 Flex References LoRA",
        "url": (
            f"https://huggingface.co/{PIKA_FLEX_REPO}/resolve/{PIKA_FLEX_REVISION}/"
            f"comfy/loras/{PIKA_FLEX_REFERENCES_FILE}"
        ),
    },
}

PIKA_FLEX_HELP = (
    "PIKA H3 FLEX — EXPERIMENTAL\n"
    "Six fixed model evaluations with the pinned Pika H3 Flex adapter, CFG 1, "
    "adapter strength 1, and video/audio shifts 12/3. Frames supports text-to-video "
    "and first/last images. References supports ordered image, video, and audio "
    "references and needs at least one image or video. Begin at 480p / 5.17 seconds. "
    "Sage2 is the dense default; SOL and SLA are optional experimental attention. "
    "Continuation, control video, extra adapters, cache, and refinement are unavailable. "
    "Maestro has no Pika 4K upsampler or stem extraction, and consumer-GPU real-time "
    "performance has not been verified. The initial experimental stack uses the stock Comfy scaled-FP8 base, BF16 Pika adapter, and INT8 ConvRot video VAE."
)


def _check_reference_flag(reference: bool) -> bool:
    if not isinstance(reference, bool):
        raise TypeError("Pika H3 Flex reference must be a boolean.")
    return reference


_PIKA_FLEX_ASPECT_KEYS = {
    "480p": ("21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"),
    "768p": ("21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"),
}


def _pika_resolution_presets() -> dict[str, dict[str, Any]]:
    return {
        tier: {
            "label": tier,
            "values": dict(zip(_PIKA_FLEX_ASPECT_KEYS[tier], resolutions)),
        }
        for tier, resolutions in PIKA_FLEX_RESOLUTION_BUCKETS.items()
    }


def pika_flex_capabilities(reference: bool) -> dict[str, Any]:
    """Return a narrowed capability profile for Frames or References."""

    reference = _check_reference_flag(reference)
    return {
        "minimax_h3_pika_flex": True,
        "minimax_h3_sampler": "euler",
        "minimax_h3_video_shift": PIKA_FLEX_VIDEO_SHIFT,
        "minimax_h3_audio_shift": PIKA_FLEX_AUDIO_SHIFT,
        "minimax_h3_unaccelerated_default_steps": PIKA_FLEX_STEPS,
        "minimax_h3_pika_flex_knots": PIKA_FLEX_KNOTS,
        "minimax_h3_pika_flex_lora_strength": PIKA_FLEX_LORA_STRENGTH,
        "fps": PIKA_FLEX_FPS,
        "dtype": "bf16",
        "frames_minimum": PIKA_FLEX_FRAME_MINIMUM,
        "frames_steps": PIKA_FLEX_FRAME_STEP,
        "frames_maximum": PIKA_FLEX_FRAME_MAXIMUM,
        "frame_alignment_modulus": PIKA_FLEX_FRAME_STEP,
        "frame_alignment_remainder": 5,
        "frame_alignment_mode": "ceil",
        "t2v_class": True,
        "i2v_class": not reference,
        "image_prompt_types_allowed": "" if reference else "TSE",
        "end_frames_always_enabled": not reference,
        "custom_frames_injection": False,
        "omni_reference": reference,
        "omni_reference_limits": dict(PIKA_FLEX_REFERENCE_LIMITS) if reference else None,
        "supports_reference_audio": reference,
        "any_audio_prompt": False,
        "audio_prompt_choices": False,
        "audio_prompt_type_sources": {},
        "infer_audio_prompt_from_guide": False,
        "returns_audio": True,
        "sliding_window": False,
        "video_continuation": False,
        "minimax_h3_multi_window": False,
        "minimax_h3_reference_sequence": False,
        "sliding_window_end_image_at_final": False,
        "sliding_window_auto_prompt_pacing": False,
        "sliding_window_memory_policy": None,
        "sliding_window_audio_history": False,
        "audio_guide_window_slicing": False,
        "video_to_video_inpaint": False,
        "guide_custom_choices": {},
        "video_guide_outpainting": [],
        "mask_preprocessing": {},
        "director_video_strategy": "",
        "director_reference_mode": "none",
        "director_memory_policy": None,
        "omni_sequence_memory_policy": None,
        "director_audio_input_mode": "none",
        "director_shot_image_support": "none",
        "director_endpoint_continuity": False,
        "custom_settings": [],
        "runtime_custom_settings": [],
        "sol_attention": True,
        "sla_attention": True,
        "sla_attention_default": False,
        "sla_attention_experimental": True,
        "veda_attention": False,
        "veda_attention_experimental": False,
        "first_block_cache": False,
        "loras_disabled": True,
        "profiles_dir": [],
        "compatible_model_paths": {},
        "compatible_model_qkv_layouts": {},
        "lock_inference_steps": True,
        "inference_steps_min": PIKA_FLEX_STEPS,
        "inference_steps_max": PIKA_FLEX_STEPS,
        "inference_steps_label": "Denoising Steps",
        "inference_steps_help": "Pika H3 Flex uses six fixed model evaluations and does not support step changes.",
        "lock_guidance_scale": True,
        "guidance_max_phases": 1,
        "lock_guidance_phases": True,
        "resolution_presets": _pika_resolution_presets(),
        "resolution_preset_order": ["480p", "768p"],
        "resolutions": [
            (f"{tier} {resolution}", resolution)
            for tier, resolutions in PIKA_FLEX_RESOLUTION_BUCKETS.items()
            for resolution in resolutions
        ],
        "supports_auto_aspect": False,
        "selector_help": PIKA_FLEX_HELP,
    }


def pika_flex_defaults(reference: bool, *, attention: Any = None) -> dict[str, Any]:
    """Return the fixed, conservative first-run generation settings."""

    reference = _check_reference_flag(reference)
    defaults: dict[str, Any] = {
        "num_inference_steps": PIKA_FLEX_STEPS,
        "guidance_phases": 1,
        "guidance_scale": PIKA_FLEX_GUIDANCE_SCALE,
        "flow_shift": PIKA_FLEX_VIDEO_SHIFT,
        "audio_flow_shift": PIKA_FLEX_AUDIO_SHIFT,
        "sample_solver": "euler",
        "denoising_strength": 1.0,

        "override_attention": normalize_pika_flex_attention(attention),
        "skip_steps_cache_type": "",
        "minimax_h3_turbo_mode": False,
        "minimax_h3_turbo_preset": "",
        "minimax_h3_extended_duration": False,
        "minimax_h3_multi_window": False,
        "minimax_h3_reference_sequence": False,
        "multi_prompts_gen_type": 2,
    }
    return defaults



def verify_pika_flex_checkpoint(
    filename: str | os.PathLike[str],
    *,
    reference: bool,
) -> None:
    """Verify the single pinned Comfy FP8 base before model allocation."""

    reference = _check_reference_flag(reference)
    try:
        filename_text = os.fspath(filename)
    except TypeError as error:
        raise TypeError("Pika H3 Flex checkpoint filename must be path-like.") from error
    if not isinstance(filename_text, str) or not filename_text:
        raise TypeError("Pika H3 Flex checkpoint filename must be a non-empty text path.")

    pin = PIKA_FLEX_CHECKPOINT_SPECS[reference]
    if Path(filename_text).name != pin["file"]:
        raise ValueError(
            f"Pika H3 Flex requires only its pinned {pin['label']}; "
            "this model has no qualified checkpoint alias."
        )

    fasth3.verify_checkpoint_pin(
        filename_text,
        PIKA_FLEX_BASE_SIZE,
        pin["sha256"],
        pin["label"],
    )
def verify_pika_flex_loras(
    paths: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
    *,
    reference: bool,
) -> None:
    """Verify the single required adapter and reject missing or extra files.

    The candidate count and basename are checked before opening the large file.
    The existing cached checkpoint-pin helper then verifies exact size and
    SHA-256. No FastH3 model flag or runtime is needed for this check.
    """

    reference = _check_reference_flag(reference)
    if isinstance(paths, (str, os.PathLike)):
        candidates = [paths]
    else:
        try:
            candidates = list(paths)
        except TypeError as error:
            raise TypeError("Pika H3 Flex adapters must be supplied as paths.") from error

    if len(candidates) != 1:
        raise ValueError("Pika H3 Flex requires exactly its one pinned adapter; extra adapters are unsupported.")

    candidate = candidates[0]
    try:
        candidate_text = os.fspath(candidate)
    except TypeError as error:
        raise TypeError("Pika H3 Flex adapter path must be path-like.") from error
    if not isinstance(candidate_text, str) or not candidate_text:
        raise TypeError("Pika H3 Flex adapter path must be a non-empty text path.")

    pin = PIKA_FLEX_LORA_SPECS[reference]
    if Path(candidate_text).name != pin["file"]:
        raise ValueError(
            f"Pika H3 Flex requires only its pinned {pin['label']}; "
            "additional or mismatched adapters are unsupported."
        )

    fasth3.verify_checkpoint_pin(
        candidate_text,
        PIKA_FLEX_LORA_SIZE,
        pin["sha256"],
        pin["label"],
    )


__all__ = [
    "PIKA_FLEX_AUDIO_SHIFT",
    "PIKA_FLEX_BASE_REPO",
    "PIKA_FLEX_BASE_REVISION",
    "PIKA_FLEX_BASE_SIZE",
    "PIKA_FLEX_CHECKPOINT_SPECS",
    "PIKA_FLEX_FRAMES_CHECKPOINT_FILE",
    "PIKA_FLEX_FRAMES_CHECKPOINT_SHA256",
    "PIKA_FLEX_REFERENCES_CHECKPOINT_FILE",
    "PIKA_FLEX_REFERENCES_CHECKPOINT_SHA256",
    "PIKA_FLEX_DEFAULT_RESOLUTION",
    "PIKA_FLEX_FPS",
    "PIKA_FLEX_FRAME_MAXIMUM",
    "PIKA_FLEX_FRAME_MINIMUM",
    "PIKA_FLEX_FRAME_STEP",
    "PIKA_FLEX_FRAMES_FILE",
    "PIKA_FLEX_FRAMES_SHA256",
    "PIKA_FLEX_GUIDANCE_SCALE",
    "PIKA_FLEX_HELP",
    "PIKA_FLEX_KNOTS",
    "PIKA_FLEX_LORA_SIZE",
    "PIKA_FLEX_LORA_SPECS",
    "PIKA_FLEX_LORA_STRENGTH",
    "PIKA_FLEX_MODEL_REPO",
    "PIKA_FLEX_MODEL_REVISION",
    "PIKA_FLEX_REFERENCE_LIMITS",
    "PIKA_FLEX_REFERENCES_FILE",
    "PIKA_FLEX_REFERENCES_SHA256",
    "PIKA_FLEX_RESOLUTION_BUCKETS",
    "PIKA_FLEX_REPO",
    "PIKA_FLEX_REVISION",
    "PIKA_FLEX_STEPS",
    "PIKA_FLEX_VIDEO_SHIFT",
    "pika_flex_capabilities",
    "pika_flex_defaults",
    "verify_pika_flex_checkpoint",
    "verify_pika_flex_loras",
]
