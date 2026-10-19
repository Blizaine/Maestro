"""Shared MiniMax H3 video VAE asset names."""

from __future__ import annotations

import os


# Full and INT8 files remain the established Maestro defaults. Keeping the
# variant list next to the LightVAE basenames gives runtime selection and the
# UI one source of truth without changing existing resolver paths.
VIDEO_VAE_FULL_FILE = "minimax_h3_video_vae_fp16.safetensors"
VIDEO_VAE_FULL_INT8_FILE = "minimax_h3_video_vae_int8_convrot.safetensors"

# WanGP v18.10 LightVAE Lite checkpoints. Maestro keeps these basenames under
# its H3 VAE asset root and resolves them through files_locator.
VIDEO_VAE_LIGHT_FILE = "minimax_h3_lightvae_lite_bf16.safetensors"
VIDEO_VAE_LIGHT_INT8_FILE = "minimax_h3_lightvae_lite_int8_convrot.safetensors"
VIDEO_VAE_LIGHT_REVISION = "630975da14fd3c84169d6cd5092da3ce1287d137"
LIGHT_VIDEO_VAE_FILES = frozenset(
    (VIDEO_VAE_LIGHT_FILE, VIDEO_VAE_LIGHT_INT8_FILE)
)

VIDEO_VAE_VARIANT_FILENAMES = {
    "full": VIDEO_VAE_FULL_FILE,
    "full_int8": VIDEO_VAE_FULL_INT8_FILE,
    "lightvae": VIDEO_VAE_LIGHT_FILE,
    "lightvae_int8": VIDEO_VAE_LIGHT_INT8_FILE,
}
VIDEO_VAE_VARIANT_CHOICES = (
    {"value": "auto", "label": "Automatic"},
    {"value": "full", "label": "Full (FP16)"},
    {"value": "full_int8", "label": "Full (INT8 ConvRot)"},
    {"value": "lightvae", "label": "LightVAE Lite (BF16)"},
    {"value": "lightvae_int8", "label": "LightVAE Lite (INT8 ConvRot)"},
)


def get_video_vae_variant_choices() -> list[dict[str, str]]:
    """Return UI-ready MiniMax H3 video VAE choices."""

    return [dict(choice) for choice in VIDEO_VAE_VARIANT_CHOICES]


def is_light_video_vae_checkpoint(filename: str | os.PathLike[str]) -> bool:
    """Recognize a LightVAE asset by basename, including resolved paths."""

    return os.path.basename(os.fspath(filename)) in LIGHT_VIDEO_VAE_FILES


__all__ = [
    "LIGHT_VIDEO_VAE_FILES",
    "VIDEO_VAE_FULL_FILE",
    "VIDEO_VAE_FULL_INT8_FILE",
    "VIDEO_VAE_LIGHT_FILE",
    "VIDEO_VAE_LIGHT_INT8_FILE",
    "VIDEO_VAE_LIGHT_REVISION",
    "VIDEO_VAE_VARIANT_CHOICES",
    "VIDEO_VAE_VARIANT_FILENAMES",
    "get_video_vae_variant_choices",
    "is_light_video_vae_checkpoint",
]
