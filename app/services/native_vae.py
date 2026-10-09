"""Lightweight validation for optional generation-time VAE decoders."""
from __future__ import annotations

H3_VAE_X2 = "h3_vae*2"


def native_vae_selection(model_def: dict, spatial: str, image_mode: int) -> str | None:
    """Return an enabled native decoder; reject unsupported saved/API choices.

    These decoders consume H3 latents during generation. They cannot be used
    as pixel upscalers for existing gallery media.
    """
    spatial = str(spatial or "")
    if not spatial.startswith("h3_vae"):
        return None
    if spatial != H3_VAE_X2:
        raise ValueError("H3 VAE upsampling currently supports 2Ã— only. Choose H3 VAE 2Ã— or None.")
    modes = (model_def.get("vae_upsamplers") or {}).get("h3_vae", [])
    if model_def.get("audio_only") or image_mode not in modes:
        raise ValueError("H3 VAE 2Ã— is available only while generating with a supported H3 model and generation mode.")
    return H3_VAE_X2
