"""Compatibility policy for DaSiWa's pinned MiniMax H3 Hybrid v3 exports."""

from __future__ import annotations

from .imported import normalize_baked_h3_attention


def validate_dasiwa_checkpoint(checkpoint: dict, qkv_layout: str) -> None:
    """Reject incompatible formats before allocating the pruned transformer."""

    formats = {part.strip().lower() for part in str(
        checkpoint.get("quantization_format") or ""
    ).split(",")}
    if (
        qkv_layout != "grouped"
        or checkpoint.get("convrot") is not True
        or "int8_tensorwise" not in formats
        or checkpoint.get("convrot_group_size") != 256
        or checkpoint.get("compressed_modulation") is not True
        or checkpoint.get("adaln_curve_grid") != 1025
        or checkpoint.get("time_embed_dim") != 8
    ):
        raise ValueError(
            "DaSiWa H3 Hybrid v3 requires the selected pruned INT8 ConvRot "
            "checkpoint with grouped QKV and [1025, 8] AdaLN curves. "
            "INT4 and stock H3 files are not interchangeable."
        )


def normalize_dasiwa_turbo_steps(value) -> int:
    """Return the creator's supported four-to-eight evaluation count."""

    if value in (None, ""):
        return 8
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float("nan")
    if isinstance(value, bool) or not numeric.is_integer() or not 4 <= numeric <= 8:
        raise ValueError("DaSiWa H3 Hybrid Turbo v3 supports 4–8 whole denoising steps.")
    return int(numeric)


def validate_dasiwa_turbo_loras(paths) -> None:
    """Keep ordinary adapters while rejecting a second distillation recipe."""

    from .fused_turbo import fused_h3_lora_incompatibility

    for path in paths or ():
        reason = fused_h3_lora_incompatibility(str(path))
        if reason:
            name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
            raise ValueError(f"DaSiWa H3 Hybrid Turbo cannot use {name}: {reason}.")


def normalize_dasiwa_turbo_request(body: dict, *, resolve_lora=None) -> None:
    """Enforce baked Turbo without adding another accelerator or cache."""

    selected = body.get("activated_loras") or ()
    validate_dasiwa_turbo_loras(selected)
    if resolve_lora is not None:
        validate_dasiwa_turbo_loras([resolve_lora(path) for path in selected])
    body["num_inference_steps"] = normalize_dasiwa_turbo_steps(body.get("num_inference_steps"))
    body["minimax_h3_turbo_mode"] = False
    body["minimax_h3_turbo_preset"] = ""
    body["guidance_scale"] = 1.0
    body["flow_shift"] = 9.0
    body["audio_flow_shift"] = 4.0
    body["skip_steps_cache_type"] = ""
    body["override_attention"] = normalize_baked_h3_attention(body.get("override_attention"))
    if (body.get("custom_settings") or {}).get("audio_refinement") == "enabled":
        raise ValueError("Audio refinement is unavailable with baked DaSiWa H3 Turbo.")
