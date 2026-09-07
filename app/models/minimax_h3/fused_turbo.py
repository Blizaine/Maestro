"""Policy helpers for the baked MATLOWAI MiniMax H3 Turbo checkpoint."""

from __future__ import annotations

import re

from .turbo import is_minimax_h3_turbo_lora, safetensors_header


FUSED_H3_DEFAULT_EVALUATIONS = 4
FUSED_H3_MIN_EVALUATIONS = 4
FUSED_H3_MAX_EVALUATIONS = 8
# Compatibility alias for callers/tests written while the recipe was fixed.
FUSED_H3_EVALUATIONS = FUSED_H3_DEFAULT_EVALUATIONS
FUSED_H3_SOLVER = "res_multistep"


def normalize_fused_h3_steps(value) -> int:
    """Return a supported experimental evaluation count.

    The publisher recommends four evaluations and also documents successful
    six- and eight-evaluation single-pass runs. Keep Maestro's control inside
    that measured ladder rather than exposing the generic 1-50 step range.
    """

    if value in (None, ""):
        return FUSED_H3_DEFAULT_EVALUATIONS
    if isinstance(value, bool):
        raise ValueError("H3 Fused Turbo total steps must be a whole number.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "H3 Fused Turbo total steps must be a whole number."
        ) from error
    if not numeric.is_integer():
        raise ValueError("H3 Fused Turbo total steps must be a whole number.")
    steps = int(numeric)
    if not FUSED_H3_MIN_EVALUATIONS <= steps <= FUSED_H3_MAX_EVALUATIONS:
        raise ValueError(
            "H3 Fused Turbo supports 4-8 total denoising steps; "
            f"received {steps}. Four is the published default."
        )
    return steps


def fused_h3_lora_incompatibility(path: str) -> str | None:
    """Identify adapters that cannot be stacked on the fused checkpoint.

    Header inspection also catches renamed PDD/VDN and DoRA files. Ordinary
    H3 LoRAs continue through the existing format, AdaLN and shape validation
    in the loader; this is not a guarantee of four-step visual quality.
    """

    path = str(path)
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    tokens = set(re.split(r"[^a-z0-9]+", name))
    if is_minimax_h3_turbo_lora(path) or tokens.intersection({"turbo", "pdd", "acc", "acceleration", "accelerator"}):
        return "Turbo is already baked in; additional acceleration recipes cannot be stacked"
    header = safetensors_header(path)
    keys = tuple(str(key).lower() for key in header if key != "__metadata__")
    if "vdn" in tokens or any(
        marker in key
        for key in keys
        for marker in (".attn.vdn.", ".attn.linear_attention.", ".attn.softmax_gate.", ".attn.to_out_linear.")
    ):
        return "VDN adapters require the dedicated H3 VDN model"
    if any("lora_magnitude_vector" in key or "dora_scale" in key for key in keys):
        return "DoRA is unsupported on INT8 ConvRot; use a standard H3 LoRA"
    if all(
        isinstance(header.get(key), dict)
        and len(header[key].get("shape", [])) == 3
        and header[key]["shape"][0] == 32
        for key in ("proj_out.weight", "audio_proj_out.weight")
    ):
        return "PDD interval heads cannot be stacked on the fused Turbo recipe"
    return None


def validate_fused_h3_loras(paths) -> None:
    """Reject conflicting adapters, preserving the user's selection on error."""

    for path in paths or []:
        reason = fused_h3_lora_incompatibility(path)
        if reason is None:
            continue
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        raise ValueError(
            f"H3 Fused 4-Step cannot use {name}: {reason}. "
            "Disable this adapter or choose its compatible H3 model."
        )


def normalize_fused_h3_request(body: dict, *, resolve_lora=None) -> None:
    """Keep ordinary adapters and their strengths while enforcing the recipe."""

    selected = body.get("activated_loras") or []
    validate_fused_h3_loras(selected)
    if resolve_lora is not None:
        validate_fused_h3_loras([resolve_lora(path) for path in selected])
    body["minimax_h3_turbo_mode"] = False
    body["minimax_h3_turbo_preset"] = ""
    body["num_inference_steps"] = normalize_fused_h3_steps(
        body.get("num_inference_steps")
    )
    body["guidance_scale"] = 1.0
    body["flow_shift"] = 12.0
    body["audio_flow_shift"] = 3.0
    body["skip_steps_cache_type"] = ""
    requested_attention = str(
        body.get("override_attention") or ""
    ).strip().lower()
    body["override_attention"] = (
        "sdpa" if requested_attention == "sdpa" else "sla"
    )


__all__ = [
    "FUSED_H3_DEFAULT_EVALUATIONS",
    "FUSED_H3_EVALUATIONS",
    "FUSED_H3_MAX_EVALUATIONS",
    "FUSED_H3_MIN_EVALUATIONS",
    "FUSED_H3_SOLVER",
    "fused_h3_lora_incompatibility",
    "normalize_fused_h3_request",
    "normalize_fused_h3_steps",
    "validate_fused_h3_loras",
]
