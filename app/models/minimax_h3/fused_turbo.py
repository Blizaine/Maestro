"""Policy helpers for the baked MATLOWAI MiniMax H3 Turbo checkpoint."""

from __future__ import annotations

from .turbo import is_minimax_h3_turbo_lora


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


def normalize_fused_h3_request(body: dict) -> int:
    """Normalize the baked recipe and remove stale managed accelerators.

    Returns the number of managed Turbo/PDD selections removed after a model
    switch. Ordinary user LoRAs are rejected instead of being silently lost.
    """

    selected = list(body.get("activated_loras") or [])
    stale_managed = [item for item in selected if is_minimax_h3_turbo_lora(item)]
    ordinary = [item for item in selected if not is_minimax_h3_turbo_lora(item)]
    if ordinary:
        names = ", ".join(str(item).rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for item in ordinary)
        raise ValueError(
            "H3 Fused 4-Step already contains its Turbo and Mystic adapters. "
            f"Disable these additional LoRAs before generating: {names}."
        )
    body["activated_loras"] = []
    body["loras_multipliers"] = ""
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
    return len(stale_managed)


__all__ = [
    "FUSED_H3_DEFAULT_EVALUATIONS",
    "FUSED_H3_EVALUATIONS",
    "FUSED_H3_MAX_EVALUATIONS",
    "FUSED_H3_MIN_EVALUATIONS",
    "FUSED_H3_SOLVER",
    "normalize_fused_h3_request",
    "normalize_fused_h3_steps",
]
