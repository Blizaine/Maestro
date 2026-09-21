"""Opt-in extended H3 output duration; reference media limits stay unchanged."""

# H3's VAE grid is 17*n+5 at 24fps. 719 frames is the last legal pass
# below 30 seconds (29.958s). This is outside the published 15s envelope.
H3_EXPERIMENTAL_MAX_FRAMES = 719
H3_EXPERIMENTAL_MAX_SECONDS = 30.0


def h3_duration_model_def(model_def: dict, inputs: dict) -> dict:
    """Return a job-local duration definition, never modifying model defaults."""
    if (
        inputs.get("minimax_h3_extended_duration") is not True
        or not str(model_def.get("architecture") or "").startswith("minimax_h3")
        or model_def.get("audio_only")
        or model_def.get("minimax_h3_viggle")
    ):
        return model_def
    return {
        **model_def,
        "frames_maximum": H3_EXPERIMENTAL_MAX_FRAMES,
        "sliding_window_defaults": {
            **(model_def.get("sliding_window_defaults") or {}),
            "window_max": H3_EXPERIMENTAL_MAX_FRAMES,
        },
    }


def apply_h3_duration_override(inputs: dict, model_def: dict) -> dict:
    """An explicit experiment must not be silently shortened by Auto sizing."""
    effective = h3_duration_model_def(model_def, inputs)
    if effective is not model_def:
        inputs["sliding_window_memory_override"] = True
        inputs["minimax_h3_sequence_memory_override"] = True
    return effective
