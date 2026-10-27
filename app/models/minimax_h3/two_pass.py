"""Validation and small pure helpers for the opt-in H3 latent two-pass path.

This file intentionally has no torch dependency so request validation and CPU
tests can exercise the settings contract without loading the H3 runtime.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any


H3_TWO_PASS_ENABLED = "h3_two_pass_enabled"
H3_PHASE2_STEPS = "h3_phase2_steps"
H3_PHASE2_NOISE = "h3_phase2_noise"
H3_PHASE2_ACCELERATOR_STRENGTH = "h3_phase2_accelerator_strength"

H3_TWO_PASS_DEFAULTS = {
    H3_TWO_PASS_ENABLED: "off",
    H3_PHASE2_STEPS: 3,
    H3_PHASE2_NOISE: 0.9035,
    H3_PHASE2_ACCELERATOR_STRENGTH: 1.0,
}

H3_PHASE2_STEPS_MIN = 2
H3_PHASE2_STEPS_MAX = 12
H3_PHASE2_NOISE_MIN = 0.05
H3_PHASE2_NOISE_MAX = 1.0
H3_PHASE2_ACCELERATOR_STRENGTH_MIN = 0.0
H3_PHASE2_ACCELERATOR_STRENGTH_MAX = 2.0


def parse_two_pass_toggle(value: Any) -> bool:
    """Parse the explicit ``off``/``on`` text control without truthiness."""

    if value is None or value is False:
        return False
    if value is True:
        return True
    if not isinstance(value, str):
        raise ValueError("H3 Two-Pass must be explicitly set to 'off' or 'on'.")
    normalized = value.strip().lower()
    if normalized == "off":
        return False
    if normalized == "on":
        return True
    raise ValueError("H3 Two-Pass must be explicitly set to 'off' or 'on'.")


def build_two_pass_generate_request(local_values: Any, generate_signature: Any) -> dict[str, Any]:
    """Keep only generate parameters, then add its intentional ``**kwargs``.

    ``generate`` imports local validation helpers before the two-pass branch;
    blindly forwarding ``locals()`` would leak those helpers into the second
    call. Signature filtering preserves optional parameters while excluding
    implementation locals and the variadic container itself.
    """

    if not isinstance(local_values, dict):
        raise TypeError("H3 two-pass request locals must be a mapping.")
    parameters = getattr(generate_signature, "parameters", generate_signature)
    if not isinstance(parameters, dict):
        parameters = dict(parameters)
    accepted = {
        name
        for name, parameter in parameters.items()
        if name != "self"
        and getattr(getattr(parameter, "kind", None), "name", "")
        not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
    }
    request = {
        name: value
        for name, value in local_values.items()
        if name in accepted
    }
    extras = local_values.get("_kwargs")
    if isinstance(extras, dict):
        request.update(extras)
    return request


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def normalize_two_pass_settings(custom_settings: Any) -> dict[str, Any]:
    """Return validated H3 two-pass settings, defaulting old jobs to Off."""

    values = custom_settings if isinstance(custom_settings, dict) else {}
    enabled = parse_two_pass_toggle(values.get(H3_TWO_PASS_ENABLED))

    raw_steps = values.get(H3_PHASE2_STEPS, H3_TWO_PASS_DEFAULTS[H3_PHASE2_STEPS])
    steps_number = _finite_number(raw_steps, name="H3 phase 2 steps")
    if not steps_number.is_integer():
        raise ValueError("H3 phase 2 steps must be a whole number.")
    steps = int(steps_number)
    if not H3_PHASE2_STEPS_MIN <= steps <= H3_PHASE2_STEPS_MAX:
        raise ValueError(
            f"H3 phase 2 steps must be between {H3_PHASE2_STEPS_MIN} and "
            f"{H3_PHASE2_STEPS_MAX}."
        )

    noise = _finite_number(
        values.get(H3_PHASE2_NOISE, H3_TWO_PASS_DEFAULTS[H3_PHASE2_NOISE]),
        name="H3 phase 2 noise",
    )
    if not H3_PHASE2_NOISE_MIN <= noise <= H3_PHASE2_NOISE_MAX:
        raise ValueError(
            f"H3 phase 2 noise must be between {H3_PHASE2_NOISE_MIN:g} and "
            f"{H3_PHASE2_NOISE_MAX:g}."
        )

    strength = _finite_number(
        values.get(
            H3_PHASE2_ACCELERATOR_STRENGTH,
            H3_TWO_PASS_DEFAULTS[H3_PHASE2_ACCELERATOR_STRENGTH],
        ),
        name="H3 phase 2 accelerator strength",
    )
    if not (
        H3_PHASE2_ACCELERATOR_STRENGTH_MIN
        <= strength
        <= H3_PHASE2_ACCELERATOR_STRENGTH_MAX
    ):
        raise ValueError("H3 phase 2 accelerator strength must be between 0 and 2.")

    return {
        H3_TWO_PASS_ENABLED: "on" if enabled else "off",
        H3_PHASE2_STEPS: steps,
        H3_PHASE2_NOISE: noise,
        H3_PHASE2_ACCELERATOR_STRENGTH: strength,
    }


def two_pass_unavailable_reason(
    model_def: Any,
    *,
    pdd_active: bool = False,
    has_control_video: bool = False,
    has_source_video: bool = False,
    has_edit_mask: bool = False,
    denoising_strength: float = 1.0,
    audio_refinement: bool = False,
    face_refinement: bool = False,
    audio_only: bool = False,
) -> str | None:
    """Explain why the generic two-pass path cannot run this H3 request."""

    definition = model_def if isinstance(model_def, dict) else {}
    incompatible_flags = (
        ("minimax_h3_ptd_refasymopd", "PTD RefAsymOPD"),
        ("minimax_h3_fused_turbo", "Fused Turbo"),
        ("minimax_h3_fasth3", "FastH3"),
        ("minimax_h3_vdn", "VDN"),
        ("vdn", "VDN"),
        ("minimax_h3_dasiwa", "DaSiWa"),
        ("minimax_h3_baked_turbo", "baked Turbo"),
        ("minimax_h3_singularity", "Singularity"),
    )
    for flag, label in incompatible_flags:
        if definition.get(flag):
            return f"H3 two-pass is unavailable for {label} recipes."
    if audio_only or definition.get("minimax_h3_audio_only"):
        return "H3 two-pass is available for generated video, not audio-only jobs."
    if pdd_active:
        return "H3 two-pass is unavailable with PDD heads or PDD accelerators."
    if has_control_video:
        return "H3 two-pass does not support Control Video workflows yet."
    if has_source_video:
        return "H3 two-pass does not support source-video continuation yet."
    if has_edit_mask:
        return "H3 two-pass does not support video edit masks or outpainting."
    try:
        requested_strength = float(denoising_strength)
    except (TypeError, ValueError):
        requested_strength = float("nan")
    if not math.isfinite(requested_strength) or requested_strength != 1.0:
        return "H3 two-pass requires full-strength generation, not partial denoising."
    if audio_refinement:
        return "H3 two-pass and Audio Refinement cannot run in the same job."
    if face_refinement:
        return "H3 two-pass and face refinement cannot run in the same job."
    return None


def resolve_h3_sample_solver(requested: Any, model_def: Any) -> str:
    """Resolve implicit WGP sampler values through the selected H3 recipe.

    WGP commonly submits ``"default"`` or an empty hidden dropdown value.
    Those are not sampler names: H3's model definition owns the native solver.
    Explicit non-default values are returned unchanged so validation can reject
    unsupported requests rather than silently replacing them.
    """

    requested_solver = "" if requested is None else str(requested).strip().lower()
    if requested_solver in ("", "default"):
        definition = model_def if isinstance(model_def, Mapping) else {}
        native_solver = str(
            definition.get("minimax_h3_sampler") or "euler"
        ).strip().lower()
        return native_solver
    return requested_solver


def _has_payload(value: Any) -> bool:
    """Treat empty WGP controls/containers as absent while failing closed."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_payload(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_has_payload(item) for item in value)
    if isinstance(value, (bytes, bytearray)):
        return bool(value)
    if hasattr(value, "numel"):
        try:
            return int(value.numel()) > 0
        except (TypeError, ValueError, OverflowError):
            return True
    try:
        return bool(value)
    except (TypeError, ValueError):
        return True


def _has_positive_outpainting(value: Any) -> bool:
    """Recognize actual WGP outpaint margins, not disabled/zero defaults."""

    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
        if not text or text.startswith("#"):
            return False
        try:
            values = [float(part) for part in text.split()]
        except (TypeError, ValueError, OverflowError):
            # Unknown non-empty text is not safe to treat as disabled.
            return True
        return any(not math.isfinite(item) or item > 0 for item in values)
    if isinstance(value, Mapping):
        return any(_has_positive_outpainting(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_has_positive_outpainting(item) for item in value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return _has_payload(value)
    return not math.isfinite(numeric) or numeric > 0


def has_h3_edit_mask_or_outpainting(inputs: Any) -> bool:
    """Detect active masks/outpaint margins in normalized WGP/API inputs."""

    values = inputs if isinstance(inputs, Mapping) else {}
    if any(
        _has_payload(values.get(name))
        for name in ("video_mask", "image_mask", "input_masks")
    ):
        return True
    return _has_positive_outpainting(values.get("outpainting_dims")) or (
        _has_positive_outpainting(values.get("video_guide_outpainting"))
    )


def phase1_dimensions(height: int, width: int, *, block_size: int = 32) -> tuple[int, int]:
    """Return the aligned half-resolution canvas used by WanGP's H3 recipe."""

    block = max(1, int(block_size))
    half_h = max(block, round(int(height) / 2.0 / block) * block)
    half_w = max(block, round(int(width) / 2.0 / block) * block)
    return half_h, half_w


def phase2_sigmas(steps: int, noise: float) -> tuple[float, ...]:
    """Build the video-only Euler phase-2 grid, including terminal zero."""

    normalized = normalize_two_pass_settings(
        {
            H3_TWO_PASS_ENABLED: "on",
            H3_PHASE2_STEPS: steps,
            H3_PHASE2_NOISE: noise,
        }
    )
    count = normalized[H3_PHASE2_STEPS]
    start = normalized[H3_PHASE2_NOISE]
    return tuple(start * (1.0 - index / count) for index in range(count + 1))


def phase2_clean_audio_timesteps(video_timesteps: Any):
    """Return clean H3 time ``t=1`` for frozen phase-two audio.

    Maestro's scheduler exposes ``t = 1 - sigma``. The upstream H3 phase-two
    call uses audio sigma zero while refining video, which therefore maps to
    time one in Maestro's convention.
    """

    new_ones = getattr(video_timesteps, "new_ones", None)
    if not callable(new_ones):
        raise TypeError("H3 phase-two audio timesteps need a tensor-like input.")
    return new_ones(video_timesteps.shape)


def phase2_lora_schedule(
    loras_slists: Any,
    loras_selected: Any,
    accelerator_paths: Any,
    accelerator_strength: float,
) -> dict[str, Any] | None:
    """Make a temporary phase-2-only schedule without mutating user settings.

    Non-accelerator adapters use their configured phase-2 values. A selected
    Turbo adapter uses the explicit phase-2 accelerator strength control.
    Shared multipliers are preserved from their shared phase-1 value.
    """

    if not isinstance(loras_slists, dict):
        return None
    schedules = copy.deepcopy(loras_slists)
    phase_keys = ("phase1", "phase2", "phase3", "shared")
    if any(not isinstance(schedules.get(key), list) for key in phase_keys):
        return None
    selected = [str(path) for path in (loras_selected or [])]
    accelerator_indexes = {
        index
        for index, path in enumerate(selected)
        if path in set(str(item) for item in (accelerator_paths or ()))
    }
    for index in range(min(len(schedules["phase1"]), len(selected))):
        source = (
            schedules["phase1"][index]
            if schedules["shared"][index]
            else schedules["phase2"][index]
        )
        if index in accelerator_indexes:
            source = float(accelerator_strength)
        for key in ("phase1", "phase2", "phase3"):
            schedules[key][index] = copy.deepcopy(source)
        schedules["shared"][index] = True
    return schedules


__all__ = [
    "H3_PHASE2_ACCELERATOR_STRENGTH",
    "H3_PHASE2_ACCELERATOR_STRENGTH_MAX",
    "H3_PHASE2_ACCELERATOR_STRENGTH_MIN",
    "H3_PHASE2_NOISE",
    "H3_PHASE2_NOISE_MAX",
    "H3_PHASE2_NOISE_MIN",
    "H3_PHASE2_STEPS",
    "H3_PHASE2_STEPS_MAX",
    "H3_PHASE2_STEPS_MIN",
    "H3_TWO_PASS_DEFAULTS",
    "build_two_pass_generate_request",
    "has_h3_edit_mask_or_outpainting",
    "H3_TWO_PASS_ENABLED",
    "normalize_two_pass_settings",
    "parse_two_pass_toggle",
    "phase1_dimensions",
    "phase2_lora_schedule",
    "phase2_clean_audio_timesteps",
    "phase2_sigmas",
    "resolve_h3_sample_solver",
    "two_pass_unavailable_reason",
]
