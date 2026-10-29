"""Small native helpers for the experimental Pika H3 Flex recipe.

This module keeps request validation and the sampler contract independent of
the H3 runtime.  PyTorch is imported only by :func:`pika_flex_step`, so merely
loading the helpers has no CUDA initialization or model-allocation side effect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


PIKA_FLEX_KNOTS = (1000, 750, 500, 250, 100, 40, 0)
PIKA_FLEX_STEPS = len(PIKA_FLEX_KNOTS) - 1


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def pika_flex_sigmas(shift: float) -> tuple[float, ...]:
    """Return Pika Flex's seven shifted knots for one modality.

    Video and audio call this independently with shifts 12 and 3 respectively.
    The shift is intentionally an argument so callers cannot accidentally
    reuse one modality's sigma ladder for the other.
    """

    shift_value = _finite_float(shift, name="Pika H3 Flex flow shift")
    if shift_value <= 0.0:
        raise ValueError("Pika H3 Flex flow shift must be greater than zero.")
    return tuple(
        shift_value * (knot / 1000.0)
        / (1.0 + (shift_value - 1.0) * (knot / 1000.0))
        for knot in PIKA_FLEX_KNOTS
    )


def pika_flex_step(
    sample: Any,
    prediction: Any,
    sigma: float,
    next_sigma: float,
    generator: Any,
) -> Any:
    """Advance one data-ward velocity evaluation in FP32 arithmetic.

    ``prediction`` is the native H3 velocity ``x0 - eps``.  For each of the
    first five intervals, the clean estimate is mixed with fresh FP32 noise at
    ``next_sigma``.  The terminal interval returns the clean estimate directly
    and does not consume RNG state.  The sampler state remains FP32 even when
    the model evaluates a BF16 or FP16 latent.

    Callers handling references should pass only the target rows to this
    function; the helper never edits either input tensor in place.
    """

    import torch

    if not isinstance(sample, torch.Tensor) or not isinstance(
        prediction, torch.Tensor
    ):
        raise TypeError("Pika H3 Flex sampling requires tensor sample and prediction.")
    if sample.shape != prediction.shape:
        raise ValueError("Pika H3 Flex sample and prediction shapes must match.")
    if sample.device != prediction.device:
        raise ValueError("Pika H3 Flex sample and prediction must be on one device.")
    if not sample.is_floating_point() or not prediction.is_floating_point():
        raise TypeError("Pika H3 Flex sample and prediction must be floating-point tensors.")
    if generator is None:
        raise ValueError("Pika H3 Flex requires an explicit per-request RNG generator.")

    current_sigma = _finite_float(sigma, name="Pika H3 Flex sigma")
    following_sigma = _finite_float(next_sigma, name="Pika H3 Flex next sigma")
    if not 0.0 <= current_sigma <= 1.0:
        raise ValueError("Pika H3 Flex sigma must be between zero and one.")
    if not 0.0 <= following_sigma <= 1.0:
        raise ValueError("Pika H3 Flex next sigma must be between zero and one.")
    if following_sigma > current_sigma:
        raise ValueError("Pika H3 Flex sigma schedule must be nonincreasing.")

    sample_fp32 = sample.to(dtype=torch.float32)
    prediction_fp32 = prediction.to(dtype=torch.float32)
    clean = sample_fp32 + current_sigma * prediction_fp32
    if following_sigma == 0.0:
        return clean

    noise = torch.randn(
        sample.shape,
        device=sample.device,
        dtype=torch.float32,
        generator=generator,
    )
    next_sample = (1.0 - following_sigma) * clean + following_sigma * noise
    return next_sample


def _has_payload(value: Any) -> bool:
    """Treat empty request controls as absent without importing torch."""

    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return len(value) > 0
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            return int(numel()) > 0
        except (TypeError, ValueError, RuntimeError):
            return True
    return True


def _is_enabled(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "off", "false", "no", "none", "0"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return bool(value)


def is_pika_flex_single_frame_start_anchor(
    *,
    image_start: Any,
    input_video: Any,
    prefix_frames_count: Any,
    window_start_frame_no: Any,
    allow_frames: bool,
) -> bool:
    """Recognize WGP's one-frame start-image carrier, not continuation video.

    WGP forwards its start image both as ``image_start`` and as a one-frame
    ``input_video`` carrier with ``prefix_frames_count=1``. H3 discards that
    carrier when an explicit start image is present. Requiring a first window,
    one frame, and the exact one-frame prefix keeps this exception away from
    continuation and multi-window requests.
    """

    if not allow_frames or not _has_payload(image_start):
        return False
    try:
        if _finite_float(prefix_frames_count, name="prefix_frames_count") != 1.0:
            return False
        if _finite_float(window_start_frame_no, name="window_start_frame_no") != 0.0:
            return False
    except (TypeError, ValueError):
        return False

    if not _has_payload(input_video):
        return False
    shape = getattr(input_video, "shape", None)
    if shape is None:
        return False
    try:
        dimensions = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError, OverflowError):
        return False
    if not dimensions or dimensions[0] != 3 or any(dimension <= 0 for dimension in dimensions):
        return False
    if len(dimensions) == 3:
        return True  # C,H,W is one frame in H3's native input convention.
    return len(dimensions) == 4 and dimensions[1] == 1  # C,T,H,W


def _require_value(inputs: Mapping[str, Any], key: str, expected: float) -> None:
    if key not in inputs or inputs[key] is None:
        return
    actual = _finite_float(inputs[key], name=f"Pika H3 Flex {key}")
    if actual != expected:
        raise ValueError(f"Pika H3 Flex requires {key}={expected:g}.")


def validate_pika_flex_request(
    inputs: Mapping[str, Any],
    *,
    allow_frames: bool = False,
    allow_references: bool = False,
) -> None:
    """Reject request options that change Pika Flex's model-owned recipe.

    Standard first/last frame conditioning can be opted into with
    ``allow_frames=True``. H3 image-reference payloads can be enabled separately
    with ``allow_references=True``. This function validates but does not rewrite
    request data.
    """

    if not isinstance(inputs, Mapping):
        raise TypeError("Pika H3 Flex request settings must be a mapping.")

    for key in ("num_inference_steps", "sampling_steps"):
        _require_value(inputs, key, float(PIKA_FLEX_STEPS))
    for key in ("guidance_scale", "cfg_scale"):
        _require_value(inputs, key, 1.0)
    _require_value(inputs, "flow_shift", 12.0)
    _require_value(inputs, "audio_flow_shift", 3.0)
    for key in ("guidance_phases", "guide_phases"):
        _require_value(inputs, key, 1.0)

    for key in ("sample_solver", "solver", "sampler"):
        value = inputs.get(key)
        solver = str(value or "").strip().lower()
        if solver and solver not in {"default", "euler"}:
            raise ValueError("Pika H3 Flex uses its fixed six-evaluation sampler.")

    for key in ("denoising_strength", "denoise_strength", "masking_strength"):
        _require_value(inputs, key, 1.0)

    selected_loras = inputs.get("activated_loras") or inputs.get("loras_selected")
    if _has_payload(selected_loras):
        raise ValueError("Pika H3 Flex does not support additional LoRAs or accelerators.")
    for key in (
        "minimax_h3_turbo_mode",
        "minimax_h3_turbo_preset",
        "minimax_h3_fused_turbo",
        "minimax_h3_baked_turbo",
        "minimax_h3_ptd_refasymopd",
        "minimax_h3_fasth3",
        "minimax_h3_vdn",
        "minimax_h3_dasiwa",
        "minimax_h3_singularity",
        "minimax_h3_pdd",
        "pdd",
        "skip_steps_cache_type",
    ):
        if _is_enabled(inputs.get(key)):
            raise ValueError("Pika H3 Flex does not support additional LoRAs or accelerators.")

    one_frame_start_prefix = (
        allow_frames
        and _has_payload(inputs.get("image_start"))
        and not _has_payload(inputs.get("input_video"))
    )
    if one_frame_start_prefix:
        try:
            one_frame_start_prefix = (
                _finite_float(
                    inputs.get("prefix_frames_count"),
                    name="prefix_frames_count",
                )
                == 1.0
                and _finite_float(
                    inputs.get("window_start_frame_no", 0),
                    name="window_start_frame_no",
                )
                == 0.0
            )
        except (TypeError, ValueError):
            one_frame_start_prefix = False

    for key in (
        "prefix_frames_count",
        "window_start_frame_no",
        "minimax_h3_multi_window",
        "minimax_h3_reference_sequence",
        "continuation",
        "continue_video",
    ):
        if key == "prefix_frames_count" and one_frame_start_prefix:
            continue
        if _is_enabled(inputs.get(key)):
            raise ValueError("Pika H3 Flex does not support continuation or multi-window generation.")

    for key in ("minimax_h3_two_pass", "minimax_h3_two_pass_enabled"):
        if _is_enabled(inputs.get(key)):
            raise ValueError("Pika H3 Flex does not support H3 Two-Pass.")

    custom = inputs.get("custom_settings")
    if isinstance(custom, Mapping):
        if _is_enabled(custom.get("h3_two_pass_enabled")):
            raise ValueError("Pika H3 Flex does not support H3 Two-Pass.")
        if str(custom.get("audio_refinement") or "").strip().lower() == "enabled":
            raise ValueError("Pika H3 Flex does not support Audio Refinement.")

    for key in ("face_refinement", "_face_refinement"):
        if _has_payload(inputs.get(key)):
            raise ValueError("Pika H3 Flex does not support face refinement.")

    reference_keys = ("image_refs", "input_ref_images", "minimax_h3_references")
    if not allow_references and any(_has_payload(inputs.get(key)) for key in reference_keys):
        raise ValueError("This Pika H3 Flex model does not accept image references.")

    frame_keys = ("image_start", "image_end")
    if not allow_frames and any(_has_payload(inputs.get(key)) for key in frame_keys):
        raise ValueError("This Pika H3 Flex model does not accept first/last frame conditioning.")

    unsupported_media = (
        "input_frames",
        "input_video",
        "video_guide",
        "video_source",
        "input_masks",
        "mask",
        "video_mask",
        "frames_to_inject",
        "input_waveform",
        "audio_guide",
        "audio_guide2",
    )
    if any(_has_payload(inputs.get(key)) for key in unsupported_media):
        raise ValueError(
            "Pika H3 Flex does not support control video, masks, timed frame injection, or source audio."
        )
    if _has_payload(inputs.get("video_prompt_type")):
        raise ValueError(
            "Pika H3 Flex does not support control video or timed frame injection."
        )
    if _has_payload(inputs.get("video_guide_outpainting")):
        raise ValueError("Pika H3 Flex does not support masks or outpainting.")
