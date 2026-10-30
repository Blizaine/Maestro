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
PIKA_FLEX_ATTENTION_MODES = ("sage2", "sdpa", "sol", "sla")
PIKA_FLEX_FRAME_MINIMUM = 124
PIKA_FLEX_FRAME_STEP = 17
PIKA_FLEX_FRAME_MAXIMUM = 362


def normalize_pika_flex_attention(value: Any) -> str:
    """Keep attention selection separate from the fixed six-step sampler."""

    mode = str(value or "sla").strip().lower()
    if mode in {"", "auto", "default"}:
        mode = "sla"
    if mode not in PIKA_FLEX_ATTENTION_MODES:
        raise ValueError("Pika H3 Flex supports Sage2, SDPA, experimental SOL or experimental SLA attention.")
    return mode


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


def pika_flex_sequence_enabled(
    inputs: Mapping[str, Any],
    *,
    reference: bool,
) -> bool:
    """Read the sequence opt-in that belongs to this Flex workflow."""

    if not isinstance(inputs, Mapping):
        raise TypeError("Pika H3 Flex request settings must be a mapping.")
    active_key = (
        "minimax_h3_reference_sequence"
        if reference
        else "minimax_h3_multi_window"
    )
    inactive_key = (
        "minimax_h3_multi_window"
        if reference
        else "minimax_h3_reference_sequence"
    )
    for key in (active_key, inactive_key):
        value = inputs.get(key)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"Pika H3 Flex requires {key} to be a boolean.")
    if inputs.get(inactive_key) is True:
        workflow = "References" if reference else "Frames"
        raise ValueError(
            f"Pika H3 Flex {workflow} uses {active_key} for native continuation."
        )
    return inputs.get(active_key) is True


def _strict_int(value: Any, *, name: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    number = _finite_float(value, name=name)
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number.")
    return int(number)


def _shape(value: Any, *, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"Pika H3 Flex continuation {name} must be a tensor.")
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Pika H3 Flex continuation {name} has an invalid shape."
        ) from error


def _validate_runtime_continuation(inputs: Mapping[str, Any]) -> None:
    """Require a legal generated visual and stereo-audio overlap on later passes."""

    window_start = _strict_int(
        inputs.get("window_start_frame_no", 0),
        name="window_start_frame_no",
        default=0,
    )
    prefix_count = _strict_int(
        inputs.get("prefix_frames_count", 0),
        name="prefix_frames_count",
        default=0,
    )
    input_video = inputs.get("input_video")
    input_waveform = inputs.get("input_waveform")
    if window_start <= 0:
        if prefix_count or _has_payload(input_video) or _has_payload(input_waveform):
            raise ValueError(
                "Pika H3 Flex continuation prefixes are valid only after the first window."
            )
        return

    if prefix_count <= 0 or not _has_payload(input_video):
        raise ValueError(
            "Pika H3 Flex continuation needs the complete generated video overlap."
        )
    video_shape = _shape(input_video, name="video prefix")
    if len(video_shape) != 4 or video_shape[0] != 3 or video_shape[1] != prefix_count:
        raise ValueError(
            "Pika H3 Flex continuation video must be a C,T,H,W RGB tensor whose "
            "frame count matches prefix_frames_count."
        )
    if prefix_count < 1 or (prefix_count - 1) % PIKA_FLEX_FRAME_STEP:
        raise ValueError(
            "Pika H3 Flex continuation overlap must use the 17*n+1 frame grid."
        )

    frame_count = _strict_int(inputs.get("frame_num"), name="frame_num")
    if (
        frame_count < PIKA_FLEX_FRAME_MINIMUM
        or frame_count > PIKA_FLEX_FRAME_MAXIMUM
        or (frame_count - 5) % PIKA_FLEX_FRAME_STEP
    ):
        raise ValueError(
            "Pika H3 Flex continuation windows must use 124–362 frames on the "
            "17*n+5 grid."
        )
    # The pass cap includes the overlap context. The newly generated portion
    # can therefore be shorter than the 124-frame standalone minimum while
    # still occupying a legal H3 target lattice (124 with an 18-frame overlap
    # generates 107 frames: 17*n+5). WGP owns the prefix extraction and its
    # audio-tail rounding, so this checks only the shared latent/frame grid.
    target_frames = frame_count - (prefix_count - 1)
    if target_frames < 5 or (target_frames - 5) % PIKA_FLEX_FRAME_STEP:
        raise ValueError(
            "Pika H3 Flex overlap must leave a legal 17*n+5 target for the next window."
        )

    if not _has_payload(input_waveform):
        raise ValueError(
            "Pika H3 Flex continuation needs the matching generated stereo-audio overlap."
        )
    audio_shape = _shape(input_waveform, name="audio prefix")
    if len(audio_shape) != 2 or 2 not in audio_shape:
        raise ValueError(
            "Pika H3 Flex continuation audio must contain both generated stereo channels."
        )
    sample_rate = _strict_int(
        inputs.get("input_waveform_sample_rate"),
        name="input_waveform_sample_rate",
    )
    if sample_rate <= 0:
        raise ValueError("Pika H3 Flex continuation audio needs a positive sample rate.")
    audio_samples = audio_shape[0] if audio_shape[1] == 2 else audio_shape[1]
    expected_samples = int(round(prefix_count / 24.0 * sample_rate))
    # WGP slices this many samples from the decoded tail. Audio VAE/codec
    # rounding can leave a small shortfall at the end of a decoded window.
    # Allow up to one video-frame of PCM rounding while still rejecting a
    # mismatched or empty history segment.
    rounding_tolerance = max(1, int(math.ceil(sample_rate / 24.0)))
    if audio_samples <= 0 or abs(audio_samples - expected_samples) > rounding_tolerance:
        raise ValueError(
            "Pika H3 Flex continuation audio length must align with its video overlap."
        )


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


def normalize_pika_flex_start_prefix(
    prefix_video: Any,
    source_frame_count: int,
    overlap_frame_count: int,
    *,
    image_start: Any,
    window_no: int,
) -> tuple[Any, int, int]:
    """Remove the generic image carrier from Flex's already complete timeline.

    Native Flex returns the first pixel frame and its corresponding audio.
    Treating that conditioning image as a source-video prefix trims native
    audio and then prepends silence. The finite concat can also clip B-frames
    at muxing. Genuine video/history prefixes keep their existing handling.
    """

    from .packing import normalize_h3_native_start_prefix

    return normalize_h3_native_start_prefix(
        prefix_video,
        source_frame_count,
        overlap_frame_count,
        image_start=image_start,
        window_no=window_no,
        architecture="minimax_h3_pika_flex",
    )


def validate_pika_flex_request(
    inputs: Mapping[str, Any],
    *,
    allow_frames: bool = False,
    allow_references: bool = False,
    allow_continuation: bool = False,
) -> None:
    """Reject request options that change Pika Flex's model-owned recipe.

    Standard first/last frame conditioning can be opted into with
    ``allow_frames=True``. H3 image-reference payloads can be enabled separately
    with ``allow_references=True``. Native rolling continuation is enabled only
    by the workflow's explicit sequence flag and is validated against its real
    generated video and stereo-audio prefixes. This function does not rewrite
    request data.
    """

    if not isinstance(inputs, Mapping):
        raise TypeError("Pika H3 Flex request settings must be a mapping.")

    normalize_pika_flex_attention(inputs.get("override_attention"))

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
    from .pika_flex_lora_compatibility import validate_pika_flex_additional_loras

    validate_pika_flex_additional_loras(
        selected_loras,
        enabled=inputs.get("minimax_h3_experimental_loras"),
    )
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
    ):
        if _is_enabled(inputs.get(key)):
            raise ValueError("Pika H3 Flex does not support additional LoRAs or accelerators.")

    if _is_enabled(inputs.get("skip_steps_cache_type")):
        raise ValueError(
            "Pika H3 Flex does not support First Block or other step-skipping caches. "
            "Disable First Block for this model; SOL and SLA attention can still be used."
        )

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

    sequence_enabled = pika_flex_sequence_enabled(
        inputs,
        reference=allow_references,
    )
    if sequence_enabled and not allow_continuation:
        raise ValueError(
            "Pika H3 Flex native continuation requires the explicit sequence workflow."
        )
    if allow_continuation and not sequence_enabled:
        raise ValueError(
            "Pika H3 Flex continuation requires its explicit sequence flag."
        )
    if not allow_continuation:
        for key in ("prefix_frames_count", "window_start_frame_no"):
            if key == "prefix_frames_count" and one_frame_start_prefix:
                continue
            if _is_enabled(inputs.get(key)):
                raise ValueError(
                    "Pika H3 Flex does not support continuation without an explicit sequence."
                )
    for key in ("continuation", "continue_video"):
        if _is_enabled(inputs.get(key)):
            raise ValueError(
                "Pika H3 Flex uses its model-specific sequence flag for continuation."
            )

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
        "video_guide",
        "video_source",
        "input_masks",
        "mask",
        "video_mask",
        "frames_to_inject",
        "audio_guide",
        "audio_guide2",
    )
    if any(_has_payload(inputs.get(key)) for key in unsupported_media):
        raise ValueError(
            "Pika H3 Flex does not support control video, masks, timed frame injection, or source audio."
        )
    has_runtime_prefix = any(
        _has_payload(inputs.get(key))
        for key in ("input_video", "input_waveform")
    )
    if has_runtime_prefix:
        if not allow_continuation:
            raise ValueError(
                "Pika H3 Flex does not support source video/audio; native generated "
                "prefixes require an explicit sequence."
            )
        _validate_runtime_continuation(inputs)
    elif allow_continuation and _is_enabled(inputs.get("window_start_frame_no")):
        _validate_runtime_continuation(inputs)
    if _has_payload(inputs.get("video_prompt_type")):
        raise ValueError(
            "Pika H3 Flex does not support control video or timed frame injection."
        )
    if _has_payload(inputs.get("video_guide_outpainting")):
        raise ValueError("Pika H3 Flex does not support masks or outpainting.")
