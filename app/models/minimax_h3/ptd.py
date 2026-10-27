"""Pinned PTD RefAsymOPD eight-hop assets and runtime validation."""

from __future__ import annotations

import math
import os
import re
from typing import Any


PTD_REPO = "DeepBeepMeep/MiniMax-H3"
PTD_REVISION = "de1028e84b0fd5e7f15b2684b704d8c7289b02be"
PTD_FOLDER = "minimax_h3/ptd_refasymopd"
PTD_HEAD_FILE = "minimax_h3_ref2va_ptd_refasymopd_8step_heads_fp32.safetensors"
PTD_NUM_STEPS = 8
PTD_REFERENCE_DETAIL = "ptd768"
PTD_PDD_INTERVALS = 32
PTD_PDD_BLOCK_SIZE = 4
PTD_PDD_RELATIVE_FORMAT = "wangp_pdd_relative_heads_v1"
PTD_SOURCE_REPOSITORY = "taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs"
PTD_SOURCE_REVISION = "5501eb8918ef8bcb858ff045b885bf47a2c4502b"
PTD_SOURCE_SHA256 = "6796802b9150110a3cd7f66891c9e6190502a432c30b0fa1aa299724cfc15cc3"

_PTD_HEAD_KEYS = (
    "pdd_heads.video_out.linear.weight",
    "pdd_heads.video_out.heads",
    "pdd_heads.video_out.linear.bias",
    "pdd_heads.video_out.head_biases",
    "pdd_heads.audio_out.linear.weight",
    "pdd_heads.audio_out.heads",
    "pdd_heads.audio_out.linear.bias",
    "pdd_heads.audio_out.head_biases",
)

PTD_LORA_FILES = tuple(
    f"minimax_h3_ref2va_ptd_refasymopd_8step_hop{step:02d}_bf16.safetensors"
    for step in range(1, PTD_NUM_STEPS + 1)
)
PTD_LORA_URLS = tuple(
    "https://huggingface.co/"
    f"{PTD_REPO}/resolve/{PTD_REVISION}/loras/ptd_refasymopd/{filename}"
    for filename in PTD_LORA_FILES
)
PTD_LORA_MULTIPLIERS = tuple(
    ",".join("1" if index == step - 1 else "0" for index in range(PTD_NUM_STEPS))
    for step in range(1, PTD_NUM_STEPS + 1)
)

_HOP_RE = re.compile(
    r"^minimax_h3_ref2va_ptd_refasymopd_8step_hop(?P<hop>0[1-8])_bf16\.safetensors$",
    re.IGNORECASE,
)


def _asset_basename(path: Any) -> str:
    value = str(path or "").split("|", 1)[0].strip().replace("\\", "/")
    return os.path.basename(value)


def ptd_hop_number(path: Any) -> int | None:
    """Return the one-based PTD hop number when a selected path is a PTD adapter."""

    match = _HOP_RE.fullmatch(_asset_basename(path))
    return int(match.group("hop")) if match is not None else None


def is_ptd_refasymopd_adapter(path: Any) -> bool:
    return ptd_hop_number(path) is not None


def ordered_ptd_adapters(paths: Any) -> tuple[str, ...]:
    """Require every independent hop LoRA exactly once and order by hop."""

    selected = [str(path) for path in (paths or ())]
    indexed = [(ptd_hop_number(path), path) for path in selected]
    found = [(hop, path) for hop, path in indexed if hop is not None]
    if len(found) != PTD_NUM_STEPS:
        raise ValueError(
            "PTD RefAsymOPD requires all eight independent hop LoRAs; "
            f"found {len(found)}. Restore the model's default PTD adapters."
        )
    numbers = [int(hop) for hop, _path in found]
    if sorted(numbers) != list(range(1, PTD_NUM_STEPS + 1)):
        raise ValueError(
            "PTD RefAsymOPD requires each hop adapter exactly once (hop01 through hop08)."
        )
    return tuple(path for _hop, path in sorted(found))


def validate_ptd_schedules(paths: Any, schedules: Any) -> None:
    """Reject profiles that do not activate exactly one matching hop per step."""

    if not isinstance(schedules, dict) or not isinstance(schedules.get("phase1"), list):
        raise ValueError("PTD RefAsymOPD needs an eight-step adapter schedule.")
    selected = [str(path) for path in (paths or ())]
    for hop in range(1, PTD_NUM_STEPS + 1):
        matches = [
            index
            for index, path in enumerate(selected)
            if ptd_hop_number(path) == hop
        ]
        if len(matches) != 1:
            raise ValueError(
                f"PTD RefAsymOPD is missing exactly one hop{hop:02d} adapter."
            )
        index = matches[0]
        if index >= len(schedules["phase1"]):
            raise ValueError("PTD RefAsymOPD adapter schedule does not match its selected LoRAs.")
        values = schedules["phase1"][index]
        if not isinstance(values, (list, tuple)) or len(values) != PTD_NUM_STEPS:
            raise ValueError(
                f"PTD RefAsymOPD hop{hop:02d} must have an eight-value schedule."
            )
        expected = [1.0 if step == hop - 1 else 0.0 for step in range(PTD_NUM_STEPS)]
        try:
            actual = [float(value) for value in values]
        except (TypeError, ValueError) as error:
            raise ValueError(f"PTD RefAsymOPD hop{hop:02d} schedule is invalid.") from error
        if actual != expected:
            raise ValueError(
                f"PTD RefAsymOPD hop{hop:02d} must be active only at denoising step {hop}."
            )


def validate_ptd_cfg_settings(kwargs: Any) -> None:
    """Require CFG one for WGP and direct MiniMax H3 calls.

    WGP currently supplies ``guide_scale`` while older/direct callers may use
    ``guidance_scale``. Validate both if present so an alias cannot bypass the
    fixed PTD recipe.
    """

    values = kwargs if isinstance(kwargs, dict) else {}
    for name in ("guide_scale", "guidance_scale"):
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, bool):
            raise ValueError("PTD RefAsymOPD requires guidance scale 1.0 (CFG 1).")
        try:
            scale = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "PTD RefAsymOPD requires guidance scale 1.0 (CFG 1)."
            ) from error
        if not math.isfinite(scale) or scale != 1.0:
            raise ValueError("PTD RefAsymOPD requires guidance scale 1.0 (CFG 1).")


def _validate_ptd_head_header(handle: Any, transformer: Any) -> None:
    """Validate the pinned base-plus-relative PDD head format without loading it."""

    metadata = handle.metadata() or {}
    expected_metadata = {
        "pdd_num_steps": str(PTD_PDD_INTERVALS),
        "pdd_block_size": str(PTD_PDD_BLOCK_SIZE),
        "ptd_format": PTD_PDD_RELATIVE_FORMAT,
        "ptd_num_hops": str(PTD_NUM_STEPS),
        "ptd_num_curv": str(PTD_NUM_STEPS),
        "source_repository": PTD_SOURCE_REPOSITORY,
        "source_revision": PTD_SOURCE_REVISION,
        "source_sha256": PTD_SOURCE_SHA256,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                "PTD RefAsymOPD head bank metadata mismatch for "
                f"'{key}': expected {expected_value!r}, got {metadata.get(key)!r}."
            )

    final_layer = getattr(transformer, "final_layer", None)
    if final_layer is None:
        raise ValueError("PTD RefAsymOPD requires a transformer final layer.")
    expected_shapes = {
        "pdd_heads.video_out.linear.weight": tuple(final_layer.video_out.weight.shape),
        "pdd_heads.video_out.heads": (
            PTD_PDD_INTERVALS - 1,
            *tuple(final_layer.video_out.weight.shape),
        ),
        "pdd_heads.video_out.linear.bias": tuple(final_layer.video_out.bias.shape),
        "pdd_heads.video_out.head_biases": (
            PTD_PDD_INTERVALS - 1,
            *tuple(final_layer.video_out.bias.shape),
        ),
        "pdd_heads.audio_out.linear.weight": tuple(final_layer.audio_out.weight.shape),
        "pdd_heads.audio_out.heads": (
            PTD_PDD_INTERVALS - 1,
            *tuple(final_layer.audio_out.weight.shape),
        ),
        "pdd_heads.audio_out.linear.bias": tuple(final_layer.audio_out.bias.shape),
        "pdd_heads.audio_out.head_biases": (
            PTD_PDD_INTERVALS - 1,
            *tuple(final_layer.audio_out.bias.shape),
        ),
    }
    available = set(handle.keys())
    missing = set(expected_shapes) - available
    unexpected = available - set(expected_shapes)
    if missing or unexpected:
        raise ValueError(
            "PTD RefAsymOPD head bank tensor keys do not match the pinned "
            f"format (missing={sorted(missing)}, unexpected={sorted(unexpected)})."
        )
    for key, expected_shape in expected_shapes.items():
        tensor_slice = handle.get_slice(key)
        shape = tuple(int(value) for value in tensor_slice.get_shape())
        if shape != expected_shape:
            raise ValueError(
                f"PTD RefAsymOPD head '{key}' shape mismatch: expected "
                f"{expected_shape}, got {shape}."
            )
        dtype = tensor_slice.get_dtype()
        if dtype != "F32":
            raise ValueError(
                f"PTD RefAsymOPD head '{key}' must be FP32, got {dtype}."
            )


def install_ptd_relative_heads(
    transformer: Any,
    checkpoint: str,
    *,
    strength: float = 1.0,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
):
    """Install PTD's pinned base-plus-relative 32-interval PDD heads.

    The first linear projection is interval zero. The remaining tensors are
    deltas from that base, as established by WanGP's PTD checkpoint
    conversion: stack(head 0, heads 1..31), then store heads 1..31 minus head
    zero. Reconstruct absolute intervals here and share Maestro's ordinary
    scheduler-aware 32x4 fusion/controller without changing its PDD file
    format.
    """

    import torch
    from safetensors import safe_open

    from . import pdd

    pdd.release_pdd_parallel_heads(transformer)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        _validate_ptd_head_header(handle, transformer)
        tensors = {
            key: handle.get_tensor(key).clone()
            for key in _PTD_HEAD_KEYS
        }

    video_weight_base = tensors["pdd_heads.video_out.linear.weight"]
    video_bias_base = tensors["pdd_heads.video_out.linear.bias"]
    audio_weight_base = tensors["pdd_heads.audio_out.linear.weight"]
    audio_bias_base = tensors["pdd_heads.audio_out.linear.bias"]
    video_weights = torch.cat(
        (video_weight_base.unsqueeze(0), video_weight_base.unsqueeze(0) + tensors["pdd_heads.video_out.heads"]),
        dim=0,
    )
    video_biases = torch.cat(
        (video_bias_base.unsqueeze(0), video_bias_base.unsqueeze(0) + tensors["pdd_heads.video_out.head_biases"]),
        dim=0,
    )
    audio_weights = torch.cat(
        (audio_weight_base.unsqueeze(0), audio_weight_base.unsqueeze(0) + tensors["pdd_heads.audio_out.heads"]),
        dim=0,
    )
    audio_biases = torch.cat(
        (audio_bias_base.unsqueeze(0), audio_bias_base.unsqueeze(0) + tensors["pdd_heads.audio_out.head_biases"]),
        dim=0,
    )
    return pdd._install_pdd_interval_heads(
        transformer,
        checkpoint,
        video_weights,
        video_biases,
        audio_weights,
        audio_biases,
        strength=strength,
        video_shift=video_shift,
        audio_shift=audio_shift,
        block_size=PTD_PDD_BLOCK_SIZE,
        adapter_name="PTD RefAsymOPD",
    )


PTD_INFOS = """## PTD RefAsymOPD Eight-Step Recipe

- The eight independent RefAsymOPD hop adapters and PDD output heads are loaded with this model. Each 8-step window activates exactly one hop adapter per step.
- The preset fixes Euler, eight evaluations, CFG 1, video/audio shifts 12/3, one phase, and the PTD 768px short-edge reference recipe. Do not add another Turbo/PDD accelerator.
- Phase 2, partial denoising, skip-step caches, and Audio Refinement are unavailable. Published examples focus on image references; other Ref2VA inputs remain experimental.
"""

PTD_DEEPY_INFOS = (
    "PTD RefAsymOPD requires eight Euler evaluations, CFG 1, video/audio shifts "
    "12/3, one guidance phase, full denoising, and 768px-short-edge reference "
    "encoding for every window. Its eight separate hop adapters activate one per "
    "step and its pinned PDD output-head bank is mandatory. Do not add Turbo/PDD "
    "accelerators or use a second denoising phase, skip-step cache, partial-schedule "
    "refinement, or Audio Refinement. Image references match published examples; "
    "other Ref2VA input modes have not been quality-validated."
)


__all__ = [
    "PTD_DEEPY_INFOS",
    "PTD_FOLDER",
    "PTD_HEAD_FILE",
    "PTD_INFOS",
    "PTD_LORA_FILES",
    "PTD_LORA_MULTIPLIERS",
    "PTD_LORA_URLS",
    "PTD_NUM_STEPS",
    "PTD_PDD_BLOCK_SIZE",
    "PTD_PDD_INTERVALS",
    "PTD_PDD_RELATIVE_FORMAT",
    "PTD_REFERENCE_DETAIL",
    "PTD_REPO",
    "PTD_REVISION",
    "PTD_SOURCE_REPOSITORY",
    "PTD_SOURCE_REVISION",
    "PTD_SOURCE_SHA256",
    "install_ptd_relative_heads",
    "is_ptd_refasymopd_adapter",
    "ordered_ptd_adapters",
    "ptd_hop_number",
    "validate_ptd_cfg_settings",
    "validate_ptd_schedules",
]
