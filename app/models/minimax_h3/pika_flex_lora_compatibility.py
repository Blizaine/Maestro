"""Fail-closed opt-in checks for ordinary LoRAs on Pika H3 Flex."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any


_VDN_TENSOR_MARKERS = (
    ".attn.linear_attention.",
    ".attn.softmax_gate.",
    ".attn.to_out_linear.",
    ".attn.vdn.",
)
_DORA_TENSOR_MARKERS = ("lora_magnitude_vector", ".dora_scale")
_ACCELERATOR_TOKENS = {
    "turbo", "pdd", "acc", "acc8step", "acceleration", "accelerator",
}
_OTHER_SPECIAL_TOKENS = {
    "dasiwa", "distillation", "distilled", "fasth3", "fastvideo",
    "lightx2v", "singularity", "dmd",
}


def _path_text(path: Any) -> str:
    try:
        value = os.fspath(path)
    except TypeError as error:
        raise TypeError("Pika H3 Flex LoRA selections must be paths.") from error
    if not isinstance(value, str):
        raise TypeError("Pika H3 Flex LoRA selections must be text paths.")
    return value.split("|", 1)[0].strip()


def _path_list(paths: Any) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        values = [paths]
    elif isinstance(paths, Sequence) and not isinstance(paths, (bytes, bytearray)):
        values = list(paths)
    else:
        raise TypeError("Pika H3 Flex LoRA selections must be a path or path list.")
    result = []
    for value in values:
        path = _path_text(value)
        if path:
            result.append(path)
    return result


def _basename(path: str) -> str:
    return os.path.basename(path.replace("\\", "/")).casefold()


def _tokens(path: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _basename(path)))


def _header(path: str) -> dict:
    if not _basename(path).endswith((".safetensors", ".sft")):
        return {}
    from .turbo import safetensors_header

    header = safetensors_header(path)
    return header if isinstance(header, dict) else {}


def _has_pdd_head_bank(header: dict) -> bool:
    expected = {"proj_out.weight": 96, "audio_proj_out.weight": 32}
    for key, width in expected.items():
        descriptor = header.get(key)
        shape = descriptor.get("shape") if isinstance(descriptor, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(size) is not int for size in shape)
            or shape[0] != 32
            or shape[1] != width
        ):
            return False
    return True


def pika_flex_lora_incompatibility(path: Any) -> str | None:
    """Explain known adapters that cannot be stacked onto the Pika recipe."""

    path = _path_text(path)
    tokens = _tokens(path)
    header = _header(path)
    keys = tuple(str(key).casefold() for key in header if key != "__metadata__")

    from .ptd import is_ptd_refasymopd_adapter
    from .turbo import is_minimax_h3_pdd_lora, is_minimax_h3_turbo_lora

    if is_ptd_refasymopd_adapter(path) or "ptd" in tokens or "refasymopd" in tokens:
        return "PTD RefAsymOPD hop adapters require their dedicated eight-step recipe"
    if (
        "vdn" in tokens
        or any(marker in key for key in keys for marker in _VDN_TENSOR_MARKERS)
    ):
        return "VDN adapters require the dedicated VDN attention recipe"
    if "pdd" in tokens or is_minimax_h3_pdd_lora(path) or _has_pdd_head_bank(header):
        return "PDD adapters use a different step schedule and special output-head bank"
    if "dora" in tokens or any(
        marker in key for key in keys for marker in _DORA_TENSOR_MARKERS
    ):
        return "DoRA adapters are unsupported by the Pika H3 Flex LoRA path"
    if (
        is_minimax_h3_turbo_lora(path)
        or tokens.intersection(_ACCELERATOR_TOKENS)
    ):
        return "Turbo and other step-acceleration adapters cannot be stacked onto Pika's fixed six-step recipe"
    if (
        tokens.intersection(_OTHER_SPECIAL_TOKENS)
        or any(key.endswith(".attn.to_gate_compress.weight") for key in keys)
    ):
        return "This adapter uses a separate distilled or special-head H3 recipe"
    return None


def _managed_pika_lora_filenames() -> set[str]:
    from .pika_flex_assets import PIKA_FLEX_LORA_SPECS

    return {str(spec["file"]).casefold() for spec in PIKA_FLEX_LORA_SPECS.values()}


def validate_pika_flex_additional_loras(
    paths: Any,
    *,
    enabled: Any,
) -> tuple[str, ...]:
    """Require literal-True opt-in for ordinary H3 style/content adapters."""

    selected = _path_list(paths)
    if not selected:
        return ()

    managed = _managed_pika_lora_filenames()
    for path in selected:
        if _basename(path) in managed:
            raise ValueError(
                "Pika H3 Flex's pinned adapter is managed by the model and cannot be selected as an extra LoRA."
            )
        reason = pika_flex_lora_incompatibility(path)
        if reason:
            raise ValueError(
                f"Pika H3 Flex cannot use {_basename(path)}: {reason}."
            )

    if enabled is not True:
        raise ValueError(
            "Pika H3 Flex style/content LoRAs are experimental and off by default. "
            "Enable minimax_h3_experimental_loras for this generation to use them."
        )
    return tuple(selected)


def validate_pika_flex_lora_selection(
    paths: Any,
    *,
    reference: bool,
    enabled: Any,
) -> tuple[str, ...]:
    """Require one verified model-owned adapter first, then optional extras."""

    if not isinstance(reference, bool):
        raise TypeError("Pika H3 Flex reference must be a boolean.")
    selected = _path_list(paths)
    from .pika_flex_assets import PIKA_FLEX_LORA_SPECS, verify_pika_flex_loras

    pinned = str(PIKA_FLEX_LORA_SPECS[reference]["file"]).casefold()
    if not selected or _basename(selected[0]) != pinned:
        raise ValueError(
            "Pika H3 Flex requires its pinned adapter as the first model-owned LoRA."
        )
    if sum(_basename(path) in _managed_pika_lora_filenames() for path in selected) != 1:
        raise ValueError(
            "Pika H3 Flex's pinned adapter must appear exactly once; managed Pika adapters cannot be selected as extras."
        )

    extras = validate_pika_flex_additional_loras(
        selected[1:], enabled=enabled,
    )
    verify_pika_flex_loras([selected[0]], reference=reference)
    return (selected[0], *extras)


def validate_pika_flex_adapter_strengths(paths: Any, strengths: Any, *, reference: bool) -> None:
    """Keep the first, model-owned Pika adapter at exactly strength 1."""

    selected = _path_list(paths)
    from .pika_flex_assets import PIKA_FLEX_LORA_SPECS

    pinned = str(PIKA_FLEX_LORA_SPECS[reference]["file"]).casefold()
    if not selected or _basename(selected[0]) != pinned:
        raise ValueError("Pika H3 Flex's pinned adapter must remain first in the LoRA list.")
    if not isinstance(strengths, Sequence) or isinstance(strengths, (str, bytes, bytearray)):
        raise ValueError("Pika H3 Flex LoRA strengths do not match the selected adapters.")
    if len(strengths) != len(selected) or not strengths:
        raise ValueError("Pika H3 Flex LoRA strengths do not match the selected adapters.")

    value = strengths[0]
    if hasattr(value, "tolist"):
        value = value.tolist()
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    try:
        normalized = [float(item) for item in values]
    except (TypeError, ValueError) as error:
        raise ValueError("Pika H3 Flex's pinned adapter strength must remain 1.0.") from error
    if not normalized or any(item != 1.0 for item in normalized):
        raise ValueError("Pika H3 Flex's pinned adapter strength must remain 1.0.")


__all__ = [
    "pika_flex_lora_incompatibility",
    "validate_pika_flex_adapter_strengths",
    "validate_pika_flex_additional_loras",
    "validate_pika_flex_lora_selection",
]
