"""Model-aware compatibility checks for MiniMax H3 LoRA adapters."""

from __future__ import annotations

import os
import re

from .turbo import safetensors_header


_VDN_TENSOR_MARKERS = (
    ".attn.linear_attention.",
    ".attn.softmax_gate.",
    ".attn.to_out_linear.",
    ".attn.vdn.",
)


def is_minimax_h3_vdn_lora(path: str) -> bool:
    """Recognize H3 VDN adapters by their name or tensor namespace.

    Header inspection reads only the safetensors JSON header and catches files
    that expose VDN-only attention namespaces. Some VDN-trained adapters use
    only shared H3 module names, so their known model-specific filename is
    still needed when the header has no dedicated marker.
    """

    path = str(path or "")
    basename = os.path.basename(path.replace("\\", "/")).lower()
    name_tokens = set(re.findall(r"[a-z0-9]+", basename))
    h3_adapter_name = {"minimax", "h3"}.issubset(name_tokens)
    if "vdn" in name_tokens and h3_adapter_name:
        return True

    if not basename.endswith((".safetensors", ".sft")):
        return False
    header = safetensors_header(path)
    return any(
        marker in str(key).lower()
        for key in header
        if key != "__metadata__"
        for marker in _VDN_TENSOR_MARKERS
    )


def minimax_h3_lora_incompatibility(
    model_def: dict | None,
    path: str,
) -> str | None:
    """Return a known H3 model/adapter incompatibility, if one is proven."""

    definition = model_def or {}
    import_profile = definition.get("minimax_h3_import_profile") or {}
    if (
        definition.get("minimax_h3_fused_turbo")
        or definition.get("minimax_h3_baked_turbo")
        or import_profile.get("baked_turbo")
        or import_profile.get("fused_turbo")
    ):
        from .fused_turbo import fused_h3_lora_incompatibility

        return fused_h3_lora_incompatibility(path)
    if not definition.get("vdn") and is_minimax_h3_vdn_lora(path):
        return "VDN adapters require the dedicated MiniMax H3 VDN model"
    return None


def validate_minimax_h3_loras(paths, model_def: dict | None) -> None:
    """Reject known incompatible H3 adapters before MMGP loads their weights."""

    for path in paths or ():
        reason = minimax_h3_lora_incompatibility(model_def, str(path))
        if reason is None:
            continue
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        raise ValueError(f"MiniMax H3 cannot use {name}: {reason}.")


_FASTH3_ACCELERATOR_TOKENS = {
    "turbo", "pdd", "acc", "acc8step", "acceleration", "accelerator",
}
_FASTH3_DORA_MARKERS = ("lora_magnitude_vector", ".dora_scale")
_FASTH3_PDD_HEAD_SHAPES = {
    "proj_out.weight": 96,
    "audio_proj_out.weight": 32,
}
_FASTH3_OMNI_OUTPUT_PREFIXES = (
    "proj_out.", "audio_proj_out.",
    "final_layer.video_out.", "final_layer.audio_out.",
)


def _has_pdd_head_bank(header: dict) -> bool:
    """Recognize the 32-interval H3 PDD output banks from descriptors only."""

    for key, width in _FASTH3_PDD_HEAD_SHAPES.items():
        descriptor = header.get(key)
        shape = descriptor.get("shape") if isinstance(descriptor, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(value) is not int for value in shape)
            or shape[0] != 32
            or shape[1] != width
        ):
            return False
    return True


def _is_omni_output_adapter_key(key: str) -> bool:
    normalized = str(key).lower()
    prefixes = ("model.diffusion_model.", "diffusion_model.", "transformer.")
    while True:
        prefix = next(
            (candidate for candidate in prefixes if normalized.startswith(candidate)),
            None,
        )
        if prefix is None:
            break
        normalized = normalized[len(prefix) :]

    # H3's name normalizer accepts Kohya/Comfy flattened module names by
    # converting a live module path such as ``final_layer.video_out`` to
    # ``lora_unet_final_layer_video_out``. Treat those spellings exactly like
    # their dotted names so OmniRef's fused PDD output heads cannot be patched
    # before the integration combines the 32 predictions.
    if normalized.startswith("lora_unet_"):
        remainder = normalized[len("lora_unet_") :]
        module_token, separator, suffix = remainder.partition(".")
        flattened_outputs = {
            "final_layer_video_out": "final_layer.video_out",
            "final_layer_audio_out": "final_layer.audio_out",
            # Keep the same legacy aliases as the dotted check below.
            "proj_out": "proj_out",
            "audio_proj_out": "audio_proj_out",
        }
        module_path = flattened_outputs.get(module_token)
        if separator and module_path is not None:
            normalized = f"{module_path}.{suffix}"

    if not normalized.startswith(_FASTH3_OMNI_OUTPUT_PREFIXES):
        return False
    return any(
        marker in normalized
        for marker in (
            ".lora_", ".lora.", ".lora/", ".lora-", ".lokr_",
            ".dora_scale", ".diff", ".weight_diff",
        )
    )


def fasth3_lora_incompatibility(
    path: str,
    *,
    omni_reference: bool = False,
) -> str | None:
    """Return a known FastH3 adapter incompatibility without loading tensors.

    FastH3 keeps its published eight-step schedule and attention path. This
    check excludes other acceleration recipes and adapter formats that need a
    different runtime. Ordinary H3 LoRAs remain experimental and are allowed
    through WanGP's normal H3 name, basis, and tensor-shape validation.
    """

    raw_path = str(path or "")
    basename = os.path.basename(raw_path.replace("\\", "/")).lower()
    tokens = set(re.findall(r"[a-z0-9]+", basename))
    header = (
        safetensors_header(raw_path)
        if basename.endswith((".safetensors", ".sft"))
        else {}
    )
    keys = tuple(str(key).lower() for key in header if key != "__metadata__")

    from .turbo import is_minimax_h3_pdd_lora, is_minimax_h3_turbo_lora

    if "pdd" in tokens or is_minimax_h3_pdd_lora(raw_path) or _has_pdd_head_bank(header):
        return "H3 PDD adapters use a different trained schedule and output-head bank"

    if is_minimax_h3_vdn_lora(raw_path):
        return "VDN adapters require their matched VDN recipe and are unavailable on FastH3"

    if "dora" in tokens or any(marker in key for key in keys for marker in _FASTH3_DORA_MARKERS):
        return "DoRA adapters are unsupported by the FastH3 LoRA path; use an ordinary H3 LoRA"

    if (
        is_minimax_h3_turbo_lora(raw_path)
        or tokens.intersection(_FASTH3_ACCELERATOR_TOKENS)
    ):
        return "FastH3 already uses its fixed eight-step distilled recipe; additional Turbo/acceleration adapters cannot be stacked"

    if omni_reference and any(_is_omni_output_adapter_key(key) for key in keys):
        return (
            "FastH3 OmniRef fuses 32 PDD output heads per evaluation; adapters targeting "
            "video_out/audio_out bypass that fusion and are unsupported"
        )

    return None


def validate_fasth3_loras(
    paths,
    *,
    enabled,
    omni_reference: bool = False,
) -> None:
    """Validate FastH3 LoRAs; opt-in must be the literal boolean ``True``."""

    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    selected = [str(path) for path in (paths or ()) if str(path or "")]
    if not selected:
        return
    if enabled is not True:
        raise ValueError(
            "FastH3 LoRAs are experimental and off by default. Enable "
            "minimax_h3_experimental_loras for this generation to use them."
        )
    for path in selected:
        reason = fasth3_lora_incompatibility(
            path,
            omni_reference=omni_reference,
        )
        if reason is None:
            continue
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        raise ValueError(f"FastH3 cannot use {name}: {reason}.")


__all__ = [
    "fasth3_lora_incompatibility",
    "is_minimax_h3_vdn_lora",
    "minimax_h3_lora_incompatibility",
    "validate_fasth3_loras",
    "validate_minimax_h3_loras",
]
