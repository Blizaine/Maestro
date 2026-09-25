"""MiniMax H3 tensor-layout helpers for WanGP ConvRot checkpoints."""

from __future__ import annotations

import json

import torch


def _reorder_qkv_rows(
    tensor: torch.Tensor,
    heads: int,
    head_dim: int,
    *,
    grouped: bool,
) -> torch.Tensor:
    tail = tuple(tensor.shape[1:])
    if grouped:
        return (
            tensor.reshape(heads, 3, head_dim, *tail)
            .permute(1, 0, 2, *range(3, 3 + len(tail)))
            .reshape_as(tensor)
            .contiguous()
        )
    return (
        tensor.reshape(3, heads, head_dim, *tail)
        .permute(1, 0, 2, *range(3, 3 + len(tail)))
        .reshape_as(tensor)
        .contiguous()
    )


def interleave_qkv_rows(tensor: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """Convert grouped Q/K/V rows back to the official head-interleaved order."""

    return _reorder_qkv_rows(tensor, heads, head_dim, grouped=False)


def group_qkv_rows(tensor: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """Convert official head-interleaved rows to contiguous Q/K/V groups."""

    return _reorder_qkv_rows(tensor, heads, head_dim, grouped=True)


def _quantization_descriptor(tensor) -> dict | None:
    if not torch.is_tensor(tensor):
        return None
    try:
        raw = bytes(tensor.detach().cpu().to(torch.uint8).reshape(-1).tolist())
        descriptor = json.loads(raw.decode("utf-8").rstrip("\0"))
        return descriptor if isinstance(descriptor, dict) else None
    except Exception:
        return None


def _is_convrot_config(tensor) -> bool:
    return bool((_quantization_descriptor(tensor) or {}).get("convrot"))


def convrot_quantization_info(state_dict: dict) -> dict[str, int | bool | str | None]:
    """Read ConvRot format and group size from concrete Comfy descriptors."""

    formats = set()
    group_sizes = set()
    convrot = False
    for key, tensor in state_dict.items():
        if not str(key).endswith(".comfy_quant"):
            continue
        descriptor = _quantization_descriptor(tensor)
        if descriptor is None:
            continue
        quantization_format = str(descriptor.get("format") or "").lower()
        if quantization_format:
            formats.add(quantization_format)
        if descriptor.get("convrot"):
            convrot = True
            group_size = descriptor.get(
                "convrot_groupsize",
                descriptor.get(
                    "convrot_group_size",
                    descriptor.get("group_size"),
                ),
            )
            try:
                group_sizes.add(int(group_size))
            except (TypeError, ValueError):
                pass
    return {
        "convrot": convrot,
        "quantization_format": ",".join(sorted(formats)),
        "convrot_group_size": (
            next(iter(group_sizes)) if len(group_sizes) == 1 else None
        ),
    }


def convrot_quantization_info_from_file(
    filename: str,
) -> dict[str, int | bool | str | None]:
    """Read only small Comfy quantization descriptors from a safetensors file."""

    from safetensors import safe_open

    descriptors = {}
    with safe_open(str(filename), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if str(key).endswith(".comfy_quant"):
                descriptors[key] = checkpoint.get_tensor(key)
    return convrot_quantization_info(descriptors)


def has_convrot_layout(state_dict: dict) -> bool:
    """Return whether checkpoint metadata declares ConvRot quantization."""

    return bool(convrot_quantization_info(state_dict)["convrot"])


def restore_interleaved_h3_qkv(state_dict: dict) -> dict:
    """Restore official H3 QKV ordering from a grouped ConvRot export."""

    if not has_convrot_layout(state_dict):
        return state_dict
    norm_key = next(
        key for key in state_dict if key.endswith("blocks.0.attn.q_norm.weight")
    )
    head_dim = int(state_dict[norm_key].shape[0])
    qkv_key = next(
        key for key in state_dict if key.endswith("blocks.0.attn.qkv_proj.weight")
    )
    heads = int(state_dict[qkv_key].shape[0]) // (3 * head_dim)
    for key in [key for key in state_dict if key.endswith(".qkv_proj.weight")]:
        base = key[: -len(".weight")]
        # A module with its own quantization descriptor is decoded by MMGP;
        # only plain/scaled tensors need a physical row reorder here.
        if base + ".comfy_quant" in state_dict:
            continue
        state_dict[key] = interleave_qkv_rows(state_dict[key], heads, head_dim)
        scale_key = base + ".weight_scale"
        if scale_key in state_dict:
            state_dict[scale_key] = interleave_qkv_rows(
                state_dict[scale_key], heads, head_dim
            )
    return state_dict


__all__ = [
    "group_qkv_rows",
    "convrot_quantization_info",
    "convrot_quantization_info_from_file",
    "has_convrot_layout",
    "interleave_qkv_rows",
    "restore_interleaved_h3_qkv",
]
