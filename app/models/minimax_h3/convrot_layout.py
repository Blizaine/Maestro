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


def _is_convrot_config(tensor) -> bool:
    if not torch.is_tensor(tensor):
        return False
    try:
        raw = bytes(tensor.detach().cpu().to(torch.uint8).reshape(-1).tolist())
        return bool(json.loads(raw.decode("utf-8")).get("convrot"))
    except Exception:
        return False


def has_convrot_layout(state_dict: dict) -> bool:
    """Return whether checkpoint metadata declares ConvRot quantization."""

    return any(
        key.endswith(".comfy_quant") and _is_convrot_config(value)
        for key, value in state_dict.items()
    )


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
    "has_convrot_layout",
    "interleave_qkv_rows",
    "restore_interleaved_h3_qkv",
]
