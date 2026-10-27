# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Small Fibo-local tensor adapters for the current PyTorch/attention APIs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rows_(tensor, operation):
    """Apply one operation to a projection's full token rows in place."""
    result = operation(tensor, 0, tensor.shape[1])
    if result is not None and result is not tensor:
        tensor.copy_(result)
    return tensor


def paired_rope_(tensor, cosine, sine):
    """Apply Qwen3-VL's split-half RoPE to ``[batch, tokens, heads, dim]``."""
    if tensor is None:
        return None
    half = tensor.shape[-1] // 2
    if tensor.shape[-1] % 2:
        raise ValueError("Rotary head dimensions must be even.")
    cosine = cosine.to(device=tensor.device, dtype=tensor.dtype)
    sine = sine.to(device=tensor.device, dtype=tensor.dtype)
    first = tensor[..., :half].clone()
    second = tensor[..., half:].clone()
    tensor[..., :half].copy_(first * cosine - second * sine)
    tensor[..., half:].copy_(second * cosine + first * sine)
    return tensor


def _interleaved_norm_rope(tensor, table, norm):
    if tensor is None:
        return None
    normalized = norm(tensor)
    pair_count = tensor.shape[-1] // 2
    if tensor.shape[-1] % 2 or table.shape[-2] != pair_count:
        raise ValueError("Fibo rotary table does not match the attention head dimension.")
    cosine = table[..., 0].to(device=tensor.device, dtype=normalized.dtype)
    sine = table[..., 1].to(device=tensor.device, dtype=normalized.dtype)
    pairs = normalized.reshape(*normalized.shape[:-1], pair_count, 2)
    even, odd = pairs.unbind(-1)
    rotated = torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1)
    tensor.copy_(rotated.flatten(-2).to(dtype=tensor.dtype))
    return tensor


def interleaved_norm_rope_(query, key, table, query_norm, key_norm):
    """Apply Fibo's adjacent-pair rotary embedding after its per-head RMSNorm."""
    _interleaved_norm_rope(query, table, query_norm)
    _interleaved_norm_rope(key, table, key_norm)
    return query, key


def norm_modulate(inputs, norm, shift, scale, segments):
    """Normalize flattened token rows and apply their row-indexed modulation."""
    value = inputs.pop()
    if value.ndim != 2:
        raise ValueError("Fibo modulation expects flattened [rows, channels] input.")
    result = torch.empty_like(value)
    for start, stop, row in segments:
        normalized = norm(value[start:stop])
        result[start:stop].copy_(
            normalized * (1 + scale[row].to(normalized)) + shift[row].to(normalized)
        )
    return result


def gated_residual_(value, residuals, gate, segments=None, tanh=False):
    """Add one or more Fibo residual streams, optionally split by modulation row."""
    if not residuals:
        return value
    residual = residuals.pop()
    if value.shape != residual.shape:
        raise ValueError("Fibo residual and target tensors must have the same shape.")
    if not value.is_contiguous():
        value = value.contiguous()
    target = value.view(-1, value.shape[-1])
    source = residual.reshape(-1, residual.shape[-1])

    def row_gate(row=None):
        selected = gate if row is None or gate.numel() == 1 else gate[row]
        return torch.tanh(selected) if tanh else selected

    if segments is None:
        target.add_(source * row_gate().to(source))
        return value
    for start, stop, row in segments:
        target[start:stop].add_(source[start:stop] * row_gate(row).to(source))
    return value


def gated_mlp(inputs, gate_proj, up_proj, down_proj, activation):
    """Run the checkpoint's SwiGLU feed-forward projection."""
    value = inputs.pop()
    return down_proj(activation(gate_proj(value)) * up_proj(value))


def mlp(inputs, gate_proj, down_proj, activation="swiglu", up=None):
    """Fibo block MLP adapter with the FFN input tensor handoff convention."""
    if not inputs:
        return None
    value = inputs.pop()
    if activation == "swiglu":
        if up is None:
            raise ValueError("SwiGLU needs both gate and up projections.")
        return down_proj(F.silu(gate_proj(value), inplace=True) * up(value))
    raise ValueError(f"Unsupported Fibo MLP activation: {activation!r}.")
