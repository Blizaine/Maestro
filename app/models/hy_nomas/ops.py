# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Small model-local PyTorch adapters for the current MMGP 4.x runtime."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def interleaved_norm_rope_(query, key, table, query_norm, key_norm):
    """Normalize and apply adjacent-pair RoPE to the supplied query/key rows.

    Hy-Nomas uses 64 rotary pairs in 128-wide heads. Text tokens use the same
    Q/K normalization without a rotary table.
    """
    for tensor, norm in ((query, query_norm), (key, key_norm)):
        if tensor is None:
            continue
        tensor.copy_(norm(tensor))
        if table is None:
            continue
        if tensor.ndim != 4 or tensor.shape[-1] % 2:
            raise ValueError("Hy-Nomas rotary input must be [batch, tokens, heads, even_dim].")
        cosine, sine = table[..., 0], table[..., 1]
        pairs = tensor.reshape(*tensor.shape[:-1], tensor.shape[-1] // 2, 2)
        if cosine.shape[-1] != pairs.shape[-2] or cosine.shape[1] != tensor.shape[1]:
            raise ValueError("Hy-Nomas rotary table does not match the token or head dimensions.")
        pairs32 = pairs.float()
        even, odd = pairs32.unbind(-1)
        cosine = cosine.to(device=tensor.device, dtype=torch.float32)
        sine = sine.to(device=tensor.device, dtype=torch.float32)
        rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
        tensor.copy_(rotated.flatten(-2).to(dtype=tensor.dtype))
    return query, key


def norm_modulate(inputs, norm, shift, scale, segments):
    value = inputs.pop()
    if value.ndim != 2:
        raise ValueError("Hy-Nomas modulation expects flattened [tokens, channels] input.")
    result = torch.empty_like(value)
    for start, stop, row in segments:
        normalized = norm(value[start:stop])
        result[start:stop].copy_(normalized * (1 + scale[row].to(normalized)) + shift[row].to(normalized))
    return result


def gated_residual_(value, residuals, gate, segments=None, tanh=False, fused=False):
    if not residuals:
        return value
    residual = residuals.pop()
    if value.shape != residual.shape:
        raise ValueError("Hy-Nomas residual and target tensors must have matching shapes.")
    target = value.reshape(-1, value.shape[-1])
    source = residual.reshape(-1, residual.shape[-1])

    def selected(row=None):
        coefficient = gate if row is None or gate.numel() == 1 else gate[row]
        return torch.tanh(coefficient) if tanh else coefficient

    if segments is None:
        target.add_(source * selected().to(source))
    else:
        for start, stop, row in segments:
            target[start:stop].add_(source[start:stop] * selected(row).to(source))
    return value


def mlp(inputs, gate_proj, down_proj, activation="gelu", up=None, residual=None, gate=None):
    if not inputs:
        return residual
    value = inputs.pop()
    if activation == "gelu":
        hidden = F.gelu(gate_proj(value), approximate="tanh")
    elif activation == "swiglu":
        if up is None:
            raise ValueError("Hy-Nomas SwiGLU needs both gate and up projections.")
        hidden = F.silu(gate_proj(value)) * up(value)
    else:
        raise ValueError(f"Unsupported Hy-Nomas MLP activation: {activation!r}.")
    update = down_proj(hidden)
    if residual is None:
        return update
    if residual.shape != update.shape:
        raise ValueError("Hy-Nomas MLP update and residual shapes do not match.")
    if gate is None:
        residual.add_(update)
    else:
        residual.add_(update * gate.to(update))
    return residual
