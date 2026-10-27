# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Attention adapters for Fibo's cross and masked paths."""

from __future__ import annotations

import torch

from shared.attention import pay_attention


def _pay_attention_mask(mask, *, batch, query_length, key_length, query_heads,
                        layout="standard"):
    """Translate standard masks to Maestro pay_attention's SDPA layout."""
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        raise TypeError("Fibo attention masks must be tensors.")
    if mask.ndim == 2:  # [batch, key]
        if mask.shape[-1] != key_length:
            raise ValueError("Fibo attention mask key length does not match the projected keys.")
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:  # [batch, query, key]
        if mask.shape[-1] != key_length or mask.shape[-2] not in (1, query_length):
            raise ValueError("Fibo attention mask query/key dimensions do not match Q/K.")
        mask = mask.unsqueeze(2)
    elif mask.ndim == 4:  # [batch, heads, query, key]
        if mask.shape[-1] != key_length:
            raise ValueError("Fibo attention mask key length does not match the projected keys.")
        if layout == "maestro":  # [batch, query, heads, key], as emitted by local Qwen3-VL
            if mask.shape[1] not in (1, query_length) or mask.shape[2] not in (1, query_heads):
                raise ValueError("Fibo Maestro-layout attention mask dimensions do not match Q/K.")
        elif layout == "standard":
            if mask.shape[-2] not in (1, query_length) or mask.shape[1] not in (1, query_heads):
                raise ValueError("Fibo attention mask head/query/key dimensions do not match Q/K.")
            mask = mask.transpose(1, 2)
        else:
            raise ValueError(f"Unsupported Fibo attention-mask layout: {layout!r}.")
    else:
        raise ValueError("Fibo attention masks must have 2, 3, or 4 dimensions.")
    if mask.shape[0] not in (1, batch):
        raise ValueError("Fibo attention mask batch must match the projected queries or be one.")
    if not (mask.dtype == torch.bool or torch.is_floating_point(mask)):
        # Tokenizer attention masks use integer 0/1 values. PyTorch SDPA's
        # boolean convention is True = allowed, matching those inputs.
        mask = mask != 0
    return mask


def attend_projected(query, key, value, *, attention_mask=None, causal=False,
                     force_attention="sdpa", attention_mask_layout="standard"):
    """Attend over projected Fibo Q/K/V tensors in [batch, tokens, heads, dim].

    Key/value batches and GQA head counts may be smaller than the query side;
    both are expanded in the same way as standard grouped-query attention.
    The sequence dimensions may differ for cross-attention.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Fibo Q/K/V tensors must have [batch, tokens, heads, dim] shape.")
    if (query.shape[-1] != key.shape[-1] or key.shape[:2] != value.shape[:2]
            or key.shape[2] != value.shape[2] or key.shape[-1] != value.shape[-1]):
        raise ValueError("Fibo Q/K/V head dimensions or key/value sequences do not match.")
    if query.shape[0] != key.shape[0]:
        if key.shape[0] != 1:
            raise ValueError("Fibo key/value batch must match the query batch or be one.")
        key = key.expand(query.shape[0], -1, -1, -1)
        value = value.expand(query.shape[0], -1, -1, -1)
    query_heads, kv_heads = query.shape[2], key.shape[2]
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("Fibo query head count must be divisible by the key/value head count.")
    if kv_heads != query_heads:
        groups = query_heads // kv_heads
        key = key[:, :, :, None, :].expand(-1, -1, -1, groups, -1).reshape(
            key.shape[0], key.shape[1], query_heads, key.shape[-1]
        )
        value = value[:, :, :, None, :].expand(-1, -1, -1, groups, -1).reshape(
            value.shape[0], value.shape[1], query_heads, value.shape[-1]
        )
    mask = _pay_attention_mask(
        attention_mask, batch=query.shape[0], query_length=query.shape[1],
        key_length=key.shape[1], query_heads=query_heads,
        layout=attention_mask_layout,
    )
    return pay_attention(
        [query, key, value],
        attention_mask=mask,
        causal=bool(causal),
        force_attention=force_attention,
    )
