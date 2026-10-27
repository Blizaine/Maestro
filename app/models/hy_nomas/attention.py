# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Model-local dense attention and Q8 history adapters.

MMGP 4.0 does not expose the newer ``joint_qkv_attention`` or ``PrefixCache``
helpers used by the upstream port. This module implements the same tensor
contract with Maestro's stable ``pay_attention`` dispatcher and moves only one
block's CPU history to the active device at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from shared.attention import pay_attention

Q8_GROUP_SIZE = 32


def broadcast_singleton_values(values, query_count):
    """Broadcast the output of a one-key camera attention group over its queries.

    Under the pinned inference profile, each image-frame query group sees
    exactly one aligned keyboard token. Its softmax weight is therefore 1,
    independent of Q, K, Q/K normalization, or RoPE. The camera module can
    project V once and broadcast the projected result without materializing
    query/key attention scores.

    ``values`` has one feature vector per leading item (normally one per
    latent frame); ``query_count`` is the number of spatial image tokens in
    each such group.
    """
    if not torch.is_tensor(values) or values.ndim < 2:
        raise ValueError("Hy-Nomas singleton camera values must have a feature dimension.")
    if not isinstance(query_count, int) or isinstance(query_count, bool) or query_count <= 0:
        raise ValueError("Hy-Nomas singleton camera query count must be a positive integer.")
    return values.unsqueeze(-2).expand(*values.shape[:-1], query_count, values.shape[-1])


def quantize_heads(tensor, group_size=Q8_GROUP_SIZE):
    """Symmetric per-token/head Q8 with one FP16 absmax scale per 32 channels."""
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] % group_size:
        raise ValueError("Hy-Nomas Q8 history expects [1, tokens, heads, dim] with 32-wide groups.")
    values = tensor[0].float().reshape(tensor.shape[1], tensor.shape[2], -1, group_size)
    scales = values.abs().amax(dim=-1, keepdim=True).div_(127.0)
    safe_scales = scales.clamp_min_(torch.finfo(torch.float32).tiny)
    quantized = torch.round(values / safe_scales).clamp_(-127, 127).to(torch.int8)
    quantized = torch.where(scales == 0, torch.zeros_like(quantized), quantized)
    return quantized.reshape(tensor.shape[1:]).contiguous(), scales.squeeze(-1).to(torch.float16).contiguous()


def _dequantize(data, scales, *, device, dtype):
    if data.ndim != 3 or scales.shape != (*data.shape[:-1], data.shape[-1] // Q8_GROUP_SIZE):
        raise ValueError("Hy-Nomas Q8 prefix data and scale shapes do not match.")
    grouped = data.to(device=device, dtype=torch.float32).reshape(*data.shape[:-1], -1, Q8_GROUP_SIZE)
    scale = scales.to(device=device, dtype=torch.float32).unsqueeze(-1)
    return (grouped * scale).reshape(1, data.shape[0], data.shape[1], data.shape[2]).to(dtype=dtype)


@dataclass(frozen=True)
class PrefixCache:
    keys: torch.Tensor
    key_scales: torch.Tensor
    values: torch.Tensor
    value_scales: torch.Tensor

    @property
    def tokens(self):
        return int(self.keys.shape[0])

    def to(self, *, device, dtype):
        return (
            _dequantize(self.keys, self.key_scales, device=device, dtype=dtype),
            _dequantize(self.values, self.value_scales, device=device, dtype=dtype),
        )


def q8_prefix(keys, key_scales, values, value_scales):
    return PrefixCache(keys, key_scales, values, value_scales)


class _HeadGroup(NamedTuple):
    q: range
    kv: range
    mean_squares: object = None


def _expand_kv_heads(query, key, value):
    if key.shape[0] == 1 and query.shape[0] != 1:
        key = key.expand(query.shape[0], -1, -1, -1)
        value = value.expand(query.shape[0], -1, -1, -1)
    if key.shape[0] != query.shape[0]:
        raise ValueError("Hy-Nomas attention key/value batch does not match the query batch.")
    q_heads, kv_heads = query.shape[2], key.shape[2]
    if kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("Hy-Nomas query heads must be divisible by key/value heads.")
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        key = key.unsqueeze(3).expand(-1, -1, -1, repeats, -1).reshape(*key.shape[:2], q_heads, key.shape[-1])
        value = value.unsqueeze(3).expand(-1, -1, -1, repeats, -1).reshape(*value.shape[:2], q_heads, value.shape[-1])
    return key, value


def attend_projected(query, key, value, *, attention_mask=None, causal=False, force_attention=None):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Hy-Nomas Q/K/V must have [batch, tokens, heads, dim] shape.")
    if key.shape[:2] != value.shape[:2] or key.shape[2:] != value.shape[2:] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Hy-Nomas key/value or query/key dimensions do not match.")
    key, value = _expand_kv_heads(query, key, value)
    return pay_attention([query, key, value], attention_mask=attention_mask, causal=bool(causal), force_attention=force_attention)


def kv_first_attention(qkv_list, force_attention=None):
    # The current dispatcher handles the complete Q/K/V tuple. The optional
    # force_attention parameter is retained for source compatibility.
    if len(qkv_list) != 3:
        raise ValueError("Hy-Nomas attention expects a Q/K/V tuple.")
    query, key, value = qkv_list
    key, value = _expand_kv_heads(query, key, value)
    return pay_attention([query, key, value], force_attention=force_attention)


def joint_qkv_attention(streams, heads, head_dim, norm_rope, kv_prefix=None, norm_v=None, attention_fn=None):
    """Project two streams, optionally prepend history, and dispatch attention.

    ``streams`` follows the upstream ``[(input_list, (q, k, v))]`` contract;
    input tensors are consumed. Outputs concatenate image then text tokens.
    """
    queries, keys, values = [], [], []
    batch = None
    for inputs, projections in streams:
        if not inputs:
            raise ValueError("Hy-Nomas joint attention received an empty input stream.")
        source = inputs.pop()
        if source.ndim == 2:
            source = source.unsqueeze(0)
        if source.ndim != 3:
            raise ValueError("Hy-Nomas joint attention inputs must be [tokens, channels] or [batch, tokens, channels].")
        batch = source.shape[0] if batch is None else batch
        if source.shape[0] != batch:
            raise ValueError("Hy-Nomas image and text streams must have matching batches.")
        q_proj, k_proj, v_proj = projections
        shape = (*source.shape[:2], heads, head_dim)
        queries.append(q_proj(source).reshape(shape))
        keys.append(k_proj(source).reshape(shape))
        values.append(v_proj(source).reshape(shape))
    query = torch.cat(queries, dim=1)
    key = torch.cat(keys, dim=1)
    value = torch.cat(values, dim=1)
    del queries, keys, values
    group = _HeadGroup(range(heads), range(heads))
    norm_rope(query, key, group)
    if norm_v is not None:
        norm_v(value, group)

    if kv_prefix is not None:
        if not isinstance(kv_prefix, PrefixCache):
            raise TypeError("Hy-Nomas history prefix must be a model-local Q8 PrefixCache.")
        prefix_key, prefix_value = kv_prefix.to(device=query.device, dtype=query.dtype)
        key = torch.cat((prefix_key, key), dim=1)
        value = torch.cat((prefix_value, value), dim=1)

    if attention_fn is None:
        return attend_projected(query, key, value)
    qkv = [query, key, value]
    try:
        return attention_fn(qkv, group)
    finally:
        # A custom masked attention consumes and clears the list. Avoid keeping
        # another reference to the large working tensors after it returns.
        qkv.clear()
