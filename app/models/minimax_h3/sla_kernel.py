# SPDX-License-Identifier: MIT
# Copyright (c) 2026 PlagueKind
"""Forward-only BLHD block-sparse attention kernel for MiniMax H3.

Adapted from ComfyUI-PlagueKind-Nodes-only-sparse (MIT) and LightX2V
(Apache-2.0). The launch ladder includes low-shared-memory configurations for
consumer GPUs and is always guarded by a dense fallback in ``sla_attention``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_forward(
    query,
    key,
    value,
    query_key_scale: tl.constexpr,
    topk: tl.constexpr,
    lookup,
    output,
    heads: tl.constexpr,
    query_length: tl.constexpr,
    key_length: tl.constexpr,
    query_blocks: tl.constexpr,
    width: tl.constexpr,
    block_query: tl.constexpr,
    block_key: tl.constexpr,
):
    query_block_index = tl.program_id(0).to(tl.int64)
    batch_head_index = tl.program_id(1).to(tl.int64)
    batch_index = batch_head_index // heads
    head_index = batch_head_index % heads
    head_stride: tl.constexpr = heads * width

    query_offset = batch_index * query_length * head_stride + head_index * width
    key_value_offset = batch_index * key_length * head_stride + head_index * width
    lookup_offset = (batch_head_index * query_blocks + query_block_index) * topk
    query_rows = query_block_index * block_query + tl.arange(0, block_query)
    key_rows = tl.arange(0, block_key)
    width_offsets = tl.arange(0, width)
    query_ptrs = query + query_offset + query_rows[:, None] * head_stride + width_offsets[None, :]
    output_ptrs = output + query_offset + query_rows[:, None] * head_stride + width_offsets[None, :]
    lookup_ptr = lookup + lookup_offset

    row_max = tl.full([block_query], -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros([block_query], dtype=tl.float32)
    accumulator = tl.zeros([block_query, width], dtype=tl.float32)
    q = tl.load(
        query_ptrs,
        mask=query_rows[:, None] < query_length,
        other=0.0,
    )
    for selected_index in tl.range(topk):
        key_block_index = tl.load(lookup_ptr + selected_index).to(tl.int64)
        key_start = key_block_index * block_key
        key_mask = (key_start + key_rows) < key_length
        key_ptrs = key + key_value_offset + (key_start + key_rows)[None, :] * head_stride + width_offsets[:, None]
        value_ptrs = value + key_value_offset + (key_start + key_rows)[:, None] * head_stride + width_offsets[None, :]
        k = tl.load(key_ptrs, mask=key_mask[None, :], other=0.0)
        logits = tl.dot(q, k) * (query_key_scale * 1.4426950408889634)
        logits = tl.where(key_mask[None, :], logits, float("-inf"))
        v = tl.load(value_ptrs, mask=key_mask[:, None], other=0.0)
        local_max = tl.max(logits, 1)
        new_max = tl.maximum(row_max, local_max)
        logits = logits - new_max[:, None]
        probabilities = tl.math.exp2(logits)
        local_sum = tl.sum(probabilities, 1)
        correction = tl.math.exp2(row_max - new_max)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(v.dtype), v)
        row_sum = row_sum * correction + local_sum
        row_max = new_max

    accumulator = accumulator / row_sum[:, None]
    tl.store(
        output_ptrs,
        accumulator.to(output.type.element_ty),
        mask=query_rows[:, None] < query_length,
    )


_LAUNCH_LADDER = {
    (128, 64): ((8, 3), (4, 3), (8, 2), (4, 1)),
    (128, 128): ((8, 2), (4, 2), (8, 1), (4, 1)),
    (64, 128): ((4, 2), (8, 2), (4, 1)),
    (64, 64): ((4, 1), (4, 3), (8, 3), (8, 1)),
}
_CHOSEN_LAUNCH: dict[tuple[int, int, int], tuple[int, int]] = {}


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lookup: torch.Tensor,
    topk: int,
    block_query: int,
    block_key: int,
    query_key_scale: float | None = None,
) -> torch.Tensor:
    """Run block-sparse attention on contiguous ``(B, L, H, D)`` tensors."""

    if not all(item.is_contiguous() for item in (query, key, value, lookup)):
        raise ValueError("SLA inputs and lookup table must be contiguous.")
    if block_query not in (64, 128) or block_key not in (64, 128):
        raise ValueError("SLA block sizes must be 64 or 128.")
    batch, query_length, heads, width = query.shape
    key_length = key.shape[1]
    scale = width ** -0.5 if query_key_scale is None else query_key_scale
    query_blocks = triton.cdiv(query_length, block_query)
    output = torch.empty_like(query)
    grid = (query_blocks, batch * heads)
    cache_key = (block_query, block_key, width)
    ladder = (
        (_CHOSEN_LAUNCH[cache_key],)
        if cache_key in _CHOSEN_LAUNCH
        else _LAUNCH_LADDER[(block_query, block_key)]
    )
    last_error: Exception | None = None
    for warps, stages in ladder:
        try:
            _attention_forward[grid](
                query,
                key,
                value,
                scale,
                int(topk),
                lookup,
                output,
                heads,
                query_length,
                key_length,
                query_blocks,
                width,
                block_query,
                block_key,
                num_warps=warps,
                num_stages=stages,
            )
        except Exception as error:  # try a smaller shared-memory launch
            last_error = error
            continue
        _CHOSEN_LAUNCH[cache_key] = (warps, stages)
        return output
    raise last_error or RuntimeError("SLA found no viable Triton launch configuration.")


__all__ = ["block_sparse_attention"]
