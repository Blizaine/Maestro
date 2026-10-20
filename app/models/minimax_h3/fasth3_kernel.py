# SPDX-License-Identifier: Apache-2.0
"""Triton kernel for FastH3 tile-routed attention.

The routing policy follows the Apache-2.0 FastVideo MiniMax-H3 VSA source at
revision ``d4f144581d6882d5bceb288ca13d767c054ca041``.  This kernel is a local
implementation; it is imported lazily by :mod:`fasth3_attention` after CUDA
preflight.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pool_qkv_kernel(
    query,
    key,
    value,
    tile_rows,
    valid_counts,
    pooled_query,
    pooled_key,
    pooled_value,
    sequence_length: tl.constexpr,
    features: tl.constexpr,
    num_tiles: tl.constexpr,
    block_tokens: tl.constexpr,
    block_features: tl.constexpr,
):
    feature_block = tl.program_id(0)
    tile_index = tl.program_id(1)
    batch_index = tl.program_id(2)
    rows = tl.arange(0, block_tokens)
    features_in_block = feature_block * block_features + tl.arange(0, block_features)
    token_rows = tl.load(
        tile_rows + tile_index * block_tokens + rows,
        mask=rows < block_tokens,
        other=-1,
    )
    token_mask = token_rows >= 0
    feature_mask = features_in_block < features
    input_offsets = (
        batch_index * sequence_length * features
        + token_rows[:, None] * features
        + features_in_block[None, :]
    )
    mask = token_mask[:, None] & feature_mask[None, :]
    q = tl.load(query + input_offsets, mask=mask, other=0.0)
    k = tl.load(key + input_offsets, mask=mask, other=0.0)
    v = tl.load(value + input_offsets, mask=mask, other=0.0)
    count = tl.load(valid_counts + tile_index).to(tl.float32)
    output_offset = (batch_index * num_tiles + tile_index) * features + features_in_block
    output_mask = feature_mask
    tl.store(
        pooled_query + output_offset,
        tl.sum(q, axis=0, dtype=tl.float32) / count,
        mask=output_mask,
    )
    tl.store(
        pooled_key + output_offset,
        tl.sum(k, axis=0, dtype=tl.float32) / count,
        mask=output_mask,
    )
    tl.store(
        pooled_value + output_offset,
        tl.sum(v, axis=0, dtype=tl.float32) / count,
        mask=output_mask,
    )


@triton.jit
def _fast_h3_attention_kernel(
    query,
    key,
    value,
    gate,
    compression,
    tile_rows,
    routes,
    output,
    sequence_length: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_tiles: tl.constexpr,
    num_prefix_tiles: tl.constexpr,
    num_video_tiles: tl.constexpr,
    topk: tl.constexpr,
    has_gate: tl.constexpr,
    query_key_scale: tl.constexpr,
    block_tokens: tl.constexpr,
    block_dim: tl.constexpr,
):
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_index = batch_head // heads
    head_index = batch_head % heads
    sequence_stride: tl.constexpr = heads * head_dim

    query_tile_rows = tl.arange(0, block_tokens)
    dimensions = tl.arange(0, block_dim)
    query_rows = tl.load(
        tile_rows + query_tile * block_tokens + query_tile_rows,
        mask=query_tile_rows < block_tokens,
        other=-1,
    )
    query_mask = query_rows >= 0
    base = batch_index * sequence_length * sequence_stride + head_index * head_dim
    query_ptrs = query + base + query_rows[:, None] * sequence_stride + dimensions[None, :]
    output_ptrs = output + base + query_rows[:, None] * sequence_stride + dimensions[None, :]
    q = tl.load(
        query_ptrs,
        mask=query_mask[:, None] & (dimensions[None, :] < head_dim),
        other=0.0,
    )

    row_max = tl.full((block_tokens,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((block_tokens,), dtype=tl.float32)
    accumulator = tl.zeros((block_tokens, block_dim), dtype=tl.float32)
    key_count = tl.where(
        query_tile < num_prefix_tiles,
        num_tiles,
        num_prefix_tiles + topk,
    )

    for selected_index in tl.range(0, key_count):
        if query_tile < num_prefix_tiles or selected_index < num_prefix_tiles:
            key_tile = selected_index
        else:
            route_index = (
                (batch_head * num_video_tiles + (query_tile - num_prefix_tiles)) * topk
                + (selected_index - num_prefix_tiles)
            )
            key_tile = tl.load(routes + route_index).to(tl.int32)

        key_tile_rows = tl.arange(0, block_tokens)
        key_rows = tl.load(
            tile_rows + key_tile * block_tokens + key_tile_rows,
            mask=key_tile_rows < block_tokens,
            other=-1,
        )
        key_mask = key_rows >= 0
        key_ptrs = (
            key
            + base
            + key_rows[None, :] * sequence_stride
            + dimensions[:, None]
        )
        value_ptrs = (
            value
            + base
            + key_rows[:, None] * sequence_stride
            + dimensions[None, :]
        )
        k = tl.load(
            key_ptrs,
            mask=key_mask[None, :] & (dimensions[:, None] < head_dim),
            other=0.0,
        )
        v = tl.load(
            value_ptrs,
            mask=key_mask[:, None] & (dimensions[None, :] < head_dim),
            other=0.0,
        )
        logits = tl.dot(q, k) * (query_key_scale * 1.4426950408889634)
        logits = tl.where(key_mask[None, :], logits, -float("inf"))
        # Invalid rows are not stored. Give them finite logits so the online
        # softmax does not form -inf - -inf for the tile padding.
        logits = tl.where(query_mask[:, None], logits, 0.0)
        local_max = tl.max(logits, axis=1)
        new_max = tl.maximum(row_max, local_max)
        probabilities = tl.math.exp2(logits - new_max[:, None])
        local_sum = tl.sum(probabilities, axis=1)
        correction = tl.math.exp2(row_max - new_max)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(v.dtype), v)
        row_sum = row_sum * correction + local_sum
        row_max = new_max

    attended = accumulator / row_sum[:, None]
    if has_gate:
        compression_offset = (
            (batch_index * num_tiles + query_tile) * sequence_stride
            + head_index * head_dim
        )
        compression_ptrs = compression + compression_offset + dimensions
        compressed = tl.load(compression_ptrs, mask=dimensions < head_dim, other=0.0)
        gate_ptrs = gate + base + query_rows[:, None] * sequence_stride + dimensions[None, :]
        gate_values = tl.load(
            gate_ptrs,
            mask=query_mask[:, None] & (dimensions[None, :] < head_dim),
            other=0.0,
        )
        # Match the trained merge boundary: round fine attention and pooled
        # compression to the activation dtype, then apply the raw gate with
        # addcmul semantics. FastVideo passes the projection through directly;
        # this is intentionally not tanh(gate).
        fine = attended.to(query.type.element_ty).to(tl.float32)
        compressed = compressed.to(query.type.element_ty).to(tl.float32)
        attended = fine + gate_values.to(tl.float32) * compressed[None, :]

    tl.store(
        output_ptrs,
        attended.to(output.type.element_ty),
        mask=query_mask[:, None] & (dimensions[None, :] < head_dim),
    )


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor | None,
    compression: torch.Tensor | None,
    tile_rows: torch.Tensor,
    routes: torch.Tensor,
    *,
    num_prefix_tiles: int,
    num_video_tiles: int,
    topk: int,
) -> torch.Tensor:
    """Run tile-routed attention on contiguous ``(B, L, H, D)`` tensors."""

    if not all(tensor.is_contiguous() for tensor in (query, key, value, tile_rows, routes)):
        raise ValueError("FastH3 VSA kernel inputs must be contiguous")
    if gate is not None and (not gate.is_contiguous() or compression is None or not compression.is_contiguous()):
        raise ValueError("FastH3 gate/compression tensors must be contiguous")
    if gate is None and compression is not None:
        raise ValueError("FastH3 compression must be absent when no gate is supplied")
    batch, sequence_length, heads, head_dim = query.shape
    if head_dim != 128:
        raise ValueError(f"FastH3 VSA Triton kernel requires 128-wide heads, got {head_dim}")
    num_tiles = int(tile_rows.shape[0])
    if tile_rows.shape[1] != 64:
        raise ValueError("FastH3 VSA tiles must have 64 rows")
    if num_prefix_tiles <= 0 or num_video_tiles <= 0 or topk <= 0:
        raise ValueError("FastH3 VSA requires non-empty prefix/video tiles and a positive top-k")
    output = torch.empty_like(query)
    _fast_h3_attention_kernel[(num_tiles, batch * heads)](
        query,
        key,
        value,
        gate if gate is not None else query,
        compression if compression is not None else query,
        tile_rows,
        routes,
        output,
        sequence_length,
        heads,
        head_dim,
        num_tiles,
        num_prefix_tiles,
        num_video_tiles,
        topk,
        gate is not None,
        head_dim ** -0.5,
        64,
        128,
        num_warps=4,
        num_stages=2,
    )
    return output


def pool_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tile_rows: torch.Tensor,
    valid_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-pool each 64-row tile in one bounded-memory Triton launch."""

    if not all(tensor.is_contiguous() for tensor in (query, key, value, tile_rows, valid_counts)):
        raise ValueError("FastH3 VSA pooling inputs must be contiguous")
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("FastH3 VSA pooling requires matching BLHD Q/K/V tensors")
    batch, sequence_length, heads, head_dim = query.shape
    num_tiles = int(tile_rows.shape[0])
    if tile_rows.shape[1] != 64 or valid_counts.numel() != num_tiles:
        raise ValueError("FastH3 VSA pooling metadata has invalid tile dimensions")
    features = heads * head_dim
    pooled = tuple(
        torch.empty((batch, num_tiles, heads, head_dim), dtype=torch.float32, device=query.device)
        for _ in range(3)
    )
    _pool_qkv_kernel[(triton.cdiv(features, 128), num_tiles, batch)](
        query,
        key,
        value,
        tile_rows,
        valid_counts,
        *pooled,
        sequence_length,
        features,
        num_tiles,
        64,
        128,
        num_warps=4,
    )
    return pooled


__all__ = ["block_sparse_attention", "pool_qkv"]
