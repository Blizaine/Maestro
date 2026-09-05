# SPDX-License-Identifier: MIT
# Copyright (c) 2026 PlagueKind
"""Block selection for MiniMax H3 SLA attention.

Adapted from ComfyUI-PlagueKind-Nodes-only-sparse (MIT), which in turn
adapts LightX2V's Apache-2.0 BLHD SLA utilities. See
``THIRD_PARTY_NOTICES.md`` for pinned sources and license information.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_kernel(
    source,
    pooled,
    length: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    block_length: tl.constexpr,
):
    block_index = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    batch_index = batch_head_index // heads
    head_index = batch_head_index - batch_index * heads

    offsets_length = block_index * block_length + tl.arange(0, block_length)
    offsets_width = tl.arange(0, width)
    source_offset = batch_index * length * heads * width + head_index * width
    pooled_offset = batch_head_index * ((length + block_length - 1) // block_length) * width
    values = tl.load(
        source
        + source_offset
        + offsets_length[:, None] * (heads * width)
        + offsets_width[None, :],
        mask=offsets_length[:, None] < length,
        other=0.0,
    )
    count = min(block_length, length - block_index * block_length)
    mean = tl.sum(values, axis=0, dtype=tl.float32) / count
    tl.store(
        pooled + pooled_offset + block_index * width + offsets_width,
        mean.to(pooled.dtype.element_ty),
    )


def mean_pool(source: torch.Tensor, block_length: int) -> torch.Tensor:
    """Pool contiguous ``(B, L, H, D)`` rows into FP32 sequence blocks."""

    if not source.is_contiguous():
        raise ValueError("SLA mean-pool input must be contiguous BLHD data.")
    batch, length, heads, width = source.shape
    block_count = (length + block_length - 1) // block_length
    pooled = torch.empty(
        (batch, heads, block_count, width),
        device=source.device,
        dtype=torch.float32,
    )
    _compress_kernel[(block_count, batch * heads)](
        source,
        pooled,
        length,
        heads,
        width,
        block_length,
    )
    return pooled


def get_block_map(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_ratio: float,
    block_query: int = 64,
    block_key: int = 64,
    protect_upto: int = 0,
) -> tuple[torch.Tensor, int]:
    """Select key blocks for every query block and pin H3's media prefix."""

    pooled_query = mean_pool(query, block_query)
    key_mean = key.mean(dim=1, dtype=torch.float32)
    pooled_key = mean_pool(key, block_key).sub_(key_mean[:, :, None, :])

    query_heads, key_heads = pooled_query.shape[1], pooled_key.shape[1]
    if query_heads != key_heads:
        if query_heads % key_heads:
            raise ValueError("SLA requires query heads divisible by key/value heads.")
        pooled_key = pooled_key.repeat_interleave(
            query_heads // key_heads,
            dim=1,
        )

    scores = pooled_query @ pooled_key.transpose(-1, -2)
    key_blocks = scores.shape[-1]
    topk = max(1, min(key_blocks, int(float(topk_ratio) * key_blocks)))
    pinned = min(
        (max(0, int(protect_upto)) + block_key - 1) // block_key,
        key_blocks,
    )
    if pinned:
        scores[..., :pinned] = float("inf")
        # Prefix blocks are additive: voice/audio/reference fidelity must not
        # displace the visual blocks selected by the requested sparsity.
        topk = min(key_blocks, topk + pinned)
    lookup = torch.topk(scores, topk, dim=-1, sorted=False).indices
    return lookup.to(torch.int32).contiguous(), topk


__all__ = ["get_block_map", "mean_pool"]
