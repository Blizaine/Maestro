# SPDX-License-Identifier: Apache-2.0
"""Native FastH3 VSA attention for packed text-audio-video H3 sequences.

The routing and gate semantics follow the Apache-2.0 FastVideo MiniMax-H3 VSA
implementation at https://github.com/hao-ai-lab/FastVideo (pinned source
revision d4f144581d6882d5bceb288ca13d767c054ca041).  The Triton kernel in
``fasth3_kernel`` is a local implementation; this module intentionally imports
it only after CUDA preflight so CPU-only imports remain safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from collections.abc import Mapping, Sequence
from typing import Any

import torch

TILE_SHAPE = (4, 4, 4)
TILE_SIZE = math.prod(TILE_SHAPE)
DEFAULT_SPARSITY = 0.80
OMNI_REF_TILE_SHAPE = (4, 4, 8)
OMNI_REF_TILE_SIZE = math.prod(OMNI_REF_TILE_SHAPE)
OMNI_REF_DEFAULT_SPARSITY = 0.90
_SCORE_ELEMENT_BUDGET = 1 << 20


@dataclass(frozen=True)
class FastH3Geometry:
    """Reusable packed row-to-tile mapping for one H3 latent geometry."""

    sequence_length: int
    prefix_segments: tuple[int, ...]
    latent_grid: tuple[int, int, int]
    num_prefix_tiles: int
    num_video_tiles: int
    topk: int
    # CPU-side, immutable-by-convention metadata.  Padding is represented by -1.
    tile_rows: torch.Tensor
    row_to_tile: torch.Tensor
    valid_counts: torch.Tensor
    tile_size: int
    # Sparse video spans are tile-index ranges.  OmniRef has one range per
    # reference video followed by the generated target; T2AV has only target.
    video_tile_spans: tuple[tuple[int, int], ...]
    span_topks: tuple[int, ...]

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles

    @property
    def route_count(self) -> int:
        """Number of sparse keys selected for each video query tile."""

        return sum(self.span_topks)


def compute_topk(sparsity: float, num_video_tiles: int) -> int:
    """Return FastVideo's per-query video-tile budget, rounded upward."""

    if isinstance(num_video_tiles, bool) or not isinstance(num_video_tiles, Integral):
        raise TypeError("num_video_tiles must be an integer")
    count = int(num_video_tiles)
    if count <= 0:
        return 0
    value = _normalize_sparsity(sparsity)
    return max(1, min(math.ceil((1.0 - value) * count), count))


def _normalize_sparsity(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("FastH3 sparsity must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"FastH3 sparsity must be between 0 and 1, got {value!r}")
    return normalized


def _shape_tuple(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a three-integer sequence")
    try:
        parts = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a three-integer sequence") from error
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three dimensions")
    result = []
    for part in parts:
        if isinstance(part, bool) or not isinstance(part, Integral):
            raise TypeError(f"{name} dimensions must be integers")
        if int(part) <= 0:
            raise ValueError(f"{name} dimensions must be positive")
        result.append(int(part))
    return tuple(result)


def _count_rows(layout: Any, field: str) -> int:
    rows = getattr(layout, field, None)
    if not torch.is_tensor(rows) or rows.ndim != 1:
        raise TypeError(f"T2AV layout must provide a one-dimensional {field} tensor")
    return int(rows.numel())


def _count_value(value: Any, name: str, *, allow_zero: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    count = int(value)
    if count < 0 or (not allow_zero and count == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return count


def _reference_value(reference: Any, field: str, default: Any = None) -> Any:
    if isinstance(reference, Mapping):
        return reference.get(field, default)
    return getattr(reference, field, default)


def _require_indices(layout: Any, field: str, expected: list[int]) -> None:
    rows = getattr(layout, field, None)
    if not torch.is_tensor(rows) or rows.ndim != 1:
        raise TypeError(f"OmniRef layout must provide a one-dimensional {field} tensor")
    expected_rows = torch.tensor(expected, dtype=rows.dtype, device=rows.device)
    if not torch.equal(rows, expected_rows):
        raise ValueError(f"OmniRef {field} do not match the prepared reference packing order")


def _append_dense_tiles(
    tile_rows: list[list[int]],
    valid_counts: list[int],
    start: int,
    count: int,
    tile_size: int,
) -> None:
    for offset in range(0, count, tile_size):
        rows = list(range(start + offset, start + min(offset + tile_size, count)))
        valid_counts.append(len(rows))
        tile_rows.append(rows + [-1] * (tile_size - len(rows)))


def _append_video_tiles(
    tile_rows: list[list[int]],
    valid_counts: list[int],
    start: int,
    grid: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    tile_size: int,
) -> None:
    grid_t, grid_h, grid_w = grid
    tile_t, tile_h, tile_w = tile_shape
    for t0 in range(0, grid_t, tile_t):
        for h0 in range(0, grid_h, tile_h):
            for w0 in range(0, grid_w, tile_w):
                rows = [
                    start + (t * grid_h + h) * grid_w + w
                    for t in range(t0, min(t0 + tile_t, grid_t))
                    for h in range(h0, min(h0 + tile_h, grid_h))
                    for w in range(w0, min(w0 + tile_w, grid_w))
                ]
                if not rows:
                    raise AssertionError("FastH3 VSA constructed an empty video tile")
                valid_counts.append(len(rows))
                tile_rows.append(rows + [-1] * (tile_size - len(rows)))


def _geometry_from_tiles(
    *,
    sequence_length: int,
    prefix_segments: tuple[int, ...],
    latent_grid: tuple[int, int, int],
    num_prefix_tiles: int,
    tile_rows: list[list[int]],
    valid_counts: list[int],
    sparsity: float,
    tile_size: int,
    video_tile_spans: tuple[tuple[int, int], ...],
) -> FastH3Geometry:
    if not video_tile_spans:
        raise ValueError("FastH3 VSA requires at least one sparse video region")
    num_video_tiles = len(tile_rows) - num_prefix_tiles
    if num_video_tiles <= 0:
        raise ValueError("FastH3 VSA requires at least one target-video tile")
    if sum(valid_counts) != sequence_length:
        raise AssertionError("FastH3 VSA tile geometry does not cover the packed sequence exactly")

    rows_tensor = torch.tensor(tile_rows, dtype=torch.int32)
    flattened = rows_tensor.flatten()
    valid = flattened >= 0
    packed_rows = flattened[valid].long()
    if packed_rows.numel() != sequence_length or torch.unique(packed_rows).numel() != sequence_length:
        raise AssertionError("FastH3 VSA tile geometry duplicates or omits packed rows")
    if int(packed_rows.min()) != 0 or int(packed_rows.max()) != sequence_length - 1:
        raise AssertionError("FastH3 VSA tile geometry contains out-of-range rows")
    tile_indices = torch.arange(len(tile_rows), dtype=torch.int32).repeat_interleave(tile_size)
    row_to_tile = torch.empty((sequence_length,), dtype=torch.int32)
    row_to_tile[packed_rows] = tile_indices[valid]

    span_topks = tuple(compute_topk(sparsity, end - start) for start, end in video_tile_spans)
    return FastH3Geometry(
        sequence_length=sequence_length,
        prefix_segments=prefix_segments,
        latent_grid=latent_grid,
        num_prefix_tiles=num_prefix_tiles,
        num_video_tiles=num_video_tiles,
        # Keep the historical target-region meaning for callers that inspect
        # ``topk``; the full per-forward route budget is ``route_count``.
        topk=span_topks[-1],
        tile_rows=rows_tensor,
        row_to_tile=row_to_tile,
        valid_counts=torch.tensor(valid_counts, dtype=torch.int32),
        tile_size=tile_size,
        video_tile_spans=video_tile_spans,
        span_topks=span_topks,
    )


def _build_omniref_geometry(
    layout: Any,
    latent_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    sparsity: float,
    references: Sequence[Any] | None,
) -> FastH3Geometry:
    """Tile packed Ref2VA media with separate sparse spans for each video."""

    references = () if references is None else references
    if isinstance(references, (str, bytes)) or not isinstance(references, Sequence):
        raise TypeError("OmniRef prepared references must be an ordered sequence")

    target_shape = _shape_tuple(latent_shape, "latent_shape")
    patch = _shape_tuple(patch_size, "patch_size")
    if any(size % patch_axis for size, patch_axis in zip(target_shape, patch)):
        raise ValueError(f"Latent dimensions {target_shape} are not divisible by patch size {patch}")
    target_grid = tuple(size // patch_axis for size, patch_axis in zip(target_shape, patch))
    sequence_length_value = getattr(layout, "sequence_length", None)
    sequence_length = _count_value(sequence_length_value, "OmniRef sequence_length", allow_zero=False)
    text_rows = _count_rows(layout, "text_indices")
    audio_rows = _count_rows(layout, "audio_indices")
    video_rows = _count_rows(layout, "video_indices")
    _require_contiguous_rows(layout, "text_indices", 0, text_rows)

    condition_video_rows = _count_value(
        getattr(layout, "num_condition_video_rows", 0), "num_condition_video_rows"
    )
    condition_audio_rows = _count_value(
        getattr(layout, "num_condition_audio_rows", 0), "num_condition_audio_rows"
    )
    if _count_value(getattr(layout, "num_target_condition_audio_latents", 0), "num_target_condition_audio_latents"):
        raise ValueError("FastH3 OmniRef does not support frozen target-audio rows")
    if _count_value(getattr(layout, "num_target_condition_video_rows", 0), "num_target_condition_video_rows"):
        raise ValueError("FastH3 OmniRef does not support frozen target-video rows")

    # Keep the true packed Ref2VA row order. Images and all audio are dense;
    # each video reference owns one independently routed latent-grid span.
    cursor = text_rows
    expected_audio: list[int] = []
    expected_video: list[int] = []
    dense_segments: list[tuple[int, int]] = [(0, text_rows)]
    reference_video_regions: list[tuple[int, tuple[int, int, int]]] = []
    reference_video_rows = 0
    reference_audio_rows = 0

    for reference_index, reference in enumerate(references):
        kind = str(
            _reference_value(reference, "kind", _reference_value(reference, "media_type", ""))
        ).strip().lower()
        if kind not in {"image", "video", "audio"}:
            raise ValueError(f"OmniRef prepared reference {reference_index} has unsupported kind {kind!r}")
        audio_latents = _count_value(
            _reference_value(reference, "num_audio_latents", 0),
            f"references[{reference_index}].num_audio_latents",
        )
        ref_audio_rows = audio_latents * 2

        if kind == "audio":
            if ref_audio_rows <= 0:
                raise ValueError(f"OmniRef audio reference {reference_index} has no prepared audio latents")
            dense_segments.append((cursor, ref_audio_rows))
            expected_audio.extend(range(cursor, cursor + ref_audio_rows))
            cursor += ref_audio_rows
            reference_audio_rows += ref_audio_rows
            continue

        ref_shape = _shape_tuple(
            (
                _reference_value(reference, "num_latent_frames", 1),
                _reference_value(reference, "latent_height", 0),
                _reference_value(reference, "latent_width", 0),
            ),
            f"references[{reference_index}] latent shape",
        )
        if any(size % patch_axis for size, patch_axis in zip(ref_shape, patch)):
            raise ValueError(
                f"OmniRef reference {reference_index} latent dimensions {ref_shape} are not divisible by patch size {patch}"
            )
        if kind == "image" and ref_shape[0] != 1:
            raise ValueError(f"OmniRef image reference {reference_index} must contain one latent frame")
        ref_grid = tuple(size // patch_axis for size, patch_axis in zip(ref_shape, patch))
        ref_video_rows = math.prod(ref_grid)
        if kind == "image":
            if ref_audio_rows:
                raise ValueError(f"OmniRef image reference {reference_index} cannot carry prepared audio rows")
            dense_segments.append((cursor, ref_video_rows))
            expected_video.extend(range(cursor, cursor + ref_video_rows))
            cursor += ref_video_rows
            reference_video_rows += ref_video_rows
            continue

        if ref_audio_rows:
            dense_segments.append((cursor, ref_audio_rows))
            expected_audio.extend(range(cursor, cursor + ref_audio_rows))
            cursor += ref_audio_rows
            reference_audio_rows += ref_audio_rows
        reference_video_regions.append((cursor, ref_grid))
        expected_video.extend(range(cursor, cursor + ref_video_rows))
        cursor += ref_video_rows
        reference_video_rows += ref_video_rows

    if condition_video_rows != reference_video_rows:
        raise ValueError(
            "FastH3 OmniRef expects reference-only visual conditioning rows; keyframe or other extra video conditions are unsupported."
        )
    if condition_audio_rows != reference_audio_rows:
        raise ValueError(
            "FastH3 OmniRef expects reference-only audio conditioning rows; keyframe or other extra audio conditions are unsupported."
        )

    target_audio_rows = audio_rows - condition_audio_rows
    if target_audio_rows <= 0:
        raise ValueError("FastH3 OmniRef requires a generated target-audio segment")
    dense_segments.append((cursor, target_audio_rows))
    expected_audio.extend(range(cursor, cursor + target_audio_rows))
    cursor += target_audio_rows

    target_video_rows = math.prod(target_grid)
    if video_rows - condition_video_rows != target_video_rows:
        raise ValueError(
            f"OmniRef layout has {video_rows - condition_video_rows} target-video rows, but latent grid {target_grid} has {target_video_rows} patches"
        )
    target_video_start = cursor
    expected_video.extend(range(target_video_start, target_video_start + target_video_rows))
    cursor += target_video_rows
    if cursor != sequence_length:
        raise ValueError(
            f"OmniRef reference and target segments reach row {cursor}, but sequence_length is {sequence_length}"
        )
    _require_indices(layout, "audio_indices", expected_audio)
    _require_indices(layout, "video_indices", expected_video)

    tile_rows: list[list[int]] = []
    valid_counts: list[int] = []
    prefix_segments: list[int] = []
    for start, count in dense_segments:
        if count:
            prefix_segments.append(count)
            _append_dense_tiles(tile_rows, valid_counts, start, count, OMNI_REF_TILE_SIZE)
    num_prefix_tiles = len(tile_rows)

    video_tile_spans: list[tuple[int, int]] = []
    for start, grid in reference_video_regions:
        tile_start = len(tile_rows)
        _append_video_tiles(
            tile_rows,
            valid_counts,
            start,
            grid,
            OMNI_REF_TILE_SHAPE,
            OMNI_REF_TILE_SIZE,
        )
        video_tile_spans.append((tile_start, len(tile_rows)))

    target_tile_start = len(tile_rows)
    _append_video_tiles(
        tile_rows,
        valid_counts,
        target_video_start,
        target_grid,
        OMNI_REF_TILE_SHAPE,
        OMNI_REF_TILE_SIZE,
    )
    video_tile_spans.append((target_tile_start, len(tile_rows)))

    return _geometry_from_tiles(
        sequence_length=sequence_length,
        prefix_segments=tuple(prefix_segments),
        latent_grid=target_grid,
        num_prefix_tiles=num_prefix_tiles,
        tile_rows=tile_rows,
        valid_counts=valid_counts,
        sparsity=sparsity,
        tile_size=OMNI_REF_TILE_SIZE,
        video_tile_spans=tuple(video_tile_spans),
    )


def _require_contiguous_rows(layout: Any, field: str, start: int, count: int) -> None:
    rows = getattr(layout, field, None)
    if not torch.is_tensor(rows) or rows.ndim != 1:
        raise TypeError(f"T2AV layout must provide a one-dimensional {field} tensor")
    if int(rows.numel()) != count:
        raise ValueError(f"T2AV layout {field} has {rows.numel()} rows; expected {count}")
    if count == 0:
        return
    expected = torch.arange(start, start + count, device=rows.device, dtype=rows.dtype)
    if not torch.equal(rows, expected):
        raise ValueError(f"FastH3 VSA requires contiguous {field} rows at sequence offset {start}")


def _build_geometry(
    layout: Any,
    latent_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    sparsity: float,
    *,
    omni_ref: bool = False,
    references: Sequence[Any] | None = None,
) -> FastH3Geometry:
    if omni_ref:
        return _build_omniref_geometry(layout, latent_shape, patch_size, sparsity, references)

    for field in (
        "num_condition_video_rows",
        "num_condition_audio_rows",
        "num_target_condition_audio_latents",
        "num_target_condition_video_rows",
    ):
        value = getattr(layout, field, 0)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"T2AV layout {field} must be an integer")
        if int(value) != 0:
            raise ValueError(
                "FastH3 VSA currently supports T2AV only; reference audio/video and frozen target frames are unsupported."
            )

    latent_t, latent_h, latent_w = _shape_tuple(latent_shape, "latent_shape")
    patch_t, patch_h, patch_w = _shape_tuple(patch_size, "patch_size")
    if any(size % patch for size, patch in zip((latent_t, latent_h, latent_w), (patch_t, patch_h, patch_w))):
        raise ValueError(f"Latent dimensions {latent_shape} are not divisible by patch size {patch_size}")
    grid = (latent_t // patch_t, latent_h // patch_h, latent_w // patch_w)

    try:
        sequence_length_value = layout.sequence_length
    except AttributeError as error:
        raise TypeError("T2AV layout must provide an integer sequence_length") from error
    if isinstance(sequence_length_value, bool) or not isinstance(sequence_length_value, Integral):
        raise TypeError("T2AV layout sequence_length must be an integer")
    sequence_length = int(sequence_length_value)
    if sequence_length <= 0:
        raise ValueError("T2AV layout sequence_length must be positive")
    text_rows = _count_rows(layout, "text_indices")
    audio_rows = _count_rows(layout, "audio_indices")
    video_rows = _count_rows(layout, "video_indices")
    expected_video_rows = math.prod(grid)
    if audio_rows <= 0:
        raise ValueError("FastH3 VSA requires a target audio segment")
    if video_rows != expected_video_rows:
        raise ValueError(
            f"T2AV layout has {video_rows} target-video rows, but latent grid {grid} has {expected_video_rows} patches"
        )
    if text_rows + audio_rows + video_rows != sequence_length:
        raise ValueError("FastH3 VSA requires one packed text, target-audio, target-video sequence")

    _require_contiguous_rows(layout, "text_indices", 0, text_rows)
    _require_contiguous_rows(layout, "audio_indices", text_rows, audio_rows)
    _require_contiguous_rows(layout, "video_indices", text_rows + audio_rows, video_rows)

    # Prefix tiles are segment-pure: text and target audio are chunked
    # independently even when the prior segment leaves a partial tile.
    tile_rows: list[list[int]] = []
    valid_counts: list[int] = []
    for start, length in ((0, text_rows), (text_rows, audio_rows)):
        for offset in range(0, length, TILE_SIZE):
            rows = list(range(start + offset, start + min(offset + TILE_SIZE, length)))
            valid_counts.append(len(rows))
            tile_rows.append(rows + [-1] * (TILE_SIZE - len(rows)))
    num_prefix_tiles = len(tile_rows)

    video_start = text_rows + audio_rows
    grid_t, grid_h, grid_w = grid
    video_tile_start = len(tile_rows)
    for t0 in range(0, grid_t, TILE_SHAPE[0]):
        for h0 in range(0, grid_h, TILE_SHAPE[1]):
            for w0 in range(0, grid_w, TILE_SHAPE[2]):
                rows = []
                for t in range(t0, min(t0 + TILE_SHAPE[0], grid_t)):
                    for h in range(h0, min(h0 + TILE_SHAPE[1], grid_h)):
                        for w in range(w0, min(w0 + TILE_SHAPE[2], grid_w)):
                            rows.append(video_start + (t * grid_h + h) * grid_w + w)
                if not rows:
                    raise AssertionError("FastH3 VSA constructed an empty video tile")
                valid_counts.append(len(rows))
                tile_rows.append(rows + [-1] * (TILE_SIZE - len(rows)))

    num_video_tiles = len(tile_rows) - video_tile_start
    if num_video_tiles <= 0:
        raise ValueError("FastH3 VSA requires at least one target-video tile")
    if sum(valid_counts) != sequence_length:
        raise AssertionError("FastH3 VSA tile geometry does not cover the packed sequence exactly")

    return _geometry_from_tiles(
        sequence_length=sequence_length,
        prefix_segments=(text_rows, audio_rows),
        latent_grid=grid,
        num_prefix_tiles=num_prefix_tiles,
        tile_rows=tile_rows,
        valid_counts=valid_counts,
        sparsity=sparsity,
        tile_size=TILE_SIZE,
        video_tile_spans=((video_tile_start, len(tile_rows)),),
    )


def _pool_tokens(
    tensor: torch.Tensor,
    row_to_tile: torch.Tensor,
    valid_counts: torch.Tensor,
    num_tiles: int,
) -> torch.Tensor:
    batch, _, heads, width = tensor.shape
    pooled = torch.zeros((batch, num_tiles, heads, width), device=tensor.device, dtype=torch.float32)
    # Cast bounded token chunks rather than materializing a full FP32 copy of
    # the packed activation tensor (which can be very large at H3 resolutions).
    for row_start in range(0, tensor.shape[1], 512):
        row_end = min(row_start + 512, tensor.shape[1])
        pooled.index_add_(
            1,
            row_to_tile[row_start:row_end],
            tensor[:, row_start:row_end].float(),
        )
    return pooled / valid_counts.to(device=tensor.device, dtype=torch.float32).view(1, num_tiles, 1, 1)


def _routing_and_compression(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    geometry: FastH3Geometry,
    *,
    sparsity: float,
    heads_per_chunk: int,
    max_score_elements: int = _SCORE_ELEMENT_BUDGET,
    need_compression: bool,
    row_to_tile: torch.Tensor | None = None,
    valid_counts: torch.Tensor | None = None,
    pooled_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute bounded-memory pooled VSA routing and, when needed, gate values."""

    batch, _, heads, width = query.shape
    if pooled_qkv is None:
        if row_to_tile is None:
            row_to_tile = geometry.row_to_tile.to(device=query.device, dtype=torch.long)
        if valid_counts is None:
            valid_counts = geometry.valid_counts.to(device=query.device)
        q_pool = _pool_tokens(query, row_to_tile, valid_counts, geometry.num_tiles)
        k_pool = _pool_tokens(key, row_to_tile, valid_counts, geometry.num_tiles)
        v_pool = _pool_tokens(value, row_to_tile, valid_counts, geometry.num_tiles)
    else:
        q_pool, k_pool, v_pool = pooled_qkv
    route_count = geometry.route_count
    route = torch.empty(
        (batch, heads, geometry.num_video_tiles, route_count),
        dtype=torch.int32,
        device=query.device,
    )
    compression = (
        torch.empty((batch, geometry.num_tiles, heads, width), dtype=torch.float32, device=query.device)
        if need_compression
        else None
    )
    head_chunk = max(1, min(int(heads_per_chunk), heads))
    scale = width ** -0.5
    for head_start in range(0, heads, head_chunk):
        head_end = min(head_start + head_chunk, heads)
        q_heads = q_pool[:, :, head_start:head_end, :].permute(0, 2, 1, 3).contiguous()
        k_heads = k_pool[:, :, head_start:head_end, :].permute(0, 2, 1, 3).contiguous()
        v_heads = v_pool[:, :, head_start:head_end, :].permute(0, 2, 1, 3).contiguous()
        chunk_batch_heads = batch * (head_end - head_start)
        candidate_keys = k_heads.transpose(-2, -1)

        if need_compression:
            query_start_limit = geometry.num_tiles
        else:
            query_start_limit = geometry.num_video_tiles

        query_chunk = max(
            1,
            max_score_elements // max(1, chunk_batch_heads * geometry.num_tiles),
        )
        for query_start in range(0, query_start_limit, query_chunk):
            query_end = min(query_start + query_chunk, query_start_limit)
            if need_compression:
                q_start = query_start
            else:
                q_start = geometry.num_prefix_tiles + query_start
            q_block = q_heads[:, :, q_start:q_start + (query_end - query_start), :]
            score_block = torch.matmul(q_block, candidate_keys) * scale

            if compression is not None:
                compressed = torch.matmul(torch.softmax(score_block, dim=-1), v_heads)
                compression[:, q_start:q_start + (query_end - query_start), head_start:head_end, :] = (
                    compressed.permute(0, 2, 1, 3)
                )

            video_query_start = max(q_start, geometry.num_prefix_tiles)
            video_query_end = min(q_start + score_block.shape[-2], geometry.num_tiles)
            if video_query_start < video_query_end:
                local_start = video_query_start - q_start
                local_end = video_query_end - q_start
                video_query_scores = score_block[:, :, local_start:local_end, :]
                query_rows = slice(
                    video_query_start - geometry.num_prefix_tiles,
                    video_query_end - geometry.num_prefix_tiles,
                )
                route_offset = 0
                for (region_start, region_end), region_topk in zip(
                    geometry.video_tile_spans, geometry.span_topks, strict=True
                ):
                    indices = torch.topk(
                        video_query_scores[..., region_start:region_end],
                        k=region_topk,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    ).indices
                    route[
                        :, head_start:head_end, query_rows, route_offset : route_offset + region_topk
                    ] = (indices.to(torch.int32) + region_start)
                    route_offset += region_topk

    return route, compression


def fast_h3_attention_status(device: Any = None, dtype: torch.dtype | None = None) -> dict[str, Any]:
    """Return whether the CUDA/Triton implementation can run on ``device``."""

    try:
        target = torch.device(device) if device is not None else torch.device("cuda")
    except (TypeError, RuntimeError) as error:
        return {"supported": False, "reason": f"Invalid FastH3 attention device: {error}"}
    if target.type != "cuda":
        return {"supported": False, "reason": "FastH3 VSA requires CUDA tensors"}
    if not torch.cuda.is_available():
        return {"supported": False, "reason": "CUDA is not available"}
    if dtype is not None and dtype not in {torch.float16, torch.bfloat16}:
        return {"supported": False, "reason": f"FastH3 VSA requires FP16/BF16 tensors, got {dtype}"}
    try:
        import triton  # noqa: F401 - intentionally lazy; not imported on CPU paths
    except Exception as error:  # noqa: BLE001 - native import errors vary by CUDA/Triton build.
        return {"supported": False, "reason": f"Triton is unavailable: {error}"}
    try:
        index = target.index if target.index is not None else torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(index)
    except (RuntimeError, TypeError, ValueError) as error:
        return {"supported": False, "reason": f"Could not query CUDA device capability: {error}"}
    if dtype == torch.bfloat16 and major < 8:
        return {"supported": False, "reason": "BF16 Triton dot products require CUDA compute capability 8.0 or newer"}
    if dtype == torch.float16 and major < 7:
        return {"supported": False, "reason": "FP16 Triton dot products require CUDA compute capability 7.0 or newer"}
    return {"supported": True, "reason": None}


def _check_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate: torch.Tensor | None, geometry: FastH3Geometry) -> None:
    tensors = (("query", query), ("key", key), ("value", value))
    if not all(torch.is_tensor(tensor) and tensor.ndim == 4 for _, tensor in tensors):
        raise ValueError("FastH3 VSA expects query/key/value tensors in BLHD layout")
    if any(tuple(tensor.shape) != tuple(query.shape) for _, tensor in tensors[1:]):
        raise ValueError("FastH3 VSA query, key, and value shapes must match")
    if query.shape[1] != geometry.sequence_length:
        raise ValueError(
            f"FastH3 VSA layout has {geometry.sequence_length} rows but Q/K/V have {query.shape[1]}"
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("FastH3 VSA query, key, and value dtypes must match")
    if query.device != key.device or query.device != value.device:
        raise ValueError("FastH3 VSA query, key, and value must be on one device")
    if gate is not None:
        if not torch.is_tensor(gate) or tuple(gate.shape) != tuple(query.shape):
            raise ValueError("FastH3 VSA gate must be None or have the same BLHD shape as query")
        if gate.device != query.device:
            raise ValueError("FastH3 VSA gate must be on the query device")
        if not gate.dtype.is_floating_point:
            raise ValueError("FastH3 VSA gate must be floating point")
        if gate.dtype != query.dtype:
            raise ValueError("FastH3 VSA gate must use the query activation dtype")


def reference_attention_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    geometry: FastH3Geometry,
    *,
    sparsity: float = DEFAULT_SPARSITY,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Small, explicit CPU oracle for tests; this is never a production fallback."""

    _check_inputs(query, key, value, gate, geometry)
    if query.device.type != "cpu":
        raise ValueError("reference_attention_cpu only accepts CPU tensors")
    _normalize_sparsity(sparsity)
    batch, _, heads, width = query.shape
    tile_rows = geometry.tile_rows
    row_to_tile = geometry.row_to_tile.long()
    valid_counts = geometry.valid_counts
    q_pool = _pool_tokens(query, row_to_tile, valid_counts, geometry.num_tiles)
    k_pool = _pool_tokens(key, row_to_tile, valid_counts, geometry.num_tiles)
    v_pool = _pool_tokens(value, row_to_tile, valid_counts, geometry.num_tiles)
    pooled_scores = torch.matmul(
        q_pool.permute(0, 2, 1, 3),
        k_pool.permute(0, 2, 3, 1),
    ) * (width ** -0.5)
    compression = None
    if gate is not None:
        compression = torch.matmul(torch.softmax(pooled_scores, dim=-1), v_pool.permute(0, 2, 1, 3))
        compression = compression.permute(0, 2, 1, 3)

    result = torch.empty_like(query)
    for batch_index in range(batch):
        for head_index in range(heads):
            for query_tile in range(geometry.num_tiles):
                if query_tile < geometry.num_prefix_tiles:
                    key_tiles = list(range(geometry.num_tiles))
                else:
                    key_tiles = list(range(geometry.num_prefix_tiles))
                    for (region_start, region_end), region_topk in zip(
                        geometry.video_tile_spans, geometry.span_topks, strict=True
                    ):
                        scores = pooled_scores[
                            batch_index,
                            head_index,
                            query_tile,
                            region_start:region_end,
                        ]
                        selected = torch.topk(
                            scores, region_topk, dim=-1, largest=True, sorted=False
                        ).indices
                        key_tiles.extend(region_start + int(index) for index in selected.tolist())
                query_rows = tile_rows[query_tile]
                query_rows = query_rows[query_rows >= 0].long()
                key_rows = tile_rows[key_tiles]
                key_rows = key_rows[key_rows >= 0].long()
                logits = torch.matmul(
                    query[batch_index, query_rows, head_index].float(),
                    key[batch_index, key_rows, head_index].float().transpose(0, 1),
                ) * (width ** -0.5)
                attended = torch.matmul(torch.softmax(logits, dim=-1), value[batch_index, key_rows, head_index].float())
                if compression is not None:
                    # The trained implementation merges after both attention
                    # branches have been rounded to the model activation dtype.
                    fine = attended.to(query.dtype)
                    compressed = compression[batch_index, query_tile, head_index].to(query.dtype)
                    gate_rows = gate[batch_index, query_rows, head_index].to(query.dtype)
                    attended = torch.addcmul(fine, compressed.unsqueeze(0), gate_rows)
                result[batch_index, query_rows, head_index] = attended.to(query.dtype)
    return result


class FastH3Attention:
    """Per-forward FastH3 VSA geometry and CUDA attention policy."""

    def __init__(
        self,
        sparsity: float | None = None,
        *,
        heads_per_chunk: int = 4,
        max_score_elements: int = _SCORE_ELEMENT_BUDGET,
        omni_ref: bool = False,
    ) -> None:
        self.omni_ref = bool(omni_ref)
        default_sparsity = OMNI_REF_DEFAULT_SPARSITY if self.omni_ref else DEFAULT_SPARSITY
        self.sparsity = _normalize_sparsity(default_sparsity if sparsity is None else sparsity)
        if isinstance(heads_per_chunk, bool) or not isinstance(heads_per_chunk, Integral) or int(heads_per_chunk) <= 0:
            raise ValueError("heads_per_chunk must be a positive integer")
        if isinstance(max_score_elements, bool) or not isinstance(max_score_elements, Integral) or int(max_score_elements) <= 0:
            raise ValueError("max_score_elements must be a positive integer")
        self.heads_per_chunk = int(heads_per_chunk)
        self.max_score_elements = int(max_score_elements)
        self.geometry: FastH3Geometry | None = None
        self._device_geometry: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def require_supported(device: Any = None, dtype: torch.dtype | None = None) -> None:
        """Raise a descriptive error when the CUDA/Triton VSA path is unavailable."""

        status = fast_h3_attention_status(device, dtype)
        if not status["supported"]:
            raise RuntimeError(status["reason"])

    def begin_forward(
        self,
        layout: Any,
        latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int] = (1, 2, 2),
        references: Sequence[Any] | None = None,
    ) -> FastH3Geometry:
        """Validate a T2AV or Ref2VA layout and prepare its segment-pure tiles."""

        if references is not None and not self.omni_ref:
            raise ValueError("Prepared references require FastH3Attention(omni_ref=True)")
        self.geometry = _build_geometry(
            layout,
            latent_shape,
            patch_size,
            self.sparsity,
            omni_ref=self.omni_ref,
            references=references,
        )
        self._device_geometry.clear()
        return self.geometry

    def _geometry_on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.geometry is None:
            raise RuntimeError("FastH3Attention.begin_forward must be called before attention")
        key = str(device)
        cached = self._device_geometry.get(key)
        if cached is None:
            cached = (
                self.geometry.tile_rows.to(device=device, dtype=torch.int32).contiguous(),
                self.geometry.valid_counts.to(device=device, dtype=torch.int32).contiguous(),
            )
            self._device_geometry[key] = cached
        return cached

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.geometry is None:
            raise RuntimeError("FastH3Attention.begin_forward must be called before attention")
        _check_inputs(query, key, value, gate, self.geometry)
        if query.shape[-1] != 128:
            raise ValueError(f"Native FastH3 VSA currently requires 128-wide heads, got {query.shape[-1]}")
        self.require_supported(query.device, query.dtype)
        from .fasth3_kernel import block_sparse_attention, pool_qkv

        query, key, value = (tensor.contiguous() for tensor in (query, key, value))
        if gate is not None:
            gate = gate.contiguous()
        tile_rows, valid_counts = self._geometry_on(query.device)
        pooled_qkv = pool_qkv(query, key, value, tile_rows, valid_counts)
        route, compression = _routing_and_compression(
            query,
            key,
            value,
            self.geometry,
            sparsity=self.sparsity,
            heads_per_chunk=self.heads_per_chunk,
            max_score_elements=self.max_score_elements,
            need_compression=gate is not None,
            pooled_qkv=pooled_qkv,
        )
        output = block_sparse_attention(
            query,
            key,
            value,
            gate,
            compression,
            tile_rows,
            route,
            num_prefix_tiles=self.geometry.num_prefix_tiles,
            num_video_tiles=self.geometry.num_video_tiles,
            topk=self.geometry.route_count,
            tile_size=self.geometry.tile_size,
        )
        return output

    def reference(self, query, key, value, gate=None) -> torch.Tensor:
        """Convenience wrapper around the test-only CPU oracle."""

        if self.geometry is None:
            raise RuntimeError("FastH3Attention.begin_forward must be called before attention")
        return reference_attention_cpu(
            query,
            key,
            value,
            self.geometry,
            sparsity=self.sparsity,
            gate=gate,
        )


__all__ = [
    "DEFAULT_SPARSITY",
    "OMNI_REF_DEFAULT_SPARSITY",
    "OMNI_REF_TILE_SHAPE",
    "OMNI_REF_TILE_SIZE",
    "TILE_SHAPE",
    "TILE_SIZE",
    "FastH3Attention",
    "FastH3Geometry",
    "compute_topk",
    "fast_h3_attention_status",
    "reference_attention_cpu",
]
