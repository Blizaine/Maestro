"""CPU contract checks for FastH3 OmniRef's multi-region VSA geometry."""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from models.minimax_h3.fasth3_attention import (
    FastH3Attention,
    OMNI_REF_DEFAULT_SPARSITY,
    OMNI_REF_TILE_SIZE,
    _build_geometry,
    reference_attention_cpu,
)
from models.minimax_h3.packing import MINIMAX_H3_TEXT_TAG
from models.minimax_h3.ref2va import MiniMaxH3PreparedReference, build_ref2va_packed_sequence


_PATCH = (1, 2, 2)


def _reference(
    kind: str,
    *,
    frames: int = 1,
    height: int = 8,
    width: int = 32,
    audio: int = 0,
) -> MiniMaxH3PreparedReference:
    return MiniMaxH3PreparedReference(
        kind=kind,
        num_latent_frames=frames,
        latent_height=height,
        latent_width=width,
        num_audio_latents=audio,
    )


def _layout(
    references: list[MiniMaxH3PreparedReference],
    *,
    text_rows: int = 5,
    target: tuple[int, int, int] = (1, 8, 32),
    target_audio_latents: int = 1,
):
    tags = torch.full((text_rows,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    return build_ref2va_packed_sequence(
        tags,
        references,
        *target,
        target_audio_latents,
        _PATCH,
    )


class FastH3OmniRefGeometryTests(unittest.TestCase):
    def test_ordered_mixed_references_make_dense_prefix_and_video_regions(self):
        references = [
            _reference("image"),
            _reference("video", frames=5, audio=4),
            _reference("audio", audio=3),
            _reference("video", frames=2),
        ]
        layout = _layout(references, target=(8, 8, 32), target_audio_latents=5)
        geometry = _build_geometry(
            layout,
            (8, 8, 32),
            _PATCH,
            OMNI_REF_DEFAULT_SPARSITY,
            omni_ref=True,
            references=references,
        )

        # Dense tiles are text, image, each reference-audio span and target
        # audio. The two reference videos and the target each retain their own
        # 4x4x8 tile region.
        self.assertEqual(geometry.tile_size, OMNI_REF_TILE_SIZE)
        self.assertEqual(geometry.prefix_segments, (5, 64, 8, 6, 10))
        self.assertEqual(geometry.num_prefix_tiles, 5)
        self.assertEqual(geometry.video_tile_spans, ((5, 9), (9, 11), (11, 15)))
        self.assertEqual(geometry.span_topks, (1, 1, 1))
        self.assertEqual(geometry.route_count, 3)
        self.assertEqual(geometry.num_tiles, 15)

        # True packed rows can be interleaved even though tile IDs group all
        # dense runs before the sparse reference and target regions.
        for start, end in ((77, 397), (403, 531), (541, 1053)):
            mapped = geometry.row_to_tile[start:end]
            self.assertTrue(bool(((mapped >= 5) & (mapped < 15)).all()))
        for start, end in ((0, 5), (5, 69), (69, 77), (397, 403), (531, 541)):
            mapped = geometry.row_to_tile[start:end]
            self.assertTrue(bool((mapped < geometry.num_prefix_tiles).all()))
        self.assertEqual(int(geometry.valid_counts.sum()), layout.sequence_length)

    def test_image_and_audio_references_stay_dense_without_reference_video_spans(self):
        references = [_reference("image"), _reference("audio", audio=2)]
        layout = _layout(references)
        policy = FastH3Attention(omni_ref=True)
        geometry = policy.begin_forward(layout, (1, 8, 32), _PATCH, references=references)

        self.assertEqual(policy.sparsity, OMNI_REF_DEFAULT_SPARSITY)
        self.assertEqual(len(geometry.video_tile_spans), 1)
        self.assertEqual(geometry.video_tile_spans[0][0], geometry.num_prefix_tiles)
        image_rows = slice(5, 5 + 64)
        self.assertTrue(bool((geometry.row_to_tile[image_rows] < geometry.num_prefix_tiles).all()))

    def test_cpu_oracle_keeps_top_tiles_from_each_video_region(self):
        references = [_reference("video"), _reference("video")]
        layout = _layout(references)
        policy = FastH3Attention(omni_ref=True)
        geometry = policy.begin_forward(layout, (1, 8, 32), _PATCH, references=references)
        self.assertEqual(geometry.video_tile_spans, ((2, 4), (4, 6), (6, 8)))
        self.assertEqual(geometry.span_topks, (1, 1, 1))

        query = torch.zeros(1, geometry.sequence_length, 1, 4)
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)
        tile_key = (0.0, 0.0, 2.0, 1.9, 1.0, 0.1, 0.0, -0.1)
        tile_value = (100.0, 200.0, 1.0, 100.0, 10.0, 1000.0, 20.0, 2000.0)
        for tile_index, (key_value, output_value) in enumerate(zip(tile_key, tile_value, strict=True)):
            rows = geometry.tile_rows[tile_index]
            rows = rows[rows >= 0].long()
            key[0, rows, 0, 0] = key_value
            value[0, rows, 0, 0] = output_value

        target_query_tile = geometry.video_tile_spans[-1][0]
        query_rows = geometry.tile_rows[target_query_tile]
        query_rows = query_rows[query_rows >= 0].long()
        query[0, query_rows, 0, 0] = 1.0

        actual = reference_attention_cpu(query, key, value, geometry)
        query_row = int(query_rows[0])
        dense_key_rows = geometry.tile_rows[: geometry.num_prefix_tiles]
        dense_key_rows = dense_key_rows[dense_key_rows >= 0].long()
        selected_key_rows = [dense_key_rows]
        # The first tile wins in every region. One global top-3 would instead
        # spend two of its slots on the first reference's competing tile.
        selected_key_rows.extend(
            geometry.tile_rows[start][geometry.tile_rows[start] >= 0].long()
            for start, _end in geometry.video_tile_spans
        )
        key_rows = torch.cat(selected_key_rows)
        logits = query[0, query_row, 0] @ key[0, key_rows, 0].T / (4**0.5)
        expected = torch.softmax(logits, dim=0) @ value[0, key_rows, 0]
        self.assertTrue(torch.allclose(actual[0, query_row, 0], expected, atol=1e-6, rtol=1e-6))

    def test_rejects_missing_reference_metadata_and_frozen_target_rows(self):
        references = [_reference("video")]
        layout = _layout(references)
        with self.assertRaisesRegex(ValueError, "reference-only visual conditioning"):
            _build_geometry(layout, (1, 8, 32), _PATCH, 0.9, omni_ref=True, references=[])

        frozen = replace(layout, num_target_condition_video_rows=1)
        with self.assertRaisesRegex(ValueError, "frozen target-video"):
            _build_geometry(frozen, (1, 8, 32), _PATCH, 0.9, omni_ref=True, references=references)

    def test_default_policy_keeps_the_existing_t2av_64_token_contract(self):
        layout = type("T2AVLayout", (), {})()
        layout.sequence_length = 3 + 64 + 32
        layout.text_indices = torch.arange(0, 3)
        layout.audio_indices = torch.arange(3, 67)
        layout.video_indices = torch.arange(67, 99)
        layout.num_condition_video_rows = 0
        layout.num_condition_audio_rows = 0
        layout.num_target_condition_audio_latents = 0
        layout.num_target_condition_video_rows = 0

        policy = FastH3Attention()
        geometry = policy.begin_forward(layout, (1, 4, 8), (1, 1, 1))
        self.assertEqual(policy.sparsity, 0.80)
        self.assertEqual(geometry.tile_size, 64)
        self.assertEqual(geometry.video_tile_spans, ((geometry.num_prefix_tiles, geometry.num_tiles),))
        self.assertEqual(geometry.route_count, geometry.topk)


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("MAESTRO_ENABLE_FASTH3_OMNIREF_CUDA_TEST") == "1",
    "set MAESTRO_ENABLE_FASTH3_OMNIREF_CUDA_TEST=1 after verifying the GPU is idle",
)
class FastH3OmniRefCudaTests(unittest.TestCase):
    def test_128_token_multi_region_attention_matches_cpu_oracle(self):
        references = [_reference("video"), _reference("video")]
        layout = _layout(references)
        policy = FastH3Attention(omni_ref=True, heads_per_chunk=1)
        geometry = policy.begin_forward(layout, (1, 8, 32), _PATCH, references=references)
        generator = torch.Generator(device="cpu").manual_seed(1907)
        query = torch.randn(1, geometry.sequence_length, 2, 128, generator=generator) * 0.02
        key = torch.randn(1, geometry.sequence_length, 2, 128, generator=generator) * 0.02
        value = torch.randn(1, geometry.sequence_length, 2, 128, generator=generator)
        gate = torch.full_like(query, 0.25)

        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            q_cuda, k_cuda, v_cuda, gate_cuda = [
                tensor.to(device="cuda", dtype=dtype) for tensor in (query, key, value, gate)
            ]
            actual = policy(q_cuda, k_cuda, v_cuda, gate_cuda).float().cpu()
            expected = reference_attention_cpu(
                q_cuda.cpu(),
                k_cuda.cpu(),
                v_cuda.cpu(),
                geometry,
                gate=gate_cuda.cpu(),
            ).float()
            self.assertTrue(torch.allclose(actual, expected, atol=3e-2, rtol=3e-2))


if __name__ == "__main__":
    unittest.main()
