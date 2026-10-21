"""CPU contract checks for native FastH3 VSA geometry and oracle behavior."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from models.minimax_h3.fasth3_attention import (
    FastH3Attention,
    _build_geometry,
    compute_topk,
    fast_h3_attention_status,
    reference_attention_cpu,
)


def _layout(text_rows: int, audio_rows: int, video_rows: int, **overrides):
    sequence_length = text_rows + audio_rows + video_rows
    layout = {
        "sequence_length": sequence_length,
        "text_indices": torch.arange(0, text_rows, dtype=torch.long),
        "audio_indices": torch.arange(text_rows, text_rows + audio_rows, dtype=torch.long),
        "video_indices": torch.arange(text_rows + audio_rows, sequence_length, dtype=torch.long),
        "num_condition_video_rows": 0,
        "num_condition_audio_rows": 0,
        "num_target_condition_audio_latents": 0,
        "num_target_condition_video_rows": 0,
    }
    layout.update(overrides)
    return SimpleNamespace(**layout)


class TestFastH3Geometry(unittest.TestCase):
    def test_prefix_segments_and_partial_3d_tiles_remain_separate(self):
        layout = _layout(text_rows=3, audio_rows=66, video_rows=35)
        geometry = _build_geometry(layout, (1, 10, 14), (1, 2, 2), 0.8)

        self.assertEqual(geometry.prefix_segments, (3, 66))
        self.assertEqual(geometry.latent_grid, (1, 5, 7))
        self.assertEqual(geometry.num_prefix_tiles, 3)
        self.assertEqual(geometry.num_video_tiles, 4)
        self.assertEqual(geometry.topk, 1)
        self.assertEqual(geometry.valid_counts.tolist(), [3, 64, 2, 16, 12, 4, 3])
        self.assertTrue(torch.equal(geometry.row_to_tile[0:3], torch.tensor([0, 0, 0])))
        self.assertTrue(torch.equal(geometry.row_to_tile[3:67], torch.ones(64, dtype=torch.int32)))
        self.assertTrue(torch.equal(geometry.row_to_tile[67:69], torch.full((2,), 2, dtype=torch.int32)))
        self.assertEqual(int(geometry.tile_rows[-1][-1]), -1)
        self.assertEqual(int(geometry.tile_rows[-1][0]), 101)

    def test_topk_rounding_matches_fastvideo_ceiling_rule(self):
        self.assertEqual(compute_topk(0.8, 4), 1)
        self.assertEqual(compute_topk(0.8, 6), 2)
        self.assertEqual(compute_topk(1.0, 6), 1)
        self.assertEqual(compute_topk(0.0, 6), 6)
        self.assertEqual(compute_topk(0.8, 0), 0)
        with self.assertRaises(ValueError):
            compute_topk(float("nan"), 4)

    def test_rejects_reference_or_frozen_condition_rows(self):
        layout = _layout(1, 1, 32, num_condition_video_rows=1)
        with self.assertRaisesRegex(ValueError, "T2AV only"):
            _build_geometry(layout, (1, 4, 8), (1, 1, 1), 0.8)

    def test_rejects_noncontiguous_packed_segments(self):
        layout = _layout(1, 1, 32)
        layout.audio_indices = torch.tensor([2])
        with self.assertRaisesRegex(ValueError, "contiguous audio_indices"):
            _build_geometry(layout, (1, 4, 8), (1, 1, 1), 0.8)


class TestFastH3CPUOracle(unittest.TestCase):
    def _geometry(self):
        # The two video tiles each contain 16 rows. Text and target audio are
        # separate prefix documents even though both fit in a single tile.
        return _build_geometry(_layout(1, 64, 32), (1, 4, 8), (1, 1, 1), 0.8)

    def _inputs(self, batch=1, heads=2, width=4):
        geometry = self._geometry()
        query = torch.zeros(batch, geometry.sequence_length, heads, width)
        key = torch.zeros_like(query)
        value = torch.empty_like(query)
        tile_values = (1.0, 2.0, 10.0, 20.0)
        for tile_index, scalar in enumerate(tile_values):
            rows = geometry.tile_rows[tile_index]
            rows = rows[rows >= 0].long()
            value[:, rows] = scalar
        if batch > 1:
            value[1] += 100.0
        return geometry, query, key, value

    def test_prefix_queries_are_dense_and_video_queries_keep_prefix_keys(self):
        geometry, query, key, value = self._inputs(batch=2)
        output = reference_attention_cpu(query, key, value, geometry)

        # With zero Q/K every selected set is uniform. Prefix rows average all
        # packed tokens. Video rows average all prefix tokens and exactly one
        # of the two video tiles selected by top-k.
        prefix_rows = geometry.tile_rows[: geometry.num_prefix_tiles]
        prefix_rows = prefix_rows[prefix_rows >= 0].long()
        all_rows = torch.arange(geometry.sequence_length)
        for batch_index in range(2):
            expected_dense = value[batch_index, all_rows].mean(dim=0)
            self.assertTrue(
                torch.allclose(output[batch_index, prefix_rows], expected_dense.expand_as(output[batch_index, prefix_rows]))
            )

            prefix_value_sum = value[batch_index, prefix_rows].sum(dim=0)
            prefix_count = prefix_rows.numel()
            candidate_means = []
            for video_tile in range(geometry.num_prefix_tiles, geometry.num_tiles):
                rows = geometry.tile_rows[video_tile]
                rows = rows[rows >= 0].long()
                candidate_means.append(
                    (prefix_value_sum + value[batch_index, rows].sum(dim=0)) / (prefix_count + rows.numel())
                )
            video_rows = geometry.tile_rows[geometry.num_prefix_tiles :]
            video_rows = video_rows[video_rows >= 0].long()
            actual = output[batch_index, video_rows]
            self.assertTrue(
                any(torch.allclose(actual, candidate.expand_as(actual)) for candidate in candidate_means)
            )

    def test_batch_routing_is_independent(self):
        geometry, query, key, value = self._inputs(batch=2)
        batched = reference_attention_cpu(query, key, value, geometry)
        separate = torch.cat(
            [
                reference_attention_cpu(query[index : index + 1], key[index : index + 1], value[index : index + 1], geometry)
                for index in range(2)
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(batched, separate))

    def test_gate_is_raw_linear_projection_and_uses_pooled_values(self):
        geometry, query, key, value = self._inputs()
        without_gate = reference_attention_cpu(query, key, value, geometry)
        gate = torch.full_like(query, 2.0)
        with_gate = reference_attention_cpu(query, key, value, geometry, gate=gate)

        # Q/K are zero, so compression attends uniformly to the four pooled
        # tile values. Raw gate=2 doubles that pooled result (no tanh).
        tile_means = []
        for tile_index in range(geometry.num_tiles):
            rows = geometry.tile_rows[tile_index]
            rows = rows[rows >= 0].long()
            tile_means.append(value[0, rows].mean(dim=0))
        compressed = torch.stack(tile_means).mean(dim=0)
        expected_delta = 2.0 * compressed
        self.assertTrue(
            torch.allclose(with_gate - without_gate, expected_delta.expand_as(without_gate))
        )

    def test_public_policy_preflights_cuda_before_kernel_import(self):
        policy = FastH3Attention()
        with self.assertRaisesRegex(RuntimeError, "CUDA tensors"):
            policy.require_supported(torch.device("cpu"), torch.float32)
        self.assertFalse(fast_h3_attention_status("cpu")["supported"])


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("MAESTRO_ENABLE_FASTH3_CUDA_TEST") == "1",
    "set MAESTRO_ENABLE_FASTH3_CUDA_TEST=1 after verifying the GPU is idle",
)
class TestFastH3CudaNumerics(unittest.TestCase):
    """Small parity check; run only when a CUDA test is deliberately enabled."""

    def test_incomplete_tiles_raw_gate_and_multiple_heads_match_cpu_oracle(self):
        geometry = _build_geometry(_layout(3, 66, 35), (1, 10, 14), (1, 2, 2), 0.8)
        generator = torch.Generator(device="cpu").manual_seed(713)
        batch, heads, width = 2, 2, 128
        query = torch.randn(batch, geometry.sequence_length, heads, width, generator=generator) * 0.02
        key = torch.randn(batch, geometry.sequence_length, heads, width, generator=generator) * 0.02
        value = torch.randn(batch, geometry.sequence_length, heads, width, generator=generator)
        gate = torch.full_like(query, 0.75)

        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            q_cuda, k_cuda, v_cuda, gate_cuda = [
                tensor.to(device="cuda", dtype=dtype) for tensor in (query, key, value, gate)
            ]
            policy = FastH3Attention(heads_per_chunk=1)
            policy.begin_forward(_layout(3, 66, 35), (1, 10, 14), (1, 2, 2))
            actual = policy(
                q_cuda,
                k_cuda,
                v_cuda,
                gate_cuda,
            ).float().cpu()
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
