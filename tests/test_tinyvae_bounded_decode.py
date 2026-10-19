"""Bounded preview batching must preserve temporal state and frame selection."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from shared.tinyvae import taehv


class RecordingSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        return x * 0.5


class BoundedDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.spatial = RecordingSpatial()
        self.model = nn.Sequential(taehv.MemBlock(2, 2), taehv.TGrow(2, 2),
                                   taehv.MemBlock(2, 2), taehv.TGrow(2, 2), self.spatial)
        self.x = torch.randn(1, 3, 2, 3, 3)

    def test_batched_pieces_preserve_all_temporal_outputs(self):
        with torch.no_grad():
            expected = taehv._apply(self.model, self.x, parallel=True)
            for max_elements in (18, 36, 1 << 23):
                with self.subTest(max_elements=max_elements), patch.object(taehv, "SEQUENTIAL_PIECE_ELEMENTS", max_elements):
                    actual = taehv._apply(self.model, self.x, parallel=False)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_unselected_frames_skip_final_spatial_work(self):
        with torch.no_grad():
            expected = taehv._apply(self.model, self.x, parallel=True)[:, [0, 4, 11]]
            self.spatial.batch_sizes.clear()
            actual = taehv._apply(self.model, self.x, parallel=False, output_indices={0, 4, 11},
                                  output_transform=lambda frame: frame + 1)
        torch.testing.assert_close(actual, expected + 1, atol=1e-6, rtol=1e-5)
        self.assertEqual(sum(self.spatial.batch_sizes), 3)

    def test_cancellation_interrupts_inside_a_latent_frame(self):
        checks = 0
        def cancelled():
            nonlocal checks
            checks += 1
            return checks >= 3
        with torch.no_grad():
            self.assertIsNone(taehv._apply(self.model, self.x, parallel=False, abort_check=cancelled))
        self.assertEqual(self.spatial.batch_sizes, [])


if __name__ == "__main__":
    unittest.main()
