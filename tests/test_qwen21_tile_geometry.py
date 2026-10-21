"""Qwen VAE context coverage and seams without loading model weights."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from models.qwen21.tiling import axis_weights, decode_image_to_cpu, image_tile_axis, tile_starts


class GeometryTests(unittest.TestCase):
    def test_all_lengths_have_full_edge_tiles_and_minimum_overlap(self):
        for length in range(1, 334):
            for requested in (256, 512, 1024):
                tile, starts = image_tile_axis(length, requested, 16)
                self.assertEqual(starts[0], 0)
                self.assertEqual(starts[-1] + tile, length)
                self.assertTrue(all(start + tile <= length for start in starts))
                if len(starts) > 1:
                    self.assertGreaterEqual(tile, 32)
                    self.assertTrue(all(b - a <= tile - 16 for a, b in zip(starts, starts[1:])))

    def test_no_zero_weight_gaps_including_three_way_overlap(self):
        for length in (33, 34, 63, 65, 68, 97, 120, 129, 172, 196):
            tile, starts = image_tile_axis(length, 512, 16)
            total = torch.zeros(length * 16)
            for start in starts:
                weights = axis_weights(tile, 16, start > 0, start + tile < length)
                total[start * 16:(start + tile) * 16] += weights
                if start:
                    self.assertTrue(torch.all(weights[:64] == 0))
                if start + tile < length:
                    self.assertTrue(torch.all(weights[-64:] == 0))
            self.assertTrue(torch.all(total > 0))

    def test_invalid_geometry_is_rejected(self):
        for args in ((0, 32, 16), (40, 0, 0), (40, 32, 32), (40, 32, -1)):
            with self.assertRaises(ValueError):
                tile_starts(*args)


class ContextDecoder(torch.nn.Module):
    """Expose errors only within four latent pixels of a padded tile edge."""
    def forward(self, z, feat_cache=None, first_chunk=False):
        if feat_cache is not None or not first_chunk:
            raise AssertionError("Single images must not retain temporal cache")
        output = z.clone()
        output[..., :4, :] = 0.95
        output[..., -4:, :] = 0.95
        output[..., :, :4] = 0.95
        output[..., :, -4:] = 0.95
        return output


class StreamingDecodeTests(unittest.TestCase):
    def vae(self, decoder=None):
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
        model.config = SimpleNamespace(patch_size=None)
        model.spatial_compression_ratio = 1
        model.tile_sample_min_height = model.tile_sample_min_width = 32
        model.use_tiling = True
        model.post_quant_conv = torch.nn.Identity()
        model.decoder = decoder or ContextDecoder()
        model.clear_cache = lambda: None
        return model

    def test_artificial_edge_errors_do_not_form_internal_lines(self):
        vae = self.vae()
        z = torch.linspace(-0.7, 0.7, 97 * 68).reshape(1, 1, 1, 97, 68).repeat(2, 4, 1, 1, 1)
        z[1].neg_()
        expected = vae.decoder(z, first_chunk=True).clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
        actual = decode_image_to_cpu(vae, z, 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.device.type, "cpu")

    def test_untiled_single_tile_and_rgb_selection_are_exact(self):
        class Identity(torch.nn.Module):
            def forward(self, z, **kwargs):
                return z
        vae = self.vae(Identity())
        vae.use_tiling = False
        z = torch.randn(1, 4, 1, 35, 49)
        actual = decode_image_to_cpu(vae, z, 3)
        expected = z[:, :3].float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_nonfinite_output_is_not_silently_quantized(self):
        class NonFinite(torch.nn.Module):
            def forward(self, z, **kwargs):
                return torch.full_like(z, float("nan"))
        with self.assertRaisesRegex(RuntimeError, "32-bit VAE"):
            decode_image_to_cpu(self.vae(NonFinite()), torch.zeros(1, 4, 1, 40, 43), 4)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
