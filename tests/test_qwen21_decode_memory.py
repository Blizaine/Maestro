"""CPU regressions for the single-frame Qwen 2.1 uint8 decoder cache path."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.qwen21.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21


def tiny_vae(patch_size=None):
    config = {
        "base_dim": 4,
        "decoder_base_dim": 4,
        "z_dim": 4,
        "dim_mult": [1, 1, 1, 1, 1],
        "num_res_blocks": 1,
        "attn_scales": [],
        "temperal_downsample": [False, True, True, True],
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
    }
    if patch_size is not None:
        # The patch decoder emits patch_size**2 channels per output channel and
        # upsamples one fewer stage before _unpatchify restores the full scale.
        config.update(
            patch_size=patch_size,
            out_channels=4 * patch_size * patch_size,
            dim_mult=[1, 1, 1, 1],
            temperal_downsample=[False, True, True],
        )
    return AutoencoderKLQwenImage21(**config).eval()


def cached_baseline(vae, latents, output_channels):
    """Run the previous decoder call shape, which supplied the feature cache."""
    original_forward = vae.decoder.forward

    def forward_with_cache(hidden_states, *args, **kwargs):
        kwargs["feat_cache"] = vae._feat_map
        kwargs["feat_idx"] = vae._conv_idx
        return original_forward(hidden_states, *args, **kwargs)

    vae.decoder.forward = forward_with_cache
    try:
        return vae.decode_to_cpu_uint8(latents.clone(), output_channels=output_channels)
    finally:
        vae.decoder.forward = original_forward


class Qwen21DecodeMemoryTests(unittest.TestCase):
    @torch.inference_mode()
    def test_single_frame_uint8_decode_matches_cached_baseline_without_retaining_features(self):
        cases = [(None, 2, (3, 5), False), (2, 2, (3, 5), False),
                 (None, 2, (33, 35), True), (2, 1, (33, 35), True)]
        for patch_size, batch, shape, tiled in cases:
            with self.subTest(patch_size=patch_size, shape=shape):
                torch.manual_seed(211)
                uncached_vae = tiny_vae(patch_size=patch_size)
                cached_vae = tiny_vae(patch_size=patch_size)
                cached_vae.load_state_dict(uncached_vae.state_dict())
                latents = torch.randn(batch, 4, 1, *shape)
                for vae in (uncached_vae, cached_vae):
                    if tiled:
                        vae.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512,
                                          tile_sample_stride_height=384, tile_sample_stride_width=384)
                cache_states = []
                original_forward = uncached_vae.decoder.forward

                def observe_uncached_forward(hidden_states, *args, **kwargs):
                    self.assertNotIn("feat_cache", kwargs)
                    self.assertNotIn("feat_idx", kwargs)
                    self.assertTrue(all(item is None for item in uncached_vae._feat_map))
                    decoded = original_forward(hidden_states, *args, **kwargs)
                    self.assertTrue(all(item is None for item in uncached_vae._feat_map))
                    cache_states.append(tuple(uncached_vae._feat_map))
                    return decoded

                uncached_vae.decoder.forward = observe_uncached_forward
                try:
                    actual = uncached_vae.decode_to_cpu_uint8(latents.clone(), output_channels=3)
                finally:
                    uncached_vae.decoder.forward = original_forward

                expected = cached_baseline(cached_vae, latents, output_channels=3)

                self.assertEqual(len(cache_states), batch * (4 if tiled else 1))
                self.assertTrue(all(all(item is None for item in state) for state in cache_states))
                self.assertEqual(tuple(actual.shape), (batch, 3, 1, shape[0] * 16, shape[1] * 16))
                self.assertEqual(actual.device.type, "cpu")
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
