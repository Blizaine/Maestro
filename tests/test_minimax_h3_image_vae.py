"""CPU checks for the standalone MiniMax H3 Image VAE."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3.image_vae import (  # noqa: E402
    IMAGE_VAE_LATENTS_MEAN,
    IMAGE_VAE_LATENTS_STD,
    MiniMaxH3ImageAttention,
    MiniMaxH3ImageVAE,
    MiniMaxH3ImageViTDecoder,
    _coerce_dtype,
    preprocess_image_vae_state_dict,
)


class MiniMaxH3ImageVAETests(unittest.TestCase):
    @staticmethod
    def make_vae(*, tile_size: int = 32, decoder_tile_size: int = 64) -> MiniMaxH3ImageVAE:
        torch.manual_seed(17)
        return MiniMaxH3ImageVAE(
            latent_channels=2,
            block_out_channels=(4, 8, 8, 16, 16, 32),
            layers_per_block=1,
            norm_num_groups=2,
            decoder_num_layers=1,
            decoder_num_attention_heads=2,
            decoder_attention_head_dim=8,
            decoder_num_register_tokens=2,
            decoder_ffn_mult=2,
            latents_mean=(0.0, 0.0),
            latents_std=(1.0, 1.0),
            encoder_tile_size=tile_size,
            encoder_tile_overlap=16 if tile_size > 16 else 0,
            decoder_tile_size=decoder_tile_size,
            decoder_tile_overlap=16 if decoder_tile_size > 16 else 0,
        ).eval().requires_grad_(False)

    def test_encode_decode_tensor_contract_and_seeded_posterior(self):
        vae = self.make_vae()
        images = torch.linspace(-1.0, 1.0, 2 * 3 * 32 * 48).view(2, 3, 32, 48)

        first = vae.encode(images)
        second = vae.encode(images.unsqueeze(2))
        self.assertEqual(tuple(first.shape), (2, 2, 1, 2, 3))
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(torch.equal(first, second))

        decoded = vae.decode(first)
        self.assertEqual(tuple(decoded.shape), (2, 3, 32, 48))
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertGreaterEqual(float(decoded.min()), 0.0)
        self.assertLessEqual(float(decoded.max()), 1.0)
        self.assertTrue(torch.equal(decoded, vae.decode(first, tiled=False)))

    def test_large_decode_uses_overlapped_tiles_and_preserves_full_path(self):
        vae = self.make_vae(decoder_tile_size=32)
        for block in vae.decoder.transformer_blocks:
            block.scale1.data.fill_(0.02)
            block.scale2.data.fill_(0.02)
        latent = torch.randn(1, 2, 1, 2, 3)
        blend_calls = []
        original_blend = vae._blend

        def record_blend(a, b, extent, dim):
            blend_calls.append((extent, dim))
            return original_blend(a, b, extent, dim)

        vae._blend = record_blend
        with torch.inference_mode():
            tiled = vae.decode(latent)
            full = vae.decode(latent, tiled=False)
        self.assertEqual(tuple(tiled.shape), (1, 3, 32, 48))
        self.assertEqual(tuple(full.shape), (1, 3, 32, 48))
        self.assertIn((16, -1), blend_calls)
        self.assertFalse(torch.equal(tiled, full))

    def test_tiled_decoder_blends_crossings_against_the_stitched_previous_row(self):
        class TileValueDecoder(torch.nn.Module):
            def forward(self, latent, *, interrupt_check=None):
                if interrupt_check is not None:
                    interrupt_check()
                value = latent[:, :1, :1, :1]
                return value.expand(-1, 3, latent.shape[-2] * 16, latent.shape[-1] * 16).clone()

        vae = self.make_vae(decoder_tile_size=32)
        vae.decoder = TileValueDecoder()
        vae.post_quant_conv = torch.nn.Identity()
        # Four latent tiles decode to constants 1, 2, 3, and 4. The upper
        # horizontal seam must be present in the strip used to blend the
        # lower-right tile, including where all four tiles meet.
        latent = torch.tensor(
            [[[[1.0, 2.0, 2.0], [3.0, 4.0, 4.0], [3.0, 4.0, 4.0]]]]
        )
        with torch.inference_mode():
            pixels = vae._decode_pixels(latent, tiled=True)
        self.assertEqual(tuple(pixels.shape), (1, 3, 48, 48))
        self.assertAlmostEqual(float(pixels[0, 0, 31, 0]), 2.875, places=6)
        self.assertAlmostEqual(float(pixels[0, 0, 31, 16]), 2.875, places=6)
        self.assertAlmostEqual(float(pixels[0, 0, 31, 31]), 3.8125, places=6)

    def test_large_encoder_inputs_use_latent_aligned_tiles(self):
        vae = self.make_vae(tile_size=32)
        images = torch.zeros(1, 3, 64, 48)
        latent = vae.encode(images)
        self.assertEqual(tuple(latent.shape), (1, 2, 1, 4, 3))

    def test_encode_and_decode_honor_runtime_interrupt_flag(self):
        vae = self.make_vae()
        vae._interrupt = True
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            vae.encode(torch.zeros(1, 3, 32, 32))
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            vae.decode(torch.zeros(1, 2, 1, 2, 2))

    def test_tiled_encoder_stitches_exact_patch_local_features(self):
        class PatchPoolMoments(torch.nn.Module):
            def forward(self, pixels):
                pooled = torch.nn.functional.avg_pool3d(
                    pixels, kernel_size=(1, 16, 16), stride=(1, 16, 16)
                )
                mean = pooled.repeat(1, 8, 1, 1, 1)
                return torch.cat((mean, torch.zeros_like(mean)), dim=1)

        vae = self.make_vae(tile_size=64)
        vae.encoder = PatchPoolMoments()
        vae.quant_conv = torch.nn.Identity()
        pixels = torch.randn(1, 3, 1, 128, 96)
        with torch.inference_mode():
            tiled = vae._encode_moments(pixels)
            vae.encoder_tile_size = 256
            full = vae._encode_moments(pixels)
        self.assertEqual(tuple(tiled.shape), (1, 48, 1, 8, 6))
        self.assertTrue(torch.allclose(tiled, full, atol=1e-6, rtol=1e-6))

    def test_decoder_outputs_patchified_image(self):
        torch.manual_seed(23)
        decoder = MiniMaxH3ImageViTDecoder(
            in_channels=4,
            out_channels=3,
            patch_size=2,
            num_layers=1,
            heads=2,
            head_dim=8,
            num_register_tokens=1,
            ffn_mult=2,
            attention_query_chunk_tokens=2,
        ).eval()
        latent = torch.randn(1, 4, 3, 4)
        with torch.inference_mode():
            chunked = decoder(latent)
        self.assertEqual(tuple(chunked.shape), (1, 3, 6, 8))

    def test_long_query_and_key_chunking_matches_one_shot_dense_attention(self):
        torch.manual_seed(27)
        attention = MiniMaxH3ImageAttention(dim=16, heads=2, head_dim=8)
        attention.query_chunk_tokens = 128
        attention.key_chunk_tokens = 257
        query = torch.randn(1, 2, 1025, 8)
        key = torch.randn(1, 2, 1025, 8)
        value = torch.randn(1, 2, 1025, 8)
        with torch.inference_mode():
            chunked = attention._attend(query, key, value)
            dense = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        self.assertTrue(torch.allclose(chunked, dense, atol=2e-5, rtol=2e-5))

    def test_online_softmax_attention_matches_sdpa(self):
        torch.manual_seed(29)
        query = torch.randn(1, 2, 9, 8)
        key = torch.randn(1, 2, 9, 8)
        value = torch.randn(1, 2, 9, 8)
        actual = MiniMaxH3ImageAttention._online_attention(
            query, key, value, key_chunk_tokens=3
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_native_checkpoint_key_adapter_maps_encoder_and_image_projection(self):
        tensors = {
            "encoder.down.0.block.0.conv1.weight": torch.zeros(1),
            "decoder.x_embedder.weight": torch.zeros(1),
            "decoder.transformer_blocks.0.attn.to_out.weight": torch.zeros(1),
            "decoder.transformer_blocks.0.ff.w1.weight": torch.zeros(1),
            "decoder.mask_token": torch.zeros(1),
            "latents_mean": torch.zeros(1),
        }
        mapped = preprocess_image_vae_state_dict(tensors)
        self.assertIn("encoder.down_blocks.0.resnets.0.conv1.weight", mapped)
        self.assertIn("decoder.x_embedder.weight", mapped)
        self.assertIn("decoder.transformer_blocks.0.attn.to_out.weight", mapped)
        self.assertIn("decoder.transformer_blocks.0.ff.w1.weight", mapped)
        self.assertNotIn("decoder.mask_token", mapped)
        self.assertNotIn("latents_mean", mapped)

    def test_published_latent_statistics_and_dtype_aliases(self):
        self.assertEqual(len(IMAGE_VAE_LATENTS_MEAN), 24)
        self.assertEqual(len(IMAGE_VAE_LATENTS_STD), 24)
        self.assertTrue(all(value > 0.0 for value in IMAGE_VAE_LATENTS_STD))
        self.assertIs(_coerce_dtype("fp16"), torch.float16)
        self.assertIs(_coerce_dtype("bf16"), torch.bfloat16)
        self.assertIs(_coerce_dtype(torch.float32), torch.float32)
        with self.assertRaises(ValueError):
            _coerce_dtype("fp8")


if __name__ == "__main__":
    unittest.main()
