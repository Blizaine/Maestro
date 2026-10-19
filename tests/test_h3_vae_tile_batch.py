"""Bounded CUDA spatial tile batching for the MiniMax H3 VAE."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import torch  # noqa: E402
from models.minimax_h3 import video_vae  # noqa: E402
from models.minimax_h3.video_vae import AutoencoderKLMiniMaxH3  # noqa: E402


GiB = 1024**3


class H3VaeTileBatchPolicyTests(unittest.TestCase):
    def choose(self, *, device, batch_size=1, tile_height=256, tile_width=256, num_tiles=2):
        return video_vae._h3_vae_decode_tile_batch_size(
            device=device,
            batch_size=batch_size,
            tile_height=tile_height,
            tile_width=tile_width,
            num_tiles=num_tiles,
        )

    def test_cpu_batch_and_oversized_or_single_tile_stay_serial_without_cuda_queries(self):
        with mock.patch.object(torch.cuda, "get_device_properties") as properties, mock.patch.object(
            torch.cuda, "mem_get_info"
        ) as memory_info:
            cases = (
                {"device": torch.device("cpu")},
                {"device": torch.device("cuda:0"), "batch_size": 2},
                {"device": torch.device("cuda:0"), "tile_height": 257},
                {"device": torch.device("cuda:0"), "tile_width": 257},
                {"device": torch.device("cuda:0"), "num_tiles": 1},
            )
            for kwargs in cases:
                with self.subTest(kwargs=kwargs):
                    self.assertEqual(self.choose(**kwargs), 1)
            properties.assert_not_called()
            memory_info.assert_not_called()

    def test_total_memory_and_free_memory_thresholds_are_inclusive(self):
        device = torch.device("cuda:0")
        with mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(total_memory=10 * GiB - 1),
        ), mock.patch.object(torch.cuda, "mem_get_info") as memory_info:
            self.assertEqual(self.choose(device=device), 1)
            memory_info.assert_not_called()

        with mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(total_memory=10 * GiB),
        ), mock.patch.object(torch.cuda, "mem_get_info", return_value=(2 * GiB - 1, 10 * GiB)):
            self.assertEqual(self.choose(device=device), 1)

        with mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(total_memory=10 * GiB),
        ), mock.patch.object(torch.cuda, "mem_get_info", return_value=(2 * GiB, 10 * GiB)):
            self.assertEqual(self.choose(device=device), 2)

    def test_unavailable_cuda_memory_probe_falls_back_to_serial(self):
        with mock.patch.object(torch.cuda, "get_device_properties", side_effect=RuntimeError("unavailable")):
            self.assertEqual(self.choose(device=torch.device("cuda:0")), 1)


class H3VaeTileBatchDecodeTests(unittest.TestCase):
    def make_vae(self):
        torch.manual_seed(42)
        vae = AutoencoderKLMiniMaxH3(
            in_channels=2,
            out_channels=2,
            latent_channels=2,
            block_out_channels=(4,),
            layers_per_block=1,
            spatial_downsample_factors=(2,),
            temporal_downsample_factors=(1,),
            norm_num_groups=1,
            decoder_num_layers=1,
            decoder_num_attention_heads=1,
            decoder_attention_head_dim=8,
            decoder_num_register_tokens=1,
            decoder_ffn_mult=2,
            clip_length=2,
            token_drop=0,
            latents_mean=(0.0, 0.0),
            latents_std=(1.0, 1.0),
        ).eval().requires_grad_(False)
        with torch.no_grad():
            for block in vae.decoder.transformer_blocks:
                block.scale1.fill_(0.7)
                block.scale2.fill_(0.6)
        vae.enable_tiling(
            tile_sample_min_height=8,
            tile_sample_min_width=8,
            tile_sample_min_overlap_height=6,
            tile_sample_min_overlap_width=2,
        )
        return vae

    def test_real_decoder_pairing_matches_serial_corners_and_reduces_calls(self):
        vae = self.make_vae()
        # 16x18 pixels on an 8x8 tile grid gives 5x3 tiles. The vertical
        # overlaps are six pixels, including triple-covered corner regions;
        # the horizontal span is not divisible by the tile width.
        y_indices, _, y_overlaps = vae._split_tiles(16, 8, 6)
        x_indices, _, x_overlaps = vae._split_tiles(18, 8, 2)
        self.assertEqual(len(y_indices), 5)
        self.assertEqual(len(x_indices), 3)
        self.assertEqual(y_overlaps, [6, 6, 6, 6])
        self.assertEqual(x_overlaps, [4, 2])

        latent = torch.randn(1, 2, 2, 8, 9)
        serial_calls = []
        serial_hook = vae.decoder.register_forward_hook(
            lambda _module, inputs, _output: serial_calls.append(inputs[0].shape[0])
        )
        with torch.inference_mode():
            serial = vae._decode_clip(latent)
        serial_hook.remove()

        paired_calls = []
        paired_hook = vae.decoder.register_forward_hook(
            lambda _module, inputs, _output: paired_calls.append(inputs[0].shape[0])
        )
        with mock.patch.object(video_vae, "_h3_vae_decode_tile_batch_size", return_value=2), mock.patch(
            "builtins.print"
        ) as diagnostic:
            with torch.inference_mode():
                paired = vae._decode_clip(latent)
            diagnostic.assert_called_once_with("[MiniMax H3 VAE] Tile batch size 2")
        paired_hook.remove()

        with mock.patch.object(video_vae, "_h3_vae_decode_tile_batch_size", return_value=2), mock.patch(
            "builtins.print"
        ) as diagnostic:
            with torch.inference_mode():
                vae._decode_clip(latent)
            diagnostic.assert_not_called()

        torch.testing.assert_close(paired, serial, rtol=1e-5, atol=1e-5)
        self.assertEqual(serial_calls, [1] * 15)
        self.assertEqual(paired_calls, [2, 1] * 5)
        self.assertEqual(len(paired_calls), 10)


if __name__ == "__main__":
    unittest.main()
