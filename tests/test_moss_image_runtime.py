"""CPU runtime contracts for the native MOSS Image model port."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.moss_image.moss_image_handler import REVISION, family_handler  # noqa: E402
from models.moss_image.pipeline import (  # noqa: E402
    _prompt_without_comment_lines,
    _unpack_vae_tile_size,
    sampling_schedule,
    validate_generation_inputs,
)
from models.moss_image.progress import GenerationAborted  # noqa: E402
from models.moss_image.text_encoder import TextEncoder, encode_prompt  # noqa: E402
from models.moss_image.transformer import MOSSImageTransformer, rope_embed  # noqa: E402
from models.moss_image.vae import MOSSImageVAE  # noqa: E402
from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402


class _TinyTokenizer:
    def __init__(self):
        self.template_args = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_args = (messages, kwargs)
        return "rendered prompt"

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[1, 2, 3, 4])


class _PointwiseDecoder(nn.Module):
    """Pixel-local decoder fixture; overlap blending must match full-frame output."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(32, 3, kernel_size=1)
        with torch.no_grad():
            self.conv_in.weight.zero_()
            self.conv_in.bias.zero_()
            for channel in range(3):
                self.conv_in.weight[channel, channel, 0, 0] = 0.5

    def forward(self, latents):
        return F.interpolate(self.conv_in(latents), scale_factor=8, mode="nearest")


def _tiny_vae():
    model = MOSSImageVAE.__new__(MOSSImageVAE)
    nn.Module.__init__(model)
    model.decoder = _PointwiseDecoder()
    model.bn_eps = 1e-4
    model.bn = nn.BatchNorm2d(128, eps=model.bn_eps, affine=False)
    model._interrupt = False
    return model


class MossImageRuntimeTests(unittest.TestCase):
    def test_sampling_schedule_is_native_nearest_training_grid_and_ends_at_zero(self):
        steps, shift = 7, 5.0
        schedule = sampling_schedule(steps, shift)
        training = torch.linspace(1, 0.001, 1000, dtype=torch.float32).flip(0)
        self.assertEqual(schedule.device.type, "cpu")
        self.assertEqual(schedule.dtype, torch.float32)
        self.assertEqual(schedule[0].item(), 1.0)
        self.assertEqual(schedule[-1].item(), 0.0)
        self.assertTrue(torch.all(schedule[:-1] > schedule[1:]))
        distances = (schedule[:-1, None] - training[None, :]).abs()
        self.assertTrue(torch.all(distances.min(dim=1).values < 1e-7))

        # The sigmas use the checkpoint's shifted flow grid before nearest
        # training-index snapping, which differs from a uniform Euler schedule.
        values = torch.linspace(1, shift / (1000 + shift - 1), steps, dtype=torch.float32)
        shifted = shift * values / (1 + (shift - 1) * values)
        source_indices = torch.stack([torch.argmin((training - sigma).abs()) for sigma in shifted])
        expected = torch.cat((torch.unique_consecutive(training[source_indices]), torch.zeros(1)))
        torch.testing.assert_close(schedule, expected)

    def test_generation_validation_rejects_unsupported_shapes_modes_and_settings(self):
        self.assertIsNone(validate_generation_inputs({"width": 1024, "height": 1024, "batch_size": 2}))
        cases = (
            ({"width": 1000, "height": 1024}, "multiples of 16"),
            ({"width": 4096, "height": 1040}, "4 megapixels"),
            ({"batch_size": 0}, "Batch size must be a positive integer"),
            ({"batch_size": 1.5}, "Batch size must be a positive integer"),
            ({"sampling_steps": True}, "Sampling steps must be a positive integer"),
            ({"flow_shift": float("nan")}, "Flow shift must be a positive finite number"),
            ({"sample_solver": "DPM++"}, "native flow-matching Euler sampler"),
            ({"image_mode": 0}, "text-to-image only"),
            ({"video_prompt_type": "I"}, "text-to-image only"),
            ({"input_ref_images": ["image.png"]}, "does not accept image, mask, video, or audio inputs"),
        )
        for settings, message in cases:
            with self.subTest(settings=settings):
                self.assertIn(message, validate_generation_inputs(settings))

    def test_prompt_comment_filter_preserves_multiline_layout_json(self):
        prompt = '{\n  # remove this whole comment\n  "caption": "Keep this line",\n  "items": []\n}'
        self.assertEqual(
            _prompt_without_comment_lines(prompt),
            '{\n  "caption": "Keep this line",\n  "items": []\n}',
        )

    def test_qwen35_cpu_conditioning_uses_local_decoder_attention_contract(self):
        config = Qwen3_5TextConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=4,
            layer_types=["full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 1],
            },
        )
        encoder = TextEncoder(config).eval()
        tokenizer = _TinyTokenizer()
        with torch.inference_mode():
            features = encode_prompt(encoder, tokenizer, "a small red bird")
        self.assertEqual(tuple(features.shape), (1, 4, 64))
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(tokenizer.template_args[0][0]["role"], "user")
        self.assertTrue(tokenizer.template_args[1]["enable_thinking"])

    def test_cfg_forward_skips_only_negative_branch_layer_22(self):
        torch.manual_seed(7)
        model = MOSSImageTransformer(
            dim=16,
            ffn_dim=32,
            num_heads=2,
            num_layers=23,
            text_dim=16,
            freq_dim=8,
            latent_channels=128,
        ).eval()
        latents = torch.randn(1, 128, 1, 1)
        contexts = [torch.randn(1, 2, 16), torch.randn(1, 2, 16)]
        ropes = [rope_embed(1, 1, 2, "cpu", head_dim=8) for _ in contexts]
        layer = model.layers[22]
        forward = layer.forward
        observed = []

        def record_skip(streams, layer_ropes, modulation, skip_branch=-1):
            observed.append(skip_branch)
            return forward(streams, layer_ropes, modulation, skip_branch)

        with patch.object(layer, "forward", side_effect=record_skip):
            with torch.inference_mode():
                outputs = model(latents, torch.tensor(500.0), contexts, ropes)
        self.assertEqual(observed, [0])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(outputs[0].shape), tuple(latents.shape))
        self.assertTrue(torch.isfinite(outputs[0]).all())
        self.assertTrue(torch.isfinite(outputs[1]).all())

    def test_tiled_vae_matches_full_frame_at_multiaxis_edges_and_returns_bytes(self):
        torch.manual_seed(11)
        vae = _tiny_vae()
        latents = torch.randn(2, 128, 6, 7) * 0.25
        with torch.inference_mode():
            full_frame = vae(latents, tile_size=0)
            tiled = vae(latents, tile_size=64)
        self.assertEqual(tuple(tiled.shape), (2, 3, 96, 112))
        self.assertEqual(tiled.dtype, torch.uint8)
        torch.testing.assert_close(tiled.float(), full_frame.float(), atol=1, rtol=0)

    def test_vae_tile_geometry_memory_tiers_and_abort_boundary(self):
        self.assertEqual(MOSSImageVAE.get_VAE_tile_size(0, 16_000, True), (True, 1024))
        self.assertEqual(MOSSImageVAE.get_VAE_tile_size(0, 8_000, True), (True, 512))
        self.assertEqual(MOSSImageVAE.get_VAE_tile_size(0, 7_999, True), (True, 256))
        self.assertEqual(_unpack_vae_tile_size((False, 512)), (False, 0))
        self.assertEqual(_unpack_vae_tile_size((True, 256)), (True, 256))

        vae = _tiny_vae()
        vae._interrupt = True
        with self.assertRaises(GenerationAborted):
            vae(torch.zeros(1, 128, 2, 2), tile_size=32)

    def test_registration_pins_public_assets_and_keeps_only_native_solver(self):
        self.assertEqual(REVISION, "0a508e986e920c8051b99df98d27329742fc3d0e")
        model_def = family_handler.query_model_def("moss_image", {})
        self.assertFalse(model_def["compile"])
        self.assertFalse(model_def["auto_quantize"])
        self.assertEqual(model_def["sample_solvers"], [("Euler", "default")])
        self.assertTrue(model_def["moss_image_prompt_helper"])
        self.assertTrue(any("tokenizer.json" in name for name in model_def["required_model_assets"]))
        self.assertTrue(any("flux2_small_decoder_bf16.safetensors" in name for name in model_def["required_model_assets"]))


if __name__ == "__main__":
    unittest.main()
