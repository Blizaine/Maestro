"""Focused CPU tests for the optional MiniMax H3 learned x2 VAE path."""
from __future__ import annotations
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from models.minimax_h3 import vae_upsampler  # noqa: E402
from models.minimax_h3.minimax_h3_main import _normalized_video_to_cpu_uint8  # noqa: E402
from models.minimax_h3.video_vae import AutoencoderKLMiniMaxH3  # noqa: E402

class H3X2VaeDecodeTests(unittest.TestCase):

    def make_vae(self, out_channels: int) -> AutoencoderKLMiniMaxH3:
        torch.manual_seed(41)
        return AutoencoderKLMiniMaxH3(
            in_channels=3,
            out_channels=out_channels,
            latent_channels=2,
            block_out_channels=(8,),
            layers_per_block=1,
            spatial_downsample_factors=(2,),
            temporal_downsample_factors=(1,),
            norm_num_groups=2,
            decoder_num_layers=1,
            decoder_num_attention_heads=1,
            decoder_attention_head_dim=8,
            decoder_num_register_tokens=1,
            decoder_ffn_mult=2,
            clip_length=2,
            token_drop=1,
            latents_mean=(0.0, 0.0),
            latents_std=(1.0, 1.0),
            native_checkpoint_layout=True,
        ).eval().requires_grad_(False)
    @staticmethod

    def compact_checkpoint(model, namespaces):
        source = {}
        for key, tensor in model.state_dict().items():
            if not key.startswith(tuple(namespaces)):
                continue
            source_key = key.replace("decoder.proj_in.", "decoder.x_embedder.")
            source_key = source_key.replace(".attn.to_out.0.", ".attn.to_out.")
            source_key = source_key.replace(".ff.net.0.proj.", ".ff.w1.")
            source_key = source_key.replace(".ff.net.2.", ".ff.w2.")
            if source_key.startswith("encoder.down_blocks."):
                level, rest = source_key.removeprefix("encoder.down_blocks.").split(".", 1)
                rest = rest.replace("resnets.", "block.", 1)
                rest = rest.replace("conv_shortcut.", "nin_shortcut.", 1)
                rest = rest.replace("downsamplers.0.", "downsample.", 1)
                source_key = f"encoder.down.{level}.{rest}"
            source[source_key] = tensor
        return source

    def expected_uint8(self, decoded: torch.Tensor, upsample: bool) -> torch.Tensor:
        if upsample:
            batch, channels, frames, height, width = decoded.shape
            packed = decoded.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
            decoded = F.pixel_shuffle(packed, 2)
            decoded = decoded.reshape(batch, frames, 3, height * 2, width * 2).permute(0, 2, 1, 3, 4)
        mean = decoded.new_tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = decoded.new_tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1, 1)
        normalized = decoded.float().mul(std).add(mean).clamp(0, 1).mul(2).sub(1)
        return normalized.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)

    def test_streaming_decode_matches_full_blended_decode_and_shuffles_after_temporal_blend(self):
        for channels, upsample in ((3, False), (12, True)):
            with self.subTest(upsample=upsample):
                vae = self.make_vae(channels)
                latent = torch.randn(1, 2, 5, 4, 4)
                calls = []
                original_blend = vae._blend
                def record_blend(a, b, blend_extent, dim):
                    calls.append((a.shape[1], b.shape[1], dim))
                    return original_blend(a, b, blend_extent, dim)
                vae._blend = record_blend
                with torch.inference_mode():
                    raw = vae._decode(latent)
                    expected = self.expected_uint8(raw, upsample)
                    actual = vae.decode_to_cpu_uint8(latent)
                self.assertTrue(calls, "test input should crossfade temporal chunks")
                self.assertTrue(all(a_channels == channels and b_channels == channels for a_channels, b_channels, _ in calls))
                self.assertTrue(all(dim == -3 for _, _, dim in calls))
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertEqual(actual.device.type, "cpu")
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(vae._decoded_frame_count(latent.shape[2]), raw.shape[2])
                if upsample:
                    self.assertEqual(actual.shape[-2:], (raw.shape[-2] * 2, raw.shape[-1] * 2))

    def test_native_345_frame_timeline_keeps_exact_tail_after_streaming(self):
        torch.manual_seed(42)
        vae = AutoencoderKLMiniMaxH3(
            in_channels=3,
            out_channels=12,
            latent_channels=2,
            block_out_channels=(8,),
            layers_per_block=1,
            spatial_downsample_factors=(2,),
            temporal_downsample_factors=(2, 2, 1),
            norm_num_groups=2,
            decoder_num_layers=1,
            decoder_num_attention_heads=1,
            decoder_attention_head_dim=8,
            decoder_num_register_tokens=1,
            decoder_ffn_mult=2,
            clip_length=17,
            token_drop=3,
            latents_mean=(0.0, 0.0),
            latents_std=(1.0, 1.0),
            native_checkpoint_layout=True,
        ).eval().requires_grad_(False)
        latent = torch.randn(1, 2, 102, 2, 2)
        with torch.inference_mode():
            raw = vae._decode(latent)
            actual = vae.decode_to_cpu_uint8(latent)
        self.assertEqual(raw.shape[2], 345)
        self.assertEqual(vae._decoded_frame_count(102), 345)
        self.assertEqual(tuple(actual.shape), (1, 3, 345, 8, 8))
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertEqual(actual.device.type, "cpu")
        self.assertTrue(torch.equal(actual, self.expected_uint8(raw, True)))

    def test_x2_loader_fills_encoder_and_decoder_without_meta_parameters(self):
        from accelerate import init_empty_weights
        from models.minimax_h3.minimax_h3_main import _load_x2_video_vae_weights
        base = self.make_vae(3)
        x2 = self.make_vae(12)
        with init_empty_weights(include_buffers=False):
            target = self.make_vae(12)
        _load_x2_video_vae_weights(
            target,
            self.compact_checkpoint(base, ("encoder.", "quant_conv.", "post_quant_conv.")),
            self.compact_checkpoint(x2, ("decoder.",)),
        )
        self.assertFalse(any(parameter.is_meta for parameter in target.parameters()))
        self.assertTrue(torch.equal(target.encoder.conv_in.weight, base.encoder.conv_in.weight.to(torch.float16)))
        self.assertTrue(torch.equal(target.quant_conv.weight, base.quant_conv.weight.to(torch.float16)))
        self.assertTrue(torch.equal(target.post_quant_conv.weight, base.post_quant_conv.weight.to(torch.float16)))
        self.assertTrue(torch.equal(target.decoder.proj_out.weight, x2.decoder.proj_out.weight.to(torch.float16)))
        self.assertEqual(target.decoder.proj_out.weight.shape[0], 12 * 1 * 2 * 2)

    def test_decoder_cancellation_is_checked_between_temporal_clips(self):
        vae = self.make_vae(12)
        vae._interrupt = True
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            vae.decode_to_cpu_uint8(torch.randn(1, 2, 5, 4, 4))

class H3X2VaeRoutingTests(unittest.TestCase):
    def test_video_only_capability_and_optional_pinned_assets_with_direct_handler_load(self):
        handler_path = APP / "models" / "minimax_h3" / "minimax_h3_handler.py"
        spec = importlib.util.spec_from_file_location("h3_x2_direct_handler_test", handler_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        direct_handler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(direct_handler)

        model_def = direct_handler.family_handler.query_model_def("minimax_h3", {})
        self.assertEqual(model_def["vae_upsamplers"], {"h3_vae": [0]})
        audio_def = direct_handler.family_handler.query_model_def("minimax_h3_voice_audio", {})
        self.assertNotIn("vae_upsamplers", audio_def)

        normal_downloads = direct_handler.family_handler.query_model_files([], "minimax_h3", model_def)
        self.assertFalse(any("MiniMax-H3-X2-Detail-v1_int8_convrot.safetensors" in str(item) for item in normal_downloads))
        x2_downloads = direct_handler.family_handler.query_model_files(
            [], "minimax_h3", model_def, VAE_upsampling="h3_vae*2"
        )
        x2 = next(item for item in x2_downloads if item.get("revision") == vae_upsampler.X2_VAE_REVISION)
        self.assertEqual(x2["repoId"], vae_upsampler.X2_VAE_REPO)
        self.assertEqual(x2["fileList"], [[
            vae_upsampler.X2_VAE_FILE,
            vae_upsampler.X2_VAE_LICENSE_FILE,
            vae_upsampler.X2_VAE_NOTICE_FILE,
        ]])
        with self.assertRaisesRegex(ValueError, "does not support"):
            direct_handler.family_handler.query_model_files([], "minimax_h3", {}, VAE_upsampling="h3_vae*2")


class H3X2VaeHelperTests(unittest.TestCase):

    def test_compact_normalization_and_same_size_resize_do_not_mutate_float_input(self):
        source = torch.linspace(-1.0, 1.0, 3 * 1 * 2 * 2, dtype=torch.float32).reshape(3, 1, 2, 2)
        original = source.clone()
        expected = ((source.clamp(-1.0, 1.0) + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8)

        compact = _normalized_video_to_cpu_uint8(source)
        self.assertTrue(torch.equal(compact, expected))
        self.assertTrue(torch.equal(source, original))

        resized = vae_upsampler.resize_video_canvas_uint8(source, 2, 2)
        self.assertTrue(torch.equal(resized, expected))
        self.assertTrue(torch.equal(source, original))

    def test_video_canvas_resize_preserves_dtype_and_cpu_placement(self):
        source = torch.linspace(-1, 1, 3 * 9 * 4 * 6, dtype=torch.float32).reshape(3, 9, 4, 6)
        resized = vae_upsampler.resize_video_canvas(source, 8, 12)
        self.assertEqual(tuple(resized.shape), (3, 9, 8, 12))
        self.assertEqual(resized.dtype, source.dtype)
        self.assertEqual(resized.device.type, "cpu")
        self.assertTrue(torch.isfinite(resized).all())
        byte_source = torch.arange(3 * 3 * 2 * 2, dtype=torch.uint8).reshape(3, 3, 2, 2)
        byte_resized = vae_upsampler.resize_video_canvas(byte_source, 4, 4)
        self.assertEqual(byte_resized.dtype, torch.uint8)
        self.assertEqual(byte_resized.device.type, "cpu")
        compact = vae_upsampler.resize_video_canvas_uint8(source, 8, 12)
        expected = ((resized.clamp(-1, 1) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        self.assertTrue(torch.equal(compact, expected))

    def test_checkpoint_verifier_fails_closed_on_wrong_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "x2.safetensors")
            with open(filename, "wb") as handle:
                handle.write(b"abc")
            with mock.patch.object(vae_upsampler, "X2_VAE_SIZE", 4):
                with self.assertRaisesRegex(ValueError, "wrong size"):
                    vae_upsampler.verify_x2_vae_checkpoint(filename)
            with mock.patch.object(vae_upsampler, "X2_VAE_SIZE", 3), mock.patch.object(
                vae_upsampler, "X2_VAE_SHA256", "0" * 64
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    vae_upsampler.verify_x2_vae_checkpoint(filename)
            with mock.patch.object(vae_upsampler, "X2_VAE_SIZE", 3), mock.patch.object(
                vae_upsampler, "X2_VAE_SHA256", hashlib.sha256(b"abc").hexdigest()
            ):
                vae_upsampler.verify_x2_vae_checkpoint(filename)
if __name__ == "__main__":
    unittest.main()
