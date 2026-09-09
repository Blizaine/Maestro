"""Native VAE checkpoint layout must avoid copies and preserve decode math."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import torch
from mmgp import offload, quant_router
from models.minimax_h3 import checkpoint
from models.minimax_h3.video_vae import MiniMaxH3VideoViTDecoder3d, video_vae_offload_models


def compact_weights(decoder, quantized=False):
    """Export the tiny native decoder in the real compact file's namespaces."""
    state = {}
    for key, tensor in decoder.state_dict().items():
        key = "decoder." + key
        key = key.replace("decoder.proj_in.", "decoder.x_embedder.")
        key = key.replace(".attn.to_out.0.", ".attn.to_out.")
        key = key.replace(".ff.net.0.proj.", ".ff.w1.").replace(".ff.net.2.", ".ff.w2.")
        if quantized and key.endswith(".weight") and (".to_qkv." in key or ".ff.w1." in key):
            base = key.removesuffix(".weight")
            scale = tensor.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 127
            state[key] = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
            state[base + ".weight_scale"] = scale
            state[base + ".comfy_quant"] = torch.tensor(list(json.dumps({
                "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4,
            }).encode()), dtype=torch.uint8)
        else:
            state[key] = tensor
    return state


class H3NativeVaeMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        quant_router.register_handler("shared.qtypes.int8_convrot")

    def decoder(self, native):
        decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=4, patch_size=2, patch_size_t=2, num_layers=2,
            num_attention_heads=2, attention_head_dim=8, ffn_mult=2,
            native_checkpoint_layout=native,
        ).eval().requires_grad_(False)
        # Checkpoint values replace the zero-initialized residual scales.
        with torch.no_grad():
            for block in decoder.transformer_blocks:
                block.scale1.fill_(0.7)
                block.scale2.fill_(0.6)
        return decoder

    def test_native_adapter_keeps_weights_scales_and_descriptors_by_identity(self):
        source = compact_weights(self.decoder(True), quantized=True)
        source["decoder.mask_token"] = torch.zeros(1)
        source["latents_mean"] = torch.zeros(4)
        converted = checkpoint.preprocess_native_video_vae_state_dict(source)
        self.assertNotIn("decoder.mask_token", converted)
        self.assertNotIn("latents_mean", converted)
        for key, value in source.items():
            if key in {"decoder.mask_token", "latents_mean"}:
                continue
            self.assertIs(converted[checkpoint._rename_video_vae_key(key)], value)

    def compare_decode(self, quantized, dtype):
        torch.manual_seed(73)
        native, legacy = self.decoder(True), self.decoder(False)
        source = compact_weights(native, quantized=quantized)
        native_state = checkpoint.preprocess_native_video_vae_state_dict(source)
        with mock.patch.object(checkpoint, "VIDEO_VAE_HEADS", 2), mock.patch.object(checkpoint, "VIDEO_VAE_HEAD_DIM", 8):
            legacy_state = checkpoint.preprocess_video_vae_state_dict(source)
        for model, state in ((native, native_state), (legacy, legacy_state)):
            state = {k.removeprefix("decoder."): v for k, v in state.items()}
            offload.load_model_data(model, (state, None), default_dtype=dtype, verboseLevel=0)
        # The production VAE also runs its FP16 decoder under autocast.
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.float16, enabled=dtype == torch.float16):
            for shape in ((1, 4, 2, 2, 3), (2, 4, 1, 3, 2)):
                latent = torch.randn(shape).to(dtype)
                expected = legacy(latent)
                actual = native(latent)
                tolerance = 3e-3 if dtype == torch.float16 else 2e-5
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_full_decoder_matches_legacy_fp32_with_rope_and_multiple_batches(self):
        self.compare_decode(False, torch.float32)

    def test_full_decoder_matches_legacy_fp16(self):
        self.compare_decode(False, torch.float16)

    def test_convrot_decoder_matches_legacy_row_scales_and_gate_order(self):
        self.compare_decode(True, torch.float32)

    def test_convrot_decoder_matches_legacy_fp16(self):
        self.compare_decode(True, torch.float16)


class H3VaeResidencyTests(unittest.TestCase):
    def codec(self):
        codec = torch.nn.Module()
        for name in ("encoder", "quant_conv", "decoder", "post_quant_conv"):
            setattr(codec, name, torch.nn.Linear(2, 2).half())
        codec._model_dtype = torch.float16
        return codec

    def test_phases_share_original_weights_and_keep_codec_dtype(self):
        codec = self.codec()
        models = video_vae_offload_models(codec, device_mem_capacity=19 * 1024**3)
        self.assertEqual(set(models), {"vae", "video_encoder"})
        self.assertEqual(models["vae"]._budget, 0)
        self.assertFalse(hasattr(models["video_encoder"], "_budget"))
        for phase, names in (("vae", ("decoder", "post_quant_conv")),
                             ("video_encoder", ("encoder", "quant_conv"))):
            self.assertEqual(models[phase]._model_dtype, torch.float16)
            for name in names:
                self.assertIs(models[phase][name], getattr(codec, name))
        self.assertEqual(sum(p.numel() for model in models.values() for p in model.parameters()),
                         sum(p.numel() for p in codec.parameters()))

    def test_small_cards_keep_profile_streaming_budget(self):
        for capacity in (0, 6 * 1024**3, 8 * 1024**3, 10 * 1024**3 - 1):
            with self.subTest(capacity=capacity):
                models = video_vae_offload_models(self.codec(), device_mem_capacity=capacity)
                self.assertFalse(hasattr(models["vae"], "_budget"))

    def test_auto_policy_uses_gpu_capacity_and_handles_cpu_only(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
                torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=10 * 1024**3)):
            self.assertEqual(video_vae_offload_models(self.codec())["vae"]._budget, 0)
        with mock.patch.object(torch.cuda, "is_available", return_value=False), mock.patch.object(
                torch.cuda, "get_device_properties") as properties:
            self.assertFalse(hasattr(video_vae_offload_models(self.codec())["vae"], "_budget"))
            properties.assert_not_called()

    def test_h3_handler_exposes_both_codec_phases_to_mmgp(self):
        from models.minimax_h3.minimax_h3_handler import family_handler
        model = SimpleNamespace(vae=self.codec(), transformer=torch.nn.Linear(2, 2),
                                audio_vae=torch.nn.Linear(2, 2), viggle=True, audio_only=False)
        with mock.patch("models.minimax_h3.minimax_h3_main.MiniMaxH3Model", return_value=model), mock.patch.object(
                torch.cuda, "is_available", return_value=False):
            result, options = family_handler.load_model("unused.safetensors", dtype=torch.float16)
        self.assertIs(result, model)
        self.assertIs(options["pipe"]["vae"]["decoder"], model.vae.decoder)
        self.assertIs(options["pipe"]["video_encoder"]["encoder"], model.vae.encoder)
        self.assertNotIn("text_encoder", options["pipe"])


if __name__ == "__main__":
    unittest.main()
