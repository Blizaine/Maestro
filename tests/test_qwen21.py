"""CPU regressions for the new architecture, reference flow and job adapter."""

import argparse
import ast
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.qwen21.qwen21_handler import MODEL_TYPE, family_handler
from models.qwen21.transformer_qwenimage21 import QwenImage21KVCache, QwenImage21Transformer2DModel
from models.qwen21.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from models.qwen21.pipeline_qwenimage21 import QwenImage21Pipeline
from models.qwen21.runtime import model_factory, GenerationCancelled
from services.enhance_guides import get_enhance_guide


def tiny_transformer():
    return QwenImage21Transformer2DModel(in_channels=4, out_channels=4, num_layers=2,
        attention_head_dim=16, num_attention_heads=2, context_in_dim=8,
        mlp_ratio=2, axes_dims_rope=(4, 6, 6)).eval()


def tiny_vae():
    return AutoencoderKLQwenImage21(base_dim=4, decoder_base_dim=4, z_dim=4,
        dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1, attn_scales=[],
        temperal_downsample=[False, True, True, True], latents_mean=[0.] * 4,
        latents_std=[1.] * 4).eval()


class ArchitectureTests(unittest.TestCase):
    def test_queue_reference_loading_preserves_alpha_only_when_supported(self):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        namespace = {"Image": Image, "has_image_file_extension": lambda name: name.endswith(".png")}
        for name in ("convert_image", "clean_image_list"):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            exec(compile(ast.Module(body=[function], type_ignores=[]), "wgp.py", "exec"), namespace)
        rgba = Image.new("RGBA", (2, 2), (30, 50, 70, 90))
        load = namespace["clean_image_list"]
        self.assertEqual(load([rgba])[0].mode, "RGB")
        actual = load([(rgba, "reference")], preserve_alpha=True)[0]
        self.assertEqual(actual.mode, "RGBA")
        self.assertEqual(actual.getpixel((0, 0)), (30, 50, 70, 90))
        generate = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "generate_video")
        start = next(i for i, node in enumerate(generate.body) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "original_image_refs")
        namespace.update(image_refs=[actual], model_def={"preserve_image_ref_alpha": True})
        exec(compile(ast.Module(body=generate.body[start:start + 3], type_ignores=[]), "wgp.py", "exec"), namespace)
        self.assertEqual(namespace["original_image_refs"][0].mode, "RGBA")
        self.assertEqual(namespace["original_image_refs"][0].getpixel((0, 0))[3], 90)
        self.assertEqual(namespace["image_refs"][0].mode, "RGB")

    @torch.inference_mode()
    def test_prefix_cache_matches_full_reference_forward(self):
        torch.manual_seed(12)
        model = tiny_transformer()
        # Text, one condition-image slot (four latent tokens), more text,
        # then the target image slot. Tests the actual interleaved layout.
        args = dict(hidden_states=torch.randn(1, 8, 4), encoder_hidden_states=torch.randn(1, 5, 8),
            encoder_hidden_states_mask=None, img_shapes=[[(1, 2, 2), (1, 2, 2)]],
            img_mask=torch.tensor([[False, True, False, False, False, True]]))
        cache = QwenImage21KVCache(2)
        model(**args, timestep=torch.tensor([0.9]), kv_cache=cache, kv_cache_mode="extract")
        full = model(**args, timestep=torch.tensor([0.3])).sample[:, -4:]
        cached = model(**args, timestep=torch.tensor([0.3]), kv_cache=cache, kv_cache_mode="cached").sample[:, -4:]
        torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)

    @torch.inference_mode()
    def test_rgba_vae_tiled_roundtrip_shapes(self):
        vae = tiny_vae()
        vae.enable_tiling(tile_sample_min_height=48, tile_sample_min_width=48,
                          tile_sample_stride_height=32, tile_sample_stride_width=32)
        sample = torch.rand(1, 4, 1, 64, 64)
        encoded = vae.encode(sample).latent_dist.mode()
        self.assertEqual(tuple(encoded.shape), (1, 4, 1, 4, 4))
        result = vae.decode(encoded).sample
        self.assertEqual(tuple(result.shape), tuple(sample.shape))
        self.assertTrue(torch.isfinite(result).all())

    @torch.inference_mode()
    def test_bf16_references_can_use_the_fp32_vae(self):
        pipe = object.__new__(QwenImage21Pipeline)
        pipe.vae = tiny_vae()
        pipe.latent_channels, pipe.vae_scale_factor = 4, 16
        latents, references = pipe.prepare_latents(
            [torch.rand(1, 4, 1, 64, 64).bfloat16()], 1, 4, 64, 64,
            torch.bfloat16, torch.device("cpu"), torch.Generator("cpu").manual_seed(1), None)
        self.assertEqual(references.dtype, torch.bfloat16)
        self.assertEqual(latents.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(references).all())

    def test_registration_and_reference_limit(self):
        settings = {}
        family_handler.update_default_settings(MODEL_TYPE, {}, settings)
        self.assertEqual(settings["num_inference_steps"], 40)
        self.assertEqual(settings["guidance_scale"], 1)
        self.assertEqual(settings["video_prompt_type"], "I")
        self.assertIsNone(family_handler.validate_generative_settings(MODEL_TYPE, {}, {"image_refs": [1] * 10}))
        self.assertIn("10", family_handler.validate_generative_settings(MODEL_TYPE, {}, {"image_refs": [1] * 11}))
        self.assertTrue(family_handler.get_lora_dir(MODEL_TYPE, SimpleNamespace(), "loras").endswith("qwen21"))
        tree = ast.parse((ROOT / "app/models/qwen/qwen_handler.py").read_text(encoding="utf-8"))
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        register = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                        and node.name == "register_lora_cli_args")
        register.decorator_list = []
        namespace = {"os": os}
        exec(compile(ast.Module(body=[register], type_ignores=[]), "qwen_handler.py", "exec"), namespace)
        parser = argparse.ArgumentParser()
        namespace["register_lora_cli_args"](parser, "loras")
        args = parser.parse_args(["--lora-dir-qwen", "old20b", "--lora-dir-qwen21", "new7b"])
        self.assertEqual(args.lora_dir_qwen, "old20b")
        self.assertEqual(family_handler.get_lora_dir(MODEL_TYPE, args, "loras"), "new7b")

    def test_model_specific_enhancement_preserves_edit_roles(self):
        with patch.dict(sys.modules, {"wgp": SimpleNamespace(get_model_def=lambda *_: {})}):
            text = get_enhance_guide(MODEL_TYPE, "image", has_images=False)
            edit = get_enhance_guide(MODEL_TYPE, "image", has_images=True)
        self.assertIn("Qwen Image 2.1", text)
        self.assertIn("<image1>", edit)
        self.assertIn("alpha channel", edit)
        self.assertNotIn("30-60 words", text)

    def test_pre_norm_encoder_does_not_retain_all_layers(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = torch.nn.Module()
                self.language_model.norm = torch.nn.LayerNorm(8)
                self.seen = None
            def forward(self, **kw):
                self.seen = kw
                self.language_model.norm(torch.full((1, 3, 8), 2.0))
        class Inputs(dict):
            def __getattr__(self, name):
                if name not in self:
                    raise AttributeError(name)
                return self[name]
            def to(self, device):
                return self
        pipe = object.__new__(QwenImage21Pipeline)
        pipe.text_encoder = Encoder()
        pipe.processor = lambda **_: Inputs(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))
        pipe.prompt_template_t2i = "{}"
        pipe._drop_idx, pipe._img_token_id = 1, 99
        embeds, _, _ = pipe._get_qwen_prompt_embeds(["cat"], device=torch.device("cpu"))
        torch.testing.assert_close(embeds, torch.full((1, 2, 8), 2.0))
        self.assertFalse(pipe.text_encoder.seen["output_hidden_states"])
        self.assertFalse(pipe.text_encoder.seen["use_cache"])
        self.assertFalse(pipe.text_encoder.language_model.norm._forward_hooks)


class JobAdapterTests(unittest.TestCase):
    def test_handler_accepts_the_shared_loader_call(self):
        # wgp.load_models supplies four positional arguments, then the encoder
        # filename by keyword. Test the real call boundary, not just runtime.
        definition = {"architecture": MODEL_TYPE}
        loaded = SimpleNamespace(transformer=object(), text_encoder=object(), vae=object())
        with patch("models.qwen21.runtime.model_factory", return_value=loaded) as factory:
            runtime, pipe = family_handler.load_model(
                ["transformer.safetensors"], MODEL_TYPE, MODEL_TYPE, definition,
                text_encoder_filename="encoder.safetensors", VAE_dtype=torch.float32,
                dtype=torch.bfloat16, quantizeTransformer=False, profile=4)
        factory.assert_called_once_with(
            ["transformer.safetensors"], "encoder.safetensors", model_def=definition,
            VAE_dtype=torch.float32, save_quantized=False, model_type=MODEL_TYPE)
        self.assertIs(runtime, loaded)
        self.assertEqual(pipe, vars(loaded))

    def runtime(self, behavior=None):
        instance = model_factory.__new__(model_factory)
        instance.device = torch.device("cpu")
        instance._abort = False
        instance.transformer = torch.nn.Linear(2, 2)
        instance.vae = SimpleNamespace(enable_tiling=Mock(), enable_slicing=Mock(), clear_cache=Mock(),
                                      decoder=torch.nn.Linear(2, 2))
        def produce(**kw):
            if behavior:
                return behavior(kw)
            kw["callback_on_step_end"](None, 0, 1, {})
            return SimpleNamespace(images=torch.full((kw["num_images_per_prompt"], 4, 32, 32), 0.5))
        instance.pipeline = Mock(side_effect=produce)
        return instance

    def test_reference_order_alpha_cfg_and_batch_output(self):
        instance = self.runtime()
        ref1, ref2 = Image.new("RGBA", (32, 32)), Image.new("RGB", (32, 32))
        progress = Mock()
        output = instance.generate(input_prompt="Combine", seed=7, video_prompt_type="I", batch_size=2,
                                   original_input_ref_images=[ref1, ref2], callback=progress, guide_scale=3)
        call = instance.pipeline.call_args.kwargs
        self.assertEqual(call["image"], [ref1, ref2])
        self.assertEqual(call["image"][0].mode, "RGBA")
        self.assertEqual(call["true_cfg_scale"], 3)
        self.assertEqual(call["negative_prompt"], " ")
        self.assertEqual(call["generator"].initial_seed(), 7)
        self.assertEqual(tuple(output.shape), (4, 2, 32, 32))
        self.assertEqual(output.dtype, torch.uint8)
        self.assertFalse(instance.transformer._forward_pre_hooks)
        instance.vae.clear_cache.assert_called_once()

    def test_disabled_references_and_default_cfg(self):
        instance = self.runtime()
        instance.generate(input_prompt="Cat", input_ref_images=[Image.new("RGB", (32, 32))])
        self.assertIsNone(instance.pipeline.call_args.kwargs["image"])
        self.assertIsNone(instance.pipeline.call_args.kwargs["negative_prompt"])

    def test_cancellation_does_not_save_partial_image(self):
        instance = self.runtime()
        def cancel(pipe, step, preview):
            instance._interrupt = True
        self.assertIsNone(instance.generate(input_prompt="Cat", callback=cancel))
        self.assertFalse(instance.transformer._forward_pre_hooks)
        instance.vae.clear_cache.assert_called_once()

    def test_exception_cleans_job_hooks(self):
        def fail(_):
            raise RuntimeError("generation error")
        instance = self.runtime(fail)
        with self.assertRaisesRegex(RuntimeError, "generation error"):
            instance.generate(input_prompt="Cat")
        self.assertFalse(instance.vae.decoder._forward_pre_hooks)
        instance.vae.clear_cache.assert_called_once()

    def test_over_limit_not_silently_truncated(self):
        instance = self.runtime()
        with self.assertRaisesRegex(ValueError, "10"):
            instance.generate(input_prompt="Combine", video_prompt_type="I", input_ref_images=[Image.new("RGB", (32, 32))] * 11)
        instance.pipeline.assert_not_called()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
