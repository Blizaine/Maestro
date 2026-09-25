"""CPU regressions for Qwen 2.1 workflows adapted from Wan2GP."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.qwen21.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from models.qwen21.pipeline_qwenimage21 import (
    QwenImage21Pipeline,
    VIGGLE_SIGMAS,
    _mask_to_latents,
    _outpainting_overlap,
    _outpainting_step_mask,
    _prepare_red_outpainting,
    qwen21_sampling_profile,
)
from models.qwen21.qwen21_handler import MODEL_TYPE, apply_acceleration_profile, family_handler
from models.qwen21.transformer_qwenimage21 import QwenImage21Transformer2DModel


def tiny_vae():
    return AutoencoderKLQwenImage21(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=[1, 1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True, True],
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    ).eval()


def tiny_transformer():
    return QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=8,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    ).eval()


def tiny_pipeline():
    class Processor(SimpleNamespace):
        pass

    class CpuPipeline(QwenImage21Pipeline):
        @property
        def _execution_device(self):
            return torch.device("cpu")

    processor = Processor(
        tokenizer=SimpleNamespace(encode=lambda _: [1]),
        apply_chat_template=lambda *_, **__: "unused by the test encoder",
    )
    transformer = tiny_transformer()
    transformer.forward = lambda *, hidden_states, **kwargs: (torch.zeros_like(hidden_states),)
    pipe = CpuPipeline(
        FlowMatchEulerDiscreteScheduler(), tiny_vae(), torch.nn.Linear(8, 8), processor, transformer
    )
    pipe.execution_device = torch.device("cpu")
    pipe.set_progress_bar_config(disable=True)

    def encode_prompt(*, image, prompt, prompt_embeds, prompt_embeds_mask, device, num_images_per_prompt):
        refs = image or []
        length = len(refs) * 2 + 1
        embeds = torch.zeros((1, length, 8), dtype=torch.float32, device=device)
        embeds_mask = torch.ones((1, length), dtype=torch.bool, device=device)
        image_mask = torch.tensor([[False, True] * len(refs) + [False]], dtype=torch.bool, device=device)
        return embeds, embeds_mask, image_mask

    pipe.encode_prompt = encode_prompt
    return pipe


def rgba_source():
    image = Image.new("RGBA", (64, 64), (42, 94, 135, 255))
    for x in range(32, 64):
        for y in range(64):
            image.putpixel((x, y), (190, 82, 35, 255))
    return image


def white_right_mask():
    mask = Image.new("L", (64, 64), 0)
    for x in range(32, 64):
        for y in range(64):
            mask.putpixel((x, y), 255)
    return mask


class ProfileAndMetadataTests(unittest.TestCase):
    def test_viggle_profiles_use_the_exact_sigma_nodes_and_reject_mismatches(self):
        expected = {
            "viggle_v01": (4, (1.0, 0.75, 0.5, 0.25)),
            "viggle_v02": (5, (1.0, 0.875, 0.75, 0.5, 0.25)),
            "viggle_v021": (6, (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)),
        }
        for solver, (steps, sigmas) in expected.items():
            with self.subTest(solver=solver):
                self.assertEqual(VIGGLE_SIGMAS[steps], sigmas)
                overrides, actual_sigmas = qwen21_sampling_profile(solver, steps)
                self.assertIsNone(overrides["shift_terminal"])
                self.assertEqual(actual_sigmas, sigmas)
        with self.assertRaisesRegex(ValueError, "requires exactly 4"):
            qwen21_sampling_profile("viggle_v01", 5)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            qwen21_sampling_profile("pruna", 8)

    def test_profile_hook_manages_only_viggle_and_is_idempotent(self):
        old_viggle = "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors"
        settings = {
            "sample_solver": "viggle_v021",
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "activated_loras": ["style_a.safetensors", old_viggle],
            "loras_multipliers": "0.35 0.4",
        }
        normalized = family_handler.apply_acceleration_profile(settings)
        self.assertEqual(normalized["sample_solver"], "viggle_v021")
        self.assertEqual(normalized["num_inference_steps"], 6)
        self.assertEqual(normalized["guidance_scale"], 1.0)
        self.assertEqual(normalized["activated_loras"][0], "style_a.safetensors")
        self.assertTrue(normalized["activated_loras"][1].endswith("6step-lora-r256.safetensors"))
        self.assertEqual(normalized["loras_multipliers"], "0.35 1")
        self.assertEqual(apply_acceleration_profile(normalized), normalized)

        base = apply_acceleration_profile({**normalized, "sample_solver": "default"})
        self.assertEqual(base["activated_loras"], ["style_a.safetensors"])
        self.assertEqual(base["loras_multipliers"], "0.35")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            apply_acceleration_profile({"sample_solver": "unknown"})

    def test_native_2k_and_edit_workflows_are_exposed_through_generic_metadata(self):
        model_def = family_handler.query_model_def(MODEL_TYPE, {})
        self.assertEqual(model_def["resolution_presets"]["2k"]["values"]["auto"], "auto_2k")
        self.assertEqual(model_def["auto_resolution_budgets"]["auto_2k"], 2048 * 2048)
        values = model_def["resolution_presets"]["2k"]["values"]
        self.assertEqual(values["1:1"], "2048x2048")
        self.assertEqual(values["16:9"], "2752x1536")
        self.assertEqual(values["3:2"], "2528x1696")
        self.assertEqual(values["2:3"], "1696x2528")
        self.assertEqual(model_def["resolution_preset_order"][-1], "2k")
        self.assertTrue(model_def["inpaint_support"])
        self.assertEqual(model_def["inpaint_video_prompt_type"], "VAG")
        self.assertTrue(model_def["image_ref_inpaint"])
        self.assertTrue(model_def["inpaint_with_image_ref"])
        self.assertEqual(model_def["runtime_custom_settings"], ["qwen21_kv_cache"])
        self.assertIn("PV", model_def["guide_preprocessing"]["selection"])
        self.assertIn("DV", model_def["guide_preprocessing"]["selection"])
        self.assertIn("EV", model_def["guide_preprocessing"]["selection"])
        self.assertEqual(model_def["model_modes"]["choices"][-1][1], 5)

    def test_reference_limit_counts_only_active_condition_images(self):
        self.assertIsNone(family_handler.validate_generative_settings(
            MODEL_TYPE, {}, {"video_prompt_type": "", "image_refs": [object()] * 12,
                             "image_guide": object()}
        ))
        self.assertIsNone(family_handler.validate_generative_settings(
            MODEL_TYPE, {}, {"video_prompt_type": "VI", "image_guide": "control.png",
                             "image_refs": [f"ref-{index}.png" for index in range(9)]}
        ))
        error = family_handler.validate_generative_settings(
            MODEL_TYPE, {}, {"video_prompt_type": "VI", "image_guide": "control.png",
                             "image_refs": [f"ref-{index}.png" for index in range(10)]}
        )
        self.assertIn("at most 9 additional references", error)
        self.assertIsNone(family_handler.validate_generative_settings(
            MODEL_TYPE, {}, {"video_prompt_type": "V", "image_mode": 2,
                             "image_guide": "masked-source.png", "image_mask": object()}
        ))


class MaskedPipelineTests(unittest.TestCase):
    @torch.inference_mode()
    def test_lanpaint_closure_keeps_cfg_caches_distinct_and_disabled_path_uncached(self):
        pipe = tiny_pipeline()
        calls = []

        def record_forward(*, hidden_states, encoder_hidden_states, kv_cache, kv_cache_mode, **kwargs):
            calls.append((float(encoder_hidden_states[0, 0, 0]), kv_cache, kv_cache_mode))
            return (torch.zeros_like(hidden_states),)

        pipe.transformer.forward = record_forward

        def encode_tagged_prompt(*, image, prompt, device, **kwargs):
            refs = image or []
            length = len(refs) * 2 + 1
            marker = -1.0 if prompt == "negative prompt" else 1.0
            embeds = torch.full((1, length, 8), marker, dtype=torch.float32, device=device)
            embeds_mask = torch.ones((1, length), dtype=torch.bool, device=device)
            image_mask = torch.tensor(
                [[False, True] * len(refs) + [False]], dtype=torch.bool, device=device
            )
            return embeds, embeds_mask, image_mask

        pipe.encode_prompt = encode_tagged_prompt
        image = rgba_source()
        cached = pipe(
            prompt="positive prompt",
            negative_prompt="negative prompt",
            true_cfg_scale=2.0,
            image=image,
            mask_image=white_right_mask(),
            inpaint_image=image,
            height=64,
            width=64,
            output_resolution=64,
            num_inference_steps=4,
            sample_solver="viggle_v01",
            model_mode=2,
            generator=torch.Generator("cpu").manual_seed(12),
            output_type="latent",
            use_kv_cache=True,
        ).images
        self.assertTrue(torch.isfinite(cached).all())

        positive_calls = [call for call in calls if call[0] > 0]
        negative_calls = [call for call in calls if call[0] < 0]
        self.assertTrue(positive_calls)
        self.assertTrue(negative_calls)
        positive_caches = {id(call[1]) for call in positive_calls}
        negative_caches = {id(call[1]) for call in negative_calls}
        self.assertEqual(len(positive_caches), 1)
        self.assertEqual(len(negative_caches), 1)
        self.assertTrue(positive_caches.isdisjoint(negative_caches))
        self.assertEqual({call[2] for call in positive_calls}, {"extract", "cached"})
        self.assertEqual({call[2] for call in negative_calls}, {"extract", "cached"})

        calls.clear()
        uncached = pipe(
            prompt="ordinary generation",
            height=64,
            width=64,
            output_resolution=64,
            num_inference_steps=2,
            generator=torch.Generator("cpu").manual_seed(13),
            output_type="latent",
            use_kv_cache=False,
        ).images
        self.assertTrue(torch.isfinite(uncached).all())
        self.assertTrue(calls)
        self.assertTrue(all(cache is None and mode is None for _, cache, mode in calls))

    @torch.inference_mode()
    def test_image_generation_accepts_inactive_engine_inpaint_mode(self):
        torch.manual_seed(30)
        pipe = tiny_pipeline()
        result = pipe(
            prompt="A quiet garden at sunrise.",
            height=64,
            width=64,
            output_resolution=64,
            num_inference_steps=2,
            model_mode=None,
            generator=torch.Generator("cpu").manual_seed(7),
            output_type="latent",
            use_kv_cache=False,
        ).images
        self.assertTrue(torch.isfinite(result).all())

    @torch.inference_mode()
    def test_reference_edit_accepts_inactive_engine_inpaint_mode(self):
        torch.manual_seed(30)
        pipe = tiny_pipeline()
        result = pipe(
            prompt="Keep the subject and add a soft sunset glow.",
            image=rgba_source(),
            height=64,
            width=64,
            output_resolution=64,
            num_inference_steps=2,
            model_mode=None,
            generator=torch.Generator("cpu").manual_seed(7),
            output_type="latent",
            use_kv_cache=False,
        ).images
        self.assertTrue(torch.isfinite(result).all())

    def test_active_mask_defaults_none_to_standard_mode(self):
        pipe = tiny_pipeline()
        image = rgba_source()
        results = []
        for model_mode in (None, 0):
            results.append(pipe(
                prompt="Edit the masked region.",
                image=image,
                mask_image=white_right_mask(),
                inpaint_image=image,
                height=64,
                width=64,
                output_resolution=64,
                num_inference_steps=2,
                model_mode=model_mode,
                generator=torch.Generator("cpu").manual_seed(11),
                output_type="latent",
                use_kv_cache=False,
            ).images)
        torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)

    def test_active_mask_rejects_explicit_invalid_inpaint_modes(self):
        pipe = tiny_pipeline()
        image = rgba_source()
        for model_mode in (1, 99):
            with self.subTest(model_mode=model_mode), self.assertRaisesRegex(ValueError, "for masked edits"):
                pipe(
                    prompt="Edit the masked region.",
                    image=image,
                    mask_image=white_right_mask(),
                    inpaint_image=image,
                    height=64,
                    width=64,
                    output_resolution=64,
                    num_inference_steps=2,
                    model_mode=model_mode,
                    output_type="latent",
                    use_kv_cache=False,
                )

    @torch.inference_mode()
    def test_masked_edit_anchors_unmasked_source_latents(self):
        torch.manual_seed(31)
        pipe = tiny_pipeline()
        image = rgba_source()
        result = pipe(
            prompt="Replace the masked half with a blue sky.",
            image=image,
            mask_image=white_right_mask(),
            inpaint_image=image,
            height=64,
            width=64,
            output_resolution=64,
            num_inference_steps=3,
            generator=torch.Generator("cpu").manual_seed(8),
            output_type="latent",
            use_kv_cache=False,
        ).images

        source_pixels = pipe.image_processor.preprocess(image.convert("RGBA"), width=64, height=64).unsqueeze(2)
        source_latents = pipe._encode_vae_image(source_pixels, torch.Generator("cpu").manual_seed(8))
        expected = source_latents[:, :, 0].flatten(2).transpose(1, 2)
        mask = _mask_to_latents(white_right_mask(), 64, 64, torch.device("cpu"), torch.float32)[0, :, 0]
        kept = mask.eq(0)
        torch.testing.assert_close(result[0, kept], expected[0, kept], atol=1e-5, rtol=1e-5)

    @torch.inference_mode()
    def test_lanpaint_remains_finite_after_the_singular_sigma_one_step(self):
        torch.manual_seed(32)
        pipe = tiny_pipeline()
        image = rgba_source()
        result = pipe(
            prompt="Replace the masked half with a blue sky.",
            image=image,
            mask_image=white_right_mask(),
            inpaint_image=image,
            height=64,
            width=64,
            output_resolution=64,
            num_inference_steps=4,
            sample_solver="viggle_v01",
            model_mode=2,
            generator=torch.Generator("cpu").manual_seed(9),
            output_type="latent",
            use_kv_cache=False,
        ).images
        self.assertTrue(torch.isfinite(result).all())

    @torch.inference_mode()
    def test_red_canvas_outpaint_preserves_the_aligned_source_crop(self):
        torch.manual_seed(33)
        pipe = tiny_pipeline()
        image = rgba_source()
        dims = (0, 0, 100, 100)
        result = pipe(
            prompt="Continue the scene beyond the image.",
            image=image,
            outpainting_dims=dims,
            height=64,
            width=64,
            output_resolution=64,
            num_inference_steps=3,
            generator=torch.Generator("cpu").manual_seed(10),
            output_type="latent",
            use_kv_cache=False,
            model_mode=None,
        ).images

        images, mask, prompt, location, red_canvas, user_mask = _prepare_red_outpainting(
            [image], 64, 64, dims, None, "Continue the scene."
        )
        self.assertEqual(location, (64, 32, 0, 32))
        self.assertEqual(red_canvas.getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(images[0].size, (32, 64))
        self.assertEqual(images[0].getpixel((0, 0)), image.getpixel((32, 0)))
        self.assertEqual(images[0].getpixel((31, 63)), image.getpixel((63, 63)))
        self.assertEqual(mask.getpixel((0, 0)), 255)
        self.assertIsNone(user_mask)
        self.assertIn("red paddings", prompt)
        self.assertTrue(torch.isfinite(result).all())

    def test_outpaint_mask_anchors_core_and_regenerates_two_latent_edge_cells(self):
        location = (128, 192, 0, 64)
        self.assertEqual(_outpainting_overlap(location, 256, 128), (0, 0, 128, 0))
        early = _outpainting_step_mask(
            location, 256, 128, torch.tensor(1.0), torch.device("cpu"), torch.float32
        ).reshape(1, 8, 16)
        late = _outpainting_step_mask(
            location, 256, 128, torch.tensor(0.5), torch.device("cpu"), torch.float32
        ).reshape(1, 8, 16)
        self.assertTrue(torch.equal(early[:, :, :4], torch.ones_like(early[:, :, :4])))
        self.assertTrue(torch.equal(early[:, :, 4:6], torch.ones_like(early[:, :, 4:6])))
        self.assertTrue(torch.equal(early[:, :, 6:16], torch.zeros_like(early[:, :, 6:16])))
        self.assertTrue(torch.all(late[:, :, 6:12] == 0.0625))
        self.assertTrue(torch.equal(late[:, :, 12:16], torch.zeros_like(late[:, :, 12:16])))


class TiledDecoderTests(unittest.TestCase):
    @torch.inference_mode()
    def test_cpu_uint8_tiles_match_the_float_tiled_decode(self):
        torch.manual_seed(34)
        vae = tiny_vae()
        vae.enable_tiling(
            tile_sample_min_height=64,
            tile_sample_min_width=64,
            tile_sample_stride_height=48,
            tile_sample_stride_width=48,
        )
        latents = torch.randn(1, 4, 1, 8, 8)
        float_tiles = vae.tiled_decode(latents.clone()).sample
        byte_tiles = vae.decode_to_cpu_uint8([latents.clone()])
        expected = float_tiles.float().add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        self.assertEqual(tuple(byte_tiles.shape), (1, 4, 1, 128, 128))
        self.assertEqual(byte_tiles.device.type, "cpu")
        torch.testing.assert_close(byte_tiles, expected, atol=0, rtol=0)

    @torch.inference_mode()
    def test_cpu_uint8_tiles_match_partial_nonsquare_float_tiles(self):
        class AmplifiedDecoder(torch.nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, *args, **kwargs):
                return self.decoder(*args, **kwargs) * 4

        torch.manual_seed(35)
        vae = tiny_vae()
        vae.decoder = AmplifiedDecoder(vae.decoder)
        vae.enable_tiling(
            tile_sample_min_height=128,
            tile_sample_min_width=128,
            tile_sample_stride_height=96,
            tile_sample_stride_width=96,
        )
        latents = torch.randn(1, 4, 1, 20, 26)
        float_tiles = vae.tiled_decode(latents.clone()).sample
        byte_tiles = vae.decode_to_cpu_uint8([latents.clone()])
        expected = float_tiles.float().add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        self.assertEqual(tuple(byte_tiles.shape), (1, 4, 1, 320, 416))
        torch.testing.assert_close(byte_tiles, expected, atol=0, rtol=0)

    @torch.inference_mode()
    def test_nonfinite_decode_fails_with_vae_precision_guidance(self):
        class NonFiniteDecoder(torch.nn.Module):
            def forward(self, latents, **kwargs):
                height, width = latents.shape[-2:]
                return latents.new_full(
                    (latents.shape[0], 4, 1, height * 16, width * 16), float("nan")
                )

        vae = tiny_vae()
        vae.decoder = NonFiniteDecoder()
        vae.enable_tiling(
            tile_sample_min_height=64,
            tile_sample_min_width=64,
            tile_sample_stride_height=48,
            tile_sample_stride_width=48,
        )
        with self.assertRaisesRegex(RuntimeError, "32-bit VAE"):
            vae.decode_to_cpu_uint8([torch.randn(1, 4, 1, 8, 8)])


if __name__ == "__main__":
    unittest.main()
