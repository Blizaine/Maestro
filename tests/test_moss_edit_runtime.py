"""CPU checks for MOSS Image reference editing, local KV reuse, and VAE inputs."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.moss_image.edit_runtime import (  # noqa: E402
    GIB,
    ReferenceCachePlan,
    _memory_budgets,
    cache_mode_log,
    edit_reference_size,
    fit_reference_image,
    has_step_varying_loras,
    is_hq_edit_variant,
    plan_reference_cache,
    validate_reference_images,
)
from models.moss_image.pipeline import MOSSImagePipeline, validate_generation_inputs  # noqa: E402
from models.moss_image.progress import GenerationAborted  # noqa: E402
from models.moss_image.transformer import MOSSImageTransformer, rope_embed  # noqa: E402
from models.moss_image.vae import MOSSImageVAE  # noqa: E402
from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import (  # noqa: E402
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from shared.reference_cache import ReferenceKVLayerCache  # noqa: E402


def _tiny_transformer(layers=3):
    model = MOSSImageTransformer(
        dim=16,
        ffn_dim=32,
        num_heads=2,
        num_layers=layers,
        text_dim=16,
        freq_dim=8,
        latent_channels=128,
    ).eval()
    model._force_device = "cpu"
    return model


def _body_ropes(contexts):
    return [rope_embed(1, 1, context.shape[1], "cpu", head_dim=8) for context in contexts]


def _sft_reference_rope(references):
    return torch.cat([
        rope_embed(
            ref.shape[-2], ref.shape[-1], 0, "cpu", head_dim=8,
            reference_index=index + 1, image_text_position=0,
        )
        for index, ref in enumerate(references)
    ], dim=1)


def _hq_reference_ropes(references, contexts):
    return [
        torch.cat([
            rope_embed(
                ref.shape[-2], ref.shape[-1], 0, "cpu", head_dim=8,
                reference_index=index + 1, image_text_position=context.shape[1],
            )
            for index, ref in enumerate(references)
        ], dim=1)
        for context in contexts
    ]


def _run_transformer(model, latents, contexts, references, reference_rope, *, cache=None, skip_layer=-1):
    with torch.inference_mode():
        return model(
            latents,
            torch.tensor(500.0),
            contexts,
            _body_ropes(contexts),
            skip_layer=skip_layer,
            reference_latents=references,
            reference_rope=reference_rope,
            kv_cache=cache,
        )


class _FixedTextCache:
    def __init__(self, contexts):
        self.contexts = contexts

    def encode(self, _encode, prompts, device=None):
        return [context.to(device) for context in self.contexts[:len(prompts)]]


class _TinyImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, kernel_size=1)

    def forward(self, pixels):
        return F.avg_pool2d(self.conv_in(pixels), kernel_size=8)


class _TinyEditingVAE(nn.Module):
    _encode_tile_starts = staticmethod(MOSSImageVAE._encode_tile_starts)
    _patch_mean = staticmethod(MOSSImageVAE._patch_mean)

    def __init__(self):
        super().__init__()
        self.encoder = _TinyImageEncoder()
        self.bn_eps = 1e-4
        self.bn = nn.BatchNorm2d(128, affine=False)
        self.bn.running_mean.zero_()
        self.bn.running_var.fill_(1)
        self._interrupt = False

    def encode(self, pixels, tile_size=384):
        return MOSSImageVAE.encode(self, pixels, tile_size=tile_size)

    def forward(self, latents, tile_size=0):
        del tile_size
        return latents.float()


def _pipeline(editing=False, variant=None, model_def=None):
    transformer = _tiny_transformer(layers=2)
    vae = _TinyEditingVAE()
    vision_kwargs = {}
    if editing and is_hq_edit_variant(variant):
        vision_kwargs = {
            "vision_encoder": nn.Identity(),
            "image_processor": object(),
            "vision_config": Qwen3_5Config(vision_config=Qwen3_5VisionConfig()),
            "vision_token_ids": {"image": 248056, "vision_start": 248053, "vision_end": 248054},
        }
    pipeline = MOSSImagePipeline(
        transformer,
        SimpleNamespace(_interrupt=False),
        tokenizer=None,
        vae=vae,
        editing=editing,
        edit_variant=variant,
        model_def=model_def or {},
        **vision_kwargs,
    )
    pipeline.text_encoder_cache = _FixedTextCache([
        torch.randn(1, 3, 16),
        torch.randn(1, 5, 16),
    ])
    return pipeline


class MossEditRuntimeTests(unittest.TestCase):
    def test_reference_validation_fit_and_model_size(self):
        image = Image.new("RGB", (600, 300), "navy")
        self.assertIsNone(validate_reference_images([image], editing=True))
        self.assertIn("one to four", validate_reference_images([], editing=True))
        self.assertIn("PIL", validate_reference_images(["image.png"], editing=True))
        self.assertIn("does not accept", validate_reference_images([image], editing=False))

        fitted = fit_reference_image(image, 256)
        self.assertEqual(fitted.size, (256, 128))
        self.assertEqual(edit_reference_size({"moss_edit_reference_size": "1024"}), 1024)
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            edit_reference_size({"moss_edit_reference_size": 255})
        self.assertTrue(is_hq_edit_variant("hq_v5"))
        self.assertFalse(is_hq_edit_variant("sft"))

    def test_generation_validation_accepts_ordered_edits_and_rejects_other_modalities(self):
        image = Image.new("RGB", (32, 16), "white")
        model_def = {"moss_edit": True, "moss_edit_variant": "hq_v5"}
        valid = {
            "image_mode": 1,
            "video_prompt_type": "I",
            "input_ref_images": [image],
            "width": 32,
            "height": 16,
        }
        self.assertIsNone(validate_generation_inputs(valid, model_def=model_def))
        self.assertIn("requires video_prompt_type I", validate_generation_inputs(
            {**valid, "video_prompt_type": ""}, model_def=model_def
        ))
        for key in ("image_mask", "video_source", "audio_guide", "video_guide", "image_guide"):
            with self.subTest(key=key):
                self.assertIn("ordered image references only", validate_generation_inputs(
                    {**valid, key: "unsupported"}, model_def=model_def
                ))
        self.assertIn(
            "does not accept image, mask, video, or audio inputs",
            validate_generation_inputs({"input_ref_images": [image]}),
        )
        self.assertIn(
            "Unknown reference KV cache mode",
            validate_generation_inputs(
                {**valid, "custom_settings": {"moss_image_kv_cache": "Not a cache mode"}},
                model_def=model_def,
            ),
        )

    def test_generate_keeps_studio_image_mode_authoritative_over_stale_model_mode(self):
        reference = Image.new("RGB", (16, 16), "white")
        for stale_mode in (0, 2):
            self.assertIsNone(validate_generation_inputs({
                "model_mode": stale_mode, "image_refs": [reference], "video_prompt_type": "I",
            }, editing=True), "An unrelated selector must not replace the default image mode")

        # WGP's production generate_video call passes both generic selectors.
        # MOSS must ignore model_mode, which may be left by a different family,
        # while still requiring the actual Studio image_mode to be 1.
        for stale_mode in (0, 2):
            with self.subTest(stale_model_mode=stale_mode):
                pipeline = _pipeline(
                    editing=True,
                    variant="sft",
                    model_def={"moss_edit_reference_size": 16},
                )

                def stop_after_mode_validation(*_args, **_kwargs):
                    raise RuntimeError("valid image mode reached generation callback")

                with self.assertRaisesRegex(RuntimeError, "valid image mode reached"):
                    pipeline.generate(
                        input_prompt="replace the hands",
                        n_prompt="",
                        sampling_steps=2,
                        width=16,
                        height=16,
                        guide_scale=1,
                        input_ref_images=[reference],
                        image_mode=1,
                        model_mode=stale_mode,
                        video_prompt_type="I",
                        VAE_tile_size=False,
                        callback=stop_after_mode_validation,
                    )

        for invalid_image_mode in (0, 2):
            with self.subTest(invalid_image_mode=invalid_image_mode):
                pipeline = _pipeline(
                    editing=True,
                    variant="sft",
                    model_def={"moss_edit_reference_size": 16},
                )
                with self.assertRaisesRegex(ValueError, "supports image generation mode 1 only"):
                    pipeline.generate(
                        input_prompt="replace the hands",
                        n_prompt="",
                        sampling_steps=2,
                        width=16,
                        height=16,
                        guide_scale=1,
                        input_ref_images=[reference],
                        image_mode=invalid_image_mode,
                        model_mode=1,
                        video_prompt_type="I",
                        VAE_tile_size=False,
                    )

    def test_managed_adapter_is_required_once_and_cannot_cross_variants(self):
        sft = _pipeline(editing=True, variant="sft")
        hq = _pipeline(editing=True, variant="hq_v5")
        base = _pipeline()
        self.assertIsNone(sft.validate_loras([
            r"C:\loras\MOSS_IMAGE_EDIT_LORA_BF16.SAFETENSORS", "style.safetensors",
        ]))
        self.assertIsNone(hq.validate_loras([
            r"C:\loras\ARMV5.SAFETENSORS", "style.safetensors",
        ]))
        self.assertIsNone(base.validate_loras(["style.safetensors"]))
        with self.assertRaisesRegex(ValueError, "requires its built-in"):
            sft.validate_loras([])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            sft.validate_loras([
                "moss_image_edit_lora_bf16.safetensors",
                "C:/loras/MOSS_IMAGE_EDIT_LORA_BF16.SAFETENSORS",
            ])
        with self.assertRaisesRegex(ValueError, "only the built-in"):
            sft.validate_loras([
                "moss_image_edit_lora_bf16.safetensors", "armV5.safetensors",
            ])
        with self.assertRaisesRegex(ValueError, "cannot use"):
            base.validate_loras(["armV5.safetensors"])

    def test_sft_shared_reference_cache_matches_uncached_and_clears(self):
        torch.manual_seed(73)
        model = _tiny_transformer(layers=3)
        contexts = [torch.randn(1, 2, 16), torch.randn(1, 4, 16)]
        references = [torch.randn(1, 128, 1, 1), torch.randn(1, 128, 1, 2)]
        latents = torch.randn(2, 128, 1, 1)
        reference_rope = _sft_reference_rope(references)
        uncached = _run_transformer(model, latents, contexts, references, reference_rope)
        cache = [ReferenceKVLayerCache("Enabled") for _ in model.layers]
        cached = _run_transformer(model, latents, contexts, references, reference_rope, cache=cache)
        for expected, actual in zip(uncached, cached):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertTrue(all(layer.k is not None and layer.v is not None for layer in cache))

        second_latents = torch.randn_like(latents)
        uncached_again = _run_transformer(model, second_latents, contexts, references, reference_rope)
        cached_again = _run_transformer(model, second_latents, contexts, references, reference_rope, cache=cache)
        for expected, actual in zip(uncached_again, cached_again):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for layer in cache:
            layer.clear()
        self.assertTrue(all(layer.k is None and layer.v is None for layer in cache))

    def test_hq_reference_cache_keeps_branches_independent_and_skips_negative_layer(self):
        torch.manual_seed(79)
        model = _tiny_transformer(layers=3)
        contexts = [torch.randn(1, 3, 16), torch.randn(1, 5, 16)]
        references = [torch.randn(1, 128, 1, 1)]
        ropes = _hq_reference_ropes(references, contexts)
        cache = [
            [ReferenceKVLayerCache("Enabled") for _ in model.layers]
            for _ in contexts
        ]
        outputs = _run_transformer(
            model, torch.randn(1, 128, 1, 1), contexts, references, ropes,
            cache=cache, skip_layer=1,
        )
        self.assertEqual(len(outputs), 2)
        self.assertIsNotNone(cache[0][0].k)
        self.assertIsNotNone(cache[1][0].k)
        self.assertFalse(torch.equal(cache[0][0].k, cache[1][0].k))
        self.assertIsNone(cache[0][1].k)
        self.assertIsNotNone(cache[1][1].k)

        # Re-run with matching body noise to check that cached and uncached
        # branch isolation stays numerically aligned through the skipped layer.
        latents = torch.randn(1, 128, 1, 1)
        uncached = _run_transformer(model, latents, contexts, references, ropes, skip_layer=1)
        cached = _run_transformer(model, latents, contexts, references, ropes, cache=cache, skip_layer=1)
        for expected, actual in zip(uncached, cached):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for branch in cache:
            for layer in branch:
                layer.clear()

    def test_int8_reference_cache_stays_within_cpu_tolerance(self):
        torch.manual_seed(83)
        model = _tiny_transformer(layers=2)
        contexts = [torch.randn(1, 3, 16), torch.randn(1, 4, 16)]
        references = [torch.randn(1, 128, 1, 2)]
        rope = _sft_reference_rope(references)
        latents = torch.randn(1, 128, 1, 1)
        expected = _run_transformer(model, latents, contexts, references, rope)
        cache = [ReferenceKVLayerCache("INT8") for _ in model.layers]
        actual = _run_transformer(model, latents, contexts, references, rope, cache=cache)
        for reference, output in zip(expected, actual):
            torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
        for layer in cache:
            layer.clear()

    def test_cache_budget_declines_before_allocation_and_reports_dynamic_loras(self):
        with patch("models.moss_image.edit_runtime._memory_budgets", return_value=(0, 0)):
            declined = plan_reference_cache(
                "Enabled",
                device="cuda",
                layers=42,
                dim=3072,
                heads=24,
                reference_tokens=256,
                body_tokens=1024,
                branches=2,
                batch_size=2,
            )
        self.assertEqual(declined.actual_mode, "Disabled")
        self.assertIn("estimated GPU need", declined.reason)
        self.assertIn("Disabled", cache_mode_log(declined))

        with patch("models.moss_image.edit_runtime._memory_budgets", return_value=(50 * 1024**3, 50 * 1024**3)):
            dynamic = plan_reference_cache(
                "Enabled",
                device="cuda",
                layers=2,
                dim=16,
                heads=2,
                reference_tokens=2,
                body_tokens=4,
                branches=1,
                dynamic_loras=True,
            )
        self.assertEqual(dynamic.actual_mode, "Disabled")
        self.assertIn("step-varying LoRA", dynamic.reason)
        with patch("models.moss_image.edit_runtime._memory_budgets", return_value=(50 * GIB, 50 * GIB)):
            enabled = plan_reference_cache(
                "Enabled",
                device="cuda",
                layers=2,
                dim=16,
                heads=2,
                reference_tokens=2,
                body_tokens=4,
                branches=1,
                denoising_steps=2,
            )
        self.assertEqual(enabled.actual_mode, "Enabled")
        self.assertEqual(
            plan_reference_cache(
                "Enabled",
                device="cuda",
                layers=2,
                dim=16,
                heads=2,
                reference_tokens=2,
                body_tokens=4,
                branches=1,
                denoising_steps=1,
            ).actual_mode,
            "Disabled",
        )
        self.assertTrue(has_step_varying_loras({"phase1": [[0.0, 1.0]]}))
        self.assertFalse(has_step_varying_loras({"phase1": [1.0]}))

    def test_cache_planner_reserves_only_non_cuda_dit_weights(self):
        model = nn.Linear(2, 2)
        non_cuda_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        )
        with (
            patch("psutil.virtual_memory", return_value=SimpleNamespace(available=8 * GIB)),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(10 * GIB, 24 * GIB)),
            patch("torch.cuda.memory_reserved", return_value=3 * GIB),
            patch("torch.cuda.memory_allocated", return_value=1 * GIB),
        ):
            gpu_capacity, ram_budget = _memory_budgets("cuda", model)
        self.assertEqual(gpu_capacity, 12 * GIB - non_cuda_bytes)
        self.assertGreater(ram_budget, 0)

    def test_encoder_mean_patchpacking_and_tiny_overlap_tiles_are_finite(self):
        moments = torch.arange(64 * 2 * 4, dtype=torch.float32).reshape(1, 64, 2, 4)
        expected = F.pixel_unshuffle(moments[:, :32], 2)
        actual = MOSSImageVAE._patch_mean(moments)
        self.assertEqual(tuple(actual.shape), (1, 128, 1, 2))
        torch.testing.assert_close(actual, expected)

        vae = MOSSImageVAE.__new__(MOSSImageVAE)
        nn.Module.__init__(vae)
        vae.encoder = _TinyImageEncoder()
        vae.bn_eps = 1e-4
        vae.bn = nn.BatchNorm2d(128, affine=False)
        vae.bn.running_mean.zero_()
        vae.bn.running_var.fill_(1)
        vae._interrupt = False
        image = torch.linspace(-1, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
        encoded = MOSSImageVAE.encode(vae, image, tile_size=32)
        self.assertEqual(tuple(encoded.shape), (1, 128, 2, 3))
        self.assertTrue(torch.isfinite(encoded).all())

        vae._interrupt = True
        with self.assertRaises(GenerationAborted):
            MOSSImageVAE.encode(vae, image, tile_size=32)

    def test_pipeline_clears_real_cache_when_cancelled_by_callback_failure(self):
        pipeline = _pipeline(editing=True, variant="sft", model_def={"moss_edit_reference_size": 16})
        created = []
        from models.moss_image import pipeline as pipeline_module

        real_create = pipeline_module.create_reference_kv_cache

        def record_cache(*args, **kwargs):
            value = real_create(*args, **kwargs)
            created.append(value)
            return value

        with patch(
            "models.moss_image.pipeline.plan_reference_cache",
            return_value=ReferenceCachePlan("Enabled", "Enabled", "test fits", 0, 1, 0, 1),
        ), patch("models.moss_image.pipeline.create_reference_kv_cache", side_effect=record_cache):
            def fail_after_step(step, *_args, **_kwargs):
                if step >= 1:
                    raise RuntimeError("stop after reference capture")

            with self.assertRaisesRegex(RuntimeError, "stop after"):
                pipeline.generate(
                    input_prompt="change the color",
                    n_prompt="",
                    sampling_steps=2,
                    width=16,
                    height=16,
                    guide_scale=1,
                    input_ref_images=[Image.new("RGB", (16, 16), "red")],
                    image_mode=1,
                    video_prompt_type="I",
                    VAE_tile_size=False,
                    custom_settings={"moss_image_kv_cache": "Enabled"},
                    callback=fail_after_step,
                )
        self.assertEqual(len(created), 1)
        self.assertTrue(all(layer.k is None and layer.v is None for layer in created[0]))


if __name__ == "__main__":
    unittest.main()
