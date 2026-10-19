"""CPU contracts for native MOSS Image LoRA loading and denoising schedules."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from mmgp import offload  # noqa: E402
from models.moss_image.moss_image_handler import family_handler  # noqa: E402
from models.moss_image.pipeline import MOSSImagePipeline  # noqa: E402
from models.moss_image.transformer import MOSSImageTransformer  # noqa: E402
from shared.utils.loras_mutipliers import parse_loras_multipliers  # noqa: E402


def _tiny_transformer():
    model = MOSSImageTransformer(
        dim=16,
        ffn_dim=32,
        num_heads=2,
        num_layers=1,
        text_dim=16,
        freq_dim=8,
        latent_channels=128,
    ).eval()
    model._force_device = "cpu"
    return model


def _hook_mmgp_lora_slots(model):
    """Install MMGP's normal Linear adapter forward path without GPU offload."""
    model._loras_model_data = {}
    model._loras_model_shortcuts = {}
    manager = object.__new__(offload.offload)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.forward = manager.hook_lora(
                module,
                model,
                "transformer",
                model._loras_model_data,
                model._loras_model_shortcuts,
                name,
            )
            module._mm_manager = manager
    return manager


def _load_cpu_adapter(model, path):
    offload.load_loras_into_model(
        model,
        [str(path)],
        [1.0],
        preprocess_sd=lambda state: model.preprocess_loras("moss_image", state),
        verboseLevel=0,
    )
    # A real MMGP profile moves these tensors to a *_GPU slot. Point those
    # slots at their CPU tensors here so the same LoRA forward code is testable
    # without initializing CUDA or creating an offload profile.
    for module_data in model._loras_model_data.values():
        for adapter_name, adapter_data in tuple(module_data.items()):
            module_data[adapter_name + "_GPU"] = adapter_data


class _FixedTextCache:
    def __init__(self, contexts):
        self.contexts = contexts

    def encode(self, _encode, _prompts, device=None):
        return [context.to(device) for context in self.contexts]


class _PassthroughVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_eps = 1e-4
        self.bn = SimpleNamespace(
            running_mean=torch.zeros(128),
            running_var=torch.ones(128),
        )
        self._interrupt = False

    def forward(self, latents, tile_size=0):
        del tile_size
        return latents.float()


def _run_generation(pipeline, loras_slists=None):
    transformer = pipeline.transformer
    steps = []
    scales = []
    video_outputs = []
    text_outputs = []

    def record_step(module, _inputs):
        steps.append(module._lora_step_no if hasattr(module, "_lora_step_no") else None)
        active = getattr(module, "_loras_scaling", {})
        scales.append({name: value[:] if isinstance(value, list) else value for name, value in active.items()})

    def record_output(store):
        def record(_module, _inputs, output):
            store.append(output.detach().clone())
        return record

    handles = [
        transformer.register_forward_pre_hook(record_step),
        transformer.video_embedder.register_forward_hook(record_output(video_outputs)),
        transformer.text_embedder.proj.register_forward_hook(record_output(text_outputs)),
    ]
    try:
        images = pipeline.generate(
            input_prompt="positive",
            n_prompt="negative",
            seed=19,
            sampling_steps=2,
            width=16,
            height=16,
            guide_scale=3.0,
            batch_size=2,
            VAE_tile_size=False,
            loras_slists=loras_slists,
        )
    finally:
        for handle in handles:
            handle.remove()
    return images, steps, scales, video_outputs, text_outputs


class MossImageLoRATests(unittest.TestCase):
    def test_base_and_sft_variants_share_the_moss_lora_directory(self):
        self.assertEqual(
            family_handler.get_lora_dir("moss_image", None, "/loras"),
            os.path.join("/loras", "moss_image"),
        )
        self.assertEqual(
            family_handler.get_lora_dir("moss_image_sft", None, "/loras"),
            os.path.join("/loras", "moss_image"),
        )

    def test_preprocessor_normalizes_native_upstream_diffusers_and_kohya_keys(self):
        model = _tiny_transformer()
        tensor = torch.ones(2, 128)
        cases = (
            "video_embedder.lora_A.weight",
            "diffusion_model.video_embedder.lora_down.weight",
            "base_model.model.transformer.video_embedder.lora_A.default.weight",
            "lora_unet_video_embedder.lora_down.weight",
        )
        for key in cases:
            with self.subTest(key=key):
                mapped = model.preprocess_loras("moss_image", {key: tensor})
                self.assertEqual(list(mapped), ["diffusion_model.video_embedder.lora_A.weight"])
                self.assertIs(mapped["diffusion_model.video_embedder.lora_A.weight"], tensor)

        tensor = torch.ones(16, 2)
        mapped = model.preprocess_loras(
            "moss_image",
            {"transformer.video_embedder.lora_B.default.weight": tensor},
        )
        self.assertEqual(list(mapped), ["diffusion_model.video_embedder.lora_B.weight"])
        self.assertIs(mapped["diffusion_model.video_embedder.lora_B.weight"], tensor)

    def test_mmgp_rejects_adapter_shapes_that_do_not_match_moss_layers(self):
        model = _tiny_transformer()
        _hook_mmgp_lora_slots(model)
        bad_adapter = {
            "transformer.video_embedder.lora_A.weight": torch.ones(2, 7),
            "transformer.video_embedder.lora_B.weight": torch.ones(16, 2),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-shape.safetensors"
            save_file(bad_adapter, str(path))
            _load_cpu_adapter(model, path)

        errors = model._loras_errors
        self.assertEqual(len(errors), 1)
        self.assertIn("Lora A dimension is not compatible", errors[0][1])
        self.assertEqual(model._loras_active_adapters, [])

    def test_generation_applies_scheduled_lora_to_cfg_branches_and_batches(self):
        torch.manual_seed(31)
        transformer = _tiny_transformer()
        _hook_mmgp_lora_slots(transformer)
        adapter = {
            "transformer.video_embedder.lora_A.weight": torch.zeros(2, 128),
            "transformer.video_embedder.lora_B.weight": torch.zeros(16, 2),
            "transformer.text_embedder.proj.lora_A.default.weight": torch.zeros(2, 16),
            "transformer.text_embedder.proj.lora_B.weight": torch.zeros(16, 2),
        }
        adapter["transformer.video_embedder.lora_A.weight"][0, 0] = 1
        adapter["transformer.video_embedder.lora_A.weight"][1, 1] = 1
        adapter["transformer.video_embedder.lora_B.weight"][0, 0] = 0.75
        adapter["transformer.video_embedder.lora_B.weight"][1, 1] = -0.5
        adapter["transformer.text_embedder.proj.lora_A.default.weight"][0, 0] = 1
        adapter["transformer.text_embedder.proj.lora_A.default.weight"][1, 1] = 1
        adapter["transformer.text_embedder.proj.lora_B.weight"][0, 0] = 0.75
        adapter["transformer.text_embedder.proj.lora_B.weight"][1, 1] = -0.5

        negative = torch.ones(1, 3, 16)
        positive = torch.arange(48, dtype=torch.float32).reshape(1, 3, 16) / 16
        pipeline = MOSSImagePipeline(
            transformer,
            SimpleNamespace(_interrupt=False),
            tokenizer=None,
            vae=_PassthroughVAE(),
        )
        pipeline.text_encoder_cache = _FixedTextCache([negative, positive])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            save_file(adapter, str(path))
            _load_cpu_adapter(transformer, path)
            try:
                offload.activate_loras(transformer, [], [])
                baseline, baseline_steps, _, baseline_video, baseline_text = _run_generation(pipeline)

                _, schedule, errors = parse_loras_multipliers(["0,1"], 1, 2, nb_phases=1)
                self.assertEqual(errors, "")
                adapted, adapted_steps, scales, adapted_video, adapted_text = _run_generation(pipeline, schedule)
            finally:
                offload.unload_loras_from_model(transformer)

        self.assertEqual(baseline_steps, [0, 0])
        self.assertEqual(adapted_steps, [0, 1])
        self.assertEqual(scales, [{"0": [0.0, 1.0]}, {"0": [0.0, 1.0]}])
        self.assertEqual(tuple(adapted.shape), tuple(baseline.shape))
        self.assertFalse(torch.allclose(adapted, baseline))

        # At the zero-weight step the actual MMGP-wrapped linears match their
        # no-adapter outputs. At the next step the same adapter changes the
        # shared batched image projection and both CFG text branches.
        torch.testing.assert_close(adapted_video[0], baseline_video[0])
        self.assertEqual(tuple(adapted_video[1].shape[:2]), (2, 1))
        for image_no in range(2):
            self.assertGreater(
                (adapted_video[1][image_no] - baseline_video[1][image_no]).abs().sum().item(),
                0.0,
            )
        self.assertEqual(len(adapted_text), 4)
        torch.testing.assert_close(adapted_text[0], baseline_text[0])
        torch.testing.assert_close(adapted_text[1], baseline_text[1])
        for branch_index in (2, 3):
            self.assertGreater(
                (adapted_text[branch_index] - baseline_text[branch_index]).abs().sum().item(),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
