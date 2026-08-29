"""Contract tests for Alibaba PAI MiniMax H3 PDD acceleration adapters."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from safetensors.torch import save_file


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
_PDD_PATH = _APP / "models" / "minimax_h3" / "pdd.py"
_TURBO_PATH = _APP / "models" / "minimax_h3" / "turbo.py"
_MODEL_PATH = _APP / "models" / "minimax_h3" / "minimax_h3_main.py"
_LAUNCH_PATH = _APP / "launch.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MiniMaxH3PDDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdd = _load_module("maestro_minimax_h3_pdd_test", _PDD_PATH)
        cls.turbo = _load_module("maestro_minimax_h3_pdd_turbo_test", _TURBO_PATH)

    def test_workflow_presets_are_filtered_and_force_eight_steps(self):
        fl_ids = {
            preset["id"]
            for preset in self.turbo.minimax_h3_turbo_presets_for_workflow("fl2va")
        }
        ref_ids = {
            preset["id"]
            for preset in self.turbo.minimax_h3_turbo_presets_for_workflow("ref2va")
        }
        self.assertIn("alibaba-pai-fl2va-pdd-8step", fl_ids)
        self.assertNotIn("alibaba-pai-ref2va-pdd-8step", fl_ids)
        self.assertIn("alibaba-pai-ref2va-pdd-8step", ref_ids)
        self.assertNotIn("alibaba-pai-fl2va-pdd-8step", ref_ids)

        pruned_ref_ids = {
            preset["id"]
            for preset in self.turbo.minimax_h3_turbo_presets_for_workflow(
                "ref2va",
                full_checkpoint=False,
            )
        }
        full_ref_ids = {
            preset["id"]
            for preset in self.turbo.minimax_h3_turbo_presets_for_workflow(
                "ref2va",
                full_checkpoint=True,
            )
        }
        self.assertIn("alibaba-pai-ref2va-pdd-8step", pruned_ref_ids)
        self.assertIn("alibaba-pai-ref2va-pdd-8step", full_ref_ids)

        body = {
            "minimax_h3_turbo_mode": True,
            "minimax_h3_turbo_preset": "alibaba-pai-fl2va-pdd-8step",
        }
        self.assertTrue(
            self.turbo.normalize_minimax_h3_turbo_request(
                body,
                full_checkpoint=False,
                workflow="fl2va",
            )
        )
        self.assertEqual(body["num_inference_steps"], 8)
        self.assertEqual(body["loras_multipliers"], "1.00")
        self.assertEqual(
            body["activated_loras"],
            ["MiniMax-H3-FL2VA-Acc-8Step.safetensors"],
        )
        with self.assertRaisesRegex(ValueError, "FL2VA"):
            self.turbo.normalize_minimax_h3_turbo_request(
                body,
                full_checkpoint=False,
                workflow="ref2va",
            )

        ref_body = {
            "minimax_h3_turbo_mode": True,
            "minimax_h3_turbo_preset": "alibaba-pai-ref2va-pdd-8step",
        }
        self.assertTrue(
            self.turbo.normalize_minimax_h3_turbo_request(
                ref_body,
                full_checkpoint=False,
                workflow="ref2va",
            )
        )
        self.assertEqual(ref_body["num_inference_steps"], 8)
        self.assertEqual(ref_body["minimax_h3_reference_detail"], "max")

    def test_ref2va_pdd_honors_explicit_reference_detail(self):
        for requested_detail in ("match", "max"):
            with self.subTest(requested_detail=requested_detail):
                body = {
                    "minimax_h3_turbo_mode": True,
                    "minimax_h3_turbo_preset": (
                        "alibaba-pai-ref2va-pdd-8step"
                    ),
                    "minimax_h3_reference_detail": requested_detail,
                }
                self.assertTrue(
                    self.turbo.normalize_minimax_h3_turbo_request(
                        body,
                        full_checkpoint=False,
                        workflow="ref2va",
                    )
                )
                self.assertEqual(
                    body["minimax_h3_reference_detail"],
                    requested_detail,
                )

    def test_ref2va_pdd_rejects_invalid_explicit_reference_detail(self):
        body = {
            "minimax_h3_turbo_mode": True,
            "minimax_h3_turbo_preset": "alibaba-pai-ref2va-pdd-8step",
            "minimax_h3_reference_detail": "unexpected",
        }
        with self.assertRaisesRegex(ValueError, "reference detail"):
            self.turbo.normalize_minimax_h3_turbo_request(
                body,
                full_checkpoint=False,
                workflow="ref2va",
            )

    def test_request_and_runtime_layers_do_not_force_high_detail(self):
        launch_source = _LAUNCH_PATH.read_text(encoding="utf-8")
        model_source = _MODEL_PATH.read_text(encoding="utf-8")
        self.assertNotIn(
            'body["minimax_h3_reference_detail"] = "max"',
            launch_source,
        )
        self.assertNotIn(
            'minimax_h3_reference_detail = "max"',
            model_source,
        )
        self.assertIn(
            "Match output (no reference upscaling)",
            launch_source,
        )
        self.assertIn(
            "Match output (no reference upscaling)",
            model_source,
        )

    def _synthetic_state(self):
        tensors = {
            "proj_out.weight": torch.zeros(32, 3, 4),
            "proj_out.bias": torch.zeros(32, 3),
            "audio_proj_out.weight": torch.zeros(32, 2, 4),
            "audio_proj_out.bias": torch.zeros(32, 2),
        }
        for offset, kind in enumerate(("q", "k", "v"), start=1):
            prefix = f"transformer_blocks.0.attn.to_{kind}"
            tensors[f"{prefix}.lora_down"] = torch.full((2, 4), float(offset))
            tensors[f"{prefix}.lora_up"] = torch.full((3, 2), float(offset + 3))
        tensors["transformer_blocks.0.attn.to_out.0.lora_down"] = torch.ones(2, 3)
        tensors["transformer_blocks.0.attn.to_out.0.lora_up"] = torch.ones(4, 2)
        return tensors

    def test_diffusers_names_map_to_split_and_fused_maestro_qkv(self):
        source = self._synthetic_state()
        self.assertTrue(self.pdd.is_pdd_state_dict(source))

        split = self.pdd.preprocess_pdd_lora_state_dict(source, split_qkv=True)
        self.assertNotIn("proj_out.weight", split)
        self.assertIn("blocks.0.attn.q_proj.lora_A.weight", split)
        self.assertIn("blocks.0.attn.out_proj.lora_B.weight", split)

        fused = self.pdd.preprocess_pdd_lora_state_dict(source, split_qkv=False)
        down = fused["blocks.0.attn.qkv_proj.lora_A.weight"]
        up = fused["blocks.0.attn.qkv_proj.lora_B.weight"]
        self.assertEqual(tuple(down.shape), (6, 4))
        self.assertEqual(tuple(up.shape), (9, 6))
        delta = up @ down
        for index in range(3):
            expected = source[
                f"transformer_blocks.0.attn.to_{'qkv'[index]}.lora_up"
            ] @ source[
                f"transformer_blocks.0.attn.to_{'qkv'[index]}.lora_down"
            ]
            self.assertTrue(torch.equal(delta[index * 3 : (index + 1) * 3], expected))

    def test_interval_heads_are_fused_and_restored(self):
        final_layer = types.SimpleNamespace(
            video_out=torch.nn.Linear(4, 3),
            audio_out=torch.nn.Linear(4, 2),
        )
        transformer = types.SimpleNamespace(final_layer=final_layer)
        original_video = final_layer.video_out
        original_audio = final_layer.audio_out
        video_weights = torch.arange(32 * 3 * 4, dtype=torch.float32).reshape(32, 3, 4)
        audio_weights = torch.arange(32 * 2 * 4, dtype=torch.float32).reshape(32, 2, 4)
        tensors = {
            "proj_out.weight": video_weights,
            "proj_out.bias": torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3),
            "audio_proj_out.weight": audio_weights,
            "audio_proj_out.bias": torch.arange(32 * 2, dtype=torch.float32).reshape(32, 2),
        }
        metadata = {"pdd_num_steps": "32", "pdd_block_size": "4"}
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "pdd.safetensors"
            save_file(tensors, str(checkpoint), metadata=metadata)
            controller = self.pdd.install_pdd_parallel_heads(
                transformer,
                str(checkpoint),
            )
            self.assertEqual(controller.num_steps, 8)
            controller.set_step(0)
            sample = torch.ones(1, 4)
            plan = self.pdd.pdd_sampling_plan(
                self.pdd.pdd_time_grid(12.0).diff(),
                0,
            )[:4].float()
            expected_weight = sum(
                video_weights[index] * plan[index]
                for index in range(4)
            )
            expected_bias = sum(
                tensors["proj_out.bias"][index] * plan[index]
                for index in range(4)
            )
            expected = torch.nn.functional.linear(sample, expected_weight, expected_bias)
            self.assertTrue(torch.allclose(transformer.final_layer.video_out(sample), expected))

            video_sigmas = self.pdd.shifted_sigma(
                12.0,
                torch.linspace(1.0, 0.0, 9, dtype=torch.float64),
            )
            audio_sigmas = self.pdd.shifted_sigma(
                3.0,
                torch.linspace(1.0, 0.0, 9, dtype=torch.float64),
            )
            controller.configure_sigmas(video_sigmas, audio_sigmas)
            self.assertEqual(controller.num_steps, 8)
            controller.set_step(7)
            self.assertEqual(transformer.final_layer.video_out.step_index, 7)

        self.pdd.release_pdd_parallel_heads(transformer)
        self.assertIs(transformer.final_layer.video_out, original_video)
        self.assertIs(transformer.final_layer.audio_out, original_audio)

    def test_runtime_sigma_plans_match_fixed_published_schedule(self):
        for shift in (12.0, 3.0):
            with self.subTest(shift=shift):
                sigmas = self.pdd.shifted_sigma(
                    shift,
                    torch.linspace(1.0, 0.0, 9, dtype=torch.float64),
                )
                runtime = self.pdd.pdd_sampling_plans_for_sigmas(
                    sigmas,
                    shift,
                )
                step_sizes = self.pdd.pdd_time_grid(shift).diff()
                fixed = torch.stack([
                    self.pdd.pdd_sampling_plan(step_sizes, start)
                    for start in range(0, 32, 4)
                ])
                self.assertTrue(torch.allclose(runtime, fixed, atol=1e-12))

    def test_runtime_sigma_plans_support_nonuniform_ranges(self):
        shift = 12.0
        fine_sigmas = self.pdd.shifted_sigma(
            shift,
            torch.linspace(1.0, 0.0, 33, dtype=torch.float64),
        )
        boundaries = fine_sigmas[[0, 3, 11, 20, 32]]
        plans = self.pdd.pdd_sampling_plans_for_sigmas(boundaries, shift)
        self.assertEqual(tuple(plans.shape), (4, 32))
        self.assertTrue(torch.allclose(plans.sum(dim=1), torch.ones(4, dtype=torch.float64)))
        self.assertEqual(torch.nonzero(plans[0] > 0).flatten().tolist(), [0, 1, 2])
        self.assertEqual(torch.nonzero(plans[-1] > 0).flatten().tolist(), list(range(20, 32)))


if __name__ == "__main__":
    unittest.main()
