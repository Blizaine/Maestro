"""Qwen-Image-2.1-Turbo Draft model: sampling profile, handler options and model definition."""

import json
import sys
import unittest
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.qwen21 import qwen21_handler  # noqa: E402
from models.qwen21.pipeline_qwenimage21 import (  # noqa: E402
    QWEN21_TURBO_SIGMAS,
    QWEN21_TURBO_SOLVER,
    qwen21_sampling_profile,
)

CONFIGS = ROOT / "app" / "models" / "qwen21" / "configs"
TURBO_DEF = json.loads((ROOT / "app" / "defaults" / "qwen_image_21_7B_turbo.json").read_text())
BASE_DEF = json.loads((ROOT / "app" / "defaults" / "qwen_image_21_7B.json").read_text())
VIGGLE_LORA = qwen21_handler._VIGGLE_PROFILES["viggle_v01"]["lora_url"]


class TurboSamplingProfileTests(unittest.TestCase):
    def test_profile_returns_published_schedule_without_dynamic_shift(self):
        overrides, sigmas = qwen21_sampling_profile(QWEN21_TURBO_SOLVER, 8)
        self.assertEqual(tuple(sigmas), QWEN21_TURBO_SIGMAS)
        self.assertEqual(overrides, {"use_dynamic_shifting": False, "shift": 1.0, "shift_terminal": None})

    def test_profile_requires_eight_steps(self):
        with self.assertRaises(ValueError):
            qwen21_sampling_profile(QWEN21_TURBO_SOLVER, 40)

    def test_scheduler_uses_sigmas_unshifted(self):
        config = json.loads((CONFIGS / "scheduler_scheduler_config.json").read_text())
        overrides, sigmas = qwen21_sampling_profile(QWEN21_TURBO_SOLVER, 8)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(config, **overrides)
        scheduler.set_timesteps(sigmas=list(sigmas), device="cpu")
        self.assertTrue(torch.allclose(scheduler.sigmas[:-1].float(),
                                       torch.tensor(QWEN21_TURBO_SIGMAS), atol=1e-5))
        self.assertEqual(float(scheduler.sigmas[-1]), 0.0)

    def test_base_profiles_unchanged(self):
        self.assertEqual(qwen21_sampling_profile("default", 40), ({}, None))
        overrides, sigmas = qwen21_sampling_profile("viggle_v01", 4)
        self.assertEqual(overrides, {"shift_terminal": None})
        self.assertEqual(len(sigmas), 4)


class TurboHandlerTests(unittest.TestCase):
    def test_turbo_definition_offers_only_turbo_profile(self):
        options = qwen21_handler.family_handler.query_model_def("qwen_image_21_7B", TURBO_DEF["model"])
        self.assertEqual(options["sample_solvers"], [("Qwen Turbo checkpoint (8 steps)", QWEN21_TURBO_SOLVER)])
        self.assertEqual(list(options["qwen21_acceleration_profiles"]), [QWEN21_TURBO_SOLVER])
        self.assertTrue(options["qwen21_turbo_checkpoint"])
        self.assertNotIn("lora_url", options["qwen21_acceleration_profiles"][QWEN21_TURBO_SOLVER])

    def test_base_definition_keeps_default_and_viggle(self):
        options = qwen21_handler.family_handler.query_model_def("qwen_image_21_7B", BASE_DEF["model"])
        solvers = [value for _, value in options["sample_solvers"]]
        self.assertEqual(solvers, ["default", "viggle_v01", "viggle_v02", "viggle_v021"])
        self.assertFalse(options["qwen21_turbo_checkpoint"])
        self.assertIn("lora_url", options["qwen21_acceleration_profiles"]["viggle_v01"])

    def test_apply_turbo_profile_sets_recipe_and_strips_managed_adapter(self):
        settings = {"sample_solver": QWEN21_TURBO_SOLVER, "num_inference_steps": 40, "guidance_scale": 4.0,
                    "activated_loras": [VIGGLE_LORA, "my_style.safetensors"], "loras_multipliers": "1 0.7"}
        out = qwen21_handler.apply_acceleration_profile(settings)
        self.assertEqual(out["num_inference_steps"], 8)
        self.assertEqual(out["guidance_scale"], 1.0)
        self.assertEqual(out["activated_loras"], ["my_style.safetensors"])
        self.assertEqual(out["loras_multipliers"], "0.7")
        self.assertEqual(qwen21_handler.apply_acceleration_profile(out), out)

    def test_ui_defaults_for_turbo_definition(self):
        defaults = {}
        qwen21_handler.family_handler.update_default_settings("qwen_image_21_7B", TURBO_DEF["model"], defaults)
        self.assertEqual((defaults["num_inference_steps"], defaults["guidance_scale"], defaults["sample_solver"]),
                         (8, 1.0, QWEN21_TURBO_SOLVER))
        base = {}
        qwen21_handler.family_handler.update_default_settings("qwen_image_21_7B", BASE_DEF["model"], base)
        self.assertEqual((base["num_inference_steps"], base["sample_solver"]), (40, "default"))


class TurboDefinitionTests(unittest.TestCase):
    def test_definition_shape(self):
        model = TURBO_DEF["model"]
        self.assertEqual(model["architecture"], "qwen_image_21_7B")
        self.assertTrue(model["qwen21_turbo_checkpoint"])
        self.assertEqual(model["URLs"], ["qwen_image_21_7B_turbo_bf16.safetensors"])
        self.assertEqual((TURBO_DEF["num_inference_steps"], TURBO_DEF["guidance_scale"], TURBO_DEF["sample_solver"]),
                         (8, 1.0, QWEN21_TURBO_SOLVER))


if __name__ == "__main__":
    unittest.main()
