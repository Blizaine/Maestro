"""CPU-only profile and schedule regressions for official Qwen Image 2.1 Turbo."""

import sys
import unittest
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.qwen21.qwen21_handler import (  # noqa: E402
    MODEL_TYPE,
    TURBO_PROFILE,
    apply_acceleration_profile,
    family_handler,
)
from models.qwen21.pipeline_qwenimage21 import (  # noqa: E402
    QWEN21_TURBO_SIGMAS,
    qwen21_sampling_profile,
    retrieve_timesteps,
)


TURBO_MODEL = {"qwen21_turbo_checkpoint": True}
BASE_MODEL = {"qwen21_turbo_checkpoint": False}
VIGGLE_V02 = (
    "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/"
    "Qwen-Image-2.1-viggle-turbo-v0.2-5step-lora-r256.safetensors"
)


class TurboHandlerTests(unittest.TestCase):
    def test_turbo_normalizes_inherited_settings_and_only_removes_managed_viggle_loras(self):
        original = {
            "sample_solver": "euler",
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "activated_loras": ["style.safetensors", VIGGLE_V02],
            "loras_multipliers": "0.35 0.8",
        }

        result = apply_acceleration_profile(original, model_def=TURBO_MODEL)

        self.assertEqual(result["sample_solver"], TURBO_PROFILE)
        self.assertEqual(result["num_inference_steps"], 8)
        self.assertEqual(result["guidance_scale"], 1.0)
        self.assertEqual(result["activated_loras"], ["style.safetensors"])
        self.assertEqual(result["loras_multipliers"], "0.35")
        self.assertEqual(original["num_inference_steps"], 40)
        self.assertEqual(apply_acceleration_profile(result, model_def=TURBO_MODEL), result)

    def test_turbo_rejects_viggle_profile(self):
        for profile in ("viggle", "viggle_v02"):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "cannot use a Viggle"):
                apply_acceleration_profile(
                    {"sample_solver": profile, "num_inference_steps": 5},
                    model_def=TURBO_MODEL,
                )

    def test_base_checkpoint_rejects_official_turbo_profile(self):
        with self.assertRaisesRegex(ValueError, "requires the official Turbo checkpoint"):
            apply_acceleration_profile(
                {"sample_solver": TURBO_PROFILE}, model_def=BASE_MODEL
            )

    def test_base_viggle_profile_keeps_its_schedule_and_adapter(self):
        result = apply_acceleration_profile(
            {
                "sample_solver": "viggle_v02",
                "num_inference_steps": 5,
                "activated_loras": ["style.safetensors"],
                "loras_multipliers": "0.35",
            },
            model_def=BASE_MODEL,
        )
        self.assertEqual(result["sample_solver"], "viggle_v02")
        self.assertEqual(result["num_inference_steps"], 5)
        self.assertEqual(result["guidance_scale"], 1.0)
        self.assertEqual(result["activated_loras"], ["style.safetensors", VIGGLE_V02])
        self.assertEqual(result["loras_multipliers"], "0.35 1")

    def test_turbo_defaults_and_inherited_workflow_metadata(self):
        defaults = {}
        family_handler.update_default_settings(MODEL_TYPE, TURBO_MODEL, defaults)
        self.assertEqual(
            (defaults["sample_solver"], defaults["num_inference_steps"], defaults["guidance_scale"]),
            (TURBO_PROFILE, 8, 1.0),
        )

        fixed = {"num_inference_steps": 40, "guidance_scale": 4.0, "sample_solver": "default"}
        family_handler.fix_settings(MODEL_TYPE, 0, {"model": TURBO_MODEL}, fixed)
        self.assertEqual(
            (fixed["sample_solver"], fixed["num_inference_steps"], fixed["guidance_scale"]),
            (TURBO_PROFILE, 8, 1.0),
        )

        options = family_handler.query_model_def(MODEL_TYPE, {"model": TURBO_MODEL})
        self.assertTrue(options["qwen21_turbo_checkpoint"])
        self.assertEqual(set(options["qwen21_acceleration_profiles"]), {TURBO_PROFILE})
        self.assertEqual(options["sample_solvers"], [(options["qwen21_acceleration_profiles"][TURBO_PROFILE]["label"], TURBO_PROFILE)])

    def test_base_defaults_and_viggle_options_remain_available(self):
        defaults = {}
        family_handler.update_default_settings(MODEL_TYPE, BASE_MODEL, defaults)
        self.assertEqual(
            (defaults["sample_solver"], defaults["num_inference_steps"], defaults["guidance_scale"]),
            ("default", 40, 4.0),
        )
        options = family_handler.query_model_def(MODEL_TYPE, BASE_MODEL)
        self.assertFalse(options["qwen21_turbo_checkpoint"])
        self.assertEqual(
            set(options["qwen21_acceleration_profiles"]),
            {"viggle_v01", "viggle_v02", "viggle_v021"},
        )

    def test_profile_utility_without_checkpoint_metadata_remains_usable(self):
        result = apply_acceleration_profile(
            {"sample_solver": TURBO_PROFILE, "num_inference_steps": 40, "guidance_scale": 4.0}
        )
        self.assertEqual(result["num_inference_steps"], 8)
        self.assertEqual(result["guidance_scale"], 1.0)


class TurboSamplingTests(unittest.TestCase):
    def test_official_sigmas_and_scheduler_configuration_are_used_exactly(self):
        overrides, sigmas = qwen21_sampling_profile(
            TURBO_PROFILE, 8, qwen21_turbo_checkpoint=True
        )
        self.assertEqual(
            sigmas,
            (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568),
        )
        self.assertEqual(overrides, {
            "shift": 1.0,
            "use_dynamic_shifting": False,
            "shift_terminal": None,
        })

        source = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(source.config, **overrides)
        timesteps, step_count = retrieve_timesteps(
            scheduler, len(sigmas), "cpu", sigmas=sigmas
        )

        self.assertEqual(step_count, 8)
        self.assertEqual(len(timesteps), 8)
        self.assertEqual(scheduler.config.shift, 1.0)
        self.assertFalse(scheduler.config.use_dynamic_shifting)
        self.assertIsNone(scheduler.config.shift_terminal)
        actual_sigmas = tuple(float(value) for value in scheduler.sigmas[:-1])
        for actual, expected in zip(actual_sigmas, QWEN21_TURBO_SIGMAS):
            self.assertAlmostEqual(actual, expected, places=7)
        self.assertEqual(float(scheduler.sigmas[-1]), 0.0)

    def test_scheduler_rejects_profile_checkpoint_mismatches_and_step_drift(self):
        with self.assertRaisesRegex(ValueError, "requires its named sampling profile"):
            qwen21_sampling_profile("default", 40, qwen21_turbo_checkpoint=True)
        with self.assertRaisesRegex(ValueError, "requires the official Turbo checkpoint"):
            qwen21_sampling_profile(TURBO_PROFILE, 8, qwen21_turbo_checkpoint=False)
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            qwen21_sampling_profile(TURBO_PROFILE, 5)
        with self.assertRaisesRegex(ValueError, "requires its named sampling profile"):
            qwen21_sampling_profile("viggle", 5, qwen21_turbo_checkpoint=True)

    def test_schedule_helper_keeps_legacy_default_and_viggle_profiles(self):
        self.assertEqual(qwen21_sampling_profile("default", 40), ({}, None))
        self.assertEqual(qwen21_sampling_profile("viggle_v02", 5)[1], (1.0, 0.875, 0.75, 0.5, 0.25))
        with self.assertRaisesRegex(ValueError, "requires exactly 5"):
            qwen21_sampling_profile("viggle_v02", 8)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
