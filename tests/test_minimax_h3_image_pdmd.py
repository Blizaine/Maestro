"""CPU schedule and catalog checks for official H3 Image pDMD profiles."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class TestMiniMaxH3ImagePDMD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.minimax_h3.image_handler import (
            PDD_LORA_URL,
            family_handler,
        )
        from models.minimax_h3.image_runtime import (
            IMAGE_PDD_SIGMAS,
            h3_image_head_plans,
            h3_image_sigmas,
        )
        from models.minimax_h3.lora_affine import _architecture

        cls.adapter = PDD_LORA_URL
        cls.handler = family_handler
        cls.native_sigmas = IMAGE_PDD_SIGMAS
        cls.head_plans = staticmethod(h3_image_head_plans)
        cls.sigmas = staticmethod(h3_image_sigmas)
        cls.lora_architecture = staticmethod(_architecture)

    def test_image_catalog_exposes_native_and_pdd_step_ranges(self):
        definition = self.handler.query_model_def("h3_image", {})
        self.assertFalse(definition["loras_disabled"])
        self.assertFalse(definition["lock_inference_steps"])
        self.assertEqual(definition["inference_steps_min"], 2)
        self.assertEqual(definition["inference_steps_max"], 6)

    def test_image_loras_use_the_shared_fl2va_adaln_basis(self):
        self.assertEqual(self.lora_architecture("h3_image"), "fl2va")

    def test_pdd_two_and_four_step_grids_match_h3_shift_twelve(self):
        for actual, expected in zip(
            self.sigmas(2), (1.0, 12.0 / 13.0, 0.0)
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(
            self.sigmas(4), (1.0, 36.0 / 37.0, 12.0 / 13.0, 0.8, 0.0)
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(self.sigmas(6), self.native_sigmas)
        for steps in (2, 4):
            plans = self.head_plans(self.sigmas(steps))
            self.assertEqual(tuple(plans.shape), (steps, 6))
            self.assertTrue((plans >= 0).all())
            self.assertTrue((plans.sum(dim=1) - 1.0).abs().max() < 1e-6)

    def test_validation_pairs_pdmd_with_two_or_four_steps_and_keeps_native_six(self):
        def validate(steps, adapters=()):
            return self.handler.validate_generative_settings(
                "h3_image",
                self.handler.query_model_def("h3_image", {}),
                {"num_inference_steps": steps, "activated_loras": list(adapters)},
            )

        self.assertIsNone(validate(2, [self.adapter]))
        self.assertIsNone(validate(4, [self.adapter]))
        self.assertIsNone(validate(6))
        self.assertIn("pDMD adapter", validate(6, [self.adapter]))
        self.assertIn("needs the pDMD adapter", validate(2))


if __name__ == "__main__":
    unittest.main()
