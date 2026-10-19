"""Model-free capability checks for MiniMax H3 SLA attention."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3.minimax_h3_handler import family_handler


class TestMiniMaxH3SLACapabilities(unittest.TestCase):
    def test_ordinary_frames_and_ref2va_models_expose_experimental_sla(self):
        for model_type in (
            "minimax_h3",
            "minimax_h3_full",
            "minimax_h3_ref2va",
            "minimax_h3_ref2va_full",
        ):
            with self.subTest(model_type=model_type):
                model_def = family_handler.query_model_def(model_type, {})

                self.assertTrue(model_def["sla_attention"])
                self.assertFalse(model_def["sla_attention_default"])
                self.assertTrue(model_def["sla_attention_experimental"])
                self.assertEqual(model_def["minimax_h3_sampler"], "euler")

                defaults = {}
                family_handler.update_default_settings(
                    model_type, model_def, defaults
                )
                self.assertEqual(defaults["num_inference_steps"], 20)
                self.assertEqual(defaults["override_attention"], "")

    def test_fused_sla_remains_enabled_by_default_and_nonexperimental(self):
        model_def = family_handler.query_model_def(
            "minimax_h3", {"minimax_h3_fused_turbo": True}
        )

        self.assertTrue(model_def["sla_attention"])
        self.assertTrue(model_def["sla_attention_default"])
        self.assertFalse(model_def["sla_attention_experimental"])
        self.assertEqual(model_def["minimax_h3_sampler"], "res_multistep")

        defaults = {}
        family_handler.update_default_settings("minimax_h3", model_def, defaults)
        self.assertEqual(defaults["num_inference_steps"], 4)
        self.assertEqual(defaults["override_attention"], "sla")

    def test_specialty_and_locked_definitions_keep_sla_excluded(self):
        imported_profile = {
            "status": "verified",
            "architectures": ["minimax_h3"],
            "native_workflow": "ref2va",
            "qkv_layout": "grouped",
            "sampler": "euler",
            "video_shift": 9.0,
            "audio_shift": 4.0,
            "default_steps": 8,
            "min_steps": 4,
            "max_steps": 8,
            "sampling_profile": "turbo",
        }
        exclusions = (
            ("minimax_h3", {"vdn": True}),
            ("minimax_h3_voice_audio", {}),
            ("viggle_animate", {}),
            ("minimax_h3", {"minimax_h3_singularity": True}),
            ("minimax_h3_ref2va", {"minimax_h3_dasiwa": True}),
            (
                "minimax_h3",
                {
                    "minimax_h3_import_profile": imported_profile,
                    "minimax_h3_baked_turbo": True,
                },
            ),
            ("minimax_h3_ref2va", {"lock_inference_steps": True}),
        )

        for model_type, model_options in exclusions:
            with self.subTest(model_type=model_type, model_options=model_options):
                model_def = family_handler.query_model_def(
                    model_type, model_options
                )

                self.assertFalse(model_def.get("sla_attention", False))
                self.assertFalse(model_def.get("sla_attention_default", False))
                self.assertFalse(
                    model_def.get("sla_attention_experimental", False)
                )


if __name__ == "__main__":
    unittest.main()
