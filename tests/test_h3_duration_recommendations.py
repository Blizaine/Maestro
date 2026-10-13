"""Regression tests for RAM-gated MiniMax H3 duration defaults."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


from models.minimax_h3.minimax_h3_handler import (
    apply_h3_omni_sequence_memory_policy,
    apply_h3_window_memory_policy,
    family_handler,
    recommended_h3_omni_sequence_profile,
    recommended_h3_window_frames,
    recommended_h3_window_profile,
)
from services.director_video_strategy import build_director_video_execution_profile


class TestH3DurationRecommendations(unittest.TestCase):
    @staticmethod
    def model(architecture="minimax_h3", settings=None):
        return family_handler.query_model_def(architecture, settings or {})

    def profile(self, model, *, vram=12, ram=32, resolution="1280x704"):
        return recommended_h3_window_profile(
            vram,
            resolution,
            model,
            ram,
        )

    def test_shared_resource_rows_are_exposed_on_video_memory_policies(self):
        model = self.model()
        recommendations = model["sliding_window_memory_policy"][
            "resource_recommendations"
        ]
        self.assertEqual(recommendations["version"], "2026-10-12")
        self.assertTrue(recommendations["evidence"])
        self.assertTrue(recommendations["tiers"])
        self.assertIn(
            "rtx3080ti-12gb-32gb-720p-repeated-345-user-report",
            {item["id"] for item in recommendations["evidence"]},
        )
        self.assertEqual(
            model["director_memory_policy"]["resource_recommendations"]["version"],
            recommendations["version"],
        )

    def test_12gb_32gib_promotes_tested_720p_checkpoint_profiles(self):
        cases = (
            ("minimax_h3", {}, 345),
            ("minimax_h3_ref2va", {}, 345),
            ("minimax_h3", {"minimax_h3_fused_turbo": True}, 345),
            (
                "minimax_h3_ref2va",
                {"minimax_h3_fused_turbo": True},
                345,
            ),
        )
        for architecture, settings, expected in cases:
            with self.subTest(architecture=architecture, settings=settings):
                model = self.model(architecture, settings)
                profile = self.profile(model)
                self.assertEqual(profile["frames"], expected)
                self.assertLessEqual(profile["frames"], 345)
                self.assertEqual(profile["resource_recommendation_version"], "2026-10-12")

        fused = self.model("minimax_h3", {"minimax_h3_fused_turbo": True})
        self.assertEqual(
            recommended_h3_window_frames(12, "1280x704", fused, 32),
            345,
        )

    def test_24gb_128gib_promotes_only_fused_reference_720p(self):
        fused_reference = self.model(
            "minimax_h3_ref2va",
            {"minimax_h3_fused_turbo": True},
        )
        self.assertEqual(
            self.profile(fused_reference, vram=24, ram=128)["frames"],
            345,
        )
        self.assertEqual(
            self.profile(fused_reference, vram=24, ram=None)["frames"],
            243,
        )
        self.assertEqual(
            self.profile(fused_reference, vram=24, ram=32)["frames"],
            243,
        )

    def test_12gb_1080p_stays_at_native_minimum(self):
        for architecture, settings in (
            ("minimax_h3", {}),
            ("minimax_h3", {"minimax_h3_fused_turbo": True}),
            ("minimax_h3_ref2va", {"minimax_h3_fused_turbo": True}),
        ):
            with self.subTest(architecture=architecture, settings=settings):
                self.assertEqual(
                    self.profile(
                        self.model(architecture, settings),
                        resolution="1920x1088",
                    )["frames"],
                    124,
                )

    def test_unknown_low_ram_and_unmeasured_canvas_keep_vram_fallback(self):
        fused = self.model("minimax_h3", {"minimax_h3_fused_turbo": True})
        self.assertEqual(self.profile(fused, ram=None)["frames"], 124)
        self.assertEqual(self.profile(fused, ram=29.9)["frames"], 124)
        self.assertEqual(
            self.profile(fused, vram=16, ram=128)["frames"],
            243,
        )
        self.assertEqual(
            self.profile(fused, resolution="1344x768")["frames"],
            124,
        )
        self.assertIsNone(
            self.profile(fused, resolution="1920x1088", ram=None)["frames"]
        )

    def test_unmeasured_vram_sizes_keep_existing_vram_only_tiers(self):
        fused = self.model("minimax_h3", {"minimax_h3_fused_turbo": True})
        expected = {6: 124, 8: 124, 10: 124, 16: 243}
        for vram, frames in expected.items():
            with self.subTest(vram=vram):
                profile = self.profile(fused, vram=vram, ram=128)
                self.assertEqual(profile["frames"], frames)
                self.assertNotIn("resource_recommendation", profile)
    def test_intermediate_canvas_keeps_existing_conservative_tier(self):
        fused = self.model("minimax_h3", {"minimax_h3_fused_turbo": True})
        profile = self.profile(fused, resolution="1344x768")
        self.assertEqual(profile["frames"], 124)
        self.assertNotIn("resource_recommendation", profile)

    def test_low_resolution_pruned_retains_its_existing_345_frame_cap(self):
        pruned = self.model()
        profile = self.profile(pruned, resolution="864x480")
        self.assertEqual(profile["frames"], 345)
        self.assertLessEqual(profile["pixels"], 901120)

    def test_omni_sequence_margin_is_overridden_only_for_measured_pruned_tier(self):
        pruned_reference = self.model("minimax_h3_ref2va")
        base = self.profile(pruned_reference)
        sequence = recommended_h3_omni_sequence_profile(
            12,
            "1280x704",
            pruned_reference,
            32,
        )
        self.assertEqual(base["frames"], 345)
        self.assertEqual(sequence["frames"], 345)
        self.assertEqual(sequence["reference_margin_frames"], 0)
        self.assertEqual(sequence["reference_margin_steps"], 0)

        fused_reference = self.model(
            "minimax_h3_ref2va",
            {"minimax_h3_fused_turbo": True},
        )
        fused_sequence = recommended_h3_omni_sequence_profile(
            12,
            "1280x704",
            fused_reference,
            32,
        )
        self.assertEqual(fused_sequence["frames"], 345)
        self.assertEqual(fused_sequence["reference_margin_frames"], 0)

        # The legacy reference margin remains active when the RAM-gated row
        # does not match, including unknown RAM and another VRAM class.
        for vram, ram in ((12, None), (12, 29.9), (16, 32)):
            with self.subTest(vram=vram, ram=ram):
                fallback = recommended_h3_omni_sequence_profile(
                    vram, "1280x704", pruned_reference, ram
                )
                self.assertNotIn("resource_recommendation", fallback)
                self.assertEqual(fallback["reference_margin_steps"], 1)
                self.assertEqual(
                    fallback["reference_margin_frames"],
                    min(17, max(0, fallback["base_frames"] - 124)),
                )
                self.assertLess(fallback["frames"], 345)

    def test_pruned_tier_resource_boundaries_remain_conservative(self):
        pruned = self.model()
        matched = self.profile(pruned, vram=12, ram=30)
        self.assertEqual(matched["frames"], 345)
        self.assertIn("resource_recommendation", matched)
        for boundary_vram in (11, 12.5):
            with self.subTest(boundary_vram=boundary_vram):
                self.assertIn(
                    "resource_recommendation",
                    self.profile(pruned, vram=boundary_vram, ram=30),
                )

        for vram, ram, resolution in (
            (12, 29.9, "1280x704"),
            (12, None, "1280x704"),
            (10.9, 32, "1280x704"),
            (12.6, 32, "1280x704"),
            (12, 32, "1344x768"),
        ):
            with self.subTest(vram=vram, ram=ram, resolution=resolution):
                profile = self.profile(
                    pruned,
                    vram=vram,
                    ram=ram,
                    resolution=resolution,
                )
                self.assertNotIn("resource_recommendation", profile)

    def test_runtime_auto_helpers_apply_measured_tier_and_keep_manual_controls(self):
        hardware = {"gpu_vram_gb": 12, "ram_gb": 32}
        pruned = self.model()
        window_inputs = {
            "video_length": 362,
            "sliding_window_size": 362,
            "resolution": "1280x704",
        }
        adjustment = apply_h3_window_memory_policy(
            window_inputs,
            pruned,
            hardware,
        )
        self.assertEqual(adjustment["effective_window_frames"], 345)
        self.assertEqual(window_inputs["sliding_window_size"], 345)

        pruned_reference = self.model("minimax_h3_ref2va")
        sequence_inputs = {
            "video_length": 362,
            "resolution": "1280x704",
            "minimax_h3_reference_sequence": True,
            "minimax_h3_sequence_clip_frames": 362,
        }
        sequence = apply_h3_omni_sequence_memory_policy(
            sequence_inputs,
            pruned_reference,
            hardware,
        )
        self.assertEqual(sequence["effective_clip_frames"], 345)
        self.assertEqual(sequence["reference_margin_frames"], 0)
        self.assertEqual(sequence_inputs["minimax_h3_sequence_clip_frames"], 345)
    def test_director_matches_studio_and_preserves_manual_controls(self):
        hardware = {"gpu_vram_gb": 12, "ram_gb": 32}
        cases = (
            ("minimax_h3", {}, False),
            ("minimax_h3", {"minimax_h3_fused_turbo": True}, False),
            ("minimax_h3_ref2va", {}, True),
            ("minimax_h3_ref2va", {"minimax_h3_fused_turbo": True}, True),
        )
        for architecture, settings, omni in cases:
            with self.subTest(architecture=architecture, settings=settings):
                model = self.model(architecture, settings)
                studio = (
                    recommended_h3_omni_sequence_profile(
                        12, "1280x704", model, 32
                    )
                    if omni
                    else self.profile(model)
                )
                director = build_director_video_execution_profile(
                    architecture,
                    model,
                    {"resolution": "1280x704"},
                    hardware,
                )
                self.assertEqual(
                    director["recommended_max_frames"],
                    studio["frames"],
                )
                self.assertEqual(director["system_ram_gb"], 32)

        pruned = self.model()
        director_manual = build_director_video_execution_profile(
            "minimax_h3",
            pruned,
            {"resolution": "1280x704"},
            hardware,
            manual_max_frames=345,
        )
        self.assertTrue(director_manual["manual_override"])
        self.assertEqual(director_manual["effective_max_frames"], 345)

        runtime_inputs = {
            "video_length": 345,
            "sliding_window_size": 345,
            "sliding_window_memory_override": True,
        }
        self.assertIsNone(
            apply_h3_window_memory_policy(runtime_inputs, pruned, hardware)
        )
        self.assertEqual(runtime_inputs["sliding_window_size"], 345)

    def test_full_audio_viggle_and_alternate_checkpoints_are_not_promoted(self):
        definitions = (
            self.model("minimax_h3_full"),
            self.model("minimax_h3_voice_audio"),
            self.model("viggle_animate"),
            self.model(
                "minimax_h3",
                {"minimax_h3_singularity": True},
            ),
        )
        for model in definitions:
            with self.subTest(model=model.get("architecture")):
                for policy_key in (
                    "sliding_window_memory_policy",
                    "omni_sequence_memory_policy",
                    "director_memory_policy",
                ):
                    policy = model.get(policy_key)
                    if isinstance(policy, dict):
                        self.assertNotIn("resource_recommendations", policy)
                profile = recommended_h3_window_profile(
                    12,
                    "1280x704",
                    model,
                    128,
                )
                self.assertNotIn("resource_recommendation", profile)


if __name__ == "__main__":
    unittest.main()
