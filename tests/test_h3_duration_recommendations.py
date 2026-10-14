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
from services.director_music_timing import resolve_clip_limits
from services.h3_duration_recommendations import load_h3_duration_recommendations


class TestH3DurationRecommendations(unittest.TestCase):
    RESOURCE_GRID = (
        (
            "24 GB / 128 GB fused frames at native 1080p",
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
            24,
            128,
            "1920x1088",
            345,
            "tested",
            "fused_4step",
        ),
        (
            "24 GB / 128 GB fused references at 720p",
            "minimax_h3_ref2va",
            {"minimax_h3_fused_turbo": True},
            24,
            128,
            "1280x704",
            345,
            "tested",
            "fused_4step_references",
        ),
        (
            "24 GB / 128 GB fused references at native 1080p",
            "minimax_h3_ref2va",
            {"minimax_h3_fused_turbo": True},
            24,
            128,
            "1920x1088",
            243,
            "estimated",
            "fused_4step_references",
        ),
        (
            "12 GB / 31.76 GiB pruned at 720p",
            "minimax_h3",
            {},
            12,
            31.76,
            "1280x704",
            345,
            "reported",
            "pruned",
        ),
        (
            "12 GB / 32 GB fused frames at 720p",
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
            12,
            32,
            "1280x704",
            345,
            "reported",
            "fused_4step",
        ),
        (
            "12 GB / 32 GB fused references at 720p",
            "minimax_h3_ref2va",
            {"minimax_h3_fused_turbo": True},
            12,
            32,
            "1280x704",
            345,
            "reported",
            "fused_4step_references",
        ),
        (
            "12 GB / 32 GB pruned at an intermediate canvas",
            "minimax_h3",
            {},
            12,
            32,
            "1344x768",
            294,
            "estimated",
            "pruned",
        ),
        (
            "12 GB / 32 GB fused frames at an intermediate canvas",
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
            12,
            32,
            "1344x768",
            294,
            "estimated",
            "fused_4step",
        ),
        (
            "12 GB / 32 GB fused references at an intermediate canvas",
            "minimax_h3_ref2va",
            {"minimax_h3_fused_turbo": True},
            12,
            32,
            "1344x768",
            294,
            "estimated",
            "fused_4step_references",
        ),
        (
            "16 GB / 32 GB fused at 720p",
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
            16,
            32,
            "1280x704",
            345,
            "estimated",
            "fused_4step",
        ),
        (
            "16 GB / 32 GB fused at native 1080p",
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
            16,
            32,
            "1920x1088",
            158,
            "estimated",
            "fused_4step",
        ),
        (
            "20 GB / 64 GB fused at native 1080p",
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
            20,
            64,
            "1920x1088",
            243,
            "estimated",
            "fused_4step",
        ),
        (
            "10 GB / 32 GB fused at 720p",
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
            10,
            32,
            "1280x704",
            243,
            "estimated",
            "fused_4step",
        ),
        (
            "8 GB / 15.8 GB fused at 480p",
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
            8,
            15.8,
            "864x480",
            124,
            "estimated",
            "fused_4step",
        ),
    )

    @staticmethod
    def model(architecture="minimax_h3", settings=None):
        return family_handler.query_model_def(architecture, settings or {})

    def profile(
        self,
        model,
        *,
        vram=12,
        ram=32,
        resolution="1280x704",
    ):
        return recommended_h3_window_profile(
            vram,
            resolution,
            model,
            ram,
        )

    def test_shared_resource_matrix_is_exposed_on_video_memory_policies(self):
        model = self.model()
        recommendations = model["sliding_window_memory_policy"][
            "resource_recommendations"
        ]
        self.assertEqual(recommendations["version"], "2026-10-14")
        self.assertTrue(recommendations["tiers"])
        self.assertTrue(recommendations["evidence"])
        self.assertIn(
            "rtx3080ti-12gb-32gb-720p-repeated-345-user-report",
            {item["id"] for item in recommendations["evidence"]},
        )
        self.assertEqual(
            model["director_memory_policy"]["resource_recommendations"],
            recommendations,
        )

    def test_every_resource_row_has_valid_provenance_and_legal_h3_frames(self):
        recommendations = load_h3_duration_recommendations()
        evidence_ids = {
            item["id"]
            for item in recommendations.get("evidence", [])
            if isinstance(item, dict) and item.get("id")
        }
        self.assertTrue(evidence_ids)
        self.assertEqual(len(evidence_ids), len(recommendations["evidence"]))

        valid_bases = {"tested", "reported", "estimated", "conservative"}
        legal_frames = {124 + 17 * step for step in range(14)}
        for index, tier in enumerate(recommendations["tiers"]):
            with self.subTest(index=index, checkpoint=tier.get("checkpoint")):
                self.assertIn(tier.get("basis"), valid_bases)
                self.assertTrue(tier.get("evidence"))
                self.assertTrue(set(tier["evidence"]).issubset(evidence_ids))
                self.assertLessEqual(tier["min_vram_gb"], tier["max_vram_gb"])
                self.assertGreater(tier["min_ram_gb"], 0)
                self.assertLessEqual(tier["min_pixels"], tier["max_pixels"])
                if "max_ram_gb" in tier:
                    self.assertGreaterEqual(
                        tier["max_ram_gb"],
                        tier["min_ram_gb"],
                    )

                frames = tier.get("frames")
                if frames is None:
                    self.assertTrue(tier.get("fallback_resolution"))
                else:
                    self.assertIs(type(frames), int)
                    self.assertIn(frames, legal_frames)

    def test_resource_grid_profiles_match_their_evidence_basis(self):
        for (
            label,
            architecture,
            settings,
            vram,
            ram,
            resolution,
            expected_frames,
            expected_basis,
            expected_checkpoint,
        ) in self.RESOURCE_GRID:
            with self.subTest(case=label):
                model = self.model(architecture, settings)
                profile = self.profile(
                    model,
                    vram=vram,
                    ram=ram,
                    resolution=resolution,
                )
                self.assertTrue(profile["supported"])
                self.assertEqual(profile["frames"], expected_frames)
                self.assertLessEqual(profile["frames"], 345)
                self.assertEqual(profile["checkpoint"], expected_checkpoint)
                self.assertEqual(
                    profile["resource_recommendation"]["basis"],
                    expected_basis,
                )
                self.assertEqual(
                    profile["resource_recommendation"]["frames"],
                    expected_frames,
                )
                self.assertTrue(profile["resource_recommendation"]["evidence"])

    def test_native_1080p_on_12gb_keeps_the_conservative_short_start(self):
        for architecture, settings, checkpoint in (
            ("minimax_h3", {}, "pruned"),
            (
                "minimax_h3",
                {"minimax_h3_fused_turbo": True},
                "fused_4step",
            ),
            (
                "minimax_h3_ref2va",
                {"minimax_h3_fused_turbo": True},
                "fused_4step_references",
            ),
        ):
            with self.subTest(checkpoint=checkpoint):
                profile = self.profile(
                    self.model(architecture, settings),
                    vram=12,
                    ram=31.76,
                    resolution="1920x1088",
                )
                self.assertEqual(profile["frames"], 124)
                self.assertEqual(
                    profile["resource_recommendation"]["basis"],
                    "conservative",
                )

    def test_adjacent_max_ram_ranges_switch_to_the_matching_resource_row(self):
        fused = self.model(
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
        )
        cases = (
            (24, 62.99, "1920x1088", 158),
            (24, 63, "1920x1088", 243),
            (8, 29.99, "864x480", 124),
            (8, 30, "864x480", 243),
        )
        for vram, ram, resolution, expected_frames in cases:
            with self.subTest(vram=vram, ram=ram, resolution=resolution):
                profile = self.profile(
                    fused,
                    vram=vram,
                    ram=ram,
                    resolution=resolution,
                )
                self.assertEqual(profile["frames"], expected_frames)
                self.assertEqual(
                    profile["resource_recommendation"]["basis"],
                    "estimated",
                )

    def test_unknown_or_below_minimum_ram_keeps_the_legacy_vram_profile(self):
        fused = self.model(
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
        )
        for ram in (None, 0, 14.99):
            with self.subTest(ram=ram):
                profile = self.profile(fused, vram=12, ram=ram)
                self.assertEqual(profile["frames"], 124)
                self.assertNotIn("resource_recommendation", profile)
                self.assertNotIn("resource_recommendation_version", profile)

    def test_8gb_16gb_native_720p_is_unsupported_and_recommends_480p(self):
        fused = self.model(
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
        )
        profile = self.profile(
            fused,
            vram=8,
            ram=16,
            resolution="1280x704",
        )
        self.assertFalse(profile["supported"])
        self.assertIsNone(profile["frames"])
        self.assertEqual(profile["fallback_resolution"], "864x480")
        self.assertEqual(
            profile["resource_recommendation"]["basis"],
            "estimated",
        )
        self.assertEqual(
            recommended_h3_window_frames(
                8,
                "1280x704",
                fused,
                16,
            ),
            0,
        )

    def test_studio_and_director_match_the_resource_grid_and_provenance(self):
        for (
            label,
            architecture,
            settings,
            vram,
            ram,
            resolution,
            _expected_frames,
            _expected_basis,
            _expected_checkpoint,
        ) in self.RESOURCE_GRID:
            with self.subTest(case=label):
                model = self.model(architecture, settings)
                studio = (
                    recommended_h3_omni_sequence_profile(
                        vram,
                        resolution,
                        model,
                        ram,
                    )
                    if architecture == "minimax_h3_ref2va"
                    else self.profile(
                        model,
                        vram=vram,
                        ram=ram,
                        resolution=resolution,
                    )
                )
                director = build_director_video_execution_profile(
                    architecture,
                    model,
                    {"resolution": resolution},
                    {"gpu_vram_gb": vram, "ram_gb": ram},
                )
                tier = studio.get("resource_recommendation") or {}
                self.assertEqual(
                    director["recommended_max_frames"],
                    studio["frames"],
                )
                self.assertEqual(
                    director["hardware_supported"],
                    studio["supported"],
                )
                self.assertEqual(director["system_ram_gb"], ram)
                self.assertEqual(
                    director["recommendation_basis"],
                    tier.get("basis", "conservative"),
                )
                self.assertEqual(
                    director["recommendation_note"],
                    tier.get("note", ""),
                )
                self.assertEqual(
                    director["resource_recommendation_version"],
                    studio.get("resource_recommendation_version"),
                )
                limits = resolve_clip_limits(model, execution_profile=director)
                self.assertEqual(limits["max_frames"], studio["frames"])
                self.assertEqual(limits["recommended_frames"], studio["frames"])
                self.assertEqual(limits["recommendation_basis"], tier["basis"])
                self.assertEqual(limits["recommendation_note"], tier.get("note", ""))
                self.assertEqual(
                    limits["recommendation_version"],
                    studio["resource_recommendation_version"],
                )

    def test_director_manual_window_survives_supported_and_unsupported_profiles(self):
        fused = self.model(
            "minimax_h3",
            {"minimax_h3_fused_turbo": True},
        )
        hardware = {"gpu_vram_gb": 12, "ram_gb": 32}
        director_manual = build_director_video_execution_profile(
            "minimax_h3",
            fused,
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
            apply_h3_window_memory_policy(runtime_inputs, fused, hardware)
        )
        self.assertEqual(runtime_inputs["sliding_window_size"], 345)

        unsupported_hardware = {"gpu_vram_gb": 8, "ram_gb": 16}
        unsupported_runtime_inputs = {
            "video_length": 362,
            "sliding_window_size": 124,
            "resolution": "1280x704",
            "sliding_window_memory_override": True,
        }
        self.assertIsNone(
            apply_h3_window_memory_policy(
                unsupported_runtime_inputs,
                fused,
                unsupported_hardware,
            )
        )
        self.assertEqual(
            unsupported_runtime_inputs["sliding_window_size"],
            124,
        )

        with self.assertRaisesRegex(ValueError, "no automatic one-pass profile"):
            build_director_video_execution_profile(
                "minimax_h3",
                fused,
                {"resolution": "1280x704"},
                unsupported_hardware,
            )

        manual_short_start = build_director_video_execution_profile(
            "minimax_h3",
            fused,
            {"resolution": "1280x704"},
            unsupported_hardware,
            manual_max_frames=124,
        )
        self.assertFalse(manual_short_start["hardware_supported"])
        self.assertTrue(manual_short_start["manual_override"])
        self.assertIsNone(manual_short_start["recommended_max_frames"])
        self.assertEqual(manual_short_start["effective_max_frames"], 124)
        self.assertEqual(manual_short_start["fallback_resolution"], "864x480")
        self.assertEqual(manual_short_start["recommendation_basis"], "estimated")

    def test_runtime_auto_helpers_use_resource_rows_without_changing_manual_values(self):
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
        self.assertEqual(
            sequence_inputs["minimax_h3_sequence_clip_frames"],
            345,
        )

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
