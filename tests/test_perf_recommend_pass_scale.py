"""
Tests for the resolution-scaled pass overhead in compute_per_job_coefficient.

User-reported OOM: 20s × 1080p × 2-stage × no-LoRA generation on RTX
4090 (24 GB VRAM) ran out of memory at stage 2. The previous flat
2 GB pass overhead per extra stage was calibrated against 720p stage
2; at 1080p, stage 2's activation peak grows ~2.25× because it
operates at full output resolution.

Fix: scale _PASS_OVERHEAD_GB_PER_EXTRA_STAGE by the output
resolution factor (output_pixels / 720p_pixels), but ONLY UPWARD.
Low-res jobs (≤720p) get unchanged behavior because the existing
2 GB constant was empirically validated for 480/540/720p and those
resolutions already work fine — loosening their caps risks regressing
working behavior. The scaling is purely additive for high-res jobs.
"""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.perf_recommend import (  # noqa: E402
    compute_h3_weight_budget,
    compute_music3_weight_budget,
    compute_per_job_coefficient,
)


class TestPassOverheadResolutionScaling(unittest.TestCase):
    """The pass_penalty for multi-stage pipelines now scales with output
    resolution. 720p baseline = 1.0×; 1080p = 2.25×; 4K = 9×."""

    def test_720p_baseline_unchanged(self):
        # 10s × 720p × 2-stage × no-LoRA × RTX 4090.
        # res_scale = 1.0 → pass_overhead unchanged from previous value.
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=2,
            resolution="1280x720",
            video_length_frames=240,  # 10s @ 24fps
        )
        # Pass penalty: 2.0 GB / 24 GB ≈ 0.083
        self.assertAlmostEqual(result["pass_penalty"], 0.083, places=2)

    def test_540p_does_not_loosen(self):
        # User confirmed 480/540/720 already work fine. Asymmetric
        # scaling with max(1.0, ...) means low-res jobs keep the
        # existing 2 GB pass overhead — never less.
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=2,
            resolution="960x540",
            video_length_frames=240,
        )
        # Should stay at 2 GB, NOT scale down to 0.56× × 2 = 1.12 GB.
        self.assertAlmostEqual(result["pass_penalty"], 0.083, places=2)

    def test_480p_does_not_loosen(self):
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=2,
            resolution="854x480",
            video_length_frames=240,
        )
        self.assertAlmostEqual(result["pass_penalty"], 0.083, places=2)

    def test_1080p_user_reported_oom_case(self):
        # The exact case from the user report: 20s × 1080p × 2-stage ×
        # no-LoRA × RTX 4090. With the resolution scaling, pass overhead
        # jumps from 2.0 → 4.5 GB (2.25× scale), tightening the cap.
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=2,
            resolution="1920x1080",
            video_length_frames=480,  # 20s @ 24fps
        )
        # Pass penalty: 4.5 GB / 24 GB ≈ 0.188 (was 0.083 before fix)
        self.assertAlmostEqual(result["pass_penalty"], 0.188, places=2)
        # Combined effective should drop from 0.513 → ~0.408
        self.assertAlmostEqual(result["effective_coef"], 0.408, places=2)

    def test_4k_extreme_case(self):
        # 4K stage 2 is 9× the baseline pixel count. Should clamp at floor.
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=2,
            resolution="3840x2160",
            video_length_frames=240,
        )
        # 9× scale × 2 GB = 18 GB / 24 = 0.75 pass penalty alone
        self.assertAlmostEqual(result["pass_penalty"], 0.75, places=2)
        # raw_effective = 0.80 - 0.75 - compute_penalty → likely floored
        # at 0.40 (model probably can't run 4K stage 2 anyway, but the
        # clamp prevents a nonsense negative coefficient)
        self.assertGreaterEqual(result["effective_coef"], 0.40)

    def test_single_stage_no_pass_penalty(self):
        # stage_count=1 → no extra stages → no pass_penalty regardless
        # of resolution.
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=1,
            resolution="1920x1080",
            video_length_frames=480,
        )
        self.assertEqual(result["pass_penalty"], 0.0)

    def test_progressive_3stage_at_1080p(self):
        # 3-stage progressive at 1080p → 2 extra stages × 2 GB × 2.25 = 9 GB
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=3,
            resolution="1920x1080",
            video_length_frames=480,
        )
        # 9 GB / 24 = 0.375
        self.assertAlmostEqual(result["pass_penalty"], 0.375, places=2)

    def test_no_resolution_keeps_baseline_penalty(self):
        # When resolution is unknown, fall back to 1.0× scale.
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=2,
            resolution=None,
            video_length_frames=240,
        )
        # res_scale = 1.0 → pass_overhead unchanged at 2.0 GB / 24 = 0.083
        self.assertAlmostEqual(result["pass_penalty"], 0.083, places=2)

    def test_reasons_log_includes_scale_note(self):
        # Heavy resolution case should mention the scale factor in the
        # reasons[] log so users can see why the penalty is higher.
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=2,
            resolution="1920x1080",
            video_length_frames=480,
        )
        joined = " ".join(result["reasons"])
        self.assertIn("resolution scale", joined)
        self.assertIn("2.25", joined)

    def test_reasons_log_omits_scale_note_at_baseline(self):
        # 720p (and below) doesn't get the scale note because the
        # asymmetric fix keeps res_scale = 1.0 there.
        result = compute_per_job_coefficient(
            base_coef=0.80,
            total_vram_gb=24.0,
            active_loras=[],
            stage_count=2,
            resolution="1280x720",
            video_length_frames=240,
        )
        joined = " ".join(result["reasons"])
        self.assertNotIn("resolution scale", joined)


class TestMiniMaxH3ActivationBudget(unittest.TestCase):
    def test_long_540p_job_restores_known_good_4090_headroom(self):
        budget = compute_h3_weight_budget(
            24.0,
            "960x544",
            336,
            runtime_workspace_gb=10.0,
        )
        self.assertLessEqual(budget["weight_budget_gb"], 14.0)
        self.assertGreaterEqual(budget["activation_reserve_gb"], 10.0)

    def test_portrait_and_landscape_have_the_same_budget(self):
        landscape = compute_h3_weight_budget(24.0, "960x544", 336)
        portrait = compute_h3_weight_budget(24.0, "544x960", 336)
        self.assertEqual(landscape, portrait)

    def test_short_480p_job_keeps_more_weights_but_never_exceeds_cap(self):
        budget = compute_h3_weight_budget(24.0, "864x480", 124)
        self.assertGreater(budget["weight_budget_gb"], 17.5)
        self.assertLessEqual(budget["weight_budget_gb"], 18.0)

    def test_video_reference_adds_full_attention_reserve(self):
        budget = compute_h3_weight_budget(24.0, "960x544", 345, 1)
        self.assertAlmostEqual(budget["activation_reserve_gb"], 17.0, places=2)
        self.assertAlmostEqual(budget["weight_budget_gb"], 7.0, places=2)

    def test_lower_vram_card_streams_more_transformer_weights(self):
        budget = compute_h3_weight_budget(
            16.0,
            "960x544",
            345,
            runtime_workspace_gb=10.0,
        )
        self.assertAlmostEqual(budget["activation_reserve_gb"], 11.0, places=2)
        self.assertAlmostEqual(budget["weight_budget_gb"], 5.0, places=2)

    def test_native_full_window_residency_honors_runtime_workspace(self):
        budget = compute_h3_weight_budget(
            24.0,
            "960x544",
            345,
            runtime_workspace_gb=10.0,
        )
        self.assertTrue(budget["runtime_scaling_active"])
        self.assertAlmostEqual(budget["scaled_runtime_workspace_gb"], 10.0, places=2)
        self.assertAlmostEqual(budget["activation_reserve_gb"], 11.0, places=2)
        self.assertAlmostEqual(budget["weight_budget_gb"], 13.0, places=2)

    def test_runtime_workspace_blend_has_no_large_threshold_jump(self):
        below = compute_h3_weight_budget(
            24.0,
            "960x544",
            310,
            runtime_workspace_gb=10.0,
        )
        above = compute_h3_weight_budget(
            24.0,
            "960x544",
            311,
            runtime_workspace_gb=10.0,
        )
        self.assertLess(
            abs(
                above["activation_reserve_gb"]
                - below["activation_reserve_gb"]
            ),
            0.1,
        )

    def test_pruned_768p_full_window_preserves_step_zero_headroom_on_4090(self):
        budget = compute_h3_weight_budget(
            24.0,
            "1344x768",
            345,
            runtime_workspace_gb=10.0,
        )
        self.assertTrue(budget["runtime_scaling_active"])
        self.assertGreater(budget["weight_budget_gb"], 6.0)
        self.assertLess(budget["weight_budget_gb"], 6.5)
        self.assertGreater(budget["activation_reserve_gb"], 17.5)
        self.assertEqual(budget["runtime_safety_margin_gb"], 1.0)

    def test_aligned_720p_retains_more_transformer_residency_than_768p(self):
        aligned_720p = compute_h3_weight_budget(
            24.0,
            "1280x704",
            345,
            runtime_workspace_gb=10.0,
        )
        high_768p = compute_h3_weight_budget(
            24.0,
            "1344x768",
            345,
            runtime_workspace_gb=10.0,
        )
        self.assertGreater(
            aligned_720p["weight_budget_gb"],
            high_768p["weight_budget_gb"],
        )
        self.assertGreater(aligned_720p["weight_budget_gb"], 7.5)
        self.assertLess(aligned_720p["weight_budget_gb"], 8.5)

    def test_recommended_720p_window_keeps_allocator_headroom_on_4090(self):
        budget = compute_h3_weight_budget(
            24.0,
            "1280x704",
            243,
            runtime_workspace_gb=10.0,
        )
        self.assertTrue(budget["runtime_scaling_active"])
        self.assertGreater(budget["activation_reserve_gb"], 12.0)
        self.assertGreater(budget["weight_budget_gb"], 11.0)
        self.assertLess(budget["weight_budget_gb"], 12.0)

    def test_pruned_768p_full_window_reaches_streaming_floor_on_16gb_card(self):
        budget = compute_h3_weight_budget(
            16.0,
            "1344x768",
            345,
            runtime_workspace_gb=10.0,
        )
        self.assertTrue(budget["runtime_scaling_active"])
        self.assertEqual(budget["weight_budget_gb"], 3.5)
        self.assertEqual(budget["activation_reserve_gb"], 12.5)

    def test_pruned_768p_short_window_uses_fixed_mmgp_workspace(self):
        budget = compute_h3_weight_budget(
            16.0,
            "1344x768",
            124,
            runtime_workspace_gb=10.0,
        )
        self.assertFalse(budget["runtime_scaling_active"])
        self.assertEqual(budget["scaled_runtime_workspace_gb"], 0.0)
        self.assertGreater(budget["weight_budget_gb"], 9.0)

    def test_1080p_window_scales_runtime_workspace_on_4090(self):
        budget = compute_h3_weight_budget(
            24.0,
            "1920x1088",
            124,
            runtime_workspace_gb=10.0,
            additional_reserve_gb=0.75,
        )
        self.assertTrue(budget["runtime_scaling_active"])
        self.assertGreater(budget["compute_ratio"], 1.4)
        self.assertGreater(budget["activation_reserve_gb"], 16.0)
        self.assertLess(budget["weight_budget_gb"], 8.0)
        self.assertEqual(budget["runtime_safety_margin_gb"], 1.0)
        self.assertEqual(budget["additional_reserve_gb"], 0.75)
        self.assertAlmostEqual(
            budget["activation_reserve_gb"],
            budget["scaled_runtime_workspace_gb"] + 1.75,
            places=6,
        )

    def test_measured_pruned_1080p_158_frame_pass_keeps_streaming_headroom(self):
        budget = compute_h3_weight_budget(
            24.0,
            "1920x1088",
            158,
            runtime_workspace_gb=10.0,
        )
        self.assertTrue(budget["runtime_scaling_active"])
        self.assertGreater(budget["activation_reserve_gb"], 19.0)
        self.assertLess(budget["activation_reserve_gb"], 19.5)
        self.assertGreater(budget["weight_budget_gb"], 4.5)
        self.assertLess(budget["weight_budget_gb"], 5.0)


class TestMiniMaxMusic3SemanticBudget(unittest.TestCase):
    def test_120_second_job_keeps_the_optimized_planner_viable_on_20gb_card(self):
        budget = compute_music3_weight_budget(20.0, 120)
        self.assertGreater(budget["kv_cache_gb"], 1.1)
        self.assertGreater(budget["runtime_reserve_gb"], 7.8)
        self.assertLess(budget["weight_budget_gb"], 12.2)
        self.assertGreater(budget["weight_budget_gb"], 12.0)

    def test_24gb_card_can_retain_the_complete_qwen_checkpoint(self):
        budget = compute_music3_weight_budget(24.0, 120)
        self.assertEqual(budget["weight_budget_gb"], 16.0)
        self.assertGreaterEqual(budget["runtime_reserve_gb"], 7.8)

    def test_longer_music_reserves_more_cache_and_streams_more_weights(self):
        short = compute_music3_weight_budget(20.0, 30)
        long = compute_music3_weight_budget(20.0, 300)
        self.assertGreater(long["kv_cache_gb"], short["kv_cache_gb"])
        self.assertGreater(
            long["runtime_reserve_gb"],
            short["runtime_reserve_gb"],
        )
        self.assertLess(long["weight_budget_gb"], short["weight_budget_gb"])

    def test_16gb_card_keeps_a_viable_streaming_budget(self):
        budget = compute_music3_weight_budget(16.0, 120)
        self.assertGreater(budget["weight_budget_gb"], 8.0)
        self.assertLess(budget["weight_budget_gb"], 8.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
