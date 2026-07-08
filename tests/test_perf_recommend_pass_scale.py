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

from services.perf_recommend import compute_per_job_coefficient  # noqa: E402


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
