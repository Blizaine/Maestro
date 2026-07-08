"""
Tests for the adjacent-shot merge that catches over-fragmented Pass 2 output.

User-reported regression: my previous commit's strict bucket rule + new
anti-fragmentation prose actively confused the LLM, which produced 36
shots of 3-8s each for a 180s target — every micro-action fragmented.
The per-shot snap-up (5s → 20s × 36) would have produced 720s total,
which the duration scale-down would have crushed back to ~5s each. Both
the worst of both worlds.

Fix: detect over-fragmentation BEFORE the per-shot post-process. When
shot count exceeds 1.3× the realistic max, merge adjacent short shots
(≤15s, no windows) into 20s shots. Concatenate video_prompts in order,
keep the first shot's image_prompt, drop keyframes (stale after merge).

Stop-merging boundaries:
  - Long shot (>15s) or multi-window shot
  - environment changes (real location boundary)
  - scene_type changes (dialogue → action)
  - Bucket would exceed 20s

The merge logic lives inline in short_film.py. This test mirrors it via
a small helper.
"""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


def _merge_short_shots(shot_dicts: list[dict], target_duration: int) -> list[dict]:
    """Mirror of the adjacent-shot merge logic in short_film.py.

    Returns the new list (may be shorter than input). Does NOT mutate
    the input list reference — caller can compare counts.
    """
    max_shots = max(2, target_duration // 15)
    if len(shot_dicts) <= max_shots * 1.3 or not shot_dicts:
        return list(shot_dicts)

    merged_shots: list[dict] = []
    bucket: list[dict] = []
    bucket_dur = 0

    def _flush():
        nonlocal bucket, bucket_dur
        if not bucket:
            return
        if len(bucket) == 1:
            merged_shots.append(bucket[0])
        else:
            head = dict(bucket[0])
            prompts = []
            for s in bucket:
                vp = (s.get("video_prompt") or "").strip()
                if vp:
                    prompts.append(vp)
            if prompts:
                head["video_prompt"] = " ".join(prompts)
            head["window_prompts"] = []
            head["duration_sec"] = 20
            head["keyframe_prompts"] = []
            all_beats: list = []
            for s in bucket:
                ab = s.get("action_beats") or []
                if isinstance(ab, list):
                    all_beats.extend(ab)
            if all_beats:
                head["action_beats"] = all_beats
            merged_shots.append(head)
        bucket = []
        bucket_dur = 0

    for sd in shot_dicts:
        if not isinstance(sd, dict):
            merged_shots.append(sd)  # type: ignore[arg-type]
            continue
        dur = int(sd.get("duration_sec", 0) or 0)
        has_windows = bool(sd.get("window_prompts"))
        is_short = (0 < dur <= 15) and not has_windows
        boundary = False
        if bucket and is_short:
            head = bucket[0]
            if (sd.get("environment") and head.get("environment")
                    and sd.get("environment") != head.get("environment")):
                boundary = True
            if (sd.get("scene_type") and head.get("scene_type")
                    and sd.get("scene_type") != head.get("scene_type")):
                boundary = True
        if not is_short or boundary:
            _flush()
            merged_shots.append(sd)
            continue
        if bucket_dur + dur > 20 and bucket:
            _flush()
        bucket.append(sd)
        bucket_dur += dur
    _flush()
    return merged_shots


class TestAdjacentShotMerge(unittest.TestCase):

    def test_no_merge_when_under_threshold(self):
        # 9 shots for 180s target — at the realistic max, no merge.
        shots = [{"duration_sec": 20, "video_prompt": f"S{i}"} for i in range(9)]
        merged = _merge_short_shots(shots, 180)
        self.assertEqual(len(merged), 9, "no merge when count is reasonable")

    def test_no_merge_when_shots_are_long(self):
        # 30 shots, all 20s — count is high but each shot is full-bucket.
        # Merge condition is "short shots only" so nothing should merge.
        shots = [{"duration_sec": 20, "video_prompt": f"S{i}"} for i in range(30)]
        merged = _merge_short_shots(shots, 180)
        # 30 > 12 * 1.3 = 15.6 → over threshold, but nothing is short
        # enough to merge.
        self.assertEqual(len(merged), 30, "long shots don't get merged")

    def test_user_reported_36_short_shots_180s(self):
        # The exact production scenario: 36 shots of 3-8s each.
        shots = []
        for i in range(36):
            dur = 3 + (i % 6)  # 3, 4, 5, 6, 7, 8, repeating
            shots.append({"duration_sec": dur, "video_prompt": f"S{i}"})
        merged = _merge_short_shots(shots, 180)
        # Should reduce significantly — exact count depends on packing.
        self.assertLess(len(merged), 36)
        # Each merged shot should be 20s.
        for s in merged:
            self.assertEqual(s["duration_sec"], 20,
                             f"merged shot has dur {s['duration_sec']}, expected 20")
        # Total should be approximately the original total (we packed
        # 3-8s into 20s buckets, so each bucket holds 3-6 shots).
        original_total = sum(s["duration_sec"] for s in shots)
        merged_total = sum(s["duration_sec"] for s in merged)
        # We're not preserving exact total — packing into 20s buckets
        # rounds up. Just verify total is in a reasonable range.
        self.assertGreaterEqual(merged_total, original_total)

    def test_video_prompts_concatenated_in_order(self):
        # 4 short shots, each 5s, should merge into one 20s shot
        # with all 4 video_prompts concatenated in order.
        shots = [
            {"duration_sec": 5, "video_prompt": "ALPHA"},
            {"duration_sec": 5, "video_prompt": "BETA"},
            {"duration_sec": 5, "video_prompt": "GAMMA"},
            {"duration_sec": 5, "video_prompt": "DELTA"},
        ] * 4  # 16 shots total to trigger merge for 60s target (max=4)
        merged = _merge_short_shots(shots, 60)
        self.assertLess(len(merged), 16)
        # First merged shot should contain ALPHA, BETA, GAMMA, DELTA in
        # that order.
        first = merged[0]
        first_vp = first.get("video_prompt", "")
        # Verify first 4 prompts are merged into the first bucket.
        self.assertIn("ALPHA", first_vp)
        self.assertIn("BETA", first_vp)
        # Order check: ALPHA must come before DELTA.
        self.assertLess(first_vp.find("ALPHA"), first_vp.find("DELTA"))

    def test_environment_change_breaks_merge(self):
        # 16 short shots in 60s target (> threshold), but the 8th has
        # a different environment — that boundary should split into
        # at least 2 distinct merged-shot groups.
        shots = []
        for i in range(16):
            shots.append({
                "duration_sec": 5,
                "video_prompt": f"S{i}",
                "environment": "courtyard" if i < 8 else "chamber",
            })
        merged = _merge_short_shots(shots, 60)
        # First merged shot should only contain courtyard prompts.
        # Find the boundary between courtyard and chamber merged shots.
        courtyard_envs = [s for s in merged if s.get("environment") == "courtyard"]
        chamber_envs = [s for s in merged if s.get("environment") == "chamber"]
        self.assertGreater(len(courtyard_envs), 0)
        self.assertGreater(len(chamber_envs), 0)
        # No merged shot should contain prompts from both environments.
        for s in merged:
            vp = s.get("video_prompt", "")
            if "S0" in vp or "S7" in vp:  # courtyard
                # Must not contain any chamber prompt
                for ci in range(8, 16):
                    self.assertNotIn(f"S{ci}", vp,
                                     f"courtyard merged shot leaked S{ci} (chamber)")

    def test_long_shot_in_middle_breaks_merge(self):
        # Several short shots, then a long shot, then more short shots.
        # The long shot should not get merged into anything; the short
        # shots before/after merge into separate groups.
        shots = [
            {"duration_sec": 5, "video_prompt": "S0"},
            {"duration_sec": 5, "video_prompt": "S1"},
            {"duration_sec": 5, "video_prompt": "S2"},
            {"duration_sec": 40, "video_prompt": "LONG", "window_prompts": ["W1", "W2"]},
            {"duration_sec": 5, "video_prompt": "S4"},
            {"duration_sec": 5, "video_prompt": "S5"},
            {"duration_sec": 5, "video_prompt": "S6"},
            {"duration_sec": 5, "video_prompt": "S7"},
            {"duration_sec": 5, "video_prompt": "S8"},
            {"duration_sec": 5, "video_prompt": "S9"},
        ] * 2  # 20 shots total
        merged = _merge_short_shots(shots, 60)
        # Long shot should appear unchanged with its windows intact.
        long_shots = [s for s in merged
                      if s.get("video_prompt") == "LONG"
                      and s.get("window_prompts")]
        self.assertGreater(len(long_shots), 0,
                           "long shot lost its window_prompts during merge")

    def test_does_not_overflow_20s_bucket(self):
        # Each merged shot's bucket should not exceed 20s. With 6s
        # individual shots, the bucket should flush after 3 shots
        # (3*6=18s, +6=24>20).
        shots = [{"duration_sec": 6, "video_prompt": f"S{i}"} for i in range(20)]
        merged = _merge_short_shots(shots, 60)
        for s in merged:
            self.assertLessEqual(s["duration_sec"], 20,
                                 f"bucket exceeded 20s: {s['duration_sec']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
