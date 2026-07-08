"""
Tests for the three-tier post-Pass-2 duration enforcement.

User-reported lessons from production:
  - 332s output for 180s target → previous fix was proportional scale.
  - 200s output for 180s target → previous fix scaled to 17s shots,
    user complained "scaling 20s shots to 17s is pointless."

Three-tier policy resolves both:
  Tier 1 (≤15% over): accept the overshoot, log it. The user-reported
    200s/180s case (11% over) sits here — no compression at all,
    clean buckets preserved.
  Tier 2 (15-50% over): bucket-aware reduction. Snap shots from larger
    buckets to smaller (40→20, 60→40, 80→60). Drops the last window's
    content per snap; preserves the bucket grid.
  Tier 3 (>50% over): proportional scale + bucket cleanup. Last resort
    for runaway LLMs. Final result is on bucket grid even if total
    misses target slightly after rounding.

Helpers below mirror the production logic in short_film.py. If
production logic changes, these helpers must be updated.
"""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


def _snap_bucket(sd: dict) -> None:
    """Mirror of the inline _snap_bucket in short_film.py."""
    d = int(sd.get("duration_sec", 0) or 0)
    if d <= 0 or d in (20, 40, 60, 80):
        return
    if d < 20:
        new_d = 20
    else:
        tail = d % 20
        if tail == 0:
            return
        new_d = (d - tail) if tail <= 10 else (d + (20 - tail))
        new_d = max(20, new_d)
    sd["duration_sec"] = new_d
    n_target = max(1, new_d // 20)
    wps = sd.get("window_prompts") or []
    if new_d == 20 and wps:
        sd["video_prompt"] = " ".join(str(w) for w in wps)
        sd["window_prompts"] = []
    elif new_d > 20 and len(wps) > n_target:
        kept = list(wps[:n_target - 1])
        merged = " ".join(str(w) for w in wps[n_target - 1:])
        kept.append(merged)
        sd["window_prompts"] = kept


def _enforce(shot_dicts: list[dict], target_duration: int) -> tuple[str, int, int]:
    """Mirror of the three-tier duration enforcement in short_film.py.

    Returns (tier, raw_total, new_total) where tier ∈
    {"accept", "bucket_down", "proportional"}.
    """
    raw_total = sum(
        int(sd.get("duration_sec", 0) or 0)
        for sd in shot_dicts
        if isinstance(sd, dict)
    )
    ceiling = int(target_duration * 1.15)
    scale_threshold = int(target_duration * 1.50)

    if raw_total <= ceiling:
        return "accept", raw_total, raw_total

    if raw_total <= scale_threshold:
        bucket_down = {80: 60, 60: 40, 40: 20}
        excess = raw_total - target_duration
        candidates = sorted(
            [sd for sd in shot_dicts
             if isinstance(sd, dict)
             and sd.get("duration_sec") in bucket_down],
            key=lambda s: -int(s.get("duration_sec", 0) or 0),
        )
        for sd in candidates:
            if excess <= 0:
                break
            cur = int(sd.get("duration_sec", 0) or 0)
            nxt = bucket_down[cur]
            wps = list(sd.get("window_prompts") or [])
            if wps:
                wps = wps[:-1]
                if nxt == 20:
                    sd["video_prompt"] = (
                        " ".join(str(w) for w in wps) if wps else
                        sd.get("video_prompt", "") or ""
                    )
                    sd["window_prompts"] = []
                else:
                    sd["window_prompts"] = wps
            sd["duration_sec"] = nxt
            excess -= (cur - nxt)
        for sd in shot_dicts:
            if isinstance(sd, dict):
                _snap_bucket(sd)
        new_total = sum(
            int(sd.get("duration_sec", 0) or 0)
            for sd in shot_dicts
            if isinstance(sd, dict)
        )
        return "bucket_down", raw_total, new_total

    # Tier 3
    scale = target_duration / raw_total if raw_total else 1.0
    for sd in shot_dicts:
        if not isinstance(sd, dict):
            continue
        old_dur = int(sd.get("duration_sec", 0) or 0)
        if old_dur <= 0:
            continue
        sd["duration_sec"] = max(3, int(old_dur * scale))
    for sd in shot_dicts:
        if isinstance(sd, dict):
            _snap_bucket(sd)
    new_total = sum(
        int(sd.get("duration_sec", 0) or 0)
        for sd in shot_dicts
        if isinstance(sd, dict)
    )
    return "proportional", raw_total, new_total


class TestTier1Accept(unittest.TestCase):
    """Within 15% of target — accept and move on."""

    def test_under_target_accepts(self):
        shots = [{"duration_sec": 20} for _ in range(8)]  # 160s
        tier, _raw, _new = _enforce(shots, 180)
        self.assertEqual(tier, "accept")
        # No durations changed
        self.assertTrue(all(s["duration_sec"] == 20 for s in shots))

    def test_user_reported_200s_over_180s_accepts(self):
        # The exact production scenario the user reported as "scaling
        # to 17s is pointless." 200s vs 180s = 11% over. With the new
        # 15% headroom, this MUST land in Tier 1 (accept) and preserve
        # all 20s buckets.
        shots = [{"duration_sec": 20} for _ in range(8)]
        shots.append({"duration_sec": 40, "window_prompts": ["W1", "W2"]})  # total 200
        tier, raw, new = _enforce(shots, 180)
        self.assertEqual(tier, "accept",
                         "200s vs 180s must NOT trigger compression — "
                         "user explicitly asked for clean buckets over exact target")
        self.assertEqual(raw, 200)
        self.assertEqual(new, 200)
        # Buckets preserved
        for s in shots:
            self.assertIn(s["duration_sec"], (20, 40, 60, 80))

    def test_just_under_ceiling_accepts(self):
        # int(180 * 1.15) = 206 due to float precision (180 * 1.15 =
        # 206.99...). Test the boundary at 206 (must accept).
        shots = [{"duration_sec": 20} for _ in range(8)]
        shots.append({"duration_sec": 46})  # total 206
        tier, _raw, _new = _enforce(shots, 180)
        self.assertEqual(tier, "accept",
                         "206s = ceiling for 180s target — must accept")


class TestTier2BucketDown(unittest.TestCase):
    """15-50% over — snap large shots to smaller buckets."""

    def test_one_40s_snaps_to_20s(self):
        # 220s vs 180s = 22% over. Has one 40s shot to snap down.
        shots = [{"duration_sec": 20} for _ in range(8)]
        shots.append({"duration_sec": 40, "window_prompts": ["W1", "W2"]})
        shots.append({"duration_sec": 20})  # total 220
        tier, raw, new = _enforce(shots, 180)
        self.assertEqual(tier, "bucket_down")
        self.assertEqual(raw, 220)
        # The 40s shot should snap to 20s, removing 20s.
        self.assertEqual(new, 200)  # 220 - 20
        # All durations should be in bucket
        for s in shots:
            self.assertIn(s["duration_sec"], (20, 40, 60, 80))

    def test_60s_snaps_to_40s_first(self):
        # 240s vs 180s = 33% over. Has 60s and 40s; should snap 60→40
        # first (preserves more sustained scenes).
        shots = [{"duration_sec": 20} for _ in range(7)]
        shots.append({"duration_sec": 40, "window_prompts": ["W1", "W2"]})
        shots.append({"duration_sec": 60, "window_prompts": ["W1", "W2", "W3"]})
        # total 7*20 + 40 + 60 = 240
        tier, raw, _new = _enforce(shots, 180)
        self.assertEqual(tier, "bucket_down")
        self.assertEqual(raw, 240)
        # The 60s shot should have been snapped first.
        sixty_shots = [s for s in shots if s["duration_sec"] == 60]
        self.assertEqual(len(sixty_shots), 0,
                         "60s shot should have been snapped down first")

    def test_no_candidates_accepts_overshoot(self):
        # All 20s shots, total way over → no bucket-down candidates.
        # Must accept the overshoot rather than chop content.
        shots = [{"duration_sec": 20} for _ in range(12)]  # 240s
        tier, raw, new = _enforce(shots, 180)
        self.assertEqual(tier, "bucket_down")
        # No snaps possible — total stays at 240.
        self.assertEqual(new, 240)


class TestTier3Proportional(unittest.TestCase):
    """>50% over — runaway LLM, fall back to proportional + cleanup."""

    def test_severe_overshoot_uses_proportional(self):
        # 332s vs 180s = 84% over. Tier 3.
        shots = []
        for _ in range(15):
            shots.append({"duration_sec": 20})
        for _ in range(8):
            shots.append({"duration_sec": 4})
        # total 15*20 + 8*4 = 332
        tier, raw, new = _enforce(shots, 180)
        self.assertEqual(tier, "proportional")
        self.assertEqual(raw, 332)
        # After scale + bucket cleanup, all shots should be on the
        # bucket grid.
        for s in shots:
            self.assertIn(s["duration_sec"], (20, 40, 60, 80),
                          f"non-bucket dur after cleanup: {s['duration_sec']}")


class TestSnapBucketHelper(unittest.TestCase):
    """The _snap_bucket helper used by Tier 2 and 3 cleanup."""

    def test_already_bucket_no_change(self):
        for d in (20, 40, 60, 80):
            shot = {"duration_sec": d}
            _snap_bucket(shot)
            self.assertEqual(shot["duration_sec"], d)

    def test_18s_rounds_up_to_20(self):
        shot = {"duration_sec": 18}
        _snap_bucket(shot)
        self.assertEqual(shot["duration_sec"], 20)

    def test_17s_rounds_up_to_20(self):
        # The user-reported "pointless 17s" scenario.
        shot = {"duration_sec": 17}
        _snap_bucket(shot)
        self.assertEqual(shot["duration_sec"], 20)

    def test_25s_rounds_down_to_20(self):
        shot = {"duration_sec": 25, "window_prompts": ["W1 long", "W2 short"]}
        _snap_bucket(shot)
        self.assertEqual(shot["duration_sec"], 20)
        # Windows should have been merged into video_prompt
        self.assertEqual(shot["window_prompts"], [])
        self.assertIn("W1 long", shot["video_prompt"])
        self.assertIn("W2 short", shot["video_prompt"])

    def test_35s_rounds_up_to_40(self):
        shot = {"duration_sec": 35, "window_prompts": ["W1", "W2"]}
        _snap_bucket(shot)
        self.assertEqual(shot["duration_sec"], 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
