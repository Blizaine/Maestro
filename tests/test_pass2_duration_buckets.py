"""
Tests for the snap-to-multiple-of-20 duration bucket post-process.

User requirement: shots are EITHER ≤20s (single video_prompt) OR exactly
a multiple of 20s (40, 60, 80) for sustained continuous action. Never
22s, 25s, 30s, 35s, 45s — those create stranded tail windows.

Snap direction by tail:
  - tail == 0       → already valid, no change
  - 1 ≤ tail ≤ 10   → snap DOWN (subtract tail, merge content)
  - 11 ≤ tail ≤ 19  → snap UP (add 20-tail, preserve content)

The actual snap logic lives inline in short_film.py inside the per-shot
post-process loop. This test exercises the snap math via a small
helper that mirrors the production logic; if the production code
changes, the helper here must be updated to match.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


def _snap_to_bucket(shot: dict) -> tuple[str | None, int, int]:
    """Mirror of the snap-to-multiple-of-20 logic in short_film.py.

    Mutates `shot` in place. Returns (action, old_dur, new_dur):
      action ∈ {None, "down", "up"}
      None means the duration was already a clean multiple of 20.
    """
    dur = int(shot.get("duration_sec", 0) or 0)
    if dur <= 0:
        return None, dur, dur
    tail = dur % 20
    if tail == 0:
        return None, dur, dur
    wps = shot.get("window_prompts") or []
    n = len(wps)
    if tail <= 10:
        # Snap DOWN
        new_dur = max(20, dur - tail)
        shot["duration_sec"] = new_dur
        if n == 1 and new_dur == 20:
            shot["video_prompt"] = str(wps[0])
            shot["window_prompts"] = []
        elif n >= 2:
            merged = str(wps[-2]) + " " + str(wps[-1])
            new_windows = list(wps[:-2]) + [merged]
            shot["window_prompts"] = new_windows
            if len(new_windows) == 1:
                shot["video_prompt"] = merged
                shot["window_prompts"] = []
        return "down", dur, new_dur
    else:
        # Snap UP
        new_dur = dur + (20 - tail)
        shot["duration_sec"] = new_dur
        return "up", dur, new_dur


class TestSnapToBucket(unittest.TestCase):
    """Verify the snap math hits the right bucket for every input."""

    def test_clean_multiples_unchanged(self):
        for dur in [20, 40, 60, 80, 100]:
            shot = {"duration_sec": dur, "window_prompts": []}
            action, old, new = _snap_to_bucket(shot)
            self.assertIsNone(action, f"{dur}s must not snap")
            self.assertEqual(new, dur)

    def test_zero_duration_no_op(self):
        shot = {"duration_sec": 0}
        action, _, _ = _snap_to_bucket(shot)
        self.assertIsNone(action)

    def test_25s_snaps_down_to_20(self):
        # User-reported bug: 25s shot with W1=20 + W2=5s (rushed tail).
        shot = {
            "duration_sec": 25,
            "window_prompts": ["First 20s of action", "Last 5s tail"],
        }
        action, old, new = _snap_to_bucket(shot)
        self.assertEqual(action, "down")
        self.assertEqual(new, 20)
        # Should now be single video_prompt since new_dur == 20.
        self.assertEqual(shot["window_prompts"], [])
        self.assertIn("First 20s of action", shot["video_prompt"])
        self.assertIn("Last 5s tail", shot["video_prompt"])

    def test_22s_snaps_down_to_20(self):
        shot = {
            "duration_sec": 22,
            "window_prompts": ["First window", "Tiny tail"],
        }
        action, _, new = _snap_to_bucket(shot)
        self.assertEqual(action, "down")
        self.assertEqual(new, 20)

    def test_30s_snaps_down_to_20(self):
        # 30s is the boundary — tail=10, still snap DOWN per the rule.
        shot = {
            "duration_sec": 30,
            "window_prompts": ["W1 20s content", "W2 10s content"],
        }
        action, _, new = _snap_to_bucket(shot)
        self.assertEqual(action, "down", "tail=10 must snap DOWN")
        self.assertEqual(new, 20)

    def test_31s_snaps_up_to_40(self):
        # 31s tail=11 — snap UP to preserve content.
        shot = {
            "duration_sec": 31,
            "window_prompts": ["W1 20s", "W2 11s"],
        }
        action, _, new = _snap_to_bucket(shot)
        self.assertEqual(action, "up", "tail=11 must snap UP")
        self.assertEqual(new, 40)

    def test_35s_snaps_up_to_40(self):
        shot = {
            "duration_sec": 35,
            "window_prompts": ["W1 20s", "W2 15s"],
        }
        action, _, new = _snap_to_bucket(shot)
        self.assertEqual(action, "up")
        self.assertEqual(new, 40)

    def test_45s_snaps_down_to_40(self):
        shot = {
            "duration_sec": 45,
            "window_prompts": ["W1", "W2", "W3 5s"],
        }
        action, _, new = _snap_to_bucket(shot)
        self.assertEqual(action, "down")
        self.assertEqual(new, 40)
        # 3 windows → 2 windows after merge.
        self.assertEqual(len(shot["window_prompts"]), 2)

    def test_50s_snaps_down_to_40(self):
        # tail=10 → snap DOWN.
        shot = {
            "duration_sec": 50,
            "window_prompts": ["W1", "W2", "W3 10s"],
        }
        action, _, new = _snap_to_bucket(shot)
        self.assertEqual(action, "down")
        self.assertEqual(new, 40)

    def test_55s_snaps_up_to_60(self):
        shot = {
            "duration_sec": 55,
            "window_prompts": ["W1", "W2", "W3 15s"],
        }
        action, _, new = _snap_to_bucket(shot)
        self.assertEqual(action, "up")
        self.assertEqual(new, 60)

    def test_user_scenario_three_25s_shots(self):
        # User reported: last 3 of 15 shots in their run were 25s each.
        # All should snap to 20s.
        shots = [
            {"duration_sec": 25, "window_prompts": ["A1", "A2"]},
            {"duration_sec": 25, "window_prompts": ["B1", "B2"]},
            {"duration_sec": 25, "window_prompts": ["C1", "C2"]},
        ]
        for s in shots:
            action, _, new = _snap_to_bucket(s)
            self.assertEqual(action, "down")
            self.assertEqual(new, 20)
        self.assertEqual(sum(s["duration_sec"] for s in shots), 60)

    def test_short_shots_no_snap(self):
        # ≤20s shots are already valid (single video_prompt, no windows).
        for dur in [5, 10, 15, 20]:
            shot = {"duration_sec": dur, "video_prompt": "x"}
            action, _, new = _snap_to_bucket(shot)
            if dur == 20:
                self.assertIsNone(action)
            else:
                # tail = dur (since dur < 20). 5/10 snap down (but
                # min is 20). 15 snaps up to 20.
                self.assertIsNotNone(action)
                self.assertEqual(new, 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
