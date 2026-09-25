"""Exact speech keeps feasible time when visual comfort estimates are squeezed."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import (
    _apply_h3_filmable_shot_clock, _h3_filmable_shot_durations,
)


class SpeechClockFloorTests(unittest.TestCase):
    def test_real_archive_clock_reduces_elastic_action_time_before_speech(self):
        weights = [2.493, 2.85, 6, 5.25, 2, 2, 2, 1]
        floors = [2.2, 2.533333333, 2, 2, 2, 2, 2, .75]
        hard = [2.2, 2.533333333, 0, 0, 0, 0, 0, 0]
        result = _h3_filmable_shot_durations(
            weights, 13.625, minimums=floors, hard_minimums=hard)
        self.assertAlmostEqual(sum(result), 13.625)
        self.assertGreaterEqual(result[0], hard[0])
        self.assertGreaterEqual(result[1], hard[1])
        self.assertTrue(all(value >= .75 for value in result[2:]))
        self.assertEqual(len(result), 8)

    def test_impossible_hard_timing_does_not_expand_or_drop_shots(self):
        result = _h3_filmable_shot_durations(
            [8, 8, 4], 8, minimums=[8, 8, 4], hard_minimums=[8, 8, 4])
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(sum(result), 8)
        self.assertTrue(any(actual < needed for actual, needed in zip(result, [8, 8, 4])))

    def test_ample_clock_keeps_prior_allocation_and_hard_floor_is_always_respected(self):
        expected = _h3_filmable_shot_durations([3, 4, 2], 14, minimums=[2, 2, 1])
        actual = _h3_filmable_shot_durations(
            [3, 4, 2], 14, minimums=[2, 2, 1], hard_minimums=[2, 0, 0])
        self.assertEqual(actual, expected)
        self.assertGreaterEqual(_h3_filmable_shot_durations(
            [1, 10], 10, minimums=[1, 1], hard_minimums=[3, 0])[0], 3)

    def test_production_clock_preserves_exact_lines_and_source_actions(self):
        lines = {"D3": {"text": "I'll stay here with the plan."},
                 "D4": {"text": "Good. Wait until I'm back outside."}}
        descriptions = ["Len speaks.", "Ada replies.", "Ada unlocks the door.",
                        "Ada opens the door.", "Ada retrieves the binder.",
                        "Ada returns outside.", "Ada locks the door.", "Len watches."]
        assignments = [[dict(beat_id=f"B{i}", source_event_ids=[f"E{i}"],
            description=description, dialogue_ids=([f"D{i+3}"] if i < 2 else []))]
            for i, description in enumerate(descriptions)]
        shots = [dict(action=description, start_seconds=i, end_seconds=i+1,
                      _timing_source_action=description)
                 for i, description in enumerate(descriptions)]
        before = deepcopy((lines, assignments, [s["action"] for s in shots]))
        _apply_h3_filmable_shot_clock(shots, assignments, lines,
            duration=13.625, only_when_squeezed=False)
        self.assertGreaterEqual(shots[0]["end_seconds"] - shots[0]["start_seconds"], 2.19)
        self.assertGreaterEqual(shots[1]["end_seconds"] - shots[1]["start_seconds"], 2.19)
        self.assertEqual(shots[-1]["end_seconds"], 13.625)
        self.assertEqual(before, (lines, assignments, [s["action"] for s in shots]))


if __name__ == "__main__":
    unittest.main()
