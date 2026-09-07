"""Speech pace, clip admission, and exact transcript boundary regressions."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.dialogue_timing import h3_dialogue_schedule
from services.director.h3_dialogue import (
    _h3_dialogue_timing_clause,
    h3_dialogue_budget_violations,
)
from services.h3_story_ledger import (
    H3DialogueTimingError,
    _prepare_render_dialogue_schedule,
)
from services.director.schema import ShotPlan
from services.director.validators.shot_validator import validate_shot_plan


class DialogueTimingTests(unittest.TestCase):
    def test_blaine_line_uses_default_pace_and_silent_tail(self):
        duration, start, end = h3_dialogue_schedule(17, 10.125)
        self.assertEqual(duration, 10.125)
        self.assertEqual(start, 0.25)
        self.assertAlmostEqual(end - start, 17 / 2.8)
        self.assertLess(end, duration)

    def test_long_line_is_not_compressed_into_fifty_five_percent(self):
        duration, start, end = h3_dialogue_schedule(25, 10)
        self.assertAlmostEqual(end - start, 25 / 2.8)
        self.assertGreater(end - start, duration * 0.55)
        self.assertGreater(duration - end, 0)

    def test_dense_lines_can_use_the_full_clip_without_forced_pauses(self):
        for word_count in (28, 29, 30):
            with self.subTest(word_count=word_count):
                self.assertEqual(h3_dialogue_schedule(word_count, 10), (10, 0, 10))

    def test_director_timing_uses_the_same_default_interval(self):
        text = " ".join(["word"] * 25)
        clause = _h3_dialogue_timing_clause([{"words": text}], 10)
        self.assertIn("about 0.25 to 9.18 seconds", clause)
        self.assertIn("silent through 10.00 seconds", clause)

    def test_director_counts_all_speakers_against_the_three_word_ceiling(self):
        shots = [{
            "duration_sec": 10,
            "dialogue_beats": [
                {"spoken_text": " ".join(["first"] * 15)},
                {"spoken_text": " ".join(["second"] * 15)},
            ],
        }]
        self.assertEqual(h3_dialogue_budget_violations(shots), [])
        shots[0]["dialogue_beats"][1]["spoken_text"] += " extra"
        violation = h3_dialogue_budget_violations(shots)[0]
        self.assertEqual((violation["word_count"], violation["word_budget"]), (31, 30))

    def test_studio_admits_three_words_per_second_but_rejects_one_extra(self):
        for count in (28, 29, 30, 31):
            with self.subTest(count=count):
                text = " ".join(f"word{index}" for index in range(count)) + "."
                args = (
                    [{"segment": 1, "dialogue_ids": ["D1"], "source_event_ids": []}],
                    [{"dialogue_id": "D1", "speaker": "Blaine", "text": text}],
                )
                kwargs = dict(segment_durations=[10], source_events=[], expected_dialogue_events={})
                if count == 31:
                    with self.assertRaisesRegex(H3DialogueTimingError, "31 spoken words.*30"):
                        _prepare_render_dialogue_schedule(*args, **kwargs)
                else:
                    _, catalog, _, fragments = _prepare_render_dialogue_schedule(*args, **kwargs)
                    self.assertEqual([item["text"] for item in catalog], [text])
                    self.assertEqual(fragments, [])

    def test_continuation_gate_uses_usable_time_after_overlap(self):
        for count in (40, 41):
            with self.subTest(count=count):
                shots = [{
                    "duration_sec": 13.625,
                    "dialogue_beats": [{"spoken_text": " ".join(["word"] * count)}],
                }]
                violations = h3_dialogue_budget_violations(shots)
                self.assertEqual(bool(violations), count == 41)
                if violations:
                    self.assertEqual(violations[0]["word_budget"], 40)

    def test_final_shot_validation_reports_the_same_ceiling(self):
        for count in (30, 31):
            with self.subTest(count=count):
                shot = ShotPlan.from_dict({
                    'shot_id': 'presenter',
                    'duration_sec': 10,
                    'dialogue_beats': [{'spoken_text': ' '.join(['word'] * count)}],
                })
                result = validate_shot_plan(shot)
                warnings = [item for item in result.warnings if 'Dialogue over-budget' in item]
                self.assertEqual(bool(warnings), count == 31)
                if warnings:
                    self.assertIn('budget 30 words at 3 words/sec', warnings[0])


if __name__ == "__main__":
    unittest.main()
