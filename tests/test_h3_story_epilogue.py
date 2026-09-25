"""An untimed source action may finish before a compatible final aftermath."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import extract_source_events, ledger_violations


class StoryEpilogueTests(unittest.TestCase):
    source = "Nora enters the empty gallery. Nora sets one blue vase on the table. No dialogue."

    def ledger(self):
        events = extract_source_events(self.source)
        self.assertEqual(len(events), 2)
        return {
            "required_final_outcome": "The single blue vase remains on the table in the empty gallery.",
            "generated_dialogue": [],
            "beats": [
                {"beat_id": "B1", "segment": 1, "description": events[0]["text"],
                 "source_event_ids": [events[0]["event_id"]], "dialogue_ids": []},
                {"beat_id": "B2", "segment": 2, "description": "Nora approaches the table with the vase still in her hands.",
                 "source_event_ids": [], "dialogue_ids": []},
                {"beat_id": "B3", "segment": 3, "description": events[1]["text"],
                 "source_event_ids": [events[1]["event_id"]], "dialogue_ids": []},
                {"beat_id": "B4", "segment": 4, "description": "Nora steps out of view; the camera holds on the vase already resting on the table.",
                 "source_event_ids": [], "dialogue_ids": []},
            ],
        }

    def check(self, value):
        return ledger_violations(self.source, value, segment_count=4, locked_dialogue=[],
                                 expect_dialogue=False, segment_durations=[8., 8., 8., 8.])

    def test_untimed_action_stays_with_its_staging_before_final_aftermath(self):
        value = self.ledger()
        original = deepcopy(value)
        self.assertEqual(self.check(value), [])
        self.assertEqual(value, original)

    def test_missing_or_repeated_action_is_still_rejected(self):
        missing = self.ledger()
        missing["beats"][2]["source_event_ids"] = []
        self.assertTrue(any("missing" in item for item in self.check(missing)))
        repeated = self.ledger()
        repeated["beats"][3]["source_event_ids"] = ["E2"]
        self.assertTrue(any("repeated" in item for item in self.check(repeated)))

    def test_order_and_final_window_coverage_are_still_required(self):
        reversed_events = self.ledger()
        reversed_events["beats"][0]["source_event_ids"] = ["E2"]
        reversed_events["beats"][2]["source_event_ids"] = ["E1"]
        self.assertTrue(any("source event order" in item for item in self.check(reversed_events)))
        no_ending = self.ledger()
        no_ending["beats"] = no_ending["beats"][:3]
        self.assertTrue(any("final segment" in item for item in self.check(no_ending)))

    def test_exact_dialogue_coverage_remains_active(self):
        from services.h3_story_ledger import extract_locked_dialogue
        source = 'Nora enters the empty gallery. Nora says, "Set it here." Nora sets one blue vase on the table.'
        catalog = extract_locked_dialogue(source)
        self.assertTrue(catalog)
        # The source-only silent ledger cannot erase the supplied speech simply
        # because its last physical action is now allowed before the epilogue.
        violations = ledger_violations(source, self.ledger(), segment_count=4,
            locked_dialogue=catalog, expect_dialogue=True, segment_durations=[8.] * 4)
        self.assertTrue(any('dialogue' in item.lower() for item in violations))

    def test_final_action_cannot_move_into_an_unusable_residual_tail(self):
        value = self.ledger()
        value['beats'][2]['source_event_ids'] = []
        value['beats'][3]['source_event_ids'] = ['E2']
        violations = ledger_violations(self.source, value, segment_count=4,
            locked_dialogue=[], expect_dialogue=False,
            segment_durations=[8., 8., 8., 0.5])
        self.assertTrue(any('last usable segment' in item for item in violations))

    def test_explicitly_timed_source_keeps_its_existing_schedule_contract(self):
        self.source = (
            "[0-4s] Nora enters the empty gallery. "
            "[4-8s] Nora sets one blue vase on the table. No dialogue."
        )
        events = extract_source_events(self.source)
        self.assertTrue(any("source_start_seconds" in event for event in events))
        self.assertTrue(any("last usable segment" in item for item in self.check(self.ledger())))


if __name__ == "__main__":
    unittest.main()
