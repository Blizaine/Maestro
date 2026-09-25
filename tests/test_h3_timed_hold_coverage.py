"""Global silent endings need visual evidence from every contributing window."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_camera_fidelity import review_missing_camera_actions, clear_confirmed_coverage_errors


class TimedHoldCoverageTests(unittest.TestCase):
    def setUp(self):
        self.text = "The 55.25-second video holds the final composition in silence for the remaining 17.923 seconds"
        self.error = "B4 shot action omits required source step: " + self.text
        self.contract = {"source_event_id": "E4", "source_event_text": self.text,
            "global_start_seconds": 37.326961, "global_end_seconds": 55.25,
            "total_hold_seconds": 17.923039, "windows": [
                {"window": 3, "local_start_seconds": 9.326961, "local_end_seconds": 13.625, "hold_seconds": 4.298039},
                {"window": 4, "local_start_seconds": 0.0, "local_end_seconds": 13.625, "hold_seconds": 13.625},
            ]}
        self.previous = [{"segment": 3, "shots": [{"start_seconds": 0., "end_seconds": 13.625,
            "action": "The performer stops at the soundtrack's final note and holds the final composition still thereafter."}]}]
        self.segment = {"segment": 4, "camera_contract": "event_cards", "shots": [
            {"beat_ids": ["B4"], "start_seconds": 0., "end_seconds": 13.625,
             "action": "The performer and composition remain still throughout this silent window."}]}
        self.response = {"check_1": {"window_decisions": {
            f"window_{i}": {"verdict": "preserved", "evidence_span_ids": [f"check_1_window_{i}_card_1_action_span_1"]}
            for i in (3, 4)
        }}}

    def review(self):
        generate = Mock(return_value=json.dumps(self.response))
        result = review_missing_camera_actions([self.error], self.segment,
            assigned_beats=[{"beat_id": "B4", "source_event_ids": ["E4"]}],
            source_events=[{"event_id": "E4", "text": self.text}], generate=generate,
            timed_hold_contracts=[self.contract], accepted_segments=self.previous)
        return result, generate

    def test_global_duration_is_split_but_source_is_unchanged(self):
        before = deepcopy(self.segment)
        receipts, generate = self.review()
        self.assertIn(self.error, receipts)
        packet = json.loads(generate.call_args.kwargs["prompt"])["check_1"]
        self.assertEqual(packet["source_requirement"], self.text)
        self.assertEqual({s["window"] for s in packet["visual_spans"]}, {3, 4})
        self.assertEqual(self.segment, before)
        self.assertEqual(clear_confirmed_coverage_errors([self.error], self.segment, receipts), [])

    def test_one_window_cannot_prove_another_windows_hold(self):
        self.response["check_1"]["window_decisions"]["window_3"]["evidence_span_ids"] = ["check_1_window_4_card_1_action_span_1"]
        self.assertFalse(self.review()[0])

    def test_missing_or_contradicted_contribution_cannot_pass(self):
        for verdict in ("missing", "contradicted"):
            self.response["check_1"]["window_decisions"]["window_3"] = {"verdict": verdict, "evidence_span_ids": []}
            self.assertFalse(self.review()[0])

    def test_clock_alone_cannot_supply_earlier_visual_evidence(self):
        self.previous = []
        receipts, generate = self.review()
        self.assertFalse(receipts)
        generate.assert_not_called()

    def test_last_instant_or_timeline_gap_cannot_cover_full_contribution(self):
        self.previous[0]["shots"][0]["start_seconds"] = 13.5
        self.assertFalse(self.review()[0])

    def test_changed_timing_invalidates_prior_approval(self):
        receipts, _ = self.review()
        self.segment["shots"][0]["start_seconds"] = 3.
        self.assertEqual(clear_confirmed_coverage_errors([self.error], self.segment, receipts), [self.error])

    def test_plain_single_check_cannot_approve_multiwindow_hold(self):
        self.response["check_1"] = {"verdict": "preserved", "evidence_span_ids": ["check_1_window_4_card_1_action_span_1"]}
        self.assertFalse(self.review()[0])


if __name__ == "__main__":
    unittest.main()
