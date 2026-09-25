"""One participant's movement cannot satisfy an explicit two-person action."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_camera_fidelity import review_missing_camera_actions


class CollectiveCameraCoverageTests(unittest.TestCase):
    def review_claimed_retreat(self, action):
        requirement = "both step back as the seated audience remains in the background"
        error = "B1 shot action omits required source step: " + requirement
        generate = Mock(return_value=json.dumps({"check_1": {"action_decisions": {
            f"action_{i}": {"verdict": "preserved", "evidence_span_ids": ["check_1_card_1_action_span_1"]}
            for i in (1, 2)
        }}}))
        return review_missing_camera_actions([error], {
            "camera_contract": "event_cards", "shots": [{"beat_ids": ["B1"], "action": action}],
        }, assigned_beats=[{"beat_id": "B1", "source_event_ids": ["E2"]}],
            source_events=[
                {"event_id": "E1", "text": "Two different adults, Tavi and Lio, prepare a wooden stand."},
                {"event_id": "E2", "text": requirement},
            ], generate=generate)

    def test_false_semantic_receipts_cannot_turn_hands_into_retreat(self):
        for action in (
            "He lets his hands settle back to a relaxed, ready position.",
            "Tavi steps back while Lio stays beside the program.",
            "Both actors are visible as Tavi steps back while Lio remains beside the stand.",
            "Tavi's hands withdraw from the stand beside Lio.",
            "Both plan to step back from the stand.",
            "Both do not step back from the stand.",
            "Both do not slowly step back from the stand.",
            "Not both actors step back from the stand.",
            "Tavi and Lio fail to step back from the stand.",
            "Tavi and Lio refuse to step back from the stand.",
            "Tavi and Lio are unable to step back from the stand.",
            "Tavi and Lio can't step back from the stand.",
            "Tavi and Lio try to step back from the stand.",
            "Tavi and Lio attempt to step back from the stand.",
            "The camera moves away from both people at the stand.",
            "Both people remain still while the camera moves backward around them.",
            "They do not step back while the camera moves backward around them.",
        ):
            with self.subTest(action=action):
                self.assertFalse(self.review_claimed_retreat(action))

    def test_collective_locomotion_paraphrases_keep_valid_receipts(self):
        for action in (
            "Tavi and Lio step back from the stand together.",
            "Both withdraw from the stand, making room for the audience.",
            "They take a half-step away from the stand.",
            "Tavi retreats from the stand and Lio walks backward toward him.",
            "They do not touch the program, but both step back from the stand.",
            "Both lower their hands and step back together.",
            "They lower their hands and step back together.",
            "Both do not touch the program, but step back together.",
        ):
            with self.subTest(action=action):
                self.assertTrue(self.review_claimed_retreat(action))

    def test_both_requires_two_independent_receipts(self):
        requirement = "both step back as the seated audience remains in the background"
        error = "B1 shot action omits required source step: " + requirement
        events = [
            {"event_id": "E1", "text": "Two different adults, Tavi and Lio, prepare a wooden stand."},
            {"event_id": "E2", "text": requirement},
        ]
        segment = {"camera_contract": "event_cards", "shots": [
            {"beat_ids": ["B1"], "action": "Tavi takes a half-step away from the stand.",
             "framing": "Wide shot includes Lio and the audience.", "camera": "Hold still."}
        ]}
        response = {"check_1": {"action_decisions": {
            "action_1": {"verdict": "preserved", "evidence_span_ids": ["check_1_card_1_action_span_1"]},
            "action_2": {"verdict": "missing", "evidence_span_ids": []},
        }}}
        generate = Mock(return_value=json.dumps(response))
        receipts = review_missing_camera_actions([error], segment,
            assigned_beats=[{"beat_id": "B1", "source_event_ids": ["E2"]}],
            source_events=events, generate=generate)
        self.assertFalse(receipts)
        checks = json.loads(generate.call_args.kwargs["prompt"])["check_1"]
        self.assertEqual({tuple(item["action_focus"]["actor_markers"])
                          for item in checks["action_obligations"]}, {("Tavi",), ("Lio",)})
        # An old broad preserved verdict cannot bypass the per-person proof.
        generate.return_value = json.dumps({"check_1": {
            "verdict": "preserved", "evidence_span_ids": ["check_1_card_1_action_span_1"]}})
        self.assertFalse(review_missing_camera_actions([error], segment,
            assigned_beats=[{"beat_id": "B1", "source_event_ids": ["E2"]}],
            source_events=events, generate=generate))


if __name__ == "__main__":
    unittest.main()
