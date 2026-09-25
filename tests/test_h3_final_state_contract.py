"""Explicit ending conditions stay states, while physical actions stay actions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import extract_source_events, segment_violations
from services.h3_camera_fidelity import review_missing_camera_actions


def check_state(source, visual, camera="Wide view", closing="The door is closed"):
    events = extract_source_events(source)
    event = events[-1]
    beat = dict(beat_id="B1", source_event_ids=[event["event_id"]],
                dialogue_ids=[], description=event["text"])
    segment = dict(segment=1, semantic_actions=True, camera_contract="event_cards",
        closing_state=closing, shots=[dict(shot=1, beat_ids=["B1"],
            start_seconds=0, end_seconds=4, action=visual, camera=camera,
            framing="Wide view", sound_effects="", dialogue=[])])
    errors = segment_violations(source, segment, segment_number=1, duration=4,
        assigned_beats=[beat], dialogue_catalog=[])
    return errors, events, beat, segment


class FinalStateContractTests(unittest.TestCase):
    def test_explicit_ending_visible_state_does_not_require_closing_again(self):
        errors, events, _, _ = check_state(
            "Priya closes the workshop door. End with the workshop door closed.",
            "The workshop door is still closed behind Priya.")
        self.assertEqual(events[-1].get("requirement_kind"), "final_state")
        self.assertFalse([e for e in errors if "omits" in e], errors)

    def test_wrong_state_or_owner_or_only_claimed_state_is_not_proof(self):
        for visual in ("The workshop door is open behind Priya.",
                       "Priya looks toward the workshop door.",
                       "A different cupboard door is closed.",
                       "Priya imagines the workshop door closed.",
                       "The workshop door is not closed.",
                       "The workshop door is closed. Then Priya opens it again.",
                       "The workshop door is closed, then Priya opens it again."):
            errors, _, _, _ = check_state(
                "End with the workshop door closed.", visual,
                closing="The workshop door is closed.")
            self.assertTrue(any("omits" in e for e in errors), (visual, errors))

    def test_end_with_an_action_still_requires_the_transition(self):
        errors, events, _, _ = check_state(
            "End with Priya closing the workshop door.",
            "Priya stands beside the workshop door. It is already closed.")
        self.assertNotIn("requirement_kind", events[-1])
        self.assertTrue(any("omits" in e for e in errors), errors)

    def test_review_gets_scoped_state_semantics_and_real_visual_receipts(self):
        errors, events, beat, segment = check_state(
            "End with the workshop door closed.",
            "The workshop entrance remains firmly shut behind Priya.")
        omission = next(e for e in errors if "omits" in e)
        before = deepcopy(segment)
        generate = Mock(return_value=json.dumps({"check_1": {
            "verdict": "preserved", "evidence_span_ids": ["check_1_card_1_action_span_1"]}}))
        receipts = review_missing_camera_actions([omission], segment,
            assigned_beats=[beat], source_events=events, generate=generate)
        self.assertIn(omission, receipts)
        payload = json.loads(generate.call_args.kwargs["prompt"])["check_1"]
        self.assertEqual(payload["requirement_kind"], "final_state")
        self.assertNotIn("action_obligations", payload)
        self.assertEqual(before, segment)

    def test_state_review_cannot_use_closing_claim_as_visual_evidence(self):
        errors, events, beat, segment = check_state(
            "End with the workshop door closed.", "Priya looks toward the workshop entrance.")
        omission = next(e for e in errors if "omits" in e)
        generate = Mock(return_value=json.dumps({"check_1": {
            "verdict": "preserved", "evidence_span_ids": ["check_1_closing_state_span_1"]}}))
        self.assertFalse(review_missing_camera_actions([omission], segment,
            assigned_beats=[beat], source_events=events, generate=generate))


if __name__ == "__main__":
    unittest.main()
