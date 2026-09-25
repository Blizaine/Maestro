"""The camera validator must use delivered speech to resolve source line anchors."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import (
    extract_locked_dialogue, extract_source_events, segment_violations,
)


class DialogueAnchorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.source = (
            'Nora says, "I found the map." '
            'Nora places the map on the desk after her first line.'
        )
        self.catalog = extract_locked_dialogue(self.source)
        event = next(item for item in extract_source_events(self.source)
                     if "places the map" in item["text"])
        self.beats = [{
            "beat_id": "B2", "source_event_ids": [event["event_id"]],
            "description": event["text"], "dialogue_ids": [],
        }]
        self.segment = {
            "segment": 2, "semantic_actions": True,
            "opening_state": "Nora holds the map beside the desk.",
            "closing_state": "The map rests on the desk.",
            "shots": [{
                "start_seconds": 0.0, "end_seconds": 8.0,
                "beat_ids": ["B2"], "dialogue": [],
                "action": "Nora places the map on the desk.",
                "camera": "An uninterrupted medium view.",
            }],
        }
        self.accepted = [{
            "index": 1, "shots": [{
                "start_seconds": 0.0, "end_seconds": 8.0,
                "dialogue": [dict(self.catalog[0])],
                "action": "Nora holds the map while speaking.",
            }],
        }]

    def _violations(self, accepted, segment=None):
        return segment_violations(
            self.source, segment or self.segment, segment_number=2, duration=8.0,
            assigned_beats=self.beats, dialogue_catalog=self.catalog,
            accepted_segments=accepted,
        )

    def test_prior_exact_speech_resolves_anchor_without_repeating_line_on_camera(self):
        self.assertEqual(self._violations(self.accepted), [])

    def test_claiming_after_first_line_cannot_replace_missing_speech(self):
        segment = deepcopy(self.segment)
        segment["shots"][0]["action"] += " This happens after her first line."
        errors = self._violations([], segment)
        self.assertTrue(any("dialogue timing" in error for error in errors), errors)

    def test_wrong_speaker_does_not_resolve_anchor(self):
        accepted = deepcopy(self.accepted)
        accepted[0]["shots"][0]["dialogue"][0]["speaker"] = "Len"
        errors = self._violations(accepted)
        self.assertTrue(any("dialogue timing" in error for error in errors), errors)

    def test_missing_physical_action_still_fails_after_exact_speech(self):
        segment = deepcopy(self.segment)
        segment["shots"][0]["action"] = "Nora holds the map beside the desk."
        errors = self._violations(self.accepted, segment)
        self.assertTrue(any("omits required source step" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
