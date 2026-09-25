"""Focused camera-compiler checks for typed endpoint conditions."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import (
    _camera_event_card_schema,
    _camera_phase_beats,
    _expand_camera_event_cards,
    _fallback_segment,
    _h3_stable_camera_context_map,
    _materialize_segment,
    extract_source_events,
)


def _camera_card(action: str) -> dict[str, str]:
    return {
        "action": action,
        "framing": "A clear medium view",
        "camera": "Hold the established axis",
        "transition": "Continue the same shot",
        "sound_effects": "Natural room tone",
    }


class FinalStateCameraContractTests(unittest.TestCase):
    def _phases(self, source: str):
        events = extract_source_events(source)
        beat = {
            "beat_id": "B1",
            "description": "Stage the requested action and visible endpoint.",
            "source_event_ids": [event["event_id"] for event in events],
            "dialogue_ids": [],
        }
        phases = _camera_phase_beats(
            [beat], source_events=events, expected_dialogue_events={},
            preserve_adaptation=True,
        )
        return events, phases

    def test_pure_endpoint_uses_a_final_condition_card(self):
        events, phases = self._phases("End with the workshop door closed.")
        self.assertEqual(events[0]["requirement_kind"], "final_state")
        self.assertTrue(phases[0]["_final_state_only"])

        schema = _camera_event_card_schema(1, phases)
        event_schema = schema["properties"]["event_cards"]["properties"]["event_1"]
        self.assertEqual(event_schema["required"], ["final_condition"])

        expanded = _expand_camera_event_cards(
            {"event_cards": {"event_1": {"final_condition": _camera_card(
                "The workshop door is closed."
            )}}},
            assigned_beats=phases, segment_number=1, duration=8.0,
        )
        action = expanded["shots"][0]["action"]
        self.assertIn("Final visible condition:", action)
        self.assertIn("The workshop door is closed", action)
        self.assertNotIn("Then", action)

    def test_mixed_action_and_endpoint_keep_both_requirements(self):
        source = "Priya opens the workshop door. End with the workshop door closed."
        events, phases = self._phases(source)
        self.assertEqual([event["event_id"] for event in events], ["E1", "E2"])
        self.assertFalse(phases[0]["_final_state_only"])
        self.assertEqual(phases[0]["_final_state_source_event_ids"], ["E2"])

        event_schema = _camera_event_card_schema(1, phases)["properties"][
            "event_cards"
        ]["properties"]["event_1"]
        self.assertEqual(event_schema["required"], ["phases", "final_condition"])
        expanded = _expand_camera_event_cards(
            {"event_cards": {"event_1": {
                "phases": [_camera_card("Priya opens the workshop door.")],
                "final_condition": "The workshop door is closed.",
            }}},
            assigned_beats=phases, segment_number=1, duration=8.0,
        )
        action = expanded["shots"][0]["action"]
        self.assertIn("Priya opens the workshop door", action)
        self.assertIn("Final visible condition: The workshop door is closed", action)
        self.assertNotIn("Then the workshop door", action)

    def test_eventive_end_wording_remains_an_action(self):
        events, phases = self._phases("End with Priya closing the workshop door.")
        self.assertNotEqual(events[0].get("requirement_kind"), "final_state")
        self.assertFalse(phases[0].get("_final_state_only", False))
        event_schema = _camera_event_card_schema(1, phases)["properties"][
            "event_cards"
        ]["properties"]["event_1"]
        self.assertIn("phases", event_schema["required"])
        self.assertNotIn("final_condition", event_schema["required"])

    def test_fallback_materialization_keeps_two_endpoints_out_of_action_order(self):
        source = (
            "Priya ties the banner to the wall hook. "
            "End with the banner hanging securely. "
            "End with the workshop door closed. No dialogue."
        )
        events, phases = self._phases(source)
        self.assertEqual(phases[0]["_final_state_source_event_ids"], ["E2", "E3"])

        draft = _fallback_segment(
            1, duration=8.0, beats=phases,
            opening_state="Priya holds the banner beside the open workshop door.",
            camera_coverage="multi_shot",
        )
        # A proposed summary cannot replace either immutable endpoint.
        draft["closing_state"] = "The banner lies on the floor and the door is open."
        rendered = _materialize_segment(
            draft, beats=phases, dialogue_catalog=[], source_events=events,
        )
        action = rendered["shots"][0]["action"]
        self.assertIn("Priya ties the banner to the wall hook", action)
        self.assertIn("Final visible condition: the banner hanging securely; the workshop door closed", action)
        self.assertNotIn("Then the banner", action)
        self.assertNotIn("Then the workshop door", action)
        self.assertNotIn("lies on the floor", action)

    def test_stable_context_filters_changed_prop_state_and_preserves_stable_facts(self):
        source = (
            "Priya lifts the banner and ties it to the wall hook. "
            "End with the banner hanging securely."
        )
        events = extract_source_events(source)
        raw = {
            "character_appearance": "Priya wears a green coat.",
            "subject_continuity": "Priya’s hands still hold the banner.",
            "setting_continuity": (
                "A workshop with a banner leaning against the far wall. "
                "The courtyard is dark. The door opens inward."
            ),
            "visual_continuity": "Soft daylight and readable framing.",
            "editing_style": "A motivated continuous reframe.",
        }
        original = dict(raw)
        stable = _h3_stable_camera_context_map(
            raw, {"perspective_contract": "", "style_contract": ""}, events,
        )

        self.assertNotIn("leaning against the far wall", stable["setting_continuity"])
        self.assertIn("courtyard is dark", stable["setting_continuity"])
        self.assertIn("door opens inward", stable["setting_continuity"])
        self.assertIn("Priya wears a green coat", stable["subject_continuity"])
        self.assertNotIn("still hold the banner", stable["subject_continuity"])
        self.assertEqual(raw, original)  # Filtered native view does not rewrite provenance.

    def test_native_context_retains_technical_mechanics_that_name_an_action(self):
        mechanics = "Priya carries the banner using both hands; no invisible attachments."
        raw = {"visual_continuity": mechanics, "motion_mechanics": mechanics}
        stable = _h3_stable_camera_context_map(
            raw, {}, extract_source_events("Priya carries the banner inside the workshop."),
        )
        self.assertIn(mechanics, stable["visual_continuity"])


if __name__ == "__main__":
    unittest.main()
