"""Dialogue must not suppress the phases of a separate compound action."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import (
    _camera_event_card_schema, _canonicalize_segment_contract,
    _h3_preview_action_frames, _h3_required_action_frames_covered,
    _deterministic_ledger, _camera_phase_beats,
    extract_h3_source_intent, extract_locked_dialogue, extract_source_events,
    segment_violations,
)


def card(action):
    return dict(action=action, camera="Track the same continuous movement.",
                framing="Medium wide", transition="continue", sound_effects="")


class MixedWindowPhaseTests(unittest.TestCase):
    def test_connective_staging_does_not_copy_the_later_payoff_as_an_assignment(self):
        prompt = "Each evening Riva leaves flowers under the clock. On the last night Sol recognizes the note and places a coin beside the flowers. No dialogue."
        events = extract_source_events(prompt)
        ledger = _deterministic_ledger(prompt, segment_count=4, segment_durations=[14.4] * 4,
            locked_dialogue=[], camera_coverage="multi_shot", reference_context="")
        connective = [beat for beat in ledger["beats"] if not beat["source_event_ids"]]
        self.assertTrue(connective)
        for beat in connective:
            self.assertNotIn(events[-1]["text"], beat["description"])
            phase = _camera_phase_beats([beat], source_events=events, expected_dialogue_events={},
                full_schedule_beats=ledger["beats"], preserve_adaptation=True)[0]
            self.assertEqual(phase["_transition_before_source_event_ids"], [events[-1]["event_id"]])
        self.assertEqual([eid for beat in ledger["beats"] for eid in beat["source_event_ids"]],
                         [event["event_id"] for event in events])

    def test_shared_prop_words_do_not_replace_a_required_placement(self):
        source = "They set the completed frame on the worktable."
        required = _h3_preview_action_frames(source, None)
        unrelated = _h3_preview_action_frames(
            "They insert the print into the frame and hold it beside the worktable.", None,
        )
        self.assertFalse(_h3_required_action_frames_covered(required, unrelated, source_text=source))
        actual = _h3_preview_action_frames("They put the frame on the worktable.", None)
        self.assertTrue(_h3_required_action_frames_covered(required, actual, source_text=source))

    def test_unknown_physical_paraphrase_needs_review_instead_of_noun_overlap(self):
        source = "They set the completed frame on the worktable."
        required = _h3_preview_action_frames(source, None)
        unknown = _h3_preview_action_frames("They guide the frame toward the worktable.", None)
        self.assertFalse(_h3_required_action_frames_covered(required, unknown, source_text=source))

    def test_compound_action_keeps_phases_and_exact_speech_ownership(self):
        prompt = 'Ivo says, "Please bring the map." Mara opens the cabinet, retrieves the map, then closes the cabinet.'
        events = extract_source_events(prompt)
        catalog = extract_locked_dialogue(prompt)
        speech = dict(beat_id="B1", source_event_ids=[events[0]["event_id"]],
                      dialogue_ids=[catalog[0]["dialogue_id"]], description="Ivo speaks.")
        action = dict(beat_id="B2", source_event_ids=[e["event_id"] for e in events[1:]],
                      dialogue_ids=[], description="Mara opens the cabinet, retrieves the map, then closes the cabinet.")
        beats = [speech, action]
        schema = _camera_event_card_schema(1, beats)
        phases = schema["properties"]["event_cards"]["properties"]["event_2"]["properties"]["phases"]
        self.assertGreaterEqual(phases["maxItems"], 3)
        draft = dict(segment=1, title="The map", coverage="Continuous", pacing="natural",
                     closing_state="Mara holds the map beside the closed cabinet.", event_cards={
            "event_1": {catalog[0]["dialogue_id"]: {"lead_in": None, "performance": card("Ivo speaks while Mara listens.")}, "follow_through": None},
            "event_2": {"phases": [card("Mara opens the cabinet."), card("Mara retrieves the map."), card("Mara closes the cabinet.")]},
        })
        result = _canonicalize_segment_contract(
            draft, segment_number=1, duration=14.4, assigned_beats=beats,
            dialogue_catalog=catalog, opening_state="Mara and Ivo stand beside the closed cabinet.",
            source_intent=extract_h3_source_intent(prompt), source_events=events,
        )
        self.assertNotIn("event_assignment_error", result)
        self.assertEqual(len(result["shots"]), 4)
        speech_rows = [d for shot in result["shots"] for d in shot.get("dialogue", [])]
        self.assertEqual([d["dialogue_id"] for d in speech_rows], [catalog[0]["dialogue_id"]])
        self.assertEqual([shot["action"] for shot in result["shots"][1:]],
                         ["Mara opens the cabinet.", "Mara retrieves the map.", "Mara closes the cabinet."])
        self.assertEqual(segment_violations(prompt, result, segment_number=1, duration=14.4,
                         assigned_beats=beats, dialogue_catalog=catalog), [])


if __name__ == "__main__":
    unittest.main()
