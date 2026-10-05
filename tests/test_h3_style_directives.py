"""Style instructions keep their global contract without posing as choreography."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import (
    _is_style_only_fragment,
    _h3_stable_camera_context_map,
    extract_h3_source_intent,
    extract_source_events,
    segment_violations,
)


STYLE = "the cinematography style must always remain live-action movie-grade with no gamified look"


def validate(source, action, camera=""):
    events = extract_source_events(source)
    beat = {"beat_id": "B1", "segment": 1,
            "description": events[0]["text"],
            "source_event_ids": [events[0]["event_id"]], "dialogue_ids": []}
    segment = {"segment": 1, "semantic_actions": True,
               "closing_state": "Mira's action is complete.",
               "shots": [{"beat_ids": ["B1"], "start_seconds": 0,
                          "end_seconds": 8, "action": action,
                          "camera": camera, "framing": "Medium shot",
                          "transition": "opening", "sound_effects": ""}]}
    return segment_violations(source, segment, segment_number=1, duration=8,
                              assigned_beats=[beat], dialogue_catalog=[])


class H3StyleDirectiveTests(unittest.TestCase):
    def test_inline_style_keeps_global_contract_and_door_action(self):
        source = f"Mira opens the wooden door, while {STYLE}."
        intent = extract_h3_source_intent(source)
        events = extract_source_events(source)
        self.assertIn(STYLE, intent["style_contract"])
        self.assertIn(STYLE, intent["global_instructions"])
        self.assertIn(STYLE, _h3_stable_camera_context_map({}, intent, events)["visual_continuity"])
        self.assertEqual([event["text"] for event in events], ["Mira opens the wooden door"])
        self.assertEqual(validate(source, "Mira opens the wooden door."), [])

    def test_style_does_not_allow_a_missing_action_to_pass(self):
        source = f"Mira opens the wooden door, while {STYLE}."
        errors = validate(source, "Mira stands beside the closed door.",
                          camera="Show Mira opening the wooden door in a live-action movie-grade view.")
        self.assertTrue(any("omits required source step" in error and "opens" in error for error in errors), errors)

    def test_authored_time_range_keeps_style_global_and_action_locked(self):
        source = (f"[0s-8s] Mira opens the wooden door, while {STYLE}. "
                  "[8s-12s] Mira lowers her hands.")
        intent = extract_h3_source_intent(source)
        events = extract_source_events(source)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["source_start_seconds"], 0)
        self.assertEqual(events[0]["source_end_seconds"], 8)
        self.assertIn("Mira opens the wooden door", events[0]["text"])
        self.assertNotIn("style", events[0]["text"])
        self.assertIn(STYLE, intent["style_contract"])
        self.assertIn(STYLE, intent["global_instructions"])
        self.assertEqual(validate(source, "Mira opens the wooden door."), [])
        self.assertTrue(any("omits required source step" in error for error in
                            validate(source, "Mira lifts a red lantern.")))

    def test_modal_style_wording_is_classified_without_a_fake_stay_action(self):
        for style in (STYLE, "cinematography style must always stay live-action movie-grade—no cheap gamified feels",
                      "the visual style should remain cinematic and gritty"):
            with self.subTest(style=style):
                self.assertTrue(_is_style_only_fragment(style))
                source = f"Mira opens the door. {style}."
                intent = extract_h3_source_intent(source)
                self.assertIn(style, intent["style_contract"])
                self.assertIn(style, intent["global_instructions"])
                self.assertFalse(any("style" in event["text"] for event in extract_source_events(source)))

    def test_narrative_and_unknown_predicates_cannot_be_hidden_in_style(self):
        for tail in ("while Mira opens the door", "while the gear slips", "and Mira discovers a passage"):
            with self.subTest(tail=tail):
                value = f"{STYLE} {tail}"
                self.assertFalse(_is_style_only_fragment(value))
                source = f"Mira stands beside the door, while {value}."
                self.assertTrue(any(tail.split()[-1] in event["text"] for event in extract_source_events(source)))

    def test_comma_delimited_physical_tail_survives_style_span(self):
        source = f"Mira stands beside the door, while {STYLE}, Mira opens the wooden door."
        intent = extract_h3_source_intent(source)
        events = extract_source_events(source)
        self.assertIn(STYLE, intent["style_contract"])
        self.assertTrue(any("Mira opens the wooden door" in event["text"] for event in events), events)

    def test_actual_camera_move_still_has_to_be_present(self):
        source = "Mira opens the wooden door, while the camera makes a slow dolly-in to the latch."
        errors = validate(source, "Mira opens the wooden door.")
        self.assertTrue(any("omits required source step" in error and "dolly" in error for error in errors), errors)
        self.assertEqual(validate(source, "Mira opens the wooden door.",
                                  "The camera makes a slow dolly-in to the latch."), [])

    def test_fighting_style_and_quoted_style_remain_story_content(self):
        self.assertFalse(_is_style_only_fragment("Mira's fighting style changes as she pulls the lever."))
        source = 'Mira reads a card marked "the visual style must remain cinematic", then opens the door.'
        self.assertNotIn("the visual style must remain cinematic", extract_h3_source_intent(source)["style_contract"])
        self.assertTrue(any("reads a card" in event["text"] for event in extract_source_events(source)))


if __name__ == "__main__":
    unittest.main()
