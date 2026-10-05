"""Authored lens moves use camera evidence without clearing human actions."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import extract_source_events, segment_violations


def validate(source, action, camera=""):
    # Imported clocks require two contiguous ranges and explicit role profiles.
    source = ("Character A: adult in a blue coat. Character B: adult in a gray coat. "
              + source + " [8s-12s] Nora lowers her hands.")
    source = source.replace("Nora", "Character A").replace("Eli", "Character B")
    action = action.replace("Nora", "Character A").replace("Eli", "Character B")
    camera = camera.replace("Nora", "Character A").replace("Eli", "Character B")
    events = extract_source_events(source)
    assert events[0]["source_start_seconds"] == 0 and events[0]["source_end_seconds"] == 8
    beat = {"beat_id": "B1", "segment": 1,
            "description": events[0]["text"],
            "source_event_ids": [events[0]["event_id"]], "dialogue_ids": []}
    segment = {"segment": 1, "semantic_actions": True,
               "closing_state": "Character A remains beside the door.",
               "shots": [{"beat_ids": ["B1"], "start_seconds": 0,
                          "end_seconds": 8, "action": action,
                          "camera": camera, "framing": "Medium shot",
                          "transition": "opening", "sound_effects": ""}]}
    return segment_violations(source, segment, segment_number=1, duration=8,
                              assigned_beats=[beat], dialogue_catalog=[])


class OpticalSourceContractTests(unittest.TestCase):
    def test_lens_requirement_can_be_satisfied_in_camera_field(self):
        source = "[0s-8s] Nora opens the blue door. The lens pulls wide to follow Eli."
        self.assertEqual(validate(source, "Nora opens the blue door.",
                                  "The lens pulls wide to follow Eli."), [])
        errors = validate(source, "Nora opens the blue door.")
        self.assertTrue(any("omits required source step" in error and "lens" in error.casefold()
                            for error in errors), errors)

    def test_inherited_camera_clauses_keep_optical_ownership(self):
        optics = "The lens glides past Nora, then cuts to Eli, finally pulls to an ultra-wide center frame."
        source = f"[0s-8s] Nora opens the blue door. {optics}"
        self.assertEqual(validate(source, "Nora opens the blue door.", optics), [])
        errors = validate(source, "Nora opens the blue door.", "The lens glides past Nora.")
        self.assertTrue(any("ultra-wide center frame" in error for error in errors), errors)

    def test_camera_fields_cannot_satisfy_a_missing_physical_action(self):
        source = "[0s-8s] Nora opens the blue door. The lens pulls wide to follow Eli."
        errors = validate(source, "Nora lifts a red lantern.",
                          "Nora opens the blue door; the lens pulls wide to follow Eli.")
        self.assertTrue(any("omits required source step" in error and "opens" in error
                            for error in errors), errors)

    def test_mixed_camera_and_human_clause_still_requires_physical_action(self):
        source = "[0s-8s] The camera follows Nora as she opens the blue door."
        errors = validate(source, "Nora lifts a red lantern.",
                          "The camera follows Nora as she opens the blue door.")
        self.assertTrue(any("omits required source step" in error for error in errors), errors)
        self.assertEqual(validate(source, "Nora opens the blue door.", "The camera follows Nora."), [])


if __name__ == "__main__":
    unittest.main()
