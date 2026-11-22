"""A supplied character reference is context, not a performed source action."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.contract_repair import apply_compact_source_physical_patch
from services.director.source_contract import (
    build_source_contract,
    format_source_contract,
    source_contract_issues,
)


GANGSTER_MIRROR_PROMPT = (
    "use the gangster character in the uploaded reference images. "
    "He has a low gruff voice and talks slowly. "
    "He is looking at himself in a full body mirror in his bedroom "
    "telling himself how good of a gangster he is."
)


class DirectorIssue197ReferenceSelectionTests(unittest.TestCase):
    def test_h3_director_shot_keeps_cast_parts_voice_and_shared_media(self):
        from services.director.reference_inputs import select_director_shot_references

        def visual(name, path):
            return {"type": "image", "path": path, "character_name": name,
                    "library_character_id": name.lower()}

        def voice(name):
            return {"type": "audio", "path": name + ".wav", "character_name": name,
                    "library_character_id": name.lower(), "audio_intent": "voice"}

        alice, body, bob = visual("Alice", "alice.png"), visual("Alice", "alice2.png"), visual("Bob", "bob.png")
        room = {"type": "image", "path": "room.png", "image_intent": "scene", "role": "room"}
        beat = {"type": "audio", "path": "beat.wav", "audio_intent": "drive"}
        garden = {"type": "image", "path": "garden.png", "image_intent": "scene", "role": "garden"}
        references = [alice, voice("Alice"), bob, voice("Bob"), room, body, beat, garden]
        plan = {
            "_director_subjects_on_screen": [{"character_id": "char_1", "speaker_name": "Bob"}],
            "_director_environment": "the room under warm light",
        }
        params = {"characters": [{"name": "Alice"}, {"name": "Bob"}]}
        self.assertEqual(select_director_shot_references(references, plan, params),
                         [bob, voice("Bob"), room, beat])
        plan["shot_reference_cast"] = ["character:alice"]
        self.assertEqual(select_director_shot_references(references, plan, params),
                         [alice, voice("Alice"), room, body, beat])
        plan["shot_reference_cast"] = []
        self.assertEqual(select_director_shot_references(references, plan, params), [room, beat])

    def test_reference_instruction_with_a_real_action_is_not_discarded(self):
        contract = build_source_contract(
            "Use the robot character in the uploaded reference images to open the door."
        )
        self.assertTrue(any("open the door" in event["text"] for event in contract["events"]))

    def test_reference_selection_is_not_an_action_but_mirror_action_is_preserved(self):
        source_contract = build_source_contract(GANGSTER_MIRROR_PROMPT)

        self.assertEqual(
            source_contract["events"],
            [{
                "event_id": "E1",
                "text": (
                    "He is looking at himself in a full body mirror in his bedroom "
                    "telling himself how good of a gangster he is"
                ),
            }],
        )
        formatted = format_source_contract(source_contract)
        self.assertIn("E1: He is looking at himself", formatted)
        self.assertNotIn("uploaded reference images", formatted)
        # Filtering applies only to the event ledger; callers retain the brief.
        self.assertIn("uploaded reference images", GANGSTER_MIRROR_PROMPT)

    def test_duplicate_action_allocation_and_unsupported_support_change_stay_rejected(self):
        source_contract = build_source_contract(GANGSTER_MIRROR_PROMPT)
        duplicate_row = {"source_event_ids": ["E1", "E1"]}
        issues = source_contract_issues([duplicate_row], source_contract)
        self.assertTrue(any(
            "Source actions must be allocated exactly once in source order" in issue
            for issue in issues
        ), issues)

        rows = [{
            "subjects_on_screen": [{"character_id": "char_0"}],
            "source_event_ids": [],
            "_director_source_contract": source_contract,
        }]
        proposal = {
            "event_shots": {"E1": 1},
            "initial_facts": {"char_0": {}},
            "event_transitions": {
                "E1": [{
                    "subject_id": "char_0",
                    "predicate": "support",
                    "to_value": "the mirror",
                    "action": "He looks at himself in the mirror",
                }],
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            "targets unknown physical fact key char_0 support",
        ):
            apply_compact_source_physical_patch(rows, proposal)


if __name__ == "__main__":
    unittest.main()
