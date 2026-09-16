"""Regressions for imperative guidance leaking into H3 cast and events."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import extract_h3_source_intent, extract_source_events


class H3InstructionCastTests(unittest.TestCase):
    def test_capitalized_creation_directive_is_not_cast(self):
        source = (
            "Mara fights Dev. Write the shot for shot continuously "
            "chained fighting choreography. Mara wins."
        )
        self.assertEqual(
            extract_h3_source_intent(source)["cast_names"],
            ["Mara", "Dev"],
        )
        self.assertFalse(any(
            event["text"].startswith("Write the shot")
            for event in extract_source_events(source)
        ))
        reproduced = (
            "Epic cinematic fight scene between a bald monk and Power Girl played by Sydney Sweeney. "
            "Write the shot for shot continuously chained fighting choreography. Power Girl can fly and has heat vision. "
            "They destroy the surrounding environment from their powerful hits, smashing each other through walls. "
            "Use dynamic camera movements. Power girl wins."
        )
        self.assertEqual(
            extract_h3_source_intent(reproduced)["cast_names"],
            ["Power Girl"],
        )

    def test_sentence_initial_name_that_is_not_a_directive_remains_cast(self):
        for name in ("Will", "Major", "Write", "Direct"):
            with self.subTest(name=name):
                source = f"{name} watches Mara while Dev enters."
                self.assertEqual(
                    extract_h3_source_intent(source)["cast_names"],
                    [name, "Mara", "Dev"],
                )

    def test_creation_word_used_as_explicit_speaker_remains_cast(self):
        for source in ('Write says, "Hold position."', 'Direct: "Move now."'):
            with self.subTest(source=source):
                self.assertIn(
                    source.split(":", 1)[0].split(" says", 1)[0],
                    extract_h3_source_intent(source)["cast_names"],
                )

    def test_continuity_direction_is_guidance_not_actor_or_event(self):
        source = (
            "In a modest architecture office, Mara stands beside Dev while Lin enters. "
            "Keep entrances, exits, listeners, room geography, and each door's open or closed "
            "state coherent across the transition."
        )
        intent = extract_h3_source_intent(source)
        self.assertEqual(intent["cast_names"], ["Mara", "Dev", "Lin"])
        self.assertIn("Keep entrances", intent["global_instructions"])
        self.assertFalse(any("Keep entrances" in item["text"] for item in extract_source_events(source)))

    def test_named_physical_keep_direction_remains_an_event(self):
        for verb in ("Keep", "Maintain"):
            with self.subTest(verb=verb):
                source = f"Mara stands beside Dev. {verb} Mara beside the door while Dev crosses the room."
                self.assertEqual(extract_h3_source_intent(source)["cast_names"], ["Mara", "Dev"])
                self.assertTrue(any("Mara beside the door" in item["text"] for item in extract_source_events(source)))

    def test_standalone_physical_direction_discovers_its_cast(self):
        for verb in ("Keep", "Maintain"):
            with self.subTest(verb=verb):
                source = f"{verb} Mara beside the door while Dev crosses the room."
                self.assertEqual(extract_h3_source_intent(source)["cast_names"], ["Mara", "Dev"])

    def test_keep_with_finite_predicate_is_a_character(self):
        intent = extract_h3_source_intent("Keep watches Mara and Dev enters.")
        self.assertEqual(intent["cast_names"], ["Keep", "Mara", "Dev"])

    def test_imperative_gerund_does_not_invent_keep_character(self):
        intent = extract_h3_source_intent("Keep walking toward Mara while Dev watches.")
        self.assertEqual(intent["cast_names"], ["Mara", "Dev"])

    def test_location_opener_is_not_cast(self):
        intent = extract_h3_source_intent("In Paris, Mara enters and Dev waves.")
        self.assertEqual(intent["cast_names"], ["Mara", "Dev"])

    def test_explicit_character_named_keep_remains_cast(self):
        for source in (
            'Keep: "Wait for me." Mara enters.',
            'Keep says, "Wait for me." Mara enters.',
            'Alice Chen as Captain Keep enters beside Mara.',
        ):
            with self.subTest(source=source):
                cast = extract_h3_source_intent(source)["cast_names"]
                self.assertTrue(any("Keep" in name for name in cast), cast)

    def test_compound_subject_named_keep_is_not_an_imperative(self):
        cast = extract_h3_source_intent("Keep and Mara enter together.")["cast_names"]
        self.assertIn("Keep", cast)
        self.assertIn("Mara", cast)

    def test_production_notes_are_guidance_not_heuristic_cast(self):
        source = (
            "[Camera Direction] Keep entrances and exits coherent across cuts. "
            "[0-4s: Scene] Mara enters and Dev watches."
        )
        intent = extract_h3_source_intent(source)
        self.assertIn("Mara", intent["cast_names"])
        self.assertIn("Dev", intent["cast_names"])
        self.assertNotIn("Camera Direction", intent["cast_names"])
        self.assertNotIn("Keep", intent["cast_names"])
        self.assertIn("Camera Direction", intent["global_instructions"])
        self.assertIn("Keep entrances", intent["global_instructions"])

    def test_explicit_speaker_in_production_notes_remains_authoritative(self):
        source = 'Keep: "Hold position."\n[Camera Direction] Preserve continuity throughout.'
        self.assertIn("Keep", extract_h3_source_intent(source)["cast_names"])


if __name__ == "__main__":
    unittest.main()
