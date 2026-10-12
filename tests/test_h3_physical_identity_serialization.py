"""Typed blocking must not leak into clip-long H3 subject identity text."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.h3_dialogue import compile_h3_clip_plans  # noqa: E402
from services.director.physical_continuity import identity_only_description  # noqa: E402


def _physical_state():
    fact = {
        "subject_id": "kitten",
        "predicate": "prop",
        "value": "sword sheathed at left hip",
    }
    return {
        "opening": [dict(fact)],
        "transitions": [],
        "closing": [dict(fact)],
    }


def _plan(*, dialogue=False, music=False):
    prompt = (
        "The Kitten looks toward the Dragon."
        if not dialogue else
        "The Kitten looks toward the Dragon and says: <d>[English] Stay close.</d>"
    )
    plan = {
        "video_prompt": prompt,
        "_director_h3_source_prompt": prompt,
        "_director_h3_model_family": "ref2va",
        "_director_h3_prompt_mode": "ref2va",
        "_director_duration_sec": 6,
        "_director_subjects_on_screen": [{
            "character_id": "kitten",
            "speaker_name": "Kitten",
            "visual_description": (
                "A small cream-colored kitten with bright green eyes, "
                "holding a long silver-bladed sword vertically in its front paws."
            ),
            "wardrobe": "silver armor and a pink bow",
        }, {
            "character_id": "dragon",
            "speaker_name": "Dragon",
            "visual_description": "A broad-winged blue dragon with gold eyes.",
        }],
        "_director_physical_state": _physical_state(),
        "_director_h3_reference_manifest": [{
            "type": "image",
            "path": "kitten-reference.png",
            "image_intent": "identity",
            "role": (
                "Kitten: a small warrior kitten wearing a pink bow, "
                "holding a sword"
            ),
        }],
    }
    if dialogue:
        plan["_director_dialogue_beats"] = [{
            "speaker_id": "kitten",
            "spoken_text": "Stay close.",
        }]
    if music:
        plan["_director_audio_plan"] = {
            "mode": "music_driven",
            "vocal_activity": "instrumental",
            "ambience": "Wind and soft wingbeats",
        }
    return plan


class H3PhysicalIdentitySerializationTests(unittest.TestCase):
    def test_bare_noun_pose_clause_keeps_preceding_appearance_and_costume(self):
        cleaned = identity_only_description(
            "A fluffy gray kitten wearing a pink bow, holding a long sword vertically."
        )
        self.assertIn("fluffy gray kitten", cleaned)
        self.assertIn("wearing a pink bow", cleaned)
        self.assertNotIn("holding a long sword", cleaned)

    def test_stable_compound_traits_are_not_mistaken_for_temporary_pose(self):
        cleaned = identity_only_description(
            "A flying squirrel with forward-facing eyes and forward facing eyes, "
            "wearing riding boots and a standing collar."
        )
        self.assertIn("flying squirrel", cleaned)
        self.assertIn("forward-facing eyes", cleaned)
        self.assertIn("forward facing eyes", cleaned)
        self.assertIn("riding boots", cleaned)
        self.assertIn("standing collar", cleaned)

    def test_actual_facing_and_held_prop_actions_are_removed(self):
        cleaned = identity_only_description(
            "A fluffy gray kitten facing the Dragon while holding a long sword."
        )
        self.assertIn("fluffy gray kitten", cleaned)
        self.assertNotIn("facing the Dragon", cleaned)
        self.assertNotIn("holding a long sword", cleaned)

    def test_shared_description_filter_keeps_identity_and_drops_pose_and_stowed_prop(self):
        description = (
            "A small cream-colored kitten with bright green eyes holding a sword. "
            "It is with a long silver-bladed sword sheathed at its left hip."
        )
        cleaned = identity_only_description(description)
        self.assertIn("cream-colored kitten", cleaned)
        self.assertIn("bright green eyes", cleaned)
        self.assertNotIn("holding a sword", cleaned)
        self.assertNotIn("sheathed", cleaned)

    def test_typed_state_sanitizes_ref2va_subject_definitions_on_dialogue_clip(self):
        plan = _plan(dialogue=True)
        original_description = plan["_director_subjects_on_screen"][0]["visual_description"]
        compile_h3_clip_plans([plan])

        definitions = re.search(
            r"(?ms)^subject_definitions:\s*(.*?)\n\nsummary:",
            plan["video_prompt"],
        )
        self.assertIsNotNone(definitions)
        serialized_subjects = definitions.group(1)
        self.assertIn("small cream-colored kitten", serialized_subjects.casefold())
        self.assertNotIn("holding a long silver-bladed sword", serialized_subjects.casefold())
        self.assertIn("silver armor and a pink bow", serialized_subjects)
        self.assertIn("<d>[English] Stay close.</d>", plan["video_prompt"])
        self.assertEqual(
            plan["_director_subjects_on_screen"][0]["visual_description"],
            original_description,
        )

    def test_typed_state_sanitizes_ref2va_subject_definitions_on_music_clip(self):
        plan = _plan(music=True)
        compile_h3_clip_plans([plan])

        definitions = re.search(
            r"(?ms)^subject_definitions:\s*(.*?)\n\nsummary:",
            plan["video_prompt"],
        )
        self.assertIsNotNone(definitions)
        serialized_subjects = definitions.group(1)
        self.assertIn("small cream-colored kitten", serialized_subjects.casefold())
        self.assertNotIn("holding a long silver-bladed sword", serialized_subjects.casefold())
        self.assertIn("silver armor and a pink bow", serialized_subjects)


if __name__ == "__main__":
    unittest.main()
