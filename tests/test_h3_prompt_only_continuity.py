"""Model-free regressions for native H3 opening-state continuity."""

from __future__ import annotations

import copy
import os
import sys
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.planners.short_film import (  # noqa: E402
    _prepare_h3_prompt_only_continuity,
    _h3_rebuilt_visual_prompt,
    _h3_identity_only_description,
)


class H3PromptOnlyContinuityTests(unittest.TestCase):
    def test_typed_state_keeps_identity_without_reference_pose_or_held_prop(self):
        raw = {
            "scene_goal": "Kitten rides through the valley",
            "subjects_on_screen": [{
                "speaker_name": "Kitten", "character_id": "kitten",
                "visual_description": (
                    "Tiny cream-colored kitten, wearing silver armor and a pink bow, "
                    "and holding a sword vertically. It is standing beside the Dragon."
                ),
                "wardrobe": "silver armor and pink bow",
            }],
            "physical_state": {
                "opening": [{"subject_id": "kitten", "predicate": "support", "value": "astride dragon"}],
                "transitions": [],
                "closing": [{"subject_id": "kitten", "predicate": "support", "value": "astride dragon"}],
            },
            "_director_carried_source_details": ["with sword sheathed at its left hip"],
            "action_beats": ["The Kitten leans with the Dragon through the bank."],
            "ending_beat": "The castle gate opens.",
        }
        prompt = _h3_rebuilt_visual_prompt(raw)
        self.assertIn("cream-colored kitten", prompt)
        self.assertIn("silver armor", prompt)
        self.assertIn("pink bow", prompt)
        self.assertNotIn("holding a sword vertically", prompt)
        self.assertNotIn("standing beside the Dragon", prompt)
        self.assertIn("sword sheathed at its left hip", prompt)
        self.assertIn("leans with the Dragon", prompt)
        self.assertIn("castle gate opens", prompt)

    def test_typed_opening_does_not_reintroduce_stale_model_blocking(self):
        state = {
            "opening": [{"subject_id": "kitten", "predicate": "support", "value": "astride dragon"}],
            "transitions": [],
            "closing": [{"subject_id": "kitten", "predicate": "support", "value": "astride dragon"}],
        }
        row = {
            "subjects_on_screen": [{"speaker_name": "Kitten", "character_id": "kitten"}],
            "physical_state": state, "spatial_setup": "Kitten floats in front of the snout",
            "environment": "misty valley", "source_event_ids": ["E1"],
            "_director_source_contract": {"events": [{"event_id": "E1", "text": "Kitten rides"}]},
            "causal_handoff": "Kitten already lands at the gate",
            "persistent_story_state": "Kitten stands on the ground",
            "action_beats": ["The Kitten rides the Dragon through the valley"],
        }
        row["video_prompt"] = _h3_rebuilt_visual_prompt(row)
        _prepare_h3_prompt_only_continuity([row])
        prompt = row["video_prompt"]
        self.assertIn("astride dragon", prompt)
        self.assertIn("misty valley", prompt)
        for stale in ("floats in front", "already lands", "stands on the ground"):
            self.assertNotIn(stale, prompt)

    def test_identity_cleanup_preserves_design_terms_and_untyped_description(self):
        description = "A flying squirrel with forward-facing eyes and silver fur"
        self.assertEqual(_h3_identity_only_description(description), description)
        self.assertIn("holding a lantern", _h3_rebuilt_visual_prompt({
            "subjects_on_screen": [{"speaker_name": "Mara", "visual_description": "Mara, holding a lantern"}],
        }))

    def test_opening_uses_static_spatial_setup_not_later_subject_pose(self):
        next_opening = (
            "The Kitten is on the ground beside the Dragon's foreleg, "
            "facing the saddle strap."
        )
        subjects = [{
            "speaker_name": "Tiny Kitten",
            "character_id": "char_0",
            "visual_description": (
                "Tiny Kitten in silver armor, already climbing the Dragon's "
                "chest strap"
            ),
            "wardrobe": "silver plate armor and a pink bow",
            "position_or_relation": "on the Dragon's chest strap, climbing",
        }]
        expected_subjects = copy.deepcopy(subjects)
        shots = [
            {
                "continuity_group": "mounting",
                "continuity_strategy": "independent",
                "spatial_setup": "The Kitten stands on the grass several feet away.",
                "subjects_on_screen": copy.deepcopy(subjects),
                "closing_blocking": "The Kitten remains standing on the grass.",
                "ending_beat": "The Kitten watches the Dragon.",
                "video_prompt": "The Kitten waits beside the Dragon. overall_soundscape: Wind.",
            },
            {
                "continuity_group": "mounting",
                "continuity_strategy": "continuous",
                "spatial_setup": next_opening,
                "subjects_on_screen": subjects,
                "video_prompt": "The Kitten approaches the Dragon. overall_soundscape: Wind.",
            },
        ]

        _prepare_h3_prompt_only_continuity(shots)

        opening_block = (
            shots[1]["video_prompt"].split("OPENING CONTINUITY:", 1)[1]
            .split("overall_soundscape:", 1)[0]
            .strip()
        )
        self.assertEqual(opening_block, next_opening + ".")
        self.assertNotIn("already climbing", opening_block)
        self.assertNotIn("chest strap", opening_block)

        previous_final = (
            shots[0]["video_prompt"].split("FINAL BLOCKING:", 1)[1]
            .split("overall_soundscape:", 1)[0]
        )
        self.assertIn("remains standing on the grass", previous_final)
        self.assertNotIn(next_opening, previous_final)
        self.assertNotIn("already climbing", previous_final)
        self.assertNotIn("chest strap", previous_final)
        self.assertEqual(shots[0]["closing_blocking"], "The Kitten remains standing on the grass.")
        self.assertEqual(shots[0]["ending_beat"], "The Kitten watches the Dragon.")
        self.assertFalse(shots[0].get("action_beats"))
        self.assertEqual(shots[1]["subjects_on_screen"], expected_subjects)


if __name__ == "__main__":
    unittest.main()
