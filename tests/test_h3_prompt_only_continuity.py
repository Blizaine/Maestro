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
)


class H3PromptOnlyContinuityTests(unittest.TestCase):
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
        self.assertIn(next_opening, previous_final)
        self.assertNotIn("already climbing", previous_final)
        self.assertNotIn("chest strap", previous_final)
        self.assertEqual(shots[0]["closing_blocking"], next_opening)
        self.assertEqual(shots[1]["subjects_on_screen"], expected_subjects)


if __name__ == "__main__":
    unittest.main()
