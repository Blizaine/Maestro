"""Visible time transitions need not use a finite light-change verb."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_recurrence import action_establishes_later_occasion, later_occasion_action_text


class NominalTimeTransitionTests(unittest.TestCase):
    def test_recorded_visual_transition_preserves_only_following_action(self):
        cue = (
            "A subtle shift in the gas lamp's amber glow and a slight change in "
            "the reflection on the cobblestones signals the passage of time into "
            "a subsequent evening. The air remains still and the platform is empty. "
        )
        text = "The florist places the first bouquet. " + cue + "The florist leaves a fresh bouquet."
        self.assertTrue(action_establishes_later_occasion(text))
        later = later_occasion_action_text(text)
        self.assertIn("leaves a fresh bouquet", later)
        self.assertNotIn("places the first bouquet", later)

    def test_unrelated_scene_and_lighting_change(self):
        text = "A change in the light establishes the next morning. The keeper waters the seedlings."
        self.assertTrue(action_establishes_later_occasion(text))
        self.assertIn("waters the seedlings", later_occasion_action_text(text))

    def test_mentions_and_ordinary_lighting_are_not_elapsed_occasions(self):
        for text in (
            "A subtle change in the light reveals the bouquet. The florist leaves a bouquet.",
            "She describes a change in the lighting into another evening. She leaves a bouquet.",
            "She imagines a shift in the amber glow into the next evening. She leaves a bouquet.",
            "A change in the lighting is planned to establish the next morning. She waters the tray.",
            "No change in the lighting occurs into the next morning. She waters the tray.",
            'She reads "a change in the lighting establishes the next morning." She waters the tray.',
        ):
            with self.subTest(text=text):
                self.assertFalse(action_establishes_later_occasion(text))
                self.assertEqual(later_occasion_action_text(text), "")


if __name__ == "__main__":
    unittest.main()
