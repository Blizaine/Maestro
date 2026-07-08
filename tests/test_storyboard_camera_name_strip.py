"""
Tests for the storyboard camera-name leak stripper.

User-reported issue: Pass 2 sometimes embeds character names inside
the camera-type parens of Format B storyboard blocks, like:
  "Shot 2 (Close-up on Henry, 7s):"
  "Shot 2 (Over-the-Shoulder from Mary, 7s):"

The IC-LoRA was trained on clean camera-type tokens (e.g. "Close-up",
"Over-the-Shoulder"). Names in the parens break the trained pattern
and the LoRA doesn't render the internal cut correctly.

Fix: strip_storyboard_camera_name_leaks() detects the "on/of/from/
with/over Name" pattern after the camera-type word and removes it
while preserving the rest of the structure.
"""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.prompt_polish import (  # noqa: E402
    strip_storyboard_camera_name_leaks,
)


class TestStripStoryboardCameraNameLeaks(unittest.TestCase):

    def test_close_up_on_henry(self):
        # The exact production text the user reported
        text = "Shot 2 (Close-up on Henry, 7s): The man pauses his cleaning."
        cleaned, count = strip_storyboard_camera_name_leaks(text)
        self.assertEqual(count, 1)
        self.assertNotIn("Henry", cleaned)
        self.assertIn("Shot 2 (Close-up, 7s):", cleaned)

    def test_medium_shot_of_mary(self):
        text = "Shot 3 (Medium Shot of Mary, 5s): She wipes the counter."
        cleaned, count = strip_storyboard_camera_name_leaks(text)
        self.assertEqual(count, 1)
        self.assertNotIn("Mary", cleaned)
        self.assertIn("Shot 3 (Medium Shot, 5s):", cleaned)

    def test_over_the_shoulder_from_mary(self):
        text = "Shot 4 (Over-the-Shoulder from Mary, 7s): The man steps forward."
        cleaned, count = strip_storyboard_camera_name_leaks(text)
        self.assertEqual(count, 1)
        self.assertNotIn("Mary", cleaned)
        self.assertIn("Shot 4 (Over-the-Shoulder, 7s):", cleaned)

    def test_multiple_shots_in_one_string(self):
        text = (
            "Shot 1 (Wide Shot, 5s): Establishing scene. "
            "Shot 2 (Close-up on Henry, 7s): He speaks. "
            "Shot 3 (Medium Shot of Mary, 5s): She replies. "
            "Shot 4 (Two-Shot, 3s): Both visible."
        )
        cleaned, count = strip_storyboard_camera_name_leaks(text)
        self.assertEqual(count, 2)  # Shots 1 and 4 are clean
        self.assertNotIn("Close-up on Henry", cleaned)
        self.assertNotIn("Medium Shot of Mary", cleaned)
        self.assertIn("Shot 2 (Close-up, 7s):", cleaned)
        self.assertIn("Shot 3 (Medium Shot, 5s):", cleaned)
        # Untouched shots stay intact
        self.assertIn("Shot 1 (Wide Shot, 5s):", cleaned)
        self.assertIn("Shot 4 (Two-Shot, 3s):", cleaned)

    def test_with_name_pattern(self):
        text = "Shot 2 (Close-up with Mary, 7s): Description."
        cleaned, count = strip_storyboard_camera_name_leaks(text)
        self.assertEqual(count, 1)
        self.assertIn("Shot 2 (Close-up, 7s):", cleaned)

    def test_clean_storyboard_unchanged(self):
        # No camera-name leaks → no changes.
        text = (
            "Shot 1 (Wide Shot, 5s): The man approaches. "
            "Shot 2 (Close-up, 7s): He turns his head. "
            "Shot 3 (Two-Shot, 3s): They face each other."
        )
        cleaned, count = strip_storyboard_camera_name_leaks(text)
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, text)

    def test_empty_input(self):
        cleaned, count = strip_storyboard_camera_name_leaks("")
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, "")
        cleaned, count = strip_storyboard_camera_name_leaks(None)  # type: ignore[arg-type]
        self.assertEqual(count, 0)

    def test_flowing_prose_not_touched(self):
        # No "Shot" prefix → no risk of false-positive on dialogue
        # text that happens to have "on Henry" / "of Mary".
        text = "The shot was aimed on Henry. Mary spoke softly to him."
        cleaned, count = strip_storyboard_camera_name_leaks(text)
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, text)

    def test_preserves_description_text_after_colon(self):
        text = "Shot 2 (Close-up on Henry, 7s): He says, 'Hello, Henry.'"
        cleaned, count = strip_storyboard_camera_name_leaks(text)
        self.assertEqual(count, 1)
        # Description text after the colon (which mentions Henry inside
        # dialogue) should be untouched.
        self.assertIn("He says, 'Hello, Henry.'", cleaned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
