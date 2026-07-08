"""
Tests for the Pass 3 polish storyboard-format guard.

User-reported regression after Multi-Shot LoRA Mode shipped:
  - Pass 2 correctly produced Format B storyboard content
    ("Shot 1 (Medium Two-Shot, 8s): The woman in dark brown dress
     wipes the display case...")
  - Pass 3 polish rewrote every storyboard block into flowing prose
    ("The woman in the dark brown dress slowly wipes... A close-up
     focuses on..."), destroying the IC-LoRA trigger pattern. The
    LoRA was trained on the literal "Shot N (Camera, Xs):"
    structure — without it, no internal cuts render.

Fix: Pass 3 detects storyboard format from the input prompt text
itself (via _is_storyboard_format) and skips the polish LLM call
entirely for storyboard-formatted prompts. Detection uses a regex
matching the literal "Shot N (Camera, Xs):" pattern, robust to the
multishot flag being missing or wrong.
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
    _is_storyboard_format,
)


class TestStoryboardDetector(unittest.TestCase):

    def test_detects_storyboard_format(self):
        # The exact production output that triggered the bug
        text = (
            "Shot 1 (Medium Two-Shot, 8s): The woman in dark brown dress "
            "wipes the glass display case slowly. "
            "Shot 2 (Close-up, 7s): She stops wiping just long enough "
            "to look up at the man."
        )
        self.assertTrue(_is_storyboard_format(text))

    def test_detects_single_shot(self):
        text = "Shot 1 (Wide Shot, 5s): A figure stands at the gate."
        self.assertTrue(_is_storyboard_format(text))

    def test_detects_case_insensitive(self):
        text = "shot 1 (wide shot, 5s): something happens."
        self.assertTrue(_is_storyboard_format(text))

    def test_detects_multi_word_camera_type(self):
        text = "Shot 1 (Over-the-Shoulder, 6s): description."
        self.assertTrue(_is_storyboard_format(text))

    def test_does_not_detect_flowing_prose(self):
        text = (
            "The woman in dark brown dress slowly wipes the glass display "
            "case. The man in the cowboy hat slowly walks toward the "
            "counter. She says, 'Same as last time.'"
        )
        self.assertFalse(_is_storyboard_format(text))

    def test_does_not_detect_polished_flowing_prose(self):
        # The buggy polish output — describes camera changes in prose
        # rather than as structured Shot blocks.
        text = (
            "The woman in the dark brown dress slowly wipes the glass "
            "display case. A close-up focuses on the woman, who stops "
            "wiping just long enough to look up at the man. The camera "
            "shifts to an over-the-shoulder shot from the woman."
        )
        self.assertFalse(_is_storyboard_format(text),
                         "polished flowing prose must NOT be detected as storyboard")

    def test_empty_input(self):
        self.assertFalse(_is_storyboard_format(""))
        self.assertFalse(_is_storyboard_format(None))  # type: ignore[arg-type]

    def test_does_not_detect_word_shot_alone(self):
        # Plain prose mentioning "shot" should not trigger
        text = "The camera captures a single shot of the doorway."
        self.assertFalse(_is_storyboard_format(text))

    def test_does_not_detect_shot_without_duration_in_seconds(self):
        # Edge case — "Shot 1 (Medium):" without duration should not match
        # the strict pattern. Storyboard format requires the (Camera, Xs)
        # structure with seconds.
        text = "Shot 1 (Medium): description"
        self.assertFalse(_is_storyboard_format(text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
