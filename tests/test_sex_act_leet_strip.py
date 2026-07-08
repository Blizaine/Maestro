"""
Tests for the sex-act leet trigger strip.

User-reported safety leak: a SFW music video about a football coach
had "bl0wj0b" in a keyframe_prompt. The LLM treated leet-coded LoRA
trigger words from the active LoRA selection as generic energy
boosters and dropped them into prompts that should never contain
them.

Leet trigger tokens like bl0wj0b, m15510n4ry, c0wg1rl name SPECIFIC
SEX ACTS. They are video LoRA triggers — they must NEVER appear in:
  - Still-image prompts (image_prompt, keyframe_prompts)
  - SFW content of any kind (music video about football, dialogue
    scenes, action sequences, etc.)

The strip_sex_act_leet_tokens function is the deterministic safety
net. The LoRA-hint warning in the system prompt explains the rule
to the LLM, but smaller models ignore it; the strip enforces the
rule unconditionally after Pass 2.
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
    strip_sex_act_leet_tokens,
)


class TestStripSexActLeetTokens(unittest.TestCase):

    def test_user_reported_keyframe_leak(self):
        # The exact production text the user reported
        text = "bl0wj0b — the man's head snaps sharply forward and back."
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertGreater(count, 0)
        self.assertNotIn("bl0wj0b", cleaned)

    def test_paren_wrapped_leet(self):
        # (c0wg1rl) The man is captured at the apex...
        text = "(c0wg1rl) The man is captured at the apex of his chest thrust."
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertGreater(count, 0)
        self.assertNotIn("c0wg1rl", cleaned)
        self.assertNotIn("(", cleaned)
        # Description text should survive
        self.assertIn("The man is captured", cleaned)

    def test_missionary_position_leet(self):
        text = "(m15510n4ry) The man leans further forward."
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertGreater(count, 0)
        self.assertNotIn("m15510n4ry", cleaned)

    def test_doggy_style_leet(self):
        text = "Shot 3 (Wide Shot, 5s): d0gg1e description here."
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertGreater(count, 0)
        self.assertNotIn("d0gg1e", cleaned)

    def test_reverse_cowgirl_leet(self):
        text = "r3v3rs3_c0wg1rl. description follows."
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertGreater(count, 0)
        self.assertNotIn("r3v3rs3_c0wg1rl", cleaned)

    def test_oral_leet(self):
        text = "(0r4l) She lowers her head."
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertGreater(count, 0)
        self.assertNotIn("0r4l", cleaned)

    def test_multiple_leet_tokens_in_one_string(self):
        text = (
            "Shot 1: bl0wj0b The man receives. "
            "Shot 2: (c0wg1rl) The woman straddles."
        )
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertGreaterEqual(count, 2)
        self.assertNotIn("bl0wj0b", cleaned)
        self.assertNotIn("c0wg1rl", cleaned)

    def test_case_insensitive(self):
        text = "BL0WJ0B description"
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertGreater(count, 0)
        self.assertNotIn("BL0WJ0B", cleaned)
        self.assertNotIn("bl0wj0b", cleaned.lower())

    def test_clean_prompt_unchanged(self):
        text = (
            "The man in red shirt stands center stage. He raises the "
            "microphone. The crowd cheers as he begins to sing."
        )
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, text)

    def test_legitimate_digit_tokens_preserved(self):
        # Real words/tokens with digits that aren't leet sex-acts.
        # Should NOT be touched.
        text = (
            "The RTX4090 GPU renders the 1920x1080 frame at 24fps. "
            "The model used Flux2 architecture."
        )
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertEqual(count, 0)
        self.assertIn("RTX4090", cleaned)
        self.assertIn("1920x1080", cleaned)
        self.assertIn("Flux2", cleaned)

    def test_empty_input(self):
        cleaned, count = strip_sex_act_leet_tokens("")
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, "")
        cleaned, count = strip_sex_act_leet_tokens(None)  # type: ignore[arg-type]
        self.assertEqual(count, 0)

    def test_preserves_surrounding_text(self):
        text = (
            "Shot 2 (Medium Two-Shot, 8s): The woman in red is on stage. "
            "(c0wg1rl) The crowd roars. She raises her arms."
        )
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertEqual(count, 1)
        self.assertIn("Shot 2 (Medium Two-Shot, 8s):", cleaned)
        self.assertIn("The crowd roars", cleaned)
        self.assertIn("She raises her arms", cleaned)
        self.assertNotIn("c0wg1rl", cleaned)

    def test_em_dash_separator(self):
        # The exact pattern from the user's bug report
        text = "bl0wj0b — the man's head snaps sharply"
        cleaned, count = strip_sex_act_leet_tokens(text)
        self.assertGreater(count, 0)
        self.assertNotIn("bl0wj0b", cleaned)
        self.assertNotIn("—", cleaned[:5])  # em-dash also gone


if __name__ == "__main__":
    unittest.main(verbosity=2)
