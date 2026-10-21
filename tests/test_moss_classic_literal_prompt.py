"""Regression coverage for literal MOSS prompts in the shared enhancer path."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from shared.prompt_enhancer.prompt_enhance_utils import generate_cinematic_prompt  # noqa: E402


class FakeMessageModel:
    def __init__(self):
        self.messages = None

    def generate_messages(self, messages, max_new_tokens, **kwargs):
        self.messages = messages
        return ["enhanced prompt"] * len(messages)


def run_enhancer(prompt: str, *, literal_prompt: bool = False):
    model = FakeMessageModel()
    result = generate_cinematic_prompt(
        image_caption_model=None,
        image_caption_processor=None,
        prompt_enhancer_model=model,
        prompt_enhancer_tokenizer=None,
        prompt=prompt,
        prompt_enhancer_instructions="Keep the supplied prompt syntax intact.",
        literal_prompt=literal_prompt,
    )
    return result, model.messages[0]


class MossClassicLiteralPromptTests(unittest.TestCase):
    def test_literal_json_prompt_keeps_email_at_artist_and_double_at_in_user_message(self):
        prompt = (
            '  {"summary":{"caption":"Portrait for email@example.com"},'
            '"items":[{"label":"text","text":"@artist @@ exact copy"}]}  '
        )

        result, messages = run_enhancer(prompt, literal_prompt=True)

        self.assertEqual(result, ["enhanced prompt"])
        self.assertEqual(messages[0], {
            "role": "system",
            "content": "Keep the supplied prompt syntax intact.",
        })
        self.assertEqual(messages[1], {"role": "user", "content": f"user_prompt: {prompt}"})

    def test_default_single_at_still_appends_user_system_suffix(self):
        _, messages = run_enhancer("A quiet city @Make the signs glow blue.")

        self.assertEqual(messages[1]["content"], "user_prompt: A quiet city")
        self.assertEqual(
            messages[0]["content"],
            "Keep the supplied prompt syntax intact.\n"
            "Follow these additional user instructions with higher priority if they conflict "
            "with the guidance above:\nMake the signs glow blue.",
        )

    def test_default_double_at_still_replaces_the_system_prompt(self):
        _, messages = run_enhancer("A quiet city @@Use only a minimal watercolor style.")

        self.assertEqual(messages[1]["content"], "user_prompt: A quiet city")
        self.assertEqual(messages[0]["content"], "Use only a minimal watercolor style.")


if __name__ == "__main__":
    unittest.main()
