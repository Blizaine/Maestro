"""H3 rejects clearly attributed dialogue whose closing quote is missing."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service
from services.h3_dialogue_source import unclosed_h3_spoken_quote_error
from services.h3_prompt_budget import H3PromptBudgetError
from services.studio_enhancement import enhancement_request


SOURCE_UNCLOSED = (
    'Episode of Seinfeld. Jerry\'s apartment. camera focused on a door as it opens '
    'and George walks in, looking very strong and buff. Jerry is standing in his '
    'apartment with Elaine. Both Jerry and Elaine are very obese. Jerry looks at '
    'him and says "George! what the hell man?!" George says in a passionate and '
    'excite tone "you guys will never believe it! Maestro can make us look any '
    'way the user wants!  look at me!" Elaine says in a frustrated voice '
    '"oh great.  thanks a lot Blizzaine.  You son of a [beep censor]. '
    'the audience laughs.'
)
SOURCE_CORRECTED = SOURCE_UNCLOSED.replace(
    'voice "oh great.  thanks a lot Blizzaine.  You son of a [beep censor]. '
    'the audience laughs.',
    'voice "oh great.  thanks a lot Blizzaine.  You son of a [beep censor]." '
    'The audience laughs.',
)
REFERENCES = [
    {"type": "image", "path": "jerry.png", "role": "Jerry"},
    {"type": "image", "path": "elaine.png", "role": "Elaine"},
    {"type": "image", "path": "george.png", "role": "George"},
]


def reference_context(prompt: str) -> str:
    payload, sequence = enhancement_request(
        {
            "prompt": prompt,
            "model_type": "minimax_h3_ref2va_fused_turbo",
            "video_length": 480,
            "sliding_window_size": 480,
            "minimax_h3_references": REFERENCES,
        },
        {"architecture": "minimax_h3_ref2va", "omni_reference": True, "fps": 24},
    )
    assert not sequence
    return payload["reference_context"]


class H3DialogueSourceTests(unittest.TestCase):
    def test_unclosed_final_seinfeld_line_fails_before_writer_with_reference_map(self):
        context = reference_context(SOURCE_UNCLOSED)
        with patch.object(llm_service, "generate") as writer:
            with self.assertRaises(H3PromptBudgetError) as raised:
                llm_service.enhance_prompt(
                    SOURCE_UNCLOSED,
                    mode="video",
                    model_type="minimax_h3_ref2va_fused_turbo",
                    duration_seconds=20,
                    reference_context=context,
                    planning_style="adaptive",
                )

        self.assertIn("unclosed quoted dialogue line", str(raised.exception))
        self.assertIn("Close the dialogue quote", str(raised.exception))
        self.assertIn("stage or sound directions", str(raised.exception))
        writer.assert_not_called()

    def test_closed_three_speaker_source_keeps_exact_lines_and_reference_owners(self):
        context = reference_context(SOURCE_CORRECTED)
        self.assertIsNone(unclosed_h3_spoken_quote_error(SOURCE_CORRECTED))
        llm_service.validate_h3_source_dialogue_duration(SOURCE_CORRECTED, 20)

        entries = llm_service._extract_h3_source_dialogue_entries(SOURCE_CORRECTED, context)
        self.assertEqual([entry["subject_id"] for entry in entries], [1, 3, 2])
        self.assertEqual([entry["speaker_id"] for entry in entries], [1, 2, 3])
        self.assertEqual(
            [entry["words"] for entry in entries],
            [
                "George! what the hell man?!",
                "you guys will never believe it! Maestro can make us look any "
                "way the user wants!  look at me!",
                "oh great.  thanks a lot Blizzaine.  You son of a [beep censor].",
            ],
        )

    def test_direct_ref2va_fallback_rejects_unclosed_cued_dialogue(self):
        with self.assertRaises(H3PromptBudgetError):
            llm_service._build_h3_ref2va_tagged_fallback(
                SOURCE_UNCLOSED,
                reference_context(SOURCE_UNCLOSED),
                duration_seconds=20,
                planning_style="adaptive",
            )

    def test_curly_quotes_and_screenplay_labels_are_checked(self):
        self.assertIsNotNone(unclosed_h3_spoken_quote_error(
            'Elaine says in a frustrated voice, “Oh great. Thanks a lot.'
        ))
        self.assertIsNotNone(unclosed_h3_spoken_quote_error(
            'ELAINE: "Oh great. Thanks a lot. The audience laughs.'
        ))

    def test_balanced_quotes_nested_quotes_and_escaped_quotes_are_admitted(self):
        for source in (
            'Elaine says, “He called "Newman" his friend.”',
            r'Elaine says, "He called \"Newman\" his friend."',
            'Jerry\'s nickname is "Newman". The episode is titled "The Contest".',
        ):
            with self.subTest(source=source):
                self.assertIsNone(unclosed_h3_spoken_quote_error(source))

    def test_unmatched_visual_titles_labels_and_tagged_text_are_not_rejected(self):
        for source in (
            'A poster reads "The Frozen North. Elaine walks into the room.',
            'Sign says "OPEN. Elaine enters the room.',
            'Title: "A bad day. The camera moves across the room.',
            'Episode title: "The Contest. George enters the apartment.',
            'STYLE: "soft contrast and warm colors.',
            '<d>[English] He said "Newman yesterday.</d> The others listen.',
            "Elaine's apartment is beside Jerry's. George enters quietly.",
        ):
            with self.subTest(source=source):
                self.assertIsNone(unclosed_h3_spoken_quote_error(source))


if __name__ == "__main__":
    unittest.main()
