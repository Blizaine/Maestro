"""H3 rejects clearly attributed dialogue whose closing quote is missing."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service
from services.h3_dialogue_source import unclosed_h3_spoken_quote_error
from services.h3_prompt_budget import H3PromptBudgetError
from services.h3_story_ledger import (
    _canonicalize_h3_source_dialogue_cues,
    extract_h3_source_intent,
    extract_locked_dialogue,
    plan_h3_story_segments,
)
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
INTERVIEW_LINE = (
    "Evan thought it would be a good idea for us to rehearse the opening before the clients arrived. "
    "He made us all label the props. But he never told the crew downstairs, so Dylan thought the visitors were here. Oops!"
)
INTERVIEW_REPLY = "you can never be too prepared."
INTERVIEW_SOURCE = (
    "Third-person over-the-shoulder view of Morgan Tate <picture 2> as he enters a rehearsal space. "
    "Mina Park <picture 4> is at the reception desk holding a clipboard. Morgan first hands her an envelope, "
    "then presents a poster to Jim Hart <picture 3> and gives a package to Noah Bell <picture 5>. "
    "He hands the final envelope to Evan Wells. Then Dylan Reed <picture 6> walks up and rings the bell. "
    "A small counter in the top left tracks the deliveries in one continuous view. "
    f'\n\nScene cuts to interview with Mina <picture 4>. She has a name badge on. She says "{INTERVIEW_LINE}" '
    f'\n\nCamera cuts back to the office with Evan Wells <picture 1>. Evans looks at the camera and says "{INTERVIEW_REPLY}"'
)
INTERVIEW_REFERENCE_ROLES = (
    "Evan Wells",
    "Morgan Tate",
    "im Hart",
    "Mina Park",
    "Noah Bell",
    "Dylan Reed",
)
INTERVIEW_REFERENCE_CONTEXT = "\n".join(
    f"<Subject {index}> is {role} from <Picture {index}>, preserving identity."
    for index, role in enumerate(INTERVIEW_REFERENCE_ROLES, start=1)
)


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
    def test_scene_cut_speakers_resolve_to_explicit_picture_owners(self):
        expected = [
            ("Mina Park", INTERVIEW_LINE),
            ("Evan Wells", INTERVIEW_REPLY),
        ]
        for reference_context in ("", INTERVIEW_REFERENCE_CONTEXT):
            with self.subTest(has_runtime_references=bool(reference_context)):
                dialogue = extract_locked_dialogue(
                    INTERVIEW_SOURCE,
                    reference_context=reference_context,
                )
                self.assertEqual(
                    [(item["speaker"], item["text"]) for item in dialogue],
                    expected,
                )
                intent = extract_h3_source_intent(
                    INTERVIEW_SOURCE,
                    reference_context=reference_context,
                )
                self.assertNotIn("She", intent["cast_names"])
                self.assertNotIn("Evans", intent["cast_names"])

    def test_scene_cut_pronoun_stays_unresolved_when_picture_owners_are_ambiguous(self):
        source = (
            "Scene cuts to an interview with Mina <picture 4> and Jim Hart <picture 3>. "
            "She says \"We are ready.\""
        )
        dialogue = extract_locked_dialogue(
            source,
            reference_context=INTERVIEW_REFERENCE_CONTEXT,
        )
        self.assertEqual([item["speaker"] for item in dialogue], ["She"])

    def test_scene_cut_pronoun_stays_unresolved_after_another_person_is_introduced(self):
        for intervening_person in (
            "Evan Wells steps into frame.",
            "Another interviewer joins Mina.",
        ):
            with self.subTest(intervening_person=intervening_person):
                source = (
                    "Scene cuts to interview with Mina <picture 4>. "
                    f"{intervening_person} She says \"We are ready.\""
                )
                dialogue = extract_locked_dialogue(
                    source,
                    reference_context=INTERVIEW_REFERENCE_CONTEXT,
                )
                self.assertEqual([item["speaker"] for item in dialogue], ["She"])

    def test_scene_cut_cue_normalization_keeps_dialogue_offsets_aligned(self):
        source = (
            "Scene cuts to interview with Mina <picture 4>. "
            "She says < d>We are ready.</ d>"
        )
        canonical = _canonicalize_h3_source_dialogue_cues(
            source,
            INTERVIEW_REFERENCE_CONTEXT,
        )
        self.assertIn("Mina Park says <d>We are ready.</d>", canonical)
        self.assertEqual(canonical.count("We are ready."), 1)

    def test_deterministic_planning_fallback_keeps_canonical_speakers_and_ids(self):
        def offline_generate(**_kwargs):
            raise RuntimeError("offline regression")

        result = plan_h3_story_segments(
            INTERVIEW_SOURCE,
            segment_durations=[14.375, 13.625],
            mode="sliding_window",
            camera_coverage="multi_shot",
            reference_context=INTERVIEW_REFERENCE_CONTEXT,
            expect_dialogue=True,
            planning_style="faithful",
            llm_generate=offline_generate,
        )
        dialogue = [
            (line["speaker"], line["speaker_id"], line["text"])
            for segment in result["segments"]
            for shot in segment["shots"]
            for line in shot.get("dialogue") or []
        ]
        self.assertEqual(
            dialogue,
            [
                ("Mina Park", "S1", INTERVIEW_LINE),
                ("Evan Wells", "S2", INTERVIEW_REPLY),
            ],
        )
        self.assertNotIn("She", result["source_intent"]["cast_names"])
        self.assertNotIn("Evans", result["source_intent"]["cast_names"])

    def test_three_window_fallback_keeps_the_interview_with_the_correct_speaker(self):
        def offline_generate(**_kwargs):
            raise RuntimeError("offline regression")

        extended_line = f"{INTERVIEW_LINE} Our rehearsal turned out to be more convincing than I expected."
        result = plan_h3_story_segments(
            INTERVIEW_SOURCE.replace(INTERVIEW_LINE, extended_line),
            segment_durations=[14.375, 13.625, 13.625],
            mode="sliding_window",
            camera_coverage="multi_shot",
            reference_context=INTERVIEW_REFERENCE_CONTEXT,
            expect_dialogue=True,
            planning_style="adaptive",
            llm_generate=offline_generate,
        )
        context = result["camera_checkpoint"]["context"]
        self.assertEqual(context["durations"], [14.375, 13.625, 13.625])
        self.assertEqual(
            [(beat["segment"], beat["dialogue_ids"])
             for beat in context["ledger"]["beats"] if beat.get("dialogue_ids")],
            [(2, ["D1"]), (3, ["D2"])],
        )
        self.assertEqual(
            [item["segments"] for item in result["dialogue_fragments"]],
            [[1, 2]],
        )

        rendered = [
            (segment_index, line["speaker"], line["speaker_id"], line["text"])
            for segment_index, segment in enumerate(result["segments"], start=1)
            for shot in segment["shots"]
            for line in (shot.get("dialogue") or [])
        ]
        mina = [item for item in rendered if item[1] == "Mina Park"]
        evan = [item for item in rendered if item[1] == "Evan Wells"]
        self.assertEqual([item[0] for item in mina], [1, 2])
        self.assertEqual([item[2] for item in mina], ["S1", "S1"])
        self.assertEqual(" ".join(item[3] for item in mina), extended_line)
        self.assertEqual([(item[0], item[2]) for item in evan], [(3, "S2")])

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
