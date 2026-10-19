"""Regressions for clip-local MiniMax H3 shot labels."""

from __future__ import annotations

import os
import sys
import unittest


_APP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "app")
)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.h3_dialogue import (  # noqa: E402
    compile_h3_clip_plans,
    compile_h3_official_prompt,
    validate_h3_prompt_contract,
)


class H3ShotNumberingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.subjects = [
            {
                "character_id": "kitten",
                "speaker_name": "Kitten",
                "visual_description": "the Kitten in silver armor",
            },
            {
                "character_id": "dragon",
                "speaker_name": "Dragon",
                "visual_description": "the Dragon with dark scales",
            },
        ]
        self.references = [{
            "type": "image",
            "role": "Kitten with pink bow standing upright holding a sword",
        }]

    @staticmethod
    def _visual(prompt: str) -> str:
        return prompt.split("detailed_description:", 1)[1].split(
            "overall_soundscape:", 1
        )[0]

    def test_empty_trailing_scene_transitions_do_not_create_cuts(self) -> None:
        for source_number in range(1, 7):
            with self.subTest(source_number=source_number):
                compiled, _ = compile_h3_official_prompt(
                    "Kitten rides Dragon through the misty field. "
                    f"<scenetrans> [Shot {source_number}] At 00:00.000 "
                    "</scenetrans>",
                    self.subjects,
                    [],
                    mode="ref2va",
                    references=self.references,
                )

                visual = self._visual(compiled)
                self.assertIn("Kitten", visual)
                self.assertIn("Dragon", visual)
                action_start = visual.index("through the misty field.")
                self.assertGreater(action_start, visual.index("[Shot 1]"))
                self.assertEqual(visual.count("[Shot 1]"), 1)
                self.assertNotIn("[Shot 2]", visual)
                self.assertNotIn("<scenetrans>", visual)
                self.assertNotIn("00:00.000", visual)
                self.assertEqual(
                    validate_h3_prompt_contract(
                        compiled,
                        mode="ref2va",
                        subjects=self.subjects,
                        references=self.references,
                    ),
                    [],
                )

    def test_timed_global_cut_labels_are_localized_without_rewriting_content(self) -> None:
        spoken = (
            "Keep [Shot 72] At 99:59.999 and <scenetrans> exactly as spoken."
        )

        for first_timestamp in ("00:02", "00:02.1", "00:02.12", "00:02.500"):
            with self.subTest(first_timestamp=first_timestamp):
                source = (
                    "Kitten rides Dragon over the valley. "
                    f"<scenetrans> [Shot 13] At {first_timestamp}: hard cut "
                    "to a close view of their faces. </scenetrans> "
                    "<scenetrans> [Shot 14] At 00:05.250: hard cut to the "
                    "Dragon banking over the valley. </scenetrans>"
                )
                compiled, _ = compile_h3_official_prompt(
                    source,
                    self.subjects,
                    [{"speaker_id": "kitten", "spoken_text": spoken}],
                    mode="ref2va",
                    references=self.references,
                    duration_seconds=8,
                )

                visual = self._visual(compiled)
                self.assertIn("[Shot 1]", visual)
                self.assertIn(
                    f"[Shot 2] At {first_timestamp}: hard cut to a close view",
                    visual,
                )
                self.assertIn("[Shot 3] At 00:05.250: hard cut to the Dragon", visual)
                self.assertNotIn("[Shot 13]", visual)
                self.assertNotIn("[Shot 14]", visual)
                self.assertIn(f"<d>[English] {spoken}</d>", visual)
                self.assertIn("(S1)", visual)
                self.assertEqual(
                    validate_h3_prompt_contract(
                        compiled,
                        [{"speaker_id": "kitten", "spoken_text": spoken}],
                        mode="ref2va",
                        subjects=self.subjects,
                        references=self.references,
                        duration_seconds=8,
                    ),
                    [],
                )

    def test_leading_global_marker_is_the_local_first_shot(self) -> None:
        compiled, _ = compile_h3_official_prompt(
            "[Shot 5] At 00:00.000: Kitten leaps onto Dragon's back and "
            "they race across the field.",
            self.subjects,
            [],
            mode="ref2va",
            references=self.references,
            duration_seconds=8,
        )

        visual = self._visual(compiled)
        self.assertIn("[Shot 1] At 00:00.000:", visual)
        self.assertNotIn("[Shot 5]", visual)
        self.assertNotIn("[Shot 2]", visual)
        self.assertEqual(
            validate_h3_prompt_contract(
                compiled,
                mode="ref2va",
                subjects=self.subjects,
                references=self.references,
                duration_seconds=8,
            ),
            [],
        )

    def test_nonmonotonic_timed_labels_are_left_for_strict_validation(self) -> None:
        compiled, _ = compile_h3_official_prompt(
            "Kitten rides Dragon across the field. "
            "[Shot 5] At 00:01.000: Kitten raises a paw. "
            "[Shot 2] At 00:02.000: Dragon turns toward camera.",
            self.subjects,
            [],
            mode="ref2va",
            references=self.references,
            duration_seconds=8,
        )

        errors = validate_h3_prompt_contract(
            compiled,
            mode="ref2va",
            subjects=self.subjects,
            references=self.references,
            duration_seconds=8,
        )
        self.assertTrue(any("visual shot numbers must be consecutive" in error for error in errors))
        visual = self._visual(compiled)
        self.assertIn("[Shot 5] At 00:01.000", visual)
        self.assertIn("[Shot 2] At 00:02.000", visual)

    def test_timed_local_numbering_gaps_are_not_compressed(self) -> None:
        compiled, _ = compile_h3_official_prompt(
            "Kitten watches Dragon. "
            "[Shot 1] At 00:00.000: Kitten raises a paw. "
            "[Shot 3] At 00:02.000: Dragon turns toward camera.",
            self.subjects,
            [],
            mode="ref2va",
            references=self.references,
            duration_seconds=8,
        )

        errors = validate_h3_prompt_contract(
            compiled,
            mode="ref2va",
            subjects=self.subjects,
            references=self.references,
            duration_seconds=8,
        )
        self.assertTrue(any("visual shot numbers must be consecutive" in error for error in errors))
        visual = self._visual(compiled)
        self.assertIn("[Shot 3] At 00:02.000", visual)

    def test_spoken_shot_literal_does_not_inflate_alignment_header(self) -> None:
        spoken = "Read [Shot 99] At 00:07.000 aloud, including <scenetrans>."
        compiled, _ = compile_h3_official_prompt(
            "Kitten watches Dragon. [Shot 8] At 00:03.500: Dragon takes "
            "flight over the field.",
            self.subjects,
            [{"speaker_id": "kitten", "spoken_text": spoken}],
            mode="l2va",
            duration_seconds=8,
        )

        self.assertIn("<Picture 1> (from [Shot 2])", compiled)
        self.assertNotIn("from [Shot 99]", compiled)
        self.assertIn(f"<d>[English] {spoken}</d>", compiled)

    def test_saved_plan_recompilation_is_idempotent_and_keeps_source(self) -> None:
        source = (
            "Kitten rides Dragon through the field. "
            "<scenetrans> [Shot 6] At 00:00.000 </scenetrans>"
        )
        plan = {
            "video_prompt": source,
            "_director_h3_prompt_mode": "ref2va",
            "_director_h3_model_family": "ref2va",
            "_director_subjects_on_screen": self.subjects,
            "_director_dialogue_beats": [],
            "_director_duration_sec": 8,
            "_director_h3_reference_manifest": self.references,
        }

        compile_h3_clip_plans([plan])
        first_compiled = plan["video_prompt"]
        self.assertEqual(plan["_director_h3_source_prompt"], source)
        compile_h3_clip_plans([plan])

        self.assertEqual(plan["video_prompt"], first_compiled)
        self.assertEqual(plan["_director_h3_compiled_prompt"], first_compiled)
        self.assertEqual(plan["_director_h3_source_prompt"], source)


if __name__ == "__main__":
    unittest.main()
