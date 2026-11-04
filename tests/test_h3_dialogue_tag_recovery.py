"""Regressions for conservative recovery of a terminal duplicated <d> opener."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service
from services.h3_prompt_budget import H3PromptBudgetError
from services.h3_story_ledger import normalize_h3_dialogue_tags


DIALOGUE_LINES = [
    "We should leave before sunset.",
    "The old bridge is closing much sooner tonight.",
    "I already checked the northern road for damage twice.",
    "Then we can reach the station before dark.",
    "Take the map and follow me.",
    "I will stay close behind.",
]


def scripted_source(*, audio_fields=False):
    turns = [
        f"Mira says <d>[English] {line}</d>" for line in DIALOGUE_LINES[:-1]
    ]
    turns.append(f"Tavi replies <d>[English] {DIALOGUE_LINES[-1]}<d>")
    source = "Scene: Mira and Tavi plan their route.\n" + "\n".join(turns)
    if audio_fields:
        source += "\n\noverall_soundscape: Rain on a station roof.\nnon_diegetic_music: N/A"
    return source


class H3DialogueTagRecoveryTests(unittest.TestCase):
    def test_terminal_duplicate_is_closed_without_changing_authored_words(self):
        source = scripted_source()
        normalized = normalize_h3_dialogue_tags(source)

        self.assertEqual(
            normalized,
            source[:-3] + "</d>",
        )
        entries = llm_service._extract_h3_source_dialogue_entries(source)
        self.assertEqual([entry["words"] for entry in entries], DIALOGUE_LINES)
        self.assertEqual(
            normalized[entries[-1]["start"]:entries[-1]["end"]],
            f"<d>[English] {DIALOGUE_LINES[-1]}</d>",
        )

    def test_terminal_duplicate_before_blank_line_audio_fields_is_repaired(self):
        source = scripted_source(audio_fields=True)
        normalized = normalize_h3_dialogue_tags(source)

        self.assertIn(f"<d>[English] {DIALOGUE_LINES[-1]}</d>\n\n", normalized)
        self.assertTrue(normalized.endswith(
            "overall_soundscape: Rain on a station roof.\nnon_diegetic_music: N/A"
        ))
        self.assertEqual(normalized.count("overall_soundscape:"), 1)
        self.assertEqual(normalized.count("non_diegetic_music:"), 1)

    def test_word_count_keeps_all_six_exact_lines_under_the_hard_cap(self):
        from services.dialogue_writing import spoken_word_count

        self.assertEqual([spoken_word_count(line) for line in DIALOGUE_LINES], [5, 8, 9, 8, 6, 5])
        self.assertEqual(sum(spoken_word_count(line) for line in DIALOGUE_LINES), 41)
        llm_service.validate_h3_source_dialogue_duration(scripted_source(), 14.4)

    def test_recovered_source_exceeding_duration_cap_fails_before_generation(self):
        with patch.object(llm_service, "generate") as writer:
            with self.assertRaises(H3PromptBudgetError) as raised:
                llm_service.enhance_prompt(
                    scripted_source(),
                    mode="video",
                    model_type="minimax_h3_fused_turbo",
                    duration_seconds=13,
                    planning_style="adaptive",
                )

        self.assertIn("41 spoken words", str(raised.exception))
        self.assertIn("at most 39", str(raised.exception))
        self.assertIn("Your original lines have not been changed", str(raised.exception))
        writer.assert_not_called()

    def test_ambiguous_duplicate_forms_remain_unchanged_and_fail_admission(self):
        ambiguous = (
            "Mira says <d>[English] We should leave <d>before sunset.<d>",
            "Mira says <d>[English] We should leave<d> extra stage direction.",
            "Mira says <d>[English] We should leave<d>\n\nsummary: another event",
            "Mira says <d>[English] We should leave<d>\n\nnon_diegetic_music: N/A\n"
            "<Subject 1> is a location reference.",
            "Mira says <d>[English] First malformed line<d>\n\n"
            "non_diegetic_music: N/A\n\nTavi says <d>[English] Second malformed line<d>",
            "Mira says </d>[English] We should leave<d>",
        )

        for source in ambiguous:
            with self.subTest(source_shape=source.splitlines()[0]):
                self.assertEqual(normalize_h3_dialogue_tags(source), source)

        with patch.object(llm_service, "generate") as writer:
            with self.assertRaises(H3PromptBudgetError):
                llm_service.enhance_prompt(
                    ambiguous[0],
                    mode="video",
                    model_type="minimax_h3_fused_turbo",
                    duration_seconds=13,
                    planning_style="adaptive",
                )
        writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
