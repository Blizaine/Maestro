"""Bounded structural contracts for generated dialogue and closed scripts."""

from pathlib import Path
import json
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service
from services.adaptive_enhancement import draft_spoken_exchange
from services.dialogue_writing import (
    creative_dialogue_violations,
    explicit_multi_speaker_dialogue,
)


class DialogueCompletionContractTests(unittest.TestCase):
    def test_anonymous_reciprocal_conversation_rejects_one_speaker(self):
        prompt = "Write a conversation between two strangers waiting for a delayed train."
        turns = [{
            "speaker": "Traveler",
            "text": "The train is late because a signal failure closed the northern track this afternoon.",
        }]

        issues = creative_dialogue_violations(prompt, turns)

        self.assertIn(
            "the requested conversation needs at least two distinct speakers",
            issues,
        )
        self.assertTrue(explicit_multi_speaker_dialogue(prompt))

    def test_adaptive_draft_repairs_incomplete_anonymous_conversation(self):
        prompt = "Write a conversation between two strangers waiting for a delayed train."
        generator = Mock(side_effect=[
            json.dumps({"turns": [{
                "speaker": "Traveler",
                "text": "The train is late because a signal failure closed the northern track this afternoon.",
            }]}),
            json.dumps({"turns": [
                {
                    "speaker": "Traveler",
                    "text": "The train is late because a signal failure closed the northern track this afternoon.",
                },
                {"speaker": "Stranger", "text": "Then we should wait by the east entrance."},
            ]}),
        ])

        result = draft_spoken_exchange(
            prompt, 8, generator, language="English",
        )

        self.assertEqual(generator.call_count, 2)
        entries = llm_service._extract_h3_source_dialogue_entries(result)
        self.assertEqual([entry["speaker_id"] for entry in entries], [1, 2])

    def test_one_turn_and_explicit_monologue_do_not_require_two_speakers(self):
        one_turn = "Write a concise one-turn conversation between two strangers waiting for a delayed train."
        monologue = "Write a monologue from a conductor explaining why the train is delayed."
        self.assertFalse(explicit_multi_speaker_dialogue(one_turn))
        self.assertFalse(explicit_multi_speaker_dialogue(monologue))
        self.assertFalse(explicit_multi_speaker_dialogue(
            "Two passengers listen while a narrator talks about the delay."
        ))

        generator = Mock(return_value=json.dumps({"turns": [
            {"speaker": "Traveler", "text": "The signal failed."},
        ]}))
        result = draft_spoken_exchange(one_turn, 8, generator, language="English")
        self.assertEqual(len(llm_service._extract_h3_source_dialogue_entries(result)), 1)
        generator.assert_called_once()

    def test_closed_adaptive_script_rejects_an_invented_guest_line(self):
        prompt = (
            "Spoken script for this adaptation. Only use these lines, in this order, beside these speakers.\n"
            "Maya (S1) says, <d>[English] The train is late.</d>"
        )
        result = _ref2va_result([
            ("Maya", 1, "The train is late."),
            ("Guest", 2, "I can show you another platform."),
        ])

        self.assertFalse(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(
            prompt, result, None,
        ))

    def test_closed_adaptive_script_requires_exact_order_count_and_content(self):
        first = "The train is late."
        second = "The platform is empty."
        prompt = (
            "Spoken script for this adaptation. Only use these lines, in this order, beside these speakers.\n"
            f"Maya (S1) says, <d>[English] {first}</d>\n"
            f"Leo (S2) says, <d>[English] {second}</d>"
        )
        valid = _ref2va_result([("Maya", 1, first), ("Leo", 2, second)])
        self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(
            prompt, valid, None,
        ))

        invalid_sequences = (
            [("Maya", 1, first), ("Maya", 1, first)],  # duplicate and missing
            [("Leo", 2, second)],  # missing
            [("Leo", 2, second), ("Maya", 1, first)],  # reordered
            [("Maya", 1, "The train has been delayed."), ("Leo", 2, second)],  # changed
        )
        for turns in invalid_sequences:
            with self.subTest(turns=turns):
                result = _ref2va_result(turns)
                self.assertFalse(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(
                    prompt, result, None,
                ))

    def test_unlocked_source_line_still_allows_a_creative_guest_reply(self):
        prompt = (
            'Maya says, <d>[English] The train is late.</d> '
            "A stranger improvises a reply."
        )
        result = _ref2va_result([
            ("Maya", 1, "The train is late."),
            ("Stranger", 2, "The east entrance is still open."),
        ])

        self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(
            prompt, result, None,
        ))


def _ref2va_result(turns):
    dialogue = " ".join(
        f"{speaker} (S{speaker_id}) says, <d>[English] {words}</d>"
        for speaker, speaker_id, words in turns
    )
    return (
        "subject_definitions: N/A\n"
        "summary: A short station scene.\n"
        "retention_analysis: Preserve the requested spoken content.\n"
        f"detailed_description: {dialogue}\n"
        "overall_soundscape: Quiet station ambience.\n"
        "non_diegetic_music: N/A"
    )


if __name__ == "__main__":
    unittest.main()
