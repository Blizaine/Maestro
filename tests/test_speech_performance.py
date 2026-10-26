"""Insertion-only Fish directing; no inference or audio is used here."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.speech_performance import (
    apply_performance_edits, direct_fish_section, needs_fish_direction,
)


SCRIPT = (
    "I kept the letter for years, though I never understood why it mattered. "
    "That evening the rain fell against the windows while I listened for the familiar footsteps. "
    "No one came to the door. I checked every lock and waited in the kitchen until the kettle stopped. "
    "Then I read the name on the envelope again. It was mine. I knew at last who had sent it."
)


class SpeechPerformanceTests(unittest.TestCase):
    def test_only_native_markup_is_inserted_without_rewriting_spoken_text(self):
        source = "Speaker 1: [calm] I checked every lock.\nSpeaker 2: The name was mine."
        response = json.dumps({"edits": [
            {"anchor": "every lock", "tag": "emphasis", "cue": "uneasy"},
            {"anchor": "mine", "tag": "whisper"},
        ]})
        result, reasons, count = apply_performance_edits(source, response)
        self.assertEqual(result, "Speaker 1: [calm] I checked [uneasy] <emphasis>every lock</emphasis>.\nSpeaker 2: The name was <whisper>mine</whisper>.")
        self.assertEqual(reasons, [])
        self.assertEqual(count, 2)

    def test_missing_repeated_partial_and_protected_anchors_cannot_change_speech(self):
        source = "Speaker 2: [sad] All is well, all is well. A wall remains."
        edits = [
            {"anchor": "Speaker 2:", "tag": "emphasis"},
            {"anchor": "sad", "tag": "soft"},
            {"anchor": "is well", "tag": "slow"},
            {"anchor": "missing", "tag": "fast"},
            {"anchor": "all", "tag": "stress"},  # also occurs inside wall
        ]
        result, reasons, count = apply_performance_edits(source, json.dumps({"edits": edits}))
        self.assertEqual(result, source)
        self.assertEqual(count, 0)
        self.assertEqual(len(reasons), 5)

    def test_valid_edits_survive_other_bad_suggestions_and_cannot_overlap(self):
        response = json.dumps({"edits": [
            {"anchor": "every lock", "tag": "stress"},
            {"anchor": "lock", "cue": "afraid"},
            {"anchor": "name", "replacement": "someone else"},
            {"anchor": "envelope", "tag": "im_start"},
            {"anchor": "kitchen", "cue": "[angry]"},
        ]})
        result, reasons, count = apply_performance_edits(SCRIPT, response)
        self.assertEqual(result, SCRIPT.replace("every lock", "<stress>every lock</stress>"))
        self.assertEqual(count, 1)
        self.assertEqual(len(reasons), 4)

    def test_complete_json_required_without_replacing_speech(self):
        for response in ('{"edits":[', '{"script":"replacement"}'):
            with self.subTest(response=response), self.assertRaises(ValueError):
                apply_performance_edits(SCRIPT, response)

    def test_excess_suggestions_apply_a_safe_bounded_subset_instead_of_failing(self):
        source = " ".join(f"phrase{index:02d}" for index in range(12))
        edits = [{"anchor": f"phrase{index:02d}", "tag": "emphasis"} for index in range(12)]
        with patch("services.speech_performance.record_advisory") as receipt:
            result, reasons, count = apply_performance_edits(source, json.dumps({"edits": edits}))
        self.assertEqual(count, 8)
        self.assertEqual(reasons, [])
        self.assertEqual(result.count("<emphasis>"), 8)
        self.assertEqual(result.replace("<emphasis>", "").replace("</emphasis>", ""), source)
        self.assertEqual(receipt.call_args.args[1]["suggested"], 12)
        self.assertEqual(receipt.call_args.args[1]["remaining_suggestions_omitted"], 4)

    def test_invalid_early_suggestion_does_not_hide_later_safe_edits(self):
        edits = [{"anchor": "missing", "tag": "emphasis"}] * 9 + [{"anchor": "letter", "tag": "soft"}]
        result, reasons, count = apply_performance_edits(SCRIPT, json.dumps({"edits": edits}))
        self.assertEqual(count, 1)
        self.assertEqual(len(reasons), 9)
        self.assertEqual(result, SCRIPT.replace("letter", "<soft>letter</soft>"))

    def test_safe_subset_and_empty_suggestions_need_no_user_repair_warning(self):
        for edits in ([], [{"anchor": "missing", "tag": "soft"}, {"anchor": "letter", "tag": "soft"}]):
            with self.subTest(edits=edits), patch("services.speech_performance.record_speech_writer_warning") as warn:
                self.direct(Mock(return_value=(json.dumps({"edits": edits}), False)))
                warn.assert_not_called()

    def test_explicit_plain_requests_existing_word_tags_and_short_speech_skip(self):
        for brief in ("Keep it verbatim.", "No emotion tags.", "Plain speech please.", "Without delivery cues."):
            self.assertFalse(needs_fish_direction(SCRIPT, brief))
        self.assertFalse(needs_fish_direction(SCRIPT, instructions="Return untagged text."))
        self.assertFalse(needs_fish_direction("One sentence."))
        self.assertFalse(needs_fish_direction(SCRIPT.replace("letter", "<soft>letter</soft>")))
        self.assertTrue(needs_fish_direction("[calm] " + SCRIPT, "Write a spooky story."))

    def direct(self, generate, **overrides):
        return direct_fish_section(text=SCRIPT, generate=generate, remaining_calls=lambda: 1,
                                   max_new_tokens=1280, brief="Tell a spooky story.", **overrides)

    def test_one_bounded_call_uses_exact_source_and_returns_insertion(self):
        generate = Mock(return_value=(json.dumps({"edits": [{"anchor": "It was mine", "tag": "emphasis", "cue": "stunned"}]}), False))
        result = self.direct(generate)
        self.assertIn("[stunned] <emphasis>It was mine</emphasis>", result)
        generate.assert_called_once()
        args = generate.call_args.kwargs
        self.assertEqual(json.loads(args["prompt"])["script"], SCRIPT)
        self.assertLessEqual(args["max_new_tokens"], 1024)

    def test_truncated_or_failed_direction_preserves_original_and_warns(self):
        for response in (("{\"edits\":[", True), ("not JSON", False)):
            with self.subTest(response=response), patch("services.speech_performance.record_speech_writer_warning") as warn:
                self.assertEqual(self.direct(Mock(return_value=response)), SCRIPT)
                self.assertTrue(warn.called)
        with patch("services.speech_performance.record_speech_writer_warning") as warn:
            self.assertEqual(self.direct(Mock(side_effect=RuntimeError("offline"))), SCRIPT)
            self.assertTrue(warn.called)

    def test_incomplete_or_reserved_budget_never_triggers_direction_call(self):
        generate = Mock()
        self.assertEqual(self.direct(generate, section_incomplete=True), SCRIPT)
        with patch("services.speech_performance.record_speech_writer_warning"):
            result = direct_fish_section(text=SCRIPT, generate=generate, remaining_calls=lambda: 0,
                                         max_new_tokens=1024, brief="Story")
        self.assertEqual(result, SCRIPT)
        generate.assert_not_called()

    def test_cancel_propagates_without_retry_or_false_success(self):
        generate = Mock(side_effect=InterruptedError("cancelled"))
        with self.assertRaises(InterruptedError):
            self.direct(generate)
        generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
