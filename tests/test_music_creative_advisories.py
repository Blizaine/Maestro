"""Subjective variety reviews cannot discard a valid music-video plan."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.music_review import repair_music_rows
from services.director.planners.music_video import MusicVideoPlanner


class CreativeAdvisoryTests(unittest.TestCase):
    def test_failed_optional_edit_preserves_valid_rows_and_previous_repairs(self):
        rows = [{"video_prompt": "Opening"}, {"video_prompt": "Middle"},
                {"video_prompt": "Ending"}]
        original = copy.deepcopy(rows)
        failures, accepted = [], []

        def replace(index, feedback, current):
            return [{"video_prompt": "Revised middle"}] if index == 1 else []

        result = repair_music_rows(
            rows, {1: "Vary the middle", 2: "Vary the ending"}, replace,
            lambda *args: None, on_failure=lambda i, reason: failures.append(i),
            on_repaired=accepted.append,
        )
        self.assertEqual(result[0], original[0])
        self.assertEqual(result[1]["video_prompt"], "Revised middle")
        self.assertEqual(result[2], original[2])
        self.assertEqual(accepted, [1])
        self.assertEqual(failures, [2])
        self.assertEqual(rows, original)

    def test_invalid_optional_edit_never_replaces_valid_source_action(self):
        original = [{"video_prompt": "The rider boards the boat.", "event": "M1"}]
        accepted, failures = [], []
        result = repair_music_rows(
            original, {0: "Try another treatment"},
            lambda *args: [{"video_prompt": "The boat departs.", "event": "M2"}],
            lambda index, row, previous: "source action changed" if row["event"] != "M1" else None,
            on_failure=lambda i, reason: failures.append(reason), on_repaired=accepted.append,
        )
        self.assertEqual(result, original)
        self.assertEqual(failures, ["source action changed"])
        self.assertEqual(accepted, [])

    def test_advisory_mode_propagates_cancellation_and_writer_failure(self):
        for error in (InterruptedError("cancelled"), RuntimeError("writer failed")):
            with self.subTest(error=type(error).__name__):
                def replace(*args):
                    raise error
                with self.assertRaises(type(error)):
                    repair_music_rows([{}], {0: "vary"}, replace, lambda *args: None,
                                      on_failure=lambda *args: self.fail("Writer error was swallowed"))

    def _planner(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"advisory": 1})
        planner._music_arc = [dict(clip_index=i, visual_beat=beat,
            purpose="Develop performance", environment="same room", camera="locked camera",
            source_event_ids=[], intentional_reprise=False, reprise_reason="")
            for i, beat in enumerate(["The dancer steps left.", "The dancer steps right."])]
        return planner

    def _finding(self, rows):
        return dict(clip_index=1, category="repeated_action", evidence=rows[1]["video_prompt"],
                    related_clip_index=0, related_evidence=rows[0]["video_prompt"],
                    repair="Vary the next gesture without changing the continuous dance.")

    def _review(self, planner, rows, repair):
        return planner._review_music_plan(rows, [{}, {}], ["", ""], "A continuous dance",
            row_validator=lambda *args: None, repair_row=repair, batch_size=3)

    def test_persistent_subjective_similarity_is_advisory_and_cached(self):
        planner = self._planner()
        rows = [{"video_prompt": "The dancer steps to the left beside the empty mirror."},
                {"video_prompt": "The dancer steps to the right beside the empty mirror."}]
        finding = self._finding(rows)
        with patch.object(planner, "_call_llm_json", return_value=[finding]) as writer:
            result = self._review(planner, rows, lambda i, feedback, current: [current[i]])
            cached = self._review(planner, result, lambda *args: self.fail("Repeated advisory repair"))
        self.assertEqual(writer.call_count, 2)
        self.assertEqual(result, rows)
        self.assertEqual(cached, rows)
        receipt = planner._planning_resume_checkpoint["music_creative_review"]
        self.assertTrue(any(note.get("severity") == "advisory" for note in receipt["review_notes"]))
        self.assertFalse(any(row["intentional_reprise"] for row in planner._music_arc))

    def test_failed_subjective_repair_keeps_validated_original_and_is_not_marked_repaired(self):
        planner = self._planner()
        rows = [{"video_prompt": "The dancer steps to the left beside the empty mirror."},
                {"video_prompt": "The dancer steps to the right beside the empty mirror."}]
        with patch.object(planner, "_call_llm_json", return_value=[self._finding(rows)]):
            result = self._review(planner, rows, lambda *args: [])
        self.assertEqual(result, rows)
        receipt = planner._planning_resume_checkpoint["music_creative_review"]
        self.assertEqual(receipt["repaired_clip_indices"], [])
        self.assertEqual(receipt["preserved_clip_indices"], [0, 1])

    def test_literal_native_prompt_recycling_stays_strict(self):
        quote = "The dancer circles the mirror twice and then raises both hands overhead."
        rows = [{"video_prompt": quote}, {"video_prompt": quote}]
        for unexplained_reprise in (False, True):
            with self.subTest(unexplained_reprise=unexplained_reprise):
                planner = self._planner()
                planner._music_arc[1]["intentional_reprise"] = unexplained_reprise
                with patch.object(planner, "_call_llm_json", return_value=[]):
                    with self.assertRaisesRegex(RuntimeError, "creative review still found issues"):
                        self._review(planner, rows, lambda i, feedback, current: [current[i]])


if __name__ == "__main__":
    unittest.main()
