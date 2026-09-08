"""Verify conditional guide routing with neutral fixtures and no live LLM calls."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.guide_loader import load_guide
from services.h3_story_ledger import plan_h3_story_segments


MARKER = "TEST SHARED CONTENT STYLE: Use a warm conversational tone."


def fixture_guide(category, name):
    return MARKER if name == "nsfw_shared" else load_guide(category, name)


class H3SharedContentGuideTests(unittest.TestCase):
    def test_frames_and_reference_both_styles_gate_every_planning_stage(self):
        for mode in ("sliding_window", "reference_sequence_continuation"):
            for style in ("faithful", "creative"):
                # Test turning the mode off after an enabled request in the same
                # process: a previously loaded guide must not leak into it.
                for enabled in (True, False):
                    with self.subTest(mode=mode, style=style, enabled=enabled):
                        prompt = (
                            'Maya says "Ready." Leo says "Let us begin."'
                            if style == "faithful" else
                            "Maya and Leo discuss how to organize their film project."
                        )
                        generator = Mock(side_effect=RuntimeError("No live LLM in this routing fixture"))
                        with patch("services.guide_loader.load_guide", side_effect=fixture_guide) as loader:
                            result = plan_h3_story_segments(
                                prompt, segment_durations=[10.1, 10.1], mode=mode,
                                camera_coverage="multi_shot", planning_style=style,
                                nsfw=enabled, llm_generate=generator,
                            )
                        self.assertEqual(len(result["segments"]), 2)
                        self.assertGreaterEqual(generator.call_count, 3)
                        names = [call.args[1] for call in loader.call_args_list]
                        self.assertIn("minimax_h3_story_treatment" if style == "faithful" else "minimax_h3_story_ledger", names)
                        self.assertIn("minimax_h3_story_segment", names)
                        self.assertEqual("nsfw_shared" in names, enabled)
                        for call in generator.call_args_list:
                            self.assertEqual(call.kwargs["system_prompt"].count(MARKER), int(enabled))
                            self.assertIn("json_schema", call.kwargs)
                        self.assertNotIn(MARKER, json.dumps(result))
                        if style == "creative":
                            self.assertTrue(any("COMPLETE SPARSE CREATIVE DIALOGUE" in call.kwargs["prompt"] for call in generator.call_args_list))

    def test_long_form_chapters_use_the_same_conditional_guide(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                generator = Mock(side_effect=RuntimeError("No live LLM in this routing fixture"))
                with patch("services.guide_loader.load_guide", side_effect=fixture_guide):
                    result = plan_h3_story_segments(
                        "A traveler crosses a strange world and finally reaches a distant city.",
                        segment_durations=[10.1] * 25, mode="sliding_window",
                        camera_coverage="multi_shot", planning_style="creative",
                        nsfw=enabled, llm_generate=generator,
                    )
                self.assertEqual(len(result["segments"]), 25)
                self.assertEqual(generator.call_count, 3)
                self.assertIn("chapters", generator.call_args_list[0].kwargs["json_schema"]["properties"])
                for call in generator.call_args_list[1:]:
                    self.assertIn("segments", call.kwargs["json_schema"]["properties"])
                for call in generator.call_args_list:
                    self.assertEqual(call.kwargs["system_prompt"].count(MARKER), int(enabled))

    def test_empty_shared_guide_preserves_original_stage_instructions(self):
        def empty_content(category, name):
            return "" if name == "nsfw_shared" else load_guide(category, name)

        generator = Mock(side_effect=RuntimeError("No live LLM in this routing fixture"))
        with patch("services.guide_loader.load_guide", side_effect=empty_content):
            result = plan_h3_story_segments(
                "A traveler walks toward the station.", segment_durations=[10.1, 10.1],
                mode="sliding_window", camera_coverage="multi_shot", nsfw=True,
                llm_generate=generator,
            )
        self.assertEqual(len(result["segments"]), 2)
        expected_guides = {
            load_guide("enhance", "minimax_h3_story_ledger"),
            load_guide("enhance", "minimax_h3_story_segment"),
        }
        for call in generator.call_args_list:
            self.assertIn(call.kwargs["system_prompt"], expected_guides)


if __name__ == "__main__":
    unittest.main()
