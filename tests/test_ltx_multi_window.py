import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.ltx_window_planner import (  # noqa: E402
    compute_ltx_window_count,
    deterministic_ltx_window_prompts,
    parse_ltx_window_prompts,
)


class LtxWindowPlannerTests(unittest.TestCase):
    def test_window_count_matches_wangp_stride(self):
        self.assertEqual(
            3,
            compute_ltx_window_count(
                600,
                241,
                overlap_frames=9,
                discard_frames=8,
            ),
        )
        self.assertEqual(
            1,
            compute_ltx_window_count(
                241,
                241,
                overlap_frames=9,
                discard_frames=8,
            ),
        )

    def test_invalid_stride_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than"):
            compute_ltx_window_count(
                300,
                17,
                overlap_frames=9,
                discard_frames=8,
            )

    def test_manual_lines_are_exact(self):
        self.assertEqual(
            ["first beat", "second beat", "final beat"],
            parse_ltx_window_prompts(
                "first beat\nsecond beat\nfinal beat",
                expected_count=3,
            ),
        )
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            parse_ltx_window_prompts(
                "first beat\nfinal beat",
                expected_count=3,
            )

    def test_ai_paragraphs_are_collapsed_to_one_line_per_window(self):
        result = parse_ltx_window_prompts(
            "First paragraph has\na wrapped line.\n\nSecond paragraph continues.",
            expected_count=2,
        )
        self.assertEqual(
            [
                "First paragraph has a wrapped line.",
                "Second paragraph continues.",
            ],
            result,
        )

    def test_fallback_advances_middle_and_final_windows(self):
        prompts = deterministic_ltx_window_prompts(
            "A woman enters. She finds a key. She opens the vault.",
            3,
        )
        self.assertEqual(3, len(prompts))
        self.assertIn("begin this beat", prompts[0])
        self.assertIn("advance only this middle beat", prompts[1])
        self.assertIn("complete the requested final beat", prompts[2])


class LtxMultiWindowWiringTests(unittest.TestCase):
    def test_every_ltx_video_handler_declares_sequence_controls(self):
        for relative in (
            "app/models/ltx_video/ltxv_handler.py",
            "app/models/ltx2/ltx2_handler.py",
            "app/models/ltx25/ltx25_handler.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('"multi_window_sequence_controls"', source, relative)

    def test_api_and_backend_publish_the_sequence_contract(self):
        launch = (APP / "launch.py").read_text(encoding="utf-8")
        self.assertIn('"multi_window_sequence_controls"', launch)
        self.assertIn('body.get("ltx_multi_window") is True', launch)
        self.assertIn('body["multi_prompts_gen_type"] = 1', launch)
        self.assertIn('response["ltx_window_plan"]', launch)

    def test_studio_exposes_toggle_modes_counts_and_manual_validation(self):
        controls = (
            ROOT
            / "ui/src/components/Sidebar/H3MultiWindowControls.tsx"
        ).read_text(encoding="utf-8")
        duration = (
            ROOT / "ui/src/components/Sidebar/DurationSlider.tsx"
        ).read_text(encoding="utf-8")
        prompt = (
            ROOT / "ui/src/components/Sidebar/PromptInput.tsx"
        ).read_text(encoding="utf-8")
        store = (ROOT / "ui/src/stores/useStore.ts").read_text(encoding="utf-8")
        self.assertIn("multi_window_sequence_controls", controls)
        self.assertIn("ltx_multi_window", controls)
        self.assertIn("ltx_window_prompt_mode", controls)
        self.assertIn("AI-planned window prompts", duration)
        self.assertIn("usesLtxManualPrompts", prompt)
        self.assertIn("Manual LTX sequence needs exactly", store)

    def test_ltx_guide_distributes_actions_chronologically(self):
        guide = (
            APP / "services/llm_guides/enhance/ltx2_video.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Distribute the requested story chronologically", guide)
        self.assertIn("never repeat", guide)
        self.assertIn("native audio", guide)


if __name__ == "__main__":
    unittest.main()
