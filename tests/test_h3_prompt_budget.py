"""Regressions for MiniMax H3 prompt measurement and safe compaction."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_prompt_budget import (  # noqa: E402
    H3_ENHANCED_TEXT_TOKEN_TARGET,
    H3_PROMPT_QUALITY_TARGET,
    fit_h3_base_prompt,
    h3_prompt_token_count,
)
from services.h3_window_planner import compile_h3_window_prompts  # noqa: E402
from services import llm_service  # noqa: E402


class H3PromptBudgetTests(unittest.TestCase):
    def setUp(self):
        # These tests exercise the conservative pre-download path so they are
        # deterministic on CI and on fresh installs without H3 tokenizer assets.
        self.tokenizer_patch = patch(
            "services.h3_prompt_budget._h3_plain_tokenizer",
            return_value=None,
        )
        self.tokenizer_patch.start()

    def tearDown(self):
        self.tokenizer_patch.stop()

    def test_short_structured_prompt_is_unchanged(self):
        prompt = (
            "integrated_multimodal_description: [Shot 1] Clark walks into frame.\n\n"
            "overall_soundscape: Quiet street ambience.\n\n"
            "non_diegetic_music: N/A"
        )
        result = fit_h3_base_prompt(prompt)
        self.assertFalse(result.compacted)
        self.assertEqual(result.prompt, prompt)
        self.assertEqual(result.token_count, h3_prompt_token_count(prompt))

    def test_compaction_preserves_alignment_dialogue_and_field_order(self):
        alignment = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
        dialogue = [
            "<d>[English] Come on! Let's go!</d>",
            "<d>[English] I don't know about this, Hermione.</d>",
        ]
        timed_clauses = " ".join(
            f"From {index:.2f} to {index + 1:.2f} seconds, [Shot {index + 1}] "
            "the handheld camera tracks the subjects through the detailed mountain "
            "landscape while their unchanged clothes and carried brooms remain visible."
            for index in range(4)
        )
        verbose_clauses = " ".join(
            "The expansive mountain environment contains richly described distant "
            "ridges, atmospheric haze, layered clouds, rocks, grasses, and sunlight."
            for _index in range(70)
        )
        prompt = (
            f"{alignment}\n\n"
            "integrated_multimodal_description: "
            f"{timed_clauses} {verbose_clauses} Hermione (S1) speaks clearly: {dialogue[0]} "
            f"Ron (S2) answers nervously: {dialogue[1]} "
            "The final frame shows them diving between the canyon walls.\n\n"
            "overall_soundscape: Continuous mountain wind, broom movement, rushing air, "
            "distant birds, fabric movement, and synchronized canyon echoes.\n\n"
            "non_diegetic_music: Epic orchestral adventure music that builds continuously."
        )

        self.assertGreater(h3_prompt_token_count(prompt), H3_ENHANCED_TEXT_TOKEN_TARGET)
        result = fit_h3_base_prompt(prompt)

        self.assertTrue(result.compacted)
        self.assertLessEqual(result.token_count, H3_ENHANCED_TEXT_TOKEN_TARGET)
        self.assertTrue(result.prompt.startswith(alignment))
        for block in dialogue:
            self.assertEqual(result.prompt.count(block), 1)
        self.assertLess(
            result.prompt.index("integrated_multimodal_description:"),
            result.prompt.index("overall_soundscape:"),
        )
        self.assertLess(
            result.prompt.index("overall_soundscape:"),
            result.prompt.index("non_diegetic_music:"),
        )
        self.assertIn("final frame", result.prompt.casefold())

    def test_dialogue_is_never_blindly_truncated(self):
        dialogue = "<d>[English] " + "verylongword " * 700 + "</d>"
        prompt = (
            f"integrated_multimodal_description: [Shot 1] Alex (S1) speaks: {dialogue}\n\n"
            "overall_soundscape: Quiet room tone.\n\n"
            "non_diegetic_music: N/A"
        )
        result = fit_h3_base_prompt(prompt, target_tokens=128)

        self.assertEqual(result.prompt.count(dialogue), 1)
        self.assertGreater(result.token_count, 128)

    def test_long_unstructured_prompt_is_preserved_instead_of_rejected(self):
        prompt = " ".join(f"word{index}" for index in range(600))
        self.assertGreater(h3_prompt_token_count(prompt), 512)

        result = fit_h3_base_prompt(prompt, target_tokens=480)

        self.assertEqual(result.prompt, prompt)
        self.assertFalse(result.compacted)
        self.assertGreater(result.token_count, 512)

    def test_structured_prompt_above_old_cutoff_is_not_compacted_unnecessarily(self):
        detail = " ".join(
            "The camera tracks the detailed room and preserves the requested action."
            for _index in range(34)
        )
        prompt = (
            "integrated_multimodal_description: [Shot 1] "
            f"{detail} The final frame shows the open doorway.\n\n"
            "overall_soundscape: Quiet room tone and footsteps.\n\n"
            "non_diegetic_music: N/A"
        )
        count = h3_prompt_token_count(prompt)
        self.assertGreater(count, 512)
        self.assertLessEqual(count, H3_PROMPT_QUALITY_TARGET)

        result = fit_h3_base_prompt(prompt)

        self.assertEqual(result.prompt, prompt)
        self.assertFalse(result.compacted)

    def test_window_compiler_records_individual_prompt_budget(self):
        boundaries = [{
            "index": 1,
            "start_frame": 0,
            "end_frame": 239,
            "start_seconds": 0.0,
            "end_seconds": 10.0,
        }]
        plan = {
            "subject_continuity": "Clark Kent in his familiar blue jacket",
            "setting_continuity": "Smallville main street at midday",
            "visual_continuity": "Live-action cinematic realism",
            "initial_state": "Clark stands beside Lana near the curb",
            "ambient_audio": "Light wind and distant traffic",
            "music": "N/A",
            "windows": [{
                "title": "The rescue",
                "opening_state": "Clark stands beside Lana near the curb",
                "coverage": "dynamic multi-shot coverage",
                "pacing": "fast real-time action",
                "shots": [{
                    "shot": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 10.0,
                    "transition": "opening composition",
                    "framing": "medium-wide shot",
                    "camera": "fast tracking camera",
                    "action": "Clark intercepts the runaway truck and stops it safely",
                    "dialogue": [],
                    "sound_effects": "tires skid and metal groans",
                }],
                "closing_state": "Clark holds the stopped truck while Lana watches",
            }],
        }

        compiled = compile_h3_window_prompts(plan, boundaries)

        self.assertEqual(len(compiled), 1)
        self.assertEqual(
            compiled[0]["prompt_quality_target"],
            H3_PROMPT_QUALITY_TARGET,
        )
        self.assertEqual(
            compiled[0]["prompt_tokens"],
            h3_prompt_token_count(compiled[0]["prompt"]),
        )
        self.assertLessEqual(
            compiled[0]["prompt_tokens"],
            H3_ENHANCED_TEXT_TOKEN_TARGET,
        )

    def test_studio_enhancer_budgets_finished_base_prompt(self):
        repeated_detail = " ".join(
            "The room contains richly described furniture, window reflections, textured "
            "walls, layered practical lighting, and small background objects."
            for _index in range(28)
        )
        verbose_result = (
            "integrated_multimodal_description: [Shot 1] From 0.00 to 8.00 seconds, "
            "a woman walks across the room and opens the far door. "
            f"{repeated_detail} The final frame shows the open doorway.\n\n"
            "overall_soundscape: Quiet room tone, footsteps, fabric movement, and the "
            "door latch opening. No human voice is audible.\n\n"
            "non_diegetic_music: N/A"
        )

        with (
            patch.object(llm_service, "generate", return_value=verbose_result),
            patch(
                "services.enhance_guides.get_enhance_guide",
                return_value="Return the required H3 Context-IR fields.",
            ),
        ):
            enhanced = llm_service.enhance_prompt(
                "A woman walks across a room and opens the far door.",
                mode="video",
                model_type="minimax_h3_fl2va_pruned",
                duration_seconds=8,
                max_new_tokens=1200,
            )

        self.assertLessEqual(
            h3_prompt_token_count(enhanced),
            H3_ENHANCED_TEXT_TOKEN_TARGET,
        )
        self.assertIn("opens the far door", enhanced)
        self.assertIn("final frame", enhanced.casefold())
        self.assertIn("overall_soundscape:", enhanced)
        self.assertIn("non_diegetic_music:", enhanced)


if __name__ == "__main__":
    unittest.main()
