"""Creative dialogue develops the brief while retaining exact scripts and timing."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.dialogue_writing import (
    conversation_brief, creative_dialogue_budget, creative_dialogue_expected,
    dialogue_forbidden, spoken_word_count,
)
from services.h3_story_ledger import (
    _canonicalize_story_ledger, _complete_creative_dialogue, _deterministic_ledger,
    _ledger_schema, extract_locked_dialogue, ledger_violations, plan_h3_story_segments,
)
from services.h3_window_planner import _creative_dialogue_expected
from services import llm_service


BRIEF = "Maya and Leo discuss how to organize their film project."
LINE_A = "Let's organize our source clips by scene first, then save a recipe so we can reuse these exact settings on tomorrow's shots."
LINE_B = "Good idea. I'll label the character references too, so neither of us has to guess which exact version belongs in each scene."


def line(text, segment=1, speaker="Maya"):
    return {"speaker": speaker, "language": "English", "delivery": "naturally", "text": text, "segment": segment}


def context_ir(*lines):
    return (
        "integrated_multimodal_description: [Shot 1] Maya and Leo sit at a desk in a sunlit room. "
        + " ".join(f"{'Maya' if index == 0 else 'Leo'} (S{index + 1}) says: <d>[English] {text}</d>" for index, text in enumerate(lines))
        + " From 0.00 to 0.25 seconds, all mouths remain closed with no voices. "
        "From 9.50 to 10.10 seconds, all mouths remain closed with no voices.\n"
        "overall_soundscape: Gentle computer fan and cloth movement.\nnon_diegetic_music: N/A"
    )


class CreativeDialogueWritingTests(unittest.TestCase):
    def test_duration_targets_use_actual_window_and_shared_speaker_budget(self):
        for duration, target, maximum in ((5.2, 12, 15), (10.1, 24, 30), (14.4, 34, 43)):
            with self.subTest(duration=duration):
                budget = creative_dialogue_budget(BRIEF, duration)
                self.assertEqual((budget.target, budget.maximum), (target, maximum))
                self.assertLessEqual(budget.target / 2.8, duration)
                self.assertIn("across all speakers", budget.instruction())
                self.assertIn("2.8 words per second", budget.instruction())
        action = creative_dialogue_budget("Maya warns Leo, then climbs the wall.", 14.4)
        self.assertLess(action.target, creative_dialogue_budget(BRIEF, 14.4).target)
        brief = creative_dialogue_budget("A space battle with short tactical dialogue.", 14.4)
        self.assertEqual(brief.minimum, 1)
        self.assertLess(brief.target, action.target)

    def test_tutorial_and_ordinary_character_names_imply_dialogue(self):
        for prompt in (
            "Blaine's tutorial about organizing a project.",
            "Maya and Leo confront each other about the missing film.",
            "Two friends meet at their usual table.",
            "A podcast discussion of camera placement.",
        ):
            with self.subTest(prompt=prompt):
                self.assertTrue(creative_dialogue_expected(prompt))
                self.assertTrue(_creative_dialogue_expected(prompt, 1))

    def test_local_silence_does_not_mute_a_later_conversation(self):
        for prompt in (
            "Start with a silent establishing shot of the landscape, then Maya and Leo discuss the plan.",
            "No dialogue until the opening shot ends; then Maya explains the recipe.",
            "Maya gives a tutorial. No voices outside the tagged speech intervals.",
        ):
            with self.subTest(prompt=prompt):
                self.assertFalse(dialogue_forbidden(prompt))
                self.assertTrue(conversation_brief(prompt))

    def test_silence_and_exact_only_requests_have_no_writing_quota(self):
        for prompt in (
            "Maya and Leo discuss the plan through gestures. No dialogue.",
            "An entirely silent film about friends who meet in a park.",
            'Maya explains the plan: "Ready?" Only these lines.',
            'Maya explains the plan: "Ready?" Use only these exact lines.',
            "A music-only tutorial montage.",
            "A landscape with moving clouds.",
            'A neon sign says "Tell me about your project" above an empty desk.',
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(creative_dialogue_budget(prompt, 14.4))

    def test_h3_quote_contract_is_mode_aware_and_times_complete_script(self):
        prompt = 'Maya explains the plan and says "Ready?"'
        faithful = llm_service._build_h3_dialogue_requirement(prompt, 14.4, "faithful")
        creative = llm_service._build_h3_dialogue_requirement(prompt, 14.4, "creative")
        exact_only = llm_service._build_h3_dialogue_requirement(prompt + " Only these lines.", 14.4, "creative")
        self.assertIn("Do not add speech", faithful)
        self.assertIn("Do not add speech", exact_only)
        self.assertIn("supporting dialogue", creative)
        self.assertIn("ALL authored spoken words", creative)
        self.assertIn("<d>[English] Ready?</d>", creative)

    def test_h3_single_window_retry_develops_short_dialogue(self):
        original = context_ir("Ready?")
        authored = context_ir(LINE_A)
        generator = Mock(return_value=authored)
        result = llm_service._complete_creative_enhancement(
            BRIEF, original, duration_seconds=10.1, structured=True, ref2va=False,
            system_prompt="Keep H3 fields.", user_prompt=BRIEF,
            generator=generator, max_new_tokens=1280, temperature=0.6,
        )
        self.assertIn(LINE_A, result)
        generator.assert_called_once()
        self.assertIn("24 spoken words", generator.call_args.kwargs["prompt"])

    def test_single_window_rejects_combined_overbudget_and_changed_quotes(self):
        prompt = BRIEF + ' Maya says "Ready?"'
        original = context_ir("Ready?")
        for replacement in (context_ir(LINE_A), context_ir("Ready?", LINE_A, LINE_B)):
            with self.subTest(replacement=replacement):
                result = llm_service._complete_creative_enhancement(
                    prompt, original, duration_seconds=10.1, structured=True, ref2va=False,
                    system_prompt="H3", user_prompt=prompt, generator=Mock(return_value=replacement),
                    max_new_tokens=1280, temperature=0.6,
                )
                self.assertEqual(result, original)

    def test_generic_enhancer_receives_creative_budget_including_raw_mode(self):
        user_prompt = llm_service._build_enhance_user_prompt(BRIEF, "video", 14.4, 1, 14.4, "ltx2", "creative")
        self.assertIn("34 spoken words", user_prompt)
        self.assertIn("CREATIVE WRITING", user_prompt)
        generator = Mock(return_value='A desk scene. Maya says "' + LINE_A + '"')
        result = llm_service._complete_creative_enhancement(
            BRIEF, 'Maya says "Ready?"', duration_seconds=10.1, structured=False, ref2va=False,
            system_prompt="Video prompt", user_prompt=user_prompt,
            generator=generator, max_new_tokens=1200, temperature=0.6,
        )
        self.assertIn(LINE_A, result)

    def test_valid_script_and_no_dialogue_do_not_trigger_retry(self):
        generator = Mock(side_effect=AssertionError("Must not call the LLM"))
        for prompt, original in ((BRIEF, context_ir(LINE_A)), (BRIEF + " No dialogue.", context_ir())):
            result = llm_service._complete_creative_enhancement(
                prompt, original, duration_seconds=10.1, structured=True, ref2va=False,
                system_prompt="H3", user_prompt=prompt, generator=generator, max_new_tokens=1280, temperature=0.6,
            )
            self.assertEqual(result, original)
        generator.assert_not_called()

    def test_real_enhance_pipeline_keeps_the_completed_h3_script(self):
        generator = Mock(side_effect=[context_ir("Ready?"), context_ir(LINE_A)])
        guide = (Path(__file__).resolve().parents[1] / "app/services/llm_guides/enhance/minimax_h3_video.md").read_text(encoding="utf-8")
        with patch.object(llm_service, "generate", generator), patch.object(llm_service, "_active_registry_entry", return_value={}), patch("services.enhance_guides.get_enhance_guide", return_value=guide):
            result = llm_service.enhance_prompt(
                BRIEF, mode="video", model_type="minimax_h3_fl2va", duration_seconds=10.1,
                planning_style="creative",
            )
        self.assertIn(LINE_A, result)
        self.assertEqual(generator.call_count, 2)
        self.assertIn("24 spoken words", generator.call_args_list[0].kwargs["system_prompt"])

    def make_ledger(self, prompt=BRIEF, durations=None):
        durations = durations or [10.1, 10.1]
        locked = extract_locked_dialogue(prompt)
        canonical = _deterministic_ledger(
            prompt, segment_count=len(durations), segment_durations=durations,
            locked_dialogue=locked, camera_coverage="multi_shot", reference_context="",
        )
        candidate = deepcopy(canonical)
        candidate["generated_dialogue"] = [line("Ready?", number) for number in range(1, len(durations) + 1)]
        ledger = _canonicalize_story_ledger(
            prompt, canonical, candidate, locked_dialogue=locked,
            segment_count=len(durations), allow_generated_dialogue=True,
        )
        return canonical, ledger, locked

    def test_sequence_completion_writes_each_window_and_keeps_story_events(self):
        canonical, ledger, locked = self.make_ledger()
        generator = Mock(side_effect=[
            json.dumps({"generated_dialogue": [line(LINE_A, 1)]}),
            json.dumps({"generated_dialogue": [line(LINE_B, 2, "Leo")]}),
        ])
        result, warnings = _complete_creative_dialogue(
            BRIEF, ledger, canonical_ledger=canonical, locked_dialogue=locked,
            durations=[10.1, 10.1], generate=generator, system_prompt="Story guide",
        )
        self.assertEqual(warnings, [])
        self.assertEqual([item["text"] for item in result["generated_dialogue"]], [LINE_A, LINE_B])
        self.assertEqual([beat["source_event_ids"] for beat in result["beats"]], [beat["source_event_ids"] for beat in ledger["beats"]])
        self.assertEqual(ledger_violations(BRIEF, result, segment_count=2, locked_dialogue=[], expect_dialogue=True, allow_generated_dialogue=True, segment_durations=[10.1, 10.1]), [])
        self.assertEqual([item["text"] for item in ledger["generated_dialogue"]], ["Ready?", "Ready?"])

    def test_sequence_completion_preserves_exact_anchor_and_its_event(self):
        prompt = BRIEF + ' Maya says "Ready?"'
        canonical, ledger, locked = self.make_ledger(prompt, [14.4])
        replacement_lines = [line(LINE_A), line("Yes, I'll start labeling the footage while you save those settings.", speaker="Leo")]
        generator = Mock(return_value=json.dumps({"generated_dialogue": replacement_lines}))
        result, warnings = _complete_creative_dialogue(
            prompt, ledger, canonical_ledger=canonical, locked_dialogue=locked,
            durations=[14.4], generate=generator, system_prompt="Story guide",
        )
        self.assertEqual(warnings, [])
        self.assertEqual(locked[0]["text"], "Ready?")
        before = [beat["source_event_ids"] for beat in ledger["beats"] if locked[0]["dialogue_id"] in beat["dialogue_ids"]]
        after = [beat["source_event_ids"] for beat in result["beats"] if locked[0]["dialogue_id"] in beat["dialogue_ids"]]
        self.assertEqual(before, after)
        self.assertEqual(len(result["generated_dialogue"]), 2)

    def test_invalid_completion_retains_previous_script_and_reports_shortfall(self):
        canonical, ledger, locked = self.make_ledger()
        for response in ("{}", json.dumps({"generated_dialogue": [line(LINE_A + " " + LINE_B, 1)]}), json.dumps({"generated_dialogue": [line(LINE_A, 99)]}), json.dumps({"generated_dialogue": [line(LINE_A, 1, "Unrequested narrator"), line(LINE_B, 2, "Leo")]})):
            with self.subTest(response=response):
                result, warnings = _complete_creative_dialogue(
                    BRIEF, ledger, canonical_ledger=canonical, locked_dialogue=locked,
                    durations=[10.1, 10.1], generate=Mock(return_value=response), system_prompt="Story guide",
                )
                self.assertEqual(result, ledger)
                self.assertTrue(warnings)

    def test_schema_allows_six_turns_per_window(self):
        schema = _ledger_schema(3, source_event_count=1, locked_dialogue_count=0, allow_generated_dialogue=True)
        self.assertEqual(schema["properties"]["generated_dialogue"]["maxItems"], 18)

    def test_faithful_and_exact_only_plans_do_not_call_creative_completion(self):
        for style, prompt in (("faithful", BRIEF), ("creative", BRIEF + ' Maya says "Ready?" Only these lines.')):
            with self.subTest(style=style), patch("services.h3_story_ledger._complete_creative_dialogue", side_effect=AssertionError("Should not author more dialogue")):
                plan = plan_h3_story_segments(
                    prompt, segment_durations=[10.1], mode="sliding_window", camera_coverage="multi_shot",
                    planning_style=style, llm_generate=Mock(side_effect=RuntimeError("offline")),
                )
                self.assertEqual(plan["ledger"]["generated_dialogue"], [])


if __name__ == "__main__":
    unittest.main()
