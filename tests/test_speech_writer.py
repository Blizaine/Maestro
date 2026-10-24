"""Bounded TTS speech writing uses fake generators and no model/GPU runtime."""
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service, speech_writer
from services.studio_enhancement import enhancement_context, enhancement_warnings


def fake_tts_prompt_modules():
    package = ModuleType("models.TTS")
    package.__path__ = []
    prompts = ModuleType("models.TTS.prompt_enhancers")
    prompts.TTS_MONOLOGUE_PROMPT = "Native monologue grammar: one plain spoken voice."
    prompts.TTS_QWEN3_DIALOGUE_PROMPT = (
        "Native dialogue grammar: exactly TWO characters, each line starts Peter: or Sarah:."
    )
    return patch.dict(sys.modules, {
        "models.TTS": package,
        "models.TTS.prompt_enhancers": prompts,
    })


class FakeGenerator:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class SpeechTargetTests(unittest.TestCase):
    def test_explicit_minutes_and_word_counts_are_inferred_and_capped(self):
        minutes = speech_writer.resolve_speech_target(
            "Write a story for about 12 minutes.", 0
        )
        self.assertEqual(minutes.minutes, 12)
        self.assertEqual(minutes.words, 1_560)
        self.assertTrue(minutes.is_long_form)

        words = speech_writer.resolve_speech_target(
            "Create exactly 1,500 words of dialogue.", 0
        )
        self.assertEqual(words.words, 1_500)
        self.assertEqual(words.source, "prompt_words")

        capped = speech_writer.resolve_speech_target(
            "Write a 75-minute dialogue.", 0
        )
        self.assertEqual(capped.minutes, 60)
        self.assertEqual(capped.words, 7_800)
        self.assertTrue(capped.capped)

    def test_english_runtime_numbers_and_composites_are_inferred(self):
        number_names = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen",
        ]
        single_digits = number_names[:9]
        for tens_word in ("twenty", "thirty", "forty", "fifty"):
            number_names.append(tens_word)
            number_names.extend(f"{tens_word}-{unit}" for unit in single_digits)
        number_names.append("sixty")
        for expected_minutes, number_name in enumerate(number_names, 1):
            target = speech_writer.resolve_speech_target(
                f"Write a {number_name}-minute story.", 0
            )
            self.assertEqual(target.minutes, expected_minutes, number_name)

        ten_minute_story = speech_writer.resolve_speech_target(
            "Write a ten-minute story about a lost compass.", 0
        )
        self.assertEqual(ten_minute_story.minutes, 10)
        self.assertEqual(ten_minute_story.words, 1_300)

        twelve_minutes_dialogue = speech_writer.resolve_speech_target(
            "Write twelve minutes of dialogue between two neighbors.", 0
        )
        self.assertEqual(twelve_minutes_dialogue.minutes, 12)

        hyphenated_composite = speech_writer.resolve_speech_target(
            "Write a twenty-one-minute conversation.", 0
        )
        self.assertEqual(hyphenated_composite.minutes, 21)

        spaced_composite = speech_writer.resolve_speech_target(
            "Write twenty one minutes of dialogue.", 0
        )
        self.assertEqual(spaced_composite.minutes, 21)

        maximum_worded_runtime = speech_writer.resolve_speech_target(
            "Write a sixty-minute monologue.", 0
        )
        self.assertEqual(maximum_worded_runtime.minutes, 60)

        unsupported_composite = speech_writer.resolve_speech_target(
            "Write a sixty-one-minute story.", 0
        )
        self.assertEqual(unsupported_composite.words, 0)

        invalid_composite = speech_writer.resolve_speech_target(
            "Write a twenty-eleven-minute story.", 0
        )
        self.assertEqual(invalid_composite.words, 0)

    def test_qualitative_long_request_defaults_to_ten_minutes_and_short_wins(self):
        long_target = speech_writer.resolve_speech_target(
            "Write a long dialogue about two sisters opening a bakery.", 0
        )
        self.assertEqual(long_target.minutes, 10)
        self.assertEqual(long_target.words, 1_300)
        self.assertEqual(long_target.source, "prompt_qualitative_long")

        explicit_short = speech_writer.resolve_speech_target(
            "Write a long story, but keep it short and brief.", 0
        )
        self.assertEqual(explicit_short.words, 0)

        explicit_minutes_win = speech_writer.resolve_speech_target(
            "Write a long story in about 2 minutes.", 0
        )
        self.assertEqual(explicit_minutes_win.minutes, 2)

        configured_minutes_win = speech_writer.resolve_speech_target(
            "Write a 20-minute story.", 2
        )
        self.assertEqual(configured_minutes_win.minutes, 2)

        narrative_time = speech_writer.resolve_speech_target(
            "Write a story about two travelers who waited ten minutes for a train.", 0
        )
        self.assertEqual(narrative_time.words, 0)

    def test_writer_instructions_are_bounded(self):
        shortened, did_truncate = speech_writer.bound_writer_instructions("x" * 8_100)
        self.assertEqual(len(shortened), 8_000)
        self.assertTrue(did_truncate)


class LongFormGenerationTests(unittest.TestCase):
    def setUp(self):
        self.target = speech_writer.resolve_speech_target("", 10)
        self.system = "Native guide: Speaker 1: / Speaker 2: with model tags."

    def run_writer(self, generator, target=None, truncated=None, max_new_tokens=2_048):
        return speech_writer.generate_long_form_script(
            generate=generator,
            prompt="Two sisters discover a letter in their bakery.",
            system_prompt=self.system,
            target=target or self.target,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            is_truncated=truncated,
        )

    def test_plan_and_sections_keep_native_grammar_and_continuity(self):
        generator = FakeGenerator([
            "Cast: Ada and Jo. Section 1: find the letter. Section 2: decide what to do.",
            "Ada: The letter was tucked beneath the flour tin.\nJo: Then let's read it together.",
            "Ada: We can keep the bakery open and honor her recipe.\nJo: And tomorrow, we begin again.",
        ])
        result = self.run_writer(generator)

        self.assertEqual(len(generator.calls), 3)
        self.assertIn("Ada: The letter", result)
        self.assertIn("tomorrow, we begin again", result)
        self.assertIn("Cast: Ada and Jo", generator.calls[1]["prompt"])
        self.assertIn("Recent script context", generator.calls[2]["prompt"])
        self.assertIn("Ada: The letter was tucked", generator.calls[2]["prompt"])
        self.assertIn("model's required native TTS format", generator.calls[1]["prompt"])
        self.assertTrue(all(call["system_prompt"] == self.system for call in generator.calls[1:]))
        self.assertTrue(all(call["max_new_tokens"] <= 1_280 for call in generator.calls))
        self.assertIn("Develop its assigned beat fully rather than summarizing it", generator.calls[1]["prompt"])

    def test_token_limited_section_gets_one_continuation_without_repeated_words(self):
        generator = FakeGenerator([
            "Cast: Mara and Len. The story ends with them restoring the bridge.",
            "Mara: I know the old road, even in the rain.",
            "I know the old road, even in the rain. But the bridge is still ahead.",
            "Len: We'll repair it together, one board at a time.",
        ])
        truncated_states = iter([False, True, False, False])
        result = self.run_writer(generator, truncated=lambda: next(truncated_states))

        self.assertEqual(len(generator.calls), 4)
        self.assertEqual(result.count("I know the old road"), 1)
        self.assertIn("But the bridge is still ahead", result)
        self.assertIn("Continue the same spoken script", generator.calls[2]["prompt"])

    def test_sixty_minute_limit_stays_under_call_and_output_budgets(self):
        target = speech_writer.resolve_speech_target("Write a 90-minute story.", 0)
        generator = FakeGenerator(["A complete plan."] + ["Speaker 1: A complete section." for _ in range(12)])
        result = self.run_writer(generator, target=target)

        self.assertEqual(target.minutes, 60)
        self.assertEqual(len(generator.calls), 13)
        self.assertLessEqual(len(generator.calls), speech_writer.MAX_GENERATION_CALLS)
        self.assertTrue(all(call["max_new_tokens"] <= 1_280 for call in generator.calls))
        self.assertTrue(result.startswith("Speaker 1:"))

    def test_section_calls_also_respect_a_lower_selected_output_budget(self):
        generator = FakeGenerator([
            "Cast: Two friends make up.",
            "Mina: I should have called you sooner.",
            "Luz: I'm glad you came back to talk.",
        ])
        self.run_writer(generator, max_new_tokens=64)

        self.assertEqual(len(generator.calls), 3)
        self.assertTrue(all(call["max_new_tokens"] <= 64 for call in generator.calls))

    def test_no_progress_continuation_and_empty_sections_are_warned_but_partial_text_survives(self):
        generator = FakeGenerator([
            "Cast: Inez and Sol. Section 1 is the departure; section 2 is the return.",
            "Inez: The train left before sunrise.",
            "Inez: The train left before sunrise.",  # duplicate continuation: no progress
            "",  # empty second section
        ])
        truncated_states = iter([False, True, False, False])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, truncated=lambda: next(truncated_states))
            warnings = enhancement_warnings()

        self.assertIn("Inez: The train left before sunrise", result)
        self.assertEqual(result.count("The train left before sunrise"), 1)
        self.assertTrue(any("made no progress" in warning for warning in warnings))
        self.assertTrue(any("section 2 returned no spoken text" in warning for warning in warnings))
        self.assertTrue(any("may be incomplete" in warning for warning in warnings))

    def test_model_error_returns_prior_draft_with_warning(self):
        generator = FakeGenerator([
            "Cast: Two friends reconcile.",
            "Nia: I kept your postcard.",
            RuntimeError("writer unavailable"),
        ])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator)
            warnings = enhancement_warnings()

        self.assertIn("Nia: I kept your postcard.", result)
        self.assertTrue(any("available script draft was retained" in warning for warning in warnings))
        self.assertTrue(any("native TTS syntax" in warning for warning in warnings))
        self.assertIn("Nia: I kept your postcard.", generator.calls[2]["prompt"])

    def test_failed_first_section_raises_instead_of_returning_blank_success(self):
        generator = FakeGenerator([
            "Cast: Two friends reconcile.",
            RuntimeError("writer unavailable"),
        ])
        with enhancement_context({}, lambda: False):
            with self.assertRaisesRegex(RuntimeError, "no spoken script text"):
                self.run_writer(generator)
            warnings = enhancement_warnings()

        self.assertTrue(any("stopped at section 1" in warning for warning in warnings))
        self.assertTrue(any("incomplete in native TTS syntax" in warning for warning in warnings))

    def test_planning_error_falls_back_to_brief_and_keeps_sections(self):
        generator = FakeGenerator([
            RuntimeError("planner unavailable"),
            "Inez: We should read the letter before we decide.",
            "Sol: Then let's make a promise to keep the bakery open.",
        ])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator)
            warnings = enhancement_warnings()

        self.assertIn("read the letter", result)
        self.assertIn("keep the bakery open", result)
        self.assertTrue(any("planning call failed" in warning for warning in warnings))

    def test_cancellation_propagates_without_becoming_a_partial_success(self):
        generator = FakeGenerator(["This response must never be used."])
        checks = iter([False, True])
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            with enhancement_context({}, lambda: next(checks)):
                self.run_writer(generator)

        self.assertEqual(generator.calls, [])

    def test_unrecovered_final_continuation_warns_about_native_syntax_and_ending(self):
        generator = FakeGenerator([
            "Cast: Mara and Len return home.",
            "Mara: The old bridge still held beneath our feet.",
            "Len: We should tell the others what we found.",
            "Len: We should tell the others what we found.",  # duplicate continuation
        ])
        truncated_states = iter([False, False, True, False])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, truncated=lambda: next(truncated_states))
            warnings = enhancement_warnings()

        self.assertIn("Mara: The old bridge still held", result)
        self.assertIn("Len: We should tell the others", result)
        self.assertTrue(any("native TTS syntax" in warning for warning in warnings))

    def test_empty_final_token_limited_section_warns_and_prior_text_survives(self):
        generator = FakeGenerator([
            "Cast: A complete story about two hikers.",
            "Nell: We found the trailhead just before sunset.",
            "",
        ])
        truncated_states = iter([False, False, True])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, truncated=lambda: next(truncated_states))
            warnings = enhancement_warnings()

        self.assertIn("We found the trailhead", result)
        self.assertTrue(any("hit the token limit before returning spoken text" in warning for warning in warnings))
        self.assertTrue(any("may be incomplete in native TTS syntax" in warning for warning in warnings))


class EnhancePromptIntegrationTests(unittest.TestCase):
    def test_speech_repeated_story_phrases_keep_the_complete_ending(self):
        story = (
            "[tense] I heard footsteps outside. I told myself I was imagining it. "
            "I told myself I was being foolish.\n"
            "[quiet] In the morning I opened the door and found the missing letter."
        )
        # This normal rhetorical repetition triggers the generic prose-loop
        # heuristic. Exercise the real speech routing, not just a cleanup flag.
        self.assertNotEqual(llm_service._clean_enhance_output(story), story)
        with fake_tts_prompt_modules(), \
                patch.object(llm_service, "generate", return_value=story) as writer:
            result = llm_service.enhance_prompt(
                "Rewrite the supplied complete story with expressive emotion.",
                mode="audio", tts_enhance_mode="monologue", max_new_tokens=2_048,
                tts_writer_instructions="A detailed first-person storyteller.",
            )
        writer.assert_called_once()
        self.assertEqual(result, story)

    def test_dialogue_keeps_repeated_refrains_and_later_turns(self):
        dialogue = (
            "Speaker 1: [insistent] We are going home. We are going home. We are going home.\n"
            "Speaker 2: [warm] Then let us take the last train together."
        )
        with fake_tts_prompt_modules(), \
                patch.object(llm_service, "generate", return_value=dialogue):
            result = llm_service.enhance_prompt(
                "Preserve this dialogue and add emotional cues.", mode="audio",
                tts_enhance_mode="dialogue", tts_voice_count=2, max_new_tokens=2_048,
            )
        self.assertEqual(result, dialogue)

    def test_style_stays_out_of_spoken_prompt_and_native_format_contract_wins(self):
        style = "Write with dry humor and an observant, warm personality."
        with fake_tts_prompt_modules(), \
                patch.object(llm_service, "generate", return_value="Ada: I brought the good biscuits.") as writer:
            result = llm_service.enhance_prompt(
                "Two friends meet at a station.",
                mode="audio",
                max_new_tokens=2_048,
                tts_enhance_mode="dialogue",
                tts_voice_count=2,
                tts_writer_instructions=style,
                raw_enhancer_mode=True,
            )

        self.assertIn("Ada: I brought the good biscuits", result)
        self.assertEqual(writer.call_count, 1)
        system = writer.call_args.kwargs["system_prompt"]
        spoken_prompt = writer.call_args.kwargs["prompt"]
        self.assertIn(style, system)
        self.assertNotIn(style, spoken_prompt)
        self.assertIn("Peter:", system)
        self.assertLess(system.index("USER-REQUESTED WRITER STYLE"), system.index("CRITICAL SPEECH SCRIPT OUTPUT CONTRACT"))
        self.assertIn("native language, speaker, voice, and tag conventions", system)

    def test_ordinary_short_tts_request_keeps_one_call_and_existing_token_budget(self):
        with fake_tts_prompt_modules(), \
                patch.object(llm_service, "generate", return_value="Mina: We should go before the rain.") as writer:
            result = llm_service.enhance_prompt(
                "Two friends decide to walk home.",
                mode="audio",
                max_new_tokens=2_048,
                tts_enhance_mode="dialogue",
                tts_voice_count=2,
            )

        self.assertIn("We should go before the rain", result)
        writer.assert_called_once()
        self.assertEqual(writer.call_args.kwargs["max_new_tokens"], 2_048)
        self.assertNotIn("USER-REQUESTED WRITER STYLE", writer.call_args.kwargs["system_prompt"])
        self.assertNotIn("CRITICAL SPEECH SCRIPT OUTPUT CONTRACT", writer.call_args.kwargs["system_prompt"])

    def test_target_only_control_bypasses_raw_enhancer_and_adds_size(self):
        with fake_tts_prompt_modules(), \
                patch.object(llm_service, "generate", return_value="Mina: The walk home took an hour.") as writer:
            result = llm_service.enhance_prompt(
                "Two friends decide to walk home.",
                mode="audio",
                max_new_tokens=512,
                tts_enhance_mode="monologue",
                tts_target_minutes=1,
                raw_enhancer_mode=True,
            )

        self.assertIn("The walk home", result)
        writer.assert_called_once()
        self.assertIn("Requested spoken-script size: about 130 words", writer.call_args.kwargs["prompt"])
        self.assertIn("CRITICAL SPEECH SCRIPT OUTPUT CONTRACT", writer.call_args.kwargs["system_prompt"])

    def test_short_tts_token_limit_keeps_draft_and_records_warning(self):
        with fake_tts_prompt_modules(), enhancement_context({}, lambda: False), \
                patch.object(llm_service, "_last_generation_metrics", {"truncated": True}), \
                patch.object(llm_service, "generate", return_value="Mina: The train doors opened.") as writer:
            result = llm_service.enhance_prompt(
                "A traveler boards the last train.",
                mode="audio",
                tts_enhance_mode="monologue",
                max_new_tokens=2_048,
            )
            warnings = enhancement_warnings()

        self.assertIn("The train doors opened", result)
        writer.assert_called_once()
        self.assertTrue(any("output token limit" in warning for warning in warnings))

    def test_three_speakers_keep_model_native_guide_and_exact_count(self):
        native = "Native speaker-reference grammar: Speaker 1{voice=...}: [cue] text"
        fake_wgp = ModuleType("wgp")
        fake_wgp.get_model_def = lambda _model_type: {
            "text_prompt_enhancer_instructions1": native,
            "text_prompt_enhancer_instructions": "Native monologue grammar.",
        }
        with fake_tts_prompt_modules(), patch.dict(sys.modules, {"wgp": fake_wgp}), \
                patch.object(llm_service, "generate", return_value="Speaker 1{voice=a}: Hello.") as writer:
            llm_service.enhance_prompt(
                "Three neighbors meet.",
                mode="audio",
                tts_enhance_mode="dialogue",
                tts_voice_count=3,
                model_type="native-three-voice-engine",
            )

        system = writer.call_args.kwargs["system_prompt"]
        self.assertIn(native, system)
        self.assertIn("EXACT SPEAKER COUNT: Write exactly 3 distinct speakers", system)
        self.assertIn("native format", system)
        self.assertNotIn("Every line must start with a character name", system)

    def test_long_prompt_uses_bounded_sections_and_passes_user_style_to_system(self):
        style = "A reflective storyteller with playful exchanges."
        replies = [
            "Cast: Uma and Kit. They find the map, argue over the route, and return home together.",
            "Uma: The map says the orchard used to touch the river.\nKit: Then we follow the old fence.",
            "Uma: We made it back before dark.\nKit: And brought home the story with us.",
        ]
        with fake_tts_prompt_modules(), patch.object(llm_service, "_last_generation_metrics", {}), \
                patch.object(llm_service, "generate", side_effect=replies) as writer:
            result = llm_service.enhance_prompt(
                "Write a long dialogue about two friends finding an old map.",
                mode="audio",
                tts_enhance_mode="dialogue",
                tts_voice_count=2,
                tts_writer_instructions=style,
                max_new_tokens=2_048,
                raw_enhancer_mode=True,
            )

        self.assertEqual(writer.call_count, 3)
        self.assertIn("Kit: Then we follow the old fence", result)
        self.assertIn(style, writer.call_args_list[1].kwargs["system_prompt"])
        self.assertIn("CRITICAL SPEECH SCRIPT OUTPUT CONTRACT", writer.call_args_list[1].kwargs["system_prompt"])
        self.assertIn("Output only the requested spoken-script portion", writer.call_args_list[1].kwargs["system_prompt"])
        self.assertTrue(all(call.kwargs["max_new_tokens"] <= 1_280 for call in writer.call_args_list))

    def test_other_enhancement_modes_ignore_speech_writer_controls(self):
        with patch.object(llm_service, "generate", return_value="A clear video prompt.") as writer:
            result = llm_service.enhance_prompt(
                "A quiet train station.",
                mode="video",
                tts_writer_instructions="This is audio-only guidance.",
                tts_target_minutes=10,
            )

        self.assertEqual(result, "A clear video prompt.")
        writer.assert_called_once()
        self.assertNotIn("audio-only guidance", writer.call_args.kwargs["system_prompt"])


if __name__ == "__main__":
    unittest.main()
