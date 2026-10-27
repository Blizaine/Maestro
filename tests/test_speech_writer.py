"""Bounded TTS speech writing uses fake generators and no model/GPU runtime."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service, speech_writer
from services.fish_script_format import FishScriptFormat
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
        self.outputs = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        self.outputs.append(reply)
        if isinstance(reply, Exception):
            raise reply
        return reply


def spoken_fixture(topic, word_count, *, offset=0):
    """Create distinct, countable speech without invoking a model."""
    tokens = []
    detail = offset
    while len(tokens) < word_count:
        tokens.extend(
            f"{topic} remembers the old street while detail{detail} changes everything.".split()
        )
        detail += 1
    return "[steady] Speaker 1{voice=sample}: " + " ".join(tokens[:word_count])


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

    def test_qualitative_long_request_accepts_bounded_adjective_phrases(self):
        prompts = [
            "tell a long, interesting and spooky story for halloween. Be creative.",
            "Write a long and suspenseful story about two sisters.",
            "Write a long story about another story.",
            "Write a long ghost story.",
            "Write a long Halloween story.",
            "Write a long science fiction story.",
            "Write an extended—carefully paced and character-focused—dialogue.",
            "Write a full-length, richly detailed monologue.",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                target = speech_writer.resolve_speech_target(prompt, 0)
                self.assertEqual(target.minutes, 10)
                self.assertEqual(target.words, 1_300)
                self.assertEqual(target.source, "prompt_qualitative_long")

    def test_qualitative_long_request_rejects_noun_phrases_before_story(self):
        prompts = [
            "Tell me the long and flowing hair story of an old portrait.",
            "Tell me the long, empty hallway story of its caretaker.",
            "Tell me the long and fulfilling life story of a lighthouse keeper.",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(speech_writer.resolve_speech_target(prompt, 0).words, 0)

    def test_short_and_explicit_size_controls_still_override_qualitative_long(self):
        prompt = "Tell a long, interesting and spooky story"

        explicit_short = speech_writer.resolve_speech_target(
            f"{prompt}, but keep it brief.", 0
        )
        self.assertEqual(explicit_short.words, 0)

        explicit_minutes = speech_writer.resolve_speech_target(
            f"{prompt} for about 2 minutes.", 0
        )
        self.assertEqual(explicit_minutes.minutes, 2)
        self.assertEqual(explicit_minutes.source, "prompt_minutes")

        explicit_words = speech_writer.resolve_speech_target(
            f"Write exactly 500 words: {prompt}.", 0
        )
        self.assertEqual(explicit_words.words, 500)
        self.assertEqual(explicit_words.source, "prompt_words")

        configured_minutes = speech_writer.resolve_speech_target(prompt, 2)
        self.assertEqual(configured_minutes.minutes, 2)
        self.assertEqual(configured_minutes.source, "setting")

    def test_clip_duration_fills_default_after_explicit_size_and_short_requests(self):
        clip = speech_writer.resolve_speech_target(
            "Two friends prepare a meal together.",
            0,
            clip_duration_seconds=1_200,
        )
        self.assertEqual(clip.minutes, 20)
        self.assertEqual(clip.words, 2_600)
        self.assertEqual(clip.source, "clip_duration")

        qualitative_long = speech_writer.resolve_speech_target(
            "Write a long, interesting and spooky story.",
            0,
            clip_duration_seconds=1_200,
        )
        self.assertEqual(qualitative_long.minutes, 20)
        self.assertEqual(qualitative_long.words, 2_600)
        self.assertEqual(qualitative_long.source, "clip_duration")

        explicit_setting = speech_writer.resolve_speech_target(
            "Write exactly 500 words.",
            12,
            clip_duration_seconds=1_200,
        )
        self.assertEqual(explicit_setting.minutes, 12)
        self.assertEqual(explicit_setting.words, 1_560)
        self.assertEqual(explicit_setting.source, "setting")

        explicit_prompt = speech_writer.resolve_speech_target(
            "Write exactly 500 words.",
            0,
            clip_duration_seconds=1_200,
        )
        self.assertEqual(explicit_prompt.words, 500)
        self.assertEqual(explicit_prompt.source, "prompt_words")

        explicit_short = speech_writer.resolve_speech_target(
            "Keep this speech brief.",
            0,
            clip_duration_seconds=1_200,
        )
        self.assertEqual(explicit_short, speech_writer.SpeechTarget())

    def test_invalid_clip_durations_do_not_create_a_target(self):
        for duration in (None, 0, -1, float("nan"), float("inf"), "invalid", True):
            with self.subTest(duration=duration):
                target = speech_writer.resolve_speech_target(
                    "Two friends prepare a meal together.",
                    0,
                    clip_duration_seconds=duration,
                )
                self.assertEqual(target, speech_writer.SpeechTarget())

        qualitative_long = speech_writer.resolve_speech_target(
            "Write a long story about two friends.",
            0,
            clip_duration_seconds=float("nan"),
        )
        self.assertEqual(qualitative_long.minutes, 10)
        self.assertEqual(qualitative_long.source, "prompt_qualitative_long")

    def test_clip_duration_uses_the_existing_sixty_minute_cap(self):
        target = speech_writer.resolve_speech_target(
            "A generic spoken script.",
            0,
            clip_duration_seconds=7_200,
        )
        self.assertEqual(target.minutes, 60)
        self.assertEqual(target.words, 7_800)
        self.assertEqual(target.source, "clip_duration")
        self.assertTrue(target.capped)

    def test_writer_instructions_are_bounded(self):
        shortened, did_truncate = speech_writer.bound_writer_instructions("x" * 8_100)
        self.assertEqual(len(shortened), 8_000)
        self.assertTrue(did_truncate)


class ScriptContextTests(unittest.TestCase):
    def test_orphan_closing_cue_is_flagged_without_rewriting_text(self):
        script = "[steady] I found the letter. A voice echoed] behind the door."
        self.assertTrue(speech_writer._has_unclosed_native_syntax(script))
        self.assertFalse(speech_writer._has_unclosed_native_syntax("[steady] I found <emphasis>the letter</emphasis>."))
        self.assertEqual(script, "[steady] I found the letter. A voice echoed] behind the door.")

    def test_prefers_complete_recent_paragraph_and_preserves_partial_ending(self):
        recent = "[calm] The next sound is [my breath"
        value = "Old context that will not fit. " * 10 + "\n\n" + recent
        self.assertEqual(speech_writer._bounded_script_context(value, len(recent) + 4), recent)
        self.assertEqual(speech_writer._bounded_script_context(recent), recent)

    def test_never_starts_inside_a_spoken_word_or_native_direction(self):
        cases = (
            ("heavy binoculars guide us home.", "eavy", "binoculars guide us home."),
            ("[hands steady while I reach across the desk] The message is clear.", "while", "The message is clear."),
            ("<emphasis>the light holds steady</emphasis> We can go home.", "light", "We can go home."),
            ("<|phoneme_start|>first second<|phoneme_end|> Stay here.", "second", "Stay here."),
        )
        for tail, clipped_start, expected in cases:
            value = "Old context. " * 10 + tail
            limit = len(value) - value.index(clipped_start)
            with self.subTest(tail=tail):
                actual = speech_writer._bounded_script_context(value, limit)
                self.assertEqual(actual, expected)
                self.assertLessEqual(len(actual), limit)
                self.assertTrue(value.endswith(actual))

    def test_omits_an_unsafe_oversized_scope_without_altering_script(self):
        value = "Before [" + "a long unfinished direction " * 30
        original = value
        self.assertEqual(speech_writer._bounded_script_context(value, 60), "")
        self.assertEqual(value, original)
        self.assertEqual(speech_writer._bounded_script_context(value, 0), "")

    def test_clipped_speaker_label_does_not_orphan_the_first_dialogue_turn(self):
        for label in ("Speaker 1:", "Mara:"):
            with self.subTest(label=label):
                value = "Earlier context. " * 10 + "\n" + label + " This turn is too long for the excerpt."
                floor = value.index(label) + 2
                self.assertEqual(speech_writer._bounded_script_context(value, len(value) - floor), "")
                next_turn = "\nSpeaker 2: We can go home."
                value += next_turn
                self.assertEqual(speech_writer._bounded_script_context(value, len(value) - floor), next_turn.lstrip())


class LongFormGenerationTests(unittest.TestCase):
    def setUp(self):
        self.target = speech_writer.resolve_speech_target("", 10)
        self.system = "Native guide: Speaker 1: / Speaker 2: with model tags."

    def run_writer(
        self, generator, target=None, truncated=None, max_new_tokens=2_048,
        *, coverage_repairs=False, finalize_section=None, script_format=None,
    ):
        # Most legacy fixtures are short on purpose and cover call sequencing.
        # Dedicated coverage tests below exercise the production repair threshold.
        minimum_coverage = speech_writer.MIN_SECTION_COVERAGE if coverage_repairs else 0.0
        with patch.object(speech_writer, "MIN_SECTION_COVERAGE", minimum_coverage):
            return speech_writer.generate_long_form_script(
                generate=generator,
                prompt="Two sisters discover a letter in their bakery.",
                system_prompt=self.system,
                target=target or self.target,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                is_truncated=truncated,
                finalize_section=finalize_section,
                script_format=script_format,
            )

    def test_spoken_word_count_excludes_directions_tags_and_speaker_labels(self):
        script = (
            "[slowly, gently] Speaker 1{voice=sample}: Wait <whisper>for me</whisper>, "
            "Mara! [sighing] Old friend."
        )

        self.assertEqual(speech_writer._spoken_word_count(script), 6)

    def test_duplicate_detector_ignores_cue_changes_and_short_refrains(self):
        first = spoken_fixture("Mara", 600)
        duplicate = first.replace("[steady]", "[urgent]").replace(
            "Speaker 1{voice=sample}:", "Speaker 2:"
        )
        refrain = "[insistent] Speaker 1: We go home. We go home. " * 30
        mostly_new = spoken_fixture("Mara", 120) + " " + spoken_fixture(
            "Len", 480, offset=500
        )

        self.assertTrue(speech_writer._has_substantial_duplicate(first, duplicate))
        self.assertFalse(speech_writer._has_substantial_duplicate(refrain, refrain))
        self.assertFalse(speech_writer._has_substantial_duplicate(first, mostly_new))

    def test_duplicate_section_gets_one_focused_replacement_and_raw_draft_is_kept(self):
        first = spoken_fixture("Mara", 600)
        duplicate = first.replace("[steady]", "[tense]")
        replacement = spoken_fixture("Len", 600)
        generator = FakeGenerator([
            "Plan: Mara leaves the bakery; Len helps her finish the repairs.",
            first,
            duplicate,
            replacement,
        ])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, coverage_repairs=True)
            warnings = enhancement_warnings()

        self.assertEqual(len(generator.calls), 4)
        self.assertEqual(result, f"{first}\n{replacement}")
        self.assertEqual(generator.outputs[2], duplicate)
        repair_prompt = generator.calls[3]["prompt"]
        # Keep the exact recent wording/ending without copying a raw tail that
        # starts inside "remembers" at this fixture's character bound.
        self.assertIn("Mara remembers the old street while detail12", repair_prompt)
        self.assertIn(duplicate[-90:], repair_prompt)
        self.assertNotIn("\nbers the old street", repair_prompt)
        self.assertIn("already part of the script", generator.calls[2]["prompt"])
        self.assertTrue(any("remains short" in warning for warning in warnings))

    def test_finished_short_final_section_keeps_its_ending_without_expansion(self):
        first = spoken_fixture("Mara", 600)
        final_excerpt = spoken_fixture("Len", 100)
        generator = FakeGenerator([
            "Plan: The friends reopen the bakery together.",
            first,
            final_excerpt,
        ])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, coverage_repairs=True)
            warnings = enhancement_warnings()

        self.assertEqual(len(generator.calls), 3)
        self.assertEqual(result, f"{first}\n{final_excerpt}")
        self.assertTrue(any("without a length-only expansion" in warning for warning in warnings))
        self.assertTrue(any("remains short" in warning for warning in warnings))

    def test_cue_only_nonfinal_expansion_does_not_count_as_spoken_progress(self):
        first = spoken_fixture("Mara", 100)
        final_excerpt = spoken_fixture("Len", 100, offset=30)
        generator = FakeGenerator([
            "Plan: The friends resolve their disagreement and reopen the bakery.",
            first,
            "[warmly]",
            final_excerpt,
        ])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, coverage_repairs=True)
            warnings = enhancement_warnings()

        self.assertEqual(result, f"{first}\n{final_excerpt}")
        self.assertTrue(any("focused expansion made no distinct progress" in warning for warning in warnings))

    def test_token_limited_repair_is_continued_until_it_finishes(self):
        first = spoken_fixture("Mara", 100)
        final_section = spoken_fixture("Len", 600)
        partial_expansion = "Mara: The door opened and the old house breathed in"
        generator = FakeGenerator([
            "Plan: The friends leave the house together.",
            first,
            partial_expansion,
            "the last of the daylight. We ran for the gate, and reached it before the lock turned.",
            final_section,
        ])
        truncated_states = iter([False, False, True, False, False])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(
                generator,
                truncated=lambda: next(truncated_states),
                coverage_repairs=True,
            )
            warnings = enhancement_warnings()

        self.assertEqual(len(generator.calls), 5)
        self.assertIn(partial_expansion, result)
        self.assertIn("the last of the daylight", result)
        self.assertIn("Continue the same spoken script", generator.calls[3]["prompt"])
        self.assertFalse(any("may be incomplete" in warning for warning in warnings))

    def test_unfinished_expansion_is_not_appended_after_a_complete_section(self):
        first = spoken_fixture("Mara", 100)
        partial_expansion = "Len: We opened the gate and saw the road ahead"
        final_section = spoken_fixture("Len", 600)
        generator = FakeGenerator([
            "Plan: The friends make it home.",
            first,
            partial_expansion,
            partial_expansion,
            final_section,
        ])
        truncated_states = iter([False, False, True, False, False])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(
                generator,
                truncated=lambda: next(truncated_states),
                coverage_repairs=True,
            )
            warnings = enhancement_warnings()

        self.assertEqual(result, f"{first}\n{final_section}")
        self.assertNotIn(partial_expansion, result)
        self.assertTrue(any("focused expansion and its continuation could not finish" in warning for warning in warnings))

    def test_multi_part_token_limited_repair_stays_inside_the_global_call_cap(self):
        first = spoken_fixture("Mara", 100)
        final_section = spoken_fixture("Len", 600)
        generator = FakeGenerator([
            "Plan: The friends find a way out.",
            first,
            "Mara: The passage narrowed around us",
            "until the stone touched our shoulders. We kept moving",
            "and found the morning beyond the gate.",
            final_section,
        ])
        truncated_states = iter([False, False, True, True, False, False])
        result = self.run_writer(
            generator,
            truncated=lambda: next(truncated_states),
            coverage_repairs=True,
        )

        self.assertEqual(len(generator.calls), 6)
        self.assertIn("stone touched our shoulders", result)
        self.assertIn("found the morning beyond the gate", result)
        self.assertLessEqual(len(generator.calls), speech_writer.MAX_GENERATION_CALLS)

    def test_finalizer_uses_shared_call_budget_and_reserves_future_finishing(self):
        first = spoken_fixture("Mara", 600)
        final_section = spoken_fixture("Len", 100)
        generator = FakeGenerator([
            "Plan: The friends repair the bakery.",
            first,
            "[warmly]",
            final_section,
            "[quietly]",
        ])
        observations = []

        def finalize(**kwargs):
            observations.append({
                "text": kwargs["text"],
                "remaining_calls": kwargs["remaining_calls"](),
                "reserve_calls": kwargs["reserve_calls"],
                "section_incomplete": kwargs["section_incomplete"],
                "max_new_tokens": kwargs["max_new_tokens"],
            })
            addition, was_truncated = kwargs["generate"](
                prompt="Add supported performance cues.",
                system_prompt=self.system,
                max_new_tokens=kwargs["max_new_tokens"],
                temperature=0.35,
            )
            self.assertFalse(was_truncated)
            return kwargs["text"] + "\n" + addition

        result = self.run_writer(generator, finalize_section=finalize)

        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0]["reserve_calls"], 3)
        self.assertGreater(observations[0]["remaining_calls"], 0)
        self.assertEqual(observations[1]["reserve_calls"], 0)
        self.assertFalse(observations[0]["section_incomplete"])
        self.assertLessEqual(observations[0]["max_new_tokens"], 1_024)
        self.assertIn("[warmly]", result)
        self.assertIn("[quietly]", result)
        self.assertEqual(len(generator.calls), 5)
        self.assertEqual(generator.calls[2]["temperature"], 0.35)

    def test_finalizer_exception_retains_section_and_cancellation_propagates(self):
        script = spoken_fixture("Mara", 300)
        generator = FakeGenerator(["Plan: The bakery opens.", script])

        def fail_finalization(**_kwargs):
            raise RuntimeError("annotation unavailable")

        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, finalize_section=fail_finalization)
            warnings = enhancement_warnings()
        self.assertEqual(result, script)
        self.assertTrue(any("finalization failed" in warning for warning in warnings))

        cancelled_generator = FakeGenerator(["Plan: The bakery opens.", script])
        with enhancement_context({}, lambda: False):
            with self.assertRaisesRegex(InterruptedError, "cancelled"):
                self.run_writer(
                    cancelled_generator,
                    finalize_section=lambda **_kwargs: (_ for _ in ()).throw(
                        InterruptedError("cancelled")
                    ),
                )

    def test_duplicate_repair_failure_retains_raw_section_and_warns(self):
        first = spoken_fixture("Mara", 600)
        duplicate = spoken_fixture("Mara", 600, offset=0)
        generator = FakeGenerator([
            "Plan: They leave the bakery together.",
            first,
            duplicate,
            duplicate,
        ])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, coverage_repairs=True)
            warnings = enhancement_warnings()

        self.assertEqual(result, f"{first}\n{duplicate}")
        self.assertIn(duplicate, generator.outputs)
        self.assertTrue(any("after one focused revision" in warning for warning in warnings))
        self.assertTrue(any("distinct spoken words" in warning for warning in warnings))

    def test_cue_only_duplicate_repair_cannot_replace_spoken_section(self):
        first = spoken_fixture("Mara", 600)
        duplicate = spoken_fixture("Mara", 600)
        generator = FakeGenerator([
            "Plan: They leave the bakery together.",
            first,
            duplicate,
            "[urgent]",
        ])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, coverage_repairs=True)
            warnings = enhancement_warnings()

        self.assertEqual(result, f"{first}\n{duplicate}")
        self.assertNotIn("[urgent]", result)
        self.assertTrue(any("still substantially repeats" in warning for warning in warnings))

    def test_empty_sections_do_not_shift_duplicate_coverage_indexes(self):
        first = spoken_fixture("Mara", 500)
        duplicate = spoken_fixture("Mara", 500)
        target = speech_writer.SpeechTarget(
            minutes=2_000 / speech_writer.WORDS_PER_MINUTE,
            words=2_000,
            source="test",
        )
        generator = FakeGenerator([
            "Plan: Mara leaves, returns, and reopens the bakery.",
            first,
            "",
            duplicate,
            duplicate,
            "",
        ])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(
                generator,
                target=target,
                coverage_repairs=True,
            )
            warnings = enhancement_warnings()

        self.assertEqual(len(generator.calls), 6)
        self.assertEqual(result, f"{first}\n{duplicate}")
        self.assertTrue(any(
            "approximately 500 distinct spoken words against a target of 2000" in warning
            for warning in warnings
        ))

    def test_continuations_and_repairs_share_the_thirty_call_cap(self):
        generator = FakeGenerator(["Plan: A long journey."] + ["Mara: The road continued."] * 100)
        target = speech_writer.resolve_speech_target("Write a 60-minute story.", 0)

        with enhancement_context({}, lambda: False):
            result = self.run_writer(
                generator,
                target=target,
                truncated=lambda: True,
                coverage_repairs=True,
            )
            warnings = enhancement_warnings()

        self.assertTrue(result)
        self.assertEqual(len(generator.calls), speech_writer.MAX_GENERATION_CALLS)
        self.assertTrue(any("30-call limit" in warning for warning in warnings))

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
        self.assertIn("Recent model-authored script context", generator.calls[2]["prompt"])
        self.assertIn("Ada: The letter was tucked", generator.calls[2]["prompt"])
        self.assertIn("model's required native TTS format", generator.calls[1]["prompt"])
        self.assertIn("LONG-FORM CONTEXT RULE", generator.calls[1]["system_prompt"])
        self.assertIn("baseline register stable across sections", generator.calls[1]["system_prompt"])
        self.assertIn("Let emotion, pace, emphasis, and intensity follow the meaning", generator.calls[1]["system_prompt"])
        self.assertIn("Honor requests for plain", generator.calls[1]["system_prompt"])
        self.assertNotIn("only when the user brief calls for a performance shift", generator.calls[1]["system_prompt"])
        self.assertTrue(all(call["system_prompt"].startswith(self.system) for call in generator.calls[1:]))
        self.assertTrue(all(call["max_new_tokens"] <= 1_280 for call in generator.calls))
        self.assertIn("Develop its assigned beat fully rather than summarizing it", generator.calls[1]["prompt"])

    def test_overlong_draft_warns_without_truncating_spoken_text(self):
        first = " ".join(f"first{i}" for i in range(900))
        second = " ".join(f"second{i}" for i in range(900))
        generator = FakeGenerator(["Plan: departure, then return.", first, second])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, coverage_repairs=True)
            warnings = enhancement_warnings()

        self.assertEqual(result, first + "\n" + second)
        self.assertEqual(len(generator.calls), 3)
        self.assertTrue(any(
            "approximately 1800 spoken words against a target of 1300" in warning
            and "substantially longer" in warning for warning in warnings
        ))

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

    def test_token_limited_partial_angle_tag_joins_inside_the_tag_name(self):
        generator = FakeGenerator([
            "Cast: Mara returns to the bridge.",
            "Mara: Wait here <whis",
            "per> until I return.</whisper>",
            "Len: I will watch the road until you come back.",
        ])
        truncated_states = iter([False, True, False, False])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, truncated=lambda: next(truncated_states))
            warnings = enhancement_warnings()

        self.assertIn("Wait here <whisper> until I return.</whisper>", result)
        self.assertFalse(any("unclosed native TTS tag or cue" in warning for warning in warnings))
        self.assertFalse(any("may be incomplete in native TTS syntax" in warning for warning in warnings))

    def test_token_limited_partial_bracket_cue_preserves_its_leading_space(self):
        generator = FakeGenerator([
            "Cast: Mara returns to the bridge.",
            "Mara: Stay close [voice starts",
            " to shake] until I return.",
            "Len: I will watch the road until you come back.",
        ])
        truncated_states = iter([False, True, False, False])
        result = self.run_writer(generator, truncated=lambda: next(truncated_states))

        self.assertIn("[voice starts to shake]", result)
        self.assertNotIn("[voice startsto shake]", result)

    def test_fish_normalization_runs_before_finalizer_and_syntax_review(self):
        target = speech_writer.SpeechTarget(minutes=0.01, words=1)
        generator = FakeGenerator([
            "Plan: Mara finds a letter.",
            "Mara: <screams, abruptly> Wait! <whispers, briefly> Stay close.",
        ])
        reviewed = []

        def finalize(**kwargs):
            reviewed.append(kwargs["text"])
            self.assertFalse(kwargs["section_incomplete"])
            return kwargs["text"]

        with enhancement_context({}, lambda: False):
            result = self.run_writer(
                generator,
                target=target,
                finalize_section=finalize,
                script_format=FishScriptFormat(),
            )
            warnings = enhancement_warnings()

        expected = "Mara: [screams, abruptly] Wait! [whispers, briefly] Stay close."
        self.assertEqual(result, expected)
        self.assertEqual(reviewed, [expected])
        self.assertFalse(any("unclosed native TTS syntax" in warning for warning in warnings))

    def test_fish_focused_completion_closes_old_cue_before_new_native_directive(self):
        target = speech_writer.SpeechTarget(minutes=0.01, words=1)
        generator = FakeGenerator([
            "Plan: Mara finds a letter.",
            "Mara: Listen. [my breath catches",
            "<forceful>Wait!</forceful> We move.",
        ])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(
                generator,
                target=target,
                script_format=FishScriptFormat(),
            )
            warnings = enhancement_warnings()

        self.assertIn("[my breath catches]", result)
        self.assertIn("<forceful>Wait!</forceful> We move.", result)
        self.assertEqual(len(generator.calls), 3)
        self.assertFalse(any("retains an unclosed native TTS" in warning for warning in warnings))

    def test_fish_finalizer_cannot_hide_unresolved_unknown_markup(self):
        target = speech_writer.SpeechTarget(minutes=0.01, words=1)
        generator = FakeGenerator([
            "Plan: Mara finds a letter.",
            "Mara: Listen <custom>closely.",
        ])
        observed_incomplete = []

        def finalize(**kwargs):
            observed_incomplete.append(kwargs["section_incomplete"])
            return kwargs["text"] + "</custom>"

        with patch.object(speech_writer, "MAX_GENERATION_CALLS", 2):
            with enhancement_context({}, lambda: False):
                result = self.run_writer(
                    generator,
                    target=target,
                    finalize_section=finalize,
                    script_format=FishScriptFormat(),
                )
                warnings = enhancement_warnings()

        self.assertEqual(observed_incomplete, [True])
        self.assertIn("Mara: Listen <custom>closely.", result)
        self.assertNotIn("</custom>", result)
        self.assertTrue(any("finalization changed an incomplete draft" in warning for warning in warnings))

    def test_plain_prose_continuation_uses_a_safe_break_and_warns_about_word_seam(self):
        generator = FakeGenerator([
            "Cast: Mara returns to the bridge.",
            "Mara: The old",
            "door opened before dawn.",
            "Len: I will watch the road until you come back.",
        ])
        truncated_states = iter([False, True, False, False])
        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, truncated=lambda: next(truncated_states))
            warnings = enhancement_warnings()

        self.assertIn("The old\ndoor opened", result)
        self.assertNotIn("The olddoor", result)
        self.assertTrue(any("intended word boundary is uncertain" in warning for warning in warnings))

    def test_unclosed_paired_tag_is_reported_to_finalizer_and_retained(self):
        generator = FakeGenerator([
            "Cast: Mara and Len return to the bridge.",
            "Mara: Listen <whisper>stay close.",
            "please follow me.",  # does not close the unbalanced native tag
            "Len: I will watch the road until you come back.",
        ])
        incomplete_flags = []

        def finalize(**kwargs):
            incomplete_flags.append(kwargs["section_incomplete"])
            return kwargs["text"]

        with enhancement_context({}, lambda: False):
            result = self.run_writer(generator, finalize_section=finalize)
            warnings = enhancement_warnings()

        self.assertIn("Mara: Listen <whisper>stay close.", result)
        self.assertEqual(incomplete_flags, [True, False])
        self.assertTrue(any("retains an unclosed native TTS tag or cue" in warning for warning in warnings))

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
            "",  # incomplete-section repair: no progress
            "",  # empty second section
        ])
        truncated_states = iter([False, True, False, False, False])
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
        self.assertIn("CRITICAL SPEECH SCRIPT OUTPUT CONTRACT", writer.call_args.kwargs["system_prompt"])
        self.assertIn("Supported native performance controls belong in the script", writer.call_args.kwargs["system_prompt"])

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
        with fake_tts_prompt_modules(), patch.object(speech_writer, "MIN_SECTION_COVERAGE", 0.0), \
                patch.object(llm_service, "_last_generation_metrics", {}), \
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

    def test_modified_halloween_story_request_uses_bounded_long_form_path(self):
        replies = [
            "Plan: The narrator finds a lantern, hears a voice upstairs, and discovers why the house is haunted.",
            "At dusk, I found a lantern waiting on the porch. A voice upstairs whispered my name before I reached the door.",
            "By morning, the house stood silent beneath the autumn sky. I kept the lantern, knowing the voice would return.",
        ]
        with fake_tts_prompt_modules(), patch.object(speech_writer, "MIN_SECTION_COVERAGE", 0.0), \
                patch.object(llm_service, "_last_generation_metrics", {}), \
                patch.object(llm_service, "generate", side_effect=replies) as writer:
            result = llm_service.enhance_prompt(
                "tell a long, interesting and spooky story for halloween. Be creative.",
                mode="audio",
                tts_enhance_mode="monologue",
                max_new_tokens=2_048,
                raw_enhancer_mode=True,
            )

        self.assertEqual(writer.call_count, 3)
        self.assertIn("Approximate target: 1300 spoken words", writer.call_args_list[0].kwargs["prompt"])
        self.assertIn("At dusk, I found a lantern", result)
        self.assertIn("knowing the voice would return", result)
        self.assertTrue(all(call.kwargs["max_new_tokens"] <= 1_280 for call in writer.call_args_list))

    def test_selected_clip_duration_sets_generic_speech_plan_and_sections(self):
        replies = [
            "Plan: The neighbors prepare a meal and share what they learned.",
            "Uma: We set the table before the rain arrived.",
            "Kit: The soup warmed everyone while we traded stories.",
            "Uma: By evening, the clouds had cleared above the garden.",
            "Kit: We walked home together, already planning the next meal.",
        ]
        with fake_tts_prompt_modules(), patch.object(speech_writer, "MIN_SECTION_COVERAGE", 0.0), \
                patch.object(llm_service, "_last_generation_metrics", {}), \
                patch.object(llm_service, "generate", side_effect=replies) as writer:
            result = llm_service.enhance_prompt(
                "Two neighbors prepare a meal together.",
                mode="audio",
                duration_seconds=1_200,
                tts_enhance_mode="monologue",
                max_new_tokens=2_048,
                raw_enhancer_mode=True,
            )

        self.assertIn("We set the table", result)
        self.assertEqual(writer.call_count, 5)
        self.assertIn("Approximate target: 2600 spoken words", writer.call_args_list[0].kwargs["prompt"])
        self.assertIn("4 chronological section beats", writer.call_args_list[0].kwargs["prompt"])
        self.assertTrue(all(call.kwargs["max_new_tokens"] <= 1_280 for call in writer.call_args_list))

    def test_other_enhancement_modes_ignore_speech_writer_controls(self):
        for mode in ("image", "video"):
            with self.subTest(mode=mode), \
                    patch.dict(sys.modules, {"wgp": SimpleNamespace(get_model_def=lambda _: {})}), \
                    patch.object(speech_writer, "resolve_speech_target", wraps=speech_writer.resolve_speech_target) as resolver, \
                    patch.object(llm_service, "generate", return_value="A clear prompt.") as writer:
                result = llm_service.enhance_prompt(
                    "A quiet train station.",
                    mode=mode,
                    duration_seconds=1_200,
                    tts_enhance_mode="monologue",
                    tts_writer_instructions="This is audio-only guidance.",
                    tts_target_minutes=10,
                )

            self.assertEqual(result, "A clear prompt.")
            writer.assert_called_once()
            resolver.assert_not_called()
            self.assertNotIn("audio-only guidance", writer.call_args.kwargs["system_prompt"])
            self.assertNotIn("spoken-script size", writer.call_args.kwargs["prompt"])
            self.assertNotIn("Approximate target:", writer.call_args.kwargs["prompt"])


if __name__ == "__main__":
    unittest.main()
