"""CPU-only contracts for the Fish Drama 3 user-facing enhancer guide."""

from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from models.TTS.fish_audio.help import DIALOGUE_ENHANCER, ENHANCER
from models.TTS.fish_audio.prompt import check_paired_tags, drama3_text_chunks
from services import llm_service, speech_writer


@contextmanager
def fish_native_writer(replies):
    """Keep real native guides, routing, section composition and cleanup."""
    prompts = types.ModuleType("models.TTS.prompt_enhancers")
    prompts.TTS_MONOLOGUE_PROMPT = "Generic fallback, which Fish must not use."
    prompts.TTS_QWEN3_DIALOGUE_PROMPT = "Generic dialogue fallback."
    registry = types.ModuleType("wgp")
    registry.get_model_def = lambda _: {
        "text_prompt_enhancer_instructions": ENHANCER,
        "text_prompt_enhancer_instructions1": DIALOGUE_ENHANCER,
    }
    with patch.dict(sys.modules, {"wgp": registry, "models.TTS.prompt_enhancers": prompts}), \
            patch("services.enhance_guides.get_enhance_guide", return_value="Unused outer audio guide."), \
            patch.object(llm_service, "_last_generation_metrics", {}), \
            patch.object(llm_service, "generate", side_effect=replies) as writer:
        yield writer


class FishPromptGuideTests(unittest.TestCase):
    def test_native_monologue_guide_reaches_writer_and_inline_controls_survive_cleanup(self):
        script = "I checked <emphasis>every</emphasis> door. [relieved] Only one was open."
        with fish_native_writer([script]) as writer:
            result = llm_service.enhance_prompt(
                "Write a short account of finding the exit.", mode="audio",
                model_type="fish_drama3_int4", tts_enhance_mode="monologue",
            )
        self.assertEqual(result, script)
        self.assertIn(ENHANCER, writer.call_args.kwargs["system_prompt"])
        self.assertNotIn("Generic fallback", writer.call_args.kwargs["system_prompt"])
        self.assertIn("Supported native performance controls belong in the script", writer.call_args.kwargs["system_prompt"])
        self.assertNotIn("No headers, no labels", writer.call_args.kwargs["system_prompt"])

    def test_native_dialogue_guide_keeps_plain_turns_and_word_level_controls(self):
        script = (
            "Speaker 1: Was it <stress>you</stress>?\n"
            "Speaker 2: I was already home. [hurt] You never asked."
        )
        with fish_native_writer([script]) as writer:
            result = llm_service.enhance_prompt(
                "Two neighbors discuss a misunderstanding.", mode="audio",
                model_type="fish_drama3_int4", tts_enhance_mode="dialogue", tts_voice_count=2,
            )
        self.assertEqual(result, script)
        self.assertIn(DIALOGUE_ENHANCER, writer.call_args.kwargs["system_prompt"])
        self.assertNotIn("Generic dialogue fallback", writer.call_args.kwargs["system_prompt"])
        self.assertIn("model-native speaker labels", writer.call_args.kwargs["system_prompt"])
        self.assertNotIn("No headers, no labels", writer.call_args.kwargs["system_prompt"])

    def test_longform_wrapper_keeps_real_native_guide_and_source_controls(self):
        first = "I found <emphasis>the original letter</emphasis>. [curious] Its seal was intact."
        second = "The address was mine. [relieved] At last, I knew who had sent it."
        with fish_native_writer(["Plan: find the letter and solve its origin.", first, second]) as writer, \
                patch.object(speech_writer, "MIN_SECTION_COVERAGE", 0.0):
            result = llm_service.enhance_prompt(
                "Tell a ten-minute story about a letter.", mode="audio", model_type="fish_drama3_int4",
                tts_enhance_mode="monologue", max_new_tokens=2048,
            )
        self.assertEqual(result, first + "\n" + second)
        self.assertEqual(writer.call_count, 3)
        for call in writer.call_args_list[1:]:
            system = call.kwargs["system_prompt"]
            self.assertIn(ENHANCER, system)
            self.assertNotIn("change delivery only when", system)
            self.assertNotIn("only when the user brief calls", system)

    def test_guide_examples_are_valid_and_lossless_through_chunking(self):
        for guide in (ENHANCER, DIALOGUE_ENHANCER):
            example = guide.split("\n\nExample:\n", 1)[1]
            with self.subTest(guide=guide[:45]):
                check_paired_tags(example, "enhancer example")
                chunks = drama3_text_chunks(example, max_speech_units=35)
                self.assertEqual("".join(chunk.source_text for chunk in chunks), example)
                for chunk in chunks:
                    check_paired_tags(chunk.prompt_text, "compiled example chunk")

    def test_plain_output_is_not_automatically_tagged_or_rejected(self):
        script = "I found the letter. Its seal was intact."
        with fish_native_writer([script]):
            result = llm_service.enhance_prompt(
                "Keep these exact words plain and untagged: " + script, mode="audio",
                model_type="fish_drama3_int4", tts_enhance_mode="monologue",
            )
        self.assertEqual(result, script)


if __name__ == "__main__":
    unittest.main()
