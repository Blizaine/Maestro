"""CPU-only regressions for Fish Drama 3 long-form chunking and assembly."""

from __future__ import annotations

import os
import sys
import types
import unicodedata
import unittest
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

import models

tts_package = sys.modules.get("models.TTS", getattr(models, "TTS", None))
if tts_package is None:
    tts_package = types.ModuleType("models.TTS")
    tts_package.__path__ = [str(APP / "models" / "TTS")]
sys.modules["models.TTS"] = tts_package
models.TTS = tts_package

from models.TTS.fish_audio import pipeline as fish_pipeline_module
from models.TTS.fish_audio.backbone import GenerationCancelled
from models.TTS.fish_audio.duration import (
    DRAMA3_PAIRED_TAGS,
    DRAMA3_PHONEME_END,
    DRAMA3_PHONEME_START,
    LONGFORM_CHUNK_MAX_SECONDS,
    duration_frame_budget,
    script_speech_units,
)
from models.TTS.fish_audio.pipeline import (
    CODEC_CONTEXT_FRAMES,
    CODEC_DECODE_TILE_FRAMES,
    FishAudioPipeline,
)
from models.TTS.fish_audio.prompt import (
    build_drama3_prompt,
    DRAMA3_GENERATION_START,
    DRAMA3_MESSAGE_END,
    PAIRED_TAGS,
    check_paired_tags,
    drama3_text_chunks,
    fit_history_to_prompt_rows,
    with_speaker_tag,
)


class FakeTokenizer:
    def encode(self, text):
        return types.SimpleNamespace(ids=list(range(len(text))))


class FakeDramaTokenizer(FakeTokenizer):
    _tokens = {
        "<|semantic:0|>": 100,
        "<|semantic:7|>": 101,
        "<|im_end|>": 102,
        DRAMA3_PHONEME_START: 103,
        DRAMA3_PHONEME_END: 104,
    }

    def token_to_id(self, token):
        return self._tokens.get(token)


class FishLongformChunkTests(unittest.TestCase):
    def test_chunks_keep_exact_source_spans_and_carry_speaker_and_opening_cue(self):
        script = "Speaker 2: [soft, reassuring] The first sentence ends here. The next sentence continues without a new line."

        chunks = drama3_text_chunks(script, max_speech_units=30)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.source_text for chunk in chunks), script)
        self.assertEqual(chunks[0].source_start, 0)
        for left, right in pairwise(chunks):
            self.assertEqual(left.source_end, right.source_start)
        self.assertIn("<|speaker:1|>", chunks[1].prompt_text)
        self.assertIn("[soft, reassuring]", chunks[1].prompt_text)
        self.assertIn("The next sentence", chunks[1].prompt_text)
        self.assertLessEqual(max(chunk.speech_units for chunk in chunks), 30)

    def test_paired_delivery_tag_state_is_closed_and_reopened_at_chunk_boundaries(self):
        script = "Speaker 2: [quiet] <whisper>Stay calm and listen carefully. Repeat these words once more.</whisper> Now go."

        chunks = drama3_text_chunks(script, max_speech_units=18)

        self.assertGreater(len(chunks), 2)
        self.assertEqual("".join(chunk.source_text for chunk in chunks), script)
        for chunk in chunks:
            self.assertEqual(chunk.prompt_text.count("<whisper>"), chunk.prompt_text.count("</whisper>"))
        self.assertTrue(any(chunk.source_start > 0 and "<whisper>" in chunk.prompt_text for chunk in chunks))

    def test_all_documented_paired_tags_validate_and_remain_balanced_in_each_chunk(self):
        opening = "".join(f"<{tag}>" for tag in PAIRED_TAGS)
        closing = "".join(f"</{tag}>" for tag in reversed(PAIRED_TAGS))
        script = f"{opening}" + " ".join(["Choose each word carefully."] * 8) + closing

        check_paired_tags(script, "script", tokenizer=FakeDramaTokenizer())
        chunks = drama3_text_chunks(script, max_speech_units=30)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.source_text for chunk in chunks), script)
        for chunk in chunks:
            check_paired_tags(chunk.prompt_text, "chunk")
            for tag in DRAMA3_PAIRED_TAGS:
                self.assertEqual(
                    chunk.prompt_text.count(f"<{tag}>"),
                    chunk.prompt_text.count(f"</{tag}>"),
                )

    def test_paired_tag_errors_cover_crossing_unclosed_or_unknown_markup(self):
        for script in (
            "<whisper><slow>crossed</whisper></slow>",
            "<soft>missing close",
            "</fast>orphan close",
            "<bold>unknown</bold>",
        ):
            with self.subTest(script=script), self.assertRaises(ValueError):
                check_paired_tags(script, "script")

    def test_phoneme_spans_are_lossless_atomic_counted_and_one_line(self):
        pronunciation = f"{DRAMA3_PHONEME_START}K W IY1 N{DRAMA3_PHONEME_END}"
        script = f"Speaker 1: Say {pronunciation} clearly, then repeat {pronunciation} once."
        check_paired_tags(script, "script", tokenizer=FakeDramaTokenizer())

        chunks = drama3_text_chunks(script, max_speech_units=8)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.source_text for chunk in chunks), script)
        self.assertEqual(sum(chunk.speech_units for chunk in chunks), script_speech_units(script))
        self.assertEqual(
            sum(chunk.source_text.count(DRAMA3_PHONEME_START) for chunk in chunks),
            2,
        )
        for chunk in chunks:
            self.assertEqual(
                chunk.source_text.count(DRAMA3_PHONEME_START),
                chunk.source_text.count(DRAMA3_PHONEME_END),
            )

        multiline = f"{DRAMA3_PHONEME_START}K W\nIY1 N{DRAMA3_PHONEME_END}"
        with self.assertRaisesRegex(ValueError, "one line"):
            check_paired_tags(multiline, "script", tokenizer=FakeDramaTokenizer())
        with self.assertRaisesRegex(ValueError, "not available in this local tokenizer"):
            check_paired_tags(
                f"{DRAMA3_PHONEME_START}K W IY1 N{DRAMA3_PHONEME_END}",
                "script",
                tokenizer=FakeTokenizer(),
            )

    def test_continuing_midline_cue_carries_but_one_off_sound_does_not(self):
        event = "[laughs]"
        script = (
            "Speaker 1: [low register, steady accent] First sentence ends here. "
            "[nervous but controlled] The next sentence changes the acting. "
            f"{event} "
            + " ".join(["The story continues without a new direction."] * 8)
        )

        chunks = drama3_text_chunks(script, max_speech_units=20)

        after_event = [
            chunk for chunk in chunks
            if chunk.source_start >= script.index(event) + len(event)
        ]
        self.assertTrue(after_event)
        self.assertTrue(all("[nervous but controlled]" in chunk.prompt_text for chunk in after_event))
        self.assertTrue(all(event not in chunk.prompt_text for chunk in after_event))

    def test_new_speaker_at_cut_keeps_its_label_and_does_not_inherit_other_voice_cue(self):
        script = (
            "Speaker 1: [tense] The first voice finishes its turn.\n"
            "Speaker 2: [calm] A different voice answers the question."
        )

        chunks = drama3_text_chunks(script, lines_per_part=1, max_speech_units=1000)

        self.assertEqual(len(chunks), 2)
        self.assertEqual("".join(chunk.source_text for chunk in chunks), script)
        self.assertTrue(chunks[1].prompt_text.startswith("<|speaker:1|>"))
        self.assertIn("[calm]", chunks[1].prompt_text)
        self.assertNotIn("[tense]", chunks[1].prompt_text)
        self.assertNotIn("Speaker 2:", chunks[1].prompt_text)

    def test_a_long_single_paragraph_splits_on_sentence_boundaries(self):
        script = " ".join(["The lantern flickers before the door opens."] * 12)

        chunks = drama3_text_chunks(script, max_speech_units=75)

        self.assertGreater(len(chunks), 4)
        self.assertEqual("".join(chunk.source_text for chunk in chunks), script)
        self.assertTrue(all(chunk.source_text.rstrip().endswith(".") for chunk in chunks[:-1]))

    def test_explicit_line_cuts_are_additional_and_trailing_blank_lines_do_not_make_a_chunk(self):
        script = "Speaker 1: First sentence. Still first turn.\nSpeaker 2: Second turn.\n\n"

        automatic = drama3_text_chunks(script, max_speech_units=1000)
        cut_by_line = drama3_text_chunks(script, lines_per_part=1, max_speech_units=1000)
        trailing_blank = drama3_text_chunks("Speaker 1: Hello.\n\n", lines_per_part=1)

        self.assertEqual("".join(chunk.source_text for chunk in automatic), script)
        self.assertEqual("".join(chunk.source_text for chunk in cut_by_line), script)
        self.assertGreater(len(cut_by_line), len(automatic))
        self.assertEqual(len(trailing_blank), 1)
        self.assertEqual(trailing_blank[0].source_text, "Speaker 1: Hello.\n\n")

    def test_unicode_fallback_does_not_split_combining_or_zwj_sequences(self):
        script = "你好🙂e\u0301👩\u200d💻🇺🇸かな文字" * 8

        chunks = drama3_text_chunks(script, max_speech_units=5)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.source_text for chunk in chunks), script)
        for chunk in chunks[:-1]:
            boundary = chunk.source_end
            self.assertFalse(unicodedata_mark(script[boundary]))
            self.assertNotEqual(script[boundary - 1], "\u200d")
            self.assertNotEqual(script[boundary], "\u200d")
            if 0x1F1E6 <= ord(script[boundary - 1]) <= 0x1F1FF:
                self.assertFalse(0x1F1E6 <= ord(script[boundary]) <= 0x1F1FF)

    def test_cue_only_script_stays_a_valid_source_chunk(self):
        chunks = drama3_text_chunks("[laughs]")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_text, "[laughs]")
        self.assertEqual(chunks[0].speech_units, 0)

    def test_history_keeps_whole_aligned_turns_and_fits_newest_before_anchor(self):
        pipeline = object.__new__(FishAudioPipeline)
        pipeline.sample_rate = FishAudioPipeline.sample_rate
        pipeline.frame_length = FishAudioPipeline.frame_length
        oversized = torch.zeros((2, 30), dtype=torch.long)
        recent = torch.ones((2, 5), dtype=torch.long)
        self.assertEqual(pipeline._history([("long turn", oversized)], 1), [])

        def turn_rows(text, codes):
            return sum(
                len(FakeTokenizer().encode(segment).ids)
                for segment in (
                    "<|im_start|>user\n",
                    with_speaker_tag(text),
                    DRAMA3_MESSAGE_END,
                    DRAMA3_GENERATION_START,
                    DRAMA3_MESSAGE_END,
                )
            ) + codes.shape[-1]

        anchor = ("first chunk", torch.full((2, 10), 2, dtype=torch.long))
        latest = ("recent chunk", recent)
        fitted = fit_history_to_prompt_rows(
            [anchor, latest], FakeTokenizer(), turn_rows(*latest)
        )

        self.assertEqual(len(fitted), 1)
        self.assertEqual(fitted[0][0], "recent chunk")
        self.assertIs(fitted[0][1], recent)

    def test_history_metadata_reports_whole_turns_after_fitting_and_actual_prompt_rows(self):
        pipeline = object.__new__(FishAudioPipeline)
        pipeline.sample_rate = 1000
        pipeline.frame_length = 100
        pipeline.tokenizer = FakeDramaTokenizer()
        pipeline.config = types.SimpleNamespace(
            audio_decoder=types.SimpleNamespace(num_codebooks=2, vocab_size=8),
            semantic_start_token_id=100,
            semantic_end_token_id=101,
            eos_token_id=102,
        )
        recent = ("recent complete turn", torch.ones((2, 450), dtype=torch.long))
        anchor = ("first chunk anchor", torch.full((2, 500), 2, dtype=torch.long))

        base = build_drama3_prompt("current", pipeline.tokenizer, pipeline.config)
        prompt = pipeline._build_bounded_prompt("current", [], [], [recent], 60, anchor)
        metadata = pipeline._last_prompt_history_context

        self.assertEqual(metadata["recent_turn_count"], 1)
        self.assertEqual(metadata["recent_audio_seconds"], 45.0)
        self.assertTrue(metadata["anchor_included"])
        self.assertEqual(metadata["anchor_audio_seconds"], 50.0)
        self.assertEqual(metadata["total_audio_seconds"], 95.0)
        self.assertEqual(metadata["history_prompt_rows"], len(prompt) - len(base))
        self.assertGreater(metadata["history_prompt_rows"], 950)

        pipeline._build_bounded_prompt("current", [], [], [recent], 20, anchor)
        self.assertEqual(pipeline._last_prompt_history_context["recent_turn_count"], 0)
        self.assertTrue(pipeline._last_prompt_history_context["anchor_included"])
        pipeline._build_bounded_prompt("current", [], [], [recent], 0, anchor)
        self.assertEqual(pipeline._last_prompt_history_context["total_turn_count"], 0)


def unicodedata_mark(character):
    return unicodedata.category(character).startswith("M")


class FishLongformPipelineTests(unittest.TestCase):
    def make_pipeline(self, *, context_rows=100000, prompt_builder=None, code_generator=None, decoder=None):
        pipeline = object.__new__(FishAudioPipeline)
        pipeline._interrupt = False
        pipeline._early_stop = False
        pipeline.config = types.SimpleNamespace(
            text=types.SimpleNamespace(max_position_embeddings=context_rows)
        )
        pipeline.tokenizer = FakeTokenizer()
        pipeline.vae = nn.Identity()
        pipeline._reference_transcripts = Mock(return_value=[])
        pipeline._reference_codes = Mock(return_value=torch.zeros((2, 1), dtype=torch.long))
        pipeline._release_decoder_caches = Mock()
        pipeline._build_bounded_prompt = Mock(
            side_effect=prompt_builder
            or (lambda *args: torch.zeros((12, 2), dtype=torch.long))
        )
        pipeline._generate_codes = Mock(
            side_effect=code_generator
            or (lambda _prompt, maximum, *args: (torch.zeros((2, min(2, maximum)), dtype=torch.long), "eos"))
        )
        pipeline._decode = Mock(
            side_effect=decoder
            or (
                lambda codes, tile: codes[0:1]
                .repeat_interleave(FishAudioPipeline.frame_length, dim=-1)
                .float()
            )
        )
        return pipeline

    def generate(self, pipeline, script, duration_seconds=0, **kwargs):
        with patch.object(
            fish_pipeline_module.runtime,
            "execution_scope",
            return_value=nullcontext(),
        ):
            return pipeline.generate(
                script,
                duration_seconds=duration_seconds,
                **kwargs,
            )

    def test_history_defaults_to_sixty_seconds_and_preserves_saved_overrides_and_rng_metadata(self):
        expected_values = ((None, 60), (20, 20), (0, 0))
        for configured, expected in expected_values:
            with self.subTest(configured=configured):
                seen_history_windows = []

                def prompt_builder(_text, _transcripts, _references, _turns, history_seconds, _anchor):
                    seen_history_windows.append(history_seconds)
                    return torch.zeros((12, 2), dtype=torch.long)

                pipeline = self.make_pipeline(prompt_builder=prompt_builder)
                settings = {"dialogue_cut_lines": 1}
                if configured is not None:
                    settings["dialogue_history_seconds"] = configured
                with patch.object(fish_pipeline_module.torch, "manual_seed") as manual_seed:
                    result = self.generate(
                        pipeline,
                        "Speaker 1: First line.\nSpeaker 2: Second line.",
                        duration_seconds=60,
                        custom_settings=settings,
                        seed=42,
                    )

                report = result["speech_generation"]
                self.assertEqual(seen_history_windows, [expected, expected])
                self.assertEqual(report["history_recent_max_seconds"], expected)
                manual_seed.assert_called_once_with(42)
                self.assertEqual(
                    report["rng"],
                    {
                        "strategy": "torch_manual_seed_once_per_job",
                        "seed": 42,
                        "reseed_per_chunk": False,
                    },
                )

    def test_eos_chunks_assemble_one_audio_tensor_and_keep_codec_context_bounded(self):
        script = " ".join(["The lantern flickers before the old door opens."] * 100)
        prompts_seen = []
        generation_budgets = []
        decode_inputs = []
        sample_rate, frame_length = 1000, 100000

        # Synthetic timing keeps the assembled tensor small while making each
        # semantic frame represent 100 seconds for this duration contract.
        self.assertGreater(
            duration_frame_budget(0, script, sample_rate, frame_length).seconds,
            600,
        )

        def prompt_builder(text, transcripts, references, turns, history_seconds, anchor):
            prompts_seen.append((text, list(turns), anchor, transcripts, references, history_seconds))
            return torch.zeros((12, 2), dtype=torch.long)

        def code_generator(_prompt, maximum, *args):
            generation_budgets.append(maximum)
            value = len(generation_budgets)
            return torch.full((2, maximum), value, dtype=torch.long), "eos"

        def decoder(codes, tile):
            decode_inputs.append((codes.clone(), tile))
            return codes[0:1].repeat_interleave(frame_length, dim=-1).float()

        pipeline = self.make_pipeline(
            prompt_builder=prompt_builder,
            code_generator=code_generator,
            decoder=decoder,
        )
        pipeline.sample_rate = sample_rate
        pipeline.frame_length = frame_length

        result = self.generate(
            pipeline,
            script,
            VAE_tile_size=512,
            custom_settings={"dialogue_history_seconds": 120},
        )

        report = result["speech_generation"]
        self.assertGreater(report["total_chunks"], 5)
        self.assertEqual(report["completed_chunks"], report["total_chunks"])
        self.assertFalse(report["incomplete"])
        self.assertEqual(report["covered_source_span"], {"start": 0, "end": len(script)})
        spans = [row["source_span"] for row in report["chunks"]]
        self.assertEqual(spans[0]["start"], 0)
        self.assertEqual(spans[-1]["end"], len(script))
        self.assertTrue(all(left["end"] == right["start"] for left, right in pairwise(spans)))
        self.assertGreater(result["x"].shape[-1] / sample_rate, 600)
        self.assertEqual(len(result["x"][0]), report["generated_frames"] * frame_length)
        self.assertEqual(pipeline.last_codes.shape[-1], report["generated_frames"])
        max_segment_frames = int(
            LONGFORM_CHUNK_MAX_SECONDS
            * sample_rate
            / frame_length
        )
        self.assertTrue(all(budget <= max_segment_frames for budget in generation_budgets))
        self.assertTrue(all(tile <= CODEC_DECODE_TILE_FRAMES for _, tile in decode_inputs))
        self.assertTrue(
            all(
                codes.shape[-1] <= CODEC_CONTEXT_FRAMES + max_segment_frames
                for codes, _ in decode_inputs
            )
        )
        self.assertEqual(decode_inputs[0][0].shape[-1], generation_budgets[0])
        self.assertTrue(any(anchor is not None for _, _, anchor, *_ in prompts_seen[1:]))
        self.assertTrue(any(turns for _, turns, *_ in prompts_seen[2:]))
        frames = result["x"].reshape(1, -1, frame_length).mean(-1).squeeze(0)
        expected = torch.tensor(
            list(range(1, report["total_chunks"] + 1)),
            dtype=torch.float32,
        )
        torch.testing.assert_close(frames, expected)
        self.assertEqual(result["generation_warnings"], [])

    def test_manual_whole_job_cap_is_shared_across_chunks(self):
        script = "Speaker 1: " + ("A careful line follows. " * 24)
        limits = []

        def code_generator(_prompt, maximum, *args):
            limits.append(maximum)
            if len(limits) == 1:
                return torch.zeros((2, 100), dtype=torch.long), "eos"
            return torch.zeros((2, maximum), dtype=torch.long), "length"

        pipeline = self.make_pipeline(code_generator=code_generator)

        result = self.generate(pipeline, script, duration_seconds=6)

        self.assertGreater(len(limits), 1)
        self.assertEqual(limits[0], duration_frame_budget(6, script, 44100, 2048).frames)
        self.assertEqual(limits[1], limits[0] - 100)
        report = result["speech_generation"]
        self.assertTrue(report["incomplete"])
        self.assertEqual(report["completed_chunks"], 1)
        self.assertEqual(report["generated_frames"], limits[0])
        self.assertEqual(report["covered_source_span"]["end"], report["chunks"][0]["source_span"]["end"])
        self.assertTrue(result["generation_warnings"])

    def test_context_limit_on_a_later_chunk_returns_partial_audio_with_exact_eos_coverage(self):
        script = "Speaker 1: " + ("A careful line follows. " * 24)
        prompt_sizes = iter([12, 20])

        def prompt_builder(*args):
            return torch.zeros((next(prompt_sizes), 2), dtype=torch.long)

        pipeline = self.make_pipeline(context_rows=20, prompt_builder=prompt_builder)
        pipeline._generate_codes = Mock(
            return_value=(torch.zeros((2, 2), dtype=torch.long), "eos")
        )

        result = self.generate(pipeline, script, duration_seconds=60)

        report = result["speech_generation"]
        self.assertTrue(report["incomplete"])
        self.assertEqual(report["completed_chunks"], 1)
        self.assertEqual(report["chunks"][0]["budget_frames"], 8)
        self.assertEqual(report["chunks"][1]["status"], "context_incomplete")
        self.assertEqual(report["covered_source_span"]["end"], report["chunks"][0]["source_span"]["end"])
        self.assertTrue(any("context limit" in warning.lower() for warning in result["generation_warnings"]))

    def test_length_and_early_stop_are_reported_incomplete_without_claiming_later_spans(self):
        script = "Speaker 1: " + ("A careful line follows. " * 24)
        pipeline = self.make_pipeline(
            code_generator=lambda _prompt, maximum, *args: (
                torch.zeros((2, min(3, maximum)), dtype=torch.long),
                "length",
            )
        )

        result = self.generate(pipeline, script, duration_seconds=60)

        report = result["speech_generation"]
        self.assertTrue(report["incomplete"])
        self.assertEqual(report["completed_chunks"], 0)
        self.assertEqual(report["covered_source_span"]["end"], 0)
        self.assertEqual(report["chunks"][0]["finish_reason"], "length")
        self.assertEqual(report["chunks"][1]["status"], "not_started")
        self.assertTrue(result["generation_warnings"])

        early = self.make_pipeline(
            code_generator=lambda _prompt, maximum, *args: (
                torch.zeros((2, 2), dtype=torch.long),
                "early_stop",
            )
        )
        early_result = self.generate(early, "A short line.", duration_seconds=60)
        self.assertEqual(early_result["speech_generation"]["status"], "incomplete")
        self.assertEqual(early_result["speech_generation"]["completed_chunks"], 0)

    def test_eos_without_audio_does_not_mark_the_script_complete(self):
        script = "Speaker 1: " + ("A careful line follows. " * 24)
        calls = 0

        def code_generator(_prompt, maximum, *args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return torch.zeros((2, 2), dtype=torch.long), "eos"
            return torch.zeros((2, 0), dtype=torch.long), "eos"

        pipeline = self.make_pipeline(code_generator=code_generator)

        result = self.generate(pipeline, script, duration_seconds=60)

        report = result["speech_generation"]
        self.assertTrue(report["incomplete"])
        self.assertEqual(report["completed_chunks"], 1)
        self.assertEqual(report["chunks"][1]["finish_reason"], "eos")
        self.assertEqual(report["chunks"][1]["status"], "incomplete")
        self.assertEqual(report["covered_source_span"]["end"], report["chunks"][0]["source_span"]["end"])
        self.assertIn("without producing audio", result["generation_warnings"][0])

    def test_cancel_returns_none_and_preserves_a_cancelled_report(self):
        pipeline = self.make_pipeline(
            code_generator=lambda *_args: (_ for _ in ()).throw(GenerationCancelled())
        )

        result = self.generate(pipeline, "A short line.")

        self.assertIsNone(result)
        self.assertEqual(pipeline.last_longform_report["status"], "cancelled")
        self.assertTrue(pipeline.last_longform_report["incomplete"])
        self.assertEqual(pipeline.last_longform_report["chunks"][0]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
