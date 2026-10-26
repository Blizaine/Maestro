"""CPU-only contracts for Fish Drama 3 Auto and manual duration budgets."""

from __future__ import annotations

import inspect
import math
import os
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

# Avoid importing Maestro's broad TTS registry while exercising the local Fish
# implementation, matching the setup used by the Fish Audio CPU contract suite.
import models

tts_package = sys.modules.get("models.TTS", getattr(models, "TTS", None))
if tts_package is None:
    tts_package = types.ModuleType("models.TTS")
    tts_package.__path__ = [str(APP / "models" / "TTS")]
sys.modules["models.TTS"] = tts_package
models.TTS = tts_package

from models.TTS.fish_audio import pipeline as fish_pipeline_module
from models.TTS.fish_audio.duration import (
    DRAMA3_PAIRED_TAGS,
    duration_frame_budget,
    script_speech_units,
)
from models.TTS.fish_audio.fish_audio_handler import DIALOGUE_HISTORY_SECONDS, family_handler
from models.TTS.fish_audio.pipeline import FishAudioPipeline


class FishDurationBudgetTests(unittest.TestCase):
    sample_rate = FishAudioPipeline.sample_rate
    frame_length = FishAudioPipeline.frame_length

    def test_estimator_ignores_speaker_and_delivery_cues_but_counts_multilingual_symbols(self):
        script = "Speaker 1: [calm] Hello <whisper>你好🙂!</whisper>\n<|speaker:1|> こんにちは。"

        self.assertEqual(script_speech_units(script), 15)

    def test_estimator_ignores_every_supported_tag_and_phoneme_markers_but_counts_phones(self):
        tagged = " ".join(f"<{tag}>x</{tag}>" for tag in DRAMA3_PAIRED_TAGS)
        script = f"{tagged} <|phoneme_start|>K W IY1 N<|phoneme_end|>"

        self.assertEqual(script_speech_units(script), len(DRAMA3_PAIRED_TAGS) + len("KWIY1N"))

    def test_auto_budget_uses_the_documented_formula_and_caps_at_one_hour(self):
        short = duration_frame_budget(0, "hello", self.sample_rate, self.frame_length)
        long = duration_frame_budget(0, "x" * 50000, self.sample_rate, self.frame_length)

        self.assertTrue(short.automatic)
        self.assertEqual(short.script_units, 5)
        self.assertEqual(short.seconds, 10)
        self.assertLess(short.frames, long.frames)
        self.assertTrue(long.automatic)
        self.assertEqual(long.seconds, 3600)
        self.assertEqual(long.frames, math.floor(3600 * self.sample_rate / self.frame_length))

    def test_manual_seconds_floor_to_frames_and_invalid_requests_are_rejected(self):
        manual = duration_frame_budget(6, "hello", self.sample_rate, self.frame_length)

        self.assertFalse(manual.automatic)
        self.assertIsNone(manual.script_units)
        self.assertEqual(manual.frames, math.floor(6 * self.sample_rate / self.frame_length))
        for value in (-1, math.nan, math.inf, -math.inf, None, 3601):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "duration_seconds"):
                duration_frame_budget(value, "hello", self.sample_rate, self.frame_length)
        exact_hour = duration_frame_budget(3600, "hello", self.sample_rate, self.frame_length)
        self.assertEqual(exact_hour.frames, math.floor(3600 * self.sample_rate / self.frame_length))

    def test_handler_advertises_auto_and_defaults_to_it(self):
        model_def = family_handler.query_model_def("fish_drama3", {})
        slider = model_def["duration_slider"]
        defaults = {}
        family_handler.update_default_settings("fish_drama3", model_def, defaults)

        self.assertTrue(model_def["speech_auto_duration"])
        self.assertTrue(model_def["duration_is_maximum"])
        self.assertEqual((slider["min"], slider["default"], slider["max"]), (0, 0, 3600))
        self.assertTrue(model_def["speech_longform"])
        self.assertEqual(model_def["audio_segment_max_seconds"], 120)
        self.assertEqual(defaults["duration_seconds"], 0)
        custom_settings = {setting["id"]: setting["default"] for setting in model_def["custom_settings"]}
        self.assertEqual(DIALOGUE_HISTORY_SECONDS["default"], 60)
        self.assertEqual(custom_settings["dialogue_history_seconds"], 60)
        self.assertEqual(
            inspect.signature(FishAudioPipeline.generate).parameters["duration_seconds"].default,
            0,
        )


class FishDurationPipelineTests(unittest.TestCase):
    def make_pipeline(self, *, prompt_rows=12, context_rows=10000, generate_codes=None):
        pipeline = object.__new__(FishAudioPipeline)
        pipeline._interrupt = False
        pipeline._early_stop = False
        pipeline.last_codes = None
        pipeline.last_finish_reason = None
        pipeline.last_duration_budget_frames = None
        pipeline.last_duration_is_auto = False
        pipeline.last_duration_context_limited = False
        pipeline.last_duration_context_limit_frames = None
        pipeline.config = types.SimpleNamespace(
            text=types.SimpleNamespace(max_position_embeddings=context_rows)
        )
        pipeline.tokenizer = object()
        pipeline.vae = nn.Identity()
        pipeline._reference_transcripts = Mock(return_value=[])
        pipeline._release_decoder_caches = Mock()
        pipeline._decode = Mock(
            side_effect=lambda codes, _tile: torch.zeros(
                (1, codes.shape[-1] * FishAudioPipeline.frame_length), dtype=torch.float32
            )
        )
        pipeline._generate_codes = Mock(
            side_effect=generate_codes
            or (lambda prompt, max_frames, *args: (torch.zeros((10, min(3, max_frames))), "eos"))
        )
        prompt = torch.zeros((prompt_rows, 2), dtype=torch.long)
        pipeline._build_bounded_prompt = Mock(return_value=prompt)
        return pipeline, prompt

    def generate(
        self,
        pipeline,
        prompt,
        script,
        duration_seconds,
        *,
        custom_settings=None,
        prompt_side_effect=None,
    ):
        if prompt_side_effect is None:
            pipeline._build_bounded_prompt.return_value = prompt
        else:
            pipeline._build_bounded_prompt.side_effect = prompt_side_effect
        with patch.object(
            fish_pipeline_module.runtime,
            "execution_scope",
            return_value=nullcontext(),
        ):
            return pipeline.generate(
                script,
                duration_seconds=duration_seconds,
                custom_settings=custom_settings,
            )

    def test_auto_budget_allows_natural_eos_before_the_derived_cap(self):
        script = "Speaker 1: [calm] Hello <whisper>there</whisper>!"
        pipeline, prompt = self.make_pipeline()
        pipeline._generate_codes = Mock(
            return_value=(torch.zeros((10, 4), dtype=torch.long), "eos")
        )
        expected = duration_frame_budget(0, script, pipeline.sample_rate, pipeline.frame_length)

        result = self.generate(pipeline, prompt, script, 0)

        self.assertIsNotNone(result)
        self.assertEqual(pipeline._generate_codes.call_args.args[1], expected.frames)
        self.assertEqual(pipeline.last_duration_budget_frames, expected.frames)
        self.assertTrue(pipeline.last_duration_is_auto)
        self.assertFalse(pipeline.last_duration_context_limited)
        self.assertEqual(pipeline.last_finish_reason, "eos")
        self.assertEqual(pipeline.last_codes.shape[-1], 4)

    def test_manual_six_second_maximum_is_passed_as_exact_floor_frame_budget(self):
        pipeline, prompt = self.make_pipeline()

        def length_limited(_prompt, max_frames, *args):
            return torch.zeros((10, max_frames), dtype=torch.long), "length"

        pipeline._generate_codes = Mock(side_effect=length_limited)
        expected_frames = math.floor(6 * pipeline.sample_rate / pipeline.frame_length)

        self.generate(pipeline, prompt, "Speaker 1: Hello there.", 6)

        self.assertEqual(pipeline._generate_codes.call_args.args[1], expected_frames)
        self.assertEqual(pipeline.last_duration_budget_frames, expected_frames)
        self.assertFalse(pipeline.last_duration_is_auto)
        self.assertFalse(pipeline.last_duration_context_limited)
        self.assertEqual(pipeline.last_finish_reason, "length")

    def test_auto_budget_is_limited_by_prompt_context_and_reports_the_finish(self):
        pipeline, prompt = self.make_pipeline(prompt_rows=20, context_rows=50)
        pipeline._generate_codes = Mock(
            return_value=(torch.zeros((10, 2), dtype=torch.long), "length")
        )
        script = "x" * 5000

        self.generate(pipeline, prompt, script, 0)

        self.assertEqual(pipeline._generate_codes.call_args.args[1], 30)
        self.assertEqual(
            pipeline.last_duration_budget_frames,
            duration_frame_budget(0, script, pipeline.sample_rate, pipeline.frame_length).frames,
        )
        self.assertTrue(pipeline.last_duration_context_limited)
        self.assertEqual(pipeline.last_duration_context_limit_frames, 30)
        self.assertEqual(pipeline.last_finish_reason, "length")

    def test_auto_budget_is_shared_across_dialogue_cuts_and_later_context_cap(self):
        script = "Speaker 1: One.\nSpeaker 2: Two."
        total_budget = duration_frame_budget(
            0, script, FishAudioPipeline.sample_rate, FishAudioPipeline.frame_length
        ).frames
        prompts = (
            torch.zeros((12, 2), dtype=torch.long),
            torch.zeros((12 + total_budget - 7, 2), dtype=torch.long),
        )
        pipeline, default_prompt = self.make_pipeline(
            prompt_rows=12,
            context_rows=12 + total_budget,
        )
        limits = []

        def generate_part(_prompt, max_frames, *args):
            limits.append(max_frames)
            if len(limits) == 1:
                return torch.zeros((10, max_frames), dtype=torch.long), "eos"
            return torch.zeros((10, max_frames), dtype=torch.long), "length"

        pipeline._generate_codes = Mock(side_effect=generate_part)
        second_part_budget = duration_frame_budget(
            0, "Speaker 2: Two.", pipeline.sample_rate, pipeline.frame_length
        ).frames

        self.generate(
            pipeline,
            default_prompt,
            script,
            0,
            custom_settings={"dialogue_cut_lines": 1},
            prompt_side_effect=prompts,
        )

        # The first per-part Auto allowance is 9 seconds. The later prompt has
        # only seven context rows left, which is tighter than the remaining
        # whole-job budget and its fresh per-part allowance.
        first_part_budget = duration_frame_budget(
            0, "Speaker 1: One.", pipeline.sample_rate, pipeline.frame_length
        ).frames
        self.assertEqual(limits, [first_part_budget, 7])
        self.assertLess(limits[1], second_part_budget)
        self.assertEqual(pipeline.last_duration_budget_frames, total_budget)
        self.assertTrue(pipeline.last_duration_context_limited)
        self.assertEqual(pipeline.last_duration_context_limit_frames, 7)
        self.assertEqual(pipeline.last_finish_reason, "length")
        self.assertEqual(pipeline.last_codes.shape[-1], first_part_budget + limits[1])

    def test_manual_maximum_keeps_the_existing_context_error(self):
        pipeline = object.__new__(FishAudioPipeline)
        pipeline.transformer = object()
        pipeline.transformer2 = object()
        pipeline.config = types.SimpleNamespace(
            text=types.SimpleNamespace(max_position_embeddings=20)
        )

        with self.assertRaisesRegex(ValueError, "context capacity"):
            pipeline._generate_codes(
                torch.zeros((12, 2), dtype=torch.long),
                9,
                0.8,
                0,
                0.95,
                None,
            )


if __name__ == "__main__":
    unittest.main()
