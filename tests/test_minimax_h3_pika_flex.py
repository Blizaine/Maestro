# ruff: noqa: E402
"""CPU contract tests for the experimental Pika H3 Flex core helpers."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
# The standalone test bootstraps the app source directory before importing its module.
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3.pika_flex import (
    PIKA_FLEX_KNOTS,
    PIKA_FLEX_STEPS,
    is_pika_flex_single_frame_start_anchor,
    pika_flex_sequence_enabled,
    pika_flex_sigmas,
    pika_flex_step,
    validate_pika_flex_request,
)


class PikaFlexSigmaTests(unittest.TestCase):
    def test_seven_independently_shifted_knots_define_six_evaluations(self):
        self.assertEqual(PIKA_FLEX_KNOTS, (1000, 750, 500, 250, 100, 40, 0))
        self.assertEqual(PIKA_FLEX_STEPS, 6)

        for shift in (12.0, 3.0):
            with self.subTest(shift=shift):
                actual = pika_flex_sigmas(shift)
                expected = tuple(
                    (shift * knot) / (1000.0 + (shift - 1.0) * knot)
                    for knot in PIKA_FLEX_KNOTS
                )
                self.assertEqual(len(actual), 7)
                for got, want in zip(actual, expected):
                    self.assertAlmostEqual(got, want, places=14)
                self.assertEqual(actual[0], 1.0)
                self.assertEqual(actual[-1], 0.0)
                self.assertTrue(all(a >= b for a, b in zip(actual, actual[1:])))

    def test_bad_shift_is_rejected(self):
        for shift in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(shift=shift), self.assertRaises(ValueError):
                pika_flex_sigmas(shift)


class PikaFlexStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("Pika H3 Flex sampler tests require PyTorch") from error
        cls.torch = torch

    def test_six_velocity_evaluations_end_cleanly_without_a_sixth_noise_draw(self):
        torch = self.torch
        sigmas = pika_flex_sigmas(12.0)
        sample = torch.tensor([[0.25, -0.5], [0.75, 0.125]], dtype=torch.float32)
        predictions = [
            torch.full_like(sample, (index - 2.0) / 10.0)
            for index in range(PIKA_FLEX_STEPS)
        ]
        generator = torch.Generator(device="cpu").manual_seed(392)
        reference_generator = torch.Generator(device="cpu").manual_seed(392)
        calls = 0
        terminal_expected = None

        for index in range(PIKA_FLEX_STEPS):
            calls += 1  # One native H3 forward supplies each velocity.
            if index == PIKA_FLEX_STEPS - 1:
                terminal_expected = (
                    sample.float() + sigmas[index] * predictions[index].float()
                )
            sample = pika_flex_step(
                sample,
                predictions[index],
                sigmas[index],
                sigmas[index + 1],
                generator,
            )
            if index < PIKA_FLEX_STEPS - 1:
                torch.randn(sample.shape, dtype=torch.float32, generator=reference_generator)

        self.assertEqual(calls, 6)
        self.assertTrue(torch.equal(generator.get_state(), reference_generator.get_state()))
        self.assertTrue(torch.equal(sample, terminal_expected))

    def test_transition_matches_explicit_fp32_reference_formula(self):
        torch = self.torch
        sample = torch.tensor([1.0000001, -2.0000003], dtype=torch.float64)
        prediction = torch.tensor([0.1000001, 0.3000003], dtype=torch.float64)
        sigma, next_sigma = 0.75, 0.4
        actual_generator = torch.Generator(device="cpu").manual_seed(71)
        reference_generator = torch.Generator(device="cpu").manual_seed(71)

        clean_fp32 = sample.float() + sigma * prediction.float()
        fresh_fp32_noise = torch.randn(
            sample.shape,
            dtype=torch.float32,
            generator=reference_generator,
        )
        expected = (1.0 - next_sigma) * clean_fp32 + next_sigma * fresh_fp32_noise
        actual = pika_flex_step(
            sample, prediction, sigma, next_sigma, actual_generator
        )

        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_generator.get_state(), reference_generator.get_state()))

    def test_video_audio_ladders_and_target_noise_are_independent(self):
        torch = self.torch
        video_sigmas = pika_flex_sigmas(12.0)
        audio_sigmas = pika_flex_sigmas(3.0)
        self.assertNotEqual(video_sigmas, audio_sigmas)

        video_generator = torch.Generator(device="cpu").manual_seed(1)
        audio_generator = torch.Generator(device="cpu").manual_seed(2)
        video_reference_generator = torch.Generator(device="cpu").manual_seed(1)
        audio_reference_generator = torch.Generator(device="cpu").manual_seed(2)
        video_target = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
        audio_target = torch.tensor([[-0.3, 0.1, 0.5]], dtype=torch.float32)
        video_prediction = torch.full_like(video_target, 0.25)
        audio_prediction = torch.full_like(audio_target, -0.2)
        ref_rows = torch.tensor([[10.0, 11.0]], dtype=torch.float32)
        refs_before = ref_rows.clone()

        video_noise = torch.randn(video_target.shape, generator=video_reference_generator)
        audio_noise = torch.randn(audio_target.shape, generator=audio_reference_generator)
        video_expected_clean = video_target + video_sigmas[1] * video_prediction
        audio_expected_clean = audio_target + audio_sigmas[1] * audio_prediction
        video_expected = (
            (1.0 - video_sigmas[2]) * video_expected_clean
            + video_sigmas[2] * video_noise
        )
        audio_expected = (
            (1.0 - audio_sigmas[2]) * audio_expected_clean
            + audio_sigmas[2] * audio_noise
        )

        video_actual = pika_flex_step(
            video_target,
            video_prediction,
            video_sigmas[1],
            video_sigmas[2],
            video_generator,
        )
        audio_actual = pika_flex_step(
            audio_target,
            audio_prediction,
            audio_sigmas[1],
            audio_sigmas[2],
            audio_generator,
        )

        self.assertTrue(torch.equal(video_actual, video_expected))
        self.assertTrue(torch.equal(audio_actual, audio_expected))
        self.assertTrue(torch.equal(ref_rows, refs_before))

    def test_seeded_generators_are_reproducible_and_isolated(self):
        torch = self.torch
        sample = torch.zeros((3, 2), dtype=torch.float32)
        prediction = torch.ones_like(sample)

        def run(seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return pika_flex_step(sample, prediction, 0.8, 0.5, generator)

        self.assertTrue(torch.equal(run(103), run(103)))
        self.assertFalse(torch.equal(run(103), run(104)))

    def test_terminal_step_returns_clean_and_preserves_rng_state(self):
        torch = self.torch
        sample = torch.tensor([[1.0, -0.25]], dtype=torch.float16)
        prediction = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(19)
        before = generator.get_state().clone()

        actual = pika_flex_step(sample, prediction, 0.125, 0.0, generator)
        expected = sample.float() + 0.125 * prediction.float()

        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(generator.get_state(), before))

    def test_bfloat16_model_latents_keep_fp32_sampler_state(self):
        torch = self.torch
        sample = torch.tensor([[0.133, -1.257]], dtype=torch.bfloat16)
        prediction = torch.tensor([[0.013, 0.527]], dtype=torch.bfloat16)
        generator = torch.Generator(device="cpu").manual_seed(2026)
        reference_generator = torch.Generator(device="cpu").manual_seed(2026)
        sigma, next_sigma = 0.6, 0.3

        clean = sample.float() + sigma * prediction.float()
        noise = torch.randn(sample.shape, dtype=torch.float32, generator=reference_generator)
        expected = (1.0 - next_sigma) * clean + next_sigma * noise
        actual = pika_flex_step(sample, prediction, sigma, next_sigma, generator)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_step_rejects_invalid_shapes_sigma_and_missing_generator(self):
        torch = self.torch
        sample = torch.zeros((2, 2), dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(1)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            pika_flex_step(sample, torch.zeros((2, 1)), 0.5, 0.25, generator)
        with self.assertRaisesRegex(ValueError, "nonincreasing"):
            pika_flex_step(sample, sample, 0.25, 0.5, generator)
        with self.assertRaisesRegex(ValueError, "explicit"):
            pika_flex_step(sample, sample, 0.5, 0.25, None)


class PikaFlexRequestTests(unittest.TestCase):
    class TensorStub:
        def __init__(self, shape):
            self.shape = shape

        def numel(self):
            import math

            return math.prod(self.shape)

    def test_valid_recipe_and_each_conditioning_surface(self):
        references_request = {
            "num_inference_steps": 6,
            "guidance_scale": 1.0,
            "flow_shift": 12,
            "audio_flow_shift": 3,
            "minimax_h3_references": [{"path": "ref.png"}],
        }
        original = copy.deepcopy(references_request)
        validate_pika_flex_request(references_request, allow_references=True)
        self.assertEqual(references_request, original)

        validate_pika_flex_request({})  # Model-owned defaults may be omitted.
        validate_pika_flex_request(
            {
                "image_start": "first.png",
                "image_end": "last.png",
                "prefix_frames_count": 1,
                "window_start_frame_no": 0,
                "input_video": None,
                "frames_to_inject": None,
            },
            allow_frames=True,
            allow_references=False,
        )

    def test_only_a_single_start_frame_can_explain_the_first_window_prefix(self):
        class OneFrameVideo:
            shape = (3, 1, 32, 32)

            @staticmethod
            def numel():
                return 3 * 32 * 32

        request = {
            "image_start": "first.png",
            "input_video": None,
            "prefix_frames_count": 1,
            "window_start_frame_no": 0,
        }
        validate_pika_flex_request(request, allow_frames=True)

        valid_anchor = {
            "image_start": "first.png",
            "input_video": OneFrameVideo(),
            "prefix_frames_count": 1,
            "window_start_frame_no": 0,
        }
        self.assertTrue(
            is_pika_flex_single_frame_start_anchor(
                **valid_anchor,
                allow_frames=True,
            )
        )
        for request in (
            {
                **valid_anchor,
                "input_video": type("TwoFrameVideo", (), {
                    "shape": (3, 2, 32, 32),
                    "numel": staticmethod(lambda: 3 * 2 * 32 * 32),
                })(),
            },
            {**valid_anchor, "prefix_frames_count": 2},
            {
                **valid_anchor,
                "input_video": type("InvalidChannels", (), {
                    "shape": (4, 1, 32, 32),
                    "numel": staticmethod(lambda: 4 * 32 * 32),
                })(),
            },
            {**valid_anchor, "window_start_frame_no": 1},
            {**valid_anchor, "image_start": None},
        ):
            self.assertFalse(
                is_pika_flex_single_frame_start_anchor(
                    **request,
                    allow_frames=True,
                )
            )
        self.assertFalse(
            is_pika_flex_single_frame_start_anchor(
                **valid_anchor,
                allow_frames=False,
            )
        )

    def test_prefix_exception_still_requires_the_runtime_to_remove_video_carrier(self):
        with self.assertRaisesRegex(ValueError, "continuation"):
            validate_pika_flex_request(
                {
                    "image_start": "first.png",
                    "input_video": object(),
                    "prefix_frames_count": 1,
                    "window_start_frame_no": 0,
                },
                allow_frames=True,
            )
        for request in (
            {"image_start": "first.png", "prefix_frames_count": 2},
            {
                "image_start": "first.png",
                "prefix_frames_count": 1,
                "window_start_frame_no": 1,
            },
        ):
            with self.subTest(request=request), self.assertRaisesRegex(
                ValueError, "continuation"
            ):
                validate_pika_flex_request(request, allow_frames=True)

    def test_explicit_sequences_validate_native_prefix_and_retain_reference_inputs(self):
        video_prefix = self.TensorStub((3, 18, 480, 832))
        # One-frame of PCM rounding shortfall is tolerated by the native H3
        # audio-tail contract; WGP slices the requested duration from output.
        audio_prefix = self.TensorStub((24_000 - 400, 2))
        frames_request = {
            "minimax_h3_multi_window": True,
            "window_start_frame_no": 106,
            "frame_num": 124,
            "prefix_frames_count": 18,
            "input_video": video_prefix,
            "input_waveform": audio_prefix,
            "input_waveform_sample_rate": 32_000,
        }
        validate_pika_flex_request(
            frames_request,
            allow_frames=True,
            allow_continuation=True,
        )
        self.assertEqual(124 - (18 - 1), 107)
        self.assertEqual((107 - 5) % 17, 0)

        references = [{"type": "image", "path": "identity.png"}]
        references_request = {
            "minimax_h3_reference_sequence": True,
            "window_start_frame_no": 106,
            "frame_num": 124,
            "prefix_frames_count": 18,
            "input_video": video_prefix,
            "input_waveform": audio_prefix,
            "input_waveform_sample_rate": 32_000,
            "minimax_h3_references": references,
        }
        validate_pika_flex_request(
            references_request,
            allow_references=True,
            allow_continuation=True,
        )
        self.assertIs(references_request["minimax_h3_references"], references)

    def test_sequence_prefix_rejects_bad_video_audio_and_role_flags(self):
        base = {
            "minimax_h3_multi_window": True,
            "window_start_frame_no": 106,
            "frame_num": 124,
            "prefix_frames_count": 18,
            "input_video": self.TensorStub((3, 18, 480, 832)),
            "input_waveform": self.TensorStub((24_000, 2)),
            "input_waveform_sample_rate": 32_000,
        }
        invalid = (
            ({**base, "input_video": self.TensorStub((3, 17, 480, 832))}, "frame count"),
            ({**base, "input_waveform": self.TensorStub((24_000, 1))}, "stereo channels"),
            ({**base, "input_waveform_sample_rate": 0}, "positive sample rate"),
            ({
                **base,
                "prefix_frames_count": 19,
                "input_video": self.TensorStub((3, 19, 480, 832)),
                "input_waveform": self.TensorStub((25_333, 2)),
            }, "17\\*n\\+1"),
            ({**base, "input_waveform": self.TensorStub((20_000, 2))}, "length"),
            ({**base, "frame_num": 242}, "17\\*n\\+5"),
        )
        for request, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_pika_flex_request(
                    request,
                    allow_frames=True,
                    allow_continuation=True,
                )

        with self.assertRaisesRegex(ValueError, "References uses"):
            pika_flex_sequence_enabled(base, reference=True)

    def test_sequence_requires_first_window_without_prefix_and_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "explicit sequence"):
            validate_pika_flex_request(
                {
                    "window_start_frame_no": 106,
                    "frame_num": 124,
                    "prefix_frames_count": 18,
                    "input_video": self.TensorStub((3, 18, 480, 832)),
                    "input_waveform": self.TensorStub((24_000, 2)),
                    "input_waveform_sample_rate": 32_000,
                },
                allow_frames=True,
            )
        validate_pika_flex_request(
            {
                "minimax_h3_multi_window": True,
                "window_start_frame_no": 0,
                "frame_num": 124,
                "prefix_frames_count": 0,
            },
            allow_frames=True,
            allow_continuation=True,
        )

    def test_rejects_wrong_recipe_values(self):
        cases = (
            ({"num_inference_steps": 5}, "num_inference_steps=6"),
            ({"sampling_steps": 8}, "sampling_steps=6"),
            ({"guidance_scale": 2.0}, "guidance_scale=1"),
            ({"cfg_scale": 0.5}, "cfg_scale=1"),
            ({"flow_shift": 11}, "flow_shift=12"),
            ({"audio_flow_shift": 4}, "audio_flow_shift=3"),
            ({"guidance_phases": 2}, "guidance_phases=1"),
            ({"denoising_strength": 0.9}, "denoising_strength=1"),
            ({"masking_strength": 0.5}, "masking_strength=1"),
        )
        for request, message in cases:
            with self.subTest(request=request), self.assertRaisesRegex(ValueError, message):
                validate_pika_flex_request(request)

    def test_rejects_unsupported_accelerators_continuation_and_controls(self):
        cases = (
            ({"activated_loras": ["character.safetensors"]}, "LoRAs"),
            ({"loras_selected": ["character.safetensors"]}, "LoRAs"),
            ({"minimax_h3_turbo_mode": True}, "accelerators"),
            ({"skip_steps_cache_type": "classic"}, "step-skipping caches"),
            ({"minimax_h3_multi_window": True}, "continuation"),
            ({"prefix_frames_count": 17}, "continuation"),
            ({"window_start_frame_no": 1}, "continuation"),
            ({"minimax_h3_reference_sequence": True}, "continuation"),
            ({"custom_settings": {"h3_two_pass_enabled": "on"}}, "Two-Pass"),
            ({"custom_settings": {"audio_refinement": "enabled"}}, "Audio Refinement"),
            ({"video_guide": "control.mp4"}, "control video"),
            ({"video_prompt_type": "GV"}, "control video"),
            ({"frames_to_inject": ["frame.png"]}, "control video"),
            ({"input_masks": ["mask.png"]}, "control video"),
            ({"input_video": "source.mp4"}, "source video/audio|control video"),
            ({"audio_guide": "source.wav"}, "control video"),
            ({"face_refinement": {"strength": 0.5}}, "face refinement"),
        )
        for request, message in cases:
            with self.subTest(request=request), self.assertRaisesRegex(ValueError, message):
                validate_pika_flex_request(request)

    def test_first_last_frames_are_opt_in_and_references_are_separately_gated(self):
        with self.assertRaisesRegex(ValueError, "first/last frame"):
            validate_pika_flex_request({"image_start": "first.png"})
        with self.assertRaisesRegex(ValueError, "image references"):
            validate_pika_flex_request({"image_refs": ["ref.png"]})
        with self.assertRaisesRegex(ValueError, "image references"):
            validate_pika_flex_request(
                {"image_refs": ["ref.png"]}, allow_references=False
            )


if __name__ == "__main__":
    unittest.main()
