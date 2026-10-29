"""CPU-only integration checks for the native Pika H3 Flex runtime wiring."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class PikaFlexRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("Pika H3 Flex runtime tests require PyTorch") from error

        cls.torch = torch
        main_path = APP / "models" / "minimax_h3" / "minimax_h3_main.py"
        tree = ast.parse(main_path.read_text(encoding="utf-8"))
        names = {
            "_h3_video_scheduler_step",
            "_pika_flex_update_generated_audio_rows",
            "_set_pika_flex_schedules",
            "_verify_pika_flex_base_checkpoint",
        }
        selected = {}
        model_methods = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                selected[node.name] = node
            elif isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Model":
                model_methods = {
                    item.name: item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                    and item.name in {"validate_loras", "generate"}
                }

        runtime_namespace = {
            "__name__": "models.minimax_h3.pika_flex_runtime_test",
            "__package__": "models.minimax_h3",
            "torch": torch,
            "MINIMAX_H3_FPS": 24,
        }
        cls.generate_ast = model_methods["generate"]
        for name, node in {**selected, **model_methods}.items():
            code = compile(ast.Module(body=[node], type_ignores=[]), str(main_path), "exec")
            exec(code, runtime_namespace)

        class MiniMaxH3ModelStub:
            def release_special_loras(self):
                self._pdd_controller = None

            validate_loras = runtime_namespace["validate_loras"]
            generate = runtime_namespace["generate"]

        class CpuScheduler:
            def __init__(self, shift=12.0, solver="euler"):
                self.shift = shift
                self.solver = solver
                self.sigmas = None
                self.timesteps = None

            def set_shift(self, shift):
                self.shift = float(shift)

            def set_solver(self, solver):
                self.solver = solver

            def set_timesteps(self, *, sigmas, device):
                self.sigmas = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
                self.timesteps = 1.0 - self.sigmas[:-1]

        cls.MiniMaxH3Model = MiniMaxH3ModelStub
        cls.h3_video_scheduler_step = staticmethod(
            runtime_namespace["_h3_video_scheduler_step"]
        )
        cls.pika_update_audio = staticmethod(
            runtime_namespace["_pika_flex_update_generated_audio_rows"]
        )
        cls.set_schedules = staticmethod(runtime_namespace["_set_pika_flex_schedules"])
        cls.verify_base_checkpoint = staticmethod(
            runtime_namespace["_verify_pika_flex_base_checkpoint"]
        )
        cls.Scheduler = CpuScheduler

    def _flex_model(self, *, reference=False):
        model = self.MiniMaxH3Model.__new__(self.MiniMaxH3Model)
        model.pika_flex = True
        model.omni_reference = reference
        model._turbo_lora_active = True
        model._turbo_lora_paths = ("stale-turbo.safetensors",)
        model._pdd_lora_active = True
        model._pdd_lora_path = "stale-pdd.safetensors"
        model._pdd_lora_strength = 0.4
        model._ptd_lora_paths = ("stale-ptd.safetensors",)
        return model

    def test_native_duration_guard_accepts_flex_362_frame_endpoint_only(self):
        body = self.generate_ast.body
        start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "duration" for t in node.targets))
        end = next(i for i in range(start, len(body))
                   if "selected duration limit" in ast.unparse(body[i]))
        code = compile(ast.Module(body=body[start:end + 1], type_ignores=[]), "native-duration", "exec")
        for pika, frames, valid in ((True, 362, True), (True, 363, False),
                                    (False, 362, False), (False, 345, True)):
            with self.subTest(pika=pika, frames=frames):
                namespace = {
                    "__package__": "models.minimax_h3", "math": math,
                    "self": SimpleNamespace(pika_flex=pika, _pdd_lora_active=False,
                                            viggle=False, audio_only=False),
                    "frame_num": frames, "fps": 24, "_face_refinement": None,
                    "minimax_h3_extended_duration": False, "MINIMAX_H3_FPS": 24,
                    "MINIMAX_H3_MIN_DURATION": 5, "MINIMAX_H3_MAX_DURATION": 15,
                }
                if valid:
                    exec(code, namespace)
                else:
                    with self.assertRaisesRegex(ValueError, "selected duration limit"):
                        exec(code, namespace)

    def test_adapter_validation_uses_the_model_owned_frames_or_references_pin(self):
        from models.minimax_h3.pika_flex_assets import (
            PIKA_FLEX_LORA_SIZE,
            PIKA_FLEX_LORA_SPECS,
        )

        for reference in (False, True):
            with self.subTest(reference=reference):
                model = self._flex_model(reference=reference)
                pin = PIKA_FLEX_LORA_SPECS[reference]
                path = str(Path("models") / pin["file"])
                with patch(
                    "models.minimax_h3.fasth3.verify_checkpoint_pin"
                ) as verify_pin:
                    model.validate_loras([path])

                verify_pin.assert_called_once_with(
                    path,
                    PIKA_FLEX_LORA_SIZE,
                    pin["sha256"],
                    pin["label"],
                )
                self.assertFalse(model._turbo_lora_active)
                self.assertEqual(model._turbo_lora_paths, ())
                self.assertFalse(model._pdd_lora_active)
                self.assertIsNone(model._pdd_lora_path)
                self.assertEqual(model._pdd_lora_strength, 1.0)
                self.assertEqual(model._ptd_lora_paths, ())

    def test_extra_adapter_fails_before_any_checkpoint_hash_read(self):
        model = self._flex_model(reference=False)
        with patch(
            "models.minimax_h3.fasth3.verify_checkpoint_pin"
        ) as verify_pin:
            with self.assertRaisesRegex(ValueError, "exactly its one pinned adapter"):
                model.validate_loras(
                    ["pika_h3_flex_fl2va_comfy_bf16.safetensors", "character.safetensors"]
                )
        verify_pin.assert_not_called()

    def test_pinned_base_checkpoint_check_rejects_multiple_files_before_verification(self):
        with patch(
            "models.minimax_h3.pika_flex_assets.verify_pika_flex_checkpoint"
        ) as verify_checkpoint:
            with self.assertRaisesRegex(ValueError, "one pinned base transformer"):
                self.verify_base_checkpoint(
                    ["first.safetensors", "second.safetensors"],
                    "first.safetensors",
                    reference=True,
                )
            verify_checkpoint.assert_not_called()

            self.verify_base_checkpoint(
                "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
                "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
                reference=True,
            )
        verify_checkpoint.assert_called_once_with(
            "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
            reference=True,
        )

    def test_video_and_audio_schedulers_get_distinct_six_step_ladders(self):
        from models.minimax_h3.pika_flex import pika_flex_sigmas

        video = self.Scheduler(shift=7.0, solver="res_multistep")
        audio = self.Scheduler(shift=8.0, solver="res_multistep")
        self.set_schedules(video, audio, "cpu")

        self.assertEqual(video.solver, "euler")
        self.assertEqual(audio.solver, "euler")
        self.assertEqual(video.shift, 12.0)
        self.assertEqual(audio.shift, 3.0)
        self.assertEqual(video.timesteps.numel(), 6)
        self.assertEqual(audio.timesteps.numel(), 6)
        self.assertTrue(
            self.torch.equal(
                video.sigmas,
                self.torch.tensor(pika_flex_sigmas(12.0), dtype=self.torch.float32),
            )
        )
        self.assertTrue(
            self.torch.equal(
                audio.sigmas,
                self.torch.tensor(pika_flex_sigmas(3.0), dtype=self.torch.float32),
            )
        )

    def test_video_update_and_preview_touch_only_generated_target_rows(self):
        torch = self.torch

        video = self.Scheduler()
        self.set_schedules(video, self.Scheduler(), "cpu")
        index = 2
        packed_rows = torch.tensor(
            [[50.0, 51.0], [60.0, 61.0], [0.25, -0.5], [1.0, 0.125]],
            dtype=torch.float32,
        )
        reference_prefix_before = packed_rows[:2].clone()
        target_before = packed_rows[2:].clone()
        prediction = torch.tensor([[0.1, 0.2], [-0.3, 0.4]], dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(12)
        expected_generator = torch.Generator(device="cpu").manual_seed(12)
        sigma = float(video.sigmas[index].item())
        next_sigma = float(video.sigmas[index + 1].item())
        clean = target_before + sigma * prediction
        noise = torch.randn(
            target_before.shape,
            generator=expected_generator,
            dtype=torch.float32,
        )
        expected = (1.0 - next_sigma) * clean + next_sigma * noise

        class PreviewCallback:
            @staticmethod
            def wants_preview(step_index):
                return step_index == index

        updated, preview_clean = self.h3_video_scheduler_step(
            video,
            prediction,
            video.timesteps[index],
            packed_rows[2:],
            callback=PreviewCallback(),
            index=index,
            audio_only=False,
            frozen_target_video=None,
            generated_video_row_count=2,
            pika_noise_generator=generator,
        )
        packed_rows[2:] = updated

        self.assertTrue(torch.equal(packed_rows[:2], reference_prefix_before))
        self.assertTrue(torch.equal(packed_rows[2:], expected))
        self.assertTrue(torch.equal(preview_clean, clean))
        self.assertTrue(torch.equal(generator.get_state(), expected_generator.get_state()))

    def test_audio_update_preserves_reference_prefix_and_target_condition_rows(self):
        torch = self.torch

        audio = self.Scheduler()
        self.set_schedules(self.Scheduler(), audio, "cpu")
        index = 3
        packed_audio = torch.tensor(
            [
                [90.0, 91.0],
                [92.0, 93.0],
                [10.0, 11.0],  # first-channel target condition
                [0.1, 0.2],    # generated audio
                [12.0, 13.0],  # second-channel target condition
                [-0.2, 0.3],   # generated audio
            ],
            dtype=torch.float32,
        )
        prefix_before = packed_audio[:2].clone()
        target_before = packed_audio[2:].clone()
        prediction = torch.tensor(
            [[0.8, 0.7], [0.1, -0.2], [-0.1, 0.3], [0.4, 0.5]],
            dtype=torch.float32,
        )
        generated_indices = torch.tensor([1, 3], dtype=torch.long)
        generator = torch.Generator(device="cpu").manual_seed(201)
        expected_generator = torch.Generator(device="cpu").manual_seed(201)
        sigma = float(audio.sigmas[index].item())
        next_sigma = float(audio.sigmas[index + 1].item())
        sample = target_before[generated_indices]
        model_output = prediction[generated_indices]
        clean = sample + sigma * model_output
        noise = torch.randn(
            sample.shape,
            generator=expected_generator,
            dtype=torch.float32,
        )
        expected_generated = (1.0 - next_sigma) * clean + next_sigma * noise

        self.pika_update_audio(
            packed_audio[2:],
            prediction,
            generated_indices,
            audio,
            index,
            generator,
        )

        self.assertTrue(torch.equal(packed_audio[:2], prefix_before))
        self.assertTrue(torch.equal(packed_audio[2], target_before[0]))
        self.assertTrue(torch.equal(packed_audio[4], target_before[2]))
        self.assertTrue(torch.equal(packed_audio[3], expected_generated[0]))
        self.assertTrue(torch.equal(packed_audio[5], expected_generated[1]))
        self.assertTrue(torch.equal(generator.get_state(), expected_generator.get_state()))

    def test_request_validation_runs_before_the_two_pass_branch(self):
        model = self._flex_model(reference=False)
        with self.assertRaisesRegex(ValueError, "does not support H3 Two-Pass"):
            model.generate(
                "a test prompt",
                sampling_steps=6,
                shift="",
                custom_settings={"h3_two_pass_enabled": "on"},
            )

    def test_request_validation_rejects_wrong_steps_and_source_audio(self):
        model = self._flex_model(reference=True)
        with self.assertRaisesRegex(ValueError, "sampling_steps=6"):
            model.generate("a test prompt", sampling_steps=5)
        with self.assertRaisesRegex(ValueError, "masking_strength=1"):
            model.generate(
                "a test prompt",
                sampling_steps=6,
                masking_strength=0.5,
            )
        with self.assertRaisesRegex(ValueError, "source-audio prompt modes"):
            model.generate(
                "a test prompt",
                sampling_steps=6,
                audio_prompt_type="A",
            )

    def test_wgp_one_frame_start_carrier_does_not_become_a_continuation(self):
        model = self._flex_model(reference=False)
        start_image = self.torch.zeros((3, 8, 8), dtype=self.torch.float32)
        # WGP's input_video carrier is C,T,H,W; image_start is C,H,W.
        # H3's explicit-start branch discards this one-frame carrier.
        start_carrier = self.torch.ones((3, 1, 8, 8), dtype=self.torch.float32)
        with self.assertRaisesRegex(ValueError, "does not support H3 Two-Pass"):
            model.generate(
                "a test prompt",
                sampling_steps=6,
                image_start=start_image,
                image_end=start_image,
                input_video=start_carrier,
                prefix_frames_count=1,
                window_start_frame_no=0,
                frames_to_inject=None,
                custom_settings={"h3_two_pass_enabled": "on"},
            )

    def test_wgp_prefix_allowance_does_not_open_later_windows_or_video_guides(self):
        model = self._flex_model(reference=False)
        start_image = self.torch.zeros((3, 8, 8), dtype=self.torch.float32)
        start_carrier = self.torch.zeros((3, 1, 8, 8), dtype=self.torch.float32)
        with self.assertRaisesRegex(ValueError, "continuation"):
            model.generate(
                "a test prompt",
                sampling_steps=6,
                image_start=start_image,
                input_video=start_carrier,
                prefix_frames_count=1,
                window_start_frame_no=17,
            )
        with self.assertRaisesRegex(ValueError, "control video"):
            model.generate(
                "a test prompt",
                sampling_steps=6,
                image_start=start_image,
                input_video=start_carrier,
                prefix_frames_count=1,
                window_start_frame_no=0,
                video_prompt_type="GV",
            )


if __name__ == "__main__":
    unittest.main()
