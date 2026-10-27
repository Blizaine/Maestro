"""CPU-only tests for the H3 two-pass toggle and phase-two helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import types
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class TestH3TwoPassHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.minimax_h3 import two_pass

        cls.two_pass = two_pass

    def test_toggle_accepts_only_explicit_boolean_or_off_on_values(self):
        parse = self.two_pass.parse_two_pass_toggle
        for value in (None, False, "off", " OFF "):
            with self.subTest(value=value):
                self.assertFalse(parse(value))
        for value in (True, "on", " ON "):
            with self.subTest(value=value):
                self.assertTrue(parse(value))
        for value in ("", "false", "true", "yes", 0, 1, 2, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse(value)

    def test_defaults_bounds_and_sigma_grid(self):
        normalize = self.two_pass.normalize_two_pass_settings
        self.assertEqual(
            normalize({}),
            {
                "h3_two_pass_enabled": "off",
                "h3_phase2_steps": 3,
                "h3_phase2_noise": 0.9035,
                "h3_phase2_accelerator_strength": 1.0,
            },
        )
        self.assertEqual(
            normalize(
                {
                    "h3_two_pass_enabled": True,
                    "h3_phase2_steps": 12,
                    "h3_phase2_noise": 0.05,
                    "h3_phase2_accelerator_strength": 2,
                }
            )["h3_two_pass_enabled"],
            "on",
        )
        self.assertEqual(
            normalize(
                {
                    "h3_phase2_steps": 2,
                    "h3_phase2_noise": 1,
                    "h3_phase2_accelerator_strength": 0,
                }
            )["h3_two_pass_enabled"],
            "off",
        )
        for key, value in (
            ("h3_phase2_steps", 1),
            ("h3_phase2_steps", 13),
            ("h3_phase2_steps", 2.5),
            ("h3_phase2_steps", True),
            ("h3_phase2_noise", 0.049),
            ("h3_phase2_noise", 1.001),
            ("h3_phase2_noise", float("nan")),
            ("h3_phase2_accelerator_strength", -0.1),
            ("h3_phase2_accelerator_strength", 2.01),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    normalize({key: value})

        sigmas = self.two_pass.phase2_sigmas(3, 0.9)
        self.assertEqual(len(sigmas), 4)
        self.assertAlmostEqual(sigmas[0], 0.9)
        self.assertAlmostEqual(sigmas[-1], 0.0)
        self.assertTrue(all(left > right for left, right in zip(sigmas, sigmas[1:])))

    def test_effective_sampler_resolves_implicit_values_but_preserves_explicit_choice(self):
        resolve = self.two_pass.resolve_h3_sample_solver
        recipe = {"minimax_h3_sampler": "euler"}
        for requested in (None, "", "   ", "default", " DEFAULT "):
            with self.subTest(requested=requested):
                self.assertEqual(resolve(requested, recipe), "euler")
        self.assertEqual(resolve("default", {}), "euler")
        self.assertEqual(resolve("unipc", recipe), "unipc")
        self.assertEqual(
            resolve("default", {"minimax_h3_sampler": "default"}),
            "default",
        )
        self.assertEqual(
            resolve("default", {"minimax_h3_sampler": "res_multistep"}),
            "res_multistep",
        )

    def test_default_mask_and_outpaint_controls_are_not_treated_as_active(self):
        detect = self.two_pass.has_h3_edit_mask_or_outpainting
        for inputs in (
            {
                "video_mask": None,
                "image_mask": "",
                "input_masks": [],
                "outpainting_dims": None,
                "video_guide_outpainting": "# disabled: 0 0 0 0",
            },
            {
                "video_mask": None,
                "image_mask": None,
                "input_masks": [None, ""],
                "outpainting_dims": [0, 0, 0, 0],
                "video_guide_outpainting": "0 0 0 0",
            },
            {
                "video_mask": None,
                "image_mask": None,
                "input_masks": [],
                "outpainting_dims": [],
                "video_guide_outpainting": "",
            },
            {
                "video_mask": "",
                "image_mask": "",
                "input_masks": "",
                "outpainting_dims": [0, 0, 0, 0],
                "video_guide_outpainting": "# disabled",
            },
        ):
            with self.subTest(inputs=inputs):
                self.assertFalse(detect(inputs))

    def test_real_masks_and_outpaint_margins_are_still_detected(self):
        detect = self.two_pass.has_h3_edit_mask_or_outpainting
        for inputs in (
            {"video_mask": "mask.mp4"},
            {"image_mask": "mask.png"},
            {"input_masks": ["mask.png"]},
            {"outpainting_dims": [0, 12, 0, 0]},
            {"video_guide_outpainting": "0 12 0 0"},
        ):
            with self.subTest(inputs=inputs):
                self.assertTrue(detect(inputs))

    def test_half_size_rounding_and_phase_two_loras_do_not_mutate_settings(self):
        self.assertEqual(self.two_pass.phase1_dimensions(512, 864), (256, 448))
        schedules = {
            "phase1": [0.8, 0.2],
            "phase2": [0.6, 0.3],
            "phase3": [0.5, 0.4],
            "shared": [False, False],
        }
        original = copy.deepcopy(schedules)
        adjusted = self.two_pass.phase2_lora_schedule(
            schedules,
            ["turbo.safetensors", "character.safetensors"],
            ["turbo.safetensors"],
            0.45,
        )
        self.assertEqual(schedules, original)
        self.assertEqual(adjusted["phase1"], [0.45, 0.3])
        self.assertEqual(adjusted["phase2"], [0.45, 0.3])
        self.assertEqual(adjusted["phase3"], [0.45, 0.3])
        self.assertEqual(adjusted["shared"], [True, True])

    def test_generate_request_filters_helper_locals_and_keeps_real_arguments(self):
        from inspect import signature

        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        helper = object()
        schedules = {
            "phase1": [0.5],
            "phase2": [0.4],
            "phase3": [0.3],
            "shared": [False],
        }
        request = self.two_pass.build_two_pass_generate_request(
            {
                "input_prompt": "a short clip",
                "height": 512,
                "width": 864,
                "custom_settings": {"h3_two_pass_enabled": "on"},
                "H3_TWO_PASS_ENABLED": helper,
                "normalize_two_pass_settings": helper,
                "phase2_clean_audio_timesteps": helper,
                "two_pass_settings": helper,
                "request": helper,
                "_kwargs": {
                    "guide_scale": 1.0,
                    "loras_slists": schedules,
                    "loras_selected": ["character.safetensors"],
                },
            },
            signature(MiniMaxH3Model.generate),
        )
        self.assertEqual(request["input_prompt"], "a short clip")
        self.assertEqual(request["height"], 512)
        self.assertEqual(request["width"], 864)
        self.assertEqual(request["custom_settings"]["h3_two_pass_enabled"], "on")
        self.assertEqual(request["guide_scale"], 1.0)
        self.assertIs(request["loras_slists"], schedules)
        self.assertEqual(request["loras_selected"], ["character.safetensors"])
        for helper_name in (
            "H3_TWO_PASS_ENABLED",
            "normalize_two_pass_settings",
            "phase2_clean_audio_timesteps",
            "two_pass_settings",
            "request",
        ):
            self.assertNotIn(helper_name, request)

    def test_frozen_phase_two_audio_uses_clean_scheduler_time(self):
        import torch

        video_timesteps = torch.tensor([0.904, 0.452, 0.0], dtype=torch.float32)
        audio_timesteps = self.two_pass.phase2_clean_audio_timesteps(video_timesteps)
        self.assertEqual(audio_timesteps.shape, video_timesteps.shape)
        self.assertEqual(audio_timesteps.dtype, video_timesteps.dtype)
        self.assertEqual(audio_timesteps.device, video_timesteps.device)
        self.assertTrue(torch.equal(audio_timesteps, torch.ones_like(video_timesteps)))

    def test_two_pass_handoff_freezes_first_pass_audio_latents(self):
        import torch
        import torch.nn.functional as F

        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        phase1_video = torch.randn(1, 24, 2, 4, 4)
        phase1_audio = torch.randn(1, 8, 5)
        calls = []

        model = object.__new__(MiniMaxH3Model)
        model.sample_solver = "euler"
        model.model_def = {}
        model._turbo_lora_active = True
        model._pdd_lora_active = False
        model.ptd_refasymopd = False
        model.audio_only = False
        model.viggle = False
        model._fused_turbo = False
        model.fasth3 = False
        model.vdn = False
        model._baked_turbo = False
        model.dasiwa = False
        model.singularity = False
        model.omni_reference = False
        model.VAE_upsampling = None
        model._interrupt = False
        model._turbo_lora_paths = []
        model.vae = types.SimpleNamespace(spatial_compression_ratio=8)
        model.latent_upscaler = object()
        real_generate = MiniMaxH3Model.generate.__get__(model, MiniMaxH3Model)
        phase2_validation_reached = []

        class _AfterTurboValidation(Exception):
            pass

        def fake_generate(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return {
                    "_h3_latents": True,
                    "video_latents": phase1_video,
                    "audio_latents": phase1_audio,
                }
            # Exercise the production second-pass generate path through Turbo's
            # step-count validation, then stop before model input construction.
            with mock.patch(
                "models.minimax_h3.minimax_h3_main._split_continuation_video",
                side_effect=_AfterTurboValidation,
            ):
                with self.assertRaises(_AfterTurboValidation):
                    real_generate(**kwargs)
            phase2_validation_reached.append(True)
            return "phase2-result"

        model.generate = fake_generate

        def fake_upscale(value, _scale, *, target_size):
            return F.interpolate(value, size=target_size, mode="trilinear", align_corners=False)

        request = {
            "input_prompt": "a short clip",
            "height": 64,
            "width": 64,
            "sampling_steps": 20,
            "seed": 17,
            "denoising_strength": 1.0,
            "custom_settings": {"h3_two_pass_enabled": "on"},
            "video_prompt_type": "TSEV",
            "input_video": None,
            "input_masks": None,
            "outpainting_dims": None,
            "_audio_segment": False,
        }
        with mock.patch(
            "models.minimax_h3.minimax_h3_main.runtime.execution_device",
            return_value="cpu",
        ), mock.patch.object(model, "latent_upscaler", side_effect=fake_upscale):
            result = model._run_h3_two_pass(
                request,
                {
                    "h3_two_pass_enabled": "on",
                    "h3_phase2_steps": 3,
                    "h3_phase2_noise": 0.9,
                    "h3_phase2_accelerator_strength": 1.0,
                },
            )

        self.assertEqual(result, "phase2-result")
        self.assertEqual(len(calls), 2)
        self.assertEqual((calls[0]["height"], calls[0]["width"]), (32, 32))
        self.assertEqual((calls[1]["height"], calls[1]["width"]), (64, 64))
        self.assertNotIn("_h3_phase2_validation_token", calls[0])
        from models.minimax_h3 import minimax_h3_main

        self.assertIs(
            calls[1]["_h3_phase2_validation_token"],
            minimax_h3_main._H3_TWO_PASS_PHASE2_VALIDATION_TOKEN,
        )
        self.assertEqual(phase2_validation_reached, [True])
        self.assertTrue(calls[0]["_h3_capture_latents"])
        self.assertTrue(calls[1]["_h3_freeze_audio"])
        self.assertFalse(calls[1]["_h3_capture_latents"])
        self.assertTrue(
            torch.equal(calls[1]["_h3_initial_audio_latents"], phase1_audio)
        )

    def test_turbo_minimum_cannot_be_bypassed_by_phase_one_or_caller_markers(self):
        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        model = object.__new__(MiniMaxH3Model)
        model._turbo_lora_active = True
        model._turbo_lora_paths = []
        model._pdd_lora_active = False
        model.ptd_refasymopd = False
        model.sample_solver = "euler"
        model.fasth3 = False
        model.audio_only = False
        model.viggle = False
        model._interrupt = False
        model._baked_turbo = False
        model.model_def = {}
        model._fused_turbo = False

        request = {
            "input_prompt": "a short clip",
            "frame_num": 124,
            "height": 64,
            "width": 64,
            "sampling_steps": 3,
            "custom_settings": {"h3_two_pass_enabled": "off"},
        }
        internal_phase1 = {**request, "_h3_two_pass_internal": True}
        incomplete_phase2 = {
            **internal_phase1,
            "_h3_initial_video_latents": object(),
            "_h3_initial_audio_latents": object(),
            "_h3_freeze_audio": True,
        }
        forged_phase2 = {
            **incomplete_phase2,
            "_h3_phase2_validation_token": object(),
        }
        for label, candidate in (
            ("ordinary request", request),
            ("internal phase one", internal_phase1),
            ("phase two without token", incomplete_phase2),
            ("phase two with untrusted token", forged_phase2),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError, "Turbo LoRA needs at least 4 denoising steps"
                ):
                    model.generate(**candidate)

    def test_unsupported_recipes_and_workflows_are_explicit(self):
        incompatible_models = (
            {"minimax_h3_ptd_refasymopd": True},
            {"minimax_h3_fused_turbo": True},
            {"minimax_h3_fasth3": True},
            {"vdn": True},
            {"minimax_h3_dasiwa": True},
            {"minimax_h3_baked_turbo": True},
            {"minimax_h3_singularity": True},
            {"minimax_h3_audio_only": True},
        )
        for model in incompatible_models:
            with self.subTest(model=model):
                self.assertIsNotNone(
                    self.two_pass.two_pass_unavailable_reason(model)
                )
        for options in (
            {"pdd_active": True},
            {"has_control_video": True},
            {"has_source_video": True},
            {"has_edit_mask": True},
            {"denoising_strength": 0.9},
            {"denoising_strength": 1.1},
            {"denoising_strength": float("nan")},
            {"denoising_strength": "invalid"},
            {"audio_refinement": True},
            {"face_refinement": True},
            {"audio_only": True},
        ):
            with self.subTest(options=options):
                self.assertIsNotNone(
                    self.two_pass.two_pass_unavailable_reason({}, **options)
                )

    def test_handler_loads_upscaler_only_when_runtime_flag_is_true(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        definition = family_handler.query_model_def("minimax_h3_ref2va", {})
        controls = {
            item["id"]: item for item in definition["custom_settings"]
        }
        self.assertEqual(controls["h3_two_pass_enabled"]["type"], "dropdown")
        self.assertEqual(controls["h3_two_pass_enabled"]["default"], "off")
        self.assertEqual(
            controls["h3_two_pass_enabled"]["choices"],
            [("Off", "off"), ("On", "on")],
        )
        self.assertEqual(
            (controls["h3_phase2_steps"]["min"], controls["h3_phase2_steps"]["max"]),
            (2, 12),
        )
        files_off = family_handler.query_model_files(
            [], "minimax_h3_ref2va", definition
        )
        resolved_off = family_handler.resolve_runtime_model_def(
            definition, {"minimax_h3_two_pass": False}
        )
        self.assertFalse(resolved_off.get("minimax_h3_two_pass_enabled", False))
        self.assertFalse(resolved_off.get("minimax_h3_latent_upscaler_enabled", False))
        self.assertNotIn(
            "minimax_h3_latent_upscaler_3d_bf16.safetensors",
            self._flatten(files_off),
        )

        resolved_on = family_handler.resolve_runtime_model_def(
            definition, {"minimax_h3_two_pass": True}
        )
        self.assertTrue(resolved_on["minimax_h3_two_pass_enabled"])
        self.assertTrue(resolved_on["minimax_h3_latent_upscaler_enabled"])
        files_on = family_handler.query_model_files(
            [], "minimax_h3_ref2va", resolved_on
        )
        self.assertIn(
            "minimax_h3_latent_upscaler_3d_bf16.safetensors",
            self._flatten(files_on),
        )
        for invalid in ("false", 1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    family_handler.resolve_runtime_model_def(
                        definition, {"minimax_h3_two_pass": invalid}
                    )

    def test_runtime_off_clears_stale_optional_asset_identity(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        stale = {
            **family_handler.query_model_def("minimax_h3_ref2va", {}),
            "minimax_h3_two_pass_enabled": True,
            "minimax_h3_latent_upscaler_enabled": True,
            "minimax_h3_latent_upscaler_filename": "stale.safetensors",
        }
        resolved = family_handler.resolve_runtime_model_def(
            stale, {"minimax_h3_two_pass": "off"}
        )
        self.assertNotIn("minimax_h3_two_pass_enabled", resolved)
        self.assertNotIn("minimax_h3_latent_upscaler_enabled", resolved)
        self.assertNotIn("minimax_h3_latent_upscaler_filename", resolved)

    def test_exception_restores_original_lora_schedule_and_step_index(self):
        from mmgp import offload
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler
        from shared.utils.loras_mutipliers import update_loras_slists

        original_schedules = {
            "phase1": [0.4],
            "phase2": [0.4],
            "phase3": [0.4],
            "shared": [True],
        }
        transformer = type("FakeTransformer", (), {})()
        video_scheduler = MiniMaxH3Scheduler(shift=12.0, solver="euler")
        audio_scheduler = MiniMaxH3Scheduler(shift=3.0, solver="euler")
        video_scheduler.set_timesteps(20)
        audio_scheduler.set_timesteps(20)
        video_before = {
            key: value.clone() if hasattr(value, "clone") else value
            for key, value in video_scheduler.__dict__.items()
        }
        audio_before = {
            key: value.clone() if hasattr(value, "clone") else value
            for key, value in audio_scheduler.__dict__.items()
        }

        def fail_after_phase_setup(_request, _settings):
            update_loras_slists(
                transformer,
                {
                    "phase1": [0.9],
                    "phase2": [0.9],
                    "phase3": [0.9],
                    "shared": [True],
                },
                3,
            )
            offload.set_step_no_for_lora(transformer, 2)
            video_scheduler.set_solver("res_multistep")
            video_scheduler.set_timesteps(4)
            audio_scheduler.set_timesteps(4)
            raise RuntimeError("phase failure")

        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        model = object.__new__(MiniMaxH3Model)
        model.transformer = transformer
        model.scheduler = video_scheduler
        model.audio_scheduler = audio_scheduler
        model._run_h3_two_pass = fail_after_phase_setup
        with mock.patch(
            "shared.utils.loras_mutipliers.update_loras_slists",
            wraps=update_loras_slists,
        ) as update:
            with self.assertRaisesRegex(RuntimeError, "phase failure"):
                model._generate_h3_two_pass(
                    {"loras_slists": original_schedules, "sampling_steps": 20},
                    {"h3_two_pass_enabled": "on"},
                )

        restore_call = update.call_args_list[-1]
        self.assertIs(restore_call.args[1], original_schedules)
        self.assertEqual(restore_call.args[2], 20)
        self.assertEqual(transformer._loras_scaling["0"], 0.4)
        self.assertEqual(transformer._lora_step_no, 0)
        self.assertEqual(video_scheduler.solver, "euler")
        self.assertEqual(video_scheduler.shift, 12.0)
        self.assertTrue(
            (video_scheduler.timesteps == video_before["timesteps"]).all()
        )
        self.assertTrue(
            (audio_scheduler.timesteps == audio_before["timesteps"]).all()
        )

    def test_cancellation_restores_original_lora_schedule_and_scheduler_state(self):
        import torch

        from mmgp import offload
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler
        from shared.utils.loras_mutipliers import update_loras_slists
        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        original_schedules = {
            "phase1": [0.4],
            "phase2": [0.4],
            "phase3": [0.4],
            "shared": [True],
        }
        temporary_schedules = {
            "phase1": [0.9],
            "phase2": [0.9],
            "phase3": [0.9],
            "shared": [True],
        }
        transformer = type("FakeTransformer", (), {})()
        video_scheduler = MiniMaxH3Scheduler(shift=12.0, solver="euler")
        audio_scheduler = MiniMaxH3Scheduler(shift=3.0, solver="euler")
        video_scheduler.set_timesteps(20)
        audio_scheduler.set_timesteps(20)
        video_before = video_scheduler.timesteps.clone()
        audio_before = audio_scheduler.timesteps.clone()

        def cancel_after_phase_setup(_request, _settings):
            update_loras_slists(transformer, temporary_schedules, 3)
            offload.set_step_no_for_lora(transformer, 2)
            video_scheduler.set_timesteps(3)
            audio_scheduler.set_timesteps(3)
            model._interrupt = True
            return None

        model = object.__new__(MiniMaxH3Model)
        model.transformer = transformer
        model.scheduler = video_scheduler
        model.audio_scheduler = audio_scheduler
        model._run_h3_two_pass = cancel_after_phase_setup
        with mock.patch(
            "shared.utils.loras_mutipliers.update_loras_slists",
            wraps=update_loras_slists,
        ) as update:
            result = model._generate_h3_two_pass(
                {"loras_slists": original_schedules, "sampling_steps": 20},
                {"h3_two_pass_enabled": "on"},
            )

        self.assertIsNone(result)
        restore_call = update.call_args_list[-1]
        self.assertIs(restore_call.args[1], original_schedules)
        self.assertEqual(restore_call.args[2], 20)
        self.assertEqual(transformer._loras_scaling["0"], 0.4)
        self.assertEqual(transformer._lora_step_no, 0)
        self.assertTrue(torch.equal(video_scheduler.timesteps, video_before))
        self.assertTrue(torch.equal(audio_scheduler.timesteps, audio_before))
        self.assertTrue(model._interrupt)

    def test_h3_pipeline_imports_with_bundled_mmgp_without_profile0(self):
        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        self.assertTrue(callable(MiniMaxH3Model.generate))

    def test_latent_upscaler_safetensors_header_is_checked_without_loading_weights(self):
        import torch
        from safetensors.torch import save_file

        from models.minimax_h3.latent_upscaler import (
            MiniMaxH3LatentUpscaler,
            validate_latent_upscaler_header,
        )

        model = MiniMaxH3LatentUpscaler(
            in_channels=32,
            blocks=0,
            channels=32,
            temporal_every=0,
        )
        tensors = {
            key: torch.zeros(tuple(tensor.shape), dtype=tensor.dtype)
            for key, tensor in model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "header-test.safetensors"
            save_file(tensors, str(filename))
            validate_latent_upscaler_header(str(filename), model)

            tensors["conv_in.weight"] = torch.zeros(1, dtype=torch.float32)
            save_file(tensors, str(filename))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                validate_latent_upscaler_header(str(filename), model)

    def test_handler_validation_rejects_incompatible_two_pass_requests(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        definition = family_handler.query_model_def("minimax_h3_ref2va", {})
        for extra, expected in (
            ({"video_guide": "control.mp4", "video_prompt_type": "GV"}, "Control Video"),
            ({"input_video": "source.mp4"}, "source-video"),
            ({"video_mask": "mask.mp4"}, "masks"),
            ({"denoising_strength": 0.7}, "full-strength"),
            ({"activated_loras": ["MiniMax-H3-Ref2VA-Acc-8Step.safetensors"]}, "PDD"),
        ):
            inputs = {
                "custom_settings": {"h3_two_pass_enabled": "on"},
                "num_inference_steps": 20,
                "video_length": 124,
                "sliding_window_size": 124,
                **extra,
            }
            with self.subTest(expected=expected):
                result = family_handler.validate_generative_settings(
                    "minimax_h3_ref2va", definition, inputs
                )
                self.assertIn(expected.lower(), result.lower())

    def test_full_wgp_defaults_allow_two_pass_for_lightx2v_turbo4_native_euler(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        definition = family_handler.query_model_def("minimax_h3_ref2va", {})
        self.assertEqual(definition.get("minimax_h3_sampler"), "euler")
        self.assertTrue(definition.get("minimax_h3_two_pass_compatible"))
        for implicit_solver in ("default", ""):
            inputs = {
                "custom_settings": {
                    "h3_two_pass_enabled": "on",
                    "h3_phase2_steps": 3,
                    "h3_phase2_noise": 0.9035,
                    "h3_phase2_accelerator_strength": 1.0,
                    "audio_refinement": "none",
                },
                "sample_solver": implicit_solver,
                "video_prompt_type": "",
                "image_prompt_type": "",
                "video_guide": None,
                "input_video": None,
                "video_mask": None,
                "image_mask": None,
                "input_masks": [],
                "outpainting_dims": [0, 0, 0, 0],
                "video_guide_outpainting": "# disabled: 0 0 0 0",
                "denoising_strength": 1.0,
                "num_inference_steps": 4,
                "video_length": 124,
                "sliding_window_size": 124,
                "sliding_window_overlap": 18,
                "minimax_h3_turbo_mode": True,
                "minimax_h3_turbo_preset": "lightx2v-ref2va-4step-v1.0-768p",
                "activated_loras": [
                    "minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v1.0_768p_bf16.safetensors"
                ],
                "loras_selected": [
                    "minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v1.0_768p_bf16.safetensors"
                ],
                "loras_multipliers": "1.0",
                "_face_refinement": False,
                "face_refinement": False,
            }
            current_definition = copy.deepcopy(definition)
            family_handler.fix_settings(
                "minimax_h3_ref2va", 2.58, current_definition, inputs
            )
            with self.subTest(sample_solver=implicit_solver):
                self.assertIsNone(
                    family_handler.validate_generative_settings(
                        "minimax_h3_ref2va", current_definition, inputs
                    )
                )
                self.assertEqual(inputs["sample_solver"], implicit_solver)

    def test_two_pass_rejects_actual_mask_and_positive_outpaint_dimensions(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        definition = {
            "architecture": "minimax_h3_ref2va",
            "omni_reference": True,
            "frames_maximum": 719,
            "minimax_h3_sampler": "euler",
            "minimax_h3_two_pass_compatible": True,
        }
        base_inputs = {
            "custom_settings": {"h3_two_pass_enabled": "on"},
            "sample_solver": "default",
            "video_prompt_type": "",
            "video_mask": None,
            "image_mask": None,
            "input_masks": [],
            "outpainting_dims": [],
            "video_guide_outpainting": "# disabled",
            "denoising_strength": 1.0,
            "num_inference_steps": 20,
            "video_length": 124,
            "sliding_window_size": 124,
        }
        for extra in (
            {"video_mask": "mask.mp4"},
            {"outpainting_dims": [0, 10, 0, 0]},
            {"video_guide_outpainting": "0 10 0 0"},
        ):
            inputs = {**base_inputs, **extra}
            with self.subTest(extra=extra):
                result = family_handler.validate_generative_settings(
                    "minimax_h3_ref2va", definition, inputs
                )
                self.assertIn("masks or outpainting", result.lower())

    def test_two_pass_rejects_explicit_non_euler_without_rewriting_it(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        definition = {
            "architecture": "minimax_h3_ref2va",
            "omni_reference": True,
            "frames_maximum": 719,
            "minimax_h3_sampler": "euler",
            "minimax_h3_two_pass_compatible": True,
        }
        inputs = {
            "custom_settings": {"h3_two_pass_enabled": "on"},
            "sample_solver": "unipc",
            "video_prompt_type": "",
            "video_mask": None,
            "image_mask": None,
            "input_masks": [],
            "outpainting_dims": [],
            "video_guide_outpainting": "# disabled",
            "denoising_strength": 1.0,
            "num_inference_steps": 20,
            "video_length": 124,
            "sliding_window_size": 124,
        }

        result = family_handler.validate_generative_settings(
            "minimax_h3_ref2va", definition, inputs
        )

        self.assertIn("effective sampler is 'unipc'", result.lower())
        self.assertEqual(inputs["sample_solver"], "unipc")

    def test_legacy_defaults_have_no_two_pass_opt_in(self):
        for filename in (
            "minimax_h3_ref2va.json",
            "minimax_h3_ref2va_full.json",
        ):
            profile = json.loads(
                (APP / "defaults" / filename).read_text(encoding="utf-8")
            )
            with self.subTest(filename=filename):
                self.assertNotIn("h3_two_pass_enabled", profile.get("custom_settings", {}))
                self.assertNotIn(
                    "minimax_h3_latent_upscaler_enabled",
                    profile["model"],
                )
                self.assertFalse(
                    self.two_pass.normalize_two_pass_settings(
                        profile.get("custom_settings")
                    )["h3_two_pass_enabled"]
                    == "on"
                )

    @staticmethod
    def _flatten(downloads):
        return [
            item
            for download in downloads
            for group in download["fileList"]
            for item in group
        ]


if __name__ == "__main__":
    unittest.main()
