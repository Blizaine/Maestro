"""CPU contract checks for the official two- and four-step pi-H3 presets."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save
from accelerate import init_empty_weights


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3.minimax_h3_handler import family_handler
from models.minimax_h3.pi_h3 import (
    _map_pi_h3_audio_sigmas,
    _scale_pi_h3_initial_noise,
    pi_h3_adapter_steps,
    pi_h3_dwd_adapter_steps,
    pi_h3_lora_incompatibility,
    pi_h3_sigmas,
)
from models.minimax_h3.lora_affine import (
    convert_adaln_loras,
    pop_source_package,
)


class PiH3PresetTests(unittest.TestCase):
    def test_official_schedule_matches_the_ordinary_profile_sigma_values(self):
        for steps in (2, 4):
            profile_path = (
                ROOT
                / "app"
                / "profiles"
                / "minimax_h3_fl2va"
                / f"pi-H3 FL2VA {steps} Steps.json"
            )
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            actual = [float(value) for value in profile["custom_sigmas"].split(",")]
            trained = pi_h3_sigmas(steps, 12.0)
            self.assertEqual(profile["num_inference_steps"], steps)
            self.assertEqual(profile["sample_solver"], "euler")
            self.assertEqual(profile["flow_shift"], 12)
            self.assertEqual(profile["guidance_phases"], 1)
            self.assertEqual(profile["guidance_scale"], 1)
            self.assertEqual(len(actual), steps)
            for got, expected in zip(actual, trained):
                self.assertTrue(math.isclose(got, expected, rel_tol=0, abs_tol=1e-5))

    def test_trained_video_grids_map_to_the_official_paired_audio_grids(self):
        expected_audio_grids = {
            2: [0.9755245, 0.61634755, 0.0],
            4: [0.9863015, 0.8715084, 0.69064754, 0.36363637, 0.0],
        }
        for steps, expected in expected_audio_grids.items():
            with self.subTest(steps=steps):
                video = torch.tensor(
                    [*pi_h3_sigmas(steps, 12.0), 0.0], dtype=torch.float32
                )
                audio = _map_pi_h3_audio_sigmas(
                    video, video_shift=12.0, audio_shift=3.0
                )
                torch.testing.assert_close(
                    audio,
                    torch.tensor(expected, dtype=torch.float32),
                    rtol=0,
                    atol=1e-6,
                )

    def test_initial_noise_scaling_preserves_attached_condition_rows(self):
        from types import SimpleNamespace

        layout = SimpleNamespace(
            num_condition_video_rows=2,
            num_target_condition_video_rows=2,
            num_condition_audio_rows=2,
            num_target_condition_audio_latents=1,
        )
        video_rows = torch.ones((8, 3), dtype=torch.float32)
        audio_rows = torch.ones((8, 3), dtype=torch.float32)
        _scale_pi_h3_initial_noise(
            video_rows,
            audio_rows,
            layout,
            num_audio_latents=3,
            audio_channels=2,
            video_sigma=torch.tensor(0.5),
            audio_sigma=torch.tensor(0.25),
        )
        torch.testing.assert_close(video_rows[:2], torch.ones((2, 3)))
        torch.testing.assert_close(video_rows[2:6], torch.full((4, 3), 0.5))
        torch.testing.assert_close(video_rows[6:], torch.ones((2, 3)))
        torch.testing.assert_close(audio_rows[:3], torch.ones((3, 3)))
        torch.testing.assert_close(audio_rows[3:5], torch.full((2, 3), 0.25))
        torch.testing.assert_close(audio_rows[5:6], torch.ones((1, 3)))
        torch.testing.assert_close(audio_rows[6:8], torch.full((2, 3), 0.25))

    def test_initial_noise_scaling_handles_zero_audio_latents(self):
        from types import SimpleNamespace

        layout = SimpleNamespace(
            num_condition_video_rows=0,
            num_target_condition_video_rows=0,
            num_condition_audio_rows=0,
            num_target_condition_audio_latents=0,
        )
        video_rows = torch.empty((0, 3), dtype=torch.float32)
        audio_rows = torch.empty((0, 3), dtype=torch.float32)
        _scale_pi_h3_initial_noise(
            video_rows,
            audio_rows,
            layout,
            num_audio_latents=0,
            audio_channels=2,
            video_sigma=torch.tensor(0.5),
            audio_sigma=torch.tensor(0.25),
        )

    def test_adapter_names_and_model_compatibility_are_workflow_specific(self):
        self.assertEqual(
            pi_h3_adapter_steps("loras/minimax_h3_pi_2step_lora_bf16.safetensors"),
            2,
        )
        self.assertEqual(
            pi_h3_adapter_steps(r"C:\\loras\\minimax_h3_pi_4step_lora_bf16.safetensors"),
            4,
        )
        self.assertEqual(
            pi_h3_dwd_adapter_steps(
                "loras/minimax_h3_fl2va_pi_2step_dwd_lora_bf16.safetensors"
            ),
            2,
        )
        self.assertIsNone(
            pi_h3_adapter_steps("minimax_h3_pi_6step_lora_bf16.safetensors")
        )
        self.assertIsNone(
            pi_h3_lora_incompatibility(
                {"architecture": "minimax_h3_full"},
                "minimax_h3_pi_2step_lora_bf16.safetensors",
            )
        )
        self.assertIn(
            "matching dedicated",
            pi_h3_lora_incompatibility(
                {"dwd": 4},
                "minimax_h3_fl2va_pi_2step_dwd_lora_bf16.safetensors",
            ),
        )
        self.assertIn(
            "cannot be used on a DWD",
            pi_h3_lora_incompatibility(
                {"architecture": "minimax_h3", "dwd": 2},
                "minimax_h3_pi_2step_lora_bf16.safetensors",
            ),
        )

    def test_dwd_model_options_lock_the_matching_one_pass_recipe(self):
        for architecture, steps in (("minimax_h3", 2), ("minimax_h3_full", 4)):
            with self.subTest(architecture=architecture, steps=steps):
                options = family_handler.query_model_def(
                    architecture,
                    {"dwd": steps},
                )
                self.assertEqual(options["dwd"], steps)
                self.assertTrue(options["minimax_h3_dwd"])
                self.assertTrue(options["lock_inference_steps"])
                self.assertEqual(
                    (options["inference_steps_min"], options["inference_steps_max"]),
                    (steps, steps),
                )
                self.assertTrue(options["lock_guidance_phases"])
                self.assertTrue(options["lock_guidance_scale"])
                self.assertEqual(options["sample_solvers"], [("Euler", "euler")])
                self.assertFalse(options["first_block_cache"])
                self.assertFalse(options["spectrum_cache"])
                self.assertFalse(options["minimax_h3_two_pass_compatible"])
                self.assertFalse(options["minimax_h3_accelerators_allowed"])
                self.assertEqual(options["profiles_dir"], [])
                self.assertEqual(options["preset_profiles_dir"], [])
                self.assertNotIn("vae_upsamplers", options)
                self.assertEqual(options["minimax_h3_model_id"],
                                 f"{architecture}_pi_dwd_{steps}step")

    def test_dwd_rejects_reference_and_other_accelerator_definitions(self):
        with self.assertRaisesRegex(ValueError, "FL2VA Frames"):
            family_handler.query_model_def("minimax_h3_ref2va", {"dwd": 2})
        with self.assertRaisesRegex(ValueError, "trained 2-step or 4-step"):
            family_handler.query_model_def("minimax_h3", {"dwd": 3})
        with self.assertRaisesRegex(ValueError, "without another H3 variant"):
            family_handler.query_model_def(
                "minimax_h3",
                {"dwd": 2, "minimax_h3_pika_flex": True},
            )

    def test_saved_dwd_settings_are_repaired_without_changing_duration(self):
        options = family_handler.query_model_def("minimax_h3", {"dwd": 2})
        settings = {
            "num_inference_steps": 20,
            "video_length": 243,
            "sliding_window_size": 243,
            "guidance_phases": 2,
            "guidance_scale": 4.0,
            "flow_shift": 7.0,
            "audio_flow_shift": 9.0,
            "sample_solver": "heun",
            "denoising_strength": 0.5,
            "skip_steps_cache_type": "first_block",
            "custom_settings": {"audio_refinement": "enabled"},
        }
        family_handler.fix_settings("minimax_h3", 3.0, options, settings)
        self.assertEqual(settings["num_inference_steps"], 2)
        self.assertEqual(settings["video_length"], 243)
        self.assertEqual(settings["flow_shift"], 12.0)
        self.assertEqual(settings["audio_flow_shift"], 3.0)
        self.assertEqual(settings["sample_solver"], "euler")
        self.assertEqual(settings["guidance_phases"], 1)
        self.assertEqual(settings["guidance_scale"], 1.0)
        self.assertEqual(settings["denoising_strength"], 1.0)
        self.assertEqual(settings["skip_steps_cache_type"], "")
        self.assertEqual(settings["custom_settings"]["audio_refinement"], "none")

    def test_dwd_decoder_imports_and_writes_without_optional_mmgp_transfer(self):
        from models.minimax_h3 import sol_decoder

        self.assertIsNotNone(
            __import__("models.minimax_h3.dwd", fromlist=["MiniMaxH3DWDHead"])
        )
        original_transfer = sol_decoder._mmgp_transfer
        sol_decoder._mmgp_transfer = None
        try:
            destination = torch.zeros((2, 3), dtype=torch.uint8)
            source = torch.full((2, 3), 177.5, dtype=torch.float32)
            with sol_decoder._ram_writer("test", depth=1, max_bytes=24) as writer:
                writer.write(destination, source)
            self.assertTrue(bool((destination == 177).all()))
        finally:
            sol_decoder._mmgp_transfer = original_transfer

    def test_dwd_rope_buffer_stays_materialized_for_legacy_weight_loading(self):
        from models.minimax_h3.dwd import MiniMaxH3DWDHead

        # _load_transformer builds this head inside the same include_buffers
        # context before calling MMGP's legacy data hydrator.
        with init_empty_weights(include_buffers=True):
            head = MiniMaxH3DWDHead(hidden=32, dtype=torch.float32)
        self.assertTrue(next(head.parameters()).is_meta)
        self.assertEqual(head.refiner.rope.inv_freq.device.type, "cpu")

        head.refiner.reset_rope()

        self.assertFalse(head.refiner.rope.inv_freq.is_meta)
        self.assertEqual(head.refiner.rope.inv_freq.device.type, "cpu")
        self.assertEqual(head.refiner.rope.inv_freq.dtype, torch.float32)
        self.assertGreater(head.refiner.rope.inv_freq.numel(), 0)

    def test_dwd_window_plan_stays_on_cpu_with_non_cpu_default_device(self):
        from models.minimax_h3.sol_decoder import (
            MiniMaxH3SolDecoder3d,
            _State,
            _VideoRotaryPosEmbed,
        )

        decoder = MiniMaxH3SolDecoder3d(
            num_layers=0,
            num_attention_heads=1,
            attention_head_dim=8,
            num_register_tokens=1,
            dtype=torch.float32,
        )
        z = torch.empty((1, 1, 5, 2, 3), device="cpu", dtype=torch.float32)
        previous_default = torch.get_default_device()
        try:
            # Meta is non-CPU but does not require a GPU, making the implicit
            # factory-device leak reproducible in the CPU test suite.
            torch.set_default_device("meta")
            rotary = _VideoRotaryPosEmbed(6)
            state = _State(decoder, z, lambda *_args: None)
        finally:
            torch.set_default_device(previous_default)

        self.assertEqual(rotary.inv_freq.device.type, "cpu")
        self.assertEqual(state.device.type, "cpu")
        for parity in state.parities:
            for _block, _lo, _hi, groups in parity.steps:
                for group in groups:
                    self.assertTrue(all(index.device.type == "cpu" for index in group))

    def test_embedded_source_table_is_removed_and_used_for_full_checkpoint_conversion(self):
        table = torch.zeros((1025, 8), dtype=torch.float32)
        affine = torch.cat(
            (
                torch.eye(8, 2688, dtype=torch.float32),
                torch.full((1, 2688), 0.01, dtype=torch.float32),
            ),
            dim=0,
        )
        package_bytes = save(
            {"adaln_t_table": table, "adaln_affine_map": affine},
            metadata={"h3_map_id": "fl2va_rank8", "full_time_fingerprint": ""},
        )
        payload = {
            "version": 1,
            "sha256": hashlib.sha256(package_bytes).hexdigest(),
            "package": base64.b64encode(package_bytes).decode("ascii"),
        }
        marker = torch.frombuffer(
            bytearray(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
            dtype=torch.uint8,
        )
        state = {
            "__wangp_h3_lora_source__": marker,
            "blocks.0.adaln_proj.linear.lora_A.weight": torch.arange(
                16, dtype=torch.float32
            ).reshape(2, 8),
            "blocks.0.adaln_proj.linear.lora_B.weight": torch.ones(
                4, 2, dtype=torch.float32
            ),
        }
        source_package = pop_source_package(state)
        self.assertEqual(source_package.name, "fl2va_rank8")
        self.assertNotIn("__wangp_h3_lora_source__", state)
        self.assertEqual(len(state), 2)

        count, architecture, source_width, target_width = convert_adaln_loras(
            "minimax_h3_full",
            state,
            source_package=source_package,
        )
        self.assertEqual(
            (count, architecture, source_width, target_width),
            (1, "fl2va", 8, 2688),
        )
        self.assertEqual(
            tuple(state["blocks.0.adaln_proj.linear.lora_A.weight"].shape),
            (2, 2688),
        )
        self.assertEqual(
            tuple(state["blocks.0.adaln_proj.linear.diff_b"].shape),
            (4,),
        )

    def test_staged_official_dwd_loras_use_the_embedded_source_tag(self):
        assets = ROOT.parent / "assets" / "h3" / "loras"
        paths = [
            assets / f"minimax_h3_fl2va_pi_{steps}step_dwd_lora_bf16.safetensors"
            for steps in (2, 4)
        ]
        if not all(path.is_file() for path in paths):
            self.skipTest("Pinned staged DWD LoRAs are not present in this checkout")
        for steps, path in zip((2, 4), paths):
            with self.subTest(steps=steps), safe_open(
                str(path), framework="pt", device="cpu"
            ) as source:
                self.assertEqual(source.metadata()["num_inference_steps"], str(steps))
                self.assertIn("__wangp_h3_lora_source__", source.keys())
                package = pop_source_package(
                    {
                        "__wangp_h3_lora_source__": source.get_tensor(
                            "__wangp_h3_lora_source__"
                        )
                    }
                )
                self.assertEqual(package.name, "fl2va_rank8")


if __name__ == "__main__":
    unittest.main()
