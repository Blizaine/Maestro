"""CPU contracts for the pinned H3 PTD RefAsymOPD recipe."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class TestH3PTDRefAsymOPD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.minimax_h3 import pdd, ptd
        from shared.utils.loras_mutipliers import parse_loras_multipliers

        cls.pdd = pdd
        cls.ptd = ptd
        cls.parse_loras_multipliers = staticmethod(parse_loras_multipliers)

    def _profile(self, filename):
        return json.loads(
            (APP / "defaults" / filename).read_text(encoding="utf-8")
        )

    def test_full_and_pruned_profiles_pin_all_eight_independent_hops(self):
        profiles = (
            self._profile("minimax_h3_ref2va_ptd_refasymopd.json"),
            self._profile("minimax_h3_ref2va_pruned_ptd_refasymopd.json"),
        )
        for profile in profiles:
            model = profile["model"]
            with self.subTest(architecture=model["architecture"]):
                self.assertTrue(model["minimax_h3_ptd_refasymopd"])
                self.assertTrue(model["pdd"])
                self.assertTrue(model["ptd"])
                self.assertEqual(model["minimax_h3_transformer_working_vram_gb"], 10)
                self.assertEqual(profile["num_inference_steps"], 8)
                self.assertEqual(profile["guidance_phases"], 1)
                self.assertEqual(profile["guidance_scale"], 1.0)
                self.assertEqual(profile["flow_shift"], 12.0)
                self.assertEqual(profile["audio_flow_shift"], 3.0)
                self.assertEqual(profile["sample_solver"], "euler")
                self.assertEqual(profile["minimax_h3_reference_detail"], "ptd768")
                self.assertEqual(len(model["loras"]), 8)
                self.assertEqual(len(model["loras_multipliers"]), 8)
                for index, (url, multiplier) in enumerate(
                    zip(model["loras"], model["loras_multipliers"]),
                    start=1,
                ):
                    self.assertIn(self.ptd.PTD_REVISION, url)
                    self.assertIn(f"hop{index:02d}_bf16.safetensors", url)
                    self.assertEqual(
                        multiplier,
                        self.ptd.PTD_LORA_MULTIPLIERS[index - 1],
                    )

    def test_one_hot_hops_activate_one_adapter_at_each_step(self):
        selected = list(reversed(self.ptd.PTD_LORA_FILES))
        ordered = self.ptd.ordered_ptd_adapters(selected)
        self.assertEqual(ordered, self.ptd.PTD_LORA_FILES)

        _choices, schedules, error = self.parse_loras_multipliers(
            list(self.ptd.PTD_LORA_MULTIPLIERS),
            self.ptd.PTD_NUM_STEPS,
            self.ptd.PTD_NUM_STEPS,
            nb_phases=1,
        )
        self.assertEqual(error, "")
        self.ptd.validate_ptd_schedules(ordered, schedules)

        import mmgp.offload as offload
        from shared.utils.loras_mutipliers import update_loras_slists

        transformer = types.SimpleNamespace()
        update_loras_slists(transformer, schedules, self.ptd.PTD_NUM_STEPS)
        for step in range(self.ptd.PTD_NUM_STEPS):
            offload.set_step_no_for_lora(transformer, step)
            active = [
                index
                for index in range(self.ptd.PTD_NUM_STEPS)
                if transformer._loras_scaling[str(index)][transformer._lora_step_no]
                == 1.0
            ]
            self.assertEqual(active, [step])

        invalid = copy.deepcopy(schedules)
        invalid["phase1"][2] = [1.0] * self.ptd.PTD_NUM_STEPS
        with self.assertRaisesRegex(ValueError, "active only"):
            self.ptd.validate_ptd_schedules(ordered, invalid)
        with self.assertRaisesRegex(ValueError, "all eight"):
            self.ptd.ordered_ptd_adapters(ordered[:-1])

    def test_ptd_cfg_one_validates_wgp_and_direct_call_aliases(self):
        validate = self.ptd.validate_ptd_cfg_settings
        validate({})
        validate({"guide_scale": 1.0})
        validate({"guidance_scale": "1.0"})
        validate({"guide_scale": 1, "guidance_scale": 1.0})
        for kwargs in (
            {"guide_scale": 2.0},
            {"guidance_scale": 0.0},
            {"guide_scale": 1.0, "guidance_scale": 1.5},
            {"guide_scale": True},
            {"guidance_scale": float("inf")},
            {"guide_scale": "cfg"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "CFG 1"):
                    validate(kwargs)

        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        model = object.__new__(MiniMaxH3Model)
        model.ptd_refasymopd = True
        model.sample_solver = "euler"
        model.transformer = types.SimpleNamespace()
        for kwargs in (
            {"guide_scale": 2.0},
            {"guidance_scale": 2.0},
        ):
            with self.subTest(direct_kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "CFG 1"):
                    model.generate(
                        input_prompt="a short clip",
                        sampling_steps=8,
                        **kwargs,
                    )

    def test_pdd_head_header_checks_shapes_before_reading_and_keeps_owned_fp32_cpu_heads(self):
        import torch
        import tempfile
        from safetensors import safe_open
        from safetensors.torch import save_file

        from models.minimax_h3 import pdd

        tensors = {
            "proj_out.weight": torch.arange(32 * 2 * 3, dtype=torch.float32).reshape(32, 2, 3),
            "proj_out.bias": torch.arange(32 * 2, dtype=torch.float32).reshape(32, 2),
            "audio_proj_out.weight": torch.arange(32 * 1 * 3, dtype=torch.float32).reshape(32, 1, 3),
            "audio_proj_out.bias": torch.arange(32, dtype=torch.float32).reshape(32, 1),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actual-header.safetensors"
            save_file(tensors, str(path))
            with safe_open(str(path), framework="pt", device="cpu") as actual_handle:
                self.assertEqual(
                    tuple(actual_handle.get_slice("proj_out.weight").get_shape()),
                    (32, 2, 3),
                )
                self.assertEqual(
                    actual_handle.get_slice("proj_out.weight").get_dtype(),
                    "F32",
                )

        class _Slice:
            def __init__(self, tensor):
                self.tensor = tensor

            def get_shape(self):
                return list(self.tensor.shape)

            def get_dtype(self):
                return "F32" if self.tensor.dtype == torch.float32 else "F16"

        class _Handle:
            def __init__(self, values):
                self.values = values
                self.reads = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def keys(self):
                return list(self.values)

            def metadata(self):
                return {"pdd_num_steps": "32", "pdd_block_size": "4"}

            def get_slice(self, key):
                return _Slice(self.values[key])

            def get_tensor(self, key):
                self.reads.append(key)
                return self.values[key]

        handle = _Handle(tensors)
        transformer = types.SimpleNamespace(
            final_layer=types.SimpleNamespace(
                video_out=torch.nn.Linear(3, 2),
                audio_out=torch.nn.Linear(3, 1),
            )
        )
        with mock.patch("safetensors.safe_open", return_value=handle):
            controller = pdd.install_pdd_parallel_heads(transformer, "pdd.safetensors")

        self.assertEqual(len(handle.reads), 4)
        self.assertTrue(all(value.device.type == "cpu" for value in controller.video_interval_weights.reshape(-1, 2, 3)))
        self.assertEqual(controller.video_interval_weights.dtype, torch.float32)
        self.assertNotEqual(
            controller.video_interval_weights.data_ptr(),
            tensors["proj_out.weight"].data_ptr(),
        )
        hidden = torch.ones((1, 3), dtype=torch.bfloat16)
        self.assertEqual(transformer.final_layer.video_out(hidden).dtype, torch.float32)

        malformed = dict(tensors)
        malformed["proj_out.weight"] = torch.zeros((32, 2, 4), dtype=torch.float32)
        bad_handle = _Handle(malformed)
        fresh_transformer = types.SimpleNamespace(
            final_layer=types.SimpleNamespace(
                video_out=torch.nn.Linear(3, 2),
                audio_out=torch.nn.Linear(3, 1),
            )
        )
        with mock.patch("safetensors.safe_open", return_value=bad_handle):
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                pdd.install_pdd_parallel_heads(fresh_transformer, "bad-pdd.safetensors")
        self.assertEqual(bad_handle.reads, [])

        wrong_dtype = dict(tensors)
        wrong_dtype["proj_out.weight"] = torch.zeros((32, 2, 3), dtype=torch.float16)
        dtype_handle = _Handle(wrong_dtype)
        with mock.patch("safetensors.safe_open", return_value=dtype_handle):
            with self.assertRaisesRegex(ValueError, "must be FP32"):
                pdd.install_pdd_parallel_heads(fresh_transformer, "wrong-dtype.safetensors")
        self.assertEqual(dtype_handle.reads, [])

    def test_pdd_output_heads_follow_hops_and_restore_base_heads(self):
        video_base = torch.nn.Linear(2, 1, bias=False)
        audio_base = torch.nn.Linear(2, 1, bias=False)
        video_head = self.pdd.MiniMaxH3PDDParallelHead(
            video_base,
            tuple(torch.full((1, 2), float(index + 1)) for index in range(8)),
            None,
            1.0,
        )
        audio_head = self.pdd.MiniMaxH3PDDParallelHead(
            audio_base,
            tuple(torch.full((1, 2), float(index + 1)) for index in range(8)),
            None,
            1.0,
        )
        transformer = types.SimpleNamespace(
            final_layer=types.SimpleNamespace(
                video_out=video_head,
                audio_out=audio_head,
            )
        )
        transformer._pdd_controller = object()
        sample = torch.ones(1, 2)
        for step in range(8):
            video_head.set_step(step)
            audio_head.set_step(step)
            self.assertTrue(torch.equal(video_head(sample), torch.tensor([[2.0 * (step + 1)]])))
            self.assertTrue(torch.equal(audio_head(sample), torch.tensor([[2.0 * (step + 1)]])))

        self.pdd.release_pdd_parallel_heads(transformer)
        self.assertIs(transformer.final_layer.video_out, video_base)
        self.assertIs(transformer.final_layer.audio_out, audio_base)
        self.assertFalse(hasattr(transformer, "_pdd_controller"))

    def test_handler_locks_recipe_and_queries_pdd_heads_without_two_pass(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        definition = family_handler.query_model_def(
            "minimax_h3_ref2va",
            {"minimax_h3_ptd_refasymopd": True, "pdd": True},
        )
        self.assertTrue(definition["minimax_h3_ptd_refasymopd"])
        self.assertTrue(definition["pdd"])
        self.assertFalse(definition["minimax_h3_two_pass_compatible"])
        self.assertEqual(definition["inference_steps_min"], 8)
        self.assertEqual(definition["inference_steps_max"], 8)
        self.assertEqual(definition["minimax_h3_sampler"], "euler")
        self.assertEqual(definition["custom_settings"], [])

        downloads = family_handler.query_model_files(
            [], "minimax_h3_ref2va", definition
        )
        flattened = [
            item
            for download in downloads
            for group in download["fileList"]
            for item in group
        ]
        self.assertIn(self.ptd.PTD_HEAD_FILE, flattened)
        self.assertNotIn("minimax_h3_latent_upscaler_3d_bf16.safetensors", flattened)

        invalid_settings = {
            "custom_settings": {"h3_two_pass_enabled": "on"},
            "num_inference_steps": 8,
            "guidance_phases": 1,
            "guidance_scale": 1,
            "flow_shift": 12,
            "audio_flow_shift": 3,
            "sample_solver": "euler",
            "video_length": 124,
        }
        self.assertIn(
            "cannot use H3 Two-Pass",
            family_handler.validate_generative_settings(
                "minimax_h3_ref2va", definition, invalid_settings
            ),
        )


if __name__ == "__main__":
    unittest.main()
