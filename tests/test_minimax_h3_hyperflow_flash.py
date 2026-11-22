"""Bounded CPU checks for the separate H3 HyperFlow Flash family."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


def _load_flash_runtime():
    """Keep CPU-only helper tests usable outside the pinned MMGP install."""
    try:
        return importlib.import_module("models.minimax_h3.hyperflow_flash")
    except ModuleNotFoundError as error:
        if error.name != "optimum":
            raise
        offload = types.ModuleType("mmgp.offload")
        offload.register_row_projection = lambda _projection: None
        sys.modules["mmgp.offload"] = offload
        import mmgp

        mmgp.offload = offload
        sys.modules.pop("models.minimax_h3.hyperflow_flash", None)
        return importlib.import_module("models.minimax_h3.hyperflow_flash")


class TestHyperFlowFlashRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flash = _load_flash_runtime()
        from models.minimax_h3.packing import (
            MINIMAX_H3_TEXT_TAG,
            build_packed_sequence,
            build_row_timesteps,
        )

        cls.MINIMAX_H3_TEXT_TAG = MINIMAX_H3_TEXT_TAG
        cls.build_packed_sequence = staticmethod(build_packed_sequence)
        cls.build_row_timesteps = staticmethod(build_row_timesteps)

    def test_native_and_refinement_grids_match_upstream_flash_intervals(self):
        expected_raw = {
            4: (1.0, 0.681872, 0.205019, 0.036496, 0.0),
            6: (1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0),
            8: (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0),
        }
        for steps, raw in expected_raw.items():
            with self.subTest(steps=steps):
                for shift in (12.0, 3.0):
                    expected = tuple(
                        shift * sigma / (1.0 + (shift - 1.0) * sigma)
                        for sigma in raw
                    )
                    self.assertEqual(
                        self.flash.schedule_sigmas(steps, shift), expected
                    )
        full = self.flash.schedule_sigmas(8, 12.0)
        for count in range(1, 9):
            self.assertEqual(
                self.flash.phase2_tail_sigmas(count), full[-(count + 1) :]
            )
        for invalid in (True, 0, 9, 3.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.flash.phase2_tail_sigmas(invalid)

    def test_checkpoint_metadata_selects_reduced_flash_architecture(self):
        state_dict = {
            "time_embedder.proj_in.weight": torch.zeros(4, 8),
            "time_embedder.proj_out.weight": torch.zeros(16, 4),
            "endpoint_time_embedder.proj_in.weight": torch.zeros(4, 8),
            "endpoint_time_embedder.proj_out.weight": torch.zeros(16, 4),
            "blocks.0.norm1.weight": torch.zeros(4),
            "blocks.49.norm1.weight": torch.zeros(4),
            "blocks.49.attn.q_proj.down.weight": torch.zeros(2, 4),
        }
        info = self.flash.probe_flash_checkpoint(
            state_dict,
            {"hyperflow_gate": "0.5", "quantization_format": "int8_convrot"},
        )
        self.assertTrue(info["hyperflow_flash"])
        self.assertEqual(info["endpoint_gate"], 0.5)
        self.assertEqual(info["time_embed_dim"], 16)
        self.assertEqual(info["dropped_blocks"], tuple(range(1, 49)))
        self.assertEqual(
            info["low_rank"], {"blocks.49.attn.q_proj": 2}
        )
        with self.assertRaisesRegex(ValueError, "endpoint gate"):
            self.flash.probe_flash_checkpoint(state_dict, {})
        malformed = dict(state_dict)
        malformed["endpoint_time_embedder.proj_out.weight"] = torch.zeros(8, 4)
        with self.assertRaisesRegex(ValueError, "compatible endpoint MLPs"):
            self.flash.probe_flash_checkpoint(
                malformed, {"hyperflow_gate": "0.5"}
            )

    def test_low_rank_linear_preserves_linear_projection_contract(self):
        from torch.nn import functional as F

        linear = self.flash.LowRankLinear(5, 3, 2, dtype=torch.float32)
        with torch.no_grad():
            linear.down.weight.copy_(torch.arange(10).reshape(2, 5) / 10)
            linear.up.weight.copy_(torch.arange(6).reshape(3, 2) / 10)
        x = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 10
        self.assertTrue(isinstance(linear, torch.nn.Linear))
        self.assertTrue(
            torch.allclose(
                linear(x),
                F.linear(x, linear.up.weight @ linear.down.weight),
            )
        )
        self.assertEqual(tuple(linear.weight.shape), (3, 5))

    def test_two_time_conditioning_keeps_reference_rows_at_their_own_time(self):
        from models.minimax_h3.hyperflow_flash import two_time_rows

        layout = self.build_packed_sequence(
            torch.tensor([self.MINIMAX_H3_TEXT_TAG]),
            1,
            2,
            2,
            1,
            (1, 2, 2),
            keyframe_anchors=("first",),
            audio_condition_anchors=(("first", 1),),
        )
        timestep, indices = self.build_row_timesteps(
            layout, 0.2, 0.3, 0.8, 1.0
        )
        unique_t, unique_r, pair_indices = two_time_rows(
            layout, timestep, indices, (0.1, 0.15), 0.8
        )
        row_t = unique_t[pair_indices]
        row_r = unique_r[pair_indices]
        self.assertAlmostEqual(float(row_t[layout.text_indices[0]]), 0.2, places=6)
        self.assertAlmostEqual(float(row_r[layout.text_indices[0]]), 0.1, places=6)
        condition_video = layout.video_indices[: layout.num_condition_video_rows]
        self.assertTrue(torch.equal(row_t[condition_video], row_r[condition_video]))
        self.assertTrue(torch.equal(row_t[layout.audio_indices[:2]], row_r[layout.audio_indices[:2]]))
        target_audio = layout.audio_indices[layout.num_condition_audio_rows :]
        self.assertTrue(torch.allclose(row_t[target_audio], torch.full_like(row_t[target_audio], 0.3)))
        self.assertTrue(torch.allclose(row_r[target_audio], torch.full_like(row_r[target_audio], 0.15)))


class TestHyperFlowFlashCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.minimax_h3.minimax_h3_handler import family_handler
        from models.minimax_h3.two_pass import normalize_two_pass_settings, phase2_lora_schedule
        from models.minimax_h3.turbo import normalize_minimax_h3_turbo_request

        cls.handler = family_handler
        cls.normalize = staticmethod(normalize_two_pass_settings)
        cls.phase2_lora_schedule = staticmethod(phase2_lora_schedule)
        cls.normalize_turbo = staticmethod(normalize_minimax_h3_turbo_request)

    def test_frames_and_references_publish_distinct_experimental_catalog_entries(self):
        expected = {
            "minimax_h3": "minimax_h3_hyperflow_fl2va.json",
            "minimax_h3_ref2va": "minimax_h3_ref2va_hyperflow.json",
        }
        for base_model_type, defaults_filename in expected.items():
            with self.subTest(base_model_type=base_model_type):
                model_def = json.loads(
                    (APP / "defaults" / defaults_filename).read_text(encoding="utf-8")
                )["model"]
                definition = self.handler.query_model_def(
                    base_model_type, model_def
                )
                self.assertTrue(definition["minimax_h3_hyperflow_flash"])
                self.assertFalse(definition["minimax_h3_accelerators_allowed"])
                self.assertTrue(definition["minimax_h3_two_pass_compatible"])
                self.assertEqual(
                    definition["minimax_h3_model_id"],
                    model_def["minimax_h3_model_id"],
                )
                self.assertTrue(
                    all(
                        "/resolve/3d3c6fb64d610dcd2d0ea5dd69a9360e69d5f24f/"
                        in url
                        for url in model_def["URLs"]
                    )
                )
                self.assertEqual(definition["inference_steps_min"], 4)
                self.assertEqual(definition["inference_steps_max"], 8)
                self.assertEqual(definition["minimax_h3_sampler"], "euler")
                self.assertNotIn(
                    "h3_phase2_accelerator_strength",
                    {setting["id"] for setting in definition["custom_settings"]},
                )
                self.assertIn("endpoint-conditioned", definition["selector_help"].lower())
                self.assertEqual(
                    definition["profiles_dir"],
                    ["minimax_h3_hyperflow_ref2va"]
                    if base_model_type == "minimax_h3_ref2va"
                    else [],
                )

    def test_validation_and_phase_two_accelerator_selection(self):
        model = self.handler.query_model_def(
            "minimax_h3_ref2va", {"minimax_h3_hyperflow_flash": True}
        )
        inputs = {
            "num_inference_steps": 6,
            "guidance_phases": 1,
            "guidance_scale": 1.0,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "sample_solver": "euler",
            "denoising_strength": 1.0,
            "skip_steps_cache_type": "",
            "video_length": 124,
            "sliding_window_size": 124,
            "sliding_window_overlap": 18,
            "resolution": "864x480",
            "fps": 24,
            "video_prompt_type": "",
            "audio_prompt_type": "",
            "custom_settings": {
                "h3_two_pass_enabled": "on",
                "h3_phase2_steps": 1,
                "h3_phase2_accelerator_strength": 1.25,
            },
        }
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "minimax_h3_ref2va", model, dict(inputs)
            )
        )
        self.assertEqual(
            self.normalize(inputs["custom_settings"], model)[
                "h3_phase2_accelerator_strength"
            ],
            0.0,
        )
        invalid = dict(inputs, num_inference_steps=5)
        self.assertIn(
            "4, 6, or 8",
            self.handler.validate_generative_settings(
                "minimax_h3_ref2va", model, invalid
            ),
        )
        masked = dict(inputs, input_masks=[object()])
        self.assertIn(
            "masks or outpainting",
            self.handler.validate_generative_settings(
                "minimax_h3_ref2va", model, masked
            ),
        )
        old_hyperflow = dict(
            inputs,
            custom_settings={"h3_two_pass_enabled": "off"},
            activated_loras=["minimax_h3_hyperflow_8step_lora_bf16.safetensors"],
        )
        self.assertIn(
            "ordinary H3 Ref2VA",
            self.handler.validate_generative_settings(
                "minimax_h3_ref2va", model, old_hyperflow
            ),
        )
        unsupported = dict(
            inputs,
            custom_settings={"h3_two_pass_enabled": "off"},
            activated_loras=["minimax_h3_turbo4_example.safetensors"],
        )
        self.assertIn(
            "does not support additional H3",
            self.handler.validate_generative_settings(
                "minimax_h3_ref2va", model, unsupported
            ),
        )

        schedule = {
            "phase1": [0.7, 0.5],
            "phase2": [0.4, 0.2],
            "phase3": [0.0, 0.0],
            "shared": [False, True],
        }
        selected = self.phase2_lora_schedule(
            schedule,
            ["turbo.safetensors", "style.safetensors"],
            ["turbo.safetensors"],
            1.25,
        )
        self.assertEqual(selected["phase2"], [1.25, 0.5])
        self.assertEqual(selected["phase3"], [1.25, 0.5])
        self.assertEqual(selected["shared"], [True, True])

    def test_pdmd_profiles_pin_the_official_adapter_revision(self):
        from models.minimax_h3.image_handler import (
            PDD_LORA_FILENAME,
            PDD_LORA_REVISION,
            PDD_LORA_SHA256,
            PDD_LORA_SIZE_BYTES,
            PDD_LORA_URL,
        )

        self.assertEqual(PDD_LORA_REVISION, "3d3c6fb64d610dcd2d0ea5dd69a9360e69d5f24f")
        self.assertEqual(PDD_LORA_FILENAME, "minimax_h3_pdmd_2nfe_bf16.safetensors")
        self.assertEqual(PDD_LORA_SIZE_BYTES, 1_383_680_648)
        self.assertEqual(
            PDD_LORA_SHA256,
            "3179e630c3fbd96b9ad7aec3a54fe23fa0ddc63e4afb64607f10cdb3d05c862a",
        )
        self.assertIn(f"/resolve/{PDD_LORA_REVISION}/", PDD_LORA_URL)
        for filename in ("pDMD 2 Steps.json", "pDMD 4 Steps.json"):
            profile = json.loads(
                (APP / "profiles" / "minimax_h3_image" / filename).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(profile["activated_loras"], [PDD_LORA_URL])
        native = json.loads(
            (
                APP
                / "profiles"
                / "minimax_h3_image"
                / "H3-Image 6 Steps (Reset).json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(native["activated_loras"], [])
        self.assertEqual(native["loras_multipliers"], "|")
        self.assertEqual(native["num_inference_steps"], 6)

    def test_flash_rejects_the_managed_turbo_toggle(self):
        body = {"minimax_h3_turbo_mode": True}
        with self.assertRaisesRegex(ValueError, "does not support H3 Turbo"):
            self.normalize_turbo(
                body,
                full_checkpoint=False,
                workflow="ref2va",
                model_def={
                    "minimax_h3_hyperflow_flash": True,
                    "minimax_h3_accelerators_allowed": False,
                },
            )

    def test_two_pass_step_bounds_are_model_specific_and_legacy_recipe_remains(self):
        flash = {"minimax_h3_hyperflow_flash": True}
        self.assertEqual(self.normalize({"h3_phase2_steps": 1}, flash)["h3_phase2_steps"], 1)
        self.assertEqual(self.normalize({"h3_phase2_steps": 8}, flash)["h3_phase2_steps"], 8)
        for steps in (0, 9):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.normalize({"h3_phase2_steps": steps}, flash)
        self.assertEqual(self.normalize({"h3_phase2_steps": 12})["h3_phase2_steps"], 12)
        presets = json.loads(
            (APP / "models" / "minimax_h3" / "turbo_presets.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            "hyperflow-ref2va-8step",
            {preset["id"] for preset in presets["presets"]},
        )


if __name__ == "__main__":
    unittest.main()
