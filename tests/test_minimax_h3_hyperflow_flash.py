"""Bounded CPU checks for the separate H3 HyperFlow Flash family."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from contextlib import nullcontext
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


def _load_flash_runtime():
    """Keep CPU-only helper tests usable outside the pinned MMGP install."""
    def import_with_test_stubs():
        offload = types.ModuleType("mmgp.offload")
        offload.register_row_projection = lambda _projection: None
        sys.modules["mmgp.offload"] = offload
        import mmgp

        mmgp.offload = offload

        diffusers = types.ModuleType("diffusers")
        diffusers.__path__ = []
        diffusers_utils = types.ModuleType("diffusers.utils")
        diffusers_utils.__path__ = []
        torch_utils = types.ModuleType("diffusers.utils.torch_utils")
        torch_utils.randn_tensor = lambda shape, **kwargs: torch.randn(shape, **kwargs)
        diffusers_names = (
            "diffusers",
            "diffusers.utils",
            "diffusers.utils.torch_utils",
        )
        saved_modules = {name: sys.modules.get(name) for name in diffusers_names}
        try:
            sys.modules["diffusers"] = diffusers
            sys.modules["diffusers.utils"] = diffusers_utils
            sys.modules["diffusers.utils.torch_utils"] = torch_utils
            sys.modules.pop("models.minimax_h3.hyperflow_flash", None)
            return importlib.import_module("models.minimax_h3.hyperflow_flash")
        finally:
            for name, previous in saved_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    if importlib.util.find_spec("optimum") is None:
        return import_with_test_stubs()
    try:
        return importlib.import_module("models.minimax_h3.hyperflow_flash")
    except ModuleNotFoundError as error:
        if error.name != "optimum":
            raise
        return import_with_test_stubs()


class TestHyperFlowFlashRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flash = _load_flash_runtime()

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

    def test_mmgp_loads_and_runs_dense_and_factorized_convrot_paths_for_both_workflows(self):
        if importlib.util.find_spec("optimum") is None:
            self.skipTest("Pinned MMGP/Quanto runtime is unavailable")
        try:
            from mmgp import offload, quant_router
            from models.minimax_h3 import minimax_h3_main
            from safetensors import safe_open
            from safetensors.torch import save_file
            from shared.qtypes import int8_convrot
        except ImportError as error:
            self.skipTest(f"Pinned MMGP/Quanto runtime is unavailable: {error}")

        handler_name = "shared.qtypes.int8_convrot"
        registered = set(quant_router._HANDLER_MODULES)
        quant_router.register_handler(handler_name)
        load_model_data = offload.load_model_data
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)

        def make_int8_state(base, rows, *, vector_scale=False):
            weight = (
                torch.arange(rows * 64, dtype=torch.int32)
                .reshape(rows, 64)
                .remainder(23)
                .sub(11)
                .to(torch.int8)
            )
            scale = torch.linspace(0.001, 0.005, rows, dtype=torch.float32).reshape(
                rows, 1
            )
            return {
                base + ".weight": weight,
                base + ".weight_scale": scale.flatten() if vector_scale else scale,
            }, weight, scale

        class TinyTransformer(torch.nn.Module):
            def __init__(self, target, block_index, dtype):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.Module() for _ in range(block_index + 1)]
                )
                self.blocks[block_index].attn = torch.nn.Module()
                setattr(
                    self.blocks[block_index].attn,
                    target.rsplit(".", 1)[-1],
                    torch.nn.Linear(64, 4, bias=False, dtype=dtype, device="meta"),
                )
                self.blocks[block_index].attn.dense_proj = torch.nn.Linear(
                    64, 5, bias=True, dtype=dtype, device="meta"
                )

            def set_qkv_layout(self, layout):
                self.test_qkv_layout = layout

        try:
            # FL2VA and Ref2VA checkpoints use different named low-rank
            # projections. Exercise each through the production H3 loader.
            for target in ("blocks.1.attn.q_proj", "blocks.4.attn.v_proj"):
                with self.subTest(target=target):
                    block_index = int(target.split(".")[1])
                    dense_base = f"blocks.{block_index}.attn.dense_proj"
                    raw_state = {}
                    dense_state, dense_weight, dense_scale = make_int8_state(dense_base, 5, vector_scale=True)
                    low_rank_state, down_weight, down_scale = make_int8_state(
                        target + ".down", 3
                    )
                    raw_state.update(dense_state)
                    raw_state.update(low_rank_state)
                    bias = torch.linspace(-0.2, 0.2, 5, dtype=torch.bfloat16)
                    up_weight = (
                        torch.arange(12, dtype=torch.float32)
                        .reshape(4, 3)
                        .div(13)
                        .to(torch.bfloat16)
                    )
                    raw_state[dense_base + ".bias"] = bias
                    raw_state[target + ".up.weight"] = up_weight

                    checkpoint = {
                        "hyperflow_flash": True,
                        "compressed_modulation": False,
                        "adaln_curve_grid": None,
                        "time_embed_dim": 16,
                        "endpoint_gate": 0.25,
                        "dropped_blocks": (),
                        "low_rank": {target: 3},
                        "quantization_format": "int8_convrot",
                    }

                    def construct_tiny(**kwargs):
                        return TinyTransformer(target, block_index, kwargs["dtype"])

                    def load_tiny_state(model, filename, **kwargs):
                        self.assertTrue(
                            all(not parameter.requires_grad for parameter in model.parameters()),
                            "H3 transformer must be frozen before MMGP installs INT8 Parameters",
                        )
                        load_model_data(model, filename, verboseLevel=0, **kwargs)

                    def read_tiny_file(filename, **_):
                        with safe_open(filename, framework="pt") as handle:
                            state = {key: handle.get_tensor(key).clone() for key in handle.keys()}
                            return state, handle.metadata()

                    with (
                        mock.patch.object(
                            minimax_h3_main,
                            "MiniMaxH3Transformer",
                            side_effect=construct_tiny,
                        ),
                        mock.patch.object(
                            minimax_h3_main,
                            "probe_h3_checkpoint",
                            return_value=checkpoint,
                        ),
                        mock.patch.object(
                            minimax_h3_main,
                            "init_empty_weights",
                            return_value=nullcontext(),
                        ),
                        mock.patch.object(
                            minimax_h3_main.offload,
                            "load_model_data",
                            side_effect=load_tiny_state,
                        ),
                        mock.patch.object(offload, "_safetensors_load_file", side_effect=read_tiny_file),
                    ):
                        filename = str(Path(scratch.name) / f"tiny-hyperflow-{block_index}.safetensors")
                        save_file(raw_state, filename, metadata={
                            "quantization_format": "int8_convrot",
                            "convrot_format": "comfy_quant_v1",
                        })
                        model = minimax_h3_main._load_transformer(
                            filename,
                            torch.bfloat16,
                            qkv_layout="contiguous",
                            hyperflow_flash=True,
                        )

                    dense = model.blocks[block_index].attn.dense_proj
                    factored = model.get_submodule(target)
                    down = factored.down
                    for module in (dense, down):
                        self.assertIsInstance(
                            module.weight, int8_convrot.Int8ConvRotWeightTensor
                        )
                        self.assertEqual(module._convrot_group_size, 64)
                        self.assertFalse(module.weight.requires_grad)

                    inputs = torch.linspace(
                        -1.0, 1.0, 128, dtype=torch.bfloat16
                    ).reshape(2, 64)
                    rotated = int8_convrot._rotate_activation(inputs, 64)

                    def expected_convrot(weight, scale, layer_bias=None):
                        accum = torch.nn.functional.linear(
                            rotated.float(), weight.float()
                        )
                        row_scales = (
                            scale.reshape(-1)
                            .to(torch.bfloat16)
                            .float()
                            .reshape(1, -1)
                        )
                        expected = (accum * row_scales).to(torch.bfloat16)
                        return expected if layer_bias is None else expected + layer_bias

                    torch.testing.assert_close(
                        dense(inputs),
                        expected_convrot(dense_weight, dense_scale, bias),
                        rtol=0.01,
                        atol=0.005,
                    )
                    expected_down = expected_convrot(down_weight, down_scale)
                    torch.testing.assert_close(
                        down(inputs), expected_down, rtol=0.01, atol=0.005
                    )
                    torch.testing.assert_close(
                        factored(inputs),
                        torch.nn.functional.linear(expected_down, up_weight),
                        rtol=0.01,
                        atol=0.005,
                    )
        finally:
            if handler_name not in registered:
                quant_router.unregister_handler(handler_name)

    def test_two_time_conditioning_keeps_reference_rows_at_their_own_time(self):
        from models.minimax_h3.hyperflow_flash import two_time_rows
        from models.minimax_h3.packing import (
            MINIMAX_H3_TEXT_TAG,
            build_packed_sequence,
            build_row_timesteps,
        )

        layout = build_packed_sequence(
            torch.tensor([MINIMAX_H3_TEXT_TAG]),
            1,
            2,
            2,
            1,
            (1, 2, 2),
            keyframe_anchors=("first",),
            audio_condition_anchors=(("first", 1),),
        )
        timestep, indices = build_row_timesteps(
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
