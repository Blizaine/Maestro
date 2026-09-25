"""Core contract tests for the optional MiniMax H3 Singularity integration."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

MODEL_DEFAULT = APP / "defaults" / "minimax_h3_ref2va_singularity.json"
TURBO_ID = "lightx2v-ref2va-turbo4-v0.1-comfy-bf16"
TURBO_FILENAME = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"


def _read_default() -> dict:
    return json.loads(MODEL_DEFAULT.read_text(encoding="utf-8"))


class TestSingularityModelDefinition(unittest.TestCase):
    def test_default_is_pinned_and_keeps_singularity_separate_from_fused(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        default = _read_default()
        model = default["model"]
        definition = family_handler.query_model_def(
            "minimax_h3_ref2va",
            model,
        )

        self.assertEqual(model["minimax_h3_model_id"], "minimax_h3_ref2va_singularity")
        self.assertEqual(model["architecture"], "minimax_h3_ref2va")
        self.assertIn("af671d9214a6e41ab8c2f43e9f871ea56246115f", model["URLs"][0])
        self.assertEqual(
            model["source_sha256"],
            "412a7b126595a958964193f3b42513d7cf2df4196ef04223a0f05e3622949cce",
        )
        self.assertEqual(model["source_size_bytes"], 20967647456)
        self.assertTrue(model["minimax_h3_turbo_mode_default"])
        self.assertEqual(model["minimax_h3_unaccelerated_default_steps"], 20)

        self.assertTrue(definition["omni_reference"])
        self.assertTrue(definition["supports_reference_audio"])
        self.assertTrue(definition["minimax_h3_singularity"])
        self.assertFalse(definition["minimax_h3_fused_turbo"])
        self.assertEqual(definition["minimax_h3_qkv_layout"], "grouped")
        self.assertEqual(definition["compatible_model_paths"], {})
        self.assertEqual(definition["compatible_model_qkv_layouts"], {})
        self.assertEqual(
            definition["minimax_h3_default_turbo_preset"],
            TURBO_ID,
        )
        self.assertEqual(
            definition["omni_sequence_memory_policy"]["reference_margin_steps"],
            1,
        )
        self.assertEqual(
            definition["omni_sequence_memory_policy"]["checkpoint"],
            "pruned",
        )

    def test_singularity_cannot_be_routed_as_full_or_fused_model(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        model = _read_default()["model"]
        with self.assertRaisesRegex(ValueError, "Ref2VA References"):
            family_handler.query_model_def("minimax_h3_ref2va_full", model)
        with self.assertRaisesRegex(ValueError, "separate checkpoint"):
            family_handler.query_model_def(
                "minimax_h3_ref2va",
                {**model, "minimax_h3_fused_turbo": True},
            )

    def test_singularity_reuses_standard_ref2va_assets(self):
        from models.minimax_h3.minimax_h3_handler import family_handler

        definition = family_handler.query_model_def(
            "minimax_h3_ref2va",
            _read_default()["model"],
        )
        downloads = family_handler.query_model_files(
            [],
            "minimax_h3_ref2va",
            definition,
        )
        flattened = [
            name
            for download in downloads
            for group in download["fileList"]
            for name in group
        ]

        self.assertIn("minimax_h3_video_vae_fp16.safetensors", flattened)
        self.assertIn("minimax_h3_audio_vae_fp32.safetensors", flattened)
        self.assertIn("config.json", flattened)
        self.assertNotIn("minimax_h3_video_vae_int8_convrot.safetensors", flattened)
        self.assertNotIn("LICENSE", flattened)
        self.assertNotIn("NOTICE", flattened)


class TestSingularityTurboRecipe(unittest.TestCase):
    def test_exact_adapter_is_pinned_and_detected_as_ref2va_turbo(self):
        from models.minimax_h3.turbo import (
            find_minimax_h3_accelerators,
            is_minimax_h3_pdd_lora,
            is_minimax_h3_turbo_lora,
            minimax_h3_turbo_preset_for_path,
        )

        preset = minimax_h3_turbo_preset_for_path(TURBO_FILENAME)
        self.assertIsNotNone(preset)
        self.assertEqual(preset["id"], TURBO_ID)
        self.assertEqual(preset["repo_id"], "lightx2v/Minimax-h3-Turbo")
        self.assertEqual(
            preset["revision"],
            "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
        )
        self.assertEqual(
            preset["sha256"],
            "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
        )
        self.assertEqual(preset["size"], 1956193000)
        self.assertEqual(preset["workflow"], "ref2va")
        self.assertEqual(preset["runtime"], "standard_lora")
        self.assertEqual(preset["steps"], 4)
        self.assertEqual(preset["weight"], 1.0)
        self.assertEqual(preset["reference_detail"], "match")
        self.assertEqual(preset["generation_settings"]["sample_solver"], "euler")
        self.assertEqual(preset["generation_settings"]["flow_shift"], 12.0)
        self.assertEqual(preset["generation_settings"]["audio_flow_shift"], 3.0)
        self.assertEqual(preset["lora_contract"]["ordinary"], {
            "rank": 128,
            "alpha": 8,
            "intrinsic_scale": 0.0625,
        })
        self.assertEqual(preset["lora_contract"]["fused_qkv"], {
            "rank": 384,
            "alpha": 24,
            "intrinsic_scale": 0.0625,
        })
        self.assertAlmostEqual(8 / 128, 24 / 384)
        self.assertTrue(is_minimax_h3_turbo_lora(TURBO_FILENAME))
        self.assertFalse(is_minimax_h3_pdd_lora(TURBO_FILENAME))
        self.assertEqual(find_minimax_h3_accelerators([TURBO_FILENAME]), [TURBO_FILENAME])
        self.assertEqual(
            len(
                find_minimax_h3_accelerators(
                    [TURBO_FILENAME, "MiniMax-H3-Ref2VA-Acc-8Step.safetensors"]
                )
            ),
            2,
        )

    def test_model_default_selects_recipe_and_disabling_uses_twenty_steps(self):
        from models.minimax_h3.turbo import normalize_minimax_h3_turbo_request

        model_def = _read_default()["model"]
        body = {
            "minimax_h3_turbo_mode": True,
            "activated_loras": ["character.safetensors"],
            "loras_multipliers": "0.45",
        }
        self.assertTrue(
            normalize_minimax_h3_turbo_request(
                body,
                full_checkpoint=False,
                workflow="ref2va",
                model_def=model_def,
            )
        )
        self.assertEqual(body["minimax_h3_turbo_preset"], TURBO_ID)
        self.assertEqual(
            body["activated_loras"],
            ["character.safetensors", TURBO_FILENAME],
        )
        self.assertEqual(body["loras_multipliers"], "0.45 1.00")
        self.assertEqual(body["num_inference_steps"], 4)
        self.assertEqual(body["guidance_scale"], 1.0)
        self.assertEqual(body["flow_shift"], 12.0)
        self.assertEqual(body["audio_flow_shift"], 3.0)
        self.assertEqual(body["sample_solver"], "euler")
        self.assertEqual(body["minimax_h3_reference_detail"], "match")

        disabled = {"minimax_h3_turbo_mode": False}
        self.assertFalse(
            normalize_minimax_h3_turbo_request(
                disabled,
                full_checkpoint=False,
                workflow="ref2va",
                model_def=model_def,
            )
        )
        self.assertEqual(disabled["num_inference_steps"], 20)
        explicit_steps = {
            "minimax_h3_turbo_mode": False,
            "num_inference_steps": 7,
        }
        normalize_minimax_h3_turbo_request(
            explicit_steps,
            full_checkpoint=False,
            workflow="ref2va",
            model_def=model_def,
        )
        self.assertEqual(explicit_steps["num_inference_steps"], 7)

    def test_only_declared_model_default_enables_turbo_when_mode_is_omitted(self):
        from models.minimax_h3.turbo import normalize_minimax_h3_turbo_request

        body = {}
        self.assertTrue(
            normalize_minimax_h3_turbo_request(
                body,
                full_checkpoint=False,
                workflow="ref2va",
                model_def=_read_default()["model"],
            )
        )
        self.assertIs(body["minimax_h3_turbo_mode"], True)
        self.assertEqual(body["minimax_h3_turbo_preset"], TURBO_ID)
        self.assertEqual(body["num_inference_steps"], 4)

        untouched = {}
        self.assertFalse(
            normalize_minimax_h3_turbo_request(
                untouched,
                full_checkpoint=False,
                workflow="ref2va",
                model_def={},
            )
        )
        self.assertEqual(untouched, {})


class TestSingularityCheckpointAndLoraContracts(unittest.TestCase):
    def test_checkpoint_gate_requires_native_grouped_int8_convrot(self):
        from models.minimax_h3.singularity import (
            validate_minimax_h3_singularity_checkpoint,
        )

        compatible = {
            "compressed_modulation": True,
            "adaln_curve_grid": 1025,
            "time_embed_dim": 8,
            "convrot": True,
            "quantization_format": "int8_tensorwise",
            "convrot_group_size": 256,
        }
        validate_minimax_h3_singularity_checkpoint(compatible, "grouped")
        for bad_probe, layout, message in (
            ({**compatible, "convrot": False}, "grouped", "ConvRot"),
            ({**compatible, "quantization_format": "fp8"}, "grouped", "int8_tensorwise"),
            ({**compatible, "convrot_group_size": 16}, "grouped", "group size 256"),
            ({**compatible, "adaln_curve_grid": 32}, "grouped", "[1025, 8]"),
            (compatible, "interleaved", "grouped"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_minimax_h3_singularity_checkpoint(bad_probe, layout)

    def test_real_quant_metadata_stubs_are_read_from_tiny_marker_only(self):
        import torch
        from mmgp import quant_router
        from safetensors.torch import save_file

        from models.minimax_h3.convrot_layout import (
            convrot_quantization_info_from_file,
        )

        descriptor = json.dumps(
            {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        marker = torch.tensor(list(descriptor), dtype=torch.uint8)
        with self.subTest("metadata-only loader returns descriptor stubs"):
            # Keep this checkpoint tiny: the 1025-row table exercises the
            # actual H3 shape probe while its marker payload remains small.
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                checkpoint_path = Path(directory) / "singularity_probe.safetensors"
                save_file(
                    {
                        "model.diffusion_model.adaln_t_table": torch.zeros(
                            (1025, 8), dtype=torch.float16
                        ),
                        "model.diffusion_model.blocks.0.attn.qkv_proj.weight": torch.zeros(
                            (12, 4), dtype=torch.int8
                        ),
                        "model.diffusion_model.blocks.0.attn.qkv_proj.comfy_quant": marker,
                    },
                    str(checkpoint_path),
                )
                metadata_state, _ = quant_router.load_metadata_state_dict(
                    str(checkpoint_path)
                )
                stub = metadata_state[
                    "model.diffusion_model.blocks.0.attn.qkv_proj.comfy_quant"
                ]
                self.assertFalse(torch.is_tensor(stub))

                info = convrot_quantization_info_from_file(str(checkpoint_path))
                self.assertTrue(info["convrot"])
                self.assertEqual(info["quantization_format"], "int8_tensorwise")
                self.assertEqual(info["convrot_group_size"], 256)

    def test_existing_namespace_converter_preserves_lightx2v_intrinsic_scale(self):
        import torch

        from models.minimax_h3.lora_vdn import normalize_diffusers_lora

        class FusedAttentionStub:
            use_adaln_curves = False

            @staticmethod
            def get_submodule(_name):
                return object()

        down = torch.ones((128, 4), dtype=torch.float32)
        up = torch.ones((2, 128), dtype=torch.float32)
        diffusers = {}
        for projection in ("q", "k", "v"):
            prefix = f"transformer_blocks.0.attn.to_{projection}"
            diffusers[f"{prefix}.lora_A.weight"] = down.clone()
            diffusers[f"{prefix}.lora_B.weight"] = up.clone()
            diffusers[f"{prefix}.alpha"] = torch.tensor(8.0)

        converted = normalize_diffusers_lora(diffusers, FusedAttentionStub())
        fused_down = converted["blocks.0.attn.qkv_proj.lora_A.weight"]
        fused_up = converted["blocks.0.attn.qkv_proj.lora_B.weight"]
        self.assertEqual(tuple(fused_down.shape), (384, 4))
        self.assertEqual(tuple(fused_up.shape), (6, 384))
        self.assertTrue(torch.allclose(fused_up[:2, :128], torch.full((2, 128), 0.0625)))
        self.assertTrue(torch.allclose(fused_up[2:4, 128:256], torch.full((2, 128), 0.0625)))
        self.assertTrue(torch.allclose(fused_up[4:, 256:], torch.full((2, 128), 0.0625)))
        self.assertTrue(torch.equal(fused_up[:2, 128:], torch.zeros((2, 256))))

        comfy = {
            "blocks.0.attn.qkv_proj.lora_A.weight": torch.ones((384, 4)),
            "blocks.0.attn.qkv_proj.lora_B.weight": torch.ones((6, 384)),
            "blocks.0.attn.qkv_proj.alpha": torch.tensor(24.0),
        }
        direct = normalize_diffusers_lora(comfy, FusedAttentionStub())
        self.assertEqual(tuple(direct["blocks.0.attn.qkv_proj.lora_A.weight"].shape), (384, 4))
        self.assertEqual(float(direct["blocks.0.attn.qkv_proj.alpha"]), 24.0)


if __name__ == "__main__":
    unittest.main()
