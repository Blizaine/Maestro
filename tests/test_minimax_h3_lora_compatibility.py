"""CPU regression coverage for MiniMax H3 LoRA compatibility checks."""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class H3AdaLNAffineRoundingTests(unittest.TestCase):
    def test_rounding_fit_stays_on_cpu_under_another_default_device(self):
        module, canonical, target, affine = self._tables()
        with patch.object(module, "_load_affine_package", return_value=(canonical, affine)):
            # Meta reproduces accidental default-device allocations without CUDA.
            with torch.device("meta"):
                rounding_floor = module._bf16_rounding_fit_error(canonical, target)
                result = module._aligned_affine_map("ref2va", target)
        self.assertIsNotNone(rounding_floor)
        self.assertTrue(math.isfinite(rounding_floor))
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.isfinite(result).all())

    def test_row_count_interpolation_stays_on_cpu_under_meta_default(self):
        from torch.nn import functional as F

        from models.minimax_h3 import lora_affine

        generator = torch.Generator().manual_seed(205)
        canonical = torch.randn(19, 8, generator=generator, dtype=torch.float64)
        # Independent CPU interpolation fixture: the checkpoint curve is sampled
        # at a different row count before a known invertible affine transform.
        sampled = F.interpolate(
            canonical.T.unsqueeze(0),
            size=37,
            mode="linear",
            align_corners=True,
        ).squeeze(0).T.contiguous()
        basis = torch.eye(8, dtype=torch.float64) + torch.randn(
            8, 8, generator=generator, dtype=torch.float64
        ) * 0.12
        offset = torch.randn(8, generator=generator, dtype=torch.float64) * 0.08
        target = sampled @ basis + offset
        affine = torch.randn(
            9, lora_affine.FULL_TIME_DIM, generator=generator, dtype=torch.float64
        )
        self.assertEqual(int(torch.linalg.matrix_rank(torch.cat(
            (sampled, torch.ones(sampled.shape[0], 1, dtype=torch.float64)), dim=1
        ))), 9)
        self.assertEqual(int(torch.linalg.matrix_rank(basis)), 8)

        with patch.object(lora_affine, "_load_affine_package", return_value=(canonical, affine)):
            # A default meta factory catches an interpolation grid that does not
            # explicitly follow the CPU checkpoint table's device.
            with torch.device("meta"):
                result = lora_affine._aligned_affine_map("ref2va", target)

        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.isfinite(result).all())
        target_h = torch.cat((target, torch.ones(37, 1, dtype=torch.float64)), dim=1)
        sampled_h = torch.cat((sampled, torch.ones(37, 1, dtype=torch.float64)), dim=1)
        actual_modulation = target_h @ result.double()
        expected_modulation = sampled_h @ affine
        torch.testing.assert_close(actual_modulation, expected_modulation, rtol=2e-5, atol=2e-5)

    def _tables(self):
        from models.minimax_h3 import lora_affine

        generator = torch.Generator().manual_seed(104)
        canonical = torch.randn(1025, 8, generator=generator) * 0.2
        change_of_basis = torch.eye(8) + torch.randn(
            8, 8, generator=generator
        ) * 0.1
        offset = torch.randn(8, generator=generator) * 0.05
        target = (canonical @ change_of_basis + offset).to(
            torch.bfloat16
        ).float()
        affine = torch.randn(
            9,
            lora_affine.FULL_TIME_DIM,
            generator=generator,
        )
        return lora_affine, canonical, target, affine

    def test_bf16_quantized_affine_curve_uses_measured_rounding_budget(self):
        module, canonical, target, affine = self._tables()
        affine = torch.zeros_like(affine)
        affine[:9, :9] = torch.eye(9)
        with patch.object(
            module,
            "_load_affine_package",
            return_value=(canonical, affine),
        ):
            result = module._aligned_affine_map("ref2va", target)

        self.assertEqual(result.shape, affine.shape)
        self.assertTrue(torch.isfinite(result).all())
        target_h = torch.cat((target.double(), torch.ones(1025, 1, dtype=torch.float64)), dim=1)
        canonical_h = torch.cat((canonical.double(), torch.ones(1025, 1, dtype=torch.float64)), dim=1)
        actual_modulation = target_h @ result.double()
        expected_modulation = canonical_h @ affine.double()
        relative_modulation_error = (
            torch.linalg.vector_norm(actual_modulation - expected_modulation)
            / torch.linalg.vector_norm(expected_modulation)
        )
        measured_budget = module._bf16_rounding_fit_error(canonical, target)
        self.assertIsNotNone(measured_budget)
        self.assertLessEqual(
            float(relative_modulation_error),
            measured_budget * module._BF16_ROUNDING_BUDGET_FACTOR + 1e-7,
        )
        self.assertGreater(float(torch.linalg.vector_norm(actual_modulation)), 0.1)

    def test_non_affine_bf16_curve_perturbation_stays_incompatible(self):
        module, canonical, target, affine = self._tables()
        perturbation = torch.sin(torch.arange(target.shape[0]) * 0.037) * 0.05
        incompatible = target.clone()
        incompatible[:, 0] = (
            incompatible[:, 0] + perturbation
        ).to(torch.bfloat16).float()
        with patch.object(
            module,
            "_load_affine_package",
            return_value=(canonical, affine),
        ):
            with self.assertRaisesRegex(ValueError, "incompatible with the canonical LoRA map"):
                module._aligned_affine_map("ref2va", incompatible)

    def test_wrong_canonical_basis_stays_incompatible_even_for_bf16_curve(self):
        module, canonical, target, affine = self._tables()
        wrong_basis = canonical.clone()
        wrong_basis[:, 0] += torch.sin(torch.arange(target.shape[0]) * 0.037) * 0.05
        with patch.object(
            module,
            "_load_affine_package",
            return_value=(wrong_basis, affine),
        ):
            with self.assertRaisesRegex(ValueError, "incompatible with the canonical LoRA map"):
                module._aligned_affine_map("ref2va", target)

    def test_float32_curve_keeps_the_original_strict_fit_guard(self):
        module, canonical, target, affine = self._tables()
        non_bf16 = target.clone()
        non_bf16[400, 0] += 0.0002
        self.assertFalse(
            torch.equal(non_bf16, non_bf16.to(torch.bfloat16).float())
        )
        with patch.object(
            module,
            "_load_affine_package",
            return_value=(canonical, affine),
        ):
            with self.assertRaisesRegex(ValueError, "incompatible with the canonical LoRA map"):
                module._aligned_affine_map("ref2va", non_bf16)


class H3LoraCompatibilityTests(unittest.TestCase):
    def test_vdn_is_rejected_on_standard_h3_but_allowed_on_vdn_model(self):
        from models.minimax_h3.lora_compatibility import (
            minimax_h3_lora_incompatibility,
            validate_minimax_h3_loras,
        )

        adapter = r"C:\loras\minimax_h3_vdn_turbo_8step_lora_bf16.safetensors"
        standard = {"architecture": "minimax_h3_ref2va"}
        self.assertIn(
            "dedicated MiniMax H3 VDN model",
            minimax_h3_lora_incompatibility(standard, adapter),
        )
        validate_minimax_h3_loras(
            ["film_style.safetensors"],
            standard,
        )
        validate_minimax_h3_loras([adapter], {**standard, "vdn": True})
        with self.assertRaisesRegex(ValueError, "minimax_h3_vdn_turbo"):
            validate_minimax_h3_loras([adapter], standard)

    def test_baked_turbo_uses_existing_acceleration_policy_and_keeps_styles(self):
        from models.minimax_h3.lora_compatibility import (
            minimax_h3_lora_incompatibility,
            validate_minimax_h3_loras,
        )

        ordinary = r"C:\loras\minimax_h3\H3_Cinematic.safetensors"
        turbo = r"C:\loras\minimax_h3\minimax_h3_turbo_v4_step600.safetensors"
        pdd = r"C:\loras\minimax_h3\MiniMax-H3-Ref2VA-Acc-8Step.safetensors"
        hybrid = {"architecture": "minimax_h3_ref2va"}
        baked = {**hybrid, "minimax_h3_baked_turbo": True}
        baked_import = {
            **hybrid,
            "minimax_h3_import_profile": {"baked_turbo": True},
        }

        self.assertIsNone(minimax_h3_lora_incompatibility(hybrid, turbo))
        validate_minimax_h3_loras([turbo], hybrid)
        validate_minimax_h3_loras([ordinary], baked)
        self.assertIn(
            "already baked",
            minimax_h3_lora_incompatibility(baked_import, turbo),
        )
        for adapter in (turbo, pdd):
            with self.subTest(adapter=adapter):
                reason = minimax_h3_lora_incompatibility(baked, adapter)
                self.assertIn("already baked", reason)
                with self.assertRaisesRegex(ValueError, "cannot use"):
                    validate_minimax_h3_loras([adapter], baked)

    def test_vdn_name_detection_does_not_reject_style_name_in_h3_folder(self):
        from models.minimax_h3.lora_compatibility import (
            is_minimax_h3_vdn_lora,
            minimax_h3_lora_incompatibility,
        )

        style = r"C:\app\loras\minimax_h3\film_vdn_reference_style.safetensors"
        self.assertFalse(is_minimax_h3_vdn_lora(style))
        self.assertIsNone(
            minimax_h3_lora_incompatibility(
                {"architecture": "minimax_h3_ref2va"}, style
            )
        )

    def test_generation_model_validates_vdn_before_lora_loading(self):
        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        model = object.__new__(MiniMaxH3Model)
        model.model_def = {"architecture": "minimax_h3_ref2va"}
        model.release_special_loras = lambda: None
        with self.assertRaisesRegex(ValueError, "dedicated MiniMax H3 VDN model"):
            model.validate_loras(
                ["minimax_h3_vdn_turbo_8step_lora_bf16.safetensors"]
            )

    def test_tensor_header_identifies_renamed_vdn_and_preserves_style_loras(self):
        from models.minimax_h3.lora_compatibility import (
            is_minimax_h3_vdn_lora,
            minimax_h3_lora_incompatibility,
        )

        with tempfile.TemporaryDirectory() as directory:
            renamed = Path(directory) / "renamed_adapter.safetensors"
            header = {
                "blocks.0.attn.linear_attention.to_q.lora_A.weight": {
                    "dtype": "BF16",
                    "shape": [8, 8],
                }
            }
            encoded = json.dumps(header).encode("utf-8")
            renamed.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
            self.assertTrue(is_minimax_h3_vdn_lora(str(renamed)))
            self.assertIsNotNone(
                minimax_h3_lora_incompatibility(
                    {"architecture": "minimax_h3"},
                    str(renamed),
                )
            )

        self.assertFalse(
            is_minimax_h3_vdn_lora("film_vdn_reference_style.safetensors")
        )

    def test_fasth3_opt_in_keeps_ordinary_loras_and_excludes_special_families(self):
        from models.minimax_h3.lora_compatibility import (
            fasth3_lora_incompatibility,
            validate_fasth3_loras,
        )

        validate_fasth3_loras(
            ["film_style.safetensors"], enabled=True, omni_reference=False,
        )
        self.assertIsNone(
            fasth3_lora_incompatibility(
                "film_style.safetensors", omni_reference=False,
            )
        )
        for adapter, expected in (
            ("minimax_h3_turbo_v4_step600.safetensors", "Turbo/acceleration"),
            ("minimax_h3_ref2va_acc_8step.safetensors", "PDD"),
            ("minimax_h3_vdn.safetensors", "VDN"),
            ("film_dora.safetensors", "DoRA"),
        ):
            with self.subTest(adapter=adapter):
                with self.assertRaisesRegex(ValueError, expected):
                    validate_fasth3_loras([adapter], enabled=True)

        for invalid_gate in (None, False, 1, "true"):
            with self.subTest(invalid_gate=invalid_gate):
                with self.assertRaisesRegex(ValueError, "experimental and off by default"):
                    validate_fasth3_loras(
                        ["film_style.safetensors"], enabled=invalid_gate,
                    )
        validate_fasth3_loras([], enabled=False)

    def test_fasth3_headers_detect_renamed_dora_vdn_and_omni_output_heads(self):
        from models.minimax_h3.lora_compatibility import (
            fasth3_lora_incompatibility,
            validate_fasth3_loras,
        )

        def write_header(path, tensors):
            encoded = json.dumps(tensors).encode("utf-8")
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renamed_vdn = root / "renamed_vdn_adapter.safetensors"
            write_header(renamed_vdn, {
                "blocks.0.attn.linear_attention.to_q.lora_A.weight": {
                    "dtype": "BF16", "shape": [8, 8],
                },
            })
            self.assertIn("VDN", fasth3_lora_incompatibility(str(renamed_vdn)))

            renamed_dora = root / "renamed_adapter.safetensors"
            write_header(renamed_dora, {
                "blocks.0.attn.qkv_proj.lora_magnitude_vector.weight": {
                    "dtype": "BF16", "shape": [8],
                },
            })
            self.assertIn("DoRA", fasth3_lora_incompatibility(str(renamed_dora)))

            omni_head = root / "ordinary_style.safetensors"
            write_header(omni_head, {
                "transformer.proj_out.lora_B.weight": {
                    "dtype": "BF16", "shape": [3072, 8],
                },
            })
            self.assertIsNone(
                fasth3_lora_incompatibility(str(omni_head), omni_reference=False)
            )
            self.assertIn(
                "32 PDD output heads",
                fasth3_lora_incompatibility(str(omni_head), omni_reference=True),
            )
            with self.assertRaisesRegex(ValueError, "video_out/audio_out"):
                validate_fasth3_loras(
                    [str(omni_head)], enabled=True, omni_reference=True,
                )

            flattened_omni_heads = root / "flattened_style.safetensors"
            write_header(flattened_omni_heads, {
                "lora_unet_final_layer_video_out.lora_A.weight": {
                    "dtype": "BF16", "shape": [8, 5376],
                },
                "lora_unet_final_layer_audio_out.lora_B.weight": {
                    "dtype": "BF16", "shape": [5376, 8],
                },
            })
            self.assertIsNone(
                fasth3_lora_incompatibility(
                    str(flattened_omni_heads), omni_reference=False,
                )
            )
            self.assertIn(
                "32 PDD output heads",
                fasth3_lora_incompatibility(
                    str(flattened_omni_heads), omni_reference=True,
                ),
            )

            # Import/picker accepts .sft, which uses the same safetensors
            # header format. Marker detection must not depend on the suffix.
            renamed_dora_sft = root / "renamed_adapter.sft"
            write_header(renamed_dora_sft, {
                "blocks.0.attn.qkv_proj.lora_magnitude_vector.weight": {
                    "dtype": "BF16", "shape": [8],
                },
            })
            self.assertIn("DoRA", fasth3_lora_incompatibility(str(renamed_dora_sft)))

            renamed_vdn_sft = root / "renamed_vdn_adapter.sft"
            write_header(renamed_vdn_sft, {
                "blocks.0.attn.linear_attention.to_q.lora_A.weight": {
                    "dtype": "BF16", "shape": [8, 8],
                },
            })
            self.assertIn("VDN", fasth3_lora_incompatibility(str(renamed_vdn_sft)))

            renamed_pdd_sft = root / "ordinary_name.sft"
            write_header(renamed_pdd_sft, {
                "proj_out.weight": {"dtype": "BF16", "shape": [32, 96, 5376]},
                "audio_proj_out.weight": {"dtype": "BF16", "shape": [32, 32, 5376]},
            })
            self.assertIn("PDD", fasth3_lora_incompatibility(str(renamed_pdd_sft)))

            renamed_pdd = root / "ordinary_name.safetensors"
            write_header(renamed_pdd, {
                "proj_out.weight": {"dtype": "BF16", "shape": [32, 96, 5376]},
                "audio_proj_out.weight": {"dtype": "BF16", "shape": [32, 32, 5376]},
            })
            self.assertIn("PDD", fasth3_lora_incompatibility(str(renamed_pdd)))

    def test_fast_model_opt_in_is_not_retained_between_jobs(self):
        from models.minimax_h3.minimax_h3_main import MiniMaxH3Model

        model = object.__new__(MiniMaxH3Model)
        model.model_def = {"architecture": "minimax_h3", "minimax_h3_fasth3": True}
        model.fasth3 = True
        model.fasth3_omniref = False
        model.release_special_loras = lambda: None
        model.validate_loras(
            ["film_style.safetensors"], minimax_h3_experimental_loras=True,
        )
        with self.assertRaisesRegex(ValueError, "experimental and off by default"):
            model.validate_loras(["film_style.safetensors"])


if __name__ == "__main__":
    unittest.main()
