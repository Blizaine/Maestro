"""CPU checks for checkpoint-shaped H3 compressed AdaLN precision."""

from __future__ import annotations

import gc
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


def _tiny_transformer():
    from models.minimax_h3.transformer import MiniMaxH3AdaLNProjection

    class TinyBlock(torch.nn.Module):
        def __init__(self, outputs):
            super().__init__()
            self.adaln_proj = MiniMaxH3AdaLNProjection(
                curve_dim=8,
                hidden_size=2,
                outputs=outputs,
                modalities=1,
                dtype=torch.float16,
            )

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_adaln_curves = True
            self.blocks = torch.nn.ModuleList(
                [TinyBlock(outputs=3), TinyBlock(outputs=3)]
            )
            self.final_layer = TinyBlock(outputs=2)

        def set_qkv_layout(self, _layout):
            pass

    return TinyTransformer()


class H3AdaLNPrecisionTests(unittest.TestCase):
    def test_probe_finds_only_fp32_compressed_adaln_weights_after_wrapper_cleanup(self):
        from models.minimax_h3 import minimax_h3_main

        state_dict = {
            "model.diffusion_model.adaln_t_table": torch.empty(
                (1025, 8), device="meta", dtype=torch.float32
            ),
            "module.diffusion_model.final_layer.adaln_proj.linear.weight": torch.empty(
                (8, 8), device="meta", dtype=torch.float32
            ),
            "model.diffusion_model.blocks.1.adaln_proj.linear.weight": torch.empty(
                (24, 8), device="meta", dtype=torch.float16
            ),
            "blocks.3.adaln_proj.linear.bias": torch.empty(
                (24,), device="meta", dtype=torch.float32
            ),
            "blocks.2.attn.qkv_proj.weight": torch.empty(
                (12, 8), device="meta", dtype=torch.float32
            ),
            "blocks.51.adaln_proj.linear.weight": torch.empty(
                (24, 8), device="meta", dtype=torch.float32
            ),
        }
        state_dict.update(
            {
                f"model.diffusion_model.blocks.{index}.adaln_proj.linear.weight": torch.empty(
                    (24, 8), device="meta", dtype=torch.float32
                )
                for index in range(50)
            }
        )
        state_dict[
            "model.diffusion_model.blocks.1.adaln_proj.linear.weight"
        ] = torch.empty((24, 8), device="meta", dtype=torch.float16)
        with mock.patch.object(
            minimax_h3_main.quant_router,
            "load_metadata_state_dict",
            return_value=(
                state_dict,
                {"ss_h3_adaln_layout": "pruned", "ss_h3_adaln_rank": "16"},
            ),
        ):
            profile = minimax_h3_main.probe_h3_checkpoint("metadata-only.safetensors")

        self.assertTrue(profile["compressed_modulation"])
        self.assertEqual(profile["adaln_curve_grid"], 1025)
        self.assertEqual(profile["time_embed_dim"], 8)
        self.assertEqual(
            profile["fp32_adaln_modules"],
            sorted(
                [
                    f"blocks.{index}.adaln_proj.linear"
                    for index in range(50)
                    if index != 1
                ]
                + ["final_layer.adaln_proj.linear"]
            ),
        )

    def test_full_checkpoint_probe_has_no_compressed_adaln_dtype_overrides(self):
        from models.minimax_h3 import minimax_h3_main

        state_dict = {
            "model.diffusion_model.time_embedder.proj_in.weight": torch.empty(
                (8, 4), device="meta", dtype=torch.float32
            ),
        }
        with mock.patch.object(
            minimax_h3_main.quant_router,
            "load_metadata_state_dict",
            return_value=(state_dict, {}),
        ):
            profile = minimax_h3_main.probe_h3_checkpoint("full.safetensors")

        self.assertFalse(profile["compressed_modulation"])
        self.assertEqual(profile["fp32_adaln_modules"], [])

    def test_transformer_loader_applies_probe_dtype_overrides_before_mmgp_load(self):
        from contextlib import nullcontext

        from models.minimax_h3 import minimax_h3_main

        checkpoint = {
            "compressed_modulation": True,
            "adaln_curve_grid": 1025,
            "time_embed_dim": 8,
            "quantization_format": "none",
            "fp32_adaln_modules": ["blocks.0.adaln_proj.linear"],
        }
        model = _tiny_transformer()
        observed = {}

        def construct_transformer(**_kwargs):
            return model

        def load_model_data(target, *_args, **_kwargs):
            observed["target"] = target
            observed["fp32_dtype"] = target.blocks[0].adaln_proj.linear._lock_dtype
            observed["stock_dtype"] = target.blocks[1].adaln_proj.linear._lock_dtype

        with (
            mock.patch.object(
                minimax_h3_main,
                "probe_h3_checkpoint",
                return_value=checkpoint,
            ),
            mock.patch.object(
                minimax_h3_main,
                "MiniMaxH3Transformer",
                side_effect=construct_transformer,
            ),
            mock.patch.object(
                minimax_h3_main,
                "init_empty_weights",
                return_value=nullcontext(),
            ),
            mock.patch.object(
                minimax_h3_main.offload,
                "load_model_data",
                side_effect=load_model_data,
            ) as load,
        ):
            result = minimax_h3_main._load_transformer(
                "probe-only.safetensors",
                torch.bfloat16,
                qkv_layout="contiguous",
            )

        self.assertIs(result, model)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(observed["fp32_dtype"], torch.float32)
        self.assertEqual(observed["stock_dtype"], torch.float16)

    def test_mmgp_preserves_selected_fp32_weights_and_stock_fp16_adaln(self):
        from mmgp import offload
        from models.minimax_h3.minimax_h3_main import _lock_fp32_adaln_modules
        from safetensors.torch import save_file

        model = _tiny_transformer()
        _lock_fp32_adaln_modules(
            model,
            [
                "blocks.0.adaln_proj.linear",
                "final_layer.adaln_proj.linear",
            ],
        )

        block_weight = torch.linspace(-0.7, 0.9, 6 * 8).reshape(6, 8)
        block_bias = torch.linspace(-0.3, 0.4, 6, dtype=torch.float16)
        stock_weight = torch.linspace(-0.5, 0.6, 6 * 8).reshape(6, 8).half()
        stock_bias = torch.linspace(-0.2, 0.2, 6, dtype=torch.float16)
        final_weight = torch.linspace(-0.8, 0.5, 4 * 8).reshape(4, 8)
        final_bias = torch.linspace(-0.15, 0.15, 4, dtype=torch.float16)
        state_dict = {
            "blocks.0.adaln_proj.linear.weight": block_weight,
            "blocks.0.adaln_proj.linear.bias": block_bias,
            "blocks.1.adaln_proj.linear.weight": stock_weight,
            "blocks.1.adaln_proj.linear.bias": stock_bias,
            "final_layer.adaln_proj.linear.weight": final_weight,
            "final_layer.adaln_proj.linear.bias": final_bias,
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny_adaln.safetensors"
            save_file(state_dict, str(checkpoint))
            offload.load_model_data(
                model,
                str(checkpoint),
                writable_tensors=False,
                default_dtype=torch.bfloat16,
                verboseLevel=0,
            )

            loaded = model.blocks[0].adaln_proj.linear
            self.assertEqual(loaded.weight.dtype, torch.float32)
            self.assertTrue(torch.equal(loaded.weight, block_weight))
            self.assertEqual(loaded.bias.dtype, torch.float32)
            self.assertTrue(torch.equal(loaded.bias, block_bias.float()))

            curve = torch.linspace(-0.4, 0.7, 16).reshape(2, 8)
            projected = torch.cat(model.blocks[0].adaln_proj(curve), dim=-1)
            expected = torch.nn.functional.linear(
                curve,
                block_weight,
                block_bias.float(),
            )
            torch.testing.assert_close(projected, expected, rtol=0, atol=0)

            stock = model.blocks[1].adaln_proj.linear
            self.assertEqual(stock.weight.dtype, torch.float16)
            self.assertTrue(torch.equal(stock.weight, stock_weight))
            self.assertEqual(stock.bias.dtype, torch.float16)
            self.assertTrue(torch.equal(stock.bias, stock_bias))

            final = model.final_layer.adaln_proj.linear
            self.assertEqual(final.weight.dtype, torch.float32)
            self.assertTrue(torch.equal(final.weight, final_weight))
            self.assertEqual(final.bias.dtype, torch.float32)
            self.assertTrue(torch.equal(final.bias, final_bias.float()))

            del loaded, stock, final, projected, expected, model
            gc.collect()

    def test_rank16_lora_rank_passes_through_width8_adaln(self):
        from models.minimax_h3.lora_affine import convert_adaln_loras

        prefix = "blocks.0.adaln_proj.linear"
        down = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
        up = torch.arange(24 * 16, dtype=torch.float32).reshape(24, 16)
        state_dict = {
            f"{prefix}.lora_A.weight": down.clone(),
            f"{prefix}.lora_B.weight": up.clone(),
        }

        count, architecture, source_width, target_width = convert_adaln_loras(
            "minimax_h3_ref2va",
            state_dict,
            torch.empty((1025, 8), dtype=torch.float32),
        )

        self.assertEqual((count, architecture), (0, "ref2va"))
        self.assertEqual((source_width, target_width), (8, 8))
        self.assertEqual(state_dict[f"{prefix}.lora_A.weight"].shape, (16, 8))
        self.assertEqual(state_dict[f"{prefix}.lora_B.weight"].shape, (24, 16))
        self.assertTrue(torch.equal(state_dict[f"{prefix}.lora_A.weight"], down))
        self.assertTrue(torch.equal(state_dict[f"{prefix}.lora_B.weight"], up))


if __name__ == "__main__":
    unittest.main()
