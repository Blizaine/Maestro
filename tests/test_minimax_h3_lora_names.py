"""CPU regressions for flattened MiniMax H3 LoRA target names."""

from pathlib import Path
import sys
import unittest


APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import torch
from torch import nn

from models.minimax_h3.lora_names import normalize_flattened_lora_names


class _Attention(nn.Module):
    def __init__(self, split_qkv=False):
        super().__init__()
        if split_qkv:
            for projection in ("q_proj", "k_proj", "v_proj"):
                setattr(self, projection, nn.Linear(4, 4, bias=False))
        else:
            self.qkv_proj = nn.Linear(4, 12, bias=False)
        self.out_proj = nn.Linear(4, 4, bias=False)


class _Block(nn.Module):
    def __init__(self, split_qkv=False):
        super().__init__()
        self.attn = _Attention(split_qkv)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(4, 8, bias=False)
        self.mlp.fc2 = nn.Linear(4, 4, bias=False)
        self.adaln_proj = nn.Module()
        self.adaln_proj.linear = nn.Linear(4, 8, bias=False)


class _TinyH3(nn.Module):
    def __init__(self, split_qkv=False):
        super().__init__()
        self.use_adaln_curves = False
        self.blocks = nn.ModuleList([_Block(split_qkv)])
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList([_Block(split_qkv)])


class MiniMaxH3LoRANamesTests(unittest.TestCase):
    def setUp(self):
        self.model = _TinyH3()

    def test_maps_supported_out_qkv_mlp_adaln_and_token_refiner_targets(self):
        tensors = {
            "out": torch.ones(2, 4),
            "qkv": torch.ones(12, 2),
            "fc1": torch.ones(8, 2),
            "adaln": torch.ones(2, 4),
            "refiner": torch.ones(4, 2),
            "alpha": torch.tensor(16.0),
        }
        source = {
            "lora_unet_blocks_0_attn_out_proj.lora_down.weight": tensors["out"],
            "lora_unet_blocks_0_attn_qkv_proj.lora_up.weight": tensors["qkv"],
            "lora_unet_blocks_0_mlp_fc1.lora_down.weight": tensors["fc1"],
            "lora_unet_blocks_0_adaln_proj_linear.lora_down.weight": tensors["adaln"],
            "lora_unet_token_refiner_blocks_0_attn_out_proj.lora_up.weight": tensors["refiner"],
            "lora_unet_blocks_0_attn_qkv_proj.alpha": tensors["alpha"],
        }

        result = normalize_flattened_lora_names(source, self.model)

        self.assertEqual(set(result), {
            "blocks.0.attn.out_proj.lora_down.weight",
            "blocks.0.attn.qkv_proj.lora_up.weight",
            "blocks.0.mlp.fc1.lora_down.weight",
            "blocks.0.adaln_proj.linear.lora_down.weight",
            "token_refiner.blocks.0.attn.out_proj.lora_up.weight",
            "blocks.0.attn.qkv_proj.alpha",
        })
        self.assertIs(result["blocks.0.attn.out_proj.lora_down.weight"], tensors["out"])
        self.assertIs(result["blocks.0.attn.qkv_proj.lora_up.weight"], tensors["qkv"])
        self.assertIs(result["blocks.0.mlp.fc1.lora_down.weight"], tensors["fc1"])
        self.assertIs(result["blocks.0.adaln_proj.linear.lora_down.weight"], tensors["adaln"])
        self.assertIs(result["token_refiner.blocks.0.attn.out_proj.lora_up.weight"], tensors["refiner"])
        self.assertIs(result["blocks.0.attn.qkv_proj.alpha"], tensors["alpha"])
        self.assertTrue(all(key.startswith("lora_unet_") for key in source))

    def test_qkv_path_is_available_when_attention_is_split_and_unknown_paths_stay_unknown(self):
        model = _TinyH3(split_qkv=True)
        value = torch.ones(12, 2)
        unknown = torch.ones(2, 4)
        source = {
            "lora_unet_blocks_0_attn_qkv_proj.lora_up.weight": value,
            "lora_unet_blocks_99_attn_out_proj.lora_down.weight": unknown,
        }

        result = normalize_flattened_lora_names(source, model)

        self.assertIn("blocks.0.attn.qkv_proj.lora_up.weight", result)
        self.assertIs(result["blocks.0.attn.qkv_proj.lora_up.weight"], value)
        self.assertIn("lora_unet_blocks_99_attn_out_proj.lora_down.weight", result)
        self.assertIs(result["lora_unet_blocks_99_attn_out_proj.lora_down.weight"], unknown)

    def test_dotted_and_peft_names_and_models_without_module_inventory_are_unchanged(self):
        values = [torch.ones(1) for _ in range(4)]
        source = {
            "blocks.0.attn.out_proj.lora_A.default.weight": values[0],
            "diffusion_model.blocks.0.attn.out_proj.lora_down.weight": values[1],
            "transformer.blocks.0.attn.qkv_proj.lora_B.weight": values[2],
            "transformer_blocks.0.attn.to_q.lora_A.weight": values[3],
        }

        result = normalize_flattened_lora_names(source, self.model)
        no_inventory = normalize_flattened_lora_names(source, object())

        self.assertEqual(set(result), set(source))
        self.assertEqual(set(no_inventory), set(source))
        for key, value in source.items():
            self.assertIs(result[key], value)
            self.assertIs(no_inventory[key], value)

    def test_ambiguous_underscore_paths_are_rejected(self):
        model = nn.Module()
        model.foo_bar = nn.Module()
        model.foo_bar.baz = nn.Linear(4, 4)
        model.foo = nn.Module()
        model.foo.bar_baz = nn.Linear(4, 4)

        with self.assertRaisesRegex(ValueError, "Ambiguous flattened H3 LoRA module"):
            normalize_flattened_lora_names(
                {"lora_unet_foo_bar_baz.lora_down.weight": torch.ones(1, 4)},
                model,
            )

    def test_flat_alias_collision_is_rejected_after_existing_diffusers_mapping(self):
        from models.minimax_h3.lora_vdn import normalize_diffusers_lora

        cases = (
            {
                "transformer_blocks.0.attn.to_out.0.lora_A.weight": torch.ones(4, 2),
                "lora_unet_blocks_0_attn_out_proj.lora_down.default.weight": torch.ones(2, 4),
            },
            {
                "lora_unet_blocks_0_attn_out_proj.lora_down.default.weight": torch.ones(2, 4),
                "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": torch.ones(4, 2),
            },
        )
        for source in cases:
            with self.subTest(keys=tuple(source)), self.assertRaisesRegex(
                ValueError, "Flattened H3 LoRA key collision"
            ):
                converted = normalize_diffusers_lora(source, self.model)
                normalize_flattened_lora_names(converted, self.model)


if __name__ == "__main__":
    unittest.main()
