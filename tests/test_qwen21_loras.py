"""CPU regressions for Qwen Image 2.1 fused and native-format LoRAs."""

import ast
import glob
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.qwen21.transformer_qwenimage21 import QwenImage21Transformer2DModel


def tiny_transformer():
    return QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=4,
        num_attention_heads=2,
        context_in_dim=4,
        mlp_ratio=3,
        axes_dims_rope=(2, 1, 1),
    ).eval()


def hook_feed_forward_linears_with_mmgp(model):
    from mmgp import offload

    manager = offload.offload.__new__(offload.offload)
    model._loras_model_data = {}
    model._loras_model_shortcuts = {}
    for name, module in model.named_modules():
        if name.endswith((".img_mlp.gate_layer", ".img_mlp.proj")):
            module._mm_manager = manager
            module.forward = manager.hook_lora(
                module,
                model,
                "transformer",
                model._loras_model_data,
                model._loras_model_shortcuts,
                name,
            )
    return model


def fused_lora_state_dict(lora_a, lora_b, alpha):
    prefix = "diffusion_model.transformer_blocks.0.img_mlp.gate_up"
    return {
        f"{prefix}.lora_A.weight": lora_a.contiguous(),
        f"{prefix}.lora_B.weight": lora_b.contiguous(),
        f"{prefix}.alpha": torch.tensor(float(alpha)),
    }


def load_adapter(model, state_dict, *, lora_strength=1.0):
    from mmgp import offload

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "adapter.safetensors")
        save_file(state_dict, path)
        loaded = offload.load_loras_into_model(
            model,
            [path],
            [lora_strength],
            pinnedLora=False,
            verboseLevel=0,
            split_linear_modules_map=model.split_linear_modules_map,
        )
        # MMGP memory-maps tensors from safetensors. Copy them before the
        # temporary adapter file is removed (Windows keeps mapped files open).
        for module_loras in model._loras_model_data.values():
            for adapter_data in module_loras.values():
                for index, tensor in enumerate(adapter_data):
                    if torch.is_tensor(tensor):
                        adapter_data[index] = tensor.clone()
        return loaded, model._loras_errors, model._loras_model_data


def install_cpu_lora_buffers(model):
    """Mirror MMGP's alias resolution while keeping adapter tensors on CPU."""
    for _, module in model.named_modules():
        for adapter_name, adapter_data in list(getattr(module, "_mm_lora_data", {}).items()):
            if adapter_name.endswith("_GPU"):
                continue
            cpu_data = list(adapter_data)
            for slot in range(min(4, len(cpu_data))):
                alias = cpu_data[slot]
                if isinstance(alias, str):
                    owner_name, owner_slot = alias.rsplit("#", 1)
                    cpu_data[slot] = model._loras_model_shortcuts[owner_name][adapter_name][
                        int(owner_slot)
                    ]
            module._mm_lora_data[adapter_name + "_GPU"] = cpu_data


class QwenImage21LoraTests(unittest.TestCase):
    def test_fused_gate_up_matches_split_swiglu_lora_math_and_alpha(self):
        torch.manual_seed(21)
        model = hook_feed_forward_linears_with_mmgp(tiny_transformer())
        ff = model.transformer_blocks[0].img_mlp
        hidden_size = model.inner_dim
        mlp_hidden_size = hidden_size * model.config.mlp_ratio
        rank = 2
        alpha = 6.0
        strength = 0.4
        lora_a = torch.randn(rank, hidden_size)
        lora_b = torch.randn(2 * mlp_hidden_size, rank)

        self.assertEqual(
            model.split_linear_modules_map["gate_up"],
            {
                "mapped_modules": ["gate_layer", "proj"],
                "split_sizes": [mlp_hidden_size, mlp_hidden_size],
            },
        )
        loaded, errors, lora_data = load_adapter(
            model, fused_lora_state_dict(lora_a, lora_b, alpha), lora_strength=strength
        )
        self.assertEqual(len(loaded), 1)
        self.assertEqual(errors, [])

        gate_data = lora_data[ff.gate_layer]["0"]
        proj_data = lora_data[ff.proj]["0"]
        if isinstance(gate_data[0], str):
            self.assertEqual(gate_data[0], "transformer_blocks.0.img_mlp.proj#0")
            shared_a = proj_data[0]
        else:
            self.assertEqual(proj_data[0], "transformer_blocks.0.img_mlp.gate_layer#0")
            shared_a = gate_data[0]
        torch.testing.assert_close(shared_a, lora_a)
        torch.testing.assert_close(gate_data[1], lora_b[:mlp_hidden_size])
        torch.testing.assert_close(proj_data[1], lora_b[mlp_hidden_size:])
        self.assertAlmostEqual(gate_data[4], alpha / rank)
        self.assertAlmostEqual(proj_data[4], alpha / rank)

        install_cpu_lora_buffers(model)

        inputs = torch.randn(3, hidden_size)
        scale = strength * alpha / rank
        fused_base = F.linear(inputs, torch.cat((ff.gate_layer.weight, ff.proj.weight), dim=0))
        fused_delta = F.linear(F.linear(inputs, lora_a), lora_b)
        fused_gate, fused_proj = (fused_base + scale * fused_delta).chunk(2, dim=-1)
        expected = ff.out(ff.activation_fn(fused_gate) * fused_proj)
        with torch.inference_mode():
            actual = ff(inputs)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_native_split_gate_and_proj_lora_format_still_loads(self):
        model = hook_feed_forward_linears_with_mmgp(tiny_transformer())
        ff = model.transformer_blocks[0].img_mlp
        hidden_size = model.inner_dim
        mlp_hidden_size = hidden_size * model.config.mlp_ratio
        rank = 2
        state_dict = {}
        for name, module in (("gate_layer", ff.gate_layer), ("proj", ff.proj)):
            prefix = f"diffusion_model.transformer_blocks.0.img_mlp.{name}"
            state_dict[f"{prefix}.lora_A.weight"] = torch.randn(rank, hidden_size)
            state_dict[f"{prefix}.lora_B.weight"] = torch.randn(mlp_hidden_size, rank)
            state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

        loaded, errors, _ = load_adapter(model, state_dict)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(errors, [])

    def test_fused_adapter_with_incompatible_input_dimension_is_rejected(self):
        model = hook_feed_forward_linears_with_mmgp(tiny_transformer())
        hidden_size = model.inner_dim
        mlp_hidden_size = hidden_size * model.config.mlp_ratio
        state_dict = fused_lora_state_dict(
            torch.randn(2, hidden_size + 1),
            torch.randn(2 * mlp_hidden_size, 2),
            2.0,
        )

        loaded, errors, _ = load_adapter(model, state_dict)
        self.assertEqual(loaded, [])
        self.assertTrue(errors)
        self.assertIn("Lora A dimension is not compatible", errors[0][1])

    def test_lora_check_only_uses_transformer_split_map_by_default(self):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        setup_loras = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "setup_loras"
        )
        namespace = {
            "glob": glob,
            "os": os,
            "get_base_model_type": lambda model_type: model_type,
            "get_lora_dir": lambda _model_type: lora_dir,
            "get_lora_search_dirs": lambda _model_type: [lora_dir],
            "get_loras_preprocessor": lambda *_args: None,
            "extract_preset": lambda *_args: ([], "", "", False, ""),
        }

        class FakeOffload:
            split_map = None

            @classmethod
            def load_loras_into_model(cls, _transformer, paths, **kwargs):
                cls.split_map = kwargs.get("split_linear_modules_map")
                return paths

        namespace["offload"] = FakeOffload
        with tempfile.TemporaryDirectory() as directory:
            lora_dir = directory
            Path(directory, "fused.safetensors").touch()
            split_map = {"gate_up": {"mapped_modules": ["gate_layer", "proj"]}}
            transformer = SimpleNamespace(split_linear_modules_map=split_map)
            exec(
                compile(ast.Module(body=[setup_loras], type_ignores=[]), "wgp.py", "exec"),
                namespace,
            )
            namespace["setup_loras"]("qwen21", transformer, None, "")
            self.assertIs(FakeOffload.split_map, split_map)

            override_map = {"custom": {"mapped_modules": ["first", "second"]}}
            namespace["setup_loras"](
                "qwen21", transformer, None, "", split_linear_modules_map=override_map
            )
            self.assertIs(FakeOffload.split_map, override_map)


if __name__ == "__main__":
    unittest.main()
