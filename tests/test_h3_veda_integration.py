"""CPU seams verify Veda selection without compiling or running GPU kernels."""

from __future__ import annotations

import ast
from functools import partial
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import torch
from torch import nn
import torch.nn.functional as F
from models.minimax_h3.minimax_h3_handler import family_handler
from models.minimax_h3 import transformer as h3
from shared import attention, attention_kit


class DenseVedaStub(nn.Module):
    """Record absolute predictor head IDs while calculating a CPU reference."""
    def __init__(self):
        super().__init__()
        self.policy = SimpleNamespace(enabled=True)
        self.calls = []

    def forward(self, qkv, group):
        self.calls.append(tuple(group.q))
        q, k, v = qkv
        qkv.clear()
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                             v.transpose(1, 2)).transpose(1, 2).contiguous()


class TestVedaIntegration(unittest.TestCase):
    def tearDown(self):
        attention_kit.configure(0)

    def test_ordinary_and_fused_modes_offer_veda_without_changing_defaults(self):
        for name in ("minimax_h3", "minimax_h3_full", "minimax_h3_ref2va", "minimax_h3_ref2va_full"):
            model_def = family_handler.query_model_def(name, {})
            self.assertTrue(model_def["veda_attention"])
            self.assertTrue(model_def["veda_attention_experimental"])
            defaults = {}
            family_handler.update_default_settings(name, model_def, defaults)
            self.assertEqual(defaults["override_attention"], "")

        for name in ("minimax_h3", "minimax_h3_ref2va"):
            model_def = family_handler.query_model_def(
                name,
                {"minimax_h3_fused_turbo": True},
            )
            self.assertTrue(model_def["veda_attention"])
            self.assertTrue(model_def["veda_attention_experimental"])
            self.assertTrue(model_def["sla_attention"])
            self.assertTrue(model_def["sla_attention_default"])
            self.assertFalse(model_def["sol_attention"])
            self.assertFalse(model_def["first_block_cache"])

            defaults = {}
            family_handler.update_default_settings(name, model_def, defaults)
            self.assertEqual(defaults["override_attention"], "sla")

            saved = {"override_attention": "veda"}
            family_handler.fix_settings(name, 2.58, model_def, saved)
            self.assertEqual(saved["override_attention"], "veda")

            body = {
                "override_attention": "veda",
                "skip_steps_cache_type": "first_block",
                "num_inference_steps": 4,
                "video_length": 243,
                "sliding_window_size": 243,
            }
            from models.minimax_h3.fused_turbo import normalize_fused_h3_request
            normalize_fused_h3_request(body)
            self.assertIsNone(
                family_handler.validate_generative_settings(name, model_def, body)
            )
            self.assertEqual(body["override_attention"], "veda")
            self.assertEqual(body["skip_steps_cache_type"], "")

            blocked = {
                "override_attention": "sol",
                "skip_steps_cache_type": "first_block",
                "num_inference_steps": 4,
                "video_length": 243,
                "sliding_window_size": 243,
            }
            self.assertIsNone(
                family_handler.validate_generative_settings(name, model_def, blocked)
            )
            self.assertEqual(blocked["override_attention"], "sla")
            self.assertEqual(blocked["skip_steps_cache_type"], "")

        for name, options in (("minimax_h3", {"vdn": True}),
                              ("minimax_h3", {"minimax_h3_fused_turbo": True, "vdn": True}),
                              ("minimax_h3_voice_audio", {}), ("viggle_animate", {}),
                              ("minimax_h3_ref2va", {"minimax_h3_singularity": True}),
                              ("minimax_h3_ref2va", {"minimax_h3_dasiwa": True}),
                              ("minimax_h3", {"lock_inference_steps": True})):
            self.assertFalse(family_handler.query_model_def(name, options)["veda_attention"])

    def test_runtime_status_rejects_cpu_old_cards_and_missing_triton(self):
        for capability, installed, supported in (((0, 0), True, False), ((7, 5), True, False),
                                                  ((8, 6), False, False), ((8, 6), True, True)):
            with (mock.patch.object(attention, "_cuda_capability", return_value=capability),
                  mock.patch.object(attention, "triton_installed", installed),
                  mock.patch.object(attention, "_triton_version_tuple", return_value=(3, 3, 1))):
                status = attention.get_veda_attention_status()
                self.assertEqual(status["supported"], supported)
                if not supported:
                    self.assertTrue(status["reason"])

    def test_loader_registers_predictors_on_meta_and_loads_both_files_through_mmgp(self):
        from accelerate import init_empty_weights
        from models.minimax_h3.veda import attention as veda
        tree = ast.parse((APP / "models/minimax_h3/minimax_h3_main.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_transformer")
        metadata = {"format": veda.FORMAT, "num_layers": "50", "num_heads": "56", "head_dim": "128",
                    "plans": json.dumps({"test": {"geometry": "test", "grid": [1, 4, 4], "shapes": ["8x4x4"],
                                                   "head_shape": [[0] * 56 for _ in range(50)]}})}
        class TinyTransformer(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Module() for _ in range(50)])
                for block in self.blocks:
                    block.attn = nn.Module()
            def set_qkv_layout(self, layout):
                self.h3_qkv_layout = layout
        factory, offload = mock.Mock(side_effect=TinyTransformer), mock.Mock()
        scope = {"__package__": "models.minimax_h3", "probe_h3_checkpoint": lambda _: {
                    "adaln_curve_grid": 2, "time_embed_dim": 8, "compressed_modulation": True},
                 "_first_path": lambda value: value, "init_empty_weights": init_empty_weights,
                 "MiniMaxH3Transformer": factory, "get_linear_split_map": lambda *a, **k: {"qkv": "split"},
                 "offload": offload, "partial": partial, "_strip_transformer_wrappers": lambda *a, **k: None}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                  function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "veda_loader", "exec"), scope)
        with mock.patch.object(veda, "read_metadata", return_value=metadata):
            result = scope["_load_transformer"]("base.safetensors", torch.bfloat16,
                                                qkv_layout="grouped", veda_filename="veda.safetensors")
        self.assertEqual(offload.load_model_data.call_args.args[1], ["base.safetensors", "veda.safetensors"])
        self.assertEqual(len(list(result.parameters())), 50 * 56 * 2)
        self.assertTrue(all(param.is_meta and not param.requires_grad for param in result.parameters()))
        self.assertIn("blocks.49.attn.veda.heads.55.proj_k.weight", result.state_dict())
        self.assertTrue(all(block.attn.veda.policy is result.veda_attention for block in result.blocks))

    def test_job_selection_reloads_on_both_edges_and_rejects_unsupported_before_loading(self):
        tree = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
        generation = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_video")
        start = next(i for i, node in enumerate(generation.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "model_kwargs" for target in node.targets))
        function = ast.parse("def select():\n    reload_needed = False\n").body[0]
        function.body += generation.body[start:start + 5]
        function.body += ast.parse("return model_kwargs, reload_needed").body
        scope = {"model_def": {"architecture": "minimax_h3", "veda_attention": True},
                 "model_type": "minimax_h3", "override_attention": "veda", "attention_mode": "sage2",
                 "get_overridden_attention": lambda _: None, "minimax_h3_text_encoder": "nvfp4_awq",
                 "minimax_h3_video_vae": "auto",
                 "get_runtime_model_def": lambda *a: {"minimax_h3_video_vae_variant": "auto"},
                 "wan_model": SimpleNamespace(veda=False, text_encoder_variant="nvfp4_awq"),
                 "send_cmd": mock.Mock()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "veda_job_gate", "exec"), scope)
        with mock.patch.object(attention, "get_veda_attention_status", return_value={"supported": True}):
            self.assertEqual(scope["select"](), ({"minimax_h3_veda": True, "minimax_h3_text_encoder": "nvfp4_awq", "minimax_h3_video_vae": "auto"}, True))
            scope["wan_model"].veda = True
            self.assertFalse(scope["select"]()[1])
            scope["override_attention"] = "sla"
            self.assertTrue(scope["select"]()[1], "turning Veda off unloads its predictor")
            scope["wan_model"].veda = False
            scope["override_attention"] = ""
            scope["attention_mode"] = "veda"
            self.assertTrue(scope["select"]()[0]["minimax_h3_veda"], "resolve effective attention before loading")
        with mock.patch.object(attention, "get_veda_attention_status", return_value={"supported": False, "reason": "unsupported GPU"}):
            self.assertIs(scope["select"](), True)
        self.assertIn("unsupported GPU", scope["send_cmd"].call_args_list[-2].args[1])

    def test_head_splitting_preserves_absolute_predictor_ids_and_matches_unsplit(self):
        torch.manual_seed(90)
        module = h3.MiniMaxH3Attention(64, 8, 8, 1e-5, torch.float32).eval()
        inputs = torch.randn(1, 13, 64)
        module.veda = DenseVedaStub()
        with torch.no_grad():
            expected = module(inputs.clone(), None)
        self.assertEqual(module.veda.calls, [tuple(range(8))])
        weight = module.qkv_proj.weight.detach().clone()
        del module.qkv_proj
        for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
            proj = nn.Linear(64, 64, bias=False)
            proj.weight.data.copy_(weight[i * 64:(i + 1) * 64])
            setattr(module, name, proj)
        module.veda.calls.clear()
        owned = inputs.clone()
        owned_ref = weakref.ref(owned)
        handoff = [owned]
        del owned
        attention_kit.configure(2)
        with (torch.no_grad(), mock.patch.object(attention_kit, "MIN_SPLIT_TOKENS", 1),
              mock.patch.object(attention_kit, "sage2_staged_settings",
                                side_effect=AssertionError("Veda must bypass Sage dispatch"))):
            actual = module(handoff, None)
        self.assertEqual(handoff, [])
        self.assertIsNone(owned_ref())
        self.assertEqual(module.veda.calls, [(i,) for i in range(8)])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
