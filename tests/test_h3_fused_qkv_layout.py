"""CPU checks for grouped and head-interleaved fused H3 QKV rows."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TestH3FusedQKVLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import models.minimax_h3.transformer as transformer

        cls.torch = torch
        cls.transformer = transformer

    def _attention_pair(self):
        torch = self.torch
        module = self.transformer
        torch.manual_seed(17)
        grouped = module.MiniMaxH3Attention(
            hidden_size=8,
            heads=2,
            head_dim=4,
            eps=1e-5,
            dtype=torch.float32,
        ).eval()
        interleaved = copy.deepcopy(grouped)
        inner = grouped.heads * grouped.head_dim
        query, key, value = grouped.qkv_proj.weight.detach().split(inner, dim=0)
        interleaved_weight = torch.stack(
            (
                query.reshape(grouped.heads, grouped.head_dim, -1),
                key.reshape(grouped.heads, grouped.head_dim, -1),
                value.reshape(grouped.heads, grouped.head_dim, -1),
            ),
            dim=1,
        ).reshape_as(interleaved.qkv_proj.weight)
        with torch.no_grad():
            interleaved.qkv_proj.weight.copy_(interleaved_weight)
        grouped.set_qkv_layout("grouped")
        interleaved.set_qkv_layout("head_interleaved")
        return grouped, interleaved

    def _tiny_transformer(self, layout):
        torch = self.torch
        transformer = self.transformer.MiniMaxH3Transformer(
            hidden_size=8,
            num_layers=1,
            token_refiner_layers=1,
            num_attention_heads=2,
            attention_head_dim=4,
            ffn_dim=12,
            video_channels=1,
            audio_channels=2,
            patch_size=(1, 1, 1),
            text_dim=6,
            curve_grid=None,
            curve_dim=4,
            timestep_input_dim=4,
            time_embed_hidden_size=8,
            rope_freq_dim=1,
            dtype=torch.float32,
        )
        transformer.set_qkv_layout(layout)
        return transformer

    def _assert_independent_lora_delta(self, transformer, converted, factors, hidden):
        torch = self.torch
        module = self.transformer
        attention = transformer.blocks[0].attn
        down = converted["blocks.0.attn.qkv_proj.lora_A.weight"]
        up = converted["blocks.0.attn.qkv_proj.lora_B.weight"]
        fused_delta = torch.nn.functional.linear(
            torch.nn.functional.linear(hidden, down),
            up,
        )
        actual = module._split_fused_qkv(
            fused_delta,
            attention.heads,
            attention.head_dim,
            attention.qkv_layout,
        )
        expected = tuple(
            torch.nn.functional.linear(
                torch.nn.functional.linear(hidden, factors[kind][0]),
                factors[kind][1],
            ).reshape(
                hidden.shape[0],
                hidden.shape[1],
                attention.heads,
                attention.head_dim,
            )
            for kind in ("q", "k", "v")
        )
        for actual_projection, expected_projection in zip(actual, expected):
            torch.testing.assert_close(
                actual_projection,
                expected_projection,
                atol=1e-6,
                rtol=1e-6,
            )

    def _stub_adaln_conversion(self):
        affine = types.ModuleType("models.minimax_h3.lora_affine")

        def no_conversion(model_type, state_dict, target_table):
            target_width = 2688 if target_table is None else int(target_table.shape[1])
            return 0, "fl2va", target_width, target_width

        affine.convert_adaln_loras = no_conversion
        return mock.patch.dict(
            sys.modules,
            {"models.minimax_h3.lora_affine": affine},
        )

    def _assert_equivalent(
        self,
        grouped,
        interleaved,
        hidden,
        *,
        chunk_size=None,
        rotary=None,
    ):
        torch = self.torch
        if chunk_size is None:
            context = mock.patch.object(
                self.transformer,
                "_activation_chunk_tokens",
                side_effect=lambda length, *_: length,
            )
        else:
            context = mock.patch.object(
                self.transformer,
                "_activation_chunk_tokens",
                side_effect=lambda length, *_: chunk_size,
            )
        with context, torch.inference_mode():
            expected = grouped(hidden, rotary)
            actual = interleaved(hidden, rotary)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_fused_splitter_maps_independent_head_qkv_rows(self):
        torch = self.torch
        module = self.transformer
        heads, head_dim = 2, 3
        logical = torch.arange(2 * 4 * heads * 3 * head_dim).reshape(
            2, 4, heads, 3, head_dim
        )
        packed = logical.reshape(2, 4, -1)
        query, key, value = module._split_fused_qkv(
            packed,
            heads,
            head_dim,
            "interleaved",
        )
        self.assertTrue(torch.equal(query, logical[..., 0, :]))
        self.assertTrue(torch.equal(key, logical[..., 1, :]))
        self.assertTrue(torch.equal(value, logical[..., 2, :]))
        self.assertTrue(query.is_contiguous())
        self.assertTrue(key.is_contiguous())
        self.assertTrue(value.is_contiguous())

    def test_interleaved_fused_projection_matches_grouped_full_and_chunked(self):
        torch = self.torch
        grouped, interleaved = self._attention_pair()
        hidden = torch.randn(2, 7, 8)
        self._assert_equivalent(grouped, interleaved, hidden)
        self._assert_equivalent(grouped, interleaved, hidden, chunk_size=2)

    def test_interleaved_fused_projection_matches_grouped_fused_rope_path(self):
        torch = self.torch
        module = self.transformer
        grouped, interleaved = self._attention_pair()
        hidden = torch.randn(1, 5, 8)
        cosine = torch.ones(5, 4)
        sine = torch.zeros(5, 4)
        rotary = (cosine, sine)

        def reference_rms_rope(query, key, rotary, start, stop, q_norm, k_norm):
            cos, sin = rotary
            return (
                module._apply_rope(q_norm(query), cos[start:stop], sin[start:stop]),
                module._apply_rope(k_norm(key), cos[start:stop], sin[start:stop]),
            )

        with (
            mock.patch.object(module.denoiser_kernels, "can_rms_rope", return_value=True),
            mock.patch.object(
                module.denoiser_kernels,
                "rms_rope",
                side_effect=reference_rms_rope,
            ),
        ):
            self._assert_equivalent(
                grouped,
                interleaved,
                hidden,
                chunk_size=2,
                rotary=rotary,
            )

    def test_interleaved_fused_projection_matches_grouped_sol_and_sla_paths(self):
        torch = self.torch
        module = self.transformer

        class ReferenceFastAttention:
            def use_for_layer(self, length, attention_mask):
                return True

            def __call__(self, qkv, enabled):
                self.asserted_enabled = enabled
                return module._run_h3_attention(*qkv, attention_mask=None)

        hidden = torch.randn(1, 5, 8)
        for policy_name in ("sol_attention", "sla_attention"):
            with self.subTest(policy=policy_name):
                grouped, interleaved = self._attention_pair()
                reference = ReferenceFastAttention()
                setattr(grouped, policy_name, reference)
                setattr(interleaved, policy_name, ReferenceFastAttention())
                self._assert_equivalent(
                    grouped,
                    interleaved,
                    hidden,
                    chunk_size=2,
                )
                self.assertTrue(reference.asserted_enabled)

    def test_interleaved_fused_projection_matches_grouped_vdn_path(self):
        torch = self.torch

        class ReferenceVDN(torch.nn.Module):
            def forward(self, hidden_chunks, raw_qkv, normalized_qkv, out_proj):
                value = normalized_qkv[2]
                return out_proj(value.reshape(value.shape[0], value.shape[1], -1))[0]

        grouped, interleaved = self._attention_pair()
        grouped.vdn = ReferenceVDN()
        interleaved.vdn = ReferenceVDN()
        hidden = torch.randn(1, 5, 8)
        with torch.inference_mode():
            expected = grouped(hidden)
            actual = interleaved(hidden)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_transformer_setter_configures_main_and_token_refiner_attention(self):
        torch = self.torch
        module = self.transformer
        transformer = module.MiniMaxH3Transformer(
            hidden_size=8,
            num_layers=2,
            token_refiner_layers=1,
            num_attention_heads=2,
            attention_head_dim=4,
            ffn_dim=12,
            video_channels=1,
            audio_channels=2,
            patch_size=(1, 1, 1),
            text_dim=6,
            curve_grid=None,
            curve_dim=4,
            timestep_input_dim=4,
            time_embed_hidden_size=8,
            rope_freq_dim=1,
            dtype=torch.float32,
        )
        transformer.set_qkv_layout("interleaved")
        attention_modules = [
            item
            for item in transformer.modules()
            if isinstance(item, module.MiniMaxH3Attention)
        ]
        self.assertEqual(len(attention_modules), 3)
        self.assertEqual({attention.qkv_layout for attention in attention_modules}, {"interleaved"})
        self.assertEqual(transformer.h3_qkv_layout, "interleaved")

    def test_layout_setting_does_not_change_split_projection_path(self):
        torch = self.torch
        grouped, _ = self._attention_pair()
        split = copy.deepcopy(grouped)
        inner = split.heads * split.head_dim
        for name, weight in zip(
            ("q_proj", "k_proj", "v_proj"),
            split.qkv_proj.weight.detach().split(inner, dim=0),
        ):
            projection = torch.nn.Linear(8, inner, bias=False)
            with torch.no_grad():
                projection.weight.copy_(weight)
            setattr(split, name, projection)
        hidden = torch.randn(1, 5, 8)
        with torch.inference_mode():
            expected = grouped(hidden)
            split.set_qkv_layout("interleaved")
            actual = split(hidden)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_diffusers_independent_qkv_loras_match_both_fused_layouts(self):
        torch = self.torch
        module = self.transformer
        torch.manual_seed(31)
        factors = {
            kind: (torch.randn(2, 8), torch.randn(8, 2))
            for kind in ("q", "k", "v")
        }
        source = {}
        for kind, (down, up) in factors.items():
            prefix = f"transformer_blocks.0.attn.to_{kind}"
            source[f"{prefix}.lora_A.weight"] = down
            source[f"{prefix}.lora_B.weight"] = up
        hidden = torch.randn(1, 5, 8)

        for layout in ("grouped", "interleaved"):
            with self.subTest(layout=layout):
                transformer = self._tiny_transformer(layout)
                from models.minimax_h3.lora_vdn import normalize_diffusers_lora

                converted = normalize_diffusers_lora(source, transformer)
                self._assert_independent_lora_delta(
                    transformer,
                    converted,
                    factors,
                    hidden,
                )

    def test_pdd_independent_qkv_loras_match_both_fused_layouts(self):
        torch = self.torch
        torch.manual_seed(43)
        factors = {
            kind: (torch.randn(2, 8), torch.randn(8, 2))
            for kind in ("q", "k", "v")
        }
        source = {
            "proj_out.weight": torch.zeros(32, 2, 8),
            "audio_proj_out.weight": torch.zeros(32, 2, 8),
        }
        for kind, (down, up) in factors.items():
            prefix = f"transformer_blocks.0.attn.to_{kind}"
            source[f"{prefix}.lora_down"] = down
            source[f"{prefix}.lora_up"] = up
        hidden = torch.randn(1, 5, 8)

        for layout in ("grouped", "interleaved"):
            with self.subTest(layout=layout):
                transformer = self._tiny_transformer(layout)
                with self._stub_adaln_conversion():
                    converted = transformer.preprocess_loras("minimax_h3", source)
                self._assert_independent_lora_delta(
                    transformer,
                    converted,
                    factors,
                    hidden,
                )

    def test_already_fused_lora_rows_remain_unchanged_when_source_is_unknown(self):
        torch = self.torch
        transformer = self._tiny_transformer("interleaved")
        down = torch.randn(6, 8)
        up = torch.randn(24, 6)
        source = {
            "blocks.0.attn.qkv_proj.lora_A.weight": down,
            "blocks.0.attn.qkv_proj.lora_B.weight": up,
        }

        with self._stub_adaln_conversion():
            converted = transformer.preprocess_loras("minimax_h3", source)

        self.assertIs(converted["blocks.0.attn.qkv_proj.lora_A.weight"], down)
        self.assertIs(converted["blocks.0.attn.qkv_proj.lora_B.weight"], up)


if __name__ == "__main__":
    unittest.main()
