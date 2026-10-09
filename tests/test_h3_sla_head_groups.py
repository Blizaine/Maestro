"""CPU routing, parity, and lifetime checks for grouped MiniMax H3 SLA."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from types import ModuleType
import unittest
import weakref
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)


def _import_project_module(name):
    return __import__(name, fromlist=["*"])


def _dense_attention(qkv):
    query, key, value = qkv
    qkv.clear()
    result = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )
    return result.transpose(1, 2).contiguous()


@contextmanager
def _stub_sla_kernels(input_ref=None, fail_at=None):
    """Provide CPU-only SLA seams without importing Triton modules."""

    map_name = "models.minimax_h3.sla_block_map"
    kernel_name = "models.minimax_h3.sla_kernel"
    package = sys.modules.get("models.minimax_h3")
    saved_attributes = {}
    if package is not None:
        for child in ("sla_block_map", "sla_kernel"):
            saved_attributes[child] = package.__dict__.get(child, _MISSING)

    map_module = ModuleType(map_name)
    kernel_module = ModuleType(kernel_name)
    map_calls = []
    kernel_calls = []

    def get_block_map(query, key, _fraction, block_query, block_key, *, protect_upto):
        map_calls.append(
            {
                "shape": tuple(query.shape),
                "protect_upto": protect_upto,
                "input_released": input_ref is not None and input_ref() is None,
            }
        )
        blocks = (query.shape[1] + block_query - 1) // block_query
        lookup = torch.zeros(
            (query.shape[0], query.shape[2], blocks, 1),
            device=query.device,
            dtype=torch.int32,
        )
        return lookup, 1

    def block_sparse_attention(query, key, value, *_args):
        call_index = len(kernel_calls)
        kernel_calls.append(tuple(query.shape))
        if call_index == fail_at:
            raise RuntimeError("injected CPU SLA kernel failure")
        return _dense_attention([query, key, value])

    map_module.get_block_map = get_block_map
    kernel_module.block_sparse_attention = block_sparse_attention

    try:
        with mock.patch.dict(
            sys.modules,
            {map_name: map_module, kernel_name: kernel_module},
        ):
            yield map_calls, kernel_calls
    finally:
        if package is not None:
            for child, previous in saved_attributes.items():
                if previous is _MISSING:
                    package.__dict__.pop(child, None)
                else:
                    package.__dict__[child] = previous


_MISSING = object()


class TestH3SLAHeadGroups(unittest.TestCase):
    def setUp(self):
        self.h3 = _import_project_module("models.minimax_h3.transformer")
        self.attention_kit = _import_project_module("shared.attention_kit")
        self.attention = _import_project_module("shared.attention")
        self.offload = _import_project_module("mmgp.offload")
        self.attention_kit.configure(0)

    def tearDown(self):
        self.attention_kit.configure(0)

    def _fixture(self):
        torch.manual_seed(101)
        heads, head_dim = 8, 64
        inner = heads * head_dim
        sla = self.h3.MiniMaxH3SLAAttention(
            {"min_seq_len": 1, "block_size": 64, "protect_audio": True}
        )
        sla.enabled = True
        sla.sink_tokens = 3
        attention = self.h3.MiniMaxH3Attention(
            inner,
            heads,
            head_dim,
            1e-5,
            torch.float32,
            sla_attention=sla,
        ).eval()
        fused_weight = attention.qkv_proj.weight.detach().clone()
        del attention.qkv_proj
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            projection = nn.Linear(inner, inner, bias=False)
            with torch.no_grad():
                projection.weight.copy_(
                    fused_weight[index * inner : (index + 1) * inner]
                )
            setattr(attention, name, projection)
        return attention

    def _inputs(self, *, requires_grad=False):
        generator = torch.Generator().manual_seed(202)
        tokens, rotary_dim = 13, 64
        hidden = torch.randn(
            1, tokens, 512, generator=generator, requires_grad=requires_grad
        )
        positions = torch.arange(tokens, dtype=torch.float32)[:, None]
        frequencies = torch.linspace(0.015, 0.21, rotary_dim // 2)[None, :]
        phase = positions * frequencies
        cosine = torch.cat((phase.cos(), phase.cos()), dim=-1)
        sine = torch.cat((phase.sin(), phase.sin()), dim=-1)
        return hidden, (cosine, sine)

    def _reference(self, attention, hidden, rotary):
        batch, tokens, _ = hidden.shape
        shape = (batch, tokens, attention.heads, attention.head_dim)
        query, key, value = [
            projection(hidden).view(shape)
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
        ]
        query = self.h3._apply_rope(attention.q_norm(query), *rotary)
        key = self.h3._apply_rope(attention.k_norm(key), *rotary)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)
        return attention.out_proj(attended.reshape(batch, tokens, -1))

    def _configure_groups(self, level=2):
        self.attention_kit.configure(level)
        return mock.patch.object(self.attention_kit, "MIN_SPLIT_TOKENS", 1)

    def test_grouped_sla_matches_dense_and_releases_owned_input(self):
        attention = self._fixture()
        values, rotary = self._inputs()
        expected = self._reference(attention, values, rotary)
        owned = values.clone()
        owned_ref = weakref.ref(owned)
        handoff = [owned]
        del owned
        row_calls = []
        real_linear_rows = self.offload.linear_rows

        def record_rows(module, prepared, start, stop):
            row_calls.append((module, start, stop))
            return real_linear_rows(module, prepared, start, stop)

        with (
            self._configure_groups(),
            _stub_sla_kernels(owned_ref) as (map_calls, kernel_calls),
            mock.patch.object(self.offload, "linear_rows", new=record_rows),
            mock.patch.object(
                self.attention_kit,
                "sage2_staged_settings",
                side_effect=AssertionError("custom SLA dispatch must bypass Sage2"),
            ) as staged_settings,
            torch.no_grad(),
        ):
            actual = attention(handoff, rotary)

        self.assertEqual(handoff, [])
        self.assertEqual(len(map_calls), 8)
        self.assertEqual(len(kernel_calls), 8)
        self.assertTrue(all(call["shape"][2] == 1 for call in map_calls))
        self.assertEqual([call["protect_upto"] for call in map_calls], [3] * 8)
        self.assertTrue(all(not call["input_released"] for call in map_calls[:-1]))
        self.assertTrue(map_calls[-1]["input_released"])
        self.assertIsNone(owned_ref())
        self.assertEqual(len(row_calls), 24)
        for group in range(8):
            start, stop = group * 64, (group + 1) * 64
            self.assertEqual(
                row_calls[group * 3 : group * 3 + 3],
                [
                    (attention.q_proj, start, stop),
                    (attention.k_proj, start, stop),
                    (attention.v_proj, start, stop),
                ],
            )
        staged_settings.assert_not_called()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_sparse_failure_falls_back_for_failed_and_remaining_groups(self):
        for fail_at in (0, 3):
            with self.subTest(fail_at=fail_at):
                attention = self._fixture()
                values, rotary = self._inputs()
                expected = self._reference(attention, values, rotary)
                dense_calls = []

                def dense_fallback(qkv, **_kwargs):
                    dense_calls.append(tuple(qkv[0].shape))
                    return _dense_attention(qkv)

                with (
                    self._configure_groups(),
                    _stub_sla_kernels(fail_at=fail_at) as (
                        map_calls, kernel_calls
                    ),
                    mock.patch.object(
                        self.attention, "pay_attention", side_effect=dense_fallback
                    ),
                    mock.patch("builtins.print"),
                    torch.no_grad(),
                ):
                    actual = attention(values.clone(), rotary)

                self.assertEqual(len(kernel_calls), fail_at + 1)
                self.assertEqual(len(map_calls), fail_at + 1)
                self.assertEqual(len(dense_calls), 8 - fail_at)
                self.assertFalse(attention.sla_attention.enabled)
                self.assertTrue(attention.sla_attention._runtime_failed)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_off_mask_training_and_unsupported_rows_keep_unsplit_routes(self):
        routes = ("off", "unsupported", "masked", "training")
        for route in routes:
            with self.subTest(route=route):
                attention = self._fixture()
                values, rotary = self._inputs(
                    requires_grad=(route == "training")
                )
                if route == "off":
                    group_patch = self._configure_groups(0)
                else:
                    group_patch = self._configure_groups(2)
                row_patch = (
                    mock.patch.object(
                        self.offload,
                        "linear_rows_supported",
                        return_value=False,
                    )
                    if route == "unsupported"
                    else mock.patch.object(
                        self.offload,
                        "linear_rows_supported",
                        return_value=True,
                    )
                )
                mask = (
                    torch.ones(13, 13, dtype=torch.bool)
                    if route == "masked"
                    else None
                )
                grad_context = (
                    torch.enable_grad()
                    if route == "training"
                    else torch.no_grad()
                )
                with (
                    group_patch,
                    row_patch,
                    _stub_sla_kernels() as (map_calls, kernel_calls),
                    grad_context,
                ):
                    actual = attention(values, rotary, attention_mask=mask)
                    if route == "training":
                        actual.sum().backward()

                if route == "masked":
                    self.assertEqual(map_calls, [])
                    self.assertEqual(kernel_calls, [])
                else:
                    self.assertEqual(len(map_calls), 1)
                    self.assertEqual(len(kernel_calls), 1)
                    self.assertEqual(map_calls[0]["shape"][2], 8)
                self.assertEqual(tuple(actual.shape), tuple(values.shape))
                self.assertEqual(values.grad is not None, route == "training")


if __name__ == "__main__":
    unittest.main()
