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
        norm_chunk_bytes = []
        real_linear_rows = self.offload.linear_rows
        real_rms_norm = self.h3._rms_norm_inplace

        def record_rows(module, prepared, start, stop):
            row_calls.append((module, start, stop))
            return real_linear_rows(module, prepared, start, stop)

        def record_rms_norm(norm, tensor, chunk_bytes=256 << 20):
            norm_chunk_bytes.append(chunk_bytes)
            return real_rms_norm(norm, tensor, chunk_bytes)

        with (
            self._configure_groups(),
            _stub_sla_kernels(owned_ref) as (map_calls, kernel_calls),
            mock.patch.object(self.offload, "linear_rows", new=record_rows),
            mock.patch.object(self.h3, "_rms_norm_inplace", new=record_rms_norm),
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
        self.assertEqual(norm_chunk_bytes, [64 << 20] * 16)
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

    def test_ordinary_grouped_and_unsplit_norm_chunk_budgets(self):
        for split_level, expected_chunk in ((2, 64 << 20), (0, 256 << 20)):
            with self.subTest(split_level=split_level):
                attention = self._fixture()
                attention.sla_attention.enabled = False
                values, rotary = self._inputs()
                expected = self._reference(attention, values, rotary)
                norm_chunk_bytes = []
                output_projection_calls = []
                real_norm = self.h3._rms_norm_inplace
                real_output_projection = (
                    self.h3._project_attention_output_reusing_storage
                )

                def record_norm(norm, tensor, chunk_bytes=256 << 20):
                    norm_chunk_bytes.append(chunk_bytes)
                    return real_norm(norm, tensor, chunk_bytes)

                def record_output_projection(projection, attended, batch, length):
                    output_projection_calls.append((batch, length))
                    return real_output_projection(
                        projection, attended, batch, length
                    )

                with (
                    self._configure_groups(split_level),
                    _stub_sla_kernels() as (map_calls, kernel_calls),
                    mock.patch.object(
                        self.attention, "pay_attention", side_effect=_dense_attention
                    ),
                    mock.patch.object(
                        self.attention_kit,
                        "sage2_staged_settings",
                        return_value=None,
                    ),
                    mock.patch.object(self.h3, "_rms_norm_inplace", new=record_norm),
                    mock.patch.object(
                        self.h3,
                        "_project_attention_output_reusing_storage",
                        new=record_output_projection,
                    ),
                    torch.no_grad(),
                ):
                    actual = attention(values.clone(), rotary)

                expected_calls = 16 if split_level else 2
                self.assertEqual(
                    norm_chunk_bytes,
                    [expected_chunk] * expected_calls,
                )
                self.assertEqual(map_calls, [])
                self.assertEqual(kernel_calls, [])
                self.assertEqual(
                    output_projection_calls,
                    [(1, 13)] if split_level else [],
                )
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_inplace_norm_matches_native_across_dtype_layout_and_tail(self):
        tolerances = {
            torch.float32: (1e-6, 1e-6),
            torch.bfloat16: (1e-2, 1e-2),
            torch.float16: (2e-3, 2e-3),
        }
        for dtype, (rtol, atol) in tolerances.items():
            for layout in ("contiguous", "strided"):
                with self.subTest(dtype=dtype, layout=layout):
                    generator = torch.Generator().manual_seed(303)
                    if layout == "contiguous":
                        source = torch.randn(
                            (1, 10, 4, 8), generator=generator, dtype=dtype
                        )
                    else:
                        source = torch.randn(
                            (1, 4, 10, 8), generator=generator, dtype=dtype
                        ).permute(0, 2, 1, 3)
                    norm = nn.RMSNorm(8, eps=1e-5, dtype=dtype)
                    actual = source.clone(memory_format=torch.preserve_format)
                    original = actual.clone(memory_format=torch.preserve_format)
                    row_bytes = actual[0, 0].numel() * actual.element_size()
                    chunk_bytes = 3 * row_bytes
                    chunk_rows = []
                    hook = norm.register_forward_pre_hook(
                        lambda _module, args: chunk_rows.append(args[0].shape[1])
                    )
                    try:
                        with torch.inference_mode():
                            expected = norm(source)
                            chunk_rows.clear()
                            returned = self.h3._rms_norm_inplace(
                                norm, actual, chunk_bytes
                            )
                    finally:
                        hook.remove()

                    self.assertIs(returned, actual)
                    self.assertEqual(chunk_rows, [3, 3, 3, 1])
                    self.assertEqual(actual.stride(), original.stride())
                    self.assertFalse(torch.equal(actual, original))
                    torch.testing.assert_close(
                        actual, expected, rtol=rtol, atol=atol
                    )

    def test_grouped_output_projection_reuses_owned_buffer_with_tail_chunk(self):
        torch.manual_seed(404)
        batch, length, heads, head_dim = 2, 7, 2, 4
        source = torch.randn(batch, length, heads, head_dim)
        projection = nn.Linear(heads * head_dim, 5, bias=True).eval()
        expected = F.linear(
            source.reshape(batch, length, -1), projection.weight, projection.bias
        )
        calls = []
        real_forward = projection.forward

        def record_forward(rows):
            calls.append(tuple(rows.shape))
            return real_forward(rows)

        with (
            mock.patch.object(
                self.h3, "MINIMAX_H3_LARGE_SEQUENCE_TOKENS", 1
            ),
            mock.patch.object(
                self.h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 4
            ),
            mock.patch.object(projection, "forward", new=record_forward),
            torch.inference_mode(),
        ):
            actual = self.h3._project_attention_output_reusing_storage(
                projection, source, batch, length
            )

        self.assertEqual(calls, [(4, 8), (4, 8), (4, 8), (2, 8)])
        self.assertEqual(tuple(actual.shape), (batch, length, 5))
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
        self.assertEqual(
            actual.untyped_storage().nbytes(),
            source.numel() * source.element_size(),
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_grouped_sla_and_ordinary_routes_use_bounded_output_projection(self):
        for route in ("sla", "ordinary"):
            with self.subTest(route=route):
                attention = self._fixture()
                attention.out_proj = nn.Linear(512, 384, bias=True).eval()
                if route == "ordinary":
                    attention.sla_attention.enabled = False
                values, rotary = self._inputs()
                expected = self._reference(attention, values, rotary)
                projection_storage = []
                real_bounded_projection = (
                    self.h3._project_attention_output_reusing_storage
                )

                def record_bounded_projection(projection, attended, batch, length):
                    source_ptr = attended.untyped_storage().data_ptr()
                    result = real_bounded_projection(
                        projection, attended, batch, length
                    )
                    projection_storage.append(
                        (
                            source_ptr,
                            result.untyped_storage().data_ptr(),
                            tuple(result.shape),
                            result.is_contiguous(),
                        )
                    )
                    return result

                context = (
                    _stub_sla_kernels()
                    if route == "sla"
                    else mock.patch.object(
                        self.attention, "pay_attention", new=_dense_attention
                    )
                )
                with (
                    self._configure_groups(),
                    mock.patch.object(
                        self.h3, "MINIMAX_H3_LARGE_SEQUENCE_TOKENS", 1
                    ),
                    mock.patch.object(
                        self.h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 4
                    ),
                    mock.patch.object(
                        self.h3,
                        "_project_attention_output_reusing_storage",
                        new=record_bounded_projection,
                    ),
                    context,
                    mock.patch.object(
                        self.attention_kit,
                        "sage2_staged_settings",
                        return_value=None,
                    ),
                    torch.no_grad(),
                ):
                    actual = attention(values.clone(), rotary)

                self.assertEqual(len(projection_storage), 1)
                source_ptr, output_ptr, shape, contiguous = projection_storage[0]
                self.assertEqual(source_ptr, output_ptr)
                self.assertEqual(shape, (1, 13, 384))
                self.assertTrue(contiguous)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_output_projection_fallbacks_preserve_input_and_autocast_dtype(self):
        torch.manual_seed(505)

        class NonlinearProjection(nn.Module):
            in_features = 8
            out_features = 8

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def forward(self, value):
                return self.linear(value).relu()

        fallback_cases = (
            ("short", torch.randn(1, 2, 2, 4), nn.Linear(8, 5)),
            (
                "noncontiguous",
                torch.randn(1, 4, 7, 2).permute(0, 2, 1, 3),
                nn.Linear(8, 5),
            ),
            ("expansion", torch.randn(1, 7, 2, 4), nn.Linear(8, 9)),
            ("nonlinear", torch.randn(1, 7, 2, 4), NonlinearProjection()),
            ("three_dimensional", torch.randn(1, 7, 8), nn.Linear(8, 5)),
        )
        for name, source, projection in fallback_cases:
            with self.subTest(route=name):
                projection.eval()
                original = source.clone(memory_format=torch.preserve_format)
                expected = projection(source.reshape(1, source.shape[1], -1))
                with (
                    mock.patch.object(
                        self.h3,
                        "MINIMAX_H3_LARGE_SEQUENCE_TOKENS",
                        50_000 if name == "short" else 1,
                    ),
                    mock.patch.object(
                        self.h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 4
                    ),
                    torch.inference_mode(),
                ):
                    actual = self.h3._project_attention_output_reusing_storage(
                        projection, source, 1, source.shape[1]
                    )
                self.assertNotEqual(
                    actual.untyped_storage().data_ptr(),
                    source.untyped_storage().data_ptr(),
                )
                self.assertTrue(torch.equal(source, original))
                torch.testing.assert_close(actual, expected)

        source = torch.randn(1, 7, 2, 4)
        projection = nn.Linear(8, 5).eval()
        with (
            mock.patch.object(
                self.h3, "MINIMAX_H3_LARGE_SEQUENCE_TOKENS", 1
            ),
            mock.patch.object(
                self.h3, "MINIMAX_H3_ACTIVATION_CHUNK_TOKENS", 4
            ),
            torch.autocast("cpu", dtype=torch.bfloat16),
            torch.inference_mode(),
        ):
            expected = projection(source.reshape(1, 7, 8))
            actual = self.h3._project_attention_output_reusing_storage(
                projection, source, 1, 7
            )
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(expected.dtype, torch.bfloat16)
        self.assertNotEqual(
            actual.untyped_storage().data_ptr(), source.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(actual, expected)

        train_source = torch.randn(1, 7, 2, 4, requires_grad=True)
        train_projection = nn.Linear(8, 5).eval()
        with torch.enable_grad():
            train_result = self.h3._project_attention_output_reusing_storage(
                train_projection, train_source, 1, 7
            )
            train_result.sum().backward()
        self.assertIsNotNone(train_source.grad)

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
