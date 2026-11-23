"""Policy and CPU parity checks for the shared Extreme attention path."""

from __future__ import annotations

import os
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

def _load_attention_kit_with_offload_stub():
    """Avoid importing the GPU/offload stack for these CPU math tests."""

    fake_offload = ModuleType("mmgp.offload")
    fake_offload.linear_rows_supported = lambda _modules, _x: False
    fake_offload.prepare_linear_input = lambda _inputs, _modules: None
    fake_offload.linear_rows = lambda *_args: None
    fake_mmgp = ModuleType("mmgp")
    fake_mmgp.offload = fake_offload
    fake_mmgp.__path__ = []
    fake_attention = ModuleType("shared.attention")
    fake_attention.pay_attention = lambda *_args, **_kwargs: None
    fake_attention.sage2_staged_settings = lambda *_args, **_kwargs: None
    path = os.path.join(_APP, "shared", "attention_kit.py")
    spec = importlib.util.spec_from_file_location("_maestro_attention_kit_cpu_test", path)
    module = importlib.util.module_from_spec(spec)

    def no_compile(fn=None, **_kwargs):
        return (lambda inner: inner) if fn is None else fn

    with mock.patch.dict(
        sys.modules,
        {"mmgp": fake_mmgp, "mmgp.offload": fake_offload, "shared.attention": fake_attention},
    ):
        sys.modules[spec.name] = module
        with mock.patch.object(torch.compiler, "disable", side_effect=no_compile):
            spec.loader.exec_module(module)
    return module, fake_offload


attention_kit, offload = _load_attention_kit_with_offload_stub()


_ROPE_PATH = os.path.join(
    _APP, "models", "ltx2", "ltx_core", "model", "transformer", "rope.py"
)
_ROPE_SPEC = importlib.util.spec_from_file_location("_maestro_ltx_rope_cpu_test", _ROPE_PATH)
ltx_rope = importlib.util.module_from_spec(_ROPE_SPEC)
_ROPE_SPEC.loader.exec_module(ltx_rope)


def _reference_attention(x, projections, out_proj, norm=None):
    batch, tokens, _ = x.shape
    q, k, v = [module(x).view(batch, tokens, 2, 4) for module in projections]
    if norm is not None:
        q, k = norm(q, k)
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    return out_proj(output.flatten(2))


def _prepared_rows(input_list, _modules):
    value = input_list[0]
    input_list.clear()
    return value


def _linear_rows(module, prepared, start, stop):
    bias = None if module.bias is None else module.bias[start:stop]
    return F.linear(prepared, module.weight[start:stop], bias)


def _cpu_attention(qkv, **_kwargs):
    query, key, value = qkv
    qkv.clear()
    return F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)


def test_chunker_scope_restores_mode_and_accepts_standard_alias():
    previous = attention_kit.current_chunker()
    try:
        assert attention_kit.configure_chunker("off") == "off"
        with attention_kit.chunker_scope("standard"):
            assert attention_kit.current_chunker() == "on"
            with attention_kit.chunker_scope("extreme"):
                assert attention_kit.current_chunker() == "extreme"
            assert attention_kit.current_chunker() == "on"
        assert attention_kit.current_chunker() == "off"
        try:
            with attention_kit.chunker_scope("extreme"):
                raise RuntimeError("exercise job-scope cleanup")
        except RuntimeError:
            pass
        assert attention_kit.current_chunker() == "off"
    finally:
        attention_kit.configure_chunker(previous)


def test_extreme_bounds_wide_partial_output_even_for_one_query_head():
    assert attention_kit._extreme_query_rows(1, 1, 128, 2, 16_384) == 16_384
    assert attention_kit._extreme_query_rows(1, 1, 128, 2, 16_384, 2048) == 4096


def test_auto_memory_decisions_are_cached_only_within_a_job_scope():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        mock.patch.object(attention_kit._chunking, "auto_room", side_effect=[100, 50]) as room,
    ):
        with attention_kit.chunker_scope("auto"):
            assert not attention_kit.chunking_needed(torch.device("cuda"), 90, ("shape", 1))
            # Repeated calls for the shape reuse the first decision.
            assert not attention_kit.chunking_needed(torch.device("cuda"), 900, ("shape", 1))
        with attention_kit.chunker_scope("auto"):
            assert attention_kit.chunking_needed(torch.device("cuda"), 90, ("shape", 1))
        assert room.call_count == 2


def test_auto_room_accounts_for_memory_used_by_other_processes():
    fake_mmgp = SimpleNamespace(allocator=SimpleNamespace(active="torch"))
    gib = 1 << 30
    with (
        mock.patch.dict(sys.modules, {"mmgp": fake_mmgp}),
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "mem_get_info", return_value=(gib, 8 * gib)),
        mock.patch.object(torch.cuda, "memory_allocated", return_value=2 * gib),
        mock.patch.object(torch.cuda, "memory_reserved", return_value=2 * gib + gib // 2),
    ):
        margin = int(8 * gib * attention_kit._chunking.AUTO_MARGIN_SHARE)
        assert attention_kit._chunking.auto_room(torch.device("cuda")) == gib + gib // 2 - margin


def test_auto_upgrades_only_when_the_extreme_estimate_fits():
    x = torch.randn(1, 9, 8)
    q, k, v, output = [nn.Linear(8, 8) for _ in range(4)]
    with (
        torch.inference_mode(),
        attention_kit.chunker_scope("auto"),
        mock.patch.object(attention_kit, "EXTREME_MIN_TOKENS", 1),
        mock.patch.object(attention_kit._chunking, "EXTREME_MIN_SAVING", 128),
        mock.patch.object(attention_kit, "chunking_needed", return_value=True),
        mock.patch.object(attention_kit._chunking, "auto_room", return_value=512),
        mock.patch.object(attention_kit, "_extreme_plan_estimates", return_value=(1024, 256)),
        mock.patch.object(attention_kit, "_extreme_qkv_attention", return_value=x) as execute,
    ):
        actual = attention_kit.qkv_attention(
            [x], q, k, v, 2, 4, lambda *_: None, out_proj=output, norm_rope_rows=True,
        )
        assert actual is x
        execute.assert_called_once()


def test_extreme_query_chunks_match_full_attention_and_bound_attention_rows():
    batch, tokens, hidden, heads, head_dim = 1, 9, 8, 2, 4
    x = torch.randn(batch, tokens, hidden)
    projections = tuple(nn.Linear(hidden, heads * head_dim) for _ in range(3))
    output = nn.Linear(hidden, 7)
    expected = _reference_attention(x, projections, output)
    seen_query_rows = []
    handed_off = [x]

    def cpu_attention(qkv, **_kwargs):
        seen_query_rows.append(qkv[0].shape[1])
        return _cpu_attention(qkv)

    previous = attention_kit.current_chunker()
    try:
        attention_kit.configure_chunker("extreme")
        with (
            mock.patch.object(attention_kit, "EXTREME_MIN_TOKENS", 1),
            mock.patch.object(attention_kit, "EXTREME_QUERY_BYTES", 192),
            mock.patch.object(offload, "linear_rows_supported", return_value=True, create=True),
            mock.patch.object(offload, "prepare_linear_input", side_effect=_prepared_rows, create=True),
            mock.patch.object(offload, "linear_rows", side_effect=_linear_rows, create=True),
            mock.patch.object(attention_kit, "pay_attention", side_effect=cpu_attention),
            torch.no_grad(),
        ):
            actual = attention_kit.qkv_attention(
                handed_off,
                *projections,
                heads,
                head_dim,
                lambda *_args: None,
                out_proj=output,
                norm_rope_rows=True,
            )
    finally:
        attention_kit.configure_chunker(previous)

    assert handed_off == []
    assert len(seen_query_rows) == 6
    assert max(seen_query_rows) == 4
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_extreme_row_norms_match_full_head_statistics():
    batch, tokens, hidden, heads, head_dim = 1, 7, 8, 2, 4
    x = torch.randn(batch, tokens, hidden)
    projections = tuple(nn.Linear(hidden, heads * head_dim) for _ in range(3))
    output = nn.Linear(hidden, hidden, bias=False)
    q_norm = SimpleNamespace(weight=torch.randn(heads * head_dim), eps=1e-5)
    k_norm = SimpleNamespace(weight=torch.randn(heads * head_dim), eps=1e-5)

    def full_norm(q, k):
        q = q * (q.float().square().mean((-2, -1), keepdim=True) + q_norm.eps).rsqrt()
        k = k * (k.float().square().mean((-2, -1), keepdim=True) + k_norm.eps).rsqrt()
        weight_shape = (1, 1, heads, head_dim)
        return q_norm.weight.view(weight_shape) * q, k_norm.weight.view(weight_shape) * k

    expected = _reference_attention(x, projections, output, norm=full_norm)

    def norm_rope(query, key, group):
        for tensor, norm, head_range, index in (
            (query, q_norm, group.q, 0),
            (key, k_norm, group.kv, 1),
        ):
            if tensor is None:
                continue
            rows = tensor.flatten(2)
            if group.mean_squares is None:
                mean = rows.float().square().mean(-1, keepdim=True)
            else:
                mean = group.mean_squares[index]
            rows.mul_((mean + norm.eps).rsqrt())
            rows.mul_(norm.weight[head_range.start * head_dim:head_range.stop * head_dim])

    previous = attention_kit.current_chunker()
    try:
        attention_kit.configure_chunker("extreme")
        with (
            mock.patch.object(attention_kit, "EXTREME_MIN_TOKENS", 1),
            mock.patch.object(attention_kit, "EXTREME_QUERY_BYTES", 192),
            mock.patch.object(offload, "linear_rows_supported", return_value=True, create=True),
            mock.patch.object(offload, "prepare_linear_input", side_effect=_prepared_rows, create=True),
            mock.patch.object(offload, "linear_rows", side_effect=_linear_rows, create=True),
            mock.patch.object(attention_kit, "pay_attention", side_effect=_cpu_attention),
            torch.no_grad(),
        ):
            actual = attention_kit.qkv_attention(
                [x], *projections, heads, head_dim, norm_rope,
                norm_spans_heads=True,
                out_proj=output,
                norm_rope_rows=True,
            )
    finally:
        attention_kit.configure_chunker(previous)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_extreme_rope_query_chunks_attend_to_full_key_sequence_in_inference_mode():
    batch, tokens, hidden, heads, head_dim = 1, 9, 8, 2, 4
    x = torch.randn(batch, tokens, hidden)
    projections = tuple(nn.Linear(hidden, heads * head_dim) for _ in range(3))
    output = nn.Linear(hidden, 7)

    positions = torch.arange(tokens, dtype=torch.float32)
    frequencies = torch.tensor([0.17, 0.43])
    angles = positions[:, None] * frequencies[None, :]
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).unsqueeze(1)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).unsqueeze(1)

    q, k, v = [module(x).view(batch, tokens, heads, head_dim) for module in projections]
    ltx_rope.apply_rotary_emb_inplace(q, (cos, sin), ltx_rope.LTXRopeType.INTERLEAVED)
    ltx_rope.apply_rotary_emb_inplace(k, (cos, sin), ltx_rope.LTXRopeType.INTERLEAVED)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    expected = output(expected.flatten(2))

    seen_shapes = []

    def cpu_attention(qkv, **_kwargs):
        seen_shapes.append((qkv[0].shape[1], qkv[1].shape[1]))
        return _cpu_attention(qkv)

    def norm_rope(query, key, _group):
        for tensor, positions_slice in ((query, _group.rows), (key, _group.rows)):
            if tensor is None:
                continue
            local_cos = cos if positions_slice is None else cos[positions_slice]
            local_sin = sin if positions_slice is None else sin[positions_slice]
            ltx_rope.apply_rotary_emb_inplace(
                tensor, (local_cos, local_sin), ltx_rope.LTXRopeType.INTERLEAVED
            )

    previous = attention_kit.current_chunker()
    try:
        attention_kit.configure_chunker("extreme")
        with (
            mock.patch.object(attention_kit, "EXTREME_MIN_TOKENS", 1),
            mock.patch.object(attention_kit, "EXTREME_QUERY_BYTES", 192),
            mock.patch.object(offload, "linear_rows_supported", return_value=True, create=True),
            mock.patch.object(offload, "prepare_linear_input", side_effect=_prepared_rows, create=True),
            mock.patch.object(offload, "linear_rows", side_effect=_linear_rows, create=True),
            mock.patch.object(attention_kit, "pay_attention", side_effect=cpu_attention),
            torch.inference_mode(),
        ):
            actual = attention_kit.qkv_attention(
                [x], *projections, heads, head_dim, norm_rope,
                out_proj=output,
                norm_rope_rows=True,
            )
    finally:
        attention_kit.configure_chunker(previous)

    assert seen_shapes
    assert max(query_rows for query_rows, _key_rows in seen_shapes) == 4
    assert all(key_rows == tokens and query_rows < key_rows for query_rows, key_rows in seen_shapes)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
