"""CPU parity and fallback checks for shared projection lifetime helpers."""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import unittest
import weakref
from types import ModuleType, SimpleNamespace
from unittest import mock


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)


def _import_project_module(module_name):
    """Import against the real project runtime; tests patch offload per call."""

    return __import__(module_name, fromlist=["*"])


_OFFLOAD = _import_project_module("mmgp.offload")
_ATTENTION = _import_project_module("shared.attention")
_ATTENTION_KIT = _import_project_module("shared.attention_kit")

import torch
import torch.nn.functional as F
from torch import nn


def _cpu_attention(qkv_list, **_kwargs):
    query, key, value = qkv_list
    qkv_list.clear()
    return F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)


_MISSING_PACKAGE_ATTRIBUTE = object()


def _import_saganaki_fwd_for_pointer_test():
    """Import Sol dispatch without retaining temporary compatibility modules."""

    package_attributes = []
    for module_name, attributes in (
        ("shared", ("sol_attn",)),
        ("shared.sol_attn", ("saganaki",)),
        ("shared.sol_attn.saganaki", ("fwd", "preprocess", "quant")),
    ):
        package = sys.modules.get(module_name)
        if package is not None:
            package_attributes.extend(
                (package, attribute, package.__dict__.get(attribute, _MISSING_PACKAGE_ATTRIBUTE))
                for attribute in attributes
            )

    try:
        # patch.dict restores sys.modules, while the snapshots below also restore
        # child-module attributes that importlib adds to existing package objects.
        with mock.patch.dict(sys.modules, {}):
            try:
                from shared.sol_attn.saganaki import fwd
            except ModuleNotFoundError as error:
                if error.name != "triton.tools.tensor_descriptor":
                    raise

                descriptor_module = ModuleType("triton.tools.tensor_descriptor")

                class _UnsupportedTensorDescriptor:
                    def __new__(cls, *_args, **_kwargs):
                        raise AssertionError(
                            "The pointer-dispatch test must not instantiate a TMA descriptor."
                        )

                    @classmethod
                    def from_tensor(cls, *_args, **_kwargs):
                        raise AssertionError(
                            "The pointer-dispatch test must not use the TMA descriptor path."
                        )

                descriptor_module.TensorDescriptor = _UnsupportedTensorDescriptor
                import triton

                autotune = triton.autotune

                def _autotune_without_result_cache(*args, **kwargs):
                    # Triton 3.3 lacks this decorator keyword; dispatch does not use it.
                    kwargs.pop("cache_results", None)
                    return autotune(*args, **kwargs)

                sys.modules["triton.tools.tensor_descriptor"] = descriptor_module
                with mock.patch.object(
                    triton, "autotune", _autotune_without_result_cache
                ):
                    from shared.sol_attn.saganaki import fwd
    finally:
        for package, attribute, previous_value in package_attributes:
            if previous_value is _MISSING_PACKAGE_ATTRIBUTE:
                package.__dict__.pop(attribute, None)
            else:
                package.__dict__[attribute] = previous_value

    return fwd


class _CustomInt8LoraProjection(nn.Module):
    """Small stand-in for a quantized projection that applies its own adapter."""

    def __init__(self, in_features, out_features, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.qweight = torch.randint(
            -8, 8, (out_features, in_features), generator=generator
        ).to(torch.int8)
        self.scale = 0.015625
        self.adapter_down = nn.Linear(in_features, 3, bias=False)
        self.adapter_up = nn.Linear(3, out_features, bias=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        base = F.linear(x, self.qweight.float() * self.scale)
        return base + self.adapter_up(self.adapter_down(x))


class TestAttentionKit(unittest.TestCase):
    def setUp(self):
        self._attention_offload_patch = mock.patch.object(
            _ATTENTION, "offload", _OFFLOAD
        )
        self._attention_kit_offload_patch = mock.patch.object(
            _ATTENTION_KIT, "offload", _OFFLOAD
        )
        self._attention_offload_patch.start()
        self._attention_kit_offload_patch.start()
        self.previous_level = _ATTENTION_KIT.head_split
        self.previous_state = dict(_OFFLOAD.shared_state)
        _OFFLOAD.shared_state.clear()
        _OFFLOAD.shared_state["_attention"] = "sdpa"
        _ATTENTION_KIT._cached_head_groups.cache_clear()

    def tearDown(self):
        _ATTENTION_KIT.head_split = self.previous_level
        _OFFLOAD.shared_state.clear()
        _OFFLOAD.shared_state.update(self.previous_state)
        _ATTENTION_KIT._cached_head_groups.cache_clear()
        self._attention_kit_offload_patch.stop()
        self._attention_offload_patch.stop()

    def test_head_groups_are_cached_and_use_head_divisors(self):
        with mock.patch.object(_ATTENTION_KIT, "MIN_SPLIT_TOKENS", 1):
            self.assertEqual(_ATTENTION_KIT.head_groups(56, 16, level=2), 8)
            first = _ATTENTION_KIT._cached_head_groups.cache_info()
            self.assertEqual(_ATTENTION_KIT.head_groups(56, 16, level=2), 8)
            second = _ATTENTION_KIT._cached_head_groups.cache_info()
            self.assertEqual(second.hits, first.hits + 1)
            self.assertEqual(_ATTENTION_KIT.head_groups(56, 16, level=3), 14)
            self.assertEqual(_ATTENTION_KIT.head_groups(56, 0, level=3), 1)

    def test_split_qkv_matches_unsplit_cpu_attention_and_releases_prepared_input(self):
        batch, tokens, hidden, heads, head_dim = 1, 12, 16, 8, 4
        torch.manual_seed(7)
        x = torch.randn(batch, tokens, hidden)
        projections = tuple(nn.Linear(hidden, heads * head_dim) for _ in range(3))
        query, key, value = [
            projection(x).view(batch, tokens, heads, head_dim)
            for projection in projections
        ]
        expected = _cpu_attention([query, key, value])
        owned_input = [x]
        prepared_refs = []
        row_calls = []

        def prepare_linear_input(input_list, _projections):
            prepared = input_list[0].clone()
            prepared_refs.append(weakref.ref(prepared))
            input_list.clear()
            return prepared

        def linear_rows(module, prepared, start, stop):
            row_calls.append((module, start, stop))
            bias = None if module.bias is None else module.bias[start:stop]
            return F.linear(prepared, module.weight[start:stop], bias)

        def norm_rope(_query, _key, _group):
            return None

        _ATTENTION_KIT.configure(2)
        with (
            mock.patch.object(_ATTENTION_KIT, "MIN_SPLIT_TOKENS", 1),
            mock.patch.object(_ATTENTION_KIT, "sage2_staged_settings", return_value=None),
            mock.patch.object(_ATTENTION_KIT, "pay_attention", side_effect=_cpu_attention),
            mock.patch.object(_OFFLOAD, "linear_rows_supported", return_value=True, create=True),
            mock.patch.object(_OFFLOAD, "prepare_linear_input", side_effect=prepare_linear_input, create=True) as prepare,
            mock.patch.object(_OFFLOAD, "linear_rows", side_effect=linear_rows, create=True),
            torch.no_grad(),
        ):
            actual = _ATTENTION_KIT.qkv_attention(
                owned_input,
                *projections,
                heads,
                head_dim,
                norm_rope,
            )

        gc.collect()
        self.assertEqual(owned_input, [])
        prepare.assert_called_once()
        self.assertEqual(len(row_calls), heads * 3)
        self.assertIsNone(prepared_refs[0]())
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_custom_int8_lora_modules_use_their_forward_fallback(self):
        batch, tokens, hidden, heads, head_dim = 1, 9, 12, 4, 3
        torch.manual_seed(19)
        x = torch.randn(batch, tokens, hidden)
        projections = tuple(
            _CustomInt8LoraProjection(hidden, heads * head_dim, seed)
            for seed in (2, 3, 5)
        )
        query, key, value = [
            projection(x).view(batch, tokens, heads, head_dim)
            for projection in projections
        ]
        expected = _cpu_attention([query, key, value])
        owned_input = [x]
        _ATTENTION_KIT.configure(3)

        with (
            mock.patch.object(_ATTENTION_KIT, "MIN_SPLIT_TOKENS", 1),
            mock.patch.object(_ATTENTION_KIT, "sage2_staged_settings", return_value=None),
            mock.patch.object(_ATTENTION_KIT, "pay_attention", side_effect=_cpu_attention),
            mock.patch.object(_OFFLOAD, "linear_rows_supported", return_value=False, create=True),
            mock.patch.object(_OFFLOAD, "prepare_linear_input", create=True) as prepare,
            mock.patch.object(_OFFLOAD, "linear_rows", create=True) as linear_rows,
            mock.patch("builtins.print"),
            torch.no_grad(),
        ):
            actual = _ATTENTION_KIT.qkv_attention(
                owned_input,
                *projections,
                heads,
                head_dim,
                lambda *_args: None,
            )

        self.assertEqual(owned_input, [])
        prepare.assert_not_called()
        linear_rows.assert_not_called()
        self.assertEqual([projection.calls for projection in projections], [2, 2, 2])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_grouped_full_head_norm_matches_unsplit_attention(self):
        batch, tokens, hidden, heads, head_dim = 1, 10, 16, 8, 4
        torch.manual_seed(21)
        x = torch.randn(batch, tokens, hidden)
        projections = tuple(nn.Linear(hidden, heads * head_dim) for _ in range(3))
        query, key, value = [
            projection(x).view(batch, tokens, heads, head_dim)
            for projection in projections
        ]
        q_weight = torch.randn(heads * head_dim)
        k_weight = torch.randn(heads * head_dim)
        epsilon = 1e-5

        def full_norm(tensor, weight):
            rows = tensor.flatten(2)
            scale = (rows.float().square().mean(-1, keepdim=True) + epsilon).rsqrt()
            rows = rows * scale
            return rows.mul(weight).view_as(tensor)

        expected = _cpu_attention(
            [full_norm(query, q_weight), full_norm(key, k_weight), value]
        )

        def norm_rope(group_query, group_key, group):
            for tensor, weight, heads_range, index in (
                (group_query, q_weight, group.q, 0),
                (group_key, k_weight, group.kv, 1),
            ):
                if tensor is None:
                    continue
                rows = tensor.flatten(2)
                if group.mean_squares is None:
                    mean_squares = rows.float().square().mean(-1, keepdim=True)
                else:
                    mean_squares = group.mean_squares[index]
                rows.mul_((mean_squares + epsilon).rsqrt())
                rows.mul_(
                    weight[
                        heads_range.start * head_dim:
                        heads_range.stop * head_dim
                    ]
                )

        def prepare_linear_input(input_list, _projections):
            prepared = input_list[0]
            input_list.clear()
            return prepared

        def linear_rows(module, prepared, start, stop):
            bias = None if module.bias is None else module.bias[start:stop]
            return F.linear(prepared, module.weight[start:stop], bias)

        _ATTENTION_KIT.configure(2)
        with (
            mock.patch.object(_ATTENTION_KIT, "MIN_SPLIT_TOKENS", 1),
            mock.patch.object(_ATTENTION_KIT, "sage2_staged_settings", return_value=None),
            mock.patch.object(_ATTENTION_KIT, "pay_attention", side_effect=_cpu_attention),
            mock.patch.object(_OFFLOAD, "linear_rows_supported", return_value=True, create=True),
            mock.patch.object(_OFFLOAD, "prepare_linear_input", side_effect=prepare_linear_input, create=True),
            mock.patch.object(_OFFLOAD, "linear_rows", side_effect=linear_rows, create=True),
            torch.no_grad(),
        ):
            actual = _ATTENTION_KIT.qkv_attention(
                [x],
                *projections,
                heads,
                head_dim,
                norm_rope,
                norm_spans_heads=True,
            )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

    def test_attention_mask_keeps_the_shared_sdpa_fallback(self):
        torch.manual_seed(23)
        query, key, value = [torch.randn(1, 6, 2, 4) for _ in range(3)]
        mask = torch.ones(1, 6, 1, 6, dtype=torch.bool)
        mask[..., 0] = False
        qkv_list = [query, key, value]
        with mock.patch.object(_ATTENTION, "sageattn2_wrapper") as sage2:
            actual = _ATTENTION.pay_attention(qkv_list, attention_mask=mask)
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask.transpose(1, 2),
        ).transpose(1, 2)
        sage2.assert_not_called()
        self.assertEqual(qkv_list, [])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_missing_runtime_attention_setting_defaults_to_sdpa(self):
        query, key, value = [torch.randn(1, 5, 2, 4) for _ in range(3)]
        qkv_list = [query, key, value]
        with mock.patch.dict(_OFFLOAD.shared_state, {}, clear=True):
            actual = _ATTENTION.pay_attention(qkv_list)
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_staged_settings_are_rejected_for_forced_dense_modes(self):
        with (
            mock.patch.object(_ATTENTION, "sageattn2", object()),
            mock.patch.object(_ATTENTION, "_sage2_staged_settings", return_value={"sentinel": True}),
            mock.patch.object(_ATTENTION, "get_default_attention_mode", return_value="sdpa"),
        ):
            self.assertIsNone(
                _ATTENTION.sage2_staged_settings(torch.device("cpu"), "sdpa")
            )
            self.assertIsNone(
                _ATTENTION.sage2_staged_settings(torch.device("cpu"), "sol")
            )
            self.assertEqual(
                _ATTENTION.sage2_staged_settings(torch.device("cpu"), "sage2"),
                {"sentinel": True},
            )
        with mock.patch.object(
            _ATTENTION, "get_default_attention_mode", return_value="sage2"
        ), mock.patch.object(
            _ATTENTION, "_sage2_staged_settings", return_value={"sentinel": True}
        ), mock.patch.object(
            _ATTENTION, "sageattn2", object()
        ):
            self.assertEqual(
                _ATTENTION.sage2_staged_settings(torch.device("cpu"), "auto"),
                {"sentinel": True},
            )

    def test_h3_inplace_norm_matches_rmsnorm_across_chunks(self):
        module = _import_project_module("models.minimax_h3.transformer")
        norm = nn.RMSNorm(8, eps=1e-5)
        torch.manual_seed(29)
        values = torch.randn(1, 13, 3, 8, dtype=torch.float32)
        expected = norm(values.clone())
        with torch.no_grad():
            actual = module._rms_norm_inplace(norm, values, chunk_bytes=48)
        self.assertIs(actual, values)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_h3_selected_head_split_uses_staged_path_with_fused_rms_rope(self):
        module = _import_project_module("models.minimax_h3.transformer")
        fake_mmgp = SimpleNamespace(offload=_OFFLOAD)
        rotary = (
            torch.ones(1, 4, 1, 64),
            torch.zeros(1, 4, 1, 64),
        )
        for group_count, settings in ((2, None), (1, {"staged": True})):
            with self.subTest(group_count=group_count, settings=settings):
                attention = module.MiniMaxH3Attention(
                    256, 4, 64, 1e-5, torch.float32
                ).eval()
                del attention.qkv_proj
                attention.q_proj = nn.Linear(256, 256, bias=False)
                attention.k_proj = nn.Linear(256, 256, bias=False)
                attention.v_proj = nn.Linear(256, 256, bias=False)
                group_width = 4 // group_count
                group_ranges = tuple(
                    slice(start, start + group_width)
                    for start in range(0, 4, group_width)
                )

                def staged_attention(
                    input_list,
                    q_proj,
                    k_proj,
                    v_proj,
                    heads,
                    head_dim,
                    norm_rope,
                ):
                    values = input_list.pop()
                    query, key, value = [
                        projection(values).view(1, 4, heads, head_dim)
                        for projection in (q_proj, k_proj, v_proj)
                    ]
                    for head_range in group_ranges:
                        norm_rope(
                            query[:, :, head_range], key[:, :, head_range], head_range
                        )
                    return F.scaled_dot_product_attention(
                        query.transpose(1, 2),
                        key.transpose(1, 2),
                        value.transpose(1, 2),
                    ).transpose(1, 2)

                with (
                    mock.patch.dict(
                        sys.modules,
                        {"mmgp": fake_mmgp, "mmgp.offload": _OFFLOAD},
                    ),
                    mock.patch.object(
                        _OFFLOAD,
                        "linear_rows_supported",
                        return_value=True,
                        create=True,
                    ),
                    mock.patch.object(
                        _ATTENTION_KIT, "head_groups", return_value=group_count
                    ),
                    mock.patch.object(
                        _ATTENTION_KIT,
                        "sage2_staged_settings",
                        return_value=settings,
                    ),
                    mock.patch.object(
                        _ATTENTION_KIT,
                        "qkv_attention",
                        side_effect=staged_attention,
                    ) as staged,
                    mock.patch.object(
                        module.denoiser_kernels,
                        "can_rms_rope",
                        return_value=True,
                    ),
                    mock.patch.object(
                        module.denoiser_kernels,
                        "rms_rope",
                        side_effect=lambda query, key, *_args: (query + 1, key + 1),
                    ) as fused_norm_rope,
                    torch.no_grad(),
                ):
                    output = attention(torch.randn(1, 4, 256), rotary)

                staged.assert_called_once()
                self.assertEqual(fused_norm_rope.call_count, group_count)
                self.assertEqual(tuple(output.shape), (1, 4, 256))

    def test_wan_fp32_rope_chunks_match_full_sequence_math(self):
        path = os.path.join(
            _APP, "models", "wan", "modules", "posemb_layers.py"
        )
        spec = importlib.util.spec_from_file_location(
            "maestro_wan_posemb_test", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(31)
        values = torch.randn(2, 7, 3, 8, dtype=torch.float16)
        cosine = torch.randn(1, 7, 1, 8, dtype=torch.float32)
        sine = torch.randn(1, 7, 1, 8, dtype=torch.float32)
        work = values.float().reshape(2, 7, 3, 4, 2)
        cosine_pairs = cosine.reshape(1, 7, 1, 4, 2)
        sine_pairs = sine.reshape(1, 7, 1, 4, 2)
        first = work[..., 0].clone()
        second = work[..., 1]
        expected = torch.stack(
            (
                first * cosine_pairs[..., 0] - second * sine_pairs[..., 0],
                second * cosine_pairs[..., 1] + first * sine_pairs[..., 1],
            ),
            dim=-1,
        ).flatten(-2).to(values.dtype)
        with mock.patch.object(module, "_ROPE_FP32_CHUNK_BYTES", 48):
            actual = module._apply_rope_inplace(
                values.clone(), cosine, sine, use_fp32=True
            )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-3, rtol=1e-3))

    def test_h3_vdn_recomputation_matches_legacy_and_releases_prepared_input(self):
        vdn_module = _import_project_module("models.minimax_h3.vdn_attention")
        offload = _OFFLOAD
        _ATTENTION_KIT.configure(0)
        torch.manual_seed(37)
        batch, tokens, hidden, heads, head_dim = 1, 4, 8, 2, 4
        values = torch.randn(tokens, hidden)
        projections = tuple(
            nn.Linear(hidden, heads * head_dim, bias=False) for _ in range(3)
        )
        q_norm, k_norm = nn.RMSNorm(head_dim), nn.RMSNorm(head_dim)
        out_proj = nn.Linear(hidden, hidden, bias=False)
        attention = vdn_module.VDNHybridAttention(hidden, heads, head_dim)
        layout = SimpleNamespace(
            sequence_length=tokens,
            text_indices=torch.tensor([0]),
        )
        attention.begin_forward(layout, 3, 1, 1, (1, 1, 1))
        # Checkpoint loading normally fills these raw Parameters. This small
        # unit-test module has no checkpoint, so initialize them explicitly
        # instead of letting allocator contents control the parity assertion.
        with torch.no_grad():
            attention.linear_attention.alpha.A_log.copy_(
                torch.tensor([-0.25, 0.35])
            )
            attention.linear_attention.alpha.dt_bias.copy_(
                torch.linspace(-0.4, 0.4, heads * head_dim)
            )

        raw = [
            projection(values.unsqueeze(0)).view(
                batch, tokens, heads, head_dim
            )
            for projection in projections
        ]
        legacy_softmax = [q_norm(raw[0]), k_norm(raw[1]), raw[2].clone()]
        with torch.no_grad(), mock.patch("builtins.print"):
            expected = attention.forward_legacy(
                [values.clone()], [tensor[0].clone() for tensor in raw], legacy_softmax,
                out_proj,
            )

        prepared_refs = []
        recomputed_refs = []
        row_calls = []

        def prepare_linear_input(input_list, _projections):
            prepared = input_list[0].clone()
            prepared_refs.append(weakref.ref(prepared))
            input_list.clear()
            return prepared

        def linear_rows(module, prepared, start, stop):
            row_calls.append((module, start, stop))
            bias = None if module.bias is None else module.bias[start:stop]
            return F.linear(prepared, module.weight[start:stop], bias)

        def norm_rope(query, key):
            if query is not None:
                query.copy_(q_norm(query))
            if key is not None:
                key.copy_(k_norm(key))

        def input_again():
            value = values.clone()
            recomputed_refs.append(weakref.ref(value))
            return value

        with (
            torch.no_grad(),
            mock.patch("builtins.print"),
            mock.patch.object(vdn_module, "offload", offload),
            mock.patch.object(vdn_module, "attention_kit", _ATTENTION_KIT),
            mock.patch.object(
                offload, "linear_rows_supported", return_value=True, create=True
            ),
            mock.patch.object(
                offload,
                "prepare_linear_input",
                side_effect=prepare_linear_input,
                create=True,
            ),
            mock.patch.object(
                offload, "linear_rows", side_effect=linear_rows, create=True
            ),
        ):
            actual = attention(
                [values.clone()],
                projections,
                norm_rope,
                out_proj,
                input_again,
            )

        gc.collect()
        self.assertEqual(len(recomputed_refs), 2)
        self.assertEqual(len(prepared_refs), 3)
        self.assertTrue(all(reference() is None for reference in prepared_refs))
        self.assertTrue(all(reference() is None for reference in recomputed_refs))
        self.assertEqual(len(row_calls), 4)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

    def test_h3_sol_staged_head_groups_match_dense_attention_and_release_inputs(self):
        sol_module = _import_project_module("models.minimax_h3.sol_attention")
        offload = _OFFLOAD
        heads, head_dim, hidden, batch, tokens = 8, 4, 32, 1, 12
        torch.manual_seed(41)
        values = torch.randn(batch, tokens, hidden)
        projections = tuple(
            nn.Linear(hidden, heads * head_dim, bias=False) for _ in range(3)
        )
        q_norm, k_norm = nn.RMSNorm(head_dim), nn.RMSNorm(head_dim)
        query, key, value = [
            projection(values).view(batch, tokens, heads, head_dim)
            for projection in projections
        ]
        query, key = q_norm(query), k_norm(key)
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)

        fake_mmgp = SimpleNamespace(offload=offload)
        fake_sol = SimpleNamespace()
        key_refs = []
        row_calls = []

        def prepare_linear_input(input_list, _projections):
            prepared = input_list.pop()
            return prepared

        def linear_rows(module, prepared, start, stop):
            row_calls.append(module)
            bias = None if module.bias is None else module.bias[start:stop]
            return F.linear(prepared, module.weight[start:stop], bias)

        def prepare_kv(raw_key, raw_value):
            key_refs.append(weakref.ref(raw_key))
            return raw_key.clone(), raw_value.clone()

        def sol_attn_prepared(query, raw_value, prepared, *, out, **_kwargs):
            prepared_key, _summary = prepared
            attended = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                prepared_key.transpose(1, 2),
                raw_value.transpose(1, 2),
            ).transpose(1, 2)
            out.copy_(attended)
            return out

        fake_sol.prepare_kv = prepare_kv
        fake_sol.sol_attn_prepared = sol_attn_prepared
        owned = values.clone()
        owned_ref = weakref.ref(owned)
        handoff = [owned]
        del owned

        def norm_rope(query_part, key_part):
            if query_part is not None:
                query_part.copy_(q_norm(query_part))
            if key_part is not None:
                key_part.copy_(k_norm(key_part))

        policy = sol_module.MiniMaxH3SolAttention()
        policy.enabled = True
        policy.sink_tokens = 3
        with (
            mock.patch.object(_ATTENTION_KIT, "head_split", 1),
            mock.patch.object(_ATTENTION_KIT, "MIN_SPLIT_TOKENS", 1),
            mock.patch.dict(
                sys.modules,
                {
                    "mmgp": fake_mmgp,
                    "mmgp.offload": offload,
                    "shared.sol_attn": fake_sol,
                    "shared.attention_kit": _ATTENTION_KIT,
                },
            ),
            mock.patch.object(
                offload, "prepare_linear_input", side_effect=prepare_linear_input,
                create=True,
            ),
            mock.patch.object(
                offload, "linear_rows", side_effect=linear_rows, create=True
            ),
            mock.patch.object(
                offload, "linear_rows_supported", return_value=True, create=True
            ),
            torch.no_grad(),
        ):
            actual = policy.attention(
                handoff,
                projections,
                norm_rope,
                heads,
                head_dim,
            )

        gc.collect()
        self.assertEqual(handoff, [])
        self.assertIsNone(owned_ref())
        self.assertEqual(len(row_calls), 12)
        for group in range(4):
            self.assertEqual(
                row_calls[group * 3 : group * 3 + 3],
                [projections[2], projections[1], projections[0]],
            )
        self.assertTrue(all(reference() is None for reference in key_refs))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_regular_sol_uses_inline_q_for_long_pointer_attention(self):
        fwd = _import_saganaki_fwd_for_pointer_test()

        batch, tokens, heads, head_dim = 1, 4096, 2, 128
        q, k, v = [
            torch.empty(batch, tokens, heads, head_dim, dtype=torch.bfloat16)
            for _ in range(3)
        ]
        prepared = tuple(torch.empty(1) for _ in range(6))
        launches = []

        class Launcher:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    launches.append((grid, args, kwargs))

                return launch

        with (
            mock.patch.object(fwd, "_validate", return_value=(8, 9)),
            mock.patch.object(fwd, "prepare_int8_kv", return_value=prepared) as prepare_kv,
            mock.patch.object(fwd, "prepare") as materialized_prepare,
            mock.patch.object(fwd, "_forward_int8_ptr", Launcher()),
        ):
            output = fwd.sol_attn(
                q,
                k,
                v,
                int8_qk=True,
                sink_blocks=(0, 1),
            )

        prepare_kv.assert_called_once_with(k, v)
        materialized_prepare.assert_not_called()
        self.assertEqual(tuple(output.shape), tuple(q.shape))
        self.assertEqual(len(launches), 1)
        _grid, args, options = launches[0]
        self.assertTrue(options["INLINE_Q"])
        self.assertIs(args[0], q)
        self.assertIs(args[4], q)
        self.assertIs(args[8], q)
        self.assertEqual(args[14:18], (0, 1, 0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
