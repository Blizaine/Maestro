"""CPU regressions for LongCat Avatar attention's temporary lifetimes."""

from __future__ import annotations

import gc
import os
import sys
import unittest
import weakref
from unittest import mock

import torch


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)


def _tiny_attention(qkv):
    """Deterministic CPU stand-in for the optional attention backend."""
    q, k, v = (tensor.float() for tensor in qkv)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    output = torch.matmul(scores.softmax(dim=-1), v).permute(0, 2, 1, 3)
    return output.to(qkv[0].dtype)


def _legacy_forward(
    module,
    x,
    *,
    shape,
    num_cond_latents,
    num_ref_latents,
    ref_img_index,
    mask_frame_range,
    return_kv=False,
    ref_target_masks=None,
):
    """Small pre-cleanup reference for the continuation branch semantics."""
    batch, tokens, channels = x.shape
    out_dtype = x.dtype
    qkv = module.qkv(x)
    if qkv.dtype != out_dtype:
        qkv = qkv.to(out_dtype)
    qkv = qkv.view(batch, tokens, 3, module.num_heads, module.head_dim)
    q, k, v = qkv.unbind(2)
    q, k = module.q_norm(q), module.k_norm(k)

    if return_kv:
        k_cache, v_cache = k.clone(), v.clone()

    q, k = module.rope_3d(q, k, shape, ref_img_index, num_ref_latents)
    frame_tokens = tokens // shape[0]
    ref_tokens = frame_tokens
    cond_tokens = num_cond_latents * frame_tokens

    q_ref = q[:, :ref_tokens].contiguous()
    k_ref = k[:, :ref_tokens].contiguous()
    v_ref = v[:, :ref_tokens].contiguous()
    q_cond = q[:, ref_tokens:cond_tokens].contiguous()
    k_cond = k[:, ref_tokens:cond_tokens].contiguous()
    v_cond = v[:, ref_tokens:cond_tokens].contiguous()
    x_ref = module._process_attn(q_ref, k_ref, v_ref, shape, out_dtype)
    x_cond = module._process_attn(q_cond, k_cond, v_cond, shape, out_dtype)

    if num_cond_latents == shape[0]:
        output = torch.cat([x_ref, x_cond], dim=1).contiguous()
    else:
        q_noise = q[:, cond_tokens:].contiguous()
        start_noise = end_noise = 0
        noisy_frames = shape[0] - num_cond_latents
        if mask_frame_range is not None and mask_frame_range > 0:
            start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents
            end_noise = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1

        if 0 <= start_noise < end_noise <= noisy_frames:
            start_pos = start_noise * frame_tokens
            end_pos = end_noise * frame_tokens
            q_front = q_noise[:, :start_pos].contiguous()
            q_mask = q_noise[:, start_pos:end_pos].contiguous()
            q_back = q_noise[:, end_pos:].contiguous()
            k_non_ref = k[:, ref_tokens:].contiguous()
            v_non_ref = v[:, ref_tokens:].contiguous()
            x_front = module._process_attn(q_front, k, v, shape, out_dtype)
            x_back = module._process_attn(q_back, k, v, shape, out_dtype)
            x_mask = module._process_attn(q_mask, k_non_ref, v_non_ref, shape, out_dtype)
            x_noise = torch.cat([x_front, x_mask, x_back], dim=1).contiguous()
        else:
            x_noise = module._process_attn(q_noise, k, v, shape, out_dtype)
        output = torch.cat([x_ref, x_cond, x_noise], dim=1).contiguous()

    output = module.proj(output.reshape(batch, tokens, channels))
    attn_map = None
    if ref_target_masks is not None:
        attn_map = module_impl.get_attn_map_with_target(
            q[:, cond_tokens:].type_as(output),
            k.type_as(output),
            shape,
            ref_target_masks=ref_target_masks,
            cp_split_hw=module.cp_split_hw,
        )
    if return_kv:
        return output, (k_cache, v_cache), attn_map
    return output, attn_map


try:
    from models.longcat.modules.avatar import attention as module_impl
except Exception as exc:  # pragma: no cover - optional runtime dependencies
    raise unittest.SkipTest(f"LongCat attention dependencies unavailable: {exc}") from exc


class TestLongCatAvatarAttentionMemory(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1741)
        self.attention = module_impl.Attention(dim=16, num_heads=2).eval()
        self.x = torch.randn(1, 8, 16)
        self.kwargs = {
            "shape": (8, 1, 1),
            "num_cond_latents": 3,
            "num_ref_latents": 1,
            "ref_img_index": 4,
            "mask_frame_range": 1,
        }

    def _run_pair(self, **overrides):
        kwargs = {**self.kwargs, **overrides}
        with mock.patch.object(module_impl, "pay_attention", side_effect=_tiny_attention):
            with torch.no_grad():
                expected = _legacy_forward(self.attention, self.x.clone(), **kwargs)
                actual = self.attention(self.x.clone(), **kwargs)
        return expected, actual

    def test_reference_mask_split_and_invalid_range_match_original_order(self):
        for name, ref_index, expected_lengths in (
            ("masked split", 4, [1, 2, 1, 1, 3]),
            ("fallback", 20, [1, 2, 5]),
        ):
            with self.subTest(name=name):
                calls = []

                def record_attention(qkv):
                    calls.append((qkv[0].shape[1], qkv[1].shape[1]))
                    return _tiny_attention(qkv)

                kwargs = {**self.kwargs, "ref_img_index": ref_index}
                with mock.patch.object(module_impl, "pay_attention", side_effect=record_attention):
                    with torch.no_grad():
                        expected = _legacy_forward(self.attention, self.x.clone(), **kwargs)
                        legacy_calls = calls[:]
                        calls.clear()
                        actual = self.attention(self.x.clone(), **kwargs)

                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                self.assertEqual([q_len for q_len, _ in legacy_calls], expected_lengths)
                self.assertEqual(calls, legacy_calls)
                self.assertIsNone(actual[1])

    def test_query_and_qkv_storage_are_released_before_projection(self):
        qkv_activations = []
        attention_outputs = []
        projection_observations = []
        self.attention.qkv.register_forward_hook(
            lambda _module, _args, output: qkv_activations.append(weakref.ref(output))
        )
        process_attention = self.attention._process_attn

        def track_attention(*args, **kwargs):
            output = process_attention(*args, **kwargs)
            attention_outputs.append(weakref.ref(output))
            return output

        self.attention._process_attn = track_attention

        def inspect_before_projection(_module, _inputs):
            gc.collect()
            projection_observations.append(
                qkv_activations[0]() is None
                and all(reference() is None for reference in attention_outputs)
            )

        self.attention.proj.register_forward_pre_hook(inspect_before_projection)
        with mock.patch.object(module_impl, "pay_attention", side_effect=_tiny_attention):
            with torch.no_grad():
                self.attention(self.x, **self.kwargs)

        self.assertEqual(projection_observations, [True])

    def test_kv_cache_and_reference_map_contracts_remain_available(self):
        expected, actual = self._run_pair(return_kv=True)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assertIsNone(actual[2])
        self.assertEqual(len(actual[1]), 2)
        for actual_cache, expected_cache in zip(actual[1], expected[1]):
            torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)

        reference_map = torch.tensor([0.25])
        target_masks = torch.ones(1, dtype=torch.bool)
        with mock.patch.object(
            module_impl,
            "get_attn_map_with_target",
            return_value=reference_map,
        ) as build_map:
            _expected, actual = self._run_pair(ref_target_masks=target_masks)
        self.assertIs(actual[1], reference_map)
        self.assertEqual(build_map.call_count, 2)
        for call in build_map.call_args_list:
            self.assertEqual(call.args[0].shape, (1, 5, 2, 8))
            self.assertEqual(call.args[1].shape, (1, 8, 2, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
