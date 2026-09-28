"""CPU numerical tests for Krea 2 Identity Edit reference-key attention bias."""
from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F
from einops import rearrange


_ROOT = Path(__file__).resolve().parents[1]
_MMDIT_PATH = _ROOT / "app" / "models" / "krea2" / "krea2_mmdit.py"


def _cpu_pay_attention(qkv_list, attention_mask=None, softmax_scale=None, recycle_q=False):
    q, k, v = qkv_list
    qkv_list.clear()
    kwargs = {} if softmax_scale is None else {"scale": softmax_scale}
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attention_mask,
        **kwargs,
    )
    return out.transpose(1, 2)


def _load_attention_functions():
    tree = ast.parse(_MMDIT_PATH.read_text(encoding="utf-8"), filename=str(_MMDIT_PATH))
    wanted = {"_attention_from_blh", "_reference_attention_mask", "_attention_with_reference_bias", "attention"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise AssertionError("Krea 2 attention helper definitions are incomplete")
    namespace = {
        "math": math,
        "torch": torch,
        "Tensor": torch.Tensor,
        "rearrange": rearrange,
        "pay_attention": _cpu_pay_attention,
    }
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_MMDIT_PATH), "exec"), namespace)
    return namespace


def _dense_oracle(q, k, v, *, mask, key_text_length, reference_token_lengths,
                  ref_boost, ref_boost_a, target_query_start, gqa=False):
    """Dense per-query oracle kept test-only; production uses broadcast key bias."""
    batch, query_length, query_heads, head_dim = q.shape
    key_length, key_heads = k.shape[1], k.shape[2]
    if gqa:
        repeat = query_heads // key_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
    dense_bias = torch.zeros((batch, 1, query_length, key_length), dtype=q.dtype, device=q.device)
    cursor = key_text_length
    for index, length in enumerate(reference_token_lengths):
        if index == len(reference_token_lengths) - 1:
            boost = ref_boost
        elif index == 0 and len(reference_token_lengths) > 1:
            boost = ref_boost_a
        else:
            boost = 1.0
        if boost != 1.0:
            dense_bias[:, :, target_query_start:, cursor:cursor + length] = math.log(max(boost, 1e-4))
        cursor += length
    if mask is not None:
        if mask.dtype == torch.bool:
            dense_bias.masked_fill_(~mask, float("-inf"))
        else:
            dense_bias += mask
    out = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        attn_mask=dense_bias,
    )
    return out.permute(0, 2, 1, 3).reshape(batch, query_length, query_heads * head_dim)


class TestKrea2ReferenceBoost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.impl = _load_attention_functions()

    def _random_stream(self, *, batch=2, queries=9, keys=9, q_heads=4, kv_heads=2, dim=4):
        torch.manual_seed(1729)
        q = torch.randn(batch, queries, q_heads, dim)
        k = torch.randn(batch, keys, kv_heads, dim)
        v = torch.randn(batch, keys, kv_heads, dim)
        return q, k, v

    def test_neutral_settings_take_identical_unmodified_attention_path(self):
        q, k, v = self._random_stream(q_heads=2, kv_heads=2)
        mask = torch.ones(2, 1, 1, 9, dtype=torch.bool)
        mask[..., -1] = False
        expected = self.impl["_attention_from_blh"](q, k, v, mask=mask)
        actual = self.impl["_attention_with_reference_bias"](
            q, k, v, mask=mask, scale=None, gqa=False, key_text_length=2,
            reference_token_lengths=(2, 1), target_len=3, ref_boost=1.0,
            ref_boost_a=1.0,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_boost_maps_first_scene_and_last_subject_only_on_target_queries(self):
        q, k, v = self._random_stream()
        mask = torch.ones(2, 1, 1, 9, dtype=torch.bool)
        mask[..., -1] = False  # padded key must remain fully masked
        actual = self.impl["_attention_with_reference_bias"](
            q, k, v, mask=mask, scale=None, gqa=True, key_text_length=2,
            reference_token_lengths=(2, 1), target_len=3, ref_boost=4.0,
            ref_boost_a=2.0,
        )
        expected = _dense_oracle(
            q, k, v, mask=mask, key_text_length=2,
            reference_token_lengths=(2, 1), ref_boost=4.0, ref_boost_a=2.0,
            target_query_start=5, gqa=True,
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

        # Production evaluates the prefix as a query-length-5 SDPA call while
        # a neutral boost takes the query-length-9 fast path. Different CPU
        # SDPA kernels may round those shapes differently, so compare against
        # the exact same unmodified prefix call instead of requiring bitwise
        # equality with a differently batched full-sequence result.
        neutral_prefix = self.impl["_attention_from_blh"](
            q[:, :5], k, v, mask=mask, scale=None, gqa=True,
        )
        self.assertTrue(torch.equal(actual[:, :5], neutral_prefix))

    def test_single_reference_uses_subject_boost_and_ignores_scene_boost(self):
        q, k, v = self._random_stream(q_heads=2, kv_heads=2)
        mask = torch.ones(2, 1, 1, 9, dtype=torch.bool)
        actual = self.impl["_attention_with_reference_bias"](
            q, k, v, mask=mask, scale=None, gqa=False, key_text_length=2,
            reference_token_lengths=(3,), target_len=4, ref_boost=3.0,
            ref_boost_a=8.0,
        )
        expected = _dense_oracle(
            q, k, v, mask=mask, key_text_length=2,
            reference_token_lengths=(3,), ref_boost=3.0, ref_boost_a=1.0,
            target_query_start=5,
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_negative_guidance_prefix_offsets_reference_keys(self):
        q, k, v = self._random_stream(batch=1, queries=4, keys=8, q_heads=2, kv_heads=2)
        mask = torch.ones(1, 1, 1, 8, dtype=torch.bool)
        mask[..., -1] = False
        actual = self.impl["_attention_with_reference_bias"](
            q, k, v, mask=mask, scale=None, gqa=False, key_text_length=3,
            reference_token_lengths=(2, 1), target_len=4, ref_boost=5.0,
            ref_boost_a=0.5, target_query_start=0,
        )
        expected = _dense_oracle(
            q, k, v, mask=mask, key_text_length=3,
            reference_token_lengths=(2, 1), ref_boost=5.0, ref_boost_a=0.5,
            target_query_start=0,
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_nag_applies_same_reference_bias_to_positive_and_negative_streams(self):
        torch.manual_seed(314)
        batch, heads, seq, dim = 1, 2, 6, 4
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        neg_k = torch.randn(batch, heads, 2, dim)
        neg_v = torch.randn(batch, heads, 2, dim)
        mask = torch.ones(batch, 1, 1, seq, dtype=torch.bool)
        neg_mask = torch.ones(batch, 2, dtype=torch.bool)
        actual = self.impl["attention"](
            [q.clone(), k.clone(), v.clone()], mask=mask, txt_len=2,
            NAG={"cap_embed_len": 2, "query_start": 4, "query_end": 6,
                 "scale": 0.6, "alpha": 0.7, "tau": 3.5},
            neg_k=neg_k, neg_v=neg_v, neg_mask=neg_mask,
            reference_token_lengths=(1, 1), target_len=2,
            ref_boost=3.0, ref_boost_a=0.5,
        )
        q_blhd, k_blhd, v_blhd = (rearrange(value, "B H L D -> B L H D") for value in (q, k, v))
        positive = _dense_oracle(
            q_blhd, k_blhd, v_blhd, mask=mask, key_text_length=2,
            reference_token_lengths=(1, 1), ref_boost=3.0, ref_boost_a=0.5,
            target_query_start=4,
        )
        negative_k = torch.cat((rearrange(neg_k, "B H L D -> B L H D"), k_blhd[:, 2:]), dim=1)
        negative_v = torch.cat((rearrange(neg_v, "B H L D -> B L H D"), v_blhd[:, 2:]), dim=1)
        negative_mask = torch.cat((neg_mask[:, None, None, :], mask[..., 2:]), dim=-1)
        guidance = _dense_oracle(
            q_blhd[:, 4:6], negative_k, negative_v, mask=negative_mask,
            key_text_length=2, reference_token_lengths=(1, 1), ref_boost=3.0,
            ref_boost_a=0.5, target_query_start=0,
        )
        positive_image = positive[:, 4:6]
        guidance = (1 - 0.6) * guidance + 0.6 * positive_image
        norm_positive = torch.norm(positive_image, p=1, dim=-1, keepdim=True)
        norm_guidance = torch.norm(guidance, p=1, dim=-1, keepdim=True)
        norm_scale = norm_guidance / norm_positive
        factor = (norm_positive * 3.5) / (norm_guidance + 1e-7)
        guidance = torch.where(norm_scale > 3.5, guidance * factor, guidance)
        guidance = 0.7 * guidance + 0.3 * positive_image
        expected = positive.clone()
        expected[:, 4:6] = guidance
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
