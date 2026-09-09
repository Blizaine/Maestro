"""Regression checks for LTX2 gated attention under mixed activation dtypes.

`to_gate_logits` is on the LTX2 never-quantize list, so it stays bf16 while the
surrounding projections are quanto int8. The injected Triton int8 kernels emit
bf16 activations, but the exact optimum.quanto path used by voice-clone /
ID-LoRA runs preserves fp32 ones, which previously crashed the gate projection
with "mat1 and mat2 must have the same dtype".
"""

from __future__ import annotations

import os
import sys
import unittest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)


class TestGatedAttentionMixedDtype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch

            from models.ltx2.ltx_core.model.transformer.attention import (
                Attention,
                AttentionFunction,
            )
        except Exception as exc:
            raise unittest.SkipTest(f"LTX2 attention dependencies unavailable: {exc}") from exc
        cls.torch = torch
        cls.Attention = Attention
        cls.AttentionFunction = AttentionFunction

    def _build(self):
        """fp32 projections with a bf16 gate — the quanto int8 layout."""
        attn = self.Attention(
            query_dim=64,
            context_dim=64,
            heads=4,
            dim_head=16,
            apply_gated_attention=True,
            attention_function=self.AttentionFunction.PYTORCH,
        ).to(self.torch.float32)
        attn.eval()
        attn.to_gate_logits.to(self.torch.bfloat16)
        return attn

    def test_self_attention_gate_accepts_fp32_activations(self):
        attn = self._build()
        x = self.torch.randn(1, 8, 64)
        with self.torch.no_grad():
            out = attn([x])
        self.assertEqual(out.shape, (1, 8, 64))
        self.assertEqual(out.dtype, self.torch.float32)

    def test_nag_cross_attention_gate_accepts_fp32_activations(self):
        attn = self._build()
        x = self.torch.randn(1, 8, 64)
        context = self.torch.randn(1, 12, 64)
        nag = {"cap_embed_len": 6, "scale": 1.5, "alpha": 0.5, "tau": 2.5}
        with self.torch.no_grad():
            out = attn([x], context_list=[context], NAG=nag)
        self.assertEqual(out.shape, (1, 8, 64))
        self.assertEqual(out.dtype, self.torch.float32)

    def test_gate_still_works_when_dtypes_already_match(self):
        attn = self._build()
        attn.to_gate_logits.to(self.torch.float32)
        x = self.torch.randn(1, 8, 64)
        with self.torch.no_grad():
            out = attn([x])
        self.assertEqual(out.dtype, self.torch.float32)


if __name__ == "__main__":
    unittest.main()
