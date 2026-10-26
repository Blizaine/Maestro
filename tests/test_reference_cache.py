"""CPU checks for portable Qwen reference-cache storage and attention fidelity."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from shared.reference_cache import (
    QuantizedReferenceTensor, ReferenceKVLayerCache, normalize_cache_mode, quantized_storage_bytes,
)
from models.qwen21.memory import GIB, reference_cache_plan
from models.qwen21.transformer_qwenimage21 import QwenImage21KVCache, QwenImage21Transformer2DModel


class StorageTests(unittest.TestCase):
    def test_roundtrip_partial_groups_batch_and_noncontiguous_input(self):
        torch.manual_seed(9)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            source = torch.randn(2, 3, 7, 45, dtype=dtype).transpose(1, 2)
            self.assertFalse(source.is_contiguous())
            for ram in (False, True):
                packed = QuantizedReferenceTensor(source, ram=ram)
                actual = packed.materialize()
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(packed.storage_bytes, quantized_storage_bytes(source.shape))
                self.assertEqual(packed.data.device.type, "cpu")
                # Per-group quantization error <= half a quantization step,
                # plus the final FP16/BF16 rounding error.
                self.assertLess((actual.float() - source.float()).abs().max().item(), 0.035)

    def test_zeros_and_large_prefix_crossing_chunk_boundary(self):
        for source in (torch.zeros(1, 9, 2, 33), torch.randn(1, 18000, 2, 8)):
            packed = QuantizedReferenceTensor(source)
            actual = packed.materialize(dtype=torch.float32)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertLess((actual - source).abs().max().item(), 0.025)

    def test_cache_does_not_alias_prefill_and_clear_releases_both_tensors(self):
        for mode in ("Enabled", "INT8", "INT8_RAM"):
            source = torch.ones(1, 11, 2, 32)
            cache = ReferenceKVLayerCache(mode)
            cache.store(source[:, :3], source[:, :3] * 2)
            source.zero_()
            key, value = cache.get(dtype=torch.float32)
            torch.testing.assert_close(key, torch.ones_like(key))
            torch.testing.assert_close(value, torch.ones_like(value) * 2)
            cache.clear()
            with self.assertRaises(RuntimeError):
                cache.get()

    def test_invalid_mode_nonfinite_and_shape_fail_clearly(self):
        self.assertEqual(normalize_cache_mode(None), "Disabled")
        with self.assertRaises(ValueError):
            ReferenceKVLayerCache("typo")
        with self.assertRaises(ValueError):
            ReferenceKVLayerCache("Disabled").store(torch.ones(1), torch.ones(1))
        with self.assertRaises(ValueError):
            QuantizedReferenceTensor(torch.ones(2, 3))
        for bad in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                QuantizedReferenceTensor(torch.full((1, 1, 2, 32), bad))


class AttentionTests(unittest.TestCase):
    @torch.inference_mode()
    def test_quantized_cache_preserves_interleaved_reference_attention(self):
        torch.manual_seed(12)
        model = QwenImage21Transformer2DModel(in_channels=4, out_channels=4, num_layers=2,
            attention_head_dim=16, num_attention_heads=2, context_in_dim=8,
            mlp_ratio=2, axes_dims_rope=(4, 6, 6)).eval()
        args = dict(hidden_states=torch.randn(1, 8, 4), encoder_hidden_states=torch.randn(1, 5, 8),
            encoder_hidden_states_mask=None, img_shapes=[[(1, 2, 2), (1, 2, 2)]],
            img_mask=torch.tensor([[False, True, False, False, False, True]]))
        expected = model(**args, timestep=torch.tensor([0.3])).sample[:, -4:]
        for mode in ("Enabled", "INT8", "INT8_RAM"):
            cache = QwenImage21KVCache(2, mode)
            model(**args, timestep=torch.tensor([0.9]), kv_cache=cache, kv_cache_mode="extract")
            actual = model(**args, timestep=torch.tensor([0.3]), kv_cache=cache, kv_cache_mode="cached").sample[:, -4:]
            tolerance = 1e-5 if mode == "Enabled" else 0.006
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def test_cfg_and_new_generation_caches_are_independent(self):
        cond, negative = QwenImage21KVCache(2, "INT8"), QwenImage21KVCache(2, "INT8_RAM")
        cond.get_layer(0).store(torch.ones(1, 2, 2, 32), torch.ones(1, 2, 2, 32))
        negative.get_layer(0).store(torch.zeros(1, 2, 2, 32), torch.zeros(1, 2, 2, 32))
        self.assertEqual(cond.get_layer(0).get()[0].sum().item(), 128)
        self.assertEqual(negative.get_layer(0).get()[0].sum().item(), 0)
        with self.assertRaises(RuntimeError):
            QwenImage21KVCache(2, "INT8").get_layer(0).get()


class BudgetTests(unittest.TestCase):
    def plan(self, mode, available=64 * GIB, cfg=False):
        model = SimpleNamespace(config=SimpleNamespace(num_attention_heads=32, attention_head_dim=128),
            transformer_blocks=[None] * 32)
        mask = torch.tensor([[False] * 96 + [True] * 1024])
        embeds = torch.zeros(2, 1120, 1, dtype=torch.bfloat16)
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=available)):
            return reference_cache_plan(model, embeds, mask, 4096, 8160, "cpu",
                embeds if cfg else None, mask if cfg else None, cache_mode=mode)[0]

    def test_persistent_storage_counts_scales_and_cfg(self):
        full = self.plan("Enabled")
        quantized = self.plan("INT8")
        self.assertEqual(quantized, full * 9 // 16)
        self.assertEqual(self.plan("INT8", cfg=True), quantized * 2)
        self.assertEqual(self.plan("INT8_RAM"), full // 32)

    def test_ram_cache_refuses_insufficient_host_memory(self):
        self.assertEqual(self.plan("INT8_RAM", available=2 * GIB), float("inf"))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
