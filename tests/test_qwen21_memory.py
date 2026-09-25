"""CPU checks for Qwen's Windows SDPA path and reference-cache sizing."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3VLConfig, Qwen3VLModel
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from models.qwen21.memory import (
    ENCODER_ATTENTION, GIB, configure_encoder_attention, encoder_attention,
    model_storage_bytes, reference_cache_plan,
)


class EncoderAttentionTests(unittest.TestCase):
    @torch.inference_mode()
    def test_expanded_heads_match_native_gqa_without_the_gqa_flag(self):
        torch.manual_seed(12)
        q = torch.randn(2, 4, 17, 16)
        k, v = torch.randn(2, 2, 17, 16), torch.randn(2, 2, 17, 16)
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        for mask in (None, torch.ones(2, 1, 17, 17, dtype=torch.bool).tril()):
            expected = sdpa_attention_forward(module, q, k, v, mask, scaling=0.25)[0]
            with patch("torch.nn.functional.scaled_dot_product_attention",
                       wraps=torch.nn.functional.scaled_dot_product_attention) as sdpa:
                actual = encoder_attention(module, q, k, v, mask, scaling=0.25)[0]
            self.assertNotIn("enable_gqa", sdpa.call_args.kwargs)
            self.assertEqual(sdpa.call_args.args[1].shape[1], q.shape[1])
            torch.testing.assert_close(actual, expected)

    @torch.inference_mode()
    def test_qwen_text_padding_causality_and_vision_are_preserved(self):
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=100, hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                out_hidden_size=64, patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
                num_position_embeddings=16, deepstack_visual_indexes=[0, 1]),
        )
        torch.manual_seed(5)
        model = Qwen3VLModel(config).eval()
        model.set_attn_implementation("sdpa")
        ids = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 7, 8, 9, 10]])
        mask = ids != 0
        pixels, grid = torch.randn(16, 12), torch.tensor([[1, 4, 4]])
        original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        expected = model.language_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        vision_expected = model.visual(pixels, grid_thw=grid)
        configure_encoder_attention(model)
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], original_sdpa)
        self.assertEqual(model.config.text_config._attn_implementation, ENCODER_ATTENTION)
        self.assertEqual(model.config.vision_config._attn_implementation, ENCODER_ATTENTION)
        actual = model.language_model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual[mask], expected[mask])
        vision_actual = model.visual(pixels, grid_thw=grid)
        torch.testing.assert_close(vision_actual, vision_expected)
        # Padding must not enter the context; future tokens must not enter
        # earlier positions when the mask builder omits an all-valid mask.
        unpadded = model.language_model(input_ids=ids[1:2, 2:], use_cache=False).last_hidden_state
        torch.testing.assert_close(unpadded[0], actual[1, 2:])
        changed = ids[:1].clone()
        changed[:, 3:] = 15
        prefix = model.language_model(input_ids=ids[:1], use_cache=False).last_hidden_state
        changed_prefix = model.language_model(input_ids=changed, use_cache=False).last_hidden_state
        torch.testing.assert_close(prefix[:, :3], changed_prefix[:, :3])


class ReferenceCacheTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(config=SimpleNamespace(num_attention_heads=32, attention_head_dim=128),
                               transformer_blocks=[None] * 32)

    def plan(self, refs=10, batch=1, cfg=False, free_gb=23):
        # Ten 1MP references + a 1920x1088 output. Image placeholder rows
        # become 4x as many latent rows; target slots must not enter the cache.
        mask = torch.tensor([[False] * 96 + [True] * (refs * 1024 + 2040)])
        embeds = torch.zeros(batch, refs * 1024 + 96, 1, dtype=torch.bfloat16)
        with patch("torch.cuda.mem_get_info", return_value=(free_gb * GIB, 24 * GIB)), \
             patch("torch.cuda.memory_reserved", return_value=GIB), \
             patch("torch.cuda.memory_allocated", return_value=GIB), \
             patch("models.qwen21.memory.model_storage_bytes", return_value=7 * GIB):
            return reference_cache_plan(self.model(), embeds, mask, refs * 4096, 8160, "cuda",
                                        embeds if cfg else None, mask if cfg else None)

    def test_ten_reference_cache_cannot_fit_but_one_reference_can(self):
        required, budget = self.plan()
        self.assertGreater(required, 20 * GIB)
        self.assertGreater(required, budget)
        required, budget = self.plan(refs=1)
        self.assertLess(required, budget)

    def test_cache_counts_both_cfg_branches_and_batch(self):
        single, _ = self.plan(refs=2)
        cfg, _ = self.plan(refs=2, cfg=True)
        batch, _ = self.plan(refs=2, batch=3)
        self.assertEqual(cfg, single * 2)
        self.assertEqual(batch, single * 3)

    def test_other_gpu_allocations_reduce_the_cache_budget(self):
        _, normal = self.plan(refs=1)
        _, busy = self.plan(refs=1, free_gb=10)
        self.assertLess(busy, normal)
        self.assertEqual(busy, 0)

    def test_weights_count_quantized_storage_without_dequantizing(self):
        weight = torch.nn.Parameter(torch.zeros(32, 64), requires_grad=False)
        data, scales = torch.zeros(32, 64, dtype=torch.int8), torch.ones(32, 1)
        weight.get_quantized_subtensors = lambda: [("data", data), ("scale", scales)]
        model = torch.nn.Module()
        model.register_parameter("weight", weight)
        model.register_buffer("buffer", torch.ones(4))
        self.assertEqual(model_storage_bytes(model), 32 * 64 + 32 * 4 + 4 * 4)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
