"""CPU-only contract tests for the Fish Audio Drama 3 port."""

# ruff: noqa: E402

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Keep this contract suite CPU-only even when run on a CUDA workstation.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

# Import the Fish package without executing the broad TTS model registry, which
# imports unrelated optional inference stacks. This test only exercises its
# local CPU code; the real application imports the same package files normally.
import models

tts_package = sys.modules.get("models.TTS", getattr(models, "TTS", None))
if tts_package is None:
    tts_package = types.ModuleType("models.TTS")
    tts_package.__path__ = [str(APP / "models" / "TTS")]
sys.modules["models.TTS"] = tts_package
# Keep the package tree and sys.modules in sync. Python 3.10's
# unittest.mock.patch("models.TTS....") resolves TTS through the parent module.
models.TTS = tts_package

# Import MMGP as the application does. The CPU tests do not load models, and
# keeping the real module available prevents this test import from masking
# later tests that exercise MMGP's embedding and ConvRot support.
import mmgp
from mmgp import offload as mmgp_offload

from mmgp.expert_compute import SwiGLUExperts
from mmgp.experts import excluded_names
from shared.llm_engines.nanovllm import vllm_support as shared_vllm_support
from shared.llm_engines.nanovllm import layerwise as shared_vllm_layerwise
from shared.llm_engines.nanovllm.layers import attention as shared_vllm_attention
from shared.llm_engines.nanovllm.models import qwen3_5 as shared_qwen3_5
from models.TTS.fish_audio import pipeline as fish_pipeline_module
from models.TTS.fish_audio import backbone as fish_backbone_module
from models.TTS.fish_audio.codec import ModelArgs as CodecModelArgs
from models.TTS.fish_audio.codec import WindowLimitedTransformer
from models.TTS.fish_audio.config import AudioDecoderConfig, FishOmniTextConfig
from models.TTS.fish_audio.layers import FullAttention, LayerContext
from models.TTS.fish_audio.tower import FishOmniAudioDecoder, TowerAttention, sample_top_k_top_p
from models.TTS.fish_audio.expert_formats import (
    EXPERT_FORMATS,
    FishSwiGLUExperts,
    Fp8BlockExperts,
    Int4GroupExperts,
)
from models.TTS.fish_audio import native_cpu
from models.TTS.fish_audio.row_ops import rows_
from models.TTS.fish_audio.native_cpu import availability as native_cpu_availability
from models.TTS.fish_audio.fish_audio_handler import family_handler
from models.TTS.fish_audio.pipeline import FishAudioPipeline
from models.TTS.fish_audio.prompt import (
    dialogue_parts,
    reference_transcripts,
    speaker_lines_to_tags,
)
from models.TTS.fish_audio.runtime_support import (
    can_place_gpu_experts,
    fish_vllm_availability,
    runtime as fish_runtime,
)


class FishSamplingTests(unittest.TestCase):
    def test_zero_top_k_keeps_candidates_and_positive_top_k_still_caps(self):
        logits = torch.ones((128, 2), dtype=torch.float32)

        torch.manual_seed(271828)
        uncapped = sample_top_k_top_p(logits, 1.0, 0, 1.0)
        self.assertEqual(set(uncapped.tolist()), {0, 1})

        torch.manual_seed(271828)
        top_one = sample_top_k_top_p(logits, 1.0, 1, 1.0)
        self.assertTrue(torch.equal(top_one, torch.zeros_like(top_one)))

    def test_scalar_and_tensor_top_k_have_identical_sampling(self):
        logits = torch.tensor([[2.0, 1.0, 0.0]] * 32)
        for top_k in (0, 2):
            with self.subTest(top_k=top_k):
                torch.manual_seed(12345)
                scalar = sample_top_k_top_p(logits, 0.8, top_k, 0.95)
                torch.manual_seed(12345)
                tensor = sample_top_k_top_p(logits, 0.8, torch.tensor(top_k), 0.95)
                self.assertTrue(torch.equal(scalar, tensor))


class FishExpertMathTests(unittest.TestCase):
    def test_row_ops_chunks_with_global_indices_and_updates_in_place(self):
        tensor = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2)
        original = tensor.clone()
        calls = []

        def transform(chunk, first, last):
            calls.append((first, last))
            return chunk + torch.arange(first, last).view(1, -1, 1)

        rows_(tensor, transform, start=1, stop=5, row_elements=4)

        expected = original.clone()
        expected[:, 1:5] += torch.arange(1, 5).view(1, -1, 1)
        self.assertEqual(calls, [(1, 2), (2, 3), (3, 4), (4, 5)])
        torch.testing.assert_close(tensor, expected)

    def test_vllm_delta_rule_uses_maestro_fla_without_optional_tuning_hook(self):
        query = torch.randn(1, 2, 2, 4)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        decay = torch.zeros(1, 2, 2)
        beta = torch.ones_like(decay)
        initial_state = torch.zeros(1, 2, 4, 3)
        expected = (torch.ones_like(query), torch.ones_like(initial_state))
        chunk_kernel = Mock(return_value=expected)
        torch_impl = Mock(side_effect=AssertionError("legacy fallback was used"))

        with (
            patch.object(shared_qwen3_5, "_DEFAULT_FAST_CHUNK_GATED_DELTA_RULE", chunk_kernel),
            patch.object(shared_qwen3_5, "configure_qwen35_fla_prefill_autotune", None, create=True),
        ):
            actual = shared_vllm_layerwise.delta_rule(
                query,
                key,
                value,
                decay,
                beta,
                initial_state,
                engine="vllm",
                recurrent=False,
                torch_impl=torch_impl,
            )

        self.assertIs(actual, expected)
        args, kwargs = chunk_kernel.call_args
        self.assertEqual(len(args), 3)
        torch.testing.assert_close(args[0], query)
        torch.testing.assert_close(args[1], key)
        torch.testing.assert_close(args[2], value)
        torch.testing.assert_close(kwargs["g"], decay)
        torch.testing.assert_close(kwargs["beta"], beta)
        self.assertIs(kwargs["initial_state"], initial_state)
        self.assertTrue(kwargs["output_final_state"])
        self.assertTrue(kwargs["use_qk_l2norm_in_kernel"])
        torch_impl.assert_not_called()

    def test_backbone_full_attention_gqa_prefill_and_decode_use_shared_contract(self):
        config = FishOmniTextConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            rope_theta=10000.0,
            partial_rotary_factor=0.5,
            rms_norm_eps=1e-6,
            max_position_embeddings=16,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
        )
        module = FullAttention(config)
        module.engine = "legacy"
        key_cache = torch.zeros(1, 4, 1, 4)
        value_cache = torch.zeros_like(key_cache)
        cache = (key_cache, value_cache)

        prefill_positions = torch.arange(3)
        prefill_context = LayerContext(
            input_pos=prefill_positions,
            cos=torch.ones(3, 2),
            sin=torch.zeros(3, 2),
            kv_len=3,
            continue_state=False,
            use_cache=False,
        )
        prefill = module(torch.randn(1, 3, 8), prefill_context, cache)

        decode_context = LayerContext(
            input_pos=torch.tensor([3]),
            cos=torch.ones(1, 2),
            sin=torch.zeros(1, 2),
            kv_len=4,
            continue_state=True,
            use_cache=True,
        )
        decode = module(torch.randn(1, 1, 8), decode_context, cache)

        self.assertEqual(tuple(prefill.shape), (1, 3, 8))
        self.assertEqual(tuple(decode.shape), (1, 1, 8))
        self.assertTrue(torch.isfinite(prefill).all())
        self.assertTrue(torch.isfinite(decode).all())
        self.assertGreater(float(key_cache[:, :4].abs().sum()), 0)

    def test_audio_tower_attention_gqa_uses_shared_group_callback(self):
        config = AudioDecoderConfig(
            dim=8,
            intermediate_size=12,
            n_head=2,
            n_local_heads=1,
            head_dim=4,
            norm_eps=1e-6,
            rope_base=10000.0,
            rotary_percent=0.5,
            vocab_size=8,
            num_codebooks=2,
            shared_enrichment_layers=1,
            vq_backbone_codebooks=0,
        )
        module = TowerAttention(config)
        module.engine = "legacy"
        cache = (torch.zeros(1, 4, 1, 4), torch.zeros(1, 4, 1, 4))
        positions = torch.arange(3)

        prefill = module(torch.randn(3, 8), positions, cache, kv_len=3)
        decode = module(torch.randn(1, 8), torch.tensor([3]), cache, kv_len=4)
        batched_cache = (torch.zeros(1, 4, 1, 4), torch.zeros(1, 4, 1, 4))
        batched = module(
            torch.randn(1, 2, 8), torch.arange(2), batched_cache, kv_len=2
        )

        self.assertEqual(tuple(prefill.shape), (3, 8))
        self.assertEqual(tuple(decode.shape), (1, 8))
        self.assertEqual(tuple(batched.shape), (1, 2, 8))
        self.assertTrue(torch.isfinite(prefill).all())
        self.assertTrue(torch.isfinite(decode).all())
        self.assertTrue(torch.isfinite(batched).all())
        self.assertGreater(float(cache[0][:, :3].abs().sum()), 0)

    def test_codec_transformer_forward_passes_mask_through_attention_callback(self):
        config = CodecModelArgs(
            n_layer=1,
            n_head=2,
            n_local_heads=1,
            dim=8,
            intermediate_size=12,
            head_dim=4,
            dropout_rate=0.0,
            channels_first=False,
        )
        module = WindowLimitedTransformer(
            config,
            input_dim=8,
            window_size=2,
            causal=True,
        )

        result = module(torch.randn(1, 4, 8))

        self.assertEqual(tuple(result.shape), (1, 4, 8))
        self.assertTrue(torch.isfinite(result).all())

    def test_fp8_block_dequantizes_each_128_by_128_scale(self):
        fmt = Fp8BlockExperts()
        weights = {
            "w13_weight": torch.full((256, 128), 2.0, dtype=torch.float8_e4m3fn),
            "w2_weight": torch.full((128, 128), -1.0, dtype=torch.float8_e4m3fn),
            "w13_scale_inv": torch.tensor([[0.5], [0.25]], dtype=torch.float32),
            "w2_scale_inv": torch.tensor([[2.0]], dtype=torch.float32),
        }

        w13, w2 = fmt.dequantize(weights)

        self.assertTrue(torch.equal(w13[:128], torch.ones((128, 128))))
        self.assertTrue(torch.equal(w13[128:], torch.full((128, 128), 0.5)))
        self.assertTrue(torch.equal(w2, torch.full((128, 128), -2.0)))

    def test_int4_unpacks_low_nibble_first_with_zero_point_eight(self):
        fmt = Int4GroupExperts()
        weights = {
            "w13_weight": torch.full((256, 64), 0x99, dtype=torch.uint8),
            "w2_weight": torch.full((128, 64), 0x77, dtype=torch.uint8),
            "w13_scales": torch.full((256, 1), 2.0, dtype=torch.float16),
            "w2_scales": torch.full((128, 1), 0.5, dtype=torch.float16),
        }

        w13, w2 = fmt.dequantize(weights)

        self.assertTrue(torch.equal(w13, torch.full((256, 128), 2.0)))
        self.assertTrue(torch.equal(w2, torch.full((128, 128), -0.5)))

    def test_routed_expert_load_and_forward_keep_bank_on_cpu(self):
        fmt = Fp8BlockExperts()
        names = fmt.tensors(2, 128, 128)
        state = {
            "w13_weight": torch.ones(names["w13_weight"][0], dtype=torch.float8_e4m3fn),
            "w2_weight": torch.ones(names["w2_weight"][0], dtype=torch.float8_e4m3fn),
            "w13_scale_inv": torch.tensor([[[0.01], [0.01]], [[0.02], [0.02]]], dtype=torch.float32),
            "w2_scale_inv": torch.tensor([[[0.01]], [[0.005]]], dtype=torch.float32),
        }
        with torch.device("meta"):
            experts = SwiGLUExperts(2, 128, 128, formats=EXPERT_FORMATS)
        experts.load_state_dict(state, assign=True)

        x = torch.ones((2, 128), dtype=torch.float32)
        ids = torch.tensor([[0], [1]])
        weights = torch.tensor([[0.25], [0.75]])
        actual = experts(x, ids, weights)
        expected = []
        for row, expert_id, route_weight in zip(x, (0, 1), (0.25, 0.75)):
            bank_weights = {
                name: tensor[expert_id]
                for name, tensor in state.items()
            }
            w13, w2 = fmt.dequantize(bank_weights)
            gate, up = F.linear(row[None], w13).chunk(2, dim=-1)
            expected.append(F.linear(F.silu(gate) * up, w2)[0] * route_weight)

        self.assertTrue(torch.allclose(actual, torch.stack(expected), atol=1e-5, rtol=1e-5))

        class Wrapper(nn.Module):
            def __init__(self, bank):
                super().__init__()
                self.experts = bank

        model = Wrapper(experts)
        self.assertEqual(excluded_names(model), {"experts"})
        model.to(dtype=torch.float16)
        self.assertTrue(all(getattr(experts, name).device.type == "cpu" for name in experts.expert_tensor_names))
        self.assertTrue(all(getattr(experts, name).dtype != torch.float16 for name in experts.expert_tensor_names))

    def test_codebook_embedding_sum_calls_offloadable_embedding_module(self):
        decoder = object.__new__(FishOmniAudioDecoder)
        nn.Module.__init__(decoder)
        decoder.config = types.SimpleNamespace(dim=4, vocab_size=8)
        decoder._model_dtype = torch.float32
        decoder.codebook_embeddings = nn.Embedding(24, 4)
        frames = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.long)
        rows = frames[:, 1:3] + torch.tensor([0, 8])
        expected = F.embedding_bag(rows, decoder.codebook_embeddings.weight, mode="sum")

        with patch.object(
            decoder.codebook_embeddings,
            "forward",
            wraps=decoder.codebook_embeddings.forward,
        ) as embedding_forward:
            actual = FishOmniAudioDecoder.embedding_sum(decoder, frames, 0, 2)

        embedding_forward.assert_called_once()
        self.assertTrue(torch.equal(actual, expected))

    def test_native_cpu_kernel_matches_portable_fp8_and_int4(self):
        available, reason = native_cpu_availability()
        if not available:
            self.skipTest(f"optional Fish native CPU library unavailable: {reason}")

        torch.manual_seed(2701)
        for fmt in (Fp8BlockExperts(), Int4GroupExperts()):
            declarations = fmt.tensors(2, 128, 128)
            if fmt.name == "fp8_block":
                state = {
                    "w13_weight": torch.randn(declarations["w13_weight"][0]).mul_(0.4).to(torch.float8_e4m3fn),
                    "w2_weight": torch.randn(declarations["w2_weight"][0]).mul_(0.4).to(torch.float8_e4m3fn),
                    "w13_scale_inv": torch.rand(declarations["w13_scale_inv"][0]) * 0.03,
                    "w2_scale_inv": torch.rand(declarations["w2_scale_inv"][0]) * 0.03,
                }
            else:
                state = {
                    "w13_weight": torch.randint(0, 256, declarations["w13_weight"][0], dtype=torch.uint8),
                    "w2_weight": torch.randint(0, 256, declarations["w2_weight"][0], dtype=torch.uint8),
                    "w13_scales": torch.rand(declarations["w13_scales"][0], dtype=torch.float16) * 0.02,
                    "w2_scales": torch.rand(declarations["w2_scales"][0], dtype=torch.float16) * 0.02,
                }

            with torch.device("meta"):
                native = FishSwiGLUExperts(2, 128, 128, formats=EXPERT_FORMATS)
                portable = SwiGLUExperts(2, 128, 128, formats=EXPERT_FORMATS)
            native.load_state_dict({name: value.clone() for name, value in state.items()}, assign=True)
            portable.load_state_dict({name: value.clone() for name, value in state.items()}, assign=True)
            x = torch.randn((4, 128), dtype=torch.float32)
            ids = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 1]], dtype=torch.int64)
            probabilities = torch.tensor(
                [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.65, 0.35]],
                dtype=torch.float32,
            )

            expected = portable(x, ids, probabilities)
            actual = native(x, ids, probabilities)

            self.assertEqual(native.execution_backend, "native_cpu")
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.allclose(actual, expected, atol=2e-3, rtol=2e-3), fmt.name)

    def test_fish_bank_falls_back_to_portable_torch_when_native_is_unavailable(self):
        fmt = Fp8BlockExperts()
        declarations = fmt.tensors(1, 128, 128)
        state = {
            "w13_weight": torch.ones(declarations["w13_weight"][0], dtype=torch.float8_e4m3fn),
            "w2_weight": torch.ones(declarations["w2_weight"][0], dtype=torch.float8_e4m3fn),
            "w13_scale_inv": torch.full(declarations["w13_scale_inv"][0], 0.01),
            "w2_scale_inv": torch.full(declarations["w2_scale_inv"][0], 0.01),
        }
        with torch.device("meta"):
            bank = FishSwiGLUExperts(1, 128, 128, formats=EXPERT_FORMATS)
        bank.load_state_dict(state, assign=True)
        x, ids, probabilities = torch.ones((1, 128)), torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 1))

        with patch.object(native_cpu, "forward", return_value=None):
            actual = bank(x, ids, probabilities)
        expected = SwiGLUExperts.forward(bank, x, ids, probabilities)

        self.assertEqual(bank.execution_backend, "portable_torch")
        self.assertTrue(torch.allclose(actual, expected))

    def test_portable_convrot_fallback_matches_native_when_available(self):
        fmt = Int4GroupExperts()
        declarations = fmt.tensors(1, 128, 128)
        state = {
            "w13_weight": torch.randint(0, 256, declarations["w13_weight"][0], dtype=torch.uint8),
            "w2_weight": torch.randint(0, 256, declarations["w2_weight"][0], dtype=torch.uint8),
            "w13_scales": torch.rand(declarations["w13_scales"][0], dtype=torch.float16) * 0.02,
            "w2_scales": torch.rand(declarations["w2_scales"][0], dtype=torch.float16) * 0.02,
            "convrot_group_size": torch.tensor(64),
        }
        with torch.device("meta"):
            bank = FishSwiGLUExperts(1, 128, 128, formats=EXPERT_FORMATS)
        bank.load_state_dict(state, assign=True)
        self.assertEqual(bank.rotation, 64)

        torch.manual_seed(6401)
        x = torch.randn((3, 128), dtype=torch.float32)
        ids = torch.zeros((3, 1), dtype=torch.long)
        probabilities = torch.ones((3, 1), dtype=torch.float32)
        expected = SwiGLUExperts.forward(bank, x, ids, probabilities)
        with patch.object(native_cpu, "forward", return_value=None):
            actual = bank(x, ids, probabilities)
        self.assertEqual(bank.execution_backend, "portable_torch")
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

        available, _ = native_cpu_availability()
        if available:
            native = native_cpu.forward(bank, x, ids, probabilities)
            self.assertTrue(torch.allclose(native, expected, atol=2e-3, rtol=2e-3))


class FishScriptContractTests(unittest.TestCase):
    def test_string_patch_resolves_fish_module_under_models_tts_parent(self):
        self.assertIs(sys.modules["models"].TTS, sys.modules["models.TTS"])
        self.assertTrue(callable(mmgp_offload.load_model_data))

        with patch("models.TTS.fish_audio.prompt.dialogue_parts") as dialogue_parts:
            self.assertIsNotNone(dialogue_parts)

    def test_vllm_capability_gate_reports_shared_runtime_failures(self):
        with patch.object(
            shared_vllm_support,
            "probe_vllm_runtime",
            return_value={
                "supported": False,
                "checks": {"triton": {"ok": False, "message": "CUDA is not available"}},
            },
        ):
            available, reason = fish_vllm_availability()
        self.assertFalse(available)
        self.assertIn("CUDA is not available", reason)

    def test_vllm_pipeline_rejects_an_unavailable_engine_before_loading(self):
        with patch.object(fish_pipeline_module, "fish_vllm_availability", return_value=(False, "FLA missing")):
            with self.assertRaisesRegex(RuntimeError, "FLA missing"):
                FishAudioPipeline(None, None, None, None, "vllm", profile=3)

    def test_vllm_engine_allows_dense_resident_profiles_with_cpu_experts(self):
        backbone, tower, vae = types.SimpleNamespace(), types.SimpleNamespace(), types.SimpleNamespace()
        with (
            patch.object(fish_pipeline_module, "fish_vllm_availability", return_value=(True, None)),
            patch.object(fish_pipeline_module, "load_transformers", return_value=(None, backbone, tower)),
            patch.object(fish_pipeline_module, "load_codec", return_value=vae),
            patch.object(FishAudioPipeline, "_silence_classifier", return_value=(None, None, None, None)),
            patch.object(fish_pipeline_module.Tokenizer, "from_file", return_value=object()),
        ):
            pipeline = FishAudioPipeline(None, None, None, None, "vllm", profile=2)

        self.assertEqual(pipeline.engine, "vllm")
        self.assertEqual(pipeline.profile, 2)
        self.assertFalse(can_place_gpu_experts(1, 64, 64, 2))

    def test_vllm_gate_rejects_missing_fla_delta_kernels(self):
        with (
            patch.object(shared_vllm_support, "probe_vllm_runtime", return_value={"supported": True, "checks": {}}),
            patch.object(shared_vllm_attention, "_DEFAULT_FLASH_ATTN_VARLEN_FUNC", Mock()),
            patch.object(shared_vllm_attention, "_DEFAULT_FLASH_ATTN_WITH_KVCACHE", Mock()),
            patch.object(shared_qwen3_5, "_DEFAULT_FAST_CHUNK_GATED_DELTA_RULE", None),
            patch.object(shared_qwen3_5, "_DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE", None),
        ):
            available, reason = fish_vllm_availability()
        self.assertFalse(available)
        self.assertIn("chunk_gated_delta_rule", reason)
        self.assertIn("fused_recurrent_gated_delta_rule", reason)

    def test_vllm_decode_uses_graphless_per_layer_path(self):
        backbone = object.__new__(fish_backbone_module.FishOmniBackbone)
        nn.Module.__init__(backbone)
        backbone.cache_capacity = 4
        backbone.abort_requested = lambda: False
        backbone.mix_embeddings = Mock(return_value=torch.zeros((1, 4)))
        backbone.forward = Mock(return_value=torch.ones((1, 1, 4)))
        backbone.lm_head = Mock(return_value=torch.ones((1, 2)))

        with (
            patch.object(fish_pipeline_module.cuda_graphs, "run", side_effect=AssertionError("CUDA graph path was invoked")) as run,
            patch.object(fish_backbone_module.runtime, "ensure_loaded"),
        ):
            hidden, logits = backbone.decode(
                torch.zeros((1,), dtype=torch.long),
                torch.zeros((1, 4)),
                torch.ones((1,), dtype=torch.bool),
                torch.zeros((1,), dtype=torch.long),
                start=0,
            )

        self.assertEqual(hidden.shape, (1, 4))
        self.assertEqual(logits.shape, (1, 2))
        backbone.forward.assert_called_once()
        run.assert_not_called()

    def test_gpu_expert_placement_keeps_a_reserve_and_requires_full_vram_profile(self):
        gib = 1024**3
        self.assertTrue(can_place_gpu_experts(16 * gib, 24 * gib, 32 * gib, 3))
        self.assertFalse(can_place_gpu_experts(16 * gib, 18 * gib, 24 * gib, 3))
        self.assertFalse(can_place_gpu_experts(1 * gib, 64 * gib, 64 * gib, 2))
        self.assertFalse(can_place_gpu_experts(1 * gib, 64 * gib, 64 * gib, 0))

    def test_managed_root_force_device_wins_over_cpu_weight_storage(self):
        model = nn.Linear(2, 2)
        model._force_device = "cuda"

        self.assertEqual(fish_runtime.execution_device(model), torch.device("cuda"))

    def test_fish_cuda_device_alias_is_resolved_to_explicit_index(self):
        with patch.object(torch.cuda, "current_device", return_value=0):
            self.assertEqual(
                FishSwiGLUExperts._canonical_cuda_device(torch.device("cuda")),
                torch.device("cuda:0"),
            )
            self.assertEqual(
                FishSwiGLUExperts._canonical_cuda_device(torch.device("cuda:1")),
                torch.device("cuda:1"),
            )

    def test_three_reference_transcripts_and_script_speakers(self):
        transcripts = reference_transcripts(
            "Speaker 1: First voice words.\nSpeaker 2: Second voice words.\nSpeaker 3: Third voice words.",
            3,
        )
        self.assertEqual(transcripts, ["First voice words.", "Second voice words.", "Third voice words."])
        tagged = speaker_lines_to_tags("Speaker 1: Hello.\nSpeaker 3: From the other side.")
        self.assertIn("<|speaker:0|>", tagged)
        self.assertIn("<|speaker:2|>", tagged)

    def test_long_dialogue_chunks_keep_continued_speaker(self):
        script = "Speaker 1: One.\nSpeaker 2: Two.\nContinues.\nSpeaker 3: Three.\nContinues again."
        chunks = dialogue_parts(speaker_lines_to_tags(script), 2)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[1].startswith("<|speaker:1|> Continues."))
        self.assertTrue(chunks[2].startswith("<|speaker:2|> Continues again."))

    def test_history_is_bounded_by_recent_audio_duration(self):
        pipeline = object.__new__(FishAudioPipeline)
        pipeline._interrupt = False
        turns = [
            ("old", torch.zeros((10, 4), dtype=torch.long)),
            ("middle", torch.zeros((10, 5), dtype=torch.long)),
            ("recent", torch.zeros((10, 2), dtype=torch.long)),
        ]
        budget_seconds = 8 * pipeline.frame_length / pipeline.sample_rate

        kept = pipeline._history(turns, budget_seconds)

        self.assertEqual([text for text, _ in kept], ["middle", "recent"])

    def test_missing_reference_transcript_uses_maestro_transcription_hook(self):
        fake_service = types.ModuleType("services.audio_analysis")
        fake_service._transcribe = Mock(return_value=[types.SimpleNamespace(text=" spoken reference ")])
        fake_service.unload_whisper = Mock()
        pipeline = object.__new__(FishAudioPipeline)
        pipeline._interrupt = False
        from shared.utils.text_encoder_cache import TextEncoderCache

        pipeline.reference_cache = TextEncoderCache()
        with tempfile.NamedTemporaryFile(delete=False) as audio:
            audio.write(b"reference")
            path = audio.name
        try:
            with patch.dict(sys.modules, {"services.audio_analysis": fake_service}):
                transcripts = pipeline._reference_transcripts(
                    [path, path, path],
                    "Speaker 1: exact first.\nSpeaker 2:\nSpeaker 3: exact third.",
                )
            self.assertEqual(transcripts, ["exact first.", "spoken reference", "exact third."])
            fake_service._transcribe.assert_called_once_with(path)
            fake_service.unload_whisper.assert_called_once()
        finally:
            os.unlink(path)

    def test_no_silence_asset_returns_clean_optional_state(self):
        pipeline = object.__new__(FishAudioPipeline)
        result = pipeline._silence_classifier()
        self.assertEqual(result, (None, None, None, None))

    def test_model_definition_exposes_three_voice_slots_and_complete_assets(self):
        model_def = family_handler.query_model_def("fish_drama3", {})
        self.assertEqual(model_def["max_voice_count"], 3)
        self.assertTrue(model_def["audio_mode_from_voice_count"])
        self.assertEqual(model_def["lm_engines"], ["legacy"])
        self.assertFalse(model_def["cuda_graphs"])
        self.assertTrue(model_def["top_k_slider"])
        self.assertTrue(model_def["top_p_slider"])
        self.assertEqual(model_def["audio_prompt_type_sources"]["selection"], ["", "A", "AB", "ABD"])

        ui_defaults = {}
        family_handler.update_default_settings("fish_drama3", model_def, ui_defaults)
        self.assertEqual(ui_defaults["top_k"], 0)
        self.assertEqual(ui_defaults["top_p"], 0.95)
        self.assertEqual(inspect.signature(FishAudioPipeline.generate).parameters["top_k"].default, 0)
        for name in ("fish_drama3", "fish_drama3_int4"):
            with self.subTest(defaults=name):
                defaults = json.loads((ROOT / "app/defaults" / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(defaults["top_k"], 0)

        downloads = family_handler.query_model_files([], "fish_drama3")
        self.assertEqual(downloads[0]["revision"], "b0cc2d3cf0752bc0d8c704562c95d4d2564e6f8e")
        self.assertEqual(downloads[0]["fileList"], [["tokenizer.json", "Fish_Drama3_codec_bf16.safetensors"]])

        for name, expert_asset, expected_ram in (
            ("fish_drama3.json", "Fish_Drama3_experts_fp8.safetensors", 48),
            ("fish_drama3_int4.json", "Fish_Drama3_experts_int4.safetensors", 24),
        ):
            config = json.loads((ROOT / "app" / "defaults" / name).read_text(encoding="utf-8"))
            model = config["model"]
            self.assertEqual(model["model_size_gb"], expected_ram)
            self.assertIn("Fish Audio Community License", model["license_name"])
            self.assertEqual(
                set(model["required_model_assets"]),
                {
                    "fish_audio/tokenizer.json",
                    "fish_audio/Fish_Drama3_codec_bf16.safetensors",
                    expert_asset,
                },
            )
            model_urls = []
            for key in ("URLs", "preload_URLs", "URLs2"):
                value = model.get(key, [])
                model_urls.extend([value] if isinstance(value, str) else value)
            for url in model_urls:
                if url.startswith("https://huggingface.co/DeepBeepMeep/TTS/"):
                    self.assertIn("/resolve/b0cc2d3cf0752bc0d8c704562c95d4d2564e6f8e/", url)


if __name__ == "__main__":
    unittest.main()
