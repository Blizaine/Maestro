"""CPU contracts for whole-frame transitions and transactional capture."""
# ruff: noqa: E402
import os
from contextlib import nullcontext
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import torch
import models

if "models.TTS" not in sys.modules:
    package = types.ModuleType("models.TTS")
    package.__path__ = [str(ROOT / "app/models/TTS")]
    sys.modules["models.TTS"] = package
    models.TTS = package

from models.TTS.fish_audio import decode_graph as graph_module
from models.TTS.fish_audio.backbone import GenerationCancelled
from models.TTS.fish_audio.decode_graph import (
    GraphStateRestoreError, WholeFrameDecoder, preserve_decode_state,
)


class TinyBackbone:
    cache_capacity = 32

    def __init__(self):
        self.kv = torch.arange(32, dtype=torch.float32)
        self.conv = torch.ones(2)
        self.state = torch.ones(2) * 2

    def compute_logits(self, hidden):
        return hidden * 2

    def decode_token(self, tokens, extra, semantic, position, capacity, continuing, kv, conv, state, *, return_logits=True):
        self.kv.index_copy_(0, position, tokens.float())
        self.conv.add_(1)
        self.state.add_(2)
        return extra + tokens[:, None], extra * 2


class TinyTower:
    cache_capacity = 32

    def __init__(self):
        self.kv = torch.arange(32, dtype=torch.float32)
        self.main_candidates = torch.arange(4)
        self.candidate_rows = torch.arange(4)
        self.finite = True
        self.top_k_seen = []

    def prepare_sampling(self):
        pass

    def sample_frame(self, logits, hidden, position, frame, history, kv, candidates, rows,
                     capacity, temperature, top_k, top_p):
        self.top_k_seen.append(top_k.clone())
        token = position + 2
        self.kv.index_copy_(0, position, token.float())
        sampled = torch.stack((token, token + 1), -1)
        # Exercise RNG consumption by warmup/capture without making assertions
        # dependent on a randomly chosen token.
        torch.rand(1)
        return sampled, torch.tensor(self.finite), hidden + 1, torch.ones(1, dtype=torch.bool)


class FishDecodeGraphTests(unittest.TestCase):
    def make_decoder(self):
        backbone, tower = TinyBackbone(), TinyTower()
        return WholeFrameDecoder(backbone, tower, torch.ones(1, 2),
                                 torch.tensor([[7, 8]]), torch.tensor([4, 5, 6, 7]),
                                 3, 0.8, 0, 0.95)

    def test_whole_frame_advances_all_state_and_preserves_uncapped_sampling(self):
        decoder = self.make_decoder()
        decoder._run_frame()
        self.assertEqual(decoder.sampled.tolist(), [[5, 6]])
        self.assertEqual(decoder.frame.tolist(), [[5, 6]])
        self.assertEqual(decoder.position.tolist(), [4])
        self.assertEqual(decoder.history[-5:].tolist(), [4, 5, 6, 7, 5])
        torch.testing.assert_close(decoder.hidden, torch.full((1, 2), 7.0))
        self.assertEqual(decoder.backbone.kv[4].item(), 5)
        self.assertEqual(decoder.tower.kv[3].item(), 5)
        decoder._run_frame()
        self.assertEqual(decoder.sampled.tolist(), [[6, 7]])
        self.assertEqual(decoder.position.tolist(), [5])
        self.assertEqual(decoder.history[-2:].tolist(), [5, 6])
        self.assertTrue(all(value.item() == 0 for value in decoder.tower.top_k_seen))
        self.assertEqual(decoder.backbone.conv.tolist(), [3, 3])
        self.assertEqual(decoder.backbone.state.tolist(), [6, 6])

    def test_finite_flag_is_kept_in_graph_result(self):
        decoder = self.make_decoder()
        decoder.tower.finite = False
        decoder._run_frame()
        self.assertFalse(decoder.finite.item())

    def test_capture_transaction_restores_cache_contents_addresses_and_rng_on_error(self):
        decoder = self.make_decoder()
        caches = decoder._cache_objects
        expected = [tensor.clone() for tensor in caches]
        pointers = [tensor.data_ptr() for tensor in caches]
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "unsupported capture"):
            with preserve_decode_state(decoder.backbone, decoder.tower, decoder.device):
                decoder._run_frame()
                raise RuntimeError("unsupported capture")
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for cache, saved, pointer in zip(caches, expected, pointers):
            self.assertTrue(torch.equal(cache, saved))
            self.assertEqual(cache.data_ptr(), pointer)

    def test_failed_rollback_is_fatal_instead_of_claiming_eager_recovery(self):
        decoder = self.make_decoder()
        with patch.object(torch.Tensor, "copy_", side_effect=RuntimeError("device failure")):
            with self.assertRaises(GraphStateRestoreError):
                with preserve_decode_state(decoder.backbone, decoder.tower, decoder.device):
                    pass

    def capture_context(self, failure=None, abort=None):
        decoder = self.make_decoder()
        graph = Mock()
        stream = Mock()
        capture = Mock(side_effect=failure) if failure else Mock(return_value=nullcontext())
        args = (decoder.backbone, decoder.tower, torch.ones(1, 2), torch.tensor([[7, 8]]),
                torch.tensor([4, 5, 6, 7]), 3, 0.8, 0, 0.95)
        expected = [value.clone() for value in decoder._cache_objects]
        rng = torch.get_rng_state().clone()
        with (
            patch.object(graph_module, "graph_eligibility", return_value=(True, None)),
            patch.object(graph_module, "weight_signature", return_value=("stable",)),
            patch.object(graph_module, "resident_capture_scope", return_value=nullcontext()),
            patch.object(torch.cuda, "mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
            patch.object(torch.cuda, "Stream", return_value=stream),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "CUDAGraph", return_value=graph),
            patch.object(torch.cuda, "graph", capture),
        ):
            result = WholeFrameDecoder.try_create(*args, engine="vllm", profile=1, check_abort=abort)
        for value, saved in zip(decoder._cache_objects, expected):
            self.assertTrue(torch.equal(value, saved))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        return result

    def test_successful_capture_restores_real_prefill_and_starts_at_original_position(self):
        (decoder, reason) = self.capture_context()
        self.assertIsNone(reason)
        self.assertEqual(decoder.position.tolist(), [3])
        decoder.graph.replay.side_effect = decoder._run_frame
        sampled, finite = decoder.step(3)
        self.assertEqual(sampled.tolist(), [[5, 6]])
        self.assertTrue(finite.item())
        sampled, _ = decoder.step(4)
        self.assertEqual(sampled.tolist(), [[6, 7]])
        self.assertEqual(decoder.graph.replay.call_count, 2)

    def test_unsupported_capture_falls_back_before_generating_a_real_frame(self):
        decoder, reason = self.capture_context(RuntimeError("capture unsupported"))
        self.assertIsNone(decoder)
        self.assertIn("capture unsupported", reason)

    def test_capture_cancellation_is_not_swallowed_into_eager_generation(self):
        abort = Mock(side_effect=[None, GenerationCancelled()])
        with self.assertRaises(GenerationCancelled):
            self.capture_context(abort=abort)

    def test_stale_cache_closed_graph_and_wrong_position_refuse_replay(self):
        decoder = self.make_decoder()
        decoder.graph = Mock()
        with self.assertRaisesRegex(RuntimeError, "position"):
            decoder.step(4)
        decoder.backbone.kv = decoder.backbone.kv.clone()
        with self.assertRaisesRegex(RuntimeError, "caches changed"):
            decoder.step(3)
        decoder.graph.replay.assert_not_called()
        decoder.close()
        with self.assertRaisesRegex(RuntimeError, "released"):
            decoder.step(3)

    def test_unsupported_profiles_never_start_cuda_capture(self):
        decoder = self.make_decoder()
        with (
            patch.object(graph_module, "graph_eligibility", return_value=(False, "offloaded")),
            patch.object(torch.cuda, "CUDAGraph", side_effect=AssertionError("unexpected capture")),
        ):
            result, reason = WholeFrameDecoder.try_create(
                decoder.backbone, decoder.tower, decoder.hidden, decoder.frame,
                decoder.history, 3, 0.8, 0, 0.95, engine="legacy", profile=4)
        self.assertIsNone(result)
        self.assertEqual(reason, "offloaded")


class FishGraphPipelineTests(unittest.TestCase):
    """Exercise the real pipeline's graph-result handling on CPU."""

    def run_codes(self, results, maximum=4, early_stop=False):
        from models.TTS.fish_audio.pipeline import FishAudioPipeline

        pipeline = object.__new__(FishAudioPipeline)
        backbone, tower = Mock(), Mock()
        backbone.model.layers = [torch.nn.Identity()]
        tower.shared_layers, tower.residual_layers = [torch.nn.Identity()], []
        tower.embed_backbone.side_effect = lambda frame, embedding: embedding
        backbone.embed.side_effect = lambda tokens: torch.ones(len(tokens), 2)
        backbone.side_effect = lambda embedding, *args: embedding
        tower.decode_step.side_effect = AssertionError("graph path used eager tower")
        pipeline.transformer, pipeline.transformer2 = backbone, tower
        pipeline.config = types.SimpleNamespace(
            text=types.SimpleNamespace(max_position_embeddings=64), eos_token_id=0)
        pipeline.engine, pipeline.profile = "vllm", 1
        pipeline._interrupt, pipeline._early_stop = False, early_stop
        pipeline.silence_probe = None
        pipeline._place_vllm_gpu_experts = Mock()
        graph = Mock(capture_seconds=0.125)
        graph.step.side_effect = results
        callback = Mock()
        with (
            patch.object(graph_module.WholeFrameDecoder, "try_create", return_value=(graph, None)),
            patch("models.TTS.fish_audio.pipeline.runtime.execution_device", return_value=torch.device("cpu")),
        ):
            codes, reason = pipeline._generate_codes(
                torch.tensor([[1, 0], [2, 0], [3, 0]]), maximum, 0.8, 0, 0.95, callback)
        return pipeline, graph, codes, reason

    def test_eos_is_not_retained_as_an_audio_frame(self):
        pipeline, graph, codes, reason = self.run_codes([
            (torch.tensor([[16, 7]]), torch.tensor(True)),
            (torch.tensor([[0, 8]]), torch.tensor(True)),
        ])
        self.assertEqual(reason, "eos")
        self.assertEqual(codes.tolist(), [[7]])
        self.assertEqual([call.args[0] for call in graph.step.call_args_list], [2, 3])
        self.assertEqual(pipeline.last_decode_backend, "cuda_graph")
        self.assertEqual(pipeline.last_graph_capture_seconds, 0.125)
        pipeline.transformer.decode.assert_not_called()

    def test_single_frame_limit_and_early_stop_remain_bounded(self):
        _, graph, codes, reason = self.run_codes([
            (torch.tensor([[16, 7]]), torch.tensor(True))], maximum=1)
        self.assertEqual((reason, codes.tolist(), graph.step.call_count), ("length", [[7]], 1))
        _, graph, codes, reason = self.run_codes([], early_stop=True)
        self.assertEqual(reason, "early_stop")
        self.assertEqual(codes.shape[-1], 0)
        graph.step.assert_not_called()

    def test_nonfinite_graph_result_is_rejected(self):
        with self.assertRaises(FloatingPointError):
            self.run_codes([(torch.tensor([[16, 7]]), torch.tensor(False))])

    def test_graph_is_closed_before_releasing_decoder_caches(self):
        from models.TTS.fish_audio.pipeline import FishAudioPipeline

        pipeline = object.__new__(FishAudioPipeline)
        order = []
        pipeline._frame_graph = Mock()
        pipeline._frame_graph.close.side_effect = lambda: order.append("graph")
        pipeline.transformer, pipeline.transformer2 = Mock(), Mock()
        pipeline.transformer.release_cache.side_effect = lambda: order.append("backbone")
        pipeline.transformer2.release_cache.side_effect = lambda: order.append("tower")
        with patch.object(torch.cuda, "is_initialized", return_value=False):
            pipeline._release_decoder_caches()
        self.assertEqual(order, ["graph", "backbone", "tower"])
        self.assertIsNone(pipeline._frame_graph)


if __name__ == "__main__":
    unittest.main()
