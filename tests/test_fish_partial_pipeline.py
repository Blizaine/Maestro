"""CPU checks for pipeline state transitions with a sampling-only graph."""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import models
import torch

if "models.TTS" not in sys.modules:
    package = types.ModuleType("models.TTS")
    package.__path__ = [str(ROOT / "app/models/TTS")]
    sys.modules["models.TTS"] = package
    models.TTS = package

from models.TTS.fish_audio.decode_graph import WholeFrameDecoder
from models.TTS.fish_audio.pipeline import FishAudioPipeline
from models.TTS.fish_audio.tower_decode_graph import TowerFrameDecoder


class FishPartialPipelineTests(unittest.TestCase):
    def run_codes(self, tokens, *, maximum=4, finite=True):
        pipeline = object.__new__(FishAudioPipeline)
        backbone, tower = Mock(), Mock()
        backbone.model.layers = [torch.nn.Identity()]
        tower.shared_layers, tower.residual_layers = [torch.nn.Identity()], []
        tower.embed_backbone.side_effect = lambda frames, embedding: embedding
        backbone.embed.side_effect = lambda ids: torch.ones(len(ids), 2)
        backbone.side_effect = lambda embedding, *args: embedding
        backbone.compute_logits.return_value = torch.ones(1, 32)
        backbone.decode.return_value = (torch.ones(1, 2), torch.ones(1, 32))
        tower.decode_step.side_effect = AssertionError("partial graph used eager tower")
        pipeline.transformer, pipeline.transformer2 = backbone, tower
        pipeline.config = types.SimpleNamespace(
            text=types.SimpleNamespace(max_position_embeddings=64), eos_token_id=0)
        pipeline.engine, pipeline.profile = "vllm", 4
        pipeline._interrupt = pipeline._early_stop = False
        pipeline.silence_probe = None
        pipeline._place_vllm_gpu_experts = Mock()
        graph = Mock(capture_seconds=0.25)
        graph.step.side_effect = [
            (torch.tensor([[token, code]]), torch.tensor(finite),
             torch.ones(1, 2), torch.tensor([True]))
            for token, code in tokens]
        with (
            patch.object(WholeFrameDecoder, "try_create", return_value=(None, "CPU experts")),
            patch.object(TowerFrameDecoder, "try_create", return_value=(graph, None)),
            patch("models.TTS.fish_audio.pipeline.runtime.execution_device", return_value=torch.device("cpu")),
        ):
            codes, reason = pipeline._generate_codes(
                torch.tensor([[1, 0], [2, 0], [3, 0]]), maximum, 0.8, 0, 0.95, Mock())
        return pipeline, graph, codes, reason

    def test_partial_graph_keeps_backbone_eager_and_drops_eos(self):
        pipeline, graph, codes, reason = self.run_codes([(16, 7), (0, 8)])
        self.assertEqual((reason, codes.tolist()), ("eos", [[7]]))
        self.assertEqual(pipeline.last_decode_backend, "tower_cuda_graph")
        self.assertEqual(pipeline.last_graph_capture_seconds, 0.25)
        self.assertEqual([call.args[0] for call in graph.step.call_args_list], [2, 3])
        # Eager decoding may queue one look-ahead before the host reads EOS;
        # it never becomes a retained audio frame.
        self.assertEqual(pipeline.transformer.decode.call_count, 2)

    def test_last_allowed_frame_does_not_queue_another_backbone_step(self):
        pipeline, graph, codes, reason = self.run_codes([(16, 7)], maximum=1)
        self.assertEqual((reason, codes.tolist()), ("length", [[7]]))
        pipeline.transformer.decode.assert_not_called()
        self.assertEqual(graph.step.call_count, 1)

    def test_partial_graph_nonfinite_output_still_fails(self):
        with self.assertRaises(FloatingPointError):
            self.run_codes([(16, 7)], maximum=1, finite=False)

    def test_partial_graph_and_slots_release_before_codec(self):
        pipeline = object.__new__(FishAudioPipeline)
        order = []
        graph, bank = Mock(), Mock()
        graph.close.side_effect = lambda: order.append("graph")
        bank.release_decode_cache.side_effect = lambda: order.append("slots")
        pipeline._frame_graph, pipeline._tower_graph = None, graph
        pipeline.transformer, pipeline.transformer2 = Mock(), Mock()
        pipeline.transformer.release_cache.side_effect = lambda: order.append("backbone")
        pipeline.transformer2.release_cache.side_effect = lambda: order.append("tower")
        pipeline._expert_banks = Mock(return_value=[bank])
        with patch.object(torch.cuda, "is_initialized", return_value=False):
            pipeline._release_decoder_caches()
        self.assertEqual(order, ["graph", "backbone", "tower", "slots"])
        self.assertIsNone(pipeline._tower_graph)

    def test_cleanup_synchronizes_the_actual_decode_device(self):
        pipeline = object.__new__(FishAudioPipeline)
        pipeline._decode_device = torch.device("cuda:1")
        pipeline._frame_graph = pipeline._tower_graph = None
        pipeline.transformer, pipeline.transformer2 = Mock(), Mock()
        pipeline._expert_banks = Mock(return_value=[])
        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(torch.cuda, "current_stream") as stream,
        ):
            pipeline._release_decoder_caches()
        stream.assert_called_once_with(torch.device("cuda:1"))
        stream.return_value.synchronize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
