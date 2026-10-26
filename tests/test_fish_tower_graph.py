"""CPU contracts for the Fish tower-only CUDA Graph."""

import importlib
import os
import sys
import types
import unittest
from contextlib import nullcontext
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

_FISH_PACKAGE = "models.TTS.fish_audio"
_BACKBONE_MODULE = f"{_FISH_PACKAGE}.backbone"
_TOWER_MODULE = f"{_FISH_PACKAGE}.tower"
_DECODE_GRAPH_MODULE = f"{_FISH_PACKAGE}.decode_graph"
_TOWER_GRAPH_MODULE = f"{_FISH_PACKAGE}.tower_decode_graph"

# This contract test does not execute Fish layers. When run alone in a small
# CPU environment, use import-only stubs for the two values the graph helper
# imports; remove the temporary modules so they cannot affect other tests.
_bootstrap_modules = not any(
    name in sys.modules for name in (_BACKBONE_MODULE, _TOWER_MODULE, _DECODE_GRAPH_MODULE)
)
if _bootstrap_modules:
    backbone_stub = types.ModuleType(_BACKBONE_MODULE)

    class GenerationCancelled(Exception):
        pass

    backbone_stub.GenerationCancelled = GenerationCancelled
    tower_stub = types.ModuleType(_TOWER_MODULE)
    tower_stub.RAS_WINDOW_FRAMES = 16
    with patch.dict(sys.modules, {_BACKBONE_MODULE: backbone_stub, _TOWER_MODULE: tower_stub}):
        graph_module = importlib.import_module(_TOWER_GRAPH_MODULE)
        decode_graph_module = importlib.import_module(_DECODE_GRAPH_MODULE)
    for name in (_TOWER_GRAPH_MODULE, _DECODE_GRAPH_MODULE):
        sys.modules.pop(name, None)
    fish_package = importlib.import_module(_FISH_PACKAGE)
    for name in ("backbone", "tower", "decode_graph", "tower_decode_graph"):
        if hasattr(fish_package, name):
            delattr(fish_package, name)
else:
    graph_module = importlib.import_module(_TOWER_GRAPH_MODULE)
    decode_graph_module = importlib.import_module(_DECODE_GRAPH_MODULE)

GenerationCancelled = graph_module.GenerationCancelled
GraphStateRestoreError = decode_graph_module.GraphStateRestoreError
TowerFrameDecoder = graph_module.TowerFrameDecoder


class TinyBackbone:
    def __init__(self):
        self.kv = torch.arange(8, dtype=torch.float32)
        self.conv = torch.ones(2)
        self.state = torch.full((2,), 2.0)


class TinyTower(torch.nn.Module):
    cache_capacity = 8

    def __init__(self):
        super().__init__()
        self.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
        self.kv = torch.arange(8, dtype=torch.float32)
        self.main_candidates = torch.tensor([0, 5, 6])
        self.candidate_rows = torch.tensor([0, 5, 6])
        self.calls = []

    def prepare_sampling(self):
        pass

    def sample_frame(self, logits, hidden, position, frame, history, kv, candidates, rows,
                     capacity, temperature, top_k, top_p):
        self.calls.append({
            "logits": logits.clone(),
            "hidden": hidden.clone(),
            "position": position.clone(),
            "frame": frame.clone(),
            "history": history.clone(),
            "top_k": top_k.clone(),
            "capacity": capacity,
        })
        main = torch.zeros((frame.shape[0],), dtype=torch.long, device=frame.device)
        sampled = torch.stack((main, frame[:, 1] + 1), dim=-1)
        kv.index_copy_(0, position, main.to(kv.dtype))
        torch.rand(())  # Verify that capture warmup restores the request RNG.
        finite = torch.isfinite(logits).all()
        return sampled, finite, hidden + 1, torch.zeros_like(main, dtype=torch.bool)


class FakeStorage:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class FakeCudaTensor:
    device = torch.device("cuda:0")
    shape = (2,)
    dtype = torch.float32
    _version = 0

    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer

    def untyped_storage(self):
        return FakeStorage(self.pointer - 8)

    def stride(self):
        return (1,)

    def storage_offset(self):
        return 0


def make_manager(roots):
    def ready_to_check_mem(self):
        return True

    def ensure_model_loaded(self, model_id):
        return model_id in self.active_models_ids

    Manager = type(
        "offload",
        (),
        {
            "__module__": "mmgp.offload",
            "ready_to_check_mem": ready_to_check_mem,
            "ensure_model_loaded": ensure_model_loaded,
        },
    )
    manager = Manager()
    manager.models = dict(roots)
    manager.active_models_ids = list(roots)
    manager.blocks_of_modules = {name: [] for name in roots}
    manager.preloaded_blocks_per_model = {name: {} for name in roots}
    manager.loaded_blocks = {name: None for name in roots}
    manager.auto_preload = None
    manager.auto_preload_requested = False
    for root in roots.values():
        root._mm_manager = manager
    return manager


class FishTowerGraphTests(unittest.TestCase):
    def make_inputs(self):
        return (
            torch.tensor([[2.0, 3.0]]),
            torch.tensor([[1.0, 4.0, 2.0, 0.0, 3.0, 5.0]]),
            torch.tensor([[8, 10]]),
            torch.tensor([11, 12, 13, 14]),
        )

    def make_models(self):
        return TinyBackbone(), TinyTower()

    def mocked_capture(self, *, capture_failure=None, abort=None, free_bytes=8 * 1024**3):
        backbone, tower = self.make_models()
        hidden, logits, frame, history = self.make_inputs()
        graph = Mock()
        stream = Mock()
        capture = Mock(side_effect=capture_failure) if capture_failure else Mock(return_value=nullcontext())
        expected_caches = tuple(
            value.clone() for value in (backbone.kv, backbone.conv, backbone.state, tower.kv)
        )
        cache_pointers = tuple(
            value.data_ptr() for value in (backbone.kv, backbone.conv, backbone.state, tower.kv)
        )
        rng = torch.get_rng_state().clone()
        patches = (
            patch.object(graph_module, "_capture_eligibility", return_value=(True, None, ("stable",))),
            patch.object(graph_module, "_tower_weight_signature", return_value=("stable",)),
            patch.object(graph_module, "resident_capture_scope", return_value=nullcontext()),
            patch.object(torch.cuda, "mem_get_info", return_value=(free_bytes, free_bytes + 1)),
            patch.object(torch.cuda, "Stream", return_value=stream),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "CUDAGraph", return_value=graph),
            patch.object(torch.cuda, "graph", capture),
        )
        entered = []
        try:
            for item in patches:
                entered.append(item)
                item.__enter__()
            decoder, reason = TowerFrameDecoder.try_create(
                backbone, tower, hidden, logits, frame, history, 3, 0.8, 0, 0.95,
                engine="vllm", check_abort=abort,
            )
        finally:
            for item in reversed(entered):
                item.__exit__(None, None, None)
        for cache, saved, pointer in zip(
            (backbone.kv, backbone.conv, backbone.state, tower.kv), expected_caches, cache_pointers
        ):
            torch.testing.assert_close(cache, saved)
            self.assertEqual(cache.data_ptr(), pointer)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        return decoder, reason, graph, backbone, tower, (hidden, logits, frame, history)

    def test_tower_signature_tracks_physical_dense_weights_only(self):
        tower = torch.nn.Module()
        tower._buffers["physical_weight"] = FakeCudaTensor(100)
        tower.kv = FakeCudaTensor(200)  # Runtime cache is not a registered buffer.
        signature = graph_module._tower_weight_signature(tower, torch.device("cuda:0"))
        self.assertEqual(signature[0], "fish-tower-physical-weights-v1")
        self.assertEqual(len(signature[1]), 1)
        tower._buffers["physical_weight"].pointer = 101
        self.assertNotEqual(signature, graph_module._tower_weight_signature(tower, torch.device("cuda:0")))

    def test_capture_requires_a_shared_known_mmgp_state_for_both_roots(self):
        backbone, tower = torch.nn.Module(), torch.nn.Module()
        tower._buffers["weight"] = FakeCudaTensor(100)
        tower.kv = FakeCudaTensor(200)
        make_manager({"backbone": backbone, "tower": tower})
        with patch.object(graph_module, "_tower_weight_signature", return_value=("stable",)):
            eligible, reason, _ = graph_module._capture_eligibility(
                backbone, tower, torch.device("cuda:0"), "vllm"
            )
        self.assertTrue(eligible, reason)

        make_manager({"backbone": backbone})
        make_manager({"tower": tower})
        with patch.object(graph_module, "_tower_weight_signature", return_value=("stable",)):
            eligible, reason, _ = graph_module._capture_eligibility(
                backbone, tower, torch.device("cuda:0"), "vllm"
            )
        self.assertFalse(eligible)
        self.assertIn("resident together", reason)

    def test_unknown_manager_fails_closed(self):
        backbone, tower = torch.nn.Module(), torch.nn.Module()
        backbone._mm_manager = object()
        with patch.object(graph_module, "_tower_weight_signature", return_value=("stable",)):
            eligible, reason, _ = graph_module._capture_eligibility(
                backbone, tower, torch.device("cuda:0"), "vllm"
            )
        self.assertFalse(eligible)
        self.assertIn("unknown MMGP manager", reason)

    def test_successful_capture_restores_all_caches_and_rng_and_steps_on_fresh_inputs(self):
        decoder, reason, graph, _, tower, inputs = self.mocked_capture()
        self.assertIsNone(reason)
        self.assertIsNotNone(decoder)
        self.assertEqual(decoder.position.tolist(), [3])
        tower.calls.clear()
        graph.replay.side_effect = decoder._sample_frame

        hidden, logits, frame, history = inputs
        sampled, finite, extra, semantic = decoder.step(3, hidden, logits, frame, history)
        self.assertEqual(sampled.tolist(), [[0, 11]])
        self.assertTrue(finite.item())
        torch.testing.assert_close(extra, hidden + 1)
        self.assertEqual(semantic.tolist(), [False])
        self.assertEqual(decoder.history.tolist(), [-1] * 12 + [11, 12, 13, 14])
        self.assertEqual(tower.calls[-1]["top_k"].item(), 0)
        self.assertEqual(tower.calls[-1]["capacity"], tower.cache_capacity)

        next_hidden, next_logits, next_frame = hidden + 4, logits.flip(-1), frame + 2
        next_history = torch.arange(30, dtype=torch.long)
        sampled, finite, _, _ = decoder.step(4, next_hidden, next_logits, next_frame, next_history)
        self.assertEqual(sampled.tolist(), [[0, 13]])
        self.assertTrue(finite.item())
        self.assertEqual(tower.calls[-1]["hidden"].tolist(), next_hidden.tolist())
        self.assertEqual(tower.calls[-1]["logits"].tolist(), next_logits.tolist())
        self.assertEqual(tower.calls[-1]["frame"].tolist(), next_frame.tolist())
        self.assertEqual(tower.calls[-1]["history"].tolist(), list(range(14, 30)))
        self.assertEqual(tower.calls[-1]["position"].tolist(), [4])

    def test_nonfinite_sampling_result_is_returned_for_the_caller_to_reject(self):
        decoder, reason, graph, _, _, inputs = self.mocked_capture()
        self.assertIsNone(reason)
        graph.replay.side_effect = decoder._sample_frame
        hidden, logits, frame, history = inputs
        logits = logits.clone()
        logits[0, 0] = torch.nan
        _, finite, _, _ = decoder.step(3, hidden, logits, frame, history)
        self.assertFalse(finite.item())

    def test_unsupported_capture_rolls_back_before_any_step_and_falls_back(self):
        decoder, reason, _, backbone, tower, _ = self.mocked_capture(
            capture_failure=RuntimeError("capture unsupported")
        )
        self.assertIsNone(decoder)
        self.assertIn("capture unsupported", reason)
        self.assertEqual(tower.kv.tolist(), list(range(8)))
        self.assertEqual(backbone.kv.tolist(), list(range(8)))

    def test_capture_cancellation_propagates_after_transactional_warmup(self):
        backbone, tower = self.make_models()
        caches = (backbone.kv, backbone.conv, backbone.state, tower.kv)
        expected = tuple(value.clone() for value in caches)
        pointers = tuple(value.data_ptr() for value in caches)
        rng = torch.get_rng_state().clone()
        abort = Mock(side_effect=[None, GenerationCancelled()])
        with (
            patch.object(self, "make_models", return_value=(backbone, tower)),
            self.assertRaises(GenerationCancelled),
        ):
            self.mocked_capture(abort=abort)
        for cache, saved, pointer in zip(caches, expected, pointers):
            torch.testing.assert_close(cache, saved)
            self.assertEqual(cache.data_ptr(), pointer)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_memory_guard_avoids_capture_without_enough_rollback_headroom(self):
        decoder, reason, graph, _, _, _ = self.mocked_capture(free_bytes=1)
        self.assertIsNone(decoder)
        self.assertIn("insufficient free VRAM", reason)
        graph.replay.assert_not_called()

    def test_wrong_cache_or_position_and_released_graph_refuse_replay(self):
        decoder, reason, graph, _, tower, inputs = self.mocked_capture()
        self.assertIsNone(reason)
        hidden, logits, frame, history = inputs
        with self.assertRaisesRegex(RuntimeError, "position"):
            decoder.step(4, hidden, logits, frame, history)
        tower.kv = tower.kv.clone()
        with self.assertRaisesRegex(RuntimeError, "cache changed"):
            decoder.step(3, hidden, logits, frame, history)
        graph.replay.assert_not_called()
        decoder.close()
        with self.assertRaisesRegex(RuntimeError, "released"):
            decoder.step(3, hidden, logits, frame, history)

    def test_graph_opt_out_skips_capture(self):
        backbone, tower = self.make_models()
        hidden, logits, frame, history = self.make_inputs()
        with (
            patch.dict(os.environ, {"MAESTRO_FISH_CUDA_GRAPH": "0"}),
            patch.object(torch.cuda, "CUDAGraph", side_effect=AssertionError("unexpected capture")),
        ):
            decoder, reason = TowerFrameDecoder.try_create(
                backbone, tower, hidden, logits, frame, history, 3, 0.8, 0, 0.95,
                engine="vllm",
            )
        self.assertIsNone(decoder)
        self.assertIn("disabled", reason)

    def test_failed_state_restore_is_fatal(self):
        # The restore context's explicit fatal type must pass through the
        # ordinary unsupported-capture fallback handler.
        backbone, tower = self.make_models()
        hidden, logits, frame, history = self.make_inputs()
        with (
            patch.object(graph_module, "_capture_eligibility", return_value=(True, None, ("stable",))),
            patch.object(graph_module, "_tower_weight_signature", return_value=("stable",)),
            patch.object(graph_module, "resident_capture_scope", return_value=nullcontext()),
            patch.object(torch.cuda, "mem_get_info", return_value=(8 * 1024**3, 8 * 1024**3)),
            patch.object(graph_module, "preserve_decode_state", side_effect=GraphStateRestoreError("restore failed")),
            self.assertRaises(GraphStateRestoreError),
        ):
            TowerFrameDecoder.try_create(
                backbone, tower, hidden, logits, frame, history, 3, 0.8, 0, 0.95,
                engine="vllm",
            )


if __name__ == "__main__":
    unittest.main()
