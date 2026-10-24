"""CPU contracts for bounded, exact selected-expert GPU cache planning."""
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

from models.TTS.fish_audio.expert_cache import (
    ExpertSlotPolicy,
    PackedExpertCache,
    cache_slot_count,
)


class FishExpertCacheTests(unittest.TestCase):
    def test_selected_route_is_protected_while_replacing_lru_slots(self):
        policy = ExpertSlotPolicy(3, 16)
        self.assertEqual(policy.route([1, 2, 3]), ([(1, 0, None), (2, 1, None), (3, 2, None)], 0))
        # Expert 1 is oldest but must be kept even though it occurs after the
        # new expert in this route. Evict 2, not the first selected expert.
        misses, hits = policy.route([4, 1, 3])
        self.assertEqual(misses, [(4, 1, 2)])
        self.assertEqual(hits, 2)
        self.assertEqual(set(policy.resident), {1, 3, 4})
        self.assertEqual(set(policy.resident.values()), {0, 1, 2})

    def test_duplicate_routes_share_slots_and_complete_hits_need_no_copy(self):
        policy = ExpertSlotPolicy(2, 4)
        self.assertEqual(len(policy.route([2, 2, 1])[0]), 2)
        self.assertEqual(policy.route([1, 2, 1]), ([], 2))

    def test_invalid_routes_do_not_change_cache_contents(self):
        policy = ExpertSlotPolicy(2, 4)
        policy.route([0, 1])
        before = dict(policy.resident)
        for route in ([-1], [4], [1, 2, 3]):
            with self.assertRaises(ValueError):
                policy.route(route)
            self.assertEqual(dict(policy.resident), before)

    def test_cache_respects_headroom_and_supports_whole_topk_or_none(self):
        gib = 1024**3
        # Production bank shape: 40 x 256 experts, 16 GiB complete bank.
        sizes, counts = [16 * gib // 40] * 40, [256] * 40
        self.assertEqual(cache_slot_count(sizes, counts, gib, 8 * gib), 0)
        self.assertEqual(cache_slot_count(sizes, counts, int(1.7 * gib), 8 * gib), 0)
        slots = cache_slot_count(sizes, counts, 4 * gib, 8 * gib)
        self.assertGreaterEqual(slots, 8)
        self.assertLessEqual(slots * sum(size // 256 for size in sizes), 4 * gib - 1536 * 1024**2)
        self.assertEqual(cache_slot_count(sizes, counts, 20 * gib, 24 * gib), 64)
        self.assertEqual(cache_slot_count(sizes, counts, 4 * gib, 8 * gib,
                                          host_free_bytes=int(0.6 * gib)), 0)
        self.assertEqual(cache_slot_count(sizes, counts, 4 * gib, 8 * gib,
                                          host_free_bytes=2 * gib), slots)

    def test_empty_or_unknown_bank_metadata_disables_placement(self):
        self.assertEqual(cache_slot_count([], [], 20, 24), 0)
        self.assertEqual(cache_slot_count([100], [0], 20, 24), 0)

    def test_staging_and_pointer_updates_keep_cpu_indices_under_another_default_device(self):
        # Exercise staging/table bookkeeping on CPU. Native CUDA parity is a
        # separate probe; this checks that MMGP's default-device context cannot
        # silently move source indices away from the CPU checkpoint storage.
        cache = object.__new__(PackedExpertCache)
        cache.policy = ExpertSlotPolicy(2, 8)
        cache.sources = tuple(torch.arange(32).reshape(8, 4) + offset for offset in range(4))
        saved = [value.clone() for value in cache.sources]
        cache.slots = tuple(torch.empty((2, 4), dtype=torch.long) for _ in range(4))
        cache.staging = tuple(torch.empty_like(value) for value in cache.slots)
        cache.host_pointers = torch.zeros((8, 4), dtype=torch.long)
        cache.pointers = torch.zeros_like(cache.host_pointers)
        cache.topk = 2
        cache.hits = cache.misses = cache.transferred_bytes = cache.frames = 0
        first, second = torch.tensor([[0, 1]]), torch.tensor([[2, 1]])
        cache.prepare(first)
        with torch.device("meta"):
            cache.prepare(second)
        self.assertEqual(cache.policy.resident, {2: 0, 1: 1})
        self.assertEqual(cache.pointers[0].tolist(), [0, 0, 0, 0])
        for index, (source, slots) in enumerate(zip(cache.sources, cache.slots)):
            torch.testing.assert_close(slots[0], source[2])
            torch.testing.assert_close(slots[1], source[1])
            self.assertEqual(cache.pointers[2, index].item(), slots[0].data_ptr())
            torch.testing.assert_close(source, saved[index])
        self.assertEqual((cache.hits, cache.misses), (1, 3))
        self.assertEqual(cache.transferred_bytes, 3 * 4 * 4 * 8)
        with self.assertRaises(ValueError):
            cache.prepare(torch.empty((1, 0), dtype=torch.long))

    def test_invalid_later_bank_releases_all_partial_cache_allocations(self):
        from models.TTS.fish_audio.pipeline import FishAudioPipeline
        pipeline = object.__new__(FishAudioPipeline)
        pipeline.engine = "vllm"
        pipeline.config = types.SimpleNamespace(text=types.SimpleNamespace(num_experts_per_tok=2))
        first, second = Mock(expert_nbytes=1024, num_experts=8), Mock(expert_nbytes=1024, num_experts=8)
        second.place_cached_experts.side_effect = ValueError("unsupported bank")
        pipeline._expert_banks = Mock(return_value=[first, second])
        with (
            patch.object(torch.cuda, "mem_get_info", return_value=(8 * 1024**3, 12 * 1024**3)),
            patch.object(torch.cuda, "empty_cache"),
            patch("models.TTS.fish_audio.pipeline.psutil.virtual_memory", return_value=Mock(available=8 * 1024**3)),
        ):
            pipeline._prepare_partial_expert_cache(torch.device("cuda:0"))
        first.release_decode_cache.assert_called_once()
        second.release_decode_cache.assert_called_once()
        self.assertIsNone(pipeline.last_expert_cache_report)

    def test_packed_staging_never_requires_cpu_float8_gather(self):
        cache = object.__new__(PackedExpertCache)
        cache.policy = ExpertSlotPolicy(2, 8)
        cache.sources = tuple(torch.arange(32, dtype=torch.float32).reshape(8, 4).to(dtype)
                              for dtype in (torch.float8_e4m3fn, torch.uint8, torch.float16, torch.float32))
        cache.slots = tuple(torch.empty((2, 4), dtype=value.dtype) for value in cache.sources)
        cache.staging = tuple(torch.empty_like(value) for value in cache.slots)
        cache.host_pointers = torch.zeros((8, 4), dtype=torch.long)
        cache.pointers = torch.zeros_like(cache.host_pointers)
        cache.topk = 2
        cache.hits = cache.misses = cache.transferred_bytes = cache.frames = 0
        gather = torch.index_select
        seen = []
        def byte_gather(source, dimension, indices, *, out):
            seen.append(source.dtype)
            self.assertEqual(source.dtype, torch.uint8)
            return gather(source, dimension, indices, out=out)
        with patch.object(torch, "index_select", side_effect=byte_gather):
            cache.prepare(torch.tensor([[2, 0]]))
        self.assertEqual(len(seen), 4)
        for source, slots in zip(cache.sources, cache.slots):
            torch.testing.assert_close(slots[0].view(torch.uint8), source[2].view(torch.uint8))
            torch.testing.assert_close(slots[1].view(torch.uint8), source[0].view(torch.uint8))


if __name__ == "__main__":
    unittest.main()
