"""CPU-only checks for FlashVSR's bounded grouped stream-cache contract."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from postprocessing.flashvsr.stream_cache import StreamCache


class _FakeRing:
    def __init__(self, modules, name):
        self.modules = modules
        self.name = name
        self.states = {}
        self.released = False

    def store(self, block, tensors, current=False):
        if block not in self.states:
            self.states[block] = [tensor.clone() for tensor in tensors]
        else:
            for target, source in zip(self.states[block], tensors):
                target.copy_(source)
        return self.states[block]

    def release(self):
        self.released = True
        self.states.clear()


class FlashVSRStreamCacheTests(unittest.TestCase):
    def test_legacy_mmgp_without_state_ring_uses_bounded_host_cache(self):
        real_import = __import__

        def legacy_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mmgp" and "runtime" in fromlist:
                raise ImportError("legacy MMGP 4.0 does not provide mmgp.runtime")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=legacy_import), patch("torch.cuda.is_available", return_value=True):
            cache = StreamCache(
                [object()], [2], window_tokens=8, heads=1, head_dim=2,
                dtype=torch.float32, device=torch.device("cuda"),
            )

        self.assertFalse(cache.rings)
        self.assertEqual(cache.storage_device, torch.device("cpu"))
        keys = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
        values = keys + 10
        cache.commit(0, keys, values, block_tokens=4)
        prefix = cache.prefix(0, 4, 2, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.equal(prefix[0], keys))
        self.assertTrue(torch.equal(prefix[1], values))
        self.assertEqual(cache.cpu_handles[0][0].shape, (16, 1, 2))
        cache.release()

    def test_grouped_cache_grows_logical_rows_within_fixed_capacity(self):
        blocks = [object(), object()]
        rings = []

        def make_ring(modules, name):
            ring = _FakeRing(modules, name)
            rings.append(ring)
            return ring

        cache = StreamCache(
            blocks, [1, 3], window_tokens=8, heads=2, head_dim=2,
            dtype=torch.float32, device=torch.device("cpu"), ring_factory=make_ring,
        )
        self.assertTrue(cache.storing)
        self.assertEqual([len(ring.modules) for ring in rings], [1, 1])

        local_k = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
        local_v = local_k + 100
        cache.commit(0, local_k, local_v, block_tokens=4)
        local_prefix = cache.prefix(0, 4, 4, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(local_prefix[0].shape), (2, 4, 4))
        self.assertTrue(torch.equal(local_prefix[0], local_k))

        first_turbo_k = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
        first_turbo_v = first_turbo_k + 200
        cache.commit(1, first_turbo_k, first_turbo_v, block_tokens=4)
        first_prefix = cache.prefix(1, 4, 4, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(first_prefix[0].shape), (4, 4, 4))
        self.assertTrue(torch.equal(first_prefix[1], first_turbo_v))

        full_turbo_k = torch.arange(6 * 4 * 4, dtype=torch.float32).reshape(6, 4, 4)
        full_turbo_v = full_turbo_k + 300
        cache.commit(1, full_turbo_k, full_turbo_v, block_tokens=4)
        full_prefix = cache.prefix(1, 4, 4, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(full_prefix[0].shape), (6, 4, 4))
        self.assertTrue(torch.equal(full_prefix[0], full_turbo_k))
        self.assertFalse(cache.storing)

        cache.release()
        self.assertTrue(all(ring.released for ring in rings))
        self.assertFalse(cache.lengths)


if __name__ == "__main__":
    unittest.main()
