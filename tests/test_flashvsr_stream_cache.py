"""CPU-only checks for FlashVSR's bounded grouped stream-cache contract."""
from __future__ import annotations

import ast
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from postprocessing.flashvsr.stream_cache import StreamCache


FLASHVSR_DIR = ROOT / "app" / "postprocessing" / "flashvsr"


def _load_flashvsr_contract_nodes(*names, namespace=None):
    source = (FLASHVSR_DIR / "wan_video_dit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing FlashVSR contract classes: {names}")
    module_namespace = {"torch": torch, "nn": torch.nn}
    if namespace:
        module_namespace.update(namespace)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(FLASHVSR_DIR / "wan_video_dit.py"),
            "exec",
        ),
        module_namespace,
    )
    return module_namespace


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
        self.assertFalse(cache.has_prefix(0))
        self.assertEqual(cache.storage_device, torch.device("cpu"))
        keys = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
        values = keys + 10
        cache.commit(0, keys, values, block_tokens=4)
        self.assertTrue(cache.has_prefix(0))
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

    def test_v11_attention_accepts_two_latent_continuation_with_stream_cache(self):
        class StopAfterShapeValidation(RuntimeError):
            pass

        def stop_partition(*_args, **_kwargs):
            raise StopAfterShapeValidation("shape accepted")

        module_namespace = _load_flashvsr_contract_nodes(
            "SelfAttention",
            namespace={
                "WindowPartition3D": SimpleNamespace(partition=stop_partition),
            },
        )
        attention_type = module_namespace["SelfAttention"]
        attention = attention_type.__new__(attention_type)
        torch.nn.Module.__init__(attention)

        def run_until_partition(frames, cached, allow_short_start=False):
            sample = torch.zeros((1, frames * 8 * 8, 4))
            cache = SimpleNamespace(has_prefix=lambda _block: cached)
            attention.forward(
                sample,
                freqs=None,
                f=frames,
                h=8,
                w=8,
                is_stream=True,
                block_id=0,
                stream_cache=cache,
                allow_short_start=allow_short_start,
            )

        with self.assertRaisesRegex(StopAfterShapeValidation, "shape accepted"):
            run_until_partition(frames=6, cached=False)
        with self.assertRaisesRegex(StopAfterShapeValidation, "shape accepted"):
            run_until_partition(frames=2, cached=True)
        with self.assertRaisesRegex(AssertionError, "startup chunk contains six"):
            run_until_partition(frames=2, cached=False)
        with self.assertRaisesRegex(StopAfterShapeValidation, "shape accepted"):
            run_until_partition(frames=2, cached=False, allow_short_start=True)

    def test_v11_continue_cache_reuses_the_output_tail(self):
        runtime_source = (FLASHVSR_DIR / "runtime.py").read_text(encoding="utf-8")
        runtime_tree = ast.parse(runtime_source)
        names = {"_apply_continue_cache", "_make_continue_cache"}
        functions = [
            node for node in runtime_tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual({node.name for node in functions}, names)
        namespace = {"torch": torch, "Any": Any, "FLASHVSR_CONTINUE_CACHE_FRAMES": 11}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=functions, type_ignores=[])
                ),
                str(FLASHVSR_DIR / "runtime.py"),
                "exec",
            ),
            namespace,
        )

        frames = torch.zeros((3, 20, 2, 2), dtype=torch.uint8)
        tail = torch.full((3, 11, 2, 2), 17, dtype=torch.uint8)
        continued = namespace["_apply_continue_cache"](
            frames, {"tail_frames": tail}
        )
        self.assertTrue(torch.all(continued[:, :11] == 17))
        self.assertTrue(torch.all(continued[:, 11:] == 0))

        next_cache = namespace["_make_continue_cache"](
            continued, 2.0, "full", overlap_frames=11
        )
        self.assertTrue(torch.equal(next_cache["tail_frames"], continued[:, -11:]))


if __name__ == "__main__":
    unittest.main()
