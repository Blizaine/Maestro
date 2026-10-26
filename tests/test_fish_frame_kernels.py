"""CPU-only dispatch contracts for Fish's opt-in single-frame Triton path."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_expert_triton_with_jit_stubs():
    """Load dispatch code without requiring Triton's native GPU runtime."""

    fake_language = types.ModuleType("triton.language")
    fake_language.constexpr = type("constexpr", (), {})
    fake_triton = types.ModuleType("triton")
    fake_triton.jit = lambda function: function
    fake_triton.language = fake_language
    old_triton = sys.modules.get("triton")
    old_language = sys.modules.get("triton.language")
    sys.modules["triton"] = fake_triton
    sys.modules["triton.language"] = fake_language
    try:
        spec = importlib.util.spec_from_file_location(
            "_fish_frame_expert_triton",
            ROOT / "app" / "models" / "TTS" / "fish_audio" / "expert_triton.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_triton is None:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = old_triton
        if old_language is None:
            sys.modules.pop("triton.language", None)
        else:
            sys.modules["triton.language"] = old_language


class RecordingKernel:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __getitem__(self, grid):
        def launch(*args):
            self.calls.append((self.name, grid, args))

        return launch


class FishFrameKernelDispatchTests(unittest.TestCase):
    def setUp(self):
        self.module = load_expert_triton_with_jit_stubs()
        self.calls = []
        gate = RecordingKernel("gate", self.calls)
        serial_down = RecordingKernel("serial_down", self.calls)
        parallel_down = RecordingKernel("parallel_down", self.calls)
        reduce = RecordingKernel("reduce", self.calls)
        for kind in ("fp8_block", "int4_group"):
            self.module._KERNELS[kind] = (gate, serial_down)
            self.module._PARALLEL_DOWN_KERNELS[kind] = parallel_down
        self.module._ordered_topk_reduce = reduce

    def inputs(self, tokens):
        hidden, intermediate, topk = 32, 16, 2
        x = torch.ones((tokens, hidden), dtype=torch.float32)
        ids = torch.zeros((tokens, topk), dtype=torch.int64)
        probabilities = torch.full((tokens, topk), 1.0 / topk)
        pointers = torch.zeros((1,), dtype=torch.int64)
        return x, ids, probabilities, pointers, hidden, intermediate, topk

    def test_single_frame_override_and_parallel_down_reuse_all_scratch(self):
        x, ids, probabilities, pointers, hidden, intermediate, topk = self.inputs(1)
        out = torch.empty((1, hidden), dtype=torch.float32)
        mid = torch.empty((topk, intermediate), dtype=torch.float32)
        down_scratch = torch.empty((topk, hidden), dtype=torch.float32)

        actual = self.module.forward(
            "fp8_block",
            x,
            ids,
            probabilities,
            pointers,
            intermediate,
            8,
            out=out,
            mid=mid,
            rows=8,
            single_token_rows=4,
            parallel_single_token_down=True,
            down_scratch=down_scratch,
        )

        self.assertIs(actual, out)
        self.assertEqual(
            [(name, grid) for name, grid, _ in self.calls],
            [("gate", (2, 4)), ("parallel_down", (2, 8)), ("reduce", (1, 8))],
        )
        self.assertIs(self.calls[0][2][3], mid)
        self.assertIs(self.calls[1][2][4], down_scratch)
        self.assertIs(self.calls[2][2][1], out)

    def test_multitoken_call_keeps_default_tile_and_serial_down(self):
        x, ids, probabilities, pointers, hidden, intermediate, _ = self.inputs(2)
        actual = self.module.forward(
            "int4_group",
            x,
            ids,
            probabilities,
            pointers,
            intermediate,
            8,
            rows=8,
            single_token_rows=4,
            parallel_single_token_down=True,
        )

        self.assertEqual(actual.shape, (2, hidden))
        self.assertEqual(
            [(name, grid) for name, grid, _ in self.calls],
            [("gate", (4, 2)), ("serial_down", (2, 4))],
        )

    def test_default_single_frame_call_preserves_existing_kernel_path(self):
        x, ids, probabilities, pointers, _hidden, intermediate, _ = self.inputs(1)

        self.module.forward(
            "fp8_block", x, ids, probabilities, pointers, intermediate, 8
        )

        self.assertEqual(
            [(name, grid) for name, grid, _ in self.calls],
            [("gate", (2, 2)), ("serial_down", (1, 4))],
        )


if __name__ == "__main__":
    unittest.main()
