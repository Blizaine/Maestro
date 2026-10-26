"""Focused CPU-only tests for Fish's trusted top-k expert route."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

# Import the Fish modules without importing the wider optional TTS registry.
import models

tts_package = sys.modules.get("models.TTS", getattr(models, "TTS", None))
if tts_package is None:
    tts_package = types.ModuleType("models.TTS")
    tts_package.__path__ = [str(ROOT / "app" / "models" / "TTS")]
sys.modules["models.TTS"] = tts_package
models.TTS = tts_package

from mmgp.expert_compute import SwiGLUExperts
from models.TTS.fish_audio import backbone, native_cpu
from models.TTS.fish_audio.expert_formats import (
    EXPERT_FORMATS,
    FishSwiGLUExperts,
    Int4GroupExperts,
)


class FishRouteValidationTests(unittest.TestCase):
    def make_gpu_bank(self):
        bank = FishSwiGLUExperts(3, 4, 4, formats=EXPERT_FORMATS, use_triton=True)
        bank.format = Int4GroupExperts()
        # These markers exercise the GPU dispatch without allocating CUDA memory.
        bank._fish_gpu_tensors = (object(),)
        bank._fish_gpu_pointers = object()
        bank._fish_gpu_device = torch.device("cuda:0")
        return bank

    def run_fake_gpu(self, bank, forward, call):
        module_name = "models.TTS.fish_audio.expert_triton"
        fake_triton = types.ModuleType(module_name)
        fake_triton.forward = forward
        fish_package = sys.modules["models.TTS.fish_audio"]
        with (
            patch.dict(sys.modules, {module_name: fake_triton}),
            patch.object(fish_package, "expert_triton", fake_triton, create=True),
            patch.object(
                FishSwiGLUExperts,
                "_canonical_cuda_device",
                return_value=torch.device("cuda:0"),
            ),
            patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True),
        ):
            return call()

    def test_direct_gpu_call_still_rejects_out_of_range_ids_by_default(self):
        bank = self.make_gpu_bank()
        triton_forward = Mock()

        def call(invalid_id):
            return bank(
                torch.ones((1, 4)),
                torch.tensor([[invalid_id]], dtype=torch.long),
                torch.ones((1, 1)),
            )

        for invalid_id in (-1, bank.num_experts):
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaisesRegex(ValueError, "outside the loaded bank"):
                    self.run_fake_gpu(
                        bank,
                        triton_forward,
                        lambda invalid_id=invalid_id: call(invalid_id),
                    )
        triton_forward.assert_not_called()

    def test_trusted_flag_keeps_ids_and_weights_shape_validation(self):
        bank = self.make_gpu_bank()
        triton_forward = Mock()

        def call():
            return bank(
                torch.ones((1, 4)),
                torch.tensor([[0]], dtype=torch.long),
                torch.ones((1, 2)),
                check_route_bounds=False,
            )

        with self.assertRaisesRegex(ValueError, "same shape"):
            self.run_fake_gpu(bank, triton_forward, call)
        triton_forward.assert_not_called()

    def test_trusted_gpu_route_skips_min_and_max_scalar_reads(self):
        bank = self.make_gpu_bank()
        triton_forward = Mock(return_value=torch.zeros((1, 4)))

        def call():
            with (
                patch.object(
                    torch.Tensor,
                    "min",
                    side_effect=AssertionError("trusted route read IDs.min()"),
                ),
                patch.object(
                    torch.Tensor,
                    "max",
                    side_effect=AssertionError("trusted route read IDs.max()"),
                ),
            ):
                return bank(
                    torch.ones((1, 4)),
                    torch.tensor([[2]], dtype=torch.long),
                    torch.ones((1, 1)),
                    check_route_bounds=False,
                )

        actual = self.run_fake_gpu(bank, triton_forward, call)
        self.assertEqual(actual.shape, (1, 4))
        triton_forward.assert_called_once()

    def test_backbone_forward_and_decode_use_module_call_with_trusted_flag(self):
        class Mixer(nn.Module):
            def forward(self, x, context, cache):
                return torch.zeros_like(x)

        class Experts(nn.Module):
            def forward(self, x, ids, weights, *, check_route_bounds=True):
                return torch.zeros_like(x)

        class Moe(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = Experts()

            def route(self, x):
                flat = x.reshape(-1, x.shape[-1])
                ids = torch.zeros((flat.shape[0], 1), dtype=torch.long)
                weights = torch.ones((flat.shape[0], 1))
                return flat, ids, weights, torch.zeros_like(flat)

        layer = backbone.DecoderLayer.__new__(backbone.DecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_type = "linear_attention"
        layer.linear_attn = Mixer()
        layer.input_layernorm = nn.Identity()
        layer.post_attention_layernorm = nn.Identity()
        layer.mlp = Moe()

        hook_calls = []
        layer.mlp.experts.register_forward_pre_hook(
            lambda module, args, kwargs: hook_calls.append(kwargs["check_route_bounds"]),
            with_kwargs=True,
        )
        x = torch.ones((1, 1, 4))

        layer(x, None, None)
        layer.decode(x, None, None)

        self.assertEqual(hook_calls, [False, False])

    def test_cpu_path_keeps_bounds_validation_and_reference_math(self):
        fmt = Int4GroupExperts()
        declarations = fmt.tensors(1, 128, 128)
        torch.manual_seed(2718)
        state = {
            "w13_weight": torch.randint(
                0, 256, declarations["w13_weight"][0], dtype=torch.uint8
            ),
            "w2_weight": torch.randint(
                0, 256, declarations["w2_weight"][0], dtype=torch.uint8
            ),
            "w13_scales": torch.rand(
                declarations["w13_scales"][0], dtype=torch.float16
            )
            * 0.02,
            "w2_scales": torch.rand(
                declarations["w2_scales"][0], dtype=torch.float16
            )
            * 0.02,
        }
        with torch.device("meta"):
            bank = FishSwiGLUExperts(1, 128, 128, formats=EXPERT_FORMATS)
        bank.load_state_dict(state, assign=True)

        x = torch.randn((2, 128))
        ids = torch.zeros((2, 1), dtype=torch.long)
        weights = torch.tensor([[0.6], [0.4]])
        expected = SwiGLUExperts.forward(bank, x, ids, weights)
        with patch.object(native_cpu, "forward", return_value=None):
            actual = bank(x, ids, weights, check_route_bounds=False)
            with self.assertRaisesRegex(ValueError, "outside the loaded bank"):
                bank(x, torch.full_like(ids, 1), weights, check_route_bounds=False)

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
