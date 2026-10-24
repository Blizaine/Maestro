"""CPU-only tests for Fish Triton format/device capability gating."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch  # noqa: I001


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import models

tts_package = sys.modules.get("models.TTS", getattr(models, "TTS", None))
if tts_package is None:
    tts_package = types.ModuleType("models.TTS")
    tts_package.__path__ = [str(ROOT / "app" / "models" / "TTS")]
sys.modules["models.TTS"] = tts_package
models.TTS = tts_package

from models.TTS.fish_audio.expert_formats import (
    EXPERT_FORMATS,
    FishSwiGLUExperts,
    Fp8BlockExperts,
    Int4GroupExperts,
)


def make_bank(kind: str) -> FishSwiGLUExperts:
    expert_format = Fp8BlockExperts() if kind == "fp8_block" else Int4GroupExperts()
    declarations = expert_format.tensors(2, 128, 128)
    if kind == "fp8_block":
        state = {
            "w13_weight": torch.ones(
                declarations["w13_weight"][0], dtype=torch.float8_e4m3fn
            ),
            "w2_weight": torch.ones(
                declarations["w2_weight"][0], dtype=torch.float8_e4m3fn
            ),
            "w13_scale_inv": torch.full(
                declarations["w13_scale_inv"][0], 0.01, dtype=torch.float32
            ),
            "w2_scale_inv": torch.full(
                declarations["w2_scale_inv"][0], 0.01, dtype=torch.float32
            ),
        }
    else:
        state = {
            "w13_weight": torch.full(
                declarations["w13_weight"][0], 0x88, dtype=torch.uint8
            ),
            "w2_weight": torch.full(
                declarations["w2_weight"][0], 0x88, dtype=torch.uint8
            ),
            "w13_scales": torch.full(
                declarations["w13_scales"][0], 0.01, dtype=torch.float16
            ),
            "w2_scales": torch.full(
                declarations["w2_scales"][0], 0.01, dtype=torch.float16
            ),
        }
    with torch.device("meta"):
        bank = FishSwiGLUExperts(
            2, 128, 128, formats=EXPERT_FORMATS, use_triton=True
        )
    bank.load_state_dict(state, strict=True, assign=True)
    return bank


def snapshot_sources(bank):
    return tuple(
        (
            getattr(bank, name),
            getattr(bank, name).clone(),
            getattr(bank, name).data_ptr(),
        )
        for name in bank.expert_tensor_names
    )


def assert_unmodified(test: unittest.TestCase, bank, snapshot):
    for name, (source, value, pointer) in zip(bank.expert_tensor_names, snapshot):
        test.assertIs(getattr(bank, name), source)
        test.assertEqual(source.device.type, "cpu")
        test.assertEqual(source.data_ptr(), pointer)
        test.assertTrue(torch.equal(source, value), name)
    test.assertIsNone(bank._fish_gpu_tensors)
    test.assertIsNone(bank._fish_gpu_pointers)
    test.assertIsNone(bank._fish_gpu_device)
    test.assertIsNone(bank._fish_gpu_cache)
    test.assertIsNone(bank._fish_decode_workspace)


class FishFormatCapabilityTests(unittest.TestCase):
    def test_support_query_requires_sm89_for_fp8_and_allows_int4(self):
        fp8 = make_bank("fp8_block")
        int4 = make_bank("int4_group")
        device = torch.device("cuda:0")

        with patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 8)
        ) as capability:
            self.assertFalse(fp8.supports_gpu_device(device))
            capability.assert_called_once_with(device)

        with patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 9)
        ) as capability:
            self.assertTrue(fp8.supports_gpu_device(device))
            capability.assert_called_once_with(device)

        with patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ):
            self.assertTrue(fp8.supports_gpu_device(device))

        with patch.object(torch.cuda, "get_device_capability") as capability:
            self.assertTrue(int4.supports_gpu_device(device))
            self.assertFalse(int4.supports_gpu_device(torch.device("cpu")))
            capability.assert_not_called()

    def test_unsupported_fp8_full_placement_rejects_before_cuda_allocations(self):
        bank = make_bank("fp8_block")
        sources = snapshot_sources(bank)
        device = torch.device("cuda:0")
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
            patch.object(torch.Tensor, "to", side_effect=AssertionError("weight copy attempted")),
            patch.object(torch, "empty", side_effect=AssertionError("CUDA allocation attempted")),
            patch.object(torch, "arange", side_effect=AssertionError("pointer allocation attempted")),
            self.assertRaisesRegex(ValueError, "capability 8\\.9.*CPU fallback"),
        ):
            bank.place_gpu_experts(device)

        assert_unmodified(self, bank, sources)

    def test_unsupported_fp8_cache_placement_rejects_before_cache_allocation(self):
        bank = make_bank("fp8_block")
        sources = snapshot_sources(bank)
        device = torch.device("cuda:0")
        cache_module = __import__(
            "models.TTS.fish_audio.expert_cache", fromlist=["PackedExpertCache"]
        )
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
            patch.object(
                cache_module,
                "PackedExpertCache",
                side_effect=AssertionError("cache constructor was called"),
            ) as cache_constructor,
            patch.object(torch, "empty", side_effect=AssertionError("CUDA allocation attempted")),
        ):
            with self.assertRaisesRegex(ValueError, "capability 8\\.9.*CPU fallback"):
                bank.place_cached_experts(capacity=1, topk=1, device=device)
            cache_constructor.assert_not_called()

        assert_unmodified(self, bank, sources)


if __name__ == "__main__":
    unittest.main()
