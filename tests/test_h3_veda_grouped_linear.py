"""CPU-only checks for shared INT8 preparation and safe grouped fallbacks."""

from __future__ import annotations

import functools
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from shared.kernels import int8_backend
from shared.qtypes import int8_convrot


class _SharedInputProjection(torch.nn.Module):
    """Small module whose ordinary forward is the real Maestro qtype forward."""

    def __init__(self, index, features=64, outputs=8):
        super().__init__()
        self.index = index
        self.in_features = features
        self.out_features = outputs
        self.qweight = types.SimpleNamespace(
            _data=torch.zeros(outputs, features, dtype=torch.int8),
            _scale=torch.ones(outputs, 1, dtype=torch.float32),
        )
        self.bias = None
        self.forward = types.MethodType(int8_convrot.QLinearInt8ConvRot.forward, self)


class _HookedLoraProjection(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self._mm_lora_data = {"active": object()}
        self.pre_hook_calls = 0
        self.forward_calls = 0
        self.register_forward_pre_hook(self._pre_hook)

    def _pre_hook(self, _module, inputs):
        self.pre_hook_calls += 1
        return inputs

    def forward(self, x):
        self.forward_calls += 1
        # Represents the wrapped ordinary/MMGP LoRA module behavior.
        return x[:, :2] + self.index


class TestVedaGroupedLinear(unittest.TestCase):
    def test_mmgp_native_wrapper_is_allowed_but_arbitrary_partial_is_not(self):
        module = types.SimpleNamespace()
        qlinear_type = int8_convrot.QLinearInt8ConvRot
        wrapper_name = (
            f"_mm_wrap_{qlinear_type.__module__.replace('.', '_')}_"
            f"{qlinear_type.__name__}"
        )

        def mmgp_wrapper(_module, *_args, **_kwargs):
            return None

        mmgp_wrapper.__name__ = wrapper_name
        module.forward = functools.partial(mmgp_wrapper, module)
        module._mm_forward = types.MethodType(qlinear_type.forward, module)
        module._mm_manager = object()
        self.assertTrue(int8_backend._grouped_linear_forward_is_native(module, qlinear_type))

        module.forward = functools.partial(lambda *_args: None, module)
        self.assertFalse(int8_backend._grouped_linear_forward_is_native(module, qlinear_type))

    def test_shared_quantization_is_consumed_by_paired_qtype_projection(self):
        modules = [_SharedInputProjection(i) for i in range(3)]
        x = (
            torch.arange(3 * 4 * 64, dtype=torch.float32)
            .remainder(11)
            .to(torch.bfloat16)
            .view(3, 4, 64)
        )
        quantized_inputs = []
        projected_inputs = []

        def quantize(rows):
            quantized_inputs.append(rows.clone())
            return rows.to(torch.int8), torch.arange(
                1, rows.shape[0] + 1, dtype=torch.float32
            ).view(-1, 1)

        def prequantized_linear(weight, bias, q, scales, out):
            self.assertIsNone(bias)
            self.assertEqual(q.dtype, torch.int8)
            projected_inputs.append((weight, q.clone(), scales.clone()))
            out.fill_(float(len(projected_inputs)))
            return out

        with (
            mock.patch.object(int8_backend, "_linear_grouped_supported", return_value=True),
            mock.patch.object(
                int8_backend,
                "_kitchen",
                types.SimpleNamespace(quantize_int8_rowwise=quantize),
            ),
            mock.patch.object(int8_convrot, "_rotate_activation", wraps=lambda tensor, group: tensor) as rotate,
            mock.patch.object(int8_backend, "kitchen_linear_prequantized", side_effect=prequantized_linear) as consume,
        ):
            result = int8_backend.linear_grouped(modules, x)

        self.assertEqual(len(quantized_inputs), 1)
        rotate.assert_called_once()
        rotation_args, _rotation_kwargs = rotate.call_args
        self.assertIs(rotation_args[0], x)
        self.assertEqual(rotation_args[1], 64)
        self.assertEqual(consume.call_count, 3)
        self.assertEqual(len(projected_inputs), 3)
        expected_q = x.to(torch.int8)
        self.assertTrue(torch.equal(quantized_inputs[0], x.reshape(-1, 64)))
        for index, (weight, q, scales) in enumerate(projected_inputs):
            self.assertIs(weight, modules[index].qweight)
            self.assertTrue(torch.equal(q, expected_q[index]))
            self.assertEqual(scales.shape, (4, 1))
            self.assertTrue(
                torch.equal(
                    scales[:, 0],
                    torch.arange(1 + index * 4, 5 + index * 4, dtype=torch.float32),
                )
            )
            self.assertFalse(hasattr(modules[index], "_wangp_prequantized_input"))
        expected = torch.stack(
            [torch.full((4, 8), i + 1, dtype=x.dtype) for i in range(3)]
        )
        self.assertTrue(torch.equal(result, expected))

    def test_malformed_paired_input_is_rejected_by_qtype_consumer(self):
        module = _SharedInputProjection(0)
        module._wangp_prequantized_input = (torch.ones(2, 64, dtype=torch.int8),)
        with self.assertRaisesRegex(ValueError, "Malformed temporary grouped INT8 input"):
            module(torch.ones(2, 64, dtype=torch.bfloat16))

    def test_hooks_and_lora_modules_take_normal_module_forward_fallback(self):
        modules = [_HookedLoraProjection(i) for i in range(3)]
        x = torch.arange(3 * 4 * 64, dtype=torch.float32).view(3, 4, 64)
        result = int8_backend.linear_grouped(modules, x)

        self.assertTrue(torch.equal(result[0], x[0, :, :2]))
        self.assertTrue(torch.equal(result[1], x[1, :, :2] + 1))
        self.assertTrue(torch.equal(result[2], x[2, :, :2] + 2))
        for module in modules:
            self.assertEqual(module.pre_hook_calls, 1)
            self.assertEqual(module.forward_calls, 1)
            self.assertFalse(hasattr(module, "_wangp_prequantized_input"))

    def test_grouped_linear_rejects_mismatched_module_count_and_rank(self):
        module = torch.nn.Linear(64, 8)
        with self.assertRaisesRegex(ValueError, r"shape \[modules, rows, features\]"):
            int8_backend.linear_grouped([module], torch.ones(2, 4, 64))
        with self.assertRaisesRegex(ValueError, r"shape \[modules, rows, features\]"):
            int8_backend.linear_grouped([module], torch.ones(4, 64))


if __name__ == "__main__":
    unittest.main()
