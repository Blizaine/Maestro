"""NVFP4 dispatch and row alignment, using CPU tensors and mocked kernels."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import torch
from shared.qtypes import nvfp4


def weight_fixture(matrix):
    return SimpleNamespace(
        shape=matrix.shape, size=matrix.size,
        _layout="legacy", _data=torch.zeros(matrix.shape[0], matrix.shape[1] // 2, dtype=torch.uint8),
        _scale=torch.ones(matrix.shape[0], matrix.shape[1] // 16),
        _input_global_scale=torch.tensor(1.0), _alpha=torch.tensor(1.0),
        dequantize=lambda dtype=None, device=None: matrix.to(dtype=dtype, device=device),
    )


class TestNvfp4Fallback(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(nvfp4, "_NVFP4_UNSUPPORTED_SHAPES", set())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_gemm_algorithm_uses_linear_and_only_caches_failed_shape(self):
        x = torch.arange(96, dtype=torch.float32).reshape(3, 32) / 100
        matrix = torch.eye(32)
        weight = weight_fixture(matrix)
        bias = torch.arange(32, dtype=torch.float64)
        with mock.patch.object(nvfp4, "_nvfp4_can_use_kernel", return_value=True), mock.patch.object(nvfp4, "_nvfp4_linear_cuda", side_effect=RuntimeError("Unable to find suitable cuBLAS GEMM algorithm")) as kernel:
            for _ in range(2):
                result = nvfp4._nvfp4_linear(x, weight, bias)
                torch.testing.assert_close(result, x + bias.float())
            self.assertEqual(kernel.call_count, 1)
            nvfp4._nvfp4_linear(x[:1], weight, bias)
            self.assertEqual(kernel.call_count, 2)

    def test_other_cuda_errors_are_not_swallowed(self):
        x = torch.zeros(1, 32)
        for error in (torch.OutOfMemoryError("CUDA out of memory"), RuntimeError("CUDA illegal memory access")):
            with self.subTest(error=str(error)), mock.patch.object(nvfp4, "_nvfp4_can_use_kernel", return_value=True), mock.patch.object(nvfp4, "_nvfp4_linear_cuda", side_effect=error):
                with self.assertRaises(type(error)):
                    nvfp4._nvfp4_linear(x, weight_fixture(torch.eye(32)))
        self.assertFalse(nvfp4._NVFP4_UNSUPPORTED_SHAPES)

    def test_lightx2v_pads_odd_rows_and_preserves_output_shape_and_bias(self):
        x = torch.arange(96, dtype=torch.bfloat16).reshape(1, 3, 32) / 100
        bias = torch.ones(32, dtype=torch.bfloat16)
        quantized_inputs = []

        def quantize(value, scale):
            quantized_inputs.append(value.clone())
            return value, torch.ones(1)

        kernel = SimpleNamespace(scaled_nvfp4_quant=quantize, cutlass_scaled_nvfp4_mm=lambda a, b, sa, sb, **kw: a + kw["bias"])
        with mock.patch.object(nvfp4, "_lx_gemm", kernel):
            result = nvfp4._nvfp4_linear_cuda_lightx2v(x, weight_fixture(torch.eye(32)), bias)
        self.assertEqual(tuple(quantized_inputs[0].shape), (16, 32))
        self.assertEqual(quantized_inputs[0][3:].count_nonzero().item(), 0)
        torch.testing.assert_close(result, x + bias)


if __name__ == "__main__":
    unittest.main()
