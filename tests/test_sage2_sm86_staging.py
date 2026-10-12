"""CPU-only regression checks for staged SageAttention on SM86."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


_ROOT = Path(__file__).resolve().parents[1]
_APP = str(_ROOT / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import torch

from shared import sage2_core


class _KernelCapture:
    def __init__(self):
        self.grid = None
        self.calls = []

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args, **kwargs):
            self.calls.append((args, kwargs))

        return launch


class Sage2SM86StagingTests(unittest.TestCase):
    def test_sm86_selects_per_block_path_without_sm89_extension(self):
        kernel = _KernelCapture()
        attention_module = SimpleNamespace(_attn_fwd=kernel)
        with (
            mock.patch.object(sage2_core, "_get_cuda_arch", return_value="sm86"),
            mock.patch.object(sage2_core, "SM89_ENABLED", False),
            mock.patch.object(
                sage2_core, "_quant_per_block_int8_kernel", kernel
            ),
            mock.patch.object(
                sage2_core, "_attn_qk_int8_per_block", attention_module
            ),
        ):
            settings = sage2_core.staged_settings(torch.device("cuda"))

        self.assertEqual(settings, {"qk_quant_gran": "per_block", "block_m": 128, "launch": None})

    def test_cpu_and_unsupported_or_unavailable_devices_fall_back(self):
        with mock.patch.object(sage2_core, "_get_cuda_arch") as get_arch:
            self.assertIsNone(sage2_core.staged_settings(torch.device("cpu")))
        get_arch.assert_not_called()

        with (
            mock.patch.object(sage2_core, "_get_cuda_arch", return_value="sm80"),
            mock.patch.object(sage2_core, "SM89_ENABLED", False),
        ):
            self.assertIsNone(sage2_core.staged_settings(torch.device("cuda")))

        with (
            mock.patch.object(sage2_core, "_get_cuda_arch", return_value="sm86"),
            mock.patch.object(sage2_core, "_quant_per_block_int8_kernel", None),
        ):
            self.assertIsNone(sage2_core.staged_settings(torch.device("cuda")))

    def test_existing_sm89_and_sm120_settings_are_retained(self):
        with (
            mock.patch.object(sage2_core, "SM89_ENABLED", True),
            mock.patch.object(sage2_core, "quant_key_per_thread_int8_kernel", object()),
            mock.patch.object(sage2_core, "quant_query_per_thread_int8_kernel", object()),
            mock.patch.object(sage2_core, "_get_cuda_arch", return_value="sm89"),
        ):
            self.assertEqual(
                sage2_core.staged_settings(torch.device("cuda")),
                sage2_core._fp8_pv_settings("sm89"),
            )

        with (
            mock.patch.object(sage2_core, "SM89_ENABLED", True),
            mock.patch.object(sage2_core, "_get_cuda_arch", return_value="sm120"),
        ):
            self.assertEqual(
                sage2_core.staged_settings(torch.device("cuda")),
                sage2_core._fp8_pv_settings("sm120"),
            )

    def test_q_and_k_keep_sage_per_block_scale_layout_and_parameters(self):
        quant_kernel = _KernelCapture()
        settings = sage2_core._PER_BLOCK_SETTINGS
        query = torch.randn((2, 129, 3, 64), dtype=torch.float16)
        key = torch.randn((1, 70, 2, 64), dtype=torch.float32)
        key_list = [key]

        with (
            mock.patch.object(sage2_core, "_quant_per_block_int8_kernel", quant_kernel),
            mock.patch.object(sage2_core, "_maybe_set_device"),
        ):
            query_int8, query_scale = sage2_core.staged_quantize_q(query, settings)
            self.assertEqual(tuple(query_int8.shape), tuple(query.shape))
            self.assertEqual(tuple(query_scale.shape), (2, 3, 2))
            self.assertEqual(quant_kernel.grid, (2, 3, 2))
            _, query_kwargs = quant_kernel.calls[-1]
            self.assertEqual(query_kwargs["BLK"], 128)
            self.assertAlmostEqual(
                query_kwargs["sm_scale"], 64**-0.5 * 1.44269504
            )

            key_int8, key_scale = sage2_core.staged_quantize_k(key_list, settings)

        self.assertEqual(key_list, [])
        self.assertEqual(tuple(key_int8.shape), tuple(key.shape))
        self.assertEqual(tuple(key_scale.shape), (1, 2, 2))
        self.assertEqual(quant_kernel.grid, (2, 2, 1))
        _, key_kwargs = quant_kernel.calls[-1]
        self.assertEqual(key_kwargs["BLK"], 64)
        self.assertEqual(key_kwargs["sm_scale"], 1.0)
        self.assertTrue(
            torch.allclose(
                key.mean(dim=1), torch.zeros_like(key.mean(dim=1)), atol=1e-6
            )
        )

    def test_sm86_stages_fp16_values_and_writes_into_strided_output(self):
        settings = sage2_core._PER_BLOCK_SETTINGS
        value_list = [torch.randn((1, 70, 1, 64), dtype=torch.bfloat16)]
        value_fp16, value_scale, value_mean = sage2_core.staged_quantize_v(
            value_list, settings
        )
        self.assertEqual(value_list, [])
        self.assertEqual(value_fp16.dtype, torch.float16)
        self.assertIsNone(value_scale)
        self.assertIsNone(value_mean)

        query_int8 = torch.empty((1, 130, 2, 64), dtype=torch.int8)
        key_int8 = torch.empty((1, 70, 1, 64), dtype=torch.int8)
        query_scale = torch.empty((1, 2, 2), dtype=torch.float32)
        key_scale = torch.empty((1, 1, 2), dtype=torch.float32)
        full_output = torch.empty((1, 130, 4, 64), dtype=torch.float16)
        output = full_output[..., 1:3, :]
        kernel = _KernelCapture()
        quantized = [
            (query_int8, query_scale),
            (key_int8, key_scale),
            (value_fp16, None, None),
        ]

        with (
            mock.patch.object(
                sage2_core,
                "_attn_qk_int8_per_block",
                SimpleNamespace(_attn_fwd=kernel),
            ),
            mock.patch.object(sage2_core, "_maybe_set_device"),
            mock.patch.object(sage2_core, "_TRITON_MASK_ARGUMENT", True),
        ):
            result = sage2_core.staged_attention(output, quantized, settings)

        self.assertIs(result, output)
        self.assertEqual(quantized, [])
        self.assertEqual(kernel.grid, (2, 2, 1))
        args, kwargs = kernel.calls[0]
        self.assertIs(args[5], output)
        self.assertIsNone(args[6])
        self.assertEqual(args[8:11], (query_int8.stride(0), query_int8.stride(2), query_int8.stride(1)))
        self.assertEqual(
            args[11:14], (key_int8.stride(0), key_int8.stride(2), key_int8.stride(1))
        )
        self.assertEqual(
            args[17:20], (output.stride(0), output.stride(2), output.stride(1))
        )
        self.assertEqual(args[20:24], (0, 0, 0, 0))
        self.assertEqual(args[24:28], (130, 70, 2, 2))
        self.assertEqual(kwargs["BLOCK_M"], 128)
        self.assertEqual(kwargs["BLOCK_N"], 64)
        self.assertEqual(kwargs["HEAD_DIM"], 64)
        self.assertEqual(kwargs["num_warps"], 4)
        self.assertEqual(kwargs["num_stages"], 3)

    def test_sm86_launch_supports_sage_kernel_without_mask_parameter(self):
        query_int8 = torch.empty((1, 130, 2, 64), dtype=torch.int8)
        key_int8 = torch.empty((1, 70, 1, 64), dtype=torch.int8)
        value = torch.empty((1, 70, 1, 64), dtype=torch.float16)
        query_scale = torch.empty((1, 2, 2), dtype=torch.float32)
        key_scale = torch.empty((1, 1, 2), dtype=torch.float32)
        output = torch.empty((1, 130, 2, 64), dtype=torch.float16)
        kernel = _KernelCapture()
        quantized = [
            (query_int8, query_scale),
            (key_int8, key_scale),
            (value, None, None),
        ]

        with (
            mock.patch.object(
                sage2_core,
                "_attn_qk_int8_per_block",
                SimpleNamespace(_attn_fwd=kernel),
            ),
            mock.patch.object(sage2_core, "_maybe_set_device"),
            mock.patch.object(sage2_core, "_TRITON_MASK_ARGUMENT", False),
        ):
            self.assertIs(
                sage2_core.staged_attention(
                    output, quantized, sage2_core._PER_BLOCK_SETTINGS
                ),
                output,
            )

        args, _ = kernel.calls[0]
        self.assertIs(args[5], output)
        self.assertEqual(tuple(args[6].shape), (0,))
        self.assertEqual(
            args[16:19], (output.stride(0), output.stride(2), output.stride(1))
        )
        self.assertEqual(args[19:23], (130, 70, 2, 2))


if __name__ == "__main__":
    unittest.main()
