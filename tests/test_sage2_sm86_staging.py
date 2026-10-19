"""CPU-only regression checks for staged SageAttention on SM86."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
import torch


def _module(name, *, package=False, **attributes):
    module = ModuleType(name)
    if package:
        module.__path__ = []
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _noop(*args, **kwargs):
    pass


def _load_sage2_core():
    """Load the real module with temporary, dependency-light Sage stubs."""
    sageattention = _module(
        "sageattention",
        package=True,
        _fused=SimpleNamespace(transpose_pad_permute_cuda=_noop),
        _qattn_sm80=SimpleNamespace(qk_int8_sv_f16_accum_f32_attn=_noop),
        _qattn_sm89=SimpleNamespace(
            qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf=_noop
        ),
        _qattn_sm90=SimpleNamespace(
            qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf=_noop
        ),
    )
    triton = _module("sageattention.triton", package=True)
    quant_per_block = _module(
        "sageattention.triton.quant_per_block",
        per_block_int8=_noop,
        quant_per_block_int8_kernel=object(),
    )
    quant_per_block_varlen = _module(
        "sageattention.triton.quant_per_block_varlen", per_block_int8=_noop
    )
    attn_kernel = SimpleNamespace(arg_names=("mask",))
    attn_per_block = _module(
        "sageattention.triton.attn_qk_int8_per_block",
        forward=_noop,
        _attn_fwd=attn_kernel,
    )
    attn_modules = {
        name: _module(f"sageattention.triton.{name}", forward=_noop)
        for name in (
            "attn_qk_int8_per_block_causal",
            "attn_qk_int8_block_varlen",
            "attn_qk_int8_per_block_causal_varlen",
        )
    }
    quant_per_thread = _module(
        "sageattention.triton.quant_per_thread",
        per_thread_int8=_noop,
        quant_key_per_thread_int8_kernel=object(),
        quant_query_per_thread_int8_kernel=object(),
    )
    quant = _module(
        "sageattention.quant",
        per_block_int8=_noop,
        per_warp_int8=_noop,
        sub_mean=_noop,
        per_channel_fp8=_noop,
    )
    sageattention.triton = triton
    sageattention.quant = quant
    triton.quant_per_block = quant_per_block
    triton.quant_per_block_varlen = quant_per_block_varlen
    triton.attn_qk_int8_per_block = attn_per_block
    triton.quant_per_thread = quant_per_thread
    for name, module in attn_modules.items():
        setattr(triton, name, module)
    fake_modules = {
        "sageattention": sageattention,
        "sageattention.triton": triton,
        "sageattention.triton.quant_per_block": quant_per_block,
        "sageattention.triton.quant_per_block_varlen": quant_per_block_varlen,
        "sageattention.triton.attn_qk_int8_per_block": attn_per_block,
        "sageattention.triton.quant_per_thread": quant_per_thread,
        "sageattention.quant": quant,
        **{
            module.__name__: module
            for module in attn_modules.values()
        },
    }
    module_name = "_sage2_core_sm86_cpu_test"

    with (
        mock.patch.dict(sys.modules, fake_modules),
        mock.patch("importlib.metadata.version", return_value="2.2.0"),
        mock.patch.object(torch.cuda, "device_count", return_value=0),
    ):
        spec = importlib.util.spec_from_file_location(
            module_name, _ROOT / "app" / "shared" / "sage2_core.py"
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {module_name: module}):
            spec.loader.exec_module(module)
        return module


sage2_core = _load_sage2_core()


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
