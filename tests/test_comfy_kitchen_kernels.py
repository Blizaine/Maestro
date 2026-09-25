from __future__ import annotations

import functools
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


_APP = Path(__file__).resolve().parents[1] / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from models.ltx2.denoiser_kernels import project_many
from models.minimax_h3 import denoiser_kernels as h3_kernels
from shared.kernels import comfy_kitchen


class _FakeQLinear(torch.nn.Module):
    def forward(self, value):
        return value


def _native_convrot_lora_forward(module, *args, **kwargs):
    return module._mm_convrot_native_forward(*args, **kwargs)


class ComfyKitchenKernelTests(unittest.TestCase):
    def test_cuda_capability_gate_rejects_unverified_architecture_before_import(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            mock.patch.object(comfy_kitchen.importlib, "import_module") as importer,
        ):
            self.assertIsNone(
                comfy_kitchen._cuda_backend({"int8_linear"}, torch.device("cuda:0"))
            )
        importer.assert_not_called()

    def test_cuda_capability_gate_uses_public_registry_and_private_symbol_probe(self):
        package = types.SimpleNamespace(
            list_backends=lambda: {
                "cuda": {
                    "available": True,
                    "disabled": False,
                    "capabilities": ["int8_linear"],
                }
            }
        )
        backend_module = types.SimpleNamespace(
            quantize_int8_rowwise_convrot64=lambda value, group_size: (value, value),
            _C=types.SimpleNamespace(cutlass_int8_dequant=lambda *args: True),
        )

        def import_backend(name):
            return package if name == "comfy_kitchen" else backend_module

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            mock.patch.object(comfy_kitchen.importlib, "import_module", side_effect=import_backend),
        ):
            result = comfy_kitchen._cuda_backend({"int8_linear"}, torch.device("cuda:0"))
        self.assertIsNotNone(result)
        self.assertEqual(result.capabilities, frozenset({"int8_linear"}))

        package.list_backends = lambda: {
            "cuda": {"available": True, "disabled": False, "capabilities": ["other"]}
        }
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            mock.patch.object(comfy_kitchen.importlib, "import_module", side_effect=import_backend) as importer,
        ):
            self.assertIsNone(
                comfy_kitchen._cuda_backend({"int8_linear"}, torch.device("cuda:0"))
            )
        importer.assert_called_once_with("comfy_kitchen")
        self.assertTrue(comfy_kitchen._has_native_int8_projection_symbols(backend_module))

    def test_disabled_or_missing_kitchen_dependency_falls_back(self):
        package = types.SimpleNamespace(
            list_backends=lambda: {
                "cuda": {
                    "available": True,
                    "disabled": True,
                    "capabilities": ["int8_linear"],
                }
            }
        )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            mock.patch.object(comfy_kitchen.importlib, "import_module", return_value=package) as importer,
        ):
            self.assertIsNone(
                comfy_kitchen._cuda_backend({"int8_linear"}, torch.device("cuda:0"))
            )
        importer.assert_called_once_with("comfy_kitchen")

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            mock.patch.object(
                comfy_kitchen.importlib,
                "import_module",
                side_effect=ImportError("Kitchen is not installed"),
            ),
        ):
            self.assertIsNone(
                comfy_kitchen._cuda_backend({"rms_rope_split_half"}, torch.device("cuda:0"))
            )

        self.assertFalse(
            comfy_kitchen._has_native_int8_projection_symbols(
                types.SimpleNamespace(
                    quantize_int8_rowwise_convrot64=lambda *args: None,
                    _C=types.SimpleNamespace(),
                )
            )
        )
        missing_native = comfy_kitchen.KitchenCudaBackend(
            package,
            types.SimpleNamespace(
                quantize_int8_rowwise_convrot64=lambda *args: None,
                _C=types.SimpleNamespace(),
            ),
            frozenset({"int8_linear"}),
        )
        with mock.patch.object(
            comfy_kitchen,
            "kitchen_backend_for",
            return_value=missing_native,
        ):
            self.assertIsNone(comfy_kitchen._int8_projection_backend(torch.device("cuda:0")))

    def test_h3_rope_keeps_cpu_rotary_on_existing_math_path(self):
        rotary = (torch.ones(3, 4), torch.zeros(3, 4))
        self.assertIs(h3_kernels.prepare_rope(rotary), rotary)

    def test_h3_norm_rope_uses_out_of_place_kernel_api(self):
        source = Path(h3_kernels.__file__).read_text(encoding="utf-8")
        self.assertIn("module.rms_rope_split_half(", source)
        self.assertNotIn("module.rms_rope_split_half_(", source)

    def test_packed_h3_rope_matches_kitchen_eager_api(self):
        try:
            import comfy_kitchen
        except ImportError:
            self.skipTest("Comfy Kitchen is optional in the CPU fallback environment")

        batch, tokens, heads, head_dim, rot_dim = 1, 5, 2, 32, 16
        pairs = rot_dim // 2
        angles = torch.randn(tokens, pairs, dtype=torch.float32)
        cosine_half, sine_half = angles.cos(), angles.sin()
        cosine = torch.cat((cosine_half, cosine_half), dim=-1)
        sine = torch.cat((sine_half, sine_half), dim=-1)
        freqs = h3_kernels._pack_split_half_rope(cosine, sine)
        self.assertEqual(tuple(freqs.shape), (1, tokens, 1, pairs, 2, 2))

        qkv = torch.randn(
            batch,
            tokens,
            heads * head_dim * 3,
            dtype=torch.bfloat16,
        )
        query_flat, key_flat, _ = qkv.chunk(3, dim=-1)
        query = query_flat.view(batch, tokens, heads, head_dim)
        key = key_flat.view(batch, tokens, heads, head_dim)
        q_norm = torch.nn.RMSNorm(head_dim, eps=1e-5, dtype=torch.bfloat16)
        k_norm = torch.nn.RMSNorm(head_dim, eps=1e-5, dtype=torch.bfloat16)
        actual_q, actual_k = comfy_kitchen.rms_rope_split_half(
            query,
            key,
            freqs,
            q_norm.weight,
            k_norm.weight,
            epsilon=q_norm.eps,
            rot_dim=rot_dim,
        )

        def local_norm_rope(value, norm):
            value = norm(value)
            first = value[..., :pairs]
            second = value[..., pairs:rot_dim]
            rotated = torch.cat((-second, first), dim=-1)
            cos = cosine.to(value.dtype)[None, :, None]
            sin = sine.to(value.dtype)[None, :, None]
            rotary = value[..., :rot_dim] * cos + rotated * sin
            return torch.cat((rotary, value[..., rot_dim:]), dim=-1)

        expected_q = local_norm_rope(query, q_norm)
        expected_k = local_norm_rope(key, k_norm)
        self.assertTrue(torch.allclose(actual_q, expected_q, atol=0.025, rtol=0.025))
        self.assertTrue(torch.allclose(actual_k, expected_k, atol=0.025, rtol=0.025))

    def test_prequantized_projection_writes_only_the_requested_output_tile(self):
        calls = []

        def cutlass(*args):
            quantized, weight, input_scale, weight_scale, bias, output = args[:6]
            calls.append((quantized.shape, weight.shape, input_scale.shape, weight_scale.shape))
            output.fill_(2.5)
            return True

        backend = types.SimpleNamespace(
            _wrap_for_dlpack=lambda value: value,
            _gemm_vector_arg=lambda value, device, dtype: value,
            _empty_cuda_tensor=lambda device, dtype: torch.empty(0, device=device, dtype=dtype),
            DTYPE_TO_CODE={torch.bfloat16: 1},
            _C=types.SimpleNamespace(cutlass_int8_dequant=cutlass),
        )
        qweight = types.SimpleNamespace(
            _data=torch.ones(8, 256, dtype=torch.int8),
            _scale=torch.ones(8, dtype=torch.float32),
        )
        quantized = torch.zeros(4, 256, dtype=torch.int8)
        input_scale = torch.ones(4, 1, dtype=torch.float32)
        output = torch.empty(4, 8, dtype=torch.bfloat16)
        result = comfy_kitchen._prequantized_kitchen_linear(
            backend,
            qweight,
            None,
            quantized,
            input_scale,
            output,
            123,
        )
        self.assertIs(result, output)
        self.assertTrue(torch.all(output == 2.5))
        self.assertEqual(calls, [((4, 256), (8, 256), (4, 1), (8,))])

    def test_mm_lora_wrappers_and_deselected_adapter_state_bypass_shared_path(self):
        module = _FakeQLinear()
        module._mm_convrot_native_forward = lambda value: value
        module.forward = functools.partial(_native_convrot_lora_forward, module)
        module._mm_lora_data = {"deselected_adapter_GPU": object()}
        self.assertTrue(comfy_kitchen._has_unhandled_mm_lora_hook(module, _FakeQLinear))

        module._mm_lora_data = {}
        self.assertFalse(comfy_kitchen._has_unhandled_mm_lora_hook(module, _FakeQLinear))
        module.register_forward_hook(lambda layer, inputs, output: output)
        self.assertTrue(comfy_kitchen._has_unhandled_mm_lora_hook(module, _FakeQLinear))

    def test_ltx_cpu_fallback_calls_each_module_and_keeps_module_hooks(self):
        calls = []

        def make_layer():
            layer = torch.nn.Linear(4, 3)
            layer.register_forward_pre_hook(lambda module, inputs: calls.append(module))
            return layer

        modules = (make_layer(), make_layer(), make_layer())
        x = torch.randn(1, 5, 4)
        results = project_many(modules, x)
        expected = tuple(module(x) for module in modules)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(torch.equal(actual, reference) for actual, reference in zip(results, expected)))
        self.assertEqual(len(calls), 6)


if __name__ == "__main__":
    unittest.main()
