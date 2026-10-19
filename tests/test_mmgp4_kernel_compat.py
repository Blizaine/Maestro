"""CPU regressions for the vendored MMGP4 and its Maestro qtype adapters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


_APP = Path(__file__).resolve().parents[1] / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))


class TestVendoredMmgp4(unittest.TestCase):
    def test_wildcard_cotenants_and_native_linear_forward_survive(self):
        import torch
        import mmgp
        from mmgp import offload

        self.assertEqual(mmgp.__version__, "4.0.0")
        self.assertTrue(offload.offload.supports_cotenant_wildcards)

        # Avoid MMGP's initializer because it queries GPU capacity. This is a
        # CPU-only test of cotenant and LoRA-forward dispatch behavior.
        manager = object.__new__(offload.offload)
        manager.active_models_ids = ["tiny_vae"]
        manager.cotenants_map = {"tiny_vae": "*", "preview_consumer": "*"}
        self.assertTrue(manager.can_model_be_cotenant("preview_consumer"))

        layer = torch.nn.Linear(3, 2)
        layer._mm_requires_native_linear_forward = True
        calls = []

        def native_forward(input, *args, **kwargs):
            calls.append(input)
            return input[:, :2] + 17

        layer._mm_lora_old_forward = native_forward
        model = SimpleNamespace(
            _loras_active_adapters=["active"],
            _loras_scaling={},
        )
        input = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        result = manager._lora_linear_forward(model, layer, {}, input)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], input)
        self.assertTrue(torch.equal(result, input[:, :2] + 17))

    def test_license_and_allocator_binaries_match_provenance(self):
        import mmgp

        package_dir = Path(mmgp.__file__).resolve().parent
        provenance = json.loads(
            (package_dir / "PROVENANCE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["upstream_commit"], mmgp.__source_commit__)
        self.assertEqual(provenance["license"], "WanGP Community License 2.0")
        self.assertIn(
            "WanGP Community License 2.0",
            (package_dir / "LICENSE.txt").read_text(encoding="utf-8"),
        )
        for relative_path, expected in provenance["allocator_binaries"].items():
            actual = hashlib.sha256(
                (package_dir / relative_path).read_bytes()
            ).hexdigest().upper()
            self.assertEqual(actual, expected.removeprefix("sha256:"))

    def test_kitchen_row_provider_matches_mmgp4_registration_api(self):
        from mmgp import offload
        from shared.kernels import int8_backend

        provider = int8_backend._KITCHEN_ROW_PROJECTION
        for method in ("supports", "key", "prepare", "rows"):
            self.assertTrue(callable(getattr(provider, method)))
        for method in ("register_row_projection", "unregister_row_projection"):
            self.assertTrue(callable(getattr(offload, method)))

        offload.register_row_projection(provider)
        try:
            self.assertIn(provider, offload._ROW_PROJECTIONS)
        finally:
            offload.unregister_row_projection(provider)


class TestCustomQtypesCpu(unittest.TestCase):
    def test_scaled_fp8_channel_scales_and_full_precision_hint(self):
        import torch
        from shared.qtypes import scaled_fp8

        data = torch.linspace(-1, 1, 16 * 16).reshape(16, 16).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.01, 0.16, 16)
        weight = scaled_fp8.ScaledFP8WeightTensor.create(
            data,
            scales,
            size=(16, 16),
            stride=(16, 1),
            dtype=torch.float32,
        )
        self.assertTrue(scaled_fp8._scaled_mm_static_ok(data, scales))
        input = torch.randn(3, 16)
        actual = weight.linear(input)
        expected = torch.nn.functional.linear(input, weight.dequantize(dtype=input.dtype))
        self.assertTrue(torch.allclose(actual, expected))

        encoded = torch.tensor(
            list(b'{"full_precision_matrix_mult": true}'), dtype=torch.uint8
        )
        self.assertTrue(scaled_fp8._decode_comfy_quant(encoded)["full_precision_matrix_mult"])
        self.assertEqual(scaled_fp8._decode_comfy_quant(torch.tensor([255], dtype=torch.uint8)), {})
        forced = scaled_fp8.ScaledFP8WeightTensor.create(
            data,
            scales,
            size=(16, 16),
            stride=(16, 1),
            dtype=torch.float32,
            force_full_precision_mm=True,
        )
        with mock.patch.object(scaled_fp8, "_scaled_mm_available", return_value=True):
            forced._set_linear_impl()
        self.assertIs(forced._linear_impl, scaled_fp8.ScaledFP8WeightTensor._linear_fallback)

    def test_scaled_fp8_activation_fallback_remains_cpu_safe(self):
        import torch
        from shared.qtypes import scaled_fp8

        input = torch.tensor([[0.0, 1.0, -2.0]], dtype=torch.float32)
        with mock.patch.object(scaled_fp8, "_get_activation_kernel", return_value=False):
            quantized, scale = scaled_fp8._quantize_activation(input, torch.float8_e4m3fn)

        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertAlmostEqual(
            float(scale),
            float(torch.tensor(2.0 / torch.finfo(torch.float8_e4m3fn).max)),
            places=8,
        )

    def test_nvfp4_pre_quant_scale_survives_router_buffer_omission(self):
        import torch
        from shared.qtypes.nvfp4 import QLinearNVFP4, NVFP4WeightTensor, _NVFP4_QTYPE

        source = torch.nn.Linear(4, 2, bias=False, dtype=torch.float32)
        source.register_buffer("pre_quant_scale", torch.empty(4), persistent=True)
        qmodule = QLinearNVFP4.qcreate(source, _NVFP4_QTYPE, device="cpu")
        scale = torch.tensor([2.0, 3.0, 4.0, 5.0])
        state_dict = {
            "weight": torch.full((2, 2), 0xFF, dtype=torch.uint8),
            "weight_scale": torch.ones((2, 1)),
            "input_global_scale": torch.ones(()),
            "alpha": torch.ones(()),
            "pre_quant_scale": scale.clone(),
        }
        qmodule._load_from_state_dict(state_dict, "", {}, False, [], [], [])

        self.assertIsInstance(qmodule.qweight, NVFP4WeightTensor)
        self.assertTrue(torch.equal(qmodule.qweight._pre_quant_scale, scale))
        self.assertIn("pre_quant_scale", dict(qmodule.qweight.get_quantized_subtensors()))

        # Simulate MMGP's routed module, which copies ordinary attrs and the
        # quantized Parameter but omits buffers.
        del qmodule._buffers["pre_quant_scale"]
        del qmodule._nvfp4_pre_quant_scale
        input = torch.ones((1, 4))
        with mock.patch.object(
            torch.nn.functional,
            "linear",
            side_effect=lambda x, weight, bias=None: x.clone(),
        ) as linear:
            qmodule(input)
        self.assertTrue(torch.equal(linear.call_args.args[0], input * scale))

    def test_tensorwise_int8_embedding_handler_is_available(self):
        import torch
        from shared.qtypes import nvfp4

        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(3, 2)
        scale = torch.tensor([[0.1], [0.2], [0.3]])
        config = torch.tensor(
            list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8
        )
        state_dict = {
            "embedding.comfy_quant": config,
            "embedding.weight_scale": scale,
        }
        quantization_map = {"embedding": {"weights": "qint8"}}
        result, _ = nvfp4.apply_pre_quantization(
            model, state_dict, quantization_map, default_dtype=torch.float32
        )

        self.assertIs(result, quantization_map)
        self.assertIsInstance(model.embedding, nvfp4.Int8TensorwiseEmbedding)
        self.assertNotIn("embedding.comfy_quant", state_dict)
        model.embedding.weight.data.copy_(torch.tensor([[1, 2], [3, 4], [5, 6]]))
        model.embedding.weight_scale.copy_(scale)
        output = model.embedding(torch.tensor([0, 2]))
        expected = torch.tensor([[0.1, 0.2], [1.5, 1.8]])
        self.assertTrue(torch.allclose(output, expected))


if __name__ == "__main__":
    unittest.main()
