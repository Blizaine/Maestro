"""Exercise the installed Quanto/MMGP dispatch, including checkpoint loading."""
import json
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

try:
    import torch
    from mmgp import offload, quant_router
    from optimum.quanto.tensor.activations.qbytes import ActivationQBytesTensor
    from optimum.quanto.tensor.qtype import qint8
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
    from shared.kernels import quanto_int8_inject as inject
    from shared.qtypes import int8_convrot
    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False


@unittest.skipUnless(RUNTIME_AVAILABLE, "Quanto/MMGP runtime dependencies are required")
class TestQuantoInt8Dispatch(unittest.TestCase):
    def setUp(self):
        inject.disable_quanto_int8_kernel()
        quant_router.register_handler("shared.qtypes.int8_convrot")

    def tearDown(self):
        inject.disable_quanto_int8_kernel()

    def load_layer(self, dtype, bias, in_features=8, out_features=4):
        data = torch.arange(in_features * out_features).remainder(32).sub(16).to(torch.int8).reshape(out_features, in_features)
        scales = torch.linspace(0.00691, 0.03237, out_features).reshape(-1, 1)
        state = {
            "linear.weight": data,
            "linear.weight_scale": scales.clone(),
            "linear.comfy_quant": torch.tensor(list(json.dumps({
                "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4,
            }).encode()), dtype=torch.uint8),
        }
        if bias:
            state["linear.bias"] = torch.linspace(-0.2, 0.4, out_features, dtype=dtype)
        model = torch.nn.Module()
        model.linear = torch.nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        offload.load_model_data(model, (state, None), default_dtype=dtype, verboseLevel=0)
        return model.linear, data, scales

    def test_loaded_convrot_retains_activation_dtype_and_original_scales(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for bias in (False, True):
                with self.subTest(dtype=dtype, bias=bias), torch.no_grad():
                    layer, data, scales = self.load_layer(dtype, bias)
                    # Batch dimensions and a non-contiguous input exercise the real reshape path.
                    x = torch.linspace(-2, 2, 96, dtype=dtype).reshape(2, 3, 16)[..., ::2]
                    rotated = int8_convrot._rotate_activation(x, 4)
                    expected = torch.nn.functional.linear(
                        rotated.float(), data.float() * scales,
                        layer.bias.float() if bias else None,
                    )
                    actual = layer(x)
                    self.assertEqual(type(layer.weight).__name__, "Int8ConvRotWeightTensor")
                    self.assertEqual(actual.shape, (2, 3, 4))
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.008)
                    self.assertEqual(layer.weight._scale.dtype, torch.float32)
                    torch.testing.assert_close(layer.weight._scale, scales, rtol=0, atol=0)

    def test_plain_quanto_weight_uses_dense_input_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype), torch.no_grad():
                data = torch.arange(-16, 16, dtype=torch.int8).reshape(4, 8)
                scales = torch.full((4, 1), 0.01357)
                weight = WeightQBytesTensor.create(qint8, 0, data.shape, data.stride(), data, scales)
                x = torch.ones(3, 8, dtype=dtype)
                actual = torch.nn.functional.linear(x, weight)
                expected = torch.nn.functional.linear(x.float(), data.float() * scales)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.008)

    def test_quantized_activations_keep_combined_scale_precision(self):
        data = torch.arange(-16, 16, dtype=torch.int8).reshape(4, 8)
        scales = torch.full((4, 1), 0.01357)
        weight = WeightQBytesTensor.create(qint8, 0, data.shape, data.stride(), data, scales)
        x = ActivationQBytesTensor.quantize(torch.ones(8, 8), qint8, torch.tensor(0.125))
        with torch.no_grad():
            actual = torch.nn.functional.linear(x, weight)
        expected = (x._data.int() @ data.int().T).float() * (x._scale * scales).T
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_convrot_lora_stays_in_activation_precision(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.no_grad():
                layer, _, _ = self.load_layer(dtype, False)
                x = torch.linspace(-2, 2, 16, dtype=dtype).reshape(2, 8)
                base = layer(x).clone()
                a = torch.full((2, 8), 0.125, dtype=dtype)
                b = torch.full((4, 2), 0.25, dtype=dtype)
                model = torch.nn.Module()
                model.linear = layer
                model._loras_active_adapters = ["turbo"]
                model._loras_scaling = {}
                layer._mm_lora_old_forward = layer.forward
                layer._mm_lora_model = model
                layer._mm_lora_data = {"turbo_GPU": [a, b, None, None, 1.0, {"type": "lora"}]}
                layer._mm_manager = SimpleNamespace(_get_lora_scaling=lambda *args: 1.0)
                self.assertEqual(int8_convrot.install_native_lora_forwards(model), 1)
                actual = layer(x)
                expected = base + (x @ a.T) @ b.T
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(os.environ.get("MAESTRO_TEST_CUDA") == "1", "Explicit idle-GPU test")
    def test_cuda_native_dispatch_and_triton_retry_after_native_oom(self):
        with torch.no_grad():
            layer, _, _ = self.load_layer(torch.bfloat16, False, 256, 256)
            layer = layer.to("cuda")
            x = torch.linspace(-2, 2, 128 * 256, device="cuda", dtype=torch.bfloat16).reshape(1, 128, 256)
            expected = layer(x)
            self.assertTrue(inject.enable_quanto_int8_kernel())
            with mock.patch.dict(os.environ, {"MAESTRO_CONVROT_KERNEL_CHECK": "1"}), \
                 mock.patch.object(inject, "_time_convrot_forward", side_effect=[1, 10]):
                actual = layer(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(any(inject._CONVROT_BACKENDS.values()))
                with mock.patch.object(torch.ops.quanto, "qbytes_mm", side_effect=torch.OutOfMemoryError("simulated native allocation failure")):
                    retried = layer(x)
            self.assertEqual(retried.dtype, torch.bfloat16)
            torch.testing.assert_close(retried, expected, rtol=0.03, atol=0.15)
            self.assertFalse(any(inject._CONVROT_BACKENDS.values()))
            self.assertFalse(inject._RUNTIME_DISABLED)


@unittest.skipUnless(RUNTIME_AVAILABLE, "Quanto/MMGP runtime dependencies are required")
class TestConvRotBackendChoice(unittest.TestCase):
    def setUp(self):
        inject._CONVROT_BACKENDS.clear()
        self.has_headroom = inject._native_convrot_has_headroom
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(inject._CONVROT_BACKENDS.clear)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(mock.patch.dict(os.environ, {"MAESTRO_CONVROT_KERNEL_CHECK": "1"}))
        self.stack.enter_context(mock.patch.object(inject, "_is_compiling_graph", return_value=False))
        self.stack.enter_context(mock.patch.object(inject, "_is_fake_tensor", return_value=False))
        self.stack.enter_context(mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        self.room = self.stack.enter_context(mock.patch.object(inject, "_native_convrot_has_headroom", return_value=True))
        self.timer = self.stack.enter_context(mock.patch.object(inject, "_time_convrot_forward", side_effect=[10, 20]))
        self.stack.enter_context(mock.patch.object(inject, "_log"))
        self.input = SimpleNamespace(shape=(1, 128, 64), stride=lambda: (8192, 64, 1),
                                     dtype=torch.bfloat16, device=torch.device("cuda:0"), is_cuda=True)
        self.weight = SimpleNamespace(qtype=SimpleNamespace(name="qint8_convrot"), shape=(256, 64))

    def choose(self):
        return inject._prefer_native_convrot_path(self.input, self.weight)

    def test_native_win_is_cached_without_rebenchmarking(self):
        self.assertTrue(self.choose())
        self.assertTrue(self.choose())
        self.assertEqual(self.timer.call_count, 2)

    def test_triton_win_or_close_result_keeps_triton(self):
        for native_ms in (9, 10, 20):
            with self.subTest(native_ms=native_ms):
                inject._CONVROT_BACKENDS.clear()
                self.timer.side_effect = [native_ms, 10]
                self.assertFalse(self.choose())

    def test_choices_are_specific_to_shape_dtype_and_device(self):
        self.assertTrue(self.choose())
        for attr, value in (("shape", (1, 16, 64)), ("dtype", torch.float16), ("device", torch.device("cuda:1"))):
            with self.subTest(attr=attr):
                setattr(self.input, attr, value)
                self.timer.side_effect = [20, 10]
                self.assertFalse(self.choose())
        self.assertEqual(self.timer.call_count, 8)

    def test_low_headroom_never_probes(self):
        self.room.return_value = False
        self.assertFalse(self.choose())
        self.timer.assert_not_called()

    def test_native_choice_can_yield_to_triton_when_residency_grows(self):
        self.assertTrue(self.choose())
        self.room.return_value = False
        self.assertFalse(self.choose())
        self.room.return_value = True
        self.assertTrue(self.choose())
        self.assertEqual(self.timer.call_count, 2)

    def test_failed_probe_does_not_disable_triton_globally(self):
        before = inject._RUNTIME_DISABLED
        self.timer.side_effect = torch.OutOfMemoryError("probe allocation")
        self.assertFalse(self.choose())
        self.assertFalse(self.choose())
        self.assertEqual(inject._RUNTIME_DISABLED, before)
        self.assertEqual(self.timer.call_count, 1)

    def test_raw_cuda_driver_failure_is_not_swallowed(self):
        self.timer.side_effect = RuntimeError("CUDA error: illegal memory access")
        with self.assertRaisesRegex(RuntimeError, "CUDA error"):
            self.choose()

    def test_non_convrot_and_non_inference_calls_are_untouched(self):
        self.weight.qtype.name = "qint8"
        self.assertFalse(self.choose())
        self.weight.qtype.name = "qint8_convrot"
        with torch.enable_grad():
            self.assertFalse(self.choose())
        self.input.is_cuda = False
        self.assertFalse(self.choose())
        self.input.is_cuda = True
        self.input.dtype = torch.float32
        self.assertFalse(self.choose())
        self.timer.assert_not_called()

    def test_compile_capture_and_opt_out_never_probe(self):
        with mock.patch.object(inject, "_is_compiling_graph", return_value=True):
            self.assertFalse(self.choose())
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertFalse(self.choose())
        with mock.patch.dict(os.environ, {"MAESTRO_CONVROT_KERNEL_CHECK": "0"}):
            self.assertFalse(self.choose())
        self.timer.assert_not_called()

    def test_probe_count_is_bounded_and_reset_with_kernel_state(self):
        with mock.patch.object(inject, "_CONVROT_BACKEND_LIMIT", 1):
            self.assertTrue(self.choose())
            self.input.shape = (1, 256, 64)
            self.assertFalse(self.choose())
            self.assertEqual(self.timer.call_count, 2)
        inject._reset_runtime_state(reset_triton_module=False)
        self.assertFalse(inject._CONVROT_BACKENDS)

    def test_headroom_counts_native_temporaries_and_allocator_cache(self):
        # Use real CPU tensors for the size estimate; CUDA readings are mocked.
        x = torch.empty((1, 128, 64), dtype=torch.bfloat16)
        weight = torch.empty((256, 64), dtype=torch.int8)
        with mock.patch.object(torch.cuda, "mem_get_info", return_value=(1024 ** 3, 24 * 1024 ** 3)), \
             mock.patch.object(torch.cuda, "memory_reserved", return_value=0), \
             mock.patch.object(torch.cuda, "memory_allocated", return_value=0):
            self.assertFalse(self.has_headroom(x, weight))
            with mock.patch.object(torch.cuda, "memory_reserved", return_value=2 * 1024 ** 3):
                self.assertTrue(self.has_headroom(x, weight))


if __name__ == "__main__":
    unittest.main()
