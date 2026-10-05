"""CPU checks for Wan2GP asymmetric W4A8 checkpoint handling."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

try:
    import torch

    from mmgp import offload, quant_router
    from shared.qtypes import asym_w4a8_int8 as w4a8
    from shared.qtypes import int8_convrot

    RUNTIME_AVAILABLE = True
except ImportError:
    torch = None
    offload = None
    quant_router = None
    w4a8 = None
    int8_convrot = None
    RUNTIME_AVAILABLE = False


_K = 256
_GROUP_SIZE = 16


def _metadata(layer="linear", *, quant_format="asym_w4a8_int8", group_size=16, convrot=256):
    return {
        "_quantization_metadata": {
            "layers": {
                layer: {
                    "format": quant_format,
                    "group_size": group_size,
                    "convrot_groupsize": convrot,
                }
            }
        }
    }


def _checkpoint(layer="linear", *, rows=5, k=_K, codebook=None, correction=False):
    if k % _GROUP_SIZE or k % 256:
        raise ValueError("fixture dimensions must be divisible by the quantization and ConvRot groups")
    generator = torch.Generator().manual_seed(4817)
    packed = torch.randint(0, 256, (rows, k // 2), generator=generator, dtype=torch.int16).to(torch.uint8).view(torch.int8)
    relative = torch.linspace(0.45, 1.25, rows * (k // _GROUP_SIZE), dtype=torch.float32)
    relative = relative.reshape(rows, k // _GROUP_SIZE).to(torch.float8_e4m3fn)
    channel = torch.linspace(0.025, 0.09, rows, dtype=torch.float32)
    state = {
        layer + ".weight": packed,
        layer + ".weight_s_rel": relative,
        layer + ".weight_s_channel": channel,
    }
    if codebook is not None:
        state[layer + ".weight_codebook"] = codebook.to(torch.float32)
    if correction:
        state[layer + ".weight_correction"] = torch.linspace(
            -0.04, 0.05, (k // _GROUP_SIZE) * rows, dtype=torch.float32
        ).reshape(k // _GROUP_SIZE, rows)
    return state


def _decode_independent(state, layer="linear", group_size=_GROUP_SIZE):
    packed = state[layer + ".weight"].to(torch.uint8)
    rows, packed_k = packed.shape
    indices = torch.empty((rows, packed_k * 2), dtype=torch.long)
    for row in range(rows):
        for column in range(packed_k):
            value = int(packed[row, column])
            indices[row, 2 * column] = value & 0x0F
            indices[row, 2 * column + 1] = value >> 4
    if layer + ".weight_codebook" in state:
        values = state[layer + ".weight_codebook"][indices]
    else:
        values = indices.to(torch.float32) - 8.0
    scales = state[layer + ".weight_s_rel"].float().repeat_interleave(group_size, dim=1)
    quantized = torch.round(values * scales).clamp(-127, 127).to(torch.int8)
    return quantized, state[layer + ".weight_s_channel"]


def _hadamard(size, dtype=torch.float32):
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
    )
    result = h4
    while result.shape[0] < size:
        result = torch.kron(result, h4)
    if result.shape[0] != size:
        raise ValueError(f"Hadamard size {size} must be a power of four")
    return result.mul(size**-0.5)


def _hadamard256():
    return _hadamard(256)


def _linear_reference(source, decoded, channel_scales, *, convrot_group_size, code_correction=None, bias=None, dtype=torch.float32):
    logical_k = int(source.shape[-1])
    source = source.to(dtype=dtype)
    if convrot_group_size:
        rotation = _hadamard(convrot_group_size, dtype=dtype)
        source = source.reshape(-1, logical_k // convrot_group_size, convrot_group_size)
        source = source.matmul(rotation).reshape(-1, logical_k)
    source_2d = source.reshape(-1, logical_k)
    activation_scale = source_2d.abs().amax(dim=-1, keepdim=True).float().div_(127).clamp_(min=1e-30)
    activation = source_2d.div(activation_scale.to(dtype)).round_().clamp_(-128, 127).to(torch.int8)
    accum = activation.to(torch.int32).matmul(decoded.to(torch.int32).t()).float()
    output = (
        accum
        * activation_scale.float()
        * channel_scales.to(torch.float32).unsqueeze(0)
    ).to(dtype)
    if code_correction is not None:
        group_size = logical_k // code_correction.shape[0]
        correction_input = activation.reshape(activation.shape[0], -1, group_size)
        correction_input = correction_input.sum(-1, dtype=torch.int32).to(dtype)
        correction_input.mul_(activation_scale.to(dtype))
        output.addmm_(correction_input, code_correction.to(dtype))
    if bias is not None:
        output.add_(bias.to(dtype))
    return output.reshape(*source.shape[:-1], decoded.shape[0])


@unittest.skipUnless(RUNTIME_AVAILABLE, "MMGP/Quanto runtime dependencies are required")
class TestAsymW4A8Int8(unittest.TestCase):
    def test_detection_and_metadata_conflicts(self):
        state = _checkpoint()
        self.assertTrue(w4a8.detect(state, metadata=_metadata())["matched"])
        self.assertTrue(w4a8.detect(state, metadata=_metadata(convrot=64))["matched"])
        self.assertFalse(
            w4a8.detect(state, metadata=_metadata(quant_format="other_quantizer"))["matched"]
        )
        malformed_metadata = {"_quantization_metadata": {"layers": {"linear": "invalid"}}}
        with self.assertRaisesRegex(ValueError, "expected an object"):
            w4a8.detect(state, metadata=malformed_metadata)
        with self.assertRaisesRegex(ValueError, "group size mismatch"):
            w4a8.detect(state, metadata=_metadata(group_size=8))

        bad_scales = dict(state)
        bad_scales["linear.weight_s_channel"] = state["linear.weight_s_channel"].half()
        with self.assertRaisesRegex(ValueError, "channel scales"):
            w4a8.detect(bad_scales, metadata=_metadata())

    def test_conversion_preserves_tensors_and_refuses_key_collisions(self):
        state = _checkpoint(codebook=torch.arange(-8, 8, dtype=torch.float32), correction=True)
        sentinel_scale = torch.tensor(3.0)
        state["linear.input_scale"] = sentinel_scale
        originals = {key: value for key, value in state.items()}
        converted = w4a8.convert_to_quanto(state, torch.float16, metadata=_metadata())
        result = converted["state_dict"]
        self.assertEqual(converted["quant_map"]["linear"]["weights"], "asym_w4a8_int8")
        self.assertIs(result["linear.weight._data"], originals["linear.weight"])
        self.assertIs(result["linear.weight._s_rel"], originals["linear.weight_s_rel"])
        self.assertIs(result["linear.weight._s_channel"], originals["linear.weight_s_channel"])
        self.assertIs(result["linear.weight._codebook"], originals["linear.weight_codebook"])
        self.assertIs(result["linear.weight._correction"], originals["linear.weight_correction"])
        self.assertIs(result["linear.input_scale"], sentinel_scale)
        self.assertNotIn("linear.weight", result)

        colliding = _checkpoint()
        colliding["linear.weight._data"] = torch.zeros_like(colliding["linear.weight"])
        original_keys = set(colliding)
        original_weight = colliding["linear.weight"]
        with self.assertRaisesRegex(ValueError, "overwrite"):
            w4a8.convert_to_quanto(colliding, torch.float16, metadata=_metadata())
        self.assertEqual(set(colliding), original_keys)
        self.assertIs(colliding["linear.weight"], original_weight)

    def test_packed_nibbles_decode_low_then_high_with_group_scales(self):
        state = _checkpoint(rows=3, codebook=None)
        expected, _ = _decode_independent(state)
        actual = w4a8._decode_w4a8_torch(
            state["linear.weight"],
            state["linear.weight_s_rel"],
            None,
            _GROUP_SIZE,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_portable_linear_matches_independent_dense_reference(self):
        codebook = torch.linspace(-5.25, 6.5, 16, dtype=torch.float32)
        state = _checkpoint(codebook=codebook, correction=True)
        state = {
            "linear.weight": state["linear.weight"],
            "linear.weight_s_rel": state["linear.weight_s_rel"],
            "linear.weight_s_channel": state["linear.weight_s_channel"],
            "linear.weight_codebook": state["linear.weight_codebook"],
            "linear.weight_correction": state["linear.weight_correction"],
        }
        decoded, channel_scales = _decode_independent(state)
        weight = w4a8.AsymW4A8Int8WeightTensor.create(
            state["linear.weight"],
            state["linear.weight_s_rel"],
            channel_scales,
            (decoded.shape[0], _K),
            (_K, 1),
            torch.float32,
            _GROUP_SIZE,
            256,
            codebook=state["linear.weight_codebook"],
            correction=state["linear.weight_correction"],
        )
        source = torch.linspace(-2.4, 2.1, 2 * _K, dtype=torch.float32).reshape(2, _K)
        bias = torch.linspace(-0.17, 0.23, decoded.shape[0], dtype=torch.float32)

        rotation = _hadamard256()
        rotated = source.reshape(-1, 1, 256).matmul(rotation).reshape_as(source)
        activation_scale = rotated.abs().amax(dim=-1, keepdim=True) / 127.0
        activation = torch.round(rotated / activation_scale).clamp(-128, 127).to(torch.int8)
        expected = activation.float().matmul(decoded.float().t())
        expected = expected * activation_scale * channel_scales.unsqueeze(0)
        correction_input = (
            activation.reshape(source.shape[0], -1, _GROUP_SIZE).to(torch.int32).sum(-1).float()
            * activation_scale
        )
        expected = expected + correction_input.matmul(state["linear.weight_correction"])
        expected = expected + bias

        actual = w4a8._w4a8_linear(source, weight, bias)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_triton_backend_uses_local_return_value_api_and_chunks_outputs(self):
        state = _checkpoint(rows=4)
        decoded, channel_scales = _decode_independent(state)
        weight = w4a8.AsymW4A8Int8WeightTensor.create(
            state["linear.weight"],
            state["linear.weight_s_rel"],
            channel_scales,
            (decoded.shape[0], _K),
            (_K, 1),
            torch.float32,
            _GROUP_SIZE,
            0,
        )

        class Backend:
            def __init__(self):
                self.calls = 0

            def scaled_int8_mm(self, activation, weights, activation_scale, weight_scale, out_dtype=None):
                self.calls += 1
                accum = activation.int().matmul(weights.int().t()).float()
                return (
                    accum
                    * activation_scale.float().unsqueeze(1)
                    * weight_scale.float().unsqueeze(0)
                ).to(out_dtype)

        backend = Backend()
        source = torch.linspace(-1.0, 1.0, 3 * _K, dtype=torch.float32).reshape(3, _K)
        with mock.patch.object(w4a8, "_kernel_backend", return_value=backend), \
             mock.patch.object(w4a8, "_workspace_rows", return_value=2), \
             mock.patch.object(
                 w4a8,
                 "_decode_w4a8_triton",
                 side_effect=lambda packed, scales, codebook_arg, group, output: w4a8._decode_w4a8_torch(
                     packed, scales, codebook_arg, group, output
                 ),
             ):
            actual = w4a8._w4a8_linear(source, weight)

        self.assertEqual(backend.calls, 2)
        activation, activation_scale = w4a8._quantize_activation(source)
        expected = activation.int().matmul(decoded.int().t()).float()
        expected *= activation_scale.unsqueeze(1) * channel_scales.unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_kernel_availability_uses_local_no_argument_api(self):
        from shared.kernels import quanto_int8_triton

        with mock.patch.object(w4a8, "_TRITON_AVAILABLE", True), \
             mock.patch.object(torch.version, "cuda", "13.0"), \
             mock.patch("torch.cuda.device"), \
             mock.patch.object(quanto_int8_triton, "is_available", return_value=True) as available:
            backend = w4a8._kernel_backend(torch.device("cuda:0"))

        self.assertIs(backend, quanto_int8_triton)
        available.assert_called_once_with()

    def test_cuda_triton_decode_and_chunked_forward_match_cpu_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        if not w4a8._TRITON_AVAILABLE:
            self.skipTest("Triton is unavailable")

        device = torch.device("cuda:0")
        backend = w4a8._kernel_backend(device)
        if backend is None:
            self.skipTest("Maestro's INT8 Triton backend is unavailable")

        rows = 9
        outputs = 24
        state = _checkpoint(
            rows=outputs,
            codebook=torch.linspace(-6.75, 7.0, 16, dtype=torch.float32),
            correction=True,
        )
        decoded, channel_scales = _decode_independent(state)
        decoded_cuda = torch.empty((outputs, _K), dtype=torch.int8, device=device)
        w4a8._decode_w4a8_triton(
            state["linear.weight"].to(device),
            state["linear.weight_s_rel"].to(device),
            state["linear.weight_codebook"].to(device),
            _GROUP_SIZE,
            decoded_cuda,
        )
        torch.testing.assert_close(decoded_cuda.cpu(), decoded, rtol=0, atol=0)

        source = torch.linspace(-2.3, 1.9, rows * _K, dtype=torch.float32).reshape(rows, _K)
        bias = torch.linspace(-0.21, 0.19, outputs, dtype=torch.float32)
        correction = state["linear.weight_correction"]
        for convrot_group_size in (64, 256):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(convrot=convrot_group_size, dtype=dtype), torch.no_grad():
                    weight = w4a8.AsymW4A8Int8WeightTensor.create(
                        state["linear.weight"],
                        state["linear.weight_s_rel"],
                        channel_scales,
                        (outputs, _K),
                        (_K, 1),
                        dtype,
                        _GROUP_SIZE,
                        convrot_group_size,
                        codebook=state["linear.weight_codebook"],
                        correction=correction,
                        device=device,
                    )
                    expected = _linear_reference(
                        source,
                        decoded,
                        channel_scales,
                        convrot_group_size=convrot_group_size,
                        code_correction=correction,
                        bias=bias,
                        dtype=dtype,
                    )
                    with mock.patch.object(w4a8, "_workspace_rows", return_value=7):
                        actual = w4a8._w4a8_linear(
                            source.to(device=device, dtype=dtype),
                            weight,
                            bias.to(device=device, dtype=dtype),
                        )
                    tolerance = 0.035 if dtype == torch.bfloat16 else 0.002
                    torch.testing.assert_close(
                        actual.cpu().float(),
                        expected.float(),
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_fused_qkv_split_splits_every_per_row_field_and_shares_metadata(self):
        state = _checkpoint("attention.qkv", rows=12, codebook=torch.arange(-8, 8), correction=True)
        original = dict(state)
        converted = w4a8.convert_to_quanto(
            state,
            torch.float16,
            metadata=_metadata("attention.qkv"),
        )
        raw = converted["state_dict"]
        split, bases = w4a8.split_fused_weights(
            raw,
            {
                "qkv": {
                    "mapped_modules": ["q_proj", "k_proj", "v_proj"],
                    "size_field": "weight._data",
                    "split_dim": 0,
                    "split_sizes": [4, 4, 4],
                }
            },
            quantization_map=converted["quant_map"],
            verboseLevel=0,
        )
        self.assertEqual(bases, ["attention.qkv"])
        self.assertNotIn("attention.qkv.qweight", split)
        for index, target in enumerate(("q_proj", "k_proj", "v_proj")):
            base = "attention." + target
            rows = slice(index * 4, (index + 1) * 4)
            torch.testing.assert_close(split[base + ".weight._data"], original["attention.qkv.weight"][rows])
            torch.testing.assert_close(split[base + ".weight._s_rel"], original["attention.qkv.weight_s_rel"][rows])
            torch.testing.assert_close(split[base + ".weight._s_channel"], original["attention.qkv.weight_s_channel"][rows])
            torch.testing.assert_close(split[base + ".weight._correction"], original["attention.qkv.weight_correction"][:, rows])
            self.assertIs(split[base + ".weight._codebook"], original["attention.qkv.weight_codebook"])
            self.assertEqual(split[base + ".weight._group_size"].item(), 16)
            self.assertEqual(split[base + ".weight._convrot_group_size"].item(), 256)

    def test_quanto_weight_cpu_offload_roundtrip_and_native_linear_route(self):
        state = _checkpoint(codebook=torch.arange(-8, 8), correction=True)
        decoded, _ = _decode_independent(state)
        weight = w4a8.AsymW4A8Int8WeightTensor.create(
            state["linear.weight"],
            state["linear.weight_s_rel"],
            state["linear.weight_s_channel"],
            (decoded.shape[0], _K),
            (_K, 1),
            torch.float32,
            _GROUP_SIZE,
            256,
            codebook=state["linear.weight_codebook"],
            correction=state["linear.weight_correction"],
        )
        copied = torch.ops.aten._to_copy.default(weight, device=torch.device("cpu"))
        detached = copied.detach()
        self.assertIsInstance(copied, w4a8.AsymW4A8Int8WeightTensor)
        self.assertIsInstance(detached, w4a8.AsymW4A8Int8WeightTensor)
        for name, original in weight.get_quantized_subtensors():
            torch.testing.assert_close(dict(detached.get_quantized_subtensors())[name], original)
        self.assertEqual(detached.__tensor_flatten__()[1]["convrot_group_size"], "256")

        quant_router.register_handler("shared.qtypes.asym_w4a8_int8")
        model = torch.nn.Module()
        model.linear = torch.nn.Linear(_K, decoded.shape[0], bias=True, dtype=torch.float32)
        load_state = _checkpoint(codebook=torch.arange(-8, 8), correction=True)
        load_state["linear.bias"] = torch.zeros(decoded.shape[0], dtype=torch.float32)
        offload.load_model_data(model, (load_state, None), default_dtype=torch.float32, verboseLevel=0)
        # MMGP 3.7.6 retains its Quanto router shell, copies the handler's
        # native implementation onto it, and uses the handler's packed weight.
        self.assertEqual(type(model.linear).__name__, "QLinearQuantoRouter")
        self.assertIsInstance(model.linear.weight, w4a8.AsymW4A8Int8WeightTensor)
        self.assertTrue(model.linear._mm_requires_native_linear_forward)
        self.assertIs(model.linear._router_forward_impl, w4a8.QLinearAsymW4A8Int8.forward)
        sample = torch.linspace(-1.0, 1.0, _K, dtype=torch.float32).reshape(1, _K)
        base_output = model.linear(sample).detach()
        self.assertEqual(base_output.shape, (1, decoded.shape[0]))

        model._loras_active_adapters = ["test"]
        model._loras_scaling = {}
        lora_a = torch.full((2, _K), 0.0125, dtype=torch.float32)
        lora_b = torch.arange(2 * decoded.shape[0], dtype=torch.float32).reshape(decoded.shape[0], 2) / 100.0
        model.linear._mm_lora_old_forward = model.linear.forward
        model.linear._mm_lora_model = model
        model.linear._mm_lora_data = {
            "test_GPU": [lora_a, lora_b, None, None, 1.0, {"type": "lora"}]
        }
        model.linear._mm_manager = SimpleNamespace(_get_lora_scaling=lambda *args: 1.0)
        self.assertEqual(int8_convrot.install_native_lora_forwards(model), 1)
        expected = base_output + (sample @ lora_a.t()) @ lora_b.t()
        torch.testing.assert_close(model.linear(sample), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
