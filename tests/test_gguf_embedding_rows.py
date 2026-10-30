from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


class _FakeQType:
    def __init__(self, name, **metadata):
        self.name = name
        self.__dict__.update(metadata)


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_handler():
    """Load the production module with only its optional registration deps stubbed."""
    q2_k = _FakeQType("Q2_K")
    q3_k = _FakeQType("Q3_K")
    q4_k = _FakeQType("Q4_K")
    q5_k = _FakeQType("Q5_K")
    q6_k = _FakeQType("Q6_K")
    q4_0 = _FakeQType("Q4_0")
    q4_1 = _FakeQType("Q4_1")
    q5_0 = _FakeQType("Q5_0")
    q5_1 = _FakeQType("Q5_1")
    q8_0 = _FakeQType("Q8_0")
    f32 = _FakeQType("F32")
    f16 = _FakeQType("F16")
    bf16 = _FakeQType("BF16")

    quant_types = SimpleNamespace(
        Q2_K=q2_k, Q3_K=q3_k, Q4_K=q4_k, Q5_K=q5_k, Q6_K=q6_k,
        Q4_0=q4_0, Q4_1=q4_1, Q5_0=q5_0, Q5_1=q5_1, Q8_0=q8_0,
        F32=f32, F16=f16, BF16=bf16,
    )
    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGMLQuantizationType = quant_types
    fake_gguf.GGML_QUANT_SIZES = {
        q2_k: (256, 84),
        q3_k: (256, 110),
        q4_k: (256, 144),
        q5_k: (256, 176),
        q6_k: (256, 210),
        q4_0: (32, 18),
        q4_1: (32, 20),
        q5_0: (32, 22),
        q5_1: (32, 24),
        q8_0: (32, 34),
    }

    class _QTensor(torch.Tensor):
        def __init__(self, qtype, axis):
            self._qtype = qtype
            self._axis = axis

    class _QModuleMixin:
        pass

    def _register_qmodule(_base):
        return lambda cls: cls

    def _quanto_qtype(name, **metadata):
        return _FakeQType(name, **metadata)

    class _BaseQEmbedding(torch.nn.Embedding):
        def __init__(
            self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
            norm_type=2.0, scale_grad_by_freq=False, sparse=False, device=None,
            dtype=None, weights=None, activations=None, optimizer=None,
            quantize_input=True,
        ):
            super().__init__(
                num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                scale_grad_by_freq, sparse, device=device, dtype=dtype,
            )
            self.weight_qtype = None

    optimum = _package("optimum")
    quanto = _package("optimum.quanto")
    quanto.QModuleMixin = _QModuleMixin
    quanto.register_qmodule = _register_qmodule
    tensor_package = _package("optimum.quanto.tensor")
    qtensor_module = types.ModuleType("optimum.quanto.tensor.qtensor")
    qtensor_module.QTensor = _QTensor
    qtype_module = types.ModuleType("optimum.quanto.tensor.qtype")
    qtype_module.qtype = _quanto_qtype
    qtype_module.qtypes = {}
    mmgp = _package("mmgp")
    mmgp_offload = types.ModuleType("mmgp.offload")
    mmgp_offload.QEmbedding = _BaseQEmbedding

    modules = {
        "gguf": fake_gguf,
        "optimum": optimum,
        "optimum.quanto": quanto,
        "optimum.quanto.tensor": tensor_package,
        "optimum.quanto.tensor.qtensor": qtensor_module,
        "optimum.quanto.tensor.qtype": qtype_module,
        "mmgp": mmgp,
        "mmgp.offload": mmgp_offload,
    }
    path = ROOT / "app" / "shared" / "qtypes" / "gguf.py"
    spec = importlib.util.spec_from_file_location("_maestro_gguf_embedding_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    module._GGUF_CUDA_KERNELS_ENABLED_CACHE = False
    return module, quant_types, fake_gguf


HANDLER, QUANT_TYPES, _FAKE_GGUF = _load_handler()


def _make_embedding(qtype_name: str, *, vocab=11, width=512, dtype=torch.float32):
    qtype = getattr(QUANT_TYPES, qtype_name)
    block_size, type_size = _FAKE_GGUF.GGML_QUANT_SIZES[qtype]
    if width % block_size:
        raise ValueError("fixture width must align to the qtype block size")
    blocks_per_row = width // block_size
    block_count = vocab * blocks_per_row
    raw = (
        (torch.arange(block_count * type_size, dtype=torch.int64) * 37 + 19) % 251
    ).to(torch.uint8).reshape(block_count, type_size)

    if qtype_name == "Q2_K":
        raw[:, -4:-2] = torch.tensor([0, 60], dtype=torch.uint8)
        raw[:, -2:] = 0
    elif qtype_name == "Q4_K":
        raw[:, :2] = torch.tensor([0, 60], dtype=torch.uint8)
        raw[:, 2:4] = 0
    elif qtype_name == "Q8_0":
        raw[:, :2] = torch.tensor([0, 60], dtype=torch.uint8)

    weight = HANDLER.GGUFWeightTensor.create(
        raw_tensor=raw,
        size=(vocab, width),
        stride=(width, 1),
        dtype=dtype,
        tensor_type=qtype,
        tensor_shape=(vocab, width),
    )
    layer = HANDLER.QEmbeddingGGUF(vocab, width, dtype=dtype)
    layer.weight_qtype = HANDLER._GGUF_QTYPE
    layer._gguf_default_dtype = dtype
    layer._parameters["weight"] = weight
    return layer, weight


class GGUFEmbeddingRowsTests(unittest.TestCase):
    def test_q2_k_and_sibling_qtypes_match_full_dequantization_without_full_table_decode(self):
        ids = torch.tensor([[8, 1, 8], [0, 5, 3]], dtype=torch.long)
        for qtype_name in ("Q2_K", "Q4_K", "Q8_0"):
            with self.subTest(qtype=qtype_name):
                layer, weight = _make_embedding(qtype_name)
                dense = weight.dequantize(dtype=torch.float32, device=torch.device("cpu"))
                expected = F.embedding(ids, dense)
                with mock.patch.object(
                    HANDLER,
                    "_gguf_dequantize_tensor",
                    side_effect=AssertionError("whole-table decode must not run"),
                ):
                    actual = layer(ids)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_duplicate_and_invalid_indices(self):
        layer, _ = _make_embedding("Q2_K")
        empty_ids = torch.empty((2, 0), dtype=torch.long)
        with mock.patch.object(
            HANDLER,
            "_gguf_dequantize_tensor",
            side_effect=AssertionError("empty lookup must not decode the table"),
        ):
            empty = layer(empty_ids)
        self.assertEqual(tuple(empty.shape), (2, 0, 512))

        ids = torch.tensor([4, 4, 2, 4], dtype=torch.long)
        dense = layer.weight.dequantize(dtype=torch.float32, device=torch.device("cpu"))
        torch.testing.assert_close(layer(ids), F.embedding(ids, dense), rtol=0, atol=0)
        with self.assertRaises(IndexError):
            layer(torch.tensor([[-1]], dtype=torch.long))
        with self.assertRaises(IndexError):
            layer(torch.tensor([[11]], dtype=torch.long))

    def test_scalar_int32_and_noncontiguous_indices_match_torch_embedding(self):
        layer, weight = _make_embedding("Q2_K")
        dense = weight.dequantize(dtype=torch.float32, device=torch.device("cpu"))

        scalar = torch.tensor(7, dtype=torch.long)
        torch.testing.assert_close(layer(scalar), F.embedding(scalar, dense), rtol=0, atol=0)

        int32_ids = torch.tensor([2, 9, 2], dtype=torch.int32)
        torch.testing.assert_close(layer(int32_ids), F.embedding(int32_ids, dense), rtol=0, atol=0)

        noncontiguous_ids = torch.tensor([[1, 4, 7], [2, 5, 8]], dtype=torch.long).t()
        self.assertFalse(noncontiguous_ids.is_contiguous())
        torch.testing.assert_close(
            layer(noncontiguous_ids),
            F.embedding(noncontiguous_ids, dense),
            rtol=0,
            atol=0,
        )

    def test_selected_dequantization_is_chunk_bounded_and_explicitly_device_placed(self):
        layer, weight = _make_embedding("Q2_K")
        ids = torch.arange(260, dtype=torch.long) % weight._tensor_shape[0]
        decoder = HANDLER._DEQUANTIZE_FUNCTIONS[QUANT_TYPES.Q2_K]
        decoded_block_counts = []

        def record_blocks(blocks, block_size, type_size, dtype):
            decoded_block_counts.append(blocks.shape[0])
            return decoder(blocks, block_size, type_size, dtype)

        with mock.patch.dict(
            HANDLER._DEQUANTIZE_FUNCTIONS,
            {QUANT_TYPES.Q2_K: record_blocks},
        ):
            # This context catches implicit allocations that inherit a CUDA
            # default device. Input and packed tensors remain CPU-resident, so
            # this check performs no CUDA allocation or device initialization.
            with torch.device("cuda"):
                actual = layer(ids)

        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(tuple(actual.shape), (260, 512))
        self.assertTrue(decoded_block_counts)
        self.assertLessEqual(max(decoded_block_counts), 128 * 2)

    def test_unaligned_layout_and_special_embedding_options_use_fallback(self):
        layer, weight = _make_embedding("Q2_K")
        input_ids = torch.tensor([1, 2], dtype=torch.long)
        unaligned = SimpleNamespace(
            _tensor_type=QUANT_TYPES.Q2_K,
            _tensor_shape=(4, 300),
            shape=(4, 300),
            _data=torch.zeros(4 * 84, dtype=torch.uint8),
        )
        self.assertIsNone(layer._dequantize_rows(unaligned, input_ids, torch.float32))
        unsupported = SimpleNamespace(
            _tensor_type=QUANT_TYPES.F32,
            _tensor_shape=(4, 512),
            shape=(4, 512),
            _data=torch.zeros(4 * 1024, dtype=torch.uint8),
        )
        self.assertIsNone(layer._dequantize_rows(unsupported, input_ids, torch.float32))

        dense = weight.dequantize(dtype=torch.float32, device=torch.device("cpu"))
        layer.max_norm = 1.0
        with (
            mock.patch.object(layer, "_dequantize_rows", side_effect=AssertionError("special path bypassed")),
            mock.patch.object(HANDLER, "_gguf_dequantize_tensor", wraps=HANDLER._gguf_dequantize_tensor) as full_decode,
        ):
            actual = layer(input_ids)
        self.assertEqual(full_decode.call_count, 1)
        expected = F.embedding(input_ids, dense, max_norm=1.0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
