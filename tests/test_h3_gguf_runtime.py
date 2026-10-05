"""Tiny real-GGUF integration checks for the MiniMax H3 MMGP runtime."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
import sys

if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

_IMPORT_ERROR = None
try:
    import gguf
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mmgp import offload, quant_router
    from models.minimax_h3.transformer import MiniMaxH3Attention
    from shared.qtypes import gguf as gguf_handler
except Exception as exc:  # Optional runtime dependencies are absent in CPU-only test envs.
    _IMPORT_ERROR = exc


_QUANT_TYPES = (
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
)
_HIDDEN = 256
_HEADS = 2
_HEAD_DIM = 128
_EPS = 1e-6


def _runtime_model(dtype=None):
    """Build only the H3 module paths covered by the small fixture."""

    if dtype is None:
        dtype = torch.float32

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MiniMaxH3Attention(
                _HIDDEN,
                _HEADS,
                _HEAD_DIM,
                eps=_EPS,
                dtype=dtype,
            )

    class FixtureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])
            self.register_buffer("adaln_t_table", torch.zeros((1025, 8), dtype=torch.float32))
            self.gguf_refs = torch.nn.ModuleDict(
                {
                    name: torch.nn.Linear(256, 3, bias=False, dtype=dtype)
                    for name in _QUANT_TYPES
                }
            )

    return FixtureModel()


def _dense_qkv(layout: str) -> np.ndarray:
    """Create visibly different Q/K/V row groups for layout interpretation."""

    rows = np.arange(3 * _HEADS * _HEAD_DIM, dtype=np.float32).reshape(3, _HEADS, _HEAD_DIM, 1)
    cols = np.arange(_HIDDEN, dtype=np.float32).reshape(1, 1, 1, _HIDDEN)
    grouped = np.ascontiguousarray(
        (np.sin(rows * 0.031 + cols * 0.017) * 0.28 + (rows // (_HEADS * _HEAD_DIM) - 1) * 0.19),
        dtype=np.float32,
    ).reshape(3 * _HEADS * _HEAD_DIM, _HIDDEN)
    if layout == "grouped":
        return grouped
    if layout == "interleaved":
        return np.ascontiguousarray(
            grouped.reshape(3, _HEADS, _HEAD_DIM, _HIDDEN)
            .transpose(1, 0, 2, 3)
            .reshape(3 * _HEADS * _HEAD_DIM, _HIDDEN)
        )
    raise ValueError(layout)


def _nontrivial_raw_quant(name: str, rows: int, width: int) -> np.ndarray:
    """Construct deterministic packed blocks with finite, nonzero scale fields."""

    qtype = getattr(gguf.GGMLQuantizationType, name)
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    blocks_per_row = width // block_size
    raw = np.arange(rows * blocks_per_row * type_size, dtype=np.uint32)
    raw = ((raw * 73 + 19) % 251).astype(np.uint8).reshape(rows, blocks_per_row * type_size)

    # gguf.quants.dequantize is the independent reference in this test. Setting
    # the block scales to exact FP16 values avoids arbitrary random NaN/Inf
    # exponents while the quantized payload bytes remain varied.
    if name in {"Q2_K", "Q4_K", "Q5_K"}:
        raw[:, -4:-2] = np.array([0x00, 0x3C], dtype=np.uint8)  # d = 1
        raw[:, -2:] = 0  # dmin = 0
    elif name in {"Q3_K", "Q6_K"}:
        raw[:, -2:] = np.array([0x00, 0x3C], dtype=np.uint8)  # d = 1
    else:
        for start in range(0, raw.shape[1], type_size):
            raw[:, start : start + 2] = np.array([0x00, 0x3C], dtype=np.uint8)
            if name in {"Q4_1", "Q5_1"}:
                raw[:, start + 2 : start + 4] = 0  # m = 0
    return raw


def _make_fixture(path: Path, layout: str):
    writer = gguf.GGUFWriter(path, "minimax_h3")
    writer.add_string("qkv_layout", layout)

    table = np.linspace(-1.0, 1.0, 1025 * 8, dtype=np.float32).reshape(1025, 8)
    q_norm = np.linspace(0.55, 1.25, _HEAD_DIM, dtype=np.float32)
    k_norm = np.linspace(1.2, 0.6, _HEAD_DIM, dtype=np.float32)
    out_proj = np.sin(
        np.arange(_HIDDEN * _HIDDEN, dtype=np.float32).reshape(_HIDDEN, _HIDDEN) * 0.013
    ) * 0.035

    writer.add_tensor("adaln_t_table", np.ascontiguousarray(table))
    writer.add_tensor("blocks.0.attn.q_norm.weight", q_norm)
    writer.add_tensor("blocks.0.attn.k_norm.weight", k_norm)
    writer.add_tensor("blocks.0.attn.out_proj.weight", np.ascontiguousarray(out_proj))

    qkv_dense = _dense_qkv(layout)
    q4_0 = gguf.GGMLQuantizationType.Q4_0
    qkv_raw = gguf.quants.quantize(qkv_dense, q4_0)
    writer.add_tensor("blocks.0.attn.qkv_proj.weight", qkv_raw, raw_dtype=q4_0)

    reference_raw = {}
    for index, name in enumerate(_QUANT_TYPES):
        qtype = getattr(gguf.GGMLQuantizationType, name)
        if name in {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"}:
            dense = np.linspace(-0.7 + index * 0.03, 0.9 - index * 0.02, 3 * 256, dtype=np.float32).reshape(3, 256)
            packed = gguf.quants.quantize(dense, qtype)
        else:
            packed = _nontrivial_raw_quant(name, rows=3, width=256)
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        reference_raw[name] = packed.copy()
        writer.add_tensor(f"gguf_refs.{name}.weight", packed, raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    expected = {
        "table": table,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "out_proj": out_proj,
        "qkv_dense": qkv_dense,
        "qkv_raw": qkv_raw,
        "qkv_dequant": gguf.quants.dequantize(qkv_raw, q4_0),
        "reference_raw": reference_raw,
    }
    return expected


def _reference_attention(model, inputs: torch.Tensor, qkv_weight: np.ndarray, layout: str) -> torch.Tensor:
    """Independent dense attention baseline using official GGUF dequantized rows."""

    device = inputs.device
    dtype = inputs.dtype
    fused = F.linear(inputs, torch.as_tensor(qkv_weight, device=device, dtype=dtype))
    batch, length, _ = fused.shape
    if layout == "grouped":
        query, key, value = fused.chunk(3, dim=-1)
        shape = (batch, length, _HEADS, _HEAD_DIM)
        query, key, value = query.reshape(shape), key.reshape(shape), value.reshape(shape)
    else:
        grouped = fused.reshape(batch, length, _HEADS, 3, _HEAD_DIM)
        query = grouped[:, :, :, 0, :].contiguous()
        key = grouped[:, :, :, 1, :].contiguous()
        value = grouped[:, :, :, 2, :].contiguous()

    q_scale = torch.as_tensor(
        model.blocks[0].attn.q_norm.weight.detach(), device=device, dtype=dtype
    )
    k_scale = torch.as_tensor(
        model.blocks[0].attn.k_norm.weight.detach(), device=device, dtype=dtype
    )
    query = F.rms_norm(query, (_HEAD_DIM,), q_scale, _EPS)
    key = F.rms_norm(key, (_HEAD_DIM,), k_scale, _EPS)
    attended = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )
    attended = attended.transpose(1, 2).reshape(batch, length, _HIDDEN)
    return F.linear(attended, model.blocks[0].attn.out_proj.weight)


def _detach_fixture_storage(model: torch.nn.Module) -> None:
    """Copy the tiny mapped test payload so Windows can remove its fixture."""

    gguf_handler.materialize_module_source_tensors(model)
    for parameter in model.parameters():
        if isinstance(parameter, gguf_handler.GGUFWeightTensor):
            parameter._data = parameter._data.clone()


class H3GGUFRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"H3 GGUF runtime dependencies unavailable: {_IMPORT_ERROR}")
        cls._previous_gguf_default_dtype = getattr(gguf_handler, "_GGUF_DEFAULT_DTYPE", None)
        quant_router.register_handler("shared.qtypes.gguf")
        quant_router.register_file_extension("gguf", gguf_handler)

    @classmethod
    def tearDownClass(cls):
        if _IMPORT_ERROR is None:
            gguf_handler._GGUF_DEFAULT_DTYPE = cls._previous_gguf_default_dtype

    def _load_fixture(self, path: Path, layout: str, dtype=None):
        if dtype is None:
            dtype = torch.float32
        expected = _make_fixture(path, layout)
        model = _runtime_model(dtype=dtype)
        offload.load_model_data(
            model,
            str(path),
            writable_tensors=True,
            default_dtype=dtype,
            ignore_unused_weights=False,
            verboseLevel=0,
        )
        _detach_fixture_storage(model)
        model.blocks[0].attn.set_qkv_layout(layout)
        return model, expected

    def _assert_loaded_packed_types(self, model, expected):
        attention = model.blocks[0].attn
        self.assertTrue(hasattr(attention, "qkv_proj"))
        self.assertFalse(hasattr(attention, "q_proj"))
        self.assertIsInstance(attention.qkv_proj.weight, gguf_handler.GGUFWeightTensor)
        self.assertEqual(tuple(attention.qkv_proj.weight._tensor_shape), (768, 256))
        self.assertEqual(attention.qkv_proj.weight._tensor_type.name, "Q4_0")

        table = model.adaln_t_table
        self.assertEqual(tuple(table.shape), (1025, 8))
        self.assertEqual(table.dtype, torch.float32)
        torch.testing.assert_close(table.detach(), torch.from_numpy(expected["table"]))

        for name in _QUANT_TYPES:
            packed = model.gguf_refs[name].weight
            self.assertIsInstance(packed, gguf_handler.GGUFWeightTensor, msg=name)
            self.assertEqual(packed._tensor_type.name, name)
            self.assertEqual(tuple(packed._tensor_shape), (3, 256))
            actual = packed.dequantize(dtype=torch.float32).cpu()
            reference = torch.from_numpy(gguf.quants.dequantize(expected["reference_raw"][name], packed._tensor_type))
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5, msg=name)

            # MMGP's packed-tensor offload contract keeps the qtype wrapper and
            # raw subtensor intact across a device copy and detach operation.
            detached = torch.ops.aten._to_copy.default(
                packed, device=torch.device("cpu")
            ).detach()
            self.assertIsInstance(detached, gguf_handler.GGUFWeightTensor, msg=name)
            self.assertEqual(detached._tensor_type.name, name)
            self.assertEqual(tuple(detached._tensor_shape), (3, 256))
            torch.testing.assert_close(
                dict(detached.get_quantized_subtensors())["data"],
                dict(packed.get_quantized_subtensors())["data"],
                rtol=0,
                atol=0,
                msg=name,
            )

    def test_grouped_fused_h3_load_cuda_linear_and_offload_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="h3-gguf-runtime-") as temp:
            root = Path(temp)
            model, expected = self._load_fixture(root / "tiny-grouped.gguf", "grouped")
            self._assert_loaded_packed_types(model, expected)

            inputs = torch.linspace(-0.5, 0.5, 3 * _HIDDEN, dtype=torch.float32).reshape(1, 3, _HIDDEN)
            actual = model.blocks[0].attn(inputs)
            baseline = _reference_attention(model, inputs, expected["qkv_dequant"], "grouped")
            torch.testing.assert_close(actual, baseline, rtol=2e-4, atol=2e-4)

            if torch.cuda.is_available():
                # Exercise every registered GGUF decoder through MMGP's
                # actual CUDA Linear route against official dequantized rows.
                cuda_inputs = torch.linspace(-0.5, 0.5, 256, dtype=torch.float32, device="cuda").reshape(1, 256)
                for name in _QUANT_TYPES:
                    layer = model.gguf_refs[name]
                    packed = layer.weight
                    packed_cuda = packed.to(device="cuda")
                    packed_back = packed_cuda.to(device="cpu").detach()
                    self.assertIsInstance(packed_back, gguf_handler.GGUFWeightTensor, msg=name)
                    self.assertEqual(packed_back._tensor_type.name, name)
                    torch.testing.assert_close(
                        dict(packed_back.get_quantized_subtensors())["data"],
                        dict(packed.get_quantized_subtensors())["data"],
                        rtol=0,
                        atol=0,
                        msg=name,
                    )
                    output = layer(cuda_inputs)
                    reference = torch.from_numpy(
                        gguf.quants.dequantize(expected["reference_raw"][name], packed._tensor_type)
                    ).to(device="cuda", dtype=torch.float32)
                    dense_output = F.linear(cuda_inputs, reference)
                    self.assertEqual(output.device.type, "cuda", msg=name)
                    torch.testing.assert_close(output, dense_output, rtol=3e-2, atol=3e-2, msg=name)

                # H3 commonly computes packed projections with BF16 inputs;
                # the AdaLN table itself must retain its F32 checkpoint type.
                bf16_model, bf16_expected = self._load_fixture(
                    root / "tiny-grouped.gguf", "grouped", dtype=torch.bfloat16
                )
                self.assertEqual(bf16_model.adaln_t_table.dtype, torch.float32)
                torch.testing.assert_close(
                    bf16_model.adaln_t_table.detach(),
                    torch.from_numpy(bf16_expected["table"]),
                )
                bf16_inputs = torch.linspace(
                    -0.5, 0.5, 256, dtype=torch.float32, device="cuda"
                ).to(torch.bfloat16).reshape(1, 256)
                for name in _QUANT_TYPES:
                    packed = bf16_model.gguf_refs[name].weight
                    output = bf16_model.gguf_refs[name](bf16_inputs)
                    reference = torch.from_numpy(
                        gguf.quants.dequantize(bf16_expected["reference_raw"][name], packed._tensor_type)
                    ).to(device="cuda", dtype=torch.bfloat16)
                    dense_output = F.linear(bf16_inputs, reference)
                    self.assertEqual(output.dtype, torch.bfloat16, msg=name)
                    self.assertTrue(torch.isfinite(output).all().item(), msg=name)
                    torch.testing.assert_close(output, dense_output, rtol=5e-2, atol=5e-2, msg=name)
                del bf16_model

            del model
            gc.collect()

    def test_interleaved_fused_h3_attention_matches_independent_baseline(self):
        with tempfile.TemporaryDirectory(prefix="h3-gguf-runtime-") as temp:
            model, expected = self._load_fixture(Path(temp) / "tiny-interleaved.gguf", "interleaved")
            self._assert_loaded_packed_types(model, expected)
            inputs = torch.linspace(0.4, -0.4, 1 * 4 * _HIDDEN, dtype=torch.float32).reshape(1, 4, _HIDDEN)
            actual = model.blocks[0].attn(inputs)
            baseline = _reference_attention(model, inputs, expected["qkv_dequant"], "interleaved")
            torch.testing.assert_close(actual, baseline, rtol=2e-4, atol=2e-4)
            del model
            gc.collect()


if __name__ == "__main__":
    unittest.main()
