import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_APP = str(Path(__file__).resolve().parents[1] / "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from models.hy_nomas.block_scaled import (
    BlockScaledWeight,
    _FP8_BLOCKS,
    _INT8_BLOCKS,
    block_scaled_linear,
    convert_to_quanto,
    dequantize_blocks,
    detect,
    QLinearBlockScaled,
)


def test_dit_int8_header_format_is_detected_and_converted():
    data = torch.randint(-120, 120, (4, 256), dtype=torch.int8)
    scales = torch.rand((4, 2), dtype=torch.float32) + 0.1
    state = {"layer.weight_q": data, "layer.weight_scale": scales, "layer.bias": torch.zeros(4)}

    found = detect(state, metadata={
        "format": "nomas-int8-bf16-single-v1",
        "group_size": "128",
        "quantization": "symmetric_int8_absmax",
        "compute_dtype": "bfloat16",
        "source_sha256": "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70",
        "layers": '{"layer": [4, 256]}',
    })
    assert found["matched"]
    assert found["details"]["formats"] == ["nomas-int8-bf16-single-v1"]

    converted = convert_to_quanto(state.copy(), torch.bfloat16, detection=found)
    assert converted["quant_map"] == {"layer": {"weights": "int8_blocks", "activations": "none"}}
    assert torch.equal(converted["state_dict"]["layer.weight._data"], data)
    assert torch.equal(converted["state_dict"]["layer.weight._scale"], scales)


def test_pixelumm_fp8_header_format_is_detected_and_converted():
    data = torch.randn((8, 256), dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.rand((1, 2), dtype=torch.float32) + 0.1
    state = {"layer.fp8_weight": data, "layer.fp8_scale": scales}

    found = detect(state, metadata={
        "format": "pixelumm_fp8_bf16_single_v1",
        "block_size": "128",
        "weight_dtype": "float8_e4m3fn",
        "scale_dtype": "float32",
        "source_sha256": "0bbfe8fd44bd2ebf1059fe65079caba50fd6f52ea8e1b5b35393e3050b9b8378",
    })
    assert found["matched"]
    converted = convert_to_quanto(state.copy(), torch.bfloat16, detection=found)
    assert converted["quant_map"] == {"layer": {"weights": "float8_e4m3fn_blocks", "activations": "none"}}
    assert torch.equal(converted["state_dict"]["layer.weight._data"].view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(converted["state_dict"]["layer.weight._scale"], scales)


def test_quantization_metadata_and_scale_shapes_are_required():
    data = torch.ones((4, 256), dtype=torch.int8)
    scales = torch.ones((4, 1), dtype=torch.float32)
    state = {"layer.weight_q": data, "layer.weight_scale": scales}

    valid_metadata = {
        "format": "nomas-int8-bf16-single-v1",
        "group_size": "128",
        "quantization": "symmetric_int8_absmax",
        "compute_dtype": "bfloat16",
        "source_sha256": "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70",
        "layers": '{"layer": [4, 256]}',
    }
    assert not detect(state, metadata={"format": "unrelated-format"})["matched"]
    assert not detect(state, metadata={"format": "nomas-int8-bf16-single-v1"})["matched"]
    assert not detect(state, metadata={**valid_metadata, "layers": '{"layer": [4, 128]}'} )["matched"]
    assert not detect({"layer.weight_q": data, "layer.weight_scale": torch.ones((3, 2))},
                      metadata=valid_metadata)["matched"]


def test_int8_block_scale_linear_matches_explicit_dequantization():
    torch.manual_seed(23)
    data = torch.randint(-30, 30, (5, 256), dtype=torch.int8)
    scales = torch.rand((5, 2), dtype=torch.float32) + 0.05
    bias = torch.randn((5,), dtype=torch.bfloat16)
    weight = BlockScaledWeight.create(data, scales, (5, 256), (256, 1), torch.bfloat16)
    inputs = torch.randn((2, 3, 256), dtype=torch.float32)

    actual = block_scaled_linear(weight, inputs, bias)
    expected_weight = dequantize_blocks(data, scales, torch.bfloat16)
    expected = F.linear(inputs.to(torch.bfloat16), expected_weight, bias)
    assert actual.shape == (2, 3, 5)
    assert torch.equal(actual, expected)
    assert weight._qtype == _INT8_BLOCKS


def test_fp8_block_scale_linear_matches_explicit_dequantization():
    torch.manual_seed(29)
    data = torch.randn((8, 256), dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.rand((1, 2), dtype=torch.float32) + 0.05
    bias = torch.randn((8,), dtype=torch.bfloat16)
    weight = BlockScaledWeight.create(data, scales, (8, 256), (256, 1), torch.bfloat16)
    inputs = torch.randn((2, 256), dtype=torch.float32)

    actual = block_scaled_linear(weight, inputs, bias)
    expected_weight = dequantize_blocks(data, scales, torch.bfloat16)
    expected = F.linear(inputs.to(torch.bfloat16), expected_weight, bias)
    assert actual.shape == (2, 8)
    assert torch.equal(actual, expected)
    assert weight._qtype == _FP8_BLOCKS


def test_block_scaled_handler_registers_and_loads_through_mmgp_quant_router():
    from mmgp import quant_router

    quant_router.register_handler("models.hy_nomas.block_scaled")
    assert quant_router._build_qmodule_cache().get(_INT8_BLOCKS) is QLinearBlockScaled

    data = torch.randint(-24, 24, (4, 256), dtype=torch.int8)
    scales = torch.rand((4, 2), dtype=torch.float32) + 0.05
    bias = torch.randn((4,), dtype=torch.bfloat16)
    metadata = {
        "format": "nomas-int8-bf16-single-v1",
        "group_size": "128",
        "quantization": "symmetric_int8_absmax",
        "compute_dtype": "bfloat16",
        "source_sha256": "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70",
        "layers": '{"0": [4, 256]}',
    }
    converted = quant_router.detect_and_convert({
        "0.weight_q": data,
        "0.weight_scale": scales,
        "0.bias": bias,
    }, torch.bfloat16, metadata=metadata)
    assert converted["kind"] == "block_scaled"

    model = torch.nn.Sequential(
        QLinearBlockScaled(256, 4, bias=True, dtype=torch.bfloat16, weights=_INT8_BLOCKS, activations=None)
    )
    missing, unexpected = model.load_state_dict(converted["state_dict"], strict=False)
    assert not missing
    assert not unexpected

    inputs = torch.randn((2, 256), dtype=torch.bfloat16)
    actual = model(inputs)
    expected = F.linear(inputs, dequantize_blocks(data, scales, torch.bfloat16), bias)
    assert torch.equal(actual, expected)
