"""H3 INT8 headers may carry complete ConvRot metadata without Comfy tensors."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.h3_checkpoint_import import H3CheckpointError, inspect_h3_header  # noqa: E402
from test_h3_checkpoint_import import QUANT_SUFFIXES, make_header  # noqa: E402


def metadata_only_int8_header(group_size=256):
    header, payloads = make_header(
        compressed=True,
        quantization="int8",
        native="ref2va",
        qkv="grouped",
        group_size=group_size,
    )
    for name in tuple(header):
        if name.endswith(".comfy_quant"):
            del header[name]
    payloads = {"adaln_t_table": payloads["adaln_t_table"]}

    metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
    for descriptor in metadata["layers"].values():
        descriptor.pop("orig_dtype", None)
    header["__metadata__"]["_quantization_metadata"] = json.dumps(
        metadata, separators=(",", ":")
    )
    header["__metadata__"].pop("quantization.bits", None)
    return header, payloads


def inspect(header, payloads, *, reader=True):
    tensor_reader = payloads.__getitem__ if reader else None
    return inspect_h3_header(
        header,
        tensor_reader=tensor_reader,
        sampling_profile="standard",
        native_workflow="ref2va",
    )


class H3HeaderQuantizationMetadataTests(unittest.TestCase):
    def test_complete_metadata_only_int8_header_reads_only_curve_payload(self):
        header, payloads = metadata_only_int8_header()
        reads = []

        profile = inspect_h3_header(
            header,
            tensor_reader=lambda key: reads.append(key) or payloads[key],
            sampling_profile="standard",
            native_workflow="ref2va",
        )

        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["quantization_format"], "int8_tensorwise")
        self.assertEqual(profile["quantized_layers"], 200)
        self.assertTrue(profile["convrot"])
        self.assertEqual(profile["convrot_group_size"], 256)
        self.assertEqual(profile["qkv_layout"], "grouped")
        self.assertFalse(any(name.endswith(".comfy_quant") for name in header))
        self.assertEqual(reads, ["adaln_t_table"])

    def test_metadata_only_full_checkpoint_needs_no_small_payload_reader(self):
        header, _ = make_header(
            compressed=False,
            quantization="int8",
            native="ref2va",
            qkv="grouped",
        )
        for name in tuple(header):
            if name.endswith(".comfy_quant"):
                del header[name]

        profile = inspect(header, {}, reader=False)

        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["quantization_format"], "int8_tensorwise")
        self.assertEqual(profile["quantized_layers"], 200)

    def test_metadata_requires_all_layers_and_one_supported_group(self):
        header, payloads = metadata_only_int8_header()
        metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
        metadata["layers"].pop("blocks.49.mlp.fc2")
        header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)
        with self.assertRaises(H3CheckpointError) as missing:
            inspect(header, payloads)
        self.assertEqual(missing.exception.code, "incomplete_quantization_metadata")

        header, payloads = metadata_only_int8_header()
        metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
        metadata["layers"]["blocks.49.mlp.fc2"]["convrot_groupsize"] = 64
        header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)
        with self.assertRaises(H3CheckpointError) as conflict:
            inspect(header, payloads)
        self.assertEqual(conflict.exception.code, "quantization_conflict")

        header, payloads = metadata_only_int8_header(group_size=128)
        with self.assertRaises(H3CheckpointError) as unsupported:
            inspect(header, payloads)
        self.assertEqual(unsupported.exception.code, "unsupported_convrot_group_size")

    def test_metadata_rejects_noninteger_or_conflicting_groups_and_unknown_formats(self):
        invalid_updates = (
            {"convrot_groupsize": "256"},
            {"convrot_groupsize": 256.0},
            {"group_size": 64},
            {"format": "int8_convrot_custom"},
        )
        for updates in invalid_updates:
            with self.subTest(updates=updates):
                header, payloads = metadata_only_int8_header()
                metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
                metadata["layers"]["blocks.0.attn.qkv_proj"].update(updates)
                header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)
                with self.assertRaises(H3CheckpointError):
                    inspect(header, payloads)

        header, payloads = metadata_only_int8_header()
        metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
        metadata["layers"]["blocks.0.attn.qkv_proj"]["convrot"] = "true"
        header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)
        with self.assertRaises(H3CheckpointError):
            inspect(header, payloads)

    def test_metadata_only_still_checks_late_block_weight_and_row_scale_shapes(self):
        header, payloads = metadata_only_int8_header()
        bad = copy.deepcopy(header)
        bad["blocks.49.mlp.fc2.weight"]["shape"] = [5375, 14336]
        with self.assertRaises(H3CheckpointError) as weight_shape:
            inspect(bad, payloads)
        self.assertEqual(weight_shape.exception.code, "tensor_shape_mismatch")

        bad = copy.deepcopy(header)
        bad["blocks.49.mlp.fc2.weight_scale"]["shape"] = [5376]
        with self.assertRaises(H3CheckpointError) as scale_shape:
            inspect(bad, payloads)
        self.assertEqual(scale_shape.exception.code, "tensor_shape_mismatch")

        bad = copy.deepcopy(header)
        bad["blocks.49.mlp.fc2.weight_scale"]["dtype"] = "F16"
        with self.assertRaises(H3CheckpointError) as scale_dtype:
            inspect(bad, payloads)
        self.assertEqual(scale_dtype.exception.code, "unsupported_tensor_dtype")

        bad = copy.deepcopy(header)
        del bad["blocks.49.mlp.fc2.weight_scale"]
        with self.assertRaises(H3CheckpointError) as missing_scale:
            inspect(bad, payloads)
        self.assertEqual(missing_scale.exception.code, "missing_tensor")

    def test_any_present_marker_is_parsed_even_when_header_metadata_is_complete(self):
        header, payloads = metadata_only_int8_header()
        marker = "blocks.49.mlp.fc2.comfy_quant"
        header[marker] = {"dtype": "U8", "shape": [4]}
        payloads[marker] = b"nope"

        with self.assertRaises(H3CheckpointError) as malformed:
            inspect(header, payloads)
        self.assertEqual(malformed.exception.code, "invalid_quantization_descriptor")

    def test_marker_backed_header_requires_reader_for_conflict_check(self):
        header, payloads = make_header(
            compressed=False,
            quantization="int8",
            native="ref2va",
            qkv="grouped",
        )
        self.assertEqual(
            sum(name.endswith(".comfy_quant") for name in header),
            len(QUANT_SUFFIXES) * 50,
        )
        with self.assertRaises(H3CheckpointError) as missing_reader:
            inspect(header, payloads, reader=False)
        self.assertEqual(missing_reader.exception.code, "small_tensor_reader_required")


if __name__ == "__main__":
    unittest.main()
