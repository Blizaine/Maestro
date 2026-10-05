"""H3 import contracts for actual packed W4A8 and GGUF logical tensor layouts."""
import copy
import json
import sys
import struct
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_h3_checkpoint_import import make_header, QUANT_SUFFIXES
from services.h3_checkpoint_import import H3CheckpointError, inspect_h3_header


def w4a8_header():
    header, payloads = make_header()
    layers = {}
    for index in range(50):
        for suffix in QUANT_SUFFIXES:
            base = f"blocks.{index}.{suffix}"
            weight = header[base + ".weight"]
            rows, columns = weight["shape"]
            weight.update(dtype="I8", shape=[rows, columns // 2])
            header[base + ".weight_s_rel"] = dict(dtype="F8_E4M3", shape=[rows, columns // 16])
            header[base + ".weight_s_channel"] = dict(dtype="F32", shape=[rows])
            header[base + ".weight_codebook"] = dict(dtype="F32", shape=[16])
            layers[base] = dict(format="asym_w4a8_int8", group_size=16, convrot=True, convrot_groupsize=256)
    header["__metadata__"].update({"quantization.bits": "W4A8", "_quantization_metadata": json.dumps({"layers": layers})})
    return header, payloads


def gguf_header():
    header, payloads = make_header()
    header["__metadata__"] = {}
    # Mixed precision also covers condition and token-refiner linears.
    for name, info in header.items():
        if name.endswith(".weight") and len(info["shape"]) == 2 and info["shape"][-1] % 256 == 0:
            info["dtype"] = "GGUF_Q4_K"
            info["gguf_type"] = 12
    return header, payloads


class H3QuantizedImportTests(unittest.TestCase):
    def inspect(self, header, payloads, **kwargs):
        return inspect_h3_header(header, tensor_reader=payloads.__getitem__, sampling_profile="standard",
                                 native_workflow="ref2va", **kwargs)

    def test_w4a8_uses_logical_dimensions_and_only_reads_curve(self):
        header, payloads = w4a8_header()
        reads = []
        profile = inspect_h3_header(header, tensor_reader=lambda key: reads.append(key) or payloads[key],
                                    sampling_profile="standard")
        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["quantization_format"], "asym_w4a8_int8")
        self.assertEqual(profile["quantized_layers"], 200)
        self.assertEqual(profile["weight_group_size"], 16)
        self.assertEqual(profile["convrot_group_size"], 256)
        self.assertEqual(profile["qkv_layout"], "grouped")
        self.assertEqual(reads, ["adaln_t_table"])

    def test_w4a8_rejects_wrong_packing_scales_codebook_or_correction(self):
        valid, payloads = w4a8_header()
        base = "blocks.3.attn.qkv_proj"
        changes = (
            (base + ".weight", "shape", [21504, 5376]),
            (base + ".weight_s_rel", "shape", [21504, 335]),
            (base + ".weight_s_channel", "dtype", "BF16"),
            (base + ".weight_codebook", "shape", [15]),
        )
        for name, field, value in changes:
            with self.subTest(name=name):
                bad = copy.deepcopy(valid); bad[name][field] = value
                with self.assertRaises(H3CheckpointError): self.inspect(bad, payloads)
        bad = copy.deepcopy(valid)
        bad[base + ".weight_correction"] = dict(dtype="F16", shape=[336, 21504])
        with self.assertRaises(H3CheckpointError): self.inspect(bad, payloads)

    def test_w4a8_requires_all_explicit_layer_metadata_and_supported_groups(self):
        header, payloads = w4a8_header()
        original = json.loads(header["__metadata__"]["_quantization_metadata"])
        for field, value in (("group_size", 32), ("convrot_groupsize", 128), ("convrot", False), ("convrot", "false"), ("format", "nunchaku_int4")):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed["layers"]["blocks.0.attn.qkv_proj"][field] = value
                header["__metadata__"]["_quantization_metadata"] = json.dumps(changed)
                with self.assertRaises(H3CheckpointError): self.inspect(header, payloads)
        original["layers"].pop("blocks.1.mlp.fc2")
        header["__metadata__"]["_quantization_metadata"] = json.dumps(original)
        with self.assertRaisesRegex(H3CheckpointError, "all 200"): self.inspect(header, payloads)

    def test_int4_label_cannot_override_a_different_quantizer(self):
        header, payloads = w4a8_header()
        header["__metadata__"]["quantization.bits"] = "INT4 custom_backend"
        with self.assertRaises(H3CheckpointError): self.inspect(header, payloads)
        header["__metadata__"]["quantization.bits"] = "W4A8"
        header["__metadata__"]["qkv_layout"] = "interleaved"
        with self.assertRaisesRegex(H3CheckpointError, "conflicting"): self.inspect(header, payloads)

    def test_gguf_mixed_linear_quants_require_a_creator_layout_selection(self):
        header, payloads = gguf_header()
        pending = self.inspect(header, payloads, file_format="gguf")
        self.assertEqual(pending["status"], "needs_selection")
        self.assertIn("qkv_layout", pending["needs_selection"])
        self.assertIsNone(pending["qkv_layout"])
        for layout in ("grouped", "interleaved"):
            profile = self.inspect(header, payloads, file_format="gguf", qkv_layout=layout)
            self.assertEqual(profile["status"], "verified")
            self.assertEqual(profile["qkv_layout_selection"], layout)
            self.assertEqual(profile["gguf_quant_types"], ["Q4_K"])
            self.assertGreater(profile["quantized_layers"], 200)
            self.assertEqual(profile["quantization_format"], "gguf")

    def test_gguf_explicit_layout_is_detected_and_conflicting_selection_blocked(self):
        header, payloads = gguf_header()
        header["__metadata__"]["qkv_layout"] = "grouped"
        self.assertEqual(self.inspect(header, payloads, file_format="gguf")["status"], "verified")
        with self.assertRaisesRegex(H3CheckpointError, "conflicts"):
            self.inspect(header, payloads, file_format="gguf", qkv_layout="interleaved")

    def test_gguf_rejects_quantized_curves_norms_unsupported_types_and_wrong_h3_geometry(self):
        original, payloads = gguf_header()
        for name, field, value in (
            ("adaln_t_table", "dtype", "GGUF_Q4_K"),
            ("blocks.0.norm1.weight", "dtype", "GGUF_Q4_K"),
            ("blocks.0.attn.qkv_proj.weight", "dtype", "GGUF_IQ4_XS"),
            ("blocks.49.attn.qkv_proj.weight", "shape", [21504, 4096]),
        ):
            with self.subTest(name=name):
                header = copy.deepcopy(original); header[name][field] = value
                with self.assertRaises(H3CheckpointError):
                    self.inspect(header, payloads, file_format="gguf", qkv_layout="grouped")

    def test_gguf_unquantized_float_curves_are_checked_for_finite_values(self):
        for dtype, finite, invalid in (
            ("F16", struct.pack("<e", 1.0), struct.pack("<e", float("inf"))),
            ("BF16", struct.pack("<H", 0x3f80), struct.pack("<H", 0x7fc0)),
        ):
            header, payloads = gguf_header()
            header["adaln_t_table"]["dtype"] = dtype
            payloads["adaln_t_table"] = finite * (1025 * 8)
            self.assertEqual(self.inspect(header, payloads, file_format="gguf", qkv_layout="grouped")["status"], "verified")
            payloads["adaln_t_table"] = invalid + payloads["adaln_t_table"][2:]
            with self.assertRaisesRegex(H3CheckpointError, "NaN or infinite"):
                self.inspect(header, payloads, file_format="gguf", qkv_layout="grouped")

    def test_safetensor_metadata_cannot_spoof_a_gguf_container(self):
        header, payloads = gguf_header()
        header["__metadata__"]["maestro.file_format"] = "gguf"
        with self.assertRaises(H3CheckpointError): self.inspect(header, payloads, qkv_layout="grouped")


if __name__ == "__main__": unittest.main()
