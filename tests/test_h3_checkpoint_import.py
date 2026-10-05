"""Focused, weight-free tests for the generic MiniMax H3 header verifier."""

from __future__ import annotations

import json
import math
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.h3_checkpoint_import import (  # noqa: E402
    H3CheckpointError,
    inspect_h3_header,
)


HIDDEN = 5376
INNER = 7168
FFN = 14336
QKV = INNER * 3
QUANT_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def descriptor(dtype, shape):
    return {"dtype": dtype, "shape": list(shape)}


def _add_float_transformer(header, *, compressed, dtype="BF16"):
    table = None
    if compressed:
        header["adaln_t_table"] = descriptor("F32", (1025, 8))
        table = bytes(1025 * 8 * 4)
        curve_dim = 8
    else:
        curve_dim = 2688
        header.update(
            {
                "time_embedder.proj_in.weight": descriptor(dtype, (5376, 256)),
                "time_embedder.proj_in.bias": descriptor(dtype, (5376,)),
                "time_embedder.proj_out.weight": descriptor(dtype, (2688, 5376)),
                "time_embedder.proj_out.bias": descriptor(dtype, (2688,)),
            }
        )

    for name, shape, tensor_dtype in (
        ("video_patch_proj.weight", (5376, 96), dtype),
        ("video_patch_proj.bias", (5376,), "F32"),
        ("audio_patch_proj.weight", (5376, 32), dtype),
        ("audio_patch_proj.bias", (5376,), "F32"),
        ("condition_proj.weight", (5376, 5120), dtype),
        ("condition_proj.bias", (5376,), dtype),
        ("final_layer.norm.weight", (5376,), dtype),
        ("final_layer.adaln_proj.linear.weight", (10752, curve_dim), dtype),
        ("final_layer.adaln_proj.linear.bias", (10752,), dtype),
        ("final_layer.video_out.weight", (96, 5376), dtype),
        ("final_layer.video_out.bias", (96,), "F32"),
        ("final_layer.audio_out.weight", (32, 5376), dtype),
        ("final_layer.audio_out.bias", (32,), "F32"),
        ("token_refiner.final_norm.weight", (5376,), dtype),
        ("rope.inv_freq", (16,), "F32"),
    ):
        header[name] = descriptor(tensor_dtype, shape)

    for index in range(50):
        prefix = f"blocks.{index}."
        values = (
            ("norm1.weight", (5376,), dtype),
            ("norm2.weight", (5376,), dtype),
            ("attn.qkv_proj.weight", (QKV, 5376), dtype),
            ("attn.q_norm.weight", (128,), dtype),
            ("attn.k_norm.weight", (128,), dtype),
            ("attn.out_proj.weight", (5376, INNER), dtype),
            ("mlp.fc1.weight", (2 * FFN, 5376), dtype),
            ("mlp.fc2.weight", (5376, FFN), dtype),
            ("adaln_proj.linear.weight", (18 * 5376, curve_dim), dtype),
            ("adaln_proj.linear.bias", (18 * 5376,), "F16" if compressed else dtype),
        )
        for name, shape, tensor_dtype in values:
            header[prefix + name] = descriptor(tensor_dtype, shape)

    for index in range(2):
        prefix = f"token_refiner.blocks.{index}."
        for name, shape in (
            ("norm1.weight", (5376,)),
            ("norm2.weight", (5376,)),
            ("attn.qkv_proj.weight", (QKV, 5376)),
            ("attn.q_norm.weight", (128,)),
            ("attn.k_norm.weight", (128,)),
            ("attn.out_proj.weight", (5376, INNER)),
            ("mlp.fc1.weight", (2 * FFN, 5376)),
            ("mlp.fc2.weight", (5376, FFN)),
        ):
            header[prefix + name] = descriptor(dtype, shape)
    return table


def make_header(*, compressed=True, quantization="plain", native="ref2va", qkv=None, group_size=256):
    header = {}
    table = _add_float_transformer(header, compressed=compressed)
    metadata = {
        "modelspec.architecture": "minimax_h3",
        "modelspec.title": f"minimax_h3_{native}_{'pruned' if compressed else 'full'}",
    }
    if qkv:
        metadata["modelspec.qkv_layout"] = qkv
    reader_payloads = {}

    if quantization == "int8":
        layers = {}
        marker_bytes = json.dumps(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size},
            separators=(",", ":"),
        ).encode("utf-8")
        marker_bytes = marker_bytes.ljust(121, b"\0")
        for index in range(50):
            prefix = f"blocks.{index}."
            for suffix in QUANT_SUFFIXES:
                base = prefix + suffix
                weight_name = base + ".weight"
                info = header[weight_name]
                info["dtype"] = "I8"
                rows = info["shape"][0]
                header[base + ".weight_scale"] = descriptor("F32", (rows, 1))
                marker = base + ".comfy_quant"
                header[marker] = descriptor("U8", (len(marker_bytes),))
                reader_payloads[marker] = marker_bytes
                layers[base] = {
                    "format": "int8_tensorwise",
                    "orig_dtype": "torch.bfloat16",
                    "convrot": True,
                    "convrot_groupsize": group_size,
                }
        metadata["quantization.bits"] = "INT8 Row-wise ConvRot (runtime)"
        metadata["_quantization_metadata"] = json.dumps(
            {"format_version": "1.0", "layers": layers}, separators=(",", ":")
        )
    elif quantization == "fp8":
        for index in range(50):
            prefix = f"blocks.{index}."
            for suffix in QUANT_SUFFIXES:
                base = prefix + suffix
                info = header[base + ".weight"]
                info["dtype"] = "F8_E4M3"
                rows = info["shape"][0]
                header[base + ".weight_scale"] = descriptor("F32", (rows, 1))
        metadata["quantization_format"] = "scaled_fp8"

    metadata["modelspec.implementation"] = "https://github.com/MiniMax-AI/MiniMax-H3" if not compressed else "community export"
    header["__metadata__"] = metadata
    if table is not None:
        reader_payloads["adaln_t_table"] = table
    return header, reader_payloads


def source_record(*, turbo=False):
    return {
        "modelId": 2877206,
        "versionId": 3374439 if turbo else 3374445,
        "fileId": 3263048 if turbo else 3263052,
        "name": "DaSiWa MiniMax H3",
        "versionName": "DaSiWa Hybrid Turbo v3" if turbo else "DaSiWa Hybrid v3",
        "filename": "creator-name-is-display-only.safetensors",
        "sha256": (
            "5da4bbf303f91b468010f504951e3959cd77420ec5f93ae007c9660b485f7df6"
            if turbo
            else "0ce4dfecce862b010823a5a74903543bd7f2ad21872272232f70f9a836d30e53"
        ),
        "size_bytes": 20967668632 if turbo else 20967668624,
        "description": "Hybrid (REF2VA + FL2VA compatible)",
        "versionDescription": "The Turbo (4 till 8-step) comes with latest distillation blend." if turbo else "Hybrid with full reference support; 20-30-step.",
    }


class H3CheckpointImportTests(unittest.TestCase):
    def test_pinned_dasiwa_standard_and_turbo_headers_verify_without_weight_reads(self):
        for turbo in (False, True):
            with self.subTest(turbo=turbo):
                header, payloads = make_header(
                    compressed=True,
                    quantization="int8",
                    native="ref2va",
                    qkv="grouped",
                )
                reads = []

                def read_small(key):
                    reads.append(key)
                    return payloads[key]

                profile = inspect_h3_header(
                    header,
                    tensor_reader=read_small,
                    source=source_record(turbo=turbo),
                )
                self.assertEqual(profile["status"], "verified")
                self.assertTrue(profile["compressed_modulation"])
                self.assertEqual((profile["adaln_curve_grid"], profile["time_embed_dim"]), (1025, 8))
                self.assertEqual(profile["quantization_format"], "int8_tensorwise")
                self.assertTrue(profile["convrot"])
                self.assertEqual(profile["convrot_group_size"], 256)
                self.assertEqual(profile["qkv_layout"], "grouped")
                self.assertEqual(profile["native_workflow"], "ref2va")
                self.assertEqual(profile["architectures"], ["minimax_h3", "minimax_h3_ref2va"])
                self.assertEqual(profile["sampling_profile"], "turbo" if turbo else "standard")
                self.assertEqual(profile["baked_turbo"], turbo)
                self.assertFalse(profile["fused_turbo"])
                if turbo:
                    self.assertEqual((profile["sampler"], profile["video_shift"], profile["audio_shift"]), ("euler", 9, 4))
                    self.assertEqual((profile["default_steps"], profile["min_steps"], profile["max_steps"]), (8, 4, 8))
                else:
                    self.assertEqual((profile["sampler"], profile["video_shift"], profile["audio_shift"]), ("euler", 11, 4))
                    self.assertEqual((profile["default_steps"], profile["min_steps"], profile["max_steps"]), (25, 20, 30))
                self.assertIn("adaln_t_table", reads)
                self.assertEqual(len([key for key in reads if key.endswith(".comfy_quant")]), 200)
                self.assertTrue(all(key.endswith(".comfy_quant") or key == "adaln_t_table" for key in reads))

    def test_ambiguous_generic_workflow_and_recipe_return_actionable_choices(self):
        header, payloads = make_header(native="hybrid", qkv="grouped")
        header["__metadata__"].pop("modelspec.title")
        header["__metadata__"]["modelspec.title"] = "minimax_h3_pruned_hybrid"
        profile = inspect_h3_header(
            header,
            tensor_reader=lambda key: payloads[key],
            source={"modelId": 999, "versionId": 111, "fileId": 222, "name": "H3 community export"},
        )
        self.assertEqual(profile["status"], "needs_selection")
        self.assertEqual(set(profile["needs_selection"]), {"native_workflow", "sampling_profile"})
        self.assertEqual(profile["selection_options"]["native_workflow"], ["fl2va", "ref2va"])
        self.assertEqual(profile["selection_options"]["sampling_profile"], ["standard", "turbo", "fused"])
        self.assertEqual(profile["architectures"], [])

    def test_user_selection_resolves_ambiguous_generic_metadata(self):
        header, payloads = make_header(native="hybrid", qkv="grouped")
        header["__metadata__"]["modelspec.title"] = "minimax_h3_pruned_hybrid"
        profile = inspect_h3_header(
            header,
            tensor_reader=lambda key: payloads[key],
            native_workflow="ref2va",
            sampling_profile="standard",
        )
        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["architectures"], ["minimax_h3", "minimax_h3_ref2va"])
        self.assertEqual((profile["sampler"], profile["video_shift"], profile["audio_shift"], profile["default_steps"]), ("euler", 12, 3, 20))
        self.assertEqual((profile["min_steps"], profile["max_steps"]), (2, 50))

    def test_explicit_fused_selection_has_complete_safe_defaults(self):
        header, payloads = make_header(native="ref2va", qkv="grouped")
        header["__metadata__"]["modelspec.title"] = "minimax_h3_ref2va_pruned_hybrid"
        profile = inspect_h3_header(
            header,
            tensor_reader=lambda key: payloads[key],
            native_workflow="ref2va",
            sampling_profile="fused",
        )
        self.assertEqual(profile["status"], "verified")
        self.assertEqual((profile["sampler"], profile["video_shift"], profile["audio_shift"]), ("res_multistep", 12, 3))
        self.assertEqual((profile["default_steps"], profile["min_steps"], profile["max_steps"]), (4, 4, 12))
        self.assertTrue(profile["baked_turbo"])
        self.assertTrue(profile["fused_turbo"])

    def test_supported_int8_convrot_group64_is_verified_when_explicit(self):
        header, payloads = make_header(quantization="int8", qkv="grouped", group_size=64)
        profile = inspect_h3_header(
            header,
            tensor_reader=lambda key: payloads[key],
            native_workflow="ref2va",
            sampling_profile="standard",
        )
        self.assertEqual(profile["convrot_group_size"], 64)

    def test_real_fl2va_int8_export_layout_accepts_retained_fp32_io_projections(self):
        # Mirrors the CivitAI 2830065 / 3193337 / 3074134 FL2VA INT8 export:
        # the four retained patch/output projection weights are F32, the 200
        # block linears are INT8 with 72-byte Comfy descriptors, and there is
        # no SafeTensor metadata or _quantization_metadata block.
        header, payloads = make_header(quantization="int8", native="fl2va", qkv="grouped")
        for name in (
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "final_layer.video_out.weight",
            "final_layer.audio_out.weight",
        ):
            header[name]["dtype"] = "F32"

        header.pop("__metadata__")
        marker_payload = json.dumps(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
            separators=(",", ":"),
        ).encode("utf-8").ljust(72, b"\0")
        for marker_name in [name for name in payloads if name.endswith(".comfy_quant")]:
            payloads[marker_name] = marker_payload
            header[marker_name]["shape"] = [72]

        source = {
            "modelId": 2830065,
            "versionId": 3193337,
            "fileId": 3074134,
            "name": "Minimax H3 INT8/INT6/INT4 ConvRot",
            "versionName": "FL2VA INT8 Pruned",
            "filename": "minimaxH3INT8INT6INT4_fl2vaINT8Pruned.safetensors",
            "sha256": "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
            "size_bytes": 20970379616,
            "description": "Both FL2VA first/last-frame and REF2VA reference-conditioned versions are available.",
            "versionDescription": None,
        }
        reads = []

        def read_small(key):
            reads.append(key)
            return payloads[key]

        profile = inspect_h3_header(
            header,
            tensor_reader=read_small,
            source=source,
            sampling_profile="standard",
        )
        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["native_workflow"], "fl2va")
        self.assertEqual(profile["architectures"], ["minimax_h3"])
        self.assertEqual(profile["quantization_format"], "int8_tensorwise")
        self.assertEqual(profile["convrot_group_size"], 256)
        self.assertEqual(profile["qkv_layout"], "grouped")
        self.assertEqual(len([key for key in reads if key.endswith(".comfy_quant")]), 200)
        self.assertEqual(set(reads), {"adaln_t_table"} | {key for key in payloads if key.endswith(".comfy_quant")})

    def test_half_precision_rotary_buffer_is_supported(self):
        for quantization in ("none", "int8"):
            for dtype in ("BF16", "F16"):
                with self.subTest(quantization=quantization, dtype=dtype):
                    header, payloads = make_header(
                        quantization=quantization, native="fl2va", qkv="grouped",
                    )
                    header["rope.inv_freq"]["dtype"] = dtype
                    profile = inspect_h3_header(
                        header,
                        tensor_reader=lambda key: payloads[key],
                        native_workflow="fl2va",
                        sampling_profile="turbo",
                    )
                    self.assertEqual(profile["status"], "verified")
                    self.assertEqual(profile["architectures"], ["minimax_h3"])

    def test_rotary_buffer_must_remain_float_with_native_shape(self):
        for dtype, shape in (("I8", [16]), ("U8", [16]), ("BF16", [32])):
            with self.subTest(dtype=dtype, shape=shape):
                header, payloads = make_header(native="fl2va", qkv="grouped")
                header["rope.inv_freq"] = descriptor(dtype, shape)
                with self.assertRaisesRegex(H3CheckpointError, "rope.inv_freq"):
                    inspect_h3_header(header, tensor_reader=lambda key: payloads[key])

    def test_unpinned_turbo_needs_recipe_confirmation_or_uses_explicit_fallback(self):
        header, payloads = make_header(native="fl2va", qkv="grouped")
        header["__metadata__"]["modelspec.title"] = "minimax_h3_fl2va_pruned_turbo"
        inferred = inspect_h3_header(
            header,
            tensor_reader=lambda key: payloads[key],
        )
        self.assertEqual(inferred["sampling_profile"], "turbo")
        self.assertEqual(inferred["status"], "needs_selection")
        self.assertIn("sampling_recipe", inferred["needs_selection"])
        self.assertIn("missing_fields", inferred["selection_options"]["sampling_recipe"])

        selected = inspect_h3_header(
            header,
            tensor_reader=lambda key: payloads[key],
            sampling_profile="turbo",
        )
        self.assertEqual(selected["status"], "verified")
        self.assertEqual((selected["sampler"], selected["video_shift"], selected["audio_shift"]), ("euler", 12, 3))
        self.assertEqual((selected["default_steps"], selected["min_steps"], selected["max_steps"]), (8, 4, 8))

    def test_creator_published_turbo_range_and_sampler_are_used(self):
        header, payloads = make_header(native="fl2va", qkv="grouped")
        header["__metadata__"]["modelspec.title"] = "minimax_h3_fl2va_pruned"
        source = {
            "name": "Heptagram",
            "versionName": "Heptagram turbo v1.0",
            "description": "Turbo recipe: Steps: 4-8; Shift video: 6-16; Shift audio: 3; Sampler: res_multistep; Scheduler: simple.",
            "versionDescription": "The model is recommended for 4-8 steps.",
        }
        profile = inspect_h3_header(
            header,
            tensor_reader=lambda key: payloads[key],
            source=source,
        )
        self.assertEqual(profile["status"], "verified")
        self.assertEqual(profile["sampling_profile"], "turbo")
        self.assertEqual((profile["sampler"], profile["video_shift"], profile["audio_shift"]), ("res_multistep", 11, 3))
        self.assertEqual((profile["default_steps"], profile["min_steps"], profile["max_steps"]), (8, 4, 8))
        self.assertEqual(profile["default_settings"]["video_shift_range"], [6.0, 16.0])

    def test_plain_pruned_checkpoint_requires_explicit_qkv_evidence(self):
        header, payloads = make_header(native="ref2va", qkv=None)
        with self.assertRaises(H3CheckpointError) as caught:
            inspect_h3_header(
                header,
                tensor_reader=lambda key: payloads[key],
                sampling_profile="standard",
                native_workflow="ref2va",
            )
        self.assertEqual(caught.exception.code, "qkv_layout_ambiguous")
        self.assertIn("qkv_layout", str(caught.exception))

    def test_full_official_bf16_checkpoint_uses_official_interleaved_layout(self):
        header, payloads = make_header(compressed=False, native="ref2va", qkv=None)
        profile = inspect_h3_header(
            header,
            source={"versionName": "Base H3"},
            sampling_profile="standard",
        )
        self.assertEqual(profile["status"], "verified")
        self.assertFalse(profile["compressed_modulation"])
        self.assertIsNone(profile["adaln_curve_grid"])
        self.assertEqual(profile["time_embed_dim"], 2688)
        self.assertEqual(profile["qkv_layout"], "interleaved")
        self.assertEqual(profile["quantization_format"], "none")
        self.assertEqual(profile["template_model_type"], "minimax_h3_ref2va_full")
        self.assertEqual(profile["architectures"], ["minimax_h3_full", "minimax_h3_ref2va_full"])

    def test_scaled_fp8_checkpoint_requires_scale_rows_and_explicit_format(self):
        header, payloads = make_header(quantization="fp8", native="fl2va", qkv="grouped")
        profile = inspect_h3_header(
            header,
            tensor_reader=lambda key: payloads[key],
            native_workflow="fl2va",
            sampling_profile="standard",
        )
        self.assertEqual(profile["quantization_format"], "scaled_fp8")
        self.assertEqual(profile["qkv_layout"], "grouped")
        self.assertFalse(profile["convrot"])
        self.assertIsNone(profile["convrot_group_size"])
        del header["blocks.49.mlp.fc2.weight_scale"]
        with self.assertRaises(H3CheckpointError) as caught:
            inspect_h3_header(
                header,
                tensor_reader=lambda key: payloads[key],
                native_workflow="fl2va",
                sampling_profile="standard",
            )
        self.assertEqual(caught.exception.code, "missing_fp8_scale")

    def test_scaled_fp8_interleaved_qkv_is_rejected_until_loader_supports_it(self):
        header, payloads = make_header(quantization="fp8", native="fl2va", qkv="interleaved")
        with self.assertRaises(H3CheckpointError) as caught:
            inspect_h3_header(
                header,
                tensor_reader=lambda key: payloads[key],
                native_workflow="fl2va",
                sampling_profile="standard",
            )
        self.assertEqual(caught.exception.code, "unsupported_qkv_layout")

    def test_scalar_scaled_fp8_matches_the_verified_comfy_release(self):
        header, payloads = make_header(quantization="fp8", native="fl2va", qkv=None)
        header.pop("__metadata__")
        for key, info in list(header.items()):
            if key.endswith(".weight_scale"):
                info["shape"] = []
                base = key[:-len(".weight_scale")]
                marker = base + ".comfy_quant"
                payloads[marker] = b'{"format":"float8_e4m3fn"}'
                header[marker] = descriptor("U8", (len(payloads[marker]),))
                header[base + ".input_scale"] = descriptor("F32", ())
        for key in ("video_patch_proj.weight", "audio_patch_proj.weight",
                    "final_layer.video_out.weight", "final_layer.audio_out.weight"):
            header[key]["dtype"] = "F32"
        source = {"modelId": 2832970, "versionId": 3196955, "fileId": 3078101,
                  "sha256": "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993",
                  "versionName": "fl2va pruned fp8 scaled"}
        actual = inspect_h3_header(header, tensor_reader=lambda key: payloads[key], source=source)
        self.assertEqual(actual["status"], "verified")
        self.assertEqual(actual["qkv_layout"], "grouped")
        self.assertEqual(actual["architectures"], ["minimax_h3"])
        self.assertEqual(actual["default_steps"], 20)
        with self.assertRaises(H3CheckpointError) as caught:
            inspect_h3_header(header, tensor_reader=lambda key: payloads[key],
                              source={**source, "sha256": "a" * 64}, sampling_profile="standard")
        self.assertEqual(caught.exception.code, "qkv_layout_ambiguous")

    def test_fp8_inverse_scales_are_not_accepted_as_direct_scales(self):
        header, payloads = make_header(quantization="fp8", native="fl2va", qkv="grouped")
        header["blocks.0.attn.qkv_proj.weight_scale_inv"] = header.pop("blocks.0.attn.qkv_proj.weight_scale")
        with self.assertRaises(H3CheckpointError) as caught:
            inspect_h3_header(header, tensor_reader=lambda key: payloads[key], sampling_profile="standard")
        self.assertEqual(caught.exception.code, "missing_fp8_scale")

    def test_rejects_nonfinite_rank8_table(self):
        header, payloads = make_header(qkv="grouped")
        table = bytearray(payloads["adaln_t_table"])
        table[0:4] = struct.pack("<f", math.nan)
        with self.assertRaises(H3CheckpointError) as caught:
            inspect_h3_header(
                header,
                tensor_reader=lambda key: table if key == "adaln_t_table" else payloads[key],
                native_workflow="ref2va",
                sampling_profile="standard",
            )
        self.assertEqual(caught.exception.code, "nonfinite_adaln_table")

    def test_rejects_missing_misshapen_or_video_only_h3_anchors(self):
        cases = (
            ("missing audio patch", lambda h: h.pop("audio_patch_proj.weight"), "missing_tensor"),
            ("wrong qkv shape", lambda h: h["blocks.49.attn.qkv_proj.weight"].update(shape=[21503, 5376]), "tensor_shape_mismatch"),
            ("missing block", lambda h: h.pop("blocks.17.attn.qkv_proj.weight"), "missing_tensor"),
        )
        for label, mutate, expected_code in cases:
            with self.subTest(label=label):
                header, payloads = make_header(qkv="grouped")
                mutate(header)
                with self.assertRaises(H3CheckpointError) as caught:
                    inspect_h3_header(
                        header,
                        tensor_reader=lambda key: payloads[key],
                        native_workflow="ref2va",
                        sampling_profile="standard",
                    )
                self.assertEqual(caught.exception.code, expected_code)

    def test_rejects_unsupported_quantization_labels_and_components(self):
        for label in ("INT4", "INT6", "NVFP4", "GGUF", "custom-fast-backend"):
            with self.subTest(label=label):
                header, payloads = make_header(qkv="grouped")
                header["__metadata__"]["quantization.bits"] = label
                with self.assertRaises(H3CheckpointError) as caught:
                    inspect_h3_header(
                        header,
                        tensor_reader=lambda key: payloads[key],
                        native_workflow="ref2va",
                        sampling_profile="standard",
                    )
                self.assertEqual(caught.exception.code, "unsupported_quantization")

        for component, tensor_name in (
            ("lora", "blocks.0.attn.qkv_proj.lora_A.weight"),
            ("vae", "vae.encoder.weight"),
            ("vdn", "blocks.0.attn.vdn.linear_attention.weight"),
        ):
            with self.subTest(component=component):
                header, payloads = make_header(qkv="grouped")
                header[tensor_name] = descriptor("BF16", (2, 2))
                with self.assertRaises(H3CheckpointError) as caught:
                    inspect_h3_header(
                        header,
                        tensor_reader=lambda key: payloads[key],
                        native_workflow="ref2va",
                        sampling_profile="standard",
                    )
                self.assertEqual(caught.exception.code, "unsupported_checkpoint_component")

    def test_rejects_quant_descriptor_conflicts_and_unpinned_hash_mismatch(self):
        header, payloads = make_header(quantization="int8", qkv="grouped")
        descriptor_payload = json.dumps(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 128},
            separators=(",", ":"),
        ).encode("utf-8").ljust(121, b"\0")
        payloads["blocks.0.attn.qkv_proj.comfy_quant"] = descriptor_payload
        with self.assertRaises(H3CheckpointError) as caught:
            inspect_h3_header(
                header,
                tensor_reader=lambda key: payloads[key],
                source=source_record(),
            )
        self.assertEqual(caught.exception.code, "quantization_conflict")

        header, payloads = make_header(quantization="int8", qkv="grouped")
        bad_source = source_record()
        bad_source["sha256"] = "0" * 64
        with self.assertRaises(H3CheckpointError) as caught:
            inspect_h3_header(
                header,
                tensor_reader=lambda key: payloads[key],
                source=bad_source,
            )
        self.assertEqual(caught.exception.code, "source_hash_mismatch")

    def test_wrapped_tensor_names_are_normalized_and_duplicate_wrappers_rejected(self):
        header, payloads = make_header(qkv="grouped")
        wrapped = {
            "__metadata__": header["__metadata__"],
            **{
                "model.diffusion_model." + key: value
                for key, value in header.items()
                if key != "__metadata__"
            },
        }
        profile = inspect_h3_header(
            wrapped,
            tensor_reader=lambda key: payloads[key.removeprefix("model.diffusion_model.")],
            native_workflow="ref2va",
            sampling_profile="standard",
        )
        self.assertEqual(profile["status"], "verified")

        wrapped["blocks.0.norm1.weight"] = descriptor("BF16", (5376,))
        with self.assertRaises(H3CheckpointError) as caught:
            inspect_h3_header(
                wrapped,
                tensor_reader=lambda key: payloads[key.removeprefix("model.diffusion_model.")],
                native_workflow="ref2va",
                sampling_profile="standard",
            )
        self.assertEqual(caught.exception.code, "duplicate_tensor")


if __name__ == "__main__":
    unittest.main()
