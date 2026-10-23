"""CPU runtime checks for metadata-backed H3 INT8 ConvRot checkpoints."""

from __future__ import annotations

import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


def int8_layer_config(group_size, *, format="int8_tensorwise"):
    return {
        "format": format,
        "convrot": True,
        "convrot_groupsize": group_size,
    }


def w4a8_layer_config(group_size=16, *, convrot_group_size=None):
    config = {"format": "asym_w4a8_int8", "group_size": group_size}
    if convrot_group_size is not None:
        config["convrot_groupsize"] = convrot_group_size
    return config


def quantization_metadata(layers):
    return {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": layers},
            separators=(",", ":"),
        )
    }


def explicit_hadamard(torch, group_size):
    """Build the normalized dense H4 Kronecker matrix as an independent oracle."""
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    )
    matrix = h4
    while matrix.shape[0] < group_size:
        matrix = torch.kron(matrix, h4)
    if matrix.shape != (group_size, group_size):
        raise ValueError(f"Unsupported explicit Hadamard group size: {group_size}")
    return matrix * (group_size**-0.5)


class H3HeaderInt8RuntimeTests(unittest.TestCase):
    def test_bf16_curve_table_loads_and_runtime_promotes_exact_values_before_interpolation(self):
        import torch

        from mmgp import offload
        from models.minimax_h3.transformer import MiniMaxH3Transformer
        from safetensors.torch import save_file

        model = MiniMaxH3Transformer(
            hidden_size=8,
            num_layers=0,
            token_refiner_layers=0,
            num_attention_heads=2,
            attention_head_dim=4,
            ffn_dim=16,
            video_channels=1,
            audio_channels=1,
            patch_size=(1, 1, 1),
            text_dim=8,
            curve_grid=1025,
            curve_dim=8,
            timestep_input_dim=4,
            time_embed_hidden_size=8,
            rope_freq_dim=4,
            dtype=torch.bfloat16,
        )
        source_table = torch.linspace(-1.25, 2.75, 1025 * 8).reshape(1025, 8).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bf16_curve.safetensors"
            save_file({"adaln_t_table": source_table}, str(checkpoint))
            offload.load_model_data(
                model,
                str(checkpoint),
                default_dtype=torch.bfloat16,
                verboseLevel=0,
                ignore_missing_keys=True,
            )
            promoted_table = source_table.float()
            self.assertEqual(model.adaln_t_table.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(model.adaln_t_table, source_table))
            self.assertTrue(torch.equal(model.adaln_t_table.float(), promoted_table))

            timestep = torch.tensor([0.0, 17.5 / 1024, 511.25 / 1024, 1.0])
            actual = model._curve_at(timestep, torch.device("cpu"))
            expected_rows = []
            for value in timestep.tolist():
                position = value * 1024
                lower = min(int(position), 1023)
                fraction = position - lower
                expected_rows.append(
                    [
                        float(promoted_table[lower, column])
                        + fraction
                        * (
                            float(promoted_table[lower + 1, column])
                            - float(promoted_table[lower, column])
                        )
                        for column in range(8)
                    ]
                )
            expected = torch.tensor(expected_rows, dtype=torch.float32)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
            del actual, expected, model
            gc.collect()

    def test_w4a8_metadata_only_keeps_peer_defaults_and_power_of_four_groups(self):
        import torch

        from shared.qtypes import asym_w4a8_int8, int8_convrot

        base = "linear"
        logical_k = 1024
        state_dict = {
            base + ".weight": torch.ones((4, logical_k // 2), dtype=torch.int8),
            base + ".weight_s_rel": torch.ones(
                (4, logical_k // 16), dtype=torch.float8_e4m3fn
            ),
            base + ".weight_s_channel": torch.ones((4,), dtype=torch.float32),
        }
        for convrot_group in (None, 0, 16, 1024):
            config = w4a8_layer_config(16, convrot_group_size=convrot_group)
            metadata = quantization_metadata({base: config})
            with self.subTest(convrot_group=convrot_group):
                self.assertFalse(int8_convrot.detect(state_dict, metadata=metadata)["matched"])
                self.assertTrue(asym_w4a8_int8.detect(state_dict, metadata=metadata)["matched"])

    def test_marker_only_w4a8_cannot_claim_a_nondefault_runtime_group(self):
        import torch

        from shared.qtypes import int8_convrot

        base = "linear"
        state_dict = {
            base + ".weight": torch.ones((4, 128), dtype=torch.int8),
            base + ".weight_s_rel": torch.ones((4, 16), dtype=torch.float8_e4m3fn),
            base + ".weight_s_channel": torch.ones((4,), dtype=torch.float32),
            base + ".comfy_quant": torch.tensor(
                list(
                    json.dumps(
                        {
                            "format": "asym_w4a8_int8",
                            "group_size": 16,
                            "convrot": True,
                            "convrot_groupsize": 64,
                        },
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                dtype=torch.uint8,
            ),
        }
        with self.assertRaisesRegex(ValueError, "needs _quantization_metadata"):
            int8_convrot.detect(state_dict)

    def test_nonobject_comfy_json_is_ignored_unless_w4a8_evidence_is_present(self):
        import torch

        from shared.qtypes import int8_convrot

        marker = torch.tensor(list(b"[]"), dtype=torch.uint8)
        unrelated = {"linear.comfy_quant": marker}
        self.assertFalse(int8_convrot.detect(unrelated)["matched"])

        base = "linear"
        w4a8_state = {
            base + ".weight": torch.ones((4, 128), dtype=torch.int8),
            base + ".weight_s_rel": torch.ones((4, 16), dtype=torch.float8_e4m3fn),
            base + ".weight_s_channel": torch.ones((4,), dtype=torch.float32),
            base + ".comfy_quant": marker,
        }
        metadata = quantization_metadata({base: w4a8_layer_config(16)})
        with self.assertRaisesRegex(ValueError, "Invalid Comfy quantization marker"):
            int8_convrot.detect(w4a8_state, metadata=metadata)

    def test_uniform_w4a8_route_consumes_its_marker_without_claiming_int8(self):
        import torch

        from mmgp import offload, quant_router
        from safetensors.torch import save_file
        from shared.qtypes import asym_w4a8_int8, int8_convrot

        handlers = (
            "shared.qtypes.asym_w4a8_int8",
            "shared.qtypes.int8_convrot",
        )
        registered = set(getattr(quant_router, "_HANDLER_MODULES", ()))
        for handler in handlers:
            quant_router.register_handler(handler)
            if handler not in registered:
                self.addCleanup(quant_router.unregister_handler, handler)

        rows, logical_k = 4, 256
        base = "w4"
        config = {
            "format": "asym_w4a8_int8",
            "group_size": 4,
            "convrot": True,
            "convrot_groupsize": 256,
        }
        metadata = quantization_metadata({base: config})
        state = {
            base + ".weight": torch.ones((rows, logical_k // 2), dtype=torch.int8),
            base + ".weight_s_rel": torch.ones(
                (rows, logical_k // 4), dtype=torch.float8_e4m3fn
            ),
            base + ".weight_s_channel": torch.ones((rows,), dtype=torch.float32),
            base + ".comfy_quant": torch.tensor(
                list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
                dtype=torch.uint8,
            ),
        }
        self.assertFalse(int8_convrot.detect(state, metadata=metadata)["matched"])
        self.assertTrue(asym_w4a8_int8.detect(state, metadata=metadata)["matched"])

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "uniform-w4.safetensors"
            save_file(state, str(checkpoint), metadata=metadata)
            model = torch.nn.Module()
            model.w4 = torch.nn.Linear(logical_k, rows, bias=False, dtype=torch.float32)
            offload.load_model_data(
                model,
                str(checkpoint),
                default_dtype=torch.float32,
                verboseLevel=0,
            )
            self.assertIsInstance(
                model.w4.weight, asym_w4a8_int8.AsymW4A8Int8WeightTensor
            )
            self.assertTrue(
                all(not key.endswith(".comfy_quant") for key in model.state_dict())
            )
            del model
            gc.collect()

    def test_header_names_match_only_supported_wrappers_and_keep_raw_state_keys(self):
        import torch

        from shared.qtypes import int8_convrot

        weight = torch.ones((4, 64), dtype=torch.int8)
        scale = torch.ones((4, 1), dtype=torch.float32)
        state_dict = {
            "module.blocks.0.attn.qkv_proj.weight": weight,
            "module.blocks.0.attn.qkv_proj.weight_scale": scale,
        }
        metadata = quantization_metadata(
            {
                "model.diffusion_model.blocks.0.attn.qkv_proj.weight": int8_layer_config(64)
            }
        )

        detection = int8_convrot.detect(state_dict, metadata=metadata)
        self.assertTrue(detection["matched"])
        self.assertEqual(detection["details"]["names"], ["module.blocks.0.attn.qkv_proj"])
        converted = int8_convrot.convert_to_quanto(
            state_dict.copy(),
            default_dtype=torch.float32,
            detection=detection,
            metadata=metadata,
        )
        self.assertIn("module.blocks.0.attn.qkv_proj.weight._data", converted["state_dict"])
        self.assertEqual(converted["quant_map"]["module.blocks.0.attn.qkv_proj"]["weights"], "qint8_convrot")

    def test_ambiguous_wrapped_layer_names_are_rejected(self):
        import torch

        from shared.qtypes import int8_convrot

        weight = torch.ones((4, 64), dtype=torch.int8)
        scale = torch.ones((4, 1), dtype=torch.float32)
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": weight,
            "blocks.0.attn.qkv_proj.weight_scale": scale,
            "module.blocks.0.attn.qkv_proj.weight": weight,
            "module.blocks.0.attn.qkv_proj.weight_scale": scale,
        }
        metadata = quantization_metadata(
            {"blocks.0.attn.qkv_proj": int8_layer_config(64)}
        )
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            int8_convrot.detect(state_dict, metadata=metadata)

    def test_metadata_fields_and_row_scale_are_strict(self):
        import torch

        from shared.qtypes import int8_convrot

        state_dict = {
            "linear.weight": torch.ones((4, 64), dtype=torch.int8),
            "linear.weight_scale": torch.ones((4, 1), dtype=torch.float32),
        }
        cases = (
            (int8_layer_config(64) | {"convrot": "true"}, "convrot"),
            (int8_layer_config(64) | {"convrot_groupsize": 64.0}, "integer group size"),
            (int8_layer_config(64) | {"group_size": 256}, "Conflicting"),
            (int8_layer_config(64, format="int8_tensorwise_custom"), "Unsupported INT8"),
        )
        for config, message in cases:
            metadata = quantization_metadata({"linear": config})
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                int8_convrot.detect(state_dict, metadata=metadata)

        with self.assertRaisesRegex(ValueError, "Unsupported INT8 ConvRot group size"):
            int8_convrot.detect(
                state_dict,
                metadata=quantization_metadata({"linear": int8_layer_config(32)}),
            )

        invalid_scale = dict(state_dict)
        invalid_scale["linear.weight_scale"] = torch.ones((4,), dtype=torch.float32)
        metadata = quantization_metadata({"linear": int8_layer_config(64)})
        with self.assertRaisesRegex(ValueError, "float32\[4, 1\]"):
            int8_convrot.detect(invalid_scale, metadata=metadata)

        invalid_scale["linear.weight_scale"] = torch.ones((4, 1), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "float32\[4, 1\]"):
            int8_convrot.detect(invalid_scale, metadata=metadata)

    def test_present_marker_is_authoritative_but_must_agree_with_header(self):
        import torch

        from shared.qtypes import int8_convrot

        base = "linear"
        weight = torch.ones((4, 64), dtype=torch.int8)
        scale = torch.ones((4, 1), dtype=torch.float32)
        state_dict = {
            base + ".weight": weight,
            base + ".weight_scale": scale,
            base + ".comfy_quant": torch.tensor(
                list(json.dumps(int8_layer_config(256)).encode("utf-8")),
                dtype=torch.uint8,
            ),
        }
        metadata = quantization_metadata(
            {
                base: {
                    "format": "int8_convrot",
                    "convrot": True,
                    "convrot_group_size": 64,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "disagree"):
            int8_convrot.detect(state_dict, metadata=metadata)

        state_dict[base + ".comfy_quant"] = torch.tensor(
            list(
                json.dumps(
                    {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "group_size": 64,
                    }
                ).encode("utf-8")
            ),
            dtype=torch.uint8,
        )
        detection = int8_convrot.detect(state_dict, metadata=metadata)
        converted = int8_convrot.convert_to_quanto(
            state_dict.copy(),
            default_dtype=torch.float32,
            detection=detection,
            metadata=metadata,
        )
        self.assertEqual(converted["state_dict"][base + ".convrot_group_size"].item(), 64)

    def test_mmgp_loads_serialized_metadata_only_qkv_and_mlp_on_cpu(self):
        import torch
        from mmgp import offload, quant_router
        from models.minimax_h3.transformer import get_linear_split_map
        from safetensors.torch import save_file
        from shared.qtypes import int8_convrot

        registered = set(getattr(quant_router, "_HANDLER_MODULES", ()))
        handler_name = "shared.qtypes.int8_convrot"
        quant_router.register_handler(handler_name)
        try:
            for group_size in (64, 256):
                with self.subTest(group_size=group_size), tempfile.TemporaryDirectory() as directory:
                    qkv_weight = torch.arange(12 * group_size, dtype=torch.int32)
                    qkv_weight = qkv_weight.reshape(12, group_size).remainder(127).to(torch.int8)
                    qkv_scale = torch.arange(1, 13, dtype=torch.float32).reshape(12, 1)
                    mlp_weight = torch.arange(8 * group_size, dtype=torch.int32)
                    mlp_weight = mlp_weight.reshape(8, group_size).remainder(127).to(torch.int8)
                    mlp_scale = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)
                    checkpoint = Path(directory) / "metadata_only.safetensors"
                    save_file(
                        {
                            "attn.qkv_proj.weight": qkv_weight,
                            "attn.qkv_proj.weight_scale": qkv_scale,
                            "mlp.fc1.weight": mlp_weight,
                            "mlp.fc1.weight_scale": mlp_scale,
                        },
                        str(checkpoint),
                        metadata=quantization_metadata(
                            {
                                "attn.qkv_proj": int8_layer_config(group_size),
                                "mlp.fc1.weight": {
                                    "format": "int8_convrot",
                                    "convrot": True,
                                    "convrot_group_size": group_size,
                                },
                            }
                        ),
                    )

                    model = torch.nn.Module()
                    model.attn = torch.nn.Module()
                    model.attn.qkv_proj = torch.nn.Linear(
                        group_size, 12, bias=False, dtype=torch.float32
                    )
                    model.mlp = torch.nn.Module()
                    model.mlp.fc1 = torch.nn.Linear(
                        group_size, 8, bias=False, dtype=torch.float32
                    )
                    split_map = get_linear_split_map(
                        4,
                        interleaved=False,
                        num_attention_heads=2,
                        attention_head_dim=2,
                    )
                    offload.split_linear_modules(model, split_map)

                    offload.load_model_data(
                        model,
                        str(checkpoint),
                        default_dtype=torch.float32,
                        fused_split_map=split_map,
                        verboseLevel=0,
                    )

                    expected_qkv_rows = {
                        "q_proj": slice(0, 4),
                        "k_proj": slice(4, 8),
                        "v_proj": slice(8, 12),
                    }
                    for name, rows in expected_qkv_rows.items():
                        module = getattr(model.attn, name)
                        self.assertTrue(torch.equal(module.weight._data, qkv_weight[rows]))
                        self.assertTrue(
                            torch.equal(module.weight._scale[:, 0], qkv_scale[rows, 0])
                        )
                        self.assertEqual(module._convrot_group_size, group_size)
                    self.assertTrue(torch.equal(model.mlp.fc1.weight._data, mlp_weight))
                    self.assertTrue(
                        torch.equal(model.mlp.fc1.weight._scale, mlp_scale)
                    )
                    self.assertEqual(model.mlp.fc1._convrot_group_size, group_size)
                    inputs = torch.linspace(
                        -1.0, 1.0, 2 * group_size, dtype=torch.float32
                    ).reshape(2, group_size)
                    rotated_inputs = inputs @ explicit_hadamard(torch, group_size)
                    expected_q = torch.nn.functional.linear(
                        rotated_inputs,
                        qkv_weight[:4].to(torch.float32) * qkv_scale[:4],
                    )
                    torch.testing.assert_close(
                        model.attn.q_proj(inputs), expected_q, rtol=1e-5, atol=1e-5
                    )
                    expected_mlp = torch.nn.functional.linear(
                        rotated_inputs,
                        mlp_weight.to(torch.float32) * mlp_scale,
                    )
                    torch.testing.assert_close(
                        model.mlp.fc1(inputs), expected_mlp, rtol=1e-5, atol=1e-5
                    )
                    self.assertFalse(any(
                        key.endswith(".comfy_quant")
                        for key in model.state_dict()
                    ))
                    # MMGP memory-maps SafeTensor storage on Windows. Release
                    # module parameters before TemporaryDirectory removes it.
                    del module
                    del model
                    gc.collect()
        finally:
            if handler_name not in registered:
                quant_router.unregister_handler(handler_name)

    def test_mmgp_loads_mixed_w4a8_int8_and_runs_both_cpu_forwards(self):
        import torch
        from mmgp import offload, quant_router
        from safetensors.torch import save_file
        from shared.qtypes import asym_w4a8_int8, int8_convrot

        registered = set(getattr(quant_router, "_HANDLER_MODULES", ()))
        handlers = (
            "shared.qtypes.asym_w4a8_int8",
            "shared.qtypes.int8_convrot",
        )
        temporary_directory = tempfile.TemporaryDirectory()
        for handler in handlers:
            quant_router.register_handler(handler)
        try:
            rows, logical_k = 8, 256
            w4_base, int8_base = "w4", "int8"
            for convrot_group in (64, 256):
                for w4_group in (4, 8, 16):
                    with self.subTest(convrot_group=convrot_group, w4_group=w4_group):
                        torch.manual_seed(7421 + convrot_group + w4_group)
                        w4_weight = torch.randint(
                            -128, 128, (rows, logical_k // 2), dtype=torch.int8
                        )
                        w4_relative = torch.linspace(
                            0.45, 1.25, rows * (logical_k // w4_group), dtype=torch.float32
                        ).reshape(rows, logical_k // w4_group).to(torch.float8_e4m3fn)
                        w4_channel = torch.linspace(0.025, 0.09, rows, dtype=torch.float32)
                        int8_weight = torch.randint(-32, 32, (rows, logical_k), dtype=torch.int8)
                        int8_scale = torch.linspace(0.01, 0.08, rows, dtype=torch.float32).reshape(rows, 1)
                        w4_config = {
                            "format": "asym_w4a8_int8",
                            "group_size": w4_group,
                            "convrot": True,
                            "convrot_groupsize": convrot_group,
                        }
                        int8_config = int8_layer_config(convrot_group)
                        checkpoint = Path(temporary_directory.name) / f"mixed-{convrot_group}-{w4_group}.safetensors"
                        save_file(
                            {
                                w4_base + ".weight": w4_weight,
                                w4_base + ".weight_s_rel": w4_relative,
                                w4_base + ".weight_s_channel": w4_channel,
                                w4_base + ".comfy_quant": torch.tensor(
                                    list(json.dumps(w4_config, separators=(",", ":")).encode("utf-8")),
                                    dtype=torch.uint8,
                                ),
                                int8_base + ".weight": int8_weight,
                                int8_base + ".weight_scale": int8_scale,
                                int8_base + ".comfy_quant": torch.tensor(
                                    list(json.dumps(int8_config, separators=(",", ":")).encode("utf-8")),
                                    dtype=torch.uint8,
                                ),
                            },
                            str(checkpoint),
                            metadata=quantization_metadata({w4_base: w4_config, int8_base: int8_config}),
                        )
                        model = torch.nn.Module()
                        model.w4 = torch.nn.Linear(logical_k, rows, bias=False, dtype=torch.float32)
                        model.int8 = torch.nn.Linear(logical_k, rows, bias=False, dtype=torch.float32)
                        offload.load_model_data(
                            model,
                            str(checkpoint),
                            default_dtype=torch.float32,
                            verboseLevel=0,
                        )

                        self.assertIsInstance(model.w4.weight, asym_w4a8_int8.AsymW4A8Int8WeightTensor)
                        self.assertIsInstance(model.int8.weight, int8_convrot.Int8ConvRotWeightTensor)
                        sample = torch.linspace(-1.0, 1.0, 2 * logical_k, dtype=torch.float32).reshape(2, logical_k)
                        hadamard = explicit_hadamard(torch, convrot_group)
                        rotated = (
                            sample.reshape(-1, logical_k // convrot_group, convrot_group) @ hadamard
                        ).reshape_as(sample)

                        expected_int8 = torch.nn.functional.linear(
                            rotated,
                            int8_weight.to(torch.float32) * int8_scale,
                        )
                        torch.testing.assert_close(
                            model.int8(sample), expected_int8, rtol=1e-5, atol=1e-5
                        )

                        packed = w4_weight.to(torch.uint8)
                        indices = torch.empty((rows, logical_k), dtype=torch.uint8)
                        indices[:, 0::2] = packed & 0x0F
                        indices[:, 1::2] = packed >> 4
                        decoded_w4 = (indices.to(torch.float32) - 8.0).reshape(
                            rows, logical_k // w4_group, w4_group
                        )
                        decoded_w4 = (
                            decoded_w4 * w4_relative.to(torch.float32).unsqueeze(-1)
                        ).round().clamp(-127, 127).reshape(rows, logical_k)
                        activation_scale = rotated.abs().amax(dim=-1, keepdim=True).float().div(127).clamp(min=1e-30)
                        quantized_activation = rotated.div(activation_scale).round().clamp(-128, 127).to(torch.int8)
                        expected_w4 = torch.nn.functional.linear(
                            quantized_activation.to(torch.float32), decoded_w4
                        ) * activation_scale * w4_channel.unsqueeze(0)
                        torch.testing.assert_close(
                            model.w4(sample), expected_w4, rtol=1e-5, atol=1e-5
                        )

                        del expected_w4, expected_int8, decoded_w4, quantized_activation
                        del activation_scale, hadamard, rotated, sample, model
                        gc.collect()
        finally:
            for handler in handlers:
                if handler not in registered:
                    quant_router.unregister_handler(handler)
            temporary_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
