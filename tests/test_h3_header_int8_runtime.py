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


if __name__ == "__main__":
    unittest.main()
