"""CPU-only verification for the four pinned official FastH3 OmniRef files."""
import hashlib
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.minimax_h3 import fasth3_omniref as omni


def _contract():
    return {
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "audio_scheduler_shift": 3.0,
        "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da",
        "conditioning": "fixed_ordered_references_target_only_flow",
        "grid_max_t": 0.999,
        "guidance_scale": 1.0,
        "model_type": "ref2va",
        "num_inference_steps": 8,
        "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
        "pdd_steps": 32,
        "schema": "fasth3-inference-contract-v1",
        "schema_version": "fasth3-inference-contract-v1",
        "transformer_component": "transformer_ref",
        "transformer_forwards": 8,
        "video_scheduler_shift": 12.0,
        "vsa_ref_keep_rate": 0.1,
        "vsa_ref_policy": "p2_multi_region",
        "vsa_sparsity": 0.9,
        "vsa_tile_size": 128,
    }


def _metadata(variant):
    if variant == "int8":
        return {
            "format": "pt",
            "minimax_h3": "adaln_basis",
            "source": "transformer_ref",
            "config": json.dumps({"transformer": {"pdd_steps": 32}}),
        }
    metadata = {
        "format": "pt",
        "source_repo": "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
        "source_revision": "88d512c5e7cc38dda499fbb65688ce58611f9c80",
        "comfyui_commit": "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c",
        "comfy_model_tools_commit": "d6797787e6bdb1a1fb0094d588a26f8e71a1c757",
        "pdd_head_storage": "head0-full;heads1-31-offset-from-head0;fp32",
        "source_inference_contract": json.dumps(_contract()),
        "config": json.dumps({"transformer": {
            "pdd_steps": 32,
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "norm_eps": 1e-5,
            "qk_norm_eps": 1e-5,
            "final_norm_eps": 1e-5,
            "time_embed_dim": 8,
        }}),
    }
    if variant in ("fp8", "nvfp4"):
        layers = "all 50 DiT blocks: fused qkv_proj, out_proj, to_gate_compress, mlp.fc1, mlp.fc2"
        dense = "token_refiner, AdaLN, norms, embeddings, patch/condition projections, FP32 PDD heads"
        metadata.update({
            "quantization_recipe": "omniref-comfy-ptq-v1",
            "quantization_source_sha256": (
                "c0fac98c0317eb85215bd186b250be9b113675baab1323add40f26262f552ed9"
                if variant == "fp8"
                else "4e244969987e3c6e16bffa449bcf62c1f7c83b5cce15e5806246d3ae8055154f"
            ),
            "quantization": json.dumps({
                "format": "float8_e4m3fn" if variant == "fp8" else "nvfp4",
                "layout": "TensorCoreFP8Layout" if variant == "fp8" else "TensorCoreNVFP4Layout",
                "weights": "per-tensor global scale" if variant == "fp8" else "per-tensor global scale plus FP8 block scales, group 16",
                "activations": "dynamic-per-tensor",
                "quantized_layers": layers,
                "dense_layers": dense,
                "qkv_treatment": "fused Q/K/V quantized as one tensor under one weight scale",
                "comfyui_commit": "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c",
            }),
        })
    return metadata


def _tensor(dtype, shape):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}


def _header(variant):
    header = {"__metadata__": _metadata(variant), "adaln_t_table": _tensor("F32", [1025, 8])}
    for name, width in (("video", 96), ("audio", 32)):
        header[f"final_layer.{name}_out.weight"] = _tensor("F32", [32 * width, 5376])
        header[f"final_layer.{name}_out.bias"] = _tensor("F32", [32 * width])

    gate = {
        "bf16": ("BF16", [7168, 5376]),
        "int8": ("I8", [7168, 5376]),
        "fp8": ("F8_E4M3", [7168, 5376]),
        "nvfp4": ("U8", [7168, 2688]),
    }[variant]
    qkv = {
        "bf16": ("BF16", [21504, 5376]),
        "int8": ("I8", [21504, 5376]),
        "fp8": ("F8_E4M3", [21504, 5376]),
        "nvfp4": ("U8", [21504, 2688]),
    }[variant]
    for index in range(50):
        gate_base = f"blocks.{index}.attn.to_gate_compress"
        qkv_base = f"blocks.{index}.attn.qkv_proj"
        header[gate_base + ".weight"] = _tensor(*gate)
        header[qkv_base + ".weight"] = _tensor(*qkv)
        if variant == "int8":
            header[gate_base + ".weight_scale"] = _tensor("F32", [7168, 1])
            header[gate_base + ".comfy_quant"] = _tensor("U8", [72])
            header[qkv_base + ".weight_scale"] = _tensor("F32", [21504, 1])
            header[qkv_base + ".comfy_quant"] = _tensor("U8", [72])
        elif variant == "fp8":
            header[gate_base + ".weight_scale"] = _tensor("F32", [])
            header[gate_base + ".comfy_quant"] = _tensor("U8", [26])
            header[qkv_base + ".weight_scale"] = _tensor("F32", [])
            header[qkv_base + ".comfy_quant"] = _tensor("U8", [26])
        elif variant == "nvfp4":
            for base, rows in ((gate_base, 7168), (qkv_base, 21504)):
                header[base + ".weight_scale"] = _tensor("F8_E4M3", [rows, 336])
                header[base + ".weight_scale_2"] = _tensor("F32", [])
                header[base + ".comfy_quant"] = _tensor("U8", [18])
    return header


class OfficialFastH3OmniRefTests(unittest.TestCase):
    def _verify(self, variant, header=None, payload=None):
        payload = payload or f"fixture for {variant}".encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{variant}.safetensors"
            path.write_bytes(payload)
            pins = {name: dict(pin) for name, pin in omni.PINNED_CHECKPOINTS.items()}
            pins[variant].update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            with patch.dict(omni.PINNED_CHECKPOINTS, pins, clear=True):
                return omni.verify_checkpoint(str(path), header or _header(variant))

    def test_all_four_official_header_contracts_are_accepted(self):
        for variant in ("bf16", "int8", "fp8", "nvfp4"):
            with self.subTest(variant=variant):
                self.assertEqual(self._verify(variant), variant)

    def test_official_pins_and_legacy_int8_aliases_are_exact(self):
        self.assertEqual(omni.REPO, "FastVideo/FastH3-Omni-Comfy")
        self.assertEqual(omni.REVISION, "c6e7d1f212c44b21227bbd6e76bfff7da70e285d")
        self.assertEqual(omni.FILE, "fasth3_omniref_8step_pruned_int8_convrot.safetensors")
        self.assertEqual(omni.SIZE, 22213723064)
        self.assertEqual(omni.SHA256, "851b2dc9670ed864b7f1e951ac9da047d83b1c04ded9bfcc1a9e5bba0ee33f80")
        self.assertEqual(omni.PINNED_CHECKPOINTS["int8"]["size"], 22213723064)
        self.assertEqual(omni.PINNED_CHECKPOINTS["int8"]["sha256"], "851b2dc9670ed864b7f1e951ac9da047d83b1c04ded9bfcc1a9e5bba0ee33f80")
        self.assertEqual(omni.PINNED_CHECKPOINTS["bf16"]["revision"], "c6e7d1f212c44b21227bbd6e76bfff7da70e285d")
        self.assertEqual(omni.PINNED_CHECKPOINTS["bf16"]["size"], 44240780704)
        self.assertEqual(omni.PINNED_CHECKPOINTS["bf16"]["sha256"], "c0fac98c0317eb85215bd186b250be9b113675baab1323add40f26262f552ed9")
        self.assertEqual(omni.PINNED_CHECKPOINTS["fp8"]["repo"], "FastVideo/FastH3-Omni-FP8-Comfy")
        self.assertEqual(omni.PINNED_CHECKPOINTS["fp8"]["revision"], "03127300d4803d5b66df66a22208b1d9238c8588")
        self.assertEqual(omni.PINNED_CHECKPOINTS["fp8"]["file"], "fasth3_omniref_fp8_comfy.safetensors")
        self.assertEqual(omni.PINNED_CHECKPOINTS["fp8"]["size"], 23046497628)
        self.assertEqual(omni.PINNED_CHECKPOINTS["fp8"]["sha256"], "d3aa751e151df957b6248db7dbe4a159549dfefe78ba884056b12b0425b019f3")
        self.assertEqual(omni.PINNED_CHECKPOINTS["nvfp4"]["repo"], "FastVideo/FastH3-Omni-NVFP4-QAD-Comfy")
        self.assertEqual(omni.PINNED_CHECKPOINTS["nvfp4"]["revision"], "e1f39b9187e3e4059b7507e0a62fa22621ff77a5")
        self.assertEqual(omni.PINNED_CHECKPOINTS["nvfp4"]["file"], "fasth3_omniref_qad_nvfp4_comfy.safetensors")
        self.assertEqual(omni.PINNED_CHECKPOINTS["nvfp4"]["size"], 13773998700)
        self.assertEqual(omni.PINNED_CHECKPOINTS["nvfp4"]["sha256"], "bf25ee58df65b4dfee349d612f7ac62a2fea0dddd1e68eb37e75c8ce8b166f5a")

    def test_wrong_schedule_or_t2av_contract_is_rejected(self):
        header = _header("bf16")
        contract = json.loads(header["__metadata__"]["source_inference_contract"])
        contract["video_scheduler_shift"] = 10.0
        header["__metadata__"]["source_inference_contract"] = json.dumps(contract)
        with self.assertRaisesRegex(ValueError, "inference contract"):
            self._verify("bf16", header)

    def test_wrong_heads_adaln_gate_count_and_nvfp4_geometry_are_rejected(self):
        cases = []
        bad_heads = _header("bf16")
        bad_heads["final_layer.video_out.weight"] = _tensor("BF16", [3072, 5376])
        cases.append(("bf16", bad_heads, "video_out.weight"))
        bad_adaln = _header("bf16")
        bad_adaln["adaln_t_table"] = _tensor("F32", [1025, 16])
        cases.append(("bf16", bad_adaln, "adaln_t_table"))
        missing_gate = _header("bf16")
        del missing_gate["blocks.49.attn.to_gate_compress.weight"]
        cases.append(("bf16", missing_gate, "exactly 50"))
        bad_nvfp4 = _header("nvfp4")
        bad_nvfp4["blocks.0.attn.to_gate_compress.weight_scale"] = _tensor("F8_E4M3", [7168, 168])
        cases.append(("nvfp4", bad_nvfp4, "weight_scale"))
        for variant, header, expected in cases:
            with self.subTest(variant=variant, expected=expected), self.assertRaisesRegex(ValueError, expected):
                self._verify(variant, header)

    def test_checksum_is_verified_after_size_based_variant_resolution(self):
        payload = b"same size"
        changed = b"bad! size"
        self.assertEqual(len(payload), len(changed))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "misleading_filename.safetensors"
            path.write_bytes(changed)
            pin = {**omni.PINNED_CHECKPOINTS["bf16"], "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            pins = {name: dict(entry) for name, entry in omni.PINNED_CHECKPOINTS.items()}
            pins["bf16"] = pin
            with patch.dict(omni.PINNED_CHECKPOINTS, pins, clear=True), self.assertRaisesRegex(ValueError, "checksum"):
                omni.verify_checkpoint(str(path), _header("bf16"))

    def test_nvfp4_fails_closed_without_blackwell_and_native_kernels(self):
        with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(RuntimeError, "requires a CUDA GPU"):
            omni.require_nvfp4_kernel_support()

        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(9, 0)), \
                self.assertRaisesRegex(RuntimeError, "Blackwell-class"):
            omni.require_nvfp4_kernel_support()

        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(10, 0)), \
                patch.object(omni, "_nvfp4_kernel_supported", return_value=False), \
                self.assertRaisesRegex(RuntimeError, "NVFP4 CUDA kernels"):
            omni.require_nvfp4_kernel_support()

        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(10, 0)), \
                patch.object(omni, "_nvfp4_kernel_supported", return_value=True):
            omni.require_nvfp4_kernel_support()

    def test_loader_routes_official_variants_and_preflights_nvfp4(self):
        from accelerate import init_empty_weights as accelerate_init_empty_weights
        from contextlib import contextmanager

        from models.minimax_h3 import fasth3_attention, turbo

        source_path = ROOT / "app" / "models" / "minimax_h3" / "minimax_h3_main.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_load_transformer"
        )
        loader_module = ast.Module(body=[function], type_ignores=[])

        for variant, expected_route in (
            ("bf16", "split"),
            ("int8", "split"),
            ("fp8", "fused"),
            ("nvfp4", "fused"),
        ):
            with self.subTest(variant=variant):
                events = []
                gate_header = {
                    f"blocks.{index}.attn.to_gate_compress.weight": {
                        "dtype": "U8",
                        "shape": [7168, 2688],
                    }
                    for index in range(50)
                }
                split_map = {"qkv_proj": "split"}

                class TinyTransformer:
                    def __init__(self, **_kwargs):
                        events.append("construct")
                        self.blocks = [
                            SimpleNamespace(attn=SimpleNamespace())
                            for _ in range(50)
                        ]
                        self.final_layer = SimpleNamespace(video_out=None, audio_out=None)
                        self.qkv_layouts = []

                    def set_qkv_layout(self, layout):
                        self.qkv_layouts.append(layout)

                    def eval(self):
                        return self

                    def requires_grad_(self, _enabled):
                        return self

                split_linear_modules = Mock(
                    side_effect=lambda *_args, **_kwargs: events.append("split")
                )
                load_model_data = Mock(
                    side_effect=lambda *_args, **_kwargs: events.append("load")
                )
                get_linear_split_map = Mock(return_value=split_map)

                @contextmanager
                def empty_weights(include_buffers=True):
                    events.append("allocation_context")
                    with accelerate_init_empty_weights(include_buffers=include_buffers):
                        yield

                namespace = {
                    "__name__": "models.minimax_h3._fasth3_official_loader_test",
                    "__package__": "models.minimax_h3",
                    "torch": torch,
                    "partial": __import__("functools").partial,
                    "init_empty_weights": empty_weights,
                    "MiniMaxH3Transformer": TinyTransformer,
                    "_first_path": lambda value: str(value[0] if isinstance(value, (list, tuple)) else value),
                    "probe_h3_checkpoint": lambda _path: (
                        events.append("probe")
                        or {
                            "compressed_modulation": True,
                            "adaln_curve_grid": 1025,
                            "time_embed_dim": 8,
                            "fp32_adaln_modules": [],
                            "quantization_format": "convrot" if variant == "int8" else "none",
                        }
                    ),
                    "_lock_fp32_adaln_modules": lambda *_args, **_kwargs: None,
                    "get_linear_split_map": get_linear_split_map,
                    "offload": SimpleNamespace(
                        split_linear_modules=split_linear_modules,
                        load_model_data=load_model_data,
                    ),
                    "_strip_transformer_wrappers": lambda state_dict, **_kwargs: state_dict,
                }
                exec(
                    compile(ast.fix_missing_locations(loader_module), str(source_path), "exec"),
                    namespace,
                )

                def resolve_variant(path):
                    events.append("resolve_variant")
                    self.assertEqual(path, f"{variant}.safetensors")
                    return variant

                def verify(path, header):
                    events.append("verify")
                    self.assertEqual(path, f"{variant}.safetensors")
                    self.assertIs(header, gate_header)
                    return variant

                def preflight(_device):
                    events.append("preflight")

                with (
                    patch.object(turbo, "safetensors_header", return_value=gate_header),
                    patch.object(omni, "checkpoint_variant_from_size", side_effect=resolve_variant),
                    patch.object(omni, "verify_checkpoint", side_effect=verify),
                    patch.object(omni, "require_nvfp4_kernel_support", side_effect=preflight),
                    patch.object(fasth3_attention, "FastH3Attention", side_effect=lambda **_kwargs: object()),
                ):
                    model = namespace["_load_transformer"](
                        f"{variant}.safetensors",
                        torch.bfloat16,
                        qkv_layout="grouped",
                        fasth3=True,
                        fasth3_omniref=True,
                    )

                self.assertEqual(len(model.blocks), 50)
                self.assertEqual(model.blocks[0].attn.to_gate_compress.weight.device.type, "meta")
                self.assertEqual(model.final_layer.video_out.weight.device.type, "meta")
                load_model_data.assert_called_once()
                if expected_route == "split":
                    get_linear_split_map.assert_called_once()
                    split_linear_modules.assert_called_once_with(model, split_map)
                    self.assertIs(load_model_data.call_args.kwargs["fused_split_map"], split_map)
                    self.assertEqual(model.qkv_layouts, [])
                else:
                    get_linear_split_map.assert_not_called()
                    split_linear_modules.assert_not_called()
                    self.assertIsNone(load_model_data.call_args.kwargs["fused_split_map"])
                    self.assertEqual(model.qkv_layouts, ["grouped"])

                self.assertLess(events.index("verify"), events.index("probe"))
                self.assertLess(events.index("probe"), events.index("allocation_context"))
                self.assertLess(events.index("allocation_context"), events.index("construct"))
                if variant == "nvfp4":
                    self.assertLess(events.index("preflight"), events.index("verify"))
                    self.assertLess(events.index("preflight"), events.index("probe"))
                    self.assertLess(events.index("preflight"), events.index("allocation_context"))
                else:
                    self.assertNotIn("preflight", events)


if __name__ == "__main__":
    unittest.main()
