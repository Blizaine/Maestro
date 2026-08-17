"""Regressions for sharing MiniMax H3 assets with linked WanGP installs."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_WGP_PATH = _ROOT / "app" / "wgp.py"
_LAUNCH_PATH = _ROOT / "app" / "launch.py"
_LOCATOR_PATH = _ROOT / "app" / "shared" / "utils" / "files_locator.py"
_CONVROT_PATH = _ROOT / "app" / "models" / "minimax_h3" / "convrot_layout.py"
_HANDLER_PATH = _ROOT / "app" / "models" / "minimax_h3" / "minimax_h3_handler.py"
_H3_MAIN_PATH = _ROOT / "app" / "models" / "minimax_h3" / "minimax_h3_main.py"
_DEFAULTS_DIR = _ROOT / "app" / "defaults"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_wgp_resolver(model_def: dict, locator):
    tree = ast.parse(_WGP_PATH.read_text(encoding="utf-8"), filename=str(_WGP_PATH))
    wanted = {"get_local_model_filename", "get_compatible_local_model_filename"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {
        "os": os,
        "fl": locator,
        "get_model_def": lambda _model_type: model_def,
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_WGP_PATH), "exec"), namespace)
    return namespace["get_compatible_local_model_filename"]


def _load_model_filename_selector(models_def: dict):
    tree = ast.parse(_WGP_PATH.read_text(encoding="utf-8"), filename=str(_WGP_PATH))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_model_filename"
    ]

    class _QuantRouter:
        @staticmethod
        def get_quantization_tokens(quantization):
            return [str(quantization).lower()]

    float16 = object()
    namespace = {
        "os": os,
        "models_def": models_def,
        "quant_router": _QuantRouter(),
        "torch": SimpleNamespace(float16=float16, bfloat16=object()),
        "get_transformer_dtype": lambda *_args, **_kwargs: object(),
        "get_base_model_type": lambda model_type: model_type,
        "get_model_recursive_prop": lambda *_args, **_kwargs: [],
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_WGP_PATH), "exec"), namespace)
    return namespace["get_model_filename"]


def _load_launch_variant_helpers(model_def: dict, locator):
    tree = ast.parse(_LAUNCH_PATH.read_text(encoding="utf-8"), filename=str(_LAUNCH_PATH))
    wanted = {"_variant_group_filenames", "_variant_group_downloaded"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {
        "os": os,
        "wgp": SimpleNamespace(
            get_model_def=lambda _model_type: model_def,
            fl=locator,
        ),
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_LAUNCH_PATH), "exec"), namespace)
    return (
        namespace["_variant_group_filenames"],
        namespace["_variant_group_downloaded"],
    )


class TestMiniMaxH3AssetSharing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.maestro_app = base / "Maestro" / "app"
        self.primary = self.maestro_app / "ckpts"
        self.linked = base / "wan.git" / "app" / "ckpts"
        self.primary.mkdir(parents=True)
        self.linked.mkdir(parents=True)
        self.locator = _load_module(
            _LOCATOR_PATH,
            f"maestro_files_locator_sharing_{id(self)}",
        )
        self.locator._APP_DIR = str(self.maestro_app)
        self.locator.set_checkpoints_paths([str(self.primary), str(self.linked)])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exact_canonical_checkpoint_wins_over_compatible_alias(self):
        canonical = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
        alternate = "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors"
        canonical_path = self.primary / canonical
        alternate_path = self.linked / alternate
        canonical_path.write_bytes(b"maestro")
        alternate_path.write_bytes(b"wangp")
        resolver = _load_wgp_resolver(
            {"compatible_model_paths": {canonical: [alternate]}},
            self.locator,
        )

        result = resolver(
            f"https://huggingface.invalid/{canonical}",
            "minimax_h3",
            file_type=0,
        )

        self.assertEqual(Path(result), canonical_path)

    def test_linked_wangp_pruned_checkpoint_prevents_duplicate_download(self):
        canonical = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
        alternate = "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors"
        alternate_path = self.linked / alternate
        alternate_path.write_bytes(b"wangp")
        resolver = _load_wgp_resolver(
            {"compatible_model_paths": {canonical: [alternate]}},
            self.locator,
        )

        result = resolver(
            f"https://huggingface.invalid/{canonical}",
            "minimax_h3",
            file_type=0,
        )

        self.assertEqual(Path(result), alternate_path)
        source = self.locator.describe_file_source(result)
        self.assertEqual(source["kind"], "linked")
        self.assertEqual(source["installation"], "wan.git")

    def test_existing_maestro_fp8_prevents_new_default_int8_download(self):
        canonical = "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors"
        legacy = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
        legacy_path = self.primary / legacy
        legacy_path.write_bytes(b"legacy maestro fp8")
        resolver = _load_wgp_resolver(
            {"compatible_model_paths": {canonical: [legacy]}},
            self.locator,
        )

        result = resolver(
            f"https://huggingface.invalid/{canonical}",
            "minimax_h3",
            file_type=0,
        )

        self.assertEqual(Path(result), legacy_path)

    def test_model_readiness_accepts_a_linked_compatible_checkpoint(self):
        canonical = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
        alternate = "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors"
        alternate_path = self.linked / alternate
        alternate_path.write_bytes(b"wangp")
        model_def = {
            "compatible_model_paths": {
                canonical: [alternate],
                alternate: [canonical],
            }
        }
        filenames, downloaded = _load_launch_variant_helpers(model_def, self.locator)
        urls = [f"https://huggingface.invalid/{canonical}"]

        self.assertEqual(
            filenames(urls, model_type="minimax_h3"),
            [canonical, alternate],
        )
        self.assertTrue(downloaded(urls, model_type="minimax_h3"))

    def test_all_four_h3_models_route_int8_and_bf16_like_wangp(self):
        cases = {
            "minimax_h3.json": (
                "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors",
                "MiniMax-H3-FL2VA-pruned_rank8_bf16.safetensors",
            ),
            "minimax_h3_ref2va.json": (
                "MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors",
                "MiniMax-H3-Ref2VA-pruned_rank8_bf16.safetensors",
            ),
            "minimax_h3_full.json": (
                "MiniMax-H3-FL2VA_int8_convrot.safetensors",
                "MiniMax-H3-FL2VA_bf16.safetensors",
            ),
            "minimax_h3_ref2va_full.json": (
                "MiniMax-H3-Ref2VA_int8_convrot.safetensors",
                "MiniMax-H3-Ref2VA_bf16.safetensors",
            ),
        }
        for filename, (expected_int8, expected_bf16) in cases.items():
            with self.subTest(defaults=filename):
                payload = json.loads((_DEFAULTS_DIR / filename).read_text(encoding="utf-8"))
                model_type = payload["model"]["architecture"]
                selector = _load_model_filename_selector({model_type: payload["model"]})
                self.assertEqual(
                    os.path.basename(selector(model_type, quantization="int8")),
                    expected_int8,
                )
                self.assertEqual(
                    os.path.basename(selector(model_type, quantization="bf16")),
                    expected_bf16,
                )

    def test_all_four_int8_models_reuse_exact_linked_wangp_checkpoints(self):
        handler = _load_module(
            _HANDLER_PATH,
            f"maestro_h3_handler_linked_matrix_{id(self)}",
        )
        cases = {
            "minimax_h3.json": "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors",
            "minimax_h3_ref2va.json": "MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors",
            "minimax_h3_full.json": "MiniMax-H3-FL2VA_int8_convrot.safetensors",
            "minimax_h3_ref2va_full.json": "MiniMax-H3-Ref2VA_int8_convrot.safetensors",
        }
        for filename, expected_checkpoint in cases.items():
            with self.subTest(defaults=filename):
                payload = json.loads((_DEFAULTS_DIR / filename).read_text(encoding="utf-8"))
                model_type = payload["model"]["architecture"]
                selector = _load_model_filename_selector({model_type: payload["model"]})
                selected_url = selector(model_type, quantization="int8")
                linked_path = self.linked / expected_checkpoint
                linked_path.write_bytes(b"linked wangp int8")
                model_def = handler.family_handler.query_model_def(model_type, {})
                resolver = _load_wgp_resolver(model_def, self.locator)

                result = resolver(selected_url, model_type, file_type=0)

                self.assertEqual(os.path.basename(selected_url), expected_checkpoint)
                self.assertEqual(Path(result), linked_path)

    def test_pruned_bf16_and_int8_use_wangp_interleaved_qkv_layout(self):
        handler = _load_module(
            _HANDLER_PATH,
            f"maestro_h3_handler_sharing_{id(self)}",
        )
        cases = {
            "minimax_h3": (
                "MiniMax-H3-FL2VA-pruned_rank8_bf16.safetensors",
                "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors",
            ),
            "minimax_h3_ref2va": (
                "MiniMax-H3-Ref2VA-pruned_rank8_bf16.safetensors",
                "MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors",
            ),
        }
        for model_type, checkpoint_names in cases.items():
            with self.subTest(model_type=model_type):
                model_def = handler.family_handler.query_model_def(model_type, {})
                layouts = model_def["compatible_model_qkv_layouts"]
                for checkpoint_name in checkpoint_names:
                    self.assertEqual(layouts[checkpoint_name], "interleaved")

    def test_qwen_encoder_resolves_wangp_folder_layout(self):
        filename = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        relative = os.path.join("Qwen3-VL-32B-Instruct", filename)
        linked_path = self.linked / relative
        linked_path.parent.mkdir(parents=True)
        linked_path.write_bytes(b"same published qwen artifact")
        resolver = _load_wgp_resolver(
            {"compatible_text_encoder_paths": {filename: [relative]}},
            self.locator,
        )

        result = resolver(
            f"https://huggingface.invalid/{filename}",
            "minimax_h3",
            file_type=2,
            extra_paths="minimax_h3",
        )

        self.assertEqual(Path(result), linked_path)

    def test_different_wangp_vae_name_is_not_treated_as_compatible(self):
        wangp_vae = self.linked / "MiniMax-H3-video_vae_fp16.safetensors"
        wangp_vae.write_bytes(b"different tensor artifact")
        resolver = _load_wgp_resolver({}, self.locator)

        result = resolver(
            "minimax_h3/vae/minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3",
            file_type=0,
        )

        self.assertIsNone(result)

    def test_loader_uses_compatible_resolver_before_download_and_load(self):
        source = _WGP_PATH.read_text(encoding="utf-8")
        self.assertIn("def get_compatible_local_model_filename(", source)
        self.assertIn(
            "local_model_filename = get_compatible_local_model_filename(",
            source,
        )
        self.assertIn(
            "text_encoder_filename = get_compatible_local_model_filename(",
            source,
        )

        launch_source = _LAUNCH_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "_variant_group_filenames(group, model_type=model_type)",
            launch_source,
        )
        self.assertGreaterEqual(
            launch_source.count("_variant_group_filenames(group, model_type=mt)"),
            2,
        )
        self.assertIn(
            "wgp.get_compatible_local_model_filename(\n"
            "                text_encoder_filename,",
            launch_source,
        )

    def test_convrot_loader_wiring_is_dependency_free(self):
        convrot_source = _CONVROT_PATH.read_text(encoding="utf-8")
        self.assertIn("def has_convrot_layout(", convrot_source)
        self.assertIn('key.endswith(".comfy_quant")', convrot_source)
        self.assertIn("and _is_convrot_config(value)", convrot_source)

        main_source = _H3_MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn('if checkpoint["convrot"]:', main_source)
        self.assertIn('qkv_layout = "interleaved"', main_source)
        self.assertIn("[MiniMax H3 Assets] Resolved component sources:", main_source)

    def test_convrot_metadata_payload_is_detected_when_torch_is_available(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("tensor-level ConvRot probe requires PyTorch")

        convrot = _load_module(
            _CONVROT_PATH,
            f"maestro_h3_convrot_sharing_{id(self)}",
        )
        payload = convrot.torch.tensor(
            list(b'{"convrot": true}'),
            dtype=convrot.torch.uint8,
        )
        self.assertTrue(
            convrot.has_convrot_layout({"blocks.0.attn.qkv_proj.comfy_quant": payload})
        )


if __name__ == "__main__":
    unittest.main()
