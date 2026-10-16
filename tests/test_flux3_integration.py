"""FLUX 3 catalog, dispatch, download and readiness contracts without a GPU server."""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.flux3.flux3_handler import (  # noqa: E402
    COMPONENT_FOLDER, ENCODER_FILES, MODEL_TYPE, REPO, REVISION, ROOT_FILES, family_handler,
)

DEFINITION = json.loads((ROOT / "app/defaults/flux3_klein.json").read_text(encoding="utf-8"))


def production_function(file, name, namespace):
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec"), namespace)
    return namespace[name]


class Flux3IntegrationTests(unittest.TestCase):
    def test_model_has_its_own_registered_native_architecture(self):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        handlers = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "family_handlers" for t in n.targets))
        self.assertIn("models.flux3.flux3_handler", ast.literal_eval(handlers))
        self.assertEqual(family_handler.query_supported_types(), [MODEL_TYPE])
        self.assertEqual(family_handler.query_family_maps(), ({}, {}))
        self.assertEqual(DEFINITION["model"]["architecture"], MODEL_TYPE)

    def test_distilled_recipe_and_active_capabilities(self):
        settings = {}
        family_handler.update_default_settings(MODEL_TYPE, {}, settings)
        self.assertEqual((settings["num_inference_steps"], settings["guidance_scale"]), (16, 1.0))
        spec = family_handler.query_model_def(MODEL_TYPE, {})
        self.assertTrue(spec["image_outputs"])
        self.assertTrue(spec["no_negative_prompt"])
        self.assertTrue(spec["loras_disabled"])
        self.assertFalse(spec["inpaint_support"])
        self.assertFalse(spec["auto_quantize"])
        self.assertEqual(spec["max_image_refs"], 10)
        self.assertEqual(spec["image_ref_choices"]["default"], "I")

    def test_named_checkpoint_pins_exact_official_flow_map_bytes(self):
        from shared.checkpoint_downloads import find_named_checkpoint_source
        model = DEFINITION["model"]
        filename = model["URLs"][0]
        self.assertNotEqual(filename, "model.safetensors")
        source = find_named_checkpoint_source(model, filename)
        self.assertEqual(source["url"], f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors")
        self.assertEqual(source["size_bytes"], 11097404512)
        self.assertEqual(source["sha256"], "c3e9813cdd049c831c836b2e9db61150e49a93fc1a4c63b256494642fee0bd7c")

    def test_guard_rejects_inert_controls_and_foreign_adapters(self):
        def validate(values):
            return family_handler.validate_generative_settings(MODEL_TYPE, {}, values)

        self.assertIsNone(validate({"image_mode": 1, "video_prompt_type": "I", "image_refs": ["x"] * 10}))
        self.assertIsNone(validate({"video_prompt_type": "", "image_refs": ["inactive"] * 12}))
        self.assertIn("10 reference", validate({"video_prompt_type": "I", "image_refs": ["x"] * 11}))
        self.assertIn("Masked inpainting", validate({"image_mode": 2}))
        self.assertIn("LoRA", validate({"activated_loras": ["flux2_adapter.safetensors"]}))
        self.assertIn("reference-image", validate({"video_prompt_type": "V"}))
        self.assertIn("Euler", validate({"sample_solver": "unipc"}))

    def test_production_download_dispatch_is_scoped_complete_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = set(ROOT_FILES) | {f"text_encoder/{f}" for f in ENCODER_FILES}
            def download(*, repo_id, revision, filename, local_dir, subfolder=""):
                self.assertEqual((repo_id, revision), (REPO, REVISION))
                remote = "/".join(x for x in (subfolder, filename) if x)
                self.assertIn(remote, allowed)
                target = Path(local_dir) / remote
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fixture")
                return str(target)
            downloader = Mock(side_effect=download)
            dispatcher = production_function("app/wgp.py", "process_files_def", {
                "os": os, "hf_download_with_public_fallback": downloader,
                "fl": SimpleNamespace(get_smart_download_root=lambda _: str(root),
                                      get_checkpoints_paths=lambda: [str(root)]),
            })
            definition = family_handler.query_model_files(None, MODEL_TYPE)[0]
            dispatcher(**definition)
            actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(actual, {f"{COMPONENT_FOLDER}/{f}" for f in allowed})
            required = set(family_handler.query_model_def(MODEL_TYPE, {})["required_model_assets"])
            self.assertEqual(actual, required)
            downloader.reset_mock()
            dispatcher(**definition)
            downloader.assert_not_called()
            missing = root / COMPONENT_FOLDER / "text_encoder/tokenizer.json"
            missing.unlink()
            dispatcher(**definition)
            self.assertEqual(downloader.call_count, 1)
            self.assertTrue(missing.exists())

    def test_ready_status_requires_every_component(self):
        model = {**DEFINITION["model"], **family_handler.query_model_def(MODEL_TYPE, {})}
        existing = set(model["required_model_assets"])
        ready = production_function("app/launch.py", "_check_model_downloaded", {
            "_model_weight_groups": lambda _: [model["URLs"]],
            "_variant_group_downloaded": lambda *a, **kw: True,
            "os": os,
            "wgp": SimpleNamespace(get_model_def=lambda _: model,
                                   get_model_recursive_prop=lambda *a, **kw: [],
                                   fl=SimpleNamespace(locate_file=lambda p, **kw: p if p in existing else None)),
        })
        self.assertTrue(ready(MODEL_TYPE))
        existing.remove(f"{COMPONENT_FOLDER}/text_encoder/model-00002-of-00002.safetensors")
        self.assertFalse(ready(MODEL_TYPE))

    def test_engine_downloads_verified_named_weights_then_component_set(self):
        definition = {**DEFINITION["model"], **family_handler.query_model_def(MODEL_TYPE, {})}
        with tempfile.TemporaryDirectory() as directory:
            destination = str(Path(directory) / DEFINITION["model"]["URLs"][0])
            components = Mock()
            namespace = {
                "os": os, "get_base_model_type": lambda _: MODEL_TYPE,
                "get_runtime_model_def": lambda _: definition,
                "ensure_audio_condition_assets": lambda _: None,
                "model_types_handlers": {MODEL_TYPE: family_handler},
                "get_compatible_local_model_filename": lambda *a, **kw: None,
                "fl": SimpleNamespace(get_smart_download_location=lambda *a, **kw: destination),
                "server_config": {}, "get_model_recursive_prop": lambda *a, **kw: [],
                "process_files_def": components,
            }
            download = production_function("app/wgp.py", "download_models", namespace)
            with patch("shared.checkpoint_downloads.download_named_checkpoint") as named:
                download(DEFINITION["model"]["URLs"][0], MODEL_TYPE)
            self.assertEqual(named.call_args.args[0], definition["download_sources"][Path(destination).name])
            self.assertEqual(named.call_args.args[1], destination)
            components.assert_called_once_with(**family_handler.query_model_files(None, MODEL_TYPE)[0])


if __name__ == "__main__":
    unittest.main()
