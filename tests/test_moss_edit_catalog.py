"""Editor asset/readiness contracts; no model download or GPU required."""
import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.moss_image.moss_image_handler import family_handler  # noqa: E402


def catalog(model_id):
    return json.loads((ROOT / f"app/defaults/{model_id}.json").read_text(encoding="utf-8"))


class MossEditCatalogTests(unittest.TestCase):
    def test_editor_recipes_reuse_pinned_sft_weights_and_distinct_adapters(self):
        base = catalog("moss_image_sft")
        for model_id, variant, guidance in (("moss_image_edit", "sft", 4), ("moss_image_edit_hq", "hq_v5", 3)):
            with self.subTest(model=model_id):
                item = catalog(model_id)
                md = item["model"]
                self.assertEqual(md["architecture"], "moss_image")
                self.assertEqual(md["URLs"], base["model"]["URLs"])
                self.assertEqual(md["download_sources"], base["model"]["download_sources"])
                self.assertTrue(md["moss_edit"])
                self.assertEqual(md["moss_edit_variant"], variant)
                self.assertEqual((item["image_mode"], item["video_prompt_type"], item["guidance_scale"]), (1, "I", guidance))
                self.assertEqual((item["num_inference_steps"], item["flow_shift"]), (50, 5))
                self.assertEqual(item["custom_settings"], {"moss_image_kv_cache": "Disabled"})
                self.assertEqual(len(md["loras"]), 1)
                url = md["loras"][0]
                self.assertRegex(url, r"/resolve/[0-9a-f]{40}/")
                provenance = md["bundled_lora_provenance"][url.split("/")[-1]]
                self.assertEqual(provenance["url"], url)
                self.assertRegex(provenance["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(provenance["size_bytes"], 200_000_000)
        self.assertNotEqual(catalog("moss_image_edit")["model"]["loras"], catalog("moss_image_edit_hq")["model"]["loras"])

    def test_capabilities_preserve_text_only_models_and_enable_only_native_edits(self):
        for model_id in ("moss_image", "moss_image_sft", "moss_image_edit", "moss_image_edit_hq"):
            md = catalog(model_id)["model"]
            definition = family_handler.query_model_def("moss_image", md)
            editing = md.get("moss_edit", False)
            self.assertEqual(definition["max_image_refs"], 4 if editing else 0)
            self.assertEqual(definition["at_least_one_image_ref_needed"], editing)
            self.assertEqual(definition["moss_image_prompt_helper"], not editing)
            self.assertFalse(definition.get("inpaint_support", False))
            self.assertEqual("image_ref_choices" in definition, editing)
            if editing:
                self.assertNotIn("Bounding Boxes", str(definition["prompt_enhancer_def"]))
                self.assertIn("what to retain", definition["text_prompt_enhancer_instructions"])
                self.assertIn("flux2_vae.safetensors", definition["required_model_assets"])
                self.assertEqual(len(definition["custom_settings"][0]["choices"]), 4)

    def test_encoder_download_is_edit_only_and_revision_pinned(self):
        plain = family_handler.query_model_files([], "moss_image", {})
        edit = family_handler.query_model_files([], "moss_image", {"moss_edit": True})
        self.assertEqual(len(plain), 1)
        self.assertEqual(edit[:1], plain)
        self.assertEqual(edit[1]["repoId"], "DeepBeepMeep/Flux2")
        self.assertRegex(edit[1]["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(edit[1]["fileList"], [["flux2_vae.safetensors"]])

    def test_hq_requires_vision_assets_without_replacing_sft_text_weights(self):
        definition = family_handler.query_model_def("moss_image", catalog("moss_image_edit_hq")["model"])
        plain = family_handler.query_model_def("moss_image", catalog("moss_image_edit")["model"])
        self.assertEqual(definition["text_encoder_URLs"], plain["text_encoder_URLs"])
        self.assertEqual(definition["moss_edit_vision_pixels"], 512 * 512)
        self.assertEqual(definition["moss_edit_reference_size"], 1024)
        required = {os.path.join("moss_image_hq_vision", name) for name in (
            "config.json", "preprocessor_config.json")}
        required.add(os.path.join("moss_image_hq_vision", "Qwen3_5_9B_Abliterated", "Qwen3.5-9B-vision_bf16.safetensors"))
        self.assertTrue(required.issubset(definition["required_model_assets"]))
        self.assertFalse(required.intersection(plain["required_model_assets"]))
        files = family_handler.query_model_files([], "moss_image", catalog("moss_image_edit_hq")["model"])
        self.assertEqual(files[2]["fileList"], [["Qwen3.5-9B-vision_bf16.safetensors"]])
        self.assertEqual(files[2]["targetFolderList"], ["moss_image_hq_vision"])
        self.assertEqual(files[3]["fileList"], [["config.json", "preprocessor_config.json"]])
        self.assertEqual(files[3]["repoId"], "Qwen/Qwen3.5-9B")
        for entry in files:
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
        self.assertNotIn("Abliterated-text", str(files))

    def test_hq_download_targets_match_required_asset_paths(self):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "process_files_def")
        with tempfile.TemporaryDirectory() as folder:
            downloaded = []

            def download(**kwargs):
                destination = Path(kwargs["local_dir"]) / kwargs.get("subfolder", "") / kwargs["filename"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"download fixture")
                downloaded.append(destination)
                return str(destination)

            namespace = {"os": os, "hf_download_with_public_fallback": download,
                         "fl": SimpleNamespace(get_smart_download_root=lambda _: folder,
                                               get_checkpoints_paths=lambda: [folder])}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "<download-dispatch>", "exec"), namespace)
            md = catalog("moss_image_edit_hq")["model"]
            files = family_handler.query_model_files([], "moss_image", md)
            for entry in files[2:]:
                namespace["process_files_def"](**entry)
            required = family_handler.query_model_def("moss_image", md)["required_model_assets"]
            expected = {Path(folder) / name for name in required if name.startswith("moss_image_hq_vision")}
            self.assertEqual(set(downloaded), expected)
            for entry in files[2:]:
                namespace["process_files_def"](**entry)
            self.assertEqual(len(downloaded), 3, "Complete assets should be reused without another download")

    def test_partial_install_is_not_reported_ready(self):
        tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_check_model_downloaded")
        namespace = {"wgp": None, "os": __import__("os"), "_model_weight_groups": lambda _: [["transformer"]], "_variant_group_downloaded": lambda *a, **k: True}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<readiness>", "exec"), namespace)
        md = family_handler.query_model_def("moss_image", {"moss_edit": True})
        # Every main weight and bundled adapter exists; only companion files
        # vary, exercising the production readiness gate rather than a copy.
        paths = set(md["required_model_assets"])
        paths.add(md["required_model_asset_groups"][0][0])
        wgp = SimpleNamespace(get_model_def=lambda _: md, get_model_recursive_prop=lambda *a, **k: [], fl=SimpleNamespace(locate_file=lambda name, **k: name if name in paths else None))
        namespace["wgp"] = wgp
        self.assertTrue(namespace["_check_model_downloaded"]("moss_image_edit"))
        for missing in ("flux2_vae.safetensors", os.path.join("moss_image", "flux2_small_decoder_bf16.safetensors")):
            with self.subTest(missing=missing):
                paths.remove(missing)
                self.assertFalse(namespace["_check_model_downloaded"]("moss_image_edit"))
                paths.add(missing)

    def test_handler_passes_variant_to_actual_validation(self):
        with patch("models.moss_image.pipeline.validate_generation_inputs", return_value="checked") as validate:
            md = {"moss_edit": True, "moss_edit_variant": "hq_v5"}
            inputs = {"image_refs": ["first.png", "second.png"]}
            self.assertEqual(family_handler.validate_generative_settings("moss_image", md, inputs), "checked")
            validate.assert_called_once_with(inputs, model_def=md)

    def test_hq_without_any_vision_companion_is_not_reported_downloaded(self):
        tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_check_model_downloaded")
        namespace = {"wgp": None, "os": os, "_model_weight_groups": lambda _: [["transformer"]],
                     "_variant_group_downloaded": lambda *a, **k: True}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<readiness>", "exec"), namespace)
        md = family_handler.query_model_def("moss_image", catalog("moss_image_edit_hq")["model"])
        paths = set(md["required_model_assets"])
        paths.add(md["required_model_asset_groups"][0][0])
        namespace["wgp"] = SimpleNamespace(
            get_model_def=lambda _: md, get_model_recursive_prop=lambda *a, **k: [],
            fl=SimpleNamespace(locate_file=lambda name, **k: name if name in paths else None))
        self.assertTrue(namespace["_check_model_downloaded"]("moss_image_edit_hq"))
        for missing in [name for name in paths if name.startswith("moss_image_hq_vision")]:
            with self.subTest(missing=missing):
                paths.remove(missing)
                self.assertFalse(namespace["_check_model_downloaded"]("moss_image_edit_hq"))
                paths.add(missing)


if __name__ == "__main__":
    unittest.main()
