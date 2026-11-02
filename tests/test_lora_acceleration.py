"""Acceleration labels are cheap technical hints, independent of display prose."""
import ast
from copy import deepcopy
import glob
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.lora_acceleration import is_acceleration_lora


class LoraAccelerationTests(unittest.TestCase):
    def test_technical_names_cover_h3_and_other_model_families(self):
        for name in (
            "minimax_h3_turbo_v4_step600.safetensors", "minimax_h3_dmad_full.safetensors",
            "minimax_h3_fl2v_turbo_4step_v1.0.safetensors", "h3_ref2va_pdd.sft",
            "Wan2.2-Lightning-I2V-4steps.safetensors", "sdxl_lcm_lora.safetensors",
            "Hyper-SD.safetensors", "Qwen-Image-2.1-viggle-turbo-4step.safetensors",
            "model_8step.safetensors", "model-16-steps.safetensors",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_acceleration_lora(name))

    def test_unknown_styles_training_step_numbers_and_parent_folders_do_not_match(self):
        for name in ("portrait.safetensors", "lighting.safetensors", "turbocharged_car.safetensors",
                     "artist_step600.safetensors", "C:/turbo/portrait.safetensors", "model_40steps.safetensors"):
            with self.subTest(name=name):
                self.assertFalse(is_acceleration_lora(name))
        self.assertFalse(is_acceleration_lora("portrait.safetensors", {"description": "Works with Turbo and Lightning", "name": "Turbo lighting"}))

    def test_explicit_metadata_is_authoritative_but_truthy_strings_are_not(self):
        self.assertTrue(is_acceleration_lora("renamed.safetensors", {"is_acceleration": True}))
        self.assertFalse(is_acceleration_lora("turbo.safetensors", {"is_acceleration": False}))
        self.assertFalse(is_acceleration_lora("renamed.safetensors", {"is_acceleration": "true"}))

    def test_real_details_route_labels_installed_and_virtual_managed_choices(self):
        from services import lora_metadata
        with tempfile.TemporaryDirectory() as directory:
            for name in ("portrait.safetensors", "sdxl_lcm.safetensors", "renamed.safetensors"):
                Path(directory, name).touch()
            Path(directory, "renamed.civitai.json").write_text(json.dumps({"is_acceleration": True}), encoding="utf-8")
            tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
            node = deepcopy(next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "list_loras_details"))
            node.decorator_list = []
            namespace = {
                "os": os, "glob": glob, "json": json,
                "wgp": SimpleNamespace(get_model_def=lambda _model: {}, get_lora_dir=lambda _model: directory,
                                       get_lora_search_dirs=lambda _model: [directory]),
                "_minimax_h3_turbo_option": lambda _model: None,
                "_managed_pika_flex_lora_names": lambda _model: set(),
                "_lora_is_compatible_with_model": lambda *_args: True,
                "_load_lora_manifest": lambda: {}, "_build_lora_max_version_map": lambda _dir: {},
                "_resolve_lora_root": lambda: directory, "_compute_lora_id": lambda name, _meta: "local:" + name,
                "_resolve_per_file_update_status": lambda **_kwargs: {}, "_classify_lora_nsfw": lambda **_kwargs: False,
                "is_acceleration_lora": is_acceleration_lora,
            }
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "launch.py", "exec"), namespace)
            namespace["_is_managed_moss_edit_lora"] = lambda _model, _path: False
            with patch.object(lora_metadata, "load_lora_display_names", return_value={}):
                rows = namespace["list_loras_details"]("test")["loras"]
            labels = {row["filename"]: row["is_acceleration"] for row in rows}
            self.assertEqual(labels, {"portrait.safetensors": False, "sdxl_lcm.safetensors": True, "renamed.safetensors": True})


if __name__ == "__main__":
    unittest.main()
