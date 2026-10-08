"""Exercise persisted memory controls and the real API handler without a GPU app."""
import ast
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.memory_settings import (apply_memory_defaults, memory_settings_snapshot,
                                     dynamic_preload_supported, preload_for_output, reserved_ram_fraction, validate_memory_updates)


class MemorySettingsTests(unittest.TestCase):
    def test_dynamic_measurements_invalidate_when_definition_refreshes_in_place(self):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "auto_preload_store")
        definition = {"architecture": "minimax_h3", "URLs": ["original"]}
        namespace = {"json": json, "auto_preload_measures": {}, "get_model_def": lambda _: definition}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "wgp.py", "exec"), namespace)
        get_store = namespace["auto_preload_store"]
        initial = get_store("imported_h3")
        initial["peak"] = 42
        self.assertIs(get_store("imported_h3"), initial)
        definition["URLs"] = ["replacement"]
        fresh = get_store("imported_h3")
        self.assertIsNot(fresh, initial)
        self.assertEqual(fresh, {})

    def test_legacy_manual_preload_migrates_without_overwriting_per_output_preferences(self):
        config = {"preload_in_VRAM": 6000, "image_preload_mode": "dynamic", "image_preload_in_VRAM": 3200,
                  "smart_memory_pinning": False, "vram_allocator": "vmm"}
        apply_memory_defaults(config)
        self.assertEqual(config["video_preload_mode"], "manual")
        self.assertEqual(config["audio_preload_in_VRAM"], 6000)
        self.assertEqual(config["image_preload_mode"], "dynamic")
        self.assertEqual(config["image_preload_in_VRAM"], 3200)
        self.assertFalse(config["smart_memory_pinning"])
        before = copy.deepcopy(config)
        apply_memory_defaults(config)
        self.assertEqual(config, before)

    def test_cli_preload_wins_and_output_modes_are_independent(self):
        config = apply_memory_defaults({"preload_in_VRAM": 6000, "image_preload_mode": "dynamic"})
        self.assertEqual(preload_for_output(config, SimpleNamespace(preload=0), "image"), ("dynamic", 0))
        self.assertEqual(preload_for_output(config, SimpleNamespace(preload=0), "audio"), ("manual", 6000))
        self.assertEqual(preload_for_output(config, SimpleNamespace(preload="2400"), "image"), ("manual", 2400))

    def test_bad_legacy_preload_uses_profile_default_and_snapshot_does_not_mutate(self):
        config = {"preload_in_VRAM": "invalid"}
        result = memory_settings_snapshot(config)
        self.assertEqual(result["video_preload_mode"], "default")
        self.assertEqual(result["vram_allocator"], "default")
        self.assertEqual(config, {"preload_in_VRAM": "invalid"})

    def test_reserved_ram_uses_fraction_for_cli_and_percent_for_settings(self):
        self.assertEqual(reserved_ram_fraction({"perc_reserved_mem_max": 50}, SimpleNamespace()), .5)
        self.assertEqual(reserved_ram_fraction({"perc_reserved_mem_max": 50}, SimpleNamespace(perc_reserved_mem_max=.4)), .4)
        self.assertEqual(reserved_ram_fraction({}, SimpleNamespace()), 0)

    def test_int8_backend_preserves_legacy_disabled_and_triton_choices(self):
        self.assertEqual(apply_memory_defaults({"enable_int8_kernels": 0})["int8_kernels"], "disabled")
        self.assertEqual(apply_memory_defaults({"enable_int8_kernels": 1})["int8_kernels"], "triton")
        self.assertEqual(apply_memory_defaults({"int8_kernels": "kitchen"})["int8_kernels"], "kitchen")

    def test_dynamic_preload_only_applies_to_async_budgeted_profiles(self):
        for profile in (2, 3, 3.5, 4, 5):
            self.assertTrue(dynamic_preload_supported(profile))
        for profile in (1, 4.5):
            self.assertFalse(dynamic_preload_supported(profile))

    def test_invalid_fields_reject_instead_of_coercing(self):
        for body in ({"read_ahead": "false"}, {"smart_memory_pinning": 1}, {"attention_head_split": True},
                     {"attention_head_split": 4}, {"perc_reserved_mem_max": 81}, {"video_preload_in_VRAM": -1},
                     {"video_preload_mode": "unlimited"}, {"vram_allocator": "other"}, {"int8_kernels": 1}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                validate_memory_updates(body)


class HTTPError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


class MemorySettingsApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config_path = Path(temporary.name) / "config.json"
        self.config = apply_memory_defaults({"attention_mode": "auto", "services": {"auto_performance": True}})
        self.wgp = SimpleNamespace(server_config=self.config, server_config_filename=str(self.config_path),
                                   args=SimpleNamespace(vram_safety_coefficient=.8), reload_needed=False)
        tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        node = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                                  and node.name == "update_system_config"))
        node.decorator_list = []
        self.namespace = {"wgp": self.wgp, "json": json, "Request": object, "HTTPException": HTTPError,
                          "PREVIEW_MODES": ("off", "rgb", "tiny_vae", "tiny_vae_video")}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "launch.py", "exec"), self.namespace)

    def save(self, body):
        async def request_json():
            return body
        return asyncio.run(self.namespace["update_system_config"](SimpleNamespace(json=request_json)))

    def test_invalid_memory_field_leaves_all_settings_and_disk_untouched(self):
        before = copy.deepcopy(self.config)
        with self.assertRaises(HTTPError):
            self.save({"attention_mode": "sdpa", "video_preload_mode": "bad"})
        self.assertEqual(self.config, before)
        self.assertFalse(self.config_path.exists())
        self.assertFalse(self.wgp.reload_needed)

    def test_valid_pinning_update_persists_and_reloads_next_generation(self):
        result = self.save({"read_ahead": True, "image_preload_mode": "dynamic", "perc_reserved_mem_max": 45})
        self.assertEqual(result["updated"]["perc_reserved_mem_max"], 45)
        self.assertEqual(json.loads(self.config_path.read_text())["image_preload_mode"], "dynamic")
        self.assertTrue(self.wgp.reload_needed)

    def test_allocator_change_requires_restart_without_reprofiling_live_models(self):
        result = self.save({"vram_allocator": "vmm_spill"})
        self.assertTrue(result["vram_allocator_restart_required"])
        self.assertEqual(result["vram_allocator_active"], "default")
        self.assertFalse(self.wgp.reload_needed)

    def test_ram_allocator_change_requires_restart_without_live_model_reload(self):
        result = self.save({"ram_allocator": "mmgp"})
        self.assertTrue(result["ram_allocator_restart_required"])
        self.assertEqual(result["ram_allocator_active"], "default")
        self.assertFalse(self.wgp.reload_needed)

    def test_head_split_does_not_reprofile_models(self):
        result = self.save({"attention_head_split": 2})
        self.assertEqual(result["updated"], {"attention_head_split": 2})
        self.assertFalse(self.wgp.reload_needed)


if __name__ == "__main__":
    unittest.main()
