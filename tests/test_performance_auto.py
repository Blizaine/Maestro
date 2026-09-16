"""CPU-only coverage for automatic profiles and persisted recommendation updates."""
from __future__ import annotations

import ast
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services import perf_recommend as perf  # noqa: E402


def hardware(ram=28.0, vram=18.7):
    return {
        "cuda_available": True,
        "gpu_name": "RTX A4500",
        "gpu_vram_gb": vram,
        "ram_gb": ram,
        "ram_tier": "high" if ram >= 64 else "low" if ram >= 32 else "very_low",
        "vram_tier": "high" if vram >= 24 else "low" if vram >= 12 else "tight",
    }


def legacy_config():
    return {
        "video_profile": 5,
        "image_profile": 5,
        "audio_profile": 3,
        "transformer_quantization": "int8",
        "vae_config": 0,
        "vram_safety_coefficient": 0.8,
        "attention_mode": "auto",
        "compile": "",
        "save_path": "my-projects",
        "services": {"auto_performance": True, "auto_performance_applied": True},
    }


class TestAutomaticProfiles(unittest.TestCase):
    def test_partial_pinning_remains_recommended_from_32_gb_ram(self):
        for ram in (32, 48, 63.9):
            for vram in (12, 18.7, 23.9):
                with self.subTest(ram=ram, vram=vram):
                    rec = perf.recommend_settings(hardware(ram, vram))
                    self.assertEqual(rec["video_profile"], 4)
                    self.assertEqual(rec["image_profile"], 4)
                    self.assertEqual(rec["audio_profile"], 3)
                    self.assertEqual(rec["vram_safety_coefficient"], 0.8)
                    self.assertEqual(rec["transformer_quantization"], "int8")

    def test_reporter_and_smaller_ram_keep_unpinned_profile_pending_benchmark(self):
        for ram in (8, 16, 23.9, 24, 28, 31.9):
            for vram in (12, 18.7, 23.9):
                with self.subTest(ram=ram, vram=vram):
                    rec = perf.recommend_settings(hardware(ram, vram))
                    self.assertEqual(rec["video_profile"], 5)
                    self.assertEqual(rec["image_profile"], 5)

    def test_other_profile_tiers_are_unchanged(self):
        for ram, vram, expected in (
            (28, 8, 5), (32, 8, 4.5), (64, 8, 4),
            (64, 19, 2), (28, 24, 3.5), (32, 24, 3), (128, 24, 1),
        ):
            with self.subTest(ram=ram, vram=vram):
                self.assertEqual(perf.recommend_settings(hardware(ram, vram))["video_profile"], expected)

    def test_unknown_ram_keeps_conservative_tier(self):
        hw = hardware()
        del hw["ram_gb"]
        self.assertEqual(perf.recommend_settings(hw)["video_profile"], 5)


class TestAutomaticProfileMigration(unittest.TestCase):
    def test_legacy_boolean_migrates_once_and_preserves_unrelated_config(self):
        config = legacy_config()
        result = perf.apply_auto_performance(config, hardware())
        self.assertEqual(result["updated"], {})
        self.assertEqual(config["save_path"], "my-projects")
        self.assertEqual(config["services"]["auto_performance_revision"], perf.AUTO_PERFORMANCE_REVISION)
        after = copy.deepcopy(config)
        self.assertIsNone(perf.apply_auto_performance(config, hardware()))
        self.assertEqual(config, after)

    def test_auto_off_or_missing_is_not_migrated(self):
        for auto in (False, None):
            with self.subTest(auto=auto):
                config = legacy_config()
                if auto is None:
                    del config["services"]["auto_performance"]
                else:
                    config["services"]["auto_performance"] = auto
                before = copy.deepcopy(config)
                self.assertIsNone(perf.apply_auto_performance(config, hardware()))
                self.assertEqual(config, before)

    def test_legacy_manual_values_are_preserved_even_if_auto_flag_was_left_on(self):
        config = legacy_config()
        config.update(image_profile=3.5, attention_mode="sage2", vram_safety_coefficient=0.65)
        result = perf.apply_auto_performance(config, hardware())
        self.assertEqual(config["video_profile"], 5)
        self.assertEqual(config["image_profile"], 3.5)
        self.assertEqual(config["attention_mode"], "sage2")
        self.assertEqual(config["vram_safety_coefficient"], 0.65)
        self.assertCountEqual(result["preserved"], ["image_profile", "attention_mode", "vram_safety_coefficient"])

    def test_stored_defaults_distinguish_auto_values_from_custom_values(self):
        config = legacy_config()
        config["services"].update(
            auto_performance_revision=1,
            auto_performance_defaults={key: config[key] for key in perf.applied_keys()},
        )
        config["video_profile"] = 4.5
        perf.apply_auto_performance(config, hardware())
        self.assertEqual(config["video_profile"], 4.5)
        self.assertEqual(config["image_profile"], 5)

    def test_unpublished_profile4_trial_refreshes_only_still_automatic_values(self):
        config = legacy_config()
        config.update(video_profile=4, image_profile=4)
        config["services"].update(
            auto_performance_revision=2,
            auto_performance_defaults={key: config[key] for key in perf.applied_keys()},
        )
        config["image_profile"] = 4.5
        result = perf.apply_auto_performance(config, hardware())
        self.assertEqual(result["updated"], {"video_profile": 5})
        self.assertEqual(config["image_profile"], 4.5)
        self.assertEqual(config["services"]["auto_performance_revision"], perf.AUTO_PERFORMANCE_REVISION)

    def test_manually_selected_profile4_is_preserved(self):
        config = legacy_config()
        config["video_profile"] = 4
        perf.apply_auto_performance(config, hardware())
        self.assertEqual(config["video_profile"], 4)
        config["services"].update(auto_performance=False, auto_performance_revision=2)
        before = copy.deepcopy(config)
        self.assertIsNone(perf.apply_auto_performance(config, hardware()))
        self.assertEqual(config, before)

    def test_explicit_apply_overrides_manual_values_and_records_revision(self):
        config = legacy_config()
        config["services"]["auto_performance"] = False
        config["video_profile"] = 4.5
        result = perf.apply_auto_performance(config, hardware(), force=True)
        self.assertEqual(config["video_profile"], 5)
        self.assertTrue(config["services"]["auto_performance"])
        self.assertFalse(perf.auto_performance_needs_refresh(config))
        self.assertEqual(result["preserved"], [])

    def test_failed_detection_does_not_consume_pending_migration(self):
        config = legacy_config()
        before = copy.deepcopy(config)
        self.assertIsNone(perf.apply_auto_performance(config, {"cuda_available": False}))
        self.assertEqual(config, before)
        self.assertTrue(perf.auto_performance_needs_refresh(config))

    def test_fresh_cpu_fallback_can_retry_when_cuda_becomes_available(self):
        config = {}
        perf.apply_auto_performance(config, {"cuda_available": False}, force=True)
        self.assertEqual(config["video_profile"], 4.5)
        self.assertTrue(perf.auto_performance_needs_refresh(config))
        perf.apply_auto_performance(config, hardware())
        self.assertEqual(config["video_profile"], 5)
        self.assertFalse(perf.auto_performance_needs_refresh(config))

    def test_future_revision_is_not_downgraded_at_startup(self):
        config = legacy_config()
        config["services"]["auto_performance_revision"] = perf.AUTO_PERFORMANCE_REVISION + 1
        before = copy.deepcopy(config)
        self.assertIsNone(perf.apply_auto_performance(config, hardware()))
        self.assertEqual(config, before)


class TestAutomaticProfileEntryPoints(unittest.TestCase):
    """Execute the real config handlers without importing the GPU application."""

    @classmethod
    def setUpClass(cls):
        cls.launch_tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        cls.wgp_tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.wgp = SimpleNamespace(
            server_config=legacy_config(),
            server_config_filename=str(Path(temp.name) / "config.json"),
            args=SimpleNamespace(vram_safety_coefficient=0.8),
            attention_mode="auto", vae_config=0, compile="", transformer_quantization="int8",
        )
        self.globals = {
            "wgp": self.wgp, "json": json, "Request": object,
            "_services": self.wgp.server_config["services"],
            "_auto_perf_needs_refresh": perf.auto_performance_needs_refresh,
        }

    def _run_startup(self):
        node = next(
            node for node in self.launch_tree.body
            if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "_auto_perf_needs_refresh"
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "launch.py", "exec"), self.globals)

    def _endpoint(self, name):
        node = copy.deepcopy(next(
            node for node in self.launch_tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == name
        ))
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), "launch.py", "exec"), self.globals)
        return self.globals[name]

    def test_startup_migrates_on_disk_once(self):
        with patch("services.hardware_detect.detect_hardware", return_value=hardware()) as detect:
            self._run_startup()
            self._run_startup()
            detect.assert_called_once()
        saved = json.loads(Path(self.wgp.server_config_filename).read_text())
        self.assertEqual(saved["video_profile"], 5)
        self.assertEqual(saved["services"]["auto_performance_revision"], perf.AUTO_PERFORMANCE_REVISION)

    def test_startup_also_refreshes_changed_runtime_settings(self):
        self.wgp.server_config["services"]["auto_performance_applied"] = False
        with patch("services.hardware_detect.detect_hardware", return_value=hardware(16, 8)):
            self._run_startup()
        self.assertEqual(self.wgp.vae_config, 3)
        self.assertEqual(self.wgp.args.vram_safety_coefficient, 0.7)

    def test_fresh_config_records_revision_and_startup_does_not_reapply(self):
        node = next(
            node for node in ast.walk(self.wgp_tree)
            if isinstance(node, ast.Try) and any(
                isinstance(child, ast.ImportFrom) and child.module == "services.perf_recommend"
                and any(alias.name == "apply_auto_performance" for alias in child.names)
                for child in node.body
            )
        )
        self.wgp.server_config.clear()
        context = {"server_config": self.wgp.server_config}
        with patch("services.hardware_detect.detect_hardware", return_value=hardware()) as detect:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "wgp.py", "exec"), context)
            self._run_startup()
            detect.assert_called_once()
        self.assertEqual(self.wgp.server_config["video_profile"], 5)
        self.assertFalse(perf.auto_performance_needs_refresh(self.wgp.server_config))

    def test_explicit_apply_endpoint_stamps_revision_and_retains_response(self):
        self.wgp.server_config["video_profile"] = 4.5
        with patch("services.hardware_detect.detect_hardware", return_value=hardware()):
            result = asyncio.run(self._endpoint("apply_system_detect")())
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["profile_changed"])
        self.assertEqual(result["applied"]["video_profile"], 5)
        self.assertFalse(perf.auto_performance_needs_refresh(self.wgp.server_config))

    def test_manual_api_change_disables_auto_but_unrelated_or_unchanged_edits_do_not(self):
        for body, expected_auto in (
            ({"video_profile": 4.5}, False),
            ({"video_profile": 5}, True),
            ({"video_output_codec": "h264"}, True),
        ):
            with self.subTest(body=body):
                self.wgp.server_config = legacy_config()

                async def request_json():
                    return body

                request = SimpleNamespace(json=request_json)
                asyncio.run(self._endpoint("update_system_config")(request))
                self.assertEqual(self.wgp.server_config["services"]["auto_performance"], expected_auto)


if __name__ == "__main__":
    unittest.main()
