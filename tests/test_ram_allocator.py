"""CPU-only coverage for optional MMGP RAM allocator startup and settings."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.memory_settings import MODEL_LOAD_KEYS, apply_memory_defaults, memory_settings_snapshot, validate_memory_updates  # noqa: E402
from shared import cuda_memory  # noqa: E402


class RAMAllocatorSettingsTests(unittest.TestCase):
    def test_ram_allocator_defaults_is_opt_in_and_is_not_a_model_reload_option(self):
        config = apply_memory_defaults({})
        self.assertEqual(config["ram_allocator"], "default")
        self.assertEqual(memory_settings_snapshot({})["ram_allocator"], "default")
        self.assertNotIn("ram_allocator", MODEL_LOAD_KEYS)
        self.assertEqual(validate_memory_updates({"ram_allocator": "mmgp"}), {"ram_allocator": "mmgp"})
        with self.assertRaises(ValueError):
            validate_memory_updates({"ram_allocator": "unknown"})


class RAMAllocatorStartupTests(unittest.TestCase):
    def setUp(self):
        importlib.reload(cuda_memory)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = Path(temporary.name) / "wgp_config.json"
        self.torch = ModuleType("torch")
        self.torch.version = SimpleNamespace(hip=None)
        self.torch.cuda = SimpleNamespace(is_available=Mock(return_value=False))
        self.torch.zeros = Mock()
        self.ram = SimpleNamespace(active=False, install=Mock(), release=Mock())

        def install():
            self.ram.active = True

        self.ram.install.side_effect = install
        allocator_module = ModuleType("mmgp.allocator")
        allocator_module.ram = self.ram
        mmgp_module = ModuleType("mmgp")
        mmgp_module.allocator = allocator_module
        self.modules = patch.dict(sys.modules, {
            "torch": self.torch,
            "mmgp": mmgp_module,
            "mmgp.allocator": allocator_module,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {"WANGP_CUDA_STACK_BYTES": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(importlib.reload, cuda_memory)

    def test_saved_opt_in_runs_on_cpu_and_reports_active(self):
        self.config.write_text(json.dumps({"ram_allocator": "mmgp"}), encoding="utf-8")
        state = cuda_memory.apply_startup_settings([], str(self.config))
        self.assertEqual(state["active"], "default")  # VRAM policy remains independent.
        self.ram.install.assert_called_once_with()
        self.assertEqual(cuda_memory.allocator_status({"ram_allocator": "mmgp"})["ram_allocator_active"], "mmgp")
        self.assertIsNone(cuda_memory.allocator_status({"ram_allocator": "mmgp"})["ram_allocator_fallback_reason"])
        self.assertFalse(cuda_memory.allocator_status({"ram_allocator": "mmgp"})["ram_allocator_restart_required"])

    def test_cli_value_wins_and_saved_change_requires_restart(self):
        self.config.write_text(json.dumps({"ram_allocator": "default"}), encoding="utf-8")
        cuda_memory.apply_startup_settings(["launch.py", "--ram-allocator=mmgp"], str(self.config))
        status = cuda_memory.allocator_status({"ram_allocator": "default"})
        self.assertEqual(status["ram_allocator_cli_override"], "mmgp")
        self.assertFalse(status["ram_allocator_restart_required"])

        importlib.reload(cuda_memory)
        self.ram.active = False
        self.ram.install.reset_mock()
        self.ram.install.side_effect = lambda: setattr(self.ram, "active", True)
        cuda_memory.apply_startup_settings([], str(self.config))
        status = cuda_memory.allocator_status({"ram_allocator": "mmgp"})
        self.assertTrue(status["ram_allocator_restart_required"])
        self.assertIsNone(status["ram_allocator_cli_override"])

    def test_supported_initialization_failures_fall_back_with_reason(self):
        for failure in (RuntimeError("missing native library"), OSError("bad native image"),
                        ImportError("psutil unavailable"), AttributeError("missing native hook")):
            with self.subTest(failure=type(failure).__name__):
                importlib.reload(cuda_memory)
                self.ram.active = False
                self.ram.install.reset_mock(side_effect=True)
                self.ram.install.side_effect = failure
                self.config.write_text(json.dumps({"ram_allocator": "mmgp"}), encoding="utf-8")
                cuda_memory.apply_startup_settings([], str(self.config))
                status = cuda_memory.allocator_status({"ram_allocator": "mmgp"})
                self.assertEqual(status["ram_allocator_active"], "default")
                self.assertIn(str(failure), status["ram_allocator_fallback_reason"])

    def test_failed_install_that_left_native_allocator_active_is_not_misreported(self):
        def partial_install():
            self.ram.active = True
            raise AttributeError("VMM release callback unavailable")

        self.ram.install.side_effect = partial_install
        self.config.write_text(json.dumps({"ram_allocator": "mmgp"}), encoding="utf-8")
        cuda_memory.apply_startup_settings([], str(self.config))
        status = cuda_memory.allocator_status({"ram_allocator": "mmgp"})
        self.assertEqual(status["ram_allocator_active"], "mmgp")
        self.assertIn("callback unavailable", status["ram_allocator_fallback_reason"])

    def test_release_is_noop_when_inactive_and_calls_native_release_when_active(self):
        self.assertFalse(cuda_memory.release_ram_cache())
        self.ram.release.assert_not_called()
        self.config.write_text(json.dumps({"ram_allocator": "mmgp"}), encoding="utf-8")
        cuda_memory.apply_startup_settings([], str(self.config))
        self.assertTrue(cuda_memory.release_ram_cache())
        self.ram.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()