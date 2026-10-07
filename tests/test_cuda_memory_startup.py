"""Allocator choice must be applied once, before any CUDA tensor is created."""
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
from shared import cuda_memory


class StartupMemoryTests(unittest.TestCase):
    def setUp(self):
        importlib.reload(cuda_memory)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = Path(temporary.name) / "wgp_config.json"
        self.torch = ModuleType("torch")
        self.torch.version = SimpleNamespace(hip=None)
        self.torch.cuda = SimpleNamespace(is_available=lambda: True)
        self.torch.zeros = Mock()
        self.allocator = SimpleNamespace(install=Mock())
        self.mmgp = ModuleType("mmgp")
        self.mmgp.allocator = self.allocator
        self.modules = patch.dict(sys.modules, {"torch": self.torch, "mmgp": self.mmgp})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {"WANGP_CUDA_STACK_BYTES": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(importlib.reload, cuda_memory)

    def test_cli_wins_over_saved_allocator(self):
        self.config.write_text(json.dumps({"vram_allocator": "vmm_spill"}))
        state = cuda_memory.apply_startup_settings(["launch.py", "--vram-allocator=vmm"], str(self.config))
        self.allocator.install.assert_called_once_with(spill=False)
        self.assertEqual(state["active"], "vmm")
        status = cuda_memory.allocator_status({"vram_allocator": "default"})
        self.assertEqual(status["vram_allocator_cli_override"], "vmm")
        self.assertFalse(status["vram_allocator_restart_required"])

    def test_saved_allocator_change_requires_restart(self):
        cuda_memory.apply_startup_settings([], str(self.config))
        status = cuda_memory.allocator_status({"vram_allocator": "vmm"})
        self.assertTrue(status["vram_allocator_restart_required"])
        self.assertIsNone(status["vram_allocator_cli_override"])

    def test_launch_and_classic_entrypoints_only_install_once(self):
        self.config.write_text(json.dumps({"vram_allocator": "vmm_spill"}))
        first = cuda_memory.apply_startup_settings([], str(self.config))
        second = cuda_memory.apply_startup_settings([], str(self.config))
        self.assertEqual(first, second)
        self.allocator.install.assert_called_once_with(spill=True)

    def test_missing_or_unsupported_native_library_falls_back(self):
        self.allocator.install.side_effect = OSError("unsupported library")
        state = cuda_memory.apply_startup_settings(["--vram-allocator", "vmm"], str(self.config))
        self.assertEqual(state["active"], "default")
        self.assertIn("unsupported library", state["fallback_reason"])
        self.assertFalse(cuda_memory.allocator_status({"vram_allocator": "vmm"})["vram_allocator_restart_required"])

    def test_cpu_and_hip_do_not_load_native_allocator_or_create_cuda_tensors(self):
        for hip, available in ((None, False), ("6.3", True)):
            with self.subTest(hip=hip, available=available):
                importlib.reload(cuda_memory)
                self.torch.version.hip = hip
                self.torch.cuda.is_available = lambda: available
                cuda_memory.apply_startup_settings(["--vram-allocator", "vmm"], str(self.config))
                self.allocator.install.assert_not_called()
                self.torch.zeros.assert_not_called()

    def test_allocator_is_installed_before_context_stack_adjustment(self):
        events = []
        self.allocator.install.side_effect = lambda **kwargs: events.append("install")
        self.torch.zeros.side_effect = lambda *args, **kwargs: events.append("tensor")
        driver = SimpleNamespace(cuCtxSetLimit=Mock(return_value=0))
        with patch.dict(os.environ, {"WANGP_CUDA_STACK_BYTES": "256"}), patch.object(cuda_memory.ctypes, "CDLL", return_value=driver), \
                patch.object(cuda_memory.ctypes, "WinDLL", return_value=driver, create=True):
            cuda_memory.apply_startup_settings(["--vram-allocator", "vmm"], str(self.config))
        self.assertEqual(events, ["install", "tensor"])

    def test_early_entrypoints_precede_hardware_detection(self):
        launch = (ROOT / "app/launch.py").read_text(encoding="utf-8")
        engine = (ROOT / "app/wgp.py").read_text(encoding="utf-8")
        self.assertLess(launch.index("apply_startup_settings(sys.argv"), launch.index("from services import safe_download"))
        self.assertLess(engine.index("apply_startup_settings(sys.argv"), engine.index("from services.optional_acceleration"))


if __name__ == "__main__":
    unittest.main()
