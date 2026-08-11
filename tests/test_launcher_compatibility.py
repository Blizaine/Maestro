"""Pinokio launcher regressions that do not require the application runtime."""
from __future__ import annotations

from pathlib import Path
import unittest


_ROOT = Path(__file__).resolve().parents[1]


class TestPinokioGpuCompatibility(unittest.TestCase):
    def test_installed_app_menu_is_not_hidden_by_early_gpu_detection(self):
        launcher = (_ROOT / "pinokio.js").read_text(encoding="utf-8")

        self.assertNotIn("if (kernel.gpu", launcher)
        self.assertIn('text: "Start"', launcher)
        self.assertIn('href: "start.js"', launcher)

    def test_fresh_install_still_uses_pinokios_documented_gpu_variable(self):
        installer = (_ROOT / "install.js").read_text(encoding="utf-8")

        self.assertIn("{{gpu !== 'nvidia'}}", installer)
        self.assertIn("This app requires an NVIDIA GPU", installer)

    def test_start_url_uses_the_required_capture_object(self):
        start = (_ROOT / "start.js").read_text(encoding="utf-8")

        self.assertIn('"event": "/(http:\\/\\/[0-9.:]+)/"', start)
        self.assertIn('url: "{{input.event[1]}}"', start)

    def test_rtx50_uses_an_isolated_cuda13_runtime(self):
        profile = (_ROOT / "launcher_profile.js").read_text(encoding="utf-8")
        installer = (_ROOT / "install.js").read_text(encoding="utf-8")
        torch_script = (_ROOT / "torch.js").read_text(encoding="utf-8")

        self.assertIn('target === "sm_120"', profile)
        self.assertIn('env: "env-rtx50"', profile)
        self.assertIn('python: "3.11"', profile)
        self.assertIn("venv_python: runtime.python", installer)
        self.assertIn("torch==2.10.0", torch_script)
        self.assertIn("/whl/cu130", torch_script)
        self.assertIn("lightx2v_kernel-0.0.2+torch2.10.0", torch_script)
        menu = (_ROOT / "pinokio.js").read_text(encoding="utf-8")
        self.assertIn("Repair RTX 50 Runtime", menu)

    def test_legacy_windows_flash_wheel_matches_wangp_documented_abi(self):
        torch_script = (_ROOT / "torch.js").read_text(encoding="utf-8")

        self.assertIn("flash_attn-2.7.4.post1+cu128torch2.7.0", torch_script)
        self.assertNotIn("flash_attn-2.8.2%2Bcu128torch2.7", torch_script)
        self.assertIn("--force-reinstall --no-deps", torch_script)

    def test_update_can_resume_a_missing_hardware_runtime(self):
        profile = (_ROOT / "launcher_profile.js").read_text(encoding="utf-8")
        updater = (_ROOT / "update.js").read_text(encoding="utf-8")

        self.assertIn("alreadyCurrentAndReady", updater)
        self.assertIn("exists('${runtime.marker}')", updater)
        self.assertIn("flashMarker", profile)
        self.assertIn("exists('${runtime.flashMarker}')", updater)
        self.assertIn("flash_only: true", updater)
        self.assertIn("venv: runtime.env", updater)

    def test_runtime_diagnostics_run_inside_the_loaded_engine(self):
        engine = (_ROOT / "app" / "wgp.py").read_text(encoding="utf-8")

        self.assertIn("from scripts.runtime_preflight import main", engine)
        self.assertIn("_runtime_preflight()", engine)

    def test_broken_optional_flash_is_guarded_before_mmgp_import(self):
        engine = (_ROOT / "app" / "wgp.py").read_text(encoding="utf-8")

        guard = engine.index("prepare_optional_flash_attention()")
        mmgp_import = engine.index("from mmgp import")
        self.assertLess(guard, mmgp_import)


if __name__ == "__main__":
    unittest.main()
