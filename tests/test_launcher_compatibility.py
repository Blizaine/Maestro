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


if __name__ == "__main__":
    unittest.main()
