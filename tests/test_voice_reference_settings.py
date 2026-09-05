"""Regression coverage for LTX-only Voice Reference settings."""

import os
import unittest


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAUNCH_PATH = os.path.join(_ROOT, "app", "launch.py")
_SERVICES_PANEL_PATH = os.path.join(
    _ROOT,
    "ui",
    "src",
    "components",
    "SettingsDrawer",
    "ServicesSettingsPanel.tsx",
)
_ADVANCED_PATH = os.path.join(
    _ROOT,
    "ui",
    "src",
    "components",
    "Sidebar",
    "AdvancedSettings.tsx",
)


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class TestVoiceReferenceSettings(unittest.TestCase):
    def test_voice_reference_and_multishot_default_off(self):
        launch = _read(_LAUNCH_PATH)
        self.assertIn(
            '"voice_reference_enabled": services.get('
            '"voice_reference_enabled", False)',
            launch,
        )
        self.assertIn(
            '"director_multishot_lora_mode": services.get('
            '"director_multishot_lora_mode", False)',
            launch,
        )
        self.assertIn(
            '"show_experimental": services.get("show_experimental", False)',
            launch,
        )

    def test_ltx_controls_moved_out_of_integrations(self):
        panel = _read(_SERVICES_PANEL_PATH)
        self.assertNotIn("Voice Reference (ID-LoRA)", panel)
        self.assertNotIn("Multi-Shot LoRA Mode", panel)

        advanced = _read(_ADVANCED_PATH)
        self.assertIn("function LtxFramesExperimentalControls()", advanced)
        self.assertIn("workflow !== 'frames'", advanced)
        self.assertIn("architecture.startsWith('ltx2')", advanced)
        self.assertIn("Voice Reference (ID-LoRA)", advanced)
        self.assertIn("Multi-Shot LoRA Prompting", advanced)
        self.assertIn("Off by default", advanced)


if __name__ == "__main__":
    unittest.main()
