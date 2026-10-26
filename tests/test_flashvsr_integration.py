"""Model-free contracts for versioned FlashVSR selections across Maestro."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.editor_projects import create_editor_project, normalize_editor_project
from services.media_info import processing_record
from services.media_processing import validate_methods


class FlashVSRSelections(unittest.TestCase):
    def test_each_version_survives_editor_normalization_and_media_validation(self):
        for prefix in ("flashvsr", "flashvsrplus"):
            for suffix, scale in (("2", 2), ("3", 3), ("4", 4), ("2pass2", 2), ("2pass4", 4)):
                method = prefix + suffix
                with self.subTest(method=method):
                    project = create_editor_project(name="Upscale")
                    project["export"]["spatial_upsampling"] = method
                    self.assertEqual(normalize_editor_project(project)["export"]["spatial_upsampling"], method)
                    validate_methods(spatial=method, check_runtime=False)
                    record = processing_record(spatial=method)
                    self.assertEqual(record["method"], method)
                    self.assertEqual(record["multiplier"], scale)
                    self.assertIn("Turbo" if prefix.endswith("plus") else "v1.1", record["method_label"])

    def test_unknown_version_is_not_silently_treated_as_plus(self):
        with self.assertRaisesRegex(ValueError, "Unknown spatial"):
            validate_methods(spatial="flashvsrplus99", check_runtime=False)


if __name__ == "__main__":
    unittest.main()
