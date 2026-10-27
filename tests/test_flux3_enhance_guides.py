"""CPU coverage for Flux 3 prompt-enhancement guide routing and scope."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
sys.path.insert(0, str(APP_ROOT))

from services.enhance_guides import get_enhance_guide


class Flux3EnhanceGuideTests(unittest.TestCase):
    def _guide(self, model_type, *, has_images=False, model_def=None):
        fake_wgp = SimpleNamespace(
            get_model_def=lambda _model_type: model_def or {},
        )
        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            return get_enhance_guide(model_type, "image", has_images=has_images)

    def test_flux3_klein_without_references_selects_text_to_image_guide(self):
        guide = self._guide("flux3_klein", has_images=False)

        self.assertIn("FLUX 3 KLEIN TEXT-TO-IMAGE GUIDE", guide)
        self.assertIn("complete image to", guide)
        self.assertNotIn("REFERENCE-IMAGE AND EDIT GUIDE", guide)

    def test_flux3_klein_with_references_selects_reference_edit_guide(self):
        guide = self._guide("flux3_klein", has_images=True)

        self.assertIn("FLUX 3 KLEIN REFERENCE-IMAGE AND EDIT GUIDE", guide)
        self.assertIn("New composition", guide)
        self.assertNotIn("TEXT-TO-IMAGE GUIDE", guide)

    def test_flux2_keeps_the_existing_generic_flux_guide(self):
        text = self._guide("flux2_klein_9b", has_images=False)
        edit = self._guide("flux2_klein_9b", has_images=True)

        self.assertIn("FLUX.2 KLEIN 9B TEXT-TO-IMAGE PROMPT GUIDE", text)
        self.assertIn("FLUX.2 KLEIN 9B IMAGE-EDIT PROMPT GUIDE", edit)

    def test_full_model_override_still_replaces_the_architecture_guide(self):
        override = "Standalone fine-tune guide."
        load_guide = Mock(return_value=override)
        with patch.dict(
            sys.modules,
            {"wgp": SimpleNamespace(get_model_def=lambda _model_type: {
                "enhance_guide": "custom_guide",
                "enhance_guide_text": "inline delta should not replace the override",
            }), "services.guide_loader": SimpleNamespace(load_guide=load_guide)},
        ):
            guide = get_enhance_guide("flux3_klein", "image", has_images=True)

        self.assertEqual(guide, override)
        load_guide.assert_called_once_with("prompt_enhancer", "custom_guide")

    def test_inline_model_delta_still_appends_to_the_flux3_base(self):
        delta = "Use the checkpoint's supplied trigger word."
        guide = self._guide(
            "flux3_klein",
            has_images=True,
            model_def={"enhance_guide_text": delta},
        )

        self.assertIn("FLUX 3 KLEIN REFERENCE-IMAGE AND EDIT GUIDE", guide)
        self.assertIn("MODEL-SPECIFIC PROMPTING NOTES", guide)
        self.assertTrue(guide.endswith(delta))

    def test_reference_guide_maps_all_slots_and_keeps_scope_supported(self):
        guide = " ".join(self._guide("flux3_klein", has_images=True).split()).casefold()
        expected_slots = [
            f"Image {index + 1} → <ref_image_{index}>".casefold()
            for index in range(10)
        ]

        for mapping in expected_slots:
            with self.subTest(mapping=mapping):
                self.assertIn(mapping, guide)
        for required in (
            "bounded edit",
            "new composition",
            "identity or style donor",
            "keep exact object and character counts.",
            "preserve quoted lettering exactly",
            "do not guess a person's",
            "do not force photorealism",
            "do not invent boxes or hard masks",
            "preserve its complete element list",
            "only the finished edit instruction or scene description in plain prose",
        ):
            with self.subTest(required=required):
                self.assertIn(required.casefold(), guide)

    def test_text_to_image_guide_preserves_user_intent_without_invention(self):
        guide = " ".join(self._guide("flux3_klein", has_images=False).split()).casefold()

        for required in (
            "named characters and brands",
            "exact quoted text",
            "never invent words",
            "keep nonhuman subjects'",
            "a still image can show a frozen action pose.",
            "do not impose photorealism",
            "do not truncate to a fixed word count.",
            "only the finished prompt in plain prose",
        ):
            with self.subTest(required=required):
                self.assertIn(required.casefold(), guide)


if __name__ == "__main__":
    unittest.main()
