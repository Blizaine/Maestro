"""CPU coverage for Krea 2 enhancer routing and native request wiring."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
sys.path.insert(0, str(APP_ROOT))

from services.enhance_guides import get_enhance_guide
from services import llm_service


_IMAGE_TITLE = "Krea 2 Image Prompt Guide"
_EDIT_TITLE = "Krea 2 Image Edit Guide"


class Krea2EnhanceGuideTests(unittest.TestCase):
    def _guide(self, model_type, *, has_images=False):
        fake_wgp = SimpleNamespace(get_model_def=lambda _model_type: {})
        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            return get_enhance_guide(model_type, "image", has_images=has_images)

    def test_raw_and_turbo_route_by_reference_presence(self):
        for model_type in ("krea2_raw", "krea2_turbo"):
            with self.subTest(model_type=model_type, has_images=False):
                self.assertIn(_IMAGE_TITLE, self._guide(model_type))
            with self.subTest(model_type=model_type, has_images=True):
                self.assertIn(_EDIT_TITLE, self._guide(model_type, has_images=True))

    def test_dedicated_edit_ids_keep_edit_contract_with_or_without_images(self):
        for model_type in ("krea2_raw_edit", "krea2_turbo_edit"):
            for has_images in (False, True):
                with self.subTest(model_type=model_type, has_images=has_images):
                    self.assertIn(
                        _EDIT_TITLE,
                        self._guide(model_type, has_images=has_images),
                    )

    def test_caption_and_edit_guides_keep_style_and_reference_constraints(self):
        caption = " ".join(self._guide("krea2_raw").split())
        edit = " ".join(self._guide("krea2_raw_edit").split())

        self.assertIn("leave the look open", caption)
        self.assertIn("do not turn it into a sequence", caption)
        self.assertIn("Do not require a lens", caption)
        self.assertIn("Follow reference roles or ordering when the user specifies them", edit)
        self.assertIn("Do not invent image tokens, numbered slots", edit)
        self.assertIn("Do not invent masks, coordinates", edit)
        self.assertIn("If no image is attached, do not claim to see one", edit)

    def test_actual_edit_request_uses_guide_without_requiring_an_attached_image(self):
        fake_wgp = SimpleNamespace(get_model_def=lambda _model_type: {})
        with (
            patch.dict(sys.modules, {"wgp": fake_wgp}),
            patch.object(llm_service, "generate", return_value="Change the scarf to blue.") as generate,
        ):
            result = llm_service.enhance_prompt(
                "Change the scarf to blue.",
                mode="image",
                model_type="krea2_turbo_edit",
            )

        self.assertEqual(result, "Change the scarf to blue.")
        self.assertEqual(generate.call_count, 1)
        request = generate.call_args.kwargs
        self.assertIn(_EDIT_TITLE, request["system_prompt"])
        self.assertIn("keep the user's edit intent", request["system_prompt"].lower())
        self.assertIsNone(request["image_paths"])
        self.assertIn("Change the scarf to blue.", request["prompt"])


if __name__ == "__main__":
    unittest.main()