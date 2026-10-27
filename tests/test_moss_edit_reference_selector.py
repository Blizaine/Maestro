"""Regression coverage for legacy Studio selectors at the MOSS edit boundary."""
from pathlib import Path
import sys
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.moss_image.moss_image_handler import family_handler  # noqa: E402


class MossEditReferenceSelectorTests(unittest.TestCase):
    def _edit_definition(self, model_type):
        return {
            "moss_edit": True,
            "moss_edit_variant": "hq_v5" if model_type.endswith("_hq") else "sft",
        }

    def _valid_request(self, video_prompt_type="KI"):
        return {
            "image_mode": 1,
            "width": 720,
            "height": 1280,
            "video_prompt_type": video_prompt_type,
            "image_refs": [
                Image.new("RGB", (720, 1280), "red"),
                Image.new("RGB", (720, 1280), "blue"),
            ],
        }

    def test_fibo_ki_alias_is_canonicalized_for_both_edit_adapters(self):
        for model_type in ("moss_image_edit", "moss_image_edit_hq"):
            with self.subTest(model_type=model_type):
                inputs = self._valid_request()
                error = family_handler.validate_generative_settings(
                    "moss_image", self._edit_definition(model_type), inputs,
                )
                self.assertIsNone(error)
                self.assertEqual(inputs["video_prompt_type"], "I")

    def test_missing_or_malformed_references_do_not_activate_the_alias(self):
        edit = self._edit_definition("moss_image_edit")

        missing = self._valid_request()
        missing["image_refs"] = []
        self.assertEqual(
            family_handler.validate_generative_settings("moss_image", edit, missing),
            "MOSS Image Edit requires video_prompt_type I.",
        )
        self.assertEqual(missing["video_prompt_type"], "KI")

        malformed = self._valid_request()
        malformed["image_refs"] = ["source.png"]
        self.assertEqual(
            family_handler.validate_generative_settings("moss_image", edit, malformed),
            "MOSS Image Edit requires video_prompt_type I.",
        )
        self.assertEqual(malformed["video_prompt_type"], "KI")

        explicit_i = self._valid_request("I")
        explicit_i["image_refs"] = []
        self.assertEqual(
            family_handler.validate_generative_settings("moss_image", edit, explicit_i),
            "MOSS Image Edit requires one to four reference images.",
        )

    def test_unknown_selector_flags_remain_invalid(self):
        edit = self._edit_definition("moss_image_edit")
        for selector in ("KIV", "KIA", "IK"):
            with self.subTest(selector=selector):
                inputs = self._valid_request(selector)
                self.assertEqual(
                    family_handler.validate_generative_settings("moss_image", edit, inputs),
                    "MOSS Image Edit requires video_prompt_type I.",
                )
                self.assertEqual(inputs["video_prompt_type"], selector)

    def test_alias_does_not_bypass_unsupported_inputs_or_image_mode(self):
        edit = self._edit_definition("moss_image_edit")
        for unsupported in ("image_mask", "video_guide", "audio_guide"):
            with self.subTest(unsupported=unsupported):
                inputs = self._valid_request()
                inputs[unsupported] = object()
                self.assertEqual(
                    family_handler.validate_generative_settings("moss_image", edit, inputs),
                    "MOSS Image Edit accepts ordered image references only; masks, video, and audio inputs are unsupported.",
                )
                self.assertEqual(inputs["video_prompt_type"], "I")

        masked_mode = self._valid_request()
        masked_mode["image_mode"] = 2
        self.assertEqual(
            family_handler.validate_generative_settings("moss_image", edit, masked_mode),
            "MOSS Image Edit supports image generation mode 1 only.",
        )
        self.assertEqual(masked_mode["video_prompt_type"], "KI")

    def test_t2i_models_do_not_accept_the_edit_alias(self):
        inputs = self._valid_request()
        error = family_handler.validate_generative_settings("moss_image", {}, inputs)
        self.assertEqual(
            error,
            "MOSS Image is text-to-image only and does not accept reference images. Clear the reference-image selection.",
        )
        self.assertEqual(inputs["video_prompt_type"], "KI")


if __name__ == "__main__":
    unittest.main()
