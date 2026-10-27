"""CPU-only tests for Director image prompt style and guide routing."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from services.director import image_prompt_rules
from services.director.guide_loader import load_guide


class DirectorImageStyleTests(unittest.TestCase):
    def test_reference_routes_load_native_guides_and_append_user_first_style_rules(self):
        cases = (
            ("flux2_klein_9b", "flux_image_edit_pass2.md", "FLUX image family"),
            ("qwen_image_edit_20B", "qwen_image_edit_reference.md", "Qwen Image"),
        )
        for model, filename, dialect in cases:
            with self.subTest(model=model), patch.object(
                image_prompt_rules, "load_guide", wraps=load_guide
            ) as loader:
                rules = image_prompt_rules.get_image_prompt_rules(
                    has_reference=True,
                    seamless=False,
                    image_model=model,
                )

            loader.assert_called_once_with(filename)
            self.assertIn(dialect, rules)
            self.assertIn("IMAGE MEDIUM AND STYLE", rules)
            self.assertIn("including anime, illustration, painting", rules)
            self.assertIn("Follow the user's requested medium and visual style", rules)
            self.assertIn(
                "An identity-only reference supplies identity, not the target medium or style",
                rules,
            )
            self.assertIn("Follow explicit user requests for medium, restyling, and lighting", rules)

    def test_no_base_reference_uses_generation_guide_without_inventing_source_constraints(self):
        with patch.object(
            image_prompt_rules, "load_guide", wraps=load_guide
        ) as loader:
            rules = image_prompt_rules.get_image_prompt_rules(
                has_reference=False,
                seamless=False,
                image_model="qwen_image_edit_20B",
            )

        loader.assert_called_once_with("flux_image_generation.md")
        self.assertIn("DIRECTOR IMAGE PROMPT", rules)
        self.assertIn("opening state", rules.casefold())
        self.assertIn("IMAGE MEDIUM AND STYLE", rules)
        self.assertNotIn("reference photo", rules.casefold())
        self.assertNotIn("attire from reference image", rules.casefold())
        self.assertNotIn("use lighting and color temp from reference image", rules.casefold())

    def test_optional_identity_references_are_available_without_a_base_image(self):
        rules = image_prompt_rules.get_image_prompt_rules(
            has_reference=False,
            num_character_refs=1,
            character_ref_labels=["red-haired person"],
            seamless=False,
            image_model="qwen_image_edit_20B",
        )

        self.assertIn("CHARACTER REFERENCE image(s)", rules)
        self.assertIn("red-haired person", rules)
        self.assertIn("alongside a main scene image when one is attached", rules)
        self.assertIn("An identity-only reference supplies identity", rules)
        self.assertNotIn("reference photo", rules.casefold())

    def test_reference_guides_keep_opening_state_without_global_motion_bans(self):
        flux = image_prompt_rules.get_image_prompt_rules(
            has_reference=True, seamless=False, image_model="flux2_klein_9b"
        )
        qwen = image_prompt_rules.get_image_prompt_rules(
            has_reference=True, seamless=False, image_model="qwen_image_edit_20B"
        )

        for rules in (flux, qwen):
            with self.subTest(header=rules.splitlines()[0]):
                self.assertIn("opening state", rules.casefold())
                self.assertTrue(
                    "before the planned video action advances" in rules.casefold()
                    or "before a planned action advances" in rules.casefold()
                )
                self.assertIn("Do not invent an extra pre-action pose", rules)
                self.assertIn("Do not globally", rules)
                self.assertNotIn("NO motion verbs", rules)
                self.assertNotIn("NO motion-photography effects", rules)
                self.assertNotIn("Use lighting and color temp from reference image.", rules)
                self.assertIn("A still", rules)

    def test_explicit_style_request_is_not_gated_on_reference_style(self):
        rules = image_prompt_rules.get_image_prompt_rules(
            has_reference=True, seamless=False, image_model="qwen_image_edit_20B"
        )

        self.assertIn("an explicit restyle takes priority", rules.casefold())
        self.assertIn("identity-only reference is not a style target", rules.casefold())
        self.assertNotIn(
            "If the reference is photorealistic, all prompts must be photorealistic",
            rules,
        )


if __name__ == "__main__":
    unittest.main()
