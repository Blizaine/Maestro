"""Image writer style contracts reach the provider without rewriting the brief."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from services import llm_service
from services.image_prompt_style import IMAGE_STYLE_GUIDANCE


MODELS = (
    "flux2_klein_9b", "flux3_klein", "flux3_klein_base",
    "qwen_image_21_7B", "qwen_image_21_7B_turbo",
    "krea2_raw", "krea2_turbo", "krea2_raw_edit", "krea2_turbo_edit",
    "moss_image", "moss_image_sft", "moss_image_edit", "moss_image_edit_hq",
    "fibo2_dev", "fibo2_edit_turbo",
)


def model_definition(model_type):
    path = APP / "defaults" / f"{model_type}.json"
    return json.loads(path.read_text(encoding="utf-8"))["model"] if path.is_file() else {}


class ImagePromptStyleTests(unittest.TestCase):
    def setUp(self):
        # Guide lookup needs registered model metadata, not the application's
        # command-line bootstrap or real model inventory.
        model_registry = patch.dict(sys.modules, {"wgp": SimpleNamespace(get_model_def=model_definition)})
        model_registry.start()
        self.addCleanup(model_registry.stop)

    def test_requested_styles_reach_guided_and_raw_writers_for_every_image_family(self):
        fake_wgp = SimpleNamespace(get_model_def=model_definition)
        for model_type in MODELS:
            for style in ("2D anime illustration", "watercolor painting", "photorealistic photograph"):
                source = f"A {style} of a fox leaping over a blue log. The sign reads \"FOX\"."
                paths = ["source.png"] if "edit" in model_type else None
                for raw in (False, True):
                    with (
                        self.subTest(model=model_type, style=style, raw=raw),
                        patch.dict(sys.modules, {"wgp": fake_wgp}),
                        patch.object(llm_service, "generate", return_value=source) as generate,
                    ):
                        result = llm_service.enhance_prompt(
                            source, mode="image", model_type=model_type,
                            image_paths=paths, raw_enhancer_mode=raw, resolution="1024x1024",
                        )
                    self.assertEqual(result, source)
                    payload = generate.call_args.kwargs
                    self.assertIn(source, payload["prompt"])
                    self.assertIn(IMAGE_STYLE_GUIDANCE, payload["prompt"])
                    self.assertEqual(payload.get("image_paths"), paths)
                    if payload["system_prompt"]:
                        self.assertIn(IMAGE_STYLE_GUIDANCE, payload["system_prompt"])
                        self.assertNotIn("STILL PHOTOGRAPH", payload["system_prompt"])

    def test_raw_image_writer_keeps_explicit_3d_style_framing(self):
        caption = "The scene, rendered in a 3D clay animation style, shows a fox leaping."
        with patch.object(llm_service, "generate", return_value=caption):
            result = llm_service.enhance_prompt(
                "A 3D clay fox leaping.", mode="image", model_type="krea2_raw",
                raw_enhancer_mode=True,
            )
        self.assertEqual(result, caption)
        # The unrelated legacy video cleanup keeps its prior behavior.
        self.assertEqual(llm_service._clean_enhancer_output(caption), "Shows a fox leaping.")

    def test_missing_guide_fallback_is_medium_neutral(self):
        with (
            patch("services.enhance_guides.get_enhance_guide", return_value=None),
            patch.object(llm_service, "generate", return_value="An anime fox.") as generate,
        ):
            llm_service.enhance_prompt("An anime fox.", mode="image", model_type="unknown_image")
        system = generate.call_args.kwargs["system_prompt"]
        self.assertIn("image prompt editor", system)
        self.assertIn(IMAGE_STYLE_GUIDANCE, system)
        self.assertNotIn("expert photographer", system)
        self.assertNotIn("STILL PHOTOGRAPH", system)

    def test_image_system_override_keeps_its_contract_and_style_context(self):
        override = "Refine only the supplied opening-frame caption."
        with patch.object(llm_service, "generate", return_value="A watercolor fox.") as generate:
            llm_service.enhance_prompt(
                "A watercolor fox.", mode="image", model_type="flux3_klein_base",
                system_override=override,
            )
        system = generate.call_args.kwargs["system_prompt"]
        self.assertTrue(system.startswith(override))
        self.assertIn(IMAGE_STYLE_GUIDANCE, system)

    def test_flux3_active_variant_context_distinguishes_base_and_distilled(self):
        fake_wgp = SimpleNamespace(get_model_def=lambda _: {})
        for model_type in ("flux3_klein", "flux3_klein_nvfp4", "flux3_klein_base", "flux3_klein_base_nvfp4"):
            with (
                self.subTest(model=model_type),
                patch.dict(sys.modules, {"wgp": fake_wgp}),
                patch.object(llm_service, "generate", return_value="An illustrated fox.") as generate,
            ):
                llm_service.enhance_prompt("An illustrated fox.", mode="image", model_type=model_type)
            system = generate.call_args.kwargs["system_prompt"]
            expected = (
                "ACTIVE FLUX 3 MODEL: Klein Base is undistilled"
                if "_base" in model_type else
                "ACTIVE FLUX 3 MODEL: Klein is the distilled checkpoint"
            )
            self.assertIn(expected, system)
            self.assertIn("Maestro owns sampling and guidance settings", system)


if __name__ == "__main__":
    unittest.main()
