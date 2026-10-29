"""Studio and enhancer routing for the MOSS reference-edit adapters."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import llm_service
from services.enhance_guides import get_enhance_guide, moss_image_enhancement_kind
from services.studio_enhancement import enhancement_request


class MossEditStudioTests(unittest.TestCase):
    def test_edit_contract_and_guide_are_separate_from_legacy_layout_models(self):
        model_defs = {
            "moss_image_edit": {"architecture": "moss_image", "moss_edit": True},
            "moss_image_edit_hq": {"architecture": "moss_image", "moss_edit": True},
            "community_moss_edit": {"architecture": "moss_image", "moss_edit": True},
            "moss_image_sft": {"architecture": "moss_image"},
            "moss_image": {"architecture": "moss_image"},
            "community_moss_t2i": {"architecture": "moss_image"},
        }
        fake_wgp = SimpleNamespace(get_model_def=lambda key: model_defs.get(key))
        with patch.dict(sys.modules, {"wgp": fake_wgp}):
            for model_type in ("moss_image_edit", "moss_image_edit_hq", "community_moss_edit"):
                with self.subTest(model_type=model_type):
                    self.assertEqual(moss_image_enhancement_kind(model_type), "edit")
                    guide = get_enhance_guide(model_type, "image", has_images=True)
                    self.assertIn("upload order", guide)
                    self.assertIn("Make only the changes the user requested", guide)
                    self.assertIn("When the user requests a composite", guide)
                    self.assertIn("never invent an unrequested", guide)
                    self.assertNotIn("bbox_coord_space", guide)
            for model_type in ("moss_image", "moss_image_sft", "community_moss_t2i"):
                with self.subTest(model_type=model_type):
                    self.assertEqual(moss_image_enhancement_kind(model_type), "layout")
                    guide = get_enhance_guide(model_type, "image", has_images=False)
                    self.assertIn("do not accept reference images", guide)

    def test_queued_enhancement_preserves_reference_order_for_edit_models(self):
        params = {
            "model_type": "moss_image_edit_hq",
            "generation_mode": "image",
            "image_mode": 1,
            "prompt": "Change only the embroidered mark.",
            "resolution": "1024x1024",
            "image_refs": ["first.png", "second.png"],
        }
        payload, sequence = enhancement_request(
            params,
            {"architecture": "moss_image", "moss_edit": True, "image_outputs": True},
        )
        self.assertFalse(sequence)
        self.assertEqual(payload["image_paths"], ["first.png", "second.png"])
        self.assertIn("image_refs order", payload["reference_context"])
        self.assertIn("roles", payload["reference_context"])

    def test_edit_enhancement_uses_plain_natural_language_and_skips_layout_compiler(self):
        model_defs = {
            "moss_image_edit": {"architecture": "moss_image", "moss_edit": True},
        }
        fake_wgp = SimpleNamespace(get_model_def=lambda key: model_defs.get(key))
        edited_prompt = "Remove only the small red label from the jar."
        with patch.dict(sys.modules, {"wgp": fake_wgp}), patch.object(
            llm_service, "generate", return_value=edited_prompt,
        ) as generate:
            result = llm_service.enhance_prompt(
                "Remove only the small red label.",
                mode="image",
                model_type="moss_image_edit",
                image_paths=["product.png", "logo.png"],
                reference_context=(
                    "Reference 1 is the product image. Reference 2 is the replacement logo."
                ),
                resolution="1024x1024",
            )
        self.assertEqual(result, edited_prompt)
        arguments = generate.call_args.kwargs
        self.assertEqual(arguments["image_paths"], ["product.png", "logo.png"])
        self.assertIn("MOSS IMAGE EDIT OUTPUT CONTRACT", arguments["system_prompt"])
        self.assertIn("Reference 1 is the product image", arguments["prompt"])
        self.assertIn("when a composite is requested", arguments["system_prompt"])
        self.assertNotIn("MOSS IMAGE OUTPUT CONTRACT", arguments["system_prompt"])
        self.assertNotIn("bbox_coord_space", arguments["system_prompt"])

    def test_native_edit_enhancer_keeps_ordered_images_without_layout_finalizer(self):
        tree = ast.parse((ROOT / "app" / "wgp.py").read_text(encoding="utf-8"))
        function = next(
            item for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "process_prompt_enhancer"
        )
        generated = unittest.mock.Mock(return_value=["Change only the jacket to dark blue."])
        enhancer_utils = ModuleType("shared.prompt_enhancer.prompt_enhance_utils")
        enhancer_utils.generate_cinematic_prompt = generated
        namespace = {
            "re": __import__("re"),
            "Image": SimpleNamespace(open=lambda path: path),
            "enhancer_offloadobj": None,
            "prompt_enhancer_image_caption_model": None,
            "prompt_enhancer_image_caption_processor": None,
            "prompt_enhancer_llm_model": None,
            "prompt_enhancer_llm_tokenizer": None,
            "server_config": {"prompt_enhancer_randomize_seed": False},
        }
        with patch.dict(sys.modules, {
            "shared.prompt_enhancer.prompt_enhance_utils": enhancer_utils,
        }):
            exec(compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                "<native MOSS edit enhancer>", "exec",
            ), namespace)
            first, second = object(), object()
            result = namespace["process_prompt_enhancer"](
                {
                    "architecture": "moss_image", "moss_edit": True,
                    "image_prompt_enhancer_instructions": "Edit the ordered references.",
                    "image_prompt_enhancer_max_tokens": 768,
                    "max_image_refs": 4,
                },
                "TI", ["Change only the jacket to dark blue."], None,
                [first, second], True, False, 123,
            )
            self.assertEqual(result, ["Change only the jacket to dark blue."])
            self.assertEqual(generated.call_args.args[5], [first, second])
            self.assertEqual(generated.call_args.kwargs["prompt_enhancer_instructions"], "Edit the ordered references.")
            self.assertFalse(generated.call_args.kwargs["literal_prompt"])

            generated.reset_mock(return_value=True)
            namespace["process_prompt_enhancer"](
                {"architecture": "qwen_image_21_7B"}, "TI", ["Edit."], None,
                [first, second], True, False, 123,
            )
            self.assertEqual(generated.call_args.args[5], [first])

    def test_text_to_image_model_can_bootstrap_first_reference_in_generate_workflow(self):
        source = (ROOT / "ui" / "src" / "components" / "Sidebar" / "ImageRefSection.tsx").read_text(encoding="utf-8")
        self.assertIn("configuredMaxRefs === 0", source)
        self.assertIn("imageRefs.length <= 1", source)
        self.assertIn("? 1", source)
        store = (ROOT / "ui" / "src" / "stores" / "useStore.ts").read_text(encoding="utf-8")
        self.assertIn("if (hasReferenceImages) return model.supports_image_edit === true", store)

    def test_required_reference_flag_reaches_model_catalog_consumers(self):
        tree = ast.parse((ROOT / "app" / "launch.py").read_text(encoding="utf-8"))
        helper = next(
            item for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "_requires_image_reference"
        )
        list_models = next(
            item for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "list_models"
        )
        list_models.decorator_list = []
        module = ast.Module(body=[helper, list_models], type_ignores=[])
        ast.fix_missing_locations(module)
        definitions = {
            "moss_image": {"architecture": "moss_image"},
            "moss_image_sft": {"architecture": "moss_image"},
            "moss_image_edit": {
                "architecture": "moss_image", "moss_edit": True,
                "image_ref_choices": {"choices": [("Ordered refs", "I")]},
                "at_least_one_image_ref_needed": True,
            },
            "moss_image_edit_hq": {
                "architecture": "moss_image", "moss_edit": True,
                "image_ref_choices": {"choices": [("Ordered refs", "I")]},
                "at_least_one_image_ref_needed": True,
            },
        }
        runtime = SimpleNamespace(
            families_infos={"moss_image": (100, "MOSS Image")},
            displayed_model_types=list(definitions),
            get_model_def=definitions.get,
            get_model_family=lambda _model, **_kwargs: "moss_image",
            get_base_model_type=lambda _model: "moss_image",
            model_types_handlers={"moss_image": object()},
            test_class_i2v=lambda _model: False,
            test_class_t2v=lambda _model: False,
        )
        director = ModuleType("services.director_model_compat")
        director.assess_director_model = lambda *args, **kwargs: {}
        preview = ModuleType("shared.preview_runtime")
        preview.model_preview_support = lambda *args, **kwargs: {}
        namespace = {"os": os, "wgp": runtime, "_check_model_downloaded": lambda _model: True}
        with patch.dict(sys.modules, {
            "services.director_model_compat": director,
            "shared.preview_runtime": preview,
        }):
            exec(compile(module, "<launch model catalog>", "exec"), namespace)
            models = {
                model["model_type"]: model
                for model in namespace["list_models"]()["models"]
            }
        requires = namespace["_requires_image_reference"]
        self.assertTrue(requires({"at_least_one_image_ref_needed": True}))
        self.assertTrue(requires({"one_image_ref_needed": True}))
        self.assertFalse(requires({"at_least_one_image_ref_needed": False}))
        for model_type in ("moss_image_edit", "moss_image_edit_hq"):
            with self.subTest(model_type=model_type):
                self.assertTrue(models[model_type]["requires_image_reference"])
                self.assertTrue(models[model_type]["supports_image_edit"])
                self.assertTrue(models[model_type]["supports_instruction_image_edit"])
        for model_type in ("moss_image", "moss_image_sft"):
            with self.subTest(model_type=model_type):
                self.assertFalse(models[model_type]["requires_image_reference"])
                self.assertFalse(models[model_type]["supports_image_edit"])
                self.assertFalse(models[model_type]["supports_instruction_image_edit"])

    def test_managed_edit_adapters_are_not_optional_user_loras(self):
        tree = ast.parse((ROOT / "app" / "launch.py").read_text(encoding="utf-8"))
        assignments = [
            item for item in tree.body
            if isinstance(item, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_MANAGED_MOSS_EDIT_LORAS" for target in item.targets)
        ]
        function = next(
            item for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "_is_managed_moss_edit_lora"
        )
        module = ast.Module(body=assignments + [function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"os": os}
        exec(compile(module, "<launch helper>", "exec"), namespace)
        managed = namespace["_is_managed_moss_edit_lora"]
        for filename in ("moss_image_edit_lora_bf16.safetensors", "armV5.safetensors"):
            self.assertTrue(managed({"architecture": "moss_image"}, filename))
        self.assertFalse(managed({"architecture": "moss_image"}, "my-style.safetensors"))
        self.assertFalse(managed({"architecture": "flux"}, "armV5.safetensors"))
        self.assertTrue(managed({"architecture": "moss_image"}, r"C:\loras\ARMV5.SAFETENSORS"))
        self.assertTrue(managed({"architecture": "moss_image"}, "/loras/ARMV5.SAFETENSORS"))

    def test_edit_types_are_enabled_through_additive_visibility_migration(self):
        source = (ROOT / "ui" / "src" / "stores" / "useStore.ts").read_text(encoding="utf-8")
        version = re.search(r"const DEFAULTS_VERSION = (\d+)", source)
        self.assertIsNotNone(version)
        self.assertGreaterEqual(int(version.group(1)), 24)
        self.assertIn("24: ['moss_image_edit', 'moss_image_edit_hq']", source)
        model_types = (ROOT / "ui" / "src" / "types" / "index.ts").read_text(encoding="utf-8")
        self.assertIn("supports_instruction_image_edit?: boolean", model_types)


if __name__ == "__main__":
    unittest.main()
