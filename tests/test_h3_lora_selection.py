"""Catalog and submission use the same H3 adapter-family guard."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


class H3LoraSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        nodes = [deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_lora_is_compatible_with_model", "_validate_h3_lora_selection"}]
        cls.resolutions = []

        def resolve(model, name):
            cls.resolutions.append((model, name))
            return name

        cls.routes = {"wgp": SimpleNamespace(resolve_lora_path=resolve)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "launch.py", "exec"), cls.routes)

    def test_vdn_adapter_is_hidden_and_rejected_on_ordinary_hybrid_before_loading(self):
        name = "minimax_h3_vdn_turbo_8step_lora_bf16.safetensors"
        model = {"architecture": "minimax_h3", "minimax_h3_lora_workflow": "ref2va"}
        self.assertFalse(self.routes["_lora_is_compatible_with_model"](model, name))
        with self.assertRaisesRegex(ValueError, "dedicated MiniMax H3 VDN model"):
            self.routes["_validate_h3_lora_selection"](model, {"model_type": "imported_frames", "activated_loras": [name]})
        self.assertIn(("imported_frames", name), self.resolutions)
        self.assertTrue(self.routes["_lora_is_compatible_with_model"]({"architecture": "minimax_h3", "vdn": True}, name))

    def test_standard_hybrid_keeps_convertible_turbo_and_ordinary_style_adapters(self):
        model = {"architecture": "minimax_h3", "minimax_h3_lora_workflow": "ref2va"}
        names = ["minimax_h3_turbo_v4_step600_ema.safetensors", "film_style.safetensors",
                 "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"]
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(self.routes["_lora_is_compatible_with_model"](model, name))
        self.routes["_validate_h3_lora_selection"](model, {"model_type": "imported_frames", "activated_loras": names})

    def test_baked_turbo_selectors_keep_style_and_omit_extra_acceleration(self):
        model = {"architecture": "minimax_h3", "minimax_h3_baked_turbo": True}
        self.assertTrue(self.routes["_lora_is_compatible_with_model"](model, "film_style.safetensors"))
        self.assertFalse(self.routes["_lora_is_compatible_with_model"](model, "minimax_h3_turbo_v4_step600.safetensors"))

    def test_fasth3_catalog_is_not_consent_but_submission_requires_explicit_opt_in(self):
        for omni in (False, True):
            model = {"architecture": "minimax_h3", "minimax_h3_fasth3": True, "minimax_h3_fasth3_omniref": omni}
            self.assertTrue(self.routes["_lora_is_compatible_with_model"](model, "portrait.safetensors"))
            self.assertFalse(self.routes["_lora_is_compatible_with_model"](model, "minimax_h3_turbo_v4.safetensors"))
            params = {"model_type": "minimax_h3_fasth3", "activated_loras": ["portrait.safetensors"]}
            with self.subTest(omni=omni), self.assertRaisesRegex(ValueError, "experimental and off by default"):
                self.routes["_validate_h3_lora_selection"](model, params)
            params["minimax_h3_experimental_loras"] = True
            self.routes["_validate_h3_lora_selection"](model, params)


if __name__ == "__main__":
    unittest.main()
