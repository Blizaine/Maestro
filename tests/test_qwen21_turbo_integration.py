"""Production dispatch/download integration without importing the GPU server."""
import ast
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from models.qwen21.qwen21_handler import apply_acceleration_profile, family_handler

ID = "qwen_image_21_7B_turbo"
DEFINITION = json.loads((ROOT / "app/defaults" / (ID + ".json")).read_text(encoding="utf-8"))

class Qwen21TurboIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        cls.functions = {node.name: node for node in source.body if isinstance(node, ast.FunctionDef)}

    def production(self, name, namespace):
        code = compile(ast.Module(body=[self.functions[name]], type_ignores=[]), "wgp.py", "exec")
        exec(code, namespace)
        return namespace[name]

    def test_standard_quantization_selector_uses_downloadable_turbo_files(self):
        namespace = {"models_def": {ID: DEFINITION["model"]}, "os": os, "torch": torch,
            "get_transformer_dtype": lambda *_: torch.bfloat16,
            "quant_router": SimpleNamespace(get_quantization_tokens=lambda key: [key])}
        select = self.production("get_model_filename", namespace)
        self.assertTrue(select(ID, "int8").endswith("qwen_image_21_turbo_7B_int8_convrot.safetensors"))
        self.assertTrue(select(ID, "bf16").endswith("qwen_image_21_turbo_7B_bf16.safetensors"))
        self.assertTrue(all(url.startswith("https://huggingface.co/") for url in DEFINITION["model"]["URLs"]))

    def test_variant_reuses_base_components(self):
        base = family_handler.query_model_files({}, "qwen_image_21_7B")
        turbo = family_handler.query_model_files(DEFINITION["model"], ID)
        self.assertEqual(turbo, base)
        self.assertTrue(DEFINITION["model"]["qwen21_turbo_checkpoint"])

    def test_server_dispatch_applies_checkpoint_identity_to_restored_settings(self):
        definitions = {ID: DEFINITION["model"], "qwen_image_21_7B": {}}
        handler = SimpleNamespace(apply_acceleration_profile=apply_acceleration_profile)
        namespace = {"get_model_handler": lambda _: handler, "get_model_def": definitions.get}
        dispatch = self.production("apply_model_acceleration_profile", namespace)
        stale = {"num_inference_steps": 40, "guidance_scale": 4, "sample_solver": "default"}
        normalized = dispatch(ID, stale)
        self.assertEqual((normalized["sample_solver"], normalized["num_inference_steps"], normalized["guidance_scale"]),
                         ("qwen21_turbo", 8, 1.0))
        self.assertEqual(stale["sample_solver"], "default")
        with self.assertRaises(ValueError):
            dispatch("qwen_image_21_7B", {**stale, "sample_solver": "qwen21_turbo"})
        self.assertEqual(dispatch("qwen_image_21_7B", stale)["num_inference_steps"], 40)

    def test_http_boundary_captures_the_actual_recipe_and_reports_mismatch_as_400(self):
        source = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        prepare = next(node for node in source.body if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == "_prepare_generation_submission")
        # Run the production boundary statement: it must precede both
        # generation preparation and the metadata snapshot, without a GPU server.
        boundary = next(node for node in prepare.body if isinstance(node, ast.If)
                        and any(isinstance(child, ast.Attribute)
                                and child.attr == "apply_model_acceleration_profile"
                                for child in ast.walk(node)))
        class HTTPError(Exception):
            def __init__(self, status_code, detail):
                self.status_code, self.detail = status_code, detail
        definitions = {ID: DEFINITION["model"], "qwen_image_21_7B": {}}
        def normalize(model_type, settings):
            return apply_acceleration_profile(settings, model_def=definitions[model_type])
        code = compile(ast.Module(body=[boundary], type_ignores=[]), "launch.py", "exec")
        body = {"model_type": ID, "sample_solver": "default", "num_inference_steps": 40,
                "guidance_scale": 4, "prompt": "A blue bird", "image_refs": ["source.png"]}
        namespace = {"body": body, "is_sfx": False, "wgp": SimpleNamespace(apply_model_acceleration_profile=normalize),
                     "HTTPException": HTTPError}
        exec(code, namespace)
        self.assertEqual((body["sample_solver"], body["num_inference_steps"], body["guidance_scale"]),
                         ("qwen21_turbo", 8, 1.0))
        self.assertEqual(body["image_refs"], ["source.png"])
        body["model_type"] = "qwen_image_21_7B"
        with self.assertRaises(HTTPError) as caught:
            exec(code, namespace)
        self.assertEqual(caught.exception.status_code, 400)
        namespace["is_sfx"] = True
        body["model_type"] = "mmaudio_sfx"
        exec(code, namespace)  # Virtual SFX models have their own renderer, no checkpoint handler.

    def test_pipeline_executes_eight_callbacks_with_official_turbo_profile(self):
        from test_qwen21_upstream import tiny_pipeline
        pipe = tiny_pipeline()
        steps = []
        result = pipe(prompt="An orange robot", width=64, height=64,
                      num_inference_steps=8, true_cfg_scale=1.0,
                      sample_solver="qwen21_turbo", qwen21_turbo_checkpoint=True,
                      use_kv_cache=False, output_type="latent",
                      callback_on_step_end=lambda _, step, timestep, tensors: (steps.append(step) or tensors))
        self.assertEqual(steps, list(range(8)))
        self.assertTrue(torch.isfinite(result.images).all())

    def test_direct_runtime_uses_the_same_checkpoint_guard_before_loading_conditioning(self):
        from models.qwen21.runtime import model_factory
        runtime = object.__new__(model_factory)
        runtime.model_def = DEFINITION["model"]
        with self.assertRaises(ValueError):
            runtime.generate(input_prompt="A blue bird", sample_solver="viggle_v01")
        runtime.model_def = {}
        with self.assertRaises(ValueError):
            runtime.generate(input_prompt="A blue bird", sample_solver="qwen21_turbo")

if __name__ == "__main__":
    unittest.main()
