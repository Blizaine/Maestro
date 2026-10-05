"""Model-free checks of the public H3 recipe and Director child-job contract."""

import ast
from contextlib import nullcontext
from functools import partial
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import director_pipeline as pipeline


MODEL = "minimax_h3_ref2va_singularity"
MODELS = ("minimax_h3_singularity", MODEL)
PRESET = "lightx2v-ref2va-turbo4-v0.1-comfy-bf16"
FILENAME = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"


def model_definition(model=MODEL):
    from models.minimax_h3.minimax_h3_handler import family_handler

    defaults = json.loads((ROOT / "app/defaults" / f"{model}.json").read_text(encoding="utf-8"))
    return {
        **defaults["model"],
        **family_handler.query_model_def(defaults["model"]["architecture"], defaults["model"]),
    }


def production_function(path, name, namespace=None, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef)
                     and node.name == class_name).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(namespace or {})
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                                           names=[ast.alias(name="annotations")], level=0),
                             function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class SingularityIntegrationTests(unittest.TestCase):
    def test_api_options_recommend_singularity_recipe_without_changing_stock_h3(self):
        option = production_function(ROOT / "app/launch.py", "_minimax_h3_turbo_option")
        for model in MODELS:
            with self.subTest(model=model):
                recipe = option(model_definition(model))
                self.assertEqual(recipe["preset_id"], PRESET)
                self.assertTrue(recipe["default_enabled"])
                self.assertEqual(recipe["steps"], 4)
                self.assertEqual(recipe["unaccelerated_steps"], 20)
                self.assertEqual(recipe["filename"], FILENAME)
                self.assertIn(PRESET, [item["id"] for item in recipe["presets"]])
        stock = option({"architecture": "minimax_h3_ref2va", "omni_reference": True})
        self.assertEqual(stock["preset_id"], "alibaba-pai-ref2va-pdd-8step")
        self.assertFalse(stock["default_enabled"])
        self.assertIsNone(option({"architecture": "minimax_h3_ref2va", "minimax_h3_fused_turbo": True}))

    def test_director_saves_recipe_and_applies_it_to_child_jobs(self):
        for model in MODELS:
            with self.subTest(model=model):
                self.check_director_child_recipe(model)

    def check_director_child_recipe(self, model):
        definition = model_definition(model)
        project = {"video_model": model, "video_params": {"resolution": "864x480"},
                   "director_max_shot_frames": 124}
        profile = pipeline._create_director_video_execution_profile(
            project, model_def=definition, hardware={"gpu_vram_gb": 24},
        )
        self.assertTrue(profile["turbo_mode"])
        self.assertEqual(profile["turbo_preset"], PRESET)
        self.assertEqual(project["video_params"]["num_inference_steps"], 4)
        child = {"model_type": model, "video_length": 124,
                 "activated_loras": ["style.safetensors", "MiniMax-H3-Ref2VA-Acc-8Step.safetensors"],
                 "loras_multipliers": "0.40 1.00"}
        with patch.object(pipeline, "_wgp", SimpleNamespace(get_model_def=lambda _: definition)):
            pipeline._apply_director_h3_optimizations(child, project["video_params"], profile)
            pipeline._prepare_director_generation_params(child)
        self.assertEqual(child["minimax_h3_turbo_preset"], PRESET)
        self.assertEqual(child["num_inference_steps"], 4)
        self.assertEqual(child["activated_loras"], ["style.safetensors", FILENAME])
        self.assertEqual(child["loras_multipliers"], "0.40 1.00")
        self.assertEqual(child["sample_solver"], "euler")
        self.assertEqual(child["flow_shift"], 12)
        self.assertEqual(child["audio_flow_shift"], 3)
        self.assertEqual(child["guidance_scale"], 1)
        self.assertEqual(child["minimax_h3_reference_detail"], "match")

    def test_director_retains_explicit_turbo_opt_out(self):
        for model in MODELS:
            with self.subTest(model=model):
                self.check_director_turbo_opt_out(model)

    def check_director_turbo_opt_out(self, model):
        project = {"video_model": model, "video_params": {
            "resolution": "864x480", "minimax_h3_turbo_mode": False, "num_inference_steps": 24,
        }, "director_max_shot_frames": 124}
        profile = pipeline._create_director_video_execution_profile(
            project, model_def=model_definition(model), hardware={"gpu_vram_gb": 24},
        )
        self.assertFalse(profile["turbo_mode"])
        self.assertEqual(project["video_params"]["num_inference_steps"], 24)
        child = {"model_type": model, "video_length": 124, "num_inference_steps": 24}
        pipeline._apply_director_h3_optimizations(child, project["video_params"], profile)
        pipeline._prepare_director_generation_params(child)
        self.assertFalse(child["minimax_h3_turbo_mode"])
        self.assertEqual(child["num_inference_steps"], 24)
        self.assertNotIn("activated_loras", child)

    def test_adapter_picker_accepts_ref_turbo_in_frames_and_keeps_stock_checks(self):
        compatible = production_function(ROOT / "app/launch.py", "_lora_is_compatible_with_model")
        for model in MODELS:
            self.assertTrue(compatible(model_definition(model), FILENAME))
        self.assertFalse(compatible({"architecture": "minimax_h3"}, FILENAME))
        self.assertTrue(compatible({"architecture": "minimax_h3_ref2va", "omni_reference": True}, FILENAME))
        self.assertFalse(compatible(model_definition(MODELS[0]), "MiniMax-H3-FL2VA-Acc-8Step.safetensors"))

    def test_native_validator_accepts_ref_adapter_for_frames_singularity_only(self):
        import os
        from models.minimax_h3 import turbo

        validate = production_function(
            ROOT / "app/models/minimax_h3/minimax_h3_main.py", "validate_loras",
            {"os": os, **{name: getattr(turbo, name) for name in (
                "find_minimax_h3_accelerators", "find_minimax_h3_turbo_loras",
                "find_minimax_h3_pdd_loras", "minimax_h3_turbo_preset_for_path",
                "minimax_h3_adapter_workflow")}}, class_name="MiniMaxH3Model",
        )
        for model in MODELS:
            definition = model_definition(model)
            stub = SimpleNamespace(model_def=definition,
                                   omni_reference=bool(definition.get("omni_reference")),
                                   release_special_loras=lambda: None)
            validate(stub, [FILENAME])
            self.assertTrue(stub._turbo_lora_active)
            with self.assertRaisesRegex(ValueError, "one accelerator"):
                validate(stub, [FILENAME, "MiniMax-H3-Ref2VA-Acc-8Step.safetensors"])
        stock_frames = SimpleNamespace(model_def={}, omni_reference=False,
                                       release_special_loras=lambda: None)
        with self.assertRaisesRegex(ValueError, "REF2VA"):
            validate(stock_frames, [FILENAME])

    def test_loader_preserves_ref2va_lora_basis_for_shared_singularity_checkpoint(self):
        from models.minimax_h3.singularity import validate_minimax_h3_singularity_checkpoint

        checkpoint = {"compressed_modulation": True, "adaln_curve_grid": 1025,
                      "time_embed_dim": 8, "convrot": True,
                      "quantization_format": "int8_tensorwise", "convrot_group_size": 256}
        for singularity in (False, True):
            with self.subTest(singularity=singularity):
                transformer = SimpleNamespace()
                transformer.eval = lambda: transformer
                transformer.requires_grad_ = lambda _value: transformer
                offload = SimpleNamespace(split_linear_modules=Mock(), load_model_data=Mock())
                load = production_function(
                    ROOT / "app/models/minimax_h3/minimax_h3_main.py", "_load_transformer",
                    {"probe_h3_checkpoint": lambda _path: checkpoint.copy(),
                     "_first_path": lambda path: path,
                     "convrot_quantization_info_from_file": lambda _path: {},
                     "validate_minimax_h3_singularity_checkpoint": validate_minimax_h3_singularity_checkpoint,
                     "init_empty_weights": lambda **_kwargs: nullcontext(),
                     "MiniMaxH3Transformer": lambda **_kwargs: transformer,
                     "get_linear_split_map": lambda *_args, **_kwargs: {"qkv": "split"},
                     "offload": offload, "partial": partial,
                     "_strip_transformer_wrappers": lambda state, **_kwargs: state},
                )
                result = load("shared.safetensors", "bf16", qkv_layout="grouped", singularity=singularity)
                self.assertIs(result, transformer)
                offload.load_model_data.assert_called_once()
                if singularity:
                    self.assertEqual(result.h3_lora_model_type, "minimax_h3_ref2va")
                else:
                    self.assertFalse(hasattr(result, "h3_lora_model_type"))


if __name__ == "__main__":
    unittest.main()
