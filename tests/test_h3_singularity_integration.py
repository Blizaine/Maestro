"""Model-free checks of the public H3 recipe and Director child-job contract."""

import ast
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import director_pipeline as pipeline


MODEL = "minimax_h3_ref2va_singularity"
PRESET = "lightx2v-ref2va-turbo4-v0.1-comfy-bf16"
FILENAME = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"


def model_definition():
    defaults = json.loads((ROOT / "app/defaults" / f"{MODEL}.json").read_text(encoding="utf-8"))
    return {
        **defaults["model"], "omni_reference": True, "fps": 24,
        "frames_minimum": 124, "frames_steps": 17, "frames_maximum": 345,
    }


class SingularityIntegrationTests(unittest.TestCase):
    def test_api_options_recommend_singularity_recipe_without_changing_stock_h3(self):
        path = ROOT / "app/launch.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_minimax_h3_turbo_option")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        option = namespace["_minimax_h3_turbo_option"]
        recipe = option(model_definition())
        self.assertEqual(recipe["preset_id"], PRESET)
        self.assertTrue(recipe["default_enabled"])
        self.assertEqual(recipe["steps"], 4)
        self.assertEqual(recipe["unaccelerated_steps"], 20)
        self.assertEqual(recipe["filename"], FILENAME)
        stock = option({"architecture": "minimax_h3_ref2va", "omni_reference": True})
        self.assertEqual(stock["preset_id"], "alibaba-pai-ref2va-pdd-8step")
        self.assertFalse(stock["default_enabled"])
        self.assertIsNone(option({"architecture": "minimax_h3_ref2va", "minimax_h3_fused_turbo": True}))

    def test_director_saves_recipe_and_applies_it_to_child_jobs(self):
        definition = model_definition()
        project = {"video_model": MODEL, "video_params": {"resolution": "864x480"},
                   "director_max_shot_frames": 124}
        profile = pipeline._create_director_video_execution_profile(
            project, model_def=definition, hardware={"gpu_vram_gb": 24},
        )
        self.assertTrue(profile["turbo_mode"])
        self.assertEqual(profile["turbo_preset"], PRESET)
        self.assertEqual(project["video_params"]["num_inference_steps"], 4)
        child = {"model_type": MODEL, "video_length": 124,
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
        project = {"video_model": MODEL, "video_params": {
            "resolution": "864x480", "minimax_h3_turbo_mode": False, "num_inference_steps": 24,
        }, "director_max_shot_frames": 124}
        profile = pipeline._create_director_video_execution_profile(
            project, model_def=model_definition(), hardware={"gpu_vram_gb": 24},
        )
        self.assertFalse(profile["turbo_mode"])
        self.assertEqual(project["video_params"]["num_inference_steps"], 24)
        child = {"model_type": MODEL, "video_length": 124, "num_inference_steps": 24}
        pipeline._apply_director_h3_optimizations(child, project["video_params"], profile)
        pipeline._prepare_director_generation_params(child)
        self.assertFalse(child["minimax_h3_turbo_mode"])
        self.assertEqual(child["num_inference_steps"], 24)
        self.assertNotIn("activated_loras", child)


if __name__ == "__main__":
    unittest.main()
