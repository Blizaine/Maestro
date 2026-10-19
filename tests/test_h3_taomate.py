import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
PATH = APP / "models/minimax_h3/turbo.py"
LAUNCH_PATH = APP / "launch.py"
spec = importlib.util.spec_from_file_location("taomate_turbo", PATH)
turbo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(turbo)
PRESET = "taomate-fl2va-3step-rank19"
DATED_PRESET = "taomate-fl2va-3step-2026-10-15"


def _load_launch_function(name):
    tree = ast.parse(LAUNCH_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(LAUNCH_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class TaoMateRecipeTests(unittest.TestCase):
    def test_recipe_replaces_accelerator_preserving_style_lora(self):
        body = {"minimax_h3_turbo_mode": True, "minimax_h3_turbo_preset": PRESET,
                "activated_loras": ["style.safetensors", "MiniMax-H3-FL2VA-Acc-8Step.safetensors"],
                "loras_multipliers": "0.75 1.0", "num_inference_steps": 20, "guidance_scale": 5}
        turbo.normalize_minimax_h3_turbo_request(body, full_checkpoint=False, workflow="fl2va")
        self.assertEqual(body["num_inference_steps"], 3)
        self.assertEqual(body["guidance_scale"], 1)
        self.assertEqual(body["sample_solver"], "euler")
        self.assertEqual(body["activated_loras"][0], "style.safetensors")
        self.assertEqual(body["loras_multipliers"], "0.75 1.00")
        self.assertEqual(len(turbo.find_minimax_h3_turbo_loras(body["activated_loras"])), 1)
        self.assertEqual(turbo.h3_scheduler_grid_points(3, turbo_active=True), 4)

    def test_taomate_is_not_offered_for_references_or_promoted_to_default(self):
        self.assertNotIn(PRESET, [p["id"] for p in turbo.minimax_h3_turbo_presets_for_workflow("ref2va", full_checkpoint=True)])
        with self.assertRaises(ValueError):
            turbo.minimax_h3_turbo_preset(PRESET, workflow="ref2va")
        self.assertEqual(turbo.minimax_h3_turbo_preset(workflow="fl2va")["steps"], 8)

    def test_dated_taomate_is_a_pinned_fl2va_selector_option(self):
        fl2va = {
            "architecture": "minimax_h3",
            "minimax_h3_full_checkpoint": False,
        }
        ref2va = {
            "architecture": "minimax_h3_ref2va",
            "minimax_h3_full_checkpoint": False,
            "minimax_h3_lora_workflow": "ref2va",
        }
        option = _load_launch_function("_minimax_h3_turbo_option")
        app_path = str(APP)
        added_app_path = app_path not in sys.path
        if added_app_path:
            sys.path.insert(0, app_path)
        try:
            fl2va_option = option(fl2va)
            ref2va_option = option(ref2va)
        finally:
            if added_app_path:
                sys.path.remove(app_path)

        visible_ids = {preset["id"] for preset in fl2va_option["presets"]}
        self.assertIn(DATED_PRESET, visible_ids)
        self.assertIn(PRESET, visible_ids)
        self.assertNotIn(
            DATED_PRESET,
            {preset["id"] for preset in ref2va_option["presets"]},
        )

        selected = next(
            preset
            for preset in fl2va_option["presets"]
            if preset["id"] == DATED_PRESET
        )
        self.assertEqual(
            selected["filename"],
            "TaoMate-H3-FL2VA-3Step-ComfyUI-BF16.safetensors",
        )
        self.assertEqual(selected["steps"], 3)
        self.assertEqual(selected["weight"], 1.0)
        self.assertEqual(selected["workflow"], "fl2va")
        self.assertEqual(
            selected["revision"],
            "630975da14fd3c84169d6cd5092da3ce1287d137",
        )
        self.assertEqual(
            selected["generation_settings"]["sample_solver"], "euler"
        )

        manifest_entry = turbo.minimax_h3_turbo_preset(
            DATED_PRESET, workflow="fl2va", full_checkpoint=False
        )
        self.assertEqual(manifest_entry["repo_id"], "DeepBeepMeep/MiniMax-H3")
        self.assertEqual(
            manifest_entry["remote_path"],
            "loras/TaoMate-H3-FL2VA-3Step-ComfyUI-BF16.safetensors",
        )
        self.assertEqual(
            manifest_entry["sha256"],
            "e0faaa6f5b4fc7cf699fe3a7e391fba21985b3ad8e87f21a200f28df35cca621",
        )
        self.assertEqual(manifest_entry["size"], 1_240_543_144)

    def test_dated_profile_and_legacy_generic_profile_are_both_preserved(self):
        old_path = APP / "profiles/minimax_h3/Turbo Taomate 3 steps.json"
        old_profile = json.loads(old_path.read_text(encoding="utf-8"))
        self.assertEqual(
            old_profile["activated_loras"],
            [
                "https://huggingface.co/DeepBeepMeep/MiniMax-H3/resolve/main/loras/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors"
            ],
        )
        self.assertEqual(old_profile["num_inference_steps"], 3)
        self.assertEqual(old_profile["guidance_scale"], 1)

        dated_path = (
            APP
            / "profiles/minimax_h3_fl2va/Turbo Taomate FL2VA 3 steps (2026-10-15).json"
        )
        dated_profile = json.loads(dated_path.read_text(encoding="utf-8"))
        self.assertEqual(
            Path(dated_profile["activated_loras"][0]).name,
            "TaoMate-H3-FL2VA-3Step-ComfyUI-BF16.safetensors",
        )
        self.assertEqual(dated_profile["loras_multipliers"], "1.0")
        self.assertEqual(dated_profile["num_inference_steps"], 3)
        self.assertEqual(dated_profile["guidance_scale"], 1)
        self.assertEqual(dated_profile["sample_solver"], "euler")

    def test_dated_taomate_applies_its_three_step_recipe_without_stacking(self):
        body = {
            "minimax_h3_turbo_mode": True,
            "minimax_h3_turbo_preset": DATED_PRESET,
            "activated_loras": [
                "style.safetensors",
                "minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors",
            ],
            "loras_multipliers": "0.75 1.0",
            "num_inference_steps": 20,
            "guidance_scale": 5,
        }
        turbo.normalize_minimax_h3_turbo_request(
            body, full_checkpoint=False, workflow="fl2va"
        )
        self.assertEqual(body["num_inference_steps"], 3)
        self.assertEqual(body["guidance_scale"], 1.0)
        self.assertEqual(body["audio_guidance_scale"], 1.0)
        self.assertEqual(body["sample_solver"], "euler")
        self.assertEqual(
            body["activated_loras"],
            ["style.safetensors", "TaoMate-H3-FL2VA-3Step-ComfyUI-BF16.safetensors"],
        )
        self.assertEqual(body["loras_multipliers"], "0.75 1.00")


if __name__ == "__main__":
    unittest.main()
