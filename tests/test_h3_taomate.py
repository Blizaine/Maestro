import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "app/models/minimax_h3/turbo.py"
spec = importlib.util.spec_from_file_location("taomate_turbo", PATH)
turbo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(turbo)
PRESET = "taomate-fl2va-3step-rank19"


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


if __name__ == "__main__":
    unittest.main()
