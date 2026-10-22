"""Offline checks for Z-Image's paired ControlNet checkpoint exports."""

import json
from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1] / "app"


class ZImageDownloadTests(unittest.TestCase):
    def test_hidden_eight_step_controlnet_uses_matching_union_exports(self):
        defaults = json.loads((APP_ROOT / "defaults/z_image_control2_1_8s.json").read_text(encoding="utf-8"))
        model = defaults["model"]
        self.assertEqual(model["architecture"], "z_image_control2_1")
        self.assertIs(model["visible"], False)
        prefix = "https://huggingface.co/DeepBeepMeep/Z-Image/resolve/d2c74b46fb8cf2c5e9821663e31f94267da26be5/"
        self.assertEqual(model["modules"], [[
            prefix + "Z-Image-Turbo-Fun-Controlnet-Union2.1-8steps_bf16.safetensors",
            prefix + "Z-Image-Turbo-Fun-Controlnet-Union2.1-8steps_quanto_bf16_int8.safetensors",
        ]])
        self.assertEqual(defaults["num_inference_steps"], 8)


if __name__ == "__main__":
    unittest.main()
