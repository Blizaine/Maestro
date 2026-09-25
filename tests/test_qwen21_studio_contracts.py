"""Studio image enhancement sees the same ordered conditioning as generation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.studio_enhancement import enhancement_request


class ImageConditioningTests(unittest.TestCase):
    def test_control_and_references_are_passed_in_runtime_order(self):
        params = {
            "model_type": "qwen_image_21_7B", "generation_mode": "image",
            "image_mode": 1, "prompt": "Move the dancer to the garden",
            "video_prompt_type": "PVI", "image_guide": "pose.png",
            "image_refs": ["identity.png", "garden.png"],
            "image_start": "stale-video-start.png", "video_length": 1,
        }
        request, sequence = enhancement_request(params, {"image_outputs": True})
        self.assertFalse(sequence)
        self.assertEqual(request["image_paths"], ["pose.png", "identity.png", "garden.png"])
        self.assertIn("source/control image", request["reference_context"])
        self.assertNotIn("seconds", request["reference_context"])

    def test_disabled_control_is_not_sent_to_the_writer(self):
        request, _ = enhancement_request({
            "generation_mode": "image", "video_prompt_type": "I",
            "image_guide": "unused.png", "image_refs": ["subject.png"],
        }, {"image_outputs": True})
        self.assertEqual(request["image_paths"], ["subject.png"])

    def test_image_api_request_does_not_require_a_ui_mode_flag(self):
        request, sequence = enhancement_request({
            "model_type": "qwen_image_21_7B", "video_prompt_type": "VI",
            "image_guide": "canvas.png", "image_refs": ["subject.png"],
        }, {"image_outputs": True})
        self.assertEqual(request["mode"], "image")
        self.assertEqual(request["image_paths"], ["canvas.png", "subject.png"])
        self.assertFalse(sequence)


if __name__ == "__main__":
    unittest.main()
