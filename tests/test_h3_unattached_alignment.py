"""Unattached Frames inputs cannot supply a first-frame alignment image."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import llm_service


class UnattachedAlignmentTests(unittest.TestCase):
    def test_only_unattached_frames_alignment_is_removed(self):
        alignment = (
            "For the target video, at 0.00 seconds into the target video, "
            "[Shot 1] (from [Shot 1]) is fully referenced.\n\n"
        )
        body = (
            "integrated_multimodal_description: [Shot 1] A courier walks down a quiet hall.\n"
            "overall_soundscape: Footsteps and room tone.\n"
            "non_diegetic_music: N/A"
        )
        for image_paths, context, expected_alignment in (
            (None, None, False),
            (["start-image.png"], None, True),
            (None, "<Picture 1> supplies the opening frame.", True),
        ):
            with self.subTest(image_paths=image_paths, context=context):
                with patch.object(llm_service, "generate", return_value=alignment + body):
                    result = llm_service.enhance_prompt(
                        "A courier walks down a quiet hall.",
                        model_type="minimax_h3_fused_turbo",
                        image_paths=image_paths, reference_context=context,
                        duration_seconds=5.2, planning_style="faithful",
                    )
                self.assertEqual(alignment.strip() in result, expected_alignment)
                self.assertIn(body, result)


if __name__ == "__main__":
    unittest.main()
