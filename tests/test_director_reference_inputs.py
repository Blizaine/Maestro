"""References must reach the writer as well as the eventual video model."""
import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.reference_inputs import planning_reference_inputs


class TestDirectorReferenceInputs(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.paths = [str(Path(temp.name) / name) for name in ("band.png", "person.png", "stage.png")]
        for path in self.paths:
            Path(path).touch()

    def test_omni_group_photo_becomes_visible_to_planner_without_changing_render_inputs(self):
        params = {"minimax_h3_references": [{"type": "image", "path": self.paths[0]}]}
        original = copy.deepcopy(params)
        result = planning_reference_inputs(params)
        self.assertEqual(result["reference_image_path"], self.paths[0])
        self.assertEqual(result["character_ref_paths"], [])
        self.assertEqual(params, original)

    def test_preserves_legacy_primary_and_named_references_and_deduplicates(self):
        result = planning_reference_inputs({
            "reference_image_path": self.paths[0],
            "character_ref_paths": [self.paths[1]], "character_ref_labels": ["Guitarist"],
            "minimax_h3_references": [
                {"type": "image", "path": self.paths[1], "name": "Guitarist"},
                {"type": "image", "path": self.paths[2], "image_intent": "scene", "name": "Stage"},
            ],
        })
        self.assertEqual(result["reference_image_path"], self.paths[0])
        self.assertEqual(result["character_ref_paths"], [self.paths[1]])
        self.assertEqual(result["character_ref_labels"], ["Guitarist"])
        self.assertEqual(result["location_ref_paths"], [self.paths[2]])
        self.assertEqual(result["location_ref_labels"], ["Stage"])

    def test_scene_image_does_not_become_a_character_reference(self):
        result = planning_reference_inputs({"minimax_h3_references": [
            {"type": "image", "path": self.paths[2], "image_intent": "scene", "role": "Stage"},
            {"type": "image", "path": self.paths[0], "image_intent": "identity"},
        ]})
        self.assertEqual(result["reference_image_path"], self.paths[0])
        self.assertEqual(result["location_ref_paths"], [self.paths[2]])
        self.assertEqual(result["location_ref_labels"], ["Stage"])

    def test_audio_video_and_missing_files_are_not_attached_as_images(self):
        result = planning_reference_inputs({"minimax_h3_references": [
            {"type": "audio", "path": self.paths[0]},
            {"type": "video", "path": self.paths[1]},
            {"type": "image", "path": "missing.png"}, None,
        ]})
        self.assertIsNone(result["reference_image_path"])
        self.assertEqual(result["character_ref_paths"], [])

    def test_pipeline_passes_group_photo_and_identity_contract_to_actual_music_writer(self):
        from services import director_pipeline as pipeline, llm_service
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps([{
                "scene_goal": "A solo view of the drummer from the band photo",
                "scene_type": "performance",
                "subjects_on_screen": [{"character_id": "drummer", "speaker_name": "Drummer",
                    "visual_description": "The long-haired drummer from the reference image",
                    "location_in_frame": "center", "facing_direction": "front"}],
                "environment": "The concert stage in the reference image",
                "visual_style": "Live concert photography", "lighting": "Stage lights",
                "mood": "Energetic", "action_beats": ["The drummer strikes the cymbal"],
                "camera_plan": {"framing": "close-up", "movement": "steady", "movement_intensity": "subtle"},
                "ending_beat": "The same drummer continues the beat",
                "video_prompt": "The long-haired drummer from the reference image strikes the cymbal.",
                "window_prompts": [],
            }])

        params = {
            "video_model": "minimax_h3_ref2va_fused_turbo", "scene_description": "Show the band photo's drummer playing.",
            "_director_shot_image_policy": "direct_references",
            "planned_clips": [{"start": 0, "end": 124/24, "duration_frames": 124, "duration_sec": 124/24}],
            "minimax_h3_references": [{"type": "image", "path": self.paths[0]}],
        }
        model = {"architecture": "minimax_h3_ref2va", "omni_reference": True,
                 "fps": 24, "frames_minimum": 124, "frames_maximum": 345, "frames_steps": 17}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(pipeline, "_wgp", SimpleNamespace(
                server_config={"services": {}}, get_model_def=lambda _: model)))
            for name in ("_update_pipeline", "_save_pipeline_state", "_capture_llm_pass"):
                stack.enter_context(mock.patch.object(pipeline, name))
            for name in ("generate", "generate_streaming"):
                stack.enter_context(mock.patch.object(llm_service, name, side_effect=generate))
            plans, _ = pipeline._run_planning_v2("reference-fixture", params, "music_video")
        self.assertTrue(calls)
        self.assertEqual(calls[0]["image_paths"], [self.paths[0]])
        self.assertIn("VISUAL REFERENCE IDENTITY", calls[0]["system_prompt"])
        self.assertIn("group photo", calls[0]["system_prompt"])
        self.assertIn("long-haired drummer", plans[0]["video_prompt"])
        self.assertEqual(params["minimax_h3_references"][0]["path"], self.paths[0])


if __name__ == "__main__":
    unittest.main()
