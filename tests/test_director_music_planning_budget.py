"""Large H3 state records must fit within bounded music-planning responses."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.planners.music_video import MusicVideoPlanner


class MusicPlanningBudgetTests(unittest.TestCase):
    def request(self, planner, count, model, **kwargs):
        return planner._plan_with_llm(
            clips=[{"start": i * 10, "end": (i + 1) * 10} for i in range(count)],
            clip_contexts=[f"Global clip {i + 1}" for i in range(count)],
            scene_description="An armored performer travels to a castle.",
            bpm=120, has_reference=False, reference_image_path=None,
            char_profiles=[], performer_map={}, video_model=model, **kwargs,
        )

    def test_h3_full_song_requests_smaller_batches_and_preserves_global_context(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        planner._uses_generated_shot_images = False
        with patch.object(planner, "_run_checkpointed_json_batches", return_value=[]) as batches:
            self.request(planner, 23, "minimax_h3_ref2va_fused_turbo")
        config = batches.call_args.kwargs
        self.assertLessEqual(config["batch_size"], 3)
        self.assertEqual(config["repair_batch_size"], 1)
        with patch.object(planner, "_call_llm_json", return_value=[]) as generate:
            config["call_batch"](1, 0, config["items"][:3], None)
            config["repair_batch"](2, 3, config["items"][3:4], None)
        first, repair = [call.kwargs for call in generate.call_args_list]
        self.assertIn("global clips 1-3 of 23", first["user_prompt"])
        self.assertIn("global clips 4-4 of 23", repair["user_prompt"])
        self.assertIn("TAIL REPAIR", repair["user_prompt"])
        self.assertGreaterEqual(first["max_tokens"], 7000)
        self.assertLessEqual(first["max_tokens"], 10000)
        self.assertEqual(first["thinking_budget"], 0)
        schema = first["json_schema"]["items"]
        self.assertNotIn("physical_state", schema["properties"])
        self.assertIn("continuity_strategy", schema["required"])
        self.assertNotIn("physical_state", first["system_prompt"])
        self.assertIn("local shot direction", first["system_prompt"])
        self.assertIn("support and contact", first["system_prompt"])

    def test_h3_short_form_keeps_the_output_allowance_for_json(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        planner._uses_generated_shot_images = False
        with patch.object(planner, "_call_llm_json", return_value=[]) as generate:
            self.request(planner, 3, "minimax_h3", _bounded_music_batch=True,
                         _preserve_short_music_reasoning=True)
        self.assertEqual(generate.call_args.kwargs["thinking_budget"], 0)
        self.assertGreaterEqual(generate.call_args.kwargs["max_tokens"], 7000)

    def test_tail_repair_receives_final_pose_without_replaying_previous_opening(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        planner._uses_generated_shot_images = False
        with patch.object(planner, "_run_checkpointed_json_batches", return_value=[]) as batches:
            self.request(planner, 23, "minimax_h3_ref2va_fused_turbo")
        config = batches.call_args.kwargs
        previous = {
            "ending_beat": "Ari finishes crouched beside the amplifier.",
            "physical_state": {
                "opening": [{"subject_id": "lead", "predicate": "posture", "value": "standing upright"}],
                "transitions": [{
                    "subject_id": "lead", "predicate": "posture",
                    "from_value": "standing upright", "to_value": "crouched",
                    "action": "Ari crouches beside the amplifier",
                }],
                "closing": [{"subject_id": "lead", "predicate": "posture", "value": "crouched"}],
            },
        }
        with patch.object(planner, "_call_llm_json", return_value=[]) as generate:
            config["repair_batch"](2, 3, config["items"][3:4], previous)
        request = generate.call_args.kwargs["user_prompt"]
        self.assertIn("ending: Ari finishes crouched beside the amplifier.", request)
        self.assertNotIn('"value": "crouched"', request)
        self.assertNotIn("standing upright", request)
        self.assertNotIn("physical_state", request)
        self.assertIn("local visual direction", request)
        self.assertIn("support, contact, grounding, prop ownership, and location", request)

    def test_ltx_retains_its_six_row_batches_and_short_form_reasoning(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        planner._uses_generated_shot_images = False
        with patch.object(planner, "_run_checkpointed_json_batches", return_value=[]) as batches:
            self.request(planner, 23, "ltx2_25")
        self.assertEqual(batches.call_args.kwargs["batch_size"], 6)
        with patch.object(planner, "_call_llm_json", return_value=[]) as generate:
            self.request(planner, 3, "ltx2_25", _bounded_music_batch=True,
                         _preserve_short_music_reasoning=True)
        self.assertEqual(generate.call_args.kwargs["thinking_budget"], 4096)
        self.assertEqual(generate.call_args.kwargs["max_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
