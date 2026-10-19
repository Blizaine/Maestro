"""Integration regression for a failed silent source/state plan."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.planners.short_film import ShortFilmPlanner

class NativeContractPatchTests(unittest.TestCase):
    def test_silent_repair_keeps_segmentation_camera_and_authored_duration(self):
        rows = []
        for index, verb in enumerate(("opens", "closes"), 1):
            rows.append({
                "title": f"Box beat {index}", "duration_sec": 10,
                "scene_goal": f"Courier {verb} the box", "narrative_role": "setup",
                "scene_type": "action", "continuity_strategy": "independent" if index == 1 else "continuous",
                "continuity_group": "room", "story_scene_number": 1,
                "causal_handoff": "The courier is beside the box", "persistent_story_state": "Courier wears a coat",
                "source_event_ids": ["E1"],
                "physical_state": {"opening": [], "transitions": [], "closing": []},
                "subjects_on_screen": [{"character_id": "char_0", "speaker_name": "Courier", "visual_description": "Adult courier", "wardrobe": "blue coat and black shoes", "position_or_relation": "standing by the box"}],
                "spatial_setup": "Courier stands beside a wooden box", "environment": "small stone room",
                "visual_style": "cinematic", "lighting": "soft daylight", "mood": "quiet",
                "action_beats": [f"Courier {verb} the box"],
                "dialogue_beats": [{"speaker_id": "char_0", "spoken_text": "Implied", "language": "English"}],
                "camera_plan": {"framing": "medium shot", "movement": "static", "movement_intensity": "none"},
                "audio_plan": {"mode": "ambient_only", "ambience": "room tone", "effects": []},
                "ending_beat": "Courier waits", "closing_blocking": "Courier stands by the box",
                "video_prompt": "Invented dialogue: <d>[English] Implied</d>", "multishot": False, "window_prompts": [],
            })
        original = copy.deepcopy(rows)
        state = {"opening": [{"subject_id": "char_0", "predicate": "grounded", "value": "grounded"}], "transitions": [], "closing": [{"subject_id": "char_0", "predicate": "grounded", "value": "grounded"}]}
        patches = [{
            "event_shots": {"E1": 1, "E2": 2},
            "initial_facts": copy.deepcopy(state["opening"]),
            "shot_transitions": [{"shot_index": i, "transitions": []} for i in (1, 2)],
        }]
        planner = ShortFilmPlanner(llm_generate=lambda **kw: "")
        planner._uses_generated_shot_images = False
        planner._video_model = "minimax_h3"
        with patch.object(planner, "_call_llm_json", side_effect=[copy.deepcopy(rows), patches]) as call:
            shots, _ = planner._plan_story_h3_native(
                story_description="A silent short film. A courier opens the box. The courier closes the box. No speech or music.",
                screenplay="A courier opens, then closes a box. COURIER says Implied.",
                reference_image_path=None, char_profiles=[], has_reference=False,
                target_duration=20, fps=24, frames_steps=17, frames_minimum=124,
                frames_maximum=345, require_source_contract=True,
            )
        self.assertEqual(call.call_count, 2)
        first, second = [x.kwargs for x in call.call_args_list]
        self.assertEqual(first["thinking_budget"], 0)
        self.assertEqual(first["json_schema"]["items"]["properties"]["dialogue_beats"]["maxItems"], 0)
        for field in ("opening", "transitions", "closing"):
            self.assertEqual(first["json_schema"]["items"]["properties"]["physical_state"]["properties"][field]["maxItems"], 0)
        event_schema = second["json_schema"]["properties"]["event_transitions"]
        self.assertEqual(event_schema["required"], ["E1", "E2"])
        self.assertEqual(set(event_schema["properties"]), {"E1", "E2"})
        self.assertNotIn("event_id", event_schema["properties"]["E1"]["items"]["properties"])
        self.assertEqual(len(shots), 2)
        for index, shot in enumerate(shots):
            self.assertEqual(shot.metadata["title"], original[index]["title"])
            self.assertEqual(shot.camera_plan.movement, "static")
            self.assertEqual(shot.metadata["_director_authored_duration_sec"], 10)
            self.assertNotIn("Implied", shot.video_prompt)
            self.assertIn(("opens" if index == 0 else "closes") + " the box", shot.video_prompt)
        self.assertEqual(rows, original)

        bad_state = copy.deepcopy(patches)
        bad_state[0]["initial_facts"][0]["value"] = "mounted"
        bad_ownership = copy.deepcopy(patches)
        bad_ownership[0]["event_shots"] = {"E1": 2, "E2": 1}
        for bad_patch, feedback in (
            (bad_state, "must be exactly 'grounded' or 'airborne'"),
            (bad_ownership, "source order"),
        ):
            with self.subTest(feedback=feedback):
                with patch.object(planner, "_call_llm_json", side_effect=[copy.deepcopy(rows), bad_patch, patches]) as retry:
                    repaired, _ = planner._plan_story_h3_native(
                        story_description="A silent short film. A courier opens the box. The courier closes the box. No speech or music.",
                        screenplay="A courier opens, then closes a box.",
                        reference_image_path=None, char_profiles=[], has_reference=False,
                        target_duration=20, fps=24, frames_steps=17, frames_minimum=124,
                        frames_maximum=345, require_source_contract=True,
                    )
                self.assertEqual(retry.call_count, 3)
                self.assertIn(feedback, retry.call_args_list[-1].kwargs["user_prompt"])
                self.assertEqual(len(repaired), 2)
                self.assertEqual(repaired[0].camera_plan.movement, "static")
                self.assertEqual(rows, original)

        with patch.object(planner, "_call_llm_json", side_effect=[copy.deepcopy(rows), bad_state, bad_state]) as exhausted:
            with self.assertRaisesRegex(RuntimeError, "No video jobs were queued"):
                planner._plan_story_h3_native(
                    story_description="A silent short film. A courier opens the box. The courier closes the box. No speech or music.",
                    screenplay="A courier opens, then closes a box.",
                    reference_image_path=None, char_profiles=[], has_reference=False,
                    target_duration=20, fps=24, frames_steps=17, frames_minimum=124,
                    frames_maximum=345, require_source_contract=True,
                )
        self.assertEqual(exhausted.call_count, 3)

if __name__ == "__main__":
    unittest.main()
