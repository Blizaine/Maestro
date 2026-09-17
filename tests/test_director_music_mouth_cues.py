"""Music analysis and performer ownership must agree with visible mouth action."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.h3_dialogue import compile_h3_clip_plans, compile_h3_official_prompt, validate_h3_prompt_contract
from services.director.music_performance import constrain_music_performance, music_performance_direction
from services.director.planners.music_video import MusicVideoPlanner
from services.director.schema import ProductionPlan


SINGER = {"speaker_name": "Mara", "visual_description": "Mara the lead singer in red", "performance_role": "vocalist"}
DRUMMER = {"speaker_name": "Cal", "visual_description": "Cal the drummer in blue", "performance_role": "instrumentalist"}


class MusicMouthCueTests(unittest.TestCase):
    def test_intro_retains_energy_camera_and_identity_without_vocalization(self):
        source = (
            "Mara the lead singer grips the mic stand. She throws her head back slightly, "
            "chest lifted, and opens her mouth wide mid-vocalization. "
            "The camera pushes in. She continues to open her mouth wide, breathing heavily."
        )
        for activity in ("unknown", "silent"):
            with self.subTest(activity=activity):
                result = constrain_music_performance(source, [SINGER], activity)
                self.assertNotIn("mid-vocalization", result)
                self.assertNotIn("open her mouth", result)
                self.assertNotIn("breathing heavily", result)
                for kept in ("grips the mic stand", "throws her head back", "chest lifted", "camera pushes in"):
                    self.assertIn(kept, result)
                self.assertIn("lips relaxed and closed", result)

    def test_drummer_shout_is_removed_without_losing_cymbal_strike(self):
        source = (
            "Cal the drummer drives the sticks onto the snare, his torso rotating forward. "
            "His mouth is open in a determined shout, and his left arm flicks up to strike the crash cymbal. "
            "Camera tilts toward his face. The crowd cheers."
        )
        result = constrain_music_performance(source, [DRUMMER], "active", project_context="The audience cheers for the band.")
        self.assertNotIn("determined shout", result)
        for kept in ("torso rotating forward", "left arm flicks up", "strike the crash cymbal", "Camera tilts", "crowd cheers"):
            self.assertIn(kept, result)
        self.assertIn("His lips stay relaxed and closed", result)

    def test_subject_id_restores_ownership_after_a_crowd_insert(self):
        source = (
            "[Shot 1] Mara (S1) holds the mic stand. The crowd cheers. "
            "[Shot 2] At 00:01.800, S1 leans forward. She continues to open her mouth wide, breathing heavily."
        )
        result = constrain_music_performance(source, [SINGER], "unknown")
        self.assertIn("The crowd cheers", result)
        self.assertIn("S1 leans forward", result)
        self.assertIn("continues to keep her lips relaxed and closed", result)
        self.assertNotIn("mouth wide", result)
        self.assertNotIn("breathing heavily", result)

    def test_mixed_band_keeps_singer_and_backing_guitarist_vocals(self):
        guitar = {"speaker_name": "Rae", "visual_description": "Rae the guitarist", "performance_role": "vocalist"}
        source = "Mara sings into the microphone. Cal the drummer shouts at his kit. Rae sings backing vocals."
        result = constrain_music_performance(source, [SINGER, DRUMMER, guitar], "active")
        self.assertIn("Mara sings into the microphone", result)
        self.assertIn("Rae sings backing vocals", result)
        self.assertNotIn("drummer shouts", result)

    def test_real_vocals_and_legacy_unknown_roles_are_not_muted(self):
        source = "Mara opens her mouth wide and sings the chorus."
        self.assertEqual(constrain_music_performance(source, [SINGER], "active"), source)
        self.assertEqual(constrain_music_performance(source, [{"visual_description": "Mara"}], "unknown"), source)

    def test_wind_embouchure_user_expression_and_negative_directions_survive(self):
        flute = {"visual_description": "the flute player", "performance_role": "instrumentalist"}
        source = "The flute player opens her mouth slightly to take a breath, then blows into the flute."
        self.assertEqual(constrain_music_performance(source, [flute], "silent"), source)
        requested = "Cal the drummer shouts triumphantly."
        self.assertEqual(constrain_music_performance(requested, [DRUMMER], "active", project_context=requested), requested)
        direction = music_performance_direction([DRUMMER], "active", project_context=requested)
        self.assertIn("may perform the user's explicitly requested expression", direction)
        self.assertNotIn("Cal the drummer in blue keeps their mouth closed", direction)
        negative = "Cal does not sing or lip-sync. His mouth stays closed."
        self.assertEqual(constrain_music_performance(negative, [DRUMMER], "active"), negative)

    def test_new_plan_carries_analysis_through_images_windows_endings_and_serialization(self):
        def generate(**kwargs):
            self.assertIn("relaxed closed lips by default", kwargs["prompt"])
            return json.dumps([{
                "subjects_on_screen": [SINGER],
                "audio_plan": {"vocal_activity": "active"},  # The LLM cannot override analysis.
                "video_prompt": "Mara bellows into the microphone. Camera pushes in.",
                "image_prompt": "Mara opens her mouth wide mid-vocalization.",
                "window_prompts": ["Mara sings the riff. Camera pans to her hands."],
                "ending_beat": "Mara opens her mouth wide mid-vocalization.",
                "action_beats": ["Mara shouts loudly."],
            }])
        planner = MusicVideoPlanner(llm_generate=generate, llm_generate_streaming=generate)
        with patch("services.director.planners.music_video.classify_vocal_intervals", return_value=["unknown"]):
            plan = planner.plan(
                clips=[{"start": 0, "end": 13, "label": "intro"}],
                scene_description="Mara sings lead. A long guitar intro starts the song.",
                video_model="minimax_h3_i2va", shot_image_policy="generated_images",
            )
        shot = ProductionPlan.from_dict(json.loads(json.dumps(plan.to_dict()))).shots[0]
        self.assertEqual(shot.audio_plan.vocal_activity, "unknown")
        for text in (shot.video_prompt, shot.image_prompt, shot.ending_beat, *shot.window_prompts, *shot.action_beats):
            self.assertNotIn("bellows", text)
            self.assertNotIn("mid-vocalization", text)
            self.assertNotIn("sings the riff", text)
            self.assertNotIn("shouts loudly", text)
        self.assertIn("Camera pushes in", shot.video_prompt)
        self.assertIn("Guitar riffs and drum hits never drive the mouth", shot.video_prompt)

    def test_h3_final_compile_catches_conflicts_in_source_and_ending(self):
        for mode in ("ref2va", "i2va", "fl2va"):
            with self.subTest(mode=mode):
                plan = {
                    "video_prompt": "Mara bellows into the microphone. Camera pushes in.",
                    "_director_subjects_on_screen": [SINGER],
                    "_director_closing_blocking": "Mara opens her mouth wide mid-vocalization.",
                    "_director_audio_plan": {"mode": "music_driven", "vocal_activity": "unknown"},
                    "_director_h3_prompt_mode": mode,
                    "_director_duration_sec": 13,
                }
                compile_h3_clip_plans([plan])
                first = plan["video_prompt"]
                self.assertNotIn("bellows", first)
                self.assertNotIn("mid-vocalization", first)
                self.assertIn("Camera pushes in", first)
                self.assertIn("Vocal activity in this interval is unconfirmed", first)
                self.assertEqual(validate_h3_prompt_contract(first, mode=mode), [])
                compile_h3_clip_plans([plan])
                self.assertEqual(plan["video_prompt"], first)

    def test_contract_and_cleaned_prose_are_idempotent_on_direct_recompile(self):
        kwargs = dict(mode="ref2va", audio_plan={"mode": "music_driven", "vocal_activity": "active"})
        prompt, _ = compile_h3_official_prompt("Cal the drummer shouts. His arm strikes the cymbal.", [DRUMMER], [], **kwargs)
        again, _ = compile_h3_official_prompt(prompt, [DRUMMER], [], **kwargs)
        self.assertIn("His arm strikes the cymbal", again)
        self.assertEqual(again.count("Vocal ownership stays"), 1)
        self.assertNotIn("drummer shouts", again)
        self.assertIn("singer continues off screen", again)

    def test_story_speech_bypasses_music_cleanup(self):
        result, _ = compile_h3_official_prompt(
            "Mara shouts across the room.", [SINGER], [{"speaker_id": "Mara", "spoken_text": "Come here!"}],
            mode="i2va", audio_plan={"mode": "dialogue_driven", "vocal_activity": "silent"},
        )
        self.assertIn("Mara shouts across the room", result)
        self.assertIn("Come here!", result)


if __name__ == "__main__":
    unittest.main()
