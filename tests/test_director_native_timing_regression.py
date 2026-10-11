"""Regressions for visual timing after native H3 duration fitting."""

from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.h3_dialogue import (  # noqa: E402
    H3DialogueContractError,
    compile_h3_clip_plans,
    validate_h3_prompt_contract,
)
from services import director_pipeline as pipeline  # noqa: E402
from services.director.planners.short_film import ShortFilmPlanner  # noqa: E402


# Exact prompt and raw durations retained from the failed Gemma Pass 2 output.
# The other shot fields below are compact schema scaffolding; these timing
# values are the production data that exposed the conversion regression.
_GEMMA_RAW_DURATIONS = [12.25, 14.38, 10.12, 12.25, 11.12]
_GEMMA_SHOT_3_PROMPT = (
    "MiniMax H3 generates synchronized picture and stereo sound. "
    "[Shot 1] EXT. ABOVE MISTY VALLEY - AIR. A medium shot shows the Dragon "
    "(char_1) smoothly initiating a bank to the right, its massive bulk "
    "shifting. The Kitten (char_0), in gleaming silver plate armor and a pink "
    "bow, is seated securely, leaning into the turn, gripping the strap. "
    "[Shot 2] At 00:03.550, the camera tracks the Dragon as it increases the "
    "angular sweep of the bank, its head tilting down to observe the distant "
    "stone castle. The Kitten subtly shifts weight forward to counteract the "
    "G-force, maintaining perfect composure. [Shot 3] At 00:06.800, the camera "
    "holds a steady medium shot, capturing the full curve of the bank as the "
    "Dragon closes the distance to the castle. The Kitten is perfectly "
    "visible, leaning into the curve, prepared for landing. overall_soundscape: "
    "Strong, focused wind noise swirls around the Dragon as it executes the "
    "bank. Practical sound: A deep, resonant *whoosh* sound increases, and the "
    "slight creak of the saddle leather under the Kitten's weight is audible. "
    "non_diegetic_music: The soaring orchestral theme transitions into a more "
    "stately, deliberate rhythm, signaling approach."
)


class DirectorNativeTimingRegressionTests(unittest.TestCase):
    @staticmethod
    def _plan(source: str, authored: float, native: float) -> dict:
        return {
            "video_prompt": source,
            "_director_h3_source_prompt": source,
            "_director_h3_prompt_mode": "t2va",
            "_director_duration_sec": native,
            "_director_authored_duration_sec": authored,
            "_director_subjects_on_screen": [],
            "_director_dialogue_beats": [],
        }

    def test_compressed_native_schedule_retimes_only_from_valid_authored_time(self):
        authored = 10.12
        native = 158 / 24
        source = (
            "A rider crosses the wet bridge in a wide view. "
            "[Shot 3] At 00:06.800: the camera finds the rider's face. "
            "The source audio cue: At 00:08.000 anchors the lyric."
        )
        plan = self._plan(source, authored, native)

        compile_h3_clip_plans([plan])

        self.assertIn("At 00:04.424", plan["video_prompt"])
        self.assertIn("At 00:08.000", plan["video_prompt"])
        self.assertEqual(plan["_director_h3_source_prompt"], source)
        self.assertEqual(
            validate_h3_prompt_contract(
                plan["video_prompt"],
                mode="t2va",
                duration_seconds=native,
            ),
            [],
        )
        compiled = plan["video_prompt"]
        compile_h3_clip_plans([plan])
        self.assertEqual(plan["video_prompt"], compiled)

    def test_expanded_native_schedule_retimes_visual_cue_proportionally(self):
        authored = 158 / 24
        native = 243 / 24
        source = (
            "A quiet close-up opens the scene. "
            "[Shot 1] At 00:05.000: the lead turns toward the window."
        )
        plan = self._plan(source, authored, native)

        compile_h3_clip_plans([plan])

        self.assertIn("At 00:07.690", plan["video_prompt"])
        self.assertEqual(
            validate_h3_prompt_contract(
                plan["video_prompt"],
                mode="t2va",
                duration_seconds=native,
            ),
            [],
        )

    def test_expansion_does_not_make_an_authored_invalid_cue_valid(self):
        source = (
            "A quiet close-up opens the scene. "
            "[Shot 1] At 00:10.500: the lead turns toward the window."
        )
        plan = self._plan(source, authored=10.12, native=12.25)

        with self.assertRaisesRegex(
            H3DialogueContractError,
            "Shot 1: authored visual timing is invalid",
        ):
            compile_h3_clip_plans([plan])

    def test_retiming_protects_dialogue_time_literals_and_source_prompt(self):
        source = (
            "A quiet close-up opens the scene. "
            "[Shot 1] At 00:06.800: the lead looks into camera. "
            "<d>[English] Read At 00:09.800 aloud.</d>"
        )
        plan = self._plan(source, authored=10.12, native=158 / 24)
        plan["_director_subjects_on_screen"] = [{
            "character_id": "lead",
            "speaker_name": "Lead",
        }]
        plan["_director_dialogue_beats"] = [{
            "speaker_id": "lead",
            "spoken_text": "Read At 00:09.800 aloud.",
        }]

        compile_h3_clip_plans([plan])

        self.assertIn("At 00:04.424", plan["video_prompt"])
        self.assertIn(
            "<d>[English] Read At 00:09.800 aloud.</d>",
            plan["video_prompt"],
        )
        self.assertEqual(plan["_director_h3_source_prompt"], source)

    def test_shot_plan_serialization_carries_authored_duration_metadata(self):
        planner = ShortFilmPlanner()
        planner._video_model = "minimax_h3_base"
        shot = planner._convert_story_shots(
            [{
                "duration_sec": 158 / 24,
                "_director_authored_duration_sec": 10.12,
                "video_prompt": "A rider crosses a bridge.",
            }],
            char_profiles=[],
            has_reference=False,
            fps=24,
            frames_steps=1,
            frames_minimum=41,
            frames_maximum=158,
        )[0]

        self.assertEqual(
            shot.metadata["_director_authored_duration_sec"], 10.12
        )
        self.assertEqual(
            shot.to_dict()["metadata"]["_director_authored_duration_sec"],
            10.12,
        )

    def test_failed_gemma_timing_flows_through_planner_and_pipeline_conversion(self):
        story = (
            "A 40-second silent fantasy short film. A tiny armored Kitten with "
            "a pink bow meets a huge dark-scaled Dragon on a stone ledge. The "
            "Dragon lowers its shoulder; the Kitten climbs up using the saddle "
            "strap and settles astride its back, with sword sheathed at its left "
            "hip. After the Kitten grips the strap securely, the Dragon takes "
            "flight, crosses a misty valley and banks gently toward a castle. "
            "During flight the Kitten stays physically seated, leaning with the "
            "Dragon. The Dragon lands, becomes stationary and folds its wings "
            "before the Kitten slides down its shoulder and puts both feet on the "
            "ground. End with the Kitten patting the Dragon snout as the castle "
            "gate opens. Preserve this ordered progression, contact and relative "
            "scale; no repeated boarding, floating, unrelated battle, speech or "
            "music. Picture 1 supplies only the Kitten identity and design."
        )
        screenplay = """EXT. STONE LEDGE - MORNING (MISTY)

A TINY KITTEN, clad in ornate silver and dark scale armor, stands perfectly still. A bright pink bow adorns its head. A sheathed sword rests against its left hip. Its gaze is fixed on a massive, dark-scaled DRAGON, which rests coiled upon the ledge. The Dragon’s head lifts slowly, its eyes tracking the Kitten.

The Dragon lowers its enormous shoulder, creating a stable, sloping ramp toward the Kitten.

The Kitten takes three crisp, purposeful steps forward, then begins to climb the Dragon’s saddle strap, its tiny claws finding purchase in the leather. It reaches the back, settles itself firmly, and shifts its weight to balance precisely over the Dragon’s spine.

The Kitten grips the saddle strap securely with both forepaws.

The Dragon pushes off the stone ledge with immense, measured force, launching itself upward into the thick morning mist.

The Dragon surges forward, soaring across a vast, rolling misty valley. The Kitten remains perfectly seated, leaning slightly into the Dragon’s forward momentum, maintaining a poised stillness against the vast expanse.

The Dragon begins a graceful bank, turning its massive body toward a distant, imposing stone castle silhouetted through the haze.

The Dragon slows its powerful flight, descending until its great talons settle firmly onto the mossy stone of the castle grounds. It becomes perfectly stationary. The Dragon smoothly folds its massive wings against its flanks.

The Kitten slides down the Dragon’s shoulder in a controlled cascade, its tiny paws sliding effortlessly down the armor. It plants both small feet onto the damp ground.

The Kitten stops, facing the Dragon’s massive snout. It raises one front paw, hesitates for a beat, then gently pats the Dragon’s snout once.

The heavy, ancient gates of the castle swing open slowly, revealing a darker entryway within.

FADE OUT."""
        compressed_screenplay = """EXT. STONE LEDGE - MORNING (MISTY)

A TINY KITTEN in silver armor and a pink bow stands before a massive DRAGON. The Dragon lifts its head, eyes fixed on the Kitten.

The Dragon lowers its shoulder.

The Kitten takes three purposeful steps, climbing the saddle strap. It reaches the back, settling firmly, gripping the strap with both paws.

The Dragon pushes off, launching into the misty air.

The Dragon surges forward across the valley. The Kitten remains perfectly seated, leaning into the momentum.

The Dragon banks gracefully toward a distant stone castle. It slows, landing solidly on the castle grounds, wings folding.

The Kitten slides down the Dragon’s shoulder, planting its feet on the damp ground. It stops, facing the Dragon’s snout. It raises one paw, pats the snout once.

The heavy castle gates swing open, revealing a dark entryway.

FADE OUT."""

        raw_shots = []
        groups = ["ledge_meeting", "valley_flight", "castle_approach", "castle_arrival", "castle_arrival"]
        for index, authored_duration in enumerate(_GEMMA_RAW_DURATIONS):
            raw_shots.append({
                "title": f"Shot {index + 1}",
                "duration_sec": authored_duration,
                "scene_goal": f"Advance ordered story beat {index + 1}.",
                "narrative_role": "setup" if index == 0 else "rising_action",
                "scene_type": "action",
                "continuity_strategy": "independent" if index == 0 else "continuous",
                "continuity_group": groups[index],
                "story_scene_number": index + 1,
                "causal_handoff": f"The preceding action causes beat {index + 1}.",
                "persistent_story_state": "The Kitten and Dragon preserve their established positions and props.",
                "subjects_on_screen": [],
                "spatial_setup": "The Kitten and Dragon occupy the described story location.",
                "environment": "A misty fantasy landscape with a stone castle.",
                "visual_style": "Grounded cinematic fantasy.",
                "lighting": "Soft morning light through mist.",
                "mood": "Quiet wonder.",
                "action_beats": [f"The Kitten and Dragon complete story beat {index + 1} in order."],
                "dialogue_beats": [],
                "camera_plan": {
                    "framing": "medium shot",
                    "movement": "gentle tracking movement",
                    "movement_intensity": "subtle",
                },
                "audio_plan": {"mode": "ambient_only", "ambience": "Wind through the mist."},
                "ending_beat": f"Beat {index + 1} reaches its visible end state.",
                "closing_blocking": "The Kitten and Dragon are clearly visible.",
                "video_prompt": (
                    _GEMMA_SHOT_3_PROMPT if index == 2 else
                    f"A medium shot shows the Kitten and Dragon completing "
                    f"ordered story beat {index + 1} in the misty landscape. "
                    "No one speaks."
                ),
                "multishot": False,
                "window_prompts": [],
            })

        responses = {
            "voice bible": "[]",
            "acclaimed screenwriter": screenplay,
            "timing editor": compressed_screenplay,
            "film director breaking a screenplay into shots for minimax h3": json.dumps(raw_shots),
        }
        calls = []

        def fake_writer(**kwargs):
            system = str(kwargs.get("system_prompt") or "").casefold()
            calls.append(kwargs)
            if "film director breaking a screenplay into shots for minimax h3" in system:
                return responses["film director breaking a screenplay into shots for minimax h3"]
            if "timing editor" in system:
                return responses["timing editor"]
            if "acclaimed screenwriter" in system:
                return responses["acclaimed screenwriter"]
            if "voice bible" in system:
                return responses["voice bible"]
            self.fail(f"Unexpected planner writer call: {system[:120]}")

        planner = ShortFilmPlanner(
            llm_generate=fake_writer,
            llm_generate_streaming=fake_writer,
        )
        production = planner.plan(
            story_description=story,
            target_duration=40.5,
            target_scenes=5,
            fps=24,
            frames_steps=17,
            frames_minimum=124,
            frames_maximum=345,
            video_model="minimax_h3_base",
            shot_image_policy="prompt_only",
        )

        self.assertEqual(len(calls), 4)
        shot3 = production.shots[2]
        authored_prompt = shot3.video_prompt
        timing = pipeline._director_shot_timing_fields(shot3)
        self.assertEqual(shot3.duration_sec, 158 / 24)
        self.assertEqual(timing["_director_authored_duration_sec"], 10.12)
        self.assertIn("At 00:06.800", authored_prompt)
        self.assertEqual(
            [round(shot.duration_sec * 24) for shot in production.shots],
            [209, 243, 158, 192, 175],
        )

        pipeline_plans = []
        for shot in production.shots:
            pipeline_plans.append({
                "video_prompt": shot.video_prompt,
                "_director_h3_source_prompt": shot.video_prompt,
                "_director_h3_prompt_mode": "t2va",
                "_director_subjects_on_screen": [],
                "_director_dialogue_beats": [],
                **pipeline._director_shot_timing_fields(shot),
            })

        compile_h3_clip_plans(pipeline_plans)

        self.assertIn("At 00:04.424", pipeline_plans[2]["video_prompt"])
        self.assertNotIn("At 00:06.800", pipeline_plans[2]["video_prompt"])
        self.assertEqual(
            pipeline_plans[2]["_director_h3_source_prompt"],
            authored_prompt,
        )
        self.assertIn("Picture 1 supplies only the Kitten identity", story)

    def test_user_edited_timing_flag_survives_save_edit_resume_before_compile(self):
        source = (
            "A medium shot shows the rider seated securely. "
            "[Shot 1] At 00:06.800: the rider looks toward the castle."
        )
        native_duration = 158 / 24
        pid = "director-native-timing-resume"
        with tempfile.TemporaryDirectory() as out_dir:
            params = {
                "pipeline_type": "short_film_story",
                "scene_description": "A rider approaches a castle.",
                "video_model": "minimax_h3_base",
                "_director_shot_image_policy": "prompt_only",
            }
            record = {
                "id": pid,
                "status": "completed",
                "phase": "completed",
                "out_dir": out_dir,
                "created_at": 1.0,
                "params": params,
                "clip_plans": [{
                    "image_prompt": "",
                    "video_prompt": "Original planner prompt.",
                    "_director_h3_source_prompt": "Original planner prompt.",
                    "_director_h3_prompt_mode": "t2va",
                    "_director_duration_sec": native_duration,
                    "_director_authored_duration_sec": 10.12,
                    "_director_prompt_user_edited": False,
                    "_director_subjects_on_screen": [],
                    "_director_dialogue_beats": [],
                }],
                "_planned_clips": [{
                    "duration_frames": 158,
                    "duration_sec": native_duration,
                }],
                "clip_images": [],
                "output_files": [],
            }
            pipelines = {pid: record}
            fake_wgp = SimpleNamespace(
                get_model_def=lambda _model: {},
            )
            with patch.object(pipeline, "_pipelines", pipelines), \
                    patch.object(pipeline, "_wgp", fake_wgp), \
                    patch.object(pipeline, "_validate_director_models"), \
                    patch.object(pipeline, "_saved_pipeline_shot_image_policy", return_value="prompt_only"), \
                    patch.object(pipeline, "_saved_director_video_execution_profile", return_value={}), \
                    patch.object(pipeline, "_validate_saved_profile_for_current_hardware"), \
                    patch.object(pipeline, "validate_director_execution_frames"), \
                    patch.object(pipeline, "_repair_saved_h3_frame_lattice", return_value=False), \
                    patch.object(pipeline, "_start_pipeline_worker"):
                self.assertTrue(pipeline._save_pipeline_state(pid))
                self.assertTrue(pipeline.update_clip_prompt(
                    out_dir,
                    pid,
                    0,
                    {"video_prompt": source},
                ))
                saved = pipeline.load_pipeline_state(out_dir, pid)
                self.assertTrue(saved["clips"][0]["_director_prompt_user_edited"])
                self.assertEqual(
                    saved["clips"][0]["_director_authored_duration_sec"],
                    10.12,
                )

                ok, message = pipeline._resume_pipeline_reserved(pid, out_dir)
                self.assertTrue(ok, message)
                resumed_plan = pipelines[pid]["clip_plans"][0]

            self.assertTrue(resumed_plan["_director_prompt_user_edited"])
            self.assertEqual(
                resumed_plan["_director_authored_duration_sec"], 10.12
            )
            with self.assertRaisesRegex(
                H3DialogueContractError,
                r"Shot 1: authored visual timing is invalid: visual timecode "
                r"00:06\.800 exceeds local clip duration 00:06\.583",
            ):
                compile_h3_clip_plans([resumed_plan])


if __name__ == "__main__":
    unittest.main()
