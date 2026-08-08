"""Model-free regression tests for Director model compatibility."""

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
_ROOT_DIR = os.path.abspath(os.path.join(_HERE, ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director_model_compat import assess_director_model  # noqa: E402
from services.director.policies import build_character_rules_block  # noqa: E402
from services.director.prompt_polish import (  # noqa: E402
    polish_prompts_third_pass,
    should_polish_director_video_prompts,
)
from services.director.planners.short_film import (  # noqa: E402
    ShortFilmPlanner,
    _route_video_pass2_guide,
    _shot_list_schema,
)
from services.director.planners.music_video import (  # noqa: E402
    MusicVideoPlanner,
    _music_shot_schema,
)
from services.director_video_strategy import (  # noqa: E402
    SHOT_IMAGE_GENERATE,
    SHOT_IMAGE_PROMPT_ONLY,
    SHOT_IMAGES_DIRECT_REFERENCES,
    adapt_bounded_timeline,
    apply_independent_shot_context,
    build_director_video_execution_profile,
    resolve_shot_image_policy,
    validate_director_execution_frames,
)


def _image_editor(**updates):
    model_def = {
        "name": "Reference editor",
        "image_outputs": True,
        "image_ref_choices": {
            "choices": [("None", ""), ("Main plus references", "KI")],
        },
    }
    model_def.update(updates)
    return model_def


def _ltx_video(**updates):
    model_def = {
        "name": "LTX",
        "image_prompt_types_allowed": "TSEV",
        "sliding_window": True,
        "any_audio_prompt": True,
        "returns_audio": True,
        "audio_guide_window_slicing": True,
        "custom_frames_injection": True,
        "auto_null_audio": True,
    }
    model_def.update(updates)
    return model_def


class TestDirectorModelAssessment(unittest.TestCase):
    def test_video_only_planner_schemas_forbid_unused_image_fields(self):
        short_schema = _shot_list_schema(
            1,
            1,
            required=["video_prompt", "image_prompt", "keyframe_prompts"],
            include_image_fields=False,
        )
        music_schema = _music_shot_schema(1, include_image_fields=False)

        for schema in (short_schema, music_schema):
            properties = schema["items"]["properties"]
            required = schema["items"]["required"]
            self.assertIn("video_prompt", properties)
            for field in (
                "image_source",
                "image_prompt",
                "visual_changes",
                "keyframe_prompts",
            ):
                self.assertNotIn(field, properties)
                self.assertNotIn(field, required)

    def test_h3_music_planner_does_not_request_or_keep_image_prompts(self):
        captured = {}

        def generate(**kwargs):
            captured.update(kwargs)
            return json.dumps([{
                "scene_goal": "Introduce the performer",
                "scene_type": "performance",
                "subjects_on_screen": [{
                    "visual_description": "Dwight from The Office in a mustard shirt",
                    "position_or_relation": "center frame",
                }],
                "environment": "A quiet conference room",
                "visual_style": "cinematic workplace comedy",
                "lighting": "soft office fluorescents",
                "mood": "dryly energetic",
                "action_beats": ["He performs directly to camera"],
                "camera_plan": {
                    "framing": "medium shot",
                    "movement": "slow push in",
                    "movement_intensity": "subtle",
                },
                "ending_beat": "He holds an awkward pose",
                "video_prompt": "Dwight from The Office performs in the conference room.",
                "window_prompts": [],
            }])

        planner = MusicVideoPlanner(
            llm_generate=generate,
            llm_generate_streaming=generate,
        )
        plan = planner.plan(
            clips=[{"start": 0.0, "end": 5.2, "label": "verse"}],
            scene_description="A workplace performance.",
            video_model="minimax_h3",
            shot_image_policy=SHOT_IMAGE_PROMPT_ONLY,
        )

        self.assertNotIn('"image_prompt":', captured["system_prompt"])
        self.assertNotIn('"image_source":', captured["system_prompt"])
        self.assertIsNone(plan.shots[0].image_prompt)
        self.assertIsNone(plan.shots[0].image_source)
        self.assertFalse(plan.shots[0].keyframe_prompts)

    def test_pipeline_renders_video_only_for_prompt_only_policy(self):
        class FakePlan:
            shots = []

            @staticmethod
            def to_dict():
                return {"skill_type": "music_video", "shots": []}

        class FakeDirector:
            def __init__(self):
                self.render_prompt_type = None

            @staticmethod
            def plan(skill_type, **kwargs):
                self.assertEqual(skill_type, "music_video")
                self.assertEqual(
                    kwargs["shot_image_policy"],
                    SHOT_IMAGE_PROMPT_ONLY,
                )
                return FakePlan()

            def render_plan(self, plan, prompt_type, **kwargs):
                self.render_prompt_type = prompt_type
                return [{"video_prompt": "A complete H3 shot."}]

            @staticmethod
            def plan_to_clip_plans(rendered):
                return [{
                    "video_prompt": rendered[0]["video_prompt"],
                    "image_prompt": rendered[0].get("image_prompt", ""),
                }]

        fake_director = FakeDirector()
        pid = "video-only-plan-test"
        previous_wgp = pipeline._wgp
        pipeline._pipelines[pid] = {"status": "running"}
        pipeline._wgp = SimpleNamespace(server_config={"services": {}})
        try:
            with patch(
                "services.director.orchestrator.DirectorOrchestrator",
                return_value=fake_director,
            ):
                clip_plans, _ = pipeline._run_planning_v2(
                    pid,
                    {
                        "_director_shot_image_policy": SHOT_IMAGE_PROMPT_ONLY,
                        "video_model": "minimax_h3",
                        "scene_description": "A direct H3 scene.",
                        "planned_clips": [{
                            "start": 0.0,
                            "end": 5.2,
                            "duration_sec": 5.2,
                        }],
                    },
                    "music_video",
                )
        finally:
            pipeline._wgp = previous_wgp
            pipeline._pipelines.pop(pid, None)

        self.assertEqual(fake_director.render_prompt_type, "video")
        self.assertEqual(clip_plans[0]["image_prompt"], "")

    def test_story_pipeline_passes_h3_native_limits_and_world_context(self):
        captured = {}
        fake_shot = SimpleNamespace(
            duration_sec=10.0,
            metadata={
                "duration_frames": 243,
                "continuity_group": "friends_kitchen",
                "closing_blocking": (
                    "Joey sits screen-left at the kitchen table"
                ),
            },
            continuity_strategy="independent",
            narrative_role="setup",
            scene_type="dialogue",
            environment="Monica's NYC apartment kitchen from Friends",
            spatial_setup=(
                "Joey stands screen-left beside the kitchen table"
            ),
            ending_beat="Joey sits at the kitchen table",
            subjects_on_screen=[
                SimpleNamespace(
                    speaker_name="Joey from Friends",
                    visual_description="Joey holding toast",
                    wardrobe=(
                        "navy overshirt, white T-shirt, blue jeans, sneakers"
                    ),
                    position_or_relation=(
                        "screen-left foreground, standing beside the table"
                    ),
                )
            ],
        )

        class FakePlan:
            shots = [fake_shot]

            @staticmethod
            def to_dict():
                return {"skill_type": "short_film", "shots": [{}]}

        class FakeDirector:
            @staticmethod
            def plan(skill_type, **kwargs):
                captured.update(kwargs)
                return FakePlan()

            @staticmethod
            def render_plan(plan, prompt_type, **kwargs):
                return [{
                    "video_prompt": (
                        "integrated_multimodal_description: [Shot 1] Joey reacts."
                    )
                }]

            @staticmethod
            def plan_to_clip_plans(rendered):
                return [{
                    "video_prompt": rendered[0]["video_prompt"],
                    "image_prompt": "",
                }]

        pid = "h3-native-planning-test"
        previous_wgp = pipeline._wgp
        pipeline._pipelines[pid] = {"status": "running"}
        pipeline._wgp = SimpleNamespace(
            server_config={"services": {}},
            get_model_def=lambda model: {
                "fps": 24,
                "frames_minimum": 124,
                "frames_maximum": 345,
                "frames_steps": 17,
                "director_video_strategy": "bounded_start_end",
                "director_shot_image_support": "optional",
                "resolutions": [("720p", "1280x704")],
                "director_memory_policy": {
                    "resolution_bands": [{
                        "min_pixels": 0,
                        "vram_tiers": [
                            {"max_vram_gb": 24, "frames": 243},
                            {"frames": 345},
                        ],
                    }],
                },
            },
        )
        try:
            with patch(
                "services.director.orchestrator.DirectorOrchestrator",
                return_value=FakeDirector(),
            ):
                plans, clips = pipeline._run_planning_v2(
                    pid,
                    {
                        "video_model": "minimax_h3",
                        "scene_description": (
                            "A Friends TV show episode in Monica's NYC apartment."
                        ),
                        "target_duration": 10,
                        "_director_shot_image_policy": SHOT_IMAGE_PROMPT_ONLY,
                        "_director_video_execution_profile": (
                            build_director_video_execution_profile(
                                "minimax_h3",
                                pipeline._wgp.get_model_def("minimax_h3"),
                                {"resolution": "1280x704"},
                                {"gpu_vram_gb": 24},
                            )
                        ),
                    },
                    "short_film_story",
                )
        finally:
            pipeline._wgp = previous_wgp
            pipeline._pipelines.pop(pid, None)

        self.assertEqual(captured["fps"], 24)
        self.assertEqual(captured["frames_minimum"], 124)
        self.assertEqual(captured["frames_maximum"], 243)
        self.assertEqual(captured["frames_steps"], 17)
        self.assertEqual(clips[0]["duration_frames"], 243)
        self.assertIn("Friends TV show", plans[0]["video_prompt"])
        self.assertIn("Monica's NYC apartment", plans[0]["video_prompt"])
        self.assertIn("navy overshirt", plans[0]["video_prompt"])
        self.assertIn("screen-left foreground", plans[0]["video_prompt"])
        self.assertIn("FINAL BLOCKING", plans[0]["video_prompt"])
        self.assertEqual(
            plans[0]["_director_continuity_group"], "friends_kitchen"
        )
        self.assertEqual(
            clips[0]["_director_closing_blocking"],
            "Joey sits screen-left at the kitchen table",
        )

    def test_h3_story_planner_omits_image_prompt_work(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            if "acclaimed screenwriter" in kwargs["system_prompt"]:
                return (
                    "INT. OFFICE - DAY\n\nDwight from The Office crosses the "
                    "conference room and begins a dry presentation.\n\n"
                    "<center>DWIGHT</center>\n"
                    "> Identity theft is not a joke, Jim.\n\n"
                    "Jim answers with a silent look, and the coworkers settle "
                    "into an awkward silence."
                )
            shots = []
            for index in range(4):
                seated = index > 0
                shots.append({
                    "title": f"Office beat {index + 1}",
                    "duration_sec": 10,
                    "scene_goal": "Advance the workplace exchange",
                    "narrative_role": "setup" if index == 0 else "resolution",
                    "scene_type": "dialogue",
                    "subjects_on_screen": [{
                        "visual_description": "Dwight from The Office in a mustard shirt",
                        "character_id": "char_0",
                        "speaker_name": "Dwight from The Office",
                        "position_or_relation": (
                            "screen-right midground, seated at the conference table"
                            if seated
                            else "screen-left foreground, standing beside the table"
                        ),
                        "wardrobe": (
                            "mustard-yellow cotton shirt, brown tie, dark slacks, "
                            "black belt, and black dress shoes"
                        ),
                    }],
                    "spatial_setup": (
                        "Dwight is seated screen-right at the conference table"
                        if seated
                        else "Dwight stands screen-left beside the conference table"
                    ),
                    "closing_blocking": (
                        "Dwight remains seated screen-right at the conference table"
                        if seated
                        else "Dwight remains standing screen-left beside the table"
                    ),
                    "continuity_strategy": (
                        "independent" if index < 2 else "continuous"
                    ),
                    "continuity_group": "dunder_mifflin_conference_room",
                    "environment": (
                        "The Dunder Mifflin conference room from The Office"
                    ),
                    "visual_style": "cinematic workplace comedy",
                    "lighting": "soft overhead office light",
                    "mood": "awkward and dry",
                    "action_beats": ["Dwight addresses the room"],
                    "dialogue_beats": ([{
                        "speaker_id": "char_0",
                        "spoken_text": "Identity theft is not a joke, Jim.",
                        "delivery": "stern and emphatic",
                        "physical_cue": "Dwight points sharply across the table.",
                        "priority": "high",
                    }] if index == 0 else []),
                    "camera_plan": {
                        "framing": "medium shot",
                        "movement": "slow push in",
                        "movement_intensity": "subtle",
                    },
                    "audio_plan": {
                        "mode": (
                            "dialogue_driven" if index == 0 else "ambient_only"
                        ),
                        "ambience": "quiet office room tone",
                        "effects": [],
                        "timing_anchor": "video",
                        "lip_sync_critical": False,
                    },
                    "ending_beat": "The room falls quiet",
                    "video_prompt": (
                        "integrated_multimodal_description: [Shot 1] Dwight "
                        "from The Office delivers a dry presentation inside "
                        "the Dunder Mifflin conference room from The Office. "
                        "overall_soundscape: Quiet office room tone. "
                        "non_diegetic_music: N/A."
                    ),
                    "multishot": False,
                    "window_prompts": [],
                })
            return json.dumps(shots)

        planner = ShortFilmPlanner(
            llm_generate=generate,
            llm_generate_streaming=generate,
        )
        plan = planner.plan(
            story_description=(
                "An episode of The Office set in the Dunder Mifflin "
                "conference room. Dwight presents an awkward idea."
            ),
            target_duration=40,
            target_scenes=2,
            video_model="minimax_h3",
            shot_image_policy=SHOT_IMAGE_PROMPT_ONLY,
            fps=24,
            frames_steps=17,
            frames_minimum=124,
            frames_maximum=345,
        )

        pass2 = next(
            call for call in calls
            if "breaking a screenplay into shots" in call["system_prompt"]
        )
        self.assertNotIn('"image_prompt":', pass2["system_prompt"])
        self.assertNotIn('"image_source":', pass2["system_prompt"])
        self.assertNotIn("keyframe_prompts", pass2["json_schema"]["items"]["properties"])
        self.assertIn("5.17-14.38 seconds", pass2["system_prompt"])
        subject_schema = pass2["json_schema"]["items"]["properties"][
            "subjects_on_screen"
        ]["items"]
        self.assertIn("wardrobe", subject_schema["required"])
        self.assertIn("position_or_relation", subject_schema["required"])
        self.assertIn("closing_blocking", pass2["json_schema"]["items"]["required"])
        self.assertIn("continuity_group", pass2["json_schema"]["items"]["required"])
        self.assertNotIn("SHOT DURATION MUST BE ONE OF: 20, 40", pass2["prompt"])
        self.assertEqual(
            pass2["json_schema"]["items"]["properties"]["window_prompts"]["maxItems"],
            0,
        )
        self.assertTrue(plan.shots)
        self.assertTrue(all(shot.image_prompt is None for shot in plan.shots))
        self.assertTrue(all(5.16 <= shot.duration_sec <= 14.38 for shot in plan.shots))
        self.assertTrue(all(not shot.window_prompts for shot in plan.shots))
        self.assertTrue(all(shot.subjects_on_screen[0].wardrobe for shot in plan.shots))
        self.assertIn("OPENING CONTINUITY", plan.shots[0].video_prompt)
        self.assertIn("FINAL BLOCKING", plan.shots[0].video_prompt)
        self.assertIn("naturally move", plan.shots[0].video_prompt)
        self.assertIn("seated screen-right", plan.shots[0].video_prompt)
        self.assertIn("naturally move", plan.shots[0].ending_beat)
        self.assertEqual(plan.shots[1].continuity_strategy, "continuous")
        self.assertIn(
            "<d>[English] Identity theft is not a joke, Jim.</d>",
            plan.shots[0].video_prompt,
        )
        self.assertIn(
            "Only these explicitly tagged lines are spoken",
            plan.shots[0].video_prompt,
        )
        self.assertIn(
            "SILENCE AND VOCAL PERFORMANCE",
            plan.shots[1].video_prompt,
        )
        self.assertIn("gibberish", plan.shots[1].video_prompt)
        self.assertEqual(
            plan.shots[0].metadata["continuity_group"],
            "dunder_mifflin_conference_room",
        )

    def test_h3_context_anchor_survives_every_independent_shot(self):
        shots = [
            SimpleNamespace(
                environment="Monica's NYC apartment kitchen from Friends",
                spatial_setup="Joey stands screen-left beside the table",
                subjects_on_screen=[
                    SimpleNamespace(
                        speaker_name="Joey from Friends",
                        visual_description="Joey holding toast",
                        wardrobe="navy overshirt, white T-shirt, blue jeans",
                        position_or_relation="screen-left foreground, standing",
                    )
                ],
            ),
            SimpleNamespace(
                environment="Monica's NYC apartment kitchen from Friends",
                spatial_setup="Ross sits screen-right at the counter",
                subjects_on_screen=[
                    SimpleNamespace(
                        speaker_name="Ross from Friends",
                        visual_description="Ross holding a coffee mug",
                        wardrobe="brown sport coat, blue shirt, dark trousers",
                        position_or_relation="screen-right midground, seated",
                    )
                ],
            ),
        ]
        plans = [
            {"video_prompt": "integrated_multimodal_description: [Shot 1] Joey reacts."},
            {"video_prompt": "integrated_multimodal_description: [Shot 1] Ross answers."},
        ]
        concept = (
            "A Friends TV show episode in Monica's NYC apartment kitchen."
        )

        apply_independent_shot_context(
            plans,
            scene_description=concept,
            shots=shots,
        )
        apply_independent_shot_context(plans)

        for plan in plans:
            self.assertIn("Friends TV show", plan["video_prompt"])
            self.assertIn("Monica's NYC apartment kitchen", plan["video_prompt"])
            self.assertEqual(plan["video_prompt"].count("PROJECT CONTINUITY"), 1)
            self.assertIn("wardrobe:", plan["video_prompt"])
            self.assertIn("opening position:", plan["video_prompt"])

    def test_h3_fallback_split_is_unique_and_has_no_rolling_window_commands(self):
        anchor = (
            "PROJECT CONTINUITY (visual world only; use only the action and "
            "dialogue specified for this shot): project world/franchise: "
            "Friends. this shot's physical setting: Monica's apartment."
        )
        plans = [{
            "video_prompt": "",
            "image_prompt": "",
            "_director_context_anchor": anchor,
            "window_prompts": [
                "Window 1 (0-20s): Joey raises toast. Monica freezes. "
                "Joey says <d>[English] What is that smell?</d>.",
                "Window 2 (20-40s): Ross sets down his mug. Ross blushes. "
                "Monica points at Ross.",
            ],
        }]
        clips = [{"start": 0.0, "end": 40.0, "duration_sec": 40.0}]

        adapted, _ = adapt_bounded_timeline(
            plans,
            clips,
            fps=24,
            minimum_frames=124,
            maximum_frames=345,
            frame_step=17,
        )

        self.assertEqual(len(adapted), 3)
        joined = " ".join(item["video_prompt"] for item in adapted)
        self.assertNotIn("Window 1", joined)
        self.assertNotIn("Window 2", joined)
        self.assertNotIn("Continue directly", joined)
        self.assertNotIn("Begin this portion", joined)
        for unique_action in (
            "Joey raises toast",
            "Monica freezes",
            "Ross sets down his mug",
            "Ross blushes",
            "Monica points at Ross",
        ):
            self.assertEqual(joined.count(unique_action), 1)
        self.assertTrue(
            all(item["video_prompt"].count("PROJECT CONTINUITY") == 1 for item in adapted)
        )

    def test_h3_planner_uses_self_contained_h3_guide(self):
        guide = _route_video_pass2_guide("minimax_h3_ref2va")

        self.assertIn("MiniMax H3", guide)
        self.assertIn("Every video prompt therefore stands on its own", guide)
        self.assertIn("integrated_multimodal_description", guide)
        self.assertNotIn("for the LTX-2 video model", guide)

    def test_h3_direct_rules_preserve_known_character_names(self):
        rules = build_character_rules_block(False, preserve_names=True)

        self.assertIn("Preserve every user-supplied proper", rules)
        self.assertIn("series, film, or franchise", rules)
        self.assertNotIn("Describe characters by appearance, not names", rules)

    def test_third_pass_preserves_h3_name_but_legacy_path_uses_descriptor(self):
        characters = [{
            "id": "char_1",
            "display_name": "Dwight from The Office",
            "physical_description": "mustard-shirted salesman with glasses",
        }]

        def make_plan():
            return [{
                "video_prompt": (
                    "Dwight from The Office enters the conference room."
                ),
                "image_prompt": "",
                "subjects_on_screen": [{
                    "character_id": "char_1",
                    "visual_description": (
                        "mustard-shirted salesman with glasses"
                    ),
                }],
            }]

        with patch(
            "services.llm_service.enhance_prompt",
            side_effect=lambda **kwargs: kwargs["prompt"],
        ):
            h3_result = polish_prompts_third_pass(
                make_plan(),
                "minimax_h3",
                "",
                characters=characters,
                preserve_video_character_names=True,
            )
            legacy_result = polish_prompts_third_pass(
                make_plan(),
                "ltx2_22B_distilled_1_1",
                "",
                characters=characters,
            )

        self.assertIn(
            "Dwight from The Office",
            h3_result[0]["video_prompt"],
        )
        self.assertNotIn(
            "Dwight from The Office",
            legacy_result[0]["video_prompt"],
        )
        self.assertIn(
            "mustard-shirted salesman with glasses",
            legacy_result[0]["video_prompt"],
        )

    def test_h3_prompt_only_polish_skips_all_creative_llm_calls(self):
        plan = [{
            "video_prompt": "A named performer crosses a quiet office.",
            "image_prompt": "A still office composition.",
            "keyframe_prompts": ["A later still composition."],
        }]

        with patch(
            "services.llm_service.enhance_prompt",
            side_effect=lambda **kwargs: kwargs["prompt"],
        ) as enhance:
            result = polish_prompts_third_pass(
                plan,
                "minimax_h3",
                "unused_image_model",
                preserve_video_character_names=True,
                polish_image_prompts=False,
            )

        self.assertEqual(enhance.call_count, 0)
        self.assertEqual(
            result[0]["video_prompt"],
            "A named performer crosses a quiet office.",
        )
        self.assertEqual(
            result[0]["image_prompt"],
            "A still office composition.",
        )
        self.assertEqual(
            result[0]["keyframe_prompts"],
            ["A later still composition."],
        )

    def test_third_pass_video_routing_is_model_aware(self):
        self.assertFalse(should_polish_director_video_prompts("minimax_h3"))
        self.assertFalse(
            should_polish_director_video_prompts("minimax_h3_full")
        )
        self.assertFalse(
            should_polish_director_video_prompts("minimax_h3_ref2va")
        )
        self.assertTrue(
            should_polish_director_video_prompts("ltx2_22B_distilled_1_1")
        )

    def test_h3_polish_cannot_drop_dialogue_or_silence_contracts(self):
        dialogue_prompt = (
            "integrated_multimodal_description: Dwight turns toward Jim. "
            "DIALOGUE AND VOCAL PERFORMANCE: (S1) Dwight speaks sternly: "
            "<d>[English] Identity theft is not a joke, Jim.</d>. Only these "
            "explicitly tagged lines are spoken. overall_soundscape: Quiet office. "
            "non_diegetic_music: N/A."
        )
        silence_prompt = (
            "integrated_multimodal_description: Jim looks into camera. "
            "SILENCE AND VOCAL PERFORMANCE: No one speaks in this shot. "
            "Generate no muttering, murmuring, gibberish, invented words, or "
            "speech-like vocalizations. overall_soundscape: Quiet office. "
            "non_diegetic_music: N/A."
        )
        plans = [
            {"video_prompt": dialogue_prompt, "image_prompt": ""},
            {"video_prompt": silence_prompt, "image_prompt": ""},
        ]

        with patch(
            "services.llm_service.enhance_prompt",
            return_value=(
                "integrated_multimodal_description: A polished office reaction. "
                "overall_soundscape: Quiet office. non_diegetic_music: N/A."
            ),
        ):
            result = polish_prompts_third_pass(
                plans,
                "minimax_h3",
                "",
                preserve_video_character_names=True,
                polish_video_prompts=True,
                polish_image_prompts=False,
            )

        self.assertEqual(result[0]["video_prompt"], dialogue_prompt)
        self.assertEqual(result[1]["video_prompt"], silence_prompt)

    def test_h3_generated_image_polish_does_not_rewrite_video_prompt(self):
        plan = [{
            "video_prompt": "Native H3 Context-IR video prompt.",
            "image_prompt": "A still opening composition.",
            "keyframe_prompts": ["A still closing composition."],
        }]

        def enhance(**kwargs):
            return f"{kwargs['prompt']} Refined."

        with patch(
            "services.llm_service.enhance_prompt",
            side_effect=enhance,
        ) as enhance_mock:
            result = polish_prompts_third_pass(
                plan,
                "minimax_h3",
                "flux2_klein_9B",
                polish_image_prompts=True,
            )

        self.assertEqual(enhance_mock.call_count, 2)
        self.assertEqual(
            [call.kwargs["mode"] for call in enhance_mock.call_args_list],
            ["image", "image"],
        )
        self.assertEqual(
            result[0]["video_prompt"],
            "Native H3 Context-IR video prompt.",
        )
        self.assertIn("Refined", result[0]["image_prompt"])
        self.assertIn("Refined", result[0]["keyframe_prompts"][0])

    def test_h3_shot_image_auto_policy_is_model_aware(self):
        fl2va = {"director_shot_image_support": "optional"}
        ref2va = {"director_shot_image_support": "direct_references"}

        self.assertEqual(
            resolve_shot_image_policy(
                fl2va, "auto", has_visual_references=False,
            ),
            SHOT_IMAGE_PROMPT_ONLY,
        )
        self.assertEqual(
            resolve_shot_image_policy(
                fl2va, "auto", has_visual_references=True,
            ),
            SHOT_IMAGE_GENERATE,
        )
        self.assertEqual(
            resolve_shot_image_policy(
                ref2va, "auto", has_visual_references=True,
            ),
            SHOT_IMAGES_DIRECT_REFERENCES,
        )
        self.assertEqual(
            resolve_shot_image_policy(
                ref2va, "generate", has_visual_references=False,
            ),
            SHOT_IMAGE_GENERATE,
        )

    def test_reference_editor_can_bootstrap_and_edit(self):
        result = assess_director_model("editor", _image_editor())
        self.assertTrue(result["image"]["compatible"])

    def test_plain_image_model_is_not_a_director_image_model(self):
        result = assess_director_model(
            "plain",
            {"name": "Plain", "image_outputs": True},
        )
        self.assertFalse(result["image"]["compatible"])
        self.assertIn("reference editing", result["image"]["reason"])

    def test_editor_that_cannot_bootstrap_is_rejected(self):
        result = assess_director_model(
            "edit-only",
            _image_editor(image_ref_choices={"choices": [("Edit", "KI")]}),
        )
        self.assertFalse(result["image"]["compatible"])
        self.assertIn("plain generation", result["image"]["reason"])

    def test_image_reference_limit_is_exposed(self):
        result = assess_director_model(
            "krea-edit",
            _image_editor(max_image_refs=2),
        )
        self.assertEqual(result["max_image_refs"], 2)

    def test_ltx_supports_all_director_workflows(self):
        result = assess_director_model(
            "ltx-custom",
            _ltx_video(),
            architecture="ltx2_22B",
        )
        for workflow in (
            "music_video",
            "short_film_audio",
            "short_film_story",
            "seamless",
        ):
            self.assertTrue(result["video"][workflow]["compatible"], workflow)
        self.assertTrue(result["supports_voice_reference"])

    def test_native_audio_output_is_not_audio_input(self):
        result = assess_director_model(
            "ovi",
            {
                "image_prompt_types_allowed": "TSVL",
                "sliding_window": True,
                "returns_audio": True,
            },
        )
        self.assertFalse(result["video"]["music_video"]["compatible"])
        self.assertTrue(result["video"]["short_film_story"]["compatible"])
        self.assertFalse(result["supports_audio_input"])
        self.assertTrue(result["generates_audio"])

    def test_fixed_length_model_is_rejected(self):
        result = assess_director_model(
            "fixed",
            {
                "image_prompt_types_allowed": "TSE",
                "sliding_window": False,
            },
        )
        self.assertFalse(result["video"]["short_film_story"]["compatible"])
        self.assertIn("sliding-window", result["video"]["short_film_story"]["reason"])

    def test_required_control_video_model_is_rejected(self):
        result = assess_director_model(
            "pose",
            {
                "image_prompt_types_allowed": "SVL",
                "sliding_window": True,
                "guide_custom_choices": {
                    "choices": [("Use pose video", "V"), ("Use pose and mask", "VA")],
                },
            },
        )
        self.assertFalse(result["video"]["short_film_story"]["compatible"])
        self.assertIn("control-video", result["video"]["short_film_story"]["reason"])

    def test_audio_avatar_without_silent_mode_is_rejected(self):
        result = assess_director_model(
            "avatar",
            {
                "image_prompt_types_allowed": "TSVL",
                "sliding_window": True,
                "audio_guidance": True,
                "any_audio_prompt": True,
            },
        )
        self.assertFalse(result["video"]["short_film_story"]["compatible"])
        self.assertIn("external speech", result["video"]["short_film_story"]["reason"])

    def test_seamless_requires_frame_injection(self):
        result = assess_director_model(
            "ordinary-i2v",
            {
                "image_prompt_types_allowed": "SEVL",
                "sliding_window": True,
            },
        )
        self.assertFalse(result["video"]["short_film_story"]["compatible"])
        self.assertIn("synchronized dialogue", result["video"]["short_film_story"]["reason"])
        self.assertFalse(result["video"]["seamless"]["compatible"])

    def test_bounded_h3_strategies_are_routed_by_workflow(self):
        fl2va = {
            "name": "H3 FL2VA",
            "image_prompt_types_allowed": "TSE",
            "sliding_window": False,
            "returns_audio": True,
            "director_video_strategy": "bounded_start_end",
            "director_audio_input_mode": "none",
            "director_reference_mode": "start_end",
            "director_shot_image_support": "optional",
            "director_endpoint_continuity": True,
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
        }
        omni = {
            "name": "H3 Ref2VA",
            "image_prompt_types_allowed": "",
            "sliding_window": False,
            "returns_audio": True,
            "omni_reference": True,
            "director_video_strategy": "omni_reference",
            "director_audio_input_mode": "reference_manifest",
            "director_reference_mode": "omni_manifest",
            "director_shot_image_support": "direct_references",
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
        }

        fl2va_result = assess_director_model(
            "minimax_h3", fl2va, architecture="minimax_h3"
        )
        self.assertTrue(fl2va_result["video"]["short_film_story"]["compatible"])
        self.assertFalse(fl2va_result["video"]["music_video"]["compatible"])
        self.assertFalse(fl2va_result["video"]["short_film_audio"]["compatible"])
        self.assertFalse(fl2va_result["video"]["seamless"]["compatible"])
        self.assertTrue(fl2va_result["supports_endpoint_continuity"])
        self.assertEqual(fl2va_result["shot_image_support"], "optional")

        omni_result = assess_director_model(
            "minimax_h3_ref2va", omni, architecture="minimax_h3_ref2va"
        )
        self.assertTrue(omni_result["video"]["music_video"]["compatible"])
        self.assertTrue(omni_result["video"]["short_film_audio"]["compatible"])
        self.assertTrue(omni_result["video"]["short_film_story"]["compatible"])
        self.assertFalse(omni_result["video"]["seamless"]["compatible"])
        self.assertTrue(omni_result["supports_voice_reference"])
        self.assertEqual(
            omni_result["shot_image_support"],
            "direct_references",
        )

    def test_h3_timeline_merges_short_cuts_and_splits_long_scenes(self):
        plans = [
            {"video_prompt": "A", "image_prompt": "image A"},
            {"video_prompt": "B", "image_prompt": "image B"},
            {
                "video_prompt": "Long scene",
                "image_prompt": "long image",
                "window_prompts": ["first movement", "second movement"],
            },
        ]
        clips = [
            {"start": 0.0, "end": 3.0, "duration_sec": 3.0, "label": "a"},
            {"start": 3.0, "end": 7.0, "duration_sec": 4.0, "label": "b"},
            {"start": 7.0, "end": 47.0, "duration_sec": 40.0, "label": "long"},
        ]

        adapted_plans, adapted_clips = adapt_bounded_timeline(
            plans,
            clips,
            fps=24,
            minimum_frames=124,
            maximum_frames=345,
            frame_step=17,
        )

        self.assertEqual(len(adapted_plans), 4)
        self.assertIn("A", adapted_plans[0]["video_prompt"])
        self.assertIn("B", adapted_plans[0]["video_prompt"])
        self.assertEqual(
            adapted_plans[0]["_director_source_clip_indices"], [0, 1]
        )
        for plan, clip in zip(adapted_plans, adapted_clips):
            frames = clip["duration_frames"]
            self.assertGreaterEqual(frames, 124)
            self.assertLessEqual(frames, 345)
            self.assertEqual((frames - 124) % 17, 0)
            self.assertEqual(plan["_director_generation_frames"], frames)
        self.assertAlmostEqual(
            adapted_clips[-1]["end"],
            sum(clip["duration_frames"] for clip in adapted_clips) / 24,
            places=6,
        )


class TestDirectorVideoExecutionProfile(unittest.TestCase):
    @staticmethod
    def _h3_model(*, full: bool = False) -> dict:
        memory_policy = {
            "resolution_bands": [
                {
                    "min_pixels": 1_800_000,
                    "vram_tiers": [
                        {
                            "max_vram_gb": 12,
                            "frames": None,
                            "fallback_resolution": "720p or lower",
                        },
                        {"max_vram_gb": 24, "frames": 124},
                        {"max_vram_gb": 32, "frames": 243},
                        {"frames": 345},
                    ],
                },
                {
                    "min_pixels": 800_000,
                    "vram_tiers": [
                        {"max_vram_gb": 12, "frames": 124},
                        {"max_vram_gb": 24, "frames": 243},
                        {"frames": 345},
                    ],
                },
                {
                    "min_pixels": 0,
                    "vram_tiers": [{"frames": 345}],
                },
            ],
        }
        return {
            "architecture": "minimax_h3",
            "director_video_strategy": "bounded_start_end",
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
            "fps": 24,
            "minimax_h3_full_checkpoint": full,
            "director_memory_policy": memory_policy,
            "resolutions": [
                ("1280x704 (16:9 720p)", "1280x704"),
                ("704x1280 (9:16 720p)", "704x1280"),
                ("1920x1088 (16:9 1080p)", "1920x1088"),
                ("1088x1920 (9:16 1080p)", "1088x1920"),
            ],
        }

    def test_auto_profile_uses_exact_canvas_vram_policy(self):
        profile = build_director_video_execution_profile(
            "minimax_h3",
            self._h3_model(),
            {"resolution": "1280x720"},
            {"gpu_vram_gb": 24},
            resolution_preset="720p",
            aspect_ratio="16:9",
        )

        self.assertEqual(profile["normalized_resolution"], "1280x704")
        self.assertEqual(profile["recommended_max_frames"], 243)
        self.assertEqual(profile["effective_max_frames"], 243)
        self.assertAlmostEqual(profile["effective_max_seconds"], 10.125)
        self.assertFalse(profile["manual_override"])

        high_resolution = build_director_video_execution_profile(
            "minimax_h3",
            self._h3_model(),
            {"resolution": "1920x1088"},
            {"gpu_vram_gb": 24},
        )
        self.assertEqual(high_resolution["effective_max_frames"], 124)
        high_resolution_32gb = build_director_video_execution_profile(
            "minimax_h3",
            self._h3_model(),
            {"resolution": "1920x1088"},
            {"gpu_vram_gb": 32},
        )
        self.assertEqual(high_resolution_32gb["effective_max_frames"], 243)

    def test_unsupported_auto_profile_requires_lower_canvas_or_override(self):
        with self.assertRaisesRegex(ValueError, "no automatic one-pass profile"):
            build_director_video_execution_profile(
                "minimax_h3",
                self._h3_model(),
                {"resolution": "1920x1088"},
                {"gpu_vram_gb": 12},
            )

        profile = build_director_video_execution_profile(
            "minimax_h3",
            self._h3_model(),
            {"resolution": "1920x1088"},
            {"gpu_vram_gb": 12},
            manual_max_frames=124,
        )
        self.assertEqual(profile["effective_max_frames"], 124)
        self.assertTrue(profile["manual_override"])
        self.assertFalse(profile["hardware_supported"])

    def test_manual_profile_must_stay_on_native_lattice(self):
        with self.assertRaisesRegex(ValueError, "124-345 frame lattice"):
            build_director_video_execution_profile(
                "minimax_h3",
                self._h3_model(),
                {"resolution": "1280x704"},
                {"gpu_vram_gb": 24},
                manual_max_frames=240,
            )

        profile = build_director_video_execution_profile(
            "minimax_h3",
            self._h3_model(),
            {"resolution": "1280x704"},
            {"gpu_vram_gb": 24},
            manual_max_frames=345,
        )
        self.assertEqual(profile["effective_max_frames"], 345)
        validate_director_execution_frames(profile, 345)

    def test_validation_rejects_runtime_window_shrink(self):
        profile = build_director_video_execution_profile(
            "minimax_h3",
            self._h3_model(),
            {"resolution": "1280x704"},
            {"gpu_vram_gb": 24},
        )
        with self.assertRaisesRegex(ValueError, "one native pass"):
            validate_director_execution_frames(profile, 345)

    def test_timeline_is_split_to_effective_profile_before_generation(self):
        profile = build_director_video_execution_profile(
            "minimax_h3",
            self._h3_model(),
            {"resolution": "1280x704"},
            {"gpu_vram_gb": 24},
        )
        plans, clips = adapt_bounded_timeline(
            [{
                "video_prompt": "One uninterrupted fourteen-second action.",
                "window_prompts": ["first beat", "second beat"],
            }],
            [{"start": 0.0, "end": 14.375, "duration_sec": 14.375}],
            fps=24,
            minimum_frames=124,
            maximum_frames=profile["effective_max_frames"],
            frame_step=17,
        )

        self.assertEqual(len(plans), 2)
        self.assertEqual(len(clips), 2)
        for clip in clips:
            self.assertLessEqual(
                clip["duration_frames"],
                profile["effective_max_frames"],
            )
            validate_director_execution_frames(
                profile,
                clip["duration_frames"],
            )

    def test_saved_auto_profile_is_rechecked_on_a_smaller_gpu(self):
        model_def = self._h3_model()
        profile = build_director_video_execution_profile(
            "minimax_h3",
            model_def,
            {"resolution": "1280x704"},
            {"gpu_vram_gb": 24},
        )
        state = {
            "video_model": "minimax_h3",
            "video_params": {"resolution": "1280x704"},
            "video_loras": {},
            "_params_snapshot": {"video_model": "minimax_h3"},
        }

        with patch.object(
            pipeline,
            "_director_hardware_snapshot",
            return_value={"gpu_vram_gb": 12},
        ):
            with self.assertRaisesRegex(ValueError, "current GPU"):
                pipeline._validate_saved_profile_for_current_hardware(
                    state,
                    profile,
                    model_def,
                    [243],
                )

    def test_director_turbo_child_keeps_one_native_pass_and_managed_recipe(self):
        model_def = self._h3_model()
        profile = build_director_video_execution_profile(
            "minimax_h3",
            model_def,
            {
                "resolution": "1280x704",
                "minimax_h3_turbo_mode": True,
            },
            {"gpu_vram_gb": 24},
        )
        params = {
            "model_type": "minimax_h3",
            "video_length": 243,
            "per_clip_frames": [243],
            "minimax_h3_turbo_mode": True,
            "_director_video_execution_profile": profile,
        }

        with patch.object(
            pipeline,
            "_wgp",
            SimpleNamespace(get_model_def=lambda _model_type: model_def),
        ):
            pipeline._prepare_director_generation_params(params)

        self.assertTrue(params["sliding_window_memory_override"])
        self.assertEqual(params["num_inference_steps"], 6)
        self.assertEqual(
            params["activated_loras"],
            ["minimax_h3_turbo_4step_ckpt500.safetensors"],
        )
        self.assertEqual(params["loras_multipliers"], "0.50")


class TestDirectorBackendValidation(unittest.TestCase):
    def setUp(self):
        self.original_wgp = pipeline._wgp
        definitions = {
            "image": _image_editor(),
            "ltx": _ltx_video(),
            "ovi": {
                "name": "Ovi",
                "image_prompt_types_allowed": "TSVL",
                "sliding_window": True,
                "returns_audio": True,
            },
        }
        pipeline._wgp = SimpleNamespace(
            get_model_def=lambda model_type: definitions.get(model_type),
            get_model_family=lambda model_type, for_ui=False: "test",
            get_base_model_type=lambda model_type: "ltx2_22B" if model_type == "ltx" else model_type,
        )

    def tearDown(self):
        pipeline._wgp = self.original_wgp

    def test_audio_workflow_rejects_native_audio_only_model(self):
        with self.assertRaisesRegex(
            pipeline.DirectorModelCompatibilityError,
            "cannot use the uploaded soundtrack",
        ):
            pipeline._validate_director_models({
                "pipeline_type": "music_video",
                "image_model": "image",
                "video_model": "ovi",
            })

    def test_story_workflow_accepts_native_audio_only_model(self):
        pipeline._validate_director_models({
            "pipeline_type": "short_film_story",
            "image_model": "image",
            "video_model": "ovi",
            "seamless": False,
        })

    def test_seamless_rejects_model_without_frame_injection(self):
        with self.assertRaisesRegex(
            pipeline.DirectorModelCompatibilityError,
            "Seamless",
        ):
            pipeline._validate_director_models({
                "pipeline_type": "short_film_story",
                "image_model": "image",
                "video_model": "ovi",
                "seamless": True,
            })

    def test_reference_limit_keeps_source_first(self):
        definitions = pipeline._wgp.get_model_def
        limited = _image_editor(max_image_refs=2)
        pipeline._wgp.get_model_def = lambda model_type: (
            limited if model_type == "image" else definitions(model_type)
        )
        refs = pipeline._limit_director_image_refs(
            "image",
            ["source.png", "character.png", "location.png"],
            pid="test",
        )
        self.assertEqual(refs, ["source.png", "character.png"])

    def test_native_window_uses_selected_model_default(self):
        frames = pipeline._director_native_window_frames(
            "ovi",
            {"settings": {"sliding_window_size": 242}},
            fps=24,
            min_frames=17,
            latent_size=8,
        )
        self.assertEqual(frames, 241)

    def test_native_window_falls_back_to_five_seconds(self):
        frames = pipeline._director_native_window_frames(
            "ovi",
            {},
            fps=25,
            min_frames=17,
            latent_size=8,
        )
        self.assertEqual(frames, 121)

    def test_voice_reference_rejects_non_ltx_model(self):
        with self.assertRaisesRegex(
            pipeline.DirectorModelCompatibilityError,
            "Voice Reference",
        ):
            pipeline._validate_director_models({
                "pipeline_type": "short_film_story",
                "image_model": "image",
                "video_model": "ovi",
                "seamless": False,
                "voice_reference": "voice.wav",
            })


class TestDirectorH3GenerationContract(unittest.TestCase):
    def setUp(self):
        self.original_wgp = pipeline._wgp
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def tearDown(self):
        pipeline._wgp = self.original_wgp

    def _file(self, name: str) -> str:
        path = os.path.join(self.temp_dir.name, name)
        with open(path, "wb") as handle:
            handle.write(b"test")
        return path

    def _install_registry(self, model_type: str, model_def: dict) -> None:
        pipeline._wgp = SimpleNamespace(
            save_path=self.temp_dir.name,
            get_model_def=lambda selected: model_def if selected == model_type else None,
            get_model_family=lambda selected, for_ui=False: "minimax_h3",
            get_base_model_type=lambda selected: model_type,
            get_model_min_frames_and_step=lambda selected: (124, 17, 17),
            get_default_settings=lambda selected: {},
            get_lora_dir=lambda selected: os.path.join(self.temp_dir.name, "loras"),
        )

    def test_fl2va_uses_native_lengths_and_endpoints_only_within_one_scene(self):
        model_type = "minimax_h3"
        model_def = {
            "name": "H3 FL2VA",
            "fps": 24,
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
            "latent_size": 17,
            "image_prompt_types_allowed": "TSE",
            "returns_audio": True,
            "director_video_strategy": "bounded_start_end",
            "director_audio_input_mode": "none",
            "director_endpoint_continuity": True,
            "director_trim_end_frames": False,
        }
        self._install_registry(model_type, model_def)
        first = self._file("first.png")
        second = self._file("second.png")
        third = self._file("third.png")
        plans = [
            {"video_prompt": "one", "_director_source_clip_indices": [0]},
            {"video_prompt": "two", "_director_source_clip_indices": [0]},
            {"video_prompt": "three", "_director_source_clip_indices": [1]},
        ]
        clips = [
            {"duration_frames": 124, "_director_source_clip_indices": [0], "_director_segment_index": 0},
            {"duration_frames": 141, "_director_source_clip_indices": [0], "_director_segment_index": 1},
            {"duration_frames": 158, "_director_source_clip_indices": [1], "_director_segment_index": 0},
        ]
        captured = {}

        def submit(params, **kwargs):
            captured.update(params)
            return ["joined.mp4"]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            pipeline._run_video_generation(
                "h3-fl",
                {
                    "video_model": model_type,
                    "pipeline_type": "short_film_story",
                    "seamless": False,
                    "video_params": {},
                },
                plans,
                clips,
                [os.path.basename(first), os.path.basename(second), os.path.basename(third)],
                out_dir=self.temp_dir.name,
            )

        self.assertEqual(captured["per_clip_frames"], [124, 141, 158])
        self.assertEqual(captured["image_end"][0], second)
        self.assertEqual(captured["image_end"][1:], ["", ""])
        self.assertEqual(captured["image_prompt_type"], "SE")
        compiled_prompts = captured["prompt"].split("\n---CLIP_BOUNDARY---\n")
        self.assertTrue(compiled_prompts[0].startswith(
            "How the reference pictures align with the target video"
        ))
        self.assertTrue(compiled_prompts[1].startswith(
            "For the target video, at 0.00 seconds into the target video"
        ))
        self.assertTrue(compiled_prompts[2].startswith(
            "For the target video, at 0.00 seconds into the target video"
        ))

    def test_fl2va_prompt_only_continues_only_split_segments_of_same_scene(self):
        model_type = "minimax_h3"
        model_def = {
            "name": "H3 FL2VA",
            "fps": 24,
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
            "latent_size": 17,
            "image_prompt_types_allowed": "TSE",
            "returns_audio": True,
            "director_video_strategy": "bounded_start_end",
            "director_shot_image_support": "optional",
            "director_audio_input_mode": "none",
            "director_trim_end_frames": False,
        }
        self._install_registry(model_type, model_def)
        plans = [
            {"video_prompt": "one", "_director_source_clip_indices": [0]},
            {"video_prompt": "two", "_director_source_clip_indices": [0]},
            {"video_prompt": "three", "_director_source_clip_indices": [1]},
        ]
        clips = [
            {"duration_frames": 124, "_director_source_clip_indices": [0], "_director_segment_index": 0},
            {"duration_frames": 141, "_director_source_clip_indices": [0], "_director_segment_index": 1},
            {"duration_frames": 158, "_director_source_clip_indices": [1], "_director_segment_index": 0},
        ]
        captured = {}

        with patch.object(
            pipeline,
            "_submit_and_wait",
            side_effect=lambda params, **kwargs: captured.update(params) or ["joined.mp4"],
        ):
            pipeline._run_video_generation(
                "h3-fl-t2v",
                {
                    "video_model": model_type,
                    "pipeline_type": "short_film_story",
                    "seamless": False,
                    "video_params": {},
                    "_director_shot_image_policy": SHOT_IMAGE_PROMPT_ONLY,
                },
                plans,
                clips,
                ["", "", ""],
                out_dir=self.temp_dir.name,
            )

        self.assertNotIn("image_start", captured)
        self.assertNotIn("image_end", captured)
        self.assertEqual(captured["image_prompt_type"], "")
        self.assertEqual(
            captured["per_clip_continue_from_previous"],
            [False, True, False],
        )
        compiled_prompts = captured["prompt"].split("\n---CLIP_BOUNDARY---\n")
        self.assertTrue(compiled_prompts[0].startswith(
            "integrated_multimodal_description: [Shot 1]"
        ))
        self.assertTrue(compiled_prompts[1].startswith(
            "For the target video, at 0.00 seconds into the target video"
        ))
        self.assertTrue(compiled_prompts[2].startswith(
            "integrated_multimodal_description: [Shot 1]"
        ))

    def test_fl2va_prompt_only_uses_final_frame_only_for_explicit_extension(self):
        model_type = "minimax_h3"
        model_def = {
            "name": "H3 FL2VA",
            "fps": 24,
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
            "latent_size": 17,
            "image_prompt_types_allowed": "TSE",
            "returns_audio": True,
            "director_video_strategy": "bounded_start_end",
            "director_shot_image_support": "optional",
            "director_audio_input_mode": "none",
            "director_trim_end_frames": False,
        }
        self._install_registry(model_type, model_def)
        plans = [
            {
                "video_prompt": "wide kitchen setup",
                "_director_continuity_group": "kitchen",
                "_director_continuity_strategy": "independent",
            },
            {
                "video_prompt": "editorial close-up in the kitchen",
                "_director_continuity_group": "kitchen",
                "_director_continuity_strategy": "continuous",
            },
            {
                "video_prompt": "literal continuation of the close-up",
                "_director_continuity_group": "kitchen",
                "_director_continuity_strategy": "extend_previous",
            },
            {
                "video_prompt": "new scene in the courtyard",
                "_director_continuity_group": "courtyard",
                "_director_continuity_strategy": "extend_previous",
            },
        ]
        clips = [{"duration_frames": 124} for _ in plans]
        captured = {}

        with patch.object(
            pipeline,
            "_submit_and_wait",
            side_effect=lambda params, **kwargs: captured.update(params) or [
                "joined.mp4"
            ],
        ):
            pipeline._run_video_generation(
                "h3-explicit-continuation",
                {
                    "video_model": model_type,
                    "pipeline_type": "short_film_story",
                    "seamless": False,
                    "video_params": {},
                    "_director_shot_image_policy": SHOT_IMAGE_PROMPT_ONLY,
                },
                plans,
                clips,
                ["", "", "", ""],
                out_dir=self.temp_dir.name,
            )

        self.assertEqual(
            captured["per_clip_continue_from_previous"],
            [False, False, True, False],
        )

    def test_ref2va_direct_mode_uses_user_references_without_composition_images(self):
        model_type = "minimax_h3_ref2va"
        model_def = {
            "name": "H3 Ref2VA",
            "fps": 24,
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
            "latent_size": 17,
            "image_prompt_types_allowed": "",
            "returns_audio": True,
            "omni_reference": True,
            "director_video_strategy": "omni_reference",
            "director_shot_image_support": "direct_references",
            "director_audio_input_mode": "reference_manifest",
            "director_trim_end_frames": False,
        }
        self._install_registry(model_type, model_def)
        character = self._file("direct-character.png")
        location = self._file("direct-location.png")
        captured = {}

        with patch.object(
            pipeline,
            "_submit_and_wait",
            side_effect=lambda params, **kwargs: captured.update(params) or ["joined.mp4"],
        ):
            pipeline._run_video_generation(
                "h3-omni-direct",
                {
                    "video_model": model_type,
                    "pipeline_type": "short_film_story",
                    "seamless": False,
                    "video_params": {},
                    "character_ref_paths": [character],
                    "location_ref_paths": [location],
                    "_director_shot_image_policy": SHOT_IMAGES_DIRECT_REFERENCES,
                },
                [{"video_prompt": "The character enters the location."}],
                [{"duration_frames": 124}],
                [""],
                out_dir=self.temp_dir.name,
            )

        self.assertNotIn("image_start", captured)
        manifest = captured["per_clip_minimax_h3_references"][0]
        self.assertEqual(
            [item.get("image_intent") for item in manifest if item["type"] == "image"],
            ["identity", "scene"],
        )
        self.assertTrue(captured["prompt"].startswith("subject_definitions:"))
        self.assertIn("<Picture 1>", captured["prompt"])
        self.assertIn("<Picture 2>", captured["prompt"])
        self.assertLess(
            captured["prompt"].index("retention_analysis:"),
            captured["prompt"].index("detailed_description:"),
        )

    def test_ref2va_builds_per_shot_native_manifests_without_fixed_start_frames(self):
        model_type = "minimax_h3_ref2va"
        model_def = {
            "name": "H3 Ref2VA",
            "fps": 24,
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
            "latent_size": 17,
            "image_prompt_types_allowed": "",
            "returns_audio": True,
            "omni_reference": True,
            "director_video_strategy": "omni_reference",
            "director_audio_input_mode": "reference_manifest",
            "director_trim_end_frames": False,
        }
        self._install_registry(model_type, model_def)
        shot = self._file("shot.png")
        character = self._file("character.png")
        location = self._file("location.png")
        voice = self._file("voice.wav")
        captured = {}

        def submit(params, **kwargs):
            captured.update(params)
            return ["joined.mp4"]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            pipeline._run_video_generation(
                "h3-omni",
                {
                    "video_model": model_type,
                    "pipeline_type": "short_film_story",
                    "seamless": False,
                    "video_params": {},
                    "character_ref_paths": [character],
                    "character_ref_labels": ["Blaine"],
                    "location_ref_paths": [location],
                    "location_ref_labels": ["dojo"],
                    "voice_reference": voice,
                },
                [{"video_prompt": "Blaine crosses the dojo."}],
                [{"duration_frames": 124}],
                [os.path.basename(shot)],
                out_dir=self.temp_dir.name,
            )

        self.assertNotIn("image_start", captured)
        self.assertEqual(captured["image_prompt_type"], "")
        manifest = captured["per_clip_minimax_h3_references"][0]
        self.assertEqual(
            [item.get("image_intent") for item in manifest if item["type"] == "image"],
            ["composition", "identity", "scene"],
        )
        self.assertEqual(manifest[-1]["audio_intent"], "voice")
        self.assertIn("Blaine", manifest[-1]["role"])
        self.assertNotIn("voice_reference", captured)

    def test_ref2va_audio_workflow_maps_shot_slice_and_keeps_clean_join_track(self):
        model_type = "minimax_h3_ref2va"
        model_def = {
            "name": "H3 Ref2VA",
            "fps": 24,
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
            "latent_size": 17,
            "image_prompt_types_allowed": "",
            "returns_audio": True,
            "omni_reference": True,
            "director_video_strategy": "omni_reference",
            "director_audio_input_mode": "reference_manifest",
            "director_trim_end_frames": False,
        }
        self._install_registry(model_type, model_def)
        shot = self._file("shot.png")
        song = self._file("song.wav")
        captured = {}
        generated_slice = []

        def slice_audio(source, start, duration, destination):
            self.assertEqual(source, song)
            self.assertAlmostEqual(start, 2.0)
            self.assertAlmostEqual(duration, 124 / 24)
            with open(destination, "wb") as handle:
                handle.write(b"slice")
            generated_slice.append(destination)

        def submit(params, **kwargs):
            captured.update(params)
            manifest_path = params["per_clip_minimax_h3_references"][0][-1]["path"]
            self.assertTrue(os.path.isfile(manifest_path))
            return ["joined.mp4"]

        with (
            patch.object(pipeline, "_slice_audio_segment", side_effect=slice_audio),
            patch.object(pipeline, "_submit_and_wait", side_effect=submit),
        ):
            pipeline._run_video_generation(
                "h3-audio",
                {
                    "video_model": model_type,
                    "pipeline_type": "music_video",
                    "seamless": False,
                    "video_params": {},
                    "audio_path": song,
                },
                [{"video_prompt": "A dancer performs."}],
                [{"start": 2.0, "duration_frames": 124}],
                [os.path.basename(shot)],
                out_dir=self.temp_dir.name,
            )

        self.assertEqual(captured["multi_clip_concat_audio"], song)
        audio_reference = captured["per_clip_minimax_h3_references"][0][-1]
        self.assertEqual(audio_reference["audio_intent"], "drive")
        self.assertTrue(generated_slice)
        self.assertFalse(os.path.exists(generated_slice[0]))


class TestDirectorUICatalogContract(unittest.TestCase):
    def test_ui_preserves_backend_director_capabilities(self):
        client_path = os.path.join(_ROOT_DIR, "ui", "src", "api", "client.ts")
        store_path = os.path.join(_ROOT_DIR, "ui", "src", "stores", "useStore.ts")
        types_path = os.path.join(_ROOT_DIR, "ui", "src", "types", "index.ts")
        chat_path = os.path.join(
            _ROOT_DIR, "ui", "src", "components", "Sidebar", "DirectorChat.tsx",
        )
        lora_selector_path = os.path.join(
            _ROOT_DIR,
            "ui",
            "src",
            "components",
            "SettingsDrawer",
            "DirectorLoraSelector.tsx",
        )
        launch_path = os.path.join(_APP_DIR, "launch.py")
        pipeline_path = os.path.join(_APP_DIR, "services", "director_pipeline.py")
        with open(client_path, encoding="utf-8") as handle:
            client = handle.read()
        with open(store_path, encoding="utf-8") as handle:
            store = handle.read()
        with open(types_path, encoding="utf-8") as handle:
            types = handle.read()
        with open(chat_path, encoding="utf-8") as handle:
            chat = handle.read()
        with open(lora_selector_path, encoding="utf-8") as handle:
            lora_selector = handle.read()
        with open(launch_path, encoding="utf-8") as handle:
            launch = handle.read()
        with open(pipeline_path, encoding="utf-8") as handle:
            pipeline_source = handle.read()

        self.assertIn("director?: DirectorModelCompatibility", client)
        self.assertIn("const backendModels: ModelDef[]", store)
        self.assertIn("...m,", store)
        self.assertIn("shot_image_guidance: directorShotImageGuidance", store)
        self.assertIn("director_max_shot_frames: directorMaxShotFrames", store)
        self.assertIn("resolution: directorVideoResolution", store)
        self.assertIn("minimax_h3_turbo_mode: directorTurboEnabled", store)
        self.assertIn("shot_image_support?", types)
        self.assertIn("Shot image guidance", chat)
        self.assertIn("Maximum planned shot", chat)
        self.assertIn("H3 Turbo", chat)
        self.assertIn("director_memory_policy", chat)
        self.assertIn("LoRA strength", lora_selector)
        self.assertIn('type="number"', lora_selector)
        self.assertIn("updateWeight(filename", lora_selector)
        self.assertIn("usesShotImages && (atStep('review')", chat)
        self.assertIn("SHOT_IMAGE_PROMPT_ONLY", pipeline_source)
        self.assertIn("_director_video_execution_profile", pipeline_source)
        self.assertIn("sliding_window_memory_override", pipeline_source)
        self.assertIn('"director_memory_policy": md.get', launch)
        self.assertIn("per_clip_continue_from_previous", launch)
        self.assertIn('"_continuation_tail_skip", 8', launch)


if __name__ == "__main__":
    unittest.main()
