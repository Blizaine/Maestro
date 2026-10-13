"""Focused checks for clip-local timing and stable cast identity in music plans."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.director.planners.music_video import (  # noqa: E402
    MusicVideoPlanner,
    _MUSIC_SHOT_PROPERTIES,
    _music_shot_schema,
)
from services.director.prompt_timing import (  # noqa: E402
    music_video_visual_timing_issue,
    normalize_music_video_visual_timecodes,
)
from services.director.h3_dialogue import (  # noqa: E402
    compile_h3_clip_plans,
    validate_h3_prompt_contract,
)


def _row(
    *,
    prompt: str,
    identity: bool = True,
    window_prompts: list[str] | None = None,
) -> dict:
    subject = {
        "visual_description": "a singer in a blue jacket",
        "performance_role": "vocalist",
    }
    if identity:
        subject["character_id"] = "lead_performer"
        subject["speaker_name"] = "Ari"
    return {
        "scene_goal": "The singer advances the chorus",
        "scene_type": "performance",
        "continuity_strategy": "continuous",
        "physical_state": {
            "opening": [],
            "transitions": [],
            "closing": [],
            "continuity": "continuous",
        },
        "subjects_on_screen": [subject],
        "environment": "A rooftop stage at dusk",
        "visual_style": "Cinematic concert photography",
        "lighting": "Amber side light",
        "mood": "Energetic",
        "action_beats": ["The singer steps toward the camera."],
        "camera_plan": {"framing": "medium shot", "movement": "slow push-in"},
        "ending_beat": "The singer holds the final pose",
        "video_prompt": prompt,
        "window_prompts": list(window_prompts or []),
    }


class MusicVideoPromptTimingTests(unittest.TestCase):
    def test_native_h3_windows_are_disallowed_without_mutating_other_schemas(self):
        original_window_schema = dict(_MUSIC_SHOT_PROPERTIES["window_prompts"])
        h3 = _music_shot_schema(
            1,
            include_image_fields=False,
            require_native_single_clip=True,
        )
        other = _music_shot_schema(1, include_image_fields=False)

        h3_windows = h3["items"]["properties"]["window_prompts"]
        other_windows = other["items"]["properties"]["window_prompts"]
        self.assertEqual(h3_windows["maxItems"], 0)
        self.assertEqual(other_windows, original_window_schema)
        self.assertNotIn("maxItems", _MUSIC_SHOT_PROPERTIES["window_prompts"])

    def test_non_h3_music_row_can_keep_window_prompts(self):
        row = _row(
            prompt=(
                "[Shot 1] Ari faces the camera. "
                "[Shot 2] At 00:05.000: Ari turns toward the stage."
            ),
            window_prompts=[
                "At 00:00.000, Ari faces the camera.",
                "At 00:00.000, Ari turns toward the stage.",
            ],
        )

        issue = MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="Ari performs on a rooftop",
            include_image_fields=False,
        )

        self.assertIsNone(issue)

    def test_h3_identity_keys_are_grammar_constrained_without_mutating_other_models(self):
        h3 = _music_shot_schema(1, include_image_fields=False, require_cast_identity=True)
        other = _music_shot_schema(1, include_image_fields=False)
        required = h3['items']['properties']['subjects_on_screen']['items']['required']
        self.assertIn('character_id', required)
        self.assertIn('speaker_name', required)
        self.assertNotIn('speaker_name', other['items']['properties']['subjects_on_screen']['items']['required'])

    def test_valid_local_visual_times_ignore_source_song_and_dialogue_times(self):
        prompt = (
            "Source-song lyric cue at 00:21.000; "
            "[Shot 1] A singer faces the camera. "
            "[Shot 2] At 00:09.500: the camera pushes closer. "
            "The singer says <d>[English] At 00:31.500</d>."
        )

        self.assertIsNone(music_video_visual_timing_issue(prompt, 10.125))

    def test_malformed_and_out_of_range_visual_times_are_reported(self):
        malformed = music_video_visual_timing_issue(
            "[Shot 2] At MM:00.01.33: the camera cuts closer.", 10.125
        )
        self.assertIn("malformed", malformed or "")

        invalid_seconds = music_video_visual_timing_issue(
            "At 00:65.000: the performer turns toward the stage.", 90.0
        )
        self.assertIn("malformed", invalid_seconds or "")

        out_of_range = music_video_visual_timing_issue(
            "At 00:11.000: the performer turns toward the stage.", 10.125
        )
        self.assertIn("exceeds local clip duration 00:10.125", out_of_range or "")

    def test_redundant_zero_fraction_is_canonicalized_without_changing_time(self):
        raw = "[Shot 1] At 00:06.21.000: Ari turns toward the stage."
        normalized = normalize_music_video_visual_timecodes(raw)

        self.assertEqual(
            normalized,
            "[Shot 1] At 00:06.210: Ari turns toward the stage.",
        )
        self.assertEqual(
            normalize_music_video_visual_timecodes(normalized), normalized
        )
        self.assertIsNone(music_video_visual_timing_issue(normalized, 6.21))
        self.assertIn(
            "exceeds local clip duration",
            music_video_visual_timing_issue(normalized, 6.1) or "",
        )

        full_precision = normalize_music_video_visual_timecodes(
            "At 00:06.210.000: Ari turns."
        )
        self.assertEqual(full_precision, "At 00:06.210: Ari turns.")

    def test_ambiguous_timecodes_source_offsets_and_dialogue_are_unchanged(self):
        malformed = (
            "At 00:06.21.001: Ari turns.",
            "At 00:06.21.00: Ari turns.",
            "At 00:06.21.000.000: Ari turns.",
            "At 00:61.21.000: Ari turns.",
        )
        for prompt in malformed:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    normalize_music_video_visual_timecodes(prompt), prompt
                )
                self.assertIn(
                    "malformed",
                    music_video_visual_timing_issue(prompt, 10.0) or "",
                )

        source_and_dialogue = (
            "Source-song offset: At 00:06.21.000. "
            "The singer says <d>[English] At 00:06.21.000</d>."
        )
        self.assertEqual(
            normalize_music_video_visual_timecodes(source_and_dialogue),
            source_and_dialogue,
        )

    def test_h3_row_canonicalizes_prompt_and_action_beat_before_validation(self):
        row = _row(
            prompt="[Shot 1] At 00:06.21.000: Ari turns toward the stage."
        )
        row["action_beats"] = ["At 00:06.21.000: Ari turns toward the stage."]

        issue = MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="Ari performs on a rooftop",
            include_image_fields=False,
            duration_seconds=6.21,
            validate_visual_timing=True,
        )

        self.assertIsNone(issue)
        self.assertIn("At 00:06.210", row["video_prompt"])
        self.assertEqual(
            row["action_beats"],
            ["At 00:06.210: Ari turns toward the stage."],
        )

    def test_plain_at_markers_accept_seconds_and_fractional_clock_forms(self):
        self.assertIsNone(music_video_visual_timing_issue(
            "At 00:05: the performer turns. At 00:09.5: the camera settles.",
            10.125,
        ))

    def test_plain_at_marker_can_follow_a_labeled_absolute_source_song_offset(self):
        prompt = (
            "Source-song offset: At 02:17.500. "
            "At 00:10.000: the camera reaches the end of this local clip."
        )
        self.assertIsNone(music_video_visual_timing_issue(prompt, 10.125))

    def test_source_song_prose_does_not_hide_a_later_visual_cut(self):
        issue = music_video_visual_timing_issue(
            'Source song begins over the stage, then At 00:17.500: the camera cuts to a drummer.',
            10.125,
        )
        self.assertIn('exceeds local clip duration', issue or '')

    def test_generic_vocal_role_is_not_a_named_identity(self):
        from services.director.planners.music_video import _has_meaningful_cast_identity
        for role in ('the vocalist', 'a singer', 'singer', 'the performer'):
            self.assertFalse(_has_meaningful_cast_identity({'character_id': 'character_1', 'speaker_name': role}), role)
        self.assertTrue(_has_meaningful_cast_identity({'character_id': 'lead_performer', 'speaker_name': ''}))

    def test_final_h3_contract_validator_receives_clip_duration(self):
        prompt = (
            "integrated_multimodal_description: [Shot 1] Ari faces the camera. "
            "[Shot 2] At 00:11.000: Ari turns toward the stage.\n\n"
            "overall_soundscape: Quiet stage ambience.\n\n"
            "non_diegetic_music: The source song continues."
        )
        errors = validate_h3_prompt_contract(prompt, duration_seconds=10.125)
        self.assertTrue(any("exceeds local clip duration" in error for error in errors), errors)

    def test_clip_context_states_actual_local_duration_and_separate_song_cues(self):
        clip = {
            "start": 120.0,
            "end": 130.125,
            "label": "chorus",
            "beat_count": 4,
            "percussion_activity": "active",
            "music_cues": [{"type": "percussion_entry", "time": 125.5}],
        }
        context = MusicVideoPlanner()._build_clip_contexts(
            [clip], None, {}, {}, None
        )[0]

        self.assertIn("Local visual duration: 10.125s", context)
        self.assertIn("clip-local timeline 00:00.000 through 00:10.125", context)
        self.assertIn("Source-song music timing", context)
        self.assertIn("song 125.50s, shot +5.50s", context)

    def test_h3_cast_identity_is_required_without_changing_other_models(self):
        row = _row(prompt="Ari performs beneath warm stage lights.", identity=False)
        issue = MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A singer performs on a rooftop",
            include_image_fields=False,
            require_stable_cast_identity=True,
        )
        self.assertIn("character_id or speaker_name", issue or "")

        self.assertIsNone(MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A singer performs on a rooftop",
            include_image_fields=False,
        ))

        guide = (ROOT / "app/services/llm_guides/director/minimax_h3_shot_breakdown.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("STABLE CAST IDENTITY", guide)
        self.assertIn("character_id", guide)
        self.assertIn("speaker_name", guide)
        self.assertIn("rider-to-mount", guide)
        self.assertIn("The target action owns the pose", guide)

    def test_numeric_slot_cast_id_needs_a_meaningful_speaker_name(self):
        row = _row(prompt="Ari performs beneath warm stage lights.")
        row["subjects_on_screen"][0]["character_id"] = "character_1"
        row["subjects_on_screen"][0].pop("speaker_name")
        issue = MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A singer performs on a rooftop",
            include_image_fields=False,
            require_stable_cast_identity=True,
        )
        self.assertIn("stable, meaningful character_id or speaker_name", issue or "")

        row["subjects_on_screen"][0]["speaker_name"] = "Ari"
        self.assertIsNone(MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A singer performs on a rooftop",
            include_image_fields=False,
            require_stable_cast_identity=True,
        ))

        row["subjects_on_screen"][0]["character_id"] = "lead_performer"
        self.assertIsNone(MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A singer performs on a rooftop",
            include_image_fields=False,
            require_stable_cast_identity=True,
        ))

        row["subjects_on_screen"][0]["character_id"] = "character_1"
        row["subjects_on_screen"][0]["speaker_name"] = "speaker_1"
        issue = MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A singer performs on a rooftop",
            include_image_fields=False,
            require_stable_cast_identity=True,
        )
        self.assertIn("stable, meaningful character_id or speaker_name", issue or "")

    def test_invalid_h3_visual_timing_enters_existing_repair_path(self):
        requests: list[str] = []

        def generate(**kwargs):
            prompt = str(kwargs.get("prompt") or "")
            requests.append(prompt)
            if "TAIL REPAIR" in prompt:
                return json.dumps([_row(
                    prompt=(
                        "[Shot 1] Ari stands under amber lights. "
                        "[Shot 2] At 00:09.500: Ari turns toward the camera."
                    )
                )])
            return json.dumps([_row(
                prompt=(
                    "[Shot 1] Ari stands under amber lights. "
                    "[Shot 2] At 00:11.000: Ari turns toward the camera."
                )
            )])

        plan = MusicVideoPlanner(llm_generate=generate).plan(
            clips=[{"start": 120.0, "end": 130.125, "label": "chorus"}],
            scene_description="Ari performs on a rooftop at dusk",
            video_model="minimax_h3_ref2va",
            shot_image_policy="direct_references",
        )

        self.assertEqual(len(requests), 2)
        self.assertIn("TAIL REPAIR", requests[1])
        self.assertIn("00:09.500", plan.shots[0].video_prompt or "")

    def test_native_h3_window_row_repairs_without_losing_valid_clip_cuts(self):
        requests: list[str] = []
        valid_cuts = (
            "[Shot 1] Ari stands under amber lights. "
            "[Shot 2] At 00:09.500: Ari turns toward the camera."
        )

        def generate(**kwargs):
            prompt = str(kwargs.get("prompt") or "")
            requests.append(prompt)
            windows = (
                []
                if "TAIL REPAIR" in prompt
                else [
                    "At 00:00.000, [Shot 1] Ari starts the movement. "
                    "[Shot 2] At 00:04.000: Ari turns.",
                    "At 00:00.000, [Shot 1] Ari continues the movement. "
                    "[Shot 2] At 00:04.000: the camera closes in.",
                ]
            )
            return json.dumps([_row(
                prompt=valid_cuts,
                window_prompts=windows,
            )])

        plan = MusicVideoPlanner(llm_generate=generate).plan(
            clips=[{"start": 120.0, "end": 130.125, "label": "chorus"}],
            scene_description="Ari performs on a rooftop at dusk",
            video_model="minimax_h3_ref2va",
            shot_image_policy="direct_references",
        )

        self.assertEqual(len(requests), 2)
        self.assertIn("TAIL REPAIR", requests[1])
        shot = plan.shots[0]
        self.assertEqual(shot.window_prompts, [])
        self.assertIn("[Shot 1]", shot.video_prompt or "")
        self.assertIn("[Shot 2] At 00:09.500", shot.video_prompt or "")

        compiled_plan = {
            "video_prompt": shot.video_prompt,
            "_director_h3_prompt_mode": "ref2va",
            "_director_h3_model_family": "ref2va",
            "_director_subjects_on_screen": shot.subjects_on_screen,
            "_director_dialogue_beats": [],
            "_director_duration_sec": shot.duration_sec,
            "_director_h3_reference_manifest": [],
            "_director_audio_plan": {"mode": "music_driven"},
            "_director_project_context": "Ari performs on a rooftop at dusk",
        }
        compile_h3_clip_plans([compiled_plan])
        self.assertEqual(
            validate_h3_prompt_contract(
                compiled_plan["video_prompt"],
                mode="ref2va",
                subjects=shot.subjects_on_screen,
                duration_seconds=shot.duration_sec,
            ),
            [],
        )
        self.assertIn("[Shot 2] At 00:09.500", compiled_plan["video_prompt"])

    def test_persistently_invalid_native_h3_windows_fail_after_bounded_repairs(self):
        calls = 0

        def generate(**kwargs):
            nonlocal calls
            calls += 1
            return json.dumps([_row(
                prompt=(
                    "[Shot 1] Ari stands under amber lights. "
                    "[Shot 2] At 00:09.500: Ari turns toward the camera."
                ),
                window_prompts=[
                    "At 00:00.000, [Shot 1] Ari starts the movement.",
                ],
            )])

        with self.assertRaisesRegex(
            RuntimeError,
            "2 failed repair attempts.*window_prompts must be empty",
        ):
            MusicVideoPlanner(llm_generate=generate).plan(
                clips=[{"start": 120.0, "end": 130.125, "label": "chorus"}],
                scene_description="Ari performs on a rooftop at dusk",
                video_model="minimax_h3_ref2va",
                shot_image_policy="direct_references",
            )

        self.assertEqual(calls, 3)

    def test_unrepaired_invalid_h3_visual_timing_is_rejected(self):
        calls = 0

        def generate(**kwargs):
            nonlocal calls
            calls += 1
            return json.dumps([_row(
                prompt=(
                    "[Shot 1] Ari stands under amber lights. "
                    "[Shot 2] At MM:00.01.33: Ari turns toward the camera."
                )
            )])

        with self.assertRaisesRegex(RuntimeError, "visual timecode.*malformed"):
            MusicVideoPlanner(llm_generate=generate).plan(
                clips=[{"start": 120.0, "end": 130.125, "label": "chorus"}],
                scene_description="Ari performs on a rooftop at dusk",
                video_model="minimax_h3_ref2va",
                shot_image_policy="direct_references",
            )

        self.assertEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
