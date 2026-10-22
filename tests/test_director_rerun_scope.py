"""Regression tests for one-clip Director reruns."""
from __future__ import annotations

import contextlib
import copy
import io
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402


class TestDirectorRerunScope(unittest.TestCase):
    def setUp(self):
        self.original_pipelines = pipeline._pipelines
        self.original_jobs = pipeline._jobs
        self.original_wgp = pipeline._wgp
        self.temp_dir = tempfile.TemporaryDirectory()
        pipeline._pipelines = {}
        pipeline._jobs = {}

    def tearDown(self):
        pipeline._pipelines = self.original_pipelines
        pipeline._jobs = self.original_jobs
        pipeline._wgp = self.original_wgp
        self.temp_dir.cleanup()

    def test_rerun_index_two_preflights_and_reports_dashboard_clip_three(self):
        pid = "pipe-rerun-scope"
        audio_path = os.path.join(self.temp_dir.name, "song.wav")
        with open(audio_path, "wb") as handle:
            handle.write(b"audio")

        plans = [
            {"video_prompt": "clip one prompt"},
            {"video_prompt": "clip two prompt"},
            {
                "video_prompt": "edited clip three prompt",
                "_director_h3_source_prompt": "edited clip three prompt",
                "_director_prompt_user_edited": True,
                "_director_audio_plan": {"mode": "music_driven"},
            },
        ]
        planned_starts = [0.0, 13.833333333333334, 28.083333333333332]
        planned_clips = [
            {
                "start": start,
                "end": start + 14.375,
                "duration_sec": 14.375,
                "duration_frames": 345,
                "output_frames": 345,
                "music_timing_version": 1,
            }
            for start in planned_starts
        ]
        profile = {
            "is_minimax_h3": True,
            "manual_override": True,
            "effective_max_frames": 345,
            "normalized_resolution": "1280x704",
        }
        pipeline._pipelines[pid] = {
            "id": pid,
            "status": "completed",
            "phase": "completed",
            "progress": {},
            "clip_plans": plans,
            "clip_images": [],
            "_planned_clips": planned_clips,
            "_clip_video_files": ["old-one.mp4", "old-two.mp4", "old-three.mp4"],
            "output_files": ["old-one.mp4", "old-two.mp4", "old-three.mp4"],
            "created_at": time.time(),
            "out_dir": self.temp_dir.name,
            "params": {
                "pipeline_type": "music_video",
                "seamless": False,
                "video_model": "minimax_h3_ref2va_fused_turbo",
                "video_params": {},
                "video_loras": {},
                "audio_path": audio_path,
                "_director_shot_image_policy": "direct_references",
                "_director_video_execution_profile": profile,
            },
        }
        self.assertTrue(pipeline._save_pipeline_state(pid))

        pipeline._wgp = SimpleNamespace(
            save_path=self.temp_dir.name,
            get_model_def=lambda _model: {
                "fps": 24,
                "frames_minimum": 124,
                "frames_steps": 17,
                "frames_maximum": 345,
                "director_video_strategy": pipeline.OMNI_REFERENCE,
            },
            get_model_min_frames_and_step=lambda _model: (124, 345, 17),
        )
        preflight_calls = []

        def fail_final_preflight(_model, clip_plans, **options):
            preflight_calls.append((clip_plans, options))
            if len(preflight_calls) == 2:
                raise ValueError("Shot 1: reference binding is invalid")
            return clip_plans

        submit = Mock(return_value=["unexpected.mp4"])
        output = io.StringIO()
        with (
            patch.object(pipeline, "_validate_director_models"),
            patch.object(pipeline, "_preflight_h3_director_prompts", side_effect=fail_final_preflight),
            patch.object(
                pipeline,
                "_director_h3_reference_manifest",
                return_value=[{"type": "image", "path": "reference.jpg"}],
            ),
            patch.object(pipeline, "_slice_audio_segment"),
            patch.object(pipeline, "_submit_and_wait", submit),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"Director clip 3 H3 prompt preflight failed: reference binding is invalid",
            ):
                pipeline.rerun_clip_video(self.temp_dir.name, pid, 2)

        self.assertEqual(len(preflight_calls), 2)
        for clip_plans, _options in preflight_calls:
            self.assertEqual(len(clip_plans), 1)
            self.assertEqual(
                clip_plans[0]["video_prompt"], "edited clip three prompt",
            )

        log = output.getvalue()
        self.assertIn("Clip 3 rerun frame budget", log)
        self.assertIn("Clip 3 rerun conditioned on song segment 28.083s-42.458s", log)
        self.assertNotIn("Clip 1 rerun", log)
        submit.assert_not_called()

        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(
            [clip["video_prompt"] for clip in saved["clips"]],
            ["clip one prompt", "clip two prompt", "edited clip three prompt"],
        )
        self.assertEqual(
            [clip["video_filename"] for clip in saved["clips"]],
            ["old-one.mp4", "old-two.mp4", "old-three.mp4"],
        )

    def test_rerun_recompiles_saved_style_and_preserves_reviewed_prompt(self):
        style = "2D anime with cel shading; avoid photorealism and 3D rendering"
        model_def = {
            "fps": 24,
            "frames_minimum": 17,
            "frames_steps": 8,
            "frames_maximum": 345,
            "director_video_strategy": pipeline.ROLLING_WINDOW,
        }
        pipeline._wgp = SimpleNamespace(
            save_path=self.temp_dir.name,
            get_model_def=lambda _model: model_def,
            get_model_min_frames_and_step=lambda _model: (17, 345, 8),
        )

        cases = (
            (
                "style-rerun-generated",
                "A performer dances beneath stage lights.",
                False,
            ),
            (
                "style-rerun-reviewed",
                "Watercolor illustration with soft pigment edges: a performer "
                "dances beneath stage lights.",
                True,
            ),
        )
        for pid, source, user_edited in cases:
            with self.subTest(user_edited=user_edited):
                clip_plan = {
                    "video_prompt": source,
                    "_director_h3_source_prompt": source,
                    "_director_h3_compiled_prompt": "",
                    "_director_h3_prompt_mode": "t2va",
                    "_director_h3_model_family": "base",
                    "_director_prompt_user_edited": user_edited,
                    "_director_visual_style": style,
                    "_director_subjects_on_screen": [],
                    "_director_dialogue_beats": [],
                }
                pipeline._pipelines[pid] = {
                    "id": pid,
                    "status": "completed",
                    "phase": "completed",
                    "progress": {},
                    "clip_plans": [clip_plan],
                    "clip_images": [],
                    "_planned_clips": [{
                        "start": 0.0,
                        "end": 8.0,
                        "duration_sec": 8.0,
                        "duration_frames": 192,
                        "output_frames": 192,
                        "music_timing_version": 0,
                    }],
                    "_clip_video_files": [],
                    "output_files": [],
                    "created_at": time.time(),
                    "out_dir": self.temp_dir.name,
                    "params": {
                        "pipeline_type": "short_film_story",
                        "scene_description": "A performer dances beneath stage lights.",
                        "seamless": False,
                        "video_model": "minimax_h3_test",
                        "video_params": {},
                        "video_loras": {},
                        "_director_shot_image_policy": "prompt_only",
                    },
                }
                self.assertTrue(pipeline._save_pipeline_state(pid))

                compiled_during_rerun = []
                real_preflight = pipeline._preflight_h3_director_prompts

                def capture_preflight(video_model, plans, **options):
                    result = real_preflight(video_model, plans, **options)
                    compiled_during_rerun.append(copy.deepcopy(plans[0]))
                    return result

                output_path = os.path.join(self.temp_dir.name, f"{pid}.mp4")
                with (
                    patch.object(pipeline, "_validate_director_models"),
                    patch.object(
                        pipeline,
                        "_saved_director_video_execution_profile",
                        return_value={},
                    ),
                    patch.object(
                        pipeline,
                        "_preflight_h3_director_prompts",
                        side_effect=capture_preflight,
                    ),
                    patch.object(
                        pipeline,
                        "_submit_and_wait",
                        return_value=[output_path],
                    ),
                ):
                    result = pipeline.rerun_clip_video(
                        self.temp_dir.name, pid, 0,
                    )

                self.assertEqual(result["filename"], output_path)
                self.assertEqual(len(compiled_during_rerun), 2)
                self.assertEqual(
                    compiled_during_rerun[-1]["_director_visual_style"], style,
                )
                self.assertEqual(
                    compiled_during_rerun[-1]["_director_prompt_user_edited"],
                    user_edited,
                )
                final_prompt = compiled_during_rerun[-1]["video_prompt"]
                saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
                self.assertEqual(
                    saved["clips"][0]["_director_visual_style"], style,
                )
                self.assertEqual(saved["clips"][0]["video_prompt"], final_prompt)

                if user_edited:
                    self.assertIn("Watercolor illustration", final_prompt)
                    self.assertNotIn(
                        "Visual style across the entire frame: 2D anime",
                        final_prompt,
                    )
                else:
                    self.assertIn(style, final_prompt)
                    self.assertIn("Visual style across the entire frame", final_prompt)


if __name__ == "__main__":
    unittest.main()
