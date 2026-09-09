"""Audio analysis, reviewed plans, and rendering must keep the same timeline."""
import copy
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from services import director_pipeline as pipeline, llm_service


H3 = {
    "architecture": "minimax_h3_ref2va", "omni_reference": True,
    "director_video_strategy": "omni_reference", "fps": 24,
    "frames_minimum": 124, "frames_maximum": 345, "frames_steps": 17,
}
LTX = {
    "architecture": "ltx2_25", "sliding_window": True, "fps": 24,
    "frames_minimum": 17, "frames_steps": 8,
    "sliding_window_defaults": {"window_default": 241, "window_max": 241},
}


class TestDirectorMusicTimeline(unittest.TestCase):
    def setUp(self):
        # Section boundaries from #117's uploaded pipeline, without prompts,
        # reference media, paths, or any other user-specific data.
        boundaries = [0, 13, 31.022, 49, 70.705, 92, 124, 150, 175.177]
        self.clips = [{"start": start, "end": end, "duration_frames": round((end - start) * 16)}
                      for start, end in zip(boundaries, boundaries[1:])]
        self.plans = [{"image_prompt": f"Scene {i}", "video_prompt": f"Action {i}. Then a reaction.", "image_source": "original"}
                      for i in range(len(self.clips))]
        self.params = {
            "video_model": "minimax_h3_ref2va", "pipeline_type": "music_video",
            "audio_path": "song.wav", "lyrics": "A singer sings.",
            "video_params": {"resolution": "864x480"},
            "_director_video_execution_profile": {"effective_max_frames": 345},
        }
        patcher = mock.patch.object(pipeline, "_wgp", SimpleNamespace(get_model_def=lambda model: LTX if model == "ltx2_25" else H3, server_config={"services": {"director_prompt_polish": "off"}}))
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_full_song(self, timeline, *, minimum, maximum, step):
        self.assertGreater(len(timeline), len(self.clips))
        self.assertLess(abs(timeline[-1]["end"] - 175.177), step / 24 + 0.05)
        cursor = 0
        for clip in timeline:
            self.assertAlmostEqual(clip["start"], cursor)
            self.assertGreaterEqual(clip["duration_frames"], minimum)
            self.assertLessEqual(clip["duration_frames"], maximum)
            self.assertEqual((clip["duration_frames"] - minimum) % step, 0)
            cursor = clip["end"]

    def test_h3_preview_and_review_preserve_full_song_and_are_idempotent(self):
        plans, timeline = pipeline.prepare_director_timeline(self.params, self.plans, self.clips)
        self.assert_full_song(timeline, minimum=124, maximum=345, step=17)
        again, same = pipeline.prepare_director_timeline(self.params, plans, timeline)
        self.assertIs(again, plans)
        self.assertIs(same, timeline)
        for plan in plans:
            source = plan["_director_source_clip_indices"][0]
            self.assertIn(self.plans[source]["image_prompt"], plan["image_prompt"])

    def test_ltx_preview_already_shows_the_shots_that_will_render(self):
        params = {**self.params, "video_model": "ltx2_25"}
        plans, timeline = pipeline.prepare_director_timeline(params, self.plans, self.clips)
        self.assert_full_song(timeline, minimum=17, maximum=241, step=8)
        self.assertIs(pipeline.prepare_director_timeline(params, plans, timeline)[1], timeline)

    def test_opening_legacy_long_plan_does_not_shorten_it(self):
        state = {"video_model": "minimax_h3_ref2va", "clips": [{"planned_clip": clip} for clip in self.clips]}
        original = copy.deepcopy(state)
        self.assertFalse(pipeline._repair_saved_h3_frame_lattice(state))
        self.assertEqual(state, original)

    def test_prepared_revision_splits_before_render_and_keeps_reviewed_images(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            images = []
            for index in range(len(self.plans)):
                path = os.path.join(directory, f"shot-{index}.png")
                with open(path, "wb") as handle:
                    handle.write(b"image")
                images.append(path)
            params = {
                **copy.deepcopy(self.params), "auto_mode": True, "seamless": False,
                "image_model": "flux2_klein_9b", "_director_shot_image_policy": "generate",
                "prepared_clip_plans": self.plans, "prepared_planned_clips": self.clips,
                "prepared_clip_image_paths": images,
            }
            record = {"id": "music-review", "status": "running", "params": params,
                      "out_dir": directory, "clip_plans": [], "clip_images": [], "output_files": [], "progress": {}}
            stack.enter_context(mock.patch.dict(pipeline._pipelines, {"music-review": record}))
            for name in ("_save_pipeline_state", "_validate_director_models", "_preflight_h3_director_prompts"):
                stack.enter_context(mock.patch.object(pipeline, name))
            stack.enter_context(mock.patch.object(pipeline, "_wait_for_gpu", return_value=True))
            stack.enter_context(mock.patch.object(llm_service, "is_loaded", return_value=False))
            planning = stack.enter_context(mock.patch.object(pipeline, "_run_planning", side_effect=AssertionError("Reviewed prompts must not be rewritten by the LLM")))
            images_gen = stack.enter_context(mock.patch.object(pipeline, "_run_image_generation", side_effect=AssertionError("Reviewed images must stay assigned")))
            render = stack.enter_context(mock.patch.object(pipeline, "_run_video_generation", return_value=[]))
            pipeline._run_pipeline("music-review")
            self.assertEqual(record["status"], "completed", record.get("error"))
            planning.assert_not_called()
            images_gen.assert_not_called()
            args = render.call_args.args
            plans, timeline, rendered_images = args[2:5]
            self.assert_full_song(timeline, minimum=124, maximum=345, step=17)
            for plan, image in zip(plans, rendered_images):
                source = plan["_director_source_clip_indices"][0]
                self.assertEqual(os.path.basename(image), f"shot-{source}.png")


if __name__ == "__main__":
    unittest.main()
