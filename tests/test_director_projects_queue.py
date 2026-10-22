"""Model-free regressions for Director project revisions and held queue."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402


class TestDirectorProjectRevisionsAndQueue(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.originals = {
            "pipelines": pipeline._pipelines,
            "wgp": pipeline._wgp,
            "queue_state": pipeline._director_queue_state,
            "queue_base": pipeline._director_queue_base,
            "queue_worker": pipeline._director_queue_worker,
        }
        pipeline._pipelines = {}
        pipeline._wgp = SimpleNamespace(save_path=self.temp_dir.name)
        pipeline._director_queue_state = None
        pipeline._director_queue_base = None
        pipeline._director_queue_worker = None

    def tearDown(self):
        pipeline._pipelines = self.originals["pipelines"]
        pipeline._wgp = self.originals["wgp"]
        pipeline._director_queue_state = self.originals["queue_state"]
        pipeline._director_queue_base = self.originals["queue_base"]
        pipeline._director_queue_worker = self.originals["queue_worker"]
        self.temp_dir.cleanup()

    def _asset(self, name: str = "reference.png") -> str:
        path = os.path.join(self.temp_dir.name, name)
        with open(path, "wb") as handle:
            handle.write(b"director-test-asset")
        return path

    def _saved_pipeline(self, clips) -> tuple[str, str]:
        pid = "prompt-edit-test"
        path = os.path.join(self.temp_dir.name, f"_director_pipeline_{pid}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"id": pid, "pipeline_id": pid, "clips": clips}, handle)
        return pid, path

    def test_shot_visual_style_persists_and_restores_for_resume(self):
        pid = "visual-style-resume"
        style = (
            "2D anime with cel shading; avoid photorealism and 3D rendering"
        )
        planned_clip = {
            "start": 0.0,
            "end": 8.0,
            "duration_sec": 8.0,
            "duration_frames": 128,
            "_director_visual_style": style,
        }
        pipeline._pipelines[pid] = {
            "id": pid,
            "status": "crashed",
            "phase": "video",
            "params": {
                "pipeline_type": "short_film_story",
                "scene_description": "A dancer performs beneath neon lights.",
                "video_model": "test-model",
                "seamless": True,
            },
            "out_dir": self.temp_dir.name,
            "clip_plans": [{
                "video_prompt": "A watercolor dancer performs beneath neon lights.",
                "_director_h3_source_prompt": (
                    "A watercolor dancer performs beneath neon lights."
                ),
                "_director_prompt_user_edited": True,
                "_director_visual_style": style,
            }],
            "_planned_clips": [planned_clip],
            "output_files": [],
        }

        self.assertTrue(pipeline._save_pipeline_state_locked(pid))
        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(saved["clips"][0]["_director_visual_style"], style)
        self.assertEqual(
            saved["clips"][0]["planned_clip"]["_director_visual_style"],
            style,
        )

        pipeline._pipelines.pop(pid)
        recovered_status = pipeline.get_pipeline_status(pid, self.temp_dir.name)
        self.assertTrue(
            recovered_status["clip_plans"][0]["_director_prompt_user_edited"]
        )
        self.assertEqual(
            recovered_status["clip_plans"][0]["_director_visual_style"],
            style,
        )

        pipeline._wgp.get_model_def = lambda _model: {}
        with (
            patch.object(pipeline, "_repair_saved_h3_frame_lattice", return_value=False),
            patch.object(pipeline, "_validate_director_models"),
            patch.object(pipeline, "_saved_director_video_execution_profile", return_value={}),
            patch.object(pipeline, "_validate_saved_profile_for_current_hardware"),
            patch.object(pipeline, "_start_pipeline_worker"),
        ):
            ok, message = pipeline._resume_pipeline_reserved(
                pid, self.temp_dir.name,
            )

        self.assertTrue(ok, message)
        resumed = pipeline._pipelines[pid]
        self.assertEqual(
            resumed["clip_plans"][0]["_director_visual_style"], style,
        )
        self.assertTrue(resumed["clip_plans"][0]["_director_prompt_user_edited"])
        self.assertEqual(
            resumed["_planned_clips"][0]["_director_visual_style"], style,
        )

    def test_native_timeline_adaptation_keeps_each_clip_style(self):
        styles = ["2D cel animation", "watercolor illustration"]
        pipeline._wgp.get_model_def = lambda _model: {
            "architecture": "minimax_h3_ref2va",
            "omni_reference": True,
            "director_video_strategy": "omni_reference",
            "fps": 24,
            "frames_minimum": 124,
            "frames_maximum": 209,
            "frames_steps": 17,
        }
        plans = [{
            "video_prompt": f"Action {index}.",
            "_director_visual_style": style,
        } for index, style in enumerate(styles)]
        clips = [{
            "start": index * 12.0,
            "end": (index + 1) * 12.0,
            "duration_sec": 12.0,
            "duration_frames": 288,
            "music_timing_version": 0,
            "_director_visual_style": style,
        } for index, style in enumerate(styles)]

        adapted_plans, adapted_clips = pipeline.prepare_director_timeline(
            {
                "video_model": "minimax_h3_ref2va",
                "pipeline_type": "music_video",
                "_director_video_execution_profile": {
                    "effective_max_frames": 209,
                },
            },
            plans,
            clips,
        )

        self.assertGreater(len(adapted_plans), len(plans))
        self.assertEqual(len(adapted_plans), len(adapted_clips))
        for plan, clip in zip(adapted_plans, adapted_clips):
            source_index = plan["_director_source_clip_indices"][0]
            self.assertEqual(plan["_director_visual_style"], styles[source_index])
            self.assertEqual(clip["_director_visual_style"], styles[source_index])

    def test_reviewed_clip_prompts_persist_and_clear_stale_h3_cache(self):
        pid, _path = self._saved_pipeline([
            {"image_prompt": "old image", "video_prompt": "old video",
             "window_prompts": ["old video"],
             "_director_h3_source_prompt": "old source",
             "_director_h3_compiled_prompt": "old compiled",
             "_director_dialogue_beats": [{"text": "old speech"}]},
            {"video_prompt": "old one\nold two", "window_prompts": ["old one", "old two"]},
        ])

        self.assertTrue(pipeline.update_clip_prompt(
            self.temp_dir.name, pid, 0, {"image_prompt": "reviewed image"},
        ))
        self.assertTrue(pipeline.update_clip_prompt(
            self.temp_dir.name, pid, 0, {"video_prompt": "reviewed video"},
        ))
        self.assertTrue(pipeline.update_clip_prompt(
            self.temp_dir.name, pid, 1,
            {"window_prompts": ["reviewed one", "reviewed two"]},
        ))

        reloaded = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(reloaded["clips"][0]["image_prompt"], "reviewed image")
        self.assertEqual(reloaded["clips"][0]["video_prompt"], "reviewed video")
        self.assertEqual(reloaded["clips"][0]["window_prompts"], ["reviewed video"])
        self.assertEqual(reloaded["clips"][0]["_director_h3_source_prompt"], "reviewed video")
        self.assertTrue(reloaded["clips"][0]["_director_prompt_user_edited"])
        self.assertEqual(reloaded["clips"][0]["_director_h3_compiled_prompt"], "")
        self.assertEqual(reloaded["clips"][0]["_director_dialogue_beats"], [])
        self.assertEqual(reloaded["clips"][1]["window_prompts"], ["reviewed one", "reviewed two"])
        self.assertEqual(reloaded["clips"][1]["video_prompt"], "reviewed one\nreviewed two")

    def test_prompt_edit_keeps_render_provenance_and_marks_media_stale(self):
        pid, _path = self._saved_pipeline([{
            "video_prompt": "rendered prompt", "video_filename": "existing.mp4",
        }])
        pipeline.update_clip_prompt(self.temp_dir.name, pid, 0, {"video_prompt": "first edit"})
        pipeline.update_clip_prompt(self.temp_dir.name, pid, 0, {"video_prompt": "second edit"})
        clip = pipeline.load_pipeline_state(self.temp_dir.name, pid)["clips"][0]
        self.assertEqual(clip["video_filename"], "existing.mp4")
        self.assertEqual(clip["_director_rendered_prompt"], "rendered prompt")
        self.assertTrue(clip["video_stale"])

    def test_reviewed_clip_prompt_rejects_ambiguous_invalid_and_busy_updates(self):
        pid, _path = self._saved_pipeline([
            {"video_prompt": "one\ntwo", "window_prompts": ["one", "two"]},
        ])
        with self.assertRaisesRegex(ValueError, "multiple window prompts"):
            pipeline.update_clip_prompt(
                self.temp_dir.name, pid, 0, {"video_prompt": "ambiguous"},
            )
        for clip_index, update, message in (
            (3, {"image_prompt": "x"}, "out of range"),
            (0, [], "JSON object"),
            (0, {}, "No prompt field"),
            (0, {"window_prompts": []}, "non-empty list"),
        ):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                pipeline.update_clip_prompt(self.temp_dir.name, pid, clip_index, update)
        with patch.object(pipeline, "_claim_pipeline_operation", return_value=False):
            with self.assertRaises(pipeline.PipelineBusyError):
                pipeline.update_clip_prompt(
                    self.temp_dir.name, pid, 0, {"window_prompts": ["one", "two"]},
                )

    def test_start_persists_complete_revision_before_worker(self):
        reference = self._asset()
        params = {
            "pipeline_type": "music_video",
            "scene_description": "Editable project",
            "reference_image_path": reference,
            "image_model": "image-test",
            "video_model": "video-test",
            "prepared_clip_plans": [{
                "image_prompt": "same opening image",
                "video_prompt": "same reviewed video prompt",
            }],
            "prepared_planned_clips": [{
                "start": 0.0,
                "end": 5.0,
                "section_label": "scene",
                "energy": 0.5,
                "suggested_prompt_hint": "",
                "beat_count": 4,
                "duration_frames": 81,
            }],
            "director_ui_snapshot": {
                "snapshot_version": 1,
                "directorSongStyle": "dark synthwave",
            },
        }

        with (
            patch.object(pipeline, "_validate_director_models"),
            patch.object(
                pipeline,
                "_create_director_video_execution_profile",
                return_value={"is_minimax_h3": False},
            ),
            patch.object(pipeline, "_start_pipeline_worker") as start_worker,
        ):
            pid = pipeline.start_pipeline(params)

        start_worker.assert_called_once_with(pid)
        state_path = os.path.join(
            self.temp_dir.name, f"_director_pipeline_{pid}.json",
        )
        self.assertTrue(os.path.isfile(state_path))
        with open(state_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)

        self.assertEqual(saved["version"], pipeline.PIPELINE_STATE_VERSION)
        self.assertIn("planning_checkpoint", saved)
        self.assertIsNone(saved["planning_checkpoint"])
        self.assertEqual(saved["project_id"], pid)
        self.assertEqual(saved["director_ui_snapshot"]["directorSongStyle"], "dark synthwave")
        self.assertEqual(saved["clips"][0]["video_prompt"], "same reviewed video prompt")
        owned_reference = saved["_params_snapshot"]["reference_image_path"]
        self.assertTrue(os.path.isfile(owned_reference))
        self.assertIn(os.path.join("_director_assets", pid), owned_reference)
        self.assertEqual(
            saved["asset_manifest"]["reference_image_path"]["serve_path"],
            os.path.relpath(owned_reference, self.temp_dir.name).replace(os.sep, "/"),
        )

    def test_held_queue_owns_assets_reorders_and_removes(self):
        first = pipeline.enqueue_director_pipeline(self.temp_dir.name, {
            "scene_description": "First project",
            "pipeline_type": "music_video",
            "reference_image_path": self._asset("first.png"),
            "video_model": "video-a",
        })
        second = pipeline.enqueue_director_pipeline(self.temp_dir.name, {
            "scene_description": "Second project",
            "pipeline_type": "short_film_story",
            "reference_image_path": self._asset("second.png"),
            "video_model": "video-b",
        })

        self.assertTrue(first["paused"])
        self.assertEqual([entry["status"] for entry in second["entries"]], ["held", "held"])
        first_id, second_id = [entry["id"] for entry in second["entries"]]
        detail = pipeline.get_director_queue_entry(self.temp_dir.name, first_id)
        self.assertIsNotNone(detail)
        self.assertTrue(detail["params"]["auto_mode"])
        owned = detail["params"]["reference_image_path"]
        self.assertTrue(os.path.isfile(owned))
        self.assertIn(os.path.join("_director_queue_assets", first_id), owned)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir.name, "_director_queue.json")))

        reordered = pipeline.reorder_director_queue(
            self.temp_dir.name, [second_id, first_id],
        )
        self.assertEqual([entry["id"] for entry in reordered["entries"]], [second_id, first_id])
        edited = pipeline.update_director_queue_entry(
            self.temp_dir.name,
            second_id,
            {
                "scene_description": "Second project, edited",
                "reference_image_path": self._asset("second-edited.png"),
            },
        )
        edited_entry = next(item for item in edited["entries"] if item["id"] == second_id)
        self.assertEqual(edited_entry["scene_description"], "Second project, edited")
        edited_detail = pipeline.get_director_queue_entry(self.temp_dir.name, second_id)
        self.assertEqual(edited_detail["params"]["scene_description"], "Second project, edited")
        self.assertIn(
            os.path.join("_director_queue_assets", second_id),
            edited_detail["params"]["reference_image_path"],
        )
        self.assertTrue(pipeline.remove_director_queue_entry(self.temp_dir.name, first_id))
        self.assertFalse(os.path.exists(owned))
        remaining = pipeline.list_director_queue(self.temp_dir.name)
        self.assertEqual([entry["id"] for entry in remaining["entries"]], [second_id])

    def test_clear_completed_history_preserves_projects_media_and_references(self):
        queue = pipeline.enqueue_director_pipeline(self.temp_dir.name, {
            "scene_description": "Finished project",
            "reference_image_path": self._asset(),
        })
        entry_id = queue["entries"][0]["id"]
        detail = pipeline.get_director_queue_entry(self.temp_dir.name, entry_id)
        reference = detail["params"]["reference_image_path"]
        video = self._asset("finished.mp4")
        _, project = self._saved_pipeline([{"video_filename": video, "reference_image_path": reference}])
        pipeline._director_queue_state["entries"][0]["status"] = "completed"

        self.assertTrue(pipeline.remove_director_queue_entry(
            self.temp_dir.name, entry_id, completed_only=True,
        ))
        pipeline._director_queue_state = None
        self.assertEqual(pipeline.list_director_queue(self.temp_dir.name)["entries"], [])
        for retained in (reference, video, project):
            with self.subTest(path=retained):
                self.assertTrue(os.path.isfile(retained))

    def test_clear_completed_rejects_entries_that_are_no_longer_completed(self):
        queue = pipeline.enqueue_director_pipeline(self.temp_dir.name, {
            "scene_description": "Keep this project",
            "reference_image_path": self._asset(),
        })
        entry_id = queue["entries"][0]["id"]
        detail = pipeline.get_director_queue_entry(self.temp_dir.name, entry_id)
        reference = detail["params"]["reference_image_path"]
        for status in ("held", "queued", "running", "failed", "cancelled"):
            with self.subTest(status=status):
                pipeline._director_queue_state["entries"][0]["status"] = status
                with self.assertRaises(pipeline.PipelineBusyError):
                    pipeline.remove_director_queue_entry(
                        self.temp_dir.name, entry_id, completed_only=True,
                    )
                self.assertEqual(pipeline.get_director_queue_entry(self.temp_dir.name, entry_id)["status"], status)
                self.assertTrue(os.path.isfile(reference))

    def test_held_queue_owns_ordered_omni_media_and_attached_soundtrack(self):
        identity = self._asset("identity.png")
        motion = self._asset("motion.mp4")
        attached = self._asset("motion-audio.wav")
        queue = pipeline.enqueue_director_pipeline(self.temp_dir.name, {
            "scene_description": "Mixed Omni references",
            "pipeline_type": "short_film_story",
            "video_model": "minimax_h3_ref2va",
            "minimax_h3_references": [
                {
                    "id": "picture",
                    "type": "image",
                    "path": identity,
                    "filename": "identity.png",
                    "role": "the hero",
                },
                {
                    "id": "motion",
                    "type": "video",
                    "path": motion,
                    "filename": "motion.mp4",
                    "role": "camera movement",
                    "audio_path": attached,
                    "include_audio": True,
                },
            ],
        })

        entry_id = queue["entries"][0]["id"]
        detail = pipeline.get_director_queue_entry(self.temp_dir.name, entry_id)
        references = detail["params"]["minimax_h3_references"]
        self.assertEqual([item["id"] for item in references], ["picture", "motion"])
        for reference in references:
            self.assertTrue(os.path.isfile(reference["path"]))
            self.assertIn(os.path.join("_director_queue_assets", entry_id), reference["path"])
        self.assertTrue(os.path.isfile(references[1]["audio_path"]))
        manifest = detail["params"]["_director_asset_manifest"][
            "minimax_h3_references"
        ]
        self.assertIn("path", manifest[0])
        self.assertIn("audio_path", manifest[1])

    def test_restart_returns_running_entry_to_held_queue(self):
        path = os.path.join(self.temp_dir.name, "_director_queue.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "version": 1,
                "paused": False,
                "running": True,
                "entries": [{
                    "id": "overnight",
                    "status": "running",
                    "message": "Rendering",
                    "params": {"scene_description": "Overnight project"},
                }],
            }, handle)

        pipeline._director_queue_state = None
        pipeline._director_queue_base = None
        restored = pipeline.list_director_queue(self.temp_dir.name)
        self.assertTrue(restored["paused"])
        self.assertFalse(restored["running"])
        self.assertEqual(restored["entries"][0]["status"], "held")
        self.assertIn("Interrupted", restored["entries"][0]["message"])

    def test_queue_dispatches_complete_projects_sequentially(self):
        pipeline.enqueue_director_pipeline(self.temp_dir.name, {
            "scene_description": "One",
        })
        pipeline.enqueue_director_pipeline(self.temp_dir.name, {
            "scene_description": "Two",
        })
        with pipeline._director_queue_lock:
            state = pipeline._load_director_queue_locked(self.temp_dir.name)
            state["paused"] = False
            for entry in state["entries"]:
                entry["status"] = "queued"

        started = []

        def fake_start(params):
            pid = f"render-{len(started) + 1}"
            started.append((pid, params["scene_description"]))
            return pid

        with (
            patch.object(pipeline, "start_pipeline", side_effect=fake_start),
            patch.object(
                pipeline,
                "get_pipeline",
                side_effect=lambda pid: {"status": "completed", "error": None},
            ),
        ):
            pipeline._run_director_queue(self.temp_dir.name)

        self.assertEqual(started, [("render-1", "One"), ("render-2", "Two")])
        queue = pipeline.list_director_queue(self.temp_dir.name)
        self.assertFalse(queue["running"])
        self.assertEqual(
            [entry["status"] for entry in queue["entries"]],
            ["completed", "completed"],
        )

    def test_queue_waits_behind_direct_pipeline_before_dispatch(self):
        queued = pipeline.enqueue_director_pipeline(self.temp_dir.name, {
            "scene_description": "Next revision",
        })
        entry_id = queued["entries"][0]["id"]
        with pipeline._director_queue_lock:
            state = pipeline._load_director_queue_locked(self.temp_dir.name)
            state["paused"] = False
            state["entries"][0]["status"] = "queued"
        pipeline._pipelines["manual-run"] = {"status": "running"}
        started = []

        def finish_active(_seconds):
            self.assertEqual(started, [])
            pipeline._pipelines["manual-run"]["status"] = "completed"

        with (
            patch.object(pipeline.time, "sleep", side_effect=finish_active),
            patch.object(
                pipeline,
                "start_pipeline",
                side_effect=lambda params: started.append(params["scene_description"]) or "queued-run",
            ),
            patch.object(
                pipeline,
                "get_pipeline",
                return_value={"status": "completed", "error": None},
            ),
        ):
            pipeline._run_director_queue(self.temp_dir.name)

        self.assertEqual(started, ["Next revision"])
        detail = pipeline.get_director_queue_entry(self.temp_dir.name, entry_id)
        self.assertEqual(detail["status"], "completed")

    def test_queue_surfaces_pipeline_gpu_wait_message(self):
        queued = pipeline.enqueue_director_pipeline(self.temp_dir.name, {
            "scene_description": "Director after Studio",
        })
        entry_id = queued["entries"][0]["id"]
        with pipeline._director_queue_lock:
            state = pipeline._load_director_queue_locked(self.temp_dir.name)
            state["paused"] = False
            state["entries"][0]["status"] = "queued"

        statuses = iter([
            {
                "status": "running",
                "progress": {
                    "message": "Waiting for GPU (generation queue)...",
                },
            },
            {"status": "completed", "error": None},
        ])

        def observe_wait(_seconds):
            detail = pipeline.get_director_queue_entry(
                self.temp_dir.name, entry_id,
            )
            self.assertEqual(
                detail["message"],
                "Waiting for GPU (generation queue)...",
            )

        with (
            patch.object(pipeline, "start_pipeline", return_value="queued-run"),
            patch.object(pipeline, "get_pipeline", side_effect=lambda _pid: next(statuses)),
            patch.object(pipeline.time, "sleep", side_effect=observe_wait),
        ):
            pipeline._run_director_queue(self.temp_dir.name)

        detail = pipeline.get_director_queue_entry(self.temp_dir.name, entry_id)
        self.assertEqual(detail["status"], "completed")

    def test_prepared_queue_revision_reaches_gpu_wait(self):
        """A frozen Director revision can enter the normal GPU waiter."""

        pid = "prepared-queue-run"
        pipeline._pipelines[pid] = {
            "id": pid,
            "status": "running",
            "phase": "planning",
            "params": {
                "pipeline_type": "music_video",
                "prepared_clip_plans": [{
                    "image_prompt": "reviewed opening frame",
                    "video_prompt": "reviewed Tom Cruise shot",
                }],
                "prepared_planned_clips": [{
                    "start": 0.0,
                    "end": 5.0,
                    "duration_frames": 81,
                }],
            },
            "out_dir": self.temp_dir.name,
            "workspace": "default",
            "clip_plans": [],
            "clip_images": [],
            "output_files": [],
            "progress": {},
        }

        # Returning False models cancellation after entering the normal GPU
        # waiter and deliberately stops before LLM or model work. The frozen
        # prepared plans must be copied successfully before that point.
        with patch.object(pipeline, "_wait_for_gpu", return_value=False) as wait:
            pipeline._run_pipeline(pid)

        wait.assert_called_once_with(pid)
        self.assertNotEqual(pipeline._pipelines[pid]["status"], "failed")
        self.assertIsNone(pipeline._pipelines[pid].get("error"))

    def test_gpu_wait_blocks_until_studio_generation_finishes(self):
        pid = "director-behind-studio"
        pipeline._pipelines[pid] = {
            "status": "running",
            "progress": {},
        }
        studio_job = {"status": "running"}

        def finish_studio(_seconds):
            studio_job["status"] = "completed"

        with (
            patch.object(pipeline, "_jobs", {"studio-job": studio_job}),
            patch.object(pipeline.time, "sleep", side_effect=finish_studio) as sleep,
        ):
            ready = pipeline._wait_for_gpu(pid, poll_interval=0.01)

        self.assertTrue(ready)
        sleep.assert_called_once_with(0.01)
        self.assertEqual(
            pipeline._pipelines[pid]["progress"]["message"],
            "Waiting for GPU (generation queue)...",
        )


if __name__ == "__main__":
    unittest.main()
