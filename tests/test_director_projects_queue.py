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

        self.assertEqual(saved["version"], 2)
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
        remaining = pipeline.list_director_queue(self.temp_dir.name)
        self.assertEqual([entry["id"] for entry in remaining["entries"]], [second_id])

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
