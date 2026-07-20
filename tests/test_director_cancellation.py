"""Model-free cancellation tests for the Director pipeline."""
from __future__ import annotations

import json
import inspect
import os
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.job_lifecycle import (  # noqa: E402
    finish_job,
    is_cancel_requested,
    record_job_outputs,
    register_abort_state,
    request_cancel,
    try_start,
    unregister_abort_state,
)


class TestDirectorCancellation(unittest.TestCase):
    def setUp(self):
        self.originals = {
            "pipelines": pipeline._pipelines,
            "jobs": pipeline._jobs,
            "active": pipeline._active_gen_states,
            "threads": pipeline._pipeline_threads,
            "child_jobs": pipeline._pipeline_child_jobs,
            "starting": pipeline._pipeline_starting,
            "operations": pipeline._pipeline_operations,
            "deleting": pipeline._pipeline_deleting,
            "run_generation": pipeline._run_generation,
            "wgp": pipeline._wgp,
            "settle_grace": pipeline._GENERATION_SETTLE_GRACE_S,
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        pipeline._pipelines = {}
        pipeline._jobs = {}
        pipeline._active_gen_states = {}
        pipeline._pipeline_threads = {}
        pipeline._pipeline_child_jobs = {}
        pipeline._pipeline_starting = set()
        pipeline._pipeline_operations = set()
        pipeline._pipeline_deleting = set()
        pipeline._run_generation = None
        pipeline._wgp = SimpleNamespace(save_path=self.temp_dir.name)
        pipeline._GENERATION_SETTLE_GRACE_S = 10.0

    def tearDown(self):
        pipeline._pipelines = self.originals["pipelines"]
        pipeline._jobs = self.originals["jobs"]
        pipeline._active_gen_states = self.originals["active"]
        pipeline._pipeline_threads = self.originals["threads"]
        pipeline._pipeline_child_jobs = self.originals["child_jobs"]
        pipeline._pipeline_starting = self.originals["starting"]
        pipeline._pipeline_operations = self.originals["operations"]
        pipeline._pipeline_deleting = self.originals["deleting"]
        pipeline._run_generation = self.originals["run_generation"]
        pipeline._wgp = self.originals["wgp"]
        pipeline._GENERATION_SETTLE_GRACE_S = self.originals["settle_grace"]
        self.temp_dir.cleanup()

    def _add_pipeline(self, pid: str = "pipe-1", status: str = "running") -> dict:
        record = {
            "id": pid,
            "status": status,
            "phase": "generating_video",
            "progress": {
                "current": 1, "total": 3, "message": "Generating...",
                "step": 1, "total_steps": 10,
            },
            "clip_plans": [],
            "clip_images": [],
            "output_files": [],
            "created_at": time.time(),
            "params": {"pipeline_type": "music_video"},
            "pause_reason": None,
            "out_dir": self.temp_dir.name,
        }
        pipeline._pipelines[pid] = record
        return record

    def test_abort_marks_queued_and_running_children_cancelled(self):
        pid = "pipe-children"
        self._add_pipeline(pid)
        queued = {
            "id": "queued", "status": "queued", "message": "Queued",
            "params": {"_director_pipeline_id": pid},
        }
        running = {
            "id": "running", "status": "queued", "message": "Queued",
            "params": {"_director_pipeline_id": pid},
        }
        unrelated = {
            "id": "other", "status": "queued", "message": "Queued",
            "params": {"_director_pipeline_id": "another-pipeline"},
        }
        pipeline._jobs.update({
            "queued": queued, "running": running, "other": unrelated,
        })

        state = {"abort": False}
        interrupt = Mock()
        self.assertTrue(try_start(running))
        self.assertTrue(register_abort_state(
            running,
            "running",
            pipeline._active_gen_states,
            state,
            interrupt_model=interrupt,
        ))
        try:
            pipeline._abort_pipeline_jobs(pid)
            self.assertEqual(queued["status"], "cancelled")
            self.assertEqual(running["status"], "cancelled")
            self.assertEqual(unrelated["status"], "queued")
            self.assertTrue(state["abort"])
            interrupt.assert_called_once_with()
        finally:
            unregister_abort_state(
                "running", pipeline._active_gen_states, state,
            )

    def test_stop_is_persisted_and_terminal_updates_are_rejected(self):
        pid = "pipe-stop"
        record = self._add_pipeline(pid)
        self.assertTrue(pipeline.stop_pipeline(pid))
        self.assertEqual(record["status"], "cancelled")
        self.assertEqual(record["progress"]["message"], "Cancelled")

        state_path = os.path.join(
            self.temp_dir.name, f"_director_pipeline_{pid}.json",
        )
        with open(state_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["status"], "cancelled")
        self.assertIsNotNone(saved["completed_at"])
        self.assertTrue(record["_state_persisted"])

        for status in ("completed", "failed", "paused"):
            with self.subTest(status=status):
                self.assertFalse(pipeline._update_pipeline(pid, status=status))
                self.assertEqual(record["status"], "cancelled")

        record["clip_plans"] = [{
            "image_prompt": "saved image", "video_prompt": "saved video",
        }]
        self.assertTrue(pipeline._update_pipeline(
            pid,
            output_files=["finished-before-stop.mp4"],
            clip_images=["image-before-stop.png"],
            _clip_keyframes=[["keyframe-before-stop.png"]],
        ))
        self.assertEqual(
            record["output_files"], ["finished-before-stop.mp4"],
        )
        self.assertEqual(record["clip_images"], ["image-before-stop.png"])
        self.assertEqual(
            record["_clip_keyframes"], [["keyframe-before-stop.png"]],
        )
        self.assertTrue(pipeline._save_pipeline_state(pid))
        settled = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(
            settled["clips"][0]["start_image_filename"],
            "image-before-stop.png",
        )
        self.assertEqual(
            settled["clips"][0]["keyframe_filenames"],
            ["keyframe-before-stop.png"],
        )

    def test_stop_does_not_replace_an_existing_terminal_result(self):
        for terminal in ("completed", "failed", "cancelled"):
            with self.subTest(terminal=terminal):
                pid = f"pipe-{terminal}"
                record = self._add_pipeline(pid, terminal)
                self.assertFalse(pipeline.stop_pipeline(pid))
                self.assertEqual(record["status"], terminal)

    def test_submit_wait_settles_cancelled_child_before_returning_outputs(self):
        pid = "pipe-late-output"
        self._add_pipeline(pid, "cancelled")
        published = threading.Event()

        def fake_generation(job_id: str):
            time.sleep(0.03)
            record_job_outputs(
                pipeline._jobs[job_id],
                ["clip-0-window-1.mp4", "clip-0-window-2.mp4"],
                clip_output_files={0: "clip-0-window-2.mp4"},
            )
            published.set()

        pipeline._run_generation = fake_generation
        outputs = pipeline._submit_and_wait(
            {"_director_pipeline_id": pid}, timeout_s=1,
            out_dir=self.temp_dir.name,
        )
        self.assertTrue(published.is_set())
        self.assertEqual(outputs, ["clip-0-window-2.mp4"])

    def test_detached_rerun_can_run_for_a_cancelled_parent_pipeline(self):
        pid = "pipe-cancelled-rerun"
        self._add_pipeline(pid, "cancelled")

        def fake_generation(job_id: str):
            job = pipeline._jobs[job_id]
            if not try_start(job):
                return
            record_job_outputs(job, ["rerun.png"])
            finish_job(job, "completed", message="Done")

        pipeline._run_generation = fake_generation
        outputs = pipeline._submit_and_wait(
            {
                "_director_pipeline_id": pid,
                "_director_detached_operation": True,
            },
            timeout_s=1,
            out_dir=self.temp_dir.name,
        )
        self.assertEqual(outputs, ["rerun.png"])

    def test_cancelled_detached_rerun_raises_with_settled_output(self):
        pid = "pipe-cancelled-detached"
        self._add_pipeline(pid, "completed")

        def fake_generation(job_id: str):
            job = pipeline._jobs[job_id]
            if not try_start(job):
                return
            request_cancel(job)
            record_job_outputs(job, ["cancelled-rerun.png"])

        pipeline._run_generation = fake_generation
        with self.assertRaises(
            pipeline.GenerationCancelledError,
        ) as caught:
            pipeline._submit_and_wait(
                {
                    "_director_pipeline_id": pid,
                    "_director_detached_operation": True,
                },
                timeout_s=1,
                out_dir=self.temp_dir.name,
            )
        self.assertEqual(
            list(caught.exception.output_files),
            ["cancelled-rerun.png"],
        )

    def test_cancelled_image_rerun_does_not_replace_saved_clip(self):
        pid = "pipe-rerun-preserve"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        record["clip_images"] = ["original.png"]
        self.assertTrue(pipeline._save_pipeline_state(pid))
        cancelled = pipeline.GenerationCancelledError(
            pipeline._DirectorOutputs(["cancelled-rerun.png"]),
        )

        with patch.object(
            pipeline, "_submit_and_wait", side_effect=cancelled,
        ):
            with self.assertRaises(pipeline.GenerationCancelledError):
                pipeline.rerun_clip_image(
                    self.temp_dir.name, pid, 0,
                )

        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(
            saved["clips"][0]["start_image_filename"],
            "original.png",
        )

    def test_submit_timeout_cancels_and_settles_child_before_raising(self):
        settled = threading.Event()

        def fake_generation(job_id: str):
            job = pipeline._jobs[job_id]
            if not try_start(job):
                return
            while not is_cancel_requested(job):
                time.sleep(0.001)
            record_job_outputs(job, ["late-timeout-output.mp4"])
            settled.set()

        pipeline._run_generation = fake_generation
        with self.assertRaises(pipeline._GenerationTimeoutError) as caught:
            pipeline._submit_and_wait(
                {}, timeout_s=0.03, out_dir=self.temp_dir.name,
            )

        self.assertTrue(settled.is_set())
        self.assertEqual(
            list(caught.exception.output_files),
            ["late-timeout-output.mp4"],
        )
        timed_out_job = next(iter(pipeline._jobs.values()))
        self.assertEqual(timed_out_job["status"], "cancelled")

    def test_timeout_is_bounded_while_child_lease_blocks_mutations(self):
        pid = "pipe-stuck-child"
        self._add_pipeline(pid, "completed")
        self.assertTrue(pipeline._save_pipeline_state(pid))
        entered = threading.Event()
        release = threading.Event()

        def non_cooperative_generation(job_id: str):
            job = pipeline._jobs[job_id]
            if not try_start(job):
                return
            entered.set()
            release.wait(timeout=2)

        pipeline._run_generation = non_cooperative_generation
        pipeline._GENERATION_SETTLE_GRACE_S = 0.02
        started = time.monotonic()
        with self.assertRaises(pipeline._GenerationTimeoutError):
            pipeline._submit_and_wait(
                {"_director_pipeline_id": pid},
                timeout_s=0.02,
                out_dir=self.temp_dir.name,
            )
        elapsed = time.monotonic() - started

        self.assertTrue(entered.is_set())
        self.assertLess(elapsed, 0.5)
        self.assertTrue(pipeline._pipeline_child_jobs.get(pid))
        self.assertTrue(pipeline.any_pipeline_active())
        self.assertFalse(pipeline._claim_pipeline_operation(pid))
        self.assertEqual(
            pipeline.delete_pipeline(self.temp_dir.name, pid),
            {"ok": False, "error": "running"},
        )
        self.assertEqual(
            pipeline.resume_pipeline(pid, self.temp_dir.name),
            (False, "Pipeline is already running."),
        )

        release.set()
        deadline = time.time() + 1
        while pipeline._pipeline_child_jobs.get(pid) and time.time() < deadline:
            time.sleep(0.005)
        self.assertFalse(pipeline._pipeline_child_jobs.get(pid))
        self.assertTrue(pipeline._claim_pipeline_operation(pid))
        pipeline._release_pipeline_operation(pid)

    def test_cancelled_detached_wait_is_bounded_and_keeps_child_lease(self):
        pid = "pipe-stuck-rerun"
        self._add_pipeline(pid, "completed")
        entered = threading.Event()
        release = threading.Event()

        def non_cooperative_generation(job_id: str):
            job = pipeline._jobs[job_id]
            if not try_start(job):
                return
            request_cancel(job)
            entered.set()
            release.wait(timeout=2)

        pipeline._run_generation = non_cooperative_generation
        pipeline._GENERATION_SETTLE_GRACE_S = 0.02
        started = time.monotonic()
        with self.assertRaises(pipeline.GenerationCancelledError):
            pipeline._submit_and_wait(
                {
                    "_director_pipeline_id": pid,
                    "_director_detached_operation": True,
                },
                timeout_s=1,
                out_dir=self.temp_dir.name,
            )
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(entered.is_set())
        self.assertTrue(pipeline._pipeline_child_jobs.get(pid))
        self.assertFalse(pipeline._claim_pipeline_operation(pid))

        release.set()
        deadline = time.time() + 1
        while pipeline._pipeline_child_jobs.get(pid) and time.time() < deadline:
            time.sleep(0.005)
        self.assertFalse(pipeline._pipeline_child_jobs.get(pid))

    def test_generation_child_lease_clears_when_thread_start_fails(self):
        pid = "pipe-child-start-failure"
        self._add_pipeline(pid, "completed")
        with patch.object(
            threading.Thread, "start", side_effect=RuntimeError("start failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                pipeline._submit_and_wait(
                    {"_director_pipeline_id": pid},
                    timeout_s=1,
                    out_dir=self.temp_dir.name,
                )
        self.assertFalse(pipeline._pipeline_child_jobs.get(pid))

    def test_start_image_timeout_aborts_phase_before_next_generation(self):
        pid = "pipe-image-timeout"
        self._add_pipeline(pid, "running")
        ref_path = os.path.join(self.temp_dir.name, "reference.png")
        with open(ref_path, "wb") as handle:
            handle.write(b"image")
        timed_out = pipeline._GenerationTimeoutError(
            pipeline._DirectorOutputs([]),
        )
        plans = [
            {"image_prompt": "shot one"},
            {"image_prompt": "shot two"},
        ]

        with patch.object(
            pipeline, "_submit_and_wait", side_effect=timed_out,
        ) as submit:
            with self.assertRaises(pipeline._GenerationTimeoutError):
                pipeline._run_image_generation(
                    pid,
                    {"reference_image_path": ref_path},
                    plans,
                    out_dir=self.temp_dir.name,
                )

        self.assertEqual(submit.call_count, 1)

    def test_keyframe_timeout_aborts_phase_before_next_generation(self):
        pid = "pipe-keyframe-timeout"
        self._add_pipeline(pid, "running")
        ref_path = os.path.join(self.temp_dir.name, "reference.png")
        with open(ref_path, "wb") as handle:
            handle.write(b"image")
        timed_out = pipeline._GenerationTimeoutError(
            pipeline._DirectorOutputs([]),
        )
        plan = {
            "image_prompt": "start",
            "keyframe_prompts": ["middle", "end"],
        }

        with patch.object(
            pipeline,
            "_submit_and_wait",
            side_effect=[["start.png"], timed_out],
        ) as submit:
            with self.assertRaises(pipeline._GenerationTimeoutError):
                pipeline._run_image_generation(
                    pid,
                    {"reference_image_path": ref_path},
                    [plan],
                    out_dir=self.temp_dir.name,
                )

        self.assertEqual(submit.call_count, 2)

    def test_cancelled_partial_video_prefix_maps_to_dashboard_clips(self):
        pid = "pipe-partial"
        record = self._add_pipeline(pid, "cancelled")
        record["params"]["seamless"] = False
        record["clip_plans"] = [
            {"image_prompt": f"image {i}", "video_prompt": f"video {i}"}
            for i in range(3)
        ]
        record["output_files"] = ["clip-1.mp4"]

        self.assertTrue(pipeline._save_pipeline_state(pid))
        state = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(
            [clip["video_filename"] for clip in state["clips"]],
            ["clip-1.mp4", None, None],
        )

    def test_director_outputs_use_final_window_per_explicit_clip_index(self):
        job = {
            "status": "completed",
            "output_files": [
                "clip0-window1.mp4",
                "clip0-window2.mp4",
                "clip1.mp4",
                "joined_MULTICLIP.WEBM",
            ],
            "clip_output_files": {
                "0": "clip0-window2.mp4",
                "2": "clip1.mp4",
            },
            "join_output_file": "joined_MULTICLIP.WEBM",
        }
        outputs = pipeline._director_job_outputs(job)
        self.assertEqual(
            outputs,
            [
                "clip0-window2.mp4",
                "clip1.mp4",
                "joined_MULTICLIP.WEBM",
            ],
        )
        self.assertEqual(
            pipeline._clip_video_slots(outputs, 3),
            ["clip0-window2.mp4", None, "clip1.mp4"],
        )

    def test_cancel_race_completion_fallback_persists_exact_clip_mapping(self):
        source = inspect.getsource(pipeline._run_pipeline)
        fallback = source.split("if not completed:", 1)[1].split(
            "_save_pipeline_state(pid)", 1,
        )[0]
        self.assertIn("output_files=output_files or []", fallback)
        self.assertIn(
            "_clip_video_files=completed_clip_videos",
            fallback,
        )

    def test_concurrent_saves_leave_latest_live_snapshot_as_valid_json(self):
        pid = "pipe-writers"
        record = self._add_pipeline(pid)
        workers = 8
        barrier = threading.Barrier(workers)
        failures: list[str] = []

        def save_value(index: int):
            try:
                barrier.wait(timeout=2)
                pipeline._update_pipeline(
                    pid, output_files=[f"clip-{index}.mp4"],
                )
                if not pipeline._save_pipeline_state(pid):
                    failures.append(f"save {index} failed")
            except Exception as exc:
                failures.append(str(exc))

        threads = [
            threading.Thread(target=save_value, args=(i,))
            for i in range(workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())

        self.assertEqual(failures, [])
        state = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(state["output_files"], record["output_files"])

    def test_failed_atomic_replace_is_reported_and_preserves_old_state(self):
        pid = "pipe-write-failure"
        record = self._add_pipeline(pid)
        record["output_files"] = ["old.mp4"]
        self.assertTrue(pipeline._save_pipeline_state(pid))
        record["output_files"] = ["new.mp4"]

        with patch.object(
            pipeline.os, "replace", side_effect=OSError("replace failed"),
        ):
            self.assertFalse(pipeline._save_pipeline_state(pid))

        state = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(state["output_files"], ["old.mp4"])
        self.assertFalse(any(
            name.endswith(".tmp") for name in os.listdir(self.temp_dir.name)
        ))

    def test_stop_exposes_when_cancelled_state_could_not_be_persisted(self):
        pid = "pipe-stop-write-failure"
        record = self._add_pipeline(pid)
        with patch.object(
            pipeline.os, "replace", side_effect=OSError("replace failed"),
        ):
            self.assertTrue(pipeline.stop_pipeline(pid))
        self.assertEqual(record["status"], "cancelled")
        self.assertFalse(record["_state_persisted"])

    def test_delete_refuses_cancelled_pipeline_until_worker_settles(self):
        pid = "pipe-settling"
        self._add_pipeline(pid, "cancelled")
        self.assertTrue(pipeline._save_pipeline_state(pid))
        release = threading.Event()
        worker = threading.Thread(target=lambda: release.wait(timeout=2))
        pipeline._pipeline_threads[pid] = worker
        worker.start()
        try:
            result = pipeline.delete_pipeline(self.temp_dir.name, pid)
            self.assertEqual(result, {"ok": False, "error": "running"})
            self.assertTrue(pipeline.any_pipeline_active())
        finally:
            release.set()
            worker.join(timeout=1)
            pipeline._pipeline_threads.pop(pid, None)

    def test_delete_cancelled_pipeline_sweeps_superseded_window_outputs(self):
        pid = "pipe-cancelled-windows"
        record = self._add_pipeline(pid, "cancelled")
        record["output_files"] = ["clip0-window2.mp4"]
        self.assertTrue(pipeline._save_pipeline_state(pid))

        filenames = ("clip0-window1.mkv", "clip0-window2.mp4")
        for filename in filenames:
            media_path = os.path.join(self.temp_dir.name, filename)
            sidecar_path = os.path.splitext(media_path)[0] + ".meta.json"
            with open(media_path, "wb") as handle:
                handle.write(b"video")
            with open(sidecar_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "director_pipeline_id": pid,
                    "output_filename": filename,
                }, handle)
            artifact_base = os.path.splitext(media_path)[0]
            with open(artifact_base + ".json", "w", encoding="utf-8") as handle:
                json.dump({"metadata": True}, handle)
            with open(artifact_base + ".zip", "wb") as handle:
                handle.write(b"alpha frames")

        result = pipeline.delete_pipeline(self.temp_dir.name, pid)

        self.assertTrue(result["ok"])
        for filename in filenames:
            media_path = os.path.join(self.temp_dir.name, filename)
            self.assertFalse(os.path.exists(media_path))
            self.assertFalse(os.path.exists(
                os.path.splitext(media_path)[0] + ".meta.json",
            ))
            self.assertFalse(os.path.exists(
                os.path.splitext(media_path)[0] + ".json",
            ))
            self.assertFalse(os.path.exists(
                os.path.splitext(media_path)[0] + ".zip",
            ))

    def test_delete_reports_failure_when_state_file_cannot_be_removed(self):
        pid = "pipe-locked-state"
        self._add_pipeline(pid, "completed")
        self.assertTrue(pipeline._save_pipeline_state(pid))
        state_path = pipeline._find_pipeline_file(self.temp_dir.name, pid)

        with patch(
            "services.win_safe_files.safe_delete",
            return_value={"deleted": False, "reason": "locked"},
        ):
            result = pipeline.delete_pipeline(self.temp_dir.name, pid)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "state_file_locked")
        self.assertTrue(os.path.isfile(state_path))
        self.assertIn(pid, pipeline._pipelines)

    def test_locked_media_preserves_sidecar_and_state_for_delete_retry(self):
        pid = "pipe-locked-media"
        record = self._add_pipeline(pid, "completed")
        filename = "locked.mp4"
        record["output_files"] = [filename]
        self.assertTrue(pipeline._save_pipeline_state(pid))
        state_path = pipeline._find_pipeline_file(self.temp_dir.name, pid)
        media_path = os.path.join(self.temp_dir.name, filename)
        sidecar_path = os.path.splitext(media_path)[0] + ".meta.json"
        with open(media_path, "wb") as handle:
            handle.write(b"video")
        with open(sidecar_path, "w", encoding="utf-8") as handle:
            json.dump({
                "director_pipeline_id": pid,
                "output_filename": filename,
            }, handle)

        from services.win_safe_files import safe_delete as real_safe_delete

        def lock_media_only(path, **kwargs):
            if os.path.normcase(path) == os.path.normcase(media_path):
                return {"deleted": False, "reason": "locked"}
            return real_safe_delete(path, **kwargs)

        with patch(
            "services.win_safe_files.safe_delete",
            side_effect=lock_media_only,
        ):
            result = pipeline.delete_pipeline(self.temp_dir.name, pid)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "media_locked")
        self.assertTrue(os.path.isfile(media_path))
        self.assertTrue(os.path.isfile(sidecar_path))
        self.assertTrue(os.path.isfile(state_path))
        self.assertIn(pid, pipeline._pipelines)

    def test_concurrent_resume_requests_are_atomically_reserved(self):
        pid = "pipe-resume-race"
        entered = threading.Event()
        release = threading.Event()
        first_result: list[tuple[bool, str]] = []

        def reserved_resume(_pid: str, _out_dir: str):
            entered.set()
            release.wait(timeout=2)
            return True, "resumed"

        with patch.object(
            pipeline, "_resume_pipeline_reserved", side_effect=reserved_resume,
        ):
            first = threading.Thread(
                target=lambda: first_result.append(
                    pipeline.resume_pipeline(pid, self.temp_dir.name),
                ),
            )
            first.start()
            self.assertTrue(entered.wait(timeout=1))
            self.assertEqual(
                pipeline.resume_pipeline(pid, self.temp_dir.name),
                (False, "Pipeline is already running."),
            )
            release.set()
            first.join(timeout=1)

        self.assertFalse(first.is_alive())
        self.assertEqual(first_result, [(True, "resumed")])
        self.assertNotIn(pid, pipeline._pipeline_starting)

    def test_dashboard_operation_blocks_delete_resume_and_tag_updates(self):
        pid = "pipe-dashboard-operation"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "image", "video_prompt": "video",
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))
        self.assertTrue(pipeline._claim_pipeline_operation(pid))
        try:
            self.assertEqual(
                pipeline.delete_pipeline(self.temp_dir.name, pid),
                {"ok": False, "error": "running"},
            )
            self.assertEqual(
                pipeline.resume_pipeline(pid, self.temp_dir.name),
                (False, "Pipeline is already running."),
            )
            with self.assertRaises(pipeline.PipelineBusyError):
                pipeline.update_clip_tag(
                    self.temp_dir.name, pid, 0, "keep",
                )
        finally:
            pipeline._release_pipeline_operation(pid)

    def test_active_status_blocks_operation_before_thread_registration(self):
        pid = "pipe-start-gap"
        self._add_pipeline(pid, "running")
        self.assertFalse(pipeline._claim_pipeline_operation(pid))

    def test_delete_reservation_blocks_late_dashboard_operation(self):
        pid = "pipe-delete-first"
        self._add_pipeline(pid, "completed")
        self.assertTrue(pipeline._claim_pipeline_delete(pid))
        try:
            self.assertFalse(pipeline._claim_pipeline_operation(pid))
            self.assertEqual(
                pipeline.resume_pipeline(pid, self.temp_dir.name),
                (False, "Pipeline is already running."),
            )
        finally:
            pipeline._release_pipeline_delete(pid)

    def test_worker_start_failure_marks_pipeline_failed_and_untracks_it(self):
        pid = "pipe-start-failure"
        record = self._add_pipeline(pid)
        with patch.object(
            pipeline.threading.Thread,
            "start",
            side_effect=RuntimeError("thread unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                pipeline._start_pipeline_worker(pid)

        self.assertEqual(record["status"], "failed")
        self.assertIn("thread unavailable", record["error"])
        self.assertNotIn(pid, pipeline._pipeline_threads)
        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(saved["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
