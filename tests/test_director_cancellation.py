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
            "repairs": pipeline._pipeline_repairs,
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
        pipeline._pipeline_repairs = {}
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
        pipeline._pipeline_repairs = self.originals["repairs"]
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

    def _write_media(self, filename: str, payload: bytes = b"media") -> str:
        filepath = os.path.join(self.temp_dir.name, filename)
        with open(filepath, "wb") as handle:
            handle.write(payload)
        return filepath

    def _wait_for_repair_terminal(
        self, pid: str, timeout: float = 3.0,
    ) -> dict:
        deadline = time.monotonic() + timeout
        last_repair = None
        while time.monotonic() < deadline:
            state = pipeline.load_pipeline_state(self.temp_dir.name, pid)
            last_repair = (state or {}).get("repair")
            with pipeline._pipeline_lock:
                tracked = pid in pipeline._pipeline_repairs
                leased = pid in pipeline._pipeline_operations
            if (
                isinstance(last_repair, dict)
                and last_repair.get("status")
                not in pipeline._REPAIR_ACTIVE_STATUSES
                and not tracked
                and not leased
            ):
                return last_repair
            time.sleep(0.01)
        self.fail(
            f"Repair {pid} did not settle; last state was {last_repair!r}"
        )

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

    def test_submit_timeout_tracks_inactivity_not_total_batch_time(self):
        def fake_generation(job_id: str):
            job = pipeline._jobs[job_id]
            if not try_start(job):
                return
            for step in range(1, 7):
                job["step"] = step
                job["total_steps"] = 6
                job["phase"] = f"Clip {step}/6"
                time.sleep(0.04)
            record_job_outputs(job, ["long-active-batch.mp4"])
            finish_job(job, "completed", message="Done")

        pipeline._run_generation = fake_generation
        started = time.monotonic()
        outputs = pipeline._submit_and_wait(
            {}, timeout_s=0.08, out_dir=self.temp_dir.name,
        )

        self.assertGreater(time.monotonic() - started, 0.20)
        self.assertEqual(outputs, ["long-active-batch.mp4"])

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

    def test_no_reference_run_persists_anchor_and_conditions_every_start(self):
        pid = "pipe-generated-anchor"
        record = self._add_pipeline(pid, "running")
        params = {
            "scene_description": "A singer performs beneath neon lights",
            "image_model": "flux2_klein_9b",
        }
        record["params"] = params
        plans = [
            {"image_prompt": "wide portrait of the singer"},
            {"image_prompt": "close portrait of the same singer"},
        ]
        generated = iter(["anchor.jpg", "shot-1.jpg", "shot-2.jpg"])
        submitted: list[dict] = []

        def fake_submit(gen_params, **_kwargs):
            submitted.append({
                **gen_params,
                "image_refs": list(gen_params.get("image_refs") or []),
            })
            filename = next(generated)
            with open(os.path.join(self.temp_dir.name, filename), "wb") as f:
                f.write(b"image")
            return [filename]

        with patch.object(
            pipeline, "_submit_and_wait", side_effect=fake_submit,
        ):
            clip_images, clip_keyframes = pipeline._run_image_generation(
                pid, params, plans, out_dir=self.temp_dir.name,
            )

        anchor_path = os.path.realpath(
            os.path.join(self.temp_dir.name, "anchor.jpg"),
        )
        self.assertEqual(clip_images, ["shot-1.jpg", "shot-2.jpg"])
        self.assertEqual(clip_keyframes, [[], []])
        self.assertEqual(len(submitted), 3)
        self.assertEqual(submitted[0]["image_refs"], [])
        self.assertEqual(submitted[0]["video_prompt_type"], "")
        self.assertIn("definitive cinematic character anchor", submitted[0]["prompt"])
        for request in submitted[1:]:
            self.assertEqual(request["image_refs"], [anchor_path])
            self.assertEqual(request["video_prompt_type"], "KI")
        self.assertEqual(
            params["generated_reference_image_filename"], "anchor.jpg",
        )

        record["clip_plans"] = plans
        record["clip_images"] = clip_images
        self.assertTrue(pipeline._save_pipeline_state(pid))
        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(
            saved["generated_reference_image_filename"], "anchor.jpg",
        )
        self.assertEqual(
            [clip["start_image_filename"] for clip in saved["clips"]],
            clip_images,
        )

    def test_generated_anchor_uses_character_refs_and_profiles_not_locations(self):
        pid = "pipe-character-anchor"
        self._add_pipeline(pid, "running")
        character_ref = os.path.join(self.temp_dir.name, "character.jpg")
        location_ref = os.path.join(self.temp_dir.name, "location.jpg")
        for path in (character_ref, location_ref):
            with open(path, "wb") as handle:
                handle.write(b"image")
        params = {
            "scene_description": "An empty moonlit train platform",
            "image_model": "flux2_klein_9b",
            "character_ref_paths": [character_ref],
            "location_ref_paths": [location_ref],
            "characters": [{
                "name": "Mara",
                "description": "a tall woman with silver braids",
            }],
        }
        plans = [{"image_prompt": "wide view of the empty platform"}]
        submitted: list[dict] = []

        def fake_submit(gen_params, **_kwargs):
            submitted.append({
                **gen_params,
                "image_refs": list(gen_params.get("image_refs") or []),
            })
            filename = "anchor.jpg" if len(submitted) == 1 else "shot.jpg"
            with open(os.path.join(self.temp_dir.name, filename), "wb") as handle:
                handle.write(b"image")
            return [filename]

        with patch.object(
            pipeline, "_submit_and_wait", side_effect=fake_submit,
        ):
            pipeline._run_image_generation(
                pid, params, plans, out_dir=self.temp_dir.name,
            )

        anchor_path = os.path.join(self.temp_dir.name, "anchor.jpg")
        self.assertEqual(submitted[0]["image_refs"], [character_ref])
        self.assertNotIn(location_ref, submitted[0]["image_refs"])
        self.assertIn(
            "Mara: a tall woman with silver braids", submitted[0]["prompt"],
        )
        self.assertIn(
            "definitive identity and appearance source", submitted[0]["prompt"],
        )
        self.assertEqual(
            submitted[1]["image_refs"],
            [anchor_path, character_ref, location_ref],
        )

    def test_user_reference_skips_generated_anchor(self):
        pid = "pipe-user-reference"
        self._add_pipeline(pid, "running")
        reference_path = os.path.join(self.temp_dir.name, "user-ref.jpg")
        with open(reference_path, "wb") as handle:
            handle.write(b"image")
        params = {
            "reference_image_path": reference_path,
            "image_model": "flux2_klein_9b",
        }
        plans = [
            {"image_prompt": "first shot"},
            {"image_prompt": "second shot"},
        ]
        generated = iter(["shot-1.jpg", "shot-2.jpg"])
        submitted: list[dict] = []

        def fake_submit(gen_params, **_kwargs):
            submitted.append({
                **gen_params,
                "image_refs": list(gen_params.get("image_refs") or []),
            })
            filename = next(generated)
            with open(os.path.join(self.temp_dir.name, filename), "wb") as f:
                f.write(b"image")
            return [filename]

        with patch.object(
            pipeline, "_submit_and_wait", side_effect=fake_submit,
        ):
            clip_images, _ = pipeline._run_image_generation(
                pid, params, plans, out_dir=self.temp_dir.name,
            )

        self.assertEqual(clip_images, ["shot-1.jpg", "shot-2.jpg"])
        self.assertEqual(len(submitted), 2)
        for request in submitted:
            self.assertEqual(request["image_refs"], [reference_path])
            self.assertEqual(request["video_prompt_type"], "KI")
        self.assertNotIn("generated_reference_image_filename", params)

    def test_reference_free_rerun_bootstraps_and_reuses_saved_anchor(self):
        pid = "pipe-rerun-anchor"
        record = self._add_pipeline(pid, "completed")
        record["params"] = {
            "pipeline_type": "music_video",
            "image_model": "flux2_klein_9b",
        }
        record["clip_plans"] = [
            {"image_prompt": "first portrait", "video_prompt": "first motion"},
            {"image_prompt": "second portrait", "video_prompt": "second motion"},
        ]
        self.assertTrue(pipeline._save_pipeline_state(pid))
        generated = iter(["bootstrap.jpg", "second.jpg"])
        submitted: list[dict] = []

        def fake_submit(gen_params, **_kwargs):
            submitted.append({
                **gen_params,
                "image_refs": list(gen_params.get("image_refs") or []),
            })
            filename = next(generated)
            with open(os.path.join(self.temp_dir.name, filename), "wb") as f:
                f.write(b"image")
            return [filename]

        with patch.object(
            pipeline, "_submit_and_wait", side_effect=fake_submit,
        ):
            pipeline.rerun_clip_image(self.temp_dir.name, pid, 0)
            pipeline.rerun_clip_image(self.temp_dir.name, pid, 1)

        bootstrap_path = os.path.join(self.temp_dir.name, "bootstrap.jpg")
        self.assertEqual(submitted[0]["image_refs"], [])
        self.assertEqual(submitted[0]["video_prompt_type"], "")
        self.assertEqual(submitted[1]["image_refs"], [bootstrap_path])
        self.assertEqual(submitted[1]["video_prompt_type"], "KI")
        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(
            saved["generated_reference_image_filename"], "bootstrap.jpg",
        )
        self.assertEqual(
            [clip["start_image_filename"] for clip in saved["clips"]],
            ["bootstrap.jpg", "second.jpg"],
        )
        self.assertEqual(
            saved["_params_snapshot"]["generated_reference_image_filename"],
            "bootstrap.jpg",
        )

    def test_image_rerun_marks_video_stale_until_video_is_replaced(self):
        pid = "pipe-stale-video"
        record = self._add_pipeline(pid, "completed")
        record["params"] = {
            "pipeline_type": "music_video",
            "image_model": "flux2_klein_9b",
            "video_model": "ltx2_22B_distilled_1_1",
        }
        record["clip_plans"] = [{
            "image_prompt": "new portrait", "video_prompt": "new motion",
        }]
        record["clip_images"] = ["old-start.jpg"]
        record["_clip_video_files"] = ["old-video.mp4"]
        record["output_files"] = ["old-video.mp4"]
        for filename in ("old-start.jpg", "old-video.mp4"):
            with open(os.path.join(self.temp_dir.name, filename), "wb") as f:
                f.write(b"media")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        def replace_image(_gen_params, **_kwargs):
            with open(
                os.path.join(self.temp_dir.name, "new-start.jpg"), "wb",
            ) as handle:
                handle.write(b"image")
            return ["new-start.jpg"]

        with patch.object(
            pipeline, "_submit_and_wait", side_effect=replace_image,
        ):
            pipeline.rerun_clip_image(self.temp_dir.name, pid, 0)

        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(saved["clips"][0]["video_filename"], "old-video.mp4")
        self.assertTrue(saved["clips"][0]["video_stale"])

        with patch.object(
            pipeline, "_submit_and_wait", return_value=["new-video.mp4"],
        ):
            pipeline.rerun_clip_video(self.temp_dir.name, pid, 0)

        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(saved["clips"][0]["video_filename"], "new-video.mp4")
        self.assertFalse(saved["clips"][0]["video_stale"])
        self.assertEqual(
            saved["output_files"],
            ["old-video.mp4", "new-video.mp4"],
        )

    def test_image_rerun_marks_backfilled_legacy_video_stale(self):
        pid = "pipe-legacy-image-rerun"
        record = self._add_pipeline(pid, "completed")
        record["params"].update({
            "seamless": False,
            "image_model": "flux2_klein_9b",
        })
        record["clip_plans"] = [{
            "image_prompt": "new portrait", "video_prompt": "old motion",
        }]
        record["clip_images"] = ["old-start.jpg"]
        record["output_files"] = ["old-video.mp4"]
        for filename in ("old-start.jpg", "old-video.mp4"):
            with open(os.path.join(self.temp_dir.name, filename), "wb") as handle:
                handle.write(b"media")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        state_path = pipeline._find_pipeline_file(self.temp_dir.name, pid)
        with open(state_path, "r", encoding="utf-8") as handle:
            raw_state = json.load(handle)
        raw_state["clips"][0]["video_filename"] = None
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(raw_state, handle)

        def replace_image(_gen_params, **_kwargs):
            with open(
                os.path.join(self.temp_dir.name, "new-start.jpg"), "wb",
            ) as handle:
                handle.write(b"image")
            return ["new-start.jpg"]

        with patch.object(
            pipeline, "_submit_and_wait", side_effect=replace_image,
        ):
            pipeline.rerun_clip_image(self.temp_dir.name, pid, 0)

        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(saved["clips"][0]["video_filename"], "old-video.mp4")
        self.assertTrue(saved["clips"][0]["video_stale"])

    def test_video_rerun_preserves_other_backfilled_legacy_clip_mappings(self):
        pid = "pipe-legacy-video-rerun"
        record = self._add_pipeline(pid, "completed")
        record["params"].update({
            "seamless": False,
            "video_model": "ltx2_22B_distilled_1_1",
        })
        record["clip_plans"] = [
            {"image_prompt": "one", "video_prompt": "motion one"},
            {"image_prompt": "two", "video_prompt": "motion two"},
        ]
        record["clip_images"] = ["start-one.jpg", "start-two.jpg"]
        record["output_files"] = ["old-one.mp4", "old-two.mp4"]
        for filename in (
            "start-one.jpg", "start-two.jpg", "old-one.mp4", "old-two.mp4",
        ):
            with open(os.path.join(self.temp_dir.name, filename), "wb") as handle:
                handle.write(b"media")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        state_path = pipeline._find_pipeline_file(self.temp_dir.name, pid)
        with open(state_path, "r", encoding="utf-8") as handle:
            raw_state = json.load(handle)
        for clip in raw_state["clips"]:
            clip["video_filename"] = None
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(raw_state, handle)

        with patch.object(
            pipeline, "_submit_and_wait", return_value=["new-one.mp4"],
        ):
            pipeline.rerun_clip_video(self.temp_dir.name, pid, 0)

        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(
            [clip["video_filename"] for clip in saved["clips"]],
            ["new-one.mp4", "old-two.mp4"],
        )
        self.assertEqual(
            saved["output_files"],
            ["old-one.mp4", "old-two.mp4", "new-one.mp4"],
        )

    def test_video_rerun_reuses_full_director_carried_frame_schedule(self):
        pid = "pipe-rerun-frame-schedule"
        record = self._add_pipeline(pid, "completed")
        audio_path = self._write_media("song.wav", b"audio")
        record["params"].update({
            "seamless": False,
            "video_model": "ltx2_22B_distilled_1_1",
            "video_params": {"num_inference_steps": 13},
            "audio_path": audio_path,
            # The model definition must win over a stale frontend fps.
            "fps": 16,
        })
        planned = [
            {"start": 2, "end": 30},
            {"start": 30, "end": 50},
            {"start": 50, "end": 69.613},
            {"start": 69.613, "end": 89},
            {"start": 89, "end": 98},
            {"start": 98, "end": 120},
        ]
        record["_planned_clips"] = planned
        record["clip_plans"] = [
            {"image_prompt": str(i), "video_prompt": f"motion {i}"}
            for i in range(len(planned))
        ]
        record["clip_images"] = [
            f"start-{i}.jpg" for i in range(len(planned))
        ]
        for filename in record["clip_images"]:
            self._write_media(filename, b"image")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        submitted = []
        audio_slices = []
        pipeline._wgp = SimpleNamespace(
            save_path=self.temp_dir.name,
            get_model_def=lambda _model: {"fps": 25},
            get_model_min_frames_and_step=lambda _model: (17, 8, 8),
        )

        with (
            patch.object(
                pipeline,
                "_slice_audio_segment",
                side_effect=lambda *args: audio_slices.append(args),
            ),
            patch.object(
                pipeline,
                "_submit_and_wait",
                side_effect=lambda params, **_kwargs: (
                    submitted.append(params) or ["replacement.mp4"]
                ),
            ),
        ):
            pipeline.rerun_clip_video(self.temp_dir.name, pid, 5)

        # Full Director's carried schedule is [697, 505, 489, 481, 225, 553].
        # Independent floor quantization used to turn this final request into
        # 545 frames, shortening the joined timeline by another 0.32 seconds.
        self.assertEqual(submitted[0]["video_length"], 553)
        self.assertEqual(submitted[0]["sliding_window_size"], 562)
        self.assertEqual(submitted[0]["num_inference_steps"], 13)
        self.assertEqual(len(audio_slices), 1)
        self.assertAlmostEqual(audio_slices[0][1], 97.88, places=3)
        self.assertAlmostEqual(audio_slices[0][2], 22.12, places=3)

        # Wider model lattices explain the public report's full one-second
        # loss: carry alternates the extra frame block instead of discarding
        # it independently from every regenerated clip.
        self.assertEqual(
            pipeline._quantize_clip_frame_schedule(
                [240, 240, 240, 240], 33, 32,
            ),
            [225, 257, 225, 257],
        )

    def test_ltx25_video_rerun_retains_source_audio_sync_contract(self):
        pid = "pipe-ltx25-rerun-sync"
        record = self._add_pipeline(pid, "completed")
        audio_path = self._write_media("song.wav", b"audio")
        record["params"].update({
            "seamless": False,
            "video_model": "ltx2_25",
            "video_params": {"resolution": "1280x704"},
            "audio_path": audio_path,
            "fps": 24,
        })
        record["_planned_clips"] = [
            {"start": 2, "end": 7, "duration_sec": 5},
        ]
        record["clip_plans"] = [{
            "image_prompt": "singer at microphone",
            "video_prompt": "The lead singer performs into a microphone.",
        }]
        record["clip_images"] = ["start.jpg"]
        self._write_media("start.jpg", b"image")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        submitted: list[dict] = []
        pipeline._wgp = SimpleNamespace(
            save_path=self.temp_dir.name,
            get_model_def=lambda _model: {
                "architecture": "ltx2_25",
                "fps": 24,
            },
            get_model_min_frames_and_step=lambda _model: (17, 8, 8),
        )

        with (
            patch.object(pipeline, "_slice_audio_segment"),
            patch.object(
                pipeline,
                "_submit_and_wait",
                side_effect=lambda params, **_kwargs: (
                    submitted.append(params) or ["replacement.mp4"]
                ),
            ),
        ):
            pipeline.rerun_clip_video(self.temp_dir.name, pid, 0)

        self.assertIn("SOURCE-AUDIO LIP SYNC", submitted[0]["prompt"])
        self.assertIn(
            "lip-syncs every vocal syllable",
            submitted[0]["prompt"],
        )
        self.assertEqual(submitted[0]["audio_prompt_type"], "A")

    def test_standard_video_uses_each_generated_start_image(self):
        pid = "pipe-video-starts"
        self._add_pipeline(pid, "running")
        clip_images = ["shot-1.jpg", "shot-2.jpg"]
        for filename in clip_images:
            with open(os.path.join(self.temp_dir.name, filename), "wb") as f:
                f.write(b"image")
        params = {
            "pipeline_type": "short_film_story",
            "seamless": False,
            "video_model": "ltx2_22B_distilled_1_1",
            "video_params": {
                "resolution": "1280x720",
                "num_inference_steps": 13,
            },
            "fps": 25,
        }
        plans = [
            {"video_prompt": "first motion"},
            {"video_prompt": "second motion"},
        ]
        planned = [
            {"start": 0, "end": 5, "duration_sec": 5},
            {"start": 5, "end": 10, "duration_sec": 5},
        ]
        submitted: list[dict] = []
        pipeline._wgp = SimpleNamespace(
            save_path=self.temp_dir.name,
            server_config={"services": {}},
            get_model_def=lambda _model: {"fps": 25},
            get_model_min_frames_and_step=lambda _model: (17, 8, 8),
        )

        def fake_submit(gen_params, **_kwargs):
            submitted.append(gen_params)
            return ["clip-1.mp4", "clip-2.mp4"]

        with patch.object(
            pipeline, "_submit_and_wait", side_effect=fake_submit,
        ):
            outputs = pipeline._run_video_generation(
                pid,
                params,
                plans,
                planned,
                clip_images,
                out_dir=self.temp_dir.name,
            )

        self.assertEqual(outputs, ["clip-1.mp4", "clip-2.mp4"])
        self.assertEqual(len(submitted), 1)
        self.assertEqual(
            submitted[0]["image_start"],
            [
                os.path.join(self.temp_dir.name, "shot-1.jpg"),
                os.path.join(self.temp_dir.name, "shot-2.jpg"),
            ],
        )
        self.assertEqual(submitted[0]["image_prompt_type"], "S")
        self.assertEqual(submitted[0]["num_inference_steps"], 13)

    def test_standard_video_uses_first_planned_time_as_audio_origin(self):
        pid = "pipe-video-audio-origin"
        self._add_pipeline(pid, "running")
        audio_path = self._write_media("song.wav", b"audio")
        for filename in ("shot-1.jpg", "shot-2.jpg"):
            self._write_media(filename, b"image")
        params = {
            "pipeline_type": "music_video",
            "seamless": False,
            "video_model": "ltx2_22B_distilled_1_1",
            "video_params": {"resolution": "1280x720"},
            "audio_path": audio_path,
            "fps": 25,
        }
        plans = [
            {"video_prompt": "first motion"},
            {"video_prompt": "second motion"},
        ]
        planned = [
            {"start": 2, "end": 7, "duration_sec": 5},
            {"start": 7, "end": 12, "duration_sec": 5},
        ]
        submitted: list[dict] = []
        pipeline._wgp = SimpleNamespace(
            save_path=self.temp_dir.name,
            server_config={"services": {}},
            get_model_def=lambda _model: {"fps": 25},
            get_model_min_frames_and_step=lambda _model: (17, 8, 8),
        )

        with patch.object(
            pipeline,
            "_submit_and_wait",
            side_effect=lambda gen_params, **_kwargs: (
                submitted.append(gen_params) or ["one.mp4", "two.mp4"]
            ),
        ):
            pipeline._run_video_generation(
                pid,
                params,
                plans,
                planned,
                ["shot-1.jpg", "shot-2.jpg"],
                out_dir=self.temp_dir.name,
            )

        self.assertEqual(submitted[0]["audio_frame_offset"], 50)
        self.assertEqual(submitted[0]["multi_clip_audio_start_sec"], 2.0)
        self.assertNotIn(
            "SOURCE-AUDIO LIP SYNC",
            submitted[0]["prompt"],
        )

    def test_ltx25_music_video_prompts_lock_visible_vocals_to_source_audio(self):
        pid = "pipe-ltx25-lip-sync"
        self._add_pipeline(pid, "running")
        audio_path = self._write_media("song.wav", b"audio")
        for filename in ("shot-1.jpg", "shot-2.jpg"):
            self._write_media(filename, b"image")
        params = {
            "pipeline_type": "music_video",
            "seamless": False,
            "video_model": "ltx2_25",
            "video_params": {"resolution": "1280x704"},
            "audio_path": audio_path,
            "fps": 24,
        }
        plans = [
            {"video_prompt": "The lead singer performs into a microphone."},
            {"video_prompt": "An atmospheric view crosses the empty stage."},
        ]
        planned = [
            {"start": 2, "end": 7, "duration_sec": 5},
            {"start": 7, "end": 12, "duration_sec": 5},
        ]
        submitted: list[dict] = []
        pipeline._wgp = SimpleNamespace(
            save_path=self.temp_dir.name,
            server_config={"services": {}},
            get_model_def=lambda _model: {
                "architecture": "ltx2_25",
                "fps": 24,
            },
            get_model_min_frames_and_step=lambda _model: (17, 8, 8),
        )

        with patch.object(
            pipeline,
            "_submit_and_wait",
            side_effect=lambda gen_params, **_kwargs: (
                submitted.append(gen_params) or ["one.mp4", "two.mp4"]
            ),
        ):
            pipeline._run_video_generation(
                pid,
                params,
                plans,
                planned,
                ["shot-1.jpg", "shot-2.jpg"],
                out_dir=self.temp_dir.name,
            )

        prompts = submitted[0]["prompt"].split(
            "\n---CLIP_BOUNDARY---\n"
        )
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn("SOURCE-AUDIO LIP SYNC", prompt)
            self.assertIn("lip-syncs every vocal syllable", prompt)
        self.assertEqual(submitted[0]["audio_prompt_type"], "A")
        self.assertEqual(submitted[0]["audio_frame_offset"], 48)

    def test_video_phase_rejects_a_recorded_start_image_missing_on_disk(self):
        with open(
            os.path.join(self.temp_dir.name, "present.jpg"), "wb",
        ) as handle:
            handle.write(b"image")
        with self.assertRaisesRegex(
            RuntimeError, "valid recorded files.*shot\(s\) 2",
        ):
            pipeline._require_video_start_images(
                ["present.jpg", "deleted.jpg"],
                2,
                self.temp_dir.name,
            )

    def test_reruns_reject_success_without_a_recorded_output(self):
        pid = "pipe-empty-rerun"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        record["clip_images"] = ["start.jpg"]
        with open(
            os.path.join(self.temp_dir.name, "start.jpg"), "wb",
        ) as handle:
            handle.write(b"image")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        with patch.object(pipeline, "_submit_and_wait", return_value=[]):
            with self.assertRaisesRegex(
                RuntimeError, "without a recorded output",
            ):
                pipeline.rerun_clip_image(self.temp_dir.name, pid, 0)
            with self.assertRaisesRegex(
                RuntimeError, "without a recorded output",
            ):
                pipeline.rerun_clip_video(self.temp_dir.name, pid, 0)

    def test_video_rerun_requires_an_existing_start_image(self):
        pid = "pipe-video-needs-start"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))

        with patch.object(pipeline, "_submit_and_wait") as submit:
            with self.assertRaisesRegex(
                ValueError, "Regenerate its start image",
            ):
                pipeline.rerun_clip_video(self.temp_dir.name, pid, 0)

        submit.assert_not_called()

    def test_prompt_only_video_rerun_does_not_require_a_start_image(self):
        pid = "pipe-video-prompt-only"
        record = self._add_pipeline(pid, "completed")
        record["params"].update({
            "pipeline_type": "short_film_story",
            "_director_shot_image_policy": "prompt_only",
        })
        record["clip_plans"] = [{
            "image_prompt": "unused still",
            "video_prompt": "A named character crosses the room.",
        }]
        record["_planned_clips"] = [{
            "start": 0,
            "end": 5,
            "duration_sec": 5,
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))
        captured = {}

        with patch.object(
            pipeline,
            "_submit_and_wait",
            side_effect=lambda params, **kwargs: captured.update(params) or ["new.mp4"],
        ):
            result = pipeline.rerun_clip_video(self.temp_dir.name, pid, 0)

        self.assertEqual(result["filename"], "new.mp4")
        self.assertNotIn("image_start", captured)
        self.assertEqual(captured["image_prompt_type"], "")

    def test_rejoin_rejects_stale_video_instead_of_omitting_clip(self):
        pid = "pipe-stale-rejoin"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [
            {"image_prompt": "one", "video_prompt": "one"},
            {"image_prompt": "two", "video_prompt": "two"},
        ]
        record["_clip_video_files"] = ["one.mp4", "two.mp4"]
        for filename in record["_clip_video_files"]:
            with open(os.path.join(self.temp_dir.name, filename), "wb") as handle:
                handle.write(b"video")
        self.assertTrue(pipeline._save_pipeline_state(pid))
        pipeline._update_saved_pipeline(
            self.temp_dir.name,
            pid,
            lambda state: state["clips"][0].__setitem__("video_stale", True),
        )
        concatenate = Mock(return_value=True)
        pipeline._wgp.concatenate_multi_clip_videos = concatenate

        with self.assertRaisesRegex(
            ValueError, "stale video clip.*1.*before rejoining",
        ):
            pipeline.rejoin_clips(self.temp_dir.name, pid)

        concatenate.assert_not_called()

    def test_rejoin_rejects_clip_whose_start_image_is_missing(self):
        pid = "pipe-missing-rejoin-start"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [
            {"image_prompt": str(index), "video_prompt": str(index)}
            for index in range(3)
        ]
        record["clip_images"] = [
            "start-one.jpg", "start-two.jpg", "start-three.jpg",
        ]
        record["_clip_video_files"] = [
            "one.mp4", "two.mp4", "three.mp4",
        ]
        for filename in (
            "start-one.jpg", "start-three.jpg",
            "one.mp4", "two.mp4", "three.mp4",
        ):
            with open(os.path.join(self.temp_dir.name, filename), "wb") as handle:
                handle.write(b"media")
        self.assertTrue(pipeline._save_pipeline_state(pid))
        concatenate = Mock(return_value=True)
        pipeline._wgp.concatenate_multi_clip_videos = concatenate

        with self.assertRaisesRegex(
            ValueError, "start image.*clip\(s\) 2.*before rejoining",
        ):
            pipeline.rejoin_clips(self.temp_dir.name, pid)

        concatenate.assert_not_called()

    def test_prompt_only_rejoin_does_not_require_start_images(self):
        pid = "pipe-prompt-only-rejoin"
        record = self._add_pipeline(pid, "completed")
        record["params"]["_director_shot_image_policy"] = "prompt_only"
        record["clip_plans"] = [
            {"image_prompt": "one", "video_prompt": "one"},
            {"image_prompt": "two", "video_prompt": "two"},
        ]
        record["_clip_video_files"] = ["one.mp4", "two.mp4"]
        for filename in record["_clip_video_files"]:
            self._write_media(filename, b"video")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        def concatenate(video_files, output_path, audio_path, **kwargs):
            self.assertEqual(len(video_files), 2)
            with open(output_path, "wb") as handle:
                handle.write(b"joined")
            return True

        pipeline._wgp.concatenate_multi_clip_videos = Mock(
            side_effect=concatenate,
        )
        result = pipeline.rejoin_clips(self.temp_dir.name, pid)
        self.assertTrue(result["filename"].endswith("_rejoin_multiclip.mp4"))

    def test_prompt_only_repair_ignores_intentionally_missing_images(self):
        pid = "pipe-prompt-only-repair"
        record = self._add_pipeline(pid, "completed")
        record["params"]["_director_shot_image_policy"] = "prompt_only"
        record["clip_plans"] = [
            {"image_prompt": "one", "video_prompt": "one"},
            {"image_prompt": "two", "video_prompt": "two"},
        ]
        record["_clip_video_files"] = ["one.mp4", None]
        self._write_media("one.mp4", b"video")
        self.assertTrue(pipeline._save_pipeline_state(pid))
        state = pipeline.load_pipeline_state(self.temp_dir.name, pid)

        plan = pipeline._plan_pipeline_repair(
            self.temp_dir.name,
            pid,
            state,
        )

        self.assertEqual(plan["image_indices"], [])
        self.assertEqual(plan["video_indices"], [1])

    def test_rejoin_rejects_recorded_video_missing_on_disk(self):
        pid = "pipe-missing-rejoin-video"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [
            {"image_prompt": str(index), "video_prompt": str(index)}
            for index in range(3)
        ]
        record["clip_images"] = [
            "start-one.jpg", "start-two.jpg", "start-three.jpg",
        ]
        record["_clip_video_files"] = [
            "one.mp4", "two.mp4", "three.mp4",
        ]
        for filename in (
            "start-one.jpg", "start-two.jpg", "start-three.jpg",
            "one.mp4", "two.mp4",
        ):
            with open(os.path.join(self.temp_dir.name, filename), "wb") as handle:
                handle.write(b"media")
        self.assertTrue(pipeline._save_pipeline_state(pid))
        concatenate = Mock(return_value=True)
        pipeline._wgp.concatenate_multi_clip_videos = concatenate

        with self.assertRaisesRegex(
            ValueError, "video clip\(s\) 3.*before rejoining",
        ):
            pipeline.rejoin_clips(self.temp_dir.name, pid)

        concatenate.assert_not_called()

    def test_rejoin_offsets_source_audio_to_first_planned_time(self):
        pid = "pipe-rejoin-audio-origin"
        record = self._add_pipeline(pid, "completed")
        audio_path = self._write_media("song.wav", b"audio")
        record["params"]["audio_path"] = audio_path
        record["clip_plans"] = [
            {"image_prompt": "one", "video_prompt": "one"},
            {"image_prompt": "two", "video_prompt": "two"},
        ]
        record["_planned_clips"] = [
            {"start": 2, "end": 7, "duration_sec": 5},
            {"start": 7, "end": 12, "duration_sec": 5},
        ]
        record["clip_images"] = ["one.jpg", "two.jpg"]
        record["_clip_video_files"] = ["one.mp4", "two.mp4"]
        for filename in (
            "one.jpg", "two.jpg", "one.mp4", "two.mp4",
        ):
            self._write_media(filename)
        self.assertTrue(pipeline._save_pipeline_state(pid))

        def concatenate(_clips, output_path, _audio, **_kwargs):
            self._write_media(os.path.basename(output_path), b"joined")
            return True

        mock_concat = Mock(side_effect=concatenate)
        pipeline._wgp.concatenate_multi_clip_videos = mock_concat

        pipeline.rejoin_clips(self.temp_dir.name, pid)

        args, kwargs = mock_concat.call_args
        self.assertEqual(args[2], audio_path)
        self.assertEqual(kwargs["audio_start_sec"], 2.0)

    def test_server_repair_skips_good_media_and_persists_joined_completion(self):
        pid = "pipe-server-repair-order"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [
            {"image_prompt": f"image {index}", "video_prompt": f"video {index}"}
            for index in range(3)
        ]
        record["clip_images"] = [None, "start-1.jpg", "start-2.jpg"]
        record["_clip_video_files"] = [
            "old-0.mp4", "old-1.mp4", "good-2.mp4",
        ]
        record["output_files"] = list(record["_clip_video_files"])
        for filename in (
            "start-1.jpg", "start-2.jpg", "old-0.mp4", "old-1.mp4",
            "good-2.mp4",
        ):
            self._write_media(filename)
        self.assertTrue(pipeline._save_pipeline_state(pid))
        pipeline._update_saved_pipeline(
            self.temp_dir.name,
            pid,
            lambda state: state["clips"][1].__setitem__(
                "video_stale", True,
            ),
        )

        order: list[tuple[str, int | None]] = []

        def fake_image(out_dir, actual_pid, clip_index):
            self.assertEqual((out_dir, actual_pid), (self.temp_dir.name, pid))
            order.append(("image", clip_index))
            filename = f"new-start-{clip_index}.jpg"
            self._write_media(filename, b"image")

            def update(state):
                clip = state["clips"][clip_index]
                clip["start_image_filename"] = filename
                clip["video_stale"] = bool(clip.get("video_filename"))
                state["generated_reference_image_filename"] = filename

            pipeline._update_saved_pipeline(out_dir, actual_pid, update)
            return {"filename": filename, "clip_index": clip_index}

        def fake_video(out_dir, actual_pid, clip_index):
            order.append(("video", clip_index))
            before = pipeline.load_pipeline_state(out_dir, actual_pid)
            start_name = before["clips"][clip_index]["start_image_filename"]
            self.assertTrue(os.path.isfile(os.path.join(out_dir, start_name)))
            filename = f"new-video-{clip_index}.mp4"
            self._write_media(filename, b"video")

            def update(state):
                clip = state["clips"][clip_index]
                clip["video_filename"] = filename
                clip["video_stale"] = False
                state.setdefault("output_files", []).append(filename)

            pipeline._update_saved_pipeline(out_dir, actual_pid, update)
            return {"filename": filename, "clip_index": clip_index}

        def fake_rejoin(out_dir, actual_pid):
            order.append(("rejoin", None))
            before = pipeline.load_pipeline_state(out_dir, actual_pid)
            self.assertTrue(all(
                clip.get("start_image_filename")
                and clip.get("video_filename")
                and not clip.get("video_stale")
                for clip in before["clips"]
            ))
            filename = "repaired-join.mp4"
            self._write_media(filename, b"joined")
            pipeline._update_saved_pipeline(
                out_dir,
                actual_pid,
                lambda state: state.setdefault("output_files", []).append(
                    filename,
                ),
            )
            return {"filename": filename}

        with (
            patch.object(
                pipeline, "_rerun_clip_image_impl", side_effect=fake_image,
            ),
            patch.object(
                pipeline, "_rerun_clip_video_impl", side_effect=fake_video,
            ),
            patch.object(
                pipeline, "_rejoin_clips_impl", side_effect=fake_rejoin,
            ),
        ):
            started = pipeline.start_pipeline_repair(self.temp_dir.name, pid)
            terminal = self._wait_for_repair_terminal(pid)

        self.assertEqual(started["repair"]["total"], 4)
        self.assertEqual(order, [
            ("image", 0),
            ("video", 0),
            ("video", 1),
            ("rejoin", None),
        ])
        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(terminal["phase"], "completed")
        self.assertEqual(terminal["current"], 4)
        self.assertEqual(terminal["total"], 4)
        self.assertEqual(terminal["result_filename"], "repaired-join.mp4")

        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(
            [clip["start_image_filename"] for clip in saved["clips"]],
            ["new-start-0.jpg", "start-1.jpg", "start-2.jpg"],
        )
        self.assertEqual(
            [clip["video_filename"] for clip in saved["clips"]],
            ["new-video-0.mp4", "new-video-1.mp4", "good-2.mp4"],
        )
        self.assertIn("repaired-join.mp4", saved["output_files"])

    def test_server_repair_holds_lease_and_duplicate_start_is_idempotent(self):
        pid = "pipe-server-repair-lease"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))

        entered = threading.Event()
        release = threading.Event()
        calls: list[tuple[str, int]] = []

        def blocking_image(out_dir, actual_pid, clip_index):
            calls.append(("image", clip_index))
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("test did not release image repair")
            filename = "lease-start.jpg"
            self._write_media(filename, b"image")
            pipeline._update_saved_pipeline(
                out_dir,
                actual_pid,
                lambda state: state["clips"][clip_index].__setitem__(
                    "start_image_filename", filename,
                ),
            )
            return {"filename": filename, "clip_index": clip_index}

        def fake_video(out_dir, actual_pid, clip_index):
            calls.append(("video", clip_index))
            filename = "lease-video.mp4"
            self._write_media(filename, b"video")

            def update(state):
                state["clips"][clip_index]["video_filename"] = filename
                state["clips"][clip_index]["video_stale"] = False

            pipeline._update_saved_pipeline(out_dir, actual_pid, update)
            return {"filename": filename, "clip_index": clip_index}

        with (
            patch.object(
                pipeline,
                "_rerun_clip_image_impl",
                side_effect=blocking_image,
            ),
            patch.object(
                pipeline, "_rerun_clip_video_impl", side_effect=fake_video,
            ),
        ):
            try:
                before = time.monotonic()
                first = pipeline.start_pipeline_repair(self.temp_dir.name, pid)
                elapsed = time.monotonic() - before
                self.assertLess(elapsed, 1.0)
                self.assertTrue(entered.wait(timeout=1))

                duplicate = pipeline.start_pipeline_repair(
                    self.temp_dir.name, pid,
                )
                self.assertEqual(
                    duplicate["repair"]["operation_id"],
                    first["repair"]["operation_id"],
                )
                self.assertEqual(calls, [("image", 0)])
                with pipeline._pipeline_lock:
                    self.assertIn(pid, pipeline._pipeline_repairs)
                    self.assertIn(pid, pipeline._pipeline_operations)

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
                        self.temp_dir.name, pid, 0, "good",
                    )
                with self.assertRaises(pipeline.PipelineBusyError):
                    pipeline.rerun_clip_video(self.temp_dir.name, pid, 0)
            finally:
                release.set()
            terminal = self._wait_for_repair_terminal(pid)

        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(calls, [("image", 0), ("video", 0)])

    def test_server_repair_failure_releases_lease_and_retry_skips_image(self):
        pid = "pipe-server-repair-retry"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [
            {"image_prompt": "image 0", "video_prompt": "video 0"},
            {"image_prompt": "image 1", "video_prompt": "video 1"},
        ]
        record["clip_images"] = [None, "good-start-1.jpg"]
        record["_clip_video_files"] = [None, "good-video-1.mp4"]
        record["output_files"] = ["good-video-1.mp4"]
        self._write_media("good-start-1.jpg", b"image")
        self._write_media("good-video-1.mp4", b"video")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        order: list[tuple[str, int | None]] = []
        video_attempts = 0

        def fake_image(out_dir, actual_pid, clip_index):
            order.append(("image", clip_index))
            filename = "retry-start-0.jpg"
            self._write_media(filename, b"image")
            pipeline._update_saved_pipeline(
                out_dir,
                actual_pid,
                lambda state: state["clips"][clip_index].__setitem__(
                    "start_image_filename", filename,
                ),
            )
            return {"filename": filename, "clip_index": clip_index}

        def flaky_video(out_dir, actual_pid, clip_index):
            nonlocal video_attempts
            video_attempts += 1
            order.append(("video", clip_index))
            if video_attempts == 1:
                raise RuntimeError("video boom")
            filename = "retry-video-0.mp4"
            self._write_media(filename, b"video")

            def update(state):
                state["clips"][clip_index]["video_filename"] = filename
                state["clips"][clip_index]["video_stale"] = False
                state.setdefault("output_files", []).append(filename)

            pipeline._update_saved_pipeline(out_dir, actual_pid, update)
            return {"filename": filename, "clip_index": clip_index}

        def fake_rejoin(out_dir, actual_pid):
            order.append(("rejoin", None))
            filename = "retry-join.mp4"
            self._write_media(filename, b"joined")
            pipeline._update_saved_pipeline(
                out_dir,
                actual_pid,
                lambda state: state.setdefault("output_files", []).append(
                    filename,
                ),
            )
            return {"filename": filename}

        with (
            patch.object(
                pipeline, "_rerun_clip_image_impl", side_effect=fake_image,
            ),
            patch.object(
                pipeline, "_rerun_clip_video_impl", side_effect=flaky_video,
            ),
            patch.object(
                pipeline, "_rejoin_clips_impl", side_effect=fake_rejoin,
            ),
        ):
            first = pipeline.start_pipeline_repair(self.temp_dir.name, pid)
            failed = self._wait_for_repair_terminal(pid)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["current"], 1)
            self.assertEqual(failed["total"], 3)
            self.assertEqual(failed["error"], "video boom")
            with pipeline._pipeline_lock:
                self.assertNotIn(pid, pipeline._pipeline_repairs)
                self.assertNotIn(pid, pipeline._pipeline_operations)

            second = pipeline.start_pipeline_repair(self.temp_dir.name, pid)
            completed = self._wait_for_repair_terminal(pid)

        self.assertNotEqual(
            first["repair"]["operation_id"],
            second["repair"]["operation_id"],
        )
        self.assertEqual(second["repair"]["total"], 2)
        self.assertEqual(order, [
            ("image", 0),
            ("video", 0),
            ("video", 0),
            ("rejoin", None),
        ])
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["current"], 2)
        self.assertEqual(completed["total"], 2)
        self.assertEqual(completed["result_filename"], "retry-join.mp4")

    def test_server_repair_cancel_is_persisted_and_stops_before_next_unit(self):
        pid = "pipe-server-repair-cancel"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))

        entered = threading.Event()
        release = threading.Event()
        video = Mock()

        def blocking_image(_out_dir, _actual_pid, _clip_index):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("test did not release cancelled repair")
            return {"filename": "ignored.jpg", "clip_index": 0}

        with (
            patch.object(
                pipeline,
                "_rerun_clip_image_impl",
                side_effect=blocking_image,
            ),
            patch.object(pipeline, "_rerun_clip_video_impl", video),
            patch.object(pipeline, "_abort_pipeline_jobs") as abort,
        ):
            try:
                pipeline.start_pipeline_repair(self.temp_dir.name, pid)
                self.assertTrue(entered.wait(timeout=1))
                cancelling = pipeline.cancel_pipeline_repair(
                    self.temp_dir.name, pid,
                )
                self.assertEqual(cancelling["status"], "cancelling")
                self.assertTrue(cancelling["cancel_requested"])
                abort.assert_called_once_with(pid)
                with pipeline._pipeline_lock:
                    self.assertIn(pid, pipeline._pipeline_operations)
            finally:
                release.set()
            terminal = self._wait_for_repair_terminal(pid)

        self.assertEqual(terminal["status"], "cancelled")
        self.assertEqual(terminal["phase"], "cancelled")
        self.assertEqual(terminal["current"], 0)
        self.assertTrue(terminal["cancel_requested"])
        video.assert_not_called()

    def test_orphaned_active_repair_is_normalized_and_listed_interrupted(self):
        pid = "pipe-orphaned-repair"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))
        pipeline._update_saved_pipeline(
            self.temp_dir.name,
            pid,
            lambda state: state.__setitem__("repair", {
                "operation_id": "orphan-operation",
                "status": "running",
                "phase": "images",
                "current": 0,
                "total": 2,
                "clip_index": 0,
                "message": "Generating start image",
                "error": None,
                "started_at": time.time() - 60,
                "updated_at": time.time() - 30,
                "completed_at": None,
                "result_filename": None,
            }),
        )

        loaded = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        repair = loaded["repair"]
        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(repair["status"], "interrupted")
        self.assertEqual(repair["phase"], "interrupted")
        self.assertIsNone(repair["clip_index"])
        self.assertIn("Maestro stopped", repair["error"])
        self.assertIsNotNone(repair["completed_at"])

        summaries = pipeline.list_pipeline_states(self.temp_dir.name)
        summary = next(item for item in summaries if item["id"] == pid)
        self.assertEqual(summary["repair_status"], "interrupted")
        persisted = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(persisted["repair"]["status"], "interrupted")

    def test_list_normalization_cannot_overwrite_newer_repair_progress(self):
        pid = "pipe-list-normalize-race"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))
        pipeline._update_saved_pipeline(
            self.temp_dir.name,
            pid,
            lambda state: state.__setitem__("repair", {
                "operation_id": "orphan-list-operation",
                "status": "running",
                "phase": "images",
                "current": 0,
                "total": 2,
                "clip_index": 0,
                "message": "Old progress",
                "error": None,
            }),
        )

        original_normalize = pipeline._normalize_interrupted_repair
        normalization_read = threading.Barrier(2)
        allow_normalize = threading.Event()
        writer_done = threading.Event()
        failures: list[BaseException] = []

        def paused_normalize(state, actual_pid):
            if actual_pid == pid:
                normalization_read.wait(timeout=2)
                if not allow_normalize.wait(timeout=2):
                    raise RuntimeError("test did not release normalization")
            return original_normalize(state, actual_pid)

        def list_states():
            try:
                pipeline.list_pipeline_states(self.temp_dir.name)
            except BaseException as exc:
                failures.append(exc)

        def publish_newer_progress():
            try:
                def update(state):
                    state["repair"].update({
                        "status": "completed",
                        "phase": "completed",
                        "current": 2,
                        "message": "Newer completed progress",
                    })

                pipeline._update_saved_pipeline(
                    self.temp_dir.name, pid, update,
                )
            except BaseException as exc:
                failures.append(exc)
            finally:
                writer_done.set()

        with patch.object(
            pipeline,
            "_normalize_interrupted_repair",
            side_effect=paused_normalize,
        ):
            reader = threading.Thread(target=list_states)
            writer = threading.Thread(target=publish_newer_progress)
            try:
                reader.start()
                normalization_read.wait(timeout=2)
                writer.start()
                # list_pipeline_states must retain the file lock from its
                # stale read through normalization and replacement.
                self.assertFalse(writer_done.wait(timeout=0.1))
            finally:
                allow_normalize.set()
            reader.join(timeout=2)
            writer.join(timeout=2)

        self.assertFalse(reader.is_alive())
        self.assertFalse(writer.is_alive())
        self.assertEqual(failures, [])
        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(saved["repair"]["status"], "completed")
        self.assertEqual(saved["repair"]["current"], 2)
        self.assertEqual(
            saved["repair"]["message"], "Newer completed progress",
        )

    def test_saved_media_validation_rejects_empty_and_wrong_kind_files(self):
        for filename in ("empty.jpg", "empty.mp4"):
            with open(
                os.path.join(self.temp_dir.name, filename), "wb",
            ):
                pass
        for filename in (
            "image-named-like-video.mp4",
            "video-named-like-image.png",
            "valid-image.webp",
            "valid-video.webm",
        ):
            self._write_media(filename)

        self.assertEqual(
            pipeline._invalid_saved_media_numbers(
                [
                    "empty.jpg",
                    "image-named-like-video.mp4",
                    "valid-image.webp",
                ],
                3,
                self.temp_dir.name,
                "image",
            ),
            [1, 2],
        )
        self.assertEqual(
            pipeline._invalid_saved_media_numbers(
                [
                    "empty.mp4",
                    "video-named-like-image.png",
                    "valid-video.webm",
                ],
                3,
                self.temp_dir.name,
                "video",
            ),
            [1, 2],
        )

    def test_duplicate_start_waits_through_atomic_repair_reservation(self):
        pid = "pipe-repair-start-handshake"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        record["clip_images"] = ["ready.jpg"]
        record["_clip_video_files"] = ["ready.mp4"]
        record["output_files"] = ["ready.mp4"]
        self._write_media("ready.jpg")
        self._write_media("ready.mp4")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        original_plan = pipeline._plan_pipeline_repair
        planner_entered = threading.Event()
        allow_plan = threading.Event()
        results: list[dict] = []
        failures: list[BaseException] = []

        def paused_plan(out_dir, actual_pid, state):
            planner_entered.set()
            if not allow_plan.wait(timeout=2):
                raise RuntimeError("test did not release repair planning")
            return original_plan(out_dir, actual_pid, state)

        def start_repair():
            try:
                results.append(
                    pipeline.start_pipeline_repair(self.temp_dir.name, pid)
                )
            except BaseException as exc:
                failures.append(exc)

        with patch.object(
            pipeline, "_plan_pipeline_repair", side_effect=paused_plan,
        ):
            first = threading.Thread(target=start_repair)
            duplicate = threading.Thread(target=start_repair)
            try:
                first.start()
                self.assertTrue(planner_entered.wait(timeout=1))
                duplicate.start()
                time.sleep(0.1)
                self.assertTrue(duplicate.is_alive())
                self.assertEqual(failures, [])
            finally:
                allow_plan.set()
            first.join(timeout=2)
            duplicate.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(duplicate.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0]["repair"]["operation_id"],
            results[1]["repair"]["operation_id"],
        )
        # This one-clip pipeline is already valid, so preserve the planner's
        # no-op result rather than adding an unnecessary unit or join.
        self.assertEqual(results[0]["repair"]["total"], 0)
        terminal = self._wait_for_repair_terminal(pid)
        self.assertEqual(terminal["status"], "completed")

    def test_fast_worker_cannot_teardown_before_ready_publication(self):
        pid = "pipe-repair-post-start-ready-gap"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        record["clip_images"] = ["ready-gap.jpg"]
        record["_clip_video_files"] = ["ready-gap.mp4"]
        record["output_files"] = ["ready-gap.mp4"]
        self._write_media("ready-gap.jpg")
        self._write_media("ready-gap.mp4")
        self.assertTrue(pipeline._save_pipeline_state(pid))

        original_plan = pipeline._plan_pipeline_repair
        original_run = pipeline._run_pipeline_repair
        original_thread_start = threading.Thread.start
        planner_entered = threading.Event()
        allow_plan = threading.Event()
        worker_waiting = threading.Event()
        post_start_gap = threading.Event()
        release_thread_start = threading.Event()
        duplicate_waiting = threading.Event()
        core_entered = threading.Event()
        results: list[dict] = []
        failures: list[BaseException] = []

        class ObservedReadyEvent:
            def __init__(self):
                self._event = threading.Event()

            def set(self):
                self._event.set()

            def wait(self, timeout=None):
                name = threading.current_thread().name
                if name == f"director-repair-{pid}":
                    worker_waiting.set()
                elif name == "repair-duplicate":
                    duplicate_waiting.set()
                return self._event.wait(timeout)

        def paused_plan(out_dir, actual_pid, state):
            planner_entered.set()
            if not allow_plan.wait(timeout=2):
                raise RuntimeError("test did not release repair planning")
            return original_plan(out_dir, actual_pid, state)

        def observed_run(out_dir, actual_pid, control, plan):
            core_entered.set()
            return original_run(out_dir, actual_pid, control, plan)

        def paused_thread_start(thread):
            original_thread_start(thread)
            if thread.name != f"director-repair-{pid}":
                return
            if not worker_waiting.wait(timeout=1):
                raise RuntimeError("repair worker did not wait for publication")
            post_start_gap.set()
            if not release_thread_start.wait(timeout=2):
                raise RuntimeError("test did not release Thread.start")

        def start_repair():
            try:
                results.append(
                    pipeline.start_pipeline_repair(self.temp_dir.name, pid)
                )
            except BaseException as exc:
                failures.append(exc)

        with (
            patch.object(
                pipeline, "_plan_pipeline_repair", side_effect=paused_plan,
            ),
            patch.object(
                pipeline, "_run_pipeline_repair", side_effect=observed_run,
            ),
            patch.object(
                pipeline.threading.Thread,
                "start",
                new=paused_thread_start,
            ),
        ):
            starter = threading.Thread(
                target=start_repair, name="repair-starter",
            )
            duplicate = threading.Thread(
                target=start_repair, name="repair-duplicate",
            )
            try:
                starter.start()
                self.assertTrue(planner_entered.wait(timeout=1))
                with pipeline._pipeline_lock:
                    control = pipeline._pipeline_repairs[pid]
                    control["ready_event"] = ObservedReadyEvent()
                allow_plan.set()
                self.assertTrue(post_start_gap.wait(timeout=1))

                duplicate.start()
                self.assertTrue(duplicate_waiting.wait(timeout=1))
                self.assertFalse(core_entered.is_set())
                with pipeline._pipeline_lock:
                    self.assertIs(pipeline._pipeline_repairs.get(pid), control)
            finally:
                allow_plan.set()
                release_thread_start.set()
            starter.join(timeout=2)
            duplicate.join(timeout=2)

        self.assertFalse(starter.is_alive())
        self.assertFalse(duplicate.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0]["repair"]["operation_id"],
            results[1]["repair"]["operation_id"],
        )
        terminal = self._wait_for_repair_terminal(pid)
        self.assertEqual(terminal["status"], "completed")

    def test_cancel_waits_for_reserved_repair_publication(self):
        original_plan = pipeline._plan_pipeline_repair

        for prior_repair in (None, {
            "operation_id": "old-operation",
            "status": "completed",
            "phase": "completed",
            "current": 1,
            "total": 1,
            "message": "Old repair complete",
        }):
            label = "none" if prior_repair is None else "old"
            with self.subTest(prior_repair=label):
                pid = f"pipe-repair-cancel-reservation-{label}"
                record = self._add_pipeline(pid, "completed")
                record["clip_plans"] = [{
                    "image_prompt": "portrait", "video_prompt": "motion",
                }]
                image_name = f"reservation-{label}.jpg"
                video_name = f"reservation-{label}.mp4"
                record["clip_images"] = [image_name]
                record["_clip_video_files"] = [video_name]
                record["output_files"] = [video_name]
                self._write_media(image_name)
                self._write_media(video_name)
                self.assertTrue(pipeline._save_pipeline_state(pid))
                if prior_repair is not None:
                    pipeline._update_saved_pipeline(
                        self.temp_dir.name,
                        pid,
                        lambda state: state.__setitem__(
                            "repair", dict(prior_repair),
                        ),
                    )

                planner_entered = threading.Event()
                allow_plan = threading.Event()
                cancel_waiting = threading.Event()
                allow_cancel_return = threading.Event()
                start_results: list[dict] = []
                cancel_results: list[dict | None] = []
                failures: list[BaseException] = []

                class DelayedCancelReadyEvent:
                    def __init__(self):
                        self._event = threading.Event()

                    def set(self):
                        self._event.set()

                    def wait(self, timeout=None):
                        is_canceller = (
                            threading.current_thread().name
                            == "repair-canceller"
                        )
                        if is_canceller:
                            cancel_waiting.set()
                        published = self._event.wait(timeout)
                        if is_canceller:
                            if published and not allow_cancel_return.wait(timeout=2):
                                raise RuntimeError(
                                    "test did not release the waiting cancel"
                                )
                        return published

                def paused_plan(out_dir, actual_pid, state):
                    planner_entered.set()
                    if not allow_plan.wait(timeout=2):
                        raise RuntimeError("test did not release repair planning")
                    return original_plan(out_dir, actual_pid, state)

                def start_repair():
                    try:
                        start_results.append(
                            pipeline.start_pipeline_repair(
                                self.temp_dir.name, pid,
                            )
                        )
                    except BaseException as exc:
                        failures.append(exc)

                def cancel_repair():
                    try:
                        cancel_results.append(
                            pipeline.cancel_pipeline_repair(
                                self.temp_dir.name, pid,
                            )
                        )
                    except BaseException as exc:
                        failures.append(exc)

                with (
                    patch.object(
                        pipeline,
                        "_plan_pipeline_repair",
                        side_effect=paused_plan,
                    ),
                    patch.object(pipeline, "_abort_pipeline_jobs") as abort,
                ):
                    starter = threading.Thread(
                        target=start_repair, name="repair-starter",
                    )
                    canceller = threading.Thread(
                        target=cancel_repair, name="repair-canceller",
                    )
                    try:
                        starter.start()
                        self.assertTrue(planner_entered.wait(timeout=1))
                        with pipeline._pipeline_lock:
                            control = pipeline._pipeline_repairs[pid]
                            control["ready_event"] = DelayedCancelReadyEvent()
                        canceller.start()
                        self.assertTrue(cancel_waiting.wait(timeout=1))
                        self.assertTrue(canceller.is_alive())
                        abort.assert_not_called()

                        before = pipeline.load_pipeline_state(
                            self.temp_dir.name, pid,
                        )
                        if prior_repair is None:
                            self.assertNotIn("repair", before)
                        else:
                            self.assertEqual(
                                before["repair"]["operation_id"],
                                "old-operation",
                            )
                            self.assertEqual(
                                before["repair"]["status"], "completed",
                            )

                        allow_plan.set()
                        terminal = self._wait_for_repair_terminal(pid)
                        self.assertEqual(terminal["status"], "completed")
                        self.assertTrue(canceller.is_alive())
                    finally:
                        allow_plan.set()
                        allow_cancel_return.set()
                    starter.join(timeout=2)
                    canceller.join(timeout=2)

                self.assertFalse(starter.is_alive())
                self.assertFalse(canceller.is_alive())
                self.assertEqual(failures, [])
                self.assertEqual(len(start_results), 1)
                self.assertEqual(len(cancel_results), 1)
                returned = cancel_results[0]
                self.assertIsNotNone(returned)
                self.assertEqual(returned["status"], "completed")
                self.assertEqual(returned["phase"], "completed")
                self.assertEqual(returned["total"], 0)
                self.assertIsNotNone(returned["completed_at"])
                self.assertFalse(returned["cancel_requested"])
                self.assertEqual(
                    returned["operation_id"],
                    start_results[0]["repair"]["operation_id"],
                )

    def test_cancel_before_detached_child_registration_prevents_generation(self):
        pid = "pipe-repair-child-registration-race"
        operation_id = "repair-child-operation"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))
        initial = {
            "operation_id": operation_id,
            "status": "running",
            "phase": "images",
            "current": 0,
            "total": 2,
            "clip_index": 0,
            "message": "Generating start image",
            "error": None,
            "cancel_requested": False,
        }
        ready_event = threading.Event()
        ready_event.set()
        control = {
            "operation_id": operation_id,
            "snapshot": dict(initial),
            "cancel_event": threading.Event(),
            "state_lock": threading.Lock(),
            "finishing": False,
            "thread": None,
            "ready_event": ready_event,
            "start_error": None,
        }
        pipeline._update_saved_pipeline(
            self.temp_dir.name,
            pid,
            lambda state: state.__setitem__("repair", dict(initial)),
        )
        with pipeline._pipeline_lock:
            pipeline._pipeline_repairs[pid] = control
            pipeline._pipeline_operations.add(pid)

        original_abort = pipeline._abort_pipeline_jobs
        abort_scan_complete = threading.Event()
        allow_cancel_release = threading.Event()
        submit_at_registration = threading.Event()
        generation_called = threading.Event()
        cancel_results: list[dict | None] = []
        submit_errors: list[BaseException] = []

        class ObservedParams(dict):
            def get(self, key, default=None):
                value = super().get(key, default)
                if key == "_director_repair_operation_id":
                    submit_at_registration.set()
                return value

        def paused_abort(actual_pid):
            self.assertEqual(actual_pid, pid)
            original_abort(actual_pid)
            abort_scan_complete.set()
            if not allow_cancel_release.wait(timeout=2):
                raise RuntimeError("test did not release cancel scan")

        def fake_generation(job_id):
            generation_called.set()
            job = pipeline._jobs[job_id]
            if try_start(job):
                finish_job(job, "completed", message="Unexpected generation")

        def cancel_repair():
            cancel_results.append(
                pipeline.cancel_pipeline_repair(self.temp_dir.name, pid)
            )

        def submit_child():
            try:
                pipeline._submit_and_wait(
                    ObservedParams({
                        "_director_pipeline_id": pid,
                        "_director_detached_operation": True,
                        "_director_repair_operation_id": operation_id,
                    }),
                    timeout_s=1,
                    out_dir=self.temp_dir.name,
                )
            except BaseException as exc:
                submit_errors.append(exc)

        pipeline._run_generation = fake_generation
        with patch.object(
            pipeline, "_abort_pipeline_jobs", side_effect=paused_abort,
        ):
            canceller = threading.Thread(target=cancel_repair)
            submitter = threading.Thread(target=submit_child)
            try:
                canceller.start()
                self.assertTrue(abort_scan_complete.wait(timeout=1))
                submitter.start()
                self.assertTrue(submit_at_registration.wait(timeout=1))
            finally:
                allow_cancel_release.set()
            canceller.join(timeout=2)
            submitter.join(timeout=2)

        self.assertFalse(canceller.is_alive())
        self.assertFalse(submitter.is_alive())
        self.assertEqual(len(cancel_results), 1)
        self.assertEqual(cancel_results[0]["status"], "cancelling")
        self.assertEqual(len(submit_errors), 1)
        self.assertIsInstance(
            submit_errors[0], pipeline.GenerationCancelledError,
        )
        self.assertFalse(generation_called.is_set())
        self.assertFalse(pipeline._pipeline_child_jobs.get(pid))
        only_job = next(iter(pipeline._jobs.values()))
        self.assertEqual(only_job["status"], "cancelled")

    def test_cancel_holds_registry_lock_while_selecting_jobs_to_abort(self):
        pid = "pipe-repair-cancel-successor-race"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))

        image_entered = threading.Event()
        release_image = threading.Event()
        abort_entered = threading.Event()
        release_abort = threading.Event()
        contender_acquired = threading.Event()
        cancel_result: list[dict | None] = []

        def blocking_image(_out_dir, _actual_pid, _clip_index):
            image_entered.set()
            if not release_image.wait(timeout=3):
                raise RuntimeError("test did not release image repair")
            return {"filename": "ignored.jpg", "clip_index": 0}

        def blocking_abort(actual_pid):
            self.assertEqual(actual_pid, pid)
            abort_entered.set()
            if not release_abort.wait(timeout=3):
                raise RuntimeError("test did not release repair abort")

        def cancel():
            cancel_result.append(
                pipeline.cancel_pipeline_repair(self.temp_dir.name, pid)
            )

        def contend_for_teardown_lock():
            with pipeline._pipeline_lock:
                contender_acquired.set()

        with (
            patch.object(
                pipeline,
                "_rerun_clip_image_impl",
                side_effect=blocking_image,
            ),
            patch.object(
                pipeline, "_abort_pipeline_jobs", side_effect=blocking_abort,
            ),
        ):
            cancel_thread = threading.Thread(target=cancel)
            contender = threading.Thread(target=contend_for_teardown_lock)
            try:
                pipeline.start_pipeline_repair(self.temp_dir.name, pid)
                self.assertTrue(image_entered.wait(timeout=1))
                cancel_thread.start()
                self.assertTrue(abort_entered.wait(timeout=1))
                contender.start()
                # A successor/teardown path uses this same lock. It cannot
                # pass the old repair while that repair selects jobs to abort.
                self.assertFalse(contender_acquired.wait(timeout=0.1))
            finally:
                release_abort.set()
                release_image.set()
            cancel_thread.join(timeout=2)
            contender.join(timeout=2)
            terminal = self._wait_for_repair_terminal(pid)

        self.assertFalse(cancel_thread.is_alive())
        self.assertFalse(contender.is_alive())
        self.assertTrue(contender_acquired.is_set())
        self.assertEqual(cancel_result[0]["status"], "cancelling")
        self.assertEqual(terminal["status"], "cancelled")

    def test_completion_decision_excludes_a_late_cancel_atomically(self):
        pid = "pipe-repair-completion-cancel-race"
        record = self._add_pipeline(pid, "completed")
        record["clip_plans"] = [{
            "image_prompt": "portrait", "video_prompt": "motion",
        }]
        self.assertTrue(pipeline._save_pipeline_state(pid))

        class PausingEvent:
            def __init__(self):
                self._event = threading.Event()
                self.check_entered = threading.Event()
                self.release_check = threading.Event()
                self._paused = False
                self._guard = threading.Lock()

            def set(self):
                self._event.set()

            def is_set(self):
                if threading.current_thread().name == "repair-finisher":
                    with self._guard:
                        should_pause = not self._paused
                        self._paused = True
                    if should_pause:
                        value = self._event.is_set()
                        self.check_entered.set()
                        if not self.release_check.wait(timeout=3):
                            raise RuntimeError(
                                "test did not release terminal decision"
                            )
                        return value
                return self._event.is_set()

        operation_id = "completion-wins-operation"
        initial = {
            "operation_id": operation_id,
            "status": "running",
            "phase": "rejoin",
            "current": 1,
            "total": 2,
            "clip_index": None,
            "message": "Finishing",
            "error": None,
        }
        cancel_event = PausingEvent()
        control = {
            "operation_id": operation_id,
            "snapshot": dict(initial),
            "cancel_event": cancel_event,
            "state_lock": threading.Lock(),
            "finishing": False,
            "thread": None,
        }
        pipeline._update_saved_pipeline(
            self.temp_dir.name,
            pid,
            lambda state: state.__setitem__("repair", dict(initial)),
        )
        with pipeline._pipeline_lock:
            pipeline._pipeline_repairs[pid] = control

        finish_result: list[dict | None] = []
        cancel_result: list[dict | None] = []

        def finish():
            finish_result.append(pipeline._finish_pipeline_repair(
                self.temp_dir.name,
                pid,
                control,
                status="completed",
                phase="completed",
                current=2,
                total=2,
                message="Repair complete",
            ))

        def cancel():
            cancel_result.append(
                pipeline.cancel_pipeline_repair(self.temp_dir.name, pid)
            )

        with patch.object(pipeline, "_abort_pipeline_jobs") as abort:
            finisher = threading.Thread(
                target=finish, name="repair-finisher",
            )
            canceller = threading.Thread(target=cancel)
            try:
                finisher.start()
                self.assertTrue(cancel_event.check_entered.wait(timeout=1))
                canceller.start()
                # The completion decision owns both state and registry locks;
                # a late cancel cannot publish "cancelling" in the gap.
                time.sleep(0.1)
                self.assertTrue(canceller.is_alive())
            finally:
                cancel_event.release_check.set()
            finisher.join(timeout=2)
            canceller.join(timeout=2)

        self.assertFalse(finisher.is_alive())
        self.assertFalse(canceller.is_alive())
        abort.assert_not_called()
        self.assertEqual(finish_result[0]["status"], "completed")
        self.assertEqual(cancel_result[0]["status"], "completed")
        saved = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        self.assertEqual(saved["repair"]["status"], "completed")
        self.assertFalse(saved["repair"]["cancel_requested"])

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

    def test_delete_does_not_trust_generated_anchor_name_without_ownership(self):
        pid = "pipe-unowned-anchor"
        record = self._add_pipeline(pid, "completed")
        self.assertTrue(pipeline._save_pipeline_state(pid))
        unrelated_path = os.path.join(self.temp_dir.name, "unrelated.jpg")
        with open(unrelated_path, "wb") as handle:
            handle.write(b"unrelated")
        pipeline._update_saved_pipeline(
            self.temp_dir.name,
            pid,
            lambda state: state.__setitem__(
                "generated_reference_image_filename", "unrelated.jpg",
            ),
        )

        result = pipeline.delete_pipeline(self.temp_dir.name, pid)

        self.assertTrue(result["ok"])
        self.assertTrue(os.path.isfile(unrelated_path))

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

    def test_fresh_start_strips_internal_generated_anchor_metadata(self):
        params = {
            "pipeline_type": "music_video",
            "generated_reference_image_filename": "unrelated.jpg",
        }

        with patch.object(pipeline, "_start_pipeline_worker") as start_worker:
            pid = pipeline.start_pipeline(params)

        start_worker.assert_called_once_with(pid)
        self.assertNotIn("generated_reference_image_filename", params)
        self.assertNotIn(
            "generated_reference_image_filename",
            pipeline._pipelines[pid]["params"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
