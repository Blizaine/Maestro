"""Per-child Director engine selection tests without model runtime dependencies."""
from __future__ import annotations

import os
import sys
import threading
import unittest
from types import SimpleNamespace


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402


class TestDirectorComfyUIEngineSnapshot(unittest.TestCase):
    def setUp(self):
        self.originals = {
            "jobs": pipeline._jobs,
            "run_generation": pipeline._run_generation,
            "wgp": pipeline._wgp,
            "gen_lock": pipeline._gen_lock,
            "active_gen_states": pipeline._active_gen_states,
            "prepare_generation_engine": pipeline._prepare_generation_engine,
        }
        self.jobs = {}
        self.started = threading.Event()

    def tearDown(self):
        pipeline._jobs = self.originals["jobs"]
        pipeline._run_generation = self.originals["run_generation"]
        pipeline._wgp = self.originals["wgp"]
        pipeline._gen_lock = self.originals["gen_lock"]
        pipeline._active_gen_states = self.originals["active_gen_states"]
        pipeline._prepare_generation_engine = self.originals["prepare_generation_engine"]

    def _run_child(self, job_id):
        job = self.jobs[job_id]
        self.started.set()
        job.update(status="completed", output_files=[])

    def test_engine_snapshot_is_captured_before_job_registration_and_worker_start(self):
        callback_calls = []
        route = {"workflow_id": "verified-video", "enabled": True}
        snapshot = {"engine": "comfyui", "route": route}

        def capture(params):
            callback_calls.append(dict(params))
            self.assertEqual(self.jobs, {})
            return snapshot

        pipeline.init(
            self.jobs,
            self._run_child,
            SimpleNamespace(),
            prepare_generation_engine=capture,
        )
        result = pipeline._submit_and_wait(
            {"model_type": "video-model", "generation_mode": "video"},
            timeout_s=2,
        )

        self.assertTrue(self.started.wait(1))
        self.assertEqual(result, [])
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0]["model_type"], "video-model")
        job = next(iter(self.jobs.values()))
        self.assertEqual(job["generation_engine_snapshot"], snapshot)

    def test_missing_callback_preserves_native_default(self):
        pipeline.init(self.jobs, self._run_child, SimpleNamespace())

        result = pipeline._submit_and_wait(
            {"model_type": "video-model", "generation_mode": "video"},
            timeout_s=2,
        )

        self.assertTrue(self.started.wait(1))
        self.assertEqual(result, [])
        self.assertNotIn("generation_engine_snapshot", next(iter(self.jobs.values())))

    def test_snapshot_failure_does_not_publish_or_start_child_job(self):
        def fail(_params):
            raise ValueError("ComfyUI route is unavailable")

        pipeline.init(
            self.jobs,
            self._run_child,
            SimpleNamespace(),
            prepare_generation_engine=fail,
        )

        with self.assertRaisesRegex(ValueError, "route is unavailable"):
            pipeline._submit_and_wait(
                {"model_type": "video-model", "generation_mode": "video"},
                timeout_s=2,
            )

        self.assertEqual(self.jobs, {})
        self.assertFalse(self.started.is_set())


if __name__ == "__main__":
    unittest.main()
