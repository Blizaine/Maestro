"""Regressions for reconnecting Director status polling after restarts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402


class TestDirectorPipelineStatusReconnect(unittest.TestCase):
    def tearDown(self):
        pipeline._pipelines.pop("deadbeef", None)

    def test_saved_running_state_is_reported_as_failed_instead_of_404(self):
        with tempfile.TemporaryDirectory() as output_dir:
            state_path = os.path.join(
                output_dir,
                "_director_pipeline_deadbeef.json",
            )
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "pipeline_id": "deadbeef",
                    "status": "running",
                    "clips": [{
                        "video_prompt": "A saved shot.",
                        "start_image_filename": "",
                        "video_filename": "clip.mp4",
                    }],
                    "output_files": ["clip.mp4"],
                }, handle)

            status = pipeline.get_pipeline_status("deadbeef", output_dir)

        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["phase"], "failed")
        self.assertTrue(status["recovered_from_disk"])
        self.assertEqual(status["progress"]["current"], 1)
        self.assertEqual(status["progress"]["total"], 1)

    def test_live_pipeline_wins_over_saved_fallback(self):
        live = {
            "id": "deadbeef",
            "status": "running",
            "phase": "generating_video",
        }
        pipeline._pipelines["deadbeef"] = live

        with tempfile.TemporaryDirectory() as output_dir:
            status = pipeline.get_pipeline_status("deadbeef", output_dir)

        self.assertEqual(status, live)
        self.assertIsNot(status, live)


if __name__ == "__main__":
    unittest.main()
