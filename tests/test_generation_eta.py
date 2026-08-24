import os
import sys
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.generation_eta import AdaptiveGenerationEta, task_workload


def _task(
    frames=100,
    steps=10,
    *,
    cache=False,
    cache_start=30,
    resolution="832x480",
):
    return {
        "params": {
            "video_length": frames,
            "num_inference_steps": steps,
            "resolution": resolution,
            "skip_steps_cache_type": "first_block" if cache else "",
            "skip_steps_start_step_perc": cache_start,
        }
    }


class AdaptiveGenerationEtaTests(unittest.TestCase):
    def test_workload_scales_with_frames_steps_and_resolution(self):
        base = task_workload(_task())
        self.assertGreater(task_workload(_task(frames=200)), base * 2)
        self.assertGreater(task_workload(_task(steps=20)), base * 1.9)
        self.assertGreater(
            task_workload(_task(resolution="1280x720")),
            base,
        )

    def test_first_block_uses_observed_post_warmup_rate(self):
        eta = AdaptiveGenerationEta([_task(cache=True)], now_fn=lambda: 0.0)
        eta.start_task(0, now=0)
        eta.observe_progress(0, 10, "Denoising", now=0)
        eta.observe_progress(1, 10, "Denoising", now=10)
        eta.observe_progress(2, 10, "Denoising", now=20)
        warmup = eta.observe_progress(3, 10, "Denoising", now=30)
        eta.observe_progress(4, 10, "Denoising", now=34)
        cached = eta.observe_progress(5, 10, "Denoising", now=38)

        self.assertEqual("live-cache-aware", cached["eta_basis"])
        self.assertLess(
            cached["clip_eta_seconds"],
            warmup["clip_eta_seconds"],
        )
        # A flat 10 seconds/step estimate would still claim ~50 seconds of
        # denoising plus finishing time. The cache-aware estimate uses the
        # measured four-second post-warm-up steps instead.
        self.assertLess(cached["clip_eta_seconds"], 50)

    def test_project_eta_scales_remaining_clip_workload(self):
        eta = AdaptiveGenerationEta(
            [_task(frames=100), _task(frames=200)],
            now_fn=lambda: 0.0,
        )
        eta.start_task(0, now=0)
        eta.observe_progress(0, 10, "Denoising", now=0)
        for step in range(1, 6):
            snapshot = eta.observe_progress(
                step,
                10,
                "Denoising",
                now=step * 5,
            )

        self.assertIsNotNone(snapshot["clip_eta_seconds"])
        self.assertIsNotNone(snapshot["project_eta_seconds"])
        self.assertGreater(
            snapshot["project_eta_seconds"],
            snapshot["clip_eta_seconds"] * 2,
        )
        self.assertGreater(
            snapshot["clip_estimates"][1]["seconds"],
            snapshot["clip_estimates"][0]["seconds"] * 2,
        )

    def test_completed_clip_calibrates_next_clip_before_first_step(self):
        eta = AdaptiveGenerationEta(
            [_task(frames=100), _task(frames=200)],
            now_fn=lambda: 0.0,
        )
        eta.start_task(0, now=0)
        eta.observe_progress(0, 10, "Denoising", now=0)
        eta.observe_progress(10, 10, "Denoising", now=50)
        eta.complete_task(60, now=60)
        eta.start_task(1, now=65)
        snapshot = eta.snapshot(now=65)

        self.assertIsNotNone(snapshot["clip_eta_seconds"])
        self.assertGreater(snapshot["clip_eta_seconds"], 120)
        self.assertEqual("completed", snapshot["clip_estimates"][0]["status"])
        self.assertEqual("current", snapshot["clip_estimates"][1]["status"])

    def test_sliding_window_eta_includes_unstarted_windows(self):
        eta = AdaptiveGenerationEta([_task()], now_fn=lambda: 0.0)
        eta.start_task(0, now=0)
        eta.observe_progress(
            0, 10, "Sliding Window 1/3 - Denoising", now=0,
        )
        eta.observe_progress(
            5, 10, "Sliding Window 1/3 - Denoising", now=25,
        )
        snapshot = eta.snapshot(now=25)

        # Five current steps are ~25s. Two unstarted windows must make the
        # whole clip estimate materially larger than the current half-window.
        self.assertGreater(snapshot["clip_eta_seconds"], 70)

    def test_sliding_window_exposes_window_and_full_generation_eta(self):
        eta = AdaptiveGenerationEta([_task()], now_fn=lambda: 0.0)
        eta.start_task(0, now=0)
        eta.observe_status(
            "Sliding Window 1/3 - Encoding Prompt",
            now=0,
        )
        eta.observe_progress(
            0, 10, "Sliding Window 1/3 - Denoising", now=2,
        )
        snapshot = eta.observe_progress(
            5, 10, "Sliding Window 1/3 - Denoising", now=27,
        )

        self.assertEqual(1, snapshot["current_window"])
        self.assertEqual(3, snapshot["total_windows"])
        self.assertIsNotNone(snapshot["window_eta_seconds"])
        self.assertIsNotNone(snapshot["generation_eta_seconds"])
        self.assertGreater(
            snapshot["generation_eta_seconds"],
            snapshot["window_eta_seconds"] * 2,
        )

    def test_window_boundary_learns_complete_window_wall_time(self):
        eta = AdaptiveGenerationEta([_task()], now_fn=lambda: 0.0)
        eta.start_task(0, now=0)
        eta.observe_status(
            "Sliding Window 1/3 - Encoding Prompt",
            now=0,
        )
        eta.observe_progress(
            0, 10, "Sliding Window 1/3 - Denoising", now=5,
        )
        eta.observe_progress(
            10, 10, "Sliding Window 1/3 - VAE Decoding", now=45,
        )
        boundary = eta.observe_status(
            "Sliding Window 2/3 - Encoding Prompt",
            now=60,
        )
        eta.observe_progress(
            0, 10, "Sliding Window 2/3 - Denoising", now=65,
        )
        snapshot = eta.observe_progress(
            5, 10, "Sliding Window 2/3 - Denoising", now=85,
        )

        self.assertEqual(2, boundary["current_window"])
        self.assertEqual(2, snapshot["current_window"])
        self.assertGreater(snapshot["generation_eta_seconds"], 50)
        self.assertGreater(
            snapshot["generation_eta_seconds"],
            snapshot["window_eta_seconds"],
        )


if __name__ == "__main__":
    unittest.main()
