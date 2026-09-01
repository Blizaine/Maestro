import os
import json
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.generation_eta import (
    AdaptiveGenerationEta,
    GenerationEtaHistory,
    task_workload,
)


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

    def test_persistent_history_seeds_matching_multi_window_render(self):
        task = _task(frames=400, steps=8, resolution="1280x704")
        task["params"].update({
            "model_type": "minimax_h3",
            "sliding_window_size": 100,
            "sliding_window_overlap": 0,
            "override_attention": "sol",
        })
        hardware = {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_vram_gb": 24,
            "gpu_capability": "sm89",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "eta.sqlite3")
            store = GenerationEtaHistory(path, hardware=hardware)
            self.assertTrue(store.record(
                task,
                400,
                wall_seconds=430,
                non_step_seconds=40,
                window_seconds=[100, 100, 100, 100],
                source_key="first-run",
            ))

            # A fresh store/estimator simulates restarting Maestro.
            fresh_store = GenerationEtaHistory(path, hardware=hardware)
            eta = AdaptiveGenerationEta(
                [task],
                history_store=fresh_store,
                now_fn=lambda: 0.0,
            )
            eta.start_task(0, now=0)
            snapshot = eta.snapshot(now=0)

            self.assertEqual(1, snapshot["eta_history_samples"])
            self.assertEqual("exact", snapshot["eta_history_match"])
            self.assertEqual("historical", snapshot["eta_basis"])
            self.assertEqual(4, snapshot["total_windows"])
            self.assertAlmostEqual(430, snapshot["clip_eta_seconds"], delta=2)

    def test_history_reuses_window_speed_for_a_different_total_duration(self):
        short = _task(frames=400, steps=8, resolution="1280x704")
        short["params"].update({
            "model_type": "minimax_h3",
            "sliding_window_size": 100,
            "sliding_window_overlap": 0,
        })
        longer = _task(frames=600, steps=8, resolution="1280x704")
        longer["params"].update({
            "model_type": "minimax_h3",
            "sliding_window_size": 100,
            "sliding_window_overlap": 0,
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            store = GenerationEtaHistory(
                os.path.join(temp_dir, "eta.sqlite3"),
                hardware={"gpu_name": "test-gpu", "gpu_vram_gb": 24},
            )
            store.record(
                short,
                400,
                wall_seconds=420,
                window_seconds=[100, 100, 100, 100],
                source_key="four-windows",
            )
            prediction = store.estimate(longer)

            self.assertIsNotNone(prediction)
            self.assertEqual("exact", prediction.match)
            self.assertAlmostEqual(600, prediction.active_seconds, delta=1)
            self.assertAlmostEqual(620, prediction.wall_seconds, delta=1)

    def test_history_keeps_different_lora_configs_separate(self):
        base = _task(frames=100, steps=8)
        base["params"]["model_type"] = "minimax_h3"
        lora = _task(frames=100, steps=8)
        lora["params"].update({
            "model_type": "minimax_h3",
            "activated_loras": ["MiniMax-H3-FL2VA-Acc-8Step.safetensors"],
            "loras_multipliers": "1.0",
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            store = GenerationEtaHistory(
                os.path.join(temp_dir, "eta.sqlite3"),
                hardware={"gpu_name": "test-gpu", "gpu_vram_gb": 24},
            )
            store.record(base, 60, source_key="without-lora")
            self.assertIsNone(store.estimate(lora))

    def test_sidecar_bootstrap_ignores_partial_multi_window_outputs(self):
        params = {
            "model_type": "minimax_h3",
            "resolution": "1280x704",
            "video_length": 400,
            "sliding_window_size": 100,
            "sliding_window_overlap": 0,
            "num_inference_steps": 8,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            complete_path = os.path.join(temp_dir, "complete.meta.json")
            partial_path = os.path.join(temp_dir, "partial.meta.json")
            with open(complete_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "generation_mode": "video",
                    "params": params,
                    "generation_time": 400,
                    "generation_time_basis": "active",
                    "job_elapsed_time": 430,
                    "multi_window_timing": {
                        "window_count": 4,
                        "completed_windows": 4,
                        "window_generation_seconds": [100, 100, 100, 100],
                        "total_generation_seconds": 400,
                    },
                }, handle)
            with open(partial_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "generation_mode": "video",
                    "params": params,
                    "generation_time": 100,
                    "multi_window_timing": {
                        "window_count": 4,
                        "completed_windows": 1,
                        "window_generation_seconds": [100],
                        "total_generation_seconds": 100,
                    },
                }, handle)

            store = GenerationEtaHistory(
                os.path.join(temp_dir, "eta.sqlite3"),
                hardware={"gpu_name": "test-gpu", "gpu_vram_gb": 24},
            )
            imported = store.bootstrap_from_sidecars(temp_dir)
            estimate = store.estimate({"params": params})

            self.assertEqual(1, imported)
            self.assertIsNotNone(estimate)
            self.assertEqual(1, estimate.sample_count)
            self.assertAlmostEqual(400, estimate.active_seconds, delta=1)


if __name__ == "__main__":
    unittest.main()
