from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from app.scripts.benchmark_memory import (
    BenchmarkError,
    BenchmarkRunner,
    LogCursor,
    _peak_metric,
    aggregate_timings,
    load_matrix,
    parse_log_evidence,
    restore_only,
    validate_base_url,
)


class FakeApi:
    def __init__(
        self,
        *,
        initially_busy=False,
        foreign_after_submission=False,
        never_finish=False,
        fail_submission=False,
        output_metadata=None,
        output_metadata_by_filename=None,
        output_files=None,
        metadata_error=None,
    ):
        self.base_url = "http://127.0.0.1:42012"
        self.system = {"video_profile": 2, "vram_allocator": "vmm"}
        self.services = {"auto_performance": True}
        self.initially_busy = initially_busy
        self.foreign_during_run = False
        self.foreign_after_submission = foreign_after_submission
        self.never_finish = never_finish
        self.fail_submission = fail_submission
        self.output_metadata = output_metadata
        self.output_metadata_by_filename = output_metadata_by_filename
        self.output_files = output_files
        self.metadata_error = metadata_error
        self.generation_request = {}
        self.metadata_paths = []
        self.job_id = None
        self.job_status = "queued"
        self.cancelled = []
        self.put_calls = []
        self.poll_index = 0

    def get(self, path):
        if path.startswith("/api/v1/outputs/") and "/metadata?" in path:
            self.metadata_paths.append(path)
            if self.metadata_error is not None:
                raise self.metadata_error
            if self.output_metadata_by_filename is not None:
                filename = path.split("/api/v1/outputs/", 1)[1].split("/metadata?", 1)[0]
                return self.output_metadata_by_filename.get(filename)
            if self.output_metadata is not None:
                return self.output_metadata
            resolution = self.generation_request.get("resolution", "864x480")
            try:
                width, height = (int(part) for part in resolution.lower().split("x", 1))
            except (AttributeError, TypeError, ValueError):
                width, height = 864, 480
            frames = self.generation_request.get("video_length", 81)
            return {
                "params": dict(self.generation_request),
                "media_info": {
                    "frames": frames,
                    "width": width,
                    "height": height,
                    "fps": 24,
                    "duration_seconds": round(frames / 24, 3),
                },
            }
        if path == "/api/v1/jobs":
            jobs = []
            if self.initially_busy:
                jobs.append({"job_id": "pre-existing", "status": "running"})
            if self.foreign_during_run:
                jobs.append({"job_id": "foreign-job", "status": "queued"})
            if self.job_id and self.job_status in {"queued", "running"}:
                jobs.append({"job_id": self.job_id, "status": self.job_status})
            return {"jobs": jobs}
        if path == "/api/v1/system-config":
            return dict(self.system)
        if path == "/api/v1/services-config":
            return dict(self.services)
        if path == "/api/v1/system-stats":
            self.poll_index += 1
            return {
                "cpu": {"percent": 31.0},
                "ram": {"percent": 40.0, "used_gb": 8.0, "total_gb": 32.0},
                "gpu": {
                    "available": True,
                    "percent": 72.0,
                    "vram_used_gb": float(self.poll_index),
                    "vram_total_gb": 24.0,
                    "vram_percent": float(self.poll_index) / 24 * 100,
                },
            }
        if path == f"/api/v1/status/{self.job_id}":
            if self.never_finish and self.job_status != "cancelled":
                self.job_status = "running"
                return {
                    "job_id": self.job_id, "status": "running", "phase": "Denoising",
                    "message": "Denoising step 1/8", "step": 1, "total_steps": 8,
                    "progress": 10, "output_files": [], "error": None,
                }
            if self.job_status == "queued":
                self.job_status = "completed"
            return {
                "job_id": self.job_id, "status": self.job_status,
                "phase": "", "message": self.job_status, "step": 0,
                "total_steps": 0, "progress": 0,
                "output_files": (
                    self.output_files
                    if self.output_files is not None and self.job_status == "completed"
                    else ["benchmark-output.mp4"] if self.job_status == "completed" else []
                ),
                "error": None,
            }
        raise AssertionError(f"Unexpected GET path: {path}")

    def post(self, path, body=None):
        if path == "/api/v1/system/release-model":
            return {"status": "ok"}
        if path == "/api/v1/generate":
            if self.fail_submission:
                raise RuntimeError("simulated submission error")
            self.generation_request = dict(body or {})
            self.job_id = "benchmark-job"
            self.job_status = "queued"
            self.foreign_during_run = self.foreign_after_submission
            return {"job_id": self.job_id, "status": "queued"}
        if path == f"/api/v1/cancel/{self.job_id}":
            self.cancelled.append(self.job_id)
            self.job_status = "cancelled"
            return {"status": "cancelled"}
        if path.startswith("/api/v1/cancel/"):
            self.cancelled.append(path.rsplit("/", 1)[-1])
            return {"status": "cancelled"}
        raise AssertionError(f"Unexpected POST path: {path}")

    def put(self, path, body):
        self.put_calls.append((path, dict(body)))
        target = self.system if path == "/api/v1/system-config" else self.services
        target.update(body)
        return {"status": "ok"}


def one_case_matrix(settings=None, *, repeats=1, reload=False):
    return {
        "request": {"generation_mode": "video", "model_type": "wan-test"},
        "cases": [{
            "name": "baseline", "settings": settings or {},
            "request": {}, "repeats": repeats, "reload": reload,
        }],
    }


class MemoryBenchmarkTests(unittest.TestCase):
    def test_loopback_validation_and_matrix_bounds(self):
        self.assertEqual(validate_base_url("http://localhost:42012/"), "http://localhost:42012")
        self.assertEqual(validate_base_url("http://[::1]:42012"), "http://[::1]:42012")
        with self.assertRaises(BenchmarkError):
            validate_base_url("https://example.com")
        with self.assertRaises(BenchmarkError):
            load_matrix(json.dumps(one_case_matrix(repeats=3)))
        duplicate_names = one_case_matrix()
        duplicate_names["cases"].append({"name": " baseline ", "settings": {}, "request": {}, "repeats": 1})
        with self.assertRaisesRegex(BenchmarkError, "unique"):
            load_matrix(json.dumps(duplicate_names))
        with self.assertRaisesRegex(BenchmarkError, "unsupported system setting"):
            load_matrix(json.dumps(one_case_matrix({"password": "should-not-be-written"})))

    def test_ram_allocator_change_is_refused_before_mutating_live_process(self):
        api = FakeApi()
        api.system["ram_allocator"] = "default"
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bench"
            runner = BenchmarkRunner(api, one_case_matrix({"ram_allocator": "mmgp"}), out, 10, .2, None, progress=False)
            with self.assertRaisesRegex(BenchmarkError, "cannot switch ram_allocator"):
                runner.execute()
            self.assertEqual(api.put_calls, [])
            self.assertEqual(api.cancelled, [])

    def test_refuses_busy_server_before_output_or_settings_mutation(self):
        api = FakeApi(initially_busy=True)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bench"
            runner = BenchmarkRunner(api, one_case_matrix({"video_profile": 3}), out, 10, 0.2, None, progress=False)
            with self.assertRaisesRegex(BenchmarkError, "queued or running"):
                runner.execute()
            self.assertFalse(out.exists())
            self.assertEqual(api.put_calls, [])
            self.assertEqual(api.cancelled, [])

    def test_successful_output_is_verified_and_effective_metadata_is_recorded(self):
        request = {
            "generation_mode": "video",
            "model_type": "minimax_h3_fused_turbo",
            "workspace": "Memory Benchmark/one",
            "resolution": "864x480",
            "video_length": 124,
            "sliding_window_size": 124,
            "sliding_window_overlap": 0,
            "sliding_window_discard_last_frames": 0,
            "minimax_h3_multi_window": False,
            "minimax_h3_reference_sequence": "auto",
            "minimax_h3_text_encoder": "q2_k",
            "settings_version": 2.58,
            "num_inference_steps": 8,
        }
        matrix = one_case_matrix()
        matrix["request"] = request
        api = FakeApi()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bench"
            result = BenchmarkRunner(
                api, matrix, out, 10, 0.2, None, progress=False
            ).execute()

        run = result["runs"][0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["benchmark_validation"]["status"], "verified")
        self.assertEqual(result["benchmark_validation"]["eligible_runs"], 1)
        self.assertEqual(run["job_terminal_status"], "completed")
        self.assertEqual(run["status"], "completed")
        self.assertTrue(run["benchmark_eligible"])
        self.assertEqual(run["output_validation"]["status"], "verified")
        output = run["output_validation"]["outputs"][0]
        self.assertEqual(output["params"]["sliding_window_size"], 124)
        self.assertEqual(output["params"]["settings_version"], 2.58)
        self.assertEqual(output["media_info"]["frames"], 124)
        self.assertEqual(run["request_parameters"]["workspace"], "Memory Benchmark/one")
        self.assertEqual(
            api.metadata_paths,
            ["/api/v1/outputs/benchmark-output.mp4/metadata?workspace=Memory%20Benchmark%2Fone"],
        )

    def test_shortened_completed_video_is_ineligible_and_settings_still_restore(self):
        request = {
            "generation_mode": "video",
            "model_type": "minimax_h3_fused_turbo",
            "workspace": "Memory-Benchmark",
            "resolution": "1280x720",
            "video_length": 243,
            "sliding_window_size": 243,
            "sliding_window_memory_override": True,
            "num_inference_steps": 5,
            "settings_version": 2.58,
        }
        shortened_metadata = {
            "params": {
                "video_length": 124,
                "sliding_window_size": 124,
                "sliding_window_memory_override": True,
                "num_inference_steps": 5,
                "resolution": "1280x720",
                "settings_version": 2.58,
            },
            "media_info": {
                "frames": 124,
                "width": 1280,
                "height": 704,
                "fps": 24,
                "duration_seconds": 5.167,
            },
        }
        matrix = one_case_matrix({"video_profile": 3})
        matrix["request"] = request
        api = FakeApi(output_metadata=shortened_metadata)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bench"
            result = BenchmarkRunner(
                api, matrix, out, 10, 0.2, None, progress=False
            ).execute()
            saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
            with (out / "results.csv").open(encoding="utf-8", newline="") as handle:
                csv_run = next(csv.DictReader(handle))

        run = result["runs"][0]
        self.assertEqual(run["job_terminal_status"], "completed")
        self.assertEqual(run["status"], "workload_mismatch")
        self.assertFalse(run["benchmark_eligible"])
        self.assertEqual(run["output_validation"]["status"], "workload_mismatch")
        mismatch_fields = {item["field"] for item in run["output_validation"]["mismatches"]}
        self.assertIn("video_length", mismatch_fields)
        self.assertIn("sliding_window_size", mismatch_fields)
        self.assertEqual(run["output_validation"]["outputs"][0]["media_info"]["frames"], 124)
        self.assertEqual(saved["runs"][0]["status"], "workload_mismatch")
        self.assertFalse(saved["runs"][0]["benchmark_eligible"])
        self.assertEqual(saved["benchmark_validation"]["status"], "workload_mismatch")
        self.assertEqual(saved["benchmark_validation"]["eligible_runs"], 0)
        self.assertEqual(saved["benchmark_validation"]["workload_mismatch_runs"], 1)
        self.assertEqual(csv_run["status"], "workload_mismatch")
        self.assertEqual(csv_run["benchmark_eligible"], "False")
        self.assertEqual(
            json.loads(csv_run["output_validation"])["status"], "workload_mismatch"
        )
        self.assertTrue(run["request_parameters"]["sliding_window_memory_override"])
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual(result["restoration"]["status"], "restored")
        self.assertEqual(api.system["video_profile"], 2)
        self.assertTrue(api.services["auto_performance"])

    def test_multi_window_intermediate_and_final_outputs_verify_as_sequence(self):
        request = {
            "generation_mode": "video",
            "model_type": "minimax_h3_fused_turbo",
            "workspace": "Memory-Benchmark",
            "resolution": "auto_720p",
            "video_length": 468,
            "num_inference_steps": 4,
            "sliding_window_size": 243,
            "sliding_window_overlap": 17,
            "sliding_window_discard_last_frames": 0,
            "minimax_h3_multi_window": True,
            "sliding_window_memory_override": True,
            "minimax_h3_reference_sequence": "auto",
            "minimax_h3_text_encoder": "q2_k",
            "settings_version": 2.58,
        }
        common = {
            "params": dict(request),
            "media_info": {
                "width": 1280,
                "height": 704,
                "fps": 24,
            },
        }
        intermediate = {
            **common,
            "media_info": {
                **common["media_info"],
                "frames": 243,
                "duration_seconds": 10.125,
            },
            "multi_window_timing": {
                "window_count": 2,
                "completed_windows": 1,
                "scene_duration_seconds": 19.5,
            },
        }
        final = {
            **common,
            "media_info": {
                **common["media_info"],
                "frames": 468,
                "duration_seconds": 19.5,
            },
            "multi_window_timing": {
                "window_count": 2,
                "completed_windows": 2,
                "scene_duration_seconds": 19.5,
            },
        }
        matrix = one_case_matrix()
        matrix["request"] = request
        api = FakeApi(
            output_files=["sequence-window-1.mp4", "sequence-final.mp4"],
            output_metadata_by_filename={
                "sequence-window-1.mp4": intermediate,
                "sequence-final.mp4": final,
            },
        )
        with tempfile.TemporaryDirectory() as temp:
            result = BenchmarkRunner(
                api, matrix, Path(temp) / "bench", 10, 0.2, None, progress=False
            ).execute()

        run = result["runs"][0]
        self.assertEqual(run["job_terminal_status"], "completed")
        self.assertEqual(run["output_validation"]["status"], "verified")
        self.assertTrue(run["benchmark_eligible"])
        outputs = run["output_validation"]["outputs"]
        self.assertEqual([item["sequence_status"] for item in outputs], ["intermediate", "final"])
        self.assertEqual(
            [item["verification_status"] for item in outputs],
            ["intermediate", "verified"],
        )
        self.assertEqual(outputs[0]["media_info"]["frames"], 243)
        self.assertEqual(
            outputs[0]["multi_window_timing"],
            {
                "window_count": 2,
                "completed_windows": 1,
                "scene_duration_seconds": 19.5,
            },
        )
        self.assertEqual(outputs[1]["media_info"]["frames"], 468)

    def test_all_multi_window_outputs_intermediate_are_unverified(self):
        request = {
            "generation_mode": "video",
            "model_type": "minimax_h3_fused_turbo",
            "workspace": "Memory-Benchmark",
            "resolution": "auto_720p",
            "video_length": 468,
            "num_inference_steps": 4,
            "sliding_window_size": 243,
            "minimax_h3_multi_window": True,
            "minimax_h3_text_encoder": "q2_k",
            "settings_version": 2.58,
        }
        partial = {
            "params": dict(request),
            "media_info": {
                "frames": 243,
                "width": 1280,
                "height": 704,
                "fps": 24,
                "duration_seconds": 10.125,
            },
            "multi_window_timing": {
                "window_count": 2,
                "completed_windows": 1,
                "scene_duration_seconds": 19.5,
            },
        }
        matrix = one_case_matrix()
        matrix["request"] = request
        api = FakeApi(
            output_files=["partial-a.mp4", "partial-b.mp4"],
            output_metadata_by_filename={
                "partial-a.mp4": partial,
                "partial-b.mp4": partial,
            },
        )
        with tempfile.TemporaryDirectory() as temp:
            result = BenchmarkRunner(
                api, matrix, Path(temp) / "bench", 10, 0.2, None, progress=False
            ).execute()

        run = result["runs"][0]
        self.assertEqual(run["job_terminal_status"], "completed")
        self.assertEqual(run["output_validation"]["status"], "unverified")
        self.assertFalse(run["benchmark_eligible"])
        self.assertEqual(
            [item["sequence_status"] for item in run["output_validation"]["outputs"]],
            ["intermediate", "intermediate"],
        )
        self.assertTrue(any("No verified final output" in warning for warning in run["output_validation"]["warnings"]))

    def test_malformed_multi_window_timing_is_unverified(self):
        request = {
            "generation_mode": "video",
            "model_type": "minimax_h3_fused_turbo",
            "workspace": "Memory-Benchmark",
            "resolution": "auto_720p",
            "video_length": 468,
            "num_inference_steps": 4,
            "sliding_window_size": 243,
            "minimax_h3_multi_window": True,
            "minimax_h3_text_encoder": "q2_k",
            "settings_version": 2.58,
        }
        malformed = {
            "params": dict(request),
            "media_info": {
                "frames": 468,
                "width": 1280,
                "height": 704,
                "fps": 24,
                "duration_seconds": 19.5,
            },
            "multi_window_timing": {
                "window_count": 2,
                "completed_windows": "2",
                "scene_duration_seconds": 19.5,
            },
        }
        matrix = one_case_matrix()
        matrix["request"] = request
        with tempfile.TemporaryDirectory() as temp:
            result = BenchmarkRunner(
                FakeApi(output_metadata=malformed), matrix, Path(temp) / "bench",
                10, 0.2, None, progress=False,
            ).execute()

        run = result["runs"][0]
        self.assertEqual(run["job_terminal_status"], "completed")
        self.assertEqual(run["output_validation"]["status"], "unverified")
        self.assertFalse(run["benchmark_eligible"])
        self.assertEqual(run["output_validation"]["outputs"][0]["sequence_status"], "unverified")
        self.assertTrue(any("malformed multi-window completion timing" in warning for warning in run["output_validation"]["warnings"]))

    def test_shortened_final_multi_window_output_is_rejected(self):
        request = {
            "generation_mode": "video",
            "model_type": "minimax_h3_fused_turbo",
            "workspace": "Memory-Benchmark",
            "resolution": "auto_720p",
            "video_length": 468,
            "num_inference_steps": 4,
            "sliding_window_size": 243,
            "minimax_h3_multi_window": True,
            "minimax_h3_text_encoder": "q2_k",
            "settings_version": 2.58,
        }
        shortened_final = {
            "params": dict(request),
            "media_info": {
                "frames": 243,
                "width": 1280,
                "height": 704,
                "fps": 24,
                "duration_seconds": 10.125,
            },
            "multi_window_timing": {
                "window_count": 2,
                "completed_windows": 2,
                "scene_duration_seconds": 19.5,
            },
        }
        matrix = one_case_matrix()
        matrix["request"] = request
        with tempfile.TemporaryDirectory() as temp:
            result = BenchmarkRunner(
                FakeApi(output_metadata=shortened_final), matrix, Path(temp) / "bench",
                10, 0.2, None, progress=False,
            ).execute()

        run = result["runs"][0]
        self.assertEqual(run["job_terminal_status"], "completed")
        self.assertEqual(run["output_validation"]["status"], "workload_mismatch")
        self.assertFalse(run["benchmark_eligible"])
        self.assertEqual(run["output_validation"]["outputs"][0]["sequence_status"], "final")
        self.assertEqual(run["output_validation"]["outputs"][0]["verification_status"], "workload_mismatch")
        self.assertTrue(any(item["field"] == "video_length" for item in run["output_validation"]["mismatches"]))

    def test_metadata_lookup_failure_is_unverified_not_generation_failure(self):
        matrix = one_case_matrix()
        matrix["request"].update({
            "workspace": "Memory-Benchmark",
            "video_length": 124,
        })
        api = FakeApi(metadata_error=RuntimeError("metadata service unavailable"))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bench"
            result = BenchmarkRunner(
                api, matrix, out, 10, 0.2, None, progress=False
            ).execute()

        run = result["runs"][0]
        self.assertEqual(run["job_terminal_status"], "completed")
        self.assertEqual(run["status"], "unverified")
        self.assertIsNone(run["error"])
        self.assertFalse(run["benchmark_eligible"])
        self.assertEqual(run["output_validation"]["status"], "unverified")
        self.assertTrue(any("metadata service unavailable" in warning for warning in run["output_validation"]["warnings"]))
        self.assertEqual(result["benchmark_validation"]["status"], "unverified")
        self.assertEqual(result["benchmark_validation"]["unverified_runs"], 1)
        self.assertEqual(result["restoration"]["status"], "restored")

    def test_matching_frame_count_with_missing_steps_and_encoder_is_unverified(self):
        request = {
            "generation_mode": "video",
            "model_type": "minimax_h3_fused_turbo",
            "workspace": "Memory-Benchmark",
            "resolution": "auto_720p",
            "video_length": 124,
            "sliding_window_size": 124,
            "sliding_window_memory_override": True,
            "num_inference_steps": 5,
            "minimax_h3_text_encoder": "q2_k",
            "settings_version": 2.58,
        }
        metadata = {
            "params": {
                "video_length": 124,
                "sliding_window_size": 124,
                "sliding_window_memory_override": True,
                "settings_version": 2.58,
            },
            "media_info": {"frames": 124, "width": 1280, "height": 704},
        }
        matrix = one_case_matrix()
        matrix["request"] = request
        with tempfile.TemporaryDirectory() as temp:
            result = BenchmarkRunner(
                FakeApi(output_metadata=metadata), matrix, Path(temp) / "bench",
                10, 0.2, None, progress=False,
            ).execute()

        run = result["runs"][0]
        self.assertEqual(run["job_terminal_status"], "completed")
        self.assertEqual(run["output_validation"]["status"], "unverified")
        self.assertEqual(run["output_validation"]["mismatches"], [])
        self.assertFalse(run["benchmark_eligible"])
        warning_text = " ".join(run["output_validation"]["warnings"])
        self.assertIn("num_inference_steps", warning_text)
        self.assertIn("minimax_h3_text_encoder", warning_text)
        self.assertEqual(result["benchmark_validation"]["status"], "unverified")

    def test_native_h3_x2_checks_the_doubled_output_and_unchanged_workload(self):
        request = {"generation_mode": "video", "model_type": "minimax_h3_fused_turbo",
                   "workspace": "Memory-Benchmark", "resolution": "960x544",
                   "video_length": 124, "num_inference_steps": 6,
                   "spatial_upsampling": "h3_vae*2", "settings_version": 2.58}
        matrix = one_case_matrix()
        matrix["request"] = request
        for output_width, expected_status in ((1920, "verified"), (960, "workload_mismatch")):
            with self.subTest(output_width=output_width):
                metadata = {"params": dict(request),
                            "media_info": {"frames": 124, "width": output_width, "height": 1088}}
                with tempfile.TemporaryDirectory() as temp:
                    result = BenchmarkRunner(FakeApi(output_metadata=metadata), matrix,
                        Path(temp) / "bench", 10, 0.2, None, progress=False).execute()
                run = result["runs"][0]
                self.assertEqual(run["output_validation"]["status"], expected_status)
                self.assertEqual(run["benchmark_eligible"], expected_status == "verified")
                self.assertEqual(run["request_parameters"]["spatial_upsampling"], "h3_vae*2")

    def test_numeric_resolution_must_match_actual_media_dimensions(self):
        request = {
            "generation_mode": "video",
            "model_type": "minimax_h3_fused_turbo",
            "workspace": "Memory-Benchmark",
            "resolution": "1280x720",
            "video_length": 124,
            "sliding_window_size": 124,
            "num_inference_steps": 5,
            "minimax_h3_text_encoder": "q2_k",
            "settings_version": 2.58,
        }
        metadata = {
            "params": dict(request),
            "media_info": {"frames": 124, "width": 1280, "height": 704},
        }
        matrix = one_case_matrix()
        matrix["request"] = request
        with tempfile.TemporaryDirectory() as temp:
            result = BenchmarkRunner(
                FakeApi(output_metadata=metadata), matrix, Path(temp) / "bench",
                10, 0.2, None, progress=False,
            ).execute()

        run = result["runs"][0]
        self.assertEqual(run["job_terminal_status"], "completed")
        self.assertEqual(run["output_validation"]["status"], "workload_mismatch")
        self.assertTrue(run["benchmark_eligible"] is False)
        self.assertEqual(run["output_validation"]["mismatches"], [{
            "field": "resolution",
            "dimension": "height",
            "requested": 720,
            "effective": 704,
            "source": "media_info",
            "filename": "benchmark-output.mp4",
        }])
        self.assertEqual(result["benchmark_validation"]["status"], "workload_mismatch")

    def test_reference_matrices_pin_current_settings_schema(self):
        script_dir = Path(__file__).resolve().parents[1] / "app" / "scripts"
        for name in ("benchmark_memory.example.json", "benchmark_memory.3080.example.json"):
            with self.subTest(name=name):
                matrix = json.loads((script_dir / name).read_text(encoding="utf-8"))
                self.assertEqual(matrix["request"]["settings_version"], 2.58)

    def test_timeout_cancels_owned_job_and_restores_both_settings(self):
        api = FakeApi(never_finish=True)
        matrix = one_case_matrix({"video_profile": 3}, reload=True)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bench"
            runner = BenchmarkRunner(api, matrix, out, 1, 0.2, None, progress=False)
            with self.assertRaisesRegex(BenchmarkError, "exceeded"):
                runner.execute()
            self.assertEqual(api.system["video_profile"], 2)
            self.assertEqual(api.services["auto_performance"], True)
            self.assertEqual(api.cancelled, ["benchmark-job"])
            result = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["restoration"]["status"], "restored")
            self.assertEqual(result["runs"][0]["status"], "timed_out")
            self.assertEqual(result["runs"][0]["job_terminal_status"], "cancelled")
            self.assertEqual(result["runs"][0]["settings_used"]["video_profile"], 3)
            self.assertEqual(result["runs"][0]["request_parameters"]["model_type"], "wan-test")

    def test_submission_error_also_restores_settings(self):
        api = FakeApi(fail_submission=True)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bench"
            runner = BenchmarkRunner(
                api, one_case_matrix({"video_profile": 4}), out, 10, 0.2, None, progress=False
            )
            with self.assertRaisesRegex(BenchmarkError, "simulated submission error"):
                runner.execute()
            self.assertEqual(api.system["video_profile"], 2)
            self.assertTrue(api.services["auto_performance"])
            result = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["restoration"]["status"], "restored")

    def test_foreign_job_aborts_and_only_owned_job_is_cancelled(self):
        api = FakeApi(foreign_after_submission=True, never_finish=True)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bench"
            runner = BenchmarkRunner(
                api, one_case_matrix({"video_profile": 3}), out, 10, 0.2, None, progress=False
            )
            with self.assertRaisesRegex(BenchmarkError, "restoration needs recovery"):
                runner.execute()
            self.assertEqual(api.cancelled, ["benchmark-job"])
            self.assertEqual(api.system["video_profile"], 3)
            self.assertTrue((out / "settings-backup.json").is_file())
            result = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "restoration_failed")

    def test_restore_only_recovers_saved_snapshot_when_idle(self):
        api = FakeApi()
        api.system["video_profile"] = 3
        api.services["auto_performance"] = False
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "settings-backup.json").write_text(json.dumps({
                "schema_version": 1,
                "base_url": "http://127.0.0.1:42012",
                "system_config": {"video_profile": 2},
                "services": {"auto_performance": True},
            }), encoding="utf-8")
            result = restore_only(api, out)
            self.assertEqual(result["status"], "restored")
            self.assertEqual(api.system["video_profile"], 2)
            self.assertEqual(api.services["auto_performance"], True)

    def test_phase_aggregation_excludes_decode_and_terminal_stale_steps(self):
        samples = [
            {"elapsed_seconds": 0, "status_detail": {
                "status": "running", "phase": "Denoising", "step": 1, "total_steps": 8,
            }},
            {"elapsed_seconds": 4, "status_detail": {
                "status": "running", "phase": "Decoding VAE", "step": 8, "total_steps": 8,
            }},
            {"elapsed_seconds": 7, "status_detail": {
                "status": "completed", "phase": "Saving output", "step": 8, "total_steps": 8,
            }},
        ]
        timings = aggregate_timings(samples, 10)
        self.assertEqual(timings["denoise_time_seconds"], 4)
        self.assertEqual(timings["other_phase_seconds"], 6)
        self.assertEqual(sum(timings["phase_durations_seconds"].values()), 10)

    def test_peak_stats_keep_nvml_physical_units_and_log_evidence(self):
        samples = [
            {"system_stats": {
                "gpu": {"available": True, "vram_used_gb": 6.25},
                "ram": {"used_gb": 8.5},
            }},
            {"system_stats": {
                "gpu": {"available": False, "vram_used_gb": 99},
                "ram": {"used_gb": 9.0},
            }},
        ]
        self.assertEqual(_peak_metric(samples, "gpu", "vram_used_gb"), 6.25)
        self.assertEqual(_peak_metric(samples, "ram", "used_gb"), 9.0)
        evidence = parse_log_evidence(
            "[H3 Perf] SageAttention head split=7\nINT8 backend: Triton\n"
            "MMGP transformer budget 8192 MB\nordinary line"
        )
        self.assertEqual(
            [item["category"] for item in evidence],
            ["h3_perf", "attention", "int8", "mmgp_budget"],
        )

    def test_progress_model_phase_replaces_emoji_for_legacy_console_encodings(self):
        runner = BenchmarkRunner(
            api=None, matrix={}, output_dir=Path("."),
            timeout_seconds=1, poll_seconds=1, log_path=None,
        )
        for encoding in ("cp1252", "ascii"):
            with self.subTest(encoding=encoding):
                buffer = io.BytesIO()
                stream = io.TextIOWrapper(buffer, encoding=encoding, errors="strict")
                with redirect_stdout(stream):
                    runner._say("START model=🐈-DaSiWa phase=loading")
                rendered = buffer.getvalue().decode(encoding)
                self.assertIn("START model=?-DaSiWa phase=loading", rendered)

    def test_log_cursor_reads_new_bytes_and_recovers_from_truncate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "latest"
            path.write_bytes(b"old session\n")
            cursor = LogCursor(path)
            with path.open("ab") as handle:
                handle.write(b"[H3 Perf] new run\n")
            cursor.read_new()
            self.assertIn("new run", cursor.text())
            self.assertNotIn("old session", cursor.text())
            path.write_bytes(b"fresh\n")
            cursor.read_new()
            self.assertTrue(cursor.truncated)
            self.assertIn("fresh", cursor.text())
            self.assertTrue(cursor.metadata()["scope_ambiguous"])

    def test_rewritten_terminal_snapshot_keeps_new_memory_evidence_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "latest"
            path.write_bytes(
                b"[H3 Perf] old attention mode\n"
                b"old automatic preload=2048MB\n"
                b"scrollback tail\n"
            )
            cursor = LogCursor(path)
            # A redraw inserts a status header, retains old scrollback, and
            # appends the current run's automatic MMGP transfer decisions.
            path.write_bytes(
                b"Terminal redraw\r\n"
                b"[H3 Perf] old attention mode\r\n"
                b"old automatic preload=2048MB\r\n"
                b"scrollback tail\r\n"
                b"Automatic preload selected: 8192MB\r\n"
                b"MMGP shuttle transfer enabled\r\n"
                b"Smart memory pinning enabled\r\n"
            )
            cursor.read_new()
            captured = cursor.text()
            evidence = parse_log_evidence(captured)
            categories = {item["category"] for item in evidence}
            self.assertNotIn("old attention mode", captured)
            self.assertNotIn("old automatic preload=2048MB", captured)
            self.assertIn("Automatic preload selected: 8192MB", captured)
            self.assertTrue({"preload", "shuttle", "pinning"}.issubset(categories))
            metadata = cursor.metadata()
            self.assertTrue(metadata["rewritten_or_sliding_snapshot"])
            self.assertTrue(metadata["scope_ambiguous"])
            self.assertIn("snapshot_delta", metadata["capture_modes_seen"])

    def test_reflowed_pre_run_evidence_fragments_are_suppressed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "latest"
            path.write_bytes(
                b"[H3 Perf] attention SDPA selected at load\n"
                b"ordinary baseline context\n"
            )
            cursor = LogCursor(path)
            path.write_bytes(
                b"[H3 Perf] attention\n"
                b"SDPA selected at load\n"
                b"ordinary baseline context\n"
                b"MMGP transformer budget 8192 MB\n"
            )
            cursor.read_new()
            captured = cursor.text()
            self.assertNotIn("[H3 Perf] attention", captured)
            self.assertNotIn("SDPA selected at load", captured)
            self.assertIn("MMGP transformer budget 8192 MB", captured)
            self.assertTrue(cursor.metadata()["scope_ambiguous"])

    def test_append_mode_keeps_a_new_line_that_repeats_baseline_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "append.log"
            line = b"Automatic preload mode selected\n"
            path.write_bytes(line)
            cursor = LogCursor(path)
            with path.open("ab") as handle:
                handle.write(line)
            cursor.read_new()
            self.assertEqual(cursor.text(), "Automatic preload mode selected")
            self.assertFalse(cursor.metadata()["scope_ambiguous"])

    def test_appended_terminal_snapshot_frame_does_not_replay_old_scrollback(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "latest"
            prior_screen = b"Old H3 Perf attention evidence\nOld preload=2GB\n"
            path.write_bytes(prior_screen)
            cursor = LogCursor(path)
            with path.open("ab") as handle:
                handle.write(prior_screen)
                handle.write(b"Automatic preload=8GB shuttle and pinning enabled\n")
            cursor.read_new()
            captured = cursor.text()
            self.assertNotIn("Old H3 Perf", captured)
            self.assertNotIn("Old preload=2GB", captured)
            self.assertIn("Automatic preload=8GB", captured)
            self.assertTrue(cursor.metadata()["rewritten_or_sliding_snapshot"])
            self.assertTrue(cursor.metadata()["scope_ambiguous"])

    def test_append_continuation_of_pre_run_partial_line_is_not_attributed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "append.log"
            path.write_bytes(b"old H3 Perf partial")
            cursor = LogCursor(path)
            with path.open("ab") as handle:
                handle.write(b" continuation\nnew H3 Perf current run\n")
            cursor.read_new()
            captured = cursor.text()
            self.assertNotIn("old H3 Perf partial", captured)
            self.assertIn("new H3 Perf current run", captured)
            self.assertTrue(cursor.metadata()["scope_ambiguous"])

    def test_rewritten_snapshot_does_not_record_an_unterminated_new_row(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "latest"
            path.write_bytes(b"old scrollback\n")
            cursor = LogCursor(path)
            path.write_bytes(b"redrawn header\nold scrollback\nAutomatic preload selected 8192MB")
            cursor.read_new()
            self.assertNotIn("Automatic preload selected", cursor.text())
            self.assertTrue(cursor.metadata()["scope_ambiguous"])
            self.assertIn("snapshot_ends_with_an_unterminated_line", cursor.metadata()["ambiguity_reasons"])

    def test_rewritten_snapshot_partial_then_complete_does_not_leak_scrollback(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "latest"
            path.write_bytes(b"old H3 Perf record\nold scrollback\n")
            cursor = LogCursor(path)
            path.write_bytes(b"[H3 Perf] current")
            cursor.read_new()
            self.assertNotIn("[H3 Perf] current", cursor.text())
            path.write_bytes(
                b"[H3 Perf] current preload decision\n"
                b"old H3 Perf record\nold scrollback\n"
                b"MMGP shuttle enabled\n"
            )
            cursor.read_new()
            captured = cursor.text()
            self.assertIn("[H3 Perf] current preload decision", captured)
            self.assertIn("MMGP shuttle enabled", captured)
            self.assertNotIn("old H3 Perf record", captured)
            self.assertTrue(cursor.metadata()["scope_ambiguous"])


if __name__ == "__main__":
    unittest.main()
