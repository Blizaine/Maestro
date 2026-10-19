"""CPU-only safety and contract tests for the local Test Bench harness."""
from __future__ import annotations

import ctypes
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import testbench.__main__ as harness  # noqa: E402 - app-local import after sys.path setup


class DigestTests(unittest.TestCase):
    def test_source_digest_normalizes_line_endings_but_detects_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lf = root / "lf" / "module.py"
            crlf = root / "crlf" / "module.py"
            lf.parent.mkdir()
            crlf.parent.mkdir()
            lf.write_bytes(b"def value():\n    return 1\n")
            crlf.write_bytes(b"def value():\r\n    return 1\r\n")
            self.assertEqual(harness._digest_named_source_files([lf]), harness._digest_named_source_files([crlf]))
            crlf.write_bytes(b"def value():\r\n    return 2\r\n")
            self.assertNotEqual(harness._digest_named_source_files([lf]), harness._digest_named_source_files([crlf]))

    def test_suite_digest_is_canonical_across_line_endings_and_detects_case_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lf = root / "lf.json"
            crlf = root / "crlf.json"
            lf.write_bytes(b'{"version":1,"cases":[{"id":"kite","prompt":"red kite"}]}\n')
            crlf.write_bytes(b'{\r\n  "cases": [{"prompt": "red kite", "id": "kite"}],\r\n  "version": 1\r\n}\r\n')
            self.assertEqual(harness.suite_digest(lf), harness.suite_digest(crlf))
            crlf.write_bytes(crlf.read_bytes().replace(b"red kite", b"blue kite"))
            self.assertNotEqual(harness.suite_digest(lf), harness.suite_digest(crlf))

    def test_dirty_fingerprint_includes_tracked_and_untracked_source_and_filters_private_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / "lf", Path(temporary) / "crlf"]
            paths = ["app/src/tracked.py", "app/src/new.js", "app/outputs/result.py", "app/.env"]
            for root, ending in zip(roots, (b"\n", b"\r\n")):
                (root / "app" / "src").mkdir(parents=True)
                (root / "app" / "outputs").mkdir(parents=True)
                (root / "app" / "src" / "tracked.py").write_bytes(b"VALUE = 1" + ending)
                (root / "app" / "src" / "new.js").write_bytes(b"const value = 2;" + ending)
                (root / "app" / "outputs" / "result.py").write_text("private output", encoding="utf-8")
                (root / "app" / ".env").write_text("TOKEN=not-included", encoding="utf-8")
            lf_digest = harness._dirty_source_fingerprint(roots[0], paths)
            crlf_digest = harness._dirty_source_fingerprint(roots[1], paths)
            self.assertEqual(lf_digest["sha256"], crlf_digest["sha256"])
            self.assertEqual(lf_digest["files_hashed"], 2)
            self.assertEqual(lf_digest["excluded_paths"], 2)
            self.assertEqual(lf_digest["status"], "complete")
            (roots[1] / "app" / "src" / "tracked.py").write_bytes(b"VALUE = 3\r\n")
            changed = harness._dirty_source_fingerprint(roots[1], paths)
            self.assertNotEqual(lf_digest["sha256"], changed["sha256"])

    def test_dirty_fingerprint_skips_secret_literals_and_reports_file_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "app" / "src").mkdir(parents=True)
            (root / "app" / "src" / "secret.py").write_text('API_KEY = "sk-very-long-secret-value"\n', encoding="utf-8")
            normal = root / "app" / "src" / "normal.py"
            normal.write_text("VALUE = 1\n", encoding="utf-8")
            secret = harness._dirty_source_fingerprint(root, ["app/src/secret.py"])
            self.assertEqual(secret["files_hashed"], 0)
            self.assertEqual(secret["skipped_paths"], 1)
            self.assertEqual(secret["status"], "partial")
            with patch.object(harness, "GIT_FINGERPRINT_MAX_FILES", 1):
                capped = harness._dirty_source_fingerprint(root, ["app/src/normal.py", "app/src/deleted.py"])
            self.assertEqual(capped["files_hashed"], 1)
            self.assertEqual(capped["status"], "partial")

    def test_git_diagnostics_hashes_file_contents_and_labels_the_filter_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "app" / "src" / "main.py"
            untracked = root / "app" / "src" / "new.py"
            (root / "app" / "src").mkdir(parents=True)
            (root / "app" / "models").mkdir(parents=True)
            source.write_bytes(b"VALUE = 1\r\n")
            untracked.write_bytes(b"NEW = 2\n")
            (root / "app" / "models" / "h3_handler.py").write_bytes(b"MODEL = 'H3'\n")

            def fake_run(command, **_kwargs):
                if command[1] == "rev-parse":
                    output = "deadbeef"
                elif command[1:3] == ["diff", "--name-only"]:
                    output = "app/src/main.py\0"
                elif command[1] == "ls-files":
                    output = "app/src/new.py\0app/outputs/not-source.py\0app/models/h3_handler.py\0"
                else:
                    raise AssertionError(command)
                return {"status": "available", "output": output, "truncated": False}

            with patch.object(harness, "ROOT", root), patch.object(harness, "_run_bounded", side_effect=fake_run):
                first = harness._git_diagnostics()
                source.write_bytes(b"VALUE = 1\n")
                normalized = harness._git_diagnostics()
                untracked.write_bytes(b"NEW = 3\n")
                changed = harness._git_diagnostics()
            self.assertEqual(first["diff_sha256"], normalized["diff_sha256"])
            self.assertNotEqual(normalized["diff_sha256"], changed["diff_sha256"])
            self.assertEqual(changed["diff_files_hashed"], 3)
            self.assertEqual(changed["diff_excluded_paths"], 1)
            self.assertEqual(changed["diff_status"], "complete")
            self.assertIn("normalized UTF-8", changed["diff_digest_scope"])
            self.assertIn("checkpoint/weight", changed["diff_digest_scope"])
            self.assertIn("app/models implementation source", changed["diff_digest_scope"])


class FakeApi:
    def __init__(self, *, busy=False, offline=False, downloaded=True, cuda=True):
        self.base_url = "http://127.0.0.1:42012"
        self.busy = busy
        self.offline = offline
        self.downloaded = downloaded
        self.cuda = cuda
        self.calls = []
        self.system = {
            "video_profile": 2, "video_preload_mode": "balanced",
            "video_preload_in_VRAM": True, "int8_kernels": False,
            "read_ahead": 2, "vram_allocator": "current-allocator",
        }

    def get(self, path):
        self.calls.append(path)
        if self.offline:
            raise harness.DeferredError("offline fake")
        if path == "/api/v1/jobs":
            return {"jobs": [{"status": "running"}] if self.busy else []}
        if path == "/api/v1/llm/stream-status":
            return {"done": not self.busy}
        if path == "/api/v1/models/downloads/status":
            return {"downloads": {}}
        if path == "/api/v1/system-config":
            return dict(self.system)
        if path == "/api/v1/services-config":
            return {"auto_performance": False}
        if path == "/api/v1/system-stats":
            return {"gpu": {"available": self.cuda, "gpu_name": "Fake CUDA", "vram_total_gb": 24}}
        if path == "/api/v1/models":
            return {"models": [{"model_type": harness.MODEL_TYPE, "architecture": "H3", "is_downloaded": self.downloaded}]}
        if path == "/api/v1/system-detect":
            return {
                "hardware": {
                    "cuda_available": self.cuda, "gpu_name": "Fake CUDA",
                    "gpu_vram_gb": 24, "gpu_capability": "sm_fake",
                    "driver_version": "fake-driver", "machine": "fake-host",
                },
                "recommended": {
                    "video_profile": 1, "vram_allocator": "different-allocator",
                    "smart_memory_pinning": True,
                },
            }
        raise AssertionError(f"Unexpected fake API path: {path}")

    def post(self, *args, **kwargs):
        raise AssertionError("Test Bench must not use FakeApi POST")


def snapshot_for(api: FakeApi) -> dict:
    return harness.server_snapshot(api)


class LoopbackTests(unittest.TestCase):
    def test_base_url_is_restricted_to_a_local_origin(self):
        self.assertEqual(harness.validate_base_url("HTTP://127.0.0.1:42012/"), "http://127.0.0.1:42012")
        self.assertEqual(harness.validate_base_url("http://[::1]:42012"), "http://[::1]:42012")
        for value in (
            "https://example.com", "http://127.0.0.1:42012/api", "http://user:pass@localhost:5",
            "http://localhost:5/?token=secret", "file://localhost/tmp",
        ):
            with self.subTest(value=value), self.assertRaises(harness.HarnessError):
                harness.validate_base_url(value)

    def test_jobs_endpoint_gets_a_separate_bounded_response_cap(self):
        class Response:
            def __init__(self):
                self.requested = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, amount):
                self.requested = amount
                return b"{}"

        response = Response()
        client = harness.ApiClient("http://127.0.0.1:42012")
        client.opener.open = lambda *_args, **_kwargs: response
        self.assertEqual(client.get("/api/v1/jobs"), {})
        self.assertEqual(response.requested, harness.JOBS_RESPONSE_LIMIT + 1)
        self.assertEqual(harness.JOBS_RESPONSE_LIMIT, 8 * 1024 * 1024)


class ReceiptTests(unittest.TestCase):
    def test_busy_prompt_mode_defers_before_any_case_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "Test-Bench"
            api = FakeApi(busy=True)
            with patch.object(harness, "diagnostics", return_value={"cpu_test": True}), \
                 patch.object(harness, "_run_prompt_stage", side_effect=AssertionError("must defer before runner")):
                bench = harness.TestBench("prompt", api.base_url, workspace=workspace, api_factory=lambda _url: api)
                self.assertEqual(bench.execute(), 2)
            latest = harness.read_json(workspace / "testbench-latest.json")
            self.assertEqual(latest["status"], "deferred")
            self.assertTrue(latest["finished_at"])
            self.assertTrue((workspace / latest["report_path"]).is_file())
            self.assertTrue((workspace / latest["artifact_filename"]).is_file())
            self.assertTrue((workspace / latest["run_id"] / "diagnostics.json").is_file())
            self.assertEqual(api.calls.count("/api/v1/jobs"), 1)

    def test_offline_smoke_receipt_is_deferred_and_keeps_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "Test-Bench"
            api = FakeApi(offline=True)
            with patch.object(harness, "diagnostics", return_value={"cpu_test": True}):
                bench = harness.TestBench("smoke", api.base_url, workspace=workspace, api_factory=lambda _url: api)
                self.assertEqual(bench.execute(), 2)
            latest = harness.read_json(workspace / "testbench-latest.json")
            self.assertEqual(latest["status"], "deferred")
            self.assertTrue(latest["finished_at"])
            self.assertTrue((workspace / latest["run_id"] / "diagnostics.json").is_file())
            self.assertTrue((workspace / latest["report_path"]).is_file())

    def test_all_modes_collect_bounded_diagnostics_even_without_an_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "Test-Bench"
            with patch.object(harness, "diagnostics", return_value={"runtime": {"python": "fake"}}):
                bench = harness.TestBench("prompt", workspace=workspace)
                self.assertEqual(bench.execute(), 2)
            latest = harness.read_json(workspace / "testbench-latest.json")
            self.assertEqual(latest["status"], "deferred")
            self.assertEqual(
                harness.read_json(workspace / latest["run_id"] / "diagnostics.json"),
                {"runtime": {"python": "fake"}},
            )

    def test_interrupted_or_uncertain_receipt_blocks_a_blind_retry(self):
        for status in ("uncertain", "interrupted", "restoration_pending"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary) / "Test-Bench"
                workspace.mkdir()
                harness.write_json(workspace / "testbench-latest.json", {"run_id": "prior", "status": status})
                def api_factory(_url):
                    self.fail("must reconcile before API access")

                bench = harness.TestBench("prompt", "http://127.0.0.1:42012", workspace=workspace,
                                          api_factory=api_factory)
                self.assertEqual(bench.execute(), 4)
                self.assertFalse((workspace / bench.run_id).exists())
                self.assertEqual(harness.read_json(workspace / "testbench-latest.json")["status"], status)


class PromptStageTests(unittest.TestCase):
    def test_shared_prompt_cases_validate_with_existing_promptbench_contract(self):
        from promptbench import runner as prompt_runner

        with tempfile.TemporaryDirectory() as temporary:
            suite_path = Path(temporary) / "suite.json"
            suite = json.loads(harness.CASES_FILE.read_text(encoding="utf-8"))
            harness.write_json(suite_path, {"version": 1, "cases": suite["prompt_cases"]})
            loaded = prompt_runner.load_suite(suite_path)
        party = loaded["cases"][0]
        self.assertIn("Jerry Seinfeld, Michael Scott", party["params"]["prompt"])
        self.assertEqual(party["params"]["video_length"] / 24, 10.125)
        self.assertEqual(party["dialogue_word_target"], {"minimum": 22, "maximum": 30})
        kite = loaded["cases"][1]
        self.assertTrue(kite["checks"]["no_dialogue"])
        self.assertEqual(kite["params"]["resolution"], "864x480")
        self.assertEqual(kite["params"]["video_length"], 124)
        self.assertEqual(kite["params"]["seed"], 424242)
        self.assertEqual(kite["params"]["num_inference_steps"], 4)

    def test_prompt_runner_receives_one_writer_two_cases_and_strict_limits(self):
        from promptbench import runner as prompt_runner

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            api = FakeApi()
            captured = {}

            def fake_run(args):
                captured.update(vars(args))
                suite = prompt_runner.load_suite(args.suite)
                attempt_dir = Path(args.output) / "attempts"
                attempt_dir.mkdir(parents=True)
                for index, case in enumerate(suite["cases"]):
                    source_warning = (
                        "The AI draft did not produce a valid H3 frame prompt. "
                        "Review the source-based draft before generating."
                    )
                    harness.write_json(attempt_dir / f"{index}.json", {
                        "case_id": case["id"], "status": "complete",
                        "result": {"prepared": {
                            "h3_window_plan": {"planned_by": "native_fallback" if index == 0 else None},
                            "enhancement_review_required": index == 1,
                            "enhancement_warnings": ["native fallback"] if index == 0 else [source_warning],
                        }},
                        "assessment": {
                            "calls": 2, "warnings": ["native fallback"] if index == 0 else [source_warning],
                            "spoken_spans": [["Jerry", "[English] one two"]],
                        },
                    })
                harness.write_json(Path(args.output) / "manifest.json", {"status": "complete"})
                (Path(args.output) / "report.html").write_text("<p>child</p>", encoding="utf-8")
                return 0

            original_client = prompt_runner.Client
            with patch.object(prompt_runner, "run", side_effect=fake_run):
                result = harness._run_prompt_stage(api, output, api.base_url)
            self.assertIs(prompt_runner.Client, original_client)
            self.assertEqual(captured["writer"], ["gemma"])
            self.assertFalse(captured["resume"])
            self.assertEqual(captured["limit"], 2)
            self.assertEqual(captured["case_timeout"], 300)
            self.assertEqual(captured["max_calls"], 12)
            self.assertEqual(captured["repetitions"], 1)
            self.assertEqual(len(prompt_runner.load_suite(captured["suite"])["cases"]), 2)
            self.assertEqual(result["attempt_count"], 2)
            self.assertEqual(result["native_fallback_count"], 2)
            self.assertEqual(result["source_based_fallback_count"], 1)
            self.assertEqual(result["attempts"][1]["fallback_type"], "source_based_draft")
            self.assertFalse(result["attempts"][1]["enhanced_success"])
            self.assertEqual(result["attempts"][0]["dialogue_words"], 2)
            self.assertFalse(result["attempts"][0]["dialogue_target_met"])

    def test_missing_gemma_is_deferred_before_any_prompt_post(self):
        from promptbench import PROTOCOL_VERSION, runner as prompt_runner

        class NoWriterClient:
            calls = []

            def __init__(self, base):
                self.base = base

            def call(self, path, data=None, timeout=30):
                self.calls.append((path, data))
                if path == "dev/prompt-bench":
                    return {"protocol": PROTOCOL_VERSION, "writers": []}
                raise AssertionError("No generation or mutating endpoint may be called")

        with tempfile.TemporaryDirectory() as temporary:
            api = FakeApi()
            with patch.object(prompt_runner, "Client", NoWriterClient):
                result = harness._run_prompt_stage(api, Path(temporary) / "run", api.base_url)
            self.assertEqual(result["status"], "deferred")
            self.assertIn("not installed", result["error"].lower())
            self.assertEqual(NoWriterClient.calls, [("dev/prompt-bench", None)])


class RenderStageTests(unittest.TestCase):
    def test_current_render_uses_two_cold_warm_runs_and_no_profile_guess(self):
        snapshot = snapshot_for(FakeApi())
        matrix = harness.build_matrix(snapshot, compare=False)
        case = matrix["cases"][0]
        self.assertEqual(case["repeats"], 2)
        self.assertTrue(case["reload"])
        self.assertEqual(case["settings"]["video_profile"], 2)
        self.assertEqual(matrix["request"]["video_length"], 124)
        self.assertEqual(matrix["request"]["resolution"], "864x480")
        self.assertEqual(matrix["request"]["seed"], 424242)
        self.assertEqual(matrix["request"]["num_inference_steps"], 4)

    def test_compare_changes_only_video_profile_and_labels_cold_sample_scope(self):
        snapshot = snapshot_for(FakeApi())
        matrix = harness.build_matrix(snapshot, compare=True)
        current, recommended = matrix["cases"]
        changed = {key for key in current["settings"] | recommended["settings"]
                   if current["settings"].get(key) != recommended["settings"].get(key)}
        self.assertEqual(changed, {"video_profile"})
        self.assertEqual([case["repeats"] for case in matrix["cases"]], [1, 1])
        self.assertTrue(all(case["reload"] for case in matrix["cases"]))
        self.assertIn("one cold sample each", matrix["selection"])
        self.assertNotEqual(recommended["settings"].get("vram_allocator"), "different-allocator")

    def test_render_requires_all_h3_components_and_cuda_before_runner_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            for api in (FakeApi(downloaded=False), FakeApi(cuda=False)):
                with self.subTest(downloaded=api.downloaded, cuda=api.cuda):
                    snapshot = snapshot_for(api)
                    with patch.object(harness, "ControlledBenchmarkRunner", side_effect=AssertionError("must not create runner")):
                        with self.assertRaises(harness.DeferredError):
                            harness._run_render_stage(api, snapshot, Path(temporary) / str(api.downloaded),
                                                      compare=False, timeout_seconds=10,
                                                      before_run=lambda: {}, stop_requested=lambda: False)

    def test_stop_and_new_busy_state_prevent_render_submit(self):
        matrix = harness.build_matrix(snapshot_for(FakeApi()), compare=False)
        with tempfile.TemporaryDirectory() as temporary:
            for stop, busy in ((True, False), (False, True)):
                reached = []
                runner = harness.ControlledBenchmarkRunner(
                    harness.memory_bench.ApiClient("http://127.0.0.1:42012"), matrix,
                    Path(temporary) / f"{stop}-{busy}", 10, 1, None,
                    before_run=lambda: {"busy": busy}, stop_requested=lambda: stop,
                    progress=False,
                )
                with patch.object(harness.memory_bench.BenchmarkRunner, "_run_one",
                                  side_effect=lambda *_args: reached.append(True)):
                    with self.assertRaises(harness.StopRequested):
                        runner._run_one(matrix["cases"][0], 1, True)
                self.assertEqual(reached, [])

    def test_runner_restoration_and_failed_generation_status_are_preserved(self):
        api = FakeApi()
        snapshot = snapshot_for(api)

        class FakeRunner:
            outcome = {}

            def __init__(self, _client, matrix, _output, *_args, **_kwargs):
                self.matrix = matrix
                self.halt_reason = None
                FakeRunner.last_matrix = matrix

            def execute(self):
                return dict(FakeRunner.outcome)

        with tempfile.TemporaryDirectory() as temporary, patch.object(harness, "ControlledBenchmarkRunner", FakeRunner):
            cases = (
                ({"status": "completed_with_failures", "restoration": {"status": "restored"},
                  "runs": [{"status": "failed", "oom": True}]}, "failed"),
                ({"status": "completed", "restoration": {"status": "pending"}, "runs": []}, "uncertain"),
                ({"status": "completed", "restoration": None, "runs": []}, "failed"),
            )
            for index, (outcome, expected) in enumerate(cases):
                FakeRunner.outcome = outcome
                result = harness._run_render_stage(api, snapshot, Path(temporary) / str(index), compare=False,
                                                   timeout_seconds=10, before_run=lambda: {},
                                                   stop_requested=lambda: False)
                self.assertEqual(result["status"], expected)
                if outcome["restoration"] is not None:
                    self.assertEqual(result["restoration"], outcome["restoration"])
            self.assertEqual(FakeRunner.last_matrix["cases"][0]["repeats"], 2)

    def test_overall_status_handles_missing_restoration_as_failure(self):
        bench = harness.TestBench("render", workspace=Path("C:/not-created"))
        bench.manifest["stages"] = {
            "render": {"status": "failed", "result": {"restoration": None}},
        }
        self.assertEqual(bench._overall_status()[0], "failed")


class DiagnosticsAndArtifactsTests(unittest.TestCase):
    def test_win32_process_check_never_uses_os_kill_and_fails_closed_on_access_denied(self):
        class Kernel:
            def __init__(self, exit_code=259, handle=123):
                self.exit_code = exit_code
                self.handle = handle
                self.closed = []

            def OpenProcess(self, _access, _inherit, _pid):
                return self.handle

            def GetExitCodeProcess(self, _handle, pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32)).contents.value = self.exit_code
                return 1

            def CloseHandle(self, handle):
                self.closed.append(handle)

        with patch.object(harness.os, "kill", side_effect=AssertionError("unsafe Windows liveness check")):
            active = Kernel()
            self.assertTrue(harness._alive(123, platform_name="win32", kernel32=active))
            self.assertEqual(active.closed, [123])
            self.assertFalse(harness._alive(123, platform_name="win32", kernel32=Kernel(exit_code=0)))
            denied = Kernel(handle=0)
            with patch.object(harness.ctypes, "get_last_error", return_value=5, create=True):
                self.assertTrue(harness._alive(123, platform_name="win32", kernel32=denied))

    def test_bundle_redacts_secrets_and_paths_and_excludes_child_html(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            harness.write_json(run_dir / "manifest.json", {
                "status": "completed", "api_key": "sk-secret", "message": "failed at C:\\Users\\me\\weights\\x.safetensors",
                "output_files": ["C:\\Users\\me\\outputs\\kite.mp4"],
            })
            harness.write_json(run_dir / "prompt" / "manifest.json", {"access_token": "secret-value"})
            (run_dir / "prompt" / "report.html").parent.mkdir(parents=True, exist_ok=True)
            (run_dir / "prompt" / "report.html").write_text("sensitive child report", encoding="utf-8")
            bundle = root / "bundle.zip"
            with patch.object(harness, "_tail_logs", return_value={
                "llama.log": b"Authorization: Bearer abcdef\nmodel C:\\private\\w.safetensors\n",
            }):
                harness.create_bundle(run_dir, bundle, {})
            self.assertLessEqual(bundle.stat().st_size, harness.BUNDLE_LIMIT)
            with zipfile.ZipFile(bundle) as archive:
                names = set(archive.namelist())
                self.assertNotIn("prompt/report.html", names)
                text = "\n".join(archive.read(name).decode("utf-8", "replace")
                                 for name in names if name.endswith((".json", ".log")))
            self.assertNotIn("sk-secret", text)
            self.assertNotIn("secret-value", text)
            self.assertNotIn("C:\\Users\\me", text)
            self.assertNotIn("C:\\private", text)
            self.assertIn("kite.mp4", text)

    def test_render_report_exposes_per_run_metrics_and_safe_output_links(self):
        manifest = {
            "run_id": "cpu-run", "mode": "render", "status": "completed_with_warnings",
            "started_at": "a", "finished_at": "b", "code_digest": "c", "suite_digest": "s",
            "artifact_filename": "cpu-run.zip", "stages": {
                "render": {"status": "completed_with_warnings", "result": {
                    "manual_visual_quality": "not tested",
                    "runs": [{
                        "case_name": "kite <cold>", "repeat": 1, "reload_before_run": True,
                        "status": "completed", "wall_time_seconds": 10.5,
                        "denoise_time_seconds": 8.0, "peak_physical_vram_gb": 20.25,
                        "peak_ram_used_gb": 40.5, "settings_used": {"video_profile": 2},
                        "output_files": [r"C:\secret\kite.mp4"],
                    }],
                }},
            },
        }
        report = harness._html_report(manifest)
        self.assertIn("10.5s", report)
        self.assertIn("8.0s", report)
        self.assertIn("20.25 GiB", report)
        self.assertIn("40.5 GiB", report)
        self.assertIn("video_profile", report)
        self.assertIn("kite.mp4", report)
        self.assertIn("workspace=Test-Bench", report)
        self.assertNotIn(r"C:\secret", report)
        self.assertIn("kite &lt;cold&gt;", report)


if __name__ == "__main__":
    unittest.main()
