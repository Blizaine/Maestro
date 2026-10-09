import io
import json
from contextlib import redirect_stderr
from email.message import Message
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.scripts.benchmark_wangp import (
    BenchmarkRunner, BenchmarkError, MCPToolError, StreamableHTTPMCP, WanGPClient,
    _sse_messages, main as benchmark_main, normalize_manifest, validate_base_url,
    write_result,
)


def make_manifest(repeats=1):
    return normalize_manifest({
        "name": "h3-pruned-comparison",
        "settings": {
            "model_type": "minimax_h3_fl2va_pruned",
            "config": "gguf_q2_k,int8_convrot",
            "prompt": "A quiet cinematic shot of a red kite over a field.",
            "resolution": "864x480",
            "video_length": 124,
            "num_inference_steps": 20,
            "flow_shift": 7.0,
            "seed": 424242,
            "repeat_generation": 1,
            "prompt_enhancer": "",
            "activated_loras": [],
            "image_prompt_type": "",
            "video_prompt_type": "",
            "audio_prompt_type": "",
            "sliding_window_size": 124,
            "sliding_window_overlap": 18,
            "sliding_window_discard_last_frames": 0,
            "override_attention": "sage2",
            "override_profile": 4,
        },
        "cases": [{"name": "480p", "repeats": repeats}],
    })


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeTransport:
    request_timeout = 20.0
    origin = "http://127.0.0.1:42019"
    endpoint = origin + "/mcp"

    def __init__(self, *, queues=None, never_finish=False, model_defaults=None, job_snapshots=None):
        self.calls = []
        self.queues = list(queues or [])
        self.never_finish = never_finish
        self.job_snapshots = list(job_snapshots or [])
        self.model_defaults = model_defaults or {
            "model_type": "minimax_h3_fl2va_pruned", "prompt": "",
            "video_length": 81, "num_inference_steps": 30, "flow_shift": 12.0,
        }
        self.job_number = 0
        self.cancelled = []
        self.tools = [
            {"name": "wangp_generate", "inputSchema": {"properties": {"wait": {"type": "boolean"}}}},
            {"name": "wangp_session", "inputSchema": {"type": "object"}},
            {"name": "wangp_model", "inputSchema": {"type": "object"}},
            {"name": "wangp_toolbox", "inputSchema": {"type": "object"}},
        ]

    def initialize(self):
        pass

    def list_tools(self):
        return self.tools

    def call_tool(self, name, arguments, timeout=None):
        self.calls.append((name, arguments, timeout))
        if name == "wangp_model":
            action = arguments["action"]
            if action == "defaults":
                return dict(self.model_defaults)
            if action == "capabilities":
                return {"properties": {"prompt": {"type": "string"}}}
            return {"profiles": [4, 5]}
        if name == "wangp_session":
            action = arguments["action"]
            if action == "list_queue":
                return self.queues.pop(0) if self.queues else {"total_count": 0}
            job_id = arguments["arguments"]["job_id"]
            if action == "get_job" and self.job_snapshots:
                return self.job_snapshots.pop(0)
            if action == "cancel_job":
                self.cancelled.append(job_id)
                return {"job_id": job_id, "done": False, "events": [{"kind": "status", "text": "cancel requested"}]}
            if self.never_finish and not self.cancelled:
                return {"job_id": job_id, "done": False, "events": [{"kind": "progress", "step": 1}]}
            return {
                "job_id": job_id, "done": True, "created_at": 100.0, "updated_at": 102.0,
                "events": [{"kind": "progress", "step": 1}, {"kind": "completed"}],
                "result": {
                    "success": True, "generated_files": ["output.mp4"], "errors": [],
                    "gallery_items": [{"media_id": "media-1", "filename": "output.mp4"}],
                },
            }
        if name == "wangp_generate":
            self.job_number += 1
            return {"job_id": f"owned-{self.job_number}", "done": False, "events": [{"kind": "started"}]}
        if name == "wangp_toolbox":
            return {"settings": {"config": "gguf_q2_k,int8_convrot", "resolution": "864x480", "video_length": 124}}
        raise AssertionError(f"Unexpected MCP tool: {name}")


class FakeHTTPResponse:
    def __init__(self, body=b"", headers=None):
        self.stream = io.BytesIO(body)
        self.headers = Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value

    def read(self, size=-1):
        return self.stream.read(size)

    def close(self):
        self.stream.close()


class FakeEndpoint:
    def __init__(self):
        self.requests = []

    def __call__(self, request, timeout):
        body = json.loads(request.data.decode("utf-8"))
        headers = {key.lower(): value for key, value in request.header_items()}
        self.requests.append((body, headers, timeout))
        if body.get("method") == "initialize":
            result = {
                "protocolVersion": "2025-03-26", "capabilities": {},
                "serverInfo": {"name": "WanGP", "version": "17.17"},
            }
            response_headers = {"Content-Type": "application/json", "Mcp-Session-Id": "session-1"}
        elif body.get("method") == "notifications/initialized":
            return FakeHTTPResponse(b"", {"Content-Type": "application/json"})
        elif body.get("method") == "tools/call":
            result = {
                "content": [{"type": "text", "text": "{\"ok\": true}"}],
                "structuredContent": {"ok": True}, "isError": False,
            }
            response_headers = {"Content-Type": "application/json"}
        else:
            raise AssertionError(f"Unexpected method: {body.get('method')}")
        return FakeHTTPResponse(
            json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode(),
            response_headers,
        )


class WangPBenchmarkTests(unittest.TestCase):
    def test_base_url_and_manifest_bounds_and_model(self):
        self.assertEqual(validate_base_url("http://localhost:42019/mcp"), (
            "http://localhost:42019", "http://localhost:42019/mcp"
        ))
        with self.assertRaises(BenchmarkError):
            validate_base_url("http://localhost:42019/mcp?token=bad")
        with self.assertRaisesRegex(BenchmarkError, "model_type"):
            normalize_manifest({
                "settings": {"model_type": "wrong", "prompt": "test"},
                "cases": [{"name": "bad"}],
            })
        expanded = make_manifest(repeats=2)["runs"]
        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[0]["settings"]["video_length"], 124)
        self.assertEqual(expanded[0]["settings"]["seed"], 424242)

    def test_unsupported_h3_settings_are_rejected_instead_of_stripped(self):
        for field in ("generation_mode", "guidance_scale", "negative_prompt"):
            with self.subTest(field=field):
                manifest = {
                    "settings": {
                        "model_type": "minimax_h3_fl2va_pruned", "prompt": "test", field: "x",
                    },
                    "cases": [{"name": "bad"}],
                }
                with self.assertRaisesRegex(BenchmarkError, field):
                    normalize_manifest(manifest)

    def test_missing_h3_default_fails_before_any_submission(self):
        transport = FakeTransport(model_defaults={
            "model_type": "minimax_h3_fl2va_pruned", "prompt": "",
            "video_length": 81, "num_inference_steps": 30,
        })
        result = BenchmarkRunner(WanGPClient(transport), make_manifest(), timeout_seconds=5).run()
        self.assertEqual(result["status"], "failed")
        self.assertIn("flow_shift", result["errors"][0])
        self.assertFalse(any(call[0] == "wangp_generate" for call in transport.calls))

    def test_streamable_http_session_headers_and_structured_result(self):
        endpoint = FakeEndpoint()
        client = StreamableHTTPMCP(
            "http://127.0.0.1:42019", request_timeout=8, opener=endpoint
        )
        client.initialize()
        self.assertEqual(client.call_tool("wangp_model", {"action": "defaults"}), {"ok": True})
        self.assertEqual(endpoint.requests[0][0]["method"], "initialize")
        self.assertEqual(endpoint.requests[1][0]["method"], "notifications/initialized")
        self.assertEqual(endpoint.requests[-1][1]["mcp-session-id"], "session-1")
        self.assertEqual(endpoint.requests[-1][1]["mcp-protocol-version"], "2025-03-26")

    def test_server_sent_event_parser(self):
        parsed = _sse_messages('event: message\ndata: {"jsonrpc":"2.0","id":3,"result":{"ok":true}}\n\n')
        self.assertEqual(parsed[0]["result"], {"ok": True})

    def test_tool_error_handles_malformed_non_list_content(self):
        client = StreamableHTTPMCP("http://127.0.0.1:42019")
        client.initialized = True
        client._rpc = lambda method, params=None, timeout=None: {
            "isError": True, "content": {"unexpected": "shape"},
        }
        with self.assertRaisesRegex(MCPToolError, "unspecified tool error"):
            client.call_tool("wangp_generate", {})

    def test_success_records_actual_output_settings_and_repeats_serially(self):
        transport = FakeTransport()
        clock = FakeClock()
        result = BenchmarkRunner(
            WanGPClient(transport), make_manifest(repeats=2),
            timeout_seconds=5, poll_interval=0.5, clock=clock, sleep=clock.sleep,
        ).run("digest")
        self.assertEqual(result["status"], "completed")
        self.assertEqual([r["job_id"] for r in result["runs"]], ["owned-1", "owned-2"])
        self.assertEqual(result["runs"][0]["requested_settings"]["num_inference_steps"], 20)
        self.assertEqual(result["runs"][0]["generated_files"], ["output.mp4"])
        output = result["runs"][0]["actual_output_settings"][0]
        self.assertEqual(output["response"]["settings"]["resolution"], "864x480")
        self.assertIn("resolution", output["settings_comparison"]["matched_fields"])
        self.assertIn("video_length", output["settings_comparison"]["matched_fields"])
        self.assertIn("config", output["settings_comparison"]["matched_fields"])
        self.assertEqual(
            result["model_discovery"]["baseline_settings_validation"]["status"], "valid"
        )
        self.assertIn(
            "override_profile",
            result["model_discovery"]["baseline_settings_validation"][
                "requested_fields_absent_from_model_defaults"
            ],
        )
        self.assertIn({"kind": "progress", "step": 1}, result["runs"][0]["progress_events"])
        submits = [call for call in transport.calls if call[0] == "wangp_generate"]
        self.assertEqual(len(submits), 2)
        self.assertTrue(all(call[1]["wait"] is False for call in submits))
        self.assertEqual(transport.cancelled, [])

    def test_busy_preflight_refuses_without_submitting_or_cancelling(self):
        transport = FakeTransport(queues=[{
            "total_count": 1, "queued_count": 0, "running_count": 1,
        }])
        result = BenchmarkRunner(WanGPClient(transport), make_manifest(), timeout_seconds=5).run()
        self.assertEqual(result["status"], "refused_busy_queue")
        self.assertEqual(result["runs"], [])
        self.assertFalse(any(call[0] == "wangp_generate" for call in transport.calls))
        queue_call = next(call for call in transport.calls if call[0] == "wangp_session")
        self.assertEqual(queue_call[1]["arguments"], {})
        self.assertEqual(transport.cancelled, [])

    def test_new_busy_queue_between_repeats_prevents_next_submission(self):
        transport = FakeTransport(queues=[
            {"total_count": 0}, {"total_count": 0},
            {"total_count": 1, "queued_count": 1, "running_count": 0},
        ])
        result = BenchmarkRunner(
            WanGPClient(transport), make_manifest(repeats=2), timeout_seconds=5
        ).run()
        self.assertEqual(result["status"], "refused_busy_queue")
        self.assertEqual(len(result["runs"]), 1)
        self.assertEqual(len([c for c in transport.calls if c[0] == "wangp_generate"]), 1)
        self.assertEqual(transport.cancelled, [])

    def test_timeout_cancels_only_owned_job_and_confirms_boundedly(self):
        transport = FakeTransport(never_finish=True)
        clock = FakeClock()
        result = BenchmarkRunner(
            WanGPClient(transport), make_manifest(), timeout_seconds=1,
            poll_interval=0.5, cancel_grace_seconds=1,
            clock=clock, sleep=clock.sleep,
        ).run()
        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(transport.cancelled, ["owned-1"])
        self.assertTrue(result["runs"][0]["timeout_cancel"]["requested"])
        self.assertTrue(result["runs"][0]["timeout_cancel"]["confirmed"])

    def test_progress_file_has_owned_id_before_first_poll(self):
        with tempfile.TemporaryDirectory() as directory:
            progress_path = Path(directory) / "live-progress.json"
            test_case = self

            class InspectingTransport(FakeTransport):
                def call_tool(self, name, arguments, timeout=None):
                    if name == "wangp_session" and arguments.get("action") == "get_job":
                        saved = json.loads(progress_path.read_text(encoding="utf-8"))
                        test_case.assertEqual(saved["job_id"], "owned-1")
                        test_case.assertEqual(saved["case"], "480p")
                    return super().call_tool(name, arguments, timeout)

            transport = InspectingTransport()
            result = BenchmarkRunner(
                WanGPClient(transport),
                make_manifest(),
                timeout_seconds=5,
                progress_callback=lambda snapshot: write_result(progress_path, snapshot),
            ).run()

            self.assertEqual(result["status"], "completed")
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["job_id"], "owned-1")
            self.assertEqual(saved["repeat"], 1)

    def test_progress_snapshots_follow_polls_and_include_final_outcome(self):
        transport = FakeTransport(job_snapshots=[
            {
                "job_id": "owned-1",
                "done": False,
                "events": [{
                    "kind": "preview",
                    "data": {
                        "current_step": 6, "total_steps": 20,
                        "phase": "inference", "status": "Denoising", "progress": 39,
                        "image": "must not be copied into the live file",
                        "text": "nested free-form text stays out",
                    },
                }],
            },
            {
                "job_id": "owned-1",
                "done": True,
                "created_at": 100.0,
                "updated_at": 102.0,
                "events": [{"kind": "completed", "text": "event history stays out"}],
                "result": {
                    "success": True, "generated_files": ["output.mp4"], "errors": [],
                    "gallery_items": [{"media_id": "media-1", "filename": "output.mp4"}],
                },
            },
        ])
        clock = FakeClock()
        snapshots = []
        result = BenchmarkRunner(
            WanGPClient(transport),
            make_manifest(),
            timeout_seconds=5,
            poll_interval=0.5,
            clock=clock,
            sleep=clock.sleep,
            progress_callback=snapshots.append,
        ).run()

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(snapshots), 4)
        self.assertEqual(snapshots[0]["job_id"], "owned-1")
        self.assertEqual(snapshots[0]["status"], "running")
        self.assertEqual(snapshots[1]["progress"], {
            "kind": "preview", "current_step": 6, "total_steps": 20,
            "phase": "inference", "status": "Denoising", "progress": 39,
        })
        self.assertEqual(snapshots[-1]["status"], "completed")
        self.assertTrue(snapshots[-1]["done"])
        serialized = json.dumps(snapshots)
        self.assertNotIn("prompt", serialized)
        self.assertNotIn("image", serialized)
        self.assertNotIn("nested free-form text stays out", serialized)
        self.assertNotIn("event history stays out", serialized)
        self.assertNotIn("requested_settings", serialized)

    def test_progress_callback_failure_does_not_retry_or_cancel_job(self):
        transport = FakeTransport()
        clock = FakeClock()
        calls = []

        def broken_progress_writer(snapshot):
            calls.append(snapshot)
            raise OSError("simulated disk full")

        runner = BenchmarkRunner(
            WanGPClient(transport),
            make_manifest(),
            timeout_seconds=5,
            clock=clock,
            sleep=clock.sleep,
            progress_callback=broken_progress_writer,
        )
        result = runner.run()

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            len([call for call in transport.calls if call[0] == "wangp_generate"]), 1
        )
        self.assertEqual(transport.cancelled, [])
        self.assertIn("simulated disk full", runner.progress_callback_warning)

    def test_timeout_progress_snapshot_records_confirmed_cancellation(self):
        transport = FakeTransport(never_finish=True)
        clock = FakeClock()
        snapshots = []
        result = BenchmarkRunner(
            WanGPClient(transport),
            make_manifest(),
            timeout_seconds=1,
            poll_interval=0.5,
            cancel_grace_seconds=1,
            clock=clock,
            sleep=clock.sleep,
            progress_callback=snapshots.append,
        ).run()

        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(transport.cancelled, ["owned-1"])
        self.assertTrue(snapshots[-1]["done"])
        self.assertEqual(snapshots[-1]["status"], "timed_out")
        self.assertEqual(snapshots[-1]["cancel_state"], {
            "status": "confirmed", "requested": True, "confirmed": True,
        })

    def test_cli_emits_one_warning_when_progress_file_write_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            progress_path = Path(directory) / "live-progress.json"
            real_write_result = write_result

            def fail_only_progress(path, result):
                if Path(path) == progress_path:
                    raise BenchmarkError("simulated progress disk failure")
                return real_write_result(path, result)

            stderr = io.StringIO()
            with patch(
                "app.scripts.benchmark_wangp.load_manifest",
                return_value=(make_manifest(), "digest"),
            ), patch(
                "app.scripts.benchmark_wangp.StreamableHTTPMCP",
                return_value=FakeTransport(),
            ), patch(
                "app.scripts.benchmark_wangp.write_result",
                side_effect=fail_only_progress,
            ), redirect_stderr(stderr):
                status = benchmark_main([
                    "--base-url", "http://127.0.0.1:42019",
                    "--manifest", "unused.json",
                    "--output", str(report_path),
                    "--progress-file", str(progress_path),
                    "--timeout", "5",
                ])

            self.assertEqual(status, 0)
            warning = stderr.getvalue()
            self.assertEqual(warning.count("Warning:"), 1)
            self.assertIn("monitoring continued", warning)
            self.assertTrue(report_path.exists())

    def test_async_disabled_is_rejected_from_tool_schema(self):
        transport = FakeTransport()
        transport.tools[0]["inputSchema"]["properties"]["wait"] = {
            "type": "boolean", "const": True,
        }
        with self.assertRaisesRegex(BenchmarkError, "mcp-async"):
            WanGPClient(transport).initialize()
        self.assertFalse(any(call[0] == "wangp_generate" for call in transport.calls))


if __name__ == "__main__":
    unittest.main()
