"""Prompt bench invariants: truthful evidence, scoped tracing, safe admission/resume."""
import asyncio
from contextlib import nullcontext
from copy import deepcopy
from functools import wraps
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from promptbench.checks import evaluate, trace_calls
from promptbench.evidence import digest, source_fingerprint
from promptbench.runner import Client, load_suite, main, report, run_lock, write_json
from promptbench.server import installed_writer, local_request, register_routes, validate_params
from promptbench.trace import capture, record_metrics, record_payload, record_response, traced


def case():
    return {"id": "quiet", "title": "Quiet", "split": "development", "params": {
        "prompt": 'A person looks at the sea. No dialogue.', "model_type": "minimax_h3_fused_turbo",
        "generation_mode": "video", "resolution": "864x480", "video_length": 345},
        "checks": {"expected_windows": 1, "no_dialogue": True}, "review": ["Meaningful silent action."]}


def result(prompt=None):
    return {"status": "complete", "seconds": 1, "generation_started": False,
        "prepared": {"params": {"prompt": prompt or 'integrated_multimodal_description: A person watches the sea.\noverall_soundscape: Surf.\nnon_diegetic_music: None.'}},
        "calls": [{"index": 1, "status": "complete", "payload": {"cache_prompt": False},
                   "metrics": {"prompt_tokens": 100, "completion_tokens": 40, "reasoning_tokens": 0, "answer_tokens": 40}}]}


class TraceTests(unittest.TestCase):
    def test_nested_generate_streaming_delegation_is_one_request(self):
        @traced
        def streaming():
            record_payload({"messages": []})
            record_metrics({"completion_tokens": 2})
            return "draft"
        @traced
        def generate():
            return streaming()
        with capture(max_calls=1) as trace:
            self.assertEqual(generate(), "draft")
        self.assertEqual(len(trace["calls"]), 1)
        self.assertEqual(trace["calls"][0]["metrics"]["completion_tokens"], 2)

    def test_real_llm_json_and_stream_capture_before_cleanup(self):
        import io
        import requests
        from services import llm_service as llm
        from services.studio_enhancement import enhancement_context
        for mode in ("json", "stream", "delegate"):
            streaming = mode != "json"
            with self.subTest(mode=mode):
                response = requests.Response()
                response.status_code = 200
                response.encoding = "utf-8"
                usage = {"prompt_tokens": 10, "completion_tokens": 8}
                if streaming:
                    chunks = [
                        {"choices": [{"delta": {"reasoning_content": "local thought"}}]},
                        {"choices": [{"delta": {"content": "draft"}, "finish_reason": "stop"}], "usage": usage},
                    ]
                    response.raw = io.BytesIO(b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n")
                else:
                    response._content = json.dumps({"choices": [{"message": {"content": "draft", "reasoning_content": "local thought"},
                        "finish_reason": "stop"}], "usage": usage}).encode()
                with patch.object(llm, "is_loaded", return_value=True), patch.object(llm, "_provider", "local"), \
                        patch.object(llm, "_model_id", "bench-test"), patch.object(llm, "_reset_idle_timer"), \
                        patch.object(llm, "_count_local_tokens", return_value=None), patch.object(requests, "post", return_value=response):
                    with capture() as trace, (enhancement_context({}, lambda: False) if mode == "delegate" else nullcontext()):
                        generate = llm.generate_streaming if mode == "stream" else llm.generate
                        self.assertEqual(generate("test", enable_thinking=False), "draft")
                self.assertEqual(len(trace["calls"]), 1)
                call = trace["calls"][0]
                self.assertEqual(call["raw_response"]["reasoning_content"], "local thought")
                self.assertEqual(call["raw_response"]["content"], "draft")
                self.assertTrue(call["raw_response"]["complete"])
                self.assertEqual(call["output"], "draft")
                self.assertEqual(call["metrics"]["prompt_tokens"], 10)
                self.assertFalse(call["payload"]["cache_prompt"])

    def test_no_trace_does_not_change_output_or_payload(self):
        payload = {"temperature": 1}
        @traced
        def generate():
            self.assertIs(record_payload(payload), payload)
            return "answer"
        self.assertEqual(generate(), "answer")

    def test_payload_image_redaction_and_metrics_survive_threads(self):
        @traced
        def generate():
            data = {"messages": [{"content": "data:image/png;base64,ABC"}], "api_key": "secret", "temperature": 1}
            record_payload(data, model_id="test")
            data["temperature"] = 99
            record_metrics({"completion_tokens": 7})
            record_response("<think>local reasoning</think>draft", "", complete=False)
            return "draft"
        async def run():
            with capture() as trace:
                await asyncio.to_thread(generate)
                return trace
        trace = asyncio.run(run())
        self.assertEqual(len(trace["calls"]), 1)
        recorded = trace["calls"][0]
        self.assertEqual(recorded["payload"]["temperature"], 1)
        self.assertNotIn("api_key", recorded["payload"])
        self.assertIn("data_url_sha256", recorded["payload"]["messages"][0]["content"])
        self.assertEqual(recorded["output"], "draft")
        self.assertIn("local reasoning", recorded["raw_response"]["content"])
        self.assertFalse(recorded["raw_response"]["complete"])

    def test_call_limit_and_scope_reset(self):
        @traced
        def generate():
            return "ok"
        with capture(max_calls=1) as trace:
            generate()
            with self.assertRaises(InterruptedError):
                generate()
        self.assertEqual(generate(), "ok")
        self.assertEqual(len(trace["calls"]), 1)

    def test_failure_is_recorded(self):
        @traced
        def generate():
            raise RuntimeError("failure")
        with capture() as trace:
            with self.assertRaises(RuntimeError):
                generate()
        self.assertEqual(trace["calls"][0]["error_type"], "RuntimeError")

    def test_concurrent_contexts_are_isolated(self):
        @traced
        def generate(value):
            return value
        async def run(value):
            with capture() as trace:
                await asyncio.to_thread(generate, value)
                return trace["calls"][0]["output"]
        async def both():
            return await asyncio.gather(run("a"), run("b"))
        self.assertEqual(asyncio.run(both()), ["a", "b"])


class EvidenceTests(unittest.TestCase):
    def test_cached_projector_alias_matches_production_loader(self):
        for legacy, current, expected in (
            (b"legacy", None, "legacy.gguf"),
            (b"legacy", b"current", "legacy.gguf"),
            (b"", b"current", "current.gguf"),
            (b"", None, None),
        ):
            with self.subTest(legacy=legacy, current=current), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder) / "writer"
                directory.mkdir()
                (directory / "model.gguf").write_bytes(b"model")
                (directory / "legacy.gguf").write_bytes(legacy)
                if current is not None:
                    (directory / "current.gguf").write_bytes(current)
                llm = types.SimpleNamespace(
                    MODEL_REGISTRY={"local/writer-GGUF": {
                        "gguf_file": "model.gguf", "mmproj_file": "current.gguf",
                        "mmproj_cache_aliases": ["legacy.gguf"],
                    }},
                    get_model_dir=lambda: folder,
                )
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "Writer asset is not installed"):
                        installed_writer(llm, "local/writer-GGUF")
                else:
                    writer = installed_writer(llm, "local/writer-GGUF")
                    self.assertEqual(Path(writer["assets"][1]["path"]).name, expected)

    def test_legacy_wrapper_does_not_double_count_inference(self):
        wrapper = {"index": 1, "function": "generate", "output": "draft", "status": "complete"}
        dispatched = {"index": 2, "function": "generate_streaming", "output": "draft", "status": "complete",
                      "payload": {"messages": []}, "metrics": {"prompt_tokens": 5}}
        failed = {"index": 3, "function": "generate", "status": "failed", "error": "not loaded"}
        calls, wrappers, failures = trace_calls([wrapper, dispatched, failed])
        self.assertEqual(calls, [dispatched])
        self.assertEqual(wrappers, [1])
        self.assertEqual(failures, [failed])

    def test_rubric_and_skill_are_part_of_source_identity(self):
        root = Path(__file__).resolve().parents[1] / "app"
        hashes = source_fingerprint(root)
        self.assertIn("promptbench/rubric.md", hashes)
        self.assertIn("promptbench/operator/SKILL.md", hashes)

    def test_public_reference_templates_require_private_media(self):
        path = Path(__file__).resolve().parents[1] / "app/promptbench/cases.json"
        with self.assertRaisesRegex(ValueError, "at least one image or video"):
            load_suite(path)

    def test_reference_missing_and_injected_cache_fields_rejected(self):
        params = case()["params"]
        with self.assertRaises(ValueError):
            validate_params({**params, "image_start": "missing-image.png"})
        with self.assertRaises(ValueError):
            validate_params({**params, "h3_window_prompts": ["cached"]})
        with self.assertRaises(ValueError):
            validate_params({**params, "_enhance_on_generation": True})

    def test_reference_models_require_visual_media_but_frames_remain_text_only(self):
        frames = case()["params"]
        self.assertEqual(validate_params(frames).get("minimax_h3_references", []), [])
        manual = {**frames, "sliding_window_size": 260, "sliding_window_overlap": 18,
                  "sliding_window_memory_override": True}
        self.assertEqual(validate_params(manual), manual)
        reference = {**frames, "model_type": "minimax_h3_ref2va_fused_turbo"}
        with self.assertRaisesRegex(ValueError, "at least one image or video"):
            validate_params(reference)
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "voice.wav"
            audio.write_bytes(b"audio")
            with self.assertRaisesRegex(ValueError, "Audio references cannot be used alone"):
                validate_params({**reference, "minimax_h3_references": [
                    {"type": "audio", "path": str(audio), "role": "Voice"},
                ]})
            image = Path(folder) / "person.png"
            image.write_bytes(b"image")
            validated = validate_params({**reference, "minimax_h3_references": [
                {"type": "image", "path": str(image), "role": "Subject"},
            ]})
            self.assertEqual(validated["minimax_h3_references"][0]["type"], "image")

    def test_bare_quote_not_mistaken_for_dialogue(self):
        data = result('integrated_multimodal_description: A "heavy silence" settles.\noverall_soundscape: Surf.\nnon_diegetic_music: None.')
        checks = {c["check"]: c for c in evaluate(case(), data)["checks"]}
        self.assertEqual(checks["silent_scene"]["status"], "pass")
        self.assertEqual(checks["creative_quality"]["status"], "review")
        self.assertEqual(checks["rendered_quality"]["status"], "not_tested")

    def test_missing_exact_line_not_hidden_by_non_speech_copy(self):
        item = case()
        item["checks"] = {"exact_dialogue": [{"speaker": "Ada", "text": "Open the door."}]}
        data = result('integrated_multimodal_description: Open the door. <d>Wait.</d>\noverall_soundscape: Surf.\nnon_diegetic_music: None.')
        checks = {c["check"]: c for c in evaluate(item, data)["checks"]}
        self.assertEqual(checks["exact_dialogue_1"]["status"], "fail")
        self.assertEqual(checks["exact_dialogue_1_owner"]["status"], "review")

    def test_partial_token_telemetry_is_not_reported_as_complete(self):
        data = result()
        data["calls"].append({"index": 2, "status": "failed", "payload": {"messages": []}})
        audit = evaluate(case(), data)
        tokens = audit["token_totals"]["prompt_tokens"]
        self.assertEqual(tokens, {"known_total": 100, "covered_calls": 1, "total_calls": 2})
        self.assertTrue(audit["review_count"] > 0)

    def test_client_rejects_remote_or_embedded_credentials(self):
        for url in ["https://example.com", "http://evil.example:42015", "http://user:pass@localhost", "http://localhost/path", "http://localhost?x=1"]:
            with self.assertRaises(ValueError):
                Client(url)
        Client("http://127.0.0.1:42015")

    def test_report_escapes_model_html_and_retains_failed_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            item = case()
            write_json(root / "manifest.json", {"specification": {"cases": [item], "candidate": "baseline", "writers": ["test"], "repetitions": 2}, "status": "limited"})
            data = result("<script>bad()</script>")
            write_json(root / "attempts/one.json", {"key": "one", "case_id": "quiet", "writer": "test", "repetition": 1,
                "status": "complete", "result": data, "assessment": evaluate(item, data)})
            write_json(root / "attempts/two.json", {"key": "two", "case_id": "quiet", "writer": "test", "repetition": 2,
                "status": "failed", "error": "timeout"})
            report(root)
            document = (root / "report.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", document)
            self.assertIn("&lt;script&gt;", document)
            self.assertIn("timeout", document)
            self.assertIn("2 recorded attempts / 2 scheduled", document)


class ServerTests(unittest.TestCase):
    def test_local_gate_requires_loopback_and_header(self):
        request = types.SimpleNamespace(client=types.SimpleNamespace(host="127.0.0.1"), headers={"x-maestro-prompt-bench": "1"})
        self.assertTrue(local_request(request))
        request.client.host = "192.168.1.5"
        self.assertFalse(local_request(request))
        request.client.host = "127.0.0.1"
        request.headers.clear()
        self.assertFalse(local_request(request))

    def test_registered_route_runs_preparation_under_context_without_jobs(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from services.studio_enhancement import current_settings
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "writer"
            model.mkdir()
            (model / "model.gguf").write_bytes(b"model")
            fake = types.SimpleNamespace(MODEL_REGISTRY={"local/writer-GGUF": {"gguf_file": "model.gguf"}},
                get_model_dir=lambda: str(root), get_status=lambda: {"loaded": False}, is_loaded=lambda: False)
            settings = {"llm_model_id": "original", "enhance_llm_model_id": "original", "nsfw_mode": True}
            events = []
            def slot(function):
                @wraps(function)
                async def wrapped(request):
                    events.append("acquired")
                    try:
                        return await function(request)
                    finally:
                        events.append("released")
                return wrapped
            async def enhance(payload):
                self.assertEqual(current_settings(settings)["llm_model_id"], "local/writer-GGUF")
                self.assertFalse(current_settings(settings)["nsfw_mode"])
                return {"enhanced": "enhanced", "warnings": []}
            async def prepare(params, *, prepare_only=False):
                self.assertTrue(prepare_only)
                return {"params": params}
            api = FastAPI()
            with patch("services.llm_service", fake, create=True):
                register_routes(api, slot=slot, settings=lambda: dict(settings),
                    get_model=lambda _: {"architecture": "minimax_h3", "fps": 24, "frames_maximum": 345},
                    enhance=enhance, prepare=prepare, app_root=root)
                client = TestClient(api, client=("127.0.0.1", 1234))
                headers = {"x-maestro-prompt-bench": "1"}
                info = client.get("/api/v1/dev/prompt-bench", headers=headers).json()
                response = client.post("/api/v1/dev/prompt-bench", headers=headers, json={
                    "writer": "local/writer-GGUF", "params": case()["params"], "source_digest": info["source_digest"]})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["status"], "complete")
            self.assertFalse(response.json()["generation_started"])
            self.assertEqual(events, ["acquired", "released"])
            self.assertEqual(settings["llm_model_id"], "original")
            self.assertTrue(settings["nsfw_mode"])

    def test_reference_without_visual_media_is_rejected_before_slot(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "writer"
            model.mkdir()
            (model / "model.gguf").write_bytes(b"model")
            fake = types.SimpleNamespace(MODEL_REGISTRY={"local/writer-GGUF": {"gguf_file": "model.gguf"}},
                get_model_dir=lambda: str(root), get_status=lambda: {"loaded": False}, is_loaded=lambda: False)
            events = []
            def slot(function):
                @wraps(function)
                async def wrapped(request):
                    events.append("acquired")
                    return await function(request)
                return wrapped
            api = FastAPI()
            with patch("services.llm_service", fake, create=True):
                register_routes(api, slot=slot, settings=dict,
                    get_model=lambda _: {"architecture": "minimax_h3_ref2va", "omni_reference": True},
                    enhance=None, prepare=None, app_root=root)
                client = TestClient(api, client=("127.0.0.1", 1234))
                response = client.post("/api/v1/dev/prompt-bench",
                    headers={"x-maestro-prompt-bench": "1"}, json={
                        "writer": "local/writer-GGUF",
                        "params": {**case()["params"], "model_type": "custom-reference-alias"},
                        "source_digest": "unused",
                    })
            self.assertEqual(response.status_code, 400)
            self.assertIn("at least one image or video", response.text)
            self.assertEqual(events, [])


class ResumeTests(unittest.TestCase):
    def test_concurrent_runner_cannot_overwrite_a_run(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with run_lock(output):
                with self.assertRaises(ValueError):
                    with run_lock(output):
                        self.fail("Second runner acquired a live run directory")
            self.assertFalse((output / ".run.lock").exists())

    def _run(self, folder, responder, resume=False):
        root = Path(folder)
        suite_path = root / "input.json"
        write_json(suite_path, {"version": 1, "cases": [case()]})
        info = {"protocol": 1, "source_digest": "abc", "writers": [{"model_id": "writer"}]}
        def call(_, path, data=None, timeout=30):
            return info if path == "dev/prompt-bench" and data is None else responder()
        def snapshot(*_):
            return {"source_digest": "abc"}
        args = ["run", "--base-url", "http://localhost:42015", "--suite", str(suite_path),
                "--writer", "writer", "--output", str(root / "out"), "--case-timeout", "30", "--max-seconds", "60"]
        if resume:
            args.append("--resume")
        with patch.object(Client, "call", call), patch.object(Client, "busy", return_value=False), patch("promptbench.runner.snapshot_source", snapshot):
            return main(args)

    def test_resume_never_resubmits_completed_or_uncertain_post(self):
        for status in ("complete", "uncertain"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                count = []
                def responder():
                    count.append(1)
                    if status == "uncertain":
                        raise TimeoutError("unknown server state")
                    return result()
                self._run(folder, responder)
                self._run(folder, responder, resume=True)
                self.assertEqual(len(count), 1)

    def test_busy_post_does_not_count_as_generated_result_and_can_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            def busy():
                import io
                raise HTTPError("local", 409, "busy", {}, io.BytesIO(b"busy"))
            self._run(folder, busy)
            attempts = list((Path(folder) / "out/attempts").glob("*.json"))
            self.assertEqual(json.loads(attempts[0].read_text())["status"], "busy")
            self._run(folder, result, resume=True)
            self.assertEqual(json.loads(attempts[0].read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
