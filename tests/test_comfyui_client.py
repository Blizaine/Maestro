"""CPU-only tests for the ComfyUI HTTP transport."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from unittest import mock

from app.services import comfyui_client
from app.services.comfyui_client import (
    ComfyClient,
    ComfyTransportError,
    ComfyValidationError,
)


class _State:
    requests: list[dict] = []
    post_prompt_mode = "ok"
    maintenance_mode = "json"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def _record(self, body: bytes = b"") -> None:
        _State.requests.append({
            "method": self.command,
            "path": self.path,
            "headers": dict(self.headers.items()),
            "body": body,
        })

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._record()
        if self.path == "/object_info":
            self._send_json({"SaveImage": {}})
        elif self.path == "/models/checkpoints":
            self._send_json(["one.safetensors"])
        elif self.path == "/system_stats":
            self._send_json({"devices": []})
        elif self.path == "/queue":
            self._send_json({"queue_running": [], "queue_pending": []})
        elif self.path.startswith("/history"):
            self._send_json({"prompt-id": {"status": {"completed": True}}})
        elif self.path.startswith("/view?"):
            body = b"image-bytes"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/bad":
            self._send_json({"detail": "sensitive server response"}, status=400)
        elif self.path == "/large":
            body = b"0123456789"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json({}, status=404)

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        self._record(body)
        if self.path in {"/queue", "/free"} and _State.maintenance_mode == "empty":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/prompt" and _State.post_prompt_mode == "validation":
            self._send_json({"error": "private token text should never escape"}, status=400)
        elif self.path == "/prompt" and _State.post_prompt_mode == "server_error":
            self._send_json({"error": "internal server detail"}, status=500)
        elif self.path == "/prompt" and _State.post_prompt_mode == "drop":
            self.close_connection = True
            self.connection.shutdown(2)
        elif self.path == "/prompt":
            self._send_json({"prompt_id": "p-123", "number": 1})
        elif self.path == "/upload/image":
            self._send_json({"name": "input.png", "subfolder": "", "type": "input"})
        elif self.path == "/queue":
            self._send_json({"deleted": ["p-123"]})
        elif self.path == "/free":
            self._send_json({"unloaded": True, "freed": True})
        else:
            self._send_json({}, status=404)


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # A deliberate socket close is used to cover ambiguous submission
        # behavior; the resulting server-side broken pipe is expected.
        return


class TestComfyClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _State.requests = []
        _State.post_prompt_mode = "ok"
        cls.server = _QuietThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        _State.requests.clear()
        _State.post_prompt_mode = "ok"
        _State.maintenance_mode = "json"

    def test_endpoint_rejects_unsafe_or_ambiguous_urls(self):
        for endpoint in (
            "ftp://127.0.0.1:8188",
            "http://user:password@127.0.0.1:8188",
            "http://127.0.0.1:8188/?token=secret",
            "http://127.0.0.1:8188/?",
            "https://example.com:8188/#fragment",
            "https://example.com:8188/#",
            "http://example.com:8188",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                ComfyClient(endpoint)
        with ComfyClient("https://example.com:8188"):
            pass
        with ComfyClient("http://localhost:8188"):
            pass
        with ComfyClient("http://192.168.1.20:8188"):
            pass

    def test_read_routes_and_authorization_header(self):
        with ComfyClient(self.endpoint, token="secret-token") as client:
            self.assertEqual(client.object_info(), {"SaveImage": {}})
            self.assertEqual(client.models("checkpoints"), ["one.safetensors"])
            self.assertEqual(client.system_stats(), {"devices": []})
            self.assertEqual(client.queue()["queue_pending"], [])
            self.assertIn("prompt-id", client.history("prompt-id"))
            self.assertEqual(client.history(), {"prompt-id": {"status": {"completed": True}}})
        self.assertTrue(_State.requests)
        self.assertTrue(all(item["headers"].get("Authorization") == "Bearer secret-token" for item in _State.requests))
        self.assertEqual(_State.requests[1]["path"], "/models/checkpoints")

    def test_submit_upload_delete_and_output_shapes(self):
        with ComfyClient(self.endpoint) as client:
            result = client.submit({"1": {"class_type": "SaveImage", "inputs": {}}}, client_id="client-1", submission_id="maestro-1")
            self.assertEqual(result["prompt_id"], "p-123")
            uploaded = client.upload_image(b"png-content", "source.png")
            self.assertEqual(uploaded["name"], "input.png")
            self.assertEqual(client.delete_queued("p-123"), {"deleted": ["p-123"]})
            self.assertEqual(client.output("final image.png", "nested folder", "output"), b"image-bytes")
        prompt_request = next(item for item in _State.requests if item["path"] == "/prompt")
        prompt = json.loads(prompt_request["body"])
        self.assertEqual(prompt["client_id"], "client-1")
        self.assertEqual(prompt["extra_data"], {"maestro_submission_id": "maestro-1", "client_id": "client-1"})
        upload_request = next(item for item in _State.requests if item["path"] == "/upload/image")
        self.assertIn(b"source.png", upload_request["body"])
        self.assertIn(b"png-content", upload_request["body"])
        queue_request = next(item for item in _State.requests if item["method"] == "POST" and item["path"] == "/queue")
        self.assertEqual(json.loads(queue_request["body"]), {"delete": ["p-123"]})
        view_request = next(item for item in _State.requests if item["path"].startswith("/view?"))
        query = parse_qs(urlsplit(view_request["path"]).query)
        self.assertEqual(query, {"filename": ["final image.png"], "subfolder": ["nested folder"], "type": ["output"]})

    def test_release_memory_posts_explicit_unload_and_free_flags(self):
        with ComfyClient(self.endpoint) as client:
            self.assertEqual(client.release_memory(), {"unloaded": True, "freed": True})
        request = next(item for item in _State.requests if item["path"] == "/free")
        self.assertEqual(request["method"], "POST")
        self.assertEqual(json.loads(request["body"]), {"unload_models": True, "free_memory": True})

    def test_core_maintenance_empty_acknowledgements_do_not_require_json(self):
        _State.maintenance_mode = "empty"
        with ComfyClient(self.endpoint) as client:
            self.assertEqual(client.release_memory(), {})
            self.assertEqual(client.delete_queued("owned-prompt"), {})
        self.assertEqual([r["path"] for r in _State.requests], ["/free", "/queue"])
        self.assertEqual(json.loads(_State.requests[1]["body"]), {"delete": ["owned-prompt"]})

    def test_empty_acknowledgements_are_limited_to_maintenance(self):
        import httpx
        empty = httpx.Response(200, content=b"")
        malformed = httpx.Response(200, content=b"not-json")
        with ComfyClient(self.endpoint) as client:
            for operation in ("submit", "upload_image"):
                with self.subTest(operation=operation), self.assertRaises(comfyui_client.ComfyResponseError) as raised:
                    client._mutation_json(empty, operation=operation)
                self.assertTrue(raised.exception.ambiguous)
            with self.assertRaises(comfyui_client.ComfyResponseError) as raised:
                client._mutation_json(malformed, operation="release_memory", allow_empty=True)
            self.assertTrue(raised.exception.ambiguous)

    def test_known_validation_response_is_safe_and_not_ambiguous(self):
        _State.post_prompt_mode = "validation"
        with ComfyClient(self.endpoint, token="secret-token") as client:
            with self.assertRaises(ComfyValidationError) as raised:
                client.submit({}, client_id="c", submission_id="s")
        self.assertEqual(raised.exception.status_code, 400)
        self.assertFalse(raised.exception.ambiguous)
        self.assertNotIn("secret-token", str(raised.exception))
        self.assertNotIn("private token text", str(raised.exception))

    def test_server_error_after_submit_is_marked_ambiguous(self):
        _State.post_prompt_mode = "server_error"
        with ComfyClient(self.endpoint) as client:
            with self.assertRaises(comfyui_client.ComfyServerError) as raised:
                client.submit({}, client_id="c", submission_id="stable-id")
        self.assertTrue(raised.exception.ambiguous)
        self.assertIn("reconcile before retrying", str(raised.exception))

    def test_connection_drop_during_submit_is_ambiguous_and_not_retried(self):
        _State.post_prompt_mode = "drop"
        with ComfyClient(self.endpoint, timeout=1) as client:
            with self.assertRaises(ComfyTransportError) as raised:
                client.submit({}, client_id="c", submission_id="stable-id")
        self.assertTrue(raised.exception.ambiguous)
        self.assertEqual(sum(item["path"] == "/prompt" for item in _State.requests), 1)
        self.assertIn("reconcile by submission ID", str(raised.exception))

    def test_connect_failure_is_known_not_ambiguous(self):
        with ComfyClient("http://127.0.0.1:1", timeout=0.2) as client:
            with self.assertRaises(ComfyTransportError) as raised:
                client.submit({}, client_id="c", submission_id="s")
        self.assertFalse(raised.exception.ambiguous)

    def test_http_errors_and_output_limit(self):
        with ComfyClient(self.endpoint) as client:
            with self.assertRaises(ComfyValidationError):
                client._get_json("/bad", operation="test")
            with mock.patch.object(comfyui_client, "MAX_OUTPUT_BYTES", 4):
                with self.assertRaisesRegex(comfyui_client.ComfyResponseError, "size limit"):
                    client.output("too-large", type="output")

    def test_inputs_are_validated_before_network_access(self):
        with ComfyClient(self.endpoint) as client:
            with self.assertRaises(ValueError):
                client.models("../checkpoints")
            with self.assertRaises(ValueError):
                client.upload_image(b"x", "../x.png")
            with self.assertRaises(ValueError):
                client.delete_queued("")
        self.assertEqual(_State.requests, [])


if __name__ == "__main__":
    unittest.main()
