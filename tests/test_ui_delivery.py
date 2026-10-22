"""Tests for the narrowly scoped startup-payload gzip middleware."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from starlette.applications import Starlette
from starlette.responses import FileResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.ui_delivery import UIStartupGZipMiddleware  # noqa: E402


class UIStartupGZipMiddlewareTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.file_path = Path(temp.name) / "fixture.bin"
        self.file_body = bytes(range(256)) * 64
        self.file_path.write_bytes(self.file_body)

        rows = [
            {
                "id": f"model-{index}",
                "description": f"Model {index}: " + (f"catalogue metadata {index} " * 360),
            }
            for index in range(80)
        ]
        self.catalogue_body = json.dumps(
            {"models": rows}, separators=(",", ":")
        ).encode("utf-8")
        self.options_body = json.dumps(
            {"options": [f"option-{index}" for index in range(5000)]},
            separators=(",", ":"),
        ).encode("utf-8")
        self.asset_body = (
            b"export const startupCatalogue = "
            + (b"{ id: 'fixture', enabled: true };\n" * 1200)
        )
        self.css_body = b".startup-panel { color: #234; }\n" * 1200
        self.other_body = b"unrelated JSON response " * 2400
        self.sse_body = b"event: ready\ndata: ok\n\nevent: done\ndata: yes\n\n"

        routes = [
            Route("/api/v1/models", self._models, methods=["GET", "POST"]),
            Route("/api/v1/model-options/{model_id}", self._model_options),
            Route("/assets/{asset_path:path}", self._asset),
            Route("/api/v1/outputs/{name:path}", self._file_response),
            Route("/api/v1/files/{name:path}", self._file_response),
            Route("/api/v1/thumbnail/{name:path}", self._file_response),
            Route("/api/v1/downloads/{name:path}", self._file_response),
            Route("/api/v1/llm/models", self._other),
            Route("/api/v1/other", self._other),
            Route("/api/v1/events", self._events),
        ]
        app = Starlette(routes=routes)
        self.client = TestClient(UIStartupGZipMiddleware(app))
        self.addCleanup(self.client.close)

    async def _models(self, request):
        return Response(self.catalogue_body, media_type="application/json")

    async def _model_options(self, request):
        return Response(self.options_body, media_type="application/json")

    async def _asset(self, request):
        path = request.path_params["asset_path"]
        if path.endswith(".css"):
            return Response(self.css_body, media_type="text/css")
        if path.endswith(".js"):
            return Response(self.asset_body, media_type="text/javascript")
        return Response(self.other_body, media_type="application/octet-stream")

    async def _file_response(self, request):
        path = request.url.path
        media_type = "application/octet-stream"
        if path.endswith(".mp4"):
            media_type = "video/mp4"
        elif path.endswith(".flac"):
            media_type = "audio/flac"
        elif path.endswith(".png"):
            media_type = "image/png"
        return FileResponse(self.file_path, media_type=media_type)

    async def _other(self, request):
        return Response(self.other_body, media_type="application/json")

    async def _events(self, request):
        async def events():
            yield b"event: ready\ndata: ok\n\n"
            yield b"event: done\ndata: yes\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    def assert_gzip_round_trip(self, response, expected_body: bytes):
        self.assertEqual(response.headers.get("content-encoding"), "gzip")
        self.assertEqual(response.content, expected_body)
        self.assertLess(int(response.headers["content-length"]), len(expected_body))
        vary_values = {
            value.strip().lower()
            for value in response.headers.get("vary", "").split(",")
            if value.strip()
        }
        self.assertIn("accept-encoding", vary_values)

    def test_large_models_and_model_options_use_starlette_gzip(self):
        self.assertGreater(len(self.catalogue_body), 300_000)
        models = self.client.get(
            "/api/v1/models", headers={"accept-encoding": "gzip"}
        )
        self.assert_gzip_round_trip(models, self.catalogue_body)

        options = self.client.get(
            "/api/v1/model-options/h3ref",
            headers={"accept-encoding": "gzip"},
        )
        self.assert_gzip_round_trip(options, self.options_body)

    def test_hashed_javascript_and_css_assets_use_gzip(self):
        for path, expected_body in (
            ("/assets/app-01234567.js", self.asset_body),
            ("/assets/theme-Az09_Bcd.css", self.css_body),
        ):
            with self.subTest(path=path):
                response = self.client.get(
                    path, headers={"accept-encoding": "gzip"}
                )
                self.assert_gzip_round_trip(response, expected_body)

    def test_identity_client_gets_uncompressed_catalogue_and_negotiation_vary(self):
        response = self.client.get(
            "/api/v1/models", headers={"accept-encoding": "identity"}
        )

        self.assertNotIn("content-encoding", response.headers)
        self.assertEqual(response.content, self.catalogue_body)
        self.assertEqual(response.headers["content-length"], str(len(self.catalogue_body)))
        self.assertIn("accept-encoding", response.headers.get("vary", "").lower())

    def test_media_files_thumbnails_and_downloads_keep_exact_bytes(self):
        paths = (
            "/api/v1/outputs/clip.mp4",
            "/api/v1/outputs/audio.flac",
            "/api/v1/files/checkpoint.bin",
            "/api/v1/thumbnail/preview.png",
            "/api/v1/downloads/archive.bin",
        )
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(
                    path, headers={"accept-encoding": "gzip"}
                )
                self.assertNotIn("content-encoding", response.headers)
                self.assertEqual(response.content, self.file_body)
                self.assertEqual(response.headers["content-length"], str(len(self.file_body)))
                self.assertNotIn("accept-encoding", response.headers.get("vary", "").lower())

    def test_other_endpoints_and_unhashed_or_nested_assets_are_untouched(self):
        paths = (
            "/api/v1/llm/models",
            "/api/v1/other",
            "/assets/runtime.js",
            "/assets/chunks/app-01234567.js",
            "/assets/app-01234567.js.map",
        )
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(
                    path, headers={"accept-encoding": "gzip"}
                )
                self.assertNotIn("content-encoding", response.headers)
                self.assertNotIn("accept-encoding", response.headers.get("vary", "").lower())

    def test_sse_is_not_buffered_or_compressed(self):
        response = self.client.get(
            "/api/v1/events", headers={"accept-encoding": "gzip"}
        )

        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertNotIn("content-encoding", response.headers)
        self.assertNotIn("accept-encoding", response.headers.get("vary", "").lower())
        self.assertEqual(response.content, self.sse_body)

    def test_head_is_allowed_and_mutating_methods_are_not_compressed(self):
        head = self.client.head(
            "/api/v1/models", headers={"accept-encoding": "gzip"}
        )
        self.assertEqual(head.headers.get("content-encoding"), "gzip")

        post = self.client.post(
            "/api/v1/models", headers={"accept-encoding": "gzip"}
        )
        self.assertNotIn("content-encoding", post.headers)
        self.assertEqual(post.content, self.catalogue_body)


if __name__ == "__main__":
    unittest.main()
