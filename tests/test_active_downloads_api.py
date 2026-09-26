"""Long-poll behaviour of GET /api/v1/downloads/active.

Uses httpx.AsyncClient over ASGITransport instead of the sync TestClient,
because a held request has to run concurrently with the state change that
releases it.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
import sys
import time
import unittest

from fastapi import FastAPI
import httpx


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
sys.path.insert(0, str(_APP))

from services.active_downloads_api import create_router  # noqa: E402


def _download(downloaded_bytes: int) -> dict:
    return {
        "file_id": "a",
        "filename": "a.safetensors",
        "status": "downloading",
        "downloaded_bytes": downloaded_bytes,
        "total_bytes": 5000,
        "seconds_since_progress": 0.0,
    }


class ActiveDownloadsLongPollTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.downloads: list[dict] = []
        self.wait_timeout = 5.0

    def _client(self) -> httpx.AsyncClient:
        app = FastAPI()
        app.include_router(create_router(
            read_downloads=lambda: list(self.downloads),
            check_interval=0.01,
            wait_timeout=self.wait_timeout,
        ))
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def _get(self, client: httpx.AsyncClient, since: str | None = None) -> httpx.Response:
        params = {} if since is None else {"since": since}
        return await asyncio.wait_for(client.get("/api/v1/downloads/active", params=params), 10)

    async def test_without_since_answers_at_once_with_a_version(self):
        self.downloads = [_download(1000)]
        async with self._client() as client:
            response = await self._get(client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        body = response.json()
        self.assertEqual(body["downloads"], [_download(1000)])
        self.assertIsInstance(body["version"], str)

    async def test_version_follows_the_download_state(self):
        async with self._client() as client:
            idle = (await self._get(client)).json()["version"]
            self.assertEqual((await self._get(client)).json()["version"], idle)
            self.downloads = [_download(1000)]
            self.assertNotEqual((await self._get(client)).json()["version"], idle)

    async def test_stale_since_answers_at_once(self):
        async with self._client() as client:
            started = time.monotonic()
            response = await self._get(client, since="stale")
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(response.json()["downloads"], [])

    async def test_current_since_is_held_until_the_downloads_change(self):
        async with self._client() as client:
            version = (await self._get(client)).json()["version"]
            held = asyncio.ensure_future(self._get(client, since=version))
            await asyncio.sleep(0.1)
            self.assertFalse(held.done(), "An unchanged state keeps the request open")
            self.downloads = [_download(1000)]
            response = await asyncio.wait_for(held, 1)
        body = response.json()
        self.assertEqual(body["downloads"], [_download(1000)])
        self.assertNotEqual(body["version"], version)

    async def test_hold_ends_after_the_wait_timeout_with_the_same_version(self):
        self.wait_timeout = 0.1
        async with self._client() as client:
            version = (await self._get(client)).json()["version"]
            response = await self._get(client, since=version)
        self.assertEqual(response.json()["version"], version)


class LaunchWiringTests(unittest.TestCase):
    def test_launch_serves_the_endpoint_through_the_router_at_the_old_position(self):
        source = (_APP / "launch.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        handlers = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_active_downloads"
        ]
        self.assertEqual(handlers, [], "launch.py no longer defines the handler itself")
        include = source.index("api.include_router(_active_downloads_router())")
        # Registered where the old handler was, so route order is unchanged.
        self.assertLess(include, source.index("def get_system_detect("))
        self.assertGreater(include, source.index("def scan_model_folders("))


if __name__ == "__main__":
    unittest.main()
