"""Exercise the actual LLM load route without importing the GPU application."""

import ast
import asyncio
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services import llm_service


class HttpError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


class LlmLoadRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        source = ast.parse((ROOT / "app" / "launch.py").read_text(encoding="utf-8-sig"))
        handler = next(node for node in source.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "llm_load")
        handler.decorator_list = []
        self.lock = threading.Lock()
        namespace = {
            "Request": object, "asyncio": asyncio, "HTTPException": HttpError,
            "_guard_interactive_llm_against_generation": lambda: None,
            "_gen_lock": self.lock, "_DEFAULT_LLM_REPO": "test-writer",
            "_llm_default_device": lambda: "cuda",
            "wgp": SimpleNamespace(server_config={}), "traceback": mock.Mock(),
        }
        exec(compile(ast.Module(body=[handler], type_ignores=[]), "llm_load", "exec"), namespace)
        self.handler = namespace["llm_load"]
        self.request = SimpleNamespace(headers={})

    async def test_slow_load_keeps_event_loop_free_and_owns_gpu_after_disconnect(self):
        started, finish, exited = threading.Event(), threading.Event(), threading.Event()
        def slow_load(**kwargs):
            started.set()
            try:
                finish.wait(timeout=5)
            finally:
                exited.set()
        with mock.patch.object(llm_service, "load_model", side_effect=slow_load), mock.patch.object(llm_service, "get_status", return_value={"loaded": True}):
            task = asyncio.create_task(self.handler(self.request))
            try:
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(started.is_set())
                self.assertFalse(exited.is_set(), "load must not block the event loop until it finishes")
                self.assertTrue(self.lock.locked())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(self.lock.locked(), "HTTP cancellation must not release the GPU while loading")
            finally:
                finish.set()
                await asyncio.to_thread(exited.wait, 2)
                for _ in range(100):
                    if not self.lock.locked():
                        break
                    await asyncio.sleep(0.01)
                if not task.done():
                    await task
        self.assertFalse(self.lock.locked())

    async def test_busy_generation_stays_409_and_does_not_load(self):
        self.lock.acquire()
        try:
            with mock.patch.object(llm_service, "load_model") as load:
                with self.assertRaises(HttpError) as caught:
                    await self.handler(self.request)
            self.assertEqual(caught.exception.status_code, 409)
            load.assert_not_called()
            self.assertTrue(self.lock.locked())
        finally:
            self.lock.release()

    async def test_failed_load_releases_gpu(self):
        with mock.patch.object(llm_service, "load_model", side_effect=RuntimeError("CUDA missing")):
            with self.assertRaises(HttpError) as caught:
                await self.handler(self.request)
        self.assertEqual(caught.exception.status_code, 500)
        self.assertFalse(self.lock.locked())


if __name__ == "__main__":
    unittest.main()
