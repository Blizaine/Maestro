"""Exercise the real management routes without importing the GPU application."""

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
from services import llm_service, lmstudio_management as management


class HttpError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


class LmStudioRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        source = ast.parse((ROOT / "app" / "launch.py").read_text(encoding="utf-8-sig"))
        handlers = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in {"llm_remote_unload", "llm_remote_loaded_models"}]
        for handler in handlers:
            handler.decorator_list = []
        self.lock = threading.Lock()
        self.services = {
            "llm_provider": "remote", "llm_remote_url": "http://localhost:1234/v1",
            "llm_remote_api_key": "private-remote-key", "openai_api_key": "never-send",
            "llm_model_id": "writer",
        }
        namespace = {
            "Request": object, "asyncio": asyncio, "HTTPException": HttpError,
            "_gen_lock": self.lock, "wgp": SimpleNamespace(server_config={"services": self.services}),
        }
        exec(compile(ast.Module(body=handlers, type_ignores=[]), "lmstudio_routes", "exec"), namespace)
        self.unload = namespace["llm_remote_unload"]
        self.list_models = namespace["llm_remote_loaded_models"]

    def request(self, body):
        return SimpleNamespace(json=mock.AsyncMock(return_value=body))

    async def test_uses_server_configuration_and_remote_credentials(self):
        with mock.patch.object(llm_service, "unload_remote_instance", return_value={"status": "unloaded"}) as unload:
            result = await self.unload(self.request({"instance_id": "selected", "server_url": "http://localhost:1234"}))
        self.assertEqual(result["status"], "unloaded")
        unload.assert_called_once_with("http://localhost:1234/v1", "private-remote-key",
                                       instance_id="selected", model_id="writer")
        with mock.patch.object(management, "loaded_models", return_value={"instances": []}) as loaded:
            self.list_models()
        loaded.assert_called_once_with("http://localhost:1234/v1", "private-remote-key")
        self.assertFalse(self.lock.locked())

    async def test_changed_server_or_provider_never_unloads(self):
        for provider, root, status in (("remote", "http://different:1234", 409),
                                      ("openai", "http://localhost:1234", 400)):
            with self.subTest(provider=provider):
                self.services["llm_provider"] = provider
                with mock.patch.object(llm_service, "unload_remote_instance") as unload:
                    with self.assertRaises(HttpError) as caught:
                        await self.unload(self.request({"instance_id": "selected", "server_url": root}))
                    self.assertEqual(caught.exception.status_code, status)
                    unload.assert_not_called()
                self.assertFalse(self.lock.locked())

    async def test_invalid_json_is_a_client_error(self):
        for request in (self.request([]), SimpleNamespace(json=mock.AsyncMock(side_effect=ValueError("invalid JSON")))):
            with mock.patch.object(llm_service, "unload_remote_instance") as unload:
                with self.assertRaises(HttpError) as caught:
                    await self.unload(request)
                self.assertEqual(caught.exception.status_code, 400)
                unload.assert_not_called()
        self.assertFalse(self.lock.locked())

    async def test_busy_generation_does_not_unload(self):
        self.lock.acquire()
        try:
            with mock.patch.object(llm_service, "unload_remote_instance") as unload:
                with self.assertRaises(HttpError) as caught:
                    await self.unload(self.request({"instance_id": "target"}))
                self.assertEqual(caught.exception.status_code, 409)
                unload.assert_not_called()
                self.assertTrue(self.lock.locked())
        finally:
            self.lock.release()

    async def test_remote_errors_release_resources_and_keep_status_codes(self):
        for error, status in ((management.ModelSelectionError("choose a model"), 400),
                              (management.ModelBusyError("writer busy"), 409),
                              (management.ModelManagementError("server rejected unload"), 502)):
            with self.subTest(status=status), mock.patch.object(llm_service, "unload_remote_instance", side_effect=error):
                with self.assertRaises(HttpError) as caught:
                    await self.unload(self.request({"instance_id": "target"}))
                self.assertEqual(caught.exception.status_code, status)
                self.assertFalse(self.lock.locked())

    async def test_disconnect_does_not_release_gpu_while_unload_is_pending(self):
        started, finish, exited = threading.Event(), threading.Event(), threading.Event()
        def slow_unload(*args, **kwargs):
            started.set()
            try:
                finish.wait(timeout=5)
                return {"status": "unloaded"}
            finally:
                exited.set()
        with mock.patch.object(llm_service, "unload_remote_instance", side_effect=slow_unload):
            task = asyncio.create_task(self.unload(self.request({"instance_id": "target"})))
            try:
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(started.is_set(), "external requests must not block the event loop")
                self.assertTrue(self.lock.locked())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(self.lock.locked(), "disconnect must not allow generation to race external cleanup")
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


if __name__ == "__main__":
    unittest.main()
