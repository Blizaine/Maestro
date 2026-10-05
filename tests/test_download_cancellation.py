"""Bounded CPU/loopback checks for scoped download cancellation."""
from __future__ import annotations

import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import safe_download
from services import download_control as dc
from shared.checkpoint_downloads import download_named_checkpoint


def checkpoint_bytes(size=2 * 1024 * 1024):
    header = json.dumps({"weight": {"dtype": "U8", "shape": [size],
                                    "data_offsets": [0, size]}}).encode()
    return len(header).to_bytes(8, "little") + header + b"x" * size


def source_for(content):
    return {"url": "https://civitai.com/api/download/models/101?fileId=11",
            "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


class FakeResponse:
    status_code = 200

    def __init__(self, content):
        self.content = content
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.content

    def close(self):
        self.closed = True


class DownloadCancellationTests(unittest.TestCase):
    def setUp(self):
        with safe_download._active_downloads_lock:
            safe_download._active_downloads.clear()

    def test_cancel_before_start_prevents_network_and_requires_worker_ack(self):
        control = dc.create_download_control()
        self.assertEqual(control.request_cancel()["status"], "cancelling")
        with patch("requests.get") as get:
            with self.assertRaises(dc.DownloadCancelled), dc.download_scope(control):
                download_named_checkpoint(source_for(checkpoint_bytes()), "unused.safetensors")
            get.assert_not_called()
        self.assertEqual(control.snapshot()["status"], "cancelling")
        self.assertEqual(control.finish("failed")["status"], "cancelled")
        self.assertEqual(control.request_cancel()["status"], "cancelled")

    def test_cancel_after_last_chunk_keeps_existing_file_and_removes_partial(self):
        content = checkpoint_bytes()
        control = dc.create_download_control()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.safetensors"
            target.write_bytes(b"existing completed checkpoint")
            response = FakeResponse(content)
            with patch("requests.get", return_value=response), dc.download_scope(control):
                with self.assertRaises(dc.DownloadCancelled):
                    download_named_checkpoint(source_for(content), target,
                        progress_hook=lambda *_: control.request_cancel())
            control.finish("failed")
            self.assertEqual(target.read_bytes(), b"existing completed checkpoint")
            self.assertEqual(list(Path(directory).iterdir()), [target])
            self.assertTrue(response.closed)
            statuses = safe_download.get_active_downloads()
            self.assertEqual([item["status"] for item in statuses], ["cancelled"])
            self.assertFalse(statuses[0]["cancellable"])
            self.assertEqual(statuses[0]["cancel_id"], control.download_id)

    def test_sealed_publication_rejects_late_cancel_and_finishes_normally(self):
        content = checkpoint_bytes()
        control = dc.create_download_control()

        def before_publish():
            dc.seal_download()
            self.assertFalse(control.snapshot()["cancellable"])
            with self.assertRaises(ValueError):
                control.request_cancel()

        with tempfile.TemporaryDirectory() as directory, \
                patch("requests.get", return_value=FakeResponse(content)), dc.download_scope(control):
            target = Path(directory) / "checkpoint.safetensors"
            download_named_checkpoint(source_for(content), target, before_publish=before_publish)
            self.assertEqual(target.read_bytes(), content)
        self.assertEqual(control.finish("completed")["status"], "completed")

    def test_real_stream_cancel_isolated_from_other_download_and_cleans_temp(self):
        content = checkpoint_bytes(4 * 1024 * 1024)
        release = {key: threading.Event() for key in ("cancel", "keep")}
        progressed = {key: threading.Event() for key in release}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                key = self.path.strip("/")
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                try:
                    self.wfile.write(content[:1024 * 1024])
                    self.wfile.flush()
                    if not release[key].wait(5):
                        return
                    self.wfile.write(content[1024 * 1024:])
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        controls = {key: dc.create_download_control() for key in release}
        results = {}
        workers = []
        import requests

        def request(*_args, **kwargs):
            control = dc.current_download_control()
            key = next(key for key, value in controls.items() if value is control)
            return requests.Session().get(f"http://127.0.0.1:{server.server_port}/{key}",
                                          stream=True, timeout=3)

        try:
            with tempfile.TemporaryDirectory() as directory, patch("requests.get", side_effect=request):
                targets = {key: Path(directory) / f"{key}.safetensors" for key in release}
                targets["cancel"].write_bytes(b"keep this existing file")

                def worker(key):
                    status = "completed"
                    try:
                        with dc.download_scope(controls[key]):
                            download_named_checkpoint(source_for(content), targets[key],
                                progress_hook=lambda *_: progressed[key].set())
                    except dc.DownloadCancelled:
                        status = "failed"
                    except Exception as exc:
                        results[key + "_error"] = repr(exc)
                        status = "failed"
                    finally:
                        results[key] = controls[key].finish(status)["status"]

                for key in release:
                    thread = threading.Thread(target=worker, args=(key,), daemon=True)
                    workers.append(thread)
                    thread.start()
                for event in progressed.values():
                    self.assertTrue(event.wait(5), "The real streamed transfer must make progress")
                tracked = safe_download.get_active_downloads()
                self.assertEqual({item["cancel_id"] for item in tracked},
                                 {control.download_id for control in controls.values()})
                self.assertTrue(all(item["cancellable"] for item in tracked))
                self.assertEqual(controls["cancel"].request_cancel()["status"], "cancelling")
                self.assertEqual(controls["keep"].snapshot()["status"], "downloading")
                release["keep"].set()
                release["cancel"].set()
                for thread in workers:
                    thread.join(5)
                    self.assertFalse(thread.is_alive(), "Cancellation must unwind the worker")
                self.assertNotIn("cancel_error", results)
                self.assertNotIn("keep_error", results)
                self.assertEqual(results["cancel"], "cancelled")
                self.assertEqual(results["keep"], "completed")
                self.assertEqual(targets["cancel"].read_bytes(), b"keep this existing file")
                self.assertEqual(targets["keep"].read_bytes(), content)
                self.assertEqual(set(Path(directory).iterdir()), set(targets.values()))
                tracked = safe_download.get_active_downloads()
                self.assertEqual([item["status"] for item in tracked], ["cancelled"])
        finally:
            for event in release.values():
                event.set()
            for thread in workers:
                thread.join(5)
            server.shutdown()
            server.server_close()
            server_thread.join(2)

    def test_transport_error_after_cancel_is_not_a_retriable_network_error(self):
        import requests
        control = dc.create_download_control()
        response = FakeResponse(b"")

        def content(**_kwargs):
            yield b"first"
            control.request_cancel()
            raise requests.ConnectionError("socket closed")

        response.iter_content = content
        with dc.download_scope(control):
            wrapped = control.wrap_response(response)
            iterator = wrapped.iter_content(chunk_size=10)
            self.assertEqual(next(iterator), b"first")
            with self.assertRaises(dc.DownloadCancelled):
                next(iterator)
        control.finish("failed")
        self.assertTrue(response.closed)

    def test_tqdm_hook_propagates_cancel_instead_of_swallowing_it(self):
        from tqdm import tqdm
        control = dc.create_download_control()
        with dc.download_scope(control):
            bar = tqdm(total=100000, unit="B", unit_scale=True, desc="same-name.bin", disable=True)
            control.request_cancel()
            with self.assertRaises(dc.DownloadCancelled):
                bar.update(1)
            bar.close()
        control.finish("failed")

    def test_urllib_cancel_preserves_completed_target_and_cleans_staging_file(self):
        import urllib.request
        content = b"u" * (128 * 1024)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/file"
        control = dc.create_download_control()

        def report(block, *_args):
            if block:
                control.request_cancel()

        try:
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "asset.bin"
                target.write_bytes(b"already installed")
                with dc.download_scope(control), self.assertRaises(dc.DownloadCancelled):
                    urllib.request.urlretrieve(url, target, report)
                control.finish("failed")
                self.assertEqual(target.read_bytes(), b"already installed")
                self.assertEqual(list(Path(directory).iterdir()), [target])
                retry = dc.create_download_control()
                with dc.download_scope(retry):
                    result, _headers = urllib.request.urlretrieve(url, target)
                retry.finish("completed")
                self.assertEqual(result, target)
                self.assertEqual(target.read_bytes(), content)
                self.assertEqual(list(Path(directory).iterdir()), [target])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_installed_hf_download_cancel_and_cache_resume(self):
        try:
            import huggingface_hub.file_download as hf
        except ImportError:
            self.skipTest("The installed Hugging Face library is unavailable")
        if not getattr(hf, "_maestro_cancellation_patched", False):
            self.fail("Cancellation must be installed for the actual HF library")
        content = b"h" * (512 * 1024)
        release = threading.Event()
        ranges = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                header = self.headers.get("Range", "bytes=0-")
                ranges.append(header)
                start = int(header.removeprefix("bytes=").split("-")[0])
                self.send_response(206)
                self.send_header("Content-Length", str(len(content) - start))
                self.send_header("Content-Range", f"bytes {start}-{len(content)-1}/{len(content)}")
                self.end_headers()
                try:
                    if start == 0 and not release.is_set():
                        self.wfile.write(content[:128 * 1024])
                        self.wfile.flush()
                        if not release.wait(5):
                            return
                        self.wfile.write(content[128 * 1024:])
                    else:
                        self.wfile.write(content[start:])
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        control = dc.create_download_control()
        result = {}
        worker = None
        try:
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(hf.constants, "DOWNLOAD_CHUNK_SIZE", 64 * 1024), \
                    patch.object(hf.constants, "HF_HUB_DOWNLOAD_TIMEOUT", 2), \
                    patch.dict(os.environ, {"TQDM_POSITION": "-1"}), \
                    patch.object(hf.constants, "HF_HUB_ENABLE_HF_TRANSFER", False):
                partial = Path(directory) / "weights.incomplete"
                target = Path(directory) / "weights.bin"
                args = (partial, target, f"http://127.0.0.1:{server.server_port}/weights",
                        None, {"X-Test": "present"}, len(content), "weights.bin", False, None, object())

                def transfer():
                    status = "completed"
                    try:
                        with dc.download_scope(control):
                            # Exercise the real installed positional signature
                            # and suppress an opaque Xet transfer locally.
                            hf._download_to_tmp_and_move(*args)
                    except dc.DownloadCancelled:
                        status = "failed"
                    except Exception as exc:
                        result["error"] = repr(exc)
                        status = "failed"
                    finally:
                        result["status"] = control.finish(status)["status"]

                worker = threading.Thread(target=transfer, daemon=True)
                worker.start()
                deadline = time.time() + 4
                while True:
                    tracked = safe_download.get_active_downloads()
                    if any(item["cancel_id"] == control.download_id
                           and item["downloaded_bytes"] >= 64 * 1024 for item in tracked):
                        break
                    self.assertLess(time.time(), deadline, result)
                    time.sleep(0.01)
                control.request_cancel()
                release.set()
                worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertNotIn("error", result)
                self.assertEqual(result["status"], "cancelled")
                self.assertFalse(target.exists())
                resumed_bytes = partial.stat().st_size
                self.assertGreater(resumed_bytes, 0)
                self.assertLess(resumed_bytes, len(content))
                self.assertEqual(len(ranges), 1, "Cancellation must not start an automatic network retry")
                retry = dc.create_download_control()
                with dc.download_scope(retry):
                    hf._download_to_tmp_and_move(*args)
                retry.finish("completed")
                self.assertEqual(target.read_bytes(), content)
                self.assertFalse(partial.exists())
                self.assertEqual(ranges[-1], f"bytes={resumed_bytes}-")
        finally:
            release.set()
            if worker is not None:
                worker.join(4)
            server.shutdown()
            server.server_close()
            server_thread.join(2)

    def test_hf_thread_pools_propagate_scope_and_http_choice_is_local(self):
        seen = []
        control = dc.create_download_control()
        original_calls = []

        def native_tmp(*_args, **kwargs):
            seen.append((module.is_xet_available(), control.snapshot()["cancellable"]))
            module.http_get("https://example.invalid/file", headers={"X-Test": "present"})
            return "downloaded"

        def pool(function, *iterables, **_kwargs):
            with ThreadPoolExecutor(max_workers=2) as executor:
                return list(executor.map(function, *iterables))

        module = SimpleNamespace(_download_to_tmp_and_move=native_tmp,
            is_xet_available=lambda: True,
            http_get=lambda *args, **kwargs: original_calls.append(kwargs),
            constants=SimpleNamespace(MAX_HTTP_DOWNLOAD_SIZE=1000))
        snapshot = SimpleNamespace(thread_map=pool)
        with patch.object(dc.importlib, "import_module", side_effect=lambda name:
                module if name.endswith("file_download") else snapshot):
            dc.install_huggingface_cancellation()
        with dc.download_scope(control):
            scopes = snapshot.thread_map(lambda _: dc.current_download_control(), [1, 2])
            self.assertEqual(scopes, [control, control])
            self.assertEqual(module._download_to_tmp_and_move(expected_size=100), "downloaded")
            self.assertEqual(seen[-1], (False, True))
            self.assertEqual(original_calls[-1]["headers"], {"X-Test": "present", "Range": "bytes=0-"})
            module._download_to_tmp_and_move(expected_size=1001)
            self.assertEqual(seen[-1], (True, False))
            self.assertNotIn("Range", original_calls[-1]["headers"])
            self.assertTrue(control.snapshot()["cancellable"])
        module._download_to_tmp_and_move(expected_size=100)
        self.assertEqual(seen[-1][0], True)
        self.assertNotIn("Range", original_calls[-1]["headers"])
        self.assertIsNone(dc.current_download_control())
        control.finish("completed")


def launch_helpers():
    tree = ast.parse((ROOT / "app" / "launch.py").read_text(encoding="utf-8"))
    names = {"download_model", "model_downloads_status", "cancel_model_download",
             "cancel_download", "_start_import_download_worker", "_update_download_record",
             "_fail_download_record"}
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            selected.append(node)

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code, self.detail = status_code, detail

    namespace = {"threading": threading, "time": time, "traceback": SimpleNamespace(print_exc=lambda: None),
                 "Request": object, "HTTPException": HTTPException,
                 "re": __import__("re"), "_model_downloads": {},
                 "_model_downloads_lock": threading.Lock(), "_civitai_downloads": {},
                 "_civitai_download_lock": threading.Lock(), "wgp": SimpleNamespace(get_model_def=lambda _: {"name": "test"}),
                 "create_download_control": dc.create_download_control,
                 "find_download_control": dc.find_download_control, "download_scope": dc.download_scope}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, "app/launch.py", "exec"), namespace)
    return namespace


class DownloadCancelApiTests(unittest.TestCase):
    def test_model_api_keeps_cancelling_active_then_allows_retry(self):
        api = launch_helpers()
        started, release = threading.Event(), threading.Event()
        controls = []

        def download(_model_type):
            control = dc.current_download_control()
            controls.append(control)
            started.set()
            release.wait(3)
            control.check()

        api["_download_model_files"] = download
        api["download_model"]("test-model")
        self.assertTrue(started.wait(2))
        control = controls[0]
        self.assertEqual(api["cancel_model_download"]("test-model")["status"], "cancelling")
        status = api["model_downloads_status"]()["downloads"]["test-model"]
        self.assertEqual(status["status"], "cancelling")
        self.assertFalse(status["cancellable"])
        self.assertEqual(api["download_model"]("test-model")["status"], "cancelling")
        self.assertEqual(len(controls), 1)
        release.set()
        deadline = time.time() + 3
        while api["model_downloads_status"]()["downloads"]["test-model"]["status"] != "cancelled":
            self.assertLess(time.time(), deadline)
            time.sleep(0.01)
        api["download_model"]("test-model")
        deadline = time.time() + 3
        while len(controls) != 2 or controls[1].snapshot()["status"] != "completed":
            self.assertLess(time.time(), deadline)
            time.sleep(0.01)
        self.assertNotEqual(controls[0].download_id, controls[1].download_id)

    def test_generic_api_validates_ids_and_preserves_completed_downloads(self):
        api = launch_helpers()
        endpoint, error = api["cancel_download"], api["HTTPException"]

        async def call(body):
            async def read():
                return body
            return await endpoint(SimpleNamespace(json=read))

        for value in ({"download_id": "../../ckpts/model"}, {}, [], {"download_id": []}):
            with self.subTest(value=value), self.assertRaises(error) as caught:
                asyncio.run(call(value))
            self.assertEqual(caught.exception.status_code, 400)
        with self.assertRaises(error) as caught:
            asyncio.run(call({"download_id": "a" * 32}))
        self.assertEqual(caught.exception.status_code, 404)
        control = dc.create_download_control()
        self.assertEqual(asyncio.run(call({"download_id": control.download_id}))["status"], "cancelling")
        control.finish("failed")
        completed = dc.create_download_control()
        completed.finish("completed")
        self.assertEqual(asyncio.run(call({"download_id": completed.download_id}))["status"], "completed")
        sealed = dc.create_download_control()
        sealed.seal()
        with self.assertRaises(error) as caught:
            asyncio.run(call({"download_id": sealed.download_id}))
        self.assertEqual(caught.exception.status_code, 409)
        sealed.finish("completed")

    def test_import_cancellation_reports_terminal_only_after_cleanup(self):
        api = launch_helpers()
        cleanup, release = threading.Event(), threading.Event()
        api["_civitai_downloads"]["import"] = {"status": "downloading", "error": None}

        def worker(_download_id):
            try:
                control = dc.current_download_control()
                while not control.cancel_requested:
                    time.sleep(0.01)
                control.check()
            finally:
                cleanup.set()
                release.wait(3)

        thread = api["_start_import_download_worker"]("import", worker)
        control = api["_civitai_downloads"]["import"]["_download_control"]
        control.request_cancel()
        self.assertTrue(cleanup.wait(2))
        self.assertEqual(control.snapshot()["status"], "cancelling")
        self.assertNotEqual(api["_civitai_downloads"]["import"]["status"], "cancelled")
        release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(api["_civitai_downloads"]["import"]["status"], "cancelled")
        self.assertIsNone(api["_civitai_downloads"]["import"]["error"])


if __name__ == "__main__":
    unittest.main()
