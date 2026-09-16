import ast
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.upload_identity import publish_identical_upload


class UploadIdentityTests(unittest.TestCase):
    def test_repeat_reuses_exact_bytes_and_preserves_mtime(self):
        with tempfile.TemporaryDirectory() as root:
            first = publish_identical_upload(b"same", ".png", root)
            stamp = os.stat(first[1]).st_mtime_ns
            time.sleep(0.002)
            second = publish_identical_upload(b"same", ".png", root)
            self.assertEqual(first, second)
            self.assertEqual(os.stat(second[1]).st_mtime_ns, stamp)
            self.assertEqual(len(list(Path(root).iterdir())), 1)

    def test_different_bytes_with_same_source_name_get_distinct_paths(self):
        with tempfile.TemporaryDirectory() as root:
            first = publish_identical_upload(b"one", ".png", root)
            second = publish_identical_upload(b"two", ".png", root)
            self.assertNotEqual(first, second)

    def test_original_extension_is_part_of_identity(self):
        with tempfile.TemporaryDirectory() as root:
            convert = lambda _source, target: Path(target).write_bytes(b"wav")
            mp3 = publish_identical_upload(b"same", ".mp3", root, transcode_to_wav=convert)
            m4a = publish_identical_upload(b"same", ".m4a", root, transcode_to_wav=convert)
            self.assertNotEqual(mp3, m4a)

    def test_concurrent_identical_publish_converges(self):
        with tempfile.TemporaryDirectory() as root:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(
                    publish_identical_upload, b"concurrent", ".mp4", root,
                ) for _ in range(8)]
                results = [future.result() for future in futures]
            self.assertEqual(len(results), 8)
            self.assertEqual(len(set(results)), 1)
            self.assertEqual(Path(results[0][1]).read_bytes(), b"concurrent")
            self.assertEqual(len(list(Path(root).iterdir())), 1)

    def test_compressed_audio_reuses_final_wav_and_cleans_failure(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def convert(source, target):
                calls.append(source)
                Path(target).write_bytes(b"wav")
            first = publish_identical_upload(b"mp3", ".mp3", root, transcode_to_wav=convert)
            second = publish_identical_upload(b"mp3", ".mp3", root, transcode_to_wav=convert)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertTrue(first[0].endswith(".wav"))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                publish_identical_upload(b"bad", ".aac", root,
                    transcode_to_wav=lambda *_: (_ for _ in ()).throw(RuntimeError("bad")))
            self.assertEqual(list(Path(root).iterdir()), [])


class UploadRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class FakeHTTPException(RuntimeError):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        launch = ROOT / "app" / "launch.py"
        tree = ast.parse(launch.read_text(encoding="utf-8"))
        route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                     and node.name == "upload_image")
        route.decorator_list = []
        namespace = {
            "Request": object, "UploadFile": object, "File": lambda *a, **k: None,
            "HTTPException": FakeHTTPException, "MAX_IMAGE_UPLOAD_BYTES": 500 * 1024 * 1024,
            "os": os, "uuid": __import__("uuid"),
        }
        exec(compile(ast.Module(body=[route], type_ignores=[]), str(launch), "exec"), namespace)
        cls.route = staticmethod(namespace["upload_image"])

    @staticmethod
    def request():
        return SimpleNamespace(headers={})

    @staticmethod
    def upload(name, content):
        async def read(): return content
        return SimpleNamespace(filename=name, read=read)

    def test_route_wires_opt_in_metadata_and_preserves_default(self):
        with tempfile.TemporaryDirectory() as root, patch.object(os, "getcwd", return_value=root):
            opted1 = asyncio.run(self.route(self.request(), self.upload("friendly.png", b"same"), True))
            stamp = os.stat(opted1["path"]).st_mtime_ns
            opted2 = asyncio.run(self.route(self.request(), self.upload("friendly.png", b"same"), True))
            self.assertEqual(opted1, opted2)
            self.assertEqual(os.stat(opted2["path"]).st_mtime_ns, stamp)
            ordinary1 = asyncio.run(self.route(self.request(), self.upload("friendly.png", b"same"), False))
            ordinary2 = asyncio.run(self.route(self.request(), self.upload("friendly.png", b"same"), False))
            self.assertNotEqual(ordinary1["filename"], ordinary2["filename"])
            self.assertEqual(Path(ordinary1["path"]).read_bytes(), b"same")

    def test_route_audio_failure_cleans_all_files(self):
        broken = SimpleNamespace(input=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad audio")))
        with tempfile.TemporaryDirectory() as root, patch.object(os, "getcwd", return_value=root), \
                patch.dict(sys.modules, {"ffmpeg": broken}):
            with self.assertRaises(RuntimeError):
                asyncio.run(self.route(self.request(), self.upload("broken.mp3", b"bad"), True))
            uploads = Path(root) / "uploads"
            self.assertEqual(list(uploads.iterdir()) if uploads.exists() else [], [])


if __name__ == "__main__":
    unittest.main()
