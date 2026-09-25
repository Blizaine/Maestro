import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))
from fastapi import HTTPException
from services.upload_media import upload_path, references_upload, delete_upload_file


class UploadMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.uploads = self.root / "uploads"
        self.uploads.mkdir()
        self.media = self.uploads / "example.mp4"
        self.media.write_bytes(b"test-only media")
        self.sidecar = self.media.with_suffix(".meta.json")
        self.sidecar.write_text("{}")

    def test_path_rejects_traversal_directories_nonmedia_and_windows_streams(self):
        for name in ("../outside.mp4", "..\\outside.mp4", "C:\\outside.mp4", "nested/example.mp4", "example.mp4:stream", "settings.json", ".hidden.mp4", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                upload_path(str(self.uploads), name)
        with patch("services.upload_media.os.path.realpath", side_effect=[str(self.uploads), str(self.root / "outside.mp4")]):
            with self.assertRaisesRegex(ValueError, "inside"):
                upload_path(str(self.uploads), "example.mp4")

    def test_nested_job_references_and_other_folder_identity(self):
        for value in (self.media.name, str(self.media), "uploads/example.mp4", "/api/v1/uploads/example.mp4", "http://localhost:42015/api/v1/uploads/example.mp4"):
            self.assertTrue(references_upload({"refs": [{"path": value}]}, str(self.media)), value)
        self.assertFalse(references_upload({"video": str(self.root / "outputs" / self.media.name)}, str(self.media)))
        self.assertFalse(references_upload("A story about example.mp4", str(self.media)))

    def test_delete_only_media_and_matching_sidecar(self):
        other = self.uploads / "other.mp4"
        other.write_bytes(b"untouched")
        def remove(path):
            os.remove(path)
            return {"deleted": True}
        result = delete_upload_file(str(self.uploads), self.media.name, safe_delete=remove)
        self.assertEqual(result["deleted"], self.media.name)
        self.assertFalse(self.media.exists())
        self.assertFalse(self.sidecar.exists())
        self.assertTrue(other.exists())
        self.assertEqual(delete_upload_file(str(self.uploads), self.media.name, safe_delete=remove), {"deleted": self.media.name})

    def test_locked_file_retains_sidecar_and_reports_error(self):
        remover = Mock(return_value={"deleted": False, "reason": "locked"})
        with self.assertRaises(PermissionError):
            delete_upload_file(str(self.uploads), self.media.name, safe_delete=remover)
        self.assertTrue(self.media.exists())
        self.assertTrue(self.sidecar.exists())
        self.assertEqual(remover.call_count, 1)

    def route(self, jobs):
        tree = ast.parse((Path(__file__).parents[1] / "app/launch.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "delete_upload")
        function.decorator_list = []
        namespace = {"os": os, "sys": sys, "HTTPException": HTTPException, "_jobs": jobs, "snapshot_job": dict}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "launch.py", "exec"), namespace)
        return namespace["delete_upload"]

    def test_route_refuses_active_inputs_and_never_calls_delete(self):
        for status in ("queued", "running", "held", "cancelling"):
            with self.subTest(status=status):
                route = self.route({"one": {"status": status, "params": {"image_refs": [self.media.name]}}})
                with patch("os.getcwd", return_value=str(self.root)), patch("services.win_safe_files.safe_delete") as remove:
                    with self.assertRaises(HTTPException) as error:
                        route(self.media.name)
                    self.assertEqual(error.exception.status_code, 409)
                    remove.assert_not_called()

    def test_route_refuses_director_review_inputs(self):
        director = SimpleNamespace(_pipeline_lock=threading.Lock(), _ACTIVE_PIPELINE_STATUSES={"review"},
            _pipelines={"test": {"status": "review", "params": {"ref": str(self.media)}}})
        with patch.dict(sys.modules, {"services.director_pipeline": director}), patch("os.getcwd", return_value=str(self.root)), patch("services.win_safe_files.safe_delete") as remove:
            with self.assertRaises(HTTPException) as error:
                self.route({})(self.media.name)
            self.assertEqual(error.exception.status_code, 409)
            remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
