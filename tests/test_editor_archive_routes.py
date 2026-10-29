"""Focused, mocked checks for Editor history and portable package routes."""
from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path
import tempfile
import time
import traceback
import unittest
from types import SimpleNamespace
import uuid


LAUNCH = Path(__file__).resolve().parents[1] / "app" / "launch.py"


class RouteError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class EditorArchiveRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        source = LAUNCH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(LAUNCH))
        names = {
            "_editor_workspace_package_root",
            "_run_editor_package",
            "editor_project_revisions",
            "editor_project_restore_revision",
            "editor_project_package",
            "editor_project_import",
        }
        selected = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                node.decorator_list = []
                node.returns = None
                for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                    arg.annotation = None
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_MAX_EDITOR_ARCHIVE_UPLOAD_BYTES"
                for target in node.targets
            ):
                selected.append(node)
        self.assertEqual(
            {node.name for node in selected if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))},
            names,
        )
        self.temp = tempfile.TemporaryDirectory(prefix="editor-archive-route-test-")
        self.root = self.temp.name
        self.project = {
            "id": "project-a", "name": "Archive test", "workspace": "workspace-a",
            "revision": 7, "assets": {}, "tracks": [],
        }
        self.jobs = {}
        self.threads = []
        self.exports = []
        self.imports = []

        class Thread:
            def __init__(inner, *, target, args, **kwargs):
                inner.target = target
                inner.args = args
                inner.kwargs = kwargs
                self.threads.append(inner)

            def start(inner):
                return None

        def workspace_directory(save_root, workspace):
            workspace = str(workspace or "default").strip() or "default"
            if workspace in {".", ".."} or os.path.basename(workspace) != workspace or any(c in workspace for c in ("/", "\\", "\0")):
                raise EditorProjectError("Invalid workspace name")
            return os.path.realpath(save_root) if workspace == "default" else os.path.join(os.path.realpath(save_root), workspace)

        class EditorProjectError(Exception):
            pass

        class EditorProjectRevisionConflict(EditorProjectError):
            pass

        class EditorProjectArchiveError(EditorProjectError):
            pass

        class EditorProjectArchiveCancelled(Exception):
            pass

        def load_project(root, workspace, project_id):
            if project_id != self.project["id"] or workspace != self.project["workspace"]:
                raise FileNotFoundError(project_id)
            return dict(self.project)

        def export_archive(project, **kwargs):
            self.exports.append((project, kwargs))
            Path(kwargs["output_path"]).write_bytes(b"mock zip")
            return kwargs["output_path"]

        def import_archive(path, **kwargs):
            self.imports.append((path, kwargs, Path(path).read_bytes()))
            return {"id": "imported-project", "workspace": kwargs["workspace"]}

        namespace = {
            "HTTPException": RouteError,
            "EditorProjectError": EditorProjectError,
            "EditorProjectRevisionConflict": EditorProjectRevisionConflict,
            "EditorProjectArchiveError": EditorProjectArchiveError,
            "EditorProjectArchiveCancelled": EditorProjectArchiveCancelled,
            "EDITOR_ARCHIVE_UNPACKED_MAX_BYTES": 20 * 1024 * 1024 * 1024,
            "workspace_directory": workspace_directory,
            "_editor_save_root": lambda: self.root,
            "_get_active_workspace": lambda: "workspace-a",
            "_editor_uploads_root": lambda: os.path.join(self.root, "uploads"),
            "_jobs": self.jobs,
            "_active_gen_states": {},
            "list_editor_project_revisions": lambda *args: [{"revision": 6}],
            "load_editor_project": load_project,
            "restore_editor_project_revision": lambda *args, **kwargs: {**self.project, "revision": 8},
            "export_editor_project_archive": export_archive,
            "import_editor_project_archive": import_archive,
            "try_start": lambda job, **kwargs: (job.update(status="running") is None) or True,
            "register_abort_state": lambda *args: True,
            "unregister_abort_state": lambda *args: None,
            "is_cancel_requested": lambda job: False,
            "update_job": lambda *args, **kwargs: True,
            "record_job_outputs": lambda job, files: job.update(output_files=files),
            "finish_job": lambda job, status, **kwargs: (job.update(status=status, **kwargs) is None) or True,
            "threading": SimpleNamespace(Thread=Thread),
            "uuid": SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="abc123def456")),
            "time": time,
            "os": os,
            "tempfile": tempfile,
            "traceback": traceback,
        }
        module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
        exec(compile(module, str(LAUNCH), "exec"), namespace)
        self.routes = namespace

    def tearDown(self):
        self.temp.cleanup()

    async def test_package_uses_exact_saved_revision_and_server_owned_path(self):
        class Request:
            headers = {}

            async def json(self):
                return {"workspace": "workspace-a", "revision": 7}

        result = await self.routes["editor_project_package"]("project-a", Request())
        self.assertEqual(result, {"job_id": "abc123def456", "status": "queued"})
        job = self.routes["_jobs"][result["job_id"]]
        self.assertEqual(job["kind"], "editor_package")
        self.assertEqual(job["params"]["project"], self.project)
        self.assertEqual(job["package_filename"], "editor-project-abc123def456.maestro-project.zip")
        expected_root = os.path.join(self.root, "workspace-a")
        self.assertEqual(os.path.realpath(job["package_root"]), os.path.realpath(expected_root))
        self.assertEqual(len(self.threads), 1)
        self.assertIs(self.threads[0].target, self.routes["_run_editor_package"])

        # Exercise the queued worker and ensure its output remains in the
        # server-selected workspace root.
        worker = self.threads[0]
        worker.target(*worker.args)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["output_files"], [job["package_filename"]])
        self.assertEqual(os.path.realpath(self.exports[0][1]["output_path"]), os.path.realpath(os.path.join(expected_root, job["package_filename"])))

    async def test_package_rejects_client_path_and_stale_revision(self):
        class Request:
            headers = {}

            def __init__(self, body):
                self.body = body

            async def json(self):
                return self.body

        with self.assertRaises(RouteError) as raised:
            await self.routes["editor_project_package"](
                "project-a", Request({"workspace": "workspace-a", "revision": 7, "output_path": "../../escape.zip"})
            )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(self.routes["_jobs"], {})

        with self.assertRaises(RouteError) as raised:
            await self.routes["editor_project_package"](
                "project-a", Request({"workspace": "workspace-a", "revision": 6})
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self.routes["_jobs"], {})

    async def test_package_worker_rejects_traversal_before_export(self):
        for filename in ("../escape.zip", "..\\escape.zip", "/escape.zip", r"C:\escape.zip"):
            with self.subTest(filename=filename):
                job = {
                    "id": "bad-package", "status": "queued", "workspace": "workspace-a",
                    "package_root": os.path.join(self.root, "workspace-a"),
                    "package_filename": filename, "params": {"project": self.project},
                }
                self.routes["_jobs"][job["id"]] = job
                self.routes["_run_editor_package"](job["id"])
                self.assertEqual(job["status"], "failed")
                self.assertEqual(self.exports, [])

    async def test_revision_listing_and_restore_route_shapes(self):
        result = self.routes["editor_project_revisions"]("project-a", "workspace-a")
        self.assertEqual(result, {"revisions": [{"revision": 6}]})

        class Request:
            async def json(self):
                return {"workspace": "workspace-a", "revision": 6, "expected_revision": 7}

        restored = await self.routes["editor_project_restore_revision"]("project-a", Request())
        self.assertEqual(restored["revision"], 8)

        conflict_type = self.routes["EditorProjectRevisionConflict"]

        def conflict(*_args, **_kwargs):
            raise conflict_type("Current revision changed")

        self.routes["restore_editor_project_revision"] = conflict
        with self.assertRaises(RouteError) as raised:
            await self.routes["editor_project_restore_revision"]("project-a", Request())
        self.assertEqual(raised.exception.status_code, 409)

    async def test_import_streams_bounded_chunks_and_cleans_staging_file(self):
        class Upload:
            def __init__(self):
                self.chunks = [b"first", b"second", b""]
                self.requested_sizes = []
                self.closed = False

            async def read(self, size):
                self.requested_sizes.append(size)
                return self.chunks.pop(0)

            async def close(self):
                self.closed = True

        upload = Upload()

        class Form:
            def get(self, key):
                return {"file": upload, "workspace": "workspace-a"}.get(key)

        class Request:
            headers = {}

            async def form(self):
                return Form()

        result = await self.routes["editor_project_import"](Request())
        self.assertEqual(result["id"], "imported-project")
        self.assertEqual(self.imports[0][2], b"firstsecond")
        self.assertEqual(upload.requested_sizes, [1024 * 1024] * 3)
        self.assertTrue(upload.closed)
        self.assertFalse(os.path.exists(self.imports[0][0]))

    async def test_import_rejects_over_limit_upload_and_removes_staging_file(self):
        self.routes["_MAX_EDITOR_ARCHIVE_UPLOAD_BYTES"] = 5

        class Upload:
            closed = False

            async def read(self, _size):
                return b"123456"

            async def close(self):
                self.closed = True

        upload = Upload()

        class Form:
            def get(self, key):
                return {"file": upload, "workspace": "workspace-a"}.get(key)

        class Request:
            headers = {}

            async def form(self):
                return Form()

        with self.assertRaises(RouteError) as raised:
            await self.routes["editor_project_import"](Request())
        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(self.imports, [])
        self.assertTrue(upload.closed)
        package_root = os.path.join(self.root, "workspace-a")
        self.assertEqual(os.listdir(package_root), [])


if __name__ == "__main__":
    unittest.main()
