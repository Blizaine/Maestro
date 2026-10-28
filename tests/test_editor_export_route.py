"""CPU-only checks for Editor export revision reuse and conflict handling."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import time
import unittest
from types import SimpleNamespace


LAUNCH = Path(__file__).resolve().parents[1] / "app" / "launch.py"


class RouteError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class EditorExportRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        source = LAUNCH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(LAUNCH))
        route = next(
            node for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "editor_export"
        )
        route.decorator_list = []
        route.returns = None
        for argument in (*route.args.posonlyargs, *route.args.args, *route.args.kwonlyargs):
            argument.annotation = None

        self.project = {
            "schema_version": 6,
            "revision": 5,
            "id": "export-project",
            "name": "Saved project",
            "workspace": "export-workspace",
            "created_at": 1,
            "updated_at": 10,
            "canvas": {"width": 1920, "height": 1080, "fps": 30, "background": "#000000"},
            "assets": {},
            "tracks": [],
            "markers": [],
            "export": {"codec": "h264"},
            "exports": [],
        }
        self.saved_project = copy.deepcopy(self.project)
        self.save_calls = []
        self.jobs = {}
        self.threads = []

        class ProjectError(Exception):
            pass

        class RevisionConflict(ProjectError):
            def __init__(inner, expected_revision, current_revision):
                inner.expected_revision = expected_revision
                inner.current_revision = current_revision
                super().__init__(
                    f"Editor project changed (expected {expected_revision}, current {current_revision})"
                )

        class Thread:
            def __init__(inner, *, target, args, **kwargs):
                inner.target = target
                inner.args = args
                inner.kwargs = kwargs
                self.threads.append(inner)

            def start(inner):
                return None

        def normalize(payload, *, workspace):
            result = copy.deepcopy(payload)
            result["workspace"] = workspace
            result.setdefault("revision", 0)
            return result

        def load_project(_root, workspace, project_id):
            if workspace != self.saved_project["workspace"] or project_id != self.saved_project["id"]:
                raise FileNotFoundError(project_id)
            return copy.deepcopy(self.saved_project)

        def save_project(_root, workspace, payload, *, expected_revision=None):
            self.save_calls.append({
                "expected_revision": expected_revision,
                "payload": copy.deepcopy(payload),
            })
            current_revision = self.saved_project["revision"]
            if expected_revision is not None and expected_revision != current_revision:
                raise RevisionConflict(expected_revision, current_revision)
            saved = normalize(payload, workspace=workspace)
            saved["revision"] = current_revision + 1
            self.saved_project = copy.deepcopy(saved)
            return saved

        namespace = {
            "HTTPException": RouteError,
            "EditorProjectError": ProjectError,
            "EditorProjectRevisionConflict": RevisionConflict,
            "normalize_editor_project": normalize,
            "load_editor_project": load_project,
            "save_editor_project": save_project,
            "_editor_save_root": lambda: "save-root",
            "_get_active_workspace": lambda: "export-workspace",
            "_workspace_dir": lambda workspace: f"workspace-root/{workspace}",
            "_jobs": self.jobs,
            "threading": SimpleNamespace(Thread=Thread),
            "uuid": SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="job123456789")),
            "time": time,
        }
        module = ast.fix_missing_locations(ast.Module(body=[route], type_ignores=[]))
        exec(compile(module, str(LAUNCH), "exec"), namespace)
        self.routes = namespace

    async def _submit(self, body):
        class Request:
            async def json(inner):
                return body

        return await self.routes["editor_export"](Request())

    async def test_exact_saved_snapshot_is_queued_without_advancing_revision(self):
        result = await self._submit({
            "workspace": self.project["workspace"],
            "project": copy.deepcopy(self.project),
            "queue_mode": "held",
        })

        self.assertEqual(result, {"job_id": "job12345", "status": "held", "revision": 5})
        self.assertEqual(self.save_calls, [])
        self.assertEqual(self.jobs[result["job_id"]]["params"]["project"], self.project)

    async def test_unsaved_payload_saves_with_expected_revision_and_returns_new_revision(self):
        payload = copy.deepcopy(self.project)
        payload["name"] = "Edited before export"

        result = await self._submit({
            "workspace": self.project["workspace"],
            "project": payload,
            "queue_mode": "held",
        })

        self.assertEqual(result["revision"], 6)
        self.assertEqual(self.save_calls[0]["expected_revision"], 5)
        self.assertEqual(self.jobs[result["job_id"]]["params"]["project"]["revision"], 6)
        self.assertEqual(self.jobs[result["job_id"]]["params"]["project"]["name"], "Edited before export")

    async def test_revisionless_legacy_payload_is_saved_before_queueing(self):
        payload = copy.deepcopy(self.project)
        payload.pop("revision")
        payload["name"] = "Legacy client project"

        result = await self._submit({
            "workspace": self.project["workspace"],
            "project": payload,
            "queue_mode": "held",
        })

        self.assertEqual(result["revision"], 6)
        self.assertIsNone(self.save_calls[0]["expected_revision"])
        self.assertEqual(self.jobs[result["job_id"]]["params"]["project"]["name"], "Legacy client project")

    async def test_stale_project_revision_returns_409_without_queueing(self):
        payload = copy.deepcopy(self.project)
        payload["revision"] = 4
        payload["name"] = "Stale client project"

        with self.assertRaises(RouteError) as raised:
            await self._submit({
                "workspace": self.project["workspace"],
                "project": payload,
                "queue_mode": "held",
            })

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self.save_calls, [])
        self.assertEqual(self.jobs, {})


if __name__ == "__main__":
    unittest.main()
