"""CPU-only checks for Editor audio API routing and revision guards."""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
import unittest
from types import SimpleNamespace


LAUNCH = Path(__file__).resolve().parents[1] / "app" / "launch.py"


class RouteError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class EditorAudioRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        source = LAUNCH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(LAUNCH))
        names = {
            "_load_editor_audio_request",
            "editor_audio_capabilities_route",
            "editor_audio_transcribe_route",
            "editor_audio_analyze_route",
        }
        functions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
        ]
        assert len(functions) == len(names)
        for node in functions:
            node.decorator_list = []
            node.returns = None
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                arg.annotation = None

        self.project = {"id": "project-a", "workspace": "workspace-a", "revision": 4}
        self.loaded = []
        self.analysis_calls = []
        self.transcription_calls = []

        async def to_thread(fn, *args, **kwargs):
            if fn is analyze:
                self.analysis_calls.append((args, kwargs))
            if fn is transcribe:
                self.transcription_calls.append((args, kwargs))
            return fn(*args, **kwargs)

        def load_project(root, workspace, project_id):
            self.loaded.append((root, workspace, project_id))
            return self.project

        def analyze(project, item_id, **kwargs):
            return {"item_id": item_id, "gain": 0.5}

        def transcribe(project, item_id, **kwargs):
            return {"item_id": item_id, "cues": []}

        namespace = {
            "HTTPException": RouteError,
            "EditorProjectError": type("EditorProjectError", (Exception,), {}),
            "asyncio": SimpleNamespace(to_thread=to_thread),
            "load_editor_project": load_project,
            "_editor_save_root": lambda: "save-root",
            "_editor_uploads_root": lambda: "uploads-root",
            "_get_active_workspace": lambda: "active-workspace",
            "_app_dir": "app-root",
            "editor_audio_capabilities": lambda root: {"device": "cpu", "downloads": False, "root": root},
            "analyze_editor_audio": analyze,
            "transcribe_editor_audio": transcribe,
        }
        module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
        exec(compile(module, str(LAUNCH), "exec"), namespace)
        self.routes = namespace

    async def test_analysis_loads_the_saved_project_and_runs_off_the_event_loop(self):
        class Request:
            async def json(self):
                return {"workspace": "workspace-a", "project_id": "project-a", "item_id": "clip-a", "revision": 4}

        result = await self.routes["editor_audio_analyze_route"](Request())
        self.assertEqual(result, {"item_id": "clip-a", "gain": 0.5})
        self.assertEqual(self.loaded, [("save-root", "workspace-a", "project-a")])
        self.assertEqual(len(self.analysis_calls), 1)
        args, kwargs = self.analysis_calls[0]
        self.assertIs(args[0], self.project)
        self.assertEqual(args[1], "clip-a")
        self.assertEqual(kwargs, {"save_root": "save-root", "uploads_root": "uploads-root"})

    async def test_stale_revision_is_rejected_before_transcription_starts(self):
        class Request:
            async def json(self):
                return {"workspace": "workspace-a", "project_id": "project-a", "item_id": "clip-a", "revision": 3}

        with self.assertRaises(RouteError) as raised:
            await self.routes["editor_audio_transcribe_route"](Request())
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self.transcription_calls, [])

    async def test_capabilities_are_local_cpu_only(self):
        result = self.routes["editor_audio_capabilities_route"]()
        self.assertEqual(result, {"device": "cpu", "downloads": False, "root": "app-root"})


if __name__ == "__main__":
    unittest.main()
