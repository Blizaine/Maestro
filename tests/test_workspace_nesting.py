"""Nested workspaces: sub-folders, rename, move and delete of output folders."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import workspace_paths  # noqa: E402
from services.workspace_paths import (  # noqa: E402
    WorkspaceNameError,
    is_within_workspace,
    normalize_workspace_name,
    rebase_workspace_name,
    scan_workspace_tree,
    split_workspace_name,
    workspace_parent,
)
from services.workspace_relocation import (  # noqa: E402
    WorkspaceRebase,
    rewrite_json_file,
    rewrite_workspace_references,
)

ROOT = Path(__file__).resolve().parents[1]


class RouteError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


_ROUTE_FUNCTIONS = {
    "_safe_join", "_get_active_workspace", "_workspace_dir", "_workspace_browse_dir",
    "_workspace_file_count", "_list_workspaces", "_persist_active_workspace",
    "_workspace_name_or_400", "_existing_workspace_dir_or_404",
    "_refuse_workspace_change_while_busy", "_relocate_workspace", "_relocate_workspace_locked",
    "list_workspaces_endpoint", "set_active_workspace", "create_workspace",
    "rename_workspace", "move_workspace", "delete_workspace",
}


def _load_workspace_routes(outputs: Path, config: Path, active: str = "default"):
    tree = ast.parse((ROOT / "app" / "launch.py").read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _ROUTE_FUNCTIONS
    ]
    missing = _ROUTE_FUNCTIONS - {node.name for node in functions}
    if missing:
        raise AssertionError(f"Missing workspace route functions: {sorted(missing)}")
    for function in functions:
        function.decorator_list = []
    server_config = {"save_path": str(outputs), "services": {"active_workspace": active}}
    namespace = {
        "os": os, "json": json, "time": time, "asyncio": asyncio, "threading": threading,
        "HTTPException": RouteError,
        "wgp": types.SimpleNamespace(
            server_config=server_config, server_config_filename=str(config),
            save_path=str(outputs), image_save_path=str(outputs),
        ),
        "_jobs": {}, "_active_gen_states": {}, "_workspace_change_lock": threading.Lock(),
        "MAX_WORKSPACE_DEPTH": workspace_paths.MAX_WORKSPACE_DEPTH,
        "_RESERVED_WORKSPACES": workspace_paths.RESERVED_TOP_LEVEL,
        "WorkspaceNameError": WorkspaceNameError,
        "is_within_workspace": is_within_workspace,
        "iter_workspace_dirs": workspace_paths.iter_workspace_dirs,
        "join_workspace_name": workspace_paths.join_workspace_name,
        "rebase_workspace_name": rebase_workspace_name,
        "scan_workspace_tree": scan_workspace_tree,
        "split_workspace_name": split_workspace_name,
        "workspace_parent": workspace_parent,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "launch.py", "exec"), namespace)
    return namespace


class WorkspaceNameTests(unittest.TestCase):
    def test_nested_names_are_canonical_folder_paths(self):
        self.assertEqual(split_workspace_name("default"), ())
        self.assertEqual(split_workspace_name(""), ())
        self.assertEqual(split_workspace_name(None), ())
        self.assertEqual(split_workspace_name(" ClientA/Shoot-1 "), ("ClientA", "Shoot-1"))
        self.assertEqual(normalize_workspace_name("ClientA\\Shoot-1"), "ClientA/Shoot-1")
        self.assertEqual(normalize_workspace_name("Old Folder/With Spaces"), "Old Folder/With Spaces")
        self.assertEqual(workspace_parent("ClientA/Shoot-1"), "ClientA")
        self.assertEqual(workspace_parent("ClientA"), "default")

    def test_traversal_drives_and_empty_segments_are_rejected(self):
        for bad in ("..", "a/..", "a/./b", "/a", "a/", "a//b", "C:/x", "a\x00b", "a\nb",
                    "/".join(["a"] * 9), "x" * 241):
            with self.subTest(bad=bad), self.assertRaises(WorkspaceNameError):
                split_workspace_name(bad)

    def test_strict_names_keep_the_existing_creation_rule(self):
        self.assertEqual(split_workspace_name("A-1/b_2", strict=True), ("A-1", "b_2"))
        for bad in ("Old Folder", "_hidden", ".dot", "-dash", "a/default", "CON", "a/lpt1", "a.b"):
            with self.subTest(bad=bad), self.assertRaises(WorkspaceNameError):
                split_workspace_name(bad, strict=True)

    def test_within_and_rebase_follow_folder_boundaries(self):
        self.assertTrue(is_within_workspace("A/B/C", "A/B"))
        self.assertTrue(is_within_workspace("A", "A"))
        self.assertTrue(is_within_workspace("A", "default"))
        self.assertFalse(is_within_workspace("AB", "A"))
        self.assertFalse(is_within_workspace("A", "A/B"))
        self.assertEqual(rebase_workspace_name("A/B/C", "A/B", "X"), "X/C")
        self.assertEqual(rebase_workspace_name("A/B", "A/B", "Z/B"), "Z/B")
        self.assertIsNone(rebase_workspace_name("AB", "A", "X"))

    def test_tree_scan_lists_parents_first_and_skips_maestro_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ("B", "A/Inner/Deep", "A/_director_assets/x", "A/.maestro_editor",
                           "Worlds/worlds/abc", "Worlds/receipts", "_director_queue_assets"):
                (root / folder).mkdir(parents=True)
            (root / "A" / "clip.mp4").write_bytes(b"x")
            (root / "A" / ".favorites.json").write_text("[]")
            (root / "A" / "Inner" / "one.png").write_bytes(b"x")
            entries = scan_workspace_tree(str(root))
            self.assertEqual([e["name"] for e in entries], ["A", "A/Inner", "A/Inner/Deep", "B", "Worlds"])
            counts = {e["name"]: e["file_count"] for e in entries}
            self.assertEqual(counts["A"], 1)
            self.assertEqual(counts["A/Inner"], 1)


class WorkspaceRebaseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.outputs = self.root / "outputs"
        self.old = self.outputs / "A" / "Shoot"
        self.new = self.outputs / "Archive" / "Shoot-2"
        self.rebase = WorkspaceRebase("A/Shoot", "Archive/Shoot-2", str(self.old), str(self.new),
                                      cwd=str(self.root))

    def test_values_pointing_into_the_moved_folder_are_rewritten(self):
        data = {
            "workspace": "A/Shoot",
            "editor_workspace": "A/Shoot/Sub",
            "other_workspace": "A/Shooting",
            "path": str(self.old / "clip.mp4"),
            "relative": "outputs/A/Shoot/img.png",
            "windows_relative": "outputs\\A\\Shoot\\img.png",
            "sibling": str(self.outputs / "A" / "Shooting" / "x.mp4"),
            "url": "/api/v1/file/clip%201.mp4?workspace=A%2FShoot&size=480",
            "prompt": "A/Shoot of a cat",
            "list": [str(self.old / "a.png"), 3, None],
        }
        rebased, changed = self.rebase.rebase(data)
        self.assertTrue(changed)
        self.assertEqual(rebased["workspace"], "Archive/Shoot-2")
        self.assertEqual(rebased["editor_workspace"], "Archive/Shoot-2/Sub")
        self.assertEqual(rebased["other_workspace"], "A/Shooting")
        self.assertEqual(rebased["path"], str(self.new / "clip.mp4"))
        self.assertEqual(rebased["relative"], "outputs/Archive/Shoot-2/img.png")
        self.assertEqual(rebased["windows_relative"], os.path.join("outputs", "Archive", "Shoot-2", "img.png"))
        self.assertEqual(rebased["sibling"], data["sibling"])
        self.assertEqual(rebased["url"], "/api/v1/file/clip%201.mp4?workspace=Archive%2FShoot-2&size=480")
        self.assertEqual(rebased["prompt"], data["prompt"])
        self.assertEqual(rebased["list"], [str(self.new / "a.png"), 3, None])
        self.assertEqual(data["workspace"], "A/Shoot", "the input is not mutated")

    def test_rewrite_keeps_mtime_and_skips_unrelated_files(self):
        self.outputs.mkdir(parents=True)
        sidecar = self.outputs / "clip.meta.json"
        sidecar.write_text(json.dumps({"params": {"workspace": "A/Shoot"}}, indent=2), encoding="utf-8")
        unrelated = self.outputs / "other.meta.json"
        unrelated.write_text(json.dumps({"params": {"workspace": "B"}}), encoding="utf-8")
        os.utime(sidecar, ns=(1_000_000_000, 1_000_000_000))
        os.utime(unrelated, ns=(2_000_000_000, 2_000_000_000))
        self.assertTrue(rewrite_json_file(str(sidecar), self.rebase))
        self.assertFalse(rewrite_json_file(str(unrelated), self.rebase))
        self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8"))["params"]["workspace"], "Archive/Shoot-2")
        self.assertEqual(os.stat(sidecar).st_mtime_ns, 1_000_000_000)
        self.assertEqual(os.stat(unrelated).st_mtime_ns, 2_000_000_000)
        self.assertIn('\n  "params"', sidecar.read_text(encoding="utf-8"), "indentation is kept")

    def test_references_update_inside_the_moved_tree_and_in_other_editor_projects(self):
        moved_sidecar = self.new / "Sub" / "clip.meta.json"
        pipeline = self.new / "_director_pipeline_p1.json"
        own_project = self.new / ".maestro_editor" / "e1.json"
        cache = self.new / ".maestro_editor" / "media_cache" / "x" / "preview.json"
        other_project = self.outputs / "B" / ".maestro_editor" / "e2.json"
        root_sidecar = self.outputs / "root.meta.json"
        for path in (moved_sidecar, pipeline, own_project, cache, other_project, root_sidecar):
            path.parent.mkdir(parents=True, exist_ok=True)
        old_path = str(self.old / "Sub" / "clip.mp4")
        moved_sidecar.write_text(json.dumps({"params": {"image_start": old_path}}), encoding="utf-8")
        pipeline.write_text(json.dumps({"workspace": "A/Shoot", "output_files": [str(self.old / "f.mp4")]}), encoding="utf-8")
        own_project.write_text(json.dumps({"workspace": "A/Shoot"}), encoding="utf-8")
        cache.write_text(json.dumps({"source": old_path}), encoding="utf-8")
        other_project.write_text(json.dumps({"workspace": "B", "assets": {"a": {
            "workspace": "A/Shoot/Sub", "path": old_path,
            "url": "/api/v1/file/clip.mp4?workspace=A%2FShoot%2FSub",
        }}}), encoding="utf-8")
        root_sidecar.write_text(json.dumps({"params": {"workspace": "A/Shoot"}}), encoding="utf-8")

        result = rewrite_workspace_references(str(self.outputs), self.rebase)

        self.assertEqual(result["files_updated"], 4)
        self.assertEqual(json.loads(moved_sidecar.read_text())["params"]["image_start"],
                         str(self.new / "Sub" / "clip.mp4"))
        self.assertEqual(json.loads(pipeline.read_text())["workspace"], "Archive/Shoot-2")
        self.assertEqual(json.loads(own_project.read_text())["workspace"], "Archive/Shoot-2")
        self.assertEqual(json.loads(cache.read_text())["source"], old_path, "preview caches rebuild themselves")
        asset = json.loads(other_project.read_text())["assets"]["a"]
        self.assertEqual(asset["workspace"], "Archive/Shoot-2/Sub")
        self.assertEqual(asset["path"], str(self.new / "Sub" / "clip.mp4"))
        self.assertEqual(asset["url"], "/api/v1/file/clip.mp4?workspace=Archive%2FShoot-2%2FSub")
        self.assertEqual(json.loads(root_sidecar.read_text())["params"]["workspace"], "A/Shoot",
                         "only Editor projects are rewritten outside the moved folder")

    def test_alias_spellings_of_a_linked_output_folder_are_rewritten(self):
        rebase = WorkspaceRebase("A", "B", str(self.root / "real" / "A"), str(self.root / "real" / "B"),
                                 aliases=[(str(self.outputs / "A"), str(self.outputs / "B"))], cwd=str(self.root))
        rebased, changed = rebase.rebase([str(self.root / "real" / "A" / "x"), str(self.outputs / "A" / "y")])
        self.assertTrue(changed)
        self.assertEqual(rebased, [str(self.root / "real" / "B" / "x"), str(self.outputs / "B" / "y")])


class WorkspaceRouteTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.outputs = self.root / "outputs"
        self.config = self.root / "wgp_config.json"
        self.outputs.mkdir()
        self.director = types.ModuleType("services.director_pipeline")
        self.director.any_pipeline_active = lambda: False
        self.director.rebase_live_workspace_state = Mock()
        modules = patch.dict(sys.modules, {"services.director_pipeline": self.director})
        modules.start()
        self.addCleanup(modules.stop)
        self.ns = _load_workspace_routes(self.outputs, self.config)

    def call(self, name, body=None, *args):
        function = self.ns[name]
        if asyncio.iscoroutinefunction(function):
            return asyncio.run(function(FakeRequest(body or {})))
        return function(*args)

    def assert_status(self, status, name, body=None, *args):
        with self.assertRaises(RouteError) as caught:
            self.call(name, body, *args)
        self.assertEqual(caught.exception.status_code, status, caught.exception.detail)
        return caught.exception.detail

    def active(self):
        return self.ns["wgp"].server_config["services"]["active_workspace"]

    def test_listing_is_a_tree_with_parent_depth_and_label(self):
        (self.outputs / "ClientA" / "Shoot-1").mkdir(parents=True)
        (self.outputs / "ClientA" / "Shoot-1" / "clip.mp4").write_bytes(b"x")
        listing = self.call("list_workspaces_endpoint")["workspaces"]
        rows = [(w["name"], w["parent"], w["depth"], w["label"], w["file_count"]) for w in listing]
        self.assertEqual(rows, [
            ("default", None, 0, "default", 0),
            ("ClientA", "default", 1, "ClientA", 0),
            ("ClientA/Shoot-1", "ClientA", 2, "Shoot-1", 1),
        ])

    def test_create_sub_folder_and_activate_it(self):
        self.call("create_workspace", {"name": "ClientA"})
        created = self.call("create_workspace", {"name": "Shoot-1", "parent": "ClientA"})
        self.assertEqual(created["name"], "ClientA/Shoot-1")
        self.assertTrue((self.outputs / "ClientA" / "Shoot-1").is_dir())
        self.call("set_active_workspace", {"name": "ClientA/Shoot-1"})
        self.assertEqual(self.active(), "ClientA/Shoot-1")
        self.assertEqual(Path(self.ns["wgp"].save_path), self.outputs / "ClientA" / "Shoot-1")
        self.assertEqual(json.loads(self.config.read_text())["services"]["active_workspace"], "ClientA/Shoot-1")

    def test_create_rejects_bad_names_missing_parents_and_worlds(self):
        (self.outputs / "Worlds").mkdir()
        self.assert_status(400, "create_workspace", {"name": "has space"})
        self.assert_status(400, "create_workspace", {"name": "../escape"})
        self.assert_status(404, "create_workspace", {"name": "x", "parent": "Missing"})
        self.assert_status(400, "create_workspace", {"name": "x", "parent": "Worlds"})
        self.assert_status(400, "create_workspace", {"name": "/".join(["d"] * 9)})
        self.assert_status(400, "set_active_workspace", {"name": "a/../../etc"})

    def test_rename_moves_children_active_folder_favorites_and_references(self):
        shoot = self.outputs / "ClientA" / "Shoot-1"
        (shoot / "Selects").mkdir(parents=True)
        (shoot / "clip.mp4").write_bytes(b"x")
        (shoot / ".favorites.json").write_text('["clip.mp4"]')
        (shoot / "clip.meta.json").write_text(json.dumps({"params": {
            "workspace": "ClientA/Shoot-1", "image_start": str(shoot / "Selects" / "ref.png"),
        }}))
        self.call("set_active_workspace", {"name": "ClientA/Shoot-1/Selects"})
        self.ns["_jobs"]["done"] = {"status": "completed", "workspace": "ClientA/Shoot-1", "out_dir": str(shoot)}

        result = self.call("rename_workspace", {"name": "ClientA/Shoot-1", "new_name": "Shoot-2"})

        moved = self.outputs / "ClientA" / "Shoot-2"
        self.assertEqual(result["new_name"], "ClientA/Shoot-2")
        self.assertEqual(result["active"], "ClientA/Shoot-2/Selects")
        self.assertFalse(shoot.exists())
        self.assertTrue((moved / "Selects").is_dir())
        self.assertEqual((moved / ".favorites.json").read_text(), '["clip.mp4"]')
        self.assertEqual(self.active(), "ClientA/Shoot-2/Selects")
        self.assertEqual(Path(self.ns["wgp"].save_path), moved / "Selects")
        params = json.loads((moved / "clip.meta.json").read_text())["params"]
        self.assertEqual(params["workspace"], "ClientA/Shoot-2")
        self.assertEqual(Path(params["image_start"]), moved / "Selects" / "ref.png")
        self.assertEqual(self.ns["_jobs"]["done"]["workspace"], "ClientA/Shoot-2")
        self.assertEqual(Path(self.ns["_jobs"]["done"]["out_dir"]), moved)
        self.director.rebase_live_workspace_state.assert_called_once()
        self.assertEqual(result["references_updated"], 1)

    def test_rename_refusals(self):
        (self.outputs / "A").mkdir()
        (self.outputs / "B").mkdir()
        (self.outputs / "Worlds").mkdir()
        self.assert_status(409, "rename_workspace", {"name": "A", "new_name": "B"})
        self.assert_status(400, "rename_workspace", {"name": "A", "new_name": "X/Y"})
        self.assert_status(400, "rename_workspace", {"name": "A", "new_name": "bad name"})
        self.assert_status(400, "rename_workspace", {"name": "default", "new_name": "X"})
        self.assert_status(400, "rename_workspace", {"name": "Worlds", "new_name": "Planets"})
        self.assert_status(404, "rename_workspace", {"name": "Missing", "new_name": "X"})
        self.ns["_jobs"]["queued"] = {"status": "queued"}
        self.assert_status(409, "rename_workspace", {"name": "A", "new_name": "C"})
        self.assertTrue((self.outputs / "A").is_dir())

    @unittest.skipUnless(os.name == "nt", "case-insensitive folders")
    def test_case_only_rename_on_windows(self):
        (self.outputs / "pika").mkdir()
        self.call("rename_workspace", {"name": "pika", "new_name": "Pika"})
        self.assertIn("Pika", os.listdir(self.outputs))

    def test_move_into_folder_and_back_to_top_level(self):
        (self.outputs / "Archive").mkdir()
        (self.outputs / "ClientA" / "Shoot-1").mkdir(parents=True)
        self.call("set_active_workspace", {"name": "ClientA/Shoot-1"})
        result = self.call("move_workspace", {"name": "ClientA", "parent": "Archive"})
        self.assertEqual(result["new_name"], "Archive/ClientA")
        self.assertEqual(self.active(), "Archive/ClientA/Shoot-1")
        self.assertTrue((self.outputs / "Archive" / "ClientA" / "Shoot-1").is_dir())
        self.call("move_workspace", {"name": "Archive/ClientA/Shoot-1", "parent": "default"})
        self.assertTrue((self.outputs / "Shoot-1").is_dir())
        self.assertEqual(self.active(), "Shoot-1")

    def test_move_refusals(self):
        (self.outputs / "A" / "B").mkdir(parents=True)
        (self.outputs / "C" / "B").mkdir(parents=True)
        (self.outputs / "Worlds").mkdir()
        self.assert_status(400, "move_workspace", {"name": "A", "parent": "A"})
        self.assert_status(400, "move_workspace", {"name": "A", "parent": "A/B"})
        self.assert_status(400, "move_workspace", {"name": "A/B", "parent": "A"})
        self.assert_status(409, "move_workspace", {"name": "A/B", "parent": "C"})
        self.assert_status(400, "move_workspace", {"name": "A", "parent": "Worlds"})
        self.assert_status(404, "move_workspace", {"name": "A", "parent": "Nowhere"})
        deep = self.outputs.joinpath(*["d"] * 7)
        deep.mkdir(parents=True)
        self.assert_status(400, "move_workspace", {"name": "A", "parent": "/".join(["d"] * 7)})
        self.assertTrue((self.outputs / "A" / "B").is_dir())

    def test_delete_parent_removes_sub_folders_and_switches_to_its_parent(self):
        (self.outputs / "ClientA" / "Shoot-1" / "Selects").mkdir(parents=True)
        (self.outputs / "ClientA" / "Shoot-1" / "Selects" / "x.mp4").write_bytes(b"x")
        self.call("set_active_workspace", {"name": "ClientA/Shoot-1/Selects"})
        result = self.call("delete_workspace", None, "ClientA/Shoot-1")
        self.assertEqual(result["switched_to"], "ClientA")
        self.assertFalse(result["switched_to_default"])
        self.assertEqual(self.active(), "ClientA")
        self.assertFalse((self.outputs / "ClientA" / "Shoot-1").exists())
        self.assert_status(400, "delete_workspace", None, "default")
        self.assert_status(400, "delete_workspace", None, "ClientA/../..")
        self.assert_status(404, "delete_workspace", None, "ClientA/Gone")


class NestedDirectorAndEditorTests(unittest.TestCase):
    def test_director_lists_and_loads_pipelines_saved_in_sub_folders(self):
        import services.director_pipeline as director

        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp)
            nested = outputs / "ClientA" / "Shoot-1"
            nested.mkdir(parents=True)
            (nested / "_director_pipeline_p1.json").write_text(json.dumps({
                "pipeline_id": "p1", "status": "completed", "created_at": 1, "clips": [],
            }))
            listed = director.list_pipeline_states(str(outputs))
            self.assertEqual([(item["id"], item["workspace"]) for item in listed], [("p1", "ClientA/Shoot-1")])
            self.assertEqual(director._find_pipeline_file(str(outputs), "p1"), str(nested / "_director_pipeline_p1.json"))
            self.assertEqual(director.load_pipeline_state(str(outputs), "p1")["pipeline_id"], "p1")

    def test_director_live_state_and_queue_follow_a_move(self):
        import services.director_pipeline as director

        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp)
            old, new = outputs / "A", outputs / "B"
            queue_state = {"version": 1, "paused": True, "running": False, "entries": [
                {"id": "q1", "status": "held", "params": {"workspace": "A", "audio": str(old / "song.wav")}},
            ]}
            pipelines = {"p1": {"workspace": "A/Sub", "out_dir": str(old / "Sub"), "status": "completed"}}
            with patch.object(director, "_pipelines", pipelines), \
                    patch.object(director, "_director_queue_state", queue_state), \
                    patch.object(director, "_director_queue_base", os.path.realpath(str(outputs))):
                director.rebase_live_workspace_state(str(outputs), WorkspaceRebase("A", "B", str(old), str(new)))
                self.assertEqual(pipelines["p1"]["workspace"], "B/Sub")
                self.assertEqual(Path(pipelines["p1"]["out_dir"]), new / "Sub")
                self.assertEqual(queue_state["entries"][0]["params"]["workspace"], "B")
                saved = json.loads((outputs / "_director_queue.json").read_text())
                self.assertEqual(Path(saved["entries"][0]["params"]["audio"]), new / "song.wav")

    def test_editor_projects_accept_nested_workspaces(self):
        from services.editor_projects import EditorProjectError, workspace_directory

        with tempfile.TemporaryDirectory() as temp:
            root = os.path.realpath(temp)
            self.assertEqual(workspace_directory(root, "ClientA/Shoot-1"), os.path.join(root, "ClientA", "Shoot-1"))
            self.assertEqual(workspace_directory(root, "default"), root)
            with self.assertRaises(EditorProjectError):
                workspace_directory(root, "ClientA/../..")


class WorkspaceUiSourceTests(unittest.TestCase):
    def test_picker_offers_tree_rename_move_and_sub_folder_actions(self):
        picker = (ROOT / "ui" / "src" / "components" / "MainContent" / "WorkspaceSelector.tsx").read_text(encoding="utf-8")
        main = (ROOT / "ui" / "src" / "components" / "MainContent" / "MainContent.tsx").read_text(encoding="utf-8")
        self.assertIn("import { WorkspaceSelector } from './WorkspaceSelector'", main)
        self.assertNotIn("function WorkspaceSelector", main)
        for text in ('title="Switch workspace"', "New Workspace", "All folders", "Uploads",
                     "New sub-folder in", "Rename", "draggable={movable}", "moveWorkspace(source, target)",
                     'role="tree"', "workspaceDropProblem"):
            self.assertIn(text, picker)


if __name__ == "__main__":
    unittest.main()
