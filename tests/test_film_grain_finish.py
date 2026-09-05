"""Model-free coverage for the standalone Film Grain finish workflow."""
from __future__ import annotations

import ast
import contextlib
import os
from pathlib import Path
import tempfile
import time
import traceback
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = ROOT / "app" / "launch.py"


def _load_worker(namespace: dict):
    source = LAUNCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(LAUNCH_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_tool_film_grain"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(LAUNCH_PATH), "exec"), namespace)
    return namespace["_run_tool_film_grain"]


class _FakeWgp:
    save_path = ""
    server_config = {"video_container": "mp4"}

    @staticmethod
    def get_available_filename(out_dir, source_name, suffix, force_extension):
        stem = Path(source_name).stem
        return str(Path(out_dir) / f"{stem}{suffix}{force_extension}")

    @staticmethod
    def format_time(_elapsed):
        return "0s"


class TestFilmGrainFinish(unittest.TestCase):
    def test_worker_creates_a_new_copy_and_records_finish_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mov"
            source.write_bytes(b"source-video")
            job = {
                "params": {
                    "video_path": str(source),
                    "intensity": 0.23,
                    "saturation": 0.61,
                },
                "workspace": "default",
                "out_dir": tmp,
                "status": "queued",
            }
            calls = {}

            @contextlib.contextmanager
            def generation_slot(_lock, _job):
                yield True

            def apply_grain(path, intensity, saturation):
                calls["grain"] = (path, intensity, saturation)
                with open(path, "ab") as handle:
                    handle.write(b"+grain")

            def update_job(target, **changes):
                target.update(changes)
                return True

            def finish_job(target, status, **changes):
                target.update(changes)
                target["status"] = status
                return True

            def write_sidecar(_out_dir, filename, **kwargs):
                calls["sidecar"] = (filename, kwargs)

            namespace = {
                "os": os,
                "time": time,
                "traceback": traceback,
                "_jobs": {"grainjob": job},
                "_gen_lock": object(),
                "_active_gen_states": {},
                "generation_slot": generation_slot,
                "try_start": lambda *_args, **_kwargs: True,
                "register_abort_state": lambda *_args, **_kwargs: True,
                "unregister_abort_state": lambda *_args, **_kwargs: None,
                "is_cancel_requested": lambda *_args, **_kwargs: False,
                "update_job": update_job,
                "finish_job": finish_job,
                "record_job_outputs": lambda target, files: target.update(
                    output_files=list(files)
                ),
                "_resolve_tool_clip_path": lambda raw, _workspace: raw,
                "_apply_film_grain_to_file": apply_grain,
                "_write_tool_sidecar": write_sidecar,
                "wgp": _FakeWgp(),
            }
            worker = _load_worker(namespace)

            self.assertTrue(worker("grainjob"))
            output = Path(tmp) / "source_film_grain.mp4"
            self.assertEqual(source.read_bytes(), b"source-video")
            self.assertEqual(output.read_bytes(), b"source-video+grain")
            self.assertEqual(calls["grain"], (str(output), 0.23, 0.61))
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["output_files"], [output.name])
            filename, sidecar = calls["sidecar"]
            self.assertEqual(filename, output.name)
            self.assertEqual(sidecar["tool"], "film_grain")
            self.assertEqual(
                sidecar["params"]["_studio_video_workflow"],
                "film_grain",
            )

    def test_api_route_and_ui_workflow_are_registered(self):
        launch_source = LAUNCH_PATH.read_text(encoding="utf-8")
        selector_source = (
            ROOT / "ui" / "src" / "components" / "Sidebar"
            / "VideoWorkflowSelector.tsx"
        ).read_text(encoding="utf-8")
        editor_source = (
            ROOT / "ui" / "src" / "editor" / "EditorInspector.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn('@api.post("/api/v1/tools/film-grain")', launch_source)
        self.assertIn("value: 'film_grain'", selector_source)
        self.assertIn("id: 'film_grain'", editor_source)


if __name__ == "__main__":
    unittest.main()
