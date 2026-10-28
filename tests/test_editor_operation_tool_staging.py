from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

LAUNCH = Path(__file__).resolve().parents[1] / "app" / "launch.py"


class EditorOperationError(ValueError):
    pass


class EditorRenderCancelled(RuntimeError):
    pass


def _load_staging_helpers(namespace: dict[str, object]) -> None:
    tree = ast.parse(LAUNCH.read_text(encoding="utf-8"), filename=str(LAUNCH))
    names = {"_stage_editor_operation_tool_source", "_cleanup_editor_operation_tool_source"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(LAUNCH), "exec"), namespace)


def _operation(source_path: str) -> dict:
    return {
        "operation_id": "editor-op-123456",
        "client_submission_id": "submission-123456",
        "tool": "upscale",
        "job_kind": "upscale",
        "job_id": "job-editor-1",
        "status": "running",
        "workspace": "workspace-a",
        # The 4-second source interval represents a 2-second timeline clip at
        # 2x speed, beginning at a nonzero source offset.
        "source_snapshot": {
            "asset_path": source_path,
            "source_in": 2.5,
            "duration": 2.0,
            "speed": 2.0,
        },
        "source_range": {"start": 2.5, "end": 6.5},
    }


class FakeProcess:
    def __init__(self, command: list[str], **_kwargs):
        self.command = command
        self.returncode = 0
        Path(command[-1]).write_bytes(b"staged-media")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_editor_tool_stages_only_the_operation_range_and_preserves_audio_mapping(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-media")
    operation = _operation(str(source))
    commands: list[list[str]] = []

    class RecordingProcess(FakeProcess):
        def __init__(self, command: list[str], **kwargs):
            commands.append(command)
            super().__init__(command, **kwargs)

    namespace = {
        "EditorOperationError": EditorOperationError,
        "EditorRenderCancelled": EditorRenderCancelled,
        "get_editor_operation_record": lambda _root, _operation_id: operation,
        "_editor_operations_root": lambda: tmp_path / "operations",
        "_resolve_tool_clip_path": lambda raw, _workspace: str(source) if str(raw) == str(source) else None,
        "_editor_operation_ffmpeg_binary": lambda: sys.executable,
        "probe_media": lambda _path: {"duration": 12.0, "has_audio": True},
        "update_job": lambda *_args, **_kwargs: True,
        "math": math,
        "os": os,
        "time": time,
    }
    _load_staging_helpers(namespace)
    job = {
        "id": "job-editor-1",
        "editor_operation_id": operation["operation_id"],
        "client_submission_id": operation["client_submission_id"],
        "workspace": operation["workspace"],
        "progress": 0,
    }

    with patch.object(subprocess, "Popen", RecordingProcess):
        temporary, staged = namespace["_stage_editor_operation_tool_source"](
            job,
            "upscale",
            str(source),
            abort_check=lambda: False,
        )
    try:
        assert Path(staged).read_bytes() == b"staged-media"
        command = commands[0]
        assert command[command.index("-ss") + 1] == "2.5"
        assert command[command.index("-t") + 1] == "4"
        assert "0:v:0" in command
        assert "0:a?" in command
        assert "aac" in command
    finally:
        namespace["_cleanup_editor_operation_tool_source"](temporary)
    assert not Path(staged).exists(), "staged source is removed after the tool finishes"


def test_editor_tool_rejects_wrong_job_kind_before_ffmpeg(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-media")
    operation = _operation(str(source))
    namespace = {
        "EditorOperationError": EditorOperationError,
        "EditorRenderCancelled": EditorRenderCancelled,
        "get_editor_operation_record": lambda _root, _operation_id: operation,
        "_editor_operations_root": lambda: tmp_path / "operations",
        "_resolve_tool_clip_path": lambda raw, _workspace: str(source) if str(raw) == str(source) else None,
        "_editor_operation_ffmpeg_binary": lambda: sys.executable,
        "probe_media": lambda _path: {"duration": 12.0},
        "update_job": lambda *_args, **_kwargs: True,
        "math": math,
        "os": os,
        "time": time,
    }
    _load_staging_helpers(namespace)
    job = {
        "id": "job-editor-1",
        "editor_operation_id": operation["operation_id"],
        "client_submission_id": operation["client_submission_id"],
        "workspace": operation["workspace"],
    }
    with patch.object(subprocess, "Popen", side_effect=AssertionError("ffmpeg must not run")):
        with pytest.raises(EditorOperationError, match="does not own"):
            namespace["_stage_editor_operation_tool_source"](
                job,
                "film_grain",
                str(source),
                abort_check=lambda: False,
            )


def test_non_editor_tool_keeps_existing_full_source_behavior(tmp_path):
    source = str(tmp_path / "source.mp4")
    namespace = {"EditorOperationError": EditorOperationError, "EditorRenderCancelled": EditorRenderCancelled}
    _load_staging_helpers(namespace)
    temporary, staged = namespace["_stage_editor_operation_tool_source"](
        {"id": "studio-job"},
        "upscale",
        source,
        abort_check=lambda: False,
    )
    assert temporary is None
    assert staged == source


def test_cancel_during_staging_terminates_ffmpeg_and_cleans_temporary_range(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-media")
    operation = _operation(str(source))
    namespace = {
        "EditorOperationError": EditorOperationError,
        "EditorRenderCancelled": EditorRenderCancelled,
        "get_editor_operation_record": lambda _root, _operation_id: operation,
        "_editor_operations_root": lambda: tmp_path / "operations",
        "_resolve_tool_clip_path": lambda raw, _workspace: str(source) if str(raw) == str(source) else None,
        "_editor_operation_ffmpeg_binary": lambda: sys.executable,
        "probe_media": lambda _path: {"duration": 12.0},
        "update_job": lambda *_args, **_kwargs: True,
        "math": math,
        "os": os,
        "time": time,
    }
    _load_staging_helpers(namespace)

    class WaitingProcess(FakeProcess):
        def poll(self):
            return None

    job = {
        "id": "job-editor-1",
        "editor_operation_id": operation["operation_id"],
        "client_submission_id": operation["client_submission_id"],
        "workspace": operation["workspace"],
    }
    with patch.object(subprocess, "Popen", WaitingProcess):
        with pytest.raises(EditorRenderCancelled):
            namespace["_stage_editor_operation_tool_source"](
                job,
                "upscale",
                str(source),
                abort_check=lambda: True,
            )
