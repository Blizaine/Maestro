"""Exercise Animate's CPU API and job preflight without importing GPU startup."""
import ast
import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from fastapi import HTTPException


def function_node(name):
    tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def load_function(name, namespace):
    node = function_node(name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), "launch-viggle", "exec"), namespace)
    return namespace[name]


class ViggleSourceApiTests(unittest.TestCase):
    def endpoint(self, resolver):
        return load_function("serve_viggle_source_info", {
            "HTTPException": HTTPException, "_resolve_tool_clip_path": resolver,
        })

    def test_api_uses_resolved_source_and_shared_video_probe(self):
        resolver = Mock(return_value="resolved-source.mp4")
        source_info = {"duration_seconds": 3.0, "container_duration_seconds": 3.2}
        with patch("services.viggle_media.probe_control_video", return_value=source_info) as probe:
            result = self.endpoint(resolver)({"video_path": "uploads/source.mp4", "workspace": "bench"})
        self.assertEqual(result, source_info)
        resolver.assert_called_once_with("uploads/source.mp4", "bench")
        probe.assert_called_once_with("resolved-source.mp4")

    def test_api_rejects_missing_invalid_and_unresolvable_sources(self):
        resolver = Mock(return_value=None)
        endpoint = self.endpoint(resolver)
        for body in ({}, {"video_path": []}, {"video_path": " "},
                     {"video_path": "a.mp4", "workspace": []}, {"video_path": "missing.mp4"}):
            with self.subTest(body=body), self.assertRaises(HTTPException) as raised:
                endpoint(body)
            self.assertEqual(raised.exception.status_code, 400)

    def test_api_exposes_media_errors_as_bad_requests(self):
        with patch("services.viggle_media.probe_control_video", side_effect=ValueError("No usable video frames")):
            with self.assertRaises(HTTPException) as raised:
                self.endpoint(lambda *_: "broken.mp4")({"video_path": "broken.mp4"})
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, "No usable video frames")


class ViggleSourceJobTests(unittest.TestCase):
    def preflight(self, model_def):
        return load_function("_normalize_viggle_job_source", {
            "wgp": SimpleNamespace(get_model_def=lambda _: model_def),
            "_resolve_tool_clip_path": lambda path, workspace: f"{workspace}/{path}",
            "is_cancel_requested": lambda job: job.get("cancel_requested", False),
        })

    def test_preflight_updates_only_the_owning_job_from_shared_normalization(self):
        job = {"params": {"model_type": "viggle_animate", "video_guide": "source.mp4",
                          "_viggle_trim_end": 3.2}, "workspace": "bench"}
        params = job["params"]

        def normalize(body, *, resolve_media, aborted):
            self.assertIs(body, params)
            self.assertEqual(resolve_media(body["video_guide"]), "bench/source.mp4")
            self.assertFalse(aborted())
            return {"_viggle_trim_end": 3.0, "video_length": 72}

        with patch("services.viggle_media.normalize_control_selection", side_effect=normalize) as helper:
            self.preflight({"minimax_h3_viggle": True})(job)
        helper.assert_called_once()
        self.assertIs(job["params"], params)
        self.assertEqual(params["_viggle_trim_end"], 3.0)
        self.assertEqual(params["video_guide"], "source.mp4")

    def test_other_models_skip_probe_and_cancellation_leaves_request_intact(self):
        job = {"params": {"model_type": "other"}}
        with patch("services.viggle_media.normalize_control_selection") as helper:
            self.preflight({})(job)
        helper.assert_not_called()
        saved = copy.deepcopy(job)
        with patch("services.viggle_media.normalize_control_selection", side_effect=InterruptedError("cancelled")):
            with self.assertRaises(InterruptedError):
                self.preflight({"minimax_h3_viggle": True})(job)
        self.assertEqual(job, saved)

    def test_worker_checks_bounds_before_character_render_and_handles_cancellation(self):
        worker = function_node("_run_generation")
        calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)]
        check = next(node for node in calls if node.func.id == "_normalize_viggle_job_source")
        character = next(node for node in calls if node.func.id == "_prepare_viggle_character_frame")
        self.assertLess(check.lineno, character.lineno)
        guards = [node for node in ast.walk(worker) if isinstance(node, ast.Try)
                  and any(child is check for statement in node.body for child in ast.walk(statement))]
        guard = min(guards, key=lambda node: node.end_lineno - node.lineno)
        handler = next(node for node in guard.handlers if isinstance(node.type, ast.Name)
                       and node.type.id == "InterruptedError")
        self.assertIn("cancelled", [node.value for node in ast.walk(handler) if isinstance(node, ast.Constant)])


if __name__ == "__main__":
    unittest.main()
