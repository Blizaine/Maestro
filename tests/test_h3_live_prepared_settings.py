"""Live-generation settings-version and reviewed H3 geometry regressions."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
import unittest

import torch

if not torch.cuda.is_available():
    raise unittest.SkipTest('Live WGP settings integration requires the installed CUDA runtime')


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
LAUNCH = APP / "launch.py"
sys.path.insert(0, str(APP))

_argv = sys.argv
_cwd = Path.cwd()
try:
    # wgp owns the app CLI parser at import time; keep unittest selectors out.
    sys.argv = [sys.argv[0]]
    os.chdir(APP)
    import wgp
finally:
    os.chdir(_cwd)
    sys.argv = _argv


def load_function(name: str):
    source = LAUNCH.read_text(encoding="utf-8")
    module = ast.parse(source)
    node = next(
        item for item in module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LAUNCH), "exec"), namespace)
    return namespace[name]


class LivePreparedSettingsTests(unittest.TestCase):
    def setUp(self):
        self.stamp = load_function("_stamp_live_generation_settings_version")
        self.stamp.__globals__["wgp"] = wgp
        self.freeze = load_function("_freeze_reviewed_h3_runtime_geometry")
        self.guard = load_function("_reviewed_h3_runtime_geometry_error")
        self.sidecar_params = load_function("_generation_sidecar_params")

    @staticmethod
    def request(version_marker=...):
        params = {
            "model_type": "minimax_h3_ref2va_fused_turbo",
            "prompt": "Frozen source",
            "video_length": 668,
            "sliding_window_size": 243,
            "sliding_window_overlap": 18,
            "multi_prompts_gen_type": 0,
            "minimax_h3_reference_sequence": True,
            "minimax_h3_sequence_prompt_mode": "adaptive",
            "minimax_h3_references": [
                {"type": "image", "path": "visual.jpg", "role": "Blaine", "image_intent": "identity"},
                {"type": "audio", "path": "voice.wav", "role": "Blaine", "audio_intent": "voice"},
            ],
            "h3_window_prompts": ["window one", "window two", "window three"],
            "h3_window_plan": {
                "total_frames": 668,
                "window_frames": 243,
                "overlap_frames": 18,
                "window_count": 3,
            },
            "_h3_window_plan_reviewed": True,
        }
        if version_marker is not ...:
            params["settings_version"] = version_marker
        return params

    def normalize(self, params):
        model_type, error = wgp._process_task_params(params, {})
        self.assertIsNone(error)
        self.assertEqual(model_type, "minimax_h3_ref2va_fused_turbo")

    def test_missing_and_null_live_versions_retain_reviewed_geometry(self):
        for marker in (..., None):
            with self.subTest(marker=marker):
                prepared = self.request(marker)
                self.stamp(prepared)
                self.assertEqual(prepared["settings_version"], wgp.settings_version)
                self.freeze(prepared, True)
                expected = prepared.pop("_reviewed_h3_runtime_geometry")
                self.normalize(prepared)
                self.assertEqual(prepared["video_length"], 668)
                self.assertEqual(prepared["sliding_window_size"], 243)
                self.assertEqual(prepared["sliding_window_overlap"], 18)
                self.assertEqual(len(prepared["h3_window_prompts"]), 3)
                self.assertEqual(prepared.get("audio_prompt_type"), "")
                self.assertEqual(prepared.get("video_prompt_type"), "")
                self.assertEqual(prepared.get("image_prompt_type"), "")
                self.assertEqual(
                    [(item["type"], item["role"]) for item in prepared["minimax_h3_references"]],
                    [("image", "Blaine"), ("audio", "Blaine")],
                )
                self.assertEqual(
                    wgp.compute_sliding_window_no(668, 243, 0, 18),
                    3,
                )
                self.assertIsNone(self.guard(expected, prepared))

    def test_explicit_old_or_unversioned_saved_manifest_still_migrates(self):
        for marker in (..., 2.1):
            with self.subTest(marker=marker):
                saved = self.request(marker)
                self.normalize(saved)
                self.assertNotEqual(saved.get("sliding_window_size"), 243)

    def test_runtime_guard_rejects_normalized_geometry_drift(self):
        reviewed = self.request(wgp.settings_version)
        self.freeze(reviewed, True)
        expected = reviewed.pop("_reviewed_h3_runtime_geometry")
        runtime = dict(reviewed)
        runtime["sliding_window_size"] = 345
        message = self.guard(expected, runtime)
        self.assertIn("expected", message)
        self.assertIn("No generation was started", message)

        for malformed in ({}, {"video_length": "bad"}, "invalid"):
            with self.subTest(malformed=malformed):
                self.assertIn("became invalid", self.guard(malformed, runtime))

    def test_reviewed_independent_sequence_skips_single_task_geometry_guard(self):
        prepared = self.request(wgp.settings_version)
        prepared["multi_prompts_gen_type"] = 3
        # All clip lengths sit on the selected H3 model's 5 + 17n lattice.
        prepared["per_clip_frames"] = [243, 243, 226]
        prepared["_reviewed_h3_runtime_geometry"] = {"stale": True}

        self.freeze(prepared, True)

        self.assertNotIn("_reviewed_h3_runtime_geometry", prepared)
        prompts = list(prepared["h3_window_prompts"])
        clip_frames = list(prepared["per_clip_frames"])
        tasks = [
            {
                **prepared,
                "prompt": prompt,
                "h3_window_prompts": [prompt],
                "video_length": frames,
                "multi_prompts_gen_type": 0,
            }
            for prompt, frames in zip(prompts, clip_frames, strict=True)
        ]
        for task in tasks:
            self.normalize(task)
        self.assertEqual(len(tasks), 3)
        self.assertEqual([task["video_length"] for task in tasks], clip_frames)
        self.assertEqual(
            [task["h3_window_prompts"] for task in tasks],
            [[prompt] for prompt in prompts],
        )
        self.assertTrue(
            all("_reviewed_h3_runtime_geometry" not in task for task in tasks),
        )

    def test_generation_sidecar_strips_only_private_runtime_geometry(self):
        original = self.request(wgp.settings_version)
        self.freeze(original, True)

        sidecar = self.sidecar_params(original)

        self.assertNotIn("_reviewed_h3_runtime_geometry", sidecar)
        self.assertEqual(sidecar["settings_version"], wgp.settings_version)
        self.assertEqual(sidecar["h3_window_plan"], original["h3_window_plan"])
        self.assertEqual(sidecar["h3_window_prompts"], original["h3_window_prompts"])
        self.assertIn("_reviewed_h3_runtime_geometry", original)

    def test_current_and_explicit_old_versions_are_not_overwritten(self):
        for version in (wgp.settings_version, 2.1, 0):
            with self.subTest(version=version):
                params = self.request(version)
                self.stamp(params)
                self.assertEqual(params["settings_version"], version)

    def test_live_preparation_calls_version_boundary(self):
        source = LAUNCH.read_text(encoding="utf-8")
        function = next(
            item for item in ast.parse(source).body
            if isinstance(item, ast.AsyncFunctionDef)
            and item.name == "_prepare_generation_submission"
        )
        text = ast.get_source_segment(source, function)
        self.assertIn("_stamp_live_generation_settings_version(body)", text)
        self.assertIn("_freeze_reviewed_h3_runtime_geometry(body, preserve_reviewed_h3_plan)", text)

        worker = next(
            item for item in ast.parse(source).body
            if isinstance(item, ast.FunctionDef) and item.name == "_run_generation"
        )
        worker_text = ast.get_source_segment(source, worker)
        self.assertLess(
            worker_text.index('raw_params.pop(\n                "_reviewed_h3_runtime_geometry"'),
            worker_text.index("manifest = []"),
        )
        self.assertLess(
            worker_text.index('raw_params.pop(\n                "_reviewed_h3_runtime_geometry"'),
            worker_text.index("wgp._parse_task_manifest(manifest, state, os.getcwd())"),
        )
        self.assertIn(
            "_reviewed_h3_runtime_geometry_error(\n                    reviewed_runtime_geometry",
            worker_text,
        )


if __name__ == "__main__":
    unittest.main()
