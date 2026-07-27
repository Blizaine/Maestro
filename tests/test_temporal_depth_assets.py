"""Model-free regressions for LTX temporal-depth preprocessor provisioning."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
_SERVICE_PATH = _APP / "services" / "managed_preprocessors.py"
_WGP_PATH = _APP / "wgp.py"
_LAUNCH_PATH = _APP / "launch.py"
_GITIGNORE_PATH = _ROOT / ".gitignore"
_VENDOR_ROOT = _APP / "preprocessing" / "video_depth_anything"


def _load_service():
    import sys

    app_path = str(_APP)
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
    spec = importlib.util.spec_from_file_location(
        "managed_preprocessors_test", _SERVICE_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load managed_preprocessors")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload: bytes, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1024 * 1024):
        for offset in range(0, len(self.payload), max(1, chunk_size)):
            yield self.payload[offset:offset + chunk_size]

    def close(self):
        self.closed = True


class TestTemporalDepthAssetRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_service()

    def test_official_variants_are_pinned_and_sized(self):
        base = self.module.VIDEO_DEPTH_CHECKPOINTS["vitb"]
        large = self.module.VIDEO_DEPTH_CHECKPOINTS["vitl"]

        self.assertEqual(base["size"], 458_247_082)
        self.assertEqual(
            base["sha256"],
            "775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b",
        )
        self.assertEqual(large["size"], 1_538_392_012)
        self.assertEqual(
            large["sha256"],
            "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa",
        )
        for checkpoint in (base, large):
            self.assertEqual(len(checkpoint["revision"]), 40)
            self.assertEqual(checkpoint["license"], "CC-BY-NC-4.0")

    def test_temporal_depth_detection_handles_ui_and_named_values(self):
        uses = self.module.uses_temporal_depth
        for value in ("TVG", "PTVG", "TEVG", "depth_temporal"):
            with self.subTest(value=value):
                self.assertTrue(uses({"video_prompt_type": value}))
        self.assertTrue(uses({"video_prompt_type": ["P", "T", "V", "G"]}))
        for value in ("", "PVG", "DVG", None):
            with self.subTest(value=value):
                self.assertFalse(uses({"video_prompt_type": value}))
        self.assertFalse(uses(None))

    def test_invalid_variant_fails_before_network_access(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported.*variant"):
            self.module.ensure_video_depth_checkpoint("vitg")


class TestTemporalDepthDownloadSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_service()

    @staticmethod
    def _spec(payload: bytes) -> dict:
        return {
            "label": "Fixture Depth",
            "repo_id": "example/depth",
            "revision": "a" * 40,
            "filename": "fixture.pth",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "license": "test-only",
        }

    def test_download_is_verified_and_atomically_published(self):
        payload = (b"temporal-depth-fixture-" * 50_000) + b"done"
        spec = self._spec(payload)
        response = _FakeResponse(payload)
        progress = []

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "depth", "fixture.pth")
            with mock.patch.object(
                self.module.requests, "get", return_value=response,
            ) as request:
                result = self.module._download_with_resume(
                    spec, target, progress.append,
                )

            self.assertEqual(result, target)
            self.assertEqual(Path(target).read_bytes(), payload)
            self.assertFalse(os.path.exists(target + ".part"))
            self.assertTrue(response.closed)
            self.assertTrue(any("Downloading Fixture Depth" in item for item in progress))
            self.assertTrue(any("Verifying Fixture Depth" in item for item in progress))
            request.assert_called_once()
            requested_url = request.call_args.args[0]
            self.assertIn(spec["revision"], requested_url)
            self.assertEqual(request.call_args.kwargs["headers"], {})

    def test_partial_download_resumes_with_range_header(self):
        payload = b"resume-this-checkpoint-safely"
        split = 9
        spec = self._spec(payload)
        response = _FakeResponse(payload[split:], status_code=206)

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "depth", "fixture.pth")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            Path(target + ".part").write_bytes(payload[:split])
            with mock.patch.object(
                self.module.requests, "get", return_value=response,
            ) as request:
                self.module._download_with_resume(spec, target)

            self.assertEqual(Path(target).read_bytes(), payload)
            self.assertEqual(
                request.call_args.kwargs["headers"],
                {"Range": f"bytes={split}-"},
            )

    def test_hash_mismatch_never_publishes_checkpoint(self):
        payload = b"expected checkpoint"
        corrupted = b"corrupt! checkpoint"
        self.assertEqual(len(payload), len(corrupted))
        spec = self._spec(payload)

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "depth", "fixture.pth")
            with mock.patch.object(
                self.module.requests,
                "get",
                return_value=_FakeResponse(corrupted),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    self.module._download_with_resume(spec, target)

            self.assertFalse(os.path.exists(target))
            self.assertFalse(os.path.exists(target + ".part"))


class TestTemporalDepthWiring(unittest.TestCase):
    def test_preflight_runs_before_managed_loras_and_model_generation(self):
        launch = _LAUNCH_PATH.read_text(encoding="utf-8")
        preflight = launch.index("ensure_video_depth_checkpoint(")
        managed_lora = launch.index(
            "_ensure_managed_loras_present(", preflight,
        )
        generation = launch.index("wgp.generate_video(", managed_lora)
        self.assertLess(preflight, managed_lora)
        self.assertLess(managed_lora, generation)

    def test_non_rest_paths_keep_a_late_provisioning_fallback(self):
        wgp = _WGP_PATH.read_text(encoding="utf-8")
        temporal_branch = wgp.index('elif process_type=="depth_temporal":')
        next_branch = wgp.index('elif process_type=="gray":', temporal_branch)
        segment = wgp[temporal_branch:next_branch]
        self.assertIn("ensure_video_depth_checkpoint(variant)", segment)
        self.assertNotIn(
            'fl.locate_file(f"depth/video_depth_anything_', segment,
        )

    def test_official_inference_source_and_license_are_vendored(self):
        expected = (
            "LICENSE",
            "VENDOR.md",
            "utils/util.py",
            "video_depth_anything/video_depth.py",
            "video_depth_anything/dpt_temporal.py",
            "video_depth_anything/motion_module/attention.py",
            "video_depth_anything/motion_module/motion_module.py",
        )
        for relative in expected:
            with self.subTest(relative=relative):
                self.assertTrue((_VENDOR_ROOT / relative).is_file())

        notice = (_VENDOR_ROOT / "VENDOR.md").read_text(encoding="utf-8")
        self.assertIn(
            "4f5ae23172ba60fd7bc11ef671cca678842c7072", notice,
        )
        gitignore = _GITIGNORE_PATH.read_text(encoding="utf-8")
        self.assertNotIn(
            "app/preprocessing/video_depth_anything/", gitignore,
        )


if __name__ == "__main__":
    unittest.main()
