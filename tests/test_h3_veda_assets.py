"""CPU-only regressions for lazy, verified Veda assets and packed geometry."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3 import veda_assets as assets


class TestVedaAssets(unittest.TestCase):
    def setUp(self):
        self.payload = b"tiny test predictor"
        self.spec = {"filename": "predictor.safetensors", "size": len(self.payload),
                     "sha256": hashlib.sha256(self.payload).hexdigest()}

    def test_predictors_are_pinned_and_distinct_by_workflow_and_precision(self):
        specs = [assets.predictor_spec(ref, quant) for ref in (False, True) for quant in (False, True)]
        self.assertEqual(len({spec["filename"] for spec in specs}), 4)
        for spec in specs:
            self.assertIn(assets.VEDA_REVISION, spec["url"])
            self.assertEqual(len(spec["sha256"]), 64)

    def test_same_size_corruption_invalidates_verification_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / self.spec["filename"]
            path.write_bytes(self.payload)
            assets.verify_predictor(str(path), self.spec)
            path.write_bytes(b"x" * len(self.payload))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                assets.verify_predictor(str(path), self.spec)

    def test_valid_linked_predictor_skips_download(self):
        from shared.utils import files_locator as fl
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimax_h3" / "minimax_h3" / self.spec["filename"]
            path.parent.mkdir(parents=True)
            path.write_bytes(self.payload)
            with (mock.patch.object(assets, "predictor_spec", return_value=self.spec),
                  mock.patch.object(fl, "get_checkpoints_paths", return_value=[tmp]),
                  mock.patch("services.managed_preprocessors._download_with_resume") as download):
                self.assertEqual(assets.ensure_predictor(True, True), str(path.resolve()))
            download.assert_not_called()

    def test_corrupt_linked_file_is_preserved_and_download_goes_to_primary(self):
        from shared.utils import files_locator as fl
        with tempfile.TemporaryDirectory() as tmp:
            linked = Path(tmp) / "linked" / "minimax_h3" / self.spec["filename"]
            linked.parent.mkdir(parents=True)
            linked.write_bytes(b"bad linked asset")
            primary = Path(tmp) / "primary" / "minimax_h3" / self.spec["filename"]
            def download(spec, destination, progress):
                self.assertEqual(spec, self.spec)
                self.assertEqual(Path(destination), primary)
                primary.parent.mkdir(parents=True)
                primary.write_bytes(self.payload)
                return destination
            with (mock.patch.object(assets, "predictor_spec", return_value=self.spec),
                  mock.patch.object(fl, "get_checkpoints_paths", return_value=[str(linked.parents[1])]),
                  mock.patch.object(fl, "get_download_location", return_value=str(primary)),
                  mock.patch("services.managed_preprocessors._download_with_resume", side_effect=download) as fetch):
                self.assertEqual(assets.ensure_predictor(False, True), str(primary))
            fetch.assert_called_once()
            self.assertEqual(linked.read_bytes(), b"bad linked asset")

    def test_mixed_reference_geometry_includes_continuation_and_excludes_audio(self):
        refs = [SimpleNamespace(kind="video", num_latent_frames=3, latent_height=8, latent_width=12),
                SimpleNamespace(kind="image", num_latent_frames=1, latent_height=4, latent_width=8),
                SimpleNamespace(kind="audio")]
        layout = SimpleNamespace(num_condition_video_rows=2 * 4 * 6 + 3 * 4 * 6 + 1 * 2 * 4)
        grid, payload = assets.geometry_payload(layout, (10, 8, 12), (2, 2), [(0, 2)], refs)
        self.assertEqual(grid, (10, 4, 6))
        self.assertEqual(payload["keyframes"], [{"latent_frame_count": 2}])
        self.assertEqual([item["kind"] for item in payload["refs"]], ["video", "image"])
        layout.num_condition_video_rows += 1
        with self.assertRaisesRegex(ValueError, "conditioning rows"):
            assets.geometry_payload(layout, (10, 8, 12), (2, 2), [(0, 2)], refs)


if __name__ == "__main__":
    unittest.main()
