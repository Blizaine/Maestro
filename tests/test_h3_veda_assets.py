"""CPU-only regressions for lazy, verified Veda assets and packed geometry."""

from __future__ import annotations

import hashlib
import math
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

    def test_frames_geometry_matches_packed_target_without_references(self):
        import torch

        from models.minimax_h3.packing import build_packed_sequence, patchify_video_latents

        patch_size = (1, 2, 2)
        latent_shape = (10, 8, 12)
        text_tags = torch.ones(1, dtype=torch.long)
        layout = build_packed_sequence(text_tags, *latent_shape, 0, patch_size)
        target_rows = patchify_video_latents(
            torch.zeros((1, 1, *latent_shape)), patch_size
        )

        grid, payload = assets.geometry_payload(layout, latent_shape, patch_size)

        self.assertEqual(layout.num_condition_video_rows, 0)
        self.assertEqual(grid, (10, 4, 6))
        self.assertEqual(math.prod(grid), target_rows.shape[0])
        self.assertEqual(payload["keyframes"], [])
        self.assertEqual(payload["refs"], [])

    def test_mixed_reference_geometry_matches_packed_rows_and_excludes_audio(self):
        import torch

        from models.minimax_h3.packing import patchify_video_latents
        from models.minimax_h3.ref2va import (
            MiniMaxH3PreparedReference,
            build_ref2va_packed_sequence,
        )

        patch_size = (1, 2, 2)
        latent_shape = (10, 8, 12)
        anchors = [("history", 2)]
        references = [
            MiniMaxH3PreparedReference(
                kind="video", num_latent_frames=3, latent_height=8,
                latent_width=12, num_audio_latents=2,
            ),
            MiniMaxH3PreparedReference(
                kind="image", num_latent_frames=1, latent_height=4,
                latent_width=8,
            ),
            MiniMaxH3PreparedReference(kind="audio", num_audio_latents=2),
        ]
        audio_anchors = [("history", 2)]
        layout = build_ref2va_packed_sequence(
            torch.ones(1, dtype=torch.long), references, *latent_shape, 4,
            patch_size, keyframe_anchors=anchors,
            audio_condition_anchors=audio_anchors,
        )

        def packed_row_count(shape):
            return patchify_video_latents(
                torch.zeros((1, 1, *shape)), patch_size
            ).shape[0]

        expected_condition_rows = sum(
            packed_row_count(shape)
            for shape in ((2, 8, 12), (3, 8, 12), (1, 4, 8))
        )
        target_rows = packed_row_count(latent_shape)
        grid, payload = assets.geometry_payload(
            layout, latent_shape, patch_size, anchors, references
        )

        self.assertEqual(layout.num_condition_video_rows, expected_condition_rows)
        self.assertGreater(layout.num_condition_audio_rows, 0)
        self.assertEqual(grid, (10, 4, 6))
        self.assertEqual(math.prod(grid), target_rows)
        self.assertEqual(payload["keyframes"], [{"latent_frame_count": 2}])
        self.assertEqual([item["kind"] for item in payload["refs"]], ["video", "image"])

        layout.num_condition_video_rows += 1
        with self.assertRaisesRegex(ValueError, "conditioning rows"):
            assets.geometry_payload(layout, latent_shape, patch_size, anchors, references)

    def test_geometry_rejects_unsupported_patch_sizes(self):
        layout = SimpleNamespace(num_condition_video_rows=0)
        with self.assertRaisesRegex(ValueError, "three-dimensional"):
            assets.geometry_payload(layout, (10, 8, 12), (2, 2))
        with self.assertRaisesRegex(ValueError, "temporal patching"):
            assets.geometry_payload(layout, (10, 8, 12), (2, 2, 2))
        with self.assertRaisesRegex(ValueError, "2x2 spatial"):
            assets.geometry_payload(layout, (10, 8, 12), (1, 4, 2))

    def test_geometry_rejects_invalid_latent_shapes(self):
        layout = SimpleNamespace(num_condition_video_rows=0)
        for latent_shape, message in (
            ((10, 8), "three-dimensional"),
            ((0, 8, 12), "positive"),
            ((10, 7, 12), "not divisible"),
        ):
            with self.subTest(latent_shape=latent_shape):
                with self.assertRaisesRegex(ValueError, message):
                    assets.geometry_payload(layout, latent_shape, (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
