"""Native view recovery, cache isolation and portable PNG selections."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import torch
from PIL import Image
from safetensors.torch import save_file
from services import character_codec as codec, character_library as library
from services import character_views as views, character_transfer as transfer, refmod


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, self.cwd)
        Path("uploads").mkdir()
        self.picture = Path("uploads/portrait.png").resolve()
        Image.new("RGB", (512, 768), (17, 49, 82)).save(self.picture)
        probe = mock.patch.object(library, "_probe_duration", return_value=3.0)
        probe.start()
        self.addCleanup(probe.stop)

    def saved(self):
        return library.create_character(name="Blaine", visual_path=str(self.picture), visual_type="image")

    def imported(self, count=12):
        tensor = torch.arange(24 * count * 4 * 6).float().reshape(1, 24, count, 4, 6) / 10000
        path = Path("source.safetensors")
        save_file({"latent": tensor}, str(path), metadata=refmod.new_refmod_metadata("Ref Person", tensor, source="stack"))
        def old_preview(_, directory, kind):
            target = directory / ("visual.mp4" if kind == "video" else "visual.png")
            target.write_bytes(b"legacy video preview" if kind == "video" else self.picture.read_bytes())
            return target
        card = library.import_character(path, decode_preview=old_preview)
        return card, tensor

    @staticmethod
    def decoder(latent, destination, metadata):
        items = []
        for index in range(latent.shape[2]):
            view_id = f"view-{index + 1:04}"
            Image.new("RGB", (96, 64), (index, 30, 70)).save(destination / f"{view_id}.png")
            items.append({"id": view_id, "latent_index": index})
        return {"source": "refmod", "items": items, "notes": []}

    def test_original_photo_uses_source_pixels_without_gpu_and_preserves_media(self):
        card = self.saved()
        decoder = mock.Mock(side_effect=AssertionError("Original photo must not be reconstructed"))
        result = library.recover_character_images(card["id"], decode_views=decoder)
        recovered = library.get_character_image(card["id"], "view-0001")
        self.assertEqual(Image.open(recovered).tobytes(), Image.open(self.picture).tobytes())
        self.assertEqual(result["image_views"]["source"], "original_image")
        self.assertEqual(Path(result["visual"]["path"]).read_bytes(), self.picture.read_bytes())
        decoder.assert_not_called()

    def test_all_views_cached_without_touching_original_latent_or_preview(self):
        card, _ = self.imported()
        paths = [Path(card["refmod"]["path"]), Path(card["visual"]["path"])]
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        decoder = mock.Mock(side_effect=self.decoder)
        result = library.recover_character_images(card["id"], decode_views=decoder)
        self.assertEqual(len(result["image_views"]["items"]), 12)
        self.assertEqual(result["image_views"]["items"][-1]["width"], 96)
        self.assertNotIn("directory", result["image_views"])
        library.recover_character_images(card["id"], decode_views=decoder)
        self.assertEqual(decoder.call_count, 1, "A completed recovery must be reused")
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])

    def test_original_png_keeps_transparency(self):
        Image.new("RGBA", (64, 64), (31, 73, 99, 123)).save(self.picture)
        card = self.saved()
        library.recover_character_images(card["id"], decode_views=self.decoder)
        with Image.open(library.get_character_image(card["id"], "view-0001")) as picture:
            self.assertEqual(picture.mode, "RGBA")
            self.assertEqual(picture.getpixel((0, 0)), (31, 73, 99, 123))

    def test_original_video_samples_the_whole_source_at_full_resolution(self):
        import cv2
        import numpy as np
        capture = mock.Mock()
        capture.get.side_effect = lambda prop: 100 if prop == cv2.CAP_PROP_FRAME_COUNT else 10
        pixels = np.full((128, 256, 3), (11, 22, 33), dtype=np.uint8)
        capture.read.return_value = (True, pixels)
        destination = Path("video-images")
        destination.mkdir()
        with mock.patch.object(cv2, "VideoCapture", return_value=capture):
            result = views.original_images(Path("source.mp4"), "video", destination, lambda _: None)
        self.assertEqual(len(result["items"]), 32)
        self.assertEqual(result["items"][0]["source_frame"], 0)
        self.assertEqual(result["items"][-1]["source_frame"], 99)
        with Image.open(destination / "view-0032.png") as picture:
            self.assertEqual(picture.size, (256, 128))
            self.assertEqual(picture.getpixel((0, 0)), (33, 22, 11))
        capture.release.assert_called_once()

    def test_cover_updates_cache_url_and_png_thumbnail(self):
        card, _ = self.imported(3)
        old = library.recover_character_images(card["id"], decode_views=self.decoder)
        selected = library.select_character_images(card["id"], ["view-0003", "view-0002"], "view-0003")
        self.assertNotEqual(old["visual"]["thumbnail_url"], selected["visual"]["thumbnail_url"])
        self.assertEqual(library.get_character_media(card["id"], "thumbnail"), library.get_character_image(card["id"], "view-0003"))
        self.assertEqual(library.get_character(card["id"])["image_views"]["selected_ids"], ["view-0003", "view-0002"])
        with zipfile.ZipFile(library.export_character_images(card["id"])) as archive:
            self.assertEqual(archive.namelist(), ["view-0003.png", "view-0002.png"])
            self.assertEqual(archive.read("view-0003.png"), library.get_character_image(card["id"], "view-0003").read_bytes())

    def test_failed_rebuild_keeps_old_selection_and_removes_partial_cache(self):
        card, _ = self.imported(2)
        library.recover_character_images(card["id"], decode_views=self.decoder)
        library.select_character_images(card["id"], ["view-0002"], "view-0002")
        directory = library._character_directory(card["id"])
        before = sorted(path.name for path in directory.iterdir())
        def fail(_, destination, __):
            (destination / "partial.png").write_bytes(b"partial")
            raise RuntimeError("Decode failed")
        with self.assertRaisesRegex(RuntimeError, "Decode failed"):
            library.recover_character_images(card["id"], decode_views=fail, force=True)
        self.assertEqual(before, sorted(path.name for path in directory.iterdir()))
        self.assertEqual(library.get_character(card["id"])["image_views"]["cover_id"], "view-0002")
        library.recover_character_images(card["id"], decode_views=self.decoder, force=True)
        self.assertEqual(library.get_character(card["id"])["image_views"]["selected_ids"], ["view-0002"])

    def test_selection_and_paths_are_constrained_to_recovered_images(self):
        card = self.saved()
        library.recover_character_images(card["id"], decode_views=self.decoder)
        for ids, cover in [([], ""), (["view-0002"], "view-0002"), (["view-0001"] * 2, "view-0001"), (["view-0001"], "../visual")]:
            with self.subTest(ids=ids, cover=cover), self.assertRaises(ValueError):
                library.select_character_images(card["id"], ids, cover)
        for value in ["../visual", "view-0002", "view-0001.png", "/visual"]:
            with self.assertRaises(ValueError): library.get_character_image(card["id"], value)
        stored = library._get_character_record(card["id"])["image_views"]
        with self.assertRaises(ValueError): views.view_path(library._character_directory(card["id"]), {**stored, "directory": "../"}, "view-0001")

    def test_portable_export_carries_selected_pngs_cover_and_unchanged_latent(self):
        card, tensor = self.imported(3)
        voice = Path("uploads/voice.wav").resolve()
        voice.write_bytes(b"original saved voice bytes")
        library.attach_character_voice(card["id"], str(voice))
        library.recover_character_images(card["id"], decode_views=self.decoder)
        library.select_character_images(card["id"], ["view-0003", "view-0001"], "view-0003")
        expected = library.get_character_image(card["id"], "view-0003").read_bytes()
        path = library.export_character(card["id"])
        info, tensors = refmod.load_refmod(path)
        self.assertTrue(torch.equal(tensors["latent"], tensor))
        self.assertEqual(len(info["character"]["image_views"]["items"]), 2)
        self.assertTrue(library.get_character(card["id"])["refmod"]["preview_generated"])
        imported = library.import_character(path, decode_preview=mock.Mock(side_effect=AssertionError("No GPU for embedded views")))
        self.assertEqual(imported["image_views"]["cover_id"], "view-0003")
        self.assertEqual(library.get_character_image(imported["id"], "view-0003").read_bytes(), expected)
        self.assertTrue(imported["refmod"]["preview_generated"])
        self.assertEqual(library.get_character_media(imported["id"], "voice").read_bytes(), voice.read_bytes())

    def test_bad_embedded_view_id_is_rejected_before_media_write(self):
        card = self.saved()
        library.recover_character_images(card["id"], decode_views=self.decoder)
        path = library.export_character(card["id"], encode_visual=lambda *_: torch.zeros(1, 24, 1, 4, 4))
        info, tensors = refmod.load_refmod(path)
        info["character"]["image_views"]["items"][0]["id"] = "../../escaped"
        info["metadata"][refmod.CHARACTER_META] = json.dumps(info["character"])
        save_file(tensors, "bad.safetensors", metadata=info["metadata"])
        with self.assertRaisesRegex(ValueError, "character image"):
            refmod.inspect_refmod("bad.safetensors")

    def test_native_decoder_preserves_grid_and_decodes_more_than_eight_views(self):
        class FakeVae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.inputs = []
            def _decode_clip(self, tensor):
                self.inputs.append(tuple(tensor.shape))
                return torch.zeros(1, 3, 1, tensor.shape[-2] * 16, tensor.shape[-1] * 16)
        vae = FakeVae()
        @contextmanager
        def fake_codec(): yield vae
        for count, height, width in [(12, 4, 6), (1, 70, 80)]:
            tensor = torch.zeros(1, 24, count, height, width)
            before = tensor.clone()
            destination = Path(f"decode-{count}")
            destination.mkdir()
            with mock.patch.object(codec, "video_codec", fake_codec):
                result = codec.recover_refmod_images(tensor, destination, {"mode": "encode", "source_shape": " +".join([f"1x{height}x{width}"] * count)})
            self.assertEqual(len(result["items"]), count)
            self.assertEqual(vae.inputs[-1], (1, 24, 1, height, width))
            with Image.open(destination / "view-0001.png") as picture:
                self.assertEqual(picture.size, (width * 16, height * 16))
            self.assertTrue(torch.equal(tensor, before))
            self.assertEqual(result["notes"], [])

    def test_recovery_routes_download_pixels_and_reject_invalid_selection(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        card = self.saved()
        app = FastAPI()
        guard = mock.Mock(side_effect=AssertionError("Original image must not use GPU"))
        app.include_router(transfer.build_router(guard))
        with TestClient(app) as client:
            job_id = client.post(f"/api/v1/characters/{card['id']}/images/recover").json()["id"]
            for _ in range(200):
                job = client.get(f"/api/v1/character-transfers/{job_id}").json()
                if job["status"] != "running": break
                time.sleep(0.01)
            self.assertEqual(job["status"], "completed", job)
            result = job["result"]["character"]
            image = client.get(result["image_views"]["items"][0]["url"])
            self.assertEqual(image.headers["content-type"], "image/png")
            self.assertEqual(Image.open(io.BytesIO(image.content)).tobytes(), Image.open(self.picture).tobytes())
            bad = client.put(f"/api/v1/characters/{card['id']}/images", json={"selected_ids": [], "cover_id": ""})
            self.assertEqual(bad.status_code, 400)
            download = client.get(f"/api/v1/characters/{card['id']}/images.zip")
            self.assertEqual(download.status_code, 200)
            self.assertIn("blaine.images.zip", download.headers["content-disposition"])
        guard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
