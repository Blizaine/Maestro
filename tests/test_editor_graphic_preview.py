"""Preview/export renderer parity and existing media-boundary regressions."""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.editor_graphic_preview import build_editor_graphic_preview


class GraphicPreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.outputs = self.root / "outputs"
        self.uploads = self.root / "uploads"
        self.outputs.mkdir(); self.uploads.mkdir()
        self.graphic = {"version": 1, "width": 640, "height": 360, "elements": [
            {"id": "panel", "type": "rect", "x": 20, "y": 200, "width": 380, "height": 100, "fill": "#18242c", "opacity": .75, "radius": 12},
            {"id": "headline", "type": "text", "x": 36, "y": 215, "width": 350, "height": 60, "text": "Editable title", "font_family": "Arial", "font_size": 24, "color": "#ffffff"},
        ]}

    def tearDown(self):
        self.tmp.cleanup()

    def render(self, **values):
        return build_editor_graphic_preview({"graphic": self.graphic, "width": 640, "height": 360, **values}, save_root=str(self.outputs), uploads_root=str(self.uploads), workspace="default")

    def test_same_renderer_as_export_and_edit_changes_pixels(self):
        from services.editor_graphics import render_editor_graphic
        before = self.render()
        self.assertEqual(before, render_editor_graphic(self.graphic, width=640, height=360))
        self.graphic["elements"][1]["text"] = "A changed title"
        self.assertNotEqual(before, self.render())
        image = Image.open(io.BytesIO(before))
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.getpixel((0, 0))[3], 0)
        self.assertGreater(image.getpixel((24, 240))[3], 0)
        self.assertLess(image.getpixel((24, 240))[3], 255)

    def test_bounded_preview_keeps_canvas_aspect(self):
        image = Image.open(io.BytesIO(self.render(width=3840, height=2160)))
        self.assertEqual(image.size, (1280, 720))

    def test_resolves_only_images_inside_existing_media_boundary(self):
        path = self.uploads / "logo.png"
        Image.new("RGBA", (32, 32), "red").save(path)
        self.graphic["elements"] = [{"id": "logo", "type": "image", "x": 0, "y": 0, "width": 32, "height": 32, "asset_id": "logo"}]
        good = {"logo": {"id": "logo", "type": "image", "origin": "upload", "path": str(path)}}
        image = Image.open(io.BytesIO(self.render(assets=good)))
        self.assertEqual(image.getpixel((10, 10)), (255, 0, 0, 255))
        outside = self.root / "private.png"
        Image.new("RGBA", (32, 32), "blue").save(outside)
        with self.assertRaises(ValueError):
            self.render(assets={"logo": {**good["logo"], "path": str(outside)}})
        with self.assertRaises(ValueError):
            self.render(assets={})

    def test_changed_source_invalidates_cached_preview(self):
        path = self.uploads / "logo.png"
        Image.new("RGBA", (32, 32), "red").save(path)
        self.graphic["elements"] = [{"id": "logo", "type": "image", "x": 0, "y": 0, "width": 32, "height": 32, "asset_id": "logo"}]
        assets = {"logo": {"id": "logo", "type": "image", "origin": "upload", "path": str(path)}}
        first = self.render(assets=assets)
        stamp = path.stat().st_mtime_ns
        Image.new("RGBA", (32, 32), "blue").save(path)
        os.utime(path, ns=(stamp + 1000000000, stamp + 1000000000))
        self.assertNotEqual(first, self.render(assets=assets))

    def test_unknown_fields_and_invalid_dimensions_fail(self):
        for values in ({"width": True}, {"height": float("nan")}, {"width": 100000}, {"raw_html": "<script>bad</script>"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.render(**values)


if __name__ == "__main__":
    unittest.main()
