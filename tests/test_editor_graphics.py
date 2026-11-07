"""Contract tests for editable native Editor graphics."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO

from PIL import Image


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.editor_graphics import (  # noqa: E402
    EditorGraphicError,
    normalize_editor_graphic,
    render_editor_graphic,
)
from services.editor_projects import (  # noqa: E402
    EditorProjectError,
    compile_editor_render,
    create_editor_project,
    normalize_editor_project,
)


def _graphic(*, fill: str = "#e02020") -> dict:
    return {
        "version": 1,
        "width": 64,
        "height": 64,
        "elements": [
            {
                "id": "panel",
                "type": "rect",
                "x": 8,
                "y": 8,
                "width": 48,
                "height": 48,
                "fill": fill,
                "opacity": 0.8,
                "radius": 4,
            },
            {
                "id": "headline",
                "type": "text",
                "x": 10,
                "y": 20,
                "width": 44,
                "height": 26,
                "text": "EDITABLE",
                "font_family": "Arial",
                "font_size": 10,
                "color": "#ffffff",
                "align": "center",
                "bold": True,
            },
        ],
    }


class TestEditorGraphics(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.save_root = os.path.join(self.temp.name, "outputs")
        self.uploads_root = os.path.join(self.temp.name, "uploads")
        os.makedirs(self.save_root)
        os.makedirs(self.uploads_root)

    def tearDown(self):
        self.temp.cleanup()

    def test_normalization_is_strict_and_bounded(self):
        graphic = _graphic()
        normalized = normalize_editor_graphic(graphic)
        self.assertEqual(normalized["width"], 64)
        self.assertEqual(normalized["elements"][0]["opacity"], 0.8)
        self.assertEqual(normalized["elements"][1]["align"], "center")

        unsafe = {**graphic, "markup": "<svg onload=alert(1)>"}
        with self.assertRaisesRegex(EditorGraphicError, "unsupported field 'markup'"):
            normalize_editor_graphic(unsafe)
        with self.assertRaisesRegex(EditorGraphicError, "unsupported field 'path'"):
            normalize_editor_graphic({
                **graphic,
                "elements": [
                    {**graphic["elements"][0], "path": "C:/private.png"},
                ],
            })
        with self.assertRaisesRegex(EditorGraphicError, "at most 40"):
            normalize_editor_graphic({
                **graphic,
                "elements": [
                    {**graphic["elements"][0], "id": f"rect-{index}"}
                    for index in range(41)
                ],
            })
        with self.assertRaisesRegex(EditorGraphicError, "opacity must be between"):
            normalize_editor_graphic({
                **graphic,
                "elements": [{**graphic["elements"][0], "opacity": 2}],
            })

    def test_graphic_spec_round_trips_as_an_editable_text_item(self):
        project = create_editor_project(name="Editable title", width=64, height=64, fps=2)
        project["tracks"][2]["z_index"] = 0
        project["tracks"][2]["items"] = [{
            "id": "title-group",
            "text": "Accessible title label",
            "start": 0,
            "duration": 1,
            "graphic": _graphic(),
        }]

        normalized = normalize_editor_project(project)
        round_trip = normalize_editor_project(normalized)
        item = round_trip["tracks"][2]["items"][0]
        self.assertEqual(item["text"], "Accessible title label")
        self.assertEqual(item["graphic"], normalize_editor_graphic(_graphic()))
        self.assertEqual(round_trip["tracks"][2]["items"][0]["graphic"], item["graphic"])

    def test_render_preserves_panel_alpha_and_design_coordinate_scaling(self):
        graphic = {
            "version": 1,
            "width": 100,
            "height": 50,
            "elements": [{
                "id": "panel",
                "type": "rect",
                "x": 10,
                "y": 10,
                "width": 40,
                "height": 20,
                "fill": "#204060",
                "opacity": 0.5,
            }],
        }
        png = render_editor_graphic(graphic, width=200, height=100)
        with Image.open(BytesIO(png)) as image:
            rgba = image.convert("RGBA")
            self.assertEqual(rgba.size, (200, 100))
            self.assertEqual(rgba.getpixel((30, 30)), (32, 64, 96, 128))
            self.assertEqual(rgba.getpixel((5, 5)), (0, 0, 0, 0))

    def test_contain_scaling_preserves_shapes_on_square_canvas(self):
        graphic = {
            "version": 1,
            "width": 1344,
            "height": 768,
            "elements": [{
                "id": "panel",
                "type": "rect",
                "x": 0,
                "y": 0,
                "width": 1344,
                "height": 768,
                "fill": "#204060",
            }],
        }
        with Image.open(BytesIO(render_editor_graphic(graphic, width=64, height=64))) as rendered:
            rgba = rendered.convert("RGBA")
            self.assertEqual(rgba.getpixel((32, 5)), (0, 0, 0, 0))
            self.assertEqual(rgba.getpixel((32, 32)), (32, 64, 96, 255))
            self.assertEqual(rgba.getpixel((32, 58)), (0, 0, 0, 0))

    def test_overlapping_translucent_elements_use_source_over_compositing(self):
        graphic = {
            "version": 1,
            "width": 64,
            "height": 64,
            "elements": [
                {
                    "id": "red",
                    "type": "rect",
                    "x": 10,
                    "y": 10,
                    "width": 30,
                    "height": 30,
                    "fill": "#ff0000",
                    "opacity": 0.5,
                },
                {
                    "id": "blue",
                    "type": "rect",
                    "x": 20,
                    "y": 20,
                    "width": 30,
                    "height": 30,
                    "fill": "#0000ff",
                    "opacity": 0.5,
                },
            ],
        }
        with Image.open(BytesIO(render_editor_graphic(graphic, width=64, height=64))) as rendered:
            self.assertEqual(
                rendered.convert("RGBA").getpixel((25, 25)),
                (85, 0, 170, 192),
            )

    def test_image_layers_require_project_asset_authorization_and_resolver(self):
        graphic = {
            "version": 1,
            "width": 64,
            "height": 64,
            "elements": [{
                "id": "logo",
                "type": "image",
                "x": 0,
                "y": 0,
                "width": 64,
                "height": 64,
                "asset_id": "brand-logo",
                "opacity": 1,
                "fit": "contain",
            }],
        }
        with self.assertRaisesRegex(EditorGraphicError, "missing image asset"):
            normalize_editor_graphic(graphic, assets={})
        with self.assertRaisesRegex(EditorGraphicError, "must be an image"):
            normalize_editor_graphic(
                graphic,
                assets={"brand-logo": {"id": "brand-logo", "type": "video"}},
            )

        image = Image.new("RGBA", (4, 4), (20, 200, 60, 255))
        seen: list[str] = []
        png = render_editor_graphic(
            normalize_editor_graphic(
                graphic,
                assets={"brand-logo": {"id": "brand-logo", "type": "image"}},
            ),
            width=64,
            height=64,
            resolve_image=lambda asset_id: seen.append(asset_id) or image,
        )
        self.assertEqual(seen, ["brand-logo"])
        with Image.open(BytesIO(png)) as rendered:
            self.assertEqual(rendered.convert("RGBA").getpixel((32, 32)), (20, 200, 60, 255))
        with self.assertRaisesRegex(EditorGraphicError, "authorized image resolver"):
            render_editor_graphic(graphic, width=64, height=64)

    def test_text_wraps_and_reports_overflow_and_unsupported_glyphs(self):
        wrapping = {
            "version": 1,
            "width": 128,
            "height": 64,
            "elements": [{
                "id": "wrap",
                "type": "text",
                "x": 0,
                "y": 0,
                "width": 60,
                "height": 64,
                "text": "Two words wrap",
                "font_family": "Arial",
                "font_size": 16,
                "color": "#ffffff",
            }],
        }
        wrapped_png = render_editor_graphic(wrapping, width=128, height=64)
        with Image.open(BytesIO(wrapped_png)) as rendered:
            self.assertGreater(rendered.convert("RGBA").getchannel("A").getbbox()[3], 16)

        too_small = {
            **wrapping,
            "elements": [{**wrapping["elements"][0], "height": 8}],
        }
        with self.assertRaisesRegex(EditorGraphicError, "does not fit its text box"):
            render_editor_graphic(too_small, width=128, height=64)
        unsupported = {
            **wrapping,
            "elements": [{**wrapping["elements"][0], "text": "\U0010FFFF"}],
        }
        with self.assertRaisesRegex(EditorGraphicError, "unsupported glyph"):
            render_editor_graphic(unsupported, width=128, height=64)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_ffmpeg_export_uses_renderer_png_and_edit_changes_exported_pixels(self):
        ffmpeg = shutil.which("ffmpeg")
        project = create_editor_project(name="Graphic export", width=64, height=64, fps=2)
        project["export"].update({"encoder": "software", "include_audio": False})
        project["tracks"][2]["z_index"] = 0
        project["tracks"][2]["items"] = [{
            "id": "title-group",
            "text": "A title",
            "start": 0,
            "duration": 1,
            "graphic": _graphic(fill="#e02020"),
            "keyframes": {
                "x": [
                    {"time": 0, "value": 0, "easing": "linear"},
                    {"time": 1, "value": 0, "easing": "linear"},
                ],
                "scale": [
                    {"time": 0, "value": 1, "easing": "linear"},
                    {"time": 1, "value": 1, "easing": "linear"},
                ],
                "opacity": [
                    {"time": 0, "value": 1, "easing": "linear"},
                    {"time": 1, "value": 1, "easing": "linear"},
                ],
            },
        }]

        decoded: list[tuple[int, int, int]] = []
        for fill, name in (("#e02020", "red"), ("#2040e0", "blue")):
            project["tracks"][2]["items"][0]["graphic"] = _graphic(fill=fill)
            filter_path = os.path.join(self.temp.name, f"{name}-filter.txt")
            output_path = os.path.join(self.temp.name, f"{name}-export.mp4")
            command, duration = compile_editor_render(
                project,
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
                filter_script_path=filter_path,
                output_path=output_path,
                ffmpeg=ffmpeg,
                encoder_capabilities={
                    "encoders": {
                        "software": True,
                        "nvidia": False,
                        "intel": False,
                        "apple": False,
                    },
                    "recommended": "software",
                },
            )
            self.assertEqual(duration, 1)
            expected_png = render_editor_graphic(
                project["tracks"][2]["items"][0]["graphic"], width=64, height=64
            )
            expected_hash = hashlib.sha256(expected_png).hexdigest()
            artifact_path = os.path.join(
                self.temp.name, f"editor-graphic-{expected_hash}.png"
            )
            with open(artifact_path, "rb") as handle:
                self.assertEqual(handle.read(), expected_png)
            result = subprocess.run(command, capture_output=True, timeout=60)
            self.assertEqual(
                result.returncode,
                0,
                result.stderr.decode("utf-8", errors="replace")[-3000:],
            )
            decoded_frame = subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "0.5",
                    "-i", output_path, "-frames:v", "1", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "pipe:1",
                ],
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(decoded_frame.returncode, 0, decoded_frame.stderr[-1000:])
            offset = (50 * 64 + 50) * 3
            decoded.append(tuple(decoded_frame.stdout[offset:offset + 3]))

        red_pixel, blue_pixel = decoded
        self.assertGreater(red_pixel[0], red_pixel[2] + 80, red_pixel)
        self.assertGreater(blue_pixel[2], blue_pixel[0] + 80, blue_pixel)
        self.assertGreater(blue_pixel[2], red_pixel[2] + 80, (red_pixel, blue_pixel))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_prores4444_export_preserves_native_graphic_alpha(self):
        ffmpeg = shutil.which("ffmpeg")
        project = create_editor_project(
            name="Graphic alpha export", width=64, height=64, fps=2
        )
        project["export"].update({
            "encoder": "software",
            "format": "prores4444",
            "include_audio": False,
        })
        graphic = {
            "version": 1,
            "width": 64,
            "height": 64,
            "elements": [
                {
                    "id": "panel",
                    "type": "rect",
                    "x": 8,
                    "y": 8,
                    "width": 48,
                    "height": 48,
                    "fill": "#204060",
                    "opacity": 0.5,
                },
                {
                    "id": "title",
                    "type": "text",
                    "x": 12,
                    "y": 18,
                    "width": 40,
                    "height": 28,
                    "text": "A",
                    "font_family": "Arial Black",
                    "font_size": 18,
                    "color": "#ffffff",
                    "bold": True,
                    "align": "center",
                },
            ],
        }
        project["tracks"][2]["items"] = [{
            "id": "alpha-title",
            "text": "Alpha title",
            "start": 0,
            "duration": 1,
            "graphic": graphic,
        }]
        expected = Image.open(BytesIO(render_editor_graphic(graphic, width=64, height=64))).convert("RGBA")
        text_pixel = next(
            (
                (x, y)
                for y in range(18, 46)
                for x in range(12, 52)
                if expected.getpixel((x, y))[3] == 255
                and expected.getpixel((x, y))[:3] == (255, 255, 255)
            ),
            None,
        )
        self.assertIsNotNone(text_pixel, "renderer should provide a fully opaque text pixel")

        filter_path = os.path.join(self.temp.name, "alpha-filter.txt")
        output_path = os.path.join(self.temp.name, "graphic-alpha.mov")
        command, duration = compile_editor_render(
            project,
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
            filter_script_path=filter_path,
            output_path=output_path,
            ffmpeg=ffmpeg,
        )
        self.assertEqual(duration, 1)
        exported = subprocess.run(command, capture_output=True, timeout=60)
        self.assertEqual(
            exported.returncode,
            0,
            exported.stderr.decode("utf-8", errors="replace")[-3000:],
        )
        decoded = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                output_path,
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(decoded.returncode, 0, decoded.stderr[-1000:])
        self.assertEqual(len(decoded.stdout), 64 * 64 * 4)

        def pixel_at(x: int, y: int) -> tuple[int, int, int, int]:
            offset = (y * 64 + x) * 4
            return tuple(decoded.stdout[offset:offset + 4])

        self.assertEqual(pixel_at(0, 0)[3], 0)
        panel_pixel = pixel_at(30, 10)
        self.assertGreater(panel_pixel[3], 110, panel_pixel)
        self.assertLess(panel_pixel[3], 150, panel_pixel)
        opaque_text_pixel = pixel_at(*text_pixel)
        self.assertGreater(opaque_text_pixel[3], 230, opaque_text_pixel)
        self.assertGreater(min(opaque_text_pixel[:3]), 200, opaque_text_pixel)


if __name__ == "__main__":
    unittest.main()
