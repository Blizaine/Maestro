"""CPU regressions for transparent ProRes 4444 Editor exports."""

from __future__ import annotations

import copy
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest.mock import patch


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.editor_projects import (  # noqa: E402
    EditorProjectError,
    compile_editor_render,
    create_editor_project,
    normalize_editor_project,
    prepare_editor_overlay_project,
    render_editor_project,
)


class TestEditorAlphaExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.save_root = os.path.join(self.temp.name, "outputs")
        self.uploads_root = os.path.join(self.temp.name, "uploads")
        os.makedirs(self.save_root)
        os.makedirs(self.uploads_root)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write_rgba_png(path: str, width: int, height: int, pixels: list[tuple[int, int, int, int]]) -> None:
        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        rows = b"".join(
            b"\x00" + bytes(channel for pixel in pixels[y * width:(y + 1) * width] for channel in pixel)
            for y in range(height)
        )
        data = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b"")
        )
        with open(path, "wb") as handle:
            handle.write(data)

    def _project_with_image(self, pixels, *, key_color="#00ff00"):
        source = os.path.join(self.save_root, "overlay-source.png")
        self._write_rgba_png(source, 64, 64, pixels)
        project = create_editor_project(name="Transparent title overlay")
        project["canvas"].update({"width": 64, "height": 64, "fps": 2})
        project["export"].update({
            "format": "prores4444",
            "encoder": "software",
            "include_audio": False,
            "filename": "Transparent title overlay",
        })
        project["assets"] = {
            "image": {
                "id": "image",
                "name": os.path.basename(source),
                "path": source,
                "type": "image",
                "origin": "output",
                "duration": 2,
                "width": 64,
                "height": 64,
                "has_audio": False,
            }
        }
        project["tracks"][0]["items"] = [{
            "id": "keyed-visual",
            "asset_id": "image",
            "start": 0,
            "duration": 2,
            "transform": {"x": 2, "y": 0, "scale": 1, "rotation": 0},
            "opacity": 0.75,
            "fade_in": 0.5,
            "fade_out": 0.5,
            "keying": {
                "enabled": True,
                "color": key_color,
                "similarity": 0.15,
                "softness": 0.35,
                "spill": 0.6,
            },
            # Alpha is produced by keying before color; this exposure makes
            # the regression catch accidental keying after color correction.
            "color": {"exposure": 1.0},
        }]
        project["tracks"][2]["items"] = [{
            "id": "title",
            "text": "TITLE",
            "start": 0,
            "duration": 2,
            "style": {"font_size": 10, "color": "#ffffff", "y": -20},
            "opacity": 0.7,
            "fade_in": 0.5,
        }]
        return project

    def _render(self, project, name="overlay.mov"):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("FFmpeg is not installed")
        with patch(
            "services.editor_projects.editor_export_capabilities",
            return_value={
                "encoders": {
                    "software": True,
                    "nvidia": False,
                    "intel": False,
                    "apple": False,
                },
                "recommended": "software",
                "prores4444": True,
            },
        ):
            output = render_editor_project(
                project,
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
                ffmpeg=ffmpeg,
                progress_callback=None,
            )
        self.assertTrue(os.path.isfile(output), output)
        self.assertEqual(os.path.splitext(output)[1].lower(), ".mov")
        return ffmpeg, output

    def _decode_rgba_frames(self, ffmpeg: str, path: str, *, width=64, height=64):
        decoded = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-i", path,
                "-vsync", "0",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(
            decoded.returncode,
            0,
            decoded.stderr.decode("utf-8", errors="replace")[-2000:],
        )
        frame_size = width * height * 4
        self.assertEqual(len(decoded.stdout) % frame_size, 0)
        frames = [
            decoded.stdout[offset:offset + frame_size]
            for offset in range(0, len(decoded.stdout), frame_size)
        ]
        self.assertGreaterEqual(len(frames), 3, "expected multiple decoded frames")
        return frames

    def _decode_rgb_frames(self, ffmpeg: str, path: str, *, width=64, height=64):
        decoded = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel", "error",
                "-i", path,
                "-vsync", "0",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(
            decoded.returncode,
            0,
            decoded.stderr.decode("utf-8", errors="replace")[-2000:],
        )
        frame_size = width * height * 3
        self.assertEqual(len(decoded.stdout) % frame_size, 0)
        return [
            decoded.stdout[offset:offset + frame_size]
            for offset in range(0, len(decoded.stdout), frame_size)
        ]

    @staticmethod
    def _pixel(frame: bytes, x: int, y: int, width: int = 64):
        offset = (y * width + x) * 4
        return tuple(frame[offset:offset + 4])

    def test_format_validation_and_mp4_default_preserve_legacy_path(self):
        project = create_editor_project()
        legacy = copy.deepcopy(project)
        legacy["export"].pop("format")
        self.assertEqual(normalize_editor_project(legacy)["export"]["format"], "mp4")

        project["export"]["format"] = "webm"
        with self.assertRaisesRegex(EditorProjectError, "export format"):
            normalize_editor_project(project)

    def test_alpha_format_rejects_upscale_and_grain(self):
        project = create_editor_project()
        project["export"].update({
            "format": "prores4444",
            "spatial_upsampling": "flashvsr2",
        })
        with self.assertRaisesRegex(EditorProjectError, "does not support"):
            normalize_editor_project(project)

        project["export"].update({
            "spatial_upsampling": "",
            "film_grain_intensity": 0.01,
        })
        with self.assertRaisesRegex(EditorProjectError, "does not support"):
            normalize_editor_project(project)

    def test_overlay_project_is_a_pure_exact_visual_selection(self):
        project = create_editor_project(name="Keep original")
        project["canvas"].update({"width": 640, "height": 360, "fps": 24})
        project["assets"] = {
            "visual": {
                "id": "visual", "name": "visual.mp4", "type": "video",
                "origin": "output", "duration": 20, "has_audio": True,
            },
            "audio": {"id": "audio", "name": "voice.wav", "type": "audio", "origin": "upload"},
        }
        project["tracks"][0]["items"] = [
            {
                "id": "selected-video", "asset_id": "visual", "start": 4, "duration": 3,
                "transform": {"x": 8, "y": -4, "scale": 1.2, "rotation": 12},
                "opacity": 0.7, "fade_in": 0.5, "fade_out": 0.25,
                "keying": {"enabled": True}, "color": {"exposure": 0.25},
                "keyframes": {"opacity": [
                    {"time": 0, "value": 0.3, "easing": "linear"},
                    {"time": 3, "value": 0.7, "easing": "linear"},
                ]},
            },
            {"id": "unselected-video", "asset_id": "visual", "start": 8, "duration": 2},
        ]
        project["tracks"][1]["items"] = [{
            "id": "audio-item", "asset_id": "audio", "start": 4, "duration": 1,
        }]
        project["tracks"][2]["items"] = [{
            "id": "title", "text": "Animated title", "start": 4.5, "duration": 2,
            "opacity": 0.8, "fade_in": 0.2,
            "keyframes": {"opacity": [
                {"time": 0, "value": 0.4, "easing": "linear"},
                {"time": 2, "value": 0.8, "easing": "linear"},
            ]},
        }]
        original = copy.deepcopy(project)

        overlay = prepare_editor_overlay_project(
            project, ["selected-video", "title"]
        )

        self.assertEqual(project, original)
        self.assertEqual(overlay["canvas"], normalize_editor_project(original)["canvas"])
        self.assertEqual(overlay["export"]["format"], "prores4444")
        self.assertTrue(overlay["export"]["include_audio"])
        self.assertTrue(overlay["assets"]["visual"]["has_audio"])
        self.assertEqual([track["type"] for track in overlay["tracks"]], ["video", "text"])
        self.assertEqual([item["id"] for item in overlay["tracks"][0]["items"]], ["selected-video"])
        self.assertEqual(overlay["tracks"][0]["items"][0]["start"], 0)
        self.assertEqual(overlay["tracks"][0]["items"][0]["transform"]["rotation"], 12)
        self.assertEqual(overlay["tracks"][0]["items"][0]["keying"]["enabled"], True)
        selected_item = overlay["tracks"][0]["items"][0]
        self.assertEqual(selected_item["opacity"], 0.7)
        self.assertEqual((selected_item["fade_in"], selected_item["fade_out"]), (0.5, 0.25))
        self.assertEqual(selected_item["color"]["exposure"], 0.25)
        self.assertEqual(selected_item["keyframes"]["opacity"], [
            {"time": 0.0, "value": 0.3, "easing": "linear"},
            {"time": 3.0, "value": 0.7, "easing": "linear"},
        ])
        self.assertEqual(overlay["tracks"][2 - 1]["items"][0]["start"], 0.5)
        self.assertEqual(set(overlay["assets"]), {"visual"})

        for invalid in ([], ["missing"], ["selected-video", "selected-video"], ["audio-item"]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(EditorProjectError):
                    prepare_editor_overlay_project(project, invalid)

    def test_cpu_prores_export_keeps_source_alpha_key_matte_text_and_timeline_effects(self):
        pixels = [(0, 255, 0, 255)] * (64 * 64)
        for y in range(64):
            for x, rgba in (
                (12, (255, 0, 0, 255)),
                (16, (255, 0, 0, 128)),
                (20, (0, 150, 0, 255)),
            ):
                pixels[y * 64 + x] = rgba
        project = self._project_with_image(pixels)
        ffmpeg, output = self._render(project)
        frames = self._decode_rgba_frames(ffmpeg, output)
        frame = frames[2]  # t=1s: both item fades have reached full strength.

        self.assertEqual(self._pixel(frame, 6, 40)[3], 0)  # keyed green over transparent base
        opaque_red = self._pixel(frame, 14, 40)
        source_alpha_red = self._pixel(frame, 18, 40)
        self.assertTrue(180 <= opaque_red[3] <= 202, opaque_red)
        self.assertTrue(85 <= source_alpha_red[3] <= 108, source_alpha_red)
        edge = self._pixel(frame, 22, 40)
        self.assertTrue(10 <= edge[3] <= 245, edge)
        self.assertLess(edge[1], 200, edge)  # green despill runs before color correction
        self.assertEqual(self._pixel(frames[0], 14, 40)[3], 0)  # fade-in starts transparent
        title_pixels = [
            self._pixel(frame, x, y)
            for y in range(2, 18)
            for x in range(8, 56)
        ]
        self.assertTrue(
            any(r > 130 and g > 130 and b > 130 and alpha > 70 for r, g, b, alpha in title_pixels),
            "the title should render over the transparent video layer",
        )

    def test_cpu_prores_export_despills_blue_screen_and_has_alpha_track(self):
        pixels = [(0, 0, 255, 255)] * (64 * 64)
        for y in range(64):
            pixels[y * 64 + 12] = (0, 0, 255, 255)
            pixels[y * 64 + 16] = (80, 80, 220, 255)
        project = self._project_with_image(pixels, key_color="#0000ff")
        project["tracks"][0]["items"][0].pop("color", None)
        # The overlay text is unrelated to this spill sample, but retained as
        # part of the generated-title workflow.
        ffmpeg, output = self._render(project, "blue-spill.mov")
        probe = subprocess.run(
            [
                shutil.which("ffprobe") or "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,profile,pix_fmt",
                "-of", "default=noprint_wrappers=1",
                output,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr[-1000:])
        self.assertIn("codec_name=prores", probe.stdout)
        self.assertRegex(probe.stdout, r"yuva444p(?:10|12)le")
        frame = self._decode_rgba_frames(ffmpeg, output)[2]
        transparent = self._pixel(frame, 6, 40)
        spill = self._pixel(frame, 18, 40)
        self.assertEqual(transparent[3], 0)
        self.assertGreater(spill[3], 0)
        self.assertLess(spill[2], 200, spill)

    def test_alpha_encoder_is_cpu_prores_with_optional_pcm_audio_and_history_roundtrips(self):
        project = create_editor_project(name="Alpha with PCM")
        project["canvas"].update({"width": 64, "height": 64, "fps": 2})
        project["export"].update({"format": "prores4444", "encoder": "nvidia", "include_audio": True})
        project["tracks"][2]["items"] = [{
            "id": "title", "text": "Title", "start": 0, "duration": 1,
        }]
        script = os.path.join(self.temp.name, "alpha-audio-filters.txt")
        command, _duration = compile_editor_render(
            project,
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
            filter_script_path=script,
            output_path=os.path.join(self.temp.name, "alpha-audio.mov"),
            ffmpeg="ffmpeg",
            encoder_capabilities={"encoders": {"software": True, "nvidia": True}},
        )
        self.assertEqual(command[command.index("-c:v") + 1], "prores_ks")
        self.assertEqual(command[command.index("-profile:v") + 1], "4")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuva444p10le")
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
        self.assertIn("-map", command)
        self.assertIn("[editor_audio]", command)

        project["exports"] = [{
            "id": "alpha-export",
            "filename": "transparent-title.mov",
            "codec": "prores4444",
            "format": "prores4444",
            "source_item_ids": ["title"],
            "workspace": "default",
            "created_at": 10,
        }]
        record = normalize_editor_project(project)["exports"][0]
        self.assertEqual(record["codec"], "prores4444")
        self.assertEqual(record["format"], "prores4444")
        self.assertEqual(record["source_item_ids"], ["title"])

    def test_transparent_title_composites_like_direct_mp4_at_interior_and_edges(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("FFmpeg is not installed")

        def title_project(output_format):
            project = create_editor_project(name="Title composite comparison")
            project["canvas"].update({
                "width": 128,
                "height": 128,
                "fps": 2,
                "background": "#606060",
            })
            project["export"].update({
                "format": output_format,
                "encoder": "software",
                "include_audio": False,
                "filename": "Title composite comparison",
            })
            project["tracks"][2]["items"] = [{
                "id": "title",
                "text": "ALPHA",
                "start": 0,
                "duration": 2,
                "style": {
                    "font_size": 24,
                    "color": "#ffffff",
                    "background_opacity": 0,
                    "y": 0,
                },
                "opacity": 0.5,
                "fade_in": 0.5,
                "fade_out": 0.5,
            }]
            return project

        capabilities = {
            "encoders": {
                "software": True,
                "nvidia": False,
                "intel": False,
                "apple": False,
            },
            "recommended": "software",
            "prores4444": True,
        }
        with patch(
            "services.editor_projects.editor_export_capabilities",
            return_value=capabilities,
        ):
            direct_mp4 = render_editor_project(
                title_project("mp4"),
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
                ffmpeg=ffmpeg,
            )
            alpha_mov = render_editor_project(
                title_project("prores4444"),
                save_root=self.save_root,
                workspace="default",
                uploads_root=self.uploads_root,
                ffmpeg=ffmpeg,
            )

        background = os.path.join(self.temp.name, "composite-background.ppm")
        with open(background, "wb") as handle:
            handle.write(b"P6\n128 128\n255\n" + bytes((96, 96, 96)) * (128 * 128))
        composite = os.path.join(self.temp.name, "alpha-composite.mp4")
        composite_graph = (
            "[0:v]format=rgba[background];[1:v]format=rgba[foreground];"
            "[background][foreground]overlay=format=auto:shortest=1,"
            "format=yuv420p[composite]"
        )
        rendered = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-loop", "1",
                "-framerate", "2",
                "-i", background,
                "-i", alpha_mov,
                "-filter_complex", composite_graph,
                "-map", "[composite]",
                "-frames:v", "4",
                "-c:v", "libx264",
                "-crf", "0",
                composite,
            ],
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(
            rendered.returncode,
            0,
            rendered.stderr.decode("utf-8", errors="replace")[-2000:],
        )

        direct_frames = self._decode_rgb_frames(ffmpeg, direct_mp4, width=128, height=128)
        composite_frames = self._decode_rgb_frames(ffmpeg, composite, width=128, height=128)
        alpha_frames = self._decode_rgba_frames(ffmpeg, alpha_mov, width=128, height=128)
        self.assertEqual((len(direct_frames), len(composite_frames), len(alpha_frames)), (4, 4, 4))

        interior = self._pixel(alpha_frames[2], 72, 67, width=128)
        self.assertTrue(250 <= interior[0] <= 255, interior)
        self.assertTrue(250 <= interior[1] <= 255, interior)
        self.assertTrue(250 <= interior[2] <= 255, interior)
        self.assertTrue(126 <= interior[3] <= 132, interior)

        edge_candidates = []
        for y in range(40, 90):
            for x in range(16, 112):
                pixel = self._pixel(alpha_frames[2], x, y, width=128)
                if 12 <= pixel[3] < 120 and min(pixel[:3]) > 240:
                    edge_candidates.append((x, y, pixel))
        self.assertTrue(edge_candidates, "expected antialiased title-edge pixels")
        edge_x, edge_y, edge_pixel = max(edge_candidates, key=lambda value: value[2][3])
        self.assertTrue(12 <= edge_pixel[3] < 120, edge_pixel)

        for frame_index, (direct, composited) in enumerate(zip(direct_frames, composite_frames)):
            max_difference = 0
            total_difference = 0
            for offset in range(0, len(direct), 3):
                difference = max(
                    abs(direct[offset + channel] - composited[offset + channel])
                    for channel in range(3)
                )
                max_difference = max(max_difference, difference)
                total_difference += difference
            mean_difference = total_difference / (128 * 128)
            self.assertLessEqual(max_difference, 8, (frame_index, max_difference, mean_difference))

        edge_offset = (edge_y * 128 + edge_x) * 3
        direct_edge = tuple(direct_frames[2][edge_offset:edge_offset + 3])
        composite_edge = tuple(composite_frames[2][edge_offset:edge_offset + 3])
        self.assertLessEqual(
            max(abs(a - b) for a, b in zip(direct_edge, composite_edge)),
            8,
            (edge_x, edge_y, edge_pixel, direct_edge, composite_edge),
        )

    def test_mp4_compiler_still_uses_legacy_h264_opaque_path(self):
        project = create_editor_project(name="Legacy MP4")
        project["canvas"].update({"width": 64, "height": 64, "fps": 2})
        project["tracks"][2]["items"] = [{"id": "title", "text": "Title", "start": 0, "duration": 1}]
        script = os.path.join(self.temp.name, "filters.txt")
        command, _duration = compile_editor_render(
            project,
            save_root=self.save_root,
            workspace="default",
            uploads_root=self.uploads_root,
            filter_script_path=script,
            output_path=os.path.join(self.temp.name, "legacy.mp4"),
            ffmpeg="ffmpeg",
            encoder_capabilities={"encoders": {"software": True}},
        )
        filters = open(script, encoding="utf-8").read()
        self.assertIn("color=c=0x000000:s=64x64", filters)
        self.assertIn("format=yuv420p[editor_video]", filters)
        self.assertIn("-pix_fmt", command)
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")


if __name__ == "__main__":
    unittest.main()
