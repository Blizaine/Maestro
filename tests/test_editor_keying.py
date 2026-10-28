"""CPU FFmpeg regressions for Editor chroma keying."""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.editor_keying import (  # noqa: E402
    editor_keying_filters,
    normalize_editor_keying,
)
from services.editor_projects import (  # noqa: E402
    EditorProjectError,
    compile_editor_render,
    create_editor_project,
    normalize_editor_project,
)


class TestEditorKeying(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.ffmpeg = shutil.which("ffmpeg")

    def tearDown(self):
        self.temp.cleanup()

    def _write_rgba_png(
        self,
        name: str,
        pixels: list[tuple[int, int, int, int]],
        *,
        height: int = 1,
    ) -> str:
        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        row = b"\x00" + b"".join(bytes(pixel) for pixel in pixels)
        image = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", len(pixels), height, 8, 6, 0, 0, 0),
            )
            + chunk(b"IDAT", zlib.compress(row * height))
            + chunk(b"IEND", b"")
        )
        path = os.path.join(self.root, name)
        with open(path, "wb") as handle:
            handle.write(image)
        return path

    def _run_keying(
        self,
        pixels: list[tuple[int, int, int, int]],
        keying: dict,
        *,
        post_filter: str = "null",
    ) -> bytes:
        if not self.ffmpeg:
            self.skipTest("FFmpeg is not installed")
        source = self._write_rgba_png("source.png", pixels)
        graph = editor_keying_filters(
            normalize_editor_keying(keying),
            input_label="0:v",
            output_label="keyed",
            suffix="test",
        )
        graph.append(f"[keyed]{post_filter}[out]")
        result = subprocess.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                source,
                "-filter_complex",
                ";".join(graph),
                "-map",
                "[out]",
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
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode("utf-8", errors="replace")[-2000:],
        )
        self.assertEqual(len(result.stdout), len(pixels) * 4)
        return result.stdout

    @staticmethod
    def _expected_alpha(
        pixel: tuple[int, int, int, int],
        key_color: tuple[int, int, int],
        *,
        similarity: float,
        softness: float,
    ) -> float:
        distance = math.sqrt(
            sum((pixel[channel] - key_color[channel]) ** 2 for channel in range(3))
            / (3 * 255 * 255)
        )
        matte = (
            min(1.0, max(0.0, (distance - similarity) / softness))
            if softness > 0.0001
            else float(distance > similarity)
        )
        return matte * pixel[3]

    def test_cpu_filter_matches_matte_formula_and_multiplies_source_alpha(self):
        pixels = [
            (0, 255, 0, 255),       # exact green: fully transparent
            (255, 0, 0, 255),       # opaque red: retained
            (0, 180, 0, 255),       # green edge: partial matte
            (255, 0, 0, 128),       # source alpha survives keying
        ]
        softness = 0.10
        actual = self._run_keying(
            pixels,
            {
                "enabled": True,
                "color": "#00ff00",
                "similarity": 0.15,
                "softness": softness,
                "spill": 0.0,
            },
        )
        for index, pixel in enumerate(pixels):
            expected = self._expected_alpha(
                pixel,
                (0, 255, 0),
                similarity=0.15,
                softness=softness,
            )
            self.assertLessEqual(
                abs(actual[index * 4 + 3] - expected),
                3,
                (index, actual[index * 4 + 3], expected),
            )
        self.assertEqual(actual[0:3], bytes((0, 255, 0)))
        self.assertEqual(actual[4:7], bytes((255, 0, 0)))

    def test_softness_at_the_hardstep_threshold_uses_binary_matte(self):
        pixels = [
            (0, 220, 0, 255),  # distance is inside similarity
            (0, 180, 0, 255),  # distance is outside similarity
        ]
        for softness in (0.0, 0.0001):
            with self.subTest(softness=softness):
                actual = self._run_keying(
                    pixels,
                    {
                        "enabled": True,
                        "color": "#00ff00",
                        "similarity": 0.15,
                        "softness": softness,
                        "spill": 0.0,
                    },
                )
                self.assertLessEqual(actual[3], 3, actual)
                self.assertGreaterEqual(actual[7], 252, actual)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_green_keys_reveal_lower_blue_layer_and_keep_opaque_red(self):
        background = self._write_rgba_png(
            "background.png",
            [(0, 0, 255, 255), (0, 0, 255, 255)],
        )
        foreground = self._write_rgba_png(
            "foreground.png",
            [(0, 255, 0, 255), (255, 0, 0, 255)],
        )
        key_filters = editor_keying_filters(
            normalize_editor_keying({
                "enabled": True,
                "color": "#00ff00",
                "similarity": 0.15,
                "softness": 0.10,
                "spill": 0.0,
            }),
            input_label="1:v",
            output_label="keyed",
            suffix="composite",
        )
        graph = [
            "[0:v]format=rgba[background]",
            *key_filters,
            "[background][keyed]overlay=shortest=1:format=auto,format=rgba[out]",
        ]
        result = subprocess.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                background,
                "-i",
                foreground,
                "-filter_complex",
                ";".join(graph),
                "-map",
                "[out]",
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
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode("utf-8", errors="replace")[-2000:],
        )
        first, second = result.stdout[:4], result.stdout[4:8]
        self.assertLessEqual(max(abs(first[i] - (0, 0, 255, 255)[i]) for i in range(4)), 3, first)
        self.assertLessEqual(max(abs(second[i] - (255, 0, 0, 255)[i]) for i in range(4)), 3, second)

    def test_green_and_blue_despill_match_channel_reduction_formula(self):
        cases = (
            (
                "#00ff00",
                (200, 220, 80, 255),
                (200, 180, 80, 255),
            ),
            (
                "#0000ff",
                (200, 80, 220, 255),
                (200, 80, 180, 255),
            ),
            (
                "#ff0000",
                (120, 180, 200, 255),
                (120, 180, 200, 255),
            ),
        )
        for color, source, expected in cases:
            with self.subTest(color=color):
                actual = self._run_keying(
                    [source],
                    {
                        "enabled": True,
                        "color": color,
                        "similarity": 0.15,
                        "softness": 0.10,
                        "spill": 0.5,
                    },
                )
                for channel in range(4):
                    self.assertLessEqual(
                        abs(actual[channel] - expected[channel]),
                        3,
                        (color, channel, actual[channel], expected[channel]),
                    )

    def test_postkey_color_adjustments_leave_the_key_matte_unchanged(self):
        pixel = (80, 255, 0, 255)
        actual = self._run_keying(
            [pixel],
            {
                "enabled": True,
                "color": "#00ff00",
                "similarity": 0.15,
                "softness": 0.20,
                "spill": 0.0,
            },
            post_filter="colorchannelmixer=rr=0:gg=1:bb=1",
        )
        expected_alpha = self._expected_alpha(
            pixel,
            (0, 255, 0),
            similarity=0.15,
            softness=0.20,
        )
        self.assertLessEqual(abs(actual[3] - expected_alpha), 3, actual)
        self.assertEqual(actual[0], 0)
        self.assertGreater(actual[3], 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_complete_editor_export_with_key_color_transform_fade_and_opacity(self):
        foreground = self._write_rgba_png("integrated.png", [(0, 255, 0, 255)] * 32 + [(255, 0, 0, 255)] * 32, height=64)
        for animated in (False, True):
            with self.subTest(animated=animated):
                project = create_editor_project(width=64, height=64, fps=8)
                project["canvas"]["background"] = "#0000ff"
                project["assets"] = {"title": {"id":"title", "name":os.path.basename(foreground), "path":foreground,
                    "type":"image", "origin":"upload", "duration":0, "width":64, "height":64}}
                item = {"id":"title", "name":"title", "asset_id":"title", "start":0, "duration":.5,
                    "source_in":0,"speed":1,"volume":0,"opacity":.75,"fit":"contain",
                    "transform":{"x":0,"y":0,"scale":1,"rotation":0},"fade_in":.125,
                    "color":{"exposure":.2,"contrast":1,"saturation":1,"temperature":0},
                    "keying":{"enabled":True,"color":"#00ff00","similarity":.15,"softness":.1,"spill":.5}}
                if animated:
                    item["keyframes"] = {"opacity":[{"time":0,"value":.75,"easing":"linear"},{"time":.5,"value":.75,"easing":"linear"}],
                        "x":[{"time":0,"value":0,"easing":"linear"},{"time":.5,"value":4,"easing":"linear"}]}
                project["tracks"][0]["items"] = [item]
                project["export"].update({"encoder":"software","include_audio":False})
                output = os.path.join(self.root, f"integrated-{animated}.mp4")
                command, _ = compile_editor_render(project, save_root=self.root, workspace="default", uploads_root=self.root,
                    filter_script_path=os.path.join(self.root, f"integrated-{animated}.txt"), output_path=output,
                    ffmpeg=self.ffmpeg, encoder_capabilities={"encoders":{"software":True},"recommended":"software"})
                rendered = subprocess.run(command, capture_output=True, timeout=30)
                self.assertEqual(rendered.returncode, 0, rendered.stderr.decode(errors="replace"))
                frame = subprocess.run([self.ffmpeg,"-v","error","-ss","0.25","-i",output,"-frames:v","1","-f","rawvideo","-pix_fmt","rgb24","pipe:1"],capture_output=True,timeout=30)
                self.assertEqual(frame.returncode, 0, frame.stderr.decode(errors="replace"))
                self.assertEqual(len(frame.stdout),64*64*3)
                left = frame.stdout[(32*64+16)*3:(32*64+16)*3+3]
                right = frame.stdout[(32*64+48)*3:(32*64+48)*3+3]
                self.assertLess(left[0],8,left)
                self.assertGreater(left[2],245,left)
                self.assertGreater(right[0],170,right)
                self.assertLess(right[1],12,right)
                self.assertTrue(40 < right[2] < 90,right)

    def test_disabled_and_absent_keying_are_graph_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            media = self._write_rgba_png("identity.png", [(255, 0, 0, 255)] * 4, height=4)
            project = create_editor_project(name="Keying identity", width=64, height=64, fps=2)
            project["export"].update({"encoder": "software", "include_audio": False})
            project["assets"] = {
                "image": {
                    "id": "image",
                    "name": os.path.basename(media),
                    "path": media,
                    "type": "image",
                    "origin": "output",
                    "duration": 1,
                    "width": 64,
                    "height": 64,
                    "has_audio": False,
                }
            }
            item = {"id": "clip", "asset_id": "image", "start": 0, "duration": 1}
            project["tracks"][0]["items"] = [dict(item)]
            _, _ = compile_editor_render(
                project,
                save_root=self.root,
                workspace="default",
                uploads_root=self.root,
                filter_script_path=os.path.join(directory, "absent.txt"),
                output_path=os.path.join(directory, "absent.mp4"),
            )
            with open(os.path.join(directory, "absent.txt"), encoding="utf-8") as handle:
                absent_graph = handle.read()

            project["tracks"][0]["items"] = [
                {**item, "keying": {"enabled": False}}
            ]
            _, _ = compile_editor_render(
                project,
                save_root=self.root,
                workspace="default",
                uploads_root=self.root,
                filter_script_path=os.path.join(directory, "disabled.txt"),
                output_path=os.path.join(directory, "disabled.mp4"),
            )
            with open(os.path.join(directory, "disabled.txt"), encoding="utf-8") as handle:
                disabled_graph = handle.read()
            self.assertEqual(disabled_graph, absent_graph)

    def test_settings_validation_and_visual_layer_restrictions(self):
        good = normalize_editor_keying({"enabled": True})
        self.assertEqual(
            good,
            {
                "enabled": True,
                "color": "#00ff00",
                "similarity": 0.15,
                "softness": 0.10,
                "spill": 0.5,
            },
        )
        self.assertIsNone(normalize_editor_keying(None))

        invalid_values = (
            {"enabled": 1},
            {"enabled": "true"},
            {"enabled": True, "color": "green"},
            {"enabled": True, "color": "#00ff0"},
            {"enabled": True, "similarity": True},
            {"enabled": True, "similarity": float("nan")},
            {"enabled": True, "spill": 10**10000},
            {"enabled": True, "similarity": 0},
            {"enabled": True, "similarity": 1.01},
            {"enabled": True, "softness": -0.01},
            {"enabled": True, "spill": 1.01},
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_editor_keying(value)

        project = create_editor_project(name="Keying only on visuals")
        project["tracks"][1]["items"] = [
            {"id": "audio", "asset_id": "sound", "duration": 1, "keying": {"enabled": True}}
        ]
        with self.assertRaisesRegex(EditorProjectError, "keying requires a video track"):
            normalize_editor_project(project)

        project = create_editor_project(name="No audio keying")
        project["assets"] = {
            "sound": {"id": "sound", "type": "audio", "duration": 1}
        }
        project["tracks"][0]["items"] = [
            {"id": "audio", "asset_id": "sound", "duration": 1, "keying": {"enabled": False}}
        ]
        with self.assertRaisesRegex(EditorProjectError, "keying requires a video or image asset"):
            normalize_editor_project(project)

    def test_compiler_runs_keying_before_transform_color_fade_and_opacity(self):
        with tempfile.TemporaryDirectory() as directory:
            media = self._write_rgba_png("ordered.png", [(0, 255, 0, 255)] * 4, height=4)
            project = create_editor_project(name="Keying order", width=64, height=64, fps=2)
            project["export"].update({"encoder": "software", "include_audio": False})
            project["assets"] = {
                "image": {
                    "id": "image",
                    "name": os.path.basename(media),
                    "path": media,
                    "type": "image",
                    "origin": "output",
                    "duration": 1,
                    "width": 64,
                    "height": 64,
                    "has_audio": False,
                }
            }
            project["tracks"][0]["items"] = [{
                "id": "clip",
                "asset_id": "image",
                "start": 0,
                "duration": 1,
                "keying": {
                    "enabled": True,
                    "color": "#00ff00",
                    "similarity": 0.15,
                    "softness": 0.10,
                    "spill": 0.5,
                },
                "transform": {"scale": 1.2, "rotation": 12},
                "color": {"exposure": 0.2},
                "fade_in": 0.25,
                "transition_in": "fade_black",
                "opacity": 0.7,
            }]
            filter_path = os.path.join(directory, "ordered.txt")
            compile_editor_render(
                project,
                save_root=self.root,
                workspace="default",
                uploads_root=self.root,
                filter_script_path=filter_path,
                output_path=os.path.join(directory, "ordered.mp4"),
            )
            with open(filter_path, encoding="utf-8") as handle:
                graph = handle.read()

        ordered_filters = (
            "colorkey=",
            "despill=",
            "scale=",
            "rotate=",
            "exposure=exposure=",
            "fade=t=in:",
            "colorchannelmixer=aa=",
        )
        positions = [graph.index(name) for name in ordered_filters]
        self.assertEqual(positions, sorted(positions), list(zip(ordered_filters, positions)))


if __name__ == "__main__":
    unittest.main()
