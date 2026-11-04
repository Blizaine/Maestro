"""Focused tests for native and joined video encoding choices."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _load_video_encoding_helper():
    source_path = APP / "services" / "video_encoding.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names = {"get_imageio_codec_params", "get_video_encoding"}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["get_video_encoding"]


def _load_imageio_codec_params():
    source_path = APP / "services" / "video_encoding.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_imageio_codec_params"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["get_imageio_codec_params"]


def _load_native_codec_params():
    source_path = APP / "shared" / "utils" / "audio_video.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_codec_params"
    )
    namespace = {"get_imageio_codec_params": _load_imageio_codec_params()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_get_codec_params"]


def _load_concat_function(server_config):
    source_path = APP / "wgp.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "concatenate_multi_clip_videos"
    )
    namespace = {
        "os": os,
        "subprocess": subprocess,
        "server_config": server_config,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["concatenate_multi_clip_videos"]


class VideoEncodingCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_path = str(APP)
        if app_path not in sys.path:
            sys.path.insert(0, app_path)

    def test_encoding_helper_matches_native_codec_and_container_choices(self):
        get_video_encoding = _load_video_encoding_helper()
        get_imageio_params = _load_imageio_codec_params()
        native_codec_params = _load_native_codec_params()
        cases = (
            ("libx264_8", "mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10"]),
            ("libx264_10", "mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0"]),
            (
                "libx265_28", "mp4",
                ["-c:v", "libx265", "-pix_fmt", "yuv420p", "-crf", "28",
                 "-x265-params", "log-level=none", "-hide_banner", "-nostats"],
            ),
            (
                "libx265_8", "mkv",
                ["-c:v", "libx265", "-pix_fmt", "yuv420p", "-crf", "8",
                 "-x265-params", "log-level=none", "-hide_banner", "-nostats"],
            ),
            ("libx264_lossless", "mkv", ["-c:v", "ffv1", "-pix_fmt", "rgb24"]),
            (
                "libx264_lossless", "mp4",
                ["-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "0"],
            ),
        )
        for codec_type, container, expected_args in cases:
            with self.subTest(codec_type=codec_type, container=container):
                resolved = get_video_encoding(codec_type, container)
                self.assertEqual(
                    native_codec_params(codec_type, container),
                    get_imageio_params(codec_type, container),
                )
                self.assertEqual(resolved["codec_type"], codec_type)
                self.assertEqual(resolved["container"], container)
                self.assertEqual(resolved["extension"], f".{container}")
                self.assertEqual(resolved["ffmpeg_args"], expected_args)

    def test_concat_command_honors_explicit_options_and_server_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_paths = []
            for index in range(2):
                clip_path = Path(temp_dir) / f"clip{index}.mp4"
                clip_path.write_bytes(b"clip")
                clip_paths.append(str(clip_path))

            server_config = {
                "video_output_codec": "libx264_lossless",
                "video_container": "mkv",
            }
            concatenate = _load_concat_function(server_config)
            commands = []
            original_run = subprocess.run

            def fake_run(command, *args, **kwargs):
                if not isinstance(command, (list, tuple)) or not command:
                    return original_run(command, *args, **kwargs)
                executable = os.path.basename(str(command[0])).lower()
                if executable not in ("ffmpeg", "ffmpeg.exe", "ffprobe", "ffprobe.exe"):
                    return original_run(command, *args, **kwargs)
                commands.append(command)
                if "-show_streams" in command:
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
                if "-show_entries" in command:
                    return subprocess.CompletedProcess(command, 0, stdout="24/1\n", stderr="")
                Path(command[-1]).write_bytes(b"joined")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with mock.patch("subprocess.run", side_effect=fake_run):
                explicit_path = Path(temp_dir) / "explicit.mp4"
                self.assertTrue(
                    concatenate(
                        clip_paths,
                        str(explicit_path),
                        codec_type="libx264_8",
                        container="mp4",
                    )
                )
                fallback_path = Path(temp_dir) / "fallback.mkv"
                self.assertTrue(concatenate(clip_paths, str(fallback_path)))

            output_commands = [
                command for command in commands if command[-1] in (str(explicit_path).replace("\\", "/"), str(fallback_path).replace("\\", "/"))
            ]
            self.assertEqual(len(output_commands), 2)
            self.assertEqual(
                output_commands[0][output_commands[0].index("-c:v"):],
                ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10",
                 str(explicit_path).replace("\\", "/")],
            )
            self.assertEqual(
                output_commands[1][output_commands[1].index("-c:v"):],
                ["-c:v", "ffv1", "-pix_fmt", "rgb24",
                 str(fallback_path).replace("\\", "/")],
            )

    def test_concat_rejects_a_container_suffix_mismatch_before_encoding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_path = Path(temp_dir) / "clip.mp4"
            clip_path.write_bytes(b"clip")
            concatenate = _load_concat_function({})
            with mock.patch("subprocess.run") as run:
                self.assertFalse(
                    concatenate(
                        [str(clip_path)],
                        str(Path(temp_dir) / "joined.mp4"),
                        codec_type="libx264_lossless",
                        container="mkv",
                    )
                )
                run.assert_not_called()

    @unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required")
    def test_tiny_joined_outputs_probe_and_decode_for_mp4_and_lossless_mkv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_paths = []
            for index, color in enumerate(("red", "blue")):
                clip_path = Path(temp_dir) / f"clip{index}.mp4"
                subprocess.run(
                    [
                        FFMPEG, "-y", "-f", "lavfi", "-i",
                        f"color=c={color}:s=64x64:r=4:d=0.5", "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                clip_paths.append(str(clip_path))

            outputs = (
                ("libx264_8", "mp4", "h264", {"yuv420p"}),
                ("libx265_8", "mp4", "hevc", {"yuv420p"}),
                ("libx264_lossless", "mp4", "h264", {"yuv444p"}),
                # FFV1 can negotiate rgb24 input as bgr0 output while keeping
                # lossless packed RGB storage.
                ("libx264_lossless", "mkv", "ffv1", {"rgb24", "bgr0"}),
            )
            for codec_type, container, expected_codec, expected_pixel_formats in outputs:
                with self.subTest(codec_type=codec_type, container=container):
                    output_path = Path(temp_dir) / f"joined-{codec_type}.{container}"
                    concatenate = _load_concat_function(
                        {
                            "video_output_codec": "libx264_8",
                            "video_container": "mp4",
                        }
                    )
                    with mock.patch.dict(
                        os.environ,
                        {"FFMPEG_BINARY": FFMPEG, "FFPROBE_BINARY": FFPROBE},
                    ):
                        self.assertTrue(
                            concatenate(
                                clip_paths,
                                str(output_path),
                                video_duration_sec=1.0,
                                codec_type=codec_type,
                                container=container,
                            )
                        )
                    probe = subprocess.run(
                        [
                            FFPROBE, "-v", "error", "-count_frames",
                            "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,pix_fmt,nb_read_frames", "-of", "json",
                            str(output_path),
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=30,
                    )
                    stream = json.loads(probe.stdout)["streams"][0]
                    self.assertEqual(stream["codec_name"], expected_codec)
                    self.assertIn(stream["pix_fmt"], expected_pixel_formats)
                    self.assertEqual(int(stream["nb_read_frames"]), 4)
                    subprocess.run(
                        [FFMPEG, "-v", "error", "-i", str(output_path), "-f", "null", "-"],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=30,
                    )


if __name__ == "__main__":
    unittest.main()
