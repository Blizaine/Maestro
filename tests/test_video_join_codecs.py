"""Focused tests for native and joined video encoding choices."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import symtable
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

    @unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required")
    def test_mixed_clip_dimensions_and_sample_aspect_join_with_soundtrack(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_specs = (
                ("red", 96, 54, "1/1"),
                ("blue", 160, 90, "1/1"),
                ("green", 128, 72, "1/1"),
                ("yellow", 64, 96, "1/1"),
                # The source pixels are 4:3, but its 2:1 SAR makes the
                # displayed image 8:3. Re-join should preserve that shape.
                ("magenta", 64, 48, "2/1"),
            )
            clip_paths = []
            for index, (color, width, height, sar) in enumerate(clip_specs):
                clip_path = Path(temp_dir) / f"clip{index}.mp4"
                subprocess.run(
                    [
                        FFMPEG, "-y", "-f", "lavfi", "-i",
                        f"color=c={color}:s={width}x{height}:r=4:d=0.5",
                        "-vf", f"setsar={sar}", "-frames:v", "2", "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(clip_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                clip_paths.append(str(clip_path))

            soundtrack = Path(temp_dir) / "soundtrack.wav"
            subprocess.run(
                [
                    FFMPEG, "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=44100:duration=2.5",
                    "-c:a", "pcm_s16le", str(soundtrack),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            output_path = Path(temp_dir) / "joined.mp4"
            concatenate = _load_concat_function({})
            with mock.patch.dict(
                os.environ,
                {"FFMPEG_BINARY": FFMPEG, "FFPROBE_BINARY": FFPROBE},
            ):
                self.assertTrue(
                    concatenate(clip_paths, str(output_path), str(soundtrack))
                )

            probe = subprocess.run(
                [
                    FFPROBE, "-v", "error", "-count_frames", "-show_streams",
                    "-show_entries",
                    "stream=codec_type,width,height,sample_aspect_ratio,nb_read_frames",
                    "-of", "json", str(output_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            streams = json.loads(probe.stdout)["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            self.assertEqual((video["width"], video["height"]), (160, 90))
            self.assertEqual(video["sample_aspect_ratio"], "1:1")
            self.assertEqual(int(video["nb_read_frames"]), 10)
            self.assertTrue(any(stream["codec_type"] == "audio" for stream in streams))

            last_frame = subprocess.run(
                [
                    FFMPEG, "-v", "error", "-i", str(output_path),
                    "-vf", r"select=eq(n\,9)", "-vsync", "0", "-frames:v", "1",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
                ],
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(last_frame.returncode, 0, last_frame.stderr.decode(errors="replace"))
            frame = last_frame.stdout
            self.assertEqual(len(frame), 160 * 90 * 3)

            def pixel(x, y):
                offset = (y * 160 + x) * 3
                return tuple(frame[offset:offset + 3])

            self.assertLess(max(pixel(80, 10)), 30, "wide source should have top letterboxing")
            red, green, blue = pixel(80, 45)
            self.assertGreater(red, 150)
            self.assertLess(green, 100)
            self.assertGreater(blue, 100)


@unittest.skipUnless(FFMPEG and FFPROBE, "requires FFmpeg")
class SoundtrackConversionTests(unittest.TestCase):
    def test_converted_song_preserves_exact_source_samples(self):
        import soundfile as sf
        import numpy as np

        path = APP / "postprocessing" / "voice_clone.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        func = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_save_converted_song")
        namespace = {"subprocess": subprocess}
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), namespace)
        with tempfile.TemporaryDirectory() as work:
            source, converted, output = (Path(work) / (name + ".wav") for name in ("source", "converted", "output"))
            sf.write(source, np.zeros((23456, 2)), 44_100)
            sf.write(converted, np.zeros((7500, 2)), 24_000)
            self.assertTrue(namespace["_save_converted_song"](source, str(converted), output))
            self.assertEqual(sf.info(output).frames, 23456)
            self.assertEqual(sf.info(output).samplerate, 44_100)
            self.assertEqual(sf.info(output).channels, 2)

    def test_stereo_instrumental_stays_stereo_after_vocal_remix(self):
        import numpy as np
        import soundfile as sf
        from postprocessing.voice_clone import _remix_vocals_with_background

        with tempfile.TemporaryDirectory() as work:
            vocals, backing, output = (Path(work) / (name + ".wav") for name in ("vocals", "backing", "output"))
            sf.write(vocals, np.full(128, 0.25), 24_000, subtype="FLOAT")
            sf.write(backing, np.tile([0.125, -0.125], (128, 1)), 24_000, subtype="FLOAT")
            _remix_vocals_with_background(vocals, backing, output, preserve_layout=True)
            audio, _ = sf.read(output)
            np.testing.assert_allclose(audio, np.tile([0.375, 0.125], (128, 1)))


@unittest.skipUnless(FFMPEG and FFPROBE, "requires FFmpeg")
class WindowVideoWriterTests(unittest.TestCase):
    def test_generation_uses_module_shutil_for_ffmpeg_detection(self):
        path = APP / "wgp.py"
        symbols = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
        generate = next(table for table in symbols.get_children() if table.get_name() == "generate_video")
        self.assertTrue(symbols.lookup("shutil").is_imported())
        self.assertTrue(generate.lookup("shutil").is_global())

    def test_cumulative_output_uses_every_window_without_reencoding_prior_frames(self):
        from shared.utils.window_video_writer import WindowVideoWriter
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            writer = WindowVideoWriter(fps=24, directory=work, ffmpeg=FFMPEG)
            encodes = []

            def encode(frames, filename):
                encodes.append(filename)
                subprocess.run([
                    FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=96x64:r=24",
                    "-frames:v", str(frames.shape[1]), "-c:v", "libx264", "-pix_fmt", "yuv420p", filename,
                ], capture_output=True, check=True, timeout=30)

            for index, count in enumerate((7, 11, 9), 1):
                writer.append(SimpleNamespace(shape=(3, count, 64, 96)), encode=encode)
                writer.write_video(str(root / f"output-{index}.mp4"))
            writer.close()
            self.assertEqual(len(encodes), 3)
            self.assertFalse(writer.root.exists())
            probe = subprocess.run([
                FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,r_frame_rate", "-of", "json",
                str(root / "output-3.mp4"),
            ], capture_output=True, check=True, timeout=30)
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(int(stream["nb_frames"]), 27)
            self.assertEqual(stream["r_frame_rate"], "24/1")


if __name__ == "__main__":
    unittest.main()
