"""CPU-only gallery excerpt tests using small synthetic media fixtures."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from services.editor_projects import probe_media
from services.gallery_media_trim import (
    GalleryMediaTrimError,
    GalleryMediaTrimTimeout,
    GalleryMediaTrimUnavailable,
    trim_gallery_media,
)


def _resolve_media_binary(name: str) -> str | None:
    """Follow the app resolver's env, bundled-project, then PATH lookup order."""
    environment_name = {"ffmpeg": "FFMPEG_BINARY", "ffprobe": "FFPROBE_BINARY"}[name]
    configured = os.environ.get(environment_name)
    if configured and Path(configured).is_file():
        return configured
    executable = name + (".exe" if os.name == "nt" else "")
    bundled = APP / "ffmpeg_bins" / executable
    if bundled.is_file():
        return str(bundled)
    return shutil.which(executable) or shutil.which(name)


FFMPEG = _resolve_media_binary("ffmpeg")
FFPROBE = _resolve_media_binary("ffprobe")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_wave(path: Path, *, duration: float = 3.0, rate: int = 32000, channels: int = 2):
    frames = round(duration * rate)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        payload = bytearray()
        for index in range(frames):
            sample = int(10000 * math.sin(2 * math.pi * 440 * index / rate))
            payload.extend(sample.to_bytes(2, "little", signed=True) * channels)
        output.writeframes(payload)


def _probe_streams(path: Path):
    result = subprocess.run(
        [
            FFPROBE, "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return json.loads(result.stdout)


class GalleryMediaTrimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FFMPEG or not FFPROBE:
            raise unittest.SkipTest("The installed CPU FFmpeg and FFprobe binaries are required.")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="maestro-gallery-trim-test-")
        self.root = Path(self.temp.name)
        self.uploads = self.root / "uploads"
        self.outputs = self.root / "outputs"
        self.uploads.mkdir()
        self.outputs.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def trim(self, asset, start, end, **overrides):
        options = {
            "save_root": str(self.outputs),
            "workspace": "default",
            "uploads_root": str(self.uploads),
            "ffmpeg": FFMPEG,
            "ffprobe": FFPROBE,
            "timeout": 30,
        }
        options.update(overrides)
        return trim_gallery_media(asset, start, end, **options)

    def test_audio_excerpt_is_sample_accurate_preserves_rate_channels_and_source(self):
        source = self.uploads / "source.wav"
        _write_wave(source)
        source_hash = _sha256(source)

        result = self.trim({"name": source.name, "origin": "upload"}, 0.37, 1.63)

        excerpt = self.uploads / result["filename"]
        self.assertEqual(result["media_type"], "audio")
        self.assertEqual(result["mime_type"], "audio/wav")
        self.assertNotEqual(excerpt, source)
        self.assertTrue(excerpt.is_file())
        with wave.open(str(excerpt), "rb") as output:
            self.assertEqual(output.getframerate(), 32000)
            self.assertEqual(output.getnchannels(), 2)
            self.assertEqual(output.getsampwidth(), 2)
            self.assertLessEqual(abs(output.getnframes() - round(1.26 * 32000)), 1)
        self.assertAlmostEqual(result["duration"], 1.26, delta=1 / 32000)
        self.assertEqual(_sha256(source), source_hash)

    def test_audio_with_embedded_album_art_is_trimmed_as_audio(self):
        cover = self.root / "cover.jpg"
        source = self.uploads / "covered.mp3"
        subprocess.run(
            [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.1",
                "-frames:v", "1", str(cover),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        subprocess.run(
            [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=3",
                "-i", str(cover), "-map", "0:a:0", "-map", "1:v:0",
                "-c:a", "libmp3lame", "-b:a", "128k", "-c:v", "copy",
                "-disposition:v:0", "attached_pic", "-id3v2_version", "3",
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        source_hash = _sha256(source)
        probed = probe_media(str(source), ffprobe=FFPROBE)
        self.assertTrue(probed["has_audio"])
        self.assertGreater(probed["width"], 0)  # FFprobe sees the attached cover as video.

        result = self.trim({"name": source.name, "origin": "upload"}, 0.25, 1.75)

        excerpt = self.uploads / result["filename"]
        self.assertEqual(result["media_type"], "audio")
        self.assertEqual(result["mime_type"], "audio/wav")
        with wave.open(str(excerpt), "rb") as output:
            self.assertEqual(output.getframerate(), 44100)
            self.assertEqual(output.getnchannels(), 1)
            self.assertEqual(output.getsampwidth(), 2)
            self.assertLessEqual(abs(output.getnframes() - round(1.5 * 44100)), 1)
        self.assertEqual(_sha256(source), source_hash)

    def _write_test_video(self, path: Path, *, with_audio: bool):
        command = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=25:duration=3",
        ]
        if with_audio:
            command.extend(["-f", "lavfi", "-i", "sine=frequency=660:sample_rate=44100:duration=3"])
        command.extend(["-t", "3", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"])
        if with_audio:
            command.extend(["-c:a", "aac", "-shortest"])
        else:
            command.append("-an")
        command.append(str(path))
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def test_video_excerpt_reencodes_fractional_range_keeps_rate_and_optional_audio(self):
        source = self.outputs / "fractional-source.mp4"
        self._write_test_video(source, with_audio=True)
        source_hash = _sha256(source)

        result = self.trim(
            {"name": source.name, "origin": "output", "workspace": "default"},
            0.31,
            1.57,
        )

        excerpt = self.uploads / result["filename"]
        data = _probe_streams(excerpt)
        video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
        audio = next(stream for stream in data["streams"] if stream["codec_type"] == "audio")
        self.assertEqual(result["media_type"], "video")
        self.assertEqual(result["mime_type"], "video/mp4")
        self.assertEqual(Path(result["filename"]).suffix, ".mp4")
        self.assertEqual(video["codec_name"], "h264")
        self.assertEqual(video["avg_frame_rate"], "25/1")
        self.assertEqual(audio["codec_name"], "aac")
        self.assertAlmostEqual(result["duration"], 1.26, delta=1 / 25)
        self.assertEqual(_sha256(source), source_hash)

    def test_video_without_audio_stays_video_only(self):
        source = self.outputs / "silent.mp4"
        self._write_test_video(source, with_audio=False)
        result = self.trim(
            {"name": source.name, "origin": "output", "workspace": "default"},
            0.2,
            1.4,
        )
        data = _probe_streams(self.uploads / result["filename"])
        self.assertEqual(result["media_type"], "video")
        self.assertTrue(any(stream["codec_name"] == "h264" for stream in data["streams"]))
        self.assertFalse(any(stream["codec_type"] == "audio" for stream in data["streams"]))

    def test_invalid_times_and_end_beyond_source_are_rejected(self):
        source = self.uploads / "source.wav"
        _write_wave(source)
        asset = {"name": source.name, "origin": "upload"}
        for start, end in ((-0.1, 1.0), (1.0, 1.0), (1.1, 1.0), (0.0, 3.01), (float("nan"), 1.0), (0.0, float("inf")), (True, 1.0)):
            with self.subTest(start=start, end=end), self.assertRaises(GalleryMediaTrimError):
                self.trim(asset, start, end)

    def test_asset_name_traversal_and_unsupported_origin_are_rejected(self):
        for name in ("../source.wav", r"..\source.wav", r"C:\private.wav", ".hidden.wav"):
            with self.subTest(name=name), self.assertRaises(GalleryMediaTrimError):
                self.trim({"name": name, "origin": "upload"}, 0, 1)
        with self.assertRaisesRegex(GalleryMediaTrimError, "origin"):
            self.trim({"name": "source.wav", "origin": "project"}, 0, 1)

    def test_output_resolution_never_falls_back_to_another_workspace_or_root(self):
        other_workspace = self.outputs / "Other"
        other_workspace.mkdir()
        _write_wave(other_workspace / "shared.wav")
        _write_wave(self.outputs / "legacy-root.wav")

        with self.assertRaises(FileNotFoundError):
            self.trim(
                {"name": "shared.wav", "origin": "output", "workspace": "Missing"},
                0,
                1,
            )
        with self.assertRaises(FileNotFoundError):
            self.trim(
                {"name": "legacy-root.wav", "origin": "output", "workspace": "Missing"},
                0,
                1,
            )
        with self.assertRaises(GalleryMediaTrimError):
            self.trim(
                {"name": "shared.wav", "origin": "output", "workspace": "../Other"},
                0,
                1,
            )

    def test_symlinked_upload_is_rejected(self):
        outside = self.root / "outside.wav"
        _write_wave(outside)
        linked = self.uploads / "linked.wav"
        try:
            os.symlink(outside, linked)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Symlink creation is unavailable: {error}")
        with self.assertRaises(GalleryMediaTrimError):
            self.trim({"name": linked.name, "origin": "upload"}, 0, 1)

    def test_ffmpeg_launch_failure_removes_partial_and_keeps_source(self):
        source = self.uploads / "source.wav"
        _write_wave(source)
        source_hash = _sha256(source)
        with self.assertRaises(GalleryMediaTrimUnavailable):
            self.trim(
                {"name": source.name, "origin": "upload"},
                0.2,
                1.2,
                ffmpeg=str(source),  # existing non-executable file; failure occurs after temp creation
            )
        self.assertEqual(_sha256(source), source_hash)
        self.assertEqual([path.name for path in self.uploads.iterdir()], [source.name])

    def test_ffmpeg_timeout_cleans_partial_and_releases_bounded_slot(self):
        source = self.uploads / "source.wav"
        _write_wave(source)
        source_hash = _sha256(source)
        asset = {"name": source.name, "origin": "upload"}

        with self.assertRaises(GalleryMediaTrimTimeout):
            self.trim(asset, 0.2, 1.2, timeout=0)

        self.assertEqual(_sha256(source), source_hash)
        self.assertEqual([path.name for path in self.uploads.iterdir()], [source.name])
        result = self.trim(asset, 0.2, 1.2)
        self.assertTrue((self.uploads / result["filename"]).is_file())


if __name__ == "__main__":
    unittest.main()
