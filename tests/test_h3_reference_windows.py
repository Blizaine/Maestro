"""Timeline slicing regressions for MiniMax H3 reference media."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.minimax_h3.ref2va import prepare_references
from models.minimax_h3.reference_manifest import validate_reference_manifest
from models.minimax_h3.reference_media import normalize_reference_manifest


class H3ReferenceWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = shutil.which("ffmpeg")
        if not cls.ffmpeg or not shutil.which("ffprobe"):
            raise unittest.SkipTest("FFmpeg and FFprobe are required for real reference-window tests.")

    def setUp(self):
        self.original_cwd = os.getcwd()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        os.chdir(self.root)
        (self.root / "uploads").mkdir()

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.tempdir.cleanup()

    def _ffmpeg(self, *args: str) -> None:
        subprocess.run(
            [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _make_timeline_clip(
        self,
        path: Path,
        *,
        segment_seconds: int = 2,
        source_fps: int = 24,
    ) -> Path:
        colors = ("red", "green", "blue")
        frequencies = (220, 440, 880)
        command: list[str] = []
        for color in colors:
            command += [
                "-f", "lavfi", "-i",
                f"color=c={color}:s=32x32:r={source_fps}:d={segment_seconds}",
            ]
        for frequency in frequencies:
            command += [
                "-f", "lavfi", "-i",
                f"sine=frequency={frequency}:sample_rate=32000:duration={segment_seconds}",
            ]
        filter_graph = (
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v];"
            "[3:a][4:a][5:a]concat=n=3:v=0:a=1[a]"
        )
        self._ffmpeg(
            *command,
            "-filter_complex", filter_graph,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
            "-c:a", "aac", "-b:a", "128k", "-ar", "32000", "-ac", "2",
            "-movflags", "+faststart", str(path),
        )
        return path

    def _make_tone(self, path: Path, *, duration: float = 10, frequency: int = 330) -> Path:
        self._ffmpeg(
            "-f", "lavfi", "-i",
            f"sine=frequency={frequency}:sample_rate=32000:duration={duration}",
            "-ar", "32000", "-ac", "2", "-c:a", "pcm_s16le", str(path),
        )
        return path

    def _make_timeline_audio(self, path: Path, *, segment_seconds: int = 2) -> Path:
        command: list[str] = []
        for frequency in (220, 440, 880):
            command += [
                "-f", "lavfi", "-i",
                f"sine=frequency={frequency}:sample_rate=32000:duration={segment_seconds}",
            ]
        self._ffmpeg(
            *command,
            "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]",
            "-map", "[a]", "-ar", "32000", "-ac", "2", "-c:a", "pcm_s16le", str(path),
        )
        return path

    @staticmethod
    def _rgb(frame) -> tuple[int, int, int]:
        center = np.asarray(frame)[16, 16].astype(np.float32)
        return tuple(int(round(value)) for value in center)

    @staticmethod
    def _dominant_frequency(waveform, start_seconds: float, *, duration: float = 0.5) -> float:
        samples = waveform[0].detach().cpu().numpy()
        start = int(round(start_seconds * 32000))
        segment = samples[start : start + int(round(duration * 32000))]
        segment = segment.astype(np.float64)
        segment -= segment.mean()
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        frequencies = np.fft.rfftfreq(len(segment), d=1 / 32000)
        return float(frequencies[int(np.argmax(spectrum[1:]) + 1)])

    def _prepare(self, manifest, *, start_frame: int, num_frames: int = 96):
        return prepare_references(
            manifest,
            num_frames=num_frames,
            target_height=32,
            target_width=32,
            timeline_start_frame=start_frame,
        )

    def test_later_windows_advance_and_overlapping_windows_share_source_frames(self):
        clip = self._make_timeline_clip(self.root / "timeline.mp4")
        manifest = [{
            "type": "video",
            "path": str(clip),
            "include_audio": False,
            "effective_duration_seconds": 4.0,
        }]

        first = self._prepare(manifest, start_frame=24)[0].frames
        later = self._prepare(manifest, start_frame=48)[0].frames

        self.assertGreater(self._rgb(first[0])[0], 220)
        self.assertGreater(self._rgb(later[0])[1], 80)
        self.assertLess(self._rgb(later[0])[0], 60)
        # Both positions depict source time 3s, despite having different local offsets.
        np.testing.assert_array_equal(first[48], later[24])

    def test_non_24fps_source_is_resampled_before_video_and_audio_window_alignment(self):
        clip = self._make_timeline_clip(self.root / "timeline-30fps.mp4", source_fps=30)
        prepared = self._prepare(
            [{"type": "video", "path": str(clip), "effective_duration_seconds": 4.0}],
            start_frame=48,
            num_frames=96,
        )[0]

        self.assertEqual(prepared.frames.shape[0], 96)
        self.assertGreater(self._rgb(prepared.frames[0])[1], 80)
        self.assertGreater(self._rgb(prepared.frames[48])[2], 80)
        self.assertAlmostEqual(self._dominant_frequency(prepared.waveform, 0.25), 440, delta=12)
        self.assertAlmostEqual(self._dominant_frequency(prepared.waveform, 2.25), 880, delta=12)

    def test_embedded_and_replacement_audio_follow_video_window_and_pad_at_eof(self):
        clip = self._make_timeline_clip(self.root / "timeline.mp4")
        replacement = self._make_timeline_audio(self.root / "replacement.wav")
        manifest = [
            {"type": "video", "path": str(clip), "effective_duration_seconds": 4.0},
            {
                "type": "video",
                "path": str(clip),
                "audio_path": str(replacement),
                "effective_duration_seconds": 4.0,
            },
        ]

        prepared = self._prepare(manifest, start_frame=60, num_frames=96)
        for reference in prepared:
            self.assertTrue(reference.has_audio)
            self.assertEqual(reference.frames.shape[0], 96)
            self.assertGreater(self._rgb(reference.frames[0])[1], 80)
            self.assertGreater(self._rgb(reference.frames[38])[2], 80)
            self.assertAlmostEqual(self._dominant_frequency(reference.waveform, 0.2), 440, delta=12)
            self.assertAlmostEqual(self._dominant_frequency(reference.waveform, 1.8), 880, delta=12)

        # This window has one second left in the source. Video holds blue and
        # both soundtrack sources become silence after their final sample.
        eof_manifest = [{
            "type": "video",
            "path": str(clip),
            "audio_path": str(replacement),
            "effective_duration_seconds": 4.0,
        }]
        eof = self._prepare(eof_manifest, start_frame=120, num_frames=96)[0]
        self.assertGreater(self._rgb(eof.frames[-1])[2], 80)
        tail = eof.waveform[:, int(1.3 * 32000) :]
        self.assertLess(float(tail.abs().max()), 1e-4)

        past_eof = self._prepare(eof_manifest, start_frame=168, num_frames=96)[0]
        self.assertTrue(all(self._rgb(frame)[2] > 80 for frame in past_eof.frames))
        self.assertLess(float(past_eof.waveform.abs().max()), 1e-4)

    def test_video_and_style_audio_share_fair_fifteen_second_pass_budgets(self):
        from models.minimax_h3.ref2va import _reference_pass_budgets

        clip = self._make_timeline_clip(self.root / "timeline.mp4")
        videos = [
            {
                "type": "video",
                "path": str(clip),
                "include_audio": False,
                "source_duration_seconds": 10.0,
            }
            for _ in range(3)
        ]
        prepared_videos = self._prepare(videos, start_frame=0, num_frames=240)
        self.assertEqual([item.frames.shape[0] for item in prepared_videos], [120, 120, 120])
        self.assertEqual(sum(item.frames.shape[0] for item in prepared_videos) / 24, 15)
        rounded_budgets = _reference_pass_budgets(
            [
                {"type": "video", "effective_duration_seconds": duration}
                for duration in (4.001, 15.0, 15.0)
            ],
            "video",
            15.0,
        )
        self.assertLessEqual(sum(rounded_budgets.values()), 15.0)
        self.assertLessEqual(sum(round(value * 24) for value in rounded_budgets.values()), 360)

        from PIL import Image

        portrait = self.root / "portrait.png"
        Image.new("RGB", (32, 32), "white").save(portrait)
        tones = self._make_tone(self.root / "style.wav", duration=10)
        style_manifest = [
            {"type": "image", "path": str(portrait)},
            *[
                {
                    "type": "audio",
                    "path": str(tones),
                    "audio_intent": "style",
                    "source_duration_seconds": 10.0,
                }
                for _ in range(3)
            ],
        ]
        prepared_audio = self._prepare(style_manifest, start_frame=0, num_frames=240)
        self.assertEqual([item.waveform.shape[-1] for item in prepared_audio[1:]], [160000] * 3)
        self.assertEqual(sum(item.waveform.shape[-1] for item in prepared_audio[1:]) / 32000, 15)

    def test_short_sound_effect_repeats_after_eof_while_style_becomes_silent(self):
        from PIL import Image

        portrait = self.root / "portrait.png"
        Image.new("RGB", (32, 32), "white").save(portrait)
        sound = self._make_tone(self.root / "blaster.wav", duration=3.226, frequency=770)
        original = sound.read_bytes()
        manifest = [
            {"type": "image", "path": str(portrait)},
            {"type": "audio", "path": str(sound), "audio_intent": "sound"},
        ]
        normalized = normalize_reference_manifest(manifest)
        self.assertEqual(normalize_reference_manifest(normalized), normalized)

        samples = []
        for start_frame in (0, 276, 552):
            prepared = self._prepare(normalized, start_frame=start_frame, num_frames=294)[1]
            self.assertEqual(prepared.audio_intent, "sound")
            self.assertTrue(prepared.has_audio)
            self.assertGreater(float(prepared.waveform.abs().max()), 0.05)
            self.assertAlmostEqual(prepared.waveform.shape[-1] / 32000, 3.226, delta=0.01)
            self.assertAlmostEqual(self._dominant_frequency(prepared.waveform, 0.25), 770, delta=12)
            samples.append(prepared.waveform.numpy())
        np.testing.assert_array_equal(samples[0], samples[1])
        np.testing.assert_array_equal(samples[0], samples[2])

        style_manifest = [manifest[0], {**manifest[1], "audio_intent": "style"}]
        first = self._prepare(style_manifest, start_frame=0, num_frames=294)[1]
        self.assertGreater(float(first.waveform.abs().max()), 0.05)
        for start_frame in (276, 552):
            later = self._prepare(style_manifest, start_frame=start_frame, num_frames=294)[1]
            self.assertEqual(float(later.waveform.abs().max()), 0.0)
        self.assertEqual(sound.read_bytes(), original)

    def test_reusable_sound_effects_share_the_existing_audio_budget(self):
        from PIL import Image

        portrait = self.root / "portrait.png"
        Image.new("RGB", (32, 32), "white").save(portrait)
        sound = self._make_tone(self.root / "effects.wav", duration=10, frequency=550)
        manifest = [
            {"type": "image", "path": str(portrait)},
            *[{
                "type": "audio", "path": str(sound), "audio_intent": "sound",
                "source_duration_seconds": 10.0,
            } for _ in range(3)],
        ]
        prepared = self._prepare(manifest, start_frame=552, num_frames=294)[1:]
        self.assertEqual([item.waveform.shape[-1] for item in prepared], [160000] * 3)
        self.assertEqual(sum(item.waveform.shape[-1] for item in prepared) / 32000, 15)
        self.assertTrue(all(float(item.waveform.abs().max()) > 0.05 for item in prepared))

    def test_saved_video_and_voice_remain_static_and_refmods_reject_timeline_following(self):
        clip = self._make_timeline_clip(self.root / "timeline.mp4")
        ordinary = validate_reference_manifest([{"type": "video", "path": str(clip)}])[0]
        saved = validate_reference_manifest([{
            "type": "video",
            "path": str(clip),
            "library_character_id": "saved-character",
        }])[0]
        character = validate_reference_manifest([{
            "type": "video",
            "path": str(clip),
            "video_intent": "character",
        }])[0]
        self.assertTrue(ordinary["follow_timeline"])
        self.assertFalse(saved["follow_timeline"])
        self.assertFalse(character["follow_timeline"])

        explicitly_static = validate_reference_manifest([{
            "type": "video",
            "path": str(clip),
            "follow_timeline": False,
        }])[0]
        self.assertFalse(explicitly_static["follow_timeline"])
        first = self._prepare([explicitly_static], start_frame=0)[0]
        later = self._prepare([explicitly_static], start_frame=72)[0]
        np.testing.assert_array_equal(first.frames, later.frames)

        (self.root / "uploads" / "characters").mkdir()
        with self.assertRaisesRegex(ValueError, "encoded RefMod video cannot follow the timeline"):
            validate_reference_manifest([{
                "type": "video",
                "path": str(clip),
                "refmod_path": str(self.root / "uploads" / "characters" / "saved.safetensors"),
                "follow_timeline": True,
            }], require_files=False)

        from PIL import Image

        portrait = self.root / "portrait.png"
        Image.new("RGB", (32, 32), "white").save(portrait)
        voice = self._make_tone(self.root / "voice.wav", duration=10, frequency=550)
        voice_manifest = [
            {"type": "image", "path": str(portrait)},
            {"type": "audio", "path": str(voice), "audio_intent": "voice"},
        ]
        voice_first = self._prepare(voice_manifest, start_frame=0)[1].waveform
        voice_later = self._prepare(voice_manifest, start_frame=72)[1].waveform
        self.assertEqual(voice_first.shape, voice_later.shape)
        self.assertTrue((voice_first == voice_later).all())

    def test_normalization_preserves_timeline_originals_and_is_idempotent(self):
        clip = self._make_timeline_clip(self.root / "timeline.mp4")
        attached = self._make_tone(self.root / "replacement.wav", duration=10, frequency=770)
        style = self._make_tone(self.root / "style.wav", duration=10, frequency=440)
        voice = self._make_tone(self.root / "voice.wav", duration=10, frequency=550)
        originals = {path: path.read_bytes() for path in (clip, attached, style, voice)}
        manifest = [
            {"type": "video", "path": str(clip), "audio_path": str(attached)},
            {"type": "audio", "path": str(style), "audio_intent": "style"},
            {"type": "audio", "path": str(voice), "audio_intent": "voice"},
        ]

        once = normalize_reference_manifest(manifest)
        twice = normalize_reference_manifest(once)

        self.assertEqual(once, twice)
        self.assertEqual(once[0]["path"], str(clip))
        self.assertEqual(once[0]["audio_path"], str(attached))
        self.assertEqual(once[1]["path"], str(style))
        self.assertNotEqual(once[2]["path"], str(voice))
        self.assertAlmostEqual(once[0]["effective_duration_seconds"], 6.0, delta=0.05)
        self.assertEqual(once[1]["effective_duration_seconds"], 7.5)
        self.assertEqual(once[2]["effective_duration_seconds"], 7.5)
        self.assertEqual({path: path.read_bytes() for path in (clip, attached, style, voice)}, originals)
        self.assertTrue(Path(once[2]["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
