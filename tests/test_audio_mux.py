"""CPU audio preparation and final mux preserve stereo across window joins."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import ffmpeg
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from shared.utils.audio_video import (  # noqa: E402
    combine_and_concatenate_video_with_audio_tracks,
    resample_audio_array,
)


class AudioPreparationTests(unittest.TestCase):
    def test_window_join_resampling_does_not_use_default_accelerator(self):
        t = np.arange(960, dtype=np.float32) / 48000
        for channels in (1, 2):
            with self.subTest(channels=channels):
                audio = np.sin(2 * np.pi * 440 * t)
                if channels == 2:
                    audio = np.stack([audio, np.sin(2 * np.pi * 880 * t)], axis=1)
                expected = resample_audio_array(audio, 48000, 32000)
                with torch.device("meta"):
                    actual = resample_audio_array(audio, 48000, 32000)
                    self.assertEqual(torch.empty(0).device.type, "meta")
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape, (640, 2) if channels == 2 else (640,))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class AudioMuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.video = str(cls.root / "silent.mp4")
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "color=c=black:s=96x64:r=24:d=2", "-c:v", "libx264", cls.video,
        ], check=True)
        cls.rate = 32000
        t = np.arange(cls.rate * 2) / cls.rate
        cls.mono = str(cls.root / "mono.wav")
        cls.stereo = str(cls.root / "stereo.wav")
        cls.prefix = str(cls.root / "prefix.wav")
        sf.write(cls.mono, 0.3 * np.sin(2 * np.pi * 660 * t), cls.rate)
        sf.write(cls.stereo, 0.3 * np.stack([
            np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t),
        ], axis=1), cls.rate)
        sf.write(cls.prefix, 0.3 * np.stack([
            np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t),
        ], axis=1), cls.rate)

    def mux(self, sources, news, prefix_seconds=0, **kwargs):
        output = str(self.root / "output.mp4")
        combine_and_concatenate_video_with_audio_tracks(
            output, self.video, sources, news, prefix_seconds, self.rate, **kwargs,
        )
        streams = [s for s in ffmpeg.probe(output)["streams"] if s["codec_type"] == "audio"]
        return output, streams

    def samples(self, output, stream=0, channels=2):
        audio = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", output, "-map", f"0:a:{stream}",
            "-f", "f32le", "-ar", str(self.rate), "-ac", str(channels), "-",
        ])
        return np.frombuffer(audio, dtype="<f4").reshape(-1, channels)

    def assert_tones(self, audio, start, frequencies):
        segment = audio[int(start * self.rate):int((start + 0.2) * self.rate)]
        self.assertEqual(len(segment), 6400)
        for channel, frequency in enumerate(frequencies):
            peak = np.argmax(np.abs(np.fft.rfft(segment[:, channel]))) * 5
            self.assertAlmostEqual(peak, frequency, delta=5)

    def test_generated_stereo_stays_stereo_in_aac_and_alac(self):
        for codec in ("aac_128", "alac"):
            with self.subTest(codec=codec):
                output, streams = self.mux([], [self.stereo], audio_codec_key=codec)
                self.assertEqual(streams[0]["channels"], 2)
                self.assertAlmostEqual(float(streams[0]["duration"]), 2, delta=0.1)
                self.assert_tones(self.samples(output), 0.2, (440, 880))

    def test_stereo_prefix_and_new_window_preserve_channels_timing_and_language(self):
        for from_start in (False, True):
            with self.subTest(from_start=from_start):
                output, streams = self.mux(
                    [self.prefix], [self.stereo], 1,
                    new_audio_from_start=from_start,
                    source_audio_metadata=[{"channels": 2, "language": "eng"}],
                )
                self.assertEqual(streams[0]["channels"], 2)
                self.assertEqual(streams[0]["tags"]["language"], "eng")
                audio = self.samples(output)
                self.assert_tones(audio, 0.2, (220, 330))
                self.assert_tones(audio, 1.2, (440, 880))

    def test_mono_and_stereo_window_join_does_not_collapse_stereo(self):
        for source, new, before, after in (
            (self.mono, self.stereo, (660, 660), (440, 880)),
            (self.prefix, self.mono, (220, 330), (660, 660)),
        ):
            with self.subTest(source=source, new=new):
                output, streams = self.mux([source], [new], 1)
                self.assertEqual(streams[0]["channels"], 2)
                audio = self.samples(output)
                self.assert_tones(audio, 0.2, before)
                self.assert_tones(audio, 1.2, after)

    def test_mono_tracks_remain_mono(self):
        for sources, prefix_seconds in (([], 0), ([self.mono], 1)):
            with self.subTest(prefix_seconds=prefix_seconds):
                output, streams = self.mux(sources, [self.mono], prefix_seconds)
                self.assertEqual(streams[0]["channels"], 1)
                self.assert_tones(self.samples(output, channels=1), 1.2, (660,))

    def test_silent_prefix_and_missing_new_audio_use_matching_channels(self):
        output, streams = self.mux([], [self.stereo], 1)
        self.assertEqual(streams[0]["channels"], 2)
        audio = self.samples(output)
        self.assertLess(float(np.abs(audio[6400:12800]).max()), 0.001)
        self.assert_tones(audio, 1.2, (440, 880))
        output, streams = self.mux([self.prefix], [], 1)
        self.assertEqual(streams[0]["channels"], 2)
        audio = self.samples(output)
        self.assert_tones(audio, 0.2, (220, 330))
        self.assertLess(float(np.abs(audio[38400:44800]).max()), 0.001)

    def test_multiple_tracks_keep_individual_layouts_and_duplicate_source_language(self):
        _, streams = self.mux([], [self.mono, self.stereo])
        self.assertEqual([s["channels"] for s in streams], [1, 2])
        output, streams = self.mux(
            [self.prefix], [self.mono, self.stereo], 1,
            source_audio_metadata=[{"channels": 2, "language": "eng"}],
        )
        self.assertEqual([s["channels"] for s in streams], [2, 2])
        self.assertEqual([s["tags"]["language"] for s in streams], ["eng", "eng"])
        self.assert_tones(self.samples(output, stream=1), 0.2, (220, 330))
        self.assert_tones(self.samples(output, stream=1), 1.2, (440, 880))


if __name__ == "__main__":
    unittest.main()
