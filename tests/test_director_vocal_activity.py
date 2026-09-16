import math
import os
import struct
import sys
import tempfile
import unittest
import wave
from unittest import mock

STAGED = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, os.path.abspath(STAGED))
from services.director.vocal_activity import classify_vocal_intervals


class TestDirectorVocalActivity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _wav(self, name, left, right=None, rate=8000):
        path = os.path.join(self.temp.name, name)
        right = left if right is None else right
        with wave.open(path, "wb") as target:
            target.setnchannels(2)
            target.setsampwidth(2)
            target.setframerate(rate)
            frames = b"".join(struct.pack("<hh", int(l * 32767), int(r * 32767)) for l, r in zip(left, right))
            target.writeframes(frames)
        return path

    def test_stereo_vocal_in_one_channel_is_active(self):
        rate = 8000
        vocal = [0.2 * math.sin(2 * math.pi * 220 * n / rate) for n in range(rate)]
        path = self._wav("stereo.wav", [0.0] * rate, vocal, rate)
        self.assertEqual(classify_vocal_intervals([{"start": 0, "end": 1}], [], path), ["active"])

    def test_digital_silence_is_silent(self):
        path = self._wav("silent.wav", [0.0] * 8000)
        self.assertEqual(classify_vocal_intervals([{"start": 0, "end": 1}], None, path), ["silent"])

    def test_timestamped_vocal_is_positive_evidence_without_stem(self):
        lyrics = [{"start": 0.25, "end": 0.75, "text": "A sung line"}]
        self.assertEqual(classify_vocal_intervals([{"start": 0, "end": 1}], lyrics, None), ["active"])

    def test_missing_stem_and_missing_transcript_are_unknown(self):
        self.assertEqual(classify_vocal_intervals([{"start": 0, "end": 1}], [], "missing.wav"), ["unknown"])

    def test_out_of_range_interval_is_unknown(self):
        path = self._wav("short.wav", [0.0] * 8000)
        self.assertEqual(classify_vocal_intervals([{"start": 1, "end": 2}], [], path), ["unknown"])

    def test_instrumental_marker_is_not_positive_vocal_evidence(self):
        lyrics = [{"start": 0, "end": 1, "text": "[Instrumental]"}]
        self.assertEqual(classify_vocal_intervals([{"start": 0, "end": 1}], lyrics, None), ["unknown"])

    def test_quiet_nonzero_stem_is_unknown_not_silent(self):
        rate = 8000
        quiet = [0.001 * math.sin(2 * math.pi * 220 * n / rate) for n in range(rate)]
        path = self._wav("quiet.wav", quiet, rate=rate)
        self.assertEqual(classify_vocal_intervals([{"start": 0, "end": 1}], [], path), ["unknown"])

    def test_reads_only_requested_slice_from_longer_file(self):
        rate = 8000
        silence = [0.0] * rate
        vocal = [0.2 * math.sin(2 * math.pi * 220 * n / rate) for n in range(rate)]
        path = self._wav("longer.wav", silence + silence + vocal + silence, rate=rate)
        self.assertEqual(classify_vocal_intervals([{"start": 2, "end": 3}], [], path), ["active"])

    def test_interval_read_error_degrades_to_unknown(self):
        path = self._wav("read-error.wav", [0.0] * 8000)
        with mock.patch("wave.Wave_read.readframes", side_effect=OSError("synthetic read failure")):
            self.assertEqual(classify_vocal_intervals([{"start": 0, "end": 1}], [], path), ["unknown"])

    def test_implausible_sample_rate_is_unknown_without_reading_samples(self):
        path = self._wav("bad-rate.wav", [0.0] * 8, rate=500000)
        with mock.patch("wave.Wave_read.readframes", side_effect=AssertionError("must not allocate")):
            self.assertEqual(classify_vocal_intervals([{"start": 0, "end": 0.00001}], [], path), ["unknown"])


if __name__ == "__main__":
    unittest.main()
