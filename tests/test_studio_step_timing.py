"""Model-free tests for Studio's live denoising rate samples."""
from __future__ import annotations

import os
import sys
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.studio_step_timing import StudioStepTiming  # noqa: E402


class TestStudioStepTiming(unittest.TestCase):
    def test_speech_frames_are_not_denoising_steps(self):
        timing = StudioStepTiming()
        for step in (1, 2, 3):
            self.assertIsNone(timing.observe(
                step, 40, "Denoising | Generating Speech | 1s", now=step,
            ))

    def test_averages_only_completed_denoising_steps(self):
        timing = StudioStepTiming()

        self.assertIsNone(timing.observe(
            1, 40, "Prompt 1/1, Sliding Window 1/2 - Denoising First Pass | 1s", now=10,
        ))
        self.assertEqual(timing.observe(
            3, 40, "Prompt 1/1, Sliding Window 1/2 - Denoising First Pass | 7s", now=16,
        ), 3)
        self.assertEqual(timing.observe(
            3, 40, "Prompt 1/1, Sliding Window 1/2 - Denoising First Pass | 7s", now=20,
        ), 3, "Duplicate progress does not add time to the sample")

        self.assertIsNone(timing.observe(0, 0, "VAE Decoding", now=50))
        self.assertIsNone(timing.observe(
            1, 40, "Prompt 1/1, Sliding Window 1/2 - Denoising Second Pass | 1s", now=100,
        ), "A new denoising pass starts a fresh sample")
        self.assertEqual(timing.observe(
            2, 40, "Prompt 1/1, Sliding Window 1/2 - Denoising Second Pass | 2s", now=104,
        ), 4)

    def test_resets_when_window_or_step_counter_changes(self):
        timing = StudioStepTiming()
        timing.observe(1, 30, "Sliding Window 1/3 - Denoising | 1s", now=1)
        self.assertEqual(timing.observe(2, 30, "Sliding Window 1/3 - Denoising | 2s", now=4), 3)

        self.assertIsNone(timing.observe(
            1, 30, "Sliding Window 2/3 - Denoising | 1s", now=100,
        ), "A new window does not include the prior window's time")
        self.assertEqual(timing.observe(2, 30, "Sliding Window 2/3 - Denoising | 2s", now=105), 5)

        self.assertIsNone(timing.observe(1, 30, "Sliding Window 2/3 - Denoising | 1s", now=200),
                          "A reset step counter starts a fresh average")
        self.assertEqual(timing.observe(2, 30, "Sliding Window 2/3 - Denoising | 2s", now=207), 7)


if __name__ == "__main__":
    unittest.main()
