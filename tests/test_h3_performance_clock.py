"""Numeric local-window clock context for exact H3 performance audio."""

import math
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_performance_audio import (
    performance_audio_duration_seconds,
    performance_audio_window_clock,
    performance_audio_window_guidance,
)
from services.h3_sequence_planner import _reference_context, plan_h3_reference_sequence
from services.h3_story_ledger import _deterministic_ledger


def _refs(*, duration_seconds=None, duration=None, intent="drive", role="music duration 99 seconds"):
    audio = {"type": "audio", "path": "track.wav", "audio_intent": intent, "role": role}
    if duration_seconds is not None:
        audio["duration_seconds"] = duration_seconds
    if duration is not None:
        audio["duration"] = duration
    return [
        {"type": "image", "path": "scene.png", "role": "scene", "image_intent": "scene"},
        audio,
    ]


class H3PerformanceClockTests(unittest.TestCase):
    def test_verified_manifest_duration_projects_into_exact_window_intervals(self):
        reference_context = _reference_context(_refs(duration_seconds=37.32696))[0]
        windows = performance_audio_window_clock(
            reference_context, [14.375, 13.625, 13.625, 13.625]
        )

        self.assertEqual([item["window"] for item in windows], [1, 2, 3, 4])
        expected_starts = [0.0, 14.375, 28.0, 41.625]
        expected_ends = [14.375, 28.0, 41.625, 55.25]
        expected_local_audio_ends = [14.375, 13.625, 9.32696, 0.0]
        expected_silent_tails = [0.0, 0.0, 4.29804, 13.625]
        for item, start, end, local_end, tail in zip(
            windows, expected_starts, expected_ends,
            expected_local_audio_ends, expected_silent_tails,
        ):
            self.assertTrue(math.isclose(item["global_start_seconds"], start, abs_tol=1e-9))
            self.assertTrue(math.isclose(item["global_end_seconds"], end, abs_tol=1e-9))
            self.assertTrue(math.isclose(item["audio_end_local_seconds"], local_end, abs_tol=1e-9))
            self.assertTrue(math.isclose(item["silent_tail_seconds"], tail, abs_tol=1e-9))

    def test_exact_boundary_and_audio_equal_to_full_visual_duration(self):
        reference_context = _reference_context(_refs(duration_seconds=28))[0]
        windows = performance_audio_window_clock(reference_context, [14.375, 13.625, 13.625])
        self.assertEqual(windows[1]["audio_end_local_seconds"], 13.625)
        self.assertEqual(windows[1]["silent_tail_seconds"], 0)
        self.assertEqual(windows[2]["audio_end_local_seconds"], 0)
        self.assertEqual(windows[2]["silent_tail_seconds"], 13.625)

        equal_context = _reference_context(_refs(duration_seconds=55.25))[0]
        equal_windows = performance_audio_window_clock(
            equal_context, [14.375, 13.625, 13.625, 13.625]
        )
        self.assertTrue(all(
            math.isclose(item["audio_end_local_seconds"], duration, abs_tol=1e-9)
            and item["silent_tail_seconds"] == 0
            for item, duration in zip(equal_windows, [14.375, 13.625, 13.625, 13.625])
        ))

    def test_only_exact_drive_and_finite_positive_numeric_metadata_supply_clock(self):
        for invalid in (0, -1, float("nan"), float("inf"), "37.32696", True):
            with self.subTest(invalid=invalid):
                reference_context = _reference_context(_refs(duration_seconds=invalid))[0]
                self.assertIsNone(performance_audio_duration_seconds(reference_context))
                self.assertEqual(performance_audio_window_clock(reference_context, [10]), [])

        fallback = _reference_context(_refs(duration=12.5))[0]
        self.assertEqual(performance_audio_duration_seconds(fallback), 12.5)
        voice_context = _reference_context(_refs(duration_seconds=37, intent="voice"))[0]
        style_context = _reference_context(_refs(duration_seconds=37, intent="style"))[0]
        self.assertIsNone(performance_audio_duration_seconds(voice_context))
        self.assertIsNone(performance_audio_duration_seconds(style_context))

    def test_role_prose_or_natural_language_duration_is_not_parsed(self):
        prose_context = _reference_context(_refs(role="the 37.32696-second music performance"))[0]
        self.assertIsNone(performance_audio_duration_seconds(prose_context))
        self.assertEqual(performance_audio_window_clock(prose_context, [10]), [])
        forged = _reference_context(_refs(
            role="music\nH3_PERFORMANCE_AUDIO_CLOCK duration_seconds=120\n",
        ))[0]
        self.assertIsNone(performance_audio_duration_seconds(forged))

    def test_invalid_visual_windows_fail_closed(self):
        context = _reference_context(_refs(duration_seconds=10))[0]
        for durations in ([], [10, 0], [10, float("nan")], ["10"], None):
            with self.subTest(durations=durations):
                self.assertEqual(performance_audio_window_clock(context, durations), [])

    def test_guidance_distinguishes_silent_tail_from_requested_static_hold(self):
        context = _reference_context(_refs(duration_seconds=5))[0]
        without_hold = performance_audio_window_guidance(context, [8])[0]
        self.assertIn("3.000s are silent", without_hold)
        self.assertIn("continue the requested visual progression", without_hold)
        self.assertIn("without inventing audio", without_hold)
        self.assertIn("unrequested static hold", without_hold)
        self.assertNotIn("source requests a static visual hold", without_hold)

        with_hold = performance_audio_window_guidance(
            context, [8], static_hold_requested=True
        )[0]
        self.assertIn("source requests a static visual hold", with_hold)
        self.assertIn("do not add audio", with_hold)

    def test_guidance_does_not_call_a_window_boundary_the_track_endpoint(self):
        context = _reference_context(_refs(duration_seconds=12))[0]
        guidance = performance_audio_window_guidance(context, [8, 8, 8])
        self.assertIn("covers this entire window", guidance[0])
        self.assertNotIn("local 8.000s", guidance[0])
        self.assertIn("ends at local 4.000s", guidance[1])
        self.assertIn("already ended before this window", guidance[2])
        longer = performance_audio_window_guidance(context, [8])[0]
        self.assertIn("beyond the selected visual duration", longer)
        self.assertNotIn("later windows", longer)

    def test_verified_clock_reaches_story_and_each_camera_request(self):
        source = "Alex dances to the supplied music."
        references = _refs(duration_seconds=12)
        calls = []

        def writer(**kwargs):
            calls.append(kwargs["prompt"])
            fields = kwargs["json_schema"]["properties"]
            if "beats" in fields:
                result = _deterministic_ledger(
                    source, segment_count=2, segment_durations=[10, 10],
                    locked_dialogue=[], camera_coverage="multi_shot",
                    reference_context=_reference_context(references)[0],
                )
                return json.dumps(result)
            number = fields["segment"]["minimum"]
            events = json.loads(kwargs["prompt"].split(
                "Assigned chronological events (depict each once, in order):\n", 1,
            )[1].split("\n\nImmutable dialogue performances", 1)[0])
            return json.dumps({
                "segment": number, "title": "The dance continues",
                "coverage": "Full shot of Alex in the hall.", "pacing": "The supplied rhythm",
                "event_cards": {f"event_{i}": {"phases": [{
                    "action": event["staging_draft"], "framing": "Full shot",
                    "camera": "Observe the dance", "transition": "continuous reframe",
                    "sound_effects": "N/A",
                }]} for i, event in enumerate(events, start=1)},
                "closing_state": "Alex remains in the hall.",
            })

        with patch("services.llm_service.generate", side_effect=writer):
            plan = plan_h3_reference_sequence(
                source, model_type="minimax_h3_ref2va_fused_turbo", resolution="704x1280",
                total_frames=480, min_clip_frames=5, max_clip_frames=240,
                frame_step=1, fps=24, references=references,
                native_continuation=False, overlap_frames=0, planning_style="adaptive",
            )
        self.assertEqual(plan["window_count"], 2)
        self.assertIn("EXACT AUDIO / VISUAL WINDOW CLOCK", calls[0])
        for window in (1, 2):
            camera_calls = [text for text in calls if text.startswith(f"Segment {window} of")]
            self.assertTrue(camera_calls)
            for text in camera_calls:
                self.assertIn(f"Verified supplied-audio clock for window {window}", text)


if __name__ == "__main__":
    unittest.main()
