"""Source-backed projection of explicit final silent holds over visual windows."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_performance_audio import (
    discover_h3_requested_static_holds,
    project_h3_requested_static_tail,
)
from services.h3_story_ledger import extract_source_events


def _drive_context(duration_seconds=37.326961, *, intent="drive"):
    if intent != "drive":
        return f"Voice reference only. H3_PERFORMANCE_AUDIO_CLOCK duration_seconds={duration_seconds}"
    return (
        "The exact target soundtrack supplies the performance and timing.\n"
        f"H3_PERFORMANCE_AUDIO_CLOCK duration_seconds={duration_seconds}"
    )


class H3PerformanceAudioTailTests(unittest.TestCase):
    def setUp(self):
        self.source = (
            "One adult shadow performer moves a single square of pale cloth across the floor, "
            "raises it as a screen, then lowers it beside one lantern. "
            "Use <Audio 1> as the exact performance-driving soundtrack, once at its supplied speed and without edits. "
            "The source track is 37.327 seconds; finish the movement at its final note. "
            "The 55.25-second video holds the final lantern-and-cloth composition in silence "
            "for the remaining 17.923 seconds."
        )
        self.events = extract_source_events(self.source)
        self.holds = discover_h3_requested_static_holds(self.events)

    def test_explicit_source_tail_projects_across_intersecting_windows(self):
        self.assertEqual(len(self.holds), 1)
        hold = self.holds[0]
        self.assertEqual(hold["source_event_id"], "E4")
        self.assertEqual(
            hold["source_event_text"],
            "The 55.25-second video holds the final lantern-and-cloth composition in silence for the remaining 17.923 seconds",
        )
        projected = project_h3_requested_static_tail(
            _drive_context(), [14.375, 13.625, 13.625, 13.625], hold
        )
        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertEqual(projected["source_event_id"], "E4")
        self.assertEqual(projected["source_event_text"], hold["source_event_text"])
        self.assertTrue(math.isclose(projected["global_start_seconds"], 37.326961, abs_tol=1e-9))
        self.assertTrue(math.isclose(projected["global_end_seconds"], 55.25, abs_tol=1e-9))
        self.assertTrue(math.isclose(projected["total_hold_seconds"], 17.923039, abs_tol=1e-9))
        self.assertEqual(len(projected["windows"]), 2)
        first, final = projected["windows"]
        self.assertEqual(first["window"], 3)
        self.assertEqual(final["window"], 4)
        self.assertAlmostEqual(first["global_start_seconds"], 37.326961, places=9)
        self.assertAlmostEqual(first["global_end_seconds"], 41.625, places=9)
        self.assertAlmostEqual(first["local_start_seconds"], 9.326961, places=9)
        self.assertAlmostEqual(first["local_end_seconds"], 13.625, places=9)
        self.assertAlmostEqual(first["hold_seconds"], 4.298039, places=9)
        self.assertAlmostEqual(final["global_start_seconds"], 41.625, places=9)
        self.assertAlmostEqual(final["global_end_seconds"], 55.25, places=9)
        self.assertAlmostEqual(final["local_start_seconds"], 0.0, places=9)
        self.assertAlmostEqual(final["local_end_seconds"], 13.625, places=9)
        self.assertAlmostEqual(final["hold_seconds"], 13.625, places=9)

    def test_source_discovery_requires_unambiguous_affirmative_static_final_state(self):
        accepted = (
            "The 55.25-second video holds the final composition in silence for the remaining 17.923 seconds.",
            "A 20-second clip holds the last frame in silence for the remaining 4 seconds.",
            "A 20-second clip holds the performer's final composition in silence for the remaining 4 seconds.",
        )
        for index, text in enumerate(accepted, start=1):
            with self.subTest(text=text):
                self.assertEqual(
                    len(discover_h3_requested_static_holds([{"event_id": f"E{index}", "text": text}])),
                    1,
                )

        rejected = (
            "If the 55.25-second video holds the final composition in silence for the remaining 17.923 seconds.",
            "The 55.25-second video does not hold the final composition in silence for the remaining 17.923 seconds.",
            'The note says, "The 55.25-second video holds the final composition in silence for the remaining 17.923 seconds."',
            "The 55.25-second video holds the final composition while the performer moves in silence for the remaining 17.923 seconds.",
            "The 55.25-second video holds the final composition as the performer turns in silence for the remaining 17.923 seconds.",
            "'The 55.25-second video holds the final composition in silence for the remaining 17.923 seconds.'",
            "The 55.25-second video holds a composition in silence for the remaining 17.923 seconds.",
            "The 55.25-second video holds the final composition in silence for the remaining 60 seconds.",
            "The video holds the final composition in silence for the remaining 17.923 seconds.",
        )
        for index, text in enumerate(rejected, start=10):
            with self.subTest(text=text):
                self.assertEqual(
                    discover_h3_requested_static_holds([{"event_id": f"E{index}", "text": text}]),
                    [],
                )

    def test_silence_alone_or_prose_duration_does_not_create_a_hold_or_audio_clock(self):
        no_hold = extract_source_events(
            "A 55.25-second video ends in silence after the source track runs for 37.327 seconds."
        )
        self.assertEqual(discover_h3_requested_static_holds(no_hold), [])
        self.assertIsNone(project_h3_requested_static_tail(
            "A role description says the track lasts 37.326961 seconds.",
            [14.375, 13.625, 13.625, 13.625],
            self.holds[0] if self.holds else {},
        ))

    def test_projection_abstains_without_exact_drive_clock_or_matching_timeline(self):
        hold = self.holds[0]
        windows = [14.375, 13.625, 13.625, 13.625]
        invalid = (
            (_drive_context(intent="voice"), windows, hold),
            (_drive_context(), [14.375, 13.625, 13.625], hold),
            (_drive_context(), [14.375, 13.625, 13.625, 10], hold),
            (_drive_context(40.0), windows, hold),
        )
        for context, durations, request in invalid:
            with self.subTest(context=context, durations=durations):
                self.assertIsNone(project_h3_requested_static_tail(context, durations, request))

        forged = dict(hold, static_hold_explicit=False)
        self.assertIsNone(project_h3_requested_static_tail(_drive_context(), windows, forged))
        forged = dict(hold, requested_tail_seconds=1.0)
        self.assertIsNone(project_h3_requested_static_tail(_drive_context(), windows, forged))


if __name__ == "__main__":
    unittest.main()
