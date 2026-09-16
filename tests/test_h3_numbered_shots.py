"""Unbracketed imported shot clocks retain story order and actor ownership."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_authored_brief import authored_timed_brief
from services.h3_story_ledger import extract_h3_source_intent, extract_source_events
from services.h3_window_planner import plan_h3_sliding_windows


TIMES = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 12), (12, 14), (14, 16), (16, 18)]
ACTIONS = [
    "Swirling beige dust fills the whole frame before anything else becomes visible.",
    "The warrior runs across the empty desert toward a dark rock formation.",
    "The warrior rotates through the air while swinging his sword; blue-white energy follows the blade.",
    "The warrior spins across the sand as an orange energy ring surrounds him.",
    "A gigantic dust vortex forms around the warrior, who stays visible at its center.",
    "A column of sand rises toward the storm clouds while the warrior remains inside it.",
    "The warrior emerges from the vortex. Blue-white energy streams from his blade. Small debris flies past the camera.",
    "The warrior holds his sword diagonally in a calm heroic pose. Silver-white hair moves in the wind.",
]


def shooting_brief(header):
    return (
        "Adapt this to 28 seconds in a barren desert using the reference image. "
        "One adult warrior wears a flowing white robe and carries a long sword.\n\n"
        + "\n\n".join(header(i + 1, *times) + "\n" + action
                       for i, (times, action) in enumerate(zip(TIMES, ACTIONS)))
        + "\n\nVISUAL STYLE:\nPhotorealistic skin and cloth under storm clouds."
        + "\n\nCAMERA:\nFast controlled tracking and motivated changes of angle."
    )


class NumberedShotTests(unittest.TestCase):
    def test_plain_and_markdown_shot_clocks_keep_eight_complete_events(self):
        headers = [
            lambda n, a, b: f"SHOT {n} — 0:{a:02}–0:{b:02}",
            lambda n, a, b: f"Shot {n}: {a}-{b}s",
            lambda n, a, b: f"## Shot {n} — 00:{a:02}.000 — 00:{b:02}.000",
            lambda n, a, b: f"**SHOT {n} — 0:{a:02}–0:{b:02}**",
            lambda n, a, b: f"- Shot {n}: {a} seconds to {b} seconds",
        ]
        for header in headers:
            with self.subTest(header=header(1, 0, 2)):
                source = shooting_brief(header)
                brief = authored_timed_brief(source)
                self.assertEqual([event["text"] for event in brief["events"]], ACTIONS)
                self.assertEqual([(e["source_start_seconds"], e["source_end_seconds"])
                                  for e in brief["events"]], TIMES)
                self.assertEqual(len(extract_source_events(source)), 8)
                for event in brief["events"]:
                    self.assertEqual(source[event["source_offset"]:event["source_end"]].strip(), event["text"])
                self.assertIn("VISUAL STYLE", brief["context"])
                self.assertNotIn("VISUAL STYLE", brief["events"][-1]["text"])
                # These sentence-initial adjectives modify energy/debris/hair.
                # They must not become three extra recurring characters.
                self.assertEqual(extract_h3_source_intent(source)["cast_names"], [])

    def test_explicit_numbered_shots_win_over_repeated_chapter_clock(self):
        source = "Chapter 1 [0s-18s] A desert action sequence.\n" + shooting_brief(
            lambda n, a, b: f"Shot {n} — 0:{a:02}–0:{b:02}"
        )
        self.assertEqual([e["text"] for e in authored_timed_brief(source)["events"]], ACTIONS)

    def test_windows_line_endings_preserve_shots_offsets_and_trailing_notes(self):
        source = shooting_brief(lambda n, a, b: f"SHOT {n} — 0:{a:02}–0:{b:02}").replace("\n", "\r\n")
        brief = authored_timed_brief(source)
        self.assertEqual([e["text"] for e in brief["events"]], ACTIONS)
        self.assertIn("VISUAL STYLE", brief["context"])
        self.assertIn("CAMERA", brief["context"])
        for event in brief["events"]:
            self.assertEqual(source[event["source_offset"]:event["source_end"]].strip(), event["text"])

    def test_ordinary_prose_and_invalid_ranges_do_not_become_a_shot_clock(self):
        for source in (
            'Alice says "Shot 1 — 0:00–0:02", then says "Shot 2 — 0:02–0:04".',
            "Shot 1 — 0:00–0:04\nHe runs.\nShot 2 — 0:02–0:06\nHe stops.",
            "Shot 1 — 0:00–0:02\nHe runs.\nShot 2 — 0:04–0:06\nHe stops.",
            "Shot 1 — 0:00–0:00\nHe runs.\nShot 2 — 0:00–0:06\nHe stops.",
        ):
            with self.subTest(source=source):
                self.assertEqual(authored_timed_brief(source)["events"], [])

    def test_adaptive_writer_receives_treatment_and_event_cards_not_a_new_schedule(self):
        source = shooting_brief(lambda n, a, b: f"SHOT {n} — 0:{a:02}–0:{b:02}")
        calls = []
        cursor = 0

        def generate(**kwargs):
            nonlocal cursor
            calls.append(kwargs)
            properties = kwargs["json_schema"]["properties"]
            self.assertNotIn("beats", properties)
            if "setting_continuity" in properties:
                return json.dumps({
                    "character_appearance": {}, "setting_continuity": "A barren desert under storm clouds.",
                    "visual_continuity": "Photorealistic skin, cloth and dust.",
                    "editing_style": "Fast tracking with motivated angle changes.",
                    "ambient_audio": "Desert wind and sand.",
                    "source_adaptation": "Retain the eight shots in order and expand their timing to the selected 28 seconds.",
                })
            segment = properties["segment"]["minimum"]
            cards = {}
            for key in properties["event_cards"]["required"]:
                cards[key] = {"phases": [{"transition": "cut", "framing": "Wide shot",
                    "camera": "Follow the action without losing its geography.",
                    "action": ACTIONS[cursor], "sound_effects": "Wind and sand."}]}
                cursor += 1
            return json.dumps({"segment": segment, "title": "Desert sequence",
                "opening_state": "Continue the established desert scene.",
                "coverage": "Clear action geography", "pacing": "Fast controlled movement",
                "closing_state": ACTIONS[cursor - 1], "event_cards": cards})

        with patch("services.llm_service.generate", side_effect=generate):
            result = plan_h3_sliding_windows(source, model_type="minimax_h3_fused_turbo",
                resolution="1280x704", total_frames=672, window_frames=345,
                overlap_frames=18, fps=24, planning_style="adaptive")
        self.assertEqual(len(calls), 3)
        self.assertEqual(cursor, 8)
        self.assertEqual(result["planning_warnings"], [])
        self.assertEqual(result["source_intent"]["cast_names"], [])
        prompts = "\n".join(result["window_prompts"])
        for action in ACTIONS:
            self.assertIn(action, prompts)
        self.assertIn(ACTIONS[-1], result["windows"][-1]["closing_state"])


if __name__ == "__main__":
    unittest.main()
