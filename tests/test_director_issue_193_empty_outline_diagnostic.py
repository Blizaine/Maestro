"""Empty music-video outlines receive a useful, evidence-based diagnostic."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.planners.music_video import MusicVideoPlanner


DIAGNOSTIC = (
    "No usable structured outline was returned after JSON repair. "
    "Check the adjacent LLM completion telemetry; if it shows "
    "finish=length and answer=0 with reasoning consuming the request limit, "
    "adjust the provider reasoning/output-token settings. Missing actions were not inferred."
)


def _clips(count: int) -> list[dict]:
    return [
        {"start": index, "end": index + 1, "label": "verse"}
        for index in range(count)
    ]


def _empty_contract() -> dict:
    return {
        "source": "A traveler crosses the country.",
        "events": [],
    }


def _planner(response_text: str):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return response_text

    planner = MusicVideoPlanner(llm_generate=generate)
    planner._music_narrative_contract = _empty_contract()
    planner._use_lyrics_context = False
    return planner, calls


class DirectorIssue193EmptyOutlineDiagnosticTests(unittest.TestCase):
    def test_empty_array_keeps_original_issue_and_uses_only_bounded_attempts(self):
        planner, calls = _planner("[]")

        with self.assertRaises(RuntimeError) as raised:
            planner._plan_music_arc(_clips(2), "A traveler crosses the country.")

        message = str(raised.exception)
        self.assertIn("received 0", message)
        self.assertIn(DIAGNOSTIC, message)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("music_creative_arc", planner._planning_resume_checkpoint)

    def test_empty_beats_container_is_recognized_but_nonempty_invalid_outline_is_not(self):
        for payload, expect_diagnostic in (
            ({"beats": []}, True),
            ({"beats": [{"wrong": "shape"}]}, False),
        ):
            with self.subTest(expect_diagnostic=expect_diagnostic):
                planner, calls = _planner(json.dumps(payload))
                with self.assertRaises(RuntimeError) as raised:
                    planner._plan_music_arc(_clips(1), "A traveler crosses the country.")

                self.assertEqual(len(calls), 2)
                self.assertEqual(DIAGNOSTIC in str(raised.exception), expect_diagnostic)

    def test_valid_outline_is_unchanged_and_needs_one_generation_call(self):
        row = {
            "clip_index": 0,
            "visual_beat": "The traveler crosses a wooden bridge into the town.",
            "purpose": "advance the visual progression",
            "environment": "a country road and wooden bridge",
            "camera": "locked medium-wide frame",
            "source_event_ids": [],
            "intentional_reprise": False,
            "reprise_reason": "",
        }
        planner, calls = _planner(json.dumps({"beats": [row]}))

        result = planner._plan_music_arc(_clips(1), "A traveler crosses the country.")

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["visual_beat"], row["visual_beat"])
        self.assertNotIn(DIAGNOSTIC, str(result))


if __name__ == "__main__":
    unittest.main()
