"""Regressions for the shared staged MiniMax H3 story contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_story_ledger import (  # noqa: E402
    extract_locked_dialogue,
    extract_source_events,
    ledger_violations,
    plan_h3_story_segments,
    sanitize_h3_prompt_text,
    segment_violations,
)
from services.h3_window_planner import (  # noqa: E402
    compile_h3_window_prompts,
    compute_h3_window_boundaries,
)


def _ledger() -> dict:
    return {
        "subject_continuity": 'Superman and Thanos remain unchanged {"S1": "Superman"}',
        "setting_continuity": "A damaged dark concrete structure",
        "visual_continuity": "Dark live-action cinematic realism",
        "editing_style": "Fast motivated action coverage",
        "initial_state": "Superman and Thanos face each other",
        "ambient_audio": "Wind and settling concrete dust",
        "music": "N/A",
        "required_final_outcome": "Thanos refuses Superman's demand",
        "beats": [
            {
                "beat_id": "B1",
                "segment": 1,
                "description": "Superman stands firm and demands the gauntlet",
                "source_event_ids": ["E1"],
                "dialogue_ids": ["D1"],
                "state_after": "Superman holds his ground while Thanos watches",
                "sound_effects": "Wind and settling grit",
            },
            {
                "beat_id": "B2",
                "segment": 2,
                "description": "Thanos raises the gauntlet and refuses",
                "source_event_ids": ["E2"],
                "dialogue_ids": ["D2"],
                "state_after": "Thanos holds the raised gauntlet toward Superman",
                "sound_effects": "The stones emit a low nonverbal hum",
            },
        ],
        "generated_dialogue": [],
    }


def _segment(number: int, *, duration: float = 10.0) -> dict:
    return {
        "segment": number,
        "title": f"Beat {number}",
        "opening_state": "The required opening composition",
        "coverage": "dynamic cinematic coverage",
        "pacing": "fast real-time action",
        "shots": [{
            "shot": 1,
            "start_seconds": 0.0,
            "end_seconds": duration,
            "transition": "opening composition",
            "framing": "medium two-shot",
            "camera": "a short motivated push in",
            "beat_ids": [f"B{number}"],
            "action": (
                "Superman stands firm and demands the gauntlet"
                if number == 1 else "Thanos raises the gauntlet and refuses"
            ),
            "dialogue": [{
                "dialogue_id": f"D{number}",
                "delivery": "calm and clear" if number == 1 else "breathy and resolute",
                "action": "holding eye contact",
            }],
            "sound_effects": "Natural synchronized effects",
        }],
        "closing_state": f"Proposed closing state {number}",
    }


class H3StoryLedgerTests(unittest.TestCase):
    def setUp(self):
        self.prompt = (
            'Superman says calmly, "Enough. Give me the glove, Thanos." '
            'Thanos raises his left hand and says in a breathy voice, "I am inevitable."'
        )
        self.locked = extract_locked_dialogue(self.prompt)

    def test_user_quotes_are_locked_with_speakers_and_order(self):
        self.assertEqual(
            [(item["dialogue_id"], item["speaker"], item["text"]) for item in self.locked],
            [
                ("D1", "Superman", "Enough. Give me the glove, Thanos."),
                ("D2", "Thanos", "I am inevitable."),
            ],
        )

    def test_quoted_title_is_not_misclassified_as_spoken_dialogue(self):
        prompt = 'A sitcom episode titled "The One With the Broken Robot" follows Alex repairing it.'
        self.assertEqual(extract_locked_dialogue(prompt), [])
        rendered = " ".join(item["text"] for item in extract_source_events(prompt))
        self.assertIn("The One With the Broken Robot", rendered)

    def test_ledger_rejects_repeated_events_and_dialogue(self):
        ledger = _ledger()
        ledger["beats"][1]["description"] = ledger["beats"][0]["description"]
        ledger["beats"][1]["dialogue_ids"] = ["D1", "D2"]
        violations = ledger_violations(
            self.prompt,
            ledger,
            segment_count=2,
            locked_dialogue=self.locked,
            expect_dialogue=True,
        )
        joined = " ".join(violations)
        self.assertIn("story event is duplicated", joined)
        self.assertIn("dialogue IDs are missing, duplicated", joined)

    def test_source_events_keep_the_requested_ending_and_drop_style_fragments(self):
        prompt = (
            "Superman and Thanos are trading punches. High-speed action movie dynamic superhero fight scenes. "
            "Superman punches Thanos through a wall. Superman looks at the gauntlet, and heat vision cuts off "
            "Thanos's arm, and Thanos yells as his knees buckle in defeat. Dark Cinematic. Sniderverse style. "
            "rated-r graphic, realistic film scene."
        )
        events = extract_source_events(prompt)
        rendered = " | ".join(item["text"] for item in events)
        self.assertIn("heat vision cuts off Thanos's arm", rendered)
        self.assertIn("Thanos yells as his knees buckle in defeat", events[-1]["text"])
        self.assertNotIn("Dark Cinematic", rendered)
        self.assertNotIn("Sniderverse style", rendered)
        self.assertNotIn("rated-r", rendered)

    def test_segment_rejects_tiny_tail_and_repeated_or_foreign_beats(self):
        segment = _segment(1)
        segment["shots"] = [
            {
                **segment["shots"][0],
                "end_seconds": 9.9,
            },
            {
                **segment["shots"][0],
                "shot": 2,
                "start_seconds": 9.9,
                "end_seconds": 10.0,
                "transition": "hard cut",
                "beat_ids": ["B2"],
                "dialogue": [],
            },
        ]
        violations = segment_violations(
            self.prompt,
            segment,
            segment_number=1,
            duration=10.0,
            assigned_beats=[_ledger()["beats"][0]],
            dialogue_catalog=self.locked,
        )
        joined = " ".join(violations)
        self.assertIn("assigned beat IDs are missing, foreign, or repeated", joined)
        self.assertIn("unusably short tail shot", joined)

    def test_staged_planner_keeps_dialogue_exact_and_repairs_only_bad_segment(self):
        invalid_second = _segment(2)
        invalid_second["shots"][0]["start_seconds"] = 9.9
        invalid_second["shots"][0]["end_seconds"] = 10.0
        responses = iter([
            json.dumps(_ledger()),
            json.dumps(_segment(1)),
            json.dumps(invalid_second),
            json.dumps(_segment(2)),
        ])
        calls: list[dict] = []

        def generate(**kwargs):
            calls.append(kwargs)
            return next(responses)

        result = plan_h3_story_segments(
            self.prompt,
            segment_durations=[10.0, 10.0],
            mode="reference_sequence",
            camera_coverage="multi_shot",
            expect_dialogue=True,
            llm_generate=generate,
        )
        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(len(calls), 4)
        self.assertNotIn("REPAIR ONLY THIS SEGMENT", calls[1]["prompt"])
        self.assertIn("REPAIR ONLY THIS SEGMENT", calls[3]["prompt"])
        rendered = json.dumps(result["segments"], ensure_ascii=False)
        self.assertEqual(rendered.count("Enough. Give me the glove, Thanos."), 1)
        self.assertEqual(rendered.count("I am inevitable."), 1)
        self.assertEqual(result["segments"][1]["opening_state"], result["segments"][0]["closing_state"])
        # The ledger owns the close; the camera response cannot silently alter it.
        self.assertEqual(
            result["segments"][1]["closing_state"],
            "Thanos holds the raised gauntlet toward Superman",
        )

    def test_compiler_neutralizes_braces_and_nested_context_labels(self):
        spans = compute_h3_window_boundaries(480, 240, fps=24, overlap_frames=0)
        plan = {
            "subject_continuity": 'Heroes {"S1": "Superman"}; summary: never nest this',
            "setting_continuity": "The same arena",
            "visual_continuity": "Cinematic realism",
            "editing_style": "Motivated cuts",
            "initial_state": "The fighters face each other",
            "ambient_audio": "Wind",
            "music": "N/A",
            "windows": [
                {
                    "window": index + 1,
                    "title": f"Beat {index + 1}",
                    "coverage": "cinematic coverage",
                    "pacing": "real-time",
                    "shots": [{
                        "shot": 1,
                        "start_seconds": 0.0,
                        "end_seconds": 10.0,
                        "transition": "opening composition",
                        "framing": "medium shot",
                        "camera": "locked camera",
                        "action": f"Action {index + 1}",
                        "dialogue": [],
                        "sound_effects": "N/A",
                    }],
                    "closing_state": f"State {index + 1}",
                }
                for index in range(2)
            ],
        }
        compiled = compile_h3_window_prompts(plan, spans)
        for item in compiled:
            prompt = item["prompt"]
            self.assertNotIn("{", prompt)
            self.assertNotIn("}", prompt)
            self.assertNotIn("summary:", prompt)
            self.assertEqual(prompt.count("integrated_multimodal_description:"), 1)
            self.assertEqual(prompt.count("overall_soundscape:"), 1)
            self.assertEqual(prompt.count("non_diegetic_music:"), 1)

    def test_sanitizer_neutralizes_template_and_context_ir_syntax(self):
        value = sanitize_h3_prompt_text('{"S1": "Neo"} detailed_description: action')
        self.assertEqual(value, '("S1": "Neo") detailed_description - action')


if __name__ == "__main__":
    unittest.main()
