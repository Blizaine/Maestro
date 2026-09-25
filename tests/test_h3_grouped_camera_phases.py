"""Focused controls for preserving adaptive silent source groups."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_camera_fidelity import (  # noqa: E402
    clear_confirmed_coverage_errors,
    review_missing_camera_actions,
)
from services.h3_story_ledger import (  # noqa: E402
    _camera_phase_beats,
    _grouped_source_order_error,
    extract_h3_source_intent,
    extract_source_events,
    segment_violations,
)


def _cast_pattern(prompt: str) -> re.Pattern[str] | None:
    names = extract_h3_source_intent(prompt).get("cast_names") or []
    if not names:
        return None
    return re.compile(
        r"(?<![\w])(?:" + "|".join(
            re.escape(name) for name in sorted(names, key=len, reverse=True)
        ) + r")(?![\w])",
        re.IGNORECASE,
    )


def _group(prompt: str, *, beat: dict | None = None, **phase_options):
    events = extract_source_events(prompt)
    assigned = beat or {
        "beat_id": "B1",
        "segment": 1,
        "source_event_ids": [event["event_id"] for event in events],
        "dialogue_ids": [],
        "description": "A connected, unbroken sequence in the writer's original staging.",
        "state_after": "The final source action is complete.",
    }
    phases = _camera_phase_beats(
        [assigned], source_events=events, expected_dialogue_events={},
        preserve_adaptation=True, **phase_options,
    )
    return events, phases


def _segment(action: str, beat: dict) -> dict:
    return {
        "segment": 1,
        "semantic_actions": True,
        "camera_contract": "event_cards",
        "shots": [{
            "shot": 1,
            "start_seconds": 0.0,
            "end_seconds": 12.0,
            "transition": "continue the same take",
            "framing": "a continuous medium-wide composition",
            "camera": "track the assigned physical actions",
            "beat_ids": [beat["beat_id"]],
            "action": action,
            "dialogue": [],
            "sound_effects": "natural synchronized effects",
        }],
        "closing_state": "The assigned source actions are complete.",
    }


class GroupedCameraPhaseTests(unittest.TestCase):
    def test_preserves_original_staging_only_for_eligible_group(self):
        prompt = "Mara unlocks the blue cabinet. Ivo retrieves the map."
        events, phases = _group(prompt)

        self.assertEqual(len(events), 2)
        self.assertEqual(len(phases), 1)
        phase = phases[0]
        self.assertEqual(phase["_grouped_source_event_ids"], ["E1", "E2"])
        self.assertEqual(phase["source_event_ids"], ["E1", "E2"])
        self.assertIn("original staging", phase["_staging_context"])
        self.assertIn("Mara unlocks", phase["description"])
        self.assertIn("Ivo retrieves", phase["description"])

    def test_keeps_timed_dialogue_recurring_and_audio_events_split(self):
        cases = [
            ("[0s-2s] Mara opens the blue case. [2s-4s] Ivo retrieves the map.", {}, False),
            ("Mara opens the blue case. Ivo retrieves the map.",
             {"beat": {"beat_id": "B1", "segment": 1,
                        "source_event_ids": ["E1", "E2"], "dialogue_ids": ["D1"],
                        "description": "The original staging."},
              "expected_dialogue_events": {"D1": "E1"}}, False),
            ("Every evening Mara opens the blue case. Ivo retrieves the map.", {}, False),
            ("Mara says a line. Ivo retrieves the map.", {}, False),
            ("Mara opens the blue case. Ivo retrieves the map.", {"audio_driven": True}, False),
        ]
        for index, (prompt, options, expected_grouping) in enumerate(cases):
            with self.subTest(index=index, prompt=prompt):
                events = extract_source_events(prompt)
                if "beat" in options:
                    beat = options.pop("beat")
                    dialogue = options.pop("expected_dialogue_events")
                    phases = _camera_phase_beats(
                        [beat], source_events=events,
                        expected_dialogue_events=dialogue,
                        preserve_adaptation=True, **options,
                    )
                else:
                    _, phases = _group(prompt, **options)
                is_grouped = any(phase.get("_grouped_source_event_ids") for phase in phases)
                self.assertEqual(is_grouped, expected_grouping)
                if not expected_grouping:
                    self.assertGreaterEqual(len(phases), 2)

    def test_explicit_reversal_is_hard_and_ordered_actions_pass(self):
        prompt = "Mara unlocks the blue cabinet. Ivo retrieves the map."
        events, phases = _group(prompt)
        beat = phases[0]
        cast = _cast_pattern(prompt)

        reversed_error = _grouped_source_order_error(
            beat, events,
            "Ivo retrieves the map. Mara unlocks the blue cabinet.", cast,
        )
        self.assertIn("visibly out of order", reversed_error)
        self.assertNotIn("shot action omits required source step", reversed_error)
        reversed_plan_errors = segment_violations(
            prompt,
            _segment("Ivo retrieves the map. Mara unlocks the blue cabinet.", beat),
            segment_number=1, duration=12.0,
            assigned_beats=phases, dialogue_catalog=[],
        )
        self.assertTrue(any("visibly out of order" in item for item in reversed_plan_errors))
        self.assertIsNone(_grouped_source_order_error(
            beat, events,
            "Mara unlocks the blue cabinet. Ivo retrieves the map.", cast,
        ))

    def test_enabling_interleave_requires_full_group_receipt(self):
        prompt = (
            "Mara carries the red crate along the corridor and into the workshop. "
            "Ivo opens the workshop door for her."
        )
        events, phases = _group(prompt)
        beat = phases[0]
        action = (
            "Mara begins carrying the red crate. Ivo opens the workshop door. "
            "Mara carries the red crate through."
        )
        order_error = _grouped_source_order_error(
            beat, events, action, _cast_pattern(prompt),
        )
        self.assertIn("shot action omits required source step", order_error)
        for event in events:
            self.assertIn(event["event_id"] + ": " + event["text"], order_error)
        self.assertIn(" Then ", order_error)

        segment = _segment(action, beat)
        errors = segment_violations(
            prompt, segment, segment_number=1, duration=12.0,
            assigned_beats=phases, dialogue_catalog=[],
        )
        self.assertIn(order_error, errors)

        received_prompt = {}

        def reject_unproven_order(**kwargs):
            checks = json.loads(kwargs["prompt"])
            received_prompt.update(checks)
            return json.dumps({
                key: {
                    "action_decisions": {
                        item["action_id"]: {
                            "verdict": "missing", "evidence_span_ids": [],
                        }
                        for item in check.get("action_obligations", [])
                    },
                }
                for key in checks
                for check in [checks[key]]
            })

        receipts = review_missing_camera_actions(
            errors, segment, assigned_beats=phases,
            source_events=events, generate=reject_unproven_order,
        )
        self.assertEqual(receipts, {})
        grouped_check = next(
            check for check in received_prompt.values()
            if "Preserve the full ordered source group" in check["source_requirement"]
        )
        self.assertIn(events[0]["text"], grouped_check["source_requirement"])
        self.assertIn(events[1]["text"], grouped_check["source_requirement"])
        self.assertEqual(grouped_check["source_event"], " ".join(
            event["text"] for event in events
        ))

        def accept_with_ordered_visual_spans(**kwargs):
            checks = json.loads(kwargs["prompt"])
            result = {}
            for key, check in checks.items():
                if "Preserve the full ordered source group" in check["source_requirement"]:
                    action_ids = [
                        span["span_id"] for span in check["visual_spans"]
                        if span["field"] == "action"
                    ]
                    result[key] = {
                        "action_decisions": {
                            item["action_id"]: {
                                "verdict": "preserved",
                                "evidence_span_ids": action_ids[:3],
                            }
                            for item in check["action_obligations"]
                        },
                    }
                else:
                    result[key] = {
                        "action_decisions": {
                            item["action_id"]: {
                                "verdict": "missing", "evidence_span_ids": [],
                            }
                            for item in check.get("action_obligations", [])
                        },
                    }
            return json.dumps(result)

        accepted = review_missing_camera_actions(
            errors, segment, assigned_beats=phases,
            source_events=events, generate=accept_with_ordered_visual_spans,
        )
        remaining = clear_confirmed_coverage_errors(errors, segment, accepted)
        self.assertNotIn(order_error, remaining)

    def test_close_before_return_is_not_silently_accepted(self):
        prompt = (
            "Ada unlocks the archive door, opens it, enters, retrieves the binder, "
            "and returns to the corridor. Then Ada closes and locks the door."
        )
        events, phases = _group(prompt)
        self.assertEqual(len(events), 3)
        self.assertEqual(len(phases), 1)
        action = (
            "Ada unlocks the archive door, opens it, and enters. "
            "Then Ada closes and locks the door. She retrieves the binder "
            "and returns to the corridor."
        )
        error = _grouped_source_order_error(
            phases[0], events, action, _cast_pattern(prompt),
        )
        self.assertTrue(
            "visibly out of order" in error
            or "Preserve the full ordered source group" in error,
            error,
        )
        plan_errors = segment_violations(
            prompt, _segment(action, phases[0]), segment_number=1,
            duration=12.0, assigned_beats=phases, dialogue_catalog=[],
        )
        self.assertTrue(any(
            "visibly out of order" in item
            or "Preserve the full ordered source group" in item
            for item in plan_errors
        ))

    def test_wrong_actor_and_distinct_wrong_object_remain_omissions(self):
        actor_prompt = "Mara opens the blue cabinet."
        actor_events = extract_source_events(actor_prompt)
        actor_beat = {
            "beat_id": "B1", "source_event_ids": ["E1"],
            "dialogue_ids": [], "description": actor_events[0]["text"],
        }
        actor_errors = segment_violations(
            actor_prompt,
            _segment("Ivo opens the blue cabinet.", actor_beat),
            segment_number=1, duration=12.0,
            assigned_beats=[actor_beat], dialogue_catalog=[],
        )
        self.assertTrue(any("omits required source step" in item for item in actor_errors))

        object_prompt = "Mara places the blue folder on the shelf."
        object_events = extract_source_events(object_prompt)
        object_beat = {
            "beat_id": "B1", "source_event_ids": ["E1"],
            "dialogue_ids": [], "description": object_events[0]["text"],
        }
        object_errors = segment_violations(
            object_prompt,
            _segment("Mara places the red box on the table.", object_beat),
            segment_number=1, duration=12.0,
            assigned_beats=[object_beat], dialogue_catalog=[],
        )
        self.assertTrue(any("omits required source step" in item for item in object_errors))


if __name__ == "__main__":
    unittest.main()
