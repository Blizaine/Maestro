"""Bounded H3 recovery keeps source order and accepted windows intact."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_plan_retry import (
    H3PlanRetryError,
    finish_retry_plan,
    validate_retry_plan,
)
from services.h3_story_ledger import (
    _canonicalize_segment_contract,
    _expand_camera_event_cards,
    plan_h3_story_segments,
)
from services.h3_window_planner import _fallback_plan


SOURCE = (
    "[0s-4s] Nora raises her right elbow, plants her left foot firmly on the wooden "
    "floor, drops her shoulder into an elbow strike against Eli's padded guard, lets "
    "the contact stop her forward movement, turns her hips to the left, steps around "
    "the red bench without touching it, lowers her right arm, shifts her weight onto "
    "both feet, looks down at the brass key, reaches toward it with her left hand, "
    "lifts it from the bench, closes her fingers around it, keeps her palm facing "
    "inward, brings the key to her chest, steadies it with her right hand, turns "
    "toward the blue gate, takes two measured steps in that direction, stops beside "
    "the gate, and keeps the key visible in her left hand while the sunlight falls "
    "through a high narrow window onto plain wooden shelves, a green cabinet, a folded "
    "blue cloth, and a white ceramic bowl beside the bench. "
    "[4s-8s] Eli places the brass key on the red bench. "
    "[8s-12s] Nora opens the blue gate. No dialogue."
)


def _schema_value(schema, field=""):
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {
            key: _schema_value(schema["properties"][key], key)
            for key in schema.get("required", [])
        }
    if kind == "array":
        return [
            _schema_value(schema["items"])
            for _ in range(schema.get("minItems", 0))
        ]
    if kind in {"integer", "number"}:
        return schema.get("minimum", 1)
    return {
        "coverage": "Readable camera coverage with a stable screen axis",
        "pacing": "Natural real-time movement",
        "framing": "A lateral medium view of the red bench",
        "camera": "A restrained lateral camera move",
        "transition": "continuous reframe",
        "sound_effects": "Quiet room tone",
    }.get(field, "Source camera test")


def _retry_plan():
    windows = [
        {"prompt": "accepted prompt with exact spacing  \n", "title": "Accepted"},
        {"prompt": "window two needs review", "title": "Review"},
    ]
    prompts = [window["prompt"] for window in windows]
    return {
        "retry_fingerprint": "fingerprint",
        "windows": windows,
        "window_prompts": prompts,
        "camera_checkpoint": {
            "version": 1,
            "context": {"durations": [4.0, 4.0]},
            "segments": [{"segment": 1}, {"segment": 2}],
            "review_windows": [2],
        },
        "planning_warnings": ["Window 2 needs review"],
    }


class H3PlanRecoveryTests(unittest.TestCase):
    def test_incomplete_saved_schedule_is_rejected_without_mutating_draft(self):
        saved = _retry_plan()
        saved["windows"].pop()
        before = deepcopy(saved)

        with self.assertRaisesRegex(H3PlanRetryError, "schedule is incomplete or malformed"):
            validate_retry_plan(saved, fingerprint="fingerprint", count=2)

        self.assertEqual(saved, before)

    def test_incomplete_repair_is_actionable_and_does_not_replace_accepted_prompt(self):
        previous = _retry_plan()
        result = {
            "window_count": 1,
            "windows": [{"prompt": "replacement"}],
            "window_prompts": ["replacement"],
        }
        before = deepcopy(previous)

        with self.assertRaisesRegex(H3PlanRetryError, "saved draft is unchanged"):
            finish_retry_plan(result, previous, "fingerprint")

        self.assertEqual(previous, before)
        self.assertEqual(result["windows"][0]["prompt"], "replacement")

    def test_empty_repair_schedule_is_not_accepted(self):
        previous = _retry_plan()
        before = deepcopy(previous)

        with self.assertRaisesRegex(H3PlanRetryError, "saved draft is unchanged"):
            finish_retry_plan({"windows": [], "window_prompts": []}, previous, "fingerprint")

        self.assertEqual(previous, before)

    def test_repair_keeps_accepted_window_prompt_byte_for_byte(self):
        previous = _retry_plan()
        regenerated = {
            "window_count": 2,
            "windows": [
                {"prompt": "rewritten first window", "title": "New"},
                {"prompt": "repaired second window", "title": "Fixed"},
            ],
            "window_prompts": ["rewritten first window", "repaired second window"],
        }

        repaired = finish_retry_plan(regenerated, previous, "fingerprint")

        self.assertEqual(repaired["windows"][0], previous["windows"][0])
        self.assertEqual(repaired["window_prompts"][0], previous["window_prompts"][0])
        self.assertEqual(repaired["windows"][1]["prompt"], "repaired second window")

    def test_fallback_uses_timed_source_events_instead_of_one_raw_paragraph(self):
        source = (
            "[0s-2s] The courier opens the gate "
            "[2s-4s] The courier crosses the yard "
            "[4s-6s] The courier puts the red bag on the bench "
            "[6s-8s] The courier opens the red bag"
        )

        plan = _fallback_plan(source, 4, window_durations=[2.0] * 4)
        actions = [window["action"] for window in plan["windows"]]
        action_text = "\n".join(actions)
        source_actions = [
            "The courier opens the gate",
            "The courier crosses the yard",
            "The courier puts the red bag on the bench",
            "The courier opens the red bag",
        ]

        self.assertEqual(len(plan["windows"]), 4)
        positions = []
        for action in source_actions:
            self.assertEqual(action_text.count(action), 1, action_text)
            positions.append(action_text.index(action))
        self.assertEqual(positions, sorted(positions))

    def test_event_indices_must_be_integers_and_remain_in_source_order(self):
        beats = [
            {"beat_id": "B1", "description": "Nora opens the case.", "source_event_ids": ["E1"]},
            {"beat_id": "B2", "description": "Nora removes the map.", "source_event_ids": ["E2"]},
        ]

        def canonicalize(indices):
            segment = {
                "segment": 1,
                "coverage": "single continuous shot",
                "pacing": "real time",
                "closing_state": "Nora holds the map beside the open case.",
                "shots": [
                    {
                        "event_indices": [value], "dialogue_ids": [],
                        "start_seconds": index * 2.0, "end_seconds": (index + 1) * 2.0,
                        "transition": "continuous reframe", "framing": "medium view",
                        "camera": "steady tracking", "action": action,
                        "sound_effects": "quiet room tone",
                    }
                    for index, (value, action) in enumerate(zip(indices, (
                        "Nora opens the case.", "Nora removes the map.",
                    )))
                ],
            }
            return _canonicalize_segment_contract(
                segment, segment_number=1, duration=4.0,
                assigned_beats=beats, dialogue_catalog=[], opening_state="Nora faces the case.",
                source_intent={},
            )

        ordered = canonicalize([1, 2])
        self.assertEqual([shot["event_indices"] for shot in ordered["shots"]], [[1], [2]])
        self.assertNotIn("event_assignment_error", ordered)

        numeric_strings = canonicalize(["1", "2"])
        self.assertIn("event_assignment_error", numeric_strings)

        reversed_plan = canonicalize(["2", "1"])
        self.assertIn("event_assignment_error", reversed_plan)

    def test_camera_event_metadata_is_ignored_but_required_cards_stay_ordered(self):
        beats = [
            {"beat_id": "B1", "description": "Nora opens the case.", "source_event_ids": ["E1"]},
        ]
        segment = {
            "segment": 1,
            "title": "One action",
            "coverage": "continuous",
            "pacing": "real time",
            "provider_trace": {"id": "ignored"},
            "event_cards": {
                "event_1": {
                    "phases": [{
                        "action": "Nora opens the case.",
                        "framing": "Medium view",
                        "camera": "A gentle push toward the case",
                        "transition": "continuous reframe",
                        "sound_effects": "Quiet room tone",
                        "provider_note": "ignored",
                    }],
                    "provider_note": "ignored",
                },
            },
        }

        expanded = _expand_camera_event_cards(
            segment, assigned_beats=beats, segment_number=1, duration=4.0,
        )

        self.assertEqual(len(expanded["shots"]), 1)
        self.assertEqual(expanded["shots"][0]["event_indices"], [1])
        self.assertEqual(expanded["shots"][0]["action"], "Nora opens the case.")
        self.assertEqual(expanded["shots"][0]["camera"], "A gentle push toward the case")

        foreign_card = deepcopy(segment)
        foreign_card["event_cards"]["event_99"] = {"invented_action": "not ignorable"}
        with self.assertRaisesRegex(ValueError, "Write every required event card"):
            _expand_camera_event_cards(
                foreign_card, assigned_beats=beats, segment_number=1, duration=4.0,
            )

    def test_failed_focused_repair_recovers_only_bad_source_camera_card(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            schema = kwargs["json_schema"]
            properties = schema.get("properties", {})
            review_keys = [key for key in properties if key.startswith("check_")]
            if review_keys:
                payload = json.loads(kwargs["prompt"])
                decisions = {}
                for key in review_keys:
                    check_schema = properties[key]["properties"]
                    action_schema = check_schema.get("action_decisions", {}).get("properties")
                    check = payload[key]
                    spans = check.get("visual_spans", [])
                    if isinstance(action_schema, dict):
                        decisions[key] = {
                            "action_decisions": {
                                action_id: {
                                    "verdict": "preserved",
                                    "evidence_span_ids": [spans[0]["span_id"]] if spans else [],
                                }
                                for action_id in action_schema
                            },
                        }
                    elif "verdict" in check_schema:
                        requirement = str(check.get("source_requirement") or "").casefold()
                        evidence = next((
                            span["span_id"] for span in spans
                            if span.get("field") == "action"
                            and requirement and requirement in str(span.get("text") or "").casefold()
                        ), None)
                        decisions[key] = {
                            "verdict": "preserved" if evidence else "missing",
                            "evidence_span_ids": [evidence] if evidence else [],
                        }
                return json.dumps(decisions)
            if "setting_continuity" in properties:
                return json.dumps({
                    "character_appearance": "Nora and Eli in plain clothing",
                    "setting_continuity": "A wooden room with a red bench and blue gate.",
                    "visual_continuity": "Natural live action",
                    "editing_style": "Readable coverage",
                    "motion_mechanics": "Physical actions with clear contact",
                    "ambient_audio": "Quiet room tone",
                })
            if "segment" not in properties or "event_cards" not in properties:
                raise AssertionError("Unexpected camera writer schema")
            if "REPAIR ONLY THIS SEGMENT" in kwargs["prompt"]:
                raise RuntimeError("repair service unavailable")
            draft = _schema_value(schema)
            if draft["segment"] == 1:
                draft["segment"] = str(draft["segment"])
                for phase in draft["event_cards"]["event_1"].values():
                    phase["camera"] = "Camera keeps the red bench in view"
                    phase["provider_note"] = "ignored"
                event_two_phase = next(iter(draft["event_cards"]["event_2"].values()))
                event_two_phase["camera"] = "Nora opens the blue gate."
                draft["provider_trace"] = {"id": "ignored"}
            return json.dumps(draft)

        with (
            patch("services.studio_enhancement.fidelity_retry_limit", return_value=1),
            patch("services.h3_camera_repair.camera_repair_advice", return_value=""),
        ):
            result = plan_h3_story_segments(
                SOURCE,
                segment_durations=[8.0, 4.0],
                mode="sliding_window",
                camera_coverage="multi_shot",
                expect_dialogue=False,
                planning_style="faithful",
                llm_generate=generate,
            )

        first_actions = " ".join(shot["action"] for shot in result["segments"][0]["shots"])
        second_actions = " ".join(shot["action"] for shot in result["segments"][1]["shots"])
        self.assertTrue(any("REPAIR ONLY THIS SEGMENT" in call["prompt"] for call in calls))
        self.assertTrue(any("Camera keeps the red bench in view" in shot["camera"]
                            for shot in result["segments"][0]["shots"]))
        self.assertNotIn("Nora opens the blue gate", first_actions)
        self.assertIn("Nora opens the blue gate", second_actions)
        self.assertEqual(result["camera_checkpoint"]["review_windows"], [1])
        self.assertTrue(any("Focused camera repair failed" in item
                            and "repair service unavailable" in item
                            for item in result["planning_diagnostics"]))
        self.assertTrue(any("preserved the other camera cards" in item
                            for item in result["planning_warnings"]))


if __name__ == "__main__":
    unittest.main()
