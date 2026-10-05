"""Bounded camera coverage reviews require complete, scoped visual evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
from services import h3_camera_fidelity as candidate


def action_sentences(count: int) -> str:
    verbs = [
        "opens the gate", "lifts the lantern", "places the lantern on the table",
        "closes the gate", "writes a note", "hands the note to Niko",
        "walks to the fountain", "turns the wheel", "unlocks the door",
        "opens the cabinet", "sets the cup on the shelf", "closes the cabinet",
        "lifts the basket", "carries the basket to the bench", "sets the basket down",
        "turns the key", "opens the shed", "walks into the shed",
    ]
    return " ".join(f"Mira {verbs[index % len(verbs)]}." for index in range(count))


def make_segment(beat_id: str, action: str, *, segment_number: int = 1) -> dict:
    return {
        "camera_contract": "event_cards",
        "segment": segment_number,
        "shots": [{
            "beat_ids": [beat_id],
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "action": action,
            "camera": "Follow the visible action in a steady medium view.",
            "framing": "Medium view of the whole action.",
        }],
    }


def make_review_fixture(requirement: str, *, beat_id: str = "B1", event_id: str = "E1"):
    error = f"{beat_id} shot action omits required source step: {requirement}"
    beats = [{"beat_id": beat_id, "source_event_ids": [event_id]}]
    events = [{"event_id": event_id, "text": requirement}]
    segment = make_segment(beat_id, requirement)
    return error, beats, events, segment


def preserved_response(kwargs: dict) -> str:
    request = json.loads(kwargs["prompt"])
    response = {}
    for check_id, check in request.items():
        spans = check["visual_spans"]
        obligations = check.get("action_obligations") or []
        contract = check.get("timed_hold_contract")
        if obligations:
            span_id = spans[0]["span_id"]
            response[check_id] = {
                "action_decisions": {
                    item["action_id"]: {
                        "verdict": "preserved",
                        "evidence_span_ids": [span_id],
                    } for item in obligations
                }
            }
        elif contract:
            windows = contract["windows"]
            response[check_id] = {
                "window_decisions": {
                    f"window_{item['window']}": {
                        "verdict": "preserved",
                        "evidence_span_ids": [next(
                            span["span_id"] for span in spans
                            if span.get("window") == item["window"]
                        )],
                    } for item in windows
                }
            }
        else:
            response[check_id] = {
                "verdict": "preserved",
                "evidence_span_ids": [spans[0]["span_id"]],
            }
    return json.dumps(response)


def assert_metadata_only_cards(test_case: unittest.TestCase, packet: dict) -> None:
    cards = packet["visual_cards"]
    metadata_fields = {"card", "window", "local_start_seconds", "local_end_seconds"}
    text_fields = {"action", "framing", "camera"}
    test_case.assertTrue(all(set(card) <= metadata_fields for card in cards))
    test_case.assertTrue(all(not (set(card) & text_fields) for card in cards))


class CameraFidelityBatchTests(unittest.TestCase):
    def test_nine_actions_split_into_stable_eight_and_one_batches(self):
        requirement = action_sentences(9)
        error, beats, events, segment = make_review_fixture(requirement)
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate,
        )

        self.assertEqual(len(calls), 2)
        action_ids = []
        span_ids = []
        for call in calls:
            packet = json.loads(call["prompt"])["check_1"]
            self.assertLessEqual(len(packet["action_obligations"]), 8)
            assert_metadata_only_cards(self, packet)
            action_ids.extend(item["action_id"] for item in packet["action_obligations"])
            span_ids.append([item["span_id"] for item in packet["visual_spans"]])
        self.assertEqual(action_ids, [f"action_{index}" for index in range(1, 10)])
        self.assertEqual(span_ids[0], span_ids[1])
        self.assertEqual(candidate.clear_confirmed_coverage_errors([error], segment, receipts), [])

    def test_window_budget_is_shared_across_review_invocations(self):
        requirement = "Mira unlocks the gate."
        error, beats, events, segment = make_review_fixture(requirement)
        budget = candidate.CameraReviewBudget(max_requests=2)
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        for _ in range(2):
            receipts = candidate.review_missing_camera_actions(
                [error], segment, assigned_beats=beats, source_events=events,
                generate=generate, review_budget=budget,
            )
            self.assertIn(error, receipts)
        self.assertEqual(len(calls), 2)
        self.assertEqual(budget.requests_dispatched, 2)

        feedback = {}
        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, repair_feedback=feedback, review_budget=budget,
        )
        self.assertEqual(len(calls), 2, "An exhausted shared budget must not dispatch another review")
        self.assertEqual(receipts, {})
        self.assertTrue(any(requirement in item for item in feedback[error]))
        self.assertTrue(any("budget of 2 per camera window was exhausted" in item
                            for item in feedback[error]))

    def test_budget_exhaustion_never_approves_a_partially_reviewed_compound(self):
        requirement = action_sentences(9)
        error, beats, events, segment = make_review_fixture(requirement)
        feedback = {}
        calls = []
        budget = candidate.CameraReviewBudget(max_requests=1)

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, repair_feedback=feedback, review_budget=budget,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(budget.requests_dispatched, 1)
        self.assertEqual(receipts, {})
        self.assertEqual(candidate.clear_confirmed_coverage_errors(
            [error], segment, receipts), [error])
        unresolved = "\n".join(feedback[error])
        self.assertIn("Mira unlocks the door", unresolved)
        self.assertIn("budget of 1 per camera window was exhausted", unresolved)

    def test_changed_cards_are_reviewed_while_shared_budget_remains(self):
        requirement = "Mira unlocks the gate."
        error, beats, events, segment = make_review_fixture(requirement)
        calls = []
        budget = candidate.CameraReviewBudget(max_requests=2)

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        first_receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, review_budget=budget,
        )
        self.assertEqual(len(calls), 1)
        segment["shots"][0]["action"] = (
            "Mira turns a brass key until the gate unlocks and swings open."
        )
        self.assertEqual(candidate.clear_confirmed_coverage_errors(
            [error], segment, first_receipts), [error],
            "A receipt for old visual text must expire after a card changes",
        )

        second_receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, review_budget=budget,
        )
        self.assertEqual(len(calls), 2, "Changed evidence must use the remaining request budget")
        packet = json.loads(calls[1]["prompt"])["check_1"]
        self.assertTrue(any("brass key" in span["text"] for span in packet["visual_spans"]))
        self.assertEqual(candidate.clear_confirmed_coverage_errors(
            [error], segment, second_receipts), [])
        self.assertEqual(budget.requests_dispatched, 2)

    def test_payload_keeps_exact_span_evidence_without_duplicate_card_prose(self):
        requirement = "Mira unlocks the gate."
        error, beats, events, segment = make_review_fixture(requirement)
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate,
        )

        packet = json.loads(calls[0]["prompt"])["check_1"]
        assert_metadata_only_cards(self, packet)
        expected_full_cards = candidate._coverage_cards(
            segment, "B1", {"B1": {"source_event_ids": ["E1"]}},
        )
        expected_card_metadata = [
            {name: card[name] for name in (
                "card", "window", "local_start_seconds", "local_end_seconds",
            ) if name in card}
            for card in expected_full_cards
        ]
        expected_spans = candidate._visual_spans(expected_full_cards, "check_1")
        self.assertEqual(packet["visual_cards"], expected_card_metadata)
        self.assertEqual(packet["visual_spans"], expected_spans)
        self.assertEqual(candidate.clear_confirmed_coverage_errors(
            [error], segment, receipts), [])

    def test_later_missing_action_keeps_original_error_and_names_that_substep(self):
        requirement = action_sentences(9)
        error, beats, events, segment = make_review_fixture(requirement)
        repair_feedback = {}
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            request = json.loads(kwargs["prompt"])
            response = json.loads(preserved_response(kwargs))
            check = request["check_1"]
            if any(item["action_id"] == "action_9" for item in check.get("action_obligations", [])):
                response["check_1"]["action_decisions"]["action_9"] = {
                    "verdict": "missing", "evidence_span_ids": [],
                }
            return json.dumps(response)

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, repair_feedback=repair_feedback,
        )

        self.assertEqual(len(calls), 2)
        self.assertNotIn(error, receipts)
        self.assertEqual(candidate.clear_confirmed_coverage_errors([error], segment, receipts), [error])
        feedback = "\n".join(repair_feedback[error])
        self.assertIn("Mira unlocks the door", feedback)
        self.assertIn("predicate: unlock", feedback)
        self.assertIn("object/target: door", feedback)
        self.assertIn("marked it missing", feedback)

    def test_missing_unknown_and_invalid_evidence_cannot_clear_compound_error(self):
        requirement = action_sentences(2)
        error, beats, events, segment = make_review_fixture(requirement)

        def run_variant(variant):
            feedback = {}

            def generate(**kwargs):
                request = json.loads(kwargs["prompt"])
                check_id = next(iter(request))
                obligations = request[check_id]["action_obligations"]
                good = preserved_response(kwargs)
                response = json.loads(good)
                decisions = response[check_id]["action_decisions"]
                if variant == "missing":
                    decisions.pop(obligations[-1]["action_id"])
                elif variant == "unknown":
                    decisions["action_999"] = decisions.pop(obligations[-1]["action_id"])
                elif variant == "unknown_check":
                    response["check_999"] = response.pop(check_id)
                elif variant == "citation":
                    decisions[obligations[-1]["action_id"]]["evidence_span_ids"] = [
                        "check_1_card_999_action_span_1"
                    ]
                elif variant == "malformed_verdict":
                    decisions[obligations[-1]["action_id"]]["verdict"] = []
                elif variant == "malformed_decision":
                    decisions[obligations[-1]["action_id"]] = ["preserved"]
                return json.dumps(response)

            receipts = candidate.review_missing_camera_actions(
                [error], segment, assigned_beats=beats, source_events=events,
                generate=generate, repair_feedback=feedback,
            )
            self.assertNotIn(error, receipts)
            self.assertEqual(candidate.clear_confirmed_coverage_errors([error], segment, receipts), [error])
            self.assertTrue(feedback[error])
            return "\n".join(feedback[error])

        self.assertIn("omitted or added action IDs", run_variant("missing"))
        self.assertIn("omitted or added action IDs", run_variant("unknown"))
        self.assertIn("no decision for this check", run_variant("unknown_check"))
        self.assertIn("outside this check's visual spans", run_variant("citation"))
        self.assertIn("malformed", run_variant("malformed_verdict"))
        self.assertIn("no decision", run_variant("malformed_decision"))

    def test_later_batch_failure_keeps_compound_error_unresolved(self):
        requirement = action_sentences(9)
        error, beats, events, segment = make_review_fixture(requirement)
        feedback = {}
        call_count = 0

        def generate(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("synthetic review failure")
            return preserved_response(kwargs)

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, repair_feedback=feedback,
        )

        self.assertEqual(call_count, 2)
        self.assertNotIn(error, receipts)
        self.assertTrue(any("no decision" in item for item in feedback[error]))

    def test_cancellation_in_later_batch_propagates(self):
        requirement = action_sentences(9)
        error, beats, events, segment = make_review_fixture(requirement)
        call_count = 0

        def generate(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise InterruptedError("cancelled")
            return preserved_response(kwargs)

        budget = candidate.CameraReviewBudget(max_requests=2)
        with self.assertRaises(InterruptedError):
            candidate.review_missing_camera_actions(
                [error], segment, assigned_beats=beats, source_events=events,
                generate=generate, review_budget=budget,
            )
        self.assertEqual(call_count, 2)
        self.assertEqual(budget.requests_dispatched, 2)

    def test_sixty_four_actions_use_at_most_eight_bounded_requests(self):
        requirement = action_sentences(64)
        error, beats, events, segment = make_review_fixture(requirement)
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate,
        )

        self.assertEqual(len(calls), 8)
        for call in calls:
            check = json.loads(call["prompt"])["check_1"]
            self.assertLessEqual(len(check["action_obligations"]), 8)
            self.assertLessEqual(call["max_new_tokens"], 4096)
        self.assertEqual(candidate.clear_confirmed_coverage_errors([error], segment, receipts), [])

    def test_shared_window_budget_caps_large_review_at_six_dispatched_requests(self):
        requirement = action_sentences(64)
        error, beats, events, segment = make_review_fixture(requirement)
        feedback = {}
        calls = []
        budget = candidate.CameraReviewBudget(max_requests=6)

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, repair_feedback=feedback, review_budget=budget,
        )

        self.assertEqual(len(calls), 6)
        self.assertEqual(budget.requests_dispatched, 6)
        self.assertEqual(receipts, {})
        self.assertEqual(candidate.clear_confirmed_coverage_errors(
            [error], segment, receipts), [error])
        self.assertTrue(any("budget of 6 per camera window was exhausted" in item
                            for item in feedback[error]))

    def test_more_than_sixty_four_actions_is_actionable_and_never_reviewed_broadly(self):
        requirement = action_sentences(65)
        error, beats, events, segment = make_review_fixture(requirement)
        feedback = {}
        generate = Mock(return_value="{}")

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, repair_feedback=feedback,
        )

        generate.assert_not_called()
        self.assertEqual(receipts, {})
        self.assertTrue(any("could not be split safely" in item for item in feedback[error]))

    def test_eight_request_budget_leaves_later_legacy_check_unresolved(self):
        compound = action_sentences(64)
        requirement = "Mira opens the blue drawer."
        error1, beats1, events1, segment = make_review_fixture(compound)
        error2 = f"B2 shot action omits required source step: {requirement}"
        segment["shots"].append({
            "beat_ids": ["B2"], "start_seconds": 0.0, "end_seconds": 4.0,
            "action": requirement, "camera": "Hold on Mira opening the drawer.",
        })
        beats = beats1 + [{"beat_id": "B2", "source_event_ids": ["E2"]}]
        events = events1 + [{"event_id": "E2", "text": requirement}]
        feedback = {}
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        receipts = candidate.review_missing_camera_actions(
            [error1, error2], segment, assigned_beats=beats, source_events=events,
            generate=generate, repair_feedback=feedback,
        )

        self.assertEqual(len(calls), 8)
        self.assertIn(error1, receipts)
        self.assertNotIn(error2, receipts)
        self.assertTrue(any("request limit of 8" in item for item in feedback[error2]))
        self.assertTrue(any(requirement in item for item in feedback[error2]))

    def test_unsupported_physical_clause_is_not_silently_dropped(self):
        requirements = [
            "Mira opens the gate, then Mira skitters the pebble into the jar.",
            "Mira opens the gate and skitters the pebble into the jar.",
        ]
        for requirement in requirements:
            with self.subTest(requirement=requirement):
                error, beats, events, segment = make_review_fixture(requirement)
                feedback = {}
                generate = Mock(return_value="{}")

                receipts = candidate.review_missing_camera_actions(
                    [error], segment, assigned_beats=beats, source_events=events,
                    generate=generate, repair_feedback=feedback,
                )

                generate.assert_not_called()
                self.assertEqual(receipts, {})
                self.assertTrue(any("could not be split safely" in item
                                    for item in feedback[error]))
                self.assertTrue(any("Mira opens the gate and skitters" in item
                                    or "Mira skitters the pebble into the jar" in item
                                    for item in feedback[error]))

    def test_requirement_outside_action_parser_keeps_legacy_full_text_review(self):
        requirement = "The stream hammers the mouth of the jug the whole time."
        error, beats, events, segment = make_review_fixture(requirement)
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate,
        )

        self.assertEqual(len(calls), 1)
        check = json.loads(calls[0]["prompt"])["check_1"]
        self.assertNotIn("action_obligations", check)
        self.assertEqual(check["source_requirement"], requirement)
        self.assertEqual(candidate.clear_confirmed_coverage_errors(
            [error], segment, receipts), [])

    def test_legacy_failed_review_feedback_names_the_exact_requirement(self):
        requirement = "The glass prisms scatter blue reflections across the desk."
        error, beats, events, segment = make_review_fixture(requirement)
        feedback = {}

        def generate(**kwargs):
            check_id = next(iter(json.loads(kwargs["prompt"])))
            return json.dumps({check_id: {
                "verdict": "missing", "evidence_span_ids": [],
            }})

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, repair_feedback=feedback,
        )

        self.assertEqual(receipts, {})
        self.assertTrue(any(requirement in item for item in feedback[error]))

    def test_legacy_final_state_and_scoped_fingerprint_behavior_remains(self):
        requirement = "Nia opens the blue door."
        error, beats, events, segment = make_review_fixture(requirement)
        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=lambda **kwargs: preserved_response(kwargs),
        )
        self.assertEqual(candidate.clear_confirmed_coverage_errors([error], segment, receipts), [])
        segment["shots"][0]["action"] = "Nia watches the closed blue door."
        self.assertEqual(candidate.clear_confirmed_coverage_errors([error], segment, receipts), [error])

        final_error, final_beats, final_events, final_segment = make_review_fixture(
            "The blue door remains closed."
        )
        final_events[0]["requirement_kind"] = "final_state"
        calls = []

        def final_state_generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        final_receipts = candidate.review_missing_camera_actions(
            [final_error], final_segment, assigned_beats=final_beats,
            source_events=final_events, generate=final_state_generate,
        )
        packet = json.loads(calls[0]["prompt"])["check_1"]
        self.assertEqual(packet["requirement_kind"], "final_state")
        self.assertNotIn("action_obligations", packet)
        self.assertEqual(candidate.clear_confirmed_coverage_errors(
            [final_error], final_segment, final_receipts), [])

    def test_timed_hold_review_keeps_per_window_evidence_and_fingerprint(self):
        requirement = "A still portrait holds after the final note."
        error, beats, events, segment = make_review_fixture(requirement)
        segment["shots"][0]["end_seconds"] = 2.0
        contracts = [{
            "source_event_id": "E1",
            "source_event_text": requirement,
            "windows": [{
                "window": 1,
                "local_start_seconds": 0.0,
                "local_end_seconds": 1.0,
            }],
        }]
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return preserved_response(kwargs)

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=generate, timed_hold_contracts=contracts,
        )

        self.assertEqual(len(calls), 1)
        packet = json.loads(calls[0]["prompt"])["check_1"]
        assert_metadata_only_cards(self, packet)
        self.assertEqual(packet["visual_cards"][0]["local_start_seconds"], 0.0)
        self.assertEqual(packet["visual_cards"][0]["local_end_seconds"], 2.0)
        span = packet["visual_spans"][0]
        self.assertEqual(span["window"], 1)
        self.assertIn("check_1_window_1_card_1_action_span_1", span["span_id"])
        self.assertEqual(packet["timed_hold_contract"]["windows"][0]["local_start_seconds"], 0.0)
        self.assertEqual(packet["timed_hold_contract"]["windows"][0]["local_end_seconds"], 1.0)
        self.assertNotEqual(
            packet["visual_cards"][0]["local_end_seconds"],
            packet["timed_hold_contract"]["windows"][0]["local_end_seconds"],
            "The reviewer must retain the card clock even when it extends beyond the required hold interval",
        )
        self.assertEqual(receipts[error]["kind"], "timed_hold")
        self.assertEqual(candidate.clear_confirmed_coverage_errors([error], segment, receipts), [])
        segment["shots"][0]["action"] = "Mira walks away from the portrait."
        self.assertEqual(candidate.clear_confirmed_coverage_errors([error], segment, receipts), [error])

    def test_collective_retreat_guard_still_requires_each_person_to_move(self):
        clause = "Both step back from the table."
        first = {
            "source_clause": clause,
            "action_focus": {"actor_markers": ["role:mara"]},
        }
        second = {
            "source_clause": clause,
            "action_focus": {"actor_markers": ["role:noel"]},
        }
        named_movement = [{"text": "Mara steps back from the table."}]
        joint_movement = [{"text": "Both Mara and Noel step back from the table."}]

        self.assertTrue(candidate._collective_retreat_evidence(first, named_movement, ["Mara", "Noel"]))
        self.assertFalse(candidate._collective_retreat_evidence(second, named_movement, ["Mara", "Noel"]))
        self.assertTrue(candidate._collective_retreat_evidence(first, joint_movement, ["Mara", "Noel"]))
        self.assertTrue(candidate._collective_retreat_evidence(second, joint_movement, ["Mara", "Noel"]))

    def test_scoped_neighbor_fingerprint_expires_when_neighbor_changes(self):
        requirement = "Mira opens the gate, then Mira closes the gate."
        error, beats, events, segment = make_review_fixture(requirement)
        beats.append({"beat_id": "B2", "source_event_ids": ["E1"]})
        segment["shots"].append({
            "beat_ids": ["B2"], "start_seconds": 4.0, "end_seconds": 8.0,
            "action": "Mira closes the gate.",
            "camera": "Hold a medium view as Mira closes the gate.",
            "framing": "The gate and Mira remain visible.",
        })

        receipts = candidate.review_missing_camera_actions(
            [error], segment, assigned_beats=beats, source_events=events,
            generate=lambda **kwargs: preserved_response(kwargs),
        )

        self.assertIn("scope", receipts[error])
        self.assertEqual(candidate.clear_confirmed_coverage_errors(
            [error], segment, receipts), [])
        segment["shots"][1]["action"] = "Mira walks away from the gate."
        self.assertEqual(candidate.clear_confirmed_coverage_errors(
            [error], segment, receipts), [error])


if __name__ == "__main__":
    unittest.main()
