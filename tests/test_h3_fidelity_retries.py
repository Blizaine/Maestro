"""Focused retry-count regressions for the staged H3 planner."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_story_ledger import (  # noqa: E402
    _deterministic_ledger,
    plan_h3_story_segments,
)
from services import studio_enhancement  # noqa: E402


def _silent_candidate(marker: str = "") -> dict:
    prompt = "Nora crosses a quiet room. No dialogue."
    candidate = _deterministic_ledger(
        prompt, segment_count=1, segment_durations=[10.0],
        locked_dialogue=[], camera_coverage="multi_shot", reference_context="",
    )
    if marker:
        candidate["retry_test_marker"] = marker
    return candidate


def _camera_draft(number: int, title: str) -> dict:
    action = "Nora walks to the table." if number == 1 else "Lee opens the door."
    return {
        "segment": number,
        "title": title,
        "opening_state": "Nora and Lee are in the room.",
        "coverage": "clear wide coverage",
        "pacing": "measured real-time action",
        "shots": [{
            "shot": 1,
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "transition": "opening composition",
            "framing": "wide view of the room",
            "camera": "locked camera",
            "beat_ids": [f"B{number}"],
            "action": action,
            "dialogue": [],
            "sound_effects": "quiet room tone",
        }],
        "closing_state": f"State after window {number}.",
    }


class H3FidelityRetryTests(unittest.TestCase):
    def run_schedule(self, retries: int | None, violations: list[list[str]], markers: list[str]):
        prompt = "Nora crosses a quiet room. No dialogue."
        candidates = [_silent_candidate(marker) for marker in markers]
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            index = len(calls) - 1
            return json.dumps(candidates[index])

        with ExitStack() as stack:
            setting = None
            if retries is not None:
                setting = stack.enter_context(patch(
                    "services.studio_enhancement.fidelity_retry_limit",
                    return_value=retries,
                ))
            else:
                setting = stack.enter_context(patch(
                    "services.studio_enhancement.fidelity_retry_limit",
                    wraps=studio_enhancement.fidelity_retry_limit,
                ))
            stack.enter_context(patch(
                "services.h3_story_ledger.ledger_violations",
                side_effect=violations,
            ))
            stack.enter_context(patch(
                "services.h3_story_ledger._render_h3_story_segments",
                side_effect=lambda context, **_kwargs: context,
            ))
            stack.enter_context(patch(
                "promptbench.experiments.action_first_enabled", return_value=False,
            ))
            stack.enter_context(patch(
                "promptbench.experiments.planning_thinking_enabled", return_value=False,
            ))
            result = plan_h3_story_segments(
                prompt, segment_durations=[10.0], mode="reference_sequence",
                camera_coverage="multi_shot", expect_dialogue=False,
                planning_style="creative", llm_generate=generate,
            )
            self.assertEqual(setting.call_count, 1)
        return result, calls

    def run_camera(self, retries: int, camera_errors: list[list[str]], *, cancel_repair=False):
        prompt = (
            "[0s-5s] Nora walks to the table.\n"
            "[5s-10s] Lee opens the door. No dialogue."
        )
        calls = []
        window_two_drafts = 0
        camera_error_values = iter(camera_errors)

        def generate(**kwargs):
            nonlocal window_two_drafts
            calls.append(kwargs)
            schema = kwargs.get("json_schema") or {}
            properties = schema.get("properties") or {}
            if "segment" not in properties:
                return json.dumps({
                    "character_appearance": {
                        "Nora": "As supplied in the source.",
                        "Lee": "As supplied in the source.",
                    },
                    "setting_continuity": "One quiet room with a table and door.",
                    "motion_mechanics": "Natural physical movement.",
                    "visual_continuity": "Naturalistic live action.",
                    "editing_style": "Readable camera coverage.",
                    "ambient_audio": "Quiet room tone.",
                })
            number = properties["segment"]["minimum"]
            if number == 1:
                return json.dumps(_camera_draft(1, "accepted first window"))
            window_two_drafts += 1
            if cancel_repair and window_two_drafts > 1:
                raise InterruptedError("cancelled during camera repair")
            return json.dumps(_camera_draft(2, f"window two draft {window_two_drafts}"))

        def check_camera(_prompt, _segment, *, segment_number, **_kwargs):
            if segment_number == 1:
                return []
            return next(camera_error_values)

        with (
            patch(
                "services.studio_enhancement.fidelity_retry_limit",
                return_value=retries,
            ) as setting,
            patch(
                "services.h3_story_ledger.segment_violations",
                side_effect=check_camera,
            ),
            patch(
                "services.h3_story_ledger._camera_repair_feedback",
                return_value=["correct the forced camera-fidelity failure"],
            ),
            patch("promptbench.experiments.action_first_enabled", return_value=False),
            patch("promptbench.experiments.planning_thinking_enabled", return_value=False),
        ):
            result = plan_h3_story_segments(
                prompt, segment_durations=[5.0, 5.0], mode="reference_sequence",
                camera_coverage="multi_shot", expect_dialogue=False,
                planning_style="faithful", llm_generate=generate,
            )
            self.assertEqual(setting.call_count, 1)
        return result, calls

    def test_schedule_retries_can_be_disabled_and_warning_reports_zero(self):
        result, calls = self.run_schedule(0, [["locked event missing"]], ["initial draft"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(
            "after 0 focused repair attempts" in warning
            for warning in result["planning_warnings"]
        ))

    def test_schedule_default_allows_one_repair_and_stops_on_success(self):
        result, calls = self.run_schedule(
            None, [["first check fails"], []], ["initial draft", "repaired draft"],
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["planning_diagnostics"], [])

    def test_schedule_retries_latest_rejected_draft_until_success(self):
        result, calls = self.run_schedule(
            5,
            [["first check fails"], ["second check fails"], []],
            ["draft zero", "draft one", "draft two"],
        )
        self.assertEqual(len(calls), 3)
        self.assertIn("draft zero", calls[1]["prompt"])
        self.assertIn("draft one", calls[2]["prompt"])
        self.assertNotIn("draft two", calls[2]["prompt"])
        self.assertEqual(result["planning_diagnostics"], [])

    def test_schedule_exhaustion_falls_back_with_actual_retry_count(self):
        result, calls = self.run_schedule(
            2,
            [["first check fails"], ["second check fails"], ["third check fails"]],
            ["draft zero", "draft one", "draft two"],
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["planned_by"], "deterministic_fallback")
        self.assertTrue(any(
            "after 2 focused repair attempts" in warning
            for warning in result["planning_warnings"]
        ))

    def test_camera_retry_limit_zero_preserves_prior_window_and_reports_zero(self):
        result, calls = self.run_camera(0, [["forced failure"]])
        repair_calls = [
            call for call in calls
            if "PREVIOUS REJECTED SEGMENT JSON" in call["prompt"]
        ]
        self.assertEqual(repair_calls, [])
        self.assertIn(
            "Nora walks to the table.",
            result["segments"][0]["shots"][0]["action"],
        )
        self.assertTrue(any(
            "after 0 focused repair attempts" in warning
            for warning in result["planning_warnings"]
        ))

    def test_camera_exhaustion_reports_the_number_of_repairs_attempted(self):
        result, calls = self.run_camera(
            2, [["initial failure"], ["repair one fails"], ["repair two fails"]],
        )
        repair_calls = [
            call for call in calls
            if "PREVIOUS REJECTED SEGMENT JSON" in call["prompt"]
        ]
        self.assertEqual(len(repair_calls), 2)
        self.assertTrue(any(
            "after 2 focused repair attempts" in warning
            for warning in result["planning_warnings"]
        ))

    def test_camera_retries_latest_draft_and_leaves_accepted_window_unchanged(self):
        result, calls = self.run_camera(
            4, [["initial failure"], ["repair one still fails"], []],
        )
        baseline, _ = self.run_camera(0, [["forced failure"]])
        repair_calls = [
            call for call in calls
            if "PREVIOUS REJECTED SEGMENT JSON" in call["prompt"]
        ]
        self.assertEqual(len(repair_calls), 2)
        self.assertIn("window two draft 1", repair_calls[0]["prompt"])
        self.assertIn("window two draft 2", repair_calls[1]["prompt"])
        self.assertEqual(result["segments"][0], baseline["segments"][0])
        self.assertIn(
            "Lee opens the door.",
            result["segments"][1]["shots"][0]["action"],
        )

    def test_schedule_cancellation_propagates_instead_of_falling_back(self):
        prompt = "Nora crosses a quiet room. No dialogue."
        candidate = _silent_candidate("initial draft")
        calls = 0

        def generate(**_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return json.dumps(candidate)
            raise InterruptedError("cancelled during schedule repair")

        with (
            patch("services.studio_enhancement.fidelity_retry_limit", return_value=2),
            patch("services.h3_story_ledger.ledger_violations", return_value=["forced failure"]),
            patch("promptbench.experiments.action_first_enabled", return_value=False),
            patch("promptbench.experiments.planning_thinking_enabled", return_value=False),
        ):
            with self.assertRaisesRegex(InterruptedError, "cancelled during schedule repair"):
                plan_h3_story_segments(
                    prompt, segment_durations=[10.0], mode="reference_sequence",
                    camera_coverage="multi_shot", expect_dialogue=False,
                    planning_style="creative", llm_generate=generate,
                )

    def test_camera_cancellation_propagates_instead_of_falling_back(self):
        with self.assertRaisesRegex(InterruptedError, "cancelled during camera repair"):
            self.run_camera(2, [["forced failure"]], cancel_repair=True)

    def test_resumed_plan_reads_current_retry_setting_once(self):
        resume = {"context": {"kept": True}, "segments": [], "retry_windows": [2]}
        with (
            patch("services.studio_enhancement.fidelity_retry_limit", return_value=3) as setting,
            patch("services.h3_story_ledger._prepare_h3_story_context") as prepare,
            patch(
                "services.h3_story_ledger._render_h3_story_segments",
                side_effect=lambda context, *, generate, resume, fidelity_retries: fidelity_retries,
            ),
        ):
            result = plan_h3_story_segments(
                "Nora waits. No dialogue.", segment_durations=[10.0],
                mode="reference_sequence", camera_coverage="multi_shot",
                llm_generate=lambda **_kwargs: "{}",
                resume=resume,
            )
        self.assertEqual(result, 3)
        self.assertEqual(setting.call_count, 1)
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
