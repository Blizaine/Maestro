"""Candidate isolation, physical-time preservation and exact/silent controls."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from promptbench.experiments import (
    action_first_enabled,
    experiment_context,
    planning_thinking_enabled,
)
from promptbench.story_time import reserve_time, speech_budget, reserve_camera_actions, action_first_clock
from services.dialogue_writing import creative_dialogue_budget
from services.h3_story_ledger import (
    _canonicalize_story_ledger, _deterministic_ledger, _ledger_schema,
    _camera_phase_beats, extract_source_events, plan_h3_story_segments,
)


class StoryTimeTests(unittest.TestCase):
    def test_experiment_is_task_local_and_resets_on_error(self):
        async def run(value):
            with experiment_context(value):
                await asyncio.sleep(0)
                return action_first_enabled()
        async def both():
            return await asyncio.gather(run("baseline"), run("action_first"))
        self.assertEqual(asyncio.run(both()), [False, True])
        with self.assertRaises(RuntimeError):
            with experiment_context("action_first"):
                raise RuntimeError()
        self.assertFalse(action_first_enabled())
        with self.assertRaises(ValueError):
            with experiment_context("unregistered"):
                pass

    def test_planning_thinking_is_request_local(self):
        self.assertFalse(planning_thinking_enabled())
        with experiment_context("planning_thinking"):
            self.assertTrue(planning_thinking_enabled())
        self.assertFalse(planning_thinking_enabled())

    def test_planning_thinking_changes_only_existing_ledger_call(self):
        prompt = "Ada lifts the latch. Bo pushes the door open. Ada catches the falling parcel. No dialogue."
        draft = _deterministic_ledger(
            prompt,
            segment_count=2,
            segment_durations=[14, 14],
            locked_dialogue=[],
            camera_coverage="multi_shot",
            reference_context="",
        )
        captures = {}
        for experiment in ("baseline", "planning_thinking"):
            calls = []

            def generate(**kwargs):
                calls.append(kwargs)
                return json.dumps(draft)

            with experiment_context(experiment):
                plan_h3_story_segments(
                    prompt,
                    segment_durations=[14, 14],
                    mode="reference_sequence_continuation",
                    camera_coverage="multi_shot",
                    reference_context="",
                    expect_dialogue=False,
                    planning_style="adaptive",
                    llm_generate=generate,
                )
            captures[experiment] = calls

        self.assertEqual(len(captures["baseline"]), len(captures["planning_thinking"]))
        baseline = captures["baseline"][0]
        candidate = captures["planning_thinking"][0]
        self.assertFalse(baseline["enable_thinking"])
        self.assertIsNotNone(baseline["json_schema"])
        self.assertTrue(candidate["enable_thinking"])
        self.assertEqual(candidate["thinking_budget"], 2048)
        self.assertEqual(candidate["reasoning_effort"], "medium")
        self.assertIsNone(candidate["json_schema"])

    def test_action_reservation_survives_speech_rewrite(self):
        prompt = "Ada carries the parcel to Bo. Bo discusses the delivery with Ada."
        canonical = _deterministic_ledger(prompt, segment_count=2, segment_durations=[14, 14],
            locked_dialogue=[], camera_coverage="multi_shot", reference_context="")
        draft = deepcopy(canonical)
        for beat in draft["beats"]:
            beat["action_seconds"] = 2
        with experiment_context("action_first"):
            compiled = _canonicalize_story_ledger(prompt, canonical, draft, locked_dialogue=[],
                segment_count=2, allow_generated_dialogue=True, preserve_adaptation=True)
            reserve_time(compiled, [14, 14])
            rewritten = _canonicalize_story_ledger(prompt, canonical, compiled, locked_dialogue=[],
                segment_count=2, allow_generated_dialogue=True, preserve_adaptation=True)
        self.assertEqual(compiled["_story_time"], rewritten["_story_time"])
        self.assertEqual([b["_action_seconds"] for b in compiled["beats"]],
                         [b["_action_seconds"] for b in rewritten["beats"]])
        self.assertEqual(compiled["required_final_outcome"], rewritten["required_final_outcome"])

    def test_movement_reduces_speech_budget_without_growing_duration(self):
        original = creative_dialogue_budget("Two people discuss a delivery.", 14)
        ledger = {"beats": [{"segment": 1, "_action_seconds": 5, "state_after": "Both inside."}]}
        reserve_time(ledger, [14])
        budget = speech_budget(ledger, 1, original)
        self.assertLess(budget.target, original.target)
        self.assertLessEqual(budget.maximum, int((9 - .8) * 3))
        self.assertEqual(ledger["_story_time"][0]["speech_seconds"], 9)

    def test_invalid_or_overcommitted_clock_is_not_silently_accepted(self):
        for value in [float('nan'), -1, True, 14]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                reserve_time({"beats": [{"segment": 1, "_action_seconds": value, "state_after": "Inside"}]}, [14])

    def test_speech_pressure_does_not_take_reserved_camera_time(self):
        shots = [{"event_indices": [1], "dialogue_ids": []},
                 {"event_indices": [1], "dialogue_ids": ["D1"]}]
        reserve_camera_actions(shots, [{"_action_seconds": 4}])
        times = action_first_clock(shots, [1, 25], 14, [0, 20])
        self.assertEqual(times, [4, 10])
        self.assertEqual(sum(times), 14)

    def test_missing_reserved_physical_card_requires_review(self):
        with self.assertRaisesRegex(ValueError, "reserved physical action"):
            reserve_camera_actions([{"event_indices": [1], "dialogue_ids": ["D1"]}], [{"_action_seconds": 3}])

    def test_candidate_plans_action_before_requesting_any_speech(self):
        prompt = "Ada opens the door. Bo enters and discusses the parcel with Ada."
        first = _deterministic_ledger(prompt, segment_count=2, segment_durations=[14, 14],
            locked_dialogue=[], camera_coverage="multi_shot", reference_context="")
        for beat in first["beats"]:
            beat["action_seconds"] = 2
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps(first) if len(calls) == 1 else "{}"
        with experiment_context("action_first"):
            result = plan_h3_story_segments(prompt, segment_durations=[14, 14],
                mode="reference_sequence_continuation", camera_coverage="multi_shot",
                reference_context="", expect_dialogue=True, planning_style="adaptive", llm_generate=generate)
        schema = calls[0]["json_schema"]
        self.assertEqual(schema["properties"]["generated_dialogue"]["maxItems"], 0)
        self.assertIn('action_seconds', schema["properties"]["beats"]["items"]["required"])
        speech_call = next(c for c in calls[1:] if 'FIT THE SPOKEN SCRIPT' in c["prompt"])
        self.assertIn('reserved_action_clock', speech_call["prompt"])
        self.assertIn('_story_time', result["ledger"])

    def test_silent_and_exact_quote_requests_use_identical_pipeline(self):
        for prompt in ['Ada and Bo fight in the courtyard. No dialogue.',
                       'Ada enters. Ada says, "Here is the key." Bo closes the door. Only use these lines.']:
            captures = []
            for experiment in ("baseline", "action_first"):
                # Intentionally invalid writer output exercises the same fallback path.
                generate = Mock(return_value="{}")
                with experiment_context(experiment):
                    result = plan_h3_story_segments(prompt, segment_durations=[14, 14],
                        mode="reference_sequence_continuation", camera_coverage="multi_shot",
                        reference_context="", expect_dialogue=False, planning_style="adaptive",
                        llm_generate=generate)
                captures.append((result, generate.call_args_list))
            self.assertEqual(captures[0], captures[1])


if __name__ == "__main__":
    unittest.main()
