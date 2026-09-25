"""Planner wiring for source-requested static audio-tail coverage."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import h3_camera_fidelity, llm_service
from services.h3_performance_audio import performance_audio_item_duration_seconds
from services.h3_story_ledger import (
    _deterministic_ledger,
    extract_source_events,
    plan_h3_story_segments,
)


_BASE_REFERENCE_CONTEXT = "The exact target soundtrack supplies the performance and timing."


def _reference_context(metadata):
    duration = performance_audio_item_duration_seconds(metadata)
    if duration is None:
        return _BASE_REFERENCE_CONTEXT
    return (
        _BASE_REFERENCE_CONTEXT
        + f"\nH3_PERFORMANCE_AUDIO_CLOCK duration_seconds={duration!r}"
    )


class TimedHoldPlannerIntegrationTests(unittest.TestCase):
    def _run_plan(self, source, *, metadata, force_review=False):
        reference_context = _reference_context(metadata)
        durations = [10.0, 10.0]
        events = extract_source_events(source)
        hold_event = next((
            event for event in events
            if "holds the final composition in silence" in event["text"].casefold()
        ), None)
        prompts = []
        review_calls = []
        semantic_review_count = 0

        def writer(**kwargs):
            nonlocal semantic_review_count
            prompt = kwargs["prompt"]
            schema = kwargs["json_schema"]
            prompts.append({"prompt": prompt, "schema": schema})
            properties = schema["properties"]
            if "beats" in properties:
                return json.dumps(_deterministic_ledger(
                    source,
                    segment_count=2,
                    segment_durations=durations,
                    locked_dialogue=[],
                    camera_coverage="multi_shot",
                    reference_context=reference_context,
                ))

            if "character_appearance" in properties and "setting_continuity" in properties:
                appearance_schema = properties["character_appearance"]
                appearance = (
                    {name: f"{name} in the established scene"
                     for name in appearance_schema.get("properties", {})}
                    if appearance_schema.get("type") == "object"
                    else "Nora in the established scene"
                )
                return json.dumps({
                    "character_appearance": appearance,
                    "setting_continuity": "The curtain, lantern, and room remain coherent.",
                    "motion_mechanics": "N/A",
                    "visual_continuity": "Keep the established final composition coherent.",
                    "editing_style": "Motivated cinematic cuts.",
                    "ambient_audio": "The supplied target soundtrack, unchanged.",
                    **({"source_relationships": []} if "source_relationships" in properties else {}),
                })

            if "event_cards" in properties:
                segment_number = properties["segment"]["minimum"]
                card_properties = properties["event_cards"]["properties"]
                cards = {}
                for event_number, (event_key, event_schema) in enumerate(card_properties.items(), start=1):
                    event_fields = event_schema["properties"]
                    action = (
                        "Nora lowers the curtain beside the lantern as the soundtrack reaches its final note."
                        if segment_number == 1
                        else "Nora and the final composition remain completely still in silence throughout this window."
                    )
                    phase = {
                        "action": action,
                        "framing": "Wide view of Nora and the final composition.",
                        "camera": "Hold a steady view of the complete arrangement.",
                        "transition": "continuous reframe",
                        "sound_effects": "N/A",
                    }
                    if "phases" in event_fields:
                        cards[event_key] = {"phases": [phase]}
                    else:
                        # This fixture is silent; the fallback keeps a useful
                        # shape if the schedule adds an optional setup card.
                        cards[event_key] = {key: phase for key in event_fields}
                return json.dumps({
                    "segment": segment_number,
                    "title": "The final arrangement",
                    "coverage": "A clear view of Nora and the final composition.",
                    "pacing": "Unhurried, with a settled ending.",
                    "event_cards": cards,
                    "closing_state": "Nora and the final composition remain still.",
                })

            # The real coverage reviewer receives a local JSON packet. Return
            # missing once to exercise the focused-repair path, then return
            # source-window-scoped IDs to confirm the repaired contract.
            packet = json.loads(prompt)
            semantic_review_count += 1
            response = {}
            for check_id, check in packet.items():
                contract = check.get("timed_hold_contract") or {}
                windows = contract.get("windows") or []
                if semantic_review_count == 1:
                    response[check_id] = {"window_decisions": {
                        f"window_{item['window']}": {
                            "verdict": "missing", "evidence_span_ids": [],
                        } for item in windows
                    }}
                    continue
                spans_by_window = {}
                for span in check.get("visual_spans") or []:
                    spans_by_window.setdefault(span.get("window"), []).append(span["span_id"])
                response[check_id] = {"window_decisions": {
                    f"window_{item['window']}": {
                        "verdict": "preserved",
                        "evidence_span_ids": spans_by_window[item["window"]][:1],
                    } for item in windows
                }}
            return json.dumps(response)

        def violations(prompt, segment, *, segment_number, assigned_beats, **kwargs):
            if not force_review or hold_event is None:
                return []
            for beat in assigned_beats:
                if hold_event["event_id"] in (beat.get("source_event_ids") or []):
                    return [
                        f"{beat['beat_id']} shot action omits required source step: {hold_event['text']}"
                    ]
            return []

        real_review = h3_camera_fidelity.review_missing_camera_actions

        def review_spy(errors, segment, **kwargs):
            review_calls.append({
                "errors": list(errors),
                "segment": segment,
                "timed_hold_contracts": kwargs.get("timed_hold_contracts"),
                "accepted_segments": deepcopy(kwargs.get("accepted_segments")),
            })
            return real_review(errors, segment, **kwargs)

        with (
            patch.object(llm_service, "keep_loaded", return_value=nullcontext()),
            patch("services.h3_story_ledger.segment_violations", side_effect=violations),
            patch("services.h3_camera_fidelity.review_missing_camera_actions", side_effect=review_spy),
        ):
            result = plan_h3_story_segments(
                source,
                segment_durations=durations,
                mode="reference_sequence",
                camera_coverage="multi_shot",
                reference_context=reference_context,
                expect_dialogue=False,
                planning_style="faithful",
                image_paths=None,
                has_start_image=False,
                nsfw=False,
                llm_generate=writer,
            )
        return result, prompts, review_calls, events, hold_event

    @staticmethod
    def _camera_prompts(prompts):
        return [item for item in prompts if "event_cards" in item["schema"]["properties"]]

    def test_verified_source_hold_reaches_each_window_and_review_retry(self):
        source = (
            "Nora lowers the curtain beside the lantern. "
            "The 20-second video holds the final composition in silence for the remaining 12 seconds."
        )
        metadata = {"type": "audio", "audio_intent": "drive", "duration_seconds": 8.0}
        result, prompts, review_calls, events, hold_event = self._run_plan(
            source, metadata=metadata, force_review=True,
        )

        self.assertIsNotNone(hold_event)
        source_text = hold_event["text"]
        source_id = hold_event["event_id"]
        camera_calls = self._camera_prompts(prompts)
        self.assertGreaterEqual(len(camera_calls), 3)  # W1, W2, and focused W2 repair.
        for segment_number, local_start in ((1, 8), (2, 0)):
            matching = [
                item for item in camera_calls
                if item["schema"]["properties"]["segment"]["minimum"] == segment_number
            ]
            self.assertTrue(matching)
            for call in matching:
                prompt = call["prompt"]
                self.assertIn(source_id, prompt)
                self.assertIn(source_text, prompt)
                self.assertIn("local_start_seconds", prompt)
                self.assertRegex(prompt, rf'local_start_seconds"?\s*[:=]\s*{local_start}(?:\.0+)?\b')
                self.assertRegex(prompt, r'local_end_seconds"?\s*[:=]\s*10(?:\.0+)?\b')

        # The first review call belongs to W1. W2's initial review and its
        # post-repair review both see the already accepted W1 materialization.
        self.assertGreaterEqual(len(review_calls), 3)
        for call in review_calls:
            contracts = call["timed_hold_contracts"]
            self.assertIsInstance(contracts, list)
            self.assertEqual(len(contracts), 1)
            contract = contracts[0]
            self.assertEqual(contract["source_event_id"], source_id)
            self.assertEqual(contract["source_event_text"], source_text)
            self.assertEqual([item["window"] for item in contract["windows"]], [1, 2])
        self.assertEqual(review_calls[0]["accepted_segments"], [])
        self.assertEqual(len(review_calls[1]["accepted_segments"]), 1)
        self.assertEqual(review_calls[1]["accepted_segments"][0]["segment"], 1)
        self.assertEqual(len(review_calls[2]["accepted_segments"]), 1)
        self.assertEqual(review_calls[2]["accepted_segments"][0]["segment"], 1)
        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(len(result["segments"]), 2)

    def test_silence_or_missing_numeric_clock_does_not_create_a_static_hold_contract(self):
        explicit_but_unclocked = (
            "Nora lowers the curtain beside the lantern. "
            "The 20-second video holds the final composition in silence for the remaining 12 seconds."
        )
        silent_but_not_static = (
            "Nora lowers the curtain beside the lantern. "
            "The soundtrack fades into silence for the final 12 seconds of the 20-second video."
        )
        cases = (
            (explicit_but_unclocked, {"type": "audio", "audio_intent": "drive"}),
            (silent_but_not_static, {"type": "audio", "audio_intent": "drive", "duration_seconds": 8.0}),
        )
        for source, metadata in cases:
            with self.subTest(source=source):
                result, prompts, review_calls, _events, _hold = self._run_plan(
                    source, metadata=metadata, force_review=False,
                )
                for call in review_calls:
                    self.assertIn(call["timed_hold_contracts"], (None, []))
                for call in self._camera_prompts(prompts):
                    self.assertNotIn('"timed_hold_contract"', call["prompt"])
                    self.assertNotIn("H3_TIMED_STATIC_HOLD", call["prompt"])
                self.assertEqual(result["planned_by"], "llm")


if __name__ == "__main__":
    unittest.main()
