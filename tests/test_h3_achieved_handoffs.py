"""Production planner regressions for achieved rather than proposed handoffs."""

from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import (
    _camera_phase_beats,
    _deterministic_ledger,
    _fallback_segment,
    _materialize_segment,
    _h3_preview_action_frames,
    extract_source_events,
    plan_h3_story_segments,
    segment_violations,
)
import services.h3_story_ledger as story_ledger_module  # noqa: E402


class AchievedHandoffTests(unittest.TestCase):
    def test_empty_handed_is_not_a_completed_handoff(self):
        self.assertFalse(any(frame[0] == "hand" for frame in _h3_preview_action_frames(
            "Tessa stands outside empty-handed", None,
        )))
        self.assertTrue(any(frame[0] == "hand" for frame in _h3_preview_action_frames(
            "Tessa handed Mara the lantern", None,
        )))

    def test_recurring_source_is_shown_on_distinct_occasions_in_fallback(self):
        source = "Each evening Nora leaves one fresh bouquet by the station clock. No dialogue."
        events = extract_source_events(source)
        beats = _camera_phase_beats(
            [{"beat_id": "B1", "description": events[0]["text"],
              "source_event_ids": ["E1"], "dialogue_ids": [], "state_after": "A bouquet is by the clock."}],
            source_events=events, expected_dialogue_events={},
        )
        self.assertEqual(len(beats), 2)
        draft = _fallback_segment(1, duration=8, beats=beats,
                                  opening_state="An empty station.", camera_coverage="multi_shot")
        rendered = _materialize_segment(draft, beats=beats, dialogue_catalog=[], source_events=events)
        self.assertIn("On one evening", rendered["shots"][0]["action"])
        self.assertIn("On a subsequent evening", rendered["shots"][1]["action"])
        draft["shots"][1]["action"] = "Nora leaves one fresh bouquet by the station clock."
        self.assertTrue(any("distinct later occasion" in issue for issue in segment_violations(
            source, draft, segment_number=1, duration=8, assigned_beats=beats, dialogue_catalog=[],
        )))

    def test_story_writer_cannot_put_a_later_retrieved_prop_in_the_opening(self):
        source = (
            "At the start, Tessa stands outside the closed archive empty-handed. "
            "Tessa opens the archive door and enters. "
            "Tessa retrieves the only lantern from the shelf. "
            "Tessa returns outside carrying the lantern. No dialogue."
        )
        events = extract_source_events(source)
        candidate = _deterministic_ledger(
            source, segment_count=2, segment_durations=[8, 8], locked_dialogue=[],
            camera_coverage="multi_shot", reference_context="",
        )
        candidate["initial_state"] = "Tessa is already carrying the lantern outside the archive."

        def generate(**kwargs):
            props = kwargs["json_schema"]["properties"]
            if "setting_continuity" in props:
                return json.dumps(candidate)
            number = props["segment"]["minimum"]
            local = [beat for beat in candidate["beats"] if beat["segment"] == number]
            card_texts = [event["text"] for beat in local for event in events
                          if event["event_id"] in beat["source_event_ids"]]
            keys = list(props["event_cards"]["properties"])
            self.assertEqual(len(keys), len(card_texts))
            return json.dumps({
                "segment": number, "coverage": "Multi shot", "closing_state": card_texts[-1],
                "event_cards": {key: {"phases": [{
                    "action": text, "camera": "Hold steady", "framing": "Medium",
                    "transition": "Hard cut", "sound_effects": "Quiet room tone",
                }]} for key, text in zip(keys, card_texts)},
            })

        result = plan_h3_story_segments(
            source, segment_durations=[8, 8], mode="sliding_window",
            planning_style="adaptive", camera_coverage="multi_shot", llm_generate=generate,
        )
        self.assertEqual(result["planning_warnings"], [])
        self.assertNotIn("already carrying", result["segments"][0]["opening_state"])
        self.assertIn("empty-handed", result["segments"][0]["opening_state"])

    def test_fallback_bucket_keeps_ordinary_events_beside_a_recurring_phase(self):
        source = (
            "Each evening Nora leaves one fresh bouquet by the station clock. "
            "Nora opens the station gate. No dialogue."
        )
        events = extract_source_events(source)
        beats = _camera_phase_beats([
            {"beat_id": f"B{index}", "description": event["text"],
             "source_event_ids": [event["event_id"]], "dialogue_ids": []}
            for index, event in enumerate(events, 1)
        ], source_events=events, expected_dialogue_events={})
        self.assertEqual(len(beats), 3)
        draft = _fallback_segment(1, duration=8, beats=beats,
                                  opening_state="An empty station.", camera_coverage="multi_shot")
        # Model the overflow bucket produced by the fallback's shot limit.
        draft["shots"][1]["beat_ids"].extend(draft["shots"][2]["beat_ids"])
        draft["shots"] = draft["shots"][:2]
        rendered = _materialize_segment(draft, beats=beats, dialogue_catalog=[], source_events=events)
        action = rendered["shots"][1]["action"]
        self.assertIn("On a subsequent evening", action)
        self.assertIn("Nora opens the station gate", action)
        self.assertLess(action.index("On a subsequent evening"), action.index("Nora opens"))

    def test_unperformed_print_insertion_cannot_cross_window_boundary(self):
        source = (
            "[0s-4s] Celia adjusts the empty frame.\n"
            "[4s-8s] Celia aligns the print beside the frame.\n"
            "[8s-12s] Celia places the print inside the frame. No dialogue."
        )
        actions = [
            "Celia adjusts the empty frame.",
            "Celia aligns the print beside the frame.",
            "Celia places the print inside the frame.",
        ]
        camera_calls = []

        def generate(**kwargs):
            properties = kwargs["json_schema"]["properties"]
            if "setting_continuity" in properties:
                return json.dumps({
                    "character_appearance": "Celia wears a blue apron.",
                    "setting_continuity": "A framing workshop.",
                    "visual_continuity": "Soft daylight.",
                    "editing_style": "Readable continuous coverage.",
                })
            self.assertIn("event_cards", properties)
            self.assertNotIn("REPAIR ONLY", kwargs["prompt"])
            index = properties["segment"]["minimum"] - 1
            camera_calls.append(kwargs["prompt"])
            return json.dumps({
                "segment": index + 1, "coverage": "Continuous",
                "closing_state": (
                    "The print is already inside the frame."
                    if index == 1 else actions[index]
                ),
                "event_cards": {"event_1": {"phases": [{
                    "action": actions[index], "camera": "Hold steady", "framing": "Medium",
                    "transition": "Continuous reframe", "sound_effects": "Quiet workshop air",
                }]}},
            })

        original_materializer = story_ledger_module._materialize_segment
        final_materialized_actions = []
        materialized_visible_action = (
            "Celia adjusts the empty frame and the frame settles into place."
        )

        def materialize_with_final_action(*args, **kwargs):
            rendered = original_materializer(*args, **kwargs)
            if rendered.get("segment") == 1 and rendered.get("shots"):
                rendered["shots"][0]["action"] = (
                    "Silent visual action, never spoken narration: "
                    + materialized_visible_action
                    + ". No words are spoken or mouthed in this shot; only explicitly "
                    "requested nonverbal reactions may be heard"
                )
                final_materialized_actions.append(rendered["shots"][0]["action"])
            return rendered

        with patch.object(
            story_ledger_module,
            "_materialize_segment",
            side_effect=materialize_with_final_action,
        ):
            result = plan_h3_story_segments(
                source, segment_durations=[4, 4, 4], mode="sliding_window",
                planning_style="adaptive", camera_coverage="continuous", llm_generate=generate,
            )
        self.assertEqual(result["planning_warnings"], [])
        self.assertEqual(len(camera_calls), 3)
        self.assertIn("Silent visual action, never spoken narration", final_materialized_actions[0])
        self.assertNotEqual(materialized_visible_action, actions[0])
        first_history = result["segments"][0]["achieved_state"][
            "accepted_visible_camera_actions"
        ]
        self.assertTrue(any(
            item["action"] == materialized_visible_action
            and item["source_event_ids"] == ["E1"]
            for item in first_history
        ), first_history)
        self.assertIn(materialized_visible_action, result["segments"][0]["closing_state"])
        self.assertNotIn("Silent visual action, never spoken narration", result["segments"][0]["closing_state"])
        self.assertNotIn("No words are spoken or mouthed", result["segments"][0]["closing_state"])
        second, third = result["segments"][1:]
        self.assertNotIn("already inside", second["closing_state"])
        self.assertNotIn("inside the frame", second["closing_state"])
        self.assertEqual(third["opening_state"], second["closing_state"])
        self.assertIn("beside the frame", third["opening_state"])
        self.assertIn("inside the frame", third["closing_state"])
        self.assertEqual(second["achieved_state"]["completed_source_event_ids"], ["E1", "E2"])
        self.assertEqual(third["achieved_state"]["completed_source_event_ids"], ["E1", "E2", "E3"])
        # The next writer gets accepted visible evidence, including more than
        # the last shot, without copying that recap into the native boundary.
        self.assertIn('accepted_prior_action_context', camera_calls[1])
        self.assertIn('Exact completed source-event text', camera_calls[1])
        self.assertIn('Accepted visible camera actions from earlier windows', camera_calls[1])
        self.assertIn(materialized_visible_action, camera_calls[1])
        self.assertIn('accepted_prior_action_context', camera_calls[2])
        self.assertIn('Celia adjusts the empty frame.', camera_calls[2])
        self.assertIn(materialized_visible_action, camera_calls[2])
        self.assertNotIn('Silent visual action, never spoken narration', camera_calls[2])
        self.assertNotIn('already inside', camera_calls[2])
        self.assertNotIn('history only', third['opening_state'])


if __name__ == "__main__":
    unittest.main()
