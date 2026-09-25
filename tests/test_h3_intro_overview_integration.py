"""Introductory source synopses must not become duplicate camera actions."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_story_ledger import (
    _deterministic_ledger,
    _h3_source_relationship_invariants,
    _normalize_h3_source_metadata,
    _plan_long_form_ledger,
    extract_source_events,
    ledger_violations,
)


PROMPT = (
    "Two adult exhibit volunteers, Sam and Priya, move one red ribbon spool "
    "from a gallery to a workshop so they can hang a banner. "
    "At the start Sam holds the spool in his right hand; it remains unused "
    "while they cross the gallery and pass through the closed workshop door. "
    "Priya opens the door, Sam enters first, and Priya closes it behind them. "
    "Only after they reach the workshop bench does Sam hand the same spool to Priya. "
    "Priya then threads it through the banner's grommets and ties the banner to the wall. "
    "End with Priya holding the spool's loose end, the banner hanging securely, "
    "and the workshop door closed. No dialogue or extra volunteers."
)


def base_ledger(prompt=PROMPT):
    return _deterministic_ledger(
        prompt, segment_count=6, segment_durations=[14.0] * 6,
        locked_dialogue=[], camera_coverage="multi_shot", reference_context="",
    )


class IntroOverviewIntegrationTests(unittest.TestCase):
    def test_canonical_and_fallback_schedule_exclude_only_the_proven_synopsis(self):
        events = extract_source_events(PROMPT)
        original = deepcopy(events)
        ledger = base_ledger()
        links = ledger["source_relationships"]
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["overview_event_id"], "E1")
        self.assertEqual(links[0]["source_evidence"]["overview"]["exact_text"], events[0]["text"])
        assigned = [eid for beat in ledger["beats"] for eid in beat["source_event_ids"]]
        self.assertCountEqual(assigned, [event["event_id"] for event in events[1:]])
        self.assertEqual(events, original)
        self.assertEqual(ledger_violations(
            PROMPT, ledger, segment_count=6, locked_dialogue=[], expect_dialogue=False,
        ), [])

    def test_missing_detail_assignment_does_not_erase_the_synopsis(self):
        links, diagnostics = _normalize_h3_source_metadata(
            PROMPT, None, assigned_event_ids=["E2"], locked_dialogue=[],
        )
        self.assertEqual(links, [])
        self.assertTrue(any("exactly once" in item for item in diagnostics))

    def test_static_context_is_recomputed_from_source_not_model_metadata(self):
        ledger = base_ledger()
        ledger["source_relationships"][0]["invariant_context"] = {
            "participants": ["Someone else"], "unique_prop": "blue suitcase",
        }
        text = _h3_source_relationship_invariants(PROMPT, ledger)
        self.assertIn("Sam, Priya", text)
        self.assertIn("one red ribbon spool", text)
        self.assertNotIn("Someone else", text)
        self.assertNotIn("suitcase", text)
        self.assertNotIn("move", text)
        self.assertNotIn("from a gallery", text)
        self.assertNotIn("hang a banner", text)

    def test_without_immediate_reset_the_intro_remains_an_action(self):
        prompt = PROMPT.replace("At the start Sam", "Afterwards Sam")
        ledger = base_ledger(prompt)
        self.assertEqual(ledger["source_relationships"], [])
        self.assertIn("E1", [eid for beat in ledger["beats"] for eid in beat["source_event_ids"]])

    def test_explicit_overrides_do_not_silently_drop_local_actions(self):
        events = extract_source_events(PROMPT)
        ledger = _deterministic_ledger(
            PROMPT, segment_count=6, segment_durations=[14.0] * 6,
            locked_dialogue=[], camera_coverage="multi_shot", reference_context="",
            source_events_override=events,
        )
        self.assertIn("E1", [eid for beat in ledger["beats"] for eid in beat["source_event_ids"]])

    def test_long_form_context_labels_accepted_synopsis_in_outline_and_chapters(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            props = kwargs["json_schema"]["properties"]
            if "chapters" in props:
                return json.dumps({"chapters": [
                    {"chapter": index + 1, "objective": "Advance the assigned details",
                     "opening_state": "Carry the last established state",
                     "closing_state": "Retain the resulting state", "continuity_notes": "No replay"}
                    for index in range(props["chapters"]["minItems"])
                ]})
            return json.dumps({"segments": [
                {"segment": index + 1, "supporting_progression": "A brief pause",
                 "resulting_state": "The established state continues",
                 "sound_effects": "Natural room tone", "dialogue": []}
                for index in range(props["segments"]["minItems"])
            ]})

        canonical = _deterministic_ledger(
            PROMPT, segment_count=25, segment_durations=[14.0] * 25,
            locked_dialogue=[], camera_coverage="multi_shot", reference_context="",
        )
        _plan_long_form_ledger(
            PROMPT, canonical_ledger=canonical, segment_durations=[14.0] * 25,
            reference_context="", generate=generate, image_paths=None,
            nsfw=False, planning_style="faithful", allow_generated_dialogue=False,
            locked_dialogue=[],
        )
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertIn("SOURCE SYNOPSES — CONTEXT ONLY", call["prompt"])
            self.assertIn("do not replay the summarized actions", call["prompt"])
            self.assertIn(extract_source_events(PROMPT)[0]["text"], call["prompt"])


if __name__ == "__main__":
    unittest.main()
