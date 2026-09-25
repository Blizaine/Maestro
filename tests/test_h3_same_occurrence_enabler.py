"""CPU checks for narrow same-passage enabler grouping."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_source_structure import (  # noqa: E402
    co_locate_h3_enabler_beats,
    propose_h3_transport_enabler_groups,
)
from services.h3_story_ledger import extract_source_events  # noqa: E402


class H3SameOccurrenceEnablerTests(unittest.TestCase):
    def _events(self) -> list[dict]:
        return [
            {
                "event_id": "E1",
                "text": (
                    "Two adult exhibit volunteers, Sam and Priya, move one red ribbon spool "
                    "from a gallery to a workshop so they can hang a banner."
                ),
            },
            {
                "event_id": "E2",
                "text": "At the start Sam holds the spool in his right hand; it remains unused.",
            },
            {
                "event_id": "E3",
                "text": "They cross the gallery and pass through the closed workshop door.",
            },
            {
                "event_id": "E4",
                "text": "Priya opens the door, Sam enters first, and Priya closes it behind them.",
            },
            {
                "event_id": "E5",
                "text": "Only after they reach the workshop bench does Sam hand the same spool to Priya.",
            },
        ]

    def _relationship(self, events: list[dict]) -> dict:
        return {
            "relation_id": "R1",
            "relation_type": "overview_of",
            "overview_event_id": "E1",
            "detail_event_ids": [event["event_id"] for event in events[1:]],
            "coverage_status": "complete",
        }

    def _groups(self, events: list[dict] | None = None) -> list[dict]:
        events = events or self._events()
        return propose_h3_transport_enabler_groups(events, [self._relationship(events)])

    def test_overview_detail_pair_proves_one_passage_and_preserves_distinct_actors(self) -> None:
        events = self._events()
        before = deepcopy(events)

        groups = self._groups(events)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["relation_type"], "same_passage_enabler")
        self.assertEqual(groups[0]["source_event_ids"], ["E3", "E4"])
        self.assertEqual(groups[0]["threshold"], "workshop door")
        self.assertEqual(groups[0]["opener"], "Priya")
        self.assertEqual(groups[0]["traveler"], "Sam")
        self.assertEqual(
            groups[0]["source_evidence"]["summary"]["exact_text"],
            events[2]["text"],
        )
        self.assertEqual(
            groups[0]["source_evidence"]["enabler"]["exact_text"],
            events[3]["text"],
        )
        self.assertEqual(events, before, "source IDs and exact source text stay immutable")

    def test_actual_ribbon_source_extraction_proposes_only_the_doorway_pair(self) -> None:
        prompt = (
            "Two adult exhibit volunteers, Sam and Priya, move one red ribbon spool from a gallery "
            "to a workshop so they can hang a banner. At the start Sam holds the spool in his right "
            "hand; it remains unused while they cross the gallery and pass through the closed workshop "
            "door. Priya opens the door, Sam enters first, and Priya closes it behind them. Only after "
            "they reach the workshop bench does Sam hand the same spool to Priya. Priya then threads it "
            "through the banner's grommets and ties the banner to the wall. End with Priya holding the "
            "spool's loose end, the banner hanging securely, and the workshop door closed. Do not preview "
            "the handoff or banner tying before the workshop; preserve who holds the spool, the door state, "
            "and the single banner and spool across every transition. No dialogue or extra volunteers."
        )
        events = extract_source_events(prompt)
        relation = {
            "relation_id": "R1",
            "relation_type": "overview_of",
            "overview_event_id": "E1",
            "detail_event_ids": [event["event_id"] for event in events[1:]],
            "coverage_status": "complete",
        }

        groups = propose_h3_transport_enabler_groups(events, [relation])

        self.assertEqual([group["source_event_ids"] for group in groups], [["E3", "E4"]])
        self.assertEqual(groups[0]["source_evidence"]["summary"]["exact_text"], events[2]["text"])
        self.assertEqual(groups[0]["source_evidence"]["enabler"]["exact_text"], events[3]["text"])

    def test_pair_requires_an_accepted_adjacent_overview_detail_link(self) -> None:
        events = self._events()
        relation = self._relationship(events)
        relation["coverage_status"] = "review"
        self.assertEqual(propose_h3_transport_enabler_groups(events, [relation]), [])
        self.assertEqual(propose_h3_transport_enabler_groups(events, []), [])

        relation = self._relationship(events)
        relation["detail_event_ids"] = ["E2", "E4", "E3", "E5"]
        self.assertEqual(propose_h3_transport_enabler_groups(events, [relation]), [])

    def test_explicit_persistent_closed_or_magical_passage_is_not_grouped(self) -> None:
        for summary in (
            "They pass through the workshop door while it remains shut.",
            "They magically pass through the closed workshop door.",
            "They phase through the closed workshop door.",
        ):
            with self.subTest(summary=summary):
                events = self._events()
                events[2]["text"] = summary
                self.assertEqual(self._groups(events), [])

    def test_hypothetical_or_failed_summary_is_not_grouped_with_a_real_opener(self) -> None:
        for summary in (
            "They plan to pass through the closed workshop door.",
            "They cannot pass through the closed workshop door.",
            "They try to pass through the closed workshop door but fail.",
        ):
            with self.subTest(summary=summary):
                events = self._events()
                events[2]["text"] = summary
                self.assertEqual(self._groups(events), [])

    def test_later_return_different_threshold_and_competing_threshold_abstain(self) -> None:
        events = self._events()
        events[3]["text"] = "On the next visit Priya opens the door, and Sam enters first."
        self.assertEqual(self._groups(events), [])

        events = self._events()
        events[3]["text"] = "Priya opens the gallery gate, and Sam enters first."
        self.assertEqual(self._groups(events), [])

        events = self._events()
        events[1]["text"] += " The separate gallery gate is nearby."
        self.assertEqual(self._groups(events), [])

    def test_wrong_or_unreal_opener_and_dialogue_do_not_create_dependency(self) -> None:
        events = self._events()
        events[3]["text"] = "Nora opens the door, and Lee enters first."
        self.assertEqual(self._groups(events), [])

        events = self._events()
        events[3]["text"] = "Priya cannot open the door, but Sam enters first."
        self.assertEqual(self._groups(events), [])

        events = self._events()
        events[3]["text"] = "If Priya opens the door, Sam enters first."
        self.assertEqual(self._groups(events), [])

        events = self._events()
        events[3]["text"] = 'Priya says, "Open the door." Sam enters first.'
        self.assertEqual(self._groups(events), [])

    def test_timing_or_dialogue_metadata_keeps_events_out_of_group(self) -> None:
        events = self._events()
        events[2]["start_seconds"] = 0
        self.assertEqual(self._groups(events), [])

        events = self._events()
        events[3]["dialogue_ids"] = ["D1"]
        self.assertEqual(self._groups(events), [])

    def test_cross_window_co_location_replaces_stale_slot_and_keeps_receipt(self) -> None:
        events = self._events()
        groups = self._groups(events)
        beats = [
            {"beat_id": "B1", "segment": 1, "source_event_ids": ["E2"], "dialogue_ids": [], "description": "Sam holds the spool."},
            {"beat_id": "B2", "segment": 2, "source_event_ids": ["E3"], "dialogue_ids": [], "description": "They pass through the closed workshop door.", "state_after": "The pair is inside."},
            {"beat_id": "B3", "segment": 3, "source_event_ids": ["E4"], "dialogue_ids": [], "description": "Priya opens the door; Sam enters.", "state_after": "The door is closed."},
            {"beat_id": "B4", "segment": 4, "source_event_ids": ["E5"], "dialogue_ids": [], "description": "Sam hands Priya the spool."},
        ]
        before = deepcopy(beats)

        result, receipts = co_locate_h3_enabler_beats(beats, groups, events)

        self.assertEqual([beat["beat_id"] for beat in result], ["B1", "B2", "B3", "B4"])
        connector, joined = result[1], result[2]
        self.assertEqual(connector["source_event_ids"], [])
        self.assertEqual(connector["segment"], 2)
        self.assertIn("previous accepted frame", connector["description"])
        self.assertIn("same ownership", connector["description"])
        self.assertIn("Unchanged from the previous accepted visible state", connector["state_after"])
        self.assertIn("ownership", connector["state_after"])
        self.assertNotIn("They pass through the closed workshop door.", str(connector))
        self.assertNotIn("Priya opens the door", str(connector))
        self.assertNotIn("Sam enters first", str(connector))
        self.assertEqual(joined["source_event_ids"], ["E3", "E4"])
        self.assertEqual(joined["segment"], 3)
        self.assertEqual(joined["_source_enabler_groups"][0]["source_evidence"]["summary"]["exact_text"], events[2]["text"])
        self.assertEqual(joined["_source_enabler_groups"][0]["source_evidence"]["enabler"]["exact_text"], events[3]["text"])
        self.assertEqual(receipts[0]["operation"], "co_located")
        self.assertEqual(receipts[0]["staging_provenance"][0]["original_beat"], before[1])
        self.assertEqual(receipts[0]["staging_provenance"][1]["original_beat"], before[2])
        self.assertEqual(beats, before, "the input schedule is not mutated")

    def test_same_window_pair_removes_duplicate_slot_and_retains_both_ids(self) -> None:
        events = self._events()
        groups = self._groups(events)
        beats = [
            {"beat_id": "B1", "segment": 1, "source_event_ids": ["E2"], "dialogue_ids": []},
            {"beat_id": "B2", "segment": 2, "source_event_ids": ["E3"], "dialogue_ids": [], "description": "stale passage"},
            {"beat_id": "B3", "segment": 2, "source_event_ids": ["E4"], "dialogue_ids": [], "description": "door opens"},
        ]

        result, receipts = co_locate_h3_enabler_beats(beats, groups, events)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["source_event_ids"], ["E3", "E4"])
        self.assertEqual(result[1]["segment"], 2)
        self.assertEqual(receipts[0]["operation"], "co_located")
        self.assertEqual(receipts[0]["from_segments"], [2, 2])

    def test_pregrouped_pair_is_idempotently_marked(self) -> None:
        events = self._events()
        groups = self._groups(events)
        beats = [
            {"beat_id": "B1", "segment": 3, "source_event_ids": ["E3", "E4"], "dialogue_ids": []},
        ]
        result, receipts = co_locate_h3_enabler_beats(beats, groups, events)
        self.assertEqual(result[0]["source_event_ids"], ["E3", "E4"])
        self.assertEqual(receipts[0]["operation"], "already_grouped")

    def test_invalid_assignments_are_returned_unchanged(self) -> None:
        events = self._events()
        groups = self._groups(events)
        base = [
            {"beat_id": "B1", "segment": 2, "source_event_ids": ["E3"], "dialogue_ids": []},
            {"beat_id": "B2", "segment": 3, "source_event_ids": ["E4"], "dialogue_ids": []},
        ]
        cases = []
        cases.append([
            base[0],
            {"beat_id": "B2", "segment": 3, "source_event_ids": ["E5"], "dialogue_ids": []},
            {**base[1], "beat_id": "B3"},
        ])
        cases.append([{**base[0], "dialogue_ids": ["D1"]}, base[1]])
        cases.append([{**base[0], "start_seconds": 0}, base[1]])
        cases.append([base[1], base[0]])
        cases.append([base[0], {**base[1], "source_event_ids": ["E4", "E4"]}])
        cases.append([base[0], {**base[1], "source_event_ids": ["E4", "E5"]}])
        cases.append([base[0], base[1], {**base[1], "beat_id": "B3"}])

        for beats in cases:
            with self.subTest(beats=beats):
                before = deepcopy(beats)
                result, receipts = co_locate_h3_enabler_beats(beats, groups, events)
                self.assertEqual(result, before)
                self.assertEqual(receipts, [])


if __name__ == "__main__":
    unittest.main()
