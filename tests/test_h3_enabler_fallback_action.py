"""Fallback passage actions are reordered only from exact source evidence."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_source_structure import (
    h3_same_passage_enabler_fallback_action,
    propose_h3_transport_enabler_groups,
)


class SamePassageFallbackActionTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {
                "event_id": "E1",
                "text": (
                    "Two adult exhibit volunteers, Sam and Priya, move one red ribbon spool "
                    "from a gallery to a workshop so they can hang a banner."
                ),
            },
            {"event_id": "E2", "text": "At the start Sam holds the spool in his right hand."},
            {
                "event_id": "E3",
                "text": "they cross the gallery and pass through the closed workshop door.",
            },
            {
                "event_id": "E4",
                "text": "Priya opens the door, Sam enters first.",
            },
            {"event_id": "E5", "text": "Priya closes it behind them."},
        ]
        self.relationships = [{
            "relation_id": "R1",
            "relation_type": "overview_of",
            "overview_event_id": "E1",
            "detail_event_ids": ["E2", "E3", "E4", "E5"],
            "coverage_status": "complete",
        }]
        groups = propose_h3_transport_enabler_groups(self.events, self.relationships)
        self.assertEqual(len(groups), 1)
        self.group = groups[0]

    def render(self, *, events=None, relationships=None, group=None):
        return h3_same_passage_enabler_fallback_action(
            self.group if group is None else group,
            self.events if events is None else events,
            self.relationships if relationships is None else relationships,
        )

    def test_reorders_only_opening_before_one_passage_and_preserves_later_close(self):
        action = self.render()
        self.assertIsNotNone(action)
        self.assertLess(action.index("cross the gallery"), action.index("Priya opens"))
        self.assertLess(action.index("Priya opens"), action.index("pass through the now-open"))
        self.assertLess(action.index("pass through the now-open"), action.index("Sam entering first"))
        self.assertEqual(action.count("pass through"), 1)
        self.assertEqual(action.count("Priya opens"), 1)
        self.assertNotIn("pass through the closed", action)
        # E5 remains a later immutable event; this pair formatter must not absorb it.
        self.assertNotIn("closes it behind them", action)
        self.assertEqual(self.events[4]["text"], "Priya closes it behind them.")

    def test_source_and_group_evidence_are_not_mutated(self):
        events = deepcopy(self.events)
        group = deepcopy(self.group)
        self.render(events=events, group=group)
        self.assertEqual(events, self.events)
        self.assertEqual(group, self.group)

    def test_forged_or_stale_source_evidence_abstains(self):
        group = deepcopy(self.group)
        group["source_evidence"]["summary"]["exact_text"] = "They pass through another door."
        self.assertIsNone(self.render(group=group))

        events = deepcopy(self.events)
        events[2]["text"] = "they cross the gallery and pass through the closed side door."
        self.assertIsNone(self.render(events=events))

    def test_unvalidated_or_conflicting_relationship_abstains(self):
        relationship = deepcopy(self.relationships[0])
        relationship["coverage_status"] = "incomplete"
        self.assertIsNone(self.render(relationships=[relationship]))

        relationship = deepcopy(self.relationships[0])
        relationship["detail_event_ids"] = ["E2", "E3", "E5", "E4"]
        self.assertIsNone(self.render(relationships=[relationship]))

    def test_different_threshold_and_unlisted_actors_abstain(self):
        events = deepcopy(self.events)
        events[3]["text"] = "Priya opens the cellar door, Sam enters first."
        self.assertIsNone(self.render(events=events))

        events = deepcopy(self.events)
        events[3]["text"] = "Lee opens the door, Sam enters first."
        self.assertIsNone(self.render(events=events))

        events = deepcopy(self.events)
        events[3]["text"] = "Priya opens the door, Lee enters first."
        self.assertIsNone(self.render(events=events))

    def test_extra_actions_are_never_dropped_by_the_narrow_formatter(self):
        events = deepcopy(self.events)
        events[2]["text"] = (
            "they cross the gallery, wave to the guard, and pass through the closed workshop door."
        )
        self.assertIsNone(self.render(events=events))

        events = deepcopy(self.events)
        events[3]["text"] = "Priya opens the door, Sam waves, and Sam enters first."
        self.assertIsNone(self.render(events=events))

    def test_repetition_negation_and_nonopening_unlock_only_abstain(self):
        events = deepcopy(self.events)
        events[2]["text"] = (
            "they cross the gallery and pass through the closed workshop door on another trip."
        )
        self.assertIsNone(self.render(events=events))

        events = deepcopy(self.events)
        events[3]["text"] = "Priya does not open the door, Sam enters first."
        self.assertIsNone(self.render(events=events))

        events = deepcopy(self.events)
        events[3]["text"] = "Priya unlocks the door, Sam enters first."
        self.assertIsNone(self.render(events=events))

    def test_timed_or_dialogue_events_abstain(self):
        events = deepcopy(self.events)
        events[2]["source_start_seconds"] = 2.0
        self.assertIsNone(self.render(events=events))

        events = deepcopy(self.events)
        events[3]["text"] = 'Priya opens the door and says, "Go." Sam enters first.'
        self.assertIsNone(self.render(events=events))

    def test_unsupported_clause_order_abstains_without_guessing(self):
        events = deepcopy(self.events)
        events[2]["text"] = "Priya approaches the closed workshop door, then Sam enters first."
        self.assertIsNone(self.render(events=events))


if __name__ == "__main__":
    unittest.main()
