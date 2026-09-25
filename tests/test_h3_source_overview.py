"""CPU regressions for source-proven introductory transport overviews."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_source_overview import (  # noqa: E402
    recognize_introductory_transport_overview,
)
from services.h3_story_ledger import extract_source_events  # noqa: E402


RIBBON_PROMPT = (
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


class H3SourceOverviewTests(unittest.TestCase):
    def _ribbon_events(self) -> list[dict]:
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
                "text": "Sam and Priya cross the gallery and pass through the closed workshop door.",
            },
            {
                "event_id": "E4",
                "text": "Priya opens the door, Sam enters first.",
            },
            {
                "event_id": "E5",
                "text": "Priya closes it behind them.",
            },
            {
                "event_id": "E6",
                "text": "Only after they reach the workshop bench does Sam hand the same spool to Priya.",
            },
            {
                "event_id": "E7",
                "text": "Priya threads it through the banner's grommets and ties the banner to the wall.",
            },
            {
                "event_id": "E8",
                "text": "End with Priya holding the spool's loose end and the banner hanging securely.",
            },
            {"event_id": "E9", "text": "the workshop door closed"},
        ]

    def _generic_events(self) -> list[dict]:
        return [
            {
                "event_id": "E1",
                "text": (
                    "Nora and Felix, museum staff, carry a single brass lantern "
                    "from a storeroom to a reading room for the evening inspection."
                ),
            },
            {"event_id": "E2", "text": "At the start Nora holds the lantern by its handle."},
            {"event_id": "E3", "text": "Nora and Felix cross the storeroom floor together."},
            {
                "event_id": "E4",
                "text": "Nora and Felix enter the reading room and reach the table with the same lantern.",
            },
        ]

    def test_ribbon_synopsis_is_related_to_ordered_trip_details_with_invariants(self) -> None:
        events = self._ribbon_events()
        before = deepcopy(events)

        proposal = recognize_introductory_transport_overview(events)

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal["relation_type"], "overview_of")
        self.assertEqual(proposal["overview_event_id"], "E1")
        self.assertEqual(proposal["detail_event_ids"], ["E2", "E3", "E4", "E5", "E6"])
        self.assertEqual(proposal["invariant_context"]["participants"], ["Sam", "Priya"])
        self.assertEqual(proposal["invariant_context"]["unique_prop"], "one red ribbon spool")
        self.assertEqual(proposal["route_evidence"]["origin"], "a gallery")
        self.assertEqual(proposal["route_evidence"]["destination"], "a workshop")
        self.assertEqual(proposal["route_evidence"]["origin_event_ids"], ["E3"])
        self.assertEqual(proposal["route_evidence"]["arrival_event_ids"], ["E6"])
        self.assertEqual(proposal["source_evidence"]["overview"]["exact_text"], events[0]["text"])
        self.assertEqual(events, before, "source event IDs and text must remain immutable")

    def test_actual_immutable_event_extraction_proves_the_ribbon_route(self) -> None:
        events = extract_source_events(RIBBON_PROMPT)

        proposal = recognize_introductory_transport_overview(events)

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal["overview_event_id"], "E1")
        self.assertEqual(proposal["detail_event_ids"], ["E2", "E3", "E4", "E5", "E6"])
        self.assertEqual(proposal["source_evidence"]["overview"]["exact_text"], events[0]["text"])

    def test_generic_names_prop_and_locations_work_without_scene_specific_rules(self) -> None:
        proposal = recognize_introductory_transport_overview(self._generic_events())

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal["invariant_context"]["participants"], ["Nora", "Felix"])
        self.assertEqual(proposal["invariant_context"]["unique_prop"], "a single brass lantern")
        self.assertEqual(proposal["route_evidence"]["origin"], "a storeroom")
        self.assertEqual(proposal["route_evidence"]["destination"], "a reading room")

    def test_existing_overview_counterexamples_without_reset_are_not_reclassified(self) -> None:
        cases = [
            [
                "Mara moves the brass lantern from the supply room to the office.",
                "Mara carries the brass lantern from the supply room down the hall.",
                "Mara carries the brass lantern from the hall into the office.",
            ],
            [
                "Mara moves the brass lantern from the supply room to the office.",
                "Nora carries the brass lantern from the supply room down the hall.",
                "Nora carries the brass lantern into the office.",
            ],
            [
                "Mara moves the brass lantern from the supply room to the office.",
                "Mara carries the red case from the supply room down the hall.",
                "Mara carries the red case into the office.",
            ],
            [
                "Mara moves the brass lantern from the supply room to the office.",
                "Mara carries the brass lantern from the supply room down the hall.",
                "Mara holds the brass lantern beside the window.",
            ],
            [
                "Mara moves the brass lantern from the supply room to the office.",
                "Mara tries to carry the brass lantern from the supply room but cannot lift it.",
                "The lantern remains on the supply-room shelf.",
            ],
        ]
        for index, texts in enumerate(cases):
            events = [
                {"event_id": f"E{event_index + 1}", "text": text}
                for event_index, text in enumerate(texts)
            ]
            with self.subTest(case=index):
                self.assertIsNone(recognize_introductory_transport_overview(events))

    def test_a_chronology_reset_alone_does_not_cover_missing_or_reversed_arrival(self) -> None:
        no_arrival = self._generic_events()
        no_arrival[-1]["text"] = "Felix waits beside the reading-room door with the lantern."
        self.assertIsNone(recognize_introductory_transport_overview(no_arrival))

        reversed_route = self._generic_events()
        reversed_route[2]["text"] = "Nora carries the lantern from the reading room back to the storeroom."
        self.assertIsNone(recognize_introductory_transport_overview(reversed_route))

    def test_conflicting_prop_color_object_or_actor_abstains(self) -> None:
        wrong_color = self._ribbon_events()
        wrong_color[1]["text"] = "At the start Sam holds the blue spool in his right hand."
        self.assertIsNone(recognize_introductory_transport_overview(wrong_color))

        second_prop = self._ribbon_events()
        second_prop[1]["text"] = "At the start Sam holds another spool in his right hand."
        self.assertIsNone(recognize_introductory_transport_overview(second_prop))

        wrong_actor = self._ribbon_events()
        wrong_actor[3]["text"] = "Nora opens the door, Sam enters first."
        self.assertIsNone(recognize_introductory_transport_overview(wrong_actor))

    def test_failed_attempt_and_explicit_alternative_route_abstain(self) -> None:
        failed = self._ribbon_events()
        failed[2]["text"] = "Sam and Priya try to cross the gallery but turn back."
        self.assertIsNone(recognize_introductory_transport_overview(failed))

        alternate_destination = self._generic_events()
        alternate_destination[2]["text"] = (
            "Nora carries the lantern from the storeroom to the basement instead."
        )
        self.assertIsNone(recognize_introductory_transport_overview(alternate_destination))

    def test_timed_or_dialogue_anchored_parent_is_not_demoted(self) -> None:
        timed = self._ribbon_events()
        timed[0]["source_start_seconds"] = 0.0
        self.assertIsNone(recognize_introductory_transport_overview(timed))

        dialogue = self._ribbon_events()
        dialogue[0]["text"] = (
            'Sam says, “We move one red ribbon spool from a gallery to a workshop.”'
        )
        self.assertIsNone(recognize_introductory_transport_overview(dialogue))

        speaker_cue = self._ribbon_events()
        speaker_cue[0]["text"] = (
            "Sam: Carry one red ribbon spool from a gallery to a workshop."
        )
        self.assertIsNone(recognize_introductory_transport_overview(speaker_cue))

        anchored = self._ribbon_events()
        anchored[0]["dialogue_ids"] = ["D1"]
        self.assertIsNone(recognize_introductory_transport_overview(anchored))

    def test_later_deliberate_return_or_second_trip_keeps_overview_executable(self) -> None:
        return_trip = self._generic_events()
        return_trip.append({
            "event_id": "E5",
            "text": "Later Nora carries the same lantern from the reading room back to the storeroom.",
        })
        self.assertIsNone(recognize_introductory_transport_overview(return_trip))

        repeated_trip = self._generic_events()
        repeated_trip.append({
            "event_id": "E5",
            "text": "Nora carries the lantern from the storeroom to the reading room again.",
        })
        self.assertIsNone(recognize_introductory_transport_overview(repeated_trip))

    def test_missing_explicit_same_prop_reference_abstains(self) -> None:
        events = self._generic_events()
        events[1]["text"] = "At the start Nora waits beside the storeroom door."
        events[3]["text"] = "Felix enters the reading room and reaches the table."
        self.assertIsNone(recognize_introductory_transport_overview(events))

    def test_prop_left_at_origin_or_people_arrive_without_it_abstains(self) -> None:
        left_behind = self._ribbon_events()
        left_behind[2]["text"] = (
            "Sam and Priya leave the spool in the gallery and cross toward the workshop."
        )
        self.assertIsNone(recognize_introductory_transport_overview(left_behind))

        empty_handed = self._ribbon_events()
        empty_handed[5]["text"] = "Only after they reach the workshop bench do they stop."
        self.assertIsNone(recognize_introductory_transport_overview(empty_handed))

    def test_unnamed_messenger_cannot_carry_the_prop_while_cast_arrives(self) -> None:
        events = self._ribbon_events()
        events[5]["text"] = (
            "A messenger carries the same spool into the workshop while Sam and Priya reach the bench."
        )
        self.assertIsNone(recognize_introductory_transport_overview(events))

    def test_overview_with_another_executable_action_is_not_demoted(self) -> None:
        for lead_in in (
            "Sam cuts a label, then Sam and Priya",
            "Sam builds a small crate, then Sam and Priya",
        ):
            with self.subTest(lead_in=lead_in):
                prefix_action = self._ribbon_events()
                prefix_action[0]["text"] = (
                    f"{lead_in} move one red ribbon spool from a gallery to a workshop "
                    "so they can hang a banner."
                )
                self.assertIsNone(recognize_introductory_transport_overview(prefix_action))

        trailing_action = self._ribbon_events()
        trailing_action[0]["text"] = (
            "Sam and Priya move one red ribbon spool from a gallery to a workshop, "
            "then hang the banner."
        )
        self.assertIsNone(recognize_introductory_transport_overview(trailing_action))


if __name__ == "__main__":
    unittest.main()
