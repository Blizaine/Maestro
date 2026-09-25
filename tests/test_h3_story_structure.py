"""CPU regressions for conservative H3 source overview relationships."""

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
    executable_h3_source_event_ids,
    normalize_h3_source_relationships,
)
from services.h3_story_ledger import (  # noqa: E402
    _canonicalize_story_ledger,
    _deterministic_ledger,
    _faithful_treatment_schema,
    _h3_overview_coverage_complete,
    extract_source_events,
    ledger_violations,
)


class H3StoryStructureTests(unittest.TestCase):
    def _events(
        self,
        details: list[str],
        *,
        parent: str = "Mara moves the brass lantern from the supply room to the office.",
        later: str | None = None,
    ) -> list[dict[str, str]]:
        events = [{"event_id": "E1", "text": parent}]
        events.extend(
            {"event_id": f"E{index + 2}", "text": text}
            for index, text in enumerate(details)
        )
        if later:
            events.append({"event_id": f"E{len(events) + 1}", "text": later})
        return events

    def _relation(self, events: list[dict[str, str]], detail_ids: list[str] | None = None) -> dict:
        return {
            "relation_id": "R1",
            "relation_type": "overview_of",
            "overview_event_id": "E1",
            "detail_event_ids": detail_ids or [
                event["event_id"] for event in events[1:3]
            ],
            "coverage_status": "complete",
        }

    def _normalize(
        self,
        events: list[dict[str, str]],
        *,
        relation: dict | None = None,
        assigned: list[str] | None = None,
        protected: tuple[str, ...] = (),
        recurring: tuple[str, ...] = (),
    ) -> tuple[list[dict], list[str]]:
        raw = [relation or self._relation(events)]
        return normalize_h3_source_relationships(
            raw,
            events,
            assigned_event_ids=assigned,
            protected_parent_ids=protected,
            recurring_event_ids=recurring,
            coverage_validator=lambda parent, details: _h3_overview_coverage_complete(
                parent,
                details,
                source_events=events,
                cast_names=["Mara", "Nora"],
            ),
        )

    def _valid_route_events(self) -> list[dict[str, str]]:
        return self._events([
            "Mara carries the brass lantern from the supply room down the hall.",
            "Mara carries the brass lantern from the hall into the office.",
        ])

    def test_complete_ordered_overview_is_metadata_and_children_remain_executable(self) -> None:
        events = self._valid_route_events()
        before = deepcopy(events)
        links, diagnostics = self._normalize(
            events,
            assigned=["E2", "E3"],
        )

        self.assertEqual(len(links), 1, diagnostics)
        self.assertEqual(links[0]["overview_event_id"], "E1")
        self.assertEqual(links[0]["detail_event_ids"], ["E2", "E3"])
        self.assertEqual(
            executable_h3_source_event_ids(events, links),
            ["E2", "E3"],
        )
        self.assertEqual(events, before, "the immutable source catalog must not be rewritten")
        self.assertEqual(
            links[0]["source_evidence"]["overview"]["exact_text"],
            events[0]["text"],
        )

    def test_model_claim_without_independent_coverage_proof_is_not_authority(self) -> None:
        events = self._valid_route_events()
        links, diagnostics = normalize_h3_source_relationships(
            [self._relation(events)],
            events,
            assigned_event_ids=["E2", "E3"],
        )
        self.assertEqual(links, [])
        self.assertTrue(any("no semantic coverage validator" in item for item in diagnostics))
        self.assertEqual(executable_h3_source_event_ids(events, links), ["E1", "E2", "E3"])

    def test_wrong_actor_does_not_cover_the_overview(self) -> None:
        events = self._events([
            "Nora carries the brass lantern from the supply room down the hall.",
            "Nora carries the brass lantern into the office.",
        ])
        links, diagnostics = self._normalize(events, assigned=["E2", "E3"])
        self.assertEqual(links, [])
        self.assertTrue(diagnostics)

    def test_wrong_prop_does_not_cover_the_overview(self) -> None:
        events = self._events([
            "Mara carries the red spool from the supply room down the hall.",
            "Mara carries the red spool into the office.",
        ])
        links, diagnostics = self._normalize(events, assigned=["E2", "E3"])
        self.assertEqual(links, [])
        self.assertTrue(diagnostics)

    def test_static_destination_mention_does_not_complete_the_route(self) -> None:
        events = self._events([
            "Mara carries the brass lantern from the supply room down the hall.",
            "Mara holds the brass lantern beside the window.",
        ])
        links, _ = self._normalize(events, assigned=["E2", "E3"])
        self.assertEqual(links, [])

    def test_reversed_route_in_one_detail_is_not_source_to_destination_coverage(self) -> None:
        events = self._events([
            "Mara carries the brass lantern from the office back to the supply room.",
            "Mara holds the brass lantern beside the window.",
        ])
        links, _ = self._normalize(events, assigned=["E2", "E3"])
        self.assertEqual(links, [])

    def test_failed_attempt_does_not_cover_a_successful_overview(self) -> None:
        events = self._events([
            "Mara tries to carry the brass lantern from the supply room but cannot lift it.",
            "The lantern remains on the supply-room shelf.",
        ])
        links, _ = self._normalize(events, assigned=["E2", "E3"])
        self.assertEqual(links, [])

    def test_every_required_overview_action_needs_detail_coverage(self) -> None:
        events = self._events(
            [
                "Mara carries the brass lantern from the supply room down the hall.",
                "Mara carries it into the office and sets it beside the desk.",
            ],
            parent=(
                "Mara carries the brass lantern from the supply room to the office "
                "and hangs it from the wall hook."
            ),
        )
        links, _ = self._normalize(events, assigned=["E2", "E3"])
        self.assertEqual(links, [])

    def test_later_independent_reuse_keeps_overview_executable(self) -> None:
        events = self._valid_route_events()
        events.append({
            "event_id": "E4",
            "text": "Mara carries the brass lantern from the office back to the supply room.",
        })
        links, diagnostics = self._normalize(
            events,
            assigned=["E2", "E3", "E4"],
        )
        self.assertEqual(links, [])
        self.assertTrue(any("later independent action" in item for item in diagnostics))

    def test_dialogue_recurrence_and_incomplete_assignments_cannot_demote_parent(self) -> None:
        events = self._valid_route_events()
        with self.subTest("dialogue anchor"):
            links, _ = self._normalize(events, assigned=["E2", "E3"], protected=("E1",))
            self.assertEqual(links, [])
        with self.subTest("recurrence"):
            links, _ = self._normalize(events, assigned=["E2", "E3"], recurring=("E1",))
            self.assertEqual(links, [])
        with self.subTest("missing detail assignment"):
            links, _ = self._normalize(events, assigned=["E2"])
            self.assertEqual(links, [])
        with self.subTest("duplicate detail assignment"):
            links, _ = self._normalize(events, assigned=["E2", "E2", "E3"])
            self.assertEqual(links, [])

    def test_source_order_and_known_ids_are_required(self) -> None:
        events = self._valid_route_events()
        reversed_details = self._relation(events, ["E3", "E2"])
        links, _ = self._normalize(events, relation=reversed_details, assigned=["E2", "E3"])
        self.assertEqual(links, [])
        unknown = self._relation(events, ["E2", "E9"])
        links, _ = self._normalize(events, relation=unknown, assigned=["E2", "E9"])
        self.assertEqual(links, [])

    def test_explicitly_timed_overview_remains_an_executable_action(self):
        events = self._valid_route_events()
        events[0].update(source_start_seconds=0.0, source_end_seconds=4.0)
        links, diagnostics = self._normalize(events, assigned=["E1", "E2", "E3"])
        self.assertEqual(links, [])
        self.assertTrue(any("explicitly timed" in item for item in diagnostics))

    def test_canonical_schedule_keeps_exact_children_and_full_catalog_evidence(self) -> None:
        prompt = (
            "Mara moves the brass lantern from the supply room to the office. "
            "Mara carries the brass lantern from the supply room down the hall. "
            "Mara carries the brass lantern from the hall into the office."
        )
        events = extract_source_events(prompt)
        self.assertEqual([item["event_id"] for item in events], ["E1", "E2", "E3"])
        canonical = _deterministic_ledger(
            prompt,
            segment_count=2,
            segment_durations=[6.0, 6.0],
            locked_dialogue=[],
            camera_coverage="continuous",
            reference_context="",
        )
        candidate = deepcopy(canonical)
        candidate["source_relationships"] = [self._relation(events)]
        candidate["beats"] = [
            {
                "segment": 1,
                "source_event_ids": ["E1", "E2"],
                "dialogue_ids": [],
                "description": "Mara carries the lantern through the hall.",
                "state_after": "The lantern is in the hall.",
                "sound_effects": "Footsteps and a soft lantern rattle.",
            },
            {
                "segment": 2,
                "source_event_ids": ["E3"],
                "dialogue_ids": [],
                "description": "Mara carries the lantern into the office.",
                "state_after": "The lantern rests in the office.",
                "sound_effects": "The lantern settles on the desk.",
            },
        ]

        ledger = _canonicalize_story_ledger(
            prompt,
            canonical,
            candidate,
            locked_dialogue=[],
            segment_count=2,
        )
        assigned_ids = [
            event_id
            for beat in ledger["beats"]
            for event_id in beat["source_event_ids"]
        ]
        self.assertEqual(assigned_ids, ["E2", "E3"])
        self.assertEqual(len(ledger["source_relationships"]), 1)
        self.assertEqual(
            ledger["source_relationships"][0]["source_evidence"]["overview"]["exact_text"],
            events[0]["text"],
        )
        self.assertEqual(
            ledger_violations(
                prompt,
                ledger,
                segment_count=2,
                locked_dialogue=[],
                expect_dialogue=False,
            ),
            [],
        )

    def test_faithful_schema_accepts_metadata_without_requiring_it(self) -> None:
        events = self._valid_route_events()
        schema = _faithful_treatment_schema(
            ["Mara"],
            source_event_ids=[event["event_id"] for event in events],
            source_events=events,
        )
        self.assertIn("source_relationships", schema["properties"])
        self.assertNotIn("state_transitions", schema["properties"])
        self.assertNotIn("source_relationships", schema["required"])
        self.assertNotIn("state_transitions", schema["required"])

    def test_source_relationships_do_not_erase_discriminating_props_or_attachment_targets(self):
        for parent, details in (
            ("Mara moves the blue case from the gallery to the table.", [
                "Mara carries the red case from the gallery through the hall.",
                "Mara carries the red case from the hall to the table.",
            ]),
            ("Mara ties the banner securely to the railing.", [
                "Mara hangs the banner on a wall hook.",
                "Mara steps away from the hook.",
            ]),
            ("Mara ties the banner to the railing.", [
                "Mara unties the banner from the railing.",
                "Mara ties the banner to the railing again.",
            ]),
        ):
            with self.subTest(parent=parent):
                links, _ = self._normalize(self._events(details, parent=parent))
                self.assertEqual(links, [])

    def test_unrelated_collective_pronoun_cannot_replace_a_named_action_owner(self):
        events = self._events([
            "Mara hands Nora the blue case; they watch.",
            "Nora carries the red crate from the supply room down the hall.",
            "Nora carries the red crate from the hall to the office.",
        ], parent=(
            "Mara hands Nora the blue case; they watch as Mara carries the red crate "
            "from the supply room to the office."
        ))
        links, _ = self._normalize(events, relation=self._relation(events, ["E2", "E3", "E4"]))
        self.assertEqual(links, [])


if __name__ == "__main__":
    unittest.main()
