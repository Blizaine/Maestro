"""Small provenance checks for H3 source-event handoffs."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_state import (  # noqa: E402
    format_h3_accepted_history_context,
    format_h3_authoritative_handoff,
    project_h3_achieved_state,
)


class H3AchievedStateTests(unittest.TestCase):
    def test_only_the_ordered_completed_prefix_is_projected(self):
        events = [
            {"event_id": "E1", "text": "Nora enters the archive."},
            {"event_id": "E2", "text": "Nora retrieves the map."},
            {"event_id": "E3", "text": "Nora returns the key."},
        ]

        snapshot = project_h3_achieved_state(events, ["E1", "E3", "E99"])

        self.assertEqual(snapshot["completed_source_event_ids"], ["E1"])
        self.assertEqual(snapshot["covered_overview_ids"], [])
        self.assertTrue(any("E99" in item for item in snapshot["diagnostics"]))
        self.assertTrue(any("E3" in item for item in snapshot["diagnostics"]))
        self.assertEqual(set(snapshot), {
            "completed_source_event_ids", "covered_overview_ids",
            "completed_source_event_quotes", "accepted_visible_camera_actions",
            "source_prop_identities", "current_window_index", "diagnostics",
        })

    def test_overview_parent_waits_for_all_ordered_detail_events(self):
        events = [
            {"event_id": "E1", "text": "Nora completes the archive task."},
            {"event_id": "E2", "text": "Nora enters the archive."},
            {"event_id": "E3", "text": "Nora retrieves the map."},
            {"event_id": "E4", "text": "Nora returns the key."},
        ]
        relationships = {"E1": ["E2", "E3"]}

        parent_only = project_h3_achieved_state(events, ["E1"],
                                                validated_overview_details=relationships)
        incomplete = project_h3_achieved_state(events, ["E2"],
                                               validated_overview_details=relationships)
        complete_details = project_h3_achieved_state(events, ["E1", "E2", "E3"],
                                                      validated_overview_details=relationships)

        self.assertEqual(parent_only["completed_source_event_ids"], [])
        self.assertEqual(parent_only["covered_overview_ids"], [])
        self.assertTrue(any("E1" in item and "non-executable" in item
                            for item in parent_only["diagnostics"]))
        self.assertEqual(incomplete["completed_source_event_ids"], ["E2"])
        self.assertEqual(incomplete["covered_overview_ids"], [])
        self.assertEqual(complete_details["completed_source_event_ids"], ["E2", "E3"])
        self.assertEqual(complete_details["covered_overview_ids"], ["E1"])

    def test_malformed_overview_order_cannot_cover_parent(self):
        events = [
            {"event_id": "E1", "text": "The courier completes the route."},
            {"event_id": "E2", "text": "The courier leaves the station."},
            {"event_id": "E3", "text": "The courier returns to the station."},
        ]

        snapshot = project_h3_achieved_state(
            events,
            ["E1", "E2", "E3"],
            validated_overview_details={"E1": ["E3", "E2"]},
        )

        self.assertEqual(snapshot["completed_source_event_ids"], ["E1", "E2", "E3"])
        self.assertEqual(snapshot["covered_overview_ids"], [])
        self.assertTrue(any("out of source order" in item for item in snapshot["diagnostics"]))

    def test_handoff_preserves_the_whole_accepted_action_without_state_inference(self):
        events = [
            {"event_id": "E1", "text": "Mara has not opened the safe."},
            {"event_id": "E2", "text": "Mara returns the key to the drawer."},
        ]
        snapshot = project_h3_achieved_state(events, ["E1"])
        # Legacy/untrusted metadata is ignored; this helper does not project
        # property proposals or interpret the wording of an action.
        snapshot["closing_state"] = "The key is still in Mara's hand."
        action = "Mara has not opened the safe; she returns the key to the drawer."

        handoff = format_h3_authoritative_handoff(
            snapshot,
            latest_accepted_visible_camera_action=action,
        )

        self.assertEqual(
            handoff,
            f"Continue from the visible result of: {action} Do not repeat it.",
        )
        self.assertIn("has not opened", handoff)
        self.assertIn("returns the key to the drawer", handoff)
        self.assertNotIn("still in Mara's hand", handoff)
        self.assertNotIn("E1", handoff)

    def test_authoritative_handoff_keeps_long_materialized_action_whole(self):
        action = (
            "Character A's strike collides with Character B's charged kick; "
            + "the platform fractures and the final physical consequence remains visible. " * 40
        ).strip()
        self.assertGreater(len(action), 600)
        snapshot = project_h3_achieved_state([], [])

        handoff = format_h3_authoritative_handoff(
            snapshot,
            latest_accepted_visible_camera_action=action,
        )

        self.assertEqual(
            handoff,
            f"Continue from the visible result of: {action} Do not repeat it.",
        )
        self.assertIn("the final physical consequence remains visible.", handoff)
        self.assertEqual(
            format_h3_authoritative_handoff(
                snapshot,
                latest_accepted_visible_camera_action="x" * 40_001,
            ),
            "",
        )

    def test_empty_handoff_does_not_invent_an_opening_or_endpoint(self):
        snapshot = project_h3_achieved_state([], [])

        self.assertEqual(format_h3_authoritative_handoff(snapshot), "")
        self.assertEqual(
            format_h3_authoritative_handoff(
                snapshot,
                latest_accepted_visible_camera_action="   ",
            ),
            "",
        )
        self.assertEqual(
            format_h3_authoritative_handoff(
                None,
                latest_accepted_visible_camera_action="Nora takes the map.",
            ),
            "",
        )

    def test_camera_history_context_keeps_source_prop_identity_separate_from_background(self):
        events = [
            {
                "event_id": "E1",
                "text": "Priya brings one wooden stand and the paper program to the hall.",
            },
            {"event_id": "E2", "text": "Priya sets the program on a low stone pedestal."},
        ]
        snapshot = project_h3_achieved_state(
            events,
            ["E1"],
            current_window_index=2,
            accepted_visible_camera_actions=[
                {
                    "accepted": True,
                    "window_index": 1,
                    "source_event_ids": ["E1"],
                    "action": "Sam places the paper program on the wooden stand.",
                },
            ],
            source_prop_identities=[
                {
                    "prop_id": "P1",
                    "identity": "wooden stand",
                    "source_event_id": "E1",
                    "source_quote": "one wooden stand",
                },
                {
                    "prop_id": "P2",
                    "identity": "paper program",
                    "source_event_id": "E1",
                    "source_quote": "paper program",
                },
                {
                    "prop_id": "P3",
                    "identity": "stone pedestal",
                    "source_event_id": "E2",
                    "source_quote": "low stone pedestal",
                },
            ],
        )
        snapshot["closing_state"] = "The program rests on the stone pedestal."

        # Prior history is deliberately kept out of the concise native handoff.
        self.assertEqual(format_h3_authoritative_handoff(snapshot), "")
        context = format_h3_accepted_history_context(snapshot)

        self.assertIn("wooden stand", context)
        self.assertIn("paper program", context)
        self.assertIn("Sam places the paper program on the wooden stand", context)
        self.assertNotIn("stone pedestal", context)
        self.assertNotIn("rests on", context)
        self.assertIn("no position or holder is implied", context)

    def test_history_orders_accepted_actions_and_prefers_latest_transfer(self):
        events = [
            {"event_id": "E1", "text": "Sam holds the folded print."},
            {"event_id": "E2", "text": "Sam hands the folded print to Priya."},
            {"event_id": "E3", "text": "Priya reads the folded print."},
        ]
        snapshot = project_h3_achieved_state(
            events,
            ["E1", "E2"],
            current_window_index=3,
            accepted_visible_camera_actions=[
                {
                    "accepted": True,
                    "window_index": 1,
                    "source_event_ids": ["E1"],
                    "action": "Sam holds the folded print at the doorway.",
                },
                {
                    "accepted": True,
                    "window_index": 2,
                    "source_event_ids": ["E2"],
                    "action": "Sam hands the folded print to Priya.",
                },
            ],
            source_prop_identities=[
                {
                    "prop_id": "P1",
                    "identity": "folded print",
                    "source_event_id": "E1",
                    "source_quote": "the folded print",
                },
            ],
        )

        context = format_h3_accepted_history_context(snapshot)

        self.assertLess(context.index("window 1:"), context.index("window 2:"))
        self.assertIn("Sam holds the folded print at the doorway", context)
        self.assertIn("Sam hands the folded print to Priya", context)
        self.assertIn("latest accepted visible action takes precedence", context)
        self.assertIn("Source-quoted prop identity only", context)
        self.assertNotIn("Priya holds", context)

    def test_unaccepted_or_current_window_moves_do_not_enter_history(self):
        events = [{"event_id": "E1", "text": "Nora enters with the key."}]
        snapshot = project_h3_achieved_state(
            events,
            ["E1"],
            current_window_index=2,
            accepted_visible_camera_actions=[
                {
                    "accepted": True,
                    "window_index": 1,
                    "source_event_ids": ["E1"],
                    "action": "Nora enters with the key.",
                },
                {
                    "accepted": False,
                    "window_index": 1,
                    "source_event_ids": ["E1"],
                    "action": "Mara takes the key from Nora.",
                },
                {
                    "accepted": True,
                    "window_index": 1,
                    "source_event_ids": ["??"],
                    "action": "The invalid-ID card transfers the key to Mara.",
                },
                {
                    "accepted": True,
                    "window_index": 1,
                    "source_event_ids": ["E99"],
                    "action": "The unknown-ID card transfers the key to Mara.",
                },
                {
                    "accepted": True,
                    "window_index": 2,
                    "source_event_ids": [],
                    "action": "Mara carries the key to the desk.",
                },
            ],
        )

        context = format_h3_accepted_history_context(snapshot)

        self.assertIn("Nora enters with the key", context)
        self.assertNotIn("Mara takes the key", context)
        self.assertNotIn("invalid-ID card", context)
        self.assertNotIn("unknown-ID card", context)
        self.assertNotIn("Mara carries the key", context)
        self.assertEqual(snapshot["completed_source_event_ids"], ["E1"])
        self.assertTrue(any("malformed source IDs" in item for item in snapshot["diagnostics"]))
        self.assertTrue(any("unknown source IDs" in item for item in snapshot["diagnostics"]))

    def test_prior_connective_action_with_empty_source_ids_is_context_only(self):
        snapshot = project_h3_achieved_state(
            [],
            [],
            current_window_index=2,
            accepted_visible_camera_actions=[
                {
                    "accepted": True,
                    "window_index": 1,
                    "source_event_ids": [],
                    "action": "The camera settles on the empty gallery doorway.",
                },
                {
                    "accepted": True,
                    "window_index": 2,
                    "source_event_ids": [],
                    "action": "The camera moves to the workshop table.",
                },
            ],
        )

        context = format_h3_accepted_history_context(snapshot)

        self.assertEqual(snapshot["completed_source_event_ids"], [])
        self.assertIn("The camera settles on the empty gallery doorway", context)
        self.assertNotIn("The camera moves to the workshop table", context)

    def test_source_quotes_and_prop_identities_need_exact_known_provenance(self):
        events = [{"event_id": "E1", "text": "Mara carries one brass key."}]
        snapshot = project_h3_achieved_state(
            events,
            ["E1"],
            current_window_index=2,
            source_prop_identities=[
                {
                    "prop_id": "P1",
                    "identity": "brass key",
                    "source_event_id": "E1",
                    "source_quote": "one brass key",
                },
                {
                    "prop_id": "P2",
                    "identity": "stone pedestal",
                    "source_event_id": "E1",
                    "source_quote": "the pedestal",
                },
                {
                    "prop_id": "P3",
                    "identity": "silver key",
                    "source_event_id": "E99",
                    "source_quote": "silver key",
                },
            ],
        )
        # Formatter output is limited to exact completed source texts and the
        # source quote validated for the one allowed identity.
        snapshot["completed_source_event_quotes"].append(
            {"source_event_id": "E99", "text": "Mara places a silver key on a pedestal."}
        )

        context = format_h3_accepted_history_context(snapshot)

        self.assertIn("brass key", context)
        self.assertNotIn("stone pedestal", context)
        self.assertNotIn("silver key", context)
        self.assertIn("Mara carries one brass key", context)

    def test_camera_history_context_has_fixed_entry_and_text_limits(self):
        events = [
            {
                "event_id": f"E{index}",
                "text": f"Nora carries prop {index}. " + ("x" * 390),
            }
            for index in range(1, 7)
        ]
        snapshot = project_h3_achieved_state(
            events,
            [f"E{index}" for index in range(1, 7)],
            current_window_index=7,
            accepted_visible_camera_actions=[
                {
                    "accepted": True,
                    "window_index": index,
                    "source_event_ids": [f"E{index}"],
                    "action": f"Nora carries prop {index}. " + ("y" * 570),
                }
                for index in range(1, 7)
            ],
            source_prop_identities=[
                {
                    "prop_id": f"P{index}",
                    "identity": f"prop {index}",
                    "source_event_id": f"E{index}",
                    "source_quote": f"prop {index}",
                }
                for index in range(1, 6)
            ],
        )

        context = format_h3_accepted_history_context(snapshot)

        self.assertEqual(len(snapshot["accepted_visible_camera_actions"]), 4)
        self.assertEqual(
            [item["source_event_id"] for item in snapshot["completed_source_event_quotes"]],
            ["E4", "E5", "E6"],
        )
        self.assertEqual(len(snapshot["source_prop_identities"]), 4)
        self.assertNotIn("window 1:", context)
        self.assertIn("window 3:", context)
        self.assertLessEqual(len(context), 4800)



if __name__ == "__main__":
    unittest.main()
