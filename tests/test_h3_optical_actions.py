"""Optical-camera clauses must not masquerade as physical cast actions."""

from __future__ import annotations

import re
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
from services.h3_optical_actions import (  # noqa: E402
    optical_only_clauses,
    physical_action_text,
)


CAST = re.compile(r"(?<![\w])(?:Nora|Eli)(?![\w])", re.IGNORECASE)


class OpticalActionTextTests(unittest.TestCase):
    def test_optical_collector_returns_explicit_and_inherited_camera_clauses(self):
        source = (
            "The lens glides past Nora, then cuts to Eli, finally pulls "
            "to an ultra-wide center frame."
        )
        self.assertEqual(
            optical_only_clauses(source, CAST),
            [
                "The lens glides past Nora",
                "then cuts to Eli",
                "finally pulls to an ultra-wide center frame.",
            ],
        )

    def test_optical_collector_excludes_a_camera_clause_with_human_gerund(self):
        source = (
            "The lens glides past Nora, then follows Eli opening the blue case."
        )
        self.assertEqual(
            optical_only_clauses(source, CAST),
            ["The lens glides past Nora"],
        )
        self.assertEqual(
            physical_action_text(source, CAST),
            "Eli opening the blue case.",
        )

    def test_optical_collector_excludes_ambiguous_retained_camera_clause(self):
        source = "The camera is picked up by Nora and rotated toward the window."
        self.assertEqual(optical_only_clauses(source, CAST), [])

    def test_optical_collector_excludes_real_human_pull_and_cut(self):
        source = "Nora pulls the brass lever and Eli cuts the rope."
        self.assertEqual(optical_only_clauses(source, CAST), [])
        self.assertEqual(physical_action_text(source, CAST), source)

    def test_optical_collector_tracks_camera_chain_without_cast_pattern(self):
        source = "The lens glides, then cuts to Nora, finally pulls wide."
        self.assertEqual(
            optical_only_clauses(source),
            ["The lens glides", "then cuts to Nora", "finally pulls wide."],
        )

    def test_lens_pull_to_follow_a_subject_is_not_a_cast_pull(self):
        self.assertEqual(
            physical_action_text("The lens pulls wide to follow Eli.", CAST),
            "",
        )

    def test_camera_chain_without_repeated_subject_is_removed(self):
        source = (
            "The lens glides past Nora, then cuts to Eli, finally pulls "
            "to an ultra-wide center frame."
        )
        self.assertEqual(physical_action_text(source, CAST), "")

    def test_first_quick_cut_and_pull_gerund_camera_chain_is_removed(self):
        source = (
            "Lens first glides past the fist, then quick-cuts to the leg, "
            "finally pulling wide to show a collision course."
        )
        self.assertEqual(physical_action_text(source, CAST), "")

    def test_lens_following_preserves_and_resolves_subordinate_human_action(self):
        self.assertEqual(
            physical_action_text("The lens follows Nora as she opens the door.", CAST),
            "Nora opens the door.",
        )

    def test_camera_follow_preserves_a_visible_gerund_human_action(self):
        cases = (
            ("The camera follows Nora visibly opening the door.", "Nora visibly opening the door."),
            ("The lens first follows Nora opening the door.", "Nora opening the door."),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(physical_action_text(source, CAST), expected)

    def test_pull_to_follow_target_keeps_the_target_s_human_gerund(self):
        self.assertEqual(
            physical_action_text(
                "The lens pulls wide to follow Eli cutting the rope.", CAST,
            ),
            "Eli cutting the rope.",
        )

    def test_follow_participle_is_retained_without_cast_pattern(self):
        self.assertEqual(
            physical_action_text("The camera follows Nora opening the door."),
            "Nora opening the door.",
        )

    def test_parser_keeps_participial_action_without_cast_pattern(self):
        from services import h3_story_ledger as ledger

        frames = ledger._h3_preview_action_frames(
            "The camera follows Nora opening the door.", None,
        )
        self.assertTrue(any(frame[0] == "open" for frame in frames), frames)

    def test_framing_target_alone_remains_optical_without_cast_pattern(self):
        self.assertEqual(physical_action_text("The camera follows Nora."), "")

    def test_relative_human_action_after_framing_target_is_retained(self):
        self.assertEqual(
            physical_action_text("The lens follows Nora who opens the door.", CAST),
            "Nora opens the door.",
        )

    def test_relative_unknown_human_predicate_fails_closed(self):
        self.assertEqual(
            physical_action_text("The lens follows Nora who whistles.", CAST),
            "Nora whistles.",
        )

    def test_camera_follow_with_participial_human_action_is_retained(self):
        self.assertEqual(
            physical_action_text("The camera follows Nora opening the door.", CAST),
            "Nora opening the door.",
        )

    def test_camera_move_after_human_action_keeps_that_action(self):
        self.assertEqual(
            physical_action_text("The camera pulls back after Eli cuts the rope.", CAST),
            "Eli cuts the rope.",
        )

    def test_camera_clause_can_precede_unjoined_human_clause(self):
        self.assertEqual(
            physical_action_text("The camera pulls back, Eli opens the door.", CAST),
            "Eli opens the door.",
        )

    def test_unjoined_human_clause_is_retained_without_cast_pattern(self):
        self.assertEqual(
            physical_action_text("The camera pulls back, Eli opens the door."),
            "Eli opens the door.",
        )

    def test_relative_finite_predicate_survives_without_cast_pattern(self):
        self.assertEqual(
            physical_action_text("The lens follows Nora who whistles."),
            "who whistles.",
        )

    def test_camera_move_does_not_remove_companion_physical_action(self):
        self.assertEqual(
            physical_action_text("The camera pulls back while Eli cuts the rope.", CAST),
            "Eli cuts the rope.",
        )

    def test_camera_clause_after_coordinated_human_actions_is_only_removed_clause(self):
        source = (
            "Nora draws back her right fist, then Eli cuts the rope, "
            "while the camera pans to their hands."
        )
        self.assertEqual(
            physical_action_text(source, CAST),
            "Nora draws back her right fist, then Eli cuts the rope",
        )

    def test_coordinated_human_subjects_remain_verbatim(self):
        source = "Nora draws back her right fist and Eli cuts the rope."
        self.assertEqual(physical_action_text(source, CAST), source)

    def test_real_human_pull_and_cut_actions_remain(self):
        cases = (
            "Nora pulls the brass lever.",
            "Eli cuts the rope.",
            "Nora pulls the lens cap off the camera.",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(physical_action_text(source, CAST), source)

    def test_human_moving_camera_equipment_is_not_an_optical_camera_predicate(self):
        source = "Nora lifts the camera and rotates its lens toward the doorway."
        self.assertEqual(physical_action_text(source, CAST), source)

    def test_unknown_human_predicate_fails_closed(self):
        source = "Nora recalibrates the antique compass while the lens pulls wide."
        self.assertEqual(
            physical_action_text(source, CAST),
            "Nora recalibrates the antique compass",
        )

    def test_unknown_or_passive_camera_predicate_is_not_masked(self):
        cases = (
            "The camera is picked up by Nora and rotated toward the window.",
            "The lens shatters beside the table.",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(physical_action_text(source, CAST), source)

    def test_camera_chain_does_not_leak_across_sentence_boundary(self):
        source = "The lens glides past Nora. Eli cuts the rope."
        self.assertEqual(physical_action_text(source, CAST), "Eli cuts the rope.")


class PreviewParserIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from services import h3_story_ledger as ledger

        cls.ledger = ledger

    def _first_event_violations(self, source: str, action: str) -> list[str]:
        ledger = self.ledger
        events = ledger.extract_source_events(source)
        self.assertGreaterEqual(len(events), 2, events)
        first = events[0]
        beat = {
            "beat_id": "B1",
            "segment": 1,
            "description": first["text"],
            "source_event_ids": [first["event_id"]],
            "dialogue_ids": [],
            "state_after": "Nora remains beside the workbench.",
        }
        segment = {
            "segment": 1,
            "semantic_actions": True,
            "opening_state": "Nora stands beside the workbench.",
            "closing_state": "Nora remains beside the workbench.",
            "shots": [{
                "shot": 1,
                "beat_ids": ["B1"],
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "transition": "continuous composition",
                "action": action,
                "camera": "Wide tracking composition",
                "framing": "Medium view of the workbench",
                "sound_effects": "",
            }],
        }
        return ledger.segment_violations(
            source, segment, segment_number=1, duration=4.0,
            assigned_beats=[beat], dialogue_catalog=[],
        )

    def test_optics_only_do_not_trip_a_future_physical_action_preview(self):
        source = (
            "[0s-4s] Nora waits beside the workbench. "
            "[4s-8s] Eli cuts the blue rope."
        )
        optical_action = (
            "The lens glides past Nora, then cuts to Eli, finally pulls "
            "to an ultra-wide center frame."
        )
        self.assertEqual(
            self.ledger._h3_preview_action_frames(optical_action, CAST),
            [],
        )
        errors = self._first_event_violations(
            source, optical_action,
        )
        self.assertFalse(any("previews later source event" in error for error in errors), errors)

    def test_human_action_after_camera_move_still_trips_future_action_guard(self):
        source = (
            "[0s-4s] Nora waits beside the workbench. "
            "[4s-8s] Eli cuts the blue rope."
        )
        errors = self._first_event_violations(
            source, "The camera pulls back while Eli cuts the blue rope."
        )
        self.assertTrue(any("previews later source event e2" in error.casefold() for error in errors), errors)

    def test_human_gerund_after_camera_follow_still_trips_future_action_guard(self):
        source = (
            "[0s-4s] Nora waits beside the workbench. "
            "[4s-8s] Eli opens the blue case."
        )
        errors = self._first_event_violations(
            source, "The lens pulls wide to follow Eli opening the blue case."
        )
        self.assertTrue(any("previews later source event e2" in error.casefold() for error in errors), errors)

    def test_subordinate_human_owner_and_action_are_preserved_for_guard(self):
        source = (
            "[0s-4s] Nora waits beside the workbench. "
            "[4s-8s] Nora opens the blue case."
        )
        errors = self._first_event_violations(
            source, "The lens follows Nora as she opens the blue case."
        )
        self.assertTrue(any("previews later source event e2" in error.casefold() for error in errors), errors)
        frames = self.ledger._h3_preview_action_frames(
            "The lens follows Nora who opens the blue case.", CAST,
        )
        self.assertTrue(
            any(frame[0] == "open" and "nora" in frame[2] for frame in frames),
            frames,
        )

    def test_camera_follow_does_not_transfer_future_action_ownership(self):
        source = (
            "[0s-4s] Nora waits beside the workbench. "
            "[4s-8s] Eli cuts the blue rope."
        )
        errors = self._first_event_violations(
            source, "The lens follows Nora as she cuts a red paper strip."
        )
        self.assertFalse(any("previews later source event" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
