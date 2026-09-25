import re
import unittest

from services.h3_story_ledger import (
    _h3_missing_relation_markers,
    _h3_relation_preserved,
    _h3_relation_preserved,
    extract_source_events,
    segment_violations,
)


class OrderedActionChronologyTests(unittest.TestCase):
    def setUp(self):
        self.cast = re.compile(r"\b(?:Sam|Priya|Mara|Dev|Nora|Len)\b", re.I)

    def missing(self, source, action):
        return _h3_missing_relation_markers(source, action, self.cast)

    def test_c6_workbench_reach_then_handoff_is_evidenced_by_action_order(self):
        prompt = (
            "Two adult exhibit volunteers, Sam and Priya, move one red ribbon spool "
            "from a gallery to a workshop. Only after they reach the workshop bench "
            "does Sam hand the same spool to Priya."
        )
        source = extract_source_events(prompt)[-1]["text"]
        action = "They reach the workshop bench. Sam hands the same spool to Priya."
        self.assertEqual(self.missing(source, action), [])
        self.assertTrue(_h3_relation_preserved(source, action, "only after", self.cast))
        self.assertTrue(_h3_relation_preserved(source, action, "only after", self.cast))
        self.assertEqual(
            self.missing(
                source,
                "Sam and Priya reach the workshop bench. "
                "Sam hands the same spool to Priya.",
            ),
            [],
        )

        # Exercise the production validator call path, including the cast
        # pattern and pronoun context passed into relation validation.
        events = extract_source_events(prompt)
        beat = {
            "beat_id": "B1", "description": source,
            "source_event_ids": [events[-1]["event_id"]],
            "dialogue_ids": [], "state_after": "Priya holds the spool at the bench",
        }
        segment = {
            "segment": 1, "semantic_actions": True,
            "opening_state": "Sam and Priya approach the workbench with the spool.",
            "closing_state": "Priya holds the spool at the workbench.",
            "shots": [{
                "shot": 1, "start_seconds": 0, "end_seconds": 8,
                "action": action, "beat_ids": ["B1"], "dialogue": [],
            }],
        }
        errors = segment_violations(
            prompt, segment, segment_number=1, duration=8,
            assigned_beats=[beat], dialogue_catalog=[],
        )
        self.assertFalse(any("chronology relation" in error for error in errors), errors)

    def test_distinct_committed_clauses_can_show_before_relation_without_marker(self):
        source = "Mara unlocks the blue gate before Dev enters the garden."
        action = "Mara unlocks the blue gate. Dev enters the garden."
        self.assertEqual(self.missing(source, action), [])

    def test_reversed_or_repeated_early_later_action_does_not_pass(self):
        source = "Mara unlocks the blue gate before Dev enters the garden."
        for action in (
            "Dev enters the garden. Mara unlocks the blue gate.",
            "Dev enters the garden. Mara unlocks the blue gate. Dev enters again.",
        ):
            with self.subTest(action=action):
                self.assertEqual(self.missing(source, action), ["before"])
                self.assertFalse(_h3_relation_preserved(source, action, "before", self.cast))
                self.assertFalse(_h3_relation_preserved(source, action, "before", self.cast))

    def test_same_clause_or_simultaneous_actions_do_not_prove_sequence(self):
        source = "Mara unlocks the blue gate before Dev enters the garden."
        for action in (
            "Mara unlocks the blue gate and Dev enters the garden.",
            "Mara unlocks the blue gate while Dev enters the garden.",
            "Mara unlocks the blue gate as Dev enters the garden.",
            "Mara unlocks the blue gate simultaneously with Dev entering the garden.",
        ):
            with self.subTest(action=action):
                self.assertEqual(self.missing(source, action), ["before"])

    def test_actor_and_affected_object_must_match_both_anchors(self):
        source = "Mara unlocks the blue gate before Dev enters the garden."
        controls = (
            ("Nora unlocks the blue gate. Dev enters the garden.", ["before"]),
            ("Mara unlocks the red chest. Dev enters the garden.", ["before"]),
            ("Mara unlocks the blue gate. Len enters the garden.", ["before"]),
            ("Mara unlocks the blue gate. Dev enters the gallery.", ["before"]),
            ("Dev enters the garden. Mara unlocks the red chest.", ["before"]),
        )
        for action, expected in controls:
            with self.subTest(action=action):
                self.assertEqual(self.missing(source, action), expected)

    def test_plural_group_anchor_requires_coordinated_subject_not_bystander(self):
        source = "Only after they reach the workshop bench does Sam hand the same spool to Priya."
        for action in (
            "Sam reaches the workshop bench while looking back at Priya. "
            "Sam hands the same spool to Priya.",
            "Sam reaches the workshop bench near Priya. "
            "Sam hands the same spool to Priya.",
        ):
            with self.subTest(action=action):
                self.assertEqual(self.missing(source, action), ["only after"])
        self.assertEqual(
            self.missing(
                source,
                "Sam and Priya reach the garden bench. "
                "Sam hands the same spool to Priya.",
            ),
            ["only after"],
        )

    def test_negated_conditional_and_intended_actions_are_not_evidence(self):
        source = "Mara unlocks the blue gate before Dev enters the garden."
        controls = (
            "Mara does not unlock the blue gate. Dev enters the garden.",
            "Mara may unlock the blue gate. Dev enters the garden.",
            "If Mara unlocks the blue gate, Dev enters the garden.",
            "Mara tries to unlock the blue gate. Dev enters the garden.",
            "Mara plans to unlock the blue gate. Dev enters the garden.",
            "Mara unlocks the blue gate. Dev might enter the garden.",
        )
        for action in controls:
            with self.subTest(action=action):
                self.assertEqual(self.missing(source, action), ["before"])

    def test_missing_physical_anchor_keeps_chronology_error(self):
        source = "Mara unlocks the blue gate before Dev enters the garden."
        self.assertEqual(self.missing(source, "Dev enters the garden."), ["before"])

    def test_explicit_temporal_markers_keep_existing_reversal_checks(self):
        source = "Nora opens the blue case after Mara unlocks its brass clasp."
        self.assertEqual(
            self.missing(source, "After Mara unlocks its brass clasp, Nora opens the blue case."),
            [],
        )
        self.assertEqual(
            self.missing(source, "Before Mara unlocks its brass clasp, Nora opens the blue case."),
            ["after"],
        )


if __name__ == "__main__":
    unittest.main()
