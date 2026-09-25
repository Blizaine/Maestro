"""Focused ownership controls for source and visible H3 action frames."""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import (
    _h3_contract_clauses,
    _h3_action_actor_overlap,
    _h3_is_averted_impact_clause,
    _h3_preview_action_frames,
    _h3_required_action_frames_covered,
    _h3_unique_source_pronoun_antecedents,
)


class H3ActionFrameRegressionTests(unittest.TestCase):
    def test_unique_event_actor_resolves_pronoun_after_clause_split(self):
        source = "A courier crosses the square; then she opens the gate."
        action_part = next(
            part for part in _h3_contract_clauses(source)
            if part.startswith("she opens")
        )
        antecedents = _h3_unique_source_pronoun_antecedents(
            source, action_part, None,
        )

        self.assertEqual(antecedents, {"she": "role:a courier"})
        required = _h3_preview_action_frames(
            action_part, None, pronoun_antecedents=antecedents,
        )
        visible = _h3_preview_action_frames(
            "A courier crosses the square and opens the gate.", None,
        )
        self.assertTrue(_h3_required_action_frames_covered(
            required, visible, source_text=action_part,
        ))

    def test_ambiguous_named_actors_do_not_resolve_the_source_pronoun(self):
        source = "Mara enters the hall; Len approaches the gate; she opens the gate."
        cast_pattern = re.compile(r"\b(?:Mara|Len)\b", re.IGNORECASE)
        action_part = next(
            part for part in _h3_contract_clauses(source)
            if part.startswith("she opens")
        )
        antecedents = _h3_unique_source_pronoun_antecedents(
            source, action_part, cast_pattern,
        )

        self.assertEqual(antecedents, {})
        required = _h3_preview_action_frames(action_part, cast_pattern)
        wrong_owner = _h3_preview_action_frames(
            "Len opens the gate.", cast_pattern,
        )
        self.assertIs(
            _h3_required_action_frames_covered(
                required, wrong_owner, source_text=action_part,
            ),
            False,
        )

    def test_named_actor_is_not_unique_when_another_role_is_in_the_event(self):
        cast_pattern = re.compile(r"\bMara\b", re.IGNORECASE)
        ambiguous = "Mara confronts a courier; she opens the gate."
        ambiguous_part = next(
            part for part in _h3_contract_clauses(ambiguous)
            if part.startswith("she opens")
        )
        self.assertEqual(
            _h3_unique_source_pronoun_antecedents(
                ambiguous, ambiguous_part, cast_pattern,
            ),
            {},
        )

        appositive_then_other = "Mara, a courier, confronts a guard; she opens the gate."
        self.assertEqual(
            _h3_unique_source_pronoun_antecedents(
                appositive_then_other, "she opens the gate.", cast_pattern,
            ),
            {},
        )

        profile = "Mara is a courier; she opens the gate."
        profile_part = next(
            part for part in _h3_contract_clauses(profile)
            if part.startswith("she opens")
        )
        self.assertEqual(
            _h3_unique_source_pronoun_antecedents(
                profile, profile_part, cast_pattern,
            ),
            {"she": "mara"},
        )

    def test_role_actor_compatibility_rejects_shared_adjectives_or_conflicting_modifiers(self):
        self.assertTrue(_h3_action_actor_overlap(
            frozenset({"role:the foam"}),
            frozenset({"role:a white foam column"}),
        ))
        self.assertFalse(_h3_action_actor_overlap(
            frozenset({"role:white foam"}),
            frozenset({"role:white courier"}),
        ))
        self.assertFalse(_h3_action_actor_overlap(
            frozenset({"role:red-coated courier"}),
            frozenset({"role:blue-coated courier"}),
        ))

    def test_physical_effect_frames_keep_subject_and_reject_wrong_actor_or_object(self):
        source = (
            "The foam strikes the courier from below and launches him up the road."
        )
        required = _h3_preview_action_frames(source, None)
        source_launch = next(frame for frame in required if frame[0] == "launch")
        self.assertEqual(source_launch[2], frozenset({"role:the foam"}))

        visible = _h3_preview_action_frames(
            "A white foam column erupts between the trucks, hits the courier "
            "from below and launches him up the road.",
            None,
        )
        self.assertTrue(_h3_required_action_frames_covered(
            required, visible, source_text=source,
        ))

        wrong_actor = _h3_preview_action_frames(
            "A truck hits the courier from below and launches him up the road.",
            None,
        )
        self.assertIs(
            _h3_required_action_frames_covered(
                required, wrong_actor, source_text=source,
            ),
            False,
        )
        wrong_object = _h3_preview_action_frames(
            "The foam launches a crate into the sky.",
            None,
        )
        self.assertIs(
            _h3_required_action_frames_covered(
                required, wrong_object, source_text=source,
            ),
            False,
        )

    def test_abstract_movement_lead_in_does_not_replace_concrete_actions(self):
        source = (
            "The fighter launches into an extraordinary martial-arts movement. "
            "She swings her sword while rotating through the air. "
            "A golden energy arc sweeps around her."
        )
        visible = (
            "The fighter leaps and rotates through the air while swinging her sword. "
            "A solid radiant golden energy arc sweeps around her."
        )
        required = _h3_preview_action_frames(source, None)
        visible_frames = _h3_preview_action_frames(visible, None)
        self.assertTrue(_h3_required_action_frames_covered(
            required, visible_frames, source_text=source,
        ))

        missing_sword = _h3_preview_action_frames(
            "The fighter leaps and rotates through the air.", None,
        )
        self.assertIs(
            _h3_required_action_frames_covered(
                required, missing_sword, source_text=source,
            ),
            False,
        )
        unrelated_subject = _h3_preview_action_frames(
            "A crate launches into the air.", None,
        )
        unrelated_leap = _h3_preview_action_frames(
            "Mara leaps into the air.", None,
        )
        self.assertIs(
            _h3_required_action_frames_covered(
                unrelated_subject, unrelated_leap,
                source_text="A crate launches into the air.",
            ),
            False,
        )

    def test_reduced_motion_clause_keeps_its_unique_subject(self):
        source = "The fighter enters the ring. She rotates through the air."
        antecedents = _h3_unique_source_pronoun_antecedents(
            source, source, None,
        )
        required = next(
            frame for frame in _h3_preview_action_frames(
                source, None, pronoun_antecedents=antecedents,
            )
            if frame[0] == "rotate"
        )
        wrong_actor = _h3_preview_action_frames(
            "A guard rotates through the air.", None,
        )
        self.assertEqual(required[2], frozenset({"role:the fighter"}))
        self.assertIs(
            _h3_required_action_frames_covered(
                [required], wrong_actor, source_text=source,
            ),
            False,
        )

    def test_loop_open_audio_label_is_not_a_door_action(self):
        self.assertEqual(
            _h3_preview_action_frames(
                "Loop-open: empty horizon, roar still going.", None,
            ),
            [],
        )
        self.assertTrue(any(
            frame[0] == "open"
            for frame in _h3_preview_action_frames(
                "Loop-open: The courier opens the gate.", None,
            )
        ))

    def test_open_door_adjective_does_not_assign_an_opening_to_a_bystander(self):
        cast_pattern = re.compile(r"\b(?:Sam|Priya)\b", re.IGNORECASE)
        required = _h3_preview_action_frames("Priya opens the door.", cast_pattern)
        visible_text = (
            "Sam pulls the door open. Priya stands near the now-open door, "
            "facing back toward the gallery entrance."
        )
        visible = _h3_preview_action_frames(visible_text, cast_pattern)
        self.assertFalse(any(
            frame[0] == "open" and frame[2] == frozenset({"priya"})
            for frame in visible
        ))
        self.assertIs(
            _h3_required_action_frames_covered(
                required, visible, source_text="Priya opens the door.",
            ),
            False,
        )

        # Resultatives and explicit opening predicates remain physical actions.
        for action in (
            "Sam pulls the door open.",
            "Priya swings it open.",
            "Priya swings the door open.",
            "Priya opens the door.",
        ):
            with self.subTest(action=action):
                self.assertTrue(any(
                    frame[0] == "open"
                    for frame in _h3_preview_action_frames(action, cast_pattern)
                ))
        sam_opening = _h3_preview_action_frames("Sam opens the door.", cast_pattern)
        sam_resultative = _h3_preview_action_frames("Sam pulls the door open.", cast_pattern)
        self.assertTrue(_h3_required_action_frames_covered(
            sam_opening, sam_resultative, source_text="Sam opens the door.",
        ))
        for static_description in (
            "Priya stands beside the open gate.",
            "Priya stands beside the already-open gate.",
            "Priya stands beside the now-open gate.",
        ):
            with self.subTest(static_description=static_description):
                self.assertFalse(any(
                    frame[0] == "open"
                    for frame in _h3_preview_action_frames(static_description, cast_pattern)
                ))

    def test_coordinated_door_state_is_not_a_locking_action(self):
        cast_pattern = re.compile(r"\b(?:Ada|Len)\b", re.IGNORECASE)
        for text in (
            "Ada and Len stand near the closed, locked archive door.",
            "Ada observes the closed and locked door.",
            "Ada studies the already closed, locked gate.",
        ):
            with self.subTest(text=text):
                self.assertFalse(any(
                    frame[0] in {"close", "lock"}
                    for frame in _h3_preview_action_frames(text, cast_pattern)
                ))
        for text in (
            "Ada closed and locked the door.",
            "Ada closes the door and locks it.",
            "Ada locks the archive door.",
            "The door is being locked by Ada.",
        ):
            with self.subTest(text=text):
                self.assertTrue(any(
                    frame[0] == "lock"
                    for frame in _h3_preview_action_frames(text, cast_pattern)
                ))

    def test_caught_impact_is_not_a_performed_step_but_other_before_actions_are(self):
        cast_pattern = re.compile(r"\bNora\b", re.IGNORECASE)
        for caught in (
            "Nora catches a tumbling lantern before it strikes the ground.",
            "Nora caught the lantern before it struck the ground.",
            "Nora is catching the lantern before it hits the ground.",
        ):
            with self.subTest(caught=caught):
                caught_parts = _h3_contract_clauses(caught)
                impact_part = next(
                    part for part in caught_parts
                    if re.search(r"\b(?:strike|struck|hit)", part, re.IGNORECASE)
                    and "ground" in part
                )
                self.assertTrue(_h3_is_averted_impact_clause(
                    caught, impact_part, cast_pattern,
                ))
                self.assertIn("catch", {
                    frame[0] for frame in _h3_preview_action_frames(
                        caught_parts[0], cast_pattern,
                    )
                })

        nonassertive_catches = (
            "Nora misses catching the lantern before it strikes the ground.",
            "Nora tries to catch the lantern before it strikes the ground.",
            "Nora attempts to catch the lantern before it strikes the ground.",
            "Nora reaches for the lantern to catch it before it strikes the ground.",
            "Nora fails to catch the lantern before it strikes the ground.",
            "Nora cannot catch the lantern before it strikes the ground.",
            "Nora would catch the lantern before it strikes the ground.",
            "Nora could catch the lantern before it strikes the ground.",
            "Nora may catch the lantern before it strikes the ground.",
            "Nora might catch the lantern before it strikes the ground.",
            "If Nora catches the lantern before it strikes the ground, she saves it.",
            "Unless Nora catches the lantern before it strikes the ground, it breaks.",
            "Nora is instructed to catch the lantern before it strikes the ground.",
            "Catch the lantern before it strikes the ground.",
            "Nora, catch the lantern before it strikes the ground.",
        )
        for source in nonassertive_catches:
            with self.subTest(source=source):
                parts = _h3_contract_clauses(source)
                impact_part = next(
                    part for part in parts
                    if re.search(r"\b(?:strike|struck|hit)", part, re.IGNORECASE)
                    and "ground" in part
                )
                self.assertFalse(_h3_is_averted_impact_clause(
                    source, impact_part, cast_pattern,
                ))

        # Do not reinterpret order: an impact that happens before the catch is
        # still an action requirement, and catching another object does not
        # avert the named lantern's impact.
        impact_first = "The lantern strikes the ground before Nora catches it."
        impact_first_part = next(
            part for part in _h3_contract_clauses(impact_first)
            if "strikes the ground" in part
        )
        self.assertFalse(_h3_is_averted_impact_clause(
            impact_first, impact_first_part, cast_pattern,
        ))
        wrong_object = "Nora catches a crate before the lantern strikes the ground."
        wrong_object_part = next(
            part for part in _h3_contract_clauses(wrong_object)
            if "strikes the ground" in part
        )
        self.assertFalse(_h3_is_averted_impact_clause(
            wrong_object, wrong_object_part, cast_pattern,
        ))
        failed_catch = "Nora does not catch the lantern before it strikes the ground."
        failed_catch_part = next(
            part for part in _h3_contract_clauses(failed_catch)
            if part.startswith("it strikes")
        )
        self.assertFalse(_h3_is_averted_impact_clause(
            failed_catch, failed_catch_part, cast_pattern,
        ))

        key_handoff = "Nora closes the door before she hands the key to Len."
        handoff_part = next(
            part for part in _h3_contract_clauses(key_handoff)
            if part.startswith("she hands")
        )
        self.assertFalse(_h3_is_averted_impact_clause(
            key_handoff, handoff_part, re.compile(r"\bNora|Len\b", re.IGNORECASE),
        ))


if __name__ == "__main__":
    unittest.main()
