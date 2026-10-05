"""Conservative shorthand expansion for source-cast action comparisons."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.h3_action_identity import expand_labeled_actor_shorthand
from services.h3_story_ledger import (
    _canonicalize_segment_contract,
    _h3_preview_action_frames,
    _h3_preview_action_matches,
    _h3_source_cast_pattern,
    extract_source_events,
    segment_violations,
)


def _preview_violations(source: str, visible_action: str) -> list[str]:
    events = extract_source_events(source)
    beat = {
        "beat_id": "B1",
        "segment": 1,
        "description": events[0]["text"],
        "source_event_ids": [events[0]["event_id"]],
        "dialogue_ids": [],
        "state_after": events[0]["text"],
    }
    draft = {
        "segment": 1,
        "title": "Courtyard actions",
        "opening_state": "Character A stands near the courtyard marker.",
        "coverage": "A clear view of the marked stone surfaces.",
        "pacing": "Real-time action.",
        "shots": [{
            "shot": 1,
            "event_indices": [1],
            "start_seconds": 0.0,
            "end_seconds": 12.0,
            "transition": "opening composition",
            "framing": "Wide view of the courtyard.",
            "camera": "Hold a clear, steady view.",
            "action": visible_action,
            "sound_effects": "",
        }],
        "closing_state": "The courtyard remains visible.",
    }
    canonical = _canonicalize_segment_contract(
        draft,
        segment_number=1,
        duration=12.0,
        assigned_beats=[beat],
        dialogue_catalog=[],
        opening_state=draft["opening_state"],
        source_intent={},
        source_events=events,
    )
    assert canonical is not None
    return segment_violations(
        source,
        canonical,
        segment_number=1,
        duration=12.0,
        assigned_beats=[beat],
        dialogue_catalog=[],
    )


class ActionIdentityTests(unittest.TestCase):
    def test_plain_uppercase_subject_and_registered_possessive_expand(self):
        pattern = _h3_source_cast_pattern(["Character A", "Character B"])
        self.assertEqual(
            expand_labeled_actor_shorthand("A breaks the stone marker.", pattern),
            "Character A breaks the stone marker.",
        )
        self.assertEqual(
            expand_labeled_actor_shorthand("A gets sideswept, smashing the pedestal.", pattern),
            "Character A gets sideswept, smashing the pedestal.",
        )
        self.assertEqual(
            expand_labeled_actor_shorthand("A's shoulder hits the ground.", pattern),
            "Character A's shoulder hits the ground.",
        )
        self.assertEqual(
            expand_labeled_actor_shorthand("A’s hands strike the shield.", pattern),
            "Character A’s hands strike the shield.",
        )

    def test_numeric_subject_alias_resolves_only_from_registered_canonical_label(self):
        pattern = _h3_source_cast_pattern(["Subject 1"])
        self.assertEqual(
            expand_labeled_actor_shorthand("1 moves the blue crate.", pattern),
            "Subject 1 moves the blue crate.",
        )
        self.assertEqual(
            expand_labeled_actor_shorthand("2 moves the blue crate.", pattern),
            "2 moves the blue crate.",
        )

    def test_articles_lowercase_unregistered_and_non_subject_tokens_stay_unchanged(self):
        pattern = _h3_source_cast_pattern(["Character A", "Character B"])
        cases = (
            "A stone statue breaks on the ledge.",
            "A very wide shot shows the courtyard.",
            "a breaks the stone marker.",
            "b hits the shield.",
            "C breaks the stone marker.",
            "The marker near A breaks.",
            "A smashing the stone marker.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(expand_labeled_actor_shorthand(text, pattern), text)

    def test_ambiguous_canonical_alias_is_left_unchanged(self):
        pattern = _h3_source_cast_pattern(["Character A", "Subject A"])
        for text in ("A breaks the marker.", "A's shoulder hits the ground."):
            with self.subTest(text=text):
                self.assertEqual(expand_labeled_actor_shorthand(text, pattern), text)

    def test_actor_a_and_actor_b_remain_distinct_in_action_frames(self):
        pattern = _h3_source_cast_pattern(["Character A", "Character B"])
        action_a = expand_labeled_actor_shorthand("A breaks the stone wall.", pattern)
        action_b = expand_labeled_actor_shorthand("B breaks the stone wall.", pattern)
        frame_a = _h3_preview_action_frames(action_a, pattern)[0]
        frame_b = _h3_preview_action_frames(action_b, pattern)[0]
        self.assertEqual(frame_a[2], frozenset({"character a"}))
        self.assertEqual(frame_b[2], frozenset({"character b"}))
        self.assertFalse(_h3_preview_action_matches(frame_a, frame_b))

    def test_manner_adverb_continuation_inherits_the_unique_prior_actor(self):
        pattern = _h3_source_cast_pattern(["Character A", "Character B"])
        frames = _h3_preview_action_frames(
            "Character A breaks the stone marker, then slowly draws back his right fist.",
            pattern,
        )
        draw = next(frame for frame in frames if frame[0] == "draw")
        self.assertEqual(draw[2], frozenset({"character a"}))

    def test_explicit_different_actor_after_adverb_overrides_prior_actor(self):
        pattern = _h3_source_cast_pattern(["Character A", "Character B"])
        frames = _h3_preview_action_frames(
            "Character A breaks the stone marker, then slowly Character B draws back his right fist.",
            pattern,
        )
        draw = next(frame for frame in frames if frame[0] == "draw")
        self.assertEqual(draw[2], frozenset({"character b"}))
        self.assertFalse(draw[2] & frozenset({"character a"}))

    def test_multi_person_lead_and_plural_pronoun_remain_ambiguous(self):
        pattern = _h3_source_cast_pattern(["Character A", "Character B"])
        unowned = _h3_preview_action_frames(
            "Character A and Character B strike the training pad, then slowly draw back their fists.",
            pattern,
        )
        draw = next(frame for frame in unowned if frame[0] == "draw")
        self.assertEqual(draw[2], frozenset())

        plural = _h3_preview_action_frames(
            "Character A and Character B strike the training pad, then slowly they draw back their fists.",
            pattern,
        )
        pronoun_draw = next(frame for frame in plural if frame[0] == "draw")
        self.assertEqual(pronoun_draw[2], frozenset({"they"}))

    def test_negated_and_attempted_actions_do_not_become_completed_frames(self):
        pattern = _h3_source_cast_pattern(["Character A"])
        negated = _h3_preview_action_frames(
            "Character A holds the blue marker but does not break it.", pattern
        )
        attempted = _h3_preview_action_frames(
            "Character A tries to shatter the blue marker.", pattern
        )
        self.assertFalse(any(frame[0] == "break" for frame in negated), negated)
        self.assertFalse(any(frame[0] == "shatter" for frame in attempted), attempted)

    def test_owned_earlier_pedestal_impact_is_not_mistaken_for_future_preview(self):
        source = (
            "[0s-6s] Character A smashes the stone statue pedestal. "
            "[6s-12s] A gets sideswiped across the courtyard, smashing another "
            "stone statue pedestal."
        )
        errors = _preview_violations(
            source, "Character A smashes the stone statue pedestal."
        )
        self.assertFalse(any("previews later" in item for item in errors), errors)

    def test_unassigned_future_wall_destruction_remains_a_preview(self):
        source = (
            "[0s-6s] Character A crosses the courtyard. "
            "[6s-12s] Character A smashes the stone wall."
        )
        errors = _preview_violations(
            source, "Character A smashes the stone wall."
        )
        self.assertTrue(any("previews later source event" in item for item in errors), errors)

    def test_adverb_actor_repair_does_not_hide_a_real_later_preview(self):
        source = (
            "[0s-6s] Character A breaks the blue marker, then slowly draws back "
            "his right fist. [6s-12s] Character A shatters the stone gate."
        )
        errors = _preview_violations(
            source,
            "Character A breaks the blue marker; Character A slowly draws back "
            "his right fist; Character A shatters the stone gate.",
        )
        self.assertTrue(
            any("previews later source event e2" in item.casefold() for item in errors),
            errors,
        )

    def test_wrong_actor_cannot_perform_a_reserved_action_early(self):
        source = (
            "[0s-6s] Character A holds the red bowl. "
            "[6s-12s] Character B breaks the red bowl."
        )
        errors = _preview_violations(source, "A breaks the red bowl.")
        self.assertTrue(any("previews later source event" in item for item in errors), errors)


if __name__ == "__main__":
    unittest.main()
