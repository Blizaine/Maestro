import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_source_grammar import (
    annotate_final_state_events,
    classify_end_with_scope,
    is_temporal_context_clause,
    mask_prenominal_participial_properties,
    mask_modal_purpose,
)


class H3SourceGrammarTests(unittest.TestCase):
    def test_masks_coordinated_prenominal_participles_without_shifting_offsets(self):
        source = (
            "Ada and Len stand stationary in front of the solid, closed, and locked "
            "dark wood archive door.\r\nAda checks the table and opens a nearby cabinet."
        )

        masked = mask_prenominal_participial_properties(source)

        self.assertEqual(len(masked), len(source))
        self.assertEqual(masked.count("\r\n"), source.count("\r\n"))
        self.assertNotIn("closed", masked.casefold())
        self.assertNotIn("locked", masked.casefold())
        self.assertIn("solid", masked)
        self.assertIn("dark wood archive door", masked)
        self.assertIn("stand stationary", masked)
        self.assertIn("opens a nearby cabinet", masked)

    def test_masks_only_the_prenominal_property_series_and_keeps_other_objects(self):
        source = (
            "The red, closed, and locked archive door stands beside the open glass case; "
            "Ada checks the case and pulls the door closed."
        )

        masked = mask_prenominal_participial_properties(source)

        self.assertIn("archive door stands beside the open glass case", masked)
        self.assertIn("Ada checks the case and pulls the door closed.", masked)
        self.assertEqual(masked.count("archive door"), source.count("archive door"))

    def test_preserves_finite_resultative_and_passive_closure_actions(self):
        for source in (
            "Ada closed and locked the door.",
            "Ada pushes the door closed and pulls it open.",
            "The door was closed and locked by Ada.",
            "The door, closed and locked by Ada, remains beside the table.",
            "Ada closes the door, and Len locks it afterward.",
        ):
            with self.subTest(source=source):
                self.assertEqual(mask_prenominal_participial_properties(source), source)

    def test_does_not_mask_quoted_property_text_or_unrelated_actions(self):
        source = (
            "The note says ‘the solid, closed, and locked archive door’. "
            "Then Ada unlocks the nearby cabinet."
        )
        self.assertEqual(mask_prenominal_participial_properties(source), source)


    def test_masks_modal_purpose_but_preserves_later_action(self):
        source = (
            "Mara carries the spool so she can hang the banner. "
            "Then she closes the workshop door."
        )
        masked = mask_modal_purpose(source)
        self.assertEqual(len(masked), len(source))
        self.assertEqual(masked.count("\n"), source.count("\n"))
        self.assertIn("Mara carries the spool", masked)
        self.assertNotIn("she can hang the banner", masked)
        self.assertIn("Then she closes the workshop door.", masked)

    def test_masks_only_through_comma_before_then_clause(self):
        source = "They move the spool so they could reach the hook, then Priya ties it."
        masked = mask_modal_purpose(source)
        self.assertNotIn("so they could reach the hook", masked)
        self.assertIn(", then Priya ties it.", masked)

    def test_modal_purpose_keeps_its_own_comma_list(self):
        source = (
            "They carry the spool so they can retrieve it, lift it, "
            "and hang the banner."
        )
        masked = mask_modal_purpose(source)
        self.assertNotIn("so they can retrieve it", masked)
        self.assertNotIn("lift it", masked)
        self.assertNotIn("hang the banner", masked)
        self.assertIn("They carry the spool", masked)

    def test_unknown_finite_action_after_comma_is_not_masked(self):
        source = "Mara carries the spool so she can hang the banner, Priya whistles."
        masked = mask_modal_purpose(source)
        self.assertNotIn("so she can hang the banner", masked)
        self.assertIn(", Priya whistles.", masked)
        lower_pronoun = "Mara carries the spool so she can hang it, she whistles."
        self.assertIn(", she whistles.", mask_modal_purpose(lower_pronoun))

    def test_quoted_purpose_is_not_masked_and_contractions_survive(self):
        source = "Mara says, ‘so she can hang the banner’; I'll keep the door open."
        self.assertEqual(mask_modal_purpose(source), source)

    def test_no_modal_purpose_is_unchanged(self):
        for source in (
            "Mara carries the spool and hangs the banner.",
            "The camera follows Priya; then she closes the door.",
            "So the story can continue, Priya opens the door.",
        ):
            self.assertEqual(mask_modal_purpose(source), source)

    def test_stative_end_with_scope_is_classified(self):
        source = (
            "End with Priya holding the spool's loose end, "
            "the banner hanging securely, and the workshop door closed."
        )
        self.assertEqual(classify_end_with_scope(source), "final_state")

    def test_eventive_and_mixed_endings_are_not_stative(self):
        self.assertEqual(
            classify_end_with_scope("End with Priya closing the workshop door."),
            "eventive",
        )
        self.assertEqual(
            classify_end_with_scope("End with Priya holding the spool, then closing the door."),
            "eventive",
        )
        self.assertEqual(
            classify_end_with_scope("End with Priya holding the spool and smiling."),
            "eventive",
        )
        self.assertIsNone(classify_end_with_scope("End with a quiet, hopeful image."))

    def test_unrecognized_action_cannot_be_hidden_beside_a_state(self):
        for action in ("shooting Ron", "singing", "breaking the window"):
            source = f"End with Mara holding the spool and {action}."
            self.assertNotEqual(classify_end_with_scope(source), "final_state", source)
            events = [
                {"event_id": "E1", "text": "End with Mara holding the spool"},
                {"event_id": "E2", "text": action},
            ]
            tagged = annotate_final_state_events(events, source_text=source)
            self.assertTrue(all("requirement_kind" not in event for event in tagged), source)

    def test_only_events_wholly_inside_the_exact_end_sentence_are_tagged(self):
        source = (
            "Priya picks up the spool. End with Priya holding the spool, "
            "the banner hanging securely, and the workshop door closed. "
            "The lights fade."
        )
        events = [
            {"event_id": "E1", "text": "Priya picks up the spool"},
            {"event_id": "E2", "text": "End with Priya holding the spool"},
            {"event_id": "E3", "text": "the banner hanging securely"},
            {"event_id": "E4", "text": "the workshop door closed"},
            {"event_id": "E5", "text": "The lights fade"},
        ]
        original = [dict(event) for event in events]
        tagged = annotate_final_state_events(events, source_text=source)
        self.assertEqual([event["event_id"] for event in tagged], ["E1", "E2", "E3", "E4", "E5"])
        self.assertEqual([event["text"] for event in tagged], [event["text"] for event in events])
        self.assertEqual(events, original)
        self.assertNotIn("requirement_kind", tagged[0])
        self.assertEqual([event.get("requirement_kind") for event in tagged[1:4]], ["final_state"] * 3)
        self.assertNotIn("requirement_kind", tagged[4])

    def test_eventive_end_event_is_retained_without_a_state_tag(self):
        source = "End with Priya closing the workshop door."
        events = [{"event_id": "E7", "text": source}]
        tagged = annotate_final_state_events(events, source_text=source)
        self.assertEqual(tagged, events)

    def test_scope_does_not_cross_sentence_or_quoted_example(self):
        source = '“End with Priya holding the spool.” Then Mara leaves. End with the door closed.'
        self.assertIsNone(classify_end_with_scope('“End with Priya holding the spool.”'))
        events = [
            {"event_id": "E1", "text": "Then Mara leaves"},
            {"event_id": "E2", "text": "End with the door closed"},
        ]
        tagged = annotate_final_state_events(events, source_text=source)
        self.assertNotIn("requirement_kind", tagged[0])
        self.assertEqual(tagged[1].get("requirement_kind"), "final_state")

    def test_quoted_state_instruction_is_never_annotated(self):
        source = 'The note reads “End with the workshop door closed.”'
        events = [{"event_id": "E1", "text": "End with the workshop door closed"}]
        self.assertNotIn(
            "requirement_kind",
            annotate_final_state_events(events, source_text=source)[0],
        )

    def test_duplicate_or_unlocatable_event_text_is_not_tagged(self):
        source = "End with the door closed. Then the door closed in another room."
        events = [{"event_id": "E1", "text": "the door closed"}]
        self.assertNotIn(
            "requirement_kind",
            annotate_final_state_events(events, source_text=source)[0],
        )
        self.assertNotIn(
            "requirement_kind",
            annotate_final_state_events(
                [{"event_id": "E2", "text": "the door remains shut"}],
                source_text="End with the door closed.",
            )[0],
        )

    def test_temporal_context_requires_a_pure_occasion_phrase(self):
        for phrase in ("on the last night", "By the following morning", "each evening"):
            self.assertTrue(is_temporal_context_clause(phrase), phrase)
        for clause in (
            "on the last night Nora leaves a bouquet",
            "each evening, Priya returns",
            "on the final night, then she closes the door",
            "the last night after the door closes",
        ):
            self.assertFalse(is_temporal_context_clause(clause), clause)


if __name__ == "__main__":
    unittest.main()
