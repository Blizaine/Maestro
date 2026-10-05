import unittest

from services.h3_contact_identity import omitted_h3_contact_instruments


class H3ContactIdentityTests(unittest.TestCase):
    def test_compound_elbow_smash_is_not_covered_by_shoulder_driving(self):
        source = (
            "Nora drops shoulder-first into a close-range explosive elbow smash on Eli."
        )
        visible = "Nora drops her weight and drives her shoulder into Eli."

        self.assertEqual(
            omitted_h3_contact_instruments(source, visible), ["elbow smash"]
        )

    def test_strike_with_elbow_is_not_covered_by_shoulder_paraphrase(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora strikes Eli with her elbow.",
                "Nora drives her shoulder into Eli.",
            ),
            ["with her elbow"],
        )

    def test_elbow_lexical_paraphrase_covers_the_same_instrument(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora strikes Eli with her elbow.",
                "Nora's elbow drives into Eli.",
            ),
            [],
        )

    def test_explicit_side_is_preserved_for_a_linked_instrument(self):
        source = "Nora strikes Eli with her left elbow."
        self.assertEqual(
            omitted_h3_contact_instruments(
                source, "Nora's right elbow drives into Eli."
            ),
            ["with her left elbow"],
        )
        self.assertEqual(
            omitted_h3_contact_instruments(
                source, "Nora's left elbow drives into Eli."
            ),
            [],
        )

    def test_unlinked_side_description_does_not_add_a_side_requirement(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora keeps her left elbow raised, then strikes Eli with her elbow.",
                "Nora's right elbow drives into Eli.",
            ),
            [],
        )

    def test_shoulder_first_and_incidental_elbows_are_not_contact_requirements(self):
        source = (
            "Nora drops shoulder-first to the ground while keeping her folded elbows tucked."
        )
        self.assertEqual(omitted_h3_contact_instruments(source, "Nora lands beside Eli."), [])
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora lands beside Eli with her elbow tucked.", "Nora lands beside Eli."
            ),
            [],
        )

    def test_a_different_contact_part_does_not_substitute_for_elbow(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora strikes Eli with her elbow.",
                "Nora drives her forearm into Eli.",
            ),
            ["with her elbow"],
        )

    def test_both_named_instruments_are_required(self):
        source = "Nora lands an elbow smash, then follows with a knee strike at Eli."
        self.assertEqual(
            omitted_h3_contact_instruments(
                source, "Nora lands an elbow smash on Eli."
            ),
            ["knee strike"],
        )

    def test_missed_or_near_missed_attempt_keeps_its_instrument(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora's elbow strike misses Eli.",
                "Nora drives her shoulder into Eli.",
            ),
            ["elbow strike"],
        )
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora almost elbowed Eli.", "Nora almost elbows Eli."
            ),
            [],
        )

    def test_direct_with_instrument_form_is_covered_by_same_instrument(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora drives her shoulder into Eli.",
                "Nora shoulder-checks Eli.",
            ),
            [],
        )

    def test_instrument_as_subject_can_contact_a_direct_target(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora strikes Eli with her elbow.", "Nora's elbow strikes Eli."
            ),
            [],
        )
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora's elbow smashes Eli.", "Nora's elbow smashes Eli."
            ),
            [],
        )
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Her fist hits Eli.", "Her fist hits Eli."
            ),
            [],
        )

    def test_plural_elbows_keep_the_same_contact_identity(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora's elbows smashed Eli.", "Nora's elbows smashed Eli."
            ),
            [],
        )

    def test_negated_source_does_not_create_a_contact_requirement(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora does not elbow Eli.", "Nora moves beside Eli."
            ),
            [],
        )
        self.assertEqual(
            omitted_h3_contact_instruments(
                "No elbow strike occurs.", "Nora moves beside Eli."
            ),
            [],
        )

    def test_negated_visible_contact_cannot_cover_a_required_instrument(self):
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora strikes Eli with her elbow.", "Nora does not elbow Eli."
            ),
            ["with her elbow"],
        )
        self.assertEqual(
            omitted_h3_contact_instruments(
                "Nora strikes Eli with her elbow.", "No elbow strike occurs."
            ),
            ["with her elbow"],
        )


if __name__ == "__main__":
    unittest.main()
