"""
Unit tests for app.services.director.safety_scan.

Run standalone:
    python tests/test_safety_scan.py

Or via pytest:
    pytest tests/test_safety_scan.py -v

"""
from __future__ import annotations

import os
import sys
import unittest

# Make the app/ folder importable when running standalone, without requiring
# the user to set PYTHONPATH manually.
_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.safety_scan import (  # noqa: E402
    SafetyViolationError,
    assert_no_minor_content,
    collect_pass2_text,
    screenplay_contains_minor_content,
)


class TestScreenplayContainsMinorContent(unittest.TestCase):
    """Co-occurrence semantics: NEITHER list alone fires; BOTH together fires."""

    def test_empty_text_returns_empty(self):
        self.assertEqual(screenplay_contains_minor_content(""), [])
        self.assertEqual(screenplay_contains_minor_content(None), [])  # type: ignore[arg-type]

    def test_minor_only_returns_empty(self):
        # Children's-content scenario — minor vocab present, no sexual vocab.
        # MUST NOT trigger; otherwise the scanner would block legitimate
        # family / kids' / school-set non-sexual screenplays.
        self.assertEqual(
            screenplay_contains_minor_content(
                "the child plays in the yard while her son rides a bike"
            ),
            [],
        )

    def test_sexual_only_returns_empty(self):
        # Adult-only NSFW scenario — sexual vocab present, no minor vocab.
        # MUST NOT trigger; this is the supported NSFW path.
        self.assertEqual(
            screenplay_contains_minor_content(
                "the woman undresses and straddles him, her breasts heavy"
            ),
            [],
        )

    def test_original_incident_phrase_triggers(self):
        # The exact phrase that surfaced in user-shared production output
        # and motivated this entire safety layer. If this assertion ever
        # goes back to passing-with-empty-list, the regression is real.
        matches = screenplay_contains_minor_content(
            "The pre-teen girl continues pumping"
        )
        self.assertTrue(matches, f"expected non-empty matches, got {matches!r}")
        self.assertIn("pre-teen", matches)
        self.assertIn("pumping", matches)

    def test_age_numeric_under_18_is_minor_evidence(self):
        matches = screenplay_contains_minor_content("a 16-year-old undresses")
        self.assertTrue(matches)
        self.assertTrue(any("16" in m for m in matches), matches)
        self.assertIn("undresses", matches)

    def test_age_numeric_variants_under_18_trigger_with_sexual_vocab(self):
        for age_text in (
            "age 16", "age: 16", "aged 16", "16 years old",
            "16-year-old", "16 yr old", "16 yo", "16 y/o", "16 y.o.",
        ):
            with self.subTest(age_text=age_text):
                matches = screenplay_contains_minor_content(
                    f"the person is {age_text} and undresses"
                )
                self.assertTrue(matches)
                self.assertTrue(any("16" in match for match in matches), matches)

    def test_age_numeric_without_sexual_vocab_returns_empty(self):
        self.assertEqual(
            screenplay_contains_minor_content("a 16-year-old kid rides a bike"),
            [],
        )

    def test_adult_age_no_co_occurrence_returns_empty(self):
        # Adult vocabulary with an over-18 age and no minor vocab. The age
        # regex is bounded to <=17, so this must NOT match.
        self.assertEqual(
            screenplay_contains_minor_content("young man, age 25, undresses"),
            [],
        )

    def test_adult_only_explicit_scene_returns_empty(self):
        # Long adult-only NSFW snippet — exercises the sexual-side regex
        # against many terms while minor vocab is absent.
        text = (
            "The woman pulls him close. She straddles him, grinding "
            "her hips. He moans as she rides him, her breasts pressed "
            "against his chest. She is fully nude."
        )
        self.assertEqual(screenplay_contains_minor_content(text), [])

    def test_familial_minor_role_triggers(self):
        # Familial roles (daughter / son / niece / nephew) are on the minor
        # list because in a sex scene they imply incest with a minor.
        matches = screenplay_contains_minor_content(
            "the daughter strips and straddles him"
        )
        self.assertTrue(matches)
        self.assertIn("daughter", matches)

    def test_canonical_minor_vocabulary_and_plurals_trigger(self):
        terms = (
            "pre-teen", "preteens", "tween", "teens", "teenager",
            "adolescents", "schoolgirl", "schoolboys", "junior",
            "elementary students", "middle schoolers", "high school students",
            "little girl", "young boys", "daughters", "sons",
            "stepdaughter", "stepsons", "niece", "nephews",
            "granddaughter", "grandsons", "under 18",
        )
        for term in terms:
            with self.subTest(term=term):
                matches = screenplay_contains_minor_content(f"the {term} is nude")
                self.assertTrue(matches, term)
                self.assertIn(term, matches)

    def test_plain_girl_and_boy_are_not_minor_terms(self):
        self.assertEqual(screenplay_contains_minor_content("the party girl is nude"), [])
        self.assertEqual(screenplay_contains_minor_content("the cowboy is nude"), [])

    def test_word_boundary_avoids_substring_false_positive(self):
        # "kindergarten" contains "kid" as a substring? Actually no — "kid"
        # is fully separate. But "teen" appears inside "canteen", "fifteen",
        # "between". Word boundaries in the regex must prevent those firing.
        text = (
            "they meet between the canteen and the fifteen lockers; "
            "the woman undresses and rides him."
        )
        # No real minor vocab here despite substring overlap — must not fire.
        self.assertEqual(screenplay_contains_minor_content(text), [])


class TestAssertNoMinorContent(unittest.TestCase):
    """Raise contract: clean text returns silently, dirty text raises with detail."""

    def test_clean_text_does_not_raise(self):
        # Should return None silently.
        self.assertIsNone(assert_no_minor_content("a clean adult scene", "test"))

    def test_dirty_text_raises_with_source_and_terms(self):
        with self.assertRaises(SafetyViolationError) as ctx:
            assert_no_minor_content(
                "The pre-teen girl continues pumping",
                source="screenplay (Pass 1)",
            )
        err = ctx.exception
        self.assertEqual(err.source, "screenplay (Pass 1)")
        self.assertIn("pre-teen", err.matched_terms)
        self.assertIn("pumping", err.matched_terms)
        # Stringified error should include the source so logs are traceable
        # to the right pipeline stage.
        self.assertIn("screenplay (Pass 1)", str(err))

    def test_empty_source_text_does_not_raise(self):
        self.assertIsNone(assert_no_minor_content("", "anywhere"))


class TestCollectPass2Text(unittest.TestCase):
    """The Pass-2 helper concatenates every text-bearing field into one blob."""

    def test_empty_input_returns_empty_string(self):
        self.assertEqual(collect_pass2_text([]), "")
        self.assertEqual(collect_pass2_text(None), "")  # type: ignore[arg-type]

    def test_collects_top_level_text_fields(self):
        shots = [{
            "title": "TITLE_TEXT",
            "image_prompt": "IMAGE_PROMPT_TEXT",
            "video_prompt": "VIDEO_PROMPT_TEXT",
        }]
        blob = collect_pass2_text(shots)
        self.assertIn("TITLE_TEXT", blob)
        self.assertIn("IMAGE_PROMPT_TEXT", blob)
        self.assertIn("VIDEO_PROMPT_TEXT", blob)

    def test_collects_array_fields(self):
        shots = [{
            "action_beats": ["BEAT_ONE", "BEAT_TWO"],
            "keyframe_prompts": ["KEYFRAME_ONE"],
        }]
        blob = collect_pass2_text(shots)
        self.assertIn("BEAT_ONE", blob)
        self.assertIn("BEAT_TWO", blob)
        self.assertIn("KEYFRAME_ONE", blob)

    def test_collects_nested_subject_and_dialogue(self):
        shots = [{
            "subjects_on_screen": [
                {"visual_description": "SUBJECT_DESC", "speaker_name": "ALICE"},
            ],
            "dialogue_beats": [
                {"spoken_text": "SPOKEN_LINE"},
            ],
        }]
        blob = collect_pass2_text(shots)
        self.assertIn("SUBJECT_DESC", blob)
        self.assertIn("ALICE", blob)
        self.assertIn("SPOKEN_LINE", blob)

    def test_collects_strings_from_arbitrarily_nested_schema_fields(self):
        shots = [{
            "spatial_setup": {"foreground": "FOREGROUND_TEXT"},
            "performance_beats": [
                {"action": "PERFORMANCE_TEXT", "constraints": ["CONSTRAINT_TEXT"]},
            ],
            "camera_plan": {"movement": "CAMERA_MOVEMENT_TEXT"},
            "audio_plan": {"effects": ["AUDIO_EFFECT_TEXT"]},
            "dialogue_beats": [{
                "spoken_text": "SPOKEN_TEXT",
                "delivery": "DELIVERY_TEXT",
                "physical_cue": "PHYSICAL_CUE_TEXT",
            }],
        }]
        blob = collect_pass2_text(shots)
        for expected in (
            "FOREGROUND_TEXT", "PERFORMANCE_TEXT", "CONSTRAINT_TEXT",
            "CAMERA_MOVEMENT_TEXT", "AUDIO_EFFECT_TEXT", "SPOKEN_TEXT",
            "DELIVERY_TEXT", "PHYSICAL_CUE_TEXT",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, blob)

    def test_pass2_blob_passes_through_safety_scan(self):
        # End-to-end: a Pass-2 shot list with minor + sexual vocab inside
        # an inner field MUST be caught by assert_no_minor_content when
        # piped through collect_pass2_text.
        dirty_shots = [{
            "title": "Scene 1",
            "image_prompt": "a clean wide establishing shot",
            "video_prompt": "the pre-teen girl continues pumping",
        }]
        blob = collect_pass2_text(dirty_shots)
        with self.assertRaises(SafetyViolationError):
            assert_no_minor_content(blob, source="shot list (Pass 2)")

    def test_pass2_blob_clean_when_all_adult(self):
        clean_shots = [{
            "title": "Scene 1",
            "image_prompt": "a woman in a red dress walks into the bar",
            "video_prompt": "she undresses and straddles him on the couch",
            "subjects_on_screen": [{"visual_description": "adult woman, 32"}],
        }]
        blob = collect_pass2_text(clean_shots)
        # Should NOT raise.
        assert_no_minor_content(blob, source="shot list (Pass 2)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
