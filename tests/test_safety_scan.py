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

    def test_age_numeric_under_18_triggers_with_minor_vocab(self):
        # The numeric-age regex is a sexual-side trigger. Co-occurring with
        # minor vocab, it should fire even when no other sexual term is
        # present in the snippet.
        matches = screenplay_contains_minor_content(
            "a 16-year-old kid kisses passionately and undresses"
        )
        self.assertTrue(matches)
        # "undresses" -> "stripped/undressing"-family is NOT in the list, but
        # "16-year-old" + "kid" should be enough on the minor-vocab side
        # plus the numeric age supplies the sexual-side trigger.
        self.assertTrue(any("16" in m for m in matches), matches)

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
