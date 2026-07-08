"""
Unit tests for the four polish-layer bug fixes shipped after the safety
guardrail work:

1. _dedupe_descriptors must consume leading article ("the he" / "a she"
   sequences must NOT appear in output).
2. _rewrite_long_possessives must handle bare possessives (no leading
   "the") AND 2-word noun phrases ("lower stomach", "white sweater").
3. _harvest_speaker_names must find screenplay-invented names like
   "Blaine" from speech-tag and vocative patterns, with stop-list
   filtering of common capitalized English words.
4. _collapse_article_pronoun safety net catches stragglers from any
   substitution chain.

Each fix targets a real symptom from production output. The string
fixtures here are minimal reproductions — kept intentionally short
so any future refactor that breaks the fix triggers a clean
regression alarm.

Run standalone:
    python tests/test_prompt_polish_fixes.py

Run via pytest:
    pytest tests/test_prompt_polish_fixes.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.prompt_polish import (  # noqa: E402
    _collapse_article_pronoun,
    _dedupe_descriptors,
    _harvest_speaker_names,
    _rewrite_long_possessives,
)


class TestDedupeArticleAware(unittest.TestCase):
    """Bug 4: 'the he' mangle. The dedupe regex must consume the leading
    article when present so the resulting pronoun stands alone."""

    def test_no_article_pronoun_after_dedupe(self):
        # Two occurrences of "the woman in white sweater" — second must
        # become "she" (or "her" after a preposition), not "the she".
        text = (
            "the woman in white sweater walks into the room. "
            "the woman in white sweater sits down."
        )
        cleaned, count = _dedupe_descriptors(
            text, {"Alice": "woman in white sweater"}
        )
        self.assertEqual(count, 1)
        self.assertNotIn("the she", cleaned.lower())
        self.assertNotIn("the he", cleaned.lower())
        self.assertIn("she sits down", cleaned)

    def test_object_pronoun_after_preposition(self):
        # When the substitution sits after a preposition, we still need
        # the article consumed AND the pronoun in object form.
        text = (
            "the man in cowboy hat reaches for the gun. "
            "she stands next to the man in cowboy hat."
        )
        cleaned, count = _dedupe_descriptors(
            text, {"Bob": "man in cowboy hat"}
        )
        self.assertEqual(count, 1)
        self.assertNotIn("the him", cleaned.lower())
        self.assertNotIn("the he", cleaned.lower())
        self.assertIn("next to him", cleaned)

    def test_no_article_present_works_unchanged(self):
        # When the descriptor in the source has no leading article, the
        # substitution should still work — pronoun stands alone.
        text = (
            "woman in white sweater walks into the room. "
            "woman in white sweater turns around."
        )
        cleaned, _count = _dedupe_descriptors(
            text, {"Alice": "woman in white sweater"}
        )
        self.assertNotIn("the she", cleaned.lower())
        self.assertIn("she turns around", cleaned)


class TestCollapseArticlePronoun(unittest.TestCase):
    """Bug 4 safety net: collapse any 'the/a/an + pronoun' that slipped
    through other code paths."""

    def test_basic_collapse(self):
        text = "the he reaches for the gun and a she runs away."
        cleaned, count = _collapse_article_pronoun(text)
        self.assertEqual(count, 2)
        self.assertNotIn("the he", cleaned)
        self.assertNotIn("a she", cleaned)
        self.assertIn("he reaches", cleaned)
        self.assertIn("she runs", cleaned)

    def test_no_op_on_clean_text(self):
        text = "the man reaches for the gun and the woman runs away."
        cleaned, count = _collapse_article_pronoun(text)
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, text)

    def test_object_pronouns_also_collapse(self):
        text = "she reaches for the him and gives a her the keys."
        cleaned, count = _collapse_article_pronoun(text)
        self.assertEqual(count, 2)
        self.assertNotIn("the him", cleaned)
        self.assertNotIn("a her", cleaned)


class TestLongPossessiveRewrite(unittest.TestCase):
    """Bug 1: D-rewrite must handle bare possessives + multi-word nouns."""

    def test_bare_possessive_no_leading_the(self):
        # Original incident: "woman in white with huge breasts's lower stomach"
        # — no leading "the". Old regex required "the" and missed this.
        text = "woman in white with huge breasts's lower stomach quivers."
        cleaned, count = _rewrite_long_possessives(text)
        self.assertEqual(count, 1)
        self.assertIn(
            "the lower stomach of the woman in white with huge breasts",
            cleaned,
        )
        # The original possessive form must be gone.
        self.assertNotIn("breasts's", cleaned)

    def test_with_leading_the_still_works(self):
        # Backward-compat — old behavior must continue.
        text = "the woman in middle with massive breasts's chest tightens."
        cleaned, count = _rewrite_long_possessives(text)
        self.assertEqual(count, 1)
        self.assertIn("the chest of the", cleaned)
        self.assertNotIn("breasts's", cleaned)

    def test_two_word_noun_phrase(self):
        # "lower stomach" is a legitimate 2-word body-part phrase. The
        # rewrite must capture both words, not leave "stomach" dangling.
        text = "woman in red dress with tied hair's right hand reaches."
        cleaned, count = _rewrite_long_possessives(text)
        self.assertEqual(count, 1)
        # The 2-word noun phrase should be preserved AND "reaches" must
        # stay as the next clause's verb.
        self.assertIn("the right hand of the woman in red dress with tied hair", cleaned)
        self.assertIn("reaches", cleaned)
        # No dangling word
        self.assertNotIn("hand of the woman in red dress with tied hair right", cleaned)

    def test_verb_after_single_word_noun_not_captured(self):
        # When the trailing word is a verb in the blocklist, it must stay
        # as a verb, not get folded into the noun phrase.
        text = "the woman in white silk gown's hand reaches forward."
        cleaned, count = _rewrite_long_possessives(text)
        self.assertEqual(count, 1)
        # "reaches" is a verb — must remain as a verb after the rewrite,
        # NOT folded into "the hand reaches of the X".
        self.assertIn("the hand of the woman in white silk gown reaches", cleaned)
        self.assertNotIn("hand reaches of the", cleaned)

    def test_short_descriptor_not_rewritten(self):
        # The 4-word minimum still applies — short possessives like
        # "the man's hand" must not be rewritten.
        text = "the man's hand grips the rope."
        cleaned, count = _rewrite_long_possessives(text)
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, text)


class TestHarvestSpeakerNames(unittest.TestCase):
    """Bug 3: harvest names from speech tags and vocatives."""

    def test_harvest_from_speech_tag(self):
        text = "Blaine reaches for the gun. Blaine says, 'I'm leaving.'"
        names = _harvest_speaker_names(text)
        self.assertIn("Blaine", names)

    def test_harvest_from_vocative_in_quoted_dialogue(self):
        text = "She turns to him and whispers 'Don't go, Blaine, please.'"
        names = _harvest_speaker_names(text)
        self.assertIn("Blaine", names)

    def test_stoplist_filters_common_words(self):
        # "The man" / "She turns" — neither "The" nor "She" should be
        # harvested because they're sentence-start English words on the
        # stop list. This test indirectly verifies the stop list is
        # active by checking that no junk names slip through.
        text = (
            "The woman walks in. She says hello. He replies 'Hey.' "
            "Today she is happy."
        )
        names = _harvest_speaker_names(text)
        # None of these are real character names — the harvest must
        # filter them all out via the stop list.
        for stopword in ("The", "She", "He", "Today"):
            self.assertNotIn(stopword, names,
                             f"stop-listed word '{stopword}' should not be harvested")

    def test_empty_input(self):
        self.assertEqual(_harvest_speaker_names(""), set())
        self.assertEqual(_harvest_speaker_names(None), set())  # type: ignore[arg-type]

    def test_multiple_distinct_names(self):
        text = (
            "Blaine reaches for the gun. Maria whispers something. "
            "Blaine says 'Wait, Maria, listen.'"
        )
        names = _harvest_speaker_names(text)
        self.assertIn("Blaine", names)
        self.assertIn("Maria", names)


class TestEndToEndDedupePlusCollapse(unittest.TestCase):
    """The dedupe function calls _collapse_article_pronoun internally as a
    safety net. Verify the integration: even pathological inputs come
    out clean."""

    def test_chained_substitution_safety(self):
        # Simulate a worst-case input: descriptor appears 3 times with
        # "the" in front each time. After dedupe, none of the 2nd+
        # occurrences should leave "the X" pronouns behind.
        text = (
            "the man in cowboy hat enters the saloon. "
            "the man in cowboy hat orders a whiskey. "
            "the man in cowboy hat tips his hat."
        )
        cleaned, count = _dedupe_descriptors(
            text, {"Cowboy": "man in cowboy hat"}
        )
        self.assertEqual(count, 2)
        # All instances of "the he" must be gone.
        self.assertNotIn("the he", cleaned.lower())
        self.assertIn("he orders a whiskey", cleaned)
        self.assertIn("he tips his hat", cleaned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
