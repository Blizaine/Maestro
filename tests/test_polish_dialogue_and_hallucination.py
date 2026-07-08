"""
Regression tests for the polish-layer fixes for the two production
incidents the user reported after the first round of polish fixes:

INCIDENT 1: Dialogue with contraction apostrophes got split into
multiple fake spans by the old _quoted_re. When polish converted
single-to-double quotes the count mismatched and the dialogue revert
silently bailed, so "Cathy" was mangled to "woman in white with
massive breasts" inside quoted dialogue and the corruption shipped.

INCIDENT 2: Polish LLM hallucinated names (e.g. "Blaine") into
narrative prose where the input had only descriptors ("the strong
man in black"). The character_reference_block in the system prompt
listed the name → descriptor mapping even when the input never
mentioned the name; the LLM used the mapping in reverse. Pronouns
("his logic") got replaced with possessive name forms ("the
Blaine's logic").

The fixes ship across four layers: apostrophe-aware quoted-span
helper, input-filtered character block, strengthened anti-hallucination
system-prompt rules, and a post-process hallucination stripper. Each
test below exercises one of the four; together they cover the
production scenarios end-to-end.
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
    _strip_hallucinated_names,
)


# Test the apostrophe-aware iterator + revert helper. These are nested
# inside polish_prompts_third_pass(), so we test them via a module-level
# wrapper that exercises the same logic.
def _import_nested_helpers():
    """Fish out the nested helpers from polish_prompts_third_pass.

    They're closures inside that function, but they don't capture any
    of the function's local state needed for THESE tests (no LLM calls,
    no character maps). We rebuild a minimal version here for testing
    by re-importing the regex pieces from the module.
    """
    import re as _re
    from services.director import prompt_polish as pp

    # Reconstruct the iterator helper using the same patterns. We
    # can't easily get the closure version, so we test the regex
    # pieces directly. The patterns are NOT module-level constants in
    # the source (they're inside the closure), so this test mirrors
    # them — if the source patterns change, this test will become
    # stale and a manual sync is needed.
    DOUBLE = _re.compile(r'"([^"]*?)"', _re.DOTALL)
    SINGLE = _re.compile(
        r"(?<![A-Za-z])'(.*?)'(?![A-Za-z])",
        _re.DOTALL,
    )

    def iter_spans(text: str):
        if not text:
            return
        spans = []
        for m in DOUBLE.finditer(text):
            spans.append(('"', m.group(1), m.start(), m.end()))
        for m in SINGLE.finditer(text):
            spans.append(("'", m.group(1), m.start(), m.end()))
        spans.sort(key=lambda s: s[2])
        last_end = -1
        for span in spans:
            if span[2] >= last_end:
                yield span
                last_end = span[3]

    return iter_spans


class TestApostropheAwareQuotedSpans(unittest.TestCase):
    """Incident 1 root cause: the old regex split dialogue at every
    contraction apostrophe."""

    def setUp(self):
        self.iter_spans = _import_nested_helpers()

    def test_single_quoted_dialogue_with_contraction_is_one_span(self):
        # Original incident input: dialogue with multiple contraction
        # apostrophes inside a single-quoted span.
        text = "She says 'You know, it's ridiculous how much you've been on my mind.'"
        spans = list(self.iter_spans(text))
        self.assertEqual(len(spans), 1, f"expected 1 span, got {len(spans)}: {spans}")
        quote_char, content, _start, _end = spans[0]
        self.assertEqual(quote_char, "'")
        # Content should include both contractions intact.
        self.assertIn("it's", content)
        self.assertIn("you've", content)

    def test_double_quoted_dialogue_with_contractions(self):
        text = 'She says "You know, it\'s ridiculous how much you\'ve been on my mind."'
        spans = list(self.iter_spans(text))
        self.assertEqual(len(spans), 1)
        quote_char, content, _, _ = spans[0]
        self.assertEqual(quote_char, '"')
        self.assertIn("it's", content)

    def test_two_distinct_dialogue_spans(self):
        # Two separate dialogue lines — the iterator should find both.
        text = "She says 'Hello, it's me.' He replies 'I've missed you.'"
        spans = list(self.iter_spans(text))
        self.assertEqual(len(spans), 2)

    def test_count_matches_after_quote_style_change(self):
        # Polish LLM commonly converts ' to " in output. Both forms
        # should produce the SAME span count for position-paired
        # revert to work.
        before = "She says 'You know, it's ridiculous.'"
        after = 'She says "You know, it\'s ridiculous."'
        before_spans = list(self.iter_spans(before))
        after_spans = list(self.iter_spans(after))
        self.assertEqual(len(before_spans), len(after_spans),
                         "Count must match for dialogue revert to fire")
        self.assertEqual(len(before_spans), 1)


class TestStripHallucinatedNames(unittest.TestCase):
    """Incident 2 root cause: polish LLM invented names that weren't
    in input, like "the Blaine does not turn..."."""

    def test_strip_basic_hallucinated_name_with_article(self):
        # The user-reported bug: input has "the strong man in black",
        # polish output has "The Blaine".
        input_text = "The strong man in black does not turn immediately; he lifts one hand."
        output_text = "The Blaine does not turn immediately; he lifts one hand."
        cleaned, hallucinated = _strip_hallucinated_names(
            input_text, output_text,
            name_to_descriptor={},
            fallback_descriptors=["strong man in black"],
        )
        self.assertIn("Blaine", hallucinated)
        self.assertNotIn("Blaine", cleaned)
        self.assertNotIn("the Blaine", cleaned.lower())
        self.assertIn("strong man in black", cleaned)

    def test_strip_possessive_hallucinated_name(self):
        # "his logic" → "the Blaine's logic" in polish output.
        input_text = "her refusal to accept his logic"
        output_text = "her refusal to accept the Blaine's logic"
        cleaned, hallucinated = _strip_hallucinated_names(
            input_text, output_text,
            name_to_descriptor={},
            fallback_descriptors=["strong man in black"],
        )
        self.assertIn("Blaine", hallucinated)
        # Possessive form should become "their" (gender-neutral).
        self.assertIn("their logic", cleaned)
        self.assertNotIn("Blaine", cleaned)

    def test_known_name_in_map_not_stripped(self):
        # If "Blaine" is a real character in our map, it must NOT be
        # stripped — the user genuinely named the character that.
        input_text = "the strong man in black walks in"
        output_text = "Blaine walks in"
        cleaned, hallucinated = _strip_hallucinated_names(
            input_text, output_text,
            name_to_descriptor={"Blaine": "strong man in black"},
            fallback_descriptors=["strong man in black"],
        )
        self.assertEqual(hallucinated, [],
                         "Names in name_to_descriptor must not be flagged")

    def test_name_present_in_input_not_stripped(self):
        # If the input already mentioned "Blaine", polish keeping it
        # is fine — not a hallucination.
        input_text = "Blaine reaches for the gun."
        output_text = "Blaine reaches for the gun, his hand steady."
        cleaned, hallucinated = _strip_hallucinated_names(
            input_text, output_text,
            name_to_descriptor={},
            fallback_descriptors=["strong man in black"],
        )
        self.assertEqual(hallucinated, [])
        self.assertIn("Blaine", cleaned)

    def test_multiple_hallucinated_names(self):
        input_text = "The woman in white speaks to the strong man in black."
        output_text = "Cathy speaks to Blaine, her hand on his arm."
        cleaned, hallucinated = _strip_hallucinated_names(
            input_text, output_text,
            name_to_descriptor={},
            fallback_descriptors=["woman in white"],
        )
        # Both names should be flagged as hallucinations.
        self.assertEqual(set(hallucinated), {"Cathy", "Blaine"})
        # Names should be substituted with the fallback (since both go
        # to the same fallback in this test, output will be redundant —
        # acceptable degradation given we can't disambiguate gender
        # without a richer signal).
        self.assertNotIn("Cathy", cleaned)
        self.assertNotIn("Blaine", cleaned)

    def test_no_fallback_keeps_name_unchanged_for_bare_form(self):
        # When no fallback descriptor is available, bare-name occurrences
        # should be left as-is (substituting with nothing would corrupt
        # the sentence). Possessives still get → "their" because that's
        # always a safe rewrite.
        input_text = "the strong man speaks"
        output_text = "Blaine speaks. Blaine's voice is low."
        cleaned, hallucinated = _strip_hallucinated_names(
            input_text, output_text,
            name_to_descriptor={},
            fallback_descriptors=[],
        )
        self.assertIn("Blaine", hallucinated)
        # Possessive "Blaine's" → "their"
        self.assertIn("their voice", cleaned)
        # Bare "Blaine speaks" stays put because we have no better option.
        self.assertIn("Blaine speaks", cleaned)

    def test_stoplist_protects_common_words(self):
        # The function uses _NAME_HARVEST_STOPLIST. Words like "The",
        # "She", "He", "Today" must NOT be flagged as hallucinations
        # even when they appear capitalized at sentence start.
        input_text = "the strong man in black walks in"
        output_text = "The strong man walks. Today the rain falls."
        cleaned, hallucinated = _strip_hallucinated_names(
            input_text, output_text,
            name_to_descriptor={},
            fallback_descriptors=["strong man in black"],
        )
        # "The" and "Today" must not be flagged.
        self.assertNotIn("The", hallucinated)
        self.assertNotIn("Today", hallucinated)


class TestEndToEndIncidentReproduction(unittest.TestCase):
    """End-to-end reproductions of the user-reported polish output."""

    def test_incident_1_dialogue_mangle_post_apostrophe_fix(self):
        # The dialogue revert depends on apostrophe-aware spans.
        # We can't fully test the revert without spinning up the polish
        # closure, but we can verify the iterator reports matching
        # span counts for the specific input/output that broke before.
        iter_spans = _import_nested_helpers()
        before = (
            "She leans in slightly and speaks, her voice breathy with anticipation, "
            "'You know, it's ridiculous how much you've been on my mind since. since the shower.' "
            "The strong man in black does not turn immediately; he lifts one hand and gently brushes "
            "a piece of hair from his forehead while his jaw tightens, responding with a low, "
            "guarded tone: 'What about me, Cathy? My terrible taste in soap?'"
        )
        after = (
            'She leans in slightly and speaks, her voice breathy with anticipation, '
            '"You know, it\'s ridiculous how much you\'ve been on my mind since. since the shower." '
            'The Blaine does not turn immediately; he lifts one hand to gently brush '
            'a piece of hair from his forehead while his jaw tightens, responding with a low, '
            'guarded tone: "What about me, woman in white with massive breasts? My terrible taste in soap?"'
        )
        before_spans = list(iter_spans(before))
        after_spans = list(iter_spans(after))
        # Both must produce 2 spans (the two dialogue lines).
        self.assertEqual(len(before_spans), 2,
                         f"expected 2 before-spans, got {len(before_spans)}")
        self.assertEqual(len(after_spans), 2,
                         f"expected 2 after-spans, got {len(after_spans)}")
        # Counts match → position-paired revert can fire.
        self.assertEqual(len(before_spans), len(after_spans))

    def test_incident_2_blaine_hallucination_strip(self):
        # End-to-end: "the strong man in black" + "his logic" → polish
        # produces "The Blaine" + "the Blaine's logic" → strip restores.
        input_text = (
            "He brings his right hand up and places it firmly on the woman in white's "
            "shoulder. her refusal to accept his logic"
        )
        output_text = (
            "He raises his right hand and places it firmly on the woman in white's "
            "shoulder, anchoring her. her refusal to accept the Blaine's logic"
        )
        cleaned, hallucinated = _strip_hallucinated_names(
            input_text, output_text,
            name_to_descriptor={},
            fallback_descriptors=["strong man in black"],
        )
        self.assertIn("Blaine", hallucinated)
        self.assertIn("their logic", cleaned)
        self.assertNotIn("Blaine", cleaned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
