"""Fish-only syntax normalization stays narrow and preserves source text."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.fish_script_format import FishScriptFormat


class FishScriptFormatTests(unittest.TestCase):
    def test_normalizes_only_clear_one_off_delivery_cues(self):
        formatter = FishScriptFormat()

        result, changed = formatter.normalize_fragment(
            'Mara: <screams, abruptly> Wait! <whispers, briefly> Stay close.'
        )

        self.assertEqual(
            result,
            'Mara: [screams, abruptly] Wait! [whispers, briefly] Stay close.',
        )
        self.assertEqual(
            changed,
            ("<screams, abruptly>", "<whispers, briefly>"),
        )

    def test_preserves_supported_tags_phonemes_and_unknown_markup(self):
        formatter = FishScriptFormat()
        script = (
            "<whisper>keep quiet</whisper> <emphasis>this word</emphasis> "
            "<|phoneme_start|>read<|phoneme_end|> "
            "<custom>markup</custom> <screams data=raw>"
        )

        result, changed = formatter.normalize_fragment(script)

        self.assertEqual(result, script)
        self.assertEqual(changed, ())

    def test_preserves_exact_source_tokens_and_user_cue_prefixes(self):
        formatter = FishScriptFormat(
            source_text="Keep <laughs, softly> verbatim and preserve [my voice steadily]."
        )

        normalized, changed = formatter.normalize_fragment(
            "<laughs, softly> [my voice steadily"
        )
        previous, addition, recovered = formatter.recover_continuation(
            "Mara: [my voice steadily",
            "<forceful>Wait!</forceful> We move.",
        )

        self.assertEqual(normalized, "<laughs, softly> [my voice steadily")
        self.assertEqual(changed, ())
        self.assertEqual(previous, "Mara: [my voice steadily")
        self.assertEqual(addition, "<forceful>Wait!</forceful> We move.")
        self.assertFalse(recovered)

    def test_closes_inherited_cue_at_a_fresh_balanced_tag_boundary(self):
        formatter = FishScriptFormat()
        previous = "Mara: [my breath catches"
        addition = "<forceful>Wait!</forceful> We move."

        joined_previous, preserved_addition, recovered = formatter.recover_continuation(
            previous, addition
        )

        self.assertEqual(joined_previous, "Mara: [my breath catches]")
        self.assertEqual(preserved_addition, addition)
        self.assertTrue(recovered)
        self.assertTrue(joined_previous.endswith("]"))
        self.assertFalse(joined_previous.endswith("</forceful>"))

    def test_closes_inherited_cue_before_a_fresh_bracket_direction(self):
        formatter = FishScriptFormat()

        previous, addition, recovered = formatter.recover_continuation(
            "Mara: [my breath catches",
            "[my voice steadies] We move.",
        )

        self.assertEqual(previous, "Mara: [my breath catches]")
        self.assertEqual(addition, "[my voice steadies] We move.")
        self.assertTrue(recovered)

    def test_refuses_tagged_continuation_that_can_split_a_spoken_word(self):
        formatter = FishScriptFormat()
        previous = "[the parchment cr"
        addition = "<stress>runching</stress> the brittle edges."

        updated_previous, updated_addition, recovered = formatter.recover_continuation(
            previous, addition
        )

        self.assertEqual(updated_previous, previous)
        self.assertEqual(updated_addition, addition)
        self.assertFalse(recovered)

    def test_checks_first_speech_after_a_leading_bracket_direction(self):
        formatter = FishScriptFormat()
        previous = "[the parchment cr"
        addition = "[my voice breaks]runching is brittle."

        updated_previous, updated_addition, recovered = formatter.recover_continuation(
            previous, addition
        )

        self.assertEqual(updated_previous, previous)
        self.assertEqual(updated_addition, addition)
        self.assertFalse(recovered)

    def test_leaves_continuing_or_ambiguous_text_and_unknown_markup_untouched(self):
        formatter = FishScriptFormat()
        cases = (
            ("Mara: [my voice starts", " to shake] and we move."),
            ("Mara: [my voice starts", "<custom>new</custom> We move."),
            ("Mara: [my voice starts", "<forceful>Wait, then we move."),
            ("Mara: [my voice starts", "ordinary words continue here."),
            ("Mara: the cue is complete [steady]", "<forceful>Wait!</forceful>"),
        )

        for previous, addition in cases:
            with self.subTest(previous=previous, addition=addition):
                updated_previous, updated_addition, recovered = formatter.recover_continuation(
                    previous, addition
                )
                self.assertEqual(updated_previous, previous)
                self.assertEqual(updated_addition, addition)
                self.assertFalse(recovered)


if __name__ == "__main__":
    unittest.main()
