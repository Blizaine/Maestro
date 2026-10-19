"""Prompt-local Ref2VA speaker proof and compatibility regressions."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3.ref2va import (
    ensure_ref2va_prompt_relationships,
    select_ref2va_window_voice_references,
)
from models.minimax_h3.speakers import (
    H3SpeakerBindingError,
    _resolve_ref2va_dialogue_owner_name,
    resolve_ref2va_dialogue_owners,
)


REFERENCES = [
    {
        "type": "image", "path": "alex.png", "role": "Alex",
        "image_intent": "identity", "character_name": "Alex",
    },
    {
        "type": "image", "path": "sam.png", "role": "Sam",
        "image_intent": "identity", "character_name": "Sam",
    },
    {"type": "audio", "path": "music.wav", "audio_intent": "style", "role": "music"},
    {
        "type": "audio", "path": "sam.wav", "audio_intent": "voice",
        "role": "Sam", "character_name": "Sam",
    },
]
ALIASES = {"alex": 1, "sam": 2}


def dialogue_spans(source: str) -> list[tuple[int, int, str]]:
    return [
        (match.start(), match.end(), match.group(1).strip())
        for match in re.finditer(r"<d(?:\s+[^>]*)?>(.*?)</d>", source, re.I | re.S)
    ]


class H3SpeakerCompatibilityTests(unittest.TestCase):
    def test_declared_subject_marker_handles_both_french_placements(self):
        for french in (
            (
                "(S1) says: <d>[French] C'est super, j'adore.</d>",
                "<d>[French] C'est super, j'adore.</d>",
            ),
            (
                "(S1) says: [French] <d>C'est super, j'adore.</d>",
                "[French] <d>C'est super, j'adore.</d>",
            ),
        ):
            line, expected_line = french
            with self.subTest(french=line):
                prompt = (
                    "subject_definitions: <Subject 1> is Alex from <Picture 1>. "
                    "<Subject 2> (S1) is Sam from <Picture 2>. "
                    "<Audio 2> is the voice-timbre reference for <Subject 2> (S1).\n"
                    "summary: Sam speaks.\n"
                    f"detailed_description: {line}"
                )
                compiled = ensure_ref2va_prompt_relationships(prompt, REFERENCES)
                self.assertIn("<Subject 2> is Sam", compiled)
                self.assertIn("<Audio 2>", compiled)
                self.assertIn(expected_line, compiled)
                self.assertEqual(
                    ensure_ref2va_prompt_relationships(compiled, REFERENCES), compiled,
                )

    def test_raw_marker_proof_is_reused_later_and_does_not_leak_between_calls(self):
        source = (
            "subject_definitions: <Subject 1> is Alex. <Subject 2> is Sam.\n"
            "detailed_description: Sam (S1) says <d>Bonjour.</d>\n"
            "[Shot 2] (S1) says <d>Encore.</d>"
        )
        spans = dialogue_spans(source)
        self.assertEqual(
            resolve_ref2va_dialogue_owners(source, spans, ALIASES, {1, 2}),
            [(2, "Sam"), (2, "")],
        )

        later_source = (
            "subject_definitions: <Subject 1> is Alex. <Subject 2> is Sam.\n"
            "detailed_description: (S1) says <d>Encore.</d>"
        )
        with self.assertRaises(H3SpeakerBindingError):
            resolve_ref2va_dialogue_owners(
                later_source, dialogue_spans(later_source), ALIASES, {1, 2},
            )

    def test_long_silent_context_cannot_override_a_proven_marker(self):
        source = (
            "subject_definitions: <Subject 1> is Alex. <Subject 2> is Sam.\n"
            "detailed_description: Alex (S1) says <d>First.</d> "
            + ("A long silent staging description. " * 20)
            + "Sam enters silently, then (S1) says <d>Second.</d>"
        )
        self.assertEqual(
            resolve_ref2va_dialogue_owners(
                source, dialogue_spans(source), ALIASES, {1, 2},
            ),
            [(1, "Alex"), (1, "")],
        )

    def test_direct_line_can_override_a_declaration_but_poisons_later_bare_use(self):
        source = (
            "subject_definitions: <Subject 1> (S2) is Alex. "
            "<Subject 2> (S1) is Sam.\n"
            "detailed_description: Alex says (S1) <d>First.</d> "
            "(S1) says <d>Second.</d>"
        )
        first_event = dialogue_spans(source)[0]
        self.assertEqual(
            resolve_ref2va_dialogue_owners(
                source, [first_event], ALIASES, {1, 2},
            ),
            [(1, "Alex")],
        )
        with self.assertRaises(H3SpeakerBindingError):
            resolve_ref2va_dialogue_owners(
                source, dialogue_spans(source), ALIASES, {1, 2},
            )

    def test_canonicalizer_ignores_definition_and_previous_shot_markers(self):
        prompt = (
            "subject_definitions: <Subject 1> (S2) is Alex. "
            "<Subject 2> (S1) is Sam.\n"
            "summary: A conversation.\n"
            "detailed_description: [Shot 1] (S2) stage cue.\n"
            "[Shot 2] Alex says <d>[English] Still here.</d>"
        )
        compiled = ensure_ref2va_prompt_relationships(prompt, REFERENCES)
        self.assertIn("[Shot 1] (S2) stage cue.", compiled)
        self.assertIn("[Shot 2] Alex says <Subject 1> (S1) <d>[English] Still here.</d>", compiled)
        self.assertEqual(
            ensure_ref2va_prompt_relationships(compiled, REFERENCES), compiled,
        )

    def test_declared_subject_number_is_independent_of_audio_ordinal(self):
        prompt = (
            "subject_definitions: <Subject 1> is Alex. "
            "<Subject 2> (S1) is Sam. "
            "<Audio 2> is the voice-timbre reference for <Subject 2> (S1).\n"
            "detailed_description: (S1) says <d>[French] Salut.</d>"
        )
        scoped_prompt, scoped, diagnostics = select_ref2va_window_voice_references(
            prompt, REFERENCES,
        )
        self.assertEqual(diagnostics["speaking_subjects"], [2])
        self.assertIn("<Subject 2>", scoped_prompt)
        self.assertIn("sam.wav", [item["path"] for item in scoped])
        self.assertNotIn("alex.wav", [item["path"] for item in scoped])

    def test_single_reference_keeps_existing_marker_fallback(self):
        prompt = "detailed_description: (S1) says <d>[French] Salut.</d>"
        references = [REFERENCES[0]]
        compiled = ensure_ref2va_prompt_relationships(prompt, references)
        self.assertIn("<Subject 1> (S1)", compiled)
        self.assertIn("[French] Salut.", compiled)

        no_subjects = "(S1) says <d>[French] Salut.</d>"
        self.assertEqual(
            resolve_ref2va_dialogue_owners(
                no_subjects, dialogue_spans(no_subjects), {}, set(),
            ),
            [(None, "")],
        )

    def test_known_guest_marker_reuse_stays_a_guest(self):
        source = (
            "subject_definitions: <Subject 1> is Alex. <Subject 2> is Sam.\n"
            "detailed_description: Nia (S4) says <d>Excuse me.</d> "
            "Then (S4) says <d>Thank you.</d>"
        )
        self.assertEqual(
            resolve_ref2va_dialogue_owners(
                source, dialogue_spans(source), ALIASES, {1, 2},
            ),
            [(None, "Nia"), (None, "Nia")],
        )

    def test_unproven_and_conflicting_markers_remain_ambiguous(self):
        unresolved = "detailed_description: (S1) says <d>Salut.</d>"
        with self.assertRaisesRegex(H3SpeakerBindingError, "could not determine"):
            ensure_ref2va_prompt_relationships(unresolved, REFERENCES[:2])

        conflicting_definition = (
            "subject_definitions: <Subject 1> (S1) is Alex. "
            "<Subject 2> (S1) is Sam.\n"
            "detailed_description: (S1) says <d>Salut.</d>"
        )
        with self.assertRaisesRegex(H3SpeakerBindingError, "could not determine"):
            ensure_ref2va_prompt_relationships(conflicting_definition, REFERENCES[:2])

        conflicting_reuse = (
            "subject_definitions: <Subject 1> is Alex. <Subject 2> is Sam.\n"
            "detailed_description: Alex (S1) says <d>Salut.</d>\n"
            "Sam (S1) says <d>Bonjour.</d>\n"
            "(S1) says <d>Encore.</d>"
        )
        with self.assertRaisesRegex(H3SpeakerBindingError, "could not determine"):
            ensure_ref2va_prompt_relationships(conflicting_reuse, REFERENCES[:2])

    def test_shot_labels_do_not_become_guest_owners(self):
        source = "detailed_description: [Shot 2] (S1) says <d>Salut.</d>"
        match = re.search(r"<d>.*?</d>", source)
        self.assertIsNotNone(match)
        self.assertEqual(_resolve_ref2va_dialogue_owner_name(source, *match.span()), "")
        with self.assertRaises(H3SpeakerBindingError):
            ensure_ref2va_prompt_relationships(source, REFERENCES[:2])

    def test_declared_voice_selection_keeps_sam_audio_for_bare_marker(self):
        prompt = (
            "subject_definitions: <Subject 1> is Alex. "
            "<Subject 2> (S1) is Sam.\n"
            "detailed_description: (S1) says <d>Bonjour.</d>"
        )
        _scoped_prompt, scoped, diagnostics = select_ref2va_window_voice_references(
            prompt, REFERENCES,
        )
        self.assertEqual(diagnostics["speaking_subjects"], [2])
        paths = [item["path"] for item in scoped]
        self.assertIn("sam.wav", paths)
        self.assertNotIn("alex.wav", paths)


if __name__ == "__main__":
    unittest.main()
