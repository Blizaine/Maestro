"""Native Subject/S cues outrank listener mentions in speaker ownership."""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3.ref2va import (
    _build_ref2va_character_bindings,
    ensure_ref2va_prompt_relationships,
    select_ref2va_window_voice_references,
)
from models.minimax_h3.speakers import H3SpeakerBindingError, resolve_ref2va_dialogue_owners
from services import llm_service


REFERENCES = [
    {"type": "image", "path": "avery.png", "role": "Avery", "library_character_id": "avery"},
    {"type": "audio", "path": "avery.wav", "role": "Avery", "audio_intent": "voice", "library_character_id": "avery"},
    {"type": "image", "path": "blair.png", "role": "Blair", "library_character_id": "blair"},
    {"type": "audio", "path": "blair.wav", "role": "Blair", "audio_intent": "voice", "library_character_id": "blair"},
    {"type": "image", "path": "cameron.png", "role": "Cameron", "library_character_id": "cameron"},
    {"type": "audio", "path": "cameron.wav", "role": "Cameron", "audio_intent": "voice", "library_character_id": "cameron"},
]

INVENTORY = "\n".join((
    'Saved character "Avery" is exactly <Subject 1>: <Picture 1> and <Audio 1> all define this one stable character.',
    'Saved character "Blair" is exactly <Subject 2>: <Picture 2> and <Audio 2> all define this one stable character.',
    'Saved character "Cameron" is exactly <Subject 3>: <Picture 3> and <Audio 3> all define this one stable character.',
))

NATIVE_PROMPT = "\n".join((
    "subject_definitions: <Subject 1> is Avery from <Picture 1>. <Subject 2> is Blair from <Picture 2>. "
    "<Subject 3> is Cameron from <Picture 3>. <Audio 1> is the voice-timbre reference for <Subject 1>. "
    "<Audio 2> is the voice-timbre reference for <Subject 2>. <Audio 3> is the voice-timbre reference for <Subject 3>.",
    "summary: Three friends hear a sound and decide what to do.",
    "retention_analysis: Preserve the three identities and their paired voice timbres.",
    "detailed_description: [Shot 1] <Subject 1> (S1) turns his head toward <Subject 2>, showing clear concern, and speaks in the voice referenced from <Audio 1>, <d>[English] Did you hear that?</d>",
    "[Shot 2] <Subject 2> (S2) replies to <Subject 3>, who watches, in the voice referenced from <Audio 2>, <d>[English] I heard it.</d>",
    "[Shot 3] <Subject 3> (S3) answers <Subject 2>, who turns toward them, in the voice referenced from <Audio 3>, <d>[English] We should leave.</d>",
    "[Shot 4] <Subject 1> (S1) says to <Subject 3>, who nods, in the voice referenced from <Audio 1>, <d>[English] Right now.</d>",
    "overall_soundscape: Quiet room tone.",
    "non_diegetic_music: N/A",
))


def spans(source: str) -> list[tuple[int, int, str]]:
    return [
        (match.start(), match.end(), match.group(1).strip())
        for match in re.finditer(r"<d(?:\s+[^>]*)?>(.*?)</d>", source, re.I | re.S)
    ]


ALIASES = _build_ref2va_character_bindings(REFERENCES)[-2]


class NativeSpeakerOwnershipTests(unittest.TestCase):
    def test_adjacent_subject_speaker_cue_wins_over_later_listener_subjects(self):
        entries = resolve_ref2va_dialogue_owners(
            NATIVE_PROMPT, spans(NATIVE_PROMPT), ALIASES, {1, 2, 3},
        )
        self.assertEqual(entries, [(1, ""), (2, ""), (3, ""), (1, "")])

        reversed_clause = (
            "subject_definitions: <Subject 1> is Avery. <Subject 2> is Blair.\n"
            "detailed_description: <Subject 2> (S2) says nothing while "
            "<Subject 1> (S1) replies <d>[English] Hello.</d>"
        )
        self.assertEqual(
            resolve_ref2va_dialogue_owners(
                reversed_clause, spans(reversed_clause), ALIASES, {1, 2},
            ),
            [(1, "")],
        )

    def test_llm_extractor_and_native_voice_selection_keep_first_vocal_order(self):
        entries = llm_service._extract_h3_source_dialogue_entries(NATIVE_PROMPT, INVENTORY)
        self.assertEqual([entry.get("subject_id") for entry in entries], [1, 2, 3, 1])
        self.assertEqual([entry["speaker_id"] for entry in entries], [1, 2, 3, 1])

        _scoped_prompt, scoped, diagnostics = select_ref2va_window_voice_references(
            NATIVE_PROMPT, REFERENCES, max_audio_references=1,
        )
        self.assertEqual(diagnostics["speaking_subjects"], [1, 2, 3])
        self.assertEqual(diagnostics["voice_kept"], 1)
        self.assertIn("avery.wav", [item["path"] for item in scoped])
        self.assertNotIn("blair.wav", [item["path"] for item in scoped])
        self.assertNotIn("cameron.wav", [item["path"] for item in scoped])

    def test_native_relationship_compilation_keeps_speaker_and_voice_pairings(self):
        compiled = ensure_ref2va_prompt_relationships(NATIVE_PROMPT, REFERENCES)
        entries = llm_service._extract_h3_source_dialogue_entries(compiled, INVENTORY)
        self.assertEqual([entry.get("subject_id") for entry in entries], [1, 2, 3, 1])
        self.assertEqual([entry["speaker_id"] for entry in entries], [1, 2, 3, 1])
        detail = compiled.split("detailed_description:", 1)[1].split("overall_soundscape:", 1)[0]
        dialogue_lines = [line for line in detail.splitlines() if "<d>" in line]
        for line, subject, speaker, audio in zip(dialogue_lines, (1, 2, 3, 1), (1, 2, 3, 1), (1, 2, 3, 1)):
            self.assertIn(f"<Subject {subject}> (S{speaker})", line)
            self.assertIn(f"<Audio {audio}>", line)

    def test_unmarked_ambiguous_line_remains_rejected(self):
        source = (
            "subject_definitions: <Subject 1> is Avery. <Subject 2> is Blair.\n"
            "detailed_description: <Subject 1> and <Subject 2> stand together; <d>Hello.</d>"
        )
        with self.assertRaises(H3SpeakerBindingError):
            resolve_ref2va_dialogue_owners(source, spans(source), ALIASES, {1, 2})


if __name__ == "__main__":
    unittest.main()
