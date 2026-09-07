"""Multi-RefMod dialogue must follow character identity, not input/line order."""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3.ref2va import ensure_ref2va_prompt_relationships
from services import llm_service


def references(names):
    return [
        {"type": "video", "path": f"{index}.mp4", "video_intent": "character",
         "include_audio": False, "character_name": name, "role": name,
         "library_character_id": f"character-{index}"}
        for index, name in enumerate(names, 1)
    ]


def inventory(names):
    return "\n".join(
        f'Saved character "{name}" is exactly <Subject {index}>: <Video {index}> '
        'all define this one stable character.'
        for index, name in enumerate(names, 1)
    )


class RefModSpeakerTests(unittest.TestCase):
    names = ["minimaxh3_sydneysweeney_v1_refmod", "minimaxh3_elizadushku_v1_refmod"]
    prompt = (
        'Eliza Dushku is standing in a modern kitchen. She says, '
        '<d>[English] Hey babe. Can I make you a sandwich?</d> '
        'Sydney Sweeney smiles and replies, <d>[English] Yes, please.</d>'
    )

    def assert_speakers(self, compiled, expected):
        actual = re.findall(r"<Subject (\d+)>\s+\(S(\d+)\)[^<]*<d>", compiled)
        self.assertEqual(actual, [(str(subject), str(speaker)) for subject, speaker in expected])

    def test_manual_pronoun_speech_matches_imported_refmod_names(self):
        self.assert_speakers(ensure_ref2va_prompt_relationships(self.prompt, references(self.names)), [(2, 1), (1, 2)])

    def test_quotes_and_tags_share_binding(self):
        prompt = 'Eliza Dushku says, "Hello." Sydney Sweeney replies, "Good morning."'
        self.assert_speakers(ensure_ref2va_prompt_relationships(prompt, references(self.names)), [(2, 1), (1, 2)])

    def test_three_refmods_keep_repeated_speaker_and_original_subject_order(self):
        names = [*self.names, "minimaxh3_carriefisher_v1_refmod.safetensors"]
        prompt = (
            'Carrie Fisher says, <d>Welcome.</d> Eliza Dushku replies, <d>Thank you.</d> '
            'Sydney Sweeney asks, <d>Shall we start?</d> Carrie Fisher replies, <d>Yes.</d>'
        )
        self.assert_speakers(ensure_ref2va_prompt_relationships(prompt, references(names)), [(3, 1), (2, 2), (1, 3), (3, 1)])

    def test_friendly_display_name_and_full_filename_remain_valid(self):
        for name in ("elizadushku v1", "Eliza Dushku", self.names[1]):
            with self.subTest(name=name):
                self.assert_speakers(ensure_ref2va_prompt_relationships(f'{name}: <d>Hello.</d>', references(self.names)), [(2, 1)])

    def test_similar_names_do_not_match_by_substring(self):
        names = ["minimaxh3_annalee_v1_refmod", "minimaxh3_annabellelee_v1_refmod"]
        prompt = 'Annabelle Lee says, <d>Hello.</d>'
        compiled = ensure_ref2va_prompt_relationships(prompt, references(names))
        self.assert_speakers(compiled, [(2, 1)])

    def test_two_versions_require_specific_version_or_subject(self):
        names = ["minimaxh3_elizadushku_v1_refmod", "minimaxh3_elizadushku_v2_refmod"]
        compiled = ensure_ref2va_prompt_relationships('elizadushku v2: <d>Hello.</d>', references(names))
        self.assert_speakers(compiled, [(2, 1)])

    def test_refmod_voices_follow_identity_when_second_character_speaks_first(self):
        refs = references(self.names)
        for index, name in enumerate(self.names, 1):
            refs.append({"type": "audio", "path": f"{index}.wav", "audio_intent": "voice",
                         "character_name": name, "role": name + " voice",
                         "library_character_id": f"character-{index}"})
        compiled = ensure_ref2va_prompt_relationships(self.prompt, refs)
        self.assertIn('<Subject 2> (S1) in the voice referenced from <Audio 2>, <d>', compiled)
        self.assertIn('<Subject 1> (S2) in the voice referenced from <Audio 1>, <d>', compiled)

    def test_enhance_keeps_named_guests_separate_from_saved_refmods(self):
        entries = llm_service._extract_h3_source_dialogue_entries(
            'Rachel asks, <d>Hello?</d> Eliza Dushku replies, <d>Welcome.</d>', inventory(self.names))
        self.assertEqual([(entry.get('subject_id'), entry['speaker_id']) for entry in entries], [(None, 1), (2, 2)])

    def test_ambiguous_pronoun_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'could not determine'):
            ensure_ref2va_prompt_relationships(
                'Eliza Dushku and Sydney Sweeney stand together. She says <d>Hello.</d>', references(self.names))

    def test_enhance_uses_the_same_natural_name_and_pronoun_bindings(self):
        entries = llm_service._extract_h3_source_dialogue_entries(self.prompt, inventory(self.names))
        self.assertEqual([(entry.get('subject_id'), entry.get('speaker_id')) for entry in entries], [(2, 1), (1, 2)])

    def test_enhance_fallback_can_be_sent_directly_to_reference_generation(self):
        enhanced = llm_service._build_h3_ref2va_tagged_fallback(self.prompt, inventory(self.names), duration_seconds=10.125)
        self.assert_speakers(ensure_ref2va_prompt_relationships(enhanced, references(self.names)), [(2, 1), (1, 2)])

    def test_enhance_repairs_unnamed_lines_with_explicit_subjects(self):
        enhanced = (
            'subject_definitions: <Subject 1> and <Subject 2>.\nsummary: A kitchen conversation.\n'
            'retention_analysis: Retain both characters.\ndetailed_description: She says (S1) '
            '<d>[English] Hey babe. Can I make you a sandwich?</d> She replies (S2) '
            '<d>[English] Yes, please.</d>\noverall_soundscape: Room tone.\nnon_diegetic_music: N/A'
        )
        repaired = llm_service._canonicalize_h3_ref2va_dialogue_speakers(enhanced, self.prompt, inventory(self.names))
        self.assert_speakers(ensure_ref2va_prompt_relationships(repaired, references(self.names)), [(2, 1), (1, 2)])

    def test_enhance_does_not_assign_an_ambiguous_line_by_round_robin(self):
        with self.assertRaisesRegex(ValueError, 'could not determine'):
            llm_service._extract_h3_source_dialogue_entries(
                'Eliza Dushku and Sydney Sweeney stand together. She says <d>Hello.</d>', inventory(self.names))


if __name__ == '__main__':
    unittest.main()
