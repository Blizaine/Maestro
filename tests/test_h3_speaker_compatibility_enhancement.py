"""Legacy H3 speaker ownership agrees in enhancement and final generation."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3.ref2va import ensure_ref2va_prompt_relationships
from models.minimax_h3.speakers import H3SpeakerBindingError
from services import llm_service


REFERENCES = [
    {"type": "image", "path": "alex.png", "role": "Alex", "library_character_id": "alex"},
    {"type": "image", "path": "sam.png", "role": "Sam", "library_character_id": "sam"},
    {"type": "audio", "path": "alex.wav", "role": "Alex", "audio_intent": "voice", "library_character_id": "alex"},
    {"type": "audio", "path": "sam.wav", "role": "Sam", "audio_intent": "voice", "library_character_id": "sam"},
]
INVENTORY = '\n'.join((
    'Saved character "Alex" is exactly <Subject 1>: <Picture 1> and <Audio 1> all define this one stable character.',
    'Saved character "Sam" is exactly <Subject 2>: <Picture 2> and <Audio 2> all define this one stable character.',
))
WORDS = ["C'est super, j'adore.", "Merci beaucoup.", "On recommence ?"]


def legacy_prompt(*, outside_language=False, first_marker=1, second_marker=2):
    def line(words):
        return f'[French] <d>{words}</d>' if outside_language else f'<d>[French] {words}</d>'
    return '\n'.join((
        f'subject_definitions: <Subject 1> (S{second_marker}) is Alex from <Picture 1>. '
        f'<Subject 2> (S{first_marker}) is Sam from <Picture 2>. '
        f'<Audio 1> is the voice-timbre reference for <Subject 1> (S{second_marker}). '
        f'<Audio 2> is the voice-timbre reference for <Subject 2> (S{first_marker}).',
        'summary: [reference generation] Two friends talk at a table.',
        'retention_analysis: Preserve both visual identities and their paired voice timbres.',
        f'detailed_description: [Shot 1] (S{first_marker}) says: {line(WORDS[0])}\n'
        f'[Shot 2] At 00:03.000, (S{second_marker}) replies: {line(WORDS[1])}\n'
        f'[Shot 3] At 00:06.000, (S{first_marker}) asks: {line(WORDS[2])}',
        'overall_soundscape: Quiet room ambience.',
        'non_diegetic_music: N/A',
    ))


class EnhancementSpeakerCompatibilityTests(unittest.TestCase):
    def assert_owners(self, source, expected):
        entries = llm_service._extract_h3_source_dialogue_entries(source, INVENTORY)
        self.assertEqual([(entry.get('subject_id'), entry['speaker_id']) for entry in entries], expected)
        return entries

    def test_french_legacy_declarations_match_generation_and_keep_exact_words(self):
        for outside in (False, True):
            with self.subTest(outside_language=outside):
                source = legacy_prompt(outside_language=outside)
                entries = self.assert_owners(source, [(2, 1), (1, 2), (2, 1)])
                self.assertEqual([entry['words'] for entry in entries], WORDS)
                self.assertEqual([entry['language'] for entry in entries], ['French'] * 3)
                compiled = ensure_ref2va_prompt_relationships(source, REFERENCES)
                self.assert_owners(compiled, [(2, 1), (1, 2), (2, 1)])
                for words in WORDS:
                    self.assertEqual(compiled.count(words), 1)

    def test_sparse_input_ids_are_canonicalized_by_vocal_event_order(self):
        source = legacy_prompt(first_marker=4, second_marker=9)
        self.assert_owners(source, [(2, 1), (1, 2), (2, 1)])
        compiled = ensure_ref2va_prompt_relationships(source, REFERENCES)
        self.assert_owners(compiled, [(2, 1), (1, 2), (2, 1)])

    def test_enhancement_and_generation_follow_permuted_audio_associations(self):
        source = legacy_prompt().replace('<Audio 1>', '<Audio TEMP>').replace(
            '<Audio 2>', '<Audio 1>',
        ).replace('<Audio TEMP>', '<Audio 2>')
        inventory = INVENTORY.replace('<Audio 1>', '<Audio TEMP>').replace(
            '<Audio 2>', '<Audio 1>',
        ).replace('<Audio TEMP>', '<Audio 2>')
        references = [*REFERENCES[:2], REFERENCES[3], REFERENCES[2]]
        result = llm_service._canonicalize_h3_ref2va_reference_fields(source, inventory, source)
        result = llm_service._canonicalize_h3_ref2va_dialogue_speakers(result, source, inventory)
        self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(source, result, inventory))
        compiled = ensure_ref2va_prompt_relationships(result, references)
        self.assert_owners(compiled, [(2, 1), (1, 2), (2, 1)])
        self.assertRegex(compiled, r'<Subject 2>\s+\(S1\)[^<]*<Audio 1>')
        self.assertRegex(compiled, r'<Subject 1>\s+\(S2\)[^<]*<Audio 2>')

    def test_known_speaker_survives_later_shot_without_repeating_the_name(self):
        source = (
            'detailed_description: [Shot 1] Sam (S1) says <d>[French] Bonjour.</d>\n'
            '[Shot 2] At 00:02.000, (S1) says <d>[French] Encore une fois.</d>'
        )
        self.assert_owners(source, [(2, 1), (2, 1)])
        self.assert_owners(ensure_ref2va_prompt_relationships(source, REFERENCES), [(2, 1), (2, 1)])

    def test_known_guest_marker_never_inherits_a_saved_portrait_or_voice(self):
        source = (
            'detailed_description: Jordan (S1) says <d>[French] Bonjour.</d>\n'
            '[Shot 2] (S1) says <d>[French] Encore une fois.</d> '
            'Alex (S2) replies <d>[French] Oui.</d>'
        )
        self.assert_owners(source, [(None, 1), (None, 1), (1, 2)])
        compiled = ensure_ref2va_prompt_relationships(source, REFERENCES)
        self.assert_owners(compiled, [(None, 1), (None, 1), (1, 2)])
        guest_lines = compiled.split('detailed_description:', 1)[1].split('Alex', 1)[0]
        self.assertNotIn('<Subject', guest_lines)
        self.assertNotIn('<Audio', guest_lines)

    def test_shot_labels_and_other_production_headers_are_not_guest_speakers(self):
        for heading in ('[Shot 2]', 'Shot 2:', '[Scene 3]', 'Camera:'):
            with self.subTest(heading=heading):
                source = f'detailed_description: {heading} (S1) says <d>[French] Bonjour.</d>'
                with self.assertRaises(H3SpeakerBindingError):
                    llm_service._extract_h3_source_dialogue_entries(source, INVENTORY)
                # Without attached identities, free-form enhancement can still
                # retain an event marker, but never invent a guest named Shot.
                entries = llm_service._extract_h3_source_dialogue_entries(source)
                self.assertEqual(len(entries), 1)
                self.assertNotIn(entries[0].get('speaker'), ('Shot', 'Shot 2', 'Scene', 'Camera'))

    def test_ambiguous_and_conflicting_declarations_still_require_ownership(self):
        ambiguous = 'detailed_description: (S1) says <d>[French] Bonjour.</d>'
        conflicting = (
            'subject_definitions: <Subject 1> (S1) is Alex. <Subject 2> (S1) is Sam.\n'
            'summary: Two friends.\ndetailed_description: (S1) says <d>[French] Bonjour.</d>'
        )
        for source in (ambiguous, conflicting):
            with self.subTest(source=source), self.assertRaises(H3SpeakerBindingError):
                llm_service._extract_h3_source_dialogue_entries(source, INVENTORY)

    def test_bindings_do_not_leak_between_enhancement_requests(self):
        self.assert_owners('Sam (S1) says <d>[French] Bonjour.</d>', [(2, 1)])
        with self.assertRaises(H3SpeakerBindingError):
            llm_service._extract_h3_source_dialogue_entries(
                'detailed_description: (S1) says <d>[French] Encore.</d>', INVENTORY,
            )

    def test_full_enhancement_flow_compiles_legacy_prompts_for_each_h3_reference_model(self):
        source = legacy_prompt()
        for model in ('minimax_h3_ref2va', 'minimax_h3_ref2va_full', 'minimax_h3_ref2va_fused_turbo'):
            with self.subTest(model=model), patch.object(
                llm_service, 'generate', return_value=source,
            ) as writer, patch(
                'services.enhance_guides.get_enhance_guide', return_value='Write a six-field H3 reference prompt.',
            ), patch(
                'services.studio_enhancement.fidelity_retry_limit', return_value=0,
            ):
                result = llm_service.enhance_prompt(
                    source, mode='video', model_type=model, reference_context=INVENTORY,
                    duration_seconds=14, planning_style='faithful',
                )
                self.assertEqual(writer.call_count, 1)
                self.assert_owners(result, [(2, 1), (1, 2), (2, 1)])
                self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(source, result, INVENTORY))
                compiled = ensure_ref2va_prompt_relationships(result, REFERENCES)
                self.assert_owners(compiled, [(2, 1), (1, 2), (2, 1)])
                self.assertRegex(compiled, r'<Subject 2>\s+\(S1\)[^<]*<Audio 2>')
                self.assertRegex(compiled, r'<Subject 1>\s+\(S2\)[^<]*<Audio 1>')
                for words in WORDS:
                    self.assertEqual(compiled.count(words), 1)


if __name__ == '__main__':
    unittest.main()
