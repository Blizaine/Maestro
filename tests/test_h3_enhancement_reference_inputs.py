"""Reference enhancement uses native bindings without rejecting layout whitespace."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3.ref2va import (
    _build_ref2va_character_bindings,
    ensure_ref2va_prompt_relationships,
    select_ref2va_window_voice_references,
)
from services import llm_service
from services.studio_enhancement import canonical_h3_enhancement_inputs, enhancement_request


REFERENCES = [
    {"type": "image", "path": "rhea.png", "role": "Rhea", "library_character_id": "rhea"},
    {"type": "image", "path": "noah.png", "role": "Noah", "library_character_id": "noah"},
    {"type": "audio", "path": "rhea.wav", "role": "Rhea", "audio_intent": "voice", "library_character_id": "rhea"},
    {"type": "audio", "path": "noah.wav", "role": "Noah", "audio_intent": "voice", "library_character_id": "noah"},
]
INVENTORY = '\n'.join((
    'Saved character "Rhea" is exactly <Subject 1>: <Picture 1> + <Audio 1> all define this one stable character.',
    'Saved character "Noah" is exactly <Subject 2>: <Picture 2> + <Audio 2> all define this one stable character.',
))
SOURCE = (
    'Use only these supplied English lines verbatim, once each and in this order. No other dialogue.\n'
    'Rhea says: "Hello,  Noah."\nNoah replies: "Hello, Rhea."\nRhea says: "Please  take a seat."'
)
DRAFT = '\n'.join((
    'subject_definitions: <Subject 1> (S1) is Rhea from <Picture 1>. '
    '<Subject 2> (S2) is Noah from <Picture 2>. '
    '<Audio 1> is the voice-timbre reference for <Subject 1> (S1). '
    '<Audio 2> is the voice-timbre reference for <Subject 2> (S2).',
    'summary: [Conversation] Two friends greet each other in a quiet room.',
    'retention_analysis: Preserve both identities and their own voice timbres.',
    'detailed_description: [Shot 1] <Subject 1> (S1) says in the voice referenced from <Audio 1>, <d>[English] Hello, Noah.</d>\n'
    '[Shot 2] <Subject 2> (S2) replies in the voice referenced from <Audio 2>, <d>[English] Hello, Rhea.</d>\n'
    '[Shot 3] <Subject 1> (S1) says in the voice referenced from <Audio 1>, <d>[English] Please take a seat.</d>',
    'overall_soundscape: Quiet room ambience. No other voices.',
    'non_diegetic_music: N/A',
))


class DialogueWhitespaceTests(unittest.TestCase):
    def test_spacing_and_line_wrapping_do_not_change_speech_or_ownership(self):
        for draft in (DRAFT, DRAFT.replace('Hello, Noah.', 'Hello,\n\tNoah.').replace('Please take', 'Please\u00a0take')):
            with self.subTest(draft=draft):
                self.assertTrue(llm_service._h3_dialogue_contract_satisfied(SOURCE, draft))
                self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(SOURCE, draft, INVENTORY))
                repaired = llm_service._canonicalize_h3_ref2va_dialogue_speakers(draft, SOURCE, INVENTORY)
                self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(SOURCE, repaired, INVENTORY))

    def test_actual_word_language_or_punctuation_changes_are_rejected(self):
        for draft in (
            DRAFT.replace('Hello, Noah.', 'Goodbye, Noah.'),
            DRAFT.replace('[English]', '[French]'),
            DRAFT.replace('Hello, Noah.', 'Hello Noah.'),
        ):
            with self.subTest(draft=draft):
                self.assertFalse(llm_service._h3_dialogue_contract_satisfied(SOURCE, draft))

    def test_closed_script_order_and_duplicate_checks_remain_strict(self):
        first = '<Subject 1> (S1) says <d>[English] Hello, Noah.</d>'
        reordered = DRAFT.replace('Hello, Noah.', 'TEMP').replace('Hello, Rhea.', 'Hello, Noah.').replace('TEMP', 'Hello, Rhea.')
        for draft in (DRAFT.replace('overall_soundscape:', first + '\noverall_soundscape:'), reordered):
            with self.subTest(draft=draft):
                self.assertFalse(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(SOURCE, draft, INVENTORY))

    def test_wrong_speaker_does_not_pass_due_to_matching_words(self):
        draft = DRAFT.replace('<Subject 1> (S1) says in the voice referenced from <Audio 1>',
                              '<Subject 2> (S2) says in the voice referenced from <Audio 2>', 1)
        self.assertFalse(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(SOURCE, draft, INVENTORY))

    def test_full_enhancement_keeps_safe_whitespace_draft_without_repairs(self):
        with patch.object(llm_service, 'generate', return_value=DRAFT) as writer, patch(
            'services.enhance_guides.get_enhance_guide', return_value='Write a six-field H3 reference prompt.',
        ), patch(
            'services.h3_single_window_continuity.review_single_window_continuity',
            side_effect=lambda **kwargs: {"prompt": kwargs['draft_prompt'], "checked": False},
        ):
            result = llm_service.enhance_prompt(
                SOURCE, mode='video', model_type='minimax_h3_ref2va_pika_flex',
                reference_context=INVENTORY, duration_seconds=15, planning_style='faithful',
            )
        self.assertEqual(writer.call_count, 1)
        self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(SOURCE, result, INVENTORY))
        scoped, refs, diagnostics = select_ref2va_window_voice_references(result, REFERENCES)
        self.assertEqual(diagnostics['voice_kept'], 2)
        self.assertEqual(diagnostics['voice_omitted'], 0)
        self.assertTrue(ensure_ref2va_prompt_relationships(scoped, refs))


class AuthoritativeReferenceInputsTests(unittest.TestCase):
    def test_enhance_now_and_queued_writer_inputs_share_actual_reference_bindings(self):
        for intent in ('identity', 'scene', 'composition', 'style', 'object'):
            with self.subTest(intent=intent):
                refs = [
                    {"type": "image", "path": "room.png", "role": "Room", "image_intent": intent},
                    {"type": "video", "path": "rhea.mp4", "role": "Rhea", "video_intent": "character", "library_character_id": "rhea", "include_audio": False},
                    {"type": "video", "path": "noah.mp4", "role": "Noah", "video_intent": "character", "library_character_id": "noah", "include_audio": False},
                    *deepcopy(REFERENCES[2:]),
                ]
                before = deepcopy(refs)
                interactive = canonical_h3_enhancement_inputs(SOURCE, refs)
                queued, _sequence = enhancement_request(
                    {"prompt": SOURCE, "model_type": "minimax_h3_ref2va_pika_flex", "video_length": 362, "minimax_h3_references": refs},
                    {"architecture": "minimax_h3_ref2va", "omni_reference": True, "fps": 24, "frames_maximum": 362},
                )
                for key in ('prompt', 'reference_context', 'image_paths', 'minimax_h3_references'):
                    self.assertEqual(interactive[key], queued[key])
                self.assertEqual(refs, before)
                aliases = _build_ref2va_character_bindings(interactive['minimax_h3_references'])[-2]
                parsed = llm_service._parse_h3_ref2va_subject_manifest(interactive['reference_context'])
                for name in ('Rhea', 'Noah'):
                    subject = next(item for item in parsed if item['name'] == name)
                    self.assertEqual(subject['index'], aliases[name.lower()])
                    self.assertTrue(subject['audios'])
                if intent in ('composition', 'object'):
                    self.assertFalse(any(item['name'] == 'Room' for item in parsed))

    def test_bad_reference_shapes_fail_before_writer_dispatch(self):
        for references in ('not a list', [{'type': 'image'}], [{'type': 'invalid', 'path': 'ref.png'}]):
            with self.subTest(references=references), self.assertRaises(ValueError):
                canonical_h3_enhancement_inputs(SOURCE, references)


if __name__ == '__main__':
    unittest.main()
