"""User-reported H3 enhancement and music-reference failures, without inference."""

from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service
from services.h3_story_ledger import extract_locked_dialogue
from models.minimax_h3.ref2va import (
    ensure_ref2va_prompt_relationships,
    select_ref2va_window_voice_references,
)


REFERENCES = [
    {"type": "image", "path": "alex.png", "role": "Alex"},
    {"type": "image", "path": "sam.png", "role": "Sam"},
    {"type": "audio", "path": "music.wav", "audio_intent": "style"},
]
INVENTORY = "\n".join((
    '<Picture 1>: visual identity/appearance reference for Alex; retention=reference',
    '<Picture 2>: visual identity/appearance reference for Sam; retention=reference',
    '<Audio 1>: background music; intent=AUDIO REFERENCE; retention=weak_reference',
))
SILENT_DRAFT = "\n".join((
    'subject_definitions: <Subject 1> is Alex from <Picture 1>. '
    '<Subject 2> is Sam from <Picture 2>. <Audio 1> supplies music style.',
    'summary: [reference generation + audio reference] Two dancers in a courtyard.',
    'retention_analysis: <Picture 1>: fully_preserved. '
    '<Picture 2>: fully_preserved. <Audio 1>: weak_reference.',
    'detailed_description: [Shot 1] Alex and Sam dance together in the courtyard.',
    'overall_soundscape: Footsteps on stone. No speech.',
    'non_diegetic_music: Use the rhythm and texture of <Audio 1>.',
))


class H3PromptRegressionTests(unittest.TestCase):
    def test_quote_descriptors_after_speech_and_silence_cues_stay_visual(self):
        cases = (
            'Music only, no dialogue. Use a "warm vintage" palette.',
            'Character: "older woman". Audio: music.',
            'The "older woman" says nothing and dances.',
            'Use the music sample "English speaker", no voice cloning.',
            'A person speaks to the "older woman" through gestures.',
        )
        for prompt in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(llm_service._extract_h3_source_dialogue_entries(prompt), [])
                self.assertEqual(extract_locked_dialogue(prompt), [])
                self.assertNotIn("<d>", ensure_ref2va_prompt_relationships(prompt, REFERENCES))

    def test_quoted_short_lines_keep_expressive_and_postposed_attribution(self):
        for prompt in (
            'Sam explains, "Go."', 'Sam muffles softly, "Go."',
            'Sam: "Go."', 'Sam (S1) "Go."',
            '"Go," Sam says.', '"Go," replies Sam.',
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(len(llm_service._extract_h3_source_dialogue_entries(prompt)), 1)
                compiled = ensure_ref2va_prompt_relationships(prompt, REFERENCES)
                self.assertEqual(compiled.count("<d>"), 1)
                self.assertIn("<Subject 2> (S1)", compiled)

    def test_live_reference_canonicalizer_without_source_dialogue(self):
        # Import the real module: a helper test must not supply missing globals.
        result = llm_service._canonicalize_h3_ref2va_reference_fields(
            SILENT_DRAFT, INVENTORY, "Two people dance. No dialogue."
        )
        self.assertIn("detailed_description: [Shot 1] Alex and Sam dance", result)
        self.assertNotIn("<d>", result)

    def test_enhancement_without_dialogue_on_all_omni_variants(self):
        for model in ("minimax_h3_ref2va", "minimax_h3_ref2va_full",
                      "minimax_h3_ref2va_fused_turbo"):
            for style in ("faithful", "creative"):
                with self.subTest(model=model, style=style), mock.patch.object(
                    llm_service, "generate", return_value=SILENT_DRAFT
                ), mock.patch(
                    "services.enhance_guides.get_enhance_guide",
                    return_value="Write a six-field H3 Ref2VA prompt.",
                ):
                    enhanced = llm_service.enhance_prompt(
                        "Alex and Sam dance together. Music only, no dialogue.",
                        mode="video", model_type=model, reference_context=INVENTORY,
                        duration_seconds=10, planning_style=style,
                    )
                    compiled = ensure_ref2va_prompt_relationships(enhanced, REFERENCES)
                    self.assertNotIn("<d>", compiled)
                    self.assertIn("<Audio 1>", compiled)

    def test_malformed_ai_speaker_output_reaches_repair_instead_of_http_error(self):
        bad = SILENT_DRAFT.replace(
            'Alex and Sam dance together in the courtyard.',
            '(S1) <d>[English] older woman</d>',
        )
        with mock.patch.object(llm_service, "generate", side_effect=[bad, SILENT_DRAFT]) as generate, mock.patch(
            "services.enhance_guides.get_enhance_guide", return_value="Write a six-field H3 prompt.",
        ):
            result = llm_service.enhance_prompt(
                'Alex and Sam dance to music. No dialogue.', model_type='minimax_h3_ref2va',
                reference_context=INVENTORY, duration_seconds=10,
            )
        self.assertEqual(generate.call_count, 2)
        self.assertNotIn('<d>', result)
        self.assertIn('<Audio 1>', result)
        self.assertIn('dance together', result)

    def test_explicit_music_only_request_rejects_even_well_formed_ai_speech(self):
        bad = SILENT_DRAFT.replace(
            'Alex and Sam dance together in the courtyard.',
            '<Subject 1> (S1) says <d>[English] Hello.</d>',
        )
        with mock.patch.object(llm_service, "generate", return_value=bad) as generate, mock.patch(
            "services.enhance_guides.get_enhance_guide", return_value="Write a six-field H3 prompt.",
        ):
            result = llm_service.enhance_prompt(
                'Alex and Sam dance to music. No dialogue.', model_type='minimax_h3_ref2va',
                reference_context=INVENTORY, duration_seconds=10,
            )
        self.assertEqual(generate.call_count, 2)
        self.assertNotIn('<d>', result)
        self.assertIn('<Audio 1>', result)

    def test_raw_quoted_descriptions_are_not_spoken_with_music(self):
        prompt = (
            'The "older woman" dances with the "English speaker" to the music. '
            'Use a "warm vintage" look. No dialogue.'
        )
        for audio in ([], [REFERENCES[-1]], [dict(REFERENCES[-1], audio_intent="drive")]):
            with self.subTest(audio=audio):
                result = ensure_ref2va_prompt_relationships(prompt, REFERENCES[:2] + audio)
                self.assertNotIn("<d>", result)
                self.assertIn('"older woman"', result)
                self.assertIn('"English speaker"', result)

    def test_enhancer_does_not_lock_quoted_character_descriptions(self):
        prompt = 'The "older woman" and the "English speaker" dance to music.'
        self.assertEqual(
            llm_service._extract_h3_source_dialogue_entries(prompt, INVENTORY), []
        )

    def test_structured_identity_names_are_not_extra_dialogue(self):
        prompt = SILENT_DRAFT.replace(
            'is Alex from', 'is the character "Alex" from'
        ).replace(
            'is Sam from', 'is the character "Sam" from'
        ).replace(
            'Alex and Sam dance together in the courtyard.',
            '<Subject 2> (S1) says <d>[English] Follow my lead.</d>',
        )
        entries = llm_service._extract_h3_source_dialogue_entries(prompt, INVENTORY)
        self.assertEqual([entry["words"] for entry in entries], ["Follow my lead."])
        self.assertEqual(entries[0]["subject_id"], 2)

    def test_speaker_count_is_independent_of_saved_reference_count(self):
        context = INVENTORY.splitlines()[0]
        source = (
            'Sam says, "We are ready." Alex replies, "Let us begin." '
            'Jordan adds, "I will help."'
        )
        draft = llm_service._build_h3_ref2va_tagged_fallback(
            source, context, duration_seconds=10,
        )
        self.assertTrue(llm_service._h3_ref2va_reference_contract_satisfied(draft, context))
        self.assertFalse(llm_service._h3_ref2va_reference_contract_satisfied(
            draft.replace('<Subject 1>', '<Subject 9>'), context,
        ))
        self.assertFalse(llm_service._h3_ref2va_reference_contract_satisfied(
            draft.replace('(S3)', '(S99)'), context,
        ))
        compiled = ensure_ref2va_prompt_relationships(draft, REFERENCES[:1])
        self.assertEqual(compiled.count('<d>'), 3)
        self.assertIn('<Subject 1> (S2)', compiled)

    def test_music_intent_overrides_voice_words_in_reference_description(self):
        for intent in ("AUDIO REFERENCE", "AUDIO REUSE / PERFORMANCE DRIVER"):
            with self.subTest(intent=intent):
                inventory = INVENTORY.replace(
                    "background music;", "English speaker voice sample music;"
                ).replace("intent=AUDIO REFERENCE", f"intent={intent}")
                subjects = llm_service._parse_h3_ref2va_subject_manifest(inventory)
                self.assertEqual([subject["audios"] for subject in subjects], [[], []])

    def test_silent_windows_keep_music_without_selecting_voices_for_quoted_labels(self):
        references = REFERENCES + [
            {"type": "audio", "path": "voice.wav", "audio_intent": "voice"},
        ]
        _, selected, _ = select_ref2va_window_voice_references(
            'Two people dance with a "vintage" look.', references,
        )
        self.assertEqual([item["path"] for item in selected], [
            "alex.png", "sam.png", "music.wav",
        ])

    def test_actual_quotes_speech_tags_and_speaker_bindings_survive(self):
        cases = (
            'Sam says, "Follow my lead." Alex replies, "I will."',
            'Sam (S1) <d>[English] Follow my lead.</d> '
            'Alex (S2) <d>[English] I will.</d>',
        )
        for prompt in cases:
            with self.subTest(prompt=prompt):
                result = ensure_ref2va_prompt_relationships(prompt, REFERENCES)
                self.assertIn('<Subject 2> (S1)', result)
                self.assertIn('<Subject 1> (S2)', result)
                self.assertEqual(result.count('<d>[English] Follow my lead.</d>'), 1)
                self.assertEqual(result.count('<d>[English] I will.</d>'), 1)

    def test_quotes_inside_tagged_speech_are_not_tagged_twice(self):
        prompt = 'Alex says <d>[English] She called it "a miracle".</d>'
        result = ensure_ref2va_prompt_relationships(prompt, REFERENCES)
        self.assertEqual(result.count("<d>"), 1)
        self.assertIn('<d>[English] She called it "a miracle".</d>', result)

    def test_real_ambiguous_dialogue_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "could not determine"):
            ensure_ref2va_prompt_relationships(
                '(S1) <d>[English] Follow my lead.</d>', REFERENCES
            )


if __name__ == "__main__":
    unittest.main()
