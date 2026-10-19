"""Multi-RefMod dialogue must follow character identity, not input/line order."""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3.ref2va import ensure_ref2va_prompt_relationships
from models.minimax_h3.speakers import _resolve_ref2va_dialogue_owner_name
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

    def test_responding_gerund_resolves_bea_and_inserts_second_speaker_id(self):
        prompt = (
            "At a quiet railway platform before dawn, adult sisters Ana and Bea wait beside one suitcase. "
            "At 0.00 seconds, Ana shifts her weight and speaks with a quiet, accusatory tone: "
            "<d>[English] You kept the blue ticket.</d> "
            "Bea does not turn away but lowers her gaze to the suitcase handle, "
            "responding with a calm, steady voice: "
            "<d>[English] I kept both of them.</d>"
        )
        second_line = re.search(r"<d>\[English\] I kept both of them\.</d>", prompt)
        self.assertIsNotNone(second_line)
        self.assertEqual(
            _resolve_ref2va_dialogue_owner_name(prompt, *second_line.span()),
            "Bea",
        )

        compiled = ensure_ref2va_prompt_relationships(
            prompt, references(["Ana", "Bea"]),
        )
        self.assert_speakers(compiled, [(1, 1), (2, 2)])
        ambiguous = (
            "Bea and Ana lower their gazes to the suitcase, "
            "responding with a calm, steady voice: "
            "<d>[English] I kept both of them.</d>"
        )
        with self.assertRaisesRegex(ValueError, "could not determine"):
            ensure_ref2va_prompt_relationships(
                ambiguous, references(["Ana", "Bea"]),
            )

    def test_responding_gerund_ignores_coordinated_prepositional_listeners(self):
        prompt = (
            'Bea looks at Ana and Tess, responding with a calm voice: '
            '<d>[English] I kept both of them.</d>'
        )
        dialogue = re.search(r"<d>.*?</d>", prompt)
        self.assertIsNotNone(dialogue)
        self.assertEqual(
            _resolve_ref2va_dialogue_owner_name(prompt, *dialogue.span()), "Bea",
        )
        compiled = ensure_ref2va_prompt_relationships(
            prompt, references(["Ana", "Bea", "Tess"]),
        )
        self.assert_speakers(compiled, [(2, 1)])

        finite_verb = (
            'Bea looked at Ana and Tess, then Tess responds: '
            '<d>[English] I kept both of them.</d>'
        )
        dialogue = re.search(r"<d>.*?</d>", finite_verb)
        self.assertEqual(
            _resolve_ref2va_dialogue_owner_name(finite_verb, *dialogue.span()),
            "Tess",
        )

    def test_parenthesized_markers_do_not_change_responding_subject_or_objects(self):
        refs = references(['Ana', 'Bea', 'Tess'])

        compound = (
            'Bea (S2) and Ana (S1) lower their gazes, responding with a calm voice: '
            '<d>[English] I kept both of them.</d>'
        )
        line = re.search(r'<d>.*?</d>', compound)
        self.assertEqual(
            _resolve_ref2va_dialogue_owner_name(compound, *line.span()), '',
        )
        with self.assertRaisesRegex(ValueError, 'could not determine'):
            ensure_ref2va_prompt_relationships(compound, refs)

        listeners = (
            'Bea looks at Ana (S1) and Tess (S3), responding with a calm voice: '
            '<d>[English] I kept both of them.</d>'
        )
        line = re.search(r'<d>.*?</d>', listeners)
        self.assertEqual(
            _resolve_ref2va_dialogue_owner_name(listeners, *line.span()), 'Bea',
        )
        self.assert_speakers(
            ensure_ref2va_prompt_relationships(listeners, refs), [(2, 1)],
        )

        finite_tess = (
            'Bea looked at Ana (S1) and Tess (S3), then Tess responds: '
            '<d>[English] I kept both of them.</d>'
        )
        line = re.search(r'<d>.*?</d>', finite_tess)
        self.assertEqual(
            _resolve_ref2va_dialogue_owner_name(finite_tess, *line.span()), 'Tess',
        )
        self.assert_speakers(
            ensure_ref2va_prompt_relationships(finite_tess, refs), [(3, 1)],
        )
    def test_leading_possessive_eyes_only_bind_with_same_subject_speech_cue(self):
        prompt = (
            'Ana shifts her weight and speaks: '
            '<d>[English] You kept the blue ticket.</d> '
            'Bea lowers her gaze to the suitcase handle, responding with a calm voice: '
            '<d>[English] I kept both of them.</d> '
            'Ana’s eyes narrow as she steps closer, her voice rising in confusion: '
            '<d>[English] Then why are you still here?</d>'
        )
        third_line = re.search(r'<d>\[English\] Then why are you still here\?</d>', prompt)
        self.assertIsNotNone(third_line)
        self.assertEqual(
            _resolve_ref2va_dialogue_owner_name(prompt, *third_line.span()), 'Ana',
        )
        marked_prompt = prompt.replace(
            'her voice rising in confusion:',
            'her voice rising in confusion: (S1)',
        )
        marked_line = re.search(
            r'<d>\[English\] Then why are you still here\?</d>', marked_prompt,
        )
        self.assertEqual(
            _resolve_ref2va_dialogue_owner_name(marked_prompt, *marked_line.span()),
            'Ana',
        )
        compiled = ensure_ref2va_prompt_relationships(
            prompt, references(['Ana', 'Bea']),
        )
        self.assert_speakers(compiled, [(1, 1), (2, 2), (1, 1)])

        michael = (
            'Michael’s eyes widen suddenly, and he leans further toward Jerry, '
            'nodding vigorously as he responds with high energy: '
            '<d>[English] Yeah! Like, they\'re the ham of sausages!</d>'
        )
        line = re.search(r'<d>.*?</d>', michael)
        self.assertEqual(
            _resolve_ref2va_dialogue_owner_name(michael, *line.span()), 'Michael',
        )
        self.assert_speakers(
            ensure_ref2va_prompt_relationships(michael, references(['Michael', 'Jerry'])),
            [(1, 1)],
        )

        for attribution in (
            'Bea watches Ana’s eyes and responds',
            'Ana’s eyes widen as Bea responds',
        ):
            with self.subTest(listener_or_switched_owner=attribution):
                source = f'{attribution}: <d>[English] I kept both of them.</d>'
                line = re.search(r'<d>.*?</d>', source)
                self.assertEqual(
                    _resolve_ref2va_dialogue_owner_name(source, *line.span()), 'Bea',
                )
                self.assert_speakers(
                    ensure_ref2va_prompt_relationships(
                        source, references(['Ana', 'Bea']),
                    ),
                    [(2, 1)],
                )

        vague = 'Ana’s eyes narrow suddenly: <d>[English] Then why?</d>'
        line = re.search(r'<d>.*?</d>', vague)
        self.assertNotEqual(
            _resolve_ref2va_dialogue_owner_name(vague, *line.span()), 'Ana',
        )

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

    def test_possessive_speech_attributions_are_kept_and_listener_objects_are_ignored(self):
        context = inventory(["Ana", "Bea"])
        entries = llm_service._extract_h3_source_dialogue_entries(
            "Ana and Bea stand together. Bea meets Ana’s eyes and answers: "
            "<d>[English] I kept both of them.</d>",
            context,
        )
        self.assertEqual([(entry.get("subject_id"), entry["speaker_id"]) for entry in entries], [(2, 1)])

        accepted = (
            "Ana's voice", "Ana’s reply", "Ana's whisper",
            "Ana's answer", "Ana’s response",
            "Ana's mouth opens as she says", "Ana’s mouth opens as she says",
            "Ana's lips move as she whispers", "Ana’s lips move as she whispers",
        )
        for attribution in accepted:
            with self.subTest(accepted_attribution=attribution):
                source = f"{attribution}: <d>[English] Hello.</d>"
                dialogue = re.search(r"<d\b.*?</d>", source)
                self.assertEqual(
                    _resolve_ref2va_dialogue_owner_name(source, *dialogue.span()), "Ana",
                )
                voice = llm_service._extract_h3_source_dialogue_entries(source, context)
                self.assertEqual(
                    [(entry.get("subject_id"), entry["speaker_id"]) for entry in voice],
                    [(1, 1)],
                )

        for attribution in (
            "Ana's voice", "Ana’s reply", "Ana's whisper",
            "Ana's answer", "Ana’s response",
        ):
            with self.subTest(comma_label=attribution):
                source = f"{attribution}, <d>[English] Hello.</d>"
                dialogue = re.search(r"<d\b.*?</d>", source)
                self.assertEqual(
                    _resolve_ref2va_dialogue_owner_name(source, *dialogue.span()), "Ana",
                )
                entries = llm_service._extract_h3_source_dialogue_entries(source, context)
                self.assertEqual(
                    [(entry.get("subject_id"), entry["speaker_id"]) for entry in entries],
                    [(1, 1)],
                )

        body_cues = (
            "Ana turns away. Ana's mouth opens as she says: <d>[English] Hello.</d>",
            "Ana turns away. Ana’s mouth opens as she says: <d>[English] Hello.</d>",
            "Then Ana's mouth opens as she says: <d>[English] Hello.</d>",
            "Then Ana’s mouth opens as she says: <d>[English] Hello.</d>",
            "Now Ana's lips move as she whispers: <d>[English] Hello.</d>",
            "Now Ana’s lips move as she whispers: <d>[English] Hello.</d>",
        )
        for source in body_cues:
            with self.subTest(body_cue=source):
                dialogue = re.search(r"<d\b.*?</d>", source)
                self.assertEqual(
                    _resolve_ref2va_dialogue_owner_name(source, *dialogue.span()), "Ana",
                )
                entries = llm_service._extract_h3_source_dialogue_entries(source, context)
                self.assertEqual(
                    [(entry.get("subject_id"), entry["speaker_id"]) for entry in entries],
                    [(1, 1)],
                )

        rejected_as_Ana = (
            "Bea watches Ana's lips and answers",
            "Bea watches Ana’s lips and answers",
            "Bea looks at Ana's mouth and says",
            "Bea looks at Ana’s mouth and says",
        )
        for attribution in rejected_as_Ana:
            with self.subTest(listener_attribution=attribution):
                source = f"{attribution}: <d>[English] I kept both of them.</d>"
                dialogue = re.search(r"<d\b.*?</d>", source)
                self.assertEqual(
                    _resolve_ref2va_dialogue_owner_name(source, *dialogue.span()), "Bea",
                )
                listener = llm_service._extract_h3_source_dialogue_entries(
                    source, context,
                )
                self.assertEqual(
                    [(entry.get("subject_id"), entry["speaker_id"]) for entry in listener],
                    [(2, 1)],
                )

    def test_postposed_speech_attribution_keeps_owner_and_source_entry(self):
        context = inventory(["Ana", "Bea"])
        for suffix in (", Ana says.", ", replies Ana."):
            with self.subTest(suffix=suffix):
                source = f"<d>[English] Hello.</d>{suffix}"
                dialogue = re.search(r"<d\b.*?</d>", source)
                self.assertEqual(
                    _resolve_ref2va_dialogue_owner_name(source, *dialogue.span()), "Ana",
                )
                entries = llm_service._extract_h3_source_dialogue_entries(source, context)
                self.assertEqual(
                    [(entry.get("subject_id"), entry["speaker_id"]) for entry in entries],
                    [(1, 1)],
                )

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
