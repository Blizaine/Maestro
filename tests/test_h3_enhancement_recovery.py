"""Recovery output keeps cast ownership and native sound fields authoritative."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services import llm_service


MOTION_ONLY_CONTEXT = "\n".join((
    "<Subject 1> is the environment and location for Harbor, defined by <Picture 1>.",
    "<Video 1> provides motion, camera, scene, or timing reference for Mira.",
    "<Audio 1> is the synchronized soundtrack paired with <Video 1>; reuse its audible timeline.",
    "<Video 2> provides motion, camera, scene, or timing reference for Tavi.",
    "<Audio 2> is the synchronized soundtrack paired with <Video 2>; reuse its audible timeline.",
    "<Video 3> provides motion, camera, scene, or timing reference for Juno.",
    "<Audio 3> is the synchronized soundtrack paired with <Video 3>; reuse its audible timeline.",
    "<Picture 1>: intent=scene",
    "<Video 1>: partially_preserved",
    "<Audio 1>: partially_copy",
    "<Video 2>: partially_preserved",
    "<Audio 2>: partially_copy",
    "<Video 3>: partially_preserved",
    "<Audio 3>: partially_copy",
))

CHARACTER_CONTEXT = "\n".join((
    "<Subject 1> is the environment and location for Harbor, defined by <Picture 1>.",
    "<Subject 2> is Mira from <Video 1>, preserving identity, appearance, and characteristic motion.",
    "<Subject 3> is Tavi from <Video 2>, preserving identity, appearance, and characteristic motion.",
    "<Subject 4> is Juno from <Video 3>, preserving identity, appearance, and characteristic motion.",
    "<Audio 1> is the voice-timbre reference for <Subject 2>.",
    "<Audio 2> is the voice-timbre reference for <Subject 3>.",
    "<Audio 3> is the voice-timbre reference for <Subject 4>.",
))


CAST_MAP = """Characters:
(S1) Mira, <Video 1> and <Audio 1> as references.
(S2) Tavi, <Video 2> and <Audio 2> as references.
(S3) Juno, <Video 3> and <Audio 3> as references.
"""


def conversation(*, native_audio_fields=False):
    source = CAST_MAP + """Scene: Mira and Tavi wait beside the harbor railing. Juno approaches.
(S1) asks (S2) with a worried expression <d>[English] Did you hear that?</d>
(S2) says in a steady voice to (S3) <d>[English] I heard it.</d>
(S3) says <d>[English] Then we should move.</d>
(S1) says <d>[English] Go now.</d>
"""
    if native_audio_fields:
        source += (
            "overall_soundscape: Rain taps the railing; a distant ferry horn fades.\n"
            "non_diegetic_music: A quiet synth pulse rises after the final line.\n"
        )
    return source


class H3EnhancementRecoveryTests(unittest.TestCase):
    def test_named_prop_does_not_take_ownership_of_a_chained_cast_reply(self):
        source = CAST_MAP + (
            'Scene: The friends meet beside the harbor.\n'
            '(S1) Mira says <d>Are you ready?</d>\n'
            '(S2) Tavi replies <d>Nearly.</d>\n'
            '(S2) Tavi then holds the Compass, Northstar, in front of him and smiles '
            'and says softly <d>I found you.</d>'
        )
        entries = llm_service._extract_h3_source_dialogue_entries(source, MOTION_ONLY_CONTEXT)
        self.assertEqual([entry['speaker'] for entry in entries], ['Mira', 'Tavi', 'Tavi'])
        self.assertEqual([entry['speaker_id'] for entry in entries], [1, 2, 2])
        fallback = llm_service._build_h3_ref2va_tagged_fallback(
            source, MOTION_ONLY_CONTEXT, duration_seconds=12,
        )
        compiled = llm_service._extract_h3_source_dialogue_entries(fallback, MOTION_ONLY_CONTEXT)
        self.assertEqual([entry['speaker_id'] for entry in compiled], [1, 2, 2])

    def test_a_named_guest_after_a_cast_action_keeps_its_own_voice(self):
        source = CAST_MAP + (
            'Scene: The friends meet beside the harbor.\n'
            '(S1) Mira says <d>Are you ready?</d>\n'
            '(S2) Tavi replies <d>Nearly.</d>\n'
            '(S2) Tavi holds the Compass and Rook says softly <d>I found you.</d>'
        )
        entries = llm_service._extract_h3_source_dialogue_entries(source, MOTION_ONLY_CONTEXT)
        self.assertEqual([entry['speaker'] for entry in entries], ['Mira', 'Tavi', 'Rook'])
        self.assertEqual([entry['speaker_id'] for entry in entries], [1, 2, 3])

    def test_rendering_cast_markers_as_names_does_not_duplicate_adjacent_names(self):
        source = '(S2) Tavi replies to Mira (S1); (S3) watches. <d>Go.</d>'
        normalized = llm_service._canonicalize_h3_source_cast_markers(
            source, {1: 'Mira', 2: 'Tavi', 3: 'Juno'},
        )
        self.assertNotIn('Tavi Tavi', normalized)
        self.assertNotIn('Mira Mira', normalized)
        self.assertIn('Juno watches.', normalized)
        self.assertIn('<d>Go.</d>', normalized)

    def test_cast_manifest_resolves_speaker_before_recipient_with_scene_subject_present(self):
        source = conversation()
        entries = llm_service._extract_h3_source_dialogue_entries(source, CHARACTER_CONTEXT)
        self.assertEqual(
            [(entry.get("subject_id"), entry.get("speaker"), entry["speaker_id"]) for entry in entries],
            [(2, "Mira", 1), (3, "Tavi", 2), (4, "Juno", 3), (2, "Mira", 1)],
        )

    def test_multiline_cast_manifest_does_not_bind_scene_subject_or_unselected_audio_as_voice(self):
        source = conversation(native_audio_fields=True)
        entries = llm_service._extract_h3_source_dialogue_entries(source, MOTION_ONLY_CONTEXT)
        self.assertEqual(
            [(entry.get("subject_id"), entry.get("speaker"), entry["speaker_id"]) for entry in entries],
            [(None, "Mira", 1), (None, "Tavi", 2), (None, "Juno", 3), (None, "Mira", 1)],
        )
        fallback = llm_service._build_h3_ref2va_tagged_fallback(
            source, MOTION_ONLY_CONTEXT, duration_seconds=12,
        )
        self.assertTrue(llm_service._has_complete_h3_ref2va_structure(fallback))
        self.assertTrue(llm_service._h3_ref2va_reference_contract_satisfied(fallback, MOTION_ONLY_CONTEXT))
        self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(
            source, fallback, MOTION_ONLY_CONTEXT,
        ))
        self.assertIn("<Subject 1> is the environment and location", fallback)
        self.assertNotRegex(fallback, r"<Subject 1>\s+\(S[123]\)")
        self.assertNotIn("Voice references supply vocal identity", fallback)
        self.assertNotIn("in the voice referenced from <Audio", fallback)
        self.assertEqual(llm_service._extract_h3_dialogue_blocks(fallback), [
            "Did you hear that?", "I heard it.", "Then we should move.", "Go now.",
        ])
        self.assertEqual(fallback.count("overall_soundscape:"), 1)
        self.assertEqual(fallback.count("non_diegetic_music:"), 1)
        self.assertEqual(fallback.count("Rain taps the railing; a distant ferry horn fades."), 1)
        self.assertEqual(fallback.count("A quiet synth pulse rises after the final line."), 1)
        detail = fallback.split("detailed_description:", 1)[1].split("overall_soundscape:", 1)[0]
        self.assertNotIn("overall_soundscape:", detail)
        self.assertNotIn("non_diegetic_music:", detail)
        self.assertIn("Juno approaches.", detail)

    def test_no_cast_header_speaker_marker_uses_actor_before_speech_verb(self):
        source = (
            "(S1) asks (S2) with a focused look <d>[English] Are we ready?</d>\n"
            "(S2) says in a calm voice to (S3) <d>[English] Almost.</d>\n"
            "(S3) replies <d>[English] Now.</d>"
        )
        entries = llm_service._extract_h3_source_dialogue_entries(source)
        self.assertEqual([entry["speaker_id"] for entry in entries], [1, 2, 3])
        compiled = llm_service._compile_h3_explicit_dialogue(source)
        self.assertIn("to (S3) (S2) <d>", compiled)

    def test_cast_aliases_are_normalized_when_first_vocal_order_differs(self):
        source = CAST_MAP + """Scene: Mira and Tavi stand near the harbor.
(S2) says in a steady voice to (S1) <d>[English] I heard it.</d>
(S1) replies <d>[English] Then we should move.</d>
"""
        entries = llm_service._extract_h3_source_dialogue_entries(source, MOTION_ONLY_CONTEXT)
        self.assertEqual(
            [(entry["speaker"], entry["speaker_id"]) for entry in entries],
            [("Tavi", 1), ("Mira", 2)],
        )
        fallback = llm_service._build_h3_ref2va_tagged_fallback(
            source, MOTION_ONLY_CONTEXT, duration_seconds=8,
        )
        detail = fallback.split("detailed_description:", 1)[1].split("overall_soundscape:", 1)[0]
        self.assertIn("Tavi says in a steady voice to Mira (S1) <d>", detail)
        self.assertNotIn("(S1) Mira", detail)
        self.assertNotIn("(S2) Tavi", detail)
        self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(
            source, fallback, MOTION_ONLY_CONTEXT,
        ))

    def test_rejected_writer_output_returns_valid_fallback_with_each_native_audio_field_once(self):
        source = conversation(native_audio_fields=True)
        with patch.object(llm_service, "generate", return_value="invented <Subject 2> and <Subject 3>") as writer, \
                patch("services.enhance_guides.get_enhance_guide", return_value="Write a six-field H3 reference prompt."), \
                patch("services.studio_enhancement.fidelity_retry_limit", return_value=0):
            result = llm_service.enhance_prompt(
                source,
                mode="video",
                model_type="minimax_h3_ref2va",
                reference_context=MOTION_ONLY_CONTEXT,
                duration_seconds=12,
                planning_style="faithful",
            )

        self.assertEqual(writer.call_count, 1)
        self.assertTrue(llm_service._has_complete_h3_ref2va_structure(result))
        self.assertTrue(llm_service._h3_ref2va_reference_contract_satisfied(result, MOTION_ONLY_CONTEXT))
        self.assertTrue(llm_service._h3_ref2va_dialogue_binding_contract_satisfied(
            source, result, MOTION_ONLY_CONTEXT,
        ))
        self.assertEqual(result.count("overall_soundscape:"), 1)
        self.assertEqual(result.count("non_diegetic_music:"), 1)
        self.assertEqual(result.count("Rain taps the railing; a distant ferry horn fades."), 1)
        self.assertEqual(result.count("A quiet synth pulse rises after the final line."), 1)
        detail = result.split("detailed_description:", 1)[1].split("overall_soundscape:", 1)[0]
        self.assertIn("to (S3) (S2) <d>", detail)

    def test_explicit_music_na_remains_authoritative_over_an_invented_score(self):
        source = (
            "Scene: A person closes a window.\n"
            "overall_soundscape: Rain against glass.\n"
            "non_diegetic_music: N/A\n"
        )
        draft = (
            "overall_soundscape: Rain and a choir.\n"
            "non_diegetic_music: A dramatic orchestral score."
        )
        enforced = llm_service._enforce_h3_music_request(draft, source, None)
        self.assertEqual(enforced, "overall_soundscape: Rain and a choir.\nnon_diegetic_music: N/A")
        preserved_sound = llm_service._enforce_h3_soundscape_silence(
            "overall_soundscape: Model ambience.\nnon_diegetic_music: N/A",
            source + "<d>[English] Close it.</d>",
        )
        self.assertIn("overall_soundscape: Rain against glass.", preserved_sound)

    def test_audio_field_extraction_ignores_headers_inside_protected_dialogue(self):
        source = (
            "Characters: Nia.\n"
            "<d>[English] The sign reads overall_soundscape: leave now.</d>\n"
            "non_diegetic_music: C:\\media\\bells.wav\n"
        )
        cleaned, fields = llm_service._extract_h3_ref2va_audio_sections(source)
        self.assertIn("<d>[English] The sign reads overall_soundscape: leave now.</d>", cleaned)
        self.assertNotIn("overall_soundscape", fields)
        self.assertEqual(fields["non_diegetic_music"], "C:\\media\\bells.wav")
        result = llm_service._enforce_h3_music_request(
            "overall_soundscape: Rain.\nnon_diegetic_music: N/A", source, None,
        )
        self.assertIn("non_diegetic_music: C:\\media\\bells.wav", result)


if __name__ == "__main__":
    unittest.main()
