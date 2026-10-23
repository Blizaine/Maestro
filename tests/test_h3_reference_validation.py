"""Focused CPU tests for H3 reference manifests and rejection explanations."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services import llm_service


NATIVE_REFERENCE_CONTEXT = "\n".join((
    "<Subject 1> is Avery from <Picture 1>, preserving identity and visible appearance.",
    "<Picture 2> is a composition and blocking reference for the room, not an identity source.",
    "<Subject 2> is the environment and location for the room, defined by <Picture 3>.",
    "<Subject 3> is the visual treatment for the scene, guided by <Picture 4>.",
    "<Video 1> provides motion, camera, scene, or timing reference for the requested movement.",
    "<Audio 1> is the soundtrack paired with <Video 1> and keeps its audible timeline.",
    "<Subject 1>: fully_preserved - preserve Avery's identity.",
    "<Picture 2>: weak_reference",
    "<Subject 2>: partially_preserved - preserve the room.",
    "<Subject 3>: weak_reference - retain broad visual treatment.",
    "<Video 1>: partially_preserved",
    "<Audio 1>: partially_copy",
))


class H3ReferenceValidationTests(unittest.TestCase):
    def test_retention_rows_do_not_create_phantom_character_subjects(self):
        manifest = llm_service._parse_h3_ref2va_subject_manifest(
            NATIVE_REFERENCE_CONTEXT
        )
        self.assertEqual([subject["index"] for subject in manifest], [1])
        self.assertEqual(manifest[0]["name"], "Avery")
        self.assertEqual(manifest[0]["pictures"], ["<Picture 1>"])
        self.assertEqual(manifest[0]["videos"], [])
        self.assertEqual(manifest[0]["audios"], [])

    def test_typed_scene_style_composition_and_motion_roles_remain_valid(self):
        candidate = "\n".join((
            "subject_definitions: <Subject 1> Avery from <Picture 1>; "
            "<Subject 2> is the environment and location for the room from <Picture 3>; "
            "<Subject 3> is the visual treatment from <Picture 4>.",
            "summary: A performer follows the requested movement.",
            "retention_analysis: <Subject 1>: fully_preserved; "
            "<Subject 2>: partially_preserved; <Subject 3>: weak_reference.",
            "detailed_description: Use <Picture 2> for composition and <Video 1> for motion.",
            "overall_soundscape: N/A",
            "non_diegetic_music: N/A",
        ))
        self.assertEqual(
            llm_service._h3_ref2va_reference_contract_issues(
                candidate, NATIVE_REFERENCE_CONTEXT
            ),
            [],
        )
        self.assertTrue(llm_service._h3_ref2va_reference_contract_satisfied(
            candidate, NATIVE_REFERENCE_CONTEXT
        ))

    def test_contract_reasons_identify_missing_invented_and_unresolved_refs(self):
        context = (
            'Saved character "Avery" is exactly <Subject 1>: <Picture 1> defines Avery.\n'
            "<Picture 1>: visual identity/appearance reference for Avery; retention=reference"
        )
        candidate = "\n".join((
            "subject_definitions: <Subject 3> is an invented person from <Video 9>.",
            "summary: A scene.",
            "retention_analysis: N/A",
            "detailed_description: <Subject N> appears.",
            "overall_soundscape: N/A",
            "non_diegetic_music: N/A",
        ))
        issues = llm_service._h3_ref2va_reference_contract_issues(candidate, context)
        self.assertTrue(any("invented reference label(s)" in issue and "<Video 9>" in issue for issue in issues))
        self.assertIn("unresolved placeholder reference <Subject N>", issues)
        self.assertIn("missing reference Subject(s): <Subject 1>", issues)
        self.assertIn("invented reference Subject(s): <Subject 3>", issues)
        self.assertTrue(any("missing <Picture 1> binding" in issue for issue in issues))
        self.assertFalse(llm_service._h3_ref2va_reference_contract_satisfied(candidate, context))

    def test_duplicate_context_ownership_is_reported_as_ambiguous(self):
        context = "\n".join((
            "<Subject 1> is Avery from <Picture 1>, preserving identity.",
            "<Subject 2> is Riley from <Picture 1>, preserving identity.",
        ))
        candidate = "\n".join((
            "subject_definitions: <Subject 1> Avery <Picture 1>; <Subject 2> Riley <Picture 1>.",
            "summary: A scene.",
            "retention_analysis: N/A",
            "detailed_description: Two people.",
            "overall_soundscape: N/A",
            "non_diegetic_music: N/A",
        ))
        issues = llm_service._h3_ref2va_reference_contract_issues(candidate, context)
        self.assertTrue(any("ambiguous reference binding <picture 1>" in issue for issue in issues))

    def test_structured_rejection_reasons_are_forwarded_with_their_stage(self):
        reasons = llm_service._h3_structured_rejection_reasons(
            "summary: draft",
            is_h3_ref2va=True,
            structure_valid=False,
            dialogue_valid=True,
            timed_silence_valid=True,
            reference_issues=["missing <Picture 1> binding from <Subject 1> definitions"],
            dialogue_binding_valid=True,
        )
        self.assertIn("missing required field subject_definitions:", reasons)
        self.assertIn("missing <Picture 1> binding from <Subject 1> definitions", reasons)
        with patch("services.enhancement_diagnostics.record_rejection") as record:
            llm_service._record_h3_enhancement_rejection(reasons, stage="repair_1")
        record.assert_called_once_with(reasons, stage="repair_1")

    def test_structural_rejection_has_fallback_reason_when_all_field_labels_exist(self):
        malformed = "\n".join((
            "subject_definitions:",
            "summary: A scene.",
            "retention_analysis: N/A",
            "detailed_description: A performer crosses the room.",
            "overall_soundscape: N/A",
            "non_diegetic_music: N/A",
        ))
        reasons = llm_service._h3_structured_rejection_reasons(
            malformed,
            is_h3_ref2va=True,
            structure_valid=False,
            dialogue_valid=True,
            timed_silence_valid=True,
            reference_issues=[],
            dialogue_binding_valid=True,
        )
        self.assertTrue(any("invalid structured field shape" in reason for reason in reasons))

    def test_speechless_fallback_keeps_motion_and_composition_without_audio_filler(self):
        context = "\n".join((
            "<Subject 1> is Avery from <Picture 1>, preserving identity and visible appearance.",
            "<Picture 2> is a composition and blocking reference for the doorway, not an identity source.",
            "<Video 1> provides motion, camera, scene, or timing reference for the movement.",
            "<Audio 1> is the soundtrack paired with <Video 1> and keeps its audible timeline.",
            "<Subject 1>: fully_preserved - preserve Avery's identity.",
            "<Picture 2>: weak_reference",
            "<Video 1>: partially_preserved",
            "<Audio 1>: partially_copy",
        ))
        fallback = llm_service._build_h3_ref2va_tagged_fallback(
            "A performer crosses the room.", context
        )
        self.assertIn("Use <Video 1> as the requested motion", fallback)
        self.assertIn("Use <Picture 2> for the requested composition and blocking", fallback)
        self.assertNotIn("room..", fallback)
        self.assertNotIn("dialogue", fallback.casefold())
        self.assertNotIn("voice", fallback.casefold())
        self.assertNotIn("ambience", fallback.casefold())
        self.assertNotIn("dialogue", fallback.casefold())
        self.assertIn("overall_soundscape: Use <Audio 1> as a reference", fallback)
        self.assertIn(
            "Preserve the supplied soundtrack's vocal/nonvocal content",
            fallback,
        )
        self.assertIn("do not add new spoken words or speech-like vocalizations", fallback)
        self.assertNotIn("No spoken words or speech-like vocalizations", fallback)

    def test_no_paired_audio_prevents_new_speech_but_keeps_requested_nonverbal_reactions(self):
        context = "\n".join((
            "<Subject 1> is Avery from <Picture 1>, preserving identity and visible appearance.",
            "<Subject 1>: fully_preserved - preserve Avery's identity.",
        ))
        fallback = llm_service._build_h3_ref2va_tagged_fallback(
            "Avery crosses the room and smiles.", context
        )
        self.assertIn("Do not add new spoken words or speech-like vocalizations", fallback)
        self.assertIn("Preserve explicitly requested nonverbal reactions", fallback)
        self.assertIn("overall_soundscape: N/A", fallback)


if __name__ == "__main__":
    unittest.main()
