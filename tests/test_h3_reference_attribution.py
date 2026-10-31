"""Focused regressions for Ref2VA prompt/reference identity alignment."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models.minimax_h3.ref2va import (
    align_ref2va_voice_reference_order,
    ensure_ref2va_prompt_relationships,
    select_ref2va_window_voice_references,
)
from models.minimax_h3.speakers import H3SpeakerBindingError
from services.h3_sequence_planner import _reference_context


def character_image(name: str, character_id: str, *, intent: str = "identity"):
    return {
        "type": "image",
        "path": f"{character_id}.png",
        "role": name,
        "character_name": name,
        "library_character_id": character_id,
        "image_intent": intent,
    }


def character_video(name: str, character_id: str):
    return {
        "type": "video",
        "path": f"{character_id}.mp4",
        "role": name,
        "character_name": name,
        "library_character_id": character_id,
        "video_intent": "character",
        "include_audio": False,
    }


def voice(name: str, character_id: str, filename: str):
    return {
        "type": "audio",
        "path": filename,
        "role": name,
        "character_name": name,
        "library_character_id": character_id,
        "audio_intent": "voice",
    }


def conversation_prompt() -> str:
    return "\n".join((
        "subject_definitions: "
        "<Subject 1> is Room from <Picture 1>, preserving the location. "
        "<Subject 2> (S1) is Rhea from <Picture 2>. "
        "<Subject 3> (S2) is Noah from <Video 1>. "
        "<Audio 1> is the voice-timbre reference for <Subject 2> (S1). "
        "<Audio 2> is the voice-timbre reference for <Subject 3> (S2).",
        "summary: Rhea and Noah discuss trusting AI in a room.",
        "retention_analysis: Preserve both character identities and paired voice timbres.",
        "detailed_description: [Shot 1] <Subject 2> (S1) says <d>[English] Trust the answer.</d> "
        "[Shot 2] <Subject 3> (S2) replies <d>[English] That seems risky.</d>",
        "overall_soundscape: Quiet room ambience.",
        "non_diegetic_music: N/A",
    ))


def scene_character_refs(room_intent: str = "scene"):
    return [
        character_image("Room", "room", intent=room_intent),
        character_image("Rhea", "rhea"),
        character_video("Noah", "noah"),
        voice("Rhea", "rhea", "rhea.wav"),
        voice("Noah", "noah", "noah.wav"),
    ]


class H3ReferenceAttributionTests(unittest.TestCase):
    def test_planner_uses_runtime_subject_order_for_all_image_intents(self):
        expected_room = {
            "identity": "<Subject 1> is Room from <Picture 1>",
            "scene": "<Subject 3> is the environment and location for Room",
            "style": "<Subject 3> is the visual treatment for Room",
            "composition": "<Picture 1> is a composition and blocking reference for Room",
            "object": "<Picture 1>",
        }
        for intent, marker in expected_room.items():
            with self.subTest(intent=intent):
                references = [
                    character_image("Room", "room", intent=intent),
                    character_image("Rhea", "rhea"),
                    character_video("Noah", "noah"),
                    voice("Rhea", "rhea", "rhea.wav"),
                    voice("Noah", "noah", "noah.wav"),
                ]
                relationships, _retention, _task_types = _reference_context(references)
                self.assertIn(marker, relationships)
                if intent == "composition":
                    self.assertNotRegex(relationships, r"<Subject \d+> is Room")
                if intent == "object":
                    self.assertNotRegex(relationships, r"<Subject \d+> is Room")
                noah_subject = "<Subject 3>" if intent == "identity" else "<Subject 2>"
                self.assertIn(
                    f"{noah_subject} is Noah from <Video 1>",
                    relationships,
                )

    def test_context_tracks_shared_character_media_and_voice_ordinals_after_permutation(self):
        references = [
            voice("Noah", "noah", "noah.wav"),
            character_image("Room", "room", intent="scene"),
            character_video("Rhea", "rhea"),
            {"type": "audio", "path": "style.wav", "role": "room tone", "audio_intent": "style"},
            character_image("Rhea", "rhea"),
            voice("Rhea", "rhea", "rhea.wav"),
            character_video("Noah", "noah"),
        ]
        relationships, _retention, _task_types = _reference_context(references)
        self.assertIn("<Subject 1> is Rhea from <Picture 2>", relationships)
        self.assertIn("<Subject 1> is Rhea from <Video 1>", relationships)
        self.assertIn("<Subject 2> is Noah from <Video 2>", relationships)
        self.assertIn("<Subject 3> is the environment and location for Room", relationships)
        self.assertIn("<Audio 1> is the voice-timbre reference for <Subject 2>", relationships)
        self.assertIn("<Audio 2> supplies sound, music, rhythm, or texture style for room tone", relationships)
        self.assertIn("<Audio 3> is the voice-timbre reference for <Subject 1>", relationships)

    def test_legacy_scene_and_composition_prompt_remaps_before_voice_pruning(self):
        for room_intent, room_subject in (("scene", "<Subject 3>"), ("composition", "")):
            with self.subTest(room_intent=room_intent):
                prompt, scoped, diagnostics = select_ref2va_window_voice_references(
                    conversation_prompt(),
                    scene_character_refs(room_intent),
                )
                self.assertIn("<Subject 1> (S1)", prompt)
                self.assertIn("<Subject 2> (S2)", prompt)
                self.assertEqual(diagnostics["speaking_subjects"], [1, 2])
                self.assertEqual(
                    {item["path"] for item in scoped if item.get("audio_intent") == "voice"},
                    {"rhea.wav", "noah.wav"},
                )
                if room_subject:
                    self.assertIn("<Subject 3> is Room", prompt)
                else:
                    self.assertNotRegex(prompt, r"<Subject \d+> is Room")
                    self.assertNotRegex(prompt.split("detailed_description:", 1)[1], r"<Subject 3>")

                compiled = ensure_ref2va_prompt_relationships(
                    prompt, scoped,
                )
                self.assertIn("<Subject 1> (S1)", compiled)
                self.assertIn("<Subject 2> (S2)", compiled)
                self.assertEqual(compiled.count("Trust the answer."), 1)
                self.assertEqual(compiled.count("That seems risky."), 1)

    def test_voice_audio_labels_follow_actual_character_media(self):
        references = [
            character_image("Rhea", "rhea"),
            character_video("Noah", "noah"),
            voice("Noah", "noah", "noah.wav"),
            {"type": "audio", "path": "texture.wav", "role": "texture", "audio_intent": "style"},
            voice("Rhea", "rhea", "rhea.wav"),
        ]
        prompt = (
            "subject_definitions: <Subject 2> is Noah from <Video 1>. "
            "<Subject 1> is Rhea from <Picture 1>. "
            "<Audio 1> is the voice-timbre reference for <Subject 2>. "
            "<Audio 3> is the voice-timbre reference for <Subject 1>.\n"
            "detailed_description: <Subject 1> says <d>Hello.</d>"
        )
        aligned, _items, maps = align_ref2va_voice_reference_order(prompt, references)
        self.assertEqual(maps.get("Audio"), None)
        self.assertIn("<Audio 1> is the voice-timbre reference for <Subject 2>", aligned)
        self.assertIn("<Audio 3> is the voice-timbre reference for <Subject 1>", aligned)

        permuted = [
            *references[:2],
            voice("Rhea", "rhea", "rhea.wav"),
            voice("Noah", "noah", "noah.wav"),
        ]
        permuted_prompt = prompt.replace("<Audio 1>", "<Audio TEMP>").replace(
            "<Audio 3>", "<Audio 1>",
        ).replace("<Audio TEMP>", "<Audio 2>")
        aligned, _items, _maps = align_ref2va_voice_reference_order(
            permuted_prompt, permuted,
        )
        self.assertIn("<Audio 2> is the voice-timbre reference for <Subject 2>", aligned)
        self.assertIn("<Audio 1> is the voice-timbre reference for <Subject 1>", aligned)

    def test_conflicting_or_ambiguous_subject_definitions_fail_early(self):
        references = [
            character_image("Alex", "alex"),
            character_image("Sam", "sam"),
        ]
        conflicting = (
            "subject_definitions: <Subject 2> is Alex from <Picture 2>.\n"
            "detailed_description: <Subject 2> says <d>Hello.</d>"
        )
        with self.assertRaisesRegex(H3SpeakerBindingError, "conflict"):
            align_ref2va_voice_reference_order(conflicting, references)

        ambiguous_refs = [
            character_image("Alex", "alex-one"),
            character_image("Alex", "alex-two"),
        ]
        ambiguous = "subject_definitions: <Subject 1> is Alex."
        with self.assertRaisesRegex(H3SpeakerBindingError, "ambiguous"):
            align_ref2va_voice_reference_order(ambiguous, ambiguous_refs)

    def test_wrong_name_same_number_is_rejected_without_explicit_definitions(self):
        references = [
            character_image("Rhea", "rhea"),
            character_image("Noah", "noah"),
        ]
        stale = (
            "detailed_description: <Subject 2> Rhea says "
            "<d>[English] Trust the answer.</d>"
        )
        with self.assertRaisesRegex(H3SpeakerBindingError, "conflict"):
            align_ref2va_voice_reference_order(stale, references)

        explained = (
            "subject_definitions: <Subject 2> is Rhea from <Picture 1>. "
            "<Subject 1> is Noah from <Picture 2>.\n"
            "detailed_description: <Subject 2> Rhea says "
            "<d>[English] Trust the answer.</d>"
        )
        aligned, _items, maps = align_ref2va_voice_reference_order(explained, references)
        self.assertEqual(maps["Subject"], {2: 1, 1: 2})
        self.assertIn("<Subject 1> Rhea says", aligned)

    def test_shot_heading_does_not_hide_a_wrong_named_subject_cue(self):
        references = [
            character_image("Rhea", "rhea"),
            character_image("Noah", "noah"),
        ]
        prompt = (
            "subject_definitions: <Subject 1> is Rhea from <Picture 1>. "
            "<Subject 2> is Noah from <Picture 2>.\n"
            "summary: Rhea and Noah have a conversation.\n"
            "detailed_description: [Shot 1] The camera settles into a close-up. "
            "Rhea <Subject 2> says <d>[English] Hello.</d>"
        )
        with self.assertRaisesRegex(H3SpeakerBindingError, "conflict"):
            align_ref2va_voice_reference_order(prompt, references)

    def test_abbreviated_name_keeps_explicit_picture_evidence(self):
        references = [character_image("Dr. Rhea Smith", "rhea")]
        prompt = (
            "subject_definitions: <Subject 4> is Dr. Rhea Smith from <Picture 1>.\n"
            "detailed_description: Rhea arrives."
        )
        aligned, _items, maps = align_ref2va_voice_reference_order(prompt, references)
        self.assertEqual(maps["Subject"], {4: 1})
        self.assertIn("<Subject 1> is Dr. Rhea Smith from <Picture 1>", aligned)

    def test_speaker_marker_alone_never_remaps_a_subject(self):
        references = [
            character_image("Rhea", "rhea"),
            character_image("Noah", "noah"),
        ]
        prompt = (
            "subject_definitions: <Subject 2> (S1) is the first speaker.\n"
            "detailed_description: <Subject 2> (S1) says <d>Hello.</d>"
        )
        aligned, _items, maps = align_ref2va_voice_reference_order(prompt, references)
        self.assertEqual(maps.get("Subject"), None)
        self.assertIn("<Subject 2> (S1)", aligned)


if __name__ == "__main__":
    unittest.main()
