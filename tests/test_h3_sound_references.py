"""Sound-effect references remain reusable conditioning, never voices or tracks."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.minimax_h3.reference_manifest import (
    split_exact_drive_audio_reference,
    validate_reference_manifest,
)
from models.minimax_h3.ref2va import (
    ensure_ref2va_prompt_relationships,
    select_ref2va_window_voice_references,
)
from services.director.h3_dialogue import compile_h3_official_prompt
from services.h3_performance_audio import has_h3_performance_audio
from services.h3_sequence_planner import (
    _reference_context,
    build_manual_h3_reference_sequence_plan,
)
from services.llm_service import (
    _canonical_h3_ref2va_subject_fields,
    _parse_h3_ref2va_subject_manifest,
)


class H3SoundReferenceTests(unittest.TestCase):
    def setUp(self):
        self.references = [
            {"type": "video", "path": "running.mp4", "include_audio": False},
            {"type": "image", "path": "chewy.png", "role": "Chewy"},
            {
                "type": "audio", "path": "voice-blaster.wav",
                "role": "Laser blaster sound effect", "audio_intent": "sound",
                "duration_seconds": 3.226,
            },
        ]

    def test_manifest_accepts_sound_and_retains_reference_limits(self):
        normalized = validate_reference_manifest(self.references, require_files=False)
        self.assertEqual(normalized[-1]["audio_intent"], "sound")
        three = [*self.references, dict(self.references[-1]), dict(self.references[-1])]
        self.assertEqual(len(validate_reference_manifest(three, require_files=False)), 5)
        with self.assertRaisesRegex(ValueError, "at most 3 audio references"):
            validate_reference_manifest([*three, dict(self.references[-1])], require_files=False)
        with self.assertRaisesRegex(ValueError, "add at least one image or video"):
            validate_reference_manifest([self.references[-1]], require_files=False)

    def test_sound_stays_a_reference_beside_an_exact_target_track(self):
        track = {"type": "audio", "path": "song.wav", "audio_intent": "drive"}
        references, path, ordinal = split_exact_drive_audio_reference([
            *self.references[:2], track, self.references[-1],
        ])
        self.assertEqual(path, "song.wav")
        self.assertEqual(ordinal, 1)
        self.assertEqual(references[-1]["audio_intent"], "sound")

    def test_queued_enhancement_describes_effects_without_a_voice_or_duration_clock(self):
        relationships, retention, task_types = _reference_context(self.references)
        self.assertIn("<Audio 1> is a reusable sound-effect reference", relationships)
        self.assertIn("each window", relationships)
        self.assertIn("<Audio 1>: reference", retention)
        self.assertIn("audio reference", task_types)
        self.assertNotIn("voice-timbre", relationships)
        self.assertNotIn("audio reuse", task_types)
        self.assertNotIn("H3_PERFORMANCE_AUDIO_CLOCK", relationships)
        self.assertFalse(has_h3_performance_audio(relationships))
        definitions, retained = _canonical_h3_ref2va_subject_fields(relationships + "\n" + retention)
        self.assertIn("reusable sound-effect reference", definitions)
        self.assertIn("<Audio 1>: reference", retained)

    def test_raw_request_compiles_effect_semantics_without_unrequested_music(self):
        prompt = ensure_ref2va_prompt_relationships(
            "Chewy runs while stormtroopers fire lasers with blaster sound effects.",
            self.references,
        )
        self.assertIn("<Audio 1> is a reusable sound-effect reference", prompt)
        self.assertIn("audio reference", prompt)
        self.assertIn("non_diegetic_music: N/A", prompt)
        self.assertNotIn("voice-timbre", prompt)

    def test_tagged_manual_windows_keep_sample_guidance_and_preserve_audio_ordinals(self):
        references = [self.references[-1], {
            **self.references[0], "has_audio": True, "include_audio": True,
        }, self.references[1]]
        prompt = ensure_ref2va_prompt_relationships(
            "Chewy runs in <Video 1>. Use <Audio 1> for the firing blasters.", references,
        )
        self.assertIn("Use <Audio 2> for the firing blasters", prompt)
        self.assertIn("timbre and texture of <Audio 2>", prompt)
        canonical = [references[1], references[2], references[0]]
        self.assertEqual(ensure_ref2va_prompt_relationships(prompt, canonical), prompt)

    def test_manual_three_window_run_keeps_effect_guidance_in_every_window(self):
        window = "Chewy runs in <Video 1> while blasters fire. Use <Audio 1> for their sound."
        result = build_manual_h3_reference_sequence_plan(
            "\n".join([window] * 3), model_type="minimax_h3_ref2va_fused_turbo",
            resolution="704x1280", total_frames=835, references=self.references,
            max_clip_frames=294, overlap_frames=18, native_continuation=True,
        )
        self.assertEqual(result["window_count"], 3)
        for window_prompt in result["window_prompts"]:
            prepared = ensure_ref2va_prompt_relationships(window_prompt, self.references)
            self.assertIn("timbre and texture of <Audio 1>", prepared)
            self.assertNotIn("voice-timbre", prepared)

    def test_enhance_now_inventory_never_binds_effect_audio_as_a_character_voice(self):
        inventory = (
            "<Picture 1>: visual identity/appearance reference for Chewy; retention=reference\n"
            "<Audio 1>: voice-blaster.wav; intent=SOUND EFFECT REFERENCE; retention=reference\n"
        )
        manifest = _parse_h3_ref2va_subject_manifest(inventory)
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["audios"], [])
        definitions, retention = _canonical_h3_ref2va_subject_fields(inventory)
        self.assertIn("<Audio 1> is a reusable sound-effect reference", definitions)
        self.assertIn("<Audio 1>: reference", retention)
        self.assertNotIn("voice-timbre", definitions)

    def test_silent_window_keeps_effect_when_unused_character_voice_is_removed(self):
        references = [
            self.references[1],
            {"type": "audio", "path": "chewy.wav", "role": "Chewy", "audio_intent": "voice"},
            self.references[-1],
        ]
        prompt, scoped, _ = select_ref2va_window_voice_references(
            "Chewy runs silently while lasers fire with sound from <Audio 2>.", references,
        )
        self.assertEqual([item.get("audio_intent") for item in scoped if item["type"] == "audio"], ["sound"])
        self.assertIn("sound from <Audio 1>", prompt)

    def test_director_compiles_sound_as_effect_guidance_without_driving_performance(self):
        prompt, _ = compile_h3_official_prompt(
            "Chewy runs while stormtroopers fire lasers with sound from <Audio 1>.",
            [{"character_id": "chewy", "speaker_name": "Chewy"}], [],
            mode="ref2va", duration_seconds=12.25, references=self.references,
        )
        self.assertIn("<Audio 1> is a reusable sound-effect reference", prompt)
        self.assertIn("requested matching sound effects", prompt)
        self.assertNotIn("voice-timbre", prompt)
        self.assertNotIn("performance-driving audio", prompt)


if __name__ == "__main__":
    unittest.main()
