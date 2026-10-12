"""Regressions for explicit Director generated-audio permissions."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.h3_dialogue import (  # noqa: E402
    H3DialogueContractError,
    compile_h3_clip_plans,
    compile_h3_official_prompt,
    validate_h3_prompt_contract,
)


SILENT_POLICY = {
    "generated_speech_allowed": False,
    "generated_music_allowed": False,
}
SUBJECTS = [
    {"character_id": "kitten", "speaker_name": "Kitten"},
    {"character_id": "dragon", "speaker_name": "Dragon"},
]
MUSIC_VARIANTS = (
    "Non-diegetic music is a soaring, hopeful orchestral theme.",
    "Non-diegetic music swells, matching the forward momentum.",
    "Non-diegetic music swells to a heroic peak.",
    "Non-diegetic music becomes gentle and resolving.",
    "Non-diegetic music resolves into a peaceful, sustained chord.",
)


class DirectorAudioPermissionTests(unittest.TestCase):
    def test_five_planner_music_styles_and_implied_tags_are_removed_idempotently(self):
        for index, music in enumerate(MUSIC_VARIANTS, start=1):
            with self.subTest(clip=index):
                prompt = (
                    "The Kitten turns toward the Dragon and pats its snout. "
                    "The Kitten says: <d>Implied</d>. " + music
                )
                compiled, _ = compile_h3_official_prompt(
                    prompt, SUBJECTS, [], audio_policy=SILENT_POLICY,
                )
                repeated, _ = compile_h3_official_prompt(
                    compiled, SUBJECTS, [], audio_policy=SILENT_POLICY,
                )
                self.assertEqual(repeated, compiled)
                self.assertNotIn("<d>", compiled)
                self.assertNotIn("Implied", compiled)
                self.assertIn("pats its snout", compiled)
                self.assertRegex(compiled, r"(?mi)^non_diegetic_music:\s*N/A$")
                self.assertEqual(
                    validate_h3_prompt_contract(
                        compiled, [], audio_policy=SILENT_POLICY,
                    ),
                    [],
                )

    def test_music_cleanup_preserves_visual_actions_and_practical_sound(self):
        prompt, _ = compile_h3_official_prompt(
            "The Kitten walks while non-diegetic music swells to a heroic peak, "
            "then pats the Dragon's snout. Wind rushes past as claws scrape stone.",
            SUBJECTS,
            [],
            audio_policy=SILENT_POLICY,
            audio_plan={"ambience": "Wind", "effects": ["Claws scrape stone"]},
        )
        self.assertIn("Kitten walks", prompt)
        self.assertIn("pats the Dragon's snout", prompt)
        self.assertIn("Wind", prompt)
        self.assertIn("Claws scrape stone", prompt)
        self.assertNotIn("heroic peak", prompt)

    def test_music_only_soundscape_clause_does_not_eat_adjacent_effects(self):
        prompt, _ = compile_h3_official_prompt(
            "The Kitten turns toward the Dragon.",
            SUBJECTS,
            [],
            audio_policy=SILENT_POLICY,
            audio_plan={
                "ambience": "Wind and a gentle score, claws scrape stone",
            },
        )
        self.assertIn("Wind", prompt)
        self.assertIn("claws scrape stone", prompt)
        self.assertNotIn("score", prompt.casefold())

    def test_music_cleanup_preserves_generic_actions_camera_moves_and_source_sfx(self):
        cases = (
            (
                "Music swells as the courier unlocks the hatch.",
                "courier unlocks the hatch",
            ),
            (
                "Music rises, then the camera pans to the knife.",
                "camera pans to the knife",
            ),
            (
                "Wind rushes while music swells as claws scrape stone.",
                "claws scrape stone",
            ),
        )
        for source, preserved in cases:
            with self.subTest(source=source):
                prompt, _ = compile_h3_official_prompt(
                    source,
                    SUBJECTS,
                    [],
                    audio_policy=SILENT_POLICY,
                )
                self.assertIn(preserved, prompt)
                self.assertNotIn("music swells", prompt.casefold())
                self.assertNotIn("music rises", prompt.casefold())

    def test_silent_policy_distinguishes_visible_text_and_nonverbal_calls(self):
        prompt, _ = compile_h3_official_prompt(
            "A sign says EXIT. The captain calls the elevator. "
            "A bird calls from the trees, and a speech bubble is visible.",
            SUBJECTS,
            [],
            audio_policy={"generated_speech_allowed": False},
        )
        self.assertIn("sign says EXIT", prompt)
        self.assertIn("captain calls the elevator", prompt)
        self.assertIn("bird calls from the trees", prompt)

        with self.assertRaisesRegex(H3DialogueContractError, "generated speech"):
            compile_h3_official_prompt(
                'The Kitten says, "We made it."',
                SUBJECTS,
                [],
                audio_policy={"generated_speech_allowed": False},
            )
        with self.assertRaisesRegex(H3DialogueContractError, "generated speech"):
            compile_h3_official_prompt(
                "The Kitten calls out to the Dragon.",
                SUBJECTS,
                [],
                audio_policy={"generated_speech_allowed": False},
            )

    def test_missing_policy_preserves_legacy_tag_behavior_and_exact_dialogue(self):
        legacy, _ = compile_h3_official_prompt(
            "The Kitten says: <d>Implied</d>.", SUBJECTS, [],
        )
        self.assertIn("<d>[English] Implied</d>", legacy)

        exact, _ = compile_h3_official_prompt(
            "The Kitten faces the Dragon. <d>[English] Stay close.</d>",
            SUBJECTS,
            [{"speaker_id": "kitten", "spoken_text": "Stay close."}],
        )
        self.assertIn("<d>[English] Stay close.</d>", exact)

    def test_empty_authoritative_manifest_does_not_promote_existing_spans(self):
        prompt, _ = compile_h3_official_prompt(
            "The Kitten speaks: <d>Implied</d> while the Dragon lands.",
            SUBJECTS,
            [],
            dialogue_manifest_authoritative=True,
        )
        self.assertNotIn("<d>", prompt)
        self.assertNotIn("Implied", prompt)
        self.assertIn("Dragon lands", prompt)
        self.assertEqual(
            validate_h3_prompt_contract(
                prompt, [], dialogue_manifest_authoritative=True,
            ),
            [],
        )

    def test_driving_audio_reference_is_retained_under_generated_audio_policy(self):
        references = [{
            "type": "audio",
            "path": "song.wav",
            "role": "source song",
            "audio_intent": "drive",
        }]
        prompt, _ = compile_h3_official_prompt(
            "The Kitten performs to the source song. <d>[English] Running free.</d> "
            "Non-diegetic music swells to a heroic peak.",
            SUBJECTS,
            [{"speaker_id": "kitten", "spoken_text": "Running free."}],
            mode="ref2va",
            references=references,
            audio_policy=SILENT_POLICY,
        )
        self.assertNotIn("<d>", prompt)
        self.assertNotIn("heroic peak", prompt)
        self.assertIn("<Audio 1>", prompt)
        self.assertIn("mapped driving audio", prompt)
        self.assertRegex(
            prompt,
            r"(?mi)^non_diegetic_music:\s*Use the mapped driving audio",
        )
        self.assertEqual(
            validate_h3_prompt_contract(
                prompt,
                [],
                mode="ref2va",
                references=references,
                subjects=SUBJECTS,
                audio_policy=SILENT_POLICY,
                has_driving_audio=True,
            ),
            [],
        )

    def test_clip_plan_consumes_policy_and_clears_planner_dialogue_mode(self):
        plan = {
            "video_prompt": "The Kitten says: <d>Implied</d>. The Dragon lands.",
            "_director_audio_policy": {"generated_speech_allowed": False},
            "_director_audio_plan": {
                "mode": "dialogue_driven",
                "timing_anchor": "audio",
                "lip_sync_critical": True,
                "ambience": "Quiet wind",
                "effects": ["Soft landing impact"],
            },
            "_director_dialogue_beats": [{"spoken_text": ""}],
            "_director_duration_sec": 8,
        }
        compile_h3_clip_plans([plan])
        self.assertEqual(plan["_director_dialogue_beats"], [])
        self.assertEqual(plan["_director_audio_plan"]["mode"], "ambient_only")
        self.assertEqual(plan["_director_audio_plan"]["timing_anchor"], "video")
        self.assertFalse(plan["_director_audio_plan"]["lip_sync_critical"])
        self.assertNotIn("<d>", plan["video_prompt"])
        self.assertNotIn("Implied", plan["video_prompt"])
        self.assertIn("Dragon lands", plan["video_prompt"])

    def test_manual_prompt_policy_violation_is_reported_without_rewriting(self):
        source = (
            "The Kitten says: <d>Implied</d>. "
            "Non-diegetic music swells to a heroic peak."
        )
        plan = {
            "video_prompt": source,
            "_director_h3_source_prompt": source,
            "_director_prompt_user_edited": True,
            "_director_audio_policy": copy.deepcopy(SILENT_POLICY),
            "_director_dialogue_beats": [],
            "_director_duration_sec": 8,
        }
        with self.assertRaisesRegex(H3DialogueContractError, "Audio policy violation"):
            compile_h3_clip_plans([plan])
        self.assertEqual(plan["video_prompt"], source)

    def test_negated_speech_and_music_text_is_not_treated_as_a_violation(self):
        prompt, _ = compile_h3_official_prompt(
            "The Kitten closes its eyes. No one speaks and music is absent. "
            "Wind and claws scraping stone remain audible.",
            SUBJECTS,
            [],
            audio_policy=SILENT_POLICY,
            audio_plan={"ambience": "Wind", "effects": ["Claws scraping stone"]},
        )
        self.assertIn("closes its eyes", prompt)
        self.assertIn("Wind", prompt)
        self.assertIn("Claws scraping stone", prompt)
        self.assertRegex(prompt, r"(?mi)^non_diegetic_music:\s*N/A$")
        self.assertEqual(
            validate_h3_prompt_contract(prompt, [], audio_policy=SILENT_POLICY),
            [],
        )

        manual = "The Kitten closes its eyes. No one speaks. Non-diegetic music is absent."
        plan = {
            "video_prompt": manual,
            "_director_h3_source_prompt": manual,
            "_director_prompt_user_edited": True,
            "_director_audio_policy": copy.deepcopy(SILENT_POLICY),
            "_director_dialogue_beats": [],
            "_director_duration_sec": 8,
        }
        compile_h3_clip_plans([plan])
        self.assertRegex(plan["video_prompt"], r"(?mi)^non_diegetic_music:\s*N/A$")


if __name__ == "__main__":
    unittest.main()
