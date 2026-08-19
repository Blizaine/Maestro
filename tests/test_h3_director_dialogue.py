"""Regressions for Director's MiniMax H3 native-dialogue contract."""

from __future__ import annotations

import os
import json
import sys
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.h3_dialogue import (  # noqa: E402
    H3DialogueContractError,
    compile_h3_clip_plans,
    compile_h3_official_prompt,
    compile_h3_vocal_contract,
    h3_dialogue_budget_violations,
    validate_h3_prompt_contract,
    validate_h3_vocal_contract,
)
from services.director.planners.short_film import (  # noqa: E402
    ShortFilmPlanner,
    _apply_h3_character_table_read,
    _coalesce_h3_dialogue_shots,
    _complete_h3_truncated_tail,
    _enforce_h3_speaker_visual_contract,
    _extract_h3_screenplay_dialogue,
    _fit_bounded_frame_schedule,
    _h3_native_structure_issues,
    _h3_planner_token_budget,
    _h3_preferred_native_durations,
    _normalize_h3_voice_bible,
    _reconcile_h3_dialogue_manifest,
    _restore_h3_dialogue_after_pacing_repair,
)


class TestH3DirectorDialogueCompiler(unittest.TestCase):
    def setUp(self):
        self.beats = [
            {
                "speaker_id": "joey",
                "spoken_text": (
                    "So, when Chandler tried to cook that thing. the casserole? "
                    "It was like, a crime scene. Just brown mush everywhere."
                ),
                "delivery": "casual and exaggerated",
            },
            {
                "speaker_id": "monica",
                "spoken_text": (
                    "It was not mush, Joey. It was rustic shepherd pie."
                ),
                "delivery": "sharp and concerned",
            },
        ]
        self.subjects = [
            {"character_id": "joey", "speaker_name": "Joey"},
            {"character_id": "monica", "speaker_name": "Monica"},
        ]

    def test_repairs_the_nested_mid_sentence_failure_from_director(self):
        broken = (
            "Joey speaks: <d><d>[English] So, when Chandler tried to cook "
            "that thing. the casserole?</d> It was like, a crime scene. Just "
            "brown mush everywhere.</d> Monica speaks: <d><d>[English] It "
            "was not mush, Joey.</d> It was rustic shepherd pie.</d> "
            "overall_soundscape: Coffee shop chatter, cups, and an espresso "
            "machine. non_diegetic_music: N/A."
        )

        repaired, _ = compile_h3_vocal_contract(
            broken, self.subjects, self.beats,
        )

        self.assertNotIn("<d><d>", repaired)
        self.assertEqual(repaired.count("<d>"), 2)
        self.assertIn(
            "<d>[English] " + self.beats[0]["spoken_text"] + "</d>",
            repaired,
        )
        self.assertIn(
            "<d>[English] " + self.beats[1]["spoken_text"] + "</d>",
            repaired,
        )
        self.assertNotIn("coffee shop chatter", repaired.lower())
        self.assertIn("No background or crowd voices are audible", repaired)
        self.assertEqual(validate_h3_vocal_contract(repaired, self.beats), [])
        self.assertEqual(
            compile_h3_vocal_contract(repaired, self.subjects, self.beats)[0],
            repaired,
        )

    def test_repairs_balanced_nested_saved_prompt_without_metadata(self):
        broken = (
            "A speaker says <d><d>[English] This entire sentence</d> must "
            "remain intact.</d> overall_soundscape: Quiet room tone."
        )

        repaired, _ = compile_h3_vocal_contract(broken, [], [])

        self.assertIn(
            "<d>[English] This entire sentence must remain intact.</d>",
            repaired,
        )
        self.assertEqual(repaired.count("<d>"), 1)
        self.assertEqual(validate_h3_vocal_contract(repaired), [])

    def test_saved_vocal_section_keeps_its_only_dialogue_line(self):
        saved = (
            "Monica glares across the table. DIALOGUE AND VOCAL PERFORMANCE: "
            "(S1) Monica speaks fiercely: <d>[English] Ross. You did this.</d>. "
            "Only this explicitly tagged line is spoken."
        )

        repaired, _ = compile_h3_vocal_contract(saved, [], [])

        self.assertIn("(S1) Monica speaks fiercely", repaired)
        self.assertIn("<d>[English] Ross. You did this.</d>", repaired)
        self.assertEqual(repaired.count("<d>"), 1)
        self.assertEqual(validate_h3_vocal_contract(repaired), [])

    def test_plain_dialogue_is_wrapped_without_touching_other_text(self):
        prompt = (
            "Joey turns and says: So, when Chandler tried to cook that thing. "
            "the casserole? It was like, a crime scene. Just brown mush "
            "everywhere. overall_soundscape: Cup clinks."
        )

        compiled, _ = compile_h3_vocal_contract(
            prompt, self.subjects, self.beats[:1],
        )

        self.assertEqual(compiled.count("<d>"), 1)
        self.assertNotIn("<d><d>", compiled)
        self.assertEqual(validate_h3_vocal_contract(compiled, self.beats[:1]), [])

    def test_unbalanced_dialogue_is_rejected_before_generation(self):
        with self.assertRaises(H3DialogueContractError):
            compile_h3_vocal_contract(
                "Joey says <d>[English] This never closes.",
                self.subjects,
                self.beats[:1],
            )

    def test_music_driven_prompt_discards_unbalanced_generated_dialogue(self):
        compiled, contract = compile_h3_official_prompt(
            (
                "A singer performs directly to camera while the band plays. "
                "The singer begins <d>[English] beep beep beep beep beep"
            ),
            [],
            [],
            mode="t2va",
            audio_plan={"mode": "music_driven", "timing_anchor": "audio"},
        )

        self.assertNotIn("<d>", compiled)
        self.assertNotIn("beep", compiled.lower())
        self.assertIn("mapped driving audio", compiled)
        self.assertEqual(contract.count("mapped driving audio"), 1)
        self.assertEqual(validate_h3_prompt_contract(compiled, mode="t2va"), [])

    def test_music_driven_prompt_discards_balanced_generated_dialogue(self):
        compiled, _ = compile_h3_official_prompt(
            (
                "A singer performs directly to camera and improvises "
                "<d>[English] words invented by the planner</d>."
            ),
            [],
            [],
            mode="t2va",
            audio_plan={"mode": "music_driven"},
        )

        self.assertNotIn("invented by the planner", compiled)
        self.assertNotIn("<d>", compiled)
        self.assertIn("mapped driving audio", compiled)
        self.assertEqual(validate_h3_prompt_contract(compiled, mode="t2va"), [])

    def test_music_driven_prompt_repairs_inline_duplicate_context_fields(self):
        compiled, _ = compile_h3_official_prompt(
            (
                "A singer runs across a moving train while lip-syncing to the "
                "supplied track. Overall soundscape features train wheels and "
                "wind; non-diegetic music is driving rock. "
                "Integrated multimodal description: [Shot 1] The singer runs "
                "toward camera. Overall soundscape: Train wheels and wind. "
                "Non-diegetic music: Driving rock. Dialogue beats: "
                "<d>Planner-invented words</d>."
            ),
            [],
            [],
            mode="t2va",
            audio_plan={"mode": "music_driven", "timing_anchor": "audio"},
        )

        self.assertNotIn("Planner-invented words", compiled)
        self.assertNotIn("<d>", compiled)
        self.assertEqual(compiled.count("integrated_multimodal_description:"), 1)
        self.assertEqual(compiled.count("overall_soundscape:"), 1)
        self.assertEqual(compiled.count("non_diegetic_music:"), 1)
        self.assertIn("mapped driving audio", compiled)
        self.assertEqual(validate_h3_prompt_contract(compiled, mode="t2va"), [])

    def test_silent_shot_gets_an_explicit_silence_contract(self):
        compiled, _ = compile_h3_vocal_contract(
            "A silent reaction. overall_soundscape: Air conditioning hum.",
            [],
            [],
        )

        self.assertIn("SILENCE AND VOCAL PERFORMANCE", compiled)
        self.assertIn("No one speaks in this shot", compiled)
        self.assertEqual(validate_h3_vocal_contract(compiled), [])

    def test_final_clip_preflight_uses_structured_dialogue(self):
        plans = [{
            "video_prompt": "Joey leans forward. overall_soundscape: Cafe chatter.",
            "_director_subjects_on_screen": self.subjects,
            "_director_dialogue_beats": self.beats[:1],
        }]

        compile_h3_clip_plans(plans)

        self.assertIn(self.beats[0]["spoken_text"], plans[0]["video_prompt"])
        self.assertEqual(plans[0]["video_prompt"].count("<d>"), 1)
        self.assertNotIn("cafe chatter", plans[0]["video_prompt"].lower())

    def test_base_compiler_uses_exact_official_fields_and_repairs_mojibake(self):
        plans = [{
            "video_prompt": (
                "PROJECT CONTINUITY (visual world only): duplicated wrapper. "
                "MiniMax H3 generates synchronized picture and stereo sound. "
                "Monica speaks: <d>[English] You\u00e2\u0080\u0099re late.</d>. "
                "The soundscape is quiet cafe chatter and cup clinks. "
                "Non-diegetic music is N/A. OPENING CONTINUITY: duplicate. "
                "FINAL BLOCKING: duplicate. DIALOGUE AND VOCAL PERFORMANCE: "
                "Only these lines are spoken."
            ),
            "_director_subjects_on_screen": [
                {"character_id": "monica", "speaker_name": "Monica"},
            ],
            "_director_dialogue_beats": [{
                "speaker_id": "monica",
                # Observed in an older saved Director project after a Windows
                # code-page round-trip: the lead UTF-8 byte became U+0101.
                "spoken_text": "You\u0101\u0080\u0099re late.",
            }],
            "_director_h3_prompt_mode": "t2va",
        }]

        compile_h3_clip_plans(plans)
        prompt = plans[0]["video_prompt"]

        self.assertTrue(prompt.startswith("integrated_multimodal_description: [Shot 1]"))
        self.assertEqual(prompt.count("integrated_multimodal_description:"), 1)
        self.assertEqual(prompt.count("overall_soundscape:"), 1)
        self.assertEqual(prompt.count("non_diegetic_music:"), 1)
        self.assertNotIn("PROJECT CONTINUITY", prompt)
        self.assertNotIn("OPENING CONTINUITY", prompt)
        self.assertNotIn("FINAL BLOCKING", prompt)
        self.assertIn("Monica (S1) speaks", prompt)
        self.assertIn("<d>[English] You\u2019re late.</d>", prompt)
        self.assertNotIn("cafe chatter", prompt.lower())
        self.assertFalse(any(0x80 <= ord(character) <= 0x9F for character in prompt))
        self.assertEqual(
            validate_h3_prompt_contract(
                prompt,
                plans[0]["_director_dialogue_beats"],
                mode="t2va",
            ),
            [],
        )

    def test_base_alignment_headers_follow_actual_frame_conditioning(self):
        source = "A person turns toward camera. overall_soundscape: Room tone."
        i2va, _ = compile_h3_official_prompt(
            source, [], [], mode="i2va", duration_seconds=8.0,
        )
        fl2va, _ = compile_h3_official_prompt(
            source, [], [], mode="fl2va", duration_seconds=10.25,
        )
        l2va, _ = compile_h3_official_prompt(
            source, [], [], mode="l2va", duration_seconds=9.5,
        )

        self.assertTrue(i2va.startswith(
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        ))
        self.assertTrue(fl2va.startswith(
            "How the reference pictures align with the target video"
        ))
        self.assertIn("10.25-second mark", fl2va)
        self.assertTrue(l2va.startswith(
            "How the reference pictures align with the target video — "
            "<Picture 1> (from [Shot 1]) aligns with the 9.50-second mark"
        ))
        self.assertEqual(validate_h3_prompt_contract(i2va, mode="i2va"), [])
        self.assertEqual(validate_h3_prompt_contract(fl2va, mode="fl2va"), [])
        self.assertEqual(validate_h3_prompt_contract(l2va, mode="l2va"), [])

    def test_ref2va_compiler_emits_six_fields_and_maps_real_manifest_order(self):
        references = [
            {
                "type": "image",
                "role": "the identity and appearance of Monica",
                "image_intent": "identity",
            },
            {
                "type": "image",
                "role": "the Central Perk environment",
                "image_intent": "scene",
            },
            {
                "type": "audio",
                "role": "the voice of Monica",
                "audio_intent": "voice",
            },
        ]
        prompt, _ = compile_h3_official_prompt(
            "Monica speaks: <d>[English] We need to talk.</d>. "
            "overall_soundscape: Espresso machine hum. non_diegetic_music: N/A.",
            [{
                "character_id": "monica",
                "speaker_name": "Monica",
                "visual_description": "Monica from Friends",
                "wardrobe": "red top and dark trousers",
            }],
            [{"speaker_id": "monica", "spoken_text": "We need to talk."}],
            mode="ref2va",
            references=references,
            speaker_registry={
                "monica": {"stable_id": "(S1)", "speaker_name": "Monica"},
            },
        )

        positions = [
            prompt.index(f"{field}:")
            for field in (
                "subject_definitions", "summary", "retention_analysis",
                "detailed_description", "overall_soundscape",
                "non_diegetic_music",
            )
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Picture 2>", prompt)
        self.assertIn("<Audio 1>", prompt)
        self.assertIn("<Subject 1> (Monica)", prompt)
        self.assertIn("identity only", prompt)
        self.assertIn("not copy its background", prompt)
        self.assertIn(
            "summary: [reference generation + audio reference]",
            prompt,
        )
        self.assertIn(
            "<Subject 1> (appears in [Shot 1]): fully_preserved -",
            prompt,
        )
        self.assertIn("<Audio 1>: reference -", prompt)
        self.assertNotIn("(retention reference)", prompt)
        self.assertEqual(
            validate_h3_prompt_contract(
                prompt,
                [{"speaker_id": "monica", "spoken_text": "We need to talk."}],
                mode="ref2va",
                references=references,
            ),
            [],
        )

    def test_project_speaker_ids_remain_stable_when_cast_order_changes(self):
        plans = [
            {
                "video_prompt": "Monica says one. overall_soundscape: Room tone.",
                "_director_subjects_on_screen": [
                    {"character_id": "ross", "speaker_name": "Ross"},
                    {"character_id": "monica", "speaker_name": "Monica"},
                ],
                "_director_dialogue_beats": [
                    {"speaker_id": "monica", "spoken_text": "One."},
                ],
            },
            {
                "video_prompt": "Ross says two. overall_soundscape: Room tone.",
                "_director_subjects_on_screen": [
                    {"character_id": "monica", "speaker_name": "Monica"},
                    {"character_id": "ross", "speaker_name": "Ross"},
                ],
                "_director_dialogue_beats": [
                    {"speaker_id": "ross", "spoken_text": "Two."},
                ],
            },
        ]

        compile_h3_clip_plans(plans)

        self.assertIn("Monica (S1)", plans[0]["video_prompt"])
        self.assertIn("Ross (S2)", plans[1]["video_prompt"])
        self.assertEqual(
            plans[0]["_director_speaker_registry"],
            plans[1]["_director_speaker_registry"],
        )

    def test_reviewed_prompt_edit_becomes_the_new_source_before_generation(self):
        plans = [{
            "video_prompt": "A wide office shot. overall_soundscape: Office hum.",
            "_director_dialogue_beats": [],
            "_director_subjects_on_screen": [],
        }]
        compile_h3_clip_plans(plans)
        plans[0]["video_prompt"] = (
            "integrated_multimodal_description: [Shot 1] A reviewed close-up "
            "of the office manager. No character speaks. "
            "overall_soundscape: Quiet office hum. non_diegetic_music: N/A"
        )

        compile_h3_clip_plans(plans, prompt_modes=["i2va"], durations=[8.0])

        self.assertIn("reviewed close-up", plans[0]["video_prompt"])
        self.assertNotIn("wide office shot", plans[0]["video_prompt"])
        self.assertTrue(plans[0]["video_prompt"].startswith(
            "For the target video, at 0.00 seconds into the target video"
        ))

    def test_ref2va_driving_audio_is_not_mistaken_for_a_silent_shot(self):
        prompt, _ = compile_h3_official_prompt(
            "A singer performs on stage. overall_soundscape: Venue ambience.",
            [],
            [],
            mode="ref2va",
            references=[{
                "type": "image",
                "role": "the singer",
                "image_intent": "identity",
            }, {
                "type": "audio",
                "role": "the exact song performance",
                "audio_intent": "drive",
            }],
        )

        self.assertIn("mapped driving audio", prompt)
        self.assertNotIn("No character speaks", prompt)
        self.assertIn("<Audio 1>", prompt)
        self.assertIn(
            "summary: [reference generation + audio reuse]",
            prompt,
        )
        self.assertIn("<Audio 1>: partially_copy -", prompt)
        self.assertEqual(
            validate_h3_prompt_contract(
                prompt,
                mode="ref2va",
                references=[{"type": "audio"}],
            ),
            [],
        )


class TestH3CharacterAuthenticity(unittest.TestCase):
    def test_voice_bible_keeps_only_supplied_cast(self):
        rows = [{
            "character_name": "Ross",
            "personality_engine": "Defensive precision hides embarrassment",
            "speech_pattern": "Over-explains, corrects wording, then backtracks",
            "relationship_behavior": "Gets more defensive when Monica presses him",
            "performance_direction": "Tight, quick, increasingly flustered delivery",
            "avoid": "Generic professor speeches and constant jargon",
        }, {
            "character_name": "Chandler",
            "personality_engine": "Deflects with jokes",
            "speech_pattern": "Punchlines",
            "relationship_behavior": "Teases friends",
            "performance_direction": "Dry delivery",
            "avoid": "Sincerity",
        }]

        bible = _normalize_h3_voice_bible(
            [{"characters": rows}],
            supported_character_text=(
                "A Friends episode starring Ross, Joey, and Monica."
            ),
        )

        self.assertEqual([entry["character_name"] for entry in bible], ["Ross"])

    def test_table_read_revises_generated_line_but_locks_user_quote(self):
        manifest = [{
            "speaker_name": "Monica",
            "spoken_text": "Do not touch that towel.",
        }, {
            "speaker_name": "Joey",
            "spoken_text": "I am experiencing a strong desire for food.",
        }]
        rows = [{
            "turn": 1,
            "speaker_name": "Monica",
            "original_text": "Do not touch that towel.",
            "revised_text": "Seriously, do not touch that towel.",
            "delivery": "fast and controlling",
        }, {
            "turn": 2,
            "speaker_name": "Joey",
            "original_text": "I am experiencing a strong desire for food.",
            "revised_text": "Wait. Is there food?",
            "delivery": "simple, hopeful, and immediate",
        }]

        revised, changed = _apply_h3_character_table_read(
            manifest,
            rows,
            story_description=(
                'Monica says, "Do not touch that towel!" Joey reacts.'
            ),
            max_spoken_words=40,
        )

        self.assertEqual(revised[0]["spoken_text"], "Do not touch that towel!")
        self.assertEqual(revised[1]["spoken_text"], "Wait. Is there food?")
        self.assertEqual(
            revised[1]["source_beat"]["delivery"],
            "simple, hopeful, and immediate",
        )
        self.assertEqual(changed, 1)

    def test_table_read_rejects_speaker_reassignment(self):
        with self.assertRaisesRegex(ValueError, "reassigned"):
            _apply_h3_character_table_read(
                [{"speaker_name": "Ross", "spoken_text": "I'm fine."}],
                [{
                    "turn": 1,
                    "speaker_name": "Joey",
                    "original_text": "I'm fine.",
                    "revised_text": "I'm good.",
                    "delivery": "casual",
                }],
                story_description="Ross insists he is fine.",
                max_spoken_words=20,
            )

    def test_speaker_visual_contract_adds_framing_and_voice_direction(self):
        shots = [{
            "subjects_on_screen": [{
                "character_id": "ross",
                "speaker_name": "Ross",
                "visual_description": "Ross from Friends",
            }],
            "spatial_setup": (
                "Ross stands center foreground while Joey watches from "
                "screen-right midground."
            ),
            "dialogue_beats": [{
                "speaker_id": "ross",
                "spoken_text": "This is not a normal situation.",
                "delivery": "flustered",
            }],
            "camera_plan": {"framing": "medium shot"},
            "audio_plan": {"mode": "dialogue_driven"},
            "video_prompt": (
                "Ross stands in the kitchen. overall_soundscape: Room tone. "
                "non_diegetic_music: N/A."
            ),
        }, {
            "subjects_on_screen": [{
                "character_id": "joey",
                "speaker_name": "Joey",
                "visual_description": "Joey from Friends",
                "wardrobe": "orange shirt, jeans, and white shoes",
                "position_or_relation": "screen-right midground, standing",
            }],
            "spatial_setup": "Joey stands screen-right midground.",
            "dialogue_beats": [],
            "camera_plan": {"framing": "medium shot"},
            "audio_plan": {"mode": "ambient_only"},
            "video_prompt": (
                "Joey waits silently. overall_soundscape: Room tone. "
                "non_diegetic_music: N/A."
            ),
        }]
        bible = [{
            "character_name": "Ross",
            "performance_direction": "precise, defensive, escalating cadence",
        }]

        _enforce_h3_speaker_visual_contract(shots, bible)

        self.assertIn(
            "precise, defensive, escalating cadence",
            shots[0]["dialogue_beats"][0]["delivery"],
        )
        self.assertIn("mouth remain unobstructed", shots[0]["camera_plan"]["reframing_notes"])
        self.assertIn("SPEAKER VISIBILITY:", shots[0]["video_prompt"])
        self.assertEqual(
            [subject["speaker_name"] for subject in shots[0]["subjects_on_screen"]],
            ["Ross", "Joey"],
        )
        self.assertEqual(
            shots[0]["subjects_on_screen"][1]["position_or_relation"],
            "in the exact position and pose stated in spatial_setup",
        )

    def test_planner_locks_validated_table_read_before_h3_shot_planning(self):
        screenplay = """INT. APARTMENT - DAY

ROSS
This is an internal biological emergency.

MONICA
I have classified this as a Level Three problem.
"""
        bible = [{
            "character_name": "Ross",
            "personality_engine": "Defensive precision hides embarrassment",
            "speech_pattern": "Over-explains and corrects himself",
            "relationship_behavior": "Gets defensive when Monica presses him",
            "performance_direction": "precise, quick, increasingly flustered",
            "avoid": "generic academic monologues",
        }, {
            "character_name": "Monica",
            "personality_engine": "Control and competence conceal anxiety",
            "speech_pattern": "Fast, direct corrections and practical questions",
            "relationship_behavior": "Challenges Ross with sibling familiarity",
            "performance_direction": "fast, grounded, sharply practical",
            "avoid": "invented rule numbers and generic neat-freak slogans",
        }]
        table_read = [{
            "turn": 1,
            "speaker_name": "Ross",
            "original_text": "This is an internal biological emergency.",
            "revised_text": "Okay, medically? This is bad.",
            "delivery": "precise, defensive, and quickly unraveling",
        }, {
            "turn": 2,
            "speaker_name": "Monica",
            "original_text": "I have classified this as a Level Three problem.",
            "revised_text": "Ross, stop diagnosing and go.",
            "delivery": "fast, blunt, and familiarly exasperated",
        }]

        def shot(name: str, speaker: str, line: str) -> dict:
            character_id = speaker.casefold()
            return {
                "title": name,
                "duration_sec": 10,
                "scene_goal": name,
                "narrative_role": "setup",
                "scene_type": "dialogue",
                "continuity_strategy": "continuous",
                "continuity_group": "apartment_day",
                "subjects_on_screen": [{
                    "visual_description": f"{speaker} from Friends",
                    "character_id": character_id,
                    "speaker_name": speaker,
                    "position_or_relation": "screen-center foreground, standing",
                    "wardrobe": "casual shirt, dark trousers, brown shoes",
                }],
                "spatial_setup": f"{speaker} stands screen-center foreground.",
                "environment": "Friends TV show apartment kitchen",
                "visual_style": "bright multi-camera sitcom",
                "lighting": "warm daylight",
                "mood": "comic concern",
                "action_beats": [f"{speaker} faces the other person."],
                "dialogue_beats": [{
                    "speaker_id": character_id,
                    "spoken_text": line,
                    "delivery": "conversational",
                    "physical_cue": f"{speaker} visibly speaks.",
                    "priority": "high",
                }],
                "camera_plan": {
                    "framing": "medium shot",
                    "movement": "static",
                    "movement_intensity": "static",
                },
                "audio_plan": {
                    "mode": "dialogue_driven",
                    "ambience": "quiet apartment room tone",
                    "effects": [],
                    "vocal_style": "natural",
                    "timing_anchor": "audio",
                    "lip_sync_critical": True,
                },
                "ending_beat": f"{speaker} holds position.",
                "closing_blocking": f"{speaker} remains center foreground.",
                "video_prompt": (
                    f"{speaker} speaks <d>[English] {line}</d>. "
                    "overall_soundscape: Quiet room tone. "
                    "non_diegetic_music: N/A."
                ),
                "multishot": False,
                "window_prompts": [],
            }

        def generate(**kwargs):
            system = kwargs["system_prompt"]
            if "character and dialogue editor" in system:
                return json.dumps(bible)
            if "acclaimed screenwriter" in system:
                self.assertIn("Defensive precision", system)
                self.assertIn("Control and competence", system)
                return screenplay
            if "H3 CHARACTER TABLE-READ" in system:
                self.assertIn("generic neat-freak slogans", kwargs["prompt"])
                return json.dumps(table_read)
            return json.dumps([
                shot(
                    "Ross admits the problem",
                    "Ross",
                    "This is an internal biological emergency.",
                ),
                shot(
                    "Monica responds",
                    "Monica",
                    "I have classified this as a Level Three problem.",
                ),
            ])

        planner = ShortFilmPlanner(
            llm_generate=generate,
            llm_generate_streaming=generate,
        )
        plan = planner.plan(
            story_description=(
                "A Friends episode starring Ross and Monica in her apartment."
            ),
            target_duration=20,
            target_scenes=2,
            video_model="minimax_h3",
            shot_image_policy="prompt_only",
            fps=24,
            frames_steps=17,
            frames_minimum=124,
            frames_maximum=345,
        )

        self.assertEqual(
            [beat.spoken_text for shot_plan in plan.shots for beat in shot_plan.dialogue_beats],
            ["Okay, medically? This is bad.", "Ross, stop diagnosing and go."],
        )
        self.assertIn(
            "quickly unraveling",
            plan.shots[0].dialogue_beats[0].delivery,
        )
        self.assertIn("quickly unraveling", plan.shots[0].video_prompt)
        self.assertNotIn(
            "This is an internal biological emergency",
            plan.shots[0].video_prompt,
        )
        self.assertIn("SPEAKER VISIBILITY:", plan.shots[0].video_prompt)


class TestH3DirectorDialogueBudget(unittest.TestCase):
    def test_preferred_durations_obey_hardware_safe_frame_ceiling(self):
        minimum_only = _h3_preferred_native_durations(
            fps=24,
            frames_minimum=124,
            frames_maximum=124,
            frames_steps=17,
        )
        self.assertEqual(minimum_only, [124 / 24])

        medium = _h3_preferred_native_durations(
            fps=24,
            frames_minimum=124,
            frames_maximum=243,
            frames_steps=17,
        )
        self.assertTrue(all(duration <= 243 / 24 for duration in medium))
        self.assertIn(192 / 24, medium)
        self.assertIn(243 / 24, medium)

        full = _h3_preferred_native_durations(
            fps=24,
            frames_minimum=124,
            frames_maximum=345,
            frames_steps=17,
        )
        self.assertIn(345 / 24, full)

    def test_reports_over_budget_without_mutating_or_truncating_lines(self):
        spoken = "one two three four five six seven eight nine ten eleven"
        shots = [{
            "title": "Crowded exchange",
            "duration_sec": 5,
            "dialogue_beats": [{"spoken_text": spoken}],
        }]

        violations = h3_dialogue_budget_violations(shots)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["word_count"], 11)
        self.assertEqual(violations[0]["word_budget"], 10)
        self.assertEqual(shots[0]["dialogue_beats"][0]["spoken_text"], spoken)

    def test_accepts_complete_lines_within_two_words_per_second(self):
        shots = [{
            "duration_sec": 5,
            "dialogue_beats": [{"spoken_text": "one two three four five six"}],
        }]

        self.assertEqual(h3_dialogue_budget_violations(shots), [])

    def test_frame_schedule_expands_only_enough_for_reported_dialogue_floors(self):
        # Mirrors the four violations from the field report: 20/17, 23/16,
        # 26/20, and 25/17 words across a seven-shot, 60-second H3 plan.
        word_counts = [20, 23, 0, 26, 0, 25, 0]
        fps = 24
        schedule = _fit_bounded_frame_schedule(
            [8.5] * len(word_counts),
            target_duration=60,
            fps=fps,
            minimum_frames=124,
            maximum_frames=345,
            frame_step=17,
            minimum_frames_by_item=[
                (word_count * fps + 1) // 2
                for word_count in word_counts
            ],
        )

        scheduled_shots = [
            {
                "duration_sec": frames / fps,
                "dialogue_beats": [{
                    "spoken_text": " ".join(
                        f"word{index}" for index in range(word_count)
                    ),
                }],
            }
            for word_count, frames in zip(word_counts, schedule)
        ]
        self.assertEqual(
            h3_dialogue_budget_violations(scheduled_shots),
            [],
        )
        self.assertAlmostEqual(sum(schedule) / fps, 63.7916667, places=5)

    def test_coalesces_adjacent_exchange_with_internal_camera_coverage(self):
        cast = [{
            "visual_description": "Ross in a brown jacket",
            "character_id": "ross",
            "speaker_name": "Ross",
            "position_or_relation": "screen-left foreground, standing",
            "wardrobe": "brown jacket, blue shirt, dark slacks, brown shoes",
        }, {
            "visual_description": "Monica in a red top",
            "character_id": "monica",
            "speaker_name": "Monica",
            "position_or_relation": "screen-right foreground, standing",
            "wardrobe": "red top, dark trousers, black shoes",
        }]

        def shot(title: str, speaker: str, line: str, closing: str) -> dict:
            return {
                "title": title,
                "duration_sec": 7,
                "scene_goal": title,
                "narrative_role": "rising_action",
                "scene_type": "dialogue",
                "continuity_strategy": "continuous",
                "continuity_group": "apartment_day",
                "subjects_on_screen": [dict(subject) for subject in cast],
                "spatial_setup": "Ross stands left and Monica stands right.",
                "environment": "Friends apartment kitchen",
                "visual_style": "bright multi-camera sitcom",
                "lighting": "warm daylight",
                "mood": "comic tension",
                "action_beats": [f"{speaker} reacts before speaking."],
                "dialogue_beats": [{
                    "speaker_id": speaker.casefold(),
                    "spoken_text": line,
                    "delivery": "conversational",
                    "physical_cue": f"{speaker} visibly speaks.",
                    "priority": "high",
                }],
                "camera_plan": {
                    "framing": f"medium shot on {speaker}",
                    "movement": "subtle push in",
                    "movement_intensity": "subtle",
                },
                "audio_plan": {
                    "mode": "dialogue_driven",
                    "ambience": "quiet kitchen room tone",
                    "effects": [],
                    "vocal_style": "natural",
                    "timing_anchor": "audio",
                    "lip_sync_critical": True,
                },
                "ending_beat": f"{speaker} finishes the line.",
                "closing_blocking": closing,
                "video_prompt": f"{speaker} speaks in the kitchen.",
                "multishot": False,
                "window_prompts": [],
            }

        original = [
            shot("Ross asks", "Ross", "What happened here?", "Ross looks right."),
            shot(
                "Monica answers",
                "Monica",
                "You happened here, and now I have to clean it.",
                "Monica points at Ross while Ross looks embarrassed.",
            ),
        ]
        original[1]["dialogue_beats"].append({
            "speaker_id": "ross",
            "spoken_text": "I understand that now.",
            "delivery": "embarrassed",
            "physical_cue": "Ross lowers his eyes.",
            "priority": "high",
        })
        compacted, merges = _coalesce_h3_dialogue_shots(
            original,
            fps=24,
            minimum_frames=124,
            maximum_frames=345,
            frame_step=17,
            minimum_shots=1,
        )

        self.assertEqual(merges, [(1, 2)])
        self.assertEqual(len(compacted), 1)
        self.assertEqual(
            [beat["spoken_text"] for beat in compacted[0]["dialogue_beats"]],
            [
                "What happened here?",
                "You happened here, and now I have to clean it.",
                "I understand that now.",
            ],
        )
        self.assertEqual(
            compacted[0]["spatial_setup"],
            original[0]["spatial_setup"],
        )
        self.assertEqual(
            compacted[0]["closing_blocking"],
            original[1]["closing_blocking"],
        )
        self.assertIn(
            "speaker-motivated internal cut or reframe",
            compacted[0]["camera_plan"]["reframing_notes"],
        )
        self.assertIn(
            "Ross, then Monica, then Ross",
            compacted[0]["camera_plan"]["reframing_notes"],
        )
        self.assertLessEqual(compacted[0]["duration_sec"], 345 / 24)

    def test_coalescer_preserves_hard_boundaries_and_minimum_clip_count(self):
        cast = [{
            "character_id": "ross",
            "speaker_name": "Ross",
            "visual_description": "Ross in a brown jacket",
        }, {
            "character_id": "monica",
            "speaker_name": "Monica",
            "visual_description": "Monica in a red top",
        }]

        def shot(group: str, speaker: str, line: str) -> dict:
            return {
                "duration_sec": 7,
                "continuity_group": group,
                "environment": f"Location {group}",
                "subjects_on_screen": [dict(subject) for subject in cast],
                "action_beats": ["The speaker reacts."],
                "dialogue_beats": [{
                    "speaker_id": speaker,
                    "spoken_text": line,
                }],
                "camera_plan": {},
                "audio_plan": {},
                "multishot": False,
            }

        across_cut = [
            shot("kitchen", "ross", "What happened?"),
            shot("hallway", "monica", "Follow me."),
        ]
        same_speaker = [
            shot("kitchen", "ross", "First thought."),
            shot("kitchen", "ross", "Second thought."),
        ]
        valid_exchange = [
            shot("kitchen", "ross", "What happened?"),
            shot("kitchen", "monica", "You happened."),
        ]

        for label, shots, minimum_shots in (
            ("continuity boundary", across_cut, 1),
            ("single speaker", same_speaker, 1),
            ("runtime floor", valid_exchange, 2),
        ):
            with self.subTest(label=label):
                compacted, merges = _coalesce_h3_dialogue_shots(
                    shots,
                    fps=24,
                    minimum_frames=124,
                    maximum_frames=345,
                    frame_step=17,
                    minimum_shots=minimum_shots,
                )
                self.assertEqual(len(compacted), 2)
                self.assertEqual(merges, [])


class TestH3DirectorDialoguePlanning(unittest.TestCase):
    def test_extracts_complete_centered_screenplay_dialogue_manifest(self):
        screenplay = """<think>private planning that must be ignored</think>
INT. APARTMENT - DAY

<center>JOEY</center>
> Morning, R. You look... intense.

<center>ROSS</center>
> (Muttering)
> It’s not a puzzle. It’s... a bio-hazard.

<center>MONICA</center>
> Ross! Did you just *shit* yourself?
"""

        manifest = _extract_h3_screenplay_dialogue(screenplay)

        self.assertEqual(
            manifest,
            [
                {
                    "speaker_name": "JOEY",
                    "spoken_text": "Morning, R. You look... intense.",
                },
                {
                    "speaker_name": "ROSS",
                    "spoken_text": "It’s not a puzzle. It’s... a bio-hazard.",
                },
                {
                    "speaker_name": "MONICA",
                    "spoken_text": "Ross! Did you just *shit* yourself?",
                },
            ],
        )

    def test_manifest_reconciliation_keeps_dialogue_in_semantic_shots(self):
        manifest = [
            {"speaker_name": "JOEY", "spoken_text": "Morning, Ross."},
            {"speaker_name": "ROSS", "spoken_text": "Morning, Joey."},
        ]
        repaired = [
            {
                "title": "Silent establishing shot",
                "subjects_on_screen": [{
                    "visual_description": "Ross Geller in a brown jacket",
                }],
                "dialogue_beats": [],
            },
            {
                "title": "Joey enters",
                "subjects_on_screen": [{
                    "visual_description": "Joey Tribbiani in an orange shirt",
                }, {
                    "visual_description": "Ross Geller in a brown jacket",
                }],
                "dialogue_beats": [{
                    "spoken_text": "Morning, Ross.",
                    "delivery": "cheerful",
                }],
            },
            {
                "title": "Ross answers",
                "subjects_on_screen": [{
                    "visual_description": "Ross Geller in a brown jacket",
                }, {
                    "visual_description": "Joey Tribbiani in an orange shirt",
                }],
                "dialogue_beats": [{
                    "spoken_text": "Morning, Joey.",
                    "delivery": "dry",
                }],
            },
        ]

        _reconcile_h3_dialogue_manifest(repaired, manifest)

        self.assertEqual(repaired[0]["dialogue_beats"], [])
        self.assertEqual(
            repaired[1]["dialogue_beats"][0]["speaker_id"],
            "dialogue_joey",
        )
        self.assertEqual(
            repaired[2]["dialogue_beats"][0]["speaker_id"],
            "dialogue_ross",
        )
        self.assertEqual(
            repaired[1]["subjects_on_screen"][0]["character_id"],
            "dialogue_joey",
        )

    def test_repair_manifest_restores_changed_turn_for_visible_speaker(self):
        manifest = [{
            "speaker_name": "ROSS",
            "spoken_text": "The exact screenplay line stays intact.",
        }]
        repaired = [{
            "subjects_on_screen": [{
                "visual_description": "Ross Geller in a brown jacket",
                "character_id": "ross",
                "speaker_name": "Ross",
            }, {
                "visual_description": "Joey Tribbiani in an orange shirt",
                "character_id": "joey",
                "speaker_name": "Joey",
            }],
            "dialogue_beats": [{
                "speaker_id": "joey",
                "spoken_text": "A duplicated line the repair invented.",
                "delivery": "excited",
                "physical_cue": "Joey waves both hands.",
            }],
        }]

        with self.assertRaisesRegex(ValueError, "changed or moved"):
            _reconcile_h3_dialogue_manifest(repaired, manifest)

        _reconcile_h3_dialogue_manifest(
            repaired,
            manifest,
            allow_manifest_restore=True,
        )

        beat = repaired[0]["dialogue_beats"][0]
        self.assertEqual(beat["speaker_id"], "ross")
        self.assertEqual(
            beat["spoken_text"],
            "The exact screenplay line stays intact.",
        )
        self.assertEqual(beat["delivery"], "natural and context-appropriate")
        self.assertIn("ross visibly delivers", beat["physical_cue"].casefold())

    def test_repair_manifest_does_not_invent_an_absent_speaker(self):
        manifest = [{
            "speaker_name": "MONICA",
            "spoken_text": "This line belongs to Monica.",
        }]
        repaired = [{
            "subjects_on_screen": [{
                "visual_description": "Ross Geller in a brown jacket",
                "character_id": "ross",
                "speaker_name": "Ross",
            }, {
                "visual_description": "Joey Tribbiani in an orange shirt",
                "character_id": "joey",
                "speaker_name": "Joey",
            }],
            "dialogue_beats": [{
                "speaker_id": "joey",
                "spoken_text": "A wrong line.",
            }],
        }]

        with self.assertRaisesRegex(ValueError, "(?i)Monica.*not visible"):
            _reconcile_h3_dialogue_manifest(
                repaired,
                manifest,
                allow_manifest_restore=True,
            )

    def test_truncated_dialogue_plan_is_rejected_not_rebucketed(self):
        manifest = [
            {"speaker_name": "JOEY", "spoken_text": "Line one."},
            {"speaker_name": "ROSS", "spoken_text": "Line two."},
            {"speaker_name": "MONICA", "spoken_text": "Line three."},
        ]
        truncated = [{
            "subjects_on_screen": [{
                "visual_description": "Joey Tribbiani",
            }],
            "dialogue_beats": [{"spoken_text": "Line one."}],
        }, {
            "subjects_on_screen": [{
                "visual_description": "Ross Geller",
            }],
            "dialogue_beats": [{"spoken_text": "Line two."}],
        }]

        with self.assertRaisesRegex(ValueError, "3 spoken turns.*2"):
            _reconcile_h3_dialogue_manifest(truncated, manifest)

    def test_incomplete_json_repair_tail_is_detected(self):
        required = ["title", "video_prompt", "multishot", "window_prompts"]
        items = [{
            "title": "Complete",
            "video_prompt": "Prompt",
            "multishot": False,
            "window_prompts": [],
        }, {
            "title": "Truncated",
            "video_prompt": "Prompt ends at the token cap",
        }]

        issues = _h3_native_structure_issues(
            items,
            required,
            minimum_items=2,
            maximum_items=4,
        )

        self.assertEqual(
            issues,
            ["shot 2 is missing multishot, window_prompts"],
        )

    def test_semantically_complete_token_capped_tail_is_recovered(self):
        required = [
            "title", "duration_sec", "scene_goal", "narrative_role",
            "scene_type", "continuity_strategy", "continuity_group",
            "subjects_on_screen", "spatial_setup", "environment",
            "visual_style", "lighting", "mood", "action_beats",
            "dialogue_beats", "camera_plan", "audio_plan", "ending_beat",
            "closing_blocking", "video_prompt", "multishot",
            "window_prompts",
        ]

        def semantic_shot(title: str, line: str) -> dict:
            return {
                "title": title,
                "duration_sec": 10,
                "scene_goal": "Ross finishes the exchange.",
                "narrative_role": "resolution",
                "scene_type": "dialogue",
                "continuity_strategy": "continuous",
                "continuity_group": "apartment_day",
                "subjects_on_screen": [{
                    "visual_description": "Ross Geller in a brown jacket",
                    "character_id": "ross",
                    "speaker_name": "Ross",
                    "position_or_relation": "screen-right foreground, standing",
                    "wardrobe": "brown jacket, blue shirt, dark slacks, brown shoes",
                }],
                "spatial_setup": "Ross stands screen-right beside the sofa.",
                "environment": "Friends TV show apartment living room",
                "visual_style": "bright multi-camera sitcom",
                "lighting": "warm daylight",
                "mood": "friendly",
                "action_beats": ["Ross smiles and lowers his coffee mug."],
                "dialogue_beats": [{
                    "speaker_id": "ross",
                    "spoken_text": line,
                    "delivery": "dry",
                    "physical_cue": "Ross looks toward Joey.",
                    "priority": "high",
                }],
            }

        complete = semantic_shot("Complete", "First line.")
        complete.update({
            "camera_plan": {"framing": "medium shot"},
            "audio_plan": {"mode": "dialogue_driven"},
            "ending_beat": "Ross settles beside the sofa.",
            "closing_blocking": "Ross remains screen-right.",
            "video_prompt": "Ross speaks <d>[English] First line.</d>",
            "multishot": False,
            "window_prompts": [],
        })
        truncated = semantic_shot("Truncated", "The final line stays exact.")
        items = [complete, truncated]

        filled = _complete_h3_truncated_tail(items, required)

        self.assertEqual(
            filled,
            [
                "camera_plan", "audio_plan", "ending_beat",
                "closing_blocking", "video_prompt", "multishot",
                "window_prompts",
            ],
        )
        self.assertEqual(
            truncated["dialogue_beats"][0]["spoken_text"],
            "The final line stays exact.",
        )
        self.assertIn("Friends TV show apartment", truncated["video_prompt"])
        self.assertEqual(
            _h3_native_structure_issues(
                items,
                required,
                minimum_items=2,
                maximum_items=3,
            ),
            [],
        )

    def test_truncated_tail_with_missing_dialogue_is_not_recovered(self):
        required = [
            "title", "duration_sec", "scene_goal", "narrative_role",
            "scene_type", "continuity_strategy", "continuity_group",
            "subjects_on_screen", "spatial_setup", "environment",
            "visual_style", "lighting", "mood", "action_beats",
            "dialogue_beats", "camera_plan", "audio_plan", "ending_beat",
            "closing_blocking", "video_prompt", "multishot",
            "window_prompts",
        ]
        unsafe_tail = {
            "title": "Unsafe truncation",
            "duration_sec": 10,
            "scene_goal": "Finish the story",
            "narrative_role": "resolution",
            "scene_type": "dialogue",
            "continuity_strategy": "continuous",
            "continuity_group": "room",
            "subjects_on_screen": [],
            "spatial_setup": "A person stands center frame.",
            "environment": "A room",
            "visual_style": "cinematic",
            "lighting": "daylight",
            "mood": "calm",
            "action_beats": ["The person reacts."],
        }

        self.assertEqual(
            _complete_h3_truncated_tail([unsafe_tail], required),
            [],
        )
        self.assertNotIn("video_prompt", unsafe_tail)

    def test_long_h3_plan_gets_tail_completion_headroom(self):
        self.assertEqual(_h3_planner_token_budget(90), 21600)
        self.assertEqual(_h3_planner_token_budget(120), 23000)

    def test_truncated_first_plan_repairs_without_shifting_dialogue_early(self):
        screenplay = """INT. APARTMENT - DAY

<center>JOEY</center>
> Morning, Ross.

Ross turns toward Joey.

<center>ROSS</center>
> Morning, Joey.
"""
        calls = []
        cast = [{
            "visual_description": "Joey Tribbiani in an orange shirt",
            "character_id": "char_1",
            "speaker_name": "Joey",
            "position_or_relation": "screen-left foreground, standing",
            "wardrobe": "orange shirt, blue jeans, white shoes",
        }, {
            "visual_description": "Ross Geller in a brown jacket",
            "character_id": "char_0",
            "speaker_name": "Ross",
            "position_or_relation": "screen-right foreground, standing",
            "wardrobe": "brown jacket, blue shirt, dark slacks, brown shoes",
        }]

        def shot(title: str, line: str = "", speaker_id: str = "") -> dict:
            beats = []
            if line:
                beat = {
                    "spoken_text": line,
                    "delivery": "conversational",
                    "physical_cue": "The speaker looks at the other person.",
                    "priority": "high",
                }
                if speaker_id:
                    beat["speaker_id"] = speaker_id
                beats.append(beat)
            return {
                "title": title,
                "duration_sec": 10,
                "scene_goal": title,
                "narrative_role": "setup",
                "scene_type": "dialogue" if line else "opening",
                "continuity_strategy": "continuous",
                "continuity_group": "apartment_day",
                "subjects_on_screen": [dict(subject) for subject in cast],
                "spatial_setup": "Joey stands left and Ross stands right.",
                "environment": "Friends TV show apartment living room",
                "visual_style": "bright multi-camera sitcom",
                "lighting": "warm daylight",
                "mood": "friendly",
                "action_beats": ["Both men hold their positions."],
                "dialogue_beats": beats,
                "camera_plan": {
                    "framing": "medium two-shot",
                    "movement": "static",
                    "movement_intensity": "static",
                },
                "audio_plan": {
                    "mode": "dialogue_driven" if line else "ambient_only",
                    "ambience": "quiet apartment room tone",
                    "effects": [],
                    "vocal_style": "natural",
                    "timing_anchor": "audio" if line else "video",
                    "lip_sync_critical": bool(line),
                },
                "ending_beat": "Both men remain in place.",
                "closing_blocking": "Joey remains left and Ross remains right.",
                "video_prompt": (
                    "integrated_multimodal_description: [Shot 1] "
                    + (
                        f"A visible speaker says <d>[English] {line}</d>. "
                        if line else
                        "Both men wait silently. "
                    )
                    + "overall_soundscape: Quiet room tone. "
                    "non_diegetic_music: N/A."
                ),
                "multishot": False,
                "window_prompts": [],
            }

        def generate(**kwargs):
            calls.append(kwargs)
            if "acclaimed screenwriter" in kwargs["system_prompt"]:
                return screenplay
            if "H3 WHOLE-PLAN REPAIR" in kwargs["prompt"]:
                return json.dumps([
                    shot("Silent establishment"),
                    shot("Joey greets Ross", "Morning, Ross."),
                    shot("Ross replies", "Morning, Joey."),
                ])
            # Simulate the field failure: json_repair recovered only the first
            # complete object from a token-capped response.
            return json.dumps([
                shot("Joey greets Ross", "Morning, Ross.", "char_1"),
            ])

        planner = ShortFilmPlanner(
            llm_generate=generate,
            llm_generate_streaming=generate,
        )
        plan = planner.plan(
            story_description="Joey and Ross exchange morning greetings.",
            target_duration=30,
            target_scenes=3,
            video_model="minimax_h3",
            shot_image_policy="prompt_only",
            fps=24,
            frames_steps=17,
            frames_minimum=124,
            frames_maximum=345,
        )

        self.assertEqual(len(plan.shots), 3)
        self.assertFalse(plan.shots[0].dialogue_beats)
        self.assertEqual(
            plan.shots[1].dialogue_beats[0].spoken_text,
            "Morning, Ross.",
        )
        self.assertEqual(
            plan.shots[2].dialogue_beats[0].spoken_text,
            "Morning, Joey.",
        )
        self.assertEqual(plan.shots[1].dialogue_beats[0].speaker_id, "char_1")
        self.assertEqual(plan.shots[2].dialogue_beats[0].speaker_id, "char_0")

    def test_token_capped_whole_plan_tail_is_completed_before_validation(self):
        screenplay = """INT. APARTMENT - DAY

<center>JOEY</center>
> Morning, Ross.

<center>ROSS</center>
> Morning, Joey.
"""
        calls = []

        def shot(title: str, speaker: str, line: str) -> dict:
            character_id = speaker.casefold()
            return {
                "title": title,
                "duration_sec": 10,
                "scene_goal": title,
                "narrative_role": "resolution",
                "scene_type": "dialogue",
                "continuity_strategy": "continuous",
                "continuity_group": "apartment_day",
                "subjects_on_screen": [{
                    "visual_description": f"{speaker} from Friends",
                    "character_id": character_id,
                    "speaker_name": speaker,
                    "position_or_relation": "screen-center foreground, standing",
                    "wardrobe": "casual shirt, dark trousers, brown shoes",
                }],
                "spatial_setup": f"{speaker} stands center foreground.",
                "environment": "Friends TV show apartment living room",
                "visual_style": "bright multi-camera sitcom",
                "lighting": "warm daylight",
                "mood": "friendly",
                "action_beats": [f"{speaker} looks toward the other person."],
                "dialogue_beats": [{
                    "speaker_id": character_id,
                    "spoken_text": line,
                    "delivery": "conversational",
                    "physical_cue": f"{speaker} smiles.",
                    "priority": "high",
                }],
                "camera_plan": {
                    "framing": "medium shot",
                    "movement": "static",
                    "movement_intensity": "static",
                },
                "audio_plan": {
                    "mode": "dialogue_driven",
                    "ambience": "quiet apartment room tone",
                    "effects": [],
                    "vocal_style": "natural",
                    "timing_anchor": "audio",
                    "lip_sync_critical": True,
                },
                "ending_beat": f"{speaker} holds the smile.",
                "closing_blocking": f"{speaker} remains center foreground.",
                "video_prompt": (
                    f"{speaker} speaks <d>[English] {line}</d>. "
                    "overall_soundscape: Quiet room tone. "
                    "non_diegetic_music: N/A."
                ),
                "multishot": False,
                "window_prompts": [],
            }

        joey = shot("Joey greets Ross", "Joey", "Morning, Ross.")
        ross = shot("Ross replies", "Ross", "Morning, Joey.")
        # Simulate the field report: the whole-plan repair duplicates an
        # earlier line and attaches it to the wrong person even though the
        # canonical speaker remains visible in the multi-person shot.
        ross["subjects_on_screen"].append(dict(joey["subjects_on_screen"][0]))
        ross["dialogue_beats"][0]["speaker_id"] = "joey"
        ross["dialogue_beats"][0]["spoken_text"] = "Morning, Ross."
        for field in (
            "camera_plan", "audio_plan", "ending_beat", "closing_blocking",
            "video_prompt", "multishot", "window_prompts",
        ):
            ross.pop(field)

        def generate(**kwargs):
            calls.append(kwargs)
            if "acclaimed screenwriter" in kwargs["system_prompt"]:
                return screenplay
            if "H3 WHOLE-PLAN REPAIR" in kwargs["prompt"]:
                return json.dumps([joey, ross])
            # Force the whole-plan repair while retaining a known good source
            # identity for the first dialogue turn.
            return json.dumps([joey])

        planner = ShortFilmPlanner(
            llm_generate=generate,
            llm_generate_streaming=generate,
        )
        plan = planner.plan(
            story_description="Joey and Ross exchange morning greetings.",
            target_duration=20,
            target_scenes=2,
            video_model="minimax_h3",
            shot_image_policy="prompt_only",
            fps=24,
            frames_steps=17,
            frames_minimum=124,
            frames_maximum=345,
        )

        self.assertEqual(len(plan.shots), 2)
        self.assertEqual(
            plan.shots[1].dialogue_beats[0].spoken_text,
            "Morning, Joey.",
        )
        self.assertEqual(plan.shots[1].camera_plan.movement, "static hold")
        self.assertIn(
            "<d>[English] Morning, Joey.</d>",
            plan.shots[1].video_prompt,
        )
        self.assertEqual(
            sum(
                "character and dialogue editor" in call["system_prompt"]
                for call in calls
            ),
            1,
        )
        self.assertEqual(
            sum(
                "H3 CHARACTER TABLE-READ" in call["system_prompt"]
                for call in calls
            ),
            1,
        )
        self.assertEqual(
            sum(
                "breaking a screenplay into shots" in call["system_prompt"]
                and "H3 WHOLE-PLAN REPAIR" not in call["prompt"]
                for call in calls
            ),
            1,
        )
        self.assertEqual(
            sum("H3 WHOLE-PLAN REPAIR" in call["prompt"] for call in calls),
            1,
        )

    def test_repair_discards_extra_turn_when_locked_turn_needs_sentence_split(self):
        long_line = (
            "I have been trying to explain this carefully because nobody in "
            "this apartment listens when the situation becomes genuinely "
            "serious. This is exactly why I asked everyone to pay attention "
            "before breakfast."
        )
        screenplay = f"""INT. APARTMENT - DAY

<center>JOEY</center>
> {long_line}

<center>ROSS</center>
> I am listening now.
"""
        calls = []
        cast = [{
            "visual_description": "Joey Tribbiani in an orange shirt",
            "character_id": "joey",
            "speaker_name": "Joey",
            "position_or_relation": "screen-left foreground, standing",
            "wardrobe": "orange shirt, blue jeans, white shoes",
        }, {
            "visual_description": "Ross Geller in a brown jacket",
            "character_id": "ross",
            "speaker_name": "Ross",
            "position_or_relation": "screen-right foreground, standing",
            "wardrobe": "brown jacket, blue shirt, dark slacks, brown shoes",
        }]

        def beat(speaker: str, text: str) -> dict:
            return {
                "speaker_id": speaker.casefold(),
                "spoken_text": text,
                "delivery": "conversational",
                "physical_cue": f"{speaker} visibly speaks.",
                "priority": "high",
            }

        def shot(title: str, beats: list[dict]) -> dict:
            return {
                "title": title,
                "duration_sec": 10,
                "scene_goal": title,
                "narrative_role": "rising_action",
                "scene_type": "dialogue",
                "continuity_strategy": "continuous",
                "continuity_group": "apartment_day",
                "subjects_on_screen": [dict(subject) for subject in cast],
                "spatial_setup": "Joey stands left and Ross stands right.",
                "environment": "Friends TV show apartment living room",
                "visual_style": "bright multi-camera sitcom",
                "lighting": "warm daylight",
                "mood": "comic tension",
                "action_beats": ["The two friends face each other."],
                "dialogue_beats": beats,
                "camera_plan": {
                    "framing": "medium two-shot",
                    "movement": "static",
                    "movement_intensity": "static",
                },
                "audio_plan": {
                    "mode": "dialogue_driven",
                    "ambience": "quiet apartment room tone",
                    "effects": [],
                    "vocal_style": "natural",
                    "timing_anchor": "audio",
                    "lip_sync_critical": True,
                },
                "ending_beat": "They hold their positions.",
                "closing_blocking": "Joey remains left and Ross remains right.",
                "video_prompt": (
                    "The two friends speak in the apartment. "
                    "overall_soundscape: Quiet room tone. "
                    "non_diegetic_music: N/A."
                ),
                "multishot": False,
                "window_prompts": [],
            }

        initial = [shot("Incomplete first pass", [beat("Joey", long_line)])]
        repaired = [
            shot("Joey explains", [
                beat("Joey", long_line),
                beat("Joey", "This sentence was invented by Pass 2."),
            ]),
            shot("Ross answers", [beat("Ross", "I am listening now.")]),
        ]

        def generate(**kwargs):
            calls.append(kwargs)
            if "acclaimed screenwriter" in kwargs["system_prompt"]:
                return screenplay
            if "H3 WHOLE-PLAN REPAIR" in kwargs["prompt"]:
                return json.dumps(repaired)
            return json.dumps(initial)

        planner = ShortFilmPlanner(
            llm_generate=generate,
            llm_generate_streaming=generate,
        )
        plan = planner.plan(
            story_description="Joey explains himself and Ross responds.",
            target_duration=20,
            target_scenes=2,
            video_model="minimax_h3",
            shot_image_policy="prompt_only",
            fps=24,
            frames_steps=17,
            frames_minimum=124,
            frames_maximum=345,
        )

        spoken = [
            dialogue.spoken_text
            for planned_shot in plan.shots
            for dialogue in (planned_shot.dialogue_beats or [])
        ]
        self.assertEqual(len(spoken), 3)
        self.assertEqual(" ".join(spoken[:2]), long_line)
        self.assertEqual(spoken[2], "I am listening now.")
        self.assertNotIn("invented by Pass 2", " ".join(spoken))
        self.assertEqual(
            sum("H3 WHOLE-PLAN REPAIR" in call["prompt"] for call in calls),
            1,
        )

    def test_changed_multispeaker_repair_restores_words_speakers_and_order(self):
        joey = {
            "character_id": "joey",
            "speaker_name": "Joey",
            "visual_description": "Joey in an orange shirt",
            "position_or_relation": "screen-left",
            "wardrobe": "orange shirt and jeans",
        }
        ross = {
            "character_id": "ross",
            "speaker_name": "Ross",
            "visual_description": "Ross in a brown jacket",
            "position_or_relation": "screen-right",
            "wardrobe": "brown jacket and slacks",
        }
        original = [{
            "subjects_on_screen": [joey],
            "dialogue_beats": [{
                "speaker_id": "joey",
                "spoken_text": "This first line stays exact.",
            }],
        }, {
            "subjects_on_screen": [ross],
            "dialogue_beats": [{
                "speaker_id": "ross",
                "spoken_text": "This second line also stays exact.",
            }],
        }]
        repaired = [{
            "scene_goal": "Joey reacts",
            "subjects_on_screen": [ross],
            "dialogue_beats": [{
                "speaker_id": "ross",
                "spoken_text": "Wrong reordered line.",
            }],
            "video_prompt": "Ross says <d>[English] Wrong reordered line.</d>",
        }, {
            "scene_goal": "Silent beat",
            "subjects_on_screen": [],
            "dialogue_beats": [],
            "video_prompt": "A silent beat.",
        }, {
            "scene_goal": "Ross answers",
            "subjects_on_screen": [joey],
            "dialogue_beats": [{
                "speaker_id": "joey",
                "spoken_text": "Another wrong line.",
            }],
            "video_prompt": "Joey says <d>[English] Another wrong line.</d>",
        }]

        restored = _restore_h3_dialogue_after_pacing_repair(
            original,
            repaired,
            [5.0, 5.0, 5.0],
        )

        restored_beats = [
            beat
            for shot in restored
            for beat in shot.get("dialogue_beats") or []
        ]
        self.assertEqual(
            [beat["spoken_text"] for beat in restored_beats],
            [
                "This first line stays exact.",
                "This second line also stays exact.",
            ],
        )
        self.assertEqual(
            [beat["speaker_id"] for beat in restored_beats],
            ["joey", "ross"],
        )
        self.assertNotIn(
            "Wrong",
            " ".join(shot["video_prompt"] for shot in restored),
        )

    def test_missing_pass2_turn_is_compiled_without_whole_plan_llm_repair(self):
        screenplay = """INT. APARTMENT - DAY

<center>JOEY</center>
> Morning, Ross.

<center>ROSS</center>
> Morning, Joey.
"""
        calls = []
        cast = [{
            "visual_description": "Joey Tribbiani in an orange shirt",
            "character_id": "joey",
            "speaker_name": "Joey",
            "position_or_relation": "screen-left foreground, standing",
            "wardrobe": "orange shirt, blue jeans, white shoes",
        }, {
            "visual_description": "Ross Geller in a brown jacket",
            "character_id": "ross",
            "speaker_name": "Ross",
            "position_or_relation": "screen-right foreground, standing",
            "wardrobe": "brown jacket, blue shirt, dark slacks, brown shoes",
        }]

        def shot(title: str, beats: list[dict]) -> dict:
            return {
                "title": title,
                "duration_sec": 10,
                "scene_goal": title,
                "narrative_role": "setup",
                "scene_type": "dialogue" if beats else "reaction",
                "continuity_strategy": "continuous",
                "continuity_group": "apartment_day",
                "subjects_on_screen": [dict(subject) for subject in cast],
                "spatial_setup": "Joey stands left and Ross stands right.",
                "environment": "Friends TV show apartment living room",
                "visual_style": "bright multi-camera sitcom",
                "lighting": "warm daylight",
                "mood": "friendly",
                "action_beats": ["The two friends look at each other."],
                "dialogue_beats": beats,
                "camera_plan": {
                    "framing": "medium two-shot",
                    "movement": "static",
                    "movement_intensity": "static",
                },
                "audio_plan": {
                    "mode": "dialogue_driven" if beats else "ambient_only",
                    "ambience": "quiet apartment room tone",
                    "effects": [],
                    "vocal_style": "natural",
                    "timing_anchor": "audio" if beats else "video",
                    "lip_sync_critical": bool(beats),
                },
                "ending_beat": "They hold their positions.",
                "closing_blocking": "Joey remains left and Ross remains right.",
                "video_prompt": (
                    "Joey speaks <d>[English] Morning, Ross.</d>. "
                    "overall_soundscape: Quiet room tone. "
                    "non_diegetic_music: N/A."
                    if beats else
                    "They react silently. overall_soundscape: Quiet room tone. "
                    "non_diegetic_music: N/A."
                ),
                "multishot": False,
                "window_prompts": [],
            }

        def generate(**kwargs):
            calls.append(kwargs)
            if "acclaimed screenwriter" in kwargs["system_prompt"]:
                return screenplay
            return json.dumps([
                shot("Joey greets Ross", [{
                    "speaker_id": "joey",
                    "spoken_text": "Morning, Ross.",
                    "delivery": "friendly",
                    "physical_cue": "Joey smiles.",
                    "priority": "high",
                }]),
                # Gemma omitted the second screenplay turn, but the visual
                # shot itself is complete and already contains Ross.
                shot("Ross answers", []),
            ])

        planner = ShortFilmPlanner(
            llm_generate=generate,
            llm_generate_streaming=generate,
        )
        plan = planner.plan(
            story_description="Joey and Ross exchange morning greetings.",
            target_duration=20,
            target_scenes=2,
            video_model="minimax_h3",
            shot_image_policy="prompt_only",
            fps=24,
            frames_steps=17,
            frames_minimum=124,
            frames_maximum=345,
        )

        pass2_calls = [
            call for call in calls
            if "breaking a screenplay into shots" in call["system_prompt"]
        ]
        self.assertEqual(len(pass2_calls), 1)
        self.assertEqual(
            [
                beat.spoken_text
                for planned_shot in plan.shots
                for beat in (planned_shot.dialogue_beats or [])
            ],
            ["Morning, Ross.", "Morning, Joey."],
        )
        self.assertIn(
            "<d>[English] Morning, Joey.</d>",
            plan.shots[1].video_prompt,
        )

    def test_eleven_manifest_turns_fit_ten_visual_slots_without_llm_repair(self):
        long_line = (
            "one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
            "twenty twenty-one twenty-two twenty-three twenty-four twenty-five "
            "twenty-six twenty-seven twenty-eight twenty-nine"
        )
        turns = [
            ("ROSS", long_line),
            ("JOEY", "Joey line two."),
            ("MONICA", "Monica line three."),
            ("ROSS", "Ross missing turn four."),
            ("JOEY", "Joey line five."),
            ("MONICA", "Monica line six."),
            ("ROSS", "Ross line seven."),
            ("JOEY", "Joey line eight."),
            ("MONICA", "Monica line nine."),
            ("ROSS", "Ross line ten."),
            ("JOEY", "Joey line eleven."),
        ]
        screenplay = "INT. APARTMENT - DAY\n\n" + "\n\n".join(
            f"<center>{speaker}</center>\n> {line}"
            for speaker, line in turns
        )
        cast = [{
            "visual_description": "Ross Geller in a brown jacket",
            "character_id": "ross",
            "speaker_name": "Ross",
            "position_or_relation": "screen-left foreground, standing",
            "wardrobe": "brown jacket, blue shirt, dark slacks, brown shoes",
        }, {
            "visual_description": "Joey Tribbiani in an orange shirt",
            "character_id": "joey",
            "speaker_name": "Joey",
            "position_or_relation": "screen-center foreground, standing",
            "wardrobe": "orange shirt, blue jeans, white shoes",
        }, {
            "visual_description": "Monica Geller in a red top",
            "character_id": "monica",
            "speaker_name": "Monica",
            "position_or_relation": "screen-right foreground, standing",
            "wardrobe": "red top, dark trousers, black shoes",
        }]
        calls = []

        def shot(index: int, speaker: str, line: str) -> dict:
            speaker_id = speaker.casefold()
            return {
                "title": f"Visual shot {index + 1}",
                "duration_sec": 9,
                "scene_goal": f"Story beat {index + 1}",
                "narrative_role": "rising_action",
                "scene_type": "dialogue",
                "continuity_strategy": "continuous",
                "continuity_group": "apartment_day",
                "subjects_on_screen": [dict(subject) for subject in cast],
                "spatial_setup": "Ross is left, Joey center, Monica right.",
                "environment": "Friends TV show apartment living room",
                "visual_style": "bright multi-camera sitcom",
                "lighting": "warm daylight",
                "mood": "comic tension",
                "action_beats": [f"The cast performs story beat {index + 1}."],
                "dialogue_beats": [{
                    "speaker_id": speaker_id,
                    "spoken_text": line,
                    "delivery": "conversational",
                    "physical_cue": f"{speaker.title()} visibly speaks.",
                    "priority": "high",
                }],
                "camera_plan": {
                    "framing": "medium three-shot",
                    "movement": "static",
                    "movement_intensity": "static",
                },
                "audio_plan": {
                    "mode": "dialogue_driven",
                    "ambience": "quiet apartment room tone",
                    "effects": [],
                    "vocal_style": "natural",
                    "timing_anchor": "audio",
                    "lip_sync_critical": True,
                },
                "ending_beat": "The cast holds for the edit.",
                "closing_blocking": "Ross remains left, Joey center, Monica right.",
                "video_prompt": (
                    f"{speaker.title()} speaks <d>[English] {line}</d>. "
                    "overall_soundscape: Quiet room tone. "
                    "non_diegetic_music: N/A."
                ),
                "multishot": False,
                "window_prompts": [],
            }

        # Drop screenplay turn four while keeping ten complete visual shots.
        planned_turns = turns[:3] + turns[4:]

        def generate(**kwargs):
            calls.append(kwargs)
            if "acclaimed screenwriter" in kwargs["system_prompt"]:
                return screenplay
            return json.dumps([
                shot(index, speaker, line)
                for index, (speaker, line) in enumerate(planned_turns)
            ])

        planner = ShortFilmPlanner(
            llm_generate=generate,
            llm_generate_streaming=generate,
        )
        plan = planner.plan(
            story_description="Ross, Joey, and Monica talk in the apartment.",
            target_duration=90,
            target_scenes=10,
            video_model="minimax_h3",
            shot_image_policy="prompt_only",
            fps=24,
            frames_steps=17,
            frames_minimum=124,
            frames_maximum=345,
        )

        pass2_calls = [
            call for call in calls
            if "breaking a screenplay into shots" in call["system_prompt"]
        ]
        self.assertEqual(len(pass2_calls), 1)
        self.assertEqual(
            [
                beat.spoken_text
                for planned_shot in plan.shots
                for beat in (planned_shot.dialogue_beats or [])
            ],
            [line for _, line in turns],
        )
        # Ten visual slots are safely compacted to the seven clips required
        # to cover a 90-second target at H3's 14.375-second maximum. Adjacent
        # short speaker turns share native conversation coverage instead of
        # becoming separate minimum-length generations.
        self.assertEqual(len(plan.shots), 7)
        self.assertTrue(any(
            len({beat.speaker_id for beat in planned_shot.dialogue_beats}) > 1
            and "internal cut or reframe" in planned_shot.video_prompt
            for planned_shot in plan.shots
            if planned_shot.dialogue_beats
        ))
        self.assertLessEqual(
            max(planned_shot.metadata["duration_frames"] for planned_shot in plan.shots),
            345,
        )
        self.assertLessEqual(
            abs(sum(planned_shot.duration_sec for planned_shot in plan.shots) - 90),
            17 / 24,
        )

    def test_over_budget_plan_is_retimed_without_cutting_a_sentence(self):
        spoken = (
            "one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
            "twenty twenty-one twenty-two twenty-three twenty-four twenty-five"
        )
        calls = []

        def shot(title: str, duration: float, line: str = "") -> dict:
            return {
                "title": title,
                "duration_sec": duration,
                "scene_goal": title,
                "narrative_role": "setup",
                "scene_type": "dialogue" if line else "reaction",
                "continuity_strategy": "independent",
                "continuity_group": "office_1",
                "subjects_on_screen": [{
                    "visual_description": "Joey in a navy shirt",
                    "character_id": "joey",
                    "speaker_name": "Joey",
                    "position_or_relation": "screen-left foreground, seated",
                    "wardrobe": "navy shirt, blue jeans, brown shoes",
                }],
                "spatial_setup": "Joey sits screen-left at a cafe table",
                "environment": "A quiet cafe",
                "visual_style": "cinematic comedy",
                "lighting": "warm daylight",
                "mood": "dry comedy",
                "action_beats": ["Joey leans forward"],
                "dialogue_beats": ([{
                    "speaker_id": "joey",
                    "spoken_text": line,
                    "delivery": "conversational",
                    "physical_cue": "Joey gestures once.",
                    "priority": "high",
                }] if line else []),
                "camera_plan": {
                    "framing": "medium shot",
                    "movement": "slow push in",
                    "movement_intensity": "subtle",
                },
                "audio_plan": {
                    "mode": "dialogue_driven" if line else "ambient_only",
                    "ambience": "cup clinks",
                    "effects": [],
                    "timing_anchor": "audio" if line else "video",
                    "lip_sync_critical": bool(line),
                },
                "ending_beat": "Joey settles back",
                "closing_blocking": "Joey remains seated screen-left",
                "video_prompt": (
                    "integrated_multimodal_description: Joey speaks: "
                    f"<d>[English] {line}</d> "
                    "overall_soundscape: Cup clinks. non_diegetic_music: N/A."
                    if line else
                    "integrated_multimodal_description: Joey reacts silently. "
                    "overall_soundscape: Cup clinks. non_diegetic_music: N/A."
                ),
                "multishot": False,
                "window_prompts": [],
            }

        def generate(**kwargs):
            calls.append(kwargs)
            if "acclaimed screenwriter" in kwargs["system_prompt"]:
                return f"INT. CAFE - DAY\n\nJOEY\n{spoken}"
            if "H3 WHOLE-PLAN REPAIR" in kwargs["prompt"]:
                altered = spoken.replace("twenty-five", "different-ending")
                return json.dumps([
                    # The repair model is allowed to reorganize shots, but its
                    # rewritten dialogue must never become authoritative. It
                    # may also return the line over budget a second time; the
                    # deterministic scheduler must fit it without another LLM
                    # call, truncation, or a fatal pre-generation error.
                    shot("Long complete line", 10, altered),
                    shot("Silent reaction", 10),
                ])
            return json.dumps([
                shot("Overstuffed line", 10, spoken),
                shot("Silent reaction", 10),
            ])

        planner = ShortFilmPlanner(
            llm_generate=generate,
            llm_generate_streaming=generate,
        )
        plan = planner.plan(
            story_description="Joey tells one complete story in a cafe.",
            target_duration=20,
            target_scenes=2,
            video_model="minimax_h3",
            shot_image_policy="prompt_only",
            fps=24,
            frames_steps=17,
            frames_minimum=124,
            frames_maximum=345,
        )

        pass2_calls = [
            call for call in calls
            if "breaking a screenplay into shots" in call["system_prompt"]
        ]
        self.assertEqual(len(pass2_calls), 1)
        self.assertNotIn("H3 WHOLE-PLAN REPAIR", pass2_calls[0]["prompt"])
        self.assertGreaterEqual(pass2_calls[0]["max_new_tokens"], 12288)
        self.assertEqual(plan.shots[0].dialogue_beats[0].spoken_text, spoken)
        self.assertIn(f"<d>[English] {spoken}</d>", plan.shots[0].video_prompt)
        self.assertNotIn("different-ending", plan.shots[0].video_prompt)
        self.assertNotIn("<d><d>", plan.shots[0].video_prompt)
        self.assertGreater(plan.shots[0].duration_sec, 12)


if __name__ == "__main__":
    unittest.main()
