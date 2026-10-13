"""Reference identities must bind to existing final H3 subjects safely."""

from __future__ import annotations

import re
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.h3_dialogue import (  # noqa: E402
    H3DialogueContractError,
    _mapped_ref2va_subject,
    _collapse_ref2va_compiler_identity_duplicates,
    compile_h3_clip_plans,
    validate_final_h3_subject_bindings,
)


def _plan(subjects, references, prompt):
    return {
        "video_prompt": prompt,
        "_director_h3_source_prompt": prompt,
        "_director_h3_model_family": "ref2va",
        "_director_h3_prompt_mode": "ref2va",
        "_director_duration_sec": 6,
        "_director_subjects_on_screen": subjects,
        "_director_physical_state": {
            "opening": [{
                "subject_id": "clip_local_blocking_slot",
                "predicate": "posture",
                "value": "leaning forward",
            }],
            "transitions": [],
            "closing": [],
        },
        "_director_h3_reference_manifest": references,
    }


def _subject_definition_lines(prompt):
    field = re.search(
        r"(?ms)^subject_definitions:\s*(.*?)\n\nsummary:",
        prompt,
    )
    if not field:
        return {}
    return {
        int(match[1]): match[2]
        for match in re.finditer(
            r"(?m)^<Subject\s+(\d+)>\s+is\s+([^\n]*)",
            field.group(1),
        )
    }


class H3FinalSubjectBindingTests(unittest.TestCase):
    def test_qualified_identity_names_bind_without_appending_another_subject(self):
        references = [{"type": "image", "role": "Kitten"}, {"type": "image", "role": "Dragon"}]
        for name in ("Northern Kingdom Dragon", "northernkingdom_dragon", "nk_realm_dragon"):
            with self.subTest(name=name):
                subjects = [
                    {"character_id": "kitten", "speaker_name": "Kitten", "visual_description": "a fluffy kitten"},
                    {"character_id": name, "speaker_name": name,
                     "visual_description": "a massive creature covered in dark scales"},
                ]
                plan = _plan(subjects, references, "[Shot 1] The kitten rides the dragon over a valley.")
                compile_h3_clip_plans([plan])
                self.assertEqual(set(_subject_definition_lines(plan["video_prompt"])), {1, 2})
                self.assertEqual(validate_final_h3_subject_bindings(plan["video_prompt"], {1: 1, 2: 2}), [])

    def test_legacy_qualified_identity_duplicate_is_repaired_without_merging_distinct_owners(self):
        references = [{"type": "image", "role": "Kitten"}, {"type": "image", "role": "Dragon"}]
        for name in ("Northern Kingdom Dragon", "northernkingdom_dragon", "nk_realm_dragon"):
            with self.subTest(name=name):
                prompt = (
                    "subject_definitions: <Subject 1> is Kitten: a fluffy kitten; identity comes from <Picture 1>.\n"
                    f"<Subject 2> is {name}: a massive creature covered in dark scales.\n"
                    "<Subject 3> is Dragon, whose facial, bodily, and character identity come from <Picture 2>.\n\n"
                    "summary: [reference generation] A kitten rides a dragon.\n\n"
                    "retention_analysis: Preserve the mapped identities.\n\n"
                    "detailed_description: Fantasy. [Shot 1] <Subject 1> raises its sword while "
                    "<Subject 2> banks left. <Subject 3> continues forward.\n\n"
                    "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A"
                )
                plan = _plan([], [], prompt)
                plan["_director_prompt_user_edited"] = True
                compile_h3_clip_plans([plan], reference_manifests=[[]])
                compile_h3_clip_plans([plan], reference_manifests=[references])
                self.assertNotIn("<Subject 3>", plan["video_prompt"])
                self.assertIn("banks left", plan["video_prompt"])
                self.assertIn("continues forward", plan["video_prompt"])
                self.assertEqual(validate_final_h3_subject_bindings(plan["video_prompt"], {1: 1, 2: 2}), [])
                distinct_owner = prompt.replace(
                    "a massive creature covered in dark scales.",
                    "a massive creature covered in dark scales; identity comes from <Picture 1>.",
                )
                self.assertEqual(_collapse_ref2va_compiler_identity_duplicates(distinct_owner, references), distinct_owner)

        with self.assertRaisesRegex(H3DialogueContractError, "multiple visible subjects"):
            _mapped_ref2va_subject("Dragon", [
                {"speaker_name": "Northern Kingdom Dragon"},
                {"speaker_name": "Southern Kingdom Dragon"},
            ])
        other_character = prompt.replace(name, "Dragon Slayer")
        self.assertEqual(_collapse_ref2va_compiler_identity_duplicates(other_character, references), other_character)

    def test_semantic_cast_ids_bind_reference_without_adding_a_second_identity(self):
        subjects = [
            {"character_id": "kitten_character", "speaker_name": "Kitten",
             "visual_description": "A fluffy kitten wearing silver armor"},
            {"character_id": "dragon_creature", "speaker_name": "dragon_creature",
             "visual_description": "A scary-looking, highly detailed dragon with dark scales"},
        ]
        original = copy.deepcopy(subjects)
        references = [
            {"type": "image", "role": "Kitten"},
            {"type": "image", "role": "Dragon"},
        ]
        plan = _plan(subjects, references, "[Shot 1] The kitten raises its left paw beside the dragon.")
        compile_h3_clip_plans([plan])
        definitions = _subject_definition_lines(plan["video_prompt"])
        self.assertEqual(set(definitions), {1, 2})
        self.assertIn("scary-looking", definitions[2])
        self.assertIn("<Picture 2>", definitions[2])
        self.assertNotIn("<Subject 3>", plan["video_prompt"])
        self.assertEqual(subjects, original)
        self.assertEqual(validate_final_h3_subject_bindings(plan["video_prompt"], {1: 1, 2: 2}), [])

    def test_legacy_reference_duplicate_is_repaired_across_both_rerun_preflights(self):
        references = [{"type": "image", "role": "Kitten"}, {"type": "image", "role": "Dragon"}]
        for existing_picture in ("", "; identity comes from <Picture 2>"):
            with self.subTest(existing_picture=existing_picture):
                prompt = (
                    "subject_definitions: <Subject 1> is Kitten: a fluffy kitten; identity comes from <Picture 1>.\n"
                    f"<Subject 2> is dragon_creature: a scary{existing_picture}.\n"
                    "<Subject 3> is Dragon, whose facial, bodily, and character identity come from <Picture 2>.\n\n"
                    "summary: [reference generation] The kitten approaches the dragon.\n\n"
                    "retention_analysis: Preserve both reference identities.\n\n"
                    "detailed_description: Cinematic fantasy. [Shot 1] <Subject 1> raises its left paw. "
                    "<Subject 2> (dragon_creature) lowers its head. <Subject 3> (Dragon) rests on the grass.\n\n"
                    "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A"
                )
                plan = _plan([
                    {"character_id": "kitten", "speaker_name": "Kitten", "visual_description": "a fluffy kitten"},
                    {"character_id": "dragon_creature", "speaker_name": "dragon_creature", "visual_description": "a scary"},
                ], [], prompt)
                plan["_director_prompt_user_edited"] = True
                compile_h3_clip_plans([plan], reference_manifests=[[]])
                self.assertEqual(set(_subject_definition_lines(plan["video_prompt"])), {1, 2})
                self.assertEqual(validate_final_h3_subject_bindings(plan["video_prompt"]), [])
                compile_h3_clip_plans([plan], reference_manifests=[references])
                result = plan["video_prompt"]
                self.assertEqual(set(_subject_definition_lines(result)), {1, 2})
                self.assertNotIn("<Subject 3>", result)
                self.assertIn("raises its left paw", result)
                self.assertIn("lowers its head", result)
                self.assertIn("rests on the grass", result)
                self.assertEqual(validate_final_h3_subject_bindings(result, {1: 1, 2: 2}), [])
                compile_h3_clip_plans([plan], reference_manifests=[references])
                self.assertEqual(plan["video_prompt"], result)

    def test_legacy_repair_preserves_authored_and_ambiguous_cast_definitions(self):
        prompt = (
            "subject_definitions: <Subject 1> is dragon_creature: a red dragon.\n"
            "<Subject 2> is Dragon, a blue dragon with a scar; identity comes from <Picture 1>.\n\n"
            "summary: [reference generation] Two different dragons.\n\n"
            "retention_analysis: Preserve distinct identities.\n\n"
            "detailed_description: Fantasy. [Shot 1] The two dragons stand apart.\n\n"
            "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A"
        )
        references = [{"type": "image", "role": "Dragon"}]
        self.assertEqual(_collapse_ref2va_compiler_identity_duplicates(prompt, references), prompt)
        with self.assertRaisesRegex(H3DialogueContractError, "multiple visible subjects"):
            _mapped_ref2va_subject("Dragon", [
                {"character_id": "dragon_creature", "speaker_name": "dragon_creature"},
                {"character_id": "dragon_actor", "speaker_name": "dragon_actor"},
            ])

    def test_reference_only_identity_is_preserved_when_no_cast_member_matches(self):
        references = [{"type": "image", "role": "Kitten"}, {"type": "image", "role": "Dragon"}]
        plan = _plan([
            {"character_id": "kitten", "speaker_name": "Kitten", "visual_description": "a fluffy kitten"},
        ], references, "[Shot 1] The kitten sings alone on the grassy ridge.")
        compile_h3_clip_plans([plan])
        first = plan["video_prompt"]
        self.assertEqual(set(_subject_definition_lines(first)), {1, 2})
        self.assertIn("<Picture 2>", _subject_definition_lines(first)[2])
        plan["_director_prompt_user_edited"] = True
        compile_h3_clip_plans([plan])
        self.assertEqual(set(_subject_definition_lines(plan["video_prompt"])), {1, 2})
        self.assertEqual(validate_final_h3_subject_bindings(plan["video_prompt"], {1: 1, 2: 2}), [])

    def test_explicit_plural_cast_is_never_collapsed_as_a_legacy_duplicate(self):
        prompt = (
            "subject_definitions: <Subject 1> is dragon_creature: a red dragon.\n"
            "<Subject 2> is Dragon, whose facial, bodily, and character identity come from <Picture 1>.\n\n"
            "summary: [reference generation] Two dragons meet.\n\n"
            "retention_analysis: Preserve both dragons.\n\n"
            "detailed_description: Fantasy. [Shot 1] Two dragons stand apart. "
            "<Subject 1> rests while <Subject 2> walks around it.\n\n"
            "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A"
        )
        references = [{"type": "image", "role": "Dragon"}]
        for description in (
            "Two dragons stand apart.", "Both dragons stand apart.",
            "Two distinct dragons stand apart.", "Dragon and another dragon stand apart.",
            "A second Dragon stands alongside the first.",
        ):
            with self.subTest(description=description):
                variant = prompt.replace("Two dragons meet.", description).replace("Two dragons stand apart.", description)
                variant = variant.replace("Preserve both dragons.", "Preserve the described identities.")
                self.assertEqual(_collapse_ref2va_compiler_identity_duplicates(variant, references), variant)
                self.assertEqual(_collapse_ref2va_compiler_identity_duplicates(
                    variant, [{"type": "image", "role": "the Dragon"}],
                ), variant)
                self.assertEqual(_collapse_ref2va_compiler_identity_duplicates(variant, []), variant)

    def test_legacy_repair_respects_each_clips_actual_prompt_mode(self):
        prompt = (
            "subject_definitions: <Subject 1> is dragon_creature: a red dragon.\n"
            "<Subject 2> is Dragon, whose facial, bodily, and character identity come from <Picture 1>.\n\n"
            "summary: [reference generation] A dragon rests.\n\n"
            "retention_analysis: Preserve the dragon.\n\n"
            "detailed_description: Fantasy. [Shot 1] <Subject 1> rests. <Subject 2> watches the ridge.\n\n"
            "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A"
        )
        first = _plan([{"character_id": "dragon_creature", "speaker_name": "dragon_creature"}], [], prompt)
        first["_director_prompt_user_edited"] = True
        second = copy.deepcopy(first)
        compile_h3_clip_plans(
            [first, second], prompt_modes=["ref2va", "t2va"],
            reference_manifests=[[{"type": "image", "role": "Dragon"}], []],
        )
        self.assertNotIn("<Subject 2>", first["_director_h3_source_prompt"])
        self.assertIn("<Subject 2>", second["_director_h3_source_prompt"])

    def test_sanitized_captured_kitten_reference_binds_native_subject_and_is_idempotent(self):
        subjects = [{
            "character_id": "director_identity_the_kitten",
            "speaker_name": "the Kitten",
            "visual_description": (
                "a cute, fluffy tabby kitten with pink bow on its head"
            ),
            "wardrobe": "silver plate armor and a flowing cape",
        }, {
            "character_id": "character_2",
            "speaker_name": "Game of Thrones style dragon",
            "visual_description": (
                "a scary looking, highly detailed dragon covered in dark scales"
            ),
        }]
        references = [{
            "type": "image",
            "image_intent": "identity",
            "role": "Kitten",
        }]
        plan = _plan(
            subjects,
            references,
            "[Shot 1] The Kitten climbs onto the Dragon's neck in a misty field.",
        )

        compile_h3_clip_plans([plan], reference_manifests=[references])

        prompt = plan["video_prompt"]
        definitions = _subject_definition_lines(prompt)
        self.assertEqual(set(definitions), {1, 2})
        self.assertIn("<Picture 1>", definitions[1])
        self.assertNotIn("<Picture 1>", definitions[2])
        self.assertIn(
            "<Subject 1> (appears in [Shot 1]): fully_preserved",
            prompt,
        )
        self.assertIn(
            "<Subject 1> uses facial, bodily, and character identity from <Picture 1>",
            prompt,
        )
        self.assertEqual(
            [subject["character_id"] for subject in subjects],
            ["director_identity_the_kitten", "character_2"],
        )
        self.assertEqual(
            validate_final_h3_subject_bindings(prompt, {1: 1}),
            [],
        )

        compile_h3_clip_plans([plan], reference_manifests=[references])
        self.assertEqual(plan["video_prompt"], prompt)

    def test_reference_alias_uses_identity_noun_when_display_name_differs(self):
        subjects = [{
            "character_id": "native_actor_41",
            "speaker_name": "Lead",
            "visual_description": (
                "a cute, fluffy tabby kitten with bright eyes and a pink bow"
            ),
        }]
        references = [{
            "type": "image",
            "image_intent": "identity",
            "role": "Kitten",
        }]
        plan = _plan(
            subjects,
            references,
            "[Shot 1] The kitten trots through a grassy field.",
        )

        compile_h3_clip_plans([plan], reference_manifests=[references])

        definitions = _subject_definition_lines(plan["video_prompt"])
        self.assertEqual(set(definitions), {1})
        self.assertIn("<Picture 1>", definitions[1])
        self.assertEqual(subjects[0]["character_id"], "native_actor_41")

    def test_generic_rider_and_vehicle_references_bind_to_their_own_native_subjects(self):
        subjects = [{
            "character_id": "native_rider_9",
            "speaker_name": "the Rider",
            "visual_description": "an armored rider in a tan coat",
        }, {
            "character_id": "native_vehicle_4",
            "speaker_name": "the Vehicle",
            "visual_description": "a red utility vehicle with a roof rack",
        }]
        references = [{
            "type": "image",
            "image_intent": "identity",
            "role": "Rider",
        }, {
            "type": "image",
            "image_intent": "identity",
            "role": "Vehicle",
        }]
        plan = _plan(
            subjects,
            references,
            "[Shot 1] The Rider drives the Vehicle along a country road.",
        )

        compile_h3_clip_plans([plan], reference_manifests=[references])

        definitions = _subject_definition_lines(plan["video_prompt"])
        self.assertEqual(set(definitions), {1, 2})
        self.assertIn("<Picture 1>", definitions[1])
        self.assertNotIn("<Picture 2>", definitions[1])
        self.assertIn("<Picture 2>", definitions[2])
        self.assertNotIn("<Picture 1>", definitions[2])
        self.assertEqual(
            validate_final_h3_subject_bindings(
                plan["video_prompt"],
                {1: 1, 2: 2},
            ),
            [],
        )

    def test_shared_generic_alias_is_ambiguous_and_pronouns_do_not_select_a_subject(self):
        subjects = [{
            "character_id": "native_guest_1",
            "speaker_name": "Guest One",
            "visual_description": "a woman with dark hair and a green coat",
        }, {
            "character_id": "native_guest_2",
            "speaker_name": "Guest Two",
            "visual_description": "a woman with dark hair and a blue coat",
        }]
        self.assertIsNone(_mapped_ref2va_subject("she", subjects))
        plan = _plan(
            subjects,
            [{"type": "image", "image_intent": "identity", "role": "woman"}],
            "[Shot 1] The two guests stand together in a station.",
        )

        with self.assertRaisesRegex(
            H3DialogueContractError,
            "matches multiple visible subjects",
        ):
            compile_h3_clip_plans(
                [plan],
                reference_manifests=[plan["_director_h3_reference_manifest"]],
            )

    def test_stage3_unnamed_cast_keeps_stable_ids_and_distinct_reference_owners(self):
        references = [{
            "type": "image",
            "image_intent": "identity",
            "role": "drummer",
        }, {
            "type": "image",
            "image_intent": "identity",
            "role": "guitarist",
        }]
        plans = []
        for clip_index in range(3):
            first_clip = clip_index == 0
            subjects = [{
                "character_id": "drummer",
                "speaker_name": "Unnamed" if first_clip else "drummer",
                "visual_description": "A drummer keeps time behind a compact kit",
            }, {
                "character_id": "guitarist",
                "speaker_name": "N/A" if first_clip else "guitarist",
                "visual_description": "A guitarist plays a bright electric guitar",
            }]
            plan = _plan(
                subjects,
                references if first_clip else [],
                (
                    "[Shot 1] The drummer keeps time while the guitarist plays "
                    "a bright chord in a rehearsal room. Camera: a steady "
                    "medium two-shot. Action: both musicians continue playing. "
                    "Lighting: warm stage light. Mood: focused. Final beat: "
                    "the music carries on."
                ),
            )
            plan["_director_dialogue_beats"] = [{
                "speaker_id": "drummer",
                "spoken_text": "Keep the beat steady.",
            }]
            if first_clip:
                # Saved registries can predate the fixed cast names too.
                plan["_director_speaker_registry"] = {
                    "drummer": {"stable_id": "(S7)", "speaker_name": "Unnamed"},
                    "guitarist": {"stable_id": "(S8)", "speaker_name": "N/A"},
                }
            plans.append(plan)

        first_subjects = plans[0]["_director_subjects_on_screen"]
        self.assertIsNone(_mapped_ref2va_subject("Unnamed", first_subjects))
        self.assertIsNone(_mapped_ref2va_subject("N/A", first_subjects))
        self.assertEqual(_mapped_ref2va_subject("drummer", first_subjects), 0)
        self.assertEqual(_mapped_ref2va_subject("guitarist", first_subjects), 1)

        compile_h3_clip_plans(
            plans,
            reference_manifests=[references, [], []],
        )

        for plan in plans:
            self.assertEqual(
                [
                    subject["character_id"]
                    for subject in plan["_director_subjects_on_screen"]
                ],
                ["drummer", "guitarist"],
            )
            self.assertEqual(
                [
                    subject["speaker_name"]
                    for subject in plan["_director_subjects_on_screen"]
                ],
                ["drummer", "guitarist"],
            )
            prompt = plan["video_prompt"]
            definitions = _subject_definition_lines(prompt)
            self.assertEqual(set(definitions), {1, 2})
            self.assertIn("drummer", definitions[1].casefold())
            self.assertIn("guitarist", definitions[2].casefold())
            self.assertNotRegex(
                prompt,
                r"(?i)(?:Unnamed|N/A)\s+\(S\d+\)\s+(?:speaks|:)",
            )
            self.assertNotIn("Unnamed", prompt)
            self.assertEqual(prompt.count("N/A"), 1)

        first_prompt = plans[0]["video_prompt"]
        first_definitions = _subject_definition_lines(first_prompt)
        self.assertIn("<Picture 1>", first_definitions[1])
        self.assertNotIn("<Picture 2>", first_definitions[1])
        self.assertIn("<Picture 2>", first_definitions[2])
        self.assertNotIn("<Picture 1>", first_definitions[2])
        self.assertNotIn(
            "Canonical identity and world: <Subject 2> (<Subject 3> (Unnamed))",
            first_prompt,
        )
        compiled_prompts = [plan["video_prompt"] for plan in plans]
        compile_h3_clip_plans(
            plans,
            reference_manifests=[references, [], []],
        )
        self.assertEqual(
            [plan["video_prompt"] for plan in plans],
            compiled_prompts,
        )

    def test_final_binding_validator_rejects_wrong_or_duplicate_owners(self):
        prompt = (
            "subject_definitions: <Subject 1> is the Rider.\n"
            "<Subject 2> is the Vehicle, whose identity comes from <Picture 1>.\n\n"
            "summary: [reference generation] A rider uses a vehicle.\n\n"
            "retention_analysis: preserve the mapped identity.\n\n"
            "detailed_description: [Shot 1] The Rider drives the Vehicle.\n\n"
            "overall_soundscape: Road ambience.\n\n"
            "non_diegetic_music: N/A"
        )
        wrong_owner = validate_final_h3_subject_bindings(prompt, {1: 1})
        self.assertEqual(len(wrong_owner), 1)
        self.assertIn("must bind only to <Subject 1>", wrong_owner[0])

        duplicate = prompt.replace(
            "<Subject 1> is the Rider.",
            "<Subject 1> is the Rider, whose identity comes from <Picture 1>.",
        )
        duplicate_errors = validate_final_h3_subject_bindings(
            duplicate,
            {1: 1},
        )
        self.assertEqual(len(duplicate_errors), 1)
        self.assertIn("found subject bindings [1, 2]", duplicate_errors[0])


if __name__ == "__main__":
    unittest.main()
