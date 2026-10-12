"""Reference identities must bind to existing final H3 subjects safely."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.h3_dialogue import (  # noqa: E402
    H3DialogueContractError,
    _mapped_ref2va_subject,
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
