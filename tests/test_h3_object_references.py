"""Object images preserve design without occupying character or voice slots."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.minimax_h3.reference_manifest import validate_reference_manifest
from models.minimax_h3.ref2va import (
    ensure_ref2va_prompt_relationships,
    prepare_references,
)
from services.director.h3_dialogue import compile_h3_official_prompt
from services.director.reference_inputs import planning_reference_inputs
from services.h3_reference_scope import prepare_scoped_reference_context
from services.h3_sequence_planner import (
    _image_reference_roles,
    _reference_context,
    build_manual_h3_reference_sequence_plan,
)
from services.h3_story_ledger import _reference_h3_cast_names
from services.llm_service import (
    _canonical_h3_ref2va_subject_fields,
    _h3_ref2va_reference_contract_satisfied,
    _parse_h3_ref2va_subject_manifest,
)


class H3ObjectReferenceTests(unittest.TestCase):
    def setUp(self):
        self.object = {
            "type": "image", "path": "atat.png", "role": "AT-AT",
            "image_intent": "object",
        }
        self.character = {
            "type": "image", "path": "chewy.png", "role": "Chewy",
            "library_character_id": "chewy", "character_name": "Chewy",
        }
        self.voice = {
            "type": "audio", "path": "chewy.wav", "role": "Chewy",
            "audio_intent": "voice", "library_character_id": "chewy",
        }

    @staticmethod
    def structured(definitions, retention, action="Two AT-ATs advance across the desert."):
        return (
            f"subject_definitions: {definitions}\n"
            "summary: [reference generation] The requested scene.\n"
            f"retention_analysis: {retention}\n"
            f"detailed_description: [Shot 1] {action}\n"
            "overall_soundscape: Wind and synchronized footsteps.\n"
            "non_diegetic_music: N/A"
        )

    def test_object_manifest_keeps_existing_limits_and_explicit_background_choice(self):
        item = validate_reference_manifest([self.object], require_files=False)[0]
        self.assertEqual(item["image_intent"], "object")
        self.assertFalse(item["remove_background"])
        isolated = {**self.object, "remove_background": True}
        self.assertTrue(validate_reference_manifest([isolated], require_files=False)[0]["remove_background"])
        for intent in ("scene", "style", "composition"):
            with self.subTest(intent=intent):
                item = validate_reference_manifest(
                    [{**isolated, "image_intent": intent}], require_files=False,
                )[0]
                self.assertFalse(item["remove_background"])
        self.assertEqual(len(validate_reference_manifest([self.object] * 9, require_files=False)), 9)
        with self.assertRaisesRegex(ValueError, "at most 9 image references"):
            validate_reference_manifest([self.object] * 10, require_files=False)
        with self.assertRaisesRegex(ValueError, "invalid image intent"):
            validate_reference_manifest([{**self.object, "image_intent": "unknown"}], require_files=False)

    def test_object_only_raw_request_preserves_design_and_requested_multiple_instances(self):
        source = "Two towering AT-ATs advance across the desert and fire their blasters."
        prompt = ensure_ref2va_prompt_relationships(source, [self.object])
        self.assertIn(source, prompt)
        self.assertIn("<Picture 1> is an object / prop reference for AT-AT", prompt)
        self.assertIn("shape, proportions, materials, colors", prompt)
        self.assertIn("count, scale, placement, and action", prompt)
        self.assertIn("<Picture 1>: fully_preserved", prompt)
        self.assertNotIn("<Subject", prompt)
        self.assertNotIn("voice-timbre", prompt)

    def test_object_position_does_not_renumber_characters_or_cross_wire_voice(self):
        for references, picture in (
            ([self.object, self.character, self.voice], 2),
            ([self.character, self.object, self.voice], 1),
        ):
            with self.subTest(character_picture=picture):
                prompt = ensure_ref2va_prompt_relationships(
                    "Chewy runs while two AT-ATs fire.", references,
                )
                self.assertIn(f"<Subject 1> is Chewy from <Picture {picture}>", prompt)
                self.assertEqual(set(re.findall(r"<Subject (\d+)>", prompt)), {"1"})
                self.assertRegex(prompt, r"<Audio 1>[^.\n]+<Subject 1>")

    def test_queued_inventory_excludes_objects_from_cast_and_voice_fallback(self):
        relationships, retention, _ = _reference_context([
            self.object, self.character, self.voice,
        ])
        context = relationships + "\n" + retention
        self.assertEqual(_reference_h3_cast_names(context), ["Chewy"])
        manifest = _parse_h3_ref2va_subject_manifest(context)
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["pictures"], ["<Picture 2>"])
        self.assertEqual(manifest[0]["audios"], ["<Audio 1>"])
        definitions, retained = _canonical_h3_ref2va_subject_fields(context)
        self.assertIn("<Picture 1> is an object / prop reference for AT-AT", definitions)
        self.assertIn("<Picture 1>: fully_preserved", retained)
        self.assertTrue(_h3_ref2va_reference_contract_satisfied(
            self.structured(definitions, retained), context,
        ))

    def test_enhance_now_object_markers_override_identity_words_and_saved_metadata(self):
        context = (
            'Saved character "AT-AT" is exactly <Subject 1>: <Picture 1> all define this one stable character.\n'
            "<Picture 1>: object / prop design and appearance reference for AT-AT; "
            "intent=OBJECT REFERENCE; image_intent=object; retention=fully_preserved\n"
            "<Picture 2>: visual identity/appearance reference for Chewy; retention=reference\n"
            "<Audio 1>: Chewy voice; intent=VOICE REFERENCE; retention=reference\n"
        )
        manifest = _parse_h3_ref2va_subject_manifest(context)
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["name"], "Chewy")
        self.assertEqual(manifest[0]["index"], 1)
        self.assertEqual(manifest[0]["pictures"], ["<Picture 2>"])
        self.assertEqual(manifest[0]["audios"], ["<Audio 1>"])
        definitions, retained = _canonical_h3_ref2va_subject_fields(context)
        self.assertIn("<Picture 1> is an object / prop reference for AT-AT", definitions)
        self.assertIn("<Picture 1>: fully_preserved", retained)
        self.assertEqual(set(re.findall(r"<Subject (\d+)>", definitions)), {"1"})
        self.assertTrue(_h3_ref2va_reference_contract_satisfied(
            self.structured(definitions, retained), context,
        ))

    def test_machine_intent_alone_and_native_object_only_context_do_not_invent_characters(self):
        contexts = [
            "<Picture 1>: AT-AT; image_intent=object; retention=fully_preserved",
            "\n".join(_reference_context([self.object])[:2]),
        ]
        for context in contexts:
            with self.subTest(context=context):
                self.assertEqual(_parse_h3_ref2va_subject_manifest(context), [])
                definitions, retained = _canonical_h3_ref2va_subject_fields(context)
                self.assertNotIn("<Subject", definitions)
                self.assertIn("object / prop", definitions)
                self.assertIn("<Picture 1>: fully_preserved", retained)

    def test_tagged_structured_manual_prompt_keeps_contract_in_the_correct_fields(self):
        source = self.structured(
            "<Subject 1> is Chewy from <Picture 1>.",
            "<Subject 1>: fully_preserved",
            "Chewy runs while two AT-ATs matching <Picture 2> fire from behind.",
        )
        prompt = ensure_ref2va_prompt_relationships(source, [self.character, self.object])
        definitions = prompt.split("summary:", 1)[0]
        retained = prompt.split("retention_analysis:", 1)[1].split("detailed_description:", 1)[0]
        self.assertIn("<Picture 2> is an object / prop reference for AT-AT", definitions)
        self.assertIn("<Picture 2>: fully_preserved", retained)
        self.assertIn("two AT-ATs matching <Picture 2> fire from behind", prompt)
        self.assertEqual(ensure_ref2va_prompt_relationships(prompt, [self.character, self.object]), prompt)
        self.assertEqual(set(re.findall(r"<Subject (\d+)>", prompt)), {"1"})

    def test_three_manual_windows_keep_objects_and_reusable_sound_effects(self):
        references = [
            {"type": "video", "path": "running.mp4", "include_audio": False},
            self.character, self.object,
            {"type": "audio", "path": "blaster.wav", "role": "Laser blaster", "audio_intent": "sound"},
        ]
        source = (
            "Chewy runs in <Video 1> while two AT-ATs matching <Picture 2> "
            "advance and fire with blaster sounds from <Audio 1>."
        )
        result = build_manual_h3_reference_sequence_plan(
            "\n".join([source] * 3), model_type="minimax_h3_ref2va_fused_turbo",
            resolution="704x1280", total_frames=835, references=references,
            max_clip_frames=294, overlap_frames=18, native_continuation=True,
        )
        self.assertEqual(result["window_count"], 3)
        for window_prompt in result["window_prompts"]:
            prepared = ensure_ref2va_prompt_relationships(window_prompt, references)
            self.assertIn("<Picture 2> is an object / prop reference for AT-AT", prepared)
            self.assertIn("<Picture 2>: fully_preserved", prepared)
            self.assertIn("timbre and texture of <Audio 1>", prepared)
            self.assertNotIn("voice-timbre", prepared)

    def test_director_object_named_after_character_does_not_become_another_subject(self):
        references = [
            self.character, {**self.object, "role": "Chewy's blaster"},
        ]
        prompt, _ = compile_h3_official_prompt(
            "Chewy carries two blasters matching <Picture 2> and fires at the walkers.",
            [{"character_id": "chewy", "speaker_name": "Chewy"}], [],
            mode="ref2va", duration_seconds=12.25, references=references,
        )
        self.assertIn("<Picture 2> is an object / prop reference for Chewy's blaster", prompt)
        self.assertIn("<Picture 2>: fully_preserved", prompt)
        self.assertEqual(set(re.findall(r"<Subject (\d+)>", prompt)), {"1"})
        self.assertIn("count, scale, placement, and action", prompt)
        self.assertNotIn("character identity come from <Picture 2>", prompt)

    def test_story_image_roles_keep_object_pixels_and_picture_order(self):
        references = [self.object, self.character]
        paths = ["chewy.png", "atat.png"]
        roles = _image_reference_roles(references, paths)
        self.assertEqual(
            [(item["image_index"], item["image_intent"]) for item in roles],
            [(2, "identity"), (1, "object")],
        )
        relationships, _, _ = _reference_context(references)
        def unexpected_summary(*_args, **_kwargs):
            self.fail("Object design must not enter the scene/style-only summary filter.")
        result = prepare_scoped_reference_context(
            unexpected_summary, "Chewy runs from two walkers.", relationships, paths, roles,
        )
        self.assertFalse(result["scoped"])
        self.assertEqual(result["image_paths"], paths)
        self.assertEqual(result["context"], relationships)

    def test_director_cast_planning_does_not_describe_object_as_a_person(self):
        with tempfile.TemporaryDirectory() as directory:
            object_path = Path(directory) / "atat.png"
            character_path = Path(directory) / "chewy.png"
            object_path.touch()
            character_path.touch()
            references = [
                {**self.object, "path": str(object_path)},
                {**self.character, "path": str(character_path)},
            ]
            result = planning_reference_inputs({"minimax_h3_references": references})
            self.assertEqual(result["reference_image_path"], str(character_path))
            self.assertEqual(result["character_ref_paths"], [])
            self.assertEqual(result["location_ref_paths"], [])
            self.assertEqual(references[0]["image_intent"], "object")
            object_only = planning_reference_inputs({"minimax_h3_references": references[:1]})
            self.assertIsNone(object_only["reference_image_path"])
            self.assertEqual(object_only["character_ref_paths"], [])

    def test_real_object_image_preparation_honors_isolation_without_overwriting_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            originals = []
            for name, color in (("isolated.png", "red"), ("plain.png", "green"), ("scene.png", "blue")):
                path = Path(directory) / name
                Image.new("RGB", (32, 32), color).save(path)
                paths.append(path)
                originals.append(path.read_bytes())
            references = [
                {**self.object, "path": str(paths[0]), "remove_background": True},
                {**self.object, "path": str(paths[1])},
                {**self.object, "path": str(paths[2]), "image_intent": "scene", "remove_background": True},
            ]
            with patch(
                "models.minimax_h3.ref2va.isolate_reference_image_background",
                return_value=Image.new("RGB", (32, 32), "white"),
            ) as isolate:
                prepared = prepare_references(
                    references, num_frames=24, target_height=32, target_width=32,
                )
            isolate.assert_called_once_with(str(paths[0]))
            self.assertEqual([item.image_intent for item in prepared], ["object", "object", "scene"])
            self.assertEqual(
                [item.image.getpixel((0, 0)) for item in prepared],
                [(255, 255, 255), (0, 128, 0), (0, 0, 255)],
            )
            self.assertEqual([path.read_bytes() for path in paths], originals)

    def test_object_isolation_failure_preserves_a_valid_original_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atat.png"
            Image.new("RGB", (32, 32), "blue").save(path)
            original = path.read_bytes()
            with patch(
                "models.minimax_h3.ref2va.isolate_reference_image_background",
                side_effect=RuntimeError("background service unavailable"),
            ):
                prepared = prepare_references(
                    [{**self.object, "path": str(path), "remove_background": True}],
                    num_frames=24, target_height=32, target_width=32,
                )
            self.assertEqual(prepared[0].image_intent, "object")
            self.assertEqual(prepared[0].image.getpixel((0, 0)), (0, 0, 255))
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
