"""Production regressions: sparse conversations, topic coverage, RefMod cast."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.dialogue_writing import dialogue_topic_covered, requested_dialogue_topics
from services.h3_dialogue_writing import _shorten_generated_dialogue, complete_creative_dialogue, creative_dialogue_windows
from services.h3_story_ledger import (
    _canonicalize_story_ledger, _deterministic_ledger, _merge_h3_cast_names,
    canonicalize_h3_reference_names, extract_h3_source_intent,
)
from services.h3_sequence_planner import compile_h3_reference_sequence_prompts, compute_h3_native_sequence_windows


KITCHEN = """Sydney Sweeney, Ana De Armas, and Scarlett Johansson are all in a modern luxury kitchen wearing aprons. They are discussing how exciting it is that Maestro v2.1.0 is live!
Features include:
• Streamlined UI & Menus
• Viggle Animate Support!
• Save & share characters w/ RefMod support
• H3 Voice Audio: generate long dialogue w/ SFX
• H3 Face Refinement
• DLSS 5 upscaling"""
CAST = ["Sydney Sweeney", "Ana De Armas", "Scarlett Johansson"]
PACKAGED = ["minimaxh3_sydneysweeney_v1_refmod", "minimaxh3_anadearmas_v1_refmod", "minimaxh3_scarlettjohansson_v1_refmod"]
CONTEXT = " ".join(f"<Subject {i}> is {name} from <Video {i}>, preserving identity. <Audio {i}> is the voice-timbre reference for <Subject {i}>." for i, name in enumerate(PACKAGED, 1))
BRIEF = "Maya and Leo discuss how to organize their film project."
LINE_A = "Let's organize the clips by scene first, then save a recipe so tomorrow we can quickly use these same exact settings again."
LINE_B = "Good idea. I'll label our character references while you do that, so we can both find the right versions for each scene."


def speech(text, segment, speaker="Maya"):
    return {"speaker": speaker, "language": "English", "delivery": "naturally", "text": text, "segment": segment}


def fixture(prompt=BRIEF, durations=None, lines=None):
    durations = durations or [10.1, 10.1]
    canonical = _deterministic_ledger(prompt, segment_count=len(durations), segment_durations=durations,
                                    locked_dialogue=[], camera_coverage="multi_shot", reference_context="")
    candidate = deepcopy(canonical)
    candidate["generated_dialogue"] = lines or [speech("Ready?", i) for i in range(1, len(durations) + 1)]
    ledger = _canonicalize_story_ledger(prompt, canonical, candidate, locked_dialogue=[], segment_count=len(durations), allow_generated_dialogue=True)
    return canonical, ledger


class H3CreativeRegressionTests(unittest.TestCase):
    def test_kitchen_cast_does_not_invent_pronoun_or_duplicate_refmods(self):
        intent = extract_h3_source_intent(KITCHEN)
        self.assertEqual(intent["cast_names"], CAST)
        self.assertNotIn("They", intent["proper_names"])
        self.assertEqual(_merge_h3_cast_names(CAST, PACKAGED, prompt=KITCHEN), CAST)
        canonical = _deterministic_ledger(KITCHEN, segment_count=5, segment_durations=[10.125] + [9.375] * 4,
                                        locked_dialogue=[], camera_coverage="multi_shot", reference_context=CONTEXT)
        self.assertEqual(canonical["source_intent"]["cast_names"], CAST)
        self.assertNotIn("prompt-native", canonical["subject_continuity"])
        for index, name in enumerate(CAST, 1):
            self.assertIn(f"<Subject {index}> is {name} from <Video {index}>", canonical["subject_continuity"])

    def test_conversational_verbs_identify_each_speaker(self):
        self.assertEqual(extract_h3_source_intent(BRIEF)["cast_names"], ["Maya", "Leo"])
        for pronoun in ("They", "She", "He", "We", "It"):
            self.assertEqual(extract_h3_source_intent(f"Maya and Leo sit together. {pronoun} looks happy.")["cast_names"], ["Maya", "Leo"])

    def test_ambiguous_refmod_versions_remain_distinct(self):
        refs = [PACKAGED[0], PACKAGED[0].replace("v1", "v2")]
        names = _merge_h3_cast_names([CAST[0]], refs)
        self.assertIn(refs[0], names)
        self.assertIn(refs[1], names)
        context = " ".join(f"<Subject {i}> is {name} from <Video {i}>." for i, name in enumerate(refs, 1))
        self.assertEqual(canonicalize_h3_reference_names(context, [CAST[0]]), context)
        self.assertEqual(_merge_h3_cast_names(["Ann Lee"], ["minimaxh3_joannlee_v1_refmod"]), ["Ann Lee", "minimaxh3_joannlee_v1_refmod"])

    def test_compiler_uses_same_subjects_and_voice_bindings_as_named_dialogue(self):
        plan = {
            "source_prompt": KITCHEN, "source_intent": extract_h3_source_intent(KITCHEN),
            "clips": [{"shots": [{"start_seconds": 0, "end_seconds": 10.125,
                "action": "Sydney Sweeney smiles at Ana De Armas and Scarlett Johansson in the kitchen.",
                "dialogue": [speech("Maestro is live!", 1, CAST[0]), speech("Let's try it together.", 1, CAST[2])]}]}],
        }
        compiled = compile_h3_reference_sequence_prompts(plan, compute_h3_native_sequence_windows(243, window_frames=243, overlap_frames=18),
                                                       reference_relationships=CONTEXT, default_retention="", task_types="video reference")
        text = compiled[0]["prompt"]
        self.assertNotIn("prompt-native", text)
        self.assertNotIn("exactly one They", text)
        self.assertIn("<Subject 1> (S1)", text)
        self.assertIn("<Subject 3> (S2)", text)
        self.assertIn("voice referenced from <Audio 3>", text)

    def test_topics_require_spoken_content_and_preserve_sfx_detail(self):
        topics = requested_dialogue_topics(KITCHEN)
        self.assertEqual(len(topics), 6)
        self.assertFalse(dialogue_topic_covered(topics[2], "Viggle Animate is a game changer!"))
        self.assertTrue(dialogue_topic_covered(topics[2], "Save and share your characters with RefMods."))
        self.assertFalse(dialogue_topic_covered(topics[3], "H3 Voice Audio is amazing for long dialogues."))
        self.assertTrue(dialogue_topic_covered(topics[3], "H3 Voice Audio generates long dialogue with sound effects."))
        self.assertTrue(dialogue_topic_covered(topics[4], "H3 can refine faces with its face refiner."))
        self.assertEqual(requested_dialogue_topics(KITCHEN + "\nNo dialogue."), [])
        self.assertEqual(requested_dialogue_topics(KITCHEN + "\nOnly these exact lines."), [])
        self.assertEqual(requested_dialogue_topics(BRIEF + "\nVisual direction:\n• Warm lighting\n• Soft focus"), [])

    def test_one_failed_window_does_not_discard_another_valid_repair(self):
        canonical, ledger = fixture()
        original = deepcopy(ledger)
        generator = Mock(side_effect=["{}", "{}", json.dumps({"generated_dialogue": [speech(LINE_B, 2, "Leo")]})])
        result, warnings = complete_creative_dialogue(BRIEF, ledger, canonical_ledger=canonical, locked_dialogue=[], durations=[10.1, 10.1], generate=generator, system_prompt="Focused guide")
        self.assertEqual([item["text"] for item in result["generated_dialogue"]], ["Ready?", LINE_B])
        self.assertEqual(warnings, [])  # A short valid line is not a timing failure.
        self.assertEqual(ledger, original)
        self.assertEqual(generator.call_count, 3)

    def test_density_development_is_one_pass_and_keeps_a_valid_shorter_exchange(self):
        canonical, ledger = fixture(lines=[speech("Ready?", 1), speech(LINE_B, 2, "Leo")])
        generator = Mock(side_effect=[json.dumps({"generated_dialogue": [speech("That sounds good!", 1)]}), json.dumps({"generated_dialogue": [speech(LINE_A, 1)]})])
        result, warnings = complete_creative_dialogue(BRIEF, ledger, canonical_ledger=canonical, locked_dialogue=[], durations=[10.1, 10.1], generate=generator, system_prompt="Focused guide")
        self.assertEqual(warnings, [])
        self.assertEqual([item["text"] for item in result["generated_dialogue"]], ["That sounds good!", LINE_B])
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(generator.call_args.kwargs["json_schema"]["properties"]["generated_dialogue"]["items"]["properties"]["speaker"]["enum"], ["Maya", "Leo"])

    def test_overlong_exchange_is_copyedited_before_another_full_rewrite(self):
        canonical, ledger = fixture(durations=[5.0], lines=[speech(LINE_A, 1)])
        writer = Mock(return_value=json.dumps({"L1": "Let's label our clips by scene."}))
        result, warnings = complete_creative_dialogue(BRIEF, ledger, canonical_ledger=canonical,
                    locked_dialogue=[], durations=[5.0], generate=writer, system_prompt="Focused guide")
        self.assertEqual(warnings, [])
        writer.assert_called_once()
        self.assertIn('SHORTEN AI-WRITTEN LINES', writer.call_args.kwargs['prompt'])
        self.assertEqual(result['generated_dialogue'][0]['text'], "Let's label our clips by scene.")

    def test_failed_copyedit_still_gets_one_structured_repair(self):
        canonical, ledger = fixture(durations=[5.0], lines=[speech(LINE_A, 1)])
        writer = Mock(side_effect=['{}', json.dumps({'generated_dialogue': [speech("Label the clips by scene.", 1)]})])
        result, warnings = complete_creative_dialogue(BRIEF, ledger, canonical_ledger=canonical,
                    locked_dialogue=[], durations=[5.0], generate=writer, system_prompt="Focused guide")
        self.assertEqual(warnings, [])
        self.assertEqual(writer.call_count, 2)
        self.assertEqual(result['generated_dialogue'][0]['text'], "Label the clips by scene.")

    def test_copyedit_batches_overlong_turns_and_retains_fitting_turn(self):
        lines = [speech('Please organize the clips by scene first.', 1),
                 speech('We can save a recipe for tomorrow.', 1, 'Leo'),
                 speech('Thanks!', 1)]
        original = deepcopy(lines)
        writer = Mock(side_effect=[
            json.dumps({'L1': 'Please label every clip by scene.', 'L2': 'Save a recipe for tomorrow.', 'L3': 'Thanks!'}),
            json.dumps({'L1': 'Label each scene.', 'L2': 'Save the recipe.'}),
        ])
        edited = _shorten_generated_dialogue(BRIEF, lines, target_words=9, per_line_targets=[4, 4, 1],
                    generate=writer, system_prompt='Copyediting guide')
        self.assertEqual(writer.call_count, 2)
        retry = writer.call_args.kwargs['prompt']
        self.assertIn('"L1"', retry)
        self.assertIn('"L2"', retry)
        self.assertNotIn('"L3"', retry)
        self.assertEqual([line['text'] for line in edited], ['Label each scene.', 'Save the recipe.', 'Thanks!'])
        self.assertEqual([line['speaker'] for line in edited], [line['speaker'] for line in original])
        self.assertEqual(lines, original)

    def test_missing_topic_triggers_repair_even_with_enough_words(self):
        prompt = BRIEF + "\nFeatures include:\n• Save & share characters w/ RefMod support"
        canonical, ledger = fixture(prompt, [10.1], [speech(LINE_A, 1)])
        audits = creative_dialogue_windows(prompt, ledger, [], [10.1])
        self.assertGreaterEqual(audits[0]["spoken_words"], audits[0]["minimum_words"])
        self.assertEqual(audits[0]["missing_topics"], ["Save & share characters w/ RefMod support"])
        replacement = "We can save and share characters with RefMods. That means we can both use the same character in our next shared project."
        result, warnings = complete_creative_dialogue(prompt, ledger, canonical_ledger=canonical, locked_dialogue=[], durations=[10.1], generate=Mock(return_value=json.dumps({"generated_dialogue": [speech(replacement, 1)]})), system_prompt="Focused guide")
        self.assertEqual(warnings, [])
        self.assertEqual(result["generated_dialogue"][0]["text"], replacement)

    def test_final_dialogue_review_accounts_for_time_spent_on_camera_action(self):
        _, ledger = fixture(BRIEF, [14.4], [speech(LINE_A, 1)])
        original = deepcopy(ledger)
        # A six-second entrance/room move leaves eight seconds to speak. The
        # revised 22-word line fits, despite missing the earlier 31-word floor.
        camera = [{"shots": [
            {"start_seconds": 0, "end_seconds": 6.4, "dialogue": []},
            {"start_seconds": 6.4, "end_seconds": 14.4, "dialogue": [{"dialogue_id": "D1"}]},
        ]}]
        self.assertTrue(creative_dialogue_windows(BRIEF, ledger, [], [14.4])[0]["writing_notes"])
        final = creative_dialogue_windows(BRIEF, ledger, [], [14.4], camera_segments=camera)[0]
        self.assertEqual(final["problems"], [])
        self.assertEqual(final["spoken_words"], 22)
        self.assertEqual(ledger, original)

        # Preferred density remains useful writing feedback, not a blocker.
        _, sparse = fixture(BRIEF, [14.4], [speech("Ready?", 1)])
        for shots in ([{"start_seconds": 0, "end_seconds": 14.4, "dialogue": [{"dialogue_id": "D1"}]}], []):
            with self.subTest(shots=shots):
                audited = creative_dialogue_windows(BRIEF, sparse, [], [14.4], camera_segments=[{"shots": shots}])
                self.assertTrue(audited[0]["writing_notes"])
                self.assertEqual(audited[0]["problems"], [])

    def test_camera_time_does_not_waive_missing_spoken_topics(self):
        prompt = BRIEF + "\nFeatures include:\n• Save & share characters w/ RefMod support"
        _, ledger = fixture(prompt, [14.4], [speech(LINE_A, 1)])
        camera = [{"shots": [{"start_seconds": 6.4, "end_seconds": 14.4,
                               "dialogue": [{"dialogue_id": "D1"}]}]}]
        audit = creative_dialogue_windows(prompt, ledger, [], [14.4], camera_segments=camera)[0]
        self.assertGreaterEqual(audit["spoken_words"], audit["minimum_words"])
        self.assertEqual(audit["missing_topics"], ["Save & share characters w/ RefMod support"])
        self.assertIn("missing spoken talking points", audit["problems"][0])

    def test_natural_copyedit_can_miss_preferred_density_without_exceeding_speech_time(self):
        # The live greeting/introduction edit kept every turn and fit its
        # camera clock, but 17 words were rejected against an 18-word floor.
        lines = [speech("Um, hi. Need help?", 1), speech("Welcome. I'm Leo. We sell paper.", 1, "Leo"),
                 speech("I'm Maya. Regional manager. Best in business.", 1)]
        _, ledger = fixture(BRIEF, [14.375], lines)
        camera = [{"shots": [
            {"start_seconds": 4.149, "end_seconds": 5.869, "dialogue": [{"dialogue_id": "D1"}]},
            {"start_seconds": 8.810, "end_seconds": 11.751, "dialogue": [{"dialogue_id": "D2"}]},
            {"start_seconds": 11.751, "end_seconds": 14.375, "dialogue": [{"dialogue_id": "D3"}]},
        ]}]
        audit = creative_dialogue_windows(BRIEF, ledger, [], [14.375], camera_segments=camera)[0]
        self.assertEqual(audit["spoken_words"], 17)
        self.assertEqual(audit["problems"], [])
        ledger["generated_dialogue"][0]["text"] += " Please take a seat over here."
        overlong = creative_dialogue_windows(BRIEF, ledger, [], [14.375], camera_segments=camera)[0]
        self.assertGreater(overlong["spoken_words"], overlong["maximum_words"])
        self.assertTrue(overlong["problems"])


if __name__ == "__main__":
    unittest.main()
