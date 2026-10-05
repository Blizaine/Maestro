"""Source-owned action survives camera writing, recovery and native compilation."""
from copy import deepcopy
import json
import re
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import (
    _h3_camera_metadata_performer_action, _recover_source_camera_events,
    _h3_source_camera_optics_only,
    _h3_preview_action_frames,
    extract_source_events, plan_h3_story_segments, segment_violations,
)
from services.h3_window_planner import compile_h3_window_prompts


SOURCE = (
    "[0s-4s] Nora raises her right elbow, plants her left foot firmly on the wooden "
    "floor, and drops shoulder-first into a close-range explosive elbow smash on Eli. "
    "Eli slips sideways past the elbow without taking the hit, keeping the brass key "
    "in his left hand throughout the movement. Nora lands beside the red bench with "
    "both feet on the floor and her right hand resting against the bench. The sunlight "
    "falls through a high narrow window onto a row of plain wooden shelves, a green "
    "cabinet, a folded blue cloth, and a white ceramic bowl beside the bench. "
    "[4s-8s] Eli places the brass key on the red bench. "
    "[8s-12s] Nora opens the blue gate. No dialogue."
)


def schema_value(schema, field=""):
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {key: schema_value(schema["properties"][key], key)
                for key in schema.get("required", [])}
    if kind == "array":
        return [schema_value(schema["items"]) for _ in range(schema.get("minItems", 0))]
    if kind in {"integer", "number"}:
        return schema.get("minimum", 1)
    return {
        "coverage": "The red bench and blue gate define a consistent left-to-right screen axis.",
        "pacing": "Readable real-time action",
        "framing": "50mm lateral medium view with the red bench in the background",
        "camera": "Gentle lateral dolly with a stable screen axis",
        "transition": "continuous reframe",
        "sound_effects": "Quiet wooden room tone",
    }.get(field, "Source camera test")


class SourceCameraIntegrationTests(unittest.TestCase):
    def test_camera_nouns_are_not_future_physical_actions_but_real_actions_remain(self):
        cast = re.compile(r"\b(?:Nora|Eli)\b", re.I)
        for text in ("Medium close-up shot of Nora.", "Mid-wide shot of Nora.",
                     "Hard cut to Nora's hands.", "Continuous take framing Nora.",
                     "Nora's close-quarters stance.",
                     "Medium close-up focused on Nora's lower body/feet."):
            self.assertEqual(_h3_preview_action_frames(text, cast), [], text)
        for text, verb in (("Nora closes the blue gate.", "close"),
                           ("Nora shoots Eli.", "shoot"),
                           ("Nora takes the brass key.", "take"),
                           ("Nora lowers her body into a crouch.", "lower"),
                           ("Close-up of Nora opening the blue gate.", "open")):
            self.assertIn(verb, [frame[0] for frame in _h3_preview_action_frames(text, cast)], text)
    def test_camera_reference_tails_are_removed_without_authorizing_standalone_actions(self):
        draft = {"coverage": "A whip pan as Nora opens the gate.", "event_cards": {
            "event_1": {"phase_1": {
                "camera": "Rapid whip-pan following Nora's trajectory as she skids across the platform.",
                "framing": "Close-up on Nora's face as she lifts her head.",
                "transition": "hard cut", "sound_effects": "Stone rumble",
            }}, "event_2": {"phase_1": {"camera": "Nora opens the blue gate."}},
        }}
        before = deepcopy(draft)
        cleaned = _h3_source_camera_optics_only(draft, ["Nora"])
        self.assertEqual(cleaned["coverage"], "A whip pan")
        phase = cleaned["event_cards"]["event_1"]["phase_1"]
        self.assertEqual(phase["camera"], "Rapid whip-pan following Nora's trajectory")
        self.assertEqual(phase["framing"], "Close-up on Nora's face")
        self.assertEqual(cleaned["event_cards"]["event_2"]["phase_1"]["camera"], "Nora opens the blue gate.")
        self.assertEqual(draft, before)
        self.assertEqual(_h3_source_camera_optics_only({"event_cards": {"event_1": {
            "phase_1": {"framing": "Wide shot of the air burst, and Nora being sideswept."},
        }}}, ["Nora"])["event_cards"]["event_1"]["phase_1"]["framing"], "Wide shot of the air burst")

    def test_compound_contact_substitution_is_hard_even_in_a_short_legacy_plan(self):
        source = "Nora drops shoulder-first into a close-range explosive elbow smash on Eli."
        prompt = "[0s-5s] " + source + " [5s-10s] Nora opens the blue gate. No dialogue."
        events = extract_source_events(prompt)
        beat = {"beat_id": "B1", "source_event_ids": ["E1"], "description": events[0]["text"], "dialogue_ids": []}
        segment = {"segment": 1, "semantic_actions": True, "shots": [{
            "shot": 1, "beat_ids": ["B1"], "start_seconds": 0, "end_seconds": 5,
            "action": "Nora drops her weight and drives her shoulder into Eli.",
            "camera": "Locked camera", "framing": "Medium view", "transition": "opening composition",
        }], "closing_state": "Nora is beside Eli."}
        errors = segment_violations(prompt, segment, segment_number=1, duration=5,
                                    assigned_beats=[beat], dialogue_catalog=[])
        error = next(item for item in errors if "changes required contact instrument" in item)
        from services.h3_camera_fidelity import clear_confirmed_coverage_errors
        self.assertIn(error, clear_confirmed_coverage_errors(errors, segment, {error: "untrusted approval"}))
        segment["shots"][0]["action"] = source
        corrected = segment_violations(prompt, segment, segment_number=1, duration=5,
                                       assigned_beats=[beat], dialogue_catalog=[])
        self.assertFalse(any("changes required contact instrument" in item for item in corrected), corrected)

    def test_camera_descriptions_do_not_turn_objects_into_action_subjects(self):
        for text in ("Camera follows Nora tightly.", "Dolly back to frame them centrally.",
                     "A wide view of Character A.", "Character A is framed against the wall.",
                     "A thunderous crack of stone and a rushing wind.",
                     "A whip pan between the two characters.",
                     "A rapid dolly along Character A's downward trajectory.",
                     "A crash against the rock face."):
            self.assertFalse(_h3_camera_metadata_performer_action(text, ["Nora", "Character A"]), text)
        for text in ("Nora corkscrews into Eli.", "A punches Character B.",
                     "A corkscrews into Character B.", "A is punching Character B.",
                     "Camera tracks Nora, then she opens the gate."):
            self.assertTrue(_h3_camera_metadata_performer_action(text, ["Nora", "Character A"]), text)

    def run_plan(self, *, bad_event=None, repair_limit=0, photographic_nouns=False,
                 anatomical_nouns=False):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            schema = kwargs["json_schema"]
            properties = schema.get("properties", {})
            draft = schema_value(schema)
            if "segment" not in properties:
                return json.dumps({
                    "character_appearance": {"Nora": "Adult in plain clothing", "Eli": "Adult in plain clothing"},
                    "setting_continuity": "One wooden room with a red bench and blue gate.",
                    "visual_continuity": "Natural live action", "editing_style": "Readable coverage",
                    "motion_mechanics": "Physical actions with clear contact", "ambient_audio": "Quiet room tone",
                })
            if photographic_nouns:
                for event in draft["event_cards"].values():
                    for phase in event.values():
                        if phase.get("framing") != "The supplied frame's exact opening composition":
                            phase["framing"] = "50mm medium shot with Nora and Eli framed beside the red bench"
                        phase["camera"] = "Quick cut to Nora's hands with the red bench in the background"
                        phase["transition"] = "Continuous take framing Nora and Eli"
            if anatomical_nouns:
                for event in draft["event_cards"].values():
                    for phase in event.values():
                        if phase.get("framing") != "The supplied frame's exact opening composition":
                            phase["framing"] = "Medium close-up focused on Nora's lower body/feet"
            if bad_event and draft["segment"] == 1:
                keys = (list(draft["event_cards"]) if bad_event == "all" else
                        ["event_2"] if bad_event == "future_view" else [bad_event])
                for event_key in keys:
                    event = draft["event_cards"][event_key]
                    event[next(iter(event))]["camera"] = (
                        "Close-up of Nora opening the blue gate." if bad_event == "future_view"
                        else "Nora opens the blue gate."
                    )
            return json.dumps(draft)

        with patch("services.studio_enhancement.fidelity_retry_limit", return_value=repair_limit):
            result = plan_h3_story_segments(
                SOURCE, segment_durations=[8.0, 4.0], mode="sliding_window",
                camera_coverage="multi_shot", expect_dialogue=False,
                planning_style="faithful", llm_generate=generate,
            )
        return result, calls

    def test_photographic_shot_cut_and_take_are_not_performer_actions(self):
        result, _ = self.run_plan(photographic_nouns=True)
        self.assertEqual(result["planning_warnings"], [])
        self.assertTrue(any("50mm medium shot" in shot["framing"]
                            for shot in result["segments"][0]["shots"]))

    def test_actual_future_action_in_a_camera_view_is_still_recovered(self):
        result, _ = self.run_plan(bad_event="future_view")
        self.assertTrue(any("event(s) 2" in value for value in result["planning_warnings"]))
        self.assertTrue(any("previews later source event E3" in value
                            for value in result["planning_diagnostics"]))

    def test_lower_body_framing_does_not_preview_a_later_lowering_action(self):
        source = SOURCE.replace("Nora opens the blue gate", "Nora lowers her body into a crouch")
        with patch(__name__ + ".SOURCE", source):
            result, _ = self.run_plan(anatomical_nouns=True)
        self.assertEqual(result["planning_warnings"], [])
        self.assertTrue(any("lower body/feet" in shot["framing"]
                            for shot in result["segments"][0]["shots"]))

    def test_locked_actions_survive_native_compilation_without_writer_action_fields(self):
        result, calls = self.run_plan()
        camera_calls = [call for call in calls if "segment" in call["json_schema"].get("properties", {})]
        self.assertEqual(len(camera_calls), 2)
        self.assertEqual(result["planning_warnings"], [])
        for call in camera_calls:
            schema = json.dumps(call["json_schema"])
            self.assertNotIn('"action"', schema)
            self.assertNotIn('"closing_state"', schema)
        actions = " ".join(shot["action"] for shot in result["segments"][0]["shots"])
        self.assertEqual(actions.count("explosive elbow smash on Eli"), 1)
        self.assertIn("brass key in his left hand throughout the movement", actions)
        self.assertIn("white ceramic bowl beside the bench", actions)
        # Opaque authority exists only during validation, never in saved JSON.
        self.assertNotIn("_source_camera_proof", json.dumps(result))
        compiled = compile_h3_window_prompts(
            {**result["ledger"], "windows": result["segments"], "source_intent": result["source_intent"]},
            [{"start_seconds": 0.0, "end_seconds": 8.0}, {"start_seconds": 8.0, "end_seconds": 12.0}],
            source_prompt=SOURCE,
        )
        self.assertIn("explosive elbow smash on Eli", compiled[0]["prompt"])
        self.assertIn("brass key in his left hand throughout the movement", compiled[0]["prompt"])
        self.assertNotIn("Nora opens the blue gate", compiled[0]["prompt"])
        self.assertIn("Nora opens the blue gate", compiled[1]["prompt"])

    def test_failed_event_recovery_keeps_other_camera_cards_and_reports_review(self):
        result, _ = self.run_plan(bad_event="event_2")
        warnings = result["planning_warnings"]
        self.assertTrue(any("event(s) 2" in item and "preserved the other camera cards" in item for item in warnings), warnings)
        first = result["segments"][0]
        self.assertTrue(any("50mm lateral medium view" in shot["framing"] for shot in first["shots"]))
        key_shot = next(shot for shot in first["shots"] if "Eli places the brass key" in shot["action"])
        self.assertNotIn("Nora opens the blue gate", key_shot["camera"])
        self.assertTrue(result["planning_diagnostics"])
        self.assertEqual(result["camera_checkpoint"]["review_windows"], [1])

    def test_ambiguous_or_still_invalid_recovery_is_rejected(self):
        class Contract:
            def fallback_event(self, key):
                return {"camera": "fallback"}
        original = {"event_cards": {"event_1": {"camera": "saved"}}}
        for errors, validate in [(["invalid whole window"], lambda _value: []),
                                 (["B1 camera failure"], lambda _value: ["still invalid"])]:
            saved = deepcopy(original)
            recovered = _recover_source_camera_events(
                original, errors=errors, assigned_beats=[{"beat_id": "B1"}],
                segment={}, source_camera=Contract(), canonicalize=lambda value: value,
                validate=validate,
            )
            self.assertIsNone(recovered)
            self.assertEqual(original, saved)

        # Rebuilding every card is whole-window fallback, not partial recovery.
        self.assertIsNone(_recover_source_camera_events(
            original, errors=["B1 camera failure"], assigned_beats=[{"beat_id": "B1"}],
            segment={}, source_camera=Contract(), canonicalize=lambda value: value,
            validate=lambda _value: [],
        ))

    def test_all_failed_cards_use_whole_window_fallback_with_source_phases(self):
        result, _ = self.run_plan(bad_event="all")
        warnings = result["planning_warnings"]
        self.assertTrue(any("compiled that window directly" in value for value in warnings), warnings)
        self.assertFalse(any("preserved the other camera cards" in value for value in warnings), warnings)
        actions = " ".join(shot["action"] for shot in result["segments"][0]["shots"])
        self.assertIn("explosive elbow smash on Eli", actions)
        self.assertIn("brass key in his left hand throughout the movement", actions)
        self.assertNotIn("Nora opens the blue gate", actions)
        self.assertGreater(len(result["segments"][0]["shots"]), 2)


if __name__ == "__main__":
    unittest.main()
