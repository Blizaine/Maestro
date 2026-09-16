"""Authored optical settings survive variable camera writing without retries."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import (
    _canonicalize_segment_contract, _materialize_segment,
    extract_source_events, plan_h3_story_segments, segment_violations,
)


SOURCE = (
    "SHOT 1 — 0:00–0:02\n"
    "Dust particles rush rapidly toward the camera, creating a dramatic transition. "
    "Strong cinematic motion blur, volumetric light rays, shallow depth of field.\n\n"
    "SHOT 2 — 0:02–0:04\n"
    "A courier crosses the square; then she opens the gate. Subtle film grain."
)


def card(action, camera="Track the visible movement"):
    return {"action": action, "camera": camera, "framing": "Medium shot",
            "transition": "cut", "sound_effects": "Natural nonverbal sounds"}


class AuthoredOpticsTests(unittest.TestCase):
    def canonical(self, source=SOURCE, first_action="Dust particles rush rapidly toward the camera.",
                  second_action="A courier crosses the square and opens the gate.",
                  first_camera="Track the dust", continuation=False):
        events = extract_source_events(source)
        beats = [{"beat_id": f"B{i}", "segment": 1, "description": e["text"],
                  "source_event_ids": [e["event_id"]], "dialogue_ids": [],
                  "state_after": e["text"], "authored_duration": 2}
                 for i, e in enumerate(events, 1)]
        phases = [card(first_action, first_camera)]
        if continuation:
            phases.append(card("The dust continues rushing toward the camera."))
        draft = {"segment": 1, "event_cards": {"event_1": {"phases": phases},
                    "event_2": {"phases": [card(second_action)]}},
                 "opening_state": "The square is dusty.", "closing_state": "The gate is open."}
        original = deepcopy(draft)
        canonical = _canonicalize_segment_contract(draft, segment_number=1, duration=8,
            assigned_beats=beats, dialogue_catalog=[], opening_state="The square is dusty.",
            source_intent={}, source_events=events)
        self.assertEqual(draft, original)
        return events, beats, canonical

    def test_missing_optics_are_retained_in_their_own_event_not_a_new_action(self):
        events, beats, segment = self.canonical()
        self.assertEqual(segment_violations(SOURCE, segment, segment_number=1, duration=8,
            assigned_beats=beats, dialogue_catalog=[]), [])
        shots = segment["shots"]
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0]["action"], "Dust particles rush rapidly toward the camera.")
        for cue in ("Strong cinematic motion blur", "volumetric light rays", "shallow depth of field"):
            self.assertIn(cue, shots[0]["camera"])
            self.assertNotIn(cue, shots[1]["camera"])
        self.assertIn("Subtle film grain", shots[1]["camera"])
        self.assertNotIn("Subtle film grain", shots[0]["camera"])
        rendered = _materialize_segment(segment, beats=beats, dialogue_catalog=[], source_events=events)
        self.assertIn("shallow depth of field", rendered["shots"][0]["camera"])

    def test_valid_optics_are_not_duplicated_across_continued_phases(self):
        _, _, segment = self.canonical(first_camera="Track with shallow depth of field", continuation=True)
        cameras = " ".join(s["camera"] for s in segment["shots"])
        self.assertEqual(cameras.count("shallow depth of field"), 1)
        self.assertEqual(cameras.count("Strong cinematic motion blur"), 1)
        self.assertNotIn("Strong cinematic motion blur", segment["shots"][1]["camera"])

    def test_source_retention_cannot_launder_a_missing_physical_event(self):
        _, beats, segment = self.canonical(second_action="A courier stands still in the square.")
        errors = segment_violations(SOURCE, segment, segment_number=1, duration=8,
            assigned_beats=beats, dialogue_catalog=[])
        self.assertTrue(any("omits required source step" in error for error in errors), errors)
        self.assertNotIn("opens the gate", segment["shots"][1]["camera"])

    def test_only_source_optics_are_carried_not_invented_ai_settings(self):
        events, beats, _ = self.canonical()
        beats[0]["description"] += " Heavy anamorphic lens flare."
        draft = {"segment": 1, "event_cards": {
            "event_1": {"phases": [card("Dust rushes toward the camera.")]},
            "event_2": {"phases": [card("The courier opens the gate.")]}}, "closing_state": "Open gate."}
        result = _canonicalize_segment_contract(draft, segment_number=1, duration=8,
            assigned_beats=beats, dialogue_catalog=[], opening_state="Dust.", source_intent={},
            source_events=events)
        self.assertNotIn("lens flare", result["shots"][0]["camera"])

    def test_detailed_action_can_paraphrase_its_descriptive_lead_in(self):
        source = (
            "[0s-2s] The fighter launches into an extraordinary martial-arts movement. "
            "She swings her sword while rotating through the air. A golden energy arc sweeps around her. "
            "The energy should look physical and luminous, illuminating the surrounding dust. "
            "[2s-4s] A courier crosses the square; then she opens the gate. Subtle film grain."
        )
        _, beats, segment = self.canonical(source=source, first_action=(
            "The fighter leaps and rotates through the air while swinging her sword. "
            "A solid radiant golden energy arc sweeps around her, lighting the surrounding dust."
        ))
        self.assertEqual(segment_violations(source, segment, segment_number=1, duration=8,
            assigned_beats=beats, dialogue_catalog=[]), [])

    def test_optical_terms_inside_actions_or_speech_are_not_copied_to_camera(self):
        source = ('[0s-2s] A courier opens the gate. The shallow depth of field changes as she turns. '
                  'She says, "Strong cinematic motion blur, volumetric light rays." '
                  '[2s-4s] She walks through the gate.')
        _, _, segment = self.canonical(source=source)
        cameras = " ".join(s["camera"] for s in segment["shots"])
        self.assertNotIn("depth of field", cameras)
        self.assertNotIn("motion blur", cameras)
        self.assertNotIn("light rays", cameras)

    def test_repeated_incomplete_optics_do_not_rewrite_good_choreography(self):
        for camera in ("Static shot", "Shallow depth of field", "Volumetric light rays"):
            calls = []
            def generate(**kwargs):
                calls.append(kwargs)
                props = kwargs["json_schema"]["properties"]
                self.assertNotIn("REPAIR ONLY", kwargs["prompt"])
                if "setting_continuity" in props:
                    return json.dumps({"character_appearance": "A courier in a blue coat.",
                        "setting_continuity": "A dusty town square.", "visual_continuity": "Natural daylight.",
                        "motion_mechanics": "Wind and walking.", "editing_style": "Two shots.",
                        "ambient_audio": "Wind.", "source_adaptation": "Two ordered events over eight seconds."})
                n = props["segment"]["minimum"]
                action = ("Dust particles rush rapidly toward the camera." if n == 1
                          else "A courier crosses the square and opens the gate.")
                return json.dumps({"segment": n, "event_cards": {"event_1": {"phases": [card(action, camera)]}},
                                   "closing_state": action, "coverage": "Clear view", "pacing": "Continuous motion"})
            with self.subTest(camera=camera):
                result = plan_h3_story_segments(SOURCE, segment_durations=[4, 4],
                    mode="reference_sequence", planning_style="adaptive", camera_coverage="multi_shot",
                    llm_generate=generate)
                self.assertEqual(len(calls), 3)
                self.assertEqual(result["planning_warnings"], [])
                self.assertEqual(result["segments"][0]["shots"][0]["camera"].casefold().count("shallow depth of field"), 1)


if __name__ == "__main__":
    unittest.main()
