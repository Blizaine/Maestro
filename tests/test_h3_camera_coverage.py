"""A developed event can progress through several camera shots without replay."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.h3_story_ledger import (
    _canonicalize_segment_contract,
    _deterministic_ledger,
    _materialize_segment,
    _enforce_materialized_vocal_staging,
    _reference_h3_cast_names,
    _segment_schema,
    _strip_planner_speech_cues,
    extract_source_events,
    plan_h3_story_segments,
    segment_violations,
)
from services.h3_dialogue_writing import fit_camera_dialogue
from services.h3_sequence_planner import (
    compile_h3_reference_sequence_prompts,
    compute_h3_sequence_clips,
)


def _beat(number, description="The fighters hold position", **extra):
    return {
        "beat_id": f"B{number}", "segment": 1, "description": description,
        "source_event_ids": [f"E{number}"], "dialogue_ids": [],
        "state_after": "The agile fighter stands over the defeated rival",
        **extra,
    }


def _camera(groups, *, actions=None, durations=None, number=1):
    actions = actions or [f"The fighter holds position beside marker {i + 1}" for i in range(len(groups))]
    durations = durations or [12 / len(groups)] * len(groups)
    shots = []
    cursor = 0
    for i, (group, action, duration) in enumerate(zip(groups, actions, durations)):
        shots.append({
            "shot": i + 1, "event_indices": group,
            "start_seconds": cursor, "end_seconds": cursor + duration,
            "transition": "opening composition" if i == 0 else "hard cut",
            "framing": "wide view of both fighters", "camera": f"tracking angle {i + 1}",
            "action": action, "sound_effects": f"Synchronized impact {i + 1}",
        })
        cursor += duration
    return {
        "segment": number, "title": "Courtyard duel", "opening_state": "The fighters face each other",
        "coverage": "motivated action coverage", "pacing": "real time", "shots": shots,
        "closing_state": "The agile fighter stands over the defeated rival",
    }


def _card(action, speaker="Nora"):
    return {"transition": "continuous reframing", "framing": f"Medium shot of {speaker}",
            "camera": f"Hold on {speaker} in the established room", "action": action,
            "sound_effects": "Quiet office room tone"}


class H3CameraCoverageTests(unittest.TestCase):
    def test_cross_and_crossed_do_not_create_a_false_future_event(self):
        source = ("[0s-4s] Nora makes crossed-blade strikes with bright sparks. "
                  "[4s-8s] Nora clears the impact crater and sends a cross strike out through the doorway.")
        first = extract_source_events(source)[0]
        beats = [_beat(1, first["text"], state_after=first["text"])]
        draft = _camera([[1]], actions=["Nora delivers a cross strike; its impact flings bright sparks out."])
        canonical = self.canonical(draft, beats)
        errors = segment_violations(source, canonical, segment_number=1, duration=12,
                                    assigned_beats=beats, dialogue_catalog=[])
        self.assertFalse(any("previews later" in error for error in errors), errors)
        canonical["shots"][0]["action"] += " Nora clears the crater and sends a strike through the doorway."
        errors = segment_violations(source, canonical, segment_number=1, duration=12,
                                    assigned_beats=beats, dialogue_catalog=[])
        self.assertTrue(any("previews later" in error for error in errors), errors)

    def test_silent_e_inflection_is_not_a_missing_action(self):
        source = "Nora raises the device; 0.3s fast charge; Nora releases the pulse."
        beats = [_beat(1, source)]
        draft = _camera([[1]], actions=[
            "Nora raises the device, charging rapidly for 0.3 seconds, and releases the pulse."
        ])
        canonical = self.canonical(draft, beats)
        self.assertEqual(segment_violations(source, canonical, segment_number=1, duration=12,
                                           assigned_beats=beats, dialogue_catalog=[]), [])
        canonical["shots"][0]["action"] = "Nora raises the device and releases the pulse."
        errors = segment_violations(source, canonical, segment_number=1, duration=12,
                                    assigned_beats=beats, dialogue_catalog=[])
        self.assertTrue(any("fast charge" in error for error in errors))

    def test_camera_repair_uses_the_writers_local_event_card_name(self):
        actions = ["Nora opens the gate.", "Nora climbs the stairs.",
                   "Nora raises the lantern; glass prisms scatter reflections.",
                   "Nora pockets the key."]
        source = "Nora wears a red coat. No dialogue. " + " ".join(
            f"[{i * 3}s-{(i + 1) * 3}s] {action}" for i, action in enumerate(actions)
        )
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            props = (kwargs.get("json_schema") or {}).get("properties", {})
            if "check_1" in props:
                # The reflection really is absent, so the semantic check must
                # leave the local event-card repair in place.
                checks = json.loads(kwargs["prompt"])
                self.assertIn("glass prisms scatter reflections", checks["check_1"]["source_requirement"])
                return json.dumps({key: {"verdict": "missing", "evidence": []}
                                   for key in checks})
            if "event_cards" not in props:
                return json.dumps({"character_appearance": {"Nora": "A courier in a red coat."},
                    "setting_continuity": "The gate leads to a stairway and balcony.",
                    "visual_continuity": "Daylight.", "editing_style": "Motivated coverage.",
                    "ambient_audio": "Footsteps."})
            number = props["segment"]["minimum"]
            assigned = actions[:2] if number == 1 else actions[2:]
            if number == 2:
                if "REPAIR ONLY THIS SEGMENT" in kwargs["prompt"]:
                    feedback = kwargs["prompt"].split("Correct these violations:", 1)[1]
                    self.assertIn("event_cards.event_1", feedback)
                    self.assertIn("glass prisms scatter reflections", feedback)
                    self.assertNotIn("B3", feedback)
                    self.assertEqual(props["event_cards"]["required"], ["event_1"])
                    # A remote provider may return extra cards despite the
                    # patch schema. It cannot overwrite already-valid work.
                    assigned = [assigned[0], "Nora throws the key into the river."]
                else:
                    assigned = ["Nora raises the lantern.", assigned[1]]
            return json.dumps({"segment": number, "title": "The courier's route",
                "coverage": "Follow Nora from gate to stairway and balcony.", "pacing": "Real time",
                "event_cards": {f"event_{i}": {"phases": [_card(action)]}
                                for i, action in enumerate(assigned, 1)},
                "closing_state": assigned[-1]})

        result = plan_h3_story_segments(source, segment_durations=[6, 6], mode="sliding_window",
            camera_coverage="multi_shot", planning_style="adaptive", llm_generate=generate)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["planning_warnings"], [])
        self.assertIn("glass prisms scatter reflections", result["segments"][1]["shots"][0]["action"])
        self.assertIn("Nora pockets the key", result["segments"][1]["shots"][1]["action"])
        self.assertNotIn("river", result["segments"][1]["closing_state"])

    def test_local_camera_fields_satisfy_authored_camera_directions(self):
        source = "Nora pockets the key; rapid push-in to a close-up of her eyes."
        beats = [_beat(1, source)]
        draft = _camera([[1]], actions=["Nora pockets the key."])
        draft["shots"][0].update(camera="Rapid push-in to Nora's eyes", framing="Close-up")
        canonical = self.canonical(draft, beats)
        self.assertEqual(segment_violations(source, canonical, segment_number=1, duration=12,
                                          assigned_beats=beats, dialogue_catalog=[]), [])
        # Future/global prose and sound descriptions cannot satisfy the same
        # missing visible direction; it must actually appear in this beat.
        canonical["shots"][0].update(camera="Static", framing="Medium shot",
                                     sound_effects=source)
        canonical["closing_state"] = source
        self.assertTrue(segment_violations(source, canonical, segment_number=1, duration=12,
                                           assigned_beats=beats, dialogue_catalog=[]))

    def test_generic_style_quality_does_not_require_a_separate_visible_action(self):
        source = ("[0s-6s] Nora pockets the key, cinematic perspective. "
                  "[6s-12s] Nora exits the room.")
        beats = [_beat(1, extract_source_events(source)[0]["text"])]
        canonical = self.canonical(_camera([[1]], actions=["Nora pockets the key."]), beats,
                                   source_events=extract_source_events(source))
        self.assertEqual(segment_violations(source, canonical, segment_number=1, duration=12,
                                          assigned_beats=beats, dialogue_catalog=[]), [])
        # The authored quality remains in the rendered source summary, even
        # though the writer does not have to repeat the same adjective.
        rendered = _materialize_segment(canonical, beats=beats, dialogue_catalog=[],
                                        source_events=extract_source_events(source))
        self.assertIn("cinematic perspective", rendered["summary"])
        self.assertIn("cinematic perspective", rendered["shots"][0]["camera"])

    def test_concrete_optics_and_actions_are_not_generic_style_quality(self):
        for requirement in ("shallow depth of field", "rapid push-in to her eyes",
                            "dramatic red light illuminates her face",
                            "the heavy oak door swings completely shut behind Nora"):
            source = f"[0s-6s] Nora pockets the key, {requirement}. [6s-12s] Nora exits the room."
            beats = [_beat(1, extract_source_events(source)[0]["text"])]
            canonical = self.canonical(_camera([[1]], actions=["Nora pockets the key."]), beats)
            with self.subTest(requirement=requirement):
                violations = segment_violations(source, canonical, segment_number=1, duration=12,
                                                 assigned_beats=beats, dialogue_catalog=[])
                self.assertTrue(any("omits required source step" in item for item in violations),
                                violations)

    def canonical(self, draft, beats, catalog=None, **kwargs):
        return _canonicalize_segment_contract(
            draft, segment_number=draft["segment"],
            duration=draft["shots"][-1]["end_seconds"], assigned_beats=beats,
            dialogue_catalog=catalog or [], opening_state=draft["opening_state"], source_intent={},
            **kwargs,
        )

    def violations(self, draft, beats, catalog=None):
        return segment_violations(
            "Two fighters duel. The agile fighter wins.", draft,
            segment_number=draft["segment"], duration=draft["shots"][-1]["end_seconds"],
            assigned_beats=beats, dialogue_catalog=catalog or [],
        )

    def test_developed_fight_event_keeps_all_three_camera_actions_in_native_prompt(self):
        actions = [
            "The heavy fighter hooks the agile fighter's jaw; she skids through glass, recovers and launches upward",
            "She drives him through the overpass, grabs his cape and spins him down onto the rooftop",
            "She stands over his defeated body while her cape settles in the wind",
        ]
        beats = [_beat(1, ". Then ".join(actions))]
        draft = _camera([[1], [1], [1]], actions=actions)
        source = deepcopy(draft)
        canonical = self.canonical(draft, beats)
        self.assertEqual(self.violations(canonical, beats), [])
        self.assertEqual(draft, source)
        self.assertEqual([shot["action"] for shot in canonical["shots"]], actions)
        rendered = _materialize_segment(canonical, beats=beats, dialogue_catalog=[], source_events=[])
        self.assertEqual(len(rendered["shots"]), 3)
        for shot, action in zip(rendered["shots"], actions):
            for clause in action.split(". "):
                self.assertIn(clause, shot["action"])
        self.assertEqual([shot["camera"] for shot in rendered["shots"]], [shot["camera"] for shot in draft["shots"]])
        self.assertEqual(rendered["closing_state"], draft["closing_state"])
        clips, _ = compute_h3_sequence_clips(288)
        compiled = compile_h3_reference_sequence_prompts(
            {"subject_definitions": "Two adult fighters", "clips": [rendered]}, clips,
            reference_relationships="", default_retention="", task_types="reference generation",
        )[0]["prompt"]
        for action in actions:
            self.assertIn(action, compiled)
        self.assertNotIn("event_indices", compiled)

    def test_adjacent_events_can_share_a_shot_and_continue_into_the_next(self):
        beats = [_beat(1), _beat(2), _beat(3)]
        for groups in ([[1], [1, 2], [2, 3]], [[1, 2], [2], [3]], [[1], [2], [3]]):
            with self.subTest(groups=groups):
                canonical = self.canonical(_camera(groups), beats)
                self.assertEqual(self.violations(canonical, beats), [])

    def test_excess_camera_shots_require_repair_instead_of_dropping_final_action(self):
        beats = [_beat(1)]
        draft = _camera([[1]] * 5)
        canonical = self.canonical(draft, beats)
        self.assertEqual(canonical["shots"], draft["shots"])
        self.assertIn("returned 5 shots instead of one to 4", self.violations(canonical, beats))

    def test_missing_foreign_revisited_and_malformed_event_coverage_requires_repair(self):
        beats = [_beat(1), _beat(2)]
        for groups in (
            [[1]], [[2]], [[1], [3]], [[2], [1]], [[1], [2], [1]],
            [[1, 1], [2]], [[1, 2], [1, 2]], [[], [1, 2]], [[True], [2]],
            [["1"], [2]], [None, [1, 2]], [1, [2]], [{"event": 1}, [2]],
        ):
            with self.subTest(groups=groups):
                canonical = self.canonical(_camera(groups), beats)
                self.assertIn("event_assignment_error", canonical)
                self.assertIn("Cover every numbered event", " ".join(self.violations(canonical, beats)))

    def test_saved_semantic_plan_cannot_revisit_or_duplicate_event_within_shot(self):
        beats = [_beat(1), _beat(2)]
        canonical = self.canonical(_camera([[1], [1], [2]]), beats)
        for ids in ([['B1'], ['B2'], ['B1']], [['B1', 'B1'], ['B1'], ['B2']]):
            with self.subTest(ids=ids):
                candidate = deepcopy(canonical)
                for shot, beat_ids in zip(candidate["shots"], ids):
                    shot["beat_ids"] = beat_ids
                self.assertIn("assigned beat coverage is missing, foreign, or out of order", self.violations(candidate, beats))

    def test_continuing_event_inserts_exact_speech_once_in_original_order(self):
        catalog = [
            {"dialogue_id": "D1", "speaker": "Mae", "text": "Stay beside me."},
            {"dialogue_id": "D2", "speaker": "Jules", "text": "I will."},
        ]
        beats = [_beat(1, dialogue_ids=["D1"]), _beat(2, dialogue_ids=["D2"])]
        original = deepcopy(beats)
        canonical = self.canonical(_camera([[1], [1], [2], [2]]), beats, catalog)
        self.assertEqual(self.violations(canonical, beats, catalog), [])
        self.assertEqual([[line["dialogue_id"] for line in shot["dialogue"]] for shot in canonical["shots"]], [["D1"], [], ["D2"], []])
        self.assertEqual(beats, original)
        rendered = _materialize_segment(canonical, beats=beats, dialogue_catalog=catalog, source_events=[])
        self.assertEqual([line["text"] for shot in rendered["shots"] for line in shot["dialogue"]], [line["text"] for line in catalog])
        duplicate = deepcopy(canonical)
        duplicate["shots"][1]["dialogue"] = deepcopy(duplicate["shots"][0]["dialogue"])
        self.assertIn("dialogue IDs are missing, duplicated, or out of order", self.violations(duplicate, beats, catalog))

    def test_speech_clock_reserves_time_only_for_the_first_coverage_shot(self):
        catalog = [{"dialogue_id": "D1", "speaker": "Mae", "text": "one two three four five six seven eight nine ten eleven twelve"}]
        beats = [_beat(1, dialogue_ids=["D1"])]
        draft = _camera([[1], [1], [1]], durations=[1, 2, 9])
        canonical = self.canonical(draft, beats, catalog)
        self.assertGreater(canonical["shots"][0]["end_seconds"], 4)
        self.assertEqual(self.violations(canonical, beats, catalog), [])
        self.assertEqual([len(shot["dialogue"]) for shot in canonical["shots"]], [1, 0, 0])

    def test_authored_duration_is_shared_across_coverage_without_doubling(self):
        beats = [_beat(1, authored_duration=8), _beat(2, authored_duration=4)]
        original = deepcopy(beats)
        canonical = self.canonical(_camera([[1], [1], [2]], durations=[1, 3, 8]), beats)
        # Event one still occupies eight seconds (split 1:3); event two gets four.
        self.assertEqual([(shot["start_seconds"], shot["end_seconds"]) for shot in canonical["shots"]], [(0, 2), (2, 8), (8, 12)])
        self.assertEqual(self.violations(canonical, beats), [])
        self.assertEqual(beats, original)

    def test_camera_places_greeting_after_entrance_and_conversation_after_both_enter(self):
        catalog = [
            {"dialogue_id": "D1", "speaker": "Eva", "text": "Can I help you?"},
            {"dialogue_id": "D2", "speaker": "Miles", "text": "Welcome to my office."},
        ]
        beats = [_beat(1, dialogue_ids=["D1"]), _beat(2, dialogue_ids=["D2"])]
        actions = [
            "Nora walks up to the reception desk",
            "Eva raises her eyes to Nora with an uncertain smile",
            "Miles leads Nora through his office doorway. Nora follows and closes the door after both are inside",
            "Miles sits opposite Nora inside the office; the closed door remains behind them",
        ]
        draft = _camera([[1], [1], [2], [2]], actions=actions)
        for shot, ids in zip(draft["shots"], [[], ["D1"], [], ["D2"]]):
            shot["dialogue_ids"] = ids
        canonical = self.canonical(draft, beats, catalog)
        self.assertEqual(self.violations(canonical, beats, catalog), [])
        rendered = _materialize_segment(canonical, beats=beats, dialogue_catalog=catalog, source_events=[])
        self.assertEqual([[line["text"] for line in shot["dialogue"]] for shot in rendered["shots"]],
                         [[], ["Can I help you?"], [], ["Welcome to my office."]])
        for shot, action in zip(rendered["shots"], actions):
            for clause in action.split(". "):
                self.assertIn(clause, shot["action"])
        schema = _segment_schema(1, event_count=2, dialogue_ids=["D1", "D2"])
        self.assertIn("dialogue_ids", schema["properties"]["shots"]["items"]["required"])
        self.assertEqual(schema["properties"]["shots"]["items"]["properties"]["dialogue_ids"]["maxItems"], 1)

    def test_camera_writes_each_turn_instead_of_summarizing_several_people_talking(self):
        catalog = [{"dialogue_id": f"D{i}", "speaker": speaker, "text": "Hello there."}
                   for i, speaker in enumerate(["Eva", "Nora"], 1)]
        beats = [_beat(1, dialogue_ids=["D1", "D2"])]
        draft = _camera([[1]])
        draft["shots"][0]["dialogue_ids"] = ["D1", "D2"]
        self.assertIn("each dialogue_id its own camera phase", self.canonical(draft, beats, catalog)["dialogue_assignment_error"])

    def test_explicit_dialogue_cannot_repeat_skip_reorder_or_move_to_another_event(self):
        catalog = [{"dialogue_id": f"D{i}", "speaker": "Eva", "text": "Hello there."} for i in (1, 2)]
        beats = [_beat(1, dialogue_ids=["D1"]), _beat(2, dialogue_ids=["D2"])]
        for groups in ([[], []], [["D1", "D1"], ["D2"]], [["D2"], ["D1"]],
                       [[], ["D1", "D2"]], [["D1"], ["D3"]], [None, ["D2"]]):
            with self.subTest(groups=groups):
                draft = _camera([[1], [2]])
                for shot, ids in zip(draft["shots"], groups):
                    shot["dialogue_ids"] = ids
                original = deepcopy(draft)
                result = self.canonical(draft, beats, catalog)
                self.assertIn("dialogue_assignment_error", result)
                self.assertEqual(draft, original)

    def test_required_camera_cards_keep_final_turn_in_both_office_windows(self):
        # Same failure shape as f696e14d: four turns with an entrance, then
        # five turns with a move into another room. Both omitted the final turn.
        for number, turn_ids in enumerate((["D1", "D2", "D3", "D4"],
                                           ["D5", "D6", "D7", "D8", "D9"]), 1):
            with self.subTest(window=number):
                split = len(turn_ids) - 2
                beats = [_beat(1, dialogue_ids=turn_ids[:split]), _beat(2, dialogue_ids=turn_ids[split:])]
                catalog = [{"dialogue_id": did, "speaker": "Nora", "text": f"Exact line {did}."} for did in turn_ids]
                events = {}
                for index, beat in enumerate(beats, 1):
                    events[f"event_{index}"] = {
                        did: {"lead_in": None, "performance": _card(f"Nora raises her eyebrows for {did}")}
                        for did in beat["dialogue_ids"]
                    }
                    events[f"event_{index}"]["follow_through"] = None
                movement = "Miles leads Nora through the door. Both enter before Miles closes it"
                events["event_2"][turn_ids[split]]["lead_in"] = _card(movement)
                draft = {"segment": number, "title": "Office visit", "coverage": "continuous tracking",
                         "pacing": "real time", "closing_state": "Both remain inside behind the closed door",
                         # Provider property ordering cannot change story order.
                         "event_cards": dict(reversed(list(events.items())))}
                original = deepcopy(draft)
                canonical = _canonicalize_segment_contract(draft, segment_number=number, duration=14,
                    assigned_beats=beats, dialogue_catalog=catalog, opening_state="Both are outside", source_intent={})
                self.assertEqual(segment_violations("An office visit", canonical, segment_number=number,
                    duration=14, assigned_beats=beats, dialogue_catalog=catalog), [])
                self.assertEqual([d["dialogue_id"] for shot in canonical["shots"] for d in shot["dialogue"]], turn_ids)
                for clause in movement.split(". "):
                    self.assertIn(clause, canonical["shots"][split]["action"])
                self.assertEqual(canonical["shots"][split]["dialogue"], [])
                self.assertEqual(canonical["shots"][-1]["dialogue"][0]["dialogue_id"], turn_ids[-1])
                self.assertEqual(draft, original)
                self.assertNotIn("event_cards", canonical)
                self.assertEqual(canonical["shots"][-1]["end_seconds"], 14)
                rendered = _materialize_segment(canonical, beats=beats, dialogue_catalog=catalog, source_events=[])
                self.assertEqual([line["text"] for shot in rendered["shots"] for line in shot["dialogue"]],
                                 [line["text"] for line in catalog])
                schema = _segment_schema(number, assigned_beats=beats)
                event_schema = schema["properties"]["event_cards"]
                self.assertEqual(event_schema["required"], ["event_1", "event_2"])
                self.assertIn(turn_ids[-1], event_schema["properties"]["event_2"]["required"])

    def test_incomplete_or_foreign_camera_cards_still_require_review(self):
        beats = [_beat(1, dialogue_ids=["D1", "D2"]), _beat(2)]
        valid = {"event_cards": {
            "event_1": {"D1": {"lead_in": None, "performance": _card("Nora nods")},
                        "D2": {"lead_in": None, "performance": _card("Nora sits")}, "follow_through": None},
            "event_2": {"phases": [_card("Nora opens the window")]},
        }}
        for path, replacement in ((["event_1", "D2"], "remove"), (["event_2"], "remove"),
                                  (["event_1", "D3"], {}), (["event_1", "D2", "performance"], None),
                                  (["event_2", "phases"], []), (["event_2", "phases"], [_card("Nora walks")] * 5)):
            with self.subTest(path=path):
                draft = deepcopy(valid)
                target = draft["event_cards"]
                for key in path[:-1]:
                    target = target[key]
                if replacement == "remove":
                    target.pop(path[-1])
                else:
                    target[path[-1]] = replacement
                result = _canonicalize_segment_contract(draft, segment_number=1, duration=14,
                    assigned_beats=beats, dialogue_catalog=[], opening_state="An office", source_intent={})
                self.assertIn("event_assignment_error", result)
                self.assertTrue(segment_violations("An office", result, segment_number=1, duration=14,
                                                 assigned_beats=beats, dialogue_catalog=[]))

    def test_silent_event_cards_preserve_camera_choreography_and_final_event(self):
        beats = [_beat(1), _beat(2)]
        actions = ["A plants his foot and punches B through a stone wall", "B rebounds from the rubble",
                   "B catches A's next punch and throws him onto the ground"]
        draft = {"title": "Duel", "coverage": "continuous", "pacing": "fast",
                 "closing_state": "A is on the ground",
                 "event_cards": {"event_1": {"phases": [_card(text) for text in actions[:2]]},
                                 "event_2": {"phases": [_card(actions[2])]}}}
        result = _canonicalize_segment_contract(draft, segment_number=1, duration=14, assigned_beats=beats,
                                               dialogue_catalog=[], opening_state="A ruined courtyard", source_intent={})
        self.assertEqual(self.violations(result, beats), [])
        self.assertEqual([shot["action"] for shot in result["shots"]], actions)
        self.assertEqual([shot["event_indices"] for shot in result["shots"]], [[1], [1], [2]])
        # The story ledger owns the cross-window state; the camera writer may
        # choreograph the event but cannot replace its resulting state.
        self.assertEqual(
            result["closing_state"],
            "The agile fighter stands over the defeated rival",
        )

    def test_shot_clock_does_not_compress_spoken_line_to_fund_silent_action(self):
        catalog = [{"dialogue_id": "D1", "speaker": "Eva",
                    "text": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"}]
        beats = [_beat(1, dialogue_ids=["D1"])]
        draft = _camera([[1], [1], [1]], durations=[4, 2, 6], actions=[
            "Nora crosses the room, walks to the desk and sits down",
            "Eva looks at Nora",
            "Nora stands, crosses the room, opens the door and steps through",
        ])
        for shot, ids in zip(draft["shots"], [[], ["D1"], []]):
            shot["dialogue_ids"] = ids
        canonical = self.canonical(draft, beats, catalog)
        speaking = canonical["shots"][1]
        self.assertGreaterEqual(speaking["end_seconds"] - speaking["start_seconds"], 16 / 3 + 0.199)
        self.assertEqual(canonical["shots"][-1]["end_seconds"], 12)
        self.assertEqual(canonical["shots"][2]["dialogue"], [])

    def test_multi_sentence_transcript_is_removed_atomically_from_visual_action(self):
        text = ('Miles says, "Welcome to the office. We are a family. A very large family." '
                'Nora smiles and sits down. A sign reads "Welcome. Please sign in."')
        result = _strip_planner_speech_cues(text)
        self.assertNotIn("We are a family", result)
        self.assertNotIn("A very large family", result)
        self.assertIn("Nora smiles and sits down", result)
        self.assertIn("Please sign in", result)

    def test_competing_voice_directions_do_not_survive_beside_sequential_tagged_turns(self):
        result = _strip_planner_speech_cues(
            "The two men continue to argue, their voices overlapping. "
            "Nora folds her arms. The overlapping shadows cross the carpet. "
            "The men are talking over one another. She listens to overlapping voices."
        )
        self.assertNotIn("voices overlapping", result)
        self.assertNotIn("talking over", result)
        self.assertNotIn("overlapping voices", result)
        self.assertIn("Nora folds her arms", result)
        self.assertIn("overlapping shadows", result)

    def test_camera_cannot_cut_to_listener_before_active_speaker_finishes(self):
        for proposed in ("Close-up of Dean, then to Nora for her reaction",
                         "Quickly cutting between close-ups of Dean and Miles as they lean in"):
            framing, camera, _ = _enforce_materialized_vocal_staging(
                framing="medium shot of Dean", camera=proposed,
                action="Dean lifts a ream of paper", dialogue_sources=[{"speaker": "Dean"}],
                known_speakers=["Dean", "Nora", "Miles"],
            )
            self.assertIn("frames Dean", camera)
            self.assertIn("only after the line ends", camera)
            self.assertNotIn("then to Nora", camera)

    def test_camera_direction_inside_action_cannot_override_the_speaking_face(self):
        _, camera, action = _enforce_materialized_vocal_staging(
            framing="medium shot on Miles", camera="Static medium shot on Miles at the desk",
            action="Miles places his hands on the desk. Nora folds her arms. The camera pushes in slowly on Nora's face.",
            dialogue_sources=[{"speaker": "Miles"}], known_speakers=["Miles", "Nora"],
        )
        self.assertIn("Miles at the desk", camera)
        self.assertIn("Miles places his hands", action)
        self.assertIn("Nora folds her arms", action)
        self.assertNotIn("camera pushes in", action)

    def test_camera_copyedit_preserves_locked_text_and_dialogue_ownership(self):
        locked = {"dialogue_id": "D1", "speaker": "Eva", "language": "English", "text": "Please sit down."}
        generated = {"dialogue_id": "D2", "speaker": "Nora", "language": "English",
                     "source_event_id": "E2", "segment": 1, "text": "Thank you very much for the wonderfully warm welcome to this extremely lovely office."}
        catalog = [deepcopy(locked), deepcopy(generated)]
        stored = [deepcopy(generated)]
        segment = {"segment": 1, "shots": [
            {"shot": 1, "start_seconds": 0, "end_seconds": 2, "dialogue": [{"dialogue_id": "D1"}]},
            {"shot": 2, "start_seconds": 2, "end_seconds": 4, "dialogue": [{"dialogue_id": "D2"}]},
        ]}
        generate = Mock(return_value=json.dumps({"L1": "Thanks for the warm welcome."}))
        warnings = fit_camera_dialogue("A workplace conversation", segment, catalog, stored,
                                      generate=generate, system_prompt="Write naturally")
        self.assertEqual(warnings, [])
        self.assertEqual(catalog[0], locked)
        self.assertEqual(stored[0], {**generated, "text": "Thanks for the warm welcome."})
        self.assertEqual(catalog[1], stored[0])
        self.assertNotIn(locked["text"], generate.call_args.kwargs["prompt"])

    def test_unedited_overlong_camera_speech_is_saved_with_review_notice(self):
        line = {"dialogue_id": "D1", "speaker": "Eva", "text": "one two three four five six seven eight"}
        segment = {"segment": 1, "shots": [{"shot": 1, "start_seconds": 0, "end_seconds": 1,
                                           "dialogue": [{"dialogue_id": "D1"}]}]}
        generator = Mock()
        result = fit_camera_dialogue("Keep exact words", segment, [line], [], generate=generator, system_prompt="")
        self.assertIn("more speaking time", result[0])
        generator.assert_not_called()
        self.assertEqual(line["text"], "one two three four five six seven eight")

    def test_copyedit_keeps_a_shorter_turn_that_fits_after_global_time_redistribution(self):
        catalog = [{"dialogue_id": f"D{i}", "speaker": name, "language": "English",
                    "text": " ".join(["word"] * count)}
                   for i, (name, count) in enumerate((("Eva", 30), ("Nora", 20)), 1)]
        stored = deepcopy(catalog)
        segment = {"segment": 1, "shots": [
            {"shot": 1, "start_seconds": 0, "end_seconds": 5, "dialogue": [{"dialogue_id": "D1"}]},
            {"shot": 2, "start_seconds": 5, "end_seconds": 10, "dialogue": [{"dialogue_id": "D2"}]},
        ]}
        # The second edit misses its old per-shot cap, but fits when the much
        # shorter first reply releases time. A second identical edit is safe.
        writer = Mock(return_value=json.dumps({"L1": "Please sit down.", "L2": " ".join(["reply"] * 18)}))
        warnings = fit_camera_dialogue("A conversation", segment, catalog, stored,
                                      generate=writer, system_prompt="Write dialogue.")
        self.assertEqual(warnings, [])
        self.assertEqual(catalog, stored)
        self.assertEqual(catalog[1]["text"], " ".join(["reply"] * 18))
        self.assertGreaterEqual(segment["shots"][1]["end_seconds"] - segment["shots"][1]["start_seconds"], 6.2)

    def test_freed_speech_time_is_rebalanced_before_requesting_another_copyedit(self):
        word_counts = [3, 8, 5, 8, 4]
        catalog = [{"dialogue_id": f"D{i}", "speaker": "Eva", "language": "English",
                    "text": " ".join(["word"] * count)} for i, count in enumerate(word_counts, 1)]
        boundaries = [0, 3.086, 4.778, 6.301, 9.387, 11.120, 12.642, 14.375]
        ids = [[], ["D1"], ["D2"], [], ["D3"], ["D4"], ["D5"]]
        segment = {"segment": 1, "shots": [
            {"shot": i + 1, "start_seconds": start, "end_seconds": end,
             "action": "Nora enters the room, crosses the carpet and sits down" if not local else "Eva looks at Nora",
             "dialogue": [{"dialogue_id": did} for did in local]}
            for i, (start, end, local) in enumerate(zip(boundaries, boundaries[1:], ids))
        ]}
        generator = Mock(side_effect=AssertionError("This draft already fits after reallocating time"))
        warnings = fit_camera_dialogue("A conversation", segment, catalog, deepcopy(catalog),
                                      generate=generator, system_prompt="")
        self.assertEqual(warnings, [])
        generator.assert_not_called()
        self.assertEqual(segment["shots"][-1]["end_seconds"], 14.375)
        by_id = {item["dialogue_id"]: item for item in catalog}
        for shot in segment["shots"]:
            seconds = shot["end_seconds"] - shot["start_seconds"]
            if shot["dialogue"]:
                words = len(by_id[shot["dialogue"][0]["dialogue_id"]]["text"].split())
                self.assertGreaterEqual(seconds + 0.001, words / 3 + 0.2)
            else:
                self.assertGreaterEqual(seconds + 0.001, 2)

    def test_unnamed_image_reference_is_not_a_fifth_person(self):
        from services.h3_sequence_planner import _ref2va_prompt_bindings, _h3_plan_cast_names
        reference = "<Subject 1> is the supplied image reference from <Picture 1>, preserving identity."
        self.assertEqual(_reference_h3_cast_names(reference), [])
        aliases, _ = _ref2va_prompt_bindings(reference)
        names = ["Nora", "Eva", "Dean", "Miles"]
        self.assertEqual(_h3_plan_cast_names({"source_intent": {"cast_names": names}}, aliases), names)

    def test_two_window_adaptive_scene_accepts_continued_camera_coverage_without_repair(self):
        prompt = "Two adult fighters duel in a ruined courtyard. The agile fighter wins. No dialogue."
        durations = [14.375, 13.625]
        events = extract_source_events(prompt)
        actions = [
            ["The heavy fighter advances and throws a hook", "The agile fighter ducks and drives an elbow into his ribs", "Both slide apart beside a cracked pillar"],
            ["The agile fighter rebounds from the pillar", "Her spinning kick connects and sends her rival to the ground", "She stands over the defeated rival and lowers her guard"],
        ]
        ledger = _deterministic_ledger(prompt, segment_count=2, segment_durations=durations,
            locked_dialogue=[], camera_coverage="multi_shot", reference_context="")
        ledger["beats"] = [
            _beat(i + 1, ". Then ".join(local_actions), segment=i + 1,
                source_event_ids=[event["event_id"] for event in (events[:-1] if i == 0 else events[-1:])])
            for i, local_actions in enumerate(actions)
        ]
        for mode in ("sliding_window", "reference_sequence", "reference_sequence_continuation"):
            with self.subTest(mode=mode):
                calls = []
                def generate(**kwargs):
                    calls.append(kwargs)
                    self.assertNotIn("REPAIR", kwargs["prompt"])
                    schema = kwargs["json_schema"]["properties"]
                    if "setting_continuity" in schema:
                        return json.dumps(ledger)
                    number = schema["segment"]["minimum"]
                    return json.dumps(_camera([[1], [1], [1]], number=number,
                        durations=[durations[number - 1] / 3] * 3, actions=actions[number - 1]))
                result = plan_h3_story_segments(prompt, segment_durations=durations, mode=mode,
                    camera_coverage="multi_shot", expect_dialogue=False, planning_style="adaptive", llm_generate=generate)
                self.assertEqual(result["planned_by"], "llm")
                self.assertEqual(result["planning_warnings"], [])
                self.assertEqual(result["planning_diagnostics"], [])
                self.assertEqual(len(calls), 3)
                for segment, expected_actions in zip(result["segments"], actions):
                    self.assertEqual(len(segment["shots"]), 3)
                    for shot, action in zip(segment["shots"], expected_actions):
                        self.assertIn(action, shot["action"])
                self.assertEqual(result["segments"][1]["opening_state"], result["segments"][0]["closing_state"])

    def test_adaptive_camera_can_correct_ai_staging_while_preserving_user_event(self):
        prompt = "Mara uses heat vision to burn a groove in the floor. No dialogue."
        ledger = _deterministic_ledger(prompt, segment_count=1, segment_durations=[8],
            locked_dialogue=[], camera_coverage="multi_shot", reference_context="")
        bad_staging = "Mara looks upward and fires heat vision, burning a groove in the floor."
        ledger["initial_state"] = "Mara kneels on the stone floor."
        ledger["beats"] = [_beat(1, bad_staging,
            state_after="Mara kneels beside a scorched groove in the floor.")]
        corrected = ("Mara lowers her gaze toward the floor ahead, firing heat vision from her eyes "
                     "down into the stone. She sweeps her gaze forward, burning a groove at the "
                     "beam's contact point, then shuts off the beam.")
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            if "setting_continuity" in kwargs["json_schema"]["properties"]:
                return json.dumps(ledger)
            serialized = kwargs["prompt"].split(
                "Assigned chronological events (depict each once, in order):\n", 1
            )[1].split("\n\nImmutable dialogue performances", 1)[0]
            assignments = json.loads(serialized)
            self.assertEqual(assignments[0]["source_requirements"], extract_source_events(prompt))
            self.assertEqual(assignments[0]["staging_draft"], bad_staging)
            self.assertNotIn("event", assignments[0])
            return json.dumps({
                "segment": 1, "title": "Scoring the stone", "coverage": "profile view",
                "pacing": "continuous motion", "event_cards": {
                    "event_1": {"phases": [_card(corrected, "Mara")]},
                }, "closing_state": "Mara kneels beside a scorched groove in the floor.",
            })
        result = plan_h3_story_segments(prompt, segment_durations=[8], mode="reference_sequence",
            camera_coverage="multi_shot", expect_dialogue=False, planning_style="adaptive",
            llm_generate=generate)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(result["planning_warnings"], [])
        action = result["segments"][0]["shots"][0]["action"]
        self.assertIn("lowers her gaze toward the floor", action)
        self.assertIn("then shuts off the beam", action)
        self.assertNotIn("looks upward", action)


if __name__ == "__main__":
    unittest.main()
