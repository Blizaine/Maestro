"""Regression tests for declared blocking state and music-video progression."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))

from services.director.physical_continuity import (
    PHYSICAL_STATE_JSON_SCHEMA,
    format_physical_state_for_prompt,
    music_video_progression_issue,
    normalize_physical_state,
    physical_state_contract_instructions,
    physical_state_subject_issues,
    source_physical_state_issues,
    validate_physical_state_sequence,
)
from services.director.planners.music_video import (
    MusicVideoPlanner,
    _music_shot_schema,
)


def fact(subject_id, predicate, value):
    return {"subject_id": subject_id, "predicate": predicate, "value": value}


def physical_state(opening, transitions=(), closing=None, continuity="continuous"):
    state = {
        "opening": list(opening),
        "transitions": list(transitions),
        "closing": list(closing if closing is not None else opening),
    }
    if continuity != "continuous":
        state["continuity"] = continuity
    return state


def transition(subject_id, predicate, from_value, to_value, action):
    return {
        "subject_id": subject_id,
        "predicate": predicate,
        "from_value": from_value,
        "to_value": to_value,
        "action": action,
    }


class PhysicalContinuityTests(unittest.TestCase):
    def test_legacy_rows_without_typed_state_remain_valid(self):
        self.assertEqual(
            validate_physical_state_sequence([
                {"scene_goal": "A wide performance shot", "metadata": {}},
                {"scene_goal": "A close performance shot"},
            ], require_state_after_first=True),
            [],
        )

    def test_schema_is_closed_and_state_property_is_optional(self):
        state_schema = _music_shot_schema(
            1, include_image_fields=False,
        )["items"]["properties"]["physical_state"]
        self.assertEqual(state_schema, PHYSICAL_STATE_JSON_SCHEMA)
        self.assertNotIn(
            "physical_state",
            _music_shot_schema(1, include_image_fields=False)["items"]["required"],
        )
        self.assertEqual(state_schema["required"], ["opening", "transitions", "closing"])
        self.assertFalse(state_schema["additionalProperties"])
        h3_schema = _music_shot_schema(
            1,
            include_image_fields=False,
            require_physical_state=True,
        )
        self.assertIn("physical_state", h3_schema["items"]["required"])
        self.assertIn("continuity_strategy", h3_schema["items"]["required"])

    def test_normalization_keeps_stable_ids_and_canonicalizes_whitespace(self):
        normalized = normalize_physical_state({
            "opening": [fact("Actor A", "Support", "  seated   on   platform ")],
            "transitions": [],
            "closing": [fact("Actor A", "support", "seated on platform")],
        })
        self.assertEqual(normalized["opening"][0]["subject_id"], "Actor A")
        self.assertEqual(normalized["opening"][0]["predicate"], "support")
        self.assertEqual(normalized["opening"][0]["value"], "seated on platform")

    def test_new_fact_values_are_compact_and_grounded_is_categorical(self):
        fact_schema = PHYSICAL_STATE_JSON_SCHEMA["properties"]["opening"]["items"]
        transition_schema = PHYSICAL_STATE_JSON_SCHEMA["properties"]["transitions"]["items"]
        self.assertEqual(fact_schema["properties"]["value"]["maxLength"], 80)
        self.assertEqual(transition_schema["properties"]["from_value"]["maxLength"], 80)
        self.assertEqual(transition_schema["properties"]["to_value"]["maxLength"], 80)
        # Every constrained fact/transition must retain its owner and values.
        # Branch-only schemas previously emitted predicate alone in live repairs.
        self.assertNotIn("anyOf", fact_schema)
        self.assertNotIn("anyOf", transition_schema)
        self.assertEqual(fact_schema["required"], ["subject_id", "predicate", "value"])
        self.assertEqual(transition_schema["required"], [
            "subject_id", "predicate", "from_value", "to_value", "action",
        ])
        self.assertEqual(
            transition_schema["properties"]["operation"]["enum"], ["establish"],
        )
        self.assertIn("null", transition_schema["properties"]["from_value"]["type"])
        with self.assertRaisesRegex(ValueError, "non-empty subject_id"):
            normalize_physical_state({
                "opening": [{"predicate": "grounded", "value": "grounded"}],
                "transitions": [], "closing": [],
            })

        legacy_short = normalize_physical_state({
            "opening": [fact("actor_a", "support", "standing on platform")],
            "transitions": [],
            "closing": [fact("actor_a", "support", "astride actor_b")],
        })
        self.assertEqual(legacy_short["opening"][0]["value"], "standing on platform")
        categorical = normalize_physical_state({
            "opening": [fact("actor_a", "grounded", " GROUNDED ")],
            "transitions": [],
            "closing": [fact("actor_a", "grounded", "AIRBORNE")],
        })
        self.assertEqual(categorical["opening"][0]["value"], "grounded")
        self.assertEqual(categorical["closing"][0]["value"], "airborne")
        with self.assertRaisesRegex(ValueError, "must be exactly 'grounded' or 'airborne'"):
            normalize_physical_state({
                "opening": [fact("actor_a", "grounded", "standing before the castle gate")],
                "transitions": [],
                "closing": [fact("actor_a", "grounded", "grounded")],
            })
        with self.assertRaisesRegex(ValueError, "exceeds 80 characters"):
            normalize_physical_state({
                "opening": [fact("actor_a", "support", "x" * 81)],
                "transitions": [],
                "closing": [fact("actor_a", "support", "short")],
            })

    def test_physical_state_generation_instructions_keep_ids_and_prose_separate(self):
        instructions = physical_state_contract_instructions()
        self.assertIn("character_id", instructions)
        self.assertIn("Never invent prop IDs", instructions)
        self.assertIn("grounded.value", instructions)
        self.assertIn("transitions[].action", instructions)
        self.assertIn("copy the preceding closing subject_id, predicate, and value exactly", instructions)
        self.assertIn("transition.from_value must exactly match", instructions)
        self.assertIn("final current value must exactly match closing", instructions)

    def test_source_relation_guard_uses_assigned_events_without_row_cooccurrence(self):
        row = {
            "shot_id": "boarding",
            "subjects_on_screen": [
                {"character_id": "actor_a"},
                {"character_id": "actor_b"},
            ],
            "action_beats": ["The actors exchange a look."],
            "physical_state": physical_state([], [], []),
        }
        issues = source_physical_state_issues(
            [row],
            [[{"event_id": "E1", "text": "Actor A climbs aboard Actor B."}]],
        )
        self.assertTrue(any("support fact in physical_state.opening" in issue for issue in issues))
        self.assertTrue(any("support fact in physical_state.closing" in issue for issue in issues))

        row["physical_state"] = physical_state(
            [fact("actor_a", "support", "standing on the stone ledge")],
            closing=[fact("actor_a", "support", "astride actor_b")],
        )
        self.assertEqual(source_physical_state_issues(
            [row], [["Actor A climbs aboard Actor B."]],
        ), [])

    def test_source_touch_allows_no_contact_at_opening_but_requires_contact_fact(self):
        row = {
            "shot_id": "touch",
            "subjects_on_screen": [
                {"character_id": "actor_a"},
                {"character_id": "actor_b"},
            ],
            "physical_state": physical_state(
                [fact("actor_a", "contact", "none")],
                closing=[fact("actor_a", "contact", "touching actor_b")],
            ),
        }
        self.assertEqual(source_physical_state_issues(
            [row], [["Actor A touches Actor B's shoulder."]],
        ), [])
        row["physical_state"] = physical_state([], [], [])
        self.assertTrue(source_physical_state_issues(
            [row], [["Actor A touches Actor B's shoulder."]],
        ))

    def test_boarding_with_saddle_strap_keeps_opening_ground_and_none_valid(self):
        row = {
            "physical_state": physical_state(
                [
                    fact("rider", "support", "on ground"),
                    fact("rider", "contact", "none"),
                ],
                closing=[
                    fact("rider", "support", "astride carrier"),
                    fact("rider", "contact", "holding saddle strap"),
                ],
            ),
        }
        source = (
            "The rider climbs up using the saddle strap and settles astride "
            "the carrier's back."
        )
        self.assertEqual(source_physical_state_issues([row], [[source]]), [])

    def test_pilot_landing_slide_and_feet_on_ground_require_support_and_grounded_facts(self):
        row = {
            "physical_state": physical_state(
                [
                    fact("actor_a", "support", "astride actor_b"),
                    fact("actor_a", "grounded", "airborne"),
                ],
                closing=[
                    fact("actor_a", "support", "standing on ground"),
                    fact("actor_a", "grounded", "grounded"),
                ],
            ),
        }
        source = (
            "Dragon lands, becomes stationary and folds wings before Kitten "
            "slides down its shoulder and puts both feet on the ground."
        )
        self.assertEqual(source_physical_state_issues([row], [[source]]), [])

        row["physical_state"] = physical_state(
            [fact("actor_a", "support", "astride actor_b")],
            closing=[fact("actor_a", "support", "standing on ground")],
        )
        issues = source_physical_state_issues([row], [[source]])
        self.assertTrue(any("grounded fact in physical_state.opening" in issue for issue in issues))
        self.assertTrue(any("grounded fact in physical_state.closing" in issue for issue in issues))

    def test_flight_event_requires_grip_contact_and_grounded_transition_endpoints(self):
        row = {
            "physical_state": physical_state(
                [
                    fact("rider", "contact", "none"),
                    fact("rider", "grounded", "grounded"),
                ],
                closing=[
                    fact("rider", "contact", "gripping saddle strap"),
                    fact("rider", "grounded", "airborne"),
                ],
            ),
        }
        source = "After the rider grips the strap securely, the carrier takes flight."
        self.assertEqual(source_physical_state_issues([row], [[source]]), [])

    def test_patting_and_gripping_are_contact_relations_but_sung_notes_are_not(self):
        state = physical_state(
            [fact("actor_a", "contact", "none")],
            closing=[fact("actor_a", "contact", "touching actor_b snout")],
        )
        self.assertEqual(source_physical_state_issues(
            [{"physical_state": state}],
            [["Kitten pats Dragon's snout, then grips the saddle strap."]],
        ), [])
        self.assertEqual(source_physical_state_issues(
            [{"physical_state": physical_state([], [], [])}],
            [["The singer holds a sustained note while the chorus carries the melody."]],
        ), [])

    def test_source_relation_guard_ignores_unrelated_or_negated_events(self):
        row = {"physical_state": physical_state([], [], [])}
        self.assertEqual(source_physical_state_issues(
            [row], [["The performer walks beside the carrier."]],
        ), [])
        self.assertEqual(source_physical_state_issues(
            [row], [["The performer does not ride the carrier."]],
        ), [])

    def test_physical_state_rejects_fabricated_prop_subject_ids(self):
        row = {
            "shot_id": "mount",
            "subjects_on_screen": [{"character_id": "actor_a"}],
            "physical_state": physical_state(
                [fact("prop_0", "support", "on the floor")],
                closing=[fact("prop_0", "support", "astride actor_b")],
            ),
        }
        self.assertTrue(any(
            "not a stable visible actor ID" in issue
            for issue in physical_state_subject_issues(row)
        ))

    def test_sequence_validator_rejects_unknown_owner_without_source_relation(self):
        row = {
            "shot_id": "quiet-portrait",
            "scene_goal": "A quiet portrait beside a window.",
            "subjects_on_screen": [{"character_id": "actor_a"}],
            "physical_state": physical_state(
                [fact("prop_0", "posture", "upright")],
            ),
        }
        issues = validate_physical_state_sequence([row])
        self.assertEqual(len(issues), 1)
        self.assertIn("subject_id 'prop_0' is not a stable visible actor ID", issues[0])

    def test_conflicting_facts_are_rejected_with_a_clear_error(self):
        row = {
            "shot_id": "handoff-1",
            "physical_state": physical_state(
                [
                    fact("actor_a", "support", "standing on platform"),
                    fact("actor_a", "support", "seated on platform"),
                ],
            ),
        }
        issues = validate_physical_state_sequence([row])
        self.assertEqual(len(issues), 1)
        self.assertIn("conflicts with another support fact", issues[0])

    def test_ordered_transitions_can_update_multiple_facts_and_close_cleanly(self):
        opening = [
            fact("actor_a", "support", "standing beside carrier"),
            fact("actor_a", "posture", "upright"),
            fact("actor_a", "prop", "holding lantern"),
        ]
        closing = [
            fact("actor_a", "support", "seated on carrier back"),
            fact("actor_a", "posture", "leaning into turn"),
            fact("actor_a", "prop", "lantern secured at belt"),
        ]
        state = physical_state(
            opening,
            [
                transition(
                    "actor_a", "support", "standing beside carrier",
                    "seated on carrier back", "climbs up and settles on the back",
                ),
                transition(
                    "actor_a", "posture", "upright", "leaning into turn",
                    "leans into the carrier's turn",
                ),
                transition(
                    "actor_a", "prop", "holding lantern", "lantern secured at belt",
                    "fastens the lantern to the belt before takeoff",
                ),
            ],
            closing,
        )
        issues = validate_physical_state_sequence([{"physical_state": state}])
        self.assertEqual(issues, [])
        self.assertIn("climbs up and settles on the back", format_physical_state_for_prompt(state))

    def test_transition_must_start_from_preceding_fact_and_produce_closing(self):
        opening = [fact("actor_a", "support", "standing beside carrier")]
        state = physical_state(
            opening,
            [transition(
                "actor_a", "support", "already airborne", "seated on carrier back",
                "climbs onto the back",
            )],
            [fact("actor_a", "support", "standing beside carrier")],
        )
        issues = validate_physical_state_sequence([{"physical_state": state}])
        self.assertTrue(any("starts at 'already airborne'" in issue for issue in issues))
        self.assertTrue(any("transitions produce 'seated on carrier back'" in issue for issue in issues))

    def test_establishment_is_rendered_and_replayed_only_at_its_event(self):
        source = "Maya grips the saddle strap with a sword sheathed at her hip."
        transition_row = {
            "shot_id": "boarding",
            "source_event_ids": ["E1"],
            "_director_source_contract": {"events": [{"event_id": "E1", "text": source}]},
            "subjects_on_screen": [
                {"character_id": "maya", "speaker_name": "Maya"},
                {"character_id": "bea", "speaker_name": "Bea"},
            ],
            "physical_state": {
                "opening": [],
                "transitions": [{
                    "subject_id": "maya", "predicate": "contact",
                    "operation": "establish", "event_id": "E1", "from_value": None,
                    "to_value": "saddle strap", "action": "grips the saddle strap",
                }],
                "closing": [fact("maya", "contact", "saddle strap")],
            },
        }
        normalized = normalize_physical_state(transition_row["physical_state"])
        self.assertEqual(normalized["transitions"][0]["operation"], "establish")
        self.assertIsNone(normalized["transitions"][0]["from_value"])
        self.assertEqual(normalized["transitions"][0]["event_id"], "E1")
        self.assertEqual(validate_physical_state_sequence([transition_row]), [])
        self.assertEqual(
            source_physical_state_issues([transition_row], [[{"event_id": "E1", "text": source}]]),
            [],
        )
        prompt_text = format_physical_state_for_prompt(normalized)
        self.assertIn("establish maya contact as saddle strap when grips the saddle strap", prompt_text)
        self.assertNotIn("changes from None", prompt_text)
        self.assertNotIn("from None", prompt_text)

        wrong_target = dict(transition_row)
        wrong_target["physical_state"] = {
            **transition_row["physical_state"],
            "transitions": [{
                **transition_row["physical_state"]["transitions"][0],
                "to_value": "sword", "action": "grips the sword",
            }],
            "closing": [fact("maya", "contact", "sword")],
        }
        self.assertTrue(any(
            "not directly supported" in issue
            for issue in source_physical_state_issues(
                [wrong_target], [[{"event_id": "E1", "text": source}]],
            )
        ))

    def test_establishment_rejects_existing_or_unproven_keys(self):
        valid_source = "Maya grips the saddle strap."
        base = {
            "source_event_ids": ["E1"],
            "_director_source_contract": {"events": [{"event_id": "E1", "text": valid_source}]},
            "subjects_on_screen": [{"character_id": "maya", "speaker_name": "Maya"}],
            "physical_state": {
                "opening": [],
                "transitions": [{
                    "subject_id": "maya", "predicate": "contact",
                    "operation": "establish", "event_id": "E1", "from_value": None,
                    "to_value": "saddle strap", "action": "grips the saddle strap",
                }],
                "closing": [fact("maya", "contact", "saddle strap")],
            },
        }
        existing = {
            **base,
            "physical_state": {
                **base["physical_state"],
                "opening": [fact("maya", "contact", "none")],
            },
        }
        self.assertTrue(any(
            "cannot establish a fact that already exists" in issue
            for issue in validate_physical_state_sequence([existing])
        ))

        no_contract = {**base, "_director_source_contract": None}
        self.assertTrue(any(
            "no embedded immutable source contract" in issue
            for issue in validate_physical_state_sequence([no_contract])
        ))
        no_closing = {
            **base,
            "physical_state": {**base["physical_state"], "closing": []},
        }
        no_closing_issues = source_physical_state_issues(
            [no_closing], [[{"event_id": "E1", "text": valid_source}]],
        )
        self.assertTrue(any("contact fact in physical_state.closing" in issue for issue in no_closing_issues))
        no_event = {
            **base,
            "physical_state": {
                **base["physical_state"],
                "transitions": [{
                    key: value for key, value in base["physical_state"]["transitions"][0].items()
                    if key != "event_id"
                }],
            },
        }
        with self.assertRaisesRegex(ValueError, "event_id"):
            normalize_physical_state(no_event["physical_state"])

    def test_establishment_rejects_ambiguous_or_unknown_named_actor(self):
        source = "The Kitten and Dragon grip the saddle strap together."
        row = {
            "source_event_ids": ["E1"],
            "_director_source_contract": {"events": [{"event_id": "E1", "text": source}]},
            "subjects_on_screen": [
                {"character_id": "char_0", "speaker_name": "Kitten"},
                {"character_id": "char_1", "speaker_name": "Dragon"},
            ],
            "physical_state": {
                "opening": [],
                "transitions": [{
                    "subject_id": "char_0", "predicate": "contact",
                    "operation": "establish", "event_id": "E1", "from_value": None,
                    "to_value": "saddle strap", "action": "grips the saddle strap",
                }],
                "closing": [fact("char_0", "contact", "saddle strap")],
            },
        }
        self.assertTrue(any(
            "not directly supported" in issue
            for issue in source_physical_state_issues([row], [[{"event_id": "E1", "text": source}]])
        ))

        unknown_source = "A waiter grips the saddle strap."
        row["_director_source_contract"] = {"events": [{"event_id": "E1", "text": unknown_source}]}
        self.assertTrue(any(
            "not directly supported" in issue
            for issue in validate_physical_state_sequence([row])
        ))

    def test_establishment_requires_positive_exact_owner_predicate_and_target(self):
        cast = [
            {"character_id": "char_0", "speaker_name": "Maya"},
            {"character_id": "char_1", "speaker_name": "Bea"},
        ]
        cases = [
            (
                "Maya does not grip the saddle strap.", "char_0", "contact",
                "saddle strap", "grips the saddle strap",
            ),
            (
                "Maya grips the saddle strap.", "char_1", "contact",
                "saddle strap", "grips the saddle strap",
            ),
            (
                "Maya grips the saddle strap.", "char_0", "prop",
                "saddle strap", "grips the saddle strap",
            ),
            (
                "Maya grips the saddle strap.", "char_0", "contact",
                "sword", "magically acquires the sword",
            ),
            (
                "Maya never settles astride Bea.", "char_0", "support",
                "astride char_1", "settles astride Bea",
            ),
        ]
        for source, owner, predicate, target, action in cases:
            with self.subTest(source=source, owner=owner, predicate=predicate, target=target):
                row = {
                    "source_event_ids": ["E1"],
                    "subjects_on_screen": cast,
                    "physical_state": {
                        "opening": [],
                        "transitions": [{
                            "subject_id": owner, "predicate": predicate,
                            "operation": "establish", "event_id": "E1", "from_value": None,
                            "to_value": target, "action": action,
                        }],
                        "closing": [fact(owner, predicate, target)],
                    },
                }
                events = [[{"event_id": "E1", "text": source}]]
                issues = source_physical_state_issues([row], events)
                self.assertTrue(any("not directly supported" in issue for issue in issues), issues)

    def test_adjacent_opening_must_match_prior_closing_unless_cut_is_declared(self):
        first = {"physical_state": physical_state(
            [fact("actor_a", "support", "standing on platform")],
        )}
        mismatch = {"physical_state": physical_state(
            [fact("actor_a", "support", "seated on platform")],
        )}
        issues = validate_physical_state_sequence([first, mismatch])
        self.assertTrue(any("preceding closing state is 'standing on platform'" in issue for issue in issues))

        time_jump = {"physical_state": physical_state(
            [fact("actor_a", "support", "seated on platform")],
            continuity="time_jump",
        )}
        self.assertEqual(validate_physical_state_sequence([first, time_jump]), [])
        self.assertIn("intentional time jump", format_physical_state_for_prompt(time_jump["physical_state"]))

    def test_declared_physical_cut_must_match_the_planned_cut_strategy(self):
        first = {"physical_state": physical_state(
            [fact("actor_a", "support", "standing on platform")],
        )}
        mismatch = {
            "continuity_strategy": "continuous",
            "physical_state": physical_state(
                [fact("actor_a", "support", "seated on platform")],
                continuity="time_jump",
            ),
        }
        issues = validate_physical_state_sequence([first, mismatch])
        self.assertTrue(any("requires a planned independent cut" in issue for issue in issues))

        planned_cut = {
            "continuity_strategy": "independent",
            "physical_state": physical_state(
                [fact("actor_a", "support", "seated on platform")],
                continuity="time_jump",
            ),
        }
        self.assertEqual(validate_physical_state_sequence([first, planned_cut]), [])

    def test_active_state_can_be_required_after_its_first_declaration(self):
        typed = {"physical_state": physical_state(
            [fact("actor_a", "support", "seated on carrier back")],
        )}
        missing = {"scene_goal": "A close portrait during flight"}
        self.assertEqual(validate_physical_state_sequence([typed, missing]), [])
        issues = validate_physical_state_sequence(
            [typed, missing], require_state_after_first=True,
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("shot 2 is missing physical_state", issues[0])


class MusicVideoProgressionTests(unittest.TestCase):
    def row(self, *, action=None, camera=None, scene_type="narrative", goal=None):
        return {
            "scene_type": scene_type,
            "scene_goal": goal or "The carrier crosses the canyon during the chorus",
            "action_beats": action or [
                "Actor A keeps both arms forward as the carrier flies toward camera",
                "The carrier holds the same line across the canyon",
            ],
            "ending_beat": "The carrier continues forward over the canyon",
            "camera_plan": camera or {
                "framing": "wide shot",
                "angle": "eye level",
                "movement": "tracking forward",
            },
        }

    def test_semantically_repeated_actions_are_detected_after_a_run(self):
        rows = [
            self.row(),
            self.row(action=[
                "Actor A holds both arms forward while the carrier is flying toward the viewer",
                "The carrier continues on the same canyon line",
            ]),
            self.row(action=[
                "Actor A extends both arms forward as the carrier flies at the camera",
                "The carrier keeps its line across the canyon",
            ]),
        ]
        issue = music_video_progression_issue(rows, sections=["chorus"] * len(rows))
        self.assertIsNotNone(issue)
        self.assertIn("repeats the same planned action", issue)

    def test_new_action_beat_or_meaningful_camera_variation_allows_a_motif(self):
        repeated = [self.row(), self.row(), self.row()]
        with_new_beat = [
            self.row(), self.row(),
            self.row(action=[
                "Actor A extends both arms as the carrier flies toward camera",
                "The carrier banks sharply around a stone arch and dives beneath it",
            ]),
        ]
        with_camera_change = [
            self.row(), self.row(),
            self.row(camera={
                "framing": "close shot", "angle": "low angle", "movement": "orbit",
            }),
        ]
        self.assertIsNotNone(music_video_progression_issue(repeated))
        self.assertIsNone(music_video_progression_issue(with_new_beat))
        self.assertIsNone(music_video_progression_issue(with_camera_change))

    def test_source_events_sections_performance_and_loops_preserve_intentional_reuse(self):
        repeated = [self.row(), self.row(), self.row()]
        self.assertIsNone(music_video_progression_issue(
            repeated,
            source_event_ids_by_index=[["E1"], ["E2"], ["E3"]],
            sections=["chorus"] * 3,
        ))
        self.assertIsNone(music_video_progression_issue(
            repeated,
            source_event_ids_by_index=[["E1"], ["E1"], ["E1"]],
            sections=["chorus"] * 3,
        ))
        self.assertIsNone(music_video_progression_issue(
            repeated, sections=["verse", "verse", "bridge"],
        ))
        self.assertIsNone(music_video_progression_issue(
            [self.row(scene_type="performance") for _ in range(3)],
        ))
        self.assertIsNone(music_video_progression_issue(
            [self.row(goal="An intentional chorus motif loops across the canyon") for _ in range(3)],
        ))


class MusicVideoPlannerPhysicalStateTests(unittest.TestCase):
    def _row_for_physical_validation(self, state, video_prompt=None):
        return {
            "scene_goal": "The rider crosses the valley astride the carrier",
            "scene_type": "narrative",
            "subjects_on_screen": [{
                "character_id": "traveler",
                "visual_description": "a traveler in a blue coat",
                "performance_role": "non_vocal",
            }],
            "environment": "an open valley",
            "visual_style": "cinematic",
            "lighting": "late afternoon",
            "mood": "determined",
            "action_beats": ["The traveler remains astride the carrier"],
            "camera_plan": {"framing": "medium shot"},
            "ending_beat": "The traveler remains seated on the carrier",
            "video_prompt": video_prompt or "The traveler rides astride the carrier across the valley.",
            "window_prompts": [],
            "physical_state": state,
        }

    def test_source_explicit_relation_requires_support_or_contact_at_both_ends(self):
        row = self._row_for_physical_validation(
            physical_state([], [], []),
        )
        issue = MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A traveler rides astride a carrier across the valley.",
            include_image_fields=False,
            require_source_physical_state=True,
        )
        self.assertIn("support/contact fact in both opening and closing", issue)

        supported = physical_state(
            [fact("traveler", "support", "astride carrier")],
            closing=[fact("traveler", "support", "astride carrier")],
        )
        row["physical_state"] = supported
        self.assertIsNone(MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A traveler rides astride a carrier across the valley.",
            include_image_fields=False,
            require_source_physical_state=True,
        ))

    def test_assigned_source_event_requires_state_even_when_row_omits_relation_words(self):
        row = self._row_for_physical_validation(physical_state([], [], []))
        row["action_beats"] = ["The performer looks toward the valley."]
        row["video_prompt"] = "A quiet close view of the performer looking outward."
        issue = MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A quiet concert beside the valley.",
            include_image_fields=False,
            require_source_physical_state=True,
            source_events_for_row=[{
                "event_id": "E7",
                "text": "The traveler rides astride the carrier across the valley.",
            }],
        )
        self.assertIn("requires a support fact in physical_state.opening", issue or "")

    def test_explicit_prompt_negation_conflicts_with_final_support_fact(self):
        state = physical_state(
            [fact("traveler", "support", "astride carrier")],
            closing=[fact("traveler", "support", "astride carrier")],
        )
        row = self._row_for_physical_validation(
            state,
            "The traveler flies unsupported beside the carrier.",
        )
        issue = MusicVideoPlanner._music_video_row_issue(
            0,
            row,
            [],
            scene_description="A traveler rides astride a carrier.",
            include_image_fields=False,
        )
        self.assertIn("directly negates the declared closing support relation", issue)

    def test_conversion_compiles_state_and_transition_into_video_prompt_and_saved_metadata(self):
        state = physical_state(
            [fact("actor_a", "support", "standing beside carrier")],
            [transition(
                "actor_a", "support", "standing beside carrier",
                "seated on carrier back", "climbs up and settles on the back",
            )],
            [fact("actor_a", "support", "seated on carrier back")],
        )
        row = {
            "scene_goal": "Board the carrier",
            "scene_type": "narrative",
            "subjects_on_screen": [{
                "character_id": "actor_a",
                "visual_description": "a traveler in a blue coat",
                "performance_role": "non_vocal",
            }],
            "environment": "an open landscape",
            "visual_style": "cinematic",
            "lighting": "late afternoon",
            "mood": "determined",
            "action_beats": ["The traveler prepares to board"],
            "camera_plan": {"framing": "medium shot", "movement": "slow dolly in"},
            "ending_beat": "The traveler is seated on the carrier back",
            "video_prompt": "The traveler climbs aboard as the carrier waits.",
            "window_prompts": [],
            "physical_state": state,
        }
        clip = {
            "label": "verse", "start": 0, "end": 5, "beat_count": 8,
            "source_event_ids": ["EVENT-7"],
        }
        shot = MusicVideoPlanner()._convert_to_shots(
            [row], [clip], [], False, {}, None, {},
        )[0]
        serialized = shot.to_dict()
        self.assertIn("Opening physical facts", serialized["video_prompt"])
        self.assertIn("Transition detail to show", serialized["video_prompt"])
        self.assertIn("settles on the back", serialized["video_prompt"])
        self.assertIn("At the final frame", serialized["video_prompt"])
        self.assertIn("final-frame facts describe only the end of the clip", serialized["video_prompt"])
        self.assertIn("seated on carrier back", serialized["video_prompt"])
        self.assertIn("climbs up and settles on the back", serialized["action_beats"])
        self.assertEqual(serialized["metadata"]["physical_state"], normalize_physical_state(state))
        self.assertEqual(serialized["metadata"]["source_event_ids"], ["EVENT-7"])


if __name__ == "__main__":
    unittest.main()
