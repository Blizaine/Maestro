"""CPU checks for bounded source/physical-state patch requests."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.contract_repair import (
    apply_compact_source_physical_patch,
    apply_source_physical_patch,
    build_compact_source_physical_repair_request,
    build_source_physical_repair_request,
)
from services.director.physical_continuity import (
    MAX_PHYSICAL_TRANSITIONS,
    source_physical_state_issues,
    validate_physical_state_sequence,
)
from services.director.source_contract import source_contract_issues


def _physical_state(value: str = "grounded") -> dict:
    return {
        "opening": [
            {"subject_id": "char_0", "predicate": "grounded", "value": value},
            {"subject_id": "char_1", "predicate": "grounded", "value": "grounded"},
        ],
        "transitions": [],
        "closing": [
            {"subject_id": "char_0", "predicate": "grounded", "value": value},
            {"subject_id": "char_1", "predicate": "grounded", "value": "grounded"},
        ],
    }


def _rows() -> list[dict]:
    return [
        {
            "title": "Meeting",
            "scene_goal": "The Kitten meets the Dragon.",
            "spatial_setup": "They face each other on the stone ledge.",
            "duration_sec": 10.0,
            "continuity_strategy": "continuous",
            "source_event_ids": ["E1"],
            "subjects_on_screen": [
                {"character_id": "char_0", "speaker_name": "Kitten", "visual_description": "silver armor"},
                {"character_id": "char_1", "speaker_name": "Dragon", "visual_description": "dark scales"},
            ],
            "action_beats": ["The Kitten stands before the Dragon."],
            "ending_beat": "The Dragon lowers its shoulder.",
            "physical_state": _physical_state("Grounded"),
            "dialogue_beats": [],
            "camera_plan": {"framing": "wide"},
            "video_prompt": "Authored prompt must stay unchanged.",
        },
        {
            "title": "Mounting",
            "scene_goal": "The Kitten boards the Dragon.",
            "spatial_setup": "The Kitten reaches the saddle strap.",
            "duration_sec": 8.0,
            "continuity_strategy": "continuous",
            "source_event_ids": ["E2"],
            "subjects_on_screen": [
                {"character_id": "char_0", "speaker_name": "Kitten"},
                {"character_id": "char_1", "speaker_name": "Dragon"},
            ],
            "action_beats": ["The Kitten climbs the strap."],
            "ending_beat": "The Kitten settles astride.",
            "physical_state": _physical_state(),
            "dialogue_beats": [{"speaker_id": "char_0", "spoken_text": "Ready."}],
            "camera_plan": {"framing": "medium"},
            "video_prompt": "Another authored prompt.",
        },
    ]


def _contract() -> dict:
    return {
        "revision": 1,
        "events": [
            {"event_id": "E1", "text": "The Dragon lowers its shoulder."},
            {"event_id": "E2", "text": "The Kitten climbs the strap and settles astride."},
        ],
        "audio_policy": {
            "generated_speech_allowed": False,
            "generated_music_allowed": False,
        },
    }


def _patches() -> list[dict]:
    return [
        {
            "shot_index": 1,
            "source_event_ids": ["E1"],
            "physical_state": _physical_state(),
        },
        {
            "shot_index": 2,
            "source_event_ids": ["E2"],
            "physical_state": _physical_state(),
        },
    ]


def _compact_response() -> dict:
    return {
        "event_shots": {"E1": 1, "E2": 2},
        "initial_facts": [
            {"subject_id": "char_0", "predicate": "grounded", "value": "grounded"},
            {"subject_id": "char_1", "predicate": "grounded", "value": "grounded"},
            {"subject_id": "char_0", "predicate": "prop", "value": "sheathed sword"},
        ],
        "shot_transitions": [
            {
                "shot_index": 1,
                "transitions": [
                    {
                        "subject_id": "char_0",
                        "predicate": "prop",
                        "to_value": "holding sword",
                        "action": "draws the sword",
                    },
                ],
            },
            {
                "shot_index": 2,
                "transitions": [
                    {
                        "subject_id": "char_0",
                        "predicate": "prop",
                        "to_value": "sheathed sword",
                        "action": "returns the sword to its sheath",
                    },
                ],
            },
        ],
    }


def _compact_object_response(cast_ids: tuple[str, ...] = ("char_0", "char_1")) -> dict:
    initial_facts = {}
    for character_id in cast_ids:
        state = {
            "support": "both paws on ledge" if character_id == "char_0" else "four feet on ledge",
            "contact": "none",
            "grounded": "grounded",
        }
        if len(cast_ids) <= 2:
            state.update({"posture": "standing", "location": "stone ledge"})
            if character_id == "char_0":
                state["prop"] = "sheathed at left hip"
        initial_facts[character_id] = state

    return {
        "event_shots": {"E1": 1, "E2": 2},
        "initial_facts": initial_facts,
        "shot_transitions": [
            {
                "shot_index": 1,
                "transitions": [{
                    "event_id": "E1",
                    "subject_id": "char_1",
                    "predicate": "support",
                    "to_value": "lowered shoulders",
                    "action": "lowers its shoulder",
                }],
            },
            {
                "shot_index": 2,
                "transitions": [
                    {
                        "event_id": "E2",
                        "subject_id": "char_0",
                        "predicate": "support",
                        "to_value": "astride char_1",
                        "action": "climbs the strap and settles astride",
                    },
                    {
                        "event_id": "E2",
                        "subject_id": "char_0",
                        "predicate": "contact",
                        "to_value": "saddle strap",
                        "action": "grips the strap securely",
                    },
                ],
            },
        ],
    }


def _compact_event_response(cast_ids: tuple[str, ...] = ("char_0", "char_1")) -> dict:
    shot_response = _compact_object_response(cast_ids)
    event_transitions = {
        event["event_id"]: [] for event in _contract()["events"]
    }
    for shot in shot_response["shot_transitions"]:
        for transition in shot["transitions"]:
            event_id = transition["event_id"]
            event_transitions[event_id].append({
                key: value for key, value in transition.items() if key != "event_id"
            })
    return {
        "event_shots": shot_response["event_shots"],
        "initial_facts": shot_response["initial_facts"],
        "event_transitions": event_transitions,
    }


def _rows_with_cast_count(count: int) -> list[dict]:
    rows = _rows()
    for character_index in range(2, count):
        for row in rows:
            row["subjects_on_screen"].append({
                "character_id": f"char_{character_index}",
                "speaker_name": f"Character {character_index}",
            })
    return rows


class DirectorContractRepairTests(unittest.TestCase):
    def test_request_is_row_bounded_and_includes_repair_evidence(self):
        system, user, schema = build_source_physical_repair_request(
            _rows(), _contract(), ["Shot 2 omits a carried support fact."],
        )
        self.assertIn("only source-event ownership and typed physical state", system)
        self.assertIn("PHYSICAL STATE CONTRACT", system)
        self.assertIn("grounded.value to exactly 'grounded' or 'airborne'", system)
        self.assertIn("Preserve the existing shot count", user)
        self.assertIn("Allocate every immutable source event ID exactly once", user)
        self.assertIn(
            'use ["E1","E2"] for the first row and ["E3"] for the later row',
            user,
        )
        self.assertIn("use the actual IDs and actions from the supplied contract", user)
        self.assertIn("copying their subject_id, predicate, and value verbatim", user)
        self.assertIn('"event_id":"E2"', user)
        self.assertIn("Shot 2 omits a carried support fact.", user)
        self.assertIn('"character_id":"char_0","speaker_name":"Kitten"', user)
        self.assertIn("The Kitten meets the Dragon.", user)
        self.assertIn("They face each other on the stone ledge.", user)
        self.assertIn("The Kitten climbs the strap.", user)
        self.assertIn("The Kitten climbs the strap and settles astride.", user)
        self.assertIn('"prior_physical_state_context"', user)
        self.assertIn("immutable source contract is authoritative", system)
        self.assertIn("draft context only, not a constraint", system)
        self.assertIn("support='astride char_1'", user)
        self.assertIn("'mounted' is not a grounded value", user)
        self.assertNotIn('"dialogue_beats"', user)
        repair_context = json.loads(user.split("REPAIR CONTEXT:\n", 1)[1])
        self.assertEqual(
            repair_context["original_shots"][0]["prior_physical_state_context"]["status"],
            "draft_revisable",
        )
        self.assertEqual(schema["minItems"], 2)
        self.assertEqual(schema["maxItems"], 2)
        self.assertEqual(
            set(schema["items"]["properties"]),
            {"shot_index", "source_event_ids", "physical_state"},
        )
        self.assertEqual(schema["items"]["required"], [
            "shot_index", "source_event_ids", "physical_state",
        ])
        self.assertFalse(schema["items"]["additionalProperties"])
        self.assertEqual(
            schema["items"]["properties"]["source_event_ids"]["items"]["enum"],
            ["E1", "E2"],
        )

    def test_invalid_prior_ledger_is_excluded_with_concrete_error(self):
        rows = _rows()
        rows[1]["physical_state"] = _physical_state("mounted")

        _, user, _ = build_source_physical_repair_request(
            rows, _contract(), ["Shot 2 has an invalid grounded value."],
        )

        repair_context = json.loads(user.split("REPAIR CONTEXT:\n", 1)[1])
        prior_state = repair_context["original_shots"][1]["prior_physical_state_context"]
        self.assertEqual(prior_state["status"], "invalid_excluded")
        self.assertFalse(prior_state["ledger_included"])
        self.assertIn("physical_state.opening[0].value", prior_state["validation_error"])
        self.assertIn("must be exactly 'grounded' or 'airborne'", prior_state["validation_error"])
        self.assertNotIn("ledger", prior_state)
        self.assertNotIn('"value":"mounted"', json.dumps(repair_context))

    def test_apply_changes_only_patch_fields_and_deep_copies(self):
        rows = _rows()
        original = copy.deepcopy(rows)
        patches = _patches()
        patches[0]["physical_state"] = _physical_state("AIRBORNE")

        patched = apply_source_physical_patch(rows, patches)

        self.assertEqual(rows, original)
        self.assertIsNot(patched, rows)
        self.assertIsNot(patched[0], rows[0])
        self.assertEqual(patched[0]["source_event_ids"], ["E1"])
        self.assertEqual(patched[0]["physical_state"]["opening"][0]["value"], "airborne")
        for key in (
            "title", "duration_sec", "continuity_strategy", "subjects_on_screen",
            "action_beats", "ending_beat", "dialogue_beats", "camera_plan", "video_prompt",
        ):
            self.assertEqual(patched[0][key], original[0][key], key)
        patched[0]["action_beats"].append("mutated")
        self.assertEqual(rows[0]["action_beats"], original[0]["action_beats"])

    def test_apply_rejects_missing_extra_or_partial_patches(self):
        with self.assertRaisesRegex(ValueError, "patch count"):
            apply_source_physical_patch(_rows(), _patches()[:1])
        with self.assertRaisesRegex(ValueError, "patch count"):
            apply_source_physical_patch(_rows(), _patches() + [_patches()[0]])

        patches = _patches()
        del patches[0]["physical_state"]
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            apply_source_physical_patch(_rows(), patches)

        patches = _patches()
        patches[0]["duration_sec"] = 4.0
        with self.assertRaisesRegex(ValueError, "unsupported keys"):
            apply_source_physical_patch(_rows(), patches)

        patches = _patches()
        del patches[1]["physical_state"]["closing"]
        with self.assertRaisesRegex(ValueError, "missing required keys: closing"):
            apply_source_physical_patch(_rows(), patches)

    def test_apply_rejects_invalid_indexes_and_nonstring_ids(self):
        for indexes in ((2, 1), (1, 1), (1, 3), (0, 2)):
            with self.subTest(indexes=indexes):
                patches = _patches()
                patches[0]["shot_index"], patches[1]["shot_index"] = indexes
                with self.assertRaisesRegex(ValueError, "shot_index"):
                    apply_source_physical_patch(_rows(), patches)

        patches = _patches()
        patches[0]["source_event_ids"] = ["E1", 2]
        with self.assertRaisesRegex(ValueError, "array of strings"):
            apply_source_physical_patch(_rows(), patches)

        patches = _patches()
        patches[0]["source_event_ids"] = "E1"
        with self.assertRaisesRegex(ValueError, "array of strings"):
            apply_source_physical_patch(_rows(), patches)

    def test_apply_rejects_null_or_unknown_physical_state_fields(self):
        patches = _patches()
        patches[0]["physical_state"] = None
        with self.assertRaisesRegex(ValueError, "must be an object"):
            apply_source_physical_patch(_rows(), patches)

        patches = _patches()
        patches[0]["physical_state"]["made_up"] = True
        with self.assertRaisesRegex(ValueError, "unsupported keys: made_up"):
            apply_source_physical_patch(_rows(), patches)

    def test_compact_request_schema_is_event_keyed_with_exact_event_ownership(self):
        rows = _rows()
        rows[1]["physical_state"] = {
            **_physical_state(),
            "continuity": "independent",
        }
        system, user, schema = build_compact_source_physical_repair_request(
            rows,
            _contract(),
            ["Shot 2 transition omitted its initial prop fact."],
        )

        self.assertIn("response's event_shots allocation are the only action authority", system)
        self.assertIn("PHYSICAL STATE CONTRACT", system)
        self.assertIn("exactly 'grounded' or 'airborne'", system)
        self.assertIn("contact only for actual touch or grip", system)
        self.assertIn(
            "Being beside or next to, approaching, noticing, or waving beside "
            "another subject does not alone establish contact",
            system,
        )
        self.assertIn("Walking is motion, not a support relation", system)
        self.assertIn("Never invent placeholder support values such as 'N/A'", system)
        self.assertIn("already-known location or posture fact", system)
        self.assertIn('{"E1":1,"E2":1,"E3":2}', user)
        self.assertIn("initial_facts is the complete canonical fact set", user)
        self.assertIn("event_transitions must be an object with exactly one key", user)
        self.assertIn("do not put event_id or shot_index inside a transition", user)
        self.assertIn("caller routes each event's list to its one assigned shot", user)
        self.assertIn("old draft action_beats, ending_beat, or source_event_ids", user)
        self.assertIn(
            f"routed to any one shot must not exceed {MAX_PHYSICAL_TRANSITIONS} transitions",
            user,
        )
        self.assertIn("Shot 2 transition omitted its initial prop fact.", user)
        context = json.loads(user.split("REPAIR CONTEXT:\n", 1)[1])
        self.assertEqual(
            context["original_shots"][1]["prior_physical_state_context"]["status"],
            "draft_revisable",
        )
        self.assertEqual(schema["required"], ["event_shots", "initial_facts", "event_transitions"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["event_shots"]["required"], ["E1", "E2"])
        self.assertFalse(schema["properties"]["event_shots"]["additionalProperties"])
        event_transitions_schema = schema["properties"]["event_transitions"]
        self.assertEqual(event_transitions_schema["required"], ["E1", "E2"])
        self.assertFalse(event_transitions_schema["additionalProperties"])
        transition_array_schema = event_transitions_schema["properties"]["E1"]
        self.assertEqual(transition_array_schema["maxItems"], MAX_PHYSICAL_TRANSITIONS)
        transition_schema = transition_array_schema["items"]
        self.assertEqual(
            set(transition_schema["properties"]),
            {"subject_id", "predicate", "to_value", "action", "operation"},
        )
        self.assertEqual(transition_schema["properties"]["operation"]["enum"], ["establish"])
        self.assertEqual(
            transition_schema["required"],
            ["subject_id", "predicate", "to_value", "action"],
        )
        self.assertNotIn("from_value", transition_schema["properties"])
        initial_facts_schema = schema["properties"]["initial_facts"]
        self.assertEqual(initial_facts_schema["type"], "object")
        self.assertEqual(initial_facts_schema["required"], ["char_0", "char_1"])
        self.assertFalse(initial_facts_schema["additionalProperties"])
        for character_schema in initial_facts_schema["properties"].values():
            self.assertEqual(character_schema["required"], [])
            self.assertEqual(
                character_schema["properties"]["grounded"]["enum"],
                ["grounded", "airborne"],
            )
            self.assertTrue(all(
                field_schema["maxLength"] == 80
                for field_schema in character_schema["properties"].values()
            ))

    def test_compact_object_facts_flatten_in_stable_order_and_replay(self):
        response = _compact_event_response()

        patched = apply_compact_source_physical_patch(_rows(), response)

        opening = patched[0]["physical_state"]["opening"]
        self.assertEqual(len(opening), 11)
        self.assertEqual(
            [(fact["subject_id"], fact["predicate"]) for fact in opening],
            [
                (character_id, predicate)
                for character_id, predicates in (
                    ("char_0", ("support", "contact", "posture", "location", "grounded", "prop")),
                    ("char_1", ("support", "contact", "posture", "location", "grounded")),
                )
                for predicate in predicates
            ],
        )
        self.assertEqual(opening[1]["value"], "none")
        self.assertEqual(patched[0]["physical_state"]["transitions"][0]["from_value"], "four feet on ledge")
        self.assertEqual(patched[0]["physical_state"]["transitions"][0]["to_value"], "lowered shoulders")
        self.assertNotIn("event_id", patched[0]["physical_state"]["transitions"][0])
        self.assertEqual(patched[1]["physical_state"]["transitions"][0]["from_value"], "both paws on ledge")
        self.assertEqual(patched[1]["physical_state"]["transitions"][1]["from_value"], "none")
        self.assertEqual(patched[1]["physical_state"]["transitions"][1]["to_value"], "saddle strap")

    def test_compact_v10_sparse_facts_can_be_established_at_their_exact_event(self):
        source_contract = {
            "events": [
                {"event_id": "E1", "text": "The Kitten and Dragon meet on the ledge."},
                {
                    "event_id": "E2",
                    "text": (
                        "The Kitten climbs up using the saddle strap and settles "
                        "astride its back, with a sword sheathed at its left hip."
                    ),
                },
                {"event_id": "E3", "text": "They wait at the castle gate."},
            ],
        }
        rows = _rows()
        rows.append(copy.deepcopy(rows[-1]))
        for row in rows:
            row["_director_source_contract"] = source_contract
        initial_facts = _compact_object_response()["initial_facts"]
        initial_facts["char_0"].pop("support")
        initial_facts["char_0"].pop("contact")
        response = {
            "event_shots": {"E1": 1, "E2": 2, "E3": 3},
            "initial_facts": initial_facts,
            "event_transitions": {
                "E1": [],
                "E2": [
                    {
                        "operation": "establish",
                        "subject_id": "char_0",
                        "predicate": "support",
                        "to_value": "astride char_1",
                        "action": "settles astride its back",
                    },
                    {
                        "operation": "establish",
                        "subject_id": "char_0",
                        "predicate": "contact",
                        "to_value": "saddle strap",
                        "action": "grips the leather saddle strap",
                    },
                ],
                "E3": [],
            },
        }

        patched = apply_compact_source_physical_patch(rows, response)

        second_state = patched[1]["physical_state"]
        opening_keys = {
            (fact["subject_id"], fact["predicate"])
            for fact in second_state["opening"]
        }
        closing = {
            (fact["subject_id"], fact["predicate"]): fact["value"]
            for fact in second_state["closing"]
        }
        self.assertNotIn(("char_0", "support"), opening_keys)
        self.assertNotIn(("char_0", "contact"), opening_keys)
        self.assertEqual(closing[("char_0", "support")], "astride char_1")
        self.assertEqual(closing[("char_0", "contact")], "saddle strap")
        self.assertEqual(
            [(item["operation"], item["event_id"]) for item in second_state["transitions"]],
            [("establish", "E2"), ("establish", "E2")],
        )
        next_opening_keys = {
            (fact["subject_id"], fact["predicate"])
            for fact in patched[2]["physical_state"]["opening"]
        }
        self.assertIn(("char_0", "support"), next_opening_keys)
        self.assertIn(("char_0", "contact"), next_opening_keys)
        assigned = [
            [event for event in source_contract["events"] if event["event_id"] == event_id]
            for event_id in ("E1", "E2", "E3")
        ]
        self.assertEqual(source_physical_state_issues(patched, assigned), [])
        self.assertEqual(validate_physical_state_sequence(patched), [])

    def test_compact_patch_does_not_establish_contact_from_proximity(self):
        source_contract = {
            "events": [
                {"event_id": "E1", "text": "The Dragon lowers its shoulder."},
                {
                    "event_id": "E2",
                    "text": (
                        "The Kitten notices the Dragon, walks beside it, and "
                        "waves beside it."
                    ),
                },
            ],
        }
        rows = _rows()
        for row in rows:
            row["_director_source_contract"] = source_contract

        response = _compact_event_response()
        response["initial_facts"]["char_0"].pop("contact")
        response["event_transitions"]["E2"] = [{
            "operation": "establish",
            "subject_id": "char_0",
            "predicate": "contact",
            "to_value": "Dragon",
            "action": "waves beside the Dragon",
        }]

        assigned = [
            [source_contract["events"][0]],
            [source_contract["events"][1]],
        ]
        patched = apply_compact_source_physical_patch(rows, response)
        issues = source_physical_state_issues(patched, assigned)
        self.assertTrue(any(
            "establishment transition" in issue and "not directly supported" in issue
            and "char_0 contact" in issue
            for issue in issues
        ), issues)

        inferred = copy.deepcopy(response)
        del inferred["event_transitions"]["E2"][0]["operation"]
        with self.assertRaisesRegex(
            ValueError,
            "unknown physical fact key char_0 contact",
        ):
            apply_compact_source_physical_patch(rows, inferred)

    def test_compact_establish_rejects_overwrite_and_non_event_keyed_input(self):
        response = _compact_event_response()
        response["event_transitions"]["E2"][0]["operation"] = "establish"
        response["event_transitions"]["E2"][0]["to_value"] = "astride char_1"
        with self.assertRaisesRegex(ValueError, "cannot establish existing physical fact key"):
            apply_compact_source_physical_patch(_rows(), response)

        legacy = _compact_response()
        legacy["shot_transitions"][0]["transitions"][0]["operation"] = "establish"
        with self.assertRaisesRegex(ValueError, "unsupported keys: operation"):
            apply_compact_source_physical_patch(_rows(), legacy)

    def test_event_keyed_transitions_route_once_and_replay_in_source_order(self):
        response = _compact_event_response()
        response["event_shots"] = {"E1": 1, "E2": 1}
        response["event_transitions"] = {
            # Deliberately reverse JSON insertion order; immutable contract order
            # controls replay, not the response object's key order.
            "E2": [{
                "subject_id": "char_0",
                "predicate": "prop",
                "to_value": "sheathed sword",
                "action": "returns the sword to its sheath",
            }],
            "E1": [{
                "subject_id": "char_0",
                "predicate": "prop",
                "to_value": "holding sword",
                "action": "draws the sword",
            }],
        }

        patched = apply_compact_source_physical_patch(_rows(), response)

        transitions = patched[0]["physical_state"]["transitions"]
        self.assertEqual([item["to_value"] for item in transitions], [
            "holding sword", "sheathed sword",
        ])
        self.assertEqual(patched[0]["source_event_ids"], ["E1", "E2"])
        self.assertEqual(patched[1]["physical_state"]["transitions"], [])
        self.assertTrue(all("event_id" not in item for item in transitions))

    def test_event_transition_keys_and_representation_are_exact(self):
        response = _compact_event_response()
        del response["event_transitions"]["E2"]
        with self.assertRaisesRegex(ValueError, "exactly match immutable source event IDs"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_event_response()
        response["event_transitions"]["E3"] = []
        with self.assertRaisesRegex(ValueError, "exactly match immutable source event IDs"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_event_response()
        response["event_transitions"]["E1"][0]["event_id"] = "E2"
        with self.assertRaisesRegex(ValueError, "unsupported keys: event_id"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_event_response()
        response["shot_transitions"] = _compact_object_response()["shot_transitions"]
        with self.assertRaisesRegex(ValueError, "exactly one of event_transitions or shot_transitions"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_event_response()
        del response["event_transitions"]
        with self.assertRaisesRegex(ValueError, "exactly one of event_transitions or shot_transitions"):
            apply_compact_source_physical_patch(_rows(), response)

    def test_event_transition_cap_is_enforced_after_shot_routing(self):
        response = _compact_event_response()
        response["event_shots"] = {"E1": 1, "E2": 1}
        first_count = MAX_PHYSICAL_TRANSITIONS // 2 + 1
        second_count = MAX_PHYSICAL_TRANSITIONS + 1 - first_count

        def make_transitions(count: int, prefix: str) -> list[dict]:
            return [{
                "subject_id": "char_0",
                "predicate": "prop",
                "to_value": f"{prefix} {index}",
                "action": f"changes prop state {prefix} {index}",
            } for index in range(count)]

        response["event_transitions"] = {
            "E1": make_transitions(first_count, "first"),
            "E2": make_transitions(second_count, "second"),
        }

        with self.assertRaisesRegex(
            ValueError,
            f"event_transitions routed to shot 1 has more than {MAX_PHYSICAL_TRANSITIONS} transitions",
        ):
            apply_compact_source_physical_patch(_rows(), response)

    def test_compact_initial_fact_object_has_exact_membership_and_sparse_states(self):
        response = _compact_object_response()
        del response["initial_facts"]["char_1"]
        with self.assertRaisesRegex(ValueError, "exactly match known cast character IDs"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_object_response()
        response["initial_facts"]["char_2"] = dict(response["initial_facts"]["char_0"])
        with self.assertRaisesRegex(ValueError, "exactly match known cast character IDs"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_object_response()
        del response["initial_facts"]["char_0"]["contact"]
        response["shot_transitions"][1]["transitions"] = [
            transition for transition in response["shot_transitions"][1]["transitions"]
            if transition["predicate"] != "contact"
        ]
        patched = apply_compact_source_physical_patch(_rows(), response)
        self.assertNotIn(
            ("char_0", "contact"),
            {(fact["subject_id"], fact["predicate"])
             for fact in patched[0]["physical_state"]["opening"]},
        )

        response = _compact_object_response()
        response["initial_facts"] = {"char_0": {}, "char_1": {}}
        response["shot_transitions"] = [
            {"shot_index": 1, "transitions": []},
            {"shot_index": 2, "transitions": []},
        ]
        patched = apply_compact_source_physical_patch(_rows(), response)
        self.assertEqual(patched[0]["physical_state"]["opening"], [])

        response = _compact_object_response()
        response["initial_facts"]["char_0"]["contact"] = "unspecified"
        with self.assertRaisesRegex(ValueError, "cannot use an unknown placeholder"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_object_response()
        response["initial_facts"]["char_0"]["unknown"] = "unspecified"
        with self.assertRaisesRegex(ValueError, "unsupported predicates: unknown"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_object_response()
        response["initial_facts"]["char_0"] = []
        with self.assertRaisesRegex(TypeError, "must be an object"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_object_response()
        response["initial_facts"] = 7
        with self.assertRaisesRegex(TypeError, "must be an array or compact character-state object"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_object_response()
        response["initial_facts"]["char_0"]["grounded"] = "astride"
        with self.assertRaisesRegex(ValueError, "exactly 'grounded' or 'airborne'"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_object_response()
        response["initial_facts"]["char_0"]["contact"] = "x" * 81
        with self.assertRaisesRegex(ValueError, "80 characters"):
            apply_compact_source_physical_patch(_rows(), response)

    def test_support_cycles_are_rejected_but_one_way_support_is_valid(self):
        row = {
            "subjects_on_screen": [
                {"character_id": "char_0"}, {"character_id": "char_1"},
            ],
            "physical_state": {
                "opening": [
                    {"subject_id": "char_0", "predicate": "support", "value": "astride char_1"},
                    {"subject_id": "char_1", "predicate": "support", "value": "supporting char_0"},
                ],
                "transitions": [],
                "closing": [
                    {"subject_id": "char_0", "predicate": "support", "value": "astride char_1"},
                    {"subject_id": "char_1", "predicate": "support", "value": "supporting char_0"},
                ],
            },
        }
        issues = validate_physical_state_sequence([row])
        self.assertTrue(any("cyclic actor support relation" in issue for issue in issues), issues)

        row["physical_state"]["opening"][1]["value"] = "stone ledge"
        row["physical_state"]["closing"][1]["value"] = "stone ledge"
        self.assertEqual(validate_physical_state_sequence([row]), [])
        row["physical_state"]["opening"][0]["value"] = "not supported by char_0"
        row["physical_state"]["closing"][0]["value"] = "not supported by char_0"
        self.assertEqual(validate_physical_state_sequence([row]), [])

    def test_compact_transition_retains_exact_source_event_for_narrow_posture_guard(self):
        response = _compact_event_response()
        response["event_transitions"]["E1"] = [{
            "subject_id": "char_1",
            "predicate": "support",
            "to_value": "char_0",
            "action": "lowers its shoulder",
        }]
        patched = apply_compact_source_physical_patch(_rows(), response)
        self.assertEqual(patched[0]["_director_transition_event_ids"], ["E1"])
        issues = source_contract_issues(patched, _contract())
        self.assertTrue(any("lowering posture/action is not supported" in issue for issue in issues), issues)

        response = _compact_event_response()
        response["event_transitions"]["E1"] = [{
            "subject_id": "char_1",
            "predicate": "posture",
            "to_value": "lowered shoulders",
            "action": "lowers its shoulder",
        }]
        patched = apply_compact_source_physical_patch(_rows(), response)
        issues = source_contract_issues(patched, _contract())
        self.assertFalse(any("lowering posture/action is not supported" in issue for issue in issues), issues)

    def test_compact_object_cast_sizes_preserve_twelve_fact_cap_and_fallback(self):
        core = ("support", "contact", "grounded")
        for count in (3, 4):
            rows = _rows_with_cast_count(count)
            _, _, schema = build_compact_source_physical_repair_request(
                rows, _contract(), [],
            )
            initial_schema = schema["properties"]["initial_facts"]
            self.assertEqual(initial_schema["type"], "object")
            self.assertEqual(initial_schema["required"], [f"char_{i}" for i in range(count)])
            for character_schema in initial_schema["properties"].values():
                self.assertEqual(character_schema["required"], [])
                self.assertTrue(set(core).issubset(character_schema["properties"]))
            if count == 3:
                self.assertIn("posture", initial_schema["properties"]["char_0"]["properties"])
                response = _compact_object_response(("char_0", "char_1", "char_2"))
                response["initial_facts"]["char_0"]["posture"] = "standing"
                response["initial_facts"]["char_1"]["location"] = "stone ledge"
                response["initial_facts"]["char_2"]["prop"] = "none"
                patched = apply_compact_source_physical_patch(rows, response)
                self.assertEqual(len(patched[0]["physical_state"]["opening"]), 12)

                response["initial_facts"]["char_2"]["location"] = "stone ledge"
                with self.assertRaisesRegex(ValueError, "more than 12 facts"):
                    apply_compact_source_physical_patch(rows, response)
            else:
                self.assertEqual(set(initial_schema["properties"]["char_0"]["properties"]), set(core))
                response = _compact_object_response(tuple(f"char_{i}" for i in range(count)))
                patched = apply_compact_source_physical_patch(rows, response)
                self.assertEqual(len(patched[0]["physical_state"]["opening"]), 12)

        rows = _rows_with_cast_count(5)
        _, _, schema = build_compact_source_physical_repair_request(rows, _contract(), [])
        self.assertEqual(schema["properties"]["initial_facts"]["type"], "array")
        self.assertEqual(schema["properties"]["initial_facts"]["maxItems"], 12)
        self.assertEqual(
            schema["required"], ["event_shots", "initial_facts", "event_transitions"],
        )
        response = _compact_event_response()
        response["initial_facts"] = [
            {
                "subject_id": character_id,
                "predicate": predicate,
                "value": value,
            }
            for character_id, state in response["initial_facts"].items()
            for predicate, value in state.items()
        ]
        patched = apply_compact_source_physical_patch(rows, response)
        self.assertEqual(len(patched[0]["physical_state"]["opening"]), 11)
        self.assertTrue(apply_compact_source_physical_patch(rows, _compact_response()))

    def test_compact_object_requires_event_ids_in_the_allocated_row_and_order(self):
        response = _compact_object_response()
        response["shot_transitions"][0]["transitions"][0]["event_id"] = "E2"
        with self.assertRaisesRegex(ValueError, "not assigned to shot 1"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_object_response()
        del response["shot_transitions"][0]["transitions"][0]["event_id"]
        with self.assertRaisesRegex(ValueError, "missing required keys: event_id"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_object_response()
        response["event_shots"] = {"E1": 1, "E2": 1}
        first_event_transition = response["shot_transitions"][0]["transitions"][0]
        second_event_transition = response["shot_transitions"][1]["transitions"][0]
        response["shot_transitions"][0]["transitions"] = [
            second_event_transition,
            first_event_transition,
        ]
        response["shot_transitions"][1]["transitions"] = []
        with self.assertRaisesRegex(ValueError, "follow source-event order"):
            apply_compact_source_physical_patch(_rows(), response)

    def test_compact_patch_replays_facts_and_preserves_other_fields(self):
        rows = _rows()
        rows[1]["physical_state"] = {
            **_physical_state(),
            "continuity": "time_jump",
        }
        original = copy.deepcopy(rows)
        response = _compact_response()

        patched = apply_compact_source_physical_patch(rows, response)

        self.assertEqual(rows, original)
        self.assertIsNot(patched, rows)
        self.assertIsNot(patched[0], rows[0])
        self.assertEqual(patched[0]["source_event_ids"], ["E1"])
        self.assertEqual(patched[1]["source_event_ids"], ["E2"])
        self.assertEqual(
            patched[0]["physical_state"]["opening"],
            response["initial_facts"],
        )
        self.assertEqual(
            patched[1]["physical_state"]["opening"],
            patched[0]["physical_state"]["closing"],
        )
        first_transition = patched[0]["physical_state"]["transitions"][0]
        self.assertEqual(first_transition["from_value"], "sheathed sword")
        self.assertEqual(first_transition["to_value"], "holding sword")
        second_transition = patched[1]["physical_state"]["transitions"][0]
        self.assertEqual(second_transition["from_value"], "holding sword")
        self.assertEqual(second_transition["to_value"], "sheathed sword")
        self.assertEqual(
            patched[1]["physical_state"]["closing"][-1]["value"],
            "sheathed sword",
        )
        self.assertEqual(patched[1]["physical_state"]["continuity"], "time_jump")
        for key in (
            "title", "duration_sec", "scene_goal", "spatial_setup",
            "continuity_strategy", "subjects_on_screen", "action_beats",
            "ending_beat", "dialogue_beats", "camera_plan", "video_prompt",
        ):
            self.assertEqual(patched[0][key], original[0][key], key)

    def test_compact_patch_requires_ordered_complete_event_mapping_and_shot_rows(self):
        response = _compact_response()
        response["event_shots"] = {"E1": 2, "E2": 1}
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_response()
        response["event_shots"] = {"E1": 1, "E3": 2}
        with self.assertRaisesRegex(ValueError, "E1..En"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_response()
        response["shot_transitions"][1]["shot_index"] = 1
        with self.assertRaisesRegex(ValueError, "shot_index values"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_response()
        response["shot_transitions"].pop()
        with self.assertRaisesRegex(ValueError, "count must match"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_response()
        response["shot_transitions"][0]["transitions"][0]["from_value"] = "invented"
        with self.assertRaisesRegex(ValueError, "unsupported keys: from_value"):
            apply_compact_source_physical_patch(_rows(), response)

    def test_compact_patch_rejects_inferred_or_malformed_physical_state(self):
        response = _compact_response()
        response["shot_transitions"][0]["transitions"][0]["predicate"] = "location"
        with self.assertRaisesRegex(ValueError, "unknown physical fact key"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_response()
        response["shot_transitions"][0]["transitions"][0]["subject_id"] = "prop_sword"
        with self.assertRaisesRegex(ValueError, "not a known cast character_id"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_response()
        response["initial_facts"].append(dict(response["initial_facts"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate subject_id/predicate"):
            apply_compact_source_physical_patch(_rows(), response)

        response = _compact_response()
        response["shot_transitions"][0]["transitions"].append({
            "subject_id": "char_0",
            "predicate": "grounded",
            "to_value": "astride char_1",
            "action": "settles on the mount",
        })
        with self.assertRaisesRegex(ValueError, "exactly 'grounded' or 'airborne'"):
            apply_compact_source_physical_patch(_rows(), response)

    def test_compact_patch_empty_contract_and_ledger_is_valid(self):
        rows = _rows()[:1]
        _, _, schema = build_compact_source_physical_repair_request(
            rows, {"events": [], "audio_policy": {}}, [],
        )
        self.assertEqual(schema["properties"]["event_shots"]["properties"], {})
        self.assertEqual(schema["properties"]["event_shots"]["required"], [])
        self.assertEqual(schema["properties"]["event_transitions"]["properties"], {})
        patched = apply_compact_source_physical_patch(rows, {
            "event_shots": {},
            "initial_facts": [],
            "event_transitions": {},
        })
        self.assertEqual(patched[0]["source_event_ids"], [])
        self.assertEqual(patched[0]["physical_state"]["opening"], [])
        self.assertEqual(patched[0]["physical_state"]["closing"], [])

    def test_compact_apply_checks_exact_embedded_contract_event_keys(self):
        rows = _rows()
        rows[0]["_director_source_contract"] = _contract()
        rows[1]["_director_source_contract"] = _contract()
        response = _compact_response()
        response["event_shots"] = {"E1": 1, "E3": 2}
        with self.assertRaisesRegex(ValueError, "exactly match"):
            apply_compact_source_physical_patch(rows, response)


if __name__ == "__main__":
    unittest.main()
