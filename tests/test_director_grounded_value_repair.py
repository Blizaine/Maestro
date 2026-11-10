"""A malformed enum may be repaired without rewriting the authored film."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.contract_repair import (
    apply_grounded_value_patch,
    build_grounded_value_repair_request,
    invalid_grounded_value_fields,
)
from services.director.physical_continuity import normalize_physical_state
from services.director.planners.short_film import ShortFilmPlanner


def _rows():
    return [{
        "title": "The rider", "duration_sec": 10,
        "dialogue_beats": [{"spoken_text": "We are flying!"}],
        "source_event_ids": ["E1"], "action_beats": ["The mount takes off"],
        "spatial_setup": "The rider sits astride a grounded mount",
        "closing_blocking": "The rider remains astride the flying mount",
        "physical_state": {
            "opening": [
                {"subject_id": "rider", "predicate": "grounded", "value": "astride"},
                {"subject_id": "rider", "predicate": "support", "value": "astride mount"},
            ],
            "transitions": [{
                "subject_id": "rider", "predicate": "grounded",
                "from_value": "on the ground", "to_value": "in flight",
                "action": "The mount takes off with the rider astride",
            }],
            "closing": [{"subject_id": "rider", "predicate": "grounded", "value": "airborne"}],
        },
    }]


class GroundedValueRepairTests(unittest.TestCase):
    def test_schema_constrains_each_target_and_keeps_valid_neighbor_anchors(self):
        _, user, schema = build_grounded_value_repair_request(
            _rows(), story_description="The mount takes off with its rider.",
            screenplay="RIDER: We are flying!", issues=[],
        )
        self.assertEqual(set(schema["required"]), {
            "shot_1_opening_1_value", "shot_1_transitions_1_from_value",
            "shot_1_transitions_1_to_value",
        })
        self.assertFalse(schema["additionalProperties"])
        for target in schema["properties"].values():
            self.assertEqual(target["enum"], ["grounded", "airborne", "unresolved"])
        context = json.loads(user.split("REPAIR CONTEXT:\n", 1)[1])
        self.assertEqual(context["fixed_grounded_anchors"][0]["value"], "airborne")
        self.assertEqual(context["shots"][0]["closing_blocking"], _rows()[0]["closing_blocking"])

    def test_takeoff_endpoints_repair_without_changing_dialogue_support_or_actions(self):
        rows = _rows()
        original = copy.deepcopy(rows)
        expected = copy.deepcopy(rows)
        state = expected[0]["physical_state"]
        state["opening"][0]["value"] = "grounded"
        state["transitions"][0]["from_value"] = "grounded"
        state["transitions"][0]["to_value"] = "airborne"
        repaired = apply_grounded_value_patch(rows, {
            "shot_1_opening_1_value": "grounded",
            "shot_1_transitions_1_from_value": "grounded",
            "shot_1_transitions_1_to_value": "airborne",
        })
        self.assertEqual(repaired, expected)
        self.assertEqual(rows, original)
        self.assertIsNotNone(normalize_physical_state(repaired[0]["physical_state"]))

    def test_rejects_partial_extra_unresolved_and_nonliteral_patches_atomically(self):
        rows = _rows()
        original = copy.deepcopy(rows)
        valid = {
            "shot_1_opening_1_value": "grounded",
            "shot_1_transitions_1_from_value": "grounded",
            "shot_1_transitions_1_to_value": "airborne",
        }
        for patch in (
            {}, {**valid, "dialogue_beats": []},
            {**valid, "shot_1_opening_1_value": "unresolved"},
            {**valid, "shot_1_opening_1_value": "seated"},
            {**valid, "shot_1_opening_1_value": True},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                apply_grounded_value_patch(rows, patch)
            self.assertEqual(rows, original)

    def test_optional_establishment_null_and_non_grounded_predicates_are_untouched(self):
        rows = _rows()
        state = rows[0]["physical_state"]
        state["opening"][0]["value"] = " Grounded "
        transition = state["transitions"][0]
        transition.update(from_value=None, to_value="airborne", operation="establish")
        self.assertEqual(invalid_grounded_value_fields(rows), {})

    def test_real_planner_json_parser_preserves_the_flat_enum_patch(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps({
                "shot_1_opening_1_value": "grounded",
                "shot_1_transitions_1_from_value": "grounded",
                "shot_1_transitions_1_to_value": "airborne",
            })

        planner = ShortFilmPlanner(llm_generate=generate)
        repaired, attempts = planner._repair_h3_grounded_state_values(
            _rows(), story_description="The mount takes off with its rider.",
            screenplay="RIDER: We are flying!", planner_token_budget=4096,
            request_budget=2,
        )

        self.assertEqual(attempts, 1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["enable_thinking"])
        self.assertEqual(calls[0]["json_schema"]["type"], "object")
        self.assertEqual(repaired[0]["physical_state"]["transitions"][0]["to_value"], "airborne")


if __name__ == "__main__":
    unittest.main()
