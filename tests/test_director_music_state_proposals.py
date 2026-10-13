"""Strict physical-state helpers and H3 music planner integration regressions."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.physical_continuity import PHYSICAL_STATE_JSON_SCHEMA
from services.director.music_narrative import build_music_narrative_contract
from services.director.planners.music_video import (
    MusicVideoPlanner,
    _bridge_h3_conflicting_openings,
    _h3_source_support_patch_spec,
    _h3_support_ground_start,
    _h3_support_value_matches_target,
    _has_explicit_physical_relation,
    _materialize_h3_physical_state_proposals,
    _music_shot_schema,
    _source_establishes_physical_relation,
    _try_repair_h3_physical_state_only,
)


def fact(subject_id, predicate, value):
    return {"subject_id": subject_id, "predicate": predicate, "value": value}


def transition(subject_id, predicate, to_value, action):
    return {
        "subject_id": subject_id,
        "predicate": predicate,
        "to_value": to_value,
        "action": action,
    }


def proposal_row(
    *, opening=(), transitions=(), continuity="continuous", strategy="continuous"
):
    return {
        "continuity_strategy": strategy,
        "subjects_on_screen": [
            {
                "character_id": "lead_performer",
                "speaker_name": "Ari",
                "visual_description": "Ari in a red coat",
                "performance_role": "non_vocal",
            }
        ],
        "scene_goal": "A close performance shot",
        "action_beats": ["Ari adjusts the sword grip"],
        "video_prompt": "Ari adjusts the sword grip in close view.",
        "physical_state": {
            "opening": list(opening),
            "transitions": list(transitions),
            "continuity": continuity,
        },
    }


def full_state(opening, transitions=(), closing=None, continuity="continuous"):
    return {
        "opening": list(opening),
        "transitions": [
            {
                "subject_id": item["subject_id"],
                "predicate": item["predicate"],
                "from_value": item["from_value"],
                "to_value": item["to_value"],
                "action": item["action"],
            }
            for item in transitions
        ],
        "closing": list(closing if closing is not None else opening),
        "continuity": continuity,
    }


def complete_shot(prompt, *, opening=(), transitions=(), ending="Ari pauses"):
    row = proposal_row(opening=opening, transitions=transitions)
    row.update(
        {
            "scene_type": "performance",
            "environment": "A rooftop stage at night",
            "visual_style": "cinematic",
            "lighting": "warm stage lights",
            "mood": "focused",
            "ending_beat": ending,
            "video_prompt": prompt,
            "camera_plan": {"framing": "medium shot"},
            "window_prompts": [],
        }
    )
    return row


class MusicStateProposalTests(unittest.TestCase):
    def _repair_callbacks(self, planner, clips):
        planner._uses_generated_shot_images = False
        with patch.object(
            planner, "_run_checkpointed_json_batches", return_value=[]
        ) as batches:
            planner._plan_with_llm(
                clips=clips,
                clip_contexts=[f"Global clip {index + 1}" for index in range(len(clips))],
                scene_description="Ari performs in a rooftop rehearsal room.",
                bpm=120,
                has_reference=False,
                reference_image_path=None,
                char_profiles=[],
                performer_map={},
                video_model="minimax_h3_ref2va_fused_turbo",
                shot_image_policy="prompt_only",
            )
        return batches.call_args.kwargs

    def _invalid_materialized_row(self, *, source_event_ids=None):
        row = complete_shot(
            "Ari raises the lantern above the rooftop stage.",
            transitions=[
                transition(
                    "lead_performer",
                    "contact",
                    "holding the lantern overhead",
                    "Ari raises the lantern",
                )
            ],
        )
        row["source_event_ids"] = list(source_event_ids or [])
        return _materialize_h3_physical_state_proposals(
            [row], preserve_invalid_tail=True,
        )[0]

    @staticmethod
    def _previous_state_row():
        row = complete_shot("Ari continues the rooftop performance.")
        row["physical_state"] = full_state(
            [fact("lead_performer", "posture", "upright")],
        )
        return row

    @staticmethod
    def _bridge_row(opening_value="leaping onto the ledge"):
        row = complete_shot(
            "Ari leaps onto the ledge and settles into a crouch.",
            opening=[fact("lead_performer", "posture", opening_value)],
        )
        row["action_beats"] = [
            "Ari leaps onto the ledge.",
            "Ari settles into a crouch.",
        ]
        row["video_prompt"] = (
            "Ari leaps onto the ledge, then settles into a crouch."
        )
        return row

    @staticmethod
    def _bridge_previous_row():
        row = complete_shot("Ari watches the rooftop stage.")
        row["physical_state"] = full_state(
            [fact("lead_performer", "posture", "looking over the valley")],
        )
        return row

    @staticmethod
    def _source_support_patch_case():
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        previous = MusicStateProposalTests._bridge_previous_row()
        grounded_contact = fact("lead_performer", "contact", "grounded")
        previous["physical_state"]["opening"].append(grounded_contact)
        previous["physical_state"]["closing"].append(grounded_contact)
        row = MusicStateProposalTests._bridge_row("leaping onto the platform")
        row["action_beats"] = [
            "Ari leaps onto the platform.",
            "Ari climbs onto the platform.",
        ]
        row["video_prompt"] = (
            "Ari leaps onto the platform, then climbs onto it."
        )
        row["source_event_ids"] = ["M1"]
        clip = {"start": 10, "end": 20, "source_event_ids": ["M1"]}
        contract = {"events": [{
            "event_id": "M1",
            "text": "climbs onto the platform",
            "kind": "boarding",
            "actor_aliases": ["Ari"],
            "target_aliases": ["platform"],
        }]}
        allocation = [{"event_id": "M1", "clip_index": 1}]
        invalid = _materialize_h3_physical_state_proposals(
            [row],
            previous_physical_state=previous["physical_state"],
            preserve_invalid_tail=True,
        )[0]
        issue = (
            "clip 2 compact physical-state proposal could not be materialized: "
            + invalid["_h3_physical_state_proposal_error"]
        )
        return planner, previous, row, invalid, issue, clip, contract, allocation

    def test_h3_schema_is_compact_and_non_h3_keeps_legacy_schema(self):
        h3_schema = _music_shot_schema(
            1,
            include_image_fields=False,
            require_cast_identity=True,
            require_physical_state=True,
        )
        state_schema = h3_schema["items"]["properties"]["physical_state"]
        self.assertEqual(
            set(state_schema["properties"]),
            {"opening", "transitions"},
        )
        self.assertNotIn(
            "from_value",
            state_schema["properties"]["transitions"]["items"]["properties"],
        )
        legacy_schema = _music_shot_schema(1, include_image_fields=False)
        self.assertEqual(
            legacy_schema["items"]["properties"]["physical_state"],
            PHYSICAL_STATE_JSON_SCHEMA,
        )
        production_h3_schema = _music_shot_schema(
            1,
            include_image_fields=False,
            require_cast_identity=True,
            omit_physical_state=True,
            require_continuity_strategy=True,
            require_native_single_clip=True,
        )
        self.assertNotIn("physical_state", production_h3_schema["items"]["properties"])
        self.assertIn("continuity_strategy", production_h3_schema["items"]["required"])

    def test_new_proposal_derives_continuity_from_single_cut_strategy(self):
        for strategy, expected in (
            ("continuous", "continuous"),
            ("extend_previous", "continuous"),
            ("independent", "independent"),
        ):
            with self.subTest(strategy=strategy):
                row = proposal_row(strategy=strategy)
                del row["physical_state"]["continuity"]
                state = _materialize_h3_physical_state_proposals([row])[0][
                    "physical_state"
                ]
                self.assertEqual(state["continuity"], expected)
                self.assertEqual(state["opening"], [])
                self.assertEqual(state["closing"], [])

    def test_independent_new_proposal_never_carries_previous_facts(self):
        row = proposal_row(
            strategy="independent",
            opening=[fact("lead_performer", "contact", "holding lantern")],
        )
        del row["physical_state"]["continuity"]
        previous = full_state([fact("lead_performer", "contact", "holding sword")])
        state = _materialize_h3_physical_state_proposals(
            [row], previous_physical_state=previous,
        )[0]["physical_state"]
        self.assertEqual(state["opening"], row["physical_state"]["opening"])
        self.assertEqual(state["closing"], row["physical_state"]["opening"])

    def test_transition_values_and_final_closing_are_derived_in_order(self):
        source = [
            proposal_row(
                opening=[fact("lead_performer", "contact", "holding sword")],
                transitions=[
                    transition(
                        "lead_performer",
                        "contact",
                        "holding sword (tighter grip)",
                        "Ari tightens the sword grip",
                    ),
                    transition(
                        "lead_performer",
                        "contact",
                        "holding sword low",
                        "Ari lowers the sword",
                    ),
                ],
            )
        ]
        result = _materialize_h3_physical_state_proposals(source)
        state = result[0]["physical_state"]
        self.assertEqual(
            [item["from_value"] for item in state["transitions"]],
            ["holding sword", "holding sword (tighter grip)"],
        )
        self.assertEqual(
            state["closing"],
            [
                fact("lead_performer", "contact", "holding sword low"),
            ],
        )
        self.assertNotIn("closing", source[0]["physical_state"])
        self.assertNotIn("from_value", source[0]["physical_state"]["transitions"][0])

    def test_continuous_state_carries_all_facts_and_allows_new_opening_keys(self):
        previous = full_state(
            [
                fact("lead_performer", "contact", "holding sword"),
                fact("lead_performer", "posture", "upright"),
            ],
            closing=[
                fact("lead_performer", "contact", "holding sword"),
                fact("lead_performer", "posture", "upright"),
            ],
        )
        result = _materialize_h3_physical_state_proposals(
            [
                proposal_row(
                    opening=[fact("lead_performer", "contact", "holding sword")],
                    transitions=[
                        transition(
                            "lead_performer",
                            "contact",
                            "sheathed sword",
                            "Ari sheathes the sword",
                        )
                    ],
                )
            ],
            previous_physical_state=previous,
        )
        opening = result[0]["physical_state"]["opening"]
        self.assertEqual(
            opening,
            [
                fact("lead_performer", "contact", "holding sword"),
                fact("lead_performer", "posture", "upright"),
            ],
        )
        self.assertEqual(
            result[0]["physical_state"]["closing"],
            [
                fact("lead_performer", "contact", "sheathed sword"),
                fact("lead_performer", "posture", "upright"),
            ],
        )

        extended = _materialize_h3_physical_state_proposals(
            [
                proposal_row(
                    opening=[fact("lead_performer", "location", "near the stage")],
                )
            ],
            previous_physical_state=previous,
        )
        self.assertEqual(
            extended[0]["physical_state"]["opening"],
            [
                fact("lead_performer", "contact", "holding sword"),
                fact("lead_performer", "posture", "upright"),
                fact("lead_performer", "location", "near the stage"),
            ],
        )

    def test_prop_manipulation_mislabeled_as_posture_keeps_carried_posture(self):
        seated = fact("lead_performer", "posture", "seated near the dragon's neck")
        prop = fact("lead_performer", "prop", "twirling sword")
        previous = full_state([seated, prop])
        row = proposal_row(transitions=[transition(
            "lead_performer", "posture", "twirling sword",
            "twirls its sword in a tight circle",
        )])
        row["action_beats"] = ["Ari twirls its sword in a tight circle."]

        result = _materialize_h3_physical_state_proposals(
            [row], previous_physical_state=previous,
        )[0]["physical_state"]

        self.assertEqual(result["closing"], [seated, prop])
        self.assertEqual(result["transitions"], [])

    def test_real_posture_change_survives_when_actor_has_prop_fact(self):
        previous = full_state([
            fact("lead_performer", "posture", "seated near the dragon's neck"),
            fact("lead_performer", "prop", "twirling sword"),
        ])
        row = proposal_row(transitions=[transition(
            "lead_performer", "posture", "standing upright", "Ari stands upright.",
        )])

        result = _materialize_h3_physical_state_proposals(
            [row], previous_physical_state=previous,
        )[0]["physical_state"]

        self.assertEqual(result["transitions"][0]["to_value"], "standing upright")
        self.assertEqual(
            next(item["value"] for item in result["closing"]
                 if item["predicate"] == "posture"),
            "standing upright",
        )

    def test_ambiguous_or_unasserted_prop_phrases_are_not_normalized(self):
        previous = full_state([
            fact("lead_performer", "posture", "seated"),
            fact("lead_performer", "prop", "twirling sword"),
        ])
        for action in (
            "Ari plans to twirl the sword.",
            "Ari talks about twirling the sword.",
            "Ari twirls or waves the sword.",
            "Mira twirls the sword.",
        ):
            with self.subTest(action=action):
                row = proposal_row(transitions=[transition(
                    "lead_performer", "posture", "twirling sword", action,
                )])
                row["action_beats"] = [action]
                result = _materialize_h3_physical_state_proposals(
                    [row], previous_physical_state=previous,
                )[0]["physical_state"]
                self.assertEqual(result["transitions"][0]["to_value"], "twirling sword")

    def test_metadata_bridge_uses_exact_carried_value_and_authored_opening_action(self):
        previous = self._bridge_previous_row()
        row = self._bridge_row()
        original = copy.deepcopy(row)
        candidate = _bridge_h3_conflicting_openings(
            row, previous["physical_state"]["closing"],
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(row, original)
        self.assertEqual(candidate["action_beats"], original["action_beats"])
        self.assertEqual(candidate["video_prompt"], original["video_prompt"])
        materialized = _materialize_h3_physical_state_proposals(
            [candidate],
            previous_physical_state=previous["physical_state"],
            preserve_invalid_tail=True,
        )[0]
        self.assertNotIn("_h3_physical_state_proposal_error", materialized)
        state = materialized["physical_state"]
        self.assertEqual(
            state["opening"],
            [fact("lead_performer", "posture", "looking over the valley")],
        )
        self.assertEqual(
            state["transitions"][0],
            {
                "subject_id": "lead_performer",
                "predicate": "posture",
                "from_value": "looking over the valley",
                "to_value": "leaping onto the ledge",
                "action": "Ari leaps onto the ledge.",
            },
        )

    def test_source_clean_metadata_bridge_skips_writer(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        previous = self._bridge_previous_row()
        row = self._bridge_row()
        invalid = _materialize_h3_physical_state_proposals(
            [row],
            previous_physical_state=previous["physical_state"],
            preserve_invalid_tail=True,
        )[0]
        issue = (
            "clip 2 compact physical-state proposal could not be materialized: "
            + invalid["_h3_physical_state_proposal_error"]
        )
        with patch.object(planner, "_call_llm_json", return_value=[]) as generate:
            result = _try_repair_h3_physical_state_only(
                planner,
                index=1,
                row=invalid,
                issue=issue,
                clip={"start": 10, "end": 20},
                previous=previous,
                contract={"events": []},
                allocation=[],
                include_image_fields=False,
            )

        self.assertEqual(generate.call_count, 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["physical_state"]["transitions"][0]["from_value"],
            "looking over the valley",
        )

    def test_grounded_source_support_obligation_is_patched_deterministically(self):
        (
            planner, previous, row, invalid, issue, clip, contract, allocation,
        ) = self._source_support_patch_case()
        original_visible = {
            key: copy.deepcopy(row.get(key))
            for key in ("action_beats", "video_prompt", "source_event_ids")
        }
        with patch.object(
            planner, "_call_llm_json",
            side_effect=AssertionError("verified additive patch should not call writer"),
        ) as generate:
            result = _try_repair_h3_physical_state_only(
                planner,
                index=1,
                row=invalid,
                issue=issue,
                clip=clip,
                previous=previous,
                contract=contract,
                allocation=allocation,
                include_image_fields=False,
            )

        self.assertEqual(generate.call_count, 0)
        self.assertEqual(len(result), 1)
        repaired = result[0]
        self.assertEqual(
            {key: repaired.get(key) for key in original_visible},
            original_visible,
        )
        state = repaired["physical_state"]
        support_opening = next(
            item for item in state["opening"] if item["predicate"] == "support"
        )
        self.assertEqual(
            support_opening,
            fact("lead_performer", "support", "on the ground"),
        )
        support_transition = next(
            item for item in state["transitions"] if item["predicate"] == "support"
        )
        self.assertEqual(
            support_transition["to_value"], "on platform",
        )
        self.assertEqual(
            support_transition["action"], "Ari climbs onto the platform.",
        )
        self.assertEqual(
            [item["predicate"] for item in state["transitions"]],
            ["posture", "support"],
        )
        row_issue = MusicVideoPlanner._music_video_row_issue(
            1,
            repaired,
            [previous],
            scene_description="Ari leaps onto and climbs onto a platform.",
            include_image_fields=False,
            require_stable_cast_identity=True,
            require_source_physical_state=True,
            require_native_single_clip=True,
            source_events_for_row=[{
                "event_id": "M1",
                "text": "climbs onto the platform",
                "actor_aliases": ["Ari"],
                "target_aliases": ["platform"],
            }],
        )
        self.assertIsNone(row_issue, row_issue)

    def test_support_patch_requires_ground_target_and_affirmative_owned_boarding(self):
        self.assertTrue(_h3_support_ground_start("on the ground"))
        for value in ("grounded", "not grounded", "above ground", "hovering over ground"):
            with self.subTest(value=value):
                self.assertFalse(_h3_support_ground_start(value))
        self.assertTrue(_h3_support_value_matches_target("on platform", ["platform"]))
        for value in (
            "contact with platform",
            "not on platform",
            "as if on platform",
            "hovering above platform",
            "on the platform as if supported",
        ):
            with self.subTest(value=value):
                self.assertFalse(
                    _h3_support_value_matches_target(value, ["platform"]),
                )

        _, _, row, _, _, _, contract, _ = self._source_support_patch_case()
        issues = [
            "source action 'climbs onto the platform' requires a support fact "
            "in physical_state.opening",
            "source action 'climbs onto the platform' requires a support fact "
            "in physical_state.closing",
        ]
        assigned_actions = [{
            "event_id": "M1",
            "text": contract["events"][0]["text"],
            "kind": "boarding",
            "actor_aliases": ["Ari"],
            "target_aliases": ["platform"],
        }]
        repair_seed = {
            "opening": [fact("lead_performer", "posture", "looking over the valley")],
            "transitions": [],
        }
        kwargs = {
            "issues": issues,
            "row": row,
            "cast_ids": [{
                "character_id": "lead_performer",
                "speaker_name": "Ari",
            }],
            "preceding_closing": [
                fact("lead_performer", "contact", "grounded"),
            ],
            "assigned_actions": assigned_actions,
            "repair_seed": repair_seed,
        }
        spec = _h3_source_support_patch_spec(**kwargs)
        self.assertEqual(spec["opening_value"], "on the ground")
        self.assertEqual(spec["to_value"], "on platform")
        self.assertEqual(spec["action"], "Ari climbs onto the platform.")

        bad_cases = []
        ungrounded = copy.deepcopy(kwargs)
        ungrounded["preceding_closing"] = [
            fact("lead_performer", "contact", "not grounded"),
        ]
        bad_cases.append(("ungrounded start", ungrounded))

        ambiguous_actor = copy.deepcopy(kwargs)
        ambiguous_actor["assigned_actions"][0]["actor_aliases"] = ["Ari", "Mira"]
        ambiguous_actor["cast_ids"].append({
            "character_id": "second_performer",
            "speaker_name": "Mira",
        })
        bad_cases.append(("ambiguous actor alias", ambiguous_actor))

        ambiguous_target = copy.deepcopy(kwargs)
        ambiguous_target["assigned_actions"][0]["target_aliases"] = [
            "platform", "the platform",
        ]
        bad_cases.append(("ambiguous target alias", ambiguous_target))

        wrong_owner = copy.deepcopy(kwargs)
        wrong_owner["row"]["action_beats"] = [
            "Ari leaps onto the platform.",
            "Mira climbs onto the platform.",
        ]
        bad_cases.append(("authored action has wrong owner", wrong_owner))

        hypothetical_source = copy.deepcopy(kwargs)
        hypothetical_source["assigned_actions"][0]["text"] = (
            "Ari plans to climb onto the platform"
        )
        bad_cases.append(("hypothetical source action", hypothetical_source))

        for label, case in bad_cases:
            with self.subTest(case=label):
                self.assertIsNone(_h3_source_support_patch_spec(**case))

    def test_metadata_bridge_fails_closed_for_unasserted_or_mismatched_actions(self):
        previous = self._bridge_previous_row()
        unsupported_beats = (
            "Ari talks about leaping onto the ledge.",
            "Ari remembers jumping onto the ledge.",
            "Ari plans to leap onto the ledge.",
            "Ari is shown leaping onto the ledge.",
            "Ari doesn't leap onto the ledge.",
            "Ari can't leap onto the ledge.",
            "Mira leaps onto the ledge.",
            "Ari leaps from the ledge toward the dragon.",
            "Ari stands beside the ledge.",
        )
        for beat in unsupported_beats:
            with self.subTest(beat=beat):
                row = self._bridge_row()
                row["action_beats"] = [beat, "Ari settles into a crouch."]
                self.assertIsNone(_bridge_h3_conflicting_openings(
                    row, previous["physical_state"]["closing"],
                ))

    def test_metadata_bridge_rejects_beat_reordering_and_later_same_key_destination(self):
        previous = self._bridge_previous_row()
        reordered = self._bridge_row()
        reordered["physical_state"]["opening"].append(
            fact("lead_performer", "contact", "gripping the ledge")
        )
        reordered["action_beats"] = [
            "Ari grips the ledge.",
            "Ari leaps onto the ledge.",
        ]
        self.assertIsNone(_bridge_h3_conflicting_openings(
            reordered, previous["physical_state"]["closing"],
        ))

        existing_early_transition = self._bridge_row()
        existing_early_transition["physical_state"]["opening"].append(
            fact("lead_performer", "contact", "holding a sword")
        )
        existing_early_transition["physical_state"]["transitions"] = [
            transition(
                "lead_performer", "contact", "twirling a sword",
                "Ari twirls the sword",
            )
        ]
        existing_early_transition["action_beats"] = [
            "Ari twirls the sword.",
            "Ari leaps onto the ledge.",
        ]
        self.assertIsNone(_bridge_h3_conflicting_openings(
            existing_early_transition, previous["physical_state"]["closing"],
        ))

        destination_after_same_key_transition = self._bridge_row()
        destination_after_same_key_transition["physical_state"]["transitions"] = [
            transition(
                "lead_performer", "posture", "upright on the rooftop",
                "Ari stands upright",
            ),
            transition(
                "lead_performer", "posture", "leaping onto the ledge",
                "Ari leaps onto the ledge",
            ),
        ]
        destination_after_same_key_transition["action_beats"] = [
            "Ari leaps onto the ledge.",
            "Ari stands upright on the rooftop.",
        ]
        self.assertIsNone(_bridge_h3_conflicting_openings(
            destination_after_same_key_transition,
            previous["physical_state"]["closing"],
        ))

        already_transitioned = self._bridge_row()
        already_transitioned["physical_state"]["transitions"] = [
            transition(
                "lead_performer", "posture", "leaping onto the ledge",
                "Ari leaps onto the ledge",
            )
        ]
        candidate = _bridge_h3_conflicting_openings(
            already_transitioned, previous["physical_state"]["closing"],
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(len(candidate["physical_state"]["transitions"]), 1)

    def test_source_obligations_fall_through_once_with_bridged_metadata_context(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        previous = self._bridge_previous_row()
        carried_support = fact(
            "lead_performer", "support", "standing on the ground",
        )
        previous["physical_state"]["opening"].append(carried_support)
        previous["physical_state"]["closing"].append(carried_support)
        row = self._bridge_row("leaping onto the platform")
        row["action_beats"] = [
            "Ari leaps onto the platform.",
            "Ari climbs onto the platform.",
        ]
        row["video_prompt"] = (
            "Ari leaps onto the platform, then climbs onto it."
        )
        row["source_event_ids"] = ["M1"]
        clip = {"start": 10, "end": 20, "source_event_ids": ["M1"]}
        contract = {"events": [{
            "event_id": "M1",
            "text": "Ari climbs onto the platform.",
            "kind": "action",
            "actor_aliases": ["Ari"],
            "target_aliases": ["platform"],
        }]}
        allocation = [{"event_id": "M1", "clip_index": 1}]
        invalid = _materialize_h3_physical_state_proposals(
            [row],
            previous_physical_state=previous["physical_state"],
            preserve_invalid_tail=True,
        )[0]
        initial_issue = invalid["_h3_physical_state_proposal_error"]
        issue = (
            "clip 2 compact physical-state proposal could not be materialized: "
            + initial_issue
        )
        repaired_state = {
            "opening": [
                fact("lead_performer", "posture", "looking over the valley"),
                fact("lead_performer", "support", "standing on the ground"),
            ],
            "transitions": [
                transition(
                    "lead_performer", "posture", "leaping onto the platform",
                    "Ari leaps onto the platform.",
                ),
                transition(
                    "lead_performer", "support", "standing on the platform",
                    "Ari climbs onto the platform.",
                ),
            ],
        }
        with patch.object(
            planner, "_call_llm_json", return_value=[{"physical_state": repaired_state}],
        ) as generate:
            result = _try_repair_h3_physical_state_only(
                planner,
                index=1,
                row=invalid,
                issue=issue,
                clip=clip,
                previous=previous,
                contract=contract,
                allocation=allocation,
                include_image_fields=False,
            )

        self.assertEqual(generate.call_count, 1)
        self.assertIsNotNone(result)
        repaired = result[0]
        state = repaired["physical_state"]
        self.assertEqual(
            state["opening"],
            [
                fact("lead_performer", "posture", "looking over the valley"),
                fact("lead_performer", "support", "standing on the ground"),
            ],
        )
        self.assertEqual(
            [item["to_value"] for item in state["transitions"]],
            ["leaping onto the platform", "standing on the platform"],
        )
        self.assertEqual(repaired["action_beats"], row["action_beats"])
        request = generate.call_args.kwargs
        context_text = request["user_prompt"]
        self.assertIn("bridge_candidate_source_physical_state_issues", context_text)
        self.assertIn("support", context_text)
        self.assertIn("climbs onto the platform", context_text)
        self.assertIn("repair_seed_compact_physical_state_proposal", context_text)
        self.assertIn("Preserve those bridges", context_text)
        self.assertIn("Returning the seed unchanged is invalid", context_text)
        self.assertIn("Closing is derived", context_text)
        self.assertIn("cannot be supplied as a field", context_text)
        self.assertIn("required opening relation may describe the supported", context_text)
        context = json.loads(context_text.rsplit("\n\n", 1)[-1])
        self.assertIn("support", context["validation_issue"])
        self.assertNotIn("conflicts with carried", context["validation_issue"])
        self.assertIn("conflicts with carried", context["resolved_original_issue"])
        self.assertIsNone(context["original_compact_physical_state_proposal"])
        self.assertEqual(context["bridged_conflicting_opening_keys"], [])
        self.assertEqual(
            context["assigned_source_actions"],
            [{
                "event_id": "M1",
                "text": "Ari climbs onto the platform.",
                "kind": "action",
                "actor_aliases": ["Ari"],
                "target_aliases": ["platform"],
            }],
        )
        known_keys = context["known_current_physical_keys"]
        self.assertEqual(len(known_keys), len({
            (item["subject_id"].casefold(), item["predicate"])
            for item in known_keys
        }))
        self.assertEqual(
            known_keys[0]["value"], "looking over the valley",
        )
        self.assertTrue(context["bridge_candidate_source_physical_state_issues"])
        self.assertEqual(
            [
                item["predicate"]
                for item in context["repair_seed_compact_physical_state_proposal"]["opening"]
            ],
            ["posture", "support"],
        )
        repair_seed = context["repair_seed_compact_physical_state_proposal"]
        self.assertEqual(set(repair_seed), {"opening", "transitions"})
        self.assertTrue(all(
            set(item) <= {"subject_id", "predicate", "to_value", "action"}
            for item in repair_seed["transitions"]
        ))
        self.assertNotIn("from_value", json.dumps(repair_seed))

        row_issue = MusicVideoPlanner._music_video_row_issue(
            1,
            repaired,
            [previous],
            scene_description="Ari performs on a rooftop stage.",
            include_image_fields=False,
            require_stable_cast_identity=True,
            require_source_physical_state=True,
            require_native_single_clip=True,
            source_events_for_row=[{
                "event_id": "M1",
                "text": "Ari climbs onto the platform.",
            }],
        )
        self.assertIsNone(row_issue, row_issue)

    def test_conflicting_carried_fact_is_rejected(self):
        previous = full_state(
            [fact("lead_performer", "contact", "holding sword")],
        )
        with self.assertRaisesRegex(ValueError, "conflicts with carried"):
            _materialize_h3_physical_state_proposals(
                [
                    proposal_row(
                        opening=[fact("lead_performer", "contact", "holding lantern")],
                    )
                ],
                previous_physical_state=previous,
            )

    def test_transition_cannot_infer_a_missing_current_fact(self):
        with self.assertRaisesRegex(ValueError, "no explicit opening/current fact"):
            _materialize_h3_physical_state_proposals(
                [
                    proposal_row(
                        transitions=[
                            transition(
                                "lead_performer",
                                "support",
                                "astride mount",
                                "Ari climbs aboard",
                            )
                        ]
                    ),
                ]
            )

    def test_independent_reset_requires_explicit_opening_when_prior_state_active(self):
        previous = full_state(
            [
                fact("lead_performer", "contact", "holding sword"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "must explicitly establish"):
            _materialize_h3_physical_state_proposals(
                [proposal_row(continuity="independent", strategy="independent")],
                previous_physical_state=previous,
            )
        reset = _materialize_h3_physical_state_proposals(
            [
                proposal_row(
                    opening=[fact("lead_performer", "location", "on stage")],
                    continuity="time_jump",
                    strategy="independent",
                )
            ],
            previous_physical_state=previous,
        )
        self.assertEqual(
            reset[0]["physical_state"]["opening"],
            [fact("lead_performer", "location", "on stage")],
        )

    def test_continuity_must_match_planned_cut_strategy(self):
        with self.assertRaisesRegex(ValueError, "requires independent or time_jump"):
            _materialize_h3_physical_state_proposals(
                [
                    proposal_row(continuity="continuous", strategy="independent"),
                ]
            )
        with self.assertRaisesRegex(ValueError, "requires continuous physical"):
            _materialize_h3_physical_state_proposals(
                [
                    proposal_row(continuity="time_jump", strategy="extend_previous"),
                ]
            )

    def test_malformed_duplicates_grounded_values_and_oversize_are_rejected(self):
        duplicate = proposal_row(
            opening=[
                fact("lead_performer", "posture", "upright"),
                fact("lead_performer", "posture", "crouched"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicates a physical fact key"):
            _materialize_h3_physical_state_proposals([duplicate])

        invalid_grounded = proposal_row(
            opening=[
                fact("lead_performer", "grounded", "standing"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "exactly 'grounded' or 'airborne'"):
            _materialize_h3_physical_state_proposals([invalid_grounded])

        extra_key = proposal_row(
            transitions=[
                {
                    **transition("lead_performer", "posture", "upright", "Ari rises"),
                    "from_value": "crouched",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "only subject_id, predicate"):
            _materialize_h3_physical_state_proposals([extra_key])

        oversized = proposal_row(
            opening=[
                fact("lead_performer", "posture", "x" * 81),
            ]
        )
        with self.assertRaisesRegex(ValueError, "exceeds 80 characters"):
            _materialize_h3_physical_state_proposals([oversized])

    def test_legacy_full_ledger_passes_through_and_rows_are_deep_copied(self):
        legacy = proposal_row()
        legacy["physical_state"] = full_state(
            [fact("lead_performer", "posture", "upright")],
        )
        legacy_before = copy.deepcopy(legacy)
        proposal = proposal_row(
            opening=[
                fact("lead_performer", "location", "near the stage"),
            ]
        )
        original_proposal = copy.deepcopy(proposal)
        result = _materialize_h3_physical_state_proposals([legacy, proposal])
        self.assertEqual(result[0], legacy_before)
        self.assertEqual(proposal, original_proposal)
        self.assertIsNot(result[0], legacy)
        self.assertEqual(
            result[1]["physical_state"]["opening"],
            [
                fact("lead_performer", "posture", "upright"),
                fact("lead_performer", "location", "near the stage"),
            ],
        )

    def test_prefix_mode_marks_invalid_row_and_keeps_unprocessed_tail(self):
        valid = proposal_row()
        invalid = proposal_row(
            transitions=[
                transition(
                    "lead_performer",
                    "support",
                    "astride mount",
                    "Ari climbs aboard",
                )
            ]
        )
        later = proposal_row(opening=[fact("lead_performer", "location", "stage")])

        with self.assertRaisesRegex(ValueError, "no explicit opening/current fact"):
            _materialize_h3_physical_state_proposals([valid, invalid, later])

        result = _materialize_h3_physical_state_proposals(
            [valid, invalid, later],
            preserve_invalid_tail=True,
        )
        self.assertEqual(result[0]["physical_state"]["closing"], [])
        self.assertIn("_h3_physical_state_proposal_error", result[1])
        self.assertEqual(result[2], later)
        issue = MusicVideoPlanner._music_video_row_issue(
            1,
            result[1],
            [result[0]],
            scene_description="A performance on stage",
            include_image_fields=False,
        )
        self.assertIn("could not be materialized", issue)

        missing_state = complete_shot("[Shot 1] Ari looks across the stage.")
        missing_state.pop("physical_state")
        preserved = _materialize_h3_physical_state_proposals(
            [missing_state],
            preserve_invalid_tail=True,
        )
        missing_state_issue = MusicVideoPlanner._music_video_row_issue(
            0,
            preserved[0],
            [],
            scene_description="A music performance on stage",
            include_image_fields=False,
            require_source_physical_state=True,
        )
        self.assertIn("could not be materialized", missing_state_issue)

    def test_holding_pose_or_camera_frame_does_not_imply_contact_state(self):
        row = complete_shot(
            "The camera holds steady in a wide frame while the kitten holds "
            "its determined pose, mid-song."
        )
        row["action_beats"] = [
            "The kitten sings with its head held high.",
            "The cape flutters in the battlefield wind.",
        ]
        row["ending_beat"] = (
            "The kitten holds its determined pose as the dust settles."
        )
        row["physical_state"] = full_state([])
        source = (
            "A dragon flies through the air carrying the kitten over a "
            "battlefield while the kitten sings."
        )

        self.assertFalse(_source_establishes_physical_relation(source, row))
        self.assertFalse(_has_explicit_physical_relation(row["video_prompt"]))
        self.assertIsNone(
            MusicVideoPlanner._music_video_row_issue(
                0,
                row,
                [],
                scene_description=source,
                include_image_fields=False,
                require_source_physical_state=True,
            )
        )

    def test_concrete_holding_and_riding_still_require_contact_state(self):
        cases = (
            (
                "Ari carries a sword across the stage.",
                "Ari holds the sword steady in the spotlight.",
            ),
            (
                "Ari touches the drummer's hand during the performance.",
                "Ari holds the drummer's hand beside the microphone.",
            ),
            (
                "The camera operator touches a camera during rehearsal.",
                "The camera operator holds a camera beside the stage.",
            ),
            (
                "Ari rides on the dragon above the rooftops.",
                "Ari rides the dragon through a bank of clouds.",
            ),
        )
        for source, prompt in cases:
            with self.subTest(prompt=prompt):
                row = complete_shot(prompt)
                row["physical_state"] = full_state([])
                issue = MusicVideoPlanner._music_video_row_issue(
                    0,
                    row,
                    [],
                    scene_description=source,
                    include_image_fields=False,
                    require_source_physical_state=True,
                )
                self.assertIn("explicitly established physical relation", issue)

    def test_abstract_music_posture_and_framing_holds_do_not_count_as_contact(self):
        source = "Ari rides a surreal wave of color while singing a sustained note."
        row = complete_shot(
            "A fixed camera holds its wide framing as Ari holds a sustained "
            "note, a focused gaze, and a poised posture."
        )
        row["action_beats"] = [
            "Ari holds a long note over the abstract visualizer.",
            "Ari keeps an upright, composed pose.",
            "The camera holds a wide shot of the singer.",
            "The camera holds on the singer as the chorus rises.",
        ]
        row["ending_beat"] = "The camera holds steady on Ari's final note."

        self.assertFalse(_source_establishes_physical_relation(source, row))
        self.assertFalse(_has_explicit_physical_relation(row["video_prompt"]))
        self.assertFalse(_has_explicit_physical_relation(row["ending_beat"]))
        self.assertFalse(
            _has_explicit_physical_relation("The camera holds a wide shot.")
        )
        self.assertFalse(
            _has_explicit_physical_relation("The camera holds on the singer.")
        )

    def test_h3_production_omits_and_discards_misclassified_physical_ledgers(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        clips = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
        callbacks = self._repair_callbacks(planner, clips)
        breath = complete_shot(
            "[Shot 1] Ari takes a deep breath, shoulders rising before she sings."
        )
        breath["action_beats"] = ["Ari takes a deep breath before singing."]
        breath["physical_state"] = {
            "opening": [],
            "transitions": [transition(
                "lead_performer", "prop", "holding a deep breath",
                "Ari takes a deep breath",
            )],
            "closing": [],
        }
        breath["_h3_physical_state_proposal_error"] = (
            "prop transition has no explicit opening/current fact"
        )

        with patch.object(planner, "_call_llm_json", return_value=[copy.deepcopy(breath)]) as generate:
            planned = callbacks["call_batch"](1, 0, [clips[0]], None)
        request = generate.call_args.kwargs
        item_schema = request["json_schema"]["items"]
        self.assertNotIn("physical_state", item_schema["properties"])
        self.assertIn("continuity_strategy", item_schema["required"])
        self.assertNotIn("physical_state", request["system_prompt"])
        self.assertIn("support and contact", request["system_prompt"])
        self.assertIn("local shot direction", request["system_prompt"])
        self.assertEqual(planned[0]["video_prompt"], breath["video_prompt"])
        self.assertEqual(planned[0]["action_beats"], breath["action_beats"])
        self.assertNotIn("physical_state", planned[0])
        self.assertNotIn("_h3_physical_state_proposal_error", planned[0])

        # The same validator is used for fresh, resumed, and checkpoint rows.
        # These two old drafts represent the observed prop/breath and
        # contact/gaze misclassifications; their visible direction survives.
        drafts = [
            breath,
            complete_shot(
                "[Shot 1] Ari's gaze shifts off-camera right as she sings."
            ),
        ]
        drafts[1]["physical_state"] = full_state([
            fact("lead_performer", "contact", "looking off-camera right"),
        ])
        drafts[1]["_h3_physical_state_proposal_error"] = (
            "gaze was mislabeled as a contact change"
        )
        strict_draft = copy.deepcopy(breath)
        strict_issue = MusicVideoPlanner._music_video_row_issue(
            0,
            strict_draft,
            [],
            scene_description="Ari performs on stage.",
            include_image_fields=False,
        )
        self.assertIn(
            "compact physical-state proposal could not be materialized",
            strict_issue,
        )
        for index, row in enumerate(drafts):
            visible = (row["video_prompt"], copy.deepcopy(row["action_beats"]))
            issue = callbacks["row_validator"](index, row, [])
            self.assertIsNone(issue, issue)
            self.assertEqual(
                (row["video_prompt"], row["action_beats"]), visible,
            )
            self.assertNotIn("physical_state", row)
            self.assertNotIn("_h3_physical_state_proposal_error", row)

    def test_h3_production_accepts_local_pose_changes_and_unknown_starting_pose(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        clips = [
            {"start": index * 10, "end": (index + 1) * 10}
            for index in range(4)
        ]
        callbacks = self._repair_callbacks(planner, clips)
        examples = (
            ("head tilt", "A kitten's head tilts, then she stands centered, looking right."),
            ("gaze", "A singer's gaze shifts toward the lights, then she stands centered."),
            ("gesture", "A sailor raises one hand in greeting, then stands centered."),
        )
        for index, (kind, prompt) in enumerate(examples):
            with self.subTest(kind=kind):
                previous = complete_shot(f"[Shot 1] The performer {kind}s toward camera.")
                previous["physical_state"] = full_state([
                    fact("lead_performer", "posture", f"{kind} directed toward camera"),
                ])
                current = complete_shot(f"[Shot 1] {prompt}")
                current["physical_state"] = full_state([
                    fact("lead_performer", "posture", "standing centered"),
                ])
                issue = callbacks["row_validator"](
                    index, current, [previous],
                )
                self.assertIsNone(issue, issue)
                self.assertNotIn("physical_state", current)
                self.assertNotIn("physical_state", previous)

        # No first-frame posture is inferred when visible prose supplies the pose.
        no_start = complete_shot(
            "[Shot 1] Ari turns toward the camera and settles into a standing pose."
        )
        no_start.pop("physical_state", None)
        issue = callbacks["row_validator"](0, no_start, [])
        self.assertIsNone(issue, issue)
        self.assertNotIn("physical_state", no_start)

    def test_h3_conversion_does_not_append_ledger_text_or_save_ledger_metadata(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        baseline = complete_shot(
            "[Shot 1] Ari turns her gaze toward the lights and takes a deep breath."
        )
        baseline["action_beats"] = ["Ari turns her gaze toward the lights."]
        with_ledger = copy.deepcopy(baseline)
        with_ledger["physical_state"] = {
            "opening": [],
            "transitions": [{
                "subject_id": "lead_performer",
                "predicate": "contact",
                "value": "holding a lantern",
            }],
            "closing": [],
        }
        with_ledger["_h3_physical_state_proposal_error"] = "stale proposal failure"
        args = (
            [{"start": 0, "end": 10, "label": "verse"}],
            [],
            False,
            {},
            None,
            {},
        )
        without_state = planner._convert_to_shots(
            [baseline], *args, ignore_physical_state_ledger=True,
        )[0]
        with_state = planner._convert_to_shots(
            [with_ledger], *args, ignore_physical_state_ledger=True,
        )[0]
        self.assertEqual(with_state.video_prompt, without_state.video_prompt)
        self.assertEqual(with_state.action_beats, without_state.action_beats)
        self.assertNotIn("physical_state", with_state.metadata)

    def test_h3_source_action_boarding_order_and_cast_guards_remain_strict(self):
        source = (
            "The kitten climbs on the dragon, then the dragon breathes fire over enemies."
        )
        contract = build_music_narrative_contract(source)
        clips = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        planner._music_narrative_contract = contract
        planner._music_narrative_allocation = [
            {"event_id": event["event_id"], "clip_index": 0}
            for event in contract["events"]
        ]
        callbacks = self._repair_callbacks(planner, clips)
        cast = [
            {
                "character_id": "kitten", "speaker_name": "Kitten",
                "visual_description": "a small kitten", "performance_role": "non_vocal",
            },
            {
                "character_id": "dragon", "speaker_name": "Dragon",
                "visual_description": "a large dragon", "performance_role": "non_vocal",
            },
        ]

        def row(prompt):
            result = complete_shot(f"[Shot 1] {prompt}")
            result["subjects_on_screen"] = copy.deepcopy(cast)
            # Invalid or stale ledger data is removed before narrative checks.
            result["physical_state"] = {"invalid": "legacy ledger"}
            return result

        valid = row(
            "The kitten stands on the ground beside the dragon. The kitten climbs "
            "on the dragon, then rides it as the dragon breathes fire over enemies."
        )
        valid_issue = callbacks["row_validator"](0, valid, [])
        self.assertIsNone(valid_issue, valid_issue)

        rejected = (
            ("The kitten sings beside the dragon.", "source verb family"),
            (
                "A puppy climbs on the dragon. The dragon breathes fire over enemies.",
                "requested actor",
            ),
            (
                "The kitten is already perched on the dragon; then the kitten climbs "
                "on the dragon. The dragon breathes fire over enemies.",
                "opens with",
            ),
            (
                "The dragon breathes fire over enemies, then the kitten climbs on the dragon.",
                "out of source order",
            ),
        )
        for prompt, expected in rejected:
            with self.subTest(expected=expected):
                issue = callbacks["row_validator"](0, row(prompt), [])
                self.assertIn(expected, issue)

        cast_planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        cast_callbacks = self._repair_callbacks(cast_planner, clips)
        missing_identity = complete_shot("[Shot 1] Ari sings to camera.")
        missing_identity["subjects_on_screen"] = [{
            "visual_description": "Ari", "performance_role": "non_vocal",
        }]
        cast_issue = cast_callbacks["row_validator"](0, missing_identity, [])
        self.assertIn("stable, meaningful character_id or speaker_name", cast_issue)

    def test_unsuccessful_narrative_patch_falls_back_to_full_row_repair(self):
        for case in ("narrative", "creative", "general"):
            with self.subTest(case=case):
                planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
                clips = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
                row = self._invalid_materialized_row(source_event_ids=[])
                previous = self._previous_state_row()
                if case == "narrative":
                    planner._music_narrative_contract = {
                        "events": [{
                            "event_id": "M1",
                            "text": "raises a lantern",
                            "kind": "action",
                            "actor_aliases": ["Ari"],
                        }],
                    }
                    planner._music_narrative_allocation = [
                        {"event_id": "M1", "clip_index": 1},
                    ]
                    row["video_prompt"] = "Ari watches the lantern and sings."
                    row["action_beats"] = ["Ari watches the lantern and sings."]
                elif case == "general":
                    row.pop("_h3_physical_state_proposal_error", None)
                    row["physical_state"] = full_state([])
                    row["mood"] = ""

                callbacks = self._repair_callbacks(planner, clips)
                if case == "creative":
                    planner._last_music_creative_issue = (
                        "Keep the assigned beat intact."
                    )
                issue = callbacks["row_validator"](1, row, [previous])
                if case == "creative":
                    self.assertIsNone(issue, issue)
                else:
                    self.assertTrue(issue)

                with patch.object(planner, "_call_llm_json", return_value=[]) as generate:
                    callbacks["repair_batch"](1, 1, [clips[1]], previous)

                self.assertEqual(generate.call_count, 2 if case == "narrative" else 1)
                if case == "narrative":
                    focused_schema = generate.call_args_list[0].kwargs["json_schema"]
                    self.assertEqual(
                        set(focused_schema["items"]["properties"]),
                        {"scene_goal", "video_prompt", "action_beats", "ending_beat"},
                    )
                if case == "creative":
                    self.assertIn(
                        "Keep the assigned beat intact.",
                        generate.call_args.kwargs["user_prompt"],
                    )
                properties = generate.call_args.kwargs["json_schema"]["items"]["properties"]
                self.assertIn("video_prompt", properties)
                self.assertIn("action_beats", properties)
                self.assertNotEqual(set(properties), {"physical_state"})


if __name__ == "__main__":
    unittest.main()
