"""Compact H3 music-state proposals materialize as strict physical ledgers."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.physical_continuity import PHYSICAL_STATE_JSON_SCHEMA
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

    def test_invalid_compact_second_row_repairs_after_checkpointing_first(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        planner._uses_generated_shot_images = False
        checkpoints = []
        planner._planning_checkpoint_callback = checkpoints.append
        clips = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
        initial_rows = [
            complete_shot(
                "[Shot 1] Ari turns toward the rooftop crowd.",
                ending="Ari turns toward the crowd",
            ),
            complete_shot(
                "[Shot 1] Ari walks toward the microphone.",
                transitions=[
                    transition(
                        "lead_performer",
                        "support",
                        "standing at microphone",
                        "Ari walks to the microphone",
                    )
                ],
                ending="Ari reaches the microphone",
            ),
        ]
        repaired_state = {
            "physical_state": {
                "opening": [
                    fact("lead_performer", "support", "standing on the rooftop stage")
                ],
                "transitions": [
                    transition(
                        "lead_performer",
                        "support",
                        "standing at microphone",
                        "Ari walks toward the microphone",
                    )
                ],
            }
        }

        with patch.object(
            planner,
            "_call_llm_json",
            side_effect=[initial_rows, [repaired_state]],
        ) as generate:
            result = planner._plan_with_llm(
                clips=clips,
                clip_contexts=["Global clip 1", "Global clip 2"],
                scene_description="A performance on a rooftop stage",
                bpm=120,
                has_reference=False,
                reference_image_path=None,
                char_profiles=[],
                performer_map={},
                video_model="minimax_h3_ref2va_fused_turbo",
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result[1]["video_prompt"], initial_rows[1]["video_prompt"])
        self.assertEqual(
            set(generate.call_args_list[1].kwargs["json_schema"]["items"]["properties"]),
            {"physical_state"},
        )
        self.assertTrue(
            any(
                batch.get("count") == 1
                and batch.get("expected_count") == 2
                and len(batch.get("rows", [])) == 1
                for checkpoint in checkpoints
                for batch in checkpoint.get("music_video_batches", {}).values()
            )
        )

    def test_batch_context_carries_previous_state_into_compact_h3_call(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        planner._uses_generated_shot_images = False
        clips = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
        prior_row = {
            "scene_goal": "Ari finishes the first shot",
            "environment": "A stage",
            "ending_beat": "Ari holds the sword",
            "subjects_on_screen": [{"visual_description": "Ari"}],
            "physical_state": full_state(
                [
                    fact("lead_performer", "contact", "holding sword"),
                ]
            ),
        }
        candidate = proposal_row(
            transitions=[
                transition(
                    "lead_performer",
                    "contact",
                    "sheathed sword",
                    "Ari sheathes the sword",
                )
            ]
        )
        with patch.object(
            planner, "_run_checkpointed_json_batches", return_value=[]
        ) as batches:
            planner._plan_with_llm(
                clips=clips,
                clip_contexts=["clip 1", "clip 2"],
                scene_description="A performer on stage",
                bpm=120,
                has_reference=False,
                reference_image_path=None,
                char_profiles=[],
                performer_map={},
                video_model="minimax_h3_ref2va_fused_turbo",
            )
        config = batches.call_args.kwargs
        with patch.object(
            planner, "_call_llm_json", return_value=[candidate]
        ) as generate:
            result = config["call_batch"](1, 1, [clips[1]], prior_row)
        state = result[0]["physical_state"]
        self.assertEqual(
            state["opening"],
            [fact("lead_performer", "contact", "holding sword")],
        )
        self.assertEqual(state["transitions"][0]["from_value"], "holding sword")
        self.assertNotIn("_music_previous_physical_state", generate.call_args.kwargs)
        schema_state = generate.call_args.kwargs["json_schema"]["items"]["properties"][
            "physical_state"
        ]
        self.assertNotIn("closing", schema_state["properties"])

    def test_metadata_only_repair_preserves_shot_and_includes_validation_context(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        clips = [
            {"start": 0, "end": 10},
            {"start": 10, "end": 20, "source_event_ids": ["M1"]},
        ]
        callbacks = self._repair_callbacks(planner, clips)
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
        original = self._invalid_materialized_row(source_event_ids=["M1"])
        issue = callbacks["row_validator"](1, original, [self._previous_state_row()])
        self.assertIn("compact physical-state proposal could not be materialized", issue)
        previous = self._previous_state_row()
        metadata = {
            "physical_state": {
                "opening": [
                    fact("lead_performer", "contact", "holding the lantern at waist")
                ],
                "transitions": [
                    transition(
                        "lead_performer",
                        "contact",
                        "holding the lantern overhead",
                        "Ari raises the lantern",
                    )
                ],
            }
        }

        with patch.object(planner, "_call_llm_json", return_value=[metadata]) as generate:
            result = callbacks["repair_batch"](1, 1, [clips[1]], previous)

        self.assertEqual(len(result), 1)
        expected = copy.deepcopy(original)
        expected.pop("_h3_physical_state_proposal_error", None)
        expected["physical_state"] = result[0]["physical_state"]
        self.assertEqual(result[0], expected)
        self.assertEqual(result[0]["source_event_ids"], ["M1"])
        self.assertEqual(result[0]["video_prompt"], original["video_prompt"])
        self.assertEqual(result[0]["action_beats"], original["action_beats"])
        self.assertIn(
            fact("lead_performer", "posture", "upright"),
            result[0]["physical_state"]["opening"],
        )

        request = generate.call_args.kwargs
        self.assertEqual(request["thinking_budget"], 0)
        self.assertLessEqual(request["max_tokens"], 2048)
        schema = request["json_schema"]
        self.assertEqual((schema["minItems"], schema["maxItems"]), (1, 1))
        self.assertEqual(
            set(schema["items"]["properties"]), {"physical_state"},
        )
        prompt = request["user_prompt"]
        for evidence in (
            "compact physical-state proposal could not be materialized",
            "target_clip_index_zero_based",
            "lead_performer",
            "preceding_authoritative_closing_facts",
            "known_current_physical_keys",
            "transition_keys_missing_prior_or_opening_fact",
            "contact",
            "raises a lantern",
            "exactly 'grounded' or 'airborne'",
            "cloth fluttering in persistent wind",
            "leaving the original visible prose intact",
            original["video_prompt"],
            json.dumps(original["physical_state"], ensure_ascii=False),
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, prompt)

    def test_incomplete_row_uses_full_repair_instead_of_state_only_repair(self):
        cases = ("missing physical_state", "missing video_prompt")
        for case in cases:
            with self.subTest(case=case):
                planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
                clips = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
                callbacks = self._repair_callbacks(planner, clips)
                if case == "missing physical_state":
                    row = complete_shot("Ari pauses on the rooftop stage.")
                    row.pop("physical_state")
                    row = _materialize_h3_physical_state_proposals(
                        [row], preserve_invalid_tail=True,
                    )[0]
                else:
                    row = self._invalid_materialized_row()
                    row.pop("video_prompt")

                previous = self._previous_state_row()
                issue = callbacks["row_validator"](1, row, [previous])
                self.assertIn("compact physical-state proposal could not be materialized", issue)

                with patch.object(planner, "_call_llm_json", return_value=[]) as generate:
                    callbacks["repair_batch"](1, 1, [clips[1]], previous)

                self.assertEqual(generate.call_count, 1)
                properties = generate.call_args.kwargs["json_schema"]["items"]["properties"]
                self.assertIn("physical_state", properties)
                self.assertIn("video_prompt", properties)
                self.assertNotEqual(set(properties), {"physical_state"})

    def test_malformed_metadata_repair_returns_empty_for_bounded_retry(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        clips = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
        callbacks = self._repair_callbacks(planner, clips)
        invalid = self._invalid_materialized_row()
        issue = callbacks["row_validator"](1, invalid, [self._previous_state_row()])
        self.assertIn("could not be materialized", issue)

        with patch.object(planner, "_call_llm_json", return_value=[]) as generate:
            result = callbacks["repair_batch"](
                1, 1, [clips[1]], self._previous_state_row(),
            )

        self.assertEqual(result, [])
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(callbacks["repair_attempts"], 2)
        self.assertEqual(
            set(generate.call_args.kwargs["json_schema"]["items"]["properties"]),
            {"physical_state"},
        )

    def test_failed_valid_metadata_is_tagged_and_second_attempt_gets_new_error(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
        clips = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
        callbacks = self._repair_callbacks(planner, clips)
        previous = self._previous_state_row()
        invalid = self._invalid_materialized_row()
        first_issue = callbacks["row_validator"](1, invalid, [previous])
        self.assertIn("no explicit opening/current fact", first_issue)

        still_invalid = {
            "physical_state": {
                "opening": [
                    fact("lead_performer", "grounded", "beside fallen sword"),
                ],
                "transitions": [
                    transition(
                        "lead_performer",
                        "contact",
                        "holding the lantern overhead",
                        "Ari raises the lantern",
                    ),
                ],
            },
        }
        repaired = {
            "physical_state": {
                "opening": [
                    fact("lead_performer", "grounded", "grounded"),
                    fact("lead_performer", "contact", "holding the lantern at waist"),
                ],
                "transitions": [
                    transition(
                        "lead_performer",
                        "contact",
                        "holding the lantern overhead",
                        "Ari raises the lantern",
                    ),
                ],
            },
        }
        with patch.object(
            planner,
            "_call_llm_json",
            side_effect=[[still_invalid], [repaired]],
        ) as generate:
            first_result = callbacks["repair_batch"](1, 1, [clips[1]], previous)
            self.assertIn("_h3_physical_state_proposal_error", first_result[0])
            grounded_issue = callbacks["row_validator"](1, first_result[0], [previous])
            self.assertIn("exactly 'grounded' or 'airborne'", grounded_issue)

            second_result = callbacks["repair_batch"](1, 1, [clips[1]], previous)

        self.assertEqual(generate.call_count, 2)
        second_prompt = generate.call_args_list[1].kwargs["user_prompt"]
        self.assertIn("transition_keys_missing_prior_or_opening_fact", second_prompt)
        self.assertIn('"predicate": "contact"', second_prompt)
        self.assertIn("exactly 'grounded' or 'airborne'", second_prompt)
        self.assertNotIn("_h3_physical_state_proposal_error", second_result[0])
        state = second_result[0]["physical_state"]
        self.assertEqual(
            next(item["value"] for item in state["opening"] if item["predicate"] == "grounded"),
            "grounded",
        )
        self.assertIsNone(callbacks["row_validator"](1, second_result[0], [previous]))

    def test_narrative_creative_and_general_errors_keep_full_row_repair(self):
        for case in ("narrative", "creative", "general"):
            with self.subTest(case=case):
                planner = MusicVideoPlanner(llm_generate=lambda **kwargs: "[]")
                clips = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
                callbacks = self._repair_callbacks(planner, clips)
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
                elif case == "general":
                    row.pop("_h3_physical_state_proposal_error", None)
                    row["physical_state"] = full_state([])
                    row["mood"] = ""

                issue = callbacks["row_validator"](1, row, [previous])
                self.assertTrue(issue)
                if case == "creative":
                    planner._last_music_creative_issue = "Keep the assigned beat intact."

                with patch.object(planner, "_call_llm_json", return_value=[]) as generate:
                    callbacks["repair_batch"](1, 1, [clips[1]], previous)

                self.assertEqual(generate.call_count, 1)
                properties = generate.call_args.kwargs["json_schema"]["items"]["properties"]
                self.assertIn("video_prompt", properties)
                self.assertIn("action_beats", properties)
                self.assertNotEqual(set(properties), {"physical_state"})


if __name__ == "__main__":
    unittest.main()
