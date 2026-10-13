"""Bounded visual repair keeps music source actions authoritative."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.director.music_narrative import (
    build_music_narrative_contract,
    format_music_narrative_context,
    music_narrative_row_issue,
)
from services.director.music_narrative_repair import (
    compile_music_narrative_bridge_row,
    compile_music_narrative_source_row,
    repair_music_narrative_visuals,
)


def visual_patch(video_prompt: str = "The boat sails across the lake.") -> dict:
    return {
        "scene_goal": "Show the boat making its crossing.",
        "video_prompt": video_prompt,
        "action_beats": ["00:00.0-00:06.5: The boat sails across the lake."],
        "ending_beat": "The boat continues toward the far shore.",
    }


def source_row(*, source_ids: list[str], video_prompt: str) -> dict:
    return {
        "scene_goal": "An old rejected goal.",
        "scene_type": "narrative",
        "environment": "Harbor pier at sunrise.",
        "visual_style": "Cinematic coastal folk film.",
        "lighting": "Warm sunrise.",
        "mood": "Hopeful.",
        "ending_beat": "An old rejected ending.",
        "video_prompt": video_prompt,
        "action_beats": ["An old rejected action beat."],
        "source_event_ids": source_ids,
        "subjects_on_screen": [
            {
                "character_id": "ari",
                "speaker_name": "",
                "visual_description": "Ari, an adult sailor in a navy coat.",
                "position_or_relation": "already aboard the yellow canoe",
                "performance_role": "non_vocal",
            },
            {
                "character_id": "sailboat",
                "speaker_name": "",
                "visual_description": "A small wooden sailboat with a cream sail.",
                "position_or_relation": "beside the pier",
                "performance_role": "non_vocal",
            },
        ],
        "camera_plan": {
            "framing": "wide shot",
            "angle": "eye level",
            "movement": "slow lateral track",
            "movement_intensity": "low",
            "lens_feel": "natural",
        },
        "window_prompts": [],
        "long_form_outline": "GLOBAL_OUTLINE_SENTINEL_" * 100,
        "lyrics_context": "LYRICS_SENTINEL_" * 100,
        "source_event_ids_metadata": {"preserve": True},
    }


class MusicNarrativeVisualRepairTests(unittest.TestCase):
    def setUp(self):
        self.contract = build_music_narrative_contract(
            "The sailor climbs onto the boat. The boat sails across the lake. "
            "Then the sailor lands at the pier."
        )
        self.allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 1},
            {"event_id": "M3", "clip_index": 2},
        ]
        self.row = {
            "scene_goal": "The boat pauses while the sailor sings.",
            "video_prompt": "The sailor stands on the deck as the boat stays near shore.",
            "action_beats": ["00:00.0-00:06.5: the sailor watches the water"],
            "ending_beat": "The boat remains beside the pier.",
            "source_event_ids": ["M2"],
            "subjects_on_screen": [{
                "character_id": "sailor",
                "speaker_name": "",
                "visual_description": "A sailor in a weathered blue coat.",
                "position_or_relation": "at the helm",
                "performance_role": "non_vocal",
            }],
            "camera_plan": {
                "framing": "medium-wide",
                "angle": "eye level",
                "movement": "slow lateral track",
                "movement_intensity": "low",
                "lens_feel": "natural",
            },
            "long_form_outline": "OUTLINE_SENTINEL_" * 200,
            "lyrics_context": "LYRICS_SENTINEL_" * 200,
            "physical_state": {"private": "PHYSICAL_STATE_SENTINEL"},
            "_h3_physical_state_proposal_error": "PRIVATE_PROPOSAL_SENTINEL",
            "untouched_metadata": {"keep": True},
        }
        self.issue = music_narrative_row_issue(
            self.row,
            contract=self.contract,
            allocation=self.allocation,
            index=1,
        )
        self.assertIsNotNone(self.issue)

    def test_single_short_prompt_repairs_only_visual_fields_and_preserves_row(self):
        original = copy.deepcopy(self.row)
        expected_context = format_music_narrative_context(
            self.contract, self.allocation, 1, 2,
        )
        patch_value = visual_patch(
            "At 00:00, the boat leaves shore and sails across the lake toward the far bank."
        )
        planner = type("Planner", (), {})()
        planner._call_llm_json = Mock(return_value=[patch_value])

        result = repair_music_narrative_visuals(
            planner,
            index=1,
            row=self.row,
            issue=self.issue,
            contract=self.contract,
            allocation=self.allocation,
            duration_seconds=6.5,
        )

        self.assertIsNotNone(result)
        self.assertEqual(planner._call_llm_json.call_count, 1)
        call = planner._call_llm_json.call_args.kwargs
        self.assertEqual(call["max_tokens"], 2300)
        self.assertEqual(call["thinking_budget"], 0)
        self.assertEqual(call["json_schema"]["minItems"], 1)
        self.assertEqual(call["json_schema"]["maxItems"], 1)
        item_schema = call["json_schema"]["items"]
        self.assertEqual(set(item_schema["required"]), set(patch_value))
        self.assertFalse(item_schema["additionalProperties"])

        prompt = call["user_prompt"]
        self.assertIn(self.issue, prompt)
        self.assertIn(expected_context, prompt)
        self.assertIn(
            "copy each required named actor plus source action clause",
            prompt.casefold(),
        )
        self.assertIn("into video_prompt unchanged and in source order", prompt)
        self.assertIn("Develop natural movement around that exact action", prompt)
        self.assertIn("boat sails across the lake", prompt)
        self.assertIn('"selected_duration_seconds":6.5', prompt)
        self.assertIn("sailor", prompt)
        self.assertIn("medium-wide", prompt)
        self.assertNotIn("OUTLINE_SENTINEL", prompt)
        self.assertNotIn("LYRICS_SENTINEL", prompt)
        self.assertNotIn("PHYSICAL_STATE_SENTINEL", prompt)
        self.assertNotIn("PRIVATE_PROPOSAL_SENTINEL", prompt)
        self.assertIn("Correct conflicting action prose in place", prompt)
        self.assertIn("do not append a contradictory action", prompt.casefold())
        request = json.loads(prompt.split("CLIP REPAIR CONTEXT (JSON):\n", 1)[1])
        self.assertEqual(request["selected_duration_seconds"], 6.5)

        expected_row = copy.deepcopy(original)
        expected_row.update(patch_value)
        self.assertEqual(result, [expected_row])
        self.assertEqual(self.row, original)
        self.assertEqual(result[0]["source_event_ids"], ["M2"])
        self.assertEqual(result[0]["untouched_metadata"], {"keep": True})
        self.assertEqual(result[0]["physical_state"], original["physical_state"])
        self.assertEqual(
            result[0]["_h3_physical_state_proposal_error"],
            original["_h3_physical_state_proposal_error"],
        )

    def test_malformed_responses_return_none_without_retry(self):
        malformed = (
            None,
            [],
            [visual_patch(), visual_patch()],
            [{**visual_patch(), "physical_state": {"opening": []}}],
            [{**visual_patch(), "video_prompt": "  "}],
            [{**visual_patch(), "action_beats": ["ok", 4]}],
        )
        for response in malformed:
            with self.subTest(response=response):
                planner = type("Planner", (), {})()
                planner._call_llm_json = Mock(return_value=response)
                result = repair_music_narrative_visuals(
                    planner,
                    index=1,
                    row=self.row,
                    issue=self.issue,
                    contract=self.contract,
                    allocation=self.allocation,
                    duration_seconds=6.5,
                )
                self.assertIsNone(result)
                self.assertEqual(planner._call_llm_json.call_count, 1)

    def test_source_checker_rejects_missing_wrong_actor_boarding_and_order(self):
        cases = (
            (
                build_music_narrative_contract("The dragon breathes fire over enemies."),
                [{"event_id": "M1", "clip_index": 0}],
                {"source_event_ids": ["M1"], "video_prompt": "The dragon circles above enemies."},
                visual_patch("The dragon circles above enemies."),
                0,
            ),
            (
                build_music_narrative_contract("The dragon breathes fire over enemies."),
                [{"event_id": "M1", "clip_index": 0}],
                {"source_event_ids": ["M1"], "video_prompt": "The dragon turns toward the field."},
                visual_patch("The kitten breathes fire over enemies."),
                0,
            ),
            (
                build_music_narrative_contract("The kitten climbs on the dragon."),
                [{"event_id": "M1", "clip_index": 0}],
                {"source_event_ids": ["M1"], "video_prompt": "The kitten stands beside the dragon."},
                visual_patch(
                    "The kitten is already mounted on the dragon, then climbs on the dragon."
                ),
                0,
            ),
            (
                build_music_narrative_contract(
                    "A passenger boards a boat. Then the boat sails across the bay."
                ),
                [
                    {"event_id": "M1", "clip_index": 0},
                    {"event_id": "M2", "clip_index": 0},
                ],
                {"source_event_ids": ["M1", "M2"], "video_prompt": "The boat waits by the dock."},
                visual_patch(
                    "The boat sails across the bay before the passenger boards it."
                ),
                0,
            ),
        )
        for contract, allocation, row, patch_value, index in cases:
            with self.subTest(row=row, patch=patch_value):
                issue = music_narrative_row_issue(
                    row, contract=contract, allocation=allocation, index=index,
                )
                self.assertIsNotNone(issue)
                planner = type("Planner", (), {})()
                planner._call_llm_json = Mock(return_value=[patch_value])
                result = repair_music_narrative_visuals(
                    planner,
                    index=index,
                    row=row,
                    issue=issue,
                    contract=contract,
                    allocation=allocation,
                    duration_seconds=6.5,
                )
                self.assertIsNone(result)
                self.assertEqual(planner._call_llm_json.call_count, 1)

    def test_interrupted_error_propagates(self):
        planner = type("Planner", (), {})()
        planner._call_llm_json = Mock(side_effect=InterruptedError("cancelled"))

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            repair_music_narrative_visuals(
                planner,
                index=1,
                row=self.row,
                issue=self.issue,
                contract=self.contract,
                allocation=self.allocation,
                duration_seconds=6.5,
            )
        self.assertEqual(planner._call_llm_json.call_count, 1)


class MusicNarrativeSourceCompilerTests(unittest.TestCase):
    def test_rebuilds_wrong_actor_and_destination_from_exact_source_action(self):
        contract = build_music_narrative_contract(
            "Ari climbs onto a small wooden sailboat. The boat sails across the bay."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 1},
        ]
        row = source_row(
            source_ids=["M1"],
            video_prompt="[Shot 1] Ari is already aboard the yellow canoe.",
        )
        original = copy.deepcopy(row)
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        )
        self.assertIsNotNone(issue)

        result = compile_music_narrative_source_row(
            index=0,
            row=row,
            issue=issue,
            contract=contract,
            allocation=allocation,
        )

        self.assertIsNotNone(result)
        compiled = result[0]
        prompt = compiled["video_prompt"]
        self.assertLess(
            prompt.index("unboarded beside"),
            prompt.index("Ari climbs onto a small wooden sailboat"),
        )
        self.assertLess(
            prompt.index("Ari climbs onto a small wooden sailboat"),
            prompt.index("securely aboard"),
        )
        self.assertIn("slow lateral track", prompt)
        self.assertNotIn("yellow canoe", prompt)
        self.assertNotIn("GLOBAL_OUTLINE_SENTINEL", prompt)
        self.assertNotIn("LYRICS_SENTINEL", prompt)
        self.assertNotIn("M1", prompt)
        self.assertEqual(compiled["source_event_ids"], ["M1"])
        self.assertEqual(compiled["source_event_ids_metadata"], {"preserve": True})
        self.assertTrue(compiled["_music_source_compiled"])
        self.assertEqual(compiled["_music_source_recovery_reason"], issue)
        self.assertEqual(row, original)
        self.assertEqual(
            music_narrative_row_issue(
                compiled, contract=contract, allocation=allocation, index=0,
            ),
            None,
        )

    def test_emits_assigned_events_in_contract_order_and_preserves_ids(self):
        contract = build_music_narrative_contract(
            "Ari climbs onto a small wooden sailboat. The boat sails across the bay."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 0},
        ]
        row = source_row(
            source_ids=["M1", "M2"],
            video_prompt="[Shot 1] The boat sails across the bay before Ari boards a canoe.",
        )
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        )
        self.assertIsNotNone(issue)

        result = compile_music_narrative_source_row(
            index=0, row=row, issue=issue, contract=contract, allocation=allocation,
        )

        self.assertIsNotNone(result)
        prompt = result[0]["video_prompt"]
        self.assertLess(
            prompt.index("Ari climbs onto a small wooden sailboat"),
            prompt.index("The boat sails across the bay"),
        )
        self.assertEqual(result[0]["source_event_ids"], ["M1", "M2"])
        self.assertIsNone(
            music_narrative_row_issue(
                result[0], contract=contract, allocation=allocation, index=0,
            )
        )

    def test_carries_completed_boarding_only_from_source_contract(self):
        contract = build_music_narrative_contract(
            "Ari climbs onto a small wooden sailboat. The boat sails across the bay."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 1},
        ]
        row = source_row(
            source_ids=["M2"],
            video_prompt="[Shot 1] Ari waits on the pier while the boat circles the bay.",
        )
        original = copy.deepcopy(row)
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=1,
        )
        self.assertIsNotNone(issue)

        result = compile_music_narrative_source_row(
            index=1, row=row, issue=issue, contract=contract, allocation=allocation,
        )

        self.assertIsNotNone(result)
        compiled = result[0]
        prompt = compiled["video_prompt"]
        self.assertLess(
            prompt.index("Ari remains securely aboard a small wooden sailboat"),
            prompt.index("The boat sails across the bay"),
        )
        self.assertEqual(
            compiled["subjects_on_screen"][0]["position_or_relation"],
            "securely aboard a small wooden sailboat",
        )
        self.assertEqual(compiled["source_event_ids"], ["M2"])
        self.assertEqual(row, original)
        self.assertIsNone(
            music_narrative_row_issue(
                compiled, contract=contract, allocation=allocation, index=1,
            )
        )

    def test_uses_source_actor_not_wrong_actor_or_wrong_destination_draft(self):
        contract = build_music_narrative_contract(
            "The dragon breathes fire over a battlefield of enemies."
        )
        allocation = [{"event_id": "M1", "clip_index": 0}]
        row = source_row(
            source_ids=["M1"],
            video_prompt="[Shot 1] The kitten breathes fire toward the viewer.",
        )
        row["subjects_on_screen"] = [
            {
                "character_id": "kitten",
                "visual_description": "A small kitten with a pink bow.",
                "position_or_relation": "center frame",
                "performance_role": "non_vocal",
            },
            {
                "character_id": "dragon",
                "visual_description": "A huge dark-scaled dragon.",
                "position_or_relation": "opposite the kitten",
                "performance_role": "non_vocal",
            },
        ]
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        )
        self.assertIsNotNone(issue)

        result = compile_music_narrative_source_row(
            index=0, row=row, issue=issue, contract=contract, allocation=allocation,
        )

        self.assertIsNotNone(result)
        prompt = result[0]["video_prompt"]
        self.assertIn("The dragon breathes fire over a battlefield of enemies", prompt)
        self.assertNotIn("kitten breathes fire", prompt)
        self.assertNotIn("toward the viewer", prompt)
        self.assertIsNone(
            music_narrative_row_issue(
                result[0], contract=contract, allocation=allocation, index=0,
            )
        )

    def test_refuses_empty_events_missing_actor_or_unanchored_human(self):
        row = source_row(source_ids=[], video_prompt="[Shot 1] Existing visual.")
        empty = {"source": "Ari waves.", "events": []}
        self.assertIsNone(
            compile_music_narrative_source_row(
                index=0, row=row, issue="missing action", contract=empty, allocation=[],
            )
        )

        contract = build_music_narrative_contract("A stranger runs across the field.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        row["source_event_ids"] = ["M1"]
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=0,
        )
        self.assertIsNotNone(issue)
        self.assertIsNone(
            compile_music_narrative_source_row(
                index=0, row=row, issue=issue, contract=contract, allocation=allocation,
            )
        )

        contract["events"][0]["actor_aliases"] = []
        self.assertIsNone(
            compile_music_narrative_source_row(
                index=0, row=row, issue=issue, contract=contract, allocation=allocation,
            )
        )

    def test_completed_source_mount_can_act_without_inventing_a_cast_id(self):
        contract = build_music_narrative_contract(
            "The kitten climbs on the dragon. The dragon swoops down and breathes fire over enemies."
        )
        allocation = [{"event_id": "M1", "clip_index": 0},
                      {"event_id": "M2", "clip_index": 1},
                      {"event_id": "M3", "clip_index": 1}]
        row = source_row(source_ids=["M2", "M3"], video_prompt="[Shot 1] The kitten watches the valley.")
        row["subjects_on_screen"] = [{
            "character_id": "kitten", "speaker_name": "",
            "visual_description": "A small kitten with a pink bow.",
            "position_or_relation": "riding the dragon",
            "performance_role": "non_vocal",
        }]
        issue = music_narrative_row_issue(row, contract=contract, allocation=allocation, index=1)
        self.assertIsNotNone(issue)
        result = compile_music_narrative_source_row(
            index=1, row=row, issue=issue, contract=contract, allocation=allocation,
        )
        self.assertIsNotNone(result)
        self.assertIn("The dragon swoops down.", result[0]["video_prompt"])
        self.assertIn("The dragon breathes fire over enemies.", result[0]["video_prompt"])
        self.assertIn("securely aboard the dragon", result[0]["video_prompt"])
        self.assertEqual([s["character_id"] for s in result[0]["subjects_on_screen"]], ["kitten"])
        # A pending mount is not authority to introduce its absent actor early.
        future = [{"event_id": "M1", "clip_index": 2},
                  {"event_id": "M2", "clip_index": 1},
                  {"event_id": "M3", "clip_index": 1}]
        self.assertIsNone(compile_music_narrative_source_row(
            index=1, row=row, issue=issue, contract=contract, allocation=future,
        ))

    def test_refuses_structurally_incomplete_row(self):
        contract = build_music_narrative_contract("Ari climbs onto a small wooden sailboat.")
        allocation = [{"event_id": "M1", "clip_index": 0}]
        row = source_row(source_ids=["M1"], video_prompt="[Shot 1] Ari stands by the pier.")
        row["camera_plan"] = {}
        self.assertIsNone(
            compile_music_narrative_source_row(
                index=0,
                row=row,
                issue="failed source action",
                contract=contract,
                allocation=allocation,
            )
        )


class MusicNarrativeBridgeCompilerTests(unittest.TestCase):
    def test_builds_sailor_boat_bridge_from_previous_state_and_recompiles_native_fields(self):
        contract = build_music_narrative_contract(
            "Ari climbs onto a small wooden sailboat. The boat sails across the bay."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 2},
        ]
        previous_source = source_row(
            source_ids=["M1"],
            video_prompt="[Shot 1] Ari is already aboard the yellow canoe.",
        )
        previous = compile_music_narrative_source_row(
            index=0,
            row=previous_source,
            issue="repair boarding",
            contract=contract,
            allocation=allocation,
        )[0]
        row = source_row(
            source_ids=[],
            video_prompt=(
                "[Shot 1] The boat sails across the bay as Ari sings; "
                "the camera follows over open water."
            ),
        )
        row["environment"] = "Open water while the boat sails across the bay."
        row["subjects_on_screen"] = [
            {
                "character_id": "ari-current",
                "speaker_name": "Ari",
                "visual_description": "Ari, an adult sailor in a navy coat.",
                "position_or_relation": "already sailing across the bay",
                "performance_role": "lead_vocalist",
            },
            {
                "character_id": "sailboat-current",
                "speaker_name": "Boat",
                "visual_description": "A small wooden sailboat with a cream sail.",
                "position_or_relation": "underway across the bay",
                "performance_role": "non_vocal",
            },
        ]
        row["camera_plan"] = {
            "framing": "medium-wide",
            "angle": "eye level",
            "movement": "slow lateral track",
            "movement_intensity": "low",
            "lens_feel": "natural",
        }
        row["native_ir"] = "detailed_description: The boat sails across the bay."
        row["video_ir"] = "The boat sails across the bay."
        row["prompt_ir"] = "The boat sails across the bay."
        row["_director_native_ir"] = "The boat sails across the bay."
        row["detailed_description"] = "The boat sails across the bay."
        row["subject_definitions"] = ["The boat is sailing across the bay."]
        row["soundtrack"] = {"path": "song.wav", "sha256": "stable-audio-id"}
        original = copy.deepcopy(row)
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=1,
        )
        self.assertIsNotNone(issue)

        result = compile_music_narrative_bridge_row(
            index=1,
            row=row,
            issue=issue,
            contract=contract,
            allocation=allocation,
            previous=previous,
        )

        self.assertIsNotNone(result)
        compiled = result[0]
        self.assertEqual(compiled["source_event_ids"], [])
        self.assertEqual(
            [subject["character_id"] for subject in compiled["subjects_on_screen"]],
            ["ari-current", "sailboat-current"],
        )
        self.assertEqual(
            [subject["performance_role"] for subject in compiled["subjects_on_screen"]],
            ["lead_vocalist", "non_vocal"],
        )
        self.assertEqual(
            compiled["subjects_on_screen"][0]["visual_description"],
            row["subjects_on_screen"][0]["visual_description"],
        )
        self.assertIn("securely aboard a small wooden sailboat", compiled["video_prompt"])
        self.assertIn("Harbor pier at sunrise", compiled["environment"])
        self.assertIn("slow lateral track", compiled["video_prompt"])
        self.assertNotIn("sails across the bay", compiled["environment"])
        self.assertNotIn("sails across the bay", compiled["video_prompt"])
        self.assertNotIn("sailing across the bay", compiled["subjects_on_screen"][0]["position_or_relation"])
        self.assertEqual(compiled["soundtrack"], original["soundtrack"])
        self.assertTrue(compiled["_music_bridge_compiled"])
        self.assertTrue(compiled["_music_source_compiled"])
        for field in ("native_ir", "video_ir", "prompt_ir", "_director_native_ir", "detailed_description"):
            self.assertEqual(compiled[field], compiled["video_prompt"])
        self.assertEqual(compiled["window_prompts"], [])
        self.assertEqual(compiled["subject_definitions"], [])
        self.assertEqual(row, original)
        self.assertIsNone(
            music_narrative_row_issue(
                compiled, contract=contract, allocation=allocation, index=1,
            )
        )

    def test_preserves_partial_kitten_contact_without_replaying_boarding_or_future_flight(self):
        contract = build_music_narrative_contract(
            "The kitten climbs on the dragon. The dragon flies through the air."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 1},
            {"event_id": "M2", "clip_index": 3},
        ]
        previous = source_row(
            source_ids=["M1"],
            video_prompt="[Shot 1] The kitten climbs on the dragon.",
        )
        previous["environment"] = "A grassy plateau beneath a bright sky."
        previous["subjects_on_screen"] = [
            {
                "character_id": "kitten-before",
                "speaker_name": "Kitten",
                "visual_description": "A cute warrior kitten with a pink bow.",
                "position_or_relation": "halfway up the dragon's foreleg",
                "performance_role": "lead_vocalist",
            },
            {
                "character_id": "dragon-before",
                "speaker_name": "Dragon",
                "visual_description": "A massive dark-scaled dragon.",
                "position_or_relation": "beside the kitten on the plateau",
                "performance_role": "non_vocal",
            },
        ]
        previous["ending_beat"] = (
            "The kitten is halfway up the dragon's foreleg after climbing aboard. "
            "The dragon remains beside the kitten."
        )
        row = source_row(
            source_ids=[],
            video_prompt=(
                "[Shot 1] The dragon flies over clouds while the kitten rides high above the battlefield."
            ),
        )
        row["environment"] = "Open sky as the dragon flies over clouds."
        row["subjects_on_screen"] = [
            {
                "character_id": "hero-cat-current",
                "speaker_name": "Kitten",
                "visual_description": "A cute warrior kitten with a pink bow.",
                "position_or_relation": "riding high above the battlefield",
                "performance_role": "lead_vocalist",
            },
            {
                "character_id": "dragon-current",
                "speaker_name": "Dragon",
                "visual_description": "A massive dark-scaled dragon.",
                "position_or_relation": "flying through clouds",
                "performance_role": "non_vocal",
            },
        ]
        original = copy.deepcopy(row)
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=2,
        )
        self.assertIsNotNone(issue)
        self.assertIsNone(
            music_narrative_row_issue(
                previous, contract=contract, allocation=allocation, index=1,
            )
        )

        result = compile_music_narrative_bridge_row(
            index=2,
            row=row,
            issue=issue,
            contract=contract,
            allocation=allocation,
            previous=previous,
        )

        self.assertIsNotNone(result)
        compiled = result[0]
        self.assertIn("halfway up the dragon's foreleg", compiled["video_prompt"])
        self.assertNotIn("climbs on the dragon", compiled["video_prompt"])
        self.assertNotIn("climbing aboard", compiled["video_prompt"])
        self.assertNotIn("flies over clouds", compiled["video_prompt"])
        self.assertIn("grassy plateau", compiled["environment"])
        self.assertEqual(compiled["source_event_ids"], [])
        self.assertEqual(
            [subject["character_id"] for subject in compiled["subjects_on_screen"]],
            ["hero-cat-current", "dragon-current"],
        )
        self.assertEqual(row, original)
        self.assertIsNone(
            music_narrative_row_issue(
                compiled, contract=contract, allocation=allocation, index=2,
            )
        )

    def test_refuses_bridge_without_accepted_previous_row_or_matching_cast(self):
        contract = build_music_narrative_contract(
            "The sailor climbs onto a boat. The boat sails across the lake."
        )
        allocation = [
            {"event_id": "M1", "clip_index": 0},
            {"event_id": "M2", "clip_index": 2},
        ]
        previous = source_row(
            source_ids=["M1"],
            video_prompt="[Shot 1] The sailor climbs onto a boat.",
        )
        row = source_row(
            source_ids=[],
            video_prompt="[Shot 1] The boat sails across the lake.",
        )
        issue = music_narrative_row_issue(
            row, contract=contract, allocation=allocation, index=1,
        )
        self.assertIsNotNone(issue)
        self.assertIsNone(
            compile_music_narrative_bridge_row(
                index=1,
                row=row,
                issue=issue,
                contract=contract,
                allocation=allocation,
                previous=None,
            )
        )

        previous["source_event_ids"] = []
        previous_issue = music_narrative_row_issue(
            previous, contract=contract, allocation=allocation, index=0,
        )
        self.assertIsNotNone(previous_issue)
        self.assertIsNone(
            compile_music_narrative_bridge_row(
                index=1,
                row=row,
                issue=issue,
                contract=contract,
                allocation=allocation,
                previous=previous,
            )
        )

        accepted_previous = source_row(
            source_ids=["M1"],
            video_prompt="[Shot 1] The sailor climbs onto a boat.",
        )
        row["subjects_on_screen"] = [{
            "character_id": "new-person",
            "speaker_name": "New Person",
            "visual_description": "A stranger in a red coat.",
            "position_or_relation": "",
            "performance_role": "non_vocal",
        }]
        self.assertIsNone(
            compile_music_narrative_bridge_row(
                index=1,
                row=row,
                issue=issue,
                contract=contract,
                allocation=allocation,
                previous=accepted_previous,
            )
        )


if __name__ == "__main__":
    unittest.main()
