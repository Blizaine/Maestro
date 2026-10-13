"""Required music-video actions survive planning, checkpoints and compilation."""
import ast
import copy
import sys
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.director.music_narrative import build_music_narrative_contract
from services.director.planners.music_video import MusicVideoPlanner, _music_shot_schema


class MusicNarrativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.contract = build_music_narrative_contract(
            "The sailor climbs onto the boat. The boat sails across the lake."
        )
        self.clips = [{"start": i * 10, "end": (i + 1) * 10} for i in range(23)]
        self.allocation = [
            {"event_id": event["event_id"], "clip_index": index + 1}
            for index, event in enumerate(self.contract["events"])
        ]
        self.planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        self.planner._configure_planning_runtime(
            {}, kind="music_video", fingerprint_payload={"test": 1},
        )
        self.planner._music_narrative_contract = self.contract
        self.planner._music_narrative_allocation = self.allocation
        self.planner._uses_generated_shot_images = False

    def _batch_callbacks(self):
        with patch.object(
            self.planner, "_run_checkpointed_json_batches", return_value=[],
        ) as batches:
            self.planner._plan_with_llm(
                clips=self.clips,
                clip_contexts=[f"Clip {i + 1}" for i in range(len(self.clips))],
                scene_description=self.contract["source"], bpm=120,
                has_reference=False, reference_image_path=None,
                char_profiles=[], performer_map={},
                video_model="minimax_h3_ref2va_fused_turbo",
            )
        return batches.call_args.kwargs

    def _row_validator(self):
        return self._batch_callbacks()["row_validator"]

    def test_allocation_is_bounded_and_durably_reused(self):
        self.assertTrue(self.contract["events"])
        published = []
        self.planner._planning_checkpoint_callback = published.append
        with patch.object(self.planner, "_call_llm_json", return_value=self.allocation) as call:
            result = self.planner._allocate_narrative_actions(
                self.contract, self.clips, self.contract["source"],
            )
            cached = self.planner._allocate_narrative_actions(
                self.contract, self.clips, self.contract["source"],
            )
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result, cached)
        self.assertEqual(published[-1]["music_narrative_allocation"], self.allocation)
        self.assertEqual(call.call_args.kwargs["thinking_budget"], 0)
        result[0]["clip_index"] = 22
        self.assertEqual(cached, self.allocation)

    def test_invalid_allocation_fails_after_two_focused_attempts(self):
        with patch.object(self.planner, "_call_llm_json", return_value=[]) as call:
            with self.assertRaisesRegex(RuntimeError, "story-action allocation failed"):
                self.planner._allocate_narrative_actions(self.contract, self.clips, "Boat story")
        self.assertEqual(call.call_count, 2)
        self.assertIn("Repair this allocation error", call.call_args.kwargs["user_prompt"])

    def test_performance_without_milestones_needs_no_allocation_call(self):
        contract = build_music_narrative_contract("A singer performs on a stage. Dramatic lights.")
        with patch.object(self.planner, "_call_llm_json") as call:
            self.assertEqual(self.planner._allocate_narrative_actions(contract, self.clips, contract["source"]), [])
        call.assert_not_called()

    def test_required_event_ids_are_grammar_constrained(self):
        ids = [event["event_id"] for event in self.contract["events"]]
        schema = _music_shot_schema(3, include_image_fields=False, narrative_event_ids=ids)
        self.assertIn("source_event_ids", schema["items"]["required"])
        self.assertEqual(schema["items"]["properties"]["source_event_ids"]["items"]["enum"], ids)

    def test_batches_receive_global_assignments_and_failure_feedback(self):
        with patch.object(self.planner, "_run_checkpointed_json_batches", return_value=[]) as batches:
            self.planner._plan_with_llm(
                clips=self.clips, clip_contexts=[f"Clip {i + 1}" for i in range(23)],
                scene_description=self.contract["source"], bpm=120, has_reference=False,
                reference_image_path=None, char_profiles=[], performer_map={},
                video_model="minimax_h3_ref2va_fused_turbo",
            )
        config = batches.call_args.kwargs
        self.assertLessEqual(config["batch_size"], 3)
        bad = {"video_prompt": "The sailor is already riding the boat.", "source_event_ids": []}
        self.assertIsNotNone(config["row_validator"](0, bad, []))
        with patch.object(self.planner, "_call_llm_json", return_value=[]) as call:
            config["repair_batch"](1, 0, self.clips[:1], None)
        request = call.call_args.kwargs["user_prompt"]
        self.assertIn("Required story-action repair", request)
        for event in self.contract["events"]:
            self.assertIn(event["event_id"], request)
        self.assertIn("FINAL REQUIRED STORY ACTIONS", request)
        with patch.object(self.planner, "_call_llm_json", return_value=[]) as call:
            config["call_batch"](3, 9, self.clips[9:12], None)
        request = call.call_args.kwargs["user_prompt"]
        final = request.split("FINAL REQUIRED STORY ACTIONS", 1)[1]
        self.assertIn("clip 10: source_event_ids must be exactly []", final)
        self.assertNotIn("clip 1: source_event_ids", final)

    def test_writer_copied_completed_event_ids_are_replaced_without_changing_prose(self):
        row_validator = self._row_validator()
        prompt = "The sailor watches the fire fade over the distant battlefield."
        row = {"video_prompt": prompt, "source_event_ids": ["M2"]}

        with patch.object(
            self.planner, "_music_video_row_issue", return_value=None,
        ) as row_check:
            issue = row_validator(10, row, [])

        self.assertIsNone(issue)
        self.assertEqual(row["source_event_ids"], [])
        self.assertEqual(row["video_prompt"], prompt)
        row_check.assert_called_once()

    def test_materialized_event_ids_follow_contract_order(self):
        event_ids = [event["event_id"] for event in self.contract["events"]]
        self.planner._music_narrative_allocation = [
            {"event_id": event_id, "clip_index": 10}
            for event_id in event_ids
        ]
        prompt = "The sailor climbs onto the boat. Then the boat sails across the lake."
        row = {"video_prompt": prompt, "source_event_ids": list(reversed(event_ids))}

        with patch.object(
            self.planner, "_music_video_row_issue", return_value=None,
        ):
            issue = self._row_validator()(10, row, [])

        self.assertIsNone(issue)
        self.assertEqual(row["source_event_ids"], event_ids)
        self.assertEqual(row["video_prompt"], prompt)

    def test_narrative_action_checks_still_reject_future_and_missing_actions(self):
        row_validator = self._row_validator()
        future = {
            "video_prompt": "The sailor climbs onto the boat.",
            "source_event_ids": ["M2"],
        }
        future_issue = row_validator(0, future, [])
        self.assertIn("performs future music narrative event M1", future_issue or "")
        self.assertEqual(future["source_event_ids"], [])

        missing = {
            "video_prompt": "The sailor waves from the shore.",
            "source_event_ids": [],
        }
        missing_issue = row_validator(1, missing, [])
        self.assertIn("must visibly perform assigned source event M1", missing_issue or "")
        self.assertEqual(missing["source_event_ids"], ["M1"])

    def test_narrative_repair_drops_rejected_outline_hint_from_full_writer_context(self):
        arc = [{
            "clip_index": i, "visual_beat": f"Distinct performance detail {i}",
            "purpose": "Musical performance", "environment": "Harbor pier",
            "camera": "Slow track", "source_event_ids": [],
            "intentional_reprise": False, "reprise_reason": "",
        } for i in range(len(self.clips))]
        arc[0]["visual_beat"] = "REJECTED_OUTLINE_HINT: the performer launches into the next act."
        arc[5]["visual_beat"] = "PRESERVED_OUTLINE_HINT: an intimate performance insert."
        self.planner._music_arc = copy.deepcopy(arc)
        callbacks = self._batch_callbacks()
        with patch.object(self.planner, "_call_llm_json", return_value=[]) as writer:
            callbacks["call_batch"](1, 0, self.clips[:1], None)
        self.assertIn("REJECTED_OUTLINE_HINT", writer.call_args.kwargs["user_prompt"])
        issue = callbacks["row_validator"](0, {
            "video_prompt": "The sailor climbs onto the boat.", "source_event_ids": [],
        }, [])
        self.assertIn("future music narrative", issue)
        with patch(
            "services.director.music_narrative_repair.repair_music_narrative_visuals", return_value=None,
        ), patch.object(self.planner, "_call_llm_json", return_value=[]) as writer:
            callbacks["repair_batch"](1, 0, self.clips[:1], None)
        repaired_context = writer.call_args.kwargs["user_prompt"]
        self.assertIn("global clip numbers, starting at 1", repaired_context)
        self.assertNotIn("ASSIGNED GROUPS FOR THIS BATCH [zero-based", repaired_context)
        self.assertNotIn("REJECTED_OUTLINE_HINT", repaired_context)
        self.assertIn("PRESERVED_OUTLINE_HINT", repaired_context)
        self.assertIn("Use the accepted previous scene", repaired_context)
        self.assertEqual(self.planner._music_arc, arc)

    def test_invalid_authoritative_allocation_fails_without_rewriting_ids(self):
        self.planner._music_narrative_allocation = self.allocation[:-1]
        row = {"video_prompt": "The sailor watches from shore.", "source_event_ids": ["M2"]}

        issue = self._row_validator()(10, row, [])

        self.assertIn("music narrative allocation is invalid", issue or "")
        self.assertEqual(row["source_event_ids"], ["M2"])

    def test_source_failure_uses_focused_visual_repair_then_existing_row_checks(self):
        with patch.object(
            self.planner, "_run_checkpointed_json_batches", return_value=[],
        ) as batches:
            self.planner._plan_with_llm(
                clips=self.clips, clip_contexts=["Clip"] * len(self.clips),
                scene_description=self.contract["source"], bpm=120,
                has_reference=False, reference_image_path=None,
                char_profiles=[], performer_map={},
                video_model="minimax_h3_ref2va_fused_turbo",
            )
        callbacks = batches.call_args.kwargs
        rejected = {"video_prompt": "The sailor waits on shore.", "source_event_ids": []}
        issue = callbacks["row_validator"](1, rejected, [])
        self.assertIn("must visibly perform assigned source event M1", issue)
        repaired = {
            "video_prompt": "The sailor climbs onto the boat.",
            "source_event_ids": ["M1"],
        }
        with patch(
            "services.director.music_narrative_repair.repair_music_narrative_visuals",
            return_value=[repaired],
        ) as focused, patch.object(self.planner, "_call_llm_json") as full:
            result = callbacks["repair_batch"](1, 1, [self.clips[1]], None)
        self.assertEqual(result, [repaired])
        focused.assert_called_once()
        self.assertEqual(focused.call_args.kwargs["row"], rejected)
        full.assert_not_called()
        # Passing the source gate does not bypass the other production checks.
        self.assertIn("missing a usable scene_goal", callbacks["row_validator"](1, repaired, []))

    def test_metadata_keeps_global_index_for_individual_clip_regeneration(self):
        rows = [{"video_prompt": event["text"], "source_event_ids": [event["event_id"]]}
                for event in self.contract["events"]]
        shots = self.planner._convert_to_shots(rows, self.clips[:len(rows)], [], False, {}, None, {})
        for index, shot in enumerate(shots):
            metadata = shot.metadata["_director_music_narrative"]
            self.assertEqual(metadata["index"], index)
            self.assertEqual(metadata["allocation"], self.allocation)
            self.assertEqual(metadata["source_event_ids"], rows[index]["source_event_ids"])

    def test_source_compiler_runs_only_after_two_failed_visual_repairs(self):
        callbacks = self._batch_callbacks()
        rejected = {"video_prompt": "The sailor waits on shore.", "source_event_ids": []}
        self.assertIn("must visibly perform", callbacks["row_validator"](1, rejected, []))
        compiled = {
            "video_prompt": "The sailor climbs onto the boat.",
            "source_event_ids": ["M1"],
            "_music_source_compiled": True,
        }
        previous = {"video_prompt": "The sailor stands on the pier."}
        with patch(
            "services.director.music_narrative_repair.repair_music_narrative_visuals",
            return_value=None,
        ) as focused, patch(
            "services.director.music_narrative_repair.compile_music_narrative_source_row",
            return_value=[compiled],
        ) as compiler, patch.object(self.planner, "_call_llm_json", return_value=[]) as full:
            callbacks["repair_batch"](1, 1, [self.clips[1]], previous)
            compiler.assert_not_called()
            result = callbacks["repair_batch"](1, 1, [self.clips[1]], previous)
        self.assertEqual(result, [compiled])
        self.assertEqual(focused.call_count, 2)
        compiler.assert_called_once()
        self.assertEqual(compiler.call_args.kwargs["previous"], previous)
        self.assertEqual(full.call_count, 1)
        # Source compilation does not skip scene, camera, cast or native validation.
        self.assertIn("missing a usable scene_goal", callbacks["row_validator"](1, compiled, []))

    def test_successful_visual_repair_does_not_compile_source(self):
        callbacks = self._batch_callbacks()
        rejected = {"video_prompt": "The sailor waits on shore.", "source_event_ids": []}
        callbacks["row_validator"](1, rejected, [])
        repaired = {"video_prompt": "The sailor climbs onto the boat.", "source_event_ids": ["M1"]}
        with patch(
            "services.director.music_narrative_repair.repair_music_narrative_visuals",
            return_value=[repaired],
        ), patch(
            "services.director.music_narrative_repair.compile_music_narrative_source_row",
        ) as compiler:
            self.assertEqual(callbacks["repair_batch"](1, 1, [self.clips[1]], None), [repaired])
        compiler.assert_not_called()

    def test_uncached_later_batch_row_still_gets_source_recovery_within_retry_cap(self):
        callbacks = self._batch_callbacks()
        rejected = {"video_prompt": "The sailor waits on shore.", "source_event_ids": []}
        # The original batch stops validation at its first bad row, so this
        # later row has no cached draft for its first repair call.
        with patch.object(self.planner, "_call_llm_json", return_value=[rejected]):
            first = callbacks["repair_batch"](1, 1, [self.clips[1]], None)
        self.assertIn("must visibly perform", callbacks["row_validator"](1, first[0], []))
        compiled = {"video_prompt": "The sailor climbs onto the boat.", "source_event_ids": ["M1"]}
        with patch(
            "services.director.music_narrative_repair.repair_music_narrative_visuals",
            return_value=None,
        ), patch(
            "services.director.music_narrative_repair.compile_music_narrative_source_row",
            return_value=[compiled],
        ) as compiler, patch.object(self.planner, "_call_llm_json") as full:
            second = callbacks["repair_batch"](1, 1, [self.clips[1]], None)
        self.assertEqual(second, [compiled])
        compiler.assert_called_once()
        full.assert_not_called()

    def test_source_compiled_recovery_is_identified_in_saved_metadata(self):
        event = self.contract["events"][0]
        row = {
            "video_prompt": event["text"], "source_event_ids": [event["event_id"]],
            "_music_source_compiled": True,
            "_music_source_recovery_reason": "Assigned action missing from writer draft",
        }
        shots = self.planner._convert_to_shots([row], self.clips[:1], [], False, {}, None, {})
        metadata = shots[0].metadata["_director_music_narrative"]
        self.assertTrue(metadata["source_compiled"])
        self.assertEqual(metadata["recovery_reason"], row["_music_source_recovery_reason"])

    def test_actual_source_recovery_passes_production_and_native_preflight(self):
        callbacks = self._batch_callbacks()
        rejected = {
            "scene_goal": "The sailor waits for departure.",
            "scene_type": "narrative",
            "environment": "A wooden pier beside a lake.",
            "visual_style": "Cinematic coastal film.",
            "lighting": "Soft daylight.",
            "mood": "Hopeful.",
            "subjects_on_screen": [{
                "character_id": "sailor", "speaker_name": "",
                "visual_description": "The sailor in a navy coat.",
                "position_or_relation": "beside the boat",
                "performance_role": "non_vocal",
            }],
            "camera_plan": {"framing": "wide shot", "movement": "slow lateral track"},
            "action_beats": ["The sailor waits on shore."],
            "ending_beat": "The sailor looks across the lake.",
            "video_prompt": "[Shot 1] The sailor waits on shore.",
            "window_prompts": [], "source_event_ids": [],
            "continuity_strategy": "continuous",
        }
        self.assertIn("must visibly perform", callbacks["row_validator"](1, rejected, []))
        # Simulate two unsuccessful model repairs while using the real source
        # compiler and the same full row validator as the running application.
        with patch(
            "services.director.music_narrative_repair.repair_music_narrative_visuals",
            return_value=None,
        ), patch.object(self.planner, "_call_llm_json", return_value=[]):
            callbacks["repair_batch"](1, 1, [self.clips[1]], None)
            result = callbacks["repair_batch"](1, 1, [self.clips[1]], None)
        self.assertEqual(len(result), 1)
        compiled = result[0]
        self.assertTrue(compiled["_music_source_compiled"])
        self.assertIsNone(callbacks["row_validator"](1, compiled, []))
        plan = {
            "video_prompt": compiled["video_prompt"],
            "_director_subjects_on_screen": compiled["subjects_on_screen"],
            "_director_project_context": self.contract["source"],
            "_director_audio_plan": {"mode": "music_driven", "vocal_activity": "instrumental"},
            "_director_music_narrative": {
                "contract": self.contract, "allocation": self.allocation,
                "index": 1, "source_event_ids": compiled["source_event_ids"],
            },
        }
        path = ROOT / "app/services/director_pipeline.py"
        node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_preflight_h3_director_prompts")
        namespace = {"Optional": Optional}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        namespace["_preflight_h3_director_prompts"](
            "minimax_h3_fused_turbo", [plan], prompt_modes=["t2va"], durations=[10.0],
        )
        self.assertIn("climbs onto the boat", plan["video_prompt"])

    def test_gap_clip_recovers_early_sailing_through_production_row_and_native_checks(self):
        self.planner._music_narrative_allocation = [
            {"event_id": "M1", "clip_index": 1},
            {"event_id": "M2", "clip_index": 3},
        ]
        callbacks = self._batch_callbacks()
        previous = {
            "scene_goal": "The sailor boards for the journey.",
            "scene_type": "narrative", "environment": "A wooden harbor pier.",
            "visual_style": "Cinematic coastal film.", "lighting": "Soft daylight.",
            "mood": "Hopeful.", "continuity_strategy": "continuous",
            "subjects_on_screen": [{
                "character_id": "sailor", "speaker_name": "The sailor",
                "visual_description": "The sailor in a navy coat.",
                "position_or_relation": "securely aboard the boat at the pier",
                "performance_role": "vocalist",
            }],
            "camera_plan": {"framing": "medium shot", "movement": "slow lateral track"},
            "video_prompt": "[Shot 1] The sailor climbs onto the boat and settles aboard.",
            "action_beats": ["The sailor climbs onto the boat."],
            "ending_beat": "The sailor is securely aboard the boat at the pier.",
            "window_prompts": [], "source_event_ids": ["M1"],
        }
        rejected = copy.deepcopy(previous)
        rejected.update(
            environment="Open water in the middle of the lake.",
            video_prompt="[Shot 1] The sailor sings aboard the boat beside the pier.",
            scene_goal="The boat is sailing across the lake.",
            ending_beat="The boat sails across the lake.",
            action_beats=["The boat sails across the lake."], source_event_ids=[],
        )
        prior_snapshot = copy.deepcopy(previous)
        self.assertIn("future music narrative event M2", callbacks["row_validator"](2, rejected, [previous]))
        with patch(
            "services.director.music_narrative_repair.repair_music_narrative_visuals", return_value=None,
        ), patch.object(self.planner, "_call_llm_json", return_value=[]):
            callbacks["repair_batch"](1, 2, self.clips[2:3], previous)
            result = callbacks["repair_batch"](1, 2, self.clips[2:3], previous)
        self.assertEqual(len(result), 1)
        recovered = result[0]
        self.assertIsNone(callbacks["row_validator"](2, recovered, [previous]))
        self.assertEqual(previous, prior_snapshot)
        self.assertEqual(recovered["source_event_ids"], [])
        self.assertEqual(recovered["environment"].rstrip("."), previous["environment"].rstrip("."))
        from services.director.h3_dialogue import compile_h3_official_prompt, validate_h3_prompt_contract
        native, _ = compile_h3_official_prompt(
            recovered["video_prompt"], recovered["subjects_on_screen"], [], mode="ref2va",
            audio_plan={"mode": "music_driven", "vocal_activity": "active"},
            closing_blocking=recovered["ending_beat"],
        )
        self.assertEqual(validate_h3_prompt_contract(native, mode="ref2va"), [])
        self.assertNotIn("climbs onto", native)
        self.assertNotIn("sails across", native)
        self.assertIn("lip-syncs only their own audible vocal part", native)

    def test_controller_rechecks_compiled_prompt_before_gpu(self):
        # Load the production preflight function without booting the GPU server.
        path = ROOT / "app/services/director_pipeline.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_preflight_h3_director_prompts")
        namespace = {"Optional": Optional}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        plan = {"video_prompt": "Sailor waits on shore.", "_director_music_narrative": {
            "contract": copy.deepcopy(self.contract), "allocation": self.allocation,
            "index": self.allocation[0]["clip_index"],
            "source_event_ids": [self.contract["events"][0]["event_id"]],
        }}
        with patch("services.director.h3_dialogue.compile_h3_clip_plans") as compiler:
            with self.assertRaisesRegex(ValueError, "story-action preflight failed"):
                namespace["_preflight_h3_director_prompts"]("minimax_h3_ref2va_fused_turbo", [plan])
        compiler.assert_called_once()


if __name__ == "__main__":
    unittest.main()
