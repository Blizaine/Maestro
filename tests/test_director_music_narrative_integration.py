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

    def _row_validator(self):
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
        return batches.call_args.kwargs["row_validator"]

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

    def test_invalid_authoritative_allocation_fails_without_rewriting_ids(self):
        self.planner._music_narrative_allocation = self.allocation[:-1]
        row = {"video_prompt": "The sailor watches from shore.", "source_event_ids": ["M2"]}

        issue = self._row_validator()(10, row, [])

        self.assertIn("music narrative allocation is invalid", issue or "")
        self.assertEqual(row["source_event_ids"], ["M2"])

    def test_metadata_keeps_global_index_for_individual_clip_regeneration(self):
        rows = [{"video_prompt": event["text"], "source_event_ids": [event["event_id"]]}
                for event in self.contract["events"]]
        shots = self.planner._convert_to_shots(rows, self.clips[:len(rows)], [], False, {}, None, {})
        for index, shot in enumerate(shots):
            metadata = shot.metadata["_director_music_narrative"]
            self.assertEqual(metadata["index"], index)
            self.assertEqual(metadata["allocation"], self.allocation)
            self.assertEqual(metadata["source_event_ids"], rows[index]["source_event_ids"])

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
