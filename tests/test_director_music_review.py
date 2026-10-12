"""Evidence-grounded creative repairs preserve accepted neighboring shots."""
import copy
import ast
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.music_review import (
    grounded_review_findings, music_visual_body, music_visual_reset_issue,
    repair_music_rows,
    repeated_music_visuals,
)
from services.director.planners.music_video import MusicVideoPlanner


def arc_row(index, beat):
    return dict(clip_index=index, visual_beat=beat, purpose="Develop the concept",
                environment="same place", camera="fixed view", source_event_ids=[],
                intentional_reprise=False, reprise_reason="")


class MusicReviewTests(unittest.TestCase):
    def test_visual_body_excludes_identity_summary_and_soundscape(self):
        prompt = ("subject_definitions: repeated identity\nsummary: repeated summary\n"
                  "detailed_description: [Shot 1] The dancer turns toward the mirror.\n"
                  "overall_soundscape: repeated song\nnon_diegetic_music: repeated song")
        self.assertEqual(music_visual_body(prompt), "[Shot 1] The dancer turns toward the mirror.")

    def test_whole_brief_reset_checks_the_visual_body_only(self):
        source = "A dancer leaves home at dawn, crosses the market, meets an old friend, and performs beside the river at sunset."
        self.assertIsNone(music_visual_reset_issue(f"summary: {source}\ndetailed_description: The dancer ties a shoe.\noverall_soundscape: Song.", source))
        self.assertIsNotNone(music_visual_reset_issue(f"detailed_description: The target follows: {source}\noverall_soundscape: Song.", source))

    def test_review_cannot_repair_an_unseen_clip_or_invent_evidence(self):
        rows = [{"video_prompt": "The dancer pauses beside the empty mirror."}]
        finding = {"clip_index": 0, "category": "arc_mismatch",
                   "evidence": "The dancer pauses beside the empty mirror.",
                   "related_clip_index": -1, "related_evidence": "",
                   "repair": "Perform the planned turn toward the audience."}
        self.assertEqual(list(grounded_review_findings([finding], rows, {0})), [0])
        finding["evidence"] = "The dancer jumps out of the window."
        with self.assertRaisesRegex(ValueError, "exact visual quote"):
            grounded_review_findings([finding], rows, {0})
        finding["clip_index"] = 7
        with self.assertRaisesRegex(ValueError, "outside"):
            grounded_review_findings([finding], rows, {0})

    def test_repetition_needs_two_actual_visual_quotes(self):
        quote = "The rider climbs onto the vehicle again."
        rows = [{"video_prompt": quote}, {"video_prompt": quote}]
        finding = dict(clip_index=1, category="repeated_action", evidence=quote,
                       related_clip_index=0, related_evidence=quote,
                       repair="Show the rider checking the route instead of boarding again.")
        self.assertIn(1, grounded_review_findings([finding], rows, {0, 1}))
        finding["related_evidence"] = "not in the prompt"
        with self.assertRaisesRegex(ValueError, "two supplied clips"):
            grounded_review_findings([finding], rows, {0, 1})

    def test_focused_repair_preserves_all_good_rows_and_rechecks_next_boundary(self):
        rows = [{"video_prompt": "Opening", "closing": "aboard"},
                {"video_prompt": "Repeated boarding", "closing": "aboard"},
                {"video_prompt": "Good POV", "opening": "aboard"}]
        original = copy.deepcopy(rows)
        requests = []
        def replace(index, feedback, current):
            requests.append((index, feedback))
            return [{"video_prompt": "Read the map", "closing": "ashore" if len(requests) == 1 else "aboard"}]
        def validate(index, row, previous):
            if index == 2 and row["opening"] != previous[-1]["closing"]:
                return "Next accepted shot requires aboard"
        result = repair_music_rows(rows, {1: "Do not repeat boarding"}, replace, validate)
        self.assertEqual([r[0] for r in requests], [1, 1])
        self.assertIn("Next accepted shot", requests[1][1])
        self.assertEqual(result[0], original[0])
        self.assertEqual(result[2], original[2])
        self.assertEqual(rows, original)

    def test_creative_review_is_cached_and_only_flagged_clip_is_rewritten(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"test": 1})
        planner._music_arc = [arc_row(0, "The performer checks the empty room."),
                              arc_row(1, "The performer turns toward the doorway.")]
        rows = [{"video_prompt": "The performer checks the empty room."},
                {"video_prompt": "The performer checks the empty room."}]
        finding = dict(clip_index=1, category="repeated_action", evidence=rows[1]["video_prompt"],
                       related_clip_index=0, related_evidence=rows[0]["video_prompt"],
                       repair="Show the planned turn toward the doorway.")
        calls = []
        def repair(index, feedback, current):
            calls.append(index)
            self.assertIn("Required closing facts", planner._last_music_creative_issue)
            self.assertIn(current[index]["video_prompt"], planner._last_music_creative_issue)
            return [{"video_prompt": "The performer turns toward the doorway."}]
        with patch.object(planner, "_call_llm_json", side_effect=[[finding], []]) as review:
            result = planner._review_music_plan(rows, [{}, {}], ["", ""], "A quiet performance",
                         row_validator=lambda *args: None, repair_row=repair, batch_size=3)
            cached = planner._review_music_plan(result, [{}, {}], ["", ""], "A quiet performance",
                         row_validator=lambda *args: None, repair_row=repair, batch_size=3)
        self.assertEqual(review.call_count, 2)
        self.assertEqual(calls, [1])
        self.assertEqual(result, cached)
        self.assertEqual(result[0], rows[0])
        receipt = planner._planning_resume_checkpoint["music_creative_review"]
        self.assertEqual(receipt["preserved_clip_indices"], [0])
        self.assertEqual(receipt["repaired_clip_indices"], [1])

    def test_arc_precedes_detail_and_roundtrips_long_cached_plan(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"test": 2})
        planner._music_narrative_contract = {"source": "An abstract pattern develops with the song.", "events": []}
        planner._use_lyrics_context = False
        clips = [{"start": i * 10, "end": (i + 1) * 10} for i in range(60)]
        outline = [arc_row(i, f"Amber lines rotate into a distinct lattice with {i+3} spokes.") for i in range(30)]
        for row in outline[1:]:
            row.update(intentional_reprise=True, reprise_reason="Develop the recurring lattice with a new spoke count.")
        with patch.object(planner, "_call_llm_json", return_value=outline) as call:
            first = planner._plan_music_arc(clips, planner._music_narrative_contract["source"])
            cached = planner._plan_music_arc(clips, planner._music_narrative_contract["source"])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(first), 60)
        self.assertEqual(first, cached)
        self.assertLessEqual(call.call_args.kwargs["json_schema"]["maxItems"], 48)

    def test_outline_shorthand_notes_do_not_replace_detailed_action_validation(self):
        from services.director.music_narrative import build_music_narrative_contract
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"test": 3})
        planner._music_narrative_contract = build_music_narrative_contract("The traveler runs across the field.")
        planner._use_lyrics_context = False
        clips = [{"start": i * 10, "end": (i + 1) * 10} for i in range(4)]
        outline = [
            arc_row(0, "The traveler stands alone and sings in the open field."),
            arc_row(1, "The traveler runs across the field toward the horizon."),
            arc_row(2, "The traveler studies a weathered sign beside the path."),
            arc_row(3, "The traveler raises a lantern as the evening light fades."),
        ]
        outline[0]["source_event_ids"] = ["M1"]
        with patch.object(planner, "_call_llm_json", return_value=outline) as writer:
            result = planner._plan_music_arc(clips, planner._music_narrative_contract["source"])
            cached = planner._plan_music_arc(clips, planner._music_narrative_contract["source"])
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(result, cached)
        self.assertEqual(result[0]["visual_beat"], outline[0]["visual_beat"])
        self.assertTrue(planner._music_arc_assignment_notes)
        self.assertEqual(planner._planning_resume_checkpoint["music_arc_assignment_notes"],
                         planner._music_arc_assignment_notes)
        from services.director.music_narrative import music_narrative_row_issue
        issue = music_narrative_row_issue(
            {"video_prompt": outline[0]["visual_beat"]},
            contract=planner._music_narrative_contract,
            allocation=[{"event_id": "M1", "clip_index": 0}], index=0,
        )
        self.assertIsNotNone(issue)  # Final detail still cannot omit the run.

    def test_keyed_outline_assignments_cannot_duplicate_action_ids_between_beats(self):
        from services.director.music_narrative import build_music_narrative_contract
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"test": 4})
        planner._music_narrative_contract = build_music_narrative_contract("The traveler runs across the field.")
        planner._use_lyrics_context = False
        clips = [{"start": i * 10, "end": (i + 1) * 10} for i in range(4)]
        beats = [
            arc_row(0, "The traveler stands alone and sings in the open field."),
            arc_row(1, "The traveler runs across the field toward the horizon."),
            arc_row(2, "The traveler studies a weathered sign beside the path."),
            arc_row(3, "The traveler raises a lantern as the evening light fades."),
        ]
        for beat in beats:
            beat.pop("source_event_ids")
        invalid = {"beats": beats, "source_event_clips": {"M1": True}}
        valid = {"beats": beats, "source_event_clips": {"M1": 1}}
        with patch.object(planner, "_call_llm_json", side_effect=[[invalid], [valid]]) as writer:
            result = planner._plan_music_arc(clips, planner._music_narrative_contract["source"])
        self.assertEqual([row["source_event_ids"] for row in result], [[], ["M1"], [], []])
        self.assertNotIn("source_event_ids", beats[0])  # Writer input is not mutated.
        schema = writer.call_args.kwargs["json_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["source_event_clips"]["required"], ["M1"])
        self.assertFalse(schema["properties"]["source_event_clips"]["additionalProperties"])
        self.assertNotIn("source_event_ids", schema["properties"]["beats"]["items"]["properties"])

    def test_malformed_keyed_outline_is_repaired_without_crashing(self):
        from services.director.music_narrative import build_music_narrative_contract
        clips = [{"start": i * 10, "end": (i + 1) * 10} for i in range(4)]
        beats = [arc_row(i, text) for i, text in enumerate([
            "The traveler sings beside a weathered sign.",
            "The traveler runs across the field toward the horizon.",
            "The traveler examines a broken lantern beside the path.",
            "The traveler raises the restored lantern in the evening light.",
        ])]
        for beat in beats:
            beat.pop("source_event_ids")
        valid = {"beats": beats, "source_event_clips": {"M1": 1}}
        malformed = []
        for index in ([], {}, True):
            item = copy.deepcopy(valid)
            item["beats"][0]["clip_index"] = index
            malformed.append(item)
        item = copy.deepcopy(valid)
        item["extra"] = "unexpected"
        malformed.append(item)
        item = copy.deepcopy(valid)
        item["beats"][0]["source_event_ids"] = ["M1"]
        malformed.append(item)
        for invalid in malformed:
            with self.subTest(invalid=invalid):
                planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
                planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"test": 5})
                planner._music_narrative_contract = build_music_narrative_contract("The traveler runs across the field.")
                planner._use_lyrics_context = False
                with patch.object(planner, "_call_llm_json", side_effect=[[invalid], [valid]]) as writer:
                    result = planner._plan_music_arc(clips, planner._music_narrative_contract["source"])
                self.assertEqual(writer.call_count, 2)
                self.assertEqual(result[1]["source_event_ids"], ["M1"])
                self.assertIn("Repair this outline", writer.call_args.kwargs["user_prompt"])

    def test_literal_recycling_is_caught_without_blocking_intentional_refrains(self):
        quote = "The performer raises both hands and walks around the empty microphone stand."
        rows = [{"video_prompt": quote}, {"video_prompt": quote}]
        arc = [arc_row(0, "Opening gesture"), arc_row(1, "New verse action")]
        self.assertEqual(list(repeated_music_visuals(rows, arc)), [1])
        arc[1]["intentional_reprise"] = True
        self.assertEqual(repeated_music_visuals(rows, arc), {})

    def test_subjective_review_cannot_rewrite_a_good_shot(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._configure_planning_runtime({}, kind="music_video", fingerprint_payload={"test": 3})
        planner._music_arc = [arc_row(0, "The performer pauses beside the empty mirror.")]
        rows = [{"video_prompt": "The performer pauses beside the empty mirror."}]
        finding = dict(clip_index=0, category="arc_mismatch", evidence=rows[0]["video_prompt"],
                       related_clip_index=-1, related_evidence="",
                       repair="Use a more exciting pose beside the mirror.")
        with patch.object(planner, "_call_llm_json", return_value=[finding]):
            result = planner._review_music_plan(rows, [{}], [""], "A quiet performance",
                        row_validator=lambda *args: None,
                        repair_row=lambda *args: self.fail("Accepted shot was rewritten"), batch_size=3)
        self.assertEqual(result, rows)
        self.assertEqual(len(planner._planning_resume_checkpoint["music_creative_review"]["review_notes"]), 1)

    def test_structural_repair_receives_the_actual_validation_error(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._uses_generated_shot_images = False
        with patch.object(planner, "_run_checkpointed_json_batches", return_value=[]) as batches:
            planner._plan_with_llm(clips=[{"start": 0, "end": 10}], clip_contexts=["Clip 1"],
                scene_description="An abstract ink pattern", bpm=120, has_reference=False,
                reference_image_path=None, char_profiles=[], performer_map={},
                video_model="minimax_h3_ref2va_fused_turbo")
        config = batches.call_args.kwargs
        draft = {"video_prompt": "The traveler runs across the battlefield, cape streaming behind."}
        issue = config["row_validator"](0, draft, [])
        self.assertIn("scene_goal", issue)
        with patch.object(planner, "_call_llm_json", return_value=[]) as writer:
            config["repair_batch"](1, 0, [{"start": 0, "end": 10}], None)
        self.assertIn("REQUIRED VALIDATION REPAIR", writer.call_args.kwargs["user_prompt"])
        self.assertIn(issue, writer.call_args.kwargs["user_prompt"])
        self.assertIn("EDIT THE REJECTED DRAFT IN PLACE", writer.call_args.kwargs["user_prompt"])
        self.assertIn(draft["video_prompt"], writer.call_args.kwargs["user_prompt"])
        self.assertNotIn('"speaker_name": "Ari"', writer.call_args.kwargs["system_prompt"])

    def test_fixed_narrative_error_is_not_repeated_during_state_repair(self):
        planner = MusicVideoPlanner(llm_generate=lambda **kw: "[]")
        planner._uses_generated_shot_images = False
        planner._music_narrative_contract = {"events": [{"event_id": "M1"}]}
        planner._music_narrative_allocation = [{"event_id": "M1", "clip_index": 0}]
        planner._last_music_narrative_issue = "Missing run from the earlier draft"
        with patch.object(planner, "_run_checkpointed_json_batches", return_value=[]) as batches:
            planner._plan_with_llm(clips=[{"start": 0, "end": 10}], clip_contexts=["Clip 1"],
                scene_description="A traveler runs through a field", bpm=120, has_reference=False,
                reference_image_path=None, char_profiles=[], performer_map={},
                video_model="minimax_h3_ref2va_fused_turbo")
        config = batches.call_args.kwargs
        with patch("services.director.planners.music_video.music_narrative_row_issue", return_value=None), \
             patch.object(planner, "_music_video_row_issue", return_value="Missing opening contact fact"):
            config["row_validator"](0, {"video_prompt": "The traveler runs through the field."}, [])
        self.assertEqual(planner._last_music_narrative_issue, "")
        self.assertEqual(planner._last_music_row_issue, "Missing opening contact fact")

    def test_final_visual_guard_preserves_explicit_user_edit(self):
        path = Path(__file__).resolve().parents[1] / "app/services/director_pipeline.py"
        node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_preflight_h3_director_prompts")
        namespace = {"Optional": Optional}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        source = "A dancer leaves home at dawn, crosses the market, meets an old friend, and performs beside the river at sunset."
        plan = {"video_prompt": "detailed_description: " + source,
                "_director_music_arc": {"source_brief": source}}
        with patch("services.director.h3_dialogue.compile_h3_clip_plans"):
            with self.assertRaisesRegex(ValueError, "visual preflight"):
                namespace["_preflight_h3_director_prompts"]("minimax_h3_ref2va", [plan])
            plan["_director_prompt_user_edited"] = True
            self.assertEqual(namespace["_preflight_h3_director_prompts"]("minimax_h3_ref2va", [plan]), [plan])


if __name__ == "__main__":
    unittest.main()
