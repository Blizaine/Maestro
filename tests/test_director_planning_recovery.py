"""Production regressions: twelve-event ads, atomic repair and saved stages."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.director.contract_repair import (
    apply_validated_compact_patch, build_compact_source_physical_repair_request,
)
from services.director.physical_continuity import normalize_physical_state
from services.director.source_contract import build_source_contract, source_contract_issues
from services.director.planners.short_film import ShortFilmPlanner, _h3_dialogue_density_targets
from services import director_pipeline as pipeline


def contract():
    return {"events": [{"event_id": f"E{i}", "text": f"The display shows service {i}."}
                       for i in range(1, 13)],
            "audio_policy": {"generated_speech_allowed": False, "generated_music_allowed": False}}


def rows(count=9):
    return [{"title": f"Service {i}", "duration_sec": 6,
             "scene_goal": f"The display shows service {i}.",
             "subjects_on_screen": [], "source_event_ids": [f"E{i}"],
             "physical_state": {"opening": [], "transitions": [], "closing": []},
             "camera_plan": {"framing": "close", "movement": "static"},
             "dialogue_beats": [], "action_beats": [f"The display shows service {i}."]}
            for i in range(1, count + 1)]


def proposal():
    return {"event_shots": {f"E{i}": min(i, 9) for i in range(1, 13)},
            "initial_facts": [], "event_transitions": {f"E{i}": [] for i in range(1, 13)}}


class ExampleDirectorRecoveryTests(unittest.TestCase):
    def test_audio_permissions_outrank_dialogue_pacing_in_a_silent_ad(self):
        source = ('An advertisement about responsive conversation and banter. '
                  'No text, logos, dialogue, subtitles, voiceover, or music.')
        self.assertIsNone(_h3_dialogue_density_targets(source, target_duration=60))

    def test_bounded_sequence_uses_owned_events_not_global_brief(self):
        planner = ShortFilmPlanner(llm_generate=lambda **kw: '[]',
            llm_generate_streaming=lambda **kw: 'INT. OFFICE - DAY\nThe display lights up. The light dims.')
        planner._video_model = 'minimax_h3_fused_turbo'
        planner._uses_generated_shot_images = False
        planner._long_form_story_blueprint_override = []
        owned = contract(); owned['events'] = owned['events'][-2:]
        planner._long_form_source_contract_override = owned
        with patch.object(planner, '_plan_story_h3_native', return_value=([], 'Example')) as native:
            planner._plan_story_driven(story_description='Global brief: twelve scenes. No speech or music.',
                reference_image_path=None, char_profiles=[], has_reference=False, target_duration=20,
                target_scenes=None, narrative_mode=False, fps=24, frames_steps=17,
                frames_minimum=124, frames_maximum=345, nsfw=False, polish_block='', multishot_lora_mode=False)
        self.assertEqual([e['event_id'] for e in native.call_args.kwargs['source_contract_override']['events']], ['E11', 'E12'])
        self.assertTrue(native.call_args.kwargs['require_source_contract'])

    def test_twelve_required_nine_assigned_focused_repair_preserves_nine(self):
        original = rows()
        before = copy.deepcopy(original)
        self.assertTrue(source_contract_issues(original, contract(), require_ids=True))
        _, prompt, schema = build_compact_source_physical_repair_request(original, contract(), [])
        self.assertIn('"missing_event_ids":["E10","E11","E12"]', prompt)
        for i in range(1, 10):
            rule = schema["properties"]["event_shots"]["properties"][f"E{i}"]
            self.assertEqual((rule["minimum"], rule["maximum"]), (i, i))
        fixed = apply_validated_compact_patch(original, proposal(), contract())
        self.assertEqual([e for row in fixed for e in row["source_event_ids"]],
                         [f"E{i}" for i in range(1, 13)])
        self.assertEqual(original, before)
        self.assertEqual(fixed[0]["camera_plan"], original[0]["camera_plan"])

    def test_bad_repairs_never_modify_original(self):
        original = rows()
        before = copy.deepcopy(original)
        bad = []
        missing = proposal(); del missing["event_shots"]["E12"]; bad.append(missing)
        order = proposal(); order["event_shots"]["E12"] = 1; bad.append(order)
        omitted = proposal(); del omitted["initial_facts"]; bad.append(omitted)
        unknown = proposal()
        unknown["event_transitions"]["E2"] = [{"subject_id": "homeowner", "predicate": "contact",
             "to_value": "phone", "action": "The homeowner touches the phone."}]
        bad.append(unknown)
        for value in bad:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                apply_validated_compact_patch(original, value, contract())
            self.assertEqual(original, before)

    def test_valid_plan_cannot_be_corrupted_by_reallocation(self):
        original = apply_validated_compact_patch(rows(), proposal(), contract())
        before = copy.deepcopy(original)
        corrupt = proposal(); corrupt["event_shots"]["E9"] = 8
        with self.assertRaisesRegex(ValueError, "moved an existing"):
            apply_validated_compact_patch(original, corrupt, contract())
        self.assertEqual(original, before)

    def test_duplicate_and_reordered_allocations_are_detected(self):
        for ids in (["E1", "E1"], ["E2", "E1"]):
            value = rows(); value[0]["source_event_ids"] = ids
            self.assertTrue(source_contract_issues(value, contract(), require_ids=True))

    def test_empty_from_value_is_actionable_and_not_invented(self):
        with self.assertRaisesRegex(ValueError, "non-empty from_value"):
            normalize_physical_state({"opening": [], "closing": [], "transitions": [{
                "subject_id": "homeowner", "predicate": "contact", "from_value": "",
                "to_value": "phone", "action": "The homeowner touches the phone."}]})

    def test_editorial_voiceover_does_not_pollute_final_source_event(self):
        source = ("SHOT 1 — 00:00–00:05 — CALL\nA phone vibrates.\n"
                  "SHOT 2 — 00:05–00:10 — CLOSE\nA light dims.\n\n"
                  "VOICEOVER — RECORD SEPARATELY IN EDITOR\n00:00–00:10: Welcome to Example Company.\n"
                  "END CARD (EDITOR GRAPHICS, NOT AI VIDEO)\nMOJO LABS\n")
        events = build_source_contract(source)["events"]
        self.assertEqual(len(events), 2)
        self.assertNotIn("VOICEOVER", events[-1]["text"])
        self.assertNotIn("MOJO LABS", events[-1]["text"])
        self.assertEqual(events[-1]["source_end_seconds"], 10)

    def test_failed_planning_stage_survives_checkpoint_and_resume(self):
        saved = []
        planner = ShortFilmPlanner(llm_generate=lambda **kw: "")
        planner._planning_checkpoint_callback = saved.append
        def fail(**kwargs):
            planner._save_story_stage("screenplay", "INT. OFFICE. NORA: We can help.")
            planner._save_story_stage("native_shots", rows())
            raise ValueError("physical_state.transitions[0] needs a non-empty from_value")
        with patch.object(planner, "_plan_story_driven_impl", side_effect=fail):
            with self.assertRaises(ValueError):
                planner._plan_story_driven(story_description="Example", target_duration=60)
        restored = ShortFilmPlanner(llm_generate=lambda **kw: "")
        restored._planning_resume_checkpoint = json.loads(json.dumps(saved[-1]))
        def continue_plan(**kwargs):
            self.assertIn("NORA", restored._cached_story_stage("screenplay"))
            self.assertEqual(len(restored._cached_story_stage("native_shots")), 9)
            return [], "Example"
        with patch.object(restored, "_plan_story_driven_impl", side_effect=continue_plan):
            restored._plan_story_driven(story_description="Example", target_duration=60)
        with patch.object(restored, "_plan_story_driven_impl",
                          side_effect=lambda **kw: self.assertIsNone(restored._cached_story_stage("screenplay"))):
            restored._plan_story_driven(story_description="Changed project", target_duration=60)

    def test_empty_plan_media_repair_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "_director_pipeline_example.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Use Resume"):
                pipeline._plan_pipeline_repair(folder, "example", {"clips": []})

    def test_auto_planning_only_and_resume_never_enter_media_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            pid = 'example-planning-only'
            record = {'id': pid, 'status': 'running', 'out_dir': folder,
                      'clip_plans': [], 'clip_images': [], 'output_files': [], 'progress': {},
                      'params': {'pipeline_type': 'short_film_story', 'auto_mode': True,
                                 'planning_only': True, 'video_model': 'test-video', 'image_model': '',
                                 'prepared_clip_plans': [{'video_prompt': 'A phone rings.'}],
                                 'prepared_planned_clips': [{'start': 0, 'end': 6, 'duration_sec': 6}]}}
            pipeline._pipelines[pid] = record
            fake = SimpleNamespace(save_path=folder, server_config={'services': {'director_prompt_polish': 'off'}},
                                   get_model_def=lambda model: {})
            try:
                with patch.object(pipeline, '_wgp', fake), patch.object(pipeline, '_wait_for_gpu', return_value=True), \
                     patch.object(pipeline, '_run_image_generation') as images, \
                     patch.object(pipeline, '_run_video_generation') as videos, \
                     patch.object(pipeline, '_run_planning') as planning:
                    pipeline._run_pipeline(pid)
                    self.assertEqual(record['phase'], 'planning_ready', record.get('error'))
                    self.assertEqual(record['status'], 'completed')
                    self.assertTrue(record['params']['auto_mode'])
                    saved = json.loads(Path(folder, f'_director_pipeline_{pid}.json').read_text(encoding='utf-8'))
                    self.assertTrue(saved['_params_snapshot']['planning_only'])
                    self.assertEqual(saved['clips'][0]['video_prompt'], 'A phone rings.')
                    check = pipeline._plan_pipeline_repair(folder, pid, saved)
                    self.assertTrue(check['validation_only'])
                    self.assertEqual(check['image_indices'], [])
                    self.assertEqual(check['video_indices'], [])
                    pipeline._run_pipeline(pid, resume=True)
                    self.assertEqual(record['phase'], 'planning_ready', record.get('error'))
                    images.assert_not_called(); videos.assert_not_called(); planning.assert_not_called()
            finally:
                pipeline._pipelines.pop(pid, None)


if __name__ == "__main__":
    unittest.main()
