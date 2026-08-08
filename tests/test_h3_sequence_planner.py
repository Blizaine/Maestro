import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_sequence_continuity import (  # noqa: E402
    append_generated_reference,
    augment_prompt_with_continuity,
)
from services.h3_sequence_planner import (  # noqa: E402
    compile_h3_reference_sequence_prompts,
    compute_h3_sequence_clips,
    plan_h3_reference_sequence,
)


class H3ReferenceSequencePlannerTests(unittest.TestCase):
    def test_clip_geometry_uses_valid_balanced_h3_lengths(self):
        clips, trim_tail = compute_h3_sequence_clips(960)
        self.assertEqual(len(clips), 3)
        self.assertGreaterEqual(sum(item["frames"] for item in clips), 960)
        self.assertEqual(
            sum(item["frames"] for item in clips) - 960,
            trim_tail,
        )
        for item in clips:
            self.assertGreaterEqual(item["frames"], 124)
            self.assertLessEqual(item["frames"], 345)
            self.assertEqual((item["frames"] - 124) % 17, 0)

    def test_clip_geometry_honors_a_smaller_memory_aware_ceiling(self):
        clips, trim_tail = compute_h3_sequence_clips(
            960,
            max_clip_frames=226,
        )
        self.assertEqual(len(clips), 5)
        self.assertGreaterEqual(sum(item["frames"] for item in clips), 960)
        self.assertEqual(sum(item["frames"] for item in clips) - 960, trim_tail)
        self.assertTrue(all(item["frames"] <= 226 for item in clips))

    def test_compiler_emits_complete_ref2va_context_per_clip(self):
        clips, _ = compute_h3_sequence_clips(500)
        plan = {
            "subject_definitions": "<Picture 1> defines Alex's identity",
            "retention_analysis": "<Picture 1>: fully_preserved for identity",
            "setting_continuity": "A rain-soaked downtown street at night",
            "visual_style": "Cinematic live action with cool practical light",
            "ambient_audio": "Rain and distant traffic",
            "music": "N/A",
            "clips": [
                {
                    "clip": 1,
                    "title": "Approach",
                    "summary": "Alex notices the danger",
                    "opening_state": "Alex enters screen left",
                    "coverage": "dynamic multi-shot action",
                    "pacing": "fast real-time pacing",
                    "shots": [{
                        "shot": 1,
                        "start_seconds": 0.0,
                        "end_seconds": clips[0]["duration_seconds"],
                        "transition": "opening composition",
                        "framing": "wide tracking shot",
                        "camera": "truck right with Alex",
                        "action": "Alex runs toward the stalled car",
                        "dialogue": [],
                        "sound_effects": "Rapid footsteps",
                    }],
                    "closing_state": "Alex reaches the car at screen center",
                },
                {
                    "clip": 2,
                    "title": "Rescue",
                    "summary": "Alex completes the rescue",
                    "opening_state": "Alex stands beside the car",
                    "coverage": "dynamic multi-shot action",
                    "pacing": "fast real-time pacing",
                    "shots": [{
                        "shot": 1,
                        "start_seconds": 0.0,
                        "end_seconds": clips[1]["duration_seconds"],
                        "transition": "opening composition",
                        "framing": "medium low angle",
                        "camera": "push in, then settle",
                        "action": "Alex pulls the driver clear",
                        "dialogue": [],
                        "sound_effects": "Door creak and rain",
                    }],
                    "closing_state": "Alex and the driver are safe on the curb",
                },
            ],
        }
        compiled = compile_h3_reference_sequence_prompts(
            plan,
            clips,
            reference_relationships=(
                "<Picture 1> defines the identity and appearance of Alex"
            ),
            default_retention="<Picture 1>: fully_preserved for identity",
            task_types="reference generation",
        )
        self.assertEqual(len(compiled), 2)
        for item in compiled:
            prompt = item["prompt"]
            self.assertEqual(prompt.count("subject_definitions:"), 1)
            self.assertEqual(prompt.count("summary:"), 1)
            self.assertEqual(prompt.count("retention_analysis:"), 1)
            self.assertEqual(prompt.count("detailed_description:"), 1)
            self.assertEqual(prompt.count("overall_soundscape:"), 1)
            self.assertEqual(prompt.count("non_diegetic_music:"), 1)
            self.assertIn("<Picture 1>", prompt)
        self.assertNotIn("pulls the driver clear", compiled[0]["prompt"])
        self.assertIn("pulls the driver clear", compiled[1]["prompt"])

    @patch("services.llm_service.generate", side_effect=RuntimeError("offline"))
    def test_planner_falls_back_to_reviewable_sequence(self, _generate):
        result = plan_h3_reference_sequence(
            "Alex runs through town and then rescues a driver",
            model_type="minimax_h3_ref2va_pruned",
            resolution="864x480",
            total_frames=720,
            references=[{
                "id": "alex",
                "type": "image",
                "path": "alex.png",
                "filename": "alex.png",
                "role": "Alex",
                "image_intent": "identity",
            }],
            camera_coverage="multi_shot",
        )
        self.assertEqual(result["plan_kind"], "reference_sequence")
        self.assertEqual(result["planned_by"], "deterministic_fallback")
        self.assertGreater(result["window_count"], 1)
        self.assertEqual(
            len(result["window_prompts"]),
            result["window_count"],
        )

    @patch("services.llm_service.generate")
    def test_staged_omni_planner_assigns_each_event_to_one_clip(self, generate):
        prompt = (
            "Alex enters the rain-soaked street. Alex spots a trapped driver. "
            "Alex pulls the driver clear. Alex watches the car settle safely."
        )
        clips, _ = compute_h3_sequence_clips(1200)
        ledger = {
            "subject_continuity": 'Alex remains unchanged {"S1": "Alex"}',
            "setting_continuity": "The same rain-soaked street at night",
            "visual_continuity": "Cinematic live-action realism",
            "editing_style": "Fast motivated editorial coverage",
            "initial_state": "Alex enters screen left",
            "ambient_audio": "Rain and distant traffic",
            "music": "N/A",
            "required_final_outcome": "Alex watches the car settle safely",
            "beats": [
                {
                    "beat_id": f"B{index + 1}",
                    "segment": index + 1,
                    "description": action,
                    "source_event_ids": [f"E{index + 1}"],
                    "dialogue_ids": [],
                    "state_after": f"State after event {index + 1}",
                    "sound_effects": "Natural synchronized effects",
                }
                for index, action in enumerate([
                    "Alex enters the rain-soaked street",
                    "Alex spots a trapped driver",
                    "Alex pulls the driver clear",
                    "Alex watches the car settle safely",
                ])
            ],
            "generated_dialogue": [],
        }
        segments = []
        for index, clip in enumerate(clips):
            number = index + 1
            segments.append({
                "segment": number,
                "title": f"Event {number}",
                "opening_state": f"Opening {number}",
                "coverage": "dynamic cinematic coverage",
                "pacing": "fast real-time pacing",
                "shots": [{
                    "shot": 1,
                    "start_seconds": 0.0,
                    "end_seconds": clip["duration_seconds"],
                    "transition": "opening composition",
                    "framing": "medium-wide shot",
                    "camera": "a motivated tracking move",
                    "beat_ids": [f"B{number}"],
                    "action": ledger["beats"][index]["description"],
                    "dialogue": [],
                    "sound_effects": "Natural synchronized effects",
                }],
                "closing_state": f"State after event {number}",
            })
        generate.side_effect = [json.dumps(ledger), *(json.dumps(item) for item in segments)]
        result = plan_h3_reference_sequence(
            prompt,
            model_type="minimax_h3_ref2va_pruned",
            resolution="864x480",
            total_frames=1200,
            references=[{
                "id": "alex",
                "type": "image",
                "path": "alex.png",
                "filename": "alex.png",
                "role": "Alex",
                "image_intent": "identity",
            }],
            camera_coverage="multi_shot",
        )
        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(generate.call_count, 5)
        self.assertEqual(result["window_count"], 4)
        for index, clip_prompt in enumerate(result["window_prompts"]):
            self.assertIn(ledger["beats"][index]["description"], clip_prompt)
            for other_index, beat in enumerate(ledger["beats"]):
                if other_index != index:
                    self.assertNotIn(beat["description"], clip_prompt)
            self.assertNotIn("{", clip_prompt)
            self.assertNotIn("}", clip_prompt)
            for label in (
                "subject_definitions:", "summary:", "retention_analysis:",
                "detailed_description:", "overall_soundscape:", "non_diegetic_music:",
            ):
                self.assertEqual(clip_prompt.count(label), 1)

    def test_generated_continuity_is_appended_as_weak_composition(self):
        references = [{
            "id": "identity",
            "type": "image",
            "path": "identity.png",
            "filename": "identity.png",
            "role": "Alex",
            "image_intent": "identity",
        }]
        with tempfile.TemporaryDirectory() as directory:
            frame_path = os.path.join(directory, "continuity.png")
            Image.new("RGB", (32, 32), (64, 96, 128)).save(frame_path)
            updated, picture_number = append_generated_reference(
                references,
                frame_path,
                role="preceding clip continuity",
            )
        self.assertEqual(picture_number, 2)
        self.assertEqual(updated[0]["image_intent"], "identity")
        self.assertEqual(updated[1]["image_intent"], "composition")

        prompt = (
            "subject_definitions: <Picture 1> defines Alex.\n\n"
            "summary: [reference generation] Alex continues.\n\n"
            "retention_analysis: <Picture 1>: fully_preserved.\n\n"
            "detailed_description: Alex moves through the scene.\n\n"
            "overall_soundscape: Natural ambience.\n\n"
            "non_diegetic_music: N/A"
        )
        augmented = augment_prompt_with_continuity(
            prompt,
            picture_number=2,
            kind="continuity",
        )
        self.assertIn("<Picture 2>: weak_reference", augmented)
        self.assertIn("not an identity source", augmented)
        self.assertIn("<Picture 1> defines Alex", augmented)

    def test_studio_sequence_wiring_preserves_source_settings(self):
        launch = (APP / "launch.py").read_text(encoding="utf-8")
        handler = (APP / "wgp.py").read_text(encoding="utf-8")
        store = (ROOT / "ui" / "src" / "stores" / "useStore.ts").read_text(
            encoding="utf-8"
        )
        duration = (
            ROOT / "ui" / "src" / "components" / "Sidebar" / "DurationSlider.tsx"
        ).read_text(encoding="utf-8")
        self.assertIn('/api/v1/llm/plan-h3-sequence', launch)
        self.assertIn('"per_clip_minimax_h3_references"', launch)
        self.assertIn('sequence_plan.get("source_prompt")', handler)
        self.assertIn('video_duration_sec=target_duration_sec', handler)
        self.assertIn('api.planH3Sequence', store)
        self.assertIn('h3ReferenceSequence', store)
        self.assertIn('minimax_h3_sequence_clip_frames', store)
        self.assertIn('apply_h3_omni_sequence_memory_policy', launch)
        self.assertIn('body["sliding_window_size"] = min(', launch)
        self.assertIn('omniReferenceSequence', duration)
        self.assertIn('Sequence Clip Length', duration)


if __name__ == "__main__":
    unittest.main()
