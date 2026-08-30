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
    _reference_context,
    build_manual_h3_reference_sequence_plan,
    compile_h3_reference_sequence_prompts,
    compute_h3_native_sequence_windows,
    compute_h3_sequence_clips,
    plan_h3_reference_sequence,
    parse_h3_manual_sequence_prompts,
    reviewed_h3_sequence_plan_matches,
    resolve_h3_sequence_source_prompt,
)


class H3ReferenceSequencePlannerTests(unittest.TestCase):
    def test_saved_video_character_and_voice_share_one_subject(self):
        relationships, retention, task_types = _reference_context([
            {
                "type": "audio",
                "path": "voice.wav",
                "role": "Blaine",
                "audio_intent": "voice",
                "library_character_id": "blaine",
            },
            {
                "type": "video",
                "path": "blaine.mp4",
                "role": "Blaine",
                "video_intent": "character",
                "include_audio": False,
                "library_character_id": "blaine",
            },
        ])
        self.assertIn("<Subject 1> is Blaine", relationships)
        self.assertIn("<Audio 1> is the voice-timbre reference", relationships)
        self.assertIn("for <Subject 1>", relationships)
        self.assertIn("fully_preserved", retention)
        self.assertIn("audio reference", task_types)

    def test_reviewed_sequence_snapshot_requires_exact_geometry(self):
        geometry = compute_h3_native_sequence_windows(
            800,
            window_frames=345,
            overlap_frames=18,
            fps=24,
        )
        prompts = [f"Reviewed Omni prompt {index}." for index in range(1, len(geometry) + 1)]
        plan = {
            "source_prompt": "A hero crosses a ruined city.",
            "plan_kind": "reference_sequence",
            "camera_coverage": "auto",
            "model_type": "minimax_h3_ref2va",
            "resolution": "1280x704",
            "window_frames": 345,
            "native_continuation": True,
            "overlap_frames": 18,
            "per_clip_frames": [item["frames"] for item in geometry],
            "windows": [
                {**item, "prompt": prompts[index]}
                for index, item in enumerate(geometry)
            ],
            "window_prompts": prompts,
        }
        common = {
            "source_prompt": plan["source_prompt"],
            "model_type": plan["model_type"],
            "resolution": plan["resolution"],
            "geometry": geometry,
            "window_frames": 345,
            "camera_coverage": "auto",
            "overlap_frames": 18,
            "native_continuation": True,
        }
        self.assertTrue(
            reviewed_h3_sequence_plan_matches(plan, prompts, **common)
        )
        self.assertFalse(
            reviewed_h3_sequence_plan_matches(
                plan,
                prompts,
                **{**common, "overlap_frames": 1},
            )
        )

    def test_compiler_uses_official_full_reference_detail_shape(self):
        clips, _ = compute_h3_sequence_clips(226)
        plan = {
            "subject_definitions": "Stable speaking identity: S1 is Alex.",
            "retention_analysis": "",
            "setting_continuity": "A rain-soaked city street at night",
            "visual_style": "cinematic live action with cool practical light",
            "ambient_audio": "Rain and distant traffic",
            "music": "N/A",
            "clips": [{
                "clip": 1,
                "title": "Rescue",
                "summary": "Alex reaches the stalled car",
                "opening_state": "Alex enters screen-left beside the car",
                "coverage": "dynamic multi-shot coverage",
                "pacing": "fast real-time pacing",
                "shots": [{
                    "shot": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 4.0,
                    "transition": "opening composition",
                    "framing": "wide tracking shot",
                    "camera": "the camera trucks right with Alex",
                    "action": "Alex runs toward the stalled car",
                    "dialogue": [],
                    "sound_effects": "Rapid footsteps",
                }, {
                    "shot": 2,
                    "start_seconds": 4.0,
                    "end_seconds": clips[0]["duration_seconds"],
                    "transition": "hard cut",
                    "framing": "low-angle medium shot",
                    "camera": "the camera pushes toward the open door",
                    "action": "Alex pulls the driver clear",
                    "dialogue": [],
                    "sound_effects": "Door metal creaks",
                }],
                "closing_state": "Alex and the driver reach the curb",
            }],
        }
        compiled = compile_h3_reference_sequence_prompts(
            plan,
            clips,
            reference_relationships=(
                "<Subject 1> is Alex, whose identity and appearance come from "
                "<Picture 1>; reject the picture's background and pose."
            ),
            default_retention=(
                "<Subject 1>: fully_preserved - preserve the identity and "
                "appearance defined by <Picture 1>."
            ),
            task_types="reference generation",
        )
        prompt = compiled[0]["prompt"]
        detailed = prompt.split("detailed_description: ", 1)[1].split(
            "\n\noverall_soundscape:", 1,
        )[0]

        self.assertTrue(detailed.startswith(
            "The target video uses cinematic live action with cool practical light."
        ))
        self.assertGreater(detailed.index("[Shot 1]"), 0)
        self.assertIn("<Subject 1> enters screen-left", detailed)
        self.assertIn("[Shot 2] At 00:04.000, hard cut", detailed)

    def test_manual_native_sequence_preserves_each_prompt_exactly(self):
        source = "First exact window prompt\nSecond exact window prompt\nThird exact window prompt"
        result = build_manual_h3_reference_sequence_plan(
            source,
            model_type="minimax_h3_ref2va",
            resolution="864x480",
            total_frames=960,
            references=[{
                "type": "image",
                "path": "hero.png",
                "role": "Hero",
            }],
            max_clip_frames=345,
            overlap_frames=18,
            native_continuation=True,
        )
        self.assertEqual(result["planned_by"], "manual")
        self.assertEqual(result["plan_kind"], "reference_sequence")
        self.assertEqual(result["window_count"], 3)
        self.assertEqual(
            result["window_prompts"],
            [
                "First exact window prompt",
                "Second exact window prompt",
                "Third exact window prompt",
            ],
        )
        self.assertEqual(
            [window["prompt"] for window in result["windows"]],
            result["window_prompts"],
        )

    def test_manual_sequence_rejects_a_prompt_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "exactly 3 non-empty prompt lines"):
            parse_h3_manual_sequence_prompts(
                "first window\nsecond window",
                expected_count=3,
                native_continuation=True,
            )

    def test_manual_hard_cut_sequence_uses_independent_clip_geometry(self):
        result = build_manual_h3_reference_sequence_plan(
            "clip one\nclip two\nclip three",
            model_type="minimax_h3_ref2va",
            resolution="864x480",
            total_frames=960,
            references=[],
            max_clip_frames=345,
            native_continuation=False,
        )
        self.assertFalse(result["native_continuation"])
        self.assertEqual(result["window_count"], 3)
        self.assertEqual(result["window_prompts"], ["clip one", "clip two", "clip three"])

    def test_saved_runtime_prompts_restore_the_original_story(self):
        prompts = ["subject_definitions: first", "subject_definitions: second"]
        source = "Alex crosses town and rescues a driver"
        restored = resolve_h3_sequence_source_prompt(
            "\n---CLIP_BOUNDARY---\n".join(prompts),
            {"source_prompt": source},
            prompts,
        )
        self.assertEqual(restored, source)

    def test_an_edited_story_is_not_replaced_by_the_cached_source(self):
        restored = resolve_h3_sequence_source_prompt(
            "Alex takes a different route through town",
            {"source_prompt": "Alex crosses town and rescues a driver"},
            ["compiled clip one", "compiled clip two"],
        )
        self.assertEqual(restored, "Alex takes a different route through town")

    def test_independent_h3_clips_receive_only_their_local_prompt(self):
        launch = (APP / "launch.py").read_text(encoding="utf-8")
        store = (ROOT / "ui" / "src" / "stores" / "useStore.ts").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'clip_params["h3_window_prompts"] = [',
            launch,
        )
        self.assertIn('clip_params["prompt"]', launch)
        self.assertIn("restoredH3SourcePrompt", store)

    def test_native_sequence_geometry_accounts_for_overlap(self):
        windows = compute_h3_native_sequence_windows(
            960,
            window_frames=345,
            overlap_frames=18,
            fps=24,
        )
        self.assertEqual(len(windows), 3)
        self.assertEqual(
            [(item["start_frame"], item["end_frame"]) for item in windows],
            [(0, 345), (345, 672), (672, 960)],
        )
        self.assertEqual([item["frames"] for item in windows], [345, 327, 288])

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

    def test_requested_nonverbal_reaction_stays_in_its_assigned_omni_clip(self):
        clips, _ = compute_h3_sequence_clips(500)
        plan = {
            "subject_definitions": "S1 is Alex",
            "retention_analysis": "Keep Alex consistent",
            "setting_continuity": "A mountain overlook",
            "visual_style": "Cinematic realism",
            "ambient_audio": "Mountain wind",
            "music": "N/A",
            "requested_nonverbal_vocals": (
                "Requested nonverbal vocalizations remain audible: laugh"
            ),
            "clips": [
                {
                    "clip": index + 1,
                    "title": f"Beat {index + 1}",
                    "summary": f"Beat {index + 1}",
                    "opening_state": "Alex stands at the overlook",
                    "coverage": "continuous coverage",
                    "pacing": "natural real-time pacing",
                    "shots": [{
                        "shot": 1,
                        "start_seconds": 0.0,
                        "end_seconds": clip["duration_seconds"],
                        "transition": "opening composition",
                        "framing": "medium shot",
                        "camera": "tracks Alex",
                        "action": (
                            "Alex laughs and starts running"
                            if index == 0 else "Alex keeps running down the trail"
                        ),
                        "dialogue": [],
                        "sound_effects": (
                            "the requested laughter"
                            if index == 0 else "rapid footsteps"
                        ),
                    }],
                    "closing_state": "Alex continues along the trail",
                }
                for index, clip in enumerate(clips)
            ],
        }
        compiled = compile_h3_reference_sequence_prompts(
            plan,
            clips,
            reference_relationships="",
            default_retention="",
            task_types="reference generation",
        )
        self.assertIn("nonverbal vocalizations remain audible: laugh", compiled[0]["prompt"])
        self.assertNotIn("nonverbal vocalizations remain audible: laugh", compiled[1]["prompt"])

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

    @patch("services.llm_service.generate", side_effect=RuntimeError("offline"))
    def test_drive_soundtrack_is_target_conditioning_not_an_omni_audio_reference(
        self,
        _generate,
    ):
        result = plan_h3_reference_sequence(
            "A singer performs the supplied song.",
            model_type="minimax_h3_ref2va",
            resolution="864x480",
            total_frames=720,
            references=[
                {
                    "type": "audio",
                    "path": "song.wav",
                    "role": "the performance",
                    "audio_intent": "drive",
                },
                {
                    "type": "image",
                    "path": "singer.png",
                    "role": "the singer",
                    "image_intent": "identity",
                },
            ],
        )
        for prompt in result["window_prompts"]:
            self.assertIn("exact target soundtrack", prompt.lower())
            self.assertNotIn("<Audio 1>", prompt)
            self.assertIn("<Picture 1>", prompt)

    @patch("services.llm_service.generate", side_effect=RuntimeError("offline"))
    def test_fallback_keeps_retention_out_of_subject_definitions(self, _generate):
        result = plan_h3_reference_sequence(
            "Alex enters the city. Alex reaches the station. Alex boards the train.",
            model_type="minimax_h3_ref2va",
            resolution="864x480",
            total_frames=960,
            references=[{
                "type": "image",
                "path": "alex.png",
                "role": "Alex",
                "image_intent": "identity",
            }],
            overlap_frames=18,
            native_continuation=True,
        )
        first = result["window_prompts"][0]
        subjects, remainder = first.split("\n\nsummary:", 1)
        self.assertNotIn("fully_preserved", subjects)
        self.assertIn("fully_preserved", remainder)

    @patch("services.llm_service.generate", side_effect=RuntimeError("offline"))
    def test_spaced_dialogue_tags_keep_character_ownership_and_event_order(
        self,
        _generate,
    ):
        from models.minimax_h3.ref2va import (
            align_ref2va_voice_reference_order,
            ensure_ref2va_prompt_relationships,
        )

        prompt = (
            "Yoda is in Dagobah swamp looking through a bag he found. "
            "Thanos is standing in the swamp and says to Yoda, "
            "< d>You could not live without Maestro. And where did it bring you? "
            "Back to me.</d> Yoda waves his hand slowly while saying, "
            "<d>Powerful, the creative tool has become. Perfectly balanced, "
            "Maestro is.</d> Thanos responds <d>as all things should be.</d> "
            "Atmospheric ambiance. Character voices sound natural in the environment. "
            "Camera pans to Blaine who waves and says "
            "<d>hey guys! check out Maestro version two! it has so many...</d> "
            "Thanos snaps his fingers. Blaine turns to dust. Yoda grunts approvingly. "
            "Cinematic film. Cinematic camera movements."
        )
        references = []
        for role in ("Yoda", "Thanos", "Blaine"):
            character_id = role.casefold()
            references.extend([
                {
                    "type": "image",
                    "path": f"{character_id}.png",
                    "role": role,
                    "image_intent": "identity",
                    "library_character_id": character_id,
                },
                {
                    "type": "audio",
                    "path": f"{character_id}.wav",
                    "role": role,
                    "audio_intent": "voice",
                    "library_character_id": character_id,
                },
            ])

        result = plan_h3_reference_sequence(
            prompt,
            model_type="minimax_h3_ref2va_pruned",
            resolution="896x512",
            total_frames=621,
            references=references,
            max_clip_frames=328,
            overlap_frames=18,
            native_continuation=True,
            camera_coverage="multi_shot",
        )
        self.assertEqual(result["window_count"], 2)
        prompts = result["window_prompts"]
        locked_lines = (
            "You could not live without Maestro. And where did it bring you? Back to me.",
            "Powerful, the creative tool has become. Perfectly balanced, Maestro is.",
            "as all things should be.",
            "hey guys! check out Maestro version two! it has so many...",
        )
        joined = "\n".join(prompts)
        for line in locked_lines:
            self.assertEqual(joined.count(line), 1)
        self.assertIn("<Subject 2> snaps his fingers", prompts[1])
        self.assertNotIn("<Subject 2> snaps his fingers", prompts[0])
        self.assertIn("<Subject 3> turns to dust", prompts[1])
        self.assertNotIn("Silent visual action, never spoken narration", prompts[1])
        self.assertNotIn("No words are spoken or mouthed", prompts[1])
        self.assertNotIn("only Thanos's mouth moves", prompts[0])
        self.assertNotIn("every other visible mouth stays closed", prompts[0])
        self.assertIn(
            "<Subject 2> (S1) says in the voice referenced from <Audio 2>",
            prompts[0],
        )
        self.assertNotIn("only on-camera speaker", joined)
        self.assertNotIn("hold Thanos's visible face", joined)
        self.assertNotIn("Atmospheric ambiance", prompts[0].split("overall_soundscape:", 1)[0])

        for window_prompt in prompts:
            aligned_prompt, aligned_refs, ordinal_maps = (
                align_ref2va_voice_reference_order(window_prompt, references)
            )
            self.assertEqual(ordinal_maps, {})
            self.assertEqual(
                [item["type"] for item in aligned_refs],
                ["image", "image", "image", "audio", "audio", "audio"],
            )
            repaired = ensure_ref2va_prompt_relationships(
                aligned_prompt,
                aligned_refs,
            )
            if "You could not live without Maestro" in repaired:
                self.assertRegex(
                    repaired,
                    r"<Subject 2>\s+\(S1\).*?<Audio 2>.*?<d>\[English\] "
                    r"You could not live without Maestro",
                )
            if "Powerful, the creative tool" in repaired:
                self.assertRegex(
                    repaired,
                    r"<Subject 1>\s+\(S2\).*?<Audio 1>.*?<d>\[English\] Powerful",
                )
            if "hey guys!" in repaired:
                self.assertRegex(
                    repaired,
                    r"<Subject 3>\s+\(S\d+\).*?<Audio 3>.*?<d>\[English\] hey guys!",
                )

    @patch("services.llm_service.generate")
    def test_two_window_three_character_plan_keeps_refs_and_dialogue_local(self, generate):
        prompt = (
            "Yoda is in Dagobah, a remote swamp-covered planet. "
            "Thanos is standing in the swamp and says, "
            "<d>Small green creature, tell me what you know of Maestro.</d> "
            "Yoda waves his hand slowly while saying, "
            "<d>Powerful, the creative tool has become.</d> "
            "Thanos responds <d>As all things should be.</d> "
            "Camera pans to Blaine who waves and says "
            "<d>Hey guys, I created Maestro version two.</d> "
            "Thanos, while standing near Yoda, snaps his fingers. "
            "Blaine turns to dust. Yoda grunts approvingly."
        )
        ledger = {
            "subject_continuity": (
                "Thanos, Yoda, and Blaine retain their requested identities, "
                "wardrobe, proportions, and stable speaking roles"
            ),
            "setting_continuity": "The same murky Dagobah swamp and jungle geography",
            "visual_continuity": "Grounded cinematic live-action realism",
            "editing_style": "Motivated dialogue coverage followed by a consequence cut",
            "initial_state": "Yoda searches the bag while Thanos stands nearby in the swamp",
            "ambient_audio": "Wetland insects, water, foliage, and synchronized movement",
            "music": "N/A",
            "required_final_outcome": "Blaine is gone and Yoda grunts approvingly",
            "beats": [
                {
                    "segment": 1,
                    "source_event_ids": ["E1"],
                    "dialogue_ids": [],
                    "state_after": "Yoda looks up from the bag toward Thanos",
                    "sound_effects": "Bag rustle and swamp ambience",
                },
                {
                    "segment": 1,
                    "source_event_ids": ["E2", "E3"],
                    "dialogue_ids": ["D1", "D2"],
                    "state_after": "Yoda lowers his hand while Thanos holds his gaze",
                    "sound_effects": "Subtle robe movement and wetland ambience",
                },
                {
                    "segment": 2,
                    "source_event_ids": ["E4", "E5"],
                    "dialogue_ids": ["D3", "D4"],
                    "state_after": "Blaine finishes speaking as Thanos raises the gauntlet",
                    "sound_effects": "A hand wave and quiet swamp movement",
                },
                {
                    "segment": 2,
                    "source_event_ids": ["E6", "E7", "E8"],
                    "dialogue_ids": [],
                    "state_after": "Blaine is gone; Yoda remains beside Thanos and grunts",
                    "sound_effects": "Finger snap, dust scattering, and Yoda's grunt",
                },
            ],
            "generated_dialogue": [],
        }

        def segment(number: int, duration: float) -> dict:
            return {
                "segment": number,
                "title": f"Dagobah beat {number}",
                "opening_state": "The supplied opening state",
                "coverage": "motivated multi-shot cinematic coverage",
                "pacing": "natural real-time pacing",
                "shots": [{
                    "shot": 1,
                    "start_seconds": 0.0,
                    "end_seconds": duration,
                    "transition": "opening composition",
                    "framing": "readable medium-wide shot",
                    "camera": "motivated pan and speaker coverage",
                    "action": f"The assigned Dagobah events for window {number} unfold once",
                    "sound_effects": "Natural synchronized swamp effects",
                }],
                "closing_state": "The supplied closing state",
            }

        references = []
        for role in ("Thanos", "Yoda", "Blaine"):
            key = role.casefold()
            references.extend([
                {
                    "type": "image",
                    "path": f"{key}.png",
                    "role": role,
                    "image_intent": "identity",
                    "library_character_id": key,
                },
                {
                    "type": "audio",
                    "path": f"{key}.wav",
                    "role": role,
                    "audio_intent": "voice",
                    "library_character_id": key,
                },
            ])
        clips = compute_h3_native_sequence_windows(
            614,
            window_frames=328,
            overlap_frames=18,
        )
        generate.side_effect = [
            json.dumps(ledger),
            json.dumps(segment(1, clips[0]["duration_seconds"])),
            json.dumps(segment(2, clips[1]["duration_seconds"])),
        ]

        result = plan_h3_reference_sequence(
            prompt,
            model_type="minimax_h3_ref2va_pruned",
            resolution="1280x704",
            total_frames=614,
            references=references,
            max_clip_frames=328,
            overlap_frames=18,
            native_continuation=True,
            camera_coverage="multi_shot",
        )

        self.assertEqual(result["planned_by"], "llm")
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(len(result["window_prompts"]), 2)
        first, second = result["window_prompts"]
        self.assertIn("Small green creature", first)
        self.assertIn("Powerful, the creative tool", first)
        self.assertNotIn("As all things should be", first)
        self.assertNotIn("Hey guys, I created Maestro", first)
        self.assertIn("As all things should be", second)
        self.assertIn("Hey guys, I created Maestro", second)
        self.assertNotIn("Small green creature", second)
        for compiled_prompt in result["window_prompts"]:
            for ordinal in range(1, 4):
                self.assertIn(f"<Subject {ordinal}>", compiled_prompt)
                self.assertIn(f"<Picture {ordinal}>", compiled_prompt)
                self.assertIn(f"<Audio {ordinal}>", compiled_prompt)
        self.assertNotIn("Thanos. Then Thanos", second)

    def test_compiler_deduplicates_semantically_identical_reference_contracts(self):
        clips, _ = compute_h3_sequence_clips(500)
        relationships = (
            "<Picture 1> defines Alex's identity. "
            "<Audio 1> supplies voice timbre for Alex Voice."
        )
        plan = {
            "subject_definitions": (
                "Stable speaking identity: S1 is Alex. "
                "<Picture 1> defines Alex's identity. "
                "<Audio 1> supplies voice timbre for Alex voice."
            ),
            "retention_analysis": "<Picture 1>: fully_preserved; <Audio 1>: reference",
            "setting_continuity": "A city street",
            "visual_style": "Cinematic realism",
            "ambient_audio": "Traffic",
            "music": "N/A",
            "clips": [
                {
                    "clip": index + 1,
                    "title": f"Clip {index + 1}",
                    "summary": f"Alex advances beat {index + 1}",
                    "opening_state": "Alex stands on the street",
                    "coverage": "cinematic coverage",
                    "pacing": "real-time pacing",
                    "shots": [{
                        "shot": 1,
                        "start_seconds": 0.0,
                        "end_seconds": clip["duration_seconds"],
                        "transition": "opening composition",
                        "framing": "medium shot",
                        "camera": "tracks Alex",
                        "action": f"Alex advances beat {index + 1}",
                        "dialogue": [],
                        "sound_effects": "Footsteps",
                    }],
                    "closing_state": f"Alex completes beat {index + 1}",
                }
                for index, clip in enumerate(clips)
            ],
        }
        compiled = compile_h3_reference_sequence_prompts(
            plan,
            clips,
            reference_relationships=relationships,
            default_retention="<Picture 1>: fully_preserved; <Audio 1>: reference",
            task_types="reference generation + audio reference",
        )
        subjects = compiled[0]["prompt"].split("\n\nsummary:", 1)[0]
        self.assertEqual(subjects.count("<Picture 1>"), 1)
        self.assertEqual(subjects.count("<Audio 1>"), 1)

    def test_compiler_uses_window_local_speaker_ids_with_stable_subject_refs(self):
        clips, _ = compute_h3_sequence_clips(500)
        relationships = (
            "<Subject 1> is Thanos, whose identity comes from <Picture 1>. "
            "<Audio 1> supplies voice timbre for <Subject 1>. "
            "<Subject 2> is Yoda, whose identity comes from <Picture 2>. "
            "<Audio 2> supplies voice timbre for <Subject 2>. "
            "<Subject 3> is Blaine, whose identity comes from <Picture 3>. "
            "<Audio 3> supplies voice timbre for <Subject 3>."
        )

        def dialogue(speaker: str, speaker_id: str, text: str) -> dict:
            return {
                "speaker": speaker,
                "speaker_id": speaker_id,
                "language": "English",
                "delivery": "speaks naturally",
                "action": "maintaining the visible performance",
                "text": text,
            }

        planned_clips = []
        for index, clip in enumerate(clips):
            lines = (
                [
                    dialogue("Thanos", "S1", "First line."),
                    dialogue("Yoda", "S2", "Second line."),
                    dialogue("Thanos", "S1", "Third line."),
                ]
                if index == 0 else
                [dialogue("Blaine", "S3", "Fourth line.")]
            )
            planned_clips.append({
                "clip": index + 1,
                "title": f"Clip {index + 1}",
                "summary": f"Story phase {index + 1}",
                "opening_state": "The same swamp scene continues",
                "coverage": "motivated speaker coverage",
                "pacing": "natural real-time pacing",
                "shots": [{
                    "shot": 1,
                    "start_seconds": 0.0,
                    "end_seconds": clip["duration_seconds"],
                    "transition": "opening composition",
                    "framing": "readable medium shot",
                    "camera": "the camera follows the active speaker",
                    "action": f"The requested phase {index + 1} unfolds",
                    "dialogue": lines,
                    "sound_effects": "Natural swamp ambience",
                }],
                "closing_state": f"Story phase {index + 1} is complete",
            })
        plan = {
            "subject_definitions": (
                "Stable speaking identities: S1 is Thanos; S2 is Yoda; S3 is Blaine. "
                "All three retain their requested appearance."
            ),
            "retention_analysis": "",
            "setting_continuity": "The same Dagobah swamp",
            "visual_style": "grounded cinematic live action",
            "ambient_audio": "Wetland insects and water",
            "music": "N/A",
            "clips": planned_clips,
        }
        compiled = compile_h3_reference_sequence_prompts(
            plan,
            clips,
            reference_relationships=relationships,
            default_retention=(
                "<Subject 1>: fully_preserved; <Audio 1>: reference; "
                "<Subject 2>: fully_preserved; <Audio 2>: reference; "
                "<Subject 3>: fully_preserved; <Audio 3>: reference"
            ),
            task_types="reference generation + audio reference",
        )

        first = compiled[0]["prompt"]
        second = compiled[1]["prompt"]
        self.assertIn("<Subject 1> (S1)", first)
        self.assertIn("<Subject 2> (S2)", first)
        self.assertIn("<Subject 3> (S1)", second)
        self.assertNotIn("<Subject 3> (S3)", second)
        self.assertNotIn("Stable speaking identities", first)
        self.assertNotIn("Stable speaking identities", second)
        for prompt in (first, second):
            for ordinal in range(1, 4):
                self.assertIn(f"<Subject {ordinal}>", prompt)
                self.assertIn(f"<Picture {ordinal}>", prompt)
                self.assertIn(f"<Audio {ordinal}>", prompt)

    @patch("services.llm_service.generate", side_effect=RuntimeError("offline"))
    def test_already_enhanced_omni_prompt_is_replanned_as_local_story_beats(self, _generate):
        expanded = (
            "subject_definitions: Stable speaking identities: S1 is Superman; S2 is Thanos. "
            "<Picture 1> defines Superman. <Picture 2> defines Thanos. "
            "<Audio 1> supplies Thanos's voice. <Picture 1> defines Superman. "
            "<Picture 2> defines Thanos. <Audio 1> supplies Thanos's voice.\n\n"
            "summary: [reference generation + audio reference] Superman is fighting Thanos "
            "Then supersonic tracking shot of Superman as he flies through a destroyed city "
            "Then cut to Superman flying through Thanos on the city street\n\n"
            "retention_analysis: <Picture 1>: fully_preserved; "
            "<Picture 2>: fully_preserved; <Audio 1>: reference\n\n"
            "detailed_description: The entire requested sequence is described here.\n\n"
            "overall_soundscape: Destruction.\n\n"
            "non_diegetic_music: N/A"
        )
        result = plan_h3_reference_sequence(
            expanded,
            model_type="minimax_h3_ref2va_full",
            resolution="864x480",
            total_frames=960,
            references=[
                {"type": "image", "path": "superman.png", "role": "Superman"},
                {"type": "image", "path": "thanos.png", "role": "Thanos"},
                {"type": "audio", "path": "thanos.wav", "role": "Thanos voice"},
            ],
            overlap_frames=18,
            native_continuation=True,
        )
        self.assertEqual(result["window_count"], 3)
        first, second, third = result["window_prompts"]
        self.assertIn("<Subject 1> is fighting <Subject 2>", first)
        self.assertNotIn("flies through a destroyed city", first)
        self.assertIn("flies through a destroyed city", second)
        self.assertNotIn("flying through Thanos", second)
        self.assertIn("flying through <Subject 2>", third)
        for prompt in result["window_prompts"]:
            subjects = prompt.split("\n\nsummary:", 1)[0]
            self.assertEqual(subjects.count("<Picture 1>"), 1)
            self.assertEqual(subjects.count("<Picture 2>"), 1)
            self.assertEqual(subjects.count("<Audio 1>"), 1)
            self.assertNotIn("concrete continuation state", prompt)

    @patch("services.llm_service.generate", side_effect=RuntimeError("offline"))
    def test_native_planner_records_continuation_geometry(self, _generate):
        result = plan_h3_reference_sequence(
            "Alex crosses town and rescues a driver",
            model_type="minimax_h3_ref2va",
            resolution="864x480",
            total_frames=960,
            references=[],
            overlap_frames=18,
            native_continuation=True,
        )
        self.assertTrue(result["native_continuation"])
        self.assertEqual(result["overlap_frames"], 18)
        self.assertEqual(result["window_count"], 3)
        self.assertEqual(result["per_clip_frames"], [345, 327, 288])

    @patch("services.llm_service.generate", side_effect=RuntimeError("offline"))
    def test_planner_can_enhance_before_omni_references_are_added(self, _generate):
        result = plan_h3_reference_sequence(
            "Alex crosses town and confronts the person following them",
            model_type="minimax_h3_ref2va",
            resolution="864x480",
            total_frames=720,
            references=[],
            camera_coverage="multi_shot",
        )
        self.assertEqual(result["plan_kind"], "reference_sequence")
        self.assertEqual(result["planned_by"], "deterministic_fallback")
        self.assertGreater(result["window_count"], 1)
        self.assertEqual(len(result["window_prompts"]), result["window_count"])

    def test_sequence_endpoint_uses_planning_not_generation_reference_rules(self):
        launch = (APP / "launch.py").read_text(encoding="utf-8")
        store = (ROOT / "ui" / "src" / "stores" / "useStore.ts").read_text(
            encoding="utf-8"
        )
        prompt_input = (
            ROOT / "ui" / "src" / "components" / "Sidebar" / "PromptInput.tsx"
        ).read_text(encoding="utf-8")
        self.assertIn("allow_empty=True", launch)
        self.assertIn("require_visual=False", launch)
        self.assertIn("promptEnhanceError", store)
        self.assertIn('role="alert"', prompt_input)

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
            expected_beat = ledger["beats"][index]["description"].replace(
                "Alex",
                "<Subject 1>",
            )
            self.assertIn(expected_beat, clip_prompt)
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
        self.assertIn(
            "{isH3 || isLtx ? 'Window Length' : 'Window Size'}",
            duration,
        )
        self.assertIn('Recommended {formatSeconds(safeWindowSeconds)}', duration)
        self.assertIn('native Omni windows', duration)
        self.assertIn('body["multi_prompts_gen_type"] = 0', launch)
        self.assertIn('body["sliding_window_overlap"] = h3_sequence_overlap', launch)


if __name__ == "__main__":
    unittest.main()
