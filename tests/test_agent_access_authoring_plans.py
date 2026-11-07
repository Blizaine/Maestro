"""CPU-only checks for native plan handoff to external authoring agents."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.agent_access.authoring_plans import (  # noqa: E402
    AuthoringPlanError,
    compute_studio_timeline,
    normalize_authored_director_plan,
    prepare_authored_generation,
    resolve_native_context,
)


def _context(workflow, model_type, model_def, params, **extra):
    context = {
        "context_id": "ctx-test-1",
        "workflow": workflow,
        "model_type": model_type,
        "model_def": model_def,
        "params": params,
        "brief": "A courier reaches the lighthouse before dawn.",
        "references": [{"id": "ref-main", "role": "image"}],
        "fps": model_def.get("fps", 24),
        **extra,
    }
    context["timeline"] = compute_studio_timeline(context)
    return context


def _window_plan(context, prompts):
    return {
        "context_id": context["context_id"],
        "windows": [
            {
                **geometry,
                "prompt": prompt,
            }
            for geometry, prompt in zip(context["timeline"]["windows"], prompts)
        ],
    }


def _music_analysis(duration=12.02, *, sections=None):
    beat_times = [
        0.0, 0.49, 1.03, 1.51, 2.02, 2.48, 3.04, 3.51, 3.66,
        4.11, 4.62, 5.14, 5.58, 6.09, 6.55, 7.02, 7.49, 7.95,
        8.44, 8.91, 9.43, 9.88, 10.37, 10.91, 11.39, 11.86,
    ]
    return {
        "duration": duration,
        "bpm": 120,
        "beats": [
            {"time": time, "strength": 0.9 if index % 4 == 0 else 0.25}
            for index, time in enumerate(beat_times)
        ],
        "downbeats": [0.0, 2.02, 3.51, 6.09, 8.44, 10.37],
        "sections": sections if sections is not None else [
            {"start": 0.0, "end": 3.57, "label": "intro", "energy": 0.24},
            {"start": 3.57, "end": 7.83, "label": "chorus", "energy": 0.92},
            {"start": 7.83, "end": duration, "label": "outro", "energy": 0.32},
        ],
        "lyrics": [
            {
                "start": 3.6,
                "end": 4.4,
                "speaker": "Singer A",
                "text": "The lights are coming back.",
            },
            {
                "start": 5.2,
                "end": 5.8,
                "speaker": "Singer A",
                "text": "Hold the line.",
            },
        ],
        "music_cues": [
            {"type": "instrument_change", "time": 3.8, "label": "synth enters"},
            {"type": "percussion_entry", "time": 8.1, "label": "drums enter"},
        ],
        "percussion_activity": [
            {"start": 3.5, "end": 7.4},
            {"start": 8.1, "end": duration},
        ],
        "beat_detection": {"status": "detected"},
        "vocal_timing": {"status": "transcribed"},
    }


def _beat_aware_music_context(analysis=None, *, energy_bias=0):
    return _context(
        "director.music_video",
        "minimax_h3_fused_turbo",
        {
            "architecture": "minimax_h3",
            "frames_minimum": 12,
            "frames_maximum": 204,
            "frames_steps": 12,
            "fps": 24,
            "director_shot_image_support": "optional",
        },
        {
            "source_duration_seconds": 12.02,
            "pipeline_type": "music_video",
            "shot_image_guidance": "prompt_only",
            "director_music_clip_seconds": 6.25,
            "energy_bias": energy_bias,
        },
        execution_profile={
            "fps": 24,
            "frames_minimum": 12,
            "frame_step": 12,
            "effective_max_frames": 204,
        },
        shot_image_policy="prompt_only",
        source_duration_seconds=12.02,
        music_analysis=_music_analysis() if analysis is None else analysis,
        music_analysis_source={"sha256": "a" * 64, "size_bytes": 12020},
    )


def _schedule_clips(context, windows, *, include_timeline_id=True):
    clips = []
    previous_id = None
    continuous = context["timeline"].get("continuity_mode") == "continuous"
    for index, window in enumerate(windows, start=1):
        clip_id = window["id"]
        clip = {
            "id": clip_id,
            "start_frame": window["start_frame"],
            "end_frame": window["end_frame"],
            "duration_frames": window["duration_frames"],
            "previous_clip_id": previous_id if continuous else None,
            "reference_ids": list(context["timeline"].get("reference_ids") or []),
            "video_prompt": f"Preserve the scheduled visual beat {index}.",
        }
        if include_timeline_id:
            clip["timeline_id"] = window["id"]
        clips.append(clip)
        previous_id = clip_id
    return clips


class StudioAuthoringPlanTests(unittest.TestCase):
    def test_native_resolver_enables_ltx_continuation_for_long_request(self):
        context = {
            "context_id": "ctx-ltx-long",
            "workflow": "video.frames",
            "model_type": "ltx2_22B_distilled_1_1",
            "model_def": {
                "multi_window_sequence_controls": True,
                "frames_maximum": 300,
                "fps": 25,
            },
            "params": {
                "video_length": 650,
                "sliding_window_size": 300,
                "sliding_window_overlap": 20,
                "sliding_window_discard_last_frames": 8,
            },
            "references": [],
        }
        resolve_native_context(context, {}, {})
        context["timeline"] = compute_studio_timeline(context)
        self.assertTrue(context["params"]["ltx_multi_window"])
        self.assertEqual(context["timeline"]["window_count"], 3)
        self.assertEqual(context["timeline"]["total_frames"], 650)

    def test_native_resolver_rejects_long_single_pass_without_continuation(self):
        context = {
            "context_id": "ctx-generic-too-long",
            "workflow": "video.frames",
            "model_type": "fixture-video",
            "model_def": {"frames_maximum": 120, "fps": 24},
            "params": {"video_length": 300, "sliding_window_size": 100},
            "references": [],
        }
        with self.assertRaisesRegex(AuthoringPlanError, "no supported external window plan"):
            resolve_native_context(context, {}, {})

    def test_director_resolver_stores_native_execution_and_image_profiles(self):
        context = {
            "context_id": "ctx-director-resolve",
            "workflow": "director.story",
            "model_type": "fixture-video",
            "model_def": {"fps": 24},
            "params": {"video_model": "fixture-video", "video_params": {}},
            "references": [{"id": "visual-1", "role": "image"}],
        }
        expected_profile = {
            "fps": 24,
            "frames_minimum": 12,
            "frame_step": 4,
            "effective_max_frames": 120,
        }
        with (
            patch(
                "services.director_video_strategy.build_director_video_execution_profile",
                return_value=expected_profile,
            ),
            patch(
                "services.director_video_strategy.resolve_shot_image_policy",
                return_value="prompt_only",
            ),
        ):
            resolve_native_context(context, {}, {"gpu_memory_gb": 8}, lora_system_hint="trusted hint")
        self.assertEqual(context["execution_profile"], expected_profile)
        self.assertEqual(context["shot_image_policy"], "prompt_only")
        self.assertEqual(context["lora_system_hint"], "trusted hint")

    def test_h3_windows_are_exact_and_server_signed_without_local_replanning(self):
        context = _context(
            "video.frames",
            "minimax_h3_fused_turbo",
            {"architecture": "minimax_h3", "frames_maximum": 124, "fps": 24},
            {
                "video_length": 250,
                "sliding_window_size": 124,
                "sliding_window_overlap": 18,
                "sliding_window_discard_last_frames": 0,
                "minimax_h3_multi_window": True,
                "minimax_h3_sequence_prompt_mode": "auto",
                "resolution": "864x480",
            },
        )
        self.assertEqual(context["timeline"]["kind"], "h3_sliding_windows")
        self.assertEqual(context["timeline"]["window_count"], 3)

        prompts = ["Open at the pier.", "Follow the courier through the fog.", "Reach the lighthouse." ]
        result = prepare_authored_generation(context, _window_plan(context, prompts))
        native = result["native_params"]
        self.assertEqual(native["h3_window_prompts"], prompts)
        self.assertEqual(native["multi_prompts_gen_type"], 2)
        self.assertEqual(native["h3_window_plan_signature"], result["plan_metadata"]["h3_window_plan_signature"])
        self.assertEqual(native["h3_window_plan"]["windows"][1]["prompt"], prompts[1])
        self.assertEqual(native["_h3_original_prompt"], context["brief"])

    def test_h3_rejects_stale_geometry_references_and_broken_continuity(self):
        context = _context(
            "video.frames",
            "minimax_h3_fused_turbo",
            {"architecture": "minimax_h3", "frames_maximum": 124, "fps": 24},
            {
                "video_length": 250,
                "sliding_window_size": 124,
                "sliding_window_overlap": 18,
                "minimax_h3_multi_window": True,
                "resolution": "864x480",
            },
        )
        prompts = ["Beat one.", "Beat two.", "Beat three."]

        wrong_id = _window_plan(context, prompts)
        wrong_id["context_id"] = "expired-context"
        with self.assertRaisesRegex(AuthoringPlanError, "different creation context"):
            prepare_authored_generation(context, wrong_id)

        wrong_geometry = _window_plan(context, prompts)
        wrong_geometry["windows"][1]["end_frame"] += 1
        with self.assertRaisesRegex(AuthoringPlanError, "current timeline value"):
            prepare_authored_generation(context, wrong_geometry)

        wrong_reference = _window_plan(context, prompts)
        wrong_reference["windows"][0]["reference_ids"] = ["other-reference"]
        with self.assertRaisesRegex(AuthoringPlanError, "does not match the current context references"):
            prepare_authored_generation(context, wrong_reference)

        broken_chain = _window_plan(context, prompts)
        broken_chain["windows"][2]["previous_window_id"] = None
        with self.assertRaisesRegex(AuthoringPlanError, "previous_window_id"):
            prepare_authored_generation(context, broken_chain)

    def test_ltx_authored_windows_use_native_manual_fields(self):
        context = _context(
            "video.frames",
            "ltx2_22B_distilled_1_1",
            {"multi_window_sequence_controls": True, "frames_maximum": 300, "fps": 25},
            {
                "video_length": 650,
                "sliding_window_size": 300,
                "sliding_window_overlap": 20,
                "sliding_window_discard_last_frames": 8,
                "ltx_multi_window": True,
                "ltx_window_prompt_mode": "auto",
            },
        )
        self.assertEqual(context["timeline"]["window_count"], 3)
        self.assertEqual(
            [(w["start_frame"], w["end_frame"]) for w in context["timeline"]["windows"]],
            [(0, 300), (272, 572), (544, 650)],
        )
        prompts = ["Establish the runway.", "Track the aircraft above the coast.", "End over the harbor."]
        result = prepare_authored_generation(context, _window_plan(context, prompts))
        self.assertEqual(result["native_params"]["ltx_window_prompt_mode"], "manual")
        self.assertEqual(result["native_params"]["ltx_window_prompts"], prompts)
        self.assertEqual(result["native_params"]["prompt"], "\n".join(prompts))
        self.assertEqual(result["native_params"]["multi_prompts_gen_type"], 1)

    def test_h3_reference_sequence_uses_manual_native_sequence_fields(self):
        context = _context(
            "video.references",
            "minimax_h3_omni",
            {
                "architecture": "minimax_h3",
                "omni_reference": True,
                "frames_minimum": 124,
                "frames_maximum": 345,
                "frames_steps": 17,
                "fps": 24,
            },
            {
                "video_length": 800,
                "sliding_window_size": 345,
                "sliding_window_overlap": 18,
                "minimax_h3_reference_sequence": True,
                "minimax_h3_sequence_continuity": False,
                "resolution": "864x480",
                "minimax_h3_references": [{"type": "image", "path": "fixture.png"}],
            },
        )
        self.assertEqual(context["timeline"]["kind"], "h3_hard_cut_sequence")
        prompts = [f"Independent cut {i}." for i in range(context["timeline"]["window_count"])]
        # This fixture covers CPU-side native sequence packing. Ref2VA
        # conditioning diagnostics are exercised by their separate tests and
        # import the model environment, which is intentionally absent here.
        with patch(
            "services.agent_access.authoring_plans._h3_reference_handoff_metadata",
            return_value=None,
        ):
            result = prepare_authored_generation(context, _window_plan(context, prompts))
        native = result["native_params"]
        self.assertEqual(native["minimax_h3_sequence_prompt_mode"], "manual")
        self.assertFalse(native["minimax_h3_sequence_continuity"])
        self.assertEqual(native["prompt"], "\n".join(prompts))
        self.assertEqual(native["h3_window_plan"]["planned_by"], "manual")
        self.assertTrue(native["h3_window_plan_signature"])

    def test_single_prompt_workflows_keep_literal_prompt_and_music_style(self):
        context = _context(
            "audio.music",
            "fixture-music",
            {"fps": 24},
            {},
        )
        result = prepare_authored_generation(
            context,
            {"context_id": context["context_id"], "prompt": "Keep these exact lyrics: blue / blue.", "style": "Dusty piano and brushed drums"},
        )
        self.assertEqual(result["native_params"], {
            "prompt": "Keep these exact lyrics: blue / blue.",
            "alt_prompt": "Dusty piano and brushed drums",
        })
        self.assertEqual(result["plan_metadata"]["window_count"], 1)


class DirectorAuthoringPlanTests(unittest.TestCase):
    def setUp(self):
        self.context = _context(
            "director.story",
            "fixture-video",
            {
                "frames_minimum": 12,
                "frames_maximum": 120,
                "frames_steps": 4,
                "fps": 24,
                "director_shot_image_support": "optional",
            },
            {"source_duration_seconds": 5, "shot_image_guidance": "prompt_only"},
            execution_profile={
                "fps": 24,
                "frames_minimum": 12,
                "frame_step": 4,
                "effective_max_frames": 120,
            },
            shot_image_policy="prompt_only",
        )

    def _plan(self):
        return {
            "context_id": self.context["context_id"],
            "clips": [
                {
                    "id": "scene-1",
                    "start_frame": 0,
                    "end_frame": 60,
                    "duration_frames": 60,
                    "previous_clip_id": None,
                    "reference_ids": ["ref-main"],
                    "video_prompt": "The courier crosses the open square.",
                },
                {
                    "id": "scene-2",
                    "start_frame": 60,
                    "end_frame": 120,
                    "duration_frames": 60,
                    "previous_clip_id": "scene-1",
                    "reference_ids": ["ref-main"],
                    "video_prompt": "The courier reaches the lighthouse door.",
                },
            ],
        }

    def test_director_normalizer_preserves_source_ids_and_native_frames(self):
        result = normalize_authored_director_plan(self.context, self._plan())
        plans = result["native_params"]["prepared_clip_plans"]
        timeline = result["native_params"]["prepared_planned_clips"]
        self.assertEqual([row["agent_source_clip_id"] for row in plans], ["scene-1", "scene-2"])
        self.assertEqual([row["duration_frames"] for row in timeline], [60, 60])
        self.assertEqual(timeline[1]["start"], 2.5)
        self.assertEqual(result["plan_metadata"]["native_total_frames"], 120)

    def test_director_rejects_bad_frame_lattice_missing_prompt_and_stale_refs(self):
        bad_frames = self._plan()
        bad_frames["clips"][0]["duration_frames"] = 62
        bad_frames["clips"][0]["end_frame"] = 62
        with self.assertRaisesRegex(AuthoringPlanError, "model accepts"):
            normalize_authored_director_plan(self.context, bad_frames)

        missing_prompt = self._plan()
        missing_prompt["clips"][0]["video_prompt"] = ""
        with self.assertRaisesRegex(AuthoringPlanError, "video_prompt is required"):
            normalize_authored_director_plan(self.context, missing_prompt)

        stale_reference = self._plan()
        stale_reference["clips"][1]["reference_ids"] = ["old-ref"]
        with self.assertRaisesRegex(AuthoringPlanError, "does not match the current context references"):
            normalize_authored_director_plan(self.context, stale_reference)

    def test_director_rejects_unaligned_window_geometry_and_missing_required_image_prompt(self):
        bad_window = self._plan()
        bad_window["clips"][0]["window_prompts"] = [{
            "id": "scene-1-window-1",
            "start_frame": 1,
            "end_frame": 60,
            "prompt": "The courier crosses the square.",
            "reference_ids": ["ref-main"],
            "previous_window_id": None,
        }]
        with self.assertRaisesRegex(AuthoringPlanError, "native frames 0–60"):
            normalize_authored_director_plan(self.context, bad_window)

        required_images = dict(self.context)
        required_images["model_def"] = {
            **self.context["model_def"],
            "director_shot_image_support": "required",
        }
        required_images.pop("timeline")
        required_images["shot_image_policy"] = "generate"
        required_images["timeline"] = compute_studio_timeline(required_images)
        image_schema = required_images["timeline"]["plan_schema"]
        self.assertTrue(image_schema["image_prompt_required"])
        self.assertIn("image_prompt", image_schema["required_clip_fields"])
        with self.assertRaisesRegex(AuthoringPlanError, "image_prompt is required"):
            normalize_authored_director_plan(required_images, self._plan())

    def test_story_schema_keeps_valid_window_prompt_support(self):
        schema = self.context["timeline"]["plan_schema"]
        self.assertTrue(schema["window_prompts_allowed"])
        self.assertIn("window_prompts", schema["clip_fields"])
        self.assertIn("prompt", schema["window_prompt_fields"])

        plan = self._plan()
        for clip in plan["clips"]:
            clip["window_prompts"] = [{
                "id": f"{clip['id']}-window-1",
                "start_frame": clip["start_frame"],
                "end_frame": clip["end_frame"],
                "prompt": f"Keep the full authored shot {clip['id']} coherent.",
                "reference_ids": ["ref-main"],
                "previous_window_id": None,
            }]
        result = normalize_authored_director_plan(self.context, plan)
        self.assertTrue(all(
            item["window_prompts"]
            for item in result["native_params"]["prepared_clip_plans"]
        ))

    def test_h3_story_accepts_only_exact_unique_window_id_as_legacy_timeline_alias(self):
        context = _context(
            "director.story",
            "minimax_h3_fused_turbo",
            {
                "architecture": "minimax_h3",
                "frames_minimum": 124,
                "frames_maximum": 243,
                "frames_steps": 17,
                "fps": 24,
                "director_shot_image_support": "optional",
            },
            {"duration_seconds": 30, "pipeline_type": "story", "shot_image_guidance": "prompt_only"},
            execution_profile={
                "fps": 24,
                "frames_minimum": 124,
                "frame_step": 17,
                "effective_max_frames": 243,
            },
            shot_image_policy="prompt_only",
        )
        schema = context["timeline"]["plan_schema"]
        self.assertIn("timeline_id", schema["clip_fields"])
        self.assertIn("timeline_id", schema["required_clip_fields"])

        windows = context["timeline"]["director_timeline"]["windows"]
        clips = _schedule_clips(context, windows, include_timeline_id=False)
        result = normalize_authored_director_plan(
            context, {"context_id": context["context_id"], "clips": clips}
        )
        prepared = result["native_params"]["prepared_clip_plans"]
        self.assertEqual(
            [item["agent_timeline_id"] for item in prepared],
            [item["id"] for item in windows],
        )
        self.assertEqual(
            [item["agent_source_clip_id"] for item in prepared],
            [item["id"] for item in windows],
        )

    def test_music_video_schema_and_legacy_alias_preserve_exact_schedule(self):
        context = _beat_aware_music_context()
        schema = context["timeline"]["plan_schema"]
        windows = context["timeline"]["windows"]
        self.assertIn("timeline_id", schema["clip_fields"])
        self.assertIn("timeline_id", schema["required_clip_fields"])
        self.assertTrue(schema["timeline_id_required"])
        self.assertFalse(schema["image_prompt_required"])
        self.assertNotIn("image_prompt", schema["required_clip_fields"])
        self.assertEqual(schema["required_clip_fields"], [
            "id", "timeline_id", "start_frame", "end_frame", "duration_frames",
            "reference_ids", "video_prompt",
        ])
        self.assertFalse(schema["window_prompts_allowed"])
        self.assertEqual(schema["window_prompt_fields"], [])
        self.assertNotIn("window_prompts", schema["clip_fields"])

        clips = _schedule_clips(context, windows, include_timeline_id=False)
        normalized = normalize_authored_director_plan(
            context, {"context_id": context["context_id"], "clips": clips}
        )
        prepared = normalized["native_params"]["prepared_clip_plans"]
        planned = normalized["native_params"]["prepared_planned_clips"]
        self.assertEqual(
            [item["agent_timeline_id"] for item in prepared],
            [item["id"] for item in windows],
        )
        self.assertEqual(
            [item["agent_source_clip_id"] for item in prepared],
            [item["id"] for item in windows],
        )
        self.assertEqual(
            [item["agent_previous_clip_id"] for item in planned],
            [clip["previous_clip_id"] for clip in clips],
        )
        self.assertEqual(
            [item["agent_reference_ids"] for item in prepared],
            [clip["reference_ids"] for clip in clips],
        )
        self.assertEqual(
            [item["output_frames"] for item in planned],
            [window["output_frames"] for window in windows],
        )

        unknown = copy.deepcopy(clips)
        unknown[0]["id"] = "agent-invented-shot-id"
        with self.assertRaisesRegex(AuthoringPlanError, "exactly matches one unique timeline window ID"):
            normalize_authored_director_plan(
                context,
                {"context_id": context["context_id"], "clips": unknown},
            )

        explicit_conflict = copy.deepcopy(clips)
        explicit_conflict[0]["timeline_id"] = windows[1]["id"]
        with self.assertRaisesRegex(AuthoringPlanError, "from the current soundtrack schedule"):
            normalize_authored_director_plan(
                context,
                {"context_id": context["context_id"], "clips": explicit_conflict},
            )

        duplicate = copy.deepcopy(clips)
        duplicate[1]["id"] = duplicate[0]["id"]
        with self.assertRaisesRegex(AuthoringPlanError, "id is duplicated"):
            normalize_authored_director_plan(
                context,
                {"context_id": context["context_id"], "clips": duplicate},
            )

        empty_explicit = copy.deepcopy(clips)
        empty_explicit[0]["timeline_id"] = ""
        with self.assertRaisesRegex(AuthoringPlanError, "timeline_id is required"):
            normalize_authored_director_plan(
                context,
                {"context_id": context["context_id"], "clips": empty_explicit},
            )

        duplicate_schedule = copy.deepcopy(context)
        duplicate_schedule["timeline"]["director_timeline"]["windows"][1]["id"] = windows[0]["id"]
        with self.assertRaisesRegex(AuthoringPlanError, "duplicate timeline window ID"):
            normalize_authored_director_plan(
                duplicate_schedule,
                {"context_id": context["context_id"], "clips": clips},
            )

        supplied_window_prompts = copy.deepcopy(clips)
        supplied_window_prompts[0]["window_prompts"] = []
        with self.assertRaisesRegex(AuthoringPlanError, "editorial music-video segment"):
            normalize_authored_director_plan(
                context,
                {"context_id": context["context_id"], "clips": supplied_window_prompts},
            )

    def test_h3_director_30_and_60_second_requests_use_native_schedule(self):
        for seconds in (30, 60):
            with self.subTest(seconds=seconds):
                context = _context(
                    "director.story",
                    "minimax_h3_fused_turbo",
                    {
                        "architecture": "minimax_h3",
                        "frames_minimum": 124,
                        "frames_maximum": 243,
                        "frames_steps": 17,
                        "fps": 24,
                        "director_shot_image_support": "optional",
                    },
                    {"duration_seconds": seconds, "pipeline_type": "story", "shot_image_guidance": "prompt_only"},
                    execution_profile={
                        "fps": 24,
                        "frames_minimum": 124,
                        "frame_step": 17,
                        "effective_max_frames": 243,
                    },
                    shot_image_policy="prompt_only",
                )
                timeline = context["timeline"]["director_timeline"]
                target_frames = seconds * 24
                self.assertEqual(timeline["kind"], "h3_native_story_schedule")
                self.assertEqual(timeline["requested_total_frames"], target_frames)
                self.assertGreaterEqual(timeline["native_total_frames"], target_frames)
                self.assertEqual(
                    timeline["padding_frames"],
                    timeline["native_total_frames"] - target_frames,
                )
                self.assertEqual(timeline["final_trim_frames"], 0)

                clips = []
                prior_source_id = None
                for index, geometry in enumerate(timeline["windows"], start=1):
                    source_id = f"screenplay-{seconds}s-scene-{index}"
                    clips.append({
                        "id": source_id,
                        "timeline_id": geometry["id"],
                        "start_frame": geometry["start_frame"],
                        "end_frame": geometry["end_frame"],
                        "duration_frames": geometry["duration_frames"],
                        "previous_clip_id": prior_source_id,
                        "reference_ids": ["ref-main"],
                        "video_prompt": f"Screenplay beat {index}, preserved as written.",
                    })
                    prior_source_id = source_id
                result = normalize_authored_director_plan(
                    context,
                    {"context_id": context["context_id"], "clips": clips},
                )
                native = result["native_params"]
                self.assertEqual(
                    [row["agent_source_clip_id"] for row in native["prepared_clip_plans"]],
                    [row["id"] for row in clips],
                )
                self.assertEqual(
                    [row["agent_timeline_id"] for row in native["prepared_clip_plans"]],
                    [row["timeline_id"] for row in clips],
                )
                self.assertEqual(
                    sum(row["duration_frames"] for row in native["prepared_planned_clips"]),
                    timeline["native_total_frames"],
                )

    def test_music_video_keeps_exact_30_and_60_second_source_timing(self):
        for seconds in (30, 60):
            with self.subTest(seconds=seconds):
                context = _context(
                    "director.music_video",
                    "minimax_h3_fused_turbo",
                    {
                        "architecture": "minimax_h3",
                        "frames_minimum": 124,
                        "frames_maximum": 243,
                        "frames_steps": 17,
                        "fps": 24,
                        "director_shot_image_support": "optional",
                    },
                    {
                        "source_duration_seconds": seconds,
                        "pipeline_type": "music_video",
                        "shot_image_guidance": "prompt_only",
                    },
                    execution_profile={
                        "fps": 24,
                        "frames_minimum": 124,
                        "frame_step": 17,
                        "effective_max_frames": 243,
                    },
                    shot_image_policy="prompt_only",
                    source_duration_seconds=seconds,
                )
                timeline = context["timeline"]["director_timeline"]
                self.assertEqual(timeline["kind"], "exact_source_audio")
                target_frames = seconds * 24
                self.assertEqual(timeline["requested_total_frames"], target_frames)
                windows = context["timeline"]["windows"]
                self.assertTrue(windows)
                self.assertEqual(sum(row["output_frames"] for row in windows), target_frames)
                self.assertEqual(
                    sum(row["native_duration_frames"] for row in windows),
                    timeline["native_total_frames"],
                )
                self.assertGreaterEqual(timeline["native_total_frames"], target_frames)
                self.assertEqual(
                    timeline["final_trim_frames"],
                    timeline["native_total_frames"] - target_frames,
                )
                clips = []
                previous_id = None
                for index, window in enumerate(windows, start=1):
                    clips.append({
                        "id": f"soundtrack-{seconds}s-scene-{index}",
                        "timeline_id": window["id"],
                        "start_frame": window["start_frame"],
                        "end_frame": window["end_frame"],
                        "duration_frames": window["duration_frames"],
                        "previous_clip_id": previous_id,
                        "reference_ids": ["ref-main"],
                        "video_prompt": f"A dancer moves through scene {index} in soundtrack order.",
                    })
                    previous_id = clips[-1]["id"]
                result = normalize_authored_director_plan(context, {
                    "context_id": context["context_id"],
                    "clips": clips,
                })
                stale_id = {
                    "context_id": context["context_id"],
                    "clips": [dict(row) for row in clips],
                }
                stale_id["clips"][0]["timeline_id"] = "old-soundtrack-window"
                with self.assertRaisesRegex(AuthoringPlanError, "from the current soundtrack schedule"):
                    normalize_authored_director_plan(context, stale_id)
                stale_bounds = {
                    "context_id": context["context_id"],
                    "clips": [dict(row) for row in clips],
                }
                stale_bounds["clips"][0]["end_frame"] += 1
                with self.assertRaisesRegex(AuthoringPlanError, "current soundtrack schedule value"):
                    normalize_authored_director_plan(context, stale_bounds)
                native = result["native_params"]
                output_frames = sum(
                    row["output_frames"] for row in native["prepared_planned_clips"]
                )
                generated_frames = sum(
                    row["duration_frames"] for row in native["prepared_planned_clips"]
                )
                self.assertEqual(output_frames, target_frames)
                self.assertGreaterEqual(generated_frames, target_frames)
                self.assertEqual(
                    generated_frames - output_frames,
                    result["plan_metadata"]["final_trim_frames"],
                )
                self.assertEqual(
                    {row["agent_source_clip_id"] for row in native["prepared_clip_plans"]},
                    {row["id"] for row in clips},
                )
                self.assertTrue(all(row["music_timing_version"] == 1 for row in native["prepared_planned_clips"]))
                self.assertEqual(
                    [row["agent_timeline_id"] for row in native["prepared_clip_plans"]],
                    [row["timeline_id"] for row in clips],
                )

    def test_music_video_uses_beats_sections_and_resolved_clip_cap(self):
        context = _beat_aware_music_context()
        timeline = context["timeline"]["director_timeline"]
        windows = context["timeline"]["windows"]
        target_frames = round(12.02 * 24)

        self.assertEqual(timeline["timing_basis"], "beat_and_section_analysis")
        self.assertEqual(timeline["analysis_duration_seconds"], 12.02)
        self.assertEqual(timeline["requested_total_frames"], target_frames)
        self.assertEqual(
            [row["end_frame"] for row in windows[:-1]],
            [86, 188],
            "section boundaries should define cuts on the source frame grid",
        )
        self.assertTrue(all(row["timing_basis"] == "beat_and_section_analysis" for row in windows))
        self.assertTrue(all(row["native_duration_frames"] <= 144 for row in windows))
        self.assertEqual(sum(row["output_frames"] for row in windows), target_frames)
        self.assertTrue(
            any(row["native_duration_frames"] > row["output_frames"] for row in windows),
            "fractional section cuts should pad to a native frame lattice and trim back to source frames",
        )
        self.assertEqual(windows[0]["section_label"], "intro")
        self.assertEqual(windows[1]["section_label"], "chorus")
        self.assertIn("opening_beat_times", windows[0])

        no_sections = _music_analysis(sections=[])
        no_section_context = _beat_aware_music_context(no_sections)
        no_section_cuts = [
            row["end_frame"] for row in no_section_context["timeline"]["windows"][:-1]
        ]
        self.assertNotEqual(no_section_cuts, [row["end_frame"] for row in windows[:-1]])

        slower_context = _beat_aware_music_context(energy_bias=-2)
        slower_cuts = [
            row["end_frame"] for row in slower_context["timeline"]["windows"][:-1]
        ]
        self.assertNotEqual(slower_cuts, [row["end_frame"] for row in windows[:-1]])

        clips = []
        previous_id = None
        for index, window in enumerate(windows, start=1):
            source_id = f"analysis-shot-{index}"
            clips.append({
                "id": source_id,
                "timeline_id": window["id"],
                "start_frame": window["start_frame"],
                "end_frame": window["end_frame"],
                "duration_frames": window["duration_frames"],
                "previous_clip_id": previous_id,
                "reference_ids": ["ref-main"],
                "video_prompt": f"A distinct visual scene for soundtrack section {index}.",
            })
            previous_id = source_id
        authored = {"context_id": context["context_id"], "clips": clips}
        normalized = normalize_authored_director_plan(context, authored)
        prepared = normalized["native_params"]["prepared_planned_clips"]
        self.assertEqual(normalized["plan_metadata"]["timing_basis"], "beat_and_section_analysis")
        self.assertEqual(sum(row["output_frames"] for row in prepared), target_frames)
        self.assertEqual(
            sum(row["duration_frames"] for row in prepared),
            timeline["native_total_frames"],
        )
        self.assertTrue(all(row["duration_frames"] <= 144 for row in prepared))
        self.assertTrue(all(row["timing_basis"] == "beat_and_section_analysis" for row in prepared))
        self.assertTrue(
            any(
                cue.get("label") == "drums enter"
                for row in prepared
                for cue in row.get("music_cues", [])
            )
        )
        self.assertTrue(
            any(
                line.get("text") == "The lights are coming back."
                and row.get("dominant_speaker") == "Singer A"
                for row in prepared
                for line in row.get("lyrics", [])
            )
        )

        stale_bounds = copy.deepcopy(authored)
        stale_bounds["clips"][0]["end_frame"] += 1
        with self.assertRaisesRegex(AuthoringPlanError, "current soundtrack schedule value"):
            normalize_authored_director_plan(context, stale_bounds)
        stale_id = copy.deepcopy(authored)
        stale_id["clips"][0]["timeline_id"] = "old-analysis-window"
        with self.assertRaisesRegex(AuthoringPlanError, "current soundtrack schedule"):
            normalize_authored_director_plan(context, stale_id)

    def test_invalid_supplied_music_analysis_does_not_use_legacy_timing(self):
        invalid = _music_analysis()
        invalid["beats"] = []
        with self.assertRaisesRegex(AuthoringPlanError, "beats is empty"):
            _beat_aware_music_context(invalid)

        context = _beat_aware_music_context()
        context["music_analysis"] = None
        with self.assertRaisesRegex(AuthoringPlanError, "context.music_analysis must be an object"):
            compute_studio_timeline(context)

    def test_music_video_accepts_explicit_no_reliable_beats_without_fabricating_readback(self):
        analysis = _music_analysis()
        analysis["beats"] = []
        analysis["bpm"] = 0
        analysis["beat_detection"] = "no_reliable_beats"
        analysis["vocal_timing"] = "not_analyzed"
        analysis.pop("lyrics")
        context = _beat_aware_music_context(analysis)
        timeline = context["timeline"]["director_timeline"]
        windows = context["timeline"]["windows"]

        self.assertEqual(timeline["timing_basis"], "section_and_duration_analysis")
        self.assertEqual(timeline["bpm"], 0)
        self.assertEqual(timeline["beat_detection"], "no_reliable_beats")
        self.assertEqual(timeline["vocal_timing"], "not_analyzed")
        self.assertTrue(windows)
        self.assertTrue(all(row["timing_basis"] == "section_and_duration_analysis" for row in windows))
        self.assertTrue(all(row["beat_count"] == 0 for row in windows))
        self.assertTrue(all(row["opening_beat_times"] == [] for row in windows))
        self.assertTrue(all(row["ending_beat_times"] == [] for row in windows))
        self.assertTrue(all(row["section_label"] for row in windows))
        self.assertTrue(all(row["vocal_timing"] == "not_analyzed" for row in windows))

        clips = []
        previous_id = None
        for index, window in enumerate(windows, start=1):
            source_id = f"unmetered-shot-{index}"
            clips.append({
                "id": source_id,
                "timeline_id": window["id"],
                "start_frame": window["start_frame"],
                "end_frame": window["end_frame"],
                "duration_frames": window["duration_frames"],
                "previous_clip_id": previous_id,
                "reference_ids": ["ref-main"],
                "video_prompt": f"A distinct visual scene for section {index}.",
            })
            previous_id = source_id
        normalized = normalize_authored_director_plan(
            context, {"context_id": context["context_id"], "clips": clips}
        )
        prepared = normalized["native_params"]["prepared_planned_clips"]
        self.assertEqual(normalized["plan_metadata"]["timing_basis"], "section_and_duration_analysis")
        self.assertTrue(all(row["beat_count"] == 0 for row in prepared))
        self.assertTrue(all(row["opening_beat_times"] == [] for row in prepared))
        self.assertTrue(all(row["vocal_timing"] == "not_analyzed" for row in prepared))
        self.assertTrue(
            any(
                cue.get("label") == "synth enters"
                for row in prepared
                for cue in row.get("music_cues", [])
            )
        )

    def test_missing_music_analysis_fixture_is_explicitly_legacy(self):
        context = _context(
            "director.music_video",
            "minimax_h3_fused_turbo",
            {
                "architecture": "minimax_h3",
                "frames_minimum": 124,
                "frames_maximum": 243,
                "frames_steps": 17,
                "fps": 24,
            },
            {"source_duration_seconds": 30, "pipeline_type": "music_video"},
            execution_profile={
                "fps": 24,
                "frames_minimum": 124,
                "frame_step": 17,
                "effective_max_frames": 243,
            },
            source_duration_seconds=30,
        )
        self.assertEqual(
            context["timeline"]["director_timeline"]["timing_basis"],
            "legacy_duration_only",
        )


if __name__ == "__main__":
    unittest.main()
