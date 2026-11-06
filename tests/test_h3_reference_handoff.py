from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest


APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3.reference_handoff import (  # noqa: E402
    compile_ref2va_prompt_handoff,
    refresh_ref2va_handoff_traces,
)
from services.agent_access.authoring_plans import (  # noqa: E402
    compute_studio_timeline,
    prepare_authored_generation,
)


def _character(name: str, character_id: str) -> list[dict]:
    return [
        {
            "id": f"{character_id}-image",
            "type": "image",
            "path": f"C:\\private\\{character_id}.png",
            "role": name,
            "character_name": name,
            "library_character_id": character_id,
            "image_intent": "identity",
        },
        {
            "id": f"{character_id}-voice",
            "type": "audio",
            "path": f"C:\\private\\{character_id}.wav",
            "role": name,
            "character_name": name,
            "library_character_id": character_id,
            "audio_intent": "voice",
        },
    ]


def _two_window_context(*, continuity: bool = True) -> dict:
    refs = _character("Mira", "mira") + _character("Rae", "rae")
    params = {
        "video_length": 672,
        "sliding_window_size": 345,
        "sliding_window_overlap": 18,
        "minimax_h3_reference_sequence": True,
        "minimax_h3_sequence_continuity": continuity,
        "resolution": "864x480",
        "minimax_h3_references": refs,
    }
    context = {
        "context_id": "ctx-handoff-test",
        "workflow": "video.references",
        "model_type": "minimax_h3_ref2va_fused_turbo",
        "model_def": {
            "architecture": "minimax_h3_ref2va",
            "omni_reference": True,
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frames_steps": 17,
            "fps": 24,
        },
        "params": params,
        "brief": "Mira and Rae meet at dawn.",
        "references": [
            {"id": row["id"], "type": row["type"], "role": row["role"]}
            for row in refs
        ],
        "fps": 24,
    }
    context["timeline"] = compute_studio_timeline(context)
    return context


class Ref2VAReferenceHandoffTests(unittest.TestCase):
    def test_named_dialogue_keeps_subject_and_first_speaker_ordinals_distinct(self):
        references = _character("Mira", "mira") + _character("Rae", "rae")
        result = compile_ref2va_prompt_handoff(
            "Rae says: <d>We go first.</d> Mira answers: <d>Then we follow.</d>",
            references,
            duration_seconds=5,
        )

        speakers = {row["local_speaker"]: row for row in result["trace"]["speaker_bindings"]}
        self.assertEqual(speakers["S1"]["subject"], "Subject 2")
        self.assertEqual(speakers["S1"]["voice_reference_id"], "rae-voice")
        self.assertEqual(speakers["S2"]["subject"], "Subject 1")
        self.assertEqual(speakers["S2"]["voice_reference_id"], "mira-voice")
        self.assertEqual(result["voice_scope"]["kept_reference_ids"], ["mira-voice", "rae-voice"])
        self.assertEqual(
            result["trace"]["prompt_sha256"],
            hashlib.sha256(result["compiled_prompt"].encode("utf-8")).hexdigest(),
        )
        self.assertNotIn("C:\\private", json.dumps(result["trace"]))
        json.dumps(result["trace"])

    def test_continuation_offsets_picture_labels_and_scopes_each_window_voice(self):
        references = _character("Mira", "mira") + _character("Rae", "rae")
        first = compile_ref2va_prompt_handoff(
            "Mira speaks: <d>We start here.</d>",
            references,
            window_index=1,
        )
        second = compile_ref2va_prompt_handoff(
            "Rae speaks: <d>We continue there.</d>",
            references,
            continuation=True,
            window_index=2,
        )

        self.assertEqual(first["voice_scope"]["kept_reference_ids"], ["mira-voice"])
        self.assertEqual(first["voice_scope"]["omitted_reference_ids"], ["rae-voice"])
        self.assertEqual(second["voice_scope"]["kept_reference_ids"], ["rae-voice"])
        self.assertEqual(second["voice_scope"]["omitted_reference_ids"], ["mira-voice"])
        self.assertIn("<Picture 1> is the exact final frame", second["compiled_prompt"])
        self.assertIn("<Picture 2>", second["compiled_prompt"])
        self.assertEqual(second["trace"]["window_index"], 2)
        self.assertTrue(second["trace"]["continuation"])
        # The source portrait remains Picture 1 in the reference map; the
        # leading continuation boundary is a separate synthetic Picture 1.
        portrait = next(
            row for row in second["trace"]["reference_bindings"]
            if row["reference_id"] == "mira-image"
        )
        self.assertEqual(portrait["label"], "Picture 1")

    def test_video_audio_disabled_does_not_shift_voice_ordinals(self):
        references = [
            {
                "id": "silent-video",
                "type": "video",
                "path": "C:\\private\\silent.mp4",
                "role": "motion",
                "video_intent": "motion",
                "has_audio": True,
                "include_audio": False,
            },
            *_character("Mira", "mira"),
        ]
        result = compile_ref2va_prompt_handoff(
            "Mira says: <d>Only the voice reference is audible.</d>",
            references,
        )

        video = next(row for row in result["trace"]["reference_bindings"] if row["type"] == "video")
        voice = next(row for row in result["trace"]["reference_bindings"] if row["reference_id"] == "mira-voice")
        self.assertFalse(video["embedded_audio_enabled"])
        self.assertNotIn("audio_label", video)
        self.assertEqual(voice["audio_label"], "Audio 1")
        self.assertEqual(result["trace"]["speaker_bindings"][0]["voice_reference_id"], "mira-voice")

    def test_embedded_video_audio_gets_audio_label_before_voice_reference(self):
        references = [
            {
                "id": "soundtrack-video",
                "type": "video",
                "path": "C:\\private\\soundtrack.mp4",
                "role": "motion",
                "video_intent": "motion",
                "has_audio": True,
                "include_audio": True,
            },
            *_character("Mira", "mira"),
        ]
        result = compile_ref2va_prompt_handoff(
            "Mira says: <d>Keep the line clear.</d>",
            references,
        )

        video = next(row for row in result["trace"]["reference_bindings"] if row["type"] == "video")
        voice = next(row for row in result["trace"]["reference_bindings"] if row["reference_id"] == "mira-voice")
        self.assertTrue(video["embedded_audio_enabled"])
        self.assertEqual(video["audio_label"], "Audio 1")
        self.assertEqual(voice["audio_label"], "Audio 2")
        self.assertEqual(result["trace"]["speaker_bindings"][0]["voice_reference_id"], "mira-voice")

    def test_audio_only_skips_voice_pruning_and_context_ir_compilation(self):
        references = [
            {
                "id": "narration",
                "type": "audio",
                "path": "C:\\private\\narration.wav",
                "role": "narration",
                "audio_intent": "drive",
            },
        ]
        result = compile_ref2va_prompt_handoff(
            "Keep this audio-only direction literal.",
            references,
            audio_only=True,
        )
        self.assertEqual(result["compiled_prompt"], "Keep this audio-only direction literal.")
        self.assertFalse(result["voice_scope"]["applied"])
        self.assertEqual(result["voice_scope"]["reason"], "audio_only")

    def test_refresh_rebinds_trace_to_native_prepared_reference_manifest(self):
        prompt = "Mira says: <d>Follow the soundtrack.</d>"
        authored = compile_ref2va_prompt_handoff(
            prompt,
            _character("Mira", "mira"),
            duration_seconds=14.375,
            window_index=2,
        )["trace"]
        authored.update({
            "timeline_start_frame": 327,
            "timeline_end_frame": 654,
            "timeline_contribution_frames": 327,
            "rendered_pass_frames": 345,
            "rendered_pass_duration_seconds": 14.375,
            "duration_basis": "native_window_frames",
        })
        runtime_references = [
            _character("Mira", "mira")[0],
            {
                "id": "quiet-video",
                "type": "video",
                "path": "C:\\private\\quiet.mp4",
                "role": "motion",
                "video_intent": "motion",
                "has_audio": True,
                "include_audio": False,
            },
            {
                "id": "video-derived-mira-voice",
                "type": "audio",
                "path": "C:\\private\\derived.wav",
                "role": "Mira",
                "character_name": "Mira",
                "library_character_id": "mira",
                "audio_intent": "voice",
            },
        ]

        refreshed = refresh_ref2va_handoff_traces(
            [authored],
            [prompt],
            runtime_references,
        )[0]
        self.assertEqual(refreshed["reference_bindings"][1]["reference_index"], 1)
        self.assertEqual(refreshed["reference_bindings"][1]["reference_id"], "quiet-video")
        self.assertEqual(refreshed["reference_bindings"][2]["reference_id"], "video-derived-mira-voice")
        self.assertEqual(refreshed["speaker_bindings"][0]["voice_reference_id"], "video-derived-mira-voice")
        self.assertEqual(refreshed["timeline_contribution_frames"], 327)
        self.assertEqual(refreshed["rendered_pass_frames"], 345)
        self.assertEqual(refreshed["duration_seconds"], 14.375)

    def test_refresh_failure_remains_a_nonblocking_diagnostic(self):
        prompt = "<d>We will see.</d>"
        authored = compile_ref2va_prompt_handoff(
            prompt,
            _character("Mira", "mira"),
        )["trace"]
        refreshed = refresh_ref2va_handoff_traces(
            [authored],
            [prompt],
            _character("Mira", "mira") + _character("Rae", "rae"),
        )[0]
        self.assertEqual(refreshed["stage"], "diagnostic_unavailable")
        self.assertIsNone(refreshed["compiled_prompt"])
        self.assertTrue(refreshed["diagnostic_error"])

    def test_unbound_dialogue_is_reported_without_guessing_a_character(self):
        result = compile_ref2va_prompt_handoff(
            "<d>Run to the hill.</d>",
            _character("Mira", "mira"),
        )
        self.assertTrue(result["trace"]["dialogue_bindings"])
        binding = result["trace"]["dialogue_bindings"][0]
        self.assertIsNone(binding["subject"])
        self.assertEqual(binding["compiled_subject"], "Subject 1")
        self.assertIsNone(binding["character_id"])
        self.assertEqual(binding["binding_status"], "unknown")

    def test_plain_subject_dialogue_cue_binds_single_character_diagnostics(self):
        result = compile_ref2va_prompt_handoff(
            "<Subject 1> speaks: <d>Run to the hill.</d>",
            _character("Mira", "mira"),
        )

        binding = result["trace"]["dialogue_bindings"][0]
        self.assertEqual(binding["subject"], "Subject 1")
        self.assertEqual(binding["compiled_subject"], "Subject 1")
        self.assertEqual(binding["character_id"], "mira")
        self.assertEqual(binding["voice_reference_id"], "mira-voice")
        self.assertEqual(binding["binding_status"], "bound")

    def test_prepare_metadata_reports_native_rendered_frames_and_preserves_prompts(self):
        context = _two_window_context()
        prompts = [
            "Mira speaks: <d>We begin at the door.</d>",
            "Rae speaks: <d>We continue through the light.</d>",
        ]
        plan = {
            "context_id": context["context_id"],
            "windows": [
                {**geometry, "prompt": prompt}
                for geometry, prompt in zip(context["timeline"]["windows"], prompts)
            ],
        }
        result = prepare_authored_generation(context, plan)

        self.assertEqual(result["native_params"]["h3_window_prompts"], prompts)
        self.assertEqual(result["native_params"]["prompt"], "\n".join(prompts))
        traces = result["plan_metadata"]["h3_reference_handoffs"]
        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0]["rendered_pass_frames"], 345)
        self.assertEqual(traces[0]["timeline_contribution_frames"], 345)
        self.assertEqual(traces[1]["rendered_pass_frames"], 345)
        self.assertEqual(traces[1]["timeline_contribution_frames"], 327)
        self.assertFalse(traces[0]["continuation"])
        self.assertTrue(traces[1]["continuation"])

    def test_hard_cut_handoffs_use_per_clip_rendered_durations(self):
        context = _two_window_context(continuity=False)
        self.assertEqual(context["timeline"]["kind"], "h3_hard_cut_sequence")
        prompts = [
            "Mira speaks: <d>Cut one.</d>",
            "Rae speaks: <d>Cut two.</d>",
        ]
        plan = {
            "context_id": context["context_id"],
            "windows": [
                {**geometry, "prompt": prompt}
                for geometry, prompt in zip(context["timeline"]["windows"], prompts)
            ],
        }
        result = prepare_authored_generation(context, plan)
        traces = result["plan_metadata"]["h3_reference_handoffs"]

        self.assertEqual(len(traces), 2)
        for trace, geometry in zip(traces, context["timeline"]["windows"]):
            self.assertEqual(trace["duration_basis"], "per_clip_frames")
            self.assertEqual(trace["rendered_pass_frames"], geometry["frames"])
            self.assertFalse(trace["continuation"])
        self.assertNotIn("exact final frame", traces[1]["compiled_prompt"])

    def test_prepare_metadata_exposes_single_window_handoff(self):
        context = _two_window_context()
        # Some installed definitions expose the H3 family even when they do
        # not repeat the architecture string on the public model row.
        context["model_def"].pop("architecture")
        context["model_def"]["family"] = "minimax_h3_ref2va"
        context["params"].update(
            video_length=180,
            minimax_h3_reference_sequence=False,
            minimax_h3_sequence_continuity=False,
        )
        context["timeline"] = compute_studio_timeline(context)
        prompt = "Mira speaks: <d>A single prepared window.</d>"
        result = prepare_authored_generation(
            context,
            {"context_id": context["context_id"], "prompt": prompt},
        )
        self.assertEqual(result["native_params"]["prompt"], prompt)
        self.assertEqual(len(result["plan_metadata"]["h3_reference_handoffs"]), 1)
        self.assertEqual(
            result["plan_metadata"]["h3_reference_handoffs"][0]["window_index"],
            1,
        )

    def test_unavailable_preview_does_not_add_an_authoring_rejection(self):
        context = _two_window_context()
        prompts = ["<d>We begin.</d>", "<d>We continue.</d>"]
        plan = {
            "context_id": context["context_id"],
            "windows": [
                {**geometry, "prompt": prompt}
                for geometry, prompt in zip(context["timeline"]["windows"], prompts)
            ],
        }

        result = prepare_authored_generation(context, plan)

        self.assertEqual(result["native_params"]["h3_window_prompts"], prompts)
        traces = result["plan_metadata"]["h3_reference_handoffs"]
        self.assertTrue(all(trace["stage"] == "diagnostic_unavailable" for trace in traces))
        self.assertTrue(all(trace["compiled_prompt"] is None for trace in traces))
        self.assertTrue(all(trace["diagnostic_error"] for trace in traces))


if __name__ == "__main__":
    unittest.main()
