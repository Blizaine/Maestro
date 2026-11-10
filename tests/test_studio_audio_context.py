"""Transcript opt-ins preserve exact audio and window-local context."""
import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.studio_audio_context import (
    append_window_transcript, conditioning_transcript_windows, generic_transcript_windows,
    load_enhancement_transcript, source_audio_driver, strip_window_transcript,
    transcript_fingerprint, transcript_plan_context, transcript_window_contexts,
)
from services.h3_performance_audio import has_h3_performance_audio, performance_audio_guidance


def transcript():
    return {"version": 1, "segments": [
        {"start": 1, "end": 2, "text": "AURORA rises."},
        {"start": 11, "end": 12, "text": "MARSH glows."},
        {"start": 22, "end": 23, "text": "RAINBOW fades."},
    ]}


class StudioAudioContextTests(unittest.TestCase):
    def test_switches_are_independent_and_voice_style_samples_never_transcribe(self):
        with patch("services.source_audio_transcript.get_source_audio_transcript", return_value=transcript()) as asr:
            params = {"audio_guide": "source.wav", "audio_prompt_type": "A"}
            self.assertIsNone(load_enhancement_transcript(params))
            self.assertIsNone(load_enhancement_transcript({**params, "source_audio_transcript_in_prompt": True}))
            asr.assert_not_called()
            enabled = load_enhancement_transcript({**params, "source_audio_transcript_for_enhancement": True})
            self.assertEqual(enabled["segments"], transcript()["segments"])
            asr.assert_called_once_with("source.wav")
        for intent in ("voice", "style", "sound"):
            with self.subTest(intent=intent):
                self.assertIsNone(source_audio_driver({
                    "minimax_h3_references": [{"type": "audio", "path": "voice.wav", "audio_intent": intent}],
                    "audio_guide": "stale-frame-audio.wav", "audio_prompt_type": "A",
                }))
        with patch("services.source_audio_transcript.get_source_audio_transcript") as asr:
            self.assertIsNone(load_enhancement_transcript({"source_audio_transcript_for_enhancement": True}))
            asr.assert_not_called()

    def test_audio_and_video_drive_roles_use_the_native_exact_driver_resolver(self):
        for reference in (
            {"type": "audio", "path": "music.wav", "audio_intent": "drive"},
            {"type": "video", "path": "performance.mp4", "video_audio_role": "drive"},
        ):
            with self.subTest(reference=reference), patch(
                "models.minimax_h3.reference_manifest.split_exact_drive_audio_reference",
                return_value=([], "extracted.wav", 1),
            ) as resolve:
                self.assertEqual(source_audio_driver({"references": [reference]}), "extracted.wav")
                resolve.assert_called_once_with([reference])
        with self.assertRaisesRegex(ValueError, "Choose one"):
            source_audio_driver({"references": [
                {"type": "audio", "audio_intent": "drive"},
                {"type": "audio", "audio_intent": "drive"},
            ]})

    def test_window_contexts_follow_overlap_and_never_copy_the_whole_track(self):
        windows = conditioning_transcript_windows([
            {"start_seconds": 0, "end_seconds": 10},
            {"start_seconds": 10, "end_seconds": 19},
        ], overlap_frames=24, fps=24)
        self.assertEqual(windows[1]["start_seconds"], 9)
        contexts = transcript_window_contexts(transcript(), windows)
        self.assertIn("AURORA", contexts[0])
        self.assertNotIn("MARSH", contexts[0])
        self.assertIn("MARSH", contexts[1])
        self.assertNotIn("AURORA", contexts[1])
        self.assertNotIn("RAINBOW", "".join(contexts))
        combined = transcript_plan_context(transcript(), windows)
        self.assertTrue(has_h3_performance_audio(combined))
        self.assertNotIn("No transcript or musical analysis is supplied", performance_audio_guidance(combined))
        self.assertIn("No transcript or musical analysis is supplied", performance_audio_guidance())

    def test_enhancement_offset_and_plan_fingerprint_track_the_actual_source(self):
        with patch("services.source_audio_transcript.get_source_audio_transcript", return_value=transcript()):
            shifted = load_enhancement_transcript({
                "audio_guide": "source.wav", "audio_prompt_type": "A",
                "source_audio_transcript_for_enhancement": True, "audio_frame_offset": 240, "fps": 24,
            })
        contexts = transcript_window_contexts(shifted, [{"start_seconds": 0, "end_seconds": 9}])
        self.assertIn("MARSH", contexts[0])
        self.assertNotIn("AURORA", contexts[0])
        self.assertNotEqual(transcript_fingerprint(shifted), transcript_fingerprint(transcript()))
        self.assertEqual(transcript_fingerprint(None), "")

    def test_prompt_insertion_preserves_audio_fields_is_idempotent_and_removes_stale_context(self):
        prompt = "integrated_multimodal_description: A landscape.\noverall_soundscape: Source audio.\nnon_diegetic_music: Source music."
        first = append_window_transcript(prompt, transcript(), 0, 10)
        self.assertIn("AURORA", first)
        self.assertLess(first.index("AURORA"), first.index("overall_soundscape:"))
        second = append_window_transcript(first, transcript(), 10, 19)
        self.assertIn("MARSH", second)
        self.assertNotIn("AURORA", second)
        self.assertEqual(second.count("[BEGIN SUPPLIED SOURCE-AUDIO TRANSCRIPT CONTEXT]"), 1)
        self.assertEqual(strip_window_transcript(second), prompt)
        self.assertEqual(append_window_transcript(second, None, 10, 19), prompt)

    def test_generic_timing_uses_full_single_clip_and_real_rolling_stride(self):
        self.assertEqual(generic_transcript_windows({"duration_seconds": 10, "window_size_seconds": 5}),
                         [{"start_seconds": 0, "end_seconds": 10}])
        windows = generic_transcript_windows({
            "duration_seconds": 25, "window_count": 3, "window_size_seconds": 10,
            "window_overlap_seconds": 1, "window_discard_seconds": 1,
        })
        self.assertEqual([row["start_seconds"] for row in windows], [0, 8, 16])
        self.assertEqual(windows[-1]["end_seconds"], 25)

    def test_actual_runtime_block_uses_the_same_audio_clock_with_offsets_and_overlap(self):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and "source_audio_transcript_in_prompt" in ast.unparse(node.test))
        scope = {"source_audio_transcript_in_prompt": True, "source_audio_transcript_data": transcript(),
                 "audio_guide": "source.wav", "reference_start_frame": 240,
                 "current_video_length": 216, "fps": 24, "prompt": "A landscape.",
                 "original_prompts": ["A landscape."], "prompts": ["A landscape."], "window_no": 2}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), "wgp.py", "exec"), scope)
        self.assertIn("MARSH", scope["prompt"])
        self.assertNotIn("AURORA", scope["prompt"])
        self.assertEqual(scope["original_prompts"][1], scope["prompt"])
        scope.update(source_audio_transcript_in_prompt=False, prompt="Unchanged.")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), "wgp.py", "exec"), scope)
        self.assertEqual(scope["prompt"], "Unchanged.")

    def test_queued_single_window_enhancement_preserves_both_switches_and_audio(self):
        from services.studio_enhancement import enhancement_request
        payload, sequence = enhancement_request({
            "prompt": "A landscape.", "model_type": "minimax_h3_fused_turbo", "video_length": 240,
            "audio_guide": "song.wav", "audio_prompt_type": "A",
            "source_audio_transcript_in_prompt": False, "source_audio_transcript_for_enhancement": True,
        }, {"architecture": "minimax_h3", "fps": 24, "frames_maximum": 345})
        self.assertFalse(sequence)
        self.assertTrue(payload["source_audio_transcript_for_enhancement"])
        self.assertFalse(payload["source_audio_transcript_in_prompt"])
        self.assertEqual(payload["audio_guide"], "song.wav")

    def test_h3_frames_planner_receives_matching_context_and_signature_invalidates(self):
        from services import h3_window_planner as planner
        arguments = dict(model_type="minimax_h3_fused_turbo", resolution="832x480", total_frames=480,
                         window_frames=243, overlap_frames=18, fps=24)
        with patch.object(planner, "plan_h3_story_segments", side_effect=RuntimeError("offline")) as writer:
            planned = planner.plan_h3_sliding_windows("A dancer performs to the supplied song.",
                                                      source_audio_transcript=transcript(), **arguments)
        kwargs = writer.call_args.kwargs
        self.assertIn("AURORA", kwargs["source_audio_context_windows"][0])
        self.assertNotIn("MARSH", kwargs["source_audio_context_windows"][0])
        self.assertIn("MARSH", kwargs["source_audio_context_windows"][1])
        self.assertFalse(kwargs["expect_dialogue"])
        with patch.object(planner, "plan_h3_story_segments", side_effect=RuntimeError("offline")):
            ordinary = planner.plan_h3_sliding_windows("A dancer performs to the supplied song.", **arguments)
        self.assertNotEqual(planned["signature"], ordinary["signature"])
        reused = append_window_transcript("A dancer performs to the supplied song.", transcript(), 0, 10)
        with patch.object(planner, "plan_h3_story_segments", side_effect=RuntimeError("offline")):
            restored = planner.plan_h3_sliding_windows(reused, **arguments)
        self.assertEqual(restored["signature"], ordinary["signature"])
        self.assertNotIn("AURORA", restored["source_prompt"])

    def test_native_h3_reference_camera_passes_receive_only_their_window_transcript(self):
        from services import llm_service
        from services.h3_sequence_planner import plan_h3_reference_sequence
        from test_h3_performance_audio import SOURCE, REFERENCES, story
        camera_contexts = []

        def writer(**kwargs):
            fields = kwargs["json_schema"]["properties"]
            if "beats" in fields:
                self.assertIn("AURORA", kwargs["prompt"])
                self.assertIn("MARSH", kwargs["prompt"])
                return json.dumps(story())
            camera_contexts.append(kwargs["prompt"].split("MATCHING SOURCE AUDIO CONTEXT:\n", 1)[1])
            number = fields["segment"]["minimum"]
            events = json.loads(kwargs["prompt"].split(
                "Assigned chronological events (depict each once, in order):\n", 1,
            )[1].split("\n\nImmutable dialogue performances", 1)[0])
            return json.dumps({
                "segment": number, "title": "The performance continues", "coverage": "Alex dancing on the lawn",
                "pacing": "Keep time with source music", "event_cards": {
                    f"event_{i}": {"phases": [{"action": event["staging_draft"], "framing": "Medium full shot of Alex",
                                             "camera": "Handheld phone camera", "transition": "continuous reframe",
                                             "sound_effects": "N/A"}]} for i, event in enumerate(events, start=1)},
                "closing_state": "Alex remains mid-dance on the lawn.",
            })

        with patch.object(llm_service, "generate", side_effect=writer):
            plan = plan_h3_reference_sequence(
                SOURCE, model_type="minimax_h3_ref2va_fused_turbo", resolution="704x1280",
                total_frames=1620, min_clip_frames=5, max_clip_frames=243, frame_step=1, fps=24,
                references=REFERENCES, native_continuation=True, overlap_frames=18,
                planning_style="adaptive", source_audio_transcript=transcript(),
            )
        self.assertEqual(len(camera_contexts), 8)
        self.assertIn("AURORA", camera_contexts[0])
        self.assertNotIn("MARSH", camera_contexts[0])
        self.assertIn("MARSH", camera_contexts[1])
        self.assertNotIn("AURORA", camera_contexts[1])
        for prompt in plan["window_prompts"]:
            self.assertNotIn("<d>", prompt)
            self.assertNotIn("BEGIN SUPPLIED SOURCE-AUDIO", prompt)  # Generation opt-in remains separate.

    def test_ltx_window_enhancement_gets_timed_context_without_changing_source_prompt(self):
        from services import llm_service
        from services.ltx_window_planner import plan_ltx_sliding_windows
        with patch.object(llm_service, "enhance_prompt", return_value="A dancer steps left.\nThe dancer turns right.") as writer:
            plan = plan_ltx_sliding_windows("A dancer performs.", model_type="ltx2_19B",
                duration_seconds=19, window_count=2, window_size_seconds=10,
                overlap_seconds=1, source_audio_transcript=transcript())
        context = writer.call_args.kwargs["reference_context"]
        self.assertIn("AURORA", context)
        self.assertIn("MARSH", context)
        self.assertNotIn("RAINBOW", context)
        self.assertEqual(plan["source_prompt"], "A dancer performs.")


if __name__ == "__main__":
    unittest.main()
