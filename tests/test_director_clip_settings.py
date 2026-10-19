"""Read-time Director gallery recipe reconstruction tests."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import types
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from app.services.director_clip_settings import (
    DIRECTOR_CLIP_SETTINGS_VERSION,
    enrich_director_output_metadata,
    json_safe_generation_params,
)


class DirectorClipSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pipeline_id = "test_pipeline"
        self.clips = [
            {
                "index": index,
                "video_filename": f"clip-{index}.mp4",
                "video_prompt": f"newer-plan-prompt-{index}",
                "planned_clip": {
                    "duration_frames": 120 + index,
                    "output_frames": 110 + index,
                    "start": index * 2.0,
                    "music_timing_version": 1,
                },
            }
            for index in range(8)
        ]
        self.state = {
            "pipeline_id": self.pipeline_id,
            "clips": self.clips,
            "video_model": "minimax_h3_ref2va_fused_turbo",
            "video_params": {
                "num_inference_steps": 4,
                "override_attention": "sla",
                "activated_loras": [],
                "loras_multipliers": "",
            },
            "video_execution_profile": {
                "fps": 24,
                "fused_turbo": True,
                "video_strategy": "omni_reference",
            },
            "shot_image_policy": "direct_references",
            "pipeline_type": "short_film_story",
            "_params_snapshot": {
                "pipeline_type": "short_film_story",
                "_director_video_execution_profile": {
                    "fused_turbo": True,
                    "fps": 24,
                },
            },
            "asset_manifest": {},
        }
        self._write_state(self.state)

    def tearDown(self):
        self.temp.cleanup()

    def _write_state(self, state):
        (self.root / f"_director_pipeline_{self.pipeline_id}.json").write_text(
            json.dumps(state), encoding="utf-8",
        )

    def _legacy_metadata(self, index):
        prompts = [f"historical prompt {item} " for item in range(8)]
        raw = "\n---CLIP_BOUNDARY---\n".join(prompts)
        return {
            "director_pipeline_id": self.pipeline_id,
            "director_clip_index": index,
            "director_artifact_kind": "clip",
            "params": {
                "prompt": raw,
                "seed": 563604453,
                "model_type": "minimax_h3_ref2va_fused_turbo",
                "video_length": 5520,
                "num_inference_steps": 6,
                "override_attention": "veda",
                "activated_loras": ["style.safetensors"],
                "loras_multipliers": "style.safetensors;0.65",
                "per_clip_frames": [345] * 8,
                "per_clip_output_frames": [332, 342, 345, 332, 343, 345, 339, 344],
                "multi_prompts_gen_type": 3,
                "repeat_generation": 4,
                "h3_window_plan": {"source_prompt": "project-wide prompt"},
                "_prompt_enhancement_payload": {"project": "stale"},
            },
        }

    def _enrich(self, name, metadata):
        return enrich_director_output_metadata(
            str(self.root), "test-workspace", name, metadata,
        )

    def test_legacy_batch_indices_zero_and_seven_restore_exact_single_clip(self):
        for index in (0, 7):
            with self.subTest(index=index):
                original = self._legacy_metadata(index)
                restored = self._enrich(f"clip-{index}.mp4", original)
                params = restored["params"]
                self.assertEqual(restored["director_clip_index"], index)
                self.assertEqual(restored["director_artifact_kind"], "clip")
                self.assertEqual(
                    restored["director_clip_settings_version"],
                    DIRECTOR_CLIP_SETTINGS_VERSION,
                )
                self.assertEqual(
                    restored["director_clip_settings_provenance"],
                    "legacy_batch_sidecar",
                )
                self.assertEqual(params["prompt"], f"historical prompt {index} ")
                self.assertEqual(params["seed"], 563604453)
                self.assertEqual(params["num_inference_steps"], 6)
                self.assertEqual(params["override_attention"], "veda")
                self.assertEqual(params["activated_loras"], ["style.safetensors"])
                self.assertEqual(params["loras_multipliers"], "style.safetensors;0.65")
                self.assertEqual(params["video_length"], 345)
                self.assertEqual(params["trim_tail_frames"], 1 if index == 7 else 13)
                self.assertEqual(params["guidance_scale"], 1.0)
                self.assertEqual(params["flow_shift"], 12.0)
                self.assertEqual(params["audio_flow_shift"], 3.0)
                self.assertEqual(params["skip_steps_cache_type"], "")
                self.assertEqual(params["multi_prompts_gen_type"], 0)
                self.assertEqual(params["repeat_generation"], 1)
                self.assertNotIn("h3_window_plan", params)
                self.assertNotIn("_prompt_enhancement_payload", params)
                self.assertAlmostEqual(params["duration_seconds"], 345 / 24)
                self.assertAlmostEqual(
                    restored["director_clip_timing"]["native_duration_seconds"],
                    14.375,
                )

    def test_joined_output_keeps_full_restore_without_clip_index(self):
        metadata = self._legacy_metadata(0)
        metadata["director_artifact_kind"] = "joined"
        metadata.pop("director_clip_index")
        expected_params = json.loads(json.dumps(metadata["params"]))
        restored = self._enrich("story_multiclip.mp4", metadata)
        self.assertEqual(restored["params"], expected_params)
        self.assertEqual(restored["director_artifact_kind"], "joined")
        self.assertNotIn("director_clip_index", restored)
        self.assertNotIn("director_clip_settings_error", restored)

    def test_rerun_without_saved_index_uses_exact_pipeline_filename(self):
        metadata = {
            "director_pipeline_id": self.pipeline_id,
            "params": {
                "prompt": "the original take prompt",
                "seed": 987654,
                "model_type": "minimax_h3_ref2va_fused_turbo",
                "video_length": 345,
                "trim_tail_frames": 1,
                "num_inference_steps": 6,
                "override_attention": "veda",
            },
        }
        restored = self._enrich("clip-7.mp4", metadata)
        self.assertEqual(restored["director_clip_index"], 7)
        self.assertEqual(
            restored["director_clip_settings_provenance"],
            "legacy_rerun_pipeline_match",
        )
        self.assertEqual(restored["params"]["prompt"], "the original take prompt")
        self.assertEqual(restored["params"]["seed"], 987654)
        self.assertEqual(restored["params"]["video_length"], 345)
        self.assertEqual(restored["params"]["trim_tail_frames"], 1)

    def test_historical_batch_prompt_wins_over_newer_pipeline_prompt(self):
        metadata = self._legacy_metadata(7)
        restored = self._enrich("clip-7.mp4", metadata)
        self.assertEqual(restored["params"]["prompt"], "historical prompt 7 ")
        self.assertNotEqual(
            restored["params"]["prompt"], self.clips[7]["video_prompt"],
        )

    def test_frames_start_image_is_restored_but_omni_identity_is_not_anchor(self):
        start = self.root / "frame-start.png"
        end = self.root / "frame-end.png"
        reference = self.root / "identity.png"
        for path in (start, end, reference):
            path.write_bytes(b"asset")

        state = json.loads(json.dumps(self.state))
        state["video_execution_profile"]["video_strategy"] = "bounded_start_end"
        state["shot_image_policy"] = "generate"
        state["clips"][0]["start_image_filename"] = start.name
        self._write_state(state)
        metadata = self._legacy_metadata(0)
        metadata["params"]["model_type"] = "wan-test"
        metadata["params"]["fps"] = 24
        metadata["params"]["image_start"] = str(start)
        metadata["params"]["image_end"] = str(end)
        metadata["params"]["image_prompt_type"] = "SE"
        frames = self._enrich("clip-0.mp4", metadata)
        self.assertEqual(frames["params"]["_studio_video_workflow"], "frames")
        self.assertEqual(frames["params"]["image_start"], str(start))
        self.assertEqual(frames["params"]["image_end"], str(end))
        metadata["params"]["image_start"] = start.name
        relative = self._enrich("clip-0.mp4", metadata)
        self.assertEqual(relative["params"]["image_start"], str(start))
        metadata["params"]["image_start"] = "missing-start.png"
        missing_start = self._enrich("clip-0.mp4", metadata)
        self.assertIsNone(missing_start["params"])
        self.assertIn("start image", missing_start["director_clip_settings_error"])

        state["video_execution_profile"]["video_strategy"] = "omni_reference"
        state["shot_image_policy"] = "direct_references"
        state["asset_manifest"] = {
            "minimax_h3_references": [
                {"path": {"path": str(reference)}, "role": "character"},
            ],
        }
        self._write_state(state)
        omni_metadata = self._legacy_metadata(0)
        omni_metadata["params"]["image_start"] = str(start)
        omni_metadata["params"]["image_end"] = str(end)
        omni_metadata["upload_filenames"] = {
            "image_start": start.name,
            "image_end": end.name,
        }
        omni = self._enrich("clip-0.mp4", omni_metadata)
        self.assertEqual(omni["params"]["_studio_video_workflow"], "references")
        self.assertNotIn("image_start", omni["params"])
        self.assertNotIn("image_end", omni["params"])
        self.assertEqual(omni["params"]["image_prompt_type"], "")
        self.assertNotIn("image_start", omni["upload_filenames"])
        self.assertNotIn("image_end", omni["upload_filenames"])
        self.assertEqual(omni["params"]["minimax_h3_references"][0]["url"],
                         "/api/v1/file/identity.png?workspace=test-workspace")

        empty_refs = self._legacy_metadata(0)
        empty_refs["params"]["minimax_h3_references"] = []
        empty_refs["params"]["image_start"] = str(start)
        no_references = self._enrich("clip-0.mp4", empty_refs)
        self.assertEqual(no_references["params"]["minimax_h3_references"], [])

    def test_changed_pipeline_model_cannot_override_historical_nonfused_settings(self):
        metadata = self._legacy_metadata(0)
        metadata["params"].update({
            "model_type": "wan-test",
            "fps": 24,
            "num_inference_steps": 8,
            "guidance_scale": 3.5,
            "flow_shift": 5.0,
            "override_attention": "",
            "skip_steps_cache_type": "",
        })
        restored = self._enrich("clip-0.mp4", metadata)
        params = restored["params"]
        self.assertEqual(params["model_type"], "wan-test")
        self.assertEqual(params["num_inference_steps"], 8)
        self.assertEqual(params["guidance_scale"], 3.5)
        self.assertEqual(params["flow_shift"], 5.0)
        self.assertEqual(params["override_attention"], "")
        self.assertEqual(params["skip_steps_cache_type"], "")

    def test_audio_fallback_slices_archived_song_at_saved_offset_atomically(self):
        source = self.root / "_director_assets" / "song.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"archived song")
        state = json.loads(json.dumps(self.state))
        state["pipeline_type"] = "music_video"
        state["_params_snapshot"].update({
            "pipeline_type": "music_video",
            "audio_path": str(source),
        })
        state["asset_manifest"] = {"audio_path": {"path": str(source)}}
        self._write_state(state)
        metadata = self._legacy_metadata(7)
        metadata["params"]["audio_guide"] = str(self.root / "_rerun_audio_deleted.wav")
        metadata["director_clip_timing"] = {
            "audio_source_path": str(self.root / "_rerun_audio_deleted.wav"),
            "audio_start_sec": 99.25,
        }
        seen = []
        started = threading.Event()
        release = threading.Event()

        def fake_slice(src, start, duration, dst):
            seen.append({
                "source": src,
                "start": start,
                "duration": duration,
                "dst": dst,
            })
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("the test did not release the audio writer")
            with wave.open(dst, "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\0\0\0\0" * 128)

        fake_pipeline = types.ModuleType("services.director_pipeline")
        fake_pipeline._slice_audio_segment = fake_slice
        with patch.dict(sys.modules, {"services.director_pipeline": fake_pipeline}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(self._enrich, "clip-7.mp4", metadata)
                    for _ in range(2)
                ]
                try:
                    self.assertTrue(started.wait(timeout=5))
                    self.assertFalse(any(future.done() for future in futures))
                finally:
                    release.set()
                restored_results = [future.result(timeout=10) for future in futures]
            for result in restored_results:
                self.assertNotIn("director_clip_settings_error", result, result.get("director_clip_settings_error"))
            # A complete stable derivative is reused without invoking ffmpeg.
            cached = self._enrich("clip-7.mp4", metadata)

        self.assertEqual(len(seen), 1)
        self.assertTrue(all(item["source"] == str(source) for item in seen))
        self.assertTrue(all(item["start"] == 99.25 for item in seen))
        self.assertTrue(all(item["duration"] == 345 / 24 for item in seen))
        paths = {result["params"]["audio_guide"] for result in restored_results}
        self.assertEqual(len(paths), 1)
        slice_path = next(iter(paths))
        params = cached["params"]
        self.assertEqual(params["audio_guide"], slice_path)
        self.assertIn("clip-7-", Path(slice_path).name)
        self.assertTrue(Path(slice_path).is_file())
        self.assertTrue(all(not Path(item["dst"]).exists() for item in seen))
        self.assertEqual(Path(slice_path).read_bytes()[:4], b"RIFF")
        self.assertNotIn("audio_frame_offset", params)
        self.assertIn("_director_clip_settings_slices", params["audio_guide"])
        self.assertIn("audio_guide", cached["director_clip_asset_urls"])

    def test_missing_required_reference_and_missing_legacy_state_fail_closed(self):
        reference = self.root / "identity-missing.png"
        state = json.loads(json.dumps(self.state))
        state["asset_manifest"] = {
            "minimax_h3_references": [
                {"path": {"path": str(reference)}, "role": "character"},
            ],
        }
        self._write_state(state)
        missing_asset = self._enrich("clip-0.mp4", self._legacy_metadata(0))
        self.assertIsNone(missing_asset["params"])
        self.assertIn("identity references", missing_asset["director_clip_settings_error"])

        (self.root / f"_director_pipeline_{self.pipeline_id}.json").unlink()
        missing_state = self._enrich("clip-0.mp4", self._legacy_metadata(0))
        self.assertIsNone(missing_state["params"])
        self.assertIn("was not found", missing_state["director_clip_settings_error"])

    def test_scoped_future_recipe_does_not_need_mutable_pipeline_file(self):
        (self.root / f"_director_pipeline_{self.pipeline_id}.json").unlink()
        metadata = {
            "director_pipeline_id": self.pipeline_id,
            "director_clip_index": 0,
            "director_artifact_kind": "clip",
            "director_clip_settings_provenance": "per_output_effective_params",
            "director_clip_timing": {
                "native_frames": 345,
                "output_frames": 342,
                "fps": 24,
            },
            "params": {
                "prompt": "exact child prompt",
                "seed": 23,
                "model_type": "wan-test",
                "video_length": 345,
                "fps": 24,
                "trim_tail_frames": 3,
                "num_inference_steps": 7,
                "image_prompt_type": "",
                "multi_prompts_gen_type": 0,
            },
        }
        restored = self._enrich("child.mp4", metadata)
        self.assertEqual(restored["params"]["prompt"], "exact child prompt")
        self.assertEqual(restored["params"]["video_length"], 345)
        self.assertEqual(restored["params"]["trim_tail_frames"], 3)
        self.assertEqual(restored["params"]["num_inference_steps"], 7)
        self.assertEqual(restored["director_clip_index"], 0)
        self.assertNotIn("director_clip_settings_error", restored)

    def test_sidecar_snapshot_skips_runtime_state_and_keeps_pil_attachment_path(self):
        from PIL import Image

        image_path = self.root / "actual-start.png"
        Image.new("RGB", (8, 8), color="red").save(image_path)
        opened = Image.open(image_path)

        class StateMustNotBeCopied:
            def __deepcopy__(self, _memo):
                raise AssertionError("live WGP state must not be copied")

        validated = {
            "state": StateMustNotBeCopied(),
            "prompt": "effective child prompt",
            "num_inference_steps": 6,
            "image_start": opened,
            "image_refs": [opened],
        }
        original_manifest_paths = {
            "image_start": str(image_path),
            "image_refs": [str(image_path)],
        }
        try:
            snapshot = json_safe_generation_params(
                validated,
                path_params=original_manifest_paths,
                attachment_keys=("image_start", "image_refs"),
            )
            encoded = json.dumps(snapshot)
        finally:
            opened.close()

        self.assertIn('"num_inference_steps": 6', encoded)
        self.assertEqual(snapshot["image_start"], str(image_path))
        self.assertEqual(snapshot["image_refs"], [str(image_path)])
        self.assertNotIn("state", snapshot)


if __name__ == "__main__":
    unittest.main()
