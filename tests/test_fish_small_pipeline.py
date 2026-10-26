"""CPU mock tests for Fish Small reports, references, and continuation."""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

import models

tts_package = sys.modules.get("models.TTS", getattr(models, "TTS", None))
if tts_package is None:
    tts_package = types.ModuleType("models.TTS")
    tts_package.__path__ = [str(APP / "models" / "TTS")]
sys.modules["models.TTS"] = tts_package
models.TTS = tts_package

import torch

from models.TTS.fish_audio_small.pipeline import FRAME_LENGTH, FishSmallPipeline
from models.TTS.fish_audio_small.small_runtime import (
    SAMPLE_RATE,
    MemoryPlan,
    NativeStreamError,
    NativeStreamInterrupted,
    StreamResult,
)


def _artifact(text: str, frames: int) -> dict:
    payload = json.dumps(
        {"text": text, "vq_codes": [[0] * frames for _ in range(10)]}
    ).encode("utf-8")
    return {
        "id": "drama3_take",
        "kind": "acoustic_tokens",
        "meta": {"family": "fish_drama3", "schema": "drama3_take_v1"},
        "payload": base64.b64encode(payload).decode("ascii"),
    }


class FakeServer:
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.requests = []
        self.cancel_calls = 0
        self.released = False
        self.release_count = 0
        self.reference_infos = []

    def stream(self, request, on_audio, *, check_cancelled=None):
        self.requests.append(request)
        references = json.loads(request["options"]["multi_reference_cond"])
        self.reference_infos.append(
            [
                (
                    sf.info(reference["audio_path"]).format,
                    sf.info(reference["audio_path"]).subtype,
                    sf.info(reference["audio_path"]).samplerate,
                    sf.info(reference["audio_path"]).channels,
                )
                for reference in references
            ]
        )
        outcome = self.outcomes.pop(0) if self.outcomes else {}
        frames = outcome.get("frames", 1)
        text = request["text"]
        pcm = np.zeros(frames * FRAME_LENGTH, dtype="<i2").tobytes()
        on_audio(pcm, SAMPLE_RATE, 1, frames * FRAME_LENGTH)
        if outcome.get("cancel"):
            outcome["cancel"]()
            raise NativeStreamInterrupted("mock disconnect")
        if outcome.get("error"):
            raise NativeStreamError(str(outcome["error"]))
        take_artifact = _artifact(text, frames)
        payload = base64.b64decode(take_artifact["payload"])
        return StreamResult(
            frames=frames,
            reached_limit=bool(outcome.get("reached_limit", False)),
            take_text=text,
            artifact=take_artifact,
            artifact_payload=payload,
            result={},
        )

    def cancel_active(self):
        self.cancel_calls += 1

    def unload_all_models(self):
        pass

    def release(self):
        self.released = True
        self.release_count += 1


def _pipeline(server=None):
    plan = MemoryPlan(
        {
            "fish_drama3.cpu_expert_layers": 0,
            "fish_drama3.disk_experts": True,
            "fish_drama3.expert_cache_size": 8,
            "fish_drama3.context_length": 4096,
            "fish_drama3.codec_encoder_chunk_samples": 65536,
            "fish_drama3.pin_experts": False,
        },
        "cuda_disk_expert_cache",
        "mock memory plan",
        8 * 1024**3,
        7 * 1024**3,
    )
    return FishSmallPipeline(
        "model.gguf",
        types.SimpleNamespace(backend="cuda", release="mock-runtime"),
        plan,
        server=server or FakeServer(),
    )


class FishSmallPipelineTests(unittest.TestCase):
    def test_output_matches_maestro_tensor_shape_and_reports_eos(self):
        pipeline = _pipeline()

        result = pipeline.generate("Speaker 1: Hello there.", seed=42)

        self.assertIsInstance(result["x"], torch.Tensor)
        self.assertEqual(tuple(result["x"].shape), (1, FRAME_LENGTH))
        self.assertEqual(result["audio_sampling_rate"], SAMPLE_RATE)
        self.assertTrue(result["speech_generation"]["chunks"][0]["eos_completed"])
        self.assertFalse(result["speech_generation"]["text_coverage_verified"])
        self.assertIn("word-by-word", result["speech_generation"]["completion_basis"].lower())

    def test_8gb_plan_and_external_cache_facade_are_explicit(self):
        pipeline = _pipeline()

        self.assertIsInstance(pipeline.model, torch.nn.Identity)
        self.assertIsNone(pipeline.model.cache)
        result = pipeline.generate("Speaker 1: Hello.")
        plan = result["speech_generation"]["memory_plan"]
        self.assertEqual(plan["session_options"]["fish_drama3.cpu_expert_layers"], 0)
        self.assertTrue(plan["session_options"]["fish_drama3.disk_experts"])

    def test_all_three_references_keep_ordered_speaker_transcripts(self):
        server = FakeServer()
        pipeline = _pipeline(server)
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for name in ("one.wav", "two.wav", "three.wav"):
                path = Path(folder) / name
                sf.write(
                    str(path),
                    np.zeros((SAMPLE_RATE, 2), dtype=np.float32),
                    SAMPLE_RATE,
                    format="WAV",
                    subtype="PCM_16",
                )
                paths.append(str(path))

            pipeline.generate(
                "Speaker 1: New words.",
                audio_guide=paths[0],
                audio_guide2=paths[1],
                audio_guide3=paths[2],
                audio_prompt_type="ABD",
                alt_prompt=(
                    "Speaker 1: first reference\n"
                    "Speaker 2: second reference\n"
                    "Speaker 3: third reference"
                ),
            )

        references = json.loads(server.requests[0]["options"]["multi_reference_cond"])
        self.assertNotEqual([ref["audio_path"] for ref in references], paths)
        self.assertEqual(
            server.reference_infos[0],
            [("WAV", "PCM_16", SAMPLE_RATE, 1)] * 3,
        )
        self.assertTrue(all(not Path(ref["audio_path"]).exists() for ref in references))
        self.assertEqual(
            [ref["text"] for ref in references],
            [
                "<|speaker:0|> first reference",
                "<|speaker:1|> second reference",
                "<|speaker:2|> third reference",
            ],
        )

    def test_long_script_passes_only_prior_take_and_does_not_replay_it(self):
        server = FakeServer([{"frames": 1} for _ in range(8)])
        pipeline = _pipeline(server)
        script = "Speaker 1: " + " ".join(["word."] * 130)

        result = pipeline.generate(script, seed=123)

        self.assertGreater(len(server.requests), 1)
        self.assertNotIn("artifacts", server.requests[0])
        self.assertEqual(
            server.requests[1]["artifacts"][0]["id"], "drama3_take"
        )
        self.assertEqual(
            [request["options"]["seed"] for request in server.requests],
            [123] * len(server.requests),
        )
        completed_frames = len(server.requests)
        self.assertEqual(result["x"].shape[-1], completed_frames * FRAME_LENGTH)
        report = result["speech_generation"]
        self.assertEqual(report["completed_chunks"], len(server.requests))
        self.assertTrue(all(chunk["history_context"]["previous_take_included"]
                            for chunk in report["chunks"][1:]))

    def test_native_frame_cap_returns_marked_partial_audio_and_stops(self):
        server = FakeServer([{"frames": 2, "reached_limit": True}])
        pipeline = _pipeline(server)

        result = pipeline.generate("Speaker 1: Hello there.")

        report = result["speech_generation"]
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["chunks"][0]["finish_reason"], "length")
        self.assertEqual(report["covered_source_span"]["end"], 0)
        self.assertEqual(result["x"].shape[-1], 2 * FRAME_LENGTH)
        self.assertTrue(result["generation_warnings"])

    def test_early_stop_returns_only_already_streamed_partial_pcm(self):
        server = FakeServer()
        pipeline = _pipeline(server)
        server.outcomes = [
            {"frames": 1, "cancel": pipeline.request_early_stop}
        ]

        result = pipeline.generate("Speaker 1: A much longer script for cancellation.")

        self.assertEqual(tuple(result["x"].shape), (1, FRAME_LENGTH))
        report = result["speech_generation"]
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["partial_streamed_pcm_samples"], FRAME_LENGTH)
        self.assertEqual(report["chunks"][0]["generated_frames"], None)
        self.assertEqual(server.release_count, 1)
        self.assertEqual(server.cancel_calls, 0)

    def test_interrupt_release_is_one_shot_and_resets_for_next_generation(self):
        server = FakeServer()
        pipeline = _pipeline(server)

        def interrupt_twice():
            pipeline._interrupt = True
            pipeline._interrupt = True

        server.outcomes = [{"cancel": interrupt_twice}, {"cancel": interrupt_twice}]
        self.assertIsNone(pipeline.generate("Speaker 1: First cancellation."))
        self.assertEqual(server.release_count, 1)
        self.assertIsNone(pipeline.generate("Speaker 1: Second cancellation."))
        self.assertEqual(server.release_count, 2)

    def test_native_protocol_error_releases_child_before_raising(self):
        server = FakeServer([{"error": "missing task.stream.done"}])
        pipeline = _pipeline(server)

        with self.assertRaisesRegex(NativeStreamError, "missing task.stream.done"):
            pipeline.generate("Speaker 1: Protocol error.")

        self.assertEqual(server.release_count, 1)

    def test_manual_duration_is_sent_as_per_part_native_frame_limit(self):
        server = FakeServer()
        pipeline = _pipeline(server)

        pipeline.generate("Speaker 1: One.", duration_seconds=1)

        self.assertEqual(server.requests[0]["options"]["max_tokens"], 21)

    def test_native_cannot_resplit_an_already_bounded_part(self):
        server = FakeServer()
        pipeline = _pipeline(server)
        pipeline.generate("Speaker 1: [a calm but deliberate and very carefully measured delivery] Hello there.")
        for request in server.requests:
            self.assertGreater(request["options"]["text_chunk_size"], len(request["text"].encode("utf8")))

    def test_unknown_custom_settings_are_not_silently_ignored(self):
        pipeline = _pipeline()

        with self.assertRaisesRegex(ValueError, "does not support custom settings"):
            pipeline.generate("Speaker 1: Hello.", custom_settings={"mystery": 1})

    def test_zero_top_p_is_rejected_before_native_submission(self):
        server = FakeServer()
        pipeline = _pipeline(server)
        with self.assertRaisesRegex(ValueError, "top_p must be greater than 0"):
            pipeline.generate("Speaker 1: Hello.", top_p=0)
        self.assertEqual(server.requests, [])

    def test_small_rejects_unsupported_loras_before_loading(self):
        from models.TTS.fish_audio_small.fish_audio_small_handler import family_handler
        self.assertTrue(family_handler.query_model_def("fish_drama3_small", {})["loras_disabled"])
        error = family_handler.validate_generative_prompt(
            "fish_drama3_small", {}, {"activated_loras": ["adapter.safetensors"]}, "Hello."
        )
        self.assertIn("does not support LoRAs", error)

    def test_wav_and_flac_references_are_cpu_normalized_to_mono_pcm16(self):
        pipeline = _pipeline()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            paths = [folder / "reference.wav", folder / "reference.flac"]
            signal = np.linspace(-0.5, 0.5, 22_050, dtype=np.float32)
            stereo = np.column_stack((signal, signal * 0.5))
            sf.write(str(paths[0]), stereo, 22_050, format="WAV", subtype="PCM_16")
            sf.write(str(paths[1]), stereo, 22_050, format="FLAC", subtype="PCM_16")

            _, references = pipeline._prepare_references(
                "AB",
                (str(paths[0]), str(paths[1]), None),
                "Speaker 1: first\nSpeaker 2: second",
                None,
            )
            self.addCleanup(pipeline._cleanup_reference_temp)

            for reference in references:
                info = sf.info(reference["audio_path"])
                self.assertEqual((info.format, info.subtype), ("WAV", "PCM_16"))
                self.assertEqual((info.samplerate, info.channels), (SAMPLE_RATE, 1))
                self.assertAlmostEqual(info.duration, 1.0, places=3)
            self.assertEqual(
                [reference["text"] for reference in references],
                ["<|speaker:0|> first", "<|speaker:1|> second"],
            )

    def test_reference_over_30_seconds_is_rejected_before_full_decode(self):
        pipeline = _pipeline()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "long.wav"
            sf.write(
                str(path),
                np.zeros(int(SAMPLE_RATE * 30.1), dtype=np.float32),
                SAMPLE_RATE,
                format="WAV",
                subtype="PCM_16",
            )

            from unittest import mock

            with mock.patch("models.TTS.fish_audio_small.pipeline.sf.read") as read:
                with self.assertRaisesRegex(ValueError, "trim it to 30 seconds"):
                    pipeline._prepare_references(
                        "A", (str(path), None, None), "Speaker 1: transcript", None
                    )
                read.assert_not_called()
            self.assertIsNone(pipeline._reference_temp_owner)

    def test_short_high_channel_reference_is_bounded_before_decode(self):
        from unittest import mock
        from models.TTS.fish_audio_small.pipeline import _normalize_voice_reference

        header = types.SimpleNamespace(samplerate=48_000, frames=48_000 * 20, channels=64)
        with mock.patch("models.TTS.fish_audio_small.pipeline.sf.info", return_value=header):
            with mock.patch("models.TTS.fish_audio_small.pipeline.sf.read") as read:
                with self.assertRaisesRegex(ValueError, "too many decoded samples"):
                    _normalize_voice_reference(Path("source.wav"), Path("dest.wav"), letter="A")
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
