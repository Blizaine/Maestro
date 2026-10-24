"""CPU-only protocol and admission tests for the Fish Small native adapter."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

import numpy as np

from models.TTS.fish_audio_small.small_runtime import (
    MAX_STREAM_SECONDS,
    SAMPLE_RATE,
    AudioCppServer,
    MemoryPlan,
    NativeStreamError,
    _consume_sse,
    choose_memory_plan,
    decode_pcm16_wav,
)


def _wav_event(samples: int, *, sample_rate: int = SAMPLE_RATE) -> str:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.zeros(samples, dtype="<i2").tobytes())
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _done_result(*, frames: int = 1, reached_limit: bool = False, text: str = "hello"):
    payload = json.dumps(
        {"text": text, "vq_codes": [[0] * frames for _ in range(10)]}
    ).encode("utf-8")
    return {
        "custom_schema_output": {
            "schema": "drama3_generation_v1",
            "data": {"frames": frames, "reached_limit": reached_limit},
        },
        "artifacts": [
            {
                "id": "drama3_take",
                "kind": "acoustic_tokens",
                "meta": {"family": "fish_drama3", "schema": "drama3_take_v1"},
                "payload": base64.b64encode(payload).decode("ascii"),
            }
        ],
    }


class FishSmallAdmissionTests(unittest.TestCase):
    gib = 1024**3

    def test_8_gib_gpu_uses_disk_backed_gpu_expert_cache(self):
        plan = choose_memory_plan(8 * self.gib, 7 * self.gib)

        self.assertEqual(plan.session_options["fish_drama3.cpu_expert_layers"], 0)
        self.assertTrue(plan.session_options["fish_drama3.disk_experts"])
        self.assertEqual(plan.session_options["fish_drama3.expert_cache_size"], 8)
        self.assertEqual(plan.session_options["fish_drama3.codec_encoder_chunk_samples"], 65536)
        self.assertFalse(plan.session_options["fish_drama3.pin_experts"])
        self.assertIn("CPU expert execution is not enabled", plan.reason)

    def test_12_gib_gpu_keeps_experts_on_gpu_with_disk_cache(self):
        plan = choose_memory_plan(12 * self.gib, 10 * self.gib)

        self.assertEqual(plan.session_options["fish_drama3.cpu_expert_layers"], 0)
        self.assertTrue(plan.session_options["fish_drama3.disk_experts"])
        self.assertEqual(plan.session_options["fish_drama3.expert_cache_size"], 8)
        self.assertIn("8-slot native GPU cache", plan.reason)
        self.assertEqual(plan.report()["minimum_dense_gpu_gib"], 6.0)

    def test_resident_experts_require_free_peak_plus_headroom(self):
        plan = choose_memory_plan(12 * self.gib, int(11.3 * self.gib))

        self.assertEqual(plan.session_options["fish_drama3.cpu_expert_layers"], 0)
        self.assertFalse(plan.session_options["fish_drama3.disk_experts"])

    def test_explicit_cpu_expert_override_is_preserved(self):
        plan = choose_memory_plan(
            12 * self.gib,
            10 * self.gib,
            overrides={"fish_drama3.cpu_expert_layers": 5},
        )

        self.assertEqual(plan.session_options["fish_drama3.cpu_expert_layers"], 5)
        self.assertIn("session uses 5 CPU expert layers", plan.reason)

    def test_insufficient_dense_cuda_memory_fails_with_actionable_message(self):
        with self.assertRaisesRegex(RuntimeError, "about 6 GiB.*close other GPU workloads"):
            choose_memory_plan(8 * self.gib, int(5.5 * self.gib))

    def test_unknown_memory_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Fish Small session option"):
            choose_memory_plan(8 * self.gib, 7 * self.gib, overrides={"fish_drama3.typo": 1})

    def test_codec_encoder_chunk_size_must_follow_upstream_granularity(self):
        with self.assertRaisesRegex(ValueError, "multiple of 512"):
            choose_memory_plan(
                8 * self.gib,
                7 * self.gib,
                overrides={"fish_drama3.codec_encoder_chunk_samples": 65535},
            )

    def test_invalid_memory_override_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_memory_plan(
                8 * self.gib,
                7 * self.gib,
                overrides={"fish_drama3.expert_cache_size": 7},
            )


class FishSmallSseTests(unittest.TestCase):
    def test_wav_event_is_bounded_pcm16_mono_44100(self):
        encoded = _wav_event(2048)
        pcm, rate, channels, samples = decode_pcm16_wav(encoded)

        self.assertEqual((rate, channels, samples), (SAMPLE_RATE, 1, 2048))
        self.assertEqual(len(pcm), 4096)

    def test_wav_event_with_wrong_rate_is_rejected(self):
        with self.assertRaises(NativeStreamError):
            decode_pcm16_wav(_wav_event(2048, sample_rate=22050))

    def test_sse_uses_native_finish_metadata_and_streamed_pcm(self):
        text = "<|speaker:0|> hello"
        done = _done_result(frames=1, text=text)
        events = [
            {"type": "task.stream.event", "event": {"audio": _wav_event(2048)}},
            {"type": "task.stream.done", "result": done},
        ]
        response = io.BytesIO(
            b"".join(
                b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"
                for event in events
            )
        )
        audio = []

        result = _consume_sse(response, lambda *args: audio.append(args), None)

        self.assertEqual(result.frames, 1)
        self.assertFalse(result.reached_limit)
        self.assertEqual(result.take_text, text)
        self.assertEqual(len(audio), 1)
        self.assertEqual(audio[0][3], 2048)  # PCM sample frames, not codec frames.

    def test_missing_native_done_marker_is_not_success(self):
        response = io.BytesIO(b'data: {"type":"task.stream.event","event":{}}\n\n')

        with self.assertRaisesRegex(NativeStreamError, "without task.stream.done"):
            _consume_sse(response, lambda *args: None, None)

    def test_native_error_event_is_not_success(self):
        response = io.BytesIO(
            b'data: {"type":"error","error":{"message":"bad request"}}\n\n'
        )

        with self.assertRaisesRegex(NativeStreamError, "bad request"):
            _consume_sse(response, lambda *args: None, None)

    def test_wav_decoder_rejects_event_beyond_30_second_bound(self):
        event = {
            "type": "task.stream.event",
            "event": {"audio": _wav_event(MAX_STREAM_SECONDS * SAMPLE_RATE + 1)},
        }
        response = io.BytesIO(
            b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"
        )

        with self.assertRaisesRegex(NativeStreamError, "larger than the 30-second bound"):
            _consume_sse(response, lambda *args: None, None)


class _FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminate_count = 0
        self.kill_count = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_count += 1
        self.returncode = 0

    def kill(self):
        self.kill_count += 1
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class _FakeSocket:
    def __init__(self):
        self.shutdown_count = 0

    def shutdown(self, _how):
        self.shutdown_count += 1


class _FakeConnection:
    def __init__(self):
        self.sock = _FakeSocket()
        self.close_count = 0

    def close(self):
        self.close_count += 1


class FishSmallChildLifecycleTests(unittest.TestCase):
    def test_release_closes_active_stream_and_terminates_owned_child_once(self):
        with tempfile.TemporaryDirectory() as folder:
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
                "mock admission plan",
                8 * 1024**3,
                7 * 1024**3,
            )
            server = AudioCppServer(
                types.SimpleNamespace(backend="cuda", release="mock"),
                Path(folder) / "model.gguf",
                plan,
            )
            process = _FakeProcess()
            connection = _FakeConnection()
            server._process = process
            server._active_connection = connection
            server._temp_directory = Path(folder)

            server.release()
            server.release()

        self.assertEqual(connection.sock.shutdown_count, 1)
        self.assertEqual(connection.close_count, 1)
        self.assertEqual(process.terminate_count, 1)
        self.assertEqual(process.kill_count, 0)
