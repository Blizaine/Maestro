"""Loopback process and bounded SSE/WAV transport for Fish Drama 3 Small.

The audio.cpp executable and GGUF are external runtimes. This module owns only
the model-scoped child server and its request protocol; Maestro's Pinokio server
continues to own its usual application lifecycle.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping


FAMILY = "fish_drama3"
MODEL_ID = "fish-drama3-small"
SAMPLE_RATE = 44_100
CHANNELS = 1
MAX_STREAM_SECONDS = 30
MAX_SSE_LINE_BYTES = 8 * 1024 * 1024
MAX_HTTP_ERROR_BYTES = 64 * 1024
HEALTH_TIMEOUT_SECONDS = 60.0


class NativeStreamError(RuntimeError):
    """The native task failed before a successful stream result arrived."""


class NativeStreamInterrupted(NativeStreamError):
    """The active native SSE request was disconnected for cancellation/stop."""


@dataclass(frozen=True)
class StreamResult:
    frames: int
    reached_limit: bool
    take_text: str
    artifact: dict[str, Any]
    artifact_payload: bytes
    result: dict[str, Any]


@dataclass(frozen=True)
class MemoryPlan:
    session_options: dict[str, Any]
    name: str
    reason: str
    gpu_total_bytes: int | None
    gpu_free_bytes: int | None

    def report(self) -> dict[str, Any]:
        gib = 1024**3
        return {
            "name": self.name,
            "reason": self.reason,
            "gpu_total_gib": (
                round(self.gpu_total_bytes / gib, 3)
                if self.gpu_total_bytes is not None
                else None
            ),
            "gpu_free_gib_at_admission": (
                round(self.gpu_free_bytes / gib, 3)
                if self.gpu_free_bytes is not None
                else None
            ),
            "estimated_cpu_expert_layers": self.session_options[
                "fish_drama3.cpu_expert_layers"
            ],
            "minimum_dense_gpu_gib": 6.0,
            "resident_experts_free_gpu_gib": 11.26,
            "session_options": dict(self.session_options),
        }


def choose_memory_plan(
    gpu_total_bytes: int | None,
    gpu_free_bytes: int | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> MemoryPlan:
    """Choose explicit GPU expert residency or a bounded disk-backed GPU cache.

    Dense backbone, tower, and codec weights remain resident in both plans.
    The minimum 6 GiB free-memory admission floor is conservative and avoids
    claiming that disk-backed experts can make a CUDA device optional. Full
    expert residency requires the measured upstream 10.26 GiB peak plus 1 GiB
    headroom. This is an admission decision, never a retry after inference.
    """
    gib = 1024**3
    minimum_dense_gpu_bytes = 6 * gib
    resident_experts_gpu_bytes = int(11.26 * gib)
    if gpu_total_bytes is None or gpu_free_bytes is None:
        raise RuntimeError(
            "Fish Small could not measure free CUDA memory. Check that the NVIDIA "
            "driver is available and retry after restarting Maestro."
        )
    if gpu_free_bytes < minimum_dense_gpu_bytes:
        raise RuntimeError(
            "Fish Small needs about 6 GiB of free CUDA memory for its resident dense "
            "backbone, tower, and codec, even when experts use disk-backed GPU caching. "
            f"Only {gpu_free_bytes / gib:.2f} GiB is free; close other GPU workloads and retry."
        )

    enough_resident_memory = gpu_free_bytes >= resident_experts_gpu_bytes
    if enough_resident_memory:
        options: dict[str, Any] = {
            "fish_drama3.cpu_expert_layers": 0,
            "fish_drama3.disk_experts": False,
            "fish_drama3.expert_cache_size": 32,
            "fish_drama3.context_length": 4096,
            "fish_drama3.codec_encoder_chunk_samples": 65536,
            "fish_drama3.pin_experts": False,
        }
        name = "cuda_resident_experts"
        reason = (
            "CUDA free memory met the upstream 10.26 GiB streaming peak plus "
            "1 GiB headroom; all expert layers remain on the CUDA backend."
        )
    else:
        options = {
            "fish_drama3.cpu_expert_layers": 0,
            "fish_drama3.disk_experts": True,
            "fish_drama3.expert_cache_size": 8,
            "fish_drama3.context_length": 4096,
            "fish_drama3.codec_encoder_chunk_samples": 65536,
            "fish_drama3.pin_experts": False,
        }
        name = "cuda_disk_expert_cache"
        reason = (
            "CUDA free memory is below the 10.26 GiB streaming peak plus 1 GiB "
            "headroom, so the full expert banks stay in the GGUF and selected "
            "experts use an 8-slot native GPU cache. Dense backbone, tower, and "
            "codec weights remain resident; CPU expert execution is not enabled."
        )

    for key, value in (overrides or {}).items():
        if key not in options:
            raise ValueError(f"Unsupported Fish Small session option: {key}")
        options[key] = value

    requested_cpu_layers = _integer_option(
        options["fish_drama3.cpu_expert_layers"],
        "fish_drama3.cpu_expert_layers",
        0,
        40,
    )
    cache_size = _integer_option(
        options["fish_drama3.expert_cache_size"],
        "fish_drama3.expert_cache_size",
        8,
        256,
    )
    context_length = _integer_option(
        options["fish_drama3.context_length"],
        "fish_drama3.context_length",
        1,
        8192,
    )
    codec_chunk_samples = _integer_option(
        options["fish_drama3.codec_encoder_chunk_samples"],
        "fish_drama3.codec_encoder_chunk_samples",
        0,
        1_048_576,
    )
    if codec_chunk_samples and codec_chunk_samples % 512:
        raise ValueError("fish_drama3.codec_encoder_chunk_samples must be 0 or a multiple of 512")
    disk_experts = _boolean_option(
        options["fish_drama3.disk_experts"], "fish_drama3.disk_experts"
    )
    pin_experts = _boolean_option(options["fish_drama3.pin_experts"], "fish_drama3.pin_experts")
    if pin_experts and not disk_experts:
        raise ValueError("fish_drama3.pin_experts requires fish_drama3.disk_experts=true")
    if not enough_resident_memory and not disk_experts and requested_cpu_layers == 0:
        raise ValueError(
            "Fish Small cannot keep all expert banks resident below 11.26 GiB free CUDA memory; "
            "enable fish_drama3.disk_experts or explicitly select CPU expert layers."
        )
    cpu_layers = requested_cpu_layers
    options.update(
        {
            "fish_drama3.cpu_expert_layers": cpu_layers,
            "fish_drama3.disk_experts": disk_experts,
            "fish_drama3.expert_cache_size": cache_size,
            "fish_drama3.context_length": context_length,
            "fish_drama3.codec_encoder_chunk_samples": codec_chunk_samples,
            "fish_drama3.pin_experts": pin_experts,
        }
    )
    if overrides:
        overridden = ", ".join(sorted(set(overrides) & set(options)))
        if overridden:
            name = f"{name}_with_user_overrides"
            reason += f" Explicit session overrides: {overridden}."
    if cpu_layers > 0:
        reason = reason.replace(
            "CPU expert execution is not enabled.",
            "An explicit session setting enables CPU expert execution.",
        )
        reason += f" The session uses {cpu_layers} CPU expert layers."
    if pin_experts:
        reason += " Explicit session setting requests pinned host expert banks."
    if cpu_layers > 0 and disk_experts:
        name = "cuda_disk_expert_cache_with_cpu_expert_layers"
    elif cpu_layers > 0:
        name = "mixed_cpu_gpu_expert_residency"
    elif disk_experts:
        name = "cuda_disk_expert_cache"
    else:
        name = "cuda_resident_experts"
    return MemoryPlan(options, name, reason, gpu_total_bytes, gpu_free_bytes)


def _integer_option(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}") from error
    if isinstance(value, bool) or (
        isinstance(value, float) and not value.is_integer()
    ) or (
        isinstance(value, str) and str(parsed) != value.strip()
    ) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return parsed


def _boolean_option(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{name} must be true or false")


def decode_pcm16_wav(encoded: str) -> tuple[bytes, int, int, int]:
    """Decode one bounded base64 WAV event to raw PCM16 bytes and its format."""
    try:
        payload = base64.b64decode(encoded, validate=True)
        with wave.open(BytesIO(payload), "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frame_count = stream.getnframes()
            if stream.getcomptype() != "NONE":
                raise NativeStreamError("Fish Small stream returned compressed WAV audio")
            if sample_width != 2:
                raise NativeStreamError(
                    f"Fish Small stream returned {sample_width * 8}-bit WAV; expected PCM16"
                )
            if channels != CHANNELS or sample_rate != SAMPLE_RATE:
                raise NativeStreamError(
                    "Fish Small stream format changed: expected 44.1 kHz mono, "
                    f"got {sample_rate} Hz with {channels} channel(s)"
                )
            pcm = stream.readframes(frame_count)
            expected_bytes = frame_count * channels * sample_width
            if len(pcm) != expected_bytes:
                raise NativeStreamError("Fish Small stream contained a truncated WAV event")
    except NativeStreamError:
        raise
    except (ValueError, wave.Error, OSError, EOFError) as error:
        raise NativeStreamError(f"Fish Small stream returned an invalid WAV event: {error}") from error
    return pcm, sample_rate, channels, frame_count


class AudioCppServer:
    """Own a hidden loopback-only audio.cpp server and its streamed requests."""

    def __init__(
        self,
        runtime_info: Any,
        model_path: str | Path,
        memory_plan: MemoryPlan,
        *,
        process_factory: Callable[..., Any] = subprocess.Popen,
        connection_factory: Callable[..., http.client.HTTPConnection] = http.client.HTTPConnection,
        temp_directory: str | Path | None = None,
        ready_timeout: float = HEALTH_TIMEOUT_SECONDS,
    ) -> None:
        self.runtime_info = runtime_info
        self.model_path = Path(model_path).expanduser().resolve()
        self.memory_plan = memory_plan
        self._process_factory = process_factory
        self._connection_factory = connection_factory
        self._temp_directory = Path(temp_directory) if temp_directory else None
        self._ready_timeout = float(ready_timeout)
        self._temp_owner: tempfile.TemporaryDirectory[str] | None = None
        self._process: Any = None
        self._log_handle: Any = None
        self._config_path: Path | None = None
        self._port: int | None = None
        self._active_connection: Any = None
        self._active_lock = threading.RLock()
        self._server_lock = threading.RLock()
        self._last_log_tail = "unavailable"

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("Fish Small native server is not running")
        return f"http://127.0.0.1:{self._port}"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_started(self, check_cancelled: Callable[[], bool] | None = None) -> None:
        with self._server_lock:
            if self.running:
                return
            executable = Path(self.runtime_info.executable).expanduser().resolve()
            if not executable.is_file():
                raise FileNotFoundError(f"audio.cpp server executable not found: {executable}")
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Fish Small GGUF not found: {self.model_path}")
            self._prepare_temp_directory()
            self._port = _unused_loopback_port()
            self._config_path = self._temp_directory / "server.json"
            config = self.build_config(self._port)
            self._config_path.write_text(
                json.dumps(config, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            log_path = self._temp_directory / "audiocpp_server.log"
            self._log_handle = log_path.open("ab")
            env = os.environ.copy()
            env.update(getattr(self.runtime_info, "environment", {}) or {})
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            try:
                self._process = self._process_factory(
                    [str(executable), "--config", str(self._config_path)],
                    cwd=str(getattr(self.runtime_info, "root", executable.parent)),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=flags,
                )
            except Exception:
                self._close_log()
                raise
            self._wait_ready(check_cancelled)

    def build_config(self, port: int) -> dict[str, Any]:
        return {
            "host": "127.0.0.1",
            "port": int(port),
            "backend": str(getattr(self.runtime_info, "backend", "cuda") or "cuda"),
            "threads": max(1, min(os.cpu_count() or 1, 8)),
            "lazy_load": True,
            "max_loaded_models": 1,
            "ui": False,
            "models": [
                {
                    "id": MODEL_ID,
                    "family": FAMILY,
                    "path": str(self.model_path),
                    "task": "tts",
                    "mode": "streaming",
                    "session_options": self.memory_plan.session_options,
                    "default_request_options": {
                        "max_tokens": int(MAX_STREAM_SECONDS * SAMPLE_RATE / 2048),
                        "top_k": 4097,
                        "top_p": 0.9,
                    },
                }
            ],
        }

    def stream(
        self,
        request: Mapping[str, Any],
        on_audio: Callable[[bytes, int, int, int], None],
        *,
        check_cancelled: Callable[[], bool] | None = None,
    ) -> StreamResult:
        self.ensure_started(check_cancelled)
        payload = json.dumps(
            {"model": MODEL_ID, "stream_format": "sse", "request": dict(request)},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        connection = self._new_connection()
        with self._active_lock:
            if self._active_connection is not None:
                connection.close()
                raise NativeStreamError("Fish Small server already has an active request")
            self._active_connection = connection
        try:
            try:
                connection.request(
                    "POST",
                    "/v1/tasks/stream",
                    body=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "Connection": "keep-alive",
                    },
                )
                response = connection.getresponse()
                if response.status < 200 or response.status >= 300:
                    body = response.read(MAX_HTTP_ERROR_BYTES + 1)
                    detail = body[:MAX_HTTP_ERROR_BYTES].decode("utf-8", errors="replace")
                    raise NativeStreamError(
                        f"audio.cpp task request failed with HTTP {response.status}: {detail}"
                    )
                result = _consume_sse(response, on_audio, check_cancelled)
                return result
            except NativeStreamInterrupted:
                raise
            except NativeStreamError:
                raise
            except (OSError, http.client.HTTPException, ValueError) as error:
                if check_cancelled is not None and check_cancelled():
                    raise NativeStreamInterrupted("Fish Small stream disconnected for cancellation") from error
                raise NativeStreamError(f"Fish Small task stream failed: {error}") from error
        finally:
            with self._active_lock:
                if self._active_connection is connection:
                    self._active_connection = None
            try:
                connection.close()
            except Exception:
                pass

    def cancel_active(self) -> None:
        with self._active_lock:
            connection, self._active_connection = self._active_connection, None
        if connection is None:
            return
        try:
            sock = getattr(connection, "sock", None)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()
        except Exception:
            pass

    def unload_all_models(self) -> None:
        if not self.running or self._port is None:
            return
        with self._active_lock:
            if self._active_connection is not None:
                # The server unload route waits for in-flight inference. Defer
                # unloading until a subsequent idle lifecycle boundary.
                return
        connection = self._new_connection(timeout=2.0)
        try:
            connection.request("POST", "/v1/tasks/unload_all_models", body=b"{}", headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            response.read(MAX_HTTP_ERROR_BYTES + 1)
            if response.status < 200 or response.status >= 300:
                raise NativeStreamError(f"audio.cpp model unload failed with HTTP {response.status}")
        except (OSError, http.client.HTTPException) as error:
            raise NativeStreamError(f"could not unload Fish Small native model: {error}") from error
        finally:
            connection.close()

    def release(self) -> None:
        self.cancel_active()
        with self._server_lock:
            process, self._process = self._process, None
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except Exception:
                    try:
                        process.kill()
                        process.wait(timeout=3)
                    except Exception:
                        pass
            self._close_log()
            if self._temp_owner is not None:
                self._temp_owner.cleanup()
                self._temp_owner = None
            self._temp_directory = None
            self._config_path = None
            self._port = None

    def _prepare_temp_directory(self) -> None:
        if self._temp_directory is not None:
            self._temp_directory.mkdir(parents=True, exist_ok=True)
            return
        self._temp_owner = tempfile.TemporaryDirectory(prefix="maestro-fish-small-")
        self._temp_directory = Path(self._temp_owner.name)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    def _new_connection(self, timeout: float | None = None) -> http.client.HTTPConnection:
        if self._port is None:
            raise RuntimeError("Fish Small native server has no loopback port")
        try:
            return self._connection_factory("127.0.0.1", self._port, timeout=timeout)
        except TypeError:
            # Small test doubles may not accept the standard timeout keyword.
            return self._connection_factory("127.0.0.1", self._port)

    def _wait_ready(self, check_cancelled: Callable[[], bool] | None) -> None:
        deadline = time.monotonic() + self._ready_timeout
        last_error = "server has not answered health checks"
        while time.monotonic() < deadline:
            if check_cancelled is not None and check_cancelled():
                self.release()
                raise NativeStreamInterrupted("Fish Small server startup was cancelled")
            if not self.running:
                self._last_log_tail = self._tail_log()
                raise RuntimeError(
                    "audio.cpp server exited before becoming ready. "
                    f"Log: {self._last_log_tail}"
                )
            connection = self._new_connection(timeout=1.0)
            try:
                connection.request("GET", "/health")
                response = connection.getresponse()
                body = response.read(MAX_HTTP_ERROR_BYTES + 1)
                if response.status == 200:
                    return
                last_error = f"health endpoint returned HTTP {response.status}: {body[:1024]!r}"
            except (OSError, http.client.HTTPException) as error:
                last_error = str(error)
            finally:
                connection.close()
            time.sleep(0.2)
        self._last_log_tail = self._tail_log()
        self.release()
        raise RuntimeError(
            f"audio.cpp server did not become ready within {self._ready_timeout:g}s: "
            f"{last_error}. Log: {self._last_log_tail}"
        )

    def _tail_log(self) -> str:
        if self._temp_directory is None:
            return "unavailable"
        path = self._temp_directory / "audiocpp_server.log"
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 8192), os.SEEK_SET)
                return handle.read().decode("utf-8", errors="replace")[-8192:]
        except OSError:
            return self._last_log_tail


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _consume_sse(
    response: Any,
    on_audio: Callable[[bytes, int, int, int], None],
    check_cancelled: Callable[[], bool] | None,
) -> StreamResult:
    data_lines: list[str] = []
    data_bytes = 0
    stream_result: StreamResult | None = None
    while True:
        if check_cancelled is not None and check_cancelled():
            raise NativeStreamInterrupted("Fish Small task stream was disconnected")
        line = response.readline(MAX_SSE_LINE_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_SSE_LINE_BYTES or (
            len(line) == MAX_SSE_LINE_BYTES + 1 and not line.endswith(b"\n")
        ):
            raise NativeStreamError("Fish Small SSE event exceeded the 8 MiB safety limit")
        if line in (b"\n", b"\r\n"):
            if not data_lines:
                data_bytes = 0
                continue
            event_text = "\n".join(data_lines)
            data_lines.clear()
            data_bytes = 0
            if event_text == "[DONE]":
                if stream_result is None:
                    raise NativeStreamError("audio.cpp ended the SSE stream without task.stream.done")
                return stream_result
            try:
                message = json.loads(event_text)
            except json.JSONDecodeError as error:
                raise NativeStreamError(f"audio.cpp sent invalid SSE JSON: {error}") from error
            event_type = message.get("type")
            if event_type == "error":
                error = message.get("error")
                detail = error.get("message") if isinstance(error, dict) else error
                raise NativeStreamError(f"audio.cpp reported a task error: {detail or 'unknown error'}")
            if event_type == "task.stream.event":
                event = message.get("event") or {}
                encoded = event.get("audio")
                if encoded:
                    pcm, sample_rate, channels, frames = decode_pcm16_wav(encoded)
                    if frames > MAX_STREAM_SECONDS * sample_rate:
                        raise NativeStreamError("Fish Small emitted one audio event larger than the 30-second bound")
                    on_audio(pcm, sample_rate, channels, frames)
            elif event_type == "task.stream.done":
                if stream_result is not None:
                    raise NativeStreamError("audio.cpp sent task.stream.done more than once")
                stream_result = _parse_done_result(message.get("result"))
                # This is the authoritative success marker. The server emits
                # [DONE] next, but the result is already complete here.
                return stream_result
            continue

        if line.startswith(b":"):
            continue
        field, separator, value = line.partition(b":")
        if not separator:
            continue
        if field == b"data":
            value = value[1:] if value.startswith(b" ") else value
            data_bytes += len(value)
            if data_bytes > MAX_SSE_LINE_BYTES:
                raise NativeStreamError("Fish Small SSE event exceeded the 8 MiB safety limit")
            data_lines.append(value.decode("utf-8", errors="strict"))
    if check_cancelled is not None and check_cancelled():
        raise NativeStreamInterrupted("Fish Small task stream was disconnected")
    if stream_result is not None:
        return stream_result
    raise NativeStreamError("audio.cpp stream ended without task.stream.done")


def _parse_done_result(result: Any) -> StreamResult:
    if not isinstance(result, dict):
        raise NativeStreamError("audio.cpp task.stream.done did not include a result object")
    schema = result.get("custom_schema_output")
    if not isinstance(schema, dict) or schema.get("schema") != "drama3_generation_v1":
        raise NativeStreamError(
            "audio.cpp task result did not include Drama 3 generation metadata"
        )
    data = schema.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("reached_limit"), bool):
        raise NativeStreamError("Drama 3 generation metadata is missing reached_limit")
    raw_frames = data.get("frames")
    if isinstance(raw_frames, bool) or not isinstance(raw_frames, int):
        raise NativeStreamError("Drama 3 generation metadata is missing a valid frame count")
    try:
        frames = int(raw_frames)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeStreamError("Drama 3 generation metadata is missing a valid frame count") from error
    if frames < 0:
        raise NativeStreamError("Drama 3 returned a negative frame count")

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise NativeStreamError("Drama 3 task result omitted its take artifact")
    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict)
            and item.get("id") == "drama3_take"
            and item.get("kind") == "acoustic_tokens"
        ),
        None,
    )
    if artifact is None:
        raise NativeStreamError("Drama 3 task result omitted the drama3_take artifact")
    meta = artifact.get("meta") or {}
    if meta.get("family") != FAMILY or meta.get("schema") != "drama3_take_v1":
        raise NativeStreamError("Drama 3 returned a take artifact with an unexpected family or schema")
    encoded_payload = artifact.get("payload")
    if not isinstance(encoded_payload, str):
        raise NativeStreamError("Drama 3 take artifact payload is not base64 text")
    try:
        payload = base64.b64decode(encoded_payload, validate=True)
        take = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeStreamError(f"Drama 3 take artifact is invalid: {error}") from error
    take_text = take.get("text") if isinstance(take, dict) else None
    if not isinstance(take_text, str):
        raise NativeStreamError("Drama 3 take artifact omitted its source text")
    books = take.get("vq_codes") if isinstance(take, dict) else None
    if not isinstance(books, list) or len(books) != 10:
        raise NativeStreamError("Drama 3 take artifact must contain ten VQ codebooks")
    if any(not isinstance(book, list) for book in books):
        raise NativeStreamError("Drama 3 take artifact contains a malformed VQ codebook")
    code_frames = len(books[0])
    if any(len(book) != code_frames for book in books):
        raise NativeStreamError("Drama 3 take artifact codebooks have different frame counts")
    if code_frames != frames:
        raise NativeStreamError(
            f"Drama 3 result reports {frames} frames but its take artifact contains {code_frames}"
        )
    return StreamResult(
        frames=frames,
        reached_limit=data["reached_limit"],
        take_text=take_text,
        artifact=artifact,
        artifact_payload=payload,
        result=result,
    )
