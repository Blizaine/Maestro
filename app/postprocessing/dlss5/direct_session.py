"""Parent-side streaming client for the isolated direct DLSS-NR worker."""

from __future__ import annotations

from collections import deque
import json
import logging
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np


PROTOCOL_VERSION = 1
MAX_CONTROL_LINE = 64 * 1024
MAX_IMAGE_PIXELS = 7680 * 4320
INIT_TIMEOUT_SECONDS = 70.0
FRAME_TIMEOUT_SECONDS = 105.0
SHUTDOWN_TIMEOUT_SECONDS = 9.0
WORKER_EXIT_TIMEOUT_SECONDS = 2.0


class DirectNRSessionError(RuntimeError):
    """The isolated direct NR worker could not complete the requested work."""


class DirectNRTimeoutError(DirectNRSessionError):
    pass


class DirectNRCancelled(DirectNRSessionError):
    pass


def _decode_control(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_CONTROL_LINE or not line.endswith(b"\n"):
        raise DirectNRSessionError("Direct NR worker sent an oversized or incomplete control message")
    try:
        response = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectNRSessionError(f"Direct NR worker sent invalid JSON: {exc}") from exc
    if not isinstance(response, dict) or response.get("protocol") != PROTOCOL_VERSION:
        raise DirectNRSessionError("Direct NR worker sent an invalid protocol response")
    return response


def _read_exact(stream, size: int) -> bytearray:
    if type(size) is not int or size < 0 or size > MAX_IMAGE_PIXELS * 3 * 4:
        raise DirectNRSessionError(f"Direct NR worker declared an invalid payload length: {size!r}")
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError(f"Direct NR worker closed after {offset} of {size} output bytes")
        offset += count
    return data


class DirectNRSession:
    """Keep one native NR instance isolated in a disposable child process.

    Frames are streamed as float32 RGB payloads. ``finished`` is a validated
    rendering receipt sent before native shutdown; if shutdown hangs, callers
    can still distinguish completed frames from incomplete rendering.
    """

    def __init__(
        self,
        width: int,
        height: int,
        frames: int,
        *,
        temporal: bool,
        intensity: float,
        runtime_dir: str | os.PathLike[str],
        abort_callback=None,
    ):
        self.width = self._positive_int(width, "width")
        self.height = self._positive_int(height, "height")
        self.frames = self._positive_int(frames, "frame count")
        if max(self.width, self.height) > 7680 or min(self.width, self.height) > 4320:
            raise ValueError("Direct NR dimensions exceed 7680x4320")
        if type(temporal) is not bool:
            raise TypeError("temporal must be a boolean")
        if isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
            raise TypeError("intensity must be numeric")
        if not math.isfinite(float(intensity)) or not 0.0 <= float(intensity) <= 2.0:
            raise ValueError("intensity must be finite and between 0 and 2")
        self.temporal = temporal
        self.intensity = float(intensity)
        self.abort_callback = abort_callback
        self._direct_root = Path(runtime_dir).expanduser().resolve()
        self._native_runtime = self._direct_root / "runtime"
        self._bridge_path = self._direct_root / "native" / "bin" / "dlss5nr_bridge.dll"
        self._worker_path = Path(__file__).with_name("direct_worker.py").resolve()
        if os.name != "nt":
            raise DirectNRSessionError("The direct DLSS-NR worker is Windows-only")
        if not self._worker_path.is_file():
            raise DirectNRSessionError(f"Direct NR worker is missing: {self._worker_path}")
        if not (self._native_runtime / "nvngx_dlssnr.dll").is_file() or not self._bridge_path.is_file():
            raise DirectNRSessionError("The direct DLSS-NR runtime or bridge is missing")
        base_python = Path(getattr(sys, "_base_executable", "") or "")
        if not base_python.is_file():
            raise DirectNRSessionError("Could not locate the base Python executable for the isolated worker")

        self._request_id = 0
        self._next_frame = 0
        self.render_finished = False
        self.native_shutdown = False
        self.cleanup_status = "running"
        self._closed = False
        self._failed = False
        self._response_queue: queue.Queue = queue.Queue(maxsize=2)
        self._logs: deque[str] = deque(maxlen=100)
        self._log_lock = threading.Lock()
        self._closed_streams = False

        command = [
            str(base_python), "-I", "-u", str(self._worker_path),
            "--parent-pid", str(os.getpid()),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self._worker_path.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if self._process.stdin is None or self._process.stdout is None or self._process.stderr is None:
                raise DirectNRSessionError("Could not create direct NR worker pipes")
            self._stdout_thread = threading.Thread(target=self._read_responses, name="dlss5nr-response-reader", daemon=True)
            self._stderr_thread = threading.Thread(target=self._read_stderr, name="dlss5nr-stderr-reader", daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()
            response = self._expect(self._exchange({
                "op": "init",
                "runtime_dir": str(self._native_runtime),
                "bridge_path": str(self._bridge_path),
                "width": self.width,
                "height": self.height,
                "frames": self.frames,
                "temporal": temporal,
                "intensity": self.intensity,
            }, timeout=INIT_TIMEOUT_SECONDS), "ready")
            if response.get("status") != "ok":
                self._raise_worker_error(response)
            if response.get("worker_pid") != self._process.pid:
                raise DirectNRSessionError(
                    "Direct NR worker PID does not match the owned process; refusing unsafe cleanup"
                )
            if response.get("frames") != self.frames:
                raise DirectNRSessionError("Direct NR worker acknowledged the wrong frame count")
            if temporal and response.get("nvof_available") is not True:
                detail = str(response.get("nvof_info") or "")
                raise DirectNRSessionError("Temporal direct NR requires NVIDIA Optical Flow (NVOFA)" + (f": {detail}" if detail else ""))
            self.bridge_version = str(response.get("bridge_version") or "unknown")
            self.gpu = str(response.get("gpu") or "unknown")
            self.nvof_info = str(response.get("nvof_info") or "")
        except BaseException:
            self._failed = True
            self._terminate_worker()
            self._close_streams()
            raise

    @staticmethod
    def _positive_int(value: int, label: str) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
        return value

    @property
    def worker_pid(self) -> int:
        return int(self._process.pid)

    @property
    def worker_logs(self) -> tuple[str, ...]:
        with self._log_lock:
            return tuple(self._logs)

    def _read_responses(self) -> None:
        stream = self._process.stdout
        try:
            while True:
                line = stream.readline(MAX_CONTROL_LINE + 1)
                if not line:
                    self._response_queue.put(EOFError("Direct NR worker closed its response pipe"))
                    return
                response = _decode_control(line)
                payload = None
                if "byte_count" in response:
                    payload = _read_exact(stream, response["byte_count"])
                self._response_queue.put((response, payload))
        except BaseException as exc:
            try:
                self._response_queue.put(exc)
            except BaseException:
                pass

    def _read_stderr(self) -> None:
        stream = self._process.stderr
        while True:
            try:
                data = stream.read(4096)
            except (OSError, ValueError):
                return
            if not data:
                return
            text = data.decode("utf-8", errors="replace")
            with self._log_lock:
                for line in text.splitlines():
                    self._logs.append(line[:2000])

    def _check_cancelled(self) -> None:
        if self.abort_callback is None:
            return
        try:
            cancelled = bool(self.abort_callback())
        except Exception as exc:
            raise DirectNRSessionError(f"Direct NR abort callback failed: {exc}") from exc
        if cancelled:
            terminated = self._terminate_worker()
            self.cleanup_status = "aborted" if terminated else "termination_failed"
            self._closed = True
            raise DirectNRCancelled("Direct NR operation cancelled")

    def _write_all(self, data: bytes | bytearray | memoryview, deadline: float) -> None:
        view = memoryview(data).cast("B")
        offset = 0
        while offset < len(view):
            self._check_cancelled()
            if time.monotonic() >= deadline:
                self._terminate_worker()
                raise DirectNRTimeoutError("Timed out writing to the direct NR worker")
            result: queue.Queue = queue.Queue(maxsize=1)

            def write_once():
                try:
                    result.put((self._process.stdin.write(view[offset:]), None))
                except BaseException as exc:
                    result.put((None, exc))

            writer = threading.Thread(target=write_once, name="dlss5nr-pipe-writer", daemon=True)
            writer.start()
            try:
                while True:
                    self._check_cancelled()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._terminate_worker()
                        raise DirectNRTimeoutError("Timed out writing to the direct NR worker")
                    try:
                        count, error = result.get(timeout=min(0.05, remaining))
                        if error is not None:
                            raise error
                        break
                    except queue.Empty:
                        if self._process.poll() is not None:
                            raise BrokenPipeError("Direct NR worker exited while input was being written")
            except BaseException:
                self._terminate_worker()
                writer.join(timeout=0.25)
                raise
            if count is None or count <= 0:
                raise DirectNRSessionError(self._failure_message("Direct NR worker pipe accepted no data"))
            offset += count

    def _exchange(self, request: dict[str, Any], *, timeout: float, payload: bytes | None = None):
        if self._closed:
            raise DirectNRSessionError("Direct NR session is closed")
        if self._failed or self._process.poll() is not None:
            raise DirectNRSessionError(self._failure_message("Direct NR worker is no longer running"))
        self._request_id += 1
        request_id = self._request_id
        message = dict(request)
        message.update(protocol=PROTOCOL_VERSION, request_id=request_id)
        if payload is not None:
            message["byte_count"] = len(payload)
        encoded = (json.dumps(message, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        if len(encoded) > MAX_CONTROL_LINE:
            raise DirectNRSessionError("Direct NR request exceeds the control message limit")
        deadline = time.monotonic() + timeout
        try:
            self._write_all(encoded, deadline)
            if payload is not None:
                self._write_all(payload, deadline)
            self._process.stdin.flush()
        except BaseException:
            self._failed = True
            self._terminate_worker()
            raise

        while True:
            self._check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._failed = True
                self._terminate_worker()
                raise DirectNRTimeoutError(self._failure_message(f"Direct NR worker timed out during {message.get('op')}"))
            try:
                item = self._response_queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                if self._process.poll() is not None:
                    # Allow the reader to enqueue the final response or EOF.
                    try:
                        item = self._response_queue.get(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
                    except queue.Empty:
                        raise DirectNRSessionError(self._failure_message("Direct NR worker exited without a response"))
                else:
                    continue
            if isinstance(item, BaseException):
                raise DirectNRSessionError(self._failure_message(str(item))) from item
            response, data = item
            if response.get("request_id") != request_id:
                raise DirectNRSessionError("Direct NR worker response request id did not match")
            if response.get("status") == "error":
                self._raise_worker_error(response)
            return response, data

    def _raise_worker_error(self, response: dict[str, Any]) -> None:
        error = str(response.get("error") or "unknown worker error")
        raise DirectNRSessionError(self._failure_message(error))

    @staticmethod
    def _expect(result, expected_op: str):
        response, payload = result
        if response.get("op") != expected_op:
            raise DirectNRSessionError(f"Expected direct NR {expected_op!r} response, got {response.get('op')!r}")
        return response

    def _failure_message(self, message: str) -> str:
        logs = self.worker_logs
        return message + ("\n" + "\n".join(logs[-12:]) if logs else "")

    def _terminate_worker(self) -> bool:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return True
        try:
            process.kill()
        except (OSError, AttributeError):
            pass
        try:
            process.wait(timeout=WORKER_EXIT_TIMEOUT_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            pass
        terminated = process.poll() is not None
        if not terminated:
            self.cleanup_status = "termination_failed"
        return terminated

    def _close_streams(self) -> None:
        if self._closed_streams:
            return
        self._closed_streams = True
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            self._closed_streams = False
            return
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name, None)
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        for name in ("_stdout_thread", "_stderr_thread"):
            thread = getattr(self, name, None)
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=0.1)

    def process_frame(self, index: int, rgba: np.ndarray, reset: bool, output: np.ndarray | None = None) -> np.ndarray:
        if self._closed or self._failed:
            raise DirectNRSessionError("Direct NR session is not available")
        if type(index) is not int or index != self._next_frame or index >= self.frames:
            raise ValueError(f"Expected frame index {self._next_frame}, received {index!r}")
        if type(reset) is not bool:
            raise TypeError("reset must be a boolean")
        if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.shape != (self.height, self.width, 4):
            raise ValueError(f"Expected uint8 RGBA frame with shape {(self.height, self.width, 4)}")
        if output is not None and (
            not isinstance(output, np.ndarray)
            or output.dtype != np.uint8
            or output.shape != (self.height, self.width, 4)
            or not output.flags.writeable
        ):
            raise ValueError("output must be a writable uint8 RGBA array with the configured frame shape")
        self._check_cancelled()
        # The public input contract is uint8 RGBA; converting those byte
        # values to float32 [0, 1] guarantees finite RGB payload values.
        rgb = np.ascontiguousarray(rgba[..., :3].astype(np.float32) / np.float32(255.0))
        payload = rgb.tobytes(order="C")
        try:
            response, raw_output = self._exchange({
                "op": "frame",
                "index": index,
                "reset": reset,
            }, timeout=FRAME_TIMEOUT_SECONDS, payload=payload)
            self._expect((response, raw_output), "frame")
            expected_bytes = self.width * self.height * 3 * 4
            if response.get("index") != index or response.get("byte_count") != expected_bytes:
                raise DirectNRSessionError("Direct NR worker returned a mismatched frame index or byte count")
            if raw_output is None or len(raw_output) != expected_bytes:
                raise DirectNRSessionError("Direct NR worker returned an incomplete frame")
            rendered = np.frombuffer(raw_output, dtype=np.float32).reshape(self.height, self.width, 3)
            if not np.isfinite(rendered).all():
                raise DirectNRSessionError("Direct NR worker returned non-finite pixels")
            # The bridge's RGB/BGR convention varies by native build. Match the
            # pinned Comfy node's inexpensive sampled channel-order heuristic.
            step_y = max(1, self.height // 128)
            step_x = max(1, self.width // 128)
            reference = rgb[::step_y, ::step_x]
            raw_sample = rendered[::step_y, ::step_x]
            swapped_sample = raw_sample[..., ::-1]
            raw_score = float(np.abs(raw_sample - reference).mean()) + float(
                np.abs(raw_sample.mean(axis=(0, 1)) - reference.mean(axis=(0, 1))).mean()
            )
            swapped_score = float(np.abs(swapped_sample - reference).mean()) + float(
                np.abs(swapped_sample.mean(axis=(0, 1)) - reference.mean(axis=(0, 1))).mean()
            )
            corrected = rendered if raw_score <= swapped_score else rendered[..., ::-1]
            result = np.empty((self.height, self.width, 4), dtype=np.uint8) if output is None else output
            result[..., :3] = np.rint(np.clip(corrected, 0.0, 1.0) * 255.0).astype(np.uint8)
            result[..., 3] = rgba[..., 3]
            self._next_frame += 1
            return result
        except BaseException:
            self._failed = True
            self._terminate_worker()
            raise

    def close(self, abort: bool = False) -> bool:
        if self._closed:
            terminated = self._terminate_worker()
            self._close_streams()
            if not terminated:
                raise DirectNRSessionError("Direct NR worker is still running after termination was requested")
            return self.native_shutdown
        if abort:
            terminated = self._terminate_worker()
            self.cleanup_status = "aborted" if terminated else "termination_failed"
            self._closed = True
            self._close_streams()
            if not terminated:
                raise DirectNRSessionError("Direct NR worker is still running after abort cleanup")
            return False
        if self._failed:
            terminated = self._terminate_worker()
            self.cleanup_status = "worker_failed" if terminated else "worker_failed_termination_failed"
            self._closed = True
            self._close_streams()
            raise DirectNRSessionError(self._failure_message("Cannot finish a failed direct NR session"))
        if self._next_frame != self.frames:
            terminated = self._terminate_worker()
            self.cleanup_status = "incomplete" if terminated else "incomplete_termination_failed"
            self._closed = True
            self._close_streams()
            raise DirectNRSessionError(f"Cannot finish: processed {self._next_frame} of {self.frames} requested frames")

        try:
            response, _ = self._exchange({"op": "finish"}, timeout=SHUTDOWN_TIMEOUT_SECONDS)
            response = self._expect((response, None), "finished")
            if response.get("frames_processed") != self.frames or response.get("frames_expected") != self.frames:
                raise DirectNRSessionError("Direct NR worker finish receipt did not match the requested frame count")
            self.render_finished = True
        except BaseException:
            terminated = self._terminate_worker()
            self.cleanup_status = "finish_failed" if terminated else "finish_failed_termination_failed"
            self._closed = True
            self._close_streams()
            raise

        try:
            response, _ = self._exchange({"op": "close"}, timeout=SHUTDOWN_TIMEOUT_SECONDS)
            response = self._expect((response, None), "closed")
            if response.get("frames_processed") != self.frames or response.get("frames_expected") != self.frames:
                raise DirectNRSessionError("Direct NR worker close receipt did not match the requested frame count")
            if response.get("native_shutdown") is not True:
                raise DirectNRSessionError("Direct NR worker did not confirm native shutdown")
            self.native_shutdown = True
            try:
                self._process.wait(timeout=WORKER_EXIT_TIMEOUT_SECONDS)
                self.cleanup_status = "native_shutdown"
            except subprocess.TimeoutExpired:
                terminated = self._terminate_worker()
                self.cleanup_status = "native_shutdown_worker_forced_exit" if terminated else "termination_failed_after_native_shutdown"
        except DirectNRCancelled:
            self._terminate_worker()
            self.cleanup_status = "aborted_after_finish"
            self._closed = True
            self._close_streams()
            raise
        except Exception as exc:
            # A validated pre-shutdown finish receipt proves frame completion.
            # Shutdown may still hang inside the driver; report that cleanup
            # separately and preserve the rendered result as successful.
            self._terminate_worker()
            self.cleanup_status = "forced_after_finish" if self._process.poll() is not None else "termination_failed_after_finish"
            logging.getLogger(__name__).warning(
                "Direct NR rendered %d/%d frames; native shutdown required bounded cleanup (%s): %s",
                self._next_frame, self.frames, self.cleanup_status, exc,
            )
        except BaseException:
            terminated = self._terminate_worker()
            self.cleanup_status = "interrupted_after_finish" if terminated else "termination_failed_after_finish"
            raise
        finally:
            self._closed = True
            self._close_streams()
        if self.cleanup_status in {"termination_failed_after_finish", "termination_failed_after_native_shutdown"}:
            raise DirectNRSessionError(
                "Direct NR rendered all frames but its isolated worker could not be terminated"
            )
        return self.native_shutdown

    def __enter__(self) -> "DirectNRSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(abort=exc is not None)

    def __del__(self):
        try:
            if hasattr(self, "_closed") and not self._closed:
                self.close(abort=True)
        except Exception:
            pass
