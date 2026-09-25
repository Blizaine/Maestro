"""Disposable process for the experimental direct DLSS-NR bridge.

This module deliberately uses only the Python standard library.  It is started
with the base interpreter so the child PID is the native-owning process, not a
Windows virtual-environment redirector.  A watchdog can therefore terminate
this process directly if a driver call or NGX teardown stops returning.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import BinaryIO, Callable


PROTOCOL_VERSION = 1
MAX_CONTROL_LINE = 64 * 1024
MAX_IMAGE_PIXELS = 7680 * 4320

INIT_TIMEOUT_SECONDS = 60.0
NVOF_TIMEOUT_SECONDS = 15.0
FRAME_TIMEOUT_SECONDS = 90.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0
PARENT_POLL_SECONDS = 0.05

EXIT_TIMEOUT = 124
EXIT_PARENT_GONE = 125
EXIT_WORKER_ERROR = 1


class WorkerProtocolError(RuntimeError):
    pass


def read_control(stream: BinaryIO) -> dict | None:
    """Read one bounded JSON control line; EOF between messages returns None."""
    line = stream.readline(MAX_CONTROL_LINE + 1)
    if not line:
        return None
    if len(line) > MAX_CONTROL_LINE or not line.endswith(b"\n"):
        raise WorkerProtocolError("Control message is too long or lacks a newline")
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError(f"Malformed control message: {exc}") from exc
    if not isinstance(message, dict):
        raise WorkerProtocolError("Control message must be a JSON object")
    if message.get("protocol") != PROTOCOL_VERSION:
        raise WorkerProtocolError("Unsupported direct-worker protocol version")
    return message


def write_all(stream: BinaryIO, data: bytes | bytearray | memoryview) -> None:
    view = memoryview(data).cast("B")
    offset = 0
    while offset < len(view):
        count = stream.write(view[offset:])
        if count is None or count <= 0:
            raise BrokenPipeError(f"Worker pipe accepted only {offset} of {len(view)} bytes")
        offset += count


def read_exact(stream: BinaryIO, size: int) -> bytearray:
    if size < 0 or size > MAX_IMAGE_PIXELS * 3 * 4:
        raise WorkerProtocolError(f"Invalid frame payload length: {size}")
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        if hasattr(stream, "readinto"):
            count = stream.readinto(view[offset:])
            if count is None:
                count = 0
        else:
            block = stream.read(size - offset)
            count = len(block)
            view[offset : offset + count] = block
        if count <= 0:
            raise EOFError(f"Parent closed the pipe after {offset} of {size} frame bytes")
        offset += count
    return data


def write_control(stream: BinaryIO, message: dict) -> None:
    payload = dict(message)
    payload["protocol"] = PROTOCOL_VERSION
    encoded = (json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_CONTROL_LINE:
        raise WorkerProtocolError("Worker control response exceeds the protocol limit")
    write_all(stream, encoded)
    stream.flush()


def _terminate_current_process(exit_code: int) -> None:
    """Terminate without running Python or DLL detach handlers on Windows."""
    if os.name != "nt":
        os._exit(exit_code)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateProcess.restype = ctypes.c_int
    if not kernel32.TerminateProcess(kernel32.GetCurrentProcess(), exit_code):
        raise ctypes.WinError(ctypes.get_last_error())


class OperationWatchdog:
    """Kill this disposable process when native work or its parent stops."""

    def __init__(
        self,
        parent_pid: int | None,
        *,
        terminate: Callable[[int], None] = _terminate_current_process,
        interval: float = PARENT_POLL_SECONDS,
    ):
        self._terminate = terminate
        self._interval = interval
        self._condition = threading.Condition()
        self._operation: str | None = None
        self._deadline: float | None = None
        self._stopped = False
        self._parent_handle = None
        self._parent_wait = None
        self._parent_close = None
        if os.name == "nt" and parent_pid:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel32.WaitForSingleObject.restype = ctypes.c_uint
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(0x00100000, 0, int(parent_pid))  # SYNCHRONIZE
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            self._parent_handle = handle
            self._parent_wait = kernel32.WaitForSingleObject
            self._parent_close = kernel32.CloseHandle
        self._thread = threading.Thread(target=self._watch, name="dlss5nr-watchdog", daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while True:
            with self._condition:
                self._condition.wait(self._interval)
                if self._stopped:
                    return
                # Re-read after waking: an operation may have finished while
                # this thread was asleep, so never act on a stale deadline.
                operation = self._operation
                deadline = self._deadline
            if self._parent_handle is not None and self._parent_wait(self._parent_handle, 0) == 0:
                reason = f"Parent process exited during {operation or 'worker idle'}"
                code = EXIT_PARENT_GONE
            elif deadline is not None and time.monotonic() >= deadline:
                reason = f"Native operation timed out: {operation}"
                code = EXIT_TIMEOUT
            else:
                continue
            try:
                sys.stderr.write(reason + "; terminating the isolated worker.\n")
                sys.stderr.flush()
            except Exception:
                pass
            try:
                self._terminate(code)
            except Exception as exc:
                try:
                    sys.stderr.write(f"TerminateProcess failed: {exc}\n")
                    sys.stderr.flush()
                except Exception:
                    pass

    def call(self, operation: str, timeout: float, function: Callable[[], object]):
        if timeout <= 0:
            raise ValueError("Native operation timeout must be positive")
        with self._condition:
            if self._operation is not None:
                raise RuntimeError(f"Native operation already active: {self._operation}")
            self._operation = operation
            self._deadline = time.monotonic() + timeout
        try:
            return function()
        finally:
            with self._condition:
                self._operation = None
                self._deadline = None
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._parent_handle is not None and self._parent_close is not None:
            self._parent_close(self._parent_handle)
            self._parent_handle = None


def _validate_init(message: dict) -> tuple[Path, Path, int, int, int, bool, float]:
    runtime_dir = Path(message.get("runtime_dir", "")).expanduser().resolve()
    bridge_path = Path(message.get("bridge_path", "")).expanduser().resolve()
    width, height, frames = message.get("width"), message.get("height"), message.get("frames")
    temporal = message.get("temporal")
    intensity = message.get("intensity")
    if any(type(value) is not int for value in (width, height, frames)):
        raise WorkerProtocolError("Width, height, and frame count must be integers")
    if width <= 0 or height <= 0 or frames <= 0:
        raise WorkerProtocolError("Width, height, and frame count must be positive")
    if max(width, height) > 7680 or min(width, height) > 4320:
        raise WorkerProtocolError("Frame dimensions exceed 7680x4320")
    if type(temporal) is not bool:
        raise WorkerProtocolError("Temporal must be a boolean")
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
        raise WorkerProtocolError("Intensity must be a number")
    intensity = float(intensity)
    if not math.isfinite(intensity) or not 0.0 <= intensity <= 2.0:
        raise WorkerProtocolError("Intensity must be finite and between 0 and 2")
    if os.name == "nt":
        required = (
            bridge_path,
            runtime_dir / "nvngx_dlssnr.dll",
            runtime_dir / "caller" / "nvngx.dll_comfy.dll",
        )
        if not required[2].is_file():
            required = (required[0], required[1], runtime_dir / "caller" / "nvngx.dll")
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Direct DLSS-NR runtime is missing: {missing[0]}")
    return runtime_dir, bridge_path, width, height, frames, temporal, intensity


class NativeBridge:
    """Thin standard-library binding for the staged C ABI bridge."""

    def __init__(self, runtime_dir: Path, bridge_path: Path):
        if os.name != "nt":
            raise RuntimeError("The direct DLSS-NR bridge is Windows-only")
        self.runtime_dir = runtime_dir
        self.bridge_path = bridge_path
        self.dll_directory_handles = []
        for directory in (bridge_path.parent, runtime_dir, runtime_dir / "caller"):
            if directory.is_dir():
                self.dll_directory_handles.append(os.add_dll_directory(str(directory)))
        self.lib = ctypes.CDLL(str(bridge_path))  # bridge exports use __cdecl
        self._bind()
        self.initialized = False

    def _bind(self) -> None:
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.lib.dlss5nr_init.argtypes = [ctypes.c_int, ctypes.c_wchar_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.dlss5nr_init.restype = ctypes.c_int
        self.lib.dlss5nr_process.argtypes = [
            float_pointer, float_pointer, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
        self.lib.dlss5nr_process.restype = ctypes.c_int
        self.lib.dlss5nr_shutdown.argtypes = []
        self.lib.dlss5nr_shutdown.restype = None
        for function in ("dlss5nr_version", "dlss5nr_gpu_name", "dlss5nr_nvof_info"):
            getattr(self.lib, function).argtypes = []
            getattr(self.lib, function).restype = ctypes.c_char_p
        self.lib.dlss5nr_nvof_available.argtypes = []
        self.lib.dlss5nr_nvof_available.restype = ctypes.c_int

    def initialize(self) -> tuple[str, str]:
        error = ctypes.create_string_buffer(4096)
        if not self.lib.dlss5nr_init(0, str(self.runtime_dir), error, len(error)):
            raise RuntimeError(_decode_native_error(error.value))
        self.initialized = True
        version = _decode_native_error(self.lib.dlss5nr_version())
        gpu = _decode_native_error(self.lib.dlss5nr_gpu_name())
        return version, gpu

    def nvof_available(self) -> bool:
        return bool(self.lib.dlss5nr_nvof_available())

    def nvof_info(self) -> str:
        return _decode_native_error(self.lib.dlss5nr_nvof_info())

    def process_frame(
        self,
        rgb_bytes: bytes | bytearray,
        width: int,
        height: int,
        reset: bool,
        intensity: float,
        temporal: bool,
    ) -> memoryview:
        sample_count = width * height * 3
        if len(rgb_bytes) != sample_count * ctypes.sizeof(ctypes.c_float):
            raise WorkerProtocolError("RGB input byte count does not match the configured dimensions")
        float_array = ctypes.c_float * sample_count
        # The worker receives a writable bytearray directly from read_exact;
        # let ctypes view that buffer instead of copying the full frame again.
        source = (
            float_array.from_buffer(rgb_bytes)
            if isinstance(rgb_bytes, bytearray)
            else float_array.from_buffer_copy(rgb_bytes)
        )
        destination = (ctypes.c_float * sample_count)()
        # Keep unwritten pixels visible to the parent's vectorized finite
        # check; zero initialization could disguise a partial native write.
        ctypes.memset(destination, 0xFF, sample_count * ctypes.sizeof(ctypes.c_float))
        error = ctypes.create_string_buffer(4096)
        ok = self.lib.dlss5nr_process(
            source,
            destination,
            width,
            height,
            1,       # natural style: directly validated in the reference node
            3,       # preset
            ctypes.c_float(intensity),
            ctypes.c_float(1.0),
            ctypes.c_float(1.0),
            ctypes.c_float(-1.0),
            0,       # auto mask
            int(reset),
            int(temporal),
            error,
            len(error),
        )
        if not ok:
            raise RuntimeError(_decode_native_error(error.value))
        # This view keeps destination alive until serve() has written the
        # output pipe, avoiding another full-frame copy in the worker.
        return memoryview(destination).cast("B")

    def shutdown(self) -> None:
        if self.initialized:
            self.lib.dlss5nr_shutdown()
            self.initialized = False


def _decode_native_error(value: bytes | None) -> str:
    if not value:
        return "Native bridge returned an empty diagnostic"
    return value.decode("utf-8", errors="replace")


def serve(
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    *,
    parent_pid: int | None,
    stderr: BinaryIO | None = None,
    native_factory: Callable[[Path, Path], object] = NativeBridge,
    watchdog_factory: Callable[[int | None], OperationWatchdog] = OperationWatchdog,
) -> tuple[int, bool]:
    """Serve one native session; return (exit code, native DLL was loaded)."""
    if stderr is None:
        stderr = getattr(sys.stderr, "buffer", sys.stderr)
    watchdog = None
    native = None
    native_loaded = False
    initialized = False

    def report_error(op: str, request_id: int, error: BaseException) -> None:
        write_control(output_stream, {
            "op": op,
            "request_id": request_id,
            "status": "error",
            "error": str(error)[:8000],
        })

    try:
        watchdog = watchdog_factory(parent_pid)
        init = read_control(input_stream)
        if init is None or init.get("op") != "init":
            raise WorkerProtocolError("The first command must initialize a direct NR session")
        request_id = init.get("request_id")
        if type(request_id) is not int or request_id <= 0:
            raise WorkerProtocolError("Invalid initialization request id")
        try:
            config = _validate_init(init)
        except Exception as exc:
            report_error("ready", request_id, exc)
            return EXIT_WORKER_ERROR, False
        runtime_dir, bridge_path, width, height, frames, temporal, intensity = config
        try:
            def load_and_initialize():
                nonlocal native, native_loaded, initialized
                native = native_factory(runtime_dir, bridge_path)
                native_loaded = True
                info = native.initialize()
                initialized = True
                return info

            version, gpu = watchdog.call("initialize", INIT_TIMEOUT_SECONDS, load_and_initialize)
            nvof_available = None
            nvof_info = ""
            if temporal:
                nvof_available = watchdog.call("NVOFA capability probe", NVOF_TIMEOUT_SECONDS, native.nvof_available)
                nvof_info = watchdog.call("NVOFA diagnostics", NVOF_TIMEOUT_SECONDS, native.nvof_info)
                if not nvof_available:
                    raise RuntimeError("Temporal direct NR requires NVIDIA Optical Flow (NVOFA); " + (nvof_info or "driver API unavailable"))
        except Exception as exc:
            report_error("ready", request_id, exc)
            return EXIT_WORKER_ERROR, native_loaded

        write_control(output_stream, {
            "op": "ready",
            "request_id": request_id,
            "status": "ok",
            "worker_pid": os.getpid(),
            "bridge_version": version,
            "gpu": gpu,
            "nvof_available": nvof_available,
            "nvof_info": nvof_info,
            "frames": frames,
        })

        expected_index = 0
        finished = False
        while True:
            message = read_control(input_stream)
            if message is None:
                try:
                    stderr.write(b"Parent pipe closed; terminating without native teardown.\n")
                    stderr.flush()
                except Exception:
                    pass
                return EXIT_PARENT_GONE, native_loaded
            op = message.get("op")
            request_id = message.get("request_id")
            if type(request_id) is not int or request_id <= 0:
                raise WorkerProtocolError("Invalid request id")
            if op == "close":
                if not finished:
                    raise WorkerProtocolError("Native shutdown requires a validated finish receipt first")
                watchdog.call("shutdown", SHUTDOWN_TIMEOUT_SECONDS, native.shutdown)
                write_control(output_stream, {
                    "op": "closed",
                    "request_id": request_id,
                    "status": "ok",
                    "native_shutdown": True,
                    "frames_processed": expected_index,
                    "frames_expected": frames,
                })
                return 0, native_loaded
            if op == "finish":
                if finished:
                    raise WorkerProtocolError("Session already has a finish receipt")
                if expected_index != frames:
                    report_error("finished", request_id, WorkerProtocolError(
                        f"Cannot finish: processed {expected_index} of {frames} requested frames"))
                    return EXIT_WORKER_ERROR, native_loaded
                write_control(output_stream, {
                    "op": "finished",
                    "request_id": request_id,
                    "status": "ok",
                    "frames_processed": expected_index,
                    "frames_expected": frames,
                })
                finished = True
                continue
            if op != "frame":
                raise WorkerProtocolError(f"Unexpected worker command: {op!r}")
            if finished:
                raise WorkerProtocolError("Frames cannot be processed after the finish receipt")
            index = message.get("index")
            reset = message.get("reset")
            byte_count = message.get("byte_count")
            expected_bytes = width * height * 3 * 4
            if type(index) is not int or index != expected_index or index >= frames:
                raise WorkerProtocolError(f"Expected frame index {expected_index}, received {index!r}")
            if type(reset) is not bool:
                raise WorkerProtocolError("Frame reset must be a boolean")
            if type(byte_count) is not int or byte_count != expected_bytes:
                raise WorkerProtocolError(f"Invalid RGB payload length for frame {index}")
            rgb_bytes = read_exact(input_stream, byte_count)
            # Only DirectNRSession can write this private pipe. It validates
            # uint8 RGBA frames and scales RGB to float32 [0, 1], so input
            # finiteness is guaranteed at that typed parent boundary.
            try:
                output = watchdog.call(
                    f"process frame {index}",
                    FRAME_TIMEOUT_SECONDS,
                    lambda: native.process_frame(rgb_bytes, width, height, reset, intensity, temporal),
                )
            except Exception as exc:
                report_error("frame", request_id, exc)
                return EXIT_WORKER_ERROR, native_loaded
            try:
                output_view = memoryview(output).cast("B")
            except (TypeError, ValueError):
                output_view = None
            if output_view is None or output_view.nbytes != expected_bytes:
                report_error("frame", request_id, WorkerProtocolError("Native bridge returned an invalid RGB output length"))
                return EXIT_WORKER_ERROR, native_loaded
            write_control(output_stream, {
                "op": "frame",
                "request_id": request_id,
                "status": "ok",
                "index": index,
                "byte_count": output_view.nbytes,
            })
            write_all(output_stream, output_view)
            output_stream.flush()
            expected_index += 1
    except (EOFError, BrokenPipeError, OSError, WorkerProtocolError) as exc:
        try:
            stderr.write((f"Direct worker protocol failure: {exc}\n").encode("utf-8", errors="replace"))
            stderr.flush()
        except Exception:
            pass
        return EXIT_WORKER_ERROR, native_loaded
    except Exception as exc:
        try:
            stderr.write((f"Direct worker native failure: {exc}\n").encode("utf-8", errors="replace"))
            stderr.flush()
        except Exception:
            pass
        return EXIT_WORKER_ERROR, native_loaded
    finally:
        if watchdog is not None:
            watchdog.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parent-pid", required=True, type=int)
    args = parser.parse_args(argv)
    code, _native_loaded = serve(
        getattr(sys.stdin, "buffer", sys.stdin),
        getattr(sys.stdout, "buffer", sys.stdout),
        parent_pid=args.parent_pid,
    )
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # This worker is disposable. Avoid Python/CRT/DLL detach paths, which can
        # block in the same native teardown that the operation watchdog bounds.
        _terminate_current_process(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
